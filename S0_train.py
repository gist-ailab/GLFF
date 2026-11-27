import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
import time
from datetime import timedelta

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,6,7"
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import argparse
from tqdm import tqdm
import timm
import numpy as np
import matplotlib.pyplot as plt
from util import read_conf, validation_accuracy, validation_accuracy_lora

import random
import rein
import adaptformer

import dino_variant
from sklearn.metrics import f1_score
from data.ham10000 import get_dataloaders
from data.ham10000_train_split import get_cls_dataloaders_train_only


def ddp_setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def rein_forward(model, inputs):
    backbone = model.module if isinstance(model, torch.nn.DataParallel) else model
    feats = backbone.forward_features(inputs)[:, 0, :]
    logits = backbone.linear(feats)
    return logits


def seed_for_init(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def set_requires_grad(model, layers_to_train):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_train):
            param.requires_grad = True
        else:
            param.requires_grad = False
            
def train():
    seed_for_init(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='ham10000')
    parser.add_argument('--adapter', '-a', type=str, default='rein')
    # parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str, default='ham10000')
    parser.add_argument('--epoch', '-e', type=str, default=50)
    parser.add_argument('--noise_rate', type=float, default=0.0, help='Label noise rate (0.0~1.0)')
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = read_conf('conf/data/'+args.data+'.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    # save_path = os.path.join(config['save_path'], args.save_path)
    save_dir = f"{args.save_path}_noise_{args.noise_rate}_epoch_{args.epoch}"
    save_path = os.path.join('./S0_train/save/', save_dir)
    # data_path = config['data_root']
    data_path = "/home/ailab-lba/ym_workspace/dataset/ham10000/"
    # batch_size = int(config['batch_size'])
    batch_size = 64
    max_epoch = int(args.epoch)
    # num_workers = int(config['num_workers'])
    
    train_loader, _, test_loader = get_dataloaders(
        data_dir=data_path,
        batch_size=batch_size,
        num_workers=4,
        random_seed=42,
        noise_rate=args.noise_rate
    )
    
    # train_loader, test_loader = get_cls_dataloaders_train_only(
    #     data_dir=data_path,
    #     batch_size=batch_size,   
    #     num_workers=4,
    #     random_seed=42
    # )


    # train_loader = torch.utils.data.DataLoader(
    #     train_loader.dataset,
    #     batch_size=128,
    #     # sampler=train_sampler,
    #     num_workers=4,
    #     pin_memory=True
    # )

    # test_loader = torch.utils.data.DataLoader(
    #     test_loader.dataset,
    #     batch_size=128,
    #     # sampler=val_sampler,
    #     num_workers=4,
    #     pin_memory=True
    # )

    os.makedirs(save_path, exist_ok=True)
        
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    elif args.netsize == 'b':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant
    elif args.netsize == 'l':
        model_load = dino_variant._large_dino
        variant = dino_variant._large_variant
    
    model = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = model.state_dict()

    model = rein.ReinsDinoVisionTransformer(
        **variant
    )
    model.load_state_dict(dino_state_dict, strict=False)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    set_requires_grad(model, ['reins', 'linear'])
    
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # print(model)
    
    print("Max epoch: ", max_epoch)
    
    criterion = torch.nn.CrossEntropyLoss()
    print("Criterion: ", criterion)
    model.eval()


    lr_decay = [int(0.5*max_epoch), int(0.75*max_epoch), int(0.9*max_epoch)]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
 
    saver = timm.utils.CheckpointSaver(
        model.module if isinstance(model, torch.nn.DataParallel) else model,
        optimizer,
        checkpoint_dir=save_path,
        max_history=1
    )

    if args.adapter == 'lora':
        scalar = GradScaler()
        
    avg_accuracy = 0.0
    best_acc = 0.0
    train_accuracies = []
    val_accuracies = []
    start_time = time.time()
        
    for epoch in range(max_epoch):
        epoch_start_time = time.time()
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{max_epoch}]", leave=False)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)    
            
            if targets.ndim > 1 and targets.size(1) > 1:
                targets = torch.argmax(targets, dim=1)
                
            if targets.ndim > 1:
                targets = targets.view(-1) 
            
            optimizer.zero_grad()
            
            outputs = rein_forward(model, inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            train_accuracy = correct / total
            avg_loss = total_loss / (batch_idx + 1)

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc":  f"{train_accuracy*100:.2f}%"
            })
                        
        epoch_duration = time.time() - epoch_start_time
        epoch_time = str(timedelta(seconds=epoch_duration))
        remaining_time = (max_epoch - (epoch + 1)) * epoch_duration
        formatted_remaining_time = str(timedelta(seconds=remaining_time))
        print(f"\nEpoch {epoch} took {epoch_time}")
        print(f"Estimated remaining training time: {formatted_remaining_time}")
        print()

        ## validation
        model.eval()
        total_loss = 0
        total = 0
        correct = 0

        if args.adapter == 'rein' or args.adapter == 'adaptformer':
            valid_accuracy = validation_accuracy(model, test_loader, device, mode=args.adapter)
        elif args.adapter == 'lora':
            valid_accuracy = validation_accuracy_lora(model, test_loader, device)
            
        train_accuracies.append(train_accuracy)
        val_accuracies.append(valid_accuracy)
        
        if epoch >= max_epoch-10:
            avg_accuracy += valid_accuracy 
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print(f'Epoch {epoch + 1}/{max_epoch} | Loss: {avg_loss:.4f} | '
            f'Train Acc: {train_accuracy:.4f} | Valid Acc: {valid_accuracy:.4f} | '
            f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print(scheduler.get_last_lr())
    
    total_duration = time.time() - start_time
    totoal_time = str(timedelta(seconds=total_duration))
    print(f"Total training time: {totoal_time}")
    
    if args.adapter == 'rein' or args.adapter == 'adaptformer':
        final_acc = validation_accuracy(model, test_loader, device, mode=args.adapter)
    elif args.adapter == 'lora':
        final_acc = validation_accuracy_lora(model, test_loader, device)
    print(f"Final Test Accuracy: {final_acc:.4f}")
    
if __name__ =='__main__':
    train()