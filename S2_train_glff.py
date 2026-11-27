import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
import time
from datetime import timedelta
from tqdm import tqdm

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,6,7"
import torch
import torch.nn as nn

import argparse
import timm
from util import read_conf
from util.metric_seg import validation_accuracy

import rein
from net_hard import GlobalLocalLesionClassifier
# from net_soft import GlobalLocalLesionClassifier

import dino_variant
from data.ham10000_seg import get_dataloaders_with_mask
from util.forward import rein_forward, lora_forward, adaptformer_forward
from util.training import seed_for_init, set_requires_grad


def train():
    seed_for_init(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='ham10000')
    parser.add_argument('--adapter', '-a', type=str, default='rein')
    # parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', choices=dino_variant.DINO_VARIANTS)
    parser.add_argument('--save_path', '-s', type=str, default='glff_ham10000_rein_s_hard_clspatch_512_warmup5')
    parser.add_argument('--epoch', '-e', type=str, default=50)
    parser.add_argument('--noise_rate', type=float, default=0.0, help='Label noise rate (0.0 ~ 1.0)')
    args = parser.parse_args()
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = read_conf('conf/data/'+args.data+'.yaml')
    # device = 'cuda:'+args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")
    # save_path = os.path.join(config['save_path'], args.save_path)
    save_dir = f"{args.save_path}_noise_{args.noise_rate}_epoch_{args.epoch}"
    save_path = os.path.join('./S2_save/', save_dir)
    # data_path = config['data_root']
    data_path = "/home/ailab-lba/ym_workspace/dataset/ham10000/"
    # batch_size = int(config['batch_size'])
    batch_size = 64
    max_epoch = int(args.epoch)
    # num_workers = int(config['num_workers'])
    
    train_loader, _, test_loader = get_dataloaders_with_mask(
        data_dir=data_path,
        batch_size=batch_size,
        num_workers=4,
        random_seed=42,
        noise_rate=args.noise_rate
    )
    
    # train_loader, test_loader = get_dataloaders_train_only(
    #     data_dir=data_path,
    #     batch_size=batch_size,
    #     num_workers=4,
    #     random_seed=42  
    # )
    
    os.makedirs(save_path, exist_ok=True)
        
    model_load, variant = dino_variant.DINO_VARIANTS[args.netsize]
    
    hub_model = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = hub_model.state_dict()

    # global/local encoder
    global_encoder = rein.ReinsDinoVisionTransformer(**variant)
    global_encoder.load_state_dict(dino_state_dict, strict=False)

    local_encoder = rein.ReinsDinoVisionTransformer(**variant)
    local_encoder.load_state_dict(dino_state_dict, strict=False)

    model = GlobalLocalLesionClassifier(
        global_encoder=global_encoder,
        local_encoder=local_encoder,
        embed_dim=variant['embed_dim'],
        num_classes=config['num_classes'],
        patch_size=variant['patch_size'],   
        tau=0.000,
        min_tokens=4,
        num_heads=8,
        cross_attn_layers=1,
        fusion="concat",        # "concat" or "gate"
        hidden_dim=512,   # 256?
        # dropout=0.2,
        cross_attn_warmup_epochs=5,
        use_global_cls=True
    )
        
    set_requires_grad(model, ['reins', 'head', 'cross_attn', 'global_encoder', 'local_encoder'])    
        
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    
    
    print(model)
    
    model.eval()

    print("Max epoch:", max_epoch)

    criterion = nn.CrossEntropyLoss()
    print("Criterion:", criterion)

    lr_decay = [int(0.5 * max_epoch), int(0.75 * max_epoch), int(0.9 * max_epoch)]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)

    saver = timm.utils.CheckpointSaver(
        model.module if isinstance(model, torch.nn.DataParallel) else model,
        optimizer,
        checkpoint_dir=save_path,
        max_history=1
    )


    start_time = time.time()

    for epoch in range(max_epoch):
        epoch_start_time = time.time()

        # === cross-attn warm-up epoch 전달 ===
        if isinstance(model, torch.nn.DataParallel):
            model.module.current_epoch = epoch
        else:
            model.current_epoch = epoch
        # -------------------------
        # Train
        # -------------------------
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"[Epoch {epoch+1}/{max_epoch}] Train",
            leave=False
        )

        for inputs, masks, targets in train_pbar:
            inputs  = inputs.to(device)
            masks   = masks.to(device)
            targets = targets.to(device)

            if targets.ndim > 1 and targets.size(1) > 1:
                targets = torch.argmax(targets, dim=1)
            if targets.ndim > 1:
                targets = targets.view(-1)

            optimizer.zero_grad()

            outputs = model(inputs, masks)     # logits
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            train_acc = correct / total

            train_pbar.set_postfix({
                "loss": f"{total_loss / (total // targets.size(0) + 1):.4f}",
                "acc": f"{train_acc*100:.2f}%"
            })

        train_avg_loss = total_loss / len(train_loader)
        train_accuracy = train_acc

        # -------------------------
        # Validation
        # -------------------------
        valid_accuracy = validation_accuracy(model, test_loader, device)


        scheduler.step()
        saver.save_checkpoint(epoch, metric=valid_accuracy)

        # -------------------------
        # Epoch summary
        # -------------------------
        epoch_duration = time.time() - epoch_start_time
        remaining_time = (max_epoch - (epoch + 1)) * epoch_duration

        print(
            f"\nEpoch {epoch+1}/{max_epoch} | "
            f"Loss: {train_avg_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Valid Acc: {valid_accuracy:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        print(f"Epoch time: {timedelta(seconds=int(epoch_duration))} | "
            f"Remaining: {timedelta(seconds=int(remaining_time))}")
        print("-" * 80)


    # -------------------------------------------------
    # Test
    # -------------------------------------------------
    final_acc = validation_accuracy(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {final_acc:.4f}")

    total_duration = time.time() - start_time
    print(f"Total training time: {timedelta(seconds=int(total_duration))}")
    
if __name__ =='__main__':
    train()