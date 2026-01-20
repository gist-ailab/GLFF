import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
import sys
sys.path.append("/SSDe/youmin_park/GLFF/")
import time
from datetime import timedelta
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import functional as F

import argparse
import timm
import numpy as np
import matplotlib.pyplot as plt
from util import read_conf
from util.metric_seg import validation_accuracy

import random
import rein
import adaptformer
from net_kd import GlobalLocalLesionClassifier, kd_loss, GlobalOnlyStudent

import dino_variant
from sklearn.metrics import f1_score
from data.ham10000_seg import get_dataloaders_with_mask


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
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--epoch', '-e', type=str, default=None)

    # KD 관련 인자 (필요 최소)
    parser.add_argument('--kd_alpha', type=float, default=0.5)  # CE:(1-alpha), KD:alpha
    parser.add_argument('--kd_T', type=float, default=4.0)      # distillation temperature
    args = parser.parse_args()

    config = read_conf('conf/data/' + args.data + '.yaml')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")

    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['data_root']
    batch_size = 128
    max_epoch = int(args.epoch)

    train_loader, valid_loader, test_loader = get_dataloaders_with_mask(
        data_dir=data_path,
        batch_size=batch_size,
        num_workers=4,
        random_seed=42
    )

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
    else:
        raise ValueError(f"Unknown netsize: {args.netsize}")

    hub_model = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = hub_model.state_dict()

    # -------------------------
    # Teacher (lesion-aware)
    # -------------------------
    teacher_global = rein.ReinsDinoVisionTransformer(**variant)
    teacher_global.load_state_dict(dino_state_dict, strict=False)

    teacher_local = rein.ReinsDinoVisionTransformer(**variant)
    teacher_local.load_state_dict(dino_state_dict, strict=False)

    teacher = GlobalLocalLesionClassifier(
        global_encoder=teacher_global,
        local_encoder=teacher_local,
        embed_dim=variant['embed_dim'],
        num_classes=config['num_classes'],
        patch_size=variant['patch_size'],
        tau=0.001,
        min_tokens=4,
        num_heads=8,
        fusion="gate",          # gate여도 동일
        hidden_dim=512,
        use_global_cls=False
    )

    # teacher 고정
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False


    # -------------------------
    # Student (lesion-free, global-only)
    # -------------------------
    student_global = rein.ReinsDinoVisionTransformer(**variant)
    student_global.load_state_dict(dino_state_dict, strict=False)

    student = GlobalOnlyStudent(
        global_encoder=student_global,
        teacher_head=teacher.head,   # 그대로 공유
        use_global_cls=False,
        head_mode="global_only"      # 또는 "zero_local"
    )


    # Student만 학습
    set_requires_grad(student, ['reins', 'head', 'global_encoder'])

    if torch.cuda.device_count() > 1:
        teacher = torch.nn.DataParallel(teacher)
        student = torch.nn.DataParallel(student)

    teacher = teacher.to(device)
    student = student.to(device)

    # print(student)  # 기존처럼 모델 출력 유지 (student가 실제 학습 대상)

    criterion = torch.nn.CrossEntropyLoss()
    print("Criterion: ", criterion)

    lr_decay = [int(0.5 * max_epoch), int(0.75 * max_epoch), int(0.9 * max_epoch)]
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)

    saver = timm.utils.CheckpointSaver(
        student.module if isinstance(student, torch.nn.DataParallel) else student,
        optimizer,
        checkpoint_dir=save_path,
        max_history=1
    )

    avg_accuracy = 0.0
    train_accuracies = []
    val_accuracies = []
    start_time = time.time()

    print("Max epoch:", max_epoch)
    print("KD alpha:", args.kd_alpha, "| KD T:", args.kd_T)

    for epoch in range(max_epoch):
        epoch_start_time = time.time()

        # -------------------------
        # Train (student only)
        # -------------------------
        student.train()
        total_loss = 0.0
        total = 0
        correct = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"[Epoch {epoch+1}/{max_epoch}] Train",
            leave=False
        )

        for inputs, masks, targets in train_pbar:
            inputs = inputs.to(device)
            masks = masks.to(device)
            targets = targets.to(device)

            if targets.ndim > 1 and targets.size(1) > 1:
                targets = torch.argmax(targets, dim=1)
            if targets.ndim > 1:
                targets = targets.view(-1)

            optimizer.zero_grad()

            # ----- Teacher logits (mask 사용) -----
            with torch.no_grad():
                logits_t = teacher(inputs, masks)  # (B,C)

            # ----- Student logits (mask 없이) -----
            logits_s = student(inputs)            # (B,C)

            # ----- Loss: CE + KD -----
            loss_ce = criterion(logits_s, targets)
            loss_kd = kd_loss(logits_s, logits_t, T=args.kd_T)
            loss = (1.0 - args.kd_alpha) * loss_ce + args.kd_alpha * loss_kd

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += targets.size(0)
            _, predicted = logits_s.max(1)
            correct += predicted.eq(targets).sum().item()

            train_acc = correct / total

            train_pbar.set_postfix({
                "loss": f"{total_loss / (total // targets.size(0) + 1):.4f}",
                "acc": f"{train_acc*100:.2f}%",
                "ce": f"{loss_ce.item():.3f}",
                "kd": f"{loss_kd.item():.3f}",
            })

        train_avg_loss = total_loss / len(train_loader)
        train_accuracy = train_acc
        train_accuracies.append(train_accuracy)

        # -------------------------
        # Validation (student only, mask-free loader)
        # -------------------------
        student.eval()
        valid_accuracy = validation_accuracy(student, test_loader, device)
        val_accuracies.append(valid_accuracy)

        if epoch >= max_epoch - 10:
            avg_accuracy += valid_accuracy

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
        print(
            f"Epoch time: {timedelta(seconds=int(epoch_duration))} | "
            f"Remaining: {timedelta(seconds=int(remaining_time))}"
        )
        print("-" * 80)

    # -------------------------------------------------
    # Test (student only)
    # -------------------------------------------------
    student.eval()
    final_acc = validation_accuracy(student, test_loader, device)
    print(f"\nFinal Test Accuracy: {final_acc:.4f}")

    total_duration = time.time() - start_time
    print(f"Total training time: {timedelta(seconds=int(total_duration))}")


if __name__ == '__main__':
    train()
