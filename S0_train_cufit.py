import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
import os
import time
from datetime import timedelta
from tqdm import tqdm

import torch
import torch.nn as nn
import argparse
import timm
import numpy as np

import rein
import dino_variant
from data.ham10000 import get_dataloaders
from util import read_conf, validation_accuracy

def seed_everything(seed=42):
    import random
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='ham10000')
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--noise_rate', type=float, default=0.2)
    parser.add_argument('--epoch', '-e', type=int, default=50)
    parser.add_argument('--save_path', '-s', type=str, default='ham10000_cufit_cls_only')
    args = parser.parse_args()

    seed_everything(42)
    config = read_conf('conf/data/'+args.data+'.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = f"{args.save_path}_noise_{args.noise_rate}_epoch_{args.epoch}"
    save_path = os.path.join('./S0_train_cufit/save/', save_dir)
    os.makedirs(save_path, exist_ok=True)

    # 1. 데이터 로더
    train_loader, _, test_loader = get_dataloaders(
        data_dir="/home/ailab-lba/ym_workspace/dataset/ham10000/",
        batch_size=32,
        num_workers=4,
        random_seed=42,
        noise_rate=args.noise_rate
    )

    # 2. 기초 모델 가중치 로드
    if args.netsize == 's':
        model_load, variant = dino_variant._small_dino, dino_variant._small_variant
    elif args.netsize == 'b':
        model_load, variant = dino_variant._base_dino, dino_variant._base_variant
    else:
        model_load, variant = dino_variant._large_dino, dino_variant._large_variant

    hub_model = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_sd = hub_model.state_dict()

    # 3. CUFIT 모듈 구성
    def build_model():
        m = rein.ReinsDinoVisionTransformer(**variant)
        m.load_state_dict(dino_sd, strict=False)
        m.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
        set_requires_grad(m, ['reins', 'linear'])
        return m.to(device)

    model_iam = build_model()
    model_lam = build_model()
    # LPM: 선형 층만 학습
    lpm_linear = nn.Linear(variant['embed_dim'], config['num_classes']).to(device)

    # 4. 옵티마이저 및 스케줄러
    optimizer = torch.optim.Adam([
            {'params': lpm_linear.parameters(), 'lr': 1e-3},
            {'params': model_iam.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-5)
    
    optimizer2 = torch.optim.Adam(model_lam.parameters(), lr=1e-3, weight_decay=1e-5)
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    lr_decay = [int(0.5*args.epoch), int(0.75*args.epoch)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, lr_decay)

    # 5. 학습 루프
    for epoch in range(args.epoch):
        model_iam.train(); model_lam.train(); lpm_linear.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # --- STEP 1: Main Update (LPM + IAM) ---
            optimizer.zero_grad()

            # [수정 포인트] LPM 가이드를 위해 어댑터가 없는 순수 특징 추출
            with torch.no_grad():
                # Reins를 거치지 않은 순수 백본 특징만 사용
                pure_feat = model_iam.forward_features_no_rein(inputs)[:, 0, :].detach()
                # pure_feat = model_iam.forward_features(inputs)[:, 0, :]
            
            lpm_out = lpm_linear(pure_feat)
            loss_linear = criterion(lpm_out, targets)

            # IAM 예측 (어댑터 포함 특징)
            iam_feat = model_iam.forward_features(inputs)[:, 0, :]
            iam_out = model_iam.linear(iam_feat)
            
            with torch.no_grad():
                # 순수 특징 기반의 LPM 결과로 마스크 생성
                linear_mask = (lpm_out.argmax(1) == targets).float().detach()
            
            loss_iam = (linear_mask * criterion(iam_out, targets))

            total_loss_main = loss_linear.mean() + loss_iam.mean()
            total_loss_main.backward()
            optimizer.step()

            # --- STEP 2: Second Update (LAM) ---
            optimizer2.zero_grad()
            
            lam_feat = model_lam.forward_features(inputs)[:, 0, :]
            lam_out = model_lam.linear(lam_feat)
            
            with torch.no_grad():
                # IAM의 예측을 가이드로 사용
                iam_mask = (iam_out.argmax(1) == targets).float().detach()
            
            loss_lam = (iam_mask * criterion(lam_out, targets))

            loss_lam.mean().backward()
            optimizer2.step()

        scheduler.step()
        scheduler2.step()
        
        valid_acc = validation_accuracy(model_lam, test_loader, device)
        print(f"Epoch {epoch+1} Valid Acc: {valid_acc:.4f} | Loss: {total_loss_main.item()} | LR: {scheduler.get_last_lr()[0]}")
        
if __name__ == '__main__':
    train()