import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
import os
import sys
sys.path.append("/home/ailab-lba/ym_workspace/GLFF/")
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

    # 1. 데이터 로더 (Label Noise 포함 버전 사용)
    train_loader, _, test_loader = get_dataloaders(
        data_dir="/home/ailab-lba/ym_workspace/dataset/ham10000/",
        batch_size=32,
        num_workers=4,
        random_seed=42,
        noise_rate=args.noise_rate
    )

    # 2. 기초 모델(DINOv2) 가중치 로드
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

    # IAM: 중간 어댑터 / LAM: 최종 어댑터 (최종 모델)
    model_iam = build_model()
    model_lam = build_model()
    # LPM: 특징 추출기를 고정하고 선형 층만 학습
    lpm_linear = nn.Linear(variant['embed_dim'], config['num_classes']).to(device)

    # 4. 옵티마이저 및 손실함수
    optimizer = torch.optim.Adam([
        {'params': lpm_linear.parameters(), 'lr': 1e-3},
        {'params': model_iam.parameters(), 'lr': 1e-3},
        {'params': model_lam.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-5)
    
    criterion = nn.CrossEntropyLoss(reduction='none') 
    
    # 5. 학습 루프
    warmup_epochs = 10 # 초기 안정화를 위한 웜업 
    for epoch in range(args.epoch):
        model_iam.train(); model_lam.train(); lpm_linear.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                feat = model_iam.forward_features(inputs)[:, 0, :]
            lpm_out = lpm_linear(feat)
            lpm_mask = (lpm_out.argmax(1) == targets).float() 

            # [IAM] Intermediate Adapter 학습
            iam_out = model_iam.forward_features(inputs)[:, 0, :]
            iam_out = model_iam.linear(iam_out)
            iam_mask = (iam_out.argmax(1) == targets).float() 

            # [LAM] Last Adapter 학습
            lam_out = model_lam.forward_features(inputs)[:, 0, :]
            lam_out = model_lam.linear(lam_out)

            loss_lpm = criterion(lpm_out, targets).mean()
            
            if epoch < warmup_epochs:
                loss_iam = criterion(iam_out, targets).mean()
                loss_lam = criterion(lam_out, targets).mean()
            else:
                # loss_iam = (lpm_mask * criterion(iam_out, targets)).mean()
                # loss_lam = (iam_mask * criterion(lam_out, targets)).mean()
                
                # 0 또는 1의 마스크는 그대로 유지하되, 하한선을 0.1로 설정
                # weight = 0.1 (틀렸을 때) / weight = 1.0 (맞았을 때)
                lpm_weight = 0.1 + 0.9 * lpm_mask
                iam_weight = 0.1 + 0.9 * iam_mask
                
                loss_iam = (lpm_weight * criterion(iam_out, targets)).mean()
    loss_lam = (iam_weight * criterion(lam_out, targets)).mean()

            total_loss = loss_lpm + loss_iam + loss_lam
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            pbar.set_postfix({"LAM_Loss": f"{loss_lam.item():.4f}", "LPM_Ratio": f"{lpm_mask.mean().item():.2f}"})

        valid_acc = validation_accuracy(model_lam, test_loader, device)
        print(f"Epoch {epoch+1} Valid Acc (LAM): {valid_acc:.4f} | LPM Ratio: {lpm_mask.mean().item():.4f}")

if __name__ == '__main__':
    train()