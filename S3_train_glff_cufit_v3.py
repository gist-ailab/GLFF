import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import timm
import numpy as np
import torch.nn.functional as F

# 사용자 정의 모듈 (파일이 동일 경로에 있어야 함)
from net_hard import GlobalLocalLesionClassifier
import rein
import dino_variant
from data.ham10000_seg import get_dataloaders_with_mask
from util import read_conf
from util.metric_seg import validation_accuracy
from util.forward import rein_forward


def set_requires_grad(model, layers_to_train):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_train):
            param.requires_grad = True
        else:
            param.requires_grad = False

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='ham10000')
    parser.add_argument('--data_dir', type=str, default="/home/ailab-lba/ym_workspace/dataset/ham10000/")
    parser.add_argument('--noise_rate', type=float, default=0.2)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_path', '-s', type=str, default='glff_cufit_ham10000')
    parser.add_argument('--warmup', type=int, default=3, help='Number of warmup epochs for co-teaching')
    args = parser.parse_args()

    config = read_conf('conf/data/'+args.data+'.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant = dino_variant._small_variant
    save_dir = f"{args.save_path}_noise_{args.noise_rate}_epoch_{args.epoch}"
    save_path = os.path.join('./S3_save/', save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 1. 데이터 로더 설정 (Label Noise 및 Mask 포함)
    # train_loader는 (이미지, 마스크, 레이블)return
    train_loader, _, test_loader = get_dataloaders_with_mask(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        noise_rate=args.noise_rate,
        random_seed=42
    )

    # 2. 기초 모델 로드 및 가중치 준비
    hub_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dino_sd = hub_model.state_dict()
    
    def get_keep_rate(epoch, total_epochs, noise_rate, warmup=5):
            if epoch < warmup:
                return 1.0 # 초기 5에포크는 모든 데이터 학습 (Warmup)
            # 이후 epoch에 따라 noise_rate까지 선형적으로 감소
            return 1.0 - min((epoch - warmup) / (total_epochs - warmup), 1.0) * noise_rate

    def create_glff_model():
            g_enc = rein.ReinsDinoVisionTransformer(**variant)
            l_enc = rein.ReinsDinoVisionTransformer(**variant)
            g_enc.load_state_dict(dino_sd, strict=False)
            l_enc.load_state_dict(dino_sd, strict=False)
            
            model = GlobalLocalLesionClassifier(
                global_encoder=g_enc,
                local_encoder=l_enc,
                embed_dim=variant['embed_dim'], # 384,
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
            
            return model.to(device)

    # CUFIT의 세 모듈: LPM(선형), IAM(GLFF 어댑터), LAM(최종 GLFF 어댑터)
    model_iam = create_glff_model() 
    model_lam = create_glff_model() 
    lpm_linear = nn.Linear(variant['embed_dim'], 7).to(device) 

    # 3. 최적화 및 손실 함수
    optimizer = torch.optim.Adam([
        {'params': lpm_linear.parameters(), 'lr': 1e-3},
        {'params': model_iam.parameters(), 'lr': 1e-3},
        {'params': model_lam.parameters(), 'lr': 1e-3}
    ])
    

    criterion_nnd = nn.CrossEntropyLoss(reduction='none') # 필터링용
    criterion_ce = nn.CrossEntropyLoss() # 기본 학습용
    kl_loss = nn.KLDivLoss(reduction='batchmean') # 전략 2: 일관성용
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(0.5*args.epoch), int(0.75*args.epoch)])
    
    
    # saver = timm.utils.CheckpointSaver(
    #     model.module if isinstance(model, torch.nn.DataParallel) else model,
    #     optimizer,
    #     checkpoint_dir=save_path,
    #     max_history=1
    # )
    
    saver = timm.utils.CheckpointSaver(
        model_lam, optimizer, checkpoint_dir=save_path, max_history=1
    )



    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    saver = timm.utils.CheckpointSaver(
        model_lam, optimizer, checkpoint_dir=save_path, max_history=1
    )

    for epoch in range(args.epoch):
        model_iam.train()
        model_lam.train()
        lpm_linear.train()
        
        # 현재 에포크에 맞는 Keep Rate 계산
        current_keep_rate = get_keep_rate(epoch, args.epoch, args.noise_rate)
        
        total_lam_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}")

        for inputs, masks, targets in pbar:
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)

            # --- [단계 1] LPM: Small-Loss Selection ---
            with torch.no_grad():
                feat = model_iam.global_encoder.forward_features_no_rein(inputs)[:, 0, :].detach()
            lpm_out = lpm_linear(feat)
            
            lpm_losses = criterion_nnd(lpm_out, targets)
            _, lpm_idx = torch.sort(lpm_losses)
            
            num_keep = int(current_keep_rate * len(targets))
            lpm_mask = torch.zeros_like(lpm_losses)
            lpm_mask[lpm_idx[:num_keep]] = 1.0

            # --- [단계 2] IAM: 학습 및 Soft Label 생성 ---
            iam_out = model_iam(inputs, masks)
            iam_prob = F.softmax(iam_out, dim=1)
            loss_iam = (lpm_mask * criterion_nnd(iam_out, targets)).mean()

            # --- [단계 3] LAM: Label Refurbishment ---
            lam_out = model_lam(inputs, masks)
            lam_prob = F.log_softmax(lam_out, dim=1)

            iam_losses = criterion_nnd(iam_out, targets)
            _, iam_idx = torch.sort(iam_losses)
            iam_mask = torch.zeros_like(iam_losses)
            iam_mask[iam_idx[:num_keep]] = 1.0 

            # Refurbishment 적용 손실
            loss_lam_clean = (iam_mask * criterion_nnd(lam_out, targets)).mean()
            loss_lam_noisy = ((1 - iam_mask) * kl_loss(lam_prob, iam_prob.detach())).mean()
            
            # Consistency & Total Loss
            loss_consistency = kl_loss(lam_prob, iam_prob.detach())
            loss_lpm = criterion_ce(lpm_out, targets)
            
            total_loss = loss_lpm + loss_iam + (loss_lam_clean + 0.5 * loss_lam_noisy) + 0.1 * loss_consistency
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_lam_loss += (loss_lam_clean + loss_lam_noisy).item()
            pbar.set_postfix({
                "Keep": f"{current_keep_rate:.2f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.6f}",
                "Loss": f"{total_lam_loss/len(pbar):.4f}"
            })

        # 에포크 종료 후 LR 스케줄러 업데이트
        lr_scheduler.step()

        # 검증 및 결과 출력
        model_lam.eval()
        with torch.no_grad():
            acc = validation_accuracy(model_lam, test_loader, device)
        
        print(f"\n[Epoch {epoch+1}] Valid Acc (LAM): {acc:.4f} | Final LR: {lr_scheduler.get_last_lr()[0]:.6f}")
        
        # 모델 저장
        saver.save_checkpoint(epoch, metric=acc)

if __name__ == "__main__":
    train()