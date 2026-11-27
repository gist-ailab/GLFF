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
    # parser.add_argument('--warmup', type=int, default=3, help='Number of warmup epochs for co-teaching')
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

    model_iam = create_glff_model() 
    model_lam = create_glff_model() 
    # LPM은 가장 안정적인 global_encoder의 출력(embed_dim)을 입력으로 받음
    lpm_linear = nn.Linear(variant['embed_dim'], config['num_classes']).to(device) 

    # 3. 최적화 분리 (공식 코드 방식)
    # optimizer: lpm_linear와 model_iam을 함께 업데이트
    optimizer = torch.optim.Adam([
        {'params': lpm_linear.parameters(), 'lr': 1e-3},
        {'params': model_iam.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-5)
    
    # optimizer2: model_lam만 전용으로 업데이트
    optimizer2 = torch.optim.Adam(model_lam.parameters(), lr=1e-3, weight_decay=1e-5)
    
    criterion = nn.CrossEntropyLoss(reduction='none')

    # 스케줄러 추가 (공식 코드 권장)
    lr_decay = [int(0.5*args.epoch), int(0.75*args.epoch)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, lr_decay)

    # 4. 학습 루프
    for epoch in range(args.epoch):
        model_iam.train()
        model_lam.train()
        lpm_linear.train()
        
        model_iam.current_epoch = epoch
        model_lam.current_epoch = epoch

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}")

        for batch_idx, (inputs, masks, targets) in enumerate(pbar):
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
            
            # --- STEP 1: Main Update (LPM + IAM) ---
            optimizer.zero_grad()

            # [LPM] GLFF의 global_encoder에서 어댑터(Reins)가 없는 순수 특징 추출
            with torch.no_grad():
                # Reins를 거치지 않은 순수 Backbone 특징만 사용하여 가이드라인 고정
                feat = model_iam.global_encoder.forward_features_no_rein(inputs)[:, 0, :].detach()
            
            lpm_out = lpm_linear(feat)
            loss_linear = criterion(lpm_out, targets) # 마스킹 없이 무조건 반영 (엔진)

            # [IAM] GLFF 전체 모델(Global+Local Fusion) 예측
            iam_out = model_iam(inputs, masks)
            
            with torch.no_grad():
                # 공식 방식: LPM 예측이 맞은 샘플만 IAM 학습용 마스크로 사용
                linear_accurate = (lpm_out.argmax(1) == targets).float().detach()
            
            loss_iam = linear_accurate * criterion(iam_out, targets)

            # 합산 후 업데이트 (loss_linear가 살아있어 학습 중단 방지)
            total_loss_main = loss_linear.mean() + loss_iam.mean()
            total_loss_main.backward()
            optimizer.step()


            # --- STEP 2: Second Update (LAM) ---
            optimizer2.zero_grad()

            # [LAM] 최종 GLFF 모델 예측
            lam_out = model_lam(inputs, masks)
            
            with torch.no_grad():
                # 공식 방식: 앞선 IAM의 예측이 맞은 샘플만 LAM 학습용 마스크로 사용
                iam_accurate = (iam_out.argmax(1) == targets).float().detach()
            
            loss_lam = iam_accurate * criterion(lam_out, targets)

            # 독립적 업데이트
            loss_lam.mean().backward()
            optimizer2.step()

            pbar.set_postfix({
                "L_Main": f"{total_loss_main.item():.3f}", 
                "L_LAM": f"{loss_lam.mean().item():.3f}",
                "Ratio": f"{linear_accurate.mean().item():.2f}"
            })

        # 스케줄러 스텝
        scheduler.step()
        scheduler2.step()

        # 검증 (최종 모델인 LAM으로 수행)
        model_iam.eval(); model_lam.eval()
        acc = validation_accuracy(model_lam, test_loader, device)
        print(f"\n[Epoch {epoch+1}] Valid Acc (LAM): {acc:.4f} | LR: {scheduler.get_last_lr()[0]}")

if __name__ == "__main__":
    train()