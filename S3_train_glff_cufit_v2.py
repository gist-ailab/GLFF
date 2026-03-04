import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
import os
import sys
# sys.path.append("/SSDd/youmin_park/GLFF/")
sys.path.append("/home/ailab-lba/ym_workspace/GLFF/")
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


def rein_forward(model, inputs):
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    output = torch.softmax(output, dim=1)
    return output

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
    
    # 각 샘플별로 클린 여부를 판단하기 위해 reduction='none' 사용
    criterion = nn.CrossEntropyLoss(reduction='none')

    # saver = timm.utils.CheckpointSaver(
    #     model.module if isinstance(model, torch.nn.DataParallel) else model,
    #     optimizer,
    #     checkpoint_dir=save_path,
    #     max_history=1
    # )
    
    saver = timm.utils.CheckpointSaver(
        model_lam, optimizer, checkpoint_dir=save_path, max_history=1
    )

    
    # 4. 학습 루프
    for epoch in range(args.epoch):
        model_iam.train()
        model_lam.train()
        lpm_linear.train()
        
        # GLFF 내부의 에폭 정보 업데이트 (Cross-attention warm-up용)
        model_iam.current_epoch = epoch
        model_lam.current_epoch = epoch

        total_lam_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}")

        for inputs, masks, targets in pbar:
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)

            # [단계 1] LPM: 특징 추출기를 고정한 채 선형 층만 학습 (노이즈에 가장 강건함)
            with torch.no_grad():
                # IAM의 global_encoder를 빌려 특징만 추출
                feat = model_iam.global_encoder.forward_features(inputs)[:, 0, :]
                # feat = model_iam.global_encoder.forward_features_no_rein(inputs)[:, 0, :].detach()
            lpm_out = lpm_linear(feat)
            # Agreement Criterion: LPM이 예측한 결과가 레이블과 같으면 클린 샘플로 간주
            lpm_mask = (lpm_out.argmax(1) == targets).float() 
            # print(f"Epoch {epoch+1} - LPM Clean Sample Ratio: {lpm_mask.mean().item():.4f}")
            
            # [단계 2] IAM: LPM이 거른 '쉬운' 데이터를 활용해 Global-Local 특징 학습
            iam_out = model_iam(inputs, masks)
            # IAM이 예측에 성공한 샘플을 LAM 학습용으로 전달
            iam_mask = (iam_out.argmax(1) == targets).float() 
            # print(f"Epoch {epoch+1} - IAM Clean Sample Ratio: {iam_mask.mean().item():.4f}") 
            
            # [단계 3] LAM: IAM이 최종 검증한 가장 깨끗한 데이터로 학습 (최종 모델)
            lam_out = model_lam(inputs, masks)
            # print(f"Epoch {epoch+1} - LAM Clean Sample Ratio: { (iam_mask * (lam_out.argmax(1) == targets).float()).mean().item():.4f} ")

            # 손실 계산: 이전 단계의 필터링 결과(Indicator)를 곱함
            loss_lpm = criterion(lpm_out, targets).mean()
            # loss_iam = (lpm_mask * criterion(iam_out, targets)).mean()
            # loss_lam = (iam_mask * criterion(lam_out, targets)).mean()
            
            warmup_epochs = args.warmup # 초기 10에포크 동안은 모든 데이터 학습

            if epoch < warmup_epochs:
                # 필터링 없이 모든 샘플에 대해 손실 계산
                loss_iam = criterion(iam_out, targets).mean()
                loss_lam = criterion(lam_out, targets).mean()
            else:
                # Warm-up 이후에는 논문대로 클린 샘플만 엄격하게 필터링
                loss_iam = (lpm_mask * criterion(iam_out, targets)).mean()
                loss_lam = (iam_mask * criterion(lam_out, targets)).mean()
            
            total_loss = loss_lpm + loss_iam + loss_lam
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_lam_loss += loss_lam.item()
            pbar.set_postfix({"LAM_Loss": f"{loss_lam.item():.4f}"})

        # 최종 모델(LAM)로 성능 측정
        acc = validation_accuracy(model_lam, test_loader, device)
        print(f"\n[Epoch {epoch+1}] Valid Acc (LAM): {acc:.4f}")
        # 각 모듈 별 클린 샘플 비율도 출력
        print(f"Epoch {epoch+1} - LPM Clean Sample Ratio: {lpm_mask.mean().item():.4f}")
        print(f"Epoch {epoch+1} - IAM Clean Sample Ratio: {iam_mask.mean().item():.4f}")
        print(f"Epoch {epoch+1} - LAM Clean Sample Ratio: { (iam_mask * (lam_out.argmax(1) == targets).float()).mean().item():.4f} ")
        print("-" * 50)

if __name__ == "__main__":
    train()