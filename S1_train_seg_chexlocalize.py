import json
import os
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as coco_mask

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

import S1_models
import S1_utils
from S1_training import (
    ddp_setup,
    freeze_encoder_except_prompt,
    is_main_process,
    load_sam_ckpt,
    save_ckpt,
    seed_all,
)


# -------------------------
# 1. CheXlocalize Dataset
# -------------------------
class CheXlocalizeDataset(Dataset):
    def __init__(self, json_path, img_dir, image_transform=None, mask_transform=None):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            self.gt_data = json.load(f)
        
        self.img_dir = img_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        
        self.samples = []
        for img_id, pathologies in self.gt_data.items():
            for path_name, rle in pathologies.items():
                if rle.get('counts'):
                    self.samples.append((img_id, path_name, rle))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, path_name, rle = self.samples[idx]
        
        rel_path = img_id.replace('_', '/', 2)
        img_path = os.path.join(self.img_dir, rel_path + ".jpg")
        
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Cannot find image for ID: {img_id}")

        image = Image.open(img_path).convert("RGB")
        mask_np = coco_mask.decode(rle)
        mask = Image.fromarray(mask_np)

        # 이미지와 마스크가 동일하게 변형(Resize)되도록 동일 시드 사용
        if self.image_transform is not None:
            seed = random.randint(0, 2**32 - 1)
            
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.image_transform(image)
            
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)

        mask = (mask > 0).float() # [1, H, W]

        # SAM 2.1 학습용 랜덤 포인트 추출 (생략 가능하면 패스)
        coords = torch.nonzero(mask[0]) 
        if len(coords) > 0:
            selected_idx = random.randint(0, len(coords) - 1)
            point = coords[selected_idx].flip(0).float() # [x, y]
            point_label = torch.tensor([1.0])
        else:
            point = torch.tensor([0.0, 0.0])
            point_label = torch.tensor([0.0])

        return {
            "inp": image,
            "gt": mask,
            "point": point,
            "point_label": point_label,
            "pathology": path_name
        }
# -------------------------
# 2. Train / Test Setup
# -------------------------
def make_chex_loaders(args, is_ddp, rank, world_size):
    # 1. 이미지용 Transform (노멀라이즈 포함)
    image_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. 마스크용 Transform (노멀라이즈 제외!)
    mask_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    base_data = os.path.join(args.data_dir, "chexlocalize")
    
    # Dataset 생성 시 두 가지 transform을 모두 전달하도록 수정
    train_set = CheXlocalizeDataset(
        json_path=os.path.join(base_data, "CheXlocalize/gt_segmentations_val.json"),
        img_dir=os.path.join(base_data, "CheXpert/val"),
        image_transform=image_transform,
        mask_transform=mask_transform
    )
    
    test_set = CheXlocalizeDataset(
        json_path=os.path.join(base_data, "CheXlocalize/gt_segmentations_test.json"),
        img_dir=os.path.join(base_data, "CheXpert/test"),
        image_transform=image_transform,
        mask_transform=mask_transform
    )

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.num_workers, pin_memory=True)

    return train_loader, test_loader

# -------------------------
# 3. Evaluation Logic
# -------------------------
@torch.no_grad()
def evaluate(loader, model, device, rank):
    model.eval()
    ious = []
    
    pbar = tqdm(loader, desc="Evaluating", leave=False) if is_main_process(rank) else loader
    
    for batch in pbar:
        inp = batch["inp"].to(device)
        gt = batch["gt"].to(device)
        
        # 추론: Stage 2/테스트를 위해 프롬프트 없는 자동 모드(infer) 사용
        pred = model.infer(inp) 
        pred = (torch.sigmoid(pred) > 0.5).float()

        # IoU 계산
        intersection = (pred * gt).sum(dim=(2, 3))
        union = (pred + gt).clamp(0, 1).sum(dim=(2, 3))
        iou = (intersection + 1e-7) / (union + 1e-7)
        
        ious.append(iou.mean().item())
    
    mIoU = np.mean(ious)
    return mIoU

# -------------------------
# 4. Main Function
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    # 파일 트리 구조에 맞춰 ym_workspace/GLFF/CheXlocalize로 기본값 설정
    parser.add_argument("--data_dir", type=str, default="/home/ailab-lba/ym_workspace/dataset/CheXlocalize")
    parser.add_argument("--sam_checkpoint", type=str, default="./S1_pretrained/sam2.1_hiera_large.pt")
    parser.add_argument("--save_dir", type=str, default="./S1_save/chex_sam2.1_seg")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--inp_size", type=int, default=512)
    parser.add_argument("--loss", type=str, default="iou")          # bce/bbce/iou
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_min", type=float, default=5e-6)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rank, local_rank, world_size, device, is_ddp = ddp_setup()
    seed_all(args.seed)

    # 데이터 로더 생성
    train_loader, test_loader = make_chex_loaders(args, is_ddp, rank, world_size)
    
    encoder_mode = {
        "name": "sam",
        "embed_dim": 768,
        "patch_size": 16,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "out_chans": 256,
        "qkv_bias": True,
        "use_rel_pos": True,
        "window_size": 14,
        "global_attn_indexes": [2, 5, 8, 11],
        "prompt_embed_dim": 256,
    }


    model_spec = {
        "name": "sam",
        "args": {
            "inp_size": args.inp_size,
            "encoder_mode": encoder_mode,
            "loss": args.loss,
        }
    }

    model = S1_models.make(model_spec).to(device)
    
    optimizer_spec = {
        "name": "adamw",
        "args": {"lr": args.lr, "weight_decay": args.weight_decay},
    }
    optimizer = S1_utils.make_optimizer(model.parameters(), optimizer_spec)
    model.optimizer = optimizer

    lr_scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr_min)

    # ckpt load + freeze
    load_sam_ckpt(model, args.sam_checkpoint, rank=rank)
    freeze_encoder_except_prompt(model)

    
    if is_main_process(rank):
        print(f"Start Training: {len(train_loader.dataset)} samples for Stage 1 adaptation.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        if is_ddp: train_loader.sampler.set_epoch(epoch)
        
        epoch_loss = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main_process(rank))
        
        for batch in pbar:
            inp = batch["inp"].to(device)
            gt = batch["gt"].to(device)
            point = batch["point"].to(device)
            label = batch["point_label"].to(device)

            # SAM 2.1 학습 (Teacher Forcing: 정답 포인트 제공)
            model.set_input(inp, gt)
            # model.optimize_parameters(optimizer)
            model.optimize_parameters()
            
            loss = model.loss_G.item() if hasattr(model, 'loss_G') else 0
            epoch_loss.append(loss)
            pbar.set_postfix(loss=f"{loss:.4f}")
            
        # define best_miou for saving best model
        if epoch == 1:
            best_miou = 0.0 

        # 정기 평가 및 저장
        if epoch % 1 == 0 or epoch == args.epochs:
            test_miou = evaluate(test_loader, model, device, rank)
            if is_main_process(rank):
                print(f"\n[*] Epoch {epoch} Test mIoU: {test_miou:.4f}")
                # 체크포인트 저장 로직 추가 가능
                save_ckpt(model, args.save_dir, f"epoch_{epoch}")

                # to save best epoch based on test mIoU
                if epoch == 1 or test_miou > best_miou:
                    best_miou = test_miou
                    save_ckpt(model, args.save_dir, "best")

    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()