import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np

import S1_models  # 사용자의 모델 정의 파일

@torch.no_grad()
def generate_masks():
    # --- 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/home/ailab-lba/ym_workspace/dataset/ham10000/"
    model_path = "./S1_save/ham10000_sam2.1_large_seg/model_epoch_best.pth" # 학습된 경로
    output_base_dir = "/home/ailab-lba/ym_workspace/dataset/ham10000/inferred_masks"
    img_size = 224
    
    # 1. 모델 로드 (학습 시와 동일한 spec)
    encoder_mode = {
        "name": "sam", "embed_dim": 768, "patch_size": 16, "depth": 12,
        "num_heads": 12, "mlp_ratio": 4, "out_chans": 256, "qkv_bias": True,
        "use_rel_pos": True, "window_size": 14, "global_attn_indexes": [2, 5, 8, 11],
        "prompt_embed_dim": 256,
    }
    model_spec = {
        "name": "sam",
        "args": {"inp_size": img_size, "encoder_mode": encoder_mode, "loss": "iou"}
    }
    
    model = S1_models.make(model_spec).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # 2. 처리할 대상 리스트 (Validation, Test)
    targets = [
        {"csv": "ISIC2018_Task3_Validation_GroundTruth.csv", "dir": "valid", "save": "valid_masks"},
        {"csv": "ISIC2018_Task3_Test_GroundTruth.csv", "dir": "test", "save": "test_masks"}
    ]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for target in targets:
        csv_path = os.path.join(data_dir, target["csv"])
        img_dir = os.path.join(data_dir, target["dir"])
        save_dir = os.path.join(output_base_dir, target["save"])
        os.makedirs(save_dir, exist_ok=True)

        df = pd.read_csv(csv_path)
        print(f"Processing {target['dir']} set...")

        for img_id in tqdm(df.iloc[:, 0]):
            img_path = os.path.join(img_dir, img_id + ".jpg")
            if not os.path.exists(img_path): continue

            # 이미지 로드 및 전처리
            raw_img = Image.open(img_path).convert("RGB")
            w, h = raw_img.size
            img_t = transform(raw_img).unsqueeze(0).to(device)

            # 추론
            pred = model.infer(img_t) # 모델 내부 infer 함수 호출
            pred_mask = torch.sigmoid(pred) > 0.5
            pred_mask = pred_mask.float().cpu().squeeze().numpy()

            # 원본 사이즈로 복원 및 저장
            mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
            mask_img = mask_img.resize((w, h), resample=Image.NEAREST)
            mask_img.save(os.path.join(save_dir, img_id + "_segmentation.png"))

    print("Mask generation completed.")

if __name__ == "__main__":
    generate_masks()