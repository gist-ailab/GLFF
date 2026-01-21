# inference.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,6,7"
import sys
# sys.path.append("/SSDd/youmin_park/GLFF/")
import torch
from tqdm import tqdm
import importlib
import argparse

import numpy as np
import torchvision
from PIL import Image

import rein
import dino_variant
from util import read_conf
from net_hard import GlobalLocalLesio,nClassifier
# from net_soft import GlobalLocalLesionClassifier
from data.ham10000_seg import get_dataloaders_with_mask
# from data.ham10000_seg_trainset_split import get_dataloaders_train_only
import S1_models 

def save_overlay_batch(
    images: torch.Tensor,      # (B,3,224,224), normalized
    masks: torch.Tensor,       # (B,1,224,224), [0,1]
    save_dir: str,
    start_idx: int,
    alpha: float = 0.5,
):
    """
    RGB 이미지 위에 mask를 red overlay로 저장
    """
    os.makedirs(save_dir, exist_ok=True)

    # ImageNet normalize 복원
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1,3,1,1)
    images = images * std + mean
    images = images.clamp(0, 1)

    images = images.cpu().numpy()
    masks = masks.cpu().numpy()

    B = images.shape[0]

    for i in range(B):
        img = images[i].transpose(1, 2, 0)          # (H,W,3)
        mask = masks[i, 0]                           # (H,W)

        # red overlay
        overlay = img.copy()
        overlay[..., 0] = np.clip(
            overlay[..., 0] * (1 - alpha) + mask * alpha,
            0, 1
        )

        out = (overlay * 255).astype(np.uint8)
        Image.fromarray(out).save(
            os.path.join(save_dir, f"{start_idx + i:06d}.png")
        )




class SegmentationInferenceWrapper(torch.nn.Module):
    def __init__(self, sam_model, threshold=0.5, use_soft=True):
        super().__init__()
        self.sam_model = sam_model
        self.threshold = threshold
        self.use_soft = use_soft

    @torch.no_grad()
    def forward(self, image):
        """
        image: (B,3,224,224)
        return: (B,1,224,224)
        """
        masks = self.sam_model.infer(image)   

        prob = torch.sigmoid(masks)

        if self.use_soft:
            return prob
        else:
            return (prob > self.threshold).float()


def load_sam2adapter_checkpoint(
    ckpt_path: str,
    device: torch.device,
    inp_size: int = 224,
    loss: str = "iou",
):

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
            "inp_size": inp_size,
            "encoder_mode": encoder_mode,
            "loss": loss,
        }
    }

    model = S1_models.make(model_spec)
    model.to(device)
    model.eval()

    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("model", sd.get("state_dict", sd))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    model.load_state_dict(sd, strict=False)

    for p in model.parameters():
        p.requires_grad_(False)

    return model



def load_cls_model(args, ckpt_path, device):
    # checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)['state_dict']
    # print(checkpoint.keys())

    config = read_conf('conf/data/'+args.data+'.yaml')
    variant = dino_variant._small_variant
    hub_model = torch.hub.load('facebookresearch/dinov2', dino_variant._small_dino)
    dino_state_dict = hub_model.state_dict()

    # global/local encoder
    global_encoder = rein.ReinsDinoVisionTransformer(**variant)
    global_encoder.load_state_dict(dino_state_dict, strict=False)

    local_encoder = rein.ReinsDinoVisionTransformer(**variant)
    local_encoder.load_state_dict(dino_state_dict, strict=False)

    model = GlobalLocalLesionClassifier(
        global_encoder=global_encoder,
        local_encoder=local_encoder,
        embed_dim=variant["embed_dim"],
        num_classes=config["num_classes"],
        patch_size=variant["patch_size"],
        tau=0.001,
        min_tokens=4,
        num_heads=8,
        cross_attn_layers=1,
        fusion="concat",        # "concat" or "gate"
        hidden_dim=512,
        # dropout=0.2,
        use_global_cls=True
    )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference(seg_model, cls_model, loader, device, save_overlay_dir=None):
    preds, labels = [], []

    for images, _, gt in tqdm(loader, desc="Inference", ncols=100):
        images = images.to(device)
        gt = gt.to(device)

        masks = seg_model(images)              # ← SAM 추론
        logits = cls_model(images, masks)      # ← 기존 모델 그대로
        
        img_counter = 0
        # 🔹 overlay 저장
        if save_overlay_dir is not None:
            save_overlay_batch(
                images,
                masks,
                save_overlay_dir,
                start_idx=img_counter,
                alpha=0.5,
            )
            img_counter += images.size(0)
            
            

        preds.append(logits.argmax(1).cpu())
        labels.append(gt.cpu())

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    acc = (preds == labels).float().mean()

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='ham10000')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SAM2Adapter model
    sam_model = load_sam2adapter_checkpoint("S1_save/ham10000_sam_final/model_epoch_last.pth", device)
    seg_model = SegmentationInferenceWrapper(sam_model).to(device)

    # Classification model
    cls_model = load_cls_model(args,"S2_save/glff_ham10000_rein_s_hard_50/last.pth.tar", device)

    _, _, test_loader = get_dataloaders_with_mask(
        data_dir="/SSDd/youmin_park/dataset/ham10000",
        batch_size=32,
        num_workers=4,
        random_seed=42
    )
    
    # _, test_loader = get_dataloaders_train_only(
    #     data_dir="/SSDd/youmin_park/dataset/ham10000",
    #     batch_size=32,
    #     num_workers=4,
    #     random_seed=42
    # )

    # Inference
    # acc = run_inference(seg_model, cls_model, test_loader, device)
    acc = run_inference(
        seg_model,
        cls_model,
        test_loader,
        device,
        save_overlay_dir="overlay_results/ham10000_test"
    )

    print(f"Final Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()