# inference.py
import torch
from tqdm import tqdm

from net_hard import GlobalLocalLesionClassifier
from data.ham10000_seg import get_dataloaders_with_mask

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
        return: mask (B,1,224,224)
        """
        logits = self.sam_model(image)  # (B,1,H,W) or (B,H,W)

        if logits.dim() == 3:
            logits = logits.unsqueeze(1)

        prob = torch.sigmoid(logits)

        if self.use_soft:
            return prob                  # soft pseudo-mask
        else:
            return (prob > self.threshold).float()


def load_cls_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    model = GlobalLocalLesionClassifier(
        global_encoder=checkpoint["global_encoder"],
        local_encoder=checkpoint["local_encoder"],
        embed_dim=checkpoint["embed_dim"],
        num_classes=checkpoint["num_classes"],
        patch_size=checkpoint["patch_size"],
        tau=checkpoint["tau"],
        min_tokens=checkpoint["min_tokens"],
        fusion=checkpoint["fusion"],
        use_global_cls=checkpoint["use_global_cls"],
    )

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference(seg_model, cls_model, loader, device):
    preds, labels = [], []

    for images, _, gt in tqdm(loader, desc="Inference", ncols=100):
        images = images.to(device)
        gt = gt.to(device)

        masks = seg_model(images)              # ← SAM 추론
        logits = cls_model(images, masks)      # ← 기존 모델 그대로

        preds.append(logits.argmax(1).cpu())
        labels.append(gt.cpu())

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    acc = (preds == labels).float().mean()

    return acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SAM2Adapter model
    sam_model = load_sam2adapter_checkpoint("./S1_train/save/ham10000_sam_final/model_epoch_last.pth")
    sam_model.to(device)
    seg_model = SegmentationInferenceWrapper(sam_model)

    # Classification model
    cls_model = load_cls_model("cls_ckpt.pth", device)

    # DataLoader 
    _, _, test_loader = get_dataloaders_with_mask(
        data_dir="/data/ham10000",
        batch_size=32,
        num_workers=4,
        random_seed=42
    )

    # Inference
    acc = run_inference(seg_model, cls_model, test_loader, device)
    print(f"Final Accuracy: {acc:.4f}")
