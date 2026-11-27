import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,6,7"

import random

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
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
# HAM10000 Dataset (Seg)
# -------------------------
class HAM10000SegmentationDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        mask_dir,
        image_transform=None,
        mask_transform=None,
        use_mask=True,
    ):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.use_mask = use_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx, 0]

        img_path = os.path.join(self.image_dir, img_id + ".jpg")
        image = Image.open(img_path).convert("RGB")

        if self.use_mask:
            mask_path = os.path.join(self.mask_dir, img_id + "_segmentation.png")
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
            else:
                w, h = image.size
                mask = Image.new("L", (w, h), 0)
        else:
            w, h = image.size
            mask = Image.new("L", (w, h), 0)

        # 동일한 spatial transform
        if self.image_transform is not None:
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.image_transform(image)

            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)

        return {
            "inp": image,               # [3,H,W]
            "gt": (mask > 0).float(),   # [1,H,W]
        }


def get_transforms(train=True, img_size=256, crop_size=224):
    from torchvision import transforms

    if train:
        image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomCrop((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop((crop_size, crop_size)),
            transforms.ToTensor(),
        ])
    else:
        image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop((crop_size, crop_size)),
            transforms.ToTensor(),
        ])
    return image_transform, mask_transform


def make_data_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    split_ratio: float,
    random_seed: int,
    img_size: int,
    crop_size: int,
    is_ddp: bool,
    rank: int,
    world_size: int,
):
    csv_file = os.path.join(data_dir, "ISIC2018_Task3_Training_GroundTruth.csv")
    image_dir = os.path.join(data_dir, "train")
    mask_dir = os.path.join(data_dir, "ham10000_mask")

    train_img_tf, train_mask_tf = get_transforms(True, img_size, crop_size)
    val_img_tf, val_mask_tf = get_transforms(False, img_size, crop_size)

    full_dataset = HAM10000SegmentationDataset(
        csv_file=csv_file,
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_transform=None,
        mask_transform=None,
        use_mask=True,
    )

    train_len = int(len(full_dataset) * split_ratio)
    val_len = len(full_dataset) - train_len

    train_set, val_set = random_split(
        full_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(random_seed),
    )

    train_set.dataset.image_transform = train_img_tf
    train_set.dataset.mask_transform = train_mask_tf
    val_set.dataset.image_transform = val_img_tf
    val_set.dataset.mask_transform = val_mask_tf

    if is_ddp:
        train_sampler = DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        train_shuffle = False
        val_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True
        val_shuffle = False

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=val_shuffle,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def make_train_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    random_seed: int,
    img_size: int,
    crop_size: int,
    is_ddp: bool,
    rank: int,
    world_size: int,
):
    # -------------------------
    # Path
    # -------------------------
    csv_file = os.path.join(data_dir, "ISIC2018_Task3_Training_GroundTruth.csv")
    image_dir = os.path.join(data_dir, "train")
    mask_dir = os.path.join(data_dir, "ham10000_mask")

    # -------------------------
    # Transform (train only)
    # -------------------------
    train_img_tf, train_mask_tf = get_transforms(
        train=True,
        img_size=img_size,
        crop_size=crop_size,
    )

    # -------------------------
    # Dataset (FULL train set)
    # -------------------------
    train_dataset = HAM10000SegmentationDataset(
        csv_file=csv_file,
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_transform=train_img_tf,
        mask_transform=train_mask_tf,
        use_mask=True,
    )

    # -------------------------
    # Sampler
    # -------------------------
    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    # -------------------------
    # DataLoader
    # -------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,   # ★ DDP 필수
    )

    return train_loader


# -------------------------
# Train / Val
# -------------------------
def train_one_epoch(train_loader, model, device, is_ddp, rank):
    model.train()
    pbar = tqdm(total=len(train_loader), leave=False, desc="train") if is_main_process(rank) else None

    loss_list = []

    for batch in train_loader:
        inp = batch["inp"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        model.set_input(inp, gt)
        model.optimize_parameters()
        loss = model.loss_G.detach()
        loss_list.append(loss)

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss_tensor = torch.stack(loss_list).mean()

    if is_ddp:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= dist.get_world_size()

    return loss_tensor.item()


@torch.no_grad()
def validate_one_epoch(val_loader, model, device, is_ddp, rank, eval_type="kvasir"):
    model.eval()

    if eval_type == "kvasir":
        metric_fn = S1_utils.calc_kvasir
        metric1, metric2 = "dice", "iou"
    else:
        # 필요하면 확장
        raise ValueError(f"Unsupported eval_type: {eval_type}")

    pbar = tqdm(total=len(val_loader), leave=False, desc="val") if is_main_process(rank) else None

    sum_m1 = 0.0
    sum_m2 = 0.0
    cnt = 0

    for batch in val_loader:
        inp = batch["inp"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        pred = torch.sigmoid(model.infer(inp))

        r1, r2, _, _ = metric_fn(pred, gt)

        bs = pred.size(0)
        sum_m1 += float(r1) * bs
        sum_m2 += float(r2) * bs
        cnt += bs

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    t_sum_m1 = torch.tensor(sum_m1, device=device)
    t_sum_m2 = torch.tensor(sum_m2, device=device)
    t_cnt = torch.tensor(cnt, device=device)

    if is_ddp:
        dist.all_reduce(t_sum_m1, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_sum_m2, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_cnt, op=dist.ReduceOp.SUM)

    m1 = (t_sum_m1 / t_cnt).item()
    m2 = (t_sum_m2 / t_cnt).item()

    return m1, m2, metric1, metric2


# -------------------------F
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/ailab-lba/ym_workspace/dataset/ham10000/")
    parser.add_argument("--sam_checkpoint", type=str, default="./S1_pretrained/sam2.1_hiera_large.pt")
    parser.add_argument("--save_dir", type=str, default="./S1_save/ham10000_sam2.1_large_seg")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--crop_size", type=int, default=224)

    parser.add_argument("--inp_size", type=int, default=224)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_min", type=float, default=5e-6)

    parser.add_argument("--eval_type", type=str, default="kvasir")  # dice/iou
    parser.add_argument("--loss", type=str, default="iou")          # bce/bbce/iou

    args = parser.parse_args()

    rank, local_rank, world_size, device, is_ddp = ddp_setup()
    seed_all(args.seed)

    log, writer = S1_utils.set_save_path(args.save_dir, remove=False)

    if is_main_process(rank):
        log(f"DDP={is_ddp} rank={rank} local_rank={local_rank} world_size={world_size}")
        log(f"data_dir={args.data_dir}")
        log(f"save_dir={args.save_dir}")

    train_loader, val_loader = make_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_ratio=args.split_ratio,
        random_seed=args.seed,
        img_size=args.img_size,
        crop_size=args.crop_size,
        is_ddp=is_ddp,
        rank=rank,
        world_size=world_size,
    )

    # train_loader = make_train_loader(
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     random_seed=args.seed,
    #     img_size=args.img_size,
    #     crop_size=args.crop_size,
    #     is_ddp=is_ddp,
    #     rank=rank,
    #     world_size=world_size,
    # )


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
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"params: trainable={trainable_params} / total={total_params}")

    best_metric = -1e18

    for epoch in range(1, args.epochs + 1):
        if is_ddp and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(train_loader, model, device, is_ddp, rank)
        lr_scheduler.step()

        # 매 epoch 검증
        val_m1, val_m2, m1_name, m2_name = validate_one_epoch(
            val_loader, model, device, is_ddp, rank, eval_type=args.eval_type
        )

        if is_main_process(rank):
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar(f"{m1_name}/val", val_m1, epoch)
            writer.add_scalar(f"{m2_name}/val", val_m2, epoch)

            log(f"epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_{m1_name}={val_m1:.4f} | val_{m2_name}={val_m2:.4f}")
            
            save_ckpt(model, args.save_dir, "last")

            if val_m1 > best_metric:
                best_metric = val_m1
                save_ckpt(model, args.save_dir, "best")

            writer.flush()
            
        # log(f"epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f}")

        # save_ckpt(model, args.save_dir, "last")

        if is_ddp:
            dist.barrier()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()