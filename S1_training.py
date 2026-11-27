import os
import random

import torch
import torch.distributed as dist


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        is_ddp = True
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_ddp = False
    return rank, local_rank, world_size, device, is_ddp


def is_main_process(rank: int) -> bool:
    return rank == 0


def load_sam_ckpt(model, ckpt_path, rank=0):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    else:
        state = ckpt

    msg = model.load_state_dict(state, strict=False)
    if is_main_process(rank):
        print(f"[CKPT] loaded: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")


def freeze_encoder_except_prompt(model):
    for name, p in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            p.requires_grad_(False)


def save_ckpt(model, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{name}.pth"))
