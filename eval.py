import os
import sys
sys.path.append("/SSDe/youmin_park/GLFF/")

import argparse
import torch
import torch.nn as nn
import numpy as np

from util import read_conf, validation_accuracy, validation_accuracy_lora
from data.ham10000 import get_dataloaders

import rein
import adaptformer
import dino_variant

# -------------------------------------------------
# Utils
# -------------------------------------------------
def seed_for_init(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def rein_forward(model, inputs):
    backbone = model.module if isinstance(model, torch.nn.DataParallel) else model
    feats = backbone.forward_features(inputs)[:, 0, :]
    logits = backbone.linear(feats)
    return logits

# -------------------------------------------------
# Main Evaluation
# -------------------------------------------------
def main():
    seed_for_init(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ham10000')
    parser.add_argument('--adapter', type=str, default='rein',
                        choices=['rein', 'adaptformer', 'lora'])
    parser.add_argument('--netsize', type=str, default='s',
                        choices=['s', 'b', 'l'])
    parser.add_argument('--ckpt', type=str, default='S0_save/only_rgb_50/last.pth.tar')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    config = read_conf(f'conf/data/{args.data}.yaml')
    data_path = config['data_root']

    _, _, test_loader = get_dataloaders(
        data_dir=data_path,
        batch_size=args.batch_size,
        num_workers=4,
        random_seed=42
    )

    test_loader = torch.utils.data.DataLoader(
        test_loader.dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )

    # -------------------------------------------------
    # Model definition
    # -------------------------------------------------
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    elif args.netsize == 'b':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant
    else:
        model_load = dino_variant._large_dino
        variant = dino_variant._large_variant

    model = rein.ReinsDinoVisionTransformer(**variant)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])

    # -------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------
    ckpt = torch.load(args.ckpt, map_location='cpu')

    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt  

    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.eval()

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------
    with torch.no_grad():
        if args.adapter in ['rein', 'adaptformer']:
            acc = validation_accuracy(
                model,
                test_loader,
                device,
                mode=args.adapter
            )
        else:
            acc = validation_accuracy_lora(
                model,
                test_loader,
                device
            )

    print("=" * 60)
    print(f"Dataset     : {args.data}")
    print(f"Adapter     : {args.adapter}")
    print(f"Net size    : {args.netsize}")
    print(f"Checkpoint  : {args.ckpt}")
    print(f"Test Acc    : {acc:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()