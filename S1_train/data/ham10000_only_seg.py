import os
import random
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class HAM10000SegmentationDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        mask_dir,
        image_transform=None,
        mask_transform=None,
        use_mask=True,   # train=True / val=False 가능
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

        # ---------------------------
        # RGB Image
        # ---------------------------
        img_path = os.path.join(self.image_dir, img_id + '.jpg')
        image = Image.open(img_path).convert('RGB')

        # ---------------------------
        # Segmentation Mask
        # ---------------------------
        if self.use_mask:
            mask_path = os.path.join(
                self.mask_dir, img_id + '_segmentation.png'
            )
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
            else:
                # mask 없는 경우 → zero mask
                w, h = image.size
                mask = Image.new('L', (w, h), 0)
        else:
            w, h = image.size
            mask = Image.new('L', (w, h), 0)

        # ---------------------------
        # 동일한 spatial transform
        # ---------------------------
        if self.image_transform is not None:
            seed = random.randint(0, 2**32)

            random.seed(seed)
            torch.manual_seed(seed)
            image = self.image_transform(image)

            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)

        # ---------------------------
        # SAM2-Adapter 학습용 반환
        # ---------------------------
        return {
            'inp': image,                 # [3, H, W]
            'gt':  (mask > 0).float()     # [1, H, W], binary
        }

def get_transforms(train=True):
    if train:
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor()  # [0,1]
        ])

    else:
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor()
        ])

    return image_transform, mask_transform

def get_segmentation_dataloaders(
    data_dir,
    batch_size=16,
    num_workers=4,
    split_ratio=0.9,
    random_seed=42
):
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    csv_file = os.path.join(data_dir, 'ISIC2018_Task3_Training_GroundTruth.csv')
    image_dir = os.path.join(data_dir, 'train')
    mask_dir  = os.path.join(data_dir, 'ham10000_mask')

    train_img_tf, train_mask_tf = get_transforms(train=True)
    val_img_tf,   val_mask_tf   = get_transforms(train=False)

    full_dataset = HAM10000SegmentationDataset(
        csv_file=csv_file,
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_transform=None,
        mask_transform=None,
        use_mask=True
    )

    train_len = int(len(full_dataset) * split_ratio)
    val_len   = len(full_dataset) - train_len

    train_set, val_set = random_split(
        full_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(random_seed)
    )

    train_set.dataset.image_transform = train_img_tf
    train_set.dataset.mask_transform  = train_mask_tf

    val_set.dataset.image_transform = val_img_tf
    val_set.dataset.mask_transform  = val_mask_tf

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


