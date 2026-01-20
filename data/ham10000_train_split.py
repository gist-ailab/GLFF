import os
import pandas as pd
import random
import torch
import time
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image


# -------------------------
# Dataset
# -------------------------
class HAM10000ClsDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        transform=None
    ):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        self.class_mapping = {
            'MEL': 0, 'NV': 1, 'BCC': 2,
            'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6
        }

        self.data['encoded_labels'] = (
            self.data.iloc[:, 1:]
            .idxmax(axis=1)
            .map(self.class_mapping)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_id + '.jpg')
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['encoded_labels']

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_cls_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# -------------------------
# Dataloader (train / test only)
# -------------------------
def get_cls_dataloaders_train_only(
    data_dir,
    batch_size=32,
    num_workers=4,
    split_ratio=0.9,
    random_seed=42
):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # paths
    train_csv = os.path.join(data_dir, 'ISIC2018_Task3_Training_GroundTruth.csv')
    image_dir = os.path.join(data_dir, 'train')

    train_tf = get_cls_transforms(train=True)
    val_tf   = get_cls_transforms(train=False)

    # full dataset
    full_dataset = HAM10000ClsDataset(
        csv_file=train_csv,
        image_dir=image_dir,
        transform=None  # split 후 주입
    )

    total_len = len(full_dataset)
    train_len = int(total_len * split_ratio)
    val_len   = total_len - train_len

    train_set, val_set = random_split(
        full_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(random_seed)
    )

    # transform 주입
    train_set.dataset.transform = train_tf
    val_set.dataset.transform   = val_tf

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

    print(f"CLS Train samples: {len(train_set)}")
    print(f"CLS Val   samples: {len(val_set)}")

    return train_loader, val_loader

# -------------------------
# Debug
# -------------------------
if __name__ == "__main__":
    data_dir = "/path/to/ham10000"

    train_loader, test_loader = get_cls_dataloaders_train_only(
        data_dir=data_dir,
        batch_size=16,
        num_workers=4,
        random_seed=42
    )

    images, labels = next(iter(train_loader))
    print(images.shape)  # [B, 3, 224, 224]
    print(labels.shape)  # [B]
