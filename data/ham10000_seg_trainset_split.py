import os
import random
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
if __package__:
    from .ham10000_seg import get_transforms
else:
    from ham10000_seg import get_transforms

class HAM10000WithMaskDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        mask_dir,
        image_transform=None,
        mask_transform=None,
        use_mask=True,          # mask 
    ):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.use_mask = use_mask

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

        # ---------------------------
        # RGB Image
        # ---------------------------
        img_path = os.path.join(self.image_dir, img_id + '.jpg')
        image = Image.open(img_path).convert('RGB')

        # ---------------------------
        # Segmentation Mask
        # ---------------------------
        if self.use_mask:
            mask_name = img_id + '_segmentation.png'
            mask_path = os.path.join(self.mask_dir, mask_name)

            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
            else:
                # train인데 mask 없는 경우 → zero-mask
                w, h = image.size
                mask = Image.new('L', (w, h), 0)
        else:
            # ⭐ Val/Test: 항상 zero-mask
            w, h = image.size
            mask = Image.new('L', (w, h), 0)

        label = self.data.iloc[idx]['encoded_labels']

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

        return image, mask, label



def get_dataloaders_train_only(
    data_dir,
    batch_size=32,
    num_workers=4,
    split_ratio=0.9,
    random_seed=42
):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # CSV (train only)
    train_csv = os.path.join(data_dir, 'ISIC2018_Task3_Training_GroundTruth.csv')
    image_dir = os.path.join(data_dir, 'train')
    mask_dir  = os.path.join(data_dir, 'ham10000_mask')

    # Transforms
    train_img_tf, train_mask_tf = get_transforms(train=True)
    test_img_tf, test_mask_tf   = get_transforms(train=False)

    # Full dataset (mask 사용)
    full_dataset = HAM10000WithMaskDataset(
        csv_file=train_csv,
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_transform=None,     # split 후 적용
        mask_transform=None,
        use_mask=True             # ⭐ teacher이므로 항상 mask
    )

    # ---------------------------
    # 9:1 split
    # ---------------------------
    total_len = len(full_dataset)
    train_len = int(total_len * split_ratio)
    test_len  = total_len - train_len

    train_set, test_set = random_split(
        full_dataset,
        [train_len, test_len],
        generator=torch.Generator().manual_seed(random_seed)
    )

    # split 이후 transform 주입
    train_set.dataset.image_transform = train_img_tf
    train_set.dataset.mask_transform  = train_mask_tf

    test_set.dataset.image_transform  = test_img_tf
    test_set.dataset.mask_transform   = test_mask_tf

    # ---------------------------
    # DataLoaders
    # ---------------------------
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Teacher Train samples: {len(train_set)} (mask ON)")
    print(f"Teacher Test  samples: {len(test_set)}  (mask ON)")

    return train_loader, test_loader


if __name__ == "__main__":
    data_dir = "/path/to/ham10000"

    train_loader, test_loader = get_teacher_dataloaders_train_only(
        data_dir=data_dir,
        batch_size=16,
        num_workers=4
    )

    images, masks, labels = next(iter(train_loader))
    print(images.shape)  # [B, 3, 224, 224]
    print(masks.shape)   # [B, 1, 224, 224]
    print(labels.shape)  # [B]