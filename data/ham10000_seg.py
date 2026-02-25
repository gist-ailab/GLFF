import os
import random
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

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

def get_dataloaders_with_mask(
    data_dir,
    batch_size=32,
    num_workers=4,
    random_seed=None
):
    if random_seed is None:
        random_seed = int(time.time()) % (2**32)

    print(f"Using random seed: {random_seed}")

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # CSV paths
    train_csv = os.path.join(data_dir, 'ISIC2018_Task3_Training_GroundTruth.csv')
    valid_csv = os.path.join(data_dir, 'ISIC2018_Task3_Validation_GroundTruth.csv')
    test_csv  = os.path.join(data_dir, 'ISIC2018_Task3_Test_GroundTruth.csv')

    # Image directories
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir  = os.path.join(data_dir, 'test')

    # Mask directory (train only 의미 있음)
    mask_dir = os.path.join(data_dir, 'ham10000_mask')

    # Transforms
    train_img_tf, train_mask_tf = get_transforms(train=True)
    test_img_tf, test_mask_tf = get_transforms(train=False)

    # ---------------------------
    # Datasets
    # ---------------------------
    # 1. 마스크 경로 설정 변경
    train_mask_dir = os.path.join(data_dir, 'ham10000_mask') # GT 마스크
    # SAM으로 생성한 마스크 경로
    sam_mask_base = os.path.join(data_dir, 'inferred_masks') 
    
    # 2. Datasets 설정 수정
    train_dataset = HAM10000WithMaskDataset(
        train_csv, train_dir, train_mask_dir,
        image_transform=train_img_tf, mask_transform=train_mask_tf,
        use_mask=True # Train은 GT 마스크 사용
    )

    valid_dataset = HAM10000WithMaskDataset(
        valid_csv, valid_dir, 
        mask_dir=os.path.join(sam_mask_base, 'valid_masks'), # SAM 마스크 경로로 변경
        image_transform=test_img_tf, mask_transform=test_mask_tf,
        use_mask=True # ⭐ False에서 True로 변경하여 마스크 로드 활성화
    )

    test_dataset = HAM10000WithMaskDataset(
        test_csv, test_dir, 
        mask_dir=os.path.join(sam_mask_base, 'test_masks'), # SAM 마스크 경로로 변경
        image_transform=test_img_tf, mask_transform=test_mask_tf,
        use_mask=True # ⭐ False에서 True로 변경
    )
    # ---------------------------
    # DataLoaders
    # ---------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)} (GT mask used)")
    print(f"Valid samples: {len(valid_dataset)} (SAM masks used)")
    print(f"Test  samples: {len(test_dataset)} (SAM masks used)")

    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    data_dir = "/path/to/ham10000"

    train_loader, val_loader, test_loader = get_dataloaders_with_mask(
        data_dir=data_dir,
        batch_size=16,
        num_workers=4
    )

    images, masks, labels = next(iter(train_loader))
    print(images.shape)  # [B, 3, 224, 224]
    print(masks.shape)   # [B, 1, 224, 224]
    print(labels.shape)  # [B]
