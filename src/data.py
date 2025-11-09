import os
import cv2
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

SPLIT_DIRS = {"train", "valid", "test"}

class BoneCancerDataset(Dataset):
    def __init__(self, root: str, split: str, labels_csv: str, img_size: int = 384, augment: bool = False):
        assert split in SPLIT_DIRS, f"split must be one of {SPLIT_DIRS}"
        self.root = root
        self.split = split
        self.dir = os.path.join(root, split)
        self.df = pd.read_csv(labels_csv)
        # normalize column names
        self.df.columns = [c.strip().lower() for c in self.df.columns]
        # expected columns: filename, cancer, normal
        for col in ["filename", "cancer", "normal"]:
            if col not in self.df.columns:
                raise ValueError(f"Missing column '{col}' in labels csv: {labels_csv}")
        # keep only rows where file exists in this split
        paths = []
        labels = []
        for _, r in self.df.iterrows():
            fname = str(r["filename"]).strip()
            # files are unique by filename across splits
            p = os.path.join(self.dir, fname)
            if os.path.exists(p):
                paths.append(p)
                # binary label: 1=cancer, 0=normal
                y = int(r["cancer"]) if pd.notnull(r["cancer"]) else 0
                labels.append(y)
        self.paths = paths
        self.labels = labels
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {self.dir} matching entries in {labels_csv}")
        self.transform = self._build_transforms(img_size, augment)

    def _build_transforms(self, size: int, augment: bool):
        train_tf = [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ]
        if augment:
            aug = [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            ]
            return A.Compose(aug + train_tf)
        return A.Compose(train_tf)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", int, str]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        transformed = self.transform(image=img)
        x = transformed["image"]
        y = int(self.labels[idx])
        return x, y, path
