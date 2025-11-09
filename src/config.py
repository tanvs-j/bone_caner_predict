import os
from dataclasses import dataclass

@dataclass
class Config:
    dataset_root: str = r"T:\\bone_can_pre\\dataset"
    train_dir: str = "train"
    valid_dir: str = "valid"
    test_dir: str = "test"
    labels_csv: str = os.path.join(dataset_root, "train", "_classes.csv")
    img_size: int = 384
    batch_size: int = 16
    epochs: int = 10
    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 0  # Windows-friendly; bump to 4+ for speed if stable
    model_name: str = "efficientnet_b0"
    ckpt_dir: str = r"T:\\bone_can_pre\\models"
