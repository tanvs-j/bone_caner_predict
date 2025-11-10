import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import Config
from src.model import build_model
from src.survival_model import SurvivalPredictor, ClinicalFeatureEncoder


class SurvivalDataset(Dataset):
    """Dataset combining images and clinical data for survival prediction"""
    
    def __init__(self, clinical_df, image_labels_csv, image_root, img_size=384, augment=False):
        self.clinical_df = clinical_df.copy()
        self.encoder = ClinicalFeatureEncoder()
        self.img_size = img_size
        
        # Load image labels
        img_df = pd.read_csv(image_labels_csv)
        img_df.columns = [c.strip().lower() for c in img_df.columns]
        
        # For this dataset, we'll randomly assign patients to images
        # In a real scenario, you'd have patient IDs matched to images
        self.image_files = [f for f in os.listdir(image_root) 
                           if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Create mapping (in real scenario, this would be from a mapping file)
        np.random.seed(42)
        num_samples = min(len(self.clinical_df), len(self.image_files))
        self.clinical_df = self.clinical_df.iloc[:num_samples].reset_index(drop=True)
        self.image_files = np.random.choice(self.image_files, num_samples, replace=False)
        
        self.image_root = image_root
        self.transform = self._build_transforms(img_size, augment)
        
    def _build_transforms(self, size, augment):
        train_tf = [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ]
        if augment:
            aug = [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.2, rotate_limit=10, 
                                  border_mode=cv2.BORDER_CONSTANT, p=0.5),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            ]
            return A.Compose(aug + train_tf)
        return A.Compose(train_tf)
    
    def __len__(self):
        return len(self.clinical_df)
    
    def __getitem__(self, idx):
        # Get clinical data
        row = self.clinical_df.iloc[idx].to_dict()
        clinical_features = self.encoder.encode(row)
        
        # Get survival status
        status = row['Status (NED, AWD, D)']
        survival_label = self.encoder.encode_status(status)
        
        # Cancer label (1 if status is AWD or D, 0 if NED)
        cancer_label = 0 if status == 'NED' else 1
        
        # Load and transform image
        img_path = os.path.join(self.image_root, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_tensor = self.transform(image=img)['image']
        
        return {
            'image': img_tensor,
            'clinical': torch.from_numpy(clinical_features),
            'cancer_label': cancer_label,
            'survival_label': survival_label,
            'age': row['Age'],
            'grade': row['Grade']
        }


def train_epoch(model, loader, criterions, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch['image'].to(device)
        clinical = batch['clinical'].to(device)
        cancer_labels = batch['cancer_label'].to(device)
        survival_labels = batch['survival_label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images, clinical)
        
        # Multi-task losses
        cancer_loss = criterions['cancer'](outputs['cancer_logits'], cancer_labels)
        survival_loss = criterions['survival'](outputs['survival_logits'], survival_labels)
        
        # Combined loss
        loss = cancer_loss + 2.0 * survival_loss  # Weight survival task more
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
    
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterions, device):
    model.eval()
    total_loss = 0.0
    cancer_preds, cancer_labels_all = [], []
    survival_preds, survival_labels_all = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            images = batch['image'].to(device)
            clinical = batch['clinical'].to(device)
            cancer_labels = batch['cancer_label'].to(device)
            survival_labels = batch['survival_label'].to(device)
            
            outputs = model(images, clinical)
            
            cancer_loss = criterions['cancer'](outputs['cancer_logits'], cancer_labels)
            survival_loss = criterions['survival'](outputs['survival_logits'], survival_labels)
            loss = cancer_loss + 2.0 * survival_loss
            
            total_loss += loss.item() * images.size(0)
            
            cancer_preds.extend(outputs['cancer_logits'].argmax(dim=1).cpu().numpy())
            cancer_labels_all.extend(cancer_labels.cpu().numpy())
            
            survival_preds.extend(outputs['survival_logits'].argmax(dim=1).cpu().numpy())
            survival_labels_all.extend(survival_labels.cpu().numpy())
    
    cancer_acc = accuracy_score(cancer_labels_all, cancer_preds)
    survival_acc = accuracy_score(survival_labels_all, survival_preds)
    survival_f1 = f1_score(survival_labels_all, survival_preds, average='macro')
    
    return (total_loss / len(loader.dataset), cancer_acc, survival_acc, survival_f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clinical-data", type=str, 
                       default=r"T:\bone_can_pre\dataset\train\Bone Tumor Dataset.csv")
    parser.add_argument("--image-labels", type=str,
                       default=r"T:\bone_can_pre\dataset\train\_classes.csv")
    parser.add_argument("--image-root", type=str,
                       default=r"T:\bone_can_pre\dataset\train")
    parser.add_argument("--pretrained-ckpt", type=str,
                       default=r"T:\bone_can_pre\models\efficientnet_b0_best.pt")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load clinical data
    print("Loading clinical data...")
    clinical_df = pd.read_csv(args.clinical_data)
    
    # Split into train/val
    train_df, val_df = train_test_split(clinical_df, test_size=0.15, random_state=42)
    
    # Create datasets
    print("Creating datasets...")
    train_ds = SurvivalDataset(train_df, args.image_labels, args.image_root, 
                               img_size=args.img_size, augment=True)
    val_ds = SurvivalDataset(val_df, args.image_labels, args.image_root,
                            img_size=args.img_size, augment=False)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # Build model
    print("Building survival model...")
    # Load pretrained cancer classifier
    base_model = build_model("efficientnet_b0", num_classes=2, pretrained=False)
    if os.path.exists(args.pretrained_ckpt):
        print(f"Loading pretrained weights from {args.pretrained_ckpt}")
        state = torch.load(args.pretrained_ckpt, map_location=device)
        base_model.load_state_dict(state['model'])
    
    # Create feature extractor (remove final classification layer)
    feature_extractor = nn.Sequential(*list(base_model.children())[:-1], nn.AdaptiveAvgPool2d(1), nn.Flatten())
    
    # Create survival predictor
    encoder = ClinicalFeatureEncoder()
    model = SurvivalPredictor(feature_extractor, num_clinical_features=encoder.feature_dim)
    model.to(device)
    
    # Loss functions
    criterions = {
        'cancer': nn.CrossEntropyLoss(),
        'survival': nn.CrossEntropyLoss()
    }
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_survival_f1 = 0.0
    os.makedirs(Config.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(Config.ckpt_dir, "survival_model_best.pt")
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_dl, criterions, optimizer, device)
        va_loss, ca_acc, sa_acc, sa_f1 = eval_epoch(model, val_dl, criterions, device)
        scheduler.step()
        
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"cancer_acc={ca_acc:.4f} survival_acc={sa_acc:.4f} survival_f1={sa_f1:.4f}")
        
        if sa_f1 > best_survival_f1:
            best_survival_f1 = sa_f1
            torch.save({
                'model': model.state_dict(),
                'survival_f1': sa_f1,
                'epoch': epoch,
                'encoder': encoder
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
    
    print(f"\nTraining completed! Best survival F1: {best_survival_f1:.4f}")
    print(f"Model saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
