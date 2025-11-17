import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from tqdm import tqdm
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.model import build_model


def seed_all(seed: int = 42):
    import random
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_transforms(img_size, augment=False):
    """Get transforms for training and validation"""
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * images.size(0)
            prob = torch.softmax(logits, dim=1)[:, 1]
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(prob.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Calculate AUC only if we have both classes
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.0
    
    return avg_loss, accuracy, auc, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description='Train bone cancer classifier on folder-based dataset')
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--img-size", type=int, default=384, help="Image size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3_small"])
    parser.add_argument("--dataset-root", type=str, default=r"T:\bone_can_pre\dataset\dataset",
                       help="Root directory containing train/valid/test folders")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loading workers")
    args = parser.parse_args()

    seed_all(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"{'='*60}\n")

    # Create datasets
    print("📂 Loading datasets...")
    train_transform = get_transforms(args.img_size, augment=True)
    val_transform = get_transforms(args.img_size, augment=False)
    
    train_dataset = datasets.ImageFolder(
        os.path.join(args.dataset_root, 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        os.path.join(args.dataset_root, 'valid'),
        transform=val_transform
    )
    
    test_dataset = datasets.ImageFolder(
        os.path.join(args.dataset_root, 'test'),
        transform=val_transform
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    print(f"✓ Classes: {train_dataset.classes}\n")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Build model
    print(f"🏗️  Building {args.model} model...")
    model = build_model(args.model, num_classes=2, pretrained=args.pretrained)
    model.to(device)
    print(f"✓ Model initialized\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    best_auc = 0.0
    best_acc = 0.0
    os.makedirs(Config.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(Config.ckpt_dir, f"{args.model}_folder_best.pt")
    
    print(f"🚀 Starting training...\n")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_auc, val_labels, val_preds = eval_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch [{epoch}/{args.epochs}] ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "auc": val_auc,
                "accuracy": val_acc,
                "model_name": args.model
            }, ckpt_path)
            print(f"  ✓ New best! Saved to {ckpt_path}")
        print()
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Completed!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Best validation accuracy: {best_acc:.4f}")
    
    # Test evaluation
    print(f"\n🧪 Evaluating on test set...")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True)['model'])
    test_loss, test_acc, test_auc, test_labels, test_preds = eval_epoch(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=train_dataset.classes))
    
    print(f"\n{'='*60}")
    print(f"✓ Model saved to: {ckpt_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
