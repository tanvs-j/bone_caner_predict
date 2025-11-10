import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

from src.config import Config
from src.data import BoneCancerDataset
from src.model import build_model


def seed_all(seed: int = 42):
    import random
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    ys, ps = [], []
    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="eval", leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            prob = torch.softmax(logits, dim=1)[:, 1]
            ys.append(y.cpu().numpy())
            ps.append(prob.cpu().numpy())
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    auc = roc_auc_score(ys, ps)
    return total_loss / len(loader.dataset), auc, ys, ps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--img-size", type=int, default=Config.img_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--model", type=str, default=Config.model_name)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--dataset-root", type=str, default=Config.dataset_root)
    parser.add_argument("--labels-csv", type=str, default=Config.labels_csv)
    args = parser.parse_args()

    seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = os.path.join(args.dataset_root, "train", "_classes.csv")
    valid_csv = os.path.join(args.dataset_root, "valid", "_classes.csv")
    
    train_ds = BoneCancerDataset(args.dataset_root, "train", train_csv, img_size=args.img_size, augment=True)
    try:
        valid_ds = BoneCancerDataset(args.dataset_root, "valid", valid_csv, img_size=args.img_size, augment=False)
        if len(valid_ds) == 0:
            raise RuntimeError("empty valid dataset")
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=Config.num_workers, pin_memory=True)
        valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=Config.num_workers, pin_memory=True)
    except Exception as e:
        # Fallback: create a random 90/10 split from the train set
        val_size = max(1, int(0.1 * len(train_ds)))
        tr_size = len(train_ds) - val_size
        train_subset, valid_subset = random_split(train_ds, [tr_size, val_size])
        print(f"[info] valid split not found ({e}); using random split {tr_size}/{val_size} from train")
        train_dl = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=Config.num_workers, pin_memory=True)
        valid_dl = DataLoader(valid_subset, batch_size=args.batch_size, shuffle=False, num_workers=Config.num_workers, pin_memory=True)

    model = build_model(args.model, num_classes=2, pretrained=(not args.no_pretrained))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = 0.0
    os.makedirs(Config.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(Config.ckpt_dir, f"{args.model}_best.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_auc, ys, ps = eval_epoch(model, valid_dl, criterion, device)
        scheduler.step()
        print(f"epoch {epoch}: train_loss={tr_loss:.4f} valid_loss={va_loss:.4f} valid_auc={va_auc:.4f}")
        if va_auc > best_auc:
            best_auc = va_auc
            torch.save({"model": model.state_dict(), "auc": va_auc, "epoch": epoch}, ckpt_path)
            print(f"saved checkpoint to {ckpt_path}")

    print(f"best valid AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()
