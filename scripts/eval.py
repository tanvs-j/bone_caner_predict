import os
import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader, Subset
import numpy as np

from src.config import Config
from src.data import BoneCancerDataset
from src.model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default=Config.dataset_root)
    parser.add_argument("--labels-csv", type=str, default=Config.labels_csv)
    parser.add_argument("--img-size", type=int, default=Config.img_size)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        ds = BoneCancerDataset(args.dataset_root, args.split, args.labels_csv, img_size=args.img_size, augment=False)
    except Exception as e:
        print(f"[info] split '{args.split}' not found ({e}); evaluating on a random subset of train")
        train_ds = BoneCancerDataset(args.dataset_root, "train", args.labels_csv, img_size=args.img_size, augment=False)
        n = len(train_ds)
        k = min(max(1, n // 10), 2000)
        idx = np.random.permutation(n)[:k]
        ds = Subset(train_ds, idx)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    base = os.path.splitext(os.path.basename(args.ckpt))[0]
    if base.startswith("mobilenet_v3_small"):
        model_name = "mobilenet_v3_small"
    elif base.startswith("efficientnet_b0"):
        model_name = "efficientnet_b0"
    else:
        model_name = base
    model = build_model(model_name, num_classes=2, pretrained=False).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    ys, ps, preds = [], [], []
    with torch.no_grad():
        for x, y, _ in dl:
            x = x.to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            ps.append(prob)
            ys.append(y.numpy())
            preds.append((prob > 0.5).astype(np.int64))

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    y_pred = np.concatenate(preds)

    print("AUC:", roc_auc_score(y_true, y_prob))
    print(classification_report(y_true, y_pred, target_names=["normal","cancer"]))

if __name__ == "__main__":
    main()
