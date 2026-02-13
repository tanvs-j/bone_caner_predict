#!/usr/bin/env python3
"""Verify that the classification fix correctly labels cancer vs normal images."""

import os, sys, glob
import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import build_model

# ── Config ──────────────────────────────────────────────────────────────
DATASET_ROOT = r"T:\bone_can_pre\dataset\dataset\test"
CKPT = os.environ.get("BONE_CKPT", r"T:\bone_can_pre\models\efficientnet_b0_best.pt")
MODEL_NAME = "efficientnet_b0" if "efficientnet" in os.path.basename(CKPT) else "mobilenet_v3_small"
NUM_SAMPLES = 5  # per class

# ── Setup ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(MODEL_NAME, num_classes=2, pretrained=False).to(device)
ckpt = torch.load(CKPT, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval()

transform = A.Compose([
    A.LongestMaxSize(max_size=224),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    x = transform(image=img_bgr)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0, 0].item()  # index 0 = cancer
    return prob

# ── Test cancer images ──────────────────────────────────────────────────
cancer_dir = os.path.join(DATASET_ROOT, "cancer")
normal_dir = os.path.join(DATASET_ROOT, "normal")

cancer_imgs = glob.glob(os.path.join(cancer_dir, "*.jpg"))[:NUM_SAMPLES]
normal_imgs = glob.glob(os.path.join(normal_dir, "*.jpg"))[:NUM_SAMPLES]

print(f"Model: {MODEL_NAME}  |  Checkpoint: {os.path.basename(CKPT)}")
print(f"Device: {device}\n")

passed = 0
total = 0

print("=== CANCER images (expected prob > 0.5) ===")
for p in cancer_imgs:
    prob = predict(p)
    ok = prob > 0.5
    print(f"  {'✅' if ok else '❌'} {os.path.basename(p)[:40]:<42} cancer_prob={prob:.4f}")
    passed += int(ok)
    total += 1

print(f"\n=== NORMAL images (expected prob < 0.5) ===")
for p in normal_imgs:
    prob = predict(p)
    ok = prob < 0.5
    print(f"  {'✅' if ok else '❌'} {os.path.basename(p)[:40]:<42} cancer_prob={prob:.4f}")
    passed += int(ok)
    total += 1

print(f"\n{'='*60}")
print(f"Result: {passed}/{total} correct  ({'PASS' if passed == total else 'PARTIAL'})")
