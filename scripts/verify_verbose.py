#!/usr/bin/env python3
"""Verbose verification of the classification fix."""
import os, sys, glob
import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import build_model

device = torch.device("cpu")
model = build_model("efficientnet_b0", num_classes=2, pretrained=False).to(device)
ckpt = torch.load(r"T:\bone_can_pre\models\efficientnet_b0_best.pt", map_location=device, weights_only=True)
model.load_state_dict(ckpt["model"])
model.eval()

t = A.Compose([
    A.LongestMaxSize(224),
    A.PadIfNeeded(224, 224, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

def pred(p):
    img = Image.open(p).convert("RGB")
    x = t(image=cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))["image"].unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    return probs[0].item(), probs[1].item()  # cancer_prob, normal_prob

print("--- CANCER images ---")
for p in sorted(glob.glob(r"T:\bone_can_pre\dataset\dataset\test\cancer\*.jpg"))[:5]:
    cp, np_ = pred(p)
    ok = cp > 0.5
    print(f"  {'OK' if ok else 'FAIL'} cancer_p={cp:.4f} normal_p={np_:.4f} {os.path.basename(p)[:35]}")

print("\n--- NORMAL images ---")
for p in sorted(glob.glob(r"T:\bone_can_pre\dataset\dataset\test\normal\*.jpg"))[:5]:
    cp, np_ = pred(p)
    ok = cp < 0.5
    print(f"  {'OK' if ok else 'FAIL'} cancer_p={cp:.4f} normal_p={np_:.4f} {os.path.basename(p)[:35]}")
