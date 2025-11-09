import os
import random
import pandas as pd
from fastapi.testclient import TestClient
from app.server import app

CSV = r"T:\\bone_can_pre\\dataset\\train\\_classes.csv"
TRAIN_DIR = r"T:\\bone_can_pre\\dataset\\train"

def pick_example(df, label_col, label_val):
    for _, r in df.iterrows():
        if int(r[label_col]) == label_val:
            p = os.path.join(TRAIN_DIR, str(r['filename']).strip())
            if os.path.exists(p):
                return p
    return None

if __name__ == "__main__":
    df = pd.read_csv(CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    client = TestClient(app)

    cancer_img = pick_example(df, 'cancer', 1)
    normal_img = pick_example(df, 'cancer', 0)

    for label, img_path in [("cancer", cancer_img), ("normal", normal_img)]:
        if not img_path:
            print(f"No example found for {label}")
            continue
        with open(img_path, 'rb') as f:
            files = {"file": (os.path.basename(img_path), f, "image/jpeg")}
            r = client.post("/predict", files=files)
            print(label, os.path.basename(img_path), r.status_code, r.json())
