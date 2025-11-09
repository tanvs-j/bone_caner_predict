import os
from fastapi.testclient import TestClient
from app.server import app

if __name__ == "__main__":
    client = TestClient(app)
    img_path = r"T:\\bone_can_pre\\dataset\\train\\-1-_jpg.rf.53eac29a51d6094f6d5dcdff51f77017.jpg"
    with open(img_path, "rb") as f:
        files = {"file": (os.path.basename(img_path), f, "image/jpeg")}
        r = client.post("/predict", files=files)
        print(r.status_code)
        print(r.json())
