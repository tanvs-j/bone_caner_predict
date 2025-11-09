import os
import io
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.model import build_model

MODEL_PATH = os.environ.get("BONE_CKPT", r"T:\\bone_can_pre\\models\\mobilenet_v3_small_best.pt")
base = os.path.splitext(os.path.basename(MODEL_PATH))[0]
if "mobilenet_v3_small" in base:
    MODEL_NAME = "mobilenet_v3_small"
elif "efficientnet_b0" in base:
    MODEL_NAME = "efficientnet_b0"
else:
    MODEL_NAME = "mobilenet_v3_small"

app = FastAPI(title="Bone Cancer Classifier", version="1.0")

size = 256
transform = A.Compose([
    A.LongestMaxSize(max_size=size),
    A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(MODEL_NAME, num_classes=2, pretrained=False).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state["model"])
model.eval()

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Bone Cancer Classifier</title>
  <style> body{font-family:Arial;margin:2rem} .card{max-width:720px} </style>
</head>
<body>
  <div class='card'>
    <h2>Bone Cancer Classifier</h2>
    <p>Upload an image. The model returns Normal vs Cancer and a probability.</p>
    <input type="file" id="file" accept="image/*" />
    <button onclick="predict()">Predict</button>
    <pre id="out"></pre>
    <h3>Lifespan</h3>
    <p>Unavailable in this model (needs survival labels). I can add this once you provide survival data.</p>
  </div>
<script>
async function predict(){
  const f = document.getElementById('file').files[0];
  if(!f){ alert('Choose an image first'); return; }
  const fd = new FormData(); fd.append('file', f, f.name);
  const res = await fetch('/predict', { method:'POST', body: fd });
  const j = await res.json();
  document.getElementById('out').textContent = JSON.stringify(j, null, 2);
}
</script>
</body>
</html>
"""

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    x = transform(image=img)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0,1].item()
    return {"cancer_probability": prob, "prediction": "cancer" if prob >= 0.5 else "normal", "lifespan": "Unavailable (needs survival model)"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
