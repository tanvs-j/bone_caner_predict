import os
import io
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn

from src.model import build_model
from src.survival_model import SurvivalPredictor, ClinicalFeatureEncoder, estimate_survival_months

# Model paths
SURVIVAL_MODEL_PATH = os.environ.get("SURVIVAL_CKPT", r"T:\bone_can_pre\models\survival_model_best.pt")
CANCER_MODEL_PATH = os.environ.get("BONE_CKPT", r"T:\bone_can_pre\models\efficientnet_b0_best.pt")

app = FastAPI(title="Bone Cancer Survival Predictor", version="2.0")

size = 384
transform = A.Compose([
    A.LongestMaxSize(max_size=size),
    A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = ClinicalFeatureEncoder()

# Load survival model if available
survival_model = None
status_names = ['NED (No Evidence of Disease)', 'AWD (Alive with Disease)', 'Dead']

if os.path.exists(SURVIVAL_MODEL_PATH):
    print(f"Loading survival model from {SURVIVAL_MODEL_PATH}")
    base_model = build_model("efficientnet_b0", num_classes=2, pretrained=False)
    feature_extractor = nn.Sequential(*list(base_model.children())[:-1], nn.AdaptiveAvgPool2d(1), nn.Flatten())
    survival_model = SurvivalPredictor(feature_extractor, num_clinical_features=encoder.feature_dim)
    state = torch.load(SURVIVAL_MODEL_PATH, map_location=device, weights_only=False)
    survival_model.load_state_dict(state['model'])
    survival_model.to(device)
    survival_model.eval()
    print("Survival model loaded successfully!")
else:
    print("Survival model not found, using cancer-only model")
    # Fallback to cancer-only model
    cancer_model = build_model("efficientnet_b0", num_classes=2, pretrained=False)
    state = torch.load(CANCER_MODEL_PATH, map_location=device)
    cancer_model.load_state_dict(state['model'])
    cancer_model.to(device)
    cancer_model.eval()


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Bone Cancer Survival Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; background: #f5f5f5; }
    .container { max-width: 900px; margin: 0 auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    h1 { color: #2c3e50; }
    .form-group { margin-bottom: 1rem; }
    label { display: block; margin-bottom: 0.5rem; font-weight: bold; color: #34495e; }
    input, select { width: 100%; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px; }
    button { background: #3498db; color: white; padding: 0.75rem 2rem; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
    button:hover { background: #2980b9; }
    #results { margin-top: 2rem; padding: 1rem; background: #ecf0f1; border-radius: 4px; display: none; }
    .result-item { margin: 0.5rem 0; }
    .prediction { font-size: 1.2em; font-weight: bold; color: #27ae60; }
    .cancer { color: #e74c3c; }
  </style>
</head>
<body>
  <div class='container'>
    <h1>🏥 Bone Cancer Survival Predictor</h1>
    <p>Upload an X-ray image and provide clinical information for survival prediction.</p>
    
    <form id="predictionForm">
      <div class="form-group">
        <label>X-ray Image:</label>
        <input type="file" id="image" accept="image/*" required />
      </div>
      
      <div class="form-group">
        <label>Sex:</label>
        <select id="sex">
          <option value="Male">Male</option>
          <option value="Female">Female</option>
        </select>
      </div>
      
      <div class="form-group">
        <label>Age:</label>
        <input type="number" id="age" value="50" min="1" max="120" required />
      </div>
      
      <div class="form-group">
        <label>Tumor Grade:</label>
        <select id="grade">
          <option value="Low">Low</option>
          <option value="Intermediate" selected>Intermediate</option>
          <option value="High">High</option>
        </select>
      </div>
      
      <div class="form-group">
        <label>Treatment (select applicable):</label>
        <div>
          <input type="checkbox" id="surgery" value="Surgery"> Surgery<br>
          <input type="checkbox" id="chemo" value="Chemotherapy"> Chemotherapy<br>
          <input type="checkbox" id="radio" value="Radiotherapy"> Radiotherapy
        </div>
      </div>
      
      <div class="form-group">
        <label>Histological Type:</label>
        <select id="histology">
          <option value="Osteosarcoma">Osteosarcoma</option>
          <option value="Leiomyosarcoma">Leiomyosarcoma</option>
          <option value="Liposarcoma">Liposarcoma</option>
          <option value="Other">Other</option>
        </select>
      </div>
      
      <button type="submit">Predict Survival</button>
    </form>
    
    <div id="results">
      <h2>Prediction Results</h2>
      <div id="output"></div>
    </div>
  </div>

<script>
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
  e.preventDefault();
  
  const file = document.getElementById('image').files[0];
  if (!file) {
    alert('Please select an image');
    return;
  }
  
  const treatments = [];
  if (document.getElementById('surgery').checked) treatments.push('Surgery');
  if (document.getElementById('chemo').checked) treatments.push('Chemotherapy');
  if (document.getElementById('radio').checked) treatments.push('Radiotherapy');
  
  const formData = new FormData();
  formData.append('file', file);
  formData.append('sex', document.getElementById('sex').value);
  formData.append('age', document.getElementById('age').value);
  formData.append('grade', document.getElementById('grade').value);
  formData.append('treatment', treatments.join(' + '));
  formData.append('histological_type', document.getElementById('histology').value);
  
  const response = await fetch('/predict_survival', { method: 'POST', body: formData });
  const data = await response.json();
  
  const resultsDiv = document.getElementById('results');
  const outputDiv = document.getElementById('output');
  
  let html = `
    <div class="result-item prediction ${data.cancer_prediction === 'cancer' ? 'cancer' : ''}">
      Cancer Prediction: ${data.cancer_prediction.toUpperCase()} (${(data.cancer_probability * 100).toFixed(1)}%)
    </div>
    <div class="result-item">
      <strong>Survival Status:</strong> ${data.survival_status}
    </div>
    <div class="result-item">
      <strong>Estimated Survival:</strong> ${data.estimated_survival.estimated_years} years 
      (${data.estimated_survival.estimated_months} months)
    </div>
    <div class="result-item">
      <strong>Confidence Range:</strong> ${data.estimated_survival.lower_bound}-${data.estimated_survival.upper_bound} months
    </div>
    <div class="result-item">
      <strong>Risk Score:</strong> ${data.risk_score.toFixed(3)}
    </div>
  `;
  
  outputDiv.innerHTML = html;
  resultsDiv.style.display = 'block';
});
</script>
</body>
</html>
"""


@app.post("/predict_survival")
async def predict_survival(
    file: UploadFile = File(...),
    sex: str = Form(...),
    age: int = Form(...),
    grade: str = Form(...),
    treatment: str = Form(...),
    histological_type: str = Form(...)
):
    # Read and transform image
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_tensor = transform(image=img)["image"].unsqueeze(0).to(device)
    
    # Encode clinical features
    clinical_data = {
        'Sex': sex,
        'Age': age,
        'Grade': grade,
        'Treatment': treatment,
        'Histological type': histological_type
    }
    clinical_features = torch.from_numpy(encoder.encode(clinical_data)).unsqueeze(0).to(device)
    
    if survival_model is not None:
        # Use survival model
        with torch.no_grad():
            outputs = survival_model(img_tensor, clinical_features)
            
            # Cancer prediction
            cancer_probs = torch.softmax(outputs['cancer_logits'], dim=1)[0]
            cancer_prob = cancer_probs[1].item()
            cancer_pred = "cancer" if cancer_prob >= 0.5 else "normal"
            
            # Survival status prediction
            survival_probs = torch.softmax(outputs['survival_logits'], dim=1)[0]
            survival_idx = survival_probs.argmax().item()
            survival_status = status_names[survival_idx]
            
            # Risk score
            risk_score = torch.tanh(outputs['risk_score']).item()
            
            # Estimate survival time
            status_short = ['NED', 'AWD', 'D'][survival_idx]
            survival_estimate = estimate_survival_months(status_short, risk_score, age, grade)
    else:
        # Fallback: use cancer-only model
        with torch.no_grad():
            cancer_probs = torch.softmax(cancer_model(img_tensor), dim=1)[0]
            cancer_prob = cancer_probs[1].item()
            cancer_pred = "cancer" if cancer_prob >= 0.5 else "normal"
            survival_status = "Unavailable (survival model not trained)"
            risk_score = 0.0
            survival_estimate = {
                'estimated_months': 0,
                'estimated_years': 0,
                'lower_bound': 0,
                'upper_bound': 0
            }
    
    return {
        "cancer_prediction": cancer_pred,
        "cancer_probability": cancer_prob,
        "survival_status": survival_status,
        "risk_score": risk_score,
        "estimated_survival": survival_estimate
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
