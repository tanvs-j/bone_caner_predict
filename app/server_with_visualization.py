import os
import io
import base64
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn

from src.model import build_model
from src.survival_model import SurvivalPredictor, ClinicalFeatureEncoder, estimate_survival_months
from src.gradcam import GradCAM, TumorDetector, get_target_layer

# Model paths
SURVIVAL_MODEL_PATH = os.environ.get("SURVIVAL_CKPT", r"T:\bone_can_pre\models\survival_model_best.pt")
CANCER_MODEL_PATH = os.environ.get("BONE_CKPT", r"T:\bone_can_pre\models\efficientnet_b0_best.pt")

app = FastAPI(title="Bone Cancer Detection & Localization", version="3.0")

size = 384
transform = A.Compose([
    A.LongestMaxSize(max_size=size),
    A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = ClinicalFeatureEncoder()

# Load models
print(f"Using device: {device}")
survival_model = None
cancer_model = None
gradcam = None
status_names = ['NED (No Evidence of Disease)', 'AWD (Alive with Disease)', 'Dead']

# Try to load survival model
if os.path.exists(SURVIVAL_MODEL_PATH):
    print(f"Loading survival model from {SURVIVAL_MODEL_PATH}")
    base_model = build_model("efficientnet_b0", num_classes=2, pretrained=False)
    feature_extractor = nn.Sequential(*list(base_model.children())[:-1], nn.AdaptiveAvgPool2d(1), nn.Flatten())
    survival_model = SurvivalPredictor(feature_extractor, num_clinical_features=encoder.feature_dim)
    state = torch.load(SURVIVAL_MODEL_PATH, map_location=device, weights_only=False)
    survival_model.load_state_dict(state['model'])
    survival_model.to(device)
    survival_model.eval()
    print("✓ Survival model loaded!")

# Load cancer-only model for visualization
print(f"Loading cancer model from {CANCER_MODEL_PATH}")
cancer_model = build_model("efficientnet_b0", num_classes=2, pretrained=False)
state = torch.load(CANCER_MODEL_PATH, map_location=device, weights_only=True)
cancer_model.load_state_dict(state['model'])
cancer_model.to(device)
cancer_model.eval()

# Initialize Grad-CAM
target_layer = get_target_layer(cancer_model, "efficientnet_b0")
gradcam = GradCAM(cancer_model, target_layer)
print("✓ Grad-CAM initialized!")


def numpy_to_base64(img_array):
    """Convert numpy array to base64 string"""
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img_pil.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Bone Cancer Detection & Localization</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 20px;
    }
    .container { 
      max-width: 1200px;
      margin: 0 auto;
      background: white;
      padding: 2rem;
      border-radius: 15px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    h1 { 
      color: #2c3e50;
      text-align: center;
      margin-bottom: 0.5rem;
      font-size: 2.5em;
    }
    .subtitle {
      text-align: center;
      color: #7f8c8d;
      margin-bottom: 2rem;
      font-size: 1.1em;
    }
    .form-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
      margin-bottom: 2rem;
    }
    .form-group {
      margin-bottom: 1rem;
    }
    .form-group.full-width {
      grid-column: span 2;
    }
    label {
      display: block;
      margin-bottom: 0.5rem;
      font-weight: 600;
      color: #34495e;
    }
    input[type="file"],
    input[type="number"],
    select {
      width: 100%;
      padding: 0.75rem;
      border: 2px solid #e0e0e0;
      border-radius: 8px;
      font-size: 1em;
      transition: border-color 0.3s;
    }
    input:focus, select:focus {
      outline: none;
      border-color: #667eea;
    }
    .checkbox-group {
      display: flex;
      gap: 1.5rem;
      flex-wrap: wrap;
    }
    .checkbox-group label {
      display: flex;
      align-items: center;
      font-weight: normal;
    }
    .checkbox-group input[type="checkbox"] {
      width: auto;
      margin-right: 0.5rem;
    }
    button {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 1rem 3rem;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-size: 1.1em;
      font-weight: 600;
      width: 100%;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    button:hover {
      transform: translateY(-2px);
      box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    #loading {
      display: none;
      text-align: center;
      padding: 2rem;
      color: #667eea;
      font-size: 1.2em;
    }
    #results {
      display: none;
      margin-top: 2rem;
      animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .results-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 1.5rem;
      margin-top: 1.5rem;
    }
    .result-card {
      background: #f8f9fa;
      padding: 1.5rem;
      border-radius: 10px;
      border-left: 4px solid #667eea;
    }
    .result-card h3 {
      color: #2c3e50;
      margin-bottom: 1rem;
      font-size: 1.3em;
    }
    .result-item {
      margin: 0.75rem 0;
      padding: 0.5rem;
      background: white;
      border-radius: 5px;
    }
    .result-label {
      font-weight: 600;
      color: #7f8c8d;
      font-size: 0.9em;
    }
    .result-value {
      color: #2c3e50;
      font-size: 1.1em;
      margin-top: 0.25rem;
    }
    .prediction-cancer { color: #e74c3c; font-weight: bold; }
    .prediction-normal { color: #27ae60; font-weight: bold; }
    .images-container {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 1.5rem;
      margin-top: 2rem;
    }
    .image-card {
      background: #f8f9fa;
      padding: 1rem;
      border-radius: 10px;
      text-align: center;
    }
    .image-card img {
      width: 100%;
      border-radius: 8px;
      box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .image-card h4 {
      margin-top: 1rem;
      color: #2c3e50;
    }
    .severity-badge {
      display: inline-block;
      padding: 0.5rem 1rem;
      border-radius: 20px;
      font-weight: 600;
      margin-top: 0.5rem;
    }
    .severity-low { background: #d4edda; color: #155724; }
    .severity-moderate { background: #fff3cd; color: #856404; }
    .severity-high { background: #f8d7da; color: #721c24; }
  </style>
</head>
<body>
  <div class='container'>
    <h1>🩻 Bone Cancer Detection & Localization</h1>
    <p class="subtitle">AI-Powered Cancer Detection with Visual Localization and Survival Prediction</p>
    
    <form id="predictionForm">
      <div class="form-grid">
        <div class="form-group full-width">
          <label>📷 Upload X-ray Image:</label>
          <input type="file" id="image" accept="image/*" required />
        </div>
        
        <div class="form-group">
          <label>👤 Sex:</label>
          <select id="sex">
            <option value="Male">Male</option>
            <option value="Female">Female</option>
          </select>
        </div>
        
        <div class="form-group">
          <label>📅 Age:</label>
          <input type="number" id="age" value="50" min="1" max="120" required />
        </div>
        
        <div class="form-group">
          <label>📊 Tumor Grade:</label>
          <select id="grade">
            <option value="Low">Low</option>
            <option value="Intermediate" selected>Intermediate</option>
            <option value="High">High</option>
          </select>
        </div>
        
        <div class="form-group">
          <label>🧬 Histological Type:</label>
          <select id="histology">
            <option value="Osteosarcoma">Osteosarcoma</option>
            <option value="Leiomyosarcoma">Leiomyosarcoma</option>
            <option value="Liposarcoma">Liposarcoma</option>
            <option value="Ewing Sarcoma">Ewing Sarcoma</option>
            <option value="Chondrosarcoma">Chondrosarcoma</option>
            <option value="Other">Other</option>
          </select>
        </div>
        
        <div class="form-group full-width">
          <label>💊 Treatment (select applicable):</label>
          <div class="checkbox-group">
            <label><input type="checkbox" id="surgery" value="Surgery"> Surgery</label>
            <label><input type="checkbox" id="chemo" value="Chemotherapy"> Chemotherapy</label>
            <label><input type="checkbox" id="radio" value="Radiotherapy"> Radiotherapy</label>
          </div>
        </div>
      </div>
      
      <button type="submit">🔍 Analyze Image</button>
    </form>
    
    <div id="loading">
      <div>⏳ Analyzing image, please wait...</div>
    </div>
    
    <div id="results"></div>
  </div>

<script>
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
  e.preventDefault();
  
  const file = document.getElementById('image').files[0];
  if (!file) {
    alert('Please select an image');
    return;
  }
  
  document.getElementById('loading').style.display = 'block';
  document.getElementById('results').style.display = 'none';
  
  const treatments = [];
  if (document.getElementById('surgery').checked) treatments.push('Surgery');
  if (document.getElementById('chemo').checked) treatments.push('Chemotherapy');
  if (document.getElementById('radio').checked) treatments.push('Radiotherapy');
  
  const formData = new FormData();
  formData.append('file', file);
  formData.append('sex', document.getElementById('sex').value);
  formData.append('age', document.getElementById('age').value);
  formData.append('grade', document.getElementById('grade').value);
  formData.append('treatment', treatments.join(' + ') || 'None');
  formData.append('histological_type', document.getElementById('histology').value);
  
  try {
    const response = await fetch('/predict_with_visualization', { method: 'POST', body: formData });
    const data = await response.json();
    
    displayResults(data);
  } catch (error) {
    alert('Error: ' + error.message);
  } finally {
    document.getElementById('loading').style.display = 'none';
  }
});

function displayResults(data) {
  const predClass = data.cancer_prediction === 'cancer' ? 'prediction-cancer' : 'prediction-normal';
  const severityClass = 'severity-' + data.tumor_analysis.severity.level.toLowerCase();
  
  const html = `
    <h2 style="color: #2c3e50; margin-bottom: 1.5rem;">📋 Analysis Results</h2>
    
    <div class="results-grid">
      <div class="result-card">
        <h3>🔬 Cancer Detection</h3>
        <div class="result-item">
          <div class="result-label">Prediction</div>
          <div class="result-value ${predClass}">${data.cancer_prediction.toUpperCase()}</div>
        </div>
        <div class="result-item">
          <div class="result-label">Confidence</div>
          <div class="result-value">${(data.cancer_probability * 100).toFixed(1)}%</div>
        </div>
      </div>
      
      <div class="result-card">
        <h3>🎯 Tumor Analysis</h3>
        <div class="result-item">
          <div class="result-label">Detected Regions</div>
          <div class="result-value">${data.tumor_analysis.num_regions}</div>
        </div>
        <div class="result-item">
          <div class="result-label">Total Affected Area</div>
          <div class="result-value">${data.tumor_analysis.total_area.toLocaleString()} pixels</div>
        </div>
        <div class="result-item">
          <div class="result-label">Severity</div>
          <div class="result-value">
            <span class="severity-badge ${severityClass}">
              Stage ${data.tumor_analysis.severity.stage} - ${data.tumor_analysis.severity.level}
            </span>
          </div>
        </div>
      </div>
      
      <div class="result-card">
        <h3>⏳ Survival Prediction</h3>
        <div class="result-item">
          <div class="result-label">Status</div>
          <div class="result-value">${data.survival_status}</div>
        </div>
        <div class="result-item">
          <div class="result-label">Estimated Survival</div>
          <div class="result-value">${data.estimated_survival.estimated_years} years</div>
        </div>
        <div class="result-item">
          <div class="result-label">Range</div>
          <div class="result-value">${data.estimated_survival.lower_bound}-${data.estimated_survival.upper_bound} months</div>
        </div>
      </div>
    </div>
    
    <div class="images-container">
      <div class="image-card">
        <img src="data:image/png;base64,${data.images.original}" alt="Original">
        <h4>Original X-ray</h4>
      </div>
      <div class="image-card">
        <img src="data:image/png;base64,${data.images.heatmap}" alt="Heatmap">
        <h4>Cancer Heatmap</h4>
      </div>
      <div class="image-card">
        <img src="data:image/png;base64,${data.images.annotated}" alt="Annotated">
        <h4>Detected Regions</h4>
      </div>
    </div>
  `;
  
  document.getElementById('results').innerHTML = html;
  document.getElementById('results').style.display = 'block';
}
</script>
</body>
</html>
"""


@app.post("/predict_with_visualization")
async def predict_with_visualization(
    file: UploadFile = File(...),
    sex: str = Form(...),
    age: int = Form(...),
    grade: str = Form(...),
    treatment: str = Form(...),
    histological_type: str = Form(...)
):
    # Read image
    content = await file.read()
    img_pil = Image.open(io.BytesIO(content)).convert("RGB")
    img_np = np.array(img_pil)
    
    # Transform for model
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_tensor = transform(image=img_bgr)["image"].unsqueeze(0).to(device)
    
    # === Cancer Detection ===
    with torch.no_grad():
        cancer_probs = torch.softmax(cancer_model(img_tensor), dim=1)[0]
        cancer_prob = cancer_probs[1].item()
        cancer_pred = "cancer" if cancer_prob >= 0.5 else "normal"
    
    # === Grad-CAM Visualization ===
    heatmap = gradcam.generate_cam(img_tensor, target_class=1)  # Class 1 = cancer
    heatmap_vis = gradcam.generate_visualization(img_np, heatmap, alpha=0.4)
    
    # === Tumor Detection ===
    tumor_mask, tumor_stats = TumorDetector.detect_tumor_regions(heatmap, threshold=0.5)
    annotated_img = TumorDetector.draw_tumor_regions(img_np, tumor_stats['regions'])
    
    # === Survival Prediction ===
    clinical_data = {
        'Sex': sex,
        'Age': age,
        'Grade': grade,
        'Treatment': treatment,
        'Histological type': histological_type
    }
    
    if survival_model is not None:
        clinical_features = torch.from_numpy(encoder.encode(clinical_data)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = survival_model(img_tensor, clinical_features)
            survival_probs = torch.softmax(outputs['survival_logits'], dim=1)[0]
            survival_idx = survival_probs.argmax().item()
            survival_status = status_names[survival_idx]
            risk_score = torch.tanh(outputs['risk_score']).item()
            
            status_short = ['NED', 'AWD', 'D'][survival_idx]
            survival_estimate = estimate_survival_months(status_short, risk_score, age, grade)
    else:
        survival_status = "Model not available"
        risk_score = 0.0
        survival_estimate = {'estimated_months': 0, 'estimated_years': 0, 'lower_bound': 0, 'upper_bound': 0}
    
    # Convert images to base64
    original_b64 = numpy_to_base64(img_np)
    heatmap_b64 = numpy_to_base64(heatmap_vis)
    annotated_b64 = numpy_to_base64(annotated_img)
    
    # Convert numpy types to Python native types for JSON serialization
    tumor_stats_clean = {
        'num_regions': int(tumor_stats['num_regions']),
        'total_area': int(tumor_stats['total_area']),
        'severity': {
            'stage': int(tumor_stats['severity']['stage']),
            'level': str(tumor_stats['severity']['level']),
            'description': str(tumor_stats['severity']['description']),
            'score': float(tumor_stats['severity']['score'])
        },
        'regions': [
            {
                'area': int(r['area']),
                'centroid': [float(r['centroid'][0]), float(r['centroid'][1])],
                'bbox': [int(r['bbox'][0]), int(r['bbox'][1]), int(r['bbox'][2]), int(r['bbox'][3])]
            }
            for r in tumor_stats['regions']
        ]
    }
    
    return {
        "cancer_prediction": cancer_pred,
        "cancer_probability": float(cancer_prob),
        "tumor_analysis": tumor_stats_clean,
        "survival_status": survival_status,
        "risk_score": float(risk_score),
        "estimated_survival": {
            'estimated_months': int(survival_estimate['estimated_months']),
            'estimated_years': float(survival_estimate['estimated_years']),
            'lower_bound': int(survival_estimate['lower_bound']),
            'upper_bound': int(survival_estimate['upper_bound'])
        },
        "images": {
            "original": original_b64,
            "heatmap": heatmap_b64,
            "annotated": annotated_b64
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
