import os
import io
import base64
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
from src.visualization import highlight_cancer_region

# Model paths
SURVIVAL_MODEL_PATH = os.environ.get("SURVIVAL_CKPT", r"T:\bone_can_pre\models\survival_model_best.pt")
CANCER_MODEL_PATH = os.environ.get("BONE_CKPT", r"T:\bone_can_pre\models\efficientnet_b0_best.pt")

app = FastAPI(title="Bone Cancer Advanced Analysis", version="3.0")

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
  <title>Bone Cancer Advanced Analysis</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 2rem;
    }
    .container { 
      max-width: 1400px;
      margin: 0 auto;
    }
    .header {
      text-align: center;
      color: white;
      margin-bottom: 2rem;
    }
    .header h1 {
      font-size: 2.5rem;
      margin-bottom: 0.5rem;
    }
    .form-card {
      background: white;
      border-radius: 12px;
      padding: 2rem;
      box-shadow: 0 10px 40px rgba(0,0,0,0.2);
      margin-bottom: 2rem;
    }
    .form-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 1.5rem;
      margin-bottom: 1.5rem;
    }
    .form-group {
      display: flex;
      flex-direction: column;
    }
    label {
      font-weight: 600;
      margin-bottom: 0.5rem;
      color: #333;
      font-size: 0.9rem;
    }
    input[type="file"], input[type="number"], select {
      padding: 0.75rem;
      border: 2px solid #e0e0e0;
      border-radius: 8px;
      font-size: 1rem;
      transition: border-color 0.3s;
    }
    input:focus, select:focus {
      outline: none;
      border-color: #667eea;
    }
    .checkbox-group {
      display: flex;
      gap: 1rem;
      flex-wrap: wrap;
    }
    .checkbox-group label {
      display: flex;
      align-items: center;
      font-weight: normal;
    }
    .checkbox-group input[type="checkbox"] {
      margin-right: 0.5rem;
      width: 18px;
      height: 18px;
    }
    .submit-btn {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 1rem 3rem;
      border: none;
      border-radius: 8px;
      font-size: 1.1rem;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.2s;
      width: 100%;
    }
    .submit-btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    .results-container {
      display: none;
    }
    .cards-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 1.5rem;
      margin-bottom: 2rem;
    }
    .card {
      background: white;
      border-radius: 12px;
      padding: 1.5rem;
      box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    .card-header {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 1rem;
      padding-bottom: 1rem;
      border-bottom: 2px solid #f0f0f0;
    }
    .card-icon {
      font-size: 1.5rem;
    }
    .card-title {
      font-size: 1.2rem;
      font-weight: 600;
      color: #333;
    }
    .card-body {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    .metric {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .metric-label {
      color: #666;
      font-size: 0.9rem;
    }
    .metric-value {
      font-weight: 600;
      font-size: 1.1rem;
      color: #333;
    }
    .status-badge {
      display: inline-block;
      padding: 0.5rem 1rem;
      border-radius: 20px;
      font-weight: 600;
      font-size: 1rem;
    }
    .status-normal {
      background: #d4edda;
      color: #155724;
    }
    .status-cancer {
      background: #f8d7da;
      color: #721c24;
    }
    .severity-low {
      background: #fff3cd;
      color: #856404;
    }
    .severity-moderate {
      background: #ffeaa7;
      color: #d63031;
    }
    .severity-high {
      background: #ff7675;
      color: white;
    }
    .images-section {
      background: white;
      border-radius: 12px;
      padding: 2rem;
      box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    .images-section h2 {
      margin-bottom: 1.5rem;
      color: #333;
    }
    .images-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 1.5rem;
    }
    .image-container {
      text-align: center;
    }
    .image-container h3 {
      margin-bottom: 1rem;
      color: #555;
      font-size: 1.1rem;
    }
    .image-container img {
      width: 100%;
      border-radius: 8px;
      box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    .loading {
      text-align: center;
      padding: 2rem;
      color: white;
      font-size: 1.2rem;
      display: none;
    }
    .spinner {
      border: 4px solid rgba(255,255,255,0.3);
      border-top: 4px solid white;
      border-radius: 50%;
      width: 40px;
      height: 40px;
      animation: spin 1s linear infinite;
      margin: 0 auto 1rem;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <div class='container'>
    <div class='header'>
      <h1>🏥 Bone Cancer Advanced Analysis System</h1>
      <p>AI-Powered Cancer Detection with Survival Prediction</p>
    </div>
    
    <div class='form-card'>
      <form id="predictionForm">
        <div class="form-group" style="margin-bottom: 1.5rem;">
          <label>📁 Upload X-ray Image for Cancer Detection</label>
          <input type="file" id="image" accept="image/*" required style="padding: 1rem; font-size: 1rem;" />
        </div>
        
        <button type="submit" class="submit-btn">🔍 Analyze Image</button>
      </form>
    </div>
    
    <div class="loading" id="loading">
      <div class="spinner"></div>
      <p>Analyzing image... Please wait</p>
    </div>
    
    <div class="results-container" id="results">
      <!-- Cancer Detection Card (Always shown) -->
      <div class="cards-row" id="detectionCard">
        <div class="card" style="max-width: 500px; margin: 0 auto;">
          <div class="card-header">
            <span class="card-icon">🔬</span>
            <span class="card-title">Cancer Detection Result</span>
          </div>
          <div class="card-body">
            <div class="metric">
              <span class="metric-label">Prediction</span>
              <span class="status-badge" id="cancerStatus">NORMAL</span>
            </div>
            <div class="metric">
              <span class="metric-label">Confidence</span>
              <span class="metric-value" id="cancerConfidence">0.0%</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Additional Analysis (Only shown if cancer detected) -->
      <div id="cancerDetailsSection" style="display: none; margin-top: 2rem;">
        <div class="cards-row">
          <div class="card">
            <div class="card-header">
              <span class="card-icon">🎯</span>
              <span class="card-title">Tumor Analysis</span>
            </div>
            <div class="card-body">
              <div class="metric">
                <span class="metric-label">Detected Regions</span>
                <span class="metric-value" id="detectedRegions">0</span>
              </div>
              <div class="metric">
                <span class="metric-label">Total Affected Area</span>
                <span class="metric-value" id="affectedArea">0 pixels</span>
              </div>
              <div class="metric">
                <span class="metric-label">Severity Stage</span>
                <span class="status-badge severity-low" id="severity">Stage 1 - Low</span>
              </div>
            </div>
          </div>
          
          <div class="card">
            <div class="card-header">
              <span class="card-icon">⏱️</span>
              <span class="card-title">Estimated Lifespan</span>
            </div>
            <div class="card-body">
              <div class="metric">
                <span class="metric-label">Survival Status</span>
                <span class="metric-value" id="survivalStatus">-</span>
              </div>
              <div class="metric">
                <span class="metric-label">Estimated Time</span>
                <span class="metric-value" id="estimatedSurvival">-</span>
              </div>
              <div class="metric">
                <span class="metric-label">Range</span>
                <span class="metric-value" id="survivalRange">-</span>
              </div>
            </div>
          </div>
        </div>
        
        <div class="images-section">
          <h2>📊 Detailed Visual Analysis</h2>
          <div class="images-grid">
            <div class="image-container">
              <h3>Original X-ray</h3>
              <img id="originalImage" alt="Original" />
            </div>
            <div class="image-container">
              <h3>Contrast RGB Highlights</h3>
              <img id="heatmapImage" alt="Contrast Highlights" />
            </div>
            <div class="image-container">
              <h3>Box Findings</h3>
              <img id="bboxImage" alt="Bounding Boxes" />
            </div>
          </div>
        </div>
      </div>
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
  
  // Show loading
  document.getElementById('loading').style.display = 'block';
  document.getElementById('results').style.display = 'none';
  
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await fetch('/predict', { method: 'POST', body: formData });
    const data = await response.json();
    
    // Hide loading
    document.getElementById('loading').style.display = 'none';
    
    // Update cancer detection card
    const cancerStatus = document.getElementById('cancerStatus');
    cancerStatus.textContent = data.cancer_prediction.toUpperCase();
    cancerStatus.className = 'status-badge ' + (data.cancer_prediction === 'cancer' ? 'status-cancer' : 'status-normal');
    document.getElementById('cancerConfidence').textContent = (data.cancer_probability * 100).toFixed(1) + '%';
    
    // Show results card
    document.getElementById('results').style.display = 'block';
    
    // If cancer detected, show additional details
    if (data.cancer_prediction === 'cancer') {
      // Update tumor analysis card
      document.getElementById('detectedRegions').textContent = data.tumor_analysis.detected_regions;
      document.getElementById('affectedArea').textContent = data.tumor_analysis.tumor_area + ' pixels';
      const severity = document.getElementById('severity');
      severity.textContent = data.tumor_analysis.severity;
      if (data.tumor_analysis.severity.includes('Low')) {
        severity.className = 'status-badge severity-low';
      } else if (data.tumor_analysis.severity.includes('Moderate')) {
        severity.className = 'status-badge severity-moderate';
      } else {
        severity.className = 'status-badge severity-high';
      }
      
      // Update survival prediction card
      document.getElementById('survivalStatus').textContent = data.survival_status;
      document.getElementById('estimatedSurvival').textContent = 
        data.estimated_survival.estimated_years + ' years (' + data.estimated_survival.estimated_months + ' months)';
      document.getElementById('survivalRange').textContent = 
        data.estimated_survival.lower_bound + '-' + data.estimated_survival.upper_bound + ' months';
      
      // Display images
      document.getElementById('originalImage').src = 'data:image/jpeg;base64,' + data.original_image;
      document.getElementById('heatmapImage').src = 'data:image/jpeg;base64,' + data.heatmap_image;
      document.getElementById('bboxImage').src = 'data:image/jpeg;base64,' + data.bbox_image;
      
      // Show cancer details section
      document.getElementById('cancerDetailsSection').style.display = 'block';
    } else {
      // Hide cancer details section for normal predictions
      document.getElementById('cancerDetailsSection').style.display = 'none';
    }
    
    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
  } catch (error) {
    document.getElementById('loading').style.display = 'none';
    alert('Error analyzing image: ' + error.message);
  }
});
</script>
</body>
</html>
"""


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # Use default clinical values for prediction
    sex = "Male"
    age = 50
    grade = "Intermediate"
    treatment = "Surgery"
    histological_type = "Osteosarcoma"
    # Read and transform image
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_tensor = transform(image=img_bgr)["image"].unsqueeze(0).to(device)
    
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
    
    # Generate three visualization styles
    heatmap_img, bbox_img, overlay_img, tumor_analysis = highlight_cancer_region(
        img_np, cancer_prob, method='advanced'
    )
    
    # Convert images to base64
    def img_to_base64(img_array):
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=90)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    original_b64 = img_to_base64(img_np)
    heatmap_b64 = img_to_base64(heatmap_img)
    bbox_b64 = img_to_base64(bbox_img)
    
    return {
        "cancer_prediction": cancer_pred,
        "cancer_probability": cancer_prob,
        "survival_status": survival_status,
        "risk_score": risk_score,
        "estimated_survival": survival_estimate,
        "original_image": original_b64,
        "heatmap_image": heatmap_b64,
        "bbox_image": bbox_b64,
        "tumor_analysis": tumor_analysis
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
