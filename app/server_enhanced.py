import os
import io
import base64
import argparse
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
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score

from src.model import build_model
from src.data import BoneCancerDataset
from src.survival_model import SurvivalPredictor, ClinicalFeatureEncoder, estimate_survival_months
from src.visualization import highlight_cancer_region
from src.config import Config

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

DEFAULT_CLINICAL = {
    'Sex': 'Male',
    'Age': 50,
    'Grade': 'Intermediate',
    'Treatment': 'Surgery',
    'Histological type': 'Osteosarcoma'
}

GLOBAL_ACCURACY = None
GLOBAL_PRECISION = None
SESSION_Y_TRUE = []
SESSION_Y_PRED = []


def build_clinical_batch(batch_size: int):
    features = encoder.encode(DEFAULT_CLINICAL)
    features = torch.from_numpy(features).to(device)
    return features.unsqueeze(0).expand(batch_size, -1)


def compute_global_metrics():
    global GLOBAL_ACCURACY, GLOBAL_PRECISION
    try:
        valid_csv = os.path.join(Config.dataset_root, "valid", "_classes.csv")
        ds = BoneCancerDataset(Config.dataset_root, "valid", valid_csv, img_size=Config.img_size, augment=False)
    except Exception as e_valid:
        try:
            test_csv = os.path.join(Config.dataset_root, "test", "_classes.csv")
            ds = BoneCancerDataset(Config.dataset_root, "test", test_csv, img_size=Config.img_size, augment=False)
        except Exception as e_test:
            print(f"Could not load validation/test dataset for metrics: {e_valid} / {e_test}")
            return

    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    model = None
    use_survival = False
    if 'survival_model' in globals() and globals().get('survival_model') is not None:
        model = globals().get('survival_model')
        use_survival = True
    elif 'cancer_model' in globals() and globals().get('cancer_model') is not None:
        model = globals().get('cancer_model')
    else:
        print("No model available for global metrics computation.")
        return

    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, y, _ in dl:
            x = x.to(device)
            if use_survival:
                clinical_batch = build_clinical_batch(x.size(0))
                outputs = model(x, clinical_batch)
                logits = outputs['cancer_logits']
            else:
                logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1]
            ys.append(y.numpy())
            preds.append((prob > 0.5).cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    GLOBAL_ACCURACY = accuracy_score(y_true, y_pred)
    GLOBAL_PRECISION = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    print(f"Loaded global metrics: accuracy={GLOBAL_ACCURACY:.4f}, precision={GLOBAL_PRECISION:.4f}")

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

compute_global_metrics()


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
    .metrics-section {
      background: white;
      border-radius: 12px;
      padding: 1.5rem;
      box-shadow: 0 5px 20px rgba(0,0,0,0.1);
      margin-top: 1.5rem;
    }
    .metrics-chart {
      display: flex;
      justify-content: space-around;
      align-items: flex-end;
      height: 180px;
      margin-top: 0.5rem;
    }
    .metric-bar {
      width: 90px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-end;
    }
    .metric-bar-inner {
      width: 100%;
      height: 0%;
      border-radius: 6px 6px 0 0;
      background: linear-gradient(180deg, #4caf50, #81c784);
      transition: height 0.6s ease;
    }
    .metric-bar-inner.precision {
      background: linear-gradient(180deg, #ff9800, #ffb74d);
    }
    .metric-bar-value {
      margin-top: 0.35rem;
      font-size: 0.85rem;
      font-weight: 600;
      color: #333;
    }
    .metric-bar-name {
      margin-top: 0.15rem;
      font-size: 0.9rem;
      color: #555;
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

        <div class="form-group" style="margin-bottom: 1.5rem;">
          <label>✅ Actual Label (optional, used for accuracy graph)</label>
          <select id="trueLabel" name="true_label">
            <option value="">-- Select if known --</option>
            <option value="normal">Normal</option>
            <option value="cancer">Cancer</option>
          </select>
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

      <div class="metrics-section" id="metricsSection" style="display: none;">
        <h2 style="margin-bottom: 1rem; color: #333; font-size: 1.2rem;">Model Accuracy &amp; Precision</h2>
        <div class="metrics-chart">
          <div class="metric-bar">
            <div class="metric-bar-inner" id="accuracyBar"></div>
            <div class="metric-bar-value" id="accuracyValue">0.0%</div>
            <div class="metric-bar-name">Accuracy</div>
          </div>
          <div class="metric-bar">
            <div class="metric-bar-inner precision" id="precisionBar"></div>
            <div class="metric-bar-value" id="precisionValue">0.0%</div>
            <div class="metric-bar-name">Precision</div>
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
  const labelInput = document.getElementById('trueLabel');
  if (labelInput && labelInput.value) {
    formData.append('true_label', labelInput.value);
  }
  
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
    
    const metricsSection = document.getElementById('metricsSection');
    if (data.metrics && data.metrics.accuracy != null && data.metrics.precision != null) {
      const acc = data.metrics.accuracy;
      const prec = data.metrics.precision;
      const accPct = (acc * 100).toFixed(1) + '%';
      const precPct = (prec * 100).toFixed(1) + '%';
      document.getElementById('accuracyValue').textContent = accPct;
      document.getElementById('precisionValue').textContent = precPct;
      document.getElementById('accuracyBar').style.height = Math.max(0, Math.min(100, acc * 100)) + '%';
      document.getElementById('precisionBar').style.height = Math.max(0, Math.min(100, prec * 100)) + '%';
      metricsSection.style.display = 'block';
    } else if (metricsSection) {
      metricsSection.style.display = 'none';
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
    file: UploadFile = File(...),
    true_label: str = Form(None)
):
    # Use default clinical values for prediction
    clinical_features = build_clinical_batch(1)
    # Read and transform image
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_tensor = transform(image=img_bgr)["image"].unsqueeze(0).to(device)
    
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
            survival_estimate = estimate_survival_months(status_short, risk_score, DEFAULT_CLINICAL['Age'], DEFAULT_CLINICAL['Grade'])
    else:
        # Fallback: use cancer-only model
        with torch.no_grad():
            cancer_probs = torch.softmax(cancer_model(img_tensor), dim=1)[0]
            cancer_prob = cancer_probs[1].item()
            cancer_pred = "cancer" if cancer_prob >= 0.5 else "normal"
            risk_score = (float(cancer_prob) * 2.0) - 1.0
            if cancer_pred == "normal":
                status_short = "NED"
                survival_status = status_names[0]
            else:
                if cancer_prob >= 0.85:
                    status_short = "D"
                    survival_status = status_names[2]
                else:
                    status_short = "AWD"
                    survival_status = status_names[1]
            survival_estimate = estimate_survival_months(
                status_short,
                risk_score,
                DEFAULT_CLINICAL['Age'],
                DEFAULT_CLINICAL['Grade'],
            )
    
    y_pred = 1 if cancer_pred == "cancer" else 0
    session_acc = None
    session_prec = None
    if true_label in ("normal", "cancer"):
        y_true = 1 if true_label == "cancer" else 0
        SESSION_Y_TRUE.append(y_true)
        SESSION_Y_PRED.append(y_pred)
        try:
            y_true_arr = np.array(SESSION_Y_TRUE)
            y_pred_arr = np.array(SESSION_Y_PRED)
            session_acc = float(accuracy_score(y_true_arr, y_pred_arr))
            session_prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
        except Exception:
            session_acc = None
            session_prec = None

    metric_acc = session_acc
    metric_prec = session_prec

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
        "tumor_analysis": tumor_analysis,
        "metrics": {
            "accuracy": metric_acc,
            "precision": metric_prec,
            "num_samples": len(SESSION_Y_TRUE)
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
    )
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
