#!/usr/bin/env python3
"""
Enhanced Bone Cancer Prediction API with Survival Prediction and Cancer Highlighting
Based on sample_models folder implementations
"""

import os
import io
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from skimage import measure, feature
from skimage.filters import sobel
from skimage.morphology import binary_erosion, binary_dilation, disk
import base64
from io import BytesIO

from src.model import build_model
from src.cancer_highlighting import (
    create_cancer_highlight_image,
    generate_analysis_report,
    create_multi_panel_visualization
)

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import joblib

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model paths
MODEL_DIR = os.path.join("T:", "bone_can_pre", "models")
CHECKPOINT_PATH = os.environ.get("BONE_CKPT", os.path.join(MODEL_DIR, "efficientnet_b0_best.pt"))
DOT_MODEL_PATH = os.path.join("T:", "bone_can_pre", "sample_models", "bone_cancer_dot_model.pkl")

# Load cancer classification model
model_name = "efficientnet_b0" if "efficientnet" in CHECKPOINT_PATH.lower() else "mobilenet_v3_small"
model = build_model(model_name, num_classes=2, pretrained=False).to(device)

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"✅ Cancer classification model loaded")
else:
    print(f"⚠️  Model checkpoint not found: {CHECKPOINT_PATH}")
    exit(1)

# Load dot-based model for cancer highlighting
try:
    dot_model = joblib.load(DOT_MODEL_PATH)
    print("✅ Dot-based cancer highlighting model loaded")
except Exception as e:
    print(f"⚠️  Could not load dot model: {e}")
    dot_model = None

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = FastAPI(title="Enhanced Bone Cancer Prediction API")

def analyze_image_with_dots(image_bgr):
    """Analyze image using dot-based approach from sample_models"""
    if dot_model is None:
        return None, None, None, None, None
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Feature extraction (similar to sample_models)
        features = []
        
        # Resize image for feature extraction
        resized = cv2.resize(enhanced, (50, 50))
        features.extend(resized.flatten())
        
        # Add statistical features
        features.extend([np.mean(enhanced), np.std(enhanced)])
        
        # Predict using dot model
        features_array = np.array(features).reshape(1, -1)
        prediction = dot_model.predict(features_array)[0]
        
        # Create tumor mask using K-means clustering
        pixels = image_bgr.reshape(-1, 3)
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(pixels)
        
        # Assume tumor is the cluster with highest intensity
        cluster_means = [np.mean(pixels[labels == i]) for i in range(3)]
        tumor_cluster = np.argmax(cluster_means)
        
        tumor_mask = (labels == tumor_cluster).reshape(image_bgr.shape[:2])
        
        # Clean up mask
        tumor_mask = binary_erosion(tumor_mask, disk(2))
        tumor_mask = binary_dilation(tumor_mask, disk(3))
        
        # Count tumor blobs
        labeled_regions = measure.label(tumor_mask.astype(np.uint8))
        tumor_blobs = len(measure.regionprops(labeled_regions))
        
        # Calculate tumor area
        tumor_area = np.sum(tumor_mask)
        
        return tumor_mask, tumor_blobs, tumor_area, enhanced, edges
        
    except Exception as e:
        print(f"Dot analysis error: {e}")
        return None, None, None, None, None

def estimate_survival_simple(cancer_prob, stage_info=None):
    """Simple survival estimation based on cancer probability"""
    if cancer_prob < 0.3:
        return "Excellent prognosis (>5 years)", "NED (No Evidence of Disease)", 0.1
    elif cancer_prob < 0.6:
        return "Good prognosis (3-5 years)", "AWD (Alive With Disease)", 0.4
    else:
        return "Guarded prognosis (1-3 years)", "High Risk", 0.8

def predict_enhanced(image):
    """Enhanced prediction with survival and highlighting"""
    # Basic prediction
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        prob = torch.sigmoid(outputs).cpu().numpy()[0][0]
    
    # Convert PIL to BGR for OpenCV processing
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Analyze with dot-based approach
    tumor_mask, tumor_blobs, tumor_area, enhanced, edges = analyze_image_with_dots(img_bgr)
    
    # Determine prediction and stage
    prediction = "cancer" if prob > 0.5 else "normal"
    
    # Estimate cancer stage
    if prediction == "cancer" and tumor_area is not None:
        if tumor_area < 1000:
            stage_text = "Stage I (Early)"
            stage_num = 1
        elif tumor_area < 5000:
            stage_text = "Stage II (Localized)"
            stage_num = 2
        elif tumor_area < 15000:
            stage_text = "Stage III (Regional)"
            stage_num = 3
        else:
            stage_text = "Stage IV (Advanced)"
            stage_num = 4
    else:
        stage_text = "No cancer detected"
        stage_num = 0
    
    # Survival prediction
    if prediction == "cancer":
        estimated_survival, survival_status, risk_score = estimate_survival_simple(prob, stage_text)
    else:
        estimated_survival = "Excellent prognosis - No cancer detected"
        survival_status = "NED (No Evidence of Disease)"
        risk_score = 0.0
    
    # Generate visualizations if cancer detected
    visualization_data = {}
    if prediction == "cancer" and tumor_mask is not None and tumor_area > 0:
        try:
            # Create highlighted image
            highlighted_image = create_cancer_highlight_image(img_bgr, tumor_mask)
            
            # Generate analysis report
            analysis_report = generate_analysis_report(tumor_mask, img_bgr, prob, stage_num)
            
            # Create multi-panel visualization
            multi_panel = create_multi_panel_visualization(img_bgr, tumor_mask, enhanced, edges)
            
            visualization_data = {
                "highlighted_image": highlighted_image,
                "multi_panel_visualization": multi_panel,
                "analysis_report": analysis_report
            }
        except Exception as e:
            print(f"Visualization error: {e}")
            visualization_data = {"error": "Visualization failed"}
    
    return {
        "cancer_probability": float(prob),
        "prediction": prediction,
        "stage": stage_text,
        "tumor_blobs": tumor_blobs if prediction == "cancer" else None,
        "tumor_area": tumor_area if prediction == "cancer" else None,
        "survival_status": survival_status,
        "risk_score": risk_score if prediction == "cancer" else None,
        "estimated_survival": estimated_survival,
        "analysis_method": "Dot-based K-means clustering + CNN classification",
        "visualization": visualization_data if prediction == "cancer" else None
    }

@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced Bone Cancer Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .upload-area { border: 2px dashed #ccc; border-radius: 10px; padding: 40px; text-align: center; margin: 20px 0; }
            .upload-area:hover { border-color: #007bff; }
            .result { margin-top: 30px; padding: 20px; border-radius: 10px; }
            .cancer-result { background: #ffe6e6; border: 1px solid #ff9999; }
            .normal-result { background: #e6ffe6; border: 1px solid #99ff99; }
            .highlight-info { background: #fff3cd; padding: 15px; border-radius: 5px; margin-top: 10px; }
            button { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #0056b3; }
            .loading { display: none; text-align: center; margin: 20px; }
            .visualization { margin-top: 20px; }
            .visualization img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🦴 Enhanced Bone Cancer Prediction</h1>
            <p>Upload a bone X-ray image to get cancer prediction, survival analysis, and cancer area highlighting.</p>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-area">
                    <input type="file" id="fileInput" name="file" accept="image/*" style="display: none;">
                    <p>📸 Click here or drag and drop an image</p>
                    <p style="color: #666; font-size: 14px;">Supported formats: JPG, PNG, JPEG</p>
                </div>
                <button type="submit">🔍 Analyze Image</button>
            </form>
            
            <div class="loading" id="loading">
                <p>⏳ Analyzing your image... This may take a moment.</p>
            </div>
            
            <div id="result" class="result" style="display:none;">
                <h3>📊 Analysis Results</h3>
                <div id="prediction"></div>
                <div id="survival"></div>
                <div id="highlighting"></div>
                <div id="visualization" style="margin-top:2rem;"></div>
            </div>
        </div>
        
        <script>
        document.querySelector('.upload-area').addEventListener('click', () => {
            document.getElementById('fileInput').click();
        });
        
        document.getElementById('fileInput').addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                document.querySelector('.upload-area p').textContent = `📁 ${e.target.files[0].name}`;
            }
        });
        
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files.length) {
                alert('Please select an image first!');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                displayResults(data);
                
            } catch (error) {
                alert('Error analyzing image: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });
        
        function displayResults(data) {
            const resultDiv = document.getElementById('result');
            const predictionDiv = document.getElementById('prediction');
            const survivalDiv = document.getElementById('survival');
            const highlightingDiv = document.getElementById('highlighting');
            const visualizationDiv = document.getElementById('visualization');
            
            // Prediction results
            const isCancer = data.prediction === 'cancer';
            predictionDiv.className = isCancer ? 'cancer-result' : 'normal-result';
            predictionDiv.innerHTML = `
                <h4>🎯 Cancer Detection</h4>
                <p><strong>Prediction:</strong> ${data.prediction.toUpperCase()}</p>
                <p><strong>Confidence:</strong> ${(data.cancer_probability * 100).toFixed(1)}%</p>
                ${data.stage ? `<p><strong>Cancer Stage:</strong> ${data.stage}</p>` : ''}
                ${data.tumor_blobs ? `<p><strong>Tumor Spots Detected:</strong> ${data.tumor_blobs}</p>` : ''}
            `;
            
            // Survival prediction
            survivalDiv.innerHTML = `
                <h4>⏳ Survival Prediction</h4>
                <p><strong>Estimated Survival:</strong> ${data.estimated_survival}</p>
                ${data.survival_status ? `<p><strong>Survival Status:</strong> ${data.survival_status}</p>` : ''}
                ${data.risk_score ? `<p><strong>Risk Score:</strong> ${data.risk_score.toFixed(3)}</p>` : ''}
            `;
            
            // Cancer highlighting info
            if (data.tumor_area) {
                highlightingDiv.innerHTML = `
                    <h4>🎯 Cancer Area Analysis</h4>
                    <div class="highlight-info">
                        <p><strong>Tumor Area:</strong> ${data.tumor_area} pixels</p>
                        <p><strong>Analysis Method:</strong> ${data.analysis_method}</p>
                        <p><em>📝 Note: The cancer areas have been highlighted in red in the processed image.</em></p>
                    </div>
                `;
            }
            
            // Visualizations
            if (data.visualization && data.visualization.highlighted_image) {
                visualizationDiv.innerHTML = `
                    <h4>🔍 Cancer Area Visualization</h4>
                    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 300px;">
                            <h5>Highlighted Cancer Areas</h5>
                            <img src="data:image/png;base64,${data.visualization.highlighted_image}" 
                                 style="width: 100%; max-width: 400px; border: 1px solid #ccc; border-radius: 5px;">
                        </div>
                        <div style="flex: 1; min-width: 300px;">
                            <h5>Multi-Panel Analysis</h5>
                            <img src="data:image/png;base64,${data.visualization.multi_panel_visualization}" 
                                 style="width: 100%; max-width: 400px; border: 1px solid #ccc; border-radius: 5px;">
                        </div>
                    </div>
                `;
                
                // Add detailed analysis report
                if (data.visualization.analysis_report) {
                    const report = data.visualization.analysis_report;
                    visualizationDiv.innerHTML += `
                        <div style="margin-top: 20px; background: #f8f9fa; padding: 15px; border-radius: 5px;">
                            <h5>📊 Detailed Analysis Report</h5>
                            <p><strong>Tumor Coverage:</strong> ${report.tumor_percentage}% of image</p>
                            <p><strong>Number of Tumor Regions:</strong> ${report.num_tumor_regions}</p>
                            <p><strong>Average Tumor Size:</strong> ${report.average_tumor_size} pixels</p>
                            <p><strong>Largest Tumor:</strong> ${report.largest_tumor_area} pixels</p>
                            <p><strong>Severity Assessment:</strong> ${report.severity_assessment}</p>
                            ${report.risk_factors.length > 0 ? 
                                `<p><strong>Risk Factors:</strong> ${report.risk_factors.join(', ')}</p>` : ''}
                        </div>
                    `;
                }
            }
            
            resultDiv.style.display = 'block';
        }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Enhanced prediction endpoint with survival and highlighting"""
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Get enhanced prediction
        result = predict_enhanced(image)
        
        return result
        
    except Exception as e:
        return {"error": str(e), "prediction": "error", "cancer_probability": 0.0}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)