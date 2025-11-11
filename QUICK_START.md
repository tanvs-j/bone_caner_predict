# 🚀 Quick Start Guide

## Run the Enhanced System

### Option 1: Enhanced Server (Recommended)
```powershell
$env:PYTHONPATH="T:\bone_can_pre"
python app/server_enhanced.py
```
**Features**: Card-based UI, 3 visualizations, bounding boxes, heatmap

### Option 2: Original Enhanced Server  
```powershell
$env:PYTHONPATH="T:\bone_can_pre"
python app/server_survival.py
```
**Features**: Basic highlighting, 2 visualizations

## Access
Open your browser and go to: **http://localhost:8000**

## What You'll See

### Enhanced System (server_enhanced.py)
1. **Three Information Cards**:
   - 🔬 Cancer Detection (Prediction + Confidence)
   - 🎯 Tumor Analysis (Regions + Area + Severity)
   - 🏆 Survival Prediction (Status + Estimated Time)

2. **Three Visualizations**:
   - Original X-ray
   - Heatmap Analysis (gradient colors)
   - Detected Regions (bounding boxes with labels)

## Example Workflow
1. Click "📁 X-ray Image" and select an image
2. Fill in patient details:
   - Sex: Male/Female
   - Age: e.g., 50
   - Tumor Grade: Intermediate
   - Histological Type: Osteosarcoma
   - Treatment: Check applicable boxes
3. Click "🔍 Analyze Image"
4. Wait 2-3 seconds for analysis
5. Scroll down to view results

## Key Differences

| Feature | server_enhanced.py | server_survival.py |
|---------|-------------------|-------------------|
| UI Design | Modern card-based | Basic HTML |
| Visualizations | 3 images | 2 images |
| Tumor Regions | Counted with boxes | Basic area only |
| Heatmap | ✅ Yes | ❌ No |
| Bounding Boxes | ✅ Yes | ❌ No |
| Loading Animation | ✅ Yes | ❌ No |

## Troubleshooting

### Port Already in Use
```powershell
# Kill process on port 8000
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force
```

### Module Not Found
```powershell
# Ensure PYTHONPATH is set
$env:PYTHONPATH="T:\bone_can_pre"
```

### Model Not Found
- Ensure models exist in `T:\bone_can_pre\models\`
- Required: `survival_model_best.pt` or `efficientnet_b0_best.pt`

## Sample Test Images
Use images from:
- `T:\bone_can_pre\dataset\test\cancer\`
- `T:\bone_can_pre\dataset\test\normal\`

## Documentation
- **Full Guide**: `ENHANCED_ANALYSIS_GUIDE.md`
- **Highlighting Feature**: `CANCER_HIGHLIGHTING_FEATURE.md`
- **Main README**: `README.md`

---
**Need Help?** Check the documentation files or review the code in `app/` and `src/` directories.
