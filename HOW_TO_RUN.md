# 📖 How to Run - Bone Cancer Prediction System

Complete step-by-step instructions for setting up and running the bone cancer detection and survival prediction system.

---

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Verify Installation](#verify-installation)
4. [Running the Application](#running-the-application)
5. [Using the Web Interface](#using-the-web-interface)
6. [Training Models (Optional)](#training-models-optional)
7. [Troubleshooting](#troubleshooting)
8. [Project Structure](#project-structure)

---

## 1. Prerequisites

### Required Software
- **Python 3.8 or higher** (Python 3.10+ recommended)
- **Git** (for cloning the repository)
- **Web Browser** (Chrome, Firefox, Edge, or Safari)
- **Windows OS** (Instructions are for Windows, but adaptable to Linux/Mac)

### Check Python Installation
Open PowerShell and run:
```powershell
python --version
```
Expected output: `Python 3.8.x` or higher

If Python is not installed:
1. Download from [python.org](https://www.python.org/downloads/)
2. Install with "Add Python to PATH" option checked
3. Restart your terminal

---

## 2. Installation

### Step 1: Navigate to Project Directory
```powershell
cd T:\bone_can_pre
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

This will install:
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision models and utilities
- `fastapi` - Web framework for the API
- `uvicorn` - ASGI server
- `opencv-python` - Image processing
- `albumentations` - Image augmentation
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning utilities
- `Pillow` - Image handling
- `python-multipart` - File upload support

**Installation time**: 5-10 minutes (depending on internet speed)

### Step 3: Verify Models are Present
```powershell
dir models\*.pt
```

Expected files:
- `efficientnet_b0_best.pt` (Cancer classification model)
- `survival_model_best.pt` (Survival prediction model)
- `mobilenet_v3_small_best.pt` (Alternative model)

✅ If files exist, you're ready to go!
❌ If files don't exist, see [Training Models](#training-models-optional)

---

## 3. Verify Installation

### Test 1: Check Python Packages
```powershell
python -c "import torch, cv2, fastapi; print('✓ All packages installed successfully')"
```

### Test 2: Check CUDA (GPU Support - Optional)
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```
- `True` = GPU acceleration available
- `False` = Will use CPU (slower but still works)

---

## 4. Running the Application

### 🎯 Option A: Enhanced Server (Recommended)

**Features**: Card-based UI, 3 visualizations, bounding boxes, heatmap

#### Step 1: Set Environment Variable
```powershell
$env:PYTHONPATH="T:\bone_can_pre"
```

#### Step 2: Start the Server
```powershell
python app/server_enhanced.py
```

#### Step 3: Wait for Startup
You should see:
```
Loading survival model from T:\bone_can_pre\models\survival_model_best.pt
Survival model loaded successfully!
INFO:     Started server process [xxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### Step 4: Open Browser
Navigate to: **http://localhost:8000**

---

### 🔧 Option B: Original Survival Server

**Features**: Basic UI, 2 visualizations

```powershell
$env:PYTHONPATH="T:\bone_can_pre"
python app/server_survival.py
```

---

### 🎨 Option C: Gradio Interface

**Features**: Simple drag-and-drop interface

```powershell
$env:PYTHONPATH="T:\bone_can_pre"
python app/ui.py
```
Access at: **server calhost:7860**

---

### 🚀 Option D: All-in-One Script

Trains models and starts server automatically:
```powershell
.\run_everything.bat
```
⚠️ **Warning**: This will take 10-30 minutes for training

---

## 5. Using the Web Interface

### Step-by-Step Usage

#### 1. **Upload X-ray Image**
   - Click the "📁 X-ray Image" button
   - Select a bone X-ray image (.jpg, .png, etc.)
   - Supported formats: JPEG, PNG, BMP

#### 2. **Enter Patient Information**
   - **Sex**: Select Male or Female
   - **Age**: Enter patient age (1-120)
   - **Tumor Grade**: Select Low, Intermediate, or High
   - **Histological Type**: Choose tumor type
     - Osteosarcoma
     - Leiomyosarcoma
     - Liposarcoma
     - Other
   - **Treatment**: Check all applicable boxes
     - ☐ Surgery
     - ☐ Chemotherapy
     - ☐ Radiotherapy

#### 3. **Analyze**
   - Click "🔍 Analyze Image" button
   - Wait 2-3 seconds for processing
   - Loading animation will appear

#### 4. **View Results**

**Three Information Cards**:

**🔬 Cancer Detection**
- Prediction: NORMAL or CANCER
- Confidence: Percentage (0-100%)

**🎯 Tumor Analysis**
- Detected Regions: Number of tumor areas
- Total Affected Area: Size in pixels
- Severity: Stage 1 (Low), 2 (Moderate), or 3 (High)

**🏆 Survival Prediction**
- Status: NED / AWD / Dead
- Estimated Survival: Years and months
- Range: Confidence interval

**📊 Visual Analysis** (3 Images):
1. **Original X-ray**: Uploaded image
2. **Heatmap Analysis**: Color-coded intensity map
3. **Detected Regions**: Bounding boxes with labels

---

## 6. Training Models (Optional)

If you need to train models from scratch:

### Train Cancer Classification Model
```powershell
$env:PYTHONPATH="T:\bone_can_pre"
python scripts/train.py --epochs 10 --batch-size 16
```
**Time**: 10-20 minutes
**Output**: `models/efficientnet_b0_best.pt`

### Train Survival Prediction Model
```powershell
$env:PYTHONPATH="T:\bone_can_pre"
python scripts/train_survival.py --epochs 15 --batch-size 16
```
**Time**: 15-25 minutes
**Output**: `models/survival_model_best.pt`

### Dataset Requirements
Ensure your dataset is organized as:
```
dataset/
├── train/
│   ├── normal/
│   │   └── *.jpg
│   ├── cancer/
│   │   └── *.jpg
│   ├── _classes.csv
│   └── Bone Tumor Dataset.csv
├── valid/
│   ├── normal/
│   ├── cancer/
│   └── _classes.csv
└── test/
    ├── normal/
    └── cancer/
```

---

## 7. Troubleshooting

### Problem: "Port already in use"
**Error**: `[Errno 10048] error while attempting to bind on address`

**Solution**:
```powershell
# Find and kill process using port 8000
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force
```

Or use a different port:
```powershell
python app/server_enhanced.py --port 8001
```
Then access at: http://localhost:8001

---

### Problem: "Module not found"
**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```powershell
# Reinstall dependencies
pip install -r requirements.txt

# Or install specific package
pip install torch torchvision
```

---

### Problem: "PYTHONPATH not set"
**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```powershell
# Set PYTHONPATH before running
$env:PYTHONPATH="T:\bone_can_pre"
```

To set permanently (Windows):
1. Right-click "This PC" → Properties
2. Advanced system settings → Environment Variables
3. Add new variable:
   - Name: `PYTHONPATH`
   - Value: `T:\bone_can_pre`

---

### Problem: "Model file not found"
**Error**: `FileNotFoundError: models/survival_model_best.pt`

**Solution 1**: Check if models exist
```powershell
dir models\
```

**Solution 2**: Train models
```powershell
.\run_everything.bat
```

**Solution 3**: Download pre-trained models (if available)
- Contact repository maintainer
- Or train from scratch using `scripts/train.py`

---

### Problem: "CUDA out of memory"
**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Use CPU instead
```powershell
$env:CUDA_VISIBLE_DEVICES=""
python app/server_enhanced.py
```

Or reduce batch size during training:
```powershell
python scripts/train.py --batch-size 8
```

---

### Problem: Slow predictions
**Cause**: Running on CPU instead of GPU

**Solutions**:
1. Check GPU availability:
   ```powershell
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. Install CUDA-enabled PyTorch:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. Accept slower CPU predictions (30-60 seconds vs 2-3 seconds)

---

### Problem: Image upload fails
**Possible causes**:
- File too large (>10MB)
- Unsupported format
- Corrupted image

**Solution**:
- Use JPEG or PNG format
- Resize large images before upload
- Test with sample images from `dataset/test/cancer/`

---

## 8. Project Structure

```
bone_can_pre/
├── app/                          # Web applications
│   ├── server_enhanced.py        # Enhanced server (RECOMMENDED)
│   ├── server_survival.py        # Basic survival server
│   ├── server.py                 # Simple cancer detection
│   └── ui.py                     # Gradio interface
│
├── src/                          # Source code
│   ├── config.py                 # Configuration settings
│   ├── data.py                   # Dataset classes
│   ├── model.py                  # Model architectures
│   ├── survival_model.py         # Survival prediction model
│   └── visualization.py          # Cancer highlighting & visualization
│
├── scripts/                      # Training scripts
│   ├── train.py                  # Train cancer classifier
│   ├── train_survival.py         # Train survival model
│   └── eval.py                   # Evaluation script
│
├── models/                       # Trained model weights
│   ├── efficientnet_b0_best.pt   # Cancer classifier
│   └── survival_model_best.pt    # Survival predictor
│
├── dataset/                      # Training data
│   ├── train/
│   ├── valid/
│   └── test/
│
├── sample_models/                # Reference implementations
│   ├── bone_cancer_highlight.py
│   ├── Bone_cancer_dots.py
│   └── model_2.py
│
├── requirements.txt              # Python dependencies
├── run_everything.bat            # All-in-one script
├── README.md                     # Project overview
├── HOW_TO_RUN.md                # This file
├── QUICK_START.md               # Quick reference
├── ENHANCED_ANALYSIS_GUIDE.md   # Feature documentation
└── CANCER_HIGHLIGHTING_FEATURE.md  # Highlighting guide
```

---

## 9. Quick Reference Commands

### Start Enhanced Server
```powershell
$env:PYTHONPATH="T:\bone_can_pre"
python app/server_enhanced.py
```

### Start Basic Server
```powershell
$env:PYTHONPATH="T:\bone_can_pre"
python app/server_survival.py
```

### Train Models
```powershell
$env:PYTHONPATH="T:\bone_can_pre"
python scripts/train.py --epochs 10 --batch-size 16
python scripts/train_survival.py --epochs 15 --batch-size 16
```

### Stop Server
Press `Ctrl+C` in the terminal

### Check Logs
Logs appear in the terminal where you started the server

---

## 10. Testing the Installation

### Quick Test
```powershell
# 1. Set environment
$env:PYTHONPATH="T:\bone_can_pre"

# 2. Start server
python app/server_enhanced.py

# 3. In browser, go to http://localhost:8000

# 4. Upload a test image from dataset/test/cancer/

# 5. Fill form and click Analyze

# 6. Verify results appear in 2-3 seconds
```

### Expected Results
- ✅ Server starts without errors
- ✅ Web page loads correctly
- ✅ Image upload works
- ✅ Analysis completes in 2-5 seconds
- ✅ Three cards display results
- ✅ Three images show visualizations
- ✅ No console errors

---

## 11. System Requirements

### Minimum Requirements
- **CPU**: Intel Core i5 or AMD Ryzen 5
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **GPU**: Not required (CPU mode available)
- **Internet**: For initial package installation

### Recommended Requirements
- **CPU**: Intel Core i7 or AMD Ryzen 7
- **RAM**: 16 GB
- **Storage**: 10 GB free space
- **GPU**: NVIDIA GPU with 4GB+ VRAM (GTX 1650 or better)
- **CUDA**: Version 11.8 or higher

### Performance
| Hardware | Prediction Time | Training Time (10 epochs) |
|----------|----------------|--------------------------|
| CPU only | 3-10 seconds   | 30-60 minutes            |
| GPU (4GB)| 1-3 seconds    | 10-20 minutes            |
| GPU (8GB+)| 0.5-2 seconds | 5-15 minutes             |

---

## 12. Next Steps

After successfully running the application:

1. **Read Documentation**
   - `ENHANCED_ANALYSIS_GUIDE.md` - Feature details
   - `CANCER_HIGHLIGHTING_FEATURE.md` - Visualization methods
   - `README.md` - Project overview

2. **Test with Sample Data**
   - Try images from `dataset/test/cancer/`
   - Try images from `dataset/test/normal/`
   - Compare different severity levels

3. **Customize Settings**
   - Adjust detection thresholds in `src/visualization.py`
   - Modify UI colors in `app/server_enhanced.py`
   - Change model parameters in `src/config.py`

4. **Train Your Own Models**
   - Prepare your dataset
   - Run training scripts
   - Evaluate performance

5. **Integrate with Your System**
   - Use API endpoints programmatically
   - Export results to database
   - Build custom frontend

---

## 13. Getting Help

### Documentation
- 📖 Full documentation in project `.md` files
- 💡 Code comments in all Python files
- 📚 Sample code in `sample_models/` directory

### Common Issues
Check the [Troubleshooting](#troubleshooting) section above

### Support
- Review error messages carefully
- Check Python and package versions
- Ensure all prerequisites are met
- Verify dataset structure

---

## ✅ Success Checklist

Before reporting issues, verify:

- [ ] Python 3.8+ installed
- [ ] All packages from `requirements.txt` installed
- [ ] `PYTHONPATH` environment variable set
- [ ] Model files exist in `models/` directory
- [ ] Server starts without errors
- [ ] Web page loads at http://localhost:8000
- [ ] Can upload images successfully
- [ ] Analysis completes and shows results

---

## 🎉 You're Ready!

If you've completed all steps successfully, your bone cancer prediction system is up and running!

**Access**: http://localhost:8000
**Documentation**: Check other `.md` files in the project root
**Questions**: Review the code and documentation

---

**Version**: 3.0  
**Last Updated**: November 2025  
**Platform**: Windows (PowerShell)  
**Python Version**: 3.8+
