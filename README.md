# Bone Cancer Prediction & Survival Analysis

A deep learning system for bone cancer detection and patient survival prediction using X-ray images and clinical data.

## 🎯 Features

- **Cancer Detection**: Binary classification (cancer vs normal) from X-ray images using EfficientNet-B0
- **Survival Prediction**: Multi-task model predicting survival status (NED/AWD/Dead)
- **Lifespan Estimation**: Estimates patient survival time in months/years with confidence intervals
- **Clinical Integration**: Incorporates patient age, sex, tumor grade, treatment type, and histology
- **Web Interface**: User-friendly FastAPI web application with real-time predictions

## 📊 Model Architecture

### 1. Cancer Classification Model
- **Architecture**: EfficientNet-B0 (pretrained on ImageNet)
- **Input**: 384x384 RGB X-ray images
- **Output**: Binary classification (cancer/normal)
- **Performance**: ~97% AUC on validation set

### 2. Survival Prediction Model
- **Architecture**: Multi-task CNN with clinical feature fusion
- **Inputs**:
  - X-ray images (extracted features from EfficientNet-B0)
  - Clinical features: age, sex, grade, treatment, histological type
- **Outputs**:
  - Cancer classification (2 classes)
  - Survival status (3 classes: NED, AWD, Dead)
  - Risk score for survival estimation
- **Performance**: 36.6% F1 score on survival prediction

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/tanvs-j/bone_caner_predict.git
cd bone_caner_predict

# Install dependencies
pip install -r requirements.txt
```

### Training

#### 1. Train Cancer Classification Model
```bash
set PYTHONPATH=%CD%
python scripts/train.py --epochs 10 --batch-size 16
```

#### 2. Train Survival Prediction Model
```bash
set PYTHONPATH=%CD%
python scripts/train_survival.py --epochs 15 --batch-size 16
```

### Running the Application

#### Option 1: FastAPI Server (with Survival Prediction)
```bash
set PYTHONPATH=%CD%
python app/server_survival.py
```
Access at: http://localhost:8000

#### Option 2: Gradio UI
```bash
set PYTHONPATH=%CD%
python app/ui.py
```
Access at: http://localhost:7860

#### Option 3: Run Everything (Train + Deploy)
```bash
run_everything.bat
```

## 📁 Project Structure

```
bone_can_pre/
├── app/
│   ├── server.py              # Basic FastAPI server (cancer only)
│   ├── server_survival.py     # Full survival prediction server
│   └── ui.py                  # Gradio interface
├── dataset/
│   ├── train/
│   │   ├── _classes.csv       # Training image labels
│   │   └── Bone Tumor Dataset.csv  # Clinical survival data
│   └── valid/
│       └── _classes.csv       # Validation image labels
├── models/
│   ├── efficientnet_b0_best.pt       # Cancer classification weights
│   └── survival_model_best.pt        # Survival prediction weights
├── scripts/
│   ├── train.py               # Train cancer classifier
│   ├── train_survival.py      # Train survival model
│   ├── eval.py                # Evaluation script
│   └── create_valid_labels_v2.py  # Generate validation labels
├── src/
│   ├── config.py              # Configuration settings
│   ├── data.py                # Dataset classes
│   ├── model.py               # Model architectures
│   └── survival_model.py      # Survival prediction model
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 📝 Data Format

### Image Labels CSV
```csv
filename, cancer, normal
image1.jpg, 1, 0
image2.jpg, 0, 1
```

### Clinical Data CSV
Required columns:
- `Patient ID`: Unique identifier
- `Sex`: Male/Female
- `Age`: Patient age in years
- `Grade`: Low/Intermediate/High
- `Histological type`: Tumor type
- `MSKCC type`: Memorial Sloan Kettering Cancer Center classification
- `Site of primary STS`: Primary tumor site
- `Status (NED, AWD, D)`: Survival status
- `Treatment`: Treatment regimen

## 🔬 Model Training Details

### Cancer Classification
- **Optimizer**: AdamW (lr=2e-4, weight_decay=1e-4)
- **Scheduler**: Cosine Annealing
- **Loss**: Cross Entropy
- **Augmentation**: Horizontal flip, rotation, brightness/contrast adjustment

### Survival Prediction
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Multi-task Loss**: Cancer loss + 2.0 × Survival loss
- **Clinical Features**: 7-dimensional encoded vector
- **Risk Estimation**: Tanh-activated risk score

## 🌐 API Endpoints

### POST `/predict_survival`
Predicts cancer status and survival time.

**Request**:
- `file`: X-ray image (multipart/form-data)
- `sex`: Male/Female
- `age`: Integer
- `grade`: Low/Intermediate/High
- `treatment`: Treatment types (e.g., "Surgery + Chemotherapy")
- `histological_type`: Tumor histology

**Response**:
```json
{
  "cancer_prediction": "cancer",
  "cancer_probability": 0.87,
  "survival_status": "AWD (Alive with Disease)",
  "risk_score": 0.234,
  "estimated_survival": {
    "estimated_months": 48,
    "estimated_years": 4.0,
    "lower_bound": 38,
    "upper_bound": 58
  }
}
```

## 📊 Dataset Information

- **Training Images**: ~7,000 X-ray images
- **Validation Images**: ~880 X-ray images
- **Clinical Records**: 500 patient records with survival data
- **Image Format**: JPG/PNG, various sizes (automatically resized)
- **Classes**: Cancer (osteosarcoma, ewing sarcoma, etc.) vs Normal bone

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- FastAPI
- Gradio
- albumentations
- OpenCV
- pandas
- scikit-learn
- numpy

See `requirements.txt` for complete list.

## 📈 Performance Metrics

| Model | Metric | Value |
|-------|--------|-------|
| Cancer Classifier | AUC | 97.2% |
| Cancer Classifier | Validation Loss | 0.26 |
| Survival Predictor | F1 Score | 36.6% |
| Survival Predictor | Cancer Accuracy | 54.7% |
| Survival Predictor | Survival Accuracy | 45.3% |

## 🔮 Future Improvements

- [ ] Add time-to-event survival analysis (Cox proportional hazards)
- [ ] Implement attention mechanisms for better interpretability
- [ ] Add tumor segmentation capabilities
- [ ] Expand dataset with more diverse patient populations
- [ ] Deploy to cloud (AWS/Azure/GCP)
- [ ] Add DICOM support for medical imaging standards

## 📄 License

This project is for educational and research purposes.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- TCGA-SARC dataset for clinical data
- Bone cancer image datasets from public repositories
- PyTorch and torchvision teams
- FastAPI and Gradio communities

---

**Note**: Model weights (`.pt` files) are not included in the repository due to size constraints. Train the models using the provided scripts to generate them.
