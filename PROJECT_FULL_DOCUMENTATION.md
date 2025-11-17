# Bone Cancer Prediction & Survival Analysis – Full Project Documentation

> Repository: `tanvs-j/bone_caner_predict`

This document provides a **single, comprehensive overview** of the bone cancer prediction project: what it does, how it is structured, how to run it, how the models are trained and evaluated, and how the web interfaces work (including the per-session accuracy/precision bar graph).

---

## 1. Project Goal & High-Level Features

- **Primary goal**: Assist clinicians and researchers by
  - Detecting **bone cancer** from X-ray images (binary: normal vs cancer).
  - Highlighting suspicious tumor regions on the image.
  - Estimating **survival-related information** using a multi-task survival model.
- **Key capabilities**:
  - Deep learning–based image classification using EfficientNet/MobileNet.
  - Multi-task survival model combining image features + clinical features.
  - Web interfaces:
    - **Enhanced FastAPI UI** (main app – port `8000`).
    - **Basic survival server** (simpler UI – port `8000` with alternate script).
    - **Gradio UI** (`ui.py` – port `7860`).
  - Visual analysis:
    - Heatmap-style highlight.
    - Bounding boxes around suspicious regions.
    - Tumor severity & stage style outputs (via visualization utilities or sample models).
  - **Per-session accuracy/precision bar graph** in the enhanced web UI.

---

## 2. Repository Structure

At a high level:

```text
bone_can_pre/
├── app/                      # Web apps & UIs
│   ├── server_enhanced.py    # Enhanced FastAPI server (recommended)
│   ├── server_survival.py    # Basic survival server
│   ├── server.py             # Simple cancer detection server
│   └── ui.py                 # Gradio interface (port 7860)
│
├── src/                      # Core library code
│   ├── config.py             # Global configuration (paths, hyperparameters)
│   ├── data.py               # Dataset & transforms (BoneCancerDataset)
│   ├── model.py              # CNN model builders (EfficientNet/MobileNet)
│   ├── survival_model.py     # Multi-task survival predictor + encoders
│   └── visualization.py      # Tumor highlighting & advanced visualization
│
├── scripts/                  # Training & evaluation scripts
│   ├── train.py              # Train image classifier
│   ├── train_survival.py     # Train survival prediction model
│   ├── eval.py               # Evaluate classifier checkpoints
│   └── ...                   # Additional utilities (e.g., folder-based training)
│
├── sample_models/            # Classical ML & visualization prototypes
│   ├── bone_cancer_highlight.py
│   ├── Bone_cancer_dots.py
│   └── model_2.py
│
├── dataset/                  # Dataset root (train/valid/test folders)
│   ├── train/
│   ├── valid/
│   └── test/
│
├── models/                   # Saved model checkpoints
│   ├── efficientnet_b0_best.pt
│   ├── mobilenet_v3_small_best.pt
│   └── survival_model_best.pt
│
├── *.md                      # Documentation (README, HOW_TO_RUN, guides)
├── requirements.txt          # Python dependencies
└── run_everything.bat        # Train models and start server (all-in-one)
```

---

## 3. Data & Dataset Format

### 3.1 Directory layout

The project expects:

```text
dataset/
├── train/
│   ├── normal/
│   ├── cancer/
│   └── _classes.csv
├── valid/
│   ├── normal/
│   ├── cancer/
│   └── _classes.csv
└── test/
    ├── normal/
    └── cancer/
```

- Image files (e.g., `.jpg`, `.png`) are placed into `normal/` or `cancer/` per split.
- `_classes.csv` contains: `filename`, `cancer`, `normal` (and possibly other columns).
  - `cancer` is typically `1` for cancer images, `0` for normal.

### 3.2 BoneCancerDataset (in `src/data.py`)

- Responsible for:
  - Reading the `_classes.csv`.
  - Matching entries to actual image files in the respective split directory.
  - Returning `(image_tensor, label_int, path)` where `label_int` is `1` (cancer) or `0` (normal).
- Uses **Albumentations** for transforms:
  - Resize with `LongestMaxSize`.
  - Padding to a fixed rectangle.
  - Normalization with ImageNet-like statistics.
  - Optional data augmentation for training: flips, shifts, brightness/contrast, etc.

---

## 4. Models

### 4.1 Classification backbone (in `src/model.py`)

- Supports at least:
  - `efficientnet_b0`
  - `mobilenet_v3_small`
- `build_model(name, num_classes, pretrained)` builds a classifier with a 2-class output (`normal`, `cancer`).
- Checkpoints stored in `models/`:
  - `efficientnet_b0_best.pt`
  - `mobilenet_v3_small_best.pt`

### 4.2 Survival model (in `src/survival_model.py`)

- `SurvivalPredictor` is a **multi-task network**:
  - Inputs:
    - Image feature vector from a CNN backbone.
    - Encoded clinical features.
  - Outputs:
    - `cancer_logits` (cancer vs normal).
    - `survival_logits` (NED, AWD, Dead).
    - `risk_score` (continuous risk measure).

- `ClinicalFeatureEncoder` maps clinical fields to a fixed-size numeric vector:
  - Encodes
    - Sex.
    - Age (normalized).
    - Tumor Grade.
    - Treatments (Surgery/Chemo/Radiotherapy) as one-hot.
    - Simplified histological type (aggressive vs not).

- `estimate_survival_months(status, risk_score, age, grade)`:
  - Uses heuristics and model outputs to estimate:
    - Expected survival in months.
    - Confidence interval (lower/upper bound).
    - Approximate years.

---

## 5. Training & Evaluation Scripts

### 5.1 Image classifier training – `scripts/train.py`

- Uses `BoneCancerDataset` with train/valid splits.
- Key steps:
  - Build model (EfficientNet/MobileNet) from `src.model`.
  - Train with cross-entropy loss.
  - Track validation AUC (ROC-AUC).
  - Save best checkpoint to `models/<model_name>_best.pt`.
- Important arguments (also defaulted from `Config`):
  - `--epochs`
  - `--batch-size`
  - `--img-size`
  - `--lr`
  - `--weight-decay`
  - `--model` (e.g., `efficientnet_b0`)

### 5.2 Survival model training – `scripts/train_survival.py`

- Trains the multi-task survival predictor on **image + clinical features**.
- Uses `SurvivalPredictor` and `ClinicalFeatureEncoder`.
- Produces `models/survival_model_best.pt`.

### 5.3 Evaluation script – `scripts/eval.py`

- Evaluates a given checkpoint on `valid` or `test` split.
- Workflow:
  - Load dataset using `BoneCancerDataset`.
  - Build model (EfficientNet/MobileNet) matching the checkpoint.
  - Compute:
    - ROC-AUC.
    - Classification report.
    - Accuracy and precision (added in this version).
  - Shows a **matplotlib bar chart** of Accuracy vs Precision when run as a script.

Usage example:

```powershell
$env:PYTHONPATH="T:\bone_can_pre"
python scripts/eval.py --ckpt models\efficientnet_b0_best.pt --split test
```

---

## 6. Web Interfaces

### 6.1 Enhanced FastAPI server – `app/server_enhanced.py`

**Purpose**: Main, feature-rich UI with:

- Cancer detection card.
- Tumor analysis details.
- Survival prediction card.
- 3-image visual analysis panel:
  - Original X-ray.
  - Heatmap/contrast highlights.
  - Bounding-box findings.
- **Per-session accuracy/precision bar graph**.

#### 6.1.1 Startup

```powershell
$env:PYTHONPATH="T:\bone_can_pre"
python app/server_enhanced.py
```

- Loads survival model from `SURVIVAL_MODEL_PATH` (or falls back to cancer-only model).
- Prepares transforms and clinical encoder.

Access in browser:

```text
http://localhost:8000
```

#### 6.1.2 Prediction flow

- Endpoint: `POST /predict`
- Input: uploaded image file (`file` field); internal default clinical values are used.
- Steps:
  1. Image is preprocessed with Albumentations.
  2. Survival model (if available) or cancer-only model predicts:
     - Cancer probability.
     - Cancer vs normal label.
     - Survival status (NED, AWD, Dead).
     - Risk score.
     - Estimated survival time and interval.
  3. `highlight_cancer_region` produces:
     - Heatmap image.
     - Bounding box image.
     - Overlay image.
     - Tumor analysis summary.
  4. Returned as JSON with base64-encoded images.

#### 6.1.3 Per-session accuracy & precision bar graph

- In this version, `server_enhanced.py` tracks per-session metrics:
  - `SESSION_Y_TRUE`: list of true labels for this server run.
  - `SESSION_Y_PRED`: list of model predictions.
- In the web form:
  - You can optionally choose **Actual Label** (`Normal` or `Cancer`).
  - This value is sent as `true_label` in the form.
- On each `/predict`:
  - If `true_label` is provided:
    - The server appends the new `(y_true, y_pred)` to the session lists.
    - Recomputes **accuracy** and **precision** on all labeled samples so far.
  - JSON response includes:

    ```json
    "metrics": {
      "accuracy": <session_accuracy_or_null>,
      "precision": <session_precision_or_null>,
      "num_samples": <number_of_labeled_samples>
    }
    ```

- Frontend behavior:
  - If metrics are available, a **vertical bar chart** appears:
    - One bar for Accuracy.
    - One bar for Precision.
  - The bar heights animate from 0 to the percentage value.
  - Bars update each time you run a labeled prediction during the same session.

> Note: restarting the server resets these session metrics.

### 6.2 Basic survival server – `app/server_survival.py`

- A simpler FastAPI server focused mainly on survival prediction with fewer UI elements.
- Can be started similarly to the enhanced server.

### 6.3 Simple detection server – `app/server.py`

- Minimal server for basic cancer detection without the full enhanced UI.

### 6.4 Gradio UI – `app/ui.py`

**Purpose**: Quick, standalone interface on port `7860`:

- Upload X-ray image.
- See class probabilities and a textual prediction.

Run it with:

```powershell
$env:PYTHONPATH="T:\bone_can_pre"
python app/ui.py
```

Access:

```text
http://localhost:7860
```

- Uses a preconfigured CNN checkpoint (by default `mobilenet_v3_small_best.pt` or EfficientNet, depending on environment variable `BONE_CKPT`).
- Returns a label dictionary `{"normal": p0, "cancer": p1}` and a prediction text.

---

## 7. Sample Models & Classical Pipelines

The `sample_models/` folder contains standalone experiments and demos:

- `bone_cancer_highlight.py`:
  - K-Means segmentation.
  - Feature extraction.
  - KNN classifier.
  - Tumor area detection and highlighting.

- `Bone_cancer_dots.py`:
  - Multi-step image processing: contrast enhancement, edge detection, segmentation.
  - Blob/dot-based tumor analysis.
  - KNN classification and stage/lifespan-like outputs.

- These scripts are **not required** for running the main web apps but are useful for understanding traditional ML approaches used during prototyping.

---

## 8. Installation & Environment

### 8.1 Requirements

- **Python**: 3.8+
- **OS**: Windows (project is tested with PowerShell commands; Linux/Mac possible with minor tweaks).

### 8.2 Install dependencies

From the project root:

```powershell
pip install -r requirements.txt
```

Key packages:

- `torch`, `torchvision` – DL framework.
- `fastapi`, `uvicorn` – Web APIs.
- `opencv-python`, `albumentations` – Image processing and augmentation.
- `pandas`, `numpy`, `scikit-learn` – Data & metrics.
- `Pillow` – Image handling.
- `gradio` – Gradio UI.

---

## 9. Running the System

### 9.1 Enhanced web server (recommended)

```powershell
$env:PYTHONPATH="T:\bone_can_pre"
python app/server_enhanced.py
```

Then open:

```text
http://localhost:8000
```

### 9.2 Basic survival server

```powershell
$env:PYTHONPATH="T:\bone_can_pre"
python app/server_survival.py
```

### 9.3 Gradio interface

```powershell
$env:PYTHONPATH="T:\bone_can_pre"
python app/ui.py
```

Access:

```text
http://localhost:7860
```

### 9.4 All-in-one script

```powershell
.\run_everything.bat
```

- Trains the main models and starts the server automatically.
- Time-consuming (can take 10–30 minutes depending on hardware).

---

## 10. Logging, Errors & Troubleshooting

### 10.1 Common issues

- **Port already in use (8000)**:
  - Symptom: `only one usage of each socket address...`.
  - Fix: stop the other process or run server on a different port (for FastAPI) or terminate previous Uvicorn process.

- **Module `src` not found**:
  - Symptom: `ModuleNotFoundError: No module named 'src'`.
  - Fix: ensure `PYTHONPATH` includes the project root:

    ```powershell
    $env:PYTHONPATH="T:\bone_can_pre"
    ```

- **Missing model checkpoint**:
  - Symptom: `FileNotFoundError: models/survival_model_best.pt` or similar.
  - Fix: train the model (`scripts/train.py`, `scripts/train_survival.py`) or copy checkpoints into `models/`.

- **CUDA out of memory**:
  - Fix: run on CPU only or reduce batch size in training scripts.

### 10.2 Debugging evaluation or metrics

- If evaluation scripts crash, check:
  - Paths to `dataset_root` and CSV files.
  - That `_classes.csv` columns include `filename`, `cancer`, `normal`.

- If the **bar graph** does not show:
  - Ensure you provided **Actual Label** in the form (enhanced UI).
  - Ensure at least one labeled sample has been processed this session.

---

## 11. Extending the Project

- **New backbones**: Add more architectures in `src/model.py` and configure them via `Config.model_name` and `scripts/train.py`.
- **More classes**: Extend the classifier from 2-class to multi-class (e.g., multiple tumor types) by adjusting dataset labels and model output layer.
- **Richer clinical features**: Extend `ClinicalFeatureEncoder` to include more fields.
- **Deployment**: Package `app/server_enhanced.py` behind a production ASGI server (e.g., gunicorn + uvicorn workers) and then front with a web server or container.

---

## 12. Summary

This project provides an end-to-end pipeline for **bone cancer detection and survival analysis**:

- Data ingestion and preprocessing for normal vs cancer X-ray images.
- Deep learning–based classification and multi-task survival modeling.
- Multiple web interfaces for interactive use.
- Visual explanation tools and per-session accuracy/precision bar graphs to help users understand model behavior.

Use this file together with:

- `README.md` – Project overview.
- `HOW_TO_RUN.md` – Step-by-step setup and run instructions.
- `ENHANCED_ANALYSIS_GUIDE.md` / `CANCER_HIGHLIGHTING_FEATURE.md` – Detailed analysis and visualization docs.

for a complete understanding of how everything works and how to modify or extend it.
