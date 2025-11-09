# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project purpose
- Binary image classifier for bone cancer vs. normal using PyTorch and TorchVision, with training/evaluation scripts and simple serving (FastAPI) and demo UI (Gradio).

Quickstart
- Python: 3.10+ recommended
- Install deps: `python -m pip install -r requirements.txt`
- Expected data layout (see README):
  - `dataset/`
    - `train/`, `valid/`, `test/` (images)
    - `train/_classes.csv` with columns: `filename, cancer, normal`

Common commands
- Train
  - `python scripts/train.py --epochs 10 --batch-size 16` (overrides in `src/config.py`)
- Evaluate
  - `python scripts/eval.py --split test --ckpt models/efficientnet_b0_best.pt`
  - If the requested split is missing, a random subset of train is used.
- Serve HTTP API (FastAPI via Uvicorn)
  - Set checkpoint, then run server:
    - POSIX (bash/zsh): `export BONE_CKPT=models/efficientnet_b0_best.pt && uvicorn app.server:app --host 0.0.0.0 --port 8000`
    - PowerShell: `$env:BONE_CKPT="models/efficientnet_b0_best.pt"; uvicorn app.server:app --host 0.0.0.0 --port 8000`
  - POST `/predict` with form `file=<image>`; returns JSON `{ cancer_probability, prediction, lifespan }`
- Demo UI (Gradio)
  - POSIX: `export BONE_CKPT=models/efficientnet_b0_best.pt && python app/ui.py`
  - PowerShell: `$env:BONE_CKPT="models/efficientnet_b0_best.pt"; python app/ui.py`
- “Tests” (no formal test suite configured)
  - Single request against the FastAPI app (uses TestClient): `python scripts/test_predict.py`
  - Sample requests pulled from training CSV: `python scripts/test_predict_from_csv.py`

Notes on tooling
- There is no build step, linter, or test framework configured in this repo. Use the above scripts for development and validation.

High-level architecture
- Configuration (`src/config.py`)
  - Centralized defaults for paths (dataset root, checkpoint dir), hyperparameters (img size, batch size, epochs, LR), model name, and DataLoader workers (Windows-friendly default `num_workers=0`). CLI flags in scripts can override these.
- Data layer (`src/data.py`)
  - `BoneCancerDataset` reads the labels CSV, normalizes column names, validates required columns (`filename`, `cancer`, `normal`), and filters rows to files that actually exist under the requested split directory (`train/`, `valid/`, or `test/`).
  - Augmentations/transforms via Albumentations: longest resize to square, pad, normalize to ImageNet stats, `ToTensorV2`. Optional train-time augmentations (flip, shift/scale/rotate, brightness/contrast).
  - Returns `(tensor, label_int, path)` per item.
- Model factory (`src/model.py`)
  - `build_model(name, num_classes=2, pretrained=True)` wraps TorchVision models and swaps the classifier head:
    - `efficientnet_b0` (default)
    - `mobilenet_v3_small`
- Training (`scripts/train.py`)
  - Optimizer: AdamW; LR schedule: CosineAnnealingLR. Loss: CrossEntropy.
  - Validation split preference: tries `valid/`; if missing or empty, creates a 90/10 random split from `train/`.
  - Metric: AUC computed from softmax probability of the cancer class. Best checkpoint saved to `models/{model_name}_best.pt` when AUC improves.
- Evaluation (`scripts/eval.py`)
  - Loads checkpoint, infers model type from checkpoint filename prefix, evaluates on `valid` or `test` (or a random subset of `train` if missing). Prints AUC and `classification_report`.
- Serving (`app/server.py`)
  - FastAPI app loads `BONE_CKPT` (falls back to a default path). Model type inferred from checkpoint filename. Endpoint `/predict` accepts an uploaded image and returns probability and label. Uses a 256px preprocessing pipeline (resize/pad/normalize) consistent with training normalization.
- Demo UI (`app/ui.py`)
  - Gradio Blocks app that loads the checkpoint (from `BONE_CKPT`) and exposes an image upload with predicted probabilities and label. “Lifespan” is currently a placeholder as survival modeling is not implemented.

Dataset and labels
- Assumes filenames are unique across splits; the dataset class filters CSV rows to each split based on actual files present.
- Binary target only: cancer vs. normal. “Stage” or survival/lifespan are not supported without additional labels; code paths and UI text call this out explicitly.
