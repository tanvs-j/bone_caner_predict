# Bone cancer classifier (deep learning)

- Train: `python scripts/train.py --epochs 10 --batch-size 16`
- Eval:  `python scripts/eval.py --split test --ckpt models/efficientnet_b0_best.pt`
- Serve: `set BONE_CKPT=models\efficientnet_b0_best.pt && uvicorn app.server:app --host 0.0.0.0 --port 8000`

Data layout expected:
```
T:\bone_can_pre\dataset\
  train\  (images)
  valid\  (images)
  test\   (images)
```
And the labels file `dataset\train\_classes.csv` with columns: `filename, cancer, normal`.

Notes:
- The dataset file names are unique across splits; the dataloader filters the CSV to files that actually exist in each split directory.
- Staging and lifespan prediction are not currently supported because there are no labels for them in `_classes.csv`. If you provide a CSV with `stage` (e.g., 0/1/2/3) or survival columns (`time,event`), I will extend the model to multi-class staging or survival analysis (DeepSurv) respectively.
