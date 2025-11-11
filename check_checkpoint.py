import torch

ckpt_path = r"T:\bone_can_pre\models\efficientnet_b0_best.pt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

print(f"Checkpoint: {ckpt_path}")
print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"AUC: {ckpt.get('auc', 'N/A'):.4f}" if 'auc' in ckpt else "AUC: N/A")
print(f"Keys in checkpoint: {list(ckpt.keys())}")
