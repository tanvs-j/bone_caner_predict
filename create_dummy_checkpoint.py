import torch
import torch.nn as nn
from src.model import build_model
import os

def create_dummy_checkpoint():
    """Create a dummy checkpoint for demonstration purposes"""
    
    # Create a simple model
    model = build_model("mobilenet_v3_small", num_classes=2, pretrained=False)
    
    # Create dummy checkpoint directory
    os.makedirs("models", exist_ok=True)
    
    # Save dummy checkpoint
    checkpoint_path = "models/mobilenet_v3_small_best.pt"
    torch.save({
        "model": model.state_dict(),
        "auc": 0.85,  # Dummy AUC
        "epoch": 1
    }, checkpoint_path)
    
    print(f"Created dummy checkpoint at: {checkpoint_path}")
    
    # Also create one for efficientnet_b0
    model2 = build_model("efficientnet_b0", num_classes=2, pretrained=False)
    checkpoint_path2 = "models/efficientnet_b0_best.pt"
    torch.save({
        "model": model2.state_dict(),
        "auc": 0.87,  # Dummy AUC
        "epoch": 1
    }, checkpoint_path2)
    
    print(f"Created dummy checkpoint at: {checkpoint_path2}")

if __name__ == "__main__":
    create_dummy_checkpoint()