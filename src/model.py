import torch
import torch.nn as nn
from torchvision import models

_DEF_OUT = 2  # cancer vs normal


def build_model(name: str = "efficientnet_b0", num_classes: int = _DEF_OUT, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        return m
    elif name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v3_small(weights=weights)
        in_features = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unsupported model: {name}")
