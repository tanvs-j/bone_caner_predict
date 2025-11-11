import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, Optional


class GradCAM:
    """
    Grad-CAM implementation for visualizing what the model is looking at
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate class activation map for the input
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index. If None, uses predicted class
            
        Returns:
            heatmap: Normalized heatmap (H, W) in range [0, 1]
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().numpy()[0]  # (C, H, W)
        activations = self.activations.cpu().numpy()[0]  # (C, H, W)
        
        # Weight the activations by gradients
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Compute weighted sum
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def generate_visualization(self, 
                              input_image: np.ndarray, 
                              heatmap: np.ndarray,
                              alpha: float = 0.4) -> np.ndarray:
        """
        Overlay heatmap on original image
        
        Args:
            input_image: Original image (H, W, 3) in range [0, 255]
            heatmap: Normalized heatmap (H, W) in range [0, 1]
            alpha: Blending factor
            
        Returns:
            vis: Visualization with heatmap overlay (H, W, 3)
        """
        # Resize heatmap to match image size
        if heatmap.shape != input_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend
        vis = np.float32(input_image) / 255.0
        vis = vis + alpha * (np.float32(heatmap_colored) / 255.0)
        vis = vis / vis.max()
        vis = np.uint8(255 * vis)
        
        return vis


class TumorDetector:
    """
    Detects and highlights tumor regions using image processing
    """
    
    @staticmethod
    def detect_tumor_regions(heatmap: np.ndarray, 
                            threshold: float = 0.5) -> Tuple[np.ndarray, dict]:
        """
        Detect tumor regions from heatmap
        
        Args:
            heatmap: Normalized heatmap (H, W) in range [0, 1]
            threshold: Threshold for tumor detection
            
        Returns:
            mask: Binary mask of tumor regions
            stats: Dictionary with tumor statistics
        """
        # Threshold the heatmap
        binary = (heatmap > threshold).astype(np.uint8) * 255
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        num_labels, labels, stats_cv, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Analyze tumor regions
        tumor_regions = []
        total_area = 0
        
        for i in range(1, num_labels):  # Skip background
            area = stats_cv[i, cv2.CC_STAT_AREA]
            if area > 100:  # Minimum area threshold
                tumor_regions.append({
                    'area': area,
                    'centroid': centroids[i],
                    'bbox': stats_cv[i][:4]  # x, y, w, h
                })
                total_area += area
        
        stats = {
            'num_regions': len(tumor_regions),
            'total_area': total_area,
            'regions': tumor_regions,
            'severity': TumorDetector.estimate_severity(total_area, len(tumor_regions))
        }
        
        return binary, stats
    
    @staticmethod
    def estimate_severity(total_area: int, num_regions: int) -> dict:
        """
        Estimate tumor severity based on area and number of regions
        
        Returns:
            Dictionary with stage, severity level, and description
        """
        # Calculate severity score
        area_score = total_area / 1000  # Normalize area
        region_score = num_regions * 0.5
        severity_score = area_score + region_score
        
        if severity_score < 2:
            stage = 1
            level = "Low"
            description = "Early stage with minimal affected area"
        elif severity_score < 5:
            stage = 2
            level = "Moderate"
            description = "Intermediate stage with moderate affected area"
        else:
            stage = 3
            level = "High"
            description = "Advanced stage with significant affected area"
        
        return {
            'stage': stage,
            'level': level,
            'description': description,
            'score': round(severity_score, 2)
        }
    
    @staticmethod
    def draw_tumor_regions(image: np.ndarray, 
                          regions: list,
                          color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """
        Draw bounding boxes around detected tumor regions
        
        Args:
            image: RGB image
            regions: List of region dictionaries
            color: RGB color for boxes
            
        Returns:
            annotated_image: Image with drawn boxes
        """
        annotated = image.copy()
        
        for i, region in enumerate(regions):
            x, y, w, h = region['bbox']
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            
            # Add label
            label = f"Region {i+1}: {region['area']}px"
            cv2.putText(annotated, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated


def get_target_layer(model, model_name: str = "efficientnet_b0"):
    """
    Get the target layer for Grad-CAM based on model architecture
    
    Args:
        model: PyTorch model
        model_name: Name of the model architecture
        
    Returns:
        target_layer: Layer to use for Grad-CAM
    """
    if "efficientnet" in model_name.lower():
        # For EfficientNet, use the last convolutional layer
        return model.features[-1]
    elif "resnet" in model_name.lower():
        return model.layer4[-1]
    elif "mobilenet" in model_name.lower():
        return model.features[-1]
    else:
        # Try to find the last convolutional layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        raise ValueError(f"Could not find suitable layer for {model_name}")
