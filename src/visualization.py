import cv2
import numpy as np
from typing import Tuple, Optional
import torch
from PIL import Image


def segment_bone_kmeans(img_rgb: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Segment bone image using K-means clustering.
    
    Args:
        img_rgb: RGB image as numpy array
        k: Number of clusters for K-means
        
    Returns:
        Segmented image
    """
    pixel_values = img_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(img_rgb.shape)
    return segmented_image


def detect_tumor_area(segmented_img: np.ndarray, method: str = 'adaptive') -> Tuple[np.ndarray, int]:
    """
    Detect tumor-like regions in segmented image.
    
    Args:
        segmented_img: Segmented RGB image
        method: Detection method - 'adaptive' or 'threshold'
        
    Returns:
        Tuple of (tumor_mask, tumor_area)
    """
    gray = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if method == 'adaptive':
        # Adaptive threshold - better for variable lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 35, 2
        )
        # Invert so bright regions are white
        binary = cv2.bitwise_not(binary)
    else:
        # Simple threshold for bright tumor regions
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    tumor_area = np.sum(mask > 0)
    return mask, tumor_area


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using CLAHE.
    
    Args:
        img: Input image (BGR or grayscale)
        
    Returns:
        Contrast-enhanced grayscale image
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def edge_detection(img: np.ndarray, low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray:
    """
    Detect edges using Canny edge detector.
    
    Args:
        img: Grayscale image
        low_threshold: Lower threshold for Canny
        high_threshold: Upper threshold for Canny
        
    Returns:
        Edge map
    """
    return cv2.Canny(img, low_threshold, high_threshold)


def analyze_tumor_with_edges(segmented_img: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Analyze tumor using both segmentation and edge information.
    
    Args:
        segmented_img: Segmented RGB image
        edges: Edge map from Canny detection
        
    Returns:
        Tuple of (tumor_mask, num_blobs, tumor_area)
    """
    gray = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 2
    )
    binary = cv2.bitwise_not(binary)
    
    # Use edges to mask out background
    edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    bone_mask = cv2.bitwise_not(edges_dilated)
    tumor_mask = cv2.bitwise_and(binary, bone_mask)
    
    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel)
    
    # Count connected components (blobs)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tumor_mask, connectivity=8)
    tumor_blobs = num_labels - 1  # subtract background
    tumor_area = np.sum(tumor_mask > 0)
    
    return tumor_mask, tumor_blobs, tumor_area


def highlight_cancer_region(original_img: np.ndarray, prediction_prob: float, 
                           method: str = 'advanced') -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Highlight potential cancer regions in the image with three visualization styles.
    
    Args:
        original_img: Original RGB image as numpy array
        prediction_prob: Cancer prediction probability from model
        method: Highlighting method - 'kmeans' or 'advanced'
        
    Returns:
        Tuple of (heatmap_img, bbox_img, overlay_img, analysis_info)
    """
    # Only highlight if cancer probability is significant
    if prediction_prob < 0.3:
        return original_img, original_img, original_img, {
            'tumor_area': 0,
            'detected_regions': 0,
            'stage': 'Normal',
            'severity': 'Stage 1 - Low',
            'method': method,
            'bounding_boxes': []
        }
    
    # Enhanced method with edge detection
    img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    enhanced = enhance_contrast(img_bgr)
    edges = edge_detection(enhanced)
    segmented = segment_bone_kmeans(original_img, k=3)
    
    mask, tumor_blobs, tumor_area = analyze_tumor_with_edges(segmented, edges)
    
    # Detect bounding boxes
    boxes, num_regions = detect_tumor_regions_with_boxes(mask, min_area=300)
    
    # 1. Create heatmap visualization (like middle image in your screenshot)
    heatmap_img = create_heatmap_visualization(original_img, mask)
    
    # 2. Create bounding box visualization (like right image in your screenshot)
    bbox_img = draw_bounding_boxes(original_img, boxes, color=(255, 0, 0), thickness=3)
    
    # 3. Create simple overlay visualization
    highlight = original_img.copy()
    highlight[mask > 0] = [255, 0, 0]  # Red overlay
    overlay_img = cv2.addWeighted(original_img, 0.7, highlight, 0.3, 0)
    
    # Determine stage
    score = num_regions * 0.5 + (tumor_area / 2000)
    if score < 2:
        stage_num = 1
        severity = "Stage 1 - Low"
    elif score < 6:
        stage_num = 2
        severity = "Stage 2 - Moderate"
    else:
        stage_num = 3
        severity = "Stage 3 - High"
    
    return heatmap_img, bbox_img, overlay_img, {
        'tumor_area': int(tumor_area),
        'detected_regions': num_regions,
        'stage': stage_num,
        'severity': severity,
        'method': method,
        'bounding_boxes': boxes
    }


def determine_stage_from_area(tumor_area: int) -> str:
    """
    Determine cancer stage based on tumor area.
    
    Args:
        tumor_area: Tumor area in pixels
        
    Returns:
        Stage description
    """
    if tumor_area < 1000:
        return "Low (Stage 1)"
    elif tumor_area < 4000:
        return "Moderate (Stage 2)"
    else:
        return "High (Stage 3)"


def create_gradcam_heatmap(model: torch.nn.Module, img_tensor: torch.Tensor, 
                          target_layer: torch.nn.Module, device: str = 'cpu') -> Optional[np.ndarray]:
    """
    Generate Grad-CAM heatmap for cancer localization.
    
    Args:
        model: PyTorch model
        img_tensor: Input image tensor (1, C, H, W)
        target_layer: Target layer for Grad-CAM
        device: Device to run on
        
    Returns:
        Heatmap as numpy array or None if failed
    """
    try:
        model.eval()
        
        # Storage for gradients and activations
        gradients = []
        activations = []
        
        def save_gradient(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        def save_activation(module, input, output):
            activations.append(output)
        
        # Register hooks
        handle_grad = target_layer.register_backward_hook(save_gradient)
        handle_fwd = target_layer.register_forward_hook(save_activation)
        
        # Forward pass
        output = model(img_tensor)
        if isinstance(output, dict):
            output = output.get('cancer_logits', output.get('logits', None))
        
        if output is None:
            return None
        
        # Get the class with highest score
        class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()
        
        # Calculate Grad-CAM
        grads = gradients[0].cpu().data.numpy()[0]
        acts = activations[0].cpu().data.numpy()[0]
        
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * acts[i]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (img_tensor.shape[3], img_tensor.shape[2]))
        cam = cam - np.min(cam)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        # Remove hooks
        handle_grad.remove()
        handle_fwd.remove()
        
        return cam
        
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        return None


def apply_heatmap_overlay(original_img: np.ndarray, heatmap: np.ndarray, 
                         colormap: int = cv2.COLORMAP_JET, alpha: float = 0.4) -> np.ndarray:
    """
    Apply heatmap overlay on original image.
    
    Args:
        original_img: Original RGB image
        heatmap: Normalized heatmap (0-1)
        colormap: OpenCV colormap to use
        alpha: Transparency factor
        
    Returns:
        Image with heatmap overlay
    """
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if heatmap_colored.shape[:2] != original_img.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (original_img.shape[1], original_img.shape[0]))
    
    # Blend images
    result = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    return result


def detect_tumor_regions_with_boxes(mask: np.ndarray, min_area: int = 500) -> Tuple[list, int]:
    """
    Detect tumor regions and return bounding boxes.
    
    Args:
        mask: Binary mask of tumor regions
        min_area: Minimum area to consider as a valid region
        
    Returns:
        Tuple of (list of bounding boxes, total regions count)
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append({
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'area': int(area)
            })
    
    return boxes, len(boxes)


def draw_bounding_boxes(img: np.ndarray, boxes: list, color: Tuple[int, int, int] = (255, 0, 0), 
                       thickness: int = 2, show_labels: bool = True) -> np.ndarray:
    """
    Draw bounding boxes on image with labels.
    
    Args:
        img: RGB image
        boxes: List of bounding box dictionaries
        color: Box color (RGB)
        thickness: Line thickness
        show_labels: Whether to show region labels
        
    Returns:
        Image with bounding boxes drawn
    """
    result = img.copy()
    
    for idx, box in enumerate(boxes, 1):
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        area = box['area']
        
        # Draw rectangle
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        
        if show_labels:
            # Draw label background
            label = f"Region {idx}: {area}px"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            label_thickness = 1
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, label_thickness)
            
            # Draw filled rectangle for text background
            cv2.rectangle(result, (x, y - label_h - 10), (x + label_w + 10, y), color, -1)
            
            # Draw text
            cv2.putText(result, label, (x + 5, y - 5), font, font_scale, (255, 255, 255), label_thickness)
    
    return result


def create_heatmap_visualization(original_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create a heatmap-style visualization of tumor regions.
    
    Args:
        original_img: Original RGB image
        mask: Binary tumor mask
        
    Returns:
        Heatmap visualization
    """
    # Create a smooth heatmap from the binary mask
    mask_float = mask.astype(np.float32) / 255.0
    
    # Apply Gaussian blur for smooth transitions
    heatmap = cv2.GaussianBlur(mask_float, (21, 21), 0)
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original
    result = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
    
    return result
