import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import base64
from io import BytesIO

def create_cancer_highlight_image(original_image: np.ndarray, 
                                  tumor_mask: np.ndarray,
                                  save_path: Optional[str] = None) -> str:
    """
    Create a highlighted cancer area image with overlay
    
    Args:
        original_image: Original RGB image as numpy array
        tumor_mask: Binary mask of detected tumor areas
        save_path: Optional path to save the image
    
    Returns:
        Base64 encoded string of the highlighted image
    """
    # Convert to RGB if needed
    if len(original_image.shape) == 2:
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif original_image.shape[2] == 4:
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
    else:
        original_rgb = original_image.copy()
    
    # Create overlay
    overlay = original_rgb.copy()
    
    # Highlight tumor areas in red
    overlay[tumor_mask > 0] = [255, 0, 0]
    
    # Blend with original (70% original, 30% overlay)
    highlighted = cv2.addWeighted(original_rgb, 0.7, overlay, 0.3, 0)
    
    # Add contour around tumor areas
    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(highlighted, contours, -1, (255, 255, 255), 2)
    
    # Add text annotations
    tumor_area = np.sum(tumor_mask > 0)
    num_tumors = len(contours)
    
    cv2.putText(highlighted, f"Tumor Areas: {num_tumors}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(highlighted, f"Area: {tumor_area} pixels", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Convert to base64
    _, buffer = cv2.imencode('.png', highlighted)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, highlighted)
    
    return img_base64

def generate_analysis_report(tumor_mask: np.ndarray, 
                           original_image: np.ndarray,
                           cancer_probability: float,
                           stage: int) -> dict:
    """
    Generate detailed analysis report of cancer areas
    
    Args:
        tumor_mask: Binary mask of detected tumor areas
        original_image: Original image
        cancer_probability: Model prediction probability
        stage: Cancer stage (1, 2, or 3)
    
    Returns:
        Dictionary with analysis metrics
    """
    # Calculate tumor metrics
    tumor_area = np.sum(tumor_mask > 0)
    total_area = original_image.shape[0] * original_image.shape[1]
    tumor_percentage = (tumor_area / total_area) * 100
    
    # Count distinct tumor regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tumor_mask, connectivity=8)
    num_tumors = num_labels - 1
    
    # Calculate average tumor size
    if num_tumors > 0:
        avg_tumor_size = tumor_area / num_tumors
        largest_tumor_area = max(stats[1:, cv2.CC_STAT_AREA])  # Skip background
    else:
        avg_tumor_size = 0
        largest_tumor_area = 0
    
    # Risk assessment based on tumor characteristics
    risk_factors = []
    if tumor_percentage > 5:
        risk_factors.append("Large tumor coverage")
    if num_tumors > 3:
        risk_factors.append("Multiple tumor sites")
    if largest_tumor_area > 1000:
        risk_factors.append("Large dominant tumor")
    
    return {
        "tumor_area_pixels": int(tumor_area),
        "tumor_percentage": round(tumor_percentage, 2),
        "num_tumor_regions": int(num_tumors),
        "average_tumor_size": round(avg_tumor_size, 1),
        "largest_tumor_area": int(largest_tumor_area),
        "risk_factors": risk_factors,
        "stage": stage,
        "severity_assessment": get_severity_assessment(stage, tumor_percentage, num_tumors)
    }

def get_severity_assessment(stage: int, tumor_percentage: float, num_tumors: int) -> str:
    """Get human-readable severity assessment"""
    if stage == 1:
        if tumor_percentage < 1:
            return "Early stage with minimal tumor presence - Excellent prognosis"
        else:
            return "Early stage with moderate tumor presence - Good prognosis"
    elif stage == 2:
        if num_tumors <= 2:
            return "Moderate stage with localized tumors - Good prognosis with treatment"
        else:
            return "Moderate stage with multiple tumor sites - Requires aggressive treatment"
    else:  # stage 3
        if tumor_percentage > 10:
            return "Advanced stage with extensive tumor coverage - Poor prognosis"
        elif num_tumors > 5:
            return "Advanced stage with widespread tumor distribution - Critical condition"
        else:
            return "Advanced stage - Immediate intensive treatment required"

def create_multi_panel_visualization(original_image: np.ndarray,
                                    tumor_mask: np.ndarray,
                                    enhanced_image: np.ndarray,
                                    edges: np.ndarray) -> str:
    """
    Create a multi-panel visualization showing different analysis stages
    
    Returns:
        Base64 encoded string of the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original X-ray Image')
    axes[0, 0].axis('off')
    
    # Enhanced contrast
    axes[0, 1].imshow(enhanced_image, cmap='gray')
    axes[0, 1].set_title('Contrast Enhanced')
    axes[0, 1].axis('off')
    
    # Edge detection
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title('Bone Edge Detection')
    axes[1, 0].axis('off')
    
    # Final highlighted result
    overlay = original_image.copy()
    overlay[tumor_mask > 0] = [255, 0, 0]
    highlighted = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
    
    axes[1, 1].imshow(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Cancer Areas Highlighted (Red)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return img_base64