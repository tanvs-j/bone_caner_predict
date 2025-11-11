# Cancer Highlighting Feature

## Overview
The bone cancer prediction system now includes advanced cancer region highlighting capabilities, inspired by the sample models. When users upload X-ray images, the system automatically detects and highlights potential cancer regions in red.

## Features Added

### 1. Visualization Module (`src/visualization.py`)
A comprehensive module containing various image processing and tumor detection algorithms:

#### Key Functions:
- **`segment_bone_kmeans()`**: Uses K-means clustering to segment bone structures
- **`detect_tumor_area()`**: Detects tumor-like regions using adaptive thresholding
- **`enhance_contrast()`**: Enhances image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **`edge_detection()`**: Detects bone edges using Canny edge detector
- **`analyze_tumor_with_edges()`**: Advanced tumor analysis combining segmentation and edge detection
- **`highlight_cancer_region()`**: Main function that highlights cancer regions with red overlay
- **`create_gradcam_heatmap()`**: Generates Grad-CAM heatmaps for deep learning model visualization
- **`apply_heatmap_overlay()`**: Applies heatmap overlays on original images

### 2. Enhanced Server (`app/server_survival.py`)
Updated to generate and return highlighted images along with predictions:

#### New Response Fields:
- `original_image`: Base64-encoded original X-ray image
- `highlighted_image`: Base64-encoded image with cancer regions highlighted in red
- `tumor_analysis`: Dictionary containing:
  - `tumor_area`: Size of detected tumor region in pixels
  - `tumor_blobs`: Number of distinct tumor regions detected
  - `stage`: Cancer stage classification (Low/Moderate/High)
  - `method`: Analysis method used ('kmeans' or 'advanced')

### 3. Improved Web Interface
The web interface now displays:
- **Side-by-side comparison**: Original X-ray vs. Highlighted image
- **Tumor Analysis Panel**: Shows detailed tumor metrics including:
  - Stage classification
  - Tumor area measurement
  - Number of detected blobs
  - Analysis method used

## Technical Details

### Highlighting Methods

#### 1. K-means Method (`method='kmeans'`)
- Segments the image into 3 clusters using K-means
- Detects bright tumor regions using adaptive thresholding
- Applies red overlay with 40% transparency
- Fast and suitable for most cases

#### 2. Advanced Method (`method='advanced'`)
- Enhances contrast using CLAHE
- Detects bone edges with Canny edge detector
- Combines segmentation with edge information
- Counts individual tumor blobs
- Provides more detailed analysis
- Currently used as default in the server

### Stage Classification

Tumor stages are determined based on:
- **Tumor Area**: Total pixels of detected tumor regions
- **Tumor Blobs**: Number of distinct cancer regions
- **Scoring**: `score = tumor_blobs * 0.5 + (tumor_area / 2000)`

Stages:
- **Stage 1 (Low)**: score < 2 or area < 1000 pixels
- **Stage 2 (Moderate)**: 2 ≤ score < 6 or 1000 ≤ area < 4000 pixels  
- **Stage 3 (High)**: score ≥ 6 or area ≥ 4000 pixels

### Threshold for Highlighting
- Only highlights if cancer prediction probability ≥ 30%
- Below 30%, returns original image unchanged

## Usage

### Running the Server
```powershell
$env:PYTHONPATH="T:\bone_can_pre"
python app/server_survival.py
```

Access at: http://localhost:8000

### API Usage
The `/predict_survival` endpoint now returns additional fields:

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
  },
  "original_image": "base64_encoded_string",
  "highlighted_image": "base64_encoded_string",
  "tumor_analysis": {
    "tumor_area": 5234,
    "tumor_blobs": 3,
    "stage": "Moderate (Stage 2)",
    "method": "advanced"
  }
}
```

## Reference Implementations

The highlighting features are based on techniques from:
- `sample_models/bone_cancer_highlight.py`: K-means segmentation and tumor detection
- `sample_models/Bone_cancer_dots.py`: Advanced edge detection and blob analysis
- `sample_models/model_2.py`: Grad-CAM visualization concepts

## Future Enhancements

Potential improvements:
- [ ] Add multiple highlighting methods selection in UI
- [ ] Implement Grad-CAM integration with the trained models
- [ ] Add adjustable sensitivity controls
- [ ] Export highlighted images as downloadable files
- [ ] Overlay tumor measurements and annotations on images
- [ ] 3D visualization for better depth perception
- [ ] Confidence heatmaps showing probability distributions

## Performance Considerations

- Image processing adds ~0.5-1.5 seconds to prediction time
- Base64 encoding increases response size (~2-4x image size)
- Consider implementing caching for repeated predictions
- For production, consider using CDN for image delivery

## Dependencies

All required packages are already in `requirements.txt`:
- OpenCV (cv2): Image processing
- NumPy: Numerical operations
- PyTorch: Deep learning model support
- PIL/Pillow: Image handling
- Albumentations: Image augmentation

## Notes

- The highlighting is based on image analysis heuristics and model predictions
- Red highlighting indicates potential cancer regions but should not be used as sole diagnostic criterion
- Medical professionals should always verify findings
- The system is for educational and research purposes
