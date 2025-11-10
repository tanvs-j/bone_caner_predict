import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class SurvivalPredictor(nn.Module):
    """
    Multi-task model for bone cancer:
    1. Cancer classification (cancer vs normal)
    2. Survival status prediction (NED, AWD, Dead)
    3. Risk score estimation
    """
    def __init__(self, feature_extractor: nn.Module, num_clinical_features: int = 0):
        super().__init__()
        self.feature_extractor = feature_extractor
        
        # Get feature dimension from the extractor
        # For EfficientNet-B0, it's 1280; for MobileNetV3-Small, it's 576
        self.feature_dim = self._get_feature_dim()
        
        # Combined feature dimension
        combined_dim = self.feature_dim + num_clinical_features
        
        # Task heads
        self.cancer_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # cancer vs normal
        )
        
        self.survival_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # NED, AWD, Dead
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # risk score for survival prediction
        )
        
    def _get_feature_dim(self):
        """Get the output dimension of the feature extractor"""
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.feature_extractor(dummy_input)
        return features.shape[1]
    
    def forward(self, images, clinical_features=None):
        # Extract image features
        img_features = self.feature_extractor(images)
        
        # Combine with clinical features if available
        if clinical_features is not None:
            features = torch.cat([img_features, clinical_features], dim=1)
        else:
            features = img_features
        
        # Multi-task predictions
        cancer_logits = self.cancer_head(features)
        survival_logits = self.survival_head(features)
        risk_score = self.risk_head(features)
        
        return {
            'cancer_logits': cancer_logits,
            'survival_logits': survival_logits,
            'risk_score': risk_score
        }


class ClinicalFeatureEncoder:
    """Encode clinical features to numerical values"""
    
    def __init__(self):
        self.sex_map = {'Male': 0, 'Female': 1, 'M': 0, 'F': 1}
        self.grade_map = {'Low': 0, 'Intermediate': 1, 'High': 2}
        self.status_map = {'NED': 0, 'AWD': 1, 'D': 2}
        
        # Treatment encoding (one-hot)
        self.treatments = ['Surgery', 'Chemotherapy', 'Radiotherapy']
        
    def encode(self, row: Dict) -> np.ndarray:
        """
        Encode a patient record to numerical features
        Returns a feature vector of size 7:
        [sex, age_normalized, grade, surgery, chemo, radio, histological_type_encoded]
        """
        features = []
        
        # Sex (0 or 1)
        sex = row.get('Sex', 'Male')
        features.append(self.sex_map.get(sex, 0))
        
        # Age (normalized to [0, 1])
        age = row.get('Age', 50)
        features.append(age / 100.0)
        
        # Grade (0, 1, or 2)
        grade = row.get('Grade', 'Intermediate')
        features.append(self.grade_map.get(grade, 1))
        
        # Treatment (one-hot encoding for 3 treatments)
        treatment = row.get('Treatment', '')
        for t in self.treatments:
            features.append(1.0 if t in treatment else 0.0)
        
        # Simplified histological type (binary: aggressive vs non-aggressive)
        hist_type = row.get('Histological type', '')
        aggressive_types = ['Leiomyosarcoma', 'Liposarcoma', 'Undifferentiated']
        features.append(1.0 if any(t in hist_type for t in aggressive_types) else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def encode_status(self, status: str) -> int:
        """Encode survival status to class index"""
        return self.status_map.get(status, 0)
    
    @property
    def feature_dim(self):
        return 7  # sex, age, grade, 3x treatment, histological


def estimate_survival_months(status: str, risk_score: float, age: int, grade: str) -> Dict[str, float]:
    """
    Estimate survival time in months based on model predictions
    This is a heuristic approximation based on clinical data
    """
    # Base survival by status
    base_survival = {
        'NED': 120,  # 10 years
        'AWD': 48,   # 4 years
        'D': 18      # 1.5 years
    }
    
    base = base_survival.get(status, 60)
    
    # Risk score adjustment (-1 to +1, lower is better)
    risk_factor = 1.0 - (risk_score * 0.3)  # 30% variation
    
    # Age factor (younger patients generally have better prognosis)
    age_factor = 1.2 if age < 40 else 1.0 if age < 60 else 0.8
    
    # Grade factor
    grade_factor = 1.2 if grade == 'Low' else 1.0 if grade == 'Intermediate' else 0.7
    
    # Calculate estimated survival
    estimated_months = base * risk_factor * age_factor * grade_factor
    
    # Confidence intervals (±20%)
    lower = estimated_months * 0.8
    upper = estimated_months * 1.2
    
    return {
        'estimated_months': max(1, int(estimated_months)),
        'lower_bound': max(1, int(lower)),
        'upper_bound': int(upper),
        'estimated_years': round(estimated_months / 12, 1)
    }
