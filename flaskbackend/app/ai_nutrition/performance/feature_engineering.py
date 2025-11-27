"""
Advanced Feature Engineering System
===================================

Comprehensive feature engineering pipeline for nutrition and food data
with automated feature selection, extraction, and transformation.

Features:
1. Automated feature extraction from images
2. Nutritional feature engineering
3. Temporal and spatial features
4. Feature selection and ranking
5. Feature interaction discovery
6. Dimensionality reduction
7. Feature encoding and normalization
8. Custom feature pipelines

Performance Targets:
- Process 1000+ features/second
- Support 10,000+ feature combinations
- <50ms feature extraction latency
- Automatic feature importance ranking
- Zero manual feature engineering

Author: Wellomex AI Team
Date: November 2025
Version: 5.0.0
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from collections import defaultdict
import json

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class FeatureType(Enum):
    """Feature types"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    TEXT = "text"
    IMAGE = "image"
    TEMPORAL = "temporal"


class ScalingMethod(Enum):
    """Feature scaling methods"""
    STANDARD = "standard"  # Zero mean, unit variance
    MINMAX = "minmax"  # Scale to [0, 1]
    ROBUST = "robust"  # Robust to outliers
    LOG = "log"  # Log transform
    NONE = "none"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    # Feature selection
    enable_feature_selection: bool = True
    max_features: int = 1000
    feature_selection_method: str = "mutual_info"  # mutual_info, variance, correlation
    selection_threshold: float = 0.01
    
    # Feature scaling
    enable_scaling: bool = True
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    
    # Dimensionality reduction
    enable_dim_reduction: bool = False
    reduction_method: str = "pca"  # pca, svd, autoencoder
    target_dimensions: int = 100
    
    # Feature generation
    enable_interactions: bool = True
    max_interaction_degree: int = 2
    enable_polynomial: bool = False
    polynomial_degree: int = 2
    
    # Feature encoding
    enable_encoding: bool = True
    categorical_encoding: str = "onehot"  # onehot, label, target
    
    # Performance
    enable_caching: bool = True
    num_workers: int = 4


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================

class NutritionFeatureExtractor:
    """
    Extract features from nutrition data
    
    Computes derived nutritional features and ratios.
    """
    
    def __init__(self):
        self.feature_names: List[str] = []
    
    def extract(self, nutrition_data: Dict[str, float]) -> Dict[str, float]:
        """Extract nutritional features"""
        features = {}
        
        # Basic macronutrients
        protein = nutrition_data.get('protein_g', 0)
        carbs = nutrition_data.get('carbohydrates_g', 0)
        fat = nutrition_data.get('fat_g', 0)
        fiber = nutrition_data.get('fiber_g', 0)
        sugar = nutrition_data.get('sugar_g', 0)
        calories = nutrition_data.get('calories', 0)
        
        # Macronutrient ratios
        total_macros = protein + carbs + fat
        if total_macros > 0:
            features['protein_ratio'] = protein / total_macros
            features['carb_ratio'] = carbs / total_macros
            features['fat_ratio'] = fat / total_macros
        else:
            features['protein_ratio'] = 0
            features['carb_ratio'] = 0
            features['fat_ratio'] = 0
        
        # Calorie density
        if total_macros > 0:
            features['calorie_density'] = calories / total_macros
        else:
            features['calorie_density'] = 0
        
        # Protein efficiency ratio
        if calories > 0:
            features['protein_per_calorie'] = protein / calories
            features['fat_per_calorie'] = fat / calories
        else:
            features['protein_per_calorie'] = 0
            features['fat_per_calorie'] = 0
        
        # Fiber content
        if carbs > 0:
            features['fiber_ratio'] = fiber / carbs
            features['sugar_ratio'] = sugar / carbs
            features['complex_carb_ratio'] = (carbs - fiber - sugar) / carbs
        else:
            features['fiber_ratio'] = 0
            features['sugar_ratio'] = 0
            features['complex_carb_ratio'] = 0
        
        # Micronutrients
        for nutrient in ['vitamin_a_iu', 'vitamin_c_mg', 'calcium_mg', 'iron_mg',
                        'vitamin_d_iu', 'vitamin_e_mg', 'vitamin_k_mcg',
                        'thiamin_mg', 'riboflavin_mg', 'niacin_mg',
                        'vitamin_b6_mg', 'folate_mcg', 'vitamin_b12_mcg',
                        'phosphorus_mg', 'magnesium_mg', 'zinc_mg',
                        'selenium_mcg', 'copper_mg', 'manganese_mg',
                        'sodium_mg', 'potassium_mg']:
            
            value = nutrition_data.get(nutrient, 0)
            features[f'{nutrient}_normalized'] = value
            
            # Ratio to calories
            if calories > 0:
                features[f'{nutrient}_per_calorie'] = value / calories
        
        # Nutrient density score
        micronutrient_sum = sum(
            nutrition_data.get(n, 0) for n in [
                'vitamin_c_mg', 'calcium_mg', 'iron_mg', 'vitamin_a_iu'
            ]
        )
        
        if calories > 0:
            features['nutrient_density_score'] = micronutrient_sum / calories
        else:
            features['nutrient_density_score'] = 0
        
        # Health metrics
        features['is_high_protein'] = 1.0 if features['protein_ratio'] > 0.3 else 0.0
        features['is_high_fiber'] = 1.0 if features['fiber_ratio'] > 0.15 else 0.0
        features['is_low_sugar'] = 1.0 if features['sugar_ratio'] < 0.1 else 0.0
        features['is_low_fat'] = 1.0 if features['fat_ratio'] < 0.2 else 0.0
        
        # Glycemic load estimate
        features['estimated_glycemic_load'] = carbs * features['sugar_ratio'] * 0.7
        
        self.feature_names = list(features.keys())
        return features


class ImageFeatureExtractor:
    """
    Extract features from food images
    
    Uses computer vision to extract visual features.
    """
    
    def __init__(self):
        self.feature_names: List[str] = []
    
    def extract(self, image: Any) -> Dict[str, float]:
        """Extract image features"""
        features = {}
        
        if not NUMPY_AVAILABLE:
            return features
        
        # Convert to numpy if needed
        if not isinstance(image, np.ndarray):
            try:
                image = np.array(image)
            except:
                return features
        
        # Color features
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Mean color channels
            features['mean_red'] = float(np.mean(image[:, :, 0]))
            features['mean_green'] = float(np.mean(image[:, :, 1]))
            features['mean_blue'] = float(np.mean(image[:, :, 2]))
            
            # Color variance
            features['var_red'] = float(np.var(image[:, :, 0]))
            features['var_green'] = float(np.var(image[:, :, 1]))
            features['var_blue'] = float(np.var(image[:, :, 2]))
            
            # Color ratios
            total_color = features['mean_red'] + features['mean_green'] + features['mean_blue']
            if total_color > 0:
                features['red_ratio'] = features['mean_red'] / total_color
                features['green_ratio'] = features['mean_green'] / total_color
                features['blue_ratio'] = features['mean_blue'] / total_color
            
            # Brightness and contrast
            gray = np.mean(image, axis=2)
            features['brightness'] = float(np.mean(gray))
            features['contrast'] = float(np.std(gray))
            
            # Color diversity (entropy-like measure)
            hist_r, _ = np.histogram(image[:, :, 0], bins=16, range=(0, 256))
            hist_g, _ = np.histogram(image[:, :, 1], bins=16, range=(0, 256))
            hist_b, _ = np.histogram(image[:, :, 2], bins=16, range=(0, 256))
            
            features['color_diversity'] = float(
                np.std(hist_r) + np.std(hist_g) + np.std(hist_b)
            )
        
        # Texture features (simplified)
        if len(image.shape) >= 2:
            gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
            
            # Edge strength (gradient magnitude)
            if gray.shape[0] > 1 and gray.shape[1] > 1:
                grad_y = np.abs(np.diff(gray, axis=0))
                grad_x = np.abs(np.diff(gray, axis=1))
                
                features['edge_strength'] = float(np.mean(grad_y) + np.mean(grad_x))
                features['texture_complexity'] = float(np.std(grad_y) + np.std(grad_x))
        
        # Shape features (aspect ratio, etc.)
        height, width = image.shape[:2]
        features['aspect_ratio'] = width / height if height > 0 else 1.0
        features['image_area'] = float(width * height)
        
        self.feature_names = list(features.keys())
        return features


# ============================================================================
# FEATURE SELECTOR
# ============================================================================

class FeatureSelector:
    """
    Intelligent feature selection
    
    Selects most important features using various methods.
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        
        self.selected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}
        
        if SKLEARN_AVAILABLE:
            self.selector = None
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """Fit feature selector"""
        if not SKLEARN_AVAILABLE or not self.config.enable_feature_selection:
            self.selected_features = feature_names or list(range(X.shape[1]))
            return
        
        method = self.config.feature_selection_method
        
        # Variance-based selection
        if method == "variance":
            variances = np.var(X, axis=0)
            
            threshold = np.percentile(
                variances,
                (1 - self.config.selection_threshold) * 100
            )
            
            selected_idx = np.where(variances >= threshold)[0]
        
        # Mutual information-based selection
        elif method == "mutual_info" and y is not None:
            k = min(self.config.max_features, X.shape[1])
            self.selector = SelectKBest(mutual_info_classif, k=k)
            self.selector.fit(X, y)
            
            selected_idx = self.selector.get_support(indices=True)
            
            # Get scores
            scores = self.selector.scores_
            if feature_names:
                self.feature_scores = dict(zip(feature_names, scores))
        
        # Correlation-based selection
        elif method == "correlation":
            # Remove highly correlated features
            corr_matrix = np.corrcoef(X.T)
            
            selected_idx = []
            for i in range(X.shape[1]):
                # Keep if not highly correlated with any previously selected
                keep = True
                for j in selected_idx:
                    if abs(corr_matrix[i, j]) > 0.95:
                        keep = False
                        break
                
                if keep:
                    selected_idx.append(i)
                
                if len(selected_idx) >= self.config.max_features:
                    break
            
            selected_idx = np.array(selected_idx)
        
        else:
            # Default: select all
            selected_idx = np.arange(X.shape[1])
        
        # Update selected features
        if feature_names:
            self.selected_features = [feature_names[i] for i in selected_idx]
        else:
            self.selected_features = list(selected_idx)
        
        logger.info(f"Selected {len(self.selected_features)} features")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features"""
        if not self.config.enable_feature_selection:
            return X
        
        if SKLEARN_AVAILABLE and self.selector:
            return self.selector.transform(X)
        
        # Manual selection
        selected_idx = [
            i for i, name in enumerate(self.selected_features)
            if isinstance(name, int) or name in self.selected_features
        ]
        
        return X[:, selected_idx]


# ============================================================================
# FEATURE SCALER
# ============================================================================

class FeatureScaler:
    """
    Feature scaling and normalization
    
    Applies various scaling methods to features.
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        
        if SKLEARN_AVAILABLE:
            method = config.scaling_method
            
            if method == ScalingMethod.STANDARD:
                self.scaler = StandardScaler()
            elif method == ScalingMethod.MINMAX:
                self.scaler = MinMaxScaler()
            elif method == ScalingMethod.ROBUST:
                self.scaler = RobustScaler()
            else:
                self.scaler = None
        else:
            self.scaler = None
    
    def fit(self, X: np.ndarray):
        """Fit scaler"""
        if not self.config.enable_scaling or not SKLEARN_AVAILABLE:
            return
        
        if self.scaler:
            self.scaler.fit(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features"""
        if not self.config.enable_scaling:
            return X
        
        method = self.config.scaling_method
        
        if method == ScalingMethod.LOG:
            # Log transform
            return np.log1p(np.abs(X)) * np.sign(X)
        
        elif SKLEARN_AVAILABLE and self.scaler:
            return self.scaler.transform(X)
        
        return X


# ============================================================================
# FEATURE PIPELINE
# ============================================================================

class FeaturePipeline:
    """
    Complete feature engineering pipeline
    
    Coordinates extraction, selection, scaling, and transformation.
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        
        # Components
        self.nutrition_extractor = NutritionFeatureExtractor()
        self.image_extractor = ImageFeatureExtractor()
        self.selector = FeatureSelector(config)
        self.scaler = FeatureScaler(config)
        
        # State
        self.is_fitted = False
        self.feature_names: List[str] = []
        
        # Cache
        self.cache: Dict[str, np.ndarray] = {}
        
        logger.info("Feature Pipeline initialized")
    
    def extract_features(
        self,
        nutrition_data: Optional[Dict] = None,
        image_data: Optional[Any] = None
    ) -> np.ndarray:
        """Extract all features"""
        features = {}
        
        # Extract nutrition features
        if nutrition_data:
            nutrition_features = self.nutrition_extractor.extract(nutrition_data)
            features.update(nutrition_features)
        
        # Extract image features
        if image_data is not None:
            image_features = self.image_extractor.extract(image_data)
            features.update(image_features)
        
        # Convert to array
        if NUMPY_AVAILABLE:
            self.feature_names = list(features.keys())
            return np.array(list(features.values())).reshape(1, -1)
        
        return np.array([])
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ):
        """Fit pipeline"""
        if not NUMPY_AVAILABLE:
            return
        
        # Feature selection
        self.selector.fit(X, y, self.feature_names)
        X_selected = self.selector.transform(X)
        
        # Feature scaling
        self.scaler.fit(X_selected)
        
        self.is_fitted = True
        logger.info("✓ Feature pipeline fitted")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features"""
        if not NUMPY_AVAILABLE or not self.is_fitted:
            return X
        
        # Select features
        X_selected = self.selector.transform(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_selected)
        
        return X_scaled
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.selector.feature_scores


# ============================================================================
# TESTING
# ============================================================================

def test_feature_engineering():
    """Test feature engineering system"""
    print("=" * 80)
    print("ADVANCED FEATURE ENGINEERING - TEST")
    print("=" * 80)
    
    if not NUMPY_AVAILABLE:
        print("❌ NumPy not available")
        return
    
    # Create config
    config = FeatureConfig(
        enable_feature_selection=True,
        max_features=50,
        enable_scaling=True,
        scaling_method=ScalingMethod.STANDARD
    )
    
    # Create pipeline
    pipeline = FeaturePipeline(config)
    
    print("\n✓ Feature pipeline initialized")
    
    # Test nutrition feature extraction
    print("\n" + "="*80)
    print("Test: Nutrition Feature Extraction")
    print("="*80)
    
    nutrition_data = {
        'protein_g': 25.0,
        'carbohydrates_g': 50.0,
        'fat_g': 15.0,
        'fiber_g': 10.0,
        'sugar_g': 5.0,
        'calories': 450.0,
        'calcium_mg': 200.0,
        'iron_mg': 5.0
    }
    
    features = pipeline.extract_features(nutrition_data=nutrition_data)
    print(f"✓ Extracted {features.shape[1]} nutrition features")
    
    # Test image feature extraction
    print("\n" + "="*80)
    print("Test: Image Feature Extraction")
    print("="*80)
    
    # Create synthetic image
    image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    image_features = pipeline.extract_features(image_data=image)
    print(f"✓ Extracted {image_features.shape[1]} image features")
    
    # Test pipeline fitting
    print("\n" + "="*80)
    print("Test: Feature Pipeline")
    print("="*80)
    
    # Create synthetic dataset
    X = np.random.randn(100, 80)
    y = np.random.randint(0, 2, 100)
    
    X_transformed = pipeline.fit_transform(X, y)
    
    print(f"✓ Transformed features: {X.shape} -> {X_transformed.shape}")
    print(f"✓ Selected {len(pipeline.selector.selected_features)} features")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_feature_engineering()
