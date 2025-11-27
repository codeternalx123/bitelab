"""
Multimodal Atomic Composition Prediction from Food Images
==========================================================

This module implements a deep learning system that predicts elemental/atomic
composition (mg/kg concentrations) from RGB/hyperspectral food images combined
with weight and metadata, leveraging ICP-MS ground truth data.

Core Features:
- Vision Transformer / EfficientNet encoder for image features
- Multimodal fusion (image + weight + metadata)
- Regression head for element concentration prediction
- Uncertainty quantification for confidence intervals
- Integration with Health Impact Analyzer for toxicity/nutrition analysis
- Support for RGB and hyperspectral imaging (400-1000nm)

Scientific Basis:
- Transition metals (Fe, Cu, Mn) affect pigment colors via oxidation states
- Organic/inorganic ratios influence texture and glossiness
- Moisture/fat/protein affect light scattering and color saturation
- Cooking/spoilage alters optical properties via Maillard reactions

References:
- ICP-MS: Inductively Coupled Plasma Mass Spectrometry (gold standard)
- FDA Total Diet Study, EFSA food composition databases
- Kubelka-Munk optical models for reflectance simulation

Author: AI Nutrition Team
Version: 0.1.0-dev
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional imports for deep learning (graceful degradation)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available - using fallback models")
    PYTORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV not available - limited image preprocessing")
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    logger.warning("PIL not available - limited image loading")
    PIL_AVAILABLE = False


# ============================================================================
# ELEMENT DEFINITIONS
# ============================================================================

class ElementCategory(Enum):
    """Categories of elements for model organization"""
    TOXIC_HEAVY_METAL = "toxic_heavy_metal"  # Pb, Cd, As, Hg
    ESSENTIAL_MICRONUTRIENT = "essential_micronutrient"  # Fe, Zn, Cu, Se
    MACROMINERALS = "macrominerals"  # Ca, Mg, Na, K, P
    TRACE_ELEMENTS = "trace_elements"  # Mn, Cr, Mo, I
    NONMETALS = "nonmetals"  # C, N, O, S (if extended to macronutrient inference)


@dataclass
class ElementInfo:
    """Metadata for each predictable element"""
    symbol: str
    name: str
    atomic_number: int
    category: ElementCategory
    typical_range_mg_kg: Tuple[float, float]  # min, max in food
    regulatory_limit_mg_kg: Optional[float]  # safety threshold
    nutritional: bool  # True if essential nutrient
    toxic: bool  # True if toxic at high levels
    rda_mg_day: Optional[float]  # Recommended Daily Allowance (if nutrient)
    
    def __post_init__(self):
        """Validation"""
        if self.toxic and self.nutritional:
            logger.debug(f"{self.symbol}: Element is both toxic and nutritional (dose-dependent)")


# Comprehensive element database
ELEMENT_DATABASE = {
    # Toxic Heavy Metals
    "Pb": ElementInfo("Pb", "Lead", 82, ElementCategory.TOXIC_HEAVY_METAL,
                     (0.001, 5.0), 0.1, False, True, None),
    "Cd": ElementInfo("Cd", "Cadmium", 48, ElementCategory.TOXIC_HEAVY_METAL,
                     (0.001, 2.0), 0.05, False, True, None),
    "As": ElementInfo("As", "Arsenic", 33, ElementCategory.TOXIC_HEAVY_METAL,
                     (0.001, 1.0), 0.1, False, True, None),
    "Hg": ElementInfo("Hg", "Mercury", 80, ElementCategory.TOXIC_HEAVY_METAL,
                     (0.001, 0.5), 0.02, False, True, None),
    
    # Essential Micronutrients (Trace Minerals)
    "Fe": ElementInfo("Fe", "Iron", 26, ElementCategory.ESSENTIAL_MICRONUTRIENT,
                     (1.0, 500.0), None, True, False, 18.0),  # RDA: 18mg/day (women)
    "Zn": ElementInfo("Zn", "Zinc", 30, ElementCategory.ESSENTIAL_MICRONUTRIENT,
                     (1.0, 200.0), None, True, False, 11.0),  # RDA: 11mg/day
    "Cu": ElementInfo("Cu", "Copper", 29, ElementCategory.ESSENTIAL_MICRONUTRIENT,
                     (0.5, 50.0), 10.0, True, True, 0.9),  # RDA: 0.9mg/day (toxic if >10mg/kg)
    "Se": ElementInfo("Se", "Selenium", 34, ElementCategory.ESSENTIAL_MICRONUTRIENT,
                     (0.01, 5.0), None, True, False, 0.055),  # RDA: 55μg/day
    
    # Macrominerals
    "Ca": ElementInfo("Ca", "Calcium", 20, ElementCategory.MACROMINERALS,
                     (10.0, 5000.0), None, True, False, 1000.0),  # RDA: 1000mg/day
    "Mg": ElementInfo("Mg", "Magnesium", 12, ElementCategory.MACROMINERALS,
                     (10.0, 2000.0), None, True, False, 400.0),  # RDA: 400mg/day
    "Na": ElementInfo("Na", "Sodium", 11, ElementCategory.MACROMINERALS,
                     (100.0, 10000.0), None, True, True, 2300.0),  # RDA: <2300mg/day (limit)
    "K": ElementInfo("K", "Potassium", 19, ElementCategory.MACROMINERALS,
                     (100.0, 10000.0), None, True, False, 3400.0),  # RDA: 3400mg/day
    "P": ElementInfo("P", "Phosphorus", 15, ElementCategory.MACROMINERALS,
                     (50.0, 5000.0), None, True, False, 700.0),  # RDA: 700mg/day
    
    # Trace Elements
    "Mn": ElementInfo("Mn", "Manganese", 25, ElementCategory.TRACE_ELEMENTS,
                     (0.1, 100.0), 11.0, True, True, 2.3),  # RDA: 2.3mg/day
    "Cr": ElementInfo("Cr", "Chromium", 24, ElementCategory.TRACE_ELEMENTS,
                     (0.01, 10.0), None, True, False, 0.035),  # RDA: 35μg/day
    "Mo": ElementInfo("Mo", "Molybdenum", 42, ElementCategory.TRACE_ELEMENTS,
                     (0.01, 5.0), None, True, False, 0.045),  # RDA: 45μg/day
    "I": ElementInfo("I", "Iodine", 53, ElementCategory.TRACE_ELEMENTS,
                     (0.01, 10.0), None, True, False, 0.15),  # RDA: 150μg/day
    
    # Nonmetals (extended for macronutrient inference)
    "C": ElementInfo("C", "Carbon", 6, ElementCategory.NONMETALS,
                     (100000.0, 500000.0), None, False, False, None),  # ~40-50% of dry mass
    "N": ElementInfo("N", "Nitrogen", 7, ElementCategory.NONMETALS,
                     (10000.0, 100000.0), None, False, False, None),  # protein = N × 6.25
    "O": ElementInfo("O", "Oxygen", 8, ElementCategory.NONMETALS,
                     (100000.0, 500000.0), None, False, False, None),
    "S": ElementInfo("S", "Sulfur", 16, ElementCategory.NONMETALS,
                     (100.0, 5000.0), None, False, False, None),
}

# Element lists for model output
TOXIC_ELEMENTS = ["Pb", "Cd", "As", "Hg"]
NUTRIENT_ELEMENTS = ["Fe", "Zn", "Cu", "Se", "Ca", "Mg", "Na", "K", "P", "Mn", "Cr", "Mo", "I"]
ALL_ELEMENTS = list(ELEMENT_DATABASE.keys())


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FoodImageData:
    """Input data for atomic prediction"""
    image: np.ndarray  # RGB (H, W, 3) or hyperspectral (H, W, C)
    weight_grams: float
    food_type: Optional[str] = None  # e.g., "leafy_vegetable", "fish", "grain"
    preparation: Optional[str] = None  # e.g., "raw", "cooked", "fried"
    source: Optional[str] = None  # e.g., "organic_farm", "supermarket", "imported"
    brand: Optional[str] = None
    imaging_mode: str = "rgb"  # "rgb" or "hyperspectral"
    
    def validate(self):
        """Input validation"""
        if self.image is None or self.image.size == 0:
            raise ValueError("Image cannot be empty")
        if self.weight_grams <= 0:
            raise ValueError(f"Weight must be positive, got {self.weight_grams}")
        if self.imaging_mode not in ["rgb", "hyperspectral"]:
            raise ValueError(f"Unknown imaging mode: {self.imaging_mode}")


@dataclass
class ElementPrediction:
    """Predicted elemental concentration with uncertainty"""
    element: str
    concentration_mg_kg: float
    uncertainty_mg_kg: float  # standard deviation
    confidence: float  # 0-1 score
    exceeds_limit: bool
    
    def get_confidence_interval(self, z_score: float = 1.96) -> Tuple[float, float]:
        """Calculate confidence interval (default: 95% CI)"""
        margin = z_score * self.uncertainty_mg_kg
        return (
            max(0, self.concentration_mg_kg - margin),
            self.concentration_mg_kg + margin
        )


@dataclass
class AtomicCompositionResult:
    """Complete prediction result"""
    predictions: List[ElementPrediction]
    timestamp: datetime
    model_version: str
    image_quality_score: float  # 0-1 quality assessment
    total_uncertainty: float  # aggregate uncertainty metric
    
    def get_element(self, symbol: str) -> Optional[ElementPrediction]:
        """Get prediction for specific element"""
        for pred in self.predictions:
            if pred.element == symbol:
                return pred
        return None
    
    def get_toxic_elements(self) -> List[ElementPrediction]:
        """Get all toxic element predictions"""
        return [p for p in self.predictions if p.element in TOXIC_ELEMENTS]
    
    def get_nutrient_elements(self) -> List[ElementPrediction]:
        """Get all nutrient predictions"""
        return [p for p in self.predictions if p.element in NUTRIENT_ELEMENTS]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Health Impact Analyzer"""
        return {
            pred.element: {
                "concentration_mg_kg": pred.concentration_mg_kg,
                "uncertainty_mg_kg": pred.uncertainty_mg_kg,
                "confidence": pred.confidence,
                "exceeds_limit": pred.exceeds_limit
            }
            for pred in self.predictions
        }


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

class ImagePreprocessor:
    """Preprocessing pipeline for food images"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True):
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array([0.485, 0.456, 0.406])  # ImageNet stats
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess(self, image: np.ndarray, mode: str = "rgb") -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image: Raw image (H, W, 3) for RGB or (H, W, C) for hyperspectral
            mode: "rgb" or "hyperspectral"
        
        Returns:
            Preprocessed image tensor
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available - minimal preprocessing")
            return self._fallback_preprocess(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        # White balance correction
        image = self._white_balance(image)
        
        # Resize
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Convert to float [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize
        if self.normalize and mode == "rgb":
            image = (image - self.mean) / self.std
        
        # Channel-first format (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def _white_balance(self, image: np.ndarray) -> np.ndarray:
        """Simple white balance using gray world assumption"""
        try:
            result = cv2.xphoto.createSimpleWB().balanceWhite(image)
            return result
        except:
            # Fallback: manual gray world
            avg = image.mean(axis=(0, 1))
            gray = avg.mean()
            scale = gray / (avg + 1e-6)
            return np.clip(image * scale, 0, 255).astype(np.uint8)
    
    def _fallback_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Minimal preprocessing without OpenCV"""
        # Simple resize (nearest neighbor)
        from scipy.ndimage import zoom
        h, w = image.shape[:2]
        zoom_h = self.target_size[0] / h
        zoom_w = self.target_size[1] / w
        if len(image.shape) == 3:
            zoomed = zoom(image, (zoom_h, zoom_w, 1), order=1)
        else:
            zoomed = zoom(image, (zoom_h, zoom_w), order=1)
        
        # Normalize
        normalized = zoomed.astype(np.float32) / 255.0
        
        # Channel-first
        if len(normalized.shape) == 3:
            normalized = np.transpose(normalized, (2, 0, 1))
        
        return normalized
    
    def assess_quality(self, image: np.ndarray) -> float:
        """
        Assess image quality (0-1 score)
        
        Factors: brightness, sharpness, noise, blur
        """
        if not CV2_AVAILABLE:
            return 0.8  # default
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        # Brightness check (ideal: 100-150 mean)
        brightness = gray.mean()
        brightness_score = 1.0 - abs(brightness - 125) / 125.0
        brightness_score = max(0, brightness_score)
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        sharpness_score = min(1.0, sharpness / 500.0)  # empirical threshold
        
        # Overall quality
        quality = 0.6 * sharpness_score + 0.4 * brightness_score
        
        return float(quality)


# ============================================================================
# VISION ENCODER (PyTorch)
# ============================================================================

if PYTORCH_AVAILABLE:
    class VisionEncoder(nn.Module):
        """
        Vision Transformer or CNN backbone for image feature extraction
        
        Supports:
        - Vision Transformer (ViT) for global context
        - EfficientNet for efficiency
        - ResNet for baseline
        """
        
        def __init__(self, backbone: str = "efficientnet_b0",
                     pretrained: bool = True,
                     feature_dim: int = 512):
            super().__init__()
            self.backbone_name = backbone
            self.feature_dim = feature_dim
            
            # Load backbone (simplified - would use timm in production)
            if "efficientnet" in backbone:
                self.backbone = self._create_efficientnet(pretrained)
            elif "vit" in backbone:
                self.backbone = self._create_vit(pretrained)
            else:
                self.backbone = self._create_resnet(pretrained)
            
            # Feature projection
            self.projector = nn.Linear(self._get_backbone_dim(), feature_dim)
        
        def _create_efficientnet(self, pretrained: bool):
            """Simplified EfficientNet (would use timm.create_model in production)"""
            # Placeholder - in production use: timm.create_model('efficientnet_b0')
            return nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
        
        def _create_vit(self, pretrained: bool):
            """Simplified Vision Transformer"""
            # Placeholder - in production use: timm.create_model('vit_base_patch16_224')
            return nn.Sequential(
                nn.Conv2d(3, 768, kernel_size=16, stride=16),  # patch embedding
                nn.Flatten(2),
                nn.Transpose(1, 2),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True),
                    num_layers=6
                ),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
        
        def _create_resnet(self, pretrained: bool):
            """Simplified ResNet"""
            return nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
        
        def _get_backbone_dim(self) -> int:
            """Get output dimension of backbone"""
            if "efficientnet" in self.backbone_name:
                return 128
            elif "vit" in self.backbone_name:
                return 768
            else:
                return 128
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, channels, height, width)
            Returns:
                features: (batch, feature_dim)
            """
            features = self.backbone(x)
            features = self.projector(features)
            return features


# ============================================================================
# MULTIMODAL FUSION MODEL (PyTorch)
# ============================================================================

if PYTORCH_AVAILABLE:
    class AtomicCompositionNet(nn.Module):
        """
        Multimodal network: Vision + Tabular → Element Concentrations
        
        Architecture:
        1. Vision encoder (ViT/EfficientNet)
        2. Tabular encoder (weight + metadata)
        3. Attention-based fusion
        4. Regression head per element
        5. Uncertainty head (variance estimation)
        """
        
        def __init__(self, num_elements: int = len(ALL_ELEMENTS),
                     vision_backbone: str = "efficientnet_b0",
                     feature_dim: int = 512,
                     dropout: float = 0.3):
            super().__init__()
            self.num_elements = num_elements
            self.element_symbols = ALL_ELEMENTS
            
            # Vision encoder
            self.vision_encoder = VisionEncoder(vision_backbone, True, feature_dim)
            
            # Tabular encoder (weight + metadata)
            self.tabular_encoder = nn.Sequential(
                nn.Linear(5, 64),  # [weight, food_type_idx, prep_idx, source_idx, brand_idx]
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, feature_dim),
            )
            
            # Fusion layer (attention-based)
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            
            # Combined feature processing
            self.fusion_mlp = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, feature_dim),
            )
            
            # Regression heads (one per element)
            self.element_regressors = nn.ModuleDict({
                element: nn.Sequential(
                    nn.Linear(feature_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),  # concentration
                )
                for element in self.element_symbols
            })
            
            # Uncertainty heads (predict log variance)
            self.uncertainty_heads = nn.ModuleDict({
                element: nn.Linear(feature_dim, 1)
                for element in self.element_symbols
            })
        
        def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> Tuple[Dict, Dict]:
            """
            Args:
                image: (batch, 3, H, W)
                tabular: (batch, 5) [weight, food_type, prep, source, brand]
            
            Returns:
                concentrations: Dict[element] -> (batch, 1) mg/kg
                uncertainties: Dict[element] -> (batch, 1) log variance
            """
            # Encode vision
            vision_features = self.vision_encoder(image)  # (batch, feature_dim)
            
            # Encode tabular
            tabular_features = self.tabular_encoder(tabular)  # (batch, feature_dim)
            
            # Attention fusion
            # Stack features: (batch, 2, feature_dim)
            stacked = torch.stack([vision_features, tabular_features], dim=1)
            fused, _ = self.fusion_attention(stacked, stacked, stacked)
            fused = fused.mean(dim=1)  # (batch, feature_dim)
            
            # Concatenate and process
            combined = torch.cat([vision_features, tabular_features], dim=-1)
            combined = self.fusion_mlp(combined)  # (batch, feature_dim)
            
            # Add residual from attention fusion
            final_features = combined + fused
            
            # Predict concentrations per element
            concentrations = {}
            uncertainties = {}
            
            for element in self.element_symbols:
                conc = self.element_regressors[element](final_features)
                log_var = self.uncertainty_heads[element](final_features)
                
                # Apply element-specific constraints
                element_info = ELEMENT_DATABASE[element]
                min_val, max_val = element_info.typical_range_mg_kg
                
                # Sigmoid to [0, 1], then scale to typical range
                conc = torch.sigmoid(conc) * (max_val - min_val) + min_val
                
                concentrations[element] = conc
                uncertainties[element] = log_var
            
            return concentrations, uncertainties
        
        def predict_with_uncertainty(self, image: torch.Tensor, 
                                     tabular: torch.Tensor,
                                     num_samples: int = 10) -> Tuple[Dict, Dict]:
            """
            Monte Carlo Dropout for uncertainty estimation
            
            Returns:
                mean_concentrations: Dict[element] -> mean prediction
                std_concentrations: Dict[element] -> standard deviation
            """
            self.train()  # Enable dropout
            
            all_predictions = {element: [] for element in self.element_symbols}
            
            with torch.no_grad():
                for _ in range(num_samples):
                    conc, _ = self.forward(image, tabular)
                    for element in self.element_symbols:
                        all_predictions[element].append(conc[element])
            
            # Compute mean and std
            mean_conc = {}
            std_conc = {}
            
            for element in self.element_symbols:
                stacked = torch.stack(all_predictions[element], dim=0)
                mean_conc[element] = stacked.mean(dim=0)
                std_conc[element] = stacked.std(dim=0)
            
            self.eval()  # Back to eval mode
            
            return mean_conc, std_conc


# ============================================================================
# ATOMIC VISION PREDICTOR (Main Interface)
# ============================================================================

class AtomicVisionPredictor:
    """
    Main interface for image-based atomic composition prediction
    
    Usage:
        predictor = AtomicVisionPredictor()
        predictor.load_model("path/to/checkpoint.pth")
        
        image_data = FoodImageData(
            image=rgb_array,
            weight_grams=150.0,
            food_type="leafy_vegetable"
        )
        
        result = predictor.predict(image_data)
        
        # Use with Health Impact Analyzer
        from health_impact_analyzer import HealthImpactAnalyzer
        analyzer = HealthImpactAnalyzer()
        analyzer.integrate_atomic_data(result.to_dict())
    """
    
    def __init__(self, model_path: Optional[str] = None,
                 use_gpu: bool = True):
        self.model_path = model_path
        self.device = self._setup_device(use_gpu)
        self.preprocessor = ImagePreprocessor()
        self.model = None
        self.model_version = "0.1.0-dev"
        
        # Metadata encoders
        self.food_type_encoder = {"leafy_vegetable": 0, "fish": 1, "grain": 2, "fruit": 3, "meat": 4}
        self.prep_encoder = {"raw": 0, "cooked": 1, "fried": 2, "baked": 3}
        self.source_encoder = {"organic_farm": 0, "supermarket": 1, "imported": 2}
        self.brand_encoder = {}  # dynamically populated
        
        if model_path and PYTORCH_AVAILABLE:
            self.load_model(model_path)
        elif not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available - using fallback heuristic model")
    
    def _setup_device(self, use_gpu: bool):
        """Setup compute device"""
        if not PYTORCH_AVAILABLE:
            return None
        
        if use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        if not PYTORCH_AVAILABLE:
            logger.error("Cannot load model - PyTorch not available")
            return
        
        try:
            self.model = AtomicCompositionNet().to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model_version = checkpoint.get('version', self.model_version)
            logger.info(f"Loaded model from {model_path} (v{self.model_version})")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def predict(self, image_data: FoodImageData, 
                use_uncertainty: bool = True) -> AtomicCompositionResult:
        """
        Predict elemental composition from food image
        
        Args:
            image_data: Input image + metadata
            use_uncertainty: Use Monte Carlo dropout for uncertainty
        
        Returns:
            AtomicCompositionResult with predictions
        """
        image_data.validate()
        
        # Assess image quality
        quality_score = self.preprocessor.assess_quality(image_data.image)
        if quality_score < 0.3:
            logger.warning(f"Low image quality ({quality_score:.2f}) - predictions may be unreliable")
        
        # Preprocess image
        processed_image = self.preprocessor.preprocess(
            image_data.image, 
            image_data.imaging_mode
        )
        
        # Encode metadata
        tabular = self._encode_metadata(image_data)
        
        # Predict
        if self.model and PYTORCH_AVAILABLE:
            predictions = self._predict_pytorch(processed_image, tabular, use_uncertainty)
        else:
            predictions = self._predict_fallback(image_data)
        
        # Build result
        result = AtomicCompositionResult(
            predictions=predictions,
            timestamp=datetime.now(),
            model_version=self.model_version,
            image_quality_score=quality_score,
            total_uncertainty=np.mean([p.uncertainty_mg_kg for p in predictions])
        )
        
        return result
    
    def _encode_metadata(self, image_data: FoodImageData) -> np.ndarray:
        """Encode metadata to numerical vector"""
        food_type_idx = self.food_type_encoder.get(image_data.food_type, 0)
        prep_idx = self.prep_encoder.get(image_data.preparation, 0)
        source_idx = self.source_encoder.get(image_data.source, 1)
        brand_idx = self.brand_encoder.get(image_data.brand, 0)
        
        return np.array([
            image_data.weight_grams,
            food_type_idx,
            prep_idx,
            source_idx,
            brand_idx
        ], dtype=np.float32)
    
    def _predict_pytorch(self, image: np.ndarray, tabular: np.ndarray,
                        use_uncertainty: bool) -> List[ElementPrediction]:
        """PyTorch model prediction"""
        # Convert to tensors
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        tabular_tensor = torch.from_numpy(tabular).unsqueeze(0).to(self.device)
        
        # Predict with uncertainty
        if use_uncertainty:
            mean_conc, std_conc = self.model.predict_with_uncertainty(
                image_tensor, tabular_tensor, num_samples=10
            )
        else:
            with torch.no_grad():
                mean_conc, log_var = self.model(image_tensor, tabular_tensor)
                std_conc = {k: torch.exp(0.5 * v) for k, v in log_var.items()}
        
        # Convert to ElementPrediction objects
        predictions = []
        for element in ALL_ELEMENTS:
            conc = mean_conc[element].item()
            uncert = std_conc[element].item()
            
            # Confidence (inverse of coefficient of variation)
            confidence = 1.0 / (1.0 + uncert / (conc + 1e-6))
            
            # Check regulatory limits
            element_info = ELEMENT_DATABASE[element]
            exceeds = False
            if element_info.regulatory_limit_mg_kg is not None:
                exceeds = conc > element_info.regulatory_limit_mg_kg
            
            predictions.append(ElementPrediction(
                element=element,
                concentration_mg_kg=conc,
                uncertainty_mg_kg=uncert,
                confidence=confidence,
                exceeds_limit=exceeds
            ))
        
        return predictions
    
    def _predict_fallback(self, image_data: FoodImageData) -> List[ElementPrediction]:
        """
        Fallback heuristic model (no ML)
        
        Uses simple rules based on food type and color analysis
        """
        logger.info("Using fallback heuristic model (ML not available)")
        
        # Analyze image color
        mean_color = image_data.image.mean(axis=(0, 1))
        
        predictions = []
        
        for element in ALL_ELEMENTS:
            element_info = ELEMENT_DATABASE[element]
            min_val, max_val = element_info.typical_range_mg_kg
            
            # Heuristic based on food type and color
            if element == "Fe":
                # Higher Fe in dark/red foods
                redness = mean_color[0] if len(mean_color) == 3 else 100
                conc = min_val + (max_val - min_val) * (redness / 255.0)
            elif element in ["Pb", "Cd", "As", "Hg"]:
                # Assume low toxic metals
                conc = min_val + (max_val - min_val) * 0.1
            else:
                # Middle of typical range
                conc = (min_val + max_val) / 2.0
            
            # High uncertainty for heuristic model
            uncert = (max_val - min_val) * 0.3
            confidence = 0.3  # low confidence
            
            exceeds = False
            if element_info.regulatory_limit_mg_kg:
                exceeds = conc > element_info.regulatory_limit_mg_kg
            
            predictions.append(ElementPrediction(
                element=element,
                concentration_mg_kg=conc,
                uncertainty_mg_kg=uncert,
                confidence=confidence,
                exceeds_limit=exceeds
            ))
        
        return predictions


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_image(image_path: str) -> np.ndarray:
    """Load image from file path"""
    if PIL_AVAILABLE:
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        return np.array(img)
    elif CV2_AVAILABLE:
        img = cv2.imread(image_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ImportError("Neither PIL nor OpenCV available for image loading")


def get_atomic_predictor(model_path: Optional[str] = None) -> AtomicVisionPredictor:
    """Factory function for AtomicVisionPredictor (singleton pattern)"""
    if not hasattr(get_atomic_predictor, '_instance'):
        get_atomic_predictor._instance = AtomicVisionPredictor(model_path)
    return get_atomic_predictor._instance


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Predict atomic composition from food image
    
    # Load image
    # image = load_image("sample_food.jpg")
    # Generate synthetic test image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create input data
    image_data = FoodImageData(
        image=image,
        weight_grams=150.0,
        food_type="leafy_vegetable",
        preparation="raw",
        source="organic_farm"
    )
    
    # Predict
    predictor = AtomicVisionPredictor()
    result = predictor.predict(image_data)
    
    # Display results
    print("\n" + "="*80)
    print("ATOMIC COMPOSITION PREDICTION RESULTS")
    print("="*80)
    print(f"Timestamp: {result.timestamp}")
    print(f"Model Version: {result.model_version}")
    print(f"Image Quality: {result.image_quality_score:.2f}/1.0")
    print(f"Overall Uncertainty: {result.total_uncertainty:.2f} mg/kg")
    print()
    
    # Toxic elements
    toxic = result.get_toxic_elements()
    if toxic:
        print("⚠️  TOXIC ELEMENTS:")
        for pred in toxic:
            ci_low, ci_high = pred.get_confidence_interval()
            status = "❌ EXCEEDS LIMIT" if pred.exceeds_limit else "✓ Safe"
            print(f"  {pred.element}: {pred.concentration_mg_kg:.3f} mg/kg "
                  f"(95% CI: {ci_low:.3f}-{ci_high:.3f}) - {status}")
    
    # Nutrients
    nutrients = result.get_nutrient_elements()
    if nutrients:
        print("\n✓ ESSENTIAL NUTRIENTS:")
        for pred in nutrients:
            element_info = ELEMENT_DATABASE[pred.element]
            print(f"  {pred.element} ({element_info.name}): {pred.concentration_mg_kg:.2f} mg/kg "
                  f"(confidence: {pred.confidence:.2f})")
    
    print("\n" + "="*80)
