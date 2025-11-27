"""
PHASE 1: Multi-Modal Feature Extraction for Visual-to-Atomic Modeling
======================================================================

This module implements the first phase of the Visual-to-Atomic pipeline:
extracting multi-modal features from cooked food images including:
- Raw ingredient identification
- Spectral signatures from RGB
- Cooking method analysis
- Visual chemometric features

The extracted features serve as the initial state for Simulated Annealing
optimization in Phase 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import signal
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CookingMethod(Enum):
    """Enumeration of cooking methods that affect spectral signatures"""
    RAW = "raw"
    BOILED = "boiled"
    STEAMED = "steamed"
    FRIED = "fried"
    GRILLED = "grilled"
    BAKED = "baked"
    ROASTED = "roasted"
    SAUTEED = "sauteed"
    DEEP_FRIED = "deep_fried"
    SMOKED = "smoked"
    MICROWAVED = "microwaved"
    PRESSURE_COOKED = "pressure_cooked"


class OilType(Enum):
    """Types of cooking oils affecting chemical composition"""
    CANOLA = "canola"
    OLIVE = "olive"
    COCONUT = "coconut"
    VEGETABLE = "vegetable"
    PEANUT = "peanut"
    SUNFLOWER = "sunflower"
    AVOCADO = "avocado"
    BUTTER = "butter"
    GHEE = "ghee"
    NONE = "none"


@dataclass
class RawIngredientPrediction:
    """Prediction of raw ingredients from cooked food"""
    ingredient_name: str
    confidence: float
    mass_fraction: float  # Estimated fraction of total mass
    atomic_signature: Dict[str, float]  # Element -> ppm
    visual_features: np.ndarray
    category: str


@dataclass
class SpectralFeatures:
    """Extracted spectral features from RGB image"""
    rgb_histogram: np.ndarray  # (3, 256)
    lab_values: np.ndarray  # (H, W, 3)
    hsv_values: np.ndarray  # (H, W, 3)
    simulated_spectrum: np.ndarray  # (num_wavelengths,)
    wavelengths: np.ndarray  # Corresponding wavelengths (nm)
    texture_features: Dict[str, float]
    color_moments: Dict[str, np.ndarray]
    
    
@dataclass
class CookingAnalysis:
    """Analysis of cooking method and its effects"""
    predicted_method: CookingMethod
    confidence: float
    temperature_estimate: float  # Celsius
    duration_estimate: float  # Minutes
    moisture_loss: float  # Fraction (0-1)
    oil_type: Optional[OilType]
    browning_index: float  # Maillard reaction indicator
    char_level: float  # Carbonization level


@dataclass
class VisualChemometricFeatures:
    """Visual features correlated with chemical composition"""
    glossiness_score: float  # Fat/moisture content indicator
    surface_roughness: float
    color_degradation: float  # Oxidation indicator
    moisture_retention: float
    protein_texture_score: float
    fat_distribution: np.ndarray
    caramelization_index: float
    estimated_water_content: float


class RawIngredientClassifier(nn.Module):
    """
    Deep learning model to identify raw ingredients from cooked food.
    
    Architecture: EfficientNet-B7 + Custom Attention Head
    - Handles multi-ingredient decomposition
    - Estimates mass fractions
    - Predicts atomic signatures per ingredient
    """
    
    def __init__(
        self,
        num_ingredients: int = 500,
        num_elements: int = 45,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Backbone: EfficientNet-B7 for feature extraction
        self.backbone = models.efficientnet_b7(pretrained=pretrained)
        backbone_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Multi-head attention for ingredient decomposition
        self.ingredient_attention = nn.MultiheadAttention(
            embed_dim=backbone_features,
            num_heads=16,
            dropout=dropout,
            batch_first=True
        )
        
        # Ingredient classification head
        self.ingredient_classifier = nn.Sequential(
            nn.Linear(backbone_features, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, num_ingredients)
        )
        
        # Mass fraction estimation head
        self.mass_fraction_head = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_ingredients),
            nn.Softmax(dim=1)  # Fractions sum to 1
        )
        
        # Atomic signature prediction head (per ingredient)
        self.atomic_head = nn.Sequential(
            nn.Linear(backbone_features + num_ingredients, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_elements),
            nn.ReLU()  # Concentrations are positive
        )
        
        # Visual feature extractor
        self.visual_feature_head = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256)
        )
        
        self.num_ingredients = num_ingredients
        self.num_elements = num_elements
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            Dictionary containing:
                - ingredient_logits: (B, num_ingredients)
                - mass_fractions: (B, num_ingredients)
                - atomic_signatures: (B, num_elements)
                - visual_features: (B, 256)
        """
        # Extract backbone features
        features = self.backbone(x)  # (B, backbone_features)
        
        # Apply self-attention for ingredient decomposition
        features_attn = features.unsqueeze(1)  # (B, 1, features)
        attn_out, attn_weights = self.ingredient_attention(
            features_attn, features_attn, features_attn
        )
        attn_features = attn_out.squeeze(1)  # (B, features)
        
        # Ingredient classification
        ingredient_logits = self.ingredient_classifier(attn_features)
        
        # Mass fraction estimation
        mass_fractions = self.mass_fraction_head(attn_features)
        
        # Atomic signature prediction
        combined_features = torch.cat([attn_features, mass_fractions], dim=1)
        atomic_signatures = self.atomic_head(combined_features)
        
        # Visual features
        visual_features = self.visual_feature_head(features)
        
        return {
            'ingredient_logits': ingredient_logits,
            'mass_fractions': mass_fractions,
            'atomic_signatures': atomic_signatures,
            'visual_features': visual_features,
            'attention_weights': attn_weights
        }
    
    def predict_ingredients(
        self,
        image: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.1
    ) -> List[RawIngredientPrediction]:
        """
        Predict raw ingredients from cooked food image
        
        Args:
            image: Input image tensor (1, 3, H, W)
            top_k: Number of top ingredients to return
            threshold: Minimum mass fraction threshold
            
        Returns:
            List of RawIngredientPrediction objects
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(image)
            
            # Get probabilities
            probs = F.softmax(outputs['ingredient_logits'], dim=1)[0]
            mass_fractions = outputs['mass_fractions'][0]
            atomic_sig = outputs['atomic_signatures'][0]
            visual_feat = outputs['visual_features'][0]
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, top_k)
            
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                mass_frac = mass_fractions[idx].item()
                
                # Only include if above threshold
                if mass_frac >= threshold:
                    pred = RawIngredientPrediction(
                        ingredient_name=self._get_ingredient_name(idx.item()),
                        confidence=prob.item(),
                        mass_fraction=mass_frac,
                        atomic_signature=self._extract_atomic_signature(atomic_sig),
                        visual_features=visual_feat.cpu().numpy(),
                        category=self._get_ingredient_category(idx.item())
                    )
                    predictions.append(pred)
            
            return predictions
    
    def _get_ingredient_name(self, idx: int) -> str:
        """Map ingredient index to name"""
        # TODO: Load from database
        ingredient_map = {
            0: "chicken_breast", 1: "salmon", 2: "beef", 3: "rice",
            4: "broccoli", 5: "potato", 6: "carrot", 7: "onion",
            # ... (500 ingredients total)
        }
        return ingredient_map.get(idx, f"ingredient_{idx}")
    
    def _get_ingredient_category(self, idx: int) -> str:
        """Get ingredient category"""
        categories = {
            0: "protein", 1: "protein", 2: "protein", 3: "grain",
            4: "vegetable", 5: "starch", 6: "vegetable", 7: "vegetable",
        }
        return categories.get(idx, "unknown")
    
    def _extract_atomic_signature(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Convert atomic signature tensor to dictionary"""
        elements = [
            'Fe', 'Zn', 'Cu', 'Mn', 'Se', 'Cr', 'Mo', 'Co', 'I',  # Essential
            'Na', 'K', 'Ca', 'Mg', 'P', 'S', 'Cl',  # Macrominerals
            'Pb', 'Hg', 'As', 'Cd', 'Al', 'Ni',  # Contaminants
            # ... (45 elements total)
        ]
        
        signature = {}
        for i, element in enumerate(elements[:self.num_elements]):
            signature[element] = tensor[i].item()
        
        return signature


class SpectralFeatureExtractor:
    """
    Extracts spectral features from RGB images to simulate
    spectroscopic measurements.
    
    Uses physics-based modeling to estimate:
    - Reflectance spectra
    - Absorption features
    - Scattering properties
    """
    
    def __init__(self, wavelength_range: Tuple[int, int] = (400, 700)):
        self.wavelength_range = wavelength_range
        self.num_wavelengths = wavelength_range[1] - wavelength_range[0]
        self.wavelengths = np.linspace(
            wavelength_range[0],
            wavelength_range[1],
            self.num_wavelengths
        )
        
        # Initialize PCA for spectral reconstruction
        self.pca = PCA(n_components=10)
        self.scaler = StandardScaler()
        
    def extract(self, image: np.ndarray) -> SpectralFeatures:
        """
        Extract spectral features from RGB image
        
        Args:
            image: RGB image (H, W, 3) in range [0, 255]
            
        Returns:
            SpectralFeatures object
        """
        # Convert to float
        image_float = image.astype(np.float32) / 255.0
        
        # Extract color spaces
        rgb_hist = self._compute_rgb_histogram(image)
        lab_values = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        hsv_values = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Simulate spectral signature
        simulated_spectrum = self._simulate_spectrum(image_float)
        
        # Extract texture features
        texture_features = self._extract_texture_features(image)
        
        # Compute color moments
        color_moments = self._compute_color_moments(image_float)
        
        return SpectralFeatures(
            rgb_histogram=rgb_hist,
            lab_values=lab_values,
            hsv_values=hsv_values,
            simulated_spectrum=simulated_spectrum,
            wavelengths=self.wavelengths,
            texture_features=texture_features,
            color_moments=color_moments
        )
    
    def _compute_rgb_histogram(self, image: np.ndarray) -> np.ndarray:
        """Compute RGB histograms"""
        hist = np.zeros((3, 256))
        for i in range(3):
            hist[i], _ = np.histogram(image[:, :, i], bins=256, range=(0, 256))
        return hist / hist.sum(axis=1, keepdims=True)  # Normalize
    
    def _simulate_spectrum(self, image: np.ndarray) -> np.ndarray:
        """
        Simulate reflectance spectrum from RGB image using
        spectral reconstruction techniques
        """
        # Get mean RGB values
        mean_rgb = image.mean(axis=(0, 1))
        
        # Convert RGB to approximate reflectance spectrum
        # Using polynomial basis functions
        spectrum = np.zeros(self.num_wavelengths)
        
        # Red channel (600-700 nm)
        red_contribution = self._gaussian_basis(
            self.wavelengths, center=650, width=50
        ) * mean_rgb[0]
        
        # Green channel (500-600 nm)
        green_contribution = self._gaussian_basis(
            self.wavelengths, center=550, width=50
        ) * mean_rgb[1]
        
        # Blue channel (400-500 nm)
        blue_contribution = self._gaussian_basis(
            self.wavelengths, center=450, width=50
        ) * mean_rgb[2]
        
        spectrum = red_contribution + green_contribution + blue_contribution
        
        # Add spectral features based on texture
        texture_variance = image.std(axis=(0, 1)).mean()
        spectrum += self._add_spectral_texture(spectrum, texture_variance)
        
        # Normalize
        spectrum = spectrum / spectrum.max() if spectrum.max() > 0 else spectrum
        
        return spectrum
    
    def _gaussian_basis(
        self,
        x: np.ndarray,
        center: float,
        width: float
    ) -> np.ndarray:
        """Gaussian basis function for spectral reconstruction"""
        return np.exp(-0.5 * ((x - center) / width) ** 2)
    
    def _add_spectral_texture(
        self,
        spectrum: np.ndarray,
        texture_variance: float
    ) -> np.ndarray:
        """Add texture-induced spectral variations"""
        # High texture variance indicates rough surface -> more scattering
        noise = np.random.normal(0, texture_variance * 0.01, len(spectrum))
        return gaussian_filter(noise, sigma=2)
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract texture features using GLCM and other methods"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Compute gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Laplacian (measure of sharpness/roughness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Compute features
        features = {
            'gradient_mean': float(gradient_magnitude.mean()),
            'gradient_std': float(gradient_magnitude.std()),
            'laplacian_variance': float(laplacian.var()),
            'edge_density': float((gradient_magnitude > 50).sum() / gradient_magnitude.size),
            'roughness_index': float(laplacian.std()),
            'homogeneity': float(1.0 / (1.0 + gradient_magnitude.std())),
        }
        
        return features
    
    def _compute_color_moments(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute statistical color moments"""
        moments = {}
        
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = image[:, :, i]
            
            # First moment: mean
            mean = channel_data.mean()
            
            # Second moment: standard deviation
            std = channel_data.std()
            
            # Third moment: skewness
            skewness = ((channel_data - mean) ** 3).mean() / (std ** 3 + 1e-8)
            
            moments[f'{channel}_mean'] = np.array([mean])
            moments[f'{channel}_std'] = np.array([std])
            moments[f'{channel}_skewness'] = np.array([skewness])
        
        return moments


class CookingMethodAnalyzer(nn.Module):
    """
    Neural network to analyze cooking method from cooked food appearance.
    
    Predicts:
    - Cooking method (fried, grilled, baked, etc.)
    - Temperature estimate
    - Duration estimate
    - Moisture loss
    - Oil type
    - Browning/charring levels
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Backbone: ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        feature_dim = 2048
        
        # Cooking method classifier
        self.method_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(CookingMethod))
        )
        
        # Temperature regression
        self.temperature_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Normalize to [0, 1], then scale
        )
        
        # Duration regression
        self.duration_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Moisture loss regression
        self.moisture_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Oil type classifier
        self.oil_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(OilType))
        )
        
        # Browning index regression
        self.browning_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Char level regression
        self.char_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        features = self.features(x).flatten(1)
        
        return {
            'method_logits': self.method_head(features),
            'temperature': self.temperature_head(features) * 300 + 25,  # 25-325°C
            'duration': self.duration_head(features) * 120,  # 0-120 minutes
            'moisture_loss': self.moisture_head(features),
            'oil_logits': self.oil_head(features),
            'browning_index': self.browning_head(features),
            'char_level': self.char_head(features)
        }
    
    def analyze(self, image: torch.Tensor) -> CookingAnalysis:
        """Analyze cooking method from image"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(image)
            
            # Get predictions
            method_probs = F.softmax(outputs['method_logits'], dim=1)[0]
            method_idx = method_probs.argmax().item()
            method_conf = method_probs[method_idx].item()
            
            oil_probs = F.softmax(outputs['oil_logits'], dim=1)[0]
            oil_idx = oil_probs.argmax().item()
            
            return CookingAnalysis(
                predicted_method=list(CookingMethod)[method_idx],
                confidence=method_conf,
                temperature_estimate=outputs['temperature'][0].item(),
                duration_estimate=outputs['duration'][0].item(),
                moisture_loss=outputs['moisture_loss'][0].item(),
                oil_type=list(OilType)[oil_idx] if method_idx in [3, 7, 8] else None,
                browning_index=outputs['browning_index'][0].item(),
                char_level=outputs['char_level'][0].item()
            )


class VisualChemometricsEngine:
    """
    Extracts visual features that correlate with chemical composition.
    
    These features bridge the gap between visual appearance and
    atomic/molecular content.
    """
    
    def __init__(self):
        self.fat_detector = self._initialize_fat_detector()
        
    def extract(self, image: np.ndarray) -> VisualChemometricFeatures:
        """Extract visual chemometric features"""
        
        # Convert to float
        image_float = image.astype(np.float32) / 255.0
        
        # Extract individual features
        glossiness = self._compute_glossiness(image_float)
        roughness = self._compute_surface_roughness(image)
        color_deg = self._compute_color_degradation(image_float)
        moisture = self._estimate_moisture_retention(image_float)
        protein_texture = self._compute_protein_texture_score(image)
        fat_dist = self._analyze_fat_distribution(image_float)
        caramel_idx = self._compute_caramelization_index(image_float)
        water_content = self._estimate_water_content(image_float, moisture)
        
        return VisualChemometricFeatures(
            glossiness_score=glossiness,
            surface_roughness=roughness,
            color_degradation=color_deg,
            moisture_retention=moisture,
            protein_texture_score=protein_texture,
            fat_distribution=fat_dist,
            caramelization_index=caramel_idx,
            estimated_water_content=water_content
        )
    
    def _compute_glossiness(self, image: np.ndarray) -> float:
        """
        Compute glossiness score (indicator of fat/moisture content)
        High gloss = high fat or moisture
        """
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Find specular highlights
        _, bright_regions = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        highlight_ratio = bright_regions.sum() / (gray.size * 255)
        
        # Compute local variance (smooth surfaces have lower variance in highlights)
        local_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Combine metrics
        glossiness = highlight_ratio * (1.0 - min(local_var / 1000, 1.0))
        
        return float(np.clip(glossiness * 10, 0, 1))
    
    def _compute_surface_roughness(self, image: np.ndarray) -> float:
        """Compute surface roughness from texture analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use Laplacian variance as roughness metric
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        roughness = laplacian.var() / 1000.0  # Normalize
        
        return float(np.clip(roughness, 0, 1))
    
    def _compute_color_degradation(self, image: np.ndarray) -> float:
        """
        Measure color degradation (oxidation indicator)
        Browning/graying indicates oxidation
        """
        # Convert to LAB
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # L* channel (lightness): lower values = more browning
        l_channel = lab[:, :, 0].astype(float) / 255.0
        
        # a* channel: positive = red, negative = green
        # b* channel: positive = yellow, negative = blue
        
        # Compute browning as reduction in lightness + shift to brown hues
        mean_lightness = l_channel.mean()
        
        # Brown indicator: low lightness, low saturation
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].astype(float) / 255.0
        mean_saturation = saturation.mean()
        
        degradation = (1.0 - mean_lightness) * (1.0 - mean_saturation)
        
        return float(np.clip(degradation, 0, 1))
    
    def _estimate_moisture_retention(self, image: np.ndarray) -> float:
        """Estimate moisture retention from visual cues"""
        # High moisture: higher saturation, less browning, more glossiness
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        saturation = hsv[:, :, 1].astype(float) / 255.0
        value = hsv[:, :, 2].astype(float) / 255.0
        
        # Moisture correlates with saturation and value
        moisture = (saturation.mean() + value.mean()) / 2.0
        
        return float(np.clip(moisture, 0, 1))
    
    def _compute_protein_texture_score(self, image: np.ndarray) -> float:
        """
        Score protein-rich texture patterns
        Protein foods have characteristic fibrous textures
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Compute oriented gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Protein texture has strong directional gradients
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_direction = np.arctan2(sobely, sobelx)
        
        # Analyze gradient coherence (proteins have aligned fibers)
        coherence = self._compute_gradient_coherence(
            gradient_magnitude, gradient_direction
        )
        
        return float(np.clip(coherence, 0, 1))
    
    def _compute_gradient_coherence(
        self,
        magnitude: np.ndarray,
        direction: np.ndarray
    ) -> float:
        """Compute coherence of gradient directions"""
        # Use structure tensor
        Jxx = gaussian_filter(magnitude * np.cos(direction) ** 2, sigma=1)
        Jxy = gaussian_filter(magnitude * np.cos(direction) * np.sin(direction), sigma=1)
        Jyy = gaussian_filter(magnitude * np.sin(direction) ** 2, sigma=1)
        
        # Compute coherence
        trace = Jxx + Jyy
        det = Jxx * Jyy - Jxy ** 2
        
        coherence = np.sqrt((trace ** 2 - 4 * det) / (trace ** 2 + 1e-8))
        
        return coherence.mean()
    
    def _analyze_fat_distribution(self, image: np.ndarray) -> np.ndarray:
        """Analyze fat distribution in food"""
        # Fat appears as glossy, light-colored regions
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Fat mask: high value, medium-high saturation
        value = hsv[:, :, 2].astype(float) / 255.0
        saturation = hsv[:, :, 1].astype(float) / 255.0
        
        fat_score = value * np.exp(-((saturation - 0.3) ** 2) / 0.1)
        
        return fat_score
    
    def _compute_caramelization_index(self, image: np.ndarray) -> float:
        """
        Compute caramelization index
        Caramelization produces characteristic brown colors
        """
        # Convert to LAB
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        l_channel = lab[:, :, 0].astype(float) / 255.0
        a_channel = lab[:, :, 1].astype(float) / 255.0
        b_channel = lab[:, :, 2].astype(float) / 255.0
        
        # Caramelization: medium lightness, positive a* (red), high positive b* (yellow)
        caramel_mask = (
            (l_channel > 0.3) & (l_channel < 0.7) &
            (a_channel > 0.4) &
            (b_channel > 0.5)
        )
        
        caramel_index = caramel_mask.sum() / caramel_mask.size
        
        return float(np.clip(caramel_index * 5, 0, 1))
    
    def _estimate_water_content(
        self,
        image: np.ndarray,
        moisture_retention: float
    ) -> float:
        """Estimate water content percentage"""
        # Combine moisture retention with visual cues
        # High water content: less browning, more gloss, higher lightness
        
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        lightness = lab[:, :, 0].astype(float).mean() / 255.0
        
        # Empirical formula
        water_content = 0.4 * moisture_retention + 0.3 * lightness + 0.3 * 0.65
        
        return float(np.clip(water_content, 0.1, 0.95))
    
    def _initialize_fat_detector(self):
        """Initialize fat detection model (placeholder)"""
        # In production, this would load a trained model
        return None


class MultiModalFeatureExtractor:
    """
    Unified interface for extracting all multi-modal features.
    
    Combines:
    - Raw ingredient classification
    - Spectral features
    - Cooking method analysis
    - Visual chemometric features
    """
    
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        
        # Initialize all extractors
        self.ingredient_classifier = RawIngredientClassifier().to(device)
        self.spectral_extractor = SpectralFeatureExtractor()
        self.cooking_analyzer = CookingMethodAnalyzer().to(device)
        self.chemometrics_engine = VisualChemometricsEngine()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info("MultiModalFeatureExtractor initialized")
    
    def extract_all_features(
        self,
        image: np.ndarray,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract all multi-modal features from food image
        
        Args:
            image: RGB image (H, W, 3) in range [0, 255]
            user_context: Optional user-provided context (cooking method, oil type, etc.)
            
        Returns:
            Dictionary containing all extracted features
        """
        logger.info("Starting multi-modal feature extraction")
        
        # Prepare image tensor
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract raw ingredients
        logger.info("Extracting raw ingredients...")
        raw_ingredients = self.ingredient_classifier.predict_ingredients(image_tensor)
        
        # Extract spectral features
        logger.info("Extracting spectral features...")
        spectral_features = self.spectral_extractor.extract(image)
        
        # Analyze cooking method
        logger.info("Analyzing cooking method...")
        cooking_analysis = self.cooking_analyzer.analyze(image_tensor)
        
        # Extract visual chemometric features
        logger.info("Extracting visual chemometric features...")
        chemometric_features = self.chemometrics_engine.extract(image)
        
        # Combine all features
        features = {
            'raw_ingredients': raw_ingredients,
            'spectral_features': spectral_features,
            'cooking_analysis': cooking_analysis,
            'chemometric_features': chemometric_features,
            'user_context': user_context or {},
            'image_shape': image.shape
        }
        
        logger.info("Multi-modal feature extraction complete")
        
        return features
    
    def load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained weights for all models"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.ingredient_classifier.load_state_dict(
            checkpoint['ingredient_classifier']
        )
        self.cooking_analyzer.load_state_dict(
            checkpoint['cooking_analyzer']
        )
        
        logger.info(f"Loaded pretrained weights from {checkpoint_path}")


# Training utilities and loss functions

class MultiTaskLoss(nn.Module):
    """Multi-task loss for joint training"""
    
    def __init__(self, task_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        self.task_weights = task_weights or {
            'ingredient_classification': 1.0,
            'mass_fraction': 2.0,
            'atomic_signature': 3.0,
            'cooking_method': 1.0,
            'temperature': 0.5,
            'duration': 0.5,
            'moisture': 1.0,
        }
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss"""
        
        losses = {}
        
        # Ingredient classification loss
        if 'ingredient_logits' in predictions:
            losses['ingredient_classification'] = F.cross_entropy(
                predictions['ingredient_logits'],
                targets['ingredient_labels']
            )
        
        # Mass fraction loss (MSE)
        if 'mass_fractions' in predictions:
            losses['mass_fraction'] = F.mse_loss(
                predictions['mass_fractions'],
                targets['mass_fractions']
            )
        
        # Atomic signature loss (Huber loss for robustness)
        if 'atomic_signatures' in predictions:
            losses['atomic_signature'] = F.smooth_l1_loss(
                predictions['atomic_signatures'],
                targets['atomic_signatures']
            )
        
        # Cooking method loss
        if 'method_logits' in predictions:
            losses['cooking_method'] = F.cross_entropy(
                predictions['method_logits'],
                targets['cooking_method']
            )
        
        # Regression losses
        for key in ['temperature', 'duration', 'moisture']:
            if key in predictions:
                losses[key] = F.mse_loss(
                    predictions[key],
                    targets[key]
                )
        
        # Weighted total loss
        total_loss = sum(
            self.task_weights.get(k, 1.0) * v
            for k, v in losses.items()
        )
        
        losses['total'] = total_loss
        
        return losses


if __name__ == "__main__":
    # Test the feature extraction pipeline
    logger.info("Testing Phase 1: Multi-Modal Feature Extraction")
    
    # Create dummy image
    test_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    # Initialize extractor
    extractor = MultiModalFeatureExtractor()
    
    # Extract features
    features = extractor.extract_all_features(test_image)
    
    logger.info(f"Extracted {len(features['raw_ingredients'])} raw ingredients")
    logger.info(f"Cooking method: {features['cooking_analysis'].predicted_method.value}")
    logger.info(f"Estimated temperature: {features['cooking_analysis'].temperature_estimate:.1f}°C")
    logger.info(f"Moisture retention: {features['chemometric_features'].moisture_retention:.2f}")
    
    logger.info("Phase 1 test complete!")
