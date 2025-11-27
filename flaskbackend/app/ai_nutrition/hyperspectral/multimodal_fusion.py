"""
Multimodal Fusion for Hyperspectral and RGB Imaging

This module provides advanced fusion techniques for combining hyperspectral and RGB
(or multispectral) imagery to leverage complementary information. Fusion improves
accuracy, spatial resolution, and robustness for atomic composition prediction.

Key Features:
- Early fusion (feature-level concatenation)
- Late fusion (decision-level combination)
- Hybrid fusion (attention-weighted integration)
- Pansharpening techniques (Gram-Schmidt, Brovey, IHS)
- Deep learning-based fusion (cross-modal attention)
- Uncertainty-aware fusion
- Spatial-spectral alignment and registration

Scientific Foundation:
- Pansharpening: Gillespie et al., 1987 (Brovey); Laben & Brower, 2000 (Gram-Schmidt)
- Multimodal fusion: Lahat et al., "Multimodal Data Fusion", IEEE, 2015
- Attention mechanisms: Vaswani et al., "Attention is All You Need", 2017
- Uncertainty fusion: Durrant-Whyte & Henderson, "Multisensor Data Fusion", 2008

Author: AI Nutrition Team
Date: 2024
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Optional dependencies
try:
    from scipy.ndimage import zoom, shift
    from scipy.optimize import minimize
    from scipy.linalg import svd
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logging.warning("SciPy not available. Some fusion features will be limited.")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionMethod(Enum):
    """Multimodal fusion methods"""
    EARLY_CONCAT = "early_concat"  # Feature-level concatenation
    EARLY_PCA = "early_pca"  # PCA-based dimensionality reduction
    LATE_AVERAGE = "late_average"  # Decision averaging
    LATE_WEIGHTED = "late_weighted"  # Weighted decision fusion
    HYBRID_ATTENTION = "hybrid_attention"  # Attention-weighted fusion
    PANSHARP_BROVEY = "pansharp_brovey"  # Brovey transform
    PANSHARP_GS = "pansharp_gs"  # Gram-Schmidt
    PANSHARP_IHS = "pansharp_ihs"  # Intensity-Hue-Saturation
    OPTIMAL_WEIGHTS = "optimal_weights"  # Data-driven weight optimization


class AlignmentMethod(Enum):
    """Spatial alignment methods"""
    NONE = "none"  # No alignment (assume already aligned)
    TRANSLATION = "translation"  # Translation only
    AFFINE = "affine"  # Affine transformation
    CORRELATION = "correlation"  # Cross-correlation based


@dataclass
class FusionConfig:
    """Configuration for multimodal fusion"""
    method: FusionMethod = FusionMethod.EARLY_CONCAT
    
    # Alignment
    alignment: AlignmentMethod = AlignmentMethod.CORRELATION
    target_resolution: Optional[Tuple[int, int]] = None  # (height, width)
    
    # Early fusion
    normalize_features: bool = True
    pca_components: Optional[int] = None  # For EARLY_PCA
    
    # Late fusion
    weights: Optional[Dict[str, float]] = None  # Modal weights for weighted fusion
    temperature: float = 1.0  # Temperature for soft fusion
    
    # Attention fusion
    attention_type: str = "channel"  # 'channel', 'spatial', 'cross'
    attention_heads: int = 4
    
    # Pansharpening
    ratio: int = 4  # Resolution ratio (HSI:RGB)
    
    # Uncertainty
    use_uncertainty: bool = False
    uncertainty_method: str = "variance"  # 'variance', 'entropy'


@dataclass
class FusionResult:
    """Result of multimodal fusion"""
    fused_features: np.ndarray  # Fused feature representation
    fused_predictions: Optional[np.ndarray] = None  # Fused predictions (late fusion)
    
    # Metadata
    weights: Optional[Dict[str, float]] = None  # Modal weights used
    attention_maps: Optional[np.ndarray] = None  # Attention visualizations
    alignment_params: Optional[Dict[str, np.ndarray]] = None  # Alignment transforms
    
    # Quality metrics
    spatial_resolution: Optional[Tuple[int, int]] = None
    spectral_bands: Optional[int] = None
    fusion_quality: Optional[float] = None  # Quality score (0-1)


class MultimodalFusion:
    """
    Multimodal fusion for hyperspectral and RGB imagery
    
    Supports various fusion strategies from simple concatenation to
    sophisticated attention-based deep learning approaches. Handles
    spatial alignment and resolution differences.
    """
    
    def __init__(self, config: FusionConfig):
        """
        Initialize multimodal fusion
        
        Args:
            config: Fusion configuration
        """
        self.config = config
        self.scaler = StandardScaler() if HAS_SKLEARN and config.normalize_features else None
        self.pca = None
        
        logger.info(f"Initialized multimodal fusion: {config.method.value}")
    
    def fuse_images(
        self,
        hyperspectral: np.ndarray,
        rgb: np.ndarray,
        hsi_wavelengths: Optional[np.ndarray] = None,
        rgb_wavelengths: Optional[np.ndarray] = None
    ) -> FusionResult:
        """
        Fuse hyperspectral and RGB images
        
        Args:
            hyperspectral: Hyperspectral image, shape (H, W, C_hsi)
            rgb: RGB image, shape (H_rgb, W_rgb, 3)
            hsi_wavelengths: HSI wavelengths (optional, for spectral matching)
            rgb_wavelengths: RGB center wavelengths (optional, ~[650, 550, 450] nm)
            
        Returns:
            Fusion result
        """
        logger.info(
            f"Fusing HSI {hyperspectral.shape} with RGB {rgb.shape} "
            f"using {self.config.method.value}"
        )
        
        # Step 1: Spatial alignment and resolution matching
        hsi_aligned, rgb_aligned, alignment_params = self._align_images(
            hyperspectral, rgb
        )
        
        # Step 2: Apply fusion method
        if self.config.method == FusionMethod.EARLY_CONCAT:
            result = self._early_fusion_concat(hsi_aligned, rgb_aligned)
        
        elif self.config.method == FusionMethod.EARLY_PCA:
            result = self._early_fusion_pca(hsi_aligned, rgb_aligned)
        
        elif self.config.method == FusionMethod.HYBRID_ATTENTION:
            result = self._hybrid_attention_fusion(hsi_aligned, rgb_aligned)
        
        elif self.config.method in [
            FusionMethod.PANSHARP_BROVEY,
            FusionMethod.PANSHARP_GS,
            FusionMethod.PANSHARP_IHS
        ]:
            result = self._pansharpening(hsi_aligned, rgb_aligned)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.config.method}")
        
        result.alignment_params = alignment_params
        result.spatial_resolution = hsi_aligned.shape[:2]
        
        # Compute fusion quality
        result.fusion_quality = self._compute_fusion_quality(result, hsi_aligned, rgb_aligned)
        
        logger.info(f"Fusion complete. Quality score: {result.fusion_quality:.3f}")
        return result
    
    def fuse_features(
        self,
        features_hsi: np.ndarray,
        features_rgb: np.ndarray,
        uncertainties: Optional[Dict[str, np.ndarray]] = None
    ) -> FusionResult:
        """
        Fuse extracted features from different modalities
        
        Args:
            features_hsi: HSI features, shape (n_samples, n_features_hsi)
            features_rgb: RGB features, shape (n_samples, n_features_rgb)
            uncertainties: Optional uncertainty estimates for each modality
            
        Returns:
            Fusion result
        """
        logger.info(
            f"Fusing features: HSI {features_hsi.shape}, RGB {features_rgb.shape}"
        )
        
        # Normalize if requested
        if self.config.normalize_features and self.scaler is not None:
            features_hsi = self.scaler.fit_transform(features_hsi)
            features_rgb = self.scaler.fit_transform(features_rgb)
        
        # Uncertainty-aware fusion
        if self.config.use_uncertainty and uncertainties is not None:
            weights = self._compute_uncertainty_weights(uncertainties)
        else:
            weights = self.config.weights or {'hsi': 0.5, 'rgb': 0.5}
        
        # Concatenate features
        fused = np.concatenate([
            features_hsi * weights['hsi'],
            features_rgb * weights['rgb']
        ], axis=1)
        
        # Optional PCA reduction
        if self.config.pca_components is not None:
            fused = self._apply_pca(fused, self.config.pca_components)
        
        return FusionResult(
            fused_features=fused,
            weights=weights
        )
    
    def fuse_predictions(
        self,
        predictions_hsi: np.ndarray,
        predictions_rgb: np.ndarray,
        uncertainties: Optional[Dict[str, np.ndarray]] = None
    ) -> FusionResult:
        """
        Fuse predictions from different modalities (late fusion)
        
        Args:
            predictions_hsi: HSI predictions, shape (n_samples, n_classes)
            predictions_rgb: RGB predictions, shape (n_samples, n_classes)
            uncertainties: Optional uncertainty estimates
            
        Returns:
            Fusion result with fused predictions
        """
        logger.info("Performing late fusion of predictions")
        
        # Compute fusion weights
        if self.config.use_uncertainty and uncertainties is not None:
            weights = self._compute_uncertainty_weights(uncertainties)
        elif self.config.weights is not None:
            weights = self.config.weights
        else:
            # Equal weighting
            weights = {'hsi': 0.5, 'rgb': 0.5}
        
        # Apply temperature scaling (for soft predictions)
        pred_hsi_temp = predictions_hsi / self.config.temperature
        pred_rgb_temp = predictions_rgb / self.config.temperature
        
        # Weighted average
        if self.config.method == FusionMethod.LATE_AVERAGE:
            fused = (pred_hsi_temp + pred_rgb_temp) / 2.0
        
        elif self.config.method == FusionMethod.LATE_WEIGHTED:
            fused = (
                weights['hsi'] * pred_hsi_temp +
                weights['rgb'] * pred_rgb_temp
            )
        
        elif self.config.method == FusionMethod.OPTIMAL_WEIGHTS:
            # Learn optimal weights (requires validation data)
            fused, weights = self._optimize_fusion_weights(
                pred_hsi_temp, pred_rgb_temp
            )
        
        else:
            raise ValueError(f"Method {self.config.method} not for late fusion")
        
        return FusionResult(
            fused_features=None,
            fused_predictions=fused,
            weights=weights
        )
    
    def _align_images(
        self,
        hyperspectral: np.ndarray,
        rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Align images spatially and match resolutions
        
        Args:
            hyperspectral: HSI image, shape (H_hsi, W_hsi, C_hsi)
            rgb: RGB image, shape (H_rgb, W_rgb, 3)
            
        Returns:
            (aligned_hsi, aligned_rgb, alignment_params)
        """
        h_hsi, w_hsi, c_hsi = hyperspectral.shape
        h_rgb, w_rgb, c_rgb = rgb.shape
        
        alignment_params = {}
        
        # Determine target resolution
        if self.config.target_resolution is not None:
            h_target, w_target = self.config.target_resolution
        else:
            # Use higher resolution (typically RGB)
            h_target = max(h_hsi, h_rgb)
            w_target = max(w_hsi, w_rgb)
        
        # Resize images to target resolution
        if (h_hsi, w_hsi) != (h_target, w_target):
            if HAS_SCIPY:
                zoom_h = h_target / h_hsi
                zoom_w = w_target / w_hsi
                hyperspectral = zoom(hyperspectral, (zoom_h, zoom_w, 1), order=1)
            else:
                logger.warning("SciPy not available. Skipping HSI resizing.")
        
        if (h_rgb, w_rgb) != (h_target, w_target):
            if HAS_SCIPY:
                zoom_h = h_target / h_rgb
                zoom_w = w_target / w_rgb
                rgb = zoom(rgb, (zoom_h, zoom_w, 1), order=1)
            else:
                logger.warning("SciPy not available. Skipping RGB resizing.")
        
        # Spatial alignment (if requested)
        if self.config.alignment == AlignmentMethod.CORRELATION:
            if HAS_SCIPY:
                # Use first HSI band and RGB grayscale for correlation
                hsi_gray = hyperspectral[:, :, 0]
                rgb_gray = np.mean(rgb, axis=2)
                
                # Find translation offset using cross-correlation
                offset = self._find_translation_offset(hsi_gray, rgb_gray)
                alignment_params['translation'] = offset
                
                # Apply translation to RGB
                if np.any(offset != 0):
                    rgb = shift(rgb, (*offset, 0), mode='constant')
                    logger.debug(f"Applied translation offset: {offset}")
        
        elif self.config.alignment == AlignmentMethod.TRANSLATION:
            # Manual translation (could be user-specified)
            pass
        
        elif self.config.alignment == AlignmentMethod.AFFINE:
            logger.warning("Affine alignment not yet implemented. Using identity.")
        
        return hyperspectral, rgb, alignment_params
    
    def _find_translation_offset(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> np.ndarray:
        """Find translation offset using cross-correlation"""
        # Normalize images
        img1 = (image1 - np.mean(image1)) / (np.std(image1) + 1e-8)
        img2 = (image2 - np.mean(image2)) / (np.std(image2) + 1e-8)
        
        # Compute cross-correlation in frequency domain
        from numpy.fft import fft2, ifft2, fftshift
        
        f1 = fft2(img1)
        f2 = fft2(img2)
        cross_corr = fftshift(ifft2(f1 * np.conj(f2)).real)
        
        # Find peak
        peak = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
        offset = np.array(peak) - np.array(img1.shape) // 2
        
        return offset
    
    def _early_fusion_concat(
        self,
        hyperspectral: np.ndarray,
        rgb: np.ndarray
    ) -> FusionResult:
        """
        Early fusion via feature concatenation
        
        Args:
            hyperspectral: HSI image, shape (H, W, C_hsi)
            rgb: RGB image, shape (H, W, 3)
            
        Returns:
            Fusion result
        """
        # Concatenate along channel dimension
        fused = np.concatenate([hyperspectral, rgb], axis=2)
        
        # Flatten spatial dimensions for feature vector
        h, w, c = fused.shape
        features = fused.reshape(-1, c)  # (H*W, C_hsi+3)
        
        # Normalize if requested
        if self.config.normalize_features and self.scaler is not None:
            features = self.scaler.fit_transform(features)
            fused = features.reshape(h, w, c)
        
        return FusionResult(
            fused_features=fused,
            spectral_bands=c
        )
    
    def _early_fusion_pca(
        self,
        hyperspectral: np.ndarray,
        rgb: np.ndarray
    ) -> FusionResult:
        """
        Early fusion with PCA dimensionality reduction
        
        Args:
            hyperspectral: HSI image, shape (H, W, C_hsi)
            rgb: RGB image, shape (H, W, 3)
            
        Returns:
            Fusion result
        """
        # First concatenate
        concat_result = self._early_fusion_concat(hyperspectral, rgb)
        features = concat_result.fused_features
        
        h, w, c = features.shape
        features_flat = features.reshape(-1, c)
        
        # Apply PCA
        n_components = self.config.pca_components or min(50, c)
        reduced = self._apply_pca(features_flat, n_components)
        
        # Reshape back
        fused = reduced.reshape(h, w, n_components)
        
        return FusionResult(
            fused_features=fused,
            spectral_bands=n_components
        )
    
    def _apply_pca(self, features: np.ndarray, n_components: int) -> np.ndarray:
        """Apply PCA dimensionality reduction"""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for PCA")
        
        if self.pca is None or self.pca.n_components != n_components:
            self.pca = PCA(n_components=n_components)
        
        reduced = self.pca.fit_transform(features)
        logger.debug(
            f"PCA: {features.shape[1]} -> {n_components} "
            f"(explained variance: {self.pca.explained_variance_ratio_.sum():.2%})"
        )
        return reduced
    
    def _hybrid_attention_fusion(
        self,
        hyperspectral: np.ndarray,
        rgb: np.ndarray
    ) -> FusionResult:
        """
        Hybrid fusion using attention mechanisms
        
        Args:
            hyperspectral: HSI image, shape (H, W, C_hsi)
            rgb: RGB image, shape (H, W, 3)
            
        Returns:
            Fusion result with attention maps
        """
        h, w, c_hsi = hyperspectral.shape
        _, _, c_rgb = rgb.shape
        
        # Compute attention weights based on feature statistics
        if self.config.attention_type == "channel":
            # Channel-wise attention (which channels are most informative)
            attn_hsi = self._channel_attention(hyperspectral)  # (C_hsi,)
            attn_rgb = self._channel_attention(rgb)  # (3,)
            
            # Apply attention
            hsi_weighted = hyperspectral * attn_hsi.reshape(1, 1, -1)
            rgb_weighted = rgb * attn_rgb.reshape(1, 1, -1)
        
        elif self.config.attention_type == "spatial":
            # Spatial attention (which spatial regions are important)
            attn_hsi = self._spatial_attention(hyperspectral)  # (H, W, 1)
            attn_rgb = self._spatial_attention(rgb)  # (H, W, 1)
            
            hsi_weighted = hyperspectral * attn_hsi
            rgb_weighted = rgb * attn_rgb
        
        elif self.config.attention_type == "cross":
            # Cross-modal attention (HSI attends to RGB and vice versa)
            hsi_weighted, rgb_weighted, attn_maps = self._cross_modal_attention(
                hyperspectral, rgb
            )
        
        else:
            raise ValueError(f"Unknown attention type: {self.config.attention_type}")
        
        # Concatenate weighted features
        fused = np.concatenate([hsi_weighted, rgb_weighted], axis=2)
        
        return FusionResult(
            fused_features=fused,
            attention_maps=attn_hsi if self.config.attention_type == "channel" else None,
            spectral_bands=c_hsi + c_rgb
        )
    
    def _channel_attention(self, features: np.ndarray) -> np.ndarray:
        """
        Compute channel-wise attention weights
        
        Args:
            features: Feature map, shape (H, W, C)
            
        Returns:
            Attention weights, shape (C,)
        """
        # Global average pooling
        gap = np.mean(features, axis=(0, 1))  # (C,)
        
        # Global max pooling
        gmp = np.max(features, axis=(0, 1))  # (C,)
        
        # Combine with learned weights (simplified: just average)
        attention = (gap + gmp) / 2.0
        
        # Normalize to [0, 1]
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        # Softmax-like normalization
        exp_attn = np.exp(attention)
        attention = exp_attn / np.sum(exp_attn)
        
        return attention
    
    def _spatial_attention(self, features: np.ndarray) -> np.ndarray:
        """
        Compute spatial attention map
        
        Args:
            features: Feature map, shape (H, W, C)
            
        Returns:
            Attention map, shape (H, W, 1)
        """
        # Channel-wise statistics
        channel_avg = np.mean(features, axis=2, keepdims=True)  # (H, W, 1)
        channel_max = np.max(features, axis=2, keepdims=True)  # (H, W, 1)
        
        # Combine
        attention = (channel_avg + channel_max) / 2.0
        
        # Normalize
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        return attention
    
    def _cross_modal_attention(
        self,
        features1: np.ndarray,
        features2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cross-modal attention: features1 attends to features2
        
        Args:
            features1: First modality, shape (H, W, C1)
            features2: Second modality, shape (H, W, C2)
            
        Returns:
            (weighted_features1, weighted_features2, attention_maps)
        """
        h, w, c1 = features1.shape
        _, _, c2 = features2.shape
        
        # Flatten spatial dimensions
        f1 = features1.reshape(-1, c1)  # (H*W, C1)
        f2 = features2.reshape(-1, c2)  # (H*W, C2)
        
        # Compute attention scores (simplified: cosine similarity)
        # Normalize features
        f1_norm = f1 / (np.linalg.norm(f1, axis=1, keepdims=True) + 1e-8)
        f2_norm = f2 / (np.linalg.norm(f2, axis=1, keepdims=True) + 1e-8)
        
        # Attention: f1 attends to f2
        attn_scores = np.dot(f1_norm, f2_norm.T)  # (H*W, H*W)
        
        # Softmax over columns
        exp_scores = np.exp(attn_scores)
        attn_weights = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-8)
        
        # Apply attention to f2
        f2_attended = np.dot(attn_weights, f2)  # (H*W, C2)
        
        # Similarly for f2 attending to f1
        attn_weights_rev = exp_scores / (np.sum(exp_scores, axis=0, keepdims=True) + 1e-8)
        f1_attended = np.dot(attn_weights_rev.T, f1)  # (H*W, C1)
        
        # Reshape back
        f1_weighted = (f1 + f1_attended).reshape(h, w, c1)
        f2_weighted = (f2 + f2_attended).reshape(h, w, c2)
        
        return f1_weighted, f2_weighted, attn_weights.reshape(h, w, h, w)
    
    def _pansharpening(
        self,
        hyperspectral: np.ndarray,
        rgb: np.ndarray
    ) -> FusionResult:
        """
        Pansharpening: enhance HSI spatial resolution using high-res RGB
        
        Args:
            hyperspectral: Low-res HSI, shape (H, W, C_hsi)
            rgb: High-res RGB, shape (H, W, 3)
            
        Returns:
            Fusion result with sharpened HSI
        """
        if self.config.method == FusionMethod.PANSHARP_BROVEY:
            fused = self._brovey_transform(hyperspectral, rgb)
        
        elif self.config.method == FusionMethod.PANSHARP_GS:
            fused = self._gram_schmidt_sharpening(hyperspectral, rgb)
        
        elif self.config.method == FusionMethod.PANSHARP_IHS:
            fused = self._ihs_transform(hyperspectral, rgb)
        
        else:
            raise ValueError(f"Unknown pansharpening method: {self.config.method}")
        
        return FusionResult(
            fused_features=fused,
            spectral_bands=hyperspectral.shape[2]
        )
    
    def _brovey_transform(
        self,
        hyperspectral: np.ndarray,
        rgb: np.ndarray
    ) -> np.ndarray:
        """
        Brovey transform pansharpening
        
        Args:
            hyperspectral: Low-res HSI, shape (H, W, C)
            rgb: High-res RGB, shape (H, W, 3)
            
        Returns:
            Sharpened HSI, shape (H, W, C)
        """
        # Compute panchromatic (intensity) from RGB
        pan = np.mean(rgb, axis=2, keepdims=True)
        
        # Compute intensity from HSI
        hsi_intensity = np.mean(hyperspectral, axis=2, keepdims=True)
        
        # Brovey transform: HSI_sharp = HSI * (Pan / HSI_intensity)
        ratio = pan / (hsi_intensity + 1e-8)
        sharpened = hyperspectral * ratio
        
        return sharpened
    
    def _gram_schmidt_sharpening(
        self,
        hyperspectral: np.ndarray,
        rgb: np.ndarray
    ) -> np.ndarray:
        """
        Gram-Schmidt pansharpening
        
        Args:
            hyperspectral: Low-res HSI, shape (H, W, C)
            rgb: High-res RGB, shape (H, W, 3)
            
        Returns:
            Sharpened HSI, shape (H, W, C)
        """
        if not HAS_SCIPY:
            logger.warning("SciPy not available. Using simplified GS.")
            return self._brovey_transform(hyperspectral, rgb)
        
        h, w, c = hyperspectral.shape
        
        # Simulate low-res panchromatic from HSI
        hsi_pan = np.mean(hyperspectral, axis=2)
        
        # High-res panchromatic from RGB
        rgb_pan = np.mean(rgb, axis=2)
        
        # Gram-Schmidt orthogonalization
        # Project HSI onto panchromatic direction, then adjust
        sharpened = hyperspectral.copy()
        
        for i in range(c):
            band = hyperspectral[:, :, i]
            
            # Compute projection coefficient
            coeff = np.sum(band * hsi_pan) / (np.sum(hsi_pan ** 2) + 1e-8)
            
            # Adjust using high-res pan
            adjustment = coeff * (rgb_pan - hsi_pan)
            sharpened[:, :, i] = band + adjustment
        
        return sharpened
    
    def _ihs_transform(
        self,
        hyperspectral: np.ndarray,
        rgb: np.ndarray
    ) -> np.ndarray:
        """
        Intensity-Hue-Saturation pansharpening
        
        Args:
            hyperspectral: Low-res HSI, shape (H, W, C)
            rgb: High-res RGB, shape (H, W, 3)
            
        Returns:
            Sharpened HSI, shape (H, W, C)
        """
        # Compute intensity from both
        hsi_intensity = np.mean(hyperspectral, axis=2, keepdims=True)
        rgb_intensity = np.mean(rgb, axis=2, keepdims=True)
        
        # Replace HSI intensity with RGB intensity
        intensity_diff = rgb_intensity - hsi_intensity
        sharpened = hyperspectral + intensity_diff
        
        # Ensure non-negative
        sharpened = np.maximum(sharpened, 0)
        
        return sharpened
    
    def _compute_uncertainty_weights(
        self,
        uncertainties: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute fusion weights based on uncertainty
        
        Args:
            uncertainties: Uncertainty estimates for each modality
            
        Returns:
            Normalized weights
        """
        # Inverse uncertainty weighting
        weights = {}
        total_inv_unc = 0.0
        
        for modal, unc in uncertainties.items():
            inv_unc = 1.0 / (np.mean(unc) + 1e-8)
            weights[modal] = inv_unc
            total_inv_unc += inv_unc
        
        # Normalize
        for modal in weights:
            weights[modal] /= total_inv_unc
        
        logger.debug(f"Uncertainty-based weights: {weights}")
        return weights
    
    def _optimize_fusion_weights(
        self,
        predictions1: np.ndarray,
        predictions2: np.ndarray,
        ground_truth: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Learn optimal fusion weights (requires validation data)
        
        Args:
            predictions1: First modality predictions
            predictions2: Second modality predictions
            ground_truth: Ground truth labels (optional)
            
        Returns:
            (fused_predictions, optimal_weights)
        """
        if ground_truth is None:
            logger.warning("No ground truth provided. Using equal weights.")
            return (predictions1 + predictions2) / 2.0, {'hsi': 0.5, 'rgb': 0.5}
        
        # Define objective: minimize prediction error
        def objective(w):
            w_norm = w[0] / (w[0] + w[1])
            fused = w_norm * predictions1 + (1 - w_norm) * predictions2
            error = np.mean((fused - ground_truth) ** 2)
            return error
        
        # Optimize
        if HAS_SCIPY:
            result = minimize(
                objective,
                x0=[0.5, 0.5],
                bounds=[(0.0, 1.0), (0.0, 1.0)],
                method='L-BFGS-B'
            )
            w_opt = result.x
            w_norm = w_opt[0] / (w_opt[0] + w_opt[1])
        else:
            w_norm = 0.5
        
        weights = {'hsi': w_norm, 'rgb': 1 - w_norm}
        fused = w_norm * predictions1 + (1 - w_norm) * predictions2
        
        logger.info(f"Optimized fusion weights: {weights}")
        return fused, weights
    
    def _compute_fusion_quality(
        self,
        result: FusionResult,
        hyperspectral: np.ndarray,
        rgb: np.ndarray
    ) -> float:
        """
        Compute fusion quality score
        
        Args:
            result: Fusion result
            hyperspectral: Original HSI
            rgb: Original RGB
            
        Returns:
            Quality score (0-1)
        """
        # Quality based on information preservation
        fused = result.fused_features
        
        if fused is None:
            return 0.5  # Neutral quality for late fusion
        
        # Compute variance retention (how much information is preserved)
        hsi_var = np.var(hyperspectral)
        rgb_var = np.var(rgb)
        fused_var = np.var(fused)
        
        # Ideally, fused variance should be similar to sum of input variances
        expected_var = hsi_var + rgb_var
        quality = min(1.0, fused_var / (expected_var + 1e-8))
        
        return float(quality)


if __name__ == "__main__":
    # Example usage and validation
    print("=" * 80)
    print("Multimodal Fusion System - Example Usage")
    print("=" * 80)
    
    # Create synthetic data
    print("\n1. Creating synthetic hyperspectral and RGB images...")
    h, w = 64, 64
    c_hsi = 100
    
    # Hyperspectral (low-res)
    hsi = np.random.rand(h // 2, w // 2, c_hsi).astype(np.float32)
    
    # RGB (high-res)
    rgb = np.random.rand(h, w, 3).astype(np.float32)
    
    print(f"HSI shape: {hsi.shape}")
    print(f"RGB shape: {rgb.shape}")
    
    # Test different fusion methods
    print("\n2. Testing fusion methods...")
    
    methods = [
        FusionMethod.EARLY_CONCAT,
        FusionMethod.EARLY_PCA,
        FusionMethod.HYBRID_ATTENTION,
        FusionMethod.PANSHARP_BROVEY
    ]
    
    for method in methods:
        config = FusionConfig(
            method=method,
            pca_components=50 if method == FusionMethod.EARLY_PCA else None,
            attention_type="channel" if method == FusionMethod.HYBRID_ATTENTION else "channel"
        )
        
        fusion = MultimodalFusion(config)
        result = fusion.fuse_images(hsi, rgb)
        
        print(f"\n{method.value.upper()}:")
        print(f"  - Fused shape: {result.fused_features.shape}")
        print(f"  - Spectral bands: {result.spectral_bands}")
        print(f"  - Quality score: {result.fusion_quality:.3f}")
        if result.weights:
            print(f"  - Weights: {result.weights}")
    
    # Test feature fusion
    print("\n3. Testing feature-level fusion...")
    n_samples = 100
    features_hsi = np.random.rand(n_samples, 200)
    features_rgb = np.random.rand(n_samples, 50)
    
    config = FusionConfig(method=FusionMethod.EARLY_CONCAT, normalize_features=True)
    fusion = MultimodalFusion(config)
    
    result = fusion.fuse_features(features_hsi, features_rgb)
    print(f"Feature fusion: {features_hsi.shape} + {features_rgb.shape} -> {result.fused_features.shape}")
    
    # Test late fusion (prediction fusion)
    print("\n4. Testing late fusion (predictions)...")
    n_classes = 20
    pred_hsi = np.random.rand(n_samples, n_classes)
    pred_rgb = np.random.rand(n_samples, n_classes)
    
    # Normalize to probabilities
    pred_hsi = pred_hsi / np.sum(pred_hsi, axis=1, keepdims=True)
    pred_rgb = pred_rgb / np.sum(pred_rgb, axis=1, keepdims=True)
    
    config = FusionConfig(
        method=FusionMethod.LATE_WEIGHTED,
        weights={'hsi': 0.7, 'rgb': 0.3}
    )
    fusion = MultimodalFusion(config)
    
    result = fusion.fuse_predictions(pred_hsi, pred_rgb)
    print(f"Prediction fusion shape: {result.fused_predictions.shape}")
    print(f"Fusion weights: {result.weights}")
    print(f"Sample fused prediction: {result.fused_predictions[0, :5]}")
    
    # Test uncertainty-aware fusion
    print("\n5. Testing uncertainty-aware fusion...")
    uncertainties = {
        'hsi': np.random.uniform(0.1, 0.5, n_samples),
        'rgb': np.random.uniform(0.2, 0.6, n_samples)
    }
    
    config = FusionConfig(
        method=FusionMethod.LATE_WEIGHTED,
        use_uncertainty=True
    )
    fusion = MultimodalFusion(config)
    
    result = fusion.fuse_predictions(pred_hsi, pred_rgb, uncertainties=uncertainties)
    print(f"Uncertainty-aware weights: {result.weights}")
    
    print("\n" + "=" * 80)
    print("Multimodal Fusion System - Validation Complete!")
    print("=" * 80)
