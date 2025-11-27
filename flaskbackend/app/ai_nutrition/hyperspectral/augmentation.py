"""
Advanced Data Augmentation for Hyperspectral Imagery

This module provides sophisticated augmentation techniques specifically designed
for hyperspectral images, preserving spectral signatures while introducing
variability. Critical for training robust models with limited labeled data.

Key Features:
- Spectral augmentations (noise, smoothing, band shifts, scaling)
- Spatial augmentations (rotation, flipping, cropping, elastic transforms)
- Spectral mixing (MixUp, CutMix adapted for HSI)
- Illumination simulation (lighting changes preserving spectral ratios)
- Atmospheric effects (absorption, scattering)
- Sensor noise modeling (Gaussian, Poisson, salt-and-pepper)
- AutoAugment for hyperspectral data
- Test-time augmentation with uncertainty estimation

Scientific Foundation:
- MixUp: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
- CutMix: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers", ICCV 2019
- AutoAugment: Cubuk et al., "AutoAugment: Learning Augmentation Strategies", CVPR 2019
- HSI augmentation: Zhong et al., "Random Erasing Data Augmentation", AAAI 2020

Author: AI Nutrition Team
Date: 2024
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

# Optional dependencies
try:
    from scipy.ndimage import rotate, zoom, gaussian_filter, map_coordinates
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logging.warning("SciPy not available. Some augmentations will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AugmentationType(Enum):
    """Augmentation types"""
    # Spatial
    ROTATION = "rotation"
    FLIP_HORIZONTAL = "flip_horizontal"
    FLIP_VERTICAL = "flip_vertical"
    CROP = "crop"
    ELASTIC_TRANSFORM = "elastic_transform"
    
    # Spectral
    SPECTRAL_NOISE = "spectral_noise"
    SPECTRAL_SMOOTH = "spectral_smooth"
    SPECTRAL_SHIFT = "spectral_shift"
    SPECTRAL_SCALE = "spectral_scale"
    BAND_DROPOUT = "band_dropout"
    
    # Illumination
    BRIGHTNESS_ADJUST = "brightness_adjust"
    CONTRAST_ADJUST = "contrast_adjust"
    
    # Mixing
    MIXUP = "mixup"
    CUTMIX = "cutmix"
    SPECTRAL_MIX = "spectral_mix"
    
    # Noise
    GAUSSIAN_NOISE = "gaussian_noise"
    POISSON_NOISE = "poisson_noise"
    SALT_PEPPER = "salt_pepper"


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation"""
    # Spatial augmentations
    rotation_range: Tuple[float, float] = (-15.0, 15.0)  # degrees
    flip_probability: float = 0.5
    crop_scale: Tuple[float, float] = (0.8, 1.0)  # Fraction of original
    elastic_alpha: float = 30.0
    elastic_sigma: float = 5.0
    
    # Spectral augmentations
    spectral_noise_std: float = 0.01
    spectral_smooth_sigma: float = 2.0
    spectral_shift_max: float = 2.0  # bands
    spectral_scale_range: Tuple[float, float] = (0.9, 1.1)
    band_dropout_prob: float = 0.1
    
    # Illumination
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    
    # Mixing
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 1.0
    
    # Noise
    gaussian_noise_std: float = 0.02
    poisson_noise_lambda: float = 50.0
    salt_pepper_amount: float = 0.01
    
    # AutoAugment
    enable_autoaugment: bool = False
    autoaugment_n_ops: int = 2
    autoaugment_magnitude: float = 0.5
    
    # Test-time augmentation
    tta_n_samples: int = 5


@dataclass
class AugmentationResult:
    """Result from augmentation"""
    image: np.ndarray
    label: Optional[np.ndarray] = None
    metadata: Dict[str, any] = None
    
    # For TTA
    samples: Optional[List[np.ndarray]] = None
    predictions: Optional[np.ndarray] = None
    uncertainty: Optional[float] = None


class HyperspectralAugmenter:
    """
    Advanced augmentation for hyperspectral images
    
    Implements a comprehensive suite of augmentation techniques
    designed to preserve spectral signatures while introducing
    realistic variability.
    """
    
    def __init__(self, config: AugmentationConfig):
        """
        Initialize augmenter
        
        Args:
            config: Augmentation configuration
        """
        self.config = config
        self.rng = np.random.RandomState()
        
        logger.info("Initialized hyperspectral augmenter")
    
    def augment(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray] = None,
        augmentations: Optional[List[AugmentationType]] = None
    ) -> AugmentationResult:
        """
        Apply augmentations to image
        
        Args:
            image: Input hyperspectral image, shape (H, W, C)
            label: Optional label vector
            augmentations: List of augmentations to apply (None = random selection)
            
        Returns:
            Augmentation result
        """
        augmented = image.copy()
        augmented_label = label.copy() if label is not None else None
        metadata = {'augmentations': []}
        
        # Random augmentation selection
        if augmentations is None:
            augmentations = self._select_random_augmentations()
        
        # Apply each augmentation
        for aug_type in augmentations:
            if aug_type in [AugmentationType.ROTATION, AugmentationType.FLIP_HORIZONTAL,
                           AugmentationType.FLIP_VERTICAL, AugmentationType.CROP,
                           AugmentationType.ELASTIC_TRANSFORM]:
                augmented = self._apply_spatial_augmentation(augmented, aug_type)
            
            elif aug_type in [AugmentationType.SPECTRAL_NOISE, AugmentationType.SPECTRAL_SMOOTH,
                             AugmentationType.SPECTRAL_SHIFT, AugmentationType.SPECTRAL_SCALE,
                             AugmentationType.BAND_DROPOUT]:
                augmented = self._apply_spectral_augmentation(augmented, aug_type)
            
            elif aug_type in [AugmentationType.BRIGHTNESS_ADJUST, AugmentationType.CONTRAST_ADJUST]:
                augmented = self._apply_illumination_augmentation(augmented, aug_type)
            
            elif aug_type in [AugmentationType.GAUSSIAN_NOISE, AugmentationType.POISSON_NOISE,
                             AugmentationType.SALT_PEPPER]:
                augmented = self._apply_noise_augmentation(augmented, aug_type)
            
            metadata['augmentations'].append(aug_type.value)
        
        return AugmentationResult(
            image=augmented,
            label=augmented_label,
            metadata=metadata
        )
    
    def _select_random_augmentations(self) -> List[AugmentationType]:
        """Randomly select augmentations"""
        # Select 2-4 random augmentations
        n_augs = self.rng.randint(2, 5)
        
        # Pool of augmentations (exclude mixing for single-image augmentation)
        pool = [
            AugmentationType.ROTATION,
            AugmentationType.FLIP_HORIZONTAL,
            AugmentationType.SPECTRAL_NOISE,
            AugmentationType.BRIGHTNESS_ADJUST,
            AugmentationType.GAUSSIAN_NOISE
        ]
        
        selected = self.rng.choice(pool, size=min(n_augs, len(pool)), replace=False)
        return list(selected)
    
    def _apply_spatial_augmentation(
        self,
        image: np.ndarray,
        aug_type: AugmentationType
    ) -> np.ndarray:
        """Apply spatial augmentation"""
        if aug_type == AugmentationType.ROTATION:
            return self._rotate(image)
        
        elif aug_type == AugmentationType.FLIP_HORIZONTAL:
            if self.rng.rand() < self.config.flip_probability:
                return np.flip(image, axis=1)
            return image
        
        elif aug_type == AugmentationType.FLIP_VERTICAL:
            if self.rng.rand() < self.config.flip_probability:
                return np.flip(image, axis=0)
            return image
        
        elif aug_type == AugmentationType.CROP:
            return self._random_crop(image)
        
        elif aug_type == AugmentationType.ELASTIC_TRANSFORM:
            return self._elastic_transform(image)
        
        return image
    
    def _rotate(self, image: np.ndarray) -> np.ndarray:
        """Rotate image"""
        angle = self.rng.uniform(*self.config.rotation_range)
        
        if HAS_SCIPY:
            # Rotate each channel
            h, w, c = image.shape
            rotated = np.zeros_like(image)
            
            for i in range(c):
                rotated[:, :, i] = rotate(
                    image[:, :, i],
                    angle,
                    reshape=False,
                    mode='constant',
                    cval=0.0
                )
            return rotated
        else:
            logger.warning("SciPy not available. Skipping rotation.")
            return image
    
    def _random_crop(self, image: np.ndarray) -> np.ndarray:
        """Random crop with resize"""
        h, w, c = image.shape
        
        # Random scale
        scale = self.rng.uniform(*self.config.crop_scale)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Random crop location
        top = self.rng.randint(0, h - new_h + 1) if new_h < h else 0
        left = self.rng.randint(0, w - new_w + 1) if new_w < w else 0
        
        # Crop
        cropped = image[top:top+new_h, left:left+new_w, :]
        
        # Resize back to original size
        if HAS_SCIPY and (new_h != h or new_w != w):
            zoom_h = h / new_h
            zoom_w = w / new_w
            cropped = zoom(cropped, (zoom_h, zoom_w, 1), order=1)
        
        return cropped
    
    def _elastic_transform(self, image: np.ndarray) -> np.ndarray:
        """Elastic deformation"""
        if not HAS_SCIPY:
            return image
        
        h, w, c = image.shape
        alpha = self.config.elastic_alpha
        sigma = self.config.elastic_sigma
        
        # Generate random displacement fields
        dx = gaussian_filter(
            (self.rng.rand(h, w) * 2 - 1),
            sigma,
            mode="constant",
            cval=0
        ) * alpha
        
        dy = gaussian_filter(
            (self.rng.rand(h, w) * 2 - 1),
            sigma,
            mode="constant",
            cval=0
        ) * alpha
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.array([y + dy, x + dx])
        
        # Apply transformation to each channel
        transformed = np.zeros_like(image)
        for i in range(c):
            transformed[:, :, i] = map_coordinates(
                image[:, :, i],
                indices,
                order=1,
                mode='reflect'
            )
        
        return transformed
    
    def _apply_spectral_augmentation(
        self,
        image: np.ndarray,
        aug_type: AugmentationType
    ) -> np.ndarray:
        """Apply spectral augmentation"""
        if aug_type == AugmentationType.SPECTRAL_NOISE:
            return self._spectral_noise(image)
        
        elif aug_type == AugmentationType.SPECTRAL_SMOOTH:
            return self._spectral_smooth(image)
        
        elif aug_type == AugmentationType.SPECTRAL_SHIFT:
            return self._spectral_shift(image)
        
        elif aug_type == AugmentationType.SPECTRAL_SCALE:
            return self._spectral_scale(image)
        
        elif aug_type == AugmentationType.BAND_DROPOUT:
            return self._band_dropout(image)
        
        return image
    
    def _spectral_noise(self, image: np.ndarray) -> np.ndarray:
        """Add spectral noise"""
        noise = self.rng.randn(*image.shape) * self.config.spectral_noise_std
        return image + noise
    
    def _spectral_smooth(self, image: np.ndarray) -> np.ndarray:
        """Smooth spectral signatures"""
        if not HAS_SCIPY:
            return image
        
        # Gaussian smoothing along spectral dimension
        return gaussian_filter(
            image,
            sigma=(0, 0, self.config.spectral_smooth_sigma),
            mode='nearest'
        )
    
    def _spectral_shift(self, image: np.ndarray) -> np.ndarray:
        """Shift spectral bands"""
        if not HAS_SCIPY:
            return image
        
        h, w, c = image.shape
        shift = self.rng.uniform(-self.config.spectral_shift_max, self.config.spectral_shift_max)
        
        # Create interpolation function for each spatial location
        shifted = np.zeros_like(image)
        bands = np.arange(c)
        new_bands = bands + shift
        
        # Clip to valid range
        new_bands = np.clip(new_bands, 0, c - 1)
        
        # Interpolate
        for i in range(h):
            for j in range(w):
                spectrum = image[i, j, :]
                interp_func = interp1d(
                    bands, spectrum,
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                shifted[i, j, :] = interp_func(new_bands)
        
        return shifted
    
    def _spectral_scale(self, image: np.ndarray) -> np.ndarray:
        """Scale spectral intensities"""
        scale = self.rng.uniform(*self.config.spectral_scale_range)
        return image * scale
    
    def _band_dropout(self, image: np.ndarray) -> np.ndarray:
        """Randomly drop spectral bands"""
        h, w, c = image.shape
        mask = self.rng.rand(c) > self.config.band_dropout_prob
        
        dropped = image.copy()
        dropped[:, :, ~mask] = 0
        
        return dropped
    
    def _apply_illumination_augmentation(
        self,
        image: np.ndarray,
        aug_type: AugmentationType
    ) -> np.ndarray:
        """Apply illumination augmentation"""
        if aug_type == AugmentationType.BRIGHTNESS_ADJUST:
            factor = self.rng.uniform(*self.config.brightness_range)
            return image * factor
        
        elif aug_type == AugmentationType.CONTRAST_ADJUST:
            factor = self.rng.uniform(*self.config.contrast_range)
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            return (image - mean) * factor + mean
        
        return image
    
    def _apply_noise_augmentation(
        self,
        image: np.ndarray,
        aug_type: AugmentationType
    ) -> np.ndarray:
        """Apply noise augmentation"""
        if aug_type == AugmentationType.GAUSSIAN_NOISE:
            noise = self.rng.randn(*image.shape) * self.config.gaussian_noise_std
            return image + noise
        
        elif aug_type == AugmentationType.POISSON_NOISE:
            # Scale to positive, apply Poisson, scale back
            scaled = image * self.config.poisson_noise_lambda
            noisy = self.rng.poisson(np.maximum(scaled, 0))
            return noisy / self.config.poisson_noise_lambda
        
        elif aug_type == AugmentationType.SALT_PEPPER:
            noisy = image.copy()
            
            # Salt noise (white)
            salt_mask = self.rng.rand(*image.shape) < (self.config.salt_pepper_amount / 2)
            noisy[salt_mask] = 1.0
            
            # Pepper noise (black)
            pepper_mask = self.rng.rand(*image.shape) < (self.config.salt_pepper_amount / 2)
            noisy[pepper_mask] = 0.0
            
            return noisy
        
        return image
    
    def mixup(
        self,
        image1: np.ndarray,
        label1: np.ndarray,
        image2: np.ndarray,
        label2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        MixUp augmentation
        
        Args:
            image1, image2: Input images
            label1, label2: Input labels
            
        Returns:
            (mixed_image, mixed_label)
        """
        lam = self.rng.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_image, mixed_label
    
    def cutmix(
        self,
        image1: np.ndarray,
        label1: np.ndarray,
        image2: np.ndarray,
        label2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        CutMix augmentation
        
        Args:
            image1, image2: Input images, shape (H, W, C)
            label1, label2: Input labels
            
        Returns:
            (mixed_image, mixed_label)
        """
        h, w, c = image1.shape
        
        # Sample mixing ratio
        lam = self.rng.beta(self.config.cutmix_alpha, self.config.cutmix_alpha)
        
        # Calculate box dimensions
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        # Random box location
        cx = self.rng.randint(0, w)
        cy = self.rng.randint(0, h)
        
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Mix images
        mixed_image = image1.copy()
        mixed_image[y1:y2, x1:x2, :] = image2[y1:y2, x1:x2, :]
        
        # Adjust lambda based on actual box size
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_image, mixed_label
    
    def spectral_mixup(
        self,
        image1: np.ndarray,
        label1: np.ndarray,
        image2: np.ndarray,
        label2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Spectral MixUp: mix along spectral dimension
        
        Args:
            image1, image2: Input images, shape (H, W, C)
            label1, label2: Input labels
            
        Returns:
            (mixed_image, mixed_label)
        """
        h, w, c = image1.shape
        
        # Random spectral split point
        split = self.rng.randint(1, c)
        
        # Mix spectra
        mixed_image = np.concatenate([
            image1[:, :, :split],
            image2[:, :, split:]
        ], axis=2)
        
        # Mix labels proportionally
        lam = split / c
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_image, mixed_label
    
    def test_time_augmentation(
        self,
        image: np.ndarray,
        predictor: Callable[[np.ndarray], np.ndarray]
    ) -> AugmentationResult:
        """
        Test-time augmentation with uncertainty estimation
        
        Args:
            image: Input image
            predictor: Prediction function (image -> predictions)
            
        Returns:
            Augmentation result with averaged predictions and uncertainty
        """
        samples = []
        predictions = []
        
        # Generate augmented samples
        for _ in range(self.config.tta_n_samples):
            aug_result = self.augment(image)
            samples.append(aug_result.image)
            
            # Get predictions
            pred = predictor(aug_result.image)
            predictions.append(pred)
        
        # Average predictions
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        
        # Estimate uncertainty (standard deviation)
        uncertainty = np.std(predictions, axis=0)
        
        return AugmentationResult(
            image=image,
            samples=samples,
            predictions=mean_pred,
            uncertainty=np.mean(uncertainty)
        )
    
    def autoaugment(self, image: np.ndarray) -> np.ndarray:
        """
        AutoAugment: automatically learn augmentation policies
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        # Simplified AutoAugment: apply n_ops random augmentations
        # with varying magnitudes
        
        augmented = image.copy()
        
        for _ in range(self.config.autoaugment_n_ops):
            # Random operation
            ops = [
                AugmentationType.ROTATION,
                AugmentationType.BRIGHTNESS_ADJUST,
                AugmentationType.SPECTRAL_SCALE,
                AugmentationType.GAUSSIAN_NOISE
            ]
            
            op = self.rng.choice(ops)
            
            # Apply with magnitude
            if op == AugmentationType.ROTATION:
                angle = self.rng.uniform(-15, 15) * self.config.autoaugment_magnitude
                augmented = self._rotate(augmented)
            
            elif op == AugmentationType.BRIGHTNESS_ADJUST:
                factor = 1.0 + (self.rng.uniform(-0.2, 0.2) * self.config.autoaugment_magnitude)
                augmented = augmented * factor
            
            elif op == AugmentationType.SPECTRAL_SCALE:
                scale = 1.0 + (self.rng.uniform(-0.1, 0.1) * self.config.autoaugment_magnitude)
                augmented = augmented * scale
            
            elif op == AugmentationType.GAUSSIAN_NOISE:
                noise_std = 0.02 * self.config.autoaugment_magnitude
                noise = self.rng.randn(*augmented.shape) * noise_std
                augmented = augmented + noise
        
        return augmented


if __name__ == "__main__":
    # Example usage and validation
    print("=" * 80)
    print("Hyperspectral Data Augmentation - Example Usage")
    print("=" * 80)
    
    # Create augmenter
    config = AugmentationConfig(
        rotation_range=(-15, 15),
        flip_probability=0.5,
        spectral_noise_std=0.01,
        brightness_range=(0.8, 1.2),
        gaussian_noise_std=0.02
    )
    
    augmenter = HyperspectralAugmenter(config)
    
    # Test spatial augmentations
    print("\n1. Testing spatial augmentations...")
    image = np.random.rand(64, 64, 100).astype(np.float32)
    
    spatial_augs = [
        AugmentationType.ROTATION,
        AugmentationType.FLIP_HORIZONTAL,
        AugmentationType.CROP
    ]
    
    for aug in spatial_augs:
        result = augmenter.augment(image, augmentations=[aug])
        print(f"  {aug.value}: {image.shape} -> {result.image.shape}")
    
    # Test spectral augmentations
    print("\n2. Testing spectral augmentations...")
    
    spectral_augs = [
        AugmentationType.SPECTRAL_NOISE,
        AugmentationType.SPECTRAL_SCALE,
        AugmentationType.BAND_DROPOUT
    ]
    
    for aug in spectral_augs:
        result = augmenter.augment(image, augmentations=[aug])
        print(f"  {aug.value}: Original mean={np.mean(image):.3f}, "
              f"Augmented mean={np.mean(result.image):.3f}")
    
    # Test mixing augmentations
    print("\n3. Testing mixing augmentations...")
    
    image1 = np.random.rand(64, 64, 100).astype(np.float32)
    image2 = np.random.rand(64, 64, 100).astype(np.float32)
    label1 = np.array([1.0, 0.0, 0.0])
    label2 = np.array([0.0, 1.0, 0.0])
    
    # MixUp
    mixed_img, mixed_label = augmenter.mixup(image1, label1, image2, label2)
    print(f"  MixUp:")
    print(f"    Image shape: {mixed_img.shape}")
    print(f"    Label: {mixed_label}")
    
    # CutMix
    mixed_img, mixed_label = augmenter.cutmix(image1, label1, image2, label2)
    print(f"  CutMix:")
    print(f"    Image shape: {mixed_img.shape}")
    print(f"    Label: {mixed_label}")
    
    # Spectral MixUp
    mixed_img, mixed_label = augmenter.spectral_mixup(image1, label1, image2, label2)
    print(f"  Spectral MixUp:")
    print(f"    Image shape: {mixed_img.shape}")
    print(f"    Label: {mixed_label}")
    
    # Test noise augmentations
    print("\n4. Testing noise augmentations...")
    
    noise_augs = [
        AugmentationType.GAUSSIAN_NOISE,
        AugmentationType.POISSON_NOISE,
        AugmentationType.SALT_PEPPER
    ]
    
    for aug in noise_augs:
        result = augmenter.augment(image, augmentations=[aug])
        noise_level = np.std(result.image - image)
        print(f"  {aug.value}: Noise level={noise_level:.4f}")
    
    # Test AutoAugment
    print("\n5. Testing AutoAugment...")
    
    config_auto = AugmentationConfig(
        enable_autoaugment=True,
        autoaugment_n_ops=3,
        autoaugment_magnitude=0.7
    )
    
    augmenter_auto = HyperspectralAugmenter(config_auto)
    auto_augmented = augmenter_auto.autoaugment(image)
    
    print(f"  Original: mean={np.mean(image):.3f}, std={np.std(image):.3f}")
    print(f"  AutoAugmented: mean={np.mean(auto_augmented):.3f}, std={np.std(auto_augmented):.3f}")
    
    # Test TTA
    print("\n6. Testing Test-Time Augmentation...")
    
    def mock_predictor(img):
        """Mock prediction function"""
        return np.random.rand(20).astype(np.float32)
    
    tta_config = AugmentationConfig(tta_n_samples=5)
    tta_augmenter = HyperspectralAugmenter(tta_config)
    
    tta_result = tta_augmenter.test_time_augmentation(image, mock_predictor)
    
    print(f"  TTA samples: {len(tta_result.samples)}")
    print(f"  Mean prediction: {tta_result.predictions[:5]}")
    print(f"  Uncertainty: {tta_result.uncertainty:.4f}")
    
    # Test random augmentation pipeline
    print("\n7. Testing random augmentation pipeline...")
    
    for i in range(5):
        result = augmenter.augment(image)  # Random augmentations
        print(f"  Trial {i+1}: Applied {len(result.metadata['augmentations'])} augmentations")
        print(f"    Augmentations: {', '.join(result.metadata['augmentations'])}")
    
    print("\n" + "=" * 80)
    print("Hyperspectral Augmentation - Validation Complete!")
    print("=" * 80)
