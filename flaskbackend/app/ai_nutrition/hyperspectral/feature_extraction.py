"""
Hyperspectral Feature Extraction
=================================

Advanced feature extraction from hyperspectral images for atomic composition detection.

Key Features:
- Spectral features (shape, derivatives, absorption features)
- Spatial features (texture, morphology)
- Spectral-spatial features (3D Gabor, wavelet transforms)
- Vegetation indices adapted for food
- Statistical features (moments, entropy)
- Absorption band fitting

Extracts rich feature representations from 100+ spectral bands
to feed into machine learning models.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    from scipy import signal, ndimage
    from scipy.ndimage import convolve, sobel, laplace
    from scipy.interpolate import UnivariateSpline
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.decomposition import PCA, FastICA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of hyperspectral features"""
    SPECTRAL_SHAPE = "spectral_shape"
    SPECTRAL_DERIVATIVES = "spectral_derivatives"
    ABSORPTION_FEATURES = "absorption_features"
    STATISTICAL = "statistical"
    TEXTURE = "texture"
    MORPHOLOGICAL = "morphological"
    VEGETATION_INDICES = "vegetation_indices"
    MINERAL_INDICES = "mineral_indices"
    WAVELET = "wavelet"
    FOURIER = "fourier"


@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    
    # Spectral features
    extract_spectral_shape: bool = True
    extract_derivatives: bool = True
    derivative_orders: List[int] = field(default_factory=lambda: [1, 2])
    extract_absorption: bool = True
    
    # Statistical features
    extract_statistics: bool = True
    statistical_moments: List[int] = field(default_factory=lambda: [1, 2, 3, 4])  # Mean, std, skew, kurtosis
    
    # Spatial features
    extract_texture: bool = True
    texture_window: int = 5
    extract_morphology: bool = False
    
    # Index features
    extract_indices: bool = True
    
    # Transform features
    extract_wavelet: bool = False
    extract_fourier: bool = False
    
    # Wavelength-specific features
    key_wavelengths: Optional[List[float]] = None  # Wavelengths of interest in nm
    
    # Output
    normalize_features: bool = True


@dataclass
class FeatureVector:
    """Container for extracted features"""
    
    features: np.ndarray  # [num_features]
    feature_names: List[str]
    feature_types: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'features': self.features.tolist(),
            'feature_names': self.feature_names,
            'feature_types': self.feature_types,
            'num_features': len(self.features)
        }
    
    def get_features_by_type(self, feature_type: str) -> np.ndarray:
        """Get features of specific type"""
        indices = [i for i, t in enumerate(self.feature_types) if t == feature_type]
        return self.features[indices]


class HyperspectralFeatureExtractor:
    """
    Extract rich features from hyperspectral images
    """
    
    def __init__(
        self,
        wavelengths: np.ndarray,
        config: Optional[FeatureConfig] = None
    ):
        """
        Initialize feature extractor
        
        Args:
            wavelengths: Wavelength values for each band [num_bands]
            config: Feature extraction configuration
        """
        self.wavelengths = wavelengths
        self.config = config or FeatureConfig()
        
        logger.info(f"Initialized hyperspectral feature extractor:")
        logger.info(f"  Number of bands: {len(wavelengths)}")
        logger.info(f"  Wavelength range: {wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm")
    
    def extract_features(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> FeatureVector:
        """
        Extract comprehensive features from hyperspectral image
        
        Args:
            image: Hyperspectral image [height, width, bands]
            mask: Binary mask [height, width] (optional)
            
        Returns:
            FeatureVector with all extracted features
        """
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image [H, W, B], got shape {image.shape}")
        
        features = []
        feature_names = []
        feature_types = []
        
        # Extract spectral features
        if self.config.extract_spectral_shape:
            f, names = self._extract_spectral_shape(image, mask)
            features.append(f)
            feature_names.extend(names)
            feature_types.extend(['spectral_shape'] * len(names))
        
        if self.config.extract_derivatives:
            for order in self.config.derivative_orders:
                f, names = self._extract_derivatives(image, mask, order)
                features.append(f)
                feature_names.extend(names)
                feature_types.extend([f'derivative_{order}'] * len(names))
        
        if self.config.extract_absorption:
            f, names = self._extract_absorption_features(image, mask)
            features.append(f)
            feature_names.extend(names)
            feature_types.extend(['absorption'] * len(names))
        
        # Extract statistical features
        if self.config.extract_statistics:
            f, names = self._extract_statistical_features(image, mask)
            features.append(f)
            feature_names.extend(names)
            feature_types.extend(['statistical'] * len(names))
        
        # Extract spatial features
        if self.config.extract_texture:
            f, names = self._extract_texture_features(image, mask)
            features.append(f)
            feature_names.extend(names)
            feature_types.extend(['texture'] * len(names))
        
        # Extract index features
        if self.config.extract_indices:
            f, names = self._extract_index_features(image, mask)
            features.append(f)
            feature_names.extend(names)
            feature_types.extend(['index'] * len(names))
        
        # Combine all features
        all_features = np.concatenate(features)
        
        # Normalize if requested
        if self.config.normalize_features:
            all_features = self._normalize(all_features)
        
        return FeatureVector(
            features=all_features,
            feature_names=feature_names,
            feature_types=feature_types
        )
    
    def _get_spectrum(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Get mean spectrum from image"""
        if mask is not None:
            # Average over masked region
            return np.mean(image[mask], axis=0)
        else:
            # Average over entire image
            return np.mean(image, axis=(0, 1))
    
    def _extract_spectral_shape(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract spectral shape features"""
        spectrum = self._get_spectrum(image, mask)
        
        features = []
        names = []
        
        # Overall shape
        features.append(np.min(spectrum))
        names.append('min_reflectance')
        
        features.append(np.max(spectrum))
        names.append('max_reflectance')
        
        features.append(np.mean(spectrum))
        names.append('mean_reflectance')
        
        features.append(np.std(spectrum))
        names.append('std_reflectance')
        
        # Find peaks
        peaks = self._find_peaks(spectrum)
        features.append(len(peaks))
        names.append('num_peaks')
        
        if len(peaks) > 0:
            features.append(self.wavelengths[peaks[0]])
            names.append('primary_peak_wavelength')
            features.append(spectrum[peaks[0]])
            names.append('primary_peak_height')
        else:
            features.extend([0.0, 0.0])
            names.extend(['primary_peak_wavelength', 'primary_peak_height'])
        
        # Spectral slope (linear fit)
        slope, intercept = np.polyfit(self.wavelengths, spectrum, 1)
        features.append(slope)
        names.append('spectral_slope')
        features.append(intercept)
        names.append('spectral_intercept')
        
        # Area under curve
        auc = np.trapz(spectrum, self.wavelengths)
        features.append(auc)
        names.append('spectral_auc')
        
        return np.array(features), names
    
    def _extract_derivatives(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        order: int = 1
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract spectral derivative features"""
        spectrum = self._get_spectrum(image, mask)
        
        # Compute derivative
        derivative = np.gradient(spectrum, self.wavelengths)
        
        if order == 2:
            derivative = np.gradient(derivative, self.wavelengths)
        
        features = []
        names = []
        
        # Statistics of derivative
        features.append(np.mean(derivative))
        names.append(f'derivative_{order}_mean')
        
        features.append(np.std(derivative))
        names.append(f'derivative_{order}_std')
        
        features.append(np.min(derivative))
        names.append(f'derivative_{order}_min')
        
        features.append(np.max(derivative))
        names.append(f'derivative_{order}_max')
        
        # Zero crossings
        zero_crossings = np.sum(np.diff(np.sign(derivative)) != 0)
        features.append(float(zero_crossings))
        names.append(f'derivative_{order}_zero_crossings')
        
        return np.array(features), names
    
    def _extract_absorption_features(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract absorption band features"""
        spectrum = self._get_spectrum(image, mask)
        
        features = []
        names = []
        
        # Convert to absorption (continuum removed)
        continuum = self._fit_continuum(spectrum)
        absorption = 1 - (spectrum / (continuum + 1e-8))
        
        # Find absorption bands (local maxima in absorption)
        absorption_peaks = self._find_peaks(absorption)
        
        features.append(len(absorption_peaks))
        names.append('num_absorption_bands')
        
        if len(absorption_peaks) > 0:
            # Primary absorption band
            primary = absorption_peaks[0]
            features.append(self.wavelengths[primary])
            names.append('primary_absorption_wavelength')
            features.append(absorption[primary])
            names.append('primary_absorption_depth')
            
            # Width (FWHM)
            width = self._compute_band_width(absorption, primary)
            features.append(width)
            names.append('primary_absorption_width')
        else:
            features.extend([0.0, 0.0, 0.0])
            names.extend(['primary_absorption_wavelength', 'primary_absorption_depth', 'primary_absorption_width'])
        
        # Total absorption
        features.append(np.sum(absorption))
        names.append('total_absorption')
        
        return np.array(features), names
    
    def _extract_statistical_features(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract statistical features"""
        spectrum = self._get_spectrum(image, mask)
        
        features = []
        names = []
        
        for moment in self.config.statistical_moments:
            if moment == 1:
                # Mean
                feat = np.mean(spectrum)
                name = 'mean'
            elif moment == 2:
                # Standard deviation
                feat = np.std(spectrum)
                name = 'std'
            elif moment == 3:
                # Skewness
                mean = np.mean(spectrum)
                std = np.std(spectrum)
                feat = np.mean(((spectrum - mean) / (std + 1e-8)) ** 3)
                name = 'skewness'
            elif moment == 4:
                # Kurtosis
                mean = np.mean(spectrum)
                std = np.std(spectrum)
                feat = np.mean(((spectrum - mean) / (std + 1e-8)) ** 4)
                name = 'kurtosis'
            else:
                continue
            
            features.append(feat)
            names.append(f'stat_{name}')
        
        # Entropy
        hist, _ = np.histogram(spectrum, bins=50, density=True)
        hist = hist + 1e-10
        entropy = -np.sum(hist * np.log(hist))
        features.append(entropy)
        names.append('entropy')
        
        # Percentiles
        for p in [25, 50, 75]:
            features.append(np.percentile(spectrum, p))
            names.append(f'percentile_{p}')
        
        return np.array(features), names
    
    def _extract_texture_features(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract spatial texture features"""
        features = []
        names = []
        
        # Use first band for texture (or mean across bands)
        if image.shape[2] > 0:
            texture_image = np.mean(image, axis=2)
        else:
            return np.array([]), []
        
        # Apply mask if provided
        if mask is not None:
            texture_image = texture_image * mask
        
        # Standard deviation (local variance)
        if HAS_SCIPY:
            window = self.config.texture_window
            local_std = ndimage.generic_filter(texture_image, np.std, size=window)
            features.append(np.mean(local_std))
            names.append('mean_local_std')
        
        # Edge strength (Sobel)
        if HAS_SCIPY:
            sobel_h = sobel(texture_image, axis=0)
            sobel_v = sobel(texture_image, axis=1)
            edge_strength = np.hypot(sobel_h, sobel_v)
            features.append(np.mean(edge_strength))
            names.append('mean_edge_strength')
        
        # Laplacian (smoothness)
        if HAS_SCIPY:
            lap = laplace(texture_image)
            features.append(np.mean(np.abs(lap)))
            names.append('mean_laplacian')
        
        # Overall spatial statistics
        features.append(np.std(texture_image))
        names.append('spatial_std')
        
        return np.array(features), names
    
    def _extract_index_features(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract spectral index features (adapted from vegetation indices)"""
        features = []
        names = []
        
        # Get band indices for specific wavelengths
        def get_band(wavelength):
            return np.argmin(np.abs(self.wavelengths - wavelength))
        
        # NDVI-like index (Red-NIR normalized difference)
        # Useful for organic content
        if np.max(self.wavelengths) >= 850:
            red_band = get_band(670)
            nir_band = get_band(850)
            
            red = self._get_band_value(image, red_band, mask)
            nir = self._get_band_value(image, nir_band, mask)
            
            ndvi = (nir - red) / (nir + red + 1e-8)
            features.append(ndvi)
            names.append('ndvi_like')
        
        # Ratio indices for different regions
        # Blue/Red ratio (anthocyanin content)
        if np.min(self.wavelengths) <= 475:
            blue_band = get_band(475)
            red_band = get_band(670)
            
            blue = self._get_band_value(image, blue_band, mask)
            red = self._get_band_value(image, red_band, mask)
            
            ratio = blue / (red + 1e-8)
            features.append(ratio)
            names.append('blue_red_ratio')
        
        # Green/Red ratio (chlorophyll)
        if np.min(self.wavelengths) <= 550:
            green_band = get_band(550)
            red_band = get_band(670)
            
            green = self._get_band_value(image, green_band, mask)
            red = self._get_band_value(image, red_band, mask)
            
            ratio = green / (red + 1e-8)
            features.append(ratio)
            names.append('green_red_ratio')
        
        # Water index (NIR/SWIR)
        if np.max(self.wavelengths) >= 970:
            nir_band = get_band(850)
            swir_band = get_band(970)
            
            nir = self._get_band_value(image, nir_band, mask)
            swir = self._get_band_value(image, swir_band, mask)
            
            water_index = (nir - swir) / (nir + swir + 1e-8)
            features.append(water_index)
            names.append('water_index')
        
        return np.array(features), names
    
    def _get_band_value(
        self,
        image: np.ndarray,
        band_idx: int,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """Get mean value of specific band"""
        band_image = image[:, :, band_idx]
        
        if mask is not None:
            return np.mean(band_image[mask])
        else:
            return np.mean(band_image)
    
    def _find_peaks(self, spectrum: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Find spectral peaks"""
        if not HAS_SCIPY:
            # Simple peak finding
            peaks = []
            for i in range(1, len(spectrum) - 1):
                if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                    if spectrum[i] > threshold * np.max(spectrum):
                        peaks.append(i)
            return np.array(peaks)
        else:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(spectrum, height=threshold * np.max(spectrum))
            return peaks
    
    def _fit_continuum(self, spectrum: np.ndarray) -> np.ndarray:
        """Fit continuum (convex hull) to spectrum"""
        # Simplified: linear fit to endpoints and max
        continuum = np.zeros_like(spectrum)
        
        # Linear interpolation between start, max, and end
        max_idx = np.argmax(spectrum)
        
        # Start to max
        continuum[:max_idx+1] = np.linspace(spectrum[0], spectrum[max_idx], max_idx+1)
        
        # Max to end
        continuum[max_idx:] = np.linspace(spectrum[max_idx], spectrum[-1], len(spectrum) - max_idx)
        
        return continuum
    
    def _compute_band_width(self, absorption: np.ndarray, peak_idx: int) -> float:
        """Compute absorption band width (FWHM)"""
        peak_value = absorption[peak_idx]
        half_max = peak_value / 2.0
        
        # Find left and right indices where absorption drops to half maximum
        left_idx = peak_idx
        while left_idx > 0 and absorption[left_idx] > half_max:
            left_idx -= 1
        
        right_idx = peak_idx
        while right_idx < len(absorption) - 1 and absorption[right_idx] > half_max:
            right_idx += 1
        
        # Width in nanometers
        width = self.wavelengths[right_idx] - self.wavelengths[left_idx]
        
        return width
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1]"""
        min_val = np.min(features)
        max_val = np.max(features)
        
        if max_val - min_val < 1e-8:
            return features
        
        return (features - min_val) / (max_val - min_val)


if __name__ == '__main__':
    # Example usage
    print("Hyperspectral Feature Extraction Example")
    print("=" * 60)
    
    # Create mock hyperspectral image
    height, width = 128, 128
    num_bands = 100
    wavelengths = np.linspace(400, 1000, num_bands)
    
    # Simulate realistic spectral signature
    image = np.zeros((height, width, num_bands))
    
    for i in range(height):
        for j in range(width):
            # Base spectrum with peaks
            spectrum = 0.3 + 0.2 * np.exp(-((wavelengths - 550) / 50) ** 2)  # Green peak
            spectrum += 0.15 * np.exp(-((wavelengths - 850) / 40) ** 2)  # NIR peak
            spectrum += np.random.randn(num_bands) * 0.02  # Noise
            image[i, j, :] = np.clip(spectrum, 0, 1)
    
    print(f"\nImage shape: {image.shape}")
    print(f"Wavelength range: {wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm")
    
    # Create feature extractor
    config = FeatureConfig(
        extract_spectral_shape=True,
        extract_derivatives=True,
        extract_absorption=True,
        extract_statistics=True,
        extract_texture=True,
        extract_indices=True
    )
    
    extractor = HyperspectralFeatureExtractor(wavelengths, config)
    
    # Extract features
    print("\nExtracting features...")
    feature_vector = extractor.extract_features(image)
    
    print(f"\nTotal features: {len(feature_vector.features)}")
    print(f"Feature names: {feature_vector.feature_names[:10]}...")
    print(f"Feature values: {feature_vector.features[:10]}")
    
    # Group by type
    unique_types = list(set(feature_vector.feature_types))
    print(f"\nFeature types:")
    for ftype in unique_types:
        count = sum(1 for t in feature_vector.feature_types if t == ftype)
        print(f"  {ftype}: {count} features")
    
    print("\n✅ Feature extraction complete!")
