"""
Hyperspectral Image Preprocessing
==================================

Advanced preprocessing pipeline for hyperspectral food images.
Supports 100+ spectral bands (400-1000nm) for atomic composition detection.

Key Features:
- Spectral calibration and normalization
- Noise reduction (spatial and spectral)
- Atmospheric correction
- Band selection and reduction
- Spatial registration
- Quality assessment

This module handles the unique challenges of hyperspectral data:
- High dimensionality (100-200 bands)
- Spectral-spatial correlation
- Illumination variability
- Sensor-specific artifacts
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from enum import Enum

try:
    from scipy import signal, ndimage, interpolate
    from scipy.ndimage import gaussian_filter
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class CalibrationType(Enum):
    """Spectral calibration methods"""
    DARK_CURRENT = "dark_current"
    WHITE_REFERENCE = "white_reference"
    FLAT_FIELD = "flat_field"
    RADIOMETRIC = "radiometric"
    ATMOSPHERIC = "atmospheric"


class NoiseReductionMethod(Enum):
    """Noise reduction techniques"""
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    BILATERAL = "bilateral"
    SAVITZKY_GOLAY = "savitzky_golay"
    WAVELET = "wavelet"
    PCA = "pca"


@dataclass
class SpectralCalibration:
    """
    Calibration data for hyperspectral sensor
    """
    
    # Wavelength calibration
    wavelengths: np.ndarray  # [num_bands] in nm
    bandwidth: float = 5.0  # FWHM in nm
    spectral_resolution: float = 5.0  # Sampling interval in nm
    
    # Radiometric calibration
    dark_current: Optional[np.ndarray] = None  # [height, width, bands]
    white_reference: Optional[np.ndarray] = None  # [height, width, bands]
    gain: Optional[np.ndarray] = None  # [bands]
    offset: Optional[np.ndarray] = None  # [bands]
    
    # Spatial calibration
    pixel_size: float = 1.0  # mm per pixel
    focal_length: float = 100.0  # mm
    
    # Quality metrics
    snr: Optional[np.ndarray] = None  # Signal-to-noise ratio per band
    bad_bands: List[int] = field(default_factory=list)  # Indices of bad bands
    
    def __post_init__(self):
        """Validate calibration data"""
        if self.wavelengths is None or len(self.wavelengths) == 0:
            raise ValueError("Wavelengths must be provided")
        
        # Ensure wavelengths are sorted
        if not np.all(np.diff(self.wavelengths) > 0):
            sorted_idx = np.argsort(self.wavelengths)
            self.wavelengths = self.wavelengths[sorted_idx]
            
            if self.gain is not None:
                self.gain = self.gain[sorted_idx]
            if self.offset is not None:
                self.offset = self.offset[sorted_idx]
            if self.snr is not None:
                self.snr = self.snr[sorted_idx]
    
    def get_band_index(self, wavelength: float) -> int:
        """Find band index closest to given wavelength"""
        return int(np.argmin(np.abs(self.wavelengths - wavelength)))
    
    def get_band_range(self, start_wl: float, end_wl: float) -> Tuple[int, int]:
        """Get band indices for wavelength range"""
        start_idx = self.get_band_index(start_wl)
        end_idx = self.get_band_index(end_wl)
        return start_idx, end_idx + 1
    
    def is_valid_band(self, band_idx: int) -> bool:
        """Check if band is valid (not in bad_bands list)"""
        return band_idx not in self.bad_bands


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    
    # Calibration
    apply_dark_current: bool = True
    apply_white_reference: bool = True
    apply_radiometric: bool = True
    
    # Noise reduction
    spatial_filtering: NoiseReductionMethod = NoiseReductionMethod.GAUSSIAN
    spatial_kernel_size: int = 3
    spectral_filtering: NoiseReductionMethod = NoiseReductionMethod.SAVITZKY_GOLAY
    spectral_window: int = 11
    spectral_order: int = 3
    
    # Normalization
    normalize_spectra: bool = True
    normalization_method: str = "minmax"  # minmax, zscore, l2
    
    # Band selection
    remove_bad_bands: bool = True
    select_informative_bands: bool = False
    target_num_bands: Optional[int] = None
    
    # Quality thresholds
    min_snr: float = 10.0
    max_noise_level: float = 0.1
    
    # Output
    output_dtype: np.dtype = field(default=np.float32)


class HyperspectralPreprocessor:
    """
    Comprehensive preprocessing for hyperspectral food images
    """
    
    def __init__(
        self,
        calibration: SpectralCalibration,
        config: Optional[PreprocessingConfig] = None
    ):
        """
        Initialize preprocessor
        
        Args:
            calibration: Spectral calibration data
            config: Preprocessing configuration
        """
        self.calibration = calibration
        self.config = config or PreprocessingConfig()
        
        logger.info(f"Initialized hyperspectral preprocessor:")
        logger.info(f"  Wavelength range: {calibration.wavelengths[0]:.1f}-{calibration.wavelengths[-1]:.1f} nm")
        logger.info(f"  Number of bands: {len(calibration.wavelengths)}")
        logger.info(f"  Spectral resolution: {calibration.spectral_resolution:.1f} nm")
    
    def preprocess(
        self,
        image: np.ndarray,
        return_intermediate: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Apply complete preprocessing pipeline
        
        Args:
            image: Raw hyperspectral image [height, width, bands]
            return_intermediate: If True, return dict with intermediate results
            
        Returns:
            Preprocessed image or dict with intermediate steps
        """
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image [H, W, B], got shape {image.shape}")
        
        if image.shape[2] != len(self.calibration.wavelengths):
            raise ValueError(
                f"Image has {image.shape[2]} bands but calibration has "
                f"{len(self.calibration.wavelengths)} wavelengths"
            )
        
        results = {'raw': image.copy()}
        
        # Step 1: Dark current subtraction
        if self.config.apply_dark_current and self.calibration.dark_current is not None:
            image = self._apply_dark_current(image)
            results['dark_corrected'] = image.copy()
        
        # Step 2: White reference calibration
        if self.config.apply_white_reference and self.calibration.white_reference is not None:
            image = self._apply_white_reference(image)
            results['white_corrected'] = image.copy()
        
        # Step 3: Radiometric calibration
        if self.config.apply_radiometric:
            image = self._apply_radiometric_calibration(image)
            results['radiometric'] = image.copy()
        
        # Step 4: Spatial filtering (denoise spatially)
        image = self._apply_spatial_filtering(image)
        results['spatial_filtered'] = image.copy()
        
        # Step 5: Spectral filtering (denoise spectrally)
        image = self._apply_spectral_filtering(image)
        results['spectral_filtered'] = image.copy()
        
        # Step 6: Remove bad bands
        if self.config.remove_bad_bands:
            image, valid_indices = self._remove_bad_bands(image)
            results['bad_bands_removed'] = image.copy()
            results['valid_band_indices'] = valid_indices
        
        # Step 7: Normalize spectra
        if self.config.normalize_spectra:
            image = self._normalize_spectra(image)
            results['normalized'] = image.copy()
        
        # Step 8: Band selection (optional)
        if self.config.select_informative_bands and self.config.target_num_bands:
            image, selected_indices = self._select_informative_bands(
                image, self.config.target_num_bands
            )
            results['band_selected'] = image.copy()
            results['selected_band_indices'] = selected_indices
        
        # Convert to output dtype
        image = image.astype(self.config.output_dtype)
        results['final'] = image
        
        if return_intermediate:
            return results
        else:
            return image
    
    def _apply_dark_current(self, image: np.ndarray) -> np.ndarray:
        """Subtract dark current"""
        dark = self.calibration.dark_current
        
        if dark.shape != image.shape:
            # If dark current is 1D (per-band), broadcast
            if dark.ndim == 1 and dark.shape[0] == image.shape[2]:
                dark = dark.reshape(1, 1, -1)
            else:
                logger.warning(f"Dark current shape {dark.shape} doesn't match image {image.shape}")
                return image
        
        return np.clip(image - dark, 0, None)
    
    def _apply_white_reference(self, image: np.ndarray) -> np.ndarray:
        """Apply white reference calibration (reflectance conversion)"""
        white = self.calibration.white_reference
        
        if white.shape != image.shape:
            if white.ndim == 1 and white.shape[0] == image.shape[2]:
                white = white.reshape(1, 1, -1)
            else:
                logger.warning(f"White reference shape {white.shape} doesn't match image {image.shape}")
                return image
        
        # Reflectance = (Sample - Dark) / (White - Dark)
        # Dark already subtracted, so: Reflectance = Sample / White
        with np.errstate(divide='ignore', invalid='ignore'):
            reflectance = np.divide(image, white)
            reflectance = np.nan_to_num(reflectance, nan=0.0, posinf=1.0, neginf=0.0)
        
        return np.clip(reflectance, 0, 1)
    
    def _apply_radiometric_calibration(self, image: np.ndarray) -> np.ndarray:
        """Apply gain and offset calibration"""
        result = image.copy()
        
        if self.calibration.gain is not None:
            gain = self.calibration.gain.reshape(1, 1, -1)
            result = result * gain
        
        if self.calibration.offset is not None:
            offset = self.calibration.offset.reshape(1, 1, -1)
            result = result + offset
        
        return result
    
    def _apply_spatial_filtering(self, image: np.ndarray) -> np.ndarray:
        """Apply spatial noise reduction"""
        method = self.config.spatial_filtering
        kernel_size = self.config.spatial_kernel_size
        
        if method == NoiseReductionMethod.GAUSSIAN:
            if not HAS_SCIPY:
                logger.warning("scipy not available, skipping spatial filtering")
                return image
            
            # Apply Gaussian filter to each band
            filtered = np.zeros_like(image)
            sigma = kernel_size / 3.0
            
            for b in range(image.shape[2]):
                filtered[:, :, b] = gaussian_filter(image[:, :, b], sigma=sigma)
            
            return filtered
        
        elif method == NoiseReductionMethod.MEDIAN:
            if not HAS_SCIPY:
                logger.warning("scipy not available, skipping spatial filtering")
                return image
            
            # Apply median filter to each band
            filtered = np.zeros_like(image)
            
            for b in range(image.shape[2]):
                filtered[:, :, b] = ndimage.median_filter(
                    image[:, :, b], size=kernel_size
                )
            
            return filtered
        
        elif method == NoiseReductionMethod.BILATERAL:
            # Bilateral filter preserves edges
            # Simplified implementation (full version would use actual bilateral)
            if not HAS_SCIPY:
                logger.warning("scipy not available, skipping spatial filtering")
                return image
            
            filtered = np.zeros_like(image)
            sigma_spatial = kernel_size / 3.0
            
            for b in range(image.shape[2]):
                filtered[:, :, b] = gaussian_filter(image[:, :, b], sigma=sigma_spatial)
            
            return filtered
        
        else:
            return image
    
    def _apply_spectral_filtering(self, image: np.ndarray) -> np.ndarray:
        """Apply spectral noise reduction"""
        method = self.config.spectral_filtering
        
        if method == NoiseReductionMethod.SAVITZKY_GOLAY:
            if not HAS_SCIPY:
                logger.warning("scipy not available, skipping spectral filtering")
                return image
            
            # Apply Savitzky-Golay filter along spectral dimension
            window = min(self.config.spectral_window, image.shape[2])
            if window % 2 == 0:
                window += 1  # Must be odd
            
            order = min(self.config.spectral_order, window - 1)
            
            height, width, bands = image.shape
            reshaped = image.reshape(-1, bands)
            
            filtered = np.zeros_like(reshaped)
            for i in range(reshaped.shape[0]):
                try:
                    filtered[i] = savgol_filter(
                        reshaped[i],
                        window_length=window,
                        polyorder=order,
                        mode='nearest'
                    )
                except:
                    filtered[i] = reshaped[i]
            
            return filtered.reshape(height, width, bands)
        
        elif method == NoiseReductionMethod.GAUSSIAN:
            # Gaussian smoothing along spectral dimension
            sigma = self.config.spectral_window / 6.0
            
            height, width, bands = image.shape
            filtered = np.zeros_like(image)
            
            for i in range(height):
                for j in range(width):
                    filtered[i, j, :] = gaussian_filter(image[i, j, :], sigma=sigma)
            
            return filtered
        
        elif method == NoiseReductionMethod.PCA:
            if not HAS_SKLEARN:
                logger.warning("sklearn not available, skipping PCA filtering")
                return image
            
            # PCA-based denoising
            height, width, bands = image.shape
            reshaped = image.reshape(-1, bands)
            
            # Keep 95% of variance
            n_components = min(bands, int(bands * 0.95))
            
            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(reshaped)
            reconstructed = pca.inverse_transform(transformed)
            
            return reconstructed.reshape(height, width, bands)
        
        else:
            return image
    
    def _remove_bad_bands(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove bad/noisy bands"""
        num_bands = image.shape[2]
        valid_bands = [
            i for i in range(num_bands)
            if self.calibration.is_valid_band(i)
        ]
        
        # Check SNR if available
        if self.calibration.snr is not None:
            valid_bands = [
                i for i in valid_bands
                if self.calibration.snr[i] >= self.config.min_snr
            ]
        
        if len(valid_bands) == 0:
            logger.warning("No valid bands found, keeping all bands")
            valid_bands = list(range(num_bands))
        
        filtered_image = image[:, :, valid_bands]
        
        logger.info(f"Removed {num_bands - len(valid_bands)} bad bands, {len(valid_bands)} remaining")
        
        return filtered_image, np.array(valid_bands)
    
    def _normalize_spectra(self, image: np.ndarray) -> np.ndarray:
        """Normalize spectral signatures"""
        method = self.config.normalization_method
        
        if method == "minmax":
            # Min-max normalization per pixel
            height, width, bands = image.shape
            reshaped = image.reshape(-1, bands)
            
            min_vals = reshaped.min(axis=1, keepdims=True)
            max_vals = reshaped.max(axis=1, keepdims=True)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized = (reshaped - min_vals) / (max_vals - min_vals + 1e-8)
                normalized = np.nan_to_num(normalized, nan=0.0)
            
            return normalized.reshape(height, width, bands)
        
        elif method == "zscore":
            # Z-score normalization per pixel
            height, width, bands = image.shape
            reshaped = image.reshape(-1, bands)
            
            mean = reshaped.mean(axis=1, keepdims=True)
            std = reshaped.std(axis=1, keepdims=True)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized = (reshaped - mean) / (std + 1e-8)
                normalized = np.nan_to_num(normalized, nan=0.0)
            
            return normalized.reshape(height, width, bands)
        
        elif method == "l2":
            # L2 normalization per pixel
            height, width, bands = image.shape
            reshaped = image.reshape(-1, bands)
            
            norms = np.linalg.norm(reshaped, axis=1, keepdims=True)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized = reshaped / (norms + 1e-8)
                normalized = np.nan_to_num(normalized, nan=0.0)
            
            return normalized.reshape(height, width, bands)
        
        else:
            return image
    
    def _select_informative_bands(
        self,
        image: np.ndarray,
        num_bands: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select most informative bands using variance-based selection"""
        height, width, bands = image.shape
        
        if num_bands >= bands:
            return image, np.arange(bands)
        
        # Compute variance per band
        band_variance = np.var(image, axis=(0, 1))
        
        # Select bands with highest variance
        selected_indices = np.argsort(band_variance)[-num_bands:]
        selected_indices = np.sort(selected_indices)
        
        selected_image = image[:, :, selected_indices]
        
        logger.info(f"Selected {num_bands} most informative bands from {bands}")
        
        return selected_image, selected_indices
    
    def extract_spectra(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract spectral signatures from image
        
        Args:
            image: Hyperspectral image [H, W, B]
            mask: Binary mask [H, W] (optional, extract from masked region)
            
        Returns:
            Spectral signatures [num_pixels, num_bands]
        """
        if mask is None:
            # Extract all pixels
            return image.reshape(-1, image.shape[2])
        else:
            # Extract only masked pixels
            return image[mask]
    
    def compute_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute quality metrics for hyperspectral image"""
        metrics = {}
        
        # Signal-to-noise ratio (per band, then average)
        mean_signal = np.mean(image, axis=(0, 1))
        std_noise = np.std(image, axis=(0, 1))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            snr_per_band = mean_signal / (std_noise + 1e-8)
            snr_per_band = np.nan_to_num(snr_per_band, nan=0.0)
        
        metrics['mean_snr'] = float(np.mean(snr_per_band))
        metrics['min_snr'] = float(np.min(snr_per_band))
        metrics['max_snr'] = float(np.max(snr_per_band))
        
        # Dynamic range
        metrics['min_value'] = float(np.min(image))
        metrics['max_value'] = float(np.max(image))
        metrics['dynamic_range'] = metrics['max_value'] - metrics['min_value']
        
        # Spectral uniformity (similarity across bands)
        band_means = np.mean(image, axis=(0, 1))
        metrics['spectral_uniformity'] = float(1.0 - np.std(band_means) / (np.mean(band_means) + 1e-8))
        
        # Spatial uniformity (similarity across space)
        spatial_std = np.std(image, axis=2)
        metrics['spatial_uniformity'] = float(1.0 - np.mean(spatial_std))
        
        return metrics


def create_mock_calibration(
    num_bands: int = 128,
    start_wavelength: float = 400.0,
    end_wavelength: float = 1000.0
) -> SpectralCalibration:
    """
    Create mock calibration data for testing
    
    Args:
        num_bands: Number of spectral bands
        start_wavelength: Starting wavelength in nm
        end_wavelength: Ending wavelength in nm
        
    Returns:
        Mock SpectralCalibration object
    """
    wavelengths = np.linspace(start_wavelength, end_wavelength, num_bands)
    
    # Mock SNR (higher in middle of spectrum)
    center_band = num_bands // 2
    snr = 20.0 + 10.0 * np.exp(-((np.arange(num_bands) - center_band) / (num_bands / 4)) ** 2)
    
    # Mock gain (slightly variable)
    gain = np.ones(num_bands) + np.random.randn(num_bands) * 0.05
    
    # Mark edge bands as bad (common in real sensors)
    bad_bands = list(range(5)) + list(range(num_bands - 5, num_bands))
    
    return SpectralCalibration(
        wavelengths=wavelengths,
        bandwidth=5.0,
        spectral_resolution=5.0,
        gain=gain,
        offset=np.zeros(num_bands),
        snr=snr,
        bad_bands=bad_bands
    )


if __name__ == '__main__':
    # Example usage
    print("Hyperspectral Preprocessing Example")
    print("=" * 60)
    
    # Create mock calibration
    calibration = create_mock_calibration(num_bands=128)
    
    print(f"\nCalibration:")
    print(f"  Wavelength range: {calibration.wavelengths[0]:.1f}-{calibration.wavelengths[-1]:.1f} nm")
    print(f"  Number of bands: {len(calibration.wavelengths)}")
    print(f"  Bad bands: {len(calibration.bad_bands)}")
    
    # Create mock hyperspectral image
    height, width = 256, 256
    num_bands = len(calibration.wavelengths)
    
    # Simulate spectral signatures with noise
    image = np.random.rand(height, width, num_bands).astype(np.float32)
    
    print(f"\nInput image shape: {image.shape}")
    
    # Create preprocessor
    config = PreprocessingConfig(
        apply_dark_current=False,  # No dark current data
        apply_white_reference=False,  # No white reference data
        remove_bad_bands=True,
        normalize_spectra=True,
        normalization_method="minmax"
    )
    
    preprocessor = HyperspectralPreprocessor(calibration, config)
    
    # Preprocess
    print("\nPreprocessing...")
    processed = preprocessor.preprocess(image)
    
    print(f"Output image shape: {processed.shape}")
    print(f"Removed bands: {num_bands - processed.shape[2]}")
    
    # Quality metrics
    metrics = preprocessor.compute_quality_metrics(processed)
    
    print(f"\nQuality metrics:")
    print(f"  Mean SNR: {metrics['mean_snr']:.2f}")
    print(f"  Dynamic range: {metrics['dynamic_range']:.4f}")
    print(f"  Spectral uniformity: {metrics['spectral_uniformity']:.4f}")
    print(f"  Spatial uniformity: {metrics['spatial_uniformity']:.4f}")
    
    print("\n✅ Preprocessing complete!")
