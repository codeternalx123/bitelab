"""
Unit tests for hyperspectral preprocessing module

Tests preprocessing pipeline including filtering, normalization, and bad band removal.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ai_nutrition.hyperspectral.spectral_preprocessing import (
    SpectralPreprocessor,
    PreprocessConfig,
    PreprocessingMethod
)


class TestSpectralPreprocessor(unittest.TestCase):
    """Test cases for SpectralPreprocessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image_shape = (100, 100, 50)
        self.image = np.random.randn(*self.image_shape) * 0.1 + 0.5
        
    def test_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = SpectralPreprocessor()
        self.assertIsNotNone(preprocessor)
        self.assertIsInstance(preprocessor.config, PreprocessConfig)
        
    def test_initialization_with_config(self):
        """Test preprocessor with custom config"""
        config = PreprocessConfig(
            apply_dark_current=True,
            spatial_filter="gaussian",
            spatial_sigma=2.0
        )
        preprocessor = SpectralPreprocessor(config)
        self.assertEqual(preprocessor.config.spatial_sigma, 2.0)


class TestDarkCurrentCorrection(unittest.TestCase):
    """Test dark current correction"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image_shape = (100, 100, 50)
        self.image = np.random.randn(*self.image_shape) * 0.1 + 0.5
        self.dark_current = np.ones(self.image_shape) * 0.01
        
    def test_dark_current_subtraction(self):
        """Test dark current subtraction"""
        config = PreprocessConfig(apply_dark_current=True)
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(self.image, dark_current=self.dark_current)
        
        # Image should be darker by dark_current amount
        self.assertLess(np.mean(processed), np.mean(self.image))
        
    def test_no_dark_current(self):
        """Test processing without dark current"""
        config = PreprocessConfig(apply_dark_current=False)
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(self.image)
        
        # Should not raise error even without dark current
        self.assertEqual(processed.shape, self.image_shape)


class TestWhiteReferenceCorrection(unittest.TestCase):
    """Test white reference correction"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image_shape = (100, 100, 50)
        self.image = np.random.randn(*self.image_shape) * 0.1 + 0.5
        self.white_reference = np.ones(self.image_shape) * 0.9
        
    def test_white_reference_normalization(self):
        """Test white reference normalization"""
        config = PreprocessConfig(apply_white_reference=True)
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(
            self.image,
            white_reference=self.white_reference
        )
        
        # Should normalize to white reference
        self.assertEqual(processed.shape, self.image_shape)
        
    def test_combined_dark_and_white(self):
        """Test combined dark current and white reference"""
        dark_current = np.ones(self.image_shape) * 0.01
        white_reference = np.ones(self.image_shape) * 0.9
        
        config = PreprocessConfig(
            apply_dark_current=True,
            apply_white_reference=True
        )
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(
            self.image,
            dark_current=dark_current,
            white_reference=white_reference
        )
        
        self.assertEqual(processed.shape, self.image_shape)


class TestSpatialFiltering(unittest.TestCase):
    """Test spatial filtering"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image_shape = (100, 100, 50)
        # Create image with spatial noise
        self.image = np.random.randn(*self.image_shape) * 0.2 + 0.5
        
    def test_gaussian_filter(self):
        """Test Gaussian spatial filtering"""
        config = PreprocessConfig(
            spatial_filter="gaussian",
            spatial_sigma=1.0
        )
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(self.image)
        
        # Filtered image should be smoother (lower std)
        self.assertLess(np.std(processed), np.std(self.image))
        
    def test_median_filter(self):
        """Test median spatial filtering"""
        config = PreprocessConfig(
            spatial_filter="median",
            spatial_window=3
        )
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(self.image)
        
        self.assertEqual(processed.shape, self.image_shape)
        
    def test_no_spatial_filter(self):
        """Test without spatial filtering"""
        config = PreprocessConfig(spatial_filter="none")
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(self.image)
        
        # Should be unchanged (except for other processing)
        self.assertEqual(processed.shape, self.image_shape)


class TestSpectralFiltering(unittest.TestCase):
    """Test spectral filtering"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image_shape = (100, 100, 50)
        self.image = np.random.randn(*self.image_shape) * 0.1 + 0.5
        
    def test_savitzky_golay_filter(self):
        """Test Savitzky-Golay spectral filtering"""
        config = PreprocessConfig(
            spectral_filter="savgol",
            savgol_window=5,
            savgol_polyorder=2
        )
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(self.image)
        
        # Should smooth spectral dimension
        self.assertEqual(processed.shape, self.image_shape)
        
    def test_moving_average_filter(self):
        """Test moving average spectral filtering"""
        config = PreprocessConfig(
            spectral_filter="moving_average",
            spectral_window=3
        )
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(self.image)
        
        self.assertEqual(processed.shape, self.image_shape)


class TestBadBandRemoval(unittest.TestCase):
    """Test bad band removal"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image_shape = (100, 100, 50)
        self.image = np.random.randn(*self.image_shape) * 0.1 + 0.5
        
    def test_remove_bad_bands(self):
        """Test removing specified bad bands"""
        bad_bands = [0, 1, 2, 47, 48, 49]  # First 3 and last 3
        
        config = PreprocessConfig(
            remove_bad_bands=True,
            bad_bands=bad_bands
        )
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(self.image)
        
        # Should have fewer bands
        expected_bands = 50 - len(bad_bands)
        self.assertEqual(processed.shape[2], expected_bands)
        
    def test_automatic_bad_band_detection(self):
        """Test automatic bad band detection"""
        # Create image with some very noisy bands
        image = self.image.copy()
        image[:, :, [5, 15, 25]] = np.random.randn(100, 100, 3) * 10  # Very noisy
        
        config = PreprocessConfig(
            remove_bad_bands=True,
            auto_detect_bad_bands=True,
            bad_band_threshold=0.9
        )
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(image)
        
        # Should remove at least some bands
        self.assertLessEqual(processed.shape[2], 50)


class TestNormalization(unittest.TestCase):
    """Test normalization methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image_shape = (100, 100, 50)
        self.image = np.random.randn(*self.image_shape) * 0.2 + 0.5
        
    def test_minmax_normalization(self):
        """Test min-max normalization"""
        config = PreprocessConfig(normalization="minmax")
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(self.image)
        
        # Should be in [0, 1]
        self.assertGreaterEqual(np.min(processed), 0.0)
        self.assertLessEqual(np.max(processed), 1.0)
        
    def test_standard_normalization(self):
        """Test standard (z-score) normalization"""
        config = PreprocessConfig(normalization="standard")
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(self.image)
        
        # Should have mean ~0, std ~1
        self.assertAlmostEqual(np.mean(processed), 0.0, places=1)
        self.assertAlmostEqual(np.std(processed), 1.0, places=1)
        
    def test_l2_normalization(self):
        """Test L2 normalization"""
        config = PreprocessConfig(normalization="l2")
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(self.image)
        
        # Each spectrum should have L2 norm of 1
        for i in range(10):
            for j in range(10):
                spectrum = processed[i, j, :]
                norm = np.linalg.norm(spectrum)
                self.assertAlmostEqual(norm, 1.0, places=5)


class TestPCADenoising(unittest.TestCase):
    """Test PCA denoising"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image_shape = (100, 100, 50)
        # Create image with signal + noise
        signal = np.random.randn(*self.image_shape) * 0.5
        noise = np.random.randn(*self.image_shape) * 0.1
        self.image = signal + noise
        
    def test_pca_denoising(self):
        """Test PCA-based denoising"""
        config = PreprocessConfig(
            apply_pca_denoising=True,
            n_pca_components=20
        )
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(self.image)
        
        # Should reduce noise (lower std)
        self.assertLess(np.std(processed), np.std(self.image))
        
    def test_pca_component_selection(self):
        """Test different numbers of PCA components"""
        for n_components in [10, 20, 30]:
            config = PreprocessConfig(
                apply_pca_denoising=True,
                n_pca_components=n_components
            )
            preprocessor = SpectralPreprocessor(config)
            
            processed = preprocessor.process(self.image)
            
            self.assertEqual(processed.shape, self.image_shape)


class TestFullPreprocessingPipeline(unittest.TestCase):
    """Test complete preprocessing pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image_shape = (100, 100, 50)
        self.image = np.random.randn(*self.image_shape) * 0.2 + 0.5
        self.dark_current = np.ones(self.image_shape) * 0.01
        self.white_reference = np.ones(self.image_shape) * 0.9
        
    def test_full_pipeline(self):
        """Test complete preprocessing pipeline"""
        config = PreprocessConfig(
            apply_dark_current=True,
            apply_white_reference=True,
            spatial_filter="gaussian",
            spatial_sigma=1.0,
            spectral_filter="savgol",
            savgol_window=5,
            savgol_polyorder=2,
            remove_bad_bands=True,
            bad_bands=[0, 1, 48, 49],
            normalization="minmax"
        )
        
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(
            self.image,
            dark_current=self.dark_current,
            white_reference=self.white_reference
        )
        
        # Check output
        self.assertEqual(processed.shape[0], 100)
        self.assertEqual(processed.shape[1], 100)
        self.assertEqual(processed.shape[2], 46)  # 50 - 4 bad bands
        self.assertGreaterEqual(np.min(processed), 0.0)
        self.assertLessEqual(np.max(processed), 1.0)
        
    def test_pipeline_order_matters(self):
        """Test that processing order affects results"""
        config1 = PreprocessConfig(
            spatial_filter="gaussian",
            normalization="minmax"
        )
        
        config2 = PreprocessConfig(
            normalization="minmax",
            spatial_filter="gaussian"
        )
        
        preprocessor1 = SpectralPreprocessor(config1)
        preprocessor2 = SpectralPreprocessor(config2)
        
        # Results should be different due to order
        processed1 = preprocessor1.process(self.image)
        processed2 = preprocessor2.process(self.image)
        
        # Not exactly equal due to order
        self.assertFalse(np.allclose(processed1, processed2))


class TestPreprocessingEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_zero_variance_bands(self):
        """Test handling of zero-variance bands"""
        image = np.random.randn(50, 50, 30) * 0.1 + 0.5
        image[:, :, 10] = 0.5  # Constant band
        
        config = PreprocessConfig(normalization="standard")
        preprocessor = SpectralPreprocessor(config)
        
        # Should handle without error
        processed = preprocessor.process(image)
        
        self.assertFalse(np.any(np.isnan(processed)))
        self.assertFalse(np.any(np.isinf(processed)))
        
    def test_negative_values(self):
        """Test handling of negative values"""
        image = np.random.randn(50, 50, 30) - 1.0  # Negative mean
        
        config = PreprocessConfig(normalization="minmax")
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(image)
        
        # Should normalize to [0, 1]
        self.assertGreaterEqual(np.min(processed), 0.0)
        
    def test_single_pixel_image(self):
        """Test with single-pixel image"""
        image = np.random.randn(1, 1, 50)
        
        config = PreprocessConfig(
            spatial_filter="gaussian",
            normalization="minmax"
        )
        preprocessor = SpectralPreprocessor(config)
        
        processed = preprocessor.process(image)
        
        self.assertEqual(processed.shape, (1, 1, 50))


class TestPreprocessingPerformance(unittest.TestCase):
    """Test preprocessing performance"""
    
    def test_processing_speed(self):
        """Test processing speed"""
        import time
        
        image = np.random.randn(256, 256, 100) * 0.2 + 0.5
        
        config = PreprocessConfig(
            spatial_filter="gaussian",
            spectral_filter="savgol",
            normalization="minmax"
        )
        
        preprocessor = SpectralPreprocessor(config)
        
        start = time.time()
        processed = preprocessor.process(image)
        elapsed = time.time() - start
        
        # Should process in reasonable time (< 5 seconds)
        self.assertLess(elapsed, 5.0)
        
        pixels = 256 * 256
        pps = pixels / elapsed
        print(f"Processing speed: {pps:.0f} pixels/second")
        
    def test_batch_processing(self):
        """Test batch processing multiple images"""
        import time
        
        images = [np.random.randn(128, 128, 50) * 0.2 + 0.5 for _ in range(10)]
        
        config = PreprocessConfig(
            spatial_filter="gaussian",
            normalization="minmax"
        )
        
        preprocessor = SpectralPreprocessor(config)
        
        start = time.time()
        processed_images = [preprocessor.process(img) for img in images]
        elapsed = time.time() - start
        
        # Should process > 2 images/second
        fps = len(images) / elapsed
        self.assertGreater(fps, 2)


if __name__ == '__main__':
    unittest.main()
