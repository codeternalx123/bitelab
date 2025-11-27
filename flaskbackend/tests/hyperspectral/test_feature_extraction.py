"""
Unit tests for hyperspectral feature extraction module

Tests all 40+ feature extraction functions.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ai_nutrition.hyperspectral.feature_extraction import (
    SpectralFeatureExtractor,
    FeatureType,
    extract_spectral_shape_features,
    extract_derivative_features,
    extract_absorption_features,
    extract_spectral_indices
)


class TestSpectralFeatureExtractor(unittest.TestCase):
    """Test cases for SpectralFeatureExtractor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.2 + 0.5
        
    def test_initialization(self):
        """Test feature extractor initialization"""
        extractor = SpectralFeatureExtractor()
        self.assertIsNotNone(extractor)
        
    def test_extract_all_features(self):
        """Test extracting all feature types"""
        extractor = SpectralFeatureExtractor()
        features = extractor.extract(self.image)
        
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)


class TestSpectralShapeFeatures(unittest.TestCase):
    """Test spectral shape feature extraction"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.2 + 0.5
        
    def test_mean_spectrum(self):
        """Test mean spectrum calculation"""
        features = extract_spectral_shape_features(self.image)
        
        self.assertIn('mean', features)
        mean = features['mean']
        self.assertEqual(mean.shape, (100, 100))
        
    def test_std_spectrum(self):
        """Test standard deviation calculation"""
        features = extract_spectral_shape_features(self.image)
        
        self.assertIn('std', features)
        std = features['std']
        self.assertEqual(std.shape, (100, 100))
        self.assertTrue(np.all(std >= 0))  # Std is non-negative
        
    def test_skewness(self):
        """Test skewness calculation"""
        features = extract_spectral_shape_features(self.image)
        
        self.assertIn('skewness', features)
        skewness = features['skewness']
        self.assertEqual(skewness.shape, (100, 100))
        
    def test_kurtosis(self):
        """Test kurtosis calculation"""
        features = extract_spectral_shape_features(self.image)
        
        self.assertIn('kurtosis', features)
        kurtosis = features['kurtosis']
        self.assertEqual(kurtosis.shape, (100, 100))
        
    def test_min_max_features(self):
        """Test min/max feature extraction"""
        features = extract_spectral_shape_features(self.image)
        
        self.assertIn('min_value', features)
        self.assertIn('max_value', features)
        
        min_val = features['min_value']
        max_val = features['max_value']
        
        # Max should be >= min
        self.assertTrue(np.all(max_val >= min_val))
        
    def test_range_feature(self):
        """Test range (max - min) feature"""
        features = extract_spectral_shape_features(self.image)
        
        self.assertIn('range', features)
        range_val = features['range']
        
        self.assertTrue(np.all(range_val >= 0))


class TestDerivativeFeatures(unittest.TestCase):
    """Test derivative feature extraction"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create image with smooth spectral profiles
        self.image = np.zeros((50, 50, 50))
        for i in range(50):
            for j in range(50):
                # Gaussian-like spectrum
                x = np.linspace(-3, 3, 50)
                self.image[i, j, :] = np.exp(-x**2 / 2) + np.random.randn(50) * 0.01
                
    def test_first_derivative(self):
        """Test first derivative calculation"""
        features = extract_derivative_features(self.image)
        
        self.assertIn('first_derivative_mean', features)
        first_deriv = features['first_derivative_mean']
        self.assertEqual(first_deriv.shape, (50, 50))
        
    def test_first_derivative_properties(self):
        """Test first derivative has expected properties"""
        features = extract_derivative_features(self.image)
        
        # First derivative mean should be small for smooth spectra
        first_deriv = features['first_derivative_mean']
        self.assertLess(np.abs(np.mean(first_deriv)), 0.1)
        
    def test_second_derivative(self):
        """Test second derivative calculation"""
        features = extract_derivative_features(self.image)
        
        self.assertIn('second_derivative_mean', features)
        second_deriv = features['second_derivative_mean']
        self.assertEqual(second_deriv.shape, (50, 50))
        
    def test_derivative_std(self):
        """Test derivative standard deviation"""
        features = extract_derivative_features(self.image)
        
        self.assertIn('first_derivative_std', features)
        self.assertIn('second_derivative_std', features)
        
        first_std = features['first_derivative_std']
        second_std = features['second_derivative_std']
        
        # Std should be non-negative
        self.assertTrue(np.all(first_std >= 0))
        self.assertTrue(np.all(second_std >= 0))


class TestAbsorptionFeatures(unittest.TestCase):
    """Test absorption feature extraction"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create image with absorption features
        self.image = np.zeros((50, 50, 50))
        for i in range(50):
            for j in range(50):
                # Baseline + absorption feature
                baseline = np.linspace(1.0, 0.8, 50)
                
                # Add absorption at band 25
                x = np.arange(50)
                absorption = -0.3 * np.exp(-((x - 25) ** 2) / (2 * 3 ** 2))
                
                self.image[i, j, :] = baseline + absorption + np.random.randn(50) * 0.02
                
    def test_absorption_depth(self):
        """Test absorption depth calculation"""
        features = extract_absorption_features(self.image)
        
        self.assertIn('absorption_depth', features)
        depth = features['absorption_depth']
        self.assertEqual(depth.shape, (50, 50))
        
        # Should detect absorption
        self.assertGreater(np.mean(depth), 0.1)
        
    def test_absorption_position(self):
        """Test absorption position detection"""
        features = extract_absorption_features(self.image)
        
        self.assertIn('absorption_position', features)
        position = features['absorption_position']
        
        # Position should be near band 25
        mean_position = np.mean(position)
        self.assertGreater(mean_position, 20)
        self.assertLess(mean_position, 30)
        
    def test_absorption_width(self):
        """Test absorption width calculation"""
        features = extract_absorption_features(self.image)
        
        self.assertIn('absorption_width', features)
        width = features['absorption_width']
        
        # Width should be positive
        self.assertTrue(np.all(width > 0))
        
    def test_absorption_asymmetry(self):
        """Test absorption asymmetry calculation"""
        features = extract_absorption_features(self.image)
        
        self.assertIn('absorption_asymmetry', features)
        asymmetry = features['absorption_asymmetry']
        
        self.assertEqual(asymmetry.shape, (50, 50))


class TestSpectralIndices(unittest.TestCase):
    """Test spectral index calculation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.rand(50, 50, 50) * 0.5 + 0.3
        
    def test_ndvi_like_index(self):
        """Test NDVI-like index calculation"""
        # (band40 - band20) / (band40 + band20)
        indices = extract_spectral_indices(self.image)
        
        self.assertIn('ndvi_like', indices)
        ndvi = indices['ndvi_like']
        
        # NDVI should be in [-1, 1]
        self.assertTrue(np.all(ndvi >= -1))
        self.assertTrue(np.all(ndvi <= 1))
        
    def test_ratio_indices(self):
        """Test ratio indices"""
        indices = extract_spectral_indices(self.image)
        
        self.assertIn('ratio_1', indices)
        self.assertIn('ratio_2', indices)
        
        ratio1 = indices['ratio_1']
        
        # Ratios should be positive
        self.assertTrue(np.all(ratio1 > 0))
        
    def test_normalized_difference_indices(self):
        """Test normalized difference indices"""
        indices = extract_spectral_indices(self.image)
        
        # Should have several ND indices
        nd_keys = [k for k in indices.keys() if 'nd_' in k]
        self.assertGreater(len(nd_keys), 0)
        
        for key in nd_keys:
            nd = indices[key]
            # Normalized difference should be in [-1, 1]
            self.assertTrue(np.all(nd >= -1.1))  # Allow small numerical error
            self.assertTrue(np.all(nd <= 1.1))


class TestMultiScaleFeatures(unittest.TestCase):
    """Test multi-scale feature extraction"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.2 + 0.5
        
    def test_texture_features(self):
        """Test texture feature extraction"""
        extractor = SpectralFeatureExtractor(include_texture=True)
        features = extractor.extract(self.image)
        
        # Should have texture features
        texture_keys = [k for k in features.keys() if 'texture' in k]
        self.assertGreater(len(texture_keys), 0)
        
    def test_spatial_features(self):
        """Test spatial feature extraction"""
        extractor = SpectralFeatureExtractor(include_spatial=True)
        features = extractor.extract(self.image)
        
        # Should have spatial features
        spatial_keys = [k for k in features.keys() if 'spatial' in k]
        self.assertGreater(len(spatial_keys), 0)


class TestFeatureVectorization(unittest.TestCase):
    """Test feature vectorization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(50, 50, 50) * 0.2 + 0.5
        
    def test_feature_vector_shape(self):
        """Test feature vector has correct shape"""
        extractor = SpectralFeatureExtractor()
        features = extractor.extract(self.image)
        
        # Convert to vector
        vector = extractor.vectorize(features)
        
        # Should be 2D: (n_pixels, n_features)
        self.assertEqual(len(vector.shape), 2)
        self.assertEqual(vector.shape[0], 50 * 50)
        
    def test_feature_vector_no_nan(self):
        """Test feature vector has no NaN values"""
        extractor = SpectralFeatureExtractor()
        features = extractor.extract(self.image)
        vector = extractor.vectorize(features)
        
        self.assertFalse(np.any(np.isnan(vector)))
        
    def test_feature_vector_no_inf(self):
        """Test feature vector has no infinite values"""
        extractor = SpectralFeatureExtractor()
        features = extractor.extract(self.image)
        vector = extractor.vectorize(features)
        
        self.assertFalse(np.any(np.isinf(vector)))


class TestFeatureSelection(unittest.TestCase):
    """Test feature selection functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(50, 50, 50) * 0.2 + 0.5
        
    def test_select_feature_subset(self):
        """Test selecting subset of features"""
        extractor = SpectralFeatureExtractor()
        
        # Extract only specific features
        feature_types = [FeatureType.MEAN, FeatureType.STD]
        features = extractor.extract(self.image, feature_types=feature_types)
        
        # Should only have requested features
        self.assertIn('mean', features)
        self.assertIn('std', features)
        
    def test_exclude_features(self):
        """Test excluding specific features"""
        extractor = SpectralFeatureExtractor()
        
        # Extract all except derivatives
        features = extractor.extract(
            self.image,
            exclude=[FeatureType.FIRST_DERIVATIVE, FeatureType.SECOND_DERIVATIVE]
        )
        
        # Should not have derivative features
        deriv_keys = [k for k in features.keys() if 'derivative' in k]
        self.assertEqual(len(deriv_keys), 0)


class TestFeatureNormalization(unittest.TestCase):
    """Test feature normalization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(50, 50, 50) * 0.2 + 0.5
        
    def test_normalize_features(self):
        """Test feature normalization"""
        extractor = SpectralFeatureExtractor(normalize=True)
        features = extractor.extract(self.image)
        vector = extractor.vectorize(features)
        
        # Check each feature column is normalized
        for i in range(vector.shape[1]):
            col = vector[:, i]
            if np.std(col) > 0:  # Skip constant features
                # Should have mean ~0, std ~1
                self.assertAlmostEqual(np.mean(col), 0.0, places=1)
                self.assertAlmostEqual(np.std(col), 1.0, places=1)


class TestFeatureExtractionEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_single_band_image(self):
        """Test with single-band image"""
        image = np.random.randn(50, 50, 1)
        
        extractor = SpectralFeatureExtractor()
        features = extractor.extract(image)
        
        # Should handle gracefully
        self.assertIsInstance(features, dict)
        
    def test_constant_spectrum(self):
        """Test with constant spectrum"""
        image = np.ones((50, 50, 50)) * 0.5
        
        extractor = SpectralFeatureExtractor()
        features = extractor.extract(image)
        
        # Std should be zero or near-zero
        if 'std' in features:
            self.assertLess(np.mean(features['std']), 0.01)
            
    def test_negative_values(self):
        """Test with negative reflectance values"""
        image = np.random.randn(50, 50, 50) - 1.0  # Negative values
        
        extractor = SpectralFeatureExtractor()
        features = extractor.extract(image)
        
        # Should complete without error
        self.assertIsInstance(features, dict)
        
    def test_zero_division_protection(self):
        """Test protection against zero division"""
        image = np.zeros((50, 50, 50))
        image[:, :, 25] = 0.001  # Very small values
        
        indices = extract_spectral_indices(image)
        
        # Should not have inf values
        for key, value in indices.items():
            self.assertFalse(np.any(np.isinf(value)))


class TestFeatureExtractionPerformance(unittest.TestCase):
    """Test feature extraction performance"""
    
    def test_extraction_speed(self):
        """Test extraction speed for large images"""
        import time
        
        image = np.random.randn(256, 256, 100) * 0.2 + 0.5
        
        extractor = SpectralFeatureExtractor()
        
        start = time.time()
        features = extractor.extract(image)
        elapsed = time.time() - start
        
        # Should complete in < 10 seconds
        self.assertLess(elapsed, 10)
        
    def test_batch_extraction(self):
        """Test extracting features for multiple images"""
        import time
        
        images = [np.random.randn(128, 128, 50) * 0.2 + 0.5 for _ in range(10)]
        
        extractor = SpectralFeatureExtractor()
        
        start = time.time()
        for image in images:
            features = extractor.extract(image)
        elapsed = time.time() - start
        
        # Should process > 1 image/second
        fps = len(images) / elapsed
        self.assertGreater(fps, 1)


class TestFeatureImportance(unittest.TestCase):
    """Test feature importance ranking"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.2 + 0.5
        
    def test_rank_features(self):
        """Test ranking features by importance"""
        extractor = SpectralFeatureExtractor()
        features = extractor.extract(self.image)
        
        # Rank by variance
        ranked = extractor.rank_features_by_variance(features)
        
        self.assertIsInstance(ranked, list)
        self.assertGreater(len(ranked), 0)
        
    def test_feature_variance(self):
        """Test computing feature variance"""
        extractor = SpectralFeatureExtractor()
        features = extractor.extract(self.image)
        
        variances = extractor.compute_feature_variances(features)
        
        # All variances should be non-negative
        for var in variances.values():
            self.assertGreaterEqual(var, 0)


class TestFeatureVisualization(unittest.TestCase):
    """Test feature visualization helpers"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(50, 50, 50) * 0.2 + 0.5
        
    def test_feature_image_generation(self):
        """Test generating feature images"""
        extractor = SpectralFeatureExtractor()
        features = extractor.extract(self.image)
        
        # Get feature as image
        mean_image = features['mean']
        
        self.assertEqual(mean_image.shape, (50, 50))
        
    def test_feature_statistics(self):
        """Test computing feature statistics"""
        extractor = SpectralFeatureExtractor()
        features = extractor.extract(self.image)
        
        stats = extractor.compute_statistics(features)
        
        # Should have min, max, mean, std for each feature
        self.assertIn('mean', stats)
        for key in ['min', 'max', 'mean', 'std']:
            self.assertIn(key, stats['mean'])


if __name__ == '__main__':
    unittest.main()
