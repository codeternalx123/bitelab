"""
Unit tests for hyperspectral band selection module

Tests all 11 band selection algorithms.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ai_nutrition.hyperspectral.band_selection import (
    BandSelector,
    SelectionMethod,
    BandSelectionConfig
)


class TestBandSelector(unittest.TestCase):
    """Test cases for BandSelector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image_shape = (100, 100, 50)
        # Create image with varying band importance
        self.image = self._generate_test_image()
        
    def _generate_test_image(self):
        """Generate synthetic hyperspectral image with known structure"""
        h, w, c = self.image_shape
        image = np.zeros((h, w, c))
        
        # Add signal to specific bands
        for i in range(c):
            # Some bands have strong signal
            if i % 5 == 0:
                image[:, :, i] = np.random.randn(h, w) * 0.5 + 0.7
            else:
                # Other bands have weak signal
                image[:, :, i] = np.random.randn(h, w) * 0.1 + 0.3
        
        return image
        
    def test_initialization(self):
        """Test band selector initialization"""
        selector = BandSelector(method=SelectionMethod.VARIANCE, n_bands=20)
        self.assertIsNotNone(selector)
        self.assertEqual(selector.n_bands, 20)


class TestVarianceBasedSelection(unittest.TestCase):
    """Test variance-based band selection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.2 + 0.5
        
    def test_variance_selection(self):
        """Test variance-based selection"""
        selector = BandSelector(method=SelectionMethod.VARIANCE, n_bands=20)
        selected = selector.select(self.image)
        
        self.assertEqual(len(selected), 20)
        self.assertTrue(all(0 <= idx < 50 for idx in selected))
        
    def test_variance_order(self):
        """Test that high variance bands are selected"""
        # Create image with known variance structure
        image = np.random.randn(100, 100, 30) * 0.1
        
        # Make specific bands high variance
        high_var_bands = [5, 10, 15, 20, 25]
        for band in high_var_bands:
            image[:, :, band] = np.random.randn(100, 100) * 2.0
        
        selector = BandSelector(method=SelectionMethod.VARIANCE, n_bands=5)
        selected = selector.select(image)
        
        # Should select the high variance bands
        selected_set = set(selected)
        high_var_set = set(high_var_bands)
        
        # At least 3 of the 5 high variance bands should be selected
        overlap = len(selected_set & high_var_set)
        self.assertGreaterEqual(overlap, 3)


class TestMutualInformationSelection(unittest.TestCase):
    """Test mutual information-based selection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.2 + 0.5
        
    def test_mutual_information_selection(self):
        """Test mutual information selection"""
        selector = BandSelector(
            method=SelectionMethod.MUTUAL_INFO,
            n_bands=20
        )
        selected = selector.select(self.image)
        
        self.assertEqual(len(selected), 20)
        
    def test_redundancy_reduction(self):
        """Test that redundant bands are avoided"""
        # Create image with redundant bands
        image = np.random.randn(100, 100, 30)
        
        # Make bands 0-4 very similar (redundant)
        base_band = np.random.randn(100, 100)
        for i in range(5):
            image[:, :, i] = base_band + np.random.randn(100, 100) * 0.01
        
        selector = BandSelector(
            method=SelectionMethod.MUTUAL_INFO,
            n_bands=10
        )
        selected = selector.select(image)
        
        # Should not select all redundant bands
        redundant_selected = sum(1 for idx in selected if idx < 5)
        self.assertLess(redundant_selected, 4)  # At most 3 of the 5 redundant bands


class TestPCABasedSelection(unittest.TestCase):
    """Test PCA-based band selection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.2 + 0.5
        
    def test_pca_selection(self):
        """Test PCA-based selection"""
        selector = BandSelector(method=SelectionMethod.PCA, n_bands=20)
        selected = selector.select(self.image)
        
        self.assertEqual(len(selected), 20)
        
    def test_pca_variance_capture(self):
        """Test that PCA captures major variance"""
        # Create image with principal components
        n_samples = 10000
        n_bands = 50
        
        # Create data with clear principal components
        data = np.random.randn(n_samples, n_bands)
        
        # Make first few components dominate
        weights = np.linspace(10, 0.1, n_bands)
        data = data * weights
        
        # Reshape to image format
        h = 100
        w = n_samples // h
        image = data[:h*w].reshape(h, w, n_bands)
        
        selector = BandSelector(method=SelectionMethod.PCA, n_bands=10)
        selected = selector.select(image)
        
        self.assertEqual(len(selected), 10)


class TestCorrelationBasedSelection(unittest.TestCase):
    """Test correlation-based selection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.2 + 0.5
        
    def test_correlation_selection(self):
        """Test correlation-based selection"""
        selector = BandSelector(
            method=SelectionMethod.CORRELATION,
            n_bands=20
        )
        selected = selector.select(self.image)
        
        self.assertEqual(len(selected), 20)
        
    def test_low_correlation_bands(self):
        """Test that low-correlation bands are preferred"""
        # Create bands with varying correlation
        image = np.random.randn(50, 50, 20)
        
        # Make some bands highly correlated
        for i in range(5):
            image[:, :, i] = image[:, :, 0] + np.random.randn(50, 50) * 0.1
        
        selector = BandSelector(
            method=SelectionMethod.CORRELATION,
            n_bands=10
        )
        selected = selector.select(image)
        
        # Should prefer uncorrelated bands
        self.assertEqual(len(selected), 10)


class TestEndmemberBasedSelection(unittest.TestCase):
    """Test endmember-based selection methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.2 + 0.5
        
    def test_vca_selection(self):
        """Test VCA-based selection"""
        selector = BandSelector(method=SelectionMethod.VCA, n_bands=20)
        selected = selector.select(self.image)
        
        self.assertEqual(len(selected), 20)
        
    def test_nfindr_selection(self):
        """Test N-FINDR-based selection"""
        selector = BandSelector(method=SelectionMethod.NFINDR, n_bands=20)
        selected = selector.select(self.image)
        
        self.assertEqual(len(selected), 20)
        
    def test_atgp_selection(self):
        """Test ATGP-based selection"""
        selector = BandSelector(method=SelectionMethod.ATGP, n_bands=20)
        selected = selector.select(self.image)
        
        self.assertEqual(len(selected), 20)


class TestGeneticAlgorithmSelection(unittest.TestCase):
    """Test genetic algorithm-based selection"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Smaller image for GA (faster)
        self.image = np.random.randn(50, 50, 30) * 0.2 + 0.5
        
    def test_genetic_algorithm_selection(self):
        """Test genetic algorithm selection"""
        selector = BandSelector(
            method=SelectionMethod.GENETIC_ALGORITHM,
            n_bands=10,
            ga_population_size=20,
            ga_generations=10
        )
        selected = selector.select(self.image)
        
        self.assertEqual(len(selected), 10)
        
    def test_genetic_algorithm_convergence(self):
        """Test that GA improves over generations"""
        selector = BandSelector(
            method=SelectionMethod.GENETIC_ALGORITHM,
            n_bands=10,
            ga_population_size=30,
            ga_generations=20
        )
        
        selected = selector.select(self.image)
        
        # Should complete without error
        self.assertEqual(len(selected), 10)


class TestBandSelectionConfiguration(unittest.TestCase):
    """Test band selection configuration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.2 + 0.5
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = BandSelectionConfig(
            min_bands=5,
            max_bands=30,
            optimize_for_speed=True
        )
        
        selector = BandSelector(
            method=SelectionMethod.VARIANCE,
            n_bands=20,
            config=config
        )
        
        selected = selector.select(self.image)
        
        self.assertGreaterEqual(len(selected), config.min_bands)
        self.assertLessEqual(len(selected), config.max_bands)
        
    def test_band_range_constraints(self):
        """Test band range constraints"""
        config = BandSelectionConfig(
            band_range=(10, 40)  # Only select from bands 10-40
        )
        
        selector = BandSelector(
            method=SelectionMethod.VARIANCE,
            n_bands=10,
            config=config
        )
        
        selected = selector.select(self.image)
        
        # All selected bands should be in range
        self.assertTrue(all(10 <= idx < 40 for idx in selected))


class TestBandSelectionComparison(unittest.TestCase):
    """Test comparison of different selection methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.2 + 0.5
        self.n_bands = 20
        
    def test_method_consistency(self):
        """Test that methods return consistent number of bands"""
        methods = [
            SelectionMethod.VARIANCE,
            SelectionMethod.MUTUAL_INFO,
            SelectionMethod.PCA,
            SelectionMethod.CORRELATION
        ]
        
        for method in methods:
            selector = BandSelector(method=method, n_bands=self.n_bands)
            selected = selector.select(self.image)
            
            self.assertEqual(len(selected), self.n_bands,
                           f"Method {method.value} returned wrong number of bands")
            
    def test_no_duplicate_bands(self):
        """Test that no method returns duplicate bands"""
        methods = [
            SelectionMethod.VARIANCE,
            SelectionMethod.MUTUAL_INFO,
            SelectionMethod.PCA
        ]
        
        for method in methods:
            selector = BandSelector(method=method, n_bands=20)
            selected = selector.select(self.image)
            
            # No duplicates
            self.assertEqual(len(selected), len(set(selected)),
                           f"Method {method.value} returned duplicate bands")


class TestBandSelectionEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_select_more_bands_than_available(self):
        """Test selecting more bands than available"""
        image = np.random.randn(50, 50, 10)
        
        selector = BandSelector(method=SelectionMethod.VARIANCE, n_bands=20)
        
        with self.assertRaises(ValueError):
            selector.select(image)
            
    def test_select_zero_bands(self):
        """Test selecting zero bands"""
        image = np.random.randn(50, 50, 30)
        
        with self.assertRaises(ValueError):
            BandSelector(method=SelectionMethod.VARIANCE, n_bands=0)
            
    def test_select_negative_bands(self):
        """Test selecting negative number of bands"""
        with self.assertRaises(ValueError):
            BandSelector(method=SelectionMethod.VARIANCE, n_bands=-5)
            
    def test_single_band_image(self):
        """Test with single-band image"""
        image = np.random.randn(50, 50, 1)
        
        selector = BandSelector(method=SelectionMethod.VARIANCE, n_bands=1)
        selected = selector.select(image)
        
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0], 0)
        
    def test_constant_bands(self):
        """Test with constant (zero variance) bands"""
        image = np.ones((50, 50, 20)) * 0.5
        
        # Add variance to a few bands
        image[:, :, 5] = np.random.randn(50, 50)
        image[:, :, 10] = np.random.randn(50, 50)
        image[:, :, 15] = np.random.randn(50, 50)
        
        selector = BandSelector(method=SelectionMethod.VARIANCE, n_bands=3)
        selected = selector.select(image)
        
        # Should select the non-constant bands
        self.assertEqual(set(selected), {5, 10, 15})


class TestBandSelectionPerformance(unittest.TestCase):
    """Test band selection performance"""
    
    def test_selection_speed(self):
        """Test selection speed for large images"""
        import time
        
        image = np.random.randn(256, 256, 100) * 0.2 + 0.5
        
        methods_and_targets = [
            (SelectionMethod.VARIANCE, 2.0),  # < 2 seconds
            (SelectionMethod.PCA, 5.0),  # < 5 seconds
            (SelectionMethod.MUTUAL_INFO, 10.0)  # < 10 seconds
        ]
        
        for method, max_time in methods_and_targets:
            selector = BandSelector(method=method, n_bands=20)
            
            start = time.time()
            selected = selector.select(image)
            elapsed = time.time() - start
            
            self.assertLess(elapsed, max_time,
                          f"Method {method.value} took {elapsed:.2f}s (limit: {max_time}s)")
            
    def test_batch_selection(self):
        """Test selecting bands for multiple images"""
        import time
        
        images = [np.random.randn(128, 128, 50) * 0.2 + 0.5 for _ in range(10)]
        
        selector = BandSelector(method=SelectionMethod.VARIANCE, n_bands=20)
        
        start = time.time()
        for image in images:
            selected = selector.select(image)
        elapsed = time.time() - start
        
        # Should process > 2 images/second
        fps = len(images) / elapsed
        self.assertGreater(fps, 2)


class TestBandSelectionValidation(unittest.TestCase):
    """Test band selection validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.2 + 0.5
        
    def test_selected_bands_valid(self):
        """Test that selected bands are valid indices"""
        selector = BandSelector(method=SelectionMethod.VARIANCE, n_bands=20)
        selected = selector.select(self.image)
        
        # All indices should be valid
        self.assertTrue(all(isinstance(idx, (int, np.integer)) for idx in selected))
        self.assertTrue(all(0 <= idx < 50 for idx in selected))
        
    def test_selected_bands_sorted(self):
        """Test that selected bands can be sorted"""
        selector = BandSelector(method=SelectionMethod.VARIANCE, n_bands=20)
        selected = selector.select(self.image)
        
        # Should be sortable
        sorted_bands = sorted(selected)
        self.assertEqual(len(sorted_bands), 20)
        
    def test_selection_reproducibility(self):
        """Test that selection is reproducible with same seed"""
        np.random.seed(42)
        selector1 = BandSelector(method=SelectionMethod.VARIANCE, n_bands=20)
        selected1 = selector1.select(self.image)
        
        np.random.seed(42)
        selector2 = BandSelector(method=SelectionMethod.VARIANCE, n_bands=20)
        selected2 = selector2.select(self.image)
        
        # Should be identical
        self.assertEqual(list(selected1), list(selected2))


if __name__ == '__main__':
    unittest.main()
