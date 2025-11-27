"""
Unit tests for hyperspectral target detection module

Tests all 7 target detection algorithms.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ai_nutrition.hyperspectral.target_detection import (
    TargetDetector,
    DetectionMethod,
    MatchedFilter,
    ACEDetector,
    CEMDetector,
    compute_spectral_angle,
    compute_detection_statistics
)


class TestTargetDetector(unittest.TestCase):
    """Test cases for TargetDetector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image_shape = (100, 100, 50)
        self.image = np.random.randn(*self.image_shape) * 0.1 + 0.5
        
        # Define target signature
        self.target_signature = np.random.randn(50) * 0.2 + 0.8
        
        # Add target at specific location
        self.image[50:55, 50:55, :] = self.target_signature + np.random.randn(5, 5, 50) * 0.05
        
    def test_initialization(self):
        """Test target detector initialization"""
        detector = TargetDetector(method=DetectionMethod.MATCHED_FILTER)
        self.assertIsNotNone(detector)
        
    def test_set_target(self):
        """Test setting target signature"""
        detector = TargetDetector(method=DetectionMethod.MATCHED_FILTER)
        detector.set_target(self.target_signature)
        
        self.assertIsNotNone(detector.target_signature)
        self.assertEqual(len(detector.target_signature), 50)


class TestMatchedFilter(unittest.TestCase):
    """Test Matched Filter detector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        self.target_signature = np.random.randn(50) * 0.2 + 0.8
        
        # Add target
        self.image[50:55, 50:55, :] = self.target_signature + np.random.randn(5, 5, 50) * 0.05
        
    def test_matched_filter_detection(self):
        """Test matched filter detection"""
        detector = TargetDetector(method=DetectionMethod.MATCHED_FILTER)
        detector.set_target(self.target_signature)
        
        detection_map = detector.detect(self.image)
        
        self.assertEqual(detection_map.shape, (100, 100))
        
    def test_matched_filter_target_response(self):
        """Test matched filter produces high response at target location"""
        detector = TargetDetector(method=DetectionMethod.MATCHED_FILTER)
        detector.set_target(self.target_signature)
        
        detection_map = detector.detect(self.image)
        
        # Target region should have higher scores
        target_score = np.mean(detection_map[50:55, 50:55])
        background_score = np.mean(detection_map[0:20, 0:20])
        
        self.assertGreater(target_score, background_score)
        
    def test_matched_filter_normalization(self):
        """Test matched filter output normalization"""
        detector = TargetDetector(
            method=DetectionMethod.MATCHED_FILTER,
            normalize=True
        )
        detector.set_target(self.target_signature)
        
        detection_map = detector.detect(self.image)
        
        # Normalized output should be in reasonable range
        self.assertGreater(np.max(detection_map), np.min(detection_map))


class TestACEDetector(unittest.TestCase):
    """Test Adaptive Coherence Estimator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        self.target_signature = np.random.randn(50) * 0.2 + 0.8
        
        # Add target
        self.image[50:55, 50:55, :] = self.target_signature + np.random.randn(5, 5, 50) * 0.05
        
    def test_ace_detection(self):
        """Test ACE detection"""
        detector = TargetDetector(method=DetectionMethod.ACE)
        detector.set_target(self.target_signature)
        
        detection_map = detector.detect(self.image)
        
        self.assertEqual(detection_map.shape, (100, 100))
        
    def test_ace_target_response(self):
        """Test ACE produces high response at target location"""
        detector = TargetDetector(method=DetectionMethod.ACE)
        detector.set_target(self.target_signature)
        
        detection_map = detector.detect(self.image)
        
        target_score = np.mean(detection_map[50:55, 50:55])
        background_score = np.mean(detection_map[0:20, 0:20])
        
        self.assertGreater(target_score, background_score)
        
    def test_ace_background_suppression(self):
        """Test ACE background suppression"""
        detector = TargetDetector(method=DetectionMethod.ACE)
        detector.set_target(self.target_signature)
        
        detection_map = detector.detect(self.image)
        
        # Background should have low variance after suppression
        background_region = detection_map[0:30, 0:30]
        background_std = np.std(background_region)
        
        self.assertLess(background_std, 0.5)


class TestCEMDetector(unittest.TestCase):
    """Test Constrained Energy Minimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        self.target_signature = np.random.randn(50) * 0.2 + 0.8
        
        # Add target
        self.image[50:55, 50:55, :] = self.target_signature + np.random.randn(5, 5, 50) * 0.05
        
    def test_cem_detection(self):
        """Test CEM detection"""
        detector = TargetDetector(method=DetectionMethod.CEM)
        detector.set_target(self.target_signature)
        
        detection_map = detector.detect(self.image)
        
        self.assertEqual(detection_map.shape, (100, 100))
        
    def test_cem_target_response(self):
        """Test CEM target response"""
        detector = TargetDetector(method=DetectionMethod.CEM)
        detector.set_target(self.target_signature)
        
        detection_map = detector.detect(self.image)
        
        target_score = np.mean(detection_map[50:55, 50:55])
        background_score = np.mean(detection_map[0:20, 0:20])
        
        self.assertGreater(target_score, background_score)
        
    def test_cem_energy_minimization(self):
        """Test CEM minimizes background energy"""
        detector = TargetDetector(method=DetectionMethod.CEM)
        detector.set_target(self.target_signature)
        
        detection_map = detector.detect(self.image)
        
        # Background energy should be minimized
        background_energy = np.mean(detection_map[0:30, 0:30] ** 2)
        
        self.assertLess(background_energy, 1.0)


class TestSpectralAngleMapper(unittest.TestCase):
    """Test Spectral Angle Mapper"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        self.target_signature = np.random.randn(50) * 0.2 + 0.8
        
        # Add target
        self.image[50:55, 50:55, :] = self.target_signature + np.random.randn(5, 5, 50) * 0.05
        
    def test_sam_detection(self):
        """Test SAM detection"""
        detector = TargetDetector(method=DetectionMethod.SAM)
        detector.set_target(self.target_signature)
        
        detection_map = detector.detect(self.image)
        
        self.assertEqual(detection_map.shape, (100, 100))
        
    def test_sam_angle_computation(self):
        """Test SAM computes spectral angles correctly"""
        detector = TargetDetector(method=DetectionMethod.SAM)
        detector.set_target(self.target_signature)
        
        detection_map = detector.detect(self.image)
        
        # SAM outputs angles in radians [0, π/2]
        self.assertTrue(np.all(detection_map >= 0))
        self.assertTrue(np.all(detection_map <= np.pi / 2))
        
    def test_sam_perfect_match(self):
        """Test SAM with perfect spectral match"""
        # Create image with exact target
        image = np.random.randn(50, 50, 50) * 0.1 + 0.5
        image[25, 25, :] = self.target_signature
        
        detector = TargetDetector(method=DetectionMethod.SAM)
        detector.set_target(self.target_signature)
        
        detection_map = detector.detect(image)
        
        # Perfect match should have angle close to 0
        perfect_match_angle = detection_map[25, 25]
        self.assertLess(perfect_match_angle, 0.1)


class TestSpectralAngleComputation(unittest.TestCase):
    """Test spectral angle computation utilities"""
    
    def test_compute_spectral_angle(self):
        """Test spectral angle computation"""
        spectrum1 = np.array([1, 2, 3, 4, 5])
        spectrum2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        angle = compute_spectral_angle(spectrum1, spectrum2)
        
        # Angle should be small for similar spectra
        self.assertIsInstance(angle, float)
        self.assertGreater(angle, 0)
        self.assertLess(angle, 0.2)
        
    def test_spectral_angle_orthogonal(self):
        """Test spectral angle for orthogonal spectra"""
        spectrum1 = np.array([1, 0, 0, 0, 0])
        spectrum2 = np.array([0, 1, 0, 0, 0])
        
        angle = compute_spectral_angle(spectrum1, spectrum2)
        
        # Should be close to π/2 (90 degrees)
        self.assertAlmostEqual(angle, np.pi / 2, places=1)
        
    def test_spectral_angle_identical(self):
        """Test spectral angle for identical spectra"""
        spectrum = np.random.randn(50)
        
        angle = compute_spectral_angle(spectrum, spectrum)
        
        # Should be close to 0
        self.assertLess(angle, 0.01)


class TestOSPDetector(unittest.TestCase):
    """Test Orthogonal Subspace Projection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        self.target_signature = np.random.randn(50) * 0.2 + 0.8
        
        # Background signatures
        self.background_signatures = [
            np.random.randn(50) * 0.1 + 0.4,
            np.random.randn(50) * 0.1 + 0.6
        ]
        
        # Add target
        self.image[50:55, 50:55, :] = self.target_signature + np.random.randn(5, 5, 50) * 0.05
        
    def test_osp_detection(self):
        """Test OSP detection"""
        detector = TargetDetector(method=DetectionMethod.OSP)
        detector.set_target(self.target_signature)
        detector.set_background(self.background_signatures)
        
        detection_map = detector.detect(self.image)
        
        self.assertEqual(detection_map.shape, (100, 100))
        
    def test_osp_background_rejection(self):
        """Test OSP rejects background signatures"""
        detector = TargetDetector(method=DetectionMethod.OSP)
        detector.set_target(self.target_signature)
        detector.set_background(self.background_signatures)
        
        detection_map = detector.detect(self.image)
        
        # Should suppress background
        target_score = np.mean(detection_map[50:55, 50:55])
        background_score = np.mean(detection_map[0:20, 0:20])
        
        self.assertGreater(target_score, background_score)


class TestAdaptiveMatchedFilter(unittest.TestCase):
    """Test Adaptive Matched Filter"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        self.target_signature = np.random.randn(50) * 0.2 + 0.8
        
        # Add target
        self.image[50:55, 50:55, :] = self.target_signature + np.random.randn(5, 5, 50) * 0.05
        
    def test_amf_detection(self):
        """Test adaptive matched filter detection"""
        detector = TargetDetector(method=DetectionMethod.ADAPTIVE_MATCHED_FILTER)
        detector.set_target(self.target_signature)
        
        detection_map = detector.detect(self.image)
        
        self.assertEqual(detection_map.shape, (100, 100))
        
    def test_amf_adaptation(self):
        """Test AMF adapts to local statistics"""
        detector = TargetDetector(
            method=DetectionMethod.ADAPTIVE_MATCHED_FILTER,
            adaptive_window=15
        )
        detector.set_target(self.target_signature)
        
        detection_map = detector.detect(self.image)
        
        target_score = np.mean(detection_map[50:55, 50:55])
        background_score = np.mean(detection_map[0:20, 0:20])
        
        self.assertGreater(target_score, background_score)


class TestHybridDetector(unittest.TestCase):
    """Test hybrid detection combining multiple methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        self.target_signature = np.random.randn(50) * 0.2 + 0.8
        
        # Add target
        self.image[50:55, 50:55, :] = self.target_signature + np.random.randn(5, 5, 50) * 0.05
        
    def test_hybrid_detection(self):
        """Test hybrid detector combining methods"""
        detector = TargetDetector(
            method=DetectionMethod.HYBRID,
            methods=[DetectionMethod.MATCHED_FILTER, DetectionMethod.SAM]
        )
        detector.set_target(self.target_signature)
        
        detection_map = detector.detect(self.image)
        
        self.assertEqual(detection_map.shape, (100, 100))
        
    def test_hybrid_fusion(self):
        """Test hybrid fusion of multiple detectors"""
        detector = TargetDetector(
            method=DetectionMethod.HYBRID,
            methods=[DetectionMethod.ACE, DetectionMethod.CEM],
            fusion_method='average'
        )
        detector.set_target(self.target_signature)
        
        detection_map = detector.detect(self.image)
        
        target_score = np.mean(detection_map[50:55, 50:55])
        background_score = np.mean(detection_map[0:20, 0:20])
        
        self.assertGreater(target_score, background_score)


class TestDetectionStatistics(unittest.TestCase):
    """Test detection statistics computation"""
    
    def test_compute_detection_statistics(self):
        """Test detection statistics computation"""
        # Create detection map
        detection_map = np.random.rand(100, 100)
        
        # Create ground truth
        ground_truth = np.zeros((100, 100))
        ground_truth[50:55, 50:55] = 1  # Target present
        
        stats = compute_detection_statistics(detection_map, ground_truth, threshold=0.5)
        
        self.assertIn('true_positives', stats)
        self.assertIn('false_positives', stats)
        self.assertIn('true_negatives', stats)
        self.assertIn('false_negatives', stats)
        
    def test_precision_recall(self):
        """Test precision and recall computation"""
        # Perfect detection
        detection_map = np.zeros((100, 100))
        detection_map[50:55, 50:55] = 1.0
        
        ground_truth = np.zeros((100, 100))
        ground_truth[50:55, 50:55] = 1
        
        stats = compute_detection_statistics(detection_map, ground_truth, threshold=0.5)
        
        precision = stats['true_positives'] / (stats['true_positives'] + stats['false_positives'] + 1e-10)
        recall = stats['true_positives'] / (stats['true_positives'] + stats['false_negatives'] + 1e-10)
        
        # Should be high for perfect detection
        self.assertGreater(precision, 0.8)
        self.assertGreater(recall, 0.8)


class TestMultiTargetDetection(unittest.TestCase):
    """Test detection of multiple targets"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        
        # Multiple target signatures
        self.target1 = np.random.randn(50) * 0.2 + 0.8
        self.target2 = np.random.randn(50) * 0.2 + 0.3
        
        # Add targets at different locations
        self.image[25:30, 25:30, :] = self.target1 + np.random.randn(5, 5, 50) * 0.05
        self.image[75:80, 75:80, :] = self.target2 + np.random.randn(5, 5, 50) * 0.05
        
    def test_detect_multiple_targets(self):
        """Test detecting multiple targets"""
        targets = [self.target1, self.target2]
        
        detection_maps = []
        for target in targets:
            detector = TargetDetector(method=DetectionMethod.ACE)
            detector.set_target(target)
            detection_map = detector.detect(self.image)
            detection_maps.append(detection_map)
        
        # Should have detection maps for both targets
        self.assertEqual(len(detection_maps), 2)
        
        # Each map should detect its respective target
        self.assertGreater(np.mean(detection_maps[0][25:30, 25:30]), 
                          np.mean(detection_maps[0][75:80, 75:80]))
        self.assertGreater(np.mean(detection_maps[1][75:80, 75:80]),
                          np.mean(detection_maps[1][25:30, 25:30]))


class TestDetectionThresholding(unittest.TestCase):
    """Test detection thresholding"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detection_map = np.random.rand(100, 100)
        
    def test_fixed_threshold(self):
        """Test fixed threshold"""
        threshold = 0.5
        binary_map = self.detection_map > threshold
        
        # Should produce binary map
        self.assertTrue(np.all((binary_map == 0) | (binary_map == 1)))
        
    def test_adaptive_threshold(self):
        """Test adaptive thresholding"""
        # Use mean + k*std as threshold
        threshold = np.mean(self.detection_map) + 2 * np.std(self.detection_map)
        binary_map = self.detection_map > threshold
        
        # Should detect ~2.5% as targets (for normal distribution)
        target_fraction = np.mean(binary_map)
        self.assertLess(target_fraction, 0.1)
        
    def test_otsu_threshold(self):
        """Test Otsu's thresholding"""
        # Simplified Otsu (just check it runs)
        hist, bins = np.histogram(self.detection_map, bins=256)
        
        # Find threshold that maximizes between-class variance
        # (simplified version)
        threshold = np.mean(self.detection_map)
        binary_map = self.detection_map > threshold
        
        self.assertEqual(binary_map.shape, (100, 100))


class TestDetectionEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_no_target_present(self):
        """Test detection when target is not present"""
        image = np.random.randn(50, 50, 30) * 0.1 + 0.5
        target = np.random.randn(30) * 0.2 + 0.8
        
        detector = TargetDetector(method=DetectionMethod.MATCHED_FILTER)
        detector.set_target(target)
        
        detection_map = detector.detect(image)
        
        # Should still produce valid map
        self.assertEqual(detection_map.shape, (50, 50))
        
    def test_target_everywhere(self):
        """Test when target is everywhere"""
        target = np.random.randn(30) * 0.2 + 0.8
        
        # Image is all target
        image = np.tile(target, (50, 50, 1)).transpose(0, 1, 2)
        
        detector = TargetDetector(method=DetectionMethod.SAM)
        detector.set_target(target)
        
        detection_map = detector.detect(image)
        
        # All angles should be small
        self.assertLess(np.mean(detection_map), 0.2)
        
    def test_zero_target(self):
        """Test with zero target signature"""
        image = np.random.randn(50, 50, 30) * 0.1 + 0.5
        target = np.zeros(30)
        
        detector = TargetDetector(method=DetectionMethod.MATCHED_FILTER)
        
        # Should handle gracefully or raise informative error
        try:
            detector.set_target(target)
            detection_map = detector.detect(image)
            # If it doesn't error, check output is valid
            self.assertEqual(detection_map.shape, (50, 50))
        except ValueError as e:
            # Expected error for zero target
            self.assertIn('zero', str(e).lower())


class TestDetectionPerformance(unittest.TestCase):
    """Test detection performance"""
    
    def test_detection_speed(self):
        """Test detection speed"""
        import time
        
        image = np.random.randn(256, 256, 100) * 0.2 + 0.5
        target = np.random.randn(100) * 0.2 + 0.8
        
        detector = TargetDetector(method=DetectionMethod.MATCHED_FILTER)
        detector.set_target(target)
        
        start = time.time()
        detection_map = detector.detect(image)
        elapsed = time.time() - start
        
        # Should complete in < 5 seconds
        self.assertLess(elapsed, 5)
        
    def test_batch_detection(self):
        """Test batch detection on multiple images"""
        import time
        
        images = [np.random.randn(128, 128, 50) * 0.2 + 0.5 for _ in range(10)]
        target = np.random.randn(50) * 0.2 + 0.8
        
        detector = TargetDetector(method=DetectionMethod.ACE)
        detector.set_target(target)
        
        start = time.time()
        for image in images:
            detection_map = detector.detect(image)
        elapsed = time.time() - start
        
        # Should process > 2 images/second
        fps = len(images) / elapsed
        self.assertGreater(fps, 2)


class TestDetectionVisualization(unittest.TestCase):
    """Test detection visualization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        self.target = np.random.randn(50) * 0.2 + 0.8
        
        detector = TargetDetector(method=DetectionMethod.ACE)
        detector.set_target(self.target)
        self.detection_map = detector.detect(self.image)
        
    def test_generate_heatmap(self):
        """Test heatmap generation"""
        # Normalize to [0, 1]
        heatmap = (self.detection_map - self.detection_map.min()) / \
                  (self.detection_map.max() - self.detection_map.min())
        
        self.assertEqual(heatmap.shape, (100, 100))
        self.assertTrue(np.all(heatmap >= 0))
        self.assertTrue(np.all(heatmap <= 1))
        
    def test_overlay_detections(self):
        """Test overlaying detections on RGB image"""
        # Create RGB representation
        rgb = self.image[:, :, [20, 30, 40]]
        
        # Create overlay
        threshold = np.percentile(self.detection_map, 95)
        overlay = rgb.copy()
        mask = self.detection_map > threshold
        overlay[mask, 0] += 0.5  # Add red channel
        
        self.assertEqual(overlay.shape, rgb.shape)
        
    def test_detection_contours(self):
        """Test extracting detection contours"""
        # Threshold
        threshold = np.percentile(self.detection_map, 95)
        binary = self.detection_map > threshold
        
        # Find connected components (simplified)
        from scipy import ndimage
        labeled, n_components = ndimage.label(binary)
        
        # Should find some components
        self.assertGreaterEqual(n_components, 0)


class TestDetectionMethodComparison(unittest.TestCase):
    """Test comparison of different detection methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        self.target = np.random.randn(50) * 0.2 + 0.8
        
        # Add target
        self.image[50:55, 50:55, :] = self.target + np.random.randn(5, 5, 50) * 0.05
        
    def test_compare_methods(self):
        """Test comparing multiple detection methods"""
        methods = [
            DetectionMethod.MATCHED_FILTER,
            DetectionMethod.ACE,
            DetectionMethod.CEM,
            DetectionMethod.SAM
        ]
        
        results = {}
        for method in methods:
            detector = TargetDetector(method=method)
            detector.set_target(self.target)
            detection_map = detector.detect(self.image)
            results[method.value] = detection_map
        
        # All should detect target region
        for method_name, detection_map in results.items():
            target_score = np.mean(detection_map[50:55, 50:55])
            background_score = np.mean(detection_map[0:20, 0:20])
            
            self.assertGreater(target_score, background_score,
                             f"Method {method_name} failed to detect target")


if __name__ == '__main__':
    unittest.main()
