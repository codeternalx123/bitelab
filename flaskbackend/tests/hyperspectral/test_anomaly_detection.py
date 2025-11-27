"""
Unit tests for hyperspectral anomaly detection module

Tests all 8 anomaly detection algorithms.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ai_nutrition.hyperspectral.anomaly_detection import (
    AnomalyDetector,
    DetectionMethod,
    RXDetector,
    LocalRXDetector,
    KernelRXDetector,
    compute_mahalanobis_distance,
    compute_roc_curve
)


class TestAnomalyDetector(unittest.TestCase):
    """Test cases for AnomalyDetector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image_shape = (100, 100, 50)
        self.image = self._generate_test_image_with_anomalies()
        
    def _generate_test_image_with_anomalies(self):
        """Generate synthetic image with known anomalies"""
        h, w, c = self.image_shape
        
        # Background (normal)
        image = np.random.randn(h, w, c) * 0.1 + 0.5
        
        # Add anomalies at specific locations
        anomaly_locations = [(25, 25), (75, 75), (50, 80)]
        for y, x in anomaly_locations:
            # Different spectral signature
            image[y-2:y+3, x-2:x+3, :] = np.random.randn(5, 5, c) * 0.3 + 0.9
        
        return image
        
    def test_initialization(self):
        """Test anomaly detector initialization"""
        detector = AnomalyDetector(method=DetectionMethod.RX)
        self.assertIsNotNone(detector)


class TestRXDetector(unittest.TestCase):
    """Test RX (Reed-Xiaoli) detector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        
        # Add anomaly
        self.image[50:55, 50:55, :] = np.random.randn(5, 5, 50) * 0.3 + 0.9
        
    def test_rx_detection(self):
        """Test RX anomaly detection"""
        detector = AnomalyDetector(method=DetectionMethod.RX)
        detection_map = detector.detect(self.image)
        
        self.assertEqual(detection_map.shape, (100, 100))
        
    def test_rx_anomaly_scores(self):
        """Test RX produces higher scores for anomalies"""
        detector = AnomalyDetector(method=DetectionMethod.RX)
        detection_map = detector.detect(self.image)
        
        # Anomaly region should have higher scores
        anomaly_score = np.mean(detection_map[50:55, 50:55])
        background_score = np.mean(detection_map[0:20, 0:20])
        
        self.assertGreater(anomaly_score, background_score)
        
    def test_rx_output_range(self):
        """Test RX output is non-negative"""
        detector = AnomalyDetector(method=DetectionMethod.RX)
        detection_map = detector.detect(self.image)
        
        # RX scores should be non-negative
        self.assertTrue(np.all(detection_map >= 0))


class TestLocalRXDetector(unittest.TestCase):
    """Test Local RX detector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        
        # Add anomaly
        self.image[50:55, 50:55, :] = np.random.randn(5, 5, 50) * 0.3 + 0.9
        
    def test_local_rx_detection(self):
        """Test Local RX detection"""
        detector = AnomalyDetector(
            method=DetectionMethod.LOCAL_RX,
            window_size=15
        )
        detection_map = detector.detect(self.image)
        
        self.assertEqual(detection_map.shape, (100, 100))
        
    def test_window_size_effect(self):
        """Test effect of different window sizes"""
        window_sizes = [7, 15, 31]
        
        for window_size in window_sizes:
            detector = AnomalyDetector(
                method=DetectionMethod.LOCAL_RX,
                window_size=window_size
            )
            detection_map = detector.detect(self.image)
            
            self.assertEqual(detection_map.shape, (100, 100))
            
    def test_local_vs_global_rx(self):
        """Test Local RX vs global RX"""
        # Global RX
        detector_global = AnomalyDetector(method=DetectionMethod.RX)
        map_global = detector_global.detect(self.image)
        
        # Local RX
        detector_local = AnomalyDetector(
            method=DetectionMethod.LOCAL_RX,
            window_size=15
        )
        map_local = detector_local.detect(self.image)
        
        # Should produce different results
        self.assertFalse(np.allclose(map_global, map_local))


class TestKernelRXDetector(unittest.TestCase):
    """Test Kernel RX detector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        
        # Add non-linear anomaly
        self.image[50:55, 50:55, :] = np.random.randn(5, 5, 50) * 0.3 + 0.9
        
    def test_kernel_rx_detection(self):
        """Test Kernel RX detection"""
        detector = AnomalyDetector(
            method=DetectionMethod.KERNEL_RX,
            kernel='rbf'
        )
        detection_map = detector.detect(self.image)
        
        self.assertEqual(detection_map.shape, (100, 100))
        
    def test_different_kernels(self):
        """Test different kernel functions"""
        kernels = ['rbf', 'poly', 'sigmoid']
        
        for kernel in kernels:
            detector = AnomalyDetector(
                method=DetectionMethod.KERNEL_RX,
                kernel=kernel
            )
            detection_map = detector.detect(self.image)
            
            self.assertEqual(detection_map.shape, (100, 100))


class TestIsolationForest(unittest.TestCase):
    """Test Isolation Forest-based detection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        
        # Add anomaly
        self.image[50:55, 50:55, :] = np.random.randn(5, 5, 50) * 0.3 + 0.9
        
    def test_isolation_forest_detection(self):
        """Test Isolation Forest detection"""
        detector = AnomalyDetector(
            method=DetectionMethod.ISOLATION_FOREST,
            contamination=0.1
        )
        detection_map = detector.detect(self.image)
        
        self.assertEqual(detection_map.shape, (100, 100))
        
    def test_contamination_parameter(self):
        """Test contamination parameter effect"""
        contaminations = [0.01, 0.05, 0.1, 0.2]
        
        for contamination in contaminations:
            detector = AnomalyDetector(
                method=DetectionMethod.ISOLATION_FOREST,
                contamination=contamination
            )
            detection_map = detector.detect(self.image)
            
            # Count detected anomalies
            anomalies = detection_map > 0.5
            anomaly_fraction = np.mean(anomalies)
            
            # Should be roughly around contamination level
            self.assertLess(abs(anomaly_fraction - contamination), 0.1)


class TestOneClassSVM(unittest.TestCase):
    """Test One-Class SVM detection"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Smaller image for SVM (faster)
        self.image = np.random.randn(50, 50, 30) * 0.1 + 0.5
        
        # Add anomaly
        self.image[25:30, 25:30, :] = np.random.randn(5, 5, 30) * 0.3 + 0.9
        
    def test_one_class_svm_detection(self):
        """Test One-Class SVM detection"""
        detector = AnomalyDetector(
            method=DetectionMethod.ONE_CLASS_SVM,
            nu=0.1
        )
        detection_map = detector.detect(self.image)
        
        self.assertEqual(detection_map.shape, (50, 50))
        
    def test_nu_parameter(self):
        """Test nu parameter (fraction of outliers)"""
        nu_values = [0.05, 0.1, 0.2]
        
        for nu in nu_values:
            detector = AnomalyDetector(
                method=DetectionMethod.ONE_CLASS_SVM,
                nu=nu
            )
            detection_map = detector.detect(self.image)
            
            self.assertEqual(detection_map.shape, (50, 50))


class TestAutoencoderDetection(unittest.TestCase):
    """Test autoencoder-based anomaly detection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        
        # Add anomaly
        self.image[50:55, 50:55, :] = np.random.randn(5, 5, 50) * 0.3 + 0.9
        
    def test_autoencoder_detection(self):
        """Test autoencoder-based detection"""
        detector = AnomalyDetector(
            method=DetectionMethod.AUTOENCODER,
            latent_dim=10
        )
        
        # Train on image (treating most as normal)
        detector.fit(self.image)
        
        # Detect anomalies
        detection_map = detector.detect(self.image)
        
        self.assertEqual(detection_map.shape, (100, 100))
        
    def test_reconstruction_error(self):
        """Test reconstruction error as anomaly score"""
        detector = AnomalyDetector(
            method=DetectionMethod.AUTOENCODER,
            latent_dim=10
        )
        
        detector.fit(self.image)
        detection_map = detector.detect(self.image)
        
        # Anomaly region should have higher reconstruction error
        anomaly_score = np.mean(detection_map[50:55, 50:55])
        background_score = np.mean(detection_map[0:20, 0:20])
        
        self.assertGreater(anomaly_score, background_score)


class TestMahalanobisDistance(unittest.TestCase):
    """Test Mahalanobis distance computation"""
    
    def test_mahalanobis_distance(self):
        """Test Mahalanobis distance calculation"""
        # Create data with known distribution
        n_samples = 1000
        n_features = 10
        
        data = np.random.randn(n_samples, n_features)
        
        # Test point
        test_point = np.random.randn(n_features)
        
        # Compute Mahalanobis distance
        distance = compute_mahalanobis_distance(test_point, data)
        
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0)
        
    def test_mahalanobis_outlier_detection(self):
        """Test outlier has larger Mahalanobis distance"""
        # Normal data
        data = np.random.randn(1000, 10) * 0.1
        
        # Normal test point
        normal_point = np.random.randn(10) * 0.1
        
        # Outlier test point
        outlier_point = np.random.randn(10) * 3.0
        
        # Compute distances
        normal_dist = compute_mahalanobis_distance(normal_point, data)
        outlier_dist = compute_mahalanobis_distance(outlier_point, data)
        
        # Outlier should have larger distance
        self.assertGreater(outlier_dist, normal_dist)


class TestROCAnalysis(unittest.TestCase):
    """Test ROC curve computation"""
    
    def test_roc_curve_computation(self):
        """Test ROC curve computation"""
        # Create synthetic scores and labels
        n_samples = 100
        
        # Normal samples (label 0)
        normal_scores = np.random.randn(80) * 0.1 + 0.3
        
        # Anomalies (label 1)
        anomaly_scores = np.random.randn(20) * 0.1 + 0.8
        
        scores = np.concatenate([normal_scores, anomaly_scores])
        labels = np.concatenate([np.zeros(80), np.ones(20)])
        
        # Compute ROC
        fpr, tpr, thresholds = compute_roc_curve(scores, labels)
        
        self.assertEqual(len(fpr), len(tpr))
        self.assertEqual(len(fpr), len(thresholds))
        
    def test_roc_auc(self):
        """Test ROC AUC computation"""
        # Perfect separation
        scores = np.concatenate([np.zeros(50), np.ones(50)])
        labels = np.concatenate([np.zeros(50), np.ones(50)])
        
        fpr, tpr, _ = compute_roc_curve(scores, labels)
        
        # Compute AUC using trapezoidal rule
        auc = np.trapz(tpr, fpr)
        
        # Should be close to 1 for perfect separation
        self.assertGreater(auc, 0.9)


class TestDetectionThresholding(unittest.TestCase):
    """Test detection thresholding"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        self.image[50:55, 50:55, :] = np.random.randn(5, 5, 50) * 0.3 + 0.9
        
    def test_percentile_thresholding(self):
        """Test percentile-based thresholding"""
        detector = AnomalyDetector(method=DetectionMethod.RX)
        detection_map = detector.detect(self.image)
        
        # Apply percentile threshold
        threshold = np.percentile(detection_map, 95)
        binary_map = detection_map > threshold
        
        # Should detect ~5% as anomalies
        anomaly_fraction = np.mean(binary_map)
        self.assertLess(abs(anomaly_fraction - 0.05), 0.02)
        
    def test_adaptive_thresholding(self):
        """Test adaptive thresholding"""
        detector = AnomalyDetector(
            method=DetectionMethod.RX,
            threshold_method='adaptive'
        )
        detection_map = detector.detect(self.image)
        binary_map = detector.apply_threshold(detection_map)
        
        # Should produce binary map
        self.assertTrue(np.all((binary_map == 0) | (binary_map == 1)))


class TestDetectionComparison(unittest.TestCase):
    """Test comparison of detection methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        self.image[50:55, 50:55, :] = np.random.randn(5, 5, 50) * 0.3 + 0.9
        
    def test_multiple_methods(self):
        """Test multiple detection methods on same image"""
        methods = [
            DetectionMethod.RX,
            DetectionMethod.LOCAL_RX,
            DetectionMethod.KERNEL_RX
        ]
        
        results = {}
        for method in methods:
            detector = AnomalyDetector(method=method)
            detection_map = detector.detect(self.image)
            results[method.value] = detection_map
            
        # All should produce valid detection maps
        for detection_map in results.values():
            self.assertEqual(detection_map.shape, (100, 100))
            
    def test_detection_consistency(self):
        """Test that all methods detect anomaly region"""
        methods = [
            DetectionMethod.RX,
            DetectionMethod.LOCAL_RX
        ]
        
        for method in methods:
            detector = AnomalyDetector(method=method)
            detection_map = detector.detect(self.image)
            
            # Anomaly region should have higher scores
            anomaly_score = np.mean(detection_map[50:55, 50:55])
            background_score = np.mean(detection_map[0:20, 0:20])
            
            self.assertGreater(anomaly_score, background_score,
                             f"Method {method.value} failed to detect anomaly")


class TestDetectionEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_no_anomalies(self):
        """Test with image containing no anomalies"""
        # Uniform image
        image = np.ones((50, 50, 30)) * 0.5 + np.random.randn(50, 50, 30) * 0.01
        
        detector = AnomalyDetector(method=DetectionMethod.RX)
        detection_map = detector.detect(image)
        
        # Should still produce valid map
        self.assertEqual(detection_map.shape, (50, 50))
        
    def test_all_anomalies(self):
        """Test with image where everything is anomalous"""
        # High variance everywhere
        image = np.random.randn(50, 50, 30) * 2.0
        
        detector = AnomalyDetector(method=DetectionMethod.RX)
        detection_map = detector.detect(image)
        
        self.assertEqual(detection_map.shape, (50, 50))
        
    def test_single_pixel_anomaly(self):
        """Test detection of single pixel anomaly"""
        image = np.random.randn(50, 50, 30) * 0.1 + 0.5
        
        # Single pixel anomaly
        image[25, 25, :] = np.random.randn(30) * 0.5 + 1.5
        
        detector = AnomalyDetector(method=DetectionMethod.RX)
        detection_map = detector.detect(image)
        
        # Single pixel should have high score
        self.assertGreater(detection_map[25, 25], np.mean(detection_map))
        
    def test_constant_image(self):
        """Test with constant image"""
        image = np.ones((50, 50, 30)) * 0.5
        
        detector = AnomalyDetector(method=DetectionMethod.RX)
        
        # Should handle gracefully
        try:
            detection_map = detector.detect(image)
            self.assertEqual(detection_map.shape, (50, 50))
        except Exception as e:
            # Should not raise exception
            self.fail(f"Detection failed on constant image: {e}")


class TestDetectionPerformance(unittest.TestCase):
    """Test detection performance"""
    
    def test_detection_speed(self):
        """Test detection speed"""
        import time
        
        image = np.random.randn(256, 256, 100) * 0.2 + 0.5
        
        detector = AnomalyDetector(method=DetectionMethod.RX)
        
        start = time.time()
        detection_map = detector.detect(image)
        elapsed = time.time() - start
        
        # Should complete in < 5 seconds
        self.assertLess(elapsed, 5)
        
    def test_batch_detection(self):
        """Test detecting anomalies in multiple images"""
        import time
        
        images = [np.random.randn(128, 128, 50) * 0.2 + 0.5 for _ in range(10)]
        
        detector = AnomalyDetector(method=DetectionMethod.RX)
        
        start = time.time()
        for image in images:
            detection_map = detector.detect(image)
        elapsed = time.time() - start
        
        # Should process > 2 images/second
        fps = len(images) / elapsed
        self.assertGreater(fps, 2)


class TestDetectionVisualization(unittest.TestCase):
    """Test detection visualization helpers"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        self.image[50:55, 50:55, :] = np.random.randn(5, 5, 50) * 0.3 + 0.9
        
    def test_generate_heatmap(self):
        """Test heatmap generation"""
        detector = AnomalyDetector(method=DetectionMethod.RX)
        detection_map = detector.detect(self.image)
        
        # Normalize to [0, 1] for visualization
        heatmap = (detection_map - detection_map.min()) / (detection_map.max() - detection_map.min())
        
        self.assertEqual(heatmap.shape, (100, 100))
        self.assertTrue(np.all(heatmap >= 0))
        self.assertTrue(np.all(heatmap <= 1))
        
    def test_overlay_on_rgb(self):
        """Test overlaying detection on RGB image"""
        detector = AnomalyDetector(method=DetectionMethod.RX)
        detection_map = detector.detect(self.image)
        
        # Create RGB representation (using first 3 bands)
        rgb = self.image[:, :, [20, 30, 40]]
        
        # Overlay detection (red channel)
        overlay = rgb.copy()
        overlay[:, :, 0] += detection_map / detection_map.max() * 0.5
        
        self.assertEqual(overlay.shape, rgb.shape)


if __name__ == '__main__':
    unittest.main()
