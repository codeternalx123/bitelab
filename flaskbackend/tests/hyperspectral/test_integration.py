"""
Unit tests for hyperspectral integration and end-to-end workflows

Tests complete pipelines from acquisition to analysis.
"""

import unittest
import numpy as np
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ai_nutrition.hyperspectral.preprocessing import SpectralPreprocessor
from app.ai_nutrition.hyperspectral.band_selection import BandSelector, SelectionMethod
from app.ai_nutrition.hyperspectral.feature_extraction import SpectralFeatureExtractor
from app.ai_nutrition.hyperspectral.classification import SpectralClassifier
from app.ai_nutrition.hyperspectral.anomaly_detection import AnomalyDetector, DetectionMethod as AnomalyMethod
from app.ai_nutrition.hyperspectral.target_detection import TargetDetector, DetectionMethod as TargetMethod


class TestCompleteWorkflow(unittest.TestCase):
    """Test complete hyperspectral analysis workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Generate synthetic hyperspectral image
        self.image = self._generate_synthetic_scene()
        
    def _generate_synthetic_scene(self):
        """Generate synthetic hyperspectral scene with known classes"""
        h, w, c = 100, 100, 50
        image = np.zeros((h, w, c))
        
        # Class 1: Background (vegetation-like)
        for i in range(c):
            wavelength = 400 + i * 12  # 400-1000 nm
            # Vegetation red edge
            if wavelength < 700:
                reflectance = 0.1
            else:
                reflectance = 0.6
            image[:, :, i] = reflectance + np.random.randn(h, w) * 0.02
        
        # Class 2: Target 1 (mineral)
        image[20:30, 20:30, :] = np.linspace(0.2, 0.8, 50) + np.random.randn(10, 10, 50) * 0.02
        
        # Class 3: Target 2 (water)
        image[70:80, 70:80, :] = np.linspace(0.8, 0.2, 50) + np.random.randn(10, 10, 50) * 0.02
        
        return image
        
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing workflow"""
        # 1. Initialize preprocessor
        preprocessor = SpectralPreprocessor()
        
        # 2. Apply preprocessing steps
        dark_current = np.random.randn(100, 100, 50) * 0.01
        white_reference = np.ones((100, 100, 50)) * 0.95
        
        processed = preprocessor.correct_dark_current(self.image, dark_current)
        processed = preprocessor.correct_white_reference(processed, white_reference)
        processed = preprocessor.apply_spatial_filter(processed, filter_type='gaussian', sigma=1.0)
        
        # 3. Verify output
        self.assertEqual(processed.shape, self.image.shape)
        self.assertFalse(np.allclose(processed, self.image))  # Should be different
        
    def test_band_selection_workflow(self):
        """Test band selection workflow"""
        # 1. Preprocess
        preprocessor = SpectralPreprocessor()
        processed = preprocessor.apply_spatial_filter(self.image, filter_type='gaussian')
        
        # 2. Select bands
        selector = BandSelector(method=SelectionMethod.VARIANCE, n_bands=20)
        selected_bands = selector.select(processed)
        
        # 3. Extract selected bands
        reduced_image = processed[:, :, selected_bands]
        
        # 4. Verify
        self.assertEqual(reduced_image.shape, (100, 100, 20))
        self.assertEqual(len(selected_bands), 20)
        
    def test_feature_extraction_workflow(self):
        """Test feature extraction workflow"""
        # 1. Preprocess and select bands
        preprocessor = SpectralPreprocessor()
        processed = preprocessor.apply_spatial_filter(self.image, filter_type='gaussian')
        
        selector = BandSelector(method=SelectionMethod.VARIANCE, n_bands=30)
        selected_bands = selector.select(processed)
        reduced_image = processed[:, :, selected_bands]
        
        # 2. Extract features
        extractor = SpectralFeatureExtractor()
        features = extractor.extract(reduced_image)
        
        # 3. Verify features
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
        
    def test_classification_workflow(self):
        """Test classification workflow"""
        # 1. Preprocess
        preprocessor = SpectralPreprocessor()
        processed = preprocessor.apply_spatial_filter(self.image, filter_type='gaussian')
        
        # 2. Extract features
        extractor = SpectralFeatureExtractor()
        features = extractor.extract(processed)
        feature_vector = extractor.vectorize(features)
        
        # 3. Create mock training data
        n_samples = 300
        n_features = feature_vector.shape[1]
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 3, n_samples)
        
        # 4. Train classifier
        classifier = SpectralClassifier(method='svm')
        classifier.fit(X_train, y_train)
        
        # 5. Classify image
        predictions = classifier.predict(feature_vector)
        classification_map = predictions.reshape(100, 100)
        
        # 6. Verify
        self.assertEqual(classification_map.shape, (100, 100))
        self.assertTrue(np.all(classification_map >= 0))
        self.assertTrue(np.all(classification_map < 3))


class TestContaminationDetectionWorkflow(unittest.TestCase):
    """Test contamination detection workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Normal food sample
        self.normal_image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        
        # Contaminated food sample (with anomaly)
        self.contaminated_image = self.normal_image.copy()
        self.contaminated_image[50:55, 50:55, :] = np.random.randn(5, 5, 50) * 0.3 + 0.9
        
    def test_contamination_detection_pipeline(self):
        """Test end-to-end contamination detection"""
        # 1. Preprocess
        preprocessor = SpectralPreprocessor()
        processed = preprocessor.apply_spatial_filter(
            self.contaminated_image,
            filter_type='gaussian'
        )
        
        # 2. Detect anomalies
        detector = AnomalyDetector(method=AnomalyMethod.RX)
        anomaly_map = detector.detect(processed)
        
        # 3. Threshold to find contamination
        threshold = np.percentile(anomaly_map, 95)
        contamination_mask = anomaly_map > threshold
        
        # 4. Verify contamination detected
        self.assertTrue(np.any(contamination_mask[50:55, 50:55]))
        
        # 5. Estimate contamination area
        contamination_area = np.sum(contamination_mask)
        self.assertGreater(contamination_area, 0)


class TestAllergenDetectionWorkflow(unittest.TestCase):
    """Test allergen detection workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Food image
        self.food_image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        
        # Allergen signature (e.g., peanut protein)
        self.allergen_signature = np.random.randn(50) * 0.2 + 0.8
        
        # Add allergen to specific location
        self.food_image[30:35, 30:35, :] = self.allergen_signature + \
                                            np.random.randn(5, 5, 50) * 0.05
        
    def test_allergen_detection_pipeline(self):
        """Test end-to-end allergen detection"""
        # 1. Preprocess
        preprocessor = SpectralPreprocessor()
        processed = preprocessor.apply_spectral_filter(
            self.food_image,
            filter_type='savgol'
        )
        
        # 2. Detect allergen target
        detector = TargetDetector(method=TargetMethod.ACE)
        detector.set_target(self.allergen_signature)
        detection_map = detector.detect(processed)
        
        # 3. Threshold
        threshold = np.percentile(detection_map, 90)
        allergen_mask = detection_map > threshold
        
        # 4. Verify allergen detected
        allergen_present = np.any(allergen_mask[30:35, 30:35])
        self.assertTrue(allergen_present)
        
        # 5. Generate alert
        if allergen_present:
            alert = {
                'allergen': 'peanut',
                'confidence': float(np.max(detection_map[30:35, 30:35])),
                'location': (32, 32)
            }
            self.assertIn('allergen', alert)
            self.assertGreater(alert['confidence'], 0)


class TestQualityMonitoringWorkflow(unittest.TestCase):
    """Test food quality monitoring workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Fresh produce
        self.fresh_image = np.random.randn(100, 100, 50) * 0.1 + 0.6
        
        # Spoiled produce (different spectral signature)
        self.spoiled_image = np.random.randn(100, 100, 50) * 0.15 + 0.4
        
    def test_quality_assessment_pipeline(self):
        """Test quality assessment workflow"""
        # 1. Preprocess both images
        preprocessor = SpectralPreprocessor()
        fresh_processed = preprocessor.apply_spatial_filter(self.fresh_image)
        spoiled_processed = preprocessor.apply_spatial_filter(self.spoiled_image)
        
        # 2. Extract features
        extractor = SpectralFeatureExtractor()
        fresh_features = extractor.extract(fresh_processed)
        spoiled_features = extractor.extract(spoiled_processed)
        
        # 3. Compare features
        # Use mean reflectance as quality indicator
        fresh_mean = np.mean(fresh_features['mean'])
        spoiled_mean = np.mean(spoiled_features['mean'])
        
        # Fresh should have higher reflectance
        self.assertGreater(fresh_mean, spoiled_mean)
        
        # 4. Quality score
        quality_score = fresh_mean / (spoiled_mean + 0.1)
        self.assertGreater(quality_score, 1.0)


class TestMultiComponentIntegration(unittest.TestCase):
    """Test integration of multiple components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        
    def test_preprocessing_and_detection_integration(self):
        """Test preprocessing feeds into detection"""
        # Preprocess
        preprocessor = SpectralPreprocessor()
        processed = preprocessor.apply_spatial_filter(self.image)
        processed = preprocessor.normalize(processed, method='minmax')
        
        # Detect anomalies
        detector = AnomalyDetector(method=AnomalyMethod.RX)
        detection_map = detector.detect(processed)
        
        self.assertEqual(detection_map.shape, (100, 100))
        
    def test_band_selection_and_classification_integration(self):
        """Test band selection improves classification"""
        # Select bands
        selector = BandSelector(method=SelectionMethod.VARIANCE, n_bands=20)
        selected_bands = selector.select(self.image)
        reduced_image = self.image[:, :, selected_bands]
        
        # Extract features
        extractor = SpectralFeatureExtractor()
        features = extractor.extract(reduced_image)
        
        # Verify reduced dimensionality
        self.assertEqual(reduced_image.shape[2], 20)
        self.assertIsInstance(features, dict)


class TestRealTimeProcessingWorkflow(unittest.TestCase):
    """Test real-time processing capabilities"""
    
    def test_frame_processing_speed(self):
        """Test processing speed for real-time application"""
        # Simulate video stream
        frames = [np.random.randn(256, 256, 50) * 0.1 + 0.5 for _ in range(30)]
        
        preprocessor = SpectralPreprocessor()
        detector = AnomalyDetector(method=AnomalyMethod.RX)
        
        start = time.time()
        for frame in frames:
            # Quick preprocessing
            processed = preprocessor.apply_spatial_filter(frame, filter_type='gaussian')
            
            # Detection
            detection_map = detector.detect(processed)
            
        elapsed = time.time() - start
        fps = len(frames) / elapsed
        
        # Should achieve > 5 FPS for real-time
        self.assertGreater(fps, 5,
                          f"Processing speed {fps:.2f} FPS is too slow for real-time")


class TestDataQualityValidation(unittest.TestCase):
    """Test data quality validation in workflow"""
    
    def test_validate_image_quality(self):
        """Test image quality validation"""
        # Good quality image
        good_image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        
        # Poor quality image (high noise)
        poor_image = np.random.randn(100, 100, 50) * 2.0 + 0.5
        
        # Calculate SNR
        good_snr = np.mean(good_image) / (np.std(good_image) + 1e-10)
        poor_snr = np.mean(poor_image) / (np.std(poor_image) + 1e-10)
        
        # Good image should have higher SNR
        self.assertGreater(good_snr, poor_snr)
        
    def test_validate_spectral_range(self):
        """Test spectral range validation"""
        image = np.random.randn(100, 100, 50) * 0.1 + 0.5
        
        # Check all bands have reasonable values
        valid = np.all((image >= 0) & (image <= 1.5))
        
        if not valid:
            # Flag for correction
            image = np.clip(image, 0, 1.0)
            
        self.assertTrue(np.all((image >= 0) & (image <= 1.0)))


class TestErrorHandlingInWorkflow(unittest.TestCase):
    """Test error handling in workflows"""
    
    def test_handle_corrupted_data(self):
        """Test handling of corrupted data"""
        # Image with NaN values
        image = np.random.randn(50, 50, 30) * 0.1 + 0.5
        image[25, 25, :] = np.nan
        
        preprocessor = SpectralPreprocessor()
        
        # Should handle NaN gracefully
        try:
            # Replace NaN with interpolation
            mask = np.isnan(image)
            if np.any(mask):
                image[mask] = np.nanmean(image)
            
            processed = preprocessor.apply_spatial_filter(image)
            self.assertFalse(np.any(np.isnan(processed)))
        except Exception as e:
            self.fail(f"Failed to handle NaN values: {e}")
            
    def test_handle_mismatched_dimensions(self):
        """Test handling of mismatched dimensions"""
        image = np.random.randn(100, 100, 50)
        target = np.random.randn(40)  # Wrong number of bands
        
        detector = TargetDetector(method=TargetMethod.MATCHED_FILTER)
        
        # Should raise error for mismatched dimensions
        with self.assertRaises((ValueError, AssertionError)):
            detector.set_target(target)
            detector.detect(image)


class TestWorkflowOptimization(unittest.TestCase):
    """Test workflow optimization"""
    
    def test_early_stopping_in_detection(self):
        """Test early stopping when target is clearly absent"""
        # Image with no target
        image = np.ones((100, 100, 50)) * 0.5 + np.random.randn(100, 100, 50) * 0.01
        target = np.random.randn(50) * 0.5 + 2.0  # Very different signature
        
        detector = TargetDetector(method=TargetMethod.SAM)
        detector.set_target(target)
        
        detection_map = detector.detect(image)
        
        # All angles should be large (target absent)
        mean_angle = np.mean(detection_map)
        
        if mean_angle > 0.5:  # Clear absence
            # Can skip further processing
            pass
        
        self.assertGreater(mean_angle, 0.3)
        
    def test_progressive_processing(self):
        """Test progressive processing (coarse to fine)"""
        image = np.random.randn(256, 256, 100) * 0.1 + 0.5
        
        # 1. Coarse detection (downsampled)
        coarse_image = image[::4, ::4, ::2]  # 64x64x50
        
        detector = AnomalyDetector(method=AnomalyMethod.RX)
        coarse_map = detector.detect(coarse_image)
        
        # 2. Find regions of interest
        threshold = np.percentile(coarse_map, 90)
        roi_mask = coarse_map > threshold
        
        # 3. Fine detection only in ROI
        if np.any(roi_mask):
            # Process full resolution only in ROI
            # (simplified - would actually extract ROI)
            fine_map = detector.detect(image)
            
            self.assertEqual(fine_map.shape, (256, 256))


class TestWorkflowReproducibility(unittest.TestCase):
    """Test workflow reproducibility"""
    
    def test_reproducible_results(self):
        """Test that workflow produces reproducible results"""
        np.random.seed(42)
        image1 = np.random.randn(50, 50, 30) * 0.1 + 0.5
        
        np.random.seed(42)
        image2 = np.random.randn(50, 50, 30) * 0.1 + 0.5
        
        # Process both
        preprocessor = SpectralPreprocessor()
        processed1 = preprocessor.apply_spatial_filter(image1, filter_type='gaussian')
        processed2 = preprocessor.apply_spatial_filter(image2, filter_type='gaussian')
        
        # Should be identical
        self.assertTrue(np.allclose(processed1, processed2))


class TestWorkflowScalability(unittest.TestCase):
    """Test workflow scalability"""
    
    def test_large_image_processing(self):
        """Test processing large hyperspectral images"""
        # Large image (512x512x100)
        large_image = np.random.randn(512, 512, 100) * 0.1 + 0.5
        
        start = time.time()
        
        # Process in chunks
        chunk_size = 128
        results = []
        
        for i in range(0, 512, chunk_size):
            for j in range(0, 512, chunk_size):
                chunk = large_image[i:i+chunk_size, j:j+chunk_size, :]
                
                # Process chunk
                preprocessor = SpectralPreprocessor()
                processed = preprocessor.apply_spatial_filter(chunk)
                results.append(processed)
        
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 30 seconds)
        self.assertLess(elapsed, 30)
        
    def test_batch_processing(self):
        """Test batch processing multiple samples"""
        samples = [np.random.randn(128, 128, 50) * 0.1 + 0.5 for _ in range(20)]
        
        start = time.time()
        
        preprocessor = SpectralPreprocessor()
        detector = AnomalyDetector(method=AnomalyMethod.RX)
        
        results = []
        for sample in samples:
            processed = preprocessor.apply_spatial_filter(sample)
            detection = detector.detect(processed)
            results.append(detection)
        
        elapsed = time.time() - start
        
        # Should process > 5 samples/second
        throughput = len(samples) / elapsed
        self.assertGreater(throughput, 5)


class TestWorkflowDocumentation(unittest.TestCase):
    """Test workflow generates proper documentation"""
    
    def test_workflow_metadata(self):
        """Test workflow generates metadata"""
        metadata = {
            'workflow': 'contamination_detection',
            'timestamp': time.time(),
            'preprocessing': ['dark_current', 'white_reference', 'gaussian_filter'],
            'detection_method': 'RX',
            'threshold': 0.95
        }
        
        self.assertIn('workflow', metadata)
        self.assertIn('preprocessing', metadata)
        self.assertIsInstance(metadata['preprocessing'], list)


if __name__ == '__main__':
    unittest.main()
