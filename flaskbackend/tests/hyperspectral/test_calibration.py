"""
Unit tests for hyperspectral calibration module

Tests radiometric, spectral, and geometric calibration functionality.
"""

import unittest
import numpy as np
import tempfile
import json
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ai_nutrition.hyperspectral.calibration import (
    HyperspectralCalibrator,
    CalibrationConfig,
    RadiometricCalibration,
    SpectralCalibration,
    GeometricCalibration,
    CalibrationCertificate
)


class TestHyperspectralCalibrator(unittest.TestCase):
    """Test cases for HyperspectralCalibrator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calibrator = HyperspectralCalibrator()
        self.image_shape = (100, 100, 50)
        
    def test_initialization(self):
        """Test calibrator initialization"""
        self.assertIsNotNone(self.calibrator)
        self.assertIsInstance(self.calibrator.config, CalibrationConfig)
        
    def test_initialization_with_config(self):
        """Test calibrator initialization with custom config"""
        config = CalibrationConfig(
            dark_current_threshold=0.02,
            min_snr=80.0
        )
        calibrator = HyperspectralCalibrator(config)
        self.assertEqual(calibrator.config.dark_current_threshold, 0.02)
        self.assertEqual(calibrator.config.min_snr, 80.0)


class TestRadiometricCalibration(unittest.TestCase):
    """Test radiometric calibration functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calibrator = HyperspectralCalibrator()
        self.image_shape = (100, 100, 50)
        
        # Generate synthetic dark frames (low signal + noise)
        self.dark_frames = [
            np.random.randn(*self.image_shape) * 0.01 + 0.005
            for _ in range(10)
        ]
        
        # Generate synthetic white frames (high signal + noise)
        self.white_frames = [
            np.random.randn(*self.image_shape) * 0.1 + 0.8
            for _ in range(10)
        ]
        
    def test_dark_frame_averaging(self):
        """Test dark current averaging"""
        cal = self.calibrator.calibrate_radiometric(
            self.dark_frames,
            self.white_frames
        )
        
        self.assertIsInstance(cal, RadiometricCalibration)
        self.assertEqual(cal.dark_current.shape, self.image_shape)
        
        # Dark current should be close to 0.005
        self.assertAlmostEqual(np.mean(cal.dark_current), 0.005, places=2)
        
    def test_white_reference_normalization(self):
        """Test white reference normalization"""
        cal = self.calibrator.calibrate_radiometric(
            self.dark_frames,
            self.white_frames
        )
        
        self.assertEqual(cal.white_reference.shape, self.image_shape)
        
        # White reference should be close to 0.8
        self.assertAlmostEqual(np.mean(cal.white_reference), 0.8, places=1)
        
    def test_gain_offset_computation(self):
        """Test gain and offset computation"""
        cal = self.calibrator.calibrate_radiometric(
            self.dark_frames,
            self.white_frames
        )
        
        self.assertEqual(cal.gain.shape, self.image_shape)
        self.assertEqual(cal.offset.shape, self.image_shape)
        
        # Gain should be positive
        self.assertTrue(np.all(cal.gain > 0))
        
    def test_apply_calibration(self):
        """Test applying radiometric calibration"""
        cal = self.calibrator.calibrate_radiometric(
            self.dark_frames,
            self.white_frames
        )
        
        # Generate raw image
        raw_image = np.random.randn(*self.image_shape) * 0.2 + 0.5
        
        # Apply calibration
        calibrated = self.calibrator.apply_radiometric_calibration(raw_image, cal)
        
        self.assertEqual(calibrated.shape, self.image_shape)
        
        # Calibrated image should be in reasonable range
        self.assertTrue(np.all(calibrated >= 0))
        self.assertTrue(np.all(calibrated <= 2.0))
        
    def test_linearity_correction(self):
        """Test linearity correction"""
        linearity_data = {
            'coefficients': np.array([1.0, 0.1, -0.01])  # polynomial coefficients
        }
        
        cal = self.calibrator.calibrate_radiometric(
            self.dark_frames,
            self.white_frames,
            linearity_data=linearity_data
        )
        
        self.assertIsNotNone(cal.linearity_coefficients)
        self.assertEqual(len(cal.linearity_coefficients), 3)
        
    def test_insufficient_frames(self):
        """Test error handling for insufficient frames"""
        with self.assertRaises(ValueError):
            self.calibrator.calibrate_radiometric(
                self.dark_frames[:2],  # Only 2 frames
                self.white_frames
            )


class TestSpectralCalibration(unittest.TestCase):
    """Test spectral calibration functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calibrator = HyperspectralCalibrator()
        
        # Nominal wavelengths
        self.wavelengths = np.linspace(400, 1000, 100)
        
        # Simulated reference peaks (Mercury emission lines)
        self.reference_peaks = np.array([435.8, 546.1, 611.9, 780.0, 894.3])
        self.reference_wavelengths = np.array([435.8, 546.1, 611.9, 780.0, 894.3])
        
    def test_wavelength_calibration(self):
        """Test wavelength calibration"""
        cal = self.calibrator.calibrate_spectral(
            self.wavelengths,
            self.reference_peaks,
            self.reference_wavelengths
        )
        
        self.assertIsInstance(cal, SpectralCalibration)
        self.assertIsNotNone(cal.calibration_coefficients)
        
    def test_wavelength_accuracy(self):
        """Test wavelength calibration accuracy"""
        # Add small errors to peaks
        measured_peaks = self.reference_peaks + np.random.randn(5) * 0.5
        
        cal = self.calibrator.calibrate_spectral(
            self.wavelengths,
            measured_peaks,
            self.reference_wavelengths
        )
        
        # RMSE should be small
        self.assertIsNotNone(cal.wavelength_rmse)
        self.assertLess(cal.wavelength_rmse, 1.0)  # < 1nm
        
    def test_fwhm_estimation(self):
        """Test FWHM estimation"""
        cal = self.calibrator.calibrate_spectral(
            self.wavelengths,
            self.reference_peaks,
            self.reference_wavelengths
        )
        
        self.assertIsNotNone(cal.fwhm)
        self.assertGreater(cal.fwhm, 0)
        self.assertLess(cal.fwhm, 20)  # Reasonable FWHM
        
    def test_invalid_peaks(self):
        """Test error handling for invalid peaks"""
        with self.assertRaises(ValueError):
            self.calibrator.calibrate_spectral(
                self.wavelengths,
                np.array([100, 200]),  # Too few peaks
                np.array([100, 200])
            )


class TestGeometricCalibration(unittest.TestCase):
    """Test geometric calibration functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calibrator = HyperspectralCalibrator()
        
        # Generate synthetic checkerboard images
        self.checkerboard_images = [
            self._generate_checkerboard(100, 100, 10, 10)
            for _ in range(5)
        ]
        self.square_size_mm = 10.0
        
    def _generate_checkerboard(self, h, w, rows, cols):
        """Generate synthetic checkerboard pattern"""
        image = np.zeros((h, w))
        square_h = h // rows
        square_w = w // cols
        
        for i in range(rows):
            for j in range(cols):
                if (i + j) % 2 == 0:
                    image[i*square_h:(i+1)*square_h, 
                          j*square_w:(j+1)*square_w] = 1.0
        
        return image
        
    def test_geometric_calibration(self):
        """Test geometric calibration"""
        cal = self.calibrator.calibrate_geometric(
            self.checkerboard_images,
            self.square_size_mm
        )
        
        self.assertIsInstance(cal, GeometricCalibration)
        
    def test_spatial_resolution(self):
        """Test spatial resolution measurement"""
        cal = self.calibrator.calibrate_geometric(
            self.checkerboard_images,
            self.square_size_mm
        )
        
        self.assertIsNotNone(cal.spatial_resolution)
        self.assertGreater(cal.spatial_resolution, 0)
        
    def test_distortion_mapping(self):
        """Test distortion map generation"""
        cal = self.calibrator.calibrate_geometric(
            self.checkerboard_images,
            self.square_size_mm
        )
        
        # Distortion map should be generated
        self.assertIsNotNone(cal.distortion_map)


class TestCalibrationValidation(unittest.TestCase):
    """Test calibration validation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calibrator = HyperspectralCalibrator()
        self.image_shape = (100, 100, 50)
        
        # Generate calibration data
        dark_frames = [
            np.random.randn(*self.image_shape) * 0.01 + 0.005
            for _ in range(10)
        ]
        white_frames = [
            np.random.randn(*self.image_shape) * 0.05 + 0.9
            for _ in range(10)
        ]
        
        self.rad_cal = self.calibrator.calibrate_radiometric(
            dark_frames,
            white_frames
        )
        
        # Generate test image
        self.test_image = np.random.randn(*self.image_shape) * 0.1 + 0.5
        self.calibrated_image = self.calibrator.apply_radiometric_calibration(
            self.test_image,
            self.rad_cal
        )
        
    def test_validation_metrics(self):
        """Test validation metric computation"""
        validation = self.calibrator.validate(
            self.calibrated_image,
            {'radiometric': self.rad_cal}
        )
        
        self.assertIsInstance(validation, dict)
        self.assertIn('dark_current_level', validation)
        self.assertIn('snr', validation)
        
    def test_snr_calculation(self):
        """Test SNR calculation"""
        validation = self.calibrator.validate(
            self.calibrated_image,
            {'radiometric': self.rad_cal}
        )
        
        self.assertGreater(validation['snr'], 0)
        
    def test_validation_passed(self):
        """Test validation pass/fail logic"""
        validation = self.calibrator.validate(
            self.calibrated_image,
            {'radiometric': self.rad_cal}
        )
        
        self.assertIn('validation_passed', validation)
        self.assertIsInstance(validation['validation_passed'], bool)


class TestCalibrationCertificate(unittest.TestCase):
    """Test calibration certificate generation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calibrator = HyperspectralCalibrator()
        self.image_shape = (100, 100, 50)
        
        # Generate calibrations
        dark_frames = [np.random.randn(*self.image_shape) * 0.01 for _ in range(10)]
        white_frames = [np.random.randn(*self.image_shape) * 0.1 + 0.9 for _ in range(10)]
        
        self.rad_cal = self.calibrator.calibrate_radiometric(dark_frames, white_frames)
        
        wavelengths = np.linspace(400, 1000, 50)
        reference_peaks = np.array([435.8, 546.1, 611.9, 780.0, 894.3])
        reference_wavelengths = reference_peaks
        
        self.spec_cal = self.calibrator.calibrate_spectral(
            wavelengths,
            reference_peaks,
            reference_wavelengths
        )
        
        # Validation results
        test_image = np.random.randn(*self.image_shape) * 0.1 + 0.5
        calibrated = self.calibrator.apply_radiometric_calibration(test_image, self.rad_cal)
        self.validation = self.calibrator.validate(
            calibrated,
            {'radiometric': self.rad_cal, 'spectral': self.spec_cal}
        )
        
    def test_certificate_generation(self):
        """Test certificate generation"""
        cert = self.calibrator.generate_certificate(
            self.rad_cal,
            self.spec_cal,
            None,  # No geometric calibration
            self.validation
        )
        
        self.assertIsInstance(cert, CalibrationCertificate)
        
    def test_certificate_contents(self):
        """Test certificate contains required fields"""
        cert = self.calibrator.generate_certificate(
            self.rad_cal,
            self.spec_cal,
            None,
            self.validation
        )
        
        cert_dict = cert.to_dict()
        
        self.assertIn('radiometric', cert_dict)
        self.assertIn('spectral', cert_dict)
        self.assertIn('validation', cert_dict)
        self.assertIn('timestamp', cert_dict)
        
    def test_certificate_serialization(self):
        """Test certificate JSON serialization"""
        cert = self.calibrator.generate_certificate(
            self.rad_cal,
            self.spec_cal,
            None,
            self.validation
        )
        
        # Should be JSON serializable
        cert_dict = cert.to_dict()
        json_str = json.dumps(cert_dict, default=str)
        
        self.assertIsInstance(json_str, str)
        
    def test_certificate_save_load(self):
        """Test saving and loading certificate"""
        cert = self.calibrator.generate_certificate(
            self.rad_cal,
            self.spec_cal,
            None,
            self.validation
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(cert.to_dict(), f, default=str)
            temp_path = f.name
        
        try:
            # Load from file
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
            
            self.assertIn('radiometric', loaded)
            self.assertIn('spectral', loaded)
        finally:
            os.unlink(temp_path)


class TestCalibrationEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calibrator = HyperspectralCalibrator()
        
    def test_single_channel_image(self):
        """Test calibration with single-channel image"""
        dark_frames = [np.random.randn(100, 100, 1) * 0.01 for _ in range(10)]
        white_frames = [np.random.randn(100, 100, 1) * 0.1 + 0.9 for _ in range(10)]
        
        cal = self.calibrator.calibrate_radiometric(dark_frames, white_frames)
        
        self.assertEqual(cal.dark_current.shape, (100, 100, 1))
        
    def test_high_dimensional_image(self):
        """Test calibration with many spectral bands"""
        dark_frames = [np.random.randn(50, 50, 200) * 0.01 for _ in range(10)]
        white_frames = [np.random.randn(50, 50, 200) * 0.1 + 0.9 for _ in range(10)]
        
        cal = self.calibrator.calibrate_radiometric(dark_frames, white_frames)
        
        self.assertEqual(cal.dark_current.shape, (50, 50, 200))
        
    def test_zero_white_reference(self):
        """Test handling of zero values in white reference"""
        dark_frames = [np.zeros((50, 50, 10)) for _ in range(10)]
        white_frames = [np.zeros((50, 50, 10)) for _ in range(10)]
        
        # Should handle division by zero
        cal = self.calibrator.calibrate_radiometric(dark_frames, white_frames)
        
        # Gain should not be inf or nan
        self.assertFalse(np.any(np.isinf(cal.gain)))
        self.assertFalse(np.any(np.isnan(cal.gain)))
        
    def test_negative_values(self):
        """Test handling of negative values"""
        dark_frames = [np.random.randn(50, 50, 10) for _ in range(10)]
        white_frames = [np.random.randn(50, 50, 10) for _ in range(10)]
        
        cal = self.calibrator.calibrate_radiometric(dark_frames, white_frames)
        
        # Should complete without errors
        self.assertIsNotNone(cal)
        
    def test_mismatched_frame_sizes(self):
        """Test error handling for mismatched frame sizes"""
        dark_frames = [np.random.randn(100, 100, 50) for _ in range(10)]
        white_frames = [np.random.randn(100, 100, 30) for _ in range(10)]  # Different bands
        
        with self.assertRaises(ValueError):
            self.calibrator.calibrate_radiometric(dark_frames, white_frames)


class TestCalibrationPerformance(unittest.TestCase):
    """Test calibration performance"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calibrator = HyperspectralCalibrator()
        
    def test_large_image_calibration(self):
        """Test calibration with large images"""
        import time
        
        # Large images: 512x512 with 100 bands
        dark_frames = [np.random.randn(512, 512, 100) * 0.01 for _ in range(10)]
        white_frames = [np.random.randn(512, 512, 100) * 0.1 + 0.9 for _ in range(10)]
        
        start = time.time()
        cal = self.calibrator.calibrate_radiometric(dark_frames, white_frames)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 10 seconds)
        self.assertLess(elapsed, 10.0)
        
    def test_calibration_application_speed(self):
        """Test calibration application speed"""
        import time
        
        # Generate calibration
        dark_frames = [np.random.randn(256, 256, 50) * 0.01 for _ in range(10)]
        white_frames = [np.random.randn(256, 256, 50) * 0.1 + 0.9 for _ in range(10)]
        cal = self.calibrator.calibrate_radiometric(dark_frames, white_frames)
        
        # Apply to many images
        images = [np.random.randn(256, 256, 50) * 0.2 + 0.5 for _ in range(100)]
        
        start = time.time()
        for img in images:
            calibrated = self.calibrator.apply_radiometric_calibration(img, cal)
        elapsed = time.time() - start
        
        # Should process > 10 images/second
        fps = len(images) / elapsed
        self.assertGreater(fps, 10)


if __name__ == '__main__':
    unittest.main()
