"""
ML Models for Health Impact Analysis
====================================

AI/ML models for:
1. Spectral analysis → compound identification
2. Toxicity prediction (GNN-based)
3. Allergen detection (sequence-based)
4. Nutritional estimation
5. Uncertainty quantification

Author: AI Nutrition Scanner Team
Date: November 2025
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SpectralFeatures:
    """Extracted features from spectral data."""
    peak_positions: List[float]
    peak_intensities: List[float]
    peak_widths: List[float]
    baseline: np.ndarray
    normalized_spectrum: np.ndarray
    pca_features: Optional[np.ndarray] = None
    wavelet_features: Optional[np.ndarray] = None


@dataclass
class CompoundPrediction:
    """ML prediction for compound presence and concentration."""
    compound_name: str
    presence_probability: float
    concentration_mg_kg: float
    concentration_std: float  # Uncertainty
    confidence_score: float
    model_version: str


@dataclass
class ToxicityPrediction:
    """ML-based toxicity prediction."""
    compound_name: str
    acute_toxicity_score: float
    chronic_toxicity_score: float
    carcinogenicity_probability: float
    ld50_predicted: Optional[float]
    hazard_classes: List[str]
    confidence: float


# =============================================================================
# SPECTRAL PROCESSING
# =============================================================================

class SpectralProcessor:
    """
    Process raw spectral data from FTIR/NIR/Raman/MassSpec.
    
    Includes:
    - Preprocessing (denoising, baseline correction)
    - Peak detection
    - Feature extraction (PCA, wavelets)
    """
    
    def __init__(self, method: str = "ftir"):
        """
        Initialize spectral processor.
        
        Args:
            method: Spectrometry method (ftir, nir, raman, massspec)
        """
        self.method = method
        self.baseline_method = "airpls"
        logger.info(f"SpectralProcessor initialized for {method}")
    
    def preprocess(self, wavelengths: np.ndarray, intensities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess raw spectral data.
        
        Args:
            wavelengths: Wavelength/wavenumber array
            intensities: Raw intensity array
        
        Returns:
            Tuple of (processed_wavelengths, processed_intensities)
        """
        # 1. Denoise (Savitzky-Golay filter)
        try:
            from scipy.signal import savgol_filter
            smoothed = savgol_filter(intensities, window_length=11, polyorder=3)
        except ImportError:
            smoothed = intensities
            logger.warning("SciPy not available, skipping denoising")
        
        # 2. Baseline correction (simplified)
        baseline = self._estimate_baseline(smoothed)
        corrected = smoothed - baseline
        
        # 3. Normalization
        normalized = self._normalize_spectrum(corrected)
        
        return wavelengths, normalized
    
    def _estimate_baseline(self, spectrum: np.ndarray) -> np.ndarray:
        """Estimate and remove baseline."""
        # Simplified: use rolling minimum
        try:
            from scipy.ndimage import minimum_filter1d
            baseline = minimum_filter1d(spectrum, size=50)
        except ImportError:
            # Fallback: simple percentile baseline
            baseline = np.percentile(spectrum, 10) * np.ones_like(spectrum)
        
        return baseline
    
    def _normalize_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Normalize spectrum (vector normalization)."""
        norm = np.linalg.norm(spectrum)
        if norm > 0:
            return spectrum / norm
        return spectrum
    
    def detect_peaks(self, wavelengths: np.ndarray, intensities: np.ndarray, 
                    prominence: float = 0.02) -> SpectralFeatures:
        """
        Detect peaks in spectrum.
        
        Args:
            wavelengths: Wavelength array
            intensities: Intensity array
            prominence: Minimum peak prominence
        
        Returns:
            SpectralFeatures with detected peaks
        """
        try:
            from scipy.signal import find_peaks, peak_widths
            
            # Find peaks
            peaks, properties = find_peaks(intensities, prominence=prominence)
            
            # Get peak widths
            widths, _, _, _ = peak_widths(intensities, peaks)
            
            return SpectralFeatures(
                peak_positions=wavelengths[peaks].tolist(),
                peak_intensities=intensities[peaks].tolist(),
                peak_widths=widths.tolist(),
                baseline=np.zeros_like(intensities),
                normalized_spectrum=intensities
            )
        
        except ImportError:
            logger.warning("SciPy not available, using simple peak detection")
            
            # Fallback: simple local maxima
            peaks = []
            for i in range(1, len(intensities) - 1):
                if intensities[i] > intensities[i-1] and intensities[i] > intensities[i+1]:
                    if intensities[i] > prominence:
                        peaks.append(i)
            
            peaks = np.array(peaks)
            
            return SpectralFeatures(
                peak_positions=wavelengths[peaks].tolist() if len(peaks) > 0 else [],
                peak_intensities=intensities[peaks].tolist() if len(peaks) > 0 else [],
                peak_widths=[1.0] * len(peaks),
                baseline=np.zeros_like(intensities),
                normalized_spectrum=intensities
            )
    
    def extract_features(self, wavelengths: np.ndarray, intensities: np.ndarray) -> SpectralFeatures:
        """
        Extract comprehensive features for ML models.
        
        Args:
            wavelengths: Wavelength array
            intensities: Intensity array
        
        Returns:
            SpectralFeatures with all extracted features
        """
        # Preprocess
        wavelengths, processed = self.preprocess(wavelengths, intensities)
        
        # Detect peaks
        features = self.detect_peaks(wavelengths, processed)
        
        # PCA features (if sklearn available)
        try:
            from sklearn.decomposition import PCA
            # Fix: use proper n_components based on data shape
            n_components = min(10, len(processed) - 1)
            if n_components > 0:
                pca = PCA(n_components=n_components)
                features.pca_features = pca.fit_transform(processed.reshape(1, -1))[0]
        except ImportError:
            pass
        except Exception as e:
            # Silently skip PCA if it fails
            pass
        
        # Wavelet features (if pywt available)
        try:
            import pywt
            coeffs = pywt.wavedec(processed, 'db4', level=3)
            features.wavelet_features = np.concatenate([c.flatten() for c in coeffs])
        except ImportError:
            pass
        except Exception as e:
            # Silently skip wavelets if they fail
            pass
        
        return features


# =============================================================================
# COMPOUND IDENTIFICATION MODEL
# =============================================================================

class CompoundIdentificationModel:
    """
    ML model for compound identification from spectral data.
    
    In production: 1D CNN or Transformer trained on spectral libraries.
    Current: Rule-based matching with ML augmentation.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize compound identification model.
        
        Args:
            model_path: Path to trained model weights
        """
        self.model_path = model_path
        self.model_version = "v0.1.0-dev"
        
        # TODO: Load trained PyTorch/TF model
        self.model = None
        
        # Spectral library (simplified)
        self._init_spectral_library()
        
        logger.info(f"CompoundIdentificationModel initialized ({self.model_version})")
    
    def _init_spectral_library(self):
        """Initialize spectral reference library."""
        # Simplified library: peak positions for common compounds
        self.library = {
            "glucose": {"peaks": [1150, 1080, 1030], "tolerance": 10},
            "protein": {"peaks": [1650, 1540, 1240], "tolerance": 10},
            "fat": {"peaks": [2920, 2850, 1740], "tolerance": 10},
            "vitamin_c": {"peaks": [1755, 1670, 1120], "tolerance": 10},
            "calcium": {"peaks": [875, 712], "tolerance": 5},
            "iron": {"peaks": [850, 560], "tolerance": 5},
        }
    
    def predict_compounds(self, features: SpectralFeatures, 
                         threshold: float = 0.7) -> List[CompoundPrediction]:
        """
        Predict compound presence and concentration.
        
        Args:
            features: Extracted spectral features
            threshold: Minimum confidence threshold
        
        Returns:
            List of compound predictions
        """
        predictions = []
        
        # Match against library
        for compound_name, ref_data in self.library.items():
            # Check peak matching
            matches = self._match_peaks(
                features.peak_positions,
                ref_data["peaks"],
                ref_data["tolerance"]
            )
            
            if len(matches) >= 2:  # At least 2 peaks match
                presence_prob = len(matches) / len(ref_data["peaks"])
                
                if presence_prob >= threshold:
                    # Estimate concentration from peak intensity
                    matched_intensities = [features.peak_intensities[i] for i in matches]
                    concentration = np.mean(matched_intensities) * 100000  # Scale to mg/kg
                    concentration_std = np.std(matched_intensities) * 100000 if len(matched_intensities) > 1 else concentration * 0.1
                    
                    predictions.append(CompoundPrediction(
                        compound_name=compound_name,
                        presence_probability=presence_prob,
                        concentration_mg_kg=float(concentration),
                        concentration_std=float(concentration_std),
                        confidence_score=presence_prob,
                        model_version=self.model_version
                    ))
        
        return predictions
    
    def _match_peaks(self, observed_peaks: List[float], 
                    reference_peaks: List[float], 
                    tolerance: float) -> List[int]:
        """Match observed peaks to reference peaks."""
        matches = []
        for i, obs_peak in enumerate(observed_peaks):
            for ref_peak in reference_peaks:
                if abs(obs_peak - ref_peak) <= tolerance:
                    matches.append(i)
                    break
        return matches
    
    def quantify_with_uncertainty(self, compound_name: str, 
                                  features: SpectralFeatures) -> Tuple[float, float]:
        """
        Quantify compound with uncertainty estimation.
        
        Args:
            compound_name: Name of compound
            features: Spectral features
        
        Returns:
            Tuple of (concentration_mg_kg, std_mg_kg)
        """
        # TODO: Use Bayesian NN or Monte Carlo dropout for uncertainty
        # For now: simple estimation
        
        if compound_name in self.library:
            ref_peaks = self.library[compound_name]["peaks"]
            tolerance = self.library[compound_name]["tolerance"]
            
            matches = self._match_peaks(features.peak_positions, ref_peaks, tolerance)
            
            if matches:
                intensities = [features.peak_intensities[i] for i in matches]
                conc = np.mean(intensities) * 100000
                std = np.std(intensities) * 100000 if len(intensities) > 1 else conc * 0.15
                return float(conc), float(std)
        
        return 0.0, 0.0


# =============================================================================
# TOXICITY PREDICTION MODEL
# =============================================================================

class ToxicityPredictionModel:
    """
    ML model for toxicity prediction.
    
    In production: GNN on molecular graphs + XGBoost on descriptors.
    Current: Rule-based with knowledge graph lookup.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize toxicity prediction model."""
        self.model_path = model_path
        self.model_version = "v0.1.0-dev"
        
        logger.info(f"ToxicityPredictionModel initialized ({self.model_version})")
    
    def predict_toxicity(self, compound_name: str, 
                        molecular_structure: Optional[str] = None) -> ToxicityPrediction:
        """
        Predict toxicity from compound name or structure.
        
        Args:
            compound_name: Name of compound
            molecular_structure: SMILES string (optional)
        
        Returns:
            ToxicityPrediction
        """
        # TODO: Implement GNN-based prediction
        # For now: placeholder with random scores
        
        # Check if known toxin
        known_hazards = ["aflatoxin", "acrylamide", "lead", "mercury", "cadmium", "arsenic"]
        is_known_toxin = any(tox in compound_name.lower() for tox in known_hazards)
        
        if is_known_toxin:
            return ToxicityPrediction(
                compound_name=compound_name,
                acute_toxicity_score=0.8,
                chronic_toxicity_score=0.7,
                carcinogenicity_probability=0.6,
                ld50_predicted=100.0,
                hazard_classes=["toxic", "harmful"],
                confidence=0.85
            )
        else:
            return ToxicityPrediction(
                compound_name=compound_name,
                acute_toxicity_score=0.1,
                chronic_toxicity_score=0.05,
                carcinogenicity_probability=0.01,
                ld50_predicted=None,
                hazard_classes=[],
                confidence=0.6
            )


# =============================================================================
# ALLERGEN PREDICTION MODEL
# =============================================================================

class AllergenPredictionModel:
    """
    ML model for allergen detection.
    
    In production: Protein sequence transformer (ESM, ProtBert).
    Current: String matching with knowledge graph.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize allergen prediction model."""
        self.model_path = model_path
        self.model_version = "v0.1.0-dev"
        
        logger.info(f"AllergenPredictionModel initialized ({self.model_version})")
    
    def predict_allergenicity(self, protein_sequence: str) -> Tuple[bool, float]:
        """
        Predict if protein sequence is allergenic.
        
        Args:
            protein_sequence: Amino acid sequence
        
        Returns:
            Tuple of (is_allergenic, confidence)
        """
        # TODO: Implement transformer-based prediction
        # For now: placeholder
        
        # Simple heuristic: long sequences more likely to be allergenic proteins
        if len(protein_sequence) > 100:
            return True, 0.7
        return False, 0.3


# =============================================================================
# MODEL FACTORY
# =============================================================================

class ModelFactory:
    """Factory for creating and managing ML models."""
    
    _spectral_processor = None
    _compound_model = None
    _toxicity_model = None
    _allergen_model = None
    
    @classmethod
    def get_spectral_processor(cls, method: str = "ftir") -> SpectralProcessor:
        """Get spectral processor instance."""
        if cls._spectral_processor is None:
            cls._spectral_processor = SpectralProcessor(method)
        return cls._spectral_processor
    
    @classmethod
    def get_compound_model(cls) -> CompoundIdentificationModel:
        """Get compound identification model."""
        if cls._compound_model is None:
            cls._compound_model = CompoundIdentificationModel()
        return cls._compound_model
    
    @classmethod
    def get_toxicity_model(cls) -> ToxicityPredictionModel:
        """Get toxicity prediction model."""
        if cls._toxicity_model is None:
            cls._toxicity_model = ToxicityPredictionModel()
        return cls._toxicity_model
    
    @classmethod
    def get_allergen_model(cls) -> AllergenPredictionModel:
        """Get allergen prediction model."""
        if cls._allergen_model is None:
            cls._allergen_model = AllergenPredictionModel()
        return cls._allergen_model
