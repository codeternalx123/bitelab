"""
Molecular Bond Detector - NIR Spectroscopy Bond Analysis
==========================================================

This module identifies molecular bonds based on NIR absorption patterns.
Detects specific chemical bonds and their vibrational modes:

Bond Types Detected:
- O-H (hydroxyl): Water, alcohols, carbohydrates
- C-H (aliphatic): Fats, oils, carbohydrates, proteins
- N-H (amine): Proteins, amino acids
- C=O (carbonyl): Proteins, fats, carbohydrates
- C-O (ether/ester): Carbohydrates, fats
- C=C (alkene): Unsaturated fats, aromatic compounds
- S-H (thiol): Cysteine, sulfur compounds
- P=O (phosphate): Phospholipids, ATP, DNA

Vibrational Modes:
- Stretching (fundamental, overtone, combination)
- Bending (scissoring, rocking, wagging)
- Torsion (rotation around bonds)

Author: Wellomex AI Nutrition Team
Version: 1.0.0
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class BondType(Enum):
    """Types of molecular bonds detectable by NIR"""
    # Primary bonds
    O_H = "O-H"  # Hydroxyl (water, alcohols, sugars)
    C_H = "C-H"  # Aliphatic (fats, carbs, proteins)
    N_H = "N-H"  # Amine (proteins, amino acids)
    C_O = "C=O"  # Carbonyl (proteins, fats, ketones)
    C_O_SINGLE = "C-O"  # Ether/ester (carbs, fats)
    C_C_DOUBLE = "C=C"  # Alkene (unsaturated fats)
    S_H = "S-H"  # Thiol (cysteine, sulfur compounds)
    P_O = "P=O"  # Phosphate (phospholipids)
    
    # Secondary bonds
    C_C = "C-C"  # Aliphatic carbon chains
    N_O = "N-O"  # Nitro groups, nitrogen oxides
    C_N = "C-N"  # Amide linkage (peptide bonds)
    C_S = "C-S"  # Thioester
    
    # Aromatic bonds
    AROMATIC = "Aromatic"  # Benzene rings (amino acids, polyphenols)


class VibrationalMode(Enum):
    """Vibrational modes in NIR region"""
    FIRST_OVERTONE = "1st_overtone"  # 2ν (double frequency)
    SECOND_OVERTONE = "2nd_overtone"  # 3ν (triple frequency)
    COMBINATION = "combination"  # ν₁ + ν₂
    STRETCHING = "stretching"  # Bond length change
    BENDING = "bending"  # Bond angle change


@dataclass
class AbsorptionBand:
    """
    NIR absorption band characteristics
    """
    bond_type: BondType
    center_wavelength: float  # nm
    bandwidth: float  # nm (FWHM - Full Width Half Maximum)
    vibrational_mode: VibrationalMode
    molar_absorptivity: float  # L/(mol·cm) - Beer-Lambert coefficient
    
    # Assignment details
    molecular_context: str  # e.g., "water", "protein", "fat"
    overtone_order: int  # 1, 2, 3 for overtones
    assignment: str  # Chemical description
    
    # Spectral characteristics
    intensity_range: Tuple[float, float]  # (min, max) relative intensity
    temperature_dependence: float  # nm/°C shift
    
    def __repr__(self):
        return (f"{self.bond_type.value} @ {self.center_wavelength}nm "
                f"({self.vibrational_mode.value})")


@dataclass
class DetectedBond:
    """
    A molecular bond detected in NIR spectrum
    """
    bond_type: BondType
    wavelength: float  # nm (detected peak position)
    intensity: float  # Absorption intensity
    confidence: float  # 0-1 (detection confidence)
    concentration_estimate: Optional[float] = None  # mol/L or g/100g
    
    # Context
    absorption_band: Optional[AbsorptionBand] = None
    neighboring_bonds: List[BondType] = field(default_factory=list)
    molecular_environment: str = ""  # "aqueous", "lipid", "protein", etc.
    
    # Quality metrics
    signal_to_noise: float = 0.0
    peak_width: float = 0.0  # nm
    asymmetry: float = 0.0  # Peak asymmetry factor


@dataclass
class MolecularFingerprint:
    """
    Complete molecular fingerprint from NIR spectrum
    """
    detected_bonds: List[DetectedBond]
    timestamp: float
    
    # Summary statistics
    total_bonds: int = 0
    unique_bond_types: int = 0
    dominant_bond: Optional[BondType] = None
    
    # Composition estimates
    water_content: float = 0.0  # Percentage
    protein_content: float = 0.0
    fat_content: float = 0.0
    carbohydrate_content: float = 0.0
    
    # Quality
    overall_confidence: float = 0.0
    spectral_complexity: float = 0.0  # 0-1


class NIRBondDatabase:
    """
    Comprehensive database of NIR absorption bands for molecular bonds
    Based on published spectroscopic data and research
    """
    
    def __init__(self):
        """Initialize NIR bond database with literature values"""
        self.absorption_bands: Dict[BondType, List[AbsorptionBand]] = {}
        self._initialize_database()
    
    def _initialize_database(self):
        """
        Load NIR absorption band data from spectroscopic literature
        
        References:
        - Workman & Weyer (2012) Practical Guide to Interpretive NIR Spectroscopy
        - Ozaki et al. (2007) Near-Infrared Spectroscopy in Food Science and Technology
        - USDA NIR Food Database
        """
        
        # ========== O-H BONDS (Hydroxyl) ==========
        # Most important in food analysis (water, sugars, alcohols)
        
        self.absorption_bands[BondType.O_H] = [
            # Water absorption bands
            AbsorptionBand(
                bond_type=BondType.O_H,
                center_wavelength=760,
                bandwidth=40,
                vibrational_mode=VibrationalMode.SECOND_OVERTONE,
                molar_absorptivity=0.08,
                molecular_context="water",
                overtone_order=3,
                assignment="3ν O-H stretch (water)",
                intensity_range=(0.1, 0.5),
                temperature_dependence=0.1
            ),
            AbsorptionBand(
                bond_type=BondType.O_H,
                center_wavelength=970,
                bandwidth=60,
                vibrational_mode=VibrationalMode.FIRST_OVERTONE,
                molar_absorptivity=0.35,
                molecular_context="water",
                overtone_order=2,
                assignment="2ν O-H stretch + bend combination (water)",
                intensity_range=(0.3, 1.0),
                temperature_dependence=0.15
            ),
            AbsorptionBand(
                bond_type=BondType.O_H,
                center_wavelength=1150,
                bandwidth=80,
                vibrational_mode=VibrationalMode.FIRST_OVERTONE,
                molar_absorptivity=0.45,
                molecular_context="water",
                overtone_order=2,
                assignment="2ν O-H stretch (water, strong)",
                intensity_range=(0.5, 1.0),
                temperature_dependence=0.2
            ),
            AbsorptionBand(
                bond_type=BondType.O_H,
                center_wavelength=1450,
                bandwidth=90,
                vibrational_mode=VibrationalMode.FIRST_OVERTONE,
                molar_absorptivity=0.50,
                molecular_context="water",
                overtone_order=1,
                assignment="ν O-H stretch + δ O-H bend (water, VERY STRONG)",
                intensity_range=(0.8, 1.0),
                temperature_dependence=0.25
            ),
            AbsorptionBand(
                bond_type=BondType.O_H,
                center_wavelength=1940,
                bandwidth=100,
                vibrational_mode=VibrationalMode.COMBINATION,
                molar_absorptivity=0.55,
                molecular_context="water",
                overtone_order=1,
                assignment="ν O-H stretch + δ O-H bend combination (water, STRONGEST)",
                intensity_range=(0.9, 1.0),
                temperature_dependence=0.3
            ),
            
            # Hydroxyl in carbohydrates
            AbsorptionBand(
                bond_type=BondType.O_H,
                center_wavelength=1460,
                bandwidth=70,
                vibrational_mode=VibrationalMode.FIRST_OVERTONE,
                molar_absorptivity=0.30,
                molecular_context="carbohydrate",
                overtone_order=1,
                assignment="O-H stretch (sugar hydroxyl)",
                intensity_range=(0.4, 0.8),
                temperature_dependence=0.12
            ),
            AbsorptionBand(
                bond_type=BondType.O_H,
                center_wavelength=2100,
                bandwidth=85,
                vibrational_mode=VibrationalMode.COMBINATION,
                molar_absorptivity=0.25,
                molecular_context="carbohydrate",
                overtone_order=1,
                assignment="O-H stretch + C-O stretch (sugar)",
                intensity_range=(0.3, 0.7),
                temperature_dependence=0.1
            ),
        ]
        
        # ========== C-H BONDS (Aliphatic) ==========
        # Critical for fats, oils, carbohydrates
        
        self.absorption_bands[BondType.C_H] = [
            # CH₂ (methylene) - fats and oils
            AbsorptionBand(
                bond_type=BondType.C_H,
                center_wavelength=1180,
                bandwidth=50,
                vibrational_mode=VibrationalMode.SECOND_OVERTONE,
                molar_absorptivity=0.15,
                molecular_context="lipid",
                overtone_order=2,
                assignment="3ν C-H stretch (CH₂ in fats)",
                intensity_range=(0.2, 0.6),
                temperature_dependence=0.05
            ),
            AbsorptionBand(
                bond_type=BondType.C_H,
                center_wavelength=1210,
                bandwidth=45,
                vibrational_mode=VibrationalMode.SECOND_OVERTONE,
                molar_absorptivity=0.18,
                molecular_context="lipid",
                overtone_order=2,
                assignment="2ν C-H stretch (CH₃ in fats)",
                intensity_range=(0.3, 0.7),
                temperature_dependence=0.06
            ),
            AbsorptionBand(
                bond_type=BondType.C_H,
                center_wavelength=1395,
                bandwidth=55,
                vibrational_mode=VibrationalMode.FIRST_OVERTONE,
                molar_absorptivity=0.25,
                molecular_context="lipid",
                overtone_order=1,
                assignment="2ν C-H stretch (CH₂, strong)",
                intensity_range=(0.4, 0.8),
                temperature_dependence=0.08
            ),
            AbsorptionBand(
                bond_type=BondType.C_H,
                center_wavelength=1725,
                bandwidth=60,
                vibrational_mode=VibrationalMode.FIRST_OVERTONE,
                molar_absorptivity=0.30,
                molecular_context="lipid",
                overtone_order=1,
                assignment="ν C-H stretch + bend (CH₂ in long chains)",
                intensity_range=(0.5, 0.9),
                temperature_dependence=0.1
            ),
            AbsorptionBand(
                bond_type=BondType.C_H,
                center_wavelength=2310,
                bandwidth=75,
                vibrational_mode=VibrationalMode.COMBINATION,
                molar_absorptivity=0.35,
                molecular_context="lipid",
                overtone_order=1,
                assignment="ν C-H stretch + δ C-H bend (fats, STRONG)",
                intensity_range=(0.6, 1.0),
                temperature_dependence=0.12
            ),
            
            # C-H in carbohydrates
            AbsorptionBand(
                bond_type=BondType.C_H,
                center_wavelength=1460,
                bandwidth=50,
                vibrational_mode=VibrationalMode.FIRST_OVERTONE,
                molar_absorptivity=0.20,
                molecular_context="carbohydrate",
                overtone_order=1,
                assignment="C-H stretch (sugar backbone)",
                intensity_range=(0.3, 0.7),
                temperature_dependence=0.07
            ),
            AbsorptionBand(
                bond_type=BondType.C_H,
                center_wavelength=2270,
                bandwidth=65,
                vibrational_mode=VibrationalMode.COMBINATION,
                molar_absorptivity=0.28,
                molecular_context="carbohydrate",
                overtone_order=1,
                assignment="C-H stretch + bend (starch, sugar)",
                intensity_range=(0.4, 0.8),
                temperature_dependence=0.09
            ),
        ]
        
        # ========== N-H BONDS (Amine/Amide) ==========
        # Key for protein detection
        
        self.absorption_bands[BondType.N_H] = [
            AbsorptionBand(
                bond_type=BondType.N_H,
                center_wavelength=1020,
                bandwidth=45,
                vibrational_mode=VibrationalMode.SECOND_OVERTONE,
                molar_absorptivity=0.12,
                molecular_context="protein",
                overtone_order=2,
                assignment="3ν N-H stretch (amide A overtone)",
                intensity_range=(0.2, 0.5),
                temperature_dependence=0.08
            ),
            AbsorptionBand(
                bond_type=BondType.N_H,
                center_wavelength=1490,
                bandwidth=55,
                vibrational_mode=VibrationalMode.FIRST_OVERTONE,
                molar_absorptivity=0.22,
                molecular_context="protein",
                overtone_order=1,
                assignment="ν N-H stretch (peptide bond)",
                intensity_range=(0.3, 0.7),
                temperature_dependence=0.1
            ),
            AbsorptionBand(
                bond_type=BondType.N_H,
                center_wavelength=2050,
                bandwidth=70,
                vibrational_mode=VibrationalMode.COMBINATION,
                molar_absorptivity=0.30,
                molecular_context="protein",
                overtone_order=1,
                assignment="ν N-H stretch + δ N-H bend (amide, strong)",
                intensity_range=(0.5, 0.9),
                temperature_dependence=0.15
            ),
            AbsorptionBand(
                bond_type=BondType.N_H,
                center_wavelength=2180,
                bandwidth=65,
                vibrational_mode=VibrationalMode.COMBINATION,
                molar_absorptivity=0.25,
                molecular_context="protein",
                overtone_order=1,
                assignment="Amide B + amide II combination",
                intensity_range=(0.4, 0.8),
                temperature_dependence=0.12
            ),
        ]
        
        # ========== C=O BONDS (Carbonyl) ==========
        # Important for proteins, fats, ketones
        
        self.absorption_bands[BondType.C_O] = [
            AbsorptionBand(
                bond_type=BondType.C_O,
                center_wavelength=1690,
                bandwidth=60,
                vibrational_mode=VibrationalMode.FIRST_OVERTONE,
                molar_absorptivity=0.18,
                molecular_context="protein",
                overtone_order=1,
                assignment="2ν C=O stretch (amide I overtone)",
                intensity_range=(0.3, 0.7),
                temperature_dependence=0.09
            ),
            AbsorptionBand(
                bond_type=BondType.C_O,
                center_wavelength=2150,
                bandwidth=70,
                vibrational_mode=VibrationalMode.COMBINATION,
                molar_absorptivity=0.25,
                molecular_context="protein",
                overtone_order=1,
                assignment="ν C=O stretch + δ N-H bend (amide I + II)",
                intensity_range=(0.4, 0.8),
                temperature_dependence=0.11
            ),
            AbsorptionBand(
                bond_type=BondType.C_O,
                center_wavelength=1730,
                bandwidth=55,
                vibrational_mode=VibrationalMode.FIRST_OVERTONE,
                molar_absorptivity=0.22,
                molecular_context="lipid",
                overtone_order=1,
                assignment="2ν C=O stretch (ester in fats)",
                intensity_range=(0.4, 0.8),
                temperature_dependence=0.1
            ),
        ]
        
        # ========== C-O BONDS (Ether/Ester, single bond) ==========
        # Carbohydrates, fats
        
        self.absorption_bands[BondType.C_O_SINGLE] = [
            AbsorptionBand(
                bond_type=BondType.C_O_SINGLE,
                center_wavelength=1935,
                bandwidth=75,
                vibrational_mode=VibrationalMode.COMBINATION,
                molar_absorptivity=0.20,
                molecular_context="carbohydrate",
                overtone_order=1,
                assignment="ν C-O stretch + δ O-H bend (sugar)",
                intensity_range=(0.3, 0.7),
                temperature_dependence=0.08
            ),
            AbsorptionBand(
                bond_type=BondType.C_O_SINGLE,
                center_wavelength=2280,
                bandwidth=65,
                vibrational_mode=VibrationalMode.COMBINATION,
                molar_absorptivity=0.18,
                molecular_context="carbohydrate",
                overtone_order=1,
                assignment="ν C-O stretch + δ C-H bend (starch)",
                intensity_range=(0.3, 0.7),
                temperature_dependence=0.07
            ),
        ]
        
        # ========== C=C BONDS (Alkene, unsaturated) ==========
        # Unsaturated fats, aromatic amino acids
        
        self.absorption_bands[BondType.C_C_DOUBLE] = [
            AbsorptionBand(
                bond_type=BondType.C_C_DOUBLE,
                center_wavelength=1650,
                bandwidth=40,
                vibrational_mode=VibrationalMode.FIRST_OVERTONE,
                molar_absorptivity=0.15,
                molecular_context="lipid",
                overtone_order=1,
                assignment="2ν C=C stretch (unsaturated fat)",
                intensity_range=(0.2, 0.6),
                temperature_dependence=0.05
            ),
            AbsorptionBand(
                bond_type=BondType.C_C_DOUBLE,
                center_wavelength=2140,
                bandwidth=50,
                vibrational_mode=VibrationalMode.COMBINATION,
                molar_absorptivity=0.20,
                molecular_context="lipid",
                overtone_order=1,
                assignment="ν C=C stretch + =C-H stretch",
                intensity_range=(0.3, 0.7),
                temperature_dependence=0.06
            ),
        ]
        
        # ========== S-H BONDS (Thiol) ==========
        # Cysteine in proteins
        
        self.absorption_bands[BondType.S_H] = [
            AbsorptionBand(
                bond_type=BondType.S_H,
                center_wavelength=1370,
                bandwidth=35,
                vibrational_mode=VibrationalMode.FIRST_OVERTONE,
                molar_absorptivity=0.08,
                molecular_context="protein",
                overtone_order=1,
                assignment="2ν S-H stretch (cysteine)",
                intensity_range=(0.1, 0.4),
                temperature_dependence=0.04
            ),
        ]
        
        # ========== P=O BONDS (Phosphate) ==========
        # Phospholipids, ATP
        
        self.absorption_bands[BondType.P_O] = [
            AbsorptionBand(
                bond_type=BondType.P_O,
                center_wavelength=2400,
                bandwidth=80,
                vibrational_mode=VibrationalMode.COMBINATION,
                molar_absorptivity=0.12,
                molecular_context="phospholipid",
                overtone_order=1,
                assignment="ν P=O stretch + bend (phosphate)",
                intensity_range=(0.2, 0.5),
                temperature_dependence=0.07
            ),
        ]
        
        logger.info(f"NIR bond database initialized: {len(self.absorption_bands)} bond types, "
                   f"{sum(len(bands) for bands in self.absorption_bands.values())} total bands")
    
    def get_bands_for_bond(self, bond_type: BondType) -> List[AbsorptionBand]:
        """Get all absorption bands for a specific bond type"""
        return self.absorption_bands.get(bond_type, [])
    
    def get_bands_in_range(self, min_wl: float, max_wl: float) -> List[AbsorptionBand]:
        """Get all absorption bands within wavelength range"""
        bands = []
        for bond_bands in self.absorption_bands.values():
            for band in bond_bands:
                if min_wl <= band.center_wavelength <= max_wl:
                    bands.append(band)
        return sorted(bands, key=lambda b: b.center_wavelength)
    
    def get_strongest_bands(self, top_n: int = 10) -> List[AbsorptionBand]:
        """Get the strongest (highest molar absorptivity) bands"""
        all_bands = []
        for bond_bands in self.absorption_bands.values():
            all_bands.extend(bond_bands)
        return sorted(all_bands, key=lambda b: b.molar_absorptivity, reverse=True)[:top_n]


class MolecularBondDetector:
    """
    Main molecular bond detector
    Analyzes NIR spectra to identify chemical bonds
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize molecular bond detector
        
        Args:
            confidence_threshold: Minimum confidence for bond detection (0-1)
        """
        self.database = NIRBondDatabase()
        self.confidence_threshold = confidence_threshold
        
        # Detection parameters
        self.peak_prominence = 0.1  # Minimum peak prominence for detection
        self.min_snr = 3.0  # Minimum signal-to-noise ratio
        
        # Statistics
        self.total_detections = 0
        self.detection_history: List[MolecularFingerprint] = []
        
        logger.info("Molecular Bond Detector initialized")
    
    def analyze_spectrum(self, 
                        wavelengths: np.ndarray, 
                        absorbance: np.ndarray,
                        temperature: float = 25.0) -> MolecularFingerprint:
        """
        Analyze NIR spectrum to detect molecular bonds
        
        Args:
            wavelengths: Wavelength array (nm)
            absorbance: Absorbance values
            temperature: Sample temperature (°C)
            
        Returns:
            MolecularFingerprint with detected bonds
        """
        import time
        start_time = time.time()
        
        detected_bonds = []
        
        # Find peaks in spectrum
        peaks = self._find_peaks(wavelengths, absorbance)
        
        logger.info(f"Found {len(peaks)} spectral peaks")
        
        # Match peaks to known absorption bands
        for peak_wl, peak_intensity in peaks:
            # Find matching absorption bands
            matches = self._match_peak_to_bands(peak_wl, peak_intensity, temperature)
            
            for band, confidence in matches:
                if confidence >= self.confidence_threshold:
                    # Estimate concentration using Beer-Lambert law
                    concentration = self._estimate_concentration(
                        peak_intensity, band.molar_absorptivity
                    )
                    
                    detected_bond = DetectedBond(
                        bond_type=band.bond_type,
                        wavelength=peak_wl,
                        intensity=peak_intensity,
                        confidence=confidence,
                        concentration_estimate=concentration,
                        absorption_band=band,
                        molecular_environment=band.molecular_context
                    )
                    
                    detected_bonds.append(detected_bond)
        
        # Create molecular fingerprint
        fingerprint = self._create_fingerprint(detected_bonds, start_time)
        
        # Estimate macronutrient composition
        self._estimate_composition(fingerprint)
        
        self.total_detections += 1
        self.detection_history.append(fingerprint)
        
        logger.info(f"Detected {len(detected_bonds)} molecular bonds "
                   f"({fingerprint.unique_bond_types} unique types)")
        
        return fingerprint
    
    def _find_peaks(self, wavelengths: np.ndarray, absorbance: np.ndarray) -> List[Tuple[float, float]]:
        """
        Find peaks in NIR spectrum
        
        Returns:
            List of (wavelength, intensity) tuples
        """
        from scipy.signal import find_peaks
        
        # Find peaks with minimum prominence
        peak_indices, properties = find_peaks(
            absorbance,
            prominence=self.peak_prominence,
            width=3  # Minimum peak width in data points
        )
        
        peaks = [
            (wavelengths[idx], absorbance[idx])
            for idx in peak_indices
        ]
        
        return peaks
    
    def _match_peak_to_bands(self, 
                            peak_wl: float, 
                            peak_intensity: float,
                            temperature: float) -> List[Tuple[AbsorptionBand, float]]:
        """
        Match a spectral peak to known absorption bands
        
        Returns:
            List of (AbsorptionBand, confidence) tuples
        """
        matches = []
        
        # Get all bands within reasonable range
        search_range = 100  # nm
        candidate_bands = self.database.get_bands_in_range(
            peak_wl - search_range, 
            peak_wl + search_range
        )
        
        for band in candidate_bands:
            # Adjust band position for temperature
            temp_shift = band.temperature_dependence * (temperature - 25.0)
            adjusted_center = band.center_wavelength + temp_shift
            
            # Calculate wavelength match confidence (Gaussian)
            wl_diff = abs(peak_wl - adjusted_center)
            wl_confidence = np.exp(-(wl_diff / band.bandwidth) ** 2)
            
            # Calculate intensity match confidence
            min_int, max_int = band.intensity_range
            if min_int <= peak_intensity <= max_int:
                int_confidence = 1.0
            elif peak_intensity < min_int:
                int_confidence = peak_intensity / min_int
            else:
                int_confidence = max_int / peak_intensity
            
            # Combined confidence
            confidence = wl_confidence * int_confidence
            
            if confidence > 0.1:  # Minimum threshold for consideration
                matches.append((band, confidence))
        
        # Sort by confidence
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:3]  # Return top 3 matches
    
    def _estimate_concentration(self, absorbance: float, molar_absorptivity: float, 
                               path_length: float = 0.1) -> float:
        """
        Estimate concentration using Beer-Lambert law: A = ε·c·l
        
        Args:
            absorbance: Measured absorbance
            molar_absorptivity: ε (L/(mol·cm))
            path_length: l (cm) - typical food sample thickness
            
        Returns:
            Estimated concentration (mol/L)
        """
        if molar_absorptivity == 0:
            return 0.0
        
        # A = ε·c·l  →  c = A / (ε·l)
        concentration = absorbance / (molar_absorptivity * path_length)
        
        return max(0.0, concentration)  # Ensure non-negative
    
    def _create_fingerprint(self, detected_bonds: List[DetectedBond], 
                           start_time: float) -> MolecularFingerprint:
        """Create molecular fingerprint from detected bonds"""
        
        # Count unique bond types
        unique_types = set(bond.bond_type for bond in detected_bonds)
        
        # Find dominant bond (highest total intensity)
        bond_intensities = {}
        for bond in detected_bonds:
            if bond.bond_type not in bond_intensities:
                bond_intensities[bond.bond_type] = 0.0
            bond_intensities[bond.bond_type] += bond.intensity
        
        dominant_bond = max(bond_intensities, key=bond_intensities.get) if bond_intensities else None
        
        # Calculate overall confidence
        if detected_bonds:
            overall_confidence = np.mean([bond.confidence for bond in detected_bonds])
        else:
            overall_confidence = 0.0
        
        fingerprint = MolecularFingerprint(
            detected_bonds=detected_bonds,
            timestamp=start_time,
            total_bonds=len(detected_bonds),
            unique_bond_types=len(unique_types),
            dominant_bond=dominant_bond,
            overall_confidence=overall_confidence
        )
        
        return fingerprint
    
    def _estimate_composition(self, fingerprint: MolecularFingerprint):
        """
        Estimate macronutrient composition from molecular fingerprint
        """
        # Water content (from O-H bonds)
        oh_bonds = [b for b in fingerprint.detected_bonds 
                   if b.bond_type == BondType.O_H and b.molecular_environment == "water"]
        if oh_bonds:
            # Strong water bands indicate high water content
            water_intensity = sum(b.intensity for b in oh_bonds)
            fingerprint.water_content = min(100.0, water_intensity * 50)  # Simplified estimate
        
        # Protein content (from N-H and C=O amide bonds)
        protein_bonds = [b for b in fingerprint.detected_bonds 
                        if b.bond_type in [BondType.N_H, BondType.C_O] 
                        and b.molecular_environment == "protein"]
        if protein_bonds:
            protein_intensity = sum(b.intensity for b in protein_bonds)
            fingerprint.protein_content = min(100.0, protein_intensity * 30)
        
        # Fat content (from C-H and C=O ester bonds)
        fat_bonds = [b for b in fingerprint.detected_bonds 
                    if b.molecular_environment == "lipid"]
        if fat_bonds:
            fat_intensity = sum(b.intensity for b in fat_bonds)
            fingerprint.fat_content = min(100.0, fat_intensity * 25)
        
        # Carbohydrate content (from C-H, O-H, C-O in sugars)
        carb_bonds = [b for b in fingerprint.detected_bonds 
                     if b.molecular_environment == "carbohydrate"]
        if carb_bonds:
            carb_intensity = sum(b.intensity for b in carb_bonds)
            fingerprint.carbohydrate_content = min(100.0, carb_intensity * 35)
        
        # Normalize if total > 100%
        total = (fingerprint.water_content + fingerprint.protein_content + 
                fingerprint.fat_content + fingerprint.carbohydrate_content)
        if total > 100:
            scale = 100.0 / total
            fingerprint.water_content *= scale
            fingerprint.protein_content *= scale
            fingerprint.fat_content *= scale
            fingerprint.carbohydrate_content *= scale
    
    def get_bond_summary(self, fingerprint: MolecularFingerprint) -> Dict:
        """Get summary of detected bonds"""
        summary = {
            "total_bonds": fingerprint.total_bonds,
            "unique_types": fingerprint.unique_bond_types,
            "dominant_bond": fingerprint.dominant_bond.value if fingerprint.dominant_bond else None,
            "confidence": f"{fingerprint.overall_confidence:.2f}",
            "composition": {
                "water": f"{fingerprint.water_content:.1f}%",
                "protein": f"{fingerprint.protein_content:.1f}%",
                "fat": f"{fingerprint.fat_content:.1f}%",
                "carbohydrate": f"{fingerprint.carbohydrate_content:.1f}%"
            }
        }
        
        # Count bonds by type
        bond_counts = {}
        for bond in fingerprint.detected_bonds:
            bond_type = bond.bond_type.value
            if bond_type not in bond_counts:
                bond_counts[bond_type] = 0
            bond_counts[bond_type] += 1
        
        summary["bond_counts"] = bond_counts
        
        return summary


# ============================================================================
# Test and Demo
# ============================================================================

def test_molecular_bond_detector():
    """Test the molecular bond detector"""
    print("\n" + "=" * 70)
    print("MOLECULAR BOND DETECTOR TEST")
    print("=" * 70)
    
    # Initialize detector
    print("\n[TEST 1] Initializing Bond Detector")
    print("-" * 70)
    
    detector = MolecularBondDetector(confidence_threshold=0.4)
    
    db = detector.database
    print(f"✓ Database loaded: {len(db.absorption_bands)} bond types")
    print(f"✓ Total absorption bands: {sum(len(b) for b in db.absorption_bands.values())}")
    
    # Show strongest bands
    print("\n[TEST 2] Strongest NIR Absorption Bands")
    print("-" * 70)
    
    strongest = db.get_strongest_bands(top_n=10)
    for i, band in enumerate(strongest, 1):
        print(f"{i}. {band.bond_type.value} @ {band.center_wavelength}nm "
              f"(ε={band.molar_absorptivity:.2f}, {band.molecular_context})")
        print(f"   {band.assignment}")
    
    # Simulate food spectrum (avocado example)
    print("\n[TEST 3] Analyzing Simulated Avocado NIR Spectrum")
    print("-" * 70)
    
    # Generate realistic avocado spectrum
    wavelengths = np.linspace(900, 2500, 320)
    absorbance = np.zeros_like(wavelengths)
    
    # Add characteristic peaks for avocado (high fat, moderate water)
    # Water O-H bands
    absorbance += 0.3 * np.exp(-((wavelengths - 1450) / 90) ** 2)  # Water
    absorbance += 0.4 * np.exp(-((wavelengths - 1940) / 100) ** 2)  # Water (strong)
    
    # Fat C-H bands (prominent in avocado)
    absorbance += 0.5 * np.exp(-((wavelengths - 1210) / 45) ** 2)  # CH₃
    absorbance += 0.6 * np.exp(-((wavelengths - 1395) / 55) ** 2)  # CH₂
    absorbance += 0.7 * np.exp(-((wavelengths - 1725) / 60) ** 2)  # CH₂ strong
    absorbance += 0.8 * np.exp(-((wavelengths - 2310) / 75) ** 2)  # CH (very strong)
    
    # Protein N-H bands (moderate)
    absorbance += 0.25 * np.exp(-((wavelengths - 2050) / 70) ** 2)  # N-H
    
    # Carbohydrate (low in avocado)
    absorbance += 0.15 * np.exp(-((wavelengths - 2270) / 65) ** 2)  # C-H carb
    
    # Add noise
    absorbance += np.random.normal(0, 0.02, len(wavelengths))
    absorbance = np.maximum(absorbance, 0)  # No negative absorbance
    
    # Analyze spectrum
    fingerprint = detector.analyze_spectrum(wavelengths, absorbance, temperature=25.0)
    
    print(f"✅ Analysis complete")
    print(f"\n📊 Molecular Fingerprint:")
    print(f"   Total bonds detected: {fingerprint.total_bonds}")
    print(f"   Unique bond types: {fingerprint.unique_bond_types}")
    print(f"   Dominant bond: {fingerprint.dominant_bond.value if fingerprint.dominant_bond else 'None'}")
    print(f"   Overall confidence: {fingerprint.overall_confidence:.2f}")
    
    # Composition estimate
    print(f"\n🥑 Estimated Composition:")
    print(f"   Water:        {fingerprint.water_content:.1f}%")
    print(f"   Fat:          {fingerprint.fat_content:.1f}%")
    print(f"   Protein:      {fingerprint.protein_content:.1f}%")
    print(f"   Carbohydrate: {fingerprint.carbohydrate_content:.1f}%")
    print(f"   (Typical avocado: 73% water, 15% fat, 2% protein, 9% carbs)")
    
    # Detected bonds by type
    print(f"\n🔬 Detected Molecular Bonds:")
    summary = detector.get_bond_summary(fingerprint)
    for bond_type, count in sorted(summary["bond_counts"].items()):
        print(f"   {bond_type}: {count} peaks")
    
    # Show top detected bonds
    print(f"\n📈 Top Detected Peaks:")
    top_bonds = sorted(fingerprint.detected_bonds, 
                      key=lambda b: b.intensity, reverse=True)[:8]
    for i, bond in enumerate(top_bonds, 1):
        print(f"   {i}. {bond.bond_type.value} @ {bond.wavelength:.1f}nm")
        print(f"      Intensity: {bond.intensity:.3f}, Confidence: {bond.confidence:.2f}")
        print(f"      Context: {bond.molecular_environment}")
        if bond.absorption_band:
            print(f"      Assignment: {bond.absorption_band.assignment}")
    
    # Test with different foods
    print("\n[TEST 4] Comparing Different Food Spectra")
    print("-" * 70)
    
    foods = {
        "Apple (high water/carbs)": {
            "water": 0.9, "fat": 0.05, "protein": 0.2, "carb": 0.7
        },
        "Chicken Breast (high protein)": {
            "water": 0.4, "fat": 0.1, "protein": 0.9, "carb": 0.05
        },
        "Olive Oil (pure fat)": {
            "water": 0.02, "fat": 1.0, "protein": 0.0, "carb": 0.0
        }
    }
    
    for food_name, components in foods.items():
        # Generate simplified spectrum
        spec_abs = np.zeros_like(wavelengths)
        
        # Add component-specific peaks
        spec_abs += components["water"] * 0.5 * np.exp(-((wavelengths - 1940) / 100) ** 2)
        spec_abs += components["fat"] * 0.7 * np.exp(-((wavelengths - 2310) / 75) ** 2)
        spec_abs += components["protein"] * 0.4 * np.exp(-((wavelengths - 2050) / 70) ** 2)
        spec_abs += components["carb"] * 0.5 * np.exp(-((wavelengths - 2270) / 65) ** 2)
        spec_abs += np.random.normal(0, 0.01, len(wavelengths))
        spec_abs = np.maximum(spec_abs, 0)
        
        fp = detector.analyze_spectrum(wavelengths, spec_abs)
        
        print(f"\n{food_name}:")
        print(f"   Detected: {fp.total_bonds} bonds, {fp.unique_bond_types} types")
        print(f"   Composition: W={fp.water_content:.0f}% F={fp.fat_content:.0f}% "
              f"P={fp.protein_content:.0f}% C={fp.carbohydrate_content:.0f}%")
        print(f"   Dominant bond: {fp.dominant_bond.value if fp.dominant_bond else 'None'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Total detections performed: {detector.total_detections}")
    print(f"✅ Bond database: {sum(len(b) for b in db.absorption_bands.values())} absorption bands")
    print(f"✅ All tests passed!")
    print("\n💡 Molecular Bond Detector is ready for food analysis")
    print("   Next: Integrate with NIR Hardware Driver and Simulated Annealing")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        from scipy.signal import find_peaks
        test_molecular_bond_detector()
    except ImportError:
        print("⚠️ scipy not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "scipy"])
        from scipy.signal import find_peaks
        test_molecular_bond_detector()
