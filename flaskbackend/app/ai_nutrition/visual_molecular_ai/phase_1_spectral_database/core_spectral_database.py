"""
Phase 1: Spectral Database System - Core Spectral Signature Database
Part 1 of 10 phases building the Visual Molecular AI System (500k LOC total)

This module implements the foundational spectral signature database that links
visual properties (color, size, texture) to molecular composition verified by
lab equipment (ICP-MS, NMR, HPLC, GC-MS).

Architecture:
- Spectral Signature Storage: Store light absorption/reflection spectra
- Color-to-Molecule Mapping: Link HSV colors to specific molecules
- Lab Data Integration: Process ICP-MS, NMR, HPLC, GC-MS data
- Validation Pipeline: Ensure data quality and consistency
- Query Engine: Fast retrieval of spectral signatures

Target: 50,000 lines of code for Phase 1
This file: ~5,000 lines (Part 1: Core Database)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import pickle
from pathlib import Path
import sqlite3
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================
# Core Data Structures
# ================================

class MoleculeCategory(Enum):
    """Categories of molecules we can detect."""
    PIGMENT = "pigment"  # Color-producing molecules
    ANTIOXIDANT = "antioxidant"  # Health-promoting antioxidants
    VITAMIN = "vitamin"  # Essential vitamins
    MINERAL = "mineral"  # Trace elements and minerals
    CARBOHYDRATE = "carbohydrate"  # Sugars, starches, fiber
    PROTEIN = "protein"  # Amino acids and proteins
    LIPID = "lipid"  # Fats and oils
    TOXIN = "toxin"  # Harmful compounds
    PHYTONUTRIENT = "phytonutrient"  # Plant compounds
    ENZYME = "enzyme"  # Biological catalysts


class LabTechnique(Enum):
    """Laboratory analysis techniques."""
    ICP_MS = "icp_ms"  # Inductively Coupled Plasma Mass Spectrometry (atoms)
    NMR = "nmr"  # Nuclear Magnetic Resonance (molecular structure)
    HPLC = "hplc"  # High-Performance Liquid Chromatography (compounds)
    GC_MS = "gc_ms"  # Gas Chromatography Mass Spectrometry (volatiles)
    UV_VIS = "uv_vis"  # UV-Visible Spectroscopy (chromophores)
    FTIR = "ftir"  # Fourier Transform Infrared (functional groups)
    SPECTROPHOTOMETRY = "spectrophotometry"  # Color measurement


class ColorSpace(Enum):
    """Color representation spaces."""
    RGB = "rgb"  # Red, Green, Blue (0-255)
    HSV = "hsv"  # Hue, Saturation, Value (0-360, 0-100, 0-100)
    LAB = "lab"  # L*a*b* (perceptual color space)
    XYZ = "xyz"  # CIE XYZ (tristimulus values)
    SPECTRAL = "spectral"  # Full spectrum (380-780nm)


@dataclass
class SpectralSignature:
    """
    Complete spectral signature of a food sample.
    Links visual properties to molecular composition.
    """
    # Identification
    signature_id: str  # Unique identifier
    food_type: str  # e.g., "avocado", "carrot", "blueberry"
    variety: str  # e.g., "Hass", "Nantes", "Bluecrop"
    sample_date: datetime
    
    # Visual Properties (What the camera sees)
    color_rgb: Tuple[int, int, int]  # RGB values (0-255)
    color_hsv: Tuple[float, float, float]  # HSV (0-360, 0-100, 0-100)
    color_lab: Tuple[float, float, float]  # L*a*b* (perceptual)
    spectral_reflectance: Dict[int, float]  # Wavelength → reflectance (380-780nm)
    
    # Size & Physical Properties
    volume_cm3: float  # Volume in cubic centimeters
    mass_g: float  # Mass in grams
    density_g_cm3: float  # Density
    surface_area_cm2: float  # Surface area
    
    # Texture Properties
    firmness_n: Optional[float] = None  # Firmness in Newtons
    roughness_um: Optional[float] = None  # Surface roughness in micrometers
    glossiness: Optional[float] = None  # Glossiness (0-1)
    
    # Molecular Composition (What the lab measures)
    molecules: Dict[str, float] = field(default_factory=dict)  # Molecule → concentration
    atoms: Dict[str, float] = field(default_factory=dict)  # Element → ppm
    
    # Lab Analysis Metadata
    lab_technique: LabTechnique = LabTechnique.ICP_MS
    lab_verified: bool = False
    confidence_score: float = 0.0  # 0-1 confidence in measurements
    
    # Geographic & Environmental
    origin_location: Optional[str] = None
    soil_type: Optional[str] = None
    growing_conditions: Dict[str, any] = field(default_factory=dict)
    
    # Processing
    processing_method: Optional[str] = None  # "raw", "cooked", "frozen", etc.
    storage_duration_days: Optional[int] = None
    
    # Quality Metrics
    freshness_score: float = 1.0  # 0-1 (1 = fresh, 0 = spoiled)
    ripeness_score: float = 0.5  # 0-1 (0 = unripe, 1 = overripe)
    contamination_score: float = 0.0  # 0-1 (0 = clean, 1 = contaminated)
    
    # Notes
    notes: str = ""


@dataclass
class MolecularProfile:
    """
    Detailed molecular profile for a specific molecule.
    Links the molecule to its visual signature (chromophore).
    """
    # Identification
    molecule_id: str  # e.g., "beta_carotene"
    common_name: str  # e.g., "Beta-Carotene"
    chemical_formula: str  # e.g., "C40H56"
    cas_number: str  # Chemical Abstracts Service number
    
    # Category
    category: MoleculeCategory
    subcategory: Optional[str] = None
    
    # Quantum Properties (Why it has color)
    absorption_peaks_nm: List[int] = field(default_factory=list)  # Wavelengths absorbed
    emission_peaks_nm: List[int] = field(default_factory=list)  # Wavelengths emitted
    conjugated_bonds: Optional[int] = None  # Number of conjugated double bonds
    chromophore_type: Optional[str] = None  # e.g., "polyene", "anthocyanin"
    
    # Visual Signature (What color it produces)
    typical_color_rgb: Tuple[int, int, int] = (0, 0, 0)
    typical_color_hsv: Tuple[float, float, float] = (0, 0, 0)
    color_intensity_factor: float = 1.0  # How strongly it colors food
    
    # Concentration Ranges
    typical_concentration_mg_100g: Tuple[float, float] = (0.0, 0.0)  # (min, max)
    detection_limit_mg_100g: float = 0.001  # Minimum detectable
    saturation_point_mg_100g: float = 1000.0  # Maximum before color saturates
    
    # Health Properties
    health_benefits: List[str] = field(default_factory=list)
    target_organs: List[str] = field(default_factory=list)
    bioavailability: float = 1.0  # 0-1 (how much body absorbs)
    
    # Stability
    light_stable: bool = True
    heat_stable: bool = True
    pH_sensitive: bool = False
    oxidation_prone: bool = False
    
    # Associated Foods
    common_foods: List[str] = field(default_factory=list)
    
    # Research
    research_papers: List[str] = field(default_factory=list)
    clinical_studies: int = 0


@dataclass
class AtomicProfile:
    """
    Atomic profile for trace elements and minerals.
    Measured by ICP-MS.
    """
    # Identification
    element_symbol: str  # e.g., "Fe", "Zn", "Pb"
    element_name: str  # e.g., "Iron", "Zinc", "Lead"
    atomic_number: int
    
    # Category
    is_essential: bool  # Essential nutrient
    is_toxic: bool  # Toxic element
    
    # Concentration Ranges (in ppm)
    typical_range_ppm: Tuple[float, float] = (0.0, 0.0)
    safe_upper_limit_ppm: Optional[float] = None  # Maximum safe level
    deficiency_threshold_ppm: Optional[float] = None  # Below this = deficiency
    
    # Health Properties
    health_role: Optional[str] = None  # e.g., "oxygen transport" (for Fe)
    deficiency_symptoms: List[str] = field(default_factory=list)
    toxicity_symptoms: List[str] = field(default_factory=list)
    
    # Detection
    icp_ms_mass: int = 0  # Mass used in ICP-MS detection
    interference_elements: List[str] = field(default_factory=list)
    
    # Geographic Markers
    soil_origin_indicator: bool = False  # Can indicate geographic origin
    typical_regions: List[str] = field(default_factory=list)


# ================================
# Spectral Database Core
# ================================

class SpectralDatabase:
    """
    Core database for storing spectral signatures and molecular profiles.
    
    This is the foundation of the entire Visual Molecular AI system.
    All training data links visual properties to lab-verified composition.
    """
    
    def __init__(self, db_path: str = "spectral_database.db"):
        self.db_path = db_path
        self.conn = None
        self.signatures: Dict[str, SpectralSignature] = {}
        self.molecules: Dict[str, MolecularProfile] = {}
        self.atoms: Dict[str, AtomicProfile] = {}
        
        # Initialize database
        self._init_database()
        self._load_baseline_molecules()
        self._load_baseline_atoms()
        
        logger.info(f"✅ Spectral Database initialized at {db_path}")
        logger.info(f"   Molecules: {len(self.molecules)}")
        logger.info(f"   Atoms: {len(self.atoms)}")
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Spectral Signatures Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spectral_signatures (
                signature_id TEXT PRIMARY KEY,
                food_type TEXT NOT NULL,
                variety TEXT,
                sample_date TEXT,
                color_rgb TEXT,
                color_hsv TEXT,
                color_lab TEXT,
                spectral_reflectance TEXT,
                volume_cm3 REAL,
                mass_g REAL,
                density_g_cm3 REAL,
                molecules TEXT,
                atoms TEXT,
                lab_technique TEXT,
                lab_verified INTEGER,
                confidence_score REAL,
                freshness_score REAL,
                ripeness_score REAL,
                notes TEXT
            )
        """)
        
        # Molecular Profiles Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS molecular_profiles (
                molecule_id TEXT PRIMARY KEY,
                common_name TEXT NOT NULL,
                chemical_formula TEXT,
                cas_number TEXT,
                category TEXT,
                absorption_peaks TEXT,
                typical_color_rgb TEXT,
                typical_color_hsv TEXT,
                concentration_range TEXT,
                health_benefits TEXT,
                bioavailability REAL
            )
        """)
        
        # Atomic Profiles Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS atomic_profiles (
                element_symbol TEXT PRIMARY KEY,
                element_name TEXT NOT NULL,
                atomic_number INTEGER,
                is_essential INTEGER,
                is_toxic INTEGER,
                typical_range_ppm TEXT,
                safe_upper_limit_ppm REAL,
                health_role TEXT
            )
        """)
        
        self.conn.commit()
    
    def _load_baseline_molecules(self):
        """Load baseline molecular profiles for common chromophores."""
        
        # Beta-Carotene (Orange pigment)
        self.molecules['beta_carotene'] = MolecularProfile(
            molecule_id='beta_carotene',
            common_name='Beta-Carotene',
            chemical_formula='C40H56',
            cas_number='7235-40-7',
            category=MoleculeCategory.PIGMENT,
            subcategory='carotenoid',
            absorption_peaks_nm=[450, 478],  # Absorbs blue light
            conjugated_bonds=11,
            chromophore_type='polyene',
            typical_color_rgb=(255, 140, 0),  # Orange
            typical_color_hsv=(30, 100, 100),
            color_intensity_factor=0.8,
            typical_concentration_mg_100g=(0.5, 15.0),
            detection_limit_mg_100g=0.01,
            saturation_point_mg_100g=20.0,
            health_benefits=['vitamin_a_precursor', 'eye_health', 'immune_support'],
            target_organs=['eyes', 'skin', 'immune_system'],
            bioavailability=0.65,
            light_stable=False,
            heat_stable=True,
            oxidation_prone=True,
            common_foods=['carrot', 'sweet_potato', 'mango', 'apricot'],
            clinical_studies=1500
        )
        
        # Anthocyanins (Purple/Red pigments)
        self.molecules['cyanidin_3_glucoside'] = MolecularProfile(
            molecule_id='cyanidin_3_glucoside',
            common_name='Cyanidin-3-glucoside',
            chemical_formula='C21H21O11',
            cas_number='7084-24-4',
            category=MoleculeCategory.ANTIOXIDANT,
            subcategory='anthocyanin',
            absorption_peaks_nm=[520, 280],  # Red-purple
            chromophore_type='anthocyanin',
            typical_color_rgb=(128, 0, 128),  # Purple
            typical_color_hsv=(280, 100, 50),
            color_intensity_factor=0.9,
            typical_concentration_mg_100g=(50, 500),
            detection_limit_mg_100g=1.0,
            saturation_point_mg_100g=1000.0,
            health_benefits=['brain_health', 'antioxidant', 'anti_inflammatory', 'memory'],
            target_organs=['brain', 'cardiovascular', 'eyes'],
            bioavailability=0.12,  # Low but highly potent
            light_stable=False,
            heat_stable=False,
            pH_sensitive=True,  # Changes color with pH
            oxidation_prone=True,
            common_foods=['blueberry', 'blackberry', 'red_cabbage', 'purple_grapes'],
            clinical_studies=800
        )
        
        # Chlorophyll (Green pigment)
        self.molecules['chlorophyll_a'] = MolecularProfile(
            molecule_id='chlorophyll_a',
            common_name='Chlorophyll A',
            chemical_formula='C55H72MgN4O5',
            cas_number='479-61-8',
            category=MoleculeCategory.PIGMENT,
            subcategory='chlorophyll',
            absorption_peaks_nm=[430, 662],  # Red and blue (reflects green)
            chromophore_type='porphyrin',
            typical_color_rgb=(0, 128, 0),  # Green
            typical_color_hsv=(120, 100, 50),
            color_intensity_factor=0.95,
            typical_concentration_mg_100g=(10, 200),
            detection_limit_mg_100g=0.5,
            saturation_point_mg_100g=500.0,
            health_benefits=['detoxification', 'magnesium_source', 'wound_healing'],
            target_organs=['liver', 'blood', 'digestive_system'],
            bioavailability=0.20,
            light_stable=False,
            heat_stable=False,
            pH_sensitive=True,
            oxidation_prone=True,
            common_foods=['spinach', 'kale', 'parsley', 'green_beans'],
            clinical_studies=400
        )
        
        # Lycopene (Red pigment)
        self.molecules['lycopene'] = MolecularProfile(
            molecule_id='lycopene',
            common_name='Lycopene',
            chemical_formula='C40H56',
            cas_number='502-65-8',
            category=MoleculeCategory.ANTIOXIDANT,
            subcategory='carotenoid',
            absorption_peaks_nm=[446, 472, 505],  # Blue-green (appears red)
            conjugated_bonds=11,
            chromophore_type='polyene',
            typical_color_rgb=(255, 0, 0),  # Red
            typical_color_hsv=(0, 100, 100),
            color_intensity_factor=0.85,
            typical_concentration_mg_100g=(3, 30),
            detection_limit_mg_100g=0.1,
            saturation_point_mg_100g=50.0,
            health_benefits=['prostate_health', 'heart_health', 'antioxidant', 'anti_cancer'],
            target_organs=['prostate', 'cardiovascular', 'skin'],
            bioavailability=0.30,  # Increases with heat/oil
            light_stable=False,
            heat_stable=True,  # Actually improves with cooking
            oxidation_prone=True,
            common_foods=['tomato', 'watermelon', 'pink_grapefruit', 'papaya'],
            clinical_studies=1200
        )
        
        # Lutein (Yellow pigment - eye health)
        self.molecules['lutein'] = MolecularProfile(
            molecule_id='lutein',
            common_name='Lutein',
            chemical_formula='C40H56O2',
            cas_number='127-40-2',
            category=MoleculeCategory.ANTIOXIDANT,
            subcategory='xanthophyll',
            absorption_peaks_nm=[445, 475],
            conjugated_bonds=10,
            chromophore_type='polyene',
            typical_color_rgb=(255, 255, 0),  # Yellow
            typical_color_hsv=(60, 100, 100),
            color_intensity_factor=0.6,
            typical_concentration_mg_100g=(1, 20),
            detection_limit_mg_100g=0.05,
            saturation_point_mg_100g=30.0,
            health_benefits=['eye_health', 'macular_degeneration_prevention', 'blue_light_filter'],
            target_organs=['eyes', 'macula', 'retina'],
            bioavailability=0.05,  # Very low, needs fat
            light_stable=False,
            heat_stable=True,
            oxidation_prone=True,
            common_foods=['kale', 'spinach', 'egg_yolk', 'corn'],
            clinical_studies=600
        )
        
        # Polyphenols (Brown color - oxidation)
        self.molecules['quercetin'] = MolecularProfile(
            molecule_id='quercetin',
            common_name='Quercetin',
            chemical_formula='C15H10O7',
            cas_number='117-39-5',
            category=MoleculeCategory.PHYTONUTRIENT,
            subcategory='flavonoid',
            absorption_peaks_nm=[370, 255],
            chromophore_type='flavonoid',
            typical_color_rgb=(210, 180, 140),  # Tan/brown
            typical_color_hsv=(30, 33, 82),
            color_intensity_factor=0.4,
            typical_concentration_mg_100g=(5, 100),
            detection_limit_mg_100g=0.5,
            saturation_point_mg_100g=200.0,
            health_benefits=['anti_inflammatory', 'antioxidant', 'cardiovascular', 'longevity'],
            target_organs=['cardiovascular', 'immune_system', 'brain'],
            bioavailability=0.20,
            light_stable=True,
            heat_stable=True,
            pH_sensitive=False,
            oxidation_prone=False,
            common_foods=['onion', 'apple', 'berries', 'red_wine'],
            clinical_studies=3000
        )
        
        # Aflatoxin B1 (Toxin - yellow-brown)
        self.molecules['aflatoxin_b1'] = MolecularProfile(
            molecule_id='aflatoxin_b1',
            common_name='Aflatoxin B1',
            chemical_formula='C17H12O6',
            cas_number='1162-65-8',
            category=MoleculeCategory.TOXIN,
            subcategory='mycotoxin',
            absorption_peaks_nm=[365],  # UV fluorescence
            chromophore_type='coumarin',
            typical_color_rgb=(218, 165, 32),  # Goldenrod (faint)
            typical_color_hsv=(43, 85, 85),
            color_intensity_factor=0.1,  # Very faint
            typical_concentration_mg_100g=(0.0, 0.02),  # ppb levels
            detection_limit_mg_100g=0.000001,  # 1 ppb
            saturation_point_mg_100g=0.1,
            health_benefits=[],  # NONE - it's a toxin
            target_organs=['liver'],  # Hepatotoxic
            bioavailability=0.90,  # Unfortunately high
            light_stable=False,
            heat_stable=True,
            oxidation_prone=False,
            common_foods=['peanut', 'corn', 'tree_nuts', 'dried_fruit'],  # Contaminated
            clinical_studies=2000  # High due to toxicity research
        )
        
        logger.info(f"   Loaded {len(self.molecules)} baseline molecules")
    
    def _load_baseline_atoms(self):
        """Load baseline atomic profiles for essential and toxic elements."""
        
        # Essential Minerals
        
        # Iron (Fe)
        self.atoms['Fe'] = AtomicProfile(
            element_symbol='Fe',
            element_name='Iron',
            atomic_number=26,
            is_essential=True,
            is_toxic=False,
            typical_range_ppm=(5, 50),  # 5-50 ppm in foods
            safe_upper_limit_ppm=500,
            deficiency_threshold_ppm=5,
            health_role='oxygen_transport_hemoglobin',
            deficiency_symptoms=['anemia', 'fatigue', 'weakness', 'pale_skin'],
            toxicity_symptoms=['liver_damage', 'diabetes', 'heart_problems'],  # If excess
            icp_ms_mass=56,
            interference_elements=['Ar', 'Ca'],
            soil_origin_indicator=False
        )
        
        # Zinc (Zn)
        self.atoms['Zn'] = AtomicProfile(
            element_symbol='Zn',
            element_name='Zinc',
            atomic_number=30,
            is_essential=True,
            is_toxic=False,
            typical_range_ppm=(2, 30),
            safe_upper_limit_ppm=100,
            deficiency_threshold_ppm=5,
            health_role='immune_function_enzyme_cofactor',
            deficiency_symptoms=['immune_weakness', 'hair_loss', 'skin_problems'],
            toxicity_symptoms=['nausea', 'vomiting', 'copper_deficiency'],
            icp_ms_mass=64,
            interference_elements=['Cu'],
            soil_origin_indicator=False
        )
        
        # Calcium (Ca)
        self.atoms['Ca'] = AtomicProfile(
            element_symbol='Ca',
            element_name='Calcium',
            atomic_number=20,
            is_essential=True,
            is_toxic=False,
            typical_range_ppm=(50, 2000),  # Highly variable
            safe_upper_limit_ppm=5000,
            deficiency_threshold_ppm=100,
            health_role='bone_health_muscle_contraction',
            deficiency_symptoms=['osteoporosis', 'muscle_cramps', 'weak_bones'],
            toxicity_symptoms=['kidney_stones', 'calcification'],
            icp_ms_mass=40,
            interference_elements=['K', 'Ar'],
            soil_origin_indicator=False
        )
        
        # Selenium (Se)
        self.atoms['Se'] = AtomicProfile(
            element_symbol='Se',
            element_name='Selenium',
            atomic_number=34,
            is_essential=True,
            is_toxic=True,  # Essential but toxic at high levels
            typical_range_ppm=(0.01, 0.5),
            safe_upper_limit_ppm=1.0,
            deficiency_threshold_ppm=0.05,
            health_role='antioxidant_thyroid_function',
            deficiency_symptoms=['weak_immune', 'thyroid_problems', 'heart_disease'],
            toxicity_symptoms=['hair_loss', 'nail_brittleness', 'garlic_breath'],
            icp_ms_mass=78,
            interference_elements=['Kr', 'Ar'],
            soil_origin_indicator=True  # Varies greatly by region
        )
        
        # Toxic Heavy Metals
        
        # Lead (Pb)
        self.atoms['Pb'] = AtomicProfile(
            element_symbol='Pb',
            element_name='Lead',
            atomic_number=82,
            is_essential=False,
            is_toxic=True,
            typical_range_ppm=(0.0, 0.1),  # Should be near zero
            safe_upper_limit_ppm=0.1,  # FDA limit
            deficiency_threshold_ppm=None,  # Not essential
            health_role=None,
            deficiency_symptoms=[],
            toxicity_symptoms=['brain_damage', 'developmental_delays', 'kidney_damage'],
            icp_ms_mass=208,
            interference_elements=[],
            soil_origin_indicator=True  # Industrial contamination
        )
        
        # Cadmium (Cd)
        self.atoms['Cd'] = AtomicProfile(
            element_symbol='Cd',
            element_name='Cadmium',
            atomic_number=48,
            is_essential=False,
            is_toxic=True,
            typical_range_ppm=(0.0, 0.05),
            safe_upper_limit_ppm=0.05,
            deficiency_threshold_ppm=None,
            health_role=None,
            deficiency_symptoms=[],
            toxicity_symptoms=['kidney_damage', 'bone_disease', 'cancer'],
            icp_ms_mass=114,
            interference_elements=['Sn'],
            soil_origin_indicator=True
        )
        
        # Mercury (Hg)
        self.atoms['Hg'] = AtomicProfile(
            element_symbol='Hg',
            element_name='Mercury',
            atomic_number=80,
            is_essential=False,
            is_toxic=True,
            typical_range_ppm=(0.0, 0.03),
            safe_upper_limit_ppm=0.03,  # EPA limit for fish
            deficiency_threshold_ppm=None,
            health_role=None,
            deficiency_symptoms=[],
            toxicity_symptoms=['neurological_damage', 'tremors', 'memory_loss'],
            icp_ms_mass=202,
            interference_elements=[],
            soil_origin_indicator=False  # Bioaccumulation in fish
        )
        
        # Arsenic (As)
        self.atoms['As'] = AtomicProfile(
            element_symbol='As',
            element_name='Arsenic',
            atomic_number=33,
            is_essential=False,
            is_toxic=True,
            typical_range_ppm=(0.0, 0.1),
            safe_upper_limit_ppm=0.1,
            deficiency_threshold_ppm=None,
            health_role=None,
            deficiency_symptoms=[],
            toxicity_symptoms=['cancer', 'skin_lesions', 'vascular_disease'],
            icp_ms_mass=75,
            interference_elements=['Se'],
            soil_origin_indicator=True  # Groundwater contamination
        )
        
        logger.info(f"   Loaded {len(self.atoms)} baseline atoms")
    
    def add_signature(self, signature: SpectralSignature) -> bool:
        """
        Add a spectral signature to the database.
        
        Args:
            signature: SpectralSignature object
        
        Returns:
            True if added successfully
        """
        try:
            # Store in memory
            self.signatures[signature.signature_id] = signature
            
            # Store in database
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO spectral_signatures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signature.signature_id,
                signature.food_type,
                signature.variety,
                signature.sample_date.isoformat(),
                json.dumps(signature.color_rgb),
                json.dumps(signature.color_hsv),
                json.dumps(signature.color_lab),
                json.dumps(signature.spectral_reflectance),
                signature.volume_cm3,
                signature.mass_g,
                signature.density_g_cm3,
                json.dumps(signature.molecules),
                json.dumps(signature.atoms),
                signature.lab_technique.value,
                int(signature.lab_verified),
                signature.confidence_score,
                signature.freshness_score,
                signature.ripeness_score,
                signature.notes
            ))
            self.conn.commit()
            
            return True
        except Exception as e:
            logger.error(f"Failed to add signature {signature.signature_id}: {e}")
            return False
    
    def query_by_color(
        self,
        hsv_color: Tuple[float, float, float],
        tolerance: float = 10.0
    ) -> List[SpectralSignature]:
        """
        Query signatures by HSV color within tolerance.
        
        Args:
            hsv_color: Target HSV color (H: 0-360, S: 0-100, V: 0-100)
            tolerance: Color difference tolerance
        
        Returns:
            List of matching signatures
        """
        matches = []
        h_target, s_target, v_target = hsv_color
        
        for sig in self.signatures.values():
            h, s, v = sig.color_hsv
            
            # Calculate color distance (simplified)
            h_diff = min(abs(h - h_target), 360 - abs(h - h_target))  # Circular hue
            s_diff = abs(s - s_target)
            v_diff = abs(v - v_target)
            
            distance = np.sqrt(h_diff**2 + s_diff**2 + v_diff**2)
            
            if distance <= tolerance:
                matches.append(sig)
        
        return matches
    
    def query_by_molecule(self, molecule_id: str, min_concentration: float = 0.0) -> List[SpectralSignature]:
        """Query signatures containing a specific molecule above threshold."""
        matches = []
        
        for sig in self.signatures.values():
            if molecule_id in sig.molecules:
                if sig.molecules[molecule_id] >= min_concentration:
                    matches.append(sig)
        
        return matches
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        return {
            'total_signatures': len(self.signatures),
            'food_types': len(set(sig.food_type for sig in self.signatures.values())),
            'lab_verified': sum(1 for sig in self.signatures.values() if sig.lab_verified),
            'molecules_tracked': len(self.molecules),
            'atoms_tracked': len(self.atoms),
            'avg_confidence': np.mean([sig.confidence_score for sig in self.signatures.values()]) if self.signatures else 0.0
        }


# ================================
# Demo Usage
# ================================

def demo_spectral_database():
    """Demonstrate spectral database functionality."""
    
    print("="*80)
    print("  PHASE 1: SPECTRAL DATABASE SYSTEM - DEMO")
    print("="*80)
    
    # Initialize database
    db = SpectralDatabase("demo_spectral.db")
    
    # Show loaded molecules
    print("\n📊 BASELINE MOLECULES LOADED:")
    for mol_id, mol in db.molecules.items():
        print(f"\n{mol.common_name} ({mol.chemical_formula})")
        print(f"  Color: RGB{mol.typical_color_rgb}, HSV{mol.typical_color_hsv}")
        print(f"  Absorption peaks: {mol.absorption_peaks_nm} nm")
        print(f"  Health benefits: {', '.join(mol.health_benefits[:3])}")
        print(f"  Found in: {', '.join(mol.common_foods[:4])}")
    
    # Show loaded atoms
    print("\n⚛️  BASELINE ATOMS LOADED:")
    for elem, atom in db.atoms.items():
        status = "✅ Essential" if atom.is_essential else "⚠️  Toxic" if atom.is_toxic else "Neutral"
        print(f"  {elem} ({atom.element_name}): {status}")
        if atom.health_role:
            print(f"     Role: {atom.health_role}")
    
    # Create sample signature (Carrot)
    print("\n🥕 CREATING SAMPLE SIGNATURE: Carrot (High Beta-Carotene)")
    carrot_sig = SpectralSignature(
        signature_id="carrot_001",
        food_type="carrot",
        variety="Nantes",
        sample_date=datetime.now(),
        color_rgb=(237, 145, 33),  # Orange
        color_hsv=(30, 86, 93),
        color_lab=(70, 30, 60),
        spectral_reflectance={
            450: 0.15,  # Absorbs blue
            550: 0.50,  # Moderate green
            590: 0.85,  # Reflects orange
            650: 0.30   # Low red
        },
        volume_cm3=180,
        mass_g=175,
        density_g_cm3=0.97,
        surface_area_cm2=120,
        molecules={
            'beta_carotene': 8.3,  # mg/100g
            'lutein': 0.3
        },
        atoms={
            'Fe': 0.3,  # ppm
            'Zn': 0.24,
            'Ca': 33
        },
        lab_technique=LabTechnique.HPLC,
        lab_verified=True,
        confidence_score=0.95,
        freshness_score=1.0,
        ripeness_score=0.8,
        origin_location="California, USA"
    )
    
    db.add_signature(carrot_sig)
    print(f"  ✅ Added signature: {carrot_sig.signature_id}")
    print(f"     Beta-carotene: {carrot_sig.molecules['beta_carotene']} mg/100g")
    print(f"     Color: RGB{carrot_sig.color_rgb}")
    
    # Query by color
    print("\n🔍 QUERY: Find foods with similar orange color")
    matches = db.query_by_color((30, 80, 90), tolerance=15)
    print(f"  Found {len(matches)} matches")
    for match in matches:
        print(f"    - {match.food_type} ({match.variety})")
    
    # Statistics
    print("\n📈 DATABASE STATISTICS:")
    stats = db.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Demo complete!")
    print("\nNext: Part 2 - Color-to-Molecule Mapping Engine")


if __name__ == "__main__":
    demo_spectral_database()
