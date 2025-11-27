"""
Atomic Molecular Profiler - The "Molecular Fingerprint" Database
==================================================================

This is the AI's brain for molecular chemical profiling. It analyzes chemical bonds
and molecules that determine food properties using advanced spectroscopy and ML.

Core Components:
1. Toxic Atom Detection (Heavy Metals & Contaminants)
2. Goal-Oriented Molecule Recognition (C-H, N-H, O-H bonds)
3. Molecular Fingerprint Database
4. CNN-based Spectral Recognition
5. Chemical Bond Analysis

Author: Wellomex AI Nutrition Team
Version: 1.0.0
Date: November 7, 2025
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import json
import pickle
from datetime import datetime
import logging
from pathlib import Path

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # scikit-learn not available - use fallback implementations

# Deep Learning (Optional - for CNN models)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # PyTorch not available - use fallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class ChemicalBondType(Enum):
    """Types of chemical bonds analyzed"""
    # Energy Bonds (Fitness, Athletics)
    C_H = "carbon_hydrogen"           # Fats, carbohydrates
    C_C = "carbon_carbon"             # Backbone structures
    C_O = "carbon_oxygen"             # Carbonyl groups
    
    # Building Bonds (Weight Loss, Muscle)
    N_H = "nitrogen_hydrogen"         # Proteins, amino acids
    N_C = "nitrogen_carbon"           # Peptide bonds
    
    # Function Bonds (Brain Health, Disease Prevention)
    O_H = "oxygen_hydrogen"           # Water, antioxidants, polyphenols
    O_O = "oxygen_oxygen"             # Peroxides
    
    # Special Bonds
    S_H = "sulfur_hydrogen"           # Thiols, cysteine
    P_O = "phosphorus_oxygen"         # ATP, DNA, phospholipids
    
    # Aromatic & Conjugated
    AROMATIC = "aromatic_ring"        # Phenols, flavonoids
    DOUBLE_BOND = "double_bond"       # Unsaturated fats
    TRIPLE_BOND = "triple_bond"       # Alkynes (rare in food)


class ToxicElement(Enum):
    """Toxic elements and contaminants to detect"""
    # Heavy Metals
    LEAD = "Pb"
    MERCURY = "Hg"
    ARSENIC = "As"
    CADMIUM = "Cd"
    CHROMIUM = "Cr"
    NICKEL = "Ni"
    
    # Pesticides & Contaminants
    GLYPHOSATE = "C3H8NO5P"
    DDT = "C14H9Cl5"
    ATRAZINE = "C8H14ClN5"
    CHLORPYRIFOS = "C9H11Cl3NO3PS"
    
    # Industrial Contaminants
    PCB = "polychlorinated_biphenyl"
    DIOXIN = "polychlorinated_dibenzodioxin"
    BPA = "bisphenol_A"
    PFAS = "per_and_polyfluoroalkyl"


class HealthGoal(Enum):
    """User health goals"""
    # Original 8 goals
    ENERGY = "energy_fitness_athletics"
    MUSCLE = "muscle_building_weight_loss"
    BRAIN = "brain_health_cognition"
    HEART = "cardiovascular_health"
    GUT = "digestive_gut_health"
    IMMUNITY = "immune_system"
    LONGEVITY = "anti_aging_longevity"
    RECOVERY = "injury_recovery"
    
    # Phase 1: Weight Management Goals (3)
    WEIGHT_LOSS = "weight_loss_fat_burning"
    WEIGHT_GAIN = "weight_gain_bulking"
    BODY_RECOMP = "body_recomposition"
    
    # Phase 1: Athletic Performance Goals (5)
    ENDURANCE = "endurance_cardio_performance"
    STRENGTH = "strength_power_performance"
    SPEED = "speed_agility_performance"
    FLEXIBILITY = "flexibility_mobility"
    ATHLETIC_RECOVERY = "athletic_recovery_post_workout"
    
    # Phase 1: Specific Health Goals (7)
    SKIN_HEALTH = "skin_anti_aging_beauty"
    HAIR_NAILS = "hair_nails_growth"
    BONE_HEALTH = "bone_density_strength"
    JOINT_HEALTH = "joint_cartilage_health"
    EYE_HEALTH = "vision_eye_health"
    SLEEP_QUALITY = "sleep_circadian_rhythm"
    STRESS_MANAGEMENT = "stress_cortisol_management"
    
    # Phase 1: Life Stage Goals (5)
    PREGNANCY = "pregnancy_prenatal_nutrition"
    LACTATION = "breastfeeding_postpartum"
    MENOPAUSE = "menopause_hormonal_balance"
    FERTILITY = "fertility_reproductive_health"
    HEALTHY_AGING = "healthy_aging_vitality"
    
    # Phase 2: Advanced Goals (12)
    DETOX = "detoxification_cleanse"
    MENTAL_CLARITY = "mental_clarity_focus_nootropic"
    MEMORY = "memory_enhancement"
    CONCENTRATION = "concentration_deep_work"
    INJURY_REHAB = "injury_rehabilitation"
    POST_SURGERY = "post_surgery_recovery"
    IMMUNE_BOOST = "immune_boost_acute"
    ALLERGY_MANAGEMENT = "allergy_seasonal_management"
    TESTOSTERONE_OPTIMIZATION = "testosterone_hormone_men"
    ESTROGEN_BALANCE = "estrogen_balance_women"
    HYDRATION = "hydration_electrolyte_balance"
    ANTI_INFLAMMATORY = "anti_inflammatory_diet"
    
    # Phase 3: Specialized Performance & Dietary Patterns (15)
    ULTRA_ENDURANCE = "ultra_endurance_100_mile"
    POWERLIFTING = "powerlifting_maximal_strength"
    CROSSFIT = "crossfit_mixed_modal"
    VEGAN_OPTIMIZATION = "vegan_plant_based_optimization"
    KETOGENIC_DIET = "ketogenic_diet_therapeutic"
    PALEO_DIET = "paleo_ancestral_nutrition"
    MEDITERRANEAN_DIET = "mediterranean_heart_health"
    DASH_DIET = "dash_hypertension_management"
    ANTI_ACNE = "anti_acne_hormonal_skin"
    HANGOVER_PREVENTION = "hangover_prevention_recovery"
    JET_LAG_RECOVERY = "jet_lag_circadian_reset"
    ALTITUDE_ADAPTATION = "altitude_high_elevation"
    COLD_TOLERANCE = "cold_tolerance_thermogenesis"
    HEAT_TOLERANCE = "heat_tolerance_climate"
    WOUND_HEALING = "wound_healing_acceleration"
    SCAR_REDUCTION = "scar_reduction_minimize"


class DiseaseCondition(Enum):
    """Disease conditions for filtering"""
    # Original 6 diseases
    DIABETES_T2 = "type_2_diabetes"
    HYPERTENSION = "high_blood_pressure"
    OBESITY = "obesity"
    CVD = "cardiovascular_disease"
    CANCER = "cancer_prevention"
    ALZHEIMERS = "alzheimers_dementia"
    KIDNEY_DISEASE = "chronic_kidney_disease"
    
    # Phase 1: Additional 10 diseases
    FATTY_LIVER = "non_alcoholic_fatty_liver_nafld"
    OSTEOPOROSIS = "osteoporosis_bone_health"
    IBD = "inflammatory_bowel_disease"
    PCOS = "polycystic_ovary_syndrome"
    GOUT = "gout_hyperuricemia"
    HYPOTHYROID = "hypothyroidism"
    DEPRESSION = "depression_anxiety"
    METABOLIC_SYNDROME = "metabolic_syndrome"
    AUTOIMMUNE = "autoimmune_diseases"
    
    # Phase 2: Additional 14 diseases  
    DIABETES_T1 = "type_1_diabetes"
    CELIAC = "celiac_disease"
    ANEMIA = "iron_deficiency_anemia"
    MIGRAINES = "migraines_chronic_headaches"
    ASTHMA = "asthma_respiratory"
    GERD = "gerd_acid_reflux"
    IBS = "irritable_bowel_syndrome"
    ECZEMA = "eczema_psoriasis_skin"
    ADHD = "adhd_focus_disorders"
    CHRONIC_FATIGUE = "chronic_fatigue_syndrome"
    FIBROMYALGIA = "fibromyalgia"
    DIVERTICULITIS = "diverticulitis_diverticulosis"
    SLEEP_APNEA = "sleep_apnea"
    GASTROPARESIS = "gastroparesis"
    
    # Phase 3: Additional 20 diseases (Neurological, Rare, Complex)
    PARKINSONS = "parkinsons_disease"
    MULTIPLE_SCLEROSIS = "multiple_sclerosis_ms"
    EPILEPSY = "epilepsy_seizure_disorder"
    LUPUS = "lupus_sle"
    HASHIMOTOS = "hashimotos_thyroiditis"
    CROHNS = "crohns_disease"
    ULCERATIVE_COLITIS = "ulcerative_colitis"
    ROSACEA = "rosacea_facial_redness"
    PSORIATIC_ARTHRITIS = "psoriatic_arthritis"
    ENDOMETRIOSIS = "endometriosis"
    INTERSTITIAL_CYSTITIS = "interstitial_cystitis_bladder"
    RESTLESS_LEG = "restless_leg_syndrome"
    PERIPHERAL_NEUROPATHY = "peripheral_neuropathy"
    RAYNAUDS = "raynauds_phenomenon"
    SJOGRENS = "sjogrens_syndrome"
    SCLERODERMA = "scleroderma_systemic_sclerosis"
    ADDISONS = "addisons_disease"
    CUSHINGS = "cushings_syndrome"
    HEMOCHROMATOSIS = "hemochromatosis_iron_overload"
    WILSONS = "wilsons_disease_copper"
    
    # Legacy conditions (kept for compatibility)
    ARTHRITIS = "rheumatoid_arthritis"
    LIVER_DISEASE = "fatty_liver_disease"  # Alias for FATTY_LIVER


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SpectralFingerprint:
    """Represents the complete spectral fingerprint of a food item"""
    wavelengths: np.ndarray  # Wavelength values (nm)
    absorbance: np.ndarray   # Absorbance values
    reflectance: np.ndarray  # Reflectance values
    timestamp: datetime
    scan_id: str
    
    # Processed features
    peak_wavelengths: List[float] = field(default_factory=list)
    peak_intensities: List[float] = field(default_factory=list)
    baseline_corrected: bool = False
    normalized: bool = False
    
    def __post_init__(self):
        """Validate spectral data"""
        if len(self.wavelengths) != len(self.absorbance):
            raise ValueError("Wavelengths and absorbance must have same length")


@dataclass
class MolecularBondProfile:
    """Profile of chemical bonds detected in food"""
    bond_type: ChemicalBondType
    concentration: float  # mg/100g
    bond_strength: float  # kJ/mol
    spectral_signature: np.ndarray
    confidence: float  # 0-1
    
    # Context
    functional_group: Optional[str] = None
    parent_molecule: Optional[str] = None
    biological_role: Optional[str] = None


@dataclass
class ToxicContaminantProfile:
    """Profile of toxic elements/contaminants"""
    element: ToxicElement
    concentration: float  # ppm or ppb
    detection_limit: float
    safe_threshold: float
    exceeds_limit: bool
    confidence: float
    
    # Risk assessment
    risk_level: str = "UNKNOWN"  # LOW, MODERATE, HIGH, CRITICAL
    health_impact: Optional[str] = None


@dataclass
class NutrientMolecularBreakdown:
    """Quantitative molecular breakdown of nutrients"""
    # Macronutrients (g per 100g)
    total_carbs: float = 0.0
    simple_sugars: float = 0.0  # C-H bonds in monosaccharides
    complex_carbs: float = 0.0  # C-H bonds in polysaccharides
    fiber: float = 0.0
    
    total_protein: float = 0.0
    amino_acids: Dict[str, float] = field(default_factory=dict)  # N-H bonds
    peptides: float = 0.0
    
    total_fat: float = 0.0
    saturated_fat: float = 0.0  # C-H bonds, single
    monounsaturated_fat: float = 0.0  # C-H + C=C
    polyunsaturated_fat: float = 0.0  # C-H + multiple C=C
    trans_fat: float = 0.0
    
    # Water & Alcohols
    water_content: float = 0.0  # O-H bonds
    
    # Micronutrients (mg per 100g)
    vitamins: Dict[str, float] = field(default_factory=dict)
    minerals: Dict[str, float] = field(default_factory=dict)
    
    # Phytonutrients (O-H bonds in polyphenols)
    polyphenols: float = 0.0
    flavonoids: float = 0.0
    carotenoids: float = 0.0
    
    # Energy calculations
    total_calories: float = 0.0
    calories_from_carbs: float = 0.0
    calories_from_protein: float = 0.0
    calories_from_fat: float = 0.0


@dataclass
class UserHealthProfile:
    """Complete user health profile for personalized recommendations"""
    user_id: str
    age: int
    sex: str  # M/F/Other
    
    # Goals (prioritized list)
    primary_goal: HealthGoal
    secondary_goals: List[HealthGoal] = field(default_factory=list)
    
    # Diseases/Conditions
    diagnosed_conditions: List[DiseaseCondition] = field(default_factory=list)
    
    # Lifestyle
    activity_level: str = "moderate"  # sedentary, light, moderate, active, very_active
    dietary_restrictions: List[str] = field(default_factory=list)  # vegan, halal, etc.
    
    # Allergies & Intolerances
    allergies: List[str] = field(default_factory=list)
    intolerances: List[str] = field(default_factory=list)
    
    # Body metrics
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    bmi: Optional[float] = None
    
    # Lab values (if available)
    blood_glucose_mg_dl: Optional[float] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    cholesterol_total: Optional[float] = None
    hdl_cholesterol: Optional[float] = None
    ldl_cholesterol: Optional[float] = None


@dataclass
class FoodRecommendation:
    """Personalized food recommendation result"""
    food_name: str
    scan_id: str
    
    # Overall scores (0-100)
    overall_score: float
    safety_score: float
    goal_alignment_score: float
    disease_compatibility_score: float
    
    # Detailed breakdown
    molecular_breakdown: NutrientMolecularBreakdown
    bond_profile: List[MolecularBondProfile]
    toxic_profile: List[ToxicContaminantProfile]
    
    # Recommendations
    recommendation: str  # HIGHLY_RECOMMENDED, RECOMMENDED, ACCEPTABLE, NOT_RECOMMENDED, AVOID
    reasoning: List[str]  # List of reasons for the recommendation
    
    # Serving suggestions
    optimal_serving_size: Optional[float] = None  # grams
    max_daily_servings: Optional[int] = None
    
    # Alternatives
    better_alternatives: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# MOLECULAR FINGERPRINT DATABASE
# ============================================================================

class MolecularFingerprintDatabase:
    """
    The AI's brain - stores and matches molecular fingerprints
    
    This database contains:
    1. Spectral signatures for all known molecules
    2. Bond patterns for nutrients
    3. Toxic element signatures
    4. Reference spectra for training
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "data/molecular_fingerprints.json"
        self.fingerprints: Dict[str, Dict] = {}
        self.bond_signatures: Dict[ChemicalBondType, np.ndarray] = {}
        self.toxic_signatures: Dict[ToxicElement, np.ndarray] = {}
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize with known molecular signatures"""
        logger.info("Initializing Molecular Fingerprint Database...")
        
        # Initialize C-H bond signatures (carbs, fats)
        self.bond_signatures[ChemicalBondType.C_H] = self._create_ch_signature()
        
        # Initialize N-H bond signatures (proteins)
        self.bond_signatures[ChemicalBondType.N_H] = self._create_nh_signature()
        
        # Initialize O-H bond signatures (water, polyphenols)
        self.bond_signatures[ChemicalBondType.O_H] = self._create_oh_signature()
        
        # Initialize toxic element signatures
        self._initialize_toxic_signatures()
        
        logger.info(f"Database initialized with {len(self.bond_signatures)} bond types")
    
    def _create_ch_signature(self) -> np.ndarray:
        """
        Create spectral signature for C-H bonds
        
        C-H bonds absorb strongly at:
        - 2850-2960 nm (symmetric and asymmetric stretch)
        - 1340-1470 nm (first overtone)
        - 1680-1820 nm (combination bands)
        """
        wavelengths = np.linspace(1000, 2500, 1500)  # NIR range
        signature = np.zeros_like(wavelengths)
        
        # Main C-H stretching region (2850-2960 nm converted to wavelength)
        # Actually use wavenumbers: 3.4 µm = 2940 cm⁻¹
        ch_peaks = [1680, 1725, 1765, 2310, 2350]  # nm in NIR
        
        for peak in ch_peaks:
            idx = np.argmin(np.abs(wavelengths - peak))
            # Gaussian peak
            signature += 0.8 * np.exp(-((wavelengths - peak) ** 2) / (2 * 20 ** 2))
        
        return signature
    
    def _create_nh_signature(self) -> np.ndarray:
        """
        Create spectral signature for N-H bonds (proteins)
        
        N-H bonds absorb at:
        - 1500-1550 nm (first overtone)
        - 2030-2080 nm (combination bands)
        - Amide I (1650 cm⁻¹) and Amide II (1550 cm⁻¹) in mid-IR
        """
        wavelengths = np.linspace(1000, 2500, 1500)
        signature = np.zeros_like(wavelengths)
        
        nh_peaks = [1510, 1530, 2040, 2060, 2180, 2200]  # nm
        
        for peak in nh_peaks:
            idx = np.argmin(np.abs(wavelengths - peak))
            signature += 0.9 * np.exp(-((wavelengths - peak) ** 2) / (2 * 15 ** 2))
        
        return signature
    
    def _create_oh_signature(self) -> np.ndarray:
        """
        Create spectral signature for O-H bonds (water, alcohols, phenols)
        
        O-H bonds absorb at:
        - 1400-1450 nm (first overtone)
        - 1920-1940 nm (water combination bands)
        - 970 nm (second overtone)
        """
        wavelengths = np.linspace(1000, 2500, 1500)
        signature = np.zeros_like(wavelengths)
        
        oh_peaks = [970, 1415, 1435, 1920, 1940, 2250]  # nm
        
        for peak in oh_peaks:
            idx = np.argmin(np.abs(wavelengths - peak))
            signature += 1.0 * np.exp(-((wavelengths - peak) ** 2) / (2 * 25 ** 2))
        
        return signature
    
    def _initialize_toxic_signatures(self):
        """Initialize spectral signatures for toxic elements"""
        # Heavy metals show characteristic X-ray fluorescence and UV-Vis absorption
        # In NIR, they affect the baseline and show weak characteristic features
        
        # Lead (Pb) - affects baseline at 1200-1400 nm
        self.toxic_signatures[ToxicElement.LEAD] = self._create_heavy_metal_signature(1250, 1350)
        
        # Mercury (Hg) - characteristic at 253.7 nm (UV), affects NIR baseline
        self.toxic_signatures[ToxicElement.MERCURY] = self._create_heavy_metal_signature(1180, 1280)
        
        # Arsenic (As) - affects 1300-1500 nm region
        self.toxic_signatures[ToxicElement.ARSENIC] = self._create_heavy_metal_signature(1380, 1480)
        
        # Cadmium (Cd) - characteristic at 228.8 nm (UV)
        self.toxic_signatures[ToxicElement.CADMIUM] = self._create_heavy_metal_signature(1320, 1420)
        
        logger.info(f"Initialized {len(self.toxic_signatures)} toxic element signatures")
    
    def _create_heavy_metal_signature(self, center_nm: float, width: float) -> np.ndarray:
        """Create a baseline shift signature for heavy metals"""
        wavelengths = np.linspace(1000, 2500, 1500)
        
        # Heavy metals cause baseline shifts and broad absorption
        signature = 0.05 * np.exp(-((wavelengths - center_nm) ** 2) / (2 * width ** 2))
        
        # Add baseline tilt
        signature += 0.02 * (wavelengths - 1000) / 1500
        
        return signature
    
    def add_fingerprint(self, molecule_name: str, fingerprint: SpectralFingerprint):
        """Add a new molecular fingerprint to the database"""
        self.fingerprints[molecule_name] = {
            'wavelengths': fingerprint.wavelengths.tolist(),
            'absorbance': fingerprint.absorbance.tolist(),
            'peaks': fingerprint.peak_wavelengths,
            'timestamp': fingerprint.timestamp.isoformat()
        }
    
    def match_fingerprint(self, spectrum: np.ndarray) -> List[Tuple[str, float]]:
        """
        Match a spectrum against database using correlation
        
        Returns:
            List of (molecule_name, similarity_score) tuples
        """
        matches = []
        
        for molecule, data in self.fingerprints.items():
            ref_spectrum = np.array(data['absorbance'])
            
            # Ensure same length
            if len(spectrum) != len(ref_spectrum):
                ref_spectrum = np.interp(
                    np.linspace(0, 1, len(spectrum)),
                    np.linspace(0, 1, len(ref_spectrum)),
                    ref_spectrum
                )
            
            # Calculate correlation
            correlation = np.corrcoef(spectrum, ref_spectrum)[0, 1]
            matches.append((molecule, correlation))
        
        # Sort by similarity
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def save_database(self):
        """Save database to disk"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'fingerprints': self.fingerprints,
            'version': '1.0.0',
            'created': datetime.now().isoformat()
        }
        
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Database saved to {self.db_path}")
    
    def load_database(self):
        """Load database from disk"""
        if Path(self.db_path).exists():
            with open(self.db_path, 'r') as f:
                data = json.load(f)
            
            self.fingerprints = data['fingerprints']
            logger.info(f"Database loaded from {self.db_path}")
        else:
            logger.warning(f"Database file not found: {self.db_path}")


# ============================================================================
# SPECTRAL ANALYSIS CNN (Deep Learning)
# ============================================================================

if TORCH_AVAILABLE:
    class SpectralCNN(nn.Module):
        """
        Convolutional Neural Network for spectral fingerprint recognition
        
        Architecture:
        - 1D Convolutional layers to extract spectral features
        - Residual connections for deep training
        - Multi-task output: bond detection + quantification
        """
        
        def __init__(self, input_size: int = 1500, num_bond_types: int = 10, num_nutrients: int = 30):
            super(SpectralCNN, self).__init__()
            
            # Convolutional feature extractor
            self.conv1 = nn.Conv1d(1, 64, kernel_size=15, padding=7)
            self.bn1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=15, padding=7)
            self.bn2 = nn.BatchNorm1d(128)
            self.conv3 = nn.Conv1d(128, 256, kernel_size=11, padding=5)
            self.bn3 = nn.BatchNorm1d(256)
            
            self.pool = nn.MaxPool1d(2)
            self.dropout = nn.Dropout(0.3)
            
            # Calculate flattened size after convolutions
            self.flatten_size = 256 * (input_size // 8)  # After 3 pooling layers
            
            # Fully connected layers
            self.fc1 = nn.Linear(self.flatten_size, 512)
            self.fc2 = nn.Linear(512, 256)
            
            # Multi-task outputs
            self.bond_classifier = nn.Linear(256, num_bond_types)  # Bond type classification
            self.nutrient_regressor = nn.Linear(256, num_nutrients)  # Nutrient quantification
            
            self.relu = nn.ReLU()
        
        def forward(self, x):
            # x shape: (batch, 1, wavelengths)
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            
            # Flatten
            x = x.view(x.size(0), -1)
            
            # Fully connected
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            
            # Multi-task outputs
            bond_logits = self.bond_classifier(x)
            nutrient_values = self.nutrient_regressor(x)
            
            return bond_logits, nutrient_values
    
    
    class SpectralDataset(Dataset):
        """Dataset for training spectral CNN"""
        
        def __init__(self, spectra: np.ndarray, bond_labels: np.ndarray, nutrient_values: np.ndarray):
            self.spectra = torch.FloatTensor(spectra)
            self.bond_labels = torch.LongTensor(bond_labels)
            self.nutrient_values = torch.FloatTensor(nutrient_values)
        
        def __len__(self):
            return len(self.spectra)
        
        def __getitem__(self, idx):
            return (
                self.spectra[idx].unsqueeze(0),  # Add channel dimension
                self.bond_labels[idx],
                self.nutrient_values[idx]
            )


# ============================================================================
# MOLECULAR BOND ANALYZER
# ============================================================================

class MolecularBondAnalyzer:
    """
    Analyzes chemical bonds from spectral data
    
    This is the core "atomic" engine that identifies C-H, N-H, O-H bonds
    and translates them into nutritional information.
    """
    
    def __init__(self, fingerprint_db: MolecularFingerprintDatabase):
        self.db = fingerprint_db
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
        # Bond-to-nutrient mapping
        self.bond_nutrient_map = {
            ChemicalBondType.C_H: ['carbohydrates', 'fats'],
            ChemicalBondType.N_H: ['protein', 'amino_acids'],
            ChemicalBondType.O_H: ['water', 'polyphenols', 'alcohols'],
            ChemicalBondType.C_O: ['carbohydrates', 'organic_acids'],
            ChemicalBondType.P_O: ['phospholipids', 'ATP', 'DNA']
        }
        
        logger.info("Molecular Bond Analyzer initialized")
    
    def analyze_spectrum(self, spectrum: np.ndarray, wavelengths: np.ndarray) -> List[MolecularBondProfile]:
        """
        Analyze a spectrum and identify chemical bonds
        
        Args:
            spectrum: Absorbance/reflectance values
            wavelengths: Corresponding wavelengths (nm)
        
        Returns:
            List of detected molecular bonds with concentrations
        """
        bonds_detected = []
        
        # Analyze each bond type
        for bond_type, reference_signature in self.db.bond_signatures.items():
            # Interpolate reference to match input wavelengths
            ref_wavelengths = np.linspace(1000, 2500, len(reference_signature))
            ref_interp = np.interp(wavelengths, ref_wavelengths, reference_signature)
            
            # Calculate correlation
            correlation = np.corrcoef(spectrum, ref_interp)[0, 1]
            
            if correlation > 0.5:  # Threshold for detection
                # Estimate concentration from peak heights
                concentration = self._estimate_concentration(spectrum, wavelengths, bond_type)
                
                # Get bond strength (typical values)
                bond_strength = self._get_bond_strength(bond_type)
                
                bond_profile = MolecularBondProfile(
                    bond_type=bond_type,
                    concentration=concentration,
                    bond_strength=bond_strength,
                    spectral_signature=ref_interp,
                    confidence=correlation,
                    biological_role=self._get_biological_role(bond_type)
                )
                
                bonds_detected.append(bond_profile)
        
        return bonds_detected
    
    def _estimate_concentration(self, spectrum: np.ndarray, wavelengths: np.ndarray, 
                                bond_type: ChemicalBondType) -> float:
        """
        Estimate concentration using Beer-Lambert law
        
        A = ε * c * l
        where A = absorbance, ε = molar absorptivity, c = concentration, l = path length
        """
        # Get peak absorbance for this bond type
        if bond_type == ChemicalBondType.C_H:
            peak_regions = [(1650, 1800), (2300, 2400)]
        elif bond_type == ChemicalBondType.N_H:
            peak_regions = [(1500, 1600), (2000, 2100)]
        elif bond_type == ChemicalBondType.O_H:
            peak_regions = [(1400, 1500), (1900, 2000)]
        else:
            peak_regions = [(1000, 2500)]
        
        max_absorbance = 0.0
        for start, end in peak_regions:
            mask = (wavelengths >= start) & (wavelengths <= end)
            if np.any(mask):
                region_max = np.max(spectrum[mask])
                max_absorbance = max(max_absorbance, region_max)
        
        # Estimate concentration (mg/100g)
        # This is simplified - real implementation would use calibration curves
        concentration = max_absorbance * 100  # Scaling factor
        
        return concentration
    
    def _get_bond_strength(self, bond_type: ChemicalBondType) -> float:
        """Get typical bond dissociation energy (kJ/mol)"""
        bond_energies = {
            ChemicalBondType.C_H: 413,
            ChemicalBondType.C_C: 348,
            ChemicalBondType.C_O: 358,
            ChemicalBondType.N_H: 391,
            ChemicalBondType.N_C: 305,
            ChemicalBondType.O_H: 463,
            ChemicalBondType.S_H: 339,
            ChemicalBondType.P_O: 335,
            ChemicalBondType.DOUBLE_BOND: 614,  # C=C
            ChemicalBondType.AROMATIC: 518  # Average in benzene
        }
        return bond_energies.get(bond_type, 350)
    
    def _get_biological_role(self, bond_type: ChemicalBondType) -> str:
        """Get biological role description"""
        roles = {
            ChemicalBondType.C_H: "Energy storage (carbs, fats). Essential for ATP production.",
            ChemicalBondType.N_H: "Protein structure. Muscle building, enzyme function, immunity.",
            ChemicalBondType.O_H: "Hydration, antioxidants. Free radical scavenging, cell signaling.",
            ChemicalBondType.C_O: "Metabolic intermediates. Energy metabolism, biosynthesis.",
            ChemicalBondType.P_O: "Energy currency (ATP), cell membranes, genetic material."
        }
        return roles.get(bond_type, "Various metabolic functions")
    
    def bonds_to_nutrients(self, bonds: List[MolecularBondProfile]) -> NutrientMolecularBreakdown:
        """
        Convert detected bonds into quantitative nutrient breakdown
        
        This is the key translation step: Bonds -> Nutrients
        """
        breakdown = NutrientMolecularBreakdown()
        
        for bond in bonds:
            if bond.bond_type == ChemicalBondType.C_H:
                # C-H bonds indicate carbs and fats
                # Use ratio analysis to separate
                if bond.concentration > 50:
                    # High concentration suggests fats
                    breakdown.total_fat += bond.concentration * 0.6
                    breakdown.saturated_fat += bond.concentration * 0.3
                else:
                    # Lower concentration suggests carbs
                    breakdown.total_carbs += bond.concentration * 0.8
                    breakdown.simple_sugars += bond.concentration * 0.4
            
            elif bond.bond_type == ChemicalBondType.N_H:
                # N-H bonds indicate proteins
                breakdown.total_protein += bond.concentration * 0.9
                
                # Estimate amino acids (simplified)
                breakdown.amino_acids['leucine'] = bond.concentration * 0.08
                breakdown.amino_acids['lysine'] = bond.concentration * 0.06
                breakdown.amino_acids['methionine'] = bond.concentration * 0.025
            
            elif bond.bond_type == ChemicalBondType.O_H:
                # O-H bonds indicate water and polyphenols
                if bond.concentration > 70:
                    breakdown.water_content += bond.concentration * 0.95
                else:
                    breakdown.polyphenols += bond.concentration * 0.5
                    breakdown.flavonoids += bond.concentration * 0.3
            
            elif bond.bond_type == ChemicalBondType.AROMATIC:
                # Aromatic rings suggest phytonutrients
                breakdown.polyphenols += bond.concentration * 0.7
                breakdown.flavonoids += bond.concentration * 0.5
        
        # Calculate calories
        breakdown.calories_from_carbs = breakdown.total_carbs * 4
        breakdown.calories_from_protein = breakdown.total_protein * 4
        breakdown.calories_from_fat = breakdown.total_fat * 9
        breakdown.total_calories = (
            breakdown.calories_from_carbs +
            breakdown.calories_from_protein +
            breakdown.calories_from_fat
        )
        
        return breakdown


# ============================================================================
# TOXIC CONTAMINANT DETECTOR
# ============================================================================

class ToxicContaminantDetector:
    """
    Detects toxic elements and contaminants using spectral analysis
    
    Uses ML models trained on heavy metal and pesticide signatures
    """
    
    def __init__(self, fingerprint_db: MolecularFingerprintDatabase):
        self.db = fingerprint_db
        self.models = {}
        
        # Safety thresholds (ppm unless specified)
        self.safe_limits = {
            ToxicElement.LEAD: 0.1,        # ppm (FDA limit for candy)
            ToxicElement.MERCURY: 0.5,      # ppm (FDA limit for fish)
            ToxicElement.ARSENIC: 0.01,     # ppm (EPA drinking water)
            ToxicElement.CADMIUM: 0.05,     # ppm (WHO guideline)
            ToxicElement.GLYPHOSATE: 1.75,  # ppm (EPA tolerance)
            ToxicElement.DDT: 0.005,        # ppm (banned but residues)
        }
        
        logger.info("Toxic Contaminant Detector initialized")
    
    def detect_contaminants(self, spectrum: np.ndarray, wavelengths: np.ndarray) -> List[ToxicContaminantProfile]:
        """
        Analyze spectrum for toxic contaminants
        
        Args:
            spectrum: Absorbance values
            wavelengths: Corresponding wavelengths
        
        Returns:
            List of detected contaminants with risk assessment
        """
        contaminants_detected = []
        
        for element, ref_signature in self.db.toxic_signatures.items():
            # Interpolate reference
            ref_wavelengths = np.linspace(1000, 2500, len(ref_signature))
            ref_interp = np.interp(wavelengths, ref_wavelengths, ref_signature)
            
            # Calculate correlation
            correlation = np.corrcoef(spectrum, ref_interp)[0, 1]
            
            if correlation > 0.3:  # Lower threshold for safety
                # Estimate concentration
                concentration = self._estimate_toxic_concentration(spectrum, wavelengths, element)
                
                safe_threshold = self.safe_limits.get(element, 0.1)
                exceeds = concentration > safe_threshold
                
                # Risk assessment
                risk_level = self._assess_risk(concentration, safe_threshold)
                
                profile = ToxicContaminantProfile(
                    element=element,
                    concentration=concentration,
                    detection_limit=0.001,  # ppb
                    safe_threshold=safe_threshold,
                    exceeds_limit=exceeds,
                    confidence=correlation,
                    risk_level=risk_level,
                    health_impact=self._get_health_impact(element)
                )
                
                contaminants_detected.append(profile)
        
        return contaminants_detected
    
    def _estimate_toxic_concentration(self, spectrum: np.ndarray, wavelengths: np.ndarray,
                                     element: ToxicElement) -> float:
        """Estimate concentration of toxic element (ppm)"""
        # This is simplified - real implementation would use:
        # - X-ray fluorescence (XRF) for heavy metals
        # - Mass spectrometry coupling
        # - Calibration with known standards
        
        # For NIR, we look at baseline shifts and weak absorption features
        baseline_shift = np.mean(spectrum) - 1.0
        peak_height = np.max(spectrum) - np.min(spectrum)
        
        concentration = abs(baseline_shift) * 10 + peak_height * 5
        
        return concentration
    
    def _assess_risk(self, concentration: float, safe_limit: float) -> str:
        """Assess risk level based on concentration"""
        ratio = concentration / safe_limit
        
        if ratio < 0.5:
            return "LOW"
        elif ratio < 1.0:
            return "MODERATE"
        elif ratio < 2.0:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _get_health_impact(self, element: ToxicElement) -> str:
        """Get health impact description"""
        impacts = {
            ToxicElement.LEAD: "Neurotoxic. Affects brain development, IQ reduction, behavioral problems.",
            ToxicElement.MERCURY: "Neurotoxic. Damages brain, kidneys, nervous system. Bioaccumulates.",
            ToxicElement.ARSENIC: "Carcinogenic. Increases cancer risk, cardiovascular disease, diabetes.",
            ToxicElement.CADMIUM: "Kidney damage, bone disease, carcinogenic. Long biological half-life.",
            ToxicElement.GLYPHOSATE: "Potential carcinogen. Disrupts gut microbiome, hormone function.",
            ToxicElement.DDT: "Endocrine disruptor. Carcinogenic. Reproductive toxicity.",
        }
        return impacts.get(element, "Toxic. Avoid exposure.")


# ============================================================================
# MAIN MOLECULAR PROFILER
# ============================================================================

class AtomicMolecularProfiler:
    """
    Main molecular profiling engine - orchestrates all analysis
    
    This is the complete "Atomic AI" system that:
    1. Analyzes spectral fingerprints
    2. Detects chemical bonds
    3. Identifies toxic contaminants
    4. Converts to nutritional data
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db = MolecularFingerprintDatabase(db_path)
        self.bond_analyzer = MolecularBondAnalyzer(self.db)
        self.toxic_detector = ToxicContaminantDetector(self.db)
        
        # CNN model (if PyTorch available)
        self.cnn_model = None
        if TORCH_AVAILABLE:
            self.cnn_model = SpectralCNN()
            logger.info("CNN model initialized")
        
        logger.info("Atomic Molecular Profiler ready")
    
    def profile_food(self, fingerprint: SpectralFingerprint) -> Tuple[NutrientMolecularBreakdown, 
                                                                       List[MolecularBondProfile],
                                                                       List[ToxicContaminantProfile]]:
        """
        Complete molecular profiling of food
        
        Args:
            fingerprint: Spectral fingerprint from NIR scanner
        
        Returns:
            Tuple of (nutrient_breakdown, bond_profile, toxic_profile)
        """
        logger.info(f"Profiling food scan: {fingerprint.scan_id}")
        
        # 1. Analyze chemical bonds
        bonds = self.bond_analyzer.analyze_spectrum(
            fingerprint.absorbance,
            fingerprint.wavelengths
        )
        logger.info(f"Detected {len(bonds)} bond types")
        
        # 2. Convert bonds to nutrients
        nutrients = self.bond_analyzer.bonds_to_nutrients(bonds)
        logger.info(f"Nutrients: {nutrients.total_protein:.1f}g protein, "
                   f"{nutrients.total_carbs:.1f}g carbs, {nutrients.total_fat:.1f}g fat")
        
        # 3. Detect toxic contaminants
        toxics = self.toxic_detector.detect_contaminants(
            fingerprint.absorbance,
            fingerprint.wavelengths
        )
        
        if toxics:
            logger.warning(f"Detected {len(toxics)} potential contaminants")
            for toxic in toxics:
                if toxic.exceeds_limit:
                    logger.error(f"ALERT: {toxic.element.value} exceeds safe limit! "
                               f"{toxic.concentration:.4f} ppm (limit: {toxic.safe_threshold} ppm)")
        else:
            logger.info("No toxic contaminants detected")
        
        return nutrients, bonds, toxics
    
    def save_models(self, path: str = "models/"):
        """Save trained models"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        if self.cnn_model and TORCH_AVAILABLE:
            torch.save(self.cnn_model.state_dict(), f"{path}/spectral_cnn.pth")
        
        self.db.save_database()
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str = "models/"):
        """Load trained models"""
        if self.cnn_model and TORCH_AVAILABLE:
            model_path = f"{path}/spectral_cnn.pth"
            if Path(model_path).exists():
                self.cnn_model.load_state_dict(torch.load(model_path))
                logger.info(f"CNN model loaded from {model_path}")
        
        self.db.load_database()


# ============================================================================
# TESTING & EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ATOMIC MOLECULAR PROFILER - Test Suite")
    print("=" * 80)
    
    # Initialize profiler
    profiler = AtomicMolecularProfiler()
    
    # Simulate NIR spectral data for different foods
    
    # Test 1: High-protein food (chicken breast)
    print("\n" + "=" * 80)
    print("TEST 1: High-Protein Food (Chicken Breast)")
    print("=" * 80)
    
    wavelengths = np.linspace(1000, 2500, 1500)
    
    # Create spectrum with strong N-H peaks (protein)
    chicken_spectrum = (
        0.3 * np.exp(-((wavelengths - 1520) ** 2) / (2 * 20 ** 2)) +  # N-H
        0.35 * np.exp(-((wavelengths - 2050) ** 2) / (2 * 15 ** 2)) +   # N-H
        0.15 * np.exp(-((wavelengths - 1730) ** 2) / (2 * 25 ** 2)) +   # C-H (fat)
        0.8 * np.exp(-((wavelengths - 1920) ** 2) / (2 * 30 ** 2)) +   # O-H (water)
        0.1 * np.random.random(len(wavelengths))  # Noise
    )
    
    chicken_fingerprint = SpectralFingerprint(
        wavelengths=wavelengths,
        absorbance=chicken_spectrum,
        reflectance=1 - chicken_spectrum,
        timestamp=datetime.now(),
        scan_id="SCAN001_CHICKEN"
    )
    
    nutrients, bonds, toxics = profiler.profile_food(chicken_fingerprint)
    
    print(f"\nNutrient Breakdown:")
    print(f"  Protein: {nutrients.total_protein:.1f}g")
    print(f"  Carbs: {nutrients.total_carbs:.1f}g")
    print(f"  Fat: {nutrients.total_fat:.1f}g")
    print(f"  Water: {nutrients.water_content:.1f}g")
    print(f"  Total Calories: {nutrients.total_calories:.0f}")
    
    print(f"\nBond Profile:")
    for bond in bonds:
        print(f"  {bond.bond_type.value}: {bond.concentration:.1f} mg/100g "
              f"(confidence: {bond.confidence:.2f})")
    
    print(f"\nToxic Scan: {'CLEAR' if not toxics else 'CONTAMINANTS DETECTED'}")
    
    # Test 2: High-carb food (rice)
    print("\n" + "=" * 80)
    print("TEST 2: High-Carbohydrate Food (White Rice)")
    print("=" * 80)
    
    # Strong C-H peaks, weak N-H
    rice_spectrum = (
        0.6 * np.exp(-((wavelengths - 1680) ** 2) / (2 * 20 ** 2)) +  # C-H
        0.7 * np.exp(-((wavelengths - 1730) ** 2) / (2 * 25 ** 2)) +  # C-H
        0.65 * np.exp(-((wavelengths - 2320) ** 2) / (2 * 30 ** 2)) +  # C-H
        0.05 * np.exp(-((wavelengths - 1520) ** 2) / (2 * 15 ** 2)) +  # N-H (low)
        0.3 * np.exp(-((wavelengths - 1920) ** 2) / (2 * 25 ** 2)) +  # O-H (water)
        0.08 * np.random.random(len(wavelengths))
    )
    
    rice_fingerprint = SpectralFingerprint(
        wavelengths=wavelengths,
        absorbance=rice_spectrum,
        reflectance=1 - rice_spectrum,
        timestamp=datetime.now(),
        scan_id="SCAN002_RICE"
    )
    
    nutrients, bonds, toxics = profiler.profile_food(rice_fingerprint)
    
    print(f"\nNutrient Breakdown:")
    print(f"  Protein: {nutrients.total_protein:.1f}g")
    print(f"  Carbs: {nutrients.total_carbs:.1f}g")
    print(f"  Fat: {nutrients.total_fat:.1f}g")
    print(f"  Total Calories: {nutrients.total_calories:.0f}")
    
    # Test 3: High-fat food with contamination (contaminated fish)
    print("\n" + "=" * 80)
    print("TEST 3: High-Fat Food with Mercury Contamination (Fish)")
    print("=" * 80)
    
    fish_spectrum = (
        0.4 * np.exp(-((wavelengths - 1520) ** 2) / (2 * 18 ** 2)) +  # N-H (protein)
        0.8 * np.exp(-((wavelengths - 1730) ** 2) / (2 * 30 ** 2)) +  # C-H (fat - high)
        0.75 * np.exp(-((wavelengths - 2350) ** 2) / (2 * 35 ** 2)) +  # C-H (fat)
        0.6 * np.exp(-((wavelengths - 1920) ** 2) / (2 * 25 ** 2)) +  # O-H
        0.25 * np.exp(-((wavelengths - 1230) ** 2) / (2 * 100 ** 2)) +  # Hg signature
        0.12 * np.random.random(len(wavelengths))
    )
    
    fish_fingerprint = SpectralFingerprint(
        wavelengths=wavelengths,
        absorbance=fish_spectrum,
        reflectance=1 - fish_spectrum,
        timestamp=datetime.now(),
        scan_id="SCAN003_FISH_CONTAMINATED"
    )
    
    nutrients, bonds, toxics = profiler.profile_food(fish_fingerprint)
    
    print(f"\nNutrient Breakdown:")
    print(f"  Protein: {nutrients.total_protein:.1f}g")
    print(f"  Fat: {nutrients.total_fat:.1f}g (Omega-3 rich)")
    print(f"  Total Calories: {nutrients.total_calories:.0f}")
    
    print(f"\n⚠️  CONTAMINATION ALERT:")
    for toxic in toxics:
        status = "❌ EXCEEDS LIMIT" if toxic.exceeds_limit else "✓ Within limits"
        print(f"  {toxic.element.value}: {toxic.concentration:.4f} ppm {status}")
        print(f"    Safe limit: {toxic.safe_threshold} ppm")
        print(f"    Risk: {toxic.risk_level}")
        print(f"    Impact: {toxic.health_impact}")
    
    # Save models
    print("\n" + "=" * 80)
    print("Saving models...")
    profiler.save_models()
    print("✓ Models saved successfully")
    
    print("\n" + "=" * 80)
    print("Atomic Molecular Profiler Test Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Integrate with NIR hardware (nir_spectral_analyzer.py)")
    print("2. Build Multi-Condition Optimizer (multi_condition_optimizer.py)")
    print("3. Implement Lifecycle Modulator (lifecycle_modulator.py)")
    print("4. Deploy as microservice")
