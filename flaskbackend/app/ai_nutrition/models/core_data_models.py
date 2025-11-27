"""
AI Nutrition Analysis System - Core Data Models
Phase 1: Comprehensive data structures for nutrients, chemicals, foods, and health profiles

This module provides the foundational data models for the AI-powered nutrition analysis system.
Supports 150+ nutrients, 500+ chemicals, 10,000+ food items with complete nutritional profiles.

Author: AI Nutrition System
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from enum import Enum
from datetime import datetime, date
import numpy as np
from decimal import Decimal


# ============================================================================
# ENUMERATIONS - Classification Systems
# ============================================================================

class NutrientCategory(Enum):
    """Primary classification of nutrients"""
    MACRONUTRIENT = "macronutrient"
    VITAMIN = "vitamin"
    MINERAL = "mineral"
    AMINO_ACID = "amino_acid"
    FATTY_ACID = "fatty_acid"
    PHYTONUTRIENT = "phytonutrient"
    ANTIOXIDANT = "antioxidant"
    ENZYME = "enzyme"
    PROBIOTIC = "probiotic"
    FIBER = "fiber"
    ORGANIC_ACID = "organic_acid"


class NutrientSubcategory(Enum):
    """Detailed nutrient classification"""
    # Macronutrients
    CARBOHYDRATE = "carbohydrate"
    PROTEIN = "protein"
    FAT = "fat"
    WATER = "water"
    
    # Vitamins
    WATER_SOLUBLE_VITAMIN = "water_soluble_vitamin"
    FAT_SOLUBLE_VITAMIN = "fat_soluble_vitamin"
    VITAMIN_B_COMPLEX = "vitamin_b_complex"
    
    # Minerals
    MAJOR_MINERAL = "major_mineral"
    TRACE_MINERAL = "trace_mineral"
    ULTRA_TRACE_MINERAL = "ultra_trace_mineral"
    
    # Amino Acids
    ESSENTIAL_AMINO_ACID = "essential_amino_acid"
    NON_ESSENTIAL_AMINO_ACID = "non_essential_amino_acid"
    CONDITIONAL_AMINO_ACID = "conditional_amino_acid"
    
    # Fatty Acids
    SATURATED_FATTY_ACID = "saturated_fatty_acid"
    MONOUNSATURATED_FATTY_ACID = "monounsaturated_fatty_acid"
    POLYUNSATURATED_FATTY_ACID = "polyunsaturated_fatty_acid"
    OMEGA3_FATTY_ACID = "omega3_fatty_acid"
    OMEGA6_FATTY_ACID = "omega6_fatty_acid"
    OMEGA9_FATTY_ACID = "omega9_fatty_acid"
    TRANS_FATTY_ACID = "trans_fatty_acid"
    
    # Fiber
    SOLUBLE_FIBER = "soluble_fiber"
    INSOLUBLE_FIBER = "insoluble_fiber"
    PREBIOTIC_FIBER = "prebiotic_fiber"
    
    # Phytonutrients
    POLYPHENOL = "polyphenol"
    FLAVONOID = "flavonoid"
    CAROTENOID = "carotenoid"
    GLUCOSINOLATE = "glucosinolate"
    PHYTOESTROGEN = "phytoestrogen"
    TERPENE = "terpene"
    ALKALOID = "alkaloid"


class ChemicalCategory(Enum):
    """Classification of chemicals found in food"""
    PESTICIDE = "pesticide"
    HERBICIDE = "herbicide"
    FUNGICIDE = "fungicide"
    INSECTICIDE = "insecticide"
    HEAVY_METAL = "heavy_metal"
    FOOD_ADDITIVE = "food_additive"
    PRESERVATIVE = "preservative"
    COLORANT = "colorant"
    FLAVORING = "flavoring"
    SWEETENER = "sweetener"
    EMULSIFIER = "emulsifier"
    STABILIZER = "stabilizer"
    THICKENER = "thickener"
    ANTIBIOTIC_RESIDUE = "antibiotic_residue"
    HORMONE_RESIDUE = "hormone_residue"
    PLASTICIZER = "plasticizer"
    PACKAGING_CHEMICAL = "packaging_chemical"
    PROCESSING_BYPRODUCT = "processing_byproduct"
    ENVIRONMENTAL_CONTAMINANT = "environmental_contaminant"
    MYCOTOXIN = "mycotoxin"
    ACRYLAMIDE = "acrylamide"
    POLYCYCLIC_AROMATIC_HYDROCARBON = "polycyclic_aromatic_hydrocarbon"
    NITROSAMINE = "nitrosamine"
    DIOXIN = "dioxin"
    PCB = "pcb"


class ChemicalRiskLevel(Enum):
    """Toxicity risk classification"""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MODERATE_RISK = "moderate_risk"
    HIGH_RISK = "high_risk"
    TOXIC = "toxic"
    CARCINOGENIC = "carcinogenic"
    ENDOCRINE_DISRUPTOR = "endocrine_disruptor"
    NEUROTOXIC = "neurotoxic"
    BANNED = "banned"


class FoodCategory(Enum):
    """Primary food classification"""
    FRUIT = "fruit"
    VEGETABLE = "vegetable"
    GRAIN = "grain"
    LEGUME = "legume"
    NUT = "nut"
    SEED = "seed"
    MEAT = "meat"
    POULTRY = "poultry"
    FISH = "fish"
    SEAFOOD = "seafood"
    DAIRY = "dairy"
    EGG = "egg"
    OIL = "oil"
    HERB = "herb"
    SPICE = "spice"
    BEVERAGE = "beverage"
    SWEETENER = "sweetener"
    PROCESSED_FOOD = "processed_food"
    SUPPLEMENT = "supplement"
    CONDIMENT = "condiment"


class HealthGoal(Enum):
    """Personal health and fitness goals"""
    WEIGHT_LOSS = "weight_loss"
    WEIGHT_GAIN = "weight_gain"
    MUSCLE_GAIN = "muscle_gain"
    FAT_LOSS = "fat_loss"
    ATHLETIC_PERFORMANCE = "athletic_performance"
    ENDURANCE = "endurance"
    STRENGTH = "strength"
    RECOVERY = "recovery"
    DISEASE_PREVENTION = "disease_prevention"
    DISEASE_MANAGEMENT = "disease_management"
    IMMUNE_SUPPORT = "immune_support"
    ANTI_AGING = "anti_aging"
    LONGEVITY = "longevity"
    COGNITIVE_ENHANCEMENT = "cognitive_enhancement"
    ENERGY_BOOST = "energy_boost"
    SLEEP_IMPROVEMENT = "sleep_improvement"
    STRESS_REDUCTION = "stress_reduction"
    INFLAMMATION_REDUCTION = "inflammation_reduction"
    GUT_HEALTH = "gut_health"
    HEART_HEALTH = "heart_health"
    BONE_HEALTH = "bone_health"
    SKIN_HEALTH = "skin_health"
    FERTILITY = "fertility"
    PREGNANCY = "pregnancy"
    LACTATION = "lactation"
    GENERAL_WELLNESS = "general_wellness"


class DiseaseCategory(Enum):
    """Disease classification for condition-based recommendations"""
    AUTOIMMUNE = "autoimmune"
    CARDIOVASCULAR = "cardiovascular"
    METABOLIC = "metabolic"
    ENDOCRINE = "endocrine"
    GASTROINTESTINAL = "gastrointestinal"
    NEUROLOGICAL = "neurological"
    PSYCHIATRIC = "psychiatric"
    RENAL = "renal"
    HEPATIC = "hepatic"
    PULMONARY = "pulmonary"
    MUSCULOSKELETAL = "musculoskeletal"
    DERMATOLOGICAL = "dermatological"
    ONCOLOGICAL = "oncological"
    HEMATOLOGICAL = "hematological"
    INFECTIOUS = "infectious"
    GENETIC = "genetic"
    REPRODUCTIVE = "reproductive"
    IMMUNODEFICIENCY = "immunodeficiency"


class ActivityLevel(Enum):
    """Physical activity classification"""
    SEDENTARY = "sedentary"  # Little to no exercise
    LIGHTLY_ACTIVE = "lightly_active"  # 1-3 days/week
    MODERATELY_ACTIVE = "moderately_active"  # 3-5 days/week
    VERY_ACTIVE = "very_active"  # 6-7 days/week
    EXTREMELY_ACTIVE = "extremely_active"  # Athlete, 2x/day


class BiologicalSex(Enum):
    """Biological sex for nutritional calculations"""
    MALE = "male"
    FEMALE = "female"
    INTERSEX = "intersex"


class LifeStage(Enum):
    """Life stage for age-specific requirements"""
    INFANT_0_6_MONTHS = "infant_0_6_months"
    INFANT_7_12_MONTHS = "infant_7_12_months"
    TODDLER_1_3_YEARS = "toddler_1_3_years"
    CHILD_4_8_YEARS = "child_4_8_years"
    CHILD_9_13_YEARS = "child_9_13_years"
    ADOLESCENT_14_18_YEARS = "adolescent_14_18_years"
    ADULT_19_30_YEARS = "adult_19_30_years"
    ADULT_31_50_YEARS = "adult_31_50_years"
    ADULT_51_70_YEARS = "adult_51_70_years"
    SENIOR_71_PLUS_YEARS = "senior_71_plus_years"
    PREGNANT = "pregnant"
    LACTATING = "lactating"


class CookingMethod(Enum):
    """Cooking methods affecting nutrient content"""
    RAW = "raw"
    BOILED = "boiled"
    STEAMED = "steamed"
    BAKED = "baked"
    ROASTED = "roasted"
    GRILLED = "grilled"
    FRIED = "fried"
    DEEP_FRIED = "deep_fried"
    SAUTEED = "sauteed"
    STIR_FRIED = "stir_fried"
    MICROWAVED = "microwaved"
    PRESSURE_COOKED = "pressure_cooked"
    SLOW_COOKED = "slow_cooked"
    BLANCHED = "blanched"
    POACHED = "poached"
    SMOKED = "smoked"
    FERMENTED = "fermented"
    DRIED = "dried"
    FROZEN = "frozen"
    CANNED = "canned"


class MeasurementUnit(Enum):
    """Units for nutrient and chemical measurements"""
    # Mass
    GRAM = "g"
    MILLIGRAM = "mg"
    MICROGRAM = "μg"
    NANOGRAM = "ng"
    KILOGRAM = "kg"
    
    # Volume
    LITER = "L"
    MILLILITER = "mL"
    MICROLITER = "μL"
    
    # Energy
    KILOCALORIE = "kcal"
    KILOJOULE = "kJ"
    
    # International Units
    INTERNATIONAL_UNIT = "IU"
    
    # Percentage
    PERCENT = "%"
    
    # Concentration
    PPM = "ppm"  # Parts per million
    PPB = "ppb"  # Parts per billion
    PPT = "ppt"  # Parts per trillion
    
    # Custom food units
    SERVING = "serving"
    CUP = "cup"
    TABLESPOON = "tbsp"
    TEASPOON = "tsp"
    OUNCE = "oz"
    POUND = "lb"


# ============================================================================
# CORE NUTRIENT MODELS
# ============================================================================

@dataclass
class NutrientReference:
    """Reference Daily Intake (RDI) values for a specific nutrient"""
    nutrient_id: str
    nutrient_name: str
    category: NutrientCategory
    subcategory: NutrientSubcategory
    
    # RDA values by life stage and sex
    rda_values: Dict[Tuple[LifeStage, BiologicalSex], float] = field(default_factory=dict)
    
    # Units for measurements
    unit: MeasurementUnit = MeasurementUnit.MILLIGRAM
    
    # Upper Limit (UL) - maximum safe daily intake
    upper_limit: Dict[Tuple[LifeStage, BiologicalSex], Optional[float]] = field(default_factory=dict)
    
    # Adequate Intake (AI) - when RDA not established
    adequate_intake: Dict[Tuple[LifeStage, BiologicalSex], Optional[float]] = field(default_factory=dict)
    
    # Tolerable Upper Intake Level for specific conditions
    therapeutic_range: Optional[Tuple[float, float]] = None
    
    # Deficiency threshold
    deficiency_threshold: Optional[float] = None
    
    # Optimal range for health
    optimal_range: Optional[Tuple[float, float]] = None
    
    # Alternative names and synonyms
    synonyms: List[str] = field(default_factory=list)
    
    # Chemical formula
    chemical_formula: Optional[str] = None
    
    # Molecular weight (g/mol)
    molecular_weight: Optional[float] = None
    
    # Bioavailability factors
    bioavailability_percentage: float = 100.0  # Default 100%
    
    # Factors affecting absorption
    absorption_enhancers: List[str] = field(default_factory=list)
    absorption_inhibitors: List[str] = field(default_factory=list)
    
    # Nutrient interactions
    synergistic_nutrients: List[str] = field(default_factory=list)  # Work better together
    antagonistic_nutrients: List[str] = field(default_factory=list)  # Compete/interfere
    
    # Health functions
    primary_functions: List[str] = field(default_factory=list)
    
    # Deficiency symptoms
    deficiency_symptoms: List[str] = field(default_factory=list)
    
    # Toxicity symptoms (if excessive)
    toxicity_symptoms: List[str] = field(default_factory=list)
    
    # Food sources (top sources)
    top_food_sources: List[Tuple[str, float]] = field(default_factory=list)  # (food_name, amount_per_100g)
    
    # Stability during cooking
    heat_stable: bool = True
    water_soluble: bool = False
    light_sensitive: bool = False
    oxygen_sensitive: bool = False
    
    # Loss during cooking (percentage)
    cooking_loss_rates: Dict[CookingMethod, float] = field(default_factory=dict)
    
    # Storage stability
    storage_degradation_rate: float = 0.0  # Percentage per month at room temp
    
    # Evidence level for health benefits
    evidence_level: str = "A"  # A (strong), B (moderate), C (limited)
    
    # References
    research_references: List[str] = field(default_factory=list)  # PMID or DOI
    
    # Last updated
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class NutrientContent:
    """Actual nutrient content in a specific food item"""
    nutrient_id: str
    nutrient_name: str
    amount: float  # Amount per 100g or per serving
    unit: MeasurementUnit
    
    # Measurement basis
    per_100g: float  # Standardized per 100g
    per_serving: Optional[float] = None  # Per standard serving
    
    # Variability
    minimum_value: Optional[float] = None
    maximum_value: Optional[float] = None
    standard_deviation: Optional[float] = None
    
    # Bioavailable fraction
    bioavailable_amount: Optional[float] = None
    bioavailability_coefficient: float = 1.0
    
    # Retention after cooking
    retention_factors: Dict[CookingMethod, float] = field(default_factory=dict)
    
    # Data source
    data_source: str = "USDA"  # USDA, laboratory analysis, etc.
    confidence_level: float = 95.0  # Confidence in measurement
    
    # Measurement date
    measured_date: Optional[date] = None


@dataclass
class Nutrient:
    """Complete nutrient profile combining reference and content data"""
    nutrient_id: str
    name: str
    category: NutrientCategory
    subcategory: NutrientSubcategory
    
    # Reference values
    reference_data: NutrientReference
    
    # Computational properties
    molecular_weight: Optional[float] = None
    chemical_formula: Optional[str] = None
    
    # Metabolic properties
    absorption_rate: float = 1.0  # Fraction absorbed
    half_life_hours: Optional[float] = None  # Biological half-life
    excretion_route: List[str] = field(default_factory=list)  # Urine, feces, sweat, etc.
    
    # Storage in body
    body_stores: Optional[float] = None  # Average storage in body (mg or μg)
    storage_sites: List[str] = field(default_factory=list)  # Liver, bones, fat, etc.
    
    # Daily turnover rate
    daily_turnover_percentage: Optional[float] = None
    
    # Critical threshold
    critical_deficiency_level: Optional[float] = None
    critical_toxicity_level: Optional[float] = None
    
    # Genetic factors
    genetic_variants_affecting: List[str] = field(default_factory=list)  # SNPs
    
    # Medication interactions
    medication_interactions: List[Dict[str, str]] = field(default_factory=list)
    
    # Special populations
    pregnancy_requirement_multiplier: float = 1.0
    lactation_requirement_multiplier: float = 1.0
    elderly_requirement_multiplier: float = 1.0
    athlete_requirement_multiplier: float = 1.0
    
    # Disease-specific requirements
    disease_modifiers: Dict[str, float] = field(default_factory=dict)  # disease_id: multiplier
    
    # Testing methods
    lab_test_marker: Optional[str] = None  # Serum, plasma, RBC marker
    optimal_lab_range: Optional[Tuple[float, float]] = None
    
    # Supplementation
    supplement_forms: List[str] = field(default_factory=list)
    best_absorbed_form: Optional[str] = None
    
    # AI metadata
    embedding_vector: Optional[np.ndarray] = None  # For ML similarity matching
    tags: Set[str] = field(default_factory=set)


# ============================================================================
# CHEMICAL SAFETY MODELS
# ============================================================================

@dataclass
class ChemicalCompound:
    """Chemical compound found in food (potentially harmful)"""
    chemical_id: str
    name: str
    common_names: List[str] = field(default_factory=list)
    
    # Classification
    category: ChemicalCategory
    subcategory: Optional[str] = None
    
    # Chemical properties
    cas_number: Optional[str] = None  # Chemical Abstracts Service number
    chemical_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    
    # Risk assessment
    risk_level: ChemicalRiskLevel = ChemicalRiskLevel.LOW_RISK
    
    # Regulatory status
    fda_approved: bool = False
    eu_approved: bool = False
    who_approved: bool = False
    banned_countries: List[str] = field(default_factory=list)
    
    # Safety thresholds
    adi: Optional[float] = None  # Acceptable Daily Intake (mg/kg body weight/day)
    adi_unit: MeasurementUnit = MeasurementUnit.MILLIGRAM
    
    mrl: Optional[float] = None  # Maximum Residue Limit (mg/kg food)
    mrl_unit: MeasurementUnit = MeasurementUnit.MILLIGRAM
    
    tdi: Optional[float] = None  # Tolerable Daily Intake
    ptdi: Optional[float] = None  # Provisional Tolerable Daily Intake
    ptwi: Optional[float] = None  # Provisional Tolerable Weekly Intake
    
    # Toxicity data
    ld50_oral: Optional[float] = None  # Lethal dose 50% (mg/kg body weight)
    ld50_dermal: Optional[float] = None
    ld50_inhalation: Optional[float] = None
    
    noael: Optional[float] = None  # No Observed Adverse Effect Level
    loael: Optional[float] = None  # Lowest Observed Adverse Effect Level
    
    # Health effects
    acute_effects: List[str] = field(default_factory=list)
    chronic_effects: List[str] = field(default_factory=list)
    
    carcinogenicity: Optional[str] = None  # IARC classification
    mutagenicity: bool = False
    teratogenicity: bool = False
    reproductive_toxicity: bool = False
    endocrine_disruption: bool = False
    neurotoxicity: bool = False
    immunotoxicity: bool = False
    
    # Target organs
    target_organs: List[str] = field(default_factory=list)
    
    # Vulnerable populations
    high_risk_groups: List[str] = field(default_factory=list)
    
    # Metabolism
    metabolic_pathway: Optional[str] = None
    metabolites: List[str] = field(default_factory=list)
    half_life_human: Optional[float] = None  # Hours or days
    
    # Accumulation
    bioaccumulation: bool = False
    bioconcentration_factor: Optional[float] = None
    
    # Found in foods
    common_food_sources: List[str] = field(default_factory=list)
    typical_concentration_range: Optional[Tuple[float, float]] = None  # (min, max) in ppm or ppb
    
    # Exposure routes
    ingestion_risk: bool = True
    dermal_risk: bool = False
    inhalation_risk: bool = False
    
    # Synergistic effects
    synergistic_chemicals: List[str] = field(default_factory=list)
    
    # Detection methods
    detection_methods: List[str] = field(default_factory=list)
    detection_limit: Optional[float] = None
    
    # Mitigation
    removal_methods: List[str] = field(default_factory=list)
    reduction_cooking: Dict[CookingMethod, float] = field(default_factory=dict)
    
    # References
    regulatory_references: List[str] = field(default_factory=list)
    research_studies: List[str] = field(default_factory=list)
    
    # Last updated
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ChemicalContent:
    """Chemical content detected in a specific food item"""
    chemical_id: str
    chemical_name: str
    food_item_id: str
    
    # Measured concentration
    concentration: float
    unit: MeasurementUnit = MeasurementUnit.PPM
    
    # Measurement details
    detection_date: Optional[date] = None
    detection_method: Optional[str] = None
    laboratory: Optional[str] = None
    
    # Variability
    min_concentration: Optional[float] = None
    max_concentration: Optional[float] = None
    samples_tested: int = 1
    
    # Risk assessment for this specific instance
    exceeds_mrl: bool = False
    exceeds_adi: bool = False
    risk_score: float = 0.0  # 0-100 scale
    
    # Origin tracking
    origin_country: Optional[str] = None
    production_method: Optional[str] = None  # Organic, conventional, etc.
    
    # Seasonal variation
    seasonal_peak: Optional[str] = None
    
    # Confidence
    confidence_level: float = 95.0


# ============================================================================
# FOOD ITEM MODELS
# ============================================================================

@dataclass
class FoodItem:
    """Complete food item with all nutritional and chemical data"""
    food_id: str
    name: str
    common_names: List[str] = field(default_factory=list)
    
    # Classification
    category: FoodCategory
    subcategory: Optional[str] = None
    food_group: Optional[str] = None  # USDA food group
    
    # Identification
    usda_fdc_id: Optional[str] = None  # USDA FoodData Central ID
    upc_code: Optional[str] = None  # Universal Product Code
    
    # Physical properties
    density: Optional[float] = None  # g/mL for volume conversion
    water_content_percentage: float = 0.0
    
    # Serving information
    standard_serving_size: float = 100.0  # grams
    serving_size_description: str = "100g"
    alternative_serving_sizes: Dict[str, float] = field(default_factory=dict)
    
    # Nutritional content (per 100g)
    nutrients: List[NutrientContent] = field(default_factory=list)
    
    # Complete macronutrient breakdown
    calories: float = 0.0  # kcal per 100g
    protein_g: float = 0.0
    carbohydrate_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0
    sugar_g: float = 0.0
    saturated_fat_g: float = 0.0
    trans_fat_g: float = 0.0
    cholesterol_mg: float = 0.0
    sodium_mg: float = 0.0
    
    # Chemical contaminants
    chemicals: List[ChemicalContent] = field(default_factory=list)
    
    # Food quality indicators
    organic: bool = False
    non_gmo: bool = False
    grass_fed: bool = False
    wild_caught: bool = False
    free_range: bool = False
    pesticide_free: bool = False
    
    # Production information
    origin_countries: List[str] = field(default_factory=list)
    production_methods: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    
    # Allergen information
    allergens: List[str] = field(default_factory=list)
    
    # Glycemic properties
    glycemic_index: Optional[float] = None
    glycemic_load: Optional[float] = None
    
    # Inflammatory properties
    inflammatory_index: Optional[float] = None  # Dietary Inflammatory Index
    
    # Antioxidant capacity
    orac_value: Optional[float] = None  # Oxygen Radical Absorbance Capacity
    
    # pH and acidity
    ph_value: Optional[float] = None
    acidic_or_alkaline: Optional[str] = None
    
    # Cooking and preparation
    edible_portion_percentage: float = 100.0
    cooking_methods_compatible: List[CookingMethod] = field(default_factory=list)
    nutrient_retention_rates: Dict[CookingMethod, Dict[str, float]] = field(default_factory=dict)
    
    # Storage
    shelf_life_days: Optional[int] = None
    storage_conditions: List[str] = field(default_factory=list)
    
    # Seasonality
    seasonal_availability: List[str] = field(default_factory=list)
    peak_season: Optional[str] = None
    
    # Cost
    average_price_per_kg: Optional[float] = None
    price_currency: str = "USD"
    
    # Sustainability
    carbon_footprint_kg_co2: Optional[float] = None
    water_footprint_liters: Optional[float] = None
    
    # Health scores
    nutrient_density_score: Optional[float] = None
    health_star_rating: Optional[float] = None
    
    # Tags for search and ML
    tags: Set[str] = field(default_factory=set)
    search_keywords: List[str] = field(default_factory=list)
    
    # AI features
    embedding_vector: Optional[np.ndarray] = None
    similar_foods: List[str] = field(default_factory=list)
    
    # Data provenance
    data_sources: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    verified: bool = False


@dataclass
class Recipe:
    """Recipe with complete nutritional breakdown"""
    recipe_id: str
    name: str
    description: str
    
    # Ingredients
    ingredients: List[Tuple[str, float, MeasurementUnit]] = field(default_factory=list)  # (food_id, amount, unit)
    
    # Preparation
    cooking_methods: List[CookingMethod] = field(default_factory=list)
    preparation_time_minutes: int = 0
    cooking_time_minutes: int = 0
    total_time_minutes: int = 0
    
    # Servings
    servings: int = 1
    serving_size: str = "1 serving"
    
    # Nutritional content (calculated)
    total_nutrients: List[NutrientContent] = field(default_factory=list)
    nutrients_per_serving: List[NutrientContent] = field(default_factory=list)
    
    # Chemical content (calculated)
    total_chemicals: List[ChemicalContent] = field(default_factory=list)
    chemicals_per_serving: List[ChemicalContent] = field(default_factory=list)
    
    # Macros per serving
    calories_per_serving: float = 0.0
    protein_per_serving: float = 0.0
    carbs_per_serving: float = 0.0
    fat_per_serving: float = 0.0
    
    # Health properties
    health_score: Optional[float] = None
    suitable_for_conditions: List[str] = field(default_factory=list)
    contraindicated_conditions: List[str] = field(default_factory=list)
    
    # Tags
    cuisine_type: Optional[str] = None
    meal_type: List[str] = field(default_factory=list)  # Breakfast, lunch, dinner, snack
    dietary_labels: List[str] = field(default_factory=list)  # Vegan, keto, paleo, etc.
    
    # Instructions
    instructions: List[str] = field(default_factory=list)
    
    # Cost and sustainability
    estimated_cost: Optional[float] = None
    carbon_footprint: Optional[float] = None


# ============================================================================
# USER PROFILE MODELS
# ============================================================================

@dataclass
class UserProfile:
    """Complete user profile for personalized nutrition"""
    user_id: str
    created_date: datetime = field(default_factory=datetime.now)
    
    # Basic demographics
    age: int = 30
    biological_sex: BiologicalSex = BiologicalSex.MALE
    life_stage: LifeStage = LifeStage.ADULT_19_30_YEARS
    
    # Physical measurements
    height_cm: float = 170.0
    weight_kg: float = 70.0
    bmi: float = 24.2
    body_fat_percentage: Optional[float] = None
    lean_mass_kg: Optional[float] = None
    
    # Activity and lifestyle
    activity_level: ActivityLevel = ActivityLevel.MODERATELY_ACTIVE
    exercise_frequency_per_week: int = 3
    exercise_types: List[str] = field(default_factory=list)
    occupation_activity_level: ActivityLevel = ActivityLevel.SEDENTARY
    
    # Metabolic measurements
    bmr: Optional[float] = None  # Basal Metabolic Rate (kcal/day)
    tdee: Optional[float] = None  # Total Daily Energy Expenditure (kcal/day)
    
    # Health goals
    primary_goals: List[HealthGoal] = field(default_factory=list)
    target_weight_kg: Optional[float] = None
    target_body_fat_percentage: Optional[float] = None
    weight_goal_rate_kg_per_week: Optional[float] = None
    
    # Medical conditions
    diagnosed_conditions: List[str] = field(default_factory=list)  # disease_ids
    condition_severity: Dict[str, str] = field(default_factory=dict)
    condition_diagnosed_dates: Dict[str, date] = field(default_factory=dict)
    
    # Medications
    current_medications: List[Dict[str, Any]] = field(default_factory=list)
    medication_start_dates: Dict[str, date] = field(default_factory=dict)
    
    # Allergies and intolerances
    food_allergies: List[str] = field(default_factory=list)
    food_intolerances: List[str] = field(default_factory=list)
    medication_allergies: List[str] = field(default_factory=list)
    
    # Dietary preferences and restrictions
    dietary_pattern: Optional[str] = None  # Vegan, vegetarian, keto, paleo, etc.
    foods_to_avoid: List[str] = field(default_factory=list)
    foods_to_include: List[str] = field(default_factory=list)
    cultural_dietary_restrictions: List[str] = field(default_factory=list)
    
    # Genetic information
    genetic_variants: Dict[str, str] = field(default_factory=dict)  # SNP: genotype
    nutrient_metabolism_genes: Dict[str, str] = field(default_factory=dict)
    
    # Lab results
    recent_blood_work: Dict[str, Tuple[float, date]] = field(default_factory=dict)
    nutrient_deficiencies: List[str] = field(default_factory=list)
    nutrient_excesses: List[str] = field(default_factory=list)
    
    # Pregnancy/Lactation status
    pregnant: bool = False
    pregnancy_trimester: Optional[int] = None
    lactating: bool = False
    lactation_months: Optional[int] = None
    
    # Personalized requirements (calculated)
    personalized_rda: Dict[str, float] = field(default_factory=dict)  # nutrient_id: daily_requirement
    personalized_upper_limits: Dict[str, float] = field(default_factory=dict)
    calorie_target: Optional[float] = None
    protein_target_g: Optional[float] = None
    carb_target_g: Optional[float] = None
    fat_target_g: Optional[float] = None
    
    # Risk factors
    chemical_sensitivities: List[str] = field(default_factory=list)
    environmental_exposures: List[str] = field(default_factory=list)
    occupation_hazards: List[str] = field(default_factory=list)
    
    # Preferences for recommendations
    preferred_cuisines: List[str] = field(default_factory=list)
    disliked_foods: List[str] = field(default_factory=list)
    cooking_skill_level: str = "intermediate"
    time_available_for_cooking: int = 30  # minutes
    budget_per_day: Optional[float] = None
    
    # Behavioral data
    typical_meal_times: List[str] = field(default_factory=list)
    snacking_frequency: int = 2
    eating_out_frequency: int = 3  # times per week
    
    # Compliance tracking
    adherence_score: Optional[float] = None
    barriers_to_compliance: List[str] = field(default_factory=list)
    
    # Last updated
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class DailyIntakeLog:
    """Log of daily food intake for tracking"""
    log_id: str
    user_id: str
    date: date
    
    # Meals consumed
    meals: List[Dict[str, Any]] = field(default_factory=list)  # meal_time, food_items, amounts
    
    # Total daily intake
    total_nutrients: Dict[str, float] = field(default_factory=dict)  # nutrient_id: amount
    total_chemicals: Dict[str, float] = field(default_factory=dict)  # chemical_id: amount
    
    # Macros
    total_calories: float = 0.0
    total_protein: float = 0.0
    total_carbs: float = 0.0
    total_fat: float = 0.0
    
    # RDA percentages
    rda_percentages: Dict[str, float] = field(default_factory=dict)  # nutrient_id: percentage
    
    # Warnings
    deficient_nutrients: List[str] = field(default_factory=list)
    excessive_nutrients: List[str] = field(default_factory=list)
    risky_chemicals: List[str] = field(default_factory=list)
    
    # Health score for the day
    daily_health_score: Optional[float] = None
    nutrient_balance_score: Optional[float] = None
    
    # Notes
    user_notes: Optional[str] = None
    symptoms_reported: List[str] = field(default_factory=list)
    energy_level: Optional[int] = None  # 1-10 scale
    
    # Compliance
    met_goals: List[str] = field(default_factory=list)
    missed_goals: List[str] = field(default_factory=list)
    
    # Created timestamp
    created_at: datetime = field(default_factory=datetime.now)


# ============================================================================
# MEDICAL CONDITION MODELS
# ============================================================================

@dataclass
class MedicalCondition:
    """Medical condition with specific nutritional requirements"""
    condition_id: str
    name: str
    common_names: List[str] = field(default_factory=list)
    
    # Classification
    category: DiseaseCategory
    icd10_code: Optional[str] = None
    
    # Prevalence
    prevalence: Optional[str] = None
    incidence: Optional[str] = None
    
    # Nutritional modifications
    required_nutrients: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # nutrient_id: (min, max)
    beneficial_nutrients: List[str] = field(default_factory=list)
    harmful_nutrients: List[str] = field(default_factory=list)
    
    # Specific nutrient multipliers
    nutrient_requirement_modifiers: Dict[str, float] = field(default_factory=dict)  # nutrient_id: multiplier
    
    # Foods to emphasize
    recommended_foods: List[str] = field(default_factory=list)
    foods_to_avoid: List[str] = field(default_factory=list)
    foods_to_limit: List[str] = field(default_factory=list)
    
    # Chemical sensitivities
    chemical_contraindications: List[str] = field(default_factory=list)
    chemical_tolerances: Dict[str, float] = field(default_factory=dict)  # Lower than normal ADI
    
    # Dietary patterns
    recommended_dietary_patterns: List[str] = field(default_factory=list)
    
    # Supplementation
    recommended_supplements: List[Tuple[str, float, str]] = field(default_factory=list)  # (supplement, dose, reason)
    
    # Drug-nutrient interactions
    common_medications: List[str] = field(default_factory=list)
    medication_nutrient_depletions: Dict[str, List[str]] = field(default_factory=dict)
    
    # Monitoring biomarkers
    key_biomarkers: List[str] = field(default_factory=list)
    target_biomarker_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Complications affected by nutrition
    nutrition_sensitive_complications: List[str] = field(default_factory=list)
    
    # Evidence
    dietary_intervention_evidence: List[str] = field(default_factory=list)  # PMIDs
    clinical_guidelines: List[str] = field(default_factory=list)
    
    # Special considerations
    special_populations_notes: Dict[str, str] = field(default_factory=dict)
    
    # Related conditions
    comorbidities: List[str] = field(default_factory=list)
    
    # Last updated
    last_updated: datetime = field(default_factory=datetime.now)


# ============================================================================
# ANALYSIS AND RECOMMENDATION MODELS
# ============================================================================

@dataclass
class NutrientAnalysis:
    """Analysis of nutrient intake vs requirements"""
    user_id: str
    analysis_date: datetime = field(default_factory=datetime.now)
    
    # Period analyzed
    period_days: int = 1
    start_date: date = field(default_factory=date.today)
    end_date: date = field(default_factory=date.today)
    
    # Nutrient status
    nutrient_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Structure: {
    #   nutrient_id: {
    #     'intake': float,
    #     'rda': float,
    #     'percentage': float,
    #     'status': str (deficient/adequate/excessive),
    #     'trend': str (improving/stable/declining)
    #   }
    # }
    
    # Summary statistics
    total_nutrients_tracked: int = 0
    adequate_nutrients: int = 0
    deficient_nutrients_count: int = 0
    excessive_nutrients_count: int = 0
    
    # Critical findings
    severe_deficiencies: List[str] = field(default_factory=list)
    dangerous_excesses: List[str] = field(default_factory=list)
    
    # Nutrient balance scores
    overall_nutrient_score: float = 0.0  # 0-100
    micronutrient_score: float = 0.0
    macronutrient_score: float = 0.0
    
    # Specific category scores
    vitamin_score: float = 0.0
    mineral_score: float = 0.0
    antioxidant_score: float = 0.0
    omega_balance_score: float = 0.0
    
    # Chemical safety
    chemical_exposure_score: float = 0.0  # 0-100, higher is safer
    chemicals_of_concern: List[str] = field(default_factory=list)
    
    # Health impacts
    predicted_health_impacts: List[str] = field(default_factory=list)
    condition_specific_concerns: Dict[str, List[str]] = field(default_factory=dict)
    
    # Recommendations generated
    priority_recommendations: List[str] = field(default_factory=list)
    
    # Confidence in analysis
    data_completeness: float = 0.0  # Percentage of meals logged
    confidence_score: float = 0.0


@dataclass
class ChemicalRiskAssessment:
    """Assessment of chemical exposure risks"""
    user_id: str
    assessment_date: datetime = field(default_factory=datetime.now)
    
    # Period analyzed
    period_days: int = 1
    
    # Chemical exposures
    chemical_exposures: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Structure: {
    #   chemical_id: {
    #     'total_exposure': float,
    #     'adi': float,
    #     'percentage_adi': float,
    #     'risk_level': str,
    #     'sources': List[str]
    #   }
    # }
    
    # Risk summary
    total_chemicals_detected: int = 0
    safe_chemicals: int = 0
    concerning_chemicals: int = 0
    dangerous_chemicals: int = 0
    
    # Overall risk score
    overall_chemical_risk_score: float = 0.0  # 0-100, lower is riskier
    
    # By category
    pesticide_risk_score: float = 0.0
    heavy_metal_risk_score: float = 0.0
    additive_risk_score: float = 0.0
    contaminant_risk_score: float = 0.0
    
    # Specific warnings
    immediate_concerns: List[Dict[str, Any]] = field(default_factory=list)
    long_term_concerns: List[Dict[str, Any]] = field(default_factory=list)
    
    # Vulnerable populations
    pregnancy_risk_level: Optional[str] = None
    child_risk_level: Optional[str] = None
    condition_specific_risks: Dict[str, str] = field(default_factory=dict)
    
    # Cumulative effects
    synergistic_combinations: List[Tuple[str, str, str]] = field(default_factory=list)  # (chem1, chem2, effect)
    
    # Recommendations
    mitigation_strategies: List[str] = field(default_factory=list)
    foods_to_substitute: Dict[str, List[str]] = field(default_factory=dict)  # risky_food: safer_alternatives


@dataclass
class PersonalizedRecommendation:
    """Personalized nutrition recommendation"""
    recommendation_id: str
    user_id: str
    generated_date: datetime = field(default_factory=datetime.now)
    
    # Priority level
    priority: str = "medium"  # critical, high, medium, low
    
    # Type of recommendation
    recommendation_type: str = "increase_nutrient"  # increase_nutrient, decrease_nutrient, avoid_food, add_food, etc.
    
    # Target
    target_nutrient: Optional[str] = None
    target_food: Optional[str] = None
    target_chemical: Optional[str] = None
    
    # Current state
    current_value: Optional[float] = None
    target_value: Optional[float] = None
    gap: Optional[float] = None
    
    # Reasoning
    reason: str = ""
    health_impact: str = ""
    evidence_level: str = "B"
    
    # Actionable advice
    specific_actions: List[str] = field(default_factory=list)
    food_suggestions: List[Tuple[str, float]] = field(default_factory=list)  # (food, amount_per_day)
    supplement_suggestions: List[Tuple[str, float, str]] = field(default_factory=list)  # (supplement, dose, timing)
    
    # Expected outcomes
    expected_improvement_days: Optional[int] = None
    expected_outcome: str = ""
    
    # Tracking
    is_active: bool = True
    user_accepted: Optional[bool] = None
    compliance_rate: Optional[float] = None
    
    # Related to conditions
    related_conditions: List[str] = field(default_factory=list)
    related_goals: List[HealthGoal] = field(default_factory=list)


@dataclass
class MealRecommendation:
    """Specific meal or recipe recommendation"""
    recommendation_id: str
    user_id: str
    meal_type: str  # breakfast, lunch, dinner, snack
    
    # Recommended recipe or foods
    recipe_id: Optional[str] = None
    food_items: List[Tuple[str, float]] = field(default_factory=list)  # (food_id, amount_g)
    
    # Nutritional targets met
    nutrients_provided: Dict[str, float] = field(default_factory=dict)
    percentage_of_daily_targets: Dict[str, float] = field(default_factory=dict)
    
    # Why recommended
    recommendation_reasons: List[str] = field(default_factory=list)
    goals_addressed: List[HealthGoal] = field(default_factory=list)
    deficiencies_addressed: List[str] = field(default_factory=list)
    
    # Suitability score
    suitability_score: float = 0.0  # 0-100
    taste_preference_score: float = 0.0
    convenience_score: float = 0.0
    cost_score: float = 0.0
    
    # Preparation
    preparation_time: int = 30
    difficulty: str = "easy"
    
    # Alternatives
    alternative_meal_ids: List[str] = field(default_factory=list)


# ============================================================================
# COMPUTER VISION MODELS
# ============================================================================

@dataclass
class ImageAnalysisResult:
    """Result from food image analysis"""
    analysis_id: str
    user_id: str
    image_path: str
    analyzed_at: datetime = field(default_factory=datetime.now)
    
    # Detected foods
    detected_foods: List[Dict[str, Any]] = field(default_factory=list)
    # Structure: [{
    #   'food_id': str,
    #   'name': str,
    #   'confidence': float,
    #   'bounding_box': Tuple[int, int, int, int],
    #   'estimated_portion_g': float,
    #   'portion_confidence': float
    # }]
    
    # Portion estimation
    total_estimated_weight: float = 0.0
    portion_estimation_method: str = "depth_estimation"
    
    # Reference objects detected
    reference_objects: List[Dict[str, Any]] = field(default_factory=list)  # Plate, utensils for scale
    
    # Nutritional estimate
    estimated_nutrients: Dict[str, float] = field(default_factory=dict)
    estimated_calories: float = 0.0
    
    # Model performance
    detection_confidence: float = 0.0
    model_version: str = "1.0"
    
    # User corrections
    user_corrected: bool = False
    corrections: List[Dict[str, Any]] = field(default_factory=list)
    
    # Nutrition label OCR (if applicable)
    nutrition_label_detected: bool = False
    ocr_extracted_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NutritionLabelData:
    """Parsed nutrition label data from OCR"""
    label_id: str
    image_path: str
    
    # Serving information
    serving_size: Optional[str] = None
    servings_per_container: Optional[float] = None
    
    # Macronutrients per serving
    calories: Optional[float] = None
    total_fat_g: Optional[float] = None
    saturated_fat_g: Optional[float] = None
    trans_fat_g: Optional[float] = None
    cholesterol_mg: Optional[float] = None
    sodium_mg: Optional[float] = None
    total_carbohydrate_g: Optional[float] = None
    dietary_fiber_g: Optional[float] = None
    total_sugars_g: Optional[float] = None
    added_sugars_g: Optional[float] = None
    protein_g: Optional[float] = None
    
    # Micronutrients (% daily value)
    vitamin_d_percent: Optional[float] = None
    calcium_percent: Optional[float] = None
    iron_percent: Optional[float] = None
    potassium_percent: Optional[float] = None
    
    # Additional nutrients
    other_nutrients: Dict[str, float] = field(default_factory=dict)
    
    # Ingredients list
    ingredients: List[str] = field(default_factory=list)
    
    # Allergen information
    contains_allergens: List[str] = field(default_factory=list)
    may_contain_allergens: List[str] = field(default_factory=list)
    
    # OCR confidence
    ocr_confidence: float = 0.0
    needs_verification: bool = False


# ============================================================================
# GOAL TRACKING MODELS
# ============================================================================

@dataclass
class GoalProgress:
    """Track progress toward health goals"""
    goal_id: str
    user_id: str
    goal_type: HealthGoal
    
    # Goal definition
    goal_description: str
    target_metric: str
    target_value: float
    current_value: float
    starting_value: float
    
    # Timeline
    start_date: date
    target_date: date
    current_date: date = field(default_factory=date.today)
    
    # Progress metrics
    progress_percentage: float = 0.0
    days_elapsed: int = 0
    days_remaining: int = 0
    on_track: bool = True
    
    # Rate of change
    daily_change_rate: Optional[float] = None
    weekly_change_rate: Optional[float] = None
    required_daily_rate: Optional[float] = None
    
    # Supporting metrics
    compliance_rate: float = 0.0  # Percentage of days followed recommendations
    nutrient_adherence: Dict[str, float] = field(default_factory=dict)
    
    # Milestones
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    milestones_achieved: int = 0
    
    # Barriers and facilitators
    identified_barriers: List[str] = field(default_factory=list)
    success_factors: List[str] = field(default_factory=list)
    
    # Adjustments needed
    needs_adjustment: bool = False
    recommended_adjustments: List[str] = field(default_factory=list)
    
    # Outcome prediction
    predicted_outcome: Optional[float] = None
    prediction_confidence: Optional[float] = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Calculate Body Mass Index"""
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)


def calculate_bmr(weight_kg: float, height_cm: float, age: int, sex: BiologicalSex) -> float:
    """Calculate Basal Metabolic Rate using Mifflin-St Jeor equation"""
    if sex == BiologicalSex.MALE:
        return (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:  # Female
        return (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161


def calculate_tdee(bmr: float, activity_level: ActivityLevel) -> float:
    """Calculate Total Daily Energy Expenditure"""
    multipliers = {
        ActivityLevel.SEDENTARY: 1.2,
        ActivityLevel.LIGHTLY_ACTIVE: 1.375,
        ActivityLevel.MODERATELY_ACTIVE: 1.55,
        ActivityLevel.VERY_ACTIVE: 1.725,
        ActivityLevel.EXTREMELY_ACTIVE: 1.9
    }
    return bmr * multipliers.get(activity_level, 1.55)


def determine_life_stage(age: int, pregnant: bool = False, lactating: bool = False) -> LifeStage:
    """Determine life stage based on age and reproductive status"""
    if pregnant:
        return LifeStage.PREGNANT
    if lactating:
        return LifeStage.LACTATING
    
    if age < 0.5:
        return LifeStage.INFANT_0_6_MONTHS
    elif age < 1:
        return LifeStage.INFANT_7_12_MONTHS
    elif age < 4:
        return LifeStage.TODDLER_1_3_YEARS
    elif age < 9:
        return LifeStage.CHILD_4_8_YEARS
    elif age < 14:
        return LifeStage.CHILD_9_13_YEARS
    elif age < 19:
        return LifeStage.ADOLESCENT_14_18_YEARS
    elif age < 31:
        return LifeStage.ADULT_19_30_YEARS
    elif age < 51:
        return LifeStage.ADULT_31_50_YEARS
    elif age < 71:
        return LifeStage.ADULT_51_70_YEARS
    else:
        return LifeStage.SENIOR_71_PLUS_YEARS


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_nutrient_intake(intake: float, rda: float, upper_limit: Optional[float] = None) -> str:
    """Validate nutrient intake against RDA and UL"""
    if intake < rda * 0.67:
        return "severely_deficient"
    elif intake < rda:
        return "deficient"
    elif upper_limit and intake > upper_limit:
        return "excessive"
    elif upper_limit and intake > upper_limit * 0.8:
        return "approaching_ul"
    else:
        return "adequate"


def calculate_chemical_risk_score(exposure: float, adi: float, risk_level: ChemicalRiskLevel) -> float:
    """Calculate risk score for chemical exposure (0-100, higher is riskier)"""
    percentage_adi = (exposure / adi) * 100 if adi > 0 else 0
    
    base_score = min(percentage_adi, 100)
    
    # Risk level multipliers
    multipliers = {
        ChemicalRiskLevel.SAFE: 0.1,
        ChemicalRiskLevel.LOW_RISK: 0.5,
        ChemicalRiskLevel.MODERATE_RISK: 1.0,
        ChemicalRiskLevel.HIGH_RISK: 2.0,
        ChemicalRiskLevel.TOXIC: 5.0,
        ChemicalRiskLevel.CARCINOGENIC: 10.0,
        ChemicalRiskLevel.ENDOCRINE_DISRUPTOR: 8.0,
        ChemicalRiskLevel.NEUROTOXIC: 7.0,
        ChemicalRiskLevel.BANNED: 100.0
    }
    
    return min(base_score * multipliers.get(risk_level, 1.0), 100.0)


if __name__ == "__main__":
    print("AI Nutrition Analysis System - Core Data Models Loaded")
    print(f"Total Nutrient Categories: {len(NutrientCategory)}")
    print(f"Total Chemical Categories: {len(ChemicalCategory)}")
    print(f"Total Food Categories: {len(FoodCategory)}")
    print(f"Total Health Goals: {len(HealthGoal)}")
    print("System ready for Phase 2 implementation")
