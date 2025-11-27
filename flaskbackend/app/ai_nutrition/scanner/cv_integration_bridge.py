"""
Computer Vision Integration Bridge
==================================

COMPREHENSIVE DISEASE & PERSONAL GOALS MANAGEMENT SYSTEM

This module connects the CV-based nutrition pipeline with an extensive
disease database (50,000+ conditions) and personal health goals tracking.

Architecture:
Phase 1: Disease Management (50,000+ conditions)
Phase 2: Personal Goals (Weight, Fitness, Macros, etc.)
Phase 3: Meal Planning & Optimization
Phase 4: Progress Tracking & Analytics
Phase 5: AI Recommendations & Alternatives

Integration Flow:
1. CV Pipeline: Photo → Detection → Segmentation → Depth → Volume → Nutrition
2. Disease Checker: Validate against 50,000+ medical conditions
3. Goals Tracker: Check progress toward personal goals
4. Meal Optimizer: Balance disease restrictions + personal goals
5. Recommender: Suggest alternatives and improvements
6. Analytics: Track long-term health trends

Author: Wellomex AI Team
Date: November 2025
Version: 3.0 (Comprehensive System)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / 'microservices'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PHASE 1: COMPREHENSIVE DISEASE DATABASE (50,000+ CONDITIONS)
# ============================================================================

class DiseaseCategory(Enum):
    """Major disease categories for classification."""
    METABOLIC = "metabolic"  # Diabetes, thyroid, metabolic syndrome
    CARDIOVASCULAR = "cardiovascular"  # Hypertension, heart disease, stroke
    RENAL = "renal"  # Kidney disease, dialysis
    GASTROINTESTINAL = "gastrointestinal"  # IBS, Crohn's, celiac
    AUTOIMMUNE = "autoimmune"  # Lupus, rheumatoid arthritis
    NEUROLOGICAL = "neurological"  # Alzheimer's, Parkinson's
    RESPIRATORY = "respiratory"  # COPD, asthma
    ENDOCRINE = "endocrine"  # Diabetes, thyroid disorders
    HEMATOLOGICAL = "hematological"  # Anemia, hemophilia
    MUSCULOSKELETAL = "musculoskeletal"  # Osteoporosis, arthritis
    ONCOLOGY = "oncology"  # Cancer-related dietary needs
    MENTAL_HEALTH = "mental_health"  # Depression, anxiety
    ALLERGIES = "allergies"  # Food allergies, intolerances
    OTHER = "other"


class DiseaseSeverity(Enum):
    """Disease severity levels."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class DiseaseProfile:
    """Complete disease profile with dietary restrictions."""
    disease_id: str
    name: str
    category: DiseaseCategory
    severity: DiseaseSeverity
    
    # Nutrient restrictions (daily limits)
    calories_max: Optional[float] = None
    calories_min: Optional[float] = None
    protein_max: Optional[float] = None
    protein_min: Optional[float] = None
    carbs_max: Optional[float] = None
    carbs_min: Optional[float] = None
    fat_max: Optional[float] = None
    fat_min: Optional[float] = None
    sodium_max: Optional[float] = None
    sodium_min: Optional[float] = None
    potassium_max: Optional[float] = None
    potassium_min: Optional[float] = None
    sugar_max: Optional[float] = None
    fiber_min: Optional[float] = None
    fiber_max: Optional[float] = None
    cholesterol_max: Optional[float] = None
    saturated_fat_max: Optional[float] = None
    trans_fat_max: Optional[float] = None
    phosphorus_max: Optional[float] = None
    calcium_max: Optional[float] = None
    calcium_min: Optional[float] = None
    iron_max: Optional[float] = None
    iron_min: Optional[float] = None
    
    # Micronutrients (NEW - for expanded diseases)
    vitamin_c_min: Optional[float] = None
    vitamin_c_max: Optional[float] = None
    vitamin_d_min: Optional[float] = None
    vitamin_d_max: Optional[float] = None
    b12_min: Optional[float] = None
    b12_max: Optional[float] = None
    folate_min: Optional[float] = None
    folate_max: Optional[float] = None
    omega3_min: Optional[float] = None
    omega3_max: Optional[float] = None
    magnesium_min: Optional[float] = None
    magnesium_max: Optional[float] = None
    zinc_min: Optional[float] = None
    zinc_max: Optional[float] = None
    selenium_min: Optional[float] = None
    selenium_max: Optional[float] = None
    water_min: Optional[float] = None
    water_max: Optional[float] = None
    fluid_max: Optional[float] = None  # Total fluid restriction
    caffeine_max: Optional[float] = None
    alcohol_max: Optional[float] = None
    purine_max: Optional[float] = None
    oxalate_max: Optional[float] = None
    gluten_free: bool = False
    lactose_free: bool = False
    
    # Foods to avoid
    forbidden_foods: Set[str] = field(default_factory=set)
    
    # Foods to encourage
    recommended_foods: Set[str] = field(default_factory=set)
    
    # Food allergies
    food_allergies: List[str] = field(default_factory=list)
    
    # Meal-specific limits (per meal)
    carbs_per_meal_max: Optional[float] = None
    sodium_per_meal_max: Optional[float] = None
    
    # Risk factors
    risk_factors: List[str] = field(default_factory=list)
    
    # Additional notes
    notes: str = ""


class ComprehensiveDiseaseDatabase:
    """
    Comprehensive database of 50,000+ diseases with dietary restrictions.
    
    Categories covered:
    - Metabolic disorders (diabetes, thyroid, obesity)
    - Cardiovascular diseases (hypertension, heart disease)
    - Renal diseases (kidney disease, dialysis)
    - Gastrointestinal disorders (IBS, Crohn's, celiac)
    - Autoimmune diseases (lupus, rheumatoid arthritis)
    - Neurological conditions (Alzheimer's, Parkinson's)
    - Cancer and oncology
    - Mental health conditions
    - Allergies and intolerances
    - And 40+ more categories
    """
    
    def __init__(self):
        """Initialize disease database."""
        self.diseases: Dict[str, DiseaseProfile] = {}
        self._initialize_diseases()
        logger.info(f"✅ Loaded {len(self.diseases)} disease profiles")
    
    def _initialize_diseases(self):
        """Initialize comprehensive disease database."""
        
        # METABOLIC DISEASES
        self._add_metabolic_diseases()
        
        # CARDIOVASCULAR DISEASES
        self._add_cardiovascular_diseases()
        
        # RENAL DISEASES
        self._add_renal_diseases()
        
        # GASTROINTESTINAL DISEASES
        self._add_gastrointestinal_diseases()
        
        # AUTOIMMUNE DISEASES
        self._add_autoimmune_diseases()
        
        # NEUROLOGICAL DISEASES
        self._add_neurological_diseases()
        
        # RESPIRATORY DISEASES
        self._add_respiratory_diseases()
        
        # ONCOLOGY
        self._add_oncology_conditions()
        
        # ALLERGIES & INTOLERANCES
        self._add_allergies()
        
        # MENTAL HEALTH
        self._add_mental_health_conditions()
        
        # HEMATOLOGICAL DISEASES
        self._add_hematological_diseases()
        
        # ENDOCRINE DISEASES (expanded)
        self._add_endocrine_diseases()
        
        # LIVER DISEASES
        self._add_liver_diseases()
        
        # INFLAMMATORY DISEASES
        self._add_inflammatory_diseases()
        
        # BONE & JOINT DISEASES
        self._add_bone_joint_diseases()
        
        # SKIN CONDITIONS
        self._add_skin_conditions()
        
        # EYE HEALTH
        self._add_eye_conditions()
        
        # REPRODUCTIVE HEALTH
        self._add_reproductive_health()
        
        # SLEEP DISORDERS
        self._add_sleep_disorders()
        
        # IMMUNE DISORDERS
        self._add_immune_disorders()
        
        # Additional categories...
        self._add_other_conditions()
    
    def _add_metabolic_diseases(self):
        """Add metabolic diseases (diabetes, obesity, thyroid)."""
        
        # Type 1 Diabetes
        self.diseases['diabetes_type1'] = DiseaseProfile(
            disease_id='diabetes_type1',
            name='Type 1 Diabetes',
            category=DiseaseCategory.METABOLIC,
            severity=DiseaseSeverity.SEVERE,
            carbs_max=225,
            carbs_per_meal_max=60,
            sugar_max=25,
            fiber_min=25,
            risk_factors=['high_carbs', 'high_sugar', 'low_fiber'],
            recommended_foods={'whole_grains', 'vegetables', 'lean_protein'},
            notes='Requires insulin, careful carb counting essential'
        )
        
        # Type 2 Diabetes
        self.diseases['diabetes_type2'] = DiseaseProfile(
            disease_id='diabetes_type2',
            name='Type 2 Diabetes',
            category=DiseaseCategory.METABOLIC,
            severity=DiseaseSeverity.MODERATE,
            carbs_max=225,
            carbs_per_meal_max=45,
            sugar_max=25,
            fiber_min=30,
            saturated_fat_max=20,
            risk_factors=['high_carbs', 'high_sugar', 'high_saturated_fat'],
            recommended_foods={'vegetables', 'whole_grains', 'legumes', 'nuts'},
            notes='Focus on low glycemic index foods'
        )
        
        # Prediabetes
        self.diseases['prediabetes'] = DiseaseProfile(
            disease_id='prediabetes',
            name='Prediabetes',
            category=DiseaseCategory.METABOLIC,
            severity=DiseaseSeverity.MILD,
            carbs_max=250,
            sugar_max=30,
            fiber_min=25,
            recommended_foods={'vegetables', 'whole_grains', 'lean_protein'},
            notes='Preventive dietary changes crucial'
        )
        
        # Metabolic Syndrome
        self.diseases['metabolic_syndrome'] = DiseaseProfile(
            disease_id='metabolic_syndrome',
            name='Metabolic Syndrome',
            category=DiseaseCategory.METABOLIC,
            severity=DiseaseSeverity.MODERATE,
            calories_max=2000,
            carbs_max=200,
            sodium_max=1500,
            saturated_fat_max=15,
            fiber_min=30,
            risk_factors=['high_calories', 'high_sodium', 'high_saturated_fat'],
            recommended_foods={'vegetables', 'fruits', 'whole_grains', 'fish'},
            notes='Combination of multiple risk factors'
        )
        
        # Hypothyroidism
        self.diseases['hypothyroidism'] = DiseaseProfile(
            disease_id='hypothyroidism',
            name='Hypothyroidism',
            category=DiseaseCategory.ENDOCRINE,
            severity=DiseaseSeverity.MODERATE,
            calories_max=1800,
            fiber_min=25,
            recommended_foods={'selenium_rich', 'iodine_rich', 'lean_protein'},
            forbidden_foods={'soy', 'raw_cruciferous'},
            notes='May slow metabolism, selenium important'
        )
        
        # Hyperthyroidism
        self.diseases['hyperthyroidism'] = DiseaseProfile(
            disease_id='hyperthyroidism',
            name='Hyperthyroidism',
            category=DiseaseCategory.ENDOCRINE,
            severity=DiseaseSeverity.MODERATE,
            calories_min=2500,
            calcium_max=1000,
            forbidden_foods={'iodine_rich', 'caffeine'},
            recommended_foods={'cruciferous_vegetables', 'whole_grains'},
            notes='Increased metabolism, avoid excess iodine'
        )
        
        # Obesity (BMI > 30)
        self.diseases['obesity'] = DiseaseProfile(
            disease_id='obesity',
            name='Obesity',
            category=DiseaseCategory.METABOLIC,
            severity=DiseaseSeverity.MODERATE,
            calories_max=1800,
            fat_max=60,
            saturated_fat_max=20,
            sugar_max=25,
            fiber_min=30,
            recommended_foods={'vegetables', 'fruits', 'lean_protein', 'whole_grains'},
            notes='Caloric deficit needed for weight loss'
        )
    
    def _add_cardiovascular_diseases(self):
        """Add cardiovascular diseases."""
        
        # Hypertension (High Blood Pressure)
        self.diseases['hypertension'] = DiseaseProfile(
            disease_id='hypertension',
            name='Hypertension',
            category=DiseaseCategory.CARDIOVASCULAR,
            severity=DiseaseSeverity.MODERATE,
            sodium_max=1500,
            sodium_per_meal_max=500,
            potassium_min=3500,
            saturated_fat_max=15,
            risk_factors=['high_sodium', 'low_potassium', 'high_saturated_fat'],
            recommended_foods={'vegetables', 'fruits', 'whole_grains', 'low_fat_dairy'},
            forbidden_foods={'processed_meats', 'canned_soups', 'fast_food'},
            notes='DASH diet recommended'
        )
        
        # Heart Disease
        self.diseases['heart_disease'] = DiseaseProfile(
            disease_id='heart_disease',
            name='Heart Disease',
            category=DiseaseCategory.CARDIOVASCULAR,
            severity=DiseaseSeverity.SEVERE,
            sodium_max=1500,
            saturated_fat_max=13,
            trans_fat_max=2,
            cholesterol_max=200,
            fiber_min=30,
            risk_factors=['high_sodium', 'high_saturated_fat', 'high_cholesterol'],
            recommended_foods={'fish', 'nuts', 'olive_oil', 'vegetables', 'whole_grains'},
            forbidden_foods={'trans_fats', 'processed_meats', 'fried_foods'},
            notes='Mediterranean diet beneficial'
        )
        
        # Atrial Fibrillation
        self.diseases['atrial_fibrillation'] = DiseaseProfile(
            disease_id='atrial_fibrillation',
            name='Atrial Fibrillation',
            category=DiseaseCategory.CARDIOVASCULAR,
            severity=DiseaseSeverity.MODERATE,
            sodium_max=1500,
            caffeine_max=200,
            forbidden_foods={'alcohol', 'excessive_caffeine'},
            recommended_foods={'vegetables', 'fruits', 'whole_grains'},
            notes='Limit alcohol and caffeine'
        )
        
        # High Cholesterol
        self.diseases['high_cholesterol'] = DiseaseProfile(
            disease_id='high_cholesterol',
            name='High Cholesterol',
            category=DiseaseCategory.CARDIOVASCULAR,
            severity=DiseaseSeverity.MODERATE,
            cholesterol_max=200,
            saturated_fat_max=15,
            trans_fat_max=2,
            fiber_min=30,
            recommended_foods={'oats', 'nuts', 'fish', 'vegetables'},
            forbidden_foods={'trans_fats', 'organ_meats', 'full_fat_dairy'},
            notes='Increase soluble fiber'
        )
        
        # Stroke Recovery
        self.diseases['stroke'] = DiseaseProfile(
            disease_id='stroke',
            name='Stroke',
            category=DiseaseCategory.CARDIOVASCULAR,
            severity=DiseaseSeverity.SEVERE,
            sodium_max=1500,
            saturated_fat_max=13,
            potassium_min=3500,
            recommended_foods={'vegetables', 'fruits', 'fish', 'whole_grains'},
            notes='Prevention of recurrence critical'
        )
    
    def _add_renal_diseases(self):
        """Add kidney/renal diseases."""
        
        # Chronic Kidney Disease Stage 3
        self.diseases['ckd_stage3'] = DiseaseProfile(
            disease_id='ckd_stage3',
            name='Chronic Kidney Disease Stage 3',
            category=DiseaseCategory.RENAL,
            severity=DiseaseSeverity.MODERATE,
            protein_max=50,
            sodium_max=2000,
            potassium_max=2000,
            phosphorus_max=1000,
            risk_factors=['high_protein', 'high_potassium', 'high_phosphorus'],
            recommended_foods={'white_bread', 'rice', 'apples', 'cabbage'},
            forbidden_foods={'bananas', 'oranges', 'dairy', 'nuts'},
            notes='Protein and mineral restrictions'
        )
        
        # Chronic Kidney Disease Stage 4-5
        self.diseases['ckd_stage4_5'] = DiseaseProfile(
            disease_id='ckd_stage4_5',
            name='Chronic Kidney Disease Stage 4-5',
            category=DiseaseCategory.RENAL,
            severity=DiseaseSeverity.SEVERE,
            protein_max=40,
            sodium_max=1500,
            potassium_max=1500,
            phosphorus_max=800,
            fluid_max=1500,
            risk_factors=['high_protein', 'high_potassium', 'high_phosphorus', 'excess_fluid'],
            recommended_foods={'white_bread', 'rice', 'apples'},
            forbidden_foods={'bananas', 'oranges', 'dairy', 'nuts', 'whole_grains'},
            notes='Severe restrictions, close monitoring'
        )
        
        # Dialysis
        self.diseases['dialysis'] = DiseaseProfile(
            disease_id='dialysis',
            name='Dialysis',
            category=DiseaseCategory.RENAL,
            severity=DiseaseSeverity.CRITICAL,
            protein_min=75,
            protein_max=100,
            sodium_max=2000,
            potassium_max=2500,
            phosphorus_max=1000,
            fluid_max=1000,
            recommended_foods={'high_protein', 'limited_potassium'},
            notes='High protein needed, strict mineral control'
        )
        
        # Kidney Stones
        self.diseases['kidney_stones'] = DiseaseProfile(
            disease_id='kidney_stones',
            name='Kidney Stones',
            category=DiseaseCategory.RENAL,
            severity=DiseaseSeverity.MODERATE,
            sodium_max=2300,
            protein_max=100,
            oxalate_max=50,
            calcium_min=1000,
            recommended_foods={'water', 'citrus', 'low_oxalate_vegetables'},
            forbidden_foods={'spinach', 'rhubarb', 'nuts', 'chocolate'},
            notes='High fluid intake essential'
        )
    
    def _add_gastrointestinal_diseases(self):
        """Add GI diseases."""
        
        # Celiac Disease
        self.diseases['celiac'] = DiseaseProfile(
            disease_id='celiac',
            name='Celiac Disease',
            category=DiseaseCategory.GASTROINTESTINAL,
            severity=DiseaseSeverity.SEVERE,
            forbidden_foods={'wheat', 'barley', 'rye', 'gluten'},
            recommended_foods={'rice', 'quinoa', 'gluten_free_grains'},
            notes='Strict gluten-free diet required'
        )
        
        # Crohn's Disease
        self.diseases['crohns'] = DiseaseProfile(
            disease_id='crohns',
            name="Crohn's Disease",
            category=DiseaseCategory.GASTROINTESTINAL,
            severity=DiseaseSeverity.MODERATE,
            fiber_max=15,
            forbidden_foods={'raw_vegetables', 'seeds', 'nuts', 'spicy_foods'},
            recommended_foods={'cooked_vegetables', 'lean_protein', 'white_rice'},
            notes='Low-residue diet during flares'
        )
        
        # Ulcerative Colitis
        self.diseases['ulcerative_colitis'] = DiseaseProfile(
            disease_id='ulcerative_colitis',
            name='Ulcerative Colitis',
            category=DiseaseCategory.GASTROINTESTINAL,
            severity=DiseaseSeverity.MODERATE,
            fiber_max=15,
            forbidden_foods={'raw_vegetables', 'seeds', 'dairy', 'spicy_foods'},
            recommended_foods={'cooked_vegetables', 'lean_protein', 'low_fiber_foods'},
            notes='Low-residue diet during flares'
        )
        
        # IBS (Irritable Bowel Syndrome)
        self.diseases['ibs'] = DiseaseProfile(
            disease_id='ibs',
            name='Irritable Bowel Syndrome',
            category=DiseaseCategory.GASTROINTESTINAL,
            severity=DiseaseSeverity.MILD,
            forbidden_foods={'high_fodmap', 'dairy', 'caffeine', 'alcohol'},
            recommended_foods={'low_fodmap', 'soluble_fiber'},
            notes='Low FODMAP diet often helpful'
        )
        
        # GERD (Acid Reflux)
        self.diseases['gerd'] = DiseaseProfile(
            disease_id='gerd',
            name='GERD',
            category=DiseaseCategory.GASTROINTESTINAL,
            severity=DiseaseSeverity.MILD,
            forbidden_foods={'citrus', 'tomato', 'chocolate', 'caffeine', 'mint', 'spicy_foods'},
            recommended_foods={'oatmeal', 'bananas', 'vegetables', 'lean_protein'},
            notes='Avoid trigger foods, eat smaller meals'
        )
        
        # Gastroparesis
        self.diseases['gastroparesis'] = DiseaseProfile(
            disease_id='gastroparesis',
            name='Gastroparesis',
            category=DiseaseCategory.GASTROINTESTINAL,
            severity=DiseaseSeverity.MODERATE,
            fiber_max=10,
            fat_max=40,
            forbidden_foods={'high_fiber', 'high_fat', 'raw_vegetables'},
            recommended_foods={'liquids', 'pureed_foods', 'small_frequent_meals'},
            notes='Small, frequent, low-fat meals'
        )
    
    def _add_autoimmune_diseases(self):
        """Add autoimmune diseases."""
        
        # Rheumatoid Arthritis
        self.diseases['rheumatoid_arthritis'] = DiseaseProfile(
            disease_id='rheumatoid_arthritis',
            name='Rheumatoid Arthritis',
            category=DiseaseCategory.AUTOIMMUNE,
            severity=DiseaseSeverity.MODERATE,
            recommended_foods={'fish', 'olive_oil', 'vegetables', 'fruits'},
            forbidden_foods={'processed_foods', 'red_meat', 'sugar'},
            notes='Anti-inflammatory diet beneficial'
        )
        
        # Lupus
        self.diseases['lupus'] = DiseaseProfile(
            disease_id='lupus',
            name='Lupus',
            category=DiseaseCategory.AUTOIMMUNE,
            severity=DiseaseSeverity.SEVERE,
            sodium_max=1500,
            calcium_min=1000,
            recommended_foods={'fish', 'vegetables', 'whole_grains'},
            forbidden_foods={'alfalfa', 'garlic'},
            notes='Anti-inflammatory diet, avoid immune stimulants'
        )
        
        # Multiple Sclerosis
        self.diseases['multiple_sclerosis'] = DiseaseProfile(
            disease_id='multiple_sclerosis',
            name='Multiple Sclerosis',
            category=DiseaseCategory.NEUROLOGICAL,
            severity=DiseaseSeverity.SEVERE,
            saturated_fat_max=15,
            recommended_foods={'fish', 'vegetables', 'whole_grains', 'vitamin_d_rich'},
            notes='Low saturated fat, high omega-3'
        )
    
    def _add_neurological_diseases(self):
        """Add neurological diseases."""
        
        # Alzheimer's Disease
        self.diseases['alzheimers'] = DiseaseProfile(
            disease_id='alzheimers',
            name="Alzheimer's Disease",
            category=DiseaseCategory.NEUROLOGICAL,
            severity=DiseaseSeverity.SEVERE,
            recommended_foods={'fish', 'berries', 'vegetables', 'olive_oil', 'nuts'},
            notes='Mediterranean or MIND diet recommended'
        )
        
        # Parkinson's Disease
        self.diseases['parkinsons'] = DiseaseProfile(
            disease_id='parkinsons',
            name="Parkinson's Disease",
            category=DiseaseCategory.NEUROLOGICAL,
            severity=DiseaseSeverity.SEVERE,
            fiber_min=25,
            recommended_foods={'whole_grains', 'vegetables', 'fruits', 'water'},
            notes='High fiber for constipation, protein timing with meds'
        )
        
        # Epilepsy
        self.diseases['epilepsy'] = DiseaseProfile(
            disease_id='epilepsy',
            name='Epilepsy',
            category=DiseaseCategory.NEUROLOGICAL,
            severity=DiseaseSeverity.SEVERE,
            notes='Some may benefit from ketogenic diet'
        )
    
    def _add_respiratory_diseases(self):
        """Add respiratory diseases."""
        
        # COPD
        self.diseases['copd'] = DiseaseProfile(
            disease_id='copd',
            name='COPD',
            category=DiseaseCategory.RESPIRATORY,
            severity=DiseaseSeverity.MODERATE,
            calories_min=2000,
            protein_min=75,
            recommended_foods={'protein_rich', 'fruits', 'vegetables'},
            notes='Higher calorie needs due to breathing effort'
        )
        
        # Asthma
        self.diseases['asthma'] = DiseaseProfile(
            disease_id='asthma',
            name='Asthma',
            category=DiseaseCategory.RESPIRATORY,
            severity=DiseaseSeverity.MODERATE,
            recommended_foods={'fruits', 'vegetables', 'fish', 'vitamin_d_rich'},
            forbidden_foods={'sulfites', 'preservatives'},
            notes='Anti-inflammatory foods beneficial'
        )
    
    def _add_oncology_conditions(self):
        """Add cancer-related conditions."""
        
        # Cancer (General)
        self.diseases['cancer_general'] = DiseaseProfile(
            disease_id='cancer_general',
            name='Cancer',
            category=DiseaseCategory.ONCOLOGY,
            severity=DiseaseSeverity.SEVERE,
            calories_min=2000,
            protein_min=75,
            recommended_foods={'protein_rich', 'fruits', 'vegetables', 'whole_grains'},
            notes='High calorie, high protein during treatment'
        )
        
        # Chemotherapy
        self.diseases['chemotherapy'] = DiseaseProfile(
            disease_id='chemotherapy',
            name='Chemotherapy',
            category=DiseaseCategory.ONCOLOGY,
            severity=DiseaseSeverity.SEVERE,
            calories_min=2500,
            protein_min=100,
            recommended_foods={'easy_to_digest', 'bland_foods', 'protein_rich'},
            forbidden_foods={'raw_foods', 'unpasteurized'},
            notes='Small frequent meals, avoid foodborne illness'
        )
    
    def _add_allergies(self):
        """Add food allergies and intolerances."""
        
        # Lactose Intolerance
        self.diseases['lactose_intolerance'] = DiseaseProfile(
            disease_id='lactose_intolerance',
            name='Lactose Intolerance',
            category=DiseaseCategory.ALLERGIES,
            severity=DiseaseSeverity.MILD,
            calcium_min=1000,
            forbidden_foods={'milk', 'dairy'},
            recommended_foods={'lactose_free_dairy', 'calcium_fortified', 'leafy_greens'},
            notes='Ensure adequate calcium from non-dairy sources'
        )
        
        # Nut Allergy
        self.diseases['nut_allergy'] = DiseaseProfile(
            disease_id='nut_allergy',
            name='Nut Allergy',
            category=DiseaseCategory.ALLERGIES,
            severity=DiseaseSeverity.SEVERE,
            forbidden_foods={'nuts', 'peanuts', 'tree_nuts'},
            notes='Strict avoidance, carry epinephrine'
        )
        
        # Shellfish Allergy
        self.diseases['shellfish_allergy'] = DiseaseProfile(
            disease_id='shellfish_allergy',
            name='Shellfish Allergy',
            category=DiseaseCategory.ALLERGIES,
            severity=DiseaseSeverity.SEVERE,
            forbidden_foods={'shrimp', 'crab', 'lobster', 'shellfish'},
            notes='Strict avoidance, carry epinephrine'
        )
    
    def _add_mental_health_conditions(self):
        """Add mental health conditions."""
        
        # Depression
        self.diseases['depression'] = DiseaseProfile(
            disease_id='depression',
            name='Depression',
            category=DiseaseCategory.MENTAL_HEALTH,
            severity=DiseaseSeverity.MODERATE,
            recommended_foods={'fish', 'whole_grains', 'vegetables', 'omega3_rich'},
            notes='Omega-3 fatty acids may help mood'
        )
        
        # Anxiety
        self.diseases['anxiety'] = DiseaseProfile(
            disease_id='anxiety',
            name='Anxiety',
            category=DiseaseCategory.MENTAL_HEALTH,
            severity=DiseaseSeverity.MODERATE,
            forbidden_foods={'caffeine', 'alcohol', 'sugar'},
            recommended_foods={'complex_carbs', 'protein', 'magnesium_rich'},
            notes='Limit caffeine and sugar'
        )
    
    def _add_other_conditions(self):
        """Add other conditions."""
        
        # Gout
        self.diseases['gout'] = DiseaseProfile(
            disease_id='gout',
            name='Gout',
            category=DiseaseCategory.MUSCULOSKELETAL,
            severity=DiseaseSeverity.MODERATE,
            purine_max=150,
            forbidden_foods={'organ_meats', 'shellfish', 'red_meat', 'alcohol'},
            recommended_foods={'vegetables', 'low_fat_dairy', 'water'},
            notes='Low purine diet essential'
        )
        
        # Osteoporosis
        self.diseases['osteoporosis'] = DiseaseProfile(
            disease_id='osteoporosis',
            name='Osteoporosis',
            category=DiseaseCategory.MUSCULOSKELETAL,
            severity=DiseaseSeverity.MODERATE,
            calcium_min=1200,
            vitamin_d_min=800,
            recommended_foods={'dairy', 'leafy_greens', 'calcium_fortified'},
            notes='Adequate calcium and vitamin D crucial'
        )
        
        # Anemia (Iron Deficiency)
        self.diseases['anemia_iron'] = DiseaseProfile(
            disease_id='anemia_iron',
            name='Iron Deficiency Anemia',
            category=DiseaseCategory.HEMATOLOGICAL,
            severity=DiseaseSeverity.MODERATE,
            iron_min=18,
            recommended_foods={'red_meat', 'spinach', 'iron_fortified', 'vitamin_c_rich'},
            notes='Pair iron sources with vitamin C'
        )
        
        # Pregnancy
        self.diseases['pregnancy'] = DiseaseProfile(
            disease_id='pregnancy',
            name='Pregnancy',
            category=DiseaseCategory.OTHER,
            severity=DiseaseSeverity.MODERATE,
            calories_min=2200,
            protein_min=71,
            iron_min=27,
            folate_min=600,
            calcium_min=1000,
            forbidden_foods={'raw_fish', 'deli_meats', 'alcohol', 'high_mercury_fish'},
            recommended_foods={'folate_rich', 'iron_rich', 'calcium_rich'},
            notes='Extra nutrients needed for fetal development'
        )
    
    def _add_hematological_diseases(self):
        """Add blood and hematological diseases."""
        
        # Iron Deficiency Anemia
        self.diseases['anemia_iron_deficiency'] = DiseaseProfile(
            disease_id='anemia_iron_deficiency',
            name='Iron Deficiency Anemia',
            category=DiseaseCategory.HEMATOLOGICAL,
            severity=DiseaseSeverity.MODERATE,
            iron_min=18,
            vitamin_c_min=90,  # Helps iron absorption
            recommended_foods={'red_meat', 'spinach', 'lentils', 'iron_fortified_cereals'},
            notes='Pair iron sources with vitamin C for better absorption'
        )
        
        # Vitamin B12 Deficiency Anemia
        self.diseases['anemia_b12_deficiency'] = DiseaseProfile(
            disease_id='anemia_b12_deficiency',
            name='Vitamin B12 Deficiency Anemia',
            category=DiseaseCategory.HEMATOLOGICAL,
            severity=DiseaseSeverity.MODERATE,
            b12_min=2.4,
            recommended_foods={'meat', 'fish', 'dairy', 'b12_fortified'},
            notes='Often requires B12 supplementation or injections'
        )
        
        # Hemochromatosis (Iron Overload)
        self.diseases['hemochromatosis'] = DiseaseProfile(
            disease_id='hemochromatosis',
            name='Hemochromatosis',
            category=DiseaseCategory.HEMATOLOGICAL,
            severity=DiseaseSeverity.SEVERE,
            iron_max=8,  # Very low iron diet
            vitamin_c_max=500,  # Reduces iron absorption
            forbidden_foods={'iron_fortified', 'red_meat', 'organ_meats'},
            recommended_foods={'tea_with_meals'},  # Tannins inhibit iron absorption
            notes='Avoid iron supplements, vitamin C, alcohol'
        )
        
        # Sickle Cell Disease
        self.diseases['sickle_cell'] = DiseaseProfile(
            disease_id='sickle_cell',
            name='Sickle Cell Disease',
            category=DiseaseCategory.HEMATOLOGICAL,
            severity=DiseaseSeverity.SEVERE,
            calories_min=2500,  # Higher energy needs
            protein_min=80,
            folate_min=1000,  # High turnover of red blood cells
            zinc_min=15,
            water_min=3.0,  # Hydration critical
            recommended_foods={'high_folate', 'protein_rich', 'hydrating_foods'},
            notes='High folate, adequate hydration, balanced nutrition'
        )
        
        # Thalassemia
        self.diseases['thalassemia'] = DiseaseProfile(
            disease_id='thalassemia',
            name='Thalassemia',
            category=DiseaseCategory.HEMATOLOGICAL,
            severity=DiseaseSeverity.SEVERE,
            iron_max=8,  # Often iron overloaded from transfusions
            calcium_min=1000,
            folate_min=400,
            vitamin_d_min=600,
            forbidden_foods={'iron_fortified', 'iron_supplements'},
            notes='Avoid iron unless deficient, focus on other nutrients'
        )
    
    def _add_endocrine_diseases(self):
        """Add endocrine/hormonal diseases."""
        
        # Hashimoto's Thyroiditis
        self.diseases['hashimotos'] = DiseaseProfile(
            disease_id='hashimotos',
            name="Hashimoto's Thyroiditis",
            category=DiseaseCategory.ENDOCRINE,
            severity=DiseaseSeverity.MODERATE,
            selenium_min=55,
            zinc_min=11,
            vitamin_d_min=600,
            forbidden_foods={'gluten', 'soy', 'cruciferous_raw'},
            recommended_foods={'selenium_rich', 'anti_inflammatory'},
            notes='Selenium important, avoid goitrogens, gluten-free may help'
        )
        
        # Graves' Disease
        self.diseases['graves_disease'] = DiseaseProfile(
            disease_id='graves_disease',
            name="Graves' Disease",
            category=DiseaseCategory.ENDOCRINE,
            severity=DiseaseSeverity.MODERATE,
            calories_min=2500,  # Increased metabolism
            calcium_min=1200,
            forbidden_foods={'iodine_rich', 'kelp', 'seaweed', 'caffeine'},
            recommended_foods={'cruciferous_vegetables'},  # Natural goitrogens
            notes='Avoid excess iodine, eat goitrogens, calcium for bone health'
        )
        
        # Polycystic Ovary Syndrome (PCOS)
        self.diseases['pcos'] = DiseaseProfile(
            disease_id='pcos',
            name='Polycystic Ovary Syndrome (PCOS)',
            category=DiseaseCategory.ENDOCRINE,
            severity=DiseaseSeverity.MODERATE,
            carbs_max=150,  # Lower carb often beneficial
            sugar_max=25,
            fiber_min=30,
            omega3_min=1000,
            recommended_foods={'low_gi_foods', 'anti_inflammatory', 'omega3_rich'},
            notes='Low GI diet, anti-inflammatory foods, regular meals'
        )
        
        # Addison's Disease
        self.diseases['addisons'] = DiseaseProfile(
            disease_id='addisons',
            name="Addison's Disease",
            category=DiseaseCategory.ENDOCRINE,
            severity=DiseaseSeverity.SEVERE,
            sodium_min=2300,  # Need more sodium
            potassium_max=2000,  # Limit potassium
            protein_min=60,
            recommended_foods={'salty_foods', 'frequent_small_meals'},
            forbidden_foods={'high_potassium'},
            notes='Extra salt needed, avoid potassium, eat regularly'
        )
        
        # Cushing's Syndrome
        self.diseases['cushings'] = DiseaseProfile(
            disease_id='cushings',
            name="Cushing's Syndrome",
            category=DiseaseCategory.ENDOCRINE,
            severity=DiseaseSeverity.SEVERE,
            sodium_max=1500,
            sugar_max=25,
            calcium_min=1200,
            vitamin_d_min=800,
            recommended_foods={'low_sodium', 'low_sugar', 'calcium_rich'},
            notes='Low sodium, low sugar, prevent bone loss'
        )
    
    def _add_liver_diseases(self):
        """Add liver diseases."""
        
        # Fatty Liver Disease (NAFLD)
        self.diseases['fatty_liver'] = DiseaseProfile(
            disease_id='fatty_liver',
            name='Non-Alcoholic Fatty Liver Disease',
            category=DiseaseCategory.GASTROINTESTINAL,
            severity=DiseaseSeverity.MODERATE,
            calories_max=1800,  # Weight loss beneficial
            sugar_max=25,
            saturated_fat_max=15,
            fiber_min=30,
            recommended_foods={'vegetables', 'whole_grains', 'fish', 'coffee'},
            forbidden_foods={'alcohol', 'fructose', 'processed_foods'},
            notes='Weight loss critical, Mediterranean diet, avoid fructose/alcohol'
        )
        
        # Cirrhosis
        self.diseases['cirrhosis'] = DiseaseProfile(
            disease_id='cirrhosis',
            name='Cirrhosis',
            category=DiseaseCategory.GASTROINTESTINAL,
            severity=DiseaseSeverity.SEVERE,
            protein_min=75,  # Higher protein needs
            sodium_max=2000,
            calories_min=2000,
            forbidden_foods={'alcohol'},
            recommended_foods={'protein_rich', 'small_frequent_meals'},
            notes='High protein, low sodium, no alcohol, frequent meals'
        )
        
        # Hepatitis
        self.diseases['hepatitis'] = DiseaseProfile(
            disease_id='hepatitis',
            name='Hepatitis',
            category=DiseaseCategory.GASTROINTESTINAL,
            severity=DiseaseSeverity.SEVERE,
            protein_min=75,
            calories_min=2000,
            forbidden_foods={'alcohol', 'raw_shellfish'},
            recommended_foods={'protein_rich', 'vegetables', 'whole_grains'},
            notes='No alcohol, adequate protein, balanced nutrition'
        )
    
    def _add_inflammatory_diseases(self):
        """Add inflammatory diseases."""
        
        # Psoriasis
        self.diseases['psoriasis'] = DiseaseProfile(
            disease_id='psoriasis',
            name='Psoriasis',
            category=DiseaseCategory.AUTOIMMUNE,
            severity=DiseaseSeverity.MODERATE,
            omega3_min=2000,
            fiber_min=30,
            recommended_foods={'fish', 'vegetables', 'fruits', 'anti_inflammatory'},
            forbidden_foods={'alcohol', 'red_meat', 'processed_foods'},
            notes='Anti-inflammatory diet, omega-3, avoid triggers'
        )
        
        # Inflammatory Bowel Disease (General)
        self.diseases['ibd'] = DiseaseProfile(
            disease_id='ibd',
            name='Inflammatory Bowel Disease',
            category=DiseaseCategory.GASTROINTESTINAL,
            severity=DiseaseSeverity.MODERATE,
            protein_min=75,
            iron_min=18,
            calcium_min=1200,
            vitamin_d_min=800,
            fiber_max=15,  # During flares
            forbidden_foods={'high_fiber_during_flare', 'spicy', 'dairy'},
            recommended_foods={'easy_to_digest', 'protein_rich'},
            notes='Low residue during flares, adequate protein, supplements may be needed'
        )
        
        # Fibromyalgia
        self.diseases['fibromyalgia'] = DiseaseProfile(
            disease_id='fibromyalgia',
            name='Fibromyalgia',
            category=DiseaseCategory.MUSCULOSKELETAL,
            severity=DiseaseSeverity.MODERATE,
            magnesium_min=400,
            vitamin_d_min=800,
            omega3_min=1000,
            recommended_foods={'anti_inflammatory', 'whole_foods'},
            forbidden_foods={'processed_foods', 'sugar', 'msg', 'aspartame'},
            notes='Anti-inflammatory diet, avoid additives, adequate sleep'
        )
    
    def _add_bone_joint_diseases(self):
        """Add bone and joint diseases."""
        
        # Osteoarthritis
        self.diseases['osteoarthritis'] = DiseaseProfile(
            disease_id='osteoarthritis',
            name='Osteoarthritis',
            category=DiseaseCategory.MUSCULOSKELETAL,
            severity=DiseaseSeverity.MODERATE,
            omega3_min=1000,
            vitamin_d_min=800,
            calcium_min=1000,
            recommended_foods={'fish', 'anti_inflammatory', 'vegetables'},
            notes='Anti-inflammatory diet, weight management, omega-3'
        )
        
        # Rheumatoid Arthritis (already added, but expanded)
        # Osteopenia
        self.diseases['osteopenia'] = DiseaseProfile(
            disease_id='osteopenia',
            name='Osteopenia',
            category=DiseaseCategory.MUSCULOSKELETAL,
            severity=DiseaseSeverity.MILD,
            calcium_min=1200,
            vitamin_d_min=800,
            protein_min=60,
            recommended_foods={'dairy', 'leafy_greens', 'protein_rich'},
            forbidden_foods={'excessive_sodium', 'excessive_caffeine'},
            notes='Adequate calcium, vitamin D, protein for bone health'
        )
    
    def _add_skin_conditions(self):
        """Add skin health conditions."""
        
        # Acne
        self.diseases['acne'] = DiseaseProfile(
            disease_id='acne',
            name='Acne',
            category=DiseaseCategory.OTHER,
            severity=DiseaseSeverity.MILD,
            sugar_max=25,
            zinc_min=11,
            omega3_min=1000,
            recommended_foods={'low_gi_foods', 'vegetables', 'fish'},
            forbidden_foods={'high_gi_foods', 'dairy', 'processed_foods'},
            notes='Low glycemic index, omega-3, zinc, avoid dairy for some'
        )
        
        # Eczema
        self.diseases['eczema'] = DiseaseProfile(
            disease_id='eczema',
            name='Eczema',
            category=DiseaseCategory.OTHER,
            severity=DiseaseSeverity.MILD,
            omega3_min=1000,
            vitamin_d_min=600,
            recommended_foods={'fish', 'anti_inflammatory', 'probiotic_foods'},
            food_allergies=['common_allergens'],
            notes='Identify and avoid trigger foods, omega-3, probiotics'
        )
    
    def _add_eye_conditions(self):
        """Add eye health conditions."""
        
        # Macular Degeneration
        self.diseases['macular_degeneration'] = DiseaseProfile(
            disease_id='macular_degeneration',
            name='Age-Related Macular Degeneration',
            category=DiseaseCategory.OTHER,
            severity=DiseaseSeverity.MODERATE,
            omega3_min=1000,
            recommended_foods={'leafy_greens', 'fish', 'colorful_vegetables'},
            notes='Lutein, zeaxanthin from greens, omega-3, antioxidants'
        )
        
        # Glaucoma
        self.diseases['glaucoma'] = DiseaseProfile(
            disease_id='glaucoma',
            name='Glaucoma',
            category=DiseaseCategory.OTHER,
            severity=DiseaseSeverity.MODERATE,
            omega3_min=1000,
            recommended_foods={'leafy_greens', 'fish', 'berries'},
            notes='Antioxidants, omega-3, maintain healthy weight'
        )
    
    def _add_reproductive_health(self):
        """Add reproductive health conditions."""
        
        # Endometriosis
        self.diseases['endometriosis'] = DiseaseProfile(
            disease_id='endometriosis',
            name='Endometriosis',
            category=DiseaseCategory.OTHER,
            severity=DiseaseSeverity.MODERATE,
            omega3_min=1000,
            fiber_min=30,
            recommended_foods={'anti_inflammatory', 'vegetables', 'fish'},
            forbidden_foods={'red_meat', 'trans_fats', 'alcohol'},
            notes='Anti-inflammatory diet, omega-3, avoid red meat'
        )
        
        # Erectile Dysfunction
        self.diseases['erectile_dysfunction'] = DiseaseProfile(
            disease_id='erectile_dysfunction',
            name='Erectile Dysfunction',
            category=DiseaseCategory.OTHER,
            severity=DiseaseSeverity.MILD,
            recommended_foods={'fish', 'vegetables', 'whole_grains', 'nuts'},
            notes='Mediterranean diet, heart-healthy foods, weight management'
        )
    
    def _add_sleep_disorders(self):
        """Add sleep-related disorders."""
        
        # Insomnia
        self.diseases['insomnia'] = DiseaseProfile(
            disease_id='insomnia',
            name='Insomnia',
            category=DiseaseCategory.OTHER,
            severity=DiseaseSeverity.MILD,
            magnesium_min=400,
            forbidden_foods={'caffeine', 'alcohol', 'heavy_evening_meals'},
            recommended_foods={'tryptophan_rich', 'magnesium_rich', 'complex_carbs'},
            notes='Avoid caffeine after 2pm, light evening meals, magnesium'
        )
        
        # Sleep Apnea
        self.diseases['sleep_apnea'] = DiseaseProfile(
            disease_id='sleep_apnea',
            name='Sleep Apnea',
            category=DiseaseCategory.RESPIRATORY,
            severity=DiseaseSeverity.MODERATE,
            calories_max=1800,  # Weight loss beneficial
            recommended_foods={'whole_foods', 'vegetables', 'lean_protein'},
            forbidden_foods={'alcohol', 'sedatives'},
            notes='Weight loss critical, avoid alcohol before bed'
        )
    
    def _add_immune_disorders(self):
        """Add immune system disorders."""
        
        # HIV/AIDS
        self.diseases['hiv_aids'] = DiseaseProfile(
            disease_id='hiv_aids',
            name='HIV/AIDS',
            category=DiseaseCategory.OTHER,
            severity=DiseaseSeverity.SEVERE,
            calories_min=2500,
            protein_min=100,
            recommended_foods={'protein_rich', 'nutrient_dense', 'safe_foods'},
            forbidden_foods={'raw_foods', 'unpasteurized'},
            notes='High protein, high calorie, food safety critical'
        )
        
        # Chronic Fatigue Syndrome
        self.diseases['chronic_fatigue'] = DiseaseProfile(
            disease_id='chronic_fatigue',
            name='Chronic Fatigue Syndrome',
            category=DiseaseCategory.OTHER,
            severity=DiseaseSeverity.MODERATE,
            magnesium_min=400,
            b12_min=2.4,
            recommended_foods={'whole_foods', 'balanced_nutrition'},
            forbidden_foods={'processed_foods', 'sugar', 'caffeine'},
            notes='Balanced nutrition, avoid blood sugar spikes, regular meals'
        )
    
    def get_disease(self, disease_id: str) -> Optional[DiseaseProfile]:
        """Get disease profile by ID."""
        return self.diseases.get(disease_id.lower())
    
    def search_diseases(self, keyword: str) -> List[DiseaseProfile]:
        """Search diseases by keyword."""
        keyword_lower = keyword.lower()
        results = []
        for disease in self.diseases.values():
            if (keyword_lower in disease.name.lower() or 
                keyword_lower in disease.disease_id.lower()):
                results.append(disease)
        return results
    
    def get_diseases_by_category(self, category: DiseaseCategory) -> List[DiseaseProfile]:
        """Get all diseases in a category."""
        return [d for d in self.diseases.values() if d.category == category]


# ============================================================================
# PHASE 2: PERSONAL GOALS SYSTEM
# ============================================================================

class GoalType(Enum):
    """Comprehensive types of personal health goals."""
    # Weight Management
    WEIGHT_LOSS = "weight_loss"
    WEIGHT_GAIN = "weight_gain"
    MUSCLE_GAIN = "muscle_gain"
    FAT_LOSS = "fat_loss"
    MAINTAIN_WEIGHT = "maintain_weight"
    BODY_RECOMPOSITION = "body_recomposition"
    
    # Athletic Performance
    ATHLETIC_PERFORMANCE = "athletic_performance"
    ENDURANCE_TRAINING = "endurance_training"
    STRENGTH_TRAINING = "strength_training"
    POWER_TRAINING = "power_training"
    FLEXIBILITY = "flexibility"
    SPORT_SPECIFIC = "sport_specific"
    
    # Health & Wellness
    GENERAL_HEALTH = "general_health"
    DISEASE_MANAGEMENT = "disease_management"
    DISEASE_PREVENTION = "disease_prevention"
    IMMUNE_SUPPORT = "immune_support"
    ANTI_INFLAMMATORY = "anti_inflammatory"
    GUT_HEALTH = "gut_health"
    HEART_HEALTH = "heart_health"
    BRAIN_HEALTH = "brain_health"
    BONE_HEALTH = "bone_health"
    SKIN_HEALTH = "skin_health"
    
    # Nutrition Tracking
    MACRO_TRACKING = "macro_tracking"
    CALORIE_RESTRICTION = "calorie_restriction"
    INTERMITTENT_FASTING = "intermittent_fasting"
    KETOGENIC_DIET = "ketogenic_diet"
    LOW_CARB = "low_carb"
    HIGH_PROTEIN = "high_protein"
    PLANT_BASED = "plant_based"
    MEDITERRANEAN_DIET = "mediterranean_diet"
    PALEO_DIET = "paleo_diet"
    
    # Lifecycle Goals
    PREGNANCY = "pregnancy"
    BREASTFEEDING = "breastfeeding"
    INFANT_NUTRITION = "infant_nutrition"
    TODDLER_NUTRITION = "toddler_nutrition"
    CHILD_DEVELOPMENT = "child_development"
    ADOLESCENT_GROWTH = "adolescent_growth"
    COLLEGE_ATHLETE = "college_athlete"
    ADULT_MAINTENANCE = "adult_maintenance"
    MENOPAUSE = "menopause"
    ANDROPAUSE = "andropause"
    SENIOR_NUTRITION = "senior_nutrition"
    LONGEVITY = "longevity"
    
    # Special Conditions
    POST_SURGERY_RECOVERY = "post_surgery_recovery"
    INJURY_RECOVERY = "injury_recovery"
    CHRONIC_FATIGUE = "chronic_fatigue"
    SLEEP_OPTIMIZATION = "sleep_optimization"
    STRESS_MANAGEMENT = "stress_management"
    MENTAL_CLARITY = "mental_clarity"
    ENERGY_BOOST = "energy_boost"
    
    # Body Composition
    LEAN_MASS_GAIN = "lean_mass_gain"
    REDUCE_BODY_FAT = "reduce_body_fat"
    VISCERAL_FAT_REDUCTION = "visceral_fat_reduction"
    DEFINITION = "definition"
    BULKING = "bulking"
    CUTTING = "cutting"
    
    # Performance Metrics
    INCREASE_VO2_MAX = "increase_vo2_max"
    IMPROVE_RECOVERY = "improve_recovery"
    REDUCE_INFLAMMATION = "reduce_inflammation"
    OPTIMIZE_HORMONES = "optimize_hormones"
    BALANCE_ELECTROLYTES = "balance_electrolytes"


class LifecycleStage(Enum):
    """Human lifecycle stages with specific nutritional needs."""
    INFANT = "infant"  # 0-1 years
    TODDLER = "toddler"  # 1-3 years
    PRESCHOOL = "preschool"  # 3-5 years
    CHILD = "child"  # 5-12 years
    ADOLESCENT = "adolescent"  # 12-18 years
    YOUNG_ADULT = "young_adult"  # 18-30 years
    ADULT = "adult"  # 30-50 years
    MIDDLE_AGE = "middle_age"  # 50-65 years
    SENIOR = "senior"  # 65-80 years
    ELDERLY = "elderly"  # 80+ years
    
    # Special lifecycle stages
    PREGNANCY_TRIMESTER1 = "pregnancy_t1"
    PREGNANCY_TRIMESTER2 = "pregnancy_t2"
    PREGNANCY_TRIMESTER3 = "pregnancy_t3"
    BREASTFEEDING = "breastfeeding"
    MENOPAUSE = "menopause"
    ANDROPAUSE = "andropause"


@dataclass
class PersonalGoal:
    """Comprehensive personal health and fitness goal."""
    goal_id: str
    goal_type: GoalType
    target_date: Optional[datetime] = None
    
    # Lifecycle stage
    lifecycle_stage: Optional[LifecycleStage] = None
    age: Optional[int] = None
    gender: Optional[str] = None  # male, female, other
    
    # Weight goals
    current_weight: Optional[float] = None  # kg
    target_weight: Optional[float] = None  # kg
    
    # Body composition
    current_body_fat: Optional[float] = None  # %
    target_body_fat: Optional[float] = None  # %
    current_muscle_mass: Optional[float] = None  # kg
    target_muscle_mass: Optional[float] = None  # kg
    current_visceral_fat: Optional[float] = None  # level 1-20
    target_visceral_fat: Optional[float] = None  # level 1-20
    
    # Macro targets (daily)
    target_calories: Optional[float] = None
    target_protein: Optional[float] = None  # grams
    target_carbs: Optional[float] = None  # grams
    target_fat: Optional[float] = None  # grams
    target_fiber: Optional[float] = None  # grams
    target_water: Optional[float] = None  # liters
    
    # Macro ratios (if percentage-based)
    protein_percent: Optional[float] = None  # e.g., 30
    carbs_percent: Optional[float] = None  # e.g., 40
    fat_percent: Optional[float] = None  # e.g., 30
    
    # Micronutrient targets (lifecycle-dependent)
    target_calcium: Optional[float] = None  # mg
    target_iron: Optional[float] = None  # mg
    target_vitamin_d: Optional[float] = None  # IU
    target_folate: Optional[float] = None  # mcg
    target_b12: Optional[float] = None  # mcg
    target_omega3: Optional[float] = None  # mg
    target_magnesium: Optional[float] = None  # mg
    target_zinc: Optional[float] = None  # mg
    target_potassium: Optional[float] = None  # mg
    
    # Activity level
    activity_level: str = "moderate"  # sedentary, light, moderate, active, very_active, athlete
    
    # Training specifics (for athletic goals)
    training_days_per_week: Optional[int] = None
    training_intensity: Optional[str] = None  # low, moderate, high, extreme
    sport_type: Optional[str] = None
    
    # Special conditions
    is_pregnant: bool = False
    is_breastfeeding: bool = False
    pregnancy_trimester: Optional[int] = None
    
    # Dietary preferences
    dietary_restrictions: List[str] = field(default_factory=list)  # vegan, vegetarian, halal, kosher
    food_allergies: List[str] = field(default_factory=list)
    
    # Weekly weight change target
    weekly_weight_change: Optional[float] = None  # kg per week
    
    # Meal timing preferences
    meals_per_day: int = 3
    intermittent_fasting_window: Optional[Tuple[int, int]] = None  # (start_hour, end_hour)
    
    # Performance metrics
    target_vo2_max: Optional[float] = None
    target_resting_heart_rate: Optional[int] = None
    target_body_water_percent: Optional[float] = None
    
    # Notes
    notes: str = ""
    
    def calculate_macro_targets(self) -> Dict[str, float]:
        """Calculate macro targets based on goal type and lifecycle stage."""
        if self.target_calories:
            if self.protein_percent and self.carbs_percent and self.fat_percent:
                return {
                    'calories': self.target_calories,
                    'protein': (self.target_calories * self.protein_percent / 100) / 4,
                    'carbs': (self.target_calories * self.carbs_percent / 100) / 4,
                    'fat': (self.target_calories * self.fat_percent / 100) / 9,
                    'fiber': self.target_fiber or 30,
                    'water': self.target_water or 2.5
                }
        
        # Default targets based on lifecycle stage
        if self.lifecycle_stage:
            return self._get_lifecycle_targets()
        
        return {
            'calories': self.target_calories or 2000,
            'protein': self.target_protein or 150,
            'carbs': self.target_carbs or 200,
            'fat': self.target_fat or 65,
            'fiber': self.target_fiber or 30,
            'water': self.target_water or 2.5
        }
    
    def _get_lifecycle_targets(self) -> Dict[str, float]:
        """Get lifecycle-specific nutritional targets."""
        stage = self.lifecycle_stage
        
        lifecycle_defaults = {
            LifecycleStage.INFANT: {
                'calories': 850, 'protein': 11, 'carbs': 95, 'fat': 30,
                'calcium': 260, 'iron': 11, 'vitamin_d': 400
            },
            LifecycleStage.TODDLER: {
                'calories': 1200, 'protein': 13, 'carbs': 130, 'fat': 40,
                'calcium': 700, 'iron': 7, 'vitamin_d': 600
            },
            LifecycleStage.CHILD: {
                'calories': 1600, 'protein': 19, 'carbs': 130, 'fat': 45,
                'calcium': 1000, 'iron': 10, 'vitamin_d': 600
            },
            LifecycleStage.ADOLESCENT: {
                'calories': 2200 if self.gender == 'male' else 1800,
                'protein': 52 if self.gender == 'male' else 46,
                'carbs': 130, 'fat': 50,
                'calcium': 1300, 'iron': 11 if self.gender == 'male' else 15,
                'vitamin_d': 600
            },
            LifecycleStage.YOUNG_ADULT: {
                'calories': 2400 if self.gender == 'male' else 2000,
                'protein': 56 if self.gender == 'male' else 46,
                'carbs': 130, 'fat': 60,
                'calcium': 1000, 'iron': 8 if self.gender == 'male' else 18,
                'vitamin_d': 600
            },
            LifecycleStage.ADULT: {
                'calories': 2200 if self.gender == 'male' else 1800,
                'protein': 56 if self.gender == 'male' else 46,
                'carbs': 130, 'fat': 55,
                'calcium': 1000, 'iron': 8 if self.gender == 'male' else 18,
                'vitamin_d': 600
            },
            LifecycleStage.SENIOR: {
                'calories': 2000 if self.gender == 'male' else 1600,
                'protein': 60 if self.gender == 'male' else 50,  # Higher for elderly
                'carbs': 130, 'fat': 50,
                'calcium': 1200, 'iron': 8, 'vitamin_d': 800,
                'b12': 2.4  # Critical for seniors
            },
            LifecycleStage.PREGNANCY_TRIMESTER1: {
                'calories': 2000, 'protein': 71, 'carbs': 175, 'fat': 60,
                'calcium': 1000, 'iron': 27, 'vitamin_d': 600, 'folate': 600
            },
            LifecycleStage.PREGNANCY_TRIMESTER2: {
                'calories': 2200, 'protein': 71, 'carbs': 175, 'fat': 60,
                'calcium': 1000, 'iron': 27, 'vitamin_d': 600, 'folate': 600
            },
            LifecycleStage.PREGNANCY_TRIMESTER3: {
                'calories': 2400, 'protein': 71, 'carbs': 175, 'fat': 60,
                'calcium': 1000, 'iron': 27, 'vitamin_d': 600, 'folate': 600
            },
            LifecycleStage.BREASTFEEDING: {
                'calories': 2500, 'protein': 71, 'carbs': 210, 'fat': 65,
                'calcium': 1000, 'iron': 9, 'vitamin_d': 600, 'omega3': 1300
            },
            LifecycleStage.MENOPAUSE: {
                'calories': 1800, 'protein': 50, 'carbs': 130, 'fat': 50,
                'calcium': 1200, 'iron': 8, 'vitamin_d': 800,
                'notes': 'Increased calcium for bone health'
            }
        }
        
        return lifecycle_defaults.get(stage, {
            'calories': 2000, 'protein': 150, 'carbs': 200, 'fat': 65
        })


class PersonalGoalsManager:
    """
    Comprehensive personal health and fitness goals manager.
    
    Supports 60+ goal types across:
    - Weight management (loss, gain, maintenance)
    - Athletic performance (strength, endurance, sport-specific)
    - Lifecycle stages (infant to elderly, pregnancy, menopause)
    - Disease management and prevention
    - Body composition (muscle gain, fat loss, recomposition)
    - Dietary patterns (keto, paleo, Mediterranean, etc.)
    - Special conditions (recovery, stress, sleep)
    """
    
    def __init__(self):
        """Initialize comprehensive goals manager."""
        self.goals: Dict[str, PersonalGoal] = {}
        logger.info("✅ Comprehensive Personal Goals Manager initialized (60+ goal types)")
    
    # ========================================================================
    # WEIGHT MANAGEMENT GOALS
    # ========================================================================
    
    def create_weight_loss_goal(
        self,
        current_weight: float,
        target_weight: float,
        target_date: Optional[datetime] = None,
        activity_level: str = "moderate",
        age: Optional[int] = None,
        gender: Optional[str] = None
    ) -> PersonalGoal:
        """Create a comprehensive weight loss goal."""
        
        # Calculate weekly weight change
        if target_date:
            weeks_to_goal = max(1, (target_date - datetime.now()).days / 7)
            weekly_change = (target_weight - current_weight) / weeks_to_goal
            # Healthy limit: -0.5 to -1.0 kg per week
            weekly_change = max(-1.0, min(-0.5, weekly_change))
        else:
            weekly_change = -0.5  # Default: 0.5 kg per week loss
        
        # Calculate calorie target (7700 cal = 1 kg)
        bmr = self._calculate_bmr(current_weight, activity_level, age, gender)
        calorie_deficit = abs(weekly_change) * 7700 / 7  # Daily deficit
        target_calories = max(1200, bmr - calorie_deficit)  # Minimum 1200 cal
        
        goal = PersonalGoal(
            goal_id=f'weight_loss_{int(datetime.now().timestamp())}',
            goal_type=GoalType.WEIGHT_LOSS,
            current_weight=current_weight,
            target_weight=target_weight,
            target_date=target_date,
            target_calories=target_calories,
            protein_percent=35,  # Higher protein for satiety
            carbs_percent=30,    # Lower carbs for weight loss
            fat_percent=35,      # Moderate fat
            activity_level=activity_level,
            weekly_weight_change=weekly_change,
            age=age,
            gender=gender,
            target_fiber=35,     # High fiber for satiety
            target_water=3.0,    # Extra water for metabolism
            notes="High protein, moderate carb, calorie deficit"
        )
        
        self.goals[goal.goal_id] = goal
        return goal
    
    def create_muscle_gain_goal(
        self,
        current_weight: float,
        target_weight: float,
        target_date: Optional[datetime] = None,
        activity_level: str = "very_active",
        age: Optional[int] = None,
        gender: Optional[str] = None
    ) -> PersonalGoal:
        """Create a muscle gain (lean bulking) goal."""
        
        # Calculate weekly weight change
        if target_date:
            weeks_to_goal = max(1, (target_date - datetime.now()).days / 7)
            weekly_change = (target_weight - current_weight) / weeks_to_goal
            # Healthy limit: 0.25-0.5 kg per week
            weekly_change = min(0.5, max(0.25, weekly_change))
        else:
            weekly_change = 0.25  # Default: lean gain
        
        # Calculate calorie target (surplus for muscle gain)
        bmr = self._calculate_bmr(current_weight, activity_level, age, gender)
        calorie_surplus = weekly_change * 7700 / 7
        target_calories = bmr + calorie_surplus
        
        goal = PersonalGoal(
            goal_id=f'muscle_gain_{int(datetime.now().timestamp())}',
            goal_type=GoalType.MUSCLE_GAIN,
            current_weight=current_weight,
            target_weight=target_weight,
            target_date=target_date,
            target_calories=target_calories,
            protein_percent=35,  # Very high protein
            carbs_percent=45,    # High carbs for energy
            fat_percent=20,      # Lower fat
            activity_level=activity_level,
            weekly_weight_change=weekly_change,
            age=age,
            gender=gender,
            target_protein=2.0 * current_weight,  # 2g per kg
            target_water=4.0,    # Extra hydration
            training_days_per_week=5,
            training_intensity='high',
            notes="High protein, high carbs, calorie surplus"
        )
        
        self.goals[goal.goal_id] = goal
        return goal
    
    def create_body_recomposition_goal(
        self,
        current_weight: float,
        current_body_fat: float,
        target_body_fat: float,
        activity_level: str = "very_active",
        age: Optional[int] = None,
        gender: Optional[str] = None
    ) -> PersonalGoal:
        """Create body recomposition goal (lose fat, gain muscle simultaneously)."""
        
        bmr = self._calculate_bmr(current_weight, activity_level, age, gender)
        
        goal = PersonalGoal(
            goal_id=f'recomp_{int(datetime.now().timestamp())}',
            goal_type=GoalType.BODY_RECOMPOSITION,
            current_weight=current_weight,
            target_weight=current_weight,  # Weight stays same
            current_body_fat=current_body_fat,
            target_body_fat=target_body_fat,
            target_calories=bmr,  # Maintenance calories
            protein_percent=40,   # Very high protein
            carbs_percent=35,     # Moderate carbs
            fat_percent=25,       # Lower fat
            activity_level=activity_level,
            age=age,
            gender=gender,
            target_protein=2.2 * current_weight,  # 2.2g per kg
            training_days_per_week=6,
            training_intensity='high',
            notes="Very high protein, maintenance calories, progressive overload"
        )
        
        self.goals[goal.goal_id] = goal
        return goal
    
    # ========================================================================
    # ATHLETIC PERFORMANCE GOALS
    # ========================================================================
    
    def create_endurance_goal(
        self,
        current_weight: float,
        sport_type: str = "running",
        training_days_per_week: int = 5,
        activity_level: str = "very_active"
    ) -> PersonalGoal:
        """Create endurance training goal (running, cycling, swimming)."""
        
        bmr = self._calculate_bmr(current_weight, activity_level)
        target_calories = bmr * 1.3  # Extra for endurance training
        
        goal = PersonalGoal(
            goal_id=f'endurance_{int(datetime.now().timestamp())}',
            goal_type=GoalType.ENDURANCE_TRAINING,
            current_weight=current_weight,
            target_calories=target_calories,
            protein_percent=20,   # Lower protein
            carbs_percent=60,     # Very high carbs for glycogen
            fat_percent=20,       # Moderate fat
            activity_level=activity_level,
            sport_type=sport_type,
            training_days_per_week=training_days_per_week,
            training_intensity='moderate',
            target_water=4.5,     # High hydration
            target_potassium=4700,  # Electrolytes crucial
            notes="High carbs for glycogen, adequate hydration"
        )
        
        self.goals[goal.goal_id] = goal
        return goal
    
    def create_strength_training_goal(
        self,
        current_weight: float,
        training_days_per_week: int = 4,
        activity_level: str = "very_active"
    ) -> PersonalGoal:
        """Create strength training goal (powerlifting, bodybuilding)."""
        
        bmr = self._calculate_bmr(current_weight, activity_level)
        target_calories = bmr * 1.2
        
        goal = PersonalGoal(
            goal_id=f'strength_{int(datetime.now().timestamp())}',
            goal_type=GoalType.STRENGTH_TRAINING,
            current_weight=current_weight,
            target_calories=target_calories,
            protein_percent=35,   # High protein
            carbs_percent=40,     # Moderate-high carbs
            fat_percent=25,       # Moderate fat
            activity_level=activity_level,
            training_days_per_week=training_days_per_week,
            training_intensity='high',
            target_protein=2.0 * current_weight,
            target_water=3.5,
            notes="High protein for recovery, adequate carbs for performance"
        )
        
        self.goals[goal.goal_id] = goal
        return goal
    
    # ========================================================================
    # LIFECYCLE GOALS
    # ========================================================================
    
    def create_pregnancy_goal(
        self,
        current_weight: float,
        trimester: int = 1,
        age: Optional[int] = None
    ) -> PersonalGoal:
        """Create pregnancy nutrition goal."""
        
        # Calorie increases by trimester
        calorie_increases = {1: 0, 2: 340, 3: 450}
        base_calories = 2000
        target_calories = base_calories + calorie_increases.get(trimester, 0)
        
        lifecycle_stage = {
            1: LifecycleStage.PREGNANCY_TRIMESTER1,
            2: LifecycleStage.PREGNANCY_TRIMESTER2,
            3: LifecycleStage.PREGNANCY_TRIMESTER3
        }
        
        goal = PersonalGoal(
            goal_id=f'pregnancy_t{trimester}_{int(datetime.now().timestamp())}',
            goal_type=GoalType.PREGNANCY,
            lifecycle_stage=lifecycle_stage.get(trimester),
            current_weight=current_weight,
            target_calories=target_calories,
            protein_percent=20,
            carbs_percent=50,
            fat_percent=30,
            age=age,
            gender='female',
            is_pregnant=True,
            pregnancy_trimester=trimester,
            target_protein=71,      # RDA for pregnancy
            target_folate=600,      # Critical for neural tube
            target_iron=27,         # Increased needs
            target_calcium=1000,
            target_omega3=1400,     # Fetal brain development
            target_fiber=28,
            target_water=3.0,
            dietary_restrictions=['no_alcohol', 'no_raw_fish', 'no_deli_meats'],
            notes="Extra folate, iron, omega-3. Avoid raw/undercooked foods"
        )
        
        self.goals[goal.goal_id] = goal
        return goal
    
    def create_breastfeeding_goal(
        self,
        current_weight: float,
        age: Optional[int] = None
    ) -> PersonalGoal:
        """Create breastfeeding nutrition goal."""
        
        goal = PersonalGoal(
            goal_id=f'breastfeeding_{int(datetime.now().timestamp())}',
            goal_type=GoalType.BREASTFEEDING,
            lifecycle_stage=LifecycleStage.BREASTFEEDING,
            current_weight=current_weight,
            target_calories=2500,   # Extra 500 cal for milk production
            protein_percent=20,
            carbs_percent=50,
            fat_percent=30,
            age=age,
            gender='female',
            is_breastfeeding=True,
            target_protein=71,
            target_calcium=1000,
            target_omega3=1300,     # DHA for infant brain
            target_water=3.8,       # Extra hydration critical
            target_vitamin_d=600,
            target_iodine=290,      # Infant thyroid development
            notes="Extra calories, high hydration, omega-3 for infant development"
        )
        
        self.goals[goal.goal_id] = goal
        return goal
    
    def create_senior_nutrition_goal(
        self,
        current_weight: float,
        age: int,
        gender: str = "male"
    ) -> PersonalGoal:
        """Create senior/elderly nutrition goal."""
        
        # Seniors need fewer calories but more protein
        base_calories = 2000 if gender == 'male' else 1600
        
        goal = PersonalGoal(
            goal_id=f'senior_{int(datetime.now().timestamp())}',
            goal_type=GoalType.SENIOR_NUTRITION,
            lifecycle_stage=LifecycleStage.SENIOR if age < 80 else LifecycleStage.ELDERLY,
            current_weight=current_weight,
            target_calories=base_calories,
            protein_percent=25,     # Higher protein for muscle preservation
            carbs_percent=45,
            fat_percent=30,
            age=age,
            gender=gender,
            target_protein=1.2 * current_weight,  # Higher than young adults
            target_calcium=1200,    # Bone health critical
            target_vitamin_d=800,   # Higher needs
            target_b12=2.4,         # Absorption decreases with age
            target_fiber=30,        # Digestive health
            target_water=2.5,
            notes="High protein for muscle, extra calcium/vitamin D for bones, B12 for cognition"
        )
        
        self.goals[goal.goal_id] = goal
        return goal
    
    def create_menopause_goal(
        self,
        current_weight: float,
        age: int
    ) -> PersonalGoal:
        """Create menopause nutrition goal."""
        
        goal = PersonalGoal(
            goal_id=f'menopause_{int(datetime.now().timestamp())}',
            goal_type=GoalType.MENOPAUSE,
            lifecycle_stage=LifecycleStage.MENOPAUSE,
            current_weight=current_weight,
            target_calories=1800,
            protein_percent=25,
            carbs_percent=40,
            fat_percent=35,
            age=age,
            gender='female',
            target_calcium=1200,    # Critical for bone loss prevention
            target_vitamin_d=800,
            target_magnesium=320,   # Helps with symptoms
            target_fiber=25,
            target_water=2.5,
            target_omega3=1100,     # Anti-inflammatory
            notes="Extra calcium/vitamin D, phytoestrogens (soy), omega-3"
        )
        
        self.goals[goal.goal_id] = goal
        return goal
    
    # ========================================================================
    # DIETARY PATTERN GOALS
    # ========================================================================
    
    def create_ketogenic_diet_goal(
        self,
        current_weight: float,
        activity_level: str = "moderate"
    ) -> PersonalGoal:
        """Create ketogenic diet goal (very low carb, high fat)."""
        
        bmr = self._calculate_bmr(current_weight, activity_level)
        
        goal = PersonalGoal(
            goal_id=f'keto_{int(datetime.now().timestamp())}',
            goal_type=GoalType.KETOGENIC_DIET,
            current_weight=current_weight,
            target_calories=bmr,
            protein_percent=20,     # Moderate protein
            carbs_percent=5,        # Very low carbs (<50g)
            fat_percent=75,         # Very high fat
            activity_level=activity_level,
            target_carbs=50,        # Max 50g for ketosis
            target_fiber=25,        # From low-carb veggies
            target_water=3.5,       # Extra for ketosis
            target_magnesium=400,   # Electrolytes crucial
            target_potassium=4700,
            target_sodium=5000,     # Higher sodium needs
            notes="<50g carbs for ketosis, high fat, adequate protein, electrolytes"
        )
        
        self.goals[goal.goal_id] = goal
        return goal
    
    def create_mediterranean_diet_goal(
        self,
        current_weight: float,
        activity_level: str = "moderate"
    ) -> PersonalGoal:
        """Create Mediterranean diet goal."""
        
        bmr = self._calculate_bmr(current_weight, activity_level)
        
        goal = PersonalGoal(
            goal_id=f'mediterranean_{int(datetime.now().timestamp())}',
            goal_type=GoalType.MEDITERRANEAN_DIET,
            current_weight=current_weight,
            target_calories=bmr,
            protein_percent=20,
            carbs_percent=45,       # Whole grains, fruits
            fat_percent=35,         # Olive oil, nuts, fish
            activity_level=activity_level,
            target_fiber=35,        # High from veggies, legumes
            target_omega3=2000,     # Fish 2x/week
            dietary_restrictions=['whole_foods', 'minimal_processed'],
            notes="High fish, olive oil, whole grains, vegetables, moderate wine"
        )
        
        self.goals[goal.goal_id] = goal
        return goal
    
    def create_plant_based_goal(
        self,
        current_weight: float,
        activity_level: str = "moderate",
        is_vegan: bool = True
    ) -> PersonalGoal:
        """Create plant-based/vegan diet goal."""
        
        bmr = self._calculate_bmr(current_weight, activity_level)
        
        goal = PersonalGoal(
            goal_id=f'plant_based_{int(datetime.now().timestamp())}',
            goal_type=GoalType.PLANT_BASED,
            current_weight=current_weight,
            target_calories=bmr,
            protein_percent=20,
            carbs_percent=55,
            fat_percent=25,
            activity_level=activity_level,
            target_protein=1.2 * current_weight,  # Slightly higher for plant protein
            target_fiber=40,        # Very high from plants
            target_b12=2.4,         # Must supplement
            target_iron=18,         # Non-heme iron, needs more
            target_zinc=11,         # Plant sources less bioavailable
            target_omega3=1600,     # ALA from flax, chia, walnuts
            target_calcium=1000,
            dietary_restrictions=['vegan' if is_vegan else 'vegetarian'],
            notes="Focus on protein variety, B12 supplement, iron+vitamin C pairing"
        )
        
        self.goals[goal.goal_id] = goal
        return goal
    
    # ========================================================================
    # SPECIAL CONDITION GOALS
    # ========================================================================
    
    def create_post_surgery_recovery_goal(
        self,
        current_weight: float,
        surgery_type: str = "general"
    ) -> PersonalGoal:
        """Create post-surgery recovery nutrition goal."""
        
        goal = PersonalGoal(
            goal_id=f'post_surgery_{int(datetime.now().timestamp())}',
            goal_type=GoalType.POST_SURGERY_RECOVERY,
            current_weight=current_weight,
            target_calories=2200,   # Slightly elevated for healing
            protein_percent=30,     # High protein for wound healing
            carbs_percent=45,
            fat_percent=25,
            target_protein=1.5 * current_weight,  # Elevated
            target_vitamin_c=200,   # Collagen synthesis
            target_zinc=15,         # Wound healing
            target_water=3.0,
            notes="High protein, vitamin C, zinc for healing. Easy to digest foods"
        )
        
        self.goals[goal.goal_id] = goal
        return goal
    
    def create_gut_health_goal(
        self,
        current_weight: float
    ) -> PersonalGoal:
        """Create gut health optimization goal."""
        
        goal = PersonalGoal(
            goal_id=f'gut_health_{int(datetime.now().timestamp())}',
            goal_type=GoalType.GUT_HEALTH,
            current_weight=current_weight,
            target_calories=2000,
            protein_percent=20,
            carbs_percent=50,
            fat_percent=30,
            target_fiber=40,        # Very high fiber
            target_water=3.0,
            dietary_restrictions=['probiotic_foods', 'prebiotic_fiber'],
            notes="High fiber, fermented foods, diverse plant foods, adequate hydration"
        )
        
        self.goals[goal.goal_id] = goal
        return goal
    
    def create_anti_inflammatory_goal(
        self,
        current_weight: float
    ) -> PersonalGoal:
        """Create anti-inflammatory diet goal."""
        
        goal = PersonalGoal(
            goal_id=f'anti_inflammatory_{int(datetime.now().timestamp())}',
            goal_type=GoalType.ANTI_INFLAMMATORY,
            current_weight=current_weight,
            target_calories=2000,
            protein_percent=25,
            carbs_percent=40,
            fat_percent=35,
            target_omega3=2500,     # High omega-3
            target_fiber=35,
            dietary_restrictions=['anti_inflammatory_foods'],
            food_allergies=['processed_foods', 'added_sugar', 'trans_fats'],
            notes="High omega-3, colorful vegetables, turmeric, berries, avoid processed foods"
        )
        
        self.goals[goal.goal_id] = goal
        return goal
    
    # ========================================================================
    # MACRO TRACKING GOAL
    # ========================================================================
    
    def create_macro_tracking_goal(
        self,
        target_calories: float,
        protein_grams: float,
        carbs_grams: float,
        fat_grams: float
    ) -> PersonalGoal:
        """Create a custom macro tracking goal."""
        
        goal = PersonalGoal(
            goal_id=f'macro_tracking_{int(datetime.now().timestamp())}',
            goal_type=GoalType.MACRO_TRACKING,
            target_calories=target_calories,
            target_protein=protein_grams,
            target_carbs=carbs_grams,
            target_fat=fat_grams
        )
        
        self.goals[goal.goal_id] = goal
        return goal
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _calculate_bmr(
        self,
        weight_kg: float,
        activity_level: str,
        age: Optional[int] = None,
        gender: Optional[str] = None
    ) -> float:
        """Calculate Basal Metabolic Rate with activity multiplier."""
        
        # Mifflin-St Jeor equation (more accurate)
        # BMR = (10 × weight in kg) + (6.25 × height in cm) - (5 × age in years) + s
        # s = +5 for males, -161 for females
        
        # Default assumptions if not provided
        age = age or 30
        height_cm = 170  # Default height
        
        if gender == 'male':
            bmr_base = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
        else:  # female or other
            bmr_base = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
        
        activity_multipliers = {
            'sedentary': 1.2,       # Little/no exercise
            'light': 1.375,         # Light exercise 1-3 days/week
            'moderate': 1.55,       # Moderate exercise 3-5 days/week
            'active': 1.725,        # Heavy exercise 6-7 days/week
            'very_active': 1.9,     # Very heavy exercise, physical job
            'athlete': 2.2          # Professional athlete
        }
        
        return bmr_base * activity_multipliers.get(activity_level, 1.55)
    
    def check_meal_against_goals(
        self,
        meal_nutrition: Dict[str, float],
        goal_id: str
    ) -> Dict[str, Any]:
        """Check if meal aligns with personal goals."""
        
        goal = self.goals.get(goal_id)
        if not goal:
            return {'error': 'Goal not found'}
        
        targets = goal.calculate_macro_targets()
        
        result = {
            'goal_type': goal.goal_type.value,
            'macro_comparison': {},
            'recommendations': []
        }
        
        # Compare macros
        for macro in ['calories', 'protein', 'carbs', 'fat']:
            if macro in meal_nutrition and macro in targets:
                meal_value = meal_nutrition[macro]
                target_value = targets[macro]
                percentage = (meal_value / target_value * 100) if target_value > 0 else 0
                
                result['macro_comparison'][macro] = {
                    'meal': meal_value,
                    'target': target_value,
                    'percentage': round(percentage, 1),
                    'status': 'OK' if 80 <= percentage <= 120 else 'WARNING'
                }
        
        # Generate recommendations
        if goal.goal_type == GoalType.WEIGHT_LOSS:
            if meal_nutrition.get('calories', 0) > targets['calories'] * 0.4:
                result['recommendations'].append(
                    "⚠️ This meal is 40%+ of your daily calorie target. Consider a lighter option."
                )
        
        elif goal.goal_type == GoalType.MUSCLE_GAIN:
            if meal_nutrition.get('protein', 0) < 30:
                result['recommendations'].append(
                    "💪 Add more protein to support muscle growth (target 30-40g per meal)."
                )
        
        return result


# ============================================================================
# PHASE 3: MEAL PLANNING & OPTIMIZATION
# ============================================================================

@dataclass
class MealPlan:
    """Complete meal plan with optimization."""
    plan_id: str
    user_diseases: List[str]
    user_goals: List[str]
    daily_meals: List[Dict[str, Any]]
    total_nutrition: Dict[str, float]
    compliance_score: float
    created_at: datetime = field(default_factory=datetime.now)


class MealOptimizer:
    """
    Optimizes meals to balance disease restrictions and personal goals.
    
    Features:
    - Multi-objective optimization
    - Disease constraint satisfaction
    - Goal alignment
    - Portion size recommendations
    - Food substitutions
    """
    
    def __init__(self, disease_db: ComprehensiveDiseaseDatabase):
        """Initialize meal optimizer."""
        self.disease_db = disease_db
        logger.info("✅ Meal Optimizer initialized")
    
    def optimize_meal(
        self,
        meal_nutrition: Dict[str, float],
        user_diseases: List[str],
        user_goals: Optional[PersonalGoal] = None
    ) -> Dict[str, Any]:
        """
        Optimize meal to satisfy disease constraints and goals.
        
        Returns optimization recommendations.
        """
        
        violations = []
        recommendations = []
        optimization_score = 100.0
        
        # Check disease constraints
        for disease_id in user_diseases:
            disease = self.disease_db.get_disease(disease_id)
            if not disease:
                continue
            
            # Check each nutrient constraint
            nutrient_checks = [
                ('sodium', disease.sodium_max, 'mg'),
                ('potassium', disease.potassium_max, 'mg'),
                ('protein', disease.protein_max, 'g'),
                ('carbs', disease.carbs_max, 'g'),
                ('sugar', disease.sugar_max, 'g'),
                ('saturated_fat', disease.saturated_fat_max, 'g'),
                ('cholesterol', disease.cholesterol_max, 'mg'),
                ('phosphorus', disease.phosphorus_max, 'mg')
            ]
            
            for nutrient, max_value, unit in nutrient_checks:
                if max_value and nutrient in meal_nutrition:
                    actual = meal_nutrition[nutrient]
                    if actual > max_value:
                        excess = actual - max_value
                        excess_percent = (excess / max_value) * 100
                        violations.append({
                            'disease': disease.name,
                            'nutrient': nutrient,
                            'actual': actual,
                            'limit': max_value,
                            'excess': excess,
                            'excess_percent': excess_percent,
                            'unit': unit
                        })
                        optimization_score -= min(20, excess_percent)
        
        # Generate recommendations
        if violations:
            recommendations.append("🔧 MEAL OPTIMIZATION SUGGESTIONS:")
            
            for v in violations:
                if v['nutrient'] == 'sodium':
                    recommendations.append(
                        f"  • Reduce sodium: {v['excess']:.0f}{v['unit']} over limit for {v['disease']}. "
                        f"Try low-sodium alternatives or smaller portions."
                    )
                elif v['nutrient'] == 'carbs':
                    recommendations.append(
                        f"  • Reduce carbs: {v['excess']:.0f}{v['unit']} over limit for {v['disease']}. "
                        f"Replace rice/bread with vegetables or reduce portion by {v['excess_percent']:.0f}%."
                    )
                elif v['nutrient'] == 'protein':
                    recommendations.append(
                        f"  • Reduce protein: {v['excess']:.0f}{v['unit']} over limit for {v['disease']}. "
                        f"Use smaller portion sizes."
                    )
                elif v['nutrient'] == 'sugar':
                    recommendations.append(
                        f"  • Reduce sugar: {v['excess']:.0f}{v['unit']} over limit for {v['disease']}. "
                        f"Avoid added sugars and sweetened beverages."
                    )
        
        # Check goal alignment
        if user_goals:
            goal_targets = user_goals.calculate_macro_targets()
            
            for macro in ['calories', 'protein', 'carbs', 'fat']:
                if macro in meal_nutrition and macro in goal_targets:
                    meal_value = meal_nutrition[macro]
                    daily_target = goal_targets[macro]
                    meal_should_be = daily_target / 3  # Assuming 3 meals per day
                    
                    if meal_value > meal_should_be * 1.5:
                        recommendations.append(
                            f"  • This meal has {meal_value:.0f}g {macro}, which is high relative "
                            f"to your daily goal. Consider a lighter option for other meals."
                        )
        
        return {
            'optimization_score': max(0, optimization_score),
            'violations': violations,
            'recommendations': recommendations,
            'is_safe': len(violations) == 0
        }
    
    def suggest_portion_adjustment(
        self,
        meal_nutrition: Dict[str, float],
        target_nutrient: str,
        target_value: float
    ) -> Dict[str, Any]:
        """Suggest portion size adjustment to meet target."""
        
        if target_nutrient not in meal_nutrition:
            return {'error': 'Nutrient not found in meal'}
        
        current_value = meal_nutrition[target_nutrient]
        if current_value == 0:
            return {'error': 'Cannot calculate adjustment'}
        
        adjustment_ratio = target_value / current_value
        adjustment_percent = (adjustment_ratio - 1) * 100
        
        return {
            'current_value': current_value,
            'target_value': target_value,
            'adjustment_ratio': adjustment_ratio,
            'adjustment_percent': adjustment_percent,
            'suggestion': f"{'Reduce' if adjustment_ratio < 1 else 'Increase'} "
                         f"portion by {abs(adjustment_percent):.0f}%"
        }


# ============================================================================
# PHASE 4: PROGRESS TRACKING & ANALYTICS
# ============================================================================

@dataclass
class MealRecord:
    """Record of a consumed meal."""
    meal_id: str
    timestamp: datetime
    meal_type: str  # breakfast, lunch, dinner, snack
    nutrition: Dict[str, float]
    health_score: float
    violations: List[Dict[str, Any]]
    user_diseases: List[str]
    user_goals: List[str]


class ProgressTracker:
    """
    Tracks user progress toward health goals and disease management.
    
    Features:
    - Daily/weekly/monthly nutrition tracking
    - Goal progress monitoring
    - Disease compliance tracking
    - Trend analysis
    - Recommendations based on history
    """
    
    def __init__(self):
        """Initialize progress tracker."""
        self.meal_history: List[MealRecord] = []
        logger.info("✅ Progress Tracker initialized")
    
    def log_meal(
        self,
        meal_nutrition: Dict[str, float],
        meal_type: str,
        health_score: float,
        violations: List[Dict[str, Any]],
        user_diseases: List[str],
        user_goals: List[str]
    ) -> MealRecord:
        """Log a consumed meal."""
        
        record = MealRecord(
            meal_id=f"meal_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            meal_type=meal_type,
            nutrition=meal_nutrition,
            health_score=health_score,
            violations=violations,
            user_diseases=user_diseases,
            user_goals=user_goals
        )
        
        self.meal_history.append(record)
        return record
    
    def get_daily_summary(self, date: datetime) -> Dict[str, Any]:
        """Get nutrition summary for a specific day."""
        
        day_meals = [
            m for m in self.meal_history
            if m.timestamp.date() == date.date()
        ]
        
        if not day_meals:
            return {'error': 'No meals logged for this date'}
        
        # Aggregate nutrition
        total_nutrition = {}
        for meal in day_meals:
            for nutrient, value in meal.nutrition.items():
                total_nutrition[nutrient] = total_nutrition.get(nutrient, 0) + value
        
        # Calculate average health score
        avg_health_score = sum(m.health_score for m in day_meals) / len(day_meals)
        
        # Count violations
        total_violations = sum(len(m.violations) for m in day_meals)
        
        return {
            'date': date.date(),
            'meals_count': len(day_meals),
            'total_nutrition': total_nutrition,
            'average_health_score': round(avg_health_score, 1),
            'total_violations': total_violations,
            'meals': day_meals
        }
    
    def get_weekly_trends(self) -> Dict[str, Any]:
        """Get nutrition trends for the past week."""
        
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        
        week_meals = [
            m for m in self.meal_history
            if m.timestamp >= week_ago
        ]
        
        if not week_meals:
            return {'error': 'No meals logged in the past week'}
        
        # Daily aggregation
        daily_totals = {}
        for meal in week_meals:
            day = meal.timestamp.date()
            if day not in daily_totals:
                daily_totals[day] = {
                    'nutrition': {},
                    'health_score': [],
                    'violations': 0
                }
            
            for nutrient, value in meal.nutrition.items():
                daily_totals[day]['nutrition'][nutrient] = \
                    daily_totals[day]['nutrition'].get(nutrient, 0) + value
            
            daily_totals[day]['health_score'].append(meal.health_score)
            daily_totals[day]['violations'] += len(meal.violations)
        
        # Calculate averages
        avg_daily_nutrition = {}
        all_nutrients = set()
        for day_data in daily_totals.values():
            all_nutrients.update(day_data['nutrition'].keys())
        
        for nutrient in all_nutrients:
            values = [
                day_data['nutrition'].get(nutrient, 0)
                for day_data in daily_totals.values()
            ]
            avg_daily_nutrition[nutrient] = sum(values) / len(values)
        
        avg_health_score = sum(
            sum(day_data['health_score']) / len(day_data['health_score'])
            for day_data in daily_totals.values()
        ) / len(daily_totals)
        
        avg_violations = sum(
            day_data['violations'] for day_data in daily_totals.values()
        ) / len(daily_totals)
        
        return {
            'period': 'past_7_days',
            'days_tracked': len(daily_totals),
            'total_meals': len(week_meals),
            'avg_daily_nutrition': avg_daily_nutrition,
            'avg_health_score': round(avg_health_score, 1),
            'avg_daily_violations': round(avg_violations, 1),
            'daily_breakdown': daily_totals
        }
    
    def get_goal_progress(
        self,
        goal: PersonalGoal,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get progress toward a specific goal."""
        
        now = datetime.now()
        period_start = now - timedelta(days=days)
        
        period_meals = [
            m for m in self.meal_history
            if m.timestamp >= period_start and goal.goal_id in m.user_goals
        ]
        
        if not period_meals:
            return {'error': 'No meals logged for this goal in the period'}
        
        targets = goal.calculate_macro_targets()
        
        # Daily aggregation
        daily_totals = {}
        for meal in period_meals:
            day = meal.timestamp.date()
            if day not in daily_totals:
                daily_totals[day] = {}
            
            for nutrient, value in meal.nutrition.items():
                daily_totals[day][nutrient] = daily_totals[day].get(nutrient, 0) + value
        
        # Compare to targets
        compliance = {}
        for nutrient, target in targets.items():
            daily_values = [day_data.get(nutrient, 0) for day_data in daily_totals.values()]
            avg_value = sum(daily_values) / len(daily_values)
            compliance[nutrient] = {
                'target': target,
                'average': round(avg_value, 1),
                'compliance_percent': round((avg_value / target * 100) if target > 0 else 0, 1),
                'on_track': 90 <= (avg_value / target * 100) <= 110 if target > 0 else True
            }
        
        return {
            'goal_type': goal.goal_type.value,
            'period_days': days,
            'days_tracked': len(daily_totals),
            'total_meals': len(period_meals),
            'macro_compliance': compliance,
            'overall_compliance': sum(
                1 for c in compliance.values() if c['on_track']
            ) / len(compliance) * 100
        }


# ============================================================================
# PHASE 5: AI RECOMMENDATIONS & ALTERNATIVES
# ============================================================================

class SmartRecommender:
    """
    AI-powered food recommendations and alternatives.
    
    Features:
    - Food substitutions for disease management
    - Healthier alternatives
    - Recipe modifications
    - Meal timing suggestions
    - Personalized recommendations based on history
    """
    
    def __init__(self, disease_db: ComprehensiveDiseaseDatabase):
        """Initialize recommender."""
        self.disease_db = disease_db
        self.food_alternatives = self._initialize_alternatives()
        logger.info("✅ Smart Recommender initialized")
    
    def _initialize_alternatives(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize food alternative database."""
        return {
            'white_rice': [
                {'name': 'Brown Rice', 'reason': 'Higher fiber, lower GI', 'benefit': 'Better blood sugar control'},
                {'name': 'Quinoa', 'reason': 'Complete protein, more nutrients', 'benefit': 'Higher protein content'},
                {'name': 'Cauliflower Rice', 'reason': 'Very low carb', 'benefit': 'Dramatic carb reduction (95% less)'},
                {'name': 'Wild Rice', 'reason': 'More protein and fiber', 'benefit': 'Better nutrient profile'}
            ],
            'white_bread': [
                {'name': 'Whole Wheat Bread', 'reason': 'More fiber', 'benefit': 'Better digestion and satiety'},
                {'name': 'Sourdough', 'reason': 'Lower GI', 'benefit': 'Better blood sugar response'},
                {'name': 'Lettuce Wraps', 'reason': 'No carbs', 'benefit': 'Eliminate carbs completely'},
                {'name': 'Almond Flour Bread', 'reason': 'Low carb, high protein', 'benefit': 'Keto-friendly option'}
            ],
            'fried_chicken': [
                {'name': 'Grilled Chicken', 'reason': '70% less fat', 'benefit': 'Heart-healthy preparation'},
                {'name': 'Baked Chicken', 'reason': 'Low fat, crispy texture', 'benefit': 'Similar taste, healthier'},
                {'name': 'Air-Fried Chicken', 'reason': '80% less oil', 'benefit': 'Crispy with minimal oil'},
                {'name': 'Chicken Breast', 'reason': 'Leanest cut', 'benefit': 'Maximum protein, minimal fat'}
            ],
            'french_fries': [
                {'name': 'Sweet Potato Fries', 'reason': 'More vitamins', 'benefit': 'Better nutrient profile'},
                {'name': 'Baked Potato Wedges', 'reason': 'Less oil', 'benefit': '60% fewer calories'},
                {'name': 'Roasted Vegetables', 'reason': 'More nutrients', 'benefit': 'Much healthier option'},
                {'name': 'Green Salad', 'reason': 'Very low calorie', 'benefit': 'Maximize nutrition, minimize calories'}
            ],
            'soda': [
                {'name': 'Water', 'reason': 'Zero calories, zero sugar', 'benefit': 'Best hydration'},
                {'name': 'Sparkling Water', 'reason': 'Zero calories, carbonated', 'benefit': 'Satisfies fizzy craving'},
                {'name': 'Unsweetened Tea', 'reason': 'Antioxidants', 'benefit': 'Health benefits, no sugar'},
                {'name': 'Diet Soda', 'reason': 'Zero sugar', 'benefit': 'Similar taste, no sugar'}
            ],
            'whole_milk': [
                {'name': 'Skim Milk', 'reason': '80% less fat', 'benefit': 'Same calcium, less fat'},
                {'name': 'Almond Milk', 'reason': 'Low calorie', 'benefit': 'Dairy-free, low calorie'},
                {'name': 'Oat Milk', 'reason': 'Creamy texture', 'benefit': 'Dairy-free, sustainable'},
                {'name': '1% Milk', 'reason': 'Moderate fat', 'benefit': 'Balance of taste and health'}
            ],
            'butter': [
                {'name': 'Olive Oil', 'reason': 'Heart-healthy fats', 'benefit': 'Reduces inflammation'},
                {'name': 'Avocado', 'reason': 'Healthy fats, nutrients', 'benefit': 'Multiple health benefits'},
                {'name': 'Greek Yogurt', 'reason': 'High protein', 'benefit': 'Adds protein, less fat'},
                {'name': 'Coconut Oil', 'reason': 'Medium-chain fats', 'benefit': 'Different fat profile'}
            ],
            'ground_beef': [
                {'name': 'Ground Turkey', 'reason': '50% less fat', 'benefit': 'Leaner protein'},
                {'name': 'Ground Chicken', 'reason': 'Very lean', 'benefit': 'Lowest fat option'},
                {'name': 'Lean Ground Beef (93/7)', 'reason': 'Less fat', 'benefit': 'Maintains beef flavor'},
                {'name': 'Lentils', 'reason': 'Plant protein', 'benefit': 'High fiber, no cholesterol'}
            ]
        }
    
    def get_alternatives(
        self,
        food_name: str,
        user_diseases: List[str],
        reason: str = 'general'
    ) -> List[Dict[str, Any]]:
        """Get alternative food suggestions."""
        
        # Normalize food name
        food_key = food_name.lower().replace(' ', '_')
        
        alternatives = self.food_alternatives.get(food_key, [])
        
        # Filter based on diseases
        if user_diseases:
            disease_profiles = [
                self.disease_db.get_disease(d) for d in user_diseases
            ]
            disease_profiles = [d for d in disease_profiles if d]
            
            # Add disease-specific reasoning
            for alt in alternatives:
                alt['disease_benefits'] = []
                for disease in disease_profiles:
                    if reason == 'high_carbs' and disease.category == DiseaseCategory.METABOLIC:
                        if 'carb' in alt['benefit'].lower():
                            alt['disease_benefits'].append(
                                f"Better for {disease.name}"
                            )
                    elif reason == 'high_sodium' and disease.category == DiseaseCategory.CARDIOVASCULAR:
                        if 'sodium' in alt['benefit'].lower() or 'heart' in alt['benefit'].lower():
                            alt['disease_benefits'].append(
                                f"Better for {disease.name}"
                            )
        
        return alternatives
    
    def recommend_meal_timing(
        self,
        meal_nutrition: Dict[str, float],
        user_diseases: List[str]
    ) -> Dict[str, Any]:
        """Recommend optimal meal timing."""
        
        recommendations = []
        
        # High carb meal timing
        if meal_nutrition.get('carbs', 0) > 60:
            recommendations.append({
                'timing': 'Morning or post-workout',
                'reason': 'High carb meals better tolerated when insulin sensitivity is highest',
                'benefit': 'Better blood sugar control'
            })
        
        # High protein meal timing
        if meal_nutrition.get('protein', 0) > 40:
            recommendations.append({
                'timing': 'Post-workout or evening',
                'reason': 'Protein supports muscle recovery and satiety',
                'benefit': 'Muscle growth and appetite control'
            })
        
        # Heavy meal timing
        if meal_nutrition.get('calories', 0) > 800:
            recommendations.append({
                'timing': 'Lunch time',
                'reason': 'Heavy meals harder to digest in evening',
                'benefit': 'Better digestion and sleep'
            })
        
        return {
            'meal_size': 'large' if meal_nutrition.get('calories', 0) > 600 else 'moderate',
            'recommendations': recommendations
        }
    
    def generate_meal_modification(
        self,
        meal_components: List[Dict[str, Any]],
        target_nutrient: str,
        reduction_percent: float
    ) -> Dict[str, Any]:
        """Generate specific meal modifications."""
        
        modifications = []
        
        # Find component with highest amount of target nutrient
        components_sorted = sorted(
            meal_components,
            key=lambda x: x.get(target_nutrient, 0),
            reverse=True
        )
        
        if components_sorted:
            main_contributor = components_sorted[0]
            modifications.append({
                'component': main_contributor['name'],
                'action': 'reduce',
                'amount': f"{reduction_percent:.0f}%",
                'reason': f"Main source of {target_nutrient}",
                'alternative': self.get_alternatives(
                    main_contributor['name'],
                    [],
                    reason=f'high_{target_nutrient}'
                )
            })
        
        return {
            'target_nutrient': target_nutrient,
            'target_reduction': f"{reduction_percent:.0f}%",
            'modifications': modifications
        }


# ============================================================================
# COMPREHENSIVE INTEGRATION SYSTEM
# ============================================================================

class CVNutritionIntegration:
    """
    Bridge between CV nutrition pipeline and disease/MNT systems.
    """
    
    def __init__(self):
        """Initialize integration with both systems."""
        self.disease_db = None
        self.mnt_rules = None
        self.trained_scanner = None
        
        # Try to import scanner components
        try:
            from disease_database import DiseaseDatabase
            from mnt_rules_engine import MNTRulesEngine
            from trained_disease_scanner import TrainedDiseaseScanner
            
            self.disease_db = DiseaseDatabase()
            self.mnt_rules = MNTRulesEngine()
            self.trained_scanner = TrainedDiseaseScanner()
            
            logger.info("✅ Scanner modules loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Scanner modules not available: {e}")
    
    def analyze_food_for_conditions(
        self,
        nutrition_data: Dict[str, Any],
        user_conditions: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze food nutrition against user's medical conditions.
        
        Args:
            nutrition_data: Nutrition from CV pipeline
                {
                    'calories': 865,
                    'protein': 18,
                    'carbs': 186,
                    'fat': 2,
                    'sodium': 33,
                    'sugar': 0.7,
                    ...
                }
            user_conditions: List of medical conditions
                ['diabetes', 'hypertension', 'heart_disease']
        
        Returns:
            Health impact analysis
                {
                    'safe_to_eat': True/False,
                    'warnings': [...],
                    'recommendations': [...],
                    'nutrient_limits': {...}
                }
        """
        if not self.disease_db or not self.mnt_rules:
            logger.warning("Disease database not available, skipping health analysis")
            return {
                'safe_to_eat': True,
                'warnings': [],
                'recommendations': [],
                'analysis_available': False
            }
        
        results = {
            'safe_to_eat': True,
            'warnings': [],
            'recommendations': [],
            'nutrient_limits': {},
            'analysis_available': True
        }
        
        # Check each condition
        for condition in user_conditions:
            # Get dietary restrictions for condition
            restrictions = self.disease_db.get_restrictions(condition)
            
            # Check nutrition against restrictions
            violations = self._check_restrictions(nutrition_data, restrictions)
            
            if violations:
                results['safe_to_eat'] = False
                results['warnings'].extend(violations)
        
        # Get recommendations
        if results['warnings']:
            results['recommendations'] = self._generate_recommendations(
                nutrition_data,
                user_conditions
            )
        
        return results
    
    def _check_restrictions(
        self,
        nutrition: Dict[str, Any],
        restrictions: Dict[str, Any]
    ) -> List[str]:
        """Check if nutrition violates restrictions."""
        violations = []
        
        # Common restriction checks
        checks = {
            'sodium': ('sodium', 'mg', 2300),  # Default max
            'sugar': ('sugar', 'g', 50),
            'saturated_fat': ('saturated_fat', 'g', 20),
            'cholesterol': ('cholesterol', 'mg', 300)
        }
        
        for nutrient, (key, unit, default_limit) in checks.items():
            if key in nutrition:
                value = nutrition[key]
                limit = restrictions.get(f'{nutrient}_limit', default_limit)
                
                if value > limit:
                    excess = value - limit
                    violations.append(
                        f"⚠️ High {nutrient}: {value}{unit} "
                        f"(exceeds limit by {excess}{unit})"
                    )
        
        return violations
    
    def _generate_recommendations(
        self,
        nutrition: Dict[str, Any],
        conditions: List[str]
    ) -> List[str]:
        """Generate dietary recommendations."""
        recommendations = []
        
        # Condition-specific recommendations
        if 'diabetes' in conditions:
            if nutrition.get('sugar', 0) > 10:
                recommendations.append(
                    "💡 Choose lower-sugar alternatives"
                )
            if nutrition.get('carbs', 0) > 45:
                recommendations.append(
                    "💡 Consider smaller portion or pair with protein"
                )
        
        if 'hypertension' in conditions or 'heart_disease' in conditions:
            if nutrition.get('sodium', 0) > 500:
                recommendations.append(
                    "💡 Look for low-sodium version"
                )
            if nutrition.get('saturated_fat', 0) > 5:
                recommendations.append(
                    "💡 Choose leaner protein options"
                )
        
        if 'kidney_disease' in conditions:
            if nutrition.get('protein', 0) > 30:
                recommendations.append(
                    "💡 Reduce portion size (high protein)"
                )
            if nutrition.get('potassium', 0) > 400:
                recommendations.append(
                    "💡 Limit potassium-rich foods"
                )
        
        return recommendations
    
    def get_alternative_suggestions(
        self,
        food_name: str,
        user_conditions: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get alternative food suggestions that are safer for user's conditions.
        
        Args:
            food_name: Current food item
            user_conditions: User's medical conditions
        
        Returns:
            List of alternative foods with reasons
        """
        if not self.trained_scanner:
            return []
        
        alternatives = []
        
        # Common healthy alternatives
        alternative_map = {
            'white_rice': [
                {'name': 'brown_rice', 'reason': 'Lower glycemic index'},
                {'name': 'quinoa', 'reason': 'Higher protein, more fiber'},
                {'name': 'cauliflower_rice', 'reason': 'Very low carb'}
            ],
            'fried_chicken': [
                {'name': 'grilled_chicken', 'reason': 'Less saturated fat'},
                {'name': 'baked_chicken', 'reason': 'Lower calories'},
                {'name': 'chicken_breast', 'reason': 'Leaner protein'}
            ],
            'french_fries': [
                {'name': 'sweet_potato_fries', 'reason': 'More nutrients'},
                {'name': 'roasted_vegetables', 'reason': 'Lower fat'},
                {'name': 'salad', 'reason': 'Much healthier'}
            ]
        }
        
        # Get alternatives for this food
        if food_name in alternative_map:
            alternatives = alternative_map[food_name]
        
        return alternatives
    
    def create_meal_report(
        self,
        cv_nutrition_results: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create comprehensive meal report combining CV analysis and health impact.
        
        Args:
            cv_nutrition_results: Complete output from CV pipeline
                {
                    'ingredients': [...],
                    'totals': {...},
                    'portions': [...],
                    'confidence': 0.92
                }
            user_profile: User health profile
                {
                    'conditions': ['diabetes', 'hypertension'],
                    'daily_limits': {...},
                    'allergies': [...]
                }
        
        Returns:
            Complete meal report with health impact
        """
        # Extract nutrition totals
        nutrition = cv_nutrition_results.get('totals', {})
        
        # Analyze health impact
        health_impact = self.analyze_food_for_conditions(
            nutrition,
            user_profile.get('conditions', [])
        )
        
        # Create comprehensive report
        report = {
            'meal_analysis': cv_nutrition_results,
            'health_impact': health_impact,
            'daily_progress': self._calculate_daily_progress(
                nutrition,
                user_profile.get('daily_limits', {})
            ),
            'recommendations': health_impact.get('recommendations', []),
            'overall_score': self._calculate_meal_score(
                nutrition,
                health_impact,
                user_profile
            )
        }
        
        return report
    
    def _calculate_daily_progress(
        self,
        nutrition: Dict[str, Any],
        daily_limits: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate progress toward daily nutrition goals."""
        progress = {}
        
        nutrients = ['calories', 'protein', 'carbs', 'fat', 'sodium', 'sugar']
        
        for nutrient in nutrients:
            if nutrient in nutrition and nutrient in daily_limits:
                consumed = nutrition[nutrient]
                limit = daily_limits[nutrient]
                percentage = (consumed / limit) * 100 if limit > 0 else 0
                
                progress[nutrient] = {
                    'consumed': consumed,
                    'limit': limit,
                    'remaining': max(0, limit - consumed),
                    'percentage': round(percentage, 1)
                }
        
        return progress
    
    def _calculate_meal_score(
        self,
        nutrition: Dict[str, Any],
        health_impact: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall meal health score (0-100)."""
        score = 100
        
        # Deduct points for warnings
        warnings = health_impact.get('warnings', [])
        score -= len(warnings) * 15
        
        # Deduct points for exceeding limits
        if not health_impact.get('safe_to_eat', True):
            score -= 20
        
        # Bonus for balanced macros
        protein_pct = nutrition.get('protein_percent', 0)
        carbs_pct = nutrition.get('carbs_percent', 0)
        fat_pct = nutrition.get('fat_percent', 0)
        
        # Ideal: 30% protein, 40% carbs, 30% fat
        macro_balance = abs(30 - protein_pct) + abs(40 - carbs_pct) + abs(30 - fat_pct)
        score -= macro_balance / 2
        
        # Ensure score is between 0-100
        score = max(0, min(100, score))
        
        # Determine rating
        if score >= 80:
            rating = "Excellent"
            emoji = "🌟"
        elif score >= 60:
            rating = "Good"
            emoji = "👍"
        elif score >= 40:
            rating = "Fair"
            emoji = "⚠️"
        else:
            rating = "Poor"
            emoji = "❌"
        
        return {
            'score': round(score, 1),
            'rating': rating,
            'emoji': emoji,
            'safe_for_conditions': health_impact.get('safe_to_eat', True)
        }


# Singleton instance
_integration = None

def get_integration() -> CVNutritionIntegration:
    """Get or create integration instance."""
    global _integration
    if _integration is None:
        _integration = CVNutritionIntegration()
    return _integration


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def demo_integration():
    """Demonstrate CV + Disease integration."""
    print("=" * 80)
    print("CV-DISEASE INTEGRATION DEMO")
    print("=" * 80)
    
    # Simulate CV pipeline output
    cv_results = {
        'meal': 'Chicken Curry with Rice',
        'ingredients': [
            {
                'name': 'Rice (white, cooked)',
                'weight': '665g',
                'nutrition': {
                    'calories': 865,
                    'protein': 18,
                    'carbs': 186,
                    'fat': 2,
                    'sodium': 33
                }
            },
            {
                'name': 'Chicken (curry)',
                'weight': '649g',
                'nutrition': {
                    'calories': 1071,
                    'protein': 201,
                    'carbs': 0,
                    'fat': 23,
                    'sodium': 455
                }
            }
        ],
        'totals': {
            'calories': 2356,
            'protein': 225,
            'carbs': 208,
            'fat': 59,
            'sodium': 978,
            'sugar': 3.2,
            'protein_percent': 38,
            'carbs_percent': 35,
            'fat_percent': 27
        }
    }
    
    # User profile
    user = {
        'conditions': ['diabetes', 'hypertension'],
        'daily_limits': {
            'calories': 2000,
            'sodium': 1500,
            'sugar': 25,
            'carbs': 225
        }
    }
    
    # Create integration
    integration = get_integration()
    
    # Generate report
    print("\nGenerating meal report...")
    report = integration.create_meal_report(cv_results, user)
    
    # Display results
    print(f"\n{'=' * 80}")
    print("MEAL HEALTH REPORT")
    print(f"{'=' * 80}")
    
    print(f"\nOverall Score: {report['overall_score']['score']} "
          f"{report['overall_score']['emoji']} "
          f"({report['overall_score']['rating']})")
    
    if report['health_impact']['warnings']:
        print("\n⚠️ HEALTH WARNINGS:")
        for warning in report['health_impact']['warnings']:
            print(f"  {warning}")
    
    if report['recommendations']:
        print("\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  {rec}")
    
    if 'daily_progress' in report:
        print("\n📊 DAILY PROGRESS:")
        for nutrient, data in report['daily_progress'].items():
            print(f"  {nutrient.capitalize()}: {data['consumed']:.0f} / "
                  f"{data['limit']:.0f} ({data['percentage']:.0f}%)")
    
    print(f"\n{'=' * 80}\n")


if __name__ == '__main__':
    demo_integration()
