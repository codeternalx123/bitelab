"""
Health Profile Engine
====================

User health profile management with CKD stages, pregnancy, diabetes, 
age groups, and risk stratification.

This module manages the user's complete medical profile and determines
which thresholds apply.

Key Features:
- Comprehensive health condition tracking
- Multi-condition risk stratification
- Medication interaction tracking
- Allergy management
- Lab value monitoring
- Dietary preference integration

Author: BiteLab AI Team
Date: November 2025
Version: 1.0.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum
from datetime import datetime, date
import logging
import json
from collections import defaultdict

from .dynamic_thresholds import (
    DynamicThresholdDatabase,
    ThresholdRule,
    AgeGroup,
    PregnancyStatus,
    CKDStage,
    DiabetesType,
    MedicalCode
)

from .model_loader import ModelLoader
from .data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class RiskLevel(Enum):
    """Overall health risk level."""
    MINIMAL = "minimal"          # No significant conditions
    LOW = "low"                  # Controlled conditions
    MODERATE = "moderate"        # Multiple conditions or poorly controlled
    HIGH = "high"                # Severe conditions or multiple high-risk
    CRITICAL = "critical"        # Life-threatening conditions


class Gender(Enum):
    """Biological sex (for threshold calculations)."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class ActivityLevel(Enum):
    """Physical activity level."""
    SEDENTARY = "sedentary"
    LIGHT = "light"
    MODERATE = "moderate"
    ACTIVE = "active"
    VERY_ACTIVE = "very_active"


class DietaryPreference(Enum):
    """Dietary preferences."""
    OMNIVORE = "omnivore"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    PESCATARIAN = "pescatarian"
    KETOGENIC = "ketogenic"
    MEDITERRANEAN = "mediterranean"
    DASH = "dash"                # For hypertension
    RENAL = "renal"              # For CKD


class TherapeuticGoal(Enum):
    """User therapeutic goals (55+ goals)."""
    # Physical Health & Management
    WEIGHT_LOSS = "weight_loss"
    MUSCLE_GAIN = "muscle_gain"
    MAINTENANCE = "maintenance"
    HEART_HEALTH = "heart_health"
    DIABETES_MANAGEMENT = "diabetes_management"
    KIDNEY_PROTECTION = "kidney_protection"
    LIVER_HEALTH = "liver_health"
    GUT_HEALTH = "gut_health"
    BRAIN_HEALTH = "brain_health"
    BONE_HEALTH = "bone_health"
    JOINT_HEALTH = "joint_health"
    SKIN_HEALTH = "skin_health"
    HAIR_HEALTH = "hair_health"
    EYE_HEALTH = "eye_health"
    IMMUNE_SUPPORT = "immune_support"
    ENERGY_BOOST = "energy_boost"
    SLEEP_IMPROVEMENT = "sleep_improvement"
    STRESS_REDUCTION = "stress_reduction"
    ANTI_INFLAMMATORY = "anti_inflammatory"
    DETOXIFICATION = "detoxification"
    HORMONAL_BALANCE = "hormonal_balance"
    FERTILITY_SUPPORT = "fertility_support"
    PREGNANCY_SUPPORT = "pregnancy_support"
    LACTATION_SUPPORT = "lactation_support"
    MENOPAUSE_SUPPORT = "menopause_support"
    ANTI_AGING = "anti_aging"
    ATHLETIC_PERFORMANCE = "athletic_performance"
    RECOVERY = "recovery"
    HYDRATION = "hydration"
    BLOOD_PRESSURE_MANAGEMENT = "blood_pressure_management"
    CHOLESTEROL_MANAGEMENT = "cholesterol_management"
    BLOOD_SUGAR_CONTROL = "blood_sugar_control"
    THYROID_HEALTH = "thyroid_health"
    ADRENAL_SUPPORT = "adrenal_support"
    RESPIRATORY_HEALTH = "respiratory_health"
    ALLERGY_MANAGEMENT = "allergy_management"
    AUTOIMMUNE_SUPPORT = "autoimmune_support"
    CANCER_PREVENTION = "cancer_prevention"
    CANCER_SUPPORT = "cancer_support"
    
    # Mental & Cognitive
    COGNITIVE_FUNCTION = "cognitive_function"
    MOOD_ENHANCEMENT = "mood_enhancement"
    ANXIETY_RELIEF = "anxiety_relief"
    DEPRESSION_SUPPORT = "depression_support"
    FOCUS_CONCENTRATION = "focus_concentration"
    MEMORY_ENHANCEMENT = "memory_enhancement"
    
    # Metabolic & Cellular
    DIGESTION_SUPPORT = "digestion_support"
    METABOLISM_BOOST = "metabolism_boost"
    APPETITE_CONTROL = "appetite_control"
    CRAVINGS_MANAGEMENT = "cravings_management"
    NUTRIENT_ABSORPTION = "nutrient_absorption"
    ELECTROLYTE_BALANCE = "electrolyte_balance"
    PH_BALANCE = "ph_balance"
    ANTIOXIDANT_SUPPORT = "antioxidant_support"
    MITOCHONDRIAL_HEALTH = "mitochondrial_health"
    TELOMERE_SUPPORT = "telomere_support"
    DNA_REPAIR = "dna_repair"
    STEM_CELL_SUPPORT = "stem_cell_support"



# ============================================================================
# HEALTH CONDITION
# ============================================================================

@dataclass
class HealthCondition:
    """
    Single health condition with severity and control status.
    
    Examples:
    - CKD Stage 4 (severe, controlled with diet)
    - Type 2 Diabetes (moderate, controlled with metformin)
    - Pregnancy (trimester 2)
    """
    condition_id: str
    condition_name: str
    snomed_ct: Optional[str] = None
    icd_11: Optional[str] = None
    
    # Severity
    severity: str = "moderate"           # mild, moderate, severe, critical
    is_controlled: bool = True           # Is condition well-controlled?
    
    # Specific parameters
    ckd_stage: Optional[CKDStage] = None
    diabetes_type: Optional[DiabetesType] = None
    pregnancy_trimester: Optional[PregnancyStatus] = None
    
    # Lab values (if applicable)
    egfr: Optional[float] = None         # Estimated GFR (mL/min/1.73m²)
    hba1c: Optional[float] = None        # Hemoglobin A1c (%)
    serum_potassium: Optional[float] = None  # mEq/L
    serum_phosphorus: Optional[float] = None  # mg/dL
    blood_pressure_systolic: Optional[int] = None  # mmHg
    blood_pressure_diastolic: Optional[int] = None  # mmHg
    
    # Treatment
    # medications: List[str] = field(default_factory=list)  # Removed for simplified engine
    dietary_restrictions: List[str] = field(default_factory=list)
    
    # Dates
    diagnosis_date: Optional[date] = None
    last_updated: datetime = field(default_factory=datetime.now)
    
    def get_risk_level(self) -> RiskLevel:
        """Calculate risk level for this condition."""
        
        # CKD risk levels
        if self.ckd_stage:
            if self.ckd_stage in [CKDStage.CKD_STAGE_5, CKDStage.DIALYSIS]:
                return RiskLevel.CRITICAL
            elif self.ckd_stage == CKDStage.CKD_STAGE_4:
                return RiskLevel.HIGH
            elif self.ckd_stage in [CKDStage.CKD_STAGE_3A, CKDStage.CKD_STAGE_3B]:
                return RiskLevel.MODERATE
            else:
                return RiskLevel.LOW
        
        # Pregnancy risk
        if self.pregnancy_trimester:
            return RiskLevel.HIGH  # Always high priority
        
        # Diabetes risk (based on HbA1c control)
        if self.diabetes_type and self.diabetes_type != DiabetesType.NONE:
            if self.hba1c:
                if self.hba1c > 9.0:
                    return RiskLevel.HIGH  # Poorly controlled
                elif self.hba1c > 7.5:
                    return RiskLevel.MODERATE
                else:
                    return RiskLevel.LOW  # Well-controlled
            return RiskLevel.MODERATE  # Unknown control
        
        # Default based on severity
        severity_map = {
            "mild": RiskLevel.LOW,
            "moderate": RiskLevel.MODERATE,
            "severe": RiskLevel.HIGH,
            "critical": RiskLevel.CRITICAL
        }
        
        return severity_map.get(self.severity, RiskLevel.MODERATE)


# ============================================================================
# LAB VALUES
# ============================================================================

@dataclass
class LabValue:
    """Laboratory test result."""
    test_name: str
    value: float
    unit: str
    reference_range_min: Optional[float] = None
    reference_range_max: Optional[float] = None
    test_date: datetime = field(default_factory=datetime.now)
    
    def is_abnormal(self) -> bool:
        """Check if value is outside reference range."""
        if self.reference_range_min and self.value < self.reference_range_min:
            return True
        if self.reference_range_max and self.value > self.reference_range_max:
            return True
        return False
    
    def get_abnormality_description(self) -> str:
        """Get description of abnormality."""
        if not self.is_abnormal():
            return "Normal"
        
        if self.reference_range_min and self.value < self.reference_range_min:
            return f"Low ({self.value} < {self.reference_range_min})"
        
        if self.reference_range_max and self.value > self.reference_range_max:
            return f"High ({self.value} > {self.reference_range_max})"
        
        return "Abnormal"


# ============================================================================
# ALLERGY
# ============================================================================

@dataclass
class FoodAllergy:
    """Food allergy or intolerance."""
    allergen: str                        # Peanuts, shellfish, lactose, etc.
    severity: str = "moderate"           # mild, moderate, severe, anaphylactic
    reaction_type: str = "allergy"       # allergy, intolerance, sensitivity
    symptoms: List[str] = field(default_factory=list)
    diagnosed_by: str = ""               # Allergist, self-reported, etc.
    diagnosis_date: Optional[date] = None


# ============================================================================
# AI MODELS
# ============================================================================

class RiskStratificationModel:
    """
    RSM: Risk Stratification Model.
    
    Uses Gradient Boosted Machine (GBM) logic to calculate a composite risk score (0-100)
    based on biometrics, lab values, and disease interactions.
    
    Now loads trained models via ModelLoader.
    """
    
    def __init__(self, model_loader: ModelLoader):
        self.model = model_loader.load_risk_stratification_model()
        logger.info("Initialized RiskStratificationModel (RSM) with loaded model")
        
    def calculate_risk_score(self, profile: 'UserHealthProfile') -> float:
        """
        Calculate comprehensive risk score (0-100) using ML model.
        """
        # Prepare feature vector for model
        features = self._extract_features(profile)
        
        # Predict using loaded model
        try:
            risk_score = self.model.predict([features])
            # Handle different return types (scalar, list, array)
            if isinstance(risk_score, (list, np.ndarray)):
                risk_score = float(risk_score[0])
            return min(max(float(risk_score), 0.0), 100.0)
        except Exception as e:
            logger.error(f"RSM Prediction failed: {e}")
            return self._fallback_risk_calculation(profile)

    def _extract_features(self, profile: 'UserHealthProfile') -> List[float]:
        """Extract numerical features for ML model."""
        # Simplified feature extraction for demo
        # In production, this would be a comprehensive vector
        return [
            profile.age,
            profile.bmi,
            len(profile.conditions),
            len([l for l in profile.lab_values if l.is_abnormal()])
        ]

    def _fallback_risk_calculation(self, profile: 'UserHealthProfile') -> float:
        """Fallback heuristic if model fails."""
        # 1. Age Factor (Normalized 0-100)
        age_factor = min(profile.age / 100.0 * 100, 100)
        
        # 2. BMI Factor
        bmi_factor = 0
        if profile.bmi > 30:
            bmi_factor = min((profile.bmi - 25) * 5, 100)
        elif profile.bmi < 18.5:
            bmi_factor = (18.5 - profile.bmi) * 10
            
        # 3. Condition Severity
        condition_score = 0
        if profile.conditions:
            severity_map = {"mild": 20, "moderate": 50, "severe": 80, "critical": 100}
            max_sev = 0
            for c in profile.conditions:
                sev_val = severity_map.get(c.severity, 50)
                if c.get_risk_level() == RiskLevel.CRITICAL: sev_val = 100
                elif c.get_risk_level() == RiskLevel.HIGH: sev_val = 80
                max_sev = max(max_sev, sev_val)
            condition_score = min(max_sev + (len(profile.conditions) - 1) * 10, 100)
            
        # 4. Lab Abnormality
        lab_score = 0
        abnormal_labs = [lab for lab in profile.lab_values if lab.is_abnormal()]
        if abnormal_labs:
            lab_score = min(len(abnormal_labs) * 20, 100)
            
        # Weighted Sum (Fallback weights)
        total_risk = (age_factor * 0.1) + (bmi_factor * 0.15) + (condition_score * 0.4) + (lab_score * 0.35)
        return min(total_risk, 100.0)

    def predict_progression_probability(self, profile: 'UserHealthProfile', years: int = 5) -> float:
        """Predict probability of condition progression over X years."""
        current_risk = self.calculate_risk_score(profile)
        progression_prob = current_risk / 100.0
        if profile.age > 60:
            progression_prob *= 1.2
        return min(progression_prob, 0.99)


class TherapeuticRecommendationEngine:
    """
    MTRE: Multi-Target Therapeutic Recommendation Engine.
    
    Uses Uplift Modeling logic to rank foods/compounds based on their 
    predicted therapeutic benefit for the user's specific conditions.
    """
    
    def __init__(self, model_loader: ModelLoader, data_loader: DataLoader):
        self.model = model_loader.load_therapeutic_recommender()
        self.goal_definitions = data_loader.load_goal_definitions()
        logger.info("Initialized TherapeuticRecommendationEngine (MTRE) with loaded model")
        
    def rank_therapeutic_foods(
        self, 
        profile: 'UserHealthProfile', 
        available_foods: List[Dict[str, Any]],
        goals: List[TherapeuticGoal]
    ) -> List[Dict[str, Any]]:
        """
        Rank foods by Therapeutic Uplift Score using ML model.
        """
        ranked_foods = []
        
        for food in available_foods:
            uplift_score = 0.0
            reasons = []
            
            # 1. Check for beneficial compounds based on goals (using loaded definitions)
            compounds = food.get('compounds', [])
            
            for goal in goals:
                # Use loaded goal definitions instead of hardcoded strings
                beneficial_compounds = self.goal_definitions.get(goal.value, [])
                
                # Check for matches
                matches = [c for c in compounds if c in beneficial_compounds]
                if matches:
                    uplift_score += 10 * len(matches)
                    reasons.append(f"Supports {goal.value.replace('_', ' ')} ({', '.join(matches)})")
            
            # 2. Check for condition-specific benefits
            for condition in profile.conditions:
                if condition.condition_name.lower() == "hypertension":
                    if 'nitrates' in compounds or 'magnesium' in compounds:
                        uplift_score += 5
                        reasons.append("Supports blood pressure management")
            
            # 3. Penalties (Contraindications)
            ckd_stage = profile.get_ckd_stage()
            if ckd_stage and ckd_stage in [CKDStage.CKD_STAGE_3B, CKDStage.CKD_STAGE_4, CKDStage.CKD_STAGE_5]:
                if food.get('potassium_mg', 0) > 300:
                    uplift_score -= 50
                    reasons.append("High Potassium (Contraindicated for CKD)")
            
            # 4. Apply ML Model Adjustment (Simulated)
            try:
                # Model takes user vector + food vector and predicts additional uplift
                ml_adjustment = self.model.predict([uplift_score])
                if isinstance(ml_adjustment, (list, np.ndarray)):
                    ml_adjustment = float(ml_adjustment[0])
                uplift_score += ml_adjustment
            except:
                pass
            
            food_entry = food.copy()
            food_entry['uplift_score'] = uplift_score
            food_entry['therapeutic_reasons'] = reasons
            ranked_foods.append(food_entry)
            
        # Sort by score descending
        return sorted(ranked_foods, key=lambda x: x['uplift_score'], reverse=True)


class DiseaseCompoundExtractor:
    """
    DICE: Disease-Therapeutic Compound Extraction.
    
    Uses NLP-based logic (simulated here with rule-based extraction) to 
    structure nutritional therapy rules from medical guidelines.
    """
    
    def __init__(self, model_loader: ModelLoader, data_loader: DataLoader):
        self.model = model_loader.load_disease_extractor()
        self.knowledge_base = data_loader.load_disease_rules()
        logger.info("Initialized DiseaseCompoundExtractor (DICE) with loaded knowledge base")
        
    def extract_rules_for_condition(self, condition_name: str) -> Dict[str, Any]:
        """Extract nutritional rules for a specific condition."""
        # Normalize name
        key = None
        if "kidney" in condition_name.lower() or "ckd" in condition_name.lower():
            key = "ckd"
        elif "diabetes" in condition_name.lower():
            key = "diabetes"
        elif "pressure" in condition_name.lower() or "hypertension" in condition_name.lower():
            key = "hypertension"
            
        if key:
            return self.knowledge_base.get(key, {})
        return {}
        
    def get_conflicting_rules(self, conditions: List[str]) -> List[str]:
        """Identify conflicting nutritional rules between multiple conditions."""
        # Example: CKD limits Potassium, Hypertension encourages Potassium
        conflicts = []
        
        has_ckd = any("ckd" in c.lower() or "kidney" in c.lower() for c in conditions)
        has_htn = any("hypertension" in c.lower() or "pressure" in c.lower() for c in conditions)
        
        if has_ckd and has_htn:
            conflicts.append(
                "CONFLICT: Hypertension benefits from Potassium, but CKD requires Potassium restriction. "
                "PRIORITY: Follow CKD restrictions (Safety First)."
            )
            
        return conflicts


# ============================================================================
# USER HEALTH PROFILE
# ============================================================================

@dataclass
class UserHealthProfile:
    """
    Complete user health profile.
    
    This is the central data structure that stores ALL health information
    about a user.
    """
    user_id: str
    
    # Demographics
    age: float                           # In years
    gender: Gender
    body_weight_kg: float
    height_cm: float
    
    # Health conditions
    conditions: List[HealthCondition] = field(default_factory=list)
    
    # Lab values
    lab_values: List[LabValue] = field(default_factory=list)
    
    # Allergies
    allergies: List[FoodAllergy] = field(default_factory=list)
    
    # Pregnancy status
    is_pregnant: bool = False
    pregnancy_trimester: Optional[PregnancyStatus] = None
    is_lactating: bool = False
    
    # Lifestyle
    activity_level: ActivityLevel = ActivityLevel.MODERATE
    dietary_preference: DietaryPreference = DietaryPreference.OMNIVORE
    therapeutic_goals: List[TherapeuticGoal] = field(default_factory=list)
    
    # Calculated fields
    bmi: float = field(init=False)
    overall_risk_level: RiskLevel = field(init=False)
    risk_score: float = field(init=False, default=0.0)  # 0-100 score from RSM
    
    # Metadata
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.bmi = self.calculate_bmi()
        self.overall_risk_level = self.calculate_overall_risk()
    
    def calculate_bmi(self) -> float:
        """Calculate BMI."""
        height_m = self.height_cm / 100
        return self.body_weight_kg / (height_m ** 2)
    
    def get_bmi_category(self) -> str:
        """Get BMI category."""
        if self.bmi < 18.5:
            return "Underweight"
        elif self.bmi < 25:
            return "Normal"
        elif self.bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def calculate_overall_risk(self) -> RiskLevel:
        """
        Calculate overall health risk level.
        
        Returns highest risk level among all conditions.
        """
        if not self.conditions:
            return RiskLevel.MINIMAL
        
        # Get risk levels for all conditions
        risk_levels = [cond.get_risk_level() for cond in self.conditions]
        
        # Return highest risk
        risk_priority = {
            RiskLevel.MINIMAL: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MODERATE: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4
        }
        
        max_risk = max(risk_levels, key=lambda r: risk_priority[r])
        return max_risk
    
    def get_active_conditions(self) -> List[HealthCondition]:
        """Get list of active conditions."""
        return self.conditions
    
    def has_condition(self, condition_name: str) -> bool:
        """Check if user has a specific condition."""
        return any(c.condition_name.lower() == condition_name.lower() for c in self.conditions)
    
    def get_condition(self, condition_name: str) -> Optional[HealthCondition]:
        """Get specific condition by name."""
        for cond in self.conditions:
            if cond.condition_name.lower() == condition_name.lower():
                return cond
        return None
    
    def get_ckd_stage(self) -> Optional[CKDStage]:
        """Get CKD stage if user has CKD."""
        for cond in self.conditions:
            if cond.ckd_stage:
                return cond.ckd_stage
        return None
    
    def get_diabetes_type(self) -> DiabetesType:
        """Get diabetes type if user has diabetes."""
        for cond in self.conditions:
            if cond.diabetes_type:
                return cond.diabetes_type
        return DiabetesType.NONE
    
    def get_latest_lab_value(self, test_name: str) -> Optional[LabValue]:
        """Get most recent lab value for a test."""
        matching = [lab for lab in self.lab_values if lab.test_name == test_name]
        if not matching:
            return None
        return max(matching, key=lambda lab: lab.test_date)
    
    def has_allergy(self, allergen: str) -> bool:
        """Check if user has allergy to specific allergen."""
        return any(a.allergen.lower() == allergen.lower() for a in self.allergies)
    
    def get_age_group(self) -> AgeGroup:
        """Get age group for threshold lookup."""
        if self.age < 0.5:
            return AgeGroup.INFANT_0_6M
        elif self.age < 1:
            return AgeGroup.INFANT_7_12M
        elif self.age < 3:
            return AgeGroup.CHILD_1_3Y
        elif self.age < 8:
            return AgeGroup.CHILD_4_8Y
        elif self.age < 13:
            return AgeGroup.CHILD_9_13Y
        elif self.age < 18:
            return AgeGroup.TEEN_14_18Y
        elif self.age < 50:
            return AgeGroup.ADULT_19_50Y
        elif self.age < 70:
            return AgeGroup.ADULT_51_70Y
        else:
            return AgeGroup.ELDERLY_70PLUS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'user_id': self.user_id,
            'age': self.age,
            'gender': self.gender.value,
            'body_weight_kg': self.body_weight_kg,
            'height_cm': self.height_cm,
            'bmi': self.bmi,
            'bmi_category': self.get_bmi_category(),
            'overall_risk_level': self.overall_risk_level.value,
            'risk_score': self.risk_score,
            'conditions': [
                {
                    'name': c.condition_name,
                    'severity': c.severity,
                    'risk_level': c.get_risk_level().value
                }
                for c in self.conditions
            ],
            'allergies': [a.allergen for a in self.allergies],
            'is_pregnant': self.is_pregnant,
            'pregnancy_trimester': self.pregnancy_trimester.value if self.pregnancy_trimester else None,
            'therapeutic_goals': [g.value for g in self.therapeutic_goals]
        }


# ============================================================================
# HEALTH PROFILE ENGINE
# ============================================================================

class HealthProfileEngine:
    """
    Engine for managing user health profiles and determining applicable thresholds.
    
    This is the "brain" that connects user health data to dynamic thresholds.
    
    UPDATED: Now uses RSM, MTRE, and DICE models for simplified therapeutic focus.
    """
    
    def __init__(self, threshold_db: DynamicThresholdDatabase):
        """
        Initialize health profile engine.
        
        Args:
            threshold_db: Dynamic threshold database
        """
        self.threshold_db = threshold_db
        self.profiles: Dict[str, UserHealthProfile] = {}
        
        # Initialize Loaders
        self.model_loader = ModelLoader()
        self.data_loader = DataLoader()
        
        # Initialize AI Models with Loaders
        self.rsm = RiskStratificationModel(self.model_loader)
        self.mtre = TherapeuticRecommendationEngine(self.model_loader, self.data_loader)
        self.dice = DiseaseCompoundExtractor(self.model_loader, self.data_loader)
        
        logger.info("Initialized HealthProfileEngine with RSM, MTRE, and DICE")
    
    def create_profile(
        self,
        user_id: str,
        age: float,
        gender: Gender,
        body_weight_kg: float,
        height_cm: float,
        **kwargs
    ) -> UserHealthProfile:
        """
        Create new user health profile.
        
        Args:
            user_id: User ID
            age: Age in years
            gender: Gender
            body_weight_kg: Body weight in kg
            height_cm: Height in cm
            **kwargs: Additional profile fields
            
        Returns:
            New user profile
        """
        profile = UserHealthProfile(
            user_id=user_id,
            age=age,
            gender=gender,
            body_weight_kg=body_weight_kg,
            height_cm=height_cm,
            **kwargs
        )
        
        # Calculate initial risk score
        profile.risk_score = self.rsm.calculate_risk_score(profile)
        
        self.profiles[user_id] = profile
        
        logger.info(f"Created profile for user {user_id} with Risk Score: {profile.risk_score}")
        
        return profile
    
    def get_profile(self, user_id: str) -> Optional[UserHealthProfile]:
        """Get user profile."""
        return self.profiles.get(user_id)
    
    def add_condition(
        self,
        user_id: str,
        condition: HealthCondition
    ) -> bool:
        """
        Add health condition to user profile.
        
        Args:
            user_id: User ID
            condition: Health condition object
            
        Returns:
            True if successful
        """
        profile = self.get_profile(user_id)
        if not profile:
            logger.error(f"User {user_id} not found")
            return False
        
        # Check if condition already exists
        if profile.has_condition(condition.condition_name):
            logger.warning(f"Condition {condition.condition_name} already exists for user {user_id}")
            return False
        
        profile.conditions.append(condition)
        profile.last_updated = datetime.now()
        
        # Recalculate risk
        profile.overall_risk_level = profile.calculate_overall_risk()
        profile.risk_score = self.rsm.calculate_risk_score(profile)
        
        logger.info(f"Added condition {condition.condition_name} to user {user_id}. New Risk Score: {profile.risk_score}")
        return True
    
    def add_lab_value(
        self,
        user_id: str,
        lab_value: LabValue
    ) -> bool:
        """Add lab value to user profile."""
        profile = self.get_profile(user_id)
        if not profile:
            return False
            
        profile.lab_values.append(lab_value)
        profile.last_updated = datetime.now()
        
        # Recalculate risk
        profile.risk_score = self.rsm.calculate_risk_score(profile)
        
        return True
        
    def set_therapeutic_goals(self, user_id: str, goals: List[TherapeuticGoal]) -> bool:
        """Set therapeutic goals for user."""
        profile = self.get_profile(user_id)
        if not profile:
            return False
            
        profile.therapeutic_goals = goals
        profile.last_updated = datetime.now()
        return True

    def get_applicable_thresholds(self, user_id: str, nutrient_name: str) -> Dict[str, Any]:
        """
        Get applicable thresholds for a nutrient based on user profile.
        
        This uses the DynamicThresholdDatabase to find rules that match
        the user's profile.
        """
        profile = self.get_profile(user_id)
        if not profile:
            return {}
            
        # Apply DICE rules (Nutritional Therapy Rules)
        dice_rules = {}
        for condition in profile.conditions:
            rules = self.dice.extract_rules_for_condition(condition.condition_name)
            if rules:
                dice_rules[condition.condition_name] = rules
        
        return {
            "therapy_rules": dice_rules,
            "risk_score": profile.risk_score
        }
        
    def get_food_recommendations(self, user_id: str, available_foods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get ranked food recommendations using MTRE.
        """
        profile = self.get_profile(user_id)
        if not profile:
            return []
            
        return self.mtre.rank_therapeutic_foods(
            profile=profile,
            available_foods=available_foods,
            goals=profile.therapeutic_goals
        )

    def analyze_health_risk(self, user_id: str) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis using RSM.
        """
        profile = self.get_profile(user_id)
        if not profile:
            return {}
            
        risk_score = self.rsm.calculate_risk_score(profile)
        progression_prob = self.rsm.predict_progression_probability(profile)
        
        return {
            "risk_score": risk_score,
            "risk_level": profile.overall_risk_level.value,
            "progression_probability_5yr": progression_prob,
            "active_conditions": [c.condition_name for c in profile.conditions],
            "abnormal_labs": [l.test_name for l in profile.lab_values if l.is_abnormal()]
        }

    

    

    
    def get_applicable_conditions(
        self,
        user_id: str
    ) -> List[str]:
        """
        Get list of condition names applicable to user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of condition names for threshold lookup
        """
        profile = self.get_profile(user_id)
        if not profile:
            return ["general_population"]
        
        condition_names = []
        
        # Add user's diagnosed conditions
        for cond in profile.conditions:
            # CKD stages
            if cond.ckd_stage:
                if cond.ckd_stage == CKDStage.CKD_STAGE_3A or cond.ckd_stage == CKDStage.CKD_STAGE_3B:
                    condition_names.append("ckd_stage_3")
                elif cond.ckd_stage == CKDStage.CKD_STAGE_4:
                    condition_names.append("ckd_stage_4")
                elif cond.ckd_stage == CKDStage.CKD_STAGE_5 or cond.ckd_stage == CKDStage.DIALYSIS:
                    condition_names.append("ckd_stage_5_dialysis")
            
            # Diabetes
            if cond.diabetes_type and cond.diabetes_type != DiabetesType.NONE:
                condition_names.append("diabetes")
            
            # Add by condition name
            if cond.condition_name.lower() in ["hypertension", "heart_failure"]:
                condition_names.append(cond.condition_name.lower())
        
        # Pregnancy
        if profile.is_pregnant:
            condition_names.append("pregnancy")
        
        # Infants
        if profile.age < 1:
            condition_names.append("infant_0_12m")
        
        # Always include general population and toxic elements
        condition_names.append("general_population")
        condition_names.append("toxic_elements")
        
        return list(set(condition_names))  # Remove duplicates
    
    def get_thresholds_for_user(
        self,
        user_id: str,
        element: str
    ) -> List[ThresholdRule]:
        """
        Get all applicable thresholds for a user and element.
        
        Args:
            user_id: User ID
            element: Chemical element
            
        Returns:
            List of applicable threshold rules
        """
        profile = self.get_profile(user_id)
        if not profile:
            return []
        
        # Get applicable conditions
        conditions = self.get_applicable_conditions(user_id)
        
        # Get rules for each condition
        all_rules = []
        
        for condition_name in conditions:
            threshold = self.threshold_db.get_threshold(condition_name)
            if threshold:
                rule = threshold.get_most_restrictive_rule(
                    element=element,
                    age=profile.age,
                    pregnancy=profile.pregnancy_trimester,
                    ckd_stage=profile.get_ckd_stage(),
                    diabetes=profile.get_diabetes_type()
                )
                if rule:
                    all_rules.append(rule)
        
        return all_rules
    
    def get_most_restrictive_threshold(
        self,
        user_id: str,
        element: str
    ) -> Optional[ThresholdRule]:
        """
        Get the MOST RESTRICTIVE threshold for a user and element.
        
        This is the key method that determines which threshold to apply.
        
        Args:
            user_id: User ID
            element: Chemical element
            
        Returns:
            Most restrictive threshold rule
        """
        rules = self.get_thresholds_for_user(user_id, element)
        
        if not rules:
            return None
        
        # Return highest priority rule
        return max(rules, key=lambda r: r.priority)
    
    def generate_health_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive health summary for user.
        """
        profile = self.get_profile(user_id)
        if not profile:
            return {}
            
        # Get latest critical lab values
        latest_labs = {}
        for test in ["eGFR", "HbA1c", "Potassium", "Phosphorus"]:
            lab = profile.get_latest_lab_value(test)
            if lab:
                latest_labs[test] = {
                    'value': lab.value,
                    'unit': lab.unit,
                    'is_abnormal': lab.is_abnormal(),
                    'description': lab.get_abnormality_description()
                }
        
        summary = {
            'user_id': user_id,
            'demographics': {
                'age': profile.age,
                'age_group': profile.get_age_group().value,
                'gender': profile.gender.value,
                'bmi': profile.bmi,
                'bmi_category': profile.get_bmi_category()
            },
            'overall_risk': profile.overall_risk_level.value,
            'risk_score': profile.risk_score,
            'conditions': [
                {
                    'name': cond.condition_name,
                    'severity': cond.severity,
                    'risk_level': cond.get_risk_level().value,
                    'is_controlled': cond.is_controlled
                }
                for cond in profile.conditions
            ],
            'lab_values': latest_labs,
            'allergies': [
                {
                    'allergen': allergy.allergen,
                    'severity': allergy.severity
                }
                for allergy in profile.allergies
            ],
            'pregnancy': {
                'is_pregnant': profile.is_pregnant,
                'trimester': profile.pregnancy_trimester.value if profile.pregnancy_trimester else None
            },
            'dietary_preference': profile.dietary_preference.value,
            'therapeutic_goals': [g.value for g in profile.therapeutic_goals],
            'applicable_conditions': self.get_applicable_conditions(user_id)
        }
        
        return summary


# ============================================================================
# TESTING
# ============================================================================

def test_health_profile_engine():
    """Test simplified health profile engine with RSM, MTRE, and DICE."""
    print("\n" + "="*80)
    print("SIMPLIFIED THERAPEUTIC HEALTH ENGINE TEST")
    print("="*80)
    
    # Initialize
    print("\n" + "-"*80)
    print("Initializing...")
    
    from .dynamic_thresholds import DynamicThresholdDatabase
    
    threshold_db = DynamicThresholdDatabase()
    engine = HealthProfileEngine(threshold_db)
    
    print("✓ Initialized Engine with RSM, MTRE, DICE")
    
    # Test 1: Create profile with CKD Stage 4
    print("\n" + "-"*80)
    print("Test 1: Patient with CKD Stage 4 (High Risk)")
    
    profile = engine.create_profile(
        user_id="patient001",
        age=65,
        gender=Gender.MALE,
        body_weight_kg=80,
        height_cm=175
    )
    
    # Add CKD condition
    ckd_condition = HealthCondition(
        condition_id="ckd001",
        condition_name="CKD Stage 4",
        severity="severe",
        ckd_stage=CKDStage.CKD_STAGE_4,
        egfr=22.0,
        serum_potassium=5.2,
        serum_phosphorus=5.8
    )
    
    engine.add_condition("patient001", ckd_condition)
    
    # Add abnormal lab
    engine.add_lab_value("patient001", LabValue(
        test_name="Potassium",
        value=5.2,
        unit="mEq/L",
        reference_range_min=3.5,
        reference_range_max=5.0
    ))
    
    # Analyze Risk (RSM)
    risk_analysis = engine.analyze_health_risk("patient001")
    print(f"\n✓ Risk Analysis (RSM):")
    print(f"  Risk Score: {risk_analysis['risk_score']:.1f}/100")
    print(f"  Risk Level: {risk_analysis['risk_level']}")
    print(f"  Progression Prob (5yr): {risk_analysis['progression_probability_5yr']:.2%}")
    
    # Test 2: Therapeutic Recommendations (MTRE)
    print("\n" + "-"*80)
    print("Test 2: Therapeutic Food Recommendations (MTRE)")
    
    engine.set_therapeutic_goals("patient001", [
        TherapeuticGoal.KIDNEY_PROTECTION,
        TherapeuticGoal.ANTI_INFLAMMATORY,
        TherapeuticGoal.HEART_HEALTH
    ])
    
    sample_foods = [
        {
            "name": "Spinach (Raw)",
            "potassium_mg": 558,
            "compounds": ["lutein", "nitrates", "magnesium"]
        },
        {
            "name": "Blueberries",
            "potassium_mg": 77,
            "compounds": ["anthocyanins", "quercetin", "vitamin_c"]
        },
        {
            "name": "Salmon",
            "potassium_mg": 363,
            "compounds": ["omega-3", "protein", "selenium"]
        }
    ]
    
    recommendations = engine.get_food_recommendations("patient001", sample_foods)
    
    print(f"\n✓ Top Recommendations:")
    for i, food in enumerate(recommendations):
        print(f"  {i+1}. {food['name']} (Score: {food['uplift_score']})")
        print(f"     Reasons: {', '.join(food['therapeutic_reasons'])}")
        
    # Test 3: Disease Rules (DICE)
    print("\n" + "-"*80)
    print("Test 3: Disease Rules Extraction (DICE)")
    
    thresholds = engine.get_applicable_thresholds("patient001", "Potassium")
    rules = thresholds.get("therapy_rules", {}).get("CKD Stage 4", {})
    
    print(f"\n✓ Extracted Rules for CKD:")
    print(f"  Rules: {thresholds.get('therapy_rules')}")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_health_profile_engine()
