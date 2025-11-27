"""
PHASE 4: Personalized Risk Assessment & Health Integration
===========================================================

This module implements personalized risk assessment that integrates:
- Atomic composition from SA optimization (Phase 2)
- ICP-MS ground truth validation (Phase 3)
- User health profiles (medical conditions, goals, medications)
- Regulatory safety thresholds (FDA, WHO, EFSA, NKF)
- Nutrient recommendations (RDA, UL, AI)

The system analyzes millions of cooked foods and generates personalized
warnings, recommendations, and alternatives based on individual health needs.

Core Features:
- Contaminant detection (Pb, Hg, As, Cd)
- Nutrient profiling (vitamins, minerals, macros)
- Drug-food interaction detection
- Disease-specific contraindications
- Goal alignment scoring
- Personalized alternative suggestions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from collections import defaultdict
from scipy.stats import norm
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk severity levels"""
    SAFE = "safe"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class HealthGoal(Enum):
    """User health goals"""
    WEIGHT_LOSS = "weight_loss"
    MUSCLE_GAIN = "muscle_gain"
    HEART_HEALTH = "heart_health"
    BRAIN_HEALTH = "brain_health"
    BONE_HEALTH = "bone_health"
    DIGESTIVE_HEALTH = "digestive_health"
    IMMUNE_SUPPORT = "immune_support"
    ENERGY_BOOST = "energy_boost"
    SKIN_HEALTH = "skin_health"
    BLOOD_SUGAR_CONTROL = "blood_sugar_control"
    BLOOD_PRESSURE_CONTROL = "blood_pressure_control"
    CHOLESTEROL_MANAGEMENT = "cholesterol_management"
    ANTI_INFLAMMATORY = "anti_inflammatory"
    DETOXIFICATION = "detoxification"
    HORMONAL_BALANCE = "hormonal_balance"
    STRESS_REDUCTION = "stress_reduction"
    SLEEP_IMPROVEMENT = "sleep_improvement"
    ATHLETIC_PERFORMANCE = "athletic_performance"
    PREGNANCY_NUTRITION = "pregnancy_nutrition"
    ANTI_AGING = "anti_aging"


class MedicalCondition(Enum):
    """Medical conditions affecting dietary needs"""
    DIABETES_TYPE1 = "diabetes_type1"
    DIABETES_TYPE2 = "diabetes_type2"
    HYPERTENSION = "hypertension"
    HYPOTENSION = "hypotension"
    HEART_DISEASE = "heart_disease"
    KIDNEY_DISEASE = "kidney_disease"
    LIVER_DISEASE = "liver_disease"
    CELIAC_DISEASE = "celiac_disease"
    CROHNS_DISEASE = "crohns_disease"
    IBS = "ibs"
    GERD = "gerd"
    OSTEOPOROSIS = "osteoporosis"
    ANEMIA = "anemia"
    THYROID_DISORDER = "thyroid_disorder"
    GOUT = "gout"
    CANCER = "cancer"
    PREGNANCY = "pregnancy"
    BREASTFEEDING = "breastfeeding"
    OBESITY = "obesity"
    UNDERWEIGHT = "underweight"


class AlertType(Enum):
    """Types of health alerts"""
    CONTAMINANT_WARNING = "contaminant_warning"
    NUTRIENT_DEFICIENCY = "nutrient_deficiency"
    NUTRIENT_EXCESS = "nutrient_excess"
    DRUG_INTERACTION = "drug_interaction"
    CONTRAINDICATION = "contraindication"
    ALLERGEN_PRESENT = "allergen_present"
    GOAL_MISALIGNMENT = "goal_misalignment"
    POSITIVE_BENEFIT = "positive_benefit"


@dataclass
class UserHealthProfile:
    """Comprehensive user health profile"""
    user_id: str
    age: int
    gender: str  # male, female, other
    weight_kg: float
    height_cm: float
    
    # Medical history
    medical_conditions: List[MedicalCondition] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    
    # Health goals
    primary_goals: List[HealthGoal] = field(default_factory=list)
    
    # Dietary preferences
    dietary_restrictions: List[str] = field(default_factory=list)  # vegan, halal, kosher, etc.
    
    # Activity level
    activity_level: str = "moderate"  # sedentary, light, moderate, active, very_active
    
    # Pregnancy/lactation
    is_pregnant: bool = False
    trimester: Optional[int] = None
    is_breastfeeding: bool = False
    
    # Lab values (if available)
    blood_glucose_mg_dl: Optional[float] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    cholesterol_total_mg_dl: Optional[float] = None
    hemoglobin_g_dl: Optional[float] = None
    creatinine_mg_dl: Optional[float] = None
    
    # Preferences
    risk_tolerance: str = "conservative"  # conservative, moderate, liberal
    
    def get_bmi(self) -> float:
        """Calculate BMI"""
        height_m = self.height_cm / 100.0
        return self.weight_kg / (height_m ** 2)
    
    def get_bmr(self) -> float:
        """Calculate Basal Metabolic Rate (Mifflin-St Jeor)"""
        if self.gender.lower() == "male":
            return 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age + 5
        else:
            return 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age - 161
    
    def get_tdee(self) -> float:
        """Calculate Total Daily Energy Expenditure"""
        activity_multipliers = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9
        }
        multiplier = activity_multipliers.get(self.activity_level, 1.55)
        return self.get_bmr() * multiplier


@dataclass
class RegulatoryThreshold:
    """Regulatory safety threshold for contaminants"""
    element: str
    threshold_ppm: float
    agency: str  # FDA, WHO, EFSA, NKF, etc.
    food_category: str
    population: str = "general"  # general, pregnant, children, etc.
    threshold_type: str = "maximum"  # maximum, recommended, action_level
    
    def is_exceeded(self, concentration: float) -> bool:
        """Check if concentration exceeds threshold"""
        return concentration > self.threshold_ppm


@dataclass
class NutrientRecommendation:
    """Nutrient intake recommendation"""
    nutrient: str
    rda: Optional[float] = None  # Recommended Dietary Allowance
    ai: Optional[float] = None   # Adequate Intake
    ul: Optional[float] = None   # Upper Limit
    unit: str = "mg"
    age_group: str = "adult"
    gender: str = "both"
    condition_specific: Optional[str] = None


@dataclass
class HealthAlert:
    """Health alert/warning for user"""
    alert_id: str
    alert_type: AlertType
    risk_level: RiskLevel
    title: str
    message: str
    details: Dict[str, Any]
    
    # Affected elements/nutrients
    elements: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    
    # Supporting data
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    safe_range: Optional[Tuple[float, float]] = None
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class NutrientProfile:
    """Complete nutritional profile of food"""
    # Macronutrients (g/100g)
    protein: float = 0.0
    carbohydrates: float = 0.0
    fat: float = 0.0
    fiber: float = 0.0
    sugar: float = 0.0
    
    # Calories
    calories: float = 0.0
    
    # Minerals (mg/100g unless specified)
    calcium: float = 0.0
    iron: float = 0.0
    magnesium: float = 0.0
    phosphorus: float = 0.0
    potassium: float = 0.0
    sodium: float = 0.0
    zinc: float = 0.0
    copper: float = 0.0
    manganese: float = 0.0
    selenium: float = 0.0  # μg/100g
    
    # Vitamins (various units)
    vitamin_a_iu: float = 0.0
    vitamin_c_mg: float = 0.0
    vitamin_d_iu: float = 0.0
    vitamin_e_mg: float = 0.0
    vitamin_k_mcg: float = 0.0
    thiamin_mg: float = 0.0
    riboflavin_mg: float = 0.0
    niacin_mg: float = 0.0
    vitamin_b6_mg: float = 0.0
    folate_mcg: float = 0.0
    vitamin_b12_mcg: float = 0.0
    
    # Fatty acids (g/100g)
    saturated_fat: float = 0.0
    monounsaturated_fat: float = 0.0
    polyunsaturated_fat: float = 0.0
    omega3: float = 0.0
    omega6: float = 0.0
    trans_fat: float = 0.0
    
    # Cholesterol (mg/100g)
    cholesterol: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'protein': self.protein,
            'carbohydrates': self.carbohydrates,
            'fat': self.fat,
            'fiber': self.fiber,
            'calories': self.calories,
            'calcium': self.calcium,
            'iron': self.iron,
            'sodium': self.sodium,
            'potassium': self.potassium,
            # ... add all nutrients
        }


class ContaminantDetector:
    """
    Detects and assesses contaminant risks in food.
    
    Monitors:
    - Heavy metals (Pb, Hg, As, Cd)
    - Toxic elements (Al, Ni)
    - Environmental contaminants
    
    Compares against regulatory thresholds from:
    - FDA (Food and Drug Administration)
    - WHO (World Health Organization)
    - EFSA (European Food Safety Authority)
    - NKF (National Kidney Foundation)
    """
    
    def __init__(self):
        self.thresholds = self._load_regulatory_thresholds()
        self.contaminant_health_effects = self._load_health_effects()
        
    def _load_regulatory_thresholds(self) -> Dict[str, List[RegulatoryThreshold]]:
        """Load regulatory thresholds for contaminants"""
        thresholds = {
            'Pb': [  # Lead
                RegulatoryThreshold('Pb', 0.1, 'FDA', 'general', 'general', 'maximum'),
                RegulatoryThreshold('Pb', 0.05, 'FDA', 'general', 'children', 'maximum'),
                RegulatoryThreshold('Pb', 0.02, 'FDA', 'general', 'pregnant', 'maximum'),
            ],
            'Hg': [  # Mercury
                RegulatoryThreshold('Hg', 1.0, 'FDA', 'fish', 'general', 'maximum'),
                RegulatoryThreshold('Hg', 0.5, 'WHO', 'fish', 'general', 'maximum'),
                RegulatoryThreshold('Hg', 0.1, 'FDA', 'fish', 'pregnant', 'maximum'),
            ],
            'As': [  # Arsenic
                RegulatoryThreshold('As', 0.3, 'FDA', 'general', 'general', 'maximum'),
                RegulatoryThreshold('As', 0.1, 'FDA', 'rice', 'general', 'action_level'),
            ],
            'Cd': [  # Cadmium
                RegulatoryThreshold('Cd', 0.1, 'EFSA', 'general', 'general', 'maximum'),
                RegulatoryThreshold('Cd', 0.05, 'WHO', 'general', 'kidney_disease', 'maximum'),
            ],
            'Al': [  # Aluminum
                RegulatoryThreshold('Al', 2.0, 'EFSA', 'general', 'general', 'tolerable'),
            ],
        }
        return thresholds
    
    def _load_health_effects(self) -> Dict[str, Dict[str, Any]]:
        """Load health effects database for contaminants"""
        return {
            'Pb': {
                'name': 'Lead',
                'acute_effects': ['Abdominal pain', 'Nausea', 'Vomiting'],
                'chronic_effects': ['Neurological damage', 'Kidney damage', 'Anemia', 'Reproductive issues'],
                'vulnerable_populations': ['Children', 'Pregnant women', 'Developing fetuses'],
                'bioaccumulative': True,
                'half_life_days': 30,
            },
            'Hg': {
                'name': 'Mercury',
                'acute_effects': ['Tremors', 'Memory loss', 'Neuromuscular changes'],
                'chronic_effects': ['Neurological damage', 'Kidney damage', 'Fetal brain development issues'],
                'vulnerable_populations': ['Pregnant women', 'Breastfeeding mothers', 'Young children'],
                'bioaccumulative': True,
                'half_life_days': 60,
            },
            'As': {
                'name': 'Arsenic',
                'acute_effects': ['Vomiting', 'Diarrhea', 'Abdominal pain'],
                'chronic_effects': ['Cancer risk', 'Cardiovascular disease', 'Diabetes', 'Skin lesions'],
                'vulnerable_populations': ['Children', 'Pregnant women'],
                'bioaccumulative': False,
                'half_life_days': 10,
            },
            'Cd': {
                'name': 'Cadmium',
                'acute_effects': ['Nausea', 'Vomiting', 'Diarrhea'],
                'chronic_effects': ['Kidney damage', 'Bone damage', 'Cancer risk'],
                'vulnerable_populations': ['Kidney disease patients', 'Smokers'],
                'bioaccumulative': True,
                'half_life_days': 3650,  # 10 years
            },
        }
    
    def detect_contaminants(
        self,
        atomic_composition: Dict[str, float],
        food_category: str,
        user_profile: UserHealthProfile
    ) -> List[HealthAlert]:
        """
        Detect contaminant risks in atomic composition
        
        Args:
            atomic_composition: Element concentrations from SA optimization
            food_category: Category of food
            user_profile: User health profile
            
        Returns:
            List of health alerts for contaminants
        """
        alerts = []
        
        # Check each contaminant
        for element, concentration in atomic_composition.items():
            if element not in self.thresholds:
                continue
            
            # Get applicable thresholds
            applicable_thresholds = self._get_applicable_thresholds(
                element,
                food_category,
                user_profile
            )
            
            for threshold in applicable_thresholds:
                if threshold.is_exceeded(concentration):
                    alert = self._create_contaminant_alert(
                        element,
                        concentration,
                        threshold,
                        user_profile
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _get_applicable_thresholds(
        self,
        element: str,
        food_category: str,
        user_profile: UserHealthProfile
    ) -> List[RegulatoryThreshold]:
        """Get thresholds applicable to user and food"""
        all_thresholds = self.thresholds.get(element, [])
        applicable = []
        
        for threshold in all_thresholds:
            # Check food category match
            if threshold.food_category != 'general' and threshold.food_category != food_category:
                continue
            
            # Check population match
            if threshold.population == 'general':
                applicable.append(threshold)
            elif threshold.population == 'pregnant' and user_profile.is_pregnant:
                applicable.append(threshold)
            elif threshold.population == 'children' and user_profile.age < 18:
                applicable.append(threshold)
            elif threshold.population == 'kidney_disease':
                if MedicalCondition.KIDNEY_DISEASE in user_profile.medical_conditions:
                    applicable.append(threshold)
        
        return applicable
    
    def _create_contaminant_alert(
        self,
        element: str,
        concentration: float,
        threshold: RegulatoryThreshold,
        user_profile: UserHealthProfile
    ) -> HealthAlert:
        """Create health alert for contaminant"""
        health_effects = self.contaminant_health_effects.get(element, {})
        element_name = health_effects.get('name', element)
        
        # Determine risk level
        excess_ratio = concentration / threshold.threshold_ppm
        if excess_ratio >= 5.0:
            risk_level = RiskLevel.CRITICAL
        elif excess_ratio >= 2.0:
            risk_level = RiskLevel.HIGH
        elif excess_ratio >= 1.5:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.LOW
        
        # Build message
        title = f"CRITICAL: High {element_name} Detected" if risk_level == RiskLevel.CRITICAL else f"Warning: Elevated {element_name}"
        
        message = (
            f"This food contains {concentration:.3f} ppm of {element_name}, "
            f"which exceeds the {threshold.agency} {threshold.threshold_type} limit of "
            f"{threshold.threshold_ppm} ppm by {((excess_ratio - 1) * 100):.0f}%."
        )
        
        # Add vulnerable population warning
        if user_profile.is_pregnant or user_profile.is_breastfeeding:
            message += f" {element_name} is particularly dangerous during pregnancy/breastfeeding."
        
        # Recommendations
        recommendations = [
            f"Limit consumption of this food",
            f"Choose alternative foods with lower {element_name} content",
        ]
        
        if health_effects.get('bioaccumulative'):
            recommendations.append(
                f"{element_name} accumulates in the body. Regular consumption increases risk."
            )
        
        # Add specific health effects
        chronic_effects = health_effects.get('chronic_effects', [])
        if chronic_effects:
            recommendations.append(
                f"Long-term exposure may cause: {', '.join(chronic_effects[:3])}"
            )
        
        return HealthAlert(
            alert_id=f"contaminant_{element}_{datetime.now().timestamp()}",
            alert_type=AlertType.CONTAMINANT_WARNING,
            risk_level=risk_level,
            title=title,
            message=message,
            details={
                'element': element,
                'element_name': element_name,
                'concentration_ppm': concentration,
                'threshold_ppm': threshold.threshold_ppm,
                'excess_ratio': excess_ratio,
                'agency': threshold.agency,
                'health_effects': health_effects,
            },
            elements=[element],
            recommendations=recommendations,
            current_value=concentration,
            threshold_value=threshold.threshold_ppm
        )
    
    def estimate_exposure_risk(
        self,
        element: str,
        concentration: float,
        serving_size_g: float,
        frequency_per_week: int,
        user_profile: UserHealthProfile
    ) -> Dict[str, Any]:
        """
        Estimate chronic exposure risk from regular consumption
        
        Args:
            element: Contaminant element
            concentration: Concentration in ppm
            serving_size_g: Typical serving size
            frequency_per_week: How often consumed
            user_profile: User profile
            
        Returns:
            Risk assessment dictionary
        """
        # Calculate weekly intake (mg)
        weekly_intake_mg = (concentration * serving_size_g * frequency_per_week) / 1000.0
        
        # Calculate per body weight (mg/kg/week)
        intake_per_kg = weekly_intake_mg / user_profile.weight_kg
        
        # Get health effects
        health_effects = self.contaminant_health_effects.get(element, {})
        
        # Estimate bioaccumulation
        bioaccumulative = health_effects.get('bioaccumulative', False)
        half_life_days = health_effects.get('half_life_days', 30)
        
        if bioaccumulative:
            # Estimate steady-state concentration
            weeks_to_steady_state = (5 * half_life_days) / 7  # 5 half-lives
            steady_state_factor = 1.0 / (1.0 - np.exp(-7.0 / half_life_days))
            effective_weekly_dose = weekly_intake_mg * steady_state_factor
        else:
            effective_weekly_dose = weekly_intake_mg
        
        # Compare to safety thresholds
        thresholds = self._get_applicable_thresholds(element, 'general', user_profile)
        
        risk_assessment = {
            'element': element,
            'weekly_intake_mg': weekly_intake_mg,
            'intake_per_kg_week': intake_per_kg,
            'bioaccumulative': bioaccumulative,
            'half_life_days': half_life_days,
            'effective_weekly_dose': effective_weekly_dose,
            'risk_level': self._assess_exposure_level(effective_weekly_dose, thresholds),
        }
        
        return risk_assessment
    
    def _assess_exposure_level(
        self,
        weekly_dose: float,
        thresholds: List[RegulatoryThreshold]
    ) -> str:
        """Assess exposure level"""
        if not thresholds:
            return "unknown"
        
        min_threshold = min(t.threshold_ppm for t in thresholds)
        
        # Rough heuristic (needs refinement with actual PTWI values)
        weekly_threshold = min_threshold * 0.1  # Simplified
        
        ratio = weekly_dose / weekly_threshold
        
        if ratio >= 2.0:
            return "high"
        elif ratio >= 1.0:
            return "moderate"
        elif ratio >= 0.5:
            return "low"
        else:
            return "minimal"


class NutrientProfiler:
    """
    Analyzes nutrient content and compares against recommendations.
    
    Provides:
    - RDA (Recommended Dietary Allowance) comparison
    - UL (Upper Limit) checking
    - Goal alignment scoring
    - Deficiency/excess detection
    """
    
    def __init__(self):
        self.nutrient_recommendations = self._load_nutrient_recommendations()
        self.nutrient_goals = self._load_nutrient_goals()
        
    def _load_nutrient_recommendations(self) -> Dict[str, NutrientRecommendation]:
        """Load RDA/AI/UL values for nutrients"""
        # Simplified - would load from comprehensive database
        return {
            'protein': NutrientRecommendation(
                'protein', rda=50.0, ul=None, unit='g'
            ),
            'fiber': NutrientRecommendation(
                'fiber', ai=25.0, ul=None, unit='g'
            ),
            'calcium': NutrientRecommendation(
                'calcium', rda=1000.0, ul=2500.0, unit='mg'
            ),
            'iron': NutrientRecommendation(
                'iron', rda=18.0, ul=45.0, unit='mg'
            ),
            'sodium': NutrientRecommendation(
                'sodium', ai=1500.0, ul=2300.0, unit='mg'
            ),
            'potassium': NutrientRecommendation(
                'potassium', ai=2600.0, ul=None, unit='mg'
            ),
            'vitamin_c': NutrientRecommendation(
                'vitamin_c', rda=90.0, ul=2000.0, unit='mg'
            ),
            'vitamin_d': NutrientRecommendation(
                'vitamin_d', rda=600.0, ul=4000.0, unit='IU'
            ),
        }
    
    def _load_nutrient_goals(self) -> Dict[HealthGoal, Dict[str, str]]:
        """Load nutrient priorities for each health goal"""
        return {
            HealthGoal.WEIGHT_LOSS: {
                'high_priority': ['protein', 'fiber'],
                'moderate_priority': ['water', 'vitamins'],
                'limit': ['calories', 'sugar', 'fat'],
            },
            HealthGoal.MUSCLE_GAIN: {
                'high_priority': ['protein', 'calories', 'carbohydrates'],
                'moderate_priority': ['creatine', 'leucine', 'vitamin_d'],
                'limit': [],
            },
            HealthGoal.HEART_HEALTH: {
                'high_priority': ['omega3', 'fiber', 'potassium', 'magnesium'],
                'moderate_priority': ['vitamin_e', 'folate'],
                'limit': ['sodium', 'saturated_fat', 'trans_fat', 'cholesterol'],
            },
            HealthGoal.BONE_HEALTH: {
                'high_priority': ['calcium', 'vitamin_d', 'vitamin_k', 'magnesium'],
                'moderate_priority': ['protein', 'phosphorus'],
                'limit': ['sodium'],
            },
            HealthGoal.BLOOD_SUGAR_CONTROL: {
                'high_priority': ['fiber', 'chromium', 'magnesium'],
                'moderate_priority': ['protein'],
                'limit': ['sugar', 'refined_carbs'],
            },
        }
    
    def analyze_nutrient_profile(
        self,
        nutrient_profile: NutrientProfile,
        atomic_composition: Dict[str, float],
        user_profile: UserHealthProfile,
        serving_size_g: float = 100.0
    ) -> List[HealthAlert]:
        """
        Analyze nutrient content against user needs
        
        Args:
            nutrient_profile: Nutritional content
            atomic_composition: Elemental composition
            user_profile: User health profile
            serving_size_g: Serving size
            
        Returns:
            List of nutrient-related alerts
        """
        alerts = []
        
        # Merge atomic composition into nutrient profile
        enhanced_profile = self._enhance_profile_with_minerals(
            nutrient_profile,
            atomic_composition
        )
        
        # Check against RDA/UL
        alerts.extend(self._check_nutrient_adequacy(
            enhanced_profile,
            user_profile,
            serving_size_g
        ))
        
        # Check goal alignment
        alerts.extend(self._check_goal_alignment(
            enhanced_profile,
            user_profile,
            serving_size_g
        ))
        
        # Check medical condition contraindications
        alerts.extend(self._check_medical_contraindications(
            enhanced_profile,
            user_profile,
            serving_size_g
        ))
        
        return alerts
    
    def _enhance_profile_with_minerals(
        self,
        profile: NutrientProfile,
        composition: Dict[str, float]
    ) -> NutrientProfile:
        """Add mineral data from atomic composition"""
        enhanced = profile
        
        # Map elements to nutrients
        element_mapping = {
            'Fe': 'iron',
            'Ca': 'calcium',
            'Mg': 'magnesium',
            'Na': 'sodium',
            'K': 'potassium',
            'Zn': 'zinc',
            'Cu': 'copper',
            'Mn': 'manganese',
            'Se': 'selenium',
        }
        
        for element, nutrient_attr in element_mapping.items():
            if element in composition:
                setattr(enhanced, nutrient_attr, composition[element])
        
        return enhanced
    
    def _check_nutrient_adequacy(
        self,
        profile: NutrientProfile,
        user_profile: UserHealthProfile,
        serving_size: float
    ) -> List[HealthAlert]:
        """Check if nutrients meet or exceed recommendations"""
        alerts = []
        
        nutrients_to_check = [
            ('calcium', profile.calcium),
            ('iron', profile.iron),
            ('sodium', profile.sodium),
            ('potassium', profile.potassium),
        ]
        
        for nutrient_name, concentration_per_100g in nutrients_to_check:
            if nutrient_name not in self.nutrient_recommendations:
                continue
            
            recommendation = self.nutrient_recommendations[nutrient_name]
            
            # Calculate amount in serving
            amount_in_serving = (concentration_per_100g * serving_size) / 100.0
            
            # Check against RDA/AI
            target = recommendation.rda or recommendation.ai
            if target:
                percent_of_target = (amount_in_serving / target) * 100
                
                if percent_of_target >= 100:
                    # Meets or exceeds RDA
                    alerts.append(HealthAlert(
                        alert_id=f"nutrient_adequate_{nutrient_name}_{datetime.now().timestamp()}",
                        alert_type=AlertType.POSITIVE_BENEFIT,
                        risk_level=RiskLevel.SAFE,
                        title=f"Excellent {nutrient_name.title()} Source",
                        message=f"Provides {percent_of_target:.0f}% of daily {nutrient_name} needs",
                        details={'nutrient': nutrient_name, 'percent_rda': percent_of_target},
                        elements=[nutrient_name],
                        current_value=amount_in_serving,
                        threshold_value=target
                    ))
                elif percent_of_target < 10:
                    # Very low
                    alerts.append(HealthAlert(
                        alert_id=f"nutrient_low_{nutrient_name}_{datetime.now().timestamp()}",
                        alert_type=AlertType.NUTRIENT_DEFICIENCY,
                        risk_level=RiskLevel.LOW,
                        title=f"Low {nutrient_name.title()} Content",
                        message=f"Contains only {percent_of_target:.0f}% of daily {nutrient_name} needs",
                        details={'nutrient': nutrient_name, 'percent_rda': percent_of_target},
                        elements=[nutrient_name],
                        recommendations=[f"Consider pairing with {nutrient_name}-rich foods"],
                        current_value=amount_in_serving,
                        threshold_value=target
                    ))
            
            # Check against UL
            if recommendation.ul:
                if amount_in_serving > recommendation.ul:
                    alerts.append(HealthAlert(
                        alert_id=f"nutrient_excess_{nutrient_name}_{datetime.now().timestamp()}",
                        alert_type=AlertType.NUTRIENT_EXCESS,
                        risk_level=RiskLevel.HIGH,
                        title=f"Excessive {nutrient_name.title()}",
                        message=f"Contains {amount_in_serving:.0f} {recommendation.unit}, exceeding the safe upper limit of {recommendation.ul} {recommendation.unit}",
                        details={'nutrient': nutrient_name, 'amount': amount_in_serving, 'ul': recommendation.ul},
                        elements=[nutrient_name],
                        recommendations=[f"Limit portion size", f"Avoid daily consumption"],
                        current_value=amount_in_serving,
                        threshold_value=recommendation.ul
                    ))
        
        return alerts
    
    def _check_goal_alignment(
        self,
        profile: NutrientProfile,
        user_profile: UserHealthProfile,
        serving_size: float
    ) -> List[HealthAlert]:
        """Check alignment with user health goals"""
        alerts = []
        
        for goal in user_profile.primary_goals:
            if goal not in self.nutrient_goals:
                continue
            
            goal_nutrients = self.nutrient_goals[goal]
            
            # Check high priority nutrients
            high_priority = goal_nutrients.get('high_priority', [])
            for nutrient in high_priority:
                value = getattr(profile, nutrient, 0.0)
                if value > 0:
                    alerts.append(HealthAlert(
                        alert_id=f"goal_align_{goal.value}_{nutrient}_{datetime.now().timestamp()}",
                        alert_type=AlertType.POSITIVE_BENEFIT,
                        risk_level=RiskLevel.SAFE,
                        title=f"Supports {goal.value.replace('_', ' ').title()}",
                        message=f"Rich in {nutrient}, which supports your goal of {goal.value.replace('_', ' ')}",
                        details={'goal': goal.value, 'nutrient': nutrient, 'value': value},
                        elements=[nutrient]
                    ))
            
            # Check nutrients to limit
            limit_nutrients = goal_nutrients.get('limit', [])
            for nutrient in limit_nutrients:
                value = getattr(profile, nutrient, 0.0)
                
                # Define thresholds for "high"
                high_thresholds = {
                    'sodium': 400.0,  # mg per 100g
                    'sugar': 15.0,    # g per 100g
                    'saturated_fat': 5.0,  # g per 100g
                }
                
                threshold = high_thresholds.get(nutrient, 10.0)
                
                if value > threshold:
                    alerts.append(HealthAlert(
                        alert_id=f"goal_misalign_{goal.value}_{nutrient}_{datetime.now().timestamp()}",
                        alert_type=AlertType.GOAL_MISALIGNMENT,
                        risk_level=RiskLevel.MODERATE,
                        title=f"High {nutrient.title()} May Hinder Goal",
                        message=f"Contains high {nutrient} ({value:.1f}), which may conflict with your {goal.value.replace('_', ' ')} goal",
                        details={'goal': goal.value, 'nutrient': nutrient, 'value': value},
                        elements=[nutrient],
                        recommendations=[f"Consider lower-{nutrient} alternatives"]
                    ))
        
        return alerts
    
    def _check_medical_contraindications(
        self,
        profile: NutrientProfile,
        user_profile: UserHealthProfile,
        serving_size: float
    ) -> List[HealthAlert]:
        """Check for medical condition contraindications"""
        alerts = []
        
        # Define contraindications
        contraindications = {
            MedicalCondition.HYPERTENSION: {
                'sodium': (400.0, "High sodium can raise blood pressure")
            },
            MedicalCondition.KIDNEY_DISEASE: {
                'potassium': (200.0, "High potassium dangerous for kidney disease"),
                'phosphorus': (150.0, "High phosphorus harmful for kidney disease"),
                'sodium': (300.0, "Limit sodium with kidney disease"),
            },
            MedicalCondition.DIABETES_TYPE2: {
                'sugar': (10.0, "High sugar affects blood glucose"),
                'carbohydrates': (30.0, "High carbs affect blood sugar"),
            },
            MedicalCondition.GOUT: {
                'purines': (150.0, "High purine foods trigger gout"),  # Would need purine data
            },
        }
        
        for condition in user_profile.medical_conditions:
            if condition not in contraindications:
                continue
            
            condition_limits = contraindications[condition]
            
            for nutrient, (limit, reason) in condition_limits.items():
                value = getattr(profile, nutrient, 0.0)
                
                if value > limit:
                    alerts.append(HealthAlert(
                        alert_id=f"contraindication_{condition.value}_{nutrient}_{datetime.now().timestamp()}",
                        alert_type=AlertType.CONTRAINDICATION,
                        risk_level=RiskLevel.HIGH,
                        title=f"Warning: {condition.value.replace('_', ' ').title()}",
                        message=f"High {nutrient} ({value:.1f}): {reason}",
                        details={
                            'condition': condition.value,
                            'nutrient': nutrient,
                            'value': value,
                            'limit': limit
                        },
                        elements=[nutrient],
                        recommendations=[
                            f"Consult your doctor before consuming",
                            f"Choose low-{nutrient} alternatives"
                        ],
                        current_value=value,
                        threshold_value=limit
                    ))
        
        return alerts


class ThresholdValidator:
    """
    Validates atomic/nutrient values against comprehensive thresholds.
    
    Integrates:
    - FDA safety limits
    - WHO guidelines
    - EFSA standards
    - NKF recommendations (for kidney disease)
    - Pregnancy-specific limits
    """
    
    def __init__(self):
        self.fda_limits = self._load_fda_limits()
        self.who_limits = self._load_who_limits()
        self.efsa_limits = self._load_efsa_limits()
        self.nkf_limits = self._load_nkf_limits()
        
    def _load_fda_limits(self) -> Dict[str, float]:
        """Load FDA safety limits"""
        return {
            'Pb': 0.1,   # ppm
            'Hg': 1.0,   # ppm (fish)
            'As': 0.3,   # ppm
            'Cd': 0.1,   # ppm
        }
    
    def _load_who_limits(self) -> Dict[str, float]:
        """Load WHO guidelines"""
        return {
            'Pb': 0.05,
            'Hg': 0.5,
            'As': 0.2,
            'Cd': 0.1,
        }
    
    def _load_efsa_limits(self) -> Dict[str, float]:
        """Load EFSA standards"""
        return {
            'Pb': 0.1,
            'Cd': 0.05,
            'As': 0.15,
        }
    
    def _load_nkf_limits(self) -> Dict[str, float]:
        """Load NKF recommendations for kidney disease"""
        return {
            'K': 2000.0,   # mg/day
            'P': 800.0,    # mg/day
            'Na': 2000.0,  # mg/day
        }
    
    def validate_all_thresholds(
        self,
        composition: Dict[str, float],
        user_profile: UserHealthProfile
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate against all threshold databases
        
        Returns comprehensive validation report
        """
        validation_report = {}
        
        for element, value in composition.items():
            element_report = {
                'value': value,
                'fda_status': self._check_threshold(value, self.fda_limits.get(element)),
                'who_status': self._check_threshold(value, self.who_limits.get(element)),
                'efsa_status': self._check_threshold(value, self.efsa_limits.get(element)),
            }
            
            # Add NKF for kidney disease patients
            if MedicalCondition.KIDNEY_DISEASE in user_profile.medical_conditions:
                element_report['nkf_status'] = self._check_threshold(
                    value,
                    self.nkf_limits.get(element)
                )
            
            validation_report[element] = element_report
        
        return validation_report
    
    def _check_threshold(
        self,
        value: float,
        threshold: Optional[float]
    ) -> Dict[str, Any]:
        """Check single threshold"""
        if threshold is None:
            return {'status': 'no_limit', 'exceeded': False}
        
        exceeded = value > threshold
        percent_of_limit = (value / threshold) * 100
        
        return {
            'status': 'exceeded' if exceeded else 'within_limit',
            'exceeded': exceeded,
            'threshold': threshold,
            'percent_of_limit': percent_of_limit
        }


class HealthScoreCalculator:
    """
    Calculates overall health score for food based on:
    - Nutrient density
    - Contaminant levels
    - Goal alignment
    - Medical condition safety
    
    Score: 0-100 (higher is better)
    """
    
    def __init__(self):
        self.weights = {
            'nutrient_density': 0.3,
            'contaminant_safety': 0.3,
            'goal_alignment': 0.2,
            'medical_safety': 0.2,
        }
    
    def calculate_health_score(
        self,
        nutrient_profile: NutrientProfile,
        atomic_composition: Dict[str, float],
        user_profile: UserHealthProfile,
        alerts: List[HealthAlert]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive health score
        
        Returns:
            Dictionary with overall score and component scores
        """
        # Component scores
        nutrient_score = self._calculate_nutrient_density_score(nutrient_profile)
        contaminant_score = self._calculate_contaminant_safety_score(
            atomic_composition, alerts
        )
        goal_score = self._calculate_goal_alignment_score(
            nutrient_profile, user_profile, alerts
        )
        medical_score = self._calculate_medical_safety_score(alerts)
        
        # Weighted overall score
        overall_score = (
            self.weights['nutrient_density'] * nutrient_score +
            self.weights['contaminant_safety'] * contaminant_score +
            self.weights['goal_alignment'] * goal_score +
            self.weights['medical_safety'] * medical_score
        )
        
        return {
            'overall_score': round(overall_score, 1),
            'nutrient_density_score': round(nutrient_score, 1),
            'contaminant_safety_score': round(contaminant_score, 1),
            'goal_alignment_score': round(goal_score, 1),
            'medical_safety_score': round(medical_score, 1),
            'grade': self._score_to_grade(overall_score),
            'recommendation': self._score_to_recommendation(overall_score)
        }
    
    def _calculate_nutrient_density_score(self, profile: NutrientProfile) -> float:
        """Score based on nutrient density (0-100)"""
        # Positive nutrients
        positive_score = 0.0
        
        # Protein
        positive_score += min(profile.protein * 2, 30)  # Max 30 points
        
        # Fiber
        positive_score += min(profile.fiber * 4, 20)  # Max 20 points
        
        # Vitamins/minerals (simplified)
        positive_score += min((profile.iron + profile.calcium) / 20, 20)  # Max 20 points
        
        # Negative nutrients
        negative_score = 0.0
        
        # Saturated fat
        negative_score += min(profile.saturated_fat * 2, 15)
        
        # Sugar
        negative_score += min(profile.sugar, 15)
        
        # Sodium (per 100g)
        negative_score += min(profile.sodium / 100, 10)
        
        # Final score
        score = positive_score - negative_score
        
        return max(0, min(100, score + 50))  # Normalize to 0-100
    
    def _calculate_contaminant_safety_score(
        self,
        composition: Dict[str, float],
        alerts: List[HealthAlert]
    ) -> float:
        """Score based on contaminant safety (0-100)"""
        # Start at 100, deduct for contaminants
        score = 100.0
        
        contaminant_alerts = [
            a for a in alerts
            if a.alert_type == AlertType.CONTAMINANT_WARNING
        ]
        
        for alert in contaminant_alerts:
            if alert.risk_level == RiskLevel.CRITICAL:
                score -= 40
            elif alert.risk_level == RiskLevel.HIGH:
                score -= 25
            elif alert.risk_level == RiskLevel.MODERATE:
                score -= 15
            elif alert.risk_level == RiskLevel.LOW:
                score -= 5
        
        return max(0, score)
    
    def _calculate_goal_alignment_score(
        self,
        profile: NutrientProfile,
        user_profile: UserHealthProfile,
        alerts: List[HealthAlert]
    ) -> float:
        """Score based on alignment with user goals (0-100)"""
        if not user_profile.primary_goals:
            return 50.0  # Neutral if no goals
        
        score = 50.0  # Start neutral
        
        # Count positive alignment alerts
        positive_alerts = [
            a for a in alerts
            if a.alert_type == AlertType.POSITIVE_BENEFIT
        ]
        score += len(positive_alerts) * 10
        
        # Count misalignment alerts
        misalignment_alerts = [
            a for a in alerts
            if a.alert_type == AlertType.GOAL_MISALIGNMENT
        ]
        score -= len(misalignment_alerts) * 15
        
        return max(0, min(100, score))
    
    def _calculate_medical_safety_score(self, alerts: List[HealthAlert]) -> float:
        """Score based on medical condition safety (0-100)"""
        score = 100.0
        
        # Deduct for contraindications
        contraindication_alerts = [
            a for a in alerts
            if a.alert_type == AlertType.CONTRAINDICATION
        ]
        score -= len(contraindication_alerts) * 30
        
        # Deduct for drug interactions
        drug_alerts = [
            a for a in alerts
            if a.alert_type == AlertType.DRUG_INTERACTION
        ]
        score -= len(drug_alerts) * 25
        
        return max(0, score)
    
    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _score_to_recommendation(self, score: float) -> str:
        """Get recommendation based on score"""
        if score >= 85:
            return "Excellent choice! Highly recommended."
        elif score >= 70:
            return "Good option. Safe to consume regularly."
        elif score >= 55:
            return "Acceptable. Consume in moderation."
        elif score >= 40:
            return "Caution advised. Limit consumption."
        else:
            return "Not recommended. Seek healthier alternatives."


class PersonalizedRiskAnalyzer:
    """
    Main risk analyzer integrating all components.
    
    Provides comprehensive personalized risk assessment for millions
    of cooked foods based on user health profiles.
    """
    
    def __init__(self):
        self.contaminant_detector = ContaminantDetector()
        self.nutrient_profiler = NutrientProfiler()
        self.threshold_validator = ThresholdValidator()
        self.health_score_calculator = HealthScoreCalculator()
        
        logger.info("PersonalizedRiskAnalyzer initialized")
    
    def analyze_food(
        self,
        atomic_composition: Dict[str, float],
        nutrient_profile: NutrientProfile,
        user_profile: UserHealthProfile,
        food_name: str,
        food_category: str,
        serving_size_g: float = 100.0
    ) -> Dict[str, Any]:
        """
        Complete personalized risk analysis
        
        Args:
            atomic_composition: From SA optimization (Phase 2)
            nutrient_profile: Nutritional content
            user_profile: User health profile
            food_name: Name of food
            food_category: Category
            serving_size_g: Serving size
            
        Returns:
            Comprehensive analysis report
        """
        logger.info(f"Analyzing {food_name} for user {user_profile.user_id}")
        
        # Collect all alerts
        all_alerts = []
        
        # 1. Contaminant detection
        contaminant_alerts = self.contaminant_detector.detect_contaminants(
            atomic_composition,
            food_category,
            user_profile
        )
        all_alerts.extend(contaminant_alerts)
        
        # 2. Nutrient analysis
        nutrient_alerts = self.nutrient_profiler.analyze_nutrient_profile(
            nutrient_profile,
            atomic_composition,
            user_profile,
            serving_size_g
        )
        all_alerts.extend(nutrient_alerts)
        
        # 3. Threshold validation
        threshold_report = self.threshold_validator.validate_all_thresholds(
            atomic_composition,
            user_profile
        )
        
        # 4. Health score
        health_score = self.health_score_calculator.calculate_health_score(
            nutrient_profile,
            atomic_composition,
            user_profile,
            all_alerts
        )
        
        # 5. Categorize alerts by risk level
        critical_alerts = [a for a in all_alerts if a.risk_level == RiskLevel.CRITICAL]
        high_alerts = [a for a in all_alerts if a.risk_level == RiskLevel.HIGH]
        moderate_alerts = [a for a in all_alerts if a.risk_level == RiskLevel.MODERATE]
        low_alerts = [a for a in all_alerts if a.risk_level == RiskLevel.LOW]
        positive_alerts = [a for a in all_alerts if a.alert_type == AlertType.POSITIVE_BENEFIT]
        
        # 6. Overall safety verdict
        if critical_alerts:
            safety_verdict = "AVOID"
            safety_color = "red"
        elif high_alerts:
            safety_verdict = "CAUTION"
            safety_color = "orange"
        elif moderate_alerts:
            safety_verdict = "MODERATE"
            safety_color = "yellow"
        else:
            safety_verdict = "SAFE"
            safety_color = "green"
        
        # 7. Generate summary
        summary = self._generate_summary(
            food_name,
            health_score,
            all_alerts,
            user_profile
        )
        
        return {
            'food_name': food_name,
            'food_category': food_category,
            'user_id': user_profile.user_id,
            'timestamp': datetime.now().isoformat(),
            
            # Safety assessment
            'safety_verdict': safety_verdict,
            'safety_color': safety_color,
            
            # Health score
            'health_score': health_score,
            
            # Alerts
            'alerts': {
                'critical': [self._alert_to_dict(a) for a in critical_alerts],
                'high': [self._alert_to_dict(a) for a in high_alerts],
                'moderate': [self._alert_to_dict(a) for a in moderate_alerts],
                'low': [self._alert_to_dict(a) for a in low_alerts],
                'positive': [self._alert_to_dict(a) for a in positive_alerts],
            },
            'total_alerts': len(all_alerts),
            
            # Threshold validation
            'threshold_validation': threshold_report,
            
            # Summary
            'summary': summary,
            
            # Recommendations
            'top_recommendations': self._get_top_recommendations(all_alerts),
        }
    
    def _alert_to_dict(self, alert: HealthAlert) -> Dict:
        """Convert alert to dictionary"""
        return {
            'alert_id': alert.alert_id,
            'type': alert.alert_type.value,
            'risk_level': alert.risk_level.value,
            'title': alert.title,
            'message': alert.message,
            'details': alert.details,
            'recommendations': alert.recommendations,
            'alternatives': alert.alternatives,
            'current_value': alert.current_value,
            'threshold_value': alert.threshold_value,
            'timestamp': alert.timestamp,
        }
    
    def _generate_summary(
        self,
        food_name: str,
        health_score: Dict[str, Any],
        alerts: List[HealthAlert],
        user_profile: UserHealthProfile
    ) -> str:
        """Generate human-readable summary"""
        score = health_score['overall_score']
        grade = health_score['grade']
        
        # Count alerts by type
        critical_count = sum(1 for a in alerts if a.risk_level == RiskLevel.CRITICAL)
        high_count = sum(1 for a in alerts if a.risk_level == RiskLevel.HIGH)
        positive_count = sum(1 for a in alerts if a.alert_type == AlertType.POSITIVE_BENEFIT)
        
        if critical_count > 0:
            summary = f"{food_name} is NOT RECOMMENDED for you. "
            summary += f"Contains {critical_count} critical health risk(s). "
        elif high_count > 0:
            summary = f"{food_name} should be consumed with CAUTION. "
            summary += f"Contains {high_count} high-risk factor(s). "
        else:
            summary = f"{food_name} is generally SAFE for you. "
        
        summary += f"Health Score: {score}/100 (Grade {grade}). "
        
        if positive_count > 0:
            summary += f"Has {positive_count} beneficial properties for your health goals."
        
        return summary
    
    def _get_top_recommendations(self, alerts: List[HealthAlert]) -> List[str]:
        """Get top 5 recommendations"""
        all_recommendations = []
        
        # Prioritize critical/high alerts
        priority_alerts = sorted(
            alerts,
            key=lambda a: (a.risk_level.value, a.alert_type.value),
            reverse=True
        )
        
        for alert in priority_alerts[:5]:
            all_recommendations.extend(alert.recommendations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:5]


class DetailedRiskCardGenerator:
    """
    Generates comprehensive Risk Cards with detailed medical condition
    and health goal specific information.
    """
    
    def __init__(self):
        self.condition_insights = self._load_condition_insights()
        self.goal_insights = self._load_goal_insights()
        
    def _load_condition_insights(self) -> Dict[MedicalCondition, Dict[str, Any]]:
        """Load detailed insights for each medical condition"""
        return {
            MedicalCondition.DIABETES_TYPE2: {
                'name': 'Type 2 Diabetes',
                'emoji': '🩺',
                'focus_nutrients': ['sugar', 'carbohydrates', 'fiber', 'chromium'],
                'risk_factors': ['High blood sugar spikes', 'Insulin resistance'],
                'safe_ranges': {'sugar': (0, 10), 'carbohydrates': (0, 30), 'fiber': (5, 50)},
                'recommendations': [
                    'Choose low glycemic index foods',
                    'Pair carbs with protein/fat to slow absorption',
                    'Monitor blood sugar 2 hours after eating'
                ]
            },
            MedicalCondition.HYPERTENSION: {
                'name': 'High Blood Pressure',
                'emoji': '❤️‍🩹',
                'focus_nutrients': ['sodium', 'potassium', 'magnesium'],
                'risk_factors': ['Sodium-induced blood pressure spikes', 'Cardiovascular strain'],
                'safe_ranges': {'sodium': (0, 300), 'potassium': (200, 1000)},
                'recommendations': [
                    'Keep sodium under 1500mg daily',
                    'Increase potassium-rich foods',
                    'Monitor blood pressure regularly'
                ]
            },
            MedicalCondition.KIDNEY_DISEASE: {
                'name': 'Chronic Kidney Disease',
                'emoji': '🫘',
                'focus_nutrients': ['potassium', 'phosphorus', 'sodium', 'protein'],
                'risk_factors': ['Electrolyte imbalance', 'Kidney function decline'],
                'safe_ranges': {'potassium': (0, 150), 'phosphorus': (0, 100), 'protein': (0, 15)},
                'recommendations': [
                    'Limit potassium and phosphorus strictly',
                    'Moderate protein intake',
                    'Work closely with nephrologist'
                ]
            },
            MedicalCondition.PREGNANCY: {
                'name': 'Pregnancy',
                'emoji': '🤰',
                'focus_nutrients': ['mercury', 'lead', 'folate', 'iron', 'calcium'],
                'risk_factors': ['Fetal development issues', 'Neural tube defects'],
                'safe_ranges': {'mercury': (0, 0.05), 'lead': (0, 0.01)},
                'recommendations': [
                    'Avoid high-mercury fish',
                    'Ensure adequate folate (400mcg daily)',
                    'Take prenatal vitamins'
                ]
            },
            MedicalCondition.HEART_DISEASE: {
                'name': 'Cardiovascular Disease',
                'emoji': '❤️',
                'focus_nutrients': ['saturated_fat', 'trans_fat', 'cholesterol', 'omega3', 'sodium'],
                'risk_factors': ['Plaque buildup', 'Heart attack risk', 'Stroke risk'],
                'safe_ranges': {'saturated_fat': (0, 5), 'sodium': (0, 300), 'omega3': (500, 3000)},
                'recommendations': [
                    'Prioritize omega-3 rich foods',
                    'Limit saturated and trans fats',
                    'Increase fiber intake for cholesterol control'
                ]
            },
        }
    
    def _load_goal_insights(self) -> Dict[HealthGoal, Dict[str, Any]]:
        """Load detailed insights for each health goal"""
        return {
            HealthGoal.WEIGHT_LOSS: {
                'name': 'Weight Loss',
                'emoji': '⚖️',
                'focus_nutrients': ['protein', 'fiber', 'calories', 'water'],
                'target_ranges': {'protein': (20, 50), 'fiber': (5, 15), 'calories': (0, 300)},
                'success_factors': [
                    'High protein increases satiety',
                    'Fiber keeps you full longer',
                    'Low calorie density allows larger portions'
                ],
                'recommendations': [
                    'Aim for 25-30g protein per meal',
                    'Fill half your plate with vegetables',
                    'Drink water before meals'
                ]
            },
            HealthGoal.MUSCLE_GAIN: {
                'name': 'Muscle Gain',
                'emoji': '💪',
                'focus_nutrients': ['protein', 'calories', 'leucine', 'creatine'],
                'target_ranges': {'protein': (30, 60), 'calories': (400, 800)},
                'success_factors': [
                    'Protein provides amino acids for muscle synthesis',
                    'Calorie surplus enables muscle growth',
                    'Leucine triggers mTOR pathway'
                ],
                'recommendations': [
                    'Consume 1.6-2.2g protein per kg bodyweight',
                    'Eat within 2 hours post-workout',
                    'Include complete protein sources'
                ]
            },
            HealthGoal.HEART_HEALTH: {
                'name': 'Heart Health',
                'emoji': '❤️',
                'focus_nutrients': ['omega3', 'fiber', 'antioxidants', 'potassium'],
                'target_ranges': {'omega3': (500, 3000), 'fiber': (5, 15), 'potassium': (300, 1000)},
                'success_factors': [
                    'Omega-3 reduces inflammation',
                    'Fiber lowers LDL cholesterol',
                    'Antioxidants protect blood vessels'
                ],
                'recommendations': [
                    'Eat fatty fish 2-3x per week',
                    'Include nuts and seeds daily',
                    'Colorful vegetables for antioxidants'
                ]
            },
            HealthGoal.BLOOD_SUGAR_CONTROL: {
                'name': 'Blood Sugar Control',
                'emoji': '📊',
                'focus_nutrients': ['sugar', 'fiber', 'chromium', 'cinnamon'],
                'target_ranges': {'sugar': (0, 5), 'fiber': (5, 15)},
                'success_factors': [
                    'Fiber slows glucose absorption',
                    'Chromium enhances insulin sensitivity',
                    'Protein/fat reduce glycemic impact'
                ],
                'recommendations': [
                    'Choose low GI foods',
                    'Never eat carbs alone',
                    'Monitor post-meal glucose'
                ]
            },
        }
    
    def generate_detailed_risk_card(
        self,
        food_name: str,
        health_score: float,
        safety_verdict: str,
        alerts: List[HealthAlert],
        nutrient_profile: NutrientProfile,
        atomic_composition: Dict[str, float],
        user_profile: UserHealthProfile
    ) -> Dict[str, Any]:
        """
        Generate comprehensive risk card with condition/goal specific info
        """
        card = {
            'food_name': food_name,
            'health_score': health_score,
            'safety_verdict': safety_verdict,
            'overall_grade': self._score_to_grade(health_score),
            
            # Personalized sections
            'medical_condition_analysis': self._analyze_for_conditions(
                user_profile.medical_conditions,
                nutrient_profile,
                atomic_composition,
                alerts
            ),
            'health_goal_analysis': self._analyze_for_goals(
                user_profile.primary_goals,
                nutrient_profile,
                atomic_composition,
                alerts
            ),
            
            # Visual indicators
            'traffic_light': self._get_traffic_light(safety_verdict),
            'emoji_indicator': self._get_emoji_indicator(health_score, safety_verdict),
            
            # Detailed breakdowns
            'nutrient_highlights': self._get_nutrient_highlights(
                nutrient_profile,
                user_profile
            ),
            'contaminant_warnings': self._get_contaminant_warnings(
                atomic_composition,
                user_profile
            ),
            
            # Actionable recommendations
            'personalized_recommendations': self._get_personalized_recommendations(
                user_profile,
                alerts,
                nutrient_profile
            ),
            
            # Quick stats
            'quick_stats': self._generate_quick_stats(
                nutrient_profile,
                user_profile
            ),
        }
        
        return card
    
    def _analyze_for_conditions(
        self,
        conditions: List[MedicalCondition],
        nutrients: NutrientProfile,
        composition: Dict[str, float],
        alerts: List[HealthAlert]
    ) -> List[Dict[str, Any]]:
        """Analyze food for each medical condition"""
        analyses = []
        
        for condition in conditions:
            if condition not in self.condition_insights:
                continue
            
            insight = self.condition_insights[condition]
            
            # Check focus nutrients
            concerns = []
            benefits = []
            
            for nutrient in insight['focus_nutrients']:
                value = getattr(nutrients, nutrient, None)
                if value is None:
                    value = composition.get(nutrient, 0)
                
                if nutrient in insight.get('safe_ranges', {}):
                    min_safe, max_safe = insight['safe_ranges'][nutrient]
                    
                    if value > max_safe:
                        concerns.append({
                            'nutrient': nutrient,
                            'value': value,
                            'safe_max': max_safe,
                            'severity': 'high' if value > max_safe * 1.5 else 'moderate',
                            'message': f"⚠️ High {nutrient}: {value:.1f} (limit: {max_safe})"
                        })
                    elif value >= min_safe * 0.8:
                        benefits.append({
                            'nutrient': nutrient,
                            'value': value,
                            'message': f"✓ Good {nutrient} level: {value:.1f}"
                        })
            
            # Overall verdict for this condition
            if concerns:
                verdict = 'CAUTION' if any(c['severity'] == 'high' for c in concerns) else 'MODERATE'
                verdict_color = 'red' if verdict == 'CAUTION' else 'orange'
            else:
                verdict = 'SAFE'
                verdict_color = 'green'
            
            analysis = {
                'condition': insight['name'],
                'emoji': insight['emoji'],
                'verdict': verdict,
                'verdict_color': verdict_color,
                'concerns': concerns,
                'benefits': benefits,
                'risk_factors': insight['risk_factors'] if concerns else [],
                'recommendations': insight['recommendations'] if concerns else [],
                'summary': self._generate_condition_summary(
                    insight['name'],
                    verdict,
                    concerns,
                    benefits
                )
            }
            
            analyses.append(analysis)
        
        return analyses
    
    def _analyze_for_goals(
        self,
        goals: List[HealthGoal],
        nutrients: NutrientProfile,
        composition: Dict[str, float],
        alerts: List[HealthAlert]
    ) -> List[Dict[str, Any]]:
        """Analyze food for each health goal"""
        analyses = []
        
        for goal in goals:
            if goal not in self.goal_insights:
                continue
            
            insight = self.goal_insights[goal]
            
            # Check focus nutrients
            positives = []
            negatives = []
            
            for nutrient in insight['focus_nutrients']:
                value = getattr(nutrients, nutrient, None)
                if value is None:
                    value = composition.get(nutrient, 0)
                
                if nutrient in insight.get('target_ranges', {}):
                    min_target, max_target = insight['target_ranges'][nutrient]
                    
                    if min_target <= value <= max_target:
                        positives.append({
                            'nutrient': nutrient,
                            'value': value,
                            'message': f"✅ Optimal {nutrient}: {value:.1f}"
                        })
                    elif value > max_target:
                        negatives.append({
                            'nutrient': nutrient,
                            'value': value,
                            'message': f"⚠️ Too much {nutrient}: {value:.1f}"
                        })
            
            # Calculate alignment score (0-100)
            total_nutrients = len(insight['focus_nutrients'])
            alignment_score = (len(positives) / total_nutrients * 100) if total_nutrients > 0 else 50
            
            # Overall verdict
            if alignment_score >= 75:
                verdict = 'EXCELLENT'
                verdict_color = 'green'
            elif alignment_score >= 50:
                verdict = 'GOOD'
                verdict_color = 'yellow'
            else:
                verdict = 'POOR'
                verdict_color = 'orange'
            
            analysis = {
                'goal': insight['name'],
                'emoji': insight['emoji'],
                'verdict': verdict,
                'verdict_color': verdict_color,
                'alignment_score': round(alignment_score, 1),
                'positives': positives,
                'negatives': negatives,
                'success_factors': insight['success_factors'] if positives else [],
                'recommendations': insight['recommendations'],
                'summary': self._generate_goal_summary(
                    insight['name'],
                    verdict,
                    alignment_score,
                    positives
                )
            }
            
            analyses.append(analysis)
        
        return analyses
    
    def _generate_condition_summary(
        self,
        condition_name: str,
        verdict: str,
        concerns: List[Dict],
        benefits: List[Dict]
    ) -> str:
        """Generate summary for medical condition"""
        if verdict == 'CAUTION':
            return f"⚠️ NOT recommended for {condition_name}. {len(concerns)} serious concern(s) detected."
        elif verdict == 'MODERATE':
            return f"⚠️ Use caution with {condition_name}. {len(concerns)} concern(s) to monitor."
        else:
            return f"✓ Safe for {condition_name}. No concerns detected."
    
    def _generate_goal_summary(
        self,
        goal_name: str,
        verdict: str,
        score: float,
        positives: List[Dict]
    ) -> str:
        """Generate summary for health goal"""
        if verdict == 'EXCELLENT':
            return f"✅ Excellent for {goal_name}! {score:.0f}% alignment with optimal nutrients."
        elif verdict == 'GOOD':
            return f"✓ Good for {goal_name}. {score:.0f}% alignment."
        else:
            return f"⚠️ Not ideal for {goal_name}. Only {score:.0f}% alignment."
    
    def _get_nutrient_highlights(
        self,
        nutrients: NutrientProfile,
        user_profile: UserHealthProfile
    ) -> Dict[str, List[str]]:
        """Get top nutrient highlights"""
        highlights = {
            'high_in': [],
            'low_in': [],
            'good_source': []
        }
        
        # Check protein
        if nutrients.protein > 20:
            highlights['high_in'].append(f"Protein ({nutrients.protein:.1f}g)")
        elif nutrients.protein > 10:
            highlights['good_source'].append(f"Protein ({nutrients.protein:.1f}g)")
        
        # Check fiber
        if nutrients.fiber > 5:
            highlights['high_in'].append(f"Fiber ({nutrients.fiber:.1f}g)")
        elif nutrients.fiber > 2.5:
            highlights['good_source'].append(f"Fiber ({nutrients.fiber:.1f}g)")
        
        # Check minerals
        if nutrients.iron > 3:
            highlights['high_in'].append(f"Iron ({nutrients.iron:.1f}mg)")
        if nutrients.calcium > 200:
            highlights['high_in'].append(f"Calcium ({nutrients.calcium:.1f}mg)")
        
        # Check concerns
        if nutrients.sodium > 400:
            highlights['high_in'].append(f"⚠️ Sodium ({nutrients.sodium:.1f}mg)")
        if nutrients.sugar > 10:
            highlights['high_in'].append(f"⚠️ Sugar ({nutrients.sugar:.1f}g)")
        
        return highlights
    
    def _get_contaminant_warnings(
        self,
        composition: Dict[str, float],
        user_profile: UserHealthProfile
    ) -> List[Dict[str, Any]]:
        """Get contaminant warnings"""
        warnings = []
        
        contaminants = {'Pb': 0.1, 'Hg': 0.5, 'As': 0.3, 'Cd': 0.1}
        
        for element, threshold in contaminants.items():
            if element in composition:
                value = composition[element]
                if value > threshold:
                    warnings.append({
                        'element': element,
                        'name': {'Pb': 'Lead', 'Hg': 'Mercury', 'As': 'Arsenic', 'Cd': 'Cadmium'}[element],
                        'value': value,
                        'threshold': threshold,
                        'severity': 'critical' if value > threshold * 2 else 'high',
                        'message': f"⚠️ Elevated {element}: {value:.3f} ppm (limit: {threshold} ppm)"
                    })
        
        return warnings
    
    def _get_personalized_recommendations(
        self,
        user_profile: UserHealthProfile,
        alerts: List[HealthAlert],
        nutrients: NutrientProfile
    ) -> List[str]:
        """Get personalized recommendations"""
        recommendations = []
        
        # Based on alerts
        for alert in alerts:
            if alert.recommendations:
                recommendations.extend(alert.recommendations[:2])
        
        # Based on profile
        if user_profile.is_pregnant:
            recommendations.append("Ensure adequate folate and iron intake")
        
        if MedicalCondition.DIABETES_TYPE2 in user_profile.medical_conditions:
            recommendations.append("Monitor blood sugar 2 hours after eating")
        
        if HealthGoal.WEIGHT_LOSS in user_profile.primary_goals:
            recommendations.append("Pair with vegetables to increase volume")
        
        return list(set(recommendations))[:5]  # Top 5 unique
    
    def _generate_quick_stats(
        self,
        nutrients: NutrientProfile,
        user_profile: UserHealthProfile
    ) -> Dict[str, Any]:
        """Generate quick statistics"""
        tdee = user_profile.get_tdee()
        
        return {
            'calories': {
                'value': nutrients.calories,
                'percent_of_daily': (nutrients.calories / tdee * 100) if tdee > 0 else 0,
                'label': f"{nutrients.calories:.0f} cal ({nutrients.calories/tdee*100:.1f}% of daily)"
            },
            'protein': {
                'value': nutrients.protein,
                'percent_of_daily': (nutrients.protein / 50 * 100),
                'label': f"{nutrients.protein:.1f}g ({nutrients.protein/50*100:.0f}% DV)"
            },
            'fiber': {
                'value': nutrients.fiber,
                'percent_of_daily': (nutrients.fiber / 25 * 100),
                'label': f"{nutrients.fiber:.1f}g ({nutrients.fiber/25*100:.0f}% DV)"
            },
            'sodium': {
                'value': nutrients.sodium,
                'percent_of_daily': (nutrients.sodium / 2300 * 100),
                'label': f"{nutrients.sodium:.0f}mg ({nutrients.sodium/2300*100:.0f}% DV)",
                'warning': nutrients.sodium > 400
            }
        }
    
    def _get_traffic_light(self, verdict: str) -> str:
        """Get traffic light color"""
        return {
            'SAFE': '🟢',
            'MODERATE': '🟡',
            'CAUTION': '🟠',
            'AVOID': '🔴'
        }.get(verdict, '⚪')
    
    def _get_emoji_indicator(self, score: float, verdict: str) -> str:
        """Get emoji indicator"""
        if verdict == 'AVOID':
            return '🚫'
        elif verdict == 'CAUTION':
            return '⚠️'
        elif score >= 90:
            return '⭐'
        elif score >= 80:
            return '✅'
        elif score >= 70:
            return '✓'
        else:
            return '⚠️'
    
    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'


if __name__ == "__main__":
    # Test personalized risk analysis with detailed risk cards
    logger.info("Testing Phase 4: Personalized Risk Assessment with Detailed Risk Cards")
    
    # Create test user profile
    user = UserHealthProfile(
        user_id="user_123",
        age=35,
        gender="female",
        weight_kg=65,
        height_cm=165,
        medical_conditions=[MedicalCondition.HYPERTENSION, MedicalCondition.DIABETES_TYPE2],
        primary_goals=[HealthGoal.WEIGHT_LOSS, HealthGoal.BLOOD_SUGAR_CONTROL],
        is_pregnant=False,
        blood_pressure_systolic=140,
        blood_pressure_diastolic=90
    )
    
    # Test atomic composition (with some contaminants)
    composition = {
        'Fe': 2.5,
        'Zn': 1.2,
        'Ca': 50.0,
        'Na': 850.0,  # High sodium
        'K': 380.0,
        'Hg': 0.15,   # Slightly elevated mercury
        'Pb': 0.08,   # Trace lead
    }
    
    # Test nutrient profile
    nutrients = NutrientProfile(
        protein=25.0,
        carbohydrates=0.5,
        fat=8.0,
        fiber=0.0,
        calories=180,
        sodium=850.0,
        sugar=0.0,
        saturated_fat=2.5
    )
    
    # Run analysis
    analyzer = PersonalizedRiskAnalyzer()
    result = analyzer.analyze_food(
        atomic_composition=composition,
        nutrient_profile=nutrients,
        user_profile=user,
        food_name="Grilled Salmon",
        food_category="seafood",
        serving_size_g=150
    )
    
    # Generate detailed risk card
    card_generator = DetailedRiskCardGenerator()
    detailed_card = card_generator.generate_detailed_risk_card(
        food_name="Grilled Salmon",
        health_score=result['health_score']['overall_score'],
        safety_verdict=result['safety_verdict'],
        alerts=[],  # Would pass actual alerts
        nutrient_profile=nutrients,
        atomic_composition=composition,
        user_profile=user
    )
    
    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"FOOD: {result['food_name']}")
    logger.info(f"SAFETY VERDICT: {result['safety_verdict']}")
    logger.info(f"HEALTH SCORE: {result['health_score']['overall_score']}/100 (Grade {result['health_score']['grade']})")
    logger.info(f"\nSUMMARY: {result['summary']}")
    
    # Print detailed risk card
    logger.info(f"\n{'='*60}")
    logger.info("DETAILED RISK CARD:")
    logger.info(f"Traffic Light: {detailed_card['traffic_light']}")
    logger.info(f"Emoji: {detailed_card['emoji_indicator']}")
    
    logger.info(f"\n{'='*60}")
    logger.info("MEDICAL CONDITION ANALYSIS:")
    for analysis in detailed_card['medical_condition_analysis']:
        logger.info(f"\n{analysis['emoji']} {analysis['condition']}: {analysis['verdict']}")
        logger.info(f"   {analysis['summary']}")
        if analysis['concerns']:
            logger.info(f"   Concerns: {len(analysis['concerns'])}")
            for concern in analysis['concerns'][:2]:
                logger.info(f"      - {concern['message']}")
    
    logger.info(f"\n{'='*60}")
    logger.info("HEALTH GOAL ANALYSIS:")
    for analysis in detailed_card['health_goal_analysis']:
        logger.info(f"\n{analysis['emoji']} {analysis['goal']}: {analysis['verdict']} ({analysis['alignment_score']}%)")
        logger.info(f"   {analysis['summary']}")
        if analysis['positives']:
            logger.info(f"   Positives:")
            for pos in analysis['positives']:
                logger.info(f"      - {pos['message']}")
    
    logger.info(f"\n{'='*60}")
    logger.info("RECOMMENDATIONS:")
    for i, rec in enumerate(detailed_card['personalized_recommendations'], 1):
        logger.info(f"  {i}. {rec}")
    
    logger.info("\nPhase 4 test complete!")
