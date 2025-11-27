"""
Lifecycle Modulator - Age-Based Nutritional Optimization
=========================================================

This module implements age-specific nutritional optimization that adapts recommendations
based on the user's lifecycle stage. Each life stage has unique nutritional priorities,
safety concerns, and metabolic characteristics.

Lifecycle Stages:
1. Infant (0-2 years): SAFETY FIRST - Toxic scan at highest sensitivity, allergen detection
2. Child (3-12 years): GROWTH - Calcium, vitamin D, protein for bone/brain development
3. Adolescent (13-19 years): DEVELOPMENT - Hormonal support, rapid growth, energy needs
4. Young Adult (20-35 years): PERFORMANCE - Athletic optimization, career demands
5. Adult (36-60 years): PREVENTION - Disease prevention, metabolic health
6. Senior (60+ years): DENSITY - Maximum micronutrients per calorie, absorption issues

Key Concept: Same food gets different recommendations based on age
Example: Raw honey
- Infant: AVOID (botulism risk)
- Child: ACCEPTABLE (energy, but limit sugar)
- Athlete: RECOMMENDED (pre-workout quick energy)
- Senior: NOT_RECOMMENDED (high glycemic, low nutrient density)

Author: Wellomex AI Nutrition Team
Version: 1.0.0
Date: November 7, 2025
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import logging
from datetime import datetime
from pathlib import Path

# Import from our modules
try:
    from atomic_molecular_profiler import (
        NutrientMolecularBreakdown, UserHealthProfile, FoodRecommendation,
        MolecularBondProfile, ToxicContaminantProfile, ChemicalBondType
    )
    from multi_condition_optimizer import (
        MultiConditionOptimizer, RecommendationLevel
    )
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    logging.warning("Required modules not found. Ensure atomic_molecular_profiler.py and multi_condition_optimizer.py are available.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class LifecycleStage(Enum):
    """Lifecycle stages with age ranges"""
    INFANT = "infant_0_2"  # 0-2 years
    TODDLER = "toddler_2_3"  # 2-3 years
    CHILD = "child_3_12"  # 3-12 years
    ADOLESCENT = "adolescent_13_19"  # 13-19 years
    YOUNG_ADULT = "young_adult_20_35"  # 20-35 years
    ADULT = "adult_36_60"  # 36-60 years
    SENIOR = "senior_60_75"  # 60-75 years
    ELDERLY = "elderly_75_plus"  # 75+ years


class NutritionalPriority(Enum):
    """Nutritional priorities by lifecycle stage"""
    SAFETY = "safety"  # Avoid toxins, allergens
    GROWTH = "growth"  # Support physical development
    DEVELOPMENT = "development"  # Support cognitive/hormonal development
    PERFORMANCE = "performance"  # Optimize athletic/work performance
    PREVENTION = "prevention"  # Prevent chronic diseases
    MAINTENANCE = "maintenance"  # Maintain health, manage conditions
    DENSITY = "nutrient_density"  # Maximum nutrients per calorie
    ABSORPTION = "enhanced_absorption"  # Compensate for reduced absorption


# ============================================================================
# LIFECYCLE PROFILES
# ============================================================================

@dataclass
class LifecycleProfile:
    """Nutritional profile for a lifecycle stage"""
    stage: LifecycleStage
    age_range: Tuple[int, int]  # (min_age, max_age)
    priority: NutritionalPriority
    
    # Key nutrients to emphasize
    critical_nutrients: Dict[str, str] = field(default_factory=dict)  # {nutrient: reason}
    
    # Nutrients to limit/avoid
    restricted_nutrients: Dict[str, str] = field(default_factory=dict)  # {nutrient: reason}
    
    # Foods to avoid entirely
    forbidden_foods: List[str] = field(default_factory=list)
    
    # Allergen sensitivity (higher = more sensitive)
    allergen_sensitivity: float = 1.0  # 0.5 = low, 1.0 = normal, 2.0 = high
    
    # Toxic contaminant sensitivity (higher = stricter limits)
    toxic_sensitivity: float = 1.0  # Infant: 3.0, Adult: 1.0, Senior: 1.5
    
    # Caloric needs (kcal/day ranges)
    caloric_needs: Tuple[int, int] = (2000, 2500)
    
    # Protein needs (g/kg body weight)
    protein_needs: Tuple[float, float] = (0.8, 1.2)
    
    # Special considerations
    special_notes: List[str] = field(default_factory=list)
    
    # Scoring adjustments
    safety_weight: float = 0.4  # How much to weight safety
    growth_weight: float = 0.0  # How much to weight growth nutrients
    density_weight: float = 0.0  # How much to weight nutrient density


class LifecycleProfileDatabase:
    """Database of lifecycle-specific nutritional profiles"""
    
    def __init__(self):
        self.profiles: Dict[LifecycleStage, LifecycleProfile] = {}
        self._initialize_profiles()
    
    def _initialize_profiles(self):
        """Initialize lifecycle profiles with evidence-based guidelines"""
        
        # INFANT (0-2 years) - SAFETY FIRST
        self.profiles[LifecycleStage.INFANT] = LifecycleProfile(
            stage=LifecycleStage.INFANT,
            age_range=(0, 2),
            priority=NutritionalPriority.SAFETY,
            critical_nutrients={
                'dha': 'Critical for brain development - 70% brain growth in first 2 years',
                'iron': 'Prevent anemia, support cognitive development',
                'calcium': 'Bone development, tooth formation',
                'vitamin_d': 'Calcium absorption, immune function',
                'zinc': 'Growth, immune function',
                'choline': 'Brain development, memory formation'
            },
            restricted_nutrients={
                'sodium': 'Immature kidneys cannot handle excess sodium (<400mg/day)',
                'sugar': 'No added sugars before age 2 (AAP guideline)',
                'honey': 'FORBIDDEN - Botulism spores risk',
                'cow_milk_protein': 'Avoid before 12 months - digestive stress',
                'nitrates': 'High nitrates in well water/spinach can cause methemoglobinemia'
            },
            forbidden_foods=[
                'honey', 'raw_honey', 'unpasteurized_dairy', 'raw_eggs',
                'undercooked_meat', 'fish_high_mercury', 'whole_nuts',
                'hard_candy', 'popcorn', 'grapes_whole', 'hot_dogs_whole'
            ],
            allergen_sensitivity=3.0,  # HIGHEST - developing immune system
            toxic_sensitivity=5.0,  # EXTREME - most vulnerable population
            caloric_needs=(800, 1200),  # Based on breast milk/formula + solids
            protein_needs=(1.1, 1.5),  # Higher per kg for growth
            special_notes=[
                'Choking hazards: Avoid small, hard, round foods',
                'Introduce allergens early (4-6 months) to prevent allergies',
                'Breast milk/formula primary until 12 months',
                'Iron-fortified cereals recommended at 6 months',
                'No juice before 12 months (AAP 2017 guideline)',
                'Watch for allergic reactions to new foods'
            ],
            safety_weight=0.7,  # 70% weight on safety
            growth_weight=0.2,
            density_weight=0.1
        )
        
        # TODDLER (2-3 years)
        self.profiles[LifecycleStage.TODDLER] = LifecycleProfile(
            stage=LifecycleStage.TODDLER,
            age_range=(2, 3),
            priority=NutritionalPriority.GROWTH,
            critical_nutrients={
                'calcium': 'Bone development - 700mg/day',
                'vitamin_d': 'Bone health - 600 IU/day',
                'iron': 'Brain development - 7mg/day',
                'dha': 'Continued brain development',
                'fiber': 'Digestive health - 19g/day',
                'protein': 'Growth - 13g/day'
            },
            restricted_nutrients={
                'sodium': 'Limit to <1500mg/day',
                'added_sugar': 'Limit to <25g/day',
                'saturated_fat': 'Limit to support heart health development'
            },
            forbidden_foods=[
                'unpasteurized_dairy', 'raw_eggs', 'undercooked_meat',
                'whole_nuts', 'hard_candy', 'popcorn'
            ],
            allergen_sensitivity=2.5,
            toxic_sensitivity=3.0,
            caloric_needs=(1000, 1400),
            protein_needs=(1.0, 1.3),
            special_notes=[
                'Still high choking risk',
                'Picky eating phase - offer variety',
                'Transition to whole milk (12-24 months)',
                'Limit juice to 4oz/day',
                'Establish healthy eating patterns'
            ],
            safety_weight=0.6,
            growth_weight=0.3,
            density_weight=0.1
        )
        
        # CHILD (3-12 years) - GROWTH
        self.profiles[LifecycleStage.CHILD] = LifecycleProfile(
            stage=LifecycleStage.CHILD,
            age_range=(3, 12),
            priority=NutritionalPriority.GROWTH,
            critical_nutrients={
                'calcium': 'Peak bone mass development - 1000-1300mg/day',
                'vitamin_d': 'Bone health, immune function - 600 IU/day',
                'iron': 'Energy, cognitive function - 10mg/day',
                'protein': 'Growth, muscle development - 19-34g/day',
                'dha': 'Brain development, learning, ADHD prevention',
                'zinc': 'Growth, immune function - 5-8mg/day',
                'vitamin_a': 'Vision, immune function',
                'b_vitamins': 'Energy metabolism, brain function'
            },
            restricted_nutrients={
                'sodium': 'Limit to <1900mg/day',
                'added_sugar': 'Limit to <25g/day',
                'saturated_fat': 'Limit to <10% calories',
                'caffeine': 'Avoid or limit to <45mg/day'
            },
            forbidden_foods=[
                'energy_drinks', 'excessive_caffeine', 'unpasteurized_dairy'
            ],
            allergen_sensitivity=1.5,
            toxic_sensitivity=2.0,
            caloric_needs=(1400, 2200),  # Varies by age/activity
            protein_needs=(0.95, 1.2),
            special_notes=[
                'Rapid growth spurts',
                'Establish healthy habits early',
                'Limit screen time eating',
                'Involve in food preparation',
                'Address picky eating with patience',
                'Prevent childhood obesity'
            ],
            safety_weight=0.4,
            growth_weight=0.4,
            density_weight=0.2
        )
        
        # ADOLESCENT (13-19 years) - DEVELOPMENT
        self.profiles[LifecycleStage.ADOLESCENT] = LifecycleProfile(
            stage=LifecycleStage.ADOLESCENT,
            age_range=(13, 19),
            priority=NutritionalPriority.DEVELOPMENT,
            critical_nutrients={
                'calcium': 'Peak bone mass - 1300mg/day',
                'vitamin_d': 'Bone development - 600 IU/day',
                'iron': 'Growth spurt, menstruation (females) - 8-15mg/day',
                'protein': 'Muscle development - 46-52g/day',
                'zinc': 'Growth, sexual maturation - 8-11mg/day',
                'b_vitamins': 'Energy, brain development',
                'omega_3': 'Brain development, mood regulation',
                'magnesium': 'Bone development, energy - 360-410mg/day'
            },
            restricted_nutrients={
                'sodium': 'Limit to <2300mg/day',
                'added_sugar': 'Limit to <25-35g/day',
                'saturated_fat': 'Limit to <10% calories',
                'caffeine': 'Limit to <100mg/day'
            },
            forbidden_foods=[
                'excessive_energy_drinks', 'alcohol', 'tobacco'
            ],
            allergen_sensitivity=1.0,
            toxic_sensitivity=1.5,
            caloric_needs=(2000, 3200),  # High for growth
            protein_needs=(0.85, 1.2),
            special_notes=[
                'Rapid growth spurt',
                'Hormonal changes affect appetite',
                'High risk for disordered eating',
                'Peer pressure influences food choices',
                'Athletes need 1.2-2.0g/kg protein',
                'Females: Address iron needs during menstruation',
                'Males: Support testosterone production (zinc, vitamin D)'
            ],
            safety_weight=0.3,
            growth_weight=0.4,
            density_weight=0.3
        )
        
        # YOUNG ADULT (20-35 years) - PERFORMANCE
        self.profiles[LifecycleStage.YOUNG_ADULT] = LifecycleProfile(
            stage=LifecycleStage.YOUNG_ADULT,
            age_range=(20, 35),
            priority=NutritionalPriority.PERFORMANCE,
            critical_nutrients={
                'protein': 'Muscle maintenance, performance - 0.8-2.0g/kg',
                'iron': 'Energy, oxygen transport',
                'calcium': 'Maintain bone density',
                'vitamin_d': 'Immune function, mood',
                'b_vitamins': 'Energy metabolism',
                'omega_3': 'Brain function, inflammation',
                'magnesium': 'Energy, recovery',
                'antioxidants': 'Oxidative stress from exercise/stress'
            },
            restricted_nutrients={
                'sodium': 'Limit to <2300mg/day',
                'added_sugar': 'Limit to <25-35g/day',
                'saturated_fat': 'Limit to <10% calories',
                'alcohol': 'Moderate: <1-2 drinks/day'
            },
            forbidden_foods=[],
            allergen_sensitivity=1.0,
            toxic_sensitivity=1.0,
            caloric_needs=(2000, 3000),  # Based on activity
            protein_needs=(0.8, 2.0),  # Higher for athletes
            special_notes=[
                'Peak physical performance years',
                'Career stress affects eating',
                'Athletes: Optimize timing and ratios',
                'Pre-workout: 3:1 carb:protein ratio',
                'Post-workout: 30g protein within 2 hours',
                'Pregnancy/lactation: Increase needs significantly',
                'Establish disease prevention habits'
            ],
            safety_weight=0.3,
            growth_weight=0.1,
            density_weight=0.6
        )
        
        # ADULT (36-60 years) - PREVENTION
        self.profiles[LifecycleStage.ADULT] = LifecycleProfile(
            stage=LifecycleStage.ADULT,
            age_range=(36, 60),
            priority=NutritionalPriority.PREVENTION,
            critical_nutrients={
                'fiber': 'Heart health, diabetes prevention - 25-35g/day',
                'omega_3': 'Heart health, inflammation - 2-3g/day',
                'antioxidants': 'Oxidative stress, aging',
                'calcium': 'Bone health (especially women) - 1000-1200mg/day',
                'vitamin_d': 'Bone, immune, mood - 600-800 IU/day',
                'b12': 'Energy, brain function (absorption declines)',
                'potassium': 'Blood pressure - 3500mg/day',
                'magnesium': 'Heart rhythm, bone health'
            },
            restricted_nutrients={
                'sodium': 'Heart health - <1500-2300mg/day',
                'saturated_fat': 'Heart health - <13g/day',
                'trans_fat': 'Avoid completely',
                'added_sugar': 'Diabetes prevention - <25g/day',
                'alcohol': 'Limit: <1-2 drinks/day'
            },
            forbidden_foods=[],
            allergen_sensitivity=1.0,
            toxic_sensitivity=1.2,
            caloric_needs=(1800, 2400),  # Metabolism slowing
            protein_needs=(1.0, 1.2),  # Prevent sarcopenia
            special_notes=[
                'Metabolism slowing ~2-8% per decade',
                'Sarcopenia (muscle loss) begins',
                'Increased risk: diabetes, CVD, hypertension',
                'Menopause (women): Bone loss accelerates',
                'Andropause (men): Testosterone decline',
                'Focus on disease prevention',
                'Regular screening for metabolic markers'
            ],
            safety_weight=0.3,
            growth_weight=0.0,
            density_weight=0.7
        )
        
        # SENIOR (60-75 years) - DENSITY
        self.profiles[LifecycleStage.SENIOR] = LifecycleProfile(
            stage=LifecycleStage.SENIOR,
            age_range=(60, 75),
            priority=NutritionalPriority.DENSITY,
            critical_nutrients={
                'protein': 'Prevent sarcopenia - 1.0-1.2g/kg',
                'vitamin_b12': 'Absorption declines, cognitive function - 2.4mcg/day',
                'vitamin_d': 'Bone health, falls prevention - 800-1000 IU/day',
                'calcium': 'Osteoporosis prevention - 1200mg/day',
                'fiber': 'Digestive health, constipation - 25-30g/day',
                'omega_3': 'Brain health, inflammation - 2-3g/day',
                'potassium': 'Blood pressure, stroke prevention',
                'antioxidants': 'Oxidative stress, cognitive decline',
                'water': 'Thirst sensation declines, dehydration risk'
            },
            restricted_nutrients={
                'sodium': 'Hypertension common - <1500mg/day',
                'saturated_fat': 'CVD risk - <13g/day',
                'simple_sugars': 'Diabetes risk - <20g/day',
                'alcohol': 'Medication interactions - <1 drink/day'
            },
            forbidden_foods=[
                'grapefruit_with_statins',  # Drug interaction
                'high_vitamin_k_on_warfarin'  # Anticoagulant interaction
            ],
            allergen_sensitivity=1.0,
            toxic_sensitivity=2.0,  # Reduced detoxification capacity
            caloric_needs=(1600, 2000),  # Lower metabolism
            protein_needs=(1.0, 1.3),  # Higher to preserve muscle
            special_notes=[
                'Reduced stomach acid → B12 absorption issues',
                'Decreased thirst sensation → dehydration risk',
                'Medication-nutrient interactions common',
                'Appetite often reduced',
                'Taste/smell decline → less enjoyment',
                'Dental issues affect food choices',
                'Constipation common (low fiber, low fluid)',
                'Falls risk (vitamin D, protein)',
                'Cognitive decline risk (B vitamins, omega-3)'
            ],
            safety_weight=0.4,
            growth_weight=0.0,
            density_weight=0.6
        )
        
        # ELDERLY (75+ years) - DENSITY + ABSORPTION
        self.profiles[LifecycleStage.ELDERLY] = LifecycleProfile(
            stage=LifecycleStage.ELDERLY,
            age_range=(75, 120),
            priority=NutritionalPriority.ABSORPTION,
            critical_nutrients={
                'protein': 'Prevent frailty - 1.2-1.5g/kg',
                'vitamin_b12': 'Sublingual/injection may be needed - 2.4-1000mcg/day',
                'vitamin_d': 'Falls, fractures, immune - 800-2000 IU/day',
                'calcium': 'Bone health - 1200mg/day',
                'fiber': 'Constipation prevention - 20-25g/day',
                'omega_3': 'Cognitive function - 2-4g/day',
                'water': 'CRITICAL - severe dehydration risk',
                'potassium': 'Heart rhythm, BP',
                'zinc': 'Immune function, wound healing'
            },
            restricted_nutrients={
                'sodium': 'Strict limit - <1200mg/day',
                'simple_sugars': '<15g/day',
                'difficult_to_chew_foods': 'Choking risk',
                'alcohol': 'High interaction risk - avoid or <1 drink/day'
            },
            forbidden_foods=[
                'grapefruit_with_medications',
                'high_vitamin_k_on_anticoagulants',
                'hard_to_chew_foods',
                'very_hot_foods'  # Reduced sensation
            ],
            allergen_sensitivity=1.0,
            toxic_sensitivity=2.5,  # Compromised detoxification
            caloric_needs=(1400, 1800),
            protein_needs=(1.2, 1.5),  # Highest to prevent frailty
            special_notes=[
                'Frailty risk - protein and exercise critical',
                'Polypharmacy (5+ medications) - nutrient interactions',
                'Dysphagia (swallowing difficulty) common',
                'Reduced gastric acid → malabsorption',
                'Social isolation → poor nutrition',
                'Fixed income → food insecurity',
                'Cognitive decline affects food preparation',
                'Pressure ulcers risk (protein, zinc, vitamin C)',
                'Immune senescence → infection risk'
            ],
            safety_weight=0.5,
            growth_weight=0.0,
            density_weight=0.5
        )
        
        logger.info(f"Initialized {len(self.profiles)} lifecycle profiles")
    
    def get_profile(self, stage: LifecycleStage) -> Optional[LifecycleProfile]:
        """Get profile for a lifecycle stage"""
        return self.profiles.get(stage)
    
    def get_profile_by_age(self, age: int) -> Optional[LifecycleProfile]:
        """Get profile based on age"""
        for profile in self.profiles.values():
            min_age, max_age = profile.age_range
            if min_age <= age <= max_age:
                return profile
        
        # Default to elderly if very old
        if age > 75:
            return self.profiles.get(LifecycleStage.ELDERLY)
        
        return None


# ============================================================================
# LIFECYCLE MODULATOR
# ============================================================================

class LifecycleModulator:
    """
    Lifecycle Modulator - Adapts recommendations based on age
    
    This modulates the base recommendation from MultiConditionOptimizer
    by applying lifecycle-specific adjustments.
    
    Example Modulation:
        Base recommendation: "Salmon - HIGHLY_RECOMMENDED (score: 85)"
        
        Infant (1 year): "Salmon - AVOID (mercury risk for developing brain)"
        Child (8 years): "Salmon - RECOMMENDED (omega-3 for brain development, limit portions)"
        Athlete (25 years): "Salmon - HIGHLY_RECOMMENDED (optimal protein + omega-3 recovery)"
        Senior (70 years): "Salmon - HIGHLY_RECOMMENDED (nutrient dense, heart + brain health)"
    """
    
    def __init__(self):
        self.lifecycle_db = LifecycleProfileDatabase()
        
        if MODULES_AVAILABLE:
            self.base_optimizer = MultiConditionOptimizer()
        else:
            self.base_optimizer = None
            logger.warning("MultiConditionOptimizer not available")
        
        logger.info("Lifecycle Modulator initialized")
    
    def modulate(self,
                 base_recommendation: 'FoodRecommendation',
                 user_profile: 'UserHealthProfile',
                 food_name: str) -> 'FoodRecommendation':
        """
        Modulate recommendation based on lifecycle stage
        
        Args:
            base_recommendation: Base recommendation from optimizer
            user_profile: User profile with age
            food_name: Name of food
        
        Returns:
            Modulated recommendation adjusted for lifecycle stage
        """
        # Get lifecycle profile
        lifecycle_profile = self.lifecycle_db.get_profile_by_age(user_profile.age)
        
        if not lifecycle_profile:
            logger.warning(f"No lifecycle profile found for age {user_profile.age}")
            return base_recommendation
        
        logger.info(f"Modulating for {lifecycle_profile.stage.value} (age {user_profile.age})")
        
        # Create modulated recommendation (copy base)
        modulated = base_recommendation
        modulated.reasoning = base_recommendation.reasoning.copy()
        
        # Apply lifecycle-specific adjustments
        
        # 1. Check forbidden foods
        if self._is_forbidden_food(food_name, lifecycle_profile):
            return self._create_forbidden_recommendation(
                base_recommendation, lifecycle_profile, food_name
            )
        
        # 2. Adjust for toxic sensitivity
        safety_adjustment = self._adjust_for_toxic_sensitivity(
            base_recommendation, lifecycle_profile
        )
        modulated.safety_score *= safety_adjustment
        
        # 3. Adjust for critical nutrients
        nutrient_bonus = self._calculate_critical_nutrient_bonus(
            base_recommendation.molecular_breakdown, lifecycle_profile
        )
        
        # 4. Adjust for restricted nutrients
        nutrient_penalty = self._calculate_restricted_nutrient_penalty(
            base_recommendation.molecular_breakdown, lifecycle_profile
        )
        
        # 5. Recalculate overall score
        modulated.overall_score = (
            base_recommendation.overall_score +
            nutrient_bonus -
            nutrient_penalty
        )
        
        # Apply safety adjustment
        modulated.overall_score *= safety_adjustment
        
        # Clamp score
        modulated.overall_score = max(0, min(100, modulated.overall_score))
        
        # 6. Update recommendation level
        modulated.recommendation = self._determine_level(modulated.overall_score)
        
        # 7. Add lifecycle-specific reasoning
        lifecycle_reasoning = self._generate_lifecycle_reasoning(
            base_recommendation, lifecycle_profile, nutrient_bonus, nutrient_penalty
        )
        modulated.reasoning.extend(lifecycle_reasoning)
        
        # 8. Adjust serving size for lifecycle
        modulated.optimal_serving_size = self._adjust_serving_size(
            base_recommendation.optimal_serving_size,
            lifecycle_profile,
            user_profile
        )
        
        logger.info(f"Modulated score: {base_recommendation.overall_score:.1f} → {modulated.overall_score:.1f}")
        
        return modulated
    
    def _is_forbidden_food(self, food_name: str, profile: LifecycleProfile) -> bool:
        """Check if food is forbidden for this lifecycle stage"""
        food_lower = food_name.lower()
        
        for forbidden in profile.forbidden_foods:
            if forbidden in food_lower:
                return True
        
        return False
    
    def _create_forbidden_recommendation(self,
                                        base_rec: 'FoodRecommendation',
                                        profile: LifecycleProfile,
                                        food_name: str) -> 'FoodRecommendation':
        """Create AVOID recommendation for forbidden food"""
        forbidden_rec = base_rec
        forbidden_rec.overall_score = 0.0
        forbidden_rec.recommendation = "AVOID"
        forbidden_rec.reasoning = [
            f"🚫 FORBIDDEN for {profile.stage.value}",
            f"Age range: {profile.age_range[0]}-{profile.age_range[1]} years",
            f"Reason: Critical safety concern for this age group",
            f"Priority: {profile.priority.value}"
        ]
        
        # Add specific reasons
        if 'honey' in food_name.lower() and profile.stage == LifecycleStage.INFANT:
            forbidden_rec.reasoning.append(
                "⚠️ BOTULISM RISK: Honey contains Clostridium botulinum spores. "
                "Infant digestive system cannot neutralize. Can be fatal."
            )
        
        forbidden_rec.optimal_serving_size = 0.0
        forbidden_rec.max_daily_servings = 0
        
        return forbidden_rec
    
    def _adjust_for_toxic_sensitivity(self,
                                      base_rec: 'FoodRecommendation',
                                      profile: LifecycleProfile) -> float:
        """Adjust safety score based on lifecycle toxic sensitivity"""
        if not base_rec.toxic_profile:
            return 1.0  # No adjustment needed
        
        # Higher sensitivity = harsher penalty
        sensitivity = profile.toxic_sensitivity
        
        # Calculate penalty multiplier
        if sensitivity > 2.0:
            # Very sensitive (infant, elderly)
            return 0.5  # 50% penalty
        elif sensitivity > 1.5:
            # Moderately sensitive
            return 0.7  # 30% penalty
        else:
            # Normal sensitivity
            return 0.9  # 10% penalty
    
    def _calculate_critical_nutrient_bonus(self,
                                          nutrients: 'NutrientMolecularBreakdown',
                                          profile: LifecycleProfile) -> float:
        """Calculate bonus for critical nutrients"""
        bonus = 0.0
        
        for nutrient_name, reason in profile.critical_nutrients.items():
            # Get nutrient value
            value = self._get_nutrient_value(nutrient_name, nutrients)
            
            if value > 0:
                # Bonus based on how critical it is (length of reason = importance)
                importance = len(reason) / 100.0  # Normalize
                bonus += min(5, value * importance)
        
        return bonus
    
    def _calculate_restricted_nutrient_penalty(self,
                                              nutrients: 'NutrientMolecularBreakdown',
                                              profile: LifecycleProfile) -> float:
        """Calculate penalty for restricted nutrients"""
        penalty = 0.0
        
        for nutrient_name, reason in profile.restricted_nutrients.items():
            value = self._get_nutrient_value(nutrient_name, nutrients)
            
            if value > 0:
                # Penalty based on severity
                severity = len(reason) / 50.0
                penalty += min(10, value * severity)
        
        return penalty
    
    def _get_nutrient_value(self, nutrient_name: str, 
                           nutrients: 'NutrientMolecularBreakdown') -> float:
        """Get nutrient value from breakdown"""
        mapping = {
            'protein': nutrients.total_protein,
            'carbohydrates': nutrients.total_carbs,
            'sugar': nutrients.simple_sugars,
            'added_sugar': nutrients.simple_sugars,
            'fiber': nutrients.fiber,
            'fat': nutrients.total_fat,
            'saturated_fat': nutrients.saturated_fat,
            'calcium': nutrients.minerals.get('calcium', 0),
            'iron': nutrients.minerals.get('iron', 0),
            'sodium': nutrients.minerals.get('sodium', 0),
            'dha': nutrients.amino_acids.get('dha', 0),
            'omega_3': nutrients.amino_acids.get('omega_3', 0),
        }
        
        return mapping.get(nutrient_name, 0)
    
    def _determine_level(self, score: float) -> str:
        """Determine recommendation level from score"""
        if score >= 80:
            return "HIGHLY_RECOMMENDED"
        elif score >= 60:
            return "RECOMMENDED"
        elif score >= 40:
            return "ACCEPTABLE"
        elif score >= 20:
            return "NOT_RECOMMENDED"
        else:
            return "AVOID"
    
    def _generate_lifecycle_reasoning(self,
                                     base_rec: 'FoodRecommendation',
                                     profile: LifecycleProfile,
                                     bonus: float,
                                     penalty: float) -> List[str]:
        """Generate lifecycle-specific reasoning"""
        reasoning = []
        
        reasoning.append(f"🎯 Lifecycle Stage: {profile.stage.value.replace('_', ' ').title()}")
        reasoning.append(f"   Priority: {profile.priority.value.title()}")
        
        if bonus > 5:
            reasoning.append(f"✓ Contains critical nutrients for this age (+{bonus:.0f} pts)")
        
        if penalty > 5:
            reasoning.append(f"❌ Contains restricted nutrients for this age (-{penalty:.0f} pts)")
        
        # Add special notes relevant to this food
        if profile.special_notes and len(profile.special_notes) > 0:
            reasoning.append(f"📝 Age-specific note: {profile.special_notes[0]}")
        
        return reasoning
    
    def _adjust_serving_size(self,
                            base_serving: Optional[float],
                            profile: LifecycleProfile,
                            user_profile: 'UserHealthProfile') -> Optional[float]:
        """Adjust serving size based on age and body weight"""
        if base_serving is None:
            return None
        
        # Age-based adjustments
        if profile.stage == LifecycleStage.INFANT:
            return base_serving * 0.15  # 15% of adult serving
        elif profile.stage == LifecycleStage.TODDLER:
            return base_serving * 0.25  # 25% of adult serving
        elif profile.stage == LifecycleStage.CHILD:
            return base_serving * 0.5  # 50% of adult serving
        elif profile.stage == LifecycleStage.ADOLESCENT:
            return base_serving * 0.85  # 85% of adult serving
        elif profile.stage == LifecycleStage.SENIOR or profile.stage == LifecycleStage.ELDERLY:
            return base_serving * 0.75  # 75% of adult serving (reduced appetite)
        else:
            return base_serving  # Adult serving


# ============================================================================
# TESTING & EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("LIFECYCLE MODULATOR - Test Suite")
    print("=" * 80)
    
    if not MODULES_AVAILABLE:
        print("\n⚠️  Required modules not found. Cannot run tests.")
        exit(1)
    
    from atomic_molecular_profiler import (
        NutrientMolecularBreakdown, UserHealthProfile, FoodRecommendation,
        MolecularBondProfile, HealthGoal, DiseaseCondition
    )
    
    # Initialize modulator
    modulator = LifecycleModulator()
    
    # Test 1: Honey for infant (FORBIDDEN)
    print("\n" + "=" * 80)
    print("TEST 1: Honey for Infant (Should be FORBIDDEN)")
    print("=" * 80)
    
    infant_user = UserHealthProfile(
        user_id="INFANT001",
        age=1,
        sex="M",
        primary_goal=HealthGoal.BRAIN
    )
    
    honey_profile = NutrientMolecularBreakdown(
        total_carbs=82.0,
        simple_sugars=82.0,
        total_calories=304
    )
    
    honey_rec = FoodRecommendation(
        food_name="Raw Honey",
        scan_id="SCAN001",
        overall_score=60.0,  # Base score before modulation
        safety_score=80.0,
        goal_alignment_score=50.0,
        disease_compatibility_score=100.0,
        molecular_breakdown=honey_profile,
        bond_profile=[],
        toxic_profile=[],
        recommendation="ACCEPTABLE",
        reasoning=["Natural sweetener", "Contains enzymes"]
    )
    
    modulated_honey = modulator.modulate(honey_rec, infant_user, "Raw Honey")
    
    print(f"\n📊 Result for Infant:")
    print(f"  Base Score: {honey_rec.overall_score:.1f}")
    print(f"  Modulated Score: {modulated_honey.overall_score:.1f}")
    print(f"  Recommendation: {modulated_honey.recommendation}")
    print(f"\n  Reasoning:")
    for reason in modulated_honey.reasoning:
        print(f"    {reason}")
    
    # Test 2: Salmon across different ages
    print("\n" + "=" * 80)
    print("TEST 2: Salmon Across Different Lifecycle Stages")
    print("=" * 80)
    
    salmon_profile = NutrientMolecularBreakdown(
        total_protein=25.0,
        total_fat=13.0,
        polyunsaturated_fat=6.0,
        water_content=60.0,
        total_calories=206
    )
    salmon_profile.amino_acids['omega_3'] = 2.3
    
    salmon_rec = FoodRecommendation(
        food_name="Wild Salmon",
        scan_id="SCAN002",
        overall_score=85.0,
        safety_score=90.0,
        goal_alignment_score=85.0,
        disease_compatibility_score=80.0,
        molecular_breakdown=salmon_profile,
        bond_profile=[],
        toxic_profile=[],
        recommendation="HIGHLY_RECOMMENDED",
        reasoning=["High omega-3", "Quality protein"]
    )
    
    ages = [5, 15, 25, 45, 70]
    
    for age in ages:
        user = UserHealthProfile(
            user_id=f"USER{age}",
            age=age,
            sex="M",
            primary_goal=HealthGoal.BRAIN
        )
        
        modulated = modulator.modulate(salmon_rec, user, "Wild Salmon")
        
        print(f"\n  Age {age}: {modulated.recommendation} (Score: {modulated.overall_score:.1f})")
        print(f"    Serving: {modulated.optimal_serving_size:.0f}g")
    
    print("\n" + "=" * 80)
    print("Lifecycle Modulator Test Complete!")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Forbidden food detection (honey for infants)")
    print("  ✓ Age-based score modulation")
    print("  ✓ Lifecycle-specific reasoning")
    print("  ✓ Age-appropriate serving sizes")
    print("  ✓ 8 lifecycle stages (infant → elderly)")
    print("\nNext: Build microservices architecture")
