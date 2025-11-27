"""
Intelligent Meal Planner with Personalization Engine
Provides personalized meal recommendations based on:
- User goals (weight loss/gain, muscle building, disease management)
- Life stage (infant, child, adolescent, adult, elderly, pregnant, lactating)
- Health conditions (diabetes, hypertension, autoimmune, allergies, etc.)
- Nutrient tracking with quality/quantity monitoring
- Precise portion recommendations

Part of AI Nutrition System - Phase 3
Author: AI Nutrition System
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from decimal import Decimal
from datetime import datetime, date, timedelta
import math


class LifeStage(Enum):
    """Life stages with different nutritional needs"""
    INFANT_0_6M = "infant_0_6_months"  # 0-6 months
    INFANT_7_12M = "infant_7_12_months"  # 7-12 months
    TODDLER = "toddler"  # 1-3 years
    CHILD = "child"  # 4-8 years
    ADOLESCENT_MALE = "adolescent_male"  # 9-18 years
    ADOLESCENT_FEMALE = "adolescent_female"  # 9-18 years
    ADULT_MALE = "adult_male"  # 19-50 years
    ADULT_FEMALE = "adult_female"  # 19-50 years
    ELDERLY_MALE = "elderly_male"  # 51+ years
    ELDERLY_FEMALE = "elderly_female"  # 51+ years
    PREGNANT = "pregnant"  # Any trimester
    LACTATING = "lactating"  # Breastfeeding


class ActivityLevel(Enum):
    """Physical activity levels for TDEE calculation"""
    SEDENTARY = 1.2  # Little or no exercise
    LIGHT = 1.375  # Light exercise 1-3 days/week
    MODERATE = 1.55  # Moderate exercise 3-5 days/week
    ACTIVE = 1.725  # Heavy exercise 6-7 days/week
    VERY_ACTIVE = 1.9  # Very intense exercise, physical job


class HealthGoal(Enum):
    """User health and fitness goals"""
    WEIGHT_LOSS = "weight_loss"
    WEIGHT_GAIN = "weight_gain"
    MUSCLE_BUILDING = "muscle_building"
    MAINTENANCE = "maintenance"
    DISEASE_MANAGEMENT = "disease_management"
    ATHLETIC_PERFORMANCE = "athletic_performance"
    GENERAL_HEALTH = "general_health"


class DietaryRestriction(Enum):
    """Common dietary restrictions"""
    VEGAN = "vegan"
    VEGETARIAN = "vegetarian"
    PESCATARIAN = "pescatarian"
    PALEO = "paleo"
    KETO = "keto"
    LOW_CARB = "low_carb"
    LOW_FAT = "low_fat"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    HALAL = "halal"
    KOSHER = "kosher"
    LOW_SODIUM = "low_sodium"
    LOW_FODMAP = "low_fodmap"
    MEDITERRANEAN = "mediterranean"


@dataclass
class UserProfile:
    """Comprehensive user profile for personalization"""
    # Basic info
    user_id: str
    age: int  # Years
    sex: str  # 'male' or 'female'
    weight: Decimal  # kg
    height: Decimal  # cm
    
    # Activity and goals
    activity_level: ActivityLevel = ActivityLevel.MODERATE
    health_goal: HealthGoal = HealthGoal.GENERAL_HEALTH
    target_weight: Optional[Decimal] = None  # kg, if applicable
    
    # Life stage (auto-calculated but can override)
    life_stage: Optional[LifeStage] = None
    is_pregnant: bool = False
    is_lactating: bool = False
    pregnancy_trimester: Optional[int] = None  # 1, 2, or 3
    
    # Health conditions
    medical_conditions: List[str] = field(default_factory=list)
    allergies: Set[str] = field(default_factory=set)
    food_intolerances: Set[str] = field(default_factory=set)
    medications: List[str] = field(default_factory=list)
    
    # Dietary preferences
    dietary_restrictions: List[DietaryRestriction] = field(default_factory=list)
    disliked_foods: Set[str] = field(default_factory=set)
    preferred_cuisines: List[str] = field(default_factory=list)
    
    # Nutrient deficiencies/concerns
    known_deficiencies: Dict[str, Decimal] = field(default_factory=dict)  # nutrient_id -> current_level
    nutrient_targets: Dict[str, Decimal] = field(default_factory=dict)  # nutrient_id -> custom_target
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class NutrientRequirement:
    """Nutrient requirements for a user"""
    nutrient_id: str
    nutrient_name: str
    rda: Decimal  # Recommended Daily Allowance
    ul: Optional[Decimal]  # Upper Limit (toxicity threshold)
    target: Decimal  # Personalized target (may differ from RDA)
    min_safe: Decimal  # Minimum safe intake
    unit: str  # mg, mcg, g, IU, etc.
    priority: int = 1  # 1=critical, 2=important, 3=beneficial
    reason: str = ""  # Why this target (e.g., "Increased for pregnancy")


@dataclass
class MealRecommendation:
    """Single meal recommendation with nutritional info"""
    meal_id: str
    meal_name: str
    foods: List[Dict]  # [{"food_id": str, "food_name": str, "portion": Decimal, "unit": str}]
    
    # Nutritional content
    total_calories: Decimal
    macronutrients: Dict[str, Decimal]  # protein, carbs, fat, fiber
    micronutrients: Dict[str, Decimal]  # all vitamins, minerals
    
    # Quality metrics
    nutrient_density_score: float  # 0-100
    diet_compliance_score: float  # 0-100 (matches user restrictions)
    health_score: float  # 0-100 (addresses user conditions)
    
    # Metadata
    meal_type: str  # breakfast, lunch, dinner, snack
    preparation_time: int  # minutes
    difficulty: str  # easy, medium, hard
    cost_estimate: str  # $, $$, $$$
    
    # Warnings/notes
    warnings: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    cooking_notes: List[str] = field(default_factory=list)


@dataclass
class DailyMealPlan:
    """Complete daily meal plan"""
    date: date
    user_id: str
    
    meals: List[MealRecommendation]
    
    # Daily totals
    total_calories: Decimal
    total_macros: Dict[str, Decimal]
    total_micros: Dict[str, Decimal]
    
    # Compliance
    rda_coverage: Dict[str, float]  # % of RDA for each nutrient
    goal_alignment: float  # 0-100
    restriction_violations: List[str] = field(default_factory=list)
    
    # Recommendations
    daily_summary: str = ""
    hydration_target: Decimal = Decimal("2.5")  # Liters
    supplements_needed: List[str] = field(default_factory=list)


class LifeStageCalculator:
    """Calculate life stage and adjust requirements accordingly"""
    
    @staticmethod
    def determine_life_stage(
        age: int,
        sex: str,
        is_pregnant: bool = False,
        is_lactating: bool = False
    ) -> LifeStage:
        """
        Determine life stage from age and sex.
        
        Args:
            age: Age in years
            sex: 'male' or 'female'
            is_pregnant: Whether currently pregnant
            is_lactating: Whether currently breastfeeding
        
        Returns:
            LifeStage enum
        """
        # Special conditions override age
        if is_pregnant:
            return LifeStage.PREGNANT
        if is_lactating:
            return LifeStage.LACTATING
        
        # Age-based stages
        if age < 1:
            if age < 0.5:  # Less than 6 months
                return LifeStage.INFANT_0_6M
            else:
                return LifeStage.INFANT_7_12M
        elif 1 <= age <= 3:
            return LifeStage.TODDLER
        elif 4 <= age <= 8:
            return LifeStage.CHILD
        elif 9 <= age <= 18:
            return LifeStage.ADOLESCENT_MALE if sex == 'male' else LifeStage.ADOLESCENT_FEMALE
        elif 19 <= age <= 50:
            return LifeStage.ADULT_MALE if sex == 'male' else LifeStage.ADULT_FEMALE
        else:  # 51+
            return LifeStage.ELDERLY_MALE if sex == 'male' else LifeStage.ELDERLY_FEMALE
    
    @staticmethod
    def get_base_calorie_needs(profile: UserProfile) -> Decimal:
        """
        Calculate Basal Metabolic Rate (BMR) using Mifflin-St Jeor equation.
        Then multiply by activity level for TDEE.
        
        Args:
            profile: User profile with weight, height, age, sex
        
        Returns:
            Total Daily Energy Expenditure (TDEE) in calories
        """
        weight_kg = float(profile.weight)
        height_cm = float(profile.height)
        age = profile.age
        
        # Mifflin-St Jeor BMR calculation
        if profile.sex == 'male':
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
        else:
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
        
        # Adjust for activity level
        tdee = bmr * profile.activity_level.value
        
        # Adjust for life stage
        if profile.life_stage == LifeStage.INFANT_0_6M:
            # Infants: ~100 kcal/kg/day
            tdee = weight_kg * 100
        elif profile.life_stage == LifeStage.INFANT_7_12M:
            # Older infants: ~90 kcal/kg/day
            tdee = weight_kg * 90
        elif profile.life_stage == LifeStage.TODDLER:
            # Toddlers need more relative to body weight
            tdee = bmr * 1.4
        elif profile.life_stage == LifeStage.PREGNANT:
            # Add 340 kcal (2nd trimester) or 452 kcal (3rd trimester)
            if profile.pregnancy_trimester == 2:
                tdee += 340
            elif profile.pregnancy_trimester == 3:
                tdee += 452
        elif profile.life_stage == LifeStage.LACTATING:
            # Add 500 kcal for milk production
            tdee += 500
        
        return Decimal(str(round(tdee)))
    
    @staticmethod
    def adjust_for_goal(base_calories: Decimal, goal: HealthGoal) -> Decimal:
        """
        Adjust calorie target based on health goal.
        
        Args:
            base_calories: Baseline TDEE
            goal: User's health goal
        
        Returns:
            Adjusted calorie target
        """
        if goal == HealthGoal.WEIGHT_LOSS:
            # 500 kcal deficit = ~0.5 kg/week loss
            return base_calories - Decimal("500")
        elif goal == HealthGoal.WEIGHT_GAIN:
            # 500 kcal surplus = ~0.5 kg/week gain
            return base_calories + Decimal("500")
        elif goal == HealthGoal.MUSCLE_BUILDING:
            # Slight surplus for muscle growth
            return base_calories + Decimal("300")
        else:
            return base_calories


class NutrientRequirementCalculator:
    """Calculate personalized nutrient requirements"""
    
    # RDA values by life stage (simplified - would be much more comprehensive)
    RDA_DATABASE = {
        LifeStage.ADULT_MALE: {
            "protein": {"rda": 56, "ul": None, "unit": "g"},
            "vitamin_d": {"rda": 15, "ul": 100, "unit": "mcg"},
            "vitamin_c": {"rda": 90, "ul": 2000, "unit": "mg"},
            "calcium": {"rda": 1000, "ul": 2500, "unit": "mg"},
            "iron": {"rda": 8, "ul": 45, "unit": "mg"},
            "vitamin_b12": {"rda": 2.4, "ul": None, "unit": "mcg"},
            "folate": {"rda": 400, "ul": 1000, "unit": "mcg"},
            "omega_3": {"rda": 1.6, "ul": None, "unit": "g"},
        },
        LifeStage.ADULT_FEMALE: {
            "protein": {"rda": 46, "ul": None, "unit": "g"},
            "vitamin_d": {"rda": 15, "ul": 100, "unit": "mcg"},
            "vitamin_c": {"rda": 75, "ul": 2000, "unit": "mg"},
            "calcium": {"rda": 1000, "ul": 2500, "unit": "mg"},
            "iron": {"rda": 18, "ul": 45, "unit": "mg"},  # Higher for menstruating women
            "vitamin_b12": {"rda": 2.4, "ul": None, "unit": "mcg"},
            "folate": {"rda": 400, "ul": 1000, "unit": "mcg"},
            "omega_3": {"rda": 1.1, "ul": None, "unit": "g"},
        },
        LifeStage.PREGNANT: {
            "protein": {"rda": 71, "ul": None, "unit": "g"},  # Increased
            "vitamin_d": {"rda": 15, "ul": 100, "unit": "mcg"},
            "vitamin_c": {"rda": 85, "ul": 2000, "unit": "mg"},
            "calcium": {"rda": 1000, "ul": 2500, "unit": "mg"},
            "iron": {"rda": 27, "ul": 45, "unit": "mg"},  # Significantly increased
            "vitamin_b12": {"rda": 2.6, "ul": None, "unit": "mcg"},
            "folate": {"rda": 600, "ul": 1000, "unit": "mcg"},  # Critical for fetal development
            "omega_3": {"rda": 1.4, "ul": None, "unit": "g"},
            "dha": {"rda": 300, "ul": None, "unit": "mg"},  # Crucial for baby brain
        },
        LifeStage.LACTATING: {
            "protein": {"rda": 71, "ul": None, "unit": "g"},
            "vitamin_d": {"rda": 15, "ul": 100, "unit": "mcg"},
            "vitamin_c": {"rda": 120, "ul": 2000, "unit": "mg"},  # Increased
            "calcium": {"rda": 1000, "ul": 2500, "unit": "mg"},
            "iron": {"rda": 9, "ul": 45, "unit": "mg"},
            "vitamin_b12": {"rda": 2.8, "ul": None, "unit": "mcg"},
            "folate": {"rda": 500, "ul": 1000, "unit": "mcg"},
            "omega_3": {"rda": 1.3, "ul": None, "unit": "g"},
        },
        LifeStage.ELDERLY_MALE: {
            "protein": {"rda": 56, "ul": None, "unit": "g"},
            "vitamin_d": {"rda": 20, "ul": 100, "unit": "mcg"},  # Increased for bone health
            "vitamin_c": {"rda": 90, "ul": 2000, "unit": "mg"},
            "calcium": {"rda": 1200, "ul": 2500, "unit": "mg"},  # Increased
            "iron": {"rda": 8, "ul": 45, "unit": "mg"},
            "vitamin_b12": {"rda": 2.4, "ul": None, "unit": "mcg"},  # Consider supplement
            "folate": {"rda": 400, "ul": 1000, "unit": "mcg"},
            "omega_3": {"rda": 1.6, "ul": None, "unit": "g"},
        },
        LifeStage.CHILD: {
            "protein": {"rda": 19, "ul": None, "unit": "g"},
            "vitamin_d": {"rda": 15, "ul": 75, "unit": "mcg"},
            "vitamin_c": {"rda": 25, "ul": 650, "unit": "mg"},
            "calcium": {"rda": 1000, "ul": 2500, "unit": "mg"},
            "iron": {"rda": 10, "ul": 40, "unit": "mg"},
            "vitamin_b12": {"rda": 1.2, "ul": None, "unit": "mcg"},
            "folate": {"rda": 200, "ul": 400, "unit": "mcg"},
            "omega_3": {"rda": 0.9, "ul": None, "unit": "g"},
        },
    }
    
    @staticmethod
    def calculate_requirements(profile: UserProfile) -> List[NutrientRequirement]:
        """
        Calculate all nutrient requirements for a user.
        
        Args:
            profile: User profile
        
        Returns:
            List of NutrientRequirement objects
        """
        requirements = []
        
        # Get base RDA for life stage
        life_stage = profile.life_stage or LifeStageCalculator.determine_life_stage(
            profile.age, profile.sex, profile.is_pregnant, profile.is_lactating
        )
        
        base_rda = NutrientRequirementCalculator.RDA_DATABASE.get(
            life_stage,
            NutrientRequirementCalculator.RDA_DATABASE[LifeStage.ADULT_MALE]
        )
        
        for nutrient_id, data in base_rda.items():
            rda = Decimal(str(data["rda"]))
            ul = Decimal(str(data["ul"])) if data["ul"] else None
            unit = data["unit"]
            
            # Start with RDA as target
            target = rda
            reason = f"Standard RDA for {life_stage.value}"
            priority = 2
            
            # Adjust for medical conditions
            target, reason, priority = NutrientRequirementCalculator._adjust_for_conditions(
                nutrient_id, target, reason, priority, profile.medical_conditions
            )
            
            # Adjust for custom targets
            if nutrient_id in profile.nutrient_targets:
                target = profile.nutrient_targets[nutrient_id]
                reason = "Custom target set by user"
            
            # Minimum safe intake (typically 50% of RDA, but varies)
            min_safe = rda * Decimal("0.5")
            
            requirements.append(NutrientRequirement(
                nutrient_id=nutrient_id,
                nutrient_name=nutrient_id.replace("_", " ").title(),
                rda=rda,
                ul=ul,
                target=target,
                min_safe=min_safe,
                unit=unit,
                priority=priority,
                reason=reason
            ))
        
        return requirements
    
    @staticmethod
    def _adjust_for_conditions(
        nutrient_id: str,
        target: Decimal,
        reason: str,
        priority: int,
        conditions: List[str]
    ) -> Tuple[Decimal, str, int]:
        """Adjust nutrient targets based on medical conditions"""
        
        for condition in conditions:
            condition_lower = condition.lower()
            
            # Diabetes
            if 'diabetes' in condition_lower:
                if nutrient_id == 'fiber':
                    target = max(target, Decimal("30"))  # Increase fiber
                    reason = "Increased for blood sugar control (diabetes)"
                    priority = 1
                elif nutrient_id == 'chromium':
                    target = max(target, Decimal("200"))  # mcg
                    reason = "Increased for insulin sensitivity (diabetes)"
                    priority = 1
            
            # Hypertension
            elif 'hypertension' in condition_lower or 'high blood pressure' in condition_lower:
                if nutrient_id == 'potassium':
                    target = max(target, Decimal("3500"))  # mg
                    reason = "Increased to lower blood pressure"
                    priority = 1
                elif nutrient_id == 'magnesium':
                    target = max(target, Decimal("400"))  # mg
                    reason = "Increased for blood pressure regulation"
                    priority = 1
            
            # Anemia
            elif 'anemia' in condition_lower:
                if nutrient_id == 'iron':
                    target = target * Decimal("1.5")  # 50% increase
                    reason = "Increased to treat anemia"
                    priority = 1
                elif nutrient_id == 'vitamin_b12':
                    target = max(target, Decimal("10"))  # mcg
                    reason = "Increased for red blood cell formation"
                    priority = 1
                elif nutrient_id == 'folate':
                    target = max(target, Decimal("800"))  # mcg
                    reason = "Increased for red blood cell formation"
                    priority = 1
            
            # Osteoporosis
            elif 'osteoporosis' in condition_lower or 'bone' in condition_lower:
                if nutrient_id == 'calcium':
                    target = max(target, Decimal("1200"))  # mg
                    reason = "Increased for bone health"
                    priority = 1
                elif nutrient_id == 'vitamin_d':
                    target = max(target, Decimal("20"))  # mcg (800 IU)
                    reason = "Increased for calcium absorption and bone health"
                    priority = 1
                elif nutrient_id == 'vitamin_k':
                    target = max(target, Decimal("120"))  # mcg
                    reason = "Increased for bone mineralization"
                    priority = 1
            
            # Autoimmune conditions
            elif any(x in condition_lower for x in ['autoimmune', 'rheumatoid', 'lupus', 'crohn', 'celiac']):
                if nutrient_id == 'omega_3':
                    target = max(target, Decimal("2.5"))  # g
                    reason = "Increased for anti-inflammatory effects"
                    priority = 1
                elif nutrient_id == 'vitamin_d':
                    target = max(target, Decimal("25"))  # mcg (1000 IU)
                    reason = "Increased for immune modulation"
                    priority = 1
        
        return target, reason, priority


# Test the system
if __name__ == "__main__":
    print("🍽️  Intelligent Meal Planner - System Test")
    print("=" * 60)
    
    # Test 1: Calculate life stage
    print("\n📊 Test 1: Life Stage Calculation")
    test_cases = [
        (0, 'female', False, False, "Infant 0-6 months"),
        (2, 'male', False, False, "Toddler"),
        (6, 'female', False, False, "Child"),
        (14, 'male', False, False, "Adolescent"),
        (30, 'female', False, False, "Adult"),
        (65, 'male', False, False, "Elderly"),
        (28, 'female', True, False, "Pregnant"),
        (32, 'female', False, True, "Lactating"),
    ]
    
    for age, sex, preg, lact, expected in test_cases:
        stage = LifeStageCalculator.determine_life_stage(age, sex, preg, lact)
        print(f"  Age {age}, {sex}, pregnant={preg}, lactating={lact}")
        print(f"  → {stage.value} ({expected})")
    
    # Test 2: Calculate calorie needs
    print("\n🔥 Test 2: Calorie Needs Calculation")
    profiles = [
        UserProfile(
            user_id="test1",
            age=30,
            sex='male',
            weight=Decimal("75"),
            height=Decimal("175"),
            activity_level=ActivityLevel.MODERATE,
            health_goal=HealthGoal.MAINTENANCE
        ),
        UserProfile(
            user_id="test2",
            age=28,
            sex='female',
            weight=Decimal("65"),
            height=Decimal("165"),
            activity_level=ActivityLevel.LIGHT,
            health_goal=HealthGoal.WEIGHT_LOSS
        ),
        UserProfile(
            user_id="test3",
            age=28,
            sex='female',
            weight=Decimal("68"),
            height=Decimal("165"),
            is_pregnant=True,
            pregnancy_trimester=3,
            activity_level=ActivityLevel.LIGHT,
            health_goal=HealthGoal.MAINTENANCE
        ),
    ]
    
    for profile in profiles:
        profile.life_stage = LifeStageCalculator.determine_life_stage(
            profile.age, profile.sex, profile.is_pregnant, profile.is_lactating
        )
        base_cal = LifeStageCalculator.get_base_calorie_needs(profile)
        target_cal = LifeStageCalculator.adjust_for_goal(base_cal, profile.health_goal)
        
        print(f"\n  {profile.sex}, age {profile.age}, {profile.weight}kg, {profile.height}cm")
        print(f"  Life stage: {profile.life_stage.value}")
        print(f"  Activity: {profile.activity_level.name}")
        print(f"  Goal: {profile.health_goal.value}")
        print(f"  Base TDEE: {base_cal} kcal")
        print(f"  Target: {target_cal} kcal")
    
    # Test 3: Calculate nutrient requirements
    print("\n💊 Test 3: Nutrient Requirements")
    test_profile = UserProfile(
        user_id="test4",
        age=30,
        sex='female',
        weight=Decimal("65"),
        height=Decimal("165"),
        medical_conditions=["type 2 diabetes", "anemia"]
    )
    test_profile.life_stage = LifeStageCalculator.determine_life_stage(
        test_profile.age, test_profile.sex
    )
    
    requirements = NutrientRequirementCalculator.calculate_requirements(test_profile)
    
    print(f"\n  Female, 30 years, with diabetes and anemia")
    print(f"  Life stage: {test_profile.life_stage.value}\n")
    
    print(f"  {'Nutrient':<15} {'RDA':<8} {'Target':<8} {'Unit':<6} {'Priority':<8} {'Reason'}")
    print(f"  {'-'*90}")
    
    for req in sorted(requirements, key=lambda x: x.priority):
        priority_str = "⚠️ CRITICAL" if req.priority == 1 else "✓ Important" if req.priority == 2 else "○ Beneficial"
        print(f"  {req.nutrient_name:<15} {req.rda:<8} {req.target:<8} {req.unit:<6} {priority_str:<12} {req.reason[:40]}")
    
    print("\n✅ System tests complete!")
    print("=" * 60)
