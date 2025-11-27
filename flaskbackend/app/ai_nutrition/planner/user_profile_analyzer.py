"""
Intelligent Meal Planner - User Profile Analysis
Analyzes user demographics, activity, health conditions, and goals
to create personalized nutrition plans.

Part of Phase 3: Intelligent Meal Planning System
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from decimal import Decimal
from datetime import datetime, date
import math


class BiologicalSex(Enum):
    """Biological sex for metabolism calculations"""
    MALE = "male"
    FEMALE = "female"


class ActivityLevel(Enum):
    """Physical activity levels with multipliers"""
    SEDENTARY = ("sedentary", 1.2, "Little to no exercise, desk job")
    LIGHT = ("light", 1.375, "Light exercise 1-3 days/week")
    MODERATE = ("moderate", 1.55, "Moderate exercise 3-5 days/week")
    ACTIVE = ("active", 1.725, "Hard exercise 6-7 days/week")
    VERY_ACTIVE = ("very_active", 1.9, "Physical job or athlete, training 2x/day")
    
    def __init__(self, key: str, multiplier: float, description: str):
        self.key = key
        self.multiplier = multiplier
        self.description = description


class LifeStage(Enum):
    """Life stages with specific nutritional needs"""
    INFANT_0_6_MONTHS = ("infant_0_6", 0, 0.5, "Exclusive breastfeeding/formula")
    INFANT_7_12_MONTHS = ("infant_7_12", 0.5, 1, "Complementary foods introduction")
    TODDLER_1_3_YEARS = ("toddler", 1, 3, "Rapid growth, high energy needs")
    CHILD_4_8_YEARS = ("child", 4, 8, "Steady growth, developing habits")
    ADOLESCENT_9_13_YEARS = ("adolescent_early", 9, 13, "Puberty onset, rapid changes")
    ADOLESCENT_14_18_YEARS = ("adolescent_late", 14, 18, "Peak growth, high calorie needs")
    ADULT_19_30_YEARS = ("adult_young", 19, 30, "Peak physical condition")
    ADULT_31_50_YEARS = ("adult_middle", 31, 50, "Maintenance, metabolism slowing")
    ADULT_51_70_YEARS = ("adult_senior", 51, 70, "Reduced calorie needs, bone health focus")
    ELDERLY_71_PLUS = ("elderly", 71, 120, "Lower calorie, higher nutrient density")
    PREGNANT = ("pregnant", 18, 45, "Increased folate, iron, calcium needs")
    LACTATING = ("lactating", 18, 45, "500+ extra calories, high fluids")
    
    def __init__(self, key: str, min_age: float, max_age: float, description: str):
        self.key = key
        self.min_age = min_age
        self.max_age = max_age
        self.description = description


class HealthGoal(Enum):
    """Health and fitness goals"""
    WEIGHT_LOSS = ("weight_loss", "Lose weight safely (0.5-1kg/week)")
    WEIGHT_GAIN = ("weight_gain", "Gain weight (0.25-0.5kg/week)")
    MUSCLE_BUILDING = ("muscle_building", "Build muscle mass (high protein)")
    MAINTENANCE = ("maintenance", "Maintain current weight")
    ATHLETIC_PERFORMANCE = ("athletic_performance", "Optimize for sports performance")
    DISEASE_MANAGEMENT = ("disease_management", "Manage chronic condition")
    GENERAL_HEALTH = ("general_health", "Overall wellness and prevention")
    LONGEVITY = ("longevity", "Anti-aging, disease prevention")
    
    def __init__(self, key: str, description: str):
        self.key = key
        self.description = description


@dataclass
class UserProfile:
    """Complete user profile for personalized nutrition"""
    # Demographics
    user_id: str
    age: float  # Years (can be fractional for infants)
    sex: BiologicalSex
    
    # Physical measurements
    weight_kg: Decimal  # Current weight in kg
    height_cm: Decimal  # Height in cm
    
    # Activity and lifestyle
    activity_level: ActivityLevel
    
    # Goals
    health_goal: HealthGoal
    target_weight_kg: Optional[Decimal] = None  # Goal weight if applicable
    
    # Health conditions (disease IDs from our autoimmune database)
    medical_conditions: Set[str] = field(default_factory=set)
    
    # Dietary restrictions
    allergens: Set[str] = field(default_factory=set)  # Known allergens
    intolerances: Set[str] = field(default_factory=set)  # Intolerances (lactose, gluten, etc.)
    dietary_preference: Optional[str] = None  # vegan, vegetarian, pescatarian, etc.
    
    # Special states
    is_pregnant: bool = False
    is_lactating: bool = False
    pregnancy_trimester: Optional[int] = None  # 1, 2, or 3
    
    # Medications (can interact with nutrients)
    medications: Set[str] = field(default_factory=set)
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class UserProfileAnalyzer:
    """
    Analyzes user profile to determine nutritional needs,
    life stage requirements, and personalized recommendations.
    """
    
    def __init__(self):
        """Initialize analyzer with reference data"""
        self.bmr_cache: Dict[str, Decimal] = {}
        self.tdee_cache: Dict[str, Decimal] = {}
    
    def determine_life_stage(self, profile: UserProfile) -> LifeStage:
        """
        Determine life stage based on age and special conditions.
        
        Args:
            profile: User profile
        
        Returns:
            LifeStage enum
        """
        # Special states override age
        if profile.is_pregnant:
            return LifeStage.PREGNANT
        
        if profile.is_lactating:
            return LifeStage.LACTATING
        
        # Age-based life stage
        age = profile.age
        
        for stage in LifeStage:
            if stage == LifeStage.PREGNANT or stage == LifeStage.LACTATING:
                continue
            
            if stage.min_age <= age < stage.max_age:
                return stage
        
        # Default to elderly if age is very high
        return LifeStage.ELDERLY_71_PLUS
    
    def calculate_bmi(self, profile: UserProfile) -> Decimal:
        """
        Calculate Body Mass Index.
        BMI = weight(kg) / (height(m))^2
        
        Args:
            profile: User profile
        
        Returns:
            BMI value
        """
        height_m = profile.height_cm / Decimal("100")
        bmi = profile.weight_kg / (height_m * height_m)
        return bmi.quantize(Decimal("0.1"))
    
    def interpret_bmi(self, bmi: Decimal, age: float) -> Tuple[str, str]:
        """
        Interpret BMI based on age-appropriate guidelines.
        
        Args:
            bmi: BMI value
            age: Age in years
        
        Returns:
            Tuple of (category, interpretation)
        """
        # For adults (18+)
        if age >= 18:
            if bmi < Decimal("18.5"):
                return ("underweight", "Below healthy weight range")
            elif bmi < Decimal("25"):
                return ("normal", "Healthy weight range")
            elif bmi < Decimal("30"):
                return ("overweight", "Above healthy weight range")
            elif bmi < Decimal("35"):
                return ("obese_class_1", "Obesity Class I - health risks")
            elif bmi < Decimal("40"):
                return ("obese_class_2", "Obesity Class II - significant health risks")
            else:
                return ("obese_class_3", "Obesity Class III - severe health risks")
        
        # For children/adolescents, BMI percentiles would be used
        # (requires CDC growth charts - simplified here)
        else:
            if bmi < Decimal("15"):
                return ("underweight", "May need nutritional support")
            elif bmi < Decimal("25"):
                return ("normal", "Healthy for age")
            else:
                return ("overweight", "May need dietary guidance")
    
    def calculate_bmr(self, profile: UserProfile) -> Decimal:
        """
        Calculate Basal Metabolic Rate using Mifflin-St Jeor equation.
        Most accurate for modern populations.
        
        Men: BMR = 10 × weight(kg) + 6.25 × height(cm) - 5 × age(years) + 5
        Women: BMR = 10 × weight(kg) + 6.25 × height(cm) - 5 × age(years) - 161
        
        Args:
            profile: User profile
        
        Returns:
            BMR in kcal/day
        """
        # Check cache
        cache_key = f"{profile.user_id}_{profile.weight_kg}_{profile.age}"
        if cache_key in self.bmr_cache:
            return self.bmr_cache[cache_key]
        
        weight = float(profile.weight_kg)
        height = float(profile.height_cm)
        age = profile.age
        
        # Base calculation
        bmr = 10 * weight + 6.25 * height - 5 * age
        
        # Sex adjustment
        if profile.sex == BiologicalSex.MALE:
            bmr += 5
        else:
            bmr -= 161
        
        bmr_decimal = Decimal(str(bmr)).quantize(Decimal("0.1"))
        
        # Cache result
        self.bmr_cache[cache_key] = bmr_decimal
        
        return bmr_decimal
    
    def calculate_tdee(self, profile: UserProfile) -> Decimal:
        """
        Calculate Total Daily Energy Expenditure.
        TDEE = BMR × Activity Multiplier
        
        Args:
            profile: User profile
        
        Returns:
            TDEE in kcal/day
        """
        # Check cache
        cache_key = f"{profile.user_id}_{profile.weight_kg}_{profile.age}_{profile.activity_level.key}"
        if cache_key in self.tdee_cache:
            return self.tdee_cache[cache_key]
        
        bmr = self.calculate_bmr(profile)
        multiplier = Decimal(str(profile.activity_level.multiplier))
        
        tdee = bmr * multiplier
        
        # Adjustments for special states
        if profile.is_pregnant:
            # Add 340 kcal (2nd trimester) or 452 kcal (3rd trimester)
            if profile.pregnancy_trimester == 2:
                tdee += Decimal("340")
            elif profile.pregnancy_trimester == 3:
                tdee += Decimal("452")
        
        if profile.is_lactating:
            # Add 500 kcal for breastfeeding
            tdee += Decimal("500")
        
        tdee = tdee.quantize(Decimal("1"))
        
        # Cache result
        self.tdee_cache[cache_key] = tdee
        
        return tdee
    
    def calculate_calorie_target(self, profile: UserProfile) -> Decimal:
        """
        Calculate daily calorie target based on goal.
        
        Args:
            profile: User profile
        
        Returns:
            Target calories per day
        """
        tdee = self.calculate_tdee(profile)
        
        if profile.health_goal == HealthGoal.WEIGHT_LOSS:
            # 500 kcal deficit = ~0.5kg/week loss
            return tdee - Decimal("500")
        
        elif profile.health_goal == HealthGoal.WEIGHT_GAIN:
            # 250-500 kcal surplus = ~0.25-0.5kg/week gain
            return tdee + Decimal("350")
        
        elif profile.health_goal == HealthGoal.MUSCLE_BUILDING:
            # Slight surplus for muscle growth
            return tdee + Decimal("250")
        
        elif profile.health_goal == HealthGoal.ATHLETIC_PERFORMANCE:
            # Maintain or slight surplus
            return tdee + Decimal("100")
        
        else:  # MAINTENANCE, DISEASE_MANAGEMENT, GENERAL_HEALTH, LONGEVITY
            return tdee
    
    def calculate_macronutrient_targets(
        self, 
        profile: UserProfile,
        calorie_target: Decimal
    ) -> Dict[str, Decimal]:
        """
        Calculate macronutrient targets (protein, carbs, fat).
        
        Args:
            profile: User profile
            calorie_target: Daily calorie target
        
        Returns:
            Dictionary with protein, carbs, fat in grams
        """
        life_stage = self.determine_life_stage(profile)
        
        # Protein targets (g/kg body weight)
        if profile.health_goal == HealthGoal.MUSCLE_BUILDING:
            protein_per_kg = Decimal("2.0")  # High protein for muscle growth
        elif profile.health_goal == HealthGoal.ATHLETIC_PERFORMANCE:
            protein_per_kg = Decimal("1.6")
        elif profile.health_goal == HealthGoal.WEIGHT_LOSS:
            protein_per_kg = Decimal("1.8")  # Preserve muscle during deficit
        elif profile.is_pregnant or profile.is_lactating:
            protein_per_kg = Decimal("1.2")
        elif life_stage in [LifeStage.ELDERLY_71_PLUS, LifeStage.ADULT_51_70_YEARS]:
            protein_per_kg = Decimal("1.2")  # Higher for elderly to prevent sarcopenia
        elif life_stage in [LifeStage.ADOLESCENT_14_18_YEARS, LifeStage.ADOLESCENT_9_13_YEARS]:
            protein_per_kg = Decimal("1.0")  # Growing adolescents
        else:
            protein_per_kg = Decimal("0.8")  # Standard RDA
        
        # Calculate protein
        protein_g = (profile.weight_kg * protein_per_kg).quantize(Decimal("0.1"))
        protein_kcal = protein_g * Decimal("4")  # 4 kcal per gram
        
        # Fat targets (20-35% of calories, adjust by goal)
        if profile.health_goal == HealthGoal.MUSCLE_BUILDING:
            fat_percent = Decimal("0.25")  # Lower fat for muscle building
        elif life_stage in [LifeStage.INFANT_0_6_MONTHS, LifeStage.INFANT_7_12_MONTHS]:
            fat_percent = Decimal("0.45")  # Infants need high fat for brain development
        elif life_stage == LifeStage.TODDLER_1_3_YEARS:
            fat_percent = Decimal("0.35")
        else:
            fat_percent = Decimal("0.30")  # Standard
        
        fat_kcal = calorie_target * fat_percent
        fat_g = (fat_kcal / Decimal("9")).quantize(Decimal("0.1"))  # 9 kcal per gram
        
        # Carbs fill the rest
        carbs_kcal = calorie_target - protein_kcal - fat_kcal
        carbs_g = (carbs_kcal / Decimal("4")).quantize(Decimal("0.1"))  # 4 kcal per gram
        
        return {
            "protein_g": protein_g,
            "carbohydrates_g": carbs_g,
            "fat_g": fat_g,
            "calories": calorie_target
        }
    
    def get_life_stage_specific_needs(self, profile: UserProfile) -> Dict[str, str]:
        """
        Get specific nutritional needs and considerations for life stage.
        
        Args:
            profile: User profile
        
        Returns:
            Dictionary of nutrient needs and recommendations
        """
        life_stage = self.determine_life_stage(profile)
        
        needs = {
            "life_stage": life_stage.key,
            "description": life_stage.description,
            "key_nutrients": [],
            "recommendations": [],
            "precautions": []
        }
        
        if life_stage == LifeStage.INFANT_0_6_MONTHS:
            needs["key_nutrients"] = ["Exclusive breast milk or formula", "Vitamin D supplement (400 IU)"]
            needs["recommendations"] = [
                "No solid foods before 6 months",
                "No water needed if exclusively breastfed",
                "Iron-fortified formula if not breastfeeding"
            ]
            needs["precautions"] = [
                "No honey (botulism risk)",
                "No cow's milk before 12 months",
                "Monitor for allergic reactions"
            ]
        
        elif life_stage == LifeStage.INFANT_7_12_MONTHS:
            needs["key_nutrients"] = ["Iron", "Zinc", "Vitamin D", "DHA (omega-3)"]
            needs["recommendations"] = [
                "Introduce complementary foods gradually",
                "Iron-rich foods: meat, beans, fortified cereals",
                "Continue breast milk or formula",
                "Introduce common allergens early (peanuts, eggs, fish)"
            ]
            needs["precautions"] = [
                "No honey until 12 months",
                "Avoid choking hazards (whole grapes, nuts)",
                "No cow's milk as main drink",
                "Watch for signs of allergies"
            ]
        
        elif life_stage == LifeStage.TODDLER_1_3_YEARS:
            needs["key_nutrients"] = ["Calcium", "Iron", "Vitamin D", "DHA"]
            needs["recommendations"] = [
                "Whole milk (full-fat) until age 2",
                "Calcium: 700mg/day (2-3 dairy servings)",
                "Iron: 7mg/day (lean meats, beans)",
                "Small, frequent meals (picky eating phase)"
            ]
            needs["precautions"] = [
                "Avoid choking hazards",
                "Limit juice to 4oz/day",
                "No added sugar",
                "Monitor for constipation (add fiber gradually)"
            ]
        
        elif life_stage in [LifeStage.CHILD_4_8_YEARS, LifeStage.ADOLESCENT_9_13_YEARS]:
            needs["key_nutrients"] = ["Calcium", "Vitamin D", "Iron", "Fiber"]
            needs["recommendations"] = [
                "Calcium: 1000-1300mg/day (bone growth)",
                "Vitamin D: 600 IU/day (sun exposure + diet)",
                "Limit processed foods and sugar",
                "Establish healthy eating patterns"
            ]
            needs["precautions"] = [
                "Watch for disordered eating signs",
                "Limit screen time during meals",
                "Avoid energy drinks",
                "Monitor growth patterns"
            ]
        
        elif life_stage == LifeStage.ADOLESCENT_14_18_YEARS:
            needs["key_nutrients"] = ["Iron (girls)", "Calcium", "Vitamin D", "Protein"]
            needs["recommendations"] = [
                "Iron: 15mg/day girls, 11mg/day boys (menstruation, growth)",
                "Calcium: 1300mg/day (peak bone mass building)",
                "Protein: 0.85-1.0g/kg (muscle development)",
                "Zinc: important for growth and development"
            ]
            needs["precautions"] = [
                "Screen for eating disorders (especially girls)",
                "Avoid energy drinks and excessive caffeine",
                "Monitor for anemia (especially athletes)",
                "Discourage crash diets"
            ]
        
        elif life_stage == LifeStage.PREGNANT:
            needs["key_nutrients"] = ["Folate/Folic Acid", "Iron", "Calcium", "DHA", "Choline"]
            needs["recommendations"] = [
                "Folate: 600mcg/day (neural tube development)",
                "Iron: 27mg/day (increased blood volume)",
                "Calcium: 1000mg/day (fetal bone development)",
                "DHA: 200-300mg/day (brain development)",
                "Choline: 450mg/day (brain and nervous system)",
                "Extra 340-450 kcal/day (2nd-3rd trimester)"
            ]
            needs["precautions"] = [
                "AVOID: Raw fish, unpasteurized cheese, deli meats, high-mercury fish",
                "AVOID: Alcohol, excess vitamin A (>10,000 IU), raw eggs",
                "Limit caffeine to 200mg/day",
                "Cook meat thoroughly (toxoplasmosis risk)",
                "Avoid liver (too much vitamin A)"
            ]
        
        elif life_stage == LifeStage.LACTATING:
            needs["key_nutrients"] = ["Calcium", "Vitamin D", "DHA", "Iodine", "Fluids"]
            needs["recommendations"] = [
                "Extra 500 kcal/day for milk production",
                "Calcium: 1000mg/day",
                "DHA: 200-300mg/day (passes to baby)",
                "Iodine: 290mcg/day (thyroid function)",
                "Hydration: 3+ liters/day",
                "Continue prenatal vitamins"
            ]
            needs["precautions"] = [
                "Limit caffeine (passes to baby)",
                "Avoid alcohol or wait 2-3 hours after drinking",
                "Monitor for food sensitivities affecting baby",
                "Avoid high-mercury fish",
                "Watch for allergenic foods affecting baby"
            ]
        
        elif life_stage in [LifeStage.ADULT_51_70_YEARS, LifeStage.ELDERLY_71_PLUS]:
            needs["key_nutrients"] = ["Vitamin B12", "Calcium", "Vitamin D", "Protein", "Fiber"]
            needs["recommendations"] = [
                "Vitamin B12: 2.4mcg/day (absorption decreases with age)",
                "Calcium: 1000-1200mg/day (bone health)",
                "Vitamin D: 800-1000 IU/day (synthesis decreases)",
                "Protein: 1.0-1.2g/kg (prevent sarcopenia)",
                "Fiber: 25-30g/day (digestive health)",
                "Omega-3: for heart and brain health"
            ]
            needs["precautions"] = [
                "Monitor sodium (blood pressure)",
                "Watch for medication-nutrient interactions",
                "Ensure adequate hydration (thirst sense decreases)",
                "Screen for malnutrition risk",
                "Check for swallowing difficulties"
            ]
        
        else:  # Young/middle adults
            needs["key_nutrients"] = ["Fiber", "Omega-3", "Vitamin D", "Magnesium"]
            needs["recommendations"] = [
                "Focus on whole foods and variety",
                "Fiber: 25-38g/day (heart health, digestion)",
                "Omega-3: 250-500mg EPA+DHA/day (heart, brain)",
                "Limit processed foods, added sugar, sodium",
                "Maintain healthy weight"
            ]
            needs["precautions"] = [
                "Limit alcohol consumption",
                "Avoid trans fats",
                "Monitor portion sizes",
                "Stay hydrated"
            ]
        
        return needs
    
    def analyze_profile(self, profile: UserProfile) -> Dict:
        """
        Complete analysis of user profile with all recommendations.
        
        Args:
            profile: User profile
        
        Returns:
            Comprehensive analysis dictionary
        """
        life_stage = self.determine_life_stage(profile)
        bmi = self.calculate_bmi(profile)
        bmi_category, bmi_interpretation = self.interpret_bmi(bmi, profile.age)
        bmr = self.calculate_bmr(profile)
        tdee = self.calculate_tdee(profile)
        calorie_target = self.calculate_calorie_target(profile)
        macros = self.calculate_macronutrient_targets(profile, calorie_target)
        life_stage_needs = self.get_life_stage_specific_needs(profile)
        
        return {
            "user_id": profile.user_id,
            "life_stage": {
                "stage": life_stage.key,
                "description": life_stage.description,
                "age": profile.age
            },
            "body_composition": {
                "weight_kg": float(profile.weight_kg),
                "height_cm": float(profile.height_cm),
                "bmi": float(bmi),
                "bmi_category": bmi_category,
                "bmi_interpretation": bmi_interpretation
            },
            "energy_needs": {
                "bmr_kcal": float(bmr),
                "tdee_kcal": float(tdee),
                "target_kcal": float(calorie_target),
                "activity_level": profile.activity_level.description
            },
            "macronutrient_targets": {
                "protein_g": float(macros["protein_g"]),
                "carbohydrates_g": float(macros["carbohydrates_g"]),
                "fat_g": float(macros["fat_g"]),
                "protein_percent": round(float(macros["protein_g"]) * 4 / float(calorie_target) * 100, 1),
                "carbs_percent": round(float(macros["carbohydrates_g"]) * 4 / float(calorie_target) * 100, 1),
                "fat_percent": round(float(macros["fat_g"]) * 9 / float(calorie_target) * 100, 1)
            },
            "life_stage_needs": life_stage_needs,
            "health_goal": {
                "goal": profile.health_goal.key,
                "description": profile.health_goal.description
            },
            "restrictions": {
                "allergens": list(profile.allergens),
                "intolerances": list(profile.intolerances),
                "dietary_preference": profile.dietary_preference,
                "medical_conditions": list(profile.medical_conditions)
            },
            "special_states": {
                "is_pregnant": profile.is_pregnant,
                "pregnancy_trimester": profile.pregnancy_trimester,
                "is_lactating": profile.is_lactating
            }
        }


def test_user_profile_analyzer():
    """Test the UserProfileAnalyzer with various scenarios"""
    print("🧪 Testing User Profile Analyzer")
    print("=" * 60)
    
    analyzer = UserProfileAnalyzer()
    
    # Test Case 1: Pregnant woman
    print("\n📋 Test Case 1: Pregnant Woman (2nd Trimester)")
    print("-" * 60)
    
    pregnant_profile = UserProfile(
        user_id="user_001",
        age=28,
        sex=BiologicalSex.FEMALE,
        weight_kg=Decimal("68"),
        height_cm=Decimal("165"),
        activity_level=ActivityLevel.LIGHT,
        health_goal=HealthGoal.GENERAL_HEALTH,
        is_pregnant=True,
        pregnancy_trimester=2,
        allergens={"shellfish"},
        dietary_preference="pescatarian"
    )
    
    analysis = analyzer.analyze_profile(pregnant_profile)
    
    print(f"Age: {analysis['life_stage']['age']} years")
    print(f"Life Stage: {analysis['life_stage']['description']}")
    print(f"BMI: {analysis['body_composition']['bmi']} ({analysis['body_composition']['bmi_category']})")
    print(f"\nEnergy Needs:")
    print(f"  BMR: {analysis['energy_needs']['bmr_kcal']} kcal/day")
    print(f"  TDEE: {analysis['energy_needs']['tdee_kcal']} kcal/day")
    print(f"  Target: {analysis['energy_needs']['target_kcal']} kcal/day")
    print(f"\nMacronutrient Targets:")
    print(f"  Protein: {analysis['macronutrient_targets']['protein_g']}g ({analysis['macronutrient_targets']['protein_percent']}%)")
    print(f"  Carbs: {analysis['macronutrient_targets']['carbohydrates_g']}g ({analysis['macronutrient_targets']['carbs_percent']}%)")
    print(f"  Fat: {analysis['macronutrient_targets']['fat_g']}g ({analysis['macronutrient_targets']['fat_percent']}%)")
    print(f"\nKey Nutrients: {', '.join(analysis['life_stage_needs']['key_nutrients'][:3])}")
    print(f"Precautions: {analysis['life_stage_needs']['precautions'][0]}")
    
    # Test Case 2: Elderly man
    print("\n\n📋 Test Case 2: Elderly Man (75 years)")
    print("-" * 60)
    
    elderly_profile = UserProfile(
        user_id="user_002",
        age=75,
        sex=BiologicalSex.MALE,
        weight_kg=Decimal("78"),
        height_cm=Decimal("175"),
        activity_level=ActivityLevel.LIGHT,
        health_goal=HealthGoal.LONGEVITY,
        medical_conditions={"type_2_diabetes", "hypertension"}
    )
    
    analysis = analyzer.analyze_profile(elderly_profile)
    
    print(f"Age: {analysis['life_stage']['age']} years")
    print(f"Life Stage: {analysis['life_stage']['description']}")
    print(f"BMI: {analysis['body_composition']['bmi']} ({analysis['body_composition']['bmi_category']})")
    print(f"\nEnergy Needs:")
    print(f"  Target: {analysis['energy_needs']['target_kcal']} kcal/day")
    print(f"\nMacronutrient Targets:")
    print(f"  Protein: {analysis['macronutrient_targets']['protein_g']}g (higher to prevent sarcopenia)")
    print(f"\nKey Nutrients: {', '.join(analysis['life_stage_needs']['key_nutrients'][:3])}")
    print(f"Medical Conditions: {', '.join(analysis['restrictions']['medical_conditions'])}")
    
    # Test Case 3: Toddler
    print("\n\n📋 Test Case 3: Toddler (2 years old)")
    print("-" * 60)
    
    toddler_profile = UserProfile(
        user_id="user_003",
        age=2,
        sex=BiologicalSex.FEMALE,
        weight_kg=Decimal("12.5"),
        height_cm=Decimal("87"),
        activity_level=ActivityLevel.MODERATE,  # Active toddler
        health_goal=HealthGoal.GENERAL_HEALTH
    )
    
    analysis = analyzer.analyze_profile(toddler_profile)
    
    print(f"Age: {analysis['life_stage']['age']} years")
    print(f"Life Stage: {analysis['life_stage']['description']}")
    print(f"\nEnergy Needs:")
    print(f"  Target: {analysis['energy_needs']['target_kcal']} kcal/day")
    print(f"\nMacronutrient Targets:")
    print(f"  Protein: {analysis['macronutrient_targets']['protein_g']}g")
    print(f"  Fat: {analysis['macronutrient_targets']['fat_g']}g ({analysis['macronutrient_targets']['fat_percent']}% - higher for brain development)")
    print(f"\nKey Recommendations:")
    for rec in analysis['life_stage_needs']['recommendations'][:3]:
        print(f"  • {rec}")
    print(f"\nPrecautions:")
    for prec in analysis['life_stage_needs']['precautions'][:2]:
        print(f"  ⚠️ {prec}")
    
    # Test Case 4: Athlete
    print("\n\n📋 Test Case 4: Athlete (Muscle Building)")
    print("-" * 60)
    
    athlete_profile = UserProfile(
        user_id="user_004",
        age=24,
        sex=BiologicalSex.MALE,
        weight_kg=Decimal("82"),
        height_cm=Decimal("180"),
        activity_level=ActivityLevel.VERY_ACTIVE,
        health_goal=HealthGoal.MUSCLE_BUILDING,
        target_weight_kg=Decimal("88")
    )
    
    analysis = analyzer.analyze_profile(athlete_profile)
    
    print(f"Age: {analysis['life_stage']['age']} years")
    print(f"Current Weight: {analysis['body_composition']['weight_kg']}kg")
    print(f"BMI: {analysis['body_composition']['bmi']}")
    print(f"\nEnergy Needs:")
    print(f"  BMR: {analysis['energy_needs']['bmr_kcal']} kcal/day")
    print(f"  TDEE: {analysis['energy_needs']['tdee_kcal']} kcal/day (very active)")
    print(f"  Target: {analysis['energy_needs']['target_kcal']} kcal/day (surplus for muscle gain)")
    print(f"\nMacronutrient Targets:")
    print(f"  Protein: {analysis['macronutrient_targets']['protein_g']}g (high for muscle building)")
    print(f"  Carbs: {analysis['macronutrient_targets']['carbohydrates_g']}g (fuel for training)")
    print(f"  Fat: {analysis['macronutrient_targets']['fat_g']}g")
    
    print("\n" + "=" * 60)
    print("✅ User Profile Analyzer tests complete!")
    print("\n💡 This analyzer can handle:")
    print("   • All life stages (infants to elderly)")
    print("   • Pregnant and lactating women")
    print("   • Various health goals")
    print("   • Medical conditions and restrictions")
    print("   • Personalized calorie and macro targets")


if __name__ == "__main__":
    test_user_profile_analyzer()
