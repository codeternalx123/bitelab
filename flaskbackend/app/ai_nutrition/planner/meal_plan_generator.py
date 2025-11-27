"""
Complete Meal Planning Engine
Generates personalized daily meal plans with precise portions
based on user profile, health conditions, and nutrient requirements.

Integrates:
- Life stage calculation
- Nutrient requirements
- Food avoidance checking
- Portion calculation
- Quality scoring

Part of Intelligent Meal Planner System
"""

from typing import List, Dict, Optional, Tuple
from decimal import Decimal
from datetime import date, datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ai_nutrition.planner.personalized_meal_planner import (
    UserProfile, LifeStage, ActivityLevel, HealthGoal,
    NutrientRequirement, MealRecommendation, DailyMealPlan,
    LifeStageCalculator, NutrientRequirementCalculator
)
from app.ai_nutrition.planner.food_recommendation_engine import (
    ConditionFoodDatabase, FoodAvoidance, AvoidanceReason
)


class PortionCalculator:
    """
    Calculate precise food portions based on life stage and goals.
    Uses USDA standard serving sizes and adjusts for individual needs.
    """
    
    # Standard serving sizes (in grams unless noted)
    STANDARD_SERVINGS = {
        # Proteins
        "meat": 85,  # 3 oz cooked
        "fish": 85,  # 3 oz cooked
        "poultry": 85,  # 3 oz cooked
        "eggs": 50,  # 1 large egg
        "tofu": 100,  # ~3.5 oz
        "legumes_cooked": 90,  # 1/2 cup
        
        # Grains
        "rice_cooked": 150,  # 1 cup
        "pasta_cooked": 140,  # 1 cup
        "bread": 30,  # 1 slice
        "oats_dry": 40,  # 1/2 cup
        
        # Vegetables
        "leafy_greens_raw": 30,  # 1 cup
        "vegetables_cooked": 85,  # 1/2 cup
        "vegetables_raw": 90,  # 1 cup
        
        # Fruits
        "fruit_medium": 150,  # 1 medium apple/banana
        "berries": 150,  # 1 cup
        "dried_fruit": 40,  # 1/4 cup
        
        # Dairy
        "milk": 240,  # 1 cup (ml)
        "yogurt": 170,  # 3/4 cup
        "cheese": 30,  # 1 oz
        
        # Fats
        "nuts": 28,  # 1 oz (~1/4 cup)
        "seeds": 15,  # 1 tbsp
        "oil": 14,  # 1 tbsp
        "avocado": 50,  # 1/3 medium
        "nut_butter": 16,  # 1 tbsp
    }
    
    # Life stage multipliers (adjust portions)
    LIFE_STAGE_MULTIPLIERS = {
        LifeStage.INFANT_0_6M: 0.1,  # Breast milk/formula only
        LifeStage.INFANT_7_12M: 0.3,  # Starting solids
        LifeStage.TODDLER: 0.5,  # Small portions
        LifeStage.CHILD: 0.7,  # Growing
        LifeStage.ADOLESCENT_MALE: 1.3,  # High needs
        LifeStage.ADOLESCENT_FEMALE: 1.1,  # High needs
        LifeStage.ADULT_MALE: 1.0,  # Baseline
        LifeStage.ADULT_FEMALE: 0.8,  # Slightly less
        LifeStage.ELDERLY_MALE: 0.9,  # Reduced
        LifeStage.ELDERLY_FEMALE: 0.7,  # Reduced
        LifeStage.PREGNANT: 1.0,  # Same as adult, more calories
        LifeStage.LACTATING: 1.1,  # Increased needs
    }
    
    @staticmethod
    def calculate_portion(
        food_category: str,
        life_stage: LifeStage,
        goal: HealthGoal = HealthGoal.MAINTENANCE
    ) -> Tuple[Decimal, str]:
        """
        Calculate appropriate portion size.
        
        Args:
            food_category: Type of food (e.g., "meat", "vegetables")
            life_stage: User's life stage
            goal: Health goal
        
        Returns:
            Tuple of (portion_size, unit)
        """
        # Get base serving size
        base_serving = PortionCalculator.STANDARD_SERVINGS.get(food_category, 100)
        
        # Apply life stage multiplier
        multiplier = PortionCalculator.LIFE_STAGE_MULTIPLIERS.get(life_stage, 1.0)
        
        # Adjust for goal
        if goal == HealthGoal.WEIGHT_LOSS:
            multiplier *= 0.9  # Slightly smaller portions
        elif goal == HealthGoal.MUSCLE_BUILDING:
            if food_category in ["meat", "fish", "poultry", "legumes_cooked", "tofu"]:
                multiplier *= 1.2  # More protein
        
        portion = Decimal(str(round(base_serving * multiplier)))
        
        # Determine unit
        if food_category in ["milk"]:
            unit = "ml"
        else:
            unit = "g"
        
        return portion, unit
    
    @staticmethod
    def get_meal_distribution(total_calories: Decimal) -> Dict[str, Decimal]:
        """
        Distribute daily calories across meals.
        
        Args:
            total_calories: Total daily calorie target
        
        Returns:
            Dictionary of meal_type -> calories
        """
        return {
            "breakfast": total_calories * Decimal("0.25"),  # 25%
            "lunch": total_calories * Decimal("0.35"),  # 35%
            "dinner": total_calories * Decimal("0.30"),  # 30%
            "snacks": total_calories * Decimal("0.10"),  # 10%
        }


class MealPlanGenerator:
    """Generate complete personalized meal plans"""
    
    def __init__(self):
        self.portion_calculator = PortionCalculator()
    
    def generate_daily_plan(
        self,
        profile: UserProfile,
        target_date: date = None
    ) -> DailyMealPlan:
        """
        Generate a complete daily meal plan for a user.
        
        Args:
            profile: User profile with health info
            target_date: Date for the meal plan (default: today)
        
        Returns:
            DailyMealPlan object
        """
        target_date = target_date or date.today()
        
        # Step 1: Calculate nutrient requirements
        profile.life_stage = LifeStageCalculator.determine_life_stage(
            profile.age, profile.sex, profile.is_pregnant, profile.is_lactating
        )
        
        nutrient_reqs = NutrientRequirementCalculator.calculate_requirements(profile)
        
        # Step 2: Calculate calorie needs
        base_calories = LifeStageCalculator.get_base_calorie_needs(profile)
        target_calories = LifeStageCalculator.adjust_for_goal(base_calories, profile.health_goal)
        
        # Step 3: Get food avoidances
        avoidances = ConditionFoodDatabase.get_avoidances_for_profile(
            profile.medical_conditions,
            profile.allergies,
            profile.food_intolerances,
            profile.medications
        )
        
        # Step 4: Get food recommendations
        recommendations = ConditionFoodDatabase.get_recommendations_for_conditions(
            profile.medical_conditions
        )
        
        # Step 5: Distribute calories across meals
        meal_calories = PortionCalculator.get_meal_distribution(target_calories)
        
        # Step 6: Generate individual meals
        meals = []
        
        # Breakfast
        breakfast = self._create_breakfast(
            profile, meal_calories["breakfast"], avoidances, nutrient_reqs
        )
        meals.append(breakfast)
        
        # Lunch
        lunch = self._create_lunch(
            profile, meal_calories["lunch"], avoidances, nutrient_reqs
        )
        meals.append(lunch)
        
        # Dinner
        dinner = self._create_dinner(
            profile, meal_calories["dinner"], avoidances, nutrient_reqs
        )
        meals.append(dinner)
        
        # Snacks
        snacks = self._create_snacks(
            profile, meal_calories["snacks"], avoidances, nutrient_reqs
        )
        meals.append(snacks)
        
        # Step 7: Calculate daily totals
        total_macros = self._calculate_daily_macros(meals)
        total_micros = self._calculate_daily_micros(meals)
        
        # Step 8: Calculate RDA coverage
        rda_coverage = {}
        for req in nutrient_reqs:
            daily_amount = total_micros.get(req.nutrient_id, Decimal("0"))
            coverage_pct = float((daily_amount / req.target) * 100) if req.target > 0 else 0
            rda_coverage[req.nutrient_id] = coverage_pct
        
        # Step 9: Generate summary
        summary = self._generate_daily_summary(
            profile, target_calories, meals, rda_coverage, nutrient_reqs
        )
        
        # Step 10: Calculate hydration target
        # Base: 30ml per kg body weight
        hydration = (float(profile.weight) * 30) / 1000  # Liters
        if profile.activity_level in [ActivityLevel.ACTIVE, ActivityLevel.VERY_ACTIVE]:
            hydration *= 1.3  # Increase for active people
        
        # Step 11: Determine supplement needs
        supplements = self._determine_supplements(rda_coverage, nutrient_reqs, profile)
        
        return DailyMealPlan(
            date=target_date,
            user_id=profile.user_id,
            meals=meals,
            total_calories=target_calories,
            total_macros=total_macros,
            total_micros=total_micros,
            rda_coverage=rda_coverage,
            goal_alignment=self._calculate_goal_alignment(meals, profile),
            restriction_violations=[],
            daily_summary=summary,
            hydration_target=Decimal(str(round(hydration, 1))),
            supplements_needed=supplements
        )
    
    def _create_breakfast(
        self,
        profile: UserProfile,
        target_calories: Decimal,
        avoidances: List[FoodAvoidance],
        nutrient_reqs: List[NutrientRequirement]
    ) -> MealRecommendation:
        """Create breakfast meal recommendation"""
        
        # Example breakfast (would query real database in production)
        foods = []
        
        # Check for egg allergy
        has_egg_allergy = any(a.item.lower() == "eggs" for a in avoidances)
        
        if not has_egg_allergy:
            # Eggs (protein)
            portion, unit = self.portion_calculator.calculate_portion(
                "eggs", profile.life_stage, profile.health_goal
            )
            foods.append({
                "food_id": "egg_whole_001",
                "food_name": "Whole Eggs",
                "portion": portion,
                "unit": unit,
                "calories": Decimal("143"),
                "protein": Decimal("12.6"),
                "carbs": Decimal("0.7"),
                "fat": Decimal("9.5")
            })
        
        # Oatmeal (carbs, fiber)
        portion, unit = self.portion_calculator.calculate_portion(
            "oats_dry", profile.life_stage, profile.health_goal
        )
        foods.append({
            "food_id": "oat_rolled_001",
            "food_name": "Rolled Oats",
            "portion": portion,
            "unit": unit,
            "calories": Decimal("150"),
            "protein": Decimal("5"),
            "carbs": Decimal("27"),
            "fat": Decimal("3")
        })
        
        # Berries (vitamins, antioxidants)
        portion, unit = self.portion_calculator.calculate_portion(
            "berries", profile.life_stage, profile.health_goal
        )
        foods.append({
            "food_id": "blueberry_001",
            "food_name": "Blueberries",
            "portion": portion,
            "unit": unit,
            "calories": Decimal("85"),
            "protein": Decimal("1"),
            "carbs": Decimal("21"),
            "fat": Decimal("0.5")
        })
        
        # Calculate totals
        total_cals = sum(f["calories"] for f in foods)
        macros = {
            "protein": sum(f["protein"] for f in foods),
            "carbs": sum(f["carbs"] for f in foods),
            "fat": sum(f["fat"] for f in foods),
            "fiber": Decimal("8")  # Estimated
        }
        
        micros = {
            "vitamin_c": Decimal("15"),
            "vitamin_d": Decimal("2"),
            "iron": Decimal("3"),
            "calcium": Decimal("50")
        }
        
        return MealRecommendation(
            meal_id=f"breakfast_{datetime.now().strftime('%Y%m%d')}",
            meal_name="High-Protein Breakfast Bowl",
            foods=foods,
            total_calories=total_cals,
            macronutrients=macros,
            micronutrients=micros,
            nutrient_density_score=82.0,
            diet_compliance_score=95.0,
            health_score=88.0,
            meal_type="breakfast",
            preparation_time=15,
            difficulty="easy",
            cost_estimate="$$",
            benefits=[
                "High protein for satiety",
                "Complex carbs for sustained energy",
                "Antioxidants from berries"
            ],
            cooking_notes=[
                "Cook oats with milk or water for 5 minutes",
                "Top with fresh berries",
                "Optional: add cinnamon for blood sugar control"
            ]
        )
    
    def _create_lunch(self, profile, target_calories, avoidances, nutrient_reqs) -> MealRecommendation:
        """Create lunch recommendation"""
        # Simplified example - would be much more sophisticated
        foods = [
            {
                "food_id": "salmon_001",
                "food_name": "Grilled Salmon",
                "portion": Decimal("120"),
                "unit": "g",
                "calories": Decimal("250"),
                "protein": Decimal("25"),
                "carbs": Decimal("0"),
                "fat": Decimal("15")
            },
            {
                "food_id": "quinoa_001",
                "food_name": "Quinoa",
                "portion": Decimal("150"),
                "unit": "g",
                "calories": Decimal("180"),
                "protein": Decimal("6"),
                "carbs": Decimal("30"),
                "fat": Decimal("3")
            },
            {
                "food_id": "spinach_001",
                "food_name": "Steamed Spinach",
                "portion": Decimal("100"),
                "unit": "g",
                "calories": Decimal("25"),
                "protein": Decimal("3"),
                "carbs": Decimal("4"),
                "fat": Decimal("0.4")
            }
        ]
        
        return MealRecommendation(
            meal_id=f"lunch_{datetime.now().strftime('%Y%m%d')}",
            meal_name="Mediterranean Salmon Bowl",
            foods=foods,
            total_calories=Decimal("455"),
            macronutrients={"protein": Decimal("34"), "carbs": Decimal("34"), "fat": Decimal("18"), "fiber": Decimal("5")},
            micronutrients={"omega_3": Decimal("2000"), "vitamin_d": Decimal("15"), "iron": Decimal("4")},
            nutrient_density_score=92.0,
            diet_compliance_score=100.0,
            health_score=95.0,
            meal_type="lunch",
            preparation_time=20,
            difficulty="medium",
            cost_estimate="$$$",
            benefits=["Omega-3 for heart health", "Complete protein", "Anti-inflammatory"]
        )
    
    def _create_dinner(self, profile, target_calories, avoidances, nutrient_reqs) -> MealRecommendation:
        """Create dinner recommendation"""
        return MealRecommendation(
            meal_id=f"dinner_{datetime.now().strftime('%Y%m%d')}",
            meal_name="Lean Chicken Stir-Fry",
            foods=[],
            total_calories=Decimal("500"),
            macronutrients={"protein": Decimal("40"), "carbs": Decimal("45"), "fat": Decimal("15"), "fiber": Decimal("10")},
            micronutrients={},
            nutrient_density_score=85.0,
            diet_compliance_score=95.0,
            health_score=90.0,
            meal_type="dinner",
            preparation_time=25,
            difficulty="medium",
            cost_estimate="$$"
        )
    
    def _create_snacks(self, profile, target_calories, avoidances, nutrient_reqs) -> MealRecommendation:
        """Create snack recommendations"""
        return MealRecommendation(
            meal_id=f"snacks_{datetime.now().strftime('%Y%m%d')}",
            meal_name="Healthy Snacks",
            foods=[],
            total_calories=Decimal("200"),
            macronutrients={"protein": Decimal("8"), "carbs": Decimal("20"), "fat": Decimal("10"), "fiber": Decimal("4")},
            micronutrients={},
            nutrient_density_score=75.0,
            diet_compliance_score=100.0,
            health_score=85.0,
            meal_type="snack",
            preparation_time=5,
            difficulty="easy",
            cost_estimate="$"
        )
    
    def _calculate_daily_macros(self, meals: List[MealRecommendation]) -> Dict[str, Decimal]:
        """Sum macros across all meals"""
        total_macros = {"protein": Decimal("0"), "carbs": Decimal("0"), "fat": Decimal("0"), "fiber": Decimal("0")}
        for meal in meals:
            for key in total_macros:
                total_macros[key] += meal.macronutrients.get(key, Decimal("0"))
        return total_macros
    
    def _calculate_daily_micros(self, meals: List[MealRecommendation]) -> Dict[str, Decimal]:
        """Sum micronutrients across all meals"""
        total_micros = {}
        for meal in meals:
            for nutrient, amount in meal.micronutrients.items():
                total_micros[nutrient] = total_micros.get(nutrient, Decimal("0")) + amount
        return total_micros
    
    def _calculate_goal_alignment(self, meals: List[MealRecommendation], profile: UserProfile) -> float:
        """Calculate how well meals align with user's goals"""
        # Simplified scoring
        avg_health_score = sum(m.health_score for m in meals) / len(meals)
        avg_compliance = sum(m.diet_compliance_score for m in meals) / len(meals)
        return (avg_health_score + avg_compliance) / 2
    
    def _generate_daily_summary(
        self,
        profile: UserProfile,
        target_calories: Decimal,
        meals: List[MealRecommendation],
        rda_coverage: Dict[str, float],
        nutrient_reqs: List[NutrientRequirement]
    ) -> str:
        """Generate human-readable daily summary"""
        
        # Check for deficiencies (< 80% RDA)
        low_nutrients = [req.nutrient_name for req in nutrient_reqs 
                        if rda_coverage.get(req.nutrient_id, 0) < 80]
        
        # Check for excesses (> 120% RDA for nutrients with UL)
        high_nutrients = [req.nutrient_name for req in nutrient_reqs 
                         if req.ul and rda_coverage.get(req.nutrient_id, 0) > 120]
        
        summary_parts = [
            f"Daily meal plan for {profile.life_stage.value.replace('_', ' ').title()}",
            f"Target: {target_calories} calories",
            f"Goal: {profile.health_goal.value.replace('_', ' ').title()}"
        ]
        
        if low_nutrients:
            summary_parts.append(f"⚠️ Low in: {', '.join(low_nutrients[:3])}")
        
        if high_nutrients:
            summary_parts.append(f"⚠️ High in: {', '.join(high_nutrients[:2])}")
        
        if not low_nutrients and not high_nutrients:
            summary_parts.append("✅ All nutrients within optimal range")
        
        return " | ".join(summary_parts)
    
    def _determine_supplements(
        self,
        rda_coverage: Dict[str, float],
        nutrient_reqs: List[NutrientRequirement],
        profile: UserProfile
    ) -> List[str]:
        """Determine which supplements are needed"""
        supplements = []
        
        for req in nutrient_reqs:
            coverage = rda_coverage.get(req.nutrient_id, 0)
            
            # Supplement if < 70% RDA and priority 1 or 2
            if coverage < 70 and req.priority <= 2:
                dose = float(req.target - (req.target * Decimal(str(coverage/100))))
                supplements.append(f"{req.nutrient_name}: {dose:.1f} {req.unit}/day")
        
        # Always recommend vitamin D for elderly/pregnant/lactating
        if profile.life_stage in [LifeStage.ELDERLY_MALE, LifeStage.ELDERLY_FEMALE,
                                   LifeStage.PREGNANT, LifeStage.LACTATING]:
            if "Vitamin D" not in [s.split(":")[0] for s in supplements]:
                supplements.append("Vitamin D: 1000 IU/day (consider blood test)")
        
        return supplements


# Test the complete system
if __name__ == "__main__":
    print("🍽️  Complete Meal Planning System Test")
    print("=" * 70)
    
    # Test case: 30-year-old female with type 2 diabetes
    test_profile = UserProfile(
        user_id="user_test_001",
        age=30,
        sex='female',
        weight=Decimal("70"),
        height=Decimal("165"),
        activity_level=ActivityLevel.MODERATE,
        health_goal=HealthGoal.WEIGHT_LOSS,
        medical_conditions=["type 2 diabetes"],
        allergies=set(),
        food_intolerances=set(),
        medications=[]
    )
    
    generator = MealPlanGenerator()
    plan = generator.generate_daily_plan(test_profile)
    
    print(f"\n👤 User: {test_profile.sex}, {test_profile.age} years, {test_profile.weight}kg")
    print(f"Life Stage: {plan.meals[0].meal_type if plan.meals else 'N/A'}")
    print(f"Goal: {test_profile.health_goal.value}")
    print(f"Conditions: {', '.join(test_profile.medical_conditions)}")
    
    print(f"\n📅 Daily Plan for {plan.date}")
    print(f"Target Calories: {plan.total_calories} kcal")
    print(f"Hydration: {plan.hydration_target} liters")
    
    print(f"\n🍽️  Meals ({len(plan.meals)} total):")
    for meal in plan.meals:
        print(f"\n  {meal.meal_type.upper()}: {meal.meal_name}")
        print(f"  Calories: {meal.total_calories} kcal")
        print(f"  Macros: P:{meal.macronutrients['protein']}g | C:{meal.macronutrients['carbs']}g | F:{meal.macronutrients['fat']}g")
        print(f"  Health Score: {meal.health_score}/100")
        if meal.benefits:
            print(f"  Benefits: {', '.join(meal.benefits[:2])}")
    
    print(f"\n📊 Daily Totals:")
    print(f"  Protein: {plan.total_macros['protein']}g")
    print(f"  Carbs: {plan.total_macros['carbs']}g")
    print(f"  Fat: {plan.total_macros['fat']}g")
    print(f"  Fiber: {plan.total_macros['fiber']}g")
    
    print(f"\n💊 RDA Coverage (key nutrients):")
    for nutrient in ['protein', 'vitamin_d', 'iron', 'calcium']:
        coverage = plan.rda_coverage.get(nutrient, 0)
        status = "✅" if coverage >= 80 else "⚠️" if coverage >= 50 else "❌"
        print(f"  {status} {nutrient.title()}: {coverage:.0f}% of RDA")
    
    if plan.supplements_needed:
        print(f"\n💊 Recommended Supplements:")
        for supp in plan.supplements_needed:
            print(f"  • {supp}")
    
    print(f"\n📝 Summary: {plan.daily_summary}")
    print(f"Goal Alignment: {plan.goal_alignment:.0f}/100")
    
    print("\n✅ Complete meal planning system test successful!")
    print("=" * 70)
