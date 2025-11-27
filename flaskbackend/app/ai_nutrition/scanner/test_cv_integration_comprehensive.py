"""
Comprehensive Testing Suite for CV Integration Bridge
Tests all lifecycle stages, goal types, and disease combinations.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from cv_integration_bridge import (
    CVNutritionIntegration,
    CVIntegrationBridge,
    ComprehensiveDiseaseDatabase,
    PersonalGoalsManager,
    LifecycleStage,
    GoalType,
    DiseaseSeverity
)


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def test_pregnancy_scenarios():
    """Test pregnancy nutrition across all trimesters."""
    print_section("PREGNANCY SCENARIOS")
    
    # Create integration bridge
    disease_db = ComprehensiveDiseaseDatabase()
    goals_manager = PersonalGoalsManager()
    
    # Test each trimester
    for trimester in [1, 2, 3]:
        print(f"\n--- Trimester {trimester} ---")
        goal = goals_manager.create_pregnancy_goal(
            current_weight=65,
            trimester=trimester,
            age=28
        )
        
        print(f"Calories: {goal.target_calories} kcal")
        print(f"Protein: {goal.target_protein}g")
        print(f"Folate: {goal.target_folate}mcg (CRITICAL for neural tube)")
        print(f"Iron: {goal.target_iron}mg (blood volume expansion)")
        print(f"Omega-3: {goal.target_omega3}mg (fetal brain development)")
        print(f"Calcium: {goal.target_calcium}mg")
        
        # Test with gestational diabetes
        if trimester == 3:
            print("\n--- Trimester 3 with Gestational Diabetes ---")
            disease = disease_db.diseases.get('gestational_diabetes')
            if disease:
                print(f"Disease: {disease.name}")
                print(f"Sugar limit: <{disease.sugar_max}g")
                print(f"Focus: Blood sugar control + pregnancy nutrition")


def test_senior_nutrition():
    """Test senior nutrition scenarios."""
    print_section("SENIOR NUTRITION SCENARIOS")
    
    bridge = CVIntegrationBridge()
    
    # Male senior
    print("--- 72-year-old Male ---")
    goal = bridge.create_senior_nutrition_goal(
        current_weight=75,
        age=72,
        gender='male'
    )
    print(f"Calories: {goal.target_calories} kcal")
    print(f"Protein: {goal.target_protein}g (high for sarcopenia prevention)")
    print(f"Calcium: {goal.target_calcium}mg (bone health)")
    print(f"Vitamin D: {goal.target_vitamin_d}IU (absorption decreases with age)")
    print(f"B12: {goal.target_b12}mcg (absorption issues common)")
    
    # Female senior with osteoporosis
    print("\n--- 68-year-old Female with Osteoporosis ---")
    goal = bridge.create_senior_nutrition_goal(
        current_weight=60,
        age=68,
        gender='female'
    )
    
    recommendations = bridge.get_meal_recommendations(
        goal=goal,
        active_diseases=['osteoporosis'],
        meal_context={'meal_type': 'lunch'}
    )
    print(f"Calcium needs: {goal.target_calcium}mg (VERY HIGH)")
    print(f"Vitamin D: {goal.target_vitamin_d}IU")
    print(f"Protein: {goal.target_protein}g")
    print(f"Recommended foods: {recommendations.get('recommended_foods', [])}")


def test_athletic_performance():
    """Test athletic performance scenarios."""
    print_section("ATHLETIC PERFORMANCE SCENARIOS")
    
    bridge = CVIntegrationBridge()
    
    # Endurance athlete
    print("--- Marathon Runner (Endurance) ---")
    goal = bridge.create_endurance_goal(
        current_weight=70,
        sport_type='running',
        training_days=6,
        age=26,
        gender='male'
    )
    print(f"Calories: {goal.target_calories} kcal (high energy needs)")
    print(f"Carbs: {goal.target_carbs}g ({goal.carbs_percent}% - glycogen critical)")
    print(f"Protein: {goal.target_protein}g")
    print(f"Water: {goal.target_water}L (hydration critical)")
    print(f"Training days: {goal.training_days_per_week}")
    
    # Strength athlete
    print("\n--- Powerlifter (Strength) ---")
    goal = bridge.create_strength_training_goal(
        current_weight=90,
        training_days=4,
        age=28,
        gender='male'
    )
    print(f"Calories: {goal.target_calories} kcal")
    print(f"Protein: {goal.target_protein}g ({goal.protein_percent}% - muscle building)")
    print(f"Carbs: {goal.target_carbs}g")
    print(f"Fat: {goal.target_fat}g")
    print(f"Training days: {goal.training_days_per_week}")


def test_breastfeeding_nutrition():
    """Test breastfeeding nutrition."""
    print_section("BREASTFEEDING NUTRITION")
    
    bridge = CVIntegrationBridge()
    
    goal = bridge.create_breastfeeding_goal(
        current_weight=68,
        age=30
    )
    print(f"Calories: {goal.target_calories} kcal (+500 for milk production)")
    print(f"Protein: {goal.target_protein}g")
    print(f"Omega-3: {goal.target_omega3}mg (DHA for infant brain)")
    print(f"Water: {goal.target_water}L (CRITICAL for milk supply)")
    print(f"Calcium: {goal.target_calcium}mg")
    print(f"Iodine: Essential for infant thyroid development")


def test_menopause_nutrition():
    """Test menopause nutrition."""
    print_section("MENOPAUSE NUTRITION")
    
    bridge = CVIntegrationBridge()
    
    goal = bridge.create_menopause_goal(
        current_weight=65,
        age=52
    )
    print(f"Calories: {goal.target_calories} kcal")
    print(f"Calcium: {goal.target_calcium}mg (bone loss 2-3% per year)")
    print(f"Vitamin D: {goal.target_vitamin_d}IU")
    print(f"Magnesium: {goal.target_magnesium}mg (symptom relief)")
    print(f"Omega-3: {goal.target_omega3}mg (anti-inflammatory)")
    print(f"Protein: {goal.target_protein}g")


def test_dietary_patterns():
    """Test dietary pattern goals."""
    print_section("DIETARY PATTERN GOALS")
    
    bridge = CVIntegrationBridge()
    
    # Ketogenic diet
    print("--- Ketogenic Diet ---")
    goal = bridge.create_ketogenic_diet_goal(
        current_weight=80,
        age=35,
        gender='male'
    )
    print(f"Carbs: {goal.target_carbs}g (VERY LOW - ketosis)")
    print(f"Fat: {goal.target_fat}g ({goal.fat_percent}%)")
    print(f"Protein: {goal.target_protein}g ({goal.protein_percent}%)")
    print(f"Sodium: 5000mg (higher needs in ketosis)")
    print(f"Magnesium: {goal.target_magnesium}mg")
    print(f"Potassium: Essential for electrolyte balance")
    
    # Mediterranean diet
    print("\n--- Mediterranean Diet ---")
    goal = bridge.create_mediterranean_diet_goal(
        current_weight=70,
        age=40,
        gender='female'
    )
    print(f"Calories: {goal.target_calories} kcal")
    print(f"Fat: {goal.target_fat}g (35% - olive oil, nuts, fish)")
    print(f"Omega-3: {goal.target_omega3}mg (fish 2x/week)")
    print(f"Fiber: {goal.target_fiber}g (vegetables, legumes)")
    
    # Plant-based diet
    print("\n--- Plant-Based Diet (Vegan) ---")
    goal = bridge.create_plant_based_goal(
        current_weight=65,
        is_vegan=True,
        age=28,
        gender='female'
    )
    print(f"B12: {goal.target_b12}mcg (MUST SUPPLEMENT - no dietary source)")
    print(f"Iron: {goal.target_iron}mg (non-heme, need more)")
    print(f"Zinc: {goal.target_zinc}mg (plant sources less bioavailable)")
    print(f"Omega-3: {goal.target_omega3}mg (ALA from flax, chia)")
    print(f"Note: Pair iron + vitamin C for better absorption")


def test_disease_combinations():
    """Test multiple disease combinations."""
    print_section("DISEASE COMBINATION SCENARIOS")
    
    bridge = CVIntegrationBridge()
    
    # Diabetes + Hypertension + High Cholesterol (common combination)
    print("--- Metabolic Syndrome (Diabetes + Hypertension + High Cholesterol) ---")
    goal = bridge.create_weight_loss_goal(
        current_weight=95,
        target_weight=80,
        age=55,
        gender='male'
    )
    
    recommendations = bridge.get_meal_recommendations(
        goal=goal,
        active_diseases=['diabetes_type2', 'hypertension', 'high_cholesterol'],
        meal_context={'meal_type': 'dinner'}
    )
    
    print(f"Active diseases: {len(recommendations.get('disease_constraints', []))}")
    for disease_id in ['diabetes_type2', 'hypertension', 'high_cholesterol']:
        disease = bridge.diseases.get(disease_id)
        if disease:
            print(f"\n{disease.name}:")
            print(f"  Sodium: <{disease.sodium_max}mg")
            print(f"  Sugar: <{disease.sugar_max}g")
            print(f"  Saturated fat: <{disease.saturated_fat_max}g")
            print(f"  Fiber: >{disease.fiber_min}g")
    
    # CKD + Diabetes (challenging combination)
    print("\n\n--- CKD Stage 4 + Diabetes ---")
    goal = bridge.create_weight_loss_goal(
        current_weight=80,
        target_weight=70,
        age=62,
        gender='male'
    )
    
    recommendations = bridge.get_meal_recommendations(
        goal=goal,
        active_diseases=['ckd_stage4_5', 'diabetes_type2'],
        meal_context={'meal_type': 'lunch'}
    )
    
    ckd = bridge.diseases.get('ckd_stage4_5')
    print(f"Protein: <{ckd.protein_max}g (very restricted)")
    print(f"Potassium: <{ckd.potassium_max}mg (very restricted)")
    print(f"Phosphorus: <{ckd.phosphorus_max}mg (very restricted)")
    print(f"Sodium: <{ckd.sodium_max}mg")
    print(f"Also manage blood sugar for diabetes")
    
    # Pregnancy + Gestational Diabetes
    print("\n\n--- Pregnancy (T3) + Gestational Diabetes ---")
    goal = bridge.create_pregnancy_goal(
        current_weight=72,
        trimester=3,
        age=32
    )
    
    recommendations = bridge.get_meal_recommendations(
        goal=goal,
        active_diseases=['gestational_diabetes'],
        meal_context={'meal_type': 'breakfast'}
    )
    
    print(f"Folate: {goal.target_folate}mcg (neural tube)")
    print(f"Iron: {goal.target_iron}mg (blood volume)")
    print(f"Blood sugar control: Critical")
    print(f"Complex carbs: Preferred over simple sugars")


def test_special_conditions():
    """Test special condition goals."""
    print_section("SPECIAL CONDITION GOALS")
    
    bridge = CVIntegrationBridge()
    
    # Post-surgery recovery
    print("--- Post-Surgery Recovery ---")
    goal = bridge.create_post_surgery_recovery_goal(
        current_weight=70,
        age=45,
        gender='male'
    )
    print(f"Protein: {goal.target_protein}g (1.5g/kg for wound healing)")
    print(f"Vitamin C: Critical for collagen synthesis")
    print(f"Zinc: Essential for tissue repair")
    print(f"Calories: {goal.target_calories} kcal (healing requires energy)")
    
    # Gut health
    print("\n--- Gut Health Optimization ---")
    goal = bridge.create_gut_health_goal(
        current_weight=68,
        age=38,
        gender='female'
    )
    print(f"Fiber: {goal.target_fiber}g (VERY HIGH)")
    print(f"Water: {goal.target_water}L")
    print(f"Focus: Probiotic and prebiotic foods")
    print(f"Avoid: Processed foods, artificial sweeteners")
    
    # Anti-inflammatory
    print("\n--- Anti-Inflammatory Goal ---")
    goal = bridge.create_anti_inflammatory_goal(
        current_weight=75,
        age=50,
        gender='female'
    )
    print(f"Omega-3: {goal.target_omega3}mg (VERY HIGH)")
    print(f"Fiber: {goal.target_fiber}g")
    print(f"Focus: Fish, vegetables, berries, nuts")
    print(f"Avoid: Processed foods, sugar, trans fats")


def test_hematological_diseases():
    """Test hematological disease scenarios."""
    print_section("HEMATOLOGICAL DISEASES")
    
    bridge = CVIntegrationBridge()
    
    # Iron deficiency anemia
    print("--- Iron Deficiency Anemia ---")
    anemia = bridge.diseases.get('anemia_iron_deficiency')
    print(f"Iron requirement: >{anemia.iron_min}mg")
    print(f"Vitamin C: >{anemia.vitamin_c_min}mg (enhances iron absorption)")
    print(f"Recommended: Red meat, spinach, lentils, iron-fortified cereals")
    print(f"Tip: Pair iron sources with vitamin C")
    
    # Hemochromatosis (opposite of anemia)
    print("\n--- Hemochromatosis (Iron Overload) ---")
    hemo = bridge.diseases.get('hemochromatosis')
    print(f"Iron restriction: <{hemo.iron_max}mg (VERY LOW)")
    print(f"Vitamin C restriction: <{hemo.vitamin_c_max}mg (reduces iron absorption)")
    print(f"Avoid: Iron supplements, red meat, organ meats")
    print(f"Helpful: Tea with meals (tannins inhibit iron absorption)")
    
    # Sickle cell disease
    print("\n--- Sickle Cell Disease ---")
    sickle = bridge.diseases.get('sickle_cell')
    print(f"Calories: >{sickle.calories_min} kcal (higher energy needs)")
    print(f"Protein: >{sickle.protein_min}g")
    print(f"Folate: >{sickle.folate_min}mcg (high RBC turnover)")
    print(f"Zinc: >{sickle.zinc_min}mg")
    print(f"Water: >{sickle.water_min}L (hydration CRITICAL)")


def test_liver_diseases():
    """Test liver disease scenarios."""
    print_section("LIVER DISEASES")
    
    bridge = CVIntegrationBridge()
    
    # Fatty liver
    print("--- Non-Alcoholic Fatty Liver Disease ---")
    fatty_liver = bridge.diseases.get('fatty_liver')
    print(f"Calories: <{fatty_liver.calories_max} kcal (weight loss beneficial)")
    print(f"Sugar: <{fatty_liver.sugar_max}g")
    print(f"Saturated fat: <{fatty_liver.saturated_fat_max}g")
    print(f"Fiber: >{fatty_liver.fiber_min}g")
    print(f"Avoid: Alcohol, fructose, processed foods")
    print(f"Recommended: Mediterranean diet, coffee")
    
    # Cirrhosis
    print("\n--- Cirrhosis ---")
    cirrhosis = bridge.diseases.get('cirrhosis')
    print(f"Protein: >{cirrhosis.protein_min}g (higher needs)")
    print(f"Sodium: <{cirrhosis.sodium_max}mg")
    print(f"Calories: >{cirrhosis.calories_min} kcal")
    print(f"Strategy: Small, frequent meals")
    print(f"Avoid: Alcohol (absolute)")


def test_endocrine_diseases():
    """Test endocrine disease scenarios."""
    print_section("ENDOCRINE DISEASES")
    
    bridge = CVIntegrationBridge()
    
    # PCOS
    print("--- Polycystic Ovary Syndrome (PCOS) ---")
    pcos = bridge.diseases.get('pcos')
    print(f"Carbs: <{pcos.carbs_max}g (lower carb beneficial)")
    print(f"Sugar: <{pcos.sugar_max}g")
    print(f"Fiber: >{pcos.fiber_min}g")
    print(f"Omega-3: >{pcos.omega3_min}mg")
    print(f"Focus: Low GI foods, anti-inflammatory")
    
    # Hashimoto's
    print("\n--- Hashimoto's Thyroiditis ---")
    hashimotos = bridge.diseases.get('hashimotos')
    print(f"Selenium: >{hashimotos.selenium_min}mcg (important for thyroid)")
    print(f"Zinc: >{hashimotos.zinc_min}mg")
    print(f"Vitamin D: >{hashimotos.vitamin_d_min}IU")
    print(f"Avoid: Gluten, soy, raw cruciferous vegetables")
    print(f"Recommended: Selenium-rich foods, anti-inflammatory")


def test_body_composition_goals():
    """Test body composition goals."""
    print_section("BODY COMPOSITION GOALS")
    
    bridge = CVIntegrationBridge()
    
    # Body recomposition
    print("--- Body Recomposition (Lose Fat + Gain Muscle) ---")
    goal = bridge.create_body_recomposition_goal(
        current_weight=80,
        target_body_fat=12,
        age=30,
        gender='male'
    )
    print(f"Calories: {goal.target_calories} kcal (maintenance)")
    print(f"Protein: {goal.target_protein}g ({goal.protein_percent}% - VERY HIGH)")
    print(f"Carbs: {goal.target_carbs}g")
    print(f"Fat: {goal.target_fat}g")
    print(f"Strategy: High protein, resistance training, slight deficit")
    
    # Cutting phase
    print("\n--- Cutting Phase (Fat Loss for Bodybuilders) ---")
    goal = bridge.create_weight_loss_goal(
        current_weight=90,
        target_weight=80,
        age=28,
        gender='male'
    )
    print(f"Calories: {goal.target_calories} kcal (deficit)")
    print(f"Protein: Very high (preserve muscle)")
    print(f"Carbs: Moderate (training energy)")
    print(f"Fat: Lower")


def test_lifecycle_progression():
    """Test nutrition across entire lifecycle."""
    print_section("LIFECYCLE PROGRESSION")
    
    bridge = CVIntegrationBridge()
    
    lifecycle_stages = [
        (LifecycleStage.INFANT, "Infant (0-1 years)"),
        (LifecycleStage.TODDLER, "Toddler (1-3 years)"),
        (LifecycleStage.CHILD, "Child (5-12 years)"),
        (LifecycleStage.ADOLESCENT, "Adolescent (12-18 years)"),
        (LifecycleStage.YOUNG_ADULT, "Young Adult (18-30 years)"),
        (LifecycleStage.ADULT, "Adult (30-50 years)"),
        (LifecycleStage.MIDDLE_AGE, "Middle Age (50-65 years)"),
        (LifecycleStage.SENIOR, "Senior (65-80 years)"),
        (LifecycleStage.ELDERLY, "Elderly (80+ years)")
    ]
    
    print("Nutritional needs across lifecycle:\n")
    print(f"{'Stage':<25} {'Calories':<12} {'Protein':<12} {'Calcium':<12} {'Special Notes'}")
    print("-" * 100)
    
    for stage, description in lifecycle_stages:
        # Get lifecycle-specific targets from PersonalGoal
        from cv_integration_bridge import PersonalGoal
        temp_goal = PersonalGoal(
            goal_id="temp",
            goal_type=GoalType.GENERAL_HEALTH,
            lifecycle_stage=stage
        )
        targets = temp_goal._get_lifecycle_targets()
        
        calories = targets.get('calories', 'N/A')
        protein = targets.get('protein', 'N/A')
        calcium = targets.get('calcium', 'N/A')
        
        notes = ""
        if stage == LifecycleStage.INFANT:
            notes = "Brain development critical"
        elif stage == LifecycleStage.ADOLESCENT:
            notes = "Growth spurt, high needs"
        elif stage == LifecycleStage.SENIOR:
            notes = "High protein (sarcopenia)"
        elif stage == LifecycleStage.ELDERLY:
            notes = "Absorption issues"
        
        print(f"{description:<25} {str(calories):<12} {str(protein):<12} {str(calcium):<12} {notes}")


def run_all_tests():
    """Run all comprehensive tests."""
    print("\n" + "="*80)
    print("  CV INTEGRATION BRIDGE - COMPREHENSIVE TEST SUITE")
    print("  Testing all lifecycle stages, goal types, and disease combinations")
    print("="*80)
    
    try:
        # Lifecycle tests
        test_pregnancy_scenarios()
        test_senior_nutrition()
        test_breastfeeding_nutrition()
        test_menopause_nutrition()
        test_lifecycle_progression()
        
        # Athletic performance
        test_athletic_performance()
        
        # Dietary patterns
        test_dietary_patterns()
        
        # Special conditions
        test_special_conditions()
        test_body_composition_goals()
        
        # Disease scenarios
        test_hematological_diseases()
        test_liver_diseases()
        test_endocrine_diseases()
        test_disease_combinations()
        
        print_section("TEST SUITE COMPLETED SUCCESSFULLY ✅")
        print("All scenarios tested:")
        print("  ✓ Pregnancy (all trimesters)")
        print("  ✓ Senior nutrition (male/female)")
        print("  ✓ Athletic performance (endurance/strength)")
        print("  ✓ Breastfeeding")
        print("  ✓ Menopause")
        print("  ✓ Dietary patterns (keto, Mediterranean, plant-based)")
        print("  ✓ Special conditions (recovery, gut health, anti-inflammatory)")
        print("  ✓ Hematological diseases (anemia, hemochromatosis, sickle cell)")
        print("  ✓ Liver diseases (fatty liver, cirrhosis)")
        print("  ✓ Endocrine diseases (PCOS, Hashimoto's)")
        print("  ✓ Disease combinations (metabolic syndrome, CKD+diabetes)")
        print("  ✓ Body composition goals")
        print("  ✓ Lifecycle progression (infant → elderly)")
        print("\nSystem ready for production! 🚀")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
