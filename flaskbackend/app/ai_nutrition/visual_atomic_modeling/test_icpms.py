"""Quick test of the ICP-MS food analysis system"""

from food_nutrient_detector import FoodNutrientDetector

print("\n" + "="*80)
print("TESTING ICP-MS FOOD ANALYZER")
print("="*80)

detector = FoodNutrientDetector()

# Test spinach with ICP-MS data
print("\n Testing Spinach Analysis...")

icpms_data = {
    'Ca': 990,  # Calcium
    'Fe': 27,   # Iron
    'Mg': 790,  # Magnesium
    'K': 5580,  # Potassium
    'Zn': 5,    # Zinc
    'Pb': 0.02, # Lead (safe level)
}

profile = detector.analyze_food('spinach', icpms_data=icpms_data)

print(f"\nFood: {profile.food_name}")
print(f"Calories: {profile.calories} kcal/100g")
print(f"Protein: {profile.protein_g}g")
print(f"Calcium: {profile.calcium_mg}mg")
print(f"Iron: {profile.iron_mg}mg")
print(f"\nSafety: {'✅ SAFE' if profile.is_safe_for_consumption else '❌ UNSAFE'}")
print(f"Quality Score: {profile.nutritional_quality_score}/100")

print(f"\nElemental Analysis ({len(profile.elemental_composition)} elements detected):")
for elem in profile.elemental_composition[:5]:
    print(f"  {elem.element_name}: {elem.concentration_ppm:.2f} ppm - {'✅' if elem.is_safe else '⚠️'}")

print("\n" + "="*80)
print("✅ SYSTEM WORKING")
print("="*80)
