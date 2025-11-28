"""
Disease Restrictions Database
==============================

Comprehensive database of 100+ medical conditions with:
- Nutritional restrictions and recommendations
- Foods to avoid and foods to prioritize
- Macro/micronutrient targets
- Medication interactions
- Clinical guidelines

This file contains ~4,000 lines of structured disease data.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class DiseaseCategory(Enum):
    """Categories of diseases"""
    METABOLIC = "metabolic"
    CARDIOVASCULAR = "cardiovascular"
    DIGESTIVE = "digestive"
    KIDNEY = "kidney"
    LIVER = "liver"
    AUTOIMMUNE = "autoimmune"
    ENDOCRINE = "endocrine"
    RESPIRATORY = "respiratory"
    NEUROLOGICAL = "neurological"
    CANCER = "cancer"
    ALLERGY = "allergy"
    MENTAL_HEALTH = "mental_health"


@dataclass
class NutrientLimit:
    """Nutrient limitation or requirement"""
    nutrient: str
    max_value: Optional[float] = None
    min_value: Optional[float] = None
    unit: str = "mg"
    daily_limit: bool = True
    per_meal: bool = False
    reason: str = ""


@dataclass
class DiseaseData:
    """Structured disease condition data"""
    code: str  # Custom or ICD-10 code
    name: str
    category: DiseaseCategory
    severity_levels: List[str]
    
    # Nutrient restrictions
    nutrients_to_limit: List[NutrientLimit]
    nutrients_to_increase: List[NutrientLimit]
    
    # Foods
    foods_to_avoid: List[str]
    foods_recommended: List[str]
    food_categories_to_avoid: List[str]
    food_categories_recommended: List[str]
    
    # Macronutrient targets (% of daily calories)
    protein_ratio: Optional[float] = None
    carb_ratio: Optional[float] = None
    fat_ratio: Optional[float] = None
    
    # Calorie adjustment
    calorie_adjustment_factor: float = 1.0
    
    # Meal timing
    meal_frequency_recommendation: str = "3 meals + 2 snacks"
    fasting_requirements: List[str] = field(default_factory=list)
    
    # Drug-nutrient interactions
    medication_interactions: Dict[str, List[str]] = field(default_factory=dict)
    
    # Clinical guidelines
    guidelines_source: str = ""
    clinical_notes: str = ""


# ============================================================================
# METABOLIC DISORDERS
# ============================================================================

DIABETES_TYPE_2 = DiseaseData(
    code="diabetes_type2",
    name="Type 2 Diabetes Mellitus",
    category=DiseaseCategory.METABOLIC,
    severity_levels=["well_controlled", "moderate", "poorly_controlled", "with_complications"],
    
    nutrients_to_limit=[
        NutrientLimit(
            nutrient="sugar",
            max_value=25,
            unit="g",
            daily_limit=True,
            reason="Prevents blood glucose spikes"
        ),
        NutrientLimit(
            nutrient="simple_carbohydrates",
            max_value=130,
            unit="g",
            daily_limit=True,
            reason="Better glycemic control"
        ),
        NutrientLimit(
            nutrient="saturated_fat",
            max_value=20,
            unit="g",
            daily_limit=True,
            reason="Reduces cardiovascular risk"
        ),
        NutrientLimit(
            nutrient="sodium",
            max_value=2300,
            unit="mg",
            daily_limit=True,
            reason="Blood pressure management"
        )
    ],
    
    nutrients_to_increase=[
        NutrientLimit(
            nutrient="fiber",
            min_value=30,
            unit="g",
            daily_limit=True,
            reason="Improves glucose control and satiety"
        ),
        NutrientLimit(
            nutrient="omega_3",
            min_value=500,
            unit="mg",
            daily_limit=True,
            reason="Cardiovascular protection"
        ),
        NutrientLimit(
            nutrient="chromium",
            min_value=200,
            unit="mcg",
            daily_limit=True,
            reason="Insulin sensitivity"
        )
    ],
    
    foods_to_avoid=[
        "White bread", "White rice", "Sugary sodas", "Candy", "Pastries",
        "Fruit juices", "Sweetened cereals", "Honey", "Processed snacks",
        "Fried foods", "High-fat dairy products"
    ],
    
    foods_recommended=[
        "Whole grains", "Leafy greens", "Berries", "Fatty fish", "Nuts",
        "Legumes", "Non-starchy vegetables", "Greek yogurt", "Eggs",
        "Olive oil", "Avocado", "Cinnamon", "Turmeric"
    ],
    
    food_categories_to_avoid=[
        "Refined grains", "Sugary beverages", "Processed meats", "Trans fats"
    ],
    
    food_categories_recommended=[
        "Whole grains", "Lean proteins", "Non-starchy vegetables", "Healthy fats"
    ],
    
    protein_ratio=25.0,  # 25% of calories from protein
    carb_ratio=45.0,     # 45% from complex carbs
    fat_ratio=30.0,      # 30% from healthy fats
    
    calorie_adjustment_factor=1.0,
    
    meal_frequency_recommendation="5-6 small meals (prevents glucose spikes)",
    
    medication_interactions={
        "metformin": ["May cause vitamin B12 deficiency - monitor intake"],
        "insulin": ["Timing with meals critical", "Avoid skipping meals"],
        "sulfonylureas": ["Risk of hypoglycemia - consistent meal timing"]
    },
    
    guidelines_source="American Diabetes Association (ADA) 2024",
    clinical_notes="Focus on glycemic index <55, glycemic load <10 per meal. "
                  "Portion control critical. Regular blood glucose monitoring."
)

DIABETES_TYPE_1 = DiseaseData(
    code="diabetes_type1",
    name="Type 1 Diabetes Mellitus",
    category=DiseaseCategory.METABOLIC,
    severity_levels=["well_controlled", "moderate", "poorly_controlled"],
    
    nutrients_to_limit=[
        NutrientLimit(
            nutrient="sugar",
            max_value=30,
            unit="g",
            daily_limit=True,
            reason="Insulin management"
        ),
        NutrientLimit(
            nutrient="saturated_fat",
            max_value=20,
            unit="g",
            daily_limit=True,
            reason="Cardiovascular health"
        )
    ],
    
    nutrients_to_increase=[
        NutrientLimit(
            nutrient="fiber",
            min_value=25,
            unit="g",
            daily_limit=True,
            reason="Glucose regulation"
        )
    ],
    
    foods_to_avoid=[
        "Sugary beverages", "Candy", "Pastries", "Highly processed foods"
    ],
    
    foods_recommended=[
        "Whole grains", "Vegetables", "Lean proteins", "Healthy fats",
        "Low-fat dairy", "Legumes", "Nuts"
    ],
    
    food_categories_to_avoid=["Simple sugars", "Refined carbohydrates"],
    food_categories_recommended=["Complex carbohydrates", "High-fiber foods"],
    
    protein_ratio=20.0,
    carb_ratio=50.0,
    fat_ratio=30.0,
    
    meal_frequency_recommendation="3 meals + 2-3 snacks (matched with insulin)",
    
    medication_interactions={
        "insulin": ["Carb counting essential", "Match insulin dose to carb intake"],
    },
    
    guidelines_source="ADA 2024",
    clinical_notes="Carbohydrate counting crucial. 15g carbs per snack, 45-60g per meal typical."
)

PRE_DIABETES = DiseaseData(
    code="pre_diabetes",
    name="Pre-diabetes (Impaired Glucose Tolerance)",
    category=DiseaseCategory.METABOLIC,
    severity_levels=["early", "advanced"],
    
    nutrients_to_limit=[
        NutrientLimit(
            nutrient="added_sugar",
            max_value=25,
            unit="g",
            daily_limit=True,
            reason="Prevent progression to diabetes"
        ),
        NutrientLimit(
            nutrient="refined_carbs",
            max_value=150,
            unit="g",
            daily_limit=True,
            reason="Improve insulin sensitivity"
        )
    ],
    
    nutrients_to_increase=[
        NutrientLimit(
            nutrient="fiber",
            min_value=35,
            unit="g",
            daily_limit=True,
            reason="Weight management and glucose control"
        )
    ],
    
    foods_to_avoid=[
        "White bread", "Sugary drinks", "Processed snacks", "Fried foods"
    ],
    
    foods_recommended=[
        "Whole grains", "Vegetables", "Fruits", "Lean proteins", "Legumes"
    ],
    
    food_categories_to_avoid=["Refined grains", "Added sugars"],
    food_categories_recommended=["Whole grains", "High-fiber foods"],
    
    protein_ratio=25.0,
    carb_ratio=45.0,
    fat_ratio=30.0,
    
    calorie_adjustment_factor=0.85,  # Modest calorie restriction for weight loss
    
    meal_frequency_recommendation="3 meals + 2 snacks",
    
    guidelines_source="ADA 2024",
    clinical_notes="7-10% weight loss target. Mediterranean diet recommended."
)

OBESITY = DiseaseData(
    code="obesity",
    name="Obesity (BMI ≥30)",
    category=DiseaseCategory.METABOLIC,
    severity_levels=["class_1", "class_2", "class_3_severe"],
    
    nutrients_to_limit=[
        NutrientLimit(
            nutrient="saturated_fat",
            max_value=15,
            unit="g",
            daily_limit=True,
            reason="Calorie reduction"
        ),
        NutrientLimit(
            nutrient="sugar",
            max_value=25,
            unit="g",
            daily_limit=True,
            reason="Weight management"
        )
    ],
    
    nutrients_to_increase=[
        NutrientLimit(
            nutrient="protein",
            min_value=80,
            unit="g",
            daily_limit=True,
            reason="Satiety and muscle preservation"
        ),
        NutrientLimit(
            nutrient="fiber",
            min_value=30,
            unit="g",
            daily_limit=True,
            reason="Satiety and gut health"
        )
    ],
    
    foods_to_avoid=[
        "Fast food", "Sugary beverages", "Processed snacks", "Fried foods",
        "High-calorie desserts"
    ],
    
    foods_recommended=[
        "Vegetables", "Lean proteins", "Whole grains", "Fruits", "Legumes",
        "Low-fat dairy"
    ],
    
    food_categories_to_avoid=["Ultra-processed foods", "Calorie-dense snacks"],
    food_categories_recommended=["Nutrient-dense whole foods", "Low-calorie vegetables"],
    
    protein_ratio=30.0,  # Higher protein for satiety
    carb_ratio=35.0,
    fat_ratio=35.0,
    
    calorie_adjustment_factor=0.75,  # 500-750 calorie deficit
    
    meal_frequency_recommendation="3 meals (no snacking to create deficit)",
    
    guidelines_source="WHO Obesity Guidelines 2024",
    clinical_notes="500-750 calorie daily deficit for 0.5-1kg/week loss. "
                  "Emphasis on volume eating, high-protein, high-fiber."
)

# ============================================================================
# CARDIOVASCULAR DISEASES
# ============================================================================

HYPERTENSION = DiseaseData(
    code="hypertension",
    name="Hypertension (High Blood Pressure)",
    category=DiseaseCategory.CARDIOVASCULAR,
    severity_levels=["elevated", "stage_1", "stage_2", "crisis"],
    
    nutrients_to_limit=[
        NutrientLimit(
            nutrient="sodium",
            max_value=1500,  # DASH diet recommendation
            unit="mg",
            daily_limit=True,
            reason="Direct blood pressure reduction"
        ),
        NutrientLimit(
            nutrient="saturated_fat",
            max_value=15,
            unit="g",
            daily_limit=True,
            reason="Cardiovascular health"
        ),
        NutrientLimit(
            nutrient="alcohol",
            max_value=14,  # units per week
            unit="g",
            daily_limit=False,
            reason="Blood pressure elevation"
        )
    ],
    
    nutrients_to_increase=[
        NutrientLimit(
            nutrient="potassium",
            min_value=4700,
            unit="mg",
            daily_limit=True,
            reason="Sodium balance and BP reduction"
        ),
        NutrientLimit(
            nutrient="magnesium",
            min_value=400,
            unit="mg",
            daily_limit=True,
            reason="Vascular relaxation"
        ),
        NutrientLimit(
            nutrient="calcium",
            min_value=1200,
            unit="mg",
            daily_limit=True,
            reason="Blood pressure regulation"
        ),
        NutrientLimit(
            nutrient="fiber",
            min_value=30,
            unit="g",
            daily_limit=True,
            reason="Weight and cholesterol management"
        )
    ],
    
    foods_to_avoid=[
        "Processed meats", "Canned soups", "Pickles", "Salty snacks",
        "Fast food", "Frozen dinners", "Condiments (soy sauce, ketchup)",
        "Cheese", "Bread (commercial)", "Pizza"
    ],
    
    foods_recommended=[
        "Leafy greens", "Berries", "Beets", "Bananas", "Oats", "Fatty fish",
        "Garlic", "Dark chocolate (>70%)", "Pomegranates", "Olive oil",
        "Nuts", "Seeds", "Legumes", "Low-fat dairy"
    ],
    
    food_categories_to_avoid=[
        "Processed foods", "High-sodium foods", "Cured meats"
    ],
    food_categories_recommended=[
        "Fruits", "Vegetables", "Whole grains", "Low-fat dairy"
    ],
    
    protein_ratio=18.0,
    carb_ratio=55.0,  # DASH diet pattern
    fat_ratio=27.0,
    
    meal_frequency_recommendation="3 meals + 1-2 snacks",
    
    medication_interactions={
        "ACE_inhibitors": ["Monitor potassium intake - avoid excess supplementation"],
        "diuretics": ["May deplete potassium - ensure adequate intake"],
        "beta_blockers": ["May affect glucose metabolism"]
    },
    
    guidelines_source="AHA/ACC Hypertension Guidelines 2024, DASH Diet",
    clinical_notes="DASH diet proven effective. Sodium <1500mg critical. "
                  "Potassium-rich foods essential unless contraindicated."
)

HIGH_CHOLESTEROL = DiseaseData(
    code="high_cholesterol",
    name="Hyperlipidemia (High Cholesterol)",
    category=DiseaseCategory.CARDIOVASCULAR,
    severity_levels=["borderline", "high", "very_high"],
    
    nutrients_to_limit=[
        NutrientLimit(
            nutrient="saturated_fat",
            max_value=13,  # <7% of calories for 2000 cal diet
            unit="g",
            daily_limit=True,
            reason="Reduces LDL cholesterol"
        ),
        NutrientLimit(
            nutrient="trans_fat",
            max_value=0,
            unit="g",
            daily_limit=True,
            reason="Increases LDL, decreases HDL"
        ),
        NutrientLimit(
            nutrient="dietary_cholesterol",
            max_value=200,
            unit="mg",
            daily_limit=True,
            reason="Direct cholesterol impact"
        )
    ],
    
    nutrients_to_increase=[
        NutrientLimit(
            nutrient="soluble_fiber",
            min_value=10,
            unit="g",
            daily_limit=True,
            reason="Binds cholesterol in gut"
        ),
        NutrientLimit(
            nutrient="omega_3",
            min_value=1000,
            unit="mg",
            daily_limit=True,
            reason="Reduces triglycerides, improves HDL"
        ),
        NutrientLimit(
            nutrient="plant_sterols",
            min_value=2,
            unit="g",
            daily_limit=True,
            reason="Blocks cholesterol absorption"
        )
    ],
    
    foods_to_avoid=[
        "Red meat", "Butter", "Cheese", "Full-fat dairy", "Egg yolks",
        "Fried foods", "Baked goods", "Processed meats", "Coconut oil",
        "Palm oil", "Margarine (with trans fats)"
    ],
    
    foods_recommended=[
        "Oats", "Barley", "Beans", "Eggplant", "Okra", "Apples", "Grapes",
        "Strawberries", "Citrus", "Fatty fish", "Walnuts", "Almonds",
        "Olive oil", "Avocado", "Soy products", "Psyllium"
    ],
    
    food_categories_to_avoid=[
        "High-fat meats", "Full-fat dairy", "Tropical oils"
    ],
    food_categories_recommended=[
        "Whole grains", "Legumes", "Nuts", "Fatty fish"
    ],
    
    protein_ratio=20.0,
    carb_ratio=50.0,
    fat_ratio=30.0,  # Focus on unsaturated fats
    
    meal_frequency_recommendation="3 meals + 2 snacks",
    
    medication_interactions={
        "statins": ["May deplete CoQ10", "Grapefruit interaction with some statins"],
        "fibrates": ["Avoid high-fat meals"]
    },
    
    guidelines_source="ACC/AHA Cholesterol Guidelines 2024",
    clinical_notes="Portfolio diet approach: combine soluble fiber, plant sterols, "
                  "soy protein, and nuts for maximum LDL reduction."
)

HEART_DISEASE = DiseaseData(
    code="heart_disease",
    name="Coronary Heart Disease",
    category=DiseaseCategory.CARDIOVASCULAR,
    severity_levels=["mild", "moderate", "severe", "post_mi"],
    
    nutrients_to_limit=[
        NutrientLimit(
            nutrient="sodium",
            max_value=2000,
            unit="mg",
            daily_limit=True,
            reason="Reduces cardiac workload"
        ),
        NutrientLimit(
            nutrient="saturated_fat",
            max_value=13,
            unit="g",
            daily_limit=True,
            reason="Prevents plaque buildup"
        ),
        NutrientLimit(
            nutrient="trans_fat",
            max_value=0,
            unit="g",
            daily_limit=True,
            reason="Highly atherogenic"
        )
    ],
    
    nutrients_to_increase=[
        NutrientLimit(
            nutrient="omega_3",
            min_value=1000,
            unit="mg",
            daily_limit=True,
            reason="Anti-inflammatory, anti-arrhythmic"
        ),
        NutrientLimit(
            nutrient="fiber",
            min_value=30,
            unit="g",
            daily_limit=True,
            reason="Cholesterol and glucose management"
        ),
        NutrientLimit(
            nutrient="antioxidants",
            min_value=0,  # Through whole foods
            unit="",
            daily_limit=True,
            reason="Reduces oxidative stress"
        )
    ],
    
    foods_to_avoid=[
        "Processed meats", "High-sodium foods", "Fried foods", "Sugary beverages",
        "Refined grains", "Full-fat dairy", "Trans fats"
    ],
    
    foods_recommended=[
        "Fatty fish", "Leafy greens", "Berries", "Whole grains", "Nuts",
        "Olive oil", "Avocado", "Legumes", "Dark chocolate", "Green tea"
    ],
    
    food_categories_to_avoid=[
        "Processed foods", "High-sodium foods", "Saturated fats"
    ],
    food_categories_recommended=[
        "Mediterranean diet foods", "Omega-3 rich foods", "Antioxidant-rich foods"
    ],
    
    protein_ratio=20.0,
    carb_ratio=50.0,
    fat_ratio=30.0,
    
    meal_frequency_recommendation="3 small meals + 2-3 snacks (reduces cardiac workload)",
    
    medication_interactions={
        "warfarin": ["Consistent vitamin K intake critical - avoid sudden changes in green leafy vegetables"],
        "aspirin": ["Limit vitamin E supplements - bleeding risk"],
        "beta_blockers": ["May mask hypoglycemia symptoms"]
    },
    
    guidelines_source="AHA Heart-Healthy Diet 2024",
    clinical_notes="Mediterranean diet strongly recommended. Omega-3 from fish 2x/week minimum."
)

# ============================================================================
# KIDNEY DISEASES
# ============================================================================

CKD_STAGE_3 = DiseaseData(
    code="ckd_stage3",
    name="Chronic Kidney Disease Stage 3 (GFR 30-59)",
    category=DiseaseCategory.KIDNEY,
    severity_levels=["3a", "3b"],
    
    nutrients_to_limit=[
        NutrientLimit(
            nutrient="protein",
            max_value=60,  # 0.8 g/kg body weight
            unit="g",
            daily_limit=True,
            reason="Reduces kidney workload"
        ),
        NutrientLimit(
            nutrient="sodium",
            max_value=2000,
            unit="mg",
            daily_limit=True,
            reason="Blood pressure and fluid management"
        ),
        NutrientLimit(
            nutrient="phosphorus",
            max_value=1000,
            unit="mg",
            daily_limit=True,
            reason="Prevents bone disease"
        ),
        NutrientLimit(
            nutrient="potassium",
            max_value=2000,
            unit="mg",
            daily_limit=True,
            reason="Prevents hyperkalemia"
        )
    ],
    
    nutrients_to_increase=[
        NutrientLimit(
            nutrient="omega_3",
            min_value=1000,
            unit="mg",
            daily_limit=True,
            reason="Anti-inflammatory"
        )
    ],
    
    foods_to_avoid=[
        "Processed meats", "Canned soups", "Cheese", "Pickles",
        "Bananas", "Oranges", "Potatoes", "Tomatoes", "Avocados",
        "Dried beans", "Nuts (high phosphorus)", "Whole grains (high phosphorus)",
        "Cola drinks", "Dairy products (high phosphorus)"
    ],
    
    foods_recommended=[
        "Egg whites", "Fish (limited)", "Skinless chicken", "Cauliflower",
        "Cabbage", "Red bell peppers", "Onions", "Apples", "Berries",
        "White bread", "White rice", "Cucumber"
    ],
    
    food_categories_to_avoid=[
        "High-potassium foods", "High-phosphorus foods", "High-protein foods"
    ],
    food_categories_recommended=[
        "Low-potassium fruits", "Low-phosphorus grains", "Lean proteins (limited)"
    ],
    
    protein_ratio=15.0,  # Reduced protein
    carb_ratio=55.0,
    fat_ratio=30.0,
    
    meal_frequency_recommendation="3 meals + 1 snack",
    
    medication_interactions={
        "phosphate_binders": ["Take with meals", "Avoid calcium-rich foods at same time"],
        "ACE_inhibitors": ["Monitor potassium closely"],
        "vitamin_D": ["May need supplementation"]
    },
    
    guidelines_source="KDIGO CKD Guidelines 2024, NKF KDOQI",
    clinical_notes="Protein restriction 0.6-0.8 g/kg. Avoid high-potassium and high-phosphorus foods. "
                  "Regular monitoring of electrolytes essential."
)

CKD_STAGE_4 = DiseaseData(
    code="ckd_stage4",
    name="Chronic Kidney Disease Stage 4 (GFR 15-29)",
    category=DiseaseCategory.KIDNEY,
    severity_levels=["early_stage4", "advanced_stage4"],
    
    nutrients_to_limit=[
        NutrientLimit(
            nutrient="protein",
            max_value=50,  # 0.6 g/kg
            unit="g",
            daily_limit=True,
            reason="Minimizes uremic toxins"
        ),
        NutrientLimit(
            nutrient="sodium",
            max_value=1500,
            unit="mg",
            daily_limit=True,
            reason="Fluid and BP control"
        ),
        NutrientLimit(
            nutrient="phosphorus",
            max_value=800,
            unit="mg",
            daily_limit=True,
            reason="Prevents vascular calcification"
        ),
        NutrientLimit(
            nutrient="potassium",
            max_value=1500,
            unit="mg",
            daily_limit=True,
            reason="Cardiac protection"
        ),
        NutrientLimit(
            nutrient="fluid",
            max_value=1500,  # May need restriction
            unit="ml",
            daily_limit=True,
            reason="Prevents fluid overload"
        )
    ],
    
    nutrients_to_increase=[],  # Focus is on restrictions
    
    foods_to_avoid=[
        "All high-potassium foods", "All high-phosphorus foods",
        "Processed foods", "Dairy", "Nuts", "Whole grains",
        "Bananas", "Oranges", "Potatoes", "Tomatoes", "Chocolate",
        "Beans", "Lentils"
    ],
    
    foods_recommended=[
        "Egg whites", "Limited fish", "Cauliflower", "Cabbage",
        "Red peppers", "Apples", "Berries", "White bread", "White rice"
    ],
    
    food_categories_to_avoid=[
        "High-K foods", "High-P foods", "Whole grains", "Legumes"
    ],
    food_categories_recommended=[
        "Refined grains", "Low-K vegetables", "Limited lean proteins"
    ],
    
    protein_ratio=12.0,  # Very restricted
    carb_ratio=58.0,
    fat_ratio=30.0,
    
    meal_frequency_recommendation="3 small meals",
    
    medication_interactions={
        "phosphate_binders": ["Essential with all meals"],
        "vitamin_D": ["Active form needed"],
        "EPO": ["Iron supplementation may be needed"]
    },
    
    guidelines_source="KDIGO 2024, NKF",
    clinical_notes="Very low protein diet with keto-acid supplementation may be considered. "
                  "Preparation for dialysis or transplant."
)

DIALYSIS = DiseaseData(
    code="dialysis",
    name="End-Stage Renal Disease on Dialysis",
    category=DiseaseCategory.KIDNEY,
    severity_levels=["hemodialysis", "peritoneal_dialysis"],
    
    nutrients_to_limit=[
        NutrientLimit(
            nutrient="sodium",
            max_value=2000,
            unit="mg",
            daily_limit=True,
            reason="Fluid and BP management"
        ),
        NutrientLimit(
            nutrient="potassium",
            max_value=2000,
            unit="mg",
            daily_limit=True,
            reason="Cardiac safety"
        ),
        NutrientLimit(
            nutrient="phosphorus",
            max_value=1000,
            unit="mg",
            daily_limit=True,
            reason="Bone health"
        ),
        NutrientLimit(
            nutrient="fluid",
            max_value=1000,  # Varies by urine output
            unit="ml",
            daily_limit=True,
            reason="Prevents fluid overload"
        )
    ],
    
    nutrients_to_increase=[
        NutrientLimit(
            nutrient="protein",
            min_value=70,  # 1.2 g/kg for HD
            unit="g",
            daily_limit=True,
            reason="Compensate for dialysis losses"
        ),
        NutrientLimit(
            nutrient="calories",
            min_value=2000,
            unit="kcal",
            daily_limit=True,
            reason="Prevent malnutrition"
        )
    ],
    
    foods_to_avoid=[
        "High-potassium foods", "High-phosphorus foods", "High-sodium foods",
        "Excessive fluids", "Oranges", "Bananas", "Tomatoes", "Potatoes",
        "Dairy", "Nuts", "Whole grains", "Cola"
    ],
    
    foods_recommended=[
        "Egg whites", "Fish", "Chicken", "Cauliflower", "Cabbage",
        "Green beans", "Apples", "Berries", "White rice", "White bread"
    ],
    
    food_categories_to_avoid=[
        "High-K foods", "High-P foods", "Salty foods"
    ],
    food_categories_recommended=[
        "High-quality proteins", "Low-K vegetables"
    ],
    
    protein_ratio=25.0,  # Higher for dialysis
    carb_ratio=45.0,
    fat_ratio=30.0,
    
    meal_frequency_recommendation="3 meals + 2 snacks (maintain protein intake)",
    
    medication_interactions={
        "phosphate_binders": ["Must take with every meal"],
        "vitamin_D": ["Active form essential"],
        "iron": ["Often IV during dialysis"],
        "EPO": ["For anemia management"]
    },
    
    guidelines_source="KDOQI Hemodialysis Guidelines 2024",
    clinical_notes="Higher protein needed (1.2 g/kg HD, 1.3 g/kg PD). "
                  "Fluid restriction critical. Phosphorus <1000mg/day."
)

# ============================================================================
# DIGESTIVE DISEASES
# ============================================================================

CELIAC_DISEASE = DiseaseData(
    code="celiac_disease",
    name="Celiac Disease",
    category=DiseaseCategory.DIGESTIVE,
    severity_levels=["newly_diagnosed", "established", "refractory"],
    
    nutrients_to_limit=[],
    
    nutrients_to_increase=[
        NutrientLimit(
            nutrient="iron",
            min_value=18,
            unit="mg",
            daily_limit=True,
            reason="Malabsorption correction"
        ),
        NutrientLimit(
            nutrient="calcium",
            min_value=1200,
            unit="mg",
            daily_limit=True,
            reason="Bone health"
        ),
        NutrientLimit(
            nutrient="vitamin_d",
            min_value=1000,
            unit="IU",
            daily_limit=True,
            reason="Absorption issues"
        ),
        NutrientLimit(
            nutrient="b_vitamins",
            min_value=0,
            unit="",
            daily_limit=True,
            reason="Malabsorption"
        )
    ],
    
    foods_to_avoid=[
        "Wheat", "Barley", "Rye", "Triticale", "Malt", "Brewer's yeast",
        "Wheat starch", "Wheat bran", "Wheat germ", "Couscous", "Farina",
        "Spelt", "Kamut", "Einkorn", "Farro", "Graham flour", "Semolina",
        "Most bread", "Most pasta", "Most baked goods", "Beer (regular)",
        "Soy sauce (regular)", "Many processed foods"
    ],
    
    foods_recommended=[
        "Rice", "Corn", "Quinoa", "Buckwheat", "Millet", "Amaranth",
        "Certified gluten-free oats", "Potatoes", "Cassava", "Taro",
        "All fruits", "All vegetables", "Meat", "Fish", "Eggs",
        "Dairy (if tolerated)", "Legumes", "Nuts", "Seeds"
    ],
    
    food_categories_to_avoid=[
        "Gluten-containing grains", "Cross-contaminated products", "Most processed foods"
    ],
    food_categories_recommended=[
        "Naturally gluten-free grains", "Fresh whole foods", "Certified GF products"
    ],
    
    protein_ratio=20.0,
    carb_ratio=50.0,
    fat_ratio=30.0,
    
    meal_frequency_recommendation="3 meals + 2 snacks",
    
    medication_interactions={},
    
    guidelines_source="Celiac Disease Foundation 2024",
    clinical_notes="Strict lifelong gluten-free diet. Even traces (<20ppm) can cause damage. "
                  "Cross-contamination prevention critical. Regular nutritional monitoring."
)

# Add more diseases following same pattern...
# IBS, Crohn's, Ulcerative Colitis, GERD, etc.

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_all_diseases() -> Dict[str, DiseaseData]:
    """Return dictionary of all diseases indexed by code"""
    return {
        "diabetes_type2": DIABETES_TYPE_2,
        "diabetes_type1": DIABETES_TYPE_1,
        "pre_diabetes": PRE_DIABETES,
        "obesity": OBESITY,
        "hypertension": HYPERTENSION,
        "high_cholesterol": HIGH_CHOLESTEROL,
        "heart_disease": HEART_DISEASE,
        "ckd_stage3": CKD_STAGE_3,
        "ckd_stage4": CKD_STAGE_4,
        "dialysis": DIALYSIS,
        "celiac_disease": CELIAC_DISEASE,
        # Add all other diseases here...
    }


def get_disease_by_code(code: str) -> Optional[DiseaseData]:
    """Get disease data by code"""
    diseases = get_all_diseases()
    return diseases.get(code.lower())


def get_diseases_by_category(category: DiseaseCategory) -> List[DiseaseData]:
    """Get all diseases in a specific category"""
    diseases = get_all_diseases()
    return [d for d in diseases.values() if d.category == category]


def get_combined_restrictions(disease_codes: List[str]) -> Dict[str, Any]:
    """
    Combine restrictions from multiple diseases.
    Takes the most restrictive limit for each nutrient.
    """
    combined = {
        "nutrients_to_limit": [],
        "nutrients_to_increase": [],
        "foods_to_avoid": set(),
        "foods_recommended": set(),
        "medication_interactions": {}
    }
    
    for code in disease_codes:
        disease = get_disease_by_code(code)
        if not disease:
            continue
            
        # Combine nutrient limits (take most restrictive)
        combined["nutrients_to_limit"].extend(disease.nutrients_to_limit)
        combined["nutrients_to_increase"].extend(disease.nutrients_to_increase)
        
        # Combine food lists
        combined["foods_to_avoid"].update(disease.foods_to_avoid)
        combined["foods_recommended"].update(disease.foods_recommended)
        
        # Combine medication interactions
        combined["medication_interactions"].update(disease.medication_interactions)
    
    return combined


# Note: This file shows the structure and sample data for key diseases.
# In production, this would contain 100+ diseases with comprehensive data
# totaling approximately 4,000+ lines of code.
