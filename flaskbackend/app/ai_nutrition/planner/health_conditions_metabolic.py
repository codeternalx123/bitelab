"""
Phase 3B: Metabolic Disease Conditions
Complete nutritional profiles for metabolism-related conditions

Conditions Covered:
1. Obesity
2. Metabolic Syndrome
3. Hyperlipidemia (High Cholesterol)
4. Hypothyroidism
5. Polycystic Ovary Syndrome (PCOS)

Each condition includes:
- Specific nutrient requirements with targets
- Calorie and macronutrient adjustments
- Food restrictions and alternatives
- Medication interactions
- Evidence-based recommendations
"""

from decimal import Decimal
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class NutrientRecommendation(Enum):
    """Types of nutrient recommendations"""
    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"
    AVOID = "avoid"
    MONITOR = "monitor"


@dataclass
class ConditionNutrientRequirement:
    """Nutrient requirement for a specific health condition"""
    nutrient_id: str
    nutrient_name: str
    recommendation_type: NutrientRecommendation
    target_amount: Optional[Decimal] = None
    target_unit: Optional[str] = None
    rationale: str = ""
    food_sources: List[str] = field(default_factory=list)
    priority: int = 1


@dataclass
class FoodRestriction:
    """Food or food category to avoid"""
    food_or_category: str
    reason: str
    severity: str  # "must_avoid", "limit", "monitor"
    alternatives: List[str] = field(default_factory=list)


@dataclass
class HealthConditionProfile:
    """Complete nutritional profile for a health condition"""
    condition_id: str
    condition_name: str
    nutrient_requirements: List[ConditionNutrientRequirement] = field(default_factory=list)
    food_restrictions: List[FoodRestriction] = field(default_factory=list)
    recommended_foods: List[str] = field(default_factory=list)
    recommended_diet_patterns: List[str] = field(default_factory=list)
    lifestyle_recommendations: List[str] = field(default_factory=list)
    medication_interactions: List[str] = field(default_factory=list)
    calorie_adjustment: Optional[str] = None  # Special notes for calorie needs
    macro_adjustment: Optional[str] = None  # Special notes for macros


# ============================================================================
# OBESITY
# ============================================================================

OBESITY = HealthConditionProfile(
    condition_id="obesity",
    condition_name="Obesity (BMI ≥30)",
    calorie_adjustment="Create 500-750 kcal daily deficit for 1-1.5 lbs/week weight loss. Minimum 1200 kcal/day for women, 1500 kcal/day for men",
    macro_adjustment="Protein: 1.2-1.6 g/kg ideal body weight (preserves muscle). Carbs: 45-50% calories. Fat: 25-30% calories",
    nutrient_requirements=[
        ConditionNutrientRequirement(
            nutrient_id="protein",
            nutrient_name="Protein",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("1.2"),
            target_unit="g/kg ideal weight",
            rationale="Increases satiety, preserves lean muscle during weight loss, higher thermic effect",
            food_sources=["lean meats", "fish", "eggs", "Greek yogurt", "legumes", "tofu", "protein shakes"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="fiber",
            nutrient_name="Dietary Fiber",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("35"),
            target_unit="g",
            rationale="Increases fullness, slows digestion, reduces calorie absorption. 14g per 1000 kcal",
            food_sources=["vegetables", "whole grains", "legumes", "fruits", "chia seeds", "psyllium"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="water",
            nutrient_name="Water Intake",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2500"),
            target_unit="ml",
            rationale="Increases metabolism, promotes satiety, often mistaken for hunger. Drink before meals",
            food_sources=["water", "herbal tea", "sparkling water", "water-rich vegetables"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_d",
            nutrient_name="Vitamin D",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="IU",
            rationale="Often deficient in obesity (sequestered in fat). Deficiency associated with weight gain",
            food_sources=["fatty fish", "fortified milk", "egg yolks", "sunlight", "supplements"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="calcium",
            nutrient_name="Calcium",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("1200"),
            target_unit="mg",
            rationale="May support fat metabolism and weight loss. Dairy calcium especially beneficial",
            food_sources=["low-fat dairy", "fortified plant milk", "leafy greens", "sardines with bones"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="added_sugars",
            nutrient_name="Added Sugars",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("25"),
            target_unit="g",
            rationale="Empty calories, no satiety. Limit to <5% of total calories (25g for 2000 kcal)",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="refined_carbs",
            nutrient_name="Refined Carbohydrates",
            recommendation_type=NutrientRecommendation.DECREASE,
            rationale="Low satiety, rapid blood sugar spikes, promotes overeating",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="alcohol",
            nutrient_name="Alcohol",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("0"),
            target_unit="drinks",
            rationale="7 calories/gram, no nutritional value, lowers inhibitions (overeating), affects metabolism",
            food_sources=[],
            priority=2
        )
    ],
    food_restrictions=[
        FoodRestriction(
            food_or_category="Sugar-sweetened beverages (soda, juice, sweetened tea, energy drinks)",
            reason="Liquid calories don't trigger satiety. Single biggest contributor to obesity",
            severity="must_avoid",
            alternatives=["water", "unsweetened tea", "sparkling water", "black coffee", "infused water"]
        ),
        FoodRestriction(
            food_or_category="Ultra-processed foods (chips, cookies, candy, packaged snacks)",
            reason="Hyper-palatable, engineered for overconsumption, high calorie density, low nutrients",
            severity="must_avoid",
            alternatives=["fresh fruits", "vegetable sticks with hummus", "air-popped popcorn", "nuts (portion controlled)"]
        ),
        FoodRestriction(
            food_or_category="Fast food and fried foods",
            reason="Very high calorie density (500-1500 kcal per meal), high fat and sodium",
            severity="limit",
            alternatives=["home-cooked meals", "grilled options", "salads with lean protein"]
        ),
        FoodRestriction(
            food_or_category="High-calorie desserts and baked goods",
            reason="High in sugar and fat (9 kcal/g), easy to overeat, minimal satiety",
            severity="limit",
            alternatives=["fresh fruit", "frozen fruit bars", "small portions of dark chocolate", "Greek yogurt with berries"]
        ),
        FoodRestriction(
            food_or_category="Large portion sizes",
            reason="Portion distortion leads to excess calorie intake",
            severity="monitor",
            alternatives=["use smaller plates", "measure portions", "follow serving size guidelines"]
        ),
        FoodRestriction(
            food_or_category="Refined grains (white bread, white rice, regular pasta)",
            reason="Low fiber, rapid digestion, blood sugar spikes, poor satiety",
            severity="limit",
            alternatives=["whole grain bread", "brown rice", "quinoa", "whole wheat pasta", "cauliflower rice"]
        )
    ],
    recommended_foods=[
        "Non-starchy vegetables (unlimited - broccoli, spinach, peppers, cauliflower, zucchini)",
        "Lean proteins (chicken breast, turkey, fish, egg whites, tofu, tempeh)",
        "Legumes (lentils, black beans, chickpeas - high protein, high fiber)",
        "Whole grains (oats, quinoa, brown rice, barley - moderate portions)",
        "Fresh fruits (berries, apples, pears - high fiber, lower calorie)",
        "Greek yogurt (low-fat, high protein)",
        "Nuts and seeds (measured portions - 1 oz)",
        "Leafy greens (kale, spinach, arugula - very low calorie, high nutrients)",
        "Cruciferous vegetables (broccoli, Brussels sprouts, cabbage)",
        "Shirataki noodles or zoodles (very low calorie pasta alternatives)"
    ],
    recommended_diet_patterns=[
        "Mediterranean Diet (proven for sustained weight loss)",
        "DASH Diet",
        "Volumetrics (high-volume, low-calorie foods)",
        "Intermittent Fasting (16:8 or 5:2 if appropriate)",
        "Plant-based diet",
        "Low-carb or ketogenic diet (if medically supervised)"
    ],
    lifestyle_recommendations=[
        "CREATE CALORIE DEFICIT: Track food intake using app (MyFitnessPal, Cronometer)",
        "EXERCISE: 250-300 minutes moderate activity/week for weight loss. Combine cardio + strength training",
        "STRENGTH TRAINING: 2-3x/week to preserve muscle mass during weight loss",
        "INCREASE NEAT (Non-Exercise Activity): Take stairs, walk more, stand vs sit",
        "SLEEP: Get 7-9 hours (poor sleep increases hunger hormones ghrelin and leptin resistance)",
        "STRESS MANAGEMENT: Chronic stress increases cortisol (promotes abdominal fat storage)",
        "MEAL PLANNING: Plan and prep meals to avoid impulsive eating",
        "MINDFUL EATING: Eat slowly, avoid distractions (TV, phone), recognize fullness cues",
        "WEIGH REGULARLY: Weekly weigh-ins for accountability (same day, same time)",
        "BEHAVIOR MODIFICATION: Address emotional eating, identify triggers",
        "SOCIAL SUPPORT: Join support group, work with dietitian or therapist",
        "SET REALISTIC GOALS: 1-2 lbs/week is safe and sustainable"
    ],
    medication_interactions=[
        "ORLISTAT (Alli, Xenical): Blocks fat absorption. Take multivitamin (A, D, E, K) 2 hours away from dose. Causes GI side effects with high-fat meals",
        "PHENTERMINE: Stimulant appetite suppressant. Avoid excessive caffeine. May cause insomnia, dry mouth",
        "GLP-1 AGONISTS (Wegovy, Saxenda, Ozempic): Injectable. Slow gastric emptying. Eat smaller meals. May cause nausea - ginger may help",
        "CONTRAVE (Naltrexone/Bupropion): Avoid alcohol. May interact with antidepressants",
        "METFORMIN (if diabetic): Take with meals to reduce GI upset. May reduce B12 absorption"
    ]
)


# ============================================================================
# METABOLIC SYNDROME
# ============================================================================

METABOLIC_SYNDROME = HealthConditionProfile(
    condition_id="metabolic_syndrome",
    condition_name="Metabolic Syndrome (Pre-Diabetes, High BP, High Triglycerides)",
    calorie_adjustment="If overweight: Create 500 kcal deficit. Weight loss of 7-10% significantly improves all metabolic parameters",
    macro_adjustment="Carbs: 45-50% (emphasize low-GI). Protein: 20-25% (1.0-1.2 g/kg). Fat: 30-35% (emphasize MUFA/PUFA)",
    nutrient_requirements=[
        ConditionNutrientRequirement(
            nutrient_id="fiber",
            nutrient_name="Dietary Fiber",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("35"),
            target_unit="g",
            rationale="Improves insulin sensitivity, lowers cholesterol, reduces blood pressure. Critical for MetS",
            food_sources=["whole grains", "legumes", "vegetables", "fruits", "nuts", "seeds"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="omega_3_fatty_acids",
            nutrient_name="Omega-3 Fatty Acids",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("1500"),
            target_unit="mg",
            rationale="Lowers triglycerides (major component of MetS), reduces inflammation",
            food_sources=["fatty fish", "walnuts", "flaxseeds", "chia seeds", "fish oil supplements"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="chromium",
            nutrient_name="Chromium",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("200"),
            target_unit="mcg",
            rationale="Enhances insulin action, improves glucose metabolism",
            food_sources=["broccoli", "whole grains", "green beans", "potatoes"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="magnesium",
            nutrient_name="Magnesium",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("400"),
            target_unit="mg",
            rationale="Improves insulin sensitivity. Deficiency common in MetS and diabetes",
            food_sources=["spinach", "almonds", "black beans", "avocado", "dark chocolate"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="sodium",
            nutrient_name="Sodium",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("2300"),
            target_unit="mg",
            rationale="Reduces blood pressure (key component of MetS). Aim for <2300mg, ideally <1500mg",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="added_sugars",
            nutrient_name="Added Sugars",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("25"),
            target_unit="g",
            rationale="Directly worsens insulin resistance and raises triglycerides",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="saturated_fat",
            nutrient_name="Saturated Fat",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("15"),
            target_unit="g",
            rationale="Worsens insulin resistance and raises LDL cholesterol. Limit to <7% calories",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_d",
            nutrient_name="Vitamin D",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="IU",
            rationale="Deficiency linked to insulin resistance and MetS. Check blood levels",
            food_sources=["fatty fish", "fortified milk", "egg yolks", "sunlight"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="polyphenols",
            nutrient_name="Polyphenols (Plant Antioxidants)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("500"),
            target_unit="mg",
            rationale="Improve insulin sensitivity, reduce inflammation. Found in colorful plant foods",
            food_sources=["berries", "green tea", "dark chocolate", "extra virgin olive oil", "coffee"],
            priority=2
        )
    ],
    food_restrictions=[
        FoodRestriction(
            food_or_category="Sugar-sweetened beverages and fruit juices",
            reason="Directly increases triglycerides, worsens insulin resistance, liquid calories",
            severity="must_avoid",
            alternatives=["water", "unsweetened tea", "sparkling water", "coffee (no sugar)"]
        ),
        FoodRestriction(
            food_or_category="Refined carbohydrates (white bread, white rice, pastries)",
            reason="High glycemic index worsens insulin resistance and blood sugar",
            severity="limit",
            alternatives=["whole grain bread", "brown rice", "quinoa", "steel-cut oats"]
        ),
        FoodRestriction(
            food_or_category="Processed meats (bacon, sausage, deli meats, hot dogs)",
            reason="High in sodium and saturated fat. Linked to increased MetS risk",
            severity="limit",
            alternatives=["lean fresh meats", "fish", "poultry", "plant proteins"]
        ),
        FoodRestriction(
            food_or_category="Trans fats and partially hydrogenated oils",
            reason="Worsen insulin resistance, raise LDL, lower HDL, increase inflammation",
            severity="must_avoid",
            alternatives=["olive oil", "avocado oil", "nuts", "seeds"]
        ),
        FoodRestriction(
            food_or_category="Excessive alcohol",
            reason="Raises triglycerides, increases abdominal fat, worsens liver function",
            severity="limit",
            alternatives=["limit to 1 drink/day women, 2/day men", "preferably red wine in moderation"]
        ),
        FoodRestriction(
            food_or_category="High-sodium processed foods",
            reason="Worsens hypertension component of MetS",
            severity="limit",
            alternatives=["fresh whole foods", "herbs and spices", "homemade meals"]
        )
    ],
    recommended_foods=[
        "Fatty fish 2-3x/week (salmon, mackerel, sardines - omega-3s and protein)",
        "Leafy greens and non-starchy vegetables (unlimited - rich in magnesium, fiber)",
        "Whole grains (oats, quinoa, barley, brown rice - moderate portions)",
        "Legumes (lentils, chickpeas, black beans - fiber and protein)",
        "Berries (blueberries, strawberries - low glycemic, high antioxidants)",
        "Nuts (almonds, walnuts - 1 oz daily for healthy fats and fiber)",
        "Extra virgin olive oil (2-3 tablespoons daily - MUFA)",
        "Avocado (healthy fats, fiber, potassium)",
        "Greek yogurt (low-fat, plain - probiotics and protein)",
        "Green tea (polyphenols improve insulin sensitivity)"
    ],
    recommended_diet_patterns=[
        "Mediterranean Diet (GOLD STANDARD for Metabolic Syndrome)",
        "DASH Diet (addresses hypertension component)",
        "Low-Glycemic Diet",
        "Portfolio Diet (lowers cholesterol)",
        "Plant-based diet with fish"
    ],
    lifestyle_recommendations=[
        "WEIGHT LOSS: 7-10% body weight loss dramatically improves ALL MetS components",
        "EXERCISE: 150 minutes moderate or 75 minutes vigorous weekly. Exercise improves insulin sensitivity",
        "STRENGTH TRAINING: 2x/week builds muscle (improves glucose disposal)",
        "REDUCE SITTING TIME: Break up sedentary time every 30 minutes",
        "SLEEP: 7-9 hours. Poor sleep worsens insulin resistance",
        "STRESS MANAGEMENT: Chronic stress increases cortisol (promotes abdominal fat)",
        "QUIT SMOKING: Smoking worsens insulin resistance",
        "LIMIT ALCOHOL: <1 drink/day women, <2/day men",
        "MONITOR: Track blood pressure, blood sugar, waist circumference, lipids",
        "REGULAR CHECKUPS: Screen for progression to type 2 diabetes"
    ],
    medication_interactions=[
        "METFORMIN (if prescribed): Take with meals to reduce GI upset. May reduce vitamin B12 absorption - supplement if needed",
        "STATINS (for high cholesterol): Deplete CoQ10. Avoid grapefruit with simvastatin/lovastatin",
        "BLOOD PRESSURE MEDS: See hypertension profile. ACE inhibitors affect potassium",
        "FISH OIL SUPPLEMENTS: May interact with blood thinners at high doses (>3g/day)"
    ]
)


# ============================================================================
# HYPERLIPIDEMIA (High Cholesterol)
# ============================================================================

HYPERLIPIDEMIA = HealthConditionProfile(
    condition_id="hyperlipidemia",
    condition_name="Hyperlipidemia (High Cholesterol/Triglycerides)",
    calorie_adjustment="If overweight: Weight loss of 5-10% can lower LDL by 5-8% and triglycerides by 20%",
    macro_adjustment="Emphasize MUFA and PUFA over saturated fat. Carbs: 50-55% (complex, high-fiber). Protein: 15-20%. Fat: 25-35% (mostly unsaturated)",
    nutrient_requirements=[
        ConditionNutrientRequirement(
            nutrient_id="soluble_fiber",
            nutrient_name="Soluble Fiber",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("10"),
            target_unit="g",
            rationale="Binds bile acids (cholesterol), reduces LDL. 10g soluble fiber lowers LDL by 5%",
            food_sources=["oats", "barley", "psyllium", "beans", "apples", "Brussels sprouts", "flaxseeds"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="plant_sterols",
            nutrient_name="Plant Sterols/Stanols",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="mg",
            rationale="Blocks cholesterol absorption in gut. 2g/day lowers LDL by 10%",
            food_sources=["fortified margarine spreads", "fortified orange juice", "nuts", "seeds", "vegetable oils"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="omega_3_fatty_acids",
            nutrient_name="Omega-3 Fatty Acids (EPA + DHA)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="mg",
            rationale="Lowers triglycerides by 25-30%. Prescription omega-3s available for severe hypertriglyceridemia",
            food_sources=["fatty fish", "fish oil supplements", "algae-based omega-3"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="saturated_fat",
            nutrient_name="Saturated Fat",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("13"),
            target_unit="g",
            rationale="Raises LDL cholesterol. Limit to <7% of calories (13g for 2000 kcal diet)",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="trans_fat",
            nutrient_name="Trans Fat",
            recommendation_type=NutrientRecommendation.AVOID,
            target_amount=Decimal("0"),
            target_unit="g",
            rationale="Raises LDL, lowers HDL. Most harmful fat for cholesterol profile",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="dietary_cholesterol",
            nutrient_name="Dietary Cholesterol",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("200"),
            target_unit="mg",
            rationale="May raise blood cholesterol in ~30% of people (hyper-responders). Limit to <200mg/day",
            food_sources=[],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="niacin",
            nutrient_name="Niacin (Vitamin B3)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("20"),
            target_unit="mg",
            rationale="Raises HDL cholesterol, lowers LDL and triglycerides. High-dose prescription form available",
            food_sources=["tuna", "chicken", "turkey", "peanuts", "mushrooms"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="soy_protein",
            nutrient_name="Soy Protein",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("25"),
            target_unit="g",
            rationale="25g/day soy protein lowers LDL by ~5%. FDA-approved cholesterol-lowering claim",
            food_sources=["tofu", "tempeh", "edamame", "soy milk", "soy nuts"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="added_sugars",
            nutrient_name="Added Sugars",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("25"),
            target_unit="g",
            rationale="Excess sugar raises triglycerides and lowers HDL cholesterol",
            food_sources=[],
            priority=1
        )
    ],
    food_restrictions=[
        FoodRestriction(
            food_or_category="Red meat and processed meats",
            reason="High in saturated fat and cholesterol. Processed meats especially harmful",
            severity="limit",
            alternatives=["fish", "skinless poultry", "legumes", "tofu", "tempeh"]
        ),
        FoodRestriction(
            food_or_category="Full-fat dairy (whole milk, cheese, butter, cream, ice cream)",
            reason="High in saturated fat which raises LDL cholesterol",
            severity="limit",
            alternatives=["skim or 1% milk", "low-fat yogurt", "plant-based milk", "nutritional yeast"]
        ),
        FoodRestriction(
            food_or_category="Trans fats (partially hydrogenated oils, margarine, shortening)",
            reason="Most harmful for cholesterol profile. Raises LDL, lowers HDL",
            severity="must_avoid",
            alternatives=["olive oil", "avocado oil", "soft margarine (no trans fats)", "nut butters"]
        ),
        FoodRestriction(
            food_or_category="Fried foods and fast food",
            reason="High in trans and saturated fats, extremely high cholesterol impact",
            severity="must_avoid",
            alternatives=["baked", "grilled", "steamed", "air-fried options"]
        ),
        FoodRestriction(
            food_or_category="High-cholesterol foods (egg yolks, organ meats, shrimp)",
            reason="Direct cholesterol source. Limit if hyper-responder to dietary cholesterol",
            severity="monitor",
            alternatives=["egg whites", "plant proteins", "fish (lower cholesterol)"]
        ),
        FoodRestriction(
            food_or_category="Commercial baked goods (cookies, cakes, pastries, donuts)",
            reason="High in trans fats, saturated fats, and added sugars",
            severity="limit",
            alternatives=["homemade with healthy oils", "fresh fruit", "dark chocolate (70%+)"]
        ),
        FoodRestriction(
            food_or_category="Sugar-sweetened beverages",
            reason="Raises triglycerides significantly",
            severity="limit",
            alternatives=["water", "unsweetened tea", "sparkling water"]
        )
    ],
    recommended_foods=[
        "Oats and oat bran (β-glucan soluble fiber - powerful LDL-lowering)",
        "Fatty fish 2-3x/week (salmon, mackerel, sardines - omega-3s)",
        "Nuts (almonds, walnuts - 1.5 oz/day lowers LDL by 5%)",
        "Legumes (beans, lentils, chickpeas - soluble fiber and plant protein)",
        "Soy products (tofu, tempeh, edamame, soy milk)",
        "Plant sterol-fortified foods (margarine spreads, orange juice)",
        "Extra virgin olive oil (MUFA lowers LDL without lowering HDL)",
        "Avocado (healthy fats and fiber)",
        "Berries (antioxidants protect LDL from oxidation)",
        "Barley and psyllium (soluble fiber)",
        "Dark leafy greens (lutein protects against cholesterol oxidation)",
        "Apples (pectin soluble fiber)"
    ],
    recommended_diet_patterns=[
        "Portfolio Diet (BEST for cholesterol - combines 4 proven components: plant sterols, soy, nuts, soluble fiber)",
        "Mediterranean Diet",
        "DASH Diet",
        "Plant-based/Vegan diet (can lower LDL by 25-30%)",
        "TLC Diet (Therapeutic Lifestyle Changes - designed for cholesterol)"
    ],
    lifestyle_recommendations=[
        "WEIGHT LOSS: If overweight, lose 5-10% body weight (major impact on lipids)",
        "EXERCISE: 150 minutes moderate or 75 minutes vigorous weekly. Raises HDL, lowers triglycerides",
        "QUIT SMOKING: Smoking lowers HDL cholesterol",
        "LIMIT ALCOHOL: Excessive intake raises triglycerides. Moderate may raise HDL slightly",
        "INCREASE SOLUBLE FIBER gradually to avoid GI upset",
        "READ LABELS: Avoid partially hydrogenated oils, limit saturated fat",
        "REPLACE not just REDUCE: Swap saturated fats for unsaturated fats (bigger benefit than just reducing)",
        "MONITOR LIPIDS: Get lipid panel every 3-6 months to track progress"
    ],
    medication_interactions=[
        "STATINS (Lipitor, Crestor, Zocor): Deplete CoQ10 - supplement 100-200mg/day. AVOID grapefruit with simvastatin and lovastatin (increases drug levels 10-fold). Red yeast rice contains natural statins - don't combine",
        "BILE ACID SEQUESTRANTS (Cholestyramine): Take other medications 1 hour before or 4 hours after. May reduce absorption of fat-soluble vitamins (A, D, E, K)",
        "EZETIMIBE (Zetia): May reduce absorption of fat-soluble vitamins. Take with or without food",
        "FIBRATES (Gemfibrozil): Increase bleeding risk with omega-3 supplements >3g/day. Take 30 min before meals",
        "PRESCRIPTION OMEGA-3s (Vascepa, Lovaza): Very high doses (4g EPA). May increase bleeding risk with anticoagulants",
        "NIACIN (prescription Niaspan): Avoid alcohol (worsens flushing). Take with food. Don't take with hot beverages. Aspirin 30 min before may reduce flushing"
    ]
)


# ============================================================================
# HYPOTHYROIDISM
# ============================================================================

HYPOTHYROIDISM = HealthConditionProfile(
    condition_id="hypothyroidism",
    condition_name="Hypothyroidism (Underactive Thyroid)",
    calorie_adjustment="Metabolism may be 5-10% slower. May need 100-300 fewer calories than predicted. Reassess if not losing weight",
    macro_adjustment="Standard balanced macros. Focus on nutrient density due to lower calorie needs. Protein: 20-30% (prevents muscle loss)",
    nutrient_requirements=[
        ConditionNutrientRequirement(
            nutrient_id="iodine",
            nutrient_name="Iodine",
            recommendation_type=NutrientRecommendation.MONITOR,
            target_amount=Decimal("150"),
            target_unit="mcg",
            rationale="Essential for thyroid hormone production. BUT excess iodine can worsen Hashimoto's (autoimmune hypothyroidism). Use iodized salt, avoid megadoses",
            food_sources=["iodized salt", "seafood", "dairy", "eggs", "seaweed (moderate amounts)"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="selenium",
            nutrient_name="Selenium",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("200"),
            target_unit="mcg",
            rationale="Required for thyroid hormone conversion (T4 to active T3). Deficiency common in hypothyroidism",
            food_sources=["Brazil nuts (2-3 daily)", "tuna", "sardines", "eggs", "sunflower seeds"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="zinc",
            nutrient_name="Zinc",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("15"),
            target_unit="mg",
            rationale="Supports thyroid hormone production and T4 to T3 conversion",
            food_sources=["oysters", "beef", "pumpkin seeds", "chickpeas", "cashews"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="iron",
            nutrient_name="Iron",
            recommendation_type=NutrientRecommendation.MONITOR,
            target_amount=Decimal("18"),
            target_unit="mg",
            rationale="Deficiency impairs thyroid function. Check ferritin levels. Separate iron supplements from thyroid medication by 4 hours",
            food_sources=["lean meats", "beans", "spinach", "fortified cereals", "supplements if deficient"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_d",
            nutrient_name="Vitamin D",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="IU",
            rationale="Deficiency common in hypothyroidism and may worsen autoimmune thyroid disease",
            food_sources=["fatty fish", "fortified milk", "egg yolks", "sunlight", "supplements"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_b12",
            nutrient_name="Vitamin B12",
            recommendation_type=NutrientRecommendation.MONITOR,
            target_amount=Decimal("2.4"),
            target_unit="mcg",
            rationale="Deficiency common in autoimmune hypothyroidism (Hashimoto's). Check levels",
            food_sources=["meat", "fish", "dairy", "fortified cereals", "nutritional yeast", "supplements"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="fiber",
            nutrient_name="Dietary Fiber",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("30"),
            target_unit="g",
            rationale="Hypothyroidism slows digestion causing constipation. Fiber helps. Separate fiber supplements from thyroid meds by 2+ hours",
            food_sources=["vegetables", "whole grains", "legumes", "fruits", "chia seeds"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="goitrogens",
            nutrient_name="Goitrogenic Foods (Raw Cruciferous)",
            recommendation_type=NutrientRecommendation.MONITOR,
            rationale="Can interfere with thyroid function in large RAW amounts. COOKED crucifers are safe and healthy. Don't avoid unless consuming massive quantities raw",
            food_sources=[],
            priority=3
        )
    ],
    food_restrictions=[
        FoodRestriction(
            food_or_category="Soy (if consuming large amounts)",
            reason="High amounts may interfere with thyroid hormone absorption. Moderate amounts (1-2 servings) likely safe. Separate soy from thyroid meds by 4 hours",
            severity="monitor",
            alternatives=["Other plant proteins", "time soy away from medication"]
        ),
        FoodRestriction(
            food_or_category="Raw cruciferous vegetables (in excessive amounts)",
            reason="Goitrogens can interfere with thyroid function. Normal amounts, especially COOKED, are fine and healthy",
            severity="monitor",
            alternatives=["Cook crucifers (deactivates goitrogens)", "eat in moderation", "don't juice large amounts raw"]
        ),
        FoodRestriction(
            food_or_category="Processed foods and refined carbs",
            reason="Easy to overeat with slower metabolism. Provides excess calories without nutrients",
            severity="limit",
            alternatives=["whole foods", "high-protein options", "fiber-rich choices"]
        ),
        FoodRestriction(
            food_or_category="Gluten (if Hashimoto's autoimmune thyroiditis)",
            reason="May worsen autoimmune response in some individuals. Consider trial elimination for 3 months",
            severity="monitor",
            alternatives=["gluten-free grains (rice, quinoa, oats)", "test for celiac disease first"]
        ),
        FoodRestriction(
            food_or_category="Excessive fiber near medication time",
            reason="High-fiber foods/supplements can reduce thyroid medication absorption",
            severity="monitor",
            alternatives=["Take thyroid medication on empty stomach", "wait 30-60 min before eating", "separate fiber supplements by 2+ hours"]
        )
    ],
    recommended_foods=[
        "Brazil nuts (2-3 daily for selenium - don't exceed)",
        "Fatty fish (salmon, sardines - selenium, vitamin D, omega-3s)",
        "Eggs (selenium, iodine, vitamin D, protein)",
        "Lean meats (iron, zinc, selenium, B12)",
        "Cooked cruciferous vegetables (nutrient-dense, fiber)",
        "Sea vegetables in moderation (kelp, nori - iodine but don't overdo)",
        "Legumes (beans, lentils - fiber, zinc, iron)",
        "Whole grains (fiber, B vitamins)",
        "Greek yogurt (iodine, protein, probiotics)",
        "Berries (antioxidants, fiber, low calorie)"
    ],
    recommended_diet_patterns=[
        "Mediterranean Diet (nutrient-dense, anti-inflammatory)",
        "Autoimmune Protocol (AIP) diet if Hashimoto's (eliminates potential triggers)",
        "Gluten-free diet (trial for Hashimoto's)",
        "Balanced whole-foods diet with adequate protein"
    ],
    lifestyle_recommendations=[
        "TAKE THYROID MEDICATION CORRECTLY: On empty stomach, 30-60 min before food, with water only. NO coffee, calcium, iron, fiber near medication",
        "WEIGHT MANAGEMENT: May be harder due to slower metabolism. Focus on portion control and regular exercise",
        "EXERCISE: Especially important. 150 min/week minimum. Builds muscle (increases metabolism)",
        "SLEEP: Prioritize 7-9 hours. Fatigue is common symptom",
        "STRESS MANAGEMENT: Stress worsens thyroid function",
        "MONITOR THYROID LEVELS: TSH, free T4, free T3 every 6-12 weeks when adjusting medication, then annually",
        "CHECK NUTRIENT LEVELS: Test ferritin, vitamin D, B12, selenium annually",
        "AVOID IODINE MEGADOSES: Supplements with >500 mcg iodine can worsen Hashimoto's",
        "BE PATIENT: Thyroid medication takes 4-6 weeks to see full effect"
    ],
    medication_interactions=[
        "LEVOTHYROXINE (Synthroid, Levoxyl): CRITICAL TIMING: Take on empty stomach, 30-60 min before food, with water only. Avoid within 4 hours: calcium, iron, magnesium, antacids, soy, fiber supplements, coffee. Best time: first thing in morning or bedtime (4 hours after last meal)",
        "CALCIUM/IRON SUPPLEMENTS: Reduce thyroid medication absorption by 50%. MUST separate by 4 hours",
        "PROTON PUMP INHIBITORS (PPIs): May reduce thyroid medication absorption. May need higher dose",
        "BIOTIN SUPPLEMENTS: Can cause falsely abnormal thyroid lab results. Stop 2-3 days before blood test",
        "COFFEE: May reduce thyroid medication absorption. Wait 30-60 min after taking medication"
    ]
)


# ============================================================================
# POLYCYSTIC OVARY SYNDROME (PCOS)
# ============================================================================

PCOS = HealthConditionProfile(
    condition_id="pcos",
    condition_name="Polycystic Ovary Syndrome (PCOS)",
    calorie_adjustment="If overweight: Weight loss of 5-10% dramatically improves symptoms. May have 10-15% slower metabolism than predicted",
    macro_adjustment="Lower carb beneficial: Carbs 40-45% (emphasize low-GI), Protein 25-30% (1.2-1.5 g/kg), Fat 30-35% (healthy fats)",
    nutrient_requirements=[
        ConditionNutrientRequirement(
            nutrient_id="fiber",
            nutrient_name="Dietary Fiber",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("35"),
            target_unit="g",
            rationale="Slows glucose absorption, improves insulin sensitivity. PCOS involves insulin resistance",
            food_sources=["vegetables", "whole grains", "legumes", "chia seeds", "flaxseeds"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="chromium",
            nutrient_name="Chromium",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("200"),
            target_unit="mcg",
            rationale="Enhances insulin action. Studies show benefit for PCOS insulin resistance",
            food_sources=["broccoli", "whole grains", "green beans", "potatoes"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="inositol",
            nutrient_name="Inositol (Myo-inositol + D-chiro-inositol)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="mg",
            rationale="Improves insulin sensitivity, ovulation, and hormone balance in PCOS. Strong evidence. Often supplemented",
            food_sources=["citrus fruits", "beans", "whole grains", "nuts", "supplements (40:1 myo:d-chiro ratio)"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="omega_3_fatty_acids",
            nutrient_name="Omega-3 Fatty Acids",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("1500"),
            target_unit="mg",
            rationale="Reduces inflammation and may improve insulin sensitivity and androgen levels",
            food_sources=["fatty fish", "walnuts", "flaxseeds", "chia seeds", "fish oil supplements"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_d",
            nutrient_name="Vitamin D",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="IU",
            rationale="70% of PCOS women are deficient. Deficiency worsens insulin resistance and hormone issues",
            food_sources=["fatty fish", "fortified milk", "egg yolks", "sunlight", "supplements"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="magnesium",
            nutrient_name="Magnesium",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("400"),
            target_unit="mg",
            rationale="Improves insulin sensitivity. Often deficient in PCOS",
            food_sources=["spinach", "almonds", "black beans", "avocado", "dark chocolate"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="zinc",
            nutrient_name="Zinc",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("15"),
            target_unit="mg",
            rationale="Supports hormone balance, reduces hirsutism (excess hair growth)",
            food_sources=["oysters", "beef", "pumpkin seeds", "chickpeas", "cashews"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="added_sugars",
            nutrient_name="Added Sugars",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("25"),
            target_unit="g",
            rationale="Worsens insulin resistance and inflammation. Limit strictly",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="refined_carbs",
            nutrient_name="Refined Carbohydrates",
            recommendation_type=NutrientRecommendation.DECREASE,
            rationale="High glycemic foods worsen insulin resistance and hormone imbalance",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="spearmint_tea",
            nutrient_name="Spearmint Tea",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2"),
            target_unit="cups",
            rationale="May reduce androgens (testosterone) and hirsutism. 2 cups daily in studies",
            food_sources=["spearmint tea"],
            priority=3
        )
    ],
    food_restrictions=[
        FoodRestriction(
            food_or_category="Refined carbohydrates and sugars",
            reason="Spike blood sugar and insulin, worsening PCOS symptoms and weight gain",
            severity="limit",
            alternatives=["whole grains", "quinoa", "steel-cut oats", "legumes", "fresh fruits (low-GI)"]
        ),
        FoodRestriction(
            food_or_category="Sugar-sweetened beverages",
            reason="Liquid sugars cause rapid insulin spikes. Worst for PCOS",
            severity="must_avoid",
            alternatives=["water", "unsweetened tea", "sparkling water", "spearmint tea"]
        ),
        FoodRestriction(
            food_or_category="Processed foods high in advanced glycation end products (AGEs)",
            reason="Increase inflammation and insulin resistance. Found in fried, grilled, roasted at high temps",
            severity="limit",
            alternatives=["steamed", "boiled", "slow-cooked foods", "marinate in acidic liquids (lemon, vinegar)"]
        ),
        FoodRestriction(
            food_or_category="Dairy (trial elimination)",
            reason="May worsen acne and androgens in some PCOS women. Consider 3-month trial elimination",
            severity="monitor",
            alternatives=["plant-based milk", "coconut yogurt", "nutritional yeast"]
        ),
        FoodRestriction(
            food_or_category="Excessive omega-6 oils (soybean, corn oil)",
            reason="Promotes inflammation. High omega-6 to omega-3 ratio worsens PCOS",
            severity="limit",
            alternatives=["olive oil", "avocado oil", "coconut oil", "balance with omega-3 foods"]
        ),
        FoodRestriction(
            food_or_category="Alcohol",
            reason="Worsens insulin resistance and liver function. Interferes with hormone metabolism",
            severity="limit",
            alternatives=["limit to occasional", "avoid if trying to conceive"]
        )
    ],
    recommended_foods=[
        "Non-starchy vegetables (unlimited - broccoli, spinach, cauliflower, peppers)",
        "Fatty fish 2-3x/week (salmon, mackerel, sardines - omega-3s)",
        "Legumes (lentils, chickpeas, black beans - protein, fiber, low-GI)",
        "Nuts and seeds (almonds, walnuts, flaxseeds, chia seeds)",
        "Whole grains in moderation (quinoa, steel-cut oats, brown rice)",
        "Berries (low-GI fruits with antioxidants)",
        "Eggs (protein, vitamin D, choline)",
        "Extra virgin olive oil (anti-inflammatory)",
        "Avocado (healthy fats, fiber)",
        "Spearmint tea (2 cups daily - anti-androgen)",
        "Cinnamon (may improve insulin sensitivity)",
        "Green leafy vegetables (folate, magnesium, anti-inflammatory)"
    ],
    recommended_diet_patterns=[
        "Low-Glycemic Diet (most evidence for PCOS)",
        "Mediterranean Diet (anti-inflammatory, improves insulin sensitivity)",
        "DASH Diet",
        "Low-Carb Diet (40-45% carbs effective for many)",
        "Anti-inflammatory Diet",
        "Dairy-free trial (3 months) if acne prominent"
    ],
    lifestyle_recommendations=[
        "WEIGHT LOSS: 5-10% body weight loss restores ovulation in 75% of women, improves all PCOS symptoms",
        "EXERCISE: 150 min/week minimum. Combination of cardio and strength training. Exercise improves insulin sensitivity significantly",
        "STRENGTH TRAINING: Build muscle to improve insulin sensitivity and metabolism",
        "MANAGE STRESS: Stress increases cortisol which worsens insulin resistance and PCOS symptoms",
        "SLEEP: 7-9 hours. Poor sleep worsens insulin resistance and hormone imbalance",
        "MEAL TIMING: Eat larger breakfast, smaller dinner (aligns with insulin sensitivity patterns)",
        "AVOID SKIPPING MEALS: Causes insulin and blood sugar fluctuations",
        "REGULAR MEAL TIMES: Helps regulate hormones and insulin",
        "CONSIDER INTERMITTENT FASTING: May improve insulin sensitivity (discuss with doctor)",
        "MONITOR CYCLES: Track periods, ovulation, symptoms",
        "MANAGE HIRSUTISM: Spearmint tea, weight loss, potential medications",
        "FERTILITY: Address PCOS early if planning pregnancy"
    ],
    medication_interactions=[
        "METFORMIN: First-line medication for PCOS insulin resistance. Take with meals to reduce GI upset. May reduce vitamin B12 absorption - supplement if levels low",
        "SPIRONOLACTONE: Anti-androgen for hirsutism/acne. INCREASES potassium - avoid potassium supplements and salt substitutes. May cause irregular periods",
        "ORAL CONTRACEPTIVES: Regulate periods, reduce androgens. Some may worsen insulin resistance - choose carefully with doctor",
        "INOSITOL SUPPLEMENTS: Very safe. Take 2-4g daily. May improve metformin effectiveness. 40:1 ratio of myo-inositol to d-chiro-inositol optimal",
        "BERBERINE: Natural supplement, similar effects to metformin. May help insulin resistance. Can interact with medications",
        "CLOMID (if trying to conceive): No specific dietary interactions but healthier lifestyle improves response"
    ]
)


# ============================================================================
# REGISTRY - All Metabolic Conditions
# ============================================================================

METABOLIC_CONDITIONS = {
    "obesity": OBESITY,
    "metabolic_syndrome": METABOLIC_SYNDROME,
    "hyperlipidemia": HYPERLIPIDEMIA,
    "hypothyroidism": HYPOTHYROIDISM,
    "pcos": PCOS
}


def get_metabolic_condition(condition_id: str) -> Optional[HealthConditionProfile]:
    """Retrieve a metabolic condition profile by ID"""
    return METABOLIC_CONDITIONS.get(condition_id)


def list_metabolic_conditions() -> List[str]:
    """Get list of all available metabolic condition IDs"""
    return list(METABOLIC_CONDITIONS.keys())


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 3B: METABOLIC DISEASES - COMPREHENSIVE PROFILES")
    print("=" * 80)
    print()
    
    for condition_id, profile in METABOLIC_CONDITIONS.items():
        print(f"\n{'=' * 80}")
        print(f"CONDITION: {profile.condition_name}")
        print(f"ID: {profile.condition_id}")
        print("=" * 80)
        
        if profile.calorie_adjustment:
            print(f"\n🔢 CALORIE ADJUSTMENT:")
            print(f"   {profile.calorie_adjustment}")
        
        if profile.macro_adjustment:
            print(f"\n🥗 MACRONUTRIENT ADJUSTMENT:")
            print(f"   {profile.macro_adjustment}")
        
        print(f"\n📊 NUTRIENT REQUIREMENTS ({len(profile.nutrient_requirements)} nutrients):")
        for req in profile.nutrient_requirements[:5]:  # Show first 5
            symbol = "⬆️" if req.recommendation_type == NutrientRecommendation.INCREASE else "⬇️" if req.recommendation_type == NutrientRecommendation.DECREASE else "⚠️"
            target = f"{req.target_amount}{req.target_unit}" if req.target_amount else "Monitor"
            print(f"  {symbol} {req.nutrient_name}: {req.recommendation_type.value.upper()}")
            print(f"     Target: {target} | Priority: {req.priority}")
        
        print(f"\n🚫 FOOD RESTRICTIONS ({len(profile.food_restrictions)} restrictions):")
        for restriction in profile.food_restrictions[:3]:  # Show first 3
            severity_icon = "❌" if restriction.severity == "must_avoid" else "⚠️" if restriction.severity == "limit" else "👁️"
            print(f"  {severity_icon} {restriction.food_or_category}")
            print(f"     Severity: {restriction.severity.upper()}")
        
        print(f"\n✅ RECOMMENDED FOODS: {len(profile.recommended_foods)} foods")
        print(f"🍽️ DIET PATTERNS: {len(profile.recommended_diet_patterns)} patterns")
        print(f"💊 MEDICATION INTERACTIONS: {len(profile.medication_interactions)} interactions")
        print(f"💡 LIFESTYLE RECOMMENDATIONS: {len(profile.lifestyle_recommendations)} recommendations")
        print()
    
    print("\n" + "=" * 80)
    print("✅ PHASE 3B COMPLETE - 5 Metabolic Conditions Fully Mapped")
    print("=" * 80)
    print(f"\nTotal Conditions: {len(METABOLIC_CONDITIONS)}")
    print("Conditions:", ", ".join([p.condition_name for p in METABOLIC_CONDITIONS.values()]))
    print("\nEach condition includes:")
    print("  ✓ Calorie and macronutrient adjustments")
    print("  ✓ Detailed nutrient requirements")
    print("  ✓ Food restrictions with alternatives")
    print("  ✓ Medication interactions")
    print("  ✓ Evidence-based recommendations")
