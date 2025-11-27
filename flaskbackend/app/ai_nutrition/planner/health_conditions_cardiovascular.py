"""
Phase 3A: Cardiovascular Disease Conditions
Complete nutritional profiles for heart-related conditions

Conditions Covered:
1. Coronary Heart Disease (CHD)
2. Atherosclerosis
3. Heart Failure
4. Atrial Fibrillation

Each condition includes:
- Specific nutrient requirements with targets
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
    priority: int = 1  # 1=critical, 2=important, 3=beneficial


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


# ============================================================================
# CORONARY HEART DISEASE (CHD)
# ============================================================================

CORONARY_HEART_DISEASE = HealthConditionProfile(
    condition_id="coronary_heart_disease",
    condition_name="Coronary Heart Disease (CHD)",
    nutrient_requirements=[
        ConditionNutrientRequirement(
            nutrient_id="omega_3_fatty_acids",
            nutrient_name="Omega-3 Fatty Acids (EPA + DHA)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("1000"),
            target_unit="mg",
            rationale="Reduces triglycerides, inflammation, and arrhythmia risk. AHA recommends 1g/day for CHD patients",
            food_sources=["fatty fish (salmon, mackerel, sardines)", "fish oil supplements", "flaxseeds", "chia seeds", "walnuts"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="saturated_fat",
            nutrient_name="Saturated Fat",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("13"),
            target_unit="g",
            rationale="Raises LDL cholesterol. Limit to <7% of total calories (13g for 2000 kcal diet)",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="trans_fat",
            nutrient_name="Trans Fat",
            recommendation_type=NutrientRecommendation.AVOID,
            target_amount=Decimal("0"),
            target_unit="g",
            rationale="Increases LDL, decreases HDL cholesterol. No safe level of intake",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="dietary_cholesterol",
            nutrient_name="Dietary Cholesterol",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("200"),
            target_unit="mg",
            rationale="May raise blood cholesterol in some individuals. Limit to <200mg/day",
            food_sources=[],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="sodium",
            nutrient_name="Sodium",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("1500"),
            target_unit="mg",
            rationale="Reduces blood pressure and cardiovascular stress. AHA recommends <1500mg for heart disease",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="fiber",
            nutrient_name="Dietary Fiber (Soluble)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("30"),
            target_unit="g",
            rationale="Lowers LDL cholesterol and improves heart health. Soluble fiber especially effective",
            food_sources=["oats", "barley", "beans", "lentils", "apples", "psyllium husk"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="potassium",
            nutrient_name="Potassium",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("4700"),
            target_unit="mg",
            rationale="Counteracts sodium effects, reduces blood pressure",
            food_sources=["bananas", "potatoes", "spinach", "avocado", "white beans", "salmon"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="coq10",
            nutrient_name="Coenzyme Q10",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("100"),
            target_unit="mg",
            rationale="Supports heart energy production. Depleted by statins. Consider supplementation",
            food_sources=["organ meats", "fatty fish", "whole grains", "spinach"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_d",
            nutrient_name="Vitamin D",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="IU",
            rationale="Deficiency linked to increased CVD risk. Many CHD patients are deficient",
            food_sources=["fatty fish", "fortified milk", "egg yolks", "sunlight exposure"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="magnesium",
            nutrient_name="Magnesium",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("420"),
            target_unit="mg",
            rationale="Supports heart rhythm, vascular function. Deficiency increases arrhythmia risk",
            food_sources=["almonds", "spinach", "black beans", "avocado", "dark chocolate"],
            priority=2
        )
    ],
    food_restrictions=[
        FoodRestriction(
            food_or_category="Red meat (beef, pork, lamb) and processed meats",
            reason="High in saturated fat and cholesterol. Processed meats contain harmful nitrates",
            severity="limit",
            alternatives=["skinless poultry", "fish", "plant proteins (legumes, tofu)", "lean game meats (venison)"]
        ),
        FoodRestriction(
            food_or_category="Fried foods, fast food, commercial baked goods",
            reason="High in trans fats, saturated fats, and sodium",
            severity="must_avoid",
            alternatives=["baked or grilled foods", "air-fried options", "homemade whole grain baked goods"]
        ),
        FoodRestriction(
            food_or_category="Full-fat dairy (whole milk, cheese, butter, cream)",
            reason="High in saturated fat and cholesterol",
            severity="limit",
            alternatives=["low-fat or fat-free dairy", "plant-based milk (almond, oat)", "Greek yogurt (low-fat)"]
        ),
        FoodRestriction(
            food_or_category="Processed and cured meats (bacon, sausage, deli meats, hot dogs)",
            reason="Very high in sodium and saturated fat. Nitrates may increase CVD risk",
            severity="must_avoid",
            alternatives=["fresh roasted turkey or chicken", "canned low-sodium tuna", "homemade meat preparations"]
        ),
        FoodRestriction(
            food_or_category="High-sodium foods (canned soups, frozen dinners, chips, pickles, soy sauce)",
            reason="Excessive sodium raises blood pressure and fluid retention",
            severity="limit",
            alternatives=["fresh or frozen vegetables", "homemade soups", "low-sodium canned goods", "herbs and spices for flavor"]
        ),
        FoodRestriction(
            food_or_category="Sugary drinks and excessive added sugars",
            reason="Contributes to obesity, inflammation, and triglyceride elevation",
            severity="limit",
            alternatives=["water", "unsweetened tea", "sparkling water", "fresh fruit"]
        ),
        FoodRestriction(
            food_or_category="Alcohol (more than 1 drink/day for women, 2 for men)",
            reason="Excessive intake raises blood pressure and triglycerides",
            severity="limit",
            alternatives=["red wine in moderation (optional)", "non-alcoholic alternatives"]
        )
    ],
    recommended_foods=[
        "Fatty fish 2-3x/week (salmon, mackerel, sardines, herring)",
        "Colorful vegetables (leafy greens, bell peppers, carrots, beets)",
        "Whole grains (oats, quinoa, brown rice, whole wheat, barley)",
        "Legumes (lentils, chickpeas, black beans, kidney beans)",
        "Nuts and seeds (walnuts, almonds, chia seeds, flaxseeds)",
        "Berries (blueberries, strawberries, blackberries)",
        "Extra virgin olive oil",
        "Avocado",
        "Garlic and onions",
        "Green tea"
    ],
    recommended_diet_patterns=[
        "Mediterranean Diet (GOLD STANDARD for CHD)",
        "DASH Diet (Dietary Approaches to Stop Hypertension)",
        "Portfolio Diet (cholesterol-lowering)",
        "Plant-based diet with fish"
    ],
    lifestyle_recommendations=[
        "Engage in 150 minutes moderate aerobic exercise weekly (walking, swimming, cycling)",
        "Maintain healthy weight (BMI 18.5-24.9)",
        "Quit smoking and avoid secondhand smoke",
        "Manage stress through meditation, yoga, or relaxation techniques",
        "Get 7-9 hours quality sleep per night",
        "Monitor blood pressure regularly",
        "Limit alcohol to moderate intake",
        "Stay hydrated (8 glasses water daily)",
        "Eat smaller, more frequent meals to avoid overworking heart"
    ],
    medication_interactions=[
        "STATINS (Lipitor, Crestor): Deplete CoQ10 - supplement 100-200mg/day. Avoid grapefruit juice (increases drug levels)",
        "BLOOD THINNERS (Warfarin): Maintain consistent vitamin K intake. Avoid excessive garlic, ginger, turmeric supplements",
        "BETA-BLOCKERS (Metoprolol): May mask hypoglycemia symptoms in diabetics",
        "ACE INHIBITORS (Lisinopril): Avoid potassium supplements and salt substitutes (can cause hyperkalemia)",
        "ASPIRIN: Take with food to reduce stomach irritation. Omega-3s may increase bleeding risk at high doses",
        "DIGOXIN: Maintain consistent fiber intake (affects absorption). Monitor potassium and magnesium levels"
    ]
)


# ============================================================================
# ATHEROSCLEROSIS
# ============================================================================

ATHEROSCLEROSIS = HealthConditionProfile(
    condition_id="atherosclerosis",
    condition_name="Atherosclerosis (Plaque Buildup in Arteries)",
    nutrient_requirements=[
        ConditionNutrientRequirement(
            nutrient_id="omega_3_fatty_acids",
            nutrient_name="Omega-3 Fatty Acids (EPA + DHA)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="mg",
            rationale="Reduces inflammation and plaque progression. Higher doses (2g) shown to stabilize plaques",
            food_sources=["fatty fish", "fish oil supplements", "algae-based omega-3"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="antioxidants",
            nutrient_name="Antioxidants (Vitamins C, E, Polyphenols)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("500"),
            target_unit="mg",
            rationale="Prevents LDL oxidation, which drives plaque formation. Focus on food sources",
            food_sources=["berries", "dark leafy greens", "citrus fruits", "nuts", "green tea", "dark chocolate"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="l_arginine",
            nutrient_name="L-Arginine",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("3000"),
            target_unit="mg",
            rationale="Precursor to nitric oxide, which relaxes blood vessels and improves endothelial function",
            food_sources=["turkey", "chicken", "pumpkin seeds", "soybeans", "peanuts", "spirulina"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="niacin",
            nutrient_name="Niacin (Vitamin B3)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("20"),
            target_unit="mg",
            rationale="Raises HDL cholesterol and may reduce plaque progression. High doses require medical supervision",
            food_sources=["tuna", "chicken breast", "turkey", "peanuts", "mushrooms", "green peas"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="saturated_fat",
            nutrient_name="Saturated Fat",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("10"),
            target_unit="g",
            rationale="Accelerates plaque formation. Limit to <5% of calories",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="trans_fat",
            nutrient_name="Trans Fat",
            recommendation_type=NutrientRecommendation.AVOID,
            target_amount=Decimal("0"),
            target_unit="g",
            rationale="Most atherogenic fat. Directly promotes plaque buildup",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="plant_sterols",
            nutrient_name="Plant Sterols/Stanols",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="mg",
            rationale="Blocks cholesterol absorption. 2g/day can lower LDL by 10%",
            food_sources=["fortified spreads", "nuts", "seeds", "vegetable oils", "whole grains"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="fiber_soluble",
            nutrient_name="Soluble Fiber",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("15"),
            target_unit="g",
            rationale="Binds cholesterol in gut, reduces LDL. 10g soluble fiber lowers LDL by 5%",
            food_sources=["oats", "barley", "psyllium", "beans", "apples", "Brussels sprouts"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_k2",
            nutrient_name="Vitamin K2 (Menaquinone)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("100"),
            target_unit="mcg",
            rationale="Prevents arterial calcification, may reverse plaque calcium. Emerging evidence",
            food_sources=["natto", "fermented foods", "hard cheeses", "egg yolks", "grass-fed butter"],
            priority=3
        )
    ],
    food_restrictions=[
        FoodRestriction(
            food_or_category="Processed meats and red meat",
            reason="High in saturated fat, heme iron (pro-oxidant), and inflammatory compounds",
            severity="limit",
            alternatives=["fish", "poultry", "legumes", "plant proteins"]
        ),
        FoodRestriction(
            food_or_category="Trans fats (margarine, shortening, fried foods)",
            reason="Directly promotes atherosclerosis and plaque instability",
            severity="must_avoid",
            alternatives=["olive oil", "avocado oil", "nuts", "seeds"]
        ),
        FoodRestriction(
            food_or_category="High-sodium processed foods",
            reason="Damages endothelium and promotes inflammation",
            severity="limit",
            alternatives=["fresh whole foods", "herbs and spices", "lemon juice"]
        ),
        FoodRestriction(
            food_or_category="Refined carbohydrates and added sugars",
            reason="Increases inflammation, triglycerides, and oxidative stress",
            severity="limit",
            alternatives=["whole grains", "fresh fruit", "complex carbohydrates"]
        ),
        FoodRestriction(
            food_or_category="Excessive omega-6 oils (corn, soybean, sunflower)",
            reason="High omega-6 to omega-3 ratio promotes inflammation",
            severity="monitor",
            alternatives=["olive oil", "avocado oil", "canola oil (moderate omega-6)"]
        )
    ],
    recommended_foods=[
        "Fatty fish 3-4x/week (wild salmon, mackerel, sardines)",
        "Extra virgin olive oil (3-4 tablespoons daily)",
        "Nuts (walnuts, almonds) 1 oz daily",
        "Berries daily (blueberries, strawberries, raspberries)",
        "Leafy greens (spinach, kale, arugula)",
        "Cruciferous vegetables (broccoli, Brussels sprouts, cauliflower)",
        "Legumes (beans, lentils, chickpeas)",
        "Whole oats and barley",
        "Garlic (fresh, 1-2 cloves daily)",
        "Green tea (2-3 cups daily)",
        "Dark chocolate (70%+ cacao, 1 oz)",
        "Pomegranate juice (100% pure, 4 oz daily)"
    ],
    recommended_diet_patterns=[
        "Mediterranean Diet (most evidence for plaque regression)",
        "Portfolio Diet (combines cholesterol-lowering foods)",
        "Ornish Diet (very low-fat, plant-based - shown to reverse atherosclerosis)",
        "Anti-inflammatory diet"
    ],
    lifestyle_recommendations=[
        "Exercise 30-60 minutes daily (improves endothelial function)",
        "Achieve and maintain healthy weight",
        "QUIT SMOKING - single most important intervention",
        "Manage stress (chronic stress accelerates plaque formation)",
        "Consider intermittent fasting (may promote autophagy and plaque clearance)",
        "Get annual lipid panel and coronary calcium score",
        "Monitor blood pressure and blood sugar",
        "Practice good sleep hygiene (7-9 hours)",
        "Consider sauna therapy (improves endothelial function)",
        "Reduce exposure to air pollution"
    ],
    medication_interactions=[
        "STATINS: Deplete CoQ10. Avoid grapefruit. Red yeast rice contains statin-like compounds",
        "BLOOD THINNERS: Limit high-dose omega-3 (>3g), vitamin E, garlic supplements",
        "NIACIN (prescription): May cause flushing. Avoid alcohol. Don't take with hot beverages",
        "EZETIMIBE: May reduce absorption of fat-soluble vitamins (A, D, E, K) - monitor levels",
        "VITAMIN K2: May interfere with warfarin. Consult doctor before supplementing"
    ]
)


# ============================================================================
# HEART FAILURE
# ============================================================================

HEART_FAILURE = HealthConditionProfile(
    condition_id="heart_failure",
    condition_name="Heart Failure (Congestive Heart Failure)",
    nutrient_requirements=[
        ConditionNutrientRequirement(
            nutrient_id="sodium",
            nutrient_name="Sodium",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("2000"),
            target_unit="mg",
            rationale="CRITICAL: Reduces fluid retention and cardiac workload. Severe HF may need <1500mg",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="fluid",
            nutrient_name="Fluid Intake",
            recommendation_type=NutrientRecommendation.MONITOR,
            target_amount=Decimal("1500"),
            target_unit="ml",
            rationale="Limit to 1.5-2L/day to prevent fluid overload. Individual limits vary - follow doctor's orders",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="thiamine",
            nutrient_name="Thiamine (Vitamin B1)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("100"),
            target_unit="mg",
            rationale="Diuretics deplete thiamine. Deficiency worsens heart failure. May need supplementation",
            food_sources=["whole grains", "pork", "legumes", "nuts", "fortified cereals"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="coq10",
            nutrient_name="Coenzyme Q10",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("300"),
            target_unit="mg",
            rationale="Improves heart energy production. Studies show 200-300mg improves ejection fraction",
            food_sources=["organ meats", "fatty fish", "whole grains", "supplements"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="magnesium",
            nutrient_name="Magnesium",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("400"),
            target_unit="mg",
            rationale="Diuretics cause loss. Deficiency increases arrhythmia risk. Essential for heart function",
            food_sources=["almonds", "spinach", "black beans", "avocado", "pumpkin seeds"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="potassium",
            nutrient_name="Potassium",
            recommendation_type=NutrientRecommendation.MONITOR,
            target_amount=Decimal("3000"),
            target_unit="mg",
            rationale="Complex: Loop diuretics deplete (need more), ACE inhibitors retain (need less). Monitor blood levels",
            food_sources=["bananas", "potatoes", "white beans", "avocado", "spinach"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="omega_3_fatty_acids",
            nutrient_name="Omega-3 Fatty Acids",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("1000"),
            target_unit="mg",
            rationale="Reduces inflammation and may improve cardiac function. Safe in heart failure",
            food_sources=["fatty fish", "fish oil supplements"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="iron",
            nutrient_name="Iron",
            recommendation_type=NutrientRecommendation.MONITOR,
            target_amount=Decimal("18"),
            target_unit="mg",
            rationale="Anemia common in HF. Iron deficiency worsens symptoms. Check ferritin levels",
            food_sources=["lean meats", "beans", "spinach", "fortified cereals", "supplements if deficient"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="protein",
            nutrient_name="Protein",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("1.2"),
            target_unit="g/kg",
            rationale="Prevents cardiac cachexia (muscle wasting). Need more than general population",
            food_sources=["lean meats", "fish", "eggs", "legumes", "dairy", "protein shakes"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_d",
            nutrient_name="Vitamin D",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="IU",
            rationale="Deficiency associated with worse outcomes. Monitor and supplement if low",
            food_sources=["fatty fish", "fortified milk", "egg yolks", "sunlight", "supplements"],
            priority=2
        )
    ],
    food_restrictions=[
        FoodRestriction(
            food_or_category="High-sodium foods (CRITICAL RESTRICTION)",
            reason="Causes fluid retention, increases cardiac workload, worsens symptoms",
            severity="must_avoid",
            alternatives=["fresh meats", "unsalted nuts", "fresh vegetables", "herbs and spices", "no-salt seasonings"]
        ),
        FoodRestriction(
            food_or_category="Canned/processed foods (soups, frozen dinners, deli meats)",
            reason="Extremely high sodium content (500-1000mg per serving)",
            severity="must_avoid",
            alternatives=["fresh or frozen vegetables", "homemade soups", "fresh-cooked meats"]
        ),
        FoodRestriction(
            food_or_category="Restaurant and fast food",
            reason="Very high sodium, often 2000+mg per meal",
            severity="limit",
            alternatives=["home-cooked meals", "request no added salt", "choose steamed or grilled options"]
        ),
        FoodRestriction(
            food_or_category="Excessive fluids (>2L/day typically)",
            reason="Fluid overload worsens symptoms and may require hospitalization",
            severity="must_avoid",
            alternatives=["measure daily fluid intake", "include all beverages, soups, ice cream", "suck on ice chips or lemon wedges for thirst"]
        ),
        FoodRestriction(
            food_or_category="Alcohol",
            reason="Directly toxic to heart muscle. Can worsen heart failure",
            severity="must_avoid",
            alternatives=["non-alcoholic beverages", "sparkling water", "herbal teas"]
        ),
        FoodRestriction(
            food_or_category="High-fat and fried foods",
            reason="Difficult to digest, may worsen symptoms",
            severity="limit",
            alternatives=["baked or grilled foods", "lean proteins", "healthy fats in moderation"]
        ),
        FoodRestriction(
            food_or_category="Caffeine (excessive amounts)",
            reason="May cause arrhythmias in sensitive individuals",
            severity="monitor",
            alternatives=["limit to 1-2 cups coffee/day", "decaf options", "herbal teas"]
        ),
        FoodRestriction(
            food_or_category="Licorice (black licorice)",
            reason="Contains glycyrrhizin which causes sodium retention and potassium loss",
            severity="must_avoid",
            alternatives=["other candies", "anise-flavored alternatives without glycyrrhizin"]
        )
    ],
    recommended_foods=[
        "Fresh fruits (berries, apples, pears - watch fluid content)",
        "Fresh vegetables (leafy greens, broccoli, carrots)",
        "Lean proteins (skinless chicken, turkey, fish)",
        "Whole grains (oats, brown rice, quinoa)",
        "Low-sodium legumes (dried beans, lentils - prepare without salt)",
        "Unsalted nuts and seeds (almonds, walnuts, pumpkin seeds)",
        "Low-fat dairy (Greek yogurt, low-sodium cottage cheese)",
        "Fresh herbs and spices (garlic, ginger, turmeric, basil)",
        "Omega-3 rich fish (salmon, mackerel, sardines)",
        "Avocado (good fats, potassium, magnesium)"
    ],
    recommended_diet_patterns=[
        "Low-Sodium DASH Diet (<2000mg sodium)",
        "Mediterranean Diet (modified for low sodium)",
        "Heart-Healthy Diet with fluid restriction"
    ],
    lifestyle_recommendations=[
        "WEIGH YOURSELF DAILY (same time, same scale) - call doctor if gain >2-3 lbs in 1 day or 5 lbs in 1 week",
        "LIMIT SODIUM to <2000mg/day (or as prescribed)",
        "RESTRICT FLUIDS as directed (typically 1.5-2L/day including all beverages and soups)",
        "Read ALL food labels - avoid foods with >200mg sodium per serving",
        "Avoid salt shakers, soy sauce, condiments",
        "Eat smaller, frequent meals (large meals increase cardiac workload)",
        "Exercise as tolerated (cardiac rehab program ideal)",
        "Elevate legs when sitting to reduce edema",
        "Monitor symptoms: shortness of breath, swelling, fatigue",
        "Take medications exactly as prescribed (diuretics, ACE inhibitors, beta-blockers)",
        "Get adequate rest - fatigue is common",
        "Avoid temperature extremes (stress heart)"
    ],
    medication_interactions=[
        "DIURETICS (Lasix, Bumetanide): DEPLETE thiamine, magnesium, potassium. May need supplements. Take in morning to avoid nighttime urination",
        "ACE INHIBITORS/ARBs: INCREASE potassium - avoid supplements and salt substitutes. May cause dry cough",
        "BETA-BLOCKERS: May mask low blood sugar. Can cause fatigue. Don't stop suddenly",
        "DIGOXIN: Narrow therapeutic window. Low potassium or magnesium increases toxicity risk. Avoid high-fiber meals near dose",
        "ALDOSTERONE ANTAGONISTS (Spironolactone): RETAIN potassium - avoid high-K foods and supplements",
        "WARFARIN: Maintain consistent vitamin K intake. Avoid excessive omega-3 and garlic supplements",
        "ENTRESTO (Sacubitril/Valsartan): Avoid salt substitutes. Monitor potassium"
    ]
)


# ============================================================================
# ATRIAL FIBRILLATION (AFib)
# ============================================================================

ATRIAL_FIBRILLATION = HealthConditionProfile(
    condition_id="atrial_fibrillation",
    condition_name="Atrial Fibrillation (AFib)",
    nutrient_requirements=[
        ConditionNutrientRequirement(
            nutrient_id="magnesium",
            nutrient_name="Magnesium",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("400"),
            target_unit="mg",
            rationale="Deficiency strongly linked to AFib. Stabilizes electrical activity. May reduce episodes",
            food_sources=["spinach", "pumpkin seeds", "almonds", "black beans", "dark chocolate", "avocado"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="potassium",
            nutrient_name="Potassium",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("4700"),
            target_unit="mg",
            rationale="Essential for heart rhythm. Low levels trigger AFib. Maintain optimal levels",
            food_sources=["bananas", "sweet potatoes", "white beans", "spinach", "avocado", "salmon"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="omega_3_fatty_acids",
            nutrient_name="Omega-3 Fatty Acids (EPA + DHA)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("1000"),
            target_unit="mg",
            rationale="Anti-inflammatory, stabilizes heart rhythm. Studies show reduced AFib episodes",
            food_sources=["fatty fish", "fish oil supplements", "algae-based omega-3"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_d",
            nutrient_name="Vitamin D",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="IU",
            rationale="Deficiency associated with increased AFib risk. Aim for blood level 40-60 ng/mL",
            food_sources=["fatty fish", "fortified milk", "egg yolks", "mushrooms", "sunlight"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="coq10",
            nutrient_name="Coenzyme Q10",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("200"),
            target_unit="mg",
            rationale="Supports heart energy. Especially if on statins (deplete CoQ10)",
            food_sources=["organ meats", "fatty fish", "whole grains", "supplements"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="sodium",
            nutrient_name="Sodium",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("2300"),
            target_unit="mg",
            rationale="Excessive sodium may trigger episodes. Moderate restriction beneficial",
            food_sources=[],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="caffeine",
            nutrient_name="Caffeine",
            recommendation_type=NutrientRecommendation.MONITOR,
            target_amount=Decimal("200"),
            target_unit="mg",
            rationale="Controversial. Most can tolerate moderate amounts. Monitor individual response",
            food_sources=[],
            priority=3
        ),
        ConditionNutrientRequirement(
            nutrient_id="alcohol",
            nutrient_name="Alcohol",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("0"),
            target_unit="drinks",
            rationale="MAJOR TRIGGER for AFib ('Holiday Heart Syndrome'). Even moderate amounts increase risk",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="taurine",
            nutrient_name="Taurine",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="mg",
            rationale="Regulates calcium in heart cells. May reduce AFib episodes. Consider supplementation",
            food_sources=["fish", "shellfish", "poultry", "eggs", "supplements"],
            priority=3
        ),
        ConditionNutrientRequirement(
            nutrient_id="antioxidants",
            nutrient_name="Antioxidants (Vitamins C, E, Polyphenols)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("500"),
            target_unit="mg",
            rationale="Oxidative stress contributes to AFib. Antioxidants may help prevent episodes",
            food_sources=["berries", "leafy greens", "nuts", "dark chocolate", "green tea"],
            priority=2
        )
    ],
    food_restrictions=[
        FoodRestriction(
            food_or_category="Alcohol (ALL TYPES)",
            reason="STRONGEST DIETARY TRIGGER for AFib. Even 1-2 drinks can cause episodes in sensitive individuals",
            severity="must_avoid",
            alternatives=["non-alcoholic beverages", "sparkling water with fruit", "mocktails", "herbal teas"]
        ),
        FoodRestriction(
            food_or_category="Energy drinks and high-caffeine beverages",
            reason="Excessive caffeine may trigger AFib in susceptible individuals. Contains other stimulants",
            severity="must_avoid",
            alternatives=["water", "herbal teas", "moderate coffee (1-2 cups if tolerated)", "green tea"]
        ),
        FoodRestriction(
            food_or_category="Large meals (especially high-carb or high-fat)",
            reason="Can trigger vagal AFib. Large meals increase cardiac workload",
            severity="limit",
            alternatives=["smaller, frequent meals", "avoid lying down after eating", "eat slowly"]
        ),
        FoodRestriction(
            food_or_category="Excessive salt/sodium",
            reason="May trigger episodes and worsens underlying heart conditions",
            severity="limit",
            alternatives=["herbs and spices", "lemon juice", "garlic", "fresh foods"]
        ),
        FoodRestriction(
            food_or_category="Processed and fried foods",
            reason="High in trans fats and inflammatory compounds that may worsen AFib",
            severity="limit",
            alternatives=["grilled or baked foods", "whole foods", "home-cooked meals"]
        ),
        FoodRestriction(
            food_or_category="Tyramine-rich foods (if on certain medications)",
            reason="Aged cheeses, cured meats, fermented foods may interact with some heart medications",
            severity="monitor",
            alternatives=["fresh cheeses", "fresh meats", "consult with doctor"]
        ),
        FoodRestriction(
            food_or_category="Grapefruit and grapefruit juice (if on certain medications)",
            reason="Interferes with many heart medications including some antiarrhythmics and statins",
            severity="must_avoid",
            alternatives=["other citrus fruits", "oranges", "berries"]
        )
    ],
    recommended_foods=[
        "Fatty fish 2-3x/week (salmon, mackerel, sardines, herring)",
        "Leafy green vegetables (spinach, kale, Swiss chard) - rich in magnesium",
        "Bananas and sweet potatoes (potassium)",
        "Nuts and seeds (almonds, pumpkin seeds, chia seeds)",
        "Legumes (black beans, lentils, chickpeas)",
        "Berries (blueberries, strawberries, raspberries)",
        "Avocado (magnesium, potassium, healthy fats)",
        "Whole grains (oats, quinoa, brown rice)",
        "Dark chocolate (70%+ cacao) - magnesium, antioxidants",
        "Green tea (antioxidants, mild caffeine)"
    ],
    recommended_diet_patterns=[
        "Mediterranean Diet (strong evidence for AFib prevention)",
        "DASH Diet",
        "Anti-inflammatory diet",
        "Low-sodium diet if hypertensive"
    ],
    lifestyle_recommendations=[
        "AVOID ALCOHOL completely or limit to very rare occasions",
        "Maintain healthy weight (obesity major AFib risk factor)",
        "Regular moderate exercise (150 min/week) - AVOID extreme endurance training",
        "Manage stress (stress and anxiety can trigger episodes)",
        "Get 7-9 hours quality sleep (sleep apnea worsens AFib - get tested)",
        "Stay well-hydrated",
        "Avoid dehydration and electrolyte imbalances",
        "Monitor for triggers (keep diary of episodes and potential triggers)",
        "Treat underlying conditions (hypertension, sleep apnea, thyroid)",
        "Limit caffeine if it triggers your episodes (individual response varies)",
        "Eat smaller, more frequent meals",
        "Avoid extreme temperature changes",
        "Practice deep breathing and relaxation techniques"
    ],
    medication_interactions=[
        "WARFARIN (Coumadin): MAINTAIN CONSISTENT VITAMIN K INTAKE. Leafy greens are healthy but keep amounts steady. Avoid excessive omega-3 (>3g), vitamin E, garlic supplements",
        "NOACs (Eliquis, Xarelto, Pradaxa): Safer than warfarin with food. Still avoid excessive anti-clotting supplements. Grapefruit OK with most NOACs",
        "ANTIARRHYTHMICS (Amiodarone, Flecainide, Sotalol): Grapefruit may interfere with some. Take consistently with or without food",
        "BETA-BLOCKERS: May mask low blood sugar. Avoid excessive caffeine. Don't stop suddenly",
        "DIGOXIN: Low potassium or magnesium increases toxicity risk. Monitor electrolytes. High-fiber meals may reduce absorption",
        "CALCIUM CHANNEL BLOCKERS: Grapefruit significantly increases levels - AVOID grapefruit",
        "STATINS: Deplete CoQ10 - supplement recommended. Avoid grapefruit with simvastatin and lovastatin"
    ]
)


# ============================================================================
# REGISTRY - All Cardiovascular Conditions
# ============================================================================

CARDIOVASCULAR_CONDITIONS = {
    "coronary_heart_disease": CORONARY_HEART_DISEASE,
    "atherosclerosis": ATHEROSCLEROSIS,
    "heart_failure": HEART_FAILURE,
    "atrial_fibrillation": ATRIAL_FIBRILLATION
}


def get_cardiovascular_condition(condition_id: str) -> Optional[HealthConditionProfile]:
    """
    Retrieve a cardiovascular condition profile by ID
    
    Args:
        condition_id: ID of the condition (e.g., 'coronary_heart_disease')
        
    Returns:
        HealthConditionProfile or None if not found
    """
    return CARDIOVASCULAR_CONDITIONS.get(condition_id)


def list_cardiovascular_conditions() -> List[str]:
    """Get list of all available cardiovascular condition IDs"""
    return list(CARDIOVASCULAR_CONDITIONS.keys())


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 3A: CARDIOVASCULAR DISEASES - COMPREHENSIVE PROFILES")
    print("=" * 80)
    print()
    
    for condition_id, profile in CARDIOVASCULAR_CONDITIONS.items():
        print(f"\n{'=' * 80}")
        print(f"CONDITION: {profile.condition_name}")
        print(f"ID: {profile.condition_id}")
        print("=" * 80)
        
        print(f"\n📊 NUTRIENT REQUIREMENTS ({len(profile.nutrient_requirements)} nutrients):")
        for req in profile.nutrient_requirements:
            symbol = "⬆️" if req.recommendation_type == NutrientRecommendation.INCREASE else "⬇️" if req.recommendation_type == NutrientRecommendation.DECREASE else "⚠️"
            target = f"{req.target_amount}{req.target_unit}" if req.target_amount else "Monitor"
            print(f"  {symbol} {req.nutrient_name}: {req.recommendation_type.value.upper()}")
            print(f"     Target: {target} | Priority: {req.priority}")
            print(f"     Rationale: {req.rationale}")
        
        print(f"\n🚫 FOOD RESTRICTIONS ({len(profile.food_restrictions)} restrictions):")
        for restriction in profile.food_restrictions:
            severity_icon = "❌" if restriction.severity == "must_avoid" else "⚠️" if restriction.severity == "limit" else "👁️"
            print(f"  {severity_icon} {restriction.food_or_category}")
            print(f"     Reason: {restriction.reason}")
            print(f"     Severity: {restriction.severity.upper()}")
            if restriction.alternatives:
                print(f"     Alternatives: {', '.join(restriction.alternatives[:3])}")
        
        print(f"\n✅ RECOMMENDED FOODS ({len(profile.recommended_foods)}):")
        for food in profile.recommended_foods[:5]:
            print(f"  • {food}")
        
        print(f"\n🍽️ DIET PATTERNS ({len(profile.recommended_diet_patterns)}):")
        for pattern in profile.recommended_diet_patterns:
            print(f"  • {pattern}")
        
        print(f"\n💊 MEDICATION INTERACTIONS ({len(profile.medication_interactions)}):")
        for interaction in profile.medication_interactions[:3]:
            print(f"  • {interaction}")
        
        print(f"\n💡 LIFESTYLE RECOMMENDATIONS ({len(profile.lifestyle_recommendations)}):")
        for rec in profile.lifestyle_recommendations[:4]:
            print(f"  • {rec}")
        
        print()
    
    print("\n" + "=" * 80)
    print("✅ PHASE 3A COMPLETE - 4 Cardiovascular Conditions Fully Mapped")
    print("=" * 80)
    print(f"\nTotal Conditions: {len(CARDIOVASCULAR_CONDITIONS)}")
    print("Conditions:", ", ".join([p.condition_name for p in CARDIOVASCULAR_CONDITIONS.values()]))
    print("\nEach condition includes:")
    print("  ✓ Detailed nutrient requirements with targets and rationale")
    print("  ✓ Food restrictions with severity levels and alternatives")
    print("  ✓ Evidence-based medication interactions")
    print("  ✓ Lifestyle recommendations")
    print("  ✓ Recommended diet patterns")
    print("\nReady to integrate into health_condition_matcher.py!")
