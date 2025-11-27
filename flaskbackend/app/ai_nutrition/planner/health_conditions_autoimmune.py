"""
Phase 3C: Autoimmune Disease Conditions
Complete nutritional profiles for autoimmune-related conditions

Conditions Covered:
1. Systemic Lupus Erythematosus (SLE)
2. Multiple Sclerosis (MS)
3. Psoriasis
4. Hashimoto's Thyroiditis
5. Inflammatory Bowel Disease (Crohn's Disease & Ulcerative Colitis)

Each condition includes:
- Anti-inflammatory nutrients and protocols
- Vitamin D optimization
- Elimination diet guidance
- Gut health protocols
- Medication interactions
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


# ============================================================================
# SYSTEMIC LUPUS ERYTHEMATOSUS (SLE)
# ============================================================================

LUPUS = HealthConditionProfile(
    condition_id="lupus",
    condition_name="Systemic Lupus Erythematosus (SLE)",
    nutrient_requirements=[
        ConditionNutrientRequirement(
            nutrient_id="omega_3_fatty_acids",
            nutrient_name="Omega-3 Fatty Acids (EPA + DHA)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("3000"),
            target_unit="mg",
            rationale="Potent anti-inflammatory. Reduces lupus disease activity, joint pain, and cardiovascular risk",
            food_sources=["fatty fish 3-4x/week", "fish oil supplements", "algae-based omega-3"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_d",
            nutrient_name="Vitamin D",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="IU",
            rationale="80% of lupus patients are deficient. Modulates immune system, reduces disease activity. Target blood level 40-60 ng/mL",
            food_sources=["fatty fish", "fortified milk", "egg yolks", "supplements (often need 2000-4000 IU)"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="calcium",
            nutrient_name="Calcium",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("1200"),
            target_unit="mg",
            rationale="Corticosteroids (prednisone) increase bone loss. Prevention of osteoporosis critical",
            food_sources=["low-fat dairy", "fortified plant milk", "leafy greens", "sardines with bones"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_c",
            nutrient_name="Vitamin C",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("500"),
            target_unit="mg",
            rationale="Antioxidant, reduces oxidative stress (high in lupus), supports immune function",
            food_sources=["citrus fruits", "berries", "bell peppers", "broccoli", "kiwi"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_e",
            nutrient_name="Vitamin E",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("400"),
            target_unit="IU",
            rationale="Antioxidant, may reduce disease activity. Studies show benefit in lupus patients",
            food_sources=["almonds", "sunflower seeds", "spinach", "avocado", "supplements"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="folate",
            nutrient_name="Folate (Folic Acid)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("1000"),
            target_unit="mcg",
            rationale="CRITICAL if on methotrexate. Prevents methotrexate side effects (mouth sores, GI upset)",
            food_sources=["leafy greens", "legumes", "fortified grains", "supplements (must take if on methotrexate)"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="sodium",
            nutrient_name="Sodium",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("2000"),
            target_unit="mg",
            rationale="Corticosteroids cause fluid retention and hypertension. Limit sodium to reduce these effects",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="alfalfa",
            nutrient_name="Alfalfa (L-canavanine)",
            recommendation_type=NutrientRecommendation.AVOID,
            rationale="Contains L-canavanine which can trigger lupus flares. Documented to worsen symptoms",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="iron",
            nutrient_name="Iron",
            recommendation_type=NutrientRecommendation.MONITOR,
            target_amount=Decimal("18"),
            target_unit="mg",
            rationale="Anemia common in lupus. Check ferritin levels. Supplement if deficient",
            food_sources=["lean meats", "beans", "spinach", "fortified cereals", "supplements if needed"],
            priority=2
        )
    ],
    food_restrictions=[
        FoodRestriction(
            food_or_category="Alfalfa sprouts and supplements",
            reason="Contains L-canavanine which can trigger lupus flares and worsen symptoms",
            severity="must_avoid",
            alternatives=["other sprouts (broccoli, mung bean)", "leafy greens", "other vegetables"]
        ),
        FoodRestriction(
            food_or_category="Garlic (in large amounts/supplements)",
            reason="May stimulate immune system and potentially trigger flares in some individuals",
            severity="monitor",
            alternatives=["other herbs and spices", "small amounts in cooking likely safe"]
        ),
        FoodRestriction(
            food_or_category="High-sodium processed foods",
            reason="Corticosteroids cause fluid retention and hypertension. Excess sodium worsens this",
            severity="limit",
            alternatives=["fresh whole foods", "herbs and spices", "low-sodium options"]
        ),
        FoodRestriction(
            food_or_category="Alcohol",
            reason="May interact with medications (NSAIDs, methotrexate). Can worsen liver function",
            severity="limit",
            alternatives=["limit to occasional", "avoid if on methotrexate", "non-alcoholic options"]
        ),
        FoodRestriction(
            food_or_category="High-fat foods (if on corticosteroids)",
            reason="Prednisone causes weight gain and increases cholesterol. High-fat foods worsen this",
            severity="limit",
            alternatives=["lean proteins", "baked/grilled foods", "plant-based fats in moderation"]
        ),
        FoodRestriction(
            food_or_category="Grapefruit",
            reason="May interfere with immunosuppressant medications",
            severity="monitor",
            alternatives=["other citrus fruits", "berries", "check with doctor about specific medications"]
        )
    ],
    recommended_foods=[
        "Fatty fish 3-4x/week (salmon, mackerel, sardines - omega-3s and vitamin D)",
        "Colorful fruits and vegetables (antioxidants, vitamin C)",
        "Leafy greens (folate, calcium, magnesium, anti-inflammatory)",
        "Berries (anthocyanins, antioxidants, anti-inflammatory)",
        "Nuts and seeds (vitamin E, healthy fats, selenium)",
        "Whole grains (fiber, B vitamins)",
        "Legumes (protein, fiber, folate, iron)",
        "Low-fat dairy or fortified alternatives (calcium, vitamin D)",
        "Green tea (polyphenols, anti-inflammatory)",
        "Turmeric/curcumin (potent anti-inflammatory - with black pepper for absorption)"
    ],
    recommended_diet_patterns=[
        "Mediterranean Diet (strong evidence for reducing inflammation and cardiovascular risk)",
        "Anti-inflammatory Diet",
        "DASH Diet (if hypertension present from steroids)"
    ],
    lifestyle_recommendations=[
        "SUN PROTECTION CRITICAL: UV light triggers lupus flares. Use SPF 30+ daily, wear protective clothing, avoid peak sun",
        "WEIGHT MANAGEMENT: Corticosteroids cause weight gain. Monitor weight, exercise regularly",
        "EXERCISE: 150 min/week moderate activity. Reduces fatigue, maintains bone density, manages weight",
        "STRESS MANAGEMENT: Stress triggers flares. Practice meditation, yoga, relaxation techniques",
        "ADEQUATE REST: Fatigue is major symptom. Prioritize 8-9 hours sleep, rest when needed",
        "AVOID SMOKING: Smoking worsens lupus, reduces medication effectiveness, increases cardiovascular risk",
        "LIMIT ALCOHOL: Especially if on methotrexate (liver toxicity)",
        "STAY HYDRATED: Adequate fluid intake important",
        "MONITOR BONE HEALTH: Annual bone density scan if on long-term steroids",
        "CARDIOVASCULAR SCREENING: Lupus increases heart disease risk. Monitor cholesterol, BP",
        "INFECTION PREVENTION: Immunosuppressants increase infection risk. Good hygiene, avoid sick contacts"
    ],
    medication_interactions=[
        "CORTICOSTEROIDS (Prednisone): Cause bone loss (need calcium 1200mg + vitamin D), weight gain (watch calories), sodium retention (limit sodium), increase blood sugar (monitor). Take with food to reduce stomach upset",
        "METHOTREXATE: MUST take folic acid 1mg daily (prevents side effects). AVOID alcohol (liver toxicity). May cause nausea - take at bedtime with food. Increases sun sensitivity",
        "HYDROXYCHLOROQUINE (Plaquenil): Take with food. Annual eye exams required. May cause nausea initially",
        "IMMUNOSUPPRESSANTS (Azathioprine, Mycophenolate): Increase infection risk. Avoid raw/undercooked foods. Grapefruit may interact with some",
        "NSAIDs: Take with food. May cause stomach upset. Limit alcohol. Can affect kidneys - stay hydrated",
        "WARFARIN (if blood clots): Maintain CONSISTENT vitamin K intake. Leafy greens healthy but keep amounts steady"
    ]
)


# ============================================================================
# MULTIPLE SCLEROSIS (MS)
# ============================================================================

MULTIPLE_SCLEROSIS = HealthConditionProfile(
    condition_id="multiple_sclerosis",
    condition_name="Multiple Sclerosis (MS)",
    nutrient_requirements=[
        ConditionNutrientRequirement(
            nutrient_id="vitamin_d",
            nutrient_name="Vitamin D",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("5000"),
            target_unit="IU",
            rationale="CRITICAL for MS. Low vitamin D linked to increased MS risk and progression. Target blood level 40-60 ng/mL. Many MS patients need 5000+ IU",
            food_sources=["fatty fish", "fortified milk", "egg yolks", "supplements (high-dose often needed)", "sun exposure"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="omega_3_fatty_acids",
            nutrient_name="Omega-3 Fatty Acids (EPA + DHA)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="mg",
            rationale="Anti-inflammatory, protects myelin sheath, may slow disease progression",
            food_sources=["fatty fish 3-4x/week", "fish oil supplements", "algae-based omega-3"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="saturated_fat",
            nutrient_name="Saturated Fat",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("15"),
            target_unit="g",
            rationale="Swank Diet (low saturated fat) shows reduced disease progression. Limit to <15g/day",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="biotin",
            nutrient_name="Biotin (High-Dose)",
            recommendation_type=NutrientRecommendation.MONITOR,
            target_amount=Decimal("300"),
            target_unit="mg",
            rationale="High-dose biotin (300mg) shows promise for progressive MS. Requires medical supervision. May affect lab tests",
            food_sources=["medical-grade supplements only (food sources insufficient for therapeutic dose)"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_b12",
            nutrient_name="Vitamin B12",
            recommendation_type=NutrientRecommendation.MONITOR,
            target_amount=Decimal("1000"),
            target_unit="mcg",
            rationale="Deficiency can mimic MS symptoms. Essential for myelin repair. Check levels annually",
            food_sources=["meat", "fish", "dairy", "fortified cereals", "supplements if deficient"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="antioxidants",
            nutrient_name="Antioxidants (Vitamins C, E, Selenium)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("500"),
            target_unit="mg",
            rationale="Reduce oxidative stress (high in MS). Support immune system. Neuroprotective",
            food_sources=["colorful fruits and vegetables", "berries", "nuts", "seeds", "green tea"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="sodium",
            nutrient_name="Sodium",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("2000"),
            target_unit="mg",
            rationale="High sodium intake linked to increased MS disease activity and relapses",
            food_sources=[],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="fiber",
            nutrient_name="Dietary Fiber",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("30"),
            target_unit="g",
            rationale="Supports gut microbiome (important for immune function). Prevents constipation (common in MS)",
            food_sources=["vegetables", "whole grains", "legumes", "fruits", "psyllium"],
            priority=2
        )
    ],
    food_restrictions=[
        FoodRestriction(
            food_or_category="High saturated fat foods (red meat, full-fat dairy, butter)",
            reason="Swank Diet research shows high saturated fat worsens MS progression",
            severity="limit",
            alternatives=["fish", "skinless poultry", "plant proteins", "low-fat dairy"]
        ),
        FoodRestriction(
            food_or_category="Processed and fried foods",
            reason="High in unhealthy fats and sodium. Pro-inflammatory",
            severity="limit",
            alternatives=["baked or grilled foods", "whole foods", "home-cooked meals"]
        ),
        FoodRestriction(
            food_or_category="High-sodium foods",
            reason="Excess sodium associated with increased MS disease activity and relapses",
            severity="limit",
            alternatives=["fresh foods", "herbs and spices", "low-sodium options"]
        ),
        FoodRestriction(
            food_or_category="Gluten (trial elimination)",
            reason="Some MS patients report symptom improvement with gluten-free diet. Consider 3-month trial",
            severity="monitor",
            alternatives=["gluten-free grains (rice, quinoa, oats)", "test for celiac disease first"]
        ),
        FoodRestriction(
            food_or_category="Dairy (trial elimination)",
            reason="Controversial. Some evidence molecular mimicry may trigger immune response. Consider trial",
            severity="monitor",
            alternatives=["plant-based milk", "calcium-fortified alternatives", "trial 3 months"]
        ),
        FoodRestriction(
            food_or_category="Alcohol (excessive)",
            reason="May worsen symptoms, interact with medications, affect balance and cognition",
            severity="limit",
            alternatives=["limit to occasional", "non-alcoholic options"]
        )
    ],
    recommended_foods=[
        "Fatty fish 3-4x/week (salmon, sardines, mackerel - omega-3s and vitamin D)",
        "Leafy greens (vitamins, minerals, antioxidants, low in saturated fat)",
        "Berries (anthocyanins, antioxidants, neuroprotective)",
        "Nuts and seeds (vitamin E, healthy fats, protein)",
        "Legumes (protein, fiber, low-fat)",
        "Whole grains (fiber, B vitamins, energy)",
        "Colorful vegetables (antioxidants, vitamins, minerals)",
        "Olive oil (monounsaturated fat, anti-inflammatory)",
        "Green tea (EGCG antioxidant, neuroprotective)",
        "Turmeric/curcumin (anti-inflammatory, neuroprotective)"
    ],
    recommended_diet_patterns=[
        "Swank Diet (very low saturated fat <15g/day - most studied MS diet, 50+ years data)",
        "Wahls Protocol (nutrient-dense, paleolithic-style, emphasizes vegetables)",
        "Mediterranean Diet (anti-inflammatory, heart-healthy)",
        "Overcoming MS Diet (plant-based, very low saturated fat)",
        "Anti-inflammatory Diet"
    ],
    lifestyle_recommendations=[
        "VITAMIN D MONITORING: Check blood levels 2-3x/year. Target 40-60 ng/mL. Supplement as needed",
        "EXERCISE: CRITICAL for MS management. Improves strength, balance, fatigue, mood. Adapt to abilities",
        "HEAT MANAGEMENT: Heat worsens symptoms (Uhthoff's phenomenon). Stay cool, use cooling vests, cold water",
        "FATIGUE MANAGEMENT: Pace activities, rest when needed, energy conservation techniques",
        "STRESS REDUCTION: Stress may trigger relapses. Meditation, yoga, counseling",
        "ADEQUATE SLEEP: 7-9 hours. Poor sleep worsens symptoms",
        "SMOKING CESSATION: Smoking accelerates MS progression. Quit immediately",
        "WEIGHT MANAGEMENT: Obesity associated with worse outcomes",
        "COGNITIVE EXERCISES: Brain training, puzzles, learning new skills",
        "SOCIAL SUPPORT: Join MS support groups, stay connected",
        "PHYSICAL THERAPY: For mobility, balance, strength training",
        "AVOID INFECTIONS: Infections can trigger relapses. Good hygiene, flu shot annually"
    ],
    medication_interactions=[
        "DISEASE-MODIFYING THERAPIES (DMTs): Various drugs (interferons, glatiramer, etc.). Follow specific dietary guidelines for each. Some require lab monitoring",
        "CORTICOSTEROIDS (for relapses): Cause bone loss (need calcium + vitamin D), increase blood sugar, sodium retention. Short-term use",
        "VITAMIN D: High doses safe but monitor blood levels. Can cause hypercalcemia if excessive (rare)",
        "HIGH-DOSE BIOTIN: May interfere with lab tests (thyroid, troponin, etc.). Inform doctors before blood work. Stop 48-72 hours before tests",
        "BACLOFEN (muscle spasms): May cause drowsiness. Avoid alcohol",
        "GILENYA (Fingolimod): Avoid live vaccines. May interact with certain foods - follow specific guidelines"
    ]
)


# ============================================================================
# PSORIASIS
# ============================================================================

PSORIASIS = HealthConditionProfile(
    condition_id="psoriasis",
    condition_name="Psoriasis (Chronic Inflammatory Skin Condition)",
    nutrient_requirements=[
        ConditionNutrientRequirement(
            nutrient_id="omega_3_fatty_acids",
            nutrient_name="Omega-3 Fatty Acids (EPA + DHA)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("3000"),
            target_unit="mg",
            rationale="Reduces inflammation, may improve psoriasis severity and reduce plaques",
            food_sources=["fatty fish 3-4x/week", "fish oil supplements", "algae-based omega-3"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_d",
            nutrient_name="Vitamin D",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="IU",
            rationale="Modulates immune system, reduces skin cell proliferation. Topical vitamin D also used. Often deficient in psoriasis",
            food_sources=["fatty fish", "fortified milk", "egg yolks", "supplements", "safe sun exposure"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_a",
            nutrient_name="Vitamin A (Beta-Carotene)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("5000"),
            target_unit="IU",
            rationale="Supports skin health, regulates cell growth. Retinoids (vitamin A derivatives) used in psoriasis treatment",
            food_sources=["sweet potatoes", "carrots", "spinach", "kale", "cantaloupe"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="selenium",
            nutrient_name="Selenium",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("200"),
            target_unit="mcg",
            rationale="Antioxidant, reduces inflammation. Low selenium associated with psoriasis severity",
            food_sources=["Brazil nuts (2-3 daily)", "seafood", "eggs", "whole grains"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="zinc",
            nutrient_name="Zinc",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("15"),
            target_unit="mg",
            rationale="Supports skin healing, immune function. Deficiency common in psoriasis",
            food_sources=["oysters", "beef", "pumpkin seeds", "chickpeas", "cashews"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="omega_6_fatty_acids",
            nutrient_name="Omega-6 Fatty Acids (Excessive)",
            recommendation_type=NutrientRecommendation.DECREASE,
            rationale="High omega-6 to omega-3 ratio promotes inflammation. Limit corn, soybean, sunflower oils",
            food_sources=[],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="alcohol",
            nutrient_name="Alcohol",
            recommendation_type=NutrientRecommendation.DECREASE,
            target_amount=Decimal("0"),
            target_unit="drinks",
            rationale="Alcohol is major trigger for psoriasis flares. Worsens inflammation, reduces treatment effectiveness",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="curcumin",
            nutrient_name="Curcumin (Turmeric)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("500"),
            target_unit="mg",
            rationale="Potent anti-inflammatory. Studies show benefit in psoriasis. Take with black pepper for absorption",
            food_sources=["turmeric spice", "curcumin supplements (with piperine)"],
            priority=2
        )
    ],
    food_restrictions=[
        FoodRestriction(
            food_or_category="Alcohol (all types)",
            reason="MAJOR TRIGGER for psoriasis flares. Increases inflammation, reduces medication effectiveness",
            severity="must_avoid",
            alternatives=["non-alcoholic beverages", "herbal teas", "sparkling water"]
        ),
        FoodRestriction(
            food_or_category="Nightshade vegetables (trial elimination)",
            reason="Some psoriasis patients report flares from nightshades (tomatoes, peppers, potatoes, eggplant). Consider 3-month trial",
            severity="monitor",
            alternatives=["other vegetables", "sweet potatoes instead of white potatoes", "track individual response"]
        ),
        FoodRestriction(
            food_or_category="Gluten (trial elimination)",
            reason="Celiac disease more common in psoriasis. Even without celiac, some report improvement gluten-free",
            severity="monitor",
            alternatives=["gluten-free grains", "test for celiac first", "trial 3 months"]
        ),
        FoodRestriction(
            food_or_category="Red meat and processed meats",
            reason="High in arachidonic acid (promotes inflammation) and saturated fat",
            severity="limit",
            alternatives=["fish", "poultry", "plant proteins", "legumes"]
        ),
        FoodRestriction(
            food_or_category="Dairy (trial elimination)",
            reason="Some report improvement dairy-free. May be related to saturated fat or proteins",
            severity="monitor",
            alternatives=["plant-based milk", "calcium-fortified alternatives", "trial 3 months"]
        ),
        FoodRestriction(
            food_or_category="Refined sugars and processed foods",
            reason="Promote inflammation, may worsen psoriasis",
            severity="limit",
            alternatives=["whole foods", "fresh fruits", "complex carbohydrates"]
        ),
        FoodRestriction(
            food_or_category="Excessive omega-6 oils (corn, soybean, sunflower)",
            reason="High omega-6 to omega-3 ratio promotes inflammation",
            severity="limit",
            alternatives=["olive oil", "avocado oil", "balance with omega-3 foods"]
        )
    ],
    recommended_foods=[
        "Fatty fish 3-4x/week (salmon, mackerel, sardines - omega-3s)",
        "Colorful fruits and vegetables (antioxidants, vitamins, anti-inflammatory)",
        "Leafy greens (vitamins A, C, K, folate)",
        "Berries (anthocyanins, antioxidants)",
        "Nuts and seeds (vitamin E, selenium, healthy fats)",
        "Olive oil (monounsaturated fat, anti-inflammatory)",
        "Turmeric/curcumin (add to foods or supplement)",
        "Green tea (polyphenols, anti-inflammatory)",
        "Probiotic foods (yogurt, kefir, sauerkraut - gut health)",
        "Whole grains (fiber, B vitamins)"
    ],
    recommended_diet_patterns=[
        "Mediterranean Diet (strong evidence for reducing psoriasis severity)",
        "Anti-inflammatory Diet",
        "Pagano Diet (alkaline diet for psoriasis - anecdotal support)",
        "Autoimmune Protocol (AIP) - elimination diet",
        "Gluten-free diet (trial for 3 months)"
    ],
    lifestyle_recommendations=[
        "WEIGHT LOSS: If overweight, losing 5-10% body weight significantly improves psoriasis",
        "LIMIT ALCOHOL: Major trigger. Abstinence ideal, or very limited intake",
        "STRESS MANAGEMENT: Stress triggers flares. Meditation, yoga, therapy",
        "AVOID SKIN TRAUMA: Koebner phenomenon (injury triggers psoriasis). Protect skin",
        "MOISTURIZE: Keep skin hydrated. Use thick creams/ointments",
        "IDENTIFY TRIGGERS: Keep diary of flares and potential triggers (foods, stress, infections)",
        "QUIT SMOKING: Smoking worsens psoriasis",
        "EXERCISE: Regular activity reduces inflammation, manages weight, reduces stress",
        "ADEQUATE SLEEP: 7-9 hours supports immune function",
        "SUN EXPOSURE: Moderate sun helps (UV light therapy used). But avoid sunburn",
        "AVOID INFECTIONS: Strep throat can trigger guttate psoriasis. Treat promptly"
    ],
    medication_interactions=[
        "METHOTREXATE: MUST take folic acid 1mg daily. AVOID alcohol (severe liver toxicity risk). Take with food. Increases sun sensitivity",
        "BIOLOGICS (Humira, Enbrel, Stelara): Increase infection risk. Avoid raw/undercooked foods. Can't take live vaccines",
        "CYCLOSPORINE: AVOID grapefruit (increases drug levels). Monitor kidney function. Increases blood pressure",
        "ACITRETIN (retinoid): AVOID alcohol (can form toxic metabolite). Avoid vitamin A supplements (toxicity). Causes birth defects - strict contraception",
        "TOPICAL VITAMIN D: Safe. May complement oral vitamin D supplementation",
        "APREMILAST (Otezla): Take with food. May cause GI upset initially. Weight loss common"
    ]
)


# ============================================================================
# HASHIMOTO'S THYROIDITIS
# ============================================================================

HASHIMOTOS = HealthConditionProfile(
    condition_id="hashimotos_thyroiditis",
    condition_name="Hashimoto's Thyroiditis (Autoimmune Hypothyroidism)",
    nutrient_requirements=[
        ConditionNutrientRequirement(
            nutrient_id="selenium",
            nutrient_name="Selenium",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("200"),
            target_unit="mcg",
            rationale="Reduces thyroid antibodies (TPO, TG), supports thyroid hormone conversion. Critical for Hashimoto's. Studies show 200mcg reduces antibodies by 40%",
            food_sources=["Brazil nuts (2-3 daily - don't exceed)", "seafood", "eggs", "supplements"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_d",
            nutrient_name="Vitamin D",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="IU",
            rationale="Modulates immune system. Deficiency very common in Hashimoto's (90%). Low levels correlate with higher antibodies",
            food_sources=["fatty fish", "fortified milk", "egg yolks", "supplements", "safe sun exposure"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="iodine",
            nutrient_name="Iodine",
            recommendation_type=NutrientRecommendation.MONITOR,
            target_amount=Decimal("150"),
            target_unit="mcg",
            rationale="COMPLEX: Needed for thyroid hormone production BUT excess iodine (>500mcg) can worsen Hashimoto's autoimmunity. Use iodized salt, avoid megadoses",
            food_sources=["iodized salt", "seafood", "dairy", "eggs", "avoid kelp supplements"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="zinc",
            nutrient_name="Zinc",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("15"),
            target_unit="mg",
            rationale="Required for thyroid hormone production and T4 to T3 conversion. Often deficient",
            food_sources=["oysters", "beef", "pumpkin seeds", "chickpeas", "cashews"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="iron",
            nutrient_name="Iron",
            recommendation_type=NutrientRecommendation.MONITOR,
            target_amount=Decimal("18"),
            target_unit="mg",
            rationale="Deficiency impairs thyroid function and worsens hypothyroid symptoms. Check ferritin. Separate from thyroid meds by 4 hours",
            food_sources=["lean meats", "beans", "spinach", "fortified cereals", "supplements if deficient"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="omega_3_fatty_acids",
            nutrient_name="Omega-3 Fatty Acids",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="mg",
            rationale="Anti-inflammatory. May reduce thyroid antibodies and autoimmune activity",
            food_sources=["fatty fish", "fish oil supplements", "flaxseeds", "chia seeds"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_b12",
            nutrient_name="Vitamin B12",
            recommendation_type=NutrientRecommendation.MONITOR,
            target_amount=Decimal("1000"),
            target_unit="mcg",
            rationale="Autoimmune conditions often cluster. Pernicious anemia (B12 deficiency) common with Hashimoto's. Check levels",
            food_sources=["meat", "fish", "dairy", "fortified cereals", "supplements if deficient"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="gluten",
            nutrient_name="Gluten",
            recommendation_type=NutrientRecommendation.MONITOR,
            rationale="Celiac disease 10x more common with Hashimoto's. Molecular mimicry: gluten proteins similar to thyroid tissue. Consider elimination trial",
            food_sources=[],
            priority=1
        )
    ],
    food_restrictions=[
        FoodRestriction(
            food_or_category="Gluten (trial elimination)",
            reason="Celiac disease 10x more common with Hashimoto's. Even without celiac, many report reduced antibodies gluten-free. 3-6 month trial recommended",
            severity="monitor",
            alternatives=["gluten-free grains (rice, quinoa, oats)", "test for celiac first", "track antibody levels during trial"]
        ),
        FoodRestriction(
            food_or_category="Soy (in large amounts)",
            reason="Can interfere with thyroid hormone absorption and function. Moderate amounts likely OK. Separate from thyroid meds by 4 hours",
            severity="monitor",
            alternatives=["other plant proteins", "limit to 1-2 servings if consuming", "time away from medication"]
        ),
        FoodRestriction(
            food_or_category="Raw cruciferous vegetables (in excess)",
            reason="Goitrogens can interfere with thyroid function in large RAW amounts. COOKED crucifers are SAFE and healthy",
            severity="monitor",
            alternatives=["cook crucifers (deactivates goitrogens)", "normal cooked amounts fine", "don't juice large amounts raw"]
        ),
        FoodRestriction(
            food_or_category="Excess iodine (kelp, seaweed supplements)",
            reason="Iodine >500mcg/day can worsen Hashimoto's autoimmunity. Iodized salt OK, but avoid megadose supplements",
            severity="limit",
            alternatives=["iodized salt (adequate)", "seafood in moderation", "avoid kelp supplements"]
        ),
        FoodRestriction(
            food_or_category="Processed foods and refined sugars",
            reason="Promote inflammation, may worsen autoimmune activity. Hypothyroidism slows metabolism - easy to gain weight",
            severity="limit",
            alternatives=["whole foods", "anti-inflammatory diet", "nutrient-dense choices"]
        ),
        FoodRestriction(
            food_or_category="Dairy (trial elimination for some)",
            reason="Lactose intolerance and dairy sensitivity more common with autoimmune conditions. Consider trial if symptoms present",
            severity="monitor",
            alternatives=["plant-based milk", "lactose-free dairy", "calcium-fortified alternatives"]
        )
    ],
    recommended_foods=[
        "Brazil nuts (2-3 daily for selenium - DON'T EXCEED)",
        "Fatty fish (salmon, sardines - selenium, omega-3s, vitamin D)",
        "Eggs (selenium, iodine, vitamin D, B12)",
        "Lean meats (iron, zinc, B12)",
        "Cooked cruciferous vegetables (nutrient-dense, healthy)",
        "Seaweed in moderation (iodine, but not excessive)",
        "Berries (antioxidants, anti-inflammatory)",
        "Leafy greens (folate, magnesium, iron)",
        "Legumes (zinc, iron, fiber)",
        "Probiotic foods (yogurt, kefir - gut health, if tolerate dairy)"
    ],
    recommended_diet_patterns=[
        "Autoimmune Protocol (AIP) Diet - most evidence for Hashimoto's (elimination, then reintroduction)",
        "Gluten-Free Diet (high priority trial - 3-6 months)",
        "Mediterranean Diet (anti-inflammatory)",
        "Paleo Diet (eliminates grains, dairy, legumes)",
        "Anti-inflammatory Diet"
    ],
    lifestyle_recommendations=[
        "GLUTEN-FREE TRIAL: Test for celiac disease first, then try strict GF for 3-6 months. Track antibody levels (TPO, TG)",
        "TAKE THYROID MEDICATION CORRECTLY: Levothyroxine on empty stomach, 30-60 min before food, with water only",
        "SUPPLEMENT TIMING: Separate calcium, iron, magnesium, fiber by 4 hours from thyroid medication",
        "STRESS MANAGEMENT: Stress worsens autoimmune conditions. Meditation, yoga, therapy",
        "ADEQUATE SLEEP: 7-9 hours. Hypothyroidism causes fatigue",
        "EXERCISE: Regular activity despite fatigue. Improves metabolism, energy, mood",
        "MONITOR ANTIBODIES: Check TPO and TG antibodies every 6-12 months to track autoimmune activity",
        "CHECK NUTRIENT LEVELS: Annual testing for vitamin D, B12, ferritin, selenium",
        "GUT HEALTH: Leaky gut may contribute to autoimmunity. Probiotics, fiber, avoid food sensitivities",
        "AVOID EXCESSIVE IODINE: No kelp supplements. Iodized salt sufficient",
        "WEIGHT MANAGEMENT: Hypothyroidism slows metabolism. Portion control, regular meals"
    ],
    medication_interactions=[
        "LEVOTHYROXINE (Synthroid): CRITICAL TIMING: Empty stomach, 30-60 min before food, with water. Wait 4 hours before: calcium, iron, magnesium, antacids, soy, fiber supplements, coffee",
        "SELENIUM: Safe 200mcg/day. Brazil nuts provide this (2-3 nuts). Don't exceed 400mcg (toxicity)",
        "CALCIUM/IRON: Reduce thyroid med absorption by 50%. MUST separate by 4 hours",
        "VITAMIN D: Safe to supplement. No interaction with thyroid medication",
        "BIOTIN: Can cause falsely abnormal thyroid lab results. Stop 2-3 days before blood test",
        "GLUTEN CROSS-REACTION: If celiac disease present, even tiny gluten amounts trigger immune response"
    ]
)


# ============================================================================
# INFLAMMATORY BOWEL DISEASE (Crohn's & Ulcerative Colitis)
# ============================================================================

IBD = HealthConditionProfile(
    condition_id="inflammatory_bowel_disease",
    condition_name="Inflammatory Bowel Disease (Crohn's Disease & Ulcerative Colitis)",
    nutrient_requirements=[
        ConditionNutrientRequirement(
            nutrient_id="vitamin_d",
            nutrient_name="Vitamin D",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="IU",
            rationale="Modulates immune system, reduces inflammation. Deficiency very common in IBD (70%). Low levels linked to worse disease activity",
            food_sources=["fatty fish", "fortified milk", "egg yolks", "supplements (often need 2000-4000 IU)"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="iron",
            nutrient_name="Iron",
            recommendation_type=NutrientRecommendation.MONITOR,
            target_amount=Decimal("18"),
            target_unit="mg",
            rationale="Anemia extremely common in IBD (blood loss, malabsorption, inflammation). Check ferritin. IV iron often needed if severe",
            food_sources=["lean meats", "beans", "spinach", "fortified cereals", "supplements/IV if deficient"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="vitamin_b12",
            nutrient_name="Vitamin B12",
            recommendation_type=NutrientRecommendation.MONITOR,
            target_amount=Decimal("1000"),
            target_unit="mcg",
            rationale="Malabsorption common, especially in Crohn's (ileum affected). Check levels annually. May need injections",
            food_sources=["meat", "fish", "dairy", "fortified cereals", "sublingual or injectable supplements if deficient"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="folate",
            nutrient_name="Folate",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("400"),
            target_unit="mcg",
            rationale="Malabsorption common. Critical if on sulfasalazine (depletes folate). Extra important if on methotrexate",
            food_sources=["leafy greens", "legumes", "fortified grains", "supplements"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="calcium",
            nutrient_name="Calcium",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("1200"),
            target_unit="mg",
            rationale="Malabsorption common. Corticosteroids increase bone loss. IBD patients at high osteoporosis risk",
            food_sources=["dairy if tolerated", "fortified plant milk", "leafy greens", "supplements"],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="zinc",
            nutrient_name="Zinc",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("15"),
            target_unit="mg",
            rationale="Lost through diarrhea. Deficiency common. Needed for healing and immune function",
            food_sources=["oysters", "beef", "pumpkin seeds", "chickpeas", "supplements"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="omega_3_fatty_acids",
            nutrient_name="Omega-3 Fatty Acids (EPA + DHA)",
            recommendation_type=NutrientRecommendation.INCREASE,
            target_amount=Decimal("2000"),
            target_unit="mg",
            rationale="Anti-inflammatory. May help maintain remission. Studies show benefit in UC",
            food_sources=["fatty fish if tolerated", "fish oil supplements", "algae-based omega-3"],
            priority=2
        ),
        ConditionNutrientRequirement(
            nutrient_id="fiber",
            nutrient_name="Dietary Fiber",
            recommendation_type=NutrientRecommendation.MONITOR,
            rationale="COMPLEX: High fiber during flares worsens symptoms. Low fiber during remission may increase relapse risk. INDIVIDUALIZE based on disease activity",
            food_sources=[],
            priority=1
        ),
        ConditionNutrientRequirement(
            nutrient_id="lactose",
            nutrient_name="Lactose",
            recommendation_type=NutrientRecommendation.MONITOR,
            rationale="Lactose intolerance more common in IBD. May worsen symptoms. Trial elimination if suspicious",
            food_sources=[],
            priority=2
        )
    ],
    food_restrictions=[
        FoodRestriction(
            food_or_category="High-fiber foods (DURING FLARES ONLY)",
            reason="During active inflammation: raw vegetables, nuts, seeds, whole grains can worsen symptoms, cause obstruction (especially Crohn's)",
            severity="limit",
            alternatives=["well-cooked vegetables", "white rice", "refined grains", "smooth nut butters", "during remission: gradually increase fiber"]
        ),
        FoodRestriction(
            food_or_category="Dairy (if lactose intolerant)",
            reason="Lactose intolerance common in IBD. May cause gas, bloating, diarrhea",
            severity="monitor",
            alternatives=["lactose-free dairy", "plant-based milk", "lactase enzyme supplements", "yogurt (lower lactose)"]
        ),
        FoodRestriction(
            food_or_category="High-fat and fried foods",
            reason="Fat malabsorption common in Crohn's. High-fat foods may worsen diarrhea",
            severity="limit",
            alternatives=["lean proteins", "baked or grilled foods", "moderate healthy fats"]
        ),
        FoodRestriction(
            food_or_category="Spicy foods",
            reason="May irritate inflamed intestines, worsen symptoms in some individuals",
            severity="monitor",
            alternatives=["mild seasonings", "herbs", "track individual tolerance"]
        ),
        FoodRestriction(
            food_or_category="Alcohol and caffeine",
            reason="Can irritate GI tract, worsen diarrhea, interact with medications",
            severity="limit",
            alternatives=["limit or avoid", "herbal teas", "decaf options"]
        ),
        FoodRestriction(
            food_or_category="Popcorn, nuts, seeds (if strictures present)",
            reason="Can cause obstruction if intestinal narrowing (strictures) present, especially in Crohn's",
            severity="must_avoid",
            alternatives=["smooth nut butters", "well-cooked foods", "avoid if strictures diagnosed"]
        ),
        FoodRestriction(
            food_or_category="FODMAPs (trial elimination)",
            reason="High-FODMAP foods may worsen symptoms (gas, bloating, diarrhea). Consider low-FODMAP trial with dietitian",
            severity="monitor",
            alternatives=["low-FODMAP alternatives", "work with registered dietitian", "reintroduce systematically"]
        )
    ],
    recommended_foods=[
        "Well-cooked vegetables (easier to digest during flares)",
        "Lean proteins (chicken, fish, eggs, tofu - easy to digest)",
        "White rice and refined grains (during flares - easier to digest)",
        "Bananas (easy to digest, potassium, pectin)",
        "Applesauce (easy to digest, pectin)",
        "Oatmeal (soluble fiber, gentle)",
        "Smooth nut butters (protein, calories)",
        "Yogurt (probiotics, calcium, protein - if tolerate dairy)",
        "Omega-3 rich fish (salmon, mackerel - if tolerate)",
        "Bone broth (nutrients, easy to digest, healing)"
    ],
    recommended_diet_patterns=[
        "Specific Carbohydrate Diet (SCD) - eliminates complex carbs, some evidence for IBD",
        "Low-FODMAP Diet (trial with dietitian - may reduce symptoms)",
        "Mediterranean Diet (during remission - anti-inflammatory)",
        "Semi-Vegetarian Diet (some evidence for maintaining Crohn's remission)",
        "INDIVIDUALIZED diet based on disease activity, strictures, symptoms"
    ],
    lifestyle_recommendations=[
        "NUTRITION SUPPORT: Work with IBD-specialized dietitian. Malnutrition common",
        "FLARE vs REMISSION DIET: Adjust diet based on disease activity. More restrictive during flares",
        "SMALL FREQUENT MEALS: Easier to digest, better nutrient absorption",
        "FOOD DIARY: Track foods and symptoms to identify personal triggers",
        "HYDRATION: CRITICAL especially during diarrhea. Electrolyte replacement",
        "SUPPLEMENT WISELY: Monitor and replace deficiencies (iron, B12, vitamin D, calcium)",
        "STRESS MANAGEMENT: Stress can trigger flares. Therapy, meditation, support groups",
        "ADEQUATE REST: Fatigue common. Prioritize sleep",
        "EXERCISE: When possible, improves bone health, reduces stress. Adapt to energy levels",
        "SMOKING CESSATION: Smoking worsens Crohn's (oddly, may help UC - but don't smoke!)",
        "MEDICATION ADHERENCE: Critical for maintaining remission",
        "REGULAR MONITORING: Colonoscopies, blood tests, bone density scans"
    ],
    medication_interactions=[
        "CORTICOSTEROIDS (Prednisone, Budesonide): Cause bone loss (need calcium 1200mg + vitamin D), increase blood sugar, sodium retention. Take with food",
        "IMMUNOSUPPRESSANTS (Azathioprine, 6-MP): Increase infection risk. May cause nausea (take at bedtime). Monitor blood counts",
        "BIOLOGICS (Remicade, Humira, Entyvio): Increase infection risk. Avoid raw/undercooked foods. Can't take live vaccines",
        "METHOTREXATE: MUST take folic acid 1mg daily. AVOID alcohol (liver toxicity). Take with food",
        "SULFASALAZINE: Depletes folate - supplement 1mg daily. May cause nausea - take with food. Can color urine/contact lenses orange",
        "IRON SUPPLEMENTS: Oral iron may worsen GI symptoms. Consider IV iron if not tolerated or severe deficiency",
        "PROBIOTICS: Generally safe. May help some patients. VSL#3 has evidence for UC"
    ]
)


# ============================================================================
# REGISTRY - All Autoimmune Conditions
# ============================================================================

AUTOIMMUNE_CONDITIONS = {
    "lupus": LUPUS,
    "multiple_sclerosis": MULTIPLE_SCLEROSIS,
    "psoriasis": PSORIASIS,
    "hashimotos_thyroiditis": HASHIMOTOS,
    "inflammatory_bowel_disease": IBD
}


def get_autoimmune_condition(condition_id: str) -> Optional[HealthConditionProfile]:
    """Retrieve an autoimmune condition profile by ID"""
    return AUTOIMMUNE_CONDITIONS.get(condition_id)


def list_autoimmune_conditions() -> List[str]:
    """Get list of all available autoimmune condition IDs"""
    return list(AUTOIMMUNE_CONDITIONS.keys())


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 3C: AUTOIMMUNE DISEASES - COMPREHENSIVE PROFILES")
    print("=" * 80)
    print()
    
    for condition_id, profile in AUTOIMMUNE_CONDITIONS.items():
        print(f"\n{'=' * 80}")
        print(f"CONDITION: {profile.condition_name}")
        print(f"ID: {profile.condition_id}")
        print("=" * 80)
        
        print(f"\n📊 NUTRIENT REQUIREMENTS ({len(profile.nutrient_requirements)} nutrients):")
        for req in profile.nutrient_requirements[:5]:
            symbol = "⬆️" if req.recommendation_type == NutrientRecommendation.INCREASE else "⬇️" if req.recommendation_type == NutrientRecommendation.DECREASE else "⚠️"
            target = f"{req.target_amount}{req.target_unit}" if req.target_amount else "Monitor"
            print(f"  {symbol} {req.nutrient_name}: {req.recommendation_type.value.upper()}")
            print(f"     Target: {target} | Priority: {req.priority}")
        
        print(f"\n🚫 FOOD RESTRICTIONS ({len(profile.food_restrictions)} restrictions):")
        for restriction in profile.food_restrictions[:3]:
            severity_icon = "❌" if restriction.severity == "must_avoid" else "⚠️" if restriction.severity == "limit" else "👁️"
            print(f"  {severity_icon} {restriction.food_or_category}")
            print(f"     Severity: {restriction.severity.upper()}")
        
        print(f"\n✅ RECOMMENDED FOODS: {len(profile.recommended_foods)} foods")
        print(f"🍽️ DIET PATTERNS: {len(profile.recommended_diet_patterns)} patterns")
        print(f"💊 MEDICATION INTERACTIONS: {len(profile.medication_interactions)} interactions")
        print(f"💡 LIFESTYLE RECOMMENDATIONS: {len(profile.lifestyle_recommendations)} recommendations")
        print()
    
    print("\n" + "=" * 80)
    print("✅ PHASE 3C COMPLETE - 5 Autoimmune Conditions Fully Mapped")
    print("=" * 80)
    print(f"\nTotal Conditions: {len(AUTOIMMUNE_CONDITIONS)}")
    print("Conditions:", ", ".join([p.condition_name for p in AUTOIMMUNE_CONDITIONS.values()]))
    print("\nKey Features:")
    print("  ✓ Anti-inflammatory protocols (omega-3, vitamin D)")
    print("  ✓ Elimination diet guidance (gluten, dairy trials)")
    print("  ✓ Medication interactions (corticosteroids, immunosuppressants)")
    print("  ✓ Gut health protocols")
    print("  ✓ Autoimmune-specific nutrition")
