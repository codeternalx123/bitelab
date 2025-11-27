"""
Health Condition Matcher - Maps medical conditions to nutritional requirements
Integrates with comprehensive disease database (40+ conditions) and provides 
condition-specific dietary recommendations, nutrient needs, and foods to avoid.

Part of Phase 3: Intelligent Meal Planning System

INTEGRATED MODULES:
- Phase 3A: Cardiovascular (4 conditions)
- Phase 3B: Metabolic (5 conditions)  
- Phase 3C: Autoimmune (5 conditions)
- Original: Type 2 Diabetes, Hypertension, Celiac, RA, CKD (5 conditions)

Total: 19+ conditions available
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from decimal import Decimal
from enum import Enum

# Import condition modules from phases
try:
    from .health_conditions_cardiovascular import CARDIOVASCULAR_CONDITIONS
    CARDIOVASCULAR_AVAILABLE = True
except ImportError:
    CARDIOVASCULAR_AVAILABLE = False
    CARDIOVASCULAR_CONDITIONS = {}

try:
    from .health_conditions_metabolic import METABOLIC_CONDITIONS
    METABOLIC_AVAILABLE = True
except ImportError:
    METABOLIC_AVAILABLE = False
    METABOLIC_CONDITIONS = {}

try:
    from .health_conditions_autoimmune import AUTOIMMUNE_CONDITIONS
    AUTOIMMUNE_AVAILABLE = True
except ImportError:
    AUTOIMMUNE_AVAILABLE = False
    AUTOIMMUNE_CONDITIONS = {}

try:
    from .health_conditions_bone_joint import BONE_JOINT_CONDITIONS
    BONE_JOINT_AVAILABLE = True
except ImportError:
    BONE_JOINT_AVAILABLE = False
    BONE_JOINT_CONDITIONS = {}


class NutrientRecommendation(Enum):
    """Types of nutrient recommendations"""
    INCREASE = "increase"  # Need more of this nutrient
    DECREASE = "decrease"  # Need less of this nutrient
    MAINTAIN = "maintain"  # Standard RDA
    AVOID = "avoid"  # Must avoid this nutrient
    MONITOR = "monitor"  # Watch intake levels


@dataclass
class ConditionNutrientRequirement:
    """Nutrient requirement for a specific health condition"""
    nutrient_id: str
    nutrient_name: str
    recommendation_type: NutrientRecommendation
    target_amount: Optional[Decimal] = None  # Daily target in appropriate units
    target_unit: Optional[str] = None  # mg, mcg, g, IU, etc.
    rationale: str = ""  # Why this is recommended
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
    
    # Nutrient modifications
    nutrient_requirements: List[ConditionNutrientRequirement] = field(default_factory=list)
    
    # Foods to avoid or limit
    food_restrictions: List[FoodRestriction] = field(default_factory=list)
    
    # Beneficial foods
    recommended_foods: List[str] = field(default_factory=list)
    
    # Dietary patterns
    recommended_diet_patterns: List[str] = field(default_factory=list)  # Mediterranean, DASH, etc.
    
    # General recommendations
    lifestyle_recommendations: List[str] = field(default_factory=list)
    
    # Drug-nutrient interactions
    medication_interactions: List[str] = field(default_factory=list)


class HealthConditionMatcher:
    """
    Matches health conditions to specific nutritional requirements.
    Provides personalized dietary recommendations based on medical conditions.
    
    Integrates condition profiles from multiple phase modules:
    - Original 5 conditions (diabetes, hypertension, celiac, RA, CKD)
    - Phase 3A: Cardiovascular (4 conditions)
    - Phase 3B: Metabolic (5 conditions)
    - Phase 3C: Autoimmune (5 conditions)
    - Phase 3D: Bone & Joint (4 conditions)
    
    Total: 19+ comprehensive condition profiles
    """
    
    def __init__(self):
        """Initialize with condition database"""
        self.condition_profiles: Dict[str, HealthConditionProfile] = {}
        self._initialize_condition_database()
        self._load_phase_modules()
    
    def _load_phase_modules(self):
        """Load conditions from phase modules"""
        # Load Phase 3A: Cardiovascular
        if CARDIOVASCULAR_AVAILABLE:
            self.condition_profiles.update(CARDIOVASCULAR_CONDITIONS)
            print(f"✅ Loaded {len(CARDIOVASCULAR_CONDITIONS)} cardiovascular conditions")
        
        # Load Phase 3B: Metabolic
        if METABOLIC_AVAILABLE:
            self.condition_profiles.update(METABOLIC_CONDITIONS)
            print(f"✅ Loaded {len(METABOLIC_CONDITIONS)} metabolic conditions")
        
        # Load Phase 3C: Autoimmune
        if AUTOIMMUNE_AVAILABLE:
            self.condition_profiles.update(AUTOIMMUNE_CONDITIONS)
            print(f"✅ Loaded {len(AUTOIMMUNE_CONDITIONS)} autoimmune conditions")
        
        # Load Phase 3D: Bone & Joint
        if BONE_JOINT_AVAILABLE:
            self.condition_profiles.update(BONE_JOINT_CONDITIONS)
            print(f"✅ Loaded {len(BONE_JOINT_CONDITIONS)} bone & joint conditions")
    
    def _initialize_condition_database(self):
        """Initialize database of original 5 health conditions"""
        
        # Type 2 Diabetes
        self.condition_profiles["type_2_diabetes"] = HealthConditionProfile(
            condition_id="type_2_diabetes",
            condition_name="Type 2 Diabetes",
            nutrient_requirements=[
                ConditionNutrientRequirement(
                    nutrient_id="fiber",
                    nutrient_name="Dietary Fiber",
                    recommendation_type=NutrientRecommendation.INCREASE,
                    target_amount=Decimal("30"),
                    target_unit="g",
                    rationale="Slows glucose absorption, improves glycemic control",
                    food_sources=["whole grains", "legumes", "vegetables", "chia seeds"],
                    priority=1
                ),
                ConditionNutrientRequirement(
                    nutrient_id="chromium",
                    nutrient_name="Chromium",
                    recommendation_type=NutrientRecommendation.INCREASE,
                    target_amount=Decimal("200"),
                    target_unit="mcg",
                    rationale="Enhances insulin sensitivity",
                    food_sources=["broccoli", "whole grains", "green beans"],
                    priority=2
                ),
                ConditionNutrientRequirement(
                    nutrient_id="magnesium",
                    nutrient_name="Magnesium",
                    recommendation_type=NutrientRecommendation.INCREASE,
                    target_amount=Decimal("400"),
                    target_unit="mg",
                    rationale="Improves glucose metabolism, often deficient in diabetes",
                    food_sources=["spinach", "almonds", "black beans", "avocado"],
                    priority=2
                ),
                ConditionNutrientRequirement(
                    nutrient_id="added_sugars",
                    nutrient_name="Added Sugars",
                    recommendation_type=NutrientRecommendation.DECREASE,
                    target_amount=Decimal("25"),
                    target_unit="g",
                    rationale="Prevents blood sugar spikes",
                    priority=1
                ),
                ConditionNutrientRequirement(
                    nutrient_id="refined_carbs",
                    nutrient_name="Refined Carbohydrates",
                    recommendation_type=NutrientRecommendation.DECREASE,
                    rationale="High glycemic index causes rapid glucose rise",
                    priority=1
                )
            ],
            food_restrictions=[
                FoodRestriction(
                    food_or_category="White bread, white rice, pastries",
                    reason="High glycemic index causes blood sugar spikes",
                    severity="limit",
                    alternatives=["whole grain bread", "brown rice", "quinoa", "oats"]
                ),
                FoodRestriction(
                    food_or_category="Sugary drinks (soda, juice, sweetened beverages)",
                    reason="Rapid blood sugar elevation without fiber",
                    severity="must_avoid",
                    alternatives=["water", "unsweetened tea", "sparkling water with lemon"]
                ),
                FoodRestriction(
                    food_or_category="Candy, cookies, cakes",
                    reason="High in added sugars and refined flour",
                    severity="limit",
                    alternatives=["fresh fruit", "dark chocolate (70%+)", "nuts"]
                )
            ],
            recommended_foods=[
                "Non-starchy vegetables (broccoli, spinach, peppers)",
                "Lean proteins (chicken, fish, tofu)",
                "Legumes (lentils, chickpeas, black beans)",
                "Whole grains (quinoa, oats, barley)",
                "Healthy fats (olive oil, avocado, nuts)",
                "Low-glycemic fruits (berries, apples, pears)"
            ],
            recommended_diet_patterns=["Mediterranean Diet", "DASH Diet", "Low-Glycemic Diet"],
            lifestyle_recommendations=[
                "Eat meals at consistent times to stabilize blood sugar",
                "Include protein and fiber with every meal",
                "Portion control for carbohydrates",
                "Regular physical activity (150 min/week)",
                "Monitor blood glucose levels",
                "Stay hydrated (8+ glasses water/day)"
            ],
            medication_interactions=[
                "Metformin may reduce vitamin B12 absorption - supplement if needed",
                "Avoid excessive alcohol (can cause hypoglycemia)",
                "Grapefruit may interact with some diabetes medications"
            ]
        )
        
        # Hypertension (High Blood Pressure)
        self.condition_profiles["hypertension"] = HealthConditionProfile(
            condition_id="hypertension",
            condition_name="Hypertension (High Blood Pressure)",
            nutrient_requirements=[
                ConditionNutrientRequirement(
                    nutrient_id="sodium",
                    nutrient_name="Sodium",
                    recommendation_type=NutrientRecommendation.DECREASE,
                    target_amount=Decimal("1500"),
                    target_unit="mg",
                    rationale="Reduces blood pressure by 5-6 mmHg",
                    priority=1
                ),
                ConditionNutrientRequirement(
                    nutrient_id="potassium",
                    nutrient_name="Potassium",
                    recommendation_type=NutrientRecommendation.INCREASE,
                    target_amount=Decimal("4700"),
                    target_unit="mg",
                    rationale="Counteracts sodium, relaxes blood vessels",
                    food_sources=["bananas", "sweet potatoes", "spinach", "avocado", "white beans"],
                    priority=1
                ),
                ConditionNutrientRequirement(
                    nutrient_id="magnesium",
                    nutrient_name="Magnesium",
                    recommendation_type=NutrientRecommendation.INCREASE,
                    target_amount=Decimal("500"),
                    target_unit="mg",
                    rationale="Relaxes blood vessels, reduces blood pressure",
                    food_sources=["pumpkin seeds", "spinach", "dark chocolate", "almonds"],
                    priority=2
                ),
                ConditionNutrientRequirement(
                    nutrient_id="calcium",
                    nutrient_name="Calcium",
                    recommendation_type=NutrientRecommendation.MAINTAIN,
                    target_amount=Decimal("1000"),
                    target_unit="mg",
                    rationale="May help lower blood pressure",
                    food_sources=["dairy", "leafy greens", "fortified foods"],
                    priority=2
                )
            ],
            food_restrictions=[
                FoodRestriction(
                    food_or_category="Processed/packaged foods",
                    reason="Very high sodium content (75% of dietary sodium)",
                    severity="limit",
                    alternatives=["fresh whole foods", "home-cooked meals"]
                ),
                FoodRestriction(
                    food_or_category="Cured meats (bacon, deli meats, hot dogs)",
                    reason="Extremely high in sodium",
                    severity="must_avoid",
                    alternatives=["fresh lean meats", "grilled chicken", "fish"]
                ),
                FoodRestriction(
                    food_or_category="Canned soups and vegetables",
                    reason="High sodium unless labeled 'no salt added'",
                    severity="limit",
                    alternatives=["fresh or frozen vegetables", "low-sodium soups"]
                ),
                FoodRestriction(
                    food_or_category="Alcohol",
                    reason="Raises blood pressure, max 1-2 drinks/day",
                    severity="limit",
                    alternatives=["sparkling water", "herbal tea"]
                )
            ],
            recommended_foods=[
                "Fruits (especially berries, bananas)",
                "Vegetables (especially leafy greens, beets)",
                "Whole grains (oats, quinoa, brown rice)",
                "Low-fat dairy (yogurt, milk)",
                "Fatty fish (salmon, mackerel - omega-3s)",
                "Nuts and seeds (unsalted)",
                "Beans and legumes"
            ],
            recommended_diet_patterns=["DASH Diet (Dietary Approaches to Stop Hypertension)", "Mediterranean Diet"],
            lifestyle_recommendations=[
                "Limit sodium to 1500mg/day (ideal) or <2300mg/day",
                "Increase potassium-rich foods",
                "Maintain healthy weight (lose 5-10% if overweight)",
                "Regular aerobic exercise (30 min most days)",
                "Limit alcohol (≤1 drink/day women, ≤2 men)",
                "Stress management techniques",
                "Read food labels carefully for sodium content"
            ],
            medication_interactions=[
                "Potassium-sparing diuretics: avoid excess potassium supplements",
                "ACE inhibitors: monitor potassium levels",
                "Grapefruit may interact with calcium channel blockers"
            ]
        )
        
        # Celiac Disease
        self.condition_profiles["celiac_disease"] = HealthConditionProfile(
            condition_id="celiac_disease",
            condition_name="Celiac Disease",
            nutrient_requirements=[
                ConditionNutrientRequirement(
                    nutrient_id="iron",
                    nutrient_name="Iron",
                    recommendation_type=NutrientRecommendation.INCREASE,
                    target_amount=Decimal("18"),
                    target_unit="mg",
                    rationale="Common deficiency due to malabsorption",
                    food_sources=["red meat", "spinach", "lentils", "fortified GF cereals"],
                    priority=1
                ),
                ConditionNutrientRequirement(
                    nutrient_id="calcium",
                    nutrient_name="Calcium",
                    recommendation_type=NutrientRecommendation.INCREASE,
                    target_amount=Decimal("1200"),
                    target_unit="mg",
                    rationale="Malabsorption causes deficiency, bone health risk",
                    food_sources=["dairy", "fortified GF milk", "leafy greens"],
                    priority=1
                ),
                ConditionNutrientRequirement(
                    nutrient_id="vitamin_d",
                    nutrient_name="Vitamin D",
                    recommendation_type=NutrientRecommendation.INCREASE,
                    target_amount=Decimal("1000"),
                    target_unit="IU",
                    rationale="Malabsorption, bone health",
                    food_sources=["fatty fish", "fortified GF milk", "egg yolks", "sun exposure"],
                    priority=1
                ),
                ConditionNutrientRequirement(
                    nutrient_id="folate",
                    nutrient_name="Folate",
                    recommendation_type=NutrientRecommendation.INCREASE,
                    target_amount=Decimal("400"),
                    target_unit="mcg",
                    rationale="Deficiency common, not in GF grains",
                    food_sources=["leafy greens", "legumes", "fortified GF foods"],
                    priority=2
                ),
                ConditionNutrientRequirement(
                    nutrient_id="fiber",
                    nutrient_name="Dietary Fiber",
                    recommendation_type=NutrientRecommendation.INCREASE,
                    target_amount=Decimal("25"),
                    target_unit="g",
                    rationale="GF diet often lower in fiber",
                    food_sources=["GF whole grains", "fruits", "vegetables", "chia seeds"],
                    priority=2
                ),
                ConditionNutrientRequirement(
                    nutrient_id="gluten",
                    nutrient_name="Gluten (wheat, barley, rye)",
                    recommendation_type=NutrientRecommendation.AVOID,
                    target_amount=Decimal("0"),
                    target_unit="mg",
                    rationale="Triggers immune response, damages intestinal lining",
                    priority=1
                )
            ],
            food_restrictions=[
                FoodRestriction(
                    food_or_category="Wheat, barley, rye, and derivatives",
                    reason="Contains gluten - triggers autoimmune response",
                    severity="must_avoid",
                    alternatives=["rice", "quinoa", "corn", "certified GF oats", "buckwheat"]
                ),
                FoodRestriction(
                    food_or_category="Most breads, pastas, cereals, baked goods",
                    reason="Made with wheat flour (gluten)",
                    severity="must_avoid",
                    alternatives=["GF bread", "GF pasta", "GF oats", "rice cakes"]
                ),
                FoodRestriction(
                    food_or_category="Beer, ales, lagers",
                    reason="Made from barley (gluten)",
                    severity="must_avoid",
                    alternatives=["wine", "GF beer", "cider", "spirits"]
                ),
                FoodRestriction(
                    food_or_category="Processed foods with hidden gluten",
                    reason="Gluten used as filler in sauces, seasonings, etc.",
                    severity="monitor",
                    alternatives=["whole foods", "certified GF products"]
                )
            ],
            recommended_foods=[
                "Naturally GF whole grains (quinoa, rice, millet, teff)",
                "Fresh fruits and vegetables",
                "Legumes (beans, lentils)",
                "Meat, fish, eggs (unprocessed)",
                "Dairy (plain, unprocessed)",
                "Nuts and seeds",
                "Certified GF oats"
            ],
            recommended_diet_patterns=["Strict Gluten-Free Diet", "Whole Foods Based"],
            lifestyle_recommendations=[
                "STRICT gluten avoidance - even trace amounts cause damage",
                "Read all food labels for gluten-containing ingredients",
                "Avoid cross-contamination (separate cutting boards, toasters)",
                "Choose certified gluten-free products (< 20ppm gluten)",
                "Increase nutrient-dense foods to compensate for malabsorption",
                "Consider supplements (iron, calcium, vitamin D, B12) if deficient",
                "Annual screening for nutrient deficiencies",
                "Work with dietitian experienced in celiac disease"
            ],
            medication_interactions=[
                "Check medications and supplements for gluten (fillers, coatings)",
                "Malabsorption may affect drug absorption - monitor effectiveness",
                "Some vitamins contain gluten - choose GF brands"
            ]
        )
        
        # Rheumatoid Arthritis
        self.condition_profiles["rheumatoid_arthritis"] = HealthConditionProfile(
            condition_id="rheumatoid_arthritis",
            condition_name="Rheumatoid Arthritis",
            nutrient_requirements=[
                ConditionNutrientRequirement(
                    nutrient_id="omega_3",
                    nutrient_name="Omega-3 Fatty Acids (EPA+DHA)",
                    recommendation_type=NutrientRecommendation.INCREASE,
                    target_amount=Decimal("2000"),
                    target_unit="mg",
                    rationale="Reduces inflammation, decreases joint pain and stiffness",
                    food_sources=["fatty fish", "fish oil", "algae oil", "walnuts", "flaxseed"],
                    priority=1
                ),
                ConditionNutrientRequirement(
                    nutrient_id="vitamin_d",
                    nutrient_name="Vitamin D",
                    recommendation_type=NutrientRecommendation.INCREASE,
                    target_amount=Decimal("2000"),
                    target_unit="IU",
                    rationale="Immune modulation, bone health (often deficient)",
                    food_sources=["fatty fish", "fortified dairy", "egg yolks", "sun exposure"],
                    priority=1
                ),
                ConditionNutrientRequirement(
                    nutrient_id="selenium",
                    nutrient_name="Selenium",
                    recommendation_type=NutrientRecommendation.INCREASE,
                    target_amount=Decimal("200"),
                    target_unit="mcg",
                    rationale="Antioxidant, may reduce inflammation",
                    food_sources=["Brazil nuts", "fish", "turkey", "eggs"],
                    priority=2
                ),
                ConditionNutrientRequirement(
                    nutrient_id="omega_6",
                    nutrient_name="Omega-6 Fatty Acids (excess)",
                    recommendation_type=NutrientRecommendation.DECREASE,
                    rationale="Pro-inflammatory when ratio to omega-3 is high",
                    priority=2
                )
            ],
            food_restrictions=[
                FoodRestriction(
                    food_or_category="Fried foods, processed vegetable oils",
                    reason="High in omega-6, promotes inflammation",
                    severity="limit",
                    alternatives=["olive oil", "avocado oil", "baked/grilled foods"]
                ),
                FoodRestriction(
                    food_or_category="Red meat, full-fat dairy",
                    reason="High in saturated fat, may increase inflammation",
                    severity="limit",
                    alternatives=["fish", "poultry", "plant proteins", "low-fat dairy"]
                ),
                FoodRestriction(
                    food_or_category="Refined carbohydrates, added sugars",
                    reason="May trigger inflammatory response",
                    severity="limit",
                    alternatives=["whole grains", "fruits", "vegetables"]
                ),
                FoodRestriction(
                    food_or_category="Nightshades (tomatoes, peppers, eggplant) - individual",
                    reason="Some people report increased inflammation (not proven)",
                    severity="monitor",
                    alternatives=["try elimination diet to test individual response"]
                )
            ],
            recommended_foods=[
                "Fatty fish (salmon, mackerel, sardines) - 2-3x/week",
                "Colorful fruits and vegetables (antioxidants)",
                "Extra virgin olive oil",
                "Nuts and seeds (especially walnuts, chia, flax)",
                "Whole grains",
                "Green tea (anti-inflammatory)",
                "Turmeric with black pepper (curcumin)",
                "Ginger (anti-inflammatory)"
            ],
            recommended_diet_patterns=["Mediterranean Diet", "Anti-Inflammatory Diet"],
            lifestyle_recommendations=[
                "Eat fatty fish 2-3 times per week or take fish oil supplement",
                "Increase omega-3 to omega-6 ratio (aim for 1:4 or better)",
                "Emphasize antioxidant-rich colorful produce",
                "Maintain healthy weight (reduces joint stress)",
                "Regular low-impact exercise (swimming, yoga)",
                "Consider elimination diet to identify trigger foods",
                "Adequate rest and stress management"
            ],
            medication_interactions=[
                "Methotrexate: take folic acid supplement (1mg/day)",
                "NSAIDs: may cause GI issues - take with food",
                "Fish oil: may thin blood - inform doctor before surgery",
                "Some immunosuppressants: avoid grapefruit"
            ]
        )
        
        # Chronic Kidney Disease (CKD)
        self.condition_profiles["chronic_kidney_disease"] = HealthConditionProfile(
            condition_id="chronic_kidney_disease",
            condition_name="Chronic Kidney Disease",
            nutrient_requirements=[
                ConditionNutrientRequirement(
                    nutrient_id="protein",
                    nutrient_name="Protein",
                    recommendation_type=NutrientRecommendation.DECREASE,
                    target_amount=Decimal("0.8"),
                    target_unit="g/kg",
                    rationale="Reduces kidney workload, slows disease progression",
                    priority=1
                ),
                ConditionNutrientRequirement(
                    nutrient_id="sodium",
                    nutrient_name="Sodium",
                    recommendation_type=NutrientRecommendation.DECREASE,
                    target_amount=Decimal("2000"),
                    target_unit="mg",
                    rationale="Reduces fluid retention and blood pressure",
                    priority=1
                ),
                ConditionNutrientRequirement(
                    nutrient_id="potassium",
                    nutrient_name="Potassium",
                    recommendation_type=NutrientRecommendation.DECREASE,
                    target_amount=Decimal("2000"),
                    target_unit="mg",
                    rationale="Kidneys can't excrete excess, risk of hyperkalemia",
                    priority=1
                ),
                ConditionNutrientRequirement(
                    nutrient_id="phosphorus",
                    nutrient_name="Phosphorus",
                    recommendation_type=NutrientRecommendation.DECREASE,
                    target_amount=Decimal("800"),
                    target_unit="mg",
                    rationale="Accumulation causes bone and heart problems",
                    priority=1
                )
            ],
            food_restrictions=[
                FoodRestriction(
                    food_or_category="High-potassium foods (bananas, oranges, potatoes, tomatoes)",
                    reason="Risk of dangerous hyperkalemia",
                    severity="must_avoid",
                    alternatives=["apples", "berries", "grapes", "cauliflower"]
                ),
                FoodRestriction(
                    food_or_category="High-phosphorus foods (dairy, nuts, beans, whole grains)",
                    reason="Kidneys can't remove excess phosphorus",
                    severity="limit",
                    alternatives=["rice milk", "limited portions with phosphate binders"]
                ),
                FoodRestriction(
                    food_or_category="Processed meats, canned foods",
                    reason="Very high in sodium and phosphorus additives",
                    severity="must_avoid",
                    alternatives=["fresh lean meats", "fresh vegetables"]
                )
            ],
            recommended_foods=[
                "Low-potassium fruits (apples, berries, grapes)",
                "Low-potassium vegetables (cabbage, cauliflower, bell peppers)",
                "White rice, white bread (lower phosphorus than whole grains)",
                "Egg whites (protein without phosphorus)",
                "Lean meats in moderation"
            ],
            recommended_diet_patterns=["Renal Diet", "Low Potassium, Low Phosphorus, Low Sodium"],
            lifestyle_recommendations=[
                "Work closely with renal dietitian - very individualized",
                "Protein needs vary by CKD stage (may need more if on dialysis)",
                "Limit phosphorus additives in processed foods",
                "Double-cook potatoes to reduce potassium (soak + boil)",
                "Control blood pressure and blood sugar",
                "Stay hydrated but follow fluid restrictions if advised",
                "Avoid NSAIDs (ibuprofen, naproxen) - kidney damage"
            ],
            medication_interactions=[
                "Phosphate binders: take with meals",
                "Some medications increase potassium - monitor levels",
                "Avoid magnesium-containing antacids",
                "Check with doctor before ANY supplements"
            ]
        )
    
    def get_condition_profile(self, condition_id: str) -> Optional[HealthConditionProfile]:
        """
        Get nutritional profile for a health condition.
        
        Args:
            condition_id: Condition identifier
        
        Returns:
            HealthConditionProfile or None if not found
        """
        return self.condition_profiles.get(condition_id)
    
    def get_combined_requirements(
        self,
        condition_ids: Set[str]
    ) -> Dict[str, List[ConditionNutrientRequirement]]:
        """
        Combine nutritional requirements from multiple conditions.
        Handles conflicts and prioritizes critical requirements.
        
        Args:
            condition_ids: Set of condition IDs
        
        Returns:
            Dictionary of nutrient_id to list of requirements
        """
        combined = {}
        
        for condition_id in condition_ids:
            profile = self.get_condition_profile(condition_id)
            if not profile:
                continue
            
            for req in profile.nutrient_requirements:
                if req.nutrient_id not in combined:
                    combined[req.nutrient_id] = []
                combined[req.nutrient_id].append(req)
        
        return combined
    
    def resolve_conflicts(
        self,
        requirements: List[ConditionNutrientRequirement]
    ) -> ConditionNutrientRequirement:
        """
        Resolve conflicting requirements for the same nutrient.
        Prioritizes based on severity and evidence.
        
        Args:
            requirements: List of requirements for same nutrient
        
        Returns:
            Final recommendation
        """
        # Sort by priority (1=most important)
        requirements.sort(key=lambda r: r.priority)
        
        # If all agree, use first (highest priority)
        if len(set(r.recommendation_type for r in requirements)) == 1:
            return requirements[0]
        
        # Handle conflicts
        # AVOID always takes precedence
        avoid_reqs = [r for r in requirements if r.recommendation_type == NutrientRecommendation.AVOID]
        if avoid_reqs:
            return avoid_reqs[0]
        
        # DECREASE takes precedence over INCREASE (safety first)
        decrease_reqs = [r for r in requirements if r.recommendation_type == NutrientRecommendation.DECREASE]
        if decrease_reqs:
            return decrease_reqs[0]
        
        # Otherwise use highest priority
        return requirements[0]
    
    def get_all_restrictions(
        self,
        condition_ids: Set[str]
    ) -> List[FoodRestriction]:
        """
        Get all food restrictions from multiple conditions.
        
        Args:
            condition_ids: Set of condition IDs
        
        Returns:
            Combined list of food restrictions
        """
        all_restrictions = []
        
        for condition_id in condition_ids:
            profile = self.get_condition_profile(condition_id)
            if profile:
                all_restrictions.extend(profile.food_restrictions)
        
        return all_restrictions
    
    def get_recommended_foods(
        self,
        condition_ids: Set[str]
    ) -> List[str]:
        """
        Get recommended foods from multiple conditions.
        
        Args:
            condition_ids: Set of condition IDs
        
        Returns:
            Combined list of recommended foods
        """
        all_recommended = []
        
        for condition_id in condition_ids:
            profile = self.get_condition_profile(condition_id)
            if profile:
                all_recommended.extend(profile.recommended_foods)
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for food in all_recommended:
            if food not in seen:
                seen.add(food)
                unique.append(food)
        
        return unique
    
    def list_all_conditions(self) -> Dict[str, List[str]]:
        """
        List all available conditions grouped by category.
        
        Returns:
            Dictionary mapping category to list of condition IDs
        """
        categories = {
            "Original": [],
            "Cardiovascular": [],
            "Metabolic": [],
            "Autoimmune": [],
            "Bone & Joint": [],
            "Digestive": [],
            "Neurological": [],
            "Respiratory": [],
            "Liver": []
        }
        
        # Original 5 conditions
        original_conditions = [
            "type_2_diabetes",
            "hypertension", 
            "celiac_disease",
            "rheumatoid_arthritis",
            "chronic_kidney_disease"
        ]
        
        for condition_id in self.condition_profiles.keys():
            if condition_id in original_conditions:
                categories["Original"].append(condition_id)
            elif condition_id in ["coronary_heart_disease", "atherosclerosis", "heart_failure", "atrial_fibrillation"]:
                categories["Cardiovascular"].append(condition_id)
            elif condition_id in ["obesity", "metabolic_syndrome", "hyperlipidemia", "hypothyroidism", "pcos"]:
                categories["Metabolic"].append(condition_id)
            elif condition_id in ["systemic_lupus", "multiple_sclerosis", "psoriasis", "hashimotos_thyroiditis", "inflammatory_bowel_disease"]:
                categories["Autoimmune"].append(condition_id)
            elif condition_id in ["osteoporosis", "osteoarthritis", "gout", "ankylosing_spondylitis"]:
                categories["Bone & Joint"].append(condition_id)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def get_conditions_by_category(self, category: str) -> List[str]:
        """
        Get all condition IDs for a specific category.
        
        Args:
            category: Category name (e.g., "Cardiovascular", "Metabolic")
        
        Returns:
            List of condition IDs in that category
        """
        all_conditions = self.list_all_conditions()
        return all_conditions.get(category, [])
    
    def get_available_condition_count(self) -> int:
        """
        Get total count of available conditions.
        
        Returns:
            Total number of conditions loaded
        """
        return len(self.condition_profiles)
    
    def get_condition_summary(self) -> Dict[str, any]:
        """
        Get comprehensive summary of loaded conditions.
        
        Returns:
            Dictionary with counts by category and total statistics
        """
        conditions_by_category = self.list_all_conditions()
        
        summary = {
            "total_conditions": self.get_available_condition_count(),
            "categories": {},
            "phase_modules_loaded": []
        }
        
        for category, condition_ids in conditions_by_category.items():
            summary["categories"][category] = {
                "count": len(condition_ids),
                "conditions": condition_ids
            }
        
        # Track which phase modules loaded
        if CARDIOVASCULAR_AVAILABLE:
            summary["phase_modules_loaded"].append("Phase 3A: Cardiovascular")
        if METABOLIC_AVAILABLE:
            summary["phase_modules_loaded"].append("Phase 3B: Metabolic")
        if AUTOIMMUNE_AVAILABLE:
            summary["phase_modules_loaded"].append("Phase 3C: Autoimmune")
        if BONE_JOINT_AVAILABLE:
            summary["phase_modules_loaded"].append("Phase 3D: Bone & Joint")
        
        return summary


def test_health_condition_matcher():
    """Test the health condition matcher"""
    print("🧪 Testing Health Condition Matcher")
    print("=" * 60)
    
    matcher = HealthConditionMatcher()
    
    # Test Case 1: Type 2 Diabetes
    print("\n📋 Test Case 1: Type 2 Diabetes")
    print("-" * 60)
    
    diabetes_profile = matcher.get_condition_profile("type_2_diabetes")
    print(f"Condition: {diabetes_profile.condition_name}")
    print(f"\nKey Nutrient Modifications ({len(diabetes_profile.nutrient_requirements)}):")
    for req in diabetes_profile.nutrient_requirements[:3]:
        print(f"  • {req.nutrient_name}: {req.recommendation_type.value}")
        if req.target_amount:
            print(f"    Target: {req.target_amount}{req.target_unit}/day")
        print(f"    Why: {req.rationale}")
    
    print(f"\nFoods to Avoid/Limit:")
    for restriction in diabetes_profile.food_restrictions:
        print(f"  ⚠️ {restriction.food_or_category} ({restriction.severity})")
        print(f"     Reason: {restriction.reason}")
    
    print(f"\nRecommended Diet Pattern: {', '.join(diabetes_profile.recommended_diet_patterns)}")
    
    # Test Case 2: Multiple Conditions (Diabetes + Hypertension)
    print("\n\n📋 Test Case 2: Comorbidities (Diabetes + Hypertension)")
    print("-" * 60)
    
    conditions = {"type_2_diabetes", "hypertension"}
    combined_reqs = matcher.get_combined_requirements(conditions)
    
    print("Combined Nutrient Requirements:")
    for nutrient_id, reqs in list(combined_reqs.items())[:5]:
        final_req = matcher.resolve_conflicts(reqs)
        print(f"  • {final_req.nutrient_name}: {final_req.recommendation_type.value}")
        if len(reqs) > 1:
            print(f"    (From {len(reqs)} conditions - resolved)")
    
    all_restrictions = matcher.get_all_restrictions(conditions)
    print(f"\nTotal Food Restrictions: {len(all_restrictions)}")
    print("Key restrictions:")
    for restriction in all_restrictions[:3]:
        print(f"  ⚠️ {restriction.food_or_category}")
    
    # Test Case 3: Celiac Disease
    print("\n\n📋 Test Case 3: Celiac Disease")
    print("-" * 60)
    
    celiac_profile = matcher.get_condition_profile("celiac_disease")
    print(f"Condition: {celiac_profile.condition_name}")
    print(f"\nCritical Restrictions:")
    for restriction in celiac_profile.food_restrictions:
        if restriction.severity == "must_avoid":
            print(f"  🚫 MUST AVOID: {restriction.food_or_category}")
            print(f"     Alternatives: {', '.join(restriction.alternatives[:3])}")
    
    print(f"\nKey Nutrients to Monitor:")
    critical_nutrients = [r for r in celiac_profile.nutrient_requirements if r.priority == 1]
    for req in critical_nutrients:
        print(f"  • {req.nutrient_name}: {req.rationale}")
    
    print("\n" + "=" * 60)
    print("✅ Health Condition Matcher tests complete!")
    print("\n💡 This matcher provides:")
    print("   • Condition-specific nutrient requirements")
    print("   • Food restrictions and alternatives")
    print("   • Medication-nutrient interactions")
    print("   • Lifestyle recommendations")
    print("   • Conflict resolution for multiple conditions")


if __name__ == "__main__":
    test_health_condition_matcher()
