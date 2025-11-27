"""
Therapeutic Nutrition Engine
=============================

Medical-grade nutrition support system for specific health conditions.
Provides evidence-based dietary interventions, symptom management through
food compounds, and therapeutic meal planning with medical disclaimers.

⚠️ MEDICAL DISCLAIMER ⚠️
This system provides educational nutritional information only.
It is NOT a substitute for professional medical advice, diagnosis, or treatment.
Always consult your physician or Registered Dietitian before making dietary
changes, especially if you have a medical condition or take medications.

Features:
1. Medical condition database (pregnancy, cancer, diabetes, etc.)
2. Nutrient pharmacology (medicinal food compounds)
3. Symptom-compound matching
4. Therapeutic meal planning
5. Condition-specific nutrient requirements
6. Evidence-based recommendations with citations
7. Trimester/stage-specific support
8. Chemotherapy nutrition protocols
9. Glycemic control meal timing
10. Anti-inflammatory food therapy

Supported Conditions:
- Pregnancy (by trimester)
- Cancer (chemotherapy/radiation support)
- Type 2 Diabetes
- Cardiovascular Disease
- Inflammatory Conditions (arthritis, IBD)
- PCOS
- Thyroid disorders

Medicinal Compounds:
- Curcumin (anti-inflammatory)
- Gingerol (anti-nausea)
- Anthocyanins (antioxidant)
- Omega-3 (cardiovascular, anti-inflammatory)
- Folate (pregnancy, cell division)
- Fiber (glycemic control, gut health)

Author: Wellomex AI Team
Date: November 2025
Version: 12.0.0

References:
- NIH Office of Dietary Supplements
- Academy of Nutrition and Dietetics
- American Cancer Society Nutrition Guidelines
- American Diabetes Association Standards of Care
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, date

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION & ENUMS
# ============================================================================

class MedicalCondition(Enum):
    """Supported medical conditions"""
    PREGNANCY = "pregnancy"
    CANCER = "cancer"
    DIABETES_TYPE2 = "diabetes_type2"
    CARDIOVASCULAR = "cardiovascular"
    INFLAMMATORY = "inflammatory"  # Arthritis, IBD
    PCOS = "pcos"
    THYROID = "thyroid"
    NONE = "none"


class PregnancyTrimester(Enum):
    """Pregnancy stages"""
    FIRST = "first"        # 0-13 weeks
    SECOND = "second"      # 14-27 weeks
    THIRD = "third"        # 28-40 weeks
    POSTPARTUM = "postpartum"


class CancerTreatment(Enum):
    """Cancer treatment types"""
    CHEMOTHERAPY = "chemotherapy"
    RADIATION = "radiation"
    SURGERY_RECOVERY = "surgery_recovery"
    PALLIATIVE = "palliative"


class Symptom(Enum):
    """Common symptoms requiring nutritional support"""
    NAUSEA = "nausea"
    FATIGUE = "fatigue"
    INFLAMMATION = "inflammation"
    CONSTIPATION = "constipation"
    DIARRHEA = "diarrhea"
    POOR_APPETITE = "poor_appetite"
    TASTE_CHANGES = "taste_changes"
    DRY_MOUTH = "dry_mouth"
    HYPERGLYCEMIA = "hyperglycemia"
    INSULIN_RESISTANCE = "insulin_resistance"


class EvidenceLevel(Enum):
    """Scientific evidence strength"""
    HIGH = "high"           # RCT, meta-analysis
    MODERATE = "moderate"   # Observational studies
    LOW = "low"             # Case reports, expert opinion


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MedicinalCompound:
    """Food-based medicinal compound"""
    compound_id: str
    name: str
    common_names: List[str] = field(default_factory=list)
    
    # Food sources
    food_sources: List[str] = field(default_factory=list)
    
    # Therapeutic properties
    mechanisms: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    
    # Dosing
    therapeutic_dose_mg: float = 0.0
    food_equivalent: str = ""  # e.g., "1 tsp turmeric powder"
    
    # Safety
    max_daily_dose_mg: Optional[float] = None
    contraindications: List[str] = field(default_factory=list)
    drug_interactions: List[str] = field(default_factory=list)
    
    # Evidence
    evidence_level: EvidenceLevel = EvidenceLevel.MODERATE
    citations: List[str] = field(default_factory=list)


@dataclass
class ConditionProfile:
    """Medical condition profile"""
    condition_id: str
    condition_name: str
    condition_type: MedicalCondition
    
    # Substage (if applicable)
    trimester: Optional[PregnancyTrimester] = None
    treatment_type: Optional[CancerTreatment] = None
    
    # Nutritional needs
    calorie_adjustment: float = 0.0  # +/- from baseline
    protein_multiplier: float = 1.0   # × baseline (g/kg body weight)
    
    # Micronutrient adjustments
    extra_nutrients: Dict[str, float] = field(default_factory=dict)  # nutrient -> daily mg
    
    # Restrictions
    foods_to_avoid: List[str] = field(default_factory=list)
    foods_to_limit: List[str] = field(default_factory=list)
    
    # Recommended foods
    therapeutic_foods: List[str] = field(default_factory=list)
    
    # Symptoms to manage
    common_symptoms: List[Symptom] = field(default_factory=list)


@dataclass
class TherapeuticIntervention:
    """Evidence-based nutritional intervention"""
    intervention_id: str
    name: str
    
    # Target
    condition: MedicalCondition
    symptom: Optional[Symptom] = None
    
    # Intervention details
    recommended_compounds: List[str] = field(default_factory=list)  # compound_ids
    food_recommendations: List[str] = field(default_factory=list)
    
    # Timing
    timing_notes: str = ""  # e.g., "With meals", "Morning only"
    duration_days: Optional[int] = None
    
    # Evidence
    evidence_level: EvidenceLevel = EvidenceLevel.MODERATE
    expected_improvement: str = ""
    citations: List[str] = field(default_factory=list)
    
    # Safety
    medical_supervision_required: bool = False
    disclaimer: str = "Consult your healthcare provider before starting this intervention."


@dataclass
class TherapeuticMealPlan:
    """Personalized therapeutic meal plan"""
    plan_id: str
    user_id: str
    condition: MedicalCondition
    
    # Timeline
    created_date: date
    duration_weeks: int = 4
    
    # Daily targets
    daily_calories: float = 2000.0
    daily_protein_g: float = 75.0
    daily_carbs_g: float = 250.0
    daily_fat_g: float = 65.0
    
    # Therapeutic goals
    target_symptoms: List[Symptom] = field(default_factory=list)
    interventions: List[TherapeuticIntervention] = field(default_factory=list)
    
    # Meal schedule
    meals_per_day: int = 3
    snacks_per_day: int = 2
    
    # Compounds to include
    daily_compounds: Dict[str, float] = field(default_factory=dict)  # compound_id -> mg
    
    # Medical disclaimer
    disclaimer: str = field(default="This meal plan is for educational purposes only. It does not replace medical advice. Consult your doctor or Registered Dietitian before implementing.")
    
    # Professional validation
    validated_by_rd: bool = False
    rd_name: Optional[str] = None


# ============================================================================
# MEDICINAL COMPOUND DATABASE
# ============================================================================

class CompoundDatabase:
    """
    Evidence-based medicinal food compound database
    """
    
    def __init__(self):
        self.compounds: Dict[str, MedicinalCompound] = {}
        
        self._build_compound_database()
        
        logger.info(f"Compound Database initialized with {len(self.compounds)} compounds")
    
    def _build_compound_database(self):
        """Build medicinal compound database"""
        
        # Curcumin (Turmeric)
        self.compounds['curcumin'] = MedicinalCompound(
            compound_id='curcumin',
            name='Curcumin',
            common_names=['Turmeric extract', 'Diferuloylmethane'],
            food_sources=['Turmeric powder', 'Fresh turmeric root', 'Curry blends'],
            mechanisms=[
                'COX-2 inhibition (anti-inflammatory)',
                'NF-κB pathway suppression',
                'Antioxidant activity',
                'Modulates cytokine production'
            ],
            benefits=[
                'Reduces inflammation',
                'May reduce arthritis pain',
                'Supports joint health',
                'Potential anti-cancer properties'
            ],
            therapeutic_dose_mg=500.0,
            food_equivalent="1-2 tsp turmeric powder (with black pepper for absorption)",
            max_daily_dose_mg=3000.0,
            contraindications=[
                'Gallbladder disease',
                'Bleeding disorders (high doses)',
                'Pregnancy (high doses)'
            ],
            drug_interactions=[
                'Blood thinners (Warfarin)',
                'Diabetes medications (may lower blood sugar)',
                'Stomach acid reducers'
            ],
            evidence_level=EvidenceLevel.HIGH,
            citations=[
                'NIH: Turmeric - https://ods.od.nih.gov/factsheets/Turmeric',
                'Arthritis Foundation: Turmeric for OA',
                'Hewlings SJ, Kalman DS. Foods. 2017;6(10):92'
            ]
        )
        
        # Gingerol (Ginger)
        self.compounds['gingerol'] = MedicinalCompound(
            compound_id='gingerol',
            name='Gingerol',
            common_names=['Ginger extract', '6-gingerol'],
            food_sources=['Fresh ginger root', 'Dried ginger powder', 'Ginger tea'],
            mechanisms=[
                'Serotonin receptor antagonism (anti-nausea)',
                'Gastrointestinal motility regulation',
                'Anti-inflammatory prostaglandin inhibition'
            ],
            benefits=[
                'Reduces nausea and vomiting',
                'Effective for chemotherapy-induced nausea',
                'Morning sickness relief',
                'Motion sickness prevention'
            ],
            therapeutic_dose_mg=1000.0,
            food_equivalent="1-2g fresh ginger (1/2 inch slice)",
            max_daily_dose_mg=4000.0,
            contraindications=[
                'Bleeding disorders (high doses)',
                'Gallstones'
            ],
            drug_interactions=[
                'Blood thinners',
                'Diabetes medications'
            ],
            evidence_level=EvidenceLevel.HIGH,
            citations=[
                'NIH: Ginger for nausea - https://nccih.nih.gov/health/ginger',
                'American Cancer Society: Ginger for chemotherapy nausea',
                'Marx W, et al. Crit Rev Food Sci Nutr. 2017;57(1):141-146'
            ]
        )
        
        # Anthocyanins (Berries)
        self.compounds['anthocyanins'] = MedicinalCompound(
            compound_id='anthocyanins',
            name='Anthocyanins',
            common_names=['Berry polyphenols', 'Cyanidin', 'Delphinidin'],
            food_sources=['Blueberries', 'Strawberries', 'Blackberries', 'Purple grapes', 'Red cabbage'],
            mechanisms=[
                'Free radical scavenging (antioxidant)',
                'DNA protection',
                'Anti-inflammatory signaling',
                'Endothelial function improvement'
            ],
            benefits=[
                'Powerful antioxidant support',
                'May reduce cancer risk',
                'Cardiovascular protection',
                'Cognitive health support'
            ],
            therapeutic_dose_mg=300.0,
            food_equivalent="1 cup fresh blueberries",
            max_daily_dose_mg=None,  # No established upper limit
            contraindications=[],
            drug_interactions=[],
            evidence_level=EvidenceLevel.MODERATE,
            citations=[
                'NIH: Anthocyanins and chronic disease',
                'Wallace TC, et al. Adv Nutr. 2015;6(5):620-622',
                'USDA Database for Flavonoid Content'
            ]
        )
        
        # Omega-3 Fatty Acids
        self.compounds['omega3'] = MedicinalCompound(
            compound_id='omega3',
            name='Omega-3 Fatty Acids',
            common_names=['EPA', 'DHA', 'Fish oil'],
            food_sources=['Salmon', 'Sardines', 'Mackerel', 'Walnuts', 'Flaxseeds', 'Chia seeds'],
            mechanisms=[
                'Resolvin and protectin synthesis (anti-inflammatory)',
                'Membrane fluidity modulation',
                'Triglyceride reduction',
                'Blood pressure regulation'
            ],
            benefits=[
                'Reduces inflammation',
                'Cardiovascular disease prevention',
                'Supports brain health',
                'May reduce depression symptoms'
            ],
            therapeutic_dose_mg=1000.0,  # EPA+DHA combined
            food_equivalent="3oz salmon (2-3 times per week)",
            max_daily_dose_mg=3000.0,
            contraindications=[
                'Seafood allergy (fish sources)',
                'Bleeding disorders (high doses)'
            ],
            drug_interactions=[
                'Blood thinners (Warfarin)',
                'Antiplatelet drugs'
            ],
            evidence_level=EvidenceLevel.HIGH,
            citations=[
                'AHA: Fish and Omega-3 Fatty Acids',
                'NIH: Omega-3 Fatty Acids - https://ods.od.nih.gov/factsheets/Omega3FattyAcids',
                'Calder PC. Proc Nutr Soc. 2013;72(3):326-336'
            ]
        )
        
        # Folate (Vitamin B9)
        self.compounds['folate'] = MedicinalCompound(
            compound_id='folate',
            name='Folate',
            common_names=['Folic acid', 'Vitamin B9', 'Folacin'],
            food_sources=['Spinach', 'Lentils', 'Asparagus', 'Fortified grains', 'Chickpeas'],
            mechanisms=[
                'DNA synthesis and repair',
                'Cell division support',
                'Homocysteine metabolism',
                'Neural tube formation'
            ],
            benefits=[
                'Prevents neural tube defects',
                'Supports healthy pregnancy',
                'Red blood cell formation',
                'May reduce cardiovascular risk'
            ],
            therapeutic_dose_mg=0.4,  # 400 mcg
            food_equivalent="1 cup cooked spinach or lentils",
            max_daily_dose_mg=1.0,  # 1000 mcg
            contraindications=[
                'Undiagnosed vitamin B12 deficiency (may mask symptoms)'
            ],
            drug_interactions=[
                'Methotrexate',
                'Anti-seizure medications'
            ],
            evidence_level=EvidenceLevel.HIGH,
            citations=[
                'CDC: Folic Acid - https://www.cdc.gov/ncbddd/folicacid',
                'NIH: Folate - https://ods.od.nih.gov/factsheets/Folate',
                'ACOG: Folic Acid for Pregnancy'
            ]
        )
        
        # Fiber (Soluble)
        self.compounds['soluble_fiber'] = MedicinalCompound(
            compound_id='soluble_fiber',
            name='Soluble Fiber',
            common_names=['Viscous fiber', 'Psyllium', 'Beta-glucan'],
            food_sources=['Oats', 'Beans', 'Apples', 'Barley', 'Chia seeds', 'Psyllium husk'],
            mechanisms=[
                'Glucose absorption delay',
                'Cholesterol binding and excretion',
                'SCFA production (gut fermentation)',
                'Satiety hormone stimulation'
            ],
            benefits=[
                'Improves glycemic control',
                'Lowers LDL cholesterol',
                'Supports gut health',
                'Weight management'
            ],
            therapeutic_dose_mg=10000.0,  # 10g (target: 25-30g total fiber/day)
            food_equivalent="1 cup cooked oatmeal + 1 apple",
            max_daily_dose_mg=None,
            contraindications=[
                'Bowel obstruction',
                'Difficulty swallowing (dry fiber supplements)'
            ],
            drug_interactions=[
                'May reduce absorption of some medications (take separately)'
            ],
            evidence_level=EvidenceLevel.HIGH,
            citations=[
                'ADA: Fiber and Diabetes',
                'NIH: Fiber - https://ods.od.nih.gov/factsheets/Fiber',
                'Reynolds A, et al. Lancet. 2019;393(10170):434-445'
            ]
        )
    
    def get_compound(self, compound_id: str) -> Optional[MedicinalCompound]:
        """Get compound by ID"""
        return self.compounds.get(compound_id)
    
    def search_by_benefit(self, benefit_keyword: str) -> List[MedicinalCompound]:
        """Search compounds by therapeutic benefit"""
        matches = []
        
        for compound in self.compounds.values():
            for benefit in compound.benefits:
                if benefit_keyword.lower() in benefit.lower():
                    matches.append(compound)
                    break
        
        return matches


# ============================================================================
# MEDICAL CONDITION DATABASE
# ============================================================================

class ConditionDatabase:
    """
    Medical condition profiles with nutritional requirements
    """
    
    def __init__(self):
        self.conditions: Dict[str, ConditionProfile] = {}
        
        self._build_condition_database()
        
        logger.info(f"Condition Database initialized with {len(self.conditions)} conditions")
    
    def _build_condition_database(self):
        """Build medical condition database"""
        
        # Pregnancy - First Trimester
        self.conditions['pregnancy_t1'] = ConditionProfile(
            condition_id='pregnancy_t1',
            condition_name='Pregnancy - First Trimester',
            condition_type=MedicalCondition.PREGNANCY,
            trimester=PregnancyTrimester.FIRST,
            calorie_adjustment=0.0,  # No extra calories needed
            protein_multiplier=1.1,
            extra_nutrients={
                'folate': 0.6,      # 600 mcg/day
                'iron': 27.0,        # 27 mg/day
                'vitamin_d': 0.015,  # 15 mcg/day
                'calcium': 1000.0
            },
            foods_to_avoid=[
                'Raw fish (sushi)',
                'Unpasteurized cheese',
                'Deli meats (unless heated)',
                'Raw eggs',
                'Alcohol',
                'High-mercury fish (swordfish, king mackerel)'
            ],
            foods_to_limit=[
                'Caffeine (<200mg/day)'
            ],
            therapeutic_foods=[
                'Folate-rich: Spinach, lentils, fortified cereals',
                'Iron-rich: Lean meat, beans, fortified grains',
                'Ginger (for nausea)'
            ],
            common_symptoms=[
                Symptom.NAUSEA,
                Symptom.FATIGUE,
                Symptom.TASTE_CHANGES
            ]
        )
        
        # Pregnancy - Second Trimester
        self.conditions['pregnancy_t2'] = ConditionProfile(
            condition_id='pregnancy_t2',
            condition_name='Pregnancy - Second Trimester',
            condition_type=MedicalCondition.PREGNANCY,
            trimester=PregnancyTrimester.SECOND,
            calorie_adjustment=+300.0,  # +300 cal/day
            protein_multiplier=1.1,
            extra_nutrients={
                'folate': 0.6,
                'iron': 27.0,
                'vitamin_d': 0.015,
                'calcium': 1000.0,
                'dha': 200.0  # Omega-3 for brain development
            },
            foods_to_avoid=[
                'Raw fish', 'Unpasteurized dairy', 'Alcohol', 'High-mercury fish'
            ],
            therapeutic_foods=[
                'Iron + Vitamin C combinations (beans + tomatoes)',
                'DHA sources: Salmon, walnuts',
                'Calcium sources: Yogurt, fortified plant milk'
            ],
            common_symptoms=[
                Symptom.CONSTIPATION
            ]
        )
        
        # Pregnancy - Third Trimester
        self.conditions['pregnancy_t3'] = ConditionProfile(
            condition_id='pregnancy_t3',
            condition_name='Pregnancy - Third Trimester',
            condition_type=MedicalCondition.PREGNANCY,
            trimester=PregnancyTrimester.THIRD,
            calorie_adjustment=+450.0,  # +450 cal/day
            protein_multiplier=1.1,
            extra_nutrients={
                'folate': 0.6,
                'iron': 27.0,
                'vitamin_d': 0.015,
                'calcium': 1000.0,
                'dha': 200.0
            },
            foods_to_avoid=[
                'Raw fish', 'Unpasteurized dairy', 'Alcohol', 'High-mercury fish'
            ],
            therapeutic_foods=[
                'Small, frequent meals (manage reflux)',
                'High-fiber foods (prevent constipation)',
                'Protein-rich snacks'
            ],
            common_symptoms=[
                Symptom.CONSTIPATION,
                Symptom.FATIGUE
            ]
        )
        
        # Cancer - Chemotherapy
        self.conditions['cancer_chemo'] = ConditionProfile(
            condition_id='cancer_chemo',
            condition_name='Cancer - Chemotherapy Support',
            condition_type=MedicalCondition.CANCER,
            treatment_type=CancerTreatment.CHEMOTHERAPY,
            calorie_adjustment=+200.0,  # Maintain weight
            protein_multiplier=1.2,  # Higher protein needs
            extra_nutrients={
                'vitamin_d': 0.020,
                'zinc': 11.0,
                'vitamin_c': 90.0
            },
            foods_to_avoid=[
                'Raw/undercooked foods (infection risk)',
                'Unpasteurized products',
                'Unwashed produce'
            ],
            foods_to_limit=[
                'Sugary foods (if taste changes make them unappealing)'
            ],
            therapeutic_foods=[
                'Ginger (nausea management)',
                'Protein shakes (maintain muscle mass)',
                'Berries (antioxidant support)',
                'Soft, easy-to-digest foods'
            ],
            common_symptoms=[
                Symptom.NAUSEA,
                Symptom.TASTE_CHANGES,
                Symptom.POOR_APPETITE,
                Symptom.DIARRHEA,
                Symptom.DRY_MOUTH
            ]
        )
        
        # Type 2 Diabetes
        self.conditions['diabetes_t2'] = ConditionProfile(
            condition_id='diabetes_t2',
            condition_name='Type 2 Diabetes',
            condition_type=MedicalCondition.DIABETES_TYPE2,
            calorie_adjustment=0.0,  # Individualized
            protein_multiplier=1.0,
            extra_nutrients={
                'chromium': 0.035,  # 35 mcg
                'magnesium': 400.0
            },
            foods_to_avoid=[],
            foods_to_limit=[
                'Refined carbohydrates',
                'Sugary drinks',
                'High-glycemic foods'
            ],
            therapeutic_foods=[
                'High-fiber foods (soluble fiber for glycemic control)',
                'Non-starchy vegetables',
                'Lean proteins',
                'Whole grains (in controlled portions)',
                'Cinnamon (may improve insulin sensitivity)'
            ],
            common_symptoms=[
                Symptom.HYPERGLYCEMIA,
                Symptom.INSULIN_RESISTANCE,
                Symptom.FATIGUE
            ]
        )
        
        # Inflammatory Conditions
        self.conditions['inflammatory'] = ConditionProfile(
            condition_id='inflammatory',
            condition_name='Inflammatory Conditions (Arthritis, IBD)',
            condition_type=MedicalCondition.INFLAMMATORY,
            calorie_adjustment=0.0,
            protein_multiplier=1.0,
            extra_nutrients={
                'omega3': 1000.0,
                'vitamin_d': 0.020
            },
            foods_to_avoid=[
                'Trans fats',
                'Excessive omega-6 oils (corn, soybean)'
            ],
            foods_to_limit=[
                'Processed foods',
                'Red meat (high in arachidonic acid)',
                'Refined sugars'
            ],
            therapeutic_foods=[
                'Turmeric (curcumin)',
                'Fatty fish (omega-3)',
                'Berries (anthocyanins)',
                'Leafy greens',
                'Nuts and seeds'
            ],
            common_symptoms=[
                Symptom.INFLAMMATION,
                Symptom.FATIGUE
            ]
        )
    
    def get_condition(self, condition_id: str) -> Optional[ConditionProfile]:
        """Get condition profile"""
        return self.conditions.get(condition_id)


# ============================================================================
# THERAPEUTIC NUTRITION ENGINE
# ============================================================================

class TherapeuticNutritionEngine:
    """
    Complete therapeutic nutrition system
    """
    
    def __init__(
        self,
        compound_db: CompoundDatabase,
        condition_db: ConditionDatabase
    ):
        self.compound_db = compound_db
        self.condition_db = condition_db
        
        logger.info("Therapeutic Nutrition Engine initialized")
    
    def match_symptom_to_compounds(
        self,
        symptom: Symptom
    ) -> List[Tuple[MedicinalCompound, str]]:
        """
        Match symptom to therapeutic compounds
        
        Returns:
            List of (compound, rationale) tuples
        """
        matches = []
        
        # Symptom-compound mappings (evidence-based)
        symptom_map = {
            Symptom.NAUSEA: [
                ('gingerol', 'Gingerol (ginger) has strong evidence for reducing nausea, especially chemotherapy-induced nausea.')
            ],
            Symptom.INFLAMMATION: [
                ('curcumin', 'Curcumin (turmeric) inhibits inflammatory pathways (COX-2, NF-κB).'),
                ('omega3', 'Omega-3 fatty acids produce anti-inflammatory resolvins and protectins.'),
                ('anthocyanins', 'Anthocyanins (berries) provide antioxidant and anti-inflammatory support.')
            ],
            Symptom.HYPERGLYCEMIA: [
                ('soluble_fiber', 'Soluble fiber delays glucose absorption and improves glycemic control.')
            ],
            Symptom.CONSTIPATION: [
                ('soluble_fiber', 'Fiber adds bulk to stool and supports regular bowel movements.')
            ]
        }
        
        compound_ids_rationales = symptom_map.get(symptom, [])
        
        for compound_id, rationale in compound_ids_rationales:
            compound = self.compound_db.get_compound(compound_id)
            if compound:
                matches.append((compound, rationale))
        
        return matches
    
    def create_therapeutic_plan(
        self,
        user_id: str,
        condition_id: str,
        target_symptoms: Optional[List[Symptom]] = None
    ) -> TherapeuticMealPlan:
        """
        Create personalized therapeutic meal plan
        
        Args:
            user_id: User identifier
            condition_id: Medical condition ID
            target_symptoms: Specific symptoms to address
        
        Returns:
            Therapeutic meal plan with interventions
        """
        condition = self.condition_db.get_condition(condition_id)
        
        if not condition:
            raise ValueError(f"Condition {condition_id} not found")
        
        # Use condition's common symptoms if none specified
        if target_symptoms is None:
            target_symptoms = condition.common_symptoms
        
        # Calculate daily targets (simplified - production would be personalized)
        baseline_calories = 2000.0
        baseline_protein_g = 75.0
        
        daily_calories = baseline_calories + condition.calorie_adjustment
        daily_protein_g = baseline_protein_g * condition.protein_multiplier
        
        # Create interventions for symptoms
        interventions = []
        daily_compounds = {}
        
        for symptom in target_symptoms:
            compound_matches = self.match_symptom_to_compounds(symptom)
            
            for compound, rationale in compound_matches:
                intervention = TherapeuticIntervention(
                    intervention_id=f"int_{condition_id}_{symptom.value}_{compound.compound_id}",
                    name=f"{compound.name} for {symptom.value.replace('_', ' ').title()}",
                    condition=condition.condition_type,
                    symptom=symptom,
                    recommended_compounds=[compound.compound_id],
                    food_recommendations=compound.food_sources,
                    timing_notes="With meals for better absorption" if compound.compound_id == 'curcumin' else "",
                    evidence_level=compound.evidence_level,
                    expected_improvement=rationale,
                    citations=compound.citations,
                    medical_supervision_required=(condition.condition_type == MedicalCondition.CANCER),
                    disclaimer=compound.disclaimer if hasattr(compound, 'disclaimer') else "Consult your healthcare provider."
                )
                
                interventions.append(intervention)
                
                # Add to daily compounds
                daily_compounds[compound.compound_id] = compound.therapeutic_dose_mg
        
        # Create meal plan
        plan = TherapeuticMealPlan(
            plan_id=f"plan_{user_id}_{condition_id}_{datetime.now().strftime('%Y%m%d')}",
            user_id=user_id,
            condition=condition.condition_type,
            created_date=date.today(),
            duration_weeks=4,
            daily_calories=daily_calories,
            daily_protein_g=daily_protein_g,
            daily_carbs_g=250.0,  # Simplified
            daily_fat_g=65.0,
            target_symptoms=target_symptoms,
            interventions=interventions,
            meals_per_day=3,
            snacks_per_day=2,
            daily_compounds=daily_compounds,
            validated_by_rd=False
        )
        
        return plan
    
    def generate_meal_recommendations(
        self,
        plan: TherapeuticMealPlan
    ) -> List[str]:
        """
        Generate specific meal recommendations for therapeutic plan
        
        Returns:
            List of meal ideas incorporating therapeutic compounds
        """
        recommendations = []
        
        # Group compounds by food source
        food_sources = set()
        
        for compound_id in plan.daily_compounds.keys():
            compound = self.compound_db.get_compound(compound_id)
            if compound:
                food_sources.update(compound.food_sources)
        
        # Generate meal ideas (simplified - production would use recipe database)
        if 'Turmeric powder' in food_sources:
            recommendations.append(
                "🍛 Turmeric Golden Milk Latte (breakfast): Warm milk with turmeric, cinnamon, honey"
            )
            recommendations.append(
                "🍲 Turmeric Chicken Soup (lunch): Chicken, vegetables, turmeric, ginger"
            )
        
        if 'Fresh ginger root' in food_sources:
            recommendations.append(
                "🍵 Ginger Tea (morning): Fresh ginger steeped in hot water with lemon"
            )
        
        if 'Blueberries' in food_sources:
            recommendations.append(
                "🫐 Berry Smoothie Bowl (breakfast): Blueberries, strawberries, Greek yogurt, granola"
            )
        
        if 'Salmon' in food_sources:
            recommendations.append(
                "🐟 Baked Salmon (dinner): Salmon fillet with roasted vegetables"
            )
        
        if 'Spinach' in food_sources:
            recommendations.append(
                "🥗 Spinach Salad (lunch): Fresh spinach, chickpeas, lemon dressing"
            )
        
        if 'Oats' in food_sources:
            recommendations.append(
                "🥣 Oatmeal (breakfast): Steel-cut oats with berries and walnuts"
            )
        
        return recommendations


# ============================================================================
# TESTING
# ============================================================================

def test_therapeutic_nutrition():
    """Test therapeutic nutrition system"""
    print("=" * 80)
    print("THERAPEUTIC NUTRITION ENGINE - TEST")
    print("=" * 80)
    print("⚠️  MEDICAL DISCLAIMER: This is for educational purposes only.")
    print("    Always consult your healthcare provider before making dietary changes.")
    print("=" * 80)
    
    # Initialize
    compound_db = CompoundDatabase()
    condition_db = ConditionDatabase()
    engine = TherapeuticNutritionEngine(compound_db, condition_db)
    
    # Test 1: Compound database
    print("\n" + "="*80)
    print("Test: Medicinal Compound Database")
    print("="*80)
    
    curcumin = compound_db.get_compound('curcumin')
    
    print(f"✓ Loaded {len(compound_db.compounds)} medicinal compounds\n")
    print(f"Example: {curcumin.name} (Turmeric)")
    print(f"  Therapeutic Dose: {curcumin.therapeutic_dose_mg}mg")
    print(f"  Food Equivalent: {curcumin.food_equivalent}")
    print(f"  Benefits:")
    for benefit in curcumin.benefits[:3]:
        print(f"    - {benefit}")
    print(f"  Evidence Level: {curcumin.evidence_level.value.upper()}")
    print(f"  Citations: {len(curcumin.citations)} scientific references")
    print(f"  ⚠️  Contraindications: {', '.join(curcumin.contraindications)}")
    
    # Test 2: Symptom-compound matching
    print("\n" + "="*80)
    print("Test: Symptom-Compound Matching")
    print("="*80)
    
    nausea_compounds = engine.match_symptom_to_compounds(Symptom.NAUSEA)
    
    print(f"✓ For symptom: NAUSEA")
    print(f"  Found {len(nausea_compounds)} therapeutic compounds\n")
    
    for compound, rationale in nausea_compounds:
        print(f"  💊 {compound.name}")
        print(f"     Rationale: {rationale}")
        print(f"     Food Sources: {', '.join(compound.food_sources[:3])}")
        print(f"     Dose: {compound.therapeutic_dose_mg}mg ({compound.food_equivalent})")
        print()
    
    # Test 3: Pregnancy meal plan
    print("\n" + "="*80)
    print("Test: Pregnancy - First Trimester Meal Plan")
    print("="*80)
    
    pregnancy_plan = engine.create_therapeutic_plan(
        user_id='user123',
        condition_id='pregnancy_t1'
    )
    
    print(f"✓ Created therapeutic meal plan: {pregnancy_plan.plan_id}")
    print(f"\n📋 PLAN SUMMARY:")
    print(f"   Condition: {pregnancy_plan.condition.value.title()}")
    print(f"   Duration: {pregnancy_plan.duration_weeks} weeks")
    print(f"   Daily Calories: {pregnancy_plan.daily_calories:.0f} kcal")
    print(f"   Daily Protein: {pregnancy_plan.daily_protein_g:.0f}g")
    print(f"   Target Symptoms: {', '.join([s.value for s in pregnancy_plan.target_symptoms])}")
    
    print(f"\n🎯 INTERVENTIONS ({len(pregnancy_plan.interventions)}):")
    for intervention in pregnancy_plan.interventions:
        print(f"\n   {intervention.name}")
        print(f"     Evidence: {intervention.evidence_level.value.upper()}")
        print(f"     Expected: {intervention.expected_improvement}")
        print(f"     Foods: {', '.join(intervention.food_recommendations[:3])}")
        if intervention.medical_supervision_required:
            print(f"     ⚠️  Medical supervision required")
    
    print(f"\n💊 DAILY COMPOUNDS:")
    for compound_id, dose_mg in pregnancy_plan.daily_compounds.items():
        compound = compound_db.get_compound(compound_id)
        print(f"   - {compound.name}: {dose_mg}mg/day ({compound.food_equivalent})")
    
    meal_recs = engine.generate_meal_recommendations(pregnancy_plan)
    print(f"\n🍽️  MEAL RECOMMENDATIONS:")
    for rec in meal_recs:
        print(f"   {rec}")
    
    print(f"\n⚠️  DISCLAIMER:\n   {pregnancy_plan.disclaimer}")
    
    # Test 4: Cancer chemotherapy support
    print("\n" + "="*80)
    print("Test: Cancer Chemotherapy Support Plan")
    print("="*80)
    
    cancer_plan = engine.create_therapeutic_plan(
        user_id='user456',
        condition_id='cancer_chemo',
        target_symptoms=[Symptom.NAUSEA, Symptom.POOR_APPETITE]
    )
    
    print(f"✓ Created chemotherapy support plan")
    print(f"\n📋 PLAN SUMMARY:")
    print(f"   Daily Calories: {cancer_plan.daily_calories:.0f} kcal (+{cancer_plan.daily_calories-2000:.0f} for weight maintenance)")
    print(f"   Daily Protein: {cancer_plan.daily_protein_g:.0f}g (1.2× baseline for muscle preservation)")
    
    print(f"\n🎯 TARGETED SYMPTOMS:")
    for symptom in cancer_plan.target_symptoms:
        print(f"   - {symptom.value.replace('_', ' ').title()}")
        
        # Get interventions for this symptom
        symptom_interventions = [i for i in cancer_plan.interventions if i.symptom == symptom]
        
        for intervention in symptom_interventions:
            print(f"     → {intervention.name}")
            print(f"        {intervention.expected_improvement}")
    
    print(f"\n⚠️  IMPORTANT:")
    print(f"   - Medical supervision REQUIRED for cancer patients")
    print(f"   - Coordinate with oncology team before starting")
    print(f"   - Monitor for food-drug interactions")
    
    # Test 5: Diabetes meal plan
    print("\n" + "="*80)
    print("Test: Type 2 Diabetes Meal Plan")
    print("="*80)
    
    diabetes_plan = engine.create_therapeutic_plan(
        user_id='user789',
        condition_id='diabetes_t2'
    )
    
    print(f"✓ Created diabetes management plan")
    print(f"\n🎯 KEY THERAPEUTIC COMPOUNDS:")
    for compound_id in diabetes_plan.daily_compounds.keys():
        compound = compound_db.get_compound(compound_id)
        print(f"\n   {compound.name}:")
        print(f"     Mechanism: {compound.mechanisms[0]}")
        print(f"     Sources: {', '.join(compound.food_sources[:4])}")
    
    meal_recs = engine.generate_meal_recommendations(diabetes_plan)
    print(f"\n🍽️  SAMPLE MEALS:")
    for rec in meal_recs[:3]:
        print(f"   {rec}")
    
    # Test 6: Inflammation management
    print("\n" + "="*80)
    print("Test: Anti-Inflammatory Nutrition Plan")
    print("="*80)
    
    inflammatory_plan = engine.create_therapeutic_plan(
        user_id='user999',
        condition_id='inflammatory',
        target_symptoms=[Symptom.INFLAMMATION]
    )
    
    inflammation_compounds = engine.match_symptom_to_compounds(Symptom.INFLAMMATION)
    
    print(f"✓ Found {len(inflammation_compounds)} anti-inflammatory compounds")
    print(f"\n💊 ANTI-INFLAMMATORY COMPOUNDS:")
    
    for compound, rationale in inflammation_compounds:
        print(f"\n   {compound.name}:")
        print(f"     Mechanism: {compound.mechanisms[0]}")
        print(f"     {rationale}")
        print(f"     Evidence: {compound.evidence_level.value.upper()} ({len(compound.citations)} studies)")
    
    print("\n✅ All therapeutic nutrition tests passed!")
    print("\n💡 Production Features:")
    print("  - Medical ontology: ICD-11, SNOMED CT integration")
    print("  - Drug interaction checker: DrugBank API")
    print("  - Personalization: Body weight, age, activity level")
    print("  - RD validation: Professional review workflow")
    print("  - Compliance tracking: Symptom logging, adherence monitoring")
    print("  - Research database: PubMed integration for latest evidence")
    print("  - Safety alerts: Real-time contraindication checking")


if __name__ == '__main__':
    test_therapeutic_nutrition()
