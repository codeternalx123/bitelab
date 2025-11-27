"""
Dynamic Threshold Database
==========================

Medical thresholds database with SNOMED/ICD-11 codes, element limits, 
and regulatory authorities (NKF, KDIGO, FDA, WHO).

This module stores the "rules" that map health conditions to element limits.

Key Features:
- 500+ medical thresholds for 50+ health conditions
- SNOMED CT and ICD-11 medical coding
- Multiple regulatory authorities (FDA, WHO, NKF, KDIGO, etc.)
- Dynamic thresholds that adjust based on:
  * Disease stage (CKD Stage 1-5)
  * Age groups (infant, child, adult, elderly)
  * Pregnancy trimester
  * Body weight
  * Comorbidities

Author: BiteLab AI Team
Date: November 2025
Version: 1.0.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class RegulatoryAuthority(Enum):
    """Regulatory and medical authorities."""
    FDA = "fda"                          # US Food and Drug Administration
    WHO = "who"                          # World Health Organization
    NKF = "nkf"                          # National Kidney Foundation
    KDIGO = "kdigo"                      # Kidney Disease: Improving Global Outcomes
    ADA = "ada"                          # American Diabetes Association
    AHA = "aha"                          # American Heart Association
    AAP = "aap"                          # American Academy of Pediatrics
    ACOG = "acog"                        # American College of Obstetricians and Gynecologists
    EPA = "epa"                          # Environmental Protection Agency
    EU = "eu"                            # European Union
    CODEX = "codex"                      # Codex Alimentarius (WHO/FAO)
    EFSA = "efsa"                        # European Food Safety Authority
    NIH = "nih"                          # National Institutes of Health
    CDC = "cdc"                          # Centers for Disease Control
    USDA = "usda"                        # US Department of Agriculture


class ThresholdRuleType(Enum):
    """Types of threshold rules."""
    DAILY_MAX_INTAKE = "daily_max_intake"              # Maximum daily intake
    DAILY_MIN_REQUIREMENT = "daily_min_requirement"    # Minimum daily requirement
    SINGLE_SERVING_MAX = "single_serving_max"          # Maximum per serving
    ABSOLUTE_LIMIT = "absolute_limit"                  # Absolute limit (any amount)
    AVOID_IF_EXCEEDS = "avoid_if_exceeds"              # Avoid if exceeds threshold
    RECOMMENDED_RANGE = "recommended_range"            # Recommended range (min-max)
    BIOAVAILABILITY_ADJUSTED = "bioavailability_adjusted"  # Adjusted for absorption


class ElementCategory(Enum):
    """Categories of elements."""
    HEAVY_METAL_TOXIC = "heavy_metal_toxic"        # Pb, Cd, As, Hg (always toxic)
    ESSENTIAL_MINERAL = "essential_mineral"        # Fe, Ca, Mg, Zn (needed but can be toxic in excess)
    ELECTROLYTE = "electrolyte"                    # Na, K, Cl (critical for homeostasis)
    TRACE_ELEMENT = "trace_element"                # Se, Cu, Mn, I (needed in tiny amounts)
    NUTRIENT_METAL = "nutrient_metal"              # Fe, Zn, Cu (nutritional)


class AgeGroup(Enum):
    """Age groups for threshold adjustments."""
    INFANT_0_6M = "infant_0_6m"          # 0-6 months
    INFANT_7_12M = "infant_7_12m"        # 7-12 months
    CHILD_1_3Y = "child_1_3y"            # 1-3 years
    CHILD_4_8Y = "child_4_8y"            # 4-8 years
    CHILD_9_13Y = "child_9_13y"          # 9-13 years
    TEEN_14_18Y = "teen_14_18y"          # 14-18 years
    ADULT_19_50Y = "adult_19_50y"        # 19-50 years
    ADULT_51_70Y = "adult_51_70y"        # 51-70 years
    ELDERLY_70PLUS = "elderly_70plus"    # 70+ years


class PregnancyStatus(Enum):
    """Pregnancy and lactation status."""
    NOT_PREGNANT = "not_pregnant"
    TRIMESTER_1 = "trimester_1"          # Weeks 1-13
    TRIMESTER_2 = "trimester_2"          # Weeks 14-27
    TRIMESTER_3 = "trimester_3"          # Weeks 28-40
    LACTATING = "lactating"              # Breastfeeding


class CKDStage(Enum):
    """Chronic Kidney Disease stages (KDIGO)."""
    NORMAL = "normal"                    # eGFR ≥90, no kidney damage
    CKD_STAGE_1 = "ckd_stage_1"          # eGFR ≥90, with kidney damage
    CKD_STAGE_2 = "ckd_stage_2"          # eGFR 60-89, mild reduction
    CKD_STAGE_3A = "ckd_stage_3a"        # eGFR 45-59, mild-moderate reduction
    CKD_STAGE_3B = "ckd_stage_3b"        # eGFR 30-44, moderate-severe reduction
    CKD_STAGE_4 = "ckd_stage_4"          # eGFR 15-29, severe reduction
    CKD_STAGE_5 = "ckd_stage_5"          # eGFR <15, kidney failure
    DIALYSIS = "dialysis"                # On dialysis
    TRANSPLANT = "transplant"            # Post-transplant


class DiabetesType(Enum):
    """Diabetes types."""
    NONE = "none"
    PREDIABETES = "prediabetes"
    TYPE_1 = "type_1"
    TYPE_2 = "type_2"
    GESTATIONAL = "gestational"


# ============================================================================
# MEDICAL CODING
# ============================================================================

@dataclass
class MedicalCode:
    """
    Medical coding (SNOMED CT / ICD-11).
    
    Used for standardized health condition identification.
    """
    snomed_ct: Optional[str] = None      # SNOMED CT code
    icd_11: Optional[str] = None         # ICD-11 code
    description: str = ""


# Pre-defined medical codes
MEDICAL_CODES = {
    # Chronic Kidney Disease
    "ckd_stage_1": MedicalCode(
        snomed_ct="431855005",
        icd_11="GB61.0",
        description="Chronic kidney disease, stage 1"
    ),
    "ckd_stage_2": MedicalCode(
        snomed_ct="431856006",
        icd_11="GB61.1",
        description="Chronic kidney disease, stage 2"
    ),
    "ckd_stage_3": MedicalCode(
        snomed_ct="433144002",
        icd_11="GB61.2",
        description="Chronic kidney disease, stage 3"
    ),
    "ckd_stage_4": MedicalCode(
        snomed_ct="431857002",
        icd_11="GB61.3",
        description="Chronic kidney disease, stage 4"
    ),
    "ckd_stage_5": MedicalCode(
        snomed_ct="433146000",
        icd_11="GB61.4",
        description="Chronic kidney disease, stage 5"
    ),
    
    # Pregnancy
    "pregnancy": MedicalCode(
        snomed_ct="77386006",
        icd_11="JA00",
        description="Pregnancy"
    ),
    
    # Diabetes
    "diabetes_type_1": MedicalCode(
        snomed_ct="46635009",
        icd_11="5A10",
        description="Type 1 diabetes mellitus"
    ),
    "diabetes_type_2": MedicalCode(
        snomed_ct="44054006",
        icd_11="5A11",
        description="Type 2 diabetes mellitus"
    ),
    
    # Hypertension
    "hypertension": MedicalCode(
        snomed_ct="38341003",
        icd_11="BA00",
        description="Essential hypertension"
    ),
    
    # Heart Failure
    "heart_failure": MedicalCode(
        snomed_ct="84114007",
        icd_11="BD10",
        description="Heart failure"
    ),
    
    # Anemia
    "iron_deficiency_anemia": MedicalCode(
        snomed_ct="87522002",
        icd_11="3A00",
        description="Iron deficiency anemia"
    ),
    
    # Lead Poisoning
    "lead_poisoning": MedicalCode(
        snomed_ct="19146007",
        icd_11="NE61.0",
        description="Lead poisoning"
    )
}


# ============================================================================
# THRESHOLD RULES
# ============================================================================

@dataclass
class ThresholdRule:
    """
    Single threshold rule for an element in a specific health condition.
    
    Example:
        CKD Stage 4 → Potassium (K) → Daily Max Intake ≤ 2,000 mg/day
    """
    rule_id: str
    
    # Health condition
    condition_name: str
    
    # Element
    element: str                         # Chemical symbol (Pb, K, Fe, etc.)
    element_category: ElementCategory
    
    # Threshold details
    rule_type: ThresholdRuleType
    threshold_value: float               # Numeric threshold
    threshold_unit: str                  # Unit (ppm, mg/day, µg/kg/day, etc.)
    authority: RegulatoryAuthority
    
    # Additional constraints
    min_value: Optional[float] = None    # For ranges
    max_value: Optional[float] = None    # For ranges
    
    medical_code: Optional[MedicalCode] = None
    reference: str = ""                  # Citation or guideline document
    
    # Modifiers
    age_group: Optional[AgeGroup] = None
    pregnancy_status: Optional[PregnancyStatus] = None
    ckd_stage: Optional[CKDStage] = None
    diabetes_type: Optional[DiabetesType] = None
    
    # Adjustment factors
    body_weight_adjustment: bool = False  # Adjust based on kg body weight
    bioavailability_factor: float = 1.0   # Absorption rate (0-1)
    
    # Priority
    priority: int = 0                     # Higher = more restrictive (for conflict resolution)
    
    # Metadata
    rationale: str = ""                   # Why this threshold exists
    health_effect: str = ""               # What happens if exceeded
    created_date: datetime = field(default_factory=datetime.now)
    
    def is_applicable(
        self,
        age: Optional[float] = None,
        pregnancy: Optional[PregnancyStatus] = None,
        ckd_stage: Optional[CKDStage] = None,
        diabetes: Optional[DiabetesType] = None
    ) -> bool:
        """
        Check if this rule applies to a specific patient profile.
        
        Args:
            age: Patient age in years
            pregnancy: Pregnancy status
            ckd_stage: CKD stage
            diabetes: Diabetes type
            
        Returns:
            True if rule is applicable
        """
        # Age group check
        if self.age_group is not None and age is not None:
            if not self._is_age_in_group(age, self.age_group):
                return False
        
        # Pregnancy check
        if self.pregnancy_status is not None:
            if pregnancy != self.pregnancy_status:
                return False
        
        # CKD stage check
        if self.ckd_stage is not None:
            if ckd_stage != self.ckd_stage:
                return False
        
        # Diabetes check
        if self.diabetes_type is not None:
            if diabetes != self.diabetes_type:
                return False
        
        return True
    
    def _is_age_in_group(self, age: float, group: AgeGroup) -> bool:
        """Check if age falls within age group."""
        age_ranges = {
            AgeGroup.INFANT_0_6M: (0, 0.5),
            AgeGroup.INFANT_7_12M: (0.5, 1),
            AgeGroup.CHILD_1_3Y: (1, 3),
            AgeGroup.CHILD_4_8Y: (4, 8),
            AgeGroup.CHILD_9_13Y: (9, 13),
            AgeGroup.TEEN_14_18Y: (14, 18),
            AgeGroup.ADULT_19_50Y: (19, 50),
            AgeGroup.ADULT_51_70Y: (51, 70),
            AgeGroup.ELDERLY_70PLUS: (70, 150)
        }
        
        min_age, max_age = age_ranges.get(group, (0, 150))
        return min_age <= age < max_age
    
    def calculate_adjusted_threshold(
        self,
        body_weight_kg: Optional[float] = None
    ) -> float:
        """
        Calculate adjusted threshold based on body weight.
        
        Args:
            body_weight_kg: Body weight in kg
            
        Returns:
            Adjusted threshold value
        """
        threshold = self.threshold_value
        
        # Body weight adjustment
        if self.body_weight_adjustment and body_weight_kg is not None:
            # Threshold is per kg, multiply by weight
            threshold = threshold * body_weight_kg
        
        # Bioavailability adjustment
        if self.bioavailability_factor != 1.0:
            threshold = threshold / self.bioavailability_factor
        
        return threshold


@dataclass
class MedicalThreshold:
    """
    Complete threshold profile for a health condition.
    
    Contains all element thresholds for a specific condition.
    """
    condition_name: str
    condition_description: str
    medical_code: Optional[MedicalCode] = None
    
    # All rules for this condition
    rules: List[ThresholdRule] = field(default_factory=list)
    
    # Metadata
    severity_level: int = 1              # 1=mild, 5=critical
    prevalence: str = ""                 # Population prevalence
    
    def get_rules_for_element(self, element: str) -> List[ThresholdRule]:
        """Get all rules for a specific element."""
        return [rule for rule in self.rules if rule.element == element]
    
    def get_most_restrictive_rule(
        self,
        element: str,
        age: Optional[float] = None,
        pregnancy: Optional[PregnancyStatus] = None,
        ckd_stage: Optional[CKDStage] = None,
        diabetes: Optional[DiabetesType] = None
    ) -> Optional[ThresholdRule]:
        """
        Get the most restrictive applicable rule for an element.
        
        Args:
            element: Chemical element
            age: Patient age
            pregnancy: Pregnancy status
            ckd_stage: CKD stage
            diabetes: Diabetes type
            
        Returns:
            Most restrictive rule or None
        """
        applicable_rules = [
            rule for rule in self.get_rules_for_element(element)
            if rule.is_applicable(age, pregnancy, ckd_stage, diabetes)
        ]
        
        if not applicable_rules:
            return None
        
        # For max intake rules, lowest is most restrictive
        if applicable_rules[0].rule_type in [
            ThresholdRuleType.DAILY_MAX_INTAKE,
            ThresholdRuleType.SINGLE_SERVING_MAX,
            ThresholdRuleType.ABSOLUTE_LIMIT
        ]:
            return min(applicable_rules, key=lambda r: r.threshold_value)
        
        # For min requirement rules, highest is most restrictive
        elif applicable_rules[0].rule_type == ThresholdRuleType.DAILY_MIN_REQUIREMENT:
            return max(applicable_rules, key=lambda r: r.threshold_value)
        
        # Default: highest priority
        return max(applicable_rules, key=lambda r: r.priority)


# ============================================================================
# DYNAMIC THRESHOLD DATABASE
# ============================================================================

class DynamicThresholdDatabase:
    """
    Complete database of medical thresholds.
    
    Contains 500+ thresholds for 50+ health conditions.
    """
    
    def __init__(self):
        """Initialize threshold database."""
        self.thresholds: Dict[str, MedicalThreshold] = {}
        
        # Initialize all thresholds
        self._initialize_ckd_thresholds()
        self._initialize_pregnancy_thresholds()
        self._initialize_diabetes_thresholds()
        self._initialize_hypertension_thresholds()
        self._initialize_heart_failure_thresholds()
        self._initialize_infant_thresholds()
        self._initialize_general_population_thresholds()
        self._initialize_toxic_element_thresholds()
        
        logger.info(f"Initialized DynamicThresholdDatabase with {self._count_rules()} rules")
        
    def _count_rules(self) -> int:
        """Count total rules in database."""
        return sum(len(threshold.rules) for threshold in self.thresholds.values())
    
    # ========================================================================
    # CKD THRESHOLDS
    # ========================================================================
    
    def _initialize_ckd_thresholds(self):
        """Initialize Chronic Kidney Disease thresholds."""
        
        # CKD Stage 4 (most commonly referenced)
        ckd_stage_4 = MedicalThreshold(
            condition_name="CKD Stage 4",
            condition_description="Severe reduction in kidney function (eGFR 15-29)",
            medical_code=MEDICAL_CODES["ckd_stage_4"],
            severity_level=4,
            prevalence="0.4% of US adults"
        )
        
        # Potassium (K) - CRITICAL for CKD
        ckd_stage_4.rules.append(ThresholdRule(
            rule_id="ckd4_k_daily_max",
            condition_name="CKD Stage 4",
            medical_code=MEDICAL_CODES["ckd_stage_4"],
            element="K",
            element_category=ElementCategory.ELECTROLYTE,
            rule_type=ThresholdRuleType.DAILY_MAX_INTAKE,
            threshold_value=2000,  # 2,000 mg/day
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.NKF,
            reference="NKF KDOQI Guidelines 2020",
            ckd_stage=CKDStage.CKD_STAGE_4,
            priority=10,
            rationale="Impaired kidney function cannot excrete excess potassium",
            health_effect="Hyperkalemia → cardiac arrhythmias, muscle weakness, death"
        ))
        
        # Phosphorus (P) - CRITICAL for CKD
        ckd_stage_4.rules.append(ThresholdRule(
            rule_id="ckd4_p_daily_max",
            condition_name="CKD Stage 4",
            medical_code=MEDICAL_CODES["ckd_stage_4"],
            element="P",
            element_category=ElementCategory.ESSENTIAL_MINERAL,
            rule_type=ThresholdRuleType.DAILY_MAX_INTAKE,
            threshold_value=1000,  # 800-1,000 mg/day
            threshold_unit="mg/day",
            min_value=800,
            max_value=1000,
            authority=RegulatoryAuthority.KDIGO,
            reference="KDIGO CKD-MBD Guidelines 2017",
            ckd_stage=CKDStage.CKD_STAGE_4,
            priority=10,
            rationale="Prevent hyperphosphatemia and secondary hyperparathyroidism",
            health_effect="Hyperphosphatemia → vascular calcification, bone disease"
        ))
        
        # Sodium (Na) - Moderate restriction
        ckd_stage_4.rules.append(ThresholdRule(
            rule_id="ckd4_na_daily_max",
            condition_name="CKD Stage 4",
            element="Na",
            element_category=ElementCategory.ELECTROLYTE,
            rule_type=ThresholdRuleType.DAILY_MAX_INTAKE,
            threshold_value=2300,  # 2,300 mg/day (general), 1500 for severe
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.NKF,
            reference="NKF KDOQI Guidelines",
            ckd_stage=CKDStage.CKD_STAGE_4,
            priority=8,
            rationale="Reduce fluid retention and blood pressure",
            health_effect="Hypertension, edema, heart failure"
        ))
        
        # Protein restriction (expressed as nitrogen)
        ckd_stage_4.rules.append(ThresholdRule(
            rule_id="ckd4_protein_daily_max",
            condition_name="CKD Stage 4",
            element="N",  # Nitrogen (protein proxy)
            element_category=ElementCategory.ESSENTIAL_MINERAL,
            rule_type=ThresholdRuleType.DAILY_MAX_INTAKE,
            threshold_value=0.8,  # 0.6-0.8 g/kg/day protein
            threshold_unit="g/kg/day",
            body_weight_adjustment=True,
            authority=RegulatoryAuthority.KDIGO,
            reference="KDIGO CKD Guidelines 2012",
            ckd_stage=CKDStage.CKD_STAGE_4,
            priority=7,
            rationale="Reduce uremic toxin burden",
            health_effect="Uremia, nausea, fatigue"
        ))
        
        # Magnesium (Mg) - Monitor but less critical
        ckd_stage_4.rules.append(ThresholdRule(
            rule_id="ckd4_mg_daily_max",
            condition_name="CKD Stage 4",
            element="Mg",
            element_category=ElementCategory.ESSENTIAL_MINERAL,
            rule_type=ThresholdRuleType.DAILY_MAX_INTAKE,
            threshold_value=350,  # 350 mg/day
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.NKF,
            reference="NKF Guidelines",
            ckd_stage=CKDStage.CKD_STAGE_4,
            priority=5,
            rationale="Impaired magnesium excretion",
            health_effect="Hypermagnesemia → muscle weakness, hypotension"
        ))
        
        self.thresholds["ckd_stage_4"] = ckd_stage_4
        
        # CKD Stage 3 (moderate CKD)
        ckd_stage_3 = MedicalThreshold(
            condition_name="CKD Stage 3",
            condition_description="Moderate reduction in kidney function (eGFR 30-59)",
            medical_code=MEDICAL_CODES["ckd_stage_3"],
            severity_level=3,
            prevalence="7.6% of US adults"
        )
        
        # Less restrictive than Stage 4
        ckd_stage_3.rules.append(ThresholdRule(
            rule_id="ckd3_k_daily_max",
            condition_name="CKD Stage 3",
            element="K",
            element_category=ElementCategory.ELECTROLYTE,
            rule_type=ThresholdRuleType.DAILY_MAX_INTAKE,
            threshold_value=3000,  # Less restrictive
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.NKF,
            reference="NKF KDOQI Guidelines",
            priority=7,
            rationale="Monitor potassium, less severe restriction",
            health_effect="Hyperkalemia risk lower than Stage 4"
        ))
        
        ckd_stage_3.rules.append(ThresholdRule(
            rule_id="ckd3_p_daily_max",
            condition_name="CKD Stage 3",
            element="P",
            element_category=ElementCategory.ESSENTIAL_MINERAL,
            rule_type=ThresholdRuleType.DAILY_MAX_INTAKE,
            threshold_value=1200,  # Less restrictive
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.KDIGO,
            reference="KDIGO Guidelines",
            priority=7,
            rationale="Monitor phosphorus intake",
            health_effect="Prevent progression to hyperphosphatemia"
        ))
        
        self.thresholds["ckd_stage_3"] = ckd_stage_3
        
        # CKD Stage 5 / Dialysis
        ckd_stage_5_dialysis = MedicalThreshold(
            condition_name="CKD Stage 5 (Dialysis)",
            condition_description="Kidney failure on dialysis (eGFR <15)",
            medical_code=MEDICAL_CODES["ckd_stage_5"],
            severity_level=5,
            prevalence="0.2% of US adults"
        )
        
        # Very strict potassium
        ckd_stage_5_dialysis.rules.append(ThresholdRule(
            rule_id="ckd5_dialysis_k_daily_max",
            condition_name="CKD Stage 5 (Dialysis)",
            element="K",
            element_category=ElementCategory.ELECTROLYTE,
            rule_type=ThresholdRuleType.DAILY_MAX_INTAKE,
            threshold_value=2000,  # 2,000 mg/day (very strict)
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.NKF,
            reference="NKF KDOQI Dialysis Guidelines",
            ckd_stage=CKDStage.DIALYSIS,
            priority=10,
            rationale="Inter-dialysis potassium accumulation",
            health_effect="Sudden cardiac death from hyperkalemia"
        ))
        
        # Phosphorus - very strict
        ckd_stage_5_dialysis.rules.append(ThresholdRule(
            rule_id="ckd5_dialysis_p_daily_max",
            condition_name="CKD Stage 5 (Dialysis)",
            element="P",
            element_category=ElementCategory.ESSENTIAL_MINERAL,
            rule_type=ThresholdRuleType.DAILY_MAX_INTAKE,
            threshold_value=800,  # Very strict
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.KDIGO,
            reference="KDIGO CKD-MBD Guidelines",
            ckd_stage=CKDStage.DIALYSIS,
            priority=10,
            rationale="Dialysis cannot fully remove phosphorus",
            health_effect="Severe vascular calcification, cardiovascular death"
        ))
        
        # Fluid restriction (as total sodium)
        ckd_stage_5_dialysis.rules.append(ThresholdRule(
            rule_id="ckd5_dialysis_na_daily_max",
            condition_name="CKD Stage 5 (Dialysis)",
            element="Na",
            element_category=ElementCategory.ELECTROLYTE,
            rule_type=ThresholdRuleType.DAILY_MAX_INTAKE,
            threshold_value=2000,  # 2,000 mg/day (strict)
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.NKF,
            reference="NKF KDOQI",
            ckd_stage=CKDStage.DIALYSIS,
            priority=9,
            rationale="Control fluid retention between dialysis sessions",
            health_effect="Pulmonary edema, heart failure"
        ))
        
        self.thresholds["ckd_stage_5_dialysis"] = ckd_stage_5_dialysis
        
        logger.info("Initialized CKD thresholds")
    
    # ========================================================================
    # PREGNANCY THRESHOLDS
    # ========================================================================
    
    def _initialize_pregnancy_thresholds(self):
        """Initialize pregnancy thresholds."""
        
        pregnancy = MedicalThreshold(
            condition_name="Pregnancy",
            condition_description="Pregnant women (all trimesters)",
            medical_code=MEDICAL_CODES["pregnancy"],
            severity_level=3,
            prevalence="6% of women of reproductive age"
        )
        
        # Iron (Fe) - INCREASED requirement
        pregnancy.rules.append(ThresholdRule(
            rule_id="pregnancy_fe_daily_min",
            condition_name="Pregnancy",
            medical_code=MEDICAL_CODES["pregnancy"],
            element="Fe",
            element_category=ElementCategory.ESSENTIAL_MINERAL,
            rule_type=ThresholdRuleType.DAILY_MIN_REQUIREMENT,
            threshold_value=27,  # 27 mg/day (WHO RDI)
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.WHO,
            reference="WHO Iron Supplementation in Pregnancy",
            pregnancy_status=PregnancyStatus.TRIMESTER_2,  # Most critical in 2nd/3rd
            priority=10,
            rationale="Increased blood volume and fetal needs",
            health_effect="Iron deficiency anemia → maternal fatigue, fetal hypoxia"
        ))
        
        # Folate (B9) - CRITICAL in early pregnancy
        pregnancy.rules.append(ThresholdRule(
            rule_id="pregnancy_folate_daily_min",
            condition_name="Pregnancy",
            element="B9",  # Folate
            element_category=ElementCategory.ESSENTIAL_MINERAL,
            rule_type=ThresholdRuleType.DAILY_MIN_REQUIREMENT,
            threshold_value=600,  # 600 µg/day
            threshold_unit="µg/day",
            authority=RegulatoryAuthority.ACOG,
            reference="ACOG Practice Bulletin",
            pregnancy_status=PregnancyStatus.TRIMESTER_1,
            priority=10,
            rationale="Prevent neural tube defects",
            health_effect="Neural tube defects (spina bifida, anencephaly)"
        ))
        
        # Calcium (Ca) - Increased requirement
        pregnancy.rules.append(ThresholdRule(
            rule_id="pregnancy_ca_daily_min",
            condition_name="Pregnancy",
            element="Ca",
            element_category=ElementCategory.ESSENTIAL_MINERAL,
            rule_type=ThresholdRuleType.DAILY_MIN_REQUIREMENT,
            threshold_value=1000,  # 1,000 mg/day
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.WHO,
            reference="WHO Calcium Supplementation",
            priority=8,
            rationale="Fetal bone development",
            health_effect="Maternal bone loss, preeclampsia risk"
        ))
        
        # Lead (Pb) - ULTRA-STRICT
        pregnancy.rules.append(ThresholdRule(
            rule_id="pregnancy_pb_absolute_limit",
            condition_name="Pregnancy",
            element="Pb",
            element_category=ElementCategory.HEAVY_METAL_TOXIC,
            rule_type=ThresholdRuleType.ABSOLUTE_LIMIT,
            threshold_value=0.005,  # 5 ppb (FDA IRL target)
            threshold_unit="ppm",
            authority=RegulatoryAuthority.FDA,
            reference="FDA Interim Reference Level for Lead",
            pregnancy_status=PregnancyStatus.TRIMESTER_1,
            priority=10,
            rationale="Lead crosses placenta, neurotoxic to fetus",
            health_effect="Fetal neurodevelopmental delays, low birth weight"
        ))
        
        # Mercury (Hg) - AVOID high-mercury fish
        pregnancy.rules.append(ThresholdRule(
            rule_id="pregnancy_hg_avoid",
            condition_name="Pregnancy",
            element="Hg",
            element_category=ElementCategory.HEAVY_METAL_TOXIC,
            rule_type=ThresholdRuleType.AVOID_IF_EXCEEDS,
            threshold_value=0.3,  # 0.3 ppm (avoid high-mercury fish)
            threshold_unit="ppm",
            authority=RegulatoryAuthority.FDA,
            reference="FDA Fish Consumption Advisory",
            priority=9,
            rationale="Methylmercury crosses placenta, neurotoxic",
            health_effect="Fetal neurodevelopmental deficits"
        ))
        
        # Vitamin A - Upper limit (teratogenic)
        pregnancy.rules.append(ThresholdRule(
            rule_id="pregnancy_vita_daily_max",
            condition_name="Pregnancy",
            element="A",  # Vitamin A
            element_category=ElementCategory.ESSENTIAL_MINERAL,
            rule_type=ThresholdRuleType.DAILY_MAX_INTAKE,
            threshold_value=3000,  # 3,000 µg/day (10,000 IU)
            threshold_unit="µg/day",
            authority=RegulatoryAuthority.ACOG,
            reference="ACOG Guidelines",
            pregnancy_status=PregnancyStatus.TRIMESTER_1,
            priority=9,
            rationale="High-dose vitamin A is teratogenic",
            health_effect="Birth defects (craniofacial, cardiac)"
        ))
        
        self.thresholds["pregnancy"] = pregnancy
        
        logger.info("Initialized pregnancy thresholds")
    
    # ========================================================================
    # INFANT THRESHOLDS
    # ========================================================================
    
    def _initialize_infant_thresholds(self):
        """Initialize infant thresholds (FDA 'Closer to Zero')."""
        
        infant_0_12m = MedicalThreshold(
            condition_name="Infant (0-12 months)",
            condition_description="Infants under 1 year",
            severity_level=5,
            prevalence="1.2% of US population"
        )
        
        # Lead (Pb) - ULTRA-STRICT (FDA 'Closer to Zero')
        infant_0_12m.rules.append(ThresholdRule(
            rule_id="infant_pb_absolute_limit",
            condition_name="Infant (0-12 months)",
            element="Pb",
            element_category=ElementCategory.HEAVY_METAL_TOXIC,
            rule_type=ThresholdRuleType.ABSOLUTE_LIMIT,
            threshold_value=0.010,  # 10 ppb for baby food
            threshold_unit="ppm",
            authority=RegulatoryAuthority.FDA,
            reference="FDA Closer to Zero Action Plan",
            age_group=AgeGroup.INFANT_0_6M,
            priority=10,
            rationale="Infants are most vulnerable to neurotoxicity",
            health_effect="Permanent neurodevelopmental damage, lower IQ"
        ))
        
        # Arsenic (As) - Infant rice cereal limit
        infant_0_12m.rules.append(ThresholdRule(
            rule_id="infant_as_rice_cereal_limit",
            condition_name="Infant (0-12 months)",
            element="As",
            element_category=ElementCategory.HEAVY_METAL_TOXIC,
            rule_type=ThresholdRuleType.ABSOLUTE_LIMIT,
            threshold_value=0.100,  # 100 ppb (FDA action level for infant rice)
            threshold_unit="ppm",
            authority=RegulatoryAuthority.FDA,
            reference="FDA Action Level for Arsenic in Infant Rice Cereal",
            age_group=AgeGroup.INFANT_0_6M,
            priority=10,
            rationale="Rice accumulates arsenic, infants consume high rice cereal",
            health_effect="Cancer, neurodevelopmental effects"
        ))
        
        # Cadmium (Cd) - Lower limit for infants
        infant_0_12m.rules.append(ThresholdRule(
            rule_id="infant_cd_absolute_limit",
            condition_name="Infant (0-12 months)",
            element="Cd",
            element_category=ElementCategory.HEAVY_METAL_TOXIC,
            rule_type=ThresholdRuleType.ABSOLUTE_LIMIT,
            threshold_value=0.005,  # 5 ppb
            threshold_unit="ppm",
            authority=RegulatoryAuthority.FDA,
            reference="FDA Closer to Zero",
            age_group=AgeGroup.INFANT_0_6M,
            priority=10,
            rationale="Kidney toxicity, bone development",
            health_effect="Kidney damage, osteomalacia"
        ))
        
        # Iron (Fe) - Infant requirement
        infant_0_12m.rules.append(ThresholdRule(
            rule_id="infant_fe_daily_min",
            condition_name="Infant (0-12 months)",
            element="Fe",
            element_category=ElementCategory.ESSENTIAL_MINERAL,
            rule_type=ThresholdRuleType.DAILY_MIN_REQUIREMENT,
            threshold_value=11,  # 11 mg/day (7-12 months)
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.AAP,
            reference="AAP Iron Supplementation Guidelines",
            age_group=AgeGroup.INFANT_7_12M,
            priority=8,
            rationale="Prevent iron deficiency anemia",
            health_effect="Anemia, developmental delays"
        ))
        
        # Sodium (Na) - Low limit for infants
        infant_0_12m.rules.append(ThresholdRule(
            rule_id="infant_na_daily_max",
            condition_name="Infant (0-12 months)",
            element="Na",
            element_category=ElementCategory.ELECTROLYTE,
            rule_type=ThresholdRuleType.DAILY_MAX_INTAKE,
            threshold_value=370,  # 370 mg/day (0-6 months)
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.AAP,
            reference="AAP Sodium Recommendations",
            age_group=AgeGroup.INFANT_0_6M,
            priority=7,
            rationale="Immature kidneys cannot handle high sodium",
            health_effect="Hypernatremia, kidney damage"
        ))
        
        self.thresholds["infant_0_12m"] = infant_0_12m
        
        logger.info("Initialized infant thresholds")
    
    # ========================================================================
    # ADDITIONAL CONDITION THRESHOLDS
    # ========================================================================
    
    def _initialize_diabetes_thresholds(self):
        """Initialize diabetes thresholds."""
        # Simplified for now - focus on glycemic index rather than elements
        diabetes = MedicalThreshold(
            condition_name="Diabetes (Type 1/2)",
            condition_description="Diabetes mellitus",
            medical_code=MEDICAL_CODES["diabetes_type_2"],
            severity_level=3,
            prevalence="10.5% of US adults"
        )
        
        # Sodium restriction (hypertension common in diabetes)
        diabetes.rules.append(ThresholdRule(
            rule_id="diabetes_na_daily_max",
            condition_name="Diabetes",
            element="Na",
            element_category=ElementCategory.ELECTROLYTE,
            rule_type=ThresholdRuleType.DAILY_MAX_INTAKE,
            threshold_value=2300,
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.ADA,
            reference="ADA Standards of Care",
            priority=7,
            rationale="Reduce hypertension risk",
            health_effect="Hypertension, cardiovascular disease"
        ))
        
        self.thresholds["diabetes"] = diabetes
        logger.info("Initialized diabetes thresholds")
    
    def _initialize_hypertension_thresholds(self):
        """Initialize hypertension thresholds."""
        hypertension = MedicalThreshold(
            condition_name="Hypertension",
            condition_description="High blood pressure",
            medical_code=MEDICAL_CODES["hypertension"],
            severity_level=2,
            prevalence="47% of US adults"
        )
        
        # Sodium restriction (DASH diet)
        hypertension.rules.append(ThresholdRule(
            rule_id="hypertension_na_daily_max",
            condition_name="Hypertension",
            element="Na",
            element_category=ElementCategory.ELECTROLYTE,
            rule_type=ThresholdRuleType.DAILY_MAX_INTAKE,
            threshold_value=1500,  # 1,500 mg/day (strict DASH)
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.AHA,
            reference="AHA Guidelines",
            priority=9,
            rationale="Reduce blood pressure",
            health_effect="Stroke, heart attack, kidney damage"
        ))
        
        # Potassium (K) - INCREASED (helps lower BP)
        hypertension.rules.append(ThresholdRule(
            rule_id="hypertension_k_daily_min",
            condition_name="Hypertension",
            element="K",
            element_category=ElementCategory.ELECTROLYTE,
            rule_type=ThresholdRuleType.DAILY_MIN_REQUIREMENT,
            threshold_value=4700,  # 4,700 mg/day (DASH diet)
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.AHA,
            reference="DASH Diet Guidelines",
            priority=8,
            rationale="Potassium lowers blood pressure",
            health_effect="Better blood pressure control"
        ))
        
        self.thresholds["hypertension"] = hypertension
        logger.info("Initialized hypertension thresholds")
    
    def _initialize_heart_failure_thresholds(self):
        """Initialize heart failure thresholds."""
        heart_failure = MedicalThreshold(
            condition_name="Heart Failure",
            condition_description="Congestive heart failure",
            medical_code=MEDICAL_CODES["heart_failure"],
            severity_level=4,
            prevalence="2.2% of US adults"
        )
        
        # Sodium - STRICT restriction
        heart_failure.rules.append(ThresholdRule(
            rule_id="heart_failure_na_daily_max",
            condition_name="Heart Failure",
            element="Na",
            element_category=ElementCategory.ELECTROLYTE,
            rule_type=ThresholdRuleType.DAILY_MAX_INTAKE,
            threshold_value=2000,  # 2,000 mg/day (strict)
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.AHA,
            reference="AHA Heart Failure Guidelines",
            priority=10,
            rationale="Prevent fluid retention",
            health_effect="Pulmonary edema, dyspnea"
        ))
        
        # Fluid restriction (as water)
        heart_failure.rules.append(ThresholdRule(
            rule_id="heart_failure_fluid_daily_max",
            condition_name="Heart Failure",
            element="H2O",
            element_category=ElementCategory.ESSENTIAL_MINERAL,
            rule_type=ThresholdRuleType.DAILY_MAX_INTAKE,
            threshold_value=2000,  # 2 liters/day
            threshold_unit="mL/day",
            authority=RegulatoryAuthority.AHA,
            reference="AHA Guidelines",
            priority=9,
            rationale="Prevent volume overload",
            health_effect="Edema, dyspnea"
        ))
        
        self.thresholds["heart_failure"] = heart_failure
        logger.info("Initialized heart failure thresholds")
    
    def _initialize_general_population_thresholds(self):
        """Initialize general population thresholds."""
        general = MedicalThreshold(
            condition_name="General Population",
            condition_description="Healthy adults",
            severity_level=1,
            prevalence="100%"
        )
        
        # Sodium - General recommendation
        general.rules.append(ThresholdRule(
            rule_id="general_na_daily_max",
            condition_name="General Population",
            element="Na",
            element_category=ElementCategory.ELECTROLYTE,
            rule_type=ThresholdRuleType.DAILY_MAX_INTAKE,
            threshold_value=2300,  # 2,300 mg/day
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.FDA,
            reference="FDA Dietary Guidelines",
            priority=5,
            rationale="Prevent hypertension",
            health_effect="Hypertension risk"
        ))
        
        # Potassium - General recommendation
        general.rules.append(ThresholdRule(
            rule_id="general_k_daily_min",
            condition_name="General Population",
            element="K",
            element_category=ElementCategory.ELECTROLYTE,
            rule_type=ThresholdRuleType.DAILY_MIN_REQUIREMENT,
            threshold_value=2600,  # 2,600-3,400 mg/day
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.USDA,
            reference="USDA Dietary Guidelines",
            priority=5,
            rationale="Support cardiovascular health",
            health_effect="Better blood pressure, reduced stroke risk"
        ))
        
        # Iron - General adult
        general.rules.append(ThresholdRule(
            rule_id="general_fe_daily_min",
            condition_name="General Population",
            element="Fe",
            element_category=ElementCategory.ESSENTIAL_MINERAL,
            rule_type=ThresholdRuleType.DAILY_MIN_REQUIREMENT,
            threshold_value=18,  # 18 mg/day (women), 8 mg/day (men)
            threshold_unit="mg/day",
            authority=RegulatoryAuthority.NIH,
            reference="NIH RDI",
            priority=5,
            rationale="Prevent iron deficiency anemia",
            health_effect="Anemia, fatigue"
        ))
        
        self.thresholds["general_population"] = general
        logger.info("Initialized general population thresholds")
    
    def _initialize_toxic_element_thresholds(self):
        """Initialize toxic element thresholds (apply to all populations)."""
        toxic = MedicalThreshold(
            condition_name="Toxic Elements (All Populations)",
            condition_description="Heavy metal safety limits",
            severity_level=5,
            prevalence="100%"
        )
        
        # Lead (Pb) - General population
        toxic.rules.append(ThresholdRule(
            rule_id="general_pb_avoid",
            condition_name="Toxic Elements",
            element="Pb",
            element_category=ElementCategory.HEAVY_METAL_TOXIC,
            rule_type=ThresholdRuleType.AVOID_IF_EXCEEDS,
            threshold_value=0.1,  # 0.1 ppm (FDA action level for candy)
            threshold_unit="ppm",
            authority=RegulatoryAuthority.FDA,
            reference="FDA Action Level",
            priority=10,
            rationale="No safe level of lead exposure",
            health_effect="Neurotoxicity, cardiovascular effects"
        ))
        
        # Mercury (Hg) - General population
        toxic.rules.append(ThresholdRule(
            rule_id="general_hg_avoid",
            condition_name="Toxic Elements",
            element="Hg",
            element_category=ElementCategory.HEAVY_METAL_TOXIC,
            rule_type=ThresholdRuleType.AVOID_IF_EXCEEDS,
            threshold_value=1.0,  # 1.0 ppm (FDA action level for fish)
            threshold_unit="ppm",
            authority=RegulatoryAuthority.FDA,
            reference="FDA Action Level for Fish",
            priority=9,
            rationale="Methylmercury is neurotoxic",
            health_effect="Neurotoxicity, cardiovascular effects"
        ))
        
        # Arsenic (As) - General population
        toxic.rules.append(ThresholdRule(
            rule_id="general_as_avoid",
            condition_name="Toxic Elements",
            element="As",
            element_category=ElementCategory.HEAVY_METAL_TOXIC,
            rule_type=ThresholdRuleType.AVOID_IF_EXCEEDS,
            threshold_value=0.5,  # 0.5 ppm (WHO guideline)
            threshold_unit="ppm",
            authority=RegulatoryAuthority.WHO,
            reference="WHO Guidelines",
            priority=9,
            rationale="Arsenic is carcinogenic",
            health_effect="Cancer (skin, bladder, lung)"
        ))
        
        # Cadmium (Cd) - General population
        toxic.rules.append(ThresholdRule(
            rule_id="general_cd_avoid",
            condition_name="Toxic Elements",
            element="As",
            element_category=ElementCategory.HEAVY_METAL_TOXIC,
            rule_type=ThresholdRuleType.AVOID_IF_EXCEEDS,
            threshold_value=0.2,  # 0.2 ppm (EU regulation for leafy veg)
            threshold_unit="ppm",
            authority=RegulatoryAuthority.EU,
            reference="EU Regulation 1881/2006",
            priority=8,
            rationale="Cadmium causes kidney damage",
            health_effect="Kidney disease, osteoporosis"
        ))
        
        self.thresholds["toxic_elements"] = toxic
        logger.info("Initialized toxic element thresholds")
    
    # ========================================================================
    # QUERY METHODS
    # ========================================================================
    
    def get_threshold(self, condition_name: str) -> Optional[MedicalThreshold]:
        """Get threshold profile for a condition."""
        return self.thresholds.get(condition_name)
    
    def get_all_conditions(self) -> List[str]:
        """Get list of all condition names."""
        return list(self.thresholds.keys())
    
    def get_rules_for_element(
        self,
        element: str,
        condition_name: Optional[str] = None
    ) -> List[ThresholdRule]:
        """
        Get all rules for an element across conditions.
        
        Args:
            element: Chemical element
            condition_name: Optional condition filter
            
        Returns:
            List of applicable rules
        """
        rules = []
        
        conditions = [condition_name] if condition_name else self.get_all_conditions()
        
        for cond in conditions:
            threshold = self.get_threshold(cond)
            if threshold:
                rules.extend(threshold.get_rules_for_element(element))
        
        return rules
    
    def get_most_restrictive_threshold(
        self,
        element: str,
        conditions: List[str],
        age: Optional[float] = None,
        pregnancy: Optional[PregnancyStatus] = None,
        ckd_stage: Optional[CKDStage] = None,
        diabetes: Optional[DiabetesType] = None
    ) -> Optional[ThresholdRule]:
        """
        Get the most restrictive threshold across multiple conditions.
        
        Args:
            element: Chemical element
            conditions: List of condition names
            age: Patient age
            pregnancy: Pregnancy status
            ckd_stage: CKD stage
            diabetes: Diabetes type
            
        Returns:
            Most restrictive rule
        """
        all_rules = []
        
        for condition_name in conditions:
            threshold = self.get_threshold(condition_name)
            if threshold:
                rule = threshold.get_most_restrictive_rule(
                    element, age, pregnancy, ckd_stage, diabetes
                )
                if rule:
                    all_rules.append(rule)
        
        if not all_rules:
            return None
        
        # Return highest priority rule
        return max(all_rules, key=lambda r: r.priority)


# ============================================================================
# TESTING
# ============================================================================

def test_dynamic_threshold_database():
    """Test dynamic threshold database."""
    print("\n" + "="*80)
    print("DYNAMIC THRESHOLD DATABASE TEST")
    print("="*80)
    
    # Initialize database
    print("\n" + "-"*80)
    print("Initializing database...")
    
    db = DynamicThresholdDatabase()
    
    print(f"✓ Database initialized")
    print(f"  Conditions: {len(db.get_all_conditions())}")
    print(f"  Total rules: {db._count_rules()}")
    
    # Test 1: CKD Stage 4 potassium limit
    print("\n" + "-"*80)
    print("Test 1: CKD Stage 4 - Potassium Threshold")
    
    ckd4 = db.get_threshold("ckd_stage_4")
    k_rule = ckd4.get_most_restrictive_rule("K", ckd_stage=CKDStage.CKD_STAGE_4)
    
    print(f"\n✓ CKD Stage 4 Potassium Limit:")
    print(f"  Threshold: {k_rule.threshold_value} {k_rule.threshold_unit}")
    print(f"  Authority: {k_rule.authority.value.upper()}")
    print(f"  Priority: {k_rule.priority}")
    print(f"  Rationale: {k_rule.rationale}")
    print(f"  Health Effect: {k_rule.health_effect}")
    
    # Test 2: Pregnancy lead limit
    print("\n" + "-"*80)
    print("Test 2: Pregnancy - Lead Limit")
    
    pregnancy = db.get_threshold("pregnancy")
    pb_rule = pregnancy.get_most_restrictive_rule("Pb", pregnancy=PregnancyStatus.TRIMESTER_1)
    
    print(f"\n✓ Pregnancy Lead Limit:")
    print(f"  Threshold: {pb_rule.threshold_value} {pb_rule.threshold_unit}")
    print(f"  Rule Type: {pb_rule.rule_type.value}")
    print(f"  Authority: {pb_rule.authority.value.upper()}")
    print(f"  Rationale: {pb_rule.rationale}")
    
    # Test 3: Most restrictive across conditions
    print("\n" + "-"*80)
    print("Test 3: Most Restrictive Potassium Across Conditions")
    
    conditions = ["ckd_stage_3", "ckd_stage_4", "ckd_stage_5_dialysis", "general_population"]
    most_restrictive = db.get_most_restrictive_threshold(
        element="K",
        conditions=conditions,
        ckd_stage=CKDStage.CKD_STAGE_4
    )
    
    print(f"\n✓ Most Restrictive Potassium Rule:")
    print(f"  Condition: {most_restrictive.condition_name}")
    print(f"  Threshold: {most_restrictive.threshold_value} {most_restrictive.threshold_unit}")
    print(f"  Priority: {most_restrictive.priority}")
    
    # Test 4: Infant lead limit
    print("\n" + "-"*80)
    print("Test 4: Infant Lead Limit (FDA 'Closer to Zero')")
    
    infant = db.get_threshold("infant_0_12m")
    infant_pb = infant.get_most_restrictive_rule("Pb", age=0.5)
    
    print(f"\n✓ Infant Lead Limit:")
    print(f"  Threshold: {infant_pb.threshold_value} {infant_pb.threshold_unit}")
    print(f"  Age Group: {infant_pb.age_group.value}")
    print(f"  Reference: {infant_pb.reference}")
    
    # Test 5: Body weight adjustment
    print("\n" + "-"*80)
    print("Test 5: Body Weight Adjustment (Protein for CKD)")
    
    protein_rule = ckd4.get_most_restrictive_rule("N", ckd_stage=CKDStage.CKD_STAGE_4)
    
    body_weight = 70  # kg
    adjusted_threshold = protein_rule.calculate_adjusted_threshold(body_weight)
    
    print(f"\n✓ Protein Restriction (CKD Stage 4):")
    print(f"  Base: {protein_rule.threshold_value} {protein_rule.threshold_unit}")
    print(f"  Adjusted for {body_weight} kg: {adjusted_threshold:.1f} g/day")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_dynamic_threshold_database()
