"""
Risk Integration Engine
======================

5-step decision process connecting atomic composition predictions
to personalized health risk assessment.

This is the CORE ENGINE that implements the critical decision logic:
    Step 1: Atomic Input Reception
    Step 2: Risk Profile Lookup
    Step 3: Hard Safety Check (Toxic Elements)
    Step 4: Nutrient Goal Check (Essential Elements)
    Step 5: Uncertainty Buffer Application

Example Flow:
------------
User scans 100g spinach → Visual Chemometrics predicts:
    Pb: 0.45 ± 0.10 ppm
    K: 450 ± 20 mg/100g
    Fe: 3.5 ± 0.3 mg/100g

User profile: Pregnant + CKD Stage 4

Step 1: Parse predictions
Step 2: Load strictest thresholds (CKD Stage 4)
Step 3: Pb = 0.45 ppm > 0.005 ppm (pregnancy limit) → CRITICAL FAIL
Step 4: K = 450 mg → 22.5% of 2,000 mg/day limit → WARNING
Step 5: Low confidence Pb → Downgrade from CRITICAL to HIGH CAUTION

Result: "DO NOT CONSUME - Lead exceeds safe limits"

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

from .dynamic_thresholds import (
    DynamicThresholdDatabase,
    ThresholdRule,
    ThresholdRuleType,
    ElementCategory,
    RegulatoryAuthority
)

from .health_profile_engine import (
    HealthProfileEngine,
    UserHealthProfile,
    RiskLevel
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class ElementRiskStatus(Enum):
    """Risk status for a single element."""
    CRITICAL_UNSAFE = "critical_unsafe"      # Exceeds limit, high confidence → DO NOT CONSUME
    HIGH_RISK = "high_risk"                  # Exceeds limit, medium confidence → AVOID
    MODERATE_RISK = "moderate_risk"          # Near limit or low confidence → CAUTION
    LOW_RISK = "low_risk"                    # Below limit, good confidence → SAFE WITH LIMITS
    SAFE = "safe"                            # Well below limit → SAFE
    BENEFICIAL = "beneficial"                # Meets nutritional need → BENEFICIAL
    UNKNOWN = "unknown"                      # Insufficient data → USE USDA
    NOT_APPLICABLE = "not_applicable"        # Element not relevant for user


class ComparisonResult(Enum):
    """Result of threshold comparison."""
    EXCEEDS_CRITICAL = "exceeds_critical"    # >150% of limit
    EXCEEDS_MODERATE = "exceeds_moderate"    # 100-150% of limit
    NEAR_LIMIT = "near_limit"                # 75-100% of limit
    ACCEPTABLE = "acceptable"                # 50-75% of limit
    OPTIMAL = "optimal"                      # 25-50% of limit
    MINIMAL = "minimal"                      # <25% of limit
    BELOW_REQUIREMENT = "below_requirement"  # Below minimum (for nutrients)


class ConfidenceLevel(Enum):
    """Prediction confidence levels."""
    VERY_HIGH = "very_high"      # >90% confidence
    HIGH = "high"                # 70-90%
    MEDIUM = "medium"            # 50-70%
    LOW = "low"                  # 30-50%
    VERY_LOW = "very_low"        # <30%


# ============================================================================
# ATOMIC INPUT STRUCTURES
# ============================================================================

@dataclass
class ElementPrediction:
    """
    Single element prediction from Visual Chemometrics.
    
    Contains:
    - Point estimate (best guess)
    - Uncertainty (standard deviation)
    - Confidence interval (95% CI)
    - Confidence level
    - Prediction method
    """
    element: str                         # Chemical symbol (Pb, K, Fe, etc.)
    
    # Prediction values
    concentration: float                 # Point estimate (ppm, mg/100g, etc.)
    unit: str                            # Unit of measurement
    
    # Uncertainty
    standard_deviation: float            # Standard deviation
    confidence_interval_95: Tuple[float, float]  # (lower, upper) 95% CI
    
    # Confidence
    confidence_level: ConfidenceLevel
    confidence_score: float              # 0-1
    
    # Metadata
    prediction_method: str = "visual_chemometrics"  # visual_chemometrics, ensemble, etc.
    model_version: str = "1.0.0"
    prediction_date: datetime = field(default_factory=datetime.now)
    
    # Quality flags
    image_quality: float = 1.0           # 0-1 (image quality score)
    food_taxonomy_confidence: float = 1.0  # 0-1 (food identification confidence)
    
    def get_upper_ci_bound(self) -> float:
        """Get upper 95% confidence interval bound (conservative for safety)."""
        return self.confidence_interval_95[1]
    
    def get_lower_ci_bound(self) -> float:
        """Get lower 95% confidence interval bound."""
        return self.confidence_interval_95[0]
    
    def is_high_confidence(self) -> bool:
        """Check if prediction is high confidence (>70%)."""
        return self.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]
    
    def calculate_coefficient_of_variation(self) -> float:
        """Calculate coefficient of variation (CV = SD / mean)."""
        if self.concentration == 0:
            return float('inf')
        return self.standard_deviation / abs(self.concentration)


@dataclass
class AtomicInput:
    """
    Complete atomic input from Visual Chemometrics.
    
    Contains predictions for all detected elements.
    """
    food_name: str
    food_category: str
    serving_size_g: float                # Serving size in grams
    
    # Element predictions
    element_predictions: Dict[str, ElementPrediction]
    
    # Overall quality
    overall_confidence: ConfidenceLevel
    overall_confidence_score: float      # 0-1
    
    # Image metadata
    image_id: str
    image_quality_score: float           # 0-1
    
    # Timestamp
    scan_date: datetime = field(default_factory=datetime.now)
    
    def get_prediction(self, element: str) -> Optional[ElementPrediction]:
        """Get prediction for specific element."""
        return self.element_predictions.get(element)
    
    def has_element(self, element: str) -> bool:
        """Check if element was predicted."""
        return element in self.element_predictions
    
    def get_all_elements(self) -> List[str]:
        """Get list of all predicted elements."""
        return list(self.element_predictions.keys())
    
    def get_toxic_elements(self) -> List[str]:
        """Get list of toxic elements (Pb, Cd, As, Hg)."""
        toxic = ['Pb', 'Cd', 'As', 'Hg']
        return [e for e in self.get_all_elements() if e in toxic]
    
    def get_essential_elements(self) -> List[str]:
        """Get list of essential elements (Fe, Ca, Mg, K, Na, P, Zn, etc.)."""
        essential = ['Fe', 'Ca', 'Mg', 'K', 'Na', 'P', 'Zn', 'Cu', 'Se', 'I']
        return [e for e in self.get_all_elements() if e in essential]


# ============================================================================
# RISK ASSESSMENT STRUCTURES
# ============================================================================

@dataclass
class ThresholdComparison:
    """
    Comparison of predicted value against threshold.
    
    Example:
        Predicted K: 450 mg/100g
        Threshold: 2,000 mg/day
        Serving: 100g
        → 450 mg in serving → 22.5% of daily limit
    """
    element: str
    
    # Predicted values
    predicted_concentration: float
    predicted_unit: str
    lower_ci: float
    upper_ci: float
    confidence_level: ConfidenceLevel
    
    # Threshold
    threshold_rule: ThresholdRule
    threshold_value: float
    threshold_unit: str
    
    # Serving size
    serving_size_g: float
    
    # Calculated comparison
    amount_in_serving: float             # Actual amount in serving
    percent_of_limit: float              # % of daily limit (for max rules)
    percent_of_requirement: float        # % of requirement (for min rules)
    
    # Comparison result
    comparison_result: ComparisonResult
    exceeds_threshold: bool
    
    # Conservative assessment (uses upper CI for toxic, lower CI for nutrients)
    conservative_exceeds: bool
    
    # Margin of safety/excess
    margin: float                        # How much over/under limit (absolute)
    margin_percent: float                # How much over/under limit (%)
    
    def __post_init__(self):
        """Calculate comparison metrics."""
        self._calculate_comparison()
    
    def _calculate_comparison(self):
        """Calculate all comparison metrics."""
        # Convert concentration to amount in serving
        # Assume concentration is per 100g
        self.amount_in_serving = (self.predicted_concentration * self.serving_size_g) / 100
        
        # For daily limits (max intake rules)
        if self.threshold_rule.rule_type in [
            ThresholdRuleType.DAILY_MAX_INTAKE,
            ThresholdRuleType.SINGLE_SERVING_MAX,
            ThresholdRuleType.ABSOLUTE_LIMIT,
            ThresholdRuleType.AVOID_IF_EXCEEDS
        ]:
            # Calculate % of limit
            self.percent_of_limit = (self.amount_in_serving / self.threshold_value) * 100
            self.percent_of_requirement = 0
            
            # Check if exceeds
            self.exceeds_threshold = self.predicted_concentration > self.threshold_value
            
            # Conservative: use upper CI for safety
            self.conservative_exceeds = self.upper_ci > self.threshold_value
            
            # Margin
            self.margin = self.amount_in_serving - self.threshold_value
            self.margin_percent = ((self.amount_in_serving - self.threshold_value) / self.threshold_value) * 100
            
            # Comparison result
            if self.percent_of_limit > 150:
                self.comparison_result = ComparisonResult.EXCEEDS_CRITICAL
            elif self.percent_of_limit > 100:
                self.comparison_result = ComparisonResult.EXCEEDS_MODERATE
            elif self.percent_of_limit > 75:
                self.comparison_result = ComparisonResult.NEAR_LIMIT
            elif self.percent_of_limit > 50:
                self.comparison_result = ComparisonResult.ACCEPTABLE
            elif self.percent_of_limit > 25:
                self.comparison_result = ComparisonResult.OPTIMAL
            else:
                self.comparison_result = ComparisonResult.MINIMAL
        
        # For minimum requirements (min intake rules)
        elif self.threshold_rule.rule_type == ThresholdRuleType.DAILY_MIN_REQUIREMENT:
            # Calculate % of requirement
            self.percent_of_requirement = (self.amount_in_serving / self.threshold_value) * 100
            self.percent_of_limit = 0
            
            # Check if meets requirement
            self.exceeds_threshold = self.predicted_concentration >= self.threshold_value
            
            # Conservative: use lower CI (less optimistic)
            self.conservative_exceeds = self.lower_ci >= self.threshold_value
            
            # Margin
            self.margin = self.amount_in_serving - self.threshold_value
            self.margin_percent = ((self.amount_in_serving - self.threshold_value) / self.threshold_value) * 100
            
            # Comparison result
            if self.percent_of_requirement < 25:
                self.comparison_result = ComparisonResult.BELOW_REQUIREMENT
            elif self.percent_of_requirement < 50:
                self.comparison_result = ComparisonResult.MINIMAL
            elif self.percent_of_requirement < 75:
                self.comparison_result = ComparisonResult.ACCEPTABLE
            else:
                self.comparison_result = ComparisonResult.OPTIMAL


@dataclass
class ElementRiskAssessment:
    """
    Risk assessment for a single element.
    
    Combines:
    - Threshold comparison
    - Confidence assessment
    - Risk status determination
    """
    element: str
    element_category: ElementCategory
    
    # Input prediction
    prediction: ElementPrediction
    
    # Threshold comparison
    threshold_comparison: Optional[ThresholdComparison]
    
    # Risk determination
    risk_status: ElementRiskStatus
    risk_score: float                    # 0-100 (higher = more risky)
    
    # Rationale
    risk_rationale: str
    health_effect: str
    
    # Action
    recommended_action: str              # AVOID, LIMIT, MONITOR, SAFE, etc.
    portion_recommendation: Optional[str] = None  # "Limit to 50g" or None
    
    # Flags
    is_primary_concern: bool = False     # Is this the main risk driver?
    requires_medical_consultation: bool = False
    
    def __post_init__(self):
        """Validate risk assessment."""
        if self.risk_status == ElementRiskStatus.CRITICAL_UNSAFE:
            self.is_primary_concern = True


@dataclass
class AtomicRiskAssessment:
    """
    Complete risk assessment for a food item.
    
    This is the OUTPUT of the Risk Integration Engine.
    """
    # Input
    atomic_input: AtomicInput
    user_profile: UserHealthProfile
    
    # Element assessments
    element_assessments: Dict[str, ElementRiskAssessment]
    
    # Overall decision
    overall_risk_status: ElementRiskStatus
    overall_risk_score: float            # 0-100
    overall_confidence: ConfidenceLevel
    
    # Primary concerns
    primary_concerns: List[str]          # Elements causing main risk
    beneficial_elements: List[str]       # Elements providing benefits
    
    # Decision
    is_safe_to_consume: bool
    consumption_recommendation: str      # "AVOID", "LIMIT TO 50g", "SAFE", etc.
    
    # Warnings and benefits
    critical_warnings: List[str]
    moderate_warnings: List[str]
    nutritional_benefits: List[str]
    
    # Rationale
    decision_rationale: str
    key_factors: List[str]               # Key decision factors
    
    # Metadata
    assessment_date: datetime = field(default_factory=datetime.now)
    engine_version: str = "1.0.0"
    
    def get_critical_elements(self) -> List[str]:
        """Get elements with CRITICAL_UNSAFE or HIGH_RISK status."""
        return [
            elem for elem, assessment in self.element_assessments.items()
            if assessment.risk_status in [ElementRiskStatus.CRITICAL_UNSAFE, ElementRiskStatus.HIGH_RISK]
        ]
    
    def get_beneficial_elements(self) -> List[str]:
        """Get elements with BENEFICIAL status."""
        return [
            elem for elem, assessment in self.element_assessments.items()
            if assessment.risk_status == ElementRiskStatus.BENEFICIAL
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            'food_name': self.atomic_input.food_name,
            'serving_size_g': self.atomic_input.serving_size_g,
            'overall_risk_status': self.overall_risk_status.value,
            'overall_risk_score': self.overall_risk_score,
            'is_safe_to_consume': self.is_safe_to_consume,
            'consumption_recommendation': self.consumption_recommendation,
            'primary_concerns': self.primary_concerns,
            'beneficial_elements': self.beneficial_elements,
            'critical_warnings': self.critical_warnings,
            'moderate_warnings': self.moderate_warnings,
            'nutritional_benefits': self.nutritional_benefits,
            'decision_rationale': self.decision_rationale,
            'overall_confidence': self.overall_confidence.value,
            'element_details': {
                elem: {
                    'concentration': assessment.prediction.concentration,
                    'unit': assessment.prediction.unit,
                    'risk_status': assessment.risk_status.value,
                    'risk_score': assessment.risk_score,
                    'recommended_action': assessment.recommended_action,
                    'risk_rationale': assessment.risk_rationale
                }
                for elem, assessment in self.element_assessments.items()
            }
        }


# ============================================================================
# RISK INTEGRATION ENGINE
# ============================================================================

class RiskIntegrationEngine:
    """
    Main Risk Integration Engine.
    
    Implements the 5-step decision process:
        1. Atomic Input Reception
        2. Risk Profile Lookup
        3. Hard Safety Check (Toxic Elements)
        4. Nutrient Goal Check (Essential Elements)
        5. Uncertainty Buffer Application
    """
    
    def __init__(
        self,
        threshold_db: DynamicThresholdDatabase,
        profile_engine: HealthProfileEngine
    ):
        """
        Initialize risk integration engine.
        
        Args:
            threshold_db: Dynamic threshold database
            profile_engine: Health profile engine
        """
        self.threshold_db = threshold_db
        self.profile_engine = profile_engine
        
        logger.info("Initialized RiskIntegrationEngine")
    
    # ========================================================================
    # STEP 1: ATOMIC INPUT RECEPTION
    # ========================================================================
    
    def receive_atomic_input(
        self,
        food_name: str,
        food_category: str,
        serving_size_g: float,
        element_predictions: Dict[str, Dict[str, Any]],
        image_quality_score: float = 1.0,
        image_id: str = ""
    ) -> AtomicInput:
        """
        Step 1: Receive and parse atomic input from Visual Chemometrics.
        
        Args:
            food_name: Name of food
            food_category: Category (leafy_vegetables, etc.)
            serving_size_g: Serving size in grams
            element_predictions: Dict of {element: {concentration, std, ci, confidence}}
            image_quality_score: Image quality (0-1)
            image_id: Image identifier
            
        Returns:
            Structured AtomicInput
        """
        logger.info(f"Step 1: Receiving atomic input for {food_name}")
        
        # Parse element predictions
        parsed_predictions = {}
        
        for element, pred_data in element_predictions.items():
            # Determine confidence level from confidence score
            conf_score = pred_data.get('confidence_score', 0.8)
            
            if conf_score > 0.9:
                conf_level = ConfidenceLevel.VERY_HIGH
            elif conf_score > 0.7:
                conf_level = ConfidenceLevel.HIGH
            elif conf_score > 0.5:
                conf_level = ConfidenceLevel.MEDIUM
            elif conf_score > 0.3:
                conf_level = ConfidenceLevel.LOW
            else:
                conf_level = ConfidenceLevel.VERY_LOW
            
            prediction = ElementPrediction(
                element=element,
                concentration=pred_data['concentration'],
                unit=pred_data.get('unit', 'ppm'),
                standard_deviation=pred_data.get('std', 0.1 * pred_data['concentration']),
                confidence_interval_95=pred_data.get('ci_95', (
                    pred_data['concentration'] - 1.96 * pred_data.get('std', 0.1 * pred_data['concentration']),
                    pred_data['concentration'] + 1.96 * pred_data.get('std', 0.1 * pred_data['concentration'])
                )),
                confidence_level=conf_level,
                confidence_score=conf_score,
                image_quality=image_quality_score
            )
            
            parsed_predictions[element] = prediction
        
        # Determine overall confidence
        if not parsed_predictions:
            overall_conf_level = ConfidenceLevel.VERY_LOW
            overall_conf_score = 0.0
        else:
            avg_conf = np.mean([p.confidence_score for p in parsed_predictions.values()])
            overall_conf_score = avg_conf
            
            if avg_conf > 0.9:
                overall_conf_level = ConfidenceLevel.VERY_HIGH
            elif avg_conf > 0.7:
                overall_conf_level = ConfidenceLevel.HIGH
            elif avg_conf > 0.5:
                overall_conf_level = ConfidenceLevel.MEDIUM
            elif avg_conf > 0.3:
                overall_conf_level = ConfidenceLevel.LOW
            else:
                overall_conf_level = ConfidenceLevel.VERY_LOW
        
        atomic_input = AtomicInput(
            food_name=food_name,
            food_category=food_category,
            serving_size_g=serving_size_g,
            element_predictions=parsed_predictions,
            overall_confidence=overall_conf_level,
            overall_confidence_score=overall_conf_score,
            image_id=image_id,
            image_quality_score=image_quality_score
        )
        
        logger.info(f"  Parsed {len(parsed_predictions)} element predictions")
        logger.info(f"  Overall confidence: {overall_conf_level.value} ({overall_conf_score:.1%})")
        
        return atomic_input
    
    # ========================================================================
    # STEP 2: RISK PROFILE LOOKUP
    # ========================================================================
    
    def lookup_risk_profile(
        self,
        user_id: str,
        atomic_input: AtomicInput
    ) -> Dict[str, ThresholdRule]:
        """
        Step 2: Lookup user's risk profile and get applicable thresholds.
        
        Args:
            user_id: User ID
            atomic_input: Atomic input data
            
        Returns:
            Dict of {element: most_restrictive_threshold}
        """
        logger.info(f"Step 2: Looking up risk profile for user {user_id}")
        
        # Get user profile
        profile = self.profile_engine.get_profile(user_id)
        if not profile:
            logger.error(f"User profile not found: {user_id}")
            return {}
        
        logger.info(f"  User: {profile.age}y, {profile.gender.value}, Risk: {profile.overall_risk_level.value}")
        
        # Get applicable conditions
        applicable_conditions = self.profile_engine.get_applicable_conditions(user_id)
        logger.info(f"  Applicable conditions: {applicable_conditions}")
        
        # Get thresholds for each element
        element_thresholds = {}
        
        for element in atomic_input.get_all_elements():
            threshold = self.profile_engine.get_most_restrictive_threshold(user_id, element)
            
            if threshold:
                element_thresholds[element] = threshold
                logger.info(f"  {element}: {threshold.threshold_value} {threshold.threshold_unit} ({threshold.condition_name})")
            else:
                logger.warning(f"  {element}: No threshold found")
        
        return element_thresholds
    
    # ========================================================================
    # STEP 3: HARD SAFETY CHECK (TOXIC ELEMENTS)
    # ========================================================================
    
    def perform_safety_check(
        self,
        atomic_input: AtomicInput,
        element_thresholds: Dict[str, ThresholdRule]
    ) -> Dict[str, ElementRiskAssessment]:
        """
        Step 3: Perform hard safety check for toxic elements (Pb, Cd, As, Hg).
        
        This is the MOST CRITICAL step - any failure here = DO NOT CONSUME.
        
        Args:
            atomic_input: Atomic input data
            element_thresholds: Threshold rules for each element
            
        Returns:
            Dict of {element: risk_assessment} for toxic elements
        """
        logger.info("Step 3: Performing hard safety check (toxic elements)")
        
        toxic_assessments = {}
        
        # Check each toxic element
        for element in atomic_input.get_toxic_elements():
            prediction = atomic_input.get_prediction(element)
            threshold = element_thresholds.get(element)
            
            if not prediction:
                continue
            
            if not threshold:
                # No threshold = use general population default
                logger.warning(f"  {element}: No threshold, skipping")
                continue
            
            # Perform threshold comparison
            comparison = ThresholdComparison(
                element=element,
                predicted_concentration=prediction.concentration,
                predicted_unit=prediction.unit,
                lower_ci=prediction.get_lower_ci_bound(),
                upper_ci=prediction.get_upper_ci_bound(),
                confidence_level=prediction.confidence_level,
                threshold_rule=threshold,
                threshold_value=threshold.threshold_value,
                threshold_unit=threshold.threshold_unit,
                serving_size_g=atomic_input.serving_size_g
            )
            
            # Determine risk status
            risk_status, risk_score, rationale, action = self._assess_toxic_element_risk(
                element, comparison, prediction
            )
            
            assessment = ElementRiskAssessment(
                element=element,
                element_category=ElementCategory.HEAVY_METAL_TOXIC,
                prediction=prediction,
                threshold_comparison=comparison,
                risk_status=risk_status,
                risk_score=risk_score,
                risk_rationale=rationale,
                health_effect=threshold.health_effect,
                recommended_action=action,
                is_primary_concern=(risk_status in [ElementRiskStatus.CRITICAL_UNSAFE, ElementRiskStatus.HIGH_RISK])
            )
            
            toxic_assessments[element] = assessment
            
            logger.info(f"  {element}: {risk_status.value} (score: {risk_score:.1f}/100)")
        
        return toxic_assessments
    
    def _assess_toxic_element_risk(
        self,
        element: str,
        comparison: ThresholdComparison,
        prediction: ElementPrediction
    ) -> Tuple[ElementRiskStatus, float, str, str]:
        """
        Assess risk for a toxic element.
        
        Returns:
            (risk_status, risk_score, rationale, action)
        """
        # Use CONSERVATIVE approach: upper CI bound
        conservative_value = comparison.upper_ci
        
        # Calculate how much over limit (conservative)
        percent_over = ((conservative_value - comparison.threshold_value) / comparison.threshold_value) * 100
        
        # High confidence predictions
        if prediction.is_high_confidence():
            if comparison.conservative_exceeds:
                if percent_over > 50:
                    # >150% of limit, high confidence → CRITICAL
                    return (
                        ElementRiskStatus.CRITICAL_UNSAFE,
                        min(100, 80 + percent_over / 10),
                        f"{element} exceeds safe limit by {percent_over:.0f}% (high confidence)",
                        "AVOID - DO NOT CONSUME"
                    )
                else:
                    # 100-150% of limit, high confidence → HIGH RISK
                    return (
                        ElementRiskStatus.HIGH_RISK,
                        min(100, 60 + percent_over / 5),
                        f"{element} exceeds safe limit by {percent_over:.0f}%",
                        "AVOID"
                    )
            elif comparison.comparison_result == ComparisonResult.NEAR_LIMIT:
                # 75-100% of limit → MODERATE RISK
                return (
                    ElementRiskStatus.MODERATE_RISK,
                    50 + comparison.percent_of_limit / 4,
                    f"{element} near safe limit ({comparison.percent_of_limit:.0f}% of limit)",
                    "LIMIT CONSUMPTION"
                )
            else:
                # <75% of limit → SAFE
                return (
                    ElementRiskStatus.SAFE,
                    comparison.percent_of_limit / 2,
                    f"{element} below safe limit ({comparison.percent_of_limit:.0f}% of limit)",
                    "SAFE"
                )
        
        # Medium/Low confidence predictions
        else:
            if comparison.conservative_exceeds:
                # Exceeds but low confidence → Downgrade to HIGH RISK (not CRITICAL)
                return (
                    ElementRiskStatus.HIGH_RISK,
                    60,
                    f"{element} may exceed safe limit (medium confidence)",
                    "CAUTION - Consider lab testing"
                )
            elif comparison.comparison_result == ComparisonResult.NEAR_LIMIT:
                return (
                    ElementRiskStatus.MODERATE_RISK,
                    40,
                    f"{element} may be near safe limit (medium confidence)",
                    "MONITOR INTAKE"
                )
            else:
                return (
                    ElementRiskStatus.LOW_RISK,
                    20,
                    f"{element} likely below limit (medium confidence)",
                    "SAFE WITH MONITORING"
                )
    
    # ========================================================================
    # STEP 4: NUTRIENT GOAL CHECK (ESSENTIAL ELEMENTS)
    # ========================================================================
    
    def perform_nutrient_check(
        self,
        atomic_input: AtomicInput,
        element_thresholds: Dict[str, ThresholdRule],
        user_id: str
    ) -> Dict[str, ElementRiskAssessment]:
        """
        Step 4: Check essential nutrients (Fe, Ca, Mg, K, P, Na, etc.).
        
        For these elements, we check:
        - Do they meet nutritional needs? (min requirements)
        - Do they exceed safety limits? (max intake)
        
        Args:
            atomic_input: Atomic input data
            element_thresholds: Threshold rules
            user_id: User ID
            
        Returns:
            Dict of {element: risk_assessment} for essential elements
        """
        logger.info("Step 4: Performing nutrient goal check (essential elements)")
        
        nutrient_assessments = {}
        
        profile = self.profile_engine.get_profile(user_id)
        
        for element in atomic_input.get_essential_elements():
            prediction = atomic_input.get_prediction(element)
            threshold = element_thresholds.get(element)
            
            if not prediction:
                continue
            
            if not threshold:
                logger.warning(f"  {element}: No threshold")
                continue
            
            # Perform threshold comparison
            comparison = ThresholdComparison(
                element=element,
                predicted_concentration=prediction.concentration,
                predicted_unit=prediction.unit,
                lower_ci=prediction.get_lower_ci_bound(),
                upper_ci=prediction.get_upper_ci_bound(),
                confidence_level=prediction.confidence_level,
                threshold_rule=threshold,
                threshold_value=threshold.threshold_value,
                threshold_unit=threshold.threshold_unit,
                serving_size_g=atomic_input.serving_size_g
            )
            
            # Assess nutrient
            risk_status, risk_score, rationale, action = self._assess_nutrient_risk(
                element, comparison, prediction, threshold, profile
            )
            
            assessment = ElementRiskAssessment(
                element=element,
                element_category=ElementCategory.ESSENTIAL_MINERAL,
                prediction=prediction,
                threshold_comparison=comparison,
                risk_status=risk_status,
                risk_score=risk_score,
                risk_rationale=rationale,
                health_effect=threshold.health_effect,
                recommended_action=action
            )
            
            nutrient_assessments[element] = assessment
            
            logger.info(f"  {element}: {risk_status.value} (score: {risk_score:.1f}/100)")
        
        return nutrient_assessments
    
    def _assess_nutrient_risk(
        self,
        element: str,
        comparison: ThresholdComparison,
        prediction: ElementPrediction,
        threshold: ThresholdRule,
        profile: UserHealthProfile
    ) -> Tuple[ElementRiskStatus, float, str, str]:
        """
        Assess risk for an essential nutrient.
        
        Returns:
            (risk_status, risk_score, rationale, action)
        """
        # Check rule type
        if threshold.rule_type == ThresholdRuleType.DAILY_MAX_INTAKE:
            # Maximum intake rules (K, P, Na for CKD)
            return self._assess_max_intake_nutrient(element, comparison, prediction, profile)
        
        elif threshold.rule_type == ThresholdRuleType.DAILY_MIN_REQUIREMENT:
            # Minimum requirement rules (Fe, Ca for pregnancy)
            return self._assess_min_requirement_nutrient(element, comparison, prediction)
        
        else:
            # Other rule types
            return (
                ElementRiskStatus.UNKNOWN,
                0,
                "Unknown rule type",
                "UNKNOWN"
            )
    
    def _assess_max_intake_nutrient(
        self,
        element: str,
        comparison: ThresholdComparison,
        prediction: ElementPrediction,
        profile: UserHealthProfile
    ) -> Tuple[ElementRiskStatus, float, str, str]:
        """Assess nutrient with maximum intake limit (e.g., K for CKD)."""
        
        # Check medication interactions
        if element == 'K':
            is_affected, med_description = profile.is_on_potassium_altering_medication()
            if is_affected and "increases" in med_description.lower():
                # User on ACE inhibitor/ARB → More strict about potassium
                comparison.percent_of_limit *= 1.2  # Increase perceived risk
        
        if prediction.is_high_confidence():
            if comparison.conservative_exceeds:
                percent_over = comparison.margin_percent
                if percent_over > 50:
                    return (
                        ElementRiskStatus.CRITICAL_UNSAFE,
                        min(100, 70 + percent_over / 5),
                        f"{element} greatly exceeds daily limit ({comparison.percent_of_limit:.0f}% of {comparison.threshold_value} {comparison.threshold_unit})",
                        "AVOID"
                    )
                else:
                    return (
                        ElementRiskStatus.HIGH_RISK,
                        min(100, 50 + percent_over / 3),
                        f"{element} exceeds daily limit ({comparison.percent_of_limit:.0f}% of limit)",
                        "LIMIT CONSUMPTION"
                    )
            elif comparison.percent_of_limit > 75:
                return (
                    ElementRiskStatus.MODERATE_RISK,
                    40 + comparison.percent_of_limit / 5,
                    f"{element} near daily limit ({comparison.percent_of_limit:.0f}% of {comparison.threshold_value} {comparison.threshold_unit})",
                    "MONITOR INTAKE - Limit portion"
                )
            elif comparison.percent_of_limit > 50:
                return (
                    ElementRiskStatus.LOW_RISK,
                    20 + comparison.percent_of_limit / 10,
                    f"{element} acceptable ({comparison.percent_of_limit:.0f}% of daily limit)",
                    "SAFE - Track daily intake"
                )
            else:
                return (
                    ElementRiskStatus.SAFE,
                    comparison.percent_of_limit / 4,
                    f"{element} well below limit ({comparison.percent_of_limit:.0f}% of daily limit)",
                    "SAFE"
                )
        else:
            # Low confidence
            if comparison.percent_of_limit > 100:
                return (
                    ElementRiskStatus.MODERATE_RISK,
                    50,
                    f"{element} may exceed daily limit (low confidence)",
                    "CAUTION"
                )
            else:
                return (
                    ElementRiskStatus.LOW_RISK,
                    30,
                    f"{element} likely acceptable (low confidence)",
                    "MONITOR"
                )
    
    def _assess_min_requirement_nutrient(
        self,
        element: str,
        comparison: ThresholdComparison,
        prediction: ElementPrediction
    ) -> Tuple[ElementRiskStatus, float, str, str]:
        """Assess nutrient with minimum requirement (e.g., Fe for pregnancy)."""
        
        if prediction.is_high_confidence():
            if comparison.percent_of_requirement >= 100:
                return (
                    ElementRiskStatus.BENEFICIAL,
                    0,  # No risk for meeting requirement
                    f"{element} meets daily requirement ({comparison.percent_of_requirement:.0f}% of {comparison.threshold_value} {comparison.threshold_unit})",
                    "BENEFICIAL - Contributes to nutritional needs"
                )
            elif comparison.percent_of_requirement >= 50:
                return (
                    ElementRiskStatus.BENEFICIAL,
                    0,
                    f"{element} provides {comparison.percent_of_requirement:.0f}% of daily requirement",
                    "BENEFICIAL - Good source"
                )
            elif comparison.percent_of_requirement >= 25:
                return (
                    ElementRiskStatus.LOW_RISK,
                    0,
                    f"{element} provides some nutritional value ({comparison.percent_of_requirement:.0f}% of requirement)",
                    "SAFE - Minor source"
                )
            else:
                return (
                    ElementRiskStatus.SAFE,
                    0,
                    f"{element} minimal contribution to requirement",
                    "SAFE"
                )
        else:
            return (
                ElementRiskStatus.UNKNOWN,
                0,
                f"{element} uncertain contribution (low confidence)",
                "UNKNOWN"
            )
    
    # ========================================================================
    # STEP 5: UNCERTAINTY BUFFER APPLICATION
    # ========================================================================
    
    def apply_uncertainty_buffer(
        self,
        all_assessments: Dict[str, ElementRiskAssessment],
        atomic_input: AtomicInput
    ) -> Dict[str, ElementRiskAssessment]:
        """
        Step 5: Apply uncertainty buffer to downgrade warnings if confidence is low.
        
        Philosophy:
        - High confidence + exceeds limit → CRITICAL
        - Low confidence + exceeds limit → HIGH RISK (not CRITICAL)
        - Very low confidence → UNKNOWN (use USDA database)
        
        Args:
            all_assessments: All element assessments
            atomic_input: Atomic input
            
        Returns:
            Adjusted assessments
        """
        logger.info("Step 5: Applying uncertainty buffer")
        
        adjusted = {}
        
        for element, assessment in all_assessments.items():
            confidence = assessment.prediction.confidence_level
            
            # Very low confidence → Downgrade to UNKNOWN
            if confidence == ConfidenceLevel.VERY_LOW:
                logger.info(f"  {element}: Very low confidence → UNKNOWN")
                assessment.risk_status = ElementRiskStatus.UNKNOWN
                assessment.risk_score = 0
                assessment.risk_rationale += " (Very low confidence - use USDA database)"
                assessment.recommended_action = "UNKNOWN - Consult USDA database"
            
            # Low confidence → Downgrade severity
            elif confidence == ConfidenceLevel.LOW:
                if assessment.risk_status == ElementRiskStatus.CRITICAL_UNSAFE:
                    logger.info(f"  {element}: Low confidence → Downgrade CRITICAL to HIGH RISK")
                    assessment.risk_status = ElementRiskStatus.HIGH_RISK
                    assessment.risk_score = max(50, assessment.risk_score - 20)
                    assessment.risk_rationale += " (Low confidence)"
                    assessment.recommended_action = "CAUTION - Consider lab testing"
            
            # Medium confidence → Add caution note
            elif confidence == ConfidenceLevel.MEDIUM:
                if assessment.risk_status in [ElementRiskStatus.CRITICAL_UNSAFE, ElementRiskStatus.HIGH_RISK]:
                    assessment.risk_rationale += " (Medium confidence)"
            
            adjusted[element] = assessment
        
        return adjusted
    
    # ========================================================================
    # MAIN ASSESSMENT METHOD
    # ========================================================================
    
    def assess_risk(
        self,
        user_id: str,
        atomic_input: AtomicInput
    ) -> AtomicRiskAssessment:
        """
        Perform complete 5-step risk assessment.
        
        This is the MAIN METHOD that orchestrates all 5 steps.
        
        Args:
            user_id: User ID
            atomic_input: Atomic input from Visual Chemometrics
            
        Returns:
            Complete AtomicRiskAssessment
        """
        logger.info("="*80)
        logger.info(f"RISK ASSESSMENT: {atomic_input.food_name} ({atomic_input.serving_size_g}g)")
        logger.info("="*80)
        
        # Get user profile
        profile = self.profile_engine.get_profile(user_id)
        if not profile:
            raise ValueError(f"User profile not found: {user_id}")
        
        # Step 2: Risk Profile Lookup
        element_thresholds = self.lookup_risk_profile(user_id, atomic_input)
        
        # Step 3: Hard Safety Check (Toxic Elements)
        toxic_assessments = self.perform_safety_check(atomic_input, element_thresholds)
        
        # Step 4: Nutrient Goal Check (Essential Elements)
        nutrient_assessments = self.perform_nutrient_check(atomic_input, element_thresholds, user_id)
        
        # Combine assessments
        all_assessments = {**toxic_assessments, **nutrient_assessments}
        
        # Step 5: Uncertainty Buffer Application
        all_assessments = self.apply_uncertainty_buffer(all_assessments, atomic_input)
        
        # Make overall decision
        overall_decision = self._make_overall_decision(all_assessments, atomic_input, profile)
        
        logger.info("="*80)
        logger.info(f"DECISION: {overall_decision['consumption_recommendation']}")
        logger.info(f"RISK: {overall_decision['overall_risk_status'].value}")
        logger.info("="*80)
        
        return AtomicRiskAssessment(
            atomic_input=atomic_input,
            user_profile=profile,
            element_assessments=all_assessments,
            **overall_decision
        )
    
    def _make_overall_decision(
        self,
        assessments: Dict[str, ElementRiskAssessment],
        atomic_input: AtomicInput,
        profile: UserHealthProfile
    ) -> Dict[str, Any]:
        """
        Make overall risk decision based on all element assessments.
        
        Returns:
            Dict with overall decision fields
        """
        # Categorize elements
        critical_elements = [
            elem for elem, a in assessments.items()
            if a.risk_status == ElementRiskStatus.CRITICAL_UNSAFE
        ]
        
        high_risk_elements = [
            elem for elem, a in assessments.items()
            if a.risk_status == ElementRiskStatus.HIGH_RISK
        ]
        
        moderate_risk_elements = [
            elem for elem, a in assessments.items()
            if a.risk_status == ElementRiskStatus.MODERATE_RISK
        ]
        
        beneficial_elements = [
            elem for elem, a in assessments.items()
            if a.risk_status == ElementRiskStatus.BENEFICIAL
        ]
        
        # Determine overall risk status
        if critical_elements:
            overall_status = ElementRiskStatus.CRITICAL_UNSAFE
            overall_score = max(a.risk_score for a in assessments.values() if a.risk_status == ElementRiskStatus.CRITICAL_UNSAFE)
            is_safe = False
            recommendation = "DO NOT CONSUME"
            rationale = f"Critical safety concern: {', '.join(critical_elements)} exceed safe limits"
        elif high_risk_elements:
            overall_status = ElementRiskStatus.HIGH_RISK
            overall_score = max(a.risk_score for a in assessments.values() if a.risk_status == ElementRiskStatus.HIGH_RISK)
            is_safe = False
            recommendation = "AVOID"
            rationale = f"High risk: {', '.join(high_risk_elements)} exceed safe limits"
        elif moderate_risk_elements:
            overall_status = ElementRiskStatus.MODERATE_RISK
            overall_score = max((a.risk_score for a in assessments.values() if a.risk_status == ElementRiskStatus.MODERATE_RISK), default=40)
            is_safe = False
            recommendation = f"LIMIT TO {atomic_input.serving_size_g // 2}g"
            rationale = f"Moderate risk: {', '.join(moderate_risk_elements)} near limits"
        else:
            overall_status = ElementRiskStatus.SAFE
            overall_score = max((a.risk_score for a in assessments.values()), default=10)
            is_safe = True
            recommendation = "SAFE TO CONSUME"
            rationale = "All elements within safe limits"
        
        # Generate warnings
        critical_warnings = [
            f"{elem}: {assessments[elem].risk_rationale}"
            for elem in critical_elements + high_risk_elements
        ]
        
        moderate_warnings = [
            f"{elem}: {assessments[elem].risk_rationale}"
            for elem in moderate_risk_elements
        ]
        
        nutritional_benefits = [
            f"{elem}: {assessments[elem].risk_rationale}"
            for elem in beneficial_elements
        ]
        
        # Key factors
        key_factors = []
        if critical_elements:
            key_factors.append(f"CRITICAL: {', '.join(critical_elements)} exceed limits")
        if high_risk_elements:
            key_factors.append(f"HIGH RISK: {', '.join(high_risk_elements)}")
        if profile.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            key_factors.append(f"User high-risk profile: {profile.overall_risk_level.value}")
        
        return {
            'overall_risk_status': overall_status,
            'overall_risk_score': overall_score,
            'overall_confidence': atomic_input.overall_confidence,
            'primary_concerns': critical_elements + high_risk_elements,
            'beneficial_elements': beneficial_elements,
            'is_safe_to_consume': is_safe,
            'consumption_recommendation': recommendation,
            'critical_warnings': critical_warnings,
            'moderate_warnings': moderate_warnings,
            'nutritional_benefits': nutritional_benefits,
            'decision_rationale': rationale,
            'key_factors': key_factors
        }


# ============================================================================
# TESTING
# ============================================================================

def test_risk_integration_engine():
    """Test risk integration engine."""
    print("\n" + "="*80)
    print("RISK INTEGRATION ENGINE TEST")
    print("="*80)
    
    # Initialize
    print("\n" + "-"*80)
    print("Initializing...")
    
    from .dynamic_thresholds import DynamicThresholdDatabase, CKDStage
    from .health_profile_engine import (
        HealthProfileEngine, Gender, HealthCondition,
        PregnancyStatus, MEDICATION_DATABASE
    )
    
    threshold_db = DynamicThresholdDatabase()
    profile_engine = HealthProfileEngine(threshold_db)
    risk_engine = RiskIntegrationEngine(threshold_db, profile_engine)
    
    print("✓ Initialized")
    
    # Create test user: CKD Stage 4 patient
    print("\n" + "-"*80)
    print("Creating test user: CKD Stage 4 patient")
    
    profile = profile_engine.create_profile(
        user_id="test_patient_001",
        age=65,
        gender=Gender.MALE,
        body_weight_kg=80,
        height_cm=175
    )
    
    ckd_condition = HealthCondition(
        condition_id="ckd001",
        condition_name="CKD Stage 4",
        severity="severe",
        ckd_stage=CKDStage.CKD_STAGE_4,
        egfr=22.0
    )
    
    profile_engine.add_condition("test_patient_001", ckd_condition)
    profile_engine.add_medication("test_patient_001", MEDICATION_DATABASE["lisinopril"])
    
    print(f"✓ Created user: {profile.age}y, {profile.gender.value}, CKD Stage 4")
    
    # Test Case 1: Safe spinach
    print("\n" + "-"*80)
    print("Test Case 1: Safe spinach (low lead, moderate potassium)")
    
    atomic_input_safe = risk_engine.receive_atomic_input(
        food_name="Organic Spinach",
        food_category="leafy_vegetables",
        serving_size_g=100,
        element_predictions={
            'Pb': {'concentration': 0.020, 'std': 0.003, 'confidence_score': 0.85, 'unit': 'ppm'},
            'K': {'concentration': 450, 'std': 20, 'confidence_score': 0.90, 'unit': 'mg/100g'},
            'Fe': {'concentration': 3.5, 'std': 0.3, 'confidence_score': 0.88, 'unit': 'mg/100g'},
            'Mg': {'concentration': 89, 'std': 8, 'confidence_score': 0.85, 'unit': 'mg/100g'}
        },
        image_quality_score=0.95
    )
    
    assessment_safe = risk_engine.assess_risk("test_patient_001", atomic_input_safe)
    
    print(f"\n✓ Assessment Complete:")
    print(f"  Overall Status: {assessment_safe.overall_risk_status.value}")
    print(f"  Risk Score: {assessment_safe.overall_risk_score:.1f}/100")
    print(f"  Recommendation: {assessment_safe.consumption_recommendation}")
    print(f"  Rationale: {assessment_safe.decision_rationale}")
    
    # Test Case 2: Contaminated spinach (high lead)
    print("\n" + "-"*80)
    print("Test Case 2: Contaminated spinach (high lead)")
    
    atomic_input_contaminated = risk_engine.receive_atomic_input(
        food_name="Contaminated Spinach",
        food_category="leafy_vegetables",
        serving_size_g=100,
        element_predictions={
            'Pb': {'concentration': 0.45, 'std': 0.10, 'confidence_score': 0.82, 'unit': 'ppm'},
            'K': {'concentration': 450, 'std': 20, 'confidence_score': 0.90, 'unit': 'mg/100g'},
            'Fe': {'concentration': 3.2, 'std': 0.4, 'confidence_score': 0.85, 'unit': 'mg/100g'}
        },
        image_quality_score=0.88
    )
    
    assessment_contaminated = risk_engine.assess_risk("test_patient_001", atomic_input_contaminated)
    
    print(f"\n✓ Assessment Complete:")
    print(f"  Overall Status: {assessment_contaminated.overall_risk_status.value}")
    print(f"  Risk Score: {assessment_contaminated.overall_risk_score:.1f}/100")
    print(f"  Recommendation: {assessment_contaminated.consumption_recommendation}")
    print(f"  Primary Concerns: {assessment_contaminated.primary_concerns}")
    print(f"  Critical Warnings: {len(assessment_contaminated.critical_warnings)}")
    
    # Test Case 3: Pregnant patient (ultra-strict lead threshold)
    print("\n" + "-"*80)
    print("Test Case 3: Pregnant patient (ultra-strict lead)")
    
    profile_pregnant = profile_engine.create_profile(
        user_id="pregnant_patient_001",
        age=28,
        gender=Gender.FEMALE,
        body_weight_kg=65,
        height_cm=165,
        is_pregnant=True,
        pregnancy_trimester=PregnancyStatus.TRIMESTER_2
    )
    
    pregnancy_condition = HealthCondition(
        condition_id="preg001",
        condition_name="Pregnancy",
        pregnancy_trimester=PregnancyStatus.TRIMESTER_2
    )
    
    profile_engine.add_condition("pregnant_patient_001", pregnancy_condition)
    
    atomic_input_pregnancy = risk_engine.receive_atomic_input(
        food_name="Spinach",
        food_category="leafy_vegetables",
        serving_size_g=100,
        element_predictions={
            'Pb': {'concentration': 0.045, 'std': 0.008, 'confidence_score': 0.90, 'unit': 'ppm'},
            'Fe': {'concentration': 3.5, 'std': 0.3, 'confidence_score': 0.92, 'unit': 'mg/100g'}
        },
        image_quality_score=0.95
    )
    
    assessment_pregnancy = risk_engine.assess_risk("pregnant_patient_001", atomic_input_pregnancy)
    
    print(f"\n✓ Assessment Complete:")
    print(f"  Overall Status: {assessment_pregnancy.overall_risk_status.value}")
    print(f"  Recommendation: {assessment_pregnancy.consumption_recommendation}")
    print(f"  Beneficial Elements: {assessment_pregnancy.beneficial_elements}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_risk_integration_engine()
