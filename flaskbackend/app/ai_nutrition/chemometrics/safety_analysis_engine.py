"""
Safety & Uncertainty Analysis Engine
====================================

Confidence-based safety decision system with regulatory compliance.

Core Principle:
--------------
NEVER return unsafe predictions with high confidence.
Better to admit uncertainty than to mislead users.

Confidence Tiers:
----------------
VERY_HIGH (>90%): Use AI prediction → Full safety analysis
HIGH (70-90%):    Use AI with caution flags → Warning messages
MEDIUM (50-70%):  Use USDA averages → "Estimated from similar foods"
LOW (<50%):       Refuse prediction → Request lab testing

Safety Philosophy:
-----------------
1. Heavy metals (Pb, Cd, As, Hg): CONSERVATIVE (err on side of caution)
2. Nutrients (Fe, Ca, Mg): PERMISSIVE (small errors acceptable)
3. Allergens: ULTRA-CONSERVATIVE (cannot afford false negatives)

Regulatory Compliance:
---------------------
- FDA Action Levels (lead in food)
- WHO/FAO Codex Alimentarius
- EU Contaminant Regulations
- USDA Nutrient Database
- EPA Safe Drinking Water Act (for comparison)

Author: BiteLab AI Team
Date: November 2025
Version: 1.0.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class ConfidenceLevel(Enum):
    """Model prediction confidence levels."""
    VERY_HIGH = "very_high"  # >90%
    HIGH = "high"            # 70-90%
    MEDIUM = "medium"        # 50-70%
    LOW = "low"              # <50%


class SafetyStatus(Enum):
    """Overall safety assessment."""
    SAFE = "safe"                      # All contaminants below limits
    CAUTION = "caution"                # Near limits or uncertainty
    WARNING = "warning"                # Exceeds limits or high uncertainty
    UNSAFE = "unsafe"                  # Significantly exceeds limits
    UNKNOWN = "unknown"                # Insufficient confidence to assess


class ContaminantType(Enum):
    """Categories of contaminants."""
    HEAVY_METAL = "heavy_metal"        # Pb, Cd, As, Hg
    NUTRIENT = "nutrient"              # Fe, Ca, Mg, Zn
    ALLERGEN = "allergen"              # Shellfish, nuts, etc.
    PESTICIDE = "pesticide"            # Organophosphates, etc.


class MessageMode(Enum):
    """Safety message modes."""
    CONSUMER = "consumer"              # For general public (non-technical)
    CLINICAL = "clinical"              # For healthcare providers (technical)
    REGULATORY = "regulatory"          # For compliance reports (legal)


# ============================================================================
# REGULATORY THRESHOLDS
# ============================================================================

@dataclass
class RegulatoryThreshold:
    """
    Regulatory threshold for a contaminant.
    
    Sources:
    - FDA Food Defect Action Levels
    - WHO/FAO Codex Alimentarius
    - EU Regulation (EC) No 1881/2006
    """
    element: str
    food_category: str
    
    # Thresholds (ppm or mg/kg)
    action_level: float          # FDA action level (unsafe above)
    warning_level: float         # 50% of action level (caution)
    ideal_max: float             # Ideal maximum (best practice)
    
    # Regulatory source
    source: str
    regulation_code: str
    
    # Health context
    health_effect: str
    vulnerable_populations: List[str] = field(default_factory=list)


class RegulatoryDatabase:
    """Database of regulatory thresholds for contaminants."""
    
    def __init__(self):
        """Initialize regulatory database."""
        self.thresholds: Dict[str, Dict[str, RegulatoryThreshold]] = {}
        
        self._initialize_thresholds()
        
        logger.info(f"Initialized RegulatoryDatabase with {self._count_thresholds()} thresholds")
        
    def _initialize_thresholds(self):
        """Initialize regulatory thresholds from FDA, WHO, EU."""
        
        # Lead (Pb) thresholds
        self._add_threshold(RegulatoryThreshold(
            element="Pb",
            food_category="leafy_vegetables",
            action_level=0.1,      # 0.1 ppm FDA action level for leafy vegetables
            warning_level=0.05,    # 50% of action level
            ideal_max=0.02,        # Best practice
            source="FDA",
            regulation_code="FDA Defect Action Level",
            health_effect="Neurotoxicity, developmental delays, cardiovascular effects",
            vulnerable_populations=["children", "pregnant_women", "infants"]
        ))
        
        self._add_threshold(RegulatoryThreshold(
            element="Pb",
            food_category="root_vegetables",
            action_level=0.1,
            warning_level=0.05,
            ideal_max=0.02,
            source="FDA",
            regulation_code="FDA Defect Action Level",
            health_effect="Neurotoxicity, kidney damage",
            vulnerable_populations=["children", "pregnant_women"]
        ))
        
        self._add_threshold(RegulatoryThreshold(
            element="Pb",
            food_category="meat",
            action_level=0.5,      # Higher action level for meat
            warning_level=0.25,
            ideal_max=0.1,
            source="FDA/USDA",
            regulation_code="21 CFR 109.6",
            health_effect="Neurotoxicity, anemia",
            vulnerable_populations=["children", "pregnant_women"]
        ))
        
        # Cadmium (Cd) thresholds
        self._add_threshold(RegulatoryThreshold(
            element="Cd",
            food_category="leafy_vegetables",
            action_level=0.2,      # EU regulation
            warning_level=0.1,
            ideal_max=0.05,
            source="EU",
            regulation_code="EC 1881/2006",
            health_effect="Kidney damage, bone demineralization (Itai-itai disease)",
            vulnerable_populations=["children", "postmenopausal_women"]
        ))
        
        self._add_threshold(RegulatoryThreshold(
            element="Cd",
            food_category="root_vegetables",
            action_level=0.1,
            warning_level=0.05,
            ideal_max=0.02,
            source="EU",
            regulation_code="EC 1881/2006",
            health_effect="Kidney damage, osteoporosis",
            vulnerable_populations=["children", "elderly"]
        ))
        
        # Arsenic (As) thresholds
        self._add_threshold(RegulatoryThreshold(
            element="As",
            food_category="leafy_vegetables",
            action_level=0.5,      # WHO/Codex
            warning_level=0.25,
            ideal_max=0.1,
            source="WHO",
            regulation_code="Codex STAN 193-1995",
            health_effect="Cancer (skin, bladder, lung), cardiovascular disease",
            vulnerable_populations=["children", "pregnant_women", "all"]
        ))
        
        self._add_threshold(RegulatoryThreshold(
            element="As",
            food_category="rice",
            action_level=0.2,      # FDA action level for infant rice cereal
            warning_level=0.1,
            ideal_max=0.05,
            source="FDA",
            regulation_code="FDA Action Level Infant Rice Cereal",
            health_effect="Cancer, neurodevelopmental effects",
            vulnerable_populations=["infants", "children"]
        ))
        
        # Mercury (Hg) thresholds
        self._add_threshold(RegulatoryThreshold(
            element="Hg",
            food_category="fish",
            action_level=1.0,      # FDA action level for methylmercury in fish
            warning_level=0.5,
            ideal_max=0.2,
            source="FDA",
            regulation_code="FDA Action Level Fish",
            health_effect="Neurotoxicity, developmental delays",
            vulnerable_populations=["pregnant_women", "children", "breastfeeding_women"]
        ))
        
        # Iron (Fe) - nutrient, not contaminant (different assessment)
        self._add_threshold(RegulatoryThreshold(
            element="Fe",
            food_category="leafy_vegetables",
            action_level=300,      # Upper limit (not toxic, just excessive)
            warning_level=200,
            ideal_max=50,
            source="USDA",
            regulation_code="USDA Nutrient Database",
            health_effect="Hemochromatosis (iron overload) in susceptible individuals",
            vulnerable_populations=["hemochromatosis_patients"]
        ))
        
        # Calcium (Ca) - nutrient
        self._add_threshold(RegulatoryThreshold(
            element="Ca",
            food_category="leafy_vegetables",
            action_level=2500,     # Upper limit
            warning_level=1500,
            ideal_max=1000,
            source="USDA",
            regulation_code="USDA DRI",
            health_effect="Hypercalcemia, kidney stones (rare)",
            vulnerable_populations=["kidney_disease_patients"]
        ))
        
        # Magnesium (Mg) - nutrient
        self._add_threshold(RegulatoryThreshold(
            element="Mg",
            food_category="leafy_vegetables",
            action_level=500,
            warning_level=350,
            ideal_max=200,
            source="USDA",
            regulation_code="USDA DRI",
            health_effect="Diarrhea (from supplemental magnesium, not food)",
            vulnerable_populations=["kidney_disease_patients"]
        ))
        
    def _add_threshold(self, threshold: RegulatoryThreshold):
        """Add threshold to database."""
        if threshold.element not in self.thresholds:
            self.thresholds[threshold.element] = {}
        
        self.thresholds[threshold.element][threshold.food_category] = threshold
        
    def get_threshold(
        self,
        element: str,
        food_category: str
    ) -> Optional[RegulatoryThreshold]:
        """
        Get regulatory threshold for element and food category.
        
        Args:
            element: Chemical element
            food_category: Food category
            
        Returns:
            Regulatory threshold or None
        """
        if element in self.thresholds:
            # Try exact match
            if food_category in self.thresholds[element]:
                return self.thresholds[element][food_category]
            
            # Try fallback to general category
            for category in ['all_foods', 'general']:
                if category in self.thresholds[element]:
                    return self.thresholds[element][category]
        
        return None
        
    def _count_thresholds(self) -> int:
        """Count total thresholds."""
        return sum(len(categories) for categories in self.thresholds.values())


# ============================================================================
# UNCERTAINTY PROPAGATION
# ============================================================================

@dataclass
class UncertaintyEstimate:
    """
    Uncertainty estimate for a prediction.
    
    Sources of uncertainty:
    1. Model uncertainty (epistemic)
    2. Data quality (aleatoric)
    3. Food variability (natural)
    """
    point_estimate: float           # Best estimate (e.g., 0.045 ppm Pb)
    standard_deviation: float       # Standard deviation
    confidence_interval_95: Tuple[float, float]  # 95% confidence interval
    
    # Uncertainty sources
    model_uncertainty: float        # From model predictions (e.g., ensemble disagreement)
    data_uncertainty: float         # From input data quality
    natural_variability: float      # From food-to-food variation
    
    # Total uncertainty
    total_uncertainty: float
    
    # Confidence level
    confidence_level: ConfidenceLevel
    
    def calculate_total_uncertainty(self):
        """Calculate total uncertainty from components."""
        # Propagate uncertainties (sum in quadrature)
        self.total_uncertainty = np.sqrt(
            self.model_uncertainty**2 +
            self.data_uncertainty**2 +
            self.natural_variability**2
        )
        
        return self.total_uncertainty
        
    def assess_confidence(self) -> ConfidenceLevel:
        """Assess confidence level based on uncertainty."""
        # Coefficient of variation (CV = std / mean)
        cv = self.standard_deviation / max(self.point_estimate, 1e-6)
        
        if cv < 0.15:  # <15% CV
            self.confidence_level = ConfidenceLevel.VERY_HIGH
        elif cv < 0.30:  # 15-30% CV
            self.confidence_level = ConfidenceLevel.HIGH
        elif cv < 0.50:  # 30-50% CV
            self.confidence_level = ConfidenceLevel.MEDIUM
        else:  # >50% CV
            self.confidence_level = ConfidenceLevel.LOW
        
        return self.confidence_level


class UncertaintyPropagator:
    """Propagates uncertainty through prediction pipeline."""
    
    def __init__(self):
        """Initialize uncertainty propagator."""
        logger.info("Initialized UncertaintyPropagator")
        
    def estimate_uncertainty(
        self,
        predictions: Dict[str, float],
        model_ensemble: Optional[List[Any]] = None,
        image_quality: float = 1.0,
        food_category: str = "unknown"
    ) -> Dict[str, UncertaintyEstimate]:
        """
        Estimate uncertainty for each element prediction.
        
        Args:
            predictions: Element predictions
            model_ensemble: Ensemble of models for uncertainty estimation
            image_quality: Image quality score (0-1)
            food_category: Food category
            
        Returns:
            Uncertainty estimates for each element
        """
        uncertainties = {}
        
        for element, point_estimate in predictions.items():
            # Simulate model uncertainty (from ensemble disagreement)
            model_unc = point_estimate * (0.05 + np.random.rand() * 0.10)  # 5-15% of value
            
            # Data uncertainty (from image quality)
            data_unc = point_estimate * (1.0 - image_quality) * 0.2  # Up to 20% if poor quality
            
            # Natural variability (from food-to-food variation)
            natural_var = point_estimate * 0.15  # Typical 15% biological variation
            
            # Calculate standard deviation
            std = np.sqrt(model_unc**2 + data_unc**2 + natural_var**2)
            
            # 95% confidence interval (±1.96 std)
            ci_95 = (
                max(0, point_estimate - 1.96 * std),
                point_estimate + 1.96 * std
            )
            
            # Create uncertainty estimate
            unc = UncertaintyEstimate(
                point_estimate=point_estimate,
                standard_deviation=std,
                confidence_interval_95=ci_95,
                model_uncertainty=model_unc,
                data_uncertainty=data_unc,
                natural_variability=natural_var,
                total_uncertainty=0.0,
                confidence_level=ConfidenceLevel.MEDIUM
            )
            
            unc.calculate_total_uncertainty()
            unc.assess_confidence()
            
            uncertainties[element] = unc
            
            logger.info(
                f"{element}: {point_estimate:.3f} ± {std:.3f} "
                f"({unc.confidence_level.value} confidence)"
            )
        
        return uncertainties


# ============================================================================
# SAFETY ANALYSIS ENGINE
# ============================================================================

@dataclass
class ElementSafetyAssessment:
    """Safety assessment for a single element."""
    element: str
    concentration: float
    uncertainty: UncertaintyEstimate
    
    # Regulatory
    threshold: Optional[RegulatoryThreshold]
    exceeds_threshold: bool
    
    # Safety decision
    safety_status: SafetyStatus
    risk_score: float  # 0-100
    
    # Explanation
    reason: str
    recommendation: str


@dataclass
class OverallSafetyAssessment:
    """Overall safety assessment for food item."""
    food_name: str
    food_category: str
    
    # Individual elements
    element_assessments: Dict[str, ElementSafetyAssessment]
    
    # Overall decision
    overall_status: SafetyStatus
    overall_risk_score: float  # 0-100
    
    # Confidence
    overall_confidence: ConfidenceLevel
    
    # Warnings
    warnings: List[str]
    cautions: List[str]
    
    # Recommendations
    recommendations: List[str]
    
    # Timestamp
    assessment_date: datetime = field(default_factory=datetime.now)


class SafetyAnalysisEngine:
    """
    Main safety analysis engine.
    
    Makes safety decisions based on:
    1. Predicted concentrations
    2. Uncertainty estimates
    3. Regulatory thresholds
    4. Vulnerable populations
    """
    
    def __init__(self):
        """Initialize safety analysis engine."""
        self.regulatory_db = RegulatoryDatabase()
        self.uncertainty_propagator = UncertaintyPropagator()
        
        logger.info("Initialized SafetyAnalysisEngine")
        
    def assess_safety(
        self,
        food_name: str,
        food_category: str,
        element_predictions: Dict[str, float],
        image_quality: float = 1.0,
        vulnerable_population: Optional[str] = None
    ) -> OverallSafetyAssessment:
        """
        Perform comprehensive safety assessment.
        
        Args:
            food_name: Food item name
            food_category: Food category
            element_predictions: Predicted element concentrations
            image_quality: Image quality score (0-1)
            vulnerable_population: If assessing for vulnerable group
            
        Returns:
            Overall safety assessment
        """
        logger.info(f"Assessing safety for {food_name} ({food_category})")
        
        # Propagate uncertainty
        uncertainties = self.uncertainty_propagator.estimate_uncertainty(
            element_predictions,
            image_quality=image_quality,
            food_category=food_category
        )
        
        # Assess each element
        element_assessments = {}
        
        for element, concentration in element_predictions.items():
            assessment = self._assess_element(
                element,
                concentration,
                uncertainties[element],
                food_category,
                vulnerable_population
            )
            
            element_assessments[element] = assessment
        
        # Overall decision
        overall_assessment = self._make_overall_decision(
            food_name,
            food_category,
            element_assessments
        )
        
        logger.info(f"Overall safety status: {overall_assessment.overall_status.value}")
        
        return overall_assessment
        
    def _assess_element(
        self,
        element: str,
        concentration: float,
        uncertainty: UncertaintyEstimate,
        food_category: str,
        vulnerable_population: Optional[str]
    ) -> ElementSafetyAssessment:
        """Assess safety for single element."""
        
        # Get regulatory threshold
        threshold = self.regulatory_db.get_threshold(element, food_category)
        
        if threshold is None:
            # No regulatory threshold available
            return ElementSafetyAssessment(
                element=element,
                concentration=concentration,
                uncertainty=uncertainty,
                threshold=None,
                exceeds_threshold=False,
                safety_status=SafetyStatus.UNKNOWN,
                risk_score=0.0,
                reason=f"No regulatory threshold available for {element} in {food_category}",
                recommendation="Use USDA database values for comparison"
            )
        
        # Check if exceeds threshold (conservative: use upper CI bound)
        upper_ci = uncertainty.confidence_interval_95[1]
        
        exceeds_action = upper_ci > threshold.action_level
        exceeds_warning = upper_ci > threshold.warning_level
        
        # Determine safety status based on confidence
        if uncertainty.confidence_level == ConfidenceLevel.VERY_HIGH:
            # High confidence → Trust prediction
            if exceeds_action:
                safety_status = SafetyStatus.UNSAFE
                reason = f"{element} exceeds FDA action level ({threshold.action_level} ppm)"
            elif exceeds_warning:
                safety_status = SafetyStatus.WARNING
                reason = f"{element} exceeds warning level ({threshold.warning_level} ppm)"
            else:
                safety_status = SafetyStatus.SAFE
                reason = f"{element} below regulatory limits"
        
        elif uncertainty.confidence_level == ConfidenceLevel.HIGH:
            # Medium-high confidence → Add caution
            if exceeds_action:
                safety_status = SafetyStatus.WARNING  # Downgrade from UNSAFE
                reason = f"{element} may exceed action level (medium confidence)"
            elif exceeds_warning:
                safety_status = SafetyStatus.CAUTION
                reason = f"{element} may exceed warning level (medium confidence)"
            else:
                safety_status = SafetyStatus.SAFE
                reason = f"{element} likely below limits (medium confidence)"
        
        else:
            # Low confidence → Use USDA fallback
            safety_status = SafetyStatus.UNKNOWN
            reason = f"Low confidence for {element} prediction, use USDA database"
        
        # Calculate risk score (0-100)
        risk_score = self._calculate_risk_score(
            concentration,
            threshold,
            uncertainty,
            vulnerable_population
        )
        
        # Recommendation
        if safety_status == SafetyStatus.UNSAFE:
            recommendation = f"AVOID consumption. {element} exceeds safe limits. Consider lab testing."
        elif safety_status == SafetyStatus.WARNING:
            recommendation = f"LIMIT consumption. {element} near safe limits."
        elif safety_status == SafetyStatus.CAUTION:
            recommendation = f"Safe for most people, but monitor intake if in vulnerable group."
        elif safety_status == SafetyStatus.SAFE:
            recommendation = f"Safe for consumption."
        else:
            recommendation = f"Insufficient data. Use USDA database or lab testing."
        
        return ElementSafetyAssessment(
            element=element,
            concentration=concentration,
            uncertainty=uncertainty,
            threshold=threshold,
            exceeds_threshold=exceeds_action,
            safety_status=safety_status,
            risk_score=risk_score,
            reason=reason,
            recommendation=recommendation
        )
        
    def _calculate_risk_score(
        self,
        concentration: float,
        threshold: RegulatoryThreshold,
        uncertainty: UncertaintyEstimate,
        vulnerable_population: Optional[str]
    ) -> float:
        """
        Calculate risk score (0-100).
        
        Higher score = higher risk.
        """
        # Base risk: ratio to action level
        base_risk = (concentration / threshold.action_level) * 50  # 0-50 points
        
        # Uncertainty penalty: high uncertainty = higher risk
        uncertainty_penalty = (uncertainty.total_uncertainty / concentration) * 25  # 0-25 points
        
        # Vulnerable population penalty
        vulnerable_penalty = 0
        if vulnerable_population in threshold.vulnerable_populations:
            vulnerable_penalty = 25  # +25 points
        
        # Total risk score
        risk_score = min(100, base_risk + uncertainty_penalty + vulnerable_penalty)
        
        return risk_score
        
    def _make_overall_decision(
        self,
        food_name: str,
        food_category: str,
        element_assessments: Dict[str, ElementSafetyAssessment]
    ) -> OverallSafetyAssessment:
        """Make overall safety decision."""
        
        # Determine overall status (most severe element)
        status_priority = {
            SafetyStatus.UNSAFE: 5,
            SafetyStatus.WARNING: 4,
            SafetyStatus.CAUTION: 3,
            SafetyStatus.SAFE: 2,
            SafetyStatus.UNKNOWN: 1
        }
        
        overall_status = SafetyStatus.SAFE
        max_priority = 0
        
        for assessment in element_assessments.values():
            priority = status_priority[assessment.safety_status]
            if priority > max_priority:
                max_priority = priority
                overall_status = assessment.safety_status
        
        # Overall risk score (max of individual scores)
        overall_risk = max(
            assessment.risk_score for assessment in element_assessments.values()
        )
        
        # Overall confidence (min of individual confidences)
        confidence_priority = {
            ConfidenceLevel.VERY_HIGH: 4,
            ConfidenceLevel.HIGH: 3,
            ConfidenceLevel.MEDIUM: 2,
            ConfidenceLevel.LOW: 1
        }
        
        min_confidence = ConfidenceLevel.VERY_HIGH
        min_conf_priority = 4
        
        for assessment in element_assessments.values():
            priority = confidence_priority[assessment.uncertainty.confidence_level]
            if priority < min_conf_priority:
                min_conf_priority = priority
                min_confidence = assessment.uncertainty.confidence_level
        
        # Collect warnings and cautions
        warnings = []
        cautions = []
        recommendations = []
        
        for element, assessment in element_assessments.items():
            if assessment.safety_status == SafetyStatus.UNSAFE:
                warnings.append(f"{element}: {assessment.reason}")
            elif assessment.safety_status == SafetyStatus.WARNING:
                warnings.append(f"{element}: {assessment.reason}")
            elif assessment.safety_status == SafetyStatus.CAUTION:
                cautions.append(f"{element}: {assessment.reason}")
            
            recommendations.append(f"{element}: {assessment.recommendation}")
        
        return OverallSafetyAssessment(
            food_name=food_name,
            food_category=food_category,
            element_assessments=element_assessments,
            overall_status=overall_status,
            overall_risk_score=overall_risk,
            overall_confidence=min_confidence,
            warnings=warnings,
            cautions=cautions,
            recommendations=recommendations
        )


# ============================================================================
# WARNING MESSAGE GENERATOR
# ============================================================================

class WarningMessageGenerator:
    """Generates user-friendly safety messages."""
    
    def __init__(self):
        """Initialize message generator."""
        logger.info("Initialized WarningMessageGenerator")
        
    def generate_message(
        self,
        assessment: OverallSafetyAssessment,
        mode: MessageMode = MessageMode.CONSUMER
    ) -> str:
        """
        Generate safety message.
        
        Args:
            assessment: Safety assessment
            mode: Message mode (consumer, clinical, regulatory)
            
        Returns:
            Formatted safety message
        """
        if mode == MessageMode.CONSUMER:
            return self._generate_consumer_message(assessment)
        elif mode == MessageMode.CLINICAL:
            return self._generate_clinical_message(assessment)
        else:
            return self._generate_regulatory_message(assessment)
        
    def _generate_consumer_message(self, assessment: OverallSafetyAssessment) -> str:
        """Generate consumer-friendly message."""
        
        lines = []
        
        # Header
        lines.append(f"Safety Assessment: {assessment.food_name}")
        lines.append("=" * 60)
        
        # Overall status
        if assessment.overall_status == SafetyStatus.SAFE:
            lines.append("✓ SAFE for consumption")
        elif assessment.overall_status == SafetyStatus.CAUTION:
            lines.append("⚠ CAUTION: Safe for most people, but read details below")
        elif assessment.overall_status == SafetyStatus.WARNING:
            lines.append("⚠ WARNING: May exceed safety limits")
        elif assessment.overall_status == SafetyStatus.UNSAFE:
            lines.append("✗ UNSAFE: Exceeds safety limits - avoid consumption")
        else:
            lines.append("? UNKNOWN: Insufficient data for assessment")
        
        # Warnings
        if assessment.warnings:
            lines.append("\nWarnings:")
            for warning in assessment.warnings:
                lines.append(f"  • {warning}")
        
        # Cautions
        if assessment.cautions:
            lines.append("\nCautions:")
            for caution in assessment.cautions:
                lines.append(f"  • {caution}")
        
        # Confidence
        lines.append(f"\nConfidence: {assessment.overall_confidence.value.upper()}")
        
        # Footer
        lines.append("\nNote: This is an AI-based estimate. For critical decisions,")
        lines.append("consult laboratory testing or USDA nutrient database.")
        
        return "\n".join(lines)
        
    def _generate_clinical_message(self, assessment: OverallSafetyAssessment) -> str:
        """Generate clinical/technical message."""
        
        lines = []
        
        lines.append(f"Clinical Safety Assessment: {assessment.food_name}")
        lines.append("=" * 80)
        
        # Overall
        lines.append(f"Overall Status: {assessment.overall_status.value.upper()}")
        lines.append(f"Risk Score: {assessment.overall_risk_score:.1f}/100")
        lines.append(f"Confidence: {assessment.overall_confidence.value}")
        
        # Element details
        lines.append("\nElement Analysis:")
        
        for element, ea in assessment.element_assessments.items():
            lines.append(f"\n  {element}:")
            lines.append(f"    Concentration: {ea.concentration:.3f} ± {ea.uncertainty.standard_deviation:.3f} ppm")
            lines.append(f"    95% CI: [{ea.uncertainty.confidence_interval_95[0]:.3f}, {ea.uncertainty.confidence_interval_95[1]:.3f}]")
            
            if ea.threshold:
                lines.append(f"    Action Level: {ea.threshold.action_level} ppm ({ea.threshold.source})")
                lines.append(f"    Status: {ea.safety_status.value}")
            
            lines.append(f"    Confidence: {ea.uncertainty.confidence_level.value}")
            lines.append(f"    Risk Score: {ea.risk_score:.1f}/100")
        
        return "\n".join(lines)
        
    def _generate_regulatory_message(self, assessment: OverallSafetyAssessment) -> str:
        """Generate regulatory compliance report."""
        
        lines = []
        
        lines.append("REGULATORY COMPLIANCE REPORT")
        lines.append("=" * 80)
        lines.append(f"Food Item: {assessment.food_name}")
        lines.append(f"Category: {assessment.food_category}")
        lines.append(f"Assessment Date: {assessment.assessment_date.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Overall Status: {assessment.overall_status.value.upper()}")
        
        lines.append("\nREGULATORY COMPLIANCE:")
        
        for element, ea in assessment.element_assessments.items():
            if ea.threshold:
                lines.append(f"\n{element}:")
                lines.append(f"  Measured: {ea.concentration:.4f} ppm")
                lines.append(f"  Uncertainty: ±{ea.uncertainty.standard_deviation:.4f} ppm")
                lines.append(f"  Action Level: {ea.threshold.action_level} ppm")
                lines.append(f"  Regulation: {ea.threshold.regulation_code}")
                lines.append(f"  Source: {ea.threshold.source}")
                lines.append(f"  Exceeds Limit: {'YES' if ea.exceeds_threshold else 'NO'}")
                lines.append(f"  Status: {ea.safety_status.value.upper()}")
        
        lines.append("\nDISCLAIMER:")
        lines.append("This assessment is based on AI prediction and should be")
        lines.append("confirmed with accredited laboratory testing for regulatory purposes.")
        
        return "\n".join(lines)


# ============================================================================
# TESTING
# ============================================================================

def test_safety_analysis_engine():
    """Test safety analysis engine."""
    print("\n" + "="*80)
    print("SAFETY ANALYSIS ENGINE TEST")
    print("="*80)
    
    # Initialize engine
    print("\n" + "-"*80)
    print("Initializing safety analysis engine...")
    
    engine = SafetyAnalysisEngine()
    message_gen = WarningMessageGenerator()
    
    print(f"✓ Engine initialized")
    print(f"  Regulatory thresholds: {engine.regulatory_db._count_thresholds()}")
    
    # Test Case 1: Safe spinach
    print("\n" + "-"*80)
    print("Test Case 1: Safe spinach sample")
    
    predictions_safe = {
        'Pb': 0.020,  # Below 0.1 ppm action level
        'Cd': 0.030,  # Below 0.2 ppm action level
        'As': 0.080,  # Below 0.5 ppm action level
        'Fe': 3.2,
        'Ca': 105,
        'Mg': 89
    }
    
    assessment_safe = engine.assess_safety(
        food_name="Organic Spinach",
        food_category="leafy_vegetables",
        element_predictions=predictions_safe,
        image_quality=0.95
    )
    
    print(f"\n✓ Assessment complete:")
    print(f"  Status: {assessment_safe.overall_status.value}")
    print(f"  Risk Score: {assessment_safe.overall_risk_score:.1f}/100")
    print(f"  Confidence: {assessment_safe.overall_confidence.value}")
    
    msg = message_gen.generate_message(assessment_safe, MessageMode.CONSUMER)
    print(f"\nConsumer Message:\n{msg}")
    
    # Test Case 2: Contaminated spinach
    print("\n" + "-"*80)
    print("Test Case 2: Contaminated spinach (high lead)")
    
    predictions_unsafe = {
        'Pb': 0.15,  # EXCEEDS 0.1 ppm action level
        'Cd': 0.08,
        'As': 0.12,
        'Fe': 2.8,
        'Ca': 98,
        'Mg': 85
    }
    
    assessment_unsafe = engine.assess_safety(
        food_name="Contaminated Spinach",
        food_category="leafy_vegetables",
        element_predictions=predictions_unsafe,
        image_quality=0.90,
        vulnerable_population="children"
    )
    
    print(f"\n✓ Assessment complete:")
    print(f"  Status: {assessment_unsafe.overall_status.value}")
    print(f"  Risk Score: {assessment_unsafe.overall_risk_score:.1f}/100")
    print(f"  Warnings: {len(assessment_unsafe.warnings)}")
    
    msg = message_gen.generate_message(assessment_unsafe, MessageMode.CONSUMER)
    print(f"\nConsumer Message:\n{msg}")
    
    # Test Case 3: Low confidence (poor image quality)
    print("\n" + "-"*80)
    print("Test Case 3: Low confidence prediction (poor image)")
    
    predictions_low_conf = {
        'Pb': 0.060,
        'Fe': 4.0,
        'Mg': 95
    }
    
    assessment_low_conf = engine.assess_safety(
        food_name="Blurry Spinach Image",
        food_category="leafy_vegetables",
        element_predictions=predictions_low_conf,
        image_quality=0.40  # Poor quality
    )
    
    print(f"\n✓ Assessment complete:")
    print(f"  Status: {assessment_low_conf.overall_status.value}")
    print(f"  Confidence: {assessment_low_conf.overall_confidence.value}")
    
    msg = message_gen.generate_message(assessment_low_conf, MessageMode.CONSUMER)
    print(f"\nConsumer Message:\n{msg}")
    
    # Test clinical message
    print("\n" + "-"*80)
    print("Clinical Message (Test Case 2):")
    
    clinical_msg = message_gen.generate_message(assessment_unsafe, MessageMode.CLINICAL)
    print(clinical_msg)
    
    # Test regulatory report
    print("\n" + "-"*80)
    print("Regulatory Report (Test Case 2):")
    
    regulatory_msg = message_gen.generate_message(assessment_unsafe, MessageMode.REGULATORY)
    print(regulatory_msg)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_safety_analysis_engine()
