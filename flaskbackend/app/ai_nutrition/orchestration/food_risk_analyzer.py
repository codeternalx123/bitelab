"""
Food Risk Analyzer with Contaminant Detection
==============================================

Advanced system that detects and analyzes food safety risks including:
- Heavy metals (lead, mercury, cadmium, arsenic) from ICPMS data
- Pesticide residues and chemical contaminants
- Nutrient level analysis from scan data
- Health goal alignment assessment
- Medical condition contraindication detection
- Personalized risk scoring

Features:
- ICPMS data integration for precise contaminant detection
- FDA/WHO/EFSA safety threshold comparisons
- Real-time risk scoring (0-100)
- Personalized health impact analysis
- Medical condition-specific warnings
- Nutritional adequacy vs. health goals
- Alternative food suggestions

Author: Wellomex AI Team
Date: November 2025
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class RiskLevel(Enum):
    """Risk severity levels"""
    SAFE = "safe"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    DANGEROUS = "dangerous"


class ContaminantType(Enum):
    """Types of food contaminants"""
    HEAVY_METAL = "heavy_metal"
    PESTICIDE = "pesticide"
    MYCOTOXIN = "mycotoxin"
    ADDITIVE = "additive"
    ENVIRONMENTAL = "environmental"
    MICROBIOLOGICAL = "microbiological"


class NutrientStatus(Enum):
    """Nutrient level status"""
    DEFICIENT = "deficient"
    LOW = "low"
    ADEQUATE = "adequate"
    HIGH = "high"
    EXCESSIVE = "excessive"


# Safety thresholds (ppm or mg/kg)
HEAVY_METAL_LIMITS = {
    "lead": {
        "general": 0.1,
        "baby_food": 0.01,
        "pregnancy": 0.05,
        "children": 0.05,
        "unit": "ppm"
    },
    "mercury": {
        "general": 0.5,
        "fish": 1.0,
        "pregnancy": 0.3,
        "children": 0.3,
        "unit": "ppm"
    },
    "cadmium": {
        "general": 0.05,
        "vegetables": 0.2,
        "pregnancy": 0.03,
        "unit": "ppm"
    },
    "arsenic": {
        "general": 0.1,
        "rice": 0.2,
        "pregnancy": 0.05,
        "unit": "ppm"
    },
    "chromium": {
        "general": 0.1,
        "unit": "ppm"
    },
    "nickel": {
        "general": 1.0,
        "unit": "ppm"
    }
}

PESTICIDE_LIMITS = {
    "glyphosate": {"limit": 1.75, "unit": "ppm"},
    "chlorpyrifos": {"limit": 0.01, "unit": "ppm"},
    "malathion": {"limit": 0.5, "unit": "ppm"},
    "permethrin": {"limit": 2.0, "unit": "ppm"},
}

# Nutrient RDA (Recommended Daily Allowance) - mg per day
NUTRIENT_RDA = {
    "vitamin_a": {"adult": 900, "pregnancy": 770, "children": 600, "unit": "mcg"},
    "vitamin_c": {"adult": 90, "pregnancy": 85, "children": 50, "unit": "mg"},
    "vitamin_d": {"adult": 20, "pregnancy": 15, "children": 15, "unit": "mcg"},
    "vitamin_e": {"adult": 15, "pregnancy": 15, "children": 11, "unit": "mg"},
    "calcium": {"adult": 1000, "pregnancy": 1000, "children": 1300, "unit": "mg"},
    "iron": {"adult": 8, "pregnancy": 27, "children": 11, "unit": "mg"},
    "magnesium": {"adult": 420, "pregnancy": 350, "children": 240, "unit": "mg"},
    "zinc": {"adult": 11, "pregnancy": 11, "children": 8, "unit": "mg"},
    "potassium": {"adult": 3400, "pregnancy": 2900, "children": 2300, "unit": "mg"},
    "selenium": {"adult": 55, "pregnancy": 60, "children": 40, "unit": "mcg"},
}


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ContaminantDetection:
    """Individual contaminant detection result"""
    contaminant_name: str
    contaminant_type: ContaminantType
    detected_level: float
    safe_limit: float
    unit: str
    risk_level: RiskLevel
    exceeds_limit: bool
    exceedance_factor: float  # How many times over the limit
    health_effects: List[str] = field(default_factory=list)
    affected_populations: List[str] = field(default_factory=list)
    regulatory_source: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "contaminant_name": self.contaminant_name,
            "contaminant_type": self.contaminant_type.value,
            "detected_level": self.detected_level,
            "safe_limit": self.safe_limit,
            "unit": self.unit,
            "risk_level": self.risk_level.value,
            "exceeds_limit": self.exceeds_limit,
            "exceedance_factor": round(self.exceedance_factor, 2),
            "health_effects": self.health_effects,
            "affected_populations": self.affected_populations,
            "regulatory_source": self.regulatory_source
        }


@dataclass
class NutrientAnalysis:
    """Nutrient level analysis"""
    nutrient_name: str
    detected_level: float
    rda_target: float
    unit: str
    status: NutrientStatus
    percent_rda: float
    is_adequate: bool
    health_benefits: List[str] = field(default_factory=list)
    deficiency_risks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "nutrient_name": self.nutrient_name,
            "detected_level": self.detected_level,
            "rda_target": self.rda_target,
            "unit": self.unit,
            "status": self.status.value,
            "percent_rda": round(self.percent_rda, 1),
            "is_adequate": self.is_adequate,
            "health_benefits": self.health_benefits,
            "deficiency_risks": self.deficiency_risks
        }


@dataclass
class HealthGoalAlignment:
    """Alignment with user's health goals"""
    goal_name: str
    alignment_score: float  # 0-100
    is_aligned: bool
    supporting_nutrients: List[str] = field(default_factory=list)
    conflicting_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "goal_name": self.goal_name,
            "alignment_score": round(self.alignment_score, 1),
            "is_aligned": self.is_aligned,
            "supporting_nutrients": self.supporting_nutrients,
            "conflicting_factors": self.conflicting_factors,
            "recommendations": self.recommendations
        }


@dataclass
class MedicalConditionCheck:
    """Medical condition contraindication check"""
    condition_name: str
    is_safe: bool
    risk_level: RiskLevel
    contraindications: List[str] = field(default_factory=list)
    beneficial_aspects: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    max_safe_serving: Optional[float] = None  # grams
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "condition_name": self.condition_name,
            "is_safe": self.is_safe,
            "risk_level": self.risk_level.value,
            "contraindications": self.contraindications,
            "beneficial_aspects": self.beneficial_aspects,
            "warnings": self.warnings,
            "max_safe_serving": self.max_safe_serving
        }


@dataclass
class ComprehensiveFoodRiskAnalysis:
    """Complete food risk and health analysis"""
    food_name: str
    analyzed_at: datetime
    
    # Contaminant analysis
    contaminants: List[ContaminantDetection] = field(default_factory=list)
    overall_contaminant_risk: RiskLevel = RiskLevel.SAFE
    is_safe_to_consume: bool = True
    
    # Nutrient analysis
    nutrients: List[NutrientAnalysis] = field(default_factory=list)
    overall_nutrient_score: float = 0.0  # 0-100
    
    # Health alignment
    goal_alignments: List[HealthGoalAlignment] = field(default_factory=list)
    overall_goal_alignment: float = 0.0  # 0-100
    
    # Medical conditions
    condition_checks: List[MedicalConditionCheck] = field(default_factory=list)
    safe_for_all_conditions: bool = True
    
    # Summary
    overall_risk_score: float = 0.0  # 0-100 (lower is better)
    overall_health_score: float = 0.0  # 0-100 (higher is better)
    recommendation: str = ""
    critical_warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # Alternatives
    safer_alternatives: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "food_name": self.food_name,
            "analyzed_at": self.analyzed_at.isoformat(),
            "contaminants": [c.to_dict() for c in self.contaminants],
            "overall_contaminant_risk": self.overall_contaminant_risk.value,
            "is_safe_to_consume": self.is_safe_to_consume,
            "nutrients": [n.to_dict() for n in self.nutrients],
            "overall_nutrient_score": round(self.overall_nutrient_score, 1),
            "goal_alignments": [g.to_dict() for g in self.goal_alignments],
            "overall_goal_alignment": round(self.overall_goal_alignment, 1),
            "condition_checks": [c.to_dict() for c in self.condition_checks],
            "safe_for_all_conditions": self.safe_for_all_conditions,
            "overall_risk_score": round(self.overall_risk_score, 1),
            "overall_health_score": round(self.overall_health_score, 1),
            "recommendation": self.recommendation,
            "critical_warnings": self.critical_warnings,
            "suggestions": self.suggestions,
            "safer_alternatives": self.safer_alternatives
        }


# ============================================================================
# FOOD RISK ANALYZER
# ============================================================================

class FoodRiskAnalyzer:
    """
    Comprehensive food risk and health analyzer
    
    Analyzes:
    1. Contaminants from ICPMS data (heavy metals, pesticides)
    2. Nutrient levels from scan data
    3. Health goal alignment
    4. Medical condition contraindications
    5. Overall safety and recommendations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize food risk analyzer
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        logger.info("FoodRiskAnalyzer initialized")
    
    async def analyze_food_risks(
        self,
        food_name: str,
        icpms_data: Optional[Dict[str, float]] = None,
        scan_data: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        serving_size_g: float = 100.0
    ) -> ComprehensiveFoodRiskAnalysis:
        """
        Perform comprehensive food risk analysis
        
        Args:
            food_name: Name of the food
            icpms_data: ICPMS element detection data {element: ppm}
            scan_data: Food scan data with nutrients
            user_profile: User health profile
            serving_size_g: Serving size in grams
            
        Returns:
            ComprehensiveFoodRiskAnalysis with all results
        """
        logger.info(f"Analyzing food risks for: {food_name}")
        
        analysis = ComprehensiveFoodRiskAnalysis(
            food_name=food_name,
            analyzed_at=datetime.now()
        )
        
        # 1. Analyze contaminants from ICPMS data
        if icpms_data:
            analysis.contaminants = self._analyze_contaminants(
                icpms_data, user_profile
            )
            analysis.overall_contaminant_risk = self._calculate_contaminant_risk(
                analysis.contaminants
            )
            analysis.is_safe_to_consume = analysis.overall_contaminant_risk not in [
                RiskLevel.CRITICAL, RiskLevel.DANGEROUS
            ]
        
        # 2. Analyze nutrients from scan data
        if scan_data:
            analysis.nutrients = self._analyze_nutrients(
                scan_data, user_profile, serving_size_g
            )
            analysis.overall_nutrient_score = self._calculate_nutrient_score(
                analysis.nutrients
            )
        
        # 3. Check health goal alignment
        if user_profile and user_profile.get("health_goals"):
            analysis.goal_alignments = self._check_goal_alignment(
                user_profile["health_goals"],
                analysis.nutrients,
                analysis.contaminants,
                scan_data or {}
            )
            analysis.overall_goal_alignment = self._calculate_goal_alignment_score(
                analysis.goal_alignments
            )
        
        # 4. Check medical condition contraindications
        if user_profile and user_profile.get("medical_conditions"):
            analysis.condition_checks = self._check_medical_conditions(
                user_profile["medical_conditions"],
                analysis.contaminants,
                analysis.nutrients,
                scan_data or {}
            )
            analysis.safe_for_all_conditions = all(
                c.is_safe for c in analysis.condition_checks
            )
        
        # 5. Calculate overall scores
        analysis.overall_risk_score = self._calculate_overall_risk_score(analysis)
        analysis.overall_health_score = self._calculate_overall_health_score(analysis)
        
        # 6. Generate recommendations
        analysis.recommendation = self._generate_recommendation(analysis)
        analysis.critical_warnings = self._generate_warnings(analysis)
        analysis.suggestions = self._generate_suggestions(analysis)
        
        # 7. Find safer alternatives if needed
        if analysis.overall_risk_score > 60 or not analysis.is_safe_to_consume:
            analysis.safer_alternatives = self._find_safer_alternatives(
                food_name, user_profile
            )
        
        logger.info(f"Risk analysis complete. Safety: {analysis.is_safe_to_consume}, "
                   f"Risk Score: {analysis.overall_risk_score:.1f}")
        
        return analysis
    
    def _analyze_contaminants(
        self,
        icpms_data: Dict[str, float],
        user_profile: Optional[Dict[str, Any]]
    ) -> List[ContaminantDetection]:
        """Analyze contaminants from ICPMS data"""
        contaminants = []
        
        # Determine user category for limits
        user_category = self._get_user_category(user_profile)
        
        # Check heavy metals
        for element, detected_ppm in icpms_data.items():
            element_lower = element.lower()
            
            # Heavy metals
            if element_lower in HEAVY_METAL_LIMITS:
                limits = HEAVY_METAL_LIMITS[element_lower]
                safe_limit = limits.get(user_category, limits["general"])
                
                exceeds = detected_ppm > safe_limit
                factor = detected_ppm / safe_limit if safe_limit > 0 else 0
                
                risk_level = self._determine_risk_level(factor)
                
                health_effects = self._get_health_effects(element_lower)
                affected_pops = self._get_affected_populations(element_lower)
                
                contaminants.append(ContaminantDetection(
                    contaminant_name=element,
                    contaminant_type=ContaminantType.HEAVY_METAL,
                    detected_level=detected_ppm,
                    safe_limit=safe_limit,
                    unit=limits["unit"],
                    risk_level=risk_level,
                    exceeds_limit=exceeds,
                    exceedance_factor=factor,
                    health_effects=health_effects,
                    affected_populations=affected_pops,
                    regulatory_source="FDA/WHO/EFSA"
                ))
        
        return contaminants
    
    def _analyze_nutrients(
        self,
        scan_data: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]],
        serving_size_g: float
    ) -> List[NutrientAnalysis]:
        """Analyze nutrient levels from scan data"""
        nutrients = []
        
        # Get user category
        user_category = self._get_user_category(user_profile)
        
        # Extract nutrients from scan data
        nutrient_data = scan_data.get("nutrients", {})
        
        for nutrient_key, detected_amount in nutrient_data.items():
            nutrient_lower = nutrient_key.lower()
            
            if nutrient_lower in NUTRIENT_RDA:
                rda_info = NUTRIENT_RDA[nutrient_lower]
                rda_target = rda_info.get(user_category, rda_info["adult"])
                
                # Scale to serving size (assume detected_amount is per 100g)
                scaled_amount = detected_amount * (serving_size_g / 100.0)
                
                percent_rda = (scaled_amount / rda_target) * 100 if rda_target > 0 else 0
                
                status = self._determine_nutrient_status(percent_rda)
                is_adequate = percent_rda >= 20  # At least 20% of RDA
                
                benefits = self._get_nutrient_benefits(nutrient_lower)
                deficiency_risks = self._get_deficiency_risks(nutrient_lower)
                
                nutrients.append(NutrientAnalysis(
                    nutrient_name=nutrient_key,
                    detected_level=scaled_amount,
                    rda_target=rda_target,
                    unit=rda_info["unit"],
                    status=status,
                    percent_rda=percent_rda,
                    is_adequate=is_adequate,
                    health_benefits=benefits,
                    deficiency_risks=deficiency_risks
                ))
        
        return nutrients
    
    def _check_goal_alignment(
        self,
        health_goals: List[str],
        nutrients: List[NutrientAnalysis],
        contaminants: List[ContaminantDetection],
        scan_data: Dict[str, Any]
    ) -> List[HealthGoalAlignment]:
        """Check alignment with health goals"""
        alignments = []
        
        for goal in health_goals:
            goal_lower = goal.lower()
            
            alignment_score = 50.0  # Neutral baseline
            supporting = []
            conflicting = []
            recommendations = []
            
            # Weight loss
            if "weight loss" in goal_lower or "lose weight" in goal_lower:
                calories = scan_data.get("calories", 0)
                protein_g = scan_data.get("protein_g", 0)
                fiber_g = scan_data.get("fiber_g", 0)
                
                if calories < 200:
                    alignment_score += 20
                    supporting.append("Low calorie content")
                if protein_g > 15:
                    alignment_score += 15
                    supporting.append("High protein for satiety")
                if fiber_g > 5:
                    alignment_score += 15
                    supporting.append("High fiber for fullness")
                
                sugar_g = scan_data.get("sugar_g", 0)
                if sugar_g > 10:
                    alignment_score -= 20
                    conflicting.append("High sugar content")
                
                recommendations.append("Pair with vegetables for volume")
            
            # Muscle gain
            elif "muscle" in goal_lower or "strength" in goal_lower:
                protein_g = scan_data.get("protein_g", 0)
                leucine = scan_data.get("leucine", 0)
                
                if protein_g > 20:
                    alignment_score += 30
                    supporting.append("Excellent protein content")
                if leucine > 1500:
                    alignment_score += 20
                    supporting.append("Rich in leucine for muscle synthesis")
                
                recommendations.append("Consume within 2 hours post-workout")
            
            # Heart health
            elif "heart" in goal_lower or "cardiovascular" in goal_lower:
                omega3 = scan_data.get("omega3_total", 0)
                fiber_g = scan_data.get("fiber_g", 0)
                sodium_mg = scan_data.get("sodium_mg", 0)
                
                if omega3 > 500:
                    alignment_score += 25
                    supporting.append("Rich in omega-3 fatty acids")
                if fiber_g > 5:
                    alignment_score += 15
                    supporting.append("High fiber lowers cholesterol")
                if sodium_mg > 500:
                    alignment_score -= 25
                    conflicting.append("High sodium raises blood pressure")
                
                # Check for lead (damages cardiovascular system)
                lead_contaminants = [c for c in contaminants if "lead" in c.contaminant_name.lower()]
                if lead_contaminants and lead_contaminants[0].exceeds_limit:
                    alignment_score -= 30
                    conflicting.append("Lead contamination harms heart health")
            
            # Brain health
            elif "brain" in goal_lower or "cognitive" in goal_lower:
                dha = scan_data.get("dha", 0)
                b_vitamins = scan_data.get("vitamin_b12", 0) + scan_data.get("folate", 0)
                
                if dha > 500:
                    alignment_score += 30
                    supporting.append("DHA supports brain function")
                if b_vitamins > 100:
                    alignment_score += 15
                    supporting.append("B vitamins support cognition")
                
                # Mercury is neurotoxic
                mercury_contaminants = [c for c in contaminants if "mercury" in c.contaminant_name.lower()]
                if mercury_contaminants and mercury_contaminants[0].exceeds_limit:
                    alignment_score -= 40
                    conflicting.append("Mercury contamination damages brain")
            
            # Diabetes management
            elif "diabetes" in goal_lower or "blood sugar" in goal_lower:
                fiber_g = scan_data.get("fiber_g", 0)
                sugar_g = scan_data.get("sugar_g", 0)
                
                if fiber_g > 5:
                    alignment_score += 25
                    supporting.append("Fiber slows glucose absorption")
                if sugar_g < 5:
                    alignment_score += 20
                    supporting.append("Low sugar content")
                elif sugar_g > 15:
                    alignment_score -= 30
                    conflicting.append("High sugar spikes blood glucose")
            
            # Bone health
            elif "bone" in goal_lower or "osteoporosis" in goal_lower:
                calcium_nutrients = [n for n in nutrients if "calcium" in n.nutrient_name.lower()]
                vitamin_d = [n for n in nutrients if "vitamin_d" in n.nutrient_name.lower()]
                
                if calcium_nutrients and calcium_nutrients[0].percent_rda > 20:
                    alignment_score += 25
                    supporting.append("Good calcium source")
                if vitamin_d and vitamin_d[0].percent_rda > 20:
                    alignment_score += 20
                    supporting.append("Vitamin D aids calcium absorption")
            
            # Ensure score is 0-100
            alignment_score = max(0, min(100, alignment_score))
            
            alignments.append(HealthGoalAlignment(
                goal_name=goal,
                alignment_score=alignment_score,
                is_aligned=alignment_score >= 60,
                supporting_nutrients=supporting,
                conflicting_factors=conflicting,
                recommendations=recommendations
            ))
        
        return alignments
    
    def _check_medical_conditions(
        self,
        medical_conditions: List[str],
        contaminants: List[ContaminantDetection],
        nutrients: List[NutrientAnalysis],
        scan_data: Dict[str, Any]
    ) -> List[MedicalConditionCheck]:
        """Check contraindications for medical conditions"""
        checks = []
        
        for condition in medical_conditions:
            condition_lower = condition.lower()
            
            is_safe = True
            risk_level = RiskLevel.SAFE
            contraindications = []
            beneficial = []
            warnings = []
            max_serving = None
            
            # Pregnancy
            if "pregnan" in condition_lower:
                # Lead is extremely dangerous during pregnancy
                lead_contaminants = [c for c in contaminants if "lead" in c.contaminant_name.lower()]
                if lead_contaminants and lead_contaminants[0].exceeds_limit:
                    is_safe = False
                    risk_level = RiskLevel.CRITICAL
                    contraindications.append("Lead crosses placenta and harms fetal brain development")
                    contraindications.append(f"Lead level: {lead_contaminants[0].detected_level:.2f} ppm (Safe limit: {lead_contaminants[0].safe_limit:.2f} ppm)")
                
                # Mercury is also dangerous
                mercury_contaminants = [c for c in contaminants if "mercury" in c.contaminant_name.lower()]
                if mercury_contaminants and mercury_contaminants[0].exceeds_limit:
                    is_safe = False
                    risk_level = RiskLevel.CRITICAL
                    contraindications.append("Methylmercury damages fetal nervous system")
                
                # Check for adequate folate
                folate_nutrients = [n for n in nutrients if "folate" in n.nutrient_name.lower() or "folic" in n.nutrient_name.lower()]
                if folate_nutrients and folate_nutrients[0].is_adequate:
                    beneficial.append("Adequate folate prevents neural tube defects")
                
                # Iron needs
                iron_nutrients = [n for n in nutrients if "iron" in n.nutrient_name.lower()]
                if iron_nutrients and iron_nutrients[0].percent_rda > 30:
                    beneficial.append("Good iron source for pregnancy")
            
            # Kidney disease
            elif "kidney" in condition_lower or "renal" in condition_lower:
                potassium_mg = scan_data.get("potassium_mg", 0)
                phosphorus_mg = scan_data.get("phosphorus_mg", 0)
                sodium_mg = scan_data.get("sodium_mg", 0)
                protein_g = scan_data.get("protein_g", 0)
                
                if potassium_mg > 400:
                    risk_level = RiskLevel.HIGH
                    warnings.append("High potassium - dangerous for kidney disease")
                    max_serving = 50.0
                
                if phosphorus_mg > 200:
                    risk_level = RiskLevel.MODERATE
                    warnings.append("High phosphorus - limit intake")
                
                if sodium_mg > 300:
                    warnings.append("High sodium - worsens kidney function")
                
                if protein_g > 20:
                    warnings.append("High protein - may strain kidneys")
                    max_serving = 75.0
            
            # Diabetes
            elif "diabetes" in condition_lower:
                sugar_g = scan_data.get("sugar_g", 0)
                carbs_g = scan_data.get("carbohydrates_g", 0)
                fiber_g = scan_data.get("fiber_g", 0)
                
                if sugar_g > 20:
                    risk_level = RiskLevel.MODERATE
                    warnings.append("High sugar - monitor blood glucose closely")
                    max_serving = 50.0
                
                if carbs_g > 30 and fiber_g < 3:
                    warnings.append("High refined carbs - may spike blood sugar")
                
                if fiber_g > 5:
                    beneficial.append("High fiber helps control blood sugar")
            
            # Hypertension
            elif "hypertension" in condition_lower or "high blood pressure" in condition_lower:
                sodium_mg = scan_data.get("sodium_mg", 0)
                potassium_mg = scan_data.get("potassium_mg", 0)
                
                if sodium_mg > 500:
                    risk_level = RiskLevel.HIGH
                    warnings.append("High sodium - raises blood pressure")
                    max_serving = 50.0
                
                if potassium_mg > 300:
                    beneficial.append("Good potassium source - helps lower blood pressure")
            
            # Celiac disease / gluten intolerance
            elif "celiac" in condition_lower or "gluten" in condition_lower:
                contains_gluten = scan_data.get("contains_gluten", False)
                if contains_gluten:
                    is_safe = False
                    risk_level = RiskLevel.CRITICAL
                    contraindications.append("Contains gluten - triggers celiac immune response")
            
            # Liver disease
            elif "liver" in condition_lower or "hepatic" in condition_lower:
                # Heavy metals are processed by liver
                critical_metals = [c for c in contaminants if c.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
                if critical_metals:
                    risk_level = RiskLevel.HIGH
                    warnings.append("Heavy metal contamination - extra burden on liver")
                
                protein_g = scan_data.get("protein_g", 0)
                if protein_g > 25:
                    warnings.append("High protein - may challenge liver function")
            
            checks.append(MedicalConditionCheck(
                condition_name=condition,
                is_safe=is_safe,
                risk_level=risk_level,
                contraindications=contraindications,
                beneficial_aspects=beneficial,
                warnings=warnings,
                max_safe_serving=max_serving
            ))
        
        return checks
    
    def _calculate_contaminant_risk(
        self,
        contaminants: List[ContaminantDetection]
    ) -> RiskLevel:
        """Calculate overall contaminant risk level"""
        if not contaminants:
            return RiskLevel.SAFE
        
        # Get highest risk level
        risk_levels = [c.risk_level for c in contaminants]
        
        if RiskLevel.DANGEROUS in risk_levels or RiskLevel.CRITICAL in risk_levels:
            return RiskLevel.CRITICAL
        elif RiskLevel.HIGH in risk_levels:
            return RiskLevel.HIGH
        elif RiskLevel.MODERATE in risk_levels:
            return RiskLevel.MODERATE
        elif RiskLevel.LOW in risk_levels:
            return RiskLevel.LOW
        else:
            return RiskLevel.SAFE
    
    def _calculate_nutrient_score(
        self,
        nutrients: List[NutrientAnalysis]
    ) -> float:
        """Calculate overall nutrient adequacy score (0-100)"""
        if not nutrients:
            return 50.0
        
        # Average percent RDA (capped at 100%)
        total_percent = sum(min(n.percent_rda, 100) for n in nutrients)
        avg_percent = total_percent / len(nutrients)
        
        return avg_percent
    
    def _calculate_goal_alignment_score(
        self,
        alignments: List[HealthGoalAlignment]
    ) -> float:
        """Calculate overall goal alignment score"""
        if not alignments:
            return 50.0
        
        total = sum(a.alignment_score for a in alignments)
        return total / len(alignments)
    
    def _calculate_overall_risk_score(
        self,
        analysis: ComprehensiveFoodRiskAnalysis
    ) -> float:
        """
        Calculate overall risk score (0-100, lower is better)
        
        Factors:
        - Contaminant risk (40%)
        - Medical condition safety (30%)
        - Goal misalignment (20%)
        - Nutrient deficiency (10%)
        """
        score = 0.0
        
        # Contaminant risk (0-40 points)
        contaminant_risk_map = {
            RiskLevel.SAFE: 0,
            RiskLevel.LOW: 10,
            RiskLevel.MODERATE: 20,
            RiskLevel.HIGH: 35,
            RiskLevel.CRITICAL: 40,
            RiskLevel.DANGEROUS: 40
        }
        score += contaminant_risk_map.get(analysis.overall_contaminant_risk, 0)
        
        # Medical condition safety (0-30 points)
        if not analysis.safe_for_all_conditions:
            unsafe_conditions = [c for c in analysis.condition_checks if not c.is_safe]
            score += min(30, len(unsafe_conditions) * 15)
        
        # Goal misalignment (0-20 points)
        if analysis.goal_alignments:
            avg_misalignment = 100 - analysis.overall_goal_alignment
            score += (avg_misalignment / 100) * 20
        
        # Nutrient deficiency (0-10 points)
        if analysis.overall_nutrient_score < 50:
            score += 10 - (analysis.overall_nutrient_score / 5)
        
        return min(100, score)
    
    def _calculate_overall_health_score(
        self,
        analysis: ComprehensiveFoodRiskAnalysis
    ) -> float:
        """
        Calculate overall health benefit score (0-100, higher is better)
        
        Inverse of risk score + nutrient quality
        """
        base_score = 100 - analysis.overall_risk_score
        
        # Boost for high nutrient content
        nutrient_boost = (analysis.overall_nutrient_score - 50) / 2
        
        # Boost for goal alignment
        goal_boost = (analysis.overall_goal_alignment - 50) / 2
        
        return max(0, min(100, base_score + nutrient_boost + goal_boost))
    
    def _generate_recommendation(
        self,
        analysis: ComprehensiveFoodRiskAnalysis
    ) -> str:
        """Generate overall recommendation"""
        if not analysis.is_safe_to_consume:
            return f"⛔ DO NOT CONSUME - Critical safety concerns detected"
        
        if analysis.overall_risk_score >= 70:
            return f"❌ NOT RECOMMENDED - High risk score ({analysis.overall_risk_score:.0f}/100)"
        elif analysis.overall_risk_score >= 50:
            return f"⚠️ CONSUME WITH CAUTION - Moderate risk factors present"
        elif analysis.overall_health_score >= 70:
            return f"✅ RECOMMENDED - Good nutritional profile and low risk"
        elif analysis.overall_health_score >= 50:
            return f"👍 ACCEPTABLE - Adequate nutrition, minimal risks"
        else:
            return f"➖ NEUTRAL - No major concerns but limited health benefits"
    
    def _generate_warnings(
        self,
        analysis: ComprehensiveFoodRiskAnalysis
    ) -> List[str]:
        """Generate critical warnings"""
        warnings = []
        
        # Contaminant warnings
        for contaminant in analysis.contaminants:
            if contaminant.exceeds_limit:
                warnings.append(
                    f"⚠️ {contaminant.contaminant_name} exceeds safe limit by "
                    f"{contaminant.exceedance_factor:.1f}x ({contaminant.detected_level:.2f} {contaminant.unit})"
                )
        
        # Medical condition warnings
        for check in analysis.condition_checks:
            if not check.is_safe:
                warnings.append(
                    f"⛔ DANGEROUS for {check.condition_name}: {', '.join(check.contraindications[:2])}"
                )
            elif check.warnings:
                warnings.append(
                    f"⚠️ {check.condition_name}: {check.warnings[0]}"
                )
        
        return warnings
    
    def _generate_suggestions(
        self,
        analysis: ComprehensiveFoodRiskAnalysis
    ) -> List[str]:
        """Generate helpful suggestions"""
        suggestions = []
        
        # Goal alignment suggestions
        for goal in analysis.goal_alignments:
            if goal.is_aligned and goal.recommendations:
                suggestions.append(f"💡 {goal.goal_name}: {goal.recommendations[0]}")
        
        # Nutrient suggestions
        deficient_nutrients = [n for n in analysis.nutrients if n.status == NutrientStatus.DEFICIENT]
        if deficient_nutrients and len(deficient_nutrients) <= 3:
            nutrient_names = ", ".join([n.nutrient_name for n in deficient_nutrients[:3]])
            suggestions.append(f"💊 Consider supplementing: {nutrient_names}")
        
        # Medical condition suggestions
        for check in analysis.condition_checks:
            if check.beneficial_aspects:
                suggestions.append(f"✨ {check.condition_name}: {check.beneficial_aspects[0]}")
        
        return suggestions
    
    def _find_safer_alternatives(
        self,
        food_name: str,
        user_profile: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Find safer alternative foods"""
        # Simple alternatives based on food type
        alternatives = []
        
        food_lower = food_name.lower()
        
        if "spinach" in food_lower:
            alternatives = ["Kale (lower heavy metals)", "Swiss chard", "Romaine lettuce"]
        elif "fish" in food_lower or "tuna" in food_lower:
            alternatives = ["Wild salmon (lower mercury)", "Sardines", "Anchovies"]
        elif "rice" in food_lower:
            alternatives = ["Quinoa (lower arsenic)", "Cauliflower rice", "Millet"]
        elif "apple" in food_lower:
            alternatives = ["Organic apples (lower pesticides)", "Pears", "Berries"]
        else:
            alternatives = ["Organic version of same food", "Locally sourced alternatives"]
        
        return alternatives
    
    # Helper methods
    
    def _get_user_category(self, user_profile: Optional[Dict[str, Any]]) -> str:
        """Determine user category for limits"""
        if not user_profile:
            return "adult"
        
        conditions = [c.lower() for c in user_profile.get("medical_conditions", [])]
        age = user_profile.get("age", 30)
        
        if any("pregnan" in c for c in conditions):
            return "pregnancy"
        elif age < 18:
            return "children"
        else:
            return "adult"
    
    def _determine_risk_level(self, exceedance_factor: float) -> RiskLevel:
        """Determine risk level from exceedance factor"""
        if exceedance_factor < 0.5:
            return RiskLevel.SAFE
        elif exceedance_factor < 1.0:
            return RiskLevel.LOW
        elif exceedance_factor < 2.0:
            return RiskLevel.MODERATE
        elif exceedance_factor < 5.0:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _determine_nutrient_status(self, percent_rda: float) -> NutrientStatus:
        """Determine nutrient status from percent RDA"""
        if percent_rda < 10:
            return NutrientStatus.DEFICIENT
        elif percent_rda < 50:
            return NutrientStatus.LOW
        elif percent_rda < 150:
            return NutrientStatus.ADEQUATE
        elif percent_rda < 300:
            return NutrientStatus.HIGH
        else:
            return NutrientStatus.EXCESSIVE
    
    def _get_health_effects(self, element: str) -> List[str]:
        """Get health effects for element"""
        effects_map = {
            "lead": ["Brain damage", "Reduced IQ in children", "Kidney damage", "Hypertension"],
            "mercury": ["Neurological damage", "Kidney damage", "Tremors", "Memory loss"],
            "cadmium": ["Kidney disease", "Bone fragility", "Lung damage", "Cancer"],
            "arsenic": ["Cancer", "Skin lesions", "Cardiovascular disease", "Diabetes"],
            "chromium": ["Skin irritation", "Lung cancer (hexavalent form)", "Liver damage"],
            "nickel": ["Allergic dermatitis", "Respiratory issues", "Cancer (inhalation)"]
        }
        return effects_map.get(element, ["Unknown health effects"])
    
    def _get_affected_populations(self, element: str) -> List[str]:
        """Get populations most affected by element"""
        populations_map = {
            "lead": ["Children", "Pregnant women", "Fetuses"],
            "mercury": ["Pregnant women", "Fetuses", "Young children"],
            "cadmium": ["Smokers", "Post-menopausal women", "People with kidney disease"],
            "arsenic": ["Children", "Pregnant women", "People with liver disease"],
            "chromium": ["Industrial workers", "People with skin sensitivity"],
            "nickel": ["People with nickel allergy", "Occupational exposure workers"]
        }
        return populations_map.get(element, ["General population"])
    
    def _get_nutrient_benefits(self, nutrient: str) -> List[str]:
        """Get health benefits of nutrient"""
        benefits_map = {
            "vitamin_a": ["Vision health", "Immune function", "Skin health"],
            "vitamin_c": ["Immune support", "Antioxidant", "Collagen production"],
            "vitamin_d": ["Bone health", "Immune function", "Mood regulation"],
            "calcium": ["Bone strength", "Muscle function", "Nerve signaling"],
            "iron": ["Oxygen transport", "Energy production", "Immune function"],
            "magnesium": ["Muscle relaxation", "Heart rhythm", "Energy production"],
            "zinc": ["Immune function", "Wound healing", "DNA synthesis"],
            "potassium": ["Blood pressure regulation", "Heart function", "Muscle function"]
        }
        return benefits_map.get(nutrient, ["General health support"])
    
    def _get_deficiency_risks(self, nutrient: str) -> List[str]:
        """Get risks of nutrient deficiency"""
        risks_map = {
            "vitamin_a": ["Night blindness", "Dry skin", "Weak immunity"],
            "vitamin_c": ["Scurvy", "Slow wound healing", "Bleeding gums"],
            "vitamin_d": ["Weak bones", "Osteoporosis", "Muscle weakness"],
            "calcium": ["Osteoporosis", "Muscle cramps", "Weak bones"],
            "iron": ["Anemia", "Fatigue", "Weakness"],
            "magnesium": ["Muscle cramps", "Fatigue", "Heart palpitations"],
            "zinc": ["Weak immunity", "Hair loss", "Poor wound healing"],
            "potassium": ["Muscle weakness", "Heart palpitations", "Fatigue"]
        }
        return risks_map.get(nutrient, ["Unknown deficiency effects"])


# ============================================================================
# TESTING
# ============================================================================

async def test_food_risk_analyzer():
    """Test the food risk analyzer"""
    print("=" * 80)
    print("FOOD RISK ANALYZER - TEST SUITE")
    print("=" * 80)
    
    analyzer = FoodRiskAnalyzer()
    
    # Test 1: Contaminated spinach for pregnant woman
    print("\n" + "=" * 80)
    print("TEST 1: Spinach with Lead Contamination - Pregnant Woman")
    print("=" * 80)
    
    icpms_data_1 = {
        "Lead": 0.45,  # Exceeds pregnancy limit of 0.05 ppm
        "Iron": 2.7,   # Good iron content
        "Calcium": 99.0
    }
    
    scan_data_1 = {
        "calories": 23,
        "protein_g": 2.9,
        "carbohydrates_g": 3.6,
        "fiber_g": 2.2,
        "nutrients": {
            "iron": 2.7,
            "calcium": 99,
            "vitamin_a": 9380,
            "folate": 194
        }
    }
    
    user_profile_1 = {
        "age": 28,
        "medical_conditions": ["Pregnancy"],
        "health_goals": ["Healthy pregnancy", "Iron intake"]
    }
    
    result_1 = await analyzer.analyze_food_risks(
        food_name="Raw Spinach",
        icpms_data=icpms_data_1,
        scan_data=scan_data_1,
        user_profile=user_profile_1,
        serving_size_g=100
    )
    
    print(f"\n📊 Analysis Results:")
    print(f"  Overall Risk Score: {result_1.overall_risk_score:.1f}/100")
    print(f"  Overall Health Score: {result_1.overall_health_score:.1f}/100")
    print(f"  Safe to Consume: {result_1.is_safe_to_consume}")
    print(f"  Recommendation: {result_1.recommendation}")
    
    print(f"\n⚠️ Critical Warnings ({len(result_1.critical_warnings)}):")
    for warning in result_1.critical_warnings:
        print(f"  {warning}")
    
    print(f"\n🔬 Contaminants Detected ({len(result_1.contaminants)}):")
    for cont in result_1.contaminants:
        if cont.exceeds_limit:
            print(f"  ⛔ {cont.contaminant_name}: {cont.detected_level:.2f} {cont.unit} "
                  f"(Limit: {cont.safe_limit:.2f}, {cont.exceedance_factor:.1f}x over)")
    
    if result_1.safer_alternatives:
        print(f"\n✅ Safer Alternatives:")
        for alt in result_1.safer_alternatives:
            print(f"  • {alt}")
    
    # Test 2: Salmon for heart health
    print("\n" + "=" * 80)
    print("TEST 2: Wild Salmon - Heart Health Goal")
    print("=" * 80)
    
    icpms_data_2 = {
        "Mercury": 0.12,  # Within safe limits
        "Selenium": 46.8  # Good selenium content
    }
    
    scan_data_2 = {
        "calories": 208,
        "protein_g": 20,
        "fat_g": 13,
        "omega3_total": 2260,
        "dha": 1400,
        "epa": 860,
        "nutrients": {
            "vitamin_d": 526,
            "vitamin_b12": 3.2,
            "selenium": 46.8
        }
    }
    
    user_profile_2 = {
        "age": 55,
        "medical_conditions": ["Hypertension"],
        "health_goals": ["Heart health", "Lower cholesterol"]
    }
    
    result_2 = await analyzer.analyze_food_risks(
        food_name="Wild Salmon",
        icpms_data=icpms_data_2,
        scan_data=scan_data_2,
        user_profile=user_profile_2,
        serving_size_g=100
    )
    
    print(f"\n📊 Analysis Results:")
    print(f"  Overall Risk Score: {result_2.overall_risk_score:.1f}/100")
    print(f"  Overall Health Score: {result_2.overall_health_score:.1f}/100")
    print(f"  Recommendation: {result_2.recommendation}")
    
    print(f"\n💪 Goal Alignments:")
    for goal in result_2.goal_alignments:
        print(f"  {goal.goal_name}: {goal.alignment_score:.0f}/100 "
              f"({'✅ Aligned' if goal.is_aligned else '❌ Not aligned'})")
        if goal.supporting_nutrients:
            print(f"    Supporting: {', '.join(goal.supporting_nutrients)}")
    
    print("\n✅ Tests completed!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_food_risk_analyzer())
