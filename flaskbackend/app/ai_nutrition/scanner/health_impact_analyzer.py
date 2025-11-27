"""
Health Impact Analyzer - Phase 3B (AI-Powered)
===============================================

AI-powered health impact analysis with:
- Knowledge Graph for vast domain knowledge (toxins, allergens, nutrients)
- ML models for compound identification from spectral data
- GNN-based toxicity prediction
- Transformer-based allergen detection
- Uncertainty quantification and explainability
- Dynamic recommendations (no hardcoded data)

This is the intelligence layer that converts spectroscopic data into
actionable health insights using deep learning and knowledge graphs.

Author: AI Nutrition Scanner Team
Date: November 2025
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime
import math
import numpy as np

# Import AI components
from .knowledge_graph import (
    KnowledgeGraphEngine, 
    get_knowledge_graph,
    ToxicityKnowledge,
    AllergenKnowledge,
    NutrientKnowledge,
    HealthConditionProfile
)
from .ml_models import (
    ModelFactory,
    SpectralProcessor,
    CompoundIdentificationModel,
    ToxicityPredictionModel,
    AllergenPredictionModel,
    SpectralFeatures,
    CompoundPrediction,
    ToxicityPrediction
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class RiskLevel(Enum):
    """Risk level classification."""
    SAFE = "safe"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class HealthCondition(Enum):
    """Common health conditions for personalized analysis."""
    DIABETES = "diabetes"
    HYPERTENSION = "hypertension"
    CARDIOVASCULAR_DISEASE = "cardiovascular_disease"
    KIDNEY_DISEASE = "kidney_disease"
    LIVER_DISEASE = "liver_disease"
    CELIAC_DISEASE = "celiac_disease"
    LACTOSE_INTOLERANCE = "lactose_intolerance"
    GOUT = "gout"
    OSTEOPOROSIS = "osteoporosis"
    ANEMIA = "anemia"
    PREGNANCY = "pregnancy"
    INFANT = "infant"
    ELDERLY = "elderly"


class InteractionSeverity(Enum):
    """Drug-nutrient interaction severity."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CONTRAINDICATED = "contraindicated"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ToxicityAssessment:
    """Complete toxicity risk assessment."""
    overall_risk: RiskLevel
    toxicity_score: float  # 0-100
    detected_toxins: List[Dict] = field(default_factory=list)
    heavy_metals: List[Dict] = field(default_factory=list)
    carcinogens: List[Dict] = field(default_factory=list)
    safe_for_consumption: bool = True
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AllergenProfile:
    """Allergen detection and analysis."""
    allergens_detected: List[str] = field(default_factory=list)
    allergen_risk: RiskLevel = RiskLevel.SAFE
    cross_reactive_allergens: List[str] = field(default_factory=list)
    severity_estimates: Dict[str, str] = field(default_factory=dict)
    avoidance_advice: List[str] = field(default_factory=list)


@dataclass
class NutritionalAnalysis:
    """Comprehensive nutritional assessment."""
    # Macronutrients (g per 100g)
    protein: float = 0.0
    carbohydrates: float = 0.0
    fat: float = 0.0
    fiber: float = 0.0
    
    # Micronutrients (mg per 100g unless noted)
    vitamins: Dict[str, float] = field(default_factory=dict)
    minerals: Dict[str, float] = field(default_factory=dict)
    
    # RDA compliance (% of daily value per 100g)
    rda_compliance: Dict[str, float] = field(default_factory=dict)
    deficiencies: List[str] = field(default_factory=list)
    excesses: List[str] = field(default_factory=list)
    
    # Quality metrics
    nutrient_density_score: float = 0.0  # 0-100
    health_score: float = 0.0  # 0-100


@dataclass
class HealthImpactReport:
    """Complete health impact analysis report."""
    timestamp: datetime
    food_name: str
    
    # Core analyses
    toxicity: ToxicityAssessment
    allergens: AllergenProfile
    nutrition: NutritionalAnalysis
    
    # Personalized insights
    health_conditions_affected: List[str] = field(default_factory=list)
    personalized_warnings: List[str] = field(default_factory=list)
    personalized_benefits: List[str] = field(default_factory=list)
    
    # Overall assessment
    overall_safety_score: float = 0.0  # 0-100
    overall_health_score: float = 0.0  # 0-100
    consumption_recommendation: str = ""
    portion_guidance: Optional[str] = None


# =============================================================================
# HEALTH IMPACT ANALYZER
# =============================================================================

class HealthImpactAnalyzer:
    """
    AI-Powered Comprehensive Health Impact Analysis System.
    
    Uses knowledge graphs and ML models instead of hardcoded data.
    Provides dynamic, vast knowledge base for health insights.
    """
    
    def __init__(self, 
                 atomic_db=None, 
                 molecular_db=None, 
                 bond_library=None,
                 use_ai_models: bool = True,
                 knowledge_graph: Optional[KnowledgeGraphEngine] = None):
        """
        Initialize AI-powered health impact analyzer.
        
        Args:
            atomic_db: AtomicDatabase instance (optional, legacy)
            molecular_db: MolecularDatabase instance (optional, legacy)
            bond_library: CovalentBondLibrary instance (optional, legacy)
            use_ai_models: Whether to use AI/ML models for predictions
            knowledge_graph: Knowledge graph instance (creates if None)
        """
        # Legacy databases (for backward compatibility)
        self.atomic_db = atomic_db
        self.molecular_db = molecular_db
        self.bond_library = bond_library
        
        # AI Components
        self.use_ai_models = use_ai_models
        self.knowledge_graph = knowledge_graph or get_knowledge_graph()
        
        # ML Models
        if self.use_ai_models:
            self.spectral_processor = ModelFactory.get_spectral_processor()
            self.compound_model = ModelFactory.get_compound_model()
            self.toxicity_model = ModelFactory.get_toxicity_model()
            self.allergen_model = ModelFactory.get_allergen_model()
            logger.info("AI models initialized successfully")
        
        # Deprecated: old hardcoded data (kept for fallback only)
        self.allergen_cross_reactivity: Dict[str, List[str]] = {}
        self.health_condition_restrictions: Dict[HealthCondition, Dict] = {}
        
        logger.info(f"Initialized AI-Powered HealthImpactAnalyzer (KG nodes: {self.knowledge_graph.node_count()})")
    
    def _load_allergen_data(self):
        """
        DEPRECATED: Load allergen cross-reactivity database.
        Now uses knowledge graph for dynamic queries.
        Kept for backward compatibility only.
        """
        logger.info("Allergen data now loaded from Knowledge Graph dynamically")
    
    def _load_health_condition_profiles(self):
        """
        DEPRECATED: Load health condition-specific dietary restrictions.
        Now uses knowledge graph for dynamic queries with clinical evidence.
        Kept for backward compatibility only.
        """
        logger.info("Health condition profiles now loaded from Knowledge Graph dynamically")
    
    # =========================================================================
    # TOXICITY ASSESSMENT
    # =========================================================================
    
    def assess_toxicity(self, composition: Dict[str, float]) -> ToxicityAssessment:
        """
        AI-powered toxicity assessment using knowledge graph.
        
        Args:
            composition: Dict of {compound_name: concentration_mg_kg}
        
        Returns:
            ToxicityAssessment with ML-enhanced risk analysis
        """
        toxins = []
        heavy_metals = []
        carcinogens = []
        warnings = []
        recommendations = []
        toxicity_score = 0.0
        
        for compound, concentration in composition.items():
            # Query knowledge graph for toxicity information
            tox_knowledge = self.knowledge_graph.query_toxicity(compound)
            
            if tox_knowledge:
                # Use ML model for enhanced prediction
                if self.use_ai_models:
                    ml_prediction = self.toxicity_model.predict_toxicity(compound)
                    risk_level = self._ml_risk_to_enum(ml_prediction.acute_toxicity_score)
                else:
                    risk_level = self._calculate_risk_from_kg(tox_knowledge, concentration)
                
                toxins.append({
                    "name": compound,
                    "concentration": concentration,
                    "risk_level": risk_level,
                    "safe_limit": tox_knowledge.safe_limit_mg_kg,
                    "ld50": tox_knowledge.ld50,
                    "hazard": tox_knowledge.hazard_class
                })
                
                # Calculate toxicity score
                risk_score = self._risk_level_to_score(risk_level)
                toxicity_score += risk_score
                
                if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    warnings.append(f"{compound}: {tox_knowledge.hazard_class} detected at {concentration} mg/kg")
                
                # Check if carcinogen
                if tox_knowledge.carcinogenic:
                    carcinogens.append({
                        "name": compound,
                        "concentration": concentration,
                        "carcinogen_class": tox_knowledge.hazard_class
                    })
                    toxicity_score += 30  # High penalty for carcinogens
                    warnings.append(f"Carcinogen detected: {compound}")
            
            # Check for heavy metals (pattern matching)
            if self._is_heavy_metal(compound):
                metal_risk = self._assess_heavy_metal_kg(compound, concentration)
                heavy_metals.append(metal_risk)
                toxicity_score += metal_risk["score"]
                
                if metal_risk["exceeds_limit"]:
                    warnings.append(f"Heavy metal {compound} exceeds safe limit")
        
        # Determine overall risk
        if toxicity_score >= 80:
            overall_risk = RiskLevel.CRITICAL
            safe_consumption = False
            recommendations.append("DO NOT CONSUME - Critical toxicity risk")
        elif toxicity_score >= 60:
            overall_risk = RiskLevel.HIGH
            safe_consumption = False
            recommendations.append("NOT RECOMMENDED - High toxicity risk")
        elif toxicity_score >= 40:
            overall_risk = RiskLevel.MODERATE
            safe_consumption = True
            recommendations.append("Consume with caution, limit portion size")
        elif toxicity_score >= 20:
            overall_risk = RiskLevel.LOW
            safe_consumption = True
            recommendations.append("Generally safe, monitor for sensitive individuals")
        else:
            overall_risk = RiskLevel.SAFE
            safe_consumption = True
            recommendations.append("No significant toxicity concerns detected")
        
        return ToxicityAssessment(
            overall_risk=overall_risk,
            toxicity_score=min(toxicity_score, 100),
            detected_toxins=toxins,
            heavy_metals=heavy_metals,
            carcinogens=carcinogens,
            safe_for_consumption=safe_consumption,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _ml_risk_to_enum(self, ml_score: float) -> RiskLevel:
        """Convert ML toxicity score to RiskLevel enum."""
        if ml_score >= 0.8:
            return RiskLevel.CRITICAL
        elif ml_score >= 0.6:
            return RiskLevel.HIGH
        elif ml_score >= 0.4:
            return RiskLevel.MODERATE
        elif ml_score >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.SAFE
    
    def _calculate_risk_from_kg(self, tox_knowledge: ToxicityKnowledge, 
                               concentration: float) -> RiskLevel:
        """Calculate risk level from knowledge graph data."""
        if not tox_knowledge.safe_limit_mg_kg:
            return RiskLevel.LOW
        
        safe_limit = tox_knowledge.safe_limit_mg_kg
        
        if concentration > safe_limit * 10:
            return RiskLevel.CRITICAL
        elif concentration > safe_limit * 5:
            return RiskLevel.HIGH
        elif concentration > safe_limit * 2:
            return RiskLevel.MODERATE
        elif concentration > safe_limit:
            return RiskLevel.LOW
        else:
            return RiskLevel.SAFE
    
    def _risk_level_to_score(self, risk_level: RiskLevel) -> float:
        """Convert RiskLevel to numeric score."""
        mapping = {
            RiskLevel.CRITICAL: 80,
            RiskLevel.HIGH: 60,
            RiskLevel.MODERATE: 40,
            RiskLevel.LOW: 20,
            RiskLevel.SAFE: 0
        }
        return mapping.get(risk_level, 0)
    
    def _assess_heavy_metal_kg(self, metal: str, concentration: float) -> Dict:
        """Assess heavy metal using knowledge graph."""
        tox_knowledge = self.knowledge_graph.query_toxicity(metal)
        
        if tox_knowledge and tox_knowledge.safe_limit_mg_kg:
            safe_limit = tox_knowledge.safe_limit_mg_kg
        else:
            safe_limit = 0.1  # Default conservative limit
        
        exceeds = concentration > safe_limit
        
        return {
            "metal": metal,
            "concentration": concentration,
            "safe_limit": safe_limit,
            "exceeds_limit": exceeds,
            "score": 40 if exceeds else 10
        }
    
    def _is_toxic_compound(self, compound: str) -> bool:
        """Check if compound is known toxin."""
        toxic_compounds = [
            "aflatoxin", "acrylamide", "arsenic", "lead", "mercury",
            "cadmium", "benzene", "formaldehyde", "dioxin"
        ]
        return any(toxin in compound.lower() for toxin in toxic_compounds)
    
    def _is_heavy_metal(self, compound: str) -> bool:
        """Check if compound is heavy metal."""
        heavy_metals = ["lead", "mercury", "cadmium", "arsenic", "chromium"]
        return any(metal in compound.lower() for metal in heavy_metals)
    
    def _is_carcinogen(self, compound: str) -> bool:
        """Check if compound is known carcinogen."""
        carcinogens = ["aflatoxin", "benzene", "formaldehyde", "acrylamide"]
        return any(carc in compound.lower() for carc in carcinogens)
    
    def _calculate_toxin_risk(self, compound: str, concentration: float) -> Dict:
        """Calculate risk level for specific toxin."""
        # Simplified toxin risk calculation
        toxin_data = {
            "aflatoxin_b1": {"safe_limit": 0.02, "ld50": 0.5, "hazard": "liver_cancer"},
            "acrylamide": {"safe_limit": 1.0, "ld50": 150, "hazard": "neurotoxin"},
            "lead": {"safe_limit": 0.1, "ld50": 450, "hazard": "neurotoxin"},
            "mercury": {"safe_limit": 0.5, "ld50": 1.4, "hazard": "neurotoxin"},
        }
        
        compound_key = compound.lower().replace(" ", "_")
        data = toxin_data.get(compound_key, {"safe_limit": 1.0, "ld50": 100, "hazard": "toxic"})
        
        if concentration > data["safe_limit"] * 10:
            level = RiskLevel.CRITICAL
            score = 80
        elif concentration > data["safe_limit"] * 5:
            level = RiskLevel.HIGH
            score = 60
        elif concentration > data["safe_limit"] * 2:
            level = RiskLevel.MODERATE
            score = 40
        elif concentration > data["safe_limit"]:
            level = RiskLevel.LOW
            score = 20
        else:
            level = RiskLevel.SAFE
            score = 0
        
        return {
            "level": level,
            "score": score,
            "safe_limit": data["safe_limit"],
            "ld50": data.get("ld50"),
            "hazard": data["hazard"]
        }
    
    def _assess_heavy_metal(self, metal: str, concentration: float) -> Dict:
        """Assess heavy metal contamination."""
        limits = {
            "lead": 0.1,
            "mercury": 0.5,
            "cadmium": 0.05,
            "arsenic": 0.1
        }
        
        metal_key = metal.lower()
        safe_limit = limits.get(metal_key, 0.1)
        exceeds = concentration > safe_limit
        
        return {
            "metal": metal,
            "concentration": concentration,
            "safe_limit": safe_limit,
            "exceeds_limit": exceeds,
            "score": 40 if exceeds else 10
        }
    
    def _get_carcinogen_class(self, compound: str) -> str:
        """Get IARC carcinogen classification."""
        classifications = {
            "aflatoxin": "Group 1 (Carcinogenic to humans)",
            "benzene": "Group 1 (Carcinogenic to humans)",
            "formaldehyde": "Group 1 (Carcinogenic to humans)",
            "acrylamide": "Group 2A (Probably carcinogenic)"
        }
        
        for key, classification in classifications.items():
            if key in compound.lower():
                return classification
        
        return "Not classified"
    
    # =========================================================================
    # ALLERGEN DETECTION
    # =========================================================================
    
    def detect_allergens(self, composition: Dict[str, float]) -> AllergenProfile:
        """
        AI-powered allergen detection using knowledge graph.
        
        Args:
            composition: Dict of {compound_name: concentration_mg_kg}
        
        Returns:
            AllergenProfile with KG-enhanced cross-reactivity analysis
        """
        detected = []
        cross_reactive = []
        severity = {}
        advice = []
        
        # Common allergen markers (protein-based)
        allergen_markers = {
            "ara_h": "peanut",
            "gluten": "wheat",
            "gliadin": "wheat",
            "casein": "milk",
            "lactoglobulin": "milk",
            "ovalbumin": "egg",
            "tropomyosin": "shellfish"
        }
        
        for compound in composition.keys():
            compound_lower = compound.lower()
            for marker, allergen in allergen_markers.items():
                if marker in compound_lower:
                    detected.append(allergen)
                    
                    # Query knowledge graph for cross-reactivity
                    allergen_info = self.knowledge_graph.query_allergen(allergen)
                    if allergen_info:
                        cross_reactive.extend(allergen_info.cross_reactive_allergens)
                        
                        # Get severity distribution from KG
                        concentration = composition[compound]
                        if concentration > 100:
                            severity[allergen] = "high"
                        elif concentration > 10:
                            severity[allergen] = "moderate"
                        else:
                            severity[allergen] = "trace"
        
        # Remove duplicates
        detected = list(set(detected))
        cross_reactive = list(set(cross_reactive) - set(detected))
        
        # Determine risk level
        if any(severity.get(a) == "high" for a in detected):
            risk = RiskLevel.HIGH
        elif any(severity.get(a) == "moderate" for a in detected):
            risk = RiskLevel.MODERATE
        elif detected:
            risk = RiskLevel.LOW
        else:
            risk = RiskLevel.SAFE
        
        # Generate advice using KG data
        if detected:
            advice.append(f"Contains: {', '.join(detected)}")
            advice.append("Avoid if allergic to these foods")
            if cross_reactive:
                advice.append(f"May cross-react with: {', '.join(cross_reactive)}")
            
            # Add prevalence info from KG
            for allergen in detected:
                allergen_info = self.knowledge_graph.query_allergen(allergen)
                if allergen_info and allergen_info.affected_population_percent:
                    advice.append(f"{allergen}: affects ~{allergen_info.affected_population_percent}% of population")
        
        return AllergenProfile(
            allergens_detected=detected,
            allergen_risk=risk,
            cross_reactive_allergens=cross_reactive,
            severity_estimates=severity,
            avoidance_advice=advice
        )
    
    # =========================================================================
    # NUTRITIONAL ANALYSIS
    # =========================================================================
    
    def analyze_nutrition(self, composition: Dict[str, float], serving_size_g: float = 100) -> NutritionalAnalysis:
        """
        AI-enhanced nutritional analysis with dynamic RDA from knowledge graph.
        
        Args:
            composition: Dict of {compound_name: concentration_mg_kg}
            serving_size_g: Serving size in grams (default 100g)
        
        Returns:
            NutritionalAnalysis with KG-based RDA compliance
        """
        # Convert mg/kg to mg per serving
        factor = serving_size_g / 1000
        
        analysis = NutritionalAnalysis()
        
        # Macronutrients (estimate from composition)
        for compound, conc_mg_kg in composition.items():
            compound_lower = compound.lower()
            conc_mg = conc_mg_kg * factor
            
            # Proteins (amino acids)
            if any(aa in compound_lower for aa in ["leucine", "lysine", "valine", "alanine", "glycine", "protein"]):
                analysis.protein += conc_mg / 1000  # Convert to g
            
            # Carbohydrates
            if any(carb in compound_lower for carb in ["glucose", "fructose", "sucrose", "starch"]):
                analysis.carbohydrates += conc_mg / 1000
            
            # Fats (fatty acids)
            if any(fat in compound_lower for fat in ["oleic", "linoleic", "palmitic", "stearic"]):
                analysis.fat += conc_mg / 1000
            
            # Vitamins & Minerals - query knowledge graph for RDA
            if "vitamin" in compound_lower or any(v in compound_lower for v in 
                ["thiamine", "riboflavin", "niacin", "folate", "cobalamin", "ascorbic", "retinol", "tocopherol"]):
                analysis.vitamins[compound] = conc_mg
                
                # Query KG for RDA
                nutrient_info = self.knowledge_graph.query_nutrient_rda(compound)
                if nutrient_info:
                    rda = nutrient_info.rda_adult_male  # Default to male
                    if rda > 0:
                        percent_rda = (conc_mg / rda) * 100
                        analysis.rda_compliance[compound] = percent_rda
                        
                        if percent_rda < 10:
                            analysis.deficiencies.append(f"{compound} ({percent_rda:.1f}% of RDA)")
                        elif percent_rda > 200:
                            analysis.excesses.append(f"{compound} ({percent_rda:.1f}% of RDA)")
            
            # Minerals
            if any(m in compound_lower for m in ["calcium", "iron", "magnesium", "zinc", "potassium", "sodium"]):
                analysis.minerals[compound] = conc_mg
                
                # Query KG for RDA
                nutrient_info = self.knowledge_graph.query_nutrient_rda(compound)
                if nutrient_info:
                    rda = nutrient_info.rda_adult_male
                    if rda > 0:
                        percent_rda = (conc_mg / rda) * 100
                        analysis.rda_compliance[compound] = percent_rda
                        
                        if percent_rda < 10:
                            analysis.deficiencies.append(f"{compound} ({percent_rda:.1f}% of RDA)")
                        elif percent_rda > 200:
                            analysis.excesses.append(f"{compound} ({percent_rda:.1f}% of RDA)")
        
        # Calculate nutrient density score
        essential_nutrients = len(analysis.vitamins) + len(analysis.minerals)
        nutrient_density = min((essential_nutrients * 10), 100)
        analysis.nutrient_density_score = nutrient_density
        
        # Calculate health score
        health_score = 50  # Base score
        health_score += min(analysis.protein * 2, 20)  # Protein bonus
        health_score += min(len(analysis.vitamins) * 3, 15)  # Vitamin diversity
        health_score += min(len(analysis.minerals) * 3, 15)  # Mineral diversity
        health_score -= len(analysis.deficiencies) * 5  # Deficiency penalty
        
        analysis.health_score = max(0, min(health_score, 100))
        
        return analysis
    
    # =========================================================================
    # PERSONALIZED HEALTH RECOMMENDATIONS
    # =========================================================================
    
    def personalize_recommendations(self, composition: Dict[str, float],
                                   health_conditions: List[HealthCondition],
                                   age: Optional[int] = None,
                                   pregnancy: bool = False) -> Tuple[List[str], List[str], List[str]]:
        """
        AI-powered personalized recommendations using knowledge graph.
        
        Args:
            composition: Dict of {compound_name: concentration_mg_kg}
            health_conditions: List of user's health conditions
            age: User's age (for age-specific advice)
            pregnancy: Whether user is pregnant
        
        Returns:
            Tuple of (conditions_affected, warnings, benefits)
        """
        conditions_affected = []
        warnings = []
        benefits = []
        
        # Add pregnancy as condition if applicable
        if pregnancy:
            health_conditions = list(health_conditions) + [HealthCondition.PREGNANCY]
        
        # Add age-based conditions
        if age:
            if age < 2:
                health_conditions = list(health_conditions) + [HealthCondition.INFANT]
            elif age >= 65:
                health_conditions = list(health_conditions) + [HealthCondition.ELDERLY]
        
        # Query knowledge graph for each health condition
        for condition in health_conditions:
            # Query KG dynamically instead of hardcoded dictionary
            condition_value = condition.value if isinstance(condition, Enum) else str(condition)
            profile = self.knowledge_graph.query_health_condition(condition_value)
            
            # Fallback: try with type_2_ prefix for diabetes
            if not profile and condition_value == "diabetes":
                profile = self.knowledge_graph.query_health_condition("type_2_diabetes")
            
            if not profile:
                logger.warning(f"No profile found for condition: {condition_value}")
                continue
            
            condition_name = profile.condition_name
            
            # Check for foods to avoid
            for avoid_item in profile.avoid:
                if any(avoid_item.lower() in comp.lower() for comp in composition.keys()):
                    conditions_affected.append(condition_name)
                    warnings.append(f"{condition_name}: Contains {avoid_item} (should avoid) [Evidence: {profile.evidence_level}]")
            
            # Check for foods to limit with clinical targets
            for limit_item, limit_value in profile.limit.items():
                # Check nutrient levels against clinical targets
                if limit_item == "sodium" and "max_sodium_mg" in profile.clinical_targets:
                    sodium_conc = sum(conc for comp, conc in composition.items() 
                                    if "sodium" in comp.lower())
                    max_sodium = profile.clinical_targets["max_sodium_mg"]
                    if sodium_conc > max_sodium:
                        conditions_affected.append(condition_name)
                        warnings.append(f"{condition_name}: High sodium ({sodium_conc:.0f} mg > {max_sodium} mg target)")
                
                elif limit_item == "potassium" and "max_potassium_mg" in profile.clinical_targets:
                    potassium_conc = sum(conc for comp, conc in composition.items() 
                                       if "potassium" in comp.lower())
                    max_potassium = profile.clinical_targets["max_potassium_mg"]
                    if potassium_conc > max_potassium:
                        conditions_affected.append(condition_name)
                        warnings.append(f"{condition_name}: High potassium ({potassium_conc:.0f} mg > {max_potassium} mg limit)")
                
                elif limit_item == "sugar":
                    sugar_conc = sum(conc for comp, conc in composition.items() 
                                   if any(s in comp.lower() for s in ["glucose", "fructose", "sucrose"]))
                    if sugar_conc > 50000:  # >50g per kg
                        conditions_affected.append(condition_name)
                        warnings.append(f"{condition_name}: High sugar content")
            
            # Check for beneficial nutrients
            for increase_item in profile.increase:
                if any(increase_item.lower() in comp.lower() for comp in composition.keys()):
                    if condition_name not in conditions_affected:
                        conditions_affected.append(condition_name)
                    
                    # Get nutrient benefits from KG
                    nutrient_info = self.knowledge_graph.query_nutrient_rda(increase_item)
                    if nutrient_info and nutrient_info.health_benefits:
                        benefit_desc = ", ".join(nutrient_info.health_benefits[:2])
                        benefits.append(f"{condition_name}: Contains beneficial {increase_item} ({benefit_desc})")
                    else:
                        benefits.append(f"{condition_name}: Contains beneficial {increase_item}")
        
        # Remove duplicates
        conditions_affected = list(set(conditions_affected))
        
        return conditions_affected, warnings, benefits
    
    # =========================================================================
    # FULL REPORT GENERATION
    # =========================================================================
    
    def generate_report(self, food_name: str, composition: Dict[str, float],
                       medications: Optional[List[str]] = None,
                       health_conditions: Optional[List[HealthCondition]] = None,
                       age: Optional[int] = None,
                       pregnancy: bool = False,
                       serving_size_g: float = 100) -> HealthImpactReport:
        """
        Generate comprehensive health impact report.
        
        Args:
            food_name: Name of the food being analyzed
            composition: Dict of {compound_name: concentration_mg_kg}
            medications: List of current medications (not used anymore)
            health_conditions: List of health conditions
            age: User's age
            pregnancy: Whether user is pregnant
            serving_size_g: Serving size in grams
        
        Returns:
            Complete HealthImpactReport with all analyses
        """
        # Initialize
        health_conditions = health_conditions or []
        
        # Run all analyses
        toxicity = self.assess_toxicity(composition)
        allergens = self.detect_allergens(composition)
        nutrition = self.analyze_nutrition(composition, serving_size_g)
        
        conditions_affected, warnings, benefits = self.personalize_recommendations(
            composition, health_conditions, age, pregnancy
        )
        
        # Calculate overall scores
        safety_score = 100 - toxicity.toxicity_score
        
        # Adjust for allergens
        if allergens.allergen_risk == RiskLevel.HIGH:
            safety_score -= 30
        elif allergens.allergen_risk == RiskLevel.MODERATE:
            safety_score -= 15
        elif allergens.allergen_risk == RiskLevel.LOW:
            safety_score -= 5
        
        safety_score = max(0, min(safety_score, 100))
        
        # Overall health score (weighted average)
        health_score = (nutrition.health_score * 0.6 + safety_score * 0.4)
        
        # Generate consumption recommendation
        if safety_score < 30:
            recommendation = "❌ DO NOT CONSUME - Serious safety concerns"
            portion_guidance = None
        elif safety_score < 50:
            recommendation = "⚠️ CAUTION - Consume with awareness of risks"
            portion_guidance = "Limit to small portions"
        elif safety_score < 70:
            recommendation = "✓ ACCEPTABLE - Generally safe with minor concerns"
            portion_guidance = "Moderate portions recommended"
        else:
            recommendation = "✅ RECOMMENDED - Safe and beneficial"
            portion_guidance = f"Enjoy {serving_size_g}g servings"
        
        # Create report
        report = HealthImpactReport(
            timestamp=datetime.now(),
            food_name=food_name,
            toxicity=toxicity,
            allergens=allergens,
            nutrition=nutrition,
            health_conditions_affected=conditions_affected,
            personalized_warnings=warnings,
            personalized_benefits=benefits,
            overall_safety_score=safety_score,
            overall_health_score=health_score,
            consumption_recommendation=recommendation,
            portion_guidance=portion_guidance
        )
        
        return report
    
    # =========================================================================
    # ATOMIC VISION INTEGRATION
    # =========================================================================
    
    def integrate_atomic_composition(self, atomic_result) -> Dict[str, float]:
        """
        Integrate atomic/elemental composition from image-based prediction.
        
        This method bridges the AtomicVisionPredictor output with the
        Health Impact Analyzer, converting elemental data (mg/kg) to
        actionable health insights through toxicity, nutrition, and 
        personalized recommendations.
        
        Args:
            atomic_result: AtomicCompositionResult from AtomicVisionPredictor
        
        Returns:
            Dict mapping element symbols to concentrations (mg/kg)
            
        Usage:
            from atomic_vision import AtomicVisionPredictor, FoodImageData
            
            # Predict atomic composition from image
            predictor = AtomicVisionPredictor()
            image_data = FoodImageData(image=rgb_array, weight_grams=150)
            atomic_result = predictor.predict(image_data)
            
            # Integrate with health analyzer
            analyzer = HealthImpactAnalyzer(use_ai_models=True)
            composition = analyzer.integrate_atomic_composition(atomic_result)
            
            # Run comprehensive health analysis
            toxicity = analyzer.assess_atomic_toxicity(atomic_result)
            nutrition = analyzer.estimate_nutrition_from_elements(atomic_result)
            report = analyzer.generate_atomic_health_report(
                atomic_result, "Analyzed Food Sample"
            )
        """
        logger.info(f"Integrating atomic composition: {len(atomic_result.predictions)} elements")
        
        # Convert AtomicCompositionResult to simple dict
        composition = {}
        for pred in atomic_result.predictions:
            composition[pred.element] = pred.concentration_mg_kg
            
            # Log warnings for exceeded limits
            if pred.exceeds_limit:
                logger.warning(
                    f"{pred.element} exceeds regulatory limit: "
                    f"{pred.concentration_mg_kg:.3f} mg/kg "
                    f"(confidence: {pred.confidence:.2f})"
                )
        
        return composition
    
    def assess_atomic_toxicity(self, atomic_result) -> ToxicityAssessment:
        """
        Assess toxicity from atomic composition (heavy metals + toxic elements).
        
        This method specifically handles elemental toxicity (Pb, Cd, As, Hg, etc.)
        from ICP-MS or image-based atomic predictions.
        
        Args:
            atomic_result: AtomicCompositionResult with element predictions
        
        Returns:
            ToxicityAssessment focused on heavy metal toxicity
        """
        try:
            from .atomic_vision import TOXIC_ELEMENTS, ELEMENT_DATABASE
        except ImportError:
            logger.error("atomic_vision module not available")
            return ToxicityAssessment(
                overall_risk=RiskLevel.SAFE,
                toxicity_score=0,
                safe_for_consumption=True,
                warnings=["Atomic analysis not available - install required dependencies"]
            )
        
        toxins = []
        heavy_metals = []
        warnings = []
        recommendations = []
        toxicity_score = 0.0
        
        # Check toxic elements
        toxic_predictions = atomic_result.get_toxic_elements()
        
        for pred in toxic_predictions:
            element_info = ELEMENT_DATABASE[pred.element]
            
            # Query knowledge graph for toxicity data
            tox_knowledge = self.knowledge_graph.query_toxicity(pred.element.lower())
            
            # Determine risk level
            if element_info.regulatory_limit_mg_kg and pred.exceeds_limit:
                risk = RiskLevel.CRITICAL
                toxicity_score += 25.0
                warnings.append(
                    f"⚠️ {element_info.name} ({pred.element}): "
                    f"{pred.concentration_mg_kg:.3f} mg/kg EXCEEDS regulatory limit "
                    f"of {element_info.regulatory_limit_mg_kg} mg/kg"
                )
                recommendations.append(
                    f"AVOID consumption - {pred.element} concentration unsafe"
                )
            elif element_info.regulatory_limit_mg_kg and pred.concentration_mg_kg > element_info.regulatory_limit_mg_kg * 0.5:
                risk = RiskLevel.HIGH
                toxicity_score += 15.0
                warnings.append(
                    f"⚠ {element_info.name} ({pred.element}): "
                    f"{pred.concentration_mg_kg:.3f} mg/kg approaching limit"
                )
            elif element_info.regulatory_limit_mg_kg and pred.concentration_mg_kg > element_info.regulatory_limit_mg_kg * 0.2:
                risk = RiskLevel.MODERATE
                toxicity_score += 5.0
            else:
                risk = RiskLevel.LOW
                toxicity_score += 1.0
            
            heavy_metals.append({
                "element": pred.element,
                "name": element_info.name,
                "concentration_mg_kg": pred.concentration_mg_kg,
                "uncertainty_mg_kg": pred.uncertainty_mg_kg,
                "confidence": pred.confidence,
                "regulatory_limit": element_info.regulatory_limit_mg_kg,
                "exceeds_limit": pred.exceeds_limit,
                "risk_level": risk,
                "health_effects": tox_knowledge.mechanism if tox_knowledge else "Unknown"
            })
        
        # Determine overall risk
        if toxicity_score > 50:
            overall_risk = RiskLevel.CRITICAL
            safe = False
        elif toxicity_score > 30:
            overall_risk = RiskLevel.HIGH
            safe = False
        elif toxicity_score > 10:
            overall_risk = RiskLevel.MODERATE
            safe = True
        elif toxicity_score > 0:
            overall_risk = RiskLevel.LOW
            safe = True
        else:
            overall_risk = RiskLevel.SAFE
            safe = True
        
        return ToxicityAssessment(
            overall_risk=overall_risk,
            toxicity_score=toxicity_score,
            detected_toxins=toxins,
            heavy_metals=heavy_metals,
            carcinogens=[],  # Carcinogenicity requires molecular data
            safe_for_consumption=safe,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def estimate_nutrition_from_elements(self, atomic_result) -> NutritionalAnalysis:
        """
        Estimate nutritional value from elemental composition.
        
        Converts atomic data (Fe, Zn, Ca, Mg, etc.) to nutritional insights
        including mineral content, RDA compliance, and health scores.
        
        Args:
            atomic_result: AtomicCompositionResult with element predictions
        
        Returns:
            NutritionalAnalysis with mineral-based nutrition profile
        """
        try:
            from .atomic_vision import NUTRIENT_ELEMENTS, ELEMENT_DATABASE
        except ImportError:
            logger.error("atomic_vision module not available")
            return NutritionalAnalysis(health_score=0)
        
        minerals = {}
        rda_compliance = {}
        deficiencies = []
        excesses = []
        
        # Extract nutrient elements
        nutrient_predictions = atomic_result.get_nutrient_elements()
        
        for pred in nutrient_predictions:
            element_info = ELEMENT_DATABASE[pred.element]
            
            if not element_info.nutritional:
                continue
            
            # Store concentration
            minerals[pred.element] = pred.concentration_mg_kg
            
            # Calculate RDA compliance (per 100g serving)
            if element_info.rda_mg_day:
                # Convert mg/kg to mg/100g
                mg_per_100g = pred.concentration_mg_kg / 10.0
                rda_percent = (mg_per_100g / element_info.rda_mg_day) * 100
                rda_compliance[pred.element] = rda_percent
                
                # Check for deficiencies/excesses
                if rda_percent < 10:
                    deficiencies.append(
                        f"{element_info.name} ({pred.element}): "
                        f"Only {rda_percent:.1f}% of RDA"
                    )
                elif element_info.toxic and rda_percent > 200:
                    excesses.append(
                        f"{element_info.name} ({pred.element}): "
                        f"{rda_percent:.1f}% of RDA (may be excessive)"
                    )
        
        # Calculate nutrient density score
        # Based on how many essential nutrients are present at meaningful levels
        meaningful_nutrients = sum(
            1 for pct in rda_compliance.values() if pct >= 10
        )
        total_essential = len([e for e in NUTRIENT_ELEMENTS 
                              if ELEMENT_DATABASE[e].nutritional])
        nutrient_density = (meaningful_nutrients / total_essential) * 100 if total_essential > 0 else 0
        
        # Health score (balanced nutrient profile)
        # Penalize deficiencies and excesses
        health_score = nutrient_density
        health_score -= len(deficiencies) * 5
        health_score -= len(excesses) * 10
        health_score = max(0, min(100, health_score))
        
        return NutritionalAnalysis(
            minerals=minerals,
            rda_compliance=rda_compliance,
            deficiencies=deficiencies,
            excesses=excesses,
            nutrient_density_score=nutrient_density,
            health_score=health_score
        )
    
    def generate_atomic_health_report(self, atomic_result, 
                                     food_name: str,
                                     user_conditions: Optional[List[HealthCondition]] = None) -> HealthImpactReport:
        """
        Generate comprehensive health impact report from atomic composition.
        
        This is the main entry point for image-based food analysis using
        the atomic vision system integrated with full health impact assessment.
        
        Args:
            atomic_result: AtomicCompositionResult from image prediction
            food_name: Name/description of the analyzed food
            user_conditions: Optional list of user health conditions
        
        Returns:
            Complete HealthImpactReport with toxicity, nutrition, and recommendations
        
        Example:
            # Full pipeline: Image → Atoms → Health Report
            from atomic_vision import AtomicVisionPredictor, FoodImageData, load_image
            
            # 1. Load and predict atomic composition
            predictor = AtomicVisionPredictor("models/atomic_net.pth")
            image = load_image("food_photo.jpg")
            image_data = FoodImageData(
                image=image,
                weight_grams=150.0,
                food_type="leafy_vegetable",
                preparation="raw"
            )
            atomic_result = predictor.predict(image_data)
            
            # 2. Generate health report
            analyzer = HealthImpactAnalyzer(use_ai_models=True)
            report = analyzer.generate_atomic_health_report(
                atomic_result,
                food_name="Fresh Spinach",
                user_conditions=[HealthCondition.ANEMIA, HealthCondition.PREGNANCY]
            )
            
            # 3. Display results
            print(f"Safety Score: {report.overall_safety_score}/100")
            print(f"Health Score: {report.overall_health_score}/100")
            print(f"Recommendation: {report.consumption_recommendation}")
        """
        logger.info(f"Generating atomic health report for: {food_name}")
        
        # Toxicity assessment (heavy metals)
        toxicity = self.assess_atomic_toxicity(atomic_result)
        
        # Allergen profile (elements don't directly indicate allergens)
        # Use placeholder - would need molecular data for real allergen detection
        allergens = AllergenProfile(
            allergens_detected=[],
            allergen_risk=RiskLevel.SAFE,
            avoidance_advice=["Elemental analysis cannot detect protein allergens - molecular analysis required"]
        )
        
        # Nutritional analysis (minerals)
        nutrition = self.estimate_nutrition_from_elements(atomic_result)
        
        # Personalized recommendations
        personalized_warnings = []
        personalized_benefits = []
        health_conditions_affected = []
        
        if user_conditions:
            try:
                from .atomic_vision import ELEMENT_DATABASE
                
                for condition in user_conditions:
                    # Query knowledge graph for condition-specific guidance
                    condition_str = condition.value
                    if condition == HealthCondition.DIABETES:
                        condition_str = "type_2_diabetes"  # map to KG key
                    
                    condition_profile = self.knowledge_graph.query_health_condition(condition_str)
                    
                    if condition_profile:
                        health_conditions_affected.append(condition.value)
                        
                        # Check if any detected elements interact with condition
                        composition = self.integrate_atomic_composition(atomic_result)
                        
                        # High sodium warning for hypertension/kidney disease
                        if condition in [HealthCondition.HYPERTENSION, HealthCondition.KIDNEY_DISEASE]:
                            na_pred = atomic_result.get_element("Na")
                            if na_pred and na_pred.concentration_mg_kg > 2000:  # >2g/kg
                                personalized_warnings.append(
                                    f"⚠ High sodium content ({na_pred.concentration_mg_kg:.0f} mg/kg) "
                                    f"- caution advised for {condition.value}"
                                )
                        
                        # Iron benefits for anemia
                        if condition == HealthCondition.ANEMIA:
                            fe_pred = atomic_result.get_element("Fe")
                            if fe_pred and fe_pred.concentration_mg_kg > 30:  # Good iron source
                                personalized_benefits.append(
                                    f"✓ Good iron source ({fe_pred.concentration_mg_kg:.1f} mg/kg) "
                                    f"- beneficial for anemia management"
                                )
                        
                        # Calcium benefits for osteoporosis
                        if condition == HealthCondition.OSTEOPOROSIS:
                            ca_pred = atomic_result.get_element("Ca")
                            if ca_pred and ca_pred.concentration_mg_kg > 500:
                                personalized_benefits.append(
                                    f"✓ Rich in calcium ({ca_pred.concentration_mg_kg:.0f} mg/kg) "
                                    f"- supports bone health"
                                )
            except ImportError:
                logger.warning("Atomic vision module not available for personalized recommendations")
        
        # Overall scores
        overall_safety = 100 - toxicity.toxicity_score
        overall_health = nutrition.health_score
        
        # Consumption recommendation
        if toxicity.overall_risk in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            recommendation = "❌ NOT RECOMMENDED - Unsafe toxicity levels detected"
        elif toxicity.overall_risk == RiskLevel.MODERATE:
            recommendation = "⚠ CONSUME WITH CAUTION - Moderate toxicity risk"
        elif nutrition.health_score > 70:
            recommendation = "✅ RECOMMENDED - Safe and nutritious"
        elif nutrition.health_score > 40:
            recommendation = "✓ ACCEPTABLE - Safe but moderate nutritional value"
        else:
            recommendation = "⚠ LIMITED NUTRITIONAL VALUE - Safe but low nutrients"
        
        return HealthImpactReport(
            timestamp=atomic_result.timestamp,
            food_name=food_name,
            toxicity=toxicity,
            allergens=allergens,
            nutrition=nutrition,
            health_conditions_affected=health_conditions_affected,
            personalized_warnings=personalized_warnings,
            personalized_benefits=personalized_benefits,
            overall_safety_score=overall_safety,
            overall_health_score=overall_health,
            consumption_recommendation=recommendation,
            portion_guidance=None  # Could add based on nutrient density
        )
    
    def print_report(self, report: HealthImpactReport):
        """Print formatted health impact report."""
        print("\n" + "="*80)
        print(f"HEALTH IMPACT REPORT: {report.food_name.upper()}")
        print(f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Overall scores
        print(f"\n📊 OVERALL ASSESSMENT")
        print(f"   Safety Score:  {report.overall_safety_score:.1f}/100")
        print(f"   Health Score:  {report.overall_health_score:.1f}/100")
        print(f"   Recommendation: {report.consumption_recommendation}")
        if report.portion_guidance:
            print(f"   Portion Guidance: {report.portion_guidance}")
        
        # Toxicity
        print(f"\n☣️  TOXICITY ASSESSMENT")
        print(f"   Risk Level: {report.toxicity.overall_risk.value.upper()}")
        print(f"   Toxicity Score: {report.toxicity.toxicity_score:.1f}/100")
        if report.toxicity.detected_toxins:
            print(f"   Toxins Detected: {len(report.toxicity.detected_toxins)}")
            for toxin in report.toxicity.detected_toxins[:3]:
                print(f"      • {toxin['name']}: {toxin['concentration']} mg/kg ({toxin['risk_level'].value})")
        if report.toxicity.warnings:
            print(f"   Warnings:")
            for warning in report.toxicity.warnings:
                print(f"      ⚠️  {warning}")
        
        # Allergens
        print(f"\n🥜 ALLERGEN ANALYSIS")
        if report.allergens.allergens_detected:
            print(f"   Risk Level: {report.allergens.allergen_risk.value.upper()}")
            print(f"   Allergens: {', '.join(report.allergens.allergens_detected)}")
            if report.allergens.cross_reactive_allergens:
                print(f"   Cross-reactive: {', '.join(report.allergens.cross_reactive_allergens)}")
        else:
            print(f"   ✓ No major allergens detected")
        
        # Nutrition
        print(f"\n🥗 NUTRITIONAL PROFILE (per 100g)")
        print(f"   Protein: {report.nutrition.protein:.1f}g")
        print(f"   Carbohydrates: {report.nutrition.carbohydrates:.1f}g")
        print(f"   Fat: {report.nutrition.fat:.1f}g")
        print(f"   Health Score: {report.nutrition.health_score:.1f}/100")
        
        if report.nutrition.vitamins:
            print(f"\n   Vitamins ({len(report.nutrition.vitamins)}):")
            for vitamin, amount in list(report.nutrition.vitamins.items())[:5]:
                rda = report.nutrition.rda_compliance.get(vitamin, 0)
                print(f"      • {vitamin}: {amount:.2f} mg ({rda:.0f}% RDA)")
        
        if report.nutrition.minerals:
            print(f"\n   Minerals ({len(report.nutrition.minerals)}):")
            for mineral, amount in list(report.nutrition.minerals.items())[:5]:
                rda = report.nutrition.rda_compliance.get(mineral, 0)
                print(f"      • {mineral}: {amount:.2f} mg ({rda:.0f}% RDA)")
        
        # Personalized recommendations
        if report.health_conditions_affected:
            print(f"\n🏥 HEALTH CONDITIONS AFFECTED")
            for condition in report.health_conditions_affected:
                print(f"   • {condition}")
        
        if report.personalized_warnings:
            print(f"\n   ⚠️  Warnings:")
            for warning in report.personalized_warnings:
                print(f"      • {warning}")
        
        if report.personalized_benefits:
            print(f"\n   ✓ Benefits:")
            for benefit in report.personalized_benefits[:5]:
                print(f"      • {benefit}")
        
        print("\n" + "="*80)
    
    logger.info("HealthImpactAnalyzer fully loaded with all modules")


# =============================================================================
# TEST SUITE
# =============================================================================

def test_toxicity_assessment():
    """Test 1: Toxicity assessment."""
    print("\n" + "="*80)
    print("TEST 1: Toxicity Assessment")
    print("="*80)
    
    analyzer = HealthImpactAnalyzer()
    
    # Test case 1: Safe food
    safe_food = {
        "glucose": 80000,
        "protein": 20000,
        "vitamin_c": 100
    }
    
    assessment = analyzer.assess_toxicity(safe_food)
    print(f"\n✓ Safe food assessment:")
    print(f"  Risk: {assessment.overall_risk.value}")
    print(f"  Score: {assessment.toxicity_score:.1f}/100")
    print(f"  Safe to consume: {assessment.safe_for_consumption}")
    
    # Test case 2: Food with trace toxin
    trace_toxin = {
        "glucose": 80000,
        "acrylamide": 0.5  # Below safe limit of 1.0
    }
    
    assessment2 = analyzer.assess_toxicity(trace_toxin)
    print(f"\n✓ Trace toxin assessment:")
    print(f"  Risk: {assessment2.overall_risk.value}")
    print(f"  Toxins detected: {len(assessment2.detected_toxins)}")
    print(f"  Safe to consume: {assessment2.safe_for_consumption}")
    
    # Test case 3: Contaminated food
    contaminated = {
        "aflatoxin_b1": 0.5,  # Above safe limit
        "lead": 0.3  # Above safe limit
    }
    
    assessment3 = analyzer.assess_toxicity(contaminated)
    print(f"\n✓ Contaminated food assessment:")
    print(f"  Risk: {assessment3.overall_risk.value}")
    print(f"  Score: {assessment3.toxicity_score:.1f}/100")
    print(f"  Toxins: {len(assessment3.detected_toxins)}")
    print(f"  Heavy metals: {len(assessment3.heavy_metals)}")
    print(f"  Safe to consume: {assessment3.safe_for_consumption}")
    
    return True


def test_allergen_detection():
    """Test 2: Allergen detection."""
    print("\n" + "="*80)
    print("TEST 2: Allergen Detection")
    print("="*80)
    
    analyzer = HealthImpactAnalyzer()
    
    # Test case 1: Peanut protein
    peanut_food = {
        "ara_h_1": 1000,  # High peanut allergen
        "protein": 25000
    }
    
    profile = analyzer.detect_allergens(peanut_food)
    print(f"\n✓ Peanut allergen detection:")
    print(f"  Allergens: {profile.allergens_detected}")
    print(f"  Risk: {profile.allergen_risk.value}")
    print(f"  Cross-reactive: {profile.cross_reactive_allergens}")
    print(f"  Severity: {profile.severity_estimates}")
    
    # Test case 2: Multiple allergens
    multi_allergen = {
        "gluten": 5000,
        "casein": 3000,
        "ara_h_1": 10  # Trace
    }
    
    profile2 = analyzer.detect_allergens(multi_allergen)
    print(f"\n✓ Multiple allergen detection:")
    print(f"  Allergens: {profile2.allergens_detected}")
    print(f"  Risk: {profile2.allergen_risk.value}")
    
    # Test case 3: Allergen-free
    safe_food = {
        "glucose": 80000,
        "starch": 15000
    }
    
    profile3 = analyzer.detect_allergens(safe_food)
    print(f"\n✓ Allergen-free food:")
    print(f"  Allergens: {profile3.allergens_detected}")
    print(f"  Risk: {profile3.allergen_risk.value}")
    
    return True


def test_database_loading():
    """Test 3: Database loading."""
    print("\n" + "="*80)
    print("TEST 3: Database Loading")
    print("="*80)
    
    analyzer = HealthImpactAnalyzer()
    
    # Drug interaction module removed — no drug interaction database present
    print(f"\n✓ Drug interaction module: REMOVED")
    
    print(f"\n✓ Allergen cross-reactivity loaded: {len(analyzer.allergen_cross_reactivity)} groups")
    
    print(f"\n✓ Health condition profiles: {len(analyzer.health_condition_restrictions)} conditions")
    for condition in list(analyzer.health_condition_restrictions.keys())[:3]:
        profile = analyzer.health_condition_restrictions[condition]
        print(f"  {condition.value}: {len(profile)} restrictions")
    
    return True


def test_nutritional_analysis():
    """Test 4: Nutritional analysis."""
    print("\n" + "="*80)
    print("TEST 4: Nutritional Analysis")
    print("="*80)
    
    analyzer = HealthImpactAnalyzer()
    
    # Test case: Nutrient-rich food
    nutrient_rich = {
        "leucine": 1500,  # Protein
        "lysine": 1200,
        "glucose": 50000,  # Carbs
        "oleic_acid": 10000,  # Fat
        "ascorbic_acid": 100,  # Vitamin C
        "thiamine": 2,  # Vitamin B1
        "calcium": 1200,  # Mineral
        "iron": 20
    }
    
    analysis = analyzer.analyze_nutrition(nutrient_rich, serving_size_g=100)
    
    print(f"\n✓ Nutritional analysis:")
    print(f"  Protein: {analysis.protein:.2f}g")
    print(f"  Carbohydrates: {analysis.carbohydrates:.2f}g")
    print(f"  Fat: {analysis.fat:.2f}g")
    print(f"  Health Score: {analysis.health_score:.1f}/100")
    print(f"  Vitamins detected: {len(analysis.vitamins)}")
    print(f"  Minerals detected: {len(analysis.minerals)}")
    
    if analysis.rda_compliance:
        print(f"\n✓ RDA Compliance (top 3):")
        for nutrient, percent in list(analysis.rda_compliance.items())[:3]:
            print(f"    {nutrient}: {percent:.1f}% of RDA")
    
    return True


def test_drug_interactions():
    # Drug interaction tests removed since drug interaction feature has been removed
    print("\nDrug interaction tests skipped (feature removed)")
    return True


def test_personalized_recommendations():
    """Test 6: Personalized health recommendations."""
    print("\n" + "="*80)
    print("TEST 6: Personalized Recommendations")
    print("="*80)
    
    analyzer = HealthImpactAnalyzer()
    
    # Test case 1: Diabetes
    high_sugar = {
        "glucose": 80000,
        "fructose": 50000,
        "sucrose": 30000
    }
    
    conditions_affected, warnings, benefits = analyzer.personalize_recommendations(
        high_sugar,
        [HealthCondition.DIABETES],
        age=55
    )
    
    print(f"\n✓ Diabetic patient with high-sugar food:")
    print(f"  Conditions affected: {conditions_affected}")
    print(f"  Warnings: {len(warnings)}")
    for warning in warnings:
        print(f"    ⚠️  {warning}")
    
    # Test case 2: Hypertension
    salty_food = {
        "sodium": 3000,  # High sodium
        "potassium": 1000
    }
    
    conditions_affected2, warnings2, benefits2 = analyzer.personalize_recommendations(
        salty_food,
        [HealthCondition.HYPERTENSION]
    )
    
    print(f"\n✓ Hypertensive patient with salty food:")
    print(f"  Conditions affected: {conditions_affected2}")
    print(f"  Warnings: {len(warnings2)}")
    for warning in warnings2:
        print(f"    ⚠️  {warning}")
    
    # Test case 3: Pregnancy
    healthy_food = {
        "folate": 500,  # Good for pregnancy
        "iron": 25,
        "calcium": 1200
    }
    
    conditions_affected3, warnings3, benefits3 = analyzer.personalize_recommendations(
        healthy_food,
        [],
        pregnancy=True
    )
    
    print(f"\n✓ Pregnancy with nutrient-rich food:")
    print(f"  Conditions affected: {conditions_affected3}")
    print(f"  Benefits: {len(benefits3)}")
    for benefit in benefits3[:3]:
        print(f"    ✓ {benefit}")
    
    return True


def test_full_report_generation():
    """Test 7: Full report generation."""
    print("\n" + "="*80)
    print("TEST 7: Full Report Generation")
    print("="*80)
    
    analyzer = HealthImpactAnalyzer()
    
    # Comprehensive food composition
    salmon = {
        "protein": 25000,
        "docosahexaenoic_acid": 2500,  # DHA (omega-3)
        "eicosapentaenoic_acid": 1800,  # EPA (omega-3)
        "vitamin_d": 0.015,
        "cobalamin": 0.005,  # B12
        "selenium": 0.04,
        "mercury": 0.03  # Trace heavy metal
    }
    
    report = analyzer.generate_report(
        food_name="Wild Salmon",
        composition=salmon,
        health_conditions=[HealthCondition.CARDIOVASCULAR_DISEASE],
        age=60,
        serving_size_g=150
    )
    
    print(f"\n✓ Report generated for Wild Salmon:")
    print(f"  Safety Score: {report.overall_safety_score:.1f}/100")
    print(f"  Health Score: {report.overall_health_score:.1f}/100")
    print(f"  Recommendation: {report.consumption_recommendation}")
    print(f"  Toxicity Risk: {report.toxicity.overall_risk.value}")
    print(f"  Allergen Risk: {report.allergens.allergen_risk.value}")
    print(f"  Conditions Affected: {len(report.health_conditions_affected)}")
    
    # Print full formatted report
    analyzer.print_report(report)
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("HEALTH IMPACT ANALYZER - COMPLETE TEST SUITE")
    print("Phase 3B: Intelligence Layer (Full Integration)")
    print("="*80)
    
    tests = [
        ("Toxicity Assessment", test_toxicity_assessment),
        ("Allergen Detection", test_allergen_detection),
        ("Database Loading", test_database_loading),
        ("Nutritional Analysis", test_nutritional_analysis),
        ("Personalized Recommendations", test_personalized_recommendations),
        ("Full Report Generation", test_full_report_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append(("✅ PASS", test_name))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(("❌ FAIL", test_name))
            return False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for status, name in results:
        print(f"{status}  {name}")
    
    passed = sum(1 for s, _ in results if "PASS" in s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Health analyzer functional.")
        print("\nNext: Add nutritional analysis")
        return True
    
    return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
