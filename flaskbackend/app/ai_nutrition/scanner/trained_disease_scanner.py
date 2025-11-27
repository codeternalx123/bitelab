"""
Trained Disease Scanner - Real-Time Food Analysis Using Trained AI

Integrates the Disease Training Engine with the MNT system to provide:
1. Real-time food scanning for ANY trained disease
2. Molecular-level requirement checking
3. Personalized recommendations based on health condition
4. Multi-disease optimization (user can have 10+ conditions)

WORKFLOW:
User has Diseases → Load Trained Profiles → Scan Food →
Check Molecular Content → Compare to Requirements →
Generate Recommendation (YES/NO/CAUTION)

KEY FEATURES:
- Supports 10,000+ trained diseases
- Molecular-level nutrient analysis
- Real-time "should I eat this?" decision
- Tells user EXACTLY what to avoid and why
- Provides molecular quantities (not just percentages)

Author: Atomic AI System
Date: November 7, 2025
Version: 1.0.0 - Trained Disease Integration
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

# Import training engine
from disease_training_engine import (
    DiseaseTrainingEngine,
    DiseaseKnowledge,
    NutrientRequirement
)

# Import existing modules
from integrated_nutrition_ai import (
    IntegratedNutritionAI,
    ComprehensiveRecommendation,
    MolecularBreakdown,
    UserAlert,
    AlertType,
    RecommendationLevel
)
from multi_condition_optimizer import (
    MultiConditionOptimizer,
    UserHealthProfile,
    DiseaseMolecularProfile
)
from mnt_api_integration import MNTAPIManager, Food

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class RequirementViolation:
    """A specific nutrient requirement that was violated"""
    nutrient_name: str
    requirement: NutrientRequirement
    actual_value: float
    actual_unit: str
    severity: str  # "critical", "high", "moderate", "low"
    explanation: str


@dataclass
class MolecularQuantityReport:
    """Detailed molecular quantities in the food"""
    food_name: str
    serving_size_g: float
    
    # Macro molecules (grams)
    protein_g: float = 0.0
    carbohydrates_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0
    sugar_g: float = 0.0
    
    # Minerals (mg)
    sodium_mg: float = 0.0
    potassium_mg: float = 0.0
    calcium_mg: float = 0.0
    iron_mg: float = 0.0
    magnesium_mg: float = 0.0
    zinc_mg: float = 0.0
    
    # Vitamins
    vitamin_d_iu: float = 0.0
    vitamin_c_mg: float = 0.0
    vitamin_b12_mcg: float = 0.0
    folate_mcg: float = 0.0
    
    # Other
    cholesterol_mg: float = 0.0
    omega_3_g: float = 0.0
    
    # Percentages (for visual display)
    protein_pct: float = 0.0
    carb_pct: float = 0.0
    fat_pct: float = 0.0


@dataclass
class DiseaseSpecificDecision:
    """Decision for a specific disease"""
    disease_name: str
    should_consume: bool
    risk_level: str  # "safe", "caution", "danger"
    violations: List[RequirementViolation] = field(default_factory=list)
    passed_requirements: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class TrainedDiseaseRecommendation:
    """Complete recommendation using trained disease data"""
    food_name: str
    overall_decision: bool  # Can user eat this?
    overall_risk: str  # "safe", "caution", "danger"
    
    # Molecular analysis
    molecular_quantities: MolecularQuantityReport
    
    # Per-disease decisions
    disease_decisions: List[DiseaseSpecificDecision] = field(default_factory=list)
    
    # What to avoid
    critical_violations: List[RequirementViolation] = field(default_factory=list)
    
    # Recommendations
    recommendation_text: str = ""
    alternative_suggestions: List[str] = field(default_factory=list)
    
    # Alerts
    alerts: List[UserAlert] = field(default_factory=list)


# ============================================================================
# TRAINED DISEASE SCANNER - MAIN CLASS
# ============================================================================

class TrainedDiseaseScanner:
    """
    Scans food using trained disease knowledge
    
    This is the "Digital Dietitian" that:
    1. Knows 10,000+ diseases and their requirements
    2. Scans food to get exact molecular quantities
    3. Compares food to EACH disease requirement
    4. Gives clear YES/NO/CAUTION decision
    5. Explains WHY (which nutrients are problems)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Core components
        self.training_engine = DiseaseTrainingEngine(config)
        self.integrated_ai = IntegratedNutritionAI(config)
        self.multi_condition_optimizer = MultiConditionOptimizer()
        self.mnt_api = MNTAPIManager(config)
        
        # Cache of trained diseases
        self.trained_diseases: Dict[str, DiseaseKnowledge] = {}
        
        logger.info("Trained Disease Scanner initialized")
    
    async def initialize(self) -> None:
        """Initialize all components"""
        await self.training_engine.initialize()
        await self.integrated_ai.initialize()
        await self.mnt_api.initialize()
        
        logger.info("Trained Disease Scanner ready")
    
    async def load_trained_diseases(self, disease_names: List[str]) -> None:
        """Load and train on specific diseases"""
        logger.info(f"Loading {len(disease_names)} trained diseases...")
        
        await self.training_engine.train_on_disease_list(disease_names)
        self.trained_diseases = self.training_engine.trained_diseases
        
        logger.info(f"✓ Loaded {len(self.trained_diseases)} trained diseases")
    
    async def scan_food_for_user(
        self,
        food_identifier: str,
        user_diseases: List[str],
        scan_mode: str = "text"
    ) -> TrainedDiseaseRecommendation:
        """
        Main scanning method
        
        Args:
            food_identifier: Food name, barcode, or NIR data
            user_diseases: List of disease names user has
            scan_mode: "text", "barcode", or "nir"
        
        Returns:
            Complete recommendation with molecular analysis
        """
        
        # Step 1: Get food data
        logger.info(f"Scanning food: {food_identifier}")
        food = await self._get_food_data(food_identifier, scan_mode)
        
        if not food:
            raise ValueError(f"Could not find food: {food_identifier}")
        
        # Step 2: Extract molecular quantities
        molecular_report = self._extract_molecular_quantities(food)
        
        # Step 3: Load disease requirements
        disease_requirements = await self._load_disease_requirements(user_diseases)
        
        # Step 4: Check EACH disease requirement
        disease_decisions = []
        all_violations = []
        
        for disease_name, requirements in disease_requirements.items():
            decision = self._check_disease_requirements(
                disease_name,
                requirements,
                molecular_report
            )
            disease_decisions.append(decision)
            all_violations.extend(decision.violations)
        
        # Step 5: Make overall decision
        overall_decision = self._make_overall_decision(disease_decisions)
        
        # Step 6: Generate recommendation text
        recommendation_text = self._generate_recommendation_text(
            disease_decisions,
            all_violations
        )
        
        # Step 7: Find critical violations
        critical_violations = [
            v for v in all_violations
            if v.severity in ["critical", "high"]
        ]
        
        # Step 8: Generate alerts
        alerts = self._generate_alerts(critical_violations)
        
        # Step 9: Suggest alternatives if needed
        alternatives = []
        if not overall_decision["should_consume"]:
            alternatives = await self._suggest_alternatives(food.name)
        
        # Create final recommendation
        recommendation = TrainedDiseaseRecommendation(
            food_name=food.name,
            overall_decision=overall_decision["should_consume"],
            overall_risk=overall_decision["risk_level"],
            molecular_quantities=molecular_report,
            disease_decisions=disease_decisions,
            critical_violations=critical_violations,
            recommendation_text=recommendation_text,
            alternative_suggestions=alternatives,
            alerts=alerts
        )
        
        logger.info(f"✓ Scan complete: {food.name} - {overall_decision['risk_level'].upper()}")
        
        return recommendation
    
    async def _get_food_data(
        self,
        food_identifier: str,
        scan_mode: str
    ) -> Optional[Food]:
        """Get food data from API"""
        
        if scan_mode == "text":
            # Search by text
            results = await self.mnt_api.search_foods(food_identifier, max_results=1)
            return results[0] if results else None
        
        elif scan_mode == "barcode":
            # Lookup by barcode
            return await self.mnt_api.get_food_by_barcode(food_identifier)
        
        elif scan_mode == "nir":
            # NIR scanning (would use integrated_ai)
            # For now, fallback to text search
            results = await self.mnt_api.search_foods(food_identifier, max_results=1)
            return results[0] if results else None
        
        return None
    
    def _extract_molecular_quantities(self, food: Food) -> MolecularQuantityReport:
        """Extract all molecular quantities from food data"""
        
        serving_g = food.serving_size_g or 100.0
        
        # Calculate percentages
        total_mass = (
            food.nutrients.get("protein", 0) +
            food.nutrients.get("carbohydrates", 0) +
            food.nutrients.get("fat", 0)
        )
        
        protein_pct = (food.nutrients.get("protein", 0) / serving_g * 100) if serving_g > 0 else 0
        carb_pct = (food.nutrients.get("carbohydrates", 0) / serving_g * 100) if serving_g > 0 else 0
        fat_pct = (food.nutrients.get("fat", 0) / serving_g * 100) if serving_g > 0 else 0
        
        return MolecularQuantityReport(
            food_name=food.name,
            serving_size_g=serving_g,
            
            # Macros
            protein_g=food.nutrients.get("protein", 0),
            carbohydrates_g=food.nutrients.get("carbohydrates", 0),
            fat_g=food.nutrients.get("fat", 0),
            fiber_g=food.nutrients.get("fiber", 0),
            sugar_g=food.nutrients.get("sugar", 0),
            
            # Minerals
            sodium_mg=food.nutrients.get("sodium", 0),
            potassium_mg=food.nutrients.get("potassium", 0),
            calcium_mg=food.nutrients.get("calcium", 0),
            iron_mg=food.nutrients.get("iron", 0),
            magnesium_mg=food.nutrients.get("magnesium", 0),
            zinc_mg=food.nutrients.get("zinc", 0),
            
            # Vitamins
            vitamin_d_iu=food.nutrients.get("vitamin_d", 0),
            vitamin_c_mg=food.nutrients.get("vitamin_c", 0),
            vitamin_b12_mcg=food.nutrients.get("vitamin_b12", 0),
            folate_mcg=food.nutrients.get("folate", 0),
            
            # Other
            cholesterol_mg=food.nutrients.get("cholesterol", 0),
            omega_3_g=food.nutrients.get("omega_3", 0),
            
            # Percentages
            protein_pct=protein_pct,
            carb_pct=carb_pct,
            fat_pct=fat_pct
        )
    
    async def _load_disease_requirements(
        self,
        disease_names: List[str]
    ) -> Dict[str, List[NutrientRequirement]]:
        """Load requirements for all user diseases"""
        
        requirements = {}
        
        for disease_name in disease_names:
            # Check if already trained
            if disease_name not in self.trained_diseases:
                # Train on this disease
                knowledge = await self.training_engine.fetch_disease_knowledge(disease_name)
                if knowledge:
                    self.trained_diseases[disease_name] = knowledge
            
            # Get requirements
            if disease_name in self.trained_diseases:
                requirements[disease_name] = self.trained_diseases[disease_name].nutrient_requirements
        
        return requirements
    
    def _check_disease_requirements(
        self,
        disease_name: str,
        requirements: List[NutrientRequirement],
        molecular_report: MolecularQuantityReport
    ) -> DiseaseSpecificDecision:
        """Check if food meets requirements for a specific disease"""
        
        violations = []
        passed = []
        
        for req in requirements:
            # Get actual value from molecular report
            actual_value = self._get_nutrient_value(req.nutrient_name, molecular_report)
            actual_unit = self._get_nutrient_unit(req.nutrient_name)
            
            # Check requirement
            is_violated = False
            
            if req.requirement_type == "limit" and req.value:
                # Should be less than value
                if actual_value > req.value:
                    is_violated = True
            
            elif req.requirement_type == "increase" and req.value:
                # Should be more than value
                if actual_value < req.value:
                    is_violated = True
            
            elif req.requirement_type == "avoid":
                # Should be zero or very low
                if actual_value > 0:
                    is_violated = True
            
            if is_violated:
                # Calculate severity
                severity = self._calculate_violation_severity(
                    req,
                    actual_value
                )
                
                violation = RequirementViolation(
                    nutrient_name=req.nutrient_name,
                    requirement=req,
                    actual_value=actual_value,
                    actual_unit=actual_unit,
                    severity=severity,
                    explanation=self._generate_violation_explanation(
                        req,
                        actual_value,
                        actual_unit
                    )
                )
                violations.append(violation)
            else:
                passed.append(req.nutrient_name)
        
        # Determine risk level for this disease
        risk_level = "safe"
        if any(v.severity == "critical" for v in violations):
            risk_level = "danger"
        elif any(v.severity == "high" for v in violations):
            risk_level = "danger"
        elif violations:
            risk_level = "caution"
        
        # Should consume?
        should_consume = risk_level != "danger"
        
        # Generate reasoning
        reasoning = self._generate_disease_reasoning(
            disease_name,
            violations,
            passed
        )
        
        return DiseaseSpecificDecision(
            disease_name=disease_name,
            should_consume=should_consume,
            risk_level=risk_level,
            violations=violations,
            passed_requirements=passed,
            reasoning=reasoning
        )
    
    def _get_nutrient_value(
        self,
        nutrient_name: str,
        molecular_report: MolecularQuantityReport
    ) -> float:
        """Get actual nutrient value from molecular report"""
        
        nutrient_map = {
            "sodium": molecular_report.sodium_mg,
            "potassium": molecular_report.potassium_mg,
            "calcium": molecular_report.calcium_mg,
            "iron": molecular_report.iron_mg,
            "magnesium": molecular_report.magnesium_mg,
            "zinc": molecular_report.zinc_mg,
            "protein": molecular_report.protein_g * 1000,  # Convert to mg
            "carbohydrates": molecular_report.carbohydrates_g * 1000,
            "fiber": molecular_report.fiber_g * 1000,
            "fat": molecular_report.fat_g * 1000,
            "sugar": molecular_report.sugar_g * 1000,
            "cholesterol": molecular_report.cholesterol_mg,
            "vitamin_d": molecular_report.vitamin_d_iu,
            "vitamin_c": molecular_report.vitamin_c_mg,
            "vitamin_b12": molecular_report.vitamin_b12_mcg,
            "folate": molecular_report.folate_mcg,
            "omega_3": molecular_report.omega_3_g * 1000
        }
        
        return nutrient_map.get(nutrient_name, 0.0)
    
    def _get_nutrient_unit(self, nutrient_name: str) -> str:
        """Get standard unit for nutrient"""
        
        unit_map = {
            "sodium": "mg",
            "potassium": "mg",
            "calcium": "mg",
            "iron": "mg",
            "magnesium": "mg",
            "zinc": "mg",
            "protein": "g",
            "carbohydrates": "g",
            "fiber": "g",
            "fat": "g",
            "sugar": "g",
            "cholesterol": "mg",
            "vitamin_d": "IU",
            "vitamin_c": "mg",
            "vitamin_b12": "mcg",
            "folate": "mcg",
            "omega_3": "g"
        }
        
        return unit_map.get(nutrient_name, "mg")
    
    def _calculate_violation_severity(
        self,
        requirement: NutrientRequirement,
        actual_value: float
    ) -> str:
        """Calculate how severe the violation is"""
        
        if requirement.requirement_type == "avoid":
            return "critical"
        
        if not requirement.value:
            return "moderate"
        
        # Calculate percentage over/under
        if requirement.requirement_type == "limit":
            ratio = actual_value / requirement.value
            if ratio > 3.0:
                return "critical"
            elif ratio > 2.0:
                return "high"
            elif ratio > 1.5:
                return "moderate"
            else:
                return "low"
        
        elif requirement.requirement_type == "increase":
            ratio = requirement.value / actual_value if actual_value > 0 else 999
            if ratio > 3.0:
                return "critical"
            elif ratio > 2.0:
                return "high"
            elif ratio > 1.5:
                return "moderate"
            else:
                return "low"
        
        return "moderate"
    
    def _generate_violation_explanation(
        self,
        requirement: NutrientRequirement,
        actual_value: float,
        actual_unit: str
    ) -> str:
        """Generate human-readable explanation of violation"""
        
        if requirement.requirement_type == "limit":
            return (
                f"{requirement.nutrient_name.upper()}: {actual_value:.1f}{actual_unit} "
                f"exceeds limit of {requirement.value}{requirement.unit or actual_unit}. "
                f"This can worsen your {requirement.reasoning}"
            )
        
        elif requirement.requirement_type == "increase":
            return (
                f"{requirement.nutrient_name.upper()}: {actual_value:.1f}{actual_unit} "
                f"is below recommended {requirement.value}{requirement.unit or actual_unit}. "
                f"You need more for: {requirement.reasoning}"
            )
        
        elif requirement.requirement_type == "avoid":
            return (
                f"{requirement.nutrient_name.upper()}: Present ({actual_value:.1f}{actual_unit}). "
                f"Should be AVOIDED because: {requirement.reasoning}"
            )
        
        return f"{requirement.nutrient_name}: Violation detected"
    
    def _generate_disease_reasoning(
        self,
        disease_name: str,
        violations: List[RequirementViolation],
        passed: List[str]
    ) -> str:
        """Generate reasoning for disease-specific decision"""
        
        if not violations:
            return f"✓ Safe for {disease_name}: All requirements met ({len(passed)} checks passed)"
        
        critical_count = sum(1 for v in violations if v.severity in ["critical", "high"])
        
        if critical_count > 0:
            return (
                f"✗ DANGER for {disease_name}: {critical_count} critical violations. "
                f"AVOID this food."
            )
        else:
            return (
                f"⚠ CAUTION for {disease_name}: {len(violations)} moderate violations. "
                f"Limit consumption."
            )
    
    def _make_overall_decision(
        self,
        disease_decisions: List[DiseaseSpecificDecision]
    ) -> Dict[str, Any]:
        """Make overall decision considering ALL diseases"""
        
        # If ANY disease says DANGER, overall is DANGER
        if any(d.risk_level == "danger" for d in disease_decisions):
            return {
                "should_consume": False,
                "risk_level": "danger"
            }
        
        # If ANY disease says CAUTION, overall is CAUTION
        if any(d.risk_level == "caution" for d in disease_decisions):
            return {
                "should_consume": True,  # Can eat but with caution
                "risk_level": "caution"
            }
        
        # All diseases say SAFE
        return {
            "should_consume": True,
            "risk_level": "safe"
        }
    
    def _generate_recommendation_text(
        self,
        disease_decisions: List[DiseaseSpecificDecision],
        all_violations: List[RequirementViolation]
    ) -> str:
        """Generate final recommendation text"""
        
        # Count risk levels
        danger_count = sum(1 for d in disease_decisions if d.risk_level == "danger")
        caution_count = sum(1 for d in disease_decisions if d.risk_level == "caution")
        safe_count = sum(1 for d in disease_decisions if d.risk_level == "safe")
        
        if danger_count > 0:
            text = f"🚫 DO NOT CONSUME\n\n"
            text += f"This food is DANGEROUS for {danger_count} of your conditions:\n"
            
            for decision in disease_decisions:
                if decision.risk_level == "danger":
                    text += f"\n• {decision.disease_name}: {len(decision.violations)} violations\n"
                    for v in decision.violations[:3]:  # Top 3
                        text += f"  - {v.explanation}\n"
        
        elif caution_count > 0:
            text = f"⚠️ CONSUME WITH CAUTION\n\n"
            text += f"This food has concerns for {caution_count} of your conditions:\n"
            
            for decision in disease_decisions:
                if decision.risk_level == "caution":
                    text += f"\n• {decision.disease_name}: {len(decision.violations)} moderate issues\n"
        
        else:
            text = f"✅ SAFE TO CONSUME\n\n"
            text += f"This food is safe for all {safe_count} of your conditions.\n"
        
        return text
    
    def _generate_alerts(
        self,
        critical_violations: List[RequirementViolation]
    ) -> List[UserAlert]:
        """Generate alerts for critical violations"""
        
        alerts = []
        
        for violation in critical_violations:
            alert = UserAlert(
                alert_type=AlertType.CONTRAINDICATION,
                severity="HIGH" if violation.severity == "critical" else "MODERATE",
                title=f"{violation.nutrient_name.upper()} Warning",
                message=violation.explanation,
                details={
                    "nutrient": violation.nutrient_name,
                    "actual": violation.actual_value,
                    "unit": violation.actual_unit,
                    "severity": violation.severity
                },
                recommended_action="Avoid this food or consult your doctor"
            )
            alerts.append(alert)
        
        return alerts
    
    async def _suggest_alternatives(self, food_name: str) -> List[str]:
        """Suggest alternative foods"""
        
        # Simple category-based suggestions
        alternatives = []
        
        if "chicken" in food_name.lower():
            alternatives = ["Turkey breast", "Tofu", "White fish"]
        elif "beef" in food_name.lower():
            alternatives = ["Lean turkey", "Chicken breast", "Legumes"]
        elif "bread" in food_name.lower():
            alternatives = ["Whole grain bread", "Oat bread", "Lettuce wrap"]
        elif "rice" in food_name.lower():
            alternatives = ["Quinoa", "Cauliflower rice", "Brown rice"]
        else:
            alternatives = ["Consult your dietitian for alternatives"]
        
        return alternatives


# ============================================================================
# EXAMPLE USAGE - REAL WORLD SCENARIO
# ============================================================================

async def real_world_example():
    """
    Real-world example: User with multiple conditions scans food
    """
    
    print("\n" + "=" * 80)
    print("TRAINED DISEASE SCANNER - REAL-WORLD EXAMPLE")
    print("=" * 80 + "\n")
    
    # Initialize scanner
    scanner = TrainedDiseaseScanner(config={
        "edamam_app_id": "DEMO",
        "edamam_app_key": "DEMO"
    })
    await scanner.initialize()
    
    # Load diseases to train on
    print("Training on user's conditions...")
    user_diseases = ["Hypertension", "Type 2 Diabetes", "Chronic Kidney Disease"]
    await scanner.load_trained_diseases(user_diseases)
    
    # User scans canned soup
    print("\n" + "=" * 80)
    print("USER SCANS: Canned Chicken Noodle Soup")
    print("=" * 80 + "\n")
    
    recommendation = await scanner.scan_food_for_user(
        food_identifier="chicken noodle soup",
        user_diseases=user_diseases,
        scan_mode="text"
    )
    
    # Display results
    print(f"Food: {recommendation.food_name}")
    print(f"Overall Decision: {'✅ YES' if recommendation.overall_decision else '🚫 NO'}")
    print(f"Risk Level: {recommendation.overall_risk.upper()}\n")
    
    print("-" * 80)
    print("MOLECULAR QUANTITIES (Per Serving)")
    print("-" * 80)
    mol = recommendation.molecular_quantities
    print(f"  Serving Size: {mol.serving_size_g}g")
    print(f"\n  Macronutrients:")
    print(f"    Protein: {mol.protein_g}g ({mol.protein_pct:.1f}%)")
    print(f"    Carbs: {mol.carbohydrates_g}g ({mol.carb_pct:.1f}%)")
    print(f"    Fat: {mol.fat_g}g ({mol.fat_pct:.1f}%)")
    print(f"    Sugar: {mol.sugar_g}g")
    print(f"    Fiber: {mol.fiber_g}g")
    print(f"\n  Minerals:")
    print(f"    Sodium: {mol.sodium_mg}mg ⚠️")
    print(f"    Potassium: {mol.potassium_mg}mg")
    print(f"    Calcium: {mol.calcium_mg}mg")
    print(f"    Iron: {mol.iron_mg}mg")
    
    print("\n" + "-" * 80)
    print("DISEASE-SPECIFIC ANALYSIS")
    print("-" * 80)
    for decision in recommendation.disease_decisions:
        icon = "✅" if decision.should_consume else "🚫"
        print(f"\n{icon} {decision.disease_name} [{decision.risk_level.upper()}]")
        print(f"   {decision.reasoning}")
        
        if decision.violations:
            print(f"\n   Violations ({len(decision.violations)}):")
            for v in decision.violations[:3]:
                print(f"     • {v.explanation}")
    
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    print(recommendation.recommendation_text)
    
    if recommendation.alternative_suggestions:
        print("\nAlternative Suggestions:")
        for alt in recommendation.alternative_suggestions:
            print(f"  • {alt}")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(real_world_example())
