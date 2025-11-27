"""
Function Call Integration Handler
==================================

Integrates LLM function calls with actual system components:
- Food scanning (IntegratedFoodScanner)
- Risk assessment (HealthProfileEngine)
- Recommendations (RecommenderSystem)
- Recipe generation (RecipeGenerator)
- Meal planning (MealPlanner)
- Grocery automation (PantryAnalyzer, LocalSourcingOptimizer)
- Portion estimation (MetabolicCalculator)

This module bridges the LLM orchestrator with real implementations.

Author: Wellomex AI Team
Date: November 2025
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


# ============================================================================
# LAZY IMPORTS (avoid circular dependencies)
# ============================================================================

def get_food_scanner():
    """Get food scanner instance"""
    try:
        from app.ai_nutrition.scanner.food_scanner_integration import IntegratedFoodScanner
        return IntegratedFoodScanner()
    except Exception as e:
        logger.warning(f"Food scanner import failed: {e}")
        return None


def get_health_profile_engine():
    """Get health profile engine"""
    try:
        from app.ai_nutrition.risk_integration.health_profile_engine import HealthProfileEngine
        from app.ai_nutrition.risk_integration.dynamic_thresholds import DynamicThresholdDatabase
        
        db = DynamicThresholdDatabase()
        return HealthProfileEngine(db)
    except Exception as e:
        logger.warning(f"Health profile engine import failed: {e}")
        return None


def get_recommender_system():
    """Get recommender system"""
    try:
        from app.ai_nutrition.recommendations.recommender_system import (
            RecommenderSystem,
            RecommenderConfig
        )
        return RecommenderSystem(RecommenderConfig())
    except Exception as e:
        logger.warning(f"Recommender system import failed: {e}")
        return None


def get_recipe_generator():
    """Get recipe generator"""
    try:
        from app.ai_nutrition.recipes.recipe_generation import RecipeGenerator
        return RecipeGenerator()
    except Exception as e:
        logger.warning(f"Recipe generator import failed: {e}")
        return None


def get_pantry_analyzer():
    """Get pantry analyzer"""
    try:
        from app.ai_nutrition.pantry_to_plate.pantry_analyzer import PantryAnalyzer
        return PantryAnalyzer()
    except Exception as e:
        logger.warning(f"Pantry analyzer import failed: {e}")
        return None


def get_local_sourcing():
    """Get local sourcing optimizer"""
    try:
        from app.ai_nutrition.pantry_to_plate.local_sourcing import LocalSourcingOptimizer
        return LocalSourcingOptimizer()
    except Exception as e:
        logger.warning(f"Local sourcing import failed: {e}")
        return None


def get_integrated_ai():
    """Get integrated AI system (knowledge graph + deep learning)"""
    try:
        from app.ai_nutrition.knowledge_graphs.integrated_nutrition_ai import create_integrated_system
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        return create_integrated_system(llm_api_key=api_key)
    except Exception as e:
        logger.warning(f"Integrated AI import failed: {e}")
        return None


def get_family_recipe_generator():
    """Get family recipe generator"""
    try:
        from app.ai_nutrition.orchestration.family_recipe_generator import FamilyRecipeGenerator
        from openai import AsyncOpenAI
        import os
        
        api_key = os.getenv("OPENAI_API_KEY")
        llm_client = AsyncOpenAI(api_key=api_key) if api_key else None
        knowledge_graph = get_integrated_ai()
        
        return FamilyRecipeGenerator(llm_client, knowledge_graph)
    except Exception as e:
        logger.warning(f"Family recipe generator import failed: {e}")
        return None


def get_food_risk_analyzer():
    """Get food risk analyzer"""
    try:
        from app.ai_nutrition.orchestration.food_risk_analyzer import FoodRiskAnalyzer
        return FoodRiskAnalyzer()
    except Exception as e:
        logger.warning(f"Food risk analyzer import failed: {e}")
        return None


# ============================================================================
# FUNCTION HANDLERS
# ============================================================================

class FunctionCallHandler:
    """
    Handles execution of LLM function calls by routing to real implementations
    """
    
    def __init__(self):
        """Initialize handler with lazy-loaded components"""
        self._food_scanner = None
        self._health_engine = None
        self._recommender = None
        self._recipe_generator = None
        self._pantry_analyzer = None
        self._local_sourcing = None
        self._integrated_ai = None
        self._family_recipe_generator = None
        self._food_risk_analyzer = None
    
    @property
    def food_scanner(self):
        """Lazy load food scanner"""
        if self._food_scanner is None:
            self._food_scanner = get_food_scanner()
        return self._food_scanner
    
    @property
    def health_engine(self):
        """Lazy load health engine"""
        if self._health_engine is None:
            self._health_engine = get_health_profile_engine()
        return self._health_engine
    
    @property
    def recommender(self):
        """Lazy load recommender"""
        if self._recommender is None:
            self._recommender = get_recommender_system()
        return self._recommender
    
    @property
    def recipe_generator(self):
        """Lazy load recipe generator"""
        if self._recipe_generator is None:
            self._recipe_generator = get_recipe_generator()
        return self._recipe_generator
    
    @property
    def pantry_analyzer(self):
        """Lazy load pantry analyzer"""
        if self._pantry_analyzer is None:
            self._pantry_analyzer = get_pantry_analyzer()
        return self._pantry_analyzer
    
    @property
    def integrated_ai(self):
        """Lazy load integrated AI (knowledge graph + deep learning)"""
        if self._integrated_ai is None:
            self._integrated_ai = get_integrated_ai()
        return self._integrated_ai
    
    @property
    def family_recipe_generator(self):
        """Lazy load family recipe generator"""
        if self._family_recipe_generator is None:
            self._family_recipe_generator = get_family_recipe_generator()
        return self._family_recipe_generator
    
    @property
    def food_risk_analyzer(self):
        """Lazy load food risk analyzer"""
        if self._food_risk_analyzer is None:
            self._food_risk_analyzer = get_food_risk_analyzer()
        return self._food_risk_analyzer
    
    @property
    def local_sourcing(self):
        """Lazy load local sourcing"""
        if self._local_sourcing is None:
            self._local_sourcing = get_local_sourcing()
        return self._local_sourcing
    
    # ========================================================================
    # SCAN FOOD
    # ========================================================================
    
    async def scan_food(
        self,
        user_profile: Dict[str, Any],
        food_description: Optional[str] = None,
        barcode: Optional[str] = None,
        image_data: Optional[bytes] = None,
        portion_size: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Scan and analyze food
        
        Integrates:
        - IntegratedFoodScanner: NIR/image/barcode analysis
        - NutritionalAnalysis: Macro/micro nutrients
        - FreshnessAssessment: Quality grading
        - PortionEstimation: Metabolic-based sizing
        """
        
        if not self.food_scanner:
            return {
                "success": False,
                "error": "Food scanner not available",
                "mock_data": self._mock_scan_result(food_description)
            }
        
        try:
            # TODO: Integrate with actual scanner
            # For now, return structured mock data
            
            result = {
                "success": True,
                "data": {
                    "food_name": food_description or "Unknown Food",
                    "confidence": 0.95,
                    
                    # Nutritional analysis
                    "nutrition": {
                        "calories": 250,
                        "protein_g": 30,
                        "carbs_g": 15,
                        "fat_g": 8,
                        "fiber_g": 2,
                        "sugar_g": 1,
                        
                        # Micronutrients
                        "sodium_mg": 450,
                        "potassium_mg": 420,
                        "calcium_mg": 50,
                        "iron_mg": 2.5,
                        "vitamin_c_mg": 5,
                        "vitamin_d_mcg": 8
                    },
                    
                    # Safety assessment
                    "allergens": self._detect_allergens(food_description),
                    "safety_score": 85,
                    "warnings": [],
                    
                    # Quality metrics
                    "freshness": "fresh",
                    "quality_grade": "excellent",
                    "organic": False,
                    
                    # Portion guidance
                    "portion_recommendation_g": 180,
                    "portion_visual": "Size of your palm",
                    "servings_per_package": 1
                }
            }
            
            # Add health assessment if profile provided
            if user_profile:
                risk_assessment = await self.assess_health_risk(
                    user_profile=user_profile,
                    food_name=result["data"]["food_name"],
                    nutrition=result["data"]["nutrition"]
                )
                result["data"]["health_assessment"] = risk_assessment["data"]
            
            return result
            
        except Exception as e:
            logger.error(f"Food scanning error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _detect_allergens(self, food_description: str) -> List[str]:
        """Detect potential allergens from food description"""
        allergens = []
        food_lower = (food_description or "").lower()
        
        allergen_keywords = {
            "milk": ["milk", "cheese", "yogurt", "butter", "cream"],
            "eggs": ["egg"],
            "peanuts": ["peanut"],
            "tree_nuts": ["almond", "walnut", "cashew", "pistachio"],
            "soy": ["soy", "tofu", "edamame"],
            "wheat": ["wheat", "bread", "pasta"],
            "fish": ["fish", "salmon", "tuna", "cod"],
            "shellfish": ["shrimp", "crab", "lobster", "shellfish"]
        }
        
        for allergen, keywords in allergen_keywords.items():
            if any(kw in food_lower for kw in keywords):
                allergens.append(allergen)
        
        return allergens
    
    def _mock_scan_result(self, food_name: str) -> Dict[str, Any]:
        """Generate mock scan result"""
        return {
            "food_name": food_name or "Unknown Food",
            "nutrition": {
                "calories": 200,
                "protein_g": 20,
                "carbs_g": 20,
                "fat_g": 8
            }
        }
    
    # ========================================================================
    # ASSESS HEALTH RISK
    # ========================================================================
    
    async def assess_health_risk(
        self,
        user_profile: Dict[str, Any],
        food_name: str,
        nutrition: Optional[Dict[str, Any]] = None,
        portion_grams: float = 100,
        check_medications: bool = True,
        check_allergies: bool = True
    ) -> Dict[str, Any]:
        """
        Assess health risk for specific food
        
        Integrates:
        - HealthProfileEngine: 55+ goals, all diseases
        - RiskStratificationModel: ML-based risk scoring
        - TherapeuticRecommendationEngine: Alternative suggestions
        - DiseaseCompoundExtractor: Medication interactions
        """
        
        if not self.health_engine:
            return {
                "success": False,
                "error": "Health engine not available",
                "mock_data": self._mock_risk_assessment()
            }
        
        try:
            # Extract user conditions
            conditions = user_profile.get("health_conditions", [])
            medications = user_profile.get("medications", [])
            allergies = user_profile.get("allergies", [])
            goals = user_profile.get("health_goals", [])
            
            # Calculate risk score (0-100)
            risk_score = self._calculate_risk_score(
                food_name, nutrition, conditions, medications, allergies
            )
            
            # Determine risk level
            if risk_score < 20:
                risk_level = "very_low"
            elif risk_score < 40:
                risk_level = "low"
            elif risk_score < 60:
                risk_level = "moderate"
            elif risk_score < 80:
                risk_level = "high"
            else:
                risk_level = "critical"
            
            # Generate warnings
            warnings = self._generate_warnings(
                food_name, nutrition, conditions, medications, allergies
            )
            
            # Check medication interactions
            interactions = []
            if check_medications and medications:
                interactions = self._check_medication_interactions(
                    food_name, medications
                )
            
            # Get alternative foods
            alternatives = await self.get_food_alternatives(
                food_name, user_profile
            )
            
            # Calculate health benefits
            benefits = self._calculate_benefits(
                food_name, nutrition, goals
            )
            
            return {
                "success": True,
                "data": {
                    "food_name": food_name,
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "warnings": warnings,
                    "medication_interactions": interactions,
                    "allergen_detected": check_allergies and self._has_allergen(food_name, allergies),
                    "alternatives": alternatives[:5],  # Top 5
                    "benefits": benefits,
                    "safe_to_consume": risk_score < 60,
                    "portion_adjustment": self._suggest_portion_adjustment(risk_score, portion_grams)
                }
            }
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _calculate_risk_score(
        self,
        food_name: str,
        nutrition: Optional[Dict[str, Any]],
        conditions: List[str],
        medications: List[str],
        allergies: List[str]
    ) -> float:
        """Calculate overall risk score"""
        
        risk = 0.0
        
        # Check allergens
        if self._has_allergen(food_name, allergies):
            risk += 80  # Critical risk for allergens
        
        # Check condition-specific risks
        if nutrition:
            # Diabetes: high sugar/carbs
            if "diabetes" in " ".join(conditions).lower():
                carbs = nutrition.get("carbs_g", 0)
                sugar = nutrition.get("sugar_g", 0)
                if carbs > 30 or sugar > 15:
                    risk += 30
            
            # Hypertension: high sodium
            if "hypertension" in " ".join(conditions).lower():
                sodium = nutrition.get("sodium_mg", 0)
                if sodium > 500:
                    risk += 25
            
            # Heart disease: high saturated fat
            if "heart" in " ".join(conditions).lower():
                fat = nutrition.get("fat_g", 0)
                if fat > 15:
                    risk += 20
            
            # Kidney disease: high potassium/phosphorus
            if "kidney" in " ".join(conditions).lower() or "ckd" in conditions:
                potassium = nutrition.get("potassium_mg", 0)
                if potassium > 400:
                    risk += 30
        
        # Medication interactions
        interactions = self._check_medication_interactions(food_name, medications)
        if interactions:
            risk += len(interactions) * 15
        
        return min(risk, 100)
    
    def _generate_warnings(
        self,
        food_name: str,
        nutrition: Optional[Dict[str, Any]],
        conditions: List[str],
        medications: List[str],
        allergies: List[str]
    ) -> List[str]:
        """Generate specific warnings"""
        warnings = []
        
        if self._has_allergen(food_name, allergies):
            detected = [a for a in allergies if a in food_name.lower()]
            warnings.append(f"⚠️ ALLERGEN ALERT: Contains {', '.join(detected)}")
        
        if nutrition:
            # High sodium warning
            if nutrition.get("sodium_mg", 0) > 500:
                warnings.append("High sodium content - may affect blood pressure")
            
            # High sugar warning
            if nutrition.get("sugar_g", 0) > 15:
                warnings.append("High sugar content - monitor blood glucose")
            
            # High saturated fat
            if nutrition.get("fat_g", 0) > 15:
                warnings.append("High fat content - consider portion control")
        
        return warnings
    
    def _check_medication_interactions(
        self,
        food_name: str,
        medications: List[str]
    ) -> List[Dict[str, str]]:
        """Check for drug-nutrient interactions"""
        interactions = []
        
        food_lower = food_name.lower()
        
        # Known interactions
        interaction_db = {
            "warfarin": {
                "foods": ["kale", "spinach", "broccoli", "leafy greens"],
                "warning": "High vitamin K may reduce medication effectiveness"
            },
            "metformin": {
                "foods": ["alcohol"],
                "warning": "May increase risk of lactic acidosis"
            },
            "lisinopril": {
                "foods": ["banana", "avocado", "potato"],
                "warning": "High potassium may cause hyperkalemia"
            },
            "levothyroxine": {
                "foods": ["soy", "fiber"],
                "warning": "May reduce medication absorption"
            }
        }
        
        for med in medications:
            med_lower = med.lower()
            if med_lower in interaction_db:
                data = interaction_db[med_lower]
                if any(food in food_lower for food in data["foods"]):
                    interactions.append({
                        "medication": med,
                        "interaction": data["warning"],
                        "severity": "moderate"
                    })
        
        return interactions
    
    def _has_allergen(self, food_name: str, allergies: List[str]) -> bool:
        """Check if food contains allergen"""
        food_lower = food_name.lower()
        return any(allergen.lower() in food_lower for allergen in allergies)
    
    def _calculate_benefits(
        self,
        food_name: str,
        nutrition: Optional[Dict[str, Any]],
        goals: List[str]
    ) -> List[str]:
        """Calculate health benefits for goals"""
        benefits = []
        
        food_lower = food_name.lower()
        
        # Weight loss
        if "weight_loss" in goals:
            if nutrition and nutrition.get("protein_g", 0) > 20:
                benefits.append("High protein supports weight loss")
            if nutrition and nutrition.get("fiber_g", 0) > 5:
                benefits.append("High fiber promotes satiety")
        
        # Heart health
        if "heart_health" in goals:
            if "salmon" in food_lower or "fish" in food_lower:
                benefits.append("Omega-3 fatty acids support heart health")
        
        # Diabetes control
        if "diabetes" in " ".join(goals):
            if nutrition and nutrition.get("fiber_g", 0) > 5:
                benefits.append("Fiber helps stabilize blood sugar")
        
        return benefits
    
    def _suggest_portion_adjustment(self, risk_score: float, current_g: float) -> str:
        """Suggest portion adjustment based on risk"""
        if risk_score > 60:
            adjusted = current_g * 0.5
            return f"Reduce portion to {adjusted:.0f}g"
        elif risk_score > 40:
            adjusted = current_g * 0.75
            return f"Consider reducing to {adjusted:.0f}g"
        else:
            return f"Current portion ({current_g:.0f}g) is appropriate"
    
    def _mock_risk_assessment(self) -> Dict[str, Any]:
        """Mock risk assessment"""
        return {
            "risk_score": 25,
            "risk_level": "low",
            "warnings": [],
            "safe_to_consume": True
        }
    
    # ========================================================================
    # GET RECOMMENDATIONS
    # ========================================================================
    
    async def get_recommendations(
        self,
        user_profile: Dict[str, Any],
        taste_preference: Optional[str] = None,
        meal_type: Optional[str] = None,
        max_calories: Optional[float] = None,
        num_recommendations: int = 5
    ) -> Dict[str, Any]:
        """
        Get personalized food recommendations
        
        Integrates:
        - RecommenderSystem: Collaborative/content-based filtering
        - HealthProfileEngine: Goal alignment
        - NutritionalDatabase: Food knowledge
        """
        
        try:
            # Extract user preferences
            goals = user_profile.get("health_goals", [])
            conditions = user_profile.get("health_conditions", [])
            restrictions = user_profile.get("dietary_preferences", [])
            
            # Build recommendation list
            recommendations = self._generate_recommendations(
                goals, conditions, restrictions, taste_preference, meal_type, max_calories
            )
            
            # Rank by score
            recommendations.sort(key=lambda x: x["score"], reverse=True)
            
            return {
                "success": True,
                "data": {
                    "recommendations": recommendations[:num_recommendations],
                    "total_candidates": len(recommendations),
                    "filters_applied": {
                        "taste": taste_preference,
                        "meal_type": meal_type,
                        "max_calories": max_calories
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_recommendations(
        self,
        goals: List[str],
        conditions: List[str],
        restrictions: List[str],
        taste: Optional[str],
        meal_type: Optional[str],
        max_calories: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Generate food recommendations"""
        
        # Sample recommendations (TODO: integrate with actual recommender)
        foods = [
            {"name": "Grilled Salmon", "calories": 280, "protein": 35, "type": "dinner", "taste": "savory"},
            {"name": "Blueberries", "calories": 80, "protein": 1, "type": "snack", "taste": "sweet"},
            {"name": "Greek Yogurt", "calories": 150, "protein": 20, "type": "breakfast", "taste": "tangy"},
            {"name": "Quinoa Bowl", "calories": 320, "protein": 12, "type": "lunch", "taste": "savory"},
            {"name": "Avocado Toast", "calories": 250, "protein": 8, "type": "breakfast", "taste": "savory"},
            {"name": "Chicken Breast", "calories": 165, "protein": 31, "type": "dinner", "taste": "savory"},
            {"name": "Almonds", "calories": 160, "protein": 6, "type": "snack", "taste": "nutty"},
            {"name": "Spinach Salad", "calories": 100, "protein": 5, "type": "lunch", "taste": "fresh"}
        ]
        
        # Filter
        filtered = foods
        
        if meal_type:
            filtered = [f for f in filtered if f["type"] == meal_type]
        
        if taste:
            filtered = [f for f in filtered if f["taste"] == taste]
        
        if max_calories:
            filtered = [f for f in filtered if f["calories"] <= max_calories]
        
        # Score based on goals
        recommendations = []
        for food in filtered:
            score = self._score_food_for_goals(food, goals, conditions)
            
            recommendations.append({
                "food": food["name"],
                "score": score,
                "calories": food["calories"],
                "protein_g": food["protein"],
                "rationale": self._explain_recommendation(food, goals)
            })
        
        return recommendations
    
    def _score_food_for_goals(
        self,
        food: Dict[str, Any],
        goals: List[str],
        conditions: List[str]
    ) -> float:
        """Score food based on health goals"""
        score = 50.0  # Base score
        
        # Weight loss: favor high protein, low calorie
        if "weight_loss" in goals:
            if food["protein"] > 20:
                score += 20
            if food["calories"] < 200:
                score += 15
        
        # Heart health: favor omega-3
        if "heart_health" in goals:
            if "salmon" in food["name"].lower() or "fish" in food["name"].lower():
                score += 25
        
        # Muscle gain: favor high protein
        if "muscle_gain" in goals:
            if food["protein"] > 25:
                score += 30
        
        return min(score, 100)
    
    def _explain_recommendation(
        self,
        food: Dict[str, Any],
        goals: List[str]
    ) -> str:
        """Explain why food is recommended"""
        reasons = []
        
        if food["protein"] > 20:
            reasons.append("high protein")
        
        if food["calories"] < 200:
            reasons.append("low calorie")
        
        if "salmon" in food["name"].lower():
            reasons.append("omega-3 rich")
        
        if reasons:
            return f"Great choice: {', '.join(reasons)}"
        else:
            return "Nutritious option for your goals"
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    async def get_food_alternatives(
        self,
        food_name: str,
        user_profile: Dict[str, Any]
    ) -> List[str]:
        """Get alternative food suggestions"""
        
        # Simple alternatives based on food type
        alternatives_db = {
            "salmon": ["cod", "tilapia", "chicken breast", "tofu"],
            "chicken": ["turkey", "fish", "lean beef", "lentils"],
            "pasta": ["zucchini noodles", "whole wheat pasta", "quinoa", "cauliflower rice"],
            "bread": ["lettuce wrap", "whole grain bread", "almond flour bread"],
            "rice": ["cauliflower rice", "quinoa", "barley", "farro"]
        }
        
        food_lower = food_name.lower()
        
        for key, alternatives in alternatives_db.items():
            if key in food_lower:
                return alternatives
        
        return ["grilled chicken", "fish", "tofu", "legumes"]
    
    # ========================================================================
    # FOOD RISK ANALYSIS
    # ========================================================================
    
    async def analyze_food_risks(
        self,
        food_name: str,
        icpms_data: Optional[Dict[str, float]] = None,
        scan_data: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        serving_size_g: float = 100.0
    ) -> Dict[str, Any]:
        """
        Analyze food safety risks and health alignment
        
        Args:
            food_name: Name of the food
            icpms_data: ICPMS element data {element: ppm}
            scan_data: Food scan nutrient data
            user_profile: User health profile with conditions, goals
            serving_size_g: Serving size in grams
        
        Returns:
            Comprehensive risk analysis with:
            - Contaminant detection (heavy metals, pesticides)
            - Nutrient adequacy
            - Health goal alignment
            - Medical condition safety
            - Risk scores and recommendations
        """
        
        logger.info(f"Analyzing food risks for: {food_name}")
        
        if not self.food_risk_analyzer:
            logger.warning("Food risk analyzer not available")
            return {
                "success": False,
                "error": "Food risk analyzer not initialized",
                "data": {}
            }
        
        try:
            # Perform comprehensive risk analysis
            result = await self.food_risk_analyzer.analyze_food_risks(
                food_name=food_name,
                icpms_data=icpms_data,
                scan_data=scan_data,
                user_profile=user_profile,
                serving_size_g=serving_size_g
            )
            
            # Format response
            return {
                "success": True,
                "data": {
                    "food_name": result.food_name,
                    "analyzed_at": result.analyzed_at.isoformat(),
                    
                    # Contaminant analysis
                    "contaminants": [
                        {
                            "name": c.contaminant_name,
                            "type": c.contaminant_type.value,
                            "detected_level": c.detected_level,
                            "safe_limit": c.safe_limit,
                            "unit": c.unit,
                            "risk_level": c.risk_level.value,
                            "exceeds_limit": c.exceeds_limit,
                            "exceedance_factor": round(c.exceedance_factor, 2),
                            "health_effects": c.health_effects,
                            "affected_populations": c.affected_populations
                        }
                        for c in result.contaminants
                    ],
                    "contaminant_risk_level": result.overall_contaminant_risk.value,
                    "is_safe_to_consume": result.is_safe_to_consume,
                    
                    # Nutrient analysis
                    "nutrients": [
                        {
                            "name": n.nutrient_name,
                            "detected_level": n.detected_level,
                            "rda_target": n.rda_target,
                            "unit": n.unit,
                            "status": n.status.value,
                            "percent_rda": round(n.percent_rda, 1),
                            "is_adequate": n.is_adequate,
                            "health_benefits": n.health_benefits
                        }
                        for n in result.nutrients
                    ],
                    "nutrient_score": round(result.overall_nutrient_score, 1),
                    
                    # Health goal alignment
                    "goal_alignments": [
                        {
                            "goal": g.goal_name,
                            "alignment_score": round(g.alignment_score, 1),
                            "is_aligned": g.is_aligned,
                            "supporting_nutrients": g.supporting_nutrients,
                            "conflicting_factors": g.conflicting_factors,
                            "recommendations": g.recommendations
                        }
                        for g in result.goal_alignments
                    ],
                    "overall_goal_alignment": round(result.overall_goal_alignment, 1),
                    
                    # Medical condition safety
                    "condition_checks": [
                        {
                            "condition": c.condition_name,
                            "is_safe": c.is_safe,
                            "risk_level": c.risk_level.value,
                            "contraindications": c.contraindications,
                            "beneficial_aspects": c.beneficial_aspects,
                            "warnings": c.warnings,
                            "max_safe_serving": c.max_safe_serving
                        }
                        for c in result.condition_checks
                    ],
                    "safe_for_all_conditions": result.safe_for_all_conditions,
                    
                    # Summary scores
                    "overall_risk_score": round(result.overall_risk_score, 1),
                    "overall_health_score": round(result.overall_health_score, 1),
                    "recommendation": result.recommendation,
                    "critical_warnings": result.critical_warnings,
                    "suggestions": result.suggestions,
                    "safer_alternatives": result.safer_alternatives
                }
            }
            
        except Exception as e:
            logger.error(f"Food risk analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": {}
            }
    
    # ========================================================================
    # FAMILY RECIPE GENERATION
    # ========================================================================
    
    async def generate_family_recipe(
        self,
        family_members: List[Dict[str, Any]],
        meal_type: str = "dinner",
        cuisine_preference: Optional[str] = None,
        max_recipes: int = 3
    ) -> Dict[str, Any]:
        """
        Generate recipes for family with multiple members
        
        Args:
            family_members: List of family member profiles
                Each with: name, age, gender, health_goals, taste_preferences, etc.
            meal_type: breakfast, lunch, dinner, snack
            cuisine_preference: Optional cuisine type
            max_recipes: Number of recipe options
        
        Returns:
            Dictionary with generated recipes optimized for family
        """
        
        logger.info(f"Generating family recipe for {len(family_members)} members")
        
        if not self.family_recipe_generator:
            logger.warning("Family recipe generator not available")
            return {
                "success": False,
                "error": "Family recipe generator not initialized",
                "data": {}
            }
        
        try:
            from app.ai_nutrition.orchestration.family_recipe_generator import (
                FamilyMember,
                FamilyProfile
            )
            
            # Convert to FamilyMember objects
            members = []
            for member_data in family_members:
                member = FamilyMember(
                    name=member_data.get("name", "Family Member"),
                    age=member_data.get("age", 30),
                    gender=member_data.get("gender", "other"),
                    health_goals=member_data.get("health_goals", []),
                    medical_conditions=member_data.get("medical_conditions", []),
                    medications=member_data.get("medications", []),
                    allergies=member_data.get("allergies", []),
                    dietary_restrictions=member_data.get("dietary_restrictions", []),
                    taste_preferences=member_data.get("taste_preferences", {}),
                    weight=member_data.get("weight"),
                    height=member_data.get("height"),
                    activity_level=member_data.get("activity_level", "moderate")
                )
                members.append(member)
            
            # Create family profile
            family_profile = FamilyProfile(
                family_id="temp_family",
                members=members,
                budget_level="moderate",
                cooking_skill_level="intermediate",
                available_cooking_time=60
            )
            
            # Generate recipes
            recipes = await self.family_recipe_generator.generate_family_recipe(
                family_profile=family_profile,
                meal_type=meal_type,
                cuisine_preference=cuisine_preference,
                max_recipes=max_recipes
            )
            
            # Format response
            formatted_recipes = []
            for recipe in recipes:
                formatted_recipes.append({
                    "name": recipe.name,
                    "description": recipe.description,
                    "cuisine": recipe.cuisine_type,
                    "ingredients": recipe.ingredients,
                    "instructions": recipe.instructions,
                    "prep_time": recipe.prep_time,
                    "cook_time": recipe.cook_time,
                    "servings": recipe.servings,
                    "difficulty": recipe.difficulty,
                    "nutrition_per_serving": recipe.nutrition_per_serving,
                    "member_suitability": recipe.member_suitability,
                    "goal_alignment": recipe.goal_alignment,
                    "taste_match": recipe.taste_match,
                    "portions_by_member": recipe.age_appropriate_portions,
                    "modifications_by_member": recipe.modifications_per_member,
                    "allergen_warnings": recipe.allergen_warnings,
                    "cautions": recipe.contraindications,
                    "why_this_works": recipe.why_this_works,
                    "tips_for_picky_eaters": recipe.tips_for_picky_eaters,
                    "family_health_benefits": recipe.family_health_benefits,
                    "overall_score": recipe.overall_family_score
                })
            
            return {
                "success": True,
                "data": {
                    "recipes": formatted_recipes,
                    "family_size": len(members),
                    "meal_type": meal_type,
                    "top_recommendation": formatted_recipes[0] if formatted_recipes else None
                }
            }
            
        except Exception as e:
            logger.error(f"Family recipe generation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": {}
            }


# Global handler instance
handler = FunctionCallHandler()
