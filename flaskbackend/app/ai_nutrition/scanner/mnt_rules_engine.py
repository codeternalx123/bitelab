"""
Medical Nutrition Therapy (MNT) Rules Engine

This module acts as the "Digital Dietitian" - translating disease molecular profiles
and health guidelines into actionable food filtering rules for the API layer.

Core Functions:
1. Convert disease profiles → API filters (e.g., "low sodium" → sodium_mg < 140)
2. Parse health guideline text → structured nutrient rules (NLP)
3. Combine multiple conditions → priority-weighted recommendations
4. Handle conflicts → safety-first resolution
5. Generate meal planning constraints

Integration Flow:
User Disease → DiseaseProfile (internal) → Rules → API Filters → Food Search → Results

Example:
  Disease: Type 2 Diabetes
  Profile: Limit carbs, increase fiber, manage sodium
  Rules: carbs_g < 45, fiber_g > 5, sodium_mg < 2300
  API Filter: {max_carbs: 45, min_fiber: 5, max_sodium: 2300}
  Food Search: Apply filters to Edamam API

Author: Atomic AI System
Date: November 7, 2025
Version: 1.0.0 - Phase 2
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import logging

# Import from our existing modules
from atomic_molecular_profiler import (
    DiseaseCondition,
    HealthGoal,
    NutrientPriority
)

from multi_condition_optimizer import (
    DiseaseMolecularProfile,
    GoalMolecularProfile,
    DiseaseProfileDatabase,
    GoalProfileDatabase
)

from mnt_api_integration import (
    FoodItem,
    DiseaseGuideline,
    NutrientData
)

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA MODELS
# ============================================================================

class FilterOperator(Enum):
    """Comparison operators for nutrient filters"""
    LESS_THAN = "lt"  # <
    LESS_THAN_EQUAL = "lte"  # <=
    GREATER_THAN = "gt"  # >
    GREATER_THAN_EQUAL = "gte"  # >=
    EQUAL = "eq"  # ==
    RANGE = "range"  # between min and max
    AVOID = "avoid"  # contraindicated


class RulePriority(Enum):
    """Priority levels for nutrition rules"""
    CRITICAL = 5  # Life-threatening if violated (e.g., allergy, drug interaction)
    HIGH = 4  # Disease management essential (e.g., diabetes carbs)
    MODERATE = 3  # Important but flexible (e.g., weight loss calories)
    LOW = 2  # Optimization (e.g., performance goals)
    OPTIONAL = 1  # Nice to have (e.g., food preferences)


@dataclass
class NutrientRule:
    """Single rule for a specific nutrient"""
    nutrient_id: str  # Standard nutrient ID (e.g., "sodium_mg")
    operator: FilterOperator
    value: float  # Target value
    value_max: Optional[float] = None  # For range operator
    
    priority: RulePriority = RulePriority.MODERATE
    reason: str = ""  # Why this rule exists
    source: str = "molecular_profile"  # molecular_profile, api_guideline, user_preference
    
    # For complex rules
    per_serving: bool = True  # vs per 100g
    per_bodyweight_kg: Optional[float] = None  # e.g., protein 2g/kg
    
    def evaluate(self, nutrient_value: float) -> bool:
        """Check if a nutrient value satisfies this rule"""
        if self.operator == FilterOperator.LESS_THAN:
            return nutrient_value < self.value
        elif self.operator == FilterOperator.LESS_THAN_EQUAL:
            return nutrient_value <= self.value
        elif self.operator == FilterOperator.GREATER_THAN:
            return nutrient_value > self.value
        elif self.operator == FilterOperator.GREATER_THAN_EQUAL:
            return nutrient_value >= self.value
        elif self.operator == FilterOperator.EQUAL:
            return abs(nutrient_value - self.value) < 0.01  # Float comparison
        elif self.operator == FilterOperator.RANGE:
            return self.value <= nutrient_value <= (self.value_max or float('inf'))
        elif self.operator == FilterOperator.AVOID:
            return nutrient_value < 0.01  # Essentially zero
        return False
    
    def get_api_filter(self) -> Dict[str, Any]:
        """Convert rule to API filter format"""
        filter_dict = {
            "nutrient": self.nutrient_id,
            "operator": self.operator.value
        }
        
        if self.operator == FilterOperator.RANGE:
            filter_dict["min"] = self.value
            filter_dict["max"] = self.value_max
        else:
            filter_dict["value"] = self.value
        
        return filter_dict


@dataclass
class FoodRecommendationRule:
    """Complete rule set for a health condition"""
    condition_name: str
    condition_type: str  # "disease" or "goal"
    
    # Nutrient rules
    nutrient_rules: List[NutrientRule] = field(default_factory=list)
    
    # Food category rules
    recommended_foods: List[str] = field(default_factory=list)
    foods_to_avoid: List[str] = field(default_factory=list)
    
    # Diet labels to prefer
    preferred_diet_labels: List[str] = field(default_factory=list)  # vegan, gluten-free, etc.
    preferred_health_labels: List[str] = field(default_factory=list)  # low-sodium, high-fiber, etc.
    
    # Meal planning constraints
    meal_frequency: Optional[str] = None
    meal_timing: Optional[str] = None
    
    # Priority for multi-condition optimization
    overall_priority: RulePriority = RulePriority.MODERATE
    severity_multiplier: float = 1.0
    
    def get_critical_rules(self) -> List[NutrientRule]:
        """Get only critical/high priority rules"""
        return [r for r in self.nutrient_rules if r.priority in [RulePriority.CRITICAL, RulePriority.HIGH]]
    
    def get_api_filters(self) -> Dict[str, Any]:
        """Generate API-compatible filter dictionary"""
        filters = {
            "nutrient_filters": [rule.get_api_filter() for rule in self.nutrient_rules],
            "diet_labels": self.preferred_diet_labels,
            "health_labels": self.preferred_health_labels,
            "exclude_foods": self.foods_to_avoid
        }
        return filters


# ============================================================================
# RULES GENERATOR - CONVERTS PROFILES TO RULES
# ============================================================================

class MNTRulesGenerator:
    """
    Generates food filtering rules from disease molecular profiles
    
    This is the core "translation layer" that converts our evidence-based
    molecular profiles into actionable API filters.
    
    Example Flow:
        Disease: Hypertension
        Profile: sodium restriction 2.5x priority, potassium 2.3x beneficial
        Rules Generated:
            - sodium_mg <= 2300 (HIGH priority)
            - potassium_mg >= 3500 (MODERATE priority)
            - recommended_foods: ["banana", "spinach", "avocado"]
            - foods_to_avoid: ["processed meats", "canned soups"]
    """
    
    # Nutrient ID mapping (molecular profile → API)
    NUTRIENT_MAPPING = {
        "sodium": "sodium_mg",
        "potassium": "potassium_mg",
        "calcium": "calcium_mg",
        "magnesium": "magnesium_mg",
        "iron": "iron_mg",
        "zinc": "zinc_mg",
        "vitamin_d": "vitamin_d_iu",
        "vitamin_c": "vitamin_c_mg",
        "vitamin_b12": "vitamin_b12_mcg",
        "folate": "folate_mcg",
        "omega_3": "omega_3_g",
        "fiber": "fiber_g",
        "protein": "protein_g",
        "carbohydrates": "carbs_g",
        "sugar": "sugar_g",
        "fat": "fat_g",
        "saturated_fat": "saturated_fat_g",
        "cholesterol": "cholesterol_mg"
    }
    
    # Standard daily limits (from USDA/FDA guidelines)
    DAILY_LIMITS = {
        "sodium_mg": {"max": 2300, "optimal": 1500},
        "sugar_g": {"max": 50, "optimal": 25},
        "saturated_fat_g": {"max": 22, "optimal": 13},
        "cholesterol_mg": {"max": 300, "optimal": 200},
        "fiber_g": {"min": 25, "optimal": 35},
        "protein_g": {"min": 50, "optimal_per_kg": 0.8},
        "calcium_mg": {"min": 1000, "optimal": 1300},
        "iron_mg": {"min": 8, "optimal": 18},
        "vitamin_d_iu": {"min": 600, "optimal": 1000}
    }
    
    def __init__(self):
        self.disease_db = DiseaseProfileDatabase()
        self.goal_db = GoalProfileDatabase()
        logger.info("MNT Rules Generator initialized with 50 diseases + 55 goals")
    
    def generate_disease_rules(
        self,
        disease: DiseaseCondition,
        bodyweight_kg: Optional[float] = None
    ) -> FoodRecommendationRule:
        """
        Generate food filtering rules from a disease profile
        
        Args:
            disease: DiseaseCondition enum
            bodyweight_kg: User's body weight for per-kg calculations
        
        Returns:
            FoodRecommendationRule object
        """
        # Get molecular profile
        profile = self.disease_db.get_profile(disease)
        if not profile:
            logger.warning(f"No profile found for {disease.value}")
            return FoodRecommendationRule(
                condition_name=disease.value,
                condition_type="disease"
            )
        
        rules = FoodRecommendationRule(
            condition_name=disease.value,
            condition_type="disease",
            overall_priority=RulePriority.HIGH,  # Diseases are high priority
            severity_multiplier=profile.severity_multiplier
        )
        
        # Process harmful molecules → MAX limits
        for molecule, weight in profile.harmful_molecules.items():
            nutrient_id = self._map_molecule_to_nutrient(molecule)
            if not nutrient_id:
                continue
            
            # Determine max value based on weight
            max_value = self._calculate_max_value(nutrient_id, weight)
            
            priority = self._determine_priority(weight, is_harmful=True)
            
            rule = NutrientRule(
                nutrient_id=nutrient_id,
                operator=FilterOperator.LESS_THAN_EQUAL,
                value=max_value,
                priority=priority,
                reason=f"Limit {molecule} for {disease.value} management",
                source="molecular_profile"
            )
            rules.nutrient_rules.append(rule)
        
        # Process beneficial molecules → MIN targets
        for molecule, weight in profile.beneficial_molecules.items():
            nutrient_id = self._map_molecule_to_nutrient(molecule)
            if not nutrient_id:
                continue
            
            # Determine min value based on weight
            min_value = self._calculate_min_value(nutrient_id, weight, bodyweight_kg)
            
            priority = self._determine_priority(weight, is_harmful=False)
            
            rule = NutrientRule(
                nutrient_id=nutrient_id,
                operator=FilterOperator.GREATER_THAN_EQUAL,
                value=min_value,
                priority=priority,
                reason=f"Increase {molecule} for {disease.value} support",
                source="molecular_profile"
            )
            rules.nutrient_rules.append(rule)
        
        # Add specific max_values from profile (e.g., sodium < 2000mg)
        for nutrient, max_val in profile.max_values.items():
            nutrient_id = self._map_molecule_to_nutrient(nutrient)
            if not nutrient_id:
                continue
            
            rule = NutrientRule(
                nutrient_id=nutrient_id,
                operator=FilterOperator.LESS_THAN_EQUAL,
                value=max_val,
                priority=RulePriority.HIGH,
                reason=f"Disease-specific limit for {disease.value}",
                source="molecular_profile"
            )
            rules.nutrient_rules.append(rule)
        
        # Add specific min_values from profile
        for nutrient, min_val in profile.min_values.items():
            nutrient_id = self._map_molecule_to_nutrient(nutrient)
            if not nutrient_id:
                continue
            
            rule = NutrientRule(
                nutrient_id=nutrient_id,
                operator=FilterOperator.GREATER_THAN_EQUAL,
                value=min_val,
                priority=RulePriority.HIGH,
                reason=f"Disease-specific requirement for {disease.value}",
                source="molecular_profile"
            )
            rules.nutrient_rules.append(rule)
        
        logger.info(f"Generated {len(rules.nutrient_rules)} rules for {disease.value}")
        return rules
    
    def generate_goal_rules(
        self,
        goal: HealthGoal,
        bodyweight_kg: Optional[float] = None
    ) -> FoodRecommendationRule:
        """
        Generate food filtering rules from a health goal profile
        
        Args:
            goal: HealthGoal enum
            bodyweight_kg: User's body weight
        
        Returns:
            FoodRecommendationRule object
        """
        profile = self.goal_db.get_profile(goal)
        if not profile:
            logger.warning(f"No profile found for {goal.value}")
            return FoodRecommendationRule(
                condition_name=goal.value,
                condition_type="goal"
            )
        
        rules = FoodRecommendationRule(
            condition_name=goal.value,
            condition_type="goal",
            overall_priority=RulePriority.MODERATE  # Goals are moderate priority
        )
        
        # Process target ranges for goals (more flexible than diseases)
        for nutrient, target_data in profile.target_ranges.items():
            nutrient_id = self._map_molecule_to_nutrient(nutrient)
            if not nutrient_id:
                continue
            
            # Goals use range operator (optimal range)
            rule = NutrientRule(
                nutrient_id=nutrient_id,
                operator=FilterOperator.RANGE,
                value=target_data.get("min", 0),
                value_max=target_data.get("max", float('inf')),
                priority=RulePriority.MODERATE,
                reason=f"Optimal range for {goal.value}",
                source="molecular_profile"
            )
            rules.nutrient_rules.append(rule)
        
        # Process priority molecules (similar to diseases)
        for molecule, weight in profile.priority_molecules.items():
            nutrient_id = self._map_molecule_to_nutrient(molecule)
            if not nutrient_id:
                continue
            
            min_value = self._calculate_min_value(nutrient_id, weight, bodyweight_kg)
            priority = self._determine_priority(weight, is_harmful=False)
            
            rule = NutrientRule(
                nutrient_id=nutrient_id,
                operator=FilterOperator.GREATER_THAN_EQUAL,
                value=min_value,
                priority=priority,
                reason=f"Support {goal.value}",
                source="molecular_profile"
            )
            rules.nutrient_rules.append(rule)
        
        logger.info(f"Generated {len(rules.nutrient_rules)} rules for {goal.value}")
        return rules
    
    def _map_molecule_to_nutrient(self, molecule: str) -> Optional[str]:
        """Map molecule name to API nutrient ID"""
        molecule_lower = molecule.lower().replace(" ", "_")
        return self.NUTRIENT_MAPPING.get(molecule_lower)
    
    def _calculate_max_value(self, nutrient_id: str, weight: float) -> float:
        """Calculate maximum allowed value based on weight"""
        limits = self.DAILY_LIMITS.get(nutrient_id, {})
        base_max = limits.get("max", limits.get("optimal", 100.0))
        
        # Higher weight = stricter limit (inverse relationship)
        if weight >= 2.5:  # Critical restriction
            return base_max * 0.5  # 50% of normal limit
        elif weight >= 2.0:
            return base_max * 0.65
        elif weight >= 1.5:
            return base_max * 0.8
        else:
            return base_max
    
    def _calculate_min_value(
        self,
        nutrient_id: str,
        weight: float,
        bodyweight_kg: Optional[float]
    ) -> float:
        """Calculate minimum required value based on weight"""
        limits = self.DAILY_LIMITS.get(nutrient_id, {})
        
        # Check if per-bodyweight calculation needed
        if bodyweight_kg and "optimal_per_kg" in limits:
            base_min = limits["optimal_per_kg"] * bodyweight_kg
        else:
            base_min = limits.get("min", limits.get("optimal", 10.0))
        
        # Higher weight = higher requirement (direct relationship)
        if weight >= 2.5:
            return base_min * 1.5  # 150% of normal
        elif weight >= 2.0:
            return base_min * 1.35
        elif weight >= 1.5:
            return base_min * 1.2
        else:
            return base_min
    
    def _determine_priority(self, weight: float, is_harmful: bool) -> RulePriority:
        """Determine rule priority based on molecular weight"""
        if weight >= 2.8:
            return RulePriority.CRITICAL
        elif weight >= 2.3:
            return RulePriority.HIGH
        elif weight >= 1.8:
            return RulePriority.MODERATE
        else:
            return RulePriority.LOW


# ============================================================================
# GUIDELINE TEXT PARSER - NLP FOR HEALTH GUIDELINES
# ============================================================================

class GuidelineTextParser:
    """
    Parse natural language health guidelines into structured rules
    
    Uses NLP patterns to extract nutrient restrictions from text like:
    - "Eat a low-sodium diet" → sodium_mg <= 2300
    - "Limit sugar to 25 grams per day" → sugar_g <= 25
    - "Increase fiber intake" → fiber_g >= 30
    """
    
    # Regex patterns for nutrient extraction
    LIMIT_PATTERNS = [
        r"limit\s+(?P<nutrient>\w+)\s+to\s+(?P<amount>\d+)\s*(?P<unit>\w+)?",
        r"restrict\s+(?P<nutrient>\w+)\s+(?:intake)?",
        r"low[- ](?P<nutrient>\w+)\s+diet",
        r"reduce\s+(?P<nutrient>\w+)\s+(?:intake|consumption)?",
        r"avoid\s+(?:high[- ])?(?P<nutrient>\w+)\s+foods"
    ]
    
    INCREASE_PATTERNS = [
        r"increase\s+(?P<nutrient>\w+)\s+(?:intake)?\s*(?:to\s+(?P<amount>\d+)\s*(?P<unit>\w+)?)?",
        r"eat\s+more\s+(?P<nutrient>\w+)",
        r"high[- ](?P<nutrient>\w+)\s+diet",
        r"boost\s+(?P<nutrient>\w+)",
        r"(?P<amount>\d+)\s*(?P<unit>\w+)?\s+of\s+(?P<nutrient>\w+)"
    ]
    
    NUTRIENT_KEYWORDS = {
        "sodium": "sodium_mg",
        "salt": "sodium_mg",
        "potassium": "potassium_mg",
        "sugar": "sugar_g",
        "fiber": "fiber_g",
        "protein": "protein_g",
        "carbohydrate": "carbs_g",
        "carbs": "carbs_g",
        "fat": "fat_g",
        "cholesterol": "cholesterol_mg",
        "calcium": "calcium_mg",
        "iron": "iron_mg"
    }
    
    def parse_guideline(self, guideline: DiseaseGuideline) -> List[NutrientRule]:
        """
        Parse guideline text into structured nutrient rules
        
        Args:
            guideline: DiseaseGuideline object with text
        
        Returns:
            List of NutrientRule objects
        """
        text = guideline.guideline_text.lower()
        rules = []
        
        # Extract limit rules
        for pattern in self.LIMIT_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                rule = self._create_limit_rule(match, guideline.disease_name)
                if rule:
                    rules.append(rule)
        
        # Extract increase rules
        for pattern in self.INCREASE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                rule = self._create_increase_rule(match, guideline.disease_name)
                if rule:
                    rules.append(rule)
        
        logger.info(f"Parsed {len(rules)} rules from guideline text for {guideline.disease_name}")
        return rules
    
    def _create_limit_rule(self, match: re.Match, disease: str) -> Optional[NutrientRule]:
        """Create a limit rule from regex match"""
        nutrient_text = match.group("nutrient")
        nutrient_id = self.NUTRIENT_KEYWORDS.get(nutrient_text.lower())
        
        if not nutrient_id:
            return None
        
        # Try to extract specific amount
        try:
            amount = float(match.group("amount"))
        except (IndexError, ValueError, TypeError):
            # Use default limits
            amount = MNTRulesGenerator.DAILY_LIMITS.get(nutrient_id, {}).get("optimal", 100.0)
        
        return NutrientRule(
            nutrient_id=nutrient_id,
            operator=FilterOperator.LESS_THAN_EQUAL,
            value=amount,
            priority=RulePriority.HIGH,
            reason=f"Guideline recommendation for {disease}",
            source="api_guideline"
        )
    
    def _create_increase_rule(self, match: re.Match, disease: str) -> Optional[NutrientRule]:
        """Create an increase rule from regex match"""
        nutrient_text = match.group("nutrient")
        nutrient_id = self.NUTRIENT_KEYWORDS.get(nutrient_text.lower())
        
        if not nutrient_id:
            return None
        
        # Try to extract specific amount
        try:
            amount = float(match.group("amount"))
        except (IndexError, ValueError, TypeError):
            # Use default minimums
            amount = MNTRulesGenerator.DAILY_LIMITS.get(nutrient_id, {}).get("min", 10.0)
        
        return NutrientRule(
            nutrient_id=nutrient_id,
            operator=FilterOperator.GREATER_THAN_EQUAL,
            value=amount,
            priority=RulePriority.MODERATE,
            reason=f"Guideline recommendation for {disease}",
            source="api_guideline"
        )


# ============================================================================
# MULTI-CONDITION RULES COMBINER
# ============================================================================

class MultiConditionRulesCombiner:
    """
    Combines rules from multiple conditions with conflict resolution
    
    Example:
        Conditions: Hypertension + Diabetes + Weight Loss
        Rules:
            - Hypertension: sodium <= 1500mg (HIGH)
            - Diabetes: carbs <= 45g (HIGH)
            - Weight Loss: calories <= 1500kcal (MODERATE)
        
        Combined: All rules applied, prioritized by severity
    """
    
    def combine_rules(
        self,
        rule_sets: List[FoodRecommendationRule]
    ) -> FoodRecommendationRule:
        """
        Combine multiple rule sets with conflict resolution
        
        Args:
            rule_sets: List of FoodRecommendationRule objects
        
        Returns:
            Combined FoodRecommendationRule
        """
        if not rule_sets:
            return FoodRecommendationRule(
                condition_name="no_conditions",
                condition_type="combined"
            )
        
        combined = FoodRecommendationRule(
            condition_name=f"combined_{len(rule_sets)}_conditions",
            condition_type="combined",
            overall_priority=RulePriority.HIGH
        )
        
        # Group rules by nutrient
        nutrient_groups: Dict[str, List[NutrientRule]] = {}
        for rule_set in rule_sets:
            for rule in rule_set.nutrient_rules:
                if rule.nutrient_id not in nutrient_groups:
                    nutrient_groups[rule.nutrient_id] = []
                nutrient_groups[rule.nutrient_id].append(rule)
        
        # Resolve conflicts for each nutrient
        for nutrient_id, rules in nutrient_groups.items():
            resolved_rule = self._resolve_conflicting_rules(rules)
            if resolved_rule:
                combined.nutrient_rules.append(resolved_rule)
        
        # Combine food lists (union of recommendations, intersection of avoidances)
        for rule_set in rule_sets:
            combined.recommended_foods.extend(rule_set.recommended_foods)
            combined.foods_to_avoid.extend(rule_set.foods_to_avoid)
        
        # Deduplicate
        combined.recommended_foods = list(set(combined.recommended_foods))
        combined.foods_to_avoid = list(set(combined.foods_to_avoid))
        
        logger.info(f"Combined {len(rule_sets)} rule sets into {len(combined.nutrient_rules)} rules")
        return combined
    
    def _resolve_conflicting_rules(self, rules: List[NutrientRule]) -> Optional[NutrientRule]:
        """
        Resolve conflicts when multiple rules target same nutrient
        
        Strategy:
        1. CRITICAL rules always win
        2. For MAX limits: Use most restrictive (lowest value)
        3. For MIN requirements: Use highest value
        4. Preserve highest priority
        """
        if not rules:
            return None
        
        if len(rules) == 1:
            return rules[0]
        
        # Sort by priority (highest first)
        rules_sorted = sorted(rules, key=lambda r: r.priority.value, reverse=True)
        
        # Check for CRITICAL rules
        critical_rules = [r for r in rules_sorted if r.priority == RulePriority.CRITICAL]
        if critical_rules:
            return critical_rules[0]  # Use first critical rule
        
        # Separate limit vs increase rules
        limit_rules = [r for r in rules_sorted if r.operator in [
            FilterOperator.LESS_THAN,
            FilterOperator.LESS_THAN_EQUAL
        ]]
        increase_rules = [r for r in rules_sorted if r.operator in [
            FilterOperator.GREATER_THAN,
            FilterOperator.GREATER_THAN_EQUAL
        ]]
        
        # For limits: Use most restrictive (lowest value)
        if limit_rules:
            most_restrictive = min(limit_rules, key=lambda r: r.value)
            return most_restrictive
        
        # For increases: Use highest requirement
        if increase_rules:
            highest_requirement = max(increase_rules, key=lambda r: r.value)
            return highest_requirement
        
        # Default: Return highest priority rule
        return rules_sorted[0]


# ============================================================================
# FOOD SCORER - EVALUATES FOODS AGAINST RULES
# ============================================================================

class FoodScorer:
    """
    Scores food items against recommendation rules
    
    Generates a score (0-100) indicating how well a food matches
    the nutritional requirements for a condition.
    """
    
    def score_food(
        self,
        food: FoodItem,
        rules: FoodRecommendationRule
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score a food item against recommendation rules
        
        Args:
            food: FoodItem to evaluate
            rules: FoodRecommendationRule to check against
        
        Returns:
            Tuple of (score, details_dict)
            score: 0-100, higher is better
            details: Breakdown of score calculation
        """
        score = 100.0
        details = {
            "passed_rules": [],
            "failed_rules": [],
            "warnings": []
        }
        
        # Evaluate each nutrient rule
        for rule in rules.nutrient_rules:
            nutrient = food.get_nutrient(rule.nutrient_id)
            
            if not nutrient:
                # Missing nutrient data
                penalty = rule.priority.value * 5  # Higher priority = bigger penalty
                score -= penalty
                details["warnings"].append(f"Missing data for {rule.nutrient_id}")
                continue
            
            # Check if nutrient value passes rule
            passes = rule.evaluate(nutrient.quantity)
            
            if passes:
                details["passed_rules"].append({
                    "nutrient": rule.nutrient_id,
                    "value": nutrient.quantity,
                    "rule": f"{rule.operator.value} {rule.value}"
                })
            else:
                # Calculate penalty based on priority
                penalty = rule.priority.value * 10
                score -= penalty
                details["failed_rules"].append({
                    "nutrient": rule.nutrient_id,
                    "value": nutrient.quantity,
                    "rule": f"{rule.operator.value} {rule.value}",
                    "penalty": penalty
                })
        
        # Bonus for diet/health labels match
        matching_diet_labels = set(food.diet_labels) & set(rules.preferred_diet_labels)
        score += len(matching_diet_labels) * 5
        
        matching_health_labels = set(food.health_labels) & set(rules.preferred_health_labels)
        score += len(matching_health_labels) * 5
        
        # Clamp score to 0-100
        score = max(0.0, min(100.0, score))
        
        details["final_score"] = score
        details["recommendation"] = self._get_recommendation(score)
        
        return score, details
    
    def _get_recommendation(self, score: float) -> str:
        """Convert score to recommendation text"""
        if score >= 90:
            return "Excellent Choice"
        elif score >= 75:
            return "Good Choice"
        elif score >= 60:
            return "Acceptable"
        elif score >= 40:
            return "Use Moderately"
        else:
            return "Not Recommended"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example usage of MNT Rules Engine"""
    
    print("\n=== MNT Rules Engine Example ===\n")
    
    # Initialize
    generator = MNTRulesGenerator()
    combiner = MultiConditionRulesCombiner()
    scorer = FoodScorer()
    
    # Example 1: Generate rules for Type 2 Diabetes
    print("Example 1: Type 2 Diabetes Rules")
    diabetes_rules = generator.generate_disease_rules(
        DiseaseCondition.TYPE_2_DIABETES,
        bodyweight_kg=75.0
    )
    print(f"  Generated {len(diabetes_rules.nutrient_rules)} rules")
    for rule in diabetes_rules.nutrient_rules[:3]:
        print(f"    - {rule.nutrient_id} {rule.operator.value} {rule.value} ({rule.priority.value})")
    
    # Example 2: Generate rules for Weight Loss goal
    print("\nExample 2: Weight Loss Goal Rules")
    weightloss_rules = generator.generate_goal_rules(
        HealthGoal.WEIGHT_LOSS,
        bodyweight_kg=75.0
    )
    print(f"  Generated {len(weightloss_rules.nutrient_rules)} rules")
    
    # Example 3: Combine multiple conditions
    print("\nExample 3: Combined Rules (Diabetes + Weight Loss)")
    combined = combiner.combine_rules([diabetes_rules, weightloss_rules])
    print(f"  Combined rules: {len(combined.nutrient_rules)}")
    print(f"  Foods to avoid: {len(combined.foods_to_avoid)}")
    
    print("\n" + "=" * 60)
    print("MNT Rules Engine - 800+ LOC")
    print("Next: Local Food Matching System")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
