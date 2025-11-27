"""
Family Recipe Generator
=======================

Multi-member recipe generation that accommodates:
- Different ages (children, adults, seniors)
- Multiple health goals per person
- Taste preferences and restrictions
- Dietary needs across family members

Generates recipes that work for the entire family while
optimizing for each member's specific health goals.

Author: Wellomex AI Team
Date: November 2025
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import asyncio

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class FamilyMember:
    """Individual family member profile"""
    name: str
    age: int
    gender: str
    health_goals: List[str] = field(default_factory=list)
    medical_conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    dietary_restrictions: List[str] = field(default_factory=list)
    taste_preferences: Dict[str, Any] = field(default_factory=dict)
    # taste_preferences format:
    # {
    #   "likes": ["sweet", "spicy", "savory"],
    #   "dislikes": ["bitter", "sour"],
    #   "favorite_cuisines": ["italian", "asian"],
    #   "favorite_ingredients": ["chicken", "pasta", "cheese"],
    #   "texture_preferences": ["crispy", "creamy"]
    # }
    
    # Metabolic data
    weight: Optional[float] = None  # kg
    height: Optional[float] = None  # cm
    activity_level: str = "moderate"  # sedentary, light, moderate, active, very_active
    
    # Special considerations
    is_child: bool = False
    is_pregnant: bool = False
    is_elderly: bool = False
    
    def __post_init__(self):
        """Auto-detect life stage"""
        if self.age < 13:
            self.is_child = True
        elif self.age >= 65:
            self.is_elderly = True


@dataclass
class FamilyProfile:
    """Complete family profile"""
    family_id: str
    members: List[FamilyMember]
    household_dietary_restrictions: List[str] = field(default_factory=list)
    budget_level: str = "moderate"  # low, moderate, high
    cooking_skill_level: str = "intermediate"  # beginner, intermediate, advanced
    available_cooking_time: int = 60  # minutes
    kitchen_equipment: List[str] = field(default_factory=list)
    
    def get_member_by_name(self, name: str) -> Optional[FamilyMember]:
        """Get family member by name"""
        return next((m for m in self.members if m.name.lower() == name.lower()), None)
    
    def get_all_allergies(self) -> List[str]:
        """Get all unique allergies across family"""
        allergies = set()
        for member in self.members:
            allergies.update(member.allergies)
        return list(allergies)
    
    def get_all_dietary_restrictions(self) -> List[str]:
        """Get all dietary restrictions"""
        restrictions = set(self.household_dietary_restrictions)
        for member in self.members:
            restrictions.update(member.dietary_restrictions)
        return list(restrictions)
    
    def get_all_health_goals(self) -> Dict[str, List[str]]:
        """Get all health goals by member"""
        return {member.name: member.health_goals for member in self.members}
    
    def get_age_groups(self) -> Dict[str, List[str]]:
        """Categorize family by age groups"""
        groups = {
            "children": [],
            "teens": [],
            "adults": [],
            "seniors": []
        }
        
        for member in self.members:
            if member.age < 13:
                groups["children"].append(member.name)
            elif 13 <= member.age < 20:
                groups["teens"].append(member.name)
            elif 20 <= member.age < 65:
                groups["adults"].append(member.name)
            else:
                groups["seniors"].append(member.name)
        
        return {k: v for k, v in groups.items() if v}  # Remove empty groups


@dataclass
class FamilyRecipe:
    """Recipe optimized for family"""
    name: str
    description: str
    cuisine_type: str
    
    # Recipe details
    ingredients: List[Dict[str, Any]]  # [{"name": "chicken", "amount": "500g", "notes": "for adults"}]
    instructions: List[str]
    prep_time: int  # minutes
    cook_time: int  # minutes
    servings: int
    difficulty: str  # easy, medium, hard
    
    # Nutritional info per serving
    nutrition_per_serving: Dict[str, float]
    
    # Family optimization
    member_suitability: Dict[str, float]  # member_name -> suitability score (0-1)
    goal_alignment: Dict[str, Dict[str, float]]  # member_name -> {goal -> score}
    taste_match: Dict[str, float]  # member_name -> taste match score (0-1)
    
    # Customization options
    age_appropriate_portions: Dict[str, str]  # member_name -> portion size
    modifications_per_member: Dict[str, List[str]]  # member_name -> [modifications]
    
    # Practical info
    cost_estimate: str  # low, moderate, high
    allergen_warnings: List[str]
    contraindications: List[str]
    
    # AI insights
    why_this_works: List[str]
    tips_for_picky_eaters: List[str]
    family_health_benefits: List[str]
    
    overall_family_score: float  # 0-1, how well it works for entire family


# ============================================================================
# FAMILY RECIPE GENERATOR
# ============================================================================

class FamilyRecipeGenerator:
    """
    Generates recipes optimized for entire families
    
    Considers:
    - Age-appropriate nutrition
    - Individual health goals
    - Taste preferences and compromises
    - Dietary restrictions across members
    - Portion adjustments by age/size
    """
    
    def __init__(self, llm_client: Any = None, knowledge_graph: Any = None):
        """
        Initialize family recipe generator
        
        Args:
            llm_client: OpenAI/Anthropic client for recipe generation
            knowledge_graph: IntegratedNutritionAI for health optimization
        """
        self.llm_client = llm_client
        self.knowledge_graph = knowledge_graph
        
    async def generate_family_recipe(
        self,
        family_profile: FamilyProfile,
        meal_type: str = "dinner",  # breakfast, lunch, dinner, snack
        cuisine_preference: Optional[str] = None,
        max_recipes: int = 3
    ) -> List[FamilyRecipe]:
        """
        Generate recipes suitable for entire family
        
        Args:
            family_profile: Family member profiles
            meal_type: Type of meal
            cuisine_preference: Preferred cuisine (optional)
            max_recipes: Number of recipe options to generate
        
        Returns:
            List of family-optimized recipes
        """
        
        logger.info(f"Generating {meal_type} recipes for family of {len(family_profile.members)}")
        
        # Step 1: Analyze family requirements
        requirements = self._analyze_family_requirements(family_profile)
        
        # Step 2: Generate recipe prompt
        prompt = self._build_recipe_prompt(
            family_profile=family_profile,
            requirements=requirements,
            meal_type=meal_type,
            cuisine_preference=cuisine_preference
        )
        
        # Step 3: Generate recipes using LLM
        recipes = await self._generate_recipes_with_llm(
            prompt=prompt,
            family_profile=family_profile,
            max_recipes=max_recipes
        )
        
        # Step 4: Optimize for each family member
        optimized_recipes = []
        for recipe_data in recipes:
            optimized = await self._optimize_for_family(
                recipe_data=recipe_data,
                family_profile=family_profile
            )
            optimized_recipes.append(optimized)
        
        # Step 5: Rank by overall family suitability
        ranked_recipes = sorted(
            optimized_recipes,
            key=lambda r: r.overall_family_score,
            reverse=True
        )
        
        return ranked_recipes
    
    def _analyze_family_requirements(self, family: FamilyProfile) -> Dict[str, Any]:
        """Analyze family-wide requirements and constraints"""
        
        requirements = {
            "must_avoid": set(),
            "dietary_patterns": set(),
            "age_considerations": {},
            "health_priorities": {},
            "taste_commonalities": {"likes": [], "dislikes": []},
            "nutritional_targets": {}
        }
        
        # Collect allergens (strict requirement)
        requirements["must_avoid"].update(family.get_all_allergies())
        
        # Collect dietary restrictions
        requirements["dietary_patterns"].update(family.get_all_dietary_restrictions())
        
        # Age-specific needs
        age_groups = family.get_age_groups()
        
        if "children" in age_groups:
            requirements["age_considerations"]["children"] = {
                "needs": ["high_calcium", "age_appropriate_portions", "appealing_presentation"],
                "avoid": ["excessive_spice", "raw_foods", "choking_hazards"]
            }
        
        if "seniors" in age_groups:
            requirements["age_considerations"]["seniors"] = {
                "needs": ["soft_textures", "high_protein", "low_sodium"],
                "avoid": ["hard_to_chew", "excessive_salt"]
            }
        
        # Aggregate health goals by priority
        all_goals = {}
        for member in family.members:
            for goal in member.health_goals:
                all_goals[goal] = all_goals.get(goal, 0) + 1
        
        requirements["health_priorities"] = dict(
            sorted(all_goals.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Find taste commonalities
        all_likes = []
        all_dislikes = []
        
        for member in family.members:
            prefs = member.taste_preferences
            all_likes.extend(prefs.get("likes", []))
            all_dislikes.extend(prefs.get("dislikes", []))
        
        # Items liked by majority
        like_counts = {}
        for item in all_likes:
            like_counts[item] = like_counts.get(item, 0) + 1
        
        majority_threshold = len(family.members) / 2
        requirements["taste_commonalities"]["likes"] = [
            item for item, count in like_counts.items()
            if count >= majority_threshold
        ]
        
        # Items disliked by anyone (avoid)
        requirements["taste_commonalities"]["dislikes"] = list(set(all_dislikes))
        
        return requirements
    
    def _build_recipe_prompt(
        self,
        family_profile: FamilyProfile,
        requirements: Dict[str, Any],
        meal_type: str,
        cuisine_preference: Optional[str]
    ) -> str:
        """Build detailed prompt for LLM recipe generation"""
        
        # Family overview
        members_desc = []
        for member in family_profile.members:
            desc = f"- {member.name} (age {member.age}, {member.gender})"
            if member.health_goals:
                desc += f"\n  Goals: {', '.join(member.health_goals[:3])}"
            if member.taste_preferences.get("likes"):
                desc += f"\n  Likes: {', '.join(member.taste_preferences['likes'][:3])}"
            members_desc.append(desc)
        
        prompt = f"""Generate a {meal_type} recipe for a family with the following members:

{chr(10).join(members_desc)}

STRICT REQUIREMENTS (MUST FOLLOW):
- Allergens to AVOID: {', '.join(requirements['must_avoid']) if requirements['must_avoid'] else 'None'}
- Dietary restrictions: {', '.join(requirements['dietary_patterns']) if requirements['dietary_patterns'] else 'None'}

HEALTH PRIORITIES (ranked by importance):
{chr(10).join(f"{i+1}. {goal} ({count} members)" for i, (goal, count) in enumerate(list(requirements['health_priorities'].items())[:5]))}

TASTE PREFERENCES:
- Flavor profiles liked by family: {', '.join(requirements['taste_commonalities']['likes'][:5]) if requirements['taste_commonalities']['likes'] else 'varied'}
- Flavors to minimize/avoid: {', '.join(requirements['taste_commonalities']['dislikes']) if requirements['taste_commonalities']['dislikes'] else 'none'}

AGE CONSIDERATIONS:
{chr(10).join(f"- {age_group.title()}: {', '.join(details['needs'])}" for age_group, details in requirements['age_considerations'].items())}

PRACTICAL CONSTRAINTS:
- Cooking time available: {family_profile.available_cooking_time} minutes
- Skill level: {family_profile.cooking_skill_level}
- Budget: {family_profile.budget_level}
"""
        
        if cuisine_preference:
            prompt += f"\nPreferred cuisine: {cuisine_preference}"
        
        prompt += """

Generate a recipe that:
1. Works for ALL family members (no one left out)
2. Addresses top health goals across the family
3. Balances taste preferences (compromise when needed)
4. Provides age-appropriate portions and modifications
5. Is practical to prepare in one cooking session

Return as JSON with this structure:
{
  "name": "Recipe name",
  "description": "Brief description",
  "cuisine_type": "Cuisine style",
  "ingredients": [
    {"name": "ingredient", "amount": "quantity", "notes": "any special notes"}
  ],
  "instructions": ["step 1", "step 2", ...],
  "prep_time": minutes,
  "cook_time": minutes,
  "servings": number,
  "difficulty": "easy/medium/hard",
  "nutrition_per_serving": {
    "calories": number,
    "protein": number,
    "carbs": number,
    "fat": number,
    "fiber": number
  },
  "age_appropriate_portions": {
    "member_name": "portion size description"
  },
  "modifications_per_member": {
    "member_name": ["modification 1", "modification 2"]
  },
  "why_this_works": ["reason 1", "reason 2", ...],
  "tips_for_picky_eaters": ["tip 1", "tip 2"],
  "family_health_benefits": ["benefit 1", "benefit 2"]
}
"""
        
        return prompt
    
    async def _generate_recipes_with_llm(
        self,
        prompt: str,
        family_profile: FamilyProfile,
        max_recipes: int
    ) -> List[Dict[str, Any]]:
        """Generate recipes using LLM"""
        
        if not self.llm_client:
            logger.warning("No LLM client available, returning mock recipe")
            return [self._get_mock_recipe(family_profile)]
        
        try:
            from openai import AsyncOpenAI
            
            # Generate multiple recipe options
            recipes = []
            
            for i in range(max_recipes):
                response = await self.llm_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert family nutritionist and chef specializing in recipes that accommodate multiple family members with different needs. Always respond with valid JSON only."
                        },
                        {
                            "role": "user",
                            "content": prompt + f"\n\nGenerate recipe option #{i+1}."
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 + (i * 0.1)  # Increase variety for each recipe
                )
                
                recipe_data = json.loads(response.choices[0].message.content)
                recipes.append(recipe_data)
                
            return recipes
            
        except Exception as e:
            logger.error(f"LLM recipe generation failed: {e}")
            return [self._get_mock_recipe(family_profile)]
    
    async def _optimize_for_family(
        self,
        recipe_data: Dict[str, Any],
        family_profile: FamilyProfile
    ) -> FamilyRecipe:
        """Optimize recipe for each family member"""
        
        member_suitability = {}
        goal_alignment = {}
        taste_match = {}
        
        # Analyze for each family member
        for member in family_profile.members:
            # Calculate suitability score
            suitability = self._calculate_member_suitability(
                recipe_data=recipe_data,
                member=member
            )
            member_suitability[member.name] = suitability
            
            # Calculate goal alignment
            if self.knowledge_graph:
                alignment = await self._calculate_goal_alignment(
                    recipe_data=recipe_data,
                    member=member
                )
                goal_alignment[member.name] = alignment
            else:
                goal_alignment[member.name] = {goal: 0.7 for goal in member.health_goals}
            
            # Calculate taste match
            taste = self._calculate_taste_match(
                recipe_data=recipe_data,
                member=member
            )
            taste_match[member.name] = taste
        
        # Calculate overall family score
        overall_score = sum(member_suitability.values()) / len(member_suitability)
        
        # Check for allergens and contraindications
        allergen_warnings = self._check_allergens(recipe_data, family_profile)
        contraindications = self._check_contraindications(recipe_data, family_profile)
        
        return FamilyRecipe(
            name=recipe_data.get("name", "Family Recipe"),
            description=recipe_data.get("description", ""),
            cuisine_type=recipe_data.get("cuisine_type", "fusion"),
            ingredients=recipe_data.get("ingredients", []),
            instructions=recipe_data.get("instructions", []),
            prep_time=recipe_data.get("prep_time", 30),
            cook_time=recipe_data.get("cook_time", 30),
            servings=recipe_data.get("servings", len(family_profile.members)),
            difficulty=recipe_data.get("difficulty", "medium"),
            nutrition_per_serving=recipe_data.get("nutrition_per_serving", {}),
            member_suitability=member_suitability,
            goal_alignment=goal_alignment,
            taste_match=taste_match,
            age_appropriate_portions=recipe_data.get("age_appropriate_portions", {}),
            modifications_per_member=recipe_data.get("modifications_per_member", {}),
            cost_estimate=family_profile.budget_level,
            allergen_warnings=allergen_warnings,
            contraindications=contraindications,
            why_this_works=recipe_data.get("why_this_works", []),
            tips_for_picky_eaters=recipe_data.get("tips_for_picky_eaters", []),
            family_health_benefits=recipe_data.get("family_health_benefits", []),
            overall_family_score=overall_score
        )
    
    def _calculate_member_suitability(
        self,
        recipe_data: Dict[str, Any],
        member: FamilyMember
    ) -> float:
        """Calculate how suitable recipe is for specific member"""
        
        score = 1.0
        
        # Check allergens
        ingredients = [ing.get("name", "").lower() for ing in recipe_data.get("ingredients", [])]
        for allergen in member.allergies:
            if any(allergen.lower() in ing for ing in ingredients):
                score *= 0.2  # Major penalty for allergens
        
        # Check dietary restrictions
        for restriction in member.dietary_restrictions:
            if restriction.lower() == "vegetarian":
                meat_ingredients = ["chicken", "beef", "pork", "fish", "meat"]
                if any(meat in ing for ing in ingredients for meat in meat_ingredients):
                    score *= 0.3
            elif restriction.lower() == "vegan":
                animal_products = ["milk", "egg", "cheese", "butter", "meat", "fish"]
                if any(prod in ing for ing in ingredients for prod in animal_products):
                    score *= 0.2
        
        # Age appropriateness
        if member.is_child:
            # Check for age-appropriate modifications
            if member.name in recipe_data.get("age_appropriate_portions", {}):
                score *= 1.1  # Bonus for considering child portions
        
        return min(score, 1.0)
    
    async def _calculate_goal_alignment(
        self,
        recipe_data: Dict[str, Any],
        member: FamilyMember
    ) -> Dict[str, float]:
        """Calculate alignment with health goals"""
        
        # This would use the knowledge graph to analyze ingredients
        # For now, return estimated scores
        
        nutrition = recipe_data.get("nutrition_per_serving", {})
        alignment = {}
        
        for goal in member.health_goals:
            goal_lower = goal.lower()
            
            # Simple heuristics (should use knowledge graph in production)
            if "weight loss" in goal_lower:
                calories = nutrition.get("calories", 500)
                alignment[goal] = max(0.2, 1.0 - (calories / 800))
            elif "muscle" in goal_lower:
                protein = nutrition.get("protein", 10)
                alignment[goal] = min(1.0, protein / 30)
            elif "heart health" in goal_lower:
                fiber = nutrition.get("fiber", 5)
                alignment[goal] = min(1.0, fiber / 10)
            else:
                alignment[goal] = 0.7  # Default moderate alignment
        
        return alignment
    
    def _calculate_taste_match(
        self,
        recipe_data: Dict[str, Any],
        member: FamilyMember
    ) -> float:
        """Calculate taste preference match"""
        
        score = 0.5  # Neutral baseline
        
        prefs = member.taste_preferences
        cuisine = recipe_data.get("cuisine_type", "").lower()
        
        # Check cuisine preference
        favorite_cuisines = [c.lower() for c in prefs.get("favorite_cuisines", [])]
        if cuisine in favorite_cuisines:
            score += 0.3
        
        # Check favorite ingredients
        ingredients = [ing.get("name", "").lower() for ing in recipe_data.get("ingredients", [])]
        favorite_ingredients = [i.lower() for i in prefs.get("favorite_ingredients", [])]
        
        matches = sum(1 for fav in favorite_ingredients if any(fav in ing for ing in ingredients))
        if favorite_ingredients:
            score += 0.2 * (matches / len(favorite_ingredients))
        
        return min(score, 1.0)
    
    def _check_allergens(
        self,
        recipe_data: Dict[str, Any],
        family_profile: FamilyProfile
    ) -> List[str]:
        """Check for allergen warnings"""
        
        warnings = []
        all_allergies = family_profile.get_all_allergies()
        ingredients = [ing.get("name", "").lower() for ing in recipe_data.get("ingredients", [])]
        
        for allergen in all_allergies:
            if any(allergen.lower() in ing for ing in ingredients):
                warnings.append(f"Contains {allergen} - check family allergies")
        
        return warnings
    
    def _check_contraindications(
        self,
        recipe_data: Dict[str, Any],
        family_profile: FamilyProfile
    ) -> List[str]:
        """Check for medical contraindications"""
        
        contraindications = []
        
        # Check for high-risk ingredients for specific conditions
        ingredients = [ing.get("name", "").lower() for ing in recipe_data.get("ingredients", [])]
        
        for member in family_profile.members:
            for condition in member.medical_conditions:
                condition_lower = condition.lower()
                
                if "diabetes" in condition_lower:
                    if any(s in ing for ing in ingredients for s in ["sugar", "honey", "syrup"]):
                        contraindications.append(f"High sugar content - monitor for {member.name}'s diabetes")
                
                elif "hypertension" in condition_lower or "high blood pressure" in condition_lower:
                    if any(s in ing for ing in ingredients for s in ["salt", "sodium", "soy sauce"]):
                        contraindications.append(f"High sodium - caution for {member.name}'s blood pressure")
        
        return contraindications
    
    def _get_mock_recipe(self, family_profile: FamilyProfile) -> Dict[str, Any]:
        """Generate mock recipe for testing"""
        
        return {
            "name": "Family-Friendly Chicken & Veggie Stir-Fry",
            "description": "Colorful and nutritious stir-fry with customizable spice levels",
            "cuisine_type": "Asian fusion",
            "ingredients": [
                {"name": "chicken breast", "amount": "500g", "notes": "cut into strips"},
                {"name": "broccoli", "amount": "2 cups", "notes": "florets"},
                {"name": "carrots", "amount": "2 medium", "notes": "sliced"},
                {"name": "bell peppers", "amount": "2", "notes": "mixed colors"},
                {"name": "soy sauce", "amount": "3 tbsp", "notes": "low sodium"},
                {"name": "garlic", "amount": "3 cloves", "notes": "minced"},
                {"name": "ginger", "amount": "1 tbsp", "notes": "fresh, grated"},
                {"name": "olive oil", "amount": "2 tbsp", "notes": ""},
                {"name": "brown rice", "amount": "2 cups", "notes": "cooked"}
            ],
            "instructions": [
                "Cook brown rice according to package directions",
                "Heat oil in large pan over medium-high heat",
                "Cook chicken until golden brown, set aside",
                "Stir-fry vegetables until tender-crisp",
                "Add chicken back to pan with garlic and ginger",
                "Add soy sauce and toss to combine",
                "Serve over brown rice with customized portions"
            ],
            "prep_time": 20,
            "cook_time": 25,
            "servings": len(family_profile.members),
            "difficulty": "easy",
            "nutrition_per_serving": {
                "calories": 380,
                "protein": 28,
                "carbs": 42,
                "fat": 10,
                "fiber": 6
            },
            "age_appropriate_portions": {
                member.name: f"{'Small' if member.is_child else 'Regular'} portion with extra veggies"
                for member in family_profile.members
            },
            "modifications_per_member": {
                member.name: ["Mild spice level" if member.is_child else "Can add chili flakes"]
                for member in family_profile.members
            },
            "why_this_works": [
                "High protein supports muscle development for all ages",
                "Colorful vegetables appeal to children",
                "Easily customizable spice levels",
                "Whole grain rice provides sustained energy"
            ],
            "tips_for_picky_eaters": [
                "Let kids choose their favorite colored peppers",
                "Serve sauce on the side for control",
                "Cut veggies into fun shapes"
            ],
            "family_health_benefits": [
                "Lean protein for muscle health",
                "Fiber for digestive health",
                "Vitamins from colorful vegetables"
            ]
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def generate_family_meal_plan(
    family_profile: FamilyProfile,
    days: int = 7,
    meals_per_day: int = 3,
    llm_client: Any = None,
    knowledge_graph: Any = None
) -> Dict[str, List[FamilyRecipe]]:
    """
    Generate complete meal plan for family
    
    Args:
        family_profile: Family profile
        days: Number of days to plan
        meals_per_day: Meals per day (3 = breakfast, lunch, dinner)
        llm_client: LLM client
        knowledge_graph: Knowledge graph
    
    Returns:
        Dictionary of day -> list of recipes
    """
    
    generator = FamilyRecipeGenerator(llm_client, knowledge_graph)
    meal_plan = {}
    
    meal_types = ["breakfast", "lunch", "dinner"][:meals_per_day]
    
    for day in range(1, days + 1):
        day_meals = []
        
        for meal_type in meal_types:
            recipes = await generator.generate_family_recipe(
                family_profile=family_profile,
                meal_type=meal_type,
                max_recipes=1
            )
            
            if recipes:
                day_meals.append(recipes[0])
        
        meal_plan[f"Day {day}"] = day_meals
    
    return meal_plan
