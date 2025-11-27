"""
Family Recipe API Endpoints
============================

API for generating family-optimized recipes that accommodate
multiple family members with different ages, health goals, and tastes.

Author: Wellomex AI Team
Date: November 2025
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from app.deps import get_current_user
from app.ai_nutrition.orchestration.family_recipe_generator import (
    FamilyRecipeGenerator,
    FamilyMember,
    FamilyProfile,
    FamilyRecipe,
    generate_family_meal_plan
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/family-recipes",
    tags=["Family Recipes"],
)


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

_family_generator: Optional[FamilyRecipeGenerator] = None

def get_family_generator() -> FamilyRecipeGenerator:
    """Get or create family recipe generator"""
    global _family_generator
    
    if _family_generator is None:
        import os
        from openai import AsyncOpenAI
        
        # Try to get LLM client
        llm_client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            llm_client = AsyncOpenAI(api_key=api_key)
        
        # Try to get knowledge graph
        knowledge_graph = None
        try:
            from app.ai_nutrition.knowledge_graphs.integrated_nutrition_ai import create_integrated_system
            knowledge_graph = create_integrated_system(llm_api_key=api_key)
        except Exception as e:
            logger.warning(f"Could not load knowledge graph: {e}")
        
        _family_generator = FamilyRecipeGenerator(llm_client, knowledge_graph)
        logger.info("Family recipe generator initialized")
    
    return _family_generator


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class FamilyMemberRequest(BaseModel):
    """Family member profile"""
    name: str
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., pattern="^(male|female|other)$")
    health_goals: List[str] = Field(default_factory=list)
    medical_conditions: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    dietary_restrictions: List[str] = Field(default_factory=list)
    taste_preferences: Dict[str, Any] = Field(default_factory=dict)
    weight: Optional[float] = None
    height: Optional[float] = None
    activity_level: str = "moderate"
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Sarah",
                "age": 35,
                "gender": "female",
                "health_goals": ["weight loss", "heart health"],
                "medical_conditions": ["prediabetes"],
                "medications": [],
                "allergies": [],
                "dietary_restrictions": [],
                "taste_preferences": {
                    "likes": ["savory", "spicy"],
                    "dislikes": ["bitter"],
                    "favorite_cuisines": ["italian", "asian"],
                    "favorite_ingredients": ["chicken", "vegetables", "pasta"]
                },
                "weight": 70.0,
                "height": 165.0,
                "activity_level": "moderate"
            }
        }


class FamilyProfileRequest(BaseModel):
    """Family profile request"""
    members: List[FamilyMemberRequest]
    household_dietary_restrictions: List[str] = Field(default_factory=list)
    budget_level: str = "moderate"
    cooking_skill_level: str = "intermediate"
    available_cooking_time: int = 60
    kitchen_equipment: List[str] = Field(default_factory=list)


class GenerateRecipeRequest(BaseModel):
    """Request for recipe generation"""
    family_profile: FamilyProfileRequest
    meal_type: str = "dinner"
    cuisine_preference: Optional[str] = None
    max_recipes: int = Field(default=3, ge=1, le=10)
    
    class Config:
        schema_extra = {
            "example": {
                "family_profile": {
                    "members": [
                        {
                            "name": "Mom",
                            "age": 35,
                            "gender": "female",
                            "health_goals": ["weight loss", "heart health"],
                            "taste_preferences": {
                                "likes": ["savory", "Mediterranean"],
                                "favorite_ingredients": ["chicken", "vegetables"]
                            }
                        },
                        {
                            "name": "Dad",
                            "age": 38,
                            "gender": "male",
                            "health_goals": ["muscle gain", "energy"],
                            "medical_conditions": ["high cholesterol"],
                            "taste_preferences": {
                                "likes": ["protein-rich", "spicy"],
                                "favorite_cuisines": ["Mexican", "Asian"]
                            }
                        },
                        {
                            "name": "Emma",
                            "age": 8,
                            "gender": "female",
                            "health_goals": ["growth", "bone health"],
                            "allergies": ["peanuts"],
                            "taste_preferences": {
                                "likes": ["mild", "sweet"],
                                "dislikes": ["spicy", "bitter"],
                                "favorite_ingredients": ["pasta", "cheese", "chicken"]
                            }
                        }
                    ],
                    "household_dietary_restrictions": [],
                    "budget_level": "moderate",
                    "cooking_skill_level": "intermediate",
                    "available_cooking_time": 45
                },
                "meal_type": "dinner",
                "cuisine_preference": "Italian",
                "max_recipes": 3
            }
        }


class FamilyRecipeResponse(BaseModel):
    """Recipe response"""
    name: str
    description: str
    cuisine_type: str
    ingredients: List[Dict[str, Any]]
    instructions: List[str]
    prep_time: int
    cook_time: int
    servings: int
    difficulty: str
    nutrition_per_serving: Dict[str, float]
    member_suitability: Dict[str, float]
    goal_alignment: Dict[str, Dict[str, float]]
    taste_match: Dict[str, float]
    age_appropriate_portions: Dict[str, str]
    modifications_per_member: Dict[str, List[str]]
    cost_estimate: str
    allergen_warnings: List[str]
    contraindications: List[str]
    why_this_works: List[str]
    tips_for_picky_eaters: List[str]
    family_health_benefits: List[str]
    overall_family_score: float


class MealPlanRequest(BaseModel):
    """Request for meal plan generation"""
    family_profile: FamilyProfileRequest
    days: int = Field(default=7, ge=1, le=30)
    meals_per_day: int = Field(default=3, ge=1, le=5)


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/generate", response_model=List[FamilyRecipeResponse])
async def generate_family_recipes(
    request: GenerateRecipeRequest,
    current_user: Dict = Depends(get_current_user),
    generator: FamilyRecipeGenerator = Depends(get_family_generator)
):
    """
    Generate recipes optimized for entire family
    
    Takes family members with their ages, health goals, and taste preferences,
    and generates recipes that work for everyone.
    
    **Features:**
    - Age-appropriate portions and modifications
    - Accommodates all health goals across family
    - Balances taste preferences
    - Checks for allergies and contraindications
    - Provides customization tips
    
    **Example Use Case:**
    Family with 3 members:
    - Mom (35): weight loss, heart health, likes savory
    - Dad (38): muscle gain, high cholesterol, likes spicy
    - Child (8): growth, bone health, peanut allergy, likes mild food
    
    System generates recipes that satisfy all requirements!
    """
    
    try:
        # Convert request to internal format
        members = [
            FamilyMember(
                name=m.name,
                age=m.age,
                gender=m.gender,
                health_goals=m.health_goals,
                medical_conditions=m.medical_conditions,
                medications=m.medications,
                allergies=m.allergies,
                dietary_restrictions=m.dietary_restrictions,
                taste_preferences=m.taste_preferences,
                weight=m.weight,
                height=m.height,
                activity_level=m.activity_level
            )
            for m in request.family_profile.members
        ]
        
        family_profile = FamilyProfile(
            family_id=current_user.get("user_id", "unknown"),
            members=members,
            household_dietary_restrictions=request.family_profile.household_dietary_restrictions,
            budget_level=request.family_profile.budget_level,
            cooking_skill_level=request.family_profile.cooking_skill_level,
            available_cooking_time=request.family_profile.available_cooking_time,
            kitchen_equipment=request.family_profile.kitchen_equipment
        )
        
        # Generate recipes
        recipes = await generator.generate_family_recipe(
            family_profile=family_profile,
            meal_type=request.meal_type,
            cuisine_preference=request.cuisine_preference,
            max_recipes=request.max_recipes
        )
        
        # Convert to response format
        response_recipes = [
            FamilyRecipeResponse(
                name=recipe.name,
                description=recipe.description,
                cuisine_type=recipe.cuisine_type,
                ingredients=recipe.ingredients,
                instructions=recipe.instructions,
                prep_time=recipe.prep_time,
                cook_time=recipe.cook_time,
                servings=recipe.servings,
                difficulty=recipe.difficulty,
                nutrition_per_serving=recipe.nutrition_per_serving,
                member_suitability=recipe.member_suitability,
                goal_alignment=recipe.goal_alignment,
                taste_match=recipe.taste_match,
                age_appropriate_portions=recipe.age_appropriate_portions,
                modifications_per_member=recipe.modifications_per_member,
                cost_estimate=recipe.cost_estimate,
                allergen_warnings=recipe.allergen_warnings,
                contraindications=recipe.contraindications,
                why_this_works=recipe.why_this_works,
                tips_for_picky_eaters=recipe.tips_for_picky_eaters,
                family_health_benefits=recipe.family_health_benefits,
                overall_family_score=recipe.overall_family_score
            )
            for recipe in recipes
        ]
        
        logger.info(f"Generated {len(response_recipes)} recipes for family of {len(members)}")
        
        return response_recipes
        
    except Exception as e:
        logger.error(f"Family recipe generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recipe generation failed: {str(e)}")


@router.post("/meal-plan")
async def generate_family_meal_plan_endpoint(
    request: MealPlanRequest,
    current_user: Dict = Depends(get_current_user),
    generator: FamilyRecipeGenerator = Depends(get_family_generator)
):
    """
    Generate complete meal plan for family
    
    Creates a multi-day meal plan with breakfast, lunch, and dinner
    recipes optimized for the entire family.
    
    **Returns:** Dictionary mapping days to meal lists
    """
    
    try:
        # Convert request to internal format
        members = [
            FamilyMember(
                name=m.name,
                age=m.age,
                gender=m.gender,
                health_goals=m.health_goals,
                medical_conditions=m.medical_conditions,
                medications=m.medications,
                allergies=m.allergies,
                dietary_restrictions=m.dietary_restrictions,
                taste_preferences=m.taste_preferences,
                weight=m.weight,
                height=m.height,
                activity_level=m.activity_level
            )
            for m in request.family_profile.members
        ]
        
        family_profile = FamilyProfile(
            family_id=current_user.get("user_id", "unknown"),
            members=members,
            household_dietary_restrictions=request.family_profile.household_dietary_restrictions,
            budget_level=request.family_profile.budget_level,
            cooking_skill_level=request.family_profile.cooking_skill_level,
            available_cooking_time=request.family_profile.available_cooking_time,
            kitchen_equipment=request.family_profile.kitchen_equipment
        )
        
        # Generate meal plan
        meal_plan = await generate_family_meal_plan(
            family_profile=family_profile,
            days=request.days,
            meals_per_day=request.meals_per_day,
            llm_client=generator.llm_client,
            knowledge_graph=generator.knowledge_graph
        )
        
        logger.info(f"Generated {request.days}-day meal plan for family of {len(members)}")
        
        return meal_plan
        
    except Exception as e:
        logger.error(f"Meal plan generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Meal plan failed: {str(e)}")


@router.post("/analyze-family")
async def analyze_family_requirements(
    request: FamilyProfileRequest,
    current_user: Dict = Depends(get_current_user),
    generator: FamilyRecipeGenerator = Depends(get_family_generator)
):
    """
    Analyze family requirements and constraints
    
    Returns an analysis of the family's collective needs:
    - Must-avoid ingredients (allergens)
    - Dietary patterns
    - Age considerations
    - Health priorities
    - Taste commonalities
    
    Useful for understanding what recipes will work before generating.
    """
    
    try:
        # Convert to internal format
        members = [
            FamilyMember(
                name=m.name,
                age=m.age,
                gender=m.gender,
                health_goals=m.health_goals,
                medical_conditions=m.medical_conditions,
                medications=m.medications,
                allergies=m.allergies,
                dietary_restrictions=m.dietary_restrictions,
                taste_preferences=m.taste_preferences,
                weight=m.weight,
                height=m.height,
                activity_level=m.activity_level
            )
            for m in request.members
        ]
        
        family_profile = FamilyProfile(
            family_id=current_user.get("user_id", "unknown"),
            members=members,
            household_dietary_restrictions=request.household_dietary_restrictions,
            budget_level=request.budget_level,
            cooking_skill_level=request.cooking_skill_level,
            available_cooking_time=request.available_cooking_time,
            kitchen_equipment=request.kitchen_equipment
        )
        
        # Analyze requirements
        requirements = generator._analyze_family_requirements(family_profile)
        
        # Add family summary
        age_groups = family_profile.get_age_groups()
        
        analysis = {
            "family_summary": {
                "total_members": len(members),
                "age_groups": age_groups,
                "has_children": bool(age_groups.get("children")),
                "has_seniors": bool(age_groups.get("seniors"))
            },
            "dietary_constraints": {
                "must_avoid_allergens": list(requirements["must_avoid"]),
                "dietary_restrictions": list(requirements["dietary_patterns"]),
                "total_allergens": len(requirements["must_avoid"])
            },
            "health_priorities": requirements["health_priorities"],
            "taste_analysis": {
                "commonly_liked": requirements["taste_commonalities"]["likes"],
                "should_avoid": requirements["taste_commonalities"]["dislikes"]
            },
            "age_considerations": requirements["age_considerations"],
            "cooking_constraints": {
                "skill_level": request.cooking_skill_level,
                "time_available": request.available_cooking_time,
                "budget": request.budget_level
            }
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Family analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
