"""
Recipes API Routes
==================

AI-powered recipe generation, adaptation, and search.

Core Features:
- Multi-LLM recipe generation (GPT-4, Claude, Gemini)
- Cultural recipe knowledge graph integration
- Dietary restriction adaptation
- Ingredient substitution engine
- Nutritional optimization
- Multi-objective recipe optimization (health, cost, simplicity)
- RAG-enhanced recipe generation

Integration with:
- ai_recipe_generator_phase3c.py (Hot/Cold path LLM generation)
- cuisine_knowledge_graph_phase3_part1.py (Cultural authenticity)
- meal_planning_service_phase1.py (Recipe transformation)
- recipes/recipe_generation.py (Core generation logic)
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

router = APIRouter()


# ============================================================================
# ENUMS
# ============================================================================

class DifficultyLevelEnum(str, Enum):
    """Recipe difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class CuisineTypeEnum(str, Enum):
    """Supported cuisine types"""
    AMERICAN = "american"
    ITALIAN = "italian"
    MEXICAN = "mexican"
    CHINESE = "chinese"
    JAPANESE = "japanese"
    INDIAN = "indian"
    FRENCH = "french"
    THAI = "thai"
    MEDITERRANEAN = "mediterranean"
    FUSION = "fusion"


class DietaryRestrictionEnum(str, Enum):
    """Dietary restrictions"""
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    NUT_FREE = "nut_free"
    KETO = "keto"
    PALEO = "paleo"
    LOW_CARB = "low_carb"
    LOW_SODIUM = "low_sodium"
    DIABETIC_FRIENDLY = "diabetic_friendly"


class MealTypeEnum(str, Enum):
    """Meal types"""
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"
    DESSERT = "dessert"
    APPETIZER = "appetizer"


class LLMProviderEnum(str, Enum):
    """LLM providers for recipe generation"""
    GPT4_TURBO = "gpt4_turbo"
    GPT35_TURBO = "gpt35_turbo"
    CLAUDE_3_OPUS = "claude_3_opus"
    CLAUDE_3_SONNET = "claude_3_sonnet"
    GEMINI_PRO = "gemini_pro"
    GEMINI_FLASH = "gemini_flash"


# ============================================================================
# REQUEST MODELS
# ============================================================================

class RecipeGenerationRequest(BaseModel):
    """Request for AI recipe generation"""
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    
    # Core requirements
    available_ingredients: Optional[List[str]] = Field(
        None,
        description="Available ingredients to use",
        example=["chicken breast", "rice", "tomatoes", "onion", "garlic"]
    )
    cuisine: Optional[CuisineTypeEnum] = Field(
        None,
        description="Desired cuisine type"
    )
    meal_type: Optional[MealTypeEnum] = Field(
        None,
        description="Type of meal"
    )
    
    # Dietary constraints
    dietary_restrictions: Optional[List[DietaryRestrictionEnum]] = Field(
        None,
        description="Dietary restrictions to follow"
    )
    medical_conditions: Optional[List[str]] = Field(
        None,
        description="Medical conditions for safety (e.g., 'diabetes', 'CKD', 'hypertension')",
        example=["diabetes", "hypertension"]
    )
    allergies: Optional[List[str]] = Field(
        None,
        description="Food allergies to avoid",
        example=["peanuts", "shellfish"]
    )
    
    # Nutritional targets
    calorie_target: Optional[int] = Field(
        None,
        description="Target calories per serving",
        ge=100,
        le=2000
    )
    protein_target_g: Optional[float] = Field(
        None,
        description="Target protein in grams"
    )
    max_sodium_mg: Optional[float] = Field(
        None,
        description="Maximum sodium in milligrams"
    )
    
    # Preferences
    difficulty: Optional[DifficultyLevelEnum] = Field(
        None,
        description="Desired difficulty level"
    )
    max_cooking_time_minutes: Optional[int] = Field(
        None,
        description="Maximum cooking time",
        ge=5,
        le=480
    )
    servings: int = Field(
        4,
        description="Number of servings",
        ge=1,
        le=20
    )
    
    # Generation settings
    llm_provider: LLMProviderEnum = Field(
        LLMProviderEnum.GPT4_TURBO,
        description="LLM provider for generation"
    )
    enable_optimization: bool = Field(
        True,
        description="Enable multi-objective optimization"
    )
    include_cultural_notes: bool = Field(
        True,
        description="Include cultural background and authenticity notes"
    )

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "available_ingredients": ["chicken", "rice", "tomatoes", "onion"],
                "cuisine": "mexican",
                "meal_type": "dinner",
                "dietary_restrictions": ["gluten_free"],
                "medical_conditions": ["diabetes"],
                "calorie_target": 600,
                "max_sodium_mg": 800,
                "difficulty": "medium",
                "max_cooking_time_minutes": 45,
                "servings": 4,
                "llm_provider": "gpt4_turbo",
                "enable_optimization": True
            }
        }


class RecipeAdaptationRequest(BaseModel):
    """Request for adapting existing recipe"""
    recipe_id: Optional[str] = Field(None, description="Existing recipe ID to adapt")
    recipe_text: Optional[str] = Field(
        None,
        description="Recipe text to adapt (if not using recipe_id)"
    )
    
    # Adaptations
    new_dietary_restrictions: Optional[List[DietaryRestrictionEnum]] = Field(
        None,
        description="New dietary restrictions to apply"
    )
    reduce_calories_by_percent: Optional[float] = Field(
        None,
        description="Percentage to reduce calories",
        ge=0,
        le=50
    )
    reduce_sodium_by_percent: Optional[float] = Field(
        None,
        description="Percentage to reduce sodium",
        ge=0,
        le=70
    )
    increase_protein_by_percent: Optional[float] = Field(
        None,
        description="Percentage to increase protein",
        ge=0,
        le=100
    )
    
    preserve_flavor: bool = Field(
        True,
        description="Preserve original flavor profile"
    )
    preserve_cuisine: bool = Field(
        True,
        description="Preserve cultural authenticity"
    )

    @validator('recipe_text')
    def validate_recipe_input(cls, v, values):
        if not v and not values.get('recipe_id'):
            raise ValueError("Either recipe_id or recipe_text must be provided")
        return v


class IngredientSubstitutionRequest(BaseModel):
    """Request for ingredient substitution suggestions"""
    ingredient_to_replace: str = Field(
        ...,
        description="Ingredient to substitute",
        example="butter"
    )
    recipe_context: Optional[str] = Field(
        None,
        description="Recipe context for better suggestions"
    )
    dietary_restrictions: Optional[List[DietaryRestrictionEnum]] = Field(
        None,
        description="Dietary restrictions for substitutes"
    )
    preserve_texture: bool = Field(
        True,
        description="Preserve texture characteristics"
    )
    preserve_flavor: bool = Field(
        True,
        description="Preserve flavor profile"
    )


class RecipeSearchRequest(BaseModel):
    """Request for recipe search"""
    query: str = Field(
        ...,
        description="Search query (keywords, ingredients, dish name)",
        example="healthy chicken pasta"
    )
    cuisines: Optional[List[CuisineTypeEnum]] = Field(
        None,
        description="Filter by cuisines"
    )
    dietary_restrictions: Optional[List[DietaryRestrictionEnum]] = Field(
        None,
        description="Filter by dietary restrictions"
    )
    max_calories: Optional[int] = Field(
        None,
        description="Maximum calories per serving"
    )
    max_cooking_time: Optional[int] = Field(
        None,
        description="Maximum cooking time in minutes"
    )
    difficulty: Optional[List[DifficultyLevelEnum]] = Field(
        None,
        description="Filter by difficulty levels"
    )
    limit: int = Field(
        20,
        description="Maximum number of results",
        ge=1,
        le=100
    )


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class NutritionalInfo(BaseModel):
    """Nutritional information per serving"""
    calories: float = Field(..., description="Calories")
    protein_g: float = Field(..., description="Protein in grams")
    carbohydrates_g: float = Field(..., description="Carbohydrates in grams")
    fat_g: float = Field(..., description="Fat in grams")
    fiber_g: Optional[float] = Field(None, description="Fiber in grams")
    sodium_mg: Optional[float] = Field(None, description="Sodium in milligrams")
    sugar_g: Optional[float] = Field(None, description="Sugar in grams")
    saturated_fat_g: Optional[float] = Field(None, description="Saturated fat in grams")


class RecipeIngredient(BaseModel):
    """Single recipe ingredient"""
    name: str = Field(..., description="Ingredient name")
    quantity: float = Field(..., description="Quantity")
    unit: str = Field(..., description="Unit of measurement")
    notes: Optional[str] = Field(None, description="Additional notes")
    is_optional: bool = Field(False, description="Whether ingredient is optional")


class RecipeStep(BaseModel):
    """Single recipe step"""
    step_number: int = Field(..., description="Step number")
    instruction: str = Field(..., description="Step instruction")
    duration_minutes: Optional[int] = Field(None, description="Duration for this step")
    tips: Optional[str] = Field(None, description="Tips for this step")


class ValidationResult(BaseModel):
    """Recipe validation result"""
    is_valid: bool = Field(..., description="Whether recipe is valid")
    passed_checks: List[str] = Field(..., description="Checks that passed")
    failed_checks: List[str] = Field(..., description="Checks that failed")
    warnings: List[str] = Field(..., description="Warning messages")
    health_risk_score: float = Field(
        ...,
        description="Health risk score (0.0=safe, 1.0=high risk)"
    )
    cultural_authenticity_score: Optional[float] = Field(
        None,
        description="Cultural authenticity score (0-1)"
    )


class OptimizationResult(BaseModel):
    """Multi-objective optimization result"""
    objectives_optimized: List[str] = Field(
        ...,
        description="Objectives that were optimized",
        example=["health_risk", "nutritional_balance", "cost_efficiency"]
    )
    improvements: Dict[str, float] = Field(
        ...,
        description="Percentage improvements per objective",
        example={
            "health_risk_reduction": 35.2,
            "nutritional_score_increase": 18.5,
            "cost_reduction": 12.0
        }
    )
    substitutions_made: List[Dict[str, str]] = Field(
        ...,
        description="Ingredient substitutions",
        example=[
            {"original": "butter", "substitute": "olive oil", "reason": "Reduce saturated fat"}
        ]
    )


class RecipeResponse(BaseModel):
    """Complete recipe response"""
    recipe_id: str = Field(..., description="Unique recipe identifier")
    recipe_name: str = Field(..., description="Recipe name")
    cuisine: Optional[CuisineTypeEnum] = Field(None, description="Cuisine type")
    meal_type: Optional[MealTypeEnum] = Field(None, description="Meal type")
    difficulty: DifficultyLevelEnum = Field(..., description="Difficulty level")
    
    # Timing
    prep_time_minutes: int = Field(..., description="Preparation time")
    cook_time_minutes: int = Field(..., description="Cooking time")
    total_time_minutes: int = Field(..., description="Total time")
    
    # Content
    servings: int = Field(..., description="Number of servings")
    ingredients: List[RecipeIngredient] = Field(..., description="Ingredients list")
    instructions: List[RecipeStep] = Field(..., description="Cooking instructions")
    
    # Nutrition
    nutritional_info: NutritionalInfo = Field(..., description="Nutritional information")
    
    # Additional info
    description: Optional[str] = Field(None, description="Recipe description")
    cultural_notes: Optional[str] = Field(None, description="Cultural background")
    health_benefits: Optional[List[str]] = Field(None, description="Health benefits")
    tips: Optional[List[str]] = Field(None, description="Cooking tips")
    
    # Metadata
    llm_provider: Optional[LLMProviderEnum] = Field(None, description="LLM used for generation")
    confidence_score: float = Field(
        ...,
        description="Generation confidence (0-1)",
        ge=0.0,
        le=1.0
    )
    tokens_used: Optional[int] = Field(None, description="Tokens used in generation")
    
    # Validation & Optimization
    validation: Optional[ValidationResult] = Field(None, description="Validation results")
    optimization: Optional[OptimizationResult] = Field(None, description="Optimization results")

    class Config:
        schema_extra = {
            "example": {
                "recipe_id": "recipe_20231115_123456",
                "recipe_name": "Healthy Grilled Chicken with Cilantro Lime Rice",
                "cuisine": "mexican",
                "meal_type": "dinner",
                "difficulty": "medium",
                "prep_time_minutes": 15,
                "cook_time_minutes": 30,
                "total_time_minutes": 45,
                "servings": 4,
                "ingredients": [
                    {
                        "name": "chicken breast",
                        "quantity": 1.5,
                        "unit": "lbs",
                        "notes": "boneless, skinless"
                    }
                ],
                "nutritional_info": {
                    "calories": 450,
                    "protein_g": 38,
                    "carbohydrates_g": 42,
                    "fat_g": 12
                },
                "confidence_score": 0.92
            }
        }


class SubstitutionSuggestion(BaseModel):
    """Ingredient substitution suggestion"""
    original_ingredient: str = Field(..., description="Original ingredient")
    substitute: str = Field(..., description="Suggested substitute")
    substitution_ratio: str = Field(
        ...,
        description="Substitution ratio",
        example="1:1"
    )
    reason: str = Field(..., description="Reason for substitution")
    texture_match: float = Field(
        ...,
        description="Texture similarity (0-1)",
        ge=0.0,
        le=1.0
    )
    flavor_match: float = Field(
        ...,
        description="Flavor similarity (0-1)",
        ge=0.0,
        le=1.0
    )
    nutritional_improvement: Optional[Dict[str, float]] = Field(
        None,
        description="Nutritional improvements",
        example={"fat_reduction_percent": 30, "fiber_increase_percent": 15}
    )


class IngredientSubstitutionResponse(BaseModel):
    """Ingredient substitution response"""
    request_id: str = Field(..., description="Request identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    original_ingredient: str = Field(..., description="Ingredient to replace")
    substitutions: List[SubstitutionSuggestion] = Field(
        ...,
        description="Substitution suggestions (ranked by suitability)"
    )


class RecipeSearchResult(BaseModel):
    """Single recipe search result"""
    recipe_id: str = Field(..., description="Recipe ID")
    recipe_name: str = Field(..., description="Recipe name")
    cuisine: Optional[CuisineTypeEnum] = Field(None, description="Cuisine type")
    difficulty: DifficultyLevelEnum = Field(..., description="Difficulty")
    total_time_minutes: int = Field(..., description="Total time")
    servings: int = Field(..., description="Servings")
    calories_per_serving: float = Field(..., description="Calories per serving")
    match_score: float = Field(
        ...,
        description="Search relevance score (0-1)",
        ge=0.0,
        le=1.0
    )
    thumbnail_url: Optional[str] = Field(None, description="Recipe thumbnail")


class RecipeSearchResponse(BaseModel):
    """Recipe search response"""
    request_id: str = Field(..., description="Search request ID")
    timestamp: datetime = Field(..., description="Search timestamp")
    query: str = Field(..., description="Search query")
    total_results: int = Field(..., description="Total matching recipes")
    results: List[RecipeSearchResult] = Field(..., description="Search results")


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "/generate",
    response_model=RecipeResponse,
    summary="Generate AI recipe",
    description="""
    Generate a complete recipe using advanced AI (GPT-4, Claude, Gemini).
    
    **Features:**
    - Multi-LLM support (GPT-4 Turbo, Claude 3 Opus, Gemini Pro)
    - Cultural recipe knowledge graph integration
    - Medical condition safety validation
    - Multi-objective optimization (health, cost, simplicity)
    - RAG-enhanced with 50,000+ recipe database
    
    **Generation Process:**
    1. **Hot Path**: Instant LLM generation with prompt engineering
    2. **RAG Enhancement**: Retrieve similar recipes for context
    3. **Validation**: Check safety, nutrition, cultural authenticity
    4. **Optimization**: Multi-objective improvements (optional)
    5. **Cold Path**: Background learning from user feedback
    
    **Optimization Objectives:**
    - **Health Risk**: Reduce risky ingredients for medical conditions
    - **Nutritional Balance**: Optimize macro/micronutrient ratios
    - **Cost Efficiency**: Minimize ingredient cost
    - **Cooking Simplicity**: Reduce steps and complexity
    """,
)
async def generate_recipe(request: RecipeGenerationRequest):
    """
    Generate a complete AI recipe.
    """
    try:
        # TODO: Integrate with actual AI recipe generator
        # from app.ai_nutrition.microservices.ai_recipe_generator_phase3c import AIRecipeGeneratorService
        # service = AIRecipeGeneratorService()
        # result = await service.generate_recipe_complete(request)
        
        # MOCK RESPONSE
        recipe_id = f"recipe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        recipe = RecipeResponse(
            recipe_id=recipe_id,
            recipe_name="Healthy Grilled Chicken with Cilantro Lime Rice",
            cuisine=request.cuisine or CuisineTypeEnum.MEXICAN,
            meal_type=request.meal_type or MealTypeEnum.DINNER,
            difficulty=request.difficulty or DifficultyLevelEnum.MEDIUM,
            prep_time_minutes=15,
            cook_time_minutes=30,
            total_time_minutes=45,
            servings=request.servings,
            ingredients=[
                RecipeIngredient(
                    name="chicken breast",
                    quantity=1.5,
                    unit="lbs",
                    notes="boneless, skinless",
                    is_optional=False
                ),
                RecipeIngredient(
                    name="white rice",
                    quantity=2,
                    unit="cups",
                    is_optional=False
                ),
                RecipeIngredient(
                    name="fresh cilantro",
                    quantity=0.25,
                    unit="cup",
                    notes="chopped",
                    is_optional=False
                ),
                RecipeIngredient(
                    name="lime juice",
                    quantity=3,
                    unit="tbsp",
                    is_optional=False
                )
            ],
            instructions=[
                RecipeStep(
                    step_number=1,
                    instruction="Season chicken breast with cumin, paprika, salt, and pepper",
                    duration_minutes=5
                ),
                RecipeStep(
                    step_number=2,
                    instruction="Grill chicken over medium-high heat for 6-7 minutes per side",
                    duration_minutes=15
                ),
                RecipeStep(
                    step_number=3,
                    instruction="Meanwhile, cook rice according to package instructions",
                    duration_minutes=20
                ),
                RecipeStep(
                    step_number=4,
                    instruction="Fluff cooked rice with fork and mix in cilantro and lime juice",
                    duration_minutes=2
                ),
                RecipeStep(
                    step_number=5,
                    instruction="Slice grilled chicken and serve over cilantro lime rice",
                    duration_minutes=3
                )
            ],
            nutritional_info=NutritionalInfo(
                calories=450,
                protein_g=38,
                carbohydrates_g=42,
                fat_g=12,
                fiber_g=2,
                sodium_mg=380,
                sugar_g=1,
                saturated_fat_g=3
            ),
            description="A healthy, diabetic-friendly Mexican-inspired chicken dish with flavorful cilantro lime rice",
            cultural_notes="Cilantro lime rice is a staple in Mexican cuisine, providing fresh, vibrant flavors",
            health_benefits=[
                "High protein supports muscle health",
                "Low sodium helps manage blood pressure",
                "Balanced macronutrients suitable for diabetes management"
            ],
            tips=[
                "Marinate chicken for 30 minutes for extra flavor",
                "Use brown rice for added fiber",
                "Grill vegetables alongside for added nutrition"
            ],
            llm_provider=request.llm_provider,
            confidence_score=0.92,
            tokens_used=1250,
            validation=ValidationResult(
                is_valid=True,
                passed_checks=["medical_safety", "nutritional_balance", "ingredient_compatibility"],
                failed_checks=[],
                warnings=[],
                health_risk_score=0.05,
                cultural_authenticity_score=0.88
            ),
            optimization=OptimizationResult(
                objectives_optimized=["health_risk", "nutritional_balance"],
                improvements={
                    "health_risk_reduction": 35.2,
                    "nutritional_score_increase": 18.5,
                    "sodium_reduction": 25.0
                },
                substitutions_made=[
                    {
                        "original": "regular salt",
                        "substitute": "reduced sodium salt",
                        "reason": "Reduce sodium for diabetes/hypertension"
                    }
                ]
            ) if request.enable_optimization else None
        )
        
        return recipe
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recipe generation error: {str(e)}"
        )


@router.post(
    "/adapt",
    response_model=RecipeResponse,
    summary="Adapt existing recipe",
    description="""
    Adapt an existing recipe to meet new dietary requirements or nutritional goals.
    
    **Adaptations:**
    - Apply new dietary restrictions (vegan, gluten-free, etc.)
    - Reduce calories, sodium, sugar
    - Increase protein, fiber
    - Preserve flavor and cultural authenticity
    
    **AI-Powered Transformations:**
    - Intelligent ingredient substitutions
    - Molecular flavor profile preservation
    - Nutritional recalculation
    - Cooking method adjustments
    """,
)
async def adapt_recipe(request: RecipeAdaptationRequest):
    """
    Adapt existing recipe to new requirements.
    """
    try:
        # TODO: Integrate with recipe transformation engine
        # from app.ai_nutrition.microservices.meal_planning_service_phase1 import AIRecipeTransformer
        # transformer = AIRecipeTransformer()
        # result = await transformer.transform_recipe(request)
        
        # MOCK RESPONSE
        recipe_id = f"recipe_adapted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Simulate adaptation
        original_calories = 650
        adapted_calories = original_calories * (1 - (request.reduce_calories_by_percent or 0) / 100)
        
        recipe = RecipeResponse(
            recipe_id=recipe_id,
            recipe_name="Adapted Healthy Pasta Primavera (Gluten-Free)",
            cuisine=CuisineTypeEnum.ITALIAN,
            meal_type=MealTypeEnum.DINNER,
            difficulty=DifficultyLevelEnum.MEDIUM,
            prep_time_minutes=15,
            cook_time_minutes=25,
            total_time_minutes=40,
            servings=4,
            ingredients=[
                RecipeIngredient(
                    name="gluten-free pasta",
                    quantity=12,
                    unit="oz",
                    notes="substituted for regular pasta"
                )
            ],
            instructions=[
                RecipeStep(
                    step_number=1,
                    instruction="Cook gluten-free pasta according to package instructions",
                    duration_minutes=12
                )
            ],
            nutritional_info=NutritionalInfo(
                calories=adapted_calories,
                protein_g=18,
                carbohydrates_g=62,
                fat_g=15
            ),
            description="Adapted for gluten-free diet while preserving Italian flavors",
            confidence_score=0.89,
            validation=ValidationResult(
                is_valid=True,
                passed_checks=["gluten_free_compliance", "nutritional_balance"],
                failed_checks=[],
                warnings=[],
                health_risk_score=0.02
            )
        )
        
        return recipe
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recipe adaptation error: {str(e)}"
        )


@router.post(
    "/substitute",
    response_model=IngredientSubstitutionResponse,
    summary="Get ingredient substitution suggestions",
    description="""
    Find suitable ingredient substitutions with nutritional and functional analysis.
    
    **Substitution Criteria:**
    - Texture compatibility
    - Flavor profile matching
    - Nutritional equivalence or improvement
    - Dietary restriction compliance
    - Cooking behavior similarity
    
    **Use Cases:**
    - Missing ingredients
    - Dietary restrictions
    - Allergy accommodations
    - Cost optimization
    - Nutritional improvement
    """,
)
async def get_substitutions(request: IngredientSubstitutionRequest):
    """
    Get ingredient substitution suggestions.
    """
    try:
        # TODO: Integrate with substitution engine
        # MOCK RESPONSE
        request_id = f"subst_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Mock substitutions for common ingredients
        substitution_db = {
            "butter": [
                SubstitutionSuggestion(
                    original_ingredient="butter",
                    substitute="olive oil",
                    substitution_ratio="1:3/4 cup",
                    reason="Healthier fat profile, reduces saturated fat",
                    texture_match=0.80,
                    flavor_match=0.75,
                    nutritional_improvement={
                        "saturated_fat_reduction_percent": 50,
                        "heart_health_score_increase": 35
                    }
                ),
                SubstitutionSuggestion(
                    original_ingredient="butter",
                    substitute="avocado",
                    substitution_ratio="1:1",
                    reason="Vegan option with healthy fats",
                    texture_match=0.85,
                    flavor_match=0.70,
                    nutritional_improvement={
                        "fiber_increase_percent": 100,
                        "saturated_fat_reduction_percent": 60
                    }
                )
            ],
            "eggs": [
                SubstitutionSuggestion(
                    original_ingredient="eggs",
                    substitute="flax eggs",
                    substitution_ratio="1 tbsp ground flax + 3 tbsp water per egg",
                    reason="Vegan option, adds omega-3 fatty acids",
                    texture_match=0.75,
                    flavor_match=0.80,
                    nutritional_improvement={
                        "omega3_increase_percent": 200,
                        "cholesterol_reduction_percent": 100
                    }
                )
            ]
        }
        
        ingredient_lower = request.ingredient_to_replace.lower()
        substitutions = substitution_db.get(ingredient_lower, [
            SubstitutionSuggestion(
                original_ingredient=request.ingredient_to_replace,
                substitute=f"alternative_{request.ingredient_to_replace}",
                substitution_ratio="1:1",
                reason="Generic substitution",
                texture_match=0.70,
                flavor_match=0.70
            )
        ])
        
        response = IngredientSubstitutionResponse(
            request_id=request_id,
            timestamp=datetime.now(),
            original_ingredient=request.ingredient_to_replace,
            substitutions=substitutions
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Substitution error: {str(e)}"
        )


@router.post(
    "/search",
    response_model=RecipeSearchResponse,
    summary="Search recipes",
    description="""
    Search recipe database with advanced filtering.
    
    **Search Features:**
    - Keyword search (ingredients, dish names, techniques)
    - Multi-filter support (cuisine, dietary restrictions, time, difficulty)
    - Semantic search (understands intent and context)
    - Nutritional filtering
    - Relevance ranking
    """,
)
async def search_recipes(request: RecipeSearchRequest):
    """
    Search recipe database.
    """
    try:
        # TODO: Integrate with recipe search engine
        # MOCK RESPONSE
        request_id = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        mock_results = [
            RecipeSearchResult(
                recipe_id="recipe_001",
                recipe_name="Healthy Chicken Pasta",
                cuisine=CuisineTypeEnum.ITALIAN,
                difficulty=DifficultyLevelEnum.MEDIUM,
                total_time_minutes=35,
                servings=4,
                calories_per_serving=450,
                match_score=0.95
            ),
            RecipeSearchResult(
                recipe_id="recipe_002",
                recipe_name="Grilled Chicken Salad",
                cuisine=CuisineTypeEnum.AMERICAN,
                difficulty=DifficultyLevelEnum.EASY,
                total_time_minutes=20,
                servings=2,
                calories_per_serving=320,
                match_score=0.88
            )
        ]
        
        # Filter by max_calories if provided
        if request.max_calories:
            mock_results = [
                r for r in mock_results
                if r.calories_per_serving <= request.max_calories
            ]
        
        # Limit results
        mock_results = mock_results[:request.limit]
        
        response = RecipeSearchResponse(
            request_id=request_id,
            timestamp=datetime.now(),
            query=request.query,
            total_results=len(mock_results),
            results=mock_results
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recipe search error: {str(e)}"
        )


@router.get(
    "/{recipe_id}",
    response_model=RecipeResponse,
    summary="Get recipe by ID",
    description="Retrieve a specific recipe by its unique identifier",
)
async def get_recipe(recipe_id: str):
    """
    Get recipe by ID.
    """
    try:
        # TODO: Retrieve from database
        # MOCK RESPONSE
        if not recipe_id.startswith("recipe_"):
            raise HTTPException(status_code=404, detail="Recipe not found")
        
        recipe = RecipeResponse(
            recipe_id=recipe_id,
            recipe_name="Mock Recipe",
            difficulty=DifficultyLevelEnum.MEDIUM,
            prep_time_minutes=15,
            cook_time_minutes=30,
            total_time_minutes=45,
            servings=4,
            ingredients=[],
            instructions=[],
            nutritional_info=NutritionalInfo(
                calories=400,
                protein_g=25,
                carbohydrates_g=45,
                fat_g=15
            ),
            confidence_score=0.90
        )
        
        return recipe
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving recipe: {str(e)}"
        )


@router.get(
    "/health",
    summary="Health check for recipes service",
)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "recipes",
        "timestamp": datetime.now(),
        "capabilities": [
            "ai_recipe_generation",
            "multi_llm_support",
            "recipe_adaptation",
            "ingredient_substitution",
            "recipe_search",
            "nutritional_optimization"
        ],
        "supported_llms": [p.value for p in LLMProviderEnum],
        "supported_cuisines": [c.value for c in CuisineTypeEnum]
    }
