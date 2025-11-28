"""
Database Models for Meal Planning System
=========================================

SQLAlchemy models for:
- User profiles with health data
- Meal plans and recipes
- Country preferences
- Disease conditions
- Ingredient databases
- Nutritional tracking
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, 
    JSON, Text, ForeignKey, Table, Enum as SQLEnum,
    UniqueConstraint, Index, CheckConstraint
)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import List, Dict, Optional
import enum

Base = declarative_base()


# ============================================================================
# ENUMS
# ============================================================================

class ActivityLevel(enum.Enum):
    """Physical activity levels"""
    SEDENTARY = "sedentary"
    LIGHTLY_ACTIVE = "lightly_active"
    MODERATELY_ACTIVE = "moderately_active"
    VERY_ACTIVE = "very_active"
    EXTREMELY_ACTIVE = "extremely_active"


class Gender(enum.Enum):
    """Gender options"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


class MealType(enum.Enum):
    """Types of meals"""
    BREAKFAST = "breakfast"
    MORNING_SNACK = "morning_snack"
    LUNCH = "lunch"
    AFTERNOON_SNACK = "afternoon_snack"
    DINNER = "dinner"
    EVENING_SNACK = "evening_snack"


class DietaryRestriction(enum.Enum):
    """Common dietary restrictions"""
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    HALAL = "halal"
    KOSHER = "kosher"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    NUT_FREE = "nut_free"
    SHELLFISH_FREE = "shellfish_free"
    SOY_FREE = "soy_free"
    EGG_FREE = "egg_free"
    PESCATARIAN = "pescatarian"
    PALEO = "paleo"
    KETO = "keto"
    LOW_FODMAP = "low_fodmap"


class DiseaseSeverity(enum.Enum):
    """Disease severity levels"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class PlanStatus(enum.Enum):
    """Meal plan status"""
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# ============================================================================
# ASSOCIATION TABLES
# ============================================================================

# Many-to-many: User <-> DiseaseCondition
user_disease_association = Table(
    'user_disease_association',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id', ondelete='CASCADE')),
    Column('disease_id', Integer, ForeignKey('disease_conditions.id', ondelete='CASCADE')),
    Column('diagnosed_date', DateTime),
    Column('severity', SQLEnum(DiseaseSeverity)),
    Column('controlled', Boolean, default=False),
    Column('notes', Text)
)

# Many-to-many: User <-> Allergen
user_allergen_association = Table(
    'user_allergen_association',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id', ondelete='CASCADE')),
    Column('allergen_id', Integer, ForeignKey('allergens.id', ondelete='CASCADE')),
    Column('severity', String(50)),  # mild, moderate, severe, anaphylaxis
    Column('diagnosed_date', DateTime)
)

# Many-to-many: Recipe <-> Ingredient
recipe_ingredient_association = Table(
    'recipe_ingredient_association',
    Base.metadata,
    Column('recipe_id', Integer, ForeignKey('recipes.id', ondelete='CASCADE')),
    Column('ingredient_id', Integer, ForeignKey('ingredients.id', ondelete='CASCADE')),
    Column('quantity', Float),
    Column('unit', String(50)),
    Column('preparation_notes', String(255)),
    Column('optional', Boolean, default=False)
)

# Many-to-many: Recipe <-> Country
recipe_country_association = Table(
    'recipe_country_association',
    Base.metadata,
    Column('recipe_id', Integer, ForeignKey('recipes.id', ondelete='CASCADE')),
    Column('country_id', Integer, ForeignKey('countries.id', ondelete='CASCADE')),
    Column('authenticity_score', Float),  # 0-1 scale
    Column('regional_variant', String(100))
)


# ============================================================================
# USER & PROFILE MODELS
# ============================================================================

class User(Base):
    """User profile with comprehensive health data"""
    __tablename__ = 'users'
    
    # Primary fields
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Personal information
    first_name = Column(String(100))
    last_name = Column(String(100))
    date_of_birth = Column(DateTime)
    gender = Column(SQLEnum(Gender))
    
    # Physical metrics
    height_cm = Column(Float)
    weight_kg = Column(Float)
    target_weight_kg = Column(Float)
    activity_level = Column(SQLEnum(ActivityLevel))
    
    # Location & preferences
    country_code = Column(String(2), index=True)  # ISO 3166-1 alpha-2
    primary_language = Column(String(10), default='en')
    timezone = Column(String(50))
    
    # Dietary preferences
    dietary_restrictions = Column(JSON)  # List of DietaryRestriction values
    cuisine_preferences = Column(JSON)  # List of preferred cuisines
    disliked_ingredients = Column(JSON)  # List of ingredient IDs to avoid
    spice_tolerance = Column(String(20))  # none, mild, medium, hot, very_hot
    
    # Health goals
    health_goals = Column(JSON)  # List of goals: weight_loss, muscle_gain, etc.
    daily_calorie_target = Column(Integer)
    protein_target_g = Column(Float)
    carb_target_g = Column(Float)
    fat_target_g = Column(Float)
    fiber_target_g = Column(Float)
    
    # Medical data
    blood_pressure_systolic = Column(Integer)
    blood_pressure_diastolic = Column(Integer)
    blood_glucose_mg_dl = Column(Float)
    cholesterol_total_mg_dl = Column(Float)
    cholesterol_ldl_mg_dl = Column(Float)
    cholesterol_hdl_mg_dl = Column(Float)
    triglycerides_mg_dl = Column(Float)
    hba1c_percent = Column(Float)
    
    # Medications & supplements
    current_medications = Column(JSON)  # List of medication names
    supplements = Column(JSON)  # List of supplements
    
    # Genetic markers (optional)
    genetic_markers = Column(JSON)  # e.g., lactose intolerance genes
    
    # Preferences
    cooking_skill_level = Column(String(20))  # beginner, intermediate, advanced
    cooking_time_max_minutes = Column(Integer, default=60)
    budget_per_day_usd = Column(Float)
    local_food_priority = Column(String(20))  # low, medium, high
    
    # Privacy & consent
    data_sharing_consent = Column(Boolean, default=False)
    research_participation = Column(Boolean, default=False)
    
    # Relationships
    diseases = relationship(
        'DiseaseCondition',
        secondary=user_disease_association,
        backref='users'
    )
    allergens = relationship(
        'Allergen',
        secondary=user_allergen_association,
        backref='users'
    )
    meal_plans = relationship('MealPlan', back_populates='user', cascade='all, delete-orphan')
    meal_ratings = relationship('MealRating', back_populates='user', cascade='all, delete-orphan')
    
    # Indexes
    __table_args__ = (
        Index('idx_user_country', 'country_code'),
        Index('idx_user_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', country='{self.country_code}')>"


# ============================================================================
# COUNTRY & CUISINE MODELS
# ============================================================================

class Country(Base):
    """Country data with cuisine information"""
    __tablename__ = 'countries'
    
    id = Column(Integer, primary_key=True)
    code = Column(String(2), unique=True, nullable=False, index=True)  # ISO 3166-1 alpha-2
    name = Column(String(100), nullable=False)
    region = Column(String(50))  # Africa, Asia, Europe, Americas, Oceania
    subregion = Column(String(50))  # East Africa, Southeast Asia, etc.
    
    # Cuisine characteristics
    traditional_cuisines = Column(JSON)  # List of cuisine names
    staple_foods = Column(JSON)  # List of staple ingredients
    common_spices = Column(JSON)  # List of commonly used spices
    cooking_methods = Column(JSON)  # List of traditional cooking methods
    
    # Meal patterns
    typical_meal_count = Column(Integer, default=3)
    breakfast_time = Column(String(10))  # e.g., "07:00"
    lunch_time = Column(String(10))
    dinner_time = Column(String(10))
    
    # Cultural notes
    religious_dietary_laws = Column(JSON)  # Common religious restrictions
    seasonal_foods = Column(JSON)  # Foods by season
    festival_foods = Column(JSON)  # Special occasion foods
    
    # Economic data
    average_food_cost_index = Column(Float)  # Relative to global average
    food_availability_score = Column(Float)  # 0-1 scale
    
    # Relationships
    recipes = relationship(
        'Recipe',
        secondary=recipe_country_association,
        backref='countries'
    )
    local_ingredients = relationship('LocalIngredient', back_populates='country')
    
    def __repr__(self):
        return f"<Country(code='{self.code}', name='{self.name}')>"


class Cuisine(Base):
    """Cuisine types and characteristics"""
    __tablename__ = 'cuisines'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    origin_country_code = Column(String(2), ForeignKey('countries.code'))
    description = Column(Text)
    
    # Flavor profile
    flavor_profile = Column(JSON)  # {sweet: 3, salty: 5, sour: 2, bitter: 1, umami: 4}
    spice_level = Column(String(20))  # mild, medium, hot, very_hot
    common_flavor_combinations = Column(JSON)
    
    # Characteristics
    signature_dishes = Column(JSON)  # List of famous dishes
    key_ingredients = Column(JSON)  # Essential ingredients
    cooking_techniques = Column(JSON)  # Common techniques
    
    # Popularity
    global_popularity_rank = Column(Integer)
    fusion_cuisines = Column(JSON)  # Related fusion cuisines
    
    def __repr__(self):
        return f"<Cuisine(name='{self.name}')>"


# ============================================================================
# DISEASE & HEALTH MODELS
# ============================================================================

class DiseaseCondition(Base):
    """Medical conditions with dietary implications"""
    __tablename__ = 'disease_conditions'
    
    id = Column(Integer, primary_key=True)
    code = Column(String(50), unique=True, nullable=False)  # ICD-10 or custom code
    name = Column(String(200), nullable=False)
    category = Column(String(100))  # metabolic, cardiovascular, digestive, etc.
    severity_levels = Column(JSON)  # Available severity classifications
    
    # Dietary restrictions
    nutrients_to_limit = Column(JSON)  # {sodium: {max_mg: 2000, reason: "..."}}
    nutrients_to_increase = Column(JSON)  # {fiber: {min_g: 30, reason: "..."}}
    foods_to_avoid = Column(JSON)  # List of food IDs or categories
    foods_recommended = Column(JSON)  # List of beneficial foods
    
    # Nutritional targets
    calorie_adjustment_factor = Column(Float, default=1.0)  # Multiplier for base calories
    protein_ratio = Column(Float)  # Target % of calories from protein
    carb_ratio = Column(Float)  # Target % of calories from carbs
    fat_ratio = Column(Float)  # Target % of calories from fat
    
    # Specific limits
    sodium_max_mg = Column(Integer)
    potassium_max_mg = Column(Integer)
    phosphorus_max_mg = Column(Integer)
    sugar_max_g = Column(Float)
    saturated_fat_max_g = Column(Float)
    cholesterol_max_mg = Column(Integer)
    
    # Meal timing
    meal_frequency_recommendation = Column(String(100))
    fasting_requirements = Column(JSON)
    
    # Drug-nutrient interactions
    medication_interactions = Column(JSON)  # {drug_name: {nutrients: [...], warning: "..."}}
    
    # Clinical guidelines
    guidelines_source = Column(String(255))  # FDA, WHO, ADA, etc.
    last_updated = Column(DateTime, default=datetime.utcnow)
    clinical_notes = Column(Text)
    
    # Relationships
    restrictions = relationship('DiseaseRestriction', back_populates='disease')
    
    __table_args__ = (
        Index('idx_disease_category', 'category'),
        Index('idx_disease_code', 'code'),
    )
    
    def __repr__(self):
        return f"<DiseaseCondition(code='{self.code}', name='{self.name}')>"


class DiseaseRestriction(Base):
    """Specific dietary restrictions for diseases"""
    __tablename__ = 'disease_restrictions'
    
    id = Column(Integer, primary_key=True)
    disease_id = Column(Integer, ForeignKey('disease_conditions.id', ondelete='CASCADE'))
    restriction_type = Column(String(50))  # nutrient, food, category, allergen
    item_identifier = Column(String(200))  # Nutrient name, food ID, etc.
    
    # Limits
    max_value = Column(Float)
    min_value = Column(Float)
    unit = Column(String(20))  # mg, g, mcg, IU, etc.
    
    # Context
    severity_specific = Column(String(50))  # Which severity this applies to
    condition_notes = Column(Text)
    evidence_level = Column(String(20))  # strong, moderate, weak
    
    # Relationships
    disease = relationship('DiseaseCondition', back_populates='restrictions')
    
    def __repr__(self):
        return f"<DiseaseRestriction(disease_id={self.disease_id}, type='{self.restriction_type}')>"


# ============================================================================
# ALLERGEN MODELS
# ============================================================================

class Allergen(Base):
    """Food allergens and intolerances"""
    __tablename__ = 'allergens'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    allergen_type = Column(String(50))  # protein, carbohydrate, compound
    common_sources = Column(JSON)  # List of foods containing this allergen
    
    # Severity data
    anaphylaxis_risk = Column(Boolean, default=False)
    cross_reactivity = Column(JSON)  # Other allergens that may cross-react
    
    # Alternative names
    aliases = Column(JSON)  # Different names for the same allergen
    hidden_sources = Column(JSON)  # Less obvious sources
    
    def __repr__(self):
        return f"<Allergen(name='{self.name}')>"


# ============================================================================
# INGREDIENT & FOOD MODELS
# ============================================================================

class Ingredient(Base):
    """Global ingredient database"""
    __tablename__ = 'ingredients'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, index=True)
    scientific_name = Column(String(200))
    category = Column(String(100))  # vegetable, fruit, grain, protein, etc.
    subcategory = Column(String(100))
    
    # Nutritional data (per 100g)
    calories = Column(Float)
    protein_g = Column(Float)
    carbs_g = Column(Float)
    fiber_g = Column(Float)
    sugar_g = Column(Float)
    fat_g = Column(Float)
    saturated_fat_g = Column(Float)
    trans_fat_g = Column(Float)
    cholesterol_mg = Column(Float)
    sodium_mg = Column(Float)
    potassium_mg = Column(Float)
    calcium_mg = Column(Float)
    iron_mg = Column(Float)
    magnesium_mg = Column(Float)
    phosphorus_mg = Column(Float)
    zinc_mg = Column(Float)
    vitamin_a_iu = Column(Float)
    vitamin_c_mg = Column(Float)
    vitamin_d_iu = Column(Float)
    vitamin_e_mg = Column(Float)
    vitamin_k_mcg = Column(Float)
    vitamin_b6_mg = Column(Float)
    vitamin_b12_mcg = Column(Float)
    folate_mcg = Column(Float)
    
    # Additional properties
    glycemic_index = Column(Integer)
    glycemic_load = Column(Float)
    water_content_percent = Column(Float)
    
    # Allergens
    contains_allergens = Column(JSON)  # List of allergen IDs
    
    # Cost & availability
    average_cost_per_kg_usd = Column(Float)
    shelf_life_days = Column(Integer)
    seasonal = Column(Boolean, default=False)
    
    # Preparation
    preparation_methods = Column(JSON)  # raw, cooked, roasted, etc.
    cooking_time_minutes = Column(Integer)
    
    # Sustainability
    carbon_footprint_kg_co2 = Column(Float)
    water_footprint_liters = Column(Float)
    
    # Relationships
    recipes = relationship(
        'Recipe',
        secondary=recipe_ingredient_association,
        backref='ingredients'
    )
    substitutes = relationship('IngredientSubstitution', 
                             foreign_keys='IngredientSubstitution.original_ingredient_id',
                             back_populates='original_ingredient')
    
    __table_args__ = (
        Index('idx_ingredient_category', 'category'),
        Index('idx_ingredient_name', 'name'),
    )
    
    def __repr__(self):
        return f"<Ingredient(name='{self.name}', category='{self.category}')>"


class LocalIngredient(Base):
    """Country-specific local ingredients"""
    __tablename__ = 'local_ingredients'
    
    id = Column(Integer, primary_key=True)
    ingredient_id = Column(Integer, ForeignKey('ingredients.id', ondelete='CASCADE'))
    country_id = Column(Integer, ForeignKey('countries.id', ondelete='CASCADE'))
    
    # Local details
    local_name = Column(String(200))
    local_aliases = Column(JSON)  # Different regional names
    cultural_significance = Column(Text)
    
    # Availability
    available_months = Column(JSON)  # List of months [1-12]
    production_regions = Column(JSON)  # Regions within country
    local_cost_per_kg_usd = Column(Float)
    
    # Usage
    traditional_uses = Column(JSON)  # How it's traditionally used
    common_preparations = Column(JSON)  # Local preparation methods
    
    # Relationships
    ingredient = relationship('Ingredient')
    country = relationship('Country', back_populates='local_ingredients')
    
    __table_args__ = (
        UniqueConstraint('ingredient_id', 'country_id'),
        Index('idx_local_ingredient_country', 'country_id'),
    )
    
    def __repr__(self):
        return f"<LocalIngredient(id={self.id}, local_name='{self.local_name}')>"


class IngredientSubstitution(Base):
    """Ingredient substitution mappings"""
    __tablename__ = 'ingredient_substitutions'
    
    id = Column(Integer, primary_key=True)
    original_ingredient_id = Column(Integer, ForeignKey('ingredients.id', ondelete='CASCADE'))
    substitute_ingredient_id = Column(Integer, ForeignKey('ingredients.id', ondelete='CASCADE'))
    
    # Substitution details
    ratio = Column(Float, default=1.0)  # Amount of substitute per 1 unit original
    reason = Column(String(100))  # allergen, cost, availability, health, etc.
    quality_score = Column(Float)  # 0-1, how good the substitution is
    
    # Nutritional comparison
    calorie_difference_percent = Column(Float)
    protein_difference_percent = Column(Float)
    
    # Notes
    preparation_differences = Column(Text)
    taste_impact = Column(String(200))
    
    # Relationships
    original_ingredient = relationship('Ingredient', 
                                      foreign_keys=[original_ingredient_id],
                                      back_populates='substitutes')
    substitute_ingredient = relationship('Ingredient', foreign_keys=[substitute_ingredient_id])
    
    __table_args__ = (
        CheckConstraint('original_ingredient_id != substitute_ingredient_id'),
    )
    
    def __repr__(self):
        return f"<IngredientSubstitution(id={self.id}, ratio={self.ratio})>"


# ============================================================================
# RECIPE MODELS
# ============================================================================

class Recipe(Base):
    """Recipe database with nutritional data"""
    __tablename__ = 'recipes'
    
    id = Column(Integer, primary_key=True)
    recipe_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    cuisine_id = Column(Integer, ForeignKey('cuisines.id'))
    
    # Recipe details
    meal_type = Column(SQLEnum(MealType))
    servings = Column(Integer, default=4)
    prep_time_minutes = Column(Integer)
    cook_time_minutes = Column(Integer)
    total_time_minutes = Column(Integer)
    difficulty_level = Column(String(20))  # easy, medium, hard
    
    # Instructions
    instructions = Column(JSON)  # List of step-by-step instructions
    cooking_methods = Column(JSON)  # boiling, frying, baking, etc.
    
    # Nutritional data (per serving)
    calories_per_serving = Column(Float)
    protein_g_per_serving = Column(Float)
    carbs_g_per_serving = Column(Float)
    fiber_g_per_serving = Column(Float)
    sugar_g_per_serving = Column(Float)
    fat_g_per_serving = Column(Float)
    saturated_fat_g_per_serving = Column(Float)
    sodium_mg_per_serving = Column(Float)
    potassium_mg_per_serving = Column(Float)
    cholesterol_mg_per_serving = Column(Float)
    
    # Additional nutritional info
    glycemic_index = Column(Integer)
    glycemic_load = Column(Float)
    
    # Dietary compatibility
    is_vegetarian = Column(Boolean, default=False)
    is_vegan = Column(Boolean, default=False)
    is_gluten_free = Column(Boolean, default=False)
    is_dairy_free = Column(Boolean, default=False)
    is_halal = Column(Boolean, default=False)
    is_kosher = Column(Boolean, default=False)
    dietary_tags = Column(JSON)  # Additional tags
    
    # Disease compatibility
    disease_safe_for = Column(JSON)  # List of disease codes this is safe for
    disease_warnings = Column(JSON)  # {disease_code: "warning message"}
    
    # Flavor profile
    flavor_profile = Column(JSON)  # {sweet: 2, salty: 4, sour: 1, bitter: 0, umami: 3}
    spice_level = Column(String(20))
    
    # Cost & sustainability
    estimated_cost_usd = Column(Float)
    sustainability_score = Column(Float)  # 0-1 scale
    
    # Media
    image_url = Column(String(500))
    video_url = Column(String(500))
    
    # Metadata
    source = Column(String(200))  # cookbook, website, user-generated, etc.
    author = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Popularity
    rating_avg = Column(Float)
    rating_count = Column(Integer, default=0)
    view_count = Column(Integer, default=0)
    
    # Relationships
    cuisine = relationship('Cuisine')
    meal_items = relationship('MealPlanItem', back_populates='recipe')
    ratings = relationship('MealRating', back_populates='recipe')
    
    __table_args__ = (
        Index('idx_recipe_cuisine', 'cuisine_id'),
        Index('idx_recipe_meal_type', 'meal_type'),
        Index('idx_recipe_difficulty', 'difficulty_level'),
    )
    
    def __repr__(self):
        return f"<Recipe(id={self.id}, name='{self.name}')>"


# ============================================================================
# MEAL PLAN MODELS
# ============================================================================

class MealPlan(Base):
    """User meal plans"""
    __tablename__ = 'meal_plans'
    
    id = Column(Integer, primary_key=True)
    plan_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    # Plan details
    name = Column(String(200))
    description = Column(Text)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    duration_days = Column(Integer, nullable=False)
    
    # Targets
    daily_calorie_target = Column(Integer)
    protein_target_g = Column(Float)
    carb_target_g = Column(Float)
    fat_target_g = Column(Float)
    
    # Configuration
    country_code = Column(String(2))
    local_food_priority = Column(String(20))  # low, medium, high
    budget_per_day_usd = Column(Float)
    
    # Generation metadata
    generation_algorithm = Column(String(100))
    ml_model_version = Column(String(50))
    personalization_score = Column(Float)  # How well it matches user preferences
    
    # Status
    status = Column(SQLEnum(PlanStatus), default=PlanStatus.DRAFT)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Adherence tracking
    adherence_percentage = Column(Float)
    meals_completed = Column(Integer, default=0)
    meals_skipped = Column(Integer, default=0)
    
    # Outcomes
    weight_change_kg = Column(Float)
    health_metrics_change = Column(JSON)  # Changes in blood pressure, glucose, etc.
    
    # Relationships
    user = relationship('User', back_populates='meal_plans')
    meal_items = relationship('MealPlanItem', back_populates='meal_plan', cascade='all, delete-orphan')
    
    __table_args__ = (
        Index('idx_meal_plan_user', 'user_id'),
        Index('idx_meal_plan_dates', 'start_date', 'end_date'),
        Index('idx_meal_plan_status', 'status'),
    )
    
    def __repr__(self):
        return f"<MealPlan(id={self.id}, name='{self.name}', user_id={self.user_id})>"


class MealPlanItem(Base):
    """Individual meals within a meal plan"""
    __tablename__ = 'meal_plan_items'
    
    id = Column(Integer, primary_key=True)
    meal_plan_id = Column(Integer, ForeignKey('meal_plans.id', ondelete='CASCADE'), nullable=False)
    recipe_id = Column(Integer, ForeignKey('recipes.id', ondelete='SET NULL'))
    
    # Meal details
    day_number = Column(Integer, nullable=False)  # 1-based day in plan
    meal_type = Column(SQLEnum(MealType), nullable=False)
    scheduled_time = Column(String(10))  # HH:MM format
    
    # Portions
    servings = Column(Float, default=1.0)
    portion_adjustment_factor = Column(Float, default=1.0)
    
    # Actual nutritional values (adjusted for servings)
    calories = Column(Float)
    protein_g = Column(Float)
    carbs_g = Column(Float)
    fat_g = Column(Float)
    
    # Status
    completed = Column(Boolean, default=False)
    completed_at = Column(DateTime)
    skipped = Column(Boolean, default=False)
    skip_reason = Column(String(200))
    
    # Substitutions made
    ingredient_substitutions = Column(JSON)  # {original_id: substitute_id}
    
    # Notes
    preparation_notes = Column(Text)
    user_notes = Column(Text)
    
    # Relationships
    meal_plan = relationship('MealPlan', back_populates='meal_items')
    recipe = relationship('Recipe', back_populates='meal_items')
    
    __table_args__ = (
        Index('idx_meal_item_plan', 'meal_plan_id'),
        Index('idx_meal_item_day', 'meal_plan_id', 'day_number'),
    )
    
    def __repr__(self):
        return f"<MealPlanItem(id={self.id}, day={self.day_number}, type={self.meal_type})>"


class MealRating(Base):
    """User ratings and feedback for meals"""
    __tablename__ = 'meal_ratings'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    recipe_id = Column(Integer, ForeignKey('recipes.id', ondelete='CASCADE'), nullable=False)
    
    # Ratings (1-5 scale)
    overall_rating = Column(Integer)
    taste_rating = Column(Integer)
    difficulty_rating = Column(Integer)
    value_rating = Column(Integer)  # Cost vs quality
    
    # Feedback
    would_make_again = Column(Boolean)
    comments = Column(Text)
    tags = Column(JSON)  # too_salty, too_bland, perfect, etc.
    
    # Context
    meal_date = Column(DateTime)
    meal_type = Column(SQLEnum(MealType))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='meal_ratings')
    recipe = relationship('Recipe', back_populates='ratings')
    
    __table_args__ = (
        Index('idx_rating_user', 'user_id'),
        Index('idx_rating_recipe', 'recipe_id'),
        CheckConstraint('overall_rating >= 1 AND overall_rating <= 5'),
    )
    
    def __repr__(self):
        return f"<MealRating(id={self.id}, rating={self.overall_rating})>"


# ============================================================================
# SHOPPING LIST MODELS
# ============================================================================

class ShoppingList(Base):
    """Generated shopping lists for meal plans"""
    __tablename__ = 'shopping_lists'
    
    id = Column(Integer, primary_key=True)
    meal_plan_id = Column(Integer, ForeignKey('meal_plans.id', ondelete='CASCADE'))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # List details
    total_estimated_cost_usd = Column(Float)
    items_count = Column(Integer)
    
    # Organization
    grouped_by_category = Column(JSON)  # {category: [items]}
    grouped_by_store_section = Column(JSON)  # {section: [items]}
    
    # Status
    completed = Column(Boolean, default=False)
    completed_at = Column(DateTime)
    
    # Relationships
    meal_plan = relationship('MealPlan')
    items = relationship('ShoppingListItem', back_populates='shopping_list', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<ShoppingList(id={self.id}, items={self.items_count})>"


class ShoppingListItem(Base):
    """Individual items in shopping list"""
    __tablename__ = 'shopping_list_items'
    
    id = Column(Integer, primary_key=True)
    shopping_list_id = Column(Integer, ForeignKey('shopping_lists.id', ondelete='CASCADE'))
    ingredient_id = Column(Integer, ForeignKey('ingredients.id', ondelete='CASCADE'))
    
    # Quantity
    quantity = Column(Float, nullable=False)
    unit = Column(String(50), nullable=False)
    
    # Details
    category = Column(String(100))
    store_section = Column(String(100))  # produce, dairy, meat, etc.
    estimated_cost_usd = Column(Float)
    
    # Status
    checked = Column(Boolean, default=False)
    
    # Alternatives
    acceptable_substitutes = Column(JSON)  # List of substitute ingredient IDs
    
    # Relationships
    shopping_list = relationship('ShoppingList', back_populates='items')
    ingredient = relationship('Ingredient')
    
    def __repr__(self):
        return f"<ShoppingListItem(id={self.id}, quantity={self.quantity} {self.unit})>"


# ============================================================================
# FLAVOR & PREFERENCE MODELS
# ============================================================================

class FlavorProfile(Base):
    """Flavor profiles for foods and recipes"""
    __tablename__ = 'flavor_profiles'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    
    # Basic tastes (0-10 scale)
    sweet = Column(Float, default=0)
    salty = Column(Float, default=0)
    sour = Column(Float, default=0)
    bitter = Column(Float, default=0)
    umami = Column(Float, default=0)
    
    # Additional characteristics
    spicy = Column(Float, default=0)
    fatty = Column(Float, default=0)
    astringent = Column(Float, default=0)
    
    # Texture
    texture_tags = Column(JSON)  # crispy, creamy, crunchy, smooth, etc.
    
    # Aroma
    aroma_compounds = Column(JSON)  # Key aroma compounds
    
    def __repr__(self):
        return f"<FlavorProfile(name='{self.name}')>"


# ============================================================================
# ANALYTICS & TRACKING MODELS
# ============================================================================

class NutritionLog(Base):
    """Daily nutrition tracking"""
    __tablename__ = 'nutrition_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    log_date = Column(DateTime, nullable=False)
    
    # Totals for the day
    total_calories = Column(Float)
    total_protein_g = Column(Float)
    total_carbs_g = Column(Float)
    total_fat_g = Column(Float)
    total_fiber_g = Column(Float)
    total_sodium_mg = Column(Float)
    total_sugar_g = Column(Float)
    
    # Water intake
    water_ml = Column(Float)
    
    # Compliance
    met_calorie_target = Column(Boolean)
    met_protein_target = Column(Boolean)
    within_sodium_limit = Column(Boolean)
    
    # Notes
    notes = Column(Text)
    energy_level = Column(Integer)  # 1-5 scale
    hunger_level = Column(Integer)  # 1-5 scale
    
    # Relationships
    user = relationship('User')
    
    __table_args__ = (
        Index('idx_nutrition_log_user_date', 'user_id', 'log_date'),
        UniqueConstraint('user_id', 'log_date'),
    )
    
    def __repr__(self):
        return f"<NutritionLog(user_id={self.user_id}, date={self.log_date})>"


class HealthMetric(Base):
    """Health metrics tracking over time"""
    __tablename__ = 'health_metrics'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    measured_at = Column(DateTime, nullable=False)
    
    # Weight & body composition
    weight_kg = Column(Float)
    body_fat_percent = Column(Float)
    muscle_mass_kg = Column(Float)
    bmi = Column(Float)
    
    # Vital signs
    blood_pressure_systolic = Column(Integer)
    blood_pressure_diastolic = Column(Integer)
    resting_heart_rate = Column(Integer)
    
    # Blood markers
    blood_glucose_mg_dl = Column(Float)
    hba1c_percent = Column(Float)
    cholesterol_total_mg_dl = Column(Float)
    cholesterol_ldl_mg_dl = Column(Float)
    cholesterol_hdl_mg_dl = Column(Float)
    triglycerides_mg_dl = Column(Float)
    
    # Other markers
    creatinine_mg_dl = Column(Float)
    gfr_ml_min = Column(Float)  # Kidney function
    
    # Context
    fasting = Column(Boolean)
    medication_taken = Column(Boolean)
    notes = Column(Text)
    
    # Relationships
    user = relationship('User')
    
    __table_args__ = (
        Index('idx_health_metric_user_date', 'user_id', 'measured_at'),
    )
    
    def __repr__(self):
        return f"<HealthMetric(user_id={self.user_id}, date={self.measured_at})>"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(engine)


def drop_tables(engine):
    """Drop all tables from the database"""
    Base.metadata.drop_all(engine)


# Export all models
__all__ = [
    'Base',
    'User',
    'Country',
    'Cuisine',
    'DiseaseCondition',
    'DiseaseRestriction',
    'Allergen',
    'Ingredient',
    'LocalIngredient',
    'IngredientSubstitution',
    'Recipe',
    'MealPlan',
    'MealPlanItem',
    'MealRating',
    'ShoppingList',
    'ShoppingListItem',
    'FlavorProfile',
    'NutritionLog',
    'HealthMetric',
    'ActivityLevel',
    'Gender',
    'MealType',
    'DietaryRestriction',
    'DiseaseSeverity',
    'PlanStatus',
]
