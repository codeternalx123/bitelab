"""
Food Knowledge Graph Models
==========================

Comprehensive data models for the food knowledge graph system supporting
millions of foods with country-specific information, nutritional data,
and complex relationships.

Core Models:
- Food entities with hierarchical taxonomy
- Country-specific food variations
- Nutritional profiles with molecular details
- Cultural and regional associations
- Preparation methods and cooking techniques
- Seasonal availability and sourcing
- Allergen and dietary restriction tracking
- Supply chain and sustainability metrics

Author: Wellomex AI Nutrition Team
Version: 1.0.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Union, Any
from uuid import UUID, uuid4
import json

class FoodCategory(Enum):
    """Primary food categories for taxonomy classification"""
    GRAINS = "grains"
    VEGETABLES = "vegetables"
    FRUITS = "fruits"
    PROTEINS = "proteins"
    DAIRY = "dairy"
    NUTS_SEEDS = "nuts_seeds"
    LEGUMES = "legumes"
    HERBS_SPICES = "herbs_spices"
    OILS_FATS = "oils_fats"
    BEVERAGES = "beverages"
    SEAFOOD = "seafood"
    MEAT = "meat"
    POULTRY = "poultry"
    FUNGI = "fungi"
    SWEETENERS = "sweeteners"
    PROCESSED = "processed"
    FERMENTED = "fermented"
    BAKED_GOODS = "baked_goods"
    CONDIMENTS = "condiments"
    SNACKS = "snacks"

class NutritionalProfile(Enum):
    """Nutritional profile classifications"""
    HIGH_PROTEIN = "high_protein"
    LOW_CARB = "low_carb"
    HIGH_FIBER = "high_fiber"
    LOW_SODIUM = "low_sodium"
    ANTIOXIDANT_RICH = "antioxidant_rich"
    OMEGA3_RICH = "omega3_rich"
    PROBIOTIC = "probiotic"
    SUPERFOOD = "superfood"
    CALORIE_DENSE = "calorie_dense"
    NUTRIENT_DENSE = "nutrient_dense"

class PreparationMethod(Enum):
    """Food preparation and cooking methods"""
    RAW = "raw"
    BOILED = "boiled"
    STEAMED = "steamed"
    GRILLED = "grilled"
    ROASTED = "roasted"
    FRIED = "fried"
    BAKED = "baked"
    SAUTEED = "sauteed"
    FERMENTED = "fermented"
    PICKLED = "pickled"
    SMOKED = "smoked"
    DRIED = "dried"
    CANNED = "canned"
    FROZEN = "frozen"
    PROCESSED = "processed"

class AllergenType(Enum):
    """Common food allergens"""
    GLUTEN = "gluten"
    DAIRY = "dairy"
    EGGS = "eggs"
    NUTS = "nuts"
    PEANUTS = "peanuts"
    SHELLFISH = "shellfish"
    FISH = "fish"
    SOY = "soy"
    SESAME = "sesame"
    SULFITES = "sulfites"

class DietaryRestriction(Enum):
    """Dietary restriction categories"""
    VEGAN = "vegan"
    VEGETARIAN = "vegetarian"
    PESCATARIAN = "pescatarian"
    KETO = "keto"
    PALEO = "paleo"
    MEDITERRANEAN = "mediterranean"
    HALAL = "halal"
    KOSHER = "kosher"
    LOW_FODMAP = "low_fodmap"
    DIABETIC_FRIENDLY = "diabetic_friendly"

class SeasonalAvailability(Enum):
    """Seasonal availability patterns"""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"
    YEAR_ROUND = "year_round"

@dataclass
class MacroNutrient:
    """Macronutrient information per 100g"""
    calories: Decimal
    carbohydrates: Decimal  # grams
    protein: Decimal        # grams
    fat: Decimal           # grams
    fiber: Decimal         # grams
    sugar: Decimal         # grams
    sodium: Decimal        # milligrams
    
    # Fat breakdown
    saturated_fat: Optional[Decimal] = None
    monounsaturated_fat: Optional[Decimal] = None
    polyunsaturated_fat: Optional[Decimal] = None
    trans_fat: Optional[Decimal] = None
    
    # Carb breakdown
    starch: Optional[Decimal] = None
    added_sugars: Optional[Decimal] = None
    
    # Additional macros
    cholesterol: Optional[Decimal] = None  # milligrams
    water_content: Optional[Decimal] = None  # grams

@dataclass
class MicroNutrient:
    """Micronutrient information per 100g"""
    # Vitamins (in various units)
    vitamin_a: Optional[Decimal] = None    # mcg RAE
    vitamin_c: Optional[Decimal] = None    # mg
    vitamin_d: Optional[Decimal] = None    # mcg
    vitamin_e: Optional[Decimal] = None    # mg
    vitamin_k: Optional[Decimal] = None    # mcg
    thiamine_b1: Optional[Decimal] = None  # mg
    riboflavin_b2: Optional[Decimal] = None # mg
    niacin_b3: Optional[Decimal] = None    # mg
    pantothenic_acid_b5: Optional[Decimal] = None # mg
    pyridoxine_b6: Optional[Decimal] = None # mg
    biotin_b7: Optional[Decimal] = None    # mcg
    folate_b9: Optional[Decimal] = None    # mcg
    cobalamin_b12: Optional[Decimal] = None # mcg
    
    # Minerals (in mg unless noted)
    calcium: Optional[Decimal] = None
    iron: Optional[Decimal] = None
    magnesium: Optional[Decimal] = None
    phosphorus: Optional[Decimal] = None
    potassium: Optional[Decimal] = None
    zinc: Optional[Decimal] = None
    copper: Optional[Decimal] = None      # mg
    manganese: Optional[Decimal] = None   # mg
    selenium: Optional[Decimal] = None    # mcg
    chromium: Optional[Decimal] = None    # mcg
    molybdenum: Optional[Decimal] = None  # mcg
    iodine: Optional[Decimal] = None      # mcg
    
    # Additional compounds
    omega3_fatty_acids: Optional[Decimal] = None  # mg
    omega6_fatty_acids: Optional[Decimal] = None  # mg
    antioxidants_orac: Optional[Decimal] = None   # units

@dataclass
class PhytochemicalProfile:
    """Plant-based bioactive compounds"""
    flavonoids: Dict[str, Decimal] = field(default_factory=dict)
    phenolic_acids: Dict[str, Decimal] = field(default_factory=dict)
    carotenoids: Dict[str, Decimal] = field(default_factory=dict)
    glucosinolates: Dict[str, Decimal] = field(default_factory=dict)
    alkaloids: Dict[str, Decimal] = field(default_factory=dict)
    terpenes: Dict[str, Decimal] = field(default_factory=dict)
    
    # Specific important compounds
    lycopene: Optional[Decimal] = None
    beta_carotene: Optional[Decimal] = None
    lutein: Optional[Decimal] = None
    zeaxanthin: Optional[Decimal] = None
    anthocyanins: Optional[Decimal] = None
    resveratrol: Optional[Decimal] = None
    quercetin: Optional[Decimal] = None
    catechins: Optional[Decimal] = None

@dataclass
class CulturalContext:
    """Cultural and regional food context"""
    origin_countries: List[str] = field(default_factory=list)
    traditional_uses: List[str] = field(default_factory=list)
    cultural_significance: str = ""
    religious_associations: List[str] = field(default_factory=list)
    festival_foods: List[str] = field(default_factory=list)
    preparation_traditions: Dict[str, str] = field(default_factory=dict)
    regional_names: Dict[str, str] = field(default_factory=dict)  # country -> local name
    cuisine_styles: List[str] = field(default_factory=list)

@dataclass
class SustainabilityMetrics:
    """Environmental and sustainability data"""
    carbon_footprint: Optional[Decimal] = None      # kg CO2 per kg food
    water_usage: Optional[Decimal] = None           # liters per kg
    land_usage: Optional[Decimal] = None            # m² per kg
    biodiversity_impact: Optional[str] = None        # impact rating
    packaging_recyclability: Optional[str] = None
    transport_distance: Optional[Decimal] = None     # average km
    seasonal_score: Optional[Decimal] = None         # 0-10 seasonal appropriateness
    organic_availability: bool = False
    fair_trade_available: bool = False
    local_sourcing_potential: Optional[Decimal] = None # 0-10 rating

@dataclass
class CountrySpecificData:
    """Country-specific food information"""
    country_code: str  # ISO 3166-1 alpha-2
    local_name: str
    common_varieties: List[str] = field(default_factory=list)
    regional_variations: Dict[str, str] = field(default_factory=dict)
    seasonal_availability: Dict[SeasonalAvailability, bool] = field(default_factory=dict)
    price_range: Optional[Dict[str, Decimal]] = None  # min, max, avg per kg/lb
    production_regions: List[str] = field(default_factory=list)
    import_sources: List[str] = field(default_factory=list)
    quality_grades: List[str] = field(default_factory=list)
    traditional_preparations: List[PreparationMethod] = field(default_factory=list)
    cultural_context: Optional[CulturalContext] = None
    sustainability_metrics: Optional[SustainabilityMetrics] = None
    market_availability: Decimal = Decimal('1.0')  # 0-1 availability score

@dataclass
class FoodEntity:
    """Core food entity with comprehensive information"""
    # Identifiers
    food_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    scientific_name: Optional[str] = None
    common_names: List[str] = field(default_factory=list)
    
    # Classification
    category: FoodCategory = FoodCategory.PROCESSED
    subcategories: List[str] = field(default_factory=list)
    food_group: str = ""
    taxonomic_family: Optional[str] = None
    
    # Nutritional Information
    macro_nutrients: Optional[MacroNutrient] = None
    micro_nutrients: Optional[MicroNutrient] = None
    phytochemicals: Optional[PhytochemicalProfile] = None
    nutritional_profiles: List[NutritionalProfile] = field(default_factory=list)
    
    # Safety and Restrictions
    allergens: List[AllergenType] = field(default_factory=list)
    dietary_restrictions: List[DietaryRestriction] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    drug_interactions: List[str] = field(default_factory=list)
    
    # Physical Properties
    typical_serving_size: Optional[Decimal] = None  # grams
    density: Optional[Decimal] = None  # g/ml
    ph_level: Optional[Decimal] = None
    glycemic_index: Optional[int] = None
    glycemic_load: Optional[Decimal] = None
    
    # Storage and Handling
    storage_temperature: Optional[str] = None
    shelf_life: Optional[int] = None  # days
    spoilage_indicators: List[str] = field(default_factory=list)
    handling_requirements: List[str] = field(default_factory=list)
    
    # Country-Specific Data
    country_data: Dict[str, CountrySpecificData] = field(default_factory=dict)
    
    # Preparation and Usage
    preparation_methods: List[PreparationMethod] = field(default_factory=list)
    cooking_time_ranges: Dict[PreparationMethod, Dict[str, int]] = field(default_factory=dict)
    flavor_profile: Dict[str, Decimal] = field(default_factory=dict)  # sweet, salty, etc.
    texture_properties: Dict[str, str] = field(default_factory=dict)
    
    # Relationships
    parent_food_id: Optional[str] = None  # for variations/processed versions
    related_foods: List[str] = field(default_factory=list)
    substitute_foods: List[str] = field(default_factory=list)
    complementary_foods: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    data_sources: List[str] = field(default_factory=list)
    confidence_score: Decimal = Decimal('1.0')  # 0-1 data reliability
    verification_status: str = "pending"  # pending, verified, disputed
    
    # API Integration
    external_ids: Dict[str, str] = field(default_factory=dict)  # API -> ID mapping
    last_api_sync: Optional[datetime] = None
    api_update_frequency: Optional[int] = None  # days between updates

@dataclass
class FoodRelationship:
    """Relationships between food entities"""
    relationship_id: str = field(default_factory=lambda: str(uuid4()))
    from_food_id: str = ""
    to_food_id: str = ""
    relationship_type: str = ""  # substitute, complement, ingredient, variant, etc.
    strength: Decimal = Decimal('1.0')  # 0-1 relationship strength
    context: Optional[str] = None  # cooking, nutrition, cultural, etc.
    country_specific: Optional[str] = None  # ISO country code if applicable
    confidence: Decimal = Decimal('1.0')
    created_at: datetime = field(default_factory=datetime.utcnow)
    data_source: Optional[str] = None

@dataclass
class NutrientInteraction:
    """Nutrient interactions and bioavailability"""
    interaction_id: str = field(default_factory=lambda: str(uuid4()))
    nutrient_a: str = ""
    nutrient_b: str = ""
    interaction_type: str = ""  # enhances, inhibits, competes, synergistic
    effect_magnitude: Decimal = Decimal('1.0')  # multiplier effect
    conditions: List[str] = field(default_factory=list)  # pH, temperature, etc.
    food_context: Optional[str] = None  # specific food combinations
    evidence_level: str = "low"  # low, medium, high, clinical
    source_studies: List[str] = field(default_factory=list)

@dataclass
class CookingTransformation:
    """How cooking methods affect nutritional content"""
    transformation_id: str = field(default_factory=lambda: str(uuid4()))
    food_id: str = ""
    preparation_method: PreparationMethod = PreparationMethod.RAW
    cooking_time: Optional[int] = None  # minutes
    temperature: Optional[int] = None  # celsius
    
    # Nutritional changes (multipliers)
    nutrient_retention: Dict[str, Decimal] = field(default_factory=dict)
    new_compounds_formed: Dict[str, Decimal] = field(default_factory=dict)
    texture_changes: Dict[str, str] = field(default_factory=dict)
    digestibility_change: Optional[Decimal] = None  # multiplier
    glycemic_impact: Optional[Decimal] = None  # GI change

@dataclass
class SeasonalNutritionVariation:
    """Seasonal variations in nutritional content"""
    food_id: str = ""
    season: SeasonalAvailability = SeasonalAvailability.YEAR_ROUND
    country_code: str = ""
    nutrient_variations: Dict[str, Decimal] = field(default_factory=dict)  # % change
    harvest_conditions: Optional[str] = None
    storage_effects: Optional[str] = None
    peak_nutrition_period: Optional[str] = None

@dataclass
class FoodSafetyData:
    """Food safety and contamination information"""
    food_id: str = ""
    common_pathogens: List[str] = field(default_factory=list)
    contamination_risks: List[str] = field(default_factory=list)
    safe_handling_practices: List[str] = field(default_factory=list)
    critical_temperatures: Dict[str, int] = field(default_factory=dict)
    high_risk_populations: List[str] = field(default_factory=list)
    regulatory_standards: Dict[str, str] = field(default_factory=dict)  # country -> standard
    recall_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class APISourceMapping:
    """Mapping between internal food IDs and external API identifiers"""
    mapping_id: str = field(default_factory=lambda: str(uuid4()))
    internal_food_id: str = ""
    api_provider: str = ""  # USDA, OpenFoodFacts, etc.
    external_id: str = ""
    external_name: str = ""
    mapping_confidence: Decimal = Decimal('1.0')
    field_mappings: Dict[str, str] = field(default_factory=dict)  # internal -> external field
    last_sync: Optional[datetime] = None
    sync_frequency: int = 7  # days
    active: bool = True

@dataclass
class DataQualityMetrics:
    """Data quality and completeness metrics"""
    food_id: str = ""
    completeness_score: Decimal = Decimal('0.0')  # 0-1
    accuracy_score: Decimal = Decimal('0.0')  # 0-1
    freshness_score: Decimal = Decimal('0.0')  # 0-1 based on last update
    source_reliability: Decimal = Decimal('0.0')  # 0-1
    
    # Field-specific quality
    nutrition_completeness: Decimal = Decimal('0.0')
    cultural_data_completeness: Decimal = Decimal('0.0')
    safety_data_completeness: Decimal = Decimal('0.0')
    relationship_completeness: Decimal = Decimal('0.0')
    
    # Validation flags
    has_conflicts: bool = False
    needs_review: bool = False
    expert_verified: bool = False
    community_verified: bool = False
    
    last_quality_check: datetime = field(default_factory=datetime.utcnow)
    quality_issues: List[str] = field(default_factory=list)

# Helper functions for model operations
def create_food_entity(name: str, category: FoodCategory, **kwargs) -> FoodEntity:
    """Create a new food entity with default values"""
    food = FoodEntity(name=name, category=category, **kwargs)
    return food

def add_country_data(food: FoodEntity, country_code: str, local_name: str, **kwargs) -> FoodEntity:
    """Add country-specific data to a food entity"""
    country_data = CountrySpecificData(
        country_code=country_code,
        local_name=local_name,
        **kwargs
    )
    food.country_data[country_code] = country_data
    return food

def calculate_nutrition_density(macro: MacroNutrient, micro: MicroNutrient) -> Decimal:
    """Calculate nutrient density score"""
    if macro.calories == 0:
        return Decimal('0')
    
    # Simple density calculation - can be enhanced with weighting
    nutrient_sum = Decimal('0')
    nutrient_count = 0
    
    # Count non-null micronutrients
    for field_name, value in micro.__dict__.items():
        if value is not None and value > 0:
            nutrient_sum += Decimal(str(value))
            nutrient_count += 1
    
    if nutrient_count == 0:
        return Decimal('0')
    
    return (nutrient_sum / nutrient_count) / macro.calories

def serialize_food_entity(food: FoodEntity) -> Dict[str, Any]:
    """Serialize food entity to JSON-compatible dictionary"""
    def decimal_converter(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        return obj
    
    food_dict = asdict(food)
    return json.loads(json.dumps(food_dict, default=decimal_converter))

def deserialize_food_entity(data: Dict[str, Any]) -> FoodEntity:
    """Deserialize dictionary back to food entity"""
    # This would need proper implementation based on the data structure
    # For now, returning a basic implementation
    return FoodEntity(**data)