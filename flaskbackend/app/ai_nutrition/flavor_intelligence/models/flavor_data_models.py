"""
Advanced Flavor Intelligence Data Models
=======================================

This module contains comprehensive data models for the Automated Flavor Intelligence Pipeline.
It implements the three-layer data architecture for flavor analysis:
- Layer A: Sensory Layer (The "Palate") - 0.0-1.0 taste intensities
- Layer B: Molecular Layer (The "Chemistry") - Chemical compounds and structures
- Layer C: Relational Layer (The "Wisdom") - Ingredient compatibility and pairings

The models support millions of flavor profiles with automated data ingestion from multiple sources.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Union, Any
from enum import Enum, auto
import numpy as np
from datetime import datetime
import json
import hashlib
from collections import defaultdict
import uuid


class FlavorCategory(Enum):
    """Primary flavor categories for ingredient classification"""
    SWEET = "sweet"
    SAVORY = "savory" 
    SPICY = "spicy"
    AROMATIC = "aromatic"
    ACIDIC = "acidic"
    BITTER = "bitter"
    UMAMI = "umami"
    FATTY = "fatty"
    FRESH = "fresh"
    EARTHY = "earthy"
    FLORAL = "floral"
    FRUITY = "fruity"
    NUTTY = "nutty"
    WOODY = "woody"
    SMOKY = "smoky"


class TasteProfile(Enum):
    """Basic taste profiles mapped to sensory perception"""
    SWEET = "sweet"
    SOUR = "sour"
    SALTY = "salty"  
    BITTER = "bitter"
    UMAMI = "umami"
    FATTY = "fatty"
    SPICY = "spicy"
    ASTRINGENT = "astringent"
    COOLING = "cooling"
    WARMING = "warming"


class MolecularClass(Enum):
    """Chemical compound classifications for molecular analysis"""
    TERPENE = "terpene"
    ESTER = "ester"
    ALDEHYDE = "aldehyde"
    KETONE = "ketone"
    ALCOHOL = "alcohol"
    ACID = "acid"
    PHENOL = "phenol"
    ALKALOID = "alkaloid"
    GLYCOSIDE = "glycoside"
    LACTONE = "lactone"
    PYRAZINE = "pyrazine"
    THIAZOLE = "thiazole"
    FURAN = "furan"
    SULFUR_COMPOUND = "sulfur_compound"
    NITROGEN_COMPOUND = "nitrogen_compound"


class CompatibilityLevel(Enum):
    """Ingredient pairing compatibility levels"""
    EXCELLENT = 5
    VERY_GOOD = 4
    GOOD = 3
    NEUTRAL = 2
    POOR = 1
    INCOMPATIBLE = 0


class DataSource(Enum):
    """Sources for flavor data collection"""
    OPENFOODFACTS = "openfoodfacts"
    USDA = "usda"
    FLAVORDB = "flavordb"
    RECIPE1M = "recipe1m"
    WIKIDATA = "wikidata"
    PUBCHEM = "pubchem"
    SPOONACULAR = "spoonacular"
    FOODCOM = "food_com"
    LLM_AUGMENTED = "llm_augmented"
    MANUAL_ENTRY = "manual_entry"


@dataclass
class SensoryProfile:
    """
    Layer A: The Sensory Layer (The "Palate")
    Represents taste intensities as perceived by human senses
    All values normalized to 0.0-1.0 scale
    """
    # Basic taste intensities (0.0 = none, 1.0 = maximum)
    sweet: float = 0.0
    sour: float = 0.0
    salty: float = 0.0
    bitter: float = 0.0
    umami: float = 0.0
    fatty: float = 0.0
    spicy: float = 0.0
    
    # Extended sensory properties
    astringent: float = 0.0  # Tannins, dryness
    cooling: float = 0.0     # Menthol, mint effect
    warming: float = 0.0     # Ginger, cinnamon effect
    numbing: float = 0.0     # Sichuan pepper effect
    
    # Aromatic intensity
    aromatic: float = 0.0
    
    # Textural properties
    creaminess: float = 0.0
    crispness: float = 0.0
    juiciness: float = 0.0
    
    # Confidence scores for each measurement
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    # Data source information
    source: DataSource = DataSource.MANUAL_ENTRY
    measured_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate sensory profile values"""
        sensory_attrs = ['sweet', 'sour', 'salty', 'bitter', 'umami', 'fatty', 
                        'spicy', 'astringent', 'cooling', 'warming', 'numbing',
                        'aromatic', 'creaminess', 'crispness', 'juiciness']
        
        for attr in sensory_attrs:
            value = getattr(self, attr)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Sensory value '{attr}' must be between 0.0 and 1.0, got {value}")
    
    def to_vector(self) -> np.ndarray:
        """Convert sensory profile to numpy vector for ML operations"""
        return np.array([
            self.sweet, self.sour, self.salty, self.bitter, self.umami,
            self.fatty, self.spicy, self.astringent, self.cooling, 
            self.warming, self.numbing, self.aromatic, self.creaminess,
            self.crispness, self.juiciness
        ])
    
    def similarity(self, other: 'SensoryProfile') -> float:
        """Calculate cosine similarity with another sensory profile"""
        vec1 = self.to_vector()
        vec2 = other.to_vector()
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def dominant_tastes(self, threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Get dominant taste components above threshold"""
        tastes = [
            ('sweet', self.sweet), ('sour', self.sour), ('salty', self.salty),
            ('bitter', self.bitter), ('umami', self.umami), ('fatty', self.fatty),
            ('spicy', self.spicy), ('aromatic', self.aromatic)
        ]
        return [(taste, intensity) for taste, intensity in tastes if intensity >= threshold]


@dataclass
class ChemicalCompound:
    """
    Individual chemical compound data structure
    Used in Layer B: Molecular Layer analysis
    """
    # Basic identification
    compound_id: str
    name: str
    cas_number: Optional[str] = None
    pubchem_cid: Optional[str] = None
    
    # Chemical properties
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    smiles: Optional[str] = None  # Simplified Molecular Input Line Entry System
    inchi: Optional[str] = None   # International Chemical Identifier
    
    # Classification
    compound_class: MolecularClass = MolecularClass.TERPENE
    functional_groups: List[str] = field(default_factory=list)
    
    # Sensory properties
    odor_threshold: Optional[float] = None  # ng/L in air
    taste_threshold: Optional[float] = None  # mg/L in water
    odor_descriptors: List[str] = field(default_factory=list)
    taste_descriptors: List[str] = field(default_factory=list)
    
    # Physical properties
    boiling_point: Optional[float] = None
    melting_point: Optional[float] = None
    solubility_water: Optional[float] = None
    vapor_pressure: Optional[float] = None
    
    # Stability and reactivity
    stability_ph_range: Tuple[float, float] = (6.0, 8.0)
    heat_stable: bool = True
    light_sensitive: bool = False
    oxygen_sensitive: bool = False
    
    # Biological activity
    antioxidant_activity: Optional[float] = None
    antimicrobial_activity: Optional[float] = None
    
    # Source information
    natural_occurrence: List[str] = field(default_factory=list)  # Foods where naturally found
    synthetic_routes: List[str] = field(default_factory=list)
    
    # Quality metrics
    confidence_score: float = 0.0
    data_source: DataSource = DataSource.PUBCHEM
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class MolecularProfile:
    """
    Layer B: The Molecular Layer (The "Chemistry")
    Complete chemical compound profile for an ingredient
    """
    ingredient_id: str
    compounds: Dict[str, ChemicalCompound] = field(default_factory=dict)
    
    # Compound concentrations (ppm or percentage)
    concentrations: Dict[str, float] = field(default_factory=dict)
    
    # Volatile organic compounds (VOCs) - key for aroma
    volatile_compounds: Set[str] = field(default_factory=set)
    
    # Non-volatile compounds - key for taste
    non_volatile_compounds: Set[str] = field(default_factory=set)
    
    # Compound interactions and synergies
    synergistic_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    antagonistic_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    
    # Processing effects on compounds
    heat_stable_compounds: Set[str] = field(default_factory=set)
    heat_labile_compounds: Set[str] = field(default_factory=set)
    
    # Extraction and availability
    water_soluble_compounds: Set[str] = field(default_factory=set)
    fat_soluble_compounds: Set[str] = field(default_factory=set)
    alcohol_soluble_compounds: Set[str] = field(default_factory=set)
    
    # Quality metrics
    completeness_score: float = 0.0  # How complete is this molecular profile
    confidence_score: float = 0.0
    last_analysis: datetime = field(default_factory=datetime.now)
    
    def add_compound(self, compound: ChemicalCompound, concentration: float = 0.0):
        """Add a chemical compound to the profile"""
        self.compounds[compound.compound_id] = compound
        if concentration > 0:
            self.concentrations[compound.compound_id] = concentration
            
        # Classify volatility based on vapor pressure
        if compound.vapor_pressure and compound.vapor_pressure > 1.0:  # mmHg at 25°C
            self.volatile_compounds.add(compound.compound_id)
        else:
            self.non_volatile_compounds.add(compound.compound_id)
            
        # Classify solubility
        if compound.solubility_water and compound.solubility_water > 1.0:  # g/L
            self.water_soluble_compounds.add(compound.compound_id)
    
    def get_compounds_by_class(self, compound_class: MolecularClass) -> List[ChemicalCompound]:
        """Get all compounds of a specific chemical class"""
        return [compound for compound in self.compounds.values() 
                if compound.compound_class == compound_class]
    
    def calculate_molecular_similarity(self, other: 'MolecularProfile') -> float:
        """Calculate Tanimoto similarity between molecular profiles"""
        compounds1 = set(self.compounds.keys())
        compounds2 = set(other.compounds.keys())
        
        intersection = len(compounds1.intersection(compounds2))
        union = len(compounds1.union(compounds2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_dominant_compounds(self, threshold: float = 0.01) -> List[Tuple[str, float]]:
        """Get compounds above concentration threshold"""
        return [(comp_id, conc) for comp_id, conc in self.concentrations.items() 
                if conc >= threshold]


@dataclass  
class IngredientCompatibility:
    """
    Represents compatibility relationship between two ingredients
    Part of Layer C: Relational Layer (The "Wisdom")
    """
    ingredient_a: str
    ingredient_b: str
    compatibility_score: CompatibilityLevel
    
    # Statistical evidence
    co_occurrence_count: int = 0
    total_recipes_a: int = 0
    total_recipes_b: int = 0
    pmi_score: float = 0.0  # Pointwise Mutual Information
    
    # Contextual information
    cuisine_contexts: List[str] = field(default_factory=list)
    cooking_methods: List[str] = field(default_factory=list)
    seasonal_patterns: Dict[str, float] = field(default_factory=dict)
    
    # Flavor rationale
    flavor_synergies: List[str] = field(default_factory=list)
    chemical_basis: List[str] = field(default_factory=list)
    
    # Quality metrics
    confidence_level: float = 0.0
    sample_size: int = 0
    data_sources: List[DataSource] = field(default_factory=list)
    
    def calculate_pmi(self):
        """Calculate Pointwise Mutual Information score"""
        if self.total_recipes_a == 0 or self.total_recipes_b == 0:
            self.pmi_score = 0.0
            return
            
        # P(A,B) = co_occurrence / total_recipes
        # P(A) = recipes_with_A / total_recipes  
        # P(B) = recipes_with_B / total_recipes
        # PMI = log2(P(A,B) / (P(A) * P(B)))
        
        total_recipes = max(self.total_recipes_a, self.total_recipes_b)
        p_ab = self.co_occurrence_count / total_recipes
        p_a = self.total_recipes_a / total_recipes
        p_b = self.total_recipes_b / total_recipes
        
        if p_a * p_b > 0:
            self.pmi_score = np.log2(p_ab / (p_a * p_b))
        else:
            self.pmi_score = 0.0


@dataclass
class RelationalProfile:
    """
    Layer C: The Relational Layer (The "Wisdom")
    Complete compatibility network for an ingredient
    """
    ingredient_id: str
    
    # Direct compatibility relationships
    compatible_ingredients: Dict[str, IngredientCompatibility] = field(default_factory=dict)
    
    # Cuisine-specific compatibility
    cuisine_compatibility: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Seasonal compatibility patterns
    seasonal_pairings: Dict[str, List[str]] = field(default_factory=dict)
    
    # Cooking method compatibility
    method_pairings: Dict[str, List[str]] = field(default_factory=dict)
    
    # Network analysis metrics
    centrality_score: float = 0.0  # How central is this ingredient in the flavor network
    cluster_assignments: List[str] = field(default_factory=list)
    
    # Substitution relationships
    primary_substitutes: List[Tuple[str, float]] = field(default_factory=list)
    secondary_substitutes: List[Tuple[str, float]] = field(default_factory=list)
    
    # Quality metrics
    network_completeness: float = 0.0
    relationship_confidence: float = 0.0
    last_network_update: datetime = field(default_factory=datetime.now)
    
    def add_compatibility(self, other_ingredient: str, compatibility: IngredientCompatibility):
        """Add a compatibility relationship"""
        self.compatible_ingredients[other_ingredient] = compatibility
    
    def get_highly_compatible(self, min_score: CompatibilityLevel = CompatibilityLevel.GOOD) -> List[str]:
        """Get ingredients with high compatibility"""
        return [ingredient for ingredient, compat in self.compatible_ingredients.items() 
                if compat.compatibility_score.value >= min_score.value]
    
    def get_cuisine_pairings(self, cuisine: str) -> List[Tuple[str, float]]:
        """Get compatible ingredients for specific cuisine"""
        if cuisine in self.cuisine_compatibility:
            return [(ingredient, score) for ingredient, score in 
                   self.cuisine_compatibility[cuisine].items()]
        return []


@dataclass
class NutritionData:
    """
    Nutritional information used for sensory profile calculation
    Maps nutrition to taste intensity using heuristic formulas
    """
    # Macronutrients (per 100g)
    calories: float = 0.0
    protein: float = 0.0
    fat: float = 0.0
    carbohydrates: float = 0.0
    fiber: float = 0.0
    sugars: float = 0.0
    
    # Minerals that affect taste
    sodium: float = 0.0      # mg - maps to saltiness
    potassium: float = 0.0   # mg
    calcium: float = 0.0     # mg
    magnesium: float = 0.0   # mg
    iron: float = 0.0        # mg
    zinc: float = 0.0        # mg
    
    # Vitamins
    vitamin_c: float = 0.0   # mg - affects sourness
    
    # Organic acids that affect sourness
    citric_acid: float = 0.0
    malic_acid: float = 0.0
    tartaric_acid: float = 0.0
    acetic_acid: float = 0.0
    
    # Compounds affecting bitterness
    caffeine: float = 0.0
    theobromine: float = 0.0
    tannins: float = 0.0
    
    # Amino acids affecting umami
    glutamate: float = 0.0
    aspartate: float = 0.0
    
    # Volatile compounds (affects aroma)
    essential_oils: float = 0.0
    
    def calculate_sweetness_intensity(self) -> float:
        """Convert sugar content to sweetness intensity (0.0-1.0)"""
        # Heuristic: 0g = 0.0, 30g+ = 1.0 (very sweet fruits/desserts)
        return min(self.sugars / 30.0, 1.0)
    
    def calculate_saltiness_intensity(self) -> float:
        """Convert sodium content to saltiness intensity (0.0-1.0)"""
        # Heuristic: 0mg = 0.0, 2000mg+ = 1.0 (very salty foods)
        return min(self.sodium / 2000.0, 1.0)
    
    def calculate_sourness_intensity(self) -> float:
        """Calculate sourness from organic acids and vitamin C"""
        total_acids = (self.citric_acid + self.malic_acid + 
                      self.tartaric_acid + self.acetic_acid + self.vitamin_c / 10.0)
        # Heuristic: 0-5000mg range for acids
        return min(total_acids / 5000.0, 1.0)
    
    def calculate_bitterness_intensity(self) -> float:
        """Calculate bitterness from alkaloids and tannins"""
        bitter_compounds = self.caffeine + self.theobromine + self.tannins
        # Heuristic: 0-200mg range for bitter compounds
        return min(bitter_compounds / 200.0, 1.0)
    
    def calculate_umami_intensity(self) -> float:
        """Calculate umami from glutamate and protein content"""
        # Glutamate is direct umami, protein is proxy
        umami_score = (self.glutamate / 100.0) + (self.protein / 50.0)
        return min(umami_score, 1.0)
    
    def calculate_fatty_intensity(self) -> float:
        """Calculate fatty mouthfeel from fat content"""
        # Heuristic: 0g = 0.0, 50g+ = 1.0 (very fatty foods)
        return min(self.fat / 50.0, 1.0)
    
    def to_sensory_profile(self) -> SensoryProfile:
        """Convert nutrition data to sensory profile using heuristics"""
        return SensoryProfile(
            sweet=self.calculate_sweetness_intensity(),
            sour=self.calculate_sourness_intensity(),
            salty=self.calculate_saltiness_intensity(),
            bitter=self.calculate_bitterness_intensity(),
            umami=self.calculate_umami_intensity(),
            fatty=self.calculate_fatty_intensity(),
            aromatic=min(self.essential_oils / 1000.0, 1.0),  # Essential oils -> aroma
            source=DataSource.USDA
        )


@dataclass
class FlavorProfile:
    """
    Complete flavor profile combining all three layers
    This is the master data structure for flavor intelligence
    """
    # Basic identification
    ingredient_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    common_names: List[str] = field(default_factory=list)
    scientific_name: Optional[str] = None
    
    # Three-layer architecture
    sensory: SensoryProfile = field(default_factory=SensoryProfile)
    molecular: MolecularProfile = field(default_factory=lambda: MolecularProfile(""))
    relational: RelationalProfile = field(default_factory=lambda: RelationalProfile(""))
    
    # Nutritional basis for sensory calculations
    nutrition: Optional[NutritionData] = None
    
    # Classification and categorization
    primary_category: FlavorCategory = FlavorCategory.SAVORY
    secondary_categories: List[FlavorCategory] = field(default_factory=list)
    
    # Geographic and cultural information
    origin_countries: List[str] = field(default_factory=list)
    cuisine_associations: Dict[str, float] = field(default_factory=dict)
    
    # Seasonal and availability information
    seasonal_availability: Dict[str, bool] = field(default_factory=dict)  # month -> available
    peak_seasons: List[str] = field(default_factory=list)
    
    # Processing and preparation effects
    raw_profile: Optional['FlavorProfile'] = None
    cooked_profiles: Dict[str, 'FlavorProfile'] = field(default_factory=dict)  # cooking method -> profile
    
    # Quality and confidence metrics
    overall_confidence: float = 0.0
    data_completeness: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    data_sources: List[DataSource] = field(default_factory=list)
    
    # ML and embedding information
    flavor_embedding: Optional[np.ndarray] = None
    embedding_version: str = "1.0"
    
    def __post_init__(self):
        """Initialize cross-references between layers"""
        if self.molecular.ingredient_id == "":
            self.molecular.ingredient_id = self.ingredient_id
        if self.relational.ingredient_id == "":
            self.relational.ingredient_id = self.ingredient_id
    
    def calculate_overall_confidence(self) -> float:
        """Calculate overall confidence based on data quality"""
        sensory_confidence = len([x for x in self.sensory.confidence_scores.values() if x > 0.5])
        molecular_confidence = self.molecular.confidence_score
        relational_confidence = self.relational.relationship_confidence
        
        # Weight the confidences: sensory 40%, molecular 30%, relational 30%
        self.overall_confidence = (
            (sensory_confidence / 15.0) * 0.4 +  # 15 sensory attributes
            molecular_confidence * 0.3 +
            relational_confidence * 0.3
        )
        return self.overall_confidence
    
    def calculate_data_completeness(self) -> float:
        """Calculate how complete the data profile is"""
        sensory_complete = sum(1 for x in self.sensory.to_vector() if x > 0) / 15.0
        molecular_complete = min(len(self.molecular.compounds) / 10.0, 1.0)  # Expect at least 10 compounds
        relational_complete = min(len(self.relational.compatible_ingredients) / 20.0, 1.0)  # Expect 20 pairings
        
        self.data_completeness = (sensory_complete + molecular_complete + relational_complete) / 3.0
        return self.data_completeness
    
    def generate_flavor_hash(self) -> str:
        """Generate unique hash for this flavor profile"""
        profile_data = {
            'sensory': self.sensory.to_vector().tolist(),
            'molecular': list(self.molecular.compounds.keys()),
            'name': self.name
        }
        return hashlib.md5(json.dumps(profile_data, sort_keys=True).encode()).hexdigest()
    
    def similarity_to(self, other: 'FlavorProfile') -> Dict[str, float]:
        """Calculate multi-layer similarity to another flavor profile"""
        return {
            'sensory': self.sensory.similarity(other.sensory),
            'molecular': self.molecular.calculate_molecular_similarity(other.molecular),
            'overall': (
                self.sensory.similarity(other.sensory) * 0.5 +
                self.molecular.calculate_molecular_similarity(other.molecular) * 0.3 +
                len(set(self.relational.compatible_ingredients.keys()).intersection(
                    set(other.relational.compatible_ingredients.keys())
                )) / max(len(self.relational.compatible_ingredients), 1) * 0.2
            )
        }
    
    def get_substitute_candidates(self, similarity_threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Get potential substitutes based on primary and secondary relationships"""
        candidates = []
        
        # Add primary substitutes
        for substitute, score in self.relational.primary_substitutes:
            if score >= similarity_threshold:
                candidates.append((substitute, score))
        
        # Add secondary substitutes with lower weight
        for substitute, score in self.relational.secondary_substitutes:
            weighted_score = score * 0.8  # Reduce secondary substitute scores
            if weighted_score >= similarity_threshold:
                candidates.append((substitute, weighted_score))
        
        return sorted(candidates, key=lambda x: x[1], reverse=True)


@dataclass
class FlavorDatabase:
    """
    Master database containing all flavor profiles
    Supports millions of ingredients with efficient indexing
    """
    # Core storage
    profiles: Dict[str, FlavorProfile] = field(default_factory=dict)
    
    # Indexing for fast lookup
    name_to_id: Dict[str, str] = field(default_factory=dict)
    category_index: Dict[FlavorCategory, List[str]] = field(default_factory=dict)
    country_index: Dict[str, List[str]] = field(default_factory=dict)
    compound_index: Dict[str, List[str]] = field(default_factory=dict)  # compound -> ingredient_ids
    
    # Statistics and metrics
    total_profiles: int = 0
    total_compounds: int = 0
    total_relationships: int = 0
    database_version: str = "1.0"
    
    # Quality tracking
    high_quality_profiles: Set[str] = field(default_factory=set)  # confidence > 0.8
    complete_profiles: Set[str] = field(default_factory=set)      # completeness > 0.8
    
    def add_profile(self, profile: FlavorProfile):
        """Add a flavor profile to the database with indexing"""
        # Store profile
        self.profiles[profile.ingredient_id] = profile
        
        # Update name index
        self.name_to_id[profile.name.lower()] = profile.ingredient_id
        for common_name in profile.common_names:
            self.name_to_id[common_name.lower()] = profile.ingredient_id
        
        # Update category index
        if profile.primary_category not in self.category_index:
            self.category_index[profile.primary_category] = []
        self.category_index[profile.primary_category].append(profile.ingredient_id)
        
        # Update country index
        for country in profile.origin_countries:
            if country not in self.country_index:
                self.country_index[country] = []
            self.country_index[country].append(profile.ingredient_id)
        
        # Update compound index
        for compound_id in profile.molecular.compounds:
            if compound_id not in self.compound_index:
                self.compound_index[compound_id] = []
            self.compound_index[compound_id].append(profile.ingredient_id)
        
        # Update statistics
        self.total_profiles += 1
        self.total_compounds += len(profile.molecular.compounds)
        self.total_relationships += len(profile.relational.compatible_ingredients)
        
        # Update quality tracking
        if profile.overall_confidence > 0.8:
            self.high_quality_profiles.add(profile.ingredient_id)
        if profile.data_completeness > 0.8:
            self.complete_profiles.add(profile.ingredient_id)
    
    def get_by_name(self, name: str) -> Optional[FlavorProfile]:
        """Get flavor profile by ingredient name"""
        ingredient_id = self.name_to_id.get(name.lower())
        return self.profiles.get(ingredient_id) if ingredient_id else None
    
    def search_by_category(self, category: FlavorCategory) -> List[FlavorProfile]:
        """Get all profiles in a category"""
        ingredient_ids = self.category_index.get(category, [])
        return [self.profiles[id] for id in ingredient_ids if id in self.profiles]
    
    def search_by_country(self, country: str) -> List[FlavorProfile]:
        """Get all profiles from a specific country"""
        ingredient_ids = self.country_index.get(country, [])
        return [self.profiles[id] for id in ingredient_ids if id in self.profiles]
    
    def find_similar_profiles(self, target_profile: FlavorProfile, 
                            limit: int = 10) -> List[Tuple[FlavorProfile, float]]:
        """Find most similar profiles to target"""
        similarities = []
        
        for profile in self.profiles.values():
            if profile.ingredient_id != target_profile.ingredient_id:
                sim_scores = target_profile.similarity_to(profile)
                similarities.append((profile, sim_scores['overall']))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:limit]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        return {
            'total_profiles': self.total_profiles,
            'total_compounds': self.total_compounds,
            'total_relationships': self.total_relationships,
            'high_quality_count': len(self.high_quality_profiles),
            'complete_profiles_count': len(self.complete_profiles),
            'categories_covered': len(self.category_index),
            'countries_covered': len(self.country_index),
            'unique_compounds': len(self.compound_index),
            'database_version': self.database_version,
            'average_confidence': sum(p.overall_confidence for p in self.profiles.values()) / max(len(self.profiles), 1),
            'average_completeness': sum(p.data_completeness for p in self.profiles.values()) / max(len(self.profiles), 1)
        }


# Global database instance
flavor_database = FlavorDatabase()


# Helper functions for data validation and processing

def validate_flavor_profile(profile: FlavorProfile) -> List[str]:
    """Validate a flavor profile and return list of issues"""
    issues = []
    
    # Check basic requirements
    if not profile.name:
        issues.append("Missing ingredient name")
    
    if not profile.ingredient_id:
        issues.append("Missing ingredient ID")
    
    # Check sensory profile
    sensory_vector = profile.sensory.to_vector()
    if np.all(sensory_vector == 0):
        issues.append("All sensory values are zero")
    
    # Check molecular profile
    if not profile.molecular.compounds:
        issues.append("No chemical compounds defined")
    
    # Check relational profile
    if not profile.relational.compatible_ingredients:
        issues.append("No compatibility relationships defined")
    
    # Check data freshness
    days_old = (datetime.now() - profile.last_updated).days
    if days_old > 365:
        issues.append(f"Data is {days_old} days old")
    
    return issues


def create_flavor_profile_from_nutrition(name: str, nutrition_data: NutritionData) -> FlavorProfile:
    """Create a basic flavor profile from nutritional data"""
    sensory = nutrition_data.to_sensory_profile()
    
    profile = FlavorProfile(
        name=name,
        sensory=sensory,
        nutrition=nutrition_data,
        data_sources=[DataSource.USDA]
    )
    
    # Calculate initial confidence and completeness
    profile.calculate_overall_confidence()
    profile.calculate_data_completeness()
    
    return profile


def merge_flavor_profiles(profiles: List[FlavorProfile]) -> FlavorProfile:
    """Merge multiple flavor profiles for the same ingredient from different sources"""
    if not profiles:
        raise ValueError("No profiles to merge")
    
    # Use first profile as base
    merged = profiles[0]
    
    # Merge data from other profiles
    for profile in profiles[1:]:
        # Average sensory values weighted by confidence
        for attr in ['sweet', 'sour', 'salty', 'bitter', 'umami', 'fatty', 'spicy']:
            current_val = getattr(merged.sensory, attr)
            other_val = getattr(profile.sensory, attr)
            
            # Simple average for now - could be weighted by confidence
            if other_val > 0:
                setattr(merged.sensory, attr, (current_val + other_val) / 2)
        
        # Merge molecular compounds
        for comp_id, compound in profile.molecular.compounds.items():
            if comp_id not in merged.molecular.compounds:
                merged.molecular.compounds[comp_id] = compound
        
        # Merge relational data
        for ingredient, compatibility in profile.relational.compatible_ingredients.items():
            if ingredient not in merged.relational.compatible_ingredients:
                merged.relational.compatible_ingredients[ingredient] = compatibility
        
        # Merge data sources
        for source in profile.data_sources:
            if source not in merged.data_sources:
                merged.data_sources.append(source)
    
    # Recalculate metrics
    merged.calculate_overall_confidence()
    merged.calculate_data_completeness()
    merged.last_updated = datetime.now()
    
    return merged


# ===== HEALTH PROFILING SYSTEM =====
# Extension for personalized health-aware food matching

from enum import Enum
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
import uuid
from datetime import datetime, date


class HealthCondition(Enum):
    """Medical conditions and health states for personalized nutrition"""
    # Cardiovascular
    HYPERTENSION = "hypertension"
    HIGH_CHOLESTEROL = "high_cholesterol"
    HEART_DISEASE = "heart_disease"
    
    # Metabolic
    DIABETES_TYPE1 = "diabetes_type1"
    DIABETES_TYPE2 = "diabetes_type2"
    PREDIABETES = "prediabetes"
    METABOLIC_SYNDROME = "metabolic_syndrome"
    OBESITY = "obesity"
    
    # Digestive
    CELIAC_DISEASE = "celiac_disease"
    LACTOSE_INTOLERANCE = "lactose_intolerance"
    IBS = "irritable_bowel_syndrome"
    CROHNS_DISEASE = "crohns_disease"
    GASTRITIS = "gastritis"
    
    # Kidney & Liver
    KIDNEY_DISEASE = "kidney_disease"
    LIVER_DISEASE = "liver_disease"
    
    # Autoimmune
    RHEUMATOID_ARTHRITIS = "rheumatoid_arthritis"
    LUPUS = "lupus"
    MULTIPLE_SCLEROSIS = "multiple_sclerosis"
    
    # Bone Health
    OSTEOPOROSIS = "osteoporosis"
    OSTEOPENIA = "osteopenia"
    
    # Neurological
    ALZHEIMERS = "alzheimers"
    PARKINSONS = "parkinsons"
    
    # Cancer (dietary considerations)
    CANCER_TREATMENT = "cancer_treatment"
    CANCER_SURVIVOR = "cancer_survivor"
    
    # Other
    ANEMIA = "anemia"
    THYROID_DISORDER = "thyroid_disorder"
    GOUT = "gout"
    FOOD_ALLERGIES = "food_allergies"


class DietaryGoal(Enum):
    """Personal dietary and wellness goals"""
    # Weight Management
    WEIGHT_LOSS = "weight_loss"
    WEIGHT_GAIN = "weight_gain"
    WEIGHT_MAINTENANCE = "weight_maintenance"
    MUSCLE_GAIN = "muscle_gain"
    
    # Performance Goals
    ATHLETIC_PERFORMANCE = "athletic_performance"
    ENDURANCE_TRAINING = "endurance_training"
    STRENGTH_TRAINING = "strength_training"
    RECOVERY_OPTIMIZATION = "recovery_optimization"
    
    # Health Optimization
    HEART_HEALTH = "heart_health"
    BRAIN_HEALTH = "brain_health"
    IMMUNE_SUPPORT = "immune_support"
    DIGESTIVE_HEALTH = "digestive_health"
    BONE_HEALTH = "bone_health"
    
    # Longevity & Wellness
    ANTI_AGING = "anti_aging"
    LONGEVITY = "longevity"
    ENERGY_OPTIMIZATION = "energy_optimization"
    SLEEP_QUALITY = "sleep_quality"
    
    # Specific Diets
    KETOGENIC = "ketogenic"
    LOW_CARB = "low_carb"
    MEDITERRANEAN = "mediterranean"
    PLANT_BASED = "plant_based"
    INTERMITTENT_FASTING = "intermittent_fasting"


class DietaryRestriction(Enum):
    """Dietary restrictions and preferences"""
    # Medical Restrictions
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    NUT_FREE = "nut_free"
    SOY_FREE = "soy_free"
    LOW_SODIUM = "low_sodium"
    LOW_SUGAR = "low_sugar"
    LOW_OXALATE = "low_oxalate"
    
    # Lifestyle Choices
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    PESCATARIAN = "pescatarian"
    KOSHER = "kosher"
    HALAL = "halal"
    
    # Special Diets
    KETO = "keto"
    PALEO = "paleo"
    RAW_FOOD = "raw_food"
    WHOLE30 = "whole30"


@dataclass
class PersonalHealthProfile:
    """Complete personal health and dietary profile"""
    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Basic Demographics
    age: Optional[int] = None
    gender: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    activity_level: str = "moderate"  # sedentary, light, moderate, active, very_active
    
    # Health Conditions
    health_conditions: Set[HealthCondition] = field(default_factory=set)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    
    # Goals and Preferences
    dietary_goals: Set[DietaryGoal] = field(default_factory=set)
    dietary_restrictions: Set[DietaryRestriction] = field(default_factory=set)
    
    # Nutritional Targets (daily)
    calorie_target: Optional[int] = None
    protein_target_g: Optional[float] = None
    carb_target_g: Optional[float] = None
    fat_target_g: Optional[float] = None
    fiber_target_g: Optional[float] = None
    sodium_limit_mg: Optional[float] = None
    sugar_limit_g: Optional[float] = None
    
    # Micronutrient Focus Areas
    key_nutrients_focus: List[str] = field(default_factory=list)  # e.g., ["iron", "b12", "calcium"]
    
    # Geographic and Cultural Context
    country: Optional[str] = None
    region: Optional[str] = None
    cultural_preferences: List[str] = field(default_factory=list)
    
    # Timing and Lifecycle
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def calculate_bmi(self) -> Optional[float]:
        """Calculate BMI if weight and height available"""
        if self.weight_kg and self.height_cm:
            height_m = self.height_cm / 100
            return self.weight_kg / (height_m ** 2)
        return None
    
    def get_health_risk_factors(self) -> List[str]:
        """Identify nutritional risk factors based on health conditions"""
        risk_factors = []
        
        if HealthCondition.HYPERTENSION in self.health_conditions:
            risk_factors.extend(["high_sodium", "excess_alcohol"])
        
        if HealthCondition.DIABETES_TYPE2 in self.health_conditions:
            risk_factors.extend(["high_sugar", "refined_carbs", "excess_calories"])
            
        if HealthCondition.HEART_DISEASE in self.health_conditions:
            risk_factors.extend(["trans_fats", "saturated_fats", "high_cholesterol_foods"])
            
        if HealthCondition.KIDNEY_DISEASE in self.health_conditions:
            risk_factors.extend(["high_protein", "high_phosphorus", "high_potassium"])
            
        return list(set(risk_factors))  # Remove duplicates


@dataclass 
class LocalFoodAvailability:
    """Local food availability and seasonality data"""
    region_id: str
    country: str
    region: str
    
    # Seasonal availability (0-1 scale, 1 = peak season)
    seasonal_foods: Dict[str, Dict[str, float]] = field(default_factory=dict)  # {food_id: {month: availability}}
    
    # Local specialties and traditional foods
    traditional_foods: List[str] = field(default_factory=list)
    regional_specialties: List[str] = field(default_factory=list)
    
    # Market data
    local_markets: List[str] = field(default_factory=list)
    price_data: Dict[str, float] = field(default_factory=dict)  # {food_id: avg_price_per_kg}
    
    # Agricultural data
    locally_grown: Set[str] = field(default_factory=set)
    imported_foods: Set[str] = field(default_factory=set)
    
    def get_seasonal_score(self, food_id: str, month: int) -> float:
        """Get seasonal availability score (0-1) for a food in given month"""
        month_name = ["jan", "feb", "mar", "apr", "may", "jun",
                      "jul", "aug", "sep", "oct", "nov", "dec"][month - 1]
        
        if food_id in self.seasonal_foods:
            return self.seasonal_foods[food_id].get(month_name, 0.5)
        return 0.5  # Default neutral availability
    
    def is_locally_available(self, food_id: str) -> bool:
        """Check if food is locally available"""
        return (food_id in self.locally_grown or 
                food_id in self.traditional_foods or
                food_id in self.regional_specialties)


@dataclass
class PersonalizedNutritionRecommendation:
    """AI-generated personalized nutrition recommendation"""
    recommendation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    profile_id: str = ""
    
    # Recommended foods (with confidence scores)
    recommended_foods: Dict[str, float] = field(default_factory=dict)  # {food_id: confidence_score}
    foods_to_limit: Dict[str, float] = field(default_factory=dict)     # {food_id: caution_level}
    foods_to_avoid: List[str] = field(default_factory=list)
    
    # Local context recommendations
    local_seasonal_picks: List[str] = field(default_factory=list)
    cultural_food_adaptations: List[str] = field(default_factory=list)
    
    # Meal timing and combinations
    optimal_meal_combinations: List[List[str]] = field(default_factory=list)
    meal_timing_suggestions: Dict[str, str] = field(default_factory=dict)
    
    # Nutritional explanations
    health_benefits_explanation: str = ""
    risk_mitigation_notes: str = ""
    
    # Confidence and validation
    overall_confidence: float = 0.0
    scientific_evidence_level: str = "moderate"  # low, moderate, high, very_high
    
    # Tracking
    generated_date: datetime = field(default_factory=datetime.now)
    expires_date: Optional[datetime] = None


class PersonalizedNutritionEngine:
    """
    AI Engine for matching local foods to personal health goals and conditions
    Integrates with the existing flavor intelligence system
    """
    
    def __init__(self, flavor_db: FlavorDatabase):
        self.flavor_db = flavor_db
        self.health_profiles: Dict[str, PersonalHealthProfile] = {}
        self.local_availability: Dict[str, LocalFoodAvailability] = {}
        self.recommendation_cache: Dict[str, PersonalizedNutritionRecommendation] = {}
    
    def add_health_profile(self, profile: PersonalHealthProfile) -> str:
        """Add or update a personal health profile"""
        self.health_profiles[profile.profile_id] = profile
        return profile.profile_id
    
    def add_local_availability(self, availability: LocalFoodAvailability) -> None:
        """Add local food availability data for a region"""
        self.local_availability[availability.region_id] = availability
    
    def generate_personalized_recommendations(
        self, 
        profile_id: str, 
        region_id: Optional[str] = None,
        current_month: Optional[int] = None
    ) -> PersonalizedNutritionRecommendation:
        """
        Generate AI-powered personalized food recommendations
        Matches local foods to health conditions and dietary goals
        """
        if profile_id not in self.health_profiles:
            raise ValueError(f"Health profile {profile_id} not found")
        
        profile = self.health_profiles[profile_id]
        current_month = current_month or datetime.now().month
        
        # Initialize recommendation
        recommendation = PersonalizedNutritionRecommendation(profile_id=profile_id)
        
        # Get available foods (local if region specified)
        available_foods = self._get_available_foods(region_id, current_month)
        
        # Analyze each food against health profile
        for food_id, food_profile in available_foods.items():
            score = self._calculate_health_compatibility_score(food_profile, profile)
            
            if score > 0.7:  # High compatibility
                recommendation.recommended_foods[food_id] = score
            elif score < 0.3:  # Low compatibility, potential risks
                recommendation.foods_to_limit[food_id] = 1.0 - score
        
        # Add local seasonal recommendations
        if region_id and region_id in self.local_availability:
            local_data = self.local_availability[region_id]
            seasonal_foods = []
            
            for food_id in recommendation.recommended_foods:
                seasonal_score = local_data.get_seasonal_score(food_id, current_month)
                if seasonal_score > 0.7:  # High seasonal availability
                    seasonal_foods.append(food_id)
            
            recommendation.local_seasonal_picks = seasonal_foods[:10]  # Top 10
        
        # Generate explanations and confidence
        recommendation.health_benefits_explanation = self._generate_health_explanation(
            recommendation, profile
        )
        recommendation.overall_confidence = self._calculate_overall_confidence(recommendation)
        
        # Cache recommendation
        self.recommendation_cache[recommendation.recommendation_id] = recommendation
        
        return recommendation
    
    def _get_available_foods(self, region_id: Optional[str], month: int) -> Dict[str, FlavorProfile]:
        """Get foods available in region, filtered by seasonality if applicable"""
        # Start with all foods in database
        available_foods = self.flavor_db.get_all_profiles()
        
        # Filter by regional availability if region specified
        if region_id and region_id in self.local_availability:
            local_data = self.local_availability[region_id]
            
            # Prioritize locally available foods
            filtered_foods = {}
            for food_id, profile in available_foods.items():
                if (local_data.is_locally_available(food_id) or 
                    local_data.get_seasonal_score(food_id, month) > 0.3):
                    filtered_foods[food_id] = profile
            
            return filtered_foods if filtered_foods else available_foods
        
        return available_foods
    
    def _calculate_health_compatibility_score(
        self, 
        food_profile: FlavorProfile, 
        health_profile: PersonalHealthProfile
    ) -> float:
        """Calculate how well a food matches a person's health profile"""
        score = 0.5  # Baseline neutral score
        
        # Check nutrition data if available
        if food_profile.nutrition:
            nutrition = food_profile.nutrition
            
            # Diabetes considerations
            if HealthCondition.DIABETES_TYPE2 in health_profile.health_conditions:
                if nutrition.sugar_g and nutrition.sugar_g > 10:  # High sugar
                    score -= 0.3
                if nutrition.fiber_g and nutrition.fiber_g > 3:  # High fiber (good)
                    score += 0.2
            
            # Heart disease considerations  
            if HealthCondition.HEART_DISEASE in health_profile.health_conditions:
                if nutrition.saturated_fat_g and nutrition.saturated_fat_g > 5:
                    score -= 0.2
                if nutrition.sodium_mg and nutrition.sodium_mg > 400:
                    score -= 0.2
            
            # Hypertension considerations
            if HealthCondition.HYPERTENSION in health_profile.health_conditions:
                if nutrition.sodium_mg and nutrition.sodium_mg > 200:
                    score -= 0.3
                if nutrition.potassium_mg and nutrition.potassium_mg > 300:  # Potassium helps
                    score += 0.2
            
            # Weight management goals
            if DietaryGoal.WEIGHT_LOSS in health_profile.dietary_goals:
                if nutrition.calories_per_100g and nutrition.calories_per_100g < 100:  # Low calorie
                    score += 0.2
                if nutrition.fiber_g and nutrition.fiber_g > 3:  # Satiating
                    score += 0.1
        
        # Check dietary restrictions
        for restriction in health_profile.dietary_restrictions:
            if restriction == DietaryRestriction.GLUTEN_FREE:
                # Would need to check ingredients for gluten
                pass  # Implement based on ingredient analysis
            elif restriction == DietaryRestriction.VEGAN:
                # Would need to check if food is plant-based
                pass  # Implement based on ingredient analysis
        
        # Ensure score stays in valid range
        return max(0.0, min(1.0, score))
    
    def _generate_health_explanation(
        self, 
        recommendation: PersonalizedNutritionRecommendation,
        profile: PersonalHealthProfile
    ) -> str:
        """Generate human-readable explanation for recommendations"""
        explanations = []
        
        if HealthCondition.DIABETES_TYPE2 in profile.health_conditions:
            explanations.append("Foods selected for stable blood sugar and high fiber content.")
        
        if HealthCondition.HEART_DISEASE in profile.health_conditions:
            explanations.append("Recommendations focus on heart-healthy nutrients and low sodium options.")
        
        if DietaryGoal.WEIGHT_LOSS in profile.dietary_goals:
            explanations.append("Foods chosen for their low calorie density and satiety factors.")
        
        return " ".join(explanations) or "Foods selected based on general nutritional quality."
    
    def _calculate_overall_confidence(self, recommendation: PersonalizedNutritionRecommendation) -> float:
        """Calculate confidence score for the recommendation"""
        # Simple confidence based on number of recommendations
        num_recommendations = len(recommendation.recommended_foods)
        if num_recommendations >= 10:
            return 0.9
        elif num_recommendations >= 5:
            return 0.7
        elif num_recommendations >= 2:
            return 0.5
        else:
            return 0.3


# Export all key classes and functions
__all__ = [
    'FlavorCategory', 'TasteProfile', 'MolecularClass', 'CompatibilityLevel', 'DataSource',
    'SensoryProfile', 'ChemicalCompound', 'MolecularProfile', 'IngredientCompatibility',
    'RelationalProfile', 'NutritionData', 'FlavorProfile', 'FlavorDatabase',
    'flavor_database', 'validate_flavor_profile', 'create_flavor_profile_from_nutrition',
    'merge_flavor_profiles',
    # Health Profiling Extensions
    'HealthCondition', 'DietaryGoal', 'DietaryRestriction', 'PersonalHealthProfile',
    'LocalFoodAvailability', 'PersonalizedNutritionRecommendation', 
    'PersonalizedNutritionEngine'
]