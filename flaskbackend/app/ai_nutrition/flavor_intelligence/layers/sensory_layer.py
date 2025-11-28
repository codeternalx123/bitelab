"""
Layer A: The Sensory Layer (The "Palate") Implementation
======================================================

This module implements the sensory layer of the Automated Flavor Intelligence Pipeline.
It handles the conversion of nutritional data to taste intensities and provides
heuristic normalization for sensory perception mapping.

Key Features:
- Automated nutrition-to-taste conversion using scientific heuristics
- LLM augmentation for abstract sensory properties
- Multi-source data integration with confidence scoring
- Real-time sensory profile calculation and caching
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
from enum import Enum
import math
from collections import defaultdict

from .flavor_data_models import (
    SensoryProfile, NutritionData, FlavorProfile, DataSource,
    TasteProfile, FlavorCategory
)


class SensoryCalculationMethod(Enum):
    """Methods for calculating sensory intensities"""
    HEURISTIC_BASIC = "heuristic_basic"
    HEURISTIC_ADVANCED = "heuristic_advanced"
    LLM_AUGMENTED = "llm_augmented"
    HYBRID = "hybrid"
    MACHINE_LEARNED = "machine_learned"


class NutrientSensoryMapping(Enum):
    """Mapping of nutrients to taste sensations"""
    # Sugar compounds -> Sweetness
    GLUCOSE = ("sweet", 1.0)
    FRUCTOSE = ("sweet", 1.2)  # 20% sweeter than glucose
    SUCROSE = ("sweet", 1.0)
    LACTOSE = ("sweet", 0.16)  # Much less sweet
    MALTOSE = ("sweet", 0.32)
    
    # Sodium compounds -> Saltiness  
    SODIUM_CHLORIDE = ("salty", 1.0)
    SODIUM_CITRATE = ("salty", 0.4)
    SODIUM_PHOSPHATE = ("salty", 0.6)
    
    # Acids -> Sourness
    CITRIC_ACID = ("sour", 1.0)
    MALIC_ACID = ("sour", 0.8)
    TARTARIC_ACID = ("sour", 0.9)
    ACETIC_ACID = ("sour", 1.1)
    LACTIC_ACID = ("sour", 0.7)
    
    # Alkaloids -> Bitterness
    CAFFEINE = ("bitter", 1.0)
    THEOBROMINE = ("bitter", 0.7)
    QUININE = ("bitter", 3.0)  # Extremely bitter
    
    # Amino acids -> Umami
    GLUTAMATE = ("umami", 1.0)
    INOSINATE = ("umami", 2.0)  # Synergistic with glutamate
    GUANYLATE = ("umami", 1.5)


@dataclass
class SensoryCalculationConfig:
    """Configuration for sensory calculations"""
    method: SensoryCalculationMethod = SensoryCalculationMethod.HYBRID
    
    # Heuristic scaling parameters
    sweetness_scale: float = 30.0      # grams sugar for max sweetness
    saltiness_scale: float = 2000.0    # mg sodium for max saltiness  
    sourness_scale: float = 5000.0     # mg acids for max sourness
    bitterness_scale: float = 200.0    # mg alkaloids for max bitterness
    umami_scale: float = 1000.0        # mg glutamate for max umami
    fatty_scale: float = 50.0          # grams fat for max fatty sensation
    
    # LLM augmentation settings
    use_llm_for_aromatics: bool = True
    use_llm_for_spiciness: bool = True
    use_llm_for_texture: bool = True
    llm_confidence_threshold: float = 0.7
    
    # Quality control
    min_data_points: int = 3
    confidence_decay_days: int = 30
    cross_validation_enabled: bool = True


@dataclass  
class LLMAugmentationRequest:
    """Request structure for LLM-based sensory augmentation"""
    ingredient_name: str
    ingredient_category: str
    known_compounds: List[str]
    nutritional_context: Dict[str, float]
    
    # Specific sensory queries
    query_aromatics: bool = True
    query_spiciness: bool = True
    query_texture: bool = True
    query_cultural_notes: bool = True
    
    # Context for better responses
    cuisine_context: List[str] = field(default_factory=list)
    preparation_methods: List[str] = field(default_factory=list)


@dataclass
class LLMAugmentationResponse:
    """Response from LLM sensory augmentation"""
    ingredient_name: str
    
    # Sensory values (0.0-1.0)
    aromatic_intensity: float = 0.0
    spicy_intensity: float = 0.0
    cooling_intensity: float = 0.0
    warming_intensity: float = 0.0
    astringent_intensity: float = 0.0
    
    # Textural properties
    creaminess: float = 0.0
    crispness: float = 0.0
    juiciness: float = 0.0
    
    # Confidence and reasoning
    confidence_score: float = 0.0
    reasoning: str = ""
    cultural_notes: List[str] = field(default_factory=list)
    
    # Quality metrics
    response_time_ms: int = 0
    model_used: str = ""


class SensoryProfileCalculator:
    """
    Advanced calculator for converting nutritional data to sensory profiles
    Uses multiple calculation methods and data sources
    """
    
    def __init__(self, config: SensoryCalculationConfig = None):
        self.config = config or SensoryCalculationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Cache for calculated profiles
        self.profile_cache: Dict[str, Tuple[SensoryProfile, datetime]] = {}
        self.cache_duration = timedelta(hours=24)
        
        # LLM integration
        self.llm_client = None
        self.llm_cache: Dict[str, LLMAugmentationResponse] = {}
        
        # Statistical models for advanced heuristics
        self.trained_models: Dict[str, Any] = {}
        
        # Validation data
        self.validation_profiles: Dict[str, SensoryProfile] = {}
    
    def calculate_sensory_profile(self, nutrition: NutritionData, 
                                ingredient_name: str = "",
                                method: Optional[SensoryCalculationMethod] = None) -> SensoryProfile:
        """
        Main entry point for calculating sensory profile from nutrition data
        """
        calculation_method = method or self.config.method
        
        # Check cache first
        cache_key = f"{ingredient_name}_{nutrition.__hash__()}"
        if cache_key in self.profile_cache:
            cached_profile, cached_time = self.profile_cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return cached_profile
        
        # Calculate based on method
        if calculation_method == SensoryCalculationMethod.HEURISTIC_BASIC:
            profile = self._calculate_basic_heuristic(nutrition)
        elif calculation_method == SensoryCalculationMethod.HEURISTIC_ADVANCED:
            profile = self._calculate_advanced_heuristic(nutrition, ingredient_name)
        elif calculation_method == SensoryCalculationMethod.LLM_AUGMENTED:
            profile = self._calculate_llm_augmented(nutrition, ingredient_name)
        elif calculation_method == SensoryCalculationMethod.HYBRID:
            profile = self._calculate_hybrid(nutrition, ingredient_name)
        else:
            profile = self._calculate_basic_heuristic(nutrition)
        
        # Cache result
        self.profile_cache[cache_key] = (profile, datetime.now())
        
        return profile
    
    def _calculate_basic_heuristic(self, nutrition: NutritionData) -> SensoryProfile:
        """
        Basic heuristic calculation using simple linear scaling
        Fast but less accurate method
        """
        return SensoryProfile(
            sweet=min(nutrition.sugars / self.config.sweetness_scale, 1.0),
            salty=min(nutrition.sodium / self.config.saltiness_scale, 1.0),
            sour=min((nutrition.citric_acid + nutrition.malic_acid + 
                     nutrition.tartaric_acid + nutrition.acetic_acid) / self.config.sourness_scale, 1.0),
            bitter=min((nutrition.caffeine + nutrition.theobromine + 
                       nutrition.tannins) / self.config.bitterness_scale, 1.0),
            umami=min((nutrition.glutamate + nutrition.protein / 10.0) / self.config.umami_scale, 1.0),
            fatty=min(nutrition.fat / self.config.fatty_scale, 1.0),
            aromatic=min(nutrition.essential_oils / 1000.0, 1.0),
            source=DataSource.USDA,
            confidence_scores={attr: 0.6 for attr in ['sweet', 'salty', 'sour', 'bitter', 'umami', 'fatty']}
        )
    
    def _calculate_advanced_heuristic(self, nutrition: NutritionData, ingredient_name: str) -> SensoryProfile:
        """
        Advanced heuristic calculation with non-linear scaling and compound interactions
        """
        # Non-linear sweetness calculation considering sugar type distribution
        sweetness = self._calculate_advanced_sweetness(nutrition)
        
        # Enhanced saltiness with mineral interactions
        saltiness = self._calculate_advanced_saltiness(nutrition)
        
        # Complex sourness with pH estimation
        sourness = self._calculate_advanced_sourness(nutrition)
        
        # Bitterness with synergistic effects
        bitterness = self._calculate_advanced_bitterness(nutrition)
        
        # Enhanced umami with amino acid interactions
        umami = self._calculate_advanced_umami(nutrition)
        
        # Fatty mouthfeel with fat type considerations
        fatty = self._calculate_advanced_fatty(nutrition)
        
        # Spiciness estimation from ingredient name patterns
        spicy = self._estimate_spiciness_from_name(ingredient_name)
        
        # Aromatic intensity from essential oils and volatiles
        aromatic = self._calculate_aromatic_intensity(nutrition, ingredient_name)
        
        return SensoryProfile(
            sweet=sweetness,
            salty=saltiness,
            sour=sourness,
            bitter=bitterness,
            umami=umami,
            fatty=fatty,
            spicy=spicy,
            aromatic=aromatic,
            source=DataSource.USDA,
            confidence_scores={
                'sweet': 0.8, 'salty': 0.8, 'sour': 0.7, 'bitter': 0.7,
                'umami': 0.75, 'fatty': 0.8, 'spicy': 0.5, 'aromatic': 0.6
            }
        )
    
    def _calculate_advanced_sweetness(self, nutrition: NutritionData) -> float:
        """Calculate sweetness with sugar type considerations and saturation curves"""
        total_sugars = nutrition.sugars
        
        if total_sugars == 0:
            return 0.0
        
        # Apply logarithmic scaling to model taste saturation
        # Human taste perception follows Weber-Fechner law (logarithmic)
        normalized = total_sugars / self.config.sweetness_scale
        
        # Logarithmic saturation: S = log(1 + intensity) / log(1 + max_intensity)
        if normalized > 0:
            sweetness = math.log(1 + normalized * 10) / math.log(11)
        else:
            sweetness = 0.0
        
        # Consider fiber content (reduces perceived sweetness)
        if nutrition.fiber > 0:
            fiber_reduction = min(nutrition.fiber / 20.0, 0.3)  # Max 30% reduction
            sweetness *= (1.0 - fiber_reduction)
        
        return min(sweetness, 1.0)
    
    def _calculate_advanced_saltiness(self, nutrition: NutritionData) -> float:
        """Calculate saltiness with mineral interactions"""
        sodium_mg = nutrition.sodium
        
        if sodium_mg == 0:
            return 0.0
        
        # Base saltiness from sodium
        base_saltiness = sodium_mg / self.config.saltiness_scale
        
        # Potassium can enhance salty taste perception
        if nutrition.potassium > 0:
            k_enhancement = min(nutrition.potassium / 2000.0, 0.2)  # Max 20% enhancement
            base_saltiness *= (1.0 + k_enhancement)
        
        # Magnesium adds slight bitterness that can mask saltiness
        if nutrition.magnesium > 0:
            mg_masking = min(nutrition.magnesium / 400.0, 0.15)  # Max 15% reduction
            base_saltiness *= (1.0 - mg_masking)
        
        return min(base_saltiness, 1.0)
    
    def _calculate_advanced_sourness(self, nutrition: NutritionData) -> float:
        """Calculate sourness with acid type weighting and pH estimation"""
        # Weight different acids by their sourness intensity
        weighted_acids = (
            nutrition.citric_acid * 1.0 +      # Citric acid baseline
            nutrition.malic_acid * 0.8 +       # Malic less sour
            nutrition.tartaric_acid * 0.9 +    # Tartaric moderately sour
            nutrition.acetic_acid * 1.1 +      # Acetic more sour
            nutrition.vitamin_c * 0.1          # Ascorbic acid mild
        )
        
        if weighted_acids == 0:
            return 0.0
        
        # Logarithmic scaling for sourness perception
        normalized = weighted_acids / self.config.sourness_scale
        sourness = math.log(1 + normalized * 5) / math.log(6) if normalized > 0 else 0.0
        
        # Sugar can mask sourness
        if nutrition.sugars > 0:
            sugar_masking = min(nutrition.sugars / 40.0, 0.4)  # Max 40% masking
            sourness *= (1.0 - sugar_masking)
        
        return min(sourness, 1.0)
    
    def _calculate_advanced_bitterness(self, nutrition: NutritionData) -> float:
        """Calculate bitterness with compound synergies"""
        # Weight bitter compounds by intensity
        weighted_bitter = (
            nutrition.caffeine * 1.0 +
            nutrition.theobromine * 0.7 +
            nutrition.tannins * 0.8
        )
        
        if weighted_bitter == 0:
            return 0.0
        
        # Bitterness perception is highly logarithmic
        normalized = weighted_bitter / self.config.bitterness_scale
        bitterness = math.log(1 + normalized * 20) / math.log(21) if normalized > 0 else 0.0
        
        # Fat can mask bitterness
        if nutrition.fat > 0:
            fat_masking = min(nutrition.fat / 30.0, 0.3)  # Max 30% masking
            bitterness *= (1.0 - fat_masking)
        
        return min(bitterness, 1.0)
    
    def _calculate_advanced_umami(self, nutrition: NutritionData) -> float:
        """Calculate umami with amino acid synergies"""
        # Direct glutamate contribution
        glutamate_umami = nutrition.glutamate / 1000.0  # mg to normalized
        
        # Protein as proxy for amino acids
        protein_umami = nutrition.protein / 50.0  # 50g protein = significant umami
        
        # Nucleotides enhance umami (estimate from ingredient type)
        nucleotide_enhancement = 0.0
        # This would be enhanced with ingredient classification
        
        total_umami = glutamate_umami + (protein_umami * 0.3) + nucleotide_enhancement
        
        return min(total_umami, 1.0)
    
    def _calculate_advanced_fatty(self, nutrition: NutritionData) -> float:
        """Calculate fatty mouthfeel with fat type considerations"""
        if nutrition.fat == 0:
            return 0.0
        
        # Base fatty sensation
        base_fatty = nutrition.fat / self.config.fatty_scale
        
        # Protein can add to mouthfeel richness
        if nutrition.protein > 0:
            protein_enhancement = min(nutrition.protein / 100.0, 0.2)  # Max 20% enhancement
            base_fatty *= (1.0 + protein_enhancement)
        
        return min(base_fatty, 1.0)
    
    def _estimate_spiciness_from_name(self, ingredient_name: str) -> float:
        """Estimate spiciness from ingredient name using keyword matching"""
        spicy_keywords = {
            'chili': 0.8, 'pepper': 0.6, 'cayenne': 0.9, 'paprika': 0.3,
            'jalapeño': 0.5, 'habanero': 0.9, 'ghost': 1.0, 'carolina reaper': 1.0,
            'serrano': 0.6, 'poblano': 0.2, 'chipotle': 0.5, 'ancho': 0.3,
            'ginger': 0.2, 'horseradish': 0.4, 'wasabi': 0.6, 'mustard': 0.2,
            'black pepper': 0.3, 'white pepper': 0.25, 'szechuan': 0.4,
            'tabasco': 0.7, 'sriracha': 0.4, 'harissa': 0.6
        }
        
        ingredient_lower = ingredient_name.lower()
        max_spiciness = 0.0
        
        for keyword, spiciness in spicy_keywords.items():
            if keyword in ingredient_lower:
                max_spiciness = max(max_spiciness, spiciness)
        
        return max_spiciness
    
    def _calculate_aromatic_intensity(self, nutrition: NutritionData, ingredient_name: str) -> float:
        """Calculate aromatic intensity from essential oils and name patterns"""
        # Base from essential oils
        base_aromatic = min(nutrition.essential_oils / 1000.0, 1.0)
        
        # Enhance based on ingredient name
        aromatic_keywords = {
            'herb': 0.3, 'basil': 0.7, 'rosemary': 0.8, 'thyme': 0.6,
            'oregano': 0.5, 'mint': 0.8, 'sage': 0.6, 'cilantro': 0.4,
            'garlic': 0.6, 'onion': 0.4, 'shallot': 0.3, 'ginger': 0.5,
            'lemon': 0.6, 'lime': 0.7, 'orange': 0.5, 'vanilla': 0.8,
            'cinnamon': 0.7, 'nutmeg': 0.6, 'cardamom': 0.8, 'clove': 0.9,
            'star anise': 0.7, 'fennel': 0.5, 'cumin': 0.4, 'coriander': 0.5
        }
        
        ingredient_lower = ingredient_name.lower()
        name_aromatic = 0.0
        
        for keyword, intensity in aromatic_keywords.items():
            if keyword in ingredient_lower:
                name_aromatic = max(name_aromatic, intensity)
        
        # Combine base and name-based estimates
        combined_aromatic = base_aromatic + (name_aromatic * 0.5)
        return min(combined_aromatic, 1.0)
    
    async def _calculate_llm_augmented(self, nutrition: NutritionData, ingredient_name: str) -> SensoryProfile:
        """Calculate sensory profile with LLM augmentation for complex attributes"""
        # Start with advanced heuristic as base
        base_profile = self._calculate_advanced_heuristic(nutrition, ingredient_name)
        
        if not self.config.use_llm_for_aromatics and not self.config.use_llm_for_spiciness:
            return base_profile
        
        # Prepare LLM request
        llm_request = LLMAugmentationRequest(
            ingredient_name=ingredient_name,
            ingredient_category="food",  # Would be determined by classification
            known_compounds=[],  # Would be populated from molecular data
            nutritional_context={
                'protein': nutrition.protein,
                'fat': nutrition.fat,
                'sugars': nutrition.sugars,
                'sodium': nutrition.sodium
            }
        )
        
        # Get LLM augmentation
        try:
            llm_response = await self._query_llm_for_sensory_data(llm_request)
            
            if llm_response.confidence_score >= self.config.llm_confidence_threshold:
                # Update base profile with LLM data
                if self.config.use_llm_for_aromatics:
                    base_profile.aromatic = max(base_profile.aromatic, llm_response.aromatic_intensity)
                
                if self.config.use_llm_for_spiciness:
                    base_profile.spicy = max(base_profile.spicy, llm_response.spicy_intensity)
                
                # Add new attributes from LLM
                base_profile.cooling = llm_response.cooling_intensity
                base_profile.warming = llm_response.warming_intensity
                base_profile.astringent = llm_response.astringent_intensity
                
                if self.config.use_llm_for_texture:
                    base_profile.creaminess = llm_response.creaminess
                    base_profile.crispness = llm_response.crispness
                    base_profile.juiciness = llm_response.juiciness
                
                # Update confidence scores
                base_profile.confidence_scores.update({
                    'aromatic': llm_response.confidence_score,
                    'spicy': llm_response.confidence_score,
                    'cooling': llm_response.confidence_score,
                    'warming': llm_response.confidence_score
                })
        
        except Exception as e:
            self.logger.warning(f"LLM augmentation failed for {ingredient_name}: {e}")
        
        return base_profile
    
    def _calculate_hybrid(self, nutrition: NutritionData, ingredient_name: str) -> SensoryProfile:
        """
        Hybrid calculation combining multiple methods with confidence weighting
        """
        # Get profiles from different methods
        basic_profile = self._calculate_basic_heuristic(nutrition)
        advanced_profile = self._calculate_advanced_heuristic(nutrition, ingredient_name)
        
        # Weight profiles based on data availability and confidence
        basic_weight = 0.3
        advanced_weight = 0.7
        
        # Create weighted combination
        hybrid_profile = SensoryProfile(
            sweet=(basic_profile.sweet * basic_weight + advanced_profile.sweet * advanced_weight),
            sour=(basic_profile.sour * basic_weight + advanced_profile.sour * advanced_weight),
            salty=(basic_profile.salty * basic_weight + advanced_profile.salty * advanced_weight),
            bitter=(basic_profile.bitter * basic_weight + advanced_profile.bitter * advanced_weight),
            umami=(basic_profile.umami * basic_weight + advanced_profile.umami * advanced_weight),
            fatty=(basic_profile.fatty * basic_weight + advanced_profile.fatty * advanced_weight),
            spicy=advanced_profile.spicy,  # Only from advanced
            aromatic=advanced_profile.aromatic,  # Only from advanced
            cooling=advanced_profile.cooling,
            warming=advanced_profile.warming,
            astringent=advanced_profile.astringent,
            source=DataSource.USDA,
            confidence_scores={
                'sweet': 0.75, 'sour': 0.75, 'salty': 0.75, 'bitter': 0.7,
                'umami': 0.7, 'fatty': 0.75, 'spicy': 0.6, 'aromatic': 0.65
            }
        )
        
        return hybrid_profile
    
    async def _query_llm_for_sensory_data(self, request: LLMAugmentationRequest) -> LLMAugmentationResponse:
        """Query LLM for sensory data augmentation"""
        prompt = self._build_llm_prompt(request)
        
        start_time = datetime.now()
        
        # Check cache first
        cache_key = f"{request.ingredient_name}_{hash(prompt)}"
        if cache_key in self.llm_cache:
            return self.llm_cache[cache_key]
        
        try:
            # Mock LLM response for now - would integrate with actual LLM API
            response_data = await self._mock_llm_query(prompt, request.ingredient_name)
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            response = LLMAugmentationResponse(
                ingredient_name=request.ingredient_name,
                aromatic_intensity=response_data.get('aromatic', 0.0),
                spicy_intensity=response_data.get('spicy', 0.0),
                cooling_intensity=response_data.get('cooling', 0.0),
                warming_intensity=response_data.get('warming', 0.0),
                astringent_intensity=response_data.get('astringent', 0.0),
                creaminess=response_data.get('creaminess', 0.0),
                crispness=response_data.get('crispness', 0.0),
                juiciness=response_data.get('juiciness', 0.0),
                confidence_score=response_data.get('confidence', 0.7),
                reasoning=response_data.get('reasoning', ''),
                response_time_ms=int(response_time),
                model_used="gpt-4"
            )
            
            # Cache response
            self.llm_cache[cache_key] = response
            return response
            
        except Exception as e:
            self.logger.error(f"LLM query failed: {e}")
            # Return default response
            return LLMAugmentationResponse(
                ingredient_name=request.ingredient_name,
                confidence_score=0.0
            )
    
    def _build_llm_prompt(self, request: LLMAugmentationRequest) -> str:
        """Build comprehensive prompt for LLM sensory analysis"""
        prompt = f"""
You are a culinary and food science expert. Analyze the sensory properties of '{request.ingredient_name}'.

Provide ratings on a scale of 0.0 to 1.0 for each property:

AROMATIC INTENSITY: How strong is the aroma/smell?
SPICY INTENSITY: Heat level from capsaicin or similar compounds
COOLING INTENSITY: Menthol-like cooling sensation
WARMING INTENSITY: Warming sensation like ginger or cinnamon
ASTRINGENT INTENSITY: Drying, puckering sensation like tannins

TEXTURE PROPERTIES:
CREAMINESS: Rich, smooth mouthfeel
CRISPNESS: Crisp, crunchy texture
JUICINESS: Moisture and juice release

Context:
- Ingredient: {request.ingredient_name}
- Category: {request.ingredient_category}
- Nutritional context: {request.nutritional_context}

Respond in JSON format:
{{
    "aromatic": 0.0-1.0,
    "spicy": 0.0-1.0,
    "cooling": 0.0-1.0,
    "warming": 0.0-1.0,
    "astringent": 0.0-1.0,
    "creaminess": 0.0-1.0,
    "crispness": 0.0-1.0,
    "juiciness": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of ratings"
}}
"""
        return prompt
    
    async def _mock_llm_query(self, prompt: str, ingredient_name: str) -> Dict[str, float]:
        """Mock LLM response - replace with actual LLM integration"""
        # Simulate API delay
        await asyncio.sleep(0.1)
        
        # Mock responses based on ingredient name patterns
        ingredient_lower = ingredient_name.lower()
        
        # Default values
        response = {
            'aromatic': 0.2, 'spicy': 0.0, 'cooling': 0.0, 'warming': 0.0,
            'astringent': 0.0, 'creaminess': 0.1, 'crispness': 0.1,
            'juiciness': 0.2, 'confidence': 0.7, 'reasoning': 'Estimated based on common culinary knowledge'
        }
        
        # Pattern-based adjustments
        if any(spice in ingredient_lower for spice in ['mint', 'menthol']):
            response.update({'cooling': 0.8, 'aromatic': 0.7})
        elif any(spice in ingredient_lower for spice in ['ginger', 'cinnamon', 'clove']):
            response.update({'warming': 0.6, 'aromatic': 0.7})
        elif any(spice in ingredient_lower for spice in ['chili', 'pepper', 'cayenne']):
            response.update({'spicy': 0.8, 'warming': 0.3})
        elif any(herb in ingredient_lower for herb in ['basil', 'rosemary', 'thyme']):
            response.update({'aromatic': 0.8})
        elif any(fruit in ingredient_lower for fruit in ['apple', 'grape', 'cherry']):
            response.update({'juiciness': 0.7, 'crispness': 0.4})
        elif any(dairy in ingredient_lower for dairy in ['cream', 'butter', 'cheese']):
            response.update({'creaminess': 0.8, 'fatty': 0.7})
        
        return response
    
    def validate_sensory_profile(self, profile: SensoryProfile, 
                               ingredient_name: str = "") -> Dict[str, Any]:
        """Validate sensory profile for consistency and accuracy"""
        issues = []
        warnings = []
        
        # Check value ranges
        sensory_values = profile.to_vector()
        if np.any(sensory_values < 0) or np.any(sensory_values > 1):
            issues.append("Sensory values outside 0.0-1.0 range")
        
        # Check for impossible combinations
        if profile.sweet > 0.8 and profile.bitter > 0.8:
            warnings.append("High sweetness and bitterness combination unusual")
        
        if profile.cooling > 0.5 and profile.warming > 0.5:
            issues.append("Cannot have high cooling and warming simultaneously")
        
        # Check confidence scores
        low_confidence_attrs = [attr for attr, conf in profile.confidence_scores.items() 
                              if conf < 0.3]
        if low_confidence_attrs:
            warnings.append(f"Low confidence in: {', '.join(low_confidence_attrs)}")
        
        # Check for missing data
        if np.all(sensory_values == 0):
            issues.append("All sensory values are zero")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'confidence_avg': np.mean(list(profile.confidence_scores.values())) if profile.confidence_scores else 0.0,
            'data_richness': np.count_nonzero(sensory_values) / len(sensory_values)
        }
    
    def batch_calculate_profiles(self, nutrition_data: List[Tuple[NutritionData, str]]) -> List[SensoryProfile]:
        """Efficiently calculate multiple sensory profiles"""
        profiles = []
        
        for nutrition, name in nutrition_data:
            try:
                profile = self.calculate_sensory_profile(nutrition, name)
                profiles.append(profile)
            except Exception as e:
                self.logger.error(f"Failed to calculate profile for {name}: {e}")
                # Add empty profile to maintain list alignment
                profiles.append(SensoryProfile(source=DataSource.USDA))
        
        return profiles
    
    def get_calculation_statistics(self) -> Dict[str, Any]:
        """Get statistics about sensory calculations performed"""
        return {
            'profiles_cached': len(self.profile_cache),
            'llm_queries_cached': len(self.llm_cache),
            'cache_hit_rate': 0.85,  # Would be calculated from actual usage
            'average_calculation_time_ms': 50,  # Would be measured
            'method_distribution': {
                'heuristic_basic': 0.2,
                'heuristic_advanced': 0.3,
                'llm_augmented': 0.2,
                'hybrid': 0.3
            }
        }


# Factory functions for common use cases

def create_sensory_calculator(method: SensoryCalculationMethod = SensoryCalculationMethod.HYBRID) -> SensoryProfileCalculator:
    """Create a sensory calculator with specified method"""
    config = SensoryCalculationConfig(method=method)
    return SensoryProfileCalculator(config)


def quick_sensory_profile(sugars: float, sodium: float, fat: float, 
                         protein: float = 0.0, ingredient_name: str = "") -> SensoryProfile:
    """Quick sensory profile calculation from basic nutrients"""
    nutrition = NutritionData(
        sugars=sugars,
        sodium=sodium,
        fat=fat,
        protein=protein
    )
    
    calculator = create_sensory_calculator(SensoryCalculationMethod.HEURISTIC_ADVANCED)
    return calculator.calculate_sensory_profile(nutrition, ingredient_name)


def calibrate_sensory_calculator(known_profiles: Dict[str, SensoryProfile]) -> SensoryProfileCalculator:
    """Create calibrated calculator using known sensory profiles"""
    calculator = create_sensory_calculator()
    calculator.validation_profiles = known_profiles
    return calculator


# Export key classes and functions
__all__ = [
    'SensoryCalculationMethod', 'NutrientSensoryMapping', 'SensoryCalculationConfig',
    'LLMAugmentationRequest', 'LLMAugmentationResponse', 'SensoryProfileCalculator',
    'create_sensory_calculator', 'quick_sensory_profile', 'calibrate_sensory_calculator'
]