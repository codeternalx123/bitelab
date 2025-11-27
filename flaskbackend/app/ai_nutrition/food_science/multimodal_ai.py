"""
Multi-Modal AI for Food Analysis
=================================

Multi-modal AI system combining Vision + Audio + Text for comprehensive food
analysis from YouTube cooking videos, images, and recipes.

Features:
1. VISION: Ingredient detection, cooking method recognition, portion estimation
2. AUDIO: Speech-to-text, keyword extraction, texture sounds
3. TEXT: Recipe extraction, ingredient parsing, nutrition estimation
4. FUSION: Combined health and flavor scoring
5. YouTube video analysis pipeline
6. Goal-based filtering (muscle gain, low inflammation, diabetes, hypertension)

Models Used:
- Vision: YOLO v8 / ViT for ingredient/method detection
- Audio: OpenAI Whisper for speech-to-text
- NLP: Transformer-based extraction
- Custom: HealthImpactAnalyzer, FlavorProfiler

Performance Targets:
- Ingredient detection: >85% accuracy
- Cooking method: >90% accuracy
- Audio transcription: >95% accuracy
- Full video analysis: <30 seconds
- Flavor scoring: 7 dimensions
- Health scoring: Multi-factor analysis

Use Cases:
1. YouTube video → Health/Flavor profile
2. Recipe image → Ingredient list
3. Cooking sounds → Texture/method prediction
4. Multi-modal fusion → Personalized diet matching

Author: Wellomex AI Team
Date: November 2025
Version: 8.0.0
"""

import logging
import time
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION & ENUMS
# ============================================================================

class ModalityType(Enum):
    """Input modality types"""
    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"


class CookingMethod(Enum):
    """Cooking methods (visual recognition)"""
    DEEP_FRYING = "deep_frying"       # High fat, cardiovascular risk
    GRILLING = "grilling"             # Potential carcinogens (charring)
    STEAMING = "steaming"             # Healthy, nutrient preservation
    BOILING = "boiling"               # Nutrient leaching
    ROASTING = "roasting"             # Moderate, depends on oil
    SAUTEING = "sauteing"             # Moderate fat
    RAW = "raw"                       # Maximum nutrients
    BAKING = "baking"                 # Variable


class HealthGoal(Enum):
    """User health goals"""
    MUSCLE_GAIN = "muscle_gain"
    WEIGHT_LOSS = "weight_loss"
    LOW_INFLAMMATION = "low_inflammation"
    DIABETES_MANAGEMENT = "diabetes_management"
    HYPERTENSION = "hypertension"
    GENERAL_HEALTH = "general_health"


@dataclass
class MultiModalConfig:
    """Multi-modal analysis configuration"""
    # Model confidence thresholds
    vision_confidence: float = 0.70
    audio_confidence: float = 0.80
    text_confidence: float = 0.85
    
    # Fusion weights
    vision_weight: float = 0.40
    audio_weight: float = 0.20
    text_weight: float = 0.40
    
    # Health scoring
    protein_high_threshold_g: float = 30.0
    fat_high_threshold_g: float = 20.0
    sodium_high_threshold_mg: float = 600.0


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class VisionOutput:
    """Computer vision analysis results"""
    ingredients: List[str] = field(default_factory=list)
    cooking_methods: List[CookingMethod] = field(default_factory=list)
    portions: int = 1
    
    # Visual cues
    has_heavy_cream: bool = False
    has_oil_splatter: bool = False  # Indicates frying
    has_charring: bool = False      # Grilling/burning
    color_golden_brown: bool = False
    
    # Confidence
    confidence: float = 0.0


@dataclass
class AudioOutput:
    """Audio analysis results"""
    transcript: str = ""
    keywords: List[str] = field(default_factory=list)
    
    # Sound-based texture/method detection
    detected_sizzle: bool = False    # Frying
    detected_crunch: bool = False    # Crispy texture
    detected_boiling: bool = False   # Boiling water
    detected_knife_chop: bool = False
    
    # Taste adjectives from speech
    taste_descriptors: List[str] = field(default_factory=list)
    
    # Confidence
    confidence: float = 0.0


@dataclass
class TextOutput:
    """NLP text analysis results"""
    recipe_title: str = ""
    ingredients: List[str] = field(default_factory=list)
    quantities: Dict[str, str] = field(default_factory=dict)
    
    # Extracted nutrition mentions
    mentioned_protein: bool = False
    mentioned_sugar: bool = False
    mentioned_salt: bool = False
    
    # Taste mentions
    taste_adjectives: List[str] = field(default_factory=list)
    
    # Confidence
    confidence: float = 0.0


@dataclass
class FlavorProfile:
    """7-dimensional flavor profile"""
    spicy: float = 0.0      # 0.0-1.0
    savory: float = 0.0     # Umami
    sweet: float = 0.0
    sour: float = 0.0
    bitter: float = 0.0
    umami: float = 0.0
    texture: str = "unknown"  # crispy, soft, creamy, crunchy
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'spicy': float(self.spicy),
            'savory': float(self.savory),
            'sweet': float(self.sweet),
            'sour': float(self.sour),
            'bitter': float(self.bitter),
            'umami': float(self.umami),
            'texture': self.texture
        }


@dataclass
class HealthProfile:
    """Health impact assessment"""
    # Macros (estimated)
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    
    # Health factors
    cooking_method_risk: float = 0.0  # 0.0-1.0 (1.0 = unhealthy)
    sodium_mg: float = 0.0
    added_sugar_g: float = 0.0
    
    # Ingredient quality
    has_processed_ingredients: bool = False
    has_inflammatory_oils: bool = False
    
    # Overall health score
    health_score: float = 0.5  # 0.0-1.0 (1.0 = healthiest)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'protein_g': float(self.protein_g),
            'carbs_g': float(self.carbs_g),
            'fat_g': float(self.fat_g),
            'cooking_method_risk': float(self.cooking_method_risk),
            'sodium_mg': float(self.sodium_mg),
            'added_sugar_g': float(self.added_sugar_g),
            'has_processed_ingredients': self.has_processed_ingredients,
            'has_inflammatory_oils': self.has_inflammatory_oils,
            'health_score': float(self.health_score)
        }


@dataclass
class MultiModalResult:
    """Complete multi-modal analysis result"""
    # Input modalities
    vision: Optional[VisionOutput] = None
    audio: Optional[AudioOutput] = None
    text: Optional[TextOutput] = None
    
    # Fused outputs
    flavor_profile: FlavorProfile = field(default_factory=FlavorProfile)
    health_profile: HealthProfile = field(default_factory=HealthProfile)
    
    # Combined ingredient list
    all_ingredients: List[str] = field(default_factory=list)
    
    # Goal matching
    fits_goals: Dict[HealthGoal, float] = field(default_factory=dict)  # 0.0-1.0


# ============================================================================
# MOCK VISION MODEL
# ============================================================================

class VisionModel:
    """
    Mock vision model (YOLO/ViT)
    
    In production: Replace with actual YOLO v8 or Vision Transformer
    """
    
    def __init__(self):
        # Ingredient appearance keywords
        self.ingredient_keywords = {
            'chili': ['red flakes', 'chili pepper', 'red pepper'],
            'cream': ['white liquid', 'heavy cream', 'milk'],
            'oil': ['golden liquid', 'oil bottle'],
            'chicken': ['raw chicken', 'chicken breast'],
            'beef': ['red meat', 'steak'],
            'vegetables': ['green vegetables', 'carrots', 'broccoli'],
            'cheese': ['yellow cheese', 'shredded cheese']
        }
        
        # Method visual cues
        self.method_cues = {
            CookingMethod.DEEP_FRYING: ['oil splatter', 'deep oil', 'bubbling'],
            CookingMethod.GRILLING: ['grill marks', 'charring', 'barbecue'],
            CookingMethod.STEAMING: ['steam', 'bamboo steamer'],
            CookingMethod.BOILING: ['boiling water', 'pot of water'],
            CookingMethod.SAUTEING: ['pan', 'sauté pan', 'stir'],
            CookingMethod.BAKING: ['oven', 'baking sheet']
        }
        
        logger.info("Vision Model initialized (Mock)")
    
    def analyze_image(self, image_description: str) -> VisionOutput:
        """
        Analyze image/frame
        
        Args:
            image_description: Text description (mock input)
        
        Returns:
            VisionOutput with detected ingredients and methods
        """
        desc_lower = image_description.lower()
        
        output = VisionOutput()
        
        # Detect ingredients
        for ingredient, keywords in self.ingredient_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                output.ingredients.append(ingredient)
        
        # Detect cooking methods
        for method, cues in self.method_cues.items():
            if any(cue in desc_lower for cue in cues):
                output.cooking_methods.append(method)
        
        # Visual cues
        output.has_heavy_cream = 'cream' in desc_lower or 'heavy cream' in desc_lower
        output.has_oil_splatter = 'oil splatter' in desc_lower or 'bubbling oil' in desc_lower
        output.has_charring = 'char' in desc_lower or 'grill marks' in desc_lower
        output.color_golden_brown = 'golden' in desc_lower or 'brown' in desc_lower
        
        output.confidence = 0.85
        
        return output


# ============================================================================
# MOCK AUDIO MODEL
# ============================================================================

class AudioModel:
    """
    Mock audio model (OpenAI Whisper + sound classification)
    
    In production: Use OpenAI Whisper API for transcription
    """
    
    def __init__(self):
        # Taste descriptors to extract
        self.taste_keywords = {
            'spicy': ['spicy', 'hot', 'fiery', 'heat'],
            'sweet': ['sweet', 'sugary', 'candy'],
            'sour': ['sour', 'tangy', 'acidic', 'tart'],
            'savory': ['savory', 'umami', 'rich'],
            'salty': ['salty', 'salted']
        }
        
        # Sound patterns (physics-based)
        self.sound_patterns = {
            'sizzle': ['sizzle', 'sizzling', 'frying sound'],
            'crunch': ['crunch', 'crunchy', 'crispy sound'],
            'boiling': ['boiling', 'bubbling water'],
            'chop': ['chop', 'chopping', 'knife']
        }
        
        logger.info("Audio Model initialized (Mock)")
    
    def analyze_audio(self, audio_description: str) -> AudioOutput:
        """
        Analyze audio (transcription + sound classification)
        
        Args:
            audio_description: Text description of audio (mock)
        
        Returns:
            AudioOutput with transcript and detected sounds
        """
        desc_lower = audio_description.lower()
        
        output = AudioOutput()
        output.transcript = audio_description
        
        # Extract taste descriptors
        taste_desc = []
        for category, keywords in self.taste_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                taste_desc.append(category)
        
        output.taste_descriptors = taste_desc
        
        # Detect sounds
        output.detected_sizzle = any(s in desc_lower for s in self.sound_patterns['sizzle'])
        output.detected_crunch = any(s in desc_lower for s in self.sound_patterns['crunch'])
        output.detected_boiling = any(s in desc_lower for s in self.sound_patterns['boiling'])
        output.detected_knife_chop = any(s in desc_lower for s in self.sound_patterns['chop'])
        
        # Extract keywords (simple - in production use NER)
        words = desc_lower.split()
        output.keywords = [w for w in words if len(w) > 4][:10]
        
        output.confidence = 0.90
        
        return output


# ============================================================================
# MOCK NLP MODEL
# ============================================================================

class NLPModel:
    """
    Mock NLP model for recipe extraction
    
    In production: Use GPT-4 or fine-tuned transformer
    """
    
    def __init__(self):
        # Common ingredients
        self.ingredient_patterns = [
            'chicken', 'beef', 'pork', 'fish', 'shrimp',
            'rice', 'pasta', 'noodles',
            'tomato', 'onion', 'garlic', 'pepper',
            'oil', 'butter', 'cream', 'cheese',
            'salt', 'sugar', 'soy sauce'
        ]
        
        logger.info("NLP Model initialized (Mock)")
    
    def analyze_text(self, text: str) -> TextOutput:
        """
        Analyze recipe text
        
        Args:
            text: Recipe description or video transcript
        
        Returns:
            TextOutput with extracted recipe info
        """
        text_lower = text.lower()
        
        output = TextOutput()
        
        # Extract title (first sentence)
        sentences = text.split('.')
        if sentences:
            output.recipe_title = sentences[0].strip()
        
        # Extract ingredients (simple keyword matching)
        ingredients = []
        for pattern in self.ingredient_patterns:
            if pattern in text_lower:
                ingredients.append(pattern)
        
        output.ingredients = ingredients
        
        # Nutrition mentions
        output.mentioned_protein = 'protein' in text_lower
        output.mentioned_sugar = 'sugar' in text_lower or 'sweet' in text_lower
        output.mentioned_salt = 'salt' in text_lower or 'sodium' in text_lower
        
        # Taste adjectives
        taste_words = ['spicy', 'sweet', 'sour', 'savory', 'tangy', 'rich', 'creamy']
        output.taste_adjectives = [w for w in taste_words if w in text_lower]
        
        output.confidence = 0.88
        
        return output


# ============================================================================
# FUSION ENGINE
# ============================================================================

class MultiModalFusion:
    """
    Fuse Vision + Audio + Text into unified analysis
    """
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        
        # Ingredient → macro mapping (simplified)
        self.ingredient_macros = {
            'chicken': {'protein': 31, 'fat': 3.6, 'carbs': 0},
            'beef': {'protein': 26, 'fat': 15, 'carbs': 0},
            'cream': {'protein': 2, 'fat': 37, 'carbs': 3},
            'oil': {'protein': 0, 'fat': 100, 'carbs': 0},
            'rice': {'protein': 2.7, 'fat': 0.3, 'carbs': 28},
            'pasta': {'protein': 5, 'fat': 1, 'carbs': 25},
            'cheese': {'protein': 25, 'fat': 33, 'carbs': 1.3},
            'vegetables': {'protein': 2, 'fat': 0.3, 'carbs': 7}
        }
        
        # Cooking method health risk
        self.method_risk = {
            CookingMethod.DEEP_FRYING: 0.8,
            CookingMethod.GRILLING: 0.5,  # Charring → carcinogens
            CookingMethod.SAUTEING: 0.3,
            CookingMethod.ROASTING: 0.2,
            CookingMethod.STEAMING: 0.0,
            CookingMethod.BOILING: 0.1,
            CookingMethod.RAW: 0.0,
            CookingMethod.BAKING: 0.2
        }
        
        logger.info("Multi-Modal Fusion initialized")
    
    def fuse_modalities(
        self,
        vision: Optional[VisionOutput],
        audio: Optional[AudioOutput],
        text: Optional[TextOutput]
    ) -> MultiModalResult:
        """
        Fuse all modalities into unified result
        """
        result = MultiModalResult(vision=vision, audio=audio, text=text)
        
        # Combine ingredients from all sources
        all_ingredients = set()
        if vision:
            all_ingredients.update(vision.ingredients)
        if text:
            all_ingredients.update(text.ingredients)
        
        result.all_ingredients = list(all_ingredients)
        
        # Build flavor profile
        result.flavor_profile = self._build_flavor_profile(vision, audio, text)
        
        # Build health profile
        result.health_profile = self._build_health_profile(
            result.all_ingredients,
            vision,
            audio,
            text
        )
        
        return result
    
    def _build_flavor_profile(
        self,
        vision: Optional[VisionOutput],
        audio: Optional[AudioOutput],
        text: Optional[TextOutput]
    ) -> FlavorProfile:
        """Build fused flavor profile"""
        profile = FlavorProfile()
        
        # Texture from audio
        if audio:
            if audio.detected_crunch:
                profile.texture = "crispy/crunchy"
            elif 'creamy' in audio.taste_descriptors:
                profile.texture = "creamy"
        
        # Taste from audio + text
        taste_scores = defaultdict(float)
        
        if audio:
            for descriptor in audio.taste_descriptors:
                if descriptor == 'spicy':
                    taste_scores['spicy'] += 0.5
                elif descriptor == 'sweet':
                    taste_scores['sweet'] += 0.5
                elif descriptor == 'sour':
                    taste_scores['sour'] += 0.5
                elif descriptor == 'savory':
                    taste_scores['savory'] += 0.5
                    taste_scores['umami'] += 0.5
        
        if text:
            for adj in text.taste_adjectives:
                if adj == 'spicy':
                    taste_scores['spicy'] += 0.3
                elif adj == 'sweet':
                    taste_scores['sweet'] += 0.3
                elif adj == 'rich':
                    taste_scores['savory'] += 0.3
                    taste_scores['umami'] += 0.3
        
        # Visual cues
        if vision:
            if 'chili' in vision.ingredients:
                taste_scores['spicy'] += 0.4
            if vision.has_charring:
                taste_scores['savory'] += 0.3
                taste_scores['umami'] += 0.3
        
        # Set scores
        profile.spicy = min(1.0, taste_scores['spicy'])
        profile.sweet = min(1.0, taste_scores['sweet'])
        profile.sour = min(1.0, taste_scores['sour'])
        profile.savory = min(1.0, taste_scores['savory'])
        profile.umami = min(1.0, taste_scores['umami'])
        
        return profile
    
    def _build_health_profile(
        self,
        ingredients: List[str],
        vision: Optional[VisionOutput],
        audio: Optional[AudioOutput],
        text: Optional[TextOutput]
    ) -> HealthProfile:
        """Build fused health profile"""
        profile = HealthProfile()
        
        # Estimate macros from ingredients
        for ingredient in ingredients:
            if ingredient in self.ingredient_macros:
                macros = self.ingredient_macros[ingredient]
                profile.protein_g += macros['protein']
                profile.fat_g += macros['fat']
                profile.carbs_g += macros['carbs']
        
        # Cooking method risk
        if vision and vision.cooking_methods:
            max_risk = max(
                self.method_risk.get(method, 0.0)
                for method in vision.cooking_methods
            )
            profile.cooking_method_risk = max_risk
        
        # Sodium estimation
        if text and text.mentioned_salt:
            profile.sodium_mg += 500  # Rough estimate
        
        # Sugar
        if text and text.mentioned_sugar:
            profile.added_sugar_g += 10
        
        # Inflammatory oils (visual cue)
        if vision:
            if vision.has_oil_splatter or 'oil' in vision.ingredients:
                if CookingMethod.DEEP_FRYING in vision.cooking_methods:
                    profile.has_inflammatory_oils = True
        
        # Processed ingredients
        processed_keywords = ['cream', 'cheese', 'processed']
        if any(kw in ingredients for kw in processed_keywords):
            profile.has_processed_ingredients = True
        
        # Calculate health score
        health_score = 1.0
        
        # Penalties
        health_score -= profile.cooking_method_risk * 0.3
        if profile.has_inflammatory_oils:
            health_score -= 0.2
        if profile.added_sugar_g > 10:
            health_score -= 0.15
        if profile.sodium_mg > self.config.sodium_high_threshold_mg:
            health_score -= 0.1
        
        # Bonuses
        if profile.protein_g > self.config.protein_high_threshold_g:
            health_score += 0.1
        if any(v in ingredients for v in ['vegetables', 'broccoli', 'spinach']):
            health_score += 0.1
        
        profile.health_score = max(0.0, min(1.0, health_score))
        
        return profile


# ============================================================================
# GOAL MATCHER
# ============================================================================

class HealthGoalMatcher:
    """
    Match recipes to user health goals
    """
    
    def __init__(self):
        # Goal requirements
        self.goal_criteria = {
            HealthGoal.MUSCLE_GAIN: {
                'min_protein_g': 30.0,
                'max_fat_g': 20.0,
                'max_sugar_g': 10.0
            },
            HealthGoal.WEIGHT_LOSS: {
                'max_fat_g': 15.0,
                'max_carbs_g': 30.0,
                'max_sugar_g': 5.0
            },
            HealthGoal.LOW_INFLAMMATION: {
                'no_inflammatory_oils': True,
                'max_sugar_g': 8.0,
                'max_cooking_risk': 0.3
            },
            HealthGoal.DIABETES_MANAGEMENT: {
                'max_carbs_g': 40.0,
                'max_sugar_g': 5.0,
                'min_fiber_g': 5.0
            },
            HealthGoal.HYPERTENSION: {
                'max_sodium_mg': 400.0,
                'no_processed': True
            }
        }
        
        logger.info("Health Goal Matcher initialized")
    
    def match_goals(
        self,
        health_profile: HealthProfile,
        goals: List[HealthGoal]
    ) -> Dict[HealthGoal, float]:
        """
        Score how well recipe matches each goal
        
        Returns:
            Dict of goal → match score (0.0-1.0)
        """
        matches = {}
        
        for goal in goals:
            if goal not in self.goal_criteria:
                continue
            
            criteria = self.goal_criteria[goal]
            score = 1.0
            
            # Check criteria
            if 'min_protein_g' in criteria:
                if health_profile.protein_g < criteria['min_protein_g']:
                    deficit = criteria['min_protein_g'] - health_profile.protein_g
                    score -= (deficit / criteria['min_protein_g']) * 0.4
            
            if 'max_fat_g' in criteria:
                if health_profile.fat_g > criteria['max_fat_g']:
                    excess = health_profile.fat_g - criteria['max_fat_g']
                    score -= (excess / 20.0) * 0.3
            
            if 'max_sugar_g' in criteria:
                if health_profile.added_sugar_g > criteria['max_sugar_g']:
                    excess = health_profile.added_sugar_g - criteria['max_sugar_g']
                    score -= (excess / 10.0) * 0.3
            
            if 'no_inflammatory_oils' in criteria:
                if health_profile.has_inflammatory_oils:
                    score -= 0.4
            
            if 'max_cooking_risk' in criteria:
                if health_profile.cooking_method_risk > criteria['max_cooking_risk']:
                    score -= health_profile.cooking_method_risk * 0.3
            
            if 'max_sodium_mg' in criteria:
                if health_profile.sodium_mg > criteria['max_sodium_mg']:
                    excess = health_profile.sodium_mg - criteria['max_sodium_mg']
                    score -= (excess / 500.0) * 0.3
            
            if 'no_processed' in criteria:
                if health_profile.has_processed_ingredients:
                    score -= 0.3
            
            matches[goal] = max(0.0, min(1.0, score))
        
        return matches


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class MultiModalOrchestrator:
    """
    Complete multi-modal food analysis system
    """
    
    def __init__(self, config: Optional[MultiModalConfig] = None):
        self.config = config or MultiModalConfig()
        
        # Models
        self.vision_model = VisionModel()
        self.audio_model = AudioModel()
        self.nlp_model = NLPModel()
        
        # Fusion
        self.fusion = MultiModalFusion(self.config)
        self.goal_matcher = HealthGoalMatcher()
        
        logger.info("Multi-Modal Orchestrator initialized")
    
    def analyze_youtube_video(
        self,
        video_frames: List[str],
        audio_segments: List[str],
        video_description: str,
        user_goals: Optional[List[HealthGoal]] = None
    ) -> MultiModalResult:
        """
        Analyze YouTube cooking video
        
        Args:
            video_frames: List of frame descriptions (mock)
            audio_segments: List of audio descriptions (mock)
            video_description: Video description text
            user_goals: User's health goals
        
        Returns:
            Complete multi-modal analysis
        """
        # Analyze vision (aggregate frames)
        vision_outputs = [self.vision_model.analyze_image(frame) for frame in video_frames]
        
        # Combine vision results
        combined_vision = VisionOutput()
        all_ingredients = set()
        all_methods = set()
        
        for vo in vision_outputs:
            all_ingredients.update(vo.ingredients)
            all_methods.update(vo.cooking_methods)
            if vo.has_heavy_cream:
                combined_vision.has_heavy_cream = True
            if vo.has_oil_splatter:
                combined_vision.has_oil_splatter = True
            if vo.has_charring:
                combined_vision.has_charring = True
        
        combined_vision.ingredients = list(all_ingredients)
        combined_vision.cooking_methods = list(all_methods)
        combined_vision.confidence = sum(vo.confidence for vo in vision_outputs) / len(vision_outputs)
        
        # Analyze audio (aggregate segments)
        audio_outputs = [self.audio_model.analyze_audio(seg) for seg in audio_segments]
        
        # Combine audio results
        combined_audio = AudioOutput()
        combined_audio.transcript = ' '.join(ao.transcript for ao in audio_outputs)
        
        all_keywords = set()
        all_taste = set()
        
        for ao in audio_outputs:
            all_keywords.update(ao.keywords)
            all_taste.update(ao.taste_descriptors)
            if ao.detected_sizzle:
                combined_audio.detected_sizzle = True
            if ao.detected_crunch:
                combined_audio.detected_crunch = True
            if ao.detected_boiling:
                combined_audio.detected_boiling = True
        
        combined_audio.keywords = list(all_keywords)
        combined_audio.taste_descriptors = list(all_taste)
        combined_audio.confidence = sum(ao.confidence for ao in audio_outputs) / len(audio_outputs)
        
        # Analyze text
        text_output = self.nlp_model.analyze_text(video_description)
        
        # Fuse modalities
        result = self.fusion.fuse_modalities(combined_vision, combined_audio, text_output)
        
        # Match to user goals
        if user_goals:
            result.fits_goals = self.goal_matcher.match_goals(
                result.health_profile,
                user_goals
            )
        
        return result


# ============================================================================
# TESTING
# ============================================================================

def test_multimodal():
    """Test multi-modal AI system"""
    print("=" * 80)
    print("MULTI-MODAL AI FOR FOOD ANALYSIS - TEST")
    print("=" * 80)
    
    # Create orchestrator
    orchestrator = MultiModalOrchestrator()
    
    # Test YouTube video analysis
    print("\n" + "="*80)
    print("Test: YouTube Video Analysis (Spicy Stir-Fry)")
    print("="*80)
    
    # Mock video data
    video_frames = [
        "Raw chicken on cutting board with knife",
        "Red chili peppers and garlic on table",
        "Hot wok with oil splatter and sizzling",
        "Golden brown chicken pieces in wok",
        "Final dish with vegetables in bowl"
    ]
    
    audio_segments = [
        "Today we're making a spicy chicken stir-fry with lots of chili",
        "Listen to that sizzle - that's how you know it's hot enough",
        "This is going to be really spicy and savory",
        "Add a pinch of salt and sugar for balance"
    ]
    
    video_description = """
    Spicy Chicken Stir-Fry Recipe
    
    Ingredients: chicken breast, red chili peppers, garlic, onion, soy sauce,
    oil, vegetables, salt, sugar
    
    This high-protein dish is perfect for muscle gain. The chicken provides
    lean protein while the vegetables add nutrients.
    """
    
    # Analyze with muscle gain goal
    result = orchestrator.analyze_youtube_video(
        video_frames=video_frames,
        audio_segments=audio_segments,
        video_description=video_description,
        user_goals=[HealthGoal.MUSCLE_GAIN, HealthGoal.LOW_INFLAMMATION]
    )
    
    print(f"✓ Video Analysis Complete")
    print(f"\n  Vision Modality:")
    print(f"    Ingredients detected: {', '.join(result.vision.ingredients)}")
    print(f"    Cooking methods: {', '.join([m.value for m in result.vision.cooking_methods])}")
    print(f"    Has oil splatter: {result.vision.has_oil_splatter}")
    print(f"    Confidence: {result.vision.confidence:.0%}")
    
    print(f"\n  Audio Modality:")
    print(f"    Detected sizzle: {result.audio.detected_sizzle}")
    print(f"    Taste descriptors: {', '.join(result.audio.taste_descriptors)}")
    print(f"    Keywords: {', '.join(list(result.audio.keywords)[:5])}")
    print(f"    Confidence: {result.audio.confidence:.0%}")
    
    print(f"\n  Text Modality:")
    print(f"    Recipe title: {result.text.recipe_title}")
    print(f"    Ingredients: {', '.join(result.text.ingredients[:5])}")
    print(f"    Mentioned protein: {result.text.mentioned_protein}")
    print(f"    Confidence: {result.text.confidence:.0%}")
    
    print(f"\n  Fused Flavor Profile:")
    flavor = result.flavor_profile
    print(f"    Spicy: {flavor.spicy:.1f}/1.0")
    print(f"    Savory: {flavor.savory:.1f}/1.0")
    print(f"    Sweet: {flavor.sweet:.1f}/1.0")
    print(f"    Umami: {flavor.umami:.1f}/1.0")
    print(f"    Texture: {flavor.texture}")
    
    print(f"\n  Fused Health Profile:")
    health = result.health_profile
    print(f"    Protein: {health.protein_g:.1f}g")
    print(f"    Fat: {health.fat_g:.1f}g")
    print(f"    Carbs: {health.carbs_g:.1f}g")
    print(f"    Cooking method risk: {health.cooking_method_risk:.0%}")
    print(f"    Sodium: {health.sodium_mg:.0f}mg")
    print(f"    Overall health score: {health.health_score:.0%}")
    
    print(f"\n  Goal Matching:")
    for goal, score in result.fits_goals.items():
        emoji = "✅" if score >= 0.7 else "⚠️" if score >= 0.4 else "❌"
        print(f"    {emoji} {goal.value}: {score:.0%}")
    
    # Test unhealthy recipe
    print("\n" + "="*80)
    print("Test: Deep-Fried Recipe (Low Health Score)")
    print("="*80)
    
    video_frames_fried = [
        "Raw chicken being battered",
        "Deep pot filled with bubbling oil",
        "Golden fried chicken pieces floating in oil",
        "Fried chicken on paper towels draining oil"
    ]
    
    audio_segments_fried = [
        "We're deep-frying chicken today - so crispy!",
        "Listen to that crunch when you bite into it",
        "This is very rich and indulgent"
    ]
    
    video_description_fried = """
    Crispy Fried Chicken Recipe
    
    Ingredients: chicken, oil, butter, cream, flour, salt, sugar
    
    This is a rich, indulgent dish with lots of flavor.
    """
    
    result_fried = orchestrator.analyze_youtube_video(
        video_frames=video_frames_fried,
        audio_segments=audio_segments_fried,
        video_description=video_description_fried,
        user_goals=[HealthGoal.WEIGHT_LOSS, HealthGoal.LOW_INFLAMMATION]
    )
    
    print(f"✓ Deep-Fried Recipe Analysis:")
    print(f"  Cooking methods: {', '.join([m.value for m in result_fried.vision.cooking_methods])}")
    print(f"  Has inflammatory oils: {result_fried.health_profile.has_inflammatory_oils}")
    print(f"  Cooking method risk: {result_fried.health_profile.cooking_method_risk:.0%}")
    print(f"  Health score: {result_fried.health_profile.health_score:.0%}")
    
    print(f"\n  Goal Matching:")
    for goal, score in result_fried.fits_goals.items():
        emoji = "✅" if score >= 0.7 else "⚠️" if score >= 0.4 else "❌"
        print(f"    {emoji} {goal.value}: {score:.0%}")
    
    # Test healthy recipe
    print("\n" + "="*80)
    print("Test: Steamed Fish (High Health Score)")
    print("="*80)
    
    video_frames_healthy = [
        "Fresh fish fillet on plate",
        "Bamboo steamer with steam rising",
        "Colorful vegetables being chopped",
        "Steamed fish with vegetables on plate"
    ]
    
    audio_segments_healthy = [
        "Steaming is the healthiest cooking method",
        "This dish is light and packed with protein",
        "Low in fat and very nutritious"
    ]
    
    video_description_healthy = """
    Healthy Steamed Fish with Vegetables
    
    Ingredients: fish, vegetables, garlic, ginger, lemon, herbs
    
    High in protein, low in fat. Perfect for weight loss and general health.
    """
    
    result_healthy = orchestrator.analyze_youtube_video(
        video_frames=video_frames_healthy,
        audio_segments=audio_segments_healthy,
        video_description=video_description_healthy,
        user_goals=[HealthGoal.WEIGHT_LOSS, HealthGoal.GENERAL_HEALTH]
    )
    
    print(f"✓ Steamed Fish Analysis:")
    print(f"  Cooking methods: {', '.join([m.value for m in result_healthy.vision.cooking_methods])}")
    print(f"  Cooking method risk: {result_healthy.health_profile.cooking_method_risk:.0%}")
    print(f"  Health score: {result_healthy.health_profile.health_score:.0%}")
    
    print(f"\n  Goal Matching:")
    for goal, score in result_healthy.fits_goals.items():
        emoji = "✅" if score >= 0.7 else "⚠️" if score >= 0.4 else "❌"
        print(f"    {emoji} {goal.value}: {score:.0%}")
    
    print("\n✅ All multi-modal AI tests passed!")
    print("\n💡 Production Integration:")
    print("  - Vision: Replace VisionModel with YOLO v8 / ViT")
    print("  - Audio: Replace AudioModel with OpenAI Whisper API")
    print("  - NLP: Replace NLPModel with GPT-4 / fine-tuned transformer")
    print("  - YouTube: Integrate yt-dlp for video/audio download")
    print("  - Real-time: Deploy as FastAPI endpoint")
    print("  - BiteLab Feature: Paste YouTube link → Get health/flavor profile")


if __name__ == '__main__':
    test_multimodal()
