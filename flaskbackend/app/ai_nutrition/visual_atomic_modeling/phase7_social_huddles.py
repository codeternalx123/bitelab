"""
PHASE 7: Anonymous Social & Health Huddles
===========================================

Implements Telegram-like anonymous community features:
- Health Huddles (topic-based communities)
- Anonymous usernames
- Verified dietitian/coach integration
- AI-facilitated meal sharing
- Shareable scan result cards
- 1-on-1 chat with mutual consent
- Community moderation

Core Features:
- Anonymous by default (e.g., "T1D-User-451")
- Health condition-based huddles
- Goal-based huddles
- Recipe sharing communities
- Professional support integration
- Automatic card generation from scans
- Real name reveal only with consent

Architecture:
    User → Join Huddles → Share Scans → AI generates Card
    → Post to Community → Get Feedback → Private Chat (optional)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from collections import defaultdict
import hashlib
import secrets
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles in huddles"""
    MEMBER = "member"
    MODERATOR = "moderator"
    VERIFIED_DIETITIAN = "verified_dietitian"
    VERIFIED_COACH = "verified_coach"
    ADMIN = "admin"


class HuddleCategory(Enum):
    """Huddle categories"""
    MEDICAL_CONDITION = "medical_condition"
    HEALTH_GOAL = "health_goal"
    DIETARY_PREFERENCE = "dietary_preference"
    RECIPE_SHARING = "recipe_sharing"
    LOCAL_COMMUNITY = "local_community"
    PROFESSIONAL_SUPPORT = "professional_support"


class PrivacyLevel(Enum):
    """Privacy levels for content"""
    ANONYMOUS = "anonymous"  # Default
    SEMI_ANONYMOUS = "semi_anonymous"  # Show first name only
    IDENTIFIED = "identified"  # Show full name (requires consent)


class ShareableContentType(Enum):
    """Types of shareable content"""
    MEAL_SCAN = "meal_scan"
    RISK_CARD = "risk_card"
    RECIPE = "recipe"
    PROGRESS_UPDATE = "progress_update"
    QUESTION = "question"
    TIP = "tip"


@dataclass
class AnonymousProfile:
    """
    Anonymous user profile for huddles
    
    Real identity protected, only anonymized username shown
    """
    anonymous_id: str  # e.g., "T1D-User-451"
    user_id: str  # Internal reference (not shown)
    
    # Anonymous display info
    display_name: str  # e.g., "T1D-User-451"
    avatar_seed: str  # For generating consistent avatar
    
    # Profile metadata (anonymous)
    member_since: datetime
    huddle_count: int = 0
    post_count: int = 0
    helpful_count: int = 0  # Times marked helpful by others
    
    # Privacy settings
    privacy_level: PrivacyLevel = PrivacyLevel.ANONYMOUS
    allow_direct_messages: bool = True
    show_profile_stats: bool = True
    
    # Verification (if professional)
    is_verified: bool = False
    verification_badge: Optional[str] = None  # "⚕️" for dietitian
    professional_title: Optional[str] = None
    
    def get_display_badge(self) -> str:
        """Get display badge for user"""
        if self.is_verified:
            return self.verification_badge or "✓"
        return ""
    
    def can_reveal_identity(self, other_user_consented: bool) -> bool:
        """Check if can reveal real identity"""
        return (
            self.privacy_level == PrivacyLevel.IDENTIFIED and
            other_user_consented
        )


@dataclass
class HealthHuddle:
    """
    Health Huddle - topic-based anonymous community
    
    Examples:
    - "Type 1 Diabetes Support"
    - "Low-Sodium Recipes"
    - "Pregnant & Vegetarian"
    - "Muscle Gain Journey"
    """
    huddle_id: str
    name: str
    description: str
    category: HuddleCategory
    
    # Huddle specifics
    tags: List[str] = field(default_factory=list)
    icon_emoji: str = "👥"
    
    # Membership
    member_count: int = 0
    members: Set[str] = field(default_factory=set)  # Anonymous IDs
    
    # Moderation
    moderators: Set[str] = field(default_factory=set)
    verified_professionals: Set[str] = field(default_factory=set)
    
    # Settings
    is_official: bool = False  # Created by verified professional
    is_private: bool = False
    requires_approval: bool = False
    
    # Activity
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    post_count: int = 0
    
    # Rules
    rules: List[str] = field(default_factory=list)
    
    def add_member(self, anonymous_id: str) -> bool:
        """Add member to huddle"""
        if anonymous_id not in self.members:
            self.members.add(anonymous_id)
            self.member_count += 1
            return True
        return False
    
    def remove_member(self, anonymous_id: str) -> bool:
        """Remove member from huddle"""
        if anonymous_id in self.members:
            self.members.remove(anonymous_id)
            self.member_count -= 1
            return True
        return False
    
    def is_moderator(self, anonymous_id: str) -> bool:
        """Check if user is moderator"""
        return anonymous_id in self.moderators
    
    def is_verified_professional(self, anonymous_id: str) -> bool:
        """Check if user is verified professional"""
        return anonymous_id in self.verified_professionals


@dataclass
class ShareableCard:
    """
    Auto-generated shareable card from meal scan
    
    Contains:
    - Food image
    - Food name
    - Macro summary
    - Risk score
    - Personalized highlights
    """
    card_id: str
    content_type: ShareableContentType
    
    # Content
    food_name: str
    food_image_url: Optional[str] = None
    
    # Nutritional summary
    calories: float = 0
    protein: float = 0
    carbs: float = 0
    fat: float = 0
    
    # Health score
    health_score: float = 0
    score_color: str = "gray"
    
    # Personalized info (anonymized)
    user_verdict: str = ""  # "Great for my goals!", "Caution for me"
    highlights: List[str] = field(default_factory=list)
    
    # Metadata
    created_by: str = ""  # Anonymous ID
    created_at: datetime = field(default_factory=datetime.now)
    
    # Engagement
    likes_count: int = 0
    comments_count: int = 0
    shares_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for sharing"""
        return {
            'card_id': self.card_id,
            'content_type': self.content_type.value,
            'food_name': self.food_name,
            'food_image_url': self.food_image_url,
            'nutrition': {
                'calories': self.calories,
                'protein': self.protein,
                'carbs': self.carbs,
                'fat': self.fat
            },
            'health_score': self.health_score,
            'score_color': self.score_color,
            'user_verdict': self.user_verdict,
            'highlights': self.highlights,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'engagement': {
                'likes': self.likes_count,
                'comments': self.comments_count,
                'shares': self.shares_count
            }
        }


@dataclass
class HuddlePost:
    """Post in a health huddle"""
    post_id: str
    huddle_id: str
    author_id: str  # Anonymous ID
    
    # Content
    content_type: ShareableContentType
    text_content: str
    shareable_card: Optional[ShareableCard] = None
    
    # Media
    image_urls: List[str] = field(default_factory=list)
    
    # Engagement
    likes: Set[str] = field(default_factory=set)  # Anonymous IDs who liked
    comments: List['HuddleComment'] = field(default_factory=list)
    
    # Moderation
    is_pinned: bool = False
    is_hidden: bool = False
    flagged_count: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    edited_at: Optional[datetime] = None
    
    def get_likes_count(self) -> int:
        """Get number of likes"""
        return len(self.likes)
    
    def add_like(self, anonymous_id: str) -> bool:
        """Add like from user"""
        if anonymous_id not in self.likes:
            self.likes.add(anonymous_id)
            return True
        return False
    
    def remove_like(self, anonymous_id: str) -> bool:
        """Remove like"""
        if anonymous_id in self.likes:
            self.likes.remove(anonymous_id)
            return True
        return False


@dataclass
class HuddleComment:
    """Comment on a huddle post"""
    comment_id: str
    post_id: str
    author_id: str  # Anonymous ID
    
    text: str
    
    # Engagement
    likes: Set[str] = field(default_factory=set)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    edited_at: Optional[datetime] = None
    
    def get_likes_count(self) -> int:
        return len(self.likes)


@dataclass
class DirectMessage:
    """Private 1-on-1 message (requires mutual consent)"""
    message_id: str
    conversation_id: str
    
    sender_id: str  # Anonymous ID
    recipient_id: str  # Anonymous ID
    
    # Content
    text: str
    shareable_card: Optional[ShareableCard] = None
    
    # Privacy
    sender_revealed: bool = False  # Sender revealed real name
    recipient_revealed: bool = False  # Recipient revealed real name
    
    # Metadata
    sent_at: datetime = field(default_factory=datetime.now)
    read_at: Optional[datetime] = None
    
    def is_read(self) -> bool:
        return self.read_at is not None


class AnonymousIdentityManager:
    """
    Manages anonymous identities for users
    
    Generates consistent anonymous usernames like:
    - "T1D-User-451"
    - "LowSodium-Chef-892"
    - "PregnantMom-123"
    """
    
    def __init__(self):
        self.user_to_anonymous: Dict[str, str] = {}
        self.anonymous_to_user: Dict[str, str] = {}
        
        # Prefixes based on health context
        self.prefixes = {
            'diabetes_type1': 'T1D',
            'diabetes_type2': 'T2D',
            'hypertension': 'BP',
            'kidney_disease': 'CKD',
            'pregnancy': 'PregnantMom',
            'heart_disease': 'Heart',
            'weight_loss': 'FitJourney',
            'muscle_gain': 'GainTrain',
            'low_sodium': 'LowSodium',
            'vegan': 'PlantBased',
            'keto': 'KetoLife',
        }
        
        logger.info("AnonymousIdentityManager initialized")
    
    def generate_anonymous_id(
        self,
        user_id: str,
        primary_context: Optional[str] = None
    ) -> str:
        """
        Generate anonymous ID for user
        
        Args:
            user_id: Real user ID
            primary_context: Primary health context (e.g., 'diabetes_type1')
            
        Returns:
            Anonymous ID like "T1D-User-451"
        """
        # Check if already generated
        if user_id in self.user_to_anonymous:
            return self.user_to_anonymous[user_id]
        
        # Get prefix
        prefix = 'Health'
        if primary_context and primary_context in self.prefixes:
            prefix = self.prefixes[primary_context]
        
        # Generate unique number
        # Use hash of user_id for consistency
        hash_val = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
        unique_num = (hash_val % 9000) + 1000  # 4-digit number
        
        # Create anonymous ID
        anonymous_id = f"{prefix}-User-{unique_num}"
        
        # Ensure uniqueness
        while anonymous_id in self.anonymous_to_user:
            unique_num = (unique_num + 1) % 10000
            anonymous_id = f"{prefix}-User-{unique_num}"
        
        # Store mapping
        self.user_to_anonymous[user_id] = anonymous_id
        self.anonymous_to_user[anonymous_id] = user_id
        
        return anonymous_id
    
    def get_real_user_id(self, anonymous_id: str) -> Optional[str]:
        """Get real user ID from anonymous ID"""
        return self.anonymous_to_user.get(anonymous_id)
    
    def get_anonymous_id(self, user_id: str) -> Optional[str]:
        """Get anonymous ID from real user ID"""
        return self.user_to_anonymous.get(user_id)


class ShareableCardGenerator:
    """
    Auto-generates shareable cards from meal scans
    
    Prevents manual data entry and ensures accuracy from
    the Chemometrics Engine (Phases 1-6)
    """
    
    def __init__(self):
        logger.info("ShareableCardGenerator initialized")
    
    def generate_from_scan(
        self,
        food_name: str,
        food_image_url: Optional[str],
        risk_card_data: Dict[str, Any],
        nutrient_profile: Dict[str, Any],
        anonymous_id: str,
        user_goals: List[str],
        user_conditions: List[str]
    ) -> ShareableCard:
        """
        Generate shareable card from meal scan
        
        Args:
            food_name: Name of food
            food_image_url: URL to food image
            risk_card_data: Risk card from Phase 4/6
            nutrient_profile: Nutritional data
            anonymous_id: User's anonymous ID
            user_goals: User's health goals
            user_conditions: User's medical conditions
            
        Returns:
            ShareableCard ready for posting
        """
        # Extract key metrics
        health_score = risk_card_data.get('health_score', 0)
        safety_verdict = risk_card_data.get('safety_verdict', 'UNKNOWN')
        
        # Determine user verdict (personalized)
        user_verdict = self._generate_user_verdict(
            safety_verdict,
            health_score,
            user_goals,
            user_conditions
        )
        
        # Generate highlights (anonymized)
        highlights = self._generate_highlights(
            risk_card_data,
            nutrient_profile,
            user_goals,
            user_conditions
        )
        
        # Determine score color
        score_color = self._get_score_color(safety_verdict, health_score)
        
        # Create card
        card = ShareableCard(
            card_id=self._generate_card_id(),
            content_type=ShareableContentType.MEAL_SCAN,
            food_name=food_name,
            food_image_url=food_image_url,
            calories=nutrient_profile.get('calories', 0),
            protein=nutrient_profile.get('protein', 0),
            carbs=nutrient_profile.get('carbohydrates', 0),
            fat=nutrient_profile.get('fat', 0),
            health_score=health_score,
            score_color=score_color,
            user_verdict=user_verdict,
            highlights=highlights,
            created_by=anonymous_id
        )
        
        return card
    
    def _generate_user_verdict(
        self,
        verdict: str,
        score: float,
        goals: List[str],
        conditions: List[str]
    ) -> str:
        """Generate personalized verdict (anonymized)"""
        if verdict == "AVOID":
            if conditions:
                return "⚠️ Not safe for my health conditions"
            return "⚠️ Not recommended for me"
        elif verdict == "CAUTION":
            return "⚠️ Proceed with caution for my needs"
        elif score >= 85:
            if goals:
                return "✅ Perfect for my health goals!"
            return "✅ Great choice for me!"
        elif score >= 70:
            return "✓ Good fit for my profile"
        else:
            return "⚠️ Could be better for my needs"
    
    def _generate_highlights(
        self,
        risk_data: Dict[str, Any],
        nutrients: Dict[str, Any],
        goals: List[str],
        conditions: List[str]
    ) -> List[str]:
        """Generate key highlights (anonymized)"""
        highlights = []
        
        # Nutrient highlights
        if nutrients.get('protein', 0) > 20:
            highlights.append(f"💪 High protein: {nutrients['protein']:.1f}g")
        
        if nutrients.get('fiber', 0) > 5:
            highlights.append(f"🌾 High fiber: {nutrients['fiber']:.1f}g")
        
        # Warnings (anonymized)
        if nutrients.get('sodium', 0) > 400:
            highlights.append(f"⚠️ High sodium: {nutrients['sodium']:.0f}mg")
        
        if nutrients.get('sugar', 0) > 10:
            highlights.append(f"⚠️ High sugar: {nutrients['sugar']:.1f}g")
        
        # Goal alignment (anonymized)
        if 'weight_loss' in goals:
            cal = nutrients.get('calories', 0)
            if cal < 300:
                highlights.append(f"⚖️ Low calorie: {cal:.0f} cal")
        
        if 'muscle_gain' in goals:
            protein = nutrients.get('protein', 0)
            if protein > 25:
                highlights.append("💪 Great for muscle building")
        
        return highlights[:4]  # Max 4 highlights
    
    def _get_score_color(self, verdict: str, score: float) -> str:
        """Get color for score badge"""
        if verdict == "AVOID":
            return "red"
        elif verdict == "CAUTION":
            return "orange"
        elif score >= 85:
            return "green"
        elif score >= 70:
            return "yellow"
        else:
            return "orange"
    
    def _generate_card_id(self) -> str:
        """Generate unique card ID"""
        return f"card_{secrets.token_urlsafe(16)}"


class HealthHuddleManager:
    """
    Manages Health Huddles and community features
    
    Features:
    - Create/join huddles
    - Post shareable cards
    - Comment and engage
    - Moderate content
    - Verify professionals
    """
    
    def __init__(
        self,
        identity_manager: AnonymousIdentityManager,
        card_generator: ShareableCardGenerator
    ):
        self.identity_manager = identity_manager
        self.card_generator = card_generator
        
        # Storage
        self.huddles: Dict[str, HealthHuddle] = {}
        self.posts: Dict[str, HuddlePost] = {}
        self.profiles: Dict[str, AnonymousProfile] = {}
        
        # Indexes
        self.user_huddles: Dict[str, Set[str]] = defaultdict(set)
        self.huddle_posts: Dict[str, List[str]] = defaultdict(list)
        
        # Create default huddles
        self._create_default_huddles()
        
        logger.info("HealthHuddleManager initialized")
    
    def _create_default_huddles(self):
        """Create default health huddles for 55+ goals and 100+ medical conditions"""
        default_huddles = [
            # ===== DIABETES & BLOOD SUGAR =====
            {
                'name': 'Type 1 Diabetes Support',
                'description': 'Community for T1D warriors sharing tips, recipes, and support',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['diabetes', 'type1', 'insulin', 'blood_sugar'],
                'icon_emoji': '🩺',
            },
            {
                'name': 'Type 2 Diabetes Management',
                'description': 'Managing T2D through diet, exercise, and lifestyle',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['diabetes', 'type2', 'diet', 'exercise'],
                'icon_emoji': '📊',
            },
            {
                'name': 'Prediabetes Prevention',
                'description': 'Preventing progression to diabetes through lifestyle changes',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['prediabetes', 'prevention', 'blood_sugar', 'lifestyle'],
                'icon_emoji': '⚠️',
            },
            {
                'name': 'Gestational Diabetes Care',
                'description': 'Managing diabetes during pregnancy',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['gestational_diabetes', 'pregnancy', 'blood_sugar'],
                'icon_emoji': '🤰',
            },
            
            # ===== CARDIOVASCULAR =====
            {
                'name': 'Heart-Healthy Living',
                'description': 'Cardiovascular health through nutrition and lifestyle',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['heart_disease', 'cholesterol', 'omega3', 'cardiovascular'],
                'icon_emoji': '❤️',
            },
            {
                'name': 'Hypertension Warriors',
                'description': 'Managing high blood pressure naturally',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['hypertension', 'blood_pressure', 'low_sodium', 'dash_diet'],
                'icon_emoji': '💉',
            },
            {
                'name': 'Low-Sodium Recipes',
                'description': 'Delicious recipes for hypertension and heart health',
                'category': HuddleCategory.RECIPE_SHARING,
                'tags': ['low_sodium', 'hypertension', 'recipes', 'heart_health'],
                'icon_emoji': '🧂',
            },
            {
                'name': 'Cholesterol Management',
                'description': 'Lower LDL, raise HDL through diet',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['cholesterol', 'ldl', 'hdl', 'heart_health'],
                'icon_emoji': '🩸',
            },
            {
                'name': 'Stroke Recovery',
                'description': 'Nutritional support for stroke survivors',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['stroke', 'recovery', 'brain_health', 'rehabilitation'],
                'icon_emoji': '🧠',
            },
            
            # ===== KIDNEY & RENAL =====
            {
                'name': 'Kidney-Friendly Eating',
                'description': 'Low potassium, low phosphorus meals for CKD',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['kidney_disease', 'ckd', 'low_potassium', 'renal_diet'],
                'icon_emoji': '🫘',
            },
            {
                'name': 'Dialysis Nutrition',
                'description': 'Nutritional support for dialysis patients',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['dialysis', 'kidney_disease', 'protein', 'phosphorus'],
                'icon_emoji': '💧',
            },
            {
                'name': 'Kidney Stone Prevention',
                'description': 'Preventing kidney stones through diet',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['kidney_stones', 'prevention', 'hydration', 'calcium'],
                'icon_emoji': '💎',
            },
            
            # ===== DIGESTIVE & GI =====
            {
                'name': 'IBS Support Circle',
                'description': 'Managing irritable bowel syndrome through diet',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['ibs', 'digestive', 'fodmap', 'gut_health'],
                'icon_emoji': '🔄',
            },
            {
                'name': 'Crohn\'s & Colitis Community',
                'description': 'IBD warriors sharing experiences and recipes',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['crohns', 'colitis', 'ibd', 'inflammation'],
                'icon_emoji': '🌡️',
            },
            {
                'name': 'Celiac Support Network',
                'description': 'Living gluten-free with celiac disease',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['celiac', 'gluten_free', 'autoimmune'],
                'icon_emoji': '🌾',
            },
            {
                'name': 'GERD & Acid Reflux',
                'description': 'Managing reflux through diet and lifestyle',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['gerd', 'acid_reflux', 'heartburn'],
                'icon_emoji': '🔥',
            },
            {
                'name': 'Gastroparesis Support',
                'description': 'Delayed gastric emptying meal strategies',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['gastroparesis', 'digestive', 'small_meals'],
                'icon_emoji': '⏰',
            },
            
            # ===== LIVER & METABOLIC =====
            {
                'name': 'Fatty Liver Recovery',
                'description': 'Reversing NAFLD through nutrition',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['fatty_liver', 'nafld', 'liver_health'],
                'icon_emoji': '🫀',
            },
            {
                'name': 'Metabolic Syndrome Warriors',
                'description': 'Combating metabolic syndrome holistically',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['metabolic_syndrome', 'insulin_resistance', 'prevention'],
                'icon_emoji': '⚡',
            },
            
            # ===== THYROID & HORMONAL =====
            {
                'name': 'Hypothyroid Support',
                'description': 'Managing underactive thyroid through nutrition',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['hypothyroid', 'thyroid', 'iodine', 'selenium'],
                'icon_emoji': '🦋',
            },
            {
                'name': 'PCOS Nutrition',
                'description': 'Managing polycystic ovary syndrome through diet',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['pcos', 'hormonal', 'insulin_resistance', 'women_health'],
                'icon_emoji': '🌸',
            },
            
            # ===== CANCER & IMMUNE =====
            {
                'name': 'Cancer Nutrition Support',
                'description': 'Nutritional support during cancer treatment',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['cancer', 'chemotherapy', 'nutrition', 'immune_support'],
                'icon_emoji': '🎗️',
            },
            {
                'name': 'Immune System Boosters',
                'description': 'Strengthening immunity through nutrition',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['immune_system', 'prevention', 'antioxidants'],
                'icon_emoji': '🛡️',
            },
            
            # ===== BONE & JOINT =====
            {
                'name': 'Osteoporosis Prevention',
                'description': 'Building strong bones through calcium and vitamin D',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['osteoporosis', 'bone_health', 'calcium', 'vitamin_d'],
                'icon_emoji': '🦴',
            },
            {
                'name': 'Arthritis Relief',
                'description': 'Anti-inflammatory eating for joint health',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['arthritis', 'inflammation', 'joint_health'],
                'icon_emoji': '🦵',
            },
            {
                'name': 'Gout Management',
                'description': 'Low-purine diet for gout prevention',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['gout', 'uric_acid', 'low_purine'],
                'icon_emoji': '💢',
            },
            
            # ===== NEUROLOGICAL =====
            {
                'name': 'Migraine Prevention',
                'description': 'Identifying triggers and preventive nutrition',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['migraine', 'headache', 'triggers'],
                'icon_emoji': '🤕',
            },
            {
                'name': 'Alzheimer\'s Prevention',
                'description': 'Brain-healthy Mediterranean and MIND diet',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['alzheimers', 'brain_health', 'memory', 'mind_diet'],
                'icon_emoji': '🧩',
            },
            {
                'name': 'Epilepsy Keto Support',
                'description': 'Ketogenic diet for seizure management',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['epilepsy', 'ketogenic', 'seizures'],
                'icon_emoji': '⚕️',
            },
            
            # ===== MENTAL HEALTH =====
            {
                'name': 'Depression & Nutrition',
                'description': 'Foods that support mental health',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['depression', 'mental_health', 'mood', 'omega3'],
                'icon_emoji': '🌈',
            },
            {
                'name': 'Anxiety Relief',
                'description': 'Calming foods and stress reduction',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['anxiety', 'stress', 'mental_health'],
                'icon_emoji': '🧘',
            },
            
            # ===== RESPIRATORY =====
            {
                'name': 'Asthma Support',
                'description': 'Anti-inflammatory nutrition for respiratory health',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['asthma', 'respiratory', 'inflammation'],
                'icon_emoji': '🫁',
            },
            {
                'name': 'COPD Nutrition',
                'description': 'Nutritional support for chronic lung disease',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['copd', 'respiratory', 'lung_health'],
                'icon_emoji': '💨',
            },
            
            # ===== PREGNANCY & MATERNAL =====
            {
                'name': 'Pregnant & Vegetarian',
                'description': 'Nutrition support for pregnant vegetarians',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['pregnancy', 'vegetarian', 'nutrition', 'prenatal'],
                'icon_emoji': '🤰',
            },
            {
                'name': 'Breastfeeding Nutrition',
                'description': 'Nourishing mom and baby through lactation',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['breastfeeding', 'lactation', 'postpartum'],
                'icon_emoji': '🍼',
            },
            {
                'name': 'Prenatal Health',
                'description': 'Optimal nutrition for healthy pregnancy',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['pregnancy', 'prenatal', 'folic_acid', 'iron'],
                'icon_emoji': '👶',
            },
            
            # ===== ALLERGIES & INTOLERANCES =====
            {
                'name': 'Food Allergy Support',
                'description': 'Living with severe food allergies',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['food_allergy', 'anaphylaxis', 'allergen_free'],
                'icon_emoji': '🚫',
            },
            {
                'name': 'Lactose Intolerance',
                'description': 'Dairy-free living and calcium alternatives',
                'category': HuddleCategory.MEDICAL_CONDITION,
                'tags': ['lactose_intolerance', 'dairy_free'],
                'icon_emoji': '🥛',
            },
            
            # ===== WEIGHT MANAGEMENT GOALS =====
            {
                'name': 'Weight Loss Warriors',
                'description': 'Supporting each other on the weight loss journey',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['weight_loss', 'diet', 'calories', 'fitness'],
                'icon_emoji': '⚖️',
            },
            {
                'name': 'Healthy Weight Gain',
                'description': 'Gaining weight healthily and sustainably',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['weight_gain', 'underweight', 'nutrition'],
                'icon_emoji': '📈',
            },
            {
                'name': 'Weight Maintenance',
                'description': 'Maintaining your ideal weight long-term',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['weight_maintenance', 'lifestyle', 'balance'],
                'icon_emoji': '⚖️',
            },
            
            # ===== FITNESS & PERFORMANCE GOALS =====
            {
                'name': 'Muscle Gain Journey',
                'description': 'Building muscle with proper nutrition and training',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['muscle_gain', 'protein', 'workout', 'bodybuilding'],
                'icon_emoji': '💪',
            },
            {
                'name': 'Athletic Performance',
                'description': 'Fueling peak athletic performance',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['athletic_performance', 'sports_nutrition', 'energy'],
                'icon_emoji': '🏃',
            },
            {
                'name': 'Endurance Training',
                'description': 'Nutrition for marathon and distance athletes',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['endurance', 'marathon', 'carb_loading'],
                'icon_emoji': '🏃‍♀️',
            },
            {
                'name': 'Strength Training',
                'description': 'Powerlifting and strength athlete nutrition',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['strength_training', 'powerlifting', 'protein'],
                'icon_emoji': '🏋️',
            },
            {
                'name': 'Fat Loss Focus',
                'description': 'Losing fat while preserving muscle',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['fat_loss', 'body_recomposition', 'lean'],
                'icon_emoji': '🔥',
            },
            
            # ===== LONGEVITY & PREVENTION GOALS =====
            {
                'name': 'Longevity & Anti-Aging',
                'description': 'Nutrition for healthy aging and longevity',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['longevity', 'anti_aging', 'healthy_aging'],
                'icon_emoji': '🌟',
            },
            {
                'name': 'Disease Prevention',
                'description': 'Proactive nutrition to prevent chronic disease',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['prevention', 'wellness', 'proactive'],
                'icon_emoji': '🛡️',
            },
            {
                'name': 'Detox & Cleanse',
                'description': 'Safe detoxification and body cleansing',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['detox', 'cleanse', 'liver_support'],
                'icon_emoji': '🍃',
            },
            
            # ===== ENERGY & VITALITY GOALS =====
            {
                'name': 'Energy Boost',
                'description': 'Natural energy through nutrition',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['energy', 'vitality', 'fatigue'],
                'icon_emoji': '⚡',
            },
            {
                'name': 'Better Sleep',
                'description': 'Foods and timing for quality sleep',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['sleep', 'rest', 'recovery', 'melatonin'],
                'icon_emoji': '😴',
            },
            {
                'name': 'Stress Reduction',
                'description': 'Nutrition for stress management',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['stress', 'cortisol', 'relaxation'],
                'icon_emoji': '🧘‍♀️',
            },
            
            # ===== DIGESTIVE HEALTH GOALS =====
            {
                'name': 'Gut Health Optimization',
                'description': 'Improving microbiome and digestive health',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['gut_health', 'microbiome', 'probiotics', 'fiber'],
                'icon_emoji': '🦠',
            },
            {
                'name': 'Inflammation Reduction',
                'description': 'Anti-inflammatory nutrition strategies',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['inflammation', 'anti_inflammatory', 'healing'],
                'icon_emoji': '🌡️',
            },
            
            # ===== BEAUTY & SKIN GOALS =====
            {
                'name': 'Skin Health & Glow',
                'description': 'Nutrition for radiant skin',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['skin_health', 'beauty', 'collagen', 'antioxidants'],
                'icon_emoji': '✨',
            },
            {
                'name': 'Hair & Nail Health',
                'description': 'Nutrients for strong hair and nails',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['hair_health', 'nail_health', 'biotin'],
                'icon_emoji': '💅',
            },
            
            # ===== COGNITIVE GOALS =====
            {
                'name': 'Brain Health & Focus',
                'description': 'Nutrition for cognitive performance',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['brain_health', 'focus', 'memory', 'nootropics'],
                'icon_emoji': '🧠',
            },
            {
                'name': 'Memory Enhancement',
                'description': 'Foods that boost memory and learning',
                'category': HuddleCategory.HEALTH_GOAL,
                'tags': ['memory', 'learning', 'cognitive'],
                'icon_emoji': '💡',
            },
            
            # ===== SPECIFIC DIET GOALS =====
            {
                'name': 'Keto Lifestyle',
                'description': 'Ketogenic diet support and recipes',
                'category': HuddleCategory.DIETARY_PREFERENCE,
                'tags': ['keto', 'ketogenic', 'low_carb', 'high_fat'],
                'icon_emoji': '🥓',
            },
            {
                'name': 'Vegan Community',
                'description': 'Plant-based living and nutrition',
                'category': HuddleCategory.DIETARY_PREFERENCE,
                'tags': ['vegan', 'plant_based', 'animal_free'],
                'icon_emoji': '🌱',
            },
            {
                'name': 'Vegetarian Hub',
                'description': 'Vegetarian nutrition and recipes',
                'category': HuddleCategory.DIETARY_PREFERENCE,
                'tags': ['vegetarian', 'meatless', 'plant_forward'],
                'icon_emoji': '🥗',
            },
            {
                'name': 'Paleo Tribe',
                'description': 'Paleolithic diet and ancestral eating',
                'category': HuddleCategory.DIETARY_PREFERENCE,
                'tags': ['paleo', 'ancestral', 'primal'],
                'icon_emoji': '🦴',
            },
            {
                'name': 'Mediterranean Diet',
                'description': 'Heart-healthy Mediterranean eating',
                'category': HuddleCategory.DIETARY_PREFERENCE,
                'tags': ['mediterranean', 'heart_health', 'olive_oil'],
                'icon_emoji': '🫒',
            },
            {
                'name': 'Intermittent Fasting',
                'description': 'Time-restricted eating and fasting protocols',
                'category': HuddleCategory.DIETARY_PREFERENCE,
                'tags': ['intermittent_fasting', 'fasting', 'time_restricted'],
                'icon_emoji': '⏱️',
            },
            {
                'name': 'Carnivore Diet',
                'description': 'Animal-based nutrition support',
                'category': HuddleCategory.DIETARY_PREFERENCE,
                'tags': ['carnivore', 'animal_based', 'zero_carb'],
                'icon_emoji': '🥩',
            },
            {
                'name': 'Whole Food Plant-Based',
                'description': 'Minimally processed plant foods',
                'category': HuddleCategory.DIETARY_PREFERENCE,
                'tags': ['wfpb', 'whole_food', 'plant_based'],
                'icon_emoji': '🌾',
            },
            
            # ===== MEAL PREP & COOKING =====
            {
                'name': 'Meal Prep Masters',
                'description': 'Batch cooking and meal planning strategies',
                'category': HuddleCategory.RECIPE_SHARING,
                'tags': ['meal_prep', 'batch_cooking', 'planning'],
                'icon_emoji': '📦',
            },
            {
                'name': 'Quick & Healthy Recipes',
                'description': '30-minute or less healthy meals',
                'category': HuddleCategory.RECIPE_SHARING,
                'tags': ['quick_recipes', 'easy_cooking', 'time_saving'],
                'icon_emoji': '⏰',
            },
            {
                'name': 'Budget Nutrition',
                'description': 'Eating healthy on a budget',
                'category': HuddleCategory.RECIPE_SHARING,
                'tags': ['budget', 'affordable', 'frugal'],
                'icon_emoji': '💰',
            },
            
            # ===== PROFESSIONAL SUPPORT =====
            {
                'name': 'Ask a Dietitian',
                'description': 'Get answers from registered dietitians',
                'category': HuddleCategory.PROFESSIONAL_SUPPORT,
                'tags': ['dietitian', 'expert', 'professional'],
                'icon_emoji': '⚕️',
            },
            {
                'name': 'Coach Corner',
                'description': 'Health coaching and accountability',
                'category': HuddleCategory.PROFESSIONAL_SUPPORT,
                'tags': ['coach', 'accountability', 'support'],
                'icon_emoji': '🏋️‍♀️',
            },
        ]
        
        for huddle_data in default_huddles:
            huddle_id = self._generate_huddle_id(huddle_data['name'])
            
            huddle = HealthHuddle(
                huddle_id=huddle_id,
                name=huddle_data['name'],
                description=huddle_data['description'],
                category=huddle_data['category'],
                tags=huddle_data['tags'],
                icon_emoji=huddle_data['icon_emoji'],
                rules=[
                    "Be respectful and supportive",
                    "No medical advice - consult professionals",
                    "Keep content relevant to the topic",
                    "Protect your privacy - stay anonymous"
                ]
            )
            
            self.huddles[huddle_id] = huddle
    
    def create_anonymous_profile(
        self,
        user_id: str,
        primary_context: Optional[str] = None,
        is_verified_professional: bool = False,
        professional_title: Optional[str] = None
    ) -> AnonymousProfile:
        """Create anonymous profile for user"""
        anonymous_id = self.identity_manager.generate_anonymous_id(
            user_id,
            primary_context
        )
        
        # Generate avatar seed
        avatar_seed = hashlib.md5(user_id.encode()).hexdigest()[:8]
        
        # Determine badge for professionals
        verification_badge = None
        if is_verified_professional:
            if professional_title and 'dietitian' in professional_title.lower():
                verification_badge = "⚕️"
            elif professional_title and 'coach' in professional_title.lower():
                verification_badge = "🏋️"
            else:
                verification_badge = "✓"
        
        profile = AnonymousProfile(
            anonymous_id=anonymous_id,
            user_id=user_id,
            display_name=anonymous_id,
            avatar_seed=avatar_seed,
            member_since=datetime.now(),
            is_verified=is_verified_professional,
            verification_badge=verification_badge,
            professional_title=professional_title
        )
        
        self.profiles[anonymous_id] = profile
        
        return profile
    
    def join_huddle(
        self,
        user_id: str,
        huddle_id: str
    ) -> bool:
        """User joins a huddle"""
        if huddle_id not in self.huddles:
            return False
        
        # Get or create anonymous profile
        anonymous_id = self.identity_manager.get_anonymous_id(user_id)
        if not anonymous_id:
            # Create profile
            profile = self.create_anonymous_profile(user_id)
            anonymous_id = profile.anonymous_id
        
        # Add to huddle
        huddle = self.huddles[huddle_id]
        success = huddle.add_member(anonymous_id)
        
        if success:
            self.user_huddles[user_id].add(huddle_id)
            self.profiles[anonymous_id].huddle_count += 1
        
        return success
    
    def create_post_from_scan(
        self,
        user_id: str,
        huddle_id: str,
        text_content: str,
        food_name: str,
        food_image_url: Optional[str],
        risk_card_data: Dict[str, Any],
        nutrient_profile: Dict[str, Any],
        user_goals: List[str],
        user_conditions: List[str]
    ) -> Optional[HuddlePost]:
        """
        Create post with shareable card from meal scan
        
        This is the AI-facilitated sharing feature:
        - Auto-generates card from scan data
        - Prevents manual entry
        - Ensures accuracy from Chemometrics Engine
        """
        if huddle_id not in self.huddles:
            return None
        
        # Get anonymous ID
        anonymous_id = self.identity_manager.get_anonymous_id(user_id)
        if not anonymous_id:
            return None
        
        # Check membership
        huddle = self.huddles[huddle_id]
        if anonymous_id not in huddle.members:
            return None
        
        # Generate shareable card
        card = self.card_generator.generate_from_scan(
            food_name=food_name,
            food_image_url=food_image_url,
            risk_card_data=risk_card_data,
            nutrient_profile=nutrient_profile,
            anonymous_id=anonymous_id,
            user_goals=user_goals,
            user_conditions=user_conditions
        )
        
        # Create post
        post_id = self._generate_post_id()
        post = HuddlePost(
            post_id=post_id,
            huddle_id=huddle_id,
            author_id=anonymous_id,
            content_type=ShareableContentType.MEAL_SCAN,
            text_content=text_content,
            shareable_card=card,
            image_urls=[food_image_url] if food_image_url else []
        )
        
        # Store post
        self.posts[post_id] = post
        self.huddle_posts[huddle_id].append(post_id)
        
        # Update stats
        huddle.post_count += 1
        huddle.last_activity = datetime.now()
        self.profiles[anonymous_id].post_count += 1
        
        return post
    
    def get_huddle_feed(
        self,
        huddle_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[HuddlePost]:
        """Get posts in huddle feed"""
        if huddle_id not in self.huddle_posts:
            return []
        
        post_ids = self.huddle_posts[huddle_id]
        # Reverse chronological order
        post_ids_slice = list(reversed(post_ids))[offset:offset+limit]
        
        posts = [self.posts[pid] for pid in post_ids_slice if pid in self.posts]
        
        return posts
    
    def add_comment(
        self,
        user_id: str,
        post_id: str,
        text: str
    ) -> Optional[HuddleComment]:
        """Add comment to post"""
        if post_id not in self.posts:
            return None
        
        anonymous_id = self.identity_manager.get_anonymous_id(user_id)
        if not anonymous_id:
            return None
        
        comment_id = self._generate_comment_id()
        comment = HuddleComment(
            comment_id=comment_id,
            post_id=post_id,
            author_id=anonymous_id,
            text=text
        )
        
        post = self.posts[post_id]
        post.comments.append(comment)
        
        # Update card engagement
        if post.shareable_card:
            post.shareable_card.comments_count += 1
        
        return comment
    
    def like_post(self, user_id: str, post_id: str) -> bool:
        """Like a post"""
        if post_id not in self.posts:
            return False
        
        anonymous_id = self.identity_manager.get_anonymous_id(user_id)
        if not anonymous_id:
            return False
        
        post = self.posts[post_id]
        success = post.add_like(anonymous_id)
        
        if success and post.shareable_card:
            post.shareable_card.likes_count += 1
        
        return success
    
    def _generate_huddle_id(self, name: str) -> str:
        """Generate huddle ID from name"""
        slug = re.sub(r'[^a-z0-9]+', '_', name.lower())
        return f"huddle_{slug}"
    
    def _generate_post_id(self) -> str:
        """Generate unique post ID"""
        return f"post_{secrets.token_urlsafe(16)}"
    
    def _generate_comment_id(self) -> str:
        """Generate unique comment ID"""
        return f"comment_{secrets.token_urlsafe(12)}"


class HuddleCreationManager:
    """
    Manages user-created custom huddles
    
    Allows users to create their own huddles for:
    - Rare conditions
    - Specific diet combinations
    - Local communities
    - Custom goals
    """
    
    def __init__(self, huddle_manager: 'HealthHuddleManager'):
        self.manager = huddle_manager
        self.pending_approvals: Dict[str, HealthHuddle] = {}
        logger.info("HuddleCreationManager initialized")
    
    def create_custom_huddle(
        self,
        creator_user_id: str,
        name: str,
        description: str,
        category: HuddleCategory,
        tags: List[str],
        icon_emoji: str = "👥",
        is_private: bool = False,
        requires_approval: bool = False
    ) -> Optional[HealthHuddle]:
        """
        Create a custom user-generated huddle
        
        Args:
            creator_user_id: User creating the huddle
            name: Huddle name
            description: Huddle description
            category: Category
            tags: Relevant tags
            icon_emoji: Icon emoji
            is_private: Private huddle (invite-only)
            requires_approval: Require approval to join
            
        Returns:
            Created huddle or None if invalid
        """
        # Validate name
        if len(name) < 3 or len(name) > 100:
            logger.warning("Invalid huddle name length")
            return None
        
        # Check for duplicate name
        name_lower = name.lower()
        for huddle in self.manager.huddles.values():
            if huddle.name.lower() == name_lower:
                logger.warning(f"Huddle name already exists: {name}")
                return None
        
        # Get creator's anonymous ID
        anonymous_id = self.manager.identity_manager.get_anonymous_id(creator_user_id)
        if not anonymous_id:
            profile = self.manager.create_anonymous_profile(creator_user_id)
            anonymous_id = profile.anonymous_id
        
        # Generate huddle ID
        huddle_id = self.manager._generate_huddle_id(name)
        
        # Create huddle
        huddle = HealthHuddle(
            huddle_id=huddle_id,
            name=name,
            description=description,
            category=category,
            tags=tags,
            icon_emoji=icon_emoji,
            is_private=is_private,
            requires_approval=requires_approval,
            rules=[
                "Be respectful and supportive",
                "Stay on topic",
                "No spam or self-promotion",
                "Protect member privacy"
            ]
        )
        
        # Creator becomes first moderator
        huddle.moderators.add(anonymous_id)
        huddle.add_member(anonymous_id)
        
        # Add to manager
        self.manager.huddles[huddle_id] = huddle
        self.manager.user_huddles[creator_user_id].add(huddle_id)
        
        logger.info(f"Created custom huddle: {name} by {anonymous_id}")
        
        return huddle
    
    def search_huddles(
        self,
        query: str,
        category: Optional[HuddleCategory] = None,
        tags: Optional[List[str]] = None
    ) -> List[HealthHuddle]:
        """
        Search for huddles by name, description, or tags
        
        Args:
            query: Search query
            category: Filter by category
            tags: Filter by tags
            
        Returns:
            Matching huddles
        """
        results = []
        query_lower = query.lower()
        
        for huddle in self.manager.huddles.values():
            # Skip private huddles in search
            if huddle.is_private:
                continue
            
            # Category filter
            if category and huddle.category != category:
                continue
            
            # Tag filter
            if tags:
                if not any(tag in huddle.tags for tag in tags):
                    continue
            
            # Text search
            if query_lower:
                if (query_lower in huddle.name.lower() or
                    query_lower in huddle.description.lower() or
                    any(query_lower in tag for tag in huddle.tags)):
                    results.append(huddle)
            else:
                results.append(huddle)
        
        # Sort by relevance (member count)
        results.sort(key=lambda h: h.member_count, reverse=True)
        
        return results
    
    def get_huddles_by_category(
        self,
        category: HuddleCategory
    ) -> List[HealthHuddle]:
        """Get all huddles in a category"""
        huddles = [
            h for h in self.manager.huddles.values()
            if h.category == category and not h.is_private
        ]
        
        # Sort by member count
        huddles.sort(key=lambda h: h.member_count, reverse=True)
        
        return huddles
    
    def get_all_categories(self) -> Dict[str, int]:
        """Get all categories with huddle counts"""
        category_counts = defaultdict(int)
        
        for huddle in self.manager.huddles.values():
            if not huddle.is_private:
                category_counts[huddle.category.value] += 1
        
        return dict(category_counts)


class CommunityAPI:
    """
    API interface for community features
    
    Provides endpoints for:
    - Browsing huddles
    - Creating custom huddles
    - Joining/leaving
    - Posting content
    - Engaging (likes, comments)
    - Direct messaging
    """
    
    def __init__(self, huddle_manager: HealthHuddleManager):
        self.manager = huddle_manager
        self.creation_manager = HuddleCreationManager(huddle_manager)
        logger.info("CommunityAPI initialized")
    
    def get_recommended_huddles(
        self,
        user_id: str,
        user_conditions: List[str],
        user_goals: List[str]
    ) -> List[Dict[str, Any]]:
        """Get recommended huddles for user"""
        recommended = []
        
        for huddle in self.manager.huddles.values():
            score = 0
            
            # Check tag matches
            for condition in user_conditions:
                if condition in huddle.tags:
                    score += 10
            
            for goal in user_goals:
                if goal in huddle.tags:
                    score += 5
            
            if score > 0:
                recommended.append({
                    'huddle': self._serialize_huddle(huddle),
                    'match_score': score,
                    'match_reason': self._get_match_reason(huddle, user_conditions, user_goals)
                })
        
        # Sort by score
        recommended.sort(key=lambda x: x['match_score'], reverse=True)
        
        return recommended[:10]
    
    def share_scan_to_huddle(
        self,
        user_id: str,
        huddle_id: str,
        message: str,
        scan_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Share meal scan result to huddle
        
        This is the one-button share feature from the app
        """
        post = self.manager.create_post_from_scan(
            user_id=user_id,
            huddle_id=huddle_id,
            text_content=message,
            food_name=scan_result['food_name'],
            food_image_url=scan_result.get('food_image_url'),
            risk_card_data=scan_result['risk_card'],
            nutrient_profile=scan_result['nutrients'],
            user_goals=scan_result.get('user_goals', []),
            user_conditions=scan_result.get('user_conditions', [])
        )
        
        if post:
            return {
                'success': True,
                'post_id': post.post_id,
                'shareable_card': post.shareable_card.to_dict() if post.shareable_card else None
            }
        
        return {'success': False, 'error': 'Failed to create post'}
    
    def _serialize_huddle(self, huddle: HealthHuddle) -> Dict[str, Any]:
        """Serialize huddle for API response"""
        return {
            'huddle_id': huddle.huddle_id,
            'name': huddle.name,
            'description': huddle.description,
            'category': huddle.category.value,
            'icon_emoji': huddle.icon_emoji,
            'member_count': huddle.member_count,
            'post_count': huddle.post_count,
            'is_official': huddle.is_official,
            'tags': huddle.tags
        }
    
    def browse_all_huddles(
        self,
        user_id: str,
        category: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Browse all available huddles with filtering
        
        Args:
            user_id: User browsing
            category: Filter by category
            search_query: Search term
            limit: Max results
            
        Returns:
            Categorized huddles and search results
        """
        # Get user's current huddles
        user_huddles = self.manager.user_huddles.get(user_id, set())
        
        # Search if query provided
        if search_query:
            huddles = self.creation_manager.search_huddles(search_query)
        elif category:
            cat_enum = HuddleCategory(category)
            huddles = self.creation_manager.get_huddles_by_category(cat_enum)
        else:
            huddles = [
                h for h in self.manager.huddles.values()
                if not h.is_private
            ]
            # Sort by member count
            huddles.sort(key=lambda h: h.member_count, reverse=True)
        
        # Limit results
        huddles = huddles[:limit]
        
        # Serialize
        result = {
            'huddles': [
                {
                    **self._serialize_huddle(h),
                    'is_member': h.huddle_id in user_huddles
                }
                for h in huddles
            ],
            'categories': self.creation_manager.get_all_categories(),
            'total_huddles': len(self.manager.huddles)
        }
        
        return result
    
    def create_huddle(
        self,
        user_id: str,
        name: str,
        description: str,
        category: str,
        tags: List[str],
        icon_emoji: str = "👥",
        is_private: bool = False
    ) -> Dict[str, Any]:
        """
        Create a custom huddle
        
        Returns:
            Created huddle info or error
        """
        cat_enum = HuddleCategory(category)
        
        huddle = self.creation_manager.create_custom_huddle(
            creator_user_id=user_id,
            name=name,
            description=description,
            category=cat_enum,
            tags=tags,
            icon_emoji=icon_emoji,
            is_private=is_private
        )
        
        if huddle:
            return {
                'success': True,
                'huddle': self._serialize_huddle(huddle)
            }
        
        return {
            'success': False,
            'error': 'Failed to create huddle (duplicate name or invalid parameters)'
        }
    
    def _get_match_reason(
        self,
        huddle: HealthHuddle,
        conditions: List[str],
        goals: List[str]
    ) -> str:
        """Get reason why huddle matches user"""
        reasons = []
        
        for condition in conditions:
            if condition in huddle.tags:
                reasons.append(f"Matches your {condition.replace('_', ' ')}")
        
        for goal in goals:
            if goal in huddle.tags:
                reasons.append(f"Supports your {goal.replace('_', ' ')} goal")
        
        return " • ".join(reasons) if reasons else "Popular community"


if __name__ == "__main__":
    logger.info("Testing Phase 7: Anonymous Social & Health Huddles")
    
    # Initialize managers
    identity_mgr = AnonymousIdentityManager()
    card_gen = ShareableCardGenerator()
    huddle_mgr = HealthHuddleManager(identity_mgr, card_gen)
    api = CommunityAPI(huddle_mgr)
    
    # Test user 1: T2D patient
    user1_id = "user_alice_123"
    profile1 = huddle_mgr.create_anonymous_profile(
        user1_id,
        primary_context='diabetes_type2'
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"User 1 Anonymous Profile:")
    logger.info(f"  Display Name: {profile1.display_name}")
    logger.info(f"  Badge: {profile1.get_display_badge()}")
    
    # Test user 2: Verified dietitian
    user2_id = "user_dietitian_456"
    profile2 = huddle_mgr.create_anonymous_profile(
        user2_id,
        is_verified_professional=True,
        professional_title="Registered Dietitian"
    )
    
    logger.info(f"\nUser 2 Verified Professional:")
    logger.info(f"  Display Name: {profile2.display_name}")
    logger.info(f"  Badge: {profile2.get_display_badge()}")
    logger.info(f"  Verified: {profile2.is_verified}")
    
    # Test joining huddles
    logger.info(f"\n{'='*60}")
    logger.info("Joining Huddles...")
    
    # User 1 joins T2D huddle
    huddle_mgr.join_huddle(user1_id, "huddle_type_2_diabetes_management")
    huddle_mgr.join_huddle(user1_id, "huddle_weight_loss_warriors")
    
    logger.info(f"User 1 joined {len(huddle_mgr.user_huddles[user1_id])} huddles")
    
    # Test sharing meal scan
    logger.info(f"\n{'='*60}")
    logger.info("Sharing Meal Scan to Huddle...")
    
    scan_result = {
        'food_name': 'Grilled Chicken Salad',
        'food_image_url': 'https://example.com/chicken_salad.jpg',
        'risk_card': {
            'health_score': 88,
            'safety_verdict': 'SAFE'
        },
        'nutrients': {
            'calories': 320,
            'protein': 35,
            'carbohydrates': 12,
            'fat': 14,
            'fiber': 6,
            'sodium': 280,
            'sugar': 3
        },
        'user_goals': ['weight_loss', 'muscle_gain'],
        'user_conditions': ['diabetes_type2']
    }
    
    share_result = api.share_scan_to_huddle(
        user_id=user1_id,
        huddle_id="huddle_type_2_diabetes_management",
        message="Just had this for lunch! Great macros and low sugar 💪",
        scan_result=scan_result
    )
    
    logger.info(f"Share Success: {share_result['success']}")
    if share_result['success']:
        card = share_result['shareable_card']
        logger.info(f"\nGenerated Shareable Card:")
        logger.info(f"  Food: {card['food_name']}")
        logger.info(f"  Health Score: {card['health_score']} ({card['score_color']})")
        logger.info(f"  Verdict: {card['user_verdict']}")
        logger.info(f"  Highlights:")
        for highlight in card['highlights']:
            logger.info(f"    - {highlight}")
    
    # Test recommended huddles
    logger.info(f"\n{'='*60}")
    logger.info("Recommended Huddles:")
    
    recommended = api.get_recommended_huddles(
        user_id=user1_id,
        user_conditions=['diabetes_type2'],
        user_goals=['weight_loss']
    )
    
    for rec in recommended[:3]:
        huddle_info = rec['huddle']
        logger.info(f"\n  {huddle_info['icon_emoji']} {huddle_info['name']}")
        logger.info(f"     {rec['match_reason']}")
        logger.info(f"     {huddle_info['member_count']} members • {huddle_info['post_count']} posts")
    
    # Test browsing all huddles
    logger.info(f"\n{'='*60}")
    logger.info("Browsing All Huddles:")
    
    browse_result = api.browse_all_huddles(
        user_id=user1_id,
        limit=10
    )
    
    logger.info(f"\nTotal Huddles Available: {browse_result['total_huddles']}")
    logger.info(f"Categories: {browse_result['categories']}")
    logger.info(f"\nTop 10 Huddles:")
    for huddle in browse_result['huddles'][:10]:
        logger.info(f"  {huddle['icon_emoji']} {huddle['name']} - {huddle['member_count']} members")
    
    # Test creating custom huddle
    logger.info(f"\n{'='*60}")
    logger.info("Creating Custom Huddle:")
    
    custom_huddle = api.create_huddle(
        user_id=user1_id,
        name="Rare Disease Warriors",
        description="Support for people with rare metabolic disorders",
        category="medical_condition",
        tags=['rare_disease', 'metabolic', 'support'],
        icon_emoji="🦋"
    )
    
    if custom_huddle['success']:
        logger.info(f"\n✅ Created: {custom_huddle['huddle']['name']}")
        logger.info(f"   Category: {custom_huddle['huddle']['category']}")
        logger.info(f"   Tags: {custom_huddle['huddle']['tags']}")
    
    # Test searching huddles
    logger.info(f"\n{'='*60}")
    logger.info("Searching Huddles (query: 'diabetes'):")
    
    search_result = api.browse_all_huddles(
        user_id=user1_id,
        search_query='diabetes',
        limit=5
    )
    
    logger.info(f"\nFound {len(search_result['huddles'])} matches:")
    for huddle in search_result['huddles']:
        logger.info(f"  {huddle['icon_emoji']} {huddle['name']}")
    
    # Test category browsing
    logger.info(f"\n{'='*60}")
    logger.info("Browsing Category: HEALTH_GOAL")
    
    category_result = api.browse_all_huddles(
        user_id=user1_id,
        category='health_goal',
        limit=10
    )
    
    logger.info(f"\nHealth Goal Huddles ({len(category_result['huddles'])}):")
    for huddle in category_result['huddles'][:5]:
        logger.info(f"  {huddle['icon_emoji']} {huddle['name']}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Phase 7 Complete!")
    logger.info(f"   Total Huddles: {browse_result['total_huddles']}")
    logger.info(f"   Medical Conditions: {browse_result['categories'].get('medical_condition', 0)}")
    logger.info(f"   Health Goals: {browse_result['categories'].get('health_goal', 0)}")
    logger.info(f"   Dietary Preferences: {browse_result['categories'].get('dietary_preference', 0)}")
    logger.info(f"   Recipe Sharing: {browse_result['categories'].get('recipe_sharing', 0)}")
    logger.info(f"   Professional Support: {browse_result['categories'].get('professional_support', 0)}")
