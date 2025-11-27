"""
Feature 8: Public Survivor Stories (Social Feed)
================================================

Anonymous publication system that allows users to share their health transformation
stories in the WATCH tab social feed, creating inspiration and community support.

Key Features:
- Story anonymization and privacy controls
- AI-powered content moderation
- Engagement metrics (views, likes, shares, comments)
- Discovery algorithms (trending, personalized, similar journeys)
- Viral growth mechanics (featured stories, hashtags, challenges)
- Story templates and formatting
- Multi-format support (text, images, videos)

Privacy & Safety:
- Opt-in sharing with granular controls
- PII removal and anonymization
- Community guidelines enforcement
- Report/flag system
- Age-appropriate content filtering

Author: AI Health Features Team
Created: November 12, 2025
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import random
import hashlib
import re
from collections import defaultdict


# ==================== ENUMS AND TYPES ====================

class StoryVisibility(Enum):
    """Story visibility levels"""
    PRIVATE = "private"  # Only visible to user
    CONNECTIONS = "connections"  # Visible to accountability squad
    PUBLIC = "public"  # Visible in WATCH feed
    FEATURED = "featured"  # Highlighted by editorial team


class StoryFormat(Enum):
    """Story content formats"""
    TEXT_ONLY = "text_only"
    TEXT_WITH_IMAGE = "text_with_image"
    BEFORE_AFTER = "before_after"
    VIDEO_STORY = "video_story"
    MILESTONE_CARD = "milestone_card"
    QUOTE_CARD = "quote_card"


class ModerationStatus(Enum):
    """Content moderation states"""
    PENDING = "pending"  # Awaiting review
    APPROVED = "approved"  # Safe to display
    FLAGGED = "flagged"  # Needs human review
    REJECTED = "rejected"  # Violates guidelines
    REMOVED = "removed"  # Removed after publication


class EngagementType(Enum):
    """Types of user engagement"""
    VIEW = "view"
    LIKE = "like"
    SHARE = "share"
    COMMENT = "comment"
    SAVE = "save"
    FOLLOW_AUTHOR = "follow_author"


class FeedAlgorithmType(Enum):
    """Feed sorting algorithms"""
    TRENDING = "trending"  # Viral stories
    PERSONALIZED = "personalized"  # Tailored to user
    RECENT = "recent"  # Chronological
    SIMILAR_JOURNEY = "similar_journey"  # Similar health conditions
    INSPIRING = "inspiring"  # High engagement + positive
    EDUCATIONAL = "educational"  # Informative content


class ReportReason(Enum):
    """Content report reasons"""
    SPAM = "spam"
    HARASSMENT = "harassment"
    MISINFORMATION = "misinformation"
    INAPPROPRIATE = "inappropriate"
    PRIVACY_VIOLATION = "privacy_violation"
    OFFENSIVE = "offensive"


# ==================== DATA MODELS ====================

@dataclass
class AnonymousAuthor:
    """Anonymized author profile"""
    anonymous_id: str  # Hashed identifier
    display_name: str  # "Health Warrior 2547"
    avatar_style: str  # Abstract avatar
    
    # Shareable demographics (aggregated)
    age_range: Optional[str] = None  # "30-35", "40-45"
    region: Optional[str] = None  # "Northeast US", "Western Europe"
    journey_duration: Optional[str] = None  # "3 months", "1 year"
    condition_type: Optional[str] = None  # "Type 2 Diabetes", "Weight Management"
    
    # Badges and credibility
    verified_transformation: bool = False  # Verified by data
    total_stories: int = 0
    total_engagement: int = 0
    
    # Privacy settings
    show_age_range: bool = True
    show_region: bool = False
    show_condition: bool = True


@dataclass
class StoryMetrics:
    """Anonymized health transformation metrics"""
    # Weight changes (optional)
    weight_change_kg: Optional[float] = None
    weight_change_percent: Optional[float] = None
    
    # Biomarker improvements (optional)
    hba1c_change: Optional[float] = None
    glucose_improvement: Optional[str] = None  # "30% better TIR"
    
    # Behavioral changes
    scan_streak_days: Optional[int] = None
    compliance_rate: Optional[float] = None
    
    # Medication changes
    medications_reduced: Optional[int] = None
    
    # Timeline
    journey_duration_days: int = 0
    
    # Verification
    data_verified: bool = False  # Backed by actual data


@dataclass
class StoryContent:
    """Story text and media content"""
    title: str
    body: str  # Main story text
    
    # Optional media
    images: List[str] = field(default_factory=list)  # Image URLs/paths
    video_url: Optional[str] = None
    
    # Formatted sections
    struggle_section: Optional[str] = None
    breakthrough_section: Optional[str] = None
    advice_section: Optional[str] = None
    
    # Hashtags and categories
    hashtags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)  # "Weight Loss", "Diabetes Reversal"
    
    # Quotes for sharing
    shareable_quote: Optional[str] = None


@dataclass
class EngagementMetrics:
    """Story engagement data"""
    views: int = 0
    likes: int = 0
    shares: int = 0
    comments: int = 0
    saves: int = 0
    
    # Engagement rate
    engagement_rate: float = 0.0  # (likes + shares + comments) / views
    
    # Virality metrics
    viral_coefficient: float = 0.0  # shares per view
    trending_score: float = 0.0  # Weighted recency + engagement
    
    # Time-based
    views_last_24h: int = 0
    views_last_7d: int = 0
    
    # User actions
    liked_by_user: bool = False
    saved_by_user: bool = False


@dataclass
class ModerationResult:
    """Content moderation analysis"""
    status: ModerationStatus
    confidence: float  # 0-1
    
    # Safety checks
    contains_pii: bool = False  # Personally identifiable info
    is_spam: bool = False
    is_inappropriate: bool = False
    is_medical_advice: bool = False  # Flagged if giving prescriptions
    
    # Detected issues
    flagged_terms: List[str] = field(default_factory=list)
    detected_patterns: List[str] = field(default_factory=list)
    
    # Review notes
    review_notes: Optional[str] = None
    reviewed_by: Optional[str] = None  # "ai_moderator" or human ID
    reviewed_at: Optional[datetime] = None


@dataclass
class SurvivorStory:
    """Complete survivor story"""
    story_id: str
    author: AnonymousAuthor
    content: StoryContent
    metrics: StoryMetrics
    engagement: EngagementMetrics
    moderation: ModerationResult
    
    # Metadata
    visibility: StoryVisibility
    format: StoryFormat
    created_at: datetime
    published_at: Optional[datetime] = None
    
    # Discoverability
    is_featured: bool = False
    featured_at: Optional[datetime] = None
    trending_rank: Optional[int] = None
    
    # User relationship
    is_own_story: bool = False
    author_followed: bool = False


# ==================== ANONYMIZATION ENGINE ====================

class StoryAnonymizer:
    """Removes PII and creates anonymous author profiles"""
    
    def __init__(self):
        self.pii_patterns = self._compile_pii_patterns()
        self.name_database = self._load_replacement_names()
    
    def anonymize_author(self, user_id: str, preferences: Dict[str, Any]) -> AnonymousAuthor:
        """Create anonymous author profile"""
        # Generate consistent anonymous ID (hashed)
        anonymous_id = self._generate_anonymous_id(user_id)
        
        # Generate friendly display name
        display_name = self._generate_display_name(anonymous_id)
        
        # Aggregate demographics
        age_range = self._aggregate_age(preferences.get('age'))
        region = self._aggregate_region(preferences.get('location'))
        
        return AnonymousAuthor(
            anonymous_id=anonymous_id,
            display_name=display_name,
            avatar_style=f"avatar_{int(anonymous_id[:8], 16) % 100}",
            age_range=age_range if preferences.get('show_age', True) else None,
            region=region if preferences.get('show_region', False) else None,
            journey_duration=preferences.get('journey_duration'),
            condition_type=preferences.get('condition_type') if preferences.get('show_condition', True) else None,
            verified_transformation=preferences.get('has_verified_data', False),
            show_age_range=preferences.get('show_age', True),
            show_region=preferences.get('show_region', False),
            show_condition=preferences.get('show_condition', True)
        )
    
    def anonymize_content(self, text: str) -> Tuple[str, List[str]]:
        """Remove PII from story content"""
        anonymized = text
        removed_items = []
        
        # Remove names (simple approach - production would use NER)
        for pattern, replacement in self.pii_patterns.items():
            matches = re.findall(pattern, anonymized, re.IGNORECASE)
            if matches:
                removed_items.extend(matches)
                anonymized = re.sub(pattern, replacement, anonymized, flags=re.IGNORECASE)
        
        return anonymized, removed_items
    
    def _generate_anonymous_id(self, user_id: str) -> str:
        """Generate consistent anonymous ID"""
        # Add salt for security
        salted = f"survivor_story_{user_id}_wellomex_2025"
        return hashlib.sha256(salted.encode()).hexdigest()[:16]
    
    def _generate_display_name(self, anonymous_id: str) -> str:
        """Generate friendly anonymous name"""
        adjectives = ["Brave", "Strong", "Resilient", "Determined", "Inspiring", 
                     "Mighty", "Powerful", "Victorious", "Thriving", "Unstoppable"]
        nouns = ["Warrior", "Champion", "Hero", "Survivor", "Fighter", 
                "Winner", "Overcomer", "Achiever", "Transformer", "Conqueror"]
        
        # Use hash to select consistent name
        adj_idx = int(anonymous_id[:4], 16) % len(adjectives)
        noun_idx = int(anonymous_id[4:8], 16) % len(nouns)
        number = int(anonymous_id[8:12], 16) % 10000
        
        return f"{adjectives[adj_idx]} {nouns[noun_idx]} {number}"
    
    def _aggregate_age(self, age: Optional[int]) -> Optional[str]:
        """Convert age to range"""
        if not age:
            return None
        
        if age < 25:
            return "18-24"
        elif age < 30:
            return "25-29"
        elif age < 35:
            return "30-34"
        elif age < 40:
            return "35-39"
        elif age < 45:
            return "40-44"
        elif age < 50:
            return "45-49"
        elif age < 55:
            return "50-54"
        elif age < 60:
            return "55-59"
        else:
            return "60+"
    
    def _aggregate_region(self, location: Optional[str]) -> Optional[str]:
        """Convert location to broad region"""
        if not location:
            return None
        
        # Simplified - production would use geocoding
        location_lower = location.lower()
        
        if any(state in location_lower for state in ['ca', 'california', 'wa', 'washington', 'or', 'oregon']):
            return "Western US"
        elif any(state in location_lower for state in ['ny', 'new york', 'ma', 'massachusetts', 'pa', 'pennsylvania']):
            return "Northeastern US"
        elif any(state in location_lower for state in ['tx', 'texas', 'fl', 'florida', 'ga', 'georgia']):
            return "Southern US"
        else:
            return "United States"
    
    def _compile_pii_patterns(self) -> Dict[str, str]:
        """Compile regex patterns for PII detection"""
        return {
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b': '[Name]',  # Full names
            r'\b\d{3}-\d{2}-\d{4}\b': '[SSN]',  # SSN
            r'\b\d{3}-\d{3}-\d{4}\b': '[Phone]',  # Phone numbers
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[Email]',  # Emails
            r'\b\d{1,5}\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr)\b': '[Address]',  # Addresses
        }
    
    def _load_replacement_names(self) -> List[str]:
        """Load generic replacement names"""
        return ["Sarah", "John", "Maria", "David", "Lisa", "Michael", "Emily", "James"]


# ==================== CONTENT MODERATOR ====================

class ContentModerator:
    """AI-powered content moderation"""
    
    def __init__(self):
        self.spam_keywords = self._load_spam_keywords()
        self.inappropriate_patterns = self._load_inappropriate_patterns()
        self.medical_advice_indicators = self._load_medical_indicators()
    
    def moderate_story(self, content: StoryContent) -> ModerationResult:
        """Analyze story content for safety"""
        # Combine all text
        full_text = f"{content.title} {content.body}"
        if content.struggle_section:
            full_text += f" {content.struggle_section}"
        if content.breakthrough_section:
            full_text += f" {content.breakthrough_section}"
        if content.advice_section:
            full_text += f" {content.advice_section}"
        
        # Run checks
        contains_pii = self._check_pii(full_text)
        is_spam = self._check_spam(full_text)
        is_inappropriate = self._check_inappropriate(full_text)
        is_medical_advice = self._check_medical_advice(full_text)
        
        # Detect specific issues
        flagged_terms = []
        detected_patterns = []
        
        if is_spam:
            flagged_terms.extend(self._find_spam_terms(full_text))
            detected_patterns.append("spam_content")
        
        if is_medical_advice:
            detected_patterns.append("medical_advice")
            flagged_terms.extend(self._find_medical_terms(full_text))
        
        # Determine status
        if contains_pii or is_spam or is_inappropriate:
            status = ModerationStatus.FLAGGED
            confidence = 0.85
        elif is_medical_advice:
            status = ModerationStatus.FLAGGED
            confidence = 0.70
        else:
            status = ModerationStatus.APPROVED
            confidence = 0.95
        
        return ModerationResult(
            status=status,
            confidence=confidence,
            contains_pii=contains_pii,
            is_spam=is_spam,
            is_inappropriate=is_inappropriate,
            is_medical_advice=is_medical_advice,
            flagged_terms=flagged_terms,
            detected_patterns=detected_patterns,
            reviewed_by="ai_moderator",
            reviewed_at=datetime.now()
        )
    
    def _check_pii(self, text: str) -> bool:
        """Check for personally identifiable information"""
        # Check for email patterns
        if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text):
            return True
        
        # Check for phone numbers
        if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
            return True
        
        # Check for SSN
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
            return True
        
        return False
    
    def _check_spam(self, text: str) -> bool:
        """Check for spam content"""
        text_lower = text.lower()
        spam_count = sum(1 for keyword in self.spam_keywords if keyword in text_lower)
        return spam_count >= 3  # 3+ spam indicators = spam
    
    def _check_inappropriate(self, text: str) -> bool:
        """Check for inappropriate content"""
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self.inappropriate_patterns)
    
    def _check_medical_advice(self, text: str) -> bool:
        """Check for medical advice (requires disclaimer)"""
        text_lower = text.lower()
        
        # Look for prescriptive language
        prescriptive_phrases = [
            "you should take", "i recommend taking", "stop taking",
            "start taking", "dosage of", "prescription for"
        ]
        
        has_prescriptive = any(phrase in text_lower for phrase in prescriptive_phrases)
        has_medical_terms = sum(1 for term in self.medical_advice_indicators if term in text_lower) >= 2
        
        return has_prescriptive and has_medical_terms
    
    def _find_spam_terms(self, text: str) -> List[str]:
        """Find specific spam terms in text"""
        text_lower = text.lower()
        return [keyword for keyword in self.spam_keywords if keyword in text_lower]
    
    def _find_medical_terms(self, text: str) -> List[str]:
        """Find medical advice indicators"""
        text_lower = text.lower()
        return [term for term in self.medical_advice_indicators if term in text_lower]
    
    def _load_spam_keywords(self) -> List[str]:
        """Load spam detection keywords"""
        return [
            "buy now", "click here", "limited time", "act now",
            "make money", "work from home", "lose weight fast",
            "miracle cure", "guaranteed results", "free trial",
            "no obligation", "risk free", "as seen on tv"
        ]
    
    def _load_inappropriate_patterns(self) -> List[str]:
        """Load inappropriate content patterns"""
        return [
            "profanity1", "profanity2", "hate_speech",  # Placeholder
            "violence", "graphic_content"
        ]
    
    def _load_medical_indicators(self) -> List[str]:
        """Load medical advice indicators"""
        return [
            "insulin", "metformin", "medication", "prescription",
            "dosage", "mg", "doctor", "physician", "diagnose"
        ]


# ==================== FEED ALGORITHM ====================

class FeedAlgorithm:
    """Personalized story discovery and ranking"""
    
    def __init__(self):
        self.trending_window_hours = 24
        self.recency_weight = 0.3
        self.engagement_weight = 0.5
        self.personalization_weight = 0.2
    
    def rank_stories(
        self,
        stories: List[SurvivorStory],
        user_profile: Dict[str, Any],
        algorithm: FeedAlgorithmType = FeedAlgorithmType.PERSONALIZED
    ) -> List[SurvivorStory]:
        """Rank stories based on algorithm"""
        if algorithm == FeedAlgorithmType.TRENDING:
            return self._rank_by_trending(stories)
        elif algorithm == FeedAlgorithmType.PERSONALIZED:
            return self._rank_by_personalization(stories, user_profile)
        elif algorithm == FeedAlgorithmType.RECENT:
            return self._rank_by_recency(stories)
        elif algorithm == FeedAlgorithmType.SIMILAR_JOURNEY:
            return self._rank_by_similarity(stories, user_profile)
        elif algorithm == FeedAlgorithmType.INSPIRING:
            return self._rank_by_inspiration(stories)
        else:
            return stories
    
    def _rank_by_trending(self, stories: List[SurvivorStory]) -> List[SurvivorStory]:
        """Rank by trending score (viral + recent)"""
        for story in stories:
            # Calculate trending score
            hours_old = (datetime.now() - story.created_at).total_seconds() / 3600
            recency_factor = max(0, 1 - (hours_old / self.trending_window_hours))
            
            engagement_rate = story.engagement.engagement_rate
            viral_coefficient = story.engagement.viral_coefficient
            
            story.engagement.trending_score = (
                recency_factor * self.recency_weight +
                engagement_rate * self.engagement_weight +
                viral_coefficient * (1 - self.recency_weight - self.engagement_weight)
            )
        
        return sorted(stories, key=lambda s: s.engagement.trending_score, reverse=True)
    
    def _rank_by_personalization(
        self,
        stories: List[SurvivorStory],
        user_profile: Dict[str, Any]
    ) -> List[SurvivorStory]:
        """Rank by personal relevance"""
        scored_stories = []
        
        for story in stories:
            score = 0.0
            
            # Similar condition (+30 points)
            if story.author.condition_type == user_profile.get('condition'):
                score += 30
            
            # Similar age range (+15 points)
            if story.author.age_range == self._get_age_range(user_profile.get('age')):
                score += 15
            
            # Similar journey duration (+10 points)
            user_days = user_profile.get('journey_days', 0)
            story_days = story.metrics.journey_duration_days
            if abs(user_days - story_days) < 90:  # Within 3 months
                score += 10
            
            # Similar goals (+20 points)
            user_goals = set(user_profile.get('goals', []))
            story_categories = set(story.content.categories)
            overlap = len(user_goals & story_categories)
            score += overlap * 5
            
            # Engagement quality (+25 points max)
            score += story.engagement.engagement_rate * 25
            
            # Verified transformation (+10 points)
            if story.metrics.data_verified:
                score += 10
            
            scored_stories.append((story, score))
        
        # Sort by score
        scored_stories.sort(key=lambda x: x[1], reverse=True)
        return [story for story, score in scored_stories]
    
    def _rank_by_recency(self, stories: List[SurvivorStory]) -> List[SurvivorStory]:
        """Rank by publication date"""
        return sorted(stories, key=lambda s: s.published_at or s.created_at, reverse=True)
    
    def _rank_by_similarity(
        self,
        stories: List[SurvivorStory],
        user_profile: Dict[str, Any]
    ) -> List[SurvivorStory]:
        """Rank by journey similarity"""
        scored_stories = []
        
        for story in stories:
            similarity_score = 0.0
            
            # Same condition (50% weight)
            if story.author.condition_type == user_profile.get('condition'):
                similarity_score += 0.5
            
            # Similar metrics (30% weight)
            user_weight_loss = user_profile.get('weight_change_kg', 0)
            story_weight_loss = story.metrics.weight_change_kg or 0
            
            if user_weight_loss and story_weight_loss:
                weight_similarity = 1 - (abs(user_weight_loss - story_weight_loss) / max(abs(user_weight_loss), abs(story_weight_loss)))
                similarity_score += weight_similarity * 0.3
            
            # Similar timeline (20% weight)
            user_days = user_profile.get('journey_days', 0)
            story_days = story.metrics.journey_duration_days
            
            if user_days and story_days:
                day_similarity = 1 - (abs(user_days - story_days) / max(user_days, story_days))
                similarity_score += day_similarity * 0.2
            
            scored_stories.append((story, similarity_score))
        
        scored_stories.sort(key=lambda x: x[1], reverse=True)
        return [story for story, score in scored_stories]
    
    def _rank_by_inspiration(self, stories: List[SurvivorStory]) -> List[SurvivorStory]:
        """Rank by inspirational value"""
        scored_stories = []
        
        for story in stories:
            inspiration_score = 0.0
            
            # Significant transformation (+40 points)
            if story.metrics.weight_change_kg and abs(story.metrics.weight_change_kg) >= 10:
                inspiration_score += 40
            
            if story.metrics.hba1c_change and abs(story.metrics.hba1c_change) >= 1.0:
                inspiration_score += 40
            
            if story.metrics.medications_reduced and story.metrics.medications_reduced > 0:
                inspiration_score += 30
            
            # Long streak (+20 points)
            if story.metrics.scan_streak_days and story.metrics.scan_streak_days >= 90:
                inspiration_score += 20
            
            # High engagement (positive response) (+20 points)
            inspiration_score += story.engagement.engagement_rate * 20
            
            # Verified data (+10 points)
            if story.metrics.data_verified:
                inspiration_score += 10
            
            scored_stories.append((story, inspiration_score))
        
        scored_stories.sort(key=lambda x: x[1], reverse=True)
        return [story for story, score in scored_stories]
    
    def _get_age_range(self, age: Optional[int]) -> Optional[str]:
        """Convert age to range"""
        if not age:
            return None
        
        if age < 25:
            return "18-24"
        elif age < 30:
            return "25-29"
        elif age < 35:
            return "30-34"
        elif age < 40:
            return "35-39"
        elif age < 45:
            return "40-44"
        elif age < 50:
            return "45-49"
        elif age < 55:
            return "50-54"
        elif age < 60:
            return "55-59"
        else:
            return "60+"


# ==================== MAIN ENGINE ====================

class PublicSurvivorStoryEngine:
    """Main engine for managing public survivor stories"""
    
    def __init__(self):
        self.anonymizer = StoryAnonymizer()
        self.moderator = ContentModerator()
        self.feed_algorithm = FeedAlgorithm()
        
        # In-memory storage (production would use database)
        self.stories: Dict[str, SurvivorStory] = {}
        self.user_stories: Dict[str, List[str]] = defaultdict(list)
    
    def create_story(
        self,
        user_id: str,
        content: StoryContent,
        metrics: StoryMetrics,
        visibility: StoryVisibility,
        format_type: StoryFormat,
        privacy_preferences: Dict[str, Any]
    ) -> SurvivorStory:
        """Create and publish survivor story"""
        # Generate story ID
        story_id = self._generate_story_id(user_id)
        
        # Anonymize author
        author = self.anonymizer.anonymize_author(user_id, privacy_preferences)
        
        # Anonymize content
        anonymized_content = self._anonymize_story_content(content)
        
        # Moderate content
        moderation = self.moderator.moderate_story(anonymized_content)
        
        # Create story
        story = SurvivorStory(
            story_id=story_id,
            author=author,
            content=anonymized_content,
            metrics=metrics,
            engagement=EngagementMetrics(),
            moderation=moderation,
            visibility=visibility,
            format=format_type,
            created_at=datetime.now(),
            published_at=datetime.now() if moderation.status == ModerationStatus.APPROVED else None,
            is_own_story=True
        )
        
        # Store story
        self.stories[story_id] = story
        self.user_stories[user_id].append(story_id)
        
        return story
    
    def get_feed(
        self,
        user_id: str,
        user_profile: Dict[str, Any],
        algorithm: FeedAlgorithmType = FeedAlgorithmType.PERSONALIZED,
        page: int = 1,
        page_size: int = 20
    ) -> List[SurvivorStory]:
        """Get personalized story feed"""
        # Get all approved public stories
        public_stories = [
            story for story in self.stories.values()
            if story.visibility in [StoryVisibility.PUBLIC, StoryVisibility.FEATURED]
            and story.moderation.status == ModerationStatus.APPROVED
        ]
        
        # Rank stories
        ranked_stories = self.feed_algorithm.rank_stories(
            public_stories,
            user_profile,
            algorithm
        )
        
        # Paginate
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        return ranked_stories[start_idx:end_idx]
    
    def engage_with_story(
        self,
        story_id: str,
        user_id: str,
        engagement_type: EngagementType
    ) -> bool:
        """Record user engagement"""
        if story_id not in self.stories:
            return False
        
        story = self.stories[story_id]
        
        if engagement_type == EngagementType.VIEW:
            story.engagement.views += 1
            story.engagement.views_last_24h += 1
            story.engagement.views_last_7d += 1
        elif engagement_type == EngagementType.LIKE:
            story.engagement.likes += 1
            story.engagement.liked_by_user = True
        elif engagement_type == EngagementType.SHARE:
            story.engagement.shares += 1
        elif engagement_type == EngagementType.COMMENT:
            story.engagement.comments += 1
        elif engagement_type == EngagementType.SAVE:
            story.engagement.saves += 1
            story.engagement.saved_by_user = True
        elif engagement_type == EngagementType.FOLLOW_AUTHOR:
            story.author_followed = True
        
        # Update engagement rate
        if story.engagement.views > 0:
            story.engagement.engagement_rate = (
                (story.engagement.likes + story.engagement.shares + story.engagement.comments) /
                story.engagement.views
            )
            story.engagement.viral_coefficient = story.engagement.shares / story.engagement.views
        
        return True
    
    def report_story(
        self,
        story_id: str,
        user_id: str,
        reason: ReportReason,
        details: Optional[str] = None
    ) -> bool:
        """Report inappropriate content"""
        if story_id not in self.stories:
            return False
        
        story = self.stories[story_id]
        
        # Update moderation status
        if story.moderation.status == ModerationStatus.APPROVED:
            story.moderation.status = ModerationStatus.FLAGGED
            story.moderation.detected_patterns.append(f"user_report_{reason.value}")
            story.moderation.review_notes = details
        
        # In production, this would trigger human review
        print(f"Story {story_id} reported for {reason.value}")
        
        return True
    
    def _anonymize_story_content(self, content: StoryContent) -> StoryContent:
        """Anonymize all story content"""
        # Anonymize title
        anon_title, _ = self.anonymizer.anonymize_content(content.title)
        
        # Anonymize body
        anon_body, _ = self.anonymizer.anonymize_content(content.body)
        
        # Anonymize sections
        anon_struggle = None
        anon_breakthrough = None
        anon_advice = None
        
        if content.struggle_section:
            anon_struggle, _ = self.anonymizer.anonymize_content(content.struggle_section)
        
        if content.breakthrough_section:
            anon_breakthrough, _ = self.anonymizer.anonymize_content(content.breakthrough_section)
        
        if content.advice_section:
            anon_advice, _ = self.anonymizer.anonymize_content(content.advice_section)
        
        return StoryContent(
            title=anon_title,
            body=anon_body,
            images=content.images,  # Images already processed
            video_url=content.video_url,
            struggle_section=anon_struggle,
            breakthrough_section=anon_breakthrough,
            advice_section=anon_advice,
            hashtags=content.hashtags,
            categories=content.categories,
            shareable_quote=content.shareable_quote
        )
    
    def _generate_story_id(self, user_id: str) -> str:
        """Generate unique story ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = ''.join(random.choices('0123456789abcdef', k=8))
        return f"story_{timestamp}_{random_suffix}"


# ==================== DEMONSTRATION ====================

def demonstrate_public_survivor_stories():
    """Demonstrate public survivor story system"""
    print("=" * 80)
    print("FEATURE 8: PUBLIC SURVIVOR STORIES (SOCIAL FEED) DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize engine
    print("🚀 Initializing Public Survivor Story Engine...")
    engine = PublicSurvivorStoryEngine()
    print("✅ Engine initialized with anonymization, moderation, and feed algorithms")
    print()
    
    # Create sample user profile
    user_id = "user_12345"
    user_profile = {
        'age': 42,
        'condition': 'Type 2 Diabetes',
        'journey_days': 180,
        'weight_change_kg': -15.0,
        'goals': ['Weight Loss', 'Diabetes Reversal', 'Medication Reduction'],
        'location': 'California'
    }
    
    privacy_prefs = {
        'age': 42,
        'show_age': True,
        'show_region': False,
        'show_condition': True,
        'location': 'California',
        'journey_duration': '6 months',
        'condition_type': 'Type 2 Diabetes',
        'has_verified_data': True
    }
    
    print("=" * 80)
    print("CREATING SURVIVOR STORY")
    print("=" * 80)
    print()
    
    # Create story content
    content = StoryContent(
        title="How I Reversed Type 2 Diabetes in 6 Months",
        body=(
            "Six months ago, I was diagnosed with Type 2 Diabetes. My HbA1c was 8.5%, "
            "I weighed 220 lbs, and I was on three medications. I felt hopeless. "
            "\n\n"
            "But then I discovered Wellomex and everything changed. The molecular scanner "
            "helped me understand exactly how different foods affected my glucose. "
            "I learned that my 'healthy' breakfast cereal was spiking my blood sugar more "
            "than a candy bar! "
            "\n\n"
            "The turning point came when I hit my first 30-day scan streak. Seeing those "
            "consistent patterns helped me make better choices. My glucose started stabilizing, "
            "the weight came off naturally, and my doctor started reducing my medications. "
            "\n\n"
            "Today, my HbA1c is 5.4% (normal range!), I've lost 35 pounds, and I'm completely "
            "off two of my three medications. My doctor says I'm in remission. "
            "\n\n"
            "If you're struggling, know this: Small consistent changes compound. You don't "
            "need perfection. You need progress. And Wellomex makes that progress visible "
            "every single day."
        ),
        struggle_section="Feeling hopeless with HbA1c 8.5% and three medications",
        breakthrough_section="30-day scan streak and discovering food-glucose patterns",
        advice_section="Focus on consistency over perfection. Scan every meal. Trust the data.",
        hashtags=["#Type2DiabetesReversal", "#WellomexSuccess", "#HealthTransformation"],
        categories=["Diabetes Reversal", "Weight Loss", "Medication Reduction"],
        shareable_quote="Six months ago: HbA1c 8.5%, 220 lbs, 3 meds. Today: HbA1c 5.4%, 185 lbs, 1 med. Type 2 Diabetes in remission! 💪"
    )
    
    # Create metrics
    metrics = StoryMetrics(
        weight_change_kg=-15.9,  # ~35 lbs
        weight_change_percent=-16.2,
        hba1c_change=-3.1,
        glucose_improvement="85% time in range",
        scan_streak_days=180,
        compliance_rate=0.89,
        medications_reduced=2,
        journey_duration_days=180,
        data_verified=True
    )
    
    # Create story
    story = engine.create_story(
        user_id=user_id,
        content=content,
        metrics=metrics,
        visibility=StoryVisibility.PUBLIC,
        format_type=StoryFormat.TEXT_ONLY,
        privacy_preferences=privacy_prefs
    )
    
    print(f"📖 Story Created: {story.story_id}")
    print(f"   Author: {story.author.display_name}")
    print(f"   Anonymous ID: {story.author.anonymous_id}")
    print(f"   Visibility: {story.visibility.value}")
    print(f"   Format: {story.format.value}")
    print()
    
    print("👤 Author Profile (Anonymized):")
    print(f"   Display Name: {story.author.display_name}")
    print(f"   Age Range: {story.author.age_range}")
    print(f"   Region: {story.author.region or 'Not Shared'}")
    print(f"   Condition: {story.author.condition_type}")
    print(f"   Journey Duration: {story.author.journey_duration}")
    print(f"   Verified Transformation: {'✅ Yes' if story.author.verified_transformation else '❌ No'}")
    print()
    
    print("📊 Transformation Metrics:")
    print(f"   Weight Loss: {metrics.weight_change_kg:.1f} kg ({metrics.weight_change_percent:.1f}%)")
    print(f"   HbA1c Change: {metrics.hba1c_change:.1f}%")
    print(f"   Glucose Control: {metrics.glucose_improvement}")
    print(f"   Scan Streak: {metrics.scan_streak_days} days")
    print(f"   Compliance Rate: {metrics.compliance_rate:.1%}")
    print(f"   Medications Reduced: {metrics.medications_reduced}")
    print(f"   Journey Duration: {metrics.journey_duration_days} days")
    print(f"   Data Verified: {'✅ Yes' if metrics.data_verified else '❌ No'}")
    print()
    
    print("🛡️ Content Moderation:")
    print(f"   Status: {story.moderation.status.value}")
    print(f"   Confidence: {story.moderation.confidence:.1%}")
    print(f"   Contains PII: {'❌ Yes' if story.moderation.contains_pii else '✅ No'}")
    print(f"   Is Spam: {'❌ Yes' if story.moderation.is_spam else '✅ No'}")
    print(f"   Is Inappropriate: {'❌ Yes' if story.moderation.is_inappropriate else '✅ No'}")
    print(f"   Is Medical Advice: {'⚠️ Yes' if story.moderation.is_medical_advice else '✅ No'}")
    print(f"   Reviewed By: {story.moderation.reviewed_by}")
    print()
    
    print("=" * 80)
    print("SIMULATING ENGAGEMENT")
    print("=" * 80)
    print()
    
    # Simulate engagement
    engagement_events = [
        (EngagementType.VIEW, 150),
        (EngagementType.LIKE, 45),
        (EngagementType.SHARE, 12),
        (EngagementType.COMMENT, 8),
        (EngagementType.SAVE, 18)
    ]
    
    for eng_type, count in engagement_events:
        for _ in range(count):
            engine.engage_with_story(story.story_id, f"viewer_{random.randint(1000, 9999)}", eng_type)
    
    print(f"📈 Engagement After 24 Hours:")
    print(f"   Views: {story.engagement.views:,}")
    print(f"   Likes: {story.engagement.likes:,}")
    print(f"   Shares: {story.engagement.shares:,}")
    print(f"   Comments: {story.engagement.comments:,}")
    print(f"   Saves: {story.engagement.saves:,}")
    print(f"   Engagement Rate: {story.engagement.engagement_rate:.1%}")
    print(f"   Viral Coefficient: {story.engagement.viral_coefficient:.3f}")
    print()
    
    print("=" * 80)
    print("CREATING ADDITIONAL STORIES FOR FEED")
    print("=" * 80)
    print()
    
    # Create 4 more stories with different profiles
    additional_stories = [
        {
            'title': "Lost 50 Pounds Without Feeling Deprived",
            'condition': "Weight Management",
            'weight_change': -22.7,
            'days': 240,
            'age': 35
        },
        {
            'title': "From Pre-Diabetic to Perfect Glucose Control",
            'condition': "Pre-Diabetes",
            'weight_change': -8.2,
            'days': 120,
            'age': 45
        },
        {
            'title': "How I Eliminated Hypertension Medication",
            'condition': "Hypertension",
            'weight_change': -12.3,
            'days': 180,
            'age': 52
        },
        {
            'title': "90-Day Transformation: My PCOS Journey",
            'condition': "PCOS",
            'weight_change': -9.5,
            'days': 90,
            'age': 28
        }
    ]
    
    for idx, story_data in enumerate(additional_stories, start=2):
        temp_content = StoryContent(
            title=story_data['title'],
            body=f"This is my {story_data['days']}-day journey with {story_data['condition']}...",
            categories=[story_data['condition']],
            hashtags=[f"#{story_data['condition'].replace(' ', '')}"]
        )
        
        temp_metrics = StoryMetrics(
            weight_change_kg=story_data['weight_change'],
            journey_duration_days=story_data['days'],
            data_verified=True
        )
        
        temp_prefs = {
            'age': story_data['age'],
            'show_age': True,
            'show_region': False,
            'show_condition': True,
            'journey_duration': f"{story_data['days']} days",
            'condition_type': story_data['condition'],
            'has_verified_data': True
        }
        
        temp_story = engine.create_story(
            user_id=f"user_{10000 + idx}",
            content=temp_content,
            metrics=temp_metrics,
            visibility=StoryVisibility.PUBLIC,
            format_type=StoryFormat.TEXT_ONLY,
            privacy_preferences=temp_prefs
        )
        
        # Simulate random engagement
        views = random.randint(50, 200)
        for _ in range(views):
            engine.engage_with_story(temp_story.story_id, f"viewer_{random.randint(1000, 9999)}", EngagementType.VIEW)
        for _ in range(int(views * 0.25)):
            engine.engage_with_story(temp_story.story_id, f"viewer_{random.randint(1000, 9999)}", EngagementType.LIKE)
        
        print(f"✅ Created: {story_data['title']}")
    
    print()
    
    print("=" * 80)
    print("GENERATING PERSONALIZED FEED")
    print("=" * 80)
    print()
    
    # Test different feed algorithms
    algorithms_to_test = [
        FeedAlgorithmType.PERSONALIZED,
        FeedAlgorithmType.TRENDING,
        FeedAlgorithmType.SIMILAR_JOURNEY
    ]
    
    for algo in algorithms_to_test:
        print(f"📱 Feed Algorithm: {algo.value.upper()}")
        print("-" * 80)
        
        feed = engine.get_feed(
            user_id=user_id,
            user_profile=user_profile,
            algorithm=algo,
            page=1,
            page_size=5
        )
        
        for rank, feed_story in enumerate(feed, start=1):
            print(f"   {rank}. {feed_story.content.title}")
            print(f"      Author: {feed_story.author.display_name} | {feed_story.author.condition_type}")
            print(f"      Metrics: {feed_story.metrics.weight_change_kg:.1f}kg, {feed_story.metrics.journey_duration_days} days")
            print(f"      Engagement: {feed_story.engagement.views} views, {feed_story.engagement.likes} likes")
            print()
        
        print()
    
    print("=" * 80)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print()
    
    print("📊 This system showcases:")
    print("   • Privacy-first anonymization (no real names, hashed IDs)")
    print("   • AI-powered content moderation (PII, spam, inappropriate)")
    print("   • Multiple feed algorithms (personalized, trending, similar)")
    print("   • Engagement tracking (views, likes, shares, comments)")
    print("   • Verified transformations with data backing")
    print("   • Viral growth mechanics (engagement rate, viral coefficient)")
    print("   • Community safety (report system, flagging)")
    print()
    
    print("🎯 Production Implementation:")
    print("   • NLP-based anonymization (NER for PII detection)")
    print("   • ML content moderation (BERT, GPT-4 vision for images)")
    print("   • Collaborative filtering for recommendations")
    print("   • Real-time trending score updates")
    print("   • Push notifications for viral stories")
    print("   • Comment threads with nested replies")
    print("   • Story reactions (heart, fire, clap, inspire)")
    print("   • Featured story editorial curation")


if __name__ == "__main__":
    demonstrate_public_survivor_stories()
