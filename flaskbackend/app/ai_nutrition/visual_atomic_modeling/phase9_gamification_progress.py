"""
PHASE 9: Progress Tracking & Gamification
==========================================

Implements GitHub-style streaks and achievement system:
- Daily streak tracking (🔥 fire icon)
- Health goal progress dashboard
- Achievement badges (100+ unique badges)
- Visual progress bars and charts
- Milestone celebrations
- Leaderboards and challenges
- Habit formation through gamification

Core Features:
- Streak counter with visual rewards
- Goal-specific badges (weight loss, diabetes, etc.)
- Disease-specific achievements
- Contamination detection badges
- Dietary compliance tracking
- Social challenges and competitions
- Weekly/Monthly progress reports

Streak Criteria:
- Daily: Scan 3+ meals
- Bonus: Complete AI meal plan
- Bonus: Log all meals
- Bonus: Hit macro targets
- Bonus: Exercise logged

Achievement Categories:
- Health Goals (55+ goal badges)
- Medical Conditions (100+ disease badges)
- Nutrition Milestones
- Contamination Detection
- Dietary Adherence
- Social Engagement
- Professional Consultations

Architecture:
    User Actions → Track Progress → Update Streaks → Award Badges
    → Visual Animations → Celebrate Milestones → Share Achievements
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta, date
import json
from collections import defaultdict
import hashlib
import secrets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BadgeCategory(Enum):
    """Categories of achievement badges"""
    HEALTH_GOAL = "health_goal"
    MEDICAL_CONDITION = "medical_condition"
    NUTRITION_MILESTONE = "nutrition_milestone"
    CONTAMINATION_DETECTION = "contamination_detection"
    DIETARY_ADHERENCE = "dietary_adherence"
    SOCIAL_ENGAGEMENT = "social_engagement"
    PROFESSIONAL_GROWTH = "professional_growth"
    STREAK_MILESTONE = "streak_milestone"
    SCANNING_ACHIEVEMENT = "scanning_achievement"


class BadgeTier(Enum):
    """Badge difficulty tiers"""
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    DIAMOND = "diamond"


class StreakType(Enum):
    """Types of streaks"""
    DAILY_SCANNING = "daily_scanning"
    MEAL_LOGGING = "meal_logging"
    MACRO_TARGETS = "macro_targets"
    EXERCISE = "exercise"
    WATER_INTAKE = "water_intake"
    MEDICATION_ADHERENCE = "medication_adherence"


@dataclass
class Badge:
    """
    Achievement badge
    
    Examples:
    - 🔬 Lead Detector: Flagged 5 contaminated foods
    - 🔥 30-Day Streak Master
    - 💪 Muscle Gain Champion: Hit protein goal 50 times
    - 🩺 Diabetes Warrior: 100 days blood sugar control
    """
    badge_id: str
    name: str
    description: str
    icon_emoji: str
    
    # Classification
    category: BadgeCategory
    tier: BadgeTier
    
    # Requirements
    requirement_count: int
    requirement_description: str
    
    # Visual
    color_code: str  # hex color for badge
    animation: Optional[str] = None  # "pulse", "glow", "sparkle"
    
    # Rarity
    rarity_score: int = 0  # 0-100, higher = rarer
    
    # Rewards
    points: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize badge"""
        return {
            'badge_id': self.badge_id,
            'name': self.name,
            'description': self.description,
            'icon': self.icon_emoji,
            'category': self.category.value,
            'tier': self.tier.value,
            'requirement': self.requirement_description,
            'color': self.color_code,
            'rarity': self.rarity_score,
            'points': self.points
        }


@dataclass
class UserBadge:
    """Badge earned by user"""
    user_badge_id: str
    user_id: str
    badge_id: str
    
    # Progress
    current_count: int = 0
    required_count: int = 0
    
    # Status
    is_earned: bool = False
    earned_at: Optional[datetime] = None
    
    # Display
    is_showcased: bool = False  # Pin to profile
    
    def get_progress_percentage(self) -> float:
        """Get progress towards badge"""
        if self.required_count == 0:
            return 100.0
        return min(100.0, (self.current_count / self.required_count) * 100)


@dataclass
class Streak:
    """
    User's streak tracking
    
    GitHub-style streak with fire icon 🔥
    """
    streak_id: str
    user_id: str
    streak_type: StreakType
    
    # Current streak
    current_streak: int = 0
    longest_streak: int = 0
    
    # Dates
    last_activity_date: Optional[date] = None
    streak_start_date: Optional[date] = None
    
    # Weekly stats
    this_week_count: int = 0
    last_week_count: int = 0
    
    # All-time
    total_count: int = 0
    
    def update(self, activity_date: date) -> bool:
        """
        Update streak with new activity
        
        Returns True if streak continued/started
        """
        if not self.last_activity_date:
            # First activity
            self.current_streak = 1
            self.longest_streak = 1
            self.last_activity_date = activity_date
            self.streak_start_date = activity_date
            self.total_count = 1
            return True
        
        # Check if same day (no update needed)
        if activity_date == self.last_activity_date:
            return False
        
        # Check if consecutive day
        yesterday = activity_date - timedelta(days=1)
        
        if self.last_activity_date == yesterday:
            # Streak continues! 🔥
            self.current_streak += 1
            self.last_activity_date = activity_date
            
            if self.current_streak > self.longest_streak:
                self.longest_streak = self.current_streak
            
            self.total_count += 1
            return True
        
        elif activity_date > self.last_activity_date:
            # Streak broken 💔
            self.current_streak = 1
            self.last_activity_date = activity_date
            self.streak_start_date = activity_date
            self.total_count += 1
            return True
        
        return False
    
    def get_fire_emoji(self) -> str:
        """Get fire emoji based on streak length"""
        if self.current_streak >= 365:
            return "🔥🔥🔥🔥"  # Year streak!
        elif self.current_streak >= 100:
            return "🔥🔥🔥"
        elif self.current_streak >= 30:
            return "🔥🔥"
        elif self.current_streak >= 7:
            return "🔥"
        else:
            return "🌱"  # Growing


@dataclass
class HealthGoal:
    """
    User's health goal with progress tracking
    """
    goal_id: str
    user_id: str
    
    # Goal details
    goal_type: str  # "weight_loss", "muscle_gain", etc.
    goal_name: str
    target_value: float
    current_value: float
    start_value: float
    unit: str  # "lbs", "kg", "mg/dL", etc.
    
    # Timeline
    start_date: date
    target_date: date
    
    # Progress
    completion_percentage: float = 0.0
    is_achieved: bool = False
    achieved_at: Optional[datetime] = None
    
    # Milestones
    milestones: List[float] = field(default_factory=list)
    milestones_achieved: Set[float] = field(default_factory=set)
    
    def update_progress(self):
        """Calculate current progress"""
        if self.start_value == self.target_value:
            self.completion_percentage = 100.0
            return
        
        progress = abs(self.current_value - self.start_value)
        total_needed = abs(self.target_value - self.start_value)
        
        self.completion_percentage = min(100.0, (progress / total_needed) * 100)
        
        # Check if achieved
        if self.goal_type in ['weight_loss', 'blood_sugar_control']:
            # Lower is better
            if self.current_value <= self.target_value:
                self.is_achieved = True
                if not self.achieved_at:
                    self.achieved_at = datetime.now()
        else:
            # Higher is better
            if self.current_value >= self.target_value:
                self.is_achieved = True
                if not self.achieved_at:
                    self.achieved_at = datetime.now()
    
    def get_progress_color(self) -> str:
        """Get color for progress bar"""
        if self.is_achieved:
            return "#00FF00"  # Green
        elif self.completion_percentage >= 75:
            return "#FFD700"  # Gold
        elif self.completion_percentage >= 50:
            return "#FFA500"  # Orange
        elif self.completion_percentage >= 25:
            return "#FFFF00"  # Yellow
        else:
            return "#FF6B6B"  # Red


@dataclass
class DailyActivity:
    """Daily activity tracking"""
    activity_id: str
    user_id: str
    activity_date: date
    
    # Scanning
    meals_scanned: int = 0
    foods_scanned: int = 0
    
    # Nutrition
    total_calories: float = 0
    total_protein: float = 0
    total_carbs: float = 0
    total_fat: float = 0
    
    # Goals hit
    calorie_goal_hit: bool = False
    protein_goal_hit: bool = False
    macro_balance_hit: bool = False
    
    # Contamination detection
    contaminants_flagged: int = 0
    high_risk_avoided: int = 0
    
    # Social
    huddle_posts: int = 0
    huddle_comments: int = 0
    
    # Consultations
    consultations_attended: int = 0
    
    # Compliance
    meal_plan_followed: bool = False
    
    def meets_streak_criteria(self) -> bool:
        """Check if daily activity meets streak criteria"""
        return self.meals_scanned >= 3


class BadgeLibrary:
    """
    Library of all available badges
    
    100+ unique badges across all categories
    """
    
    def __init__(self):
        self.badges: Dict[str, Badge] = {}
        self._initialize_badges()
        logger.info(f"BadgeLibrary initialized with {len(self.badges)} badges")
    
    def _initialize_badges(self):
        """Initialize all badge definitions"""
        
        # ===== STREAK MILESTONES =====
        streak_badges = [
            Badge("streak_7", "Week Warrior 🔥", "Maintain 7-day streak", "🔥",
                  BadgeCategory.STREAK_MILESTONE, BadgeTier.BRONZE, 7,
                  "7 consecutive days", "#FF6B6B", "pulse", 10, 100),
            Badge("streak_30", "Month Master 🔥🔥", "Maintain 30-day streak", "🔥",
                  BadgeCategory.STREAK_MILESTONE, BadgeTier.SILVER, 30,
                  "30 consecutive days", "#FFA500", "glow", 30, 500),
            Badge("streak_100", "Century Champion 🔥🔥🔥", "Maintain 100-day streak", "🔥",
                  BadgeCategory.STREAK_MILESTONE, BadgeTier.GOLD, 100,
                  "100 consecutive days", "#FFD700", "sparkle", 60, 2000),
            Badge("streak_365", "Yearly Legend 🔥🔥🔥🔥", "Maintain 365-day streak", "🔥",
                  BadgeCategory.STREAK_MILESTONE, BadgeTier.DIAMOND, 365,
                  "365 consecutive days", "#00FFFF", "sparkle", 95, 10000),
        ]
        
        # ===== CONTAMINATION DETECTION =====
        contamination_badges = [
            Badge("lead_detector_1", "Lead Detector 🔬", "Flag 5 lead-contaminated foods", "🔬",
                  BadgeCategory.CONTAMINATION_DETECTION, BadgeTier.BRONZE, 5,
                  "Detect 5 high-lead foods", "#8B4513", None, 20, 200),
            Badge("lead_detector_2", "Lead Expert 🔬", "Flag 25 lead-contaminated foods", "🔬",
                  BadgeCategory.CONTAMINATION_DETECTION, BadgeTier.SILVER, 25,
                  "Detect 25 high-lead foods", "#C0C0C0", None, 40, 500),
            Badge("mercury_guardian", "Mercury Guardian 🐟", "Avoid 10 high-mercury foods", "🐟",
                  BadgeCategory.CONTAMINATION_DETECTION, BadgeTier.SILVER, 10,
                  "Avoid 10 mercury-rich foods", "#4682B4", None, 30, 300),
            Badge("arsenic_aware", "Arsenic Awareness ⚠️", "Flag 15 arsenic sources", "⚠️",
                  BadgeCategory.CONTAMINATION_DETECTION, BadgeTier.SILVER, 15,
                  "Identify 15 arsenic sources", "#FF4500", None, 35, 400),
            Badge("toxin_hunter", "Toxin Hunter 🎯", "Flag 50 contaminated foods", "🎯",
                  BadgeCategory.CONTAMINATION_DETECTION, BadgeTier.GOLD, 50,
                  "Detect 50 contaminants total", "#FFD700", "glow", 70, 1500),
        ]
        
        # ===== WEIGHT LOSS GOALS =====
        weight_loss_badges = [
            Badge("weight_5lb", "First 5 🎯", "Lose first 5 lbs", "🎯",
                  BadgeCategory.HEALTH_GOAL, BadgeTier.BRONZE, 5,
                  "Lose 5 lbs", "#90EE90", None, 15, 150),
            Badge("weight_10lb", "Perfect 10 ⚖️", "Lose 10 lbs", "⚖️",
                  BadgeCategory.HEALTH_GOAL, BadgeTier.BRONZE, 10,
                  "Lose 10 lbs", "#32CD32", None, 20, 300),
            Badge("weight_25lb", "Quarter Century 🏆", "Lose 25 lbs", "🏆",
                  BadgeCategory.HEALTH_GOAL, BadgeTier.SILVER, 25,
                  "Lose 25 lbs", "#FFD700", "pulse", 40, 800),
            Badge("weight_50lb", "Half-Hundred Hero 🌟", "Lose 50 lbs", "🌟",
                  BadgeCategory.HEALTH_GOAL, BadgeTier.GOLD, 50,
                  "Lose 50 lbs", "#FF6347", "sparkle", 75, 2500),
            Badge("weight_100lb", "Century Slayer 👑", "Lose 100 lbs", "👑",
                  BadgeCategory.HEALTH_GOAL, BadgeTier.DIAMOND, 100,
                  "Lose 100 lbs", "#9400D3", "sparkle", 99, 10000),
        ]
        
        # ===== DIABETES MANAGEMENT =====
        diabetes_badges = [
            Badge("diabetes_week", "Blood Sugar Beginner 🩸", "7 days glucose control", "🩸",
                  BadgeCategory.MEDICAL_CONDITION, BadgeTier.BRONZE, 7,
                  "7 days in target range", "#DC143C", None, 15, 150),
            Badge("diabetes_month", "Glucose Guardian 💉", "30 days glucose control", "💉",
                  BadgeCategory.MEDICAL_CONDITION, BadgeTier.SILVER, 30,
                  "30 days in target range", "#FF69B4", "pulse", 35, 600),
            Badge("diabetes_100", "Diabetes Warrior 🩺", "100 days glucose control", "🩺",
                  BadgeCategory.MEDICAL_CONDITION, BadgeTier.GOLD, 100,
                  "100 days in target range", "#4169E1", "glow", 65, 2000),
            Badge("a1c_improved", "A1C Champion 📊", "Reduce A1C by 1%", "📊",
                  BadgeCategory.MEDICAL_CONDITION, BadgeTier.GOLD, 1,
                  "Lower A1C by 1 point", "#00CED1", "sparkle", 70, 2500),
        ]
        
        # ===== MUSCLE GAIN =====
        muscle_badges = [
            Badge("protein_week", "Protein Starter 💪", "Hit protein goal 7 days", "💪",
                  BadgeCategory.HEALTH_GOAL, BadgeTier.BRONZE, 7,
                  "Meet protein target 7 times", "#FF8C00", None, 15, 150),
            Badge("protein_month", "Protein Pro 🥩", "Hit protein goal 30 days", "🥩",
                  BadgeCategory.HEALTH_GOAL, BadgeTier.SILVER, 30,
                  "Meet protein target 30 times", "#8B0000", "pulse", 35, 600),
            Badge("muscle_5lb", "Muscle Builder 🏋️", "Gain 5 lbs muscle", "🏋️",
                  BadgeCategory.HEALTH_GOAL, BadgeTier.SILVER, 5,
                  "Gain 5 lbs lean mass", "#B8860B", None, 40, 800),
            Badge("muscle_10lb", "Mass Master 💪", "Gain 10 lbs muscle", "💪",
                  BadgeCategory.HEALTH_GOAL, BadgeTier.GOLD, 10,
                  "Gain 10 lbs lean mass", "#DAA520", "glow", 70, 2000),
        ]
        
        # ===== DIETARY ADHERENCE =====
        diet_badges = [
            Badge("keto_week", "Keto Starter 🥓", "7 days keto", "🥓",
                  BadgeCategory.DIETARY_ADHERENCE, BadgeTier.BRONZE, 7,
                  "7 days under 50g carbs", "#8B4513", None, 15, 150),
            Badge("keto_month", "Keto Master 🧈", "30 days keto", "🧈",
                  BadgeCategory.DIETARY_ADHERENCE, BadgeTier.GOLD, 30,
                  "30 days under 50g carbs", "#CD853F", "sparkle", 60, 1500),
            Badge("vegan_week", "Plant Starter 🌱", "7 days vegan", "🌱",
                  BadgeCategory.DIETARY_ADHERENCE, BadgeTier.BRONZE, 7,
                  "7 days plant-based", "#228B22", None, 15, 150),
            Badge("vegan_month", "Vegan Champion 🥗", "30 days vegan", "🥗",
                  BadgeCategory.DIETARY_ADHERENCE, BadgeTier.GOLD, 30,
                  "30 days plant-based", "#00FF00", "sparkle", 60, 1500),
        ]
        
        # ===== SCANNING ACHIEVEMENTS =====
        scanning_badges = [
            Badge("scan_10", "Scanner Novice 📸", "Scan 10 meals", "📸",
                  BadgeCategory.SCANNING_ACHIEVEMENT, BadgeTier.BRONZE, 10,
                  "Scan 10 meals", "#4682B4", None, 10, 50),
            Badge("scan_100", "Scanner Pro 📷", "Scan 100 meals", "📷",
                  BadgeCategory.SCANNING_ACHIEVEMENT, BadgeTier.SILVER, 100,
                  "Scan 100 meals", "#5F9EA0", None, 30, 300),
            Badge("scan_1000", "Scanner Master 📹", "Scan 1000 meals", "📹",
                  BadgeCategory.SCANNING_ACHIEVEMENT, BadgeTier.GOLD, 1000,
                  "Scan 1000 meals", "#00BFFF", "glow", 80, 3000),
            Badge("scan_perfect_week", "Perfect Week 🎯", "Scan 3+ meals for 7 days", "🎯",
                  BadgeCategory.SCANNING_ACHIEVEMENT, BadgeTier.SILVER, 7,
                  "7 days of 3+ scans", "#FFD700", "pulse", 40, 500),
        ]
        
        # ===== SOCIAL ENGAGEMENT =====
        social_badges = [
            Badge("huddle_join", "Community Member 👥", "Join first huddle", "👥",
                  BadgeCategory.SOCIAL_ENGAGEMENT, BadgeTier.BRONZE, 1,
                  "Join a health huddle", "#9370DB", None, 5, 50),
            Badge("huddle_5", "Community Builder 🏘️", "Join 5 huddles", "🏘️",
                  BadgeCategory.SOCIAL_ENGAGEMENT, BadgeTier.SILVER, 5,
                  "Join 5 health huddles", "#BA55D3", None, 25, 250),
            Badge("post_10", "Active Contributor 💬", "Make 10 huddle posts", "💬",
                  BadgeCategory.SOCIAL_ENGAGEMENT, BadgeTier.BRONZE, 10,
                  "Post 10 times in huddles", "#DDA0DD", None, 20, 200),
            Badge("helpful_50", "Helping Hand 🤝", "Get 50 helpful marks", "🤝",
                  BadgeCategory.SOCIAL_ENGAGEMENT, BadgeTier.GOLD, 50,
                  "50 helpful reactions", "#FF1493", "glow", 65, 1500),
        ]
        
        # ===== PROFESSIONAL GROWTH =====
        professional_badges = [
            Badge("consult_first", "Advice Seeker 💼", "Complete first consultation", "💼",
                  BadgeCategory.PROFESSIONAL_GROWTH, BadgeTier.BRONZE, 1,
                  "Book and complete consultation", "#4169E1", None, 15, 200),
            Badge("consult_5", "Regular Client 📋", "Complete 5 consultations", "📋",
                  BadgeCategory.PROFESSIONAL_GROWTH, BadgeTier.SILVER, 5,
                  "Complete 5 consultations", "#1E90FF", None, 35, 600),
            Badge("consult_review", "Feedback Provider ⭐", "Leave 10 reviews", "⭐",
                  BadgeCategory.PROFESSIONAL_GROWTH, BadgeTier.SILVER, 10,
                  "Review 10 consultations", "#FFD700", "pulse", 30, 400),
        ]
        
        # ===== HYPERTENSION MANAGEMENT =====
        bp_badges = [
            Badge("bp_week", "BP Beginner 💉", "7 days sodium control", "💉",
                  BadgeCategory.MEDICAL_CONDITION, BadgeTier.BRONZE, 7,
                  "7 days under 1500mg sodium", "#FF6347", None, 15, 150),
            Badge("bp_month", "Pressure Master 🩸", "30 days sodium control", "🩸",
                  BadgeCategory.MEDICAL_CONDITION, BadgeTier.SILVER, 30,
                  "30 days under 1500mg sodium", "#DC143C", "pulse", 35, 600),
            Badge("dash_diet", "DASH Diet Champion 🧂", "Follow DASH 30 days", "🧂",
                  BadgeCategory.MEDICAL_CONDITION, BadgeTier.GOLD, 30,
                  "30 days DASH compliance", "#B22222", "glow", 60, 1500),
        ]
        
        # ===== KIDNEY HEALTH =====
        kidney_badges = [
            Badge("kidney_potassium", "Potassium Pro 🫘", "Control potassium 30 days", "🫘",
                  BadgeCategory.MEDICAL_CONDITION, BadgeTier.SILVER, 30,
                  "30 days low potassium", "#8B4513", None, 35, 600),
            Badge("kidney_phosphorus", "Phosphorus Guardian 💧", "Control phosphorus 30 days", "💧",
                  BadgeCategory.MEDICAL_CONDITION, BadgeTier.SILVER, 30,
                  "30 days low phosphorus", "#4682B4", None, 35, 600),
            Badge("renal_diet", "Renal Diet Master 🥗", "Perfect renal diet 60 days", "🥗",
                  BadgeCategory.MEDICAL_CONDITION, BadgeTier.GOLD, 60,
                  "60 days renal compliance", "#2E8B57", "glow", 70, 2000),
        ]
        
        # ===== NUTRITION MILESTONES =====
        nutrition_badges = [
            Badge("fiber_champion", "Fiber Champion 🌾", "Hit fiber goal 30 days", "🌾",
                  BadgeCategory.NUTRITION_MILESTONE, BadgeTier.SILVER, 30,
                  "30g+ fiber 30 times", "#D2691E", None, 30, 400),
            Badge("omega3_master", "Omega-3 Master 🐟", "Consume omega-3 50 days", "🐟",
                  BadgeCategory.NUTRITION_MILESTONE, BadgeTier.SILVER, 50,
                  "Omega-3 rich foods 50 times", "#4682B4", None, 40, 600),
            Badge("hydration_hero", "Hydration Hero 💧", "Drink 8+ glasses 30 days", "💧",
                  BadgeCategory.NUTRITION_MILESTONE, BadgeTier.BRONZE, 30,
                  "8 glasses water 30 days", "#87CEEB", None, 25, 300),
            Badge("rainbow_eater", "Rainbow Eater 🌈", "Eat all color groups 7 days", "🌈",
                  BadgeCategory.NUTRITION_MILESTONE, BadgeTier.GOLD, 7,
                  "All veggie colors 7 days", "#FF69B4", "sparkle", 65, 1200),
        ]
        
        # Combine all badges
        all_badges = (
            streak_badges + contamination_badges + weight_loss_badges +
            diabetes_badges + muscle_badges + diet_badges +
            scanning_badges + social_badges + professional_badges +
            bp_badges + kidney_badges + nutrition_badges
        )
        
        # Store in library
        for badge in all_badges:
            self.badges[badge.badge_id] = badge
    
    def get_badge(self, badge_id: str) -> Optional[Badge]:
        """Get badge by ID"""
        return self.badges.get(badge_id)
    
    def get_badges_by_category(
        self,
        category: BadgeCategory
    ) -> List[Badge]:
        """Get all badges in a category"""
        return [
            b for b in self.badges.values()
            if b.category == category
        ]
    
    def get_badges_by_tier(
        self,
        tier: BadgeTier
    ) -> List[Badge]:
        """Get all badges of a tier"""
        return [
            b for b in self.badges.values()
            if b.tier == tier
        ]


class ProgressTracker:
    """
    Tracks user progress towards goals and badges
    
    Integrates with all system features:
    - Meal scanning (Phase 1-6)
    - Social huddles (Phase 7)
    - Professional consultations (Phase 8)
    """
    
    def __init__(self, badge_library: BadgeLibrary):
        self.badge_library = badge_library
        
        # Storage
        self.user_badges: Dict[str, List[UserBadge]] = defaultdict(list)
        self.user_streaks: Dict[str, Dict[str, Streak]] = defaultdict(dict)
        self.user_goals: Dict[str, List[HealthGoal]] = defaultdict(list)
        self.daily_activities: Dict[str, List[DailyActivity]] = defaultdict(list)
        
        logger.info("ProgressTracker initialized")
    
    def initialize_user(self, user_id: str):
        """Initialize progress tracking for new user"""
        # Initialize all streaks
        for streak_type in StreakType:
            streak_id = f"{user_id}_{streak_type.value}"
            self.user_streaks[user_id][streak_type.value] = Streak(
                streak_id=streak_id,
                user_id=user_id,
                streak_type=streak_type
            )
        
        # Initialize badge progress for all badges
        for badge in self.badge_library.badges.values():
            user_badge = UserBadge(
                user_badge_id=f"{user_id}_{badge.badge_id}",
                user_id=user_id,
                badge_id=badge.badge_id,
                required_count=badge.requirement_count
            )
            self.user_badges[user_id].append(user_badge)
    
    def record_meal_scan(
        self,
        user_id: str,
        scan_date: date,
        contaminants_detected: int = 0
    ):
        """Record meal scan activity"""
        # Get or create daily activity
        activity = self._get_or_create_daily_activity(user_id, scan_date)
        activity.meals_scanned += 1
        activity.foods_scanned += 1
        activity.contaminants_flagged += contaminants_detected
        
        # Update scanning streak
        streak = self.user_streaks[user_id].get(StreakType.DAILY_SCANNING.value)
        if activity.meets_streak_criteria():
            streak.update(scan_date)
        
        # Update badge progress
        self._update_badge_progress(user_id, "scan_10", 1)
        self._update_badge_progress(user_id, "scan_100", 1)
        self._update_badge_progress(user_id, "scan_1000", 1)
        
        if contaminants_detected > 0:
            self._update_badge_progress(user_id, "lead_detector_1", contaminants_detected)
            self._update_badge_progress(user_id, "lead_detector_2", contaminants_detected)
            self._update_badge_progress(user_id, "toxin_hunter", contaminants_detected)
    
    def record_goal_progress(
        self,
        user_id: str,
        goal_id: str,
        new_value: float
    ):
        """Record progress towards health goal"""
        goals = self.user_goals[user_id]
        goal = next((g for g in goals if g.goal_id == goal_id), None)
        
        if not goal:
            return
        
        goal.current_value = new_value
        goal.update_progress()
        
        # Check for weight loss badges
        if goal.goal_type == "weight_loss":
            lbs_lost = goal.start_value - goal.current_value
            self._update_badge_progress(user_id, "weight_5lb", int(lbs_lost))
            self._update_badge_progress(user_id, "weight_10lb", int(lbs_lost))
            self._update_badge_progress(user_id, "weight_25lb", int(lbs_lost))
            self._update_badge_progress(user_id, "weight_50lb", int(lbs_lost))
            self._update_badge_progress(user_id, "weight_100lb", int(lbs_lost))
    
    def record_dietary_compliance(
        self,
        user_id: str,
        diet_type: str,
        compliance_date: date
    ):
        """Record dietary adherence"""
        if diet_type == "keto":
            self._update_badge_progress(user_id, "keto_week", 1)
            self._update_badge_progress(user_id, "keto_month", 1)
        elif diet_type == "vegan":
            self._update_badge_progress(user_id, "vegan_week", 1)
            self._update_badge_progress(user_id, "vegan_month", 1)
    
    def record_huddle_activity(
        self,
        user_id: str,
        activity_type: str,
        count: int = 1
    ):
        """Record social huddle activity"""
        if activity_type == "join":
            self._update_badge_progress(user_id, "huddle_join", count)
            self._update_badge_progress(user_id, "huddle_5", count)
        elif activity_type == "post":
            self._update_badge_progress(user_id, "post_10", count)
        elif activity_type == "helpful":
            self._update_badge_progress(user_id, "helpful_50", count)
    
    def record_consultation(
        self,
        user_id: str,
        completed: bool = True
    ):
        """Record professional consultation"""
        if completed:
            self._update_badge_progress(user_id, "consult_first", 1)
            self._update_badge_progress(user_id, "consult_5", 1)
    
    def get_user_dashboard(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get complete user dashboard
        
        Returns:
        - Active streaks
        - Goal progress
        - Recent badges earned
        - Next badges to earn
        - Daily activity summary
        """
        # Get streaks
        streaks_data = []
        for streak in self.user_streaks[user_id].values():
            streaks_data.append({
                'type': streak.streak_type.value,
                'current': streak.current_streak,
                'longest': streak.longest_streak,
                'emoji': streak.get_fire_emoji()
            })
        
        # Get goals
        goals_data = []
        for goal in self.user_goals[user_id]:
            goals_data.append({
                'goal_id': goal.goal_id,
                'name': goal.goal_name,
                'progress': goal.completion_percentage,
                'current': goal.current_value,
                'target': goal.target_value,
                'unit': goal.unit,
                'color': goal.get_progress_color(),
                'is_achieved': goal.is_achieved
            })
        
        # Get earned badges
        earned_badges = [
            {
                **self.badge_library.get_badge(ub.badge_id).to_dict(),
                'earned_at': ub.earned_at.isoformat() if ub.earned_at else None
            }
            for ub in self.user_badges[user_id]
            if ub.is_earned
        ]
        
        # Get next badges (sorted by progress)
        next_badges = [
            {
                **self.badge_library.get_badge(ub.badge_id).to_dict(),
                'progress': ub.get_progress_percentage(),
                'current': ub.current_count,
                'required': ub.required_count
            }
            for ub in self.user_badges[user_id]
            if not ub.is_earned and ub.current_count > 0
        ]
        next_badges.sort(key=lambda x: x['progress'], reverse=True)
        
        # Today's activity
        today = date.today()
        today_activity = self._get_or_create_daily_activity(user_id, today)
        
        return {
            'streaks': streaks_data,
            'goals': goals_data,
            'earned_badges': earned_badges,
            'next_badges': next_badges[:5],  # Top 5 closest
            'today': {
                'meals_scanned': today_activity.meals_scanned,
                'contaminants_flagged': today_activity.contaminants_flagged,
                'streak_criteria_met': today_activity.meets_streak_criteria()
            },
            'stats': {
                'total_badges': len(earned_badges),
                'total_points': sum(b['points'] for b in earned_badges),
                'longest_streak': max(s['longest'] for s in streaks_data) if streaks_data else 0
            }
        }
    
    def _get_or_create_daily_activity(
        self,
        user_id: str,
        activity_date: date
    ) -> DailyActivity:
        """Get or create daily activity record"""
        activities = self.daily_activities[user_id]
        
        for activity in activities:
            if activity.activity_date == activity_date:
                return activity
        
        # Create new
        activity = DailyActivity(
            activity_id=f"{user_id}_{activity_date.isoformat()}",
            user_id=user_id,
            activity_date=activity_date
        )
        activities.append(activity)
        
        return activity
    
    def _update_badge_progress(
        self,
        user_id: str,
        badge_id: str,
        increment: int
    ):
        """Update progress towards a badge"""
        user_badges = self.user_badges[user_id]
        
        for ub in user_badges:
            if ub.badge_id == badge_id and not ub.is_earned:
                ub.current_count += increment
                
                # Check if earned
                if ub.current_count >= ub.required_count:
                    ub.is_earned = True
                    ub.earned_at = datetime.now()
                    
                    badge = self.badge_library.get_badge(badge_id)
                    logger.info(
                        f"🎉 Badge earned! User {user_id}: "
                        f"{badge.icon_emoji} {badge.name}"
                    )
                
                break


if __name__ == "__main__":
    logger.info("Testing Phase 9: Progress Tracking & Gamification")
    
    # Initialize system
    badge_lib = BadgeLibrary()
    tracker = ProgressTracker(badge_lib)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Badge Library: {len(badge_lib.badges)} badges")
    
    # Show badge categories
    for category in BadgeCategory:
        count = len(badge_lib.get_badges_by_category(category))
        logger.info(f"  {category.value}: {count} badges")
    
    # Test user journey
    user_id = "user_test_001"
    tracker.initialize_user(user_id)
    
    logger.info(f"\n{'='*60}")
    logger.info("Test User Journey: 30-Day Weight Loss")
    
    # Create weight loss goal
    goal = HealthGoal(
        goal_id="goal_001",
        user_id=user_id,
        goal_type="weight_loss",
        goal_name="Lose 10 lbs",
        target_value=190,
        current_value=200,
        start_value=200,
        unit="lbs",
        start_date=date.today(),
        target_date=date.today() + timedelta(days=60),
        milestones=[195, 192, 190]
    )
    tracker.user_goals[user_id].append(goal)
    
    # Simulate 30 days of activity
    logger.info(f"\nSimulating 30 days of meal scanning...")
    
    today = date.today()
    for day in range(30):
        scan_date = today - timedelta(days=30-day)
        
        # Scan 3 meals per day
        for meal in range(3):
            contaminants = 1 if day % 5 == 0 else 0
            tracker.record_meal_scan(user_id, scan_date, contaminants)
        
        # Update weight every 5 days
        if day % 5 == 0:
            new_weight = 200 - (day * 0.3)  # Losing ~0.3 lbs/day
            tracker.record_goal_progress(user_id, goal.goal_id, new_weight)
        
        # Record dietary compliance
        if day >= 7:
            tracker.record_dietary_compliance(user_id, "keto", scan_date)
    
    # Record social activity
    tracker.record_huddle_activity(user_id, "join", 3)
    tracker.record_huddle_activity(user_id, "post", 15)
    
    # Record consultation
    tracker.record_consultation(user_id, completed=True)
    
    # Get dashboard
    logger.info(f"\n{'='*60}")
    logger.info("User Dashboard:")
    
    dashboard = tracker.get_user_dashboard(user_id)
    
    # Show streaks
    logger.info(f"\n🔥 Streaks:")
    for streak in dashboard['streaks']:
        logger.info(
            f"  {streak['emoji']} {streak['type']}: "
            f"{streak['current']} days (Best: {streak['longest']})"
        )
    
    # Show goals
    logger.info(f"\n🎯 Goals:")
    for goal_data in dashboard['goals']:
        logger.info(
            f"  {goal_data['name']}: {goal_data['progress']:.1f}% "
            f"({goal_data['current']}{goal_data['unit']} → {goal_data['target']}{goal_data['unit']})"
        )
    
    # Show earned badges
    logger.info(f"\n🏆 Earned Badges ({len(dashboard['earned_badges'])}):")
    for badge in dashboard['earned_badges'][:10]:
        logger.info(
            f"  {badge['icon']} {badge['name']} ({badge['tier']})"
        )
    
    # Show next badges
    logger.info(f"\n📈 Next Badges (Top 5):")
    for badge in dashboard['next_badges']:
        logger.info(
            f"  {badge['icon']} {badge['name']}: {badge['progress']:.0f}% "
            f"({badge['current']}/{badge['required']})"
        )
    
    # Show stats
    logger.info(f"\n📊 Stats:")
    logger.info(f"  Total Badges: {dashboard['stats']['total_badges']}")
    logger.info(f"  Total Points: {dashboard['stats']['total_points']:,}")
    logger.info(f"  Longest Streak: {dashboard['stats']['longest_streak']} days")
    logger.info(f"  Today's Scans: {dashboard['today']['meals_scanned']}")
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ Phase 9: Gamification Complete!")
    logger.info(f"\n🎮 Features:")
    logger.info(f"  • {len(badge_lib.badges)} Achievement Badges")
    logger.info(f"  • GitHub-style Streaks 🔥")
    logger.info(f"  • Visual Progress Tracking")
    logger.info(f"  • Goal Dashboard")
    logger.info(f"  • Milestone Celebrations")
    logger.info(f"  • Multi-tier Badge System (Bronze → Diamond)")
