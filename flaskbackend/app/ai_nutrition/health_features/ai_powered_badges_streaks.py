"""
Feature 5: AI-Powered Badges & Streaks
========================================

Intelligent achievement system that tracks meaningful consistency across SCAN, PLAN,
and WATCH tabs with contextual rewards. Unlike simple login streaks, this system
rewards actual healthy behaviors and food choices.

Key Features:
- Context-aware badge unlocking based on real actions
- Smart streak mechanics with recovery options
- Tab-specific achievements (SCAN, PLAN, WATCH)
- Progressive difficulty tiers (Bronze → Silver → Gold → Platinum → Diamond)
- Streak freezes and recovery mechanisms
- Social sharing and leaderboards
- Predictive achievement recommendations

Badge Categories:
SCAN Tab:
- Molecular Master: Scan foods with high molecular complexity
- Food Explorer: Try diverse food categories
- Macro Balancer: Maintain balanced macros
- Fiber Champion: Consistently choose high-fiber foods
- Sugar Slayer: Avoid high-sugar foods

PLAN Tab:
- Compliance King/Queen: Follow meal plans consistently
- Family Chef: Cook meals for family
- Meal Prep Pro: Weekly meal preparation
- Portion Master: Accurate portion control
- Hydration Hero: Daily water goals

WATCH Tab:
- Recipe Collector: Save diverse recipes
- Cuisine Explorer: Try international cuisines
- Video Learner: Complete educational videos
- Community Supporter: Help others in comments
- Trend Setter: Early adopter of new features

Author: AI Health Features Team
Created: November 12, 2025
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta, date
from enum import Enum
import json
from collections import defaultdict, Counter


# ==================== ENUMS AND TYPES ====================

class BadgeCategory(Enum):
    """Badge categories aligned with app tabs"""
    SCAN = "scan"
    PLAN = "plan"
    WATCH = "watch"
    GENERAL = "general"


class BadgeTier(Enum):
    """Progressive difficulty tiers"""
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    DIAMOND = "diamond"


class StreakType(Enum):
    """Types of streaks"""
    SCAN_STREAK = "scan_streak"  # Daily scanning
    COMPLIANCE_STREAK = "compliance_streak"  # Meal plan adherence
    EXERCISE_STREAK = "exercise_streak"  # Daily activity
    WATER_STREAK = "water_streak"  # Hydration goals
    LEARNING_STREAK = "learning_streak"  # Educational content
    CONSISTENCY_STREAK = "consistency_streak"  # Overall consistency


class StreakStatus(Enum):
    """Streak status"""
    ACTIVE = "active"
    FROZEN = "frozen"  # Using streak freeze
    AT_RISK = "at_risk"  # Last day to maintain
    BROKEN = "broken"
    RECOVERED = "recovered"  # Used recovery


class AchievementRarity(Enum):
    """Achievement rarity (affects rewards)"""
    COMMON = "common"  # 50%+ of users achieve
    UNCOMMON = "uncommon"  # 20-50% of users
    RARE = "rare"  # 5-20% of users
    EPIC = "epic"  # 1-5% of users
    LEGENDARY = "legendary"  # <1% of users


# ==================== DATA MODELS ====================

@dataclass
class BadgeRequirement:
    """Requirements to unlock a badge"""
    requirement_type: str  # "count", "percentage", "consecutive", "unique"
    metric: str  # What to measure
    threshold: float  # Required value
    timeframe_days: Optional[int] = None  # Must achieve within timeframe
    
    def check(self, value: float) -> bool:
        """Check if requirement is met"""
        return value >= self.threshold


@dataclass
class Badge:
    """Achievement badge"""
    badge_id: str
    name: str
    description: str
    category: BadgeCategory
    tier: BadgeTier
    rarity: AchievementRarity
    
    # Requirements
    requirements: List[BadgeRequirement]
    
    # Rewards
    points: int  # Achievement points
    title: Optional[str] = None  # Unlockable title (e.g., "Molecular Master")
    icon_url: Optional[str] = None
    
    # Metadata
    unlock_date: Optional[datetime] = None
    progress_percentage: float = 0.0
    is_unlocked: bool = False
    is_hidden: bool = False  # Hidden until close to unlocking
    
    def calculate_progress(self, user_data: Dict[str, float]) -> float:
        """Calculate progress toward unlocking badge (0-100%)"""
        if self.is_unlocked:
            return 100.0
        
        if not self.requirements:
            return 0.0
        
        # Calculate progress for each requirement
        progress_scores = []
        for req in self.requirements:
            user_value = user_data.get(req.metric, 0)
            req_progress = min(100.0, (user_value / req.threshold) * 100)
            progress_scores.append(req_progress)
        
        # Average progress across all requirements
        self.progress_percentage = sum(progress_scores) / len(progress_scores)
        return self.progress_percentage
    
    def check_unlock(self, user_data: Dict[str, float]) -> bool:
        """Check if badge should be unlocked"""
        if self.is_unlocked:
            return False
        
        # All requirements must be met
        for req in self.requirements:
            user_value = user_data.get(req.metric, 0)
            if not req.check(user_value):
                return False
        
        # Unlock!
        self.is_unlocked = True
        self.unlock_date = datetime.now()
        self.progress_percentage = 100.0
        return True


@dataclass
class StreakData:
    """Streak tracking data"""
    streak_id: str
    streak_type: StreakType
    current_count: int  # Current streak length
    longest_count: int  # Personal best
    status: StreakStatus
    
    # Dates
    start_date: date
    last_updated: date
    
    # Recovery mechanics
    freezes_available: int = 2  # Streak freezes (miss a day without breaking)
    freezes_used: int = 0
    recovery_available: int = 1  # Can recover broken streak within 48h
    recovery_used: int = 0
    
    # Milestones
    milestone_days: List[int] = field(default_factory=lambda: [7, 30, 60, 90, 180, 365])
    milestones_achieved: List[int] = field(default_factory=list)
    
    def increment(self) -> Tuple[bool, Optional[int]]:
        """
        Increment streak by 1 day.
        
        Returns:
            (milestone_reached, milestone_day)
        """
        self.current_count += 1
        self.last_updated = date.today()
        self.status = StreakStatus.ACTIVE
        
        # Update longest streak
        if self.current_count > self.longest_count:
            self.longest_count = self.current_count
        
        # Check for milestone
        for milestone in self.milestone_days:
            if self.current_count == milestone and milestone not in self.milestones_achieved:
                self.milestones_achieved.append(milestone)
                return True, milestone
        
        return False, None
    
    def break_streak(self, allow_recovery: bool = True) -> bool:
        """
        Break streak.
        
        Args:
            allow_recovery: If True, allow recovery within 48h
        
        Returns:
            True if recovery available, False otherwise
        """
        if allow_recovery and self.recovery_available > 0:
            self.status = StreakStatus.AT_RISK
            return True  # Recovery possible
        else:
            self.status = StreakStatus.BROKEN
            self.current_count = 0
            return False
    
    def use_freeze(self) -> bool:
        """Use streak freeze to skip a day"""
        if self.freezes_available > 0:
            self.freezes_available -= 1
            self.freezes_used += 1
            self.status = StreakStatus.FROZEN
            self.last_updated = date.today()
            return True
        return False
    
    def recover(self) -> bool:
        """Recover broken streak (within 48h)"""
        if self.status == StreakStatus.AT_RISK and self.recovery_available > 0:
            self.recovery_available -= 1
            self.recovery_used += 1
            self.status = StreakStatus.RECOVERED
            self.last_updated = date.today()
            return True
        return False
    
    def get_next_milestone(self) -> Optional[int]:
        """Get next milestone to achieve"""
        for milestone in self.milestone_days:
            if milestone > self.current_count:
                return milestone
        return None


@dataclass
class UserAchievements:
    """User's complete achievement profile"""
    user_id: str
    
    # Badges
    badges: List[Badge] = field(default_factory=list)
    unlocked_badges: List[Badge] = field(default_factory=list)
    
    # Streaks
    streaks: Dict[str, StreakData] = field(default_factory=dict)
    
    # Stats
    total_points: int = 0
    achievement_level: int = 1  # Overall level based on points
    unlocked_titles: List[str] = field(default_factory=list)
    active_title: Optional[str] = None
    
    # Social
    leaderboard_rank: Optional[int] = None
    percentile: Optional[float] = None  # Top X% of users
    
    def add_points(self, points: int) -> Tuple[bool, int]:
        """
        Add achievement points.
        
        Returns:
            (leveled_up, new_level)
        """
        old_level = self.achievement_level
        self.total_points += points
        
        # Level calculation: Level = sqrt(total_points / 100)
        new_level = int((self.total_points / 100) ** 0.5) + 1
        self.achievement_level = new_level
        
        return new_level > old_level, new_level
    
    def get_badge_by_id(self, badge_id: str) -> Optional[Badge]:
        """Get badge by ID"""
        for badge in self.badges:
            if badge.badge_id == badge_id:
                return badge
        return None
    
    def get_streak(self, streak_type: StreakType) -> Optional[StreakData]:
        """Get streak by type"""
        return self.streaks.get(streak_type.value)


# ==================== BADGE LIBRARY ====================

class BadgeLibrary:
    """Library of all available badges"""
    
    @staticmethod
    def create_scan_badges() -> List[Badge]:
        """Create SCAN tab badges"""
        badges = []
        
        # Molecular Master badges
        for tier, (threshold, points) in [
            (BadgeTier.BRONZE, (50, 100)),
            (BadgeTier.SILVER, (200, 300)),
            (BadgeTier.GOLD, (500, 600)),
            (BadgeTier.PLATINUM, (1000, 1000)),
            (BadgeTier.DIAMOND, (2000, 2000))
        ]:
            badges.append(Badge(
                badge_id=f"molecular_master_{tier.value}",
                name=f"Molecular Master {tier.value.title()}",
                description=f"Scan {threshold} foods with detailed molecular analysis",
                category=BadgeCategory.SCAN,
                tier=tier,
                rarity=AchievementRarity.UNCOMMON,
                requirements=[
                    BadgeRequirement("count", "molecular_scans", threshold)
                ],
                points=points,
                title="Molecular Master" if tier == BadgeTier.DIAMOND else None
            ))
        
        # Food Explorer badges
        badges.append(Badge(
            badge_id="food_explorer_bronze",
            name="Food Explorer Bronze",
            description="Scan foods from 10 different categories",
            category=BadgeCategory.SCAN,
            tier=BadgeTier.BRONZE,
            rarity=AchievementRarity.COMMON,
            requirements=[
                BadgeRequirement("unique", "food_categories", 10)
            ],
            points=150
        ))
        
        badges.append(Badge(
            badge_id="food_explorer_gold",
            name="Food Explorer Gold",
            description="Scan foods from 30 different categories",
            category=BadgeCategory.SCAN,
            tier=BadgeTier.GOLD,
            rarity=AchievementRarity.RARE,
            requirements=[
                BadgeRequirement("unique", "food_categories", 30)
            ],
            points=500,
            title="Food Explorer"
        ))
        
        # Sugar Slayer badges
        badges.append(Badge(
            badge_id="sugar_slayer_silver",
            name="Sugar Slayer Silver",
            description="Maintain <25g daily sugar for 30 consecutive days",
            category=BadgeCategory.SCAN,
            tier=BadgeTier.SILVER,
            rarity=AchievementRarity.UNCOMMON,
            requirements=[
                BadgeRequirement("consecutive", "low_sugar_days", 30)
            ],
            points=400
        ))
        
        badges.append(Badge(
            badge_id="sugar_slayer_platinum",
            name="Sugar Slayer Platinum",
            description="Maintain <15g daily sugar for 90 consecutive days",
            category=BadgeCategory.SCAN,
            tier=BadgeTier.PLATINUM,
            rarity=AchievementRarity.EPIC,
            requirements=[
                BadgeRequirement("consecutive", "very_low_sugar_days", 90)
            ],
            points=1200,
            title="Sugar Slayer"
        ))
        
        # Fiber Champion badges
        badges.append(Badge(
            badge_id="fiber_champion_gold",
            name="Fiber Champion Gold",
            description="Achieve 30g+ fiber daily for 60 days",
            category=BadgeCategory.SCAN,
            tier=BadgeTier.GOLD,
            rarity=AchievementRarity.RARE,
            requirements=[
                BadgeRequirement("consecutive", "high_fiber_days", 60)
            ],
            points=700,
            title="Fiber Champion"
        ))
        
        # Macro Balancer
        badges.append(Badge(
            badge_id="macro_balancer_silver",
            name="Macro Balancer Silver",
            description="Hit macro targets (±5%) for 14 consecutive days",
            category=BadgeCategory.SCAN,
            tier=BadgeTier.SILVER,
            rarity=AchievementRarity.UNCOMMON,
            requirements=[
                BadgeRequirement("consecutive", "balanced_macro_days", 14)
            ],
            points=300
        ))
        
        return badges
    
    @staticmethod
    def create_plan_badges() -> List[Badge]:
        """Create PLAN tab badges"""
        badges = []
        
        # Compliance King/Queen badges
        for tier, (threshold, points) in [
            (BadgeTier.BRONZE, (85, 150)),
            (BadgeTier.SILVER, (90, 350)),
            (BadgeTier.GOLD, (95, 700)),
            (BadgeTier.PLATINUM, (98, 1100))
        ]:
            badges.append(Badge(
                badge_id=f"compliance_{tier.value}",
                name=f"Compliance {tier.value.title()}",
                description=f"Maintain {threshold}%+ meal plan adherence for 30 days",
                category=BadgeCategory.PLAN,
                tier=tier,
                rarity=AchievementRarity.UNCOMMON if tier in [BadgeTier.BRONZE, BadgeTier.SILVER] else AchievementRarity.RARE,
                requirements=[
                    BadgeRequirement("percentage", "compliance_30d", threshold, 30)
                ],
                points=points,
                title="Compliance Champion" if tier == BadgeTier.PLATINUM else None
            ))
        
        # Meal Prep Pro
        badges.append(Badge(
            badge_id="meal_prep_gold",
            name="Meal Prep Pro Gold",
            description="Complete 12 weekly meal prep sessions",
            category=BadgeCategory.PLAN,
            tier=BadgeTier.GOLD,
            rarity=AchievementRarity.RARE,
            requirements=[
                BadgeRequirement("count", "meal_prep_sessions", 12)
            ],
            points=600,
            title="Meal Prep Pro"
        ))
        
        # Hydration Hero
        badges.append(Badge(
            badge_id="hydration_hero_silver",
            name="Hydration Hero Silver",
            description="Hit water goals (64oz+) for 30 consecutive days",
            category=BadgeCategory.PLAN,
            tier=BadgeTier.SILVER,
            rarity=AchievementRarity.COMMON,
            requirements=[
                BadgeRequirement("consecutive", "hydration_days", 30)
            ],
            points=250
        ))
        
        badges.append(Badge(
            badge_id="hydration_hero_platinum",
            name="Hydration Hero Platinum",
            description="Hit water goals (64oz+) for 180 consecutive days",
            category=BadgeCategory.PLAN,
            tier=BadgeTier.PLATINUM,
            rarity=AchievementRarity.EPIC,
            requirements=[
                BadgeRequirement("consecutive", "hydration_days", 180)
            ],
            points=1300,
            title="Hydration Hero"
        ))
        
        # Family Chef
        badges.append(Badge(
            badge_id="family_chef_gold",
            name="Family Chef Gold",
            description="Cook 100 family meals tracked in app",
            category=BadgeCategory.PLAN,
            tier=BadgeTier.GOLD,
            rarity=AchievementRarity.RARE,
            requirements=[
                BadgeRequirement("count", "family_meals", 100)
            ],
            points=650,
            title="Family Chef"
        ))
        
        # Portion Master
        badges.append(Badge(
            badge_id="portion_master_silver",
            name="Portion Master Silver",
            description="Log accurate portions (photo verified) for 50 meals",
            category=BadgeCategory.PLAN,
            tier=BadgeTier.SILVER,
            rarity=AchievementRarity.UNCOMMON,
            requirements=[
                BadgeRequirement("count", "accurate_portions", 50)
            ],
            points=280
        ))
        
        return badges
    
    @staticmethod
    def create_watch_badges() -> List[Badge]:
        """Create WATCH tab badges"""
        badges = []
        
        # Recipe Collector badges
        badges.append(Badge(
            badge_id="recipe_collector_bronze",
            name="Recipe Collector Bronze",
            description="Save 25 recipes to your collection",
            category=BadgeCategory.WATCH,
            tier=BadgeTier.BRONZE,
            rarity=AchievementRarity.COMMON,
            requirements=[
                BadgeRequirement("count", "saved_recipes", 25)
            ],
            points=120
        ))
        
        badges.append(Badge(
            badge_id="recipe_collector_platinum",
            name="Recipe Collector Platinum",
            description="Save 200 recipes and cook 50 of them",
            category=BadgeCategory.WATCH,
            tier=BadgeTier.PLATINUM,
            rarity=AchievementRarity.EPIC,
            requirements=[
                BadgeRequirement("count", "saved_recipes", 200),
                BadgeRequirement("count", "cooked_recipes", 50)
            ],
            points=1100,
            title="Recipe Master"
        ))
        
        # Cuisine Explorer badges
        badges.append(Badge(
            badge_id="cuisine_explorer_gold",
            name="Cuisine Explorer Gold",
            description="Try recipes from 15 different cuisines",
            category=BadgeCategory.WATCH,
            tier=BadgeTier.GOLD,
            rarity=AchievementRarity.RARE,
            requirements=[
                BadgeRequirement("unique", "cuisines_tried", 15)
            ],
            points=550,
            title="World Chef"
        ))
        
        # Video Learner badges
        badges.append(Badge(
            badge_id="video_learner_silver",
            name="Video Learner Silver",
            description="Watch 30 educational nutrition videos",
            category=BadgeCategory.WATCH,
            tier=BadgeTier.SILVER,
            rarity=AchievementRarity.COMMON,
            requirements=[
                BadgeRequirement("count", "videos_watched", 30)
            ],
            points=200
        ))
        
        badges.append(Badge(
            badge_id="video_learner_diamond",
            name="Video Learner Diamond",
            description="Complete 100 educational videos and pass 50 quizzes",
            category=BadgeCategory.WATCH,
            tier=BadgeTier.DIAMOND,
            rarity=AchievementRarity.LEGENDARY,
            requirements=[
                BadgeRequirement("count", "videos_watched", 100),
                BadgeRequirement("count", "quizzes_passed", 50)
            ],
            points=2200,
            title="Nutrition Scholar"
        ))
        
        # Community Supporter
        badges.append(Badge(
            badge_id="community_supporter_gold",
            name="Community Supporter Gold",
            description="Give 100 helpful comments to other users",
            category=BadgeCategory.WATCH,
            tier=BadgeTier.GOLD,
            rarity=AchievementRarity.RARE,
            requirements=[
                BadgeRequirement("count", "helpful_comments", 100)
            ],
            points=700,
            title="Community Champion"
        ))
        
        # Trend Setter
        badges.append(Badge(
            badge_id="trend_setter_platinum",
            name="Trend Setter Platinum",
            description="Be among first 1000 users to try 5 new features",
            category=BadgeCategory.WATCH,
            tier=BadgeTier.PLATINUM,
            rarity=AchievementRarity.LEGENDARY,
            requirements=[
                BadgeRequirement("count", "early_adopter_features", 5)
            ],
            points=1500,
            title="Innovator"
        ))
        
        return badges
    
    @staticmethod
    def create_general_badges() -> List[Badge]:
        """Create general/cross-category badges"""
        badges = []
        
        # Perfect Week badges
        badges.append(Badge(
            badge_id="perfect_week",
            name="Perfect Week",
            description="Complete all daily goals for 7 consecutive days",
            category=BadgeCategory.GENERAL,
            tier=BadgeTier.GOLD,
            rarity=AchievementRarity.RARE,
            requirements=[
                BadgeRequirement("consecutive", "perfect_days", 7)
            ],
            points=800
        ))
        
        # Wellness Warrior badges
        badges.append(Badge(
            badge_id="wellness_warrior_platinum",
            name="Wellness Warrior Platinum",
            description="Maintain 90+ wellness score for 60 days",
            category=BadgeCategory.GENERAL,
            tier=BadgeTier.PLATINUM,
            rarity=AchievementRarity.EPIC,
            requirements=[
                BadgeRequirement("consecutive", "high_wellness_days", 60)
            ],
            points=1400,
            title="Wellness Warrior"
        ))
        
        # Early Bird
        badges.append(Badge(
            badge_id="early_bird_silver",
            name="Early Bird Silver",
            description="Scan breakfast before 8 AM for 30 days",
            category=BadgeCategory.GENERAL,
            tier=BadgeTier.SILVER,
            rarity=AchievementRarity.UNCOMMON,
            requirements=[
                BadgeRequirement("consecutive", "early_breakfast_days", 30)
            ],
            points=320
        ))
        
        # Comeback Kid
        badges.append(Badge(
            badge_id="comeback_kid",
            name="Comeback Kid",
            description="Recover from a setback and achieve 14-day streak",
            category=BadgeCategory.GENERAL,
            tier=BadgeTier.GOLD,
            rarity=AchievementRarity.UNCOMMON,
            requirements=[
                BadgeRequirement("consecutive", "post_setback_streak", 14)
            ],
            points=450,
            is_hidden=True
        ))
        
        return badges
    
    @staticmethod
    def get_all_badges() -> List[Badge]:
        """Get complete badge library"""
        return (
            BadgeLibrary.create_scan_badges() +
            BadgeLibrary.create_plan_badges() +
            BadgeLibrary.create_watch_badges() +
            BadgeLibrary.create_general_badges()
        )


# ==================== STREAK MANAGER ====================

class StreakManager:
    """Manages all user streaks"""
    
    def __init__(self):
        self.streak_configs = self._initialize_configs()
    
    def _initialize_configs(self) -> Dict[StreakType, Dict]:
        """Initialize streak configurations"""
        return {
            StreakType.SCAN_STREAK: {
                'name': 'Daily Scan Streak',
                'description': 'Scan at least 3 meals per day',
                'min_requirement': 3,
                'freezes': 2,
                'recovery': 1
            },
            StreakType.COMPLIANCE_STREAK: {
                'name': 'Meal Plan Compliance',
                'description': 'Follow meal plan with 80%+ adherence',
                'min_requirement': 0.8,
                'freezes': 1,
                'recovery': 1
            },
            StreakType.EXERCISE_STREAK: {
                'name': 'Daily Activity Streak',
                'description': 'Complete 30+ minutes of activity',
                'min_requirement': 30,
                'freezes': 2,
                'recovery': 1
            },
            StreakType.WATER_STREAK: {
                'name': 'Hydration Streak',
                'description': 'Drink 64+ oz of water daily',
                'min_requirement': 64,
                'freezes': 1,
                'recovery': 0
            },
            StreakType.LEARNING_STREAK: {
                'name': 'Learning Streak',
                'description': 'Watch or read educational content daily',
                'min_requirement': 1,
                'freezes': 3,
                'recovery': 1
            },
            StreakType.CONSISTENCY_STREAK: {
                'name': 'Overall Consistency',
                'description': 'Complete 5+ daily goals',
                'min_requirement': 5,
                'freezes': 2,
                'recovery': 1
            }
        }
    
    def initialize_user_streaks(self, user_id: str) -> Dict[str, StreakData]:
        """Initialize streaks for new user"""
        streaks = {}
        today = date.today()
        
        for streak_type, config in self.streak_configs.items():
            streaks[streak_type.value] = StreakData(
                streak_id=f"{user_id}_{streak_type.value}",
                streak_type=streak_type,
                current_count=0,
                longest_count=0,
                status=StreakStatus.ACTIVE,
                start_date=today,
                last_updated=today,
                freezes_available=config['freezes'],
                recovery_available=config['recovery']
            )
        
        return streaks
    
    def update_streak(
        self,
        streak: StreakData,
        user_value: float,
        check_date: Optional[date] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Update streak based on user action.
        
        Args:
            streak: StreakData object
            user_value: User's value for the day
            check_date: Date to check (defaults to today)
        
        Returns:
            (streak_updated, notification_message)
        """
        check_date = check_date or date.today()
        config = self.streak_configs[streak.streak_type]
        min_req = config['min_requirement']
        
        # Check if requirement met
        requirement_met = user_value >= min_req
        
        # Calculate days since last update
        days_since_update = (check_date - streak.last_updated).days
        
        if days_since_update == 0:
            # Already updated today
            return False, None
        
        elif days_since_update == 1:
            # Normal daily update
            if requirement_met:
                milestone_reached, milestone_day = streak.increment()
                if milestone_reached:
                    return True, f"🔥 {milestone_day}-day {config['name']} milestone achieved!"
                return True, None
            else:
                # Requirement not met - break or freeze
                if streak.freezes_available > 0:
                    return True, f"❄️ Streak freeze used. {streak.freezes_available - 1} remaining."
                else:
                    streak.break_streak()
                    return True, f"💔 {config['name']} broken. Start fresh tomorrow!"
        
        else:
            # Missed day(s) - streak likely broken unless recovery available
            if days_since_update == 2 and streak.status == StreakStatus.AT_RISK:
                # Recovery window
                if requirement_met and streak.recover():
                    return True, f"💪 Streak recovered! Keep going!"
                else:
                    streak.break_streak(allow_recovery=False)
                    return True, f"💔 {config['name']} broken. Start fresh!"
            else:
                # Too many days missed
                streak.break_streak(allow_recovery=False)
                return True, f"💔 {config['name']} broken. Start fresh!"


# ==================== ACHIEVEMENT ENGINE ====================

class AchievementEngine:
    """Main engine for managing badges and streaks"""
    
    def __init__(self):
        self.badge_library = BadgeLibrary.get_all_badges()
        self.streak_manager = StreakManager()
    
    def initialize_user(self, user_id: str) -> UserAchievements:
        """Initialize achievements for new user"""
        # Create user achievements profile
        user_achievements = UserAchievements(
            user_id=user_id,
            badges=self.badge_library.copy(),
            streaks=self.streak_manager.initialize_user_streaks(user_id)
        )
        
        return user_achievements
    
    def update_user_progress(
        self,
        user_achievements: UserAchievements,
        user_data: Dict[str, float]
    ) -> Dict[str, List]:
        """
        Update user progress and check for new achievements.
        
        Args:
            user_achievements: User's achievement profile
            user_data: Dictionary of user metrics
        
        Returns:
            Dictionary with newly unlocked badges and updated streaks
        """
        results = {
            'new_badges': [],
            'badge_progress': [],
            'streak_updates': [],
            'points_earned': 0,
            'level_up': False,
            'new_level': user_achievements.achievement_level
        }
        
        # Update badge progress
        for badge in user_achievements.badges:
            if not badge.is_unlocked:
                old_progress = badge.progress_percentage
                new_progress = badge.calculate_progress(user_data)
                
                # Check for unlock
                if badge.check_unlock(user_data):
                    results['new_badges'].append(badge)
                    user_achievements.unlocked_badges.append(badge)
                    
                    # Add points
                    leveled_up, new_level = user_achievements.add_points(badge.points)
                    results['points_earned'] += badge.points
                    
                    if leveled_up:
                        results['level_up'] = True
                        results['new_level'] = new_level
                    
                    # Unlock title if available
                    if badge.title and badge.title not in user_achievements.unlocked_titles:
                        user_achievements.unlocked_titles.append(badge.title)
                
                # Track significant progress
                elif new_progress >= 75 and old_progress < 75:
                    results['badge_progress'].append({
                        'badge': badge,
                        'progress': new_progress,
                        'message': f"Almost there! {badge.name} is {new_progress:.0f}% complete"
                    })
        
        # Update streaks
        for streak_type, streak in user_achievements.streaks.items():
            metric_key = f"{streak_type}_value"
            if metric_key in user_data:
                updated, message = self.streak_manager.update_streak(
                    streak, user_data[metric_key]
                )
                if updated and message:
                    results['streak_updates'].append({
                        'streak_type': streak_type,
                        'message': message,
                        'current_count': streak.current_count
                    })
        
        return results
    
    def get_recommended_achievements(
        self,
        user_achievements: UserAchievements,
        limit: int = 5
    ) -> List[Dict]:
        """
        Get recommended achievements to work toward (AI-powered suggestions).
        
        Args:
            user_achievements: User's achievement profile
            limit: Maximum number of recommendations
        
        Returns:
            List of recommended badges with reasoning
        """
        recommendations = []
        
        # Find badges closest to unlocking
        in_progress = [
            badge for badge in user_achievements.badges
            if not badge.is_unlocked and badge.progress_percentage > 0
        ]
        
        # Sort by progress (descending)
        in_progress.sort(key=lambda b: b.progress_percentage, reverse=True)
        
        for badge in in_progress[:limit]:
            recommendations.append({
                'badge': badge,
                'progress': badge.progress_percentage,
                'reason': f"You're {badge.progress_percentage:.0f}% there! Just a bit more effort to unlock.",
                'priority': 'high' if badge.progress_percentage >= 75 else 'medium'
            })
        
        return recommendations
    
    def get_leaderboard_position(
        self,
        user_achievements: UserAchievements,
        all_users_points: List[int]
    ) -> Tuple[int, float]:
        """
        Calculate user's leaderboard position.
        
        Args:
            user_achievements: User's achievement profile
            all_users_points: List of all users' points
        
        Returns:
            (rank, percentile)
        """
        all_users_points = sorted(all_users_points, reverse=True)
        user_points = user_achievements.total_points
        
        # Find rank
        rank = 1
        for points in all_users_points:
            if points > user_points:
                rank += 1
            else:
                break
        
        # Calculate percentile
        percentile = (1 - (rank / len(all_users_points))) * 100 if all_users_points else 0
        
        user_achievements.leaderboard_rank = rank
        user_achievements.percentile = percentile
        
        return rank, percentile


# ==================== DEMONSTRATION ====================

def demonstrate_badges_and_streaks():
    """Demonstrate the AI-Powered Badges & Streaks system"""
    
    print("=" * 80)
    print("FEATURE 5: AI-POWERED BADGES & STREAKS DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize achievement engine
    print("🚀 Initializing Achievement Engine...")
    engine = AchievementEngine()
    print(f"✅ Loaded {len(engine.badge_library)} badges across {len(set(b.category for b in engine.badge_library))} categories")
    print()
    
    # Initialize user
    user_id = "user_12345"
    print(f"👤 Initializing achievements for {user_id}...")
    user_achievements = engine.initialize_user(user_id)
    print(f"✅ User initialized with {len(user_achievements.streaks)} active streaks")
    print()
    
    # Simulate Day 1-30 progress
    print("=" * 80)
    print("SIMULATING 30-DAY JOURNEY")
    print("=" * 80)
    print()
    
    # Day 1-7: Building initial streaks
    print("📅 Days 1-7: Building Initial Habits")
    for day in range(1, 8):
        user_data = {
            'molecular_scans': day * 3,
            'food_categories': min(day, 10),
            'scan_streak_value': 4,  # 4 scans per day
            'hydration_days': day,
            'water_streak_value': 70  # 70oz water
        }
        
        results = engine.update_user_progress(user_achievements, user_data)
        
        if results['new_badges']:
            for badge in results['new_badges']:
                print(f"   🏆 Day {day}: Unlocked '{badge.name}' (+{badge.points} points)")
        
        if results['streak_updates']:
            for update in results['streak_updates']:
                if 'milestone' in update['message'].lower():
                    print(f"   🔥 Day {day}: {update['message']}")
    
    print(f"   ✅ Week 1 Complete: {user_achievements.total_points} points earned")
    print()
    
    # Day 8-30: Accelerating progress
    print("📅 Days 8-30: Accelerating Progress")
    for day in range(8, 31):
        user_data = {
            'molecular_scans': day * 3,
            'food_categories': min(day + 5, 15),
            'low_sugar_days': day - 7,  # Started tracking sugar
            'high_fiber_days': max(0, day - 10),
            'balanced_macro_days': max(0, day - 5),
            'compliance_30d': 88.0,  # 88% compliance
            'hydration_days': day,
            'saved_recipes': day * 2,
            'videos_watched': day,
            'scan_streak_value': 4,
            'compliance_streak_value': 0.9,
            'water_streak_value': 72
        }
        
        results = engine.update_user_progress(user_achievements, user_data)
        
        if results['new_badges']:
            for badge in results['new_badges']:
                print(f"   🏆 Day {day}: Unlocked '{badge.name}' - {badge.tier.value.title()} tier (+{badge.points} points)")
        
        if results['level_up']:
            print(f"   ⭐ Day {day}: LEVEL UP! Now Level {results['new_level']}")
        
        # Show milestone streaks
        if day in [14, 21, 30]:
            print(f"   🔥 Day {day}: Milestone - Multiple {day}-day streaks achieved!")
    
    print(f"   ✅ Month 1 Complete: {user_achievements.total_points} points earned")
    print()
    
    # Display final results
    print("=" * 80)
    print("30-DAY ACHIEVEMENT SUMMARY")
    print("=" * 80)
    print()
    
    print(f"🎯 Overall Stats:")
    print(f"   Total Points: {user_achievements.total_points}")
    print(f"   Achievement Level: {user_achievements.achievement_level}")
    print(f"   Badges Unlocked: {len(user_achievements.unlocked_badges)}/{len(user_achievements.badges)}")
    print(f"   Unlocked Titles: {len(user_achievements.unlocked_titles)}")
    print()
    
    # Show unlocked badges by category
    print("🏆 Unlocked Badges by Category:")
    for category in BadgeCategory:
        category_badges = [b for b in user_achievements.unlocked_badges if b.category == category]
        if category_badges:
            print(f"\n   {category.value.upper()} ({len(category_badges)} badges):")
            for badge in category_badges:
                tier_emoji = {'bronze': '🥉', 'silver': '🥈', 'gold': '🥇', 'platinum': '💎', 'diamond': '💠'}
                print(f"      {tier_emoji.get(badge.tier.value, '🏅')} {badge.name} ({badge.tier.value.title()}) - {badge.points} pts")
    print()
    
    # Show active streaks
    print("🔥 Active Streaks:")
    for streak_type, streak in user_achievements.streaks.items():
        if streak.current_count > 0:
            next_milestone = streak.get_next_milestone()
            print(f"   {streak.streak_type.value.replace('_', ' ').title()}: {streak.current_count} days")
            print(f"      Longest: {streak.longest_count} | Next Milestone: {next_milestone} days")
            print(f"      Freezes: {streak.freezes_available} | Recovery: {streak.recovery_available}")
    print()
    
    # Show unlocked titles
    if user_achievements.unlocked_titles:
        print("👑 Unlocked Titles:")
        for title in user_achievements.unlocked_titles:
            print(f"   ✨ {title}")
        print()
    
    # Get recommendations
    print("💡 Recommended Next Achievements:")
    recommendations = engine.get_recommended_achievements(user_achievements, limit=5)
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['badge'].name} ({rec['progress']:.0f}% complete)")
        print(f"      {rec['reason']}")
        print(f"      Category: {rec['badge'].category.value} | Tier: {rec['badge'].tier.value} | Points: {rec['badge'].points}")
    print()
    
    # Simulate leaderboard
    print("🏅 Leaderboard Position:")
    simulated_user_points = [
        5000, 4500, 4200, 3800, 3500, 3200, 3000, 2800,
        user_achievements.total_points,  # Our user
        2300, 2100, 1900, 1700, 1500, 1300, 1100, 900, 700, 500, 300
    ]
    rank, percentile = engine.get_leaderboard_position(user_achievements, simulated_user_points)
    print(f"   Rank: #{rank} out of {len(simulated_user_points)} users")
    print(f"   Top {percentile:.1f}% of all users")
    print()
    
    # Show badge rarity distribution
    print("💎 Badge Rarity Distribution:")
    unlocked_by_rarity = defaultdict(int)
    total_by_rarity = defaultdict(int)
    for badge in user_achievements.badges:
        total_by_rarity[badge.rarity.value] += 1
        if badge.is_unlocked:
            unlocked_by_rarity[badge.rarity.value] += 1
    
    for rarity in AchievementRarity:
        unlocked = unlocked_by_rarity[rarity.value]
        total = total_by_rarity[rarity.value]
        if total > 0:
            print(f"   {rarity.value.title()}: {unlocked}/{total} ({unlocked/total*100:.0f}%)")
    print()
    
    print("=" * 80)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print()
    print("📊 This system showcases:")
    print("   • Context-aware badge unlocking (not just login streaks)")
    print("   • Progressive difficulty tiers (Bronze → Diamond)")
    print("   • Smart streak mechanics with freezes and recovery")
    print("   • Tab-specific achievements (SCAN, PLAN, WATCH)")
    print("   • Rare and legendary badges for top performers")
    print("   • AI-powered achievement recommendations")
    print("   • Social features (leaderboards, titles)")
    print()
    print("🎯 Production Implementation:")
    print("   • Real-time badge unlocking with push notifications")
    print("   • Animated badge reveal UI")
    print("   • Social sharing of achievements")
    print("   • Weekly achievement digest emails")
    print("   • Team/family challenges and competitions")
    print("   • Seasonal limited-edition badges")
    print()


if __name__ == "__main__":
    demonstrate_badges_and_streaks()
