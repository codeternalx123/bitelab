"""
Feature 9: AI-Matched Accountability Squad
==========================================

Intelligent matchmaking system that creates accountability groups (squads) of 3-6 members
with similar health goals, conditions, and lifestyles for mutual support and motivation.

Key Features:
- Multi-dimensional matching algorithm (age, goals, preferences, conditions, timezone)
- Group dynamics optimization (personality fit, activity levels)
- Shared challenges and competitions
- Group chat integration
- Progress tracking and leaderboards
- Conflict resolution and group health monitoring
- Squad achievements and badges

Matching Dimensions:
- Demographics: Age (±5 years), gender preference, location/timezone
- Health: Conditions, goals, current metrics, journey stage
- Lifestyle: Activity level, dietary preferences, schedule
- Personality: Communication style, motivation type, competitiveness
- Experience: App usage, streak length, engagement level

Author: AI Health Features Team
Created: November 12, 2025
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from datetime import datetime, timedelta, time
from enum import Enum
import random
import math
from collections import defaultdict


# ==================== ENUMS AND TYPES ====================

class MatchingCriteria(Enum):
    """Matching importance levels"""
    CRITICAL = "critical"  # Must match (e.g., language)
    IMPORTANT = "important"  # Strong preference (e.g., condition)
    PREFERRED = "preferred"  # Nice to have (e.g., age range)
    OPTIONAL = "optional"  # Low priority (e.g., hobbies)


class SquadSize(Enum):
    """Squad size options"""
    TRIO = 3  # Intimate group
    QUAD = 4  # Balanced
    QUINTET = 5  # Dynamic
    SQUAD = 6  # Full team


class SquadStatus(Enum):
    """Squad lifecycle states"""
    FORMING = "forming"  # Accepting members
    ACTIVE = "active"  # Fully operational
    DORMANT = "dormant"  # Low activity
    DISBANDED = "disbanded"  # Permanently closed


class ChallengeType(Enum):
    """Squad challenge types"""
    SCAN_STREAK = "scan_streak"  # Consecutive scan days
    COMPLIANCE = "compliance"  # Meal plan adherence
    WEIGHT_LOSS = "weight_loss"  # Combined weight loss
    STEPS = "steps"  # Total steps
    LEARNING = "learning"  # Educational content
    CONSISTENCY = "consistency"  # Daily goals hit


class CommunicationStyle(Enum):
    """User communication preferences"""
    CHATTY = "chatty"  # Frequent messages
    MODERATE = "moderate"  # Regular check-ins
    MINIMAL = "minimal"  # Occasional updates


class MotivationType(Enum):
    """Motivation style preferences"""
    COMPETITIVE = "competitive"  # Leaderboards, challenges
    COLLABORATIVE = "collaborative"  # Team goals
    SUPPORTIVE = "supportive"  # Emotional support
    EDUCATIONAL = "educational"  # Learning together


# ==================== DATA MODELS ====================

@dataclass
class UserProfile:
    """Complete user profile for matching"""
    user_id: str
    
    # Demographics
    age: int
    gender: Optional[str] = None
    timezone: str = "UTC"
    language: str = "en"
    
    # Health profile
    conditions: List[str] = field(default_factory=list)  # ["Type 2 Diabetes", "Hypertension"]
    goals: List[str] = field(default_factory=list)  # ["Weight Loss", "Medication Reduction"]
    current_weight_kg: Optional[float] = None
    target_weight_kg: Optional[float] = None
    journey_days: int = 0
    
    # Lifestyle
    activity_level: str = "moderate"  # sedentary, light, moderate, active, very_active
    dietary_preferences: List[str] = field(default_factory=list)  # ["vegetarian", "gluten_free"]
    typical_schedule: str = "9to5"  # 9to5, shift_work, flexible, night_owl
    
    # Personality
    communication_style: CommunicationStyle = CommunicationStyle.MODERATE
    motivation_type: MotivationType = MotivationType.COLLABORATIVE
    competitiveness: float = 0.5  # 0-1 scale
    
    # App engagement
    days_active: int = 0
    current_streak: int = 0
    avg_scans_per_day: float = 0.0
    engagement_score: float = 0.0  # 0-1, composite metric
    
    # Squad preferences
    preferred_squad_size: SquadSize = SquadSize.QUAD
    max_squads: int = 2  # Can be in multiple squads
    current_squad_count: int = 0


@dataclass
class MatchingScore:
    """Detailed matching score breakdown"""
    total_score: float  # 0-100
    
    # Component scores
    demographic_score: float = 0.0
    health_similarity: float = 0.0
    lifestyle_fit: float = 0.0
    personality_compatibility: float = 0.0
    engagement_alignment: float = 0.0
    
    # Weights applied
    demographic_weight: float = 0.15
    health_weight: float = 0.35
    lifestyle_weight: float = 0.20
    personality_weight: float = 0.20
    engagement_weight: float = 0.10
    
    # Match quality
    is_excellent_match: bool = False  # 85+
    is_good_match: bool = False  # 70-84
    is_acceptable_match: bool = False  # 60-69
    
    # Compatibility notes
    shared_conditions: List[str] = field(default_factory=list)
    shared_goals: List[str] = field(default_factory=list)
    potential_issues: List[str] = field(default_factory=list)


@dataclass
class SquadMember:
    """Squad member with role and activity"""
    user_id: str
    display_name: str
    avatar: str
    
    # Squad role
    is_creator: bool = False
    is_moderator: bool = False
    joined_at: datetime = field(default_factory=datetime.now)
    
    # Activity tracking
    messages_sent: int = 0
    challenges_completed: int = 0
    support_given: int = 0  # Reactions, encouragements
    last_active: datetime = field(default_factory=datetime.now)
    
    # Performance
    current_streak: int = 0
    points_earned: int = 0
    achievements_unlocked: int = 0


@dataclass
class SquadChallenge:
    """Squad challenge definition"""
    challenge_id: str
    challenge_type: ChallengeType
    name: str
    description: str
    
    # Duration
    start_date: datetime
    end_date: datetime
    duration_days: int
    
    # Goals
    individual_target: Optional[float] = None  # Per member
    team_target: Optional[float] = None  # Combined
    
    # Progress
    participants: List[str] = field(default_factory=list)  # user_ids
    individual_progress: Dict[str, float] = field(default_factory=dict)
    team_progress: float = 0.0
    
    # Rewards
    points_reward: int = 0
    badge_unlocked: Optional[str] = None
    
    # Status
    is_completed: bool = False
    completion_rate: float = 0.0  # 0-1


@dataclass
class SquadStats:
    """Squad aggregate statistics"""
    total_members: int
    active_members_7d: int
    
    # Engagement
    total_messages: int
    messages_per_day: float
    avg_response_time_hours: float
    
    # Performance
    avg_member_streak: float
    total_scans: int
    total_weight_loss_kg: float
    combined_points: int
    
    # Challenges
    active_challenges: int
    completed_challenges: int
    challenge_completion_rate: float
    
    # Health
    squad_health_score: float  # 0-100
    activity_trend: str  # "increasing", "stable", "declining"
    risk_of_disbanding: float  # 0-1


@dataclass
class AccountabilitySquad:
    """Complete squad data"""
    squad_id: str
    name: str
    description: str
    
    # Members
    members: List[SquadMember]
    max_size: int
    
    # Status
    status: SquadStatus
    created_at: datetime
    
    # Matching
    matching_criteria: Dict[str, Any]
    avg_compatibility_score: float
    
    # Challenges
    active_challenges: List[SquadChallenge]
    completed_challenges: List[SquadChallenge]
    
    # Stats
    stats: SquadStats
    
    # Squad culture
    communication_level: str = "moderate"  # low, moderate, high
    competitiveness_level: float = 0.5  # 0-1
    support_level: float = 0.0  # 0-1 based on interactions


# ==================== MATCHING ALGORITHM ====================

class MatchingAlgorithm:
    """AI-powered squad matching"""
    
    def __init__(self):
        # Matching weights (sum to 1.0)
        self.demographic_weight = 0.15
        self.health_weight = 0.35
        self.lifestyle_weight = 0.20
        self.personality_weight = 0.20
        self.engagement_weight = 0.10
    
    def calculate_match_score(
        self,
        user1: UserProfile,
        user2: UserProfile
    ) -> MatchingScore:
        """Calculate compatibility between two users"""
        # Component scores
        demo_score = self._calculate_demographic_score(user1, user2)
        health_score = self._calculate_health_similarity(user1, user2)
        lifestyle_score = self._calculate_lifestyle_fit(user1, user2)
        personality_score = self._calculate_personality_compatibility(user1, user2)
        engagement_score = self._calculate_engagement_alignment(user1, user2)
        
        # Weighted total
        total = (
            demo_score * self.demographic_weight +
            health_score * self.health_weight +
            lifestyle_score * self.lifestyle_weight +
            personality_score * self.personality_weight +
            engagement_score * self.engagement_weight
        )
        
        # Identify shared elements
        shared_conditions = list(set(user1.conditions) & set(user2.conditions))
        shared_goals = list(set(user1.goals) & set(user2.goals))
        
        # Check for potential issues
        issues = []
        if abs(user1.age - user2.age) > 20:
            issues.append("large_age_gap")
        if user1.communication_style == CommunicationStyle.CHATTY and user2.communication_style == CommunicationStyle.MINIMAL:
            issues.append("communication_mismatch")
        if abs(user1.competitiveness - user2.competitiveness) > 0.6:
            issues.append("competitiveness_mismatch")
        
        # Classify match quality
        is_excellent = total >= 85
        is_good = 70 <= total < 85
        is_acceptable = 60 <= total < 70
        
        return MatchingScore(
            total_score=total,
            demographic_score=demo_score,
            health_similarity=health_score,
            lifestyle_fit=lifestyle_score,
            personality_compatibility=personality_score,
            engagement_alignment=engagement_score,
            demographic_weight=self.demographic_weight,
            health_weight=self.health_weight,
            lifestyle_weight=self.lifestyle_weight,
            personality_weight=self.personality_weight,
            engagement_weight=self.engagement_weight,
            is_excellent_match=is_excellent,
            is_good_match=is_good,
            is_acceptable_match=is_acceptable,
            shared_conditions=shared_conditions,
            shared_goals=shared_goals,
            potential_issues=issues
        )
    
    def find_best_squad(
        self,
        user: UserProfile,
        available_squads: List[AccountabilitySquad]
    ) -> Optional[Tuple[AccountabilitySquad, float]]:
        """Find best existing squad for user"""
        best_squad = None
        best_score = 0.0
        
        for squad in available_squads:
            # Check if squad has space
            if len(squad.members) >= squad.max_size:
                continue
            
            # Check if squad is accepting members
            if squad.status not in [SquadStatus.FORMING, SquadStatus.ACTIVE]:
                continue
            
            # Calculate compatibility with existing members
            compatibility_scores = []
            for member in squad.members:
                # Would need to load full UserProfile for member
                # Simplified: use squad average compatibility
                pass
            
            # Use squad's average compatibility as baseline
            squad_score = squad.avg_compatibility_score
            
            # Bonus for matching criteria
            if user.conditions and squad.matching_criteria.get('conditions'):
                if any(c in squad.matching_criteria['conditions'] for c in user.conditions):
                    squad_score += 10
            
            if user.goals and squad.matching_criteria.get('goals'):
                if any(g in squad.matching_criteria['goals'] for g in user.goals):
                    squad_score += 10
            
            # Track best match
            if squad_score > best_score:
                best_score = squad_score
                best_squad = squad
        
        if best_squad and best_score >= 60:  # Minimum acceptable score
            return (best_squad, best_score)
        
        return None
    
    def create_optimal_squad(
        self,
        seed_user: UserProfile,
        candidate_users: List[UserProfile],
        target_size: int = 4
    ) -> List[Tuple[UserProfile, MatchingScore]]:
        """Create optimal squad from candidates"""
        selected = []
        remaining = candidate_users.copy()
        
        # Start with seed user
        current_group = [seed_user]
        
        # Iteratively add best matches
        for _ in range(target_size - 1):
            if not remaining:
                break
            
            best_candidate = None
            best_avg_score = 0.0
            best_score_obj = None
            
            # Find candidate with highest average compatibility
            for candidate in remaining:
                # Calculate compatibility with all current members
                scores = []
                for member in current_group:
                    match_score = self.calculate_match_score(member, candidate)
                    scores.append(match_score.total_score)
                
                avg_score = sum(scores) / len(scores)
                
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_candidate = candidate
                    best_score_obj = self.calculate_match_score(seed_user, candidate)
            
            # Add best candidate if acceptable
            if best_candidate and best_avg_score >= 60:
                current_group.append(best_candidate)
                selected.append((best_candidate, best_score_obj))
                remaining.remove(best_candidate)
            else:
                break  # No more acceptable matches
        
        return selected
    
    def _calculate_demographic_score(self, user1: UserProfile, user2: UserProfile) -> float:
        """Score demographic similarity"""
        score = 0.0
        
        # Age similarity (40 points max)
        age_diff = abs(user1.age - user2.age)
        if age_diff <= 5:
            score += 40
        elif age_diff <= 10:
            score += 30
        elif age_diff <= 15:
            score += 20
        elif age_diff <= 20:
            score += 10
        
        # Language match (CRITICAL - 50 points)
        if user1.language == user2.language:
            score += 50
        
        # Timezone compatibility (10 points)
        # Simplified: exact match only
        if user1.timezone == user2.timezone:
            score += 10
        
        return score
    
    def _calculate_health_similarity(self, user1: UserProfile, user2: UserProfile) -> float:
        """Score health profile similarity"""
        score = 0.0
        
        # Shared conditions (40 points max)
        shared_conditions = set(user1.conditions) & set(user2.conditions)
        if shared_conditions:
            score += min(40, len(shared_conditions) * 20)
        
        # Shared goals (40 points max)
        shared_goals = set(user1.goals) & set(user2.goals)
        if shared_goals:
            score += min(40, len(shared_goals) * 15)
        
        # Similar journey stage (20 points)
        if user1.journey_days > 0 and user2.journey_days > 0:
            day_ratio = min(user1.journey_days, user2.journey_days) / max(user1.journey_days, user2.journey_days)
            score += day_ratio * 20
        
        return score
    
    def _calculate_lifestyle_fit(self, user1: UserProfile, user2: UserProfile) -> float:
        """Score lifestyle compatibility"""
        score = 0.0
        
        # Activity level match (30 points)
        activity_levels = ["sedentary", "light", "moderate", "active", "very_active"]
        if user1.activity_level in activity_levels and user2.activity_level in activity_levels:
            idx1 = activity_levels.index(user1.activity_level)
            idx2 = activity_levels.index(user2.activity_level)
            activity_diff = abs(idx1 - idx2)
            score += max(0, 30 - activity_diff * 10)
        
        # Dietary preferences overlap (30 points)
        if user1.dietary_preferences and user2.dietary_preferences:
            shared_prefs = set(user1.dietary_preferences) & set(user2.dietary_preferences)
            total_prefs = set(user1.dietary_preferences) | set(user2.dietary_preferences)
            if total_prefs:
                overlap_ratio = len(shared_prefs) / len(total_prefs)
                score += overlap_ratio * 30
        else:
            score += 15  # Neutral if neither has preferences
        
        # Schedule compatibility (40 points)
        if user1.typical_schedule == user2.typical_schedule:
            score += 40
        elif user1.typical_schedule == "flexible" or user2.typical_schedule == "flexible":
            score += 30  # Flexible can adapt
        else:
            score += 20  # Different schedules, lower score
        
        return score
    
    def _calculate_personality_compatibility(self, user1: UserProfile, user2: UserProfile) -> float:
        """Score personality fit"""
        score = 0.0
        
        # Communication style match (30 points)
        comm_map = {CommunicationStyle.MINIMAL: 0, CommunicationStyle.MODERATE: 1, CommunicationStyle.CHATTY: 2}
        comm_diff = abs(comm_map[user1.communication_style] - comm_map[user2.communication_style])
        score += max(0, 30 - comm_diff * 15)
        
        # Motivation type compatibility (40 points)
        # Some types work well together
        if user1.motivation_type == user2.motivation_type:
            score += 40  # Perfect match
        elif user1.motivation_type == MotivationType.SUPPORTIVE or user2.motivation_type == MotivationType.SUPPORTIVE:
            score += 35  # Supportive works with everyone
        elif user1.motivation_type == MotivationType.COLLABORATIVE or user2.motivation_type == MotivationType.COLLABORATIVE:
            score += 30  # Collaborative is flexible
        else:
            score += 20  # Competitive + Educational can work
        
        # Competitiveness alignment (30 points)
        comp_diff = abs(user1.competitiveness - user2.competitiveness)
        score += (1 - comp_diff) * 30
        
        return score
    
    def _calculate_engagement_alignment(self, user1: UserProfile, user2: UserProfile) -> float:
        """Score engagement level similarity"""
        score = 0.0
        
        # Streak similarity (40 points)
        if user1.current_streak > 0 and user2.current_streak > 0:
            streak_ratio = min(user1.current_streak, user2.current_streak) / max(user1.current_streak, user2.current_streak)
            score += streak_ratio * 40
        elif user1.current_streak == 0 and user2.current_streak == 0:
            score += 20  # Both starting fresh
        else:
            score += 10  # Mismatch
        
        # Scan frequency similarity (30 points)
        if user1.avg_scans_per_day > 0 and user2.avg_scans_per_day > 0:
            scan_ratio = min(user1.avg_scans_per_day, user2.avg_scans_per_day) / max(user1.avg_scans_per_day, user2.avg_scans_per_day)
            score += scan_ratio * 30
        
        # Overall engagement similarity (30 points)
        if user1.engagement_score > 0 and user2.engagement_score > 0:
            engagement_ratio = min(user1.engagement_score, user2.engagement_score) / max(user1.engagement_score, user2.engagement_score)
            score += engagement_ratio * 30
        
        return score


# ==================== SQUAD MANAGER ====================

class SquadManager:
    """Manage squad lifecycle and operations"""
    
    def __init__(self):
        self.matching_algorithm = MatchingAlgorithm()
        
        # In-memory storage
        self.squads: Dict[str, AccountabilitySquad] = {}
        self.user_squads: Dict[str, List[str]] = defaultdict(list)
    
    def create_squad(
        self,
        creator: UserProfile,
        name: str,
        description: str,
        max_size: int = 4
    ) -> AccountabilitySquad:
        """Create new squad"""
        squad_id = self._generate_squad_id()
        
        # Create creator member
        creator_member = SquadMember(
            user_id=creator.user_id,
            display_name=f"User{creator.user_id[-4:]}",
            avatar=f"avatar_{creator.user_id[-2:]}",
            is_creator=True,
            is_moderator=True
        )
        
        # Initial stats
        stats = SquadStats(
            total_members=1,
            active_members_7d=1,
            total_messages=0,
            messages_per_day=0.0,
            avg_response_time_hours=0.0,
            avg_member_streak=creator.current_streak,
            total_scans=0,
            total_weight_loss_kg=0.0,
            combined_points=0,
            active_challenges=0,
            completed_challenges=0,
            challenge_completion_rate=0.0,
            squad_health_score=100.0,
            activity_trend="stable",
            risk_of_disbanding=0.0
        )
        
        # Create squad
        squad = AccountabilitySquad(
            squad_id=squad_id,
            name=name,
            description=description,
            members=[creator_member],
            max_size=max_size,
            status=SquadStatus.FORMING,
            created_at=datetime.now(),
            matching_criteria={
                'conditions': creator.conditions,
                'goals': creator.goals,
                'age_range': (creator.age - 5, creator.age + 5),
                'timezone': creator.timezone
            },
            avg_compatibility_score=100.0,  # Perfect for creator
            active_challenges=[],
            completed_challenges=[],
            stats=stats
        )
        
        # Store
        self.squads[squad_id] = squad
        self.user_squads[creator.user_id].append(squad_id)
        
        return squad
    
    def find_squad_for_user(
        self,
        user: UserProfile,
        min_score: float = 60.0
    ) -> Optional[Tuple[AccountabilitySquad, float]]:
        """Find best squad match for user"""
        available = [s for s in self.squads.values() if s.status in [SquadStatus.FORMING, SquadStatus.ACTIVE]]
        result = self.matching_algorithm.find_best_squad(user, available)
        
        if result:
            squad, score = result
            if score >= min_score:
                return result
        
        return None
    
    def add_member_to_squad(
        self,
        squad_id: str,
        user: UserProfile
    ) -> bool:
        """Add member to existing squad"""
        if squad_id not in self.squads:
            return False
        
        squad = self.squads[squad_id]
        
        # Check capacity
        if len(squad.members) >= squad.max_size:
            return False
        
        # Create member
        new_member = SquadMember(
            user_id=user.user_id,
            display_name=f"User{user.user_id[-4:]}",
            avatar=f"avatar_{user.user_id[-2:]}"
        )
        
        # Add to squad
        squad.members.append(new_member)
        self.user_squads[user.user_id].append(squad_id)
        
        # Update stats
        squad.stats.total_members = len(squad.members)
        squad.stats.active_members_7d = len(squad.members)  # Assume all active
        
        # Activate if reaching minimum size
        if len(squad.members) >= 3 and squad.status == SquadStatus.FORMING:
            squad.status = SquadStatus.ACTIVE
        
        return True
    
    def create_challenge(
        self,
        squad_id: str,
        challenge_type: ChallengeType,
        name: str,
        duration_days: int = 7,
        individual_target: Optional[float] = None,
        team_target: Optional[float] = None
    ) -> Optional[SquadChallenge]:
        """Create squad challenge"""
        if squad_id not in self.squads:
            return None
        
        squad = self.squads[squad_id]
        challenge_id = f"challenge_{squad_id}_{len(squad.active_challenges) + 1}"
        
        challenge = SquadChallenge(
            challenge_id=challenge_id,
            challenge_type=challenge_type,
            name=name,
            description=f"{name} for {squad.name}",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=duration_days),
            duration_days=duration_days,
            individual_target=individual_target,
            team_target=team_target,
            participants=[m.user_id for m in squad.members],
            points_reward=duration_days * 10
        )
        
        squad.active_challenges.append(challenge)
        squad.stats.active_challenges = len(squad.active_challenges)
        
        return challenge
    
    def update_challenge_progress(
        self,
        squad_id: str,
        challenge_id: str,
        user_id: str,
        progress: float
    ) -> bool:
        """Update member's challenge progress"""
        if squad_id not in self.squads:
            return False
        
        squad = self.squads[squad_id]
        
        # Find challenge
        challenge = None
        for ch in squad.active_challenges:
            if ch.challenge_id == challenge_id:
                challenge = ch
                break
        
        if not challenge:
            return False
        
        # Update progress
        challenge.individual_progress[user_id] = progress
        
        # Calculate team progress
        challenge.team_progress = sum(challenge.individual_progress.values())
        
        # Check completion
        if challenge.team_target and challenge.team_progress >= challenge.team_target:
            challenge.is_completed = True
            challenge.completion_rate = 1.0
            
            # Move to completed
            squad.active_challenges.remove(challenge)
            squad.completed_challenges.append(challenge)
            
            # Update stats
            squad.stats.active_challenges = len(squad.active_challenges)
            squad.stats.completed_challenges = len(squad.completed_challenges)
            squad.stats.combined_points += challenge.points_reward
        
        return True
    
    def calculate_squad_health(self, squad_id: str) -> float:
        """Calculate squad health score"""
        if squad_id not in self.squads:
            return 0.0
        
        squad = self.squads[squad_id]
        health_score = 100.0
        
        # Activity factor (40 points)
        if squad.stats.total_members > 0:
            activity_ratio = squad.stats.active_members_7d / squad.stats.total_members
            health_score -= (1 - activity_ratio) * 40
        
        # Engagement factor (30 points)
        if squad.stats.messages_per_day < 1:
            health_score -= 20
        elif squad.stats.messages_per_day < 3:
            health_score -= 10
        
        # Challenge participation (30 points)
        if squad.stats.active_challenges == 0 and squad.stats.completed_challenges == 0:
            health_score -= 30
        elif squad.stats.challenge_completion_rate < 0.5:
            health_score -= 15
        
        squad.stats.squad_health_score = max(0, health_score)
        
        # Determine risk
        if health_score < 50:
            squad.stats.risk_of_disbanding = 0.8
            squad.status = SquadStatus.DORMANT
        elif health_score < 70:
            squad.stats.risk_of_disbanding = 0.4
        else:
            squad.stats.risk_of_disbanding = 0.1
        
        return health_score
    
    def _generate_squad_id(self) -> str:
        """Generate unique squad ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = ''.join(random.choices('0123456789abcdef', k=6))
        return f"squad_{timestamp}_{random_suffix}"


# ==================== DEMONSTRATION ====================

def demonstrate_accountability_squad():
    """Demonstrate AI-matched accountability squad system"""
    print("=" * 80)
    print("FEATURE 9: AI-MATCHED ACCOUNTABILITY SQUAD DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize manager
    print("🚀 Initializing Squad Manager...")
    manager = SquadManager()
    print("✅ Manager initialized with AI matching algorithm")
    print()
    
    print("=" * 80)
    print("CREATING USER PROFILES")
    print("=" * 80)
    print()
    
    # Create diverse user profiles
    users = [
        UserProfile(
            user_id="user_001",
            age=42,
            timezone="America/Los_Angeles",
            language="en",
            conditions=["Type 2 Diabetes", "Hypertension"],
            goals=["Weight Loss", "Diabetes Reversal", "Medication Reduction"],
            current_weight_kg=95.0,
            target_weight_kg=75.0,
            journey_days=180,
            activity_level="moderate",
            dietary_preferences=["low_carb"],
            typical_schedule="9to5",
            communication_style=CommunicationStyle.MODERATE,
            motivation_type=MotivationType.COLLABORATIVE,
            competitiveness=0.6,
            days_active=150,
            current_streak=30,
            avg_scans_per_day=3.2,
            engagement_score=0.85,
            preferred_squad_size=SquadSize.QUAD
        ),
        UserProfile(
            user_id="user_002",
            age=38,
            timezone="America/Los_Angeles",
            language="en",
            conditions=["Type 2 Diabetes"],
            goals=["Weight Loss", "Glucose Control"],
            current_weight_kg=88.0,
            target_weight_kg=70.0,
            journey_days=120,
            activity_level="active",
            dietary_preferences=["low_carb", "high_protein"],
            typical_schedule="9to5",
            communication_style=CommunicationStyle.CHATTY,
            motivation_type=MotivationType.COMPETITIVE,
            competitiveness=0.8,
            days_active=100,
            current_streak=25,
            avg_scans_per_day=3.5,
            engagement_score=0.90,
            preferred_squad_size=SquadSize.QUAD
        ),
        UserProfile(
            user_id="user_003",
            age=45,
            timezone="America/Los_Angeles",
            language="en",
            conditions=["Type 2 Diabetes", "High Cholesterol"],
            goals=["Diabetes Reversal", "Heart Health"],
            current_weight_kg=92.0,
            target_weight_kg=78.0,
            journey_days=200,
            activity_level="moderate",
            dietary_preferences=["mediterranean"],
            typical_schedule="9to5",
            communication_style=CommunicationStyle.MODERATE,
            motivation_type=MotivationType.SUPPORTIVE,
            competitiveness=0.4,
            days_active=180,
            current_streak=45,
            avg_scans_per_day=3.0,
            engagement_score=0.88,
            preferred_squad_size=SquadSize.QUAD
        ),
        UserProfile(
            user_id="user_004",
            age=40,
            timezone="America/Los_Angeles",
            language="en",
            conditions=["Pre-Diabetes"],
            goals=["Weight Loss", "Prevention"],
            current_weight_kg=82.0,
            target_weight_kg=70.0,
            journey_days=60,
            activity_level="moderate",
            dietary_preferences=["vegetarian"],
            typical_schedule="flexible",
            communication_style=CommunicationStyle.MODERATE,
            motivation_type=MotivationType.EDUCATIONAL,
            competitiveness=0.5,
            days_active=55,
            current_streak=15,
            avg_scans_per_day=2.8,
            engagement_score=0.75,
            preferred_squad_size=SquadSize.QUAD
        )
    ]
    
    for user in users:
        print(f"👤 {user.user_id}")
        print(f"   Age: {user.age} | Conditions: {', '.join(user.conditions)}")
        print(f"   Goals: {', '.join(user.goals)}")
        print(f"   Streak: {user.current_streak} days | Engagement: {user.engagement_score:.0%}")
        print()
    
    print("=" * 80)
    print("CALCULATING MATCH SCORES")
    print("=" * 80)
    print()
    
    # Calculate pairwise match scores
    print("🔍 Compatibility Matrix:")
    print()
    
    for i, user1 in enumerate(users):
        for j, user2 in enumerate(users):
            if i < j:  # Avoid duplicates
                score = manager.matching_algorithm.calculate_match_score(user1, user2)
                
                quality = "⭐ EXCELLENT" if score.is_excellent_match else "✓ GOOD" if score.is_good_match else "○ ACCEPTABLE" if score.is_acceptable_match else "✗ POOR"
                
                print(f"{user1.user_id} ↔ {user2.user_id}: {score.total_score:.1f}/100 {quality}")
                print(f"   Demographics: {score.demographic_score:.1f} | Health: {score.health_similarity:.1f} | Lifestyle: {score.lifestyle_fit:.1f}")
                print(f"   Personality: {score.personality_compatibility:.1f} | Engagement: {score.engagement_alignment:.1f}")
                
                if score.shared_conditions:
                    print(f"   Shared Conditions: {', '.join(score.shared_conditions)}")
                if score.shared_goals:
                    print(f"   Shared Goals: {', '.join(score.shared_goals)}")
                if score.potential_issues:
                    print(f"   ⚠️  Issues: {', '.join(score.potential_issues)}")
                
                print()
    
    print("=" * 80)
    print("CREATING OPTIMAL SQUAD")
    print("=" * 80)
    print()
    
    # Create squad with user_001 as seed
    seed_user = users[0]
    other_users = users[1:]
    
    squad = manager.create_squad(
        creator=seed_user,
        name="Type 2 Warriors",
        description="Support group for reversing Type 2 Diabetes through lifestyle changes",
        max_size=4
    )
    
    print(f"📋 Squad Created: {squad.name}")
    print(f"   ID: {squad.squad_id}")
    print(f"   Description: {squad.description}")
    print(f"   Status: {squad.status.value}")
    print(f"   Size: {len(squad.members)}/{squad.max_size}")
    print()
    
    # Find best matches for squad
    print("🔍 Finding Best Matches...")
    matched = manager.matching_algorithm.create_optimal_squad(seed_user, other_users, target_size=4)
    
    print(f"✅ Found {len(matched)} compatible members:")
    print()
    
    for user, match_score in matched:
        print(f"   • {user.user_id}")
        print(f"     Match Score: {match_score.total_score:.1f}/100")
        print(f"     Conditions: {', '.join(user.conditions)}")
        print(f"     Motivation: {user.motivation_type.value}")
        
        # Add to squad
        manager.add_member_to_squad(squad.squad_id, user)
        print(f"     ✅ Added to squad")
        print()
    
    print(f"📊 Squad Status: {squad.status.value} ({len(squad.members)} members)")
    print()
    
    print("=" * 80)
    print("CREATING SQUAD CHALLENGES")
    print("=" * 80)
    print()
    
    # Create challenges
    challenges = [
        {
            'type': ChallengeType.SCAN_STREAK,
            'name': '7-Day Scan Streak',
            'duration': 7,
            'individual_target': 7.0,
            'team_target': 28.0  # 4 members × 7 days
        },
        {
            'type': ChallengeType.COMPLIANCE,
            'name': 'Meal Plan Mastery',
            'duration': 7,
            'individual_target': 0.85,  # 85% compliance
            'team_target': None
        },
        {
            'type': ChallengeType.WEIGHT_LOSS,
            'name': 'Team Weight Loss Challenge',
            'duration': 30,
            'individual_target': 2.0,  # 2kg per member
            'team_target': 8.0  # 8kg total
        }
    ]
    
    created_challenges = []
    for ch_data in challenges:
        challenge = manager.create_challenge(
            squad_id=squad.squad_id,
            challenge_type=ch_data['type'],
            name=ch_data['name'],
            duration_days=ch_data['duration'],
            individual_target=ch_data['individual_target'],
            team_target=ch_data['team_target']
        )
        
        created_challenges.append(challenge)
        
        print(f"🎯 Challenge Created: {challenge.name}")
        print(f"   Type: {challenge.challenge_type.value}")
        print(f"   Duration: {challenge.duration_days} days")
        print(f"   Participants: {len(challenge.participants)}")
        print(f"   Team Target: {challenge.team_target}")
        print(f"   Reward: {challenge.points_reward} points")
        print()
    
    print("=" * 80)
    print("SIMULATING CHALLENGE PROGRESS")
    print("=" * 80)
    print()
    
    # Simulate progress on first challenge (7-Day Scan Streak)
    challenge = created_challenges[0]
    
    print(f"📈 Simulating progress for: {challenge.name}")
    print()
    
    # Each member makes progress
    progress_updates = [
        (users[0].user_id, 7.0, "100%"),  # Perfect
        (users[1].user_id, 7.0, "100%"),  # Perfect
        (users[2].user_id, 6.0, "86%"),   # Missed 1 day
        (users[3].user_id, 7.0, "100%")   # Perfect
    ]
    
    for user_id, progress, percent in progress_updates:
        manager.update_challenge_progress(squad.squad_id, challenge.challenge_id, user_id, progress)
        print(f"   • {user_id}: {progress}/{challenge.individual_target} days ({percent})")
    
    print()
    print(f"🏆 Team Progress: {challenge.team_progress}/{challenge.team_target}")
    print(f"   Status: {'✅ COMPLETED!' if challenge.is_completed else '🔄 In Progress'}")
    
    if challenge.is_completed:
        print(f"   Reward: {challenge.points_reward} points earned!")
    
    print()
    
    print("=" * 80)
    print("SQUAD STATISTICS")
    print("=" * 80)
    print()
    
    # Calculate and display squad health
    health_score = manager.calculate_squad_health(squad.squad_id)
    
    print(f"📊 {squad.name} Statistics:")
    print(f"   Total Members: {squad.stats.total_members}")
    print(f"   Active (7d): {squad.stats.active_members_7d}")
    print(f"   Combined Points: {squad.stats.combined_points}")
    print(f"   Avg Member Streak: {squad.stats.avg_member_streak:.1f} days")
    print()
    
    print(f"🎯 Challenges:")
    print(f"   Active: {squad.stats.active_challenges}")
    print(f"   Completed: {squad.stats.completed_challenges}")
    print(f"   Completion Rate: {squad.stats.challenge_completion_rate:.0%}")
    print()
    
    print(f"💚 Squad Health:")
    print(f"   Health Score: {squad.stats.squad_health_score:.1f}/100")
    print(f"   Activity Trend: {squad.stats.activity_trend}")
    print(f"   Risk of Disbanding: {squad.stats.risk_of_disbanding:.0%}")
    print()
    
    print("=" * 80)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print()
    
    print("📊 This system showcases:")
    print("   • Multi-dimensional matching (demographics, health, lifestyle, personality)")
    print("   • Optimal squad creation (3-6 members, balanced compatibility)")
    print("   • Shared challenges with individual and team goals")
    print("   • Progress tracking and leaderboards")
    print("   • Squad health monitoring and risk assessment")
    print("   • Engagement metrics and activity trends")
    print()
    
    print("🎯 Production Implementation:")
    print("   • Real-time chat integration (WebSocket, Firebase)")
    print("   • Push notifications for squad activities")
    print("   • Video calls and voice messages")
    print("   • Advanced matching: ML-based compatibility prediction")
    print("   • Dynamic rebalancing: Suggest new members for inactive squads")
    print("   • Conflict resolution: Mediation system, member voting")
    print("   • Squad achievements: Collective badges and rewards")
    print("   • Cross-squad competitions and tournaments")


if __name__ == "__main__":
    demonstrate_accountability_squad()
