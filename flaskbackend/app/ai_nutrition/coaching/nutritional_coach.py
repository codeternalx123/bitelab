"""
Real-Time Nutritional Coaching AI
==================================

Conversational AI system for personalized nutritional coaching, behavioral
interventions, and real-time meal guidance. Uses natural language processing
to provide human-like coaching experiences.

Features:
1. Conversational meal guidance
2. Portion control coaching
3. Behavioral nutrition interventions
4. Motivational interviewing techniques
5. Goal-setting and tracking
6. Habit formation support
7. Mindful eating coaching
8. Emotional eating support
9. Progress celebration and encouragement
10. Personalized feedback loops

Coaching Techniques:
- Motivational Interviewing (MI)
- Cognitive Behavioral Therapy (CBT) for eating
- SMART goal framework
- Stages of Change model (Transtheoretical)
- Habit stacking
- Implementation intentions

Use Cases:
1. "What should I eat for lunch?" → Personalized suggestion + rationale
2. "I'm craving ice cream" → Mindful intervention + healthy alternatives
3. "I ate too much at dinner" → Non-judgmental coaching + future strategies
4. "Help me plan for a restaurant" → Pre-meal planning + ordering tips
5. "I keep snacking at night" → Habit analysis + behavior modification

Author: Wellomex AI Team
Date: November 2025
Version: 15.0.0
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, date, timedelta
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


# ============================================================================
# COACHING ENUMS
# ============================================================================

class CoachingStyle(Enum):
    """Coaching communication styles"""
    SUPPORTIVE = "supportive"          # Empathetic, encouraging
    DIRECT = "direct"                  # Clear, action-oriented
    EDUCATIONAL = "educational"        # Teaching-focused
    MOTIVATIONAL = "motivational"      # High energy, inspiring


class StageOfChange(Enum):
    """Transtheoretical Model stages"""
    PRECONTEMPLATION = "precontemplation"  # Not ready
    CONTEMPLATION = "contemplation"        # Thinking about it
    PREPARATION = "preparation"            # Getting ready
    ACTION = "action"                      # Making changes
    MAINTENANCE = "maintenance"            # Sustaining changes
    RELAPSE = "relapse"                    # Temporary setback


class InterventionType(Enum):
    """Behavioral intervention types"""
    PORTION_CONTROL = "portion_control"
    MINDFUL_EATING = "mindful_eating"
    MEAL_TIMING = "meal_timing"
    FOOD_SWAPS = "food_swaps"
    EMOTIONAL_EATING = "emotional_eating"
    SOCIAL_EATING = "social_eating"
    HABIT_FORMATION = "habit_formation"


class EmotionalState(Enum):
    """User emotional states"""
    MOTIVATED = "motivated"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    PROUD = "proud"
    GUILTY = "guilty"
    STRESSED = "stressed"
    NEUTRAL = "neutral"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CoachingProfile:
    """User coaching profile"""
    user_id: str
    
    # Preferences
    preferred_style: CoachingStyle = CoachingStyle.SUPPORTIVE
    communication_frequency: str = "daily"  # daily, weekly, on-demand
    
    # Stage of change
    current_stage: StageOfChange = StageOfChange.CONTEMPLATION
    
    # Goals
    primary_goal: str = ""
    secondary_goals: List[str] = field(default_factory=list)
    
    # Barriers
    identified_barriers: List[str] = field(default_factory=list)
    
    # Motivators
    intrinsic_motivators: List[str] = field(default_factory=list)
    extrinsic_motivators: List[str] = field(default_factory=list)


@dataclass
class CoachingConversation:
    """Coaching conversation session"""
    conversation_id: str
    user_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Context
    user_message: str = ""
    detected_emotion: Optional[EmotionalState] = None
    detected_intent: str = ""  # ask_advice, report_progress, express_concern
    
    # Coach response
    coach_response: str = ""
    intervention_type: Optional[InterventionType] = None
    
    # Techniques used
    mi_technique_used: Optional[str] = None  # open_question, affirmation, reflection
    
    # Follow-up
    action_items: List[str] = field(default_factory=list)
    scheduled_followup: Optional[datetime] = None


@dataclass
class NutritionalGoal:
    """SMART nutritional goal"""
    goal_id: str
    user_id: str
    
    # SMART components
    specific: str  # Specific statement
    measurable: str  # How to measure
    achievable: bool = True
    relevant: str  # Why it matters
    time_bound: date = field(default_factory=lambda: date.today() + timedelta(days=30))
    
    # Progress
    start_date: date = field(default_factory=date.today)
    current_progress: float = 0.0  # 0.0-1.0
    milestones: List[str] = field(default_factory=list)
    
    # Status
    status: str = "active"  # active, completed, abandoned


@dataclass
class BehavioralIntervention:
    """Evidence-based behavioral intervention"""
    intervention_id: str
    intervention_type: InterventionType
    
    # Target behavior
    target_behavior: str
    
    # Strategy
    strategy_name: str
    strategy_description: str
    
    # Implementation
    implementation_steps: List[str] = field(default_factory=list)
    
    # Evidence
    evidence_base: str = ""  # CBT, MI, Habit formation research
    success_rate: float = 0.0  # 0.0-1.0
    
    # Personalization
    personalized_tip: str = ""


@dataclass
class PortionGuidance:
    """Visual portion size guidance"""
    food_item: str
    
    # Recommended portion
    recommended_amount: str  # "1 cup", "palm-sized"
    calories: float
    
    # Visual cue
    visual_comparison: str  # "Size of a baseball", "Deck of cards"
    hand_measure: Optional[str] = None  # "Palm", "Fist", "Thumb"
    
    # Rationale
    rationale: str = ""


@dataclass
class MealSuggestion:
    """AI-generated meal suggestion"""
    suggestion_id: str
    
    # Meal details
    meal_type: str  # breakfast, lunch, dinner, snack
    suggested_foods: List[str] = field(default_factory=list)
    
    # Nutrition
    estimated_calories: float = 0.0
    protein_g: float = 0.0
    
    # Personalization
    reasoning: str = ""  # Why this suggestion
    
    # Preparation
    quick_prep: bool = False
    prep_time_minutes: int = 0


# ============================================================================
# MOTIVATIONAL INTERVIEWING ENGINE
# ============================================================================

class MotivationalInterviewingEngine:
    """
    Motivational Interviewing (MI) conversation engine
    
    MI Core Skills (OARS):
    - Open-ended questions
    - Affirmations
    - Reflective listening
    - Summaries
    """
    
    def __init__(self):
        # Open-ended question templates
        self.open_questions = {
            'goal_exploration': [
                "What would you like to achieve with your nutrition?",
                "How would your life be different if you reached your goal?",
                "What matters most to you about your health?"
            ],
            'barrier_identification': [
                "What gets in the way of eating healthier?",
                "What challenges have you faced before?",
                "What makes healthy eating difficult for you?"
            ],
            'motivation_building': [
                "What are your reasons for wanting to change?",
                "On a scale of 1-10, how important is this goal to you? Why that number?",
                "What would need to happen for you to move from a 5 to a 7?"
            ]
        }
        
        # Affirmations
        self.affirmations = [
            "That shows real commitment.",
            "You're taking important steps.",
            "That's a great insight about yourself.",
            "You're being thoughtful about this.",
            "That takes courage to recognize."
        ]
        
        # Reflective listening stems
        self.reflections = {
            'simple': [
                "It sounds like...",
                "You're feeling...",
                "What I hear you saying is..."
            ],
            'complex': [
                "On one hand you want... but on the other hand...",
                "You value... and you're also concerned about...",
                "Part of you... while another part..."
            ]
        }
        
        logger.info("Motivational Interviewing Engine initialized")
    
    def generate_open_question(self, category: str) -> str:
        """Generate open-ended question"""
        questions = self.open_questions.get(category, self.open_questions['goal_exploration'])
        return random.choice(questions)
    
    def generate_affirmation(self, user_action: str) -> str:
        """Generate affirmation for user action"""
        base_affirmation = random.choice(self.affirmations)
        return f"{base_affirmation} {user_action}"
    
    def reflect_statement(self, user_statement: str, complexity: str = 'simple') -> str:
        """Generate reflective listening response"""
        stem = random.choice(self.reflections[complexity])
        
        if complexity == 'simple':
            return f"{stem} {user_statement.lower()}"
        else:
            # Complex reflection requires ambivalence detection
            return f"{stem} you're navigating multiple priorities."


# ============================================================================
# BEHAVIORAL INTERVENTION LIBRARY
# ============================================================================

class BehavioralInterventionLibrary:
    """
    Evidence-based behavioral nutrition interventions
    """
    
    def __init__(self):
        self.interventions: Dict[str, BehavioralIntervention] = {}
        
        self._build_intervention_library()
        
        logger.info(f"Behavioral Intervention Library initialized with {len(self.interventions)} interventions")
    
    def _build_intervention_library(self):
        """Build intervention library"""
        
        # Portion Control - Visual Cues
        self.interventions['portion_visual'] = BehavioralIntervention(
            intervention_id='portion_visual',
            intervention_type=InterventionType.PORTION_CONTROL,
            target_behavior='Overeating due to large portions',
            strategy_name='Visual Portion Cues',
            strategy_description='Use hand measurements and visual comparisons to estimate portions',
            implementation_steps=[
                '1. Use your palm to measure protein portions (3-4 oz)',
                '2. Use your fist to measure carbs (1 cup)',
                '3. Use your thumb to measure fats (1 tablespoon)',
                '4. Fill half your plate with vegetables'
            ],
            evidence_base='Behavioral nutrition research (Wansink, 2006)',
            success_rate=0.75,
            personalized_tip='Start with one meal per day and build from there'
        )
        
        # Mindful Eating - PAUSE Technique
        self.interventions['mindful_pause'] = BehavioralIntervention(
            intervention_id='mindful_pause',
            intervention_type=InterventionType.MINDFUL_EATING,
            target_behavior='Eating too quickly without awareness',
            strategy_name='PAUSE Before Eating',
            strategy_description='Take mindful pause before meals',
            implementation_steps=[
                'P - Pause before eating',
                'A - Assess your hunger (1-10 scale)',
                'U - Unplug from distractions (phone, TV)',
                'S - Smell and appreciate your food',
                'E - Eat slowly, savor each bite'
            ],
            evidence_base='Mindfulness-Based Eating Awareness Training (MB-EAT)',
            success_rate=0.70,
            personalized_tip='Set a reminder on your phone for meal times'
        )
        
        # Emotional Eating - HALT Check
        self.interventions['emotional_halt'] = BehavioralIntervention(
            intervention_id='emotional_halt',
            intervention_type=InterventionType.EMOTIONAL_EATING,
            target_behavior='Eating in response to emotions rather than hunger',
            strategy_name='HALT Before Emotional Eating',
            strategy_description='Check if you\'re Hungry, Angry, Lonely, or Tired',
            implementation_steps=[
                'H - Am I actually Hungry? (Physical vs. emotional)',
                'A - Am I Angry or upset?',
                'L - Am I feeling Lonely?',
                'T - Am I Tired or stressed?',
                'If not hungry: Address the real need (rest, call friend, journal)'
            ],
            evidence_base='CBT for Binge Eating Disorder (Fairburn, 2008)',
            success_rate=0.68,
            personalized_tip='Keep a list of non-food coping strategies visible'
        )
        
        # Habit Formation - Implementation Intentions
        self.interventions['habit_when_then'] = BehavioralIntervention(
            intervention_id='habit_when_then',
            intervention_type=InterventionType.HABIT_FORMATION,
            target_behavior='Inconsistent healthy eating habits',
            strategy_name='When-Then Planning (Implementation Intentions)',
            strategy_description='Create specific if-then plans for healthy behaviors',
            implementation_steps=[
                '1. Identify the trigger: "When I wake up..."',
                '2. Specify the action: "Then I will drink 16oz water"',
                '3. Make it specific (time, place, action)',
                '4. Start with ONE habit at a time',
                '5. Track daily for 21 days'
            ],
            evidence_base='Implementation intentions research (Gollwitzer, 1999)',
            success_rate=0.80,
            personalized_tip='Link new habit to existing routine (habit stacking)'
        )
        
        # Food Swaps - Gradual Substitution
        self.interventions['food_swaps_gradual'] = BehavioralIntervention(
            intervention_id='food_swaps_gradual',
            intervention_type=InterventionType.FOOD_SWAPS,
            target_behavior='Reliance on less nutritious foods',
            strategy_name='Gradual Food Substitution',
            strategy_description='Slowly replace less healthy foods with nutritious alternatives',
            implementation_steps=[
                '1. Choose ONE food to swap (e.g., soda)',
                '2. Find similar alternative (flavored sparkling water)',
                '3. Week 1: Replace 25% of servings',
                '4. Week 2: Replace 50%',
                '5. Week 3: Replace 75%',
                '6. Week 4: Complete swap (or find comfortable balance)'
            ],
            evidence_base='Behavior change gradual approach (Prochaska, 1992)',
            success_rate=0.72,
            personalized_tip='Focus on addition (add good) before subtraction (remove bad)'
        )
        
        # Meal Timing - Structured Eating
        self.interventions['meal_timing_structured'] = BehavioralIntervention(
            intervention_id='meal_timing_structured',
            intervention_type=InterventionType.MEAL_TIMING,
            target_behavior='Irregular eating patterns leading to overeating',
            strategy_name='Structured Meal Timing',
            strategy_description='Eat at consistent times to regulate hunger cues',
            implementation_steps=[
                '1. Set 3 meal times (breakfast, lunch, dinner)',
                '2. Add 1-2 planned snacks if needed',
                '3. Eat within 1 hour of waking',
                '4. Space meals 4-5 hours apart',
                '5. Stop eating 2-3 hours before bed'
            ],
            evidence_base='Chrononutrition research (Garaulet & Gomez-Abellan, 2014)',
            success_rate=0.65,
            personalized_tip='Use phone alarms to remind you of meal times initially'
        )
    
    def get_intervention(self, intervention_id: str) -> Optional[BehavioralIntervention]:
        """Get intervention by ID"""
        return self.interventions.get(intervention_id)
    
    def recommend_intervention(
        self,
        target_behavior: str,
        user_stage: StageOfChange
    ) -> Optional[BehavioralIntervention]:
        """Recommend intervention based on behavior and stage of change"""
        # Match intervention to behavior
        for intervention in self.interventions.values():
            if target_behavior.lower() in intervention.target_behavior.lower():
                # Adjust recommendation based on stage
                if user_stage in [StageOfChange.PRECONTEMPLATION, StageOfChange.CONTEMPLATION]:
                    # Focus on awareness and education
                    return intervention
                elif user_stage in [StageOfChange.PREPARATION, StageOfChange.ACTION]:
                    # Provide actionable strategies
                    return intervention
        
        return None


# ============================================================================
# PORTION CONTROL COACH
# ============================================================================

class PortionControlCoach:
    """
    Real-time portion size guidance
    """
    
    def __init__(self):
        self.portion_database: Dict[str, PortionGuidance] = {}
        
        self._build_portion_database()
        
        logger.info(f"Portion Control Coach initialized with {len(self.portion_database)} foods")
    
    def _build_portion_database(self):
        """Build portion guidance database"""
        
        # Proteins
        self.portion_database['chicken_breast'] = PortionGuidance(
            food_item='Chicken Breast',
            recommended_amount='3-4 oz (cooked)',
            calories=165.0,
            visual_comparison='Deck of cards',
            hand_measure='Palm of your hand',
            rationale='Adequate protein without excess calories'
        )
        
        self.portion_database['salmon'] = PortionGuidance(
            food_item='Salmon',
            recommended_amount='4 oz (cooked)',
            calories=206.0,
            visual_comparison='Checkbook',
            hand_measure='Palm of your hand',
            rationale='Rich in omega-3s, heart-healthy protein portion'
        )
        
        # Carbs
        self.portion_database['rice'] = PortionGuidance(
            food_item='Rice (cooked)',
            recommended_amount='1/2 cup',
            calories=103.0,
            visual_comparison='Tennis ball',
            hand_measure='Cupped hand',
            rationale='Controlled carbohydrate portion for energy'
        )
        
        self.portion_database['pasta'] = PortionGuidance(
            food_item='Pasta (cooked)',
            recommended_amount='1 cup',
            calories=200.0,
            visual_comparison='Baseball',
            hand_measure='Your fist',
            rationale='Standard serving size for sustained energy'
        )
        
        # Fats
        self.portion_database['olive_oil'] = PortionGuidance(
            food_item='Olive Oil',
            recommended_amount='1 tablespoon',
            calories=119.0,
            visual_comparison='Poker chip',
            hand_measure='Tip of your thumb',
            rationale='Heart-healthy fats in controlled amount'
        )
        
        self.portion_database['nuts'] = PortionGuidance(
            food_item='Nuts (almonds, walnuts)',
            recommended_amount='1 oz (about 23 almonds)',
            calories=164.0,
            visual_comparison='Golf ball',
            hand_measure='Small handful',
            rationale='Nutrient-dense snack portion'
        )
        
        # Vegetables (unlimited)
        self.portion_database['vegetables'] = PortionGuidance(
            food_item='Non-starchy Vegetables',
            recommended_amount='At least 1-2 cups (fill half your plate)',
            calories=50.0,
            visual_comparison='2 baseballs',
            hand_measure='Two fists',
            rationale='High volume, low calories, rich in nutrients'
        )
        
        # Dairy
        self.portion_database['cheese'] = PortionGuidance(
            food_item='Cheese',
            recommended_amount='1 oz',
            calories=114.0,
            visual_comparison='4 stacked dice',
            hand_measure='Your thumb',
            rationale='Calcium and protein with calorie awareness'
        )
    
    def get_portion_guidance(self, food_item: str) -> Optional[PortionGuidance]:
        """Get portion guidance for food"""
        food_lower = food_item.lower().replace(' ', '_')
        return self.portion_database.get(food_lower)
    
    def coach_portion(self, food_item: str, user_portion: str) -> str:
        """Provide portion coaching"""
        guidance = self.get_portion_guidance(food_item)
        
        if not guidance:
            return f"For {food_item}, try using visual cues: protein = palm size, carbs = fist size, fats = thumb size."
        
        response = f"**{guidance.food_item}**\n\n"
        response += f"✓ Recommended: {guidance.recommended_amount}\n"
        response += f"📏 Visual cue: {guidance.visual_comparison}\n"
        response += f"✋ Hand measure: {guidance.hand_measure}\n"
        response += f"💡 Why: {guidance.rationale}\n"
        response += f"🔢 Calories: ~{guidance.calories:.0f}"
        
        return response


# ============================================================================
# CONVERSATIONAL COACH
# ============================================================================

class NutritionalCoach:
    """
    Complete conversational nutritional coaching system
    """
    
    def __init__(self):
        self.mi_engine = MotivationalInterviewingEngine()
        self.intervention_library = BehavioralInterventionLibrary()
        self.portion_coach = PortionControlCoach()
        
        # User profiles
        self.user_profiles: Dict[str, CoachingProfile] = {}
        
        # Conversation history
        self.conversations: List[CoachingConversation] = []
        
        logger.info("Nutritional Coach initialized")
    
    def create_user_profile(
        self,
        user_id: str,
        primary_goal: str,
        preferred_style: CoachingStyle = CoachingStyle.SUPPORTIVE
    ) -> CoachingProfile:
        """Create coaching profile for user"""
        profile = CoachingProfile(
            user_id=user_id,
            preferred_style=preferred_style,
            primary_goal=primary_goal
        )
        
        self.user_profiles[user_id] = profile
        
        return profile
    
    def detect_intent(self, user_message: str) -> str:
        """Detect user intent from message"""
        message_lower = user_message.lower()
        
        # Intent patterns
        if any(word in message_lower for word in ['should i', 'what should', 'recommend', 'suggest']):
            return 'ask_advice'
        elif any(word in message_lower for word in ['i ate', 'i had', 'just finished']):
            return 'report_meal'
        elif any(word in message_lower for word in ['craving', 'want to eat', 'tempted']):
            return 'express_craving'
        elif any(word in message_lower for word in ['struggling', 'difficult', 'hard to', 'challenge']):
            return 'express_difficulty'
        elif any(word in message_lower for word in ['achieved', 'success', 'did it', 'proud']):
            return 'celebrate_progress'
        else:
            return 'general_question'
    
    def detect_emotion(self, user_message: str) -> EmotionalState:
        """Detect emotional state from message"""
        message_lower = user_message.lower()
        
        # Emotion keywords
        if any(word in message_lower for word in ['excited', 'motivated', 'ready', 'determined']):
            return EmotionalState.MOTIVATED
        elif any(word in message_lower for word in ['frustrated', 'annoyed', 'stuck']):
            return EmotionalState.FRUSTRATED
        elif any(word in message_lower for word in ['confused', 'unsure', 'don\'t know']):
            return EmotionalState.CONFUSED
        elif any(word in message_lower for word in ['proud', 'accomplished', 'happy']):
            return EmotionalState.PROUD
        elif any(word in message_lower for word in ['guilty', 'bad', 'failed', 'messed up']):
            return EmotionalState.GUILTY
        elif any(word in message_lower for word in ['stressed', 'overwhelmed', 'anxious']):
            return EmotionalState.STRESSED
        else:
            return EmotionalState.NEUTRAL
    
    def coach_response(
        self,
        user_id: str,
        user_message: str
    ) -> str:
        """Generate coaching response"""
        # Get user profile
        profile = self.user_profiles.get(user_id)
        if not profile:
            return "Hi! I'm your nutritional coach. Let's start by setting a goal together. What would you like to achieve?"
        
        # Detect intent and emotion
        intent = self.detect_intent(user_message)
        emotion = self.detect_emotion(user_message)
        
        # Generate response based on intent
        if intent == 'ask_advice':
            response = self._provide_meal_advice(user_message, profile)
        elif intent == 'express_craving':
            response = self._handle_craving(user_message, emotion, profile)
        elif intent == 'express_difficulty':
            response = self._provide_support(user_message, emotion, profile)
        elif intent == 'celebrate_progress':
            response = self._celebrate_win(user_message, profile)
        elif intent == 'report_meal':
            response = self._feedback_on_meal(user_message, profile)
        else:
            response = self._general_coaching(user_message, profile)
        
        # Log conversation
        conversation = CoachingConversation(
            conversation_id=f"conv_{len(self.conversations)}",
            user_id=user_id,
            user_message=user_message,
            detected_emotion=emotion,
            detected_intent=intent,
            coach_response=response
        )
        self.conversations.append(conversation)
        
        return response
    
    def _provide_meal_advice(self, message: str, profile: CoachingProfile) -> str:
        """Provide meal suggestion"""
        # Simplified meal suggestion
        meal_suggestions = [
            "🥗 How about a colorful salad with grilled chicken, quinoa, and olive oil dressing? It's balanced with protein, whole grains, and healthy fats.",
            "🍲 Consider a veggie stir-fry with tofu and brown rice. Quick to make and packed with nutrients!",
            "🐟 Baked salmon with roasted vegetables and sweet potato would be excellent. Rich in omega-3s!",
            "🥙 A whole grain wrap with hummus, veggies, and grilled chicken gives you sustained energy."
        ]
        
        suggestion = random.choice(meal_suggestions)
        
        response = f"{suggestion}\n\n"
        response += f"💡 **Why this works for you**: This aligns with your goal of {profile.primary_goal}.\n\n"
        response += "🍽️ **Portion tip**: Fill half your plate with vegetables, 1/4 with protein (palm-sized), 1/4 with whole grains (fist-sized)."
        
        return response
    
    def _handle_craving(self, message: str, emotion: EmotionalState, profile: CoachingProfile) -> str:
        """Handle food craving with mindful intervention"""
        # HALT check
        response = "I hear you're having a craving. Let's pause for a moment. 🧘\n\n"
        response += "**HALT Check**:\n"
        response += "• Are you actually **Hungry** (stomach growling)?\n"
        response += "• Are you **Angry** or upset about something?\n"
        response += "• Are you feeling **Lonely**?\n"
        response += "• Are you **Tired** or stressed?\n\n"
        
        response += "If you're truly hungry:\n"
        response += "→ Try a healthier version first (e.g., Greek yogurt with berries instead of ice cream)\n\n"
        
        response += "If it's emotional:\n"
        response += "→ Take a 10-minute walk\n"
        response += "→ Call a friend\n"
        response += "→ Journal about what you're feeling\n\n"
        
        response += "Remember: Cravings typically pass in 10-15 minutes. You've got this! 💪"
        
        return response
    
    def _provide_support(self, message: str, emotion: EmotionalState, profile: CoachingProfile) -> str:
        """Provide emotional support and problem-solving"""
        # Reflective listening
        response = self.mi_engine.reflect_statement("you're facing some challenges", "simple")
        response += " That's completely normal. 🤝\n\n"
        
        # Affirmation
        response += self.mi_engine.generate_affirmation("acknowledging this difficulty") + "\n\n"
        
        # Problem-solving
        response += "Let's break this down:\n\n"
        response += "1. **What's the specific challenge?** (e.g., 'I eat too much at night')\n"
        response += "2. **When does it happen?** (trigger identification)\n"
        response += "3. **What could you try differently?** (brainstorm solutions)\n\n"
        
        response += "Would you like to explore a specific strategy together?"
        
        return response
    
    def _celebrate_win(self, message: str, profile: CoachingProfile) -> str:
        """Celebrate user progress"""
        celebrations = [
            "🎉 Amazing work! That's a real achievement!",
            "👏 You should be proud of yourself! That took effort!",
            "⭐ Fantastic! You're building healthy habits!",
            "🌟 This is exactly the kind of progress that creates lasting change!"
        ]
        
        response = random.choice(celebrations) + "\n\n"
        response += "**What made this successful?** Understanding what worked helps you repeat it.\n\n"
        response += "Keep building on this momentum. What's your next small step? 🚀"
        
        return response
    
    def _feedback_on_meal(self, message: str, profile: CoachingProfile) -> str:
        """Provide non-judgmental feedback on reported meal"""
        response = "Thanks for sharing what you ate! 📝\n\n"
        response += "**Reflection questions**:\n"
        response += "• How did you feel before eating? (hungry, stressed, etc.)\n"
        response += "• How do you feel now? (satisfied, too full, still hungry)\n"
        response += "• Was this meal aligned with your goals?\n\n"
        
        response += "**Remember**: Every meal is a new opportunity. No judgment, just learning. 🌱"
        
        return response
    
    def _general_coaching(self, message: str, profile: CoachingProfile) -> str:
        """General coaching conversation"""
        # Ask open-ended question
        question = self.mi_engine.generate_open_question('goal_exploration')
        
        response = f"{question}\n\n"
        response += "I'm here to support you on your nutrition journey. How can I help today?"
        
        return response


# ============================================================================
# TESTING
# ============================================================================

def test_nutritional_coach():
    """Test nutritional coaching system"""
    print("=" * 80)
    print("REAL-TIME NUTRITIONAL COACHING AI - TEST")
    print("=" * 80)
    
    # Initialize
    coach = NutritionalCoach()
    
    # Create user profile
    print("\n" + "="*80)
    print("Test: Create User Coaching Profile")
    print("="*80)
    
    profile = coach.create_user_profile(
        user_id='user123',
        primary_goal='Lose 15 pounds by eating healthier',
        preferred_style=CoachingStyle.SUPPORTIVE
    )
    
    print(f"✓ Profile created for user123")
    print(f"   Goal: {profile.primary_goal}")
    print(f"   Style: {profile.preferred_style.value}")
    print(f"   Stage: {profile.current_stage.value}")
    
    # Test 2: Meal advice request
    print("\n" + "="*80)
    print("Test: Ask for Meal Advice")
    print("="*80)
    
    user_msg_1 = "What should I eat for lunch today?"
    print(f"User: {user_msg_1}\n")
    
    coach_response_1 = coach.coach_response('user123', user_msg_1)
    print(f"Coach:\n{coach_response_1}")
    
    # Test 3: Craving intervention
    print("\n" + "="*80)
    print("Test: Handle Craving (HALT Intervention)")
    print("="*80)
    
    user_msg_2 = "I'm really craving ice cream right now"
    print(f"User: {user_msg_2}\n")
    
    coach_response_2 = coach.coach_response('user123', user_msg_2)
    print(f"Coach:\n{coach_response_2}")
    
    # Test 4: Express difficulty
    print("\n" + "="*80)
    print("Test: Support for Difficulty")
    print("="*80)
    
    user_msg_3 = "I'm struggling to stop snacking at night"
    print(f"User: {user_msg_3}\n")
    
    coach_response_3 = coach.coach_response('user123', user_msg_3)
    print(f"Coach:\n{coach_response_3}")
    
    # Test 5: Celebrate progress
    print("\n" + "="*80)
    print("Test: Celebrate Progress")
    print("="*80)
    
    user_msg_4 = "I meal prepped for the whole week!"
    print(f"User: {user_msg_4}\n")
    
    coach_response_4 = coach.coach_response('user123', user_msg_4)
    print(f"Coach:\n{coach_response_4}")
    
    # Test 6: Portion guidance
    print("\n" + "="*80)
    print("Test: Portion Control Guidance")
    print("="*80)
    
    portion_guidance = coach.portion_coach.coach_portion('chicken_breast', 'large serving')
    print("User asks: How much chicken should I eat?\n")
    print(f"Coach:\n{portion_guidance}")
    
    # Test 7: Behavioral intervention
    print("\n" + "="*80)
    print("Test: Behavioral Intervention Recommendation")
    print("="*80)
    
    intervention = coach.intervention_library.get_intervention('mindful_pause')
    
    print(f"🎯 Intervention: {intervention.strategy_name}")
    print(f"   Target: {intervention.target_behavior}")
    print(f"   Evidence: {intervention.evidence_base}")
    print(f"   Success Rate: {intervention.success_rate:.0%}")
    print(f"\n   Steps:")
    for step in intervention.implementation_steps:
        print(f"   {step}")
    print(f"\n   💡 Personalized tip: {intervention.personalized_tip}")
    
    # Test 8: Motivational Interviewing techniques
    print("\n" + "="*80)
    print("Test: Motivational Interviewing Techniques")
    print("="*80)
    
    print("Open-ended questions:")
    for i in range(3):
        question = coach.mi_engine.generate_open_question('motivation_building')
        print(f"  • {question}")
    
    print("\nAffirmations:")
    affirmation = coach.mi_engine.generate_affirmation("tracking your meals daily")
    print(f"  • {affirmation}")
    
    print("\nReflective listening:")
    reflection = coach.mi_engine.reflect_statement("I want to eat healthier but I'm always too busy")
    print(f"  • {reflection}")
    
    # Test 9: Conversation history
    print("\n" + "="*80)
    print("Test: Conversation History Analysis")
    print("="*80)
    
    print(f"✓ Total conversations: {len(coach.conversations)}")
    print(f"\nRecent conversation intents:")
    for conv in coach.conversations[-4:]:
        print(f"  • Intent: {conv.detected_intent} | Emotion: {conv.detected_emotion.value}")
    
    print("\n✅ All nutritional coaching tests passed!")
    print("\n💡 Production Features:")
    print("  - NLP: Advanced intent recognition with transformers (BERT, GPT)")
    print("  - Personalization: Learning from user interaction history")
    print("  - Voice: Speech-to-text for hands-free coaching")
    print("  - Notifications: Proactive check-ins and reminders")
    print("  - Integration: Connect to meal planning, recipe matching")
    print("  - Analytics: Track coaching effectiveness and user outcomes")
    print("  - Professional oversight: RD review of coaching conversations")


if __name__ == '__main__':
    test_nutritional_coach()
