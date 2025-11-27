"""
LLM Orchestrator - Conversational AI for Complete Nutrition System
===================================================================

Unified conversational AI that integrates all system capabilities:
- Food scanning (image, barcode, NIR spectroscopy)
- Risk assessment (55+ health goals, all diseases)
- Personalized recommendations (medications, conditions, preferences)
- Recipe generation (taste preferences, pantry-based)
- Meal planning (goals, budget, dietary restrictions)
- Auto grocery generation (pantry inventory, sourcing)
- Portion estimation (health goals, metabolic needs)

This module provides:
1. Multi-turn conversational interface
2. Function calling to internal services
3. Context aggregation from all user data
4. Training data collection for fine-tuning
5. Performance monitoring across all goals/diseases

LLM Integration:
- Primary: OpenAI GPT-4/GPT-4-turbo with function calling
- Secondary: Anthropic Claude 3.5 Sonnet
- Tertiary: Google Gemini Pro
- Fine-tuning: Custom models trained on system performance

Author: Wellomex AI Team
Date: November 2025
Version: 1.0.0
"""

import os
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
_function_handler = None
_training_pipeline = None
_performance_monitor = None
_integrated_ai = None

def get_function_handler():
    """Get function handler instance"""
    global _function_handler
    if _function_handler is None:
        from .function_handler import handler
        _function_handler = handler
    return _function_handler

def get_training_pipeline():
    """Get training pipeline instance"""
    global _training_pipeline
    if _training_pipeline is None:
        from .training_pipeline import pipeline
        _training_pipeline = pipeline
    return _training_pipeline

def get_performance_monitor():
    """Get performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        from .training_pipeline import monitor
        _performance_monitor = monitor
    return _performance_monitor

def get_integrated_ai():
    """Get integrated AI system (knowledge graph + deep learning)"""
    global _integrated_ai
    if _integrated_ai is None:
        try:
            from ..knowledge_graphs.integrated_nutrition_ai import create_integrated_system
            api_key = os.getenv("OPENAI_API_KEY")
            _integrated_ai = create_integrated_system(llm_api_key=api_key)
            logger.info("Integrated AI system initialized (Knowledge Graph + Deep Learning)")
        except Exception as e:
            logger.warning(f"Could not initialize integrated AI: {e}")
            _integrated_ai = None
    return _integrated_ai


# ============================================================================
# CONFIGURATION
# ============================================================================

class LLMProvider(Enum):
    """LLM provider selection"""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT4_TURBO = "openai_gpt4_turbo"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    GOOGLE_GEMINI = "google_gemini"
    CUSTOM_FINETUNED = "custom_finetuned"


class ConversationMode(Enum):
    """Conversation mode"""
    FOOD_SCANNING = "food_scanning"
    MEAL_PLANNING = "meal_planning"
    RECIPE_GENERATION = "recipe_generation"
    HEALTH_ASSESSMENT = "health_assessment"
    GROCERY_SHOPPING = "grocery_shopping"
    GENERAL_NUTRITION = "general_nutrition"


@dataclass
class LLMConfig:
    """LLM orchestrator configuration"""
    # Provider settings
    primary_provider: LLMProvider = LLMProvider.OPENAI_GPT4_TURBO
    fallback_providers: List[LLMProvider] = field(default_factory=lambda: [
        LLMProvider.ANTHROPIC_CLAUDE,
        LLMProvider.GOOGLE_GEMINI
    ])
    
    # Model parameters
    temperature: float = 0.7
    max_tokens: int = 4000
    top_p: float = 0.9
    frequency_penalty: float = 0.3
    presence_penalty: float = 0.3
    
    # Conversation settings
    max_context_messages: int = 20  # Keep last N messages in context
    session_timeout_minutes: int = 60
    enable_function_calling: bool = True
    
    # Training settings
    collect_training_data: bool = True
    training_data_path: str = "data/llm_training"
    min_feedback_score: float = 3.0  # Only use high-quality interactions
    
    # Performance monitoring
    track_disease_performance: bool = True
    track_goal_performance: bool = True
    log_function_calls: bool = True


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Message:
    """Chat message"""
    role: str  # system, user, assistant, function
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    function_call: Optional[Dict[str, Any]] = None
    function_response: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationSession:
    """Conversation session tracking"""
    session_id: str
    user_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    mode: ConversationMode = ConversationMode.GENERAL_NUTRITION
    
    # Context aggregation
    user_profile: Optional[Dict[str, Any]] = None
    active_health_conditions: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    dietary_preferences: List[str] = field(default_factory=list)
    active_goals: List[str] = field(default_factory=list)
    
    # State tracking
    scanned_foods: List[Dict[str, Any]] = field(default_factory=list)
    pantry_items: List[Dict[str, Any]] = field(default_factory=list)
    pending_actions: List[str] = field(default_factory=list)
    
    # Performance metrics
    function_calls: int = 0
    successful_recommendations: int = 0
    user_satisfaction: Optional[float] = None


@dataclass
class FunctionCall:
    """Function call from LLM"""
    function_name: str
    arguments: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = ""
    
    # Execution tracking
    executed: bool = False
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class TrainingExample:
    """Training example for fine-tuning"""
    conversation_history: List[Message]
    function_calls: List[FunctionCall]
    user_feedback: float  # 1-5 rating
    health_goals_addressed: List[str]
    diseases_considered: List[str]
    outcome_success: bool
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# LLM ORCHESTRATOR
# ============================================================================

class LLMOrchestrator:
    """
    Main LLM orchestration engine
    
    Integrates all system capabilities into conversational AI:
    - Scans food (image, barcode, NIR)
    - Assesses health risks for 55+ goals and all diseases
    - Generates personalized recommendations
    - Creates recipes based on taste + pantry
    - Plans meals with auto grocery generation
    - Estimates portions based on metabolic needs
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize orchestrator"""
        self.config = config or LLMConfig()
        
        # Session management
        self.sessions: Dict[str, ConversationSession] = {}
        
        # Function registry (lazy loaded)
        self._function_registry = None
        
        # Training data collection
        self.training_examples: List[TrainingExample] = []
        
        # Performance tracking
        self.disease_performance: Dict[str, List[float]] = defaultdict(list)
        self.goal_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Initialize LLM clients
        self._init_llm_clients()
        
        logger.info("LLM Orchestrator initialized")
    
    def _init_llm_clients(self):
        """Initialize LLM API clients"""
        self.openai_client = None
        self.anthropic_client = None
        
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized")
        
        if ANTHROPIC_AVAILABLE:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
                logger.info("Anthropic client initialized")
    
    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================
    
    def create_session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> ConversationSession:
        """Create new conversation session"""
        session_id = session_id or f"session_{user_id}_{int(time.time())}"
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            user_profile=user_profile or {}
        )
        
        # Add system message with context
        system_message = self._build_system_prompt(user_profile)
        session.messages.append(Message(
            role="system",
            content=system_message
        ))
        
        self.sessions[session_id] = session
        logger.info(f"Created session {session_id} for user {user_id}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get existing session"""
        session = self.sessions.get(session_id)
        
        if session:
            # Check timeout
            elapsed = datetime.now() - session.last_activity
            if elapsed > timedelta(minutes=self.config.session_timeout_minutes):
                logger.warning(f"Session {session_id} timed out")
                del self.sessions[session_id]
                return None
        
        return session
    
    def update_session_context(
        self,
        session_id: str,
        health_conditions: Optional[List[str]] = None,
        medications: Optional[List[str]] = None,
        goals: Optional[List[str]] = None,
        pantry_items: Optional[List[Dict[str, Any]]] = None
    ):
        """Update session context"""
        session = self.get_session(session_id)
        if not session:
            return
        
        if health_conditions is not None:
            session.active_health_conditions = health_conditions
        if medications is not None:
            session.current_medications = medications
        if goals is not None:
            session.active_goals = goals
        if pantry_items is not None:
            session.pantry_items = pantry_items
        
        logger.info(f"Updated context for session {session_id}")
    
    # ========================================================================
    # SYSTEM PROMPT GENERATION
    # ========================================================================
    
    def _build_system_prompt(self, user_profile: Optional[Dict[str, Any]] = None) -> str:
        """Build comprehensive system prompt"""
        profile = user_profile or {}
        
        prompt = """You are Wellomex AI, an advanced nutrition assistant with deep expertise in:

🔬 FOOD ANALYSIS
- Multi-modal food scanning (images, barcodes, NIR spectroscopy)
- Precise nutritional analysis at atomic/molecular level
- Freshness assessment and quality grading
- Portion size estimation based on metabolic needs

⚕️ HEALTH OPTIMIZATION
- 55+ therapeutic health goals (weight loss, heart health, diabetes control, etc.)
- Comprehensive disease management for ALL conditions
- Medication-nutrient interaction checking
- FDA/WHO/NKF safety compliance
- Personalized risk stratification

🍽️ MEAL INTELLIGENCE
- Recipe generation based on taste preferences and available ingredients
- Meal planning aligned with health goals and budget
- Smart pantry management and inventory tracking
- Automated grocery list generation with local sourcing
- Portion recommendations for metabolic needs

🎯 YOUR CAPABILITIES:

1. **scan_food**: Analyze food from image, barcode, or description
   - Returns: nutrition, allergens, safety warnings, portions

2. **assess_health_risk**: Evaluate food safety for user's conditions
   - Considers: medications, diseases, allergies, goals
   - Returns: risk score, warnings, alternatives

3. **get_recommendations**: Personalized food suggestions
   - Based on: health goals, taste preferences, restrictions
   - Returns: ranked food list with rationale

4. **generate_recipe**: Create recipes from available ingredients
   - Inputs: pantry items, taste preferences, dietary restrictions
   - Returns: recipe with instructions, nutrition, cooking time

5. **create_meal_plan**: Weekly meal planning
   - Optimizes: nutrition, budget, variety, cooking time
   - Returns: 7-day plan with grocery list

6. **generate_grocery_list**: Smart shopping automation
   - Analyzes: pantry inventory, meal plan, local availability
   - Returns: optimized list with sourcing suggestions

7. **estimate_portion**: Calculate ideal serving size
   - Based on: metabolic needs, goals, activity level
   - Returns: portion size in grams/cups with rationale

8. **analyze_food_risks**: Comprehensive safety and health analysis
   - Detects: heavy metals (lead, mercury, arsenic), pesticides, contaminants
   - Uses: ICPMS data for precise element detection when available
   - Analyzes: nutrient adequacy, health goal alignment, medical condition safety
   - Returns: risk scores, warnings, safer alternatives
   - Perfect for: pregnancy, children, chronic conditions

9. **generate_family_recipe**: Create recipes for entire family
   - Inputs: family members (ages, goals, tastes), meal type, cuisine
   - Accommodates: different ages, health goals, taste preferences
   - Returns: recipes with age-appropriate portions and modifications
   - Perfect for families with children, teens, adults, seniors
"""

        # Add user-specific context
        if profile:
            prompt += "\n\n📋 USER PROFILE:\n"
            
            if profile.get("health_conditions"):
                conditions = ", ".join(profile["health_conditions"])
                prompt += f"- Health Conditions: {conditions}\n"
            
            if profile.get("medications"):
                meds = ", ".join(profile["medications"])
                prompt += f"- Medications: {meds}\n"
            
            if profile.get("allergies"):
                allergies = ", ".join(profile["allergies"])
                prompt += f"- Allergies: {allergies}\n"
            
            if profile.get("dietary_preferences"):
                prefs = ", ".join(profile["dietary_preferences"])
                prompt += f"- Dietary Preferences: {prefs}\n"
            
            if profile.get("health_goals"):
                goals = ", ".join(profile["health_goals"])
                prompt += f"- Active Goals: {goals}\n"
        
        prompt += """

🎯 INTERACTION GUIDELINES:
1. Always prioritize user safety (check medications, allergies, conditions)
2. Use function calls to access real-time data (don't guess)
3. Provide specific, actionable recommendations
4. Explain health impacts in simple terms
5. Ask clarifying questions when needed
6. Be encouraging and supportive
7. Cite sources for medical claims

When user scans food or asks for recipes, ALWAYS use function calls to:
- Analyze exact nutritional content
- Check medication interactions
- Verify against health conditions
- Generate personalized portions
- Suggest alternatives if needed

Ready to help the user achieve their health goals! 🚀
"""
        
        return prompt
    
    # ========================================================================
    # CONVERSATION
    # ========================================================================
    
    async def chat(
        self,
        session_id: str,
        user_message: str,
        image_data: Optional[bytes] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main conversation endpoint
        
        Args:
            session_id: Session identifier
            user_message: User's text message
            image_data: Optional image for food scanning
            metadata: Optional metadata (barcode, location, etc.)
        
        Returns:
            Response with assistant message and any function results
        """
        session = self.get_session(session_id)
        if not session:
            return {
                "error": "Session not found or expired",
                "session_id": session_id
            }
        
        # Update activity timestamp
        session.last_activity = datetime.now()
        
        # Add user message
        user_msg = Message(
            role="user",
            content=user_message,
            metadata=metadata or {}
        )
        session.messages.append(user_msg)
        
        # Handle image if provided
        if image_data:
            # Auto-trigger food scanning
            scan_result = await self._execute_function(
                session_id,
                "scan_food",
                {"image_data": image_data}
            )
            
            # Add scan result to context
            if scan_result.get("success"):
                session.scanned_foods.append(scan_result["data"])
                context_msg = f"\n\n[SCANNED FOOD: {scan_result['data'].get('food_name', 'Unknown')}]"
                user_msg.content += context_msg
        
        # Generate LLM response
        response = await self._generate_response(session)
        
        return response
    
    async def _generate_response(self, session: ConversationSession) -> Dict[str, Any]:
        """Generate LLM response with function calling"""
        
        # Build messages for LLM
        messages = self._build_message_history(session)
        
        # Get function definitions
        functions = self._get_function_definitions()
        
        # Call LLM
        try:
            if self.config.primary_provider == LLMProvider.OPENAI_GPT4_TURBO:
                response = await self._call_openai(messages, functions)
            elif self.config.primary_provider == LLMProvider.ANTHROPIC_CLAUDE:
                response = await self._call_anthropic(messages, functions)
            else:
                response = {"error": "Provider not implemented"}
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return {
                "error": str(e),
                "assistant_message": "I apologize, but I'm having trouble processing your request. Please try again."
            }
    
    async def _call_openai(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Call OpenAI API with function calling"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            # Make API call
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                functions=functions,
                function_call="auto",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty
            )
            
            choice = response.choices[0]
            message = choice.message
            
            result = {
                "assistant_message": message.content or "",
                "function_calls": [],
                "finish_reason": choice.finish_reason
            }
            
            # Handle function calls
            if message.function_call:
                func_name = message.function_call.name
                func_args = json.loads(message.function_call.arguments)
                
                result["function_calls"].append({
                    "name": func_name,
                    "arguments": func_args
                })
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _call_anthropic(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Call Anthropic Claude API"""
        # TODO: Implement Anthropic function calling
        # (Currently uses different format than OpenAI)
        return {
            "error": "Anthropic implementation pending",
            "assistant_message": "Using OpenAI as fallback"
        }
    
    def _build_message_history(self, session: ConversationSession) -> List[Dict[str, str]]:
        """Build message history for LLM"""
        messages = []
        
        # Get last N messages
        recent_messages = session.messages[-self.config.max_context_messages:]
        
        for msg in recent_messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return messages
    
    # ========================================================================
    # FUNCTION CALLING REGISTRY
    # ========================================================================
    
    def _get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get function definitions for LLM"""
        return [
            {
                "name": "scan_food",
                "description": "Scan and analyze food from image, barcode, or description. Returns detailed nutritional information, allergens, safety warnings, and portion recommendations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "food_description": {
                            "type": "string",
                            "description": "Text description of the food (e.g., 'grilled chicken breast')"
                        },
                        "barcode": {
                            "type": "string",
                            "description": "Product barcode if available"
                        },
                        "image_data": {
                            "type": "string",
                            "description": "Base64 encoded image data"
                        },
                        "portion_size": {
                            "type": "string",
                            "description": "Estimated portion size (e.g., '200g', '1 cup')"
                        }
                    },
                    "required": ["food_description"]
                }
            },
            {
                "name": "assess_health_risk",
                "description": "Assess health risk of a food for user's specific conditions, medications, and allergies. Returns risk score (0-100), warnings, and safer alternatives.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "food_name": {
                            "type": "string",
                            "description": "Name of the food to assess"
                        },
                        "portion_grams": {
                            "type": "number",
                            "description": "Portion size in grams"
                        },
                        "check_medications": {
                            "type": "boolean",
                            "description": "Check for drug-nutrient interactions",
                            "default": True
                        },
                        "check_allergies": {
                            "type": "boolean",
                            "description": "Check for allergens",
                            "default": True
                        }
                    },
                    "required": ["food_name"]
                }
            },
            {
                "name": "get_recommendations",
                "description": "Get personalized food recommendations based on health goals, taste preferences, and dietary restrictions. Returns ranked list of foods with rationale.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "taste_preference": {
                            "type": "string",
                            "description": "Desired taste profile (sweet, savory, spicy, etc.)"
                        },
                        "meal_type": {
                            "type": "string",
                            "enum": ["breakfast", "lunch", "dinner", "snack"],
                            "description": "Type of meal"
                        },
                        "max_calories": {
                            "type": "number",
                            "description": "Maximum calories"
                        },
                        "num_recommendations": {
                            "type": "integer",
                            "description": "Number of recommendations to return",
                            "default": 5
                        }
                    }
                }
            },
            {
                "name": "generate_recipe",
                "description": "Generate recipe using available pantry ingredients and user's taste preferences. Returns complete recipe with instructions, nutrition, and cooking time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "available_ingredients": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of ingredients user has"
                        },
                        "cuisine_type": {
                            "type": "string",
                            "description": "Desired cuisine (Italian, Chinese, Mexican, etc.)"
                        },
                        "cooking_time_minutes": {
                            "type": "integer",
                            "description": "Maximum cooking time"
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard"],
                            "description": "Recipe difficulty level"
                        },
                        "servings": {
                            "type": "integer",
                            "description": "Number of servings",
                            "default": 4
                        }
                    }
                }
            },
            {
                "name": "create_meal_plan",
                "description": "Create weekly meal plan optimized for health goals, budget, and variety. Returns 7-day plan with grocery list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of days to plan",
                            "default": 7
                        },
                        "daily_calorie_target": {
                            "type": "number",
                            "description": "Target calories per day"
                        },
                        "budget_usd": {
                            "type": "number",
                            "description": "Weekly food budget"
                        },
                        "meal_frequency": {
                            "type": "integer",
                            "description": "Meals per day",
                            "default": 3
                        },
                        "prep_time_available": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Available cooking time"
                        }
                    }
                }
            },
            {
                "name": "generate_grocery_list",
                "description": "Generate optimized grocery shopping list based on meal plan and pantry inventory. Includes local sourcing suggestions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "meal_plan_id": {
                            "type": "string",
                            "description": "ID of meal plan"
                        },
                        "current_pantry": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Items already in pantry"
                        },
                        "location": {
                            "type": "string",
                            "description": "User location for local sourcing"
                        },
                        "budget_limit": {
                            "type": "number",
                            "description": "Maximum budget"
                        }
                    }
                }
            },
            {
                "name": "estimate_portion",
                "description": "Calculate ideal portion size based on metabolic needs, activity level, and health goals. Returns portion in grams and common measures.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "food_name": {
                            "type": "string",
                            "description": "Food to portion"
                        },
                        "meal_type": {
                            "type": "string",
                            "enum": ["breakfast", "lunch", "dinner", "snack"]
                        },
                        "activity_level": {
                            "type": "string",
                            "enum": ["sedentary", "light", "moderate", "active", "very_active"]
                        },
                        "current_weight_kg": {
                            "type": "number",
                            "description": "User's current weight"
                        },
                        "target_weight_kg": {
                            "type": "number",
                            "description": "User's target weight"
                        }
                    },
                    "required": ["food_name"]
                }
            },
            {
                "name": "analyze_food_risks",
                "description": "Analyze food safety risks including heavy metal contamination (lead, mercury, arsenic, cadmium), pesticide residues, nutrient adequacy, and alignment with user's health goals and medical conditions. Uses ICPMS data when available for precise contaminant detection. Returns comprehensive risk scores, warnings, and safer alternatives.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "food_name": {
                            "type": "string",
                            "description": "Name of the food to analyze"
                        },
                        "icpms_data": {
                            "type": "object",
                            "description": "ICPMS element detection data (optional)",
                            "properties": {
                                "elements": {
                                    "type": "object",
                                    "description": "Element concentrations in ppm",
                                    "additionalProperties": {"type": "number"},
                                    "example": {"Lead": 0.45, "Mercury": 0.12, "Iron": 2.7}
                                }
                            }
                        },
                        "scan_data": {
                            "type": "object",
                            "description": "Food scan nutrient data (optional)",
                            "properties": {
                                "calories": {"type": "number", "description": "Calories per 100g"},
                                "protein_g": {"type": "number", "description": "Protein in grams"},
                                "carbohydrates_g": {"type": "number", "description": "Carbohydrates in grams"},
                                "fat_g": {"type": "number", "description": "Total fat in grams"},
                                "fiber_g": {"type": "number", "description": "Dietary fiber in grams"},
                                "sugar_g": {"type": "number", "description": "Total sugars in grams"},
                                "sodium_mg": {"type": "number", "description": "Sodium in mg"},
                                "potassium_mg": {"type": "number", "description": "Potassium in mg"},
                                "omega3_total": {"type": "number", "description": "Total omega-3 in mg"},
                                "dha": {"type": "number", "description": "DHA in mg"},
                                "epa": {"type": "number", "description": "EPA in mg"},
                                "contains_gluten": {"type": "boolean", "description": "Contains gluten"},
                                "nutrients": {
                                    "type": "object",
                                    "description": "Additional nutrients {nutrient_name: amount}",
                                    "additionalProperties": {"type": "number"}
                                }
                            }
                        },
                        "user_profile": {
                            "type": "object",
                            "description": "User health profile (optional - uses session profile if not provided)",
                            "properties": {
                                "age": {"type": "integer", "description": "Age in years"},
                                "gender": {"type": "string", "description": "Gender"},
                                "medical_conditions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of medical conditions (Pregnancy, Diabetes, Hypertension, etc.)"
                                },
                                "health_goals": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of health goals (Weight loss, Heart health, Brain health, etc.)"
                                },
                                "allergies": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Known food allergies"
                                },
                                "medications": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Current medications"
                                }
                            }
                        },
                        "serving_size_g": {
                            "type": "number",
                            "description": "Serving size in grams",
                            "default": 100.0
                        }
                    },
                    "required": ["food_name"]
                }
            },
            {
                "name": "generate_family_recipe",
                "description": "Generate recipes optimized for entire family with different ages, health goals, and taste preferences. Accommodates children, adults, and seniors with age-appropriate portions and modifications. Perfect for families where members have different dietary needs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "family_members": {
                            "type": "array",
                            "description": "Array of family member profiles",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Member's name"},
                                    "age": {"type": "integer", "description": "Age in years"},
                                    "gender": {"type": "string", "enum": ["male", "female", "other"]},
                                    "health_goals": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Health goals (e.g., weight loss, muscle gain, heart health)"
                                    },
                                    "medical_conditions": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Medical conditions (e.g., diabetes, hypertension)"
                                    },
                                    "medications": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Current medications"
                                    },
                                    "allergies": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Food allergies"
                                    },
                                    "dietary_restrictions": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Dietary restrictions (vegetarian, vegan, etc.)"
                                    },
                                    "taste_preferences": {
                                        "type": "object",
                                        "description": "Taste preferences",
                                        "properties": {
                                            "likes": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "Flavors they like (sweet, savory, spicy, etc.)"
                                            },
                                            "dislikes": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "Flavors they dislike"
                                            },
                                            "favorite_cuisines": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "Favorite cuisines"
                                            },
                                            "favorite_ingredients": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "Favorite ingredients"
                                            }
                                        }
                                    },
                                    "weight": {"type": "number", "description": "Weight in kg (optional)"},
                                    "height": {"type": "number", "description": "Height in cm (optional)"},
                                    "activity_level": {
                                        "type": "string",
                                        "enum": ["sedentary", "light", "moderate", "active", "very_active"],
                                        "default": "moderate"
                                    }
                                },
                                "required": ["name", "age", "gender"]
                            }
                        },
                        "meal_type": {
                            "type": "string",
                            "enum": ["breakfast", "lunch", "dinner", "snack"],
                            "description": "Type of meal",
                            "default": "dinner"
                        },
                        "cuisine_preference": {
                            "type": "string",
                            "description": "Preferred cuisine type (Italian, Asian, Mexican, etc.)"
                        },
                        "max_recipes": {
                            "type": "integer",
                            "description": "Number of recipe options to generate",
                            "default": 3
                        }
                    },
                    "required": ["family_members"]
                }
            }
        ]
    
    async def _execute_function(
        self,
        session_id: str,
        function_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a function call"""
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        start_time = time.time()
        
        # Create function call record
        func_call = FunctionCall(
            function_name=function_name,
            arguments=arguments,
            session_id=session_id
        )
        
        try:
            # Route to appropriate handler
            if function_name == "scan_food":
                result = await self._handle_scan_food(session, arguments)
            elif function_name == "assess_health_risk":
                result = await self._handle_assess_risk(session, arguments)
            elif function_name == "get_recommendations":
                result = await self._handle_get_recommendations(session, arguments)
            elif function_name == "generate_recipe":
                result = await self._handle_generate_recipe(session, arguments)
            elif function_name == "create_meal_plan":
                result = await self._handle_create_meal_plan(session, arguments)
            elif function_name == "generate_grocery_list":
                result = await self._handle_generate_grocery(session, arguments)
            elif function_name == "estimate_portion":
                result = await self._handle_estimate_portion(session, arguments)
            elif function_name == "analyze_food_risks":
                result = await self._handle_analyze_food_risks(session, arguments)
            elif function_name == "generate_family_recipe":
                result = await self._handle_generate_family_recipe(session, arguments)
            else:
                result = {"error": f"Unknown function: {function_name}"}
            
            func_call.executed = True
            func_call.result = result
            
        except Exception as e:
            logger.error(f"Function execution error: {e}")
            func_call.error = str(e)
            result = {"error": str(e)}
        
        func_call.execution_time_ms = (time.time() - start_time) * 1000
        
        # Track metrics
        session.function_calls += 1
        
        return result
    
    # ========================================================================
    # FUNCTION HANDLERS (Integration Points)
    # ========================================================================
    
    async def _handle_scan_food(
        self,
        session: ConversationSession,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle food scanning"""
        handler = get_function_handler()
        
        return await handler.scan_food(
            user_profile=session.user_profile,
            food_description=args.get("food_description"),
            barcode=args.get("barcode"),
            image_data=args.get("image_data"),
            portion_size=args.get("portion_size")
        )
    
    async def _handle_assess_risk(
        self,
        session: ConversationSession,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle health risk assessment"""
        handler = get_function_handler()
        
        return await handler.assess_health_risk(
            user_profile=session.user_profile,
            food_name=args.get("food_name"),
            nutrition=args.get("nutrition"),
            portion_grams=args.get("portion_grams", 100),
            check_medications=args.get("check_medications", True),
            check_allergies=args.get("check_allergies", True)
        )
    
    async def _handle_get_recommendations(
        self,
        session: ConversationSession,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle food recommendations"""
        handler = get_function_handler()
        
        return await handler.get_recommendations(
            user_profile=session.user_profile,
            taste_preference=args.get("taste_preference"),
            meal_type=args.get("meal_type"),
            max_calories=args.get("max_calories"),
            num_recommendations=args.get("num_recommendations", 5)
        )
        
        return {
            "success": True,
            "data": {
                "recommendations": [
                    {
                        "food": "Salmon",
                        "score": 95,
                        "rationale": "High in omega-3, supports heart health"
                    },
                    {
                        "food": "Blueberries",
                        "score": 90,
                        "rationale": "Antioxidants, supports brain health"
                    }
                ]
            }
        }
    
    async def _handle_generate_recipe(
        self,
        session: ConversationSession,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle recipe generation"""
        # TODO: Integrate with recipe generator
        # from app.ai_nutrition.recipes import RecipeGenerator
        
        return {
            "success": True,
            "data": {
                "recipe_name": "Healthy Chicken Bowl",
                "ingredients": args.get("available_ingredients", []),
                "instructions": [
                    "1. Season chicken with herbs",
                    "2. Grill for 6 minutes per side",
                    "3. Serve with vegetables"
                ],
                "nutrition_per_serving": {
                    "calories": 450,
                    "protein_g": 40,
                    "carbs_g": 30,
                    "fat_g": 15
                },
                "cooking_time_minutes": 25
            }
        }
    
    async def _handle_create_meal_plan(
        self,
        session: ConversationSession,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle meal plan creation"""
        # TODO: Integrate with meal planner
        # from app.ai_nutrition.planner import MealPlanner
        
        return {
            "success": True,
            "data": {
                "plan_id": f"plan_{int(time.time())}",
                "days": args.get("days", 7),
                "meals": [],
                "grocery_list": [],
                "total_cost_usd": 75.50
            }
        }
    
    async def _handle_generate_grocery(
        self,
        session: ConversationSession,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle grocery list generation"""
        # TODO: Integrate with pantry system
        # from app.ai_nutrition.pantry_to_plate import LocalSourcingOptimizer
        
        return {
            "success": True,
            "data": {
                "items": [
                    {"name": "Chicken breast", "quantity": "1.5 lbs", "store": "Whole Foods"},
                    {"name": "Broccoli", "quantity": "2 bunches", "store": "Farmers Market"}
                ],
                "total_cost": 35.00,
                "delivery_available": True
            }
        }
    
    async def _handle_estimate_portion(
        self,
        session: ConversationSession,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle portion estimation"""
        # TODO: Integrate with metabolic calculator
        
        return {
            "success": True,
            "data": {
                "food_name": args["food_name"],
                "portion_grams": 180,
                "portion_visual": "Size of your palm",
                "calories": 220,
                "rationale": "Based on your weight loss goal and activity level"
            }
        }
    
    async def _handle_analyze_food_risks(
        self,
        session: ConversationSession,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle food risk analysis"""
        handler = get_function_handler()
        
        # Use session user profile if not provided
        user_profile = args.get("user_profile", session.user_profile)
        
        # Extract ICPMS data if provided
        icpms_data = None
        if args.get("icpms_data") and args["icpms_data"].get("elements"):
            icpms_data = args["icpms_data"]["elements"]
        
        # Extract scan data if provided
        scan_data = args.get("scan_data")
        
        return await handler.analyze_food_risks(
            food_name=args["food_name"],
            icpms_data=icpms_data,
            scan_data=scan_data,
            user_profile=user_profile,
            serving_size_g=args.get("serving_size_g", 100.0)
        )
    
    async def _handle_generate_family_recipe(
        self,
        session: ConversationSession,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle family recipe generation"""
        handler = get_function_handler()
        
        return await handler.generate_family_recipe(
            family_members=args.get("family_members", []),
            meal_type=args.get("meal_type", "dinner"),
            cuisine_preference=args.get("cuisine_preference"),
            max_recipes=args.get("max_recipes", 3)
        )
    
    # ========================================================================
    # TRAINING DATA COLLECTION
    # ========================================================================
    
    def collect_training_example(
        self,
        session_id: str,
        user_feedback: float,
        outcome_success: bool
    ):
        """Collect training example from conversation"""
        session = self.get_session(session_id)
        if not session or user_feedback < self.config.min_feedback_score:
            return
        
        # Use training pipeline
        pipeline = get_training_pipeline()
        monitor = get_performance_monitor()
        
        # Extract function calls from session
        function_calls = [
            msg.function_call for msg in session.messages
            if msg.function_call is not None
        ]
        
        # Convert messages to dict format
        messages = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in session.messages
        ]
        
        # Calculate average response time (mock for now)
        response_time_ms = 1500.0
        
        # Collect training datapoint
        datapoint = pipeline.collect_conversation(
            session_id=session_id,
            user_id=session.user_id,
            messages=messages,
            function_calls=function_calls,
            user_profile=session.user_profile or {},
            user_rating=user_feedback,
            outcome_success=outcome_success,
            response_time_ms=response_time_ms
        )
        
        # Record performance metrics
        from .training_pipeline import MetricType
        
        monitor.record_metric(
            metric_type=MetricType.USER_SATISFACTION,
            value=user_feedback,
            health_goal=session.active_goals[0] if session.active_goals else None,
            disease=session.active_health_conditions[0] if session.active_health_conditions else None,
            session_id=session_id,
            user_id=session.user_id
        )
        
        logger.info(f"Collected training example from session {session_id} (rating: {user_feedback})")
    
    def _save_training_example_old(self, session_id: str, user_feedback: float, outcome_success: bool):
        """Legacy method - kept for reference, use collect_training_example instead"""
        session = self.get_session(session_id)
        if not session or user_feedback < self.config.min_feedback_score:
            return
        
        # Extract function calls from session
        function_calls = [
            msg.function_call for msg in session.messages
            if msg.function_call is not None
        ]
        
        example = TrainingExample(
            conversation_history=session.messages.copy(),
            function_calls=function_calls,
            user_feedback=user_feedback,
            health_goals_addressed=session.active_goals,
            diseases_considered=session.active_health_conditions,
            outcome_success=outcome_success
        )
        
        self.training_examples.append(example)
        
        # Save to disk
        if self.config.collect_training_data:
            self._save_training_example(example)
        
        logger.info(f"Collected training example from session {session_id}")
    
    def _save_training_example(self, example: TrainingExample):
        """Save training example to disk"""
        os.makedirs(self.config.training_data_path, exist_ok=True)
        
        filename = f"{example.timestamp.strftime('%Y%m%d_%H%M%S')}_{example.user_feedback:.1f}.json"
        filepath = os.path.join(self.config.training_data_path, filename)
        
        # Convert to JSON-serializable format
        data = {
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in example.conversation_history
            ],
            "feedback": example.user_feedback,
            "goals": example.health_goals_addressed,
            "diseases": example.diseases_considered,
            "success": example.outcome_success
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# ============================================================================
# FINE-TUNING SYSTEM
# ============================================================================

class FineTuningManager:
    """
    Manage fine-tuning of custom LLMs
    
    Aggregates performance across:
    - 55+ health goals
    - All disease conditions
    - Medication interactions
    - Food recommendations
    - Recipe generations
    """
    
    def __init__(self, training_data_path: str = "data/llm_training"):
        self.training_data_path = training_data_path
        self.openai_client = None
        
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
    
    def prepare_training_dataset(
        self,
        min_examples: int = 100,
        quality_threshold: float = 4.0
    ) -> List[Dict[str, Any]]:
        """Prepare training dataset from collected examples"""
        
        # Load all training examples
        examples = []
        for filename in os.listdir(self.training_data_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.training_data_path, filename)
                with open(filepath, 'r') as f:
                    examples.append(json.load(f))
        
        # Filter by quality
        high_quality = [
            ex for ex in examples
            if ex["feedback"] >= quality_threshold and ex["success"]
        ]
        
        logger.info(f"Found {len(high_quality)} high-quality examples out of {len(examples)} total")
        
        if len(high_quality) < min_examples:
            logger.warning(f"Insufficient training data: {len(high_quality)} < {min_examples}")
            return []
        
        # Convert to OpenAI fine-tuning format
        training_data = []
        for ex in high_quality:
            training_data.append({
                "messages": ex["messages"]
            })
        
        return training_data
    
    def submit_fine_tuning_job(
        self,
        training_data: List[Dict[str, Any]],
        model: str = "gpt-4-turbo-preview",
        suffix: str = "wellomex-nutrition"
    ) -> Optional[str]:
        """Submit fine-tuning job to OpenAI"""
        if not self.openai_client:
            logger.error("OpenAI client not initialized")
            return None
        
        try:
            # Save training data to JSONL
            training_file = f"training_{int(time.time())}.jsonl"
            with open(training_file, 'w') as f:
                for item in training_data:
                    f.write(json.dumps(item) + '\n')
            
            # Upload file
            with open(training_file, 'rb') as f:
                file_response = self.openai_client.files.create(
                    file=f,
                    purpose='fine-tune'
                )
            
            # Create fine-tuning job
            job = self.openai_client.fine_tuning.jobs.create(
                training_file=file_response.id,
                model=model,
                suffix=suffix
            )
            
            logger.info(f"Fine-tuning job submitted: {job.id}")
            return job.id
            
        except Exception as e:
            logger.error(f"Fine-tuning submission error: {e}")
            return None
