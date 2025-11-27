"""
Conversational Chat API
=======================

FastAPI endpoints for conversational AI nutrition assistant.

Endpoints:
- POST /chat/session: Create new conversation session
- POST /chat/message: Send message to assistant
- POST /chat/scan-and-ask: Scan food image and ask question
- GET /chat/history/{session_id}: Get conversation history
- POST /chat/feedback: Submit feedback for training
- GET /chat/sessions: List user's sessions

Author: Wellomex AI Team
Date: November 2025
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import json
import asyncio

from app.deps import get_current_user, rate_limit
from app.ai_nutrition.orchestration.llm_orchestrator import (
    LLMOrchestrator,
    LLMConfig,
    ConversationMode,
    LLMProvider
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Conversational AI"],
    responses={404: {"description": "Not found"}}
)

# Global orchestrator instance
orchestrator = LLMOrchestrator(LLMConfig())


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class CreateSessionRequest(BaseModel):
    """Request to create conversation session"""
    mode: Optional[ConversationMode] = ConversationMode.GENERAL_NUTRITION
    user_profile: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "mode": "general_nutrition",
                "user_profile": {
                    "health_conditions": ["type2_diabetes", "hypertension"],
                    "medications": ["metformin", "lisinopril"],
                    "allergies": ["peanuts", "shellfish"],
                    "dietary_preferences": ["low_carb", "heart_healthy"],
                    "health_goals": ["weight_loss", "blood_sugar_control"]
                }
            }
        }


class CreateSessionResponse(BaseModel):
    """Response with session details"""
    session_id: str
    user_id: str
    created_at: datetime
    mode: str
    message: str = "Session created successfully"


class SendMessageRequest(BaseModel):
    """Request to send message"""
    session_id: str
    message: str
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_user123_1732123456",
                "message": "I just scanned grilled salmon. Is this good for my diabetes?",
                "metadata": {
                    "location": "home",
                    "meal_time": "dinner"
                }
            }
        }


class MessageResponse(BaseModel):
    """Response with assistant message"""
    session_id: str
    assistant_message: str
    function_calls: List[Dict[str, Any]] = []
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_user123_1732123456",
                "assistant_message": "Excellent choice! Grilled salmon is highly beneficial for managing type 2 diabetes...",
                "function_calls": [
                    {
                        "name": "assess_health_risk",
                        "result": {
                            "risk_score": 15,
                            "risk_level": "very_low",
                            "benefits": ["omega3", "low_glycemic"]
                        }
                    }
                ],
                "timestamp": "2025-11-20T10:30:00"
            }
        }


class ScanAndAskRequest(BaseModel):
    """Request for scan + question"""
    session_id: str
    question: Optional[str] = "What did I scan and is it safe for me?"
    portion_estimate: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_user123_1732123456",
                "question": "How much of this should I eat for dinner?",
                "portion_estimate": "200g"
            }
        }


class FeedbackRequest(BaseModel):
    """Feedback for training"""
    session_id: str
    rating: float = Field(..., ge=1.0, le=5.0, description="Rating 1-5")
    outcome_success: bool = Field(default=True, description="Was outcome successful?")
    comments: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_user123_1732123456",
                "rating": 5.0,
                "outcome_success": True,
                "comments": "Very helpful recommendations!"
            }
        }


class ConversationHistoryResponse(BaseModel):
    """Conversation history"""
    session_id: str
    messages: List[Dict[str, Any]]
    total_messages: int
    function_calls_count: int
    user_satisfaction: Optional[float] = None


class SessionSummary(BaseModel):
    """Session summary"""
    session_id: str
    created_at: datetime
    last_activity: datetime
    mode: str
    message_count: int
    active_goals: List[str]


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "/session",
    response_model=CreateSessionResponse,
    summary="Create Conversation Session",
    description="""
    Create a new conversation session with the AI nutrition assistant.
    
    The session maintains context including:
    - User health profile (conditions, medications, allergies)
    - Active health goals
    - Conversation history
    - Scanned foods and pantry inventory
    
    Sessions expire after 60 minutes of inactivity.
    """
)
async def create_session(
    request: CreateSessionRequest,
    current_user: Dict = Depends(get_current_user)
) -> CreateSessionResponse:
    """Create new conversation session"""
    
    try:
        user_id = current_user["user_id"]
        
        # Create session
        session = orchestrator.create_session(
            user_id=user_id,
            user_profile=request.user_profile
        )
        
        # Update mode
        session.mode = request.mode
        
        logger.info(f"Created session {session.session_id} for user {user_id}")
        
        return CreateSessionResponse(
            session_id=session.session_id,
            user_id=user_id,
            created_at=session.created_at,
            mode=session.mode.value
        )
        
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/message",
    response_model=MessageResponse,
    summary="Send Message",
    description="""
    Send a message to the AI assistant and get a response.
    
    The assistant will:
    1. Understand your question/request
    2. Call appropriate functions (scan food, assess risk, etc.)
    3. Provide personalized recommendations
    4. Consider your health profile and goals
    
    Example questions:
    - "I just ate salmon. Was that a good choice?"
    - "What should I cook for dinner with chicken and broccoli?"
    - "Create a meal plan for this week"
    - "Is this food safe with my medications?"
    """
)
async def send_message(
    request: SendMessageRequest,
    current_user: Dict = Depends(get_current_user),
    _rate_limit: None = Depends(rate_limit)
) -> MessageResponse:
    """Send message to assistant"""
    
    try:
        # Verify session belongs to user
        session = orchestrator.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        if session.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Session does not belong to user")
        
        # Generate response
        response = await orchestrator.chat(
            session_id=request.session_id,
            user_message=request.message,
            metadata=request.metadata
        )
        
        if "error" in response:
            raise HTTPException(status_code=500, detail=response["error"])
        
        return MessageResponse(
            session_id=request.session_id,
            assistant_message=response.get("assistant_message", ""),
            function_calls=response.get("function_calls", []),
            timestamp=datetime.now(),
            metadata=response.get("metadata")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Message processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/scan-and-ask",
    response_model=MessageResponse,
    summary="Scan Food and Ask Question",
    description="""
    Upload food image, scan it, and ask a question about it.
    
    This combines food scanning with conversational AI:
    1. Scans the food image (nutrition, allergens, quality)
    2. Assesses health risk for your conditions
    3. Answers your specific question
    4. Provides portion recommendations
    
    Perfect for:
    - "Is this safe for me?"
    - "How much should I eat?"
    - "What are the health benefits?"
    - "Should I eat this before my workout?"
    """
)
async def scan_and_ask(
    session_id: str = Form(...),
    question: Optional[str] = Form("What did I scan and is it safe for me?"),
    portion_estimate: Optional[str] = Form(None),
    image: UploadFile = File(...),
    current_user: Dict = Depends(get_current_user),
    _rate_limit: None = Depends(rate_limit)
) -> MessageResponse:
    """Scan food image and ask question"""
    
    try:
        # Verify session
        session = orchestrator.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        # Read image data
        image_data = await image.read()
        
        # Build message with image
        metadata = {
            "has_image": True,
            "image_filename": image.filename,
            "portion_estimate": portion_estimate
        }
        
        # Process with LLM
        response = await orchestrator.chat(
            session_id=session_id,
            user_message=question,
            image_data=image_data,
            metadata=metadata
        )
        
        if "error" in response:
            raise HTTPException(status_code=500, detail=response["error"])
        
        return MessageResponse(
            session_id=session_id,
            assistant_message=response.get("assistant_message", ""),
            function_calls=response.get("function_calls", []),
            timestamp=datetime.now(),
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scan and ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/history/{session_id}",
    response_model=ConversationHistoryResponse,
    summary="Get Conversation History",
    description="Retrieve full conversation history for a session"
)
async def get_history(
    session_id: str,
    current_user: Dict = Depends(get_current_user)
) -> ConversationHistoryResponse:
    """Get conversation history"""
    
    try:
        session = orchestrator.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        # Convert messages to dict
        messages = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "function_call": msg.function_call
            }
            for msg in session.messages
        ]
        
        return ConversationHistoryResponse(
            session_id=session_id,
            messages=messages,
            total_messages=len(messages),
            function_calls_count=session.function_calls,
            user_satisfaction=session.user_satisfaction
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/feedback",
    summary="Submit Feedback",
    description="""
    Submit feedback about conversation quality.
    
    This feedback is used to:
    1. Collect training data for fine-tuning
    2. Track performance across health goals and diseases
    3. Improve recommendation quality
    4. Optimize function calling accuracy
    
    High-quality conversations (rating >= 4.0) are used for training.
    """
)
async def submit_feedback(
    request: FeedbackRequest,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, str]:
    """Submit conversation feedback"""
    
    try:
        session = orchestrator.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        # Update session satisfaction
        session.user_satisfaction = request.rating
        
        # Collect training example
        orchestrator.collect_training_example(
            session_id=request.session_id,
            user_feedback=request.rating,
            outcome_success=request.outcome_success
        )
        
        logger.info(f"Feedback submitted for session {request.session_id}: {request.rating}/5.0")
        
        return {
            "status": "success",
            "message": "Thank you for your feedback!"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/sessions",
    response_model=List[SessionSummary],
    summary="List User Sessions",
    description="Get list of all conversation sessions for current user"
)
async def list_sessions(
    current_user: Dict = Depends(get_current_user)
) -> List[SessionSummary]:
    """List user's sessions"""
    
    try:
        user_id = current_user["user_id"]
        
        # Filter sessions for current user
        user_sessions = [
            session for session in orchestrator.sessions.values()
            if session.user_id == user_id
        ]
        
        # Build summaries
        summaries = [
            SessionSummary(
                session_id=session.session_id,
                created_at=session.created_at,
                last_activity=session.last_activity,
                mode=session.mode.value,
                message_count=len(session.messages),
                active_goals=session.active_goals
            )
            for session in user_sessions
        ]
        
        return summaries
        
    except Exception as e:
        logger.error(f"Session listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/stream",
    summary="Stream Message Response (TODO)",
    description="Stream assistant response in real-time (SSE)"
)
async def stream_message(
    request: SendMessageRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Stream message response (TODO: implement SSE streaming)"""
    # TODO: Implement Server-Sent Events streaming for real-time responses
    raise HTTPException(status_code=501, detail="Streaming not yet implemented")
