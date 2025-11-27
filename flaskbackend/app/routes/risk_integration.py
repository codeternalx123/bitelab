"""
Risk Integration API Routes

This module provides RESTful API endpoints for the Dynamic Risk Integration Layer,
connecting chemometric element detection to personalized health risk assessment.

Endpoints:
==========

1. POST /api/v1/risk-integration/assess
   - Assess food safety based on element predictions and user health profile
   - Input: Element predictions + user profile
   - Output: Risk assessment with warnings

2. POST /api/v1/risk-integration/warnings
   - Generate personalized warning messages
   - Modes: consumer, clinical, regulatory
   - Output: Formatted warnings with actionable insights

3. POST /api/v1/risk-integration/alternatives
   - Find safer alternative foods
   - Input: Problematic food + search criteria
   - Output: Ranked alternatives with comparisons

4. GET /api/v1/risk-integration/thresholds/{condition}
   - Get medical thresholds for a health condition
   - Output: Element limits and regulatory citations

5. POST /api/v1/risk-integration/health-profile
   - Create or update user health profile
   - Input: Medical conditions, age, weight, medications
   - Output: Complete health profile with risk stratification

6. GET /api/v1/risk-integration/health-profile/{user_id}
   - Retrieve user health profile
   - Output: Current health profile and conditions

7. POST /api/v1/risk-integration/batch-assess
   - Batch assessment for multiple food samples
   - Input: Array of predictions
   - Output: Array of assessments

8. GET /api/v1/risk-integration/food-database/search
   - Search food database by name or category
   - Output: Foods with element profiles

Author: Wellomex AI Nutrition System
Version: 1.0.0
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# Import risk integration components
import sys
sys.path.append('..')

from app.ai_nutrition.risk_integration.risk_integration_engine import (
    RiskIntegrationEngine,
    ElementPrediction,
    RiskAssessment
)
from app.ai_nutrition.risk_integration.personalized_warning_system import (
    WarningMessageGenerator,
    MessageMode,
    ComprehensiveWarning
)
from app.ai_nutrition.risk_integration.alternative_food_finder import (
    AlternativeFoodFinder,
    FoodDatabase,
    SearchCriteria,
    FoodCategory,
    AlternativeScore
)
from app.ai_nutrition.risk_integration.dynamic_thresholds import DynamicThresholdDatabase
from app.ai_nutrition.risk_integration.health_profile_engine import (
    HealthProfileEngine,
    UserHealthProfile,
    MedicalCondition
)

router = APIRouter()

# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class ElementPredictionRequest(BaseModel):
    """Element prediction from chemometric model."""
    element: str = Field(..., description="Element symbol (e.g., 'Pb', 'K', 'Fe')")
    concentration: float = Field(..., description="Measured concentration", ge=0)
    uncertainty: Optional[float] = Field(0.0, description="Measurement uncertainty", ge=0)
    confidence: float = Field(..., description="Prediction confidence (0.0-1.0)", ge=0, le=1)
    unit: str = Field("ppm", description="Unit of measurement (ppm, mg/kg, mg/100g)")
    measurement_method: str = Field("visual_chemometrics", description="Detection method")
    
    class Config:
        schema_extra = {
            "example": {
                "element": "Pb",
                "concentration": 0.45,
                "uncertainty": 0.10,
                "confidence": 0.92,
                "unit": "ppm",
                "measurement_method": "visual_chemometrics"
            }
        }


class MedicalConditionRequest(BaseModel):
    """User medical condition."""
    snomed_code: str = Field(..., description="SNOMED CT code")
    name: str = Field(..., description="Condition name")
    severity: Optional[str] = Field("active", description="Severity level")
    diagnosed_date: Optional[datetime] = Field(None, description="Diagnosis date")
    notes: Optional[str] = Field("", description="Additional notes")
    
    class Config:
        schema_extra = {
            "example": {
                "snomed_code": "77386006",
                "name": "Pregnancy",
                "severity": "active",
                "diagnosed_date": "2024-01-15T00:00:00",
                "notes": "Second trimester, uncomplicated"
            }
        }


class HealthProfileRequest(BaseModel):
    """User health profile creation/update request."""
    user_id: str = Field(..., description="User identifier")
    conditions: List[MedicalConditionRequest] = Field(..., description="Medical conditions")
    age: Optional[int] = Field(None, description="Age in years", ge=0, le=150)
    weight_kg: Optional[float] = Field(None, description="Weight in kilograms", ge=0, le=500)
    height_cm: Optional[float] = Field(None, description="Height in centimeters", ge=0, le=300)
    medications: Optional[List[str]] = Field(default_factory=list, description="Current medications")
    allergies: Optional[List[str]] = Field(default_factory=list, description="Known allergies")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_12345",
                "conditions": [
                    {
                        "snomed_code": "77386006",
                        "name": "Pregnancy",
                        "severity": "active"
                    },
                    {
                        "snomed_code": "431857002",
                        "name": "CKD Stage 4",
                        "severity": "severe"
                    }
                ],
                "age": 32,
                "weight_kg": 68.0,
                "medications": ["Prenatal vitamins", "Iron supplement"]
            }
        }


class RiskAssessmentRequest(BaseModel):
    """Request for food safety risk assessment."""
    predictions: List[ElementPredictionRequest] = Field(..., description="Element predictions from chemometric model")
    user_profile: HealthProfileRequest = Field(..., description="User health profile")
    food_item: str = Field(..., description="Name of food being assessed")
    serving_size: float = Field(100.0, description="Serving size in grams", ge=1, le=10000)
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "element": "Pb",
                        "concentration": 0.45,
                        "uncertainty": 0.10,
                        "confidence": 0.92,
                        "unit": "ppm"
                    },
                    {
                        "element": "K",
                        "concentration": 450.0,
                        "confidence": 0.88,
                        "unit": "mg/100g"
                    }
                ],
                "user_profile": {
                    "user_id": "user_12345",
                    "conditions": [
                        {"snomed_code": "77386006", "name": "Pregnancy"}
                    ],
                    "age": 32
                },
                "food_item": "Spinach",
                "serving_size": 100.0
            }
        }


class MessageModeEnum(str, Enum):
    """Warning message modes."""
    consumer = "consumer"
    clinical = "clinical"
    regulatory = "regulatory"


class WarningRequest(BaseModel):
    """Request for personalized warning generation."""
    predictions: List[ElementPredictionRequest]
    user_profile: HealthProfileRequest
    food_item: str
    serving_size: float = 100.0
    message_mode: MessageModeEnum = MessageModeEnum.consumer
    batch_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {"element": "Pb", "concentration": 0.45, "confidence": 0.92, "unit": "ppm"}
                ],
                "user_profile": {
                    "user_id": "user_12345",
                    "conditions": [{"snomed_code": "77386006", "name": "Pregnancy"}],
                    "age": 32
                },
                "food_item": "Spinach",
                "serving_size": 100.0,
                "message_mode": "consumer"
            }
        }


class SearchCriteriaRequest(BaseModel):
    """Criteria for finding alternative foods."""
    problem_elements: List[str] = Field(..., description="Elements to minimize (e.g., ['Pb', 'K'])")
    preserve_nutrients: List[str] = Field(default_factory=list, description="Elements to preserve (e.g., ['Fe', 'Ca'])")
    category_preference: Optional[str] = Field(None, description="Preferred food category")
    max_price_increase_pct: float = Field(100.0, description="Maximum acceptable price increase (%)", ge=0, le=500)
    require_seasonal: bool = Field(False, description="Only show seasonal foods")
    
    class Config:
        schema_extra = {
            "example": {
                "problem_elements": ["Pb", "Cd"],
                "preserve_nutrients": ["Fe", "Ca"],
                "category_preference": "leafy_greens",
                "max_price_increase_pct": 50.0,
                "require_seasonal": False
            }
        }


class AlternativesRequest(BaseModel):
    """Request for alternative food recommendations."""
    original_food_name: str = Field(..., description="Name of problematic food")
    search_criteria: SearchCriteriaRequest
    user_profile: Optional[HealthProfileRequest] = None
    top_n: int = Field(10, description="Number of alternatives to return", ge=1, le=50)
    
    class Config:
        schema_extra = {
            "example": {
                "original_food_name": "Spinach",
                "search_criteria": {
                    "problem_elements": ["Pb"],
                    "preserve_nutrients": ["Fe", "Ca"],
                    "max_price_increase_pct": 50.0
                },
                "top_n": 5
            }
        }


# ============================================================================
# Initialize Risk Integration Components
# ============================================================================

# Singleton instances (would use dependency injection in production)
threshold_db = DynamicThresholdDatabase()
profile_engine = HealthProfileEngine()
risk_engine = RiskIntegrationEngine(threshold_db, profile_engine)
warning_system = WarningMessageGenerator(risk_engine)
food_db = FoodDatabase()
alternative_finder = AlternativeFoodFinder(food_db)


# ============================================================================
# Helper Functions
# ============================================================================

def convert_prediction_request(req: ElementPredictionRequest) -> ElementPrediction:
    """Convert API request to ElementPrediction object."""
    return ElementPrediction(
        element=req.element,
        concentration=req.concentration,
        uncertainty=req.uncertainty,
        confidence=req.confidence,
        unit=req.unit,
        measurement_method=req.measurement_method
    )


def convert_health_profile_request(req: HealthProfileRequest) -> UserHealthProfile:
    """Convert API request to UserHealthProfile object."""
    conditions = [
        MedicalCondition(
            condition_id=f"cond_{i}",
            snomed_code=c.snomed_code,
            name=c.name,
            severity=c.severity or "active",
            diagnosed_date=c.diagnosed_date or datetime.now(),
            notes=c.notes or ""
        )
        for i, c in enumerate(req.conditions)
    ]
    
    return UserHealthProfile(
        user_id=req.user_id,
        conditions=conditions,
        age=req.age,
        weight_kg=req.weight_kg,
        height_cm=req.height_cm,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


# ============================================================================
# API Endpoints
# ============================================================================

@router.post(
    "/assess",
    summary="Assess Food Safety Risk",
    description="""
    Perform comprehensive food safety risk assessment based on element predictions 
    and user health profile.
    
    This endpoint:
    1. Processes atomic element predictions from chemometric model
    2. Matches user's health conditions to applicable thresholds
    3. Checks toxic elements against regulatory limits
    4. Evaluates nutrients against daily targets
    5. Adjusts risk based on prediction confidence
    
    Returns complete risk assessment with safety failures, nutrient warnings, 
    and beneficial nutrients detected.
    """,
    response_description="Complete risk assessment",
    tags=["Risk Integration"]
)
async def assess_food_risk(request: RiskAssessmentRequest):
    """
    Assess food safety risk.
    
    **Example Response:**
    ```json
    {
        "overall_risk_level": "CRITICAL",
        "confidence": 0.92,
        "safety_failures": [
            {
                "element": "Pb",
                "measured_value": 0.45,
                "threshold": 0.1,
                "exceeds_limit": true,
                "exceedance_percentage": 350.0,
                "severity": "CRITICAL"
            }
        ],
        "nutrient_warnings": [
            {
                "element": "K",
                "contribution_percentage": 22.5,
                "is_restricted": true
            }
        ],
        "nutrient_benefits": [
            {
                "element": "Fe",
                "contribution_percentage": 13.0,
                "is_beneficial": true
            }
        ]
    }
    ```
    """
    try:
        # Convert request to internal objects
        predictions = [convert_prediction_request(p) for p in request.predictions]
        user_profile = convert_health_profile_request(request.user_profile)
        
        # Perform risk assessment
        risk_assessment = risk_engine.assess_risk(
            predictions=predictions,
            user_profile=user_profile,
            food_item=request.food_item,
            serving_size=request.serving_size
        )
        
        # Convert to dict for JSON response
        return {
            "overall_risk_level": risk_assessment.overall_risk_level,
            "confidence": risk_assessment.confidence,
            "safety_failures": [
                {
                    "element": f.element,
                    "measured_value": f.measured_value,
                    "threshold": f.threshold,
                    "exceeds_limit": f.exceeds_limit,
                    "exceedance_percentage": f.exceedance_percentage,
                    "regulatory_authority": f.regulatory_authority,
                    "severity": f.get_severity() if hasattr(f, 'get_severity') else "UNKNOWN"
                }
                for f in risk_assessment.safety_failures
            ],
            "nutrient_warnings": [
                {
                    "element": n.element,
                    "measured_value": n.measured_value,
                    "daily_target": n.daily_target,
                    "contribution_percentage": n.contribution_percentage,
                    "is_restricted": n.is_restricted,
                    "recommendation": n.get_recommendation() if hasattr(n, 'get_recommendation') else ""
                }
                for n in risk_assessment.nutrient_warnings
            ],
            "nutrient_benefits": [
                {
                    "element": n.element,
                    "measured_value": n.measured_value,
                    "contribution_percentage": n.contribution_percentage,
                    "is_beneficial": n.is_beneficial
                }
                for n in risk_assessment.nutrient_benefits
            ],
            "uncertainty_flags": risk_assessment.uncertainty_flags
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")


@router.post(
    "/warnings",
    summary="Generate Personalized Warnings",
    description="""
    Generate personalized warning messages in three modes:
    
    - **Consumer Mode**: Simple, actionable language for end-users
    - **Clinical Mode**: Medical terminology for healthcare professionals  
    - **Regulatory Mode**: Formal compliance reports for authorities
    
    Returns formatted warnings with risk levels, actionable insights, 
    alternative recommendations, and next steps.
    """,
    response_description="Personalized warning messages",
    tags=["Risk Integration"]
)
async def generate_warnings(request: WarningRequest):
    """
    Generate personalized warning messages.
    
    **Example Consumer Mode Response:**
    ```json
    {
        "banner": {
            "icon": "⛔",
            "message": "DO NOT CONSUME - Unsafe for Pregnancy",
            "risk_level": "CRITICAL"
        },
        "primary_concerns": [
            {
                "icon": "⛔",
                "element": "Lead",
                "message": "Lead levels are dangerously high (4.5x safe limit for pregnancy)",
                "action": "DO NOT CONSUME - Immediate health threat"
            }
        ],
        "alternatives": [
            "Try kale instead (similar iron, 90% less lead)",
            "Swiss chard (good iron source, lower potassium for CKD)"
        ],
        "next_steps": [
            "Avoid this specific spinach batch completely",
            "Speak with your OB-GYN about lead exposure"
        ]
    }
    ```
    """
    try:
        # Convert request to internal objects
        predictions = [convert_prediction_request(p) for p in request.predictions]
        user_profile = convert_health_profile_request(request.user_profile)
        
        # Map enum to MessageMode
        mode_mapping = {
            MessageModeEnum.consumer: MessageMode.CONSUMER,
            MessageModeEnum.clinical: MessageMode.CLINICAL,
            MessageModeEnum.regulatory: MessageMode.REGULATORY
        }
        message_mode = mode_mapping[request.message_mode]
        
        # Generate warnings
        warnings = warning_system.generate_warnings(
            predictions=predictions,
            user_profile=user_profile,
            food_item=request.food_item,
            serving_size=request.serving_size,
            message_mode=message_mode,
            batch_id=request.batch_id
        )
        
        return warnings
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warning generation failed: {str(e)}")


@router.post(
    "/alternatives",
    summary="Find Safer Alternative Foods",
    description="""
    AI-powered search for safer alternative foods that:
    - Minimize problem elements (toxic elements, restricted nutrients)
    - Preserve beneficial nutrients (iron, calcium, etc.)
    - Match taste and cooking methods when possible
    - Consider seasonal availability and price
    
    Returns ranked alternatives with:
    - Total score (risk reduction + nutrient preservation + availability + price)
    - Element-by-element comparisons
    - Risk level improvement (CRITICAL → SAFE)
    - Preparation tips
    """,
    response_description="Ranked alternative foods",
    tags=["Risk Integration"]
)
async def find_alternatives(request: AlternativesRequest):
    """
    Find safer alternative foods.
    
    **Example Response:**
    ```json
    {
        "alternatives": [
            {
                "food_name": "Kale",
                "total_score": 92.0,
                "risk_reduction_score": 90.0,
                "nutrient_preservation_score": 91.0,
                "risk_improvement": "CRITICAL → SAFE",
                "element_improvements": {
                    "Pb": 90.0,
                    "Cd": 75.0
                },
                "price": "$5.00/kg",
                "seasonality": "In season"
            }
        ]
    }
    ```
    """
    try:
        # Search for original food in database
        foods = food_db.search_by_name(request.original_food_name)
        if not foods:
            raise HTTPException(
                status_code=404, 
                detail=f"Food '{request.original_food_name}' not found in database"
            )
        
        original_food = foods[0]
        
        # Convert search criteria
        category = None
        if request.search_criteria.category_preference:
            try:
                category = FoodCategory(request.search_criteria.category_preference)
            except ValueError:
                pass
        
        search_criteria = SearchCriteria(
            problem_elements=request.search_criteria.problem_elements,
            preserve_nutrients=request.search_criteria.preserve_nutrients,
            category_preference=category,
            max_price_increase_pct=request.search_criteria.max_price_increase_pct,
            require_seasonal=request.search_criteria.require_seasonal
        )
        
        # Convert user profile if provided
        user_profile = None
        if request.user_profile:
            user_profile = convert_health_profile_request(request.user_profile)
        
        # Find alternatives
        alternatives = alternative_finder.find_alternatives(
            original_food=original_food,
            search_criteria=search_criteria,
            user_profile=user_profile,
            top_n=request.top_n
        )
        
        # Format response
        return {
            "original_food": original_food.name,
            "alternatives": [
                {
                    "food_name": alt.food_item.name,
                    "food_id": alt.food_item.food_id,
                    "total_score": round(alt.total_score, 1),
                    "risk_reduction_score": round(alt.risk_reduction_score, 1),
                    "nutrient_preservation_score": round(alt.nutrient_preservation_score, 1),
                    "availability_score": round(alt.availability_score, 1),
                    "price_score": round(alt.price_score, 1),
                    "risk_improvement": alt.risk_level_improvement,
                    "element_improvements": {
                        elem: round(reduction, 1) 
                        for elem, reduction in alt.element_improvements.items()
                    },
                    "element_comparisons": {
                        elem: {
                            "original": round(orig, 2),
                            "alternative": round(alt_val, 2),
                            "reduction_pct": round((orig - alt_val) / orig * 100, 1) if orig > 0 else 0
                        }
                        for elem, (orig, alt_val) in alt.element_comparisons.items()
                    },
                    "price": f"${alt.food_item.price_per_kg:.2f}/kg",
                    "seasonality": "In season" if alt.food_item.seasonality.is_in_season() else "Available",
                    "preparation_tips": alt.food_item.preparation_tips[:3]
                }
                for alt in alternatives
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alternative search failed: {str(e)}")


@router.get(
    "/thresholds/{condition}",
    summary="Get Medical Thresholds for Condition",
    description="""
    Retrieve element thresholds and limits for a specific health condition.
    
    Supports conditions:
    - Pregnancy (SNOMED: 77386006)
    - CKD stages (SNOMED: 431855005, 431856006, 431857002, 433144002, 433146000)
    - Diabetes Type 1/2
    - Infant (<2 years)
    
    Returns regulatory limits from FDA, WHO, NKF, KDIGO, ADA, etc.
    """,
    response_description="Medical thresholds for condition",
    tags=["Risk Integration"]
)
async def get_thresholds(
    condition: str = Query(..., description="SNOMED CT code or condition name"),
):
    """
    Get medical thresholds for a health condition.
    
    **Example Response:**
    ```json
    {
        "condition": "Pregnancy",
        "snomed_code": "77386006",
        "thresholds": [
            {
                "element": "Pb",
                "limit_value": 0.1,
                "unit": "mg/kg",
                "authority": "FDA",
                "regulation": "Defect Action Level"
            }
        ]
    }
    ```
    """
    try:
        # Try to get thresholds by SNOMED code
        thresholds = threshold_db.get_thresholds_for_condition(condition)
        
        if not thresholds:
            # Try by condition name
            thresholds = threshold_db.get_thresholds_by_name(condition)
        
        if not thresholds:
            raise HTTPException(
                status_code=404,
                detail=f"No thresholds found for condition '{condition}'"
            )
        
        return {
            "condition": condition,
            "thresholds": [
                {
                    "element": t.element,
                    "limit_value": t.limit_value,
                    "unit": t.unit,
                    "threshold_type": t.threshold_type,
                    "authority": t.authority,
                    "regulation": t.regulation_name,
                    "citation": t.citation
                }
                for t in thresholds
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Threshold retrieval failed: {str(e)}")


@router.post(
    "/health-profile",
    summary="Create/Update Health Profile",
    description="""
    Create or update user health profile with medical conditions.
    
    Stores:
    - Medical conditions (with SNOMED CT codes)
    - Age, weight, height
    - Current medications
    - Known allergies
    
    Returns complete profile with risk stratification.
    """,
    response_description="Created/updated health profile",
    tags=["Risk Integration"]
)
async def create_health_profile(request: HealthProfileRequest):
    """
    Create or update user health profile.
    
    **Example Response:**
    ```json
    {
        "user_id": "user_12345",
        "conditions": [
            {
                "name": "Pregnancy",
                "snomed_code": "77386006",
                "severity": "active"
            }
        ],
        "age": 32,
        "weight_kg": 68.0,
        "risk_level": "HIGH"
    }
    ```
    """
    try:
        # Convert and store profile
        user_profile = convert_health_profile_request(request)
        
        # Store in profile engine (would persist to database in production)
        profile_engine.create_profile(user_profile)
        
        # Calculate risk stratification
        risk_score = profile_engine.calculate_risk_score(user_profile)
        
        return {
            "user_id": user_profile.user_id,
            "conditions": [
                {
                    "condition_id": c.condition_id,
                    "name": c.name,
                    "snomed_code": c.snomed_code,
                    "severity": c.severity,
                    "diagnosed_date": c.diagnosed_date.isoformat() if c.diagnosed_date else None
                }
                for c in user_profile.conditions
            ],
            "age": user_profile.age,
            "weight_kg": user_profile.weight_kg,
            "height_cm": user_profile.height_cm,
            "risk_score": risk_score,
            "risk_level": "HIGH" if risk_score > 80 else "MODERATE" if risk_score > 50 else "LOW",
            "created_at": user_profile.created_at.isoformat(),
            "updated_at": user_profile.updated_at.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile creation failed: {str(e)}")


@router.get(
    "/health-profile/{user_id}",
    summary="Get User Health Profile",
    description="""
    Retrieve user health profile by user ID.
    
    Returns complete profile with all medical conditions and risk stratification.
    """,
    response_description="User health profile",
    tags=["Risk Integration"]
)
async def get_health_profile(user_id: str):
    """
    Get user health profile.
    
    **Example Response:**
    ```json
    {
        "user_id": "user_12345",
        "conditions": [
            {
                "name": "Pregnancy",
                "snomed_code": "77386006"
            }
        ],
        "age": 32
    }
    ```
    """
    try:
        # Retrieve profile (would query database in production)
        user_profile = profile_engine.get_profile(user_id)
        
        if not user_profile:
            raise HTTPException(
                status_code=404,
                detail=f"Profile not found for user '{user_id}'"
            )
        
        return {
            "user_id": user_profile.user_id,
            "conditions": [
                {
                    "name": c.name,
                    "snomed_code": c.snomed_code,
                    "severity": c.severity
                }
                for c in user_profile.conditions
            ],
            "age": user_profile.age,
            "weight_kg": user_profile.weight_kg,
            "height_cm": user_profile.height_cm
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile retrieval failed: {str(e)}")


@router.get(
    "/food-database/search",
    summary="Search Food Database",
    description="""
    Search food database by name or category.
    
    Returns foods with complete element profiles, nutrient data, 
    seasonal availability, and pricing.
    """,
    response_description="Food items matching search",
    tags=["Risk Integration"]
)
async def search_food_database(
    query: Optional[str] = Query(None, description="Food name search query"),
    category: Optional[str] = Query(None, description="Food category filter")
):
    """
    Search food database.
    
    **Example Response:**
    ```json
    {
        "foods": [
            {
                "food_id": "kale_001",
                "name": "Kale",
                "category": "leafy_greens",
                "elements": {
                    "Pb": 0.05,
                    "Fe": 32.0,
                    "K": 4910.0
                },
                "price_per_kg": 5.00,
                "in_season": true
            }
        ]
    }
    ```
    """
    try:
        foods = []
        
        if query:
            foods = food_db.search_by_name(query)
        elif category:
            try:
                cat = FoodCategory(category)
                foods = food_db.search_by_category(cat)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid category '{category}'"
                )
        else:
            # Return all foods (limit to first 50)
            foods = list(food_db.foods.values())[:50]
        
        return {
            "count": len(foods),
            "foods": [
                {
                    "food_id": f.food_id,
                    "name": f.name,
                    "category": f.category.value,
                    "subcategory": f.subcategory,
                    "elements": f.element_profile.to_flat_dict(),
                    "nutrients": {
                        "protein_g": f.nutrient_profile.protein_g,
                        "fiber_g": f.nutrient_profile.fiber_g,
                        "calories_kcal": f.nutrient_profile.calories_kcal
                    },
                    "price_per_kg": f.price_per_kg,
                    "in_season": f.seasonality.is_in_season(),
                    "preparation_tips": f.preparation_tips[:2]
                }
                for f in foods
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Food search failed: {str(e)}")


@router.get(
    "/health",
    summary="Health Check",
    description="Check if Risk Integration API is operational",
    tags=["Risk Integration"]
)
async def health_check():
    """Risk Integration API health check."""
    return {
        "status": "healthy",
        "service": "Risk Integration API",
        "version": "1.0.0",
        "components": {
            "threshold_database": "operational",
            "profile_engine": "operational",
            "risk_engine": "operational",
            "warning_system": "operational",
            "food_database": "operational",
            "alternative_finder": "operational"
        }
    }
