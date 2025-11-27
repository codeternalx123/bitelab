"""
Food Risk Analysis API Routes
==============================

RESTful API endpoints for comprehensive food safety and health analysis:
- Contaminant detection from ICPMS data
- Nutrient level analysis
- Health goal alignment
- Medical condition safety checks
- Risk scoring and recommendations

Endpoints:
- POST /food-risk/analyze: Analyze food risks
- POST /food-risk/batch-analyze: Batch analyze multiple foods
- GET /food-risk/contaminant-limits: Get safety limits for contaminants
- POST /food-risk/compare-foods: Compare risk profiles of multiple foods

Author: Wellomex AI Team
Date: November 2025
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from app.ai_nutrition.orchestration.food_risk_analyzer import (
    FoodRiskAnalyzer,
    ComprehensiveFoodRiskAnalysis,
    RiskLevel
)

router = APIRouter()

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ICPMSDataRequest(BaseModel):
    """ICPMS element detection data"""
    elements: Dict[str, float] = Field(
        ...,
        description="Element concentrations {element_name: ppm}",
        example={"Lead": 0.45, "Mercury": 0.12, "Iron": 2.7}
    )


class NutrientDataRequest(BaseModel):
    """Nutrient scan data"""
    calories: Optional[float] = Field(None, description="Calories per 100g")
    protein_g: Optional[float] = Field(None, description="Protein in grams")
    carbohydrates_g: Optional[float] = Field(None, description="Carbohydrates in grams")
    fat_g: Optional[float] = Field(None, description="Total fat in grams")
    fiber_g: Optional[float] = Field(None, description="Dietary fiber in grams")
    sugar_g: Optional[float] = Field(None, description="Total sugars in grams")
    sodium_mg: Optional[float] = Field(None, description="Sodium in mg")
    potassium_mg: Optional[float] = Field(None, description="Potassium in mg")
    
    nutrients: Dict[str, float] = Field(
        default_factory=dict,
        description="Additional nutrients {nutrient_name: amount}",
        example={"iron": 2.7, "calcium": 99, "vitamin_a": 9380}
    )
    
    # Omega-3 fatty acids
    omega3_total: Optional[float] = Field(None, description="Total omega-3 in mg")
    dha: Optional[float] = Field(None, description="DHA in mg")
    epa: Optional[float] = Field(None, description="EPA in mg")
    
    # Additional markers
    contains_gluten: Optional[bool] = Field(False, description="Contains gluten")


class UserProfileRequest(BaseModel):
    """User health profile"""
    age: Optional[int] = Field(None, description="Age in years", ge=0, le=150)
    gender: Optional[str] = Field(None, description="Gender (male/female/other)")
    weight_kg: Optional[float] = Field(None, description="Weight in kg")
    height_cm: Optional[float] = Field(None, description="Height in cm")
    
    medical_conditions: List[str] = Field(
        default_factory=list,
        description="List of medical conditions",
        example=["Pregnancy", "Hypertension", "Diabetes"]
    )
    
    health_goals: List[str] = Field(
        default_factory=list,
        description="List of health goals",
        example=["Weight loss", "Heart health", "Muscle gain"]
    )
    
    allergies: List[str] = Field(
        default_factory=list,
        description="Known allergies"
    )
    
    medications: List[str] = Field(
        default_factory=list,
        description="Current medications"
    )


class FoodRiskAnalysisRequest(BaseModel):
    """Request for food risk analysis"""
    food_name: str = Field(..., description="Name of the food")
    
    icpms_data: Optional[ICPMSDataRequest] = Field(
        None,
        description="ICPMS contaminant detection data"
    )
    
    scan_data: Optional[NutrientDataRequest] = Field(
        None,
        description="Nutrient scan data"
    )
    
    user_profile: Optional[UserProfileRequest] = Field(
        None,
        description="User health profile"
    )
    
    serving_size_g: float = Field(
        100.0,
        description="Serving size in grams",
        ge=1,
        le=1000
    )
    
    class Config:
        schema_extra = {
            "example": {
                "food_name": "Raw Spinach",
                "icpms_data": {
                    "elements": {
                        "Lead": 0.45,
                        "Iron": 2.7,
                        "Calcium": 99.0
                    }
                },
                "scan_data": {
                    "calories": 23,
                    "protein_g": 2.9,
                    "fiber_g": 2.2,
                    "nutrients": {
                        "iron": 2.7,
                        "calcium": 99,
                        "vitamin_a": 9380
                    }
                },
                "user_profile": {
                    "age": 28,
                    "medical_conditions": ["Pregnancy"],
                    "health_goals": ["Healthy pregnancy"]
                },
                "serving_size_g": 100
            }
        }


class BatchFoodRequest(BaseModel):
    """Batch food analysis request"""
    foods: List[FoodRiskAnalysisRequest] = Field(
        ...,
        description="List of foods to analyze"
    )
    
    user_profile: Optional[UserProfileRequest] = Field(
        None,
        description="Shared user profile for all foods"
    )


class FoodComparisonRequest(BaseModel):
    """Compare risk profiles of multiple foods"""
    foods: List[FoodRiskAnalysisRequest] = Field(
        ...,
        description="Foods to compare",
        min_items=2,
        max_items=10
    )
    
    comparison_criteria: List[str] = Field(
        default_factory=lambda: ["contaminant_risk", "nutrient_quality", "health_alignment"],
        description="Criteria to compare"
    )


class ContaminantLimitsResponse(BaseModel):
    """Response with contaminant safety limits"""
    contaminant_name: str
    contaminant_type: str
    
    general_limit: float
    unit: str
    
    population_specific_limits: Dict[str, float] = Field(
        default_factory=dict,
        description="Limits for specific populations"
    )
    
    health_effects: List[str] = Field(default_factory=list)
    regulatory_sources: List[str] = Field(default_factory=list)


# ============================================================================
# API ENDPOINTS
# ============================================================================

# Lazy-loaded analyzer
_risk_analyzer = None

def get_risk_analyzer() -> FoodRiskAnalyzer:
    """Get or create risk analyzer instance"""
    global _risk_analyzer
    if _risk_analyzer is None:
        _risk_analyzer = FoodRiskAnalyzer()
    return _risk_analyzer


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_food_risks(request: FoodRiskAnalysisRequest):
    """
    Analyze food safety risks and health alignment
    
    Performs comprehensive analysis including:
    - Heavy metal and contaminant detection (from ICPMS data)
    - Nutrient level assessment
    - Health goal alignment
    - Medical condition safety checks
    - Risk scoring and recommendations
    
    **Example Request:**
    ```json
    {
        "food_name": "Wild Salmon",
        "icpms_data": {
            "elements": {"Mercury": 0.12, "Selenium": 46.8}
        },
        "scan_data": {
            "protein_g": 20,
            "omega3_total": 2260,
            "nutrients": {"vitamin_d": 526}
        },
        "user_profile": {
            "age": 55,
            "medical_conditions": ["Hypertension"],
            "health_goals": ["Heart health"]
        }
    }
    ```
    
    **Returns:**
    - Contaminant analysis with risk levels
    - Nutrient adequacy scores
    - Health goal alignment scores
    - Medical condition safety checks
    - Overall risk and health scores
    - Personalized recommendations
    """
    try:
        analyzer = get_risk_analyzer()
        
        # Convert request models to dicts
        icpms_dict = None
        if request.icpms_data:
            icpms_dict = request.icpms_data.elements
        
        scan_dict = None
        if request.scan_data:
            scan_dict = request.scan_data.dict()
        
        user_dict = None
        if request.user_profile:
            user_dict = request.user_profile.dict()
        
        # Perform analysis
        result = await analyzer.analyze_food_risks(
            food_name=request.food_name,
            icpms_data=icpms_dict,
            scan_data=scan_dict,
            user_profile=user_dict,
            serving_size_g=request.serving_size_g
        )
        
        return {
            "success": True,
            "analysis": result.to_dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/batch-analyze", response_model=Dict[str, Any])
async def batch_analyze_foods(request: BatchFoodRequest, background_tasks: BackgroundTasks):
    """
    Analyze multiple foods in batch
    
    Useful for:
    - Meal composition analysis
    - Grocery shopping safety checks
    - Diet plan evaluation
    
    **Returns:**
    - Individual analysis for each food
    - Aggregate risk scores
    - Combined recommendations
    """
    try:
        analyzer = get_risk_analyzer()
        
        results = []
        total_risk = 0.0
        total_health = 0.0
        all_warnings = []
        
        for food_request in request.foods:
            # Use shared profile if individual not provided
            user_profile = food_request.user_profile or request.user_profile
            
            icpms_dict = None
            if food_request.icpms_data:
                icpms_dict = food_request.icpms_data.elements
            
            scan_dict = None
            if food_request.scan_data:
                scan_dict = food_request.scan_data.dict()
            
            user_dict = None
            if user_profile:
                user_dict = user_profile.dict()
            
            result = await analyzer.analyze_food_risks(
                food_name=food_request.food_name,
                icpms_data=icpms_dict,
                scan_data=scan_dict,
                user_profile=user_dict,
                serving_size_g=food_request.serving_size_g
            )
            
            results.append(result.to_dict())
            total_risk += result.overall_risk_score
            total_health += result.overall_health_score
            all_warnings.extend(result.critical_warnings)
        
        num_foods = len(request.foods)
        
        return {
            "success": True,
            "num_foods_analyzed": num_foods,
            "individual_results": results,
            "aggregate": {
                "average_risk_score": round(total_risk / num_foods, 1),
                "average_health_score": round(total_health / num_foods, 1),
                "total_warnings": len(all_warnings),
                "critical_warnings": all_warnings[:10]  # Top 10
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.post("/compare-foods", response_model=Dict[str, Any])
async def compare_food_risks(request: FoodComparisonRequest):
    """
    Compare risk profiles of multiple foods
    
    Helps users choose the safest and healthiest option among alternatives.
    
    **Example:**
    Compare spinach vs. kale for a pregnant woman to see which has lower lead.
    
    **Returns:**
    - Side-by-side comparison
    - Ranked by safety and health scores
    - Recommendation for best choice
    """
    try:
        analyzer = get_risk_analyzer()
        
        analyses = []
        
        for food_request in request.foods:
            icpms_dict = None
            if food_request.icpms_data:
                icpms_dict = food_request.icpms_data.elements
            
            scan_dict = None
            if food_request.scan_data:
                scan_dict = food_request.scan_data.dict()
            
            user_dict = None
            if food_request.user_profile:
                user_dict = food_request.user_profile.dict()
            
            result = await analyzer.analyze_food_risks(
                food_name=food_request.food_name,
                icpms_data=icpms_dict,
                scan_data=scan_dict,
                user_profile=user_dict,
                serving_size_g=food_request.serving_size_g
            )
            
            analyses.append({
                "food_name": food_request.food_name,
                "overall_risk_score": result.overall_risk_score,
                "overall_health_score": result.overall_health_score,
                "is_safe_to_consume": result.is_safe_to_consume,
                "contaminant_risk": result.overall_contaminant_risk.value,
                "critical_warnings": result.critical_warnings,
                "recommendation": result.recommendation
            })
        
        # Rank by health score (higher is better)
        ranked_by_health = sorted(analyses, key=lambda x: x["overall_health_score"], reverse=True)
        
        # Rank by risk score (lower is better)
        ranked_by_safety = sorted(analyses, key=lambda x: x["overall_risk_score"])
        
        # Best overall (highest health, lowest risk)
        best_overall = sorted(
            analyses,
            key=lambda x: (x["overall_health_score"] - x["overall_risk_score"]),
            reverse=True
        )[0]
        
        return {
            "success": True,
            "comparison": {
                "foods_compared": len(request.foods),
                "all_analyses": analyses,
                "ranked_by_health": ranked_by_health,
                "ranked_by_safety": ranked_by_safety,
                "best_overall": best_overall,
                "recommendation": f"✅ We recommend: {best_overall['food_name']} "
                                f"(Health Score: {best_overall['overall_health_score']:.0f}, "
                                f"Risk Score: {best_overall['overall_risk_score']:.0f})"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.get("/contaminant-limits", response_model=Dict[str, Any])
async def get_contaminant_limits(
    contaminant: Optional[str] = None,
    population: Optional[str] = None
):
    """
    Get safety limits for contaminants
    
    **Parameters:**
    - contaminant: Specific contaminant name (e.g., "lead", "mercury")
    - population: Population group (e.g., "pregnancy", "children", "adult")
    
    **Returns:**
    - Safety limits from FDA/WHO/EFSA
    - Population-specific thresholds
    - Health effects information
    """
    try:
        from app.ai_nutrition.orchestration.food_risk_analyzer import (
            HEAVY_METAL_LIMITS,
            PESTICIDE_LIMITS
        )
        
        limits = []
        
        # Get all heavy metal limits
        for metal, data in HEAVY_METAL_LIMITS.items():
            if contaminant is None or contaminant.lower() in metal.lower():
                pop_limits = {}
                for key, val in data.items():
                    if key not in ["unit", "general"] and isinstance(val, (int, float)):
                        pop_limits[key] = val
                
                analyzer = get_risk_analyzer()
                health_effects = analyzer._get_health_effects(metal)
                
                limits.append({
                    "contaminant_name": metal.capitalize(),
                    "contaminant_type": "heavy_metal",
                    "general_limit": data["general"],
                    "unit": data["unit"],
                    "population_specific_limits": pop_limits,
                    "health_effects": health_effects,
                    "regulatory_sources": ["FDA", "WHO", "EFSA"]
                })
        
        # Get pesticide limits
        for pesticide, data in PESTICIDE_LIMITS.items():
            if contaminant is None or contaminant.lower() in pesticide.lower():
                limits.append({
                    "contaminant_name": pesticide.capitalize(),
                    "contaminant_type": "pesticide",
                    "general_limit": data["limit"],
                    "unit": data["unit"],
                    "population_specific_limits": {},
                    "health_effects": ["Neurotoxicity", "Endocrine disruption"],
                    "regulatory_sources": ["EPA", "FDA"]
                })
        
        if not limits:
            raise HTTPException(status_code=404, detail=f"No limits found for: {contaminant}")
        
        return {
            "success": True,
            "contaminant_limits": limits
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve limits: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "food-risk-analysis",
        "version": "1.0.0"
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
Example API calls:

1. Analyze spinach for pregnant woman:
   POST /api/v1/food-risk/analyze
   {
       "food_name": "Raw Spinach",
       "icpms_data": {"elements": {"Lead": 0.45, "Iron": 2.7}},
       "scan_data": {"protein_g": 2.9, "nutrients": {"iron": 2.7}},
       "user_profile": {"age": 28, "medical_conditions": ["Pregnancy"]}
   }

2. Compare foods:
   POST /api/v1/food-risk/compare-foods
   {
       "foods": [
           {"food_name": "Spinach", "icpms_data": {"elements": {"Lead": 0.45}}},
           {"food_name": "Kale", "icpms_data": {"elements": {"Lead": 0.05}}}
       ]
   }

3. Get lead limits:
   GET /api/v1/food-risk/contaminant-limits?contaminant=lead
"""
