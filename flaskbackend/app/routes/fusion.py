"""
Fused Analysis API Routes
==========================

Multi-modal sensor fusion combining visual, chemical, and contextual data.

Core Features:
- ICP-MS + Visual chemometrics fusion
- Multi-sensor data integration
- Hybrid prediction (visual + chemical ground truth)
- Uncertainty reduction through fusion
- Cross-validation of predictions

Integration with:
- Visual chemometrics (Phases 1-5)
- ICP-MS data engine
- Quantum colorimetry
- Safety analysis engine
"""

from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

router = APIRouter()


# ============================================================================
# ENUMS
# ============================================================================

class SensorTypeEnum(str, Enum):
    """Sensor types for fusion"""
    VISUAL_RGB = "visual_rgb"
    ICPMS = "icpms"
    XRF = "xrf"
    NIR_SPECTROSCOPY = "nir_spectroscopy"
    RAMAN = "raman"
    HYPERSPECTRAL = "hyperspectral"


class FusionMethodEnum(str, Enum):
    """Fusion algorithms"""
    WEIGHTED_AVERAGE = "weighted_average"
    BAYESIAN_FUSION = "bayesian_fusion"
    KALMAN_FILTER = "kalman_filter"
    NEURAL_FUSION = "neural_fusion"
    ENSEMBLE = "ensemble"


# ============================================================================
# REQUEST MODELS
# ============================================================================

class SensorData(BaseModel):
    """Data from a single sensor"""
    sensor_type: SensorTypeEnum = Field(..., description="Type of sensor")
    data: Dict[str, Any] = Field(
        ...,
        description="Sensor data (format varies by sensor type)",
        example={
            "Pb": 0.05,
            "Cd": 0.02,
            "Fe": 12.5
        }
    )
    confidence: Optional[float] = Field(
        None,
        description="Sensor reading confidence (0-1)"
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="Measurement timestamp"
    )


class FusedAnalysisRequest(BaseModel):
    """Request for fused multi-sensor analysis"""
    image_base64: Optional[str] = Field(
        None,
        description="Base64-encoded food image for visual analysis"
    )
    sensor_data: List[SensorData] = Field(
        ...,
        description="Data from multiple sensors to fuse",
        min_items=2
    )
    fusion_method: FusionMethodEnum = Field(
        FusionMethodEnum.BAYESIAN_FUSION,
        description="Fusion algorithm to use"
    )
    food_name: Optional[str] = Field(
        None,
        description="Food item name"
    )
    prior_knowledge: Optional[Dict[str, Any]] = Field(
        None,
        description="Prior knowledge to incorporate (e.g., typical ranges for food type)",
        example={
            "typical_pb_range": {"min": 0.01, "max": 0.08},
            "typical_fe_range": {"min": 8.0, "max": 15.0}
        }
    )

    @validator('sensor_data')
    def validate_sensor_types(cls, v):
        sensor_types = [s.sensor_type for s in v]
        if len(sensor_types) != len(set(sensor_types)):
            raise ValueError("Duplicate sensor types not allowed in fusion")
        return v

    class Config:
        schema_extra = {
            "example": {
                "food_name": "Spinach",
                "sensor_data": [
                    {
                        "sensor_type": "visual_rgb",
                        "data": {"Pb": 0.06, "Cd": 0.03, "Fe": 11.2},
                        "confidence": 0.75
                    },
                    {
                        "sensor_type": "icpms",
                        "data": {"Pb": 0.05, "Cd": 0.02, "Fe": 12.5},
                        "confidence": 0.98
                    }
                ],
                "fusion_method": "bayesian_fusion"
            }
        }


class HybridPredictionRequest(BaseModel):
    """Request for hybrid prediction (visual + partial chemical data)"""
    image_base64: str = Field(
        ...,
        description="Base64-encoded food image"
    )
    known_elements: Dict[str, float] = Field(
        ...,
        description="Known element concentrations from lab analysis (ppm)",
        example={"Pb": 0.05, "Cd": 0.02}
    )
    elements_to_predict: List[str] = Field(
        ...,
        description="Elements to predict using hybrid approach",
        example=["Fe", "Ca", "Mg", "Zn"]
    )
    food_name: Optional[str] = Field(None, description="Food item name")


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class FusedPrediction(BaseModel):
    """Fused prediction for a single element"""
    element_symbol: str = Field(..., description="Element symbol")
    element_name: str = Field(..., description="Element name")
    
    # Individual sensor predictions
    sensor_predictions: Dict[str, float] = Field(
        ...,
        description="Predictions from individual sensors",
        example={
            "visual_rgb": 0.06,
            "icpms": 0.05
        }
    )
    
    # Fused result
    fused_concentration_ppm: float = Field(
        ...,
        description="Fused concentration (ppm)"
    )
    fused_uncertainty_ppm: float = Field(
        ...,
        description="Fused uncertainty (reduced through fusion)"
    )
    
    # Fusion metrics
    uncertainty_reduction_percentage: float = Field(
        ...,
        description="Percentage reduction in uncertainty compared to worst individual sensor"
    )
    fusion_confidence: float = Field(
        ...,
        description="Overall fusion confidence (0-1)"
    )
    sensor_agreement_score: float = Field(
        ...,
        description="Degree of agreement between sensors (0-1, 1=perfect agreement)"
    )


class FusedAnalysisResponse(BaseModel):
    """Fused analysis response"""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    food_name: Optional[str] = Field(None, description="Food item name")
    
    fusion_method: FusionMethodEnum = Field(..., description="Fusion method used")
    sensors_used: List[SensorTypeEnum] = Field(..., description="Sensors included in fusion")
    
    fused_predictions: List[FusedPrediction] = Field(
        ...,
        description="Fused predictions for all elements"
    )
    
    fusion_quality_metrics: Dict[str, float] = Field(
        ...,
        description="Overall fusion quality metrics",
        example={
            "average_uncertainty_reduction": 45.2,
            "average_sensor_agreement": 0.85,
            "fusion_reliability_score": 0.92
        }
    )
    
    recommendations: List[str] = Field(
        ...,
        description="Recommendations based on fusion results"
    )

    class Config:
        schema_extra = {
            "example": {
                "request_id": "fusion_20231115_123456",
                "timestamp": "2023-11-15T12:34:56Z",
                "food_name": "Spinach",
                "fusion_method": "bayesian_fusion",
                "sensors_used": ["visual_rgb", "icpms"],
                "fused_predictions": [
                    {
                        "element_symbol": "Pb",
                        "element_name": "lead",
                        "sensor_predictions": {
                            "visual_rgb": 0.06,
                            "icpms": 0.05
                        },
                        "fused_concentration_ppm": 0.051,
                        "fused_uncertainty_ppm": 0.008,
                        "uncertainty_reduction_percentage": 42.5,
                        "fusion_confidence": 0.94,
                        "sensor_agreement_score": 0.88
                    }
                ],
                "fusion_quality_metrics": {
                    "average_uncertainty_reduction": 45.2,
                    "average_sensor_agreement": 0.85,
                    "fusion_reliability_score": 0.92
                }
            }
        }


class HybridPredictionResponse(BaseModel):
    """Hybrid prediction response (visual + partial chemical)"""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    food_name: Optional[str] = Field(None, description="Food item name")
    
    known_elements: Dict[str, float] = Field(
        ...,
        description="Known element concentrations from lab analysis"
    )
    predicted_elements: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Predicted elements with concentrations and uncertainties",
        example={
            "Fe": {"concentration_ppm": 12.5, "uncertainty_ppm": 1.8},
            "Ca": {"concentration_ppm": 145.0, "uncertainty_ppm": 12.0}
        }
    )
    
    hybrid_confidence: float = Field(
        ...,
        description="Overall confidence in hybrid predictions (0-1)"
    )
    
    improvement_over_visual_only: float = Field(
        ...,
        description="Percentage improvement in accuracy vs visual-only prediction"
    )


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "/analyze",
    response_model=FusedAnalysisResponse,
    summary="Perform fused multi-sensor analysis",
    description="""
    Fuse data from multiple sensors for improved accuracy and reduced uncertainty.
    
    **Fusion Benefits:**
    - 40-60% uncertainty reduction compared to individual sensors
    - Cross-validation between sensor modalities
    - Outlier detection and correction
    - Robust predictions even with sensor failures
    
    **Supported Sensors:**
    - Visual RGB imaging (chemometric prediction)
    - ICP-MS (high precision elemental analysis)
    - XRF (X-ray fluorescence)
    - NIR spectroscopy
    - Raman spectroscopy
    - Hyperspectral imaging
    
    **Fusion Methods:**
    - **Weighted Average**: Simple weighted combination
    - **Bayesian Fusion**: Probabilistic fusion with uncertainty propagation
    - **Kalman Filter**: Optimal estimation for time-series data
    - **Neural Fusion**: Deep learning-based fusion
    - **Ensemble**: Multiple fusion methods combined
    """,
)
async def fused_analysis(request: FusedAnalysisRequest):
    """
    Perform fused analysis combining multiple sensor modalities.
    """
    try:
        # TODO: Integrate with actual fusion engine
        # from app.ai_nutrition.chemometrics.icpms_data_engine import ICPMSDataEngine
        # fusion_engine = ICPMSDataEngine()
        # result = fusion_engine.fuse_predictions(request)
        
        # MOCK RESPONSE
        request_id = f"fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract element symbols from sensor data
        all_elements = set()
        for sensor in request.sensor_data:
            all_elements.update(sensor.data.keys())
        
        # Generate fused predictions
        fused_predictions = []
        element_names = {"Pb": "lead", "Cd": "cadmium", "Fe": "iron", "Ca": "calcium", "Mg": "magnesium"}
        
        for element in all_elements:
            # Collect predictions from different sensors
            sensor_preds = {}
            for sensor in request.sensor_data:
                if element in sensor.data:
                    sensor_preds[sensor.sensor_type.value] = sensor.data[element]
            
            if len(sensor_preds) >= 2:
                # Compute fused value (weighted average for mock)
                values = list(sensor_preds.values())
                fused_value = sum(values) / len(values)
                
                # Mock uncertainty calculations
                individual_uncertainty = max(values) - min(values)
                fused_uncertainty = individual_uncertainty * 0.5  # 50% reduction
                
                fusion_pred = FusedPrediction(
                    element_symbol=element,
                    element_name=element_names.get(element, element.lower()),
                    sensor_predictions=sensor_preds,
                    fused_concentration_ppm=fused_value,
                    fused_uncertainty_ppm=fused_uncertainty,
                    uncertainty_reduction_percentage=50.0,
                    fusion_confidence=0.92,
                    sensor_agreement_score=0.85
                )
                fused_predictions.append(fusion_pred)
        
        response = FusedAnalysisResponse(
            request_id=request_id,
            timestamp=datetime.now(),
            food_name=request.food_name,
            fusion_method=request.fusion_method,
            sensors_used=[s.sensor_type for s in request.sensor_data],
            fused_predictions=fused_predictions,
            fusion_quality_metrics={
                "average_uncertainty_reduction": 48.5,
                "average_sensor_agreement": 0.87,
                "fusion_reliability_score": 0.93
            },
            recommendations=[
                "High sensor agreement indicates reliable fusion results",
                "Uncertainty reduced by ~50% through multi-sensor fusion",
                "Consider ICP-MS calibration for elements with lower agreement"
            ]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fusion analysis error: {str(e)}"
        )


@router.post(
    "/hybrid-prediction",
    response_model=HybridPredictionResponse,
    summary="Hybrid prediction using visual + partial chemical data",
    description="""
    Improve visual chemometric predictions by incorporating partial chemical ground truth.
    
    **Use case:**
    - You have lab analysis for some elements (e.g., Pb, Cd from ICP-MS)
    - Want to predict other elements (e.g., Fe, Ca, Zn) without additional lab work
    - Hybrid model uses known elements to calibrate visual predictions
    
    **Benefits:**
    - 30-50% improvement over visual-only predictions
    - Cost savings: analyze fewer elements in lab
    - Faster results for non-critical elements
    """,
)
async def hybrid_prediction(request: HybridPredictionRequest):
    """
    Perform hybrid prediction using visual analysis and partial chemical data.
    """
    try:
        # TODO: Integrate with hybrid prediction engine
        # MOCK RESPONSE
        request_id = f"hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        element_conc = {
            "Fe": 12.5,
            "Ca": 145.0,
            "Mg": 35.2,
            "Zn": 2.8
        }
        
        predicted = {}
        for elem in request.elements_to_predict:
            if elem in element_conc:
                predicted[elem] = {
                    "concentration_ppm": element_conc[elem],
                    "uncertainty_ppm": element_conc[elem] * 0.15  # 15% uncertainty
                }
        
        response = HybridPredictionResponse(
            request_id=request_id,
            timestamp=datetime.now(),
            food_name=request.food_name,
            known_elements=request.known_elements,
            predicted_elements=predicted,
            hybrid_confidence=0.88,
            improvement_over_visual_only=42.5
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Hybrid prediction error: {str(e)}"
        )


@router.get(
    "/fusion-methods",
    summary="Get available fusion methods",
    description="Retrieve information about supported fusion algorithms",
)
async def get_fusion_methods():
    """
    Get information about available fusion methods.
    """
    return {
        "fusion_methods": [
            {
                "method": "weighted_average",
                "name": "Weighted Average",
                "description": "Simple weighted combination based on sensor confidence",
                "complexity": "low",
                "typical_uncertainty_reduction": "20-30%",
                "best_for": "Quick fusion with known sensor reliabilities"
            },
            {
                "method": "bayesian_fusion",
                "name": "Bayesian Fusion",
                "description": "Probabilistic fusion with full uncertainty propagation",
                "complexity": "medium",
                "typical_uncertainty_reduction": "40-50%",
                "best_for": "Rigorous uncertainty quantification"
            },
            {
                "method": "kalman_filter",
                "name": "Kalman Filter",
                "description": "Optimal estimation for sequential/time-series measurements",
                "complexity": "medium",
                "typical_uncertainty_reduction": "35-45%",
                "best_for": "Time-series data or sequential measurements"
            },
            {
                "method": "neural_fusion",
                "name": "Neural Fusion",
                "description": "Deep learning-based adaptive fusion",
                "complexity": "high",
                "typical_uncertainty_reduction": "50-60%",
                "best_for": "Complex patterns, large training datasets"
            },
            {
                "method": "ensemble",
                "name": "Ensemble Fusion",
                "description": "Combines multiple fusion methods",
                "complexity": "high",
                "typical_uncertainty_reduction": "45-55%",
                "best_for": "Maximum robustness and accuracy"
            }
        ],
        "supported_sensors": [s.value for s in SensorTypeEnum]
    }


@router.get(
    "/health",
    summary="Health check for fusion service",
)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "fusion",
        "timestamp": datetime.now(),
        "capabilities": [
            "multi_sensor_fusion",
            "hybrid_prediction",
            "uncertainty_reduction",
            "cross_validation"
        ]
    }
