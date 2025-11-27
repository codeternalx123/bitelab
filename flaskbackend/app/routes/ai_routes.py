"""
Deep Learning & Knowledge Graph API
====================================

FastAPI routes for:
1. Food analysis with deep learning models
2. Knowledge graph querying
3. LLM-based knowledge expansion
4. Training model from LLM data

Author: Wellomex AI Team
Date: November 2025
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from app.ai_nutrition.knowledge_graphs.integrated_nutrition_ai import (
    IntegratedNutritionAI,
    UserContext,
    FoodItem,
    NutritionRecommendation,
    create_integrated_system
)
from app.ai_nutrition.knowledge_graphs.food_knowledge_graph import EntityType, RelationType
from app.core.security import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["AI & Deep Learning"])


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Singleton integrated AI system
_integrated_ai: Optional[IntegratedNutritionAI] = None

def get_integrated_ai() -> IntegratedNutritionAI:
    """Get or create integrated AI system"""
    global _integrated_ai
    
    if _integrated_ai is None:
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        _integrated_ai = create_integrated_system(llm_api_key=api_key)
        logger.info("Integrated AI system initialized")
    
    return _integrated_ai


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class UserContextRequest(BaseModel):
    """User health context"""
    age: int = Field(..., ge=0, le=150)
    gender: str = Field(..., pattern="^(male|female|other)$")
    weight: float = Field(..., gt=0, description="Weight in kg")
    height: float = Field(..., gt=0, description="Height in cm")
    activity_level: str = Field(..., pattern="^(sedentary|light|moderate|active|very_active)$")
    health_goals: List[str] = Field(default_factory=list)
    medical_conditions: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    dietary_restrictions: List[str] = Field(default_factory=list)
    biomarkers: Dict[str, float] = Field(default_factory=dict)


class FoodItemRequest(BaseModel):
    """Food item for analysis"""
    name: str
    ingredients: List[str] = Field(default_factory=list)
    nutrients: Dict[str, float] = Field(default_factory=dict)
    portion_size: float = Field(default=100, description="Portion in grams")
    preparation_method: Optional[str] = None


class AnalyzeFoodRequest(BaseModel):
    """Request for food analysis"""
    food: FoodItemRequest
    user_context: UserContextRequest
    use_graph: bool = Field(default=True, description="Use knowledge graph")
    use_deep_model: bool = Field(default=True, description="Use deep learning")
    use_llm: bool = Field(default=False, description="Query LLM for validation")


class AnalyzeFoodResponse(BaseModel):
    """Response from food analysis"""
    food_name: str
    recommendation_score: float
    alignment_with_goals: Dict[str, float]
    disease_risk_impacts: Dict[str, float]
    portion_recommendation: float
    reasoning: List[str]
    confidence: float
    alternatives: List[str]
    cautions: List[str]


class KnowledgeQueryRequest(BaseModel):
    """Query knowledge graph"""
    entity_name: str
    relation_type: Optional[str] = None
    max_results: int = Field(default=10, le=100)


class KnowledgeQueryResponse(BaseModel):
    """Knowledge graph query response"""
    entity: str
    relationships: List[Dict[str, Any]]


class ExpandKnowledgeRequest(BaseModel):
    """Request to expand knowledge from LLM"""
    entities: List[str]
    entity_type: str = Field(default="FOOD")


class TrainFromLLMRequest(BaseModel):
    """Request to train model from LLM data"""
    foods: List[str]
    health_goals: List[str]
    diseases: List[str]
    num_samples: int = Field(default=100, ge=10, le=1000)


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/analyze-food", response_model=AnalyzeFoodResponse)
async def analyze_food_with_ai(
    request: AnalyzeFoodRequest,
    current_user: Dict = Depends(get_current_user),
    ai_system: IntegratedNutritionAI = Depends(get_integrated_ai)
):
    """
    Analyze food using integrated AI system
    
    Combines:
    - Knowledge graph relationships
    - Deep learning predictions
    - Optional LLM validation
    
    Returns comprehensive nutrition recommendation
    """
    
    try:
        # Convert request models to internal types
        food_item = FoodItem(
            name=request.food.name,
            ingredients=request.food.ingredients,
            nutrients=request.food.nutrients,
            portion_size=request.food.portion_size,
            preparation_method=request.food.preparation_method
        )
        
        user_context = UserContext(
            user_id=current_user.get("user_id", "unknown"),
            age=request.user_context.age,
            gender=request.user_context.gender,
            weight=request.user_context.weight,
            height=request.user_context.height,
            activity_level=request.user_context.activity_level,
            health_goals=request.user_context.health_goals,
            medical_conditions=request.user_context.medical_conditions,
            medications=request.user_context.medications,
            allergies=request.user_context.allergies,
            dietary_restrictions=request.user_context.dietary_restrictions,
            biomarkers=request.user_context.biomarkers
        )
        
        # Run analysis
        recommendation = await ai_system.analyze_food(
            food=food_item,
            user_context=user_context,
            use_graph=request.use_graph,
            use_deep_model=request.use_deep_model,
            use_llm=request.use_llm
        )
        
        # Convert to response
        return AnalyzeFoodResponse(
            food_name=recommendation.food_name,
            recommendation_score=recommendation.recommendation_score,
            alignment_with_goals=recommendation.alignment_with_goals,
            disease_risk_impacts=recommendation.disease_risk_impacts,
            portion_recommendation=recommendation.portion_recommendation,
            reasoning=recommendation.reasoning,
            confidence=recommendation.confidence,
            alternatives=recommendation.alternatives,
            cautions=recommendation.cautions
        )
        
    except Exception as e:
        logger.error(f"Food analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/knowledge/query", response_model=KnowledgeQueryResponse)
async def query_knowledge_graph(
    request: KnowledgeQueryRequest,
    current_user: Dict = Depends(get_current_user),
    ai_system: IntegratedNutritionAI = Depends(get_integrated_ai)
):
    """
    Query the knowledge graph
    
    Returns entity relationships, properties, and connection strengths
    """
    
    try:
        relation_type = None
        if request.relation_type:
            try:
                relation_type = RelationType[request.relation_type.upper()]
            except KeyError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid relation type: {request.relation_type}"
                )
        
        relationships = ai_system.knowledge_graph.query(
            entity_name=request.entity_name,
            relation_type=relation_type
        )
        
        # Limit results
        relationships = relationships[:request.max_results]
        
        return KnowledgeQueryResponse(
            entity=request.entity_name,
            relationships=relationships
        )
        
    except Exception as e:
        logger.error(f"Knowledge query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/knowledge/entities")
async def list_entities(
    entity_type: Optional[str] = None,
    limit: int = 100,
    current_user: Dict = Depends(get_current_user),
    ai_system: IntegratedNutritionAI = Depends(get_integrated_ai)
):
    """
    List entities in knowledge graph
    
    Optionally filter by entity type (FOOD, NUTRIENT, HEALTH_GOAL, DISEASE, etc.)
    """
    
    try:
        entities = ai_system.knowledge_graph.entities
        
        # Filter by type if specified
        if entity_type:
            try:
                entity_type_enum = EntityType[entity_type.upper()]
                entities = {
                    name: entity
                    for name, entity in entities.items()
                    if entity.entity_type == entity_type_enum
                }
            except KeyError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid entity type: {entity_type}"
                )
        
        # Convert to list and limit
        entity_list = [
            {
                "name": name,
                "type": entity.entity_type.value,
                "properties": entity.properties
            }
            for name, entity in list(entities.items())[:limit]
        ]
        
        return {
            "total": len(entities),
            "returned": len(entity_list),
            "entities": entity_list
        }
        
    except Exception as e:
        logger.error(f"Entity listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Listing failed: {str(e)}")


@router.post("/knowledge/expand")
async def expand_knowledge(
    request: ExpandKnowledgeRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user),
    ai_system: IntegratedNutritionAI = Depends(get_integrated_ai)
):
    """
    Expand knowledge graph using LLM
    
    Queries GPT-4 for entity relationships and adds to graph.
    Runs in background for large requests.
    """
    
    try:
        entity_type_enum = EntityType[request.entity_type.upper()]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid entity type: {request.entity_type}"
        )
    
    # Run expansion in background
    background_tasks.add_task(
        ai_system.expand_knowledge,
        entities=request.entities,
        entity_type=entity_type_enum
    )
    
    return {
        "message": f"Knowledge expansion started for {len(request.entities)} entities",
        "entities": request.entities,
        "status": "processing"
    }


@router.post("/train/llm-data")
async def train_from_llm(
    request: TrainFromLLMRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user),
    ai_system: IntegratedNutritionAI = Depends(get_integrated_ai)
):
    """
    Generate training data from LLM and train deep models
    
    This is a long-running operation that:
    1. Queries GPT-4 for food-health relationships
    2. Generates synthetic training examples
    3. Trains deep learning models
    
    Runs in background and returns immediately.
    """
    
    # Validate admin access
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=403,
            detail="Admin access required for model training"
        )
    
    # Run training in background
    background_tasks.add_task(
        ai_system.train_from_llm,
        foods=request.foods,
        health_goals=request.health_goals,
        diseases=request.diseases,
        num_samples=request.num_samples
    )
    
    return {
        "message": "Training started from LLM data",
        "config": {
            "foods": len(request.foods),
            "health_goals": len(request.health_goals),
            "diseases": len(request.diseases),
            "samples": request.num_samples
        },
        "status": "training"
    }


@router.get("/knowledge/stats")
async def get_knowledge_stats(
    current_user: Dict = Depends(get_current_user),
    ai_system: IntegratedNutritionAI = Depends(get_integrated_ai)
):
    """
    Get knowledge graph statistics
    
    Returns entity counts, relationship counts, coverage metrics
    """
    
    try:
        graph = ai_system.knowledge_graph
        
        # Count entities by type
        entity_counts = {}
        for entity_type in EntityType:
            count = sum(
                1 for e in graph.entities.values()
                if e.entity_type == entity_type
            )
            entity_counts[entity_type.value] = count
        
        # Count relationships by type
        relation_counts = {}
        total_relations = 0
        
        for entity_name in graph.entities.keys():
            for relation_type in RelationType:
                rels = graph.query(entity_name, relation_type)
                relation_counts[relation_type.value] = relation_counts.get(
                    relation_type.value, 0
                ) + len(rels)
                total_relations += len(rels)
        
        return {
            "entities": {
                "total": len(graph.entities),
                "by_type": entity_counts
            },
            "relationships": {
                "total": total_relations,
                "by_type": relation_counts
            },
            "graph_density": total_relations / max(len(graph.entities), 1)
        }
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")


@router.post("/save")
async def save_ai_system(
    directory: str = "ai_models",
    current_user: Dict = Depends(get_current_user),
    ai_system: IntegratedNutritionAI = Depends(get_integrated_ai)
):
    """
    Save AI system state (knowledge graph + deep models)
    
    Admin only - saves to specified directory
    """
    
    if not current_user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        ai_system.save(directory)
        
        return {
            "message": "AI system saved successfully",
            "directory": directory
        }
        
    except Exception as e:
        logger.error(f"Save failed: {e}")
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")


@router.post("/load")
async def load_ai_system(
    directory: str = "ai_models",
    current_user: Dict = Depends(get_current_user),
    ai_system: IntegratedNutritionAI = Depends(get_integrated_ai)
):
    """
    Load AI system state from disk
    
    Admin only - loads from specified directory
    """
    
    if not current_user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        ai_system.load(directory)
        
        return {
            "message": "AI system loaded successfully",
            "directory": directory
        }
        
    except Exception as e:
        logger.error(f"Load failed: {e}")
        raise HTTPException(status_code=500, detail=f"Load failed: {str(e)}")
