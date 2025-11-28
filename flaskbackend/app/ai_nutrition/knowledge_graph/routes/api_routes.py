"""
FastAPI Routes for Food Knowledge Graph
======================================

REST API endpoints for the food knowledge graph system,
providing comprehensive food data access, search, analysis, and management.

Available Endpoints:
- Food Search and Discovery
- Food Details and Management
- Nutritional Analysis
- Food Substitution Recommendations
- Cultural Food Pattern Analysis
- Seasonal Availability Predictions
- Data Ingestion and Synchronization
- System Analytics and Quality Assessment

Author: Wellomex AI Nutrition Team
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
from fastapi.responses import JSONResponse
from pydantic import Field

from ..services.food_knowledge_service import (
    FoodKnowledgeService, 
    FoodKnowledgeServiceFactory,
    FoodSearchQuery,
    FoodCreateRequest,
    SubstitutionRequest,
    CulturalAnalysisRequest,
    SeasonalPredictionRequest,
    DataIngestionRequest
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/food-knowledge", tags=["Food Knowledge Graph"])

# Dependency to get service instance
async def get_food_knowledge_service() -> FoodKnowledgeService:
    """Dependency to get the food knowledge service"""
    try:
        return await FoodKnowledgeServiceFactory.get_service()
    except Exception as e:
        logger.error(f"Failed to get food knowledge service: {e}")
        raise HTTPException(
            status_code=503, 
            detail="Food Knowledge Graph service is currently unavailable"
        )

@router.get("/health", summary="Health Check")
async def health_check():
    """Health check endpoint for the food knowledge graph service"""
    return {
        "status": "healthy",
        "service": "Food Knowledge Graph",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

# Food Search and Discovery Endpoints

@router.post("/search", summary="Search Foods", response_model=Dict[str, Any])
async def search_foods(
    query: FoodSearchQuery = Body(..., description="Search parameters"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Search for foods using advanced knowledge graph queries with ML enhancement.
    
    Supports:
    - Text-based search with natural language processing
    - Category and nutritional filtering
    - Country-specific availability
    - Allergen and dietary restriction filtering
    - ML-powered result ranking and similarity matching
    """
    return await service.search_foods(query)

@router.get("/search", summary="Search Foods (GET)", response_model=Dict[str, Any])
async def search_foods_get(
    q: str = Query("", description="Search query"),
    category: Optional[str] = Query(None, description="Food category"),
    country: Optional[str] = Query(None, description="Country code"),
    allergen_free: Optional[str] = Query(None, description="Comma-separated allergens to avoid"),
    dietary: Optional[str] = Query(None, description="Comma-separated dietary restrictions"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Result offset"),
    include_similar: bool = Query(False, description="Include similar foods"),
    use_ml_ranking: bool = Query(True, description="Use ML ranking"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    GET version of food search for simple queries and API exploration.
    """
    # Convert GET parameters to search query
    search_query = FoodSearchQuery(
        query=q,
        category=category,
        country_codes=[country] if country else None,
        allergen_free=allergen_free.split(',') if allergen_free else None,
        dietary_compatible=dietary.split(',') if dietary else None,
        limit=limit,
        offset=offset,
        include_similar=include_similar,
        use_ml_ranking=use_ml_ranking
    )
    
    return await service.search_foods(search_query)

@router.get("/foods/{food_id}", summary="Get Food Details")
async def get_food_details(
    food_id: str = Path(..., description="Unique food identifier"),
    enrich_ml: bool = Query(True, description="Include ML-generated insights"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Get comprehensive details about a specific food including:
    - Complete nutritional profile
    - Country-specific availability and variations
    - Related foods and substitutes
    - ML-generated insights and recommendations
    - Cultural and seasonal information
    """
    return await service.get_food_details(food_id, enrich_with_ml=enrich_ml)

# Food Management Endpoints

@router.post("/foods", summary="Create Food", response_model=Dict[str, Any])
async def create_food(
    food_request: FoodCreateRequest = Body(..., description="Food creation data"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Create a new food entity in the knowledge graph.
    
    Features:
    - Automatic nutritional data validation
    - ML-powered data enrichment
    - Automatic relationship inference
    - Quality scoring and verification
    """
    return await service.create_food(food_request)

@router.put("/foods/{food_id}", summary="Update Food")
async def update_food(
    food_id: str = Path(..., description="Food identifier to update"),
    updates: Dict[str, Any] = Body(..., description="Fields to update"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Update an existing food entity with new information.
    """
    # This would call a service method to update the food
    # For now, return a placeholder response
    return {
        "success": True,
        "food_id": food_id,
        "message": "Food update functionality will be implemented",
        "updated_fields": list(updates.keys()),
        "timestamp": datetime.utcnow().isoformat()
    }

# Food Relationship and Analysis Endpoints

@router.post("/substitutes", summary="Find Food Substitutes")
async def find_food_substitutes(
    request: SubstitutionRequest = Body(..., description="Substitution request parameters"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Find intelligent food substitutes using ML algorithms.
    
    Considers:
    - Nutritional similarity and compatibility
    - Cultural and culinary appropriateness
    - Availability and cost factors
    - Dietary restrictions and allergens
    - Cooking method compatibility
    """
    return await service.find_substitutes(request)

@router.get("/foods/{food_id}/substitutes", summary="Get Food Substitutes (Simple)")
async def get_food_substitutes_simple(
    food_id: str = Path(..., description="Original food ID"),
    context: str = Query("general", description="Substitution context"),
    country: Optional[str] = Query(None, description="Country code for availability"),
    limit: int = Query(10, ge=1, le=50, description="Maximum substitutes"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Simplified GET endpoint for food substitutes.
    """
    request = SubstitutionRequest(
        food_id=food_id,
        context=context,
        country_code=country,
        limit=limit
    )
    return await service.find_substitutes(request)

@router.get("/foods/{food_id}/similar", summary="Find Similar Foods")
async def find_similar_foods(
    food_id: str = Path(..., description="Food ID for similarity search"),
    country: Optional[str] = Query(None, description="Country code filter"),
    limit: int = Query(10, ge=1, le=50, description="Maximum similar foods"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Find nutritionally and culturally similar foods using ML similarity algorithms.
    """
    # Get food details with similar foods included
    details = await service.get_food_details(food_id, enrich_with_ml=True)
    
    if not details:
        raise HTTPException(status_code=404, detail="Food not found")
    
    # Extract similar foods from relationships
    similar_foods = []
    relationships = details.get('relationships', {})
    
    if 'similar_nutritionally' in relationships:
        similar_foods.extend(relationships['similar_nutritionally'][:limit])
    
    return {
        'food_id': food_id,
        'similar_foods': similar_foods,
        'country_filter': country,
        'limit': limit,
        'timestamp': datetime.utcnow().isoformat()
    }

# Cultural and Regional Analysis Endpoints

@router.post("/cultural-analysis", summary="Analyze Cultural Food Patterns")
async def analyze_cultural_patterns(
    request: CulturalAnalysisRequest = Body(..., description="Cultural analysis parameters"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Perform comprehensive cultural food pattern analysis for a specific country.
    
    Provides insights on:
    - Traditional food categories and preferences
    - Seasonal eating patterns
    - Regional variations and specialties
    - Nutritional characteristics of cultural diet
    - Food network analysis and influential foods
    """
    return await service.analyze_cultural_patterns(request)

@router.get("/countries/{country_code}/food-culture", summary="Get Country Food Culture")
async def get_country_food_culture(
    country_code: str = Path(..., description="ISO 2-letter country code"),
    include_seasonal: bool = Query(True, description="Include seasonal patterns"),
    include_nutrition: bool = Query(True, description="Include nutritional analysis"),
    include_network: bool = Query(True, description="Include network analysis"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Get comprehensive food culture information for a specific country.
    """
    request = CulturalAnalysisRequest(
        country_code=country_code,
        include_seasonal=include_seasonal,
        include_nutrition=include_nutrition,
        include_network=include_network
    )
    return await service.analyze_cultural_patterns(request)

@router.get("/countries/{country_code}/availability", summary="Get Country Food Availability")
async def get_country_availability(
    country_code: str = Path(..., description="ISO 2-letter country code"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Get comprehensive food availability data for a specific country.
    """
    return await service.get_country_food_availability(country_code)

# Seasonal and Temporal Analysis Endpoints

@router.post("/seasonal-prediction", summary="Predict Seasonal Availability")
async def predict_seasonal_availability(
    request: SeasonalPredictionRequest = Body(..., description="Seasonal prediction parameters"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Predict seasonal food availability using ML models and historical data.
    
    Factors considered:
    - Historical seasonal patterns
    - Climate and growing conditions
    - Import/export dependencies
    - Regional production capabilities
    """
    return await service.predict_seasonal_availability(request)

@router.get("/foods/{food_id}/seasonal/{country_code}", summary="Get Seasonal Info")
async def get_seasonal_info(
    food_id: str = Path(..., description="Food identifier"),
    country_code: str = Path(..., description="Country code"),
    target_date: Optional[datetime] = Query(None, description="Target date for prediction"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Get seasonal availability information for a specific food and country.
    """
    request = SeasonalPredictionRequest(
        food_id=food_id,
        country_code=country_code,
        target_date=target_date
    )
    return await service.predict_seasonal_availability(request)

# Data Management and Synchronization Endpoints

@router.post("/ingest", summary="Ingest Data from APIs")
async def ingest_data_from_apis(
    request: DataIngestionRequest = Body(..., description="Data ingestion parameters"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Ingest food data from external APIs and integrate into the knowledge graph.
    
    Supported APIs:
    - USDA FoodData Central
    - Open Food Facts
    - Nutritionix
    - Spoonacular
    - Regional food databases
    """
    return await service.ingest_data_from_apis(request)

@router.post("/sync", summary="Synchronize with External APIs")
async def sync_with_apis(
    force_sync: bool = Query(False, description="Force synchronization even if not scheduled"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Synchronize existing food data with external APIs to ensure freshness and accuracy.
    """
    return await service.sync_with_apis(force_sync=force_sync)

@router.get("/sync/status", summary="Get Sync Status")
async def get_sync_status(
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Get current synchronization status and statistics.
    """
    analytics = await service.get_system_analytics()
    
    return {
        'sync_status': analytics.get('sync_status', {}),
        'api_stats': analytics.get('api_stats', {}),
        'last_updated': analytics.get('timestamp')
    }

# Quality Assessment and Analytics Endpoints

@router.post("/quality-assessment", summary="Run Quality Assessment")
async def run_quality_assessment(
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Run comprehensive data quality assessment across the entire food knowledge graph.
    
    Evaluates:
    - Data completeness and accuracy
    - Relationship consistency
    - API data quality
    - ML model performance
    - System health metrics
    """
    return await service.run_quality_assessment()

@router.get("/analytics", summary="Get System Analytics")
async def get_system_analytics(
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Get comprehensive system analytics and performance metrics.
    """
    return await service.get_system_analytics()

@router.get("/metrics", summary="Get Performance Metrics")
async def get_performance_metrics(
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Get real-time performance metrics for the food knowledge graph system.
    """
    analytics = await service.get_system_analytics()
    
    return {
        'performance_metrics': analytics.get('performance_metrics', {}),
        'service_metrics': analytics.get('service_metrics', {}),
        'quality_metrics': analytics.get('quality_metrics', {}),
        'timestamp': analytics.get('timestamp')
    }

# Advanced Search and Discovery Endpoints

@router.get("/discover", summary="Discover Foods")
async def discover_foods(
    category: Optional[str] = Query(None, description="Food category to discover"),
    country: Optional[str] = Query(None, description="Country for regional discovery"),
    seasonal_only: bool = Query(False, description="Only show seasonal foods"),
    trending: bool = Query(False, description="Show trending foods"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Discover interesting and relevant foods based on various criteria.
    """
    # Build discovery query based on parameters
    search_query = FoodSearchQuery(
        query="",  # Empty query for discovery
        category=category,
        country_codes=[country] if country else None,
        limit=limit,
        use_ml_ranking=True
    )
    
    results = await service.search_foods(search_query)
    
    # Add discovery-specific metadata
    results['discovery_type'] = 'general'
    if seasonal_only:
        results['discovery_type'] = 'seasonal'
    elif trending:
        results['discovery_type'] = 'trending'
    
    return results

@router.get("/recommendations/{food_id}", summary="Get Food Recommendations")
async def get_food_recommendations(
    food_id: str = Path(..., description="Base food for recommendations"),
    recommendation_type: str = Query("similar", description="Type: similar, complementary, substitute"),
    country: Optional[str] = Query(None, description="Country context"),
    limit: int = Query(10, ge=1, le=50, description="Maximum recommendations"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Get AI-powered food recommendations based on a specific food.
    """
    if recommendation_type == "substitute":
        request = SubstitutionRequest(
            food_id=food_id,
            country_code=country,
            limit=limit
        )
        return await service.find_substitutes(request)
    else:
        # Get food details with relationships
        details = await service.get_food_details(food_id, enrich_with_ml=True)
        
        if not details:
            raise HTTPException(status_code=404, detail="Food not found")
        
        relationships = details.get('relationships', {})
        
        if recommendation_type == "similar":
            recommendations = relationships.get('similar_nutritionally', [])
        elif recommendation_type == "complementary":
            recommendations = relationships.get('complementary', [])
        else:
            recommendations = []
        
        return {
            'food_id': food_id,
            'recommendation_type': recommendation_type,
            'recommendations': recommendations[:limit],
            'country_context': country,
            'timestamp': datetime.utcnow().isoformat()
        }

# Batch Operations Endpoints

@router.post("/batch/search", summary="Batch Food Search")
async def batch_food_search(
    queries: List[str] = Body(..., description="List of search queries"),
    country: Optional[str] = Body(None, description="Country filter for all queries"),
    limit_per_query: int = Body(10, ge=1, le=50, description="Results per query"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Perform batch food searches for multiple queries efficiently.
    """
    results = []
    
    for query in queries:
        search_query = FoodSearchQuery(
            query=query,
            country_codes=[country] if country else None,
            limit=limit_per_query,
            use_ml_ranking=True
        )
        
        try:
            result = await service.search_foods(search_query)
            results.append({
                'query': query,
                'success': True,
                'foods': result.get('foods', []),
                'total_found': result.get('total_found', 0)
            })
        except Exception as e:
            results.append({
                'query': query,
                'success': False,
                'error': str(e),
                'foods': [],
                'total_found': 0
            })
    
    return {
        'batch_results': results,
        'total_queries': len(queries),
        'successful_queries': sum(1 for r in results if r['success']),
        'timestamp': datetime.utcnow().isoformat()
    }

@router.post("/batch/details", summary="Batch Food Details")
async def batch_food_details(
    food_ids: List[str] = Body(..., description="List of food IDs"),
    enrich_ml: bool = Body(True, description="Include ML enrichment"),
    service: FoodKnowledgeService = Depends(get_food_knowledge_service)
):
    """
    Get detailed information for multiple foods in a single request.
    """
    results = []
    
    for food_id in food_ids:
        try:
            details = await service.get_food_details(food_id, enrich_with_ml=enrich_ml)
            if details:
                results.append({
                    'food_id': food_id,
                    'success': True,
                    'details': details
                })
            else:
                results.append({
                    'food_id': food_id,
                    'success': False,
                    'error': 'Food not found',
                    'details': None
                })
        except Exception as e:
            results.append({
                'food_id': food_id,
                'success': False,
                'error': str(e),
                'details': None
            })
    
    return {
        'batch_results': results,
        'total_requested': len(food_ids),
        'successful_requests': sum(1 for r in results if r['success']),
        'timestamp': datetime.utcnow().isoformat()
    }

# Error handling
@router.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@router.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception in food knowledge API: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )