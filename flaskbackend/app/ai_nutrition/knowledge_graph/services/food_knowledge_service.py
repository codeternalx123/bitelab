"""
Food Knowledge Graph Services
============================

High-level service layer for food knowledge graph operations,
providing business logic and API endpoints for the food AI system.

Services:
- Food Search and Discovery
- Nutritional Analysis
- Cultural Food Insights  
- Substitution Recommendations
- Seasonal Availability
- Quality Assessment
- Data Synchronization

Author: Wellomex AI Nutrition Team
Version: 1.0.0
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from uuid import uuid4
import aioredis
from fastapi import HTTPException

from ..models.food_knowledge_models import (
    FoodEntity, FoodCategory, AllergenType, DietaryRestriction,
    MacroNutrient, MicroNutrient, CountrySpecificData
)
from ..core.knowledge_graph_engine import FoodKnowledgeGraphEngine, SearchRequest
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses

class FoodSearchQuery(BaseModel):
    """Food search query parameters"""
    query: str = Field("", description="Search query text")
    category: Optional[str] = Field(None, description="Food category filter")
    country_codes: Optional[List[str]] = Field(None, description="Country codes for filtering")
    allergen_free: Optional[List[str]] = Field(None, description="Allergens to avoid")
    dietary_compatible: Optional[List[str]] = Field(None, description="Dietary restrictions to match")
    nutritional_profiles: Optional[List[str]] = Field(None, description="Nutritional profile filters")
    include_similar: bool = Field(False, description="Include similar foods")
    use_ml_ranking: bool = Field(True, description="Use ML for result ranking")
    limit: int = Field(50, ge=1, le=200, description="Maximum results")
    offset: int = Field(0, ge=0, description="Result offset for pagination")

class FoodCreateRequest(BaseModel):
    """Request to create new food entity"""
    name: str = Field(..., description="Food name")
    scientific_name: Optional[str] = Field(None, description="Scientific name")
    common_names: List[str] = Field(default_factory=list, description="Alternative names")
    category: str = Field(..., description="Primary food category")
    subcategories: List[str] = Field(default_factory=list, description="Subcategories")
    
    # Nutritional data
    calories: Optional[float] = Field(None, description="Calories per 100g")
    protein: Optional[float] = Field(None, description="Protein grams per 100g")
    fat: Optional[float] = Field(None, description="Fat grams per 100g")
    carbohydrates: Optional[float] = Field(None, description="Carb grams per 100g")
    fiber: Optional[float] = Field(None, description="Fiber grams per 100g")
    sodium: Optional[float] = Field(None, description="Sodium mg per 100g")
    
    # Additional properties
    allergens: List[str] = Field(default_factory=list, description="Known allergens")
    dietary_restrictions: List[str] = Field(default_factory=list, description="Compatible diets")
    country_data: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Country-specific data")
    
    auto_enrich: bool = Field(True, description="Auto-enrich with ML")

class SubstitutionRequest(BaseModel):
    """Request for food substitution recommendations"""
    food_id: str = Field(..., description="Original food ID")
    context: str = Field("general", description="Substitution context (cooking, dietary, etc.)")
    country_code: Optional[str] = Field(None, description="Country for availability")
    dietary_restrictions: Optional[List[str]] = Field(None, description="Required dietary constraints")
    limit: int = Field(10, ge=1, le=50, description="Maximum substitutes")

class CulturalAnalysisRequest(BaseModel):
    """Request for cultural food analysis"""
    country_code: str = Field(..., description="Country code (ISO 2-letter)")
    include_seasonal: bool = Field(True, description="Include seasonal patterns")
    include_nutrition: bool = Field(True, description="Include nutritional analysis")
    include_network: bool = Field(True, description="Include network analysis")

class SeasonalPredictionRequest(BaseModel):
    """Request for seasonal availability prediction"""
    food_id: str = Field(..., description="Food ID")
    country_code: str = Field(..., description="Country code")
    target_date: Optional[datetime] = Field(None, description="Target date for prediction")

class DataIngestionRequest(BaseModel):
    """Request for data ingestion from APIs"""
    search_queries: List[str] = Field(..., description="Queries to search for")
    country_codes: Optional[List[str]] = Field(None, description="Priority countries")
    max_foods_per_query: int = Field(100, ge=10, le=500, description="Max foods per query")
    force_update: bool = Field(False, description="Force update existing foods")

class FoodKnowledgeService:
    """Main service class for food knowledge graph operations"""
    
    def __init__(self, engine: FoodKnowledgeGraphEngine):
        self.engine = engine
        
        # Service metrics
        self.service_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'last_reset': datetime.utcnow()
        }
    
    async def search_foods(self, query: FoodSearchQuery) -> Dict[str, Any]:
        """Search for foods using the knowledge graph"""
        start_time = datetime.utcnow()
        
        try:
            self.service_metrics['total_requests'] += 1
            
            # Convert to internal search request
            search_request = SearchRequest(
                query=query.query,
                category=FoodCategory(query.category) if query.category else None,
                country_codes=query.country_codes,
                allergen_free=[AllergenType(a) for a in query.allergen_free] if query.allergen_free else None,
                dietary_compatible=[DietaryRestriction(d) for d in query.dietary_compatible] if query.dietary_compatible else None,
                nutritional_profiles=query.nutritional_profiles,
                include_similar=query.include_similar,
                use_ml_ranking=query.use_ml_ranking,
                limit=query.limit,
                offset=query.offset
            )
            
            # Execute search
            results = await self.engine.search_foods(search_request)
            
            # Convert food entities to serializable format
            serialized_foods = []
            for food in results.get('foods', []):
                food_dict = self._serialize_food_entity(food)
                serialized_foods.append(food_dict)
            
            response = {
                'foods': serialized_foods,
                'total_found': results.get('total_found', 0),
                'has_more': len(serialized_foods) == query.limit,
                'pagination': {
                    'limit': query.limit,
                    'offset': query.offset,
                    'next_offset': query.offset + len(serialized_foods) if len(serialized_foods) == query.limit else None
                },
                'metadata': {
                    'source': results.get('source', 'unknown'),
                    'ml_enhanced': results.get('ml_enhanced', False),
                    'api_sources': results.get('api_sources', []),
                    'response_time': results.get('response_time', 0.0),
                    'timestamp': results.get('timestamp')
                }
            }
            
            # Include similar foods if requested
            if query.include_similar and results.get('similar_foods'):
                similar_foods = []
                for food, similarity in results['similar_foods']:
                    similar_dict = self._serialize_food_entity(food)
                    similar_dict['similarity_score'] = float(similarity)
                    similar_foods.append(similar_dict)
                response['similar_foods'] = similar_foods
            
            self.service_metrics['successful_requests'] += 1
            self._update_response_time((datetime.utcnow() - start_time).total_seconds())
            
            return response
            
        except Exception as e:
            self.service_metrics['failed_requests'] += 1
            logger.error(f"Food search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    async def get_food_details(self, food_id: str, enrich_with_ml: bool = True) -> Dict[str, Any]:
        """Get detailed information about a specific food"""
        
        try:
            details = await self.engine.get_food_details(food_id, enrich_with_ml=enrich_with_ml)
            
            if not details:
                raise HTTPException(status_code=404, detail="Food not found")
            
            # Serialize the response
            response = {
                'food': self._serialize_food_entity(details['food']),
                'quality_score': details['quality_score'],
                'relationships': {},
                'ml_insights': details.get('ml_insights', {}),
                'metadata': {
                    'last_updated': details['food'].updated_at.isoformat(),
                    'data_sources': details['food'].data_sources,
                    'confidence_score': float(details['food'].confidence_score)
                }
            }
            
            # Process relationships
            relationships = details.get('relationships', {})
            for rel_type, rel_foods in relationships.items():
                if isinstance(rel_foods, list):
                    serialized_rels = []
                    for item in rel_foods:
                        if isinstance(item, tuple):
                            food_entity, score = item
                            rel_dict = self._serialize_food_entity(food_entity)
                            rel_dict['relationship_score'] = float(score) if isinstance(score, (int, float, Decimal)) else score
                            serialized_rels.append(rel_dict)
                        else:
                            serialized_rels.append(self._serialize_food_entity(item))
                    response['relationships'][rel_type] = serialized_rels
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get food details failed for {food_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get food details: {str(e)}")
    
    async def create_food(self, request: FoodCreateRequest) -> Dict[str, Any]:
        """Create a new food entity"""
        
        try:
            # Convert request to food entity data
            food_data = {
                'name': request.name,
                'scientific_name': request.scientific_name,
                'common_names': request.common_names,
                'category': request.category,
                'subcategories': request.subcategories,
                'allergens': request.allergens,
                'dietary_restrictions': request.dietary_restrictions,
                'country_data': request.country_data
            }
            
            # Add nutritional data if provided
            if any([request.calories, request.protein, request.fat, request.carbohydrates]):
                food_data['macro_nutrients'] = {
                    'calories': request.calories or 0,
                    'protein': request.protein or 0,
                    'fat': request.fat or 0,
                    'carbohydrates': request.carbohydrates or 0,
                    'fiber': request.fiber or 0,
                    'sodium': request.sodium or 0
                }
            
            # Create the food entity
            success, food_id = await self.engine.create_food_entity(
                food_data=food_data,
                source="api_create",
                auto_enrich=request.auto_enrich
            )
            
            if success:
                return {
                    'success': True,
                    'food_id': food_id,
                    'message': 'Food created successfully',
                    'auto_enriched': request.auto_enrich
                }
            else:
                raise HTTPException(status_code=400, detail="Failed to create food entity")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Create food failed: {e}")
            raise HTTPException(status_code=500, detail=f"Food creation failed: {str(e)}")
    
    async def find_substitutes(self, request: SubstitutionRequest) -> Dict[str, Any]:
        """Find food substitutes using ML engine"""
        
        try:
            # Convert dietary restrictions
            dietary_restrictions = None
            if request.dietary_restrictions:
                dietary_restrictions = [DietaryRestriction(d) for d in request.dietary_restrictions]
            
            # Get substitutes
            substitutes = await self.engine.find_food_substitutes_ml(
                food_id=request.food_id,
                context=request.context,
                country_code=request.country_code,
                dietary_restrictions=dietary_restrictions,
                limit=request.limit
            )
            
            # Serialize response
            substitute_list = []
            for food_entity, score, insights in substitutes:
                substitute_dict = {
                    'food': self._serialize_food_entity(food_entity),
                    'substitution_score': float(score),
                    'insights': insights
                }
                substitute_list.append(substitute_dict)
            
            return {
                'original_food_id': request.food_id,
                'context': request.context,
                'country_code': request.country_code,
                'substitutes': substitute_list,
                'total_found': len(substitute_list),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Find substitutes failed for {request.food_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Substitution search failed: {str(e)}")
    
    async def analyze_cultural_patterns(self, request: CulturalAnalysisRequest) -> Dict[str, Any]:
        """Analyze cultural food patterns for a country"""
        
        try:
            analysis = await self.engine.analyze_cultural_food_patterns(request.country_code)
            
            if 'error' in analysis:
                raise HTTPException(status_code=404, detail=analysis['error'])
            
            # Enhance with additional data if requested
            if request.include_network and 'network_insights' not in analysis:
                # Add network analysis if not already included
                pass
            
            return {
                'country_code': request.country_code,
                'analysis': analysis,
                'parameters': {
                    'include_seasonal': request.include_seasonal,
                    'include_nutrition': request.include_nutrition,
                    'include_network': request.include_network
                },
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Cultural analysis failed for {request.country_code}: {e}")
            raise HTTPException(status_code=500, detail=f"Cultural analysis failed: {str(e)}")
    
    async def predict_seasonal_availability(self, request: SeasonalPredictionRequest) -> Dict[str, Any]:
        """Predict seasonal food availability"""
        
        try:
            prediction = await self.engine.predict_seasonal_availability(
                food_id=request.food_id,
                country_code=request.country_code,
                target_date=request.target_date
            )
            
            if 'error' in prediction:
                raise HTTPException(status_code=404, detail=prediction['error'])
            
            return {
                'prediction': prediction,
                'request': {
                    'food_id': request.food_id,
                    'country_code': request.country_code,
                    'target_date': request.target_date.isoformat() if request.target_date else None
                },
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Seasonal prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Seasonal prediction failed: {str(e)}")
    
    async def ingest_data_from_apis(self, request: DataIngestionRequest) -> Dict[str, Any]:
        """Ingest food data from external APIs"""
        
        try:
            result = await self.engine.ingest_from_apis(
                search_queries=request.search_queries,
                country_codes=request.country_codes,
                max_foods_per_query=request.max_foods_per_query
            )
            
            # Convert Decimal values to float for JSON serialization
            api_costs = {}
            for api, cost in result.api_costs.items():
                api_costs[api] = float(cost)
            
            return {
                'ingestion_result': {
                    'total_processed': result.total_processed,
                    'successful_imports': result.successful_imports,
                    'failed_imports': result.failed_imports,
                    'duplicates_found': result.duplicates_found,
                    'new_relationships': result.new_relationships,
                    'processing_time': result.processing_time,
                    'api_costs': api_costs,
                    'errors': result.errors
                },
                'request_parameters': {
                    'search_queries': request.search_queries,
                    'country_codes': request.country_codes,
                    'max_foods_per_query': request.max_foods_per_query,
                    'force_update': request.force_update
                },
                'completed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise HTTPException(status_code=500, detail=f"Data ingestion failed: {str(e)}")
    
    async def sync_with_apis(self, force_sync: bool = False) -> Dict[str, Any]:
        """Synchronize existing data with external APIs"""
        
        try:
            result = await self.engine.sync_with_external_apis(force_sync=force_sync)
            
            # Convert Decimal values for JSON serialization
            if 'api_costs' in result:
                api_costs = {}
                for api, cost in result['api_costs'].items():
                    api_costs[api] = float(cost) if isinstance(cost, Decimal) else cost
                result['api_costs'] = api_costs
            
            return {
                'sync_result': result,
                'force_sync': force_sync,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"API synchronization failed: {e}")
            raise HTTPException(status_code=500, detail=f"API sync failed: {str(e)}")
    
    async def run_quality_assessment(self) -> Dict[str, Any]:
        """Run comprehensive data quality assessment"""
        
        try:
            assessment = await self.engine.run_quality_assessment()
            
            return {
                'quality_assessment': assessment,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            raise HTTPException(status_code=500, detail=f"Quality assessment failed: {str(e)}")
    
    async def get_system_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics"""
        
        try:
            analytics = await self.engine.get_system_analytics()
            
            # Add service-level metrics
            analytics['service_metrics'] = self.service_metrics
            
            return analytics
            
        except Exception as e:
            logger.error(f"System analytics failed: {e}")
            raise HTTPException(status_code=500, detail=f"System analytics failed: {str(e)}")
    
    async def get_country_food_availability(self, country_code: str) -> Dict[str, Any]:
        """Get food availability data for a specific country"""
        
        try:
            availability = await self.engine.db_manager.get_country_food_availability(country_code)
            
            if not availability:
                raise HTTPException(status_code=404, detail=f"No data found for country {country_code}")
            
            return {
                'country_availability': availability,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Country availability failed for {country_code}: {e}")
            raise HTTPException(status_code=500, detail=f"Country availability failed: {str(e)}")
    
    # Helper Methods
    
    def _serialize_food_entity(self, food: FoodEntity) -> Dict[str, Any]:
        """Convert FoodEntity to JSON-serializable dictionary"""
        
        food_dict = {
            'food_id': food.food_id,
            'name': food.name,
            'scientific_name': food.scientific_name,
            'common_names': food.common_names,
            'category': food.category.value if food.category else None,
            'subcategories': food.subcategories,
            'food_group': food.food_group,
            'taxonomic_family': food.taxonomic_family,
            'confidence_score': float(food.confidence_score),
            'verification_status': food.verification_status,
            'data_sources': food.data_sources,
            'created_at': food.created_at.isoformat(),
            'updated_at': food.updated_at.isoformat()
        }
        
        # Add nutritional data
        if food.macro_nutrients:
            macro = food.macro_nutrients
            food_dict['macro_nutrients'] = {
                'calories': float(macro.calories),
                'protein': float(macro.protein),
                'fat': float(macro.fat),
                'carbohydrates': float(macro.carbohydrates),
                'fiber': float(macro.fiber),
                'sugar': float(macro.sugar),
                'sodium': float(macro.sodium)
            }
            
            # Add optional macro nutrients
            if macro.saturated_fat is not None:
                food_dict['macro_nutrients']['saturated_fat'] = float(macro.saturated_fat)
        
        if food.micro_nutrients:
            micro = food.micro_nutrients
            micro_dict = {}
            
            # Add non-null micronutrients
            for attr_name in ['vitamin_c', 'vitamin_d', 'calcium', 'iron', 'potassium']:
                value = getattr(micro, attr_name, None)
                if value is not None:
                    micro_dict[attr_name] = float(value)
            
            if micro_dict:
                food_dict['micro_nutrients'] = micro_dict
        
        # Add allergens and dietary restrictions
        food_dict['allergens'] = [a.value for a in food.allergens]
        food_dict['dietary_restrictions'] = [d.value for d in food.dietary_restrictions]
        food_dict['nutritional_profiles'] = [p.value for p in food.nutritional_profiles]
        
        # Add physical properties
        if food.glycemic_index:
            food_dict['glycemic_index'] = food.glycemic_index
        if food.glycemic_load:
            food_dict['glycemic_load'] = float(food.glycemic_load)
        if food.typical_serving_size:
            food_dict['typical_serving_size'] = float(food.typical_serving_size)
        
        # Add country-specific data
        if food.country_data:
            country_data_dict = {}
            for country_code, country_data in food.country_data.items():
                country_data_dict[country_code] = {
                    'local_name': country_data.local_name,
                    'common_varieties': country_data.common_varieties,
                    'market_availability': float(country_data.market_availability),
                    'production_regions': country_data.production_regions,
                    'import_sources': country_data.import_sources,
                    'quality_grades': country_data.quality_grades,
                    'traditional_preparations': [p.value for p in country_data.traditional_preparations]
                }
                
                # Add seasonal availability
                if country_data.seasonal_availability:
                    seasonal_dict = {}
                    for season, available in country_data.seasonal_availability.items():
                        seasonal_dict[season.value] = available
                    country_data_dict[country_code]['seasonal_availability'] = seasonal_dict
            
            food_dict['country_data'] = country_data_dict
        
        # Add external IDs
        if food.external_ids:
            food_dict['external_ids'] = food.external_ids
        
        return food_dict
    
    def _update_response_time(self, response_time: float):
        """Update average response time metric"""
        current_avg = self.service_metrics['avg_response_time']
        total_requests = self.service_metrics['total_requests']
        
        if total_requests > 1:
            new_avg = ((current_avg * (total_requests - 1)) + response_time) / total_requests
            self.service_metrics['avg_response_time'] = new_avg

class FoodKnowledgeServiceFactory:
    """Factory for creating food knowledge service instances"""
    
    _instance = None
    _engine = None
    
    @classmethod
    async def get_service(
        cls,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password", 
        redis_url: str = "redis://localhost:6379"
    ) -> FoodKnowledgeService:
        """Get singleton service instance"""
        
        if cls._instance is None:
            # Initialize engine if needed
            if cls._engine is None:
                from ..core.knowledge_graph_engine import create_food_knowledge_engine
                cls._engine = await create_food_knowledge_engine(
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_password=neo4j_password,
                    redis_url=redis_url
                )
            
            cls._instance = FoodKnowledgeService(cls._engine)
        
        return cls._instance
    
    @classmethod
    async def shutdown(cls):
        """Shutdown the service and engine"""
        if cls._engine:
            await cls._engine.close()
            cls._engine = None
        cls._instance = None