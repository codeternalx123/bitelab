"""
Food Knowledge Graph Core Engine
===============================

Central orchestrator for the food knowledge graph system, integrating
all components including database management, API clients, ML engines,
and real-time data synchronization.

Core Responsibilities:
- Coordinate data ingestion from multiple APIs
- Manage knowledge graph relationships and inference
- Orchestrate ML model training and inference
- Handle real-time data updates and synchronization
- Provide unified interface for food knowledge operations
- Manage caching and performance optimization
- Handle data quality and consistency validation

Author: Wellomex AI Nutrition Team
Version: 1.0.0
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from uuid import uuid4
import aioredis
from contextlib import asynccontextmanager

from ..models.food_knowledge_models import (
    FoodEntity, FoodRelationship, CountrySpecificData,
    FoodCategory, AllergenType, DietaryRestriction,
    APISourceMapping, DataQualityMetrics
)
from ..graph_db.neo4j_manager import GraphDatabaseManager, FoodNetworkAnalyzer
from ..api_clients.api_manager import APIClientManager, APIResponse
from ..ml_engines.food_ai_engine import FoodKnowledgeMLEngine

logger = logging.getLogger(__name__)

@dataclass
class SyncConfiguration:
    """Configuration for data synchronization"""
    enabled: bool = True
    sync_interval_hours: int = 24
    batch_size: int = 100
    max_concurrent_requests: int = 10
    priority_countries: List[str] = field(default_factory=lambda: ['US', 'GB', 'FR', 'DE', 'JP'])
    auto_quality_check: bool = True
    ml_inference_enabled: bool = True

@dataclass
class SearchRequest:
    """Standardized search request structure"""
    query: str = ""
    category: Optional[FoodCategory] = None
    country_codes: Optional[List[str]] = None
    allergen_free: Optional[List[AllergenType]] = None
    dietary_compatible: Optional[List[DietaryRestriction]] = None
    nutritional_profiles: Optional[List[str]] = None
    include_similar: bool = False
    use_ml_ranking: bool = True
    limit: int = 50
    offset: int = 0

@dataclass
class IngestionResult:
    """Result of data ingestion operation"""
    total_processed: int = 0
    successful_imports: int = 0
    failed_imports: int = 0
    duplicates_found: int = 0
    new_relationships: int = 0
    errors: List[str] = field(default_factory=list)
    processing_time: Optional[float] = None
    api_costs: Dict[str, Decimal] = field(default_factory=dict)

class FoodKnowledgeGraphEngine:
    """Main engine for food knowledge graph operations"""
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j", 
        neo4j_password: str = "password",
        redis_url: str = "redis://localhost:6379"
    ):
        # Core components
        self.db_manager = GraphDatabaseManager(neo4j_uri, neo4j_user, neo4j_password, redis_url)
        self.api_manager = APIClientManager(redis_url)
        self.ml_engine = FoodKnowledgeMLEngine(self.db_manager)
        self.network_analyzer = FoodNetworkAnalyzer(self.db_manager)
        
        # Configuration
        self.sync_config = SyncConfiguration()
        
        # State tracking
        self.is_initialized = False
        self.last_sync_time: Optional[datetime] = None
        self.active_sync_tasks: Set[str] = set()
        
        # Performance metrics
        self.performance_metrics = {
            'total_queries': 0,
            'avg_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'api_success_rate': 0.0,
            'last_updated': datetime.utcnow()
        }
        
        # Data quality tracking
        self.quality_metrics = {
            'total_foods': 0,
            'verified_foods': 0,
            'avg_completeness_score': 0.0,
            'relationship_count': 0,
            'last_quality_check': None
        }
        
    async def initialize(self):
        """Initialize all engine components"""
        logger.info("Initializing Food Knowledge Graph Engine")
        
        start_time = datetime.utcnow()
        
        try:
            # Initialize database
            await self.db_manager.initialize()
            
            # Initialize API clients
            await self.api_manager.initialize()
            
            # Initialize ML engines
            await self.ml_engine.initialize_models()
            
            # Load existing data metrics
            await self._load_metrics()
            
            # Start background tasks
            asyncio.create_task(self._background_sync_scheduler())
            asyncio.create_task(self._background_quality_monitor())
            
            self.is_initialized = True
            initialization_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"Food Knowledge Graph Engine initialized in {initialization_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            return False
    
    async def close(self):
        """Gracefully shutdown the engine"""
        logger.info("Shutting down Food Knowledge Graph Engine")
        
        # Cancel active sync tasks
        for task_id in self.active_sync_tasks.copy():
            await self._cancel_sync_task(task_id)
        
        # Close components
        await self.ml_engine.close() if hasattr(self.ml_engine, 'close') else None
        await self.api_manager.close()
        await self.db_manager.close()
        
        logger.info("Engine shutdown complete")

    # Core Food Operations
    
    async def search_foods(self, request: SearchRequest) -> Dict[str, Any]:
        """Comprehensive food search with ML enhancement"""
        start_time = datetime.utcnow()
        
        try:
            # Update metrics
            self.performance_metrics['total_queries'] += 1
            
            # Search in local database first
            local_results = await self.db_manager.search_foods(
                query=request.query,
                category=request.category,
                country_codes=request.country_codes,
                allergen_free=request.allergen_free,
                dietary_compatible=request.dietary_compatible,
                nutritional_profiles=request.nutritional_profiles,
                limit=request.limit,
                offset=request.offset
            )
            
            results = {
                'foods': local_results,
                'total_found': len(local_results),
                'source': 'local',
                'ml_enhanced': False,
                'api_sources': []
            }
            
            # If insufficient local results, search external APIs
            if len(local_results) < request.limit // 2 and request.query:
                api_results = await self._search_external_apis(request)
                if api_results:
                    results['foods'].extend(api_results['foods'])
                    results['total_found'] = len(results['foods'])
                    results['api_sources'] = api_results['sources']
            
            # Apply ML ranking if enabled
            if request.use_ml_ranking and results['foods']:
                ranked_foods = await self._apply_ml_ranking(results['foods'], request)
                results['foods'] = ranked_foods
                results['ml_enhanced'] = True
            
            # Find similar foods if requested
            if request.include_similar and results['foods']:
                similar_foods = await self._find_similar_foods_ml(
                    results['foods'][0], 
                    request.country_codes[0] if request.country_codes else None
                )
                results['similar_foods'] = similar_foods
            
            # Update performance metrics
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_response_time_metric(response_time)
            
            results['response_time'] = response_time
            results['timestamp'] = datetime.utcnow().isoformat()
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {'error': str(e), 'foods': []}
    
    async def get_food_details(self, food_id: str, enrich_with_ml: bool = True) -> Optional[Dict[str, Any]]:
        """Get comprehensive food details with ML enrichment"""
        
        # Get base food data
        food = await self.db_manager.get_food_entity(food_id)
        if not food:
            return None
        
        details = {
            'food': food,
            'relationships': {},
            'ml_insights': {},
            'quality_score': 0.0
        }
        
        # Get relationships
        details['relationships'] = {
            'substitutes': await self.db_manager.find_food_substitutes(food_id, limit=10),
            'similar_nutritionally': await self.db_manager.get_nutritionally_similar_foods(food_id, limit=10),
            'complementary': await self._find_complementary_foods(food_id)
        }
        
        # Add ML insights if enabled
        if enrich_with_ml and self.ml_engine:
            details['ml_insights'] = {
                'cultural_analysis': await self._get_cultural_insights(food),
                'seasonal_predictions': await self._get_seasonal_predictions(food),
                'health_scores': await self._calculate_health_scores(food),
                'sustainability_metrics': await self._get_sustainability_insights(food)
            }
        
        # Calculate quality score
        details['quality_score'] = await self._calculate_food_quality_score(food)
        
        return details
    
    async def create_food_entity(
        self, 
        food_data: Dict[str, Any], 
        source: str = "manual",
        auto_enrich: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """Create new food entity with optional ML enrichment"""
        
        try:
            # Create basic food entity
            food = self._dict_to_food_entity(food_data)
            food.data_sources.append(source)
            
            # Auto-enrich with ML if enabled
            if auto_enrich:
                food = await self._enrich_food_with_ml(food)
            
            # Save to database
            success = await self.db_manager.create_food_entity(food)
            
            if success:
                # Create automatic relationships
                await self._create_automatic_relationships(food)
                
                # Update quality metrics
                await self._update_quality_metrics()
                
                return True, food.food_id
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"Failed to create food entity: {e}")
            return False, None
    
    async def update_food_entity(
        self, 
        food_id: str, 
        updates: Dict[str, Any],
        source: str = "manual"
    ) -> bool:
        """Update existing food entity"""
        
        try:
            # Get existing food
            food = await self.db_manager.get_food_entity(food_id)
            if not food:
                return False
            
            # Apply updates
            for field, value in updates.items():
                if hasattr(food, field):
                    setattr(food, field, value)
            
            # Update metadata
            food.updated_at = datetime.utcnow()
            if source not in food.data_sources:
                food.data_sources.append(source)
            
            # Save changes
            return await self.db_manager.update_food_entity(food)
            
        except Exception as e:
            logger.error(f"Failed to update food entity {food_id}: {e}")
            return False

    # Data Ingestion and Synchronization
    
    async def ingest_from_apis(
        self, 
        search_queries: List[str],
        country_codes: Optional[List[str]] = None,
        max_foods_per_query: int = 100
    ) -> IngestionResult:
        """Ingest food data from external APIs"""
        
        start_time = datetime.utcnow()
        result = IngestionResult()
        
        try:
            country_codes = country_codes or self.sync_config.priority_countries
            
            for query in search_queries:
                logger.info(f"Ingesting data for query: '{query}'")
                
                # Search multiple API sources
                api_results = await self.api_manager.search_foods_multi_source(
                    query=query,
                    country_code=country_codes[0] if country_codes else None,
                    max_sources=3
                )
                
                batch_processed = 0
                for food_entity, api_source in api_results[:max_foods_per_query]:
                    try:
                        # Check for duplicates
                        existing = await self._find_duplicate_food(food_entity)
                        if existing:
                            result.duplicates_found += 1
                            # Merge data if it's an improvement
                            if await self._should_merge_food_data(existing, food_entity):
                                await self._merge_food_entities(existing, food_entity)
                            continue
                        
                        # Create new food entity
                        success = await self.db_manager.create_food_entity(food_entity)
                        if success:
                            result.successful_imports += 1
                            
                            # Create API source mapping
                            await self._create_api_mapping(food_entity, api_source)
                            
                        else:
                            result.failed_imports += 1
                        
                        result.total_processed += 1
                        batch_processed += 1
                        
                        # Batch processing with rate limiting
                        if batch_processed % self.sync_config.batch_size == 0:
                            await asyncio.sleep(1)  # Rate limiting
                        
                    except Exception as e:
                        result.failed_imports += 1
                        result.errors.append(f"Failed to process food from {api_source}: {str(e)}")
                        logger.error(f"Error processing food: {e}")
                
                # Small delay between queries
                await asyncio.sleep(0.5)
            
            # Generate relationships for new foods
            if result.successful_imports > 0:
                new_relationships = await self._generate_automatic_relationships()
                result.new_relationships = new_relationships
            
            result.processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"Ingestion completed: {result.successful_imports}/{result.total_processed} successful")
            
            return result
            
        except Exception as e:
            result.errors.append(f"Ingestion failed: {str(e)}")
            logger.error(f"API ingestion failed: {e}")
            return result
    
    async def sync_with_external_apis(self, force_sync: bool = False) -> Dict[str, Any]:
        """Synchronize existing food data with external APIs"""
        
        if not force_sync and not self._should_sync():
            return {'message': 'Sync not needed', 'skipped': True}
        
        sync_id = str(uuid4())
        self.active_sync_tasks.add(sync_id)
        
        try:
            logger.info("Starting food data synchronization")
            
            # Get foods that need updates
            stale_foods = await self._get_stale_foods()
            
            sync_results = {
                'sync_id': sync_id,
                'started_at': datetime.utcnow().isoformat(),
                'foods_to_update': len(stale_foods),
                'updated_foods': 0,
                'failed_updates': 0,
                'api_costs': {},
                'errors': []
            }
            
            # Process in batches
            for i in range(0, len(stale_foods), self.sync_config.batch_size):
                batch = stale_foods[i:i + self.sync_config.batch_size]
                
                # Process batch concurrently
                batch_tasks = []
                for food in batch:
                    task = self._sync_single_food(food)
                    batch_tasks.append(task)
                
                # Wait for batch completion with limited concurrency
                semaphore = asyncio.Semaphore(self.sync_config.max_concurrent_requests)
                async def limited_sync(food_sync_task):
                    async with semaphore:
                        return await food_sync_task
                
                batch_results = await asyncio.gather(
                    *[limited_sync(task) for task in batch_tasks],
                    return_exceptions=True
                )
                
                # Process results
                for result in batch_results:
                    if isinstance(result, Exception):
                        sync_results['failed_updates'] += 1
                        sync_results['errors'].append(str(result))
                    elif result.get('success'):
                        sync_results['updated_foods'] += 1
                    else:
                        sync_results['failed_updates'] += 1
            
            # Update sync timestamp
            self.last_sync_time = datetime.utcnow()
            sync_results['completed_at'] = self.last_sync_time.isoformat()
            
            # Get API statistics
            api_stats = await self.api_manager.get_api_statistics()
            sync_results['api_costs'] = {
                api: stats.get('total_cost', Decimal('0.0'))
                for api, stats in api_stats.get('apis', {}).items()
            }
            
            logger.info(f"Sync completed: {sync_results['updated_foods']} updated, {sync_results['failed_updates']} failed")
            
            return sync_results
            
        except Exception as e:
            logger.error(f"Synchronization failed: {e}")
            return {'error': str(e), 'sync_id': sync_id}
        
        finally:
            self.active_sync_tasks.discard(sync_id)

    # ML-Enhanced Operations
    
    async def find_food_substitutes_ml(
        self,
        food_id: str,
        context: str = "general",
        country_code: Optional[str] = None,
        dietary_restrictions: Optional[List[DietaryRestriction]] = None,
        limit: int = 10
    ) -> List[Tuple[FoodEntity, float, Dict[str, Any]]]:
        """Find food substitutes using ML engine"""
        
        # Get base food
        original_food = await self.db_manager.get_food_entity(food_id)
        if not original_food:
            return []
        
        # Get candidate foods from database
        candidates = await self.db_manager.search_foods(
            category=original_food.category,
            country_codes=[country_code] if country_code else None,
            dietary_compatible=dietary_restrictions,
            limit=500
        )
        
        # Use ML engine for substitution analysis
        ml_substitutes = await self.ml_engine.substitution_engine.find_substitutes(
            original_food=original_food,
            candidate_foods=candidates,
            context=context,
            country_code=country_code,
            top_k=limit
        )
        
        # Enhance results with additional insights
        enhanced_results = []
        for substitute_food, score in ml_substitutes:
            insights = {
                'substitution_reason': await self._get_substitution_reason(original_food, substitute_food, context),
                'nutritional_comparison': await self._compare_nutrition(original_food, substitute_food),
                'availability_score': await self._get_availability_score(substitute_food, country_code),
                'cost_comparison': await self._estimate_cost_comparison(original_food, substitute_food, country_code)
            }
            enhanced_results.append((substitute_food, score, insights))
        
        return enhanced_results
    
    async def analyze_cultural_food_patterns(self, country_code: str) -> Dict[str, Any]:
        """Analyze cultural food patterns using ML"""
        
        # Get country-specific foods
        country_foods = await self.db_manager.search_foods(
            country_codes=[country_code],
            limit=1000
        )
        
        if not country_foods:
            return {'error': f'No food data found for country {country_code}'}
        
        # Use ML engine for cultural analysis
        cultural_analysis = await self.ml_engine.cultural_analyzer.analyze_cultural_patterns(
            foods=country_foods,
            country_code=country_code
        )
        
        # Enhance with network analysis
        network_metrics = await self.network_analyzer.calculate_food_centrality(country_code)
        cultural_analysis['network_insights'] = {
            'influential_foods': self._get_most_influential_foods(network_metrics),
            'food_communities': await self.network_analyzer.find_food_communities(country_code)
        }
        
        # Add recommendations
        cultural_analysis['recommendations'] = await self._generate_cultural_recommendations(
            cultural_analysis, country_code
        )
        
        return cultural_analysis
    
    async def predict_seasonal_availability(
        self,
        food_id: str,
        country_code: str,
        target_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Predict seasonal availability using ML models"""
        
        target_date = target_date or datetime.utcnow()
        
        # Get food and historical data
        food = await self.db_manager.get_food_entity(food_id)
        if not food:
            return {'error': 'Food not found'}
        
        country_data = food.country_data.get(country_code)
        if not country_data:
            return {'error': f'No data for country {country_code}'}
        
        # Use ML for seasonal prediction
        prediction = {
            'food_id': food_id,
            'country_code': country_code,
            'target_date': target_date.isoformat(),
            'availability_score': 0.0,
            'peak_season': None,
            'factors': {},
            'confidence': 0.0
        }
        
        # Simple rule-based prediction (can be enhanced with ML)
        month = target_date.month
        seasonal_patterns = country_data.seasonal_availability
        
        # Calculate base availability
        base_availability = float(country_data.market_availability)
        
        # Adjust for season
        seasonal_multiplier = 1.0
        if seasonal_patterns:
            from ..models.food_knowledge_models import SeasonalAvailability
            
            if month in [3, 4, 5] and SeasonalAvailability.SPRING in seasonal_patterns:
                seasonal_multiplier = 1.3
            elif month in [6, 7, 8] and SeasonalAvailability.SUMMER in seasonal_patterns:
                seasonal_multiplier = 1.3
            elif month in [9, 10, 11] and SeasonalAvailability.FALL in seasonal_patterns:
                seasonal_multiplier = 1.3
            elif month in [12, 1, 2] and SeasonalAvailability.WINTER in seasonal_patterns:
                seasonal_multiplier = 1.3
            elif SeasonalAvailability.YEAR_ROUND in seasonal_patterns:
                seasonal_multiplier = 1.0
            else:
                seasonal_multiplier = 0.7
        
        prediction['availability_score'] = min(1.0, base_availability * seasonal_multiplier)
        prediction['confidence'] = 0.8 if seasonal_patterns else 0.5
        
        # Determine peak season
        if seasonal_patterns:
            peak_seasons = [season.value for season in seasonal_patterns.keys() if seasonal_patterns[season]]
            prediction['peak_season'] = peak_seasons[0] if peak_seasons else None
        
        prediction['factors'] = {
            'base_availability': base_availability,
            'seasonal_multiplier': seasonal_multiplier,
            'production_regions': country_data.production_regions,
            'import_dependency': len(country_data.import_sources) > 0
        }
        
        return prediction

    # Quality and Analytics
    
    async def run_quality_assessment(self) -> Dict[str, Any]:
        """Run comprehensive data quality assessment"""
        
        logger.info("Starting quality assessment")
        
        assessment = {
            'started_at': datetime.utcnow().isoformat(),
            'food_quality': {},
            'relationship_quality': {},
            'api_quality': {},
            'ml_model_performance': {},
            'recommendations': []
        }
        
        try:
            # Assess food data quality
            assessment['food_quality'] = await self._assess_food_data_quality()
            
            # Assess relationship quality
            assessment['relationship_quality'] = await self._assess_relationship_quality()
            
            # Assess API data quality
            assessment['api_quality'] = await self.api_manager.get_api_statistics()
            
            # Assess ML model performance
            assessment['ml_model_performance'] = await self.ml_engine.get_model_performance_metrics()
            
            # Generate recommendations
            assessment['recommendations'] = await self._generate_quality_recommendations(assessment)
            
            assessment['completed_at'] = datetime.utcnow().isoformat()
            
            # Update quality metrics
            self.quality_metrics.update({
                'last_quality_check': datetime.utcnow(),
                'avg_completeness_score': assessment['food_quality'].get('avg_completeness', 0.0),
                'total_foods': assessment['food_quality'].get('total_foods', 0),
                'verified_foods': assessment['food_quality'].get('verified_foods', 0)
            })
            
            logger.info("Quality assessment completed")
            return assessment
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            assessment['error'] = str(e)
            return assessment
    
    async def get_system_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics"""
        
        analytics = {
            'timestamp': datetime.utcnow().isoformat(),
            'performance_metrics': self.performance_metrics,
            'quality_metrics': self.quality_metrics,
            'database_stats': {},
            'api_stats': {},
            'ml_stats': {},
            'sync_status': {}
        }
        
        try:
            # Database statistics
            analytics['database_stats'] = await self._get_database_statistics()
            
            # API statistics
            analytics['api_stats'] = await self.api_manager.get_api_statistics()
            
            # ML model statistics
            analytics['ml_stats'] = await self.ml_engine.get_model_performance_metrics()
            
            # Sync status
            analytics['sync_status'] = {
                'last_sync': self.last_sync_time.isoformat() if self.last_sync_time else None,
                'active_sync_tasks': len(self.active_sync_tasks),
                'sync_enabled': self.sync_config.enabled
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get system analytics: {e}")
            analytics['error'] = str(e)
            return analytics

    # Helper Methods
    
    async def _search_external_apis(self, request: SearchRequest) -> Optional[Dict[str, Any]]:
        """Search external APIs for additional results"""
        
        try:
            api_results = await self.api_manager.search_foods_multi_source(
                query=request.query,
                country_code=request.country_codes[0] if request.country_codes else None,
                max_sources=2
            )
            
            # Convert to standardized format
            foods = [food for food, source in api_results]
            sources = list(set([source for food, source in api_results]))
            
            return {'foods': foods, 'sources': sources}
            
        except Exception as e:
            logger.error(f"External API search failed: {e}")
            return None
    
    async def _apply_ml_ranking(self, foods: List[FoodEntity], request: SearchRequest) -> List[FoodEntity]:
        """Apply ML-based ranking to search results"""
        
        if not self.ml_engine or not foods:
            return foods
        
        try:
            # Simple ranking based on confidence scores and completeness
            def rank_food(food):
                score = float(food.confidence_score)
                
                # Boost score for foods with complete nutritional data
                if food.macro_nutrients and food.micro_nutrients:
                    score += 0.2
                
                # Boost score for foods with country data
                if request.country_codes:
                    for country in request.country_codes:
                        if country in food.country_data:
                            score += 0.1
                            break
                
                # Boost score for foods with dietary restriction compatibility
                if request.dietary_compatible:
                    compatible_restrictions = set(request.dietary_compatible)
                    food_restrictions = set(food.dietary_restrictions)
                    if compatible_restrictions.issubset(food_restrictions):
                        score += 0.15
                
                return score
            
            ranked_foods = sorted(foods, key=rank_food, reverse=True)
            return ranked_foods
            
        except Exception as e:
            logger.error(f"ML ranking failed: {e}")
            return foods
    
    async def _find_similar_foods_ml(
        self, 
        food: FoodEntity, 
        country_code: Optional[str] = None
    ) -> List[Tuple[FoodEntity, float]]:
        """Find similar foods using ML similarity engine"""
        
        if not self.ml_engine.similarity_engine.is_trained:
            return []
        
        try:
            # Get candidate foods
            candidates = await self.db_manager.search_foods(
                category=food.category,
                country_codes=[country_code] if country_code else None,
                limit=200
            )
            
            # Use ML engine
            similar_foods = self.ml_engine.similarity_engine.find_similar_foods(
                target_food=food,
                candidate_foods=candidates,
                top_k=10
            )
            
            return similar_foods
            
        except Exception as e:
            logger.error(f"ML similarity search failed: {e}")
            return []
    
    def _should_sync(self) -> bool:
        """Check if synchronization should be performed"""
        if not self.sync_config.enabled:
            return False
        
        if not self.last_sync_time:
            return True
        
        time_since_sync = datetime.utcnow() - self.last_sync_time
        sync_interval = timedelta(hours=self.sync_config.sync_interval_hours)
        
        return time_since_sync >= sync_interval
    
    async def _get_stale_foods(self, limit: int = 1000) -> List[FoodEntity]:
        """Get foods that need updating"""
        # This would query for foods with old last_api_sync timestamps
        # For now, return empty list as placeholder
        return []
    
    async def _sync_single_food(self, food: FoodEntity) -> Dict[str, Any]:
        """Synchronize a single food with external APIs"""
        
        try:
            # Get updated data from APIs
            updated_food = await self.api_manager.get_food_details_multi_source(food.external_ids)
            
            if updated_food:
                # Merge new data
                merged_food = await self._merge_food_entities(food, updated_food)
                
                # Update in database
                success = await self.db_manager.update_food_entity(merged_food)
                return {'success': success, 'food_id': food.food_id}
            else:
                return {'success': False, 'error': 'No updated data found'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _background_sync_scheduler(self):
        """Background task for scheduled synchronization"""
        while True:
            try:
                if self._should_sync():
                    await self.sync_with_external_apis()
                
                # Sleep for 1 hour before checking again
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Background sync error: {e}")
                await asyncio.sleep(3600)
    
    async def _background_quality_monitor(self):
        """Background task for quality monitoring"""
        while True:
            try:
                if self.sync_config.auto_quality_check:
                    await self.run_quality_assessment()
                
                # Run quality check every 6 hours
                await asyncio.sleep(21600)
                
            except Exception as e:
                logger.error(f"Background quality monitor error: {e}")
                await asyncio.sleep(21600)
    
    def _update_response_time_metric(self, response_time: float):
        """Update average response time metric"""
        current_avg = self.performance_metrics['avg_response_time']
        total_queries = self.performance_metrics['total_queries']
        
        new_avg = ((current_avg * (total_queries - 1)) + response_time) / total_queries
        self.performance_metrics['avg_response_time'] = new_avg
        self.performance_metrics['last_updated'] = datetime.utcnow()
    
    # Placeholder methods for functionality to be implemented
    
    def _dict_to_food_entity(self, data: Dict[str, Any]) -> FoodEntity:
        """Convert dictionary to FoodEntity"""
        # Implementation depends on the data structure
        return FoodEntity()
    
    async def _enrich_food_with_ml(self, food: FoodEntity) -> FoodEntity:
        """Enrich food entity with ML predictions"""
        return food
    
    async def _create_automatic_relationships(self, food: FoodEntity):
        """Create automatic relationships for new food"""
        pass
    
    async def _update_quality_metrics(self):
        """Update system quality metrics"""
        pass
    
    async def _find_duplicate_food(self, food: FoodEntity) -> Optional[FoodEntity]:
        """Find potential duplicate food in database"""
        return None
    
    async def _should_merge_food_data(self, existing: FoodEntity, new: FoodEntity) -> bool:
        """Determine if food data should be merged"""
        return False
    
    async def _merge_food_entities(self, food1: FoodEntity, food2: FoodEntity) -> FoodEntity:
        """Merge two food entities"""
        return food1
    
    async def _create_api_mapping(self, food: FoodEntity, api_source: str):
        """Create API source mapping"""
        pass
    
    async def _generate_automatic_relationships(self) -> int:
        """Generate automatic relationships between foods"""
        return 0
    
    async def _load_metrics(self):
        """Load existing performance metrics"""
        pass
    
    async def _cancel_sync_task(self, task_id: str):
        """Cancel active sync task"""
        pass
    
    async def _find_complementary_foods(self, food_id: str) -> List[FoodEntity]:
        """Find complementary foods"""
        return []
    
    async def _get_cultural_insights(self, food: FoodEntity) -> Dict[str, Any]:
        """Get cultural insights for food"""
        return {}
    
    async def _get_seasonal_predictions(self, food: FoodEntity) -> Dict[str, Any]:
        """Get seasonal predictions"""
        return {}
    
    async def _calculate_health_scores(self, food: FoodEntity) -> Dict[str, float]:
        """Calculate health scores"""
        return {}
    
    async def _get_sustainability_insights(self, food: FoodEntity) -> Dict[str, Any]:
        """Get sustainability insights"""
        return {}
    
    async def _calculate_food_quality_score(self, food: FoodEntity) -> float:
        """Calculate food data quality score"""
        return 0.8
    
    async def _get_substitution_reason(self, original: FoodEntity, substitute: FoodEntity, context: str) -> str:
        """Get reason for substitution"""
        return "Similar nutritional profile"
    
    async def _compare_nutrition(self, food1: FoodEntity, food2: FoodEntity) -> Dict[str, Any]:
        """Compare nutritional profiles"""
        return {}
    
    async def _get_availability_score(self, food: FoodEntity, country_code: Optional[str]) -> float:
        """Get availability score"""
        return 0.8
    
    async def _estimate_cost_comparison(self, food1: FoodEntity, food2: FoodEntity, country_code: Optional[str]) -> Dict[str, float]:
        """Estimate cost comparison"""
        return {'ratio': 1.0, 'confidence': 0.5}
    
    def _get_most_influential_foods(self, network_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Get most influential foods from network metrics"""
        return []
    
    async def _generate_cultural_recommendations(self, analysis: Dict[str, Any], country_code: str) -> List[str]:
        """Generate cultural food recommendations"""
        return []
    
    async def _assess_food_data_quality(self) -> Dict[str, Any]:
        """Assess food data quality"""
        return {'total_foods': 0, 'verified_foods': 0, 'avg_completeness': 0.0}
    
    async def _assess_relationship_quality(self) -> Dict[str, Any]:
        """Assess relationship quality"""
        return {}
    
    async def _generate_quality_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations"""
        return []
    
    async def _get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {}

# Factory function for easy instantiation
async def create_food_knowledge_engine(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    redis_url: str = "redis://localhost:6379"
) -> FoodKnowledgeGraphEngine:
    """Factory function to create and initialize the food knowledge engine"""
    
    engine = FoodKnowledgeGraphEngine(neo4j_uri, neo4j_user, neo4j_password, redis_url)
    
    success = await engine.initialize()
    if not success:
        raise RuntimeError("Failed to initialize Food Knowledge Graph Engine")
    
    return engine