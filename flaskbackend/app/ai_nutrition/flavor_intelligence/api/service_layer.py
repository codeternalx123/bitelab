"""
API Service Layer for Flavor Intelligence Pipeline
=================================================

This module implements comprehensive REST API endpoints for the Automated
Flavor Intelligence Pipeline. It provides FastAPI-based services for
flavor analysis, ingredient substitution, recipe optimization, and
knowledge graph queries.

Key Features:
- Flavor Profile Analysis API
- Ingredient Similarity & Substitution API  
- Recipe Analysis & Optimization API
- Knowledge Graph Query API
- Real-time Flavor Recommendation Engine
- Batch Processing API for Large Datasets
- Authentication & Rate Limiting
- Comprehensive Monitoring & Logging
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from typing import Dict, List, Optional, Union, Any, Tuple
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass, asdict
import asyncio
import logging
from datetime import datetime, timedelta
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import hashlib
import io
from contextlib import asynccontextmanager

# Database and caching
import redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import motor.motor_asyncio

# Monitoring and metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog

# Internal imports
from ..models.flavor_data_models import (
    FlavorProfile, SensoryProfile, NutritionData, ChemicalCompound,
    FlavorCategory, MolecularClass, DataSource
)
from ..layers.sensory_layer import SensoryProfileCalculator
from ..layers.molecular_layer import MolecularAnalyzer
from ..layers.relational_layer import RecipeDataProcessor, Recipe, IngredientCompatibility
from ..graphrag.graphrag_engine import Neo4jFlavorGraphDB, GraphRAGQueryProcessor
from ..scrapers.data_scrapers import DataScrapingOrchestrator
from ..models.neural_networks import FlavorIntelligenceInference, ModelConfig


# Pydantic models for API requests/responses

class SensoryProfileRequest(BaseModel):
    """Request model for sensory profile"""
    sweet: float = Field(..., ge=0, le=10, description="Sweetness intensity (0-10)")
    sour: float = Field(..., ge=0, le=10, description="Sourness intensity (0-10)")
    salty: float = Field(..., ge=0, le=10, description="Saltiness intensity (0-10)")
    bitter: float = Field(..., ge=0, le=10, description="Bitterness intensity (0-10)")
    umami: float = Field(..., ge=0, le=10, description="Umami intensity (0-10)")
    fatty: float = Field(..., ge=0, le=10, description="Fattiness intensity (0-10)")
    spicy: float = Field(..., ge=0, le=10, description="Spiciness intensity (0-10)")
    aromatic: float = Field(..., ge=0, le=10, description="Aromatic intensity (0-10)")

class NutritionDataRequest(BaseModel):
    """Request model for nutrition data"""
    calories: float = Field(..., ge=0, description="Calories per 100g")
    protein: float = Field(..., ge=0, description="Protein in grams per 100g")
    fat: float = Field(..., ge=0, description="Fat in grams per 100g")
    carbohydrates: float = Field(..., ge=0, description="Carbs in grams per 100g")
    fiber: Optional[float] = Field(None, ge=0, description="Fiber in grams per 100g")
    sugars: Optional[float] = Field(None, ge=0, description="Sugars in grams per 100g")
    sodium: Optional[float] = Field(None, ge=0, description="Sodium in mg per 100g")
    calcium: Optional[float] = Field(None, ge=0, description="Calcium in mg per 100g")
    iron: Optional[float] = Field(None, ge=0, description="Iron in mg per 100g")
    vitamin_c: Optional[float] = Field(None, ge=0, description="Vitamin C in mg per 100g")

class FlavorProfileRequest(BaseModel):
    """Request model for creating flavor profile"""
    name: str = Field(..., description="Ingredient name")
    sensory: Optional[SensoryProfileRequest] = None
    nutrition: Optional[NutritionDataRequest] = None
    category: Optional[str] = Field(None, description="Primary flavor category")
    origin_countries: Optional[List[str]] = Field(None, description="Countries of origin")
    description: Optional[str] = Field(None, description="Ingredient description")

class FlavorProfileResponse(BaseModel):
    """Response model for flavor profile"""
    ingredient_id: str
    name: str
    sensory: Dict[str, float]
    nutrition: Optional[Dict[str, float]]
    category: str
    confidence: float
    data_completeness: float
    origin_countries: List[str]
    created_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class IngredientSimilarityRequest(BaseModel):
    """Request for ingredient similarity analysis"""
    target_ingredient: str = Field(..., description="Target ingredient name or ID")
    comparison_ingredients: Optional[List[str]] = Field(None, description="Specific ingredients to compare")
    category_filter: Optional[str] = Field(None, description="Filter by category")
    max_results: int = Field(10, ge=1, le=100, description="Maximum results to return")
    similarity_threshold: float = Field(0.5, ge=0, le=1, description="Minimum similarity score")

class IngredientSimilarityResponse(BaseModel):
    """Response for ingredient similarity"""
    target_ingredient: FlavorProfileResponse
    similar_ingredients: List[Dict[str, Any]]
    similarity_scores: List[float]
    analysis_metadata: Dict[str, Any]

class RecipeAnalysisRequest(BaseModel):
    """Request for recipe analysis"""
    recipe_title: str = Field(..., description="Recipe title")
    ingredients: List[str] = Field(..., min_items=2, description="List of ingredients")
    cuisine: Optional[str] = Field(None, description="Cuisine type")
    dietary_restrictions: Optional[List[str]] = Field(None, description="Dietary restrictions")
    analyze_compatibility: bool = Field(True, description="Analyze ingredient compatibility")
    suggest_substitutions: bool = Field(False, description="Suggest ingredient substitutions")

class RecipeAnalysisResponse(BaseModel):
    """Response for recipe analysis"""
    recipe_id: str
    title: str
    analyzed_ingredients: List[FlavorProfileResponse]
    compatibility_matrix: Dict[str, Dict[str, float]]
    flavor_harmony_score: float
    suggested_substitutions: Optional[List[Dict[str, Any]]]
    nutritional_summary: Dict[str, float]
    improvement_suggestions: List[str]

class GraphQueryRequest(BaseModel):
    """Request for graph-based queries"""
    query: str = Field(..., description="Natural language query")
    query_type: str = Field("general", description="Query type: general, similarity, substitution, pairing")
    max_results: int = Field(20, ge=1, le=100, description="Maximum results")
    include_explanations: bool = Field(True, description="Include AI explanations")

class GraphQueryResponse(BaseModel):
    """Response for graph queries"""
    query: str
    query_type: str
    results: List[Dict[str, Any]]
    explanations: Optional[List[str]]
    confidence_scores: List[float]
    execution_time_ms: float

class BatchProcessingRequest(BaseModel):
    """Request for batch processing"""
    operation: str = Field(..., description="Operation: analyze_profiles, find_similarities, analyze_recipes")
    data: List[Dict[str, Any]] = Field(..., description="Input data for processing")
    options: Optional[Dict[str, Any]] = Field(None, description="Processing options")

class BatchProcessingResponse(BaseModel):
    """Response for batch processing"""
    job_id: str
    operation: str
    status: str
    total_items: int
    processed_items: int
    results: Optional[List[Dict[str, Any]]]
    errors: List[str]
    processing_time_seconds: float


# Application configuration and setup

@dataclass
class APIConfig:
    """Configuration for API service"""
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    
    # Database connections
    postgres_url: str = "postgresql+asyncpg://user:pass@localhost/flavordb"
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    redis_url: str = "redis://localhost:6379"
    
    # Model paths
    model_paths: Dict[str, str] = None
    
    # Authentication
    enable_auth: bool = False
    api_keys: List[str] = None
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst: int = 10
    
    # Caching
    cache_ttl_seconds: int = 3600
    enable_response_caching: bool = True
    
    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"


# Metrics and monitoring
api_requests_total = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
api_request_duration = Histogram('api_request_duration_seconds', 'API request duration')
api_active_connections = Gauge('api_active_connections', 'Active API connections')
flavor_analyses_total = Counter('flavor_analyses_total', 'Total flavor analyses performed')
similarity_calculations_total = Counter('similarity_calculations_total', 'Total similarity calculations')


class FlavorIntelligenceAPI:
    """Main API application class"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.logger = structlog.get_logger()
        
        # Initialize components
        self._initialize_connections()
        self._initialize_processors()
        self._initialize_ml_models()
        
        # Create FastAPI app
        self.app = self._create_app()
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        
        # Background tasks
        self.background_tasks = {}
    
    def _initialize_connections(self):
        """Initialize database connections"""
        
        # PostgreSQL for structured data
        self.postgres_engine = create_async_engine(self.config.postgres_url)
        self.async_session = sessionmaker(
            self.postgres_engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # Neo4j for graph data
        self.graph_db = Neo4jFlavorGraphDB(
            uri=self.config.neo4j_url,
            user=self.config.neo4j_user,
            password=self.config.neo4j_password
        )
        
        # Redis for caching
        self.redis_client = redis.from_url(self.config.redis_url, decode_responses=True)
        
        self.logger.info("Database connections initialized")
    
    def _initialize_processors(self):
        """Initialize processing components"""
        
        # Core processors
        self.sensory_calculator = SensoryProfileCalculator()
        self.molecular_analyzer = MolecularAnalyzer()
        self.recipe_processor = RecipeDataProcessor(None)
        self.graphrag_processor = GraphRAGQueryProcessor(self.graph_db)
        
        # Data orchestrator
        from ..scrapers.data_scrapers import ScrapingConfig
        scraping_config = ScrapingConfig()
        self.data_orchestrator = DataScrapingOrchestrator(scraping_config)
        
        self.logger.info("Processing components initialized")
    
    def _initialize_ml_models(self):
        """Initialize ML models for inference"""
        
        if self.config.model_paths:
            model_config = ModelConfig()
            self.ml_inference = FlavorIntelligenceInference(
                self.config.model_paths, 
                model_config
            )
        else:
            self.ml_inference = None
            self.logger.warning("No ML model paths provided - ML features disabled")
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            self.logger.info("Starting Flavor Intelligence API")
            api_active_connections.inc()
            yield
            # Shutdown
            await self._cleanup_connections()
            self.logger.info("Flavor Intelligence API shutdown complete")
        
        app = FastAPI(
            title="Flavor Intelligence API",
            description="Comprehensive flavor analysis and ingredient intelligence platform",
            version="1.0.0",
            lifespan=lifespan,
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        return app
    
    def _setup_middleware(self):
        """Setup middleware"""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Compression middleware
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request, call_next):
            start_time = time.time()
            
            # Process request
            response = await call_next(request)
            
            # Log metrics
            process_time = time.time() - start_time
            api_request_duration.observe(process_time)
            api_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            # Log request
            self.logger.info(
                "Request processed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                process_time=process_time
            )
            
            return response
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        # Metrics endpoint
        @self.app.get("/metrics")
        async def get_metrics():
            return generate_latest()
        
        # Flavor profile endpoints
        self._setup_flavor_profile_routes()
        
        # Similarity analysis endpoints
        self._setup_similarity_routes()
        
        # Recipe analysis endpoints
        self._setup_recipe_routes()
        
        # Graph query endpoints
        self._setup_graph_routes()
        
        # Batch processing endpoints
        self._setup_batch_routes()
        
        # Data management endpoints
        self._setup_data_routes()
    
    def _setup_flavor_profile_routes(self):
        """Setup flavor profile related routes"""
        
        @self.app.post("/api/v1/flavor-profiles", response_model=FlavorProfileResponse)
        async def create_flavor_profile(profile_request: FlavorProfileRequest):
            """Create or analyze a flavor profile"""
            
            try:
                # Convert request to internal format
                sensory_profile = None
                if profile_request.sensory:
                    sensory_profile = SensoryProfile(
                        sweet=profile_request.sensory.sweet,
                        sour=profile_request.sensory.sour,
                        salty=profile_request.sensory.salty,
                        bitter=profile_request.sensory.bitter,
                        umami=profile_request.sensory.umami,
                        fatty=profile_request.sensory.fatty,
                        spicy=profile_request.sensory.spicy,
                        aromatic=profile_request.sensory.aromatic
                    )
                
                nutrition_data = None
                if profile_request.nutrition:
                    nutrition_data = NutritionData(
                        calories=profile_request.nutrition.calories,
                        protein=profile_request.nutrition.protein,
                        fat=profile_request.nutrition.fat,
                        carbohydrates=profile_request.nutrition.carbohydrates,
                        fiber=profile_request.nutrition.fiber or 0,
                        sugars=profile_request.nutrition.sugars or 0,
                        sodium=profile_request.nutrition.sodium or 0,
                        calcium=profile_request.nutrition.calcium or 0,
                        iron=profile_request.nutrition.iron or 0,
                        vitamin_c=profile_request.nutrition.vitamin_c or 0
                    )
                
                # If no sensory profile provided, calculate from nutrition
                if not sensory_profile and nutrition_data:
                    sensory_profile = self.sensory_calculator.calculate_sensory_profile(
                        nutrition_data, profile_request.name
                    )
                
                # Determine category
                category = FlavorCategory.SAVORY  # Default
                if profile_request.category:
                    try:
                        category = FlavorCategory(profile_request.category.lower())
                    except ValueError:
                        pass
                
                # Create flavor profile
                flavor_profile = FlavorProfile(
                    ingredient_id=f"api_{hashlib.md5(profile_request.name.encode()).hexdigest()[:8]}",
                    name=profile_request.name,
                    sensory=sensory_profile,
                    nutrition=nutrition_data,
                    primary_category=category,
                    origin_countries=profile_request.origin_countries or [],
                    data_sources=[DataSource.USER_INPUT],
                    last_updated=datetime.now()
                )
                
                # Calculate quality metrics
                flavor_profile.calculate_overall_confidence()
                flavor_profile.calculate_data_completeness()
                
                # Store in database (if configured)
                # await self._store_flavor_profile(flavor_profile)
                
                # Convert to response
                response = FlavorProfileResponse(
                    ingredient_id=flavor_profile.ingredient_id,
                    name=flavor_profile.name,
                    sensory={
                        "sweet": flavor_profile.sensory.sweet if flavor_profile.sensory else 0,
                        "sour": flavor_profile.sensory.sour if flavor_profile.sensory else 0,
                        "salty": flavor_profile.sensory.salty if flavor_profile.sensory else 0,
                        "bitter": flavor_profile.sensory.bitter if flavor_profile.sensory else 0,
                        "umami": flavor_profile.sensory.umami if flavor_profile.sensory else 0,
                        "fatty": flavor_profile.sensory.fatty if flavor_profile.sensory else 0,
                        "spicy": flavor_profile.sensory.spicy if flavor_profile.sensory else 0,
                        "aromatic": flavor_profile.sensory.aromatic if flavor_profile.sensory else 0
                    },
                    nutrition={
                        "calories": nutrition_data.calories if nutrition_data else 0,
                        "protein": nutrition_data.protein if nutrition_data else 0,
                        "fat": nutrition_data.fat if nutrition_data else 0,
                        "carbohydrates": nutrition_data.carbohydrates if nutrition_data else 0
                    } if nutrition_data else None,
                    category=flavor_profile.primary_category.value,
                    confidence=flavor_profile.overall_confidence,
                    data_completeness=flavor_profile.data_completeness,
                    origin_countries=flavor_profile.origin_countries,
                    created_at=flavor_profile.last_updated
                )
                
                # Increment counter
                flavor_analyses_total.inc()
                
                return response
            
            except Exception as e:
                self.logger.error("Failed to create flavor profile", error=str(e))
                raise HTTPException(status_code=500, detail=f"Failed to create flavor profile: {str(e)}")
        
        @self.app.get("/api/v1/flavor-profiles/{ingredient_id}", response_model=FlavorProfileResponse)
        async def get_flavor_profile(ingredient_id: str):
            """Retrieve a specific flavor profile"""
            
            # Check cache first
            cached_profile = await self._get_cached_profile(ingredient_id)
            if cached_profile:
                return cached_profile
            
            # Search in graph database
            try:
                profile_data = await self.graph_db.get_ingredient_profile(ingredient_id)
                if not profile_data:
                    raise HTTPException(status_code=404, detail="Flavor profile not found")
                
                # Convert to response format
                response = self._convert_profile_to_response(profile_data)
                
                # Cache the result
                await self._cache_profile(ingredient_id, response)
                
                return response
            
            except Exception as e:
                self.logger.error("Failed to retrieve flavor profile", ingredient_id=ingredient_id, error=str(e))
                raise HTTPException(status_code=500, detail=f"Failed to retrieve flavor profile: {str(e)}")
        
        @self.app.get("/api/v1/flavor-profiles", response_model=List[FlavorProfileResponse])
        async def search_flavor_profiles(
            query: str = None,
            category: str = None,
            limit: int = 20,
            offset: int = 0
        ):
            """Search flavor profiles by various criteria"""
            
            try:
                # Build search parameters
                search_params = {
                    "limit": limit,
                    "offset": offset
                }
                
                if query:
                    search_params["name_contains"] = query
                
                if category:
                    search_params["category"] = category
                
                # Execute search
                profiles = await self.graph_db.search_ingredients(**search_params)
                
                # Convert to response format
                responses = []
                for profile_data in profiles:
                    response = self._convert_profile_to_response(profile_data)
                    responses.append(response)
                
                return responses
            
            except Exception as e:
                self.logger.error("Failed to search flavor profiles", error=str(e))
                raise HTTPException(status_code=500, detail=f"Failed to search flavor profiles: {str(e)}")
    
    def _setup_similarity_routes(self):
        """Setup similarity analysis routes"""
        
        @self.app.post("/api/v1/similarity/analyze", response_model=IngredientSimilarityResponse)
        async def analyze_similarity(similarity_request: IngredientSimilarityRequest):
            """Analyze ingredient similarity"""
            
            try:
                # Get target ingredient profile
                target_profile = await self._get_ingredient_profile(similarity_request.target_ingredient)
                if not target_profile:
                    raise HTTPException(status_code=404, detail="Target ingredient not found")
                
                # Get comparison ingredients
                if similarity_request.comparison_ingredients:
                    comparison_profiles = []
                    for ingredient in similarity_request.comparison_ingredients:
                        profile = await self._get_ingredient_profile(ingredient)
                        if profile:
                            comparison_profiles.append(profile)
                else:
                    # Search similar ingredients from database
                    search_params = {
                        "category": similarity_request.category_filter,
                        "limit": similarity_request.max_results * 2  # Get more for filtering
                    }
                    comparison_profiles = await self.graph_db.find_similar_ingredients(
                        target_profile.ingredient_id, **search_params
                    )
                
                # Calculate similarities using ML model if available
                similarities = []
                if self.ml_inference:
                    for comp_profile in comparison_profiles:
                        similarity_score = self.ml_inference.calculate_similarity(
                            target_profile, comp_profile
                        )
                        if similarity_score >= similarity_request.similarity_threshold:
                            similarities.append({
                                'ingredient': comp_profile,
                                'similarity_score': similarity_score
                            })
                else:
                    # Fallback to rule-based similarity
                    similarities = await self._calculate_rule_based_similarity(
                        target_profile, comparison_profiles, similarity_request.similarity_threshold
                    )
                
                # Sort by similarity score
                similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
                similarities = similarities[:similarity_request.max_results]
                
                # Prepare response
                similar_ingredients = []
                similarity_scores = []
                
                for sim_data in similarities:
                    ingredient_response = self._convert_profile_to_response(sim_data['ingredient'])
                    similar_ingredients.append({
                        'ingredient': ingredient_response.dict(),
                        'explanation': f"Similar due to {self._get_similarity_explanation(target_profile, sim_data['ingredient'])}"
                    })
                    similarity_scores.append(sim_data['similarity_score'])
                
                # Create response
                response = IngredientSimilarityResponse(
                    target_ingredient=self._convert_profile_to_response(target_profile),
                    similar_ingredients=similar_ingredients,
                    similarity_scores=similarity_scores,
                    analysis_metadata={
                        'method': 'ml_model' if self.ml_inference else 'rule_based',
                        'total_comparisons': len(comparison_profiles),
                        'threshold_used': similarity_request.similarity_threshold,
                        'analysis_timestamp': datetime.now().isoformat()
                    }
                )
                
                # Increment counter
                similarity_calculations_total.inc()
                
                return response
            
            except Exception as e:
                self.logger.error("Failed to analyze similarity", error=str(e))
                raise HTTPException(status_code=500, detail=f"Failed to analyze similarity: {str(e)}")
        
        @self.app.get("/api/v1/similarity/substitutes/{ingredient_id}")
        async def find_substitutes(
            ingredient_id: str,
            max_results: int = 5,
            dietary_restrictions: Optional[List[str]] = None,
            cuisine_preference: Optional[str] = None
        ):
            """Find ingredient substitutes"""
            
            try:
                # Get target ingredient
                target_profile = await self._get_ingredient_profile(ingredient_id)
                if not target_profile:
                    raise HTTPException(status_code=404, detail="Ingredient not found")
                
                # Find substitutes using ML model
                if self.ml_inference:
                    # Get candidate ingredients (same category or compatible categories)
                    candidates = await self.graph_db.get_substitute_candidates(
                        ingredient_id, dietary_restrictions, cuisine_preference
                    )
                    
                    # Use ML model to find best substitutes
                    substitutes = self.ml_inference.find_substitutes(
                        target_profile, candidates, max_results
                    )
                else:
                    # Use graph-based substitute finding
                    substitutes = await self.graph_db.find_ingredient_substitutes(
                        ingredient_id, max_results, dietary_restrictions
                    )
                
                # Format response
                substitute_list = []
                for substitute, confidence in substitutes:
                    substitute_response = self._convert_profile_to_response(substitute)
                    substitute_list.append({
                        'substitute': substitute_response.dict(),
                        'confidence': confidence,
                        'substitution_ratio': 1.0,  # Default 1:1 ratio
                        'notes': f"Good substitute for {target_profile.name}"
                    })
                
                return {
                    'target_ingredient': ingredient_id,
                    'substitutes': substitute_list,
                    'analysis_method': 'ml_model' if self.ml_inference else 'graph_based',
                    'timestamp': datetime.now().isoformat()
                }
            
            except Exception as e:
                self.logger.error("Failed to find substitutes", error=str(e))
                raise HTTPException(status_code=500, detail=f"Failed to find substitutes: {str(e)}")
    
    def _setup_recipe_routes(self):
        """Setup recipe analysis routes"""
        
        @self.app.post("/api/v1/recipes/analyze", response_model=RecipeAnalysisResponse)
        async def analyze_recipe(recipe_request: RecipeAnalysisRequest):
            """Analyze recipe for flavor harmony and compatibility"""
            
            try:
                # Generate recipe ID
                recipe_id = hashlib.md5(
                    f"{recipe_request.recipe_title}_{len(recipe_request.ingredients)}".encode()
                ).hexdigest()[:12]
                
                # Analyze each ingredient
                analyzed_ingredients = []
                ingredient_profiles = []
                
                for ingredient_name in recipe_request.ingredients:
                    # Try to find existing profile
                    profile = await self._find_ingredient_by_name(ingredient_name)
                    
                    if not profile:
                        # Create basic profile for unknown ingredient
                        profile = FlavorProfile(
                            ingredient_id=f"temp_{hashlib.md5(ingredient_name.encode()).hexdigest()[:8]}",
                            name=ingredient_name,
                            primary_category=FlavorCategory.SAVORY,  # Default
                            data_sources=[DataSource.USER_INPUT],
                            last_updated=datetime.now()
                        )
                        
                        # Try to infer sensory profile from name
                        inferred_sensory = self.sensory_calculator.infer_from_name(ingredient_name)
                        if inferred_sensory:
                            profile.sensory = inferred_sensory
                    
                    ingredient_profiles.append(profile)
                    analyzed_ingredients.append(self._convert_profile_to_response(profile))
                
                # Calculate compatibility matrix
                compatibility_matrix = {}
                if recipe_request.analyze_compatibility:
                    compatibility_matrix = await self._calculate_compatibility_matrix(ingredient_profiles)
                
                # Calculate overall flavor harmony score
                harmony_score = self._calculate_flavor_harmony(ingredient_profiles, compatibility_matrix)
                
                # Generate substitution suggestions
                suggested_substitutions = []
                if recipe_request.suggest_substitutions:
                    suggested_substitutions = await self._generate_substitution_suggestions(
                        ingredient_profiles, recipe_request.dietary_restrictions
                    )
                
                # Calculate nutritional summary
                nutritional_summary = self._calculate_nutritional_summary(ingredient_profiles)
                
                # Generate improvement suggestions
                improvement_suggestions = self._generate_improvement_suggestions(
                    ingredient_profiles, harmony_score, recipe_request.cuisine
                )
                
                # Create response
                response = RecipeAnalysisResponse(
                    recipe_id=recipe_id,
                    title=recipe_request.recipe_title,
                    analyzed_ingredients=analyzed_ingredients,
                    compatibility_matrix=compatibility_matrix,
                    flavor_harmony_score=harmony_score,
                    suggested_substitutions=suggested_substitutions if recipe_request.suggest_substitutions else None,
                    nutritional_summary=nutritional_summary,
                    improvement_suggestions=improvement_suggestions
                )
                
                return response
            
            except Exception as e:
                self.logger.error("Failed to analyze recipe", error=str(e))
                raise HTTPException(status_code=500, detail=f"Failed to analyze recipe: {str(e)}")
        
        @self.app.post("/api/v1/recipes/optimize")
        async def optimize_recipe(
            recipe_request: RecipeAnalysisRequest,
            optimization_goals: List[str] = ["flavor_harmony", "nutrition_balance"]
        ):
            """Optimize recipe for specific goals"""
            
            try:
                # First analyze the current recipe
                current_analysis = await analyze_recipe(recipe_request)
                
                # Generate optimization suggestions based on goals
                optimizations = {}
                
                if "flavor_harmony" in optimization_goals:
                    flavor_optimizations = await self._optimize_for_flavor_harmony(
                        recipe_request.ingredients, current_analysis.flavor_harmony_score
                    )
                    optimizations["flavor_improvements"] = flavor_optimizations
                
                if "nutrition_balance" in optimization_goals:
                    nutrition_optimizations = await self._optimize_for_nutrition(
                        recipe_request.ingredients, current_analysis.nutritional_summary
                    )
                    optimizations["nutrition_improvements"] = nutrition_optimizations
                
                if "cost_reduction" in optimization_goals:
                    cost_optimizations = await self._optimize_for_cost(recipe_request.ingredients)
                    optimizations["cost_improvements"] = cost_optimizations
                
                return {
                    'recipe_id': current_analysis.recipe_id,
                    'current_analysis': current_analysis.dict(),
                    'optimizations': optimizations,
                    'optimization_timestamp': datetime.now().isoformat()
                }
            
            except Exception as e:
                self.logger.error("Failed to optimize recipe", error=str(e))
                raise HTTPException(status_code=500, detail=f"Failed to optimize recipe: {str(e)}")
    
    def _setup_graph_routes(self):
        """Setup graph query routes"""
        
        @self.app.post("/api/v1/graph/query", response_model=GraphQueryResponse)
        async def query_knowledge_graph(query_request: GraphQueryRequest):
            """Query the flavor knowledge graph using natural language"""
            
            try:
                start_time = time.time()
                
                # Process query through GraphRAG
                results = await self.graphrag_processor.process_query(
                    query_request.query,
                    query_type=query_request.query_type,
                    max_results=query_request.max_results,
                    include_explanations=query_request.include_explanations
                )
                
                execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                # Format response
                response = GraphQueryResponse(
                    query=query_request.query,
                    query_type=query_request.query_type,
                    results=results.get('results', []),
                    explanations=results.get('explanations') if query_request.include_explanations else None,
                    confidence_scores=results.get('confidence_scores', []),
                    execution_time_ms=execution_time
                )
                
                return response
            
            except Exception as e:
                self.logger.error("Failed to process graph query", error=str(e))
                raise HTTPException(status_code=500, detail=f"Failed to process graph query: {str(e)}")
        
        @self.app.get("/api/v1/graph/stats")
        async def get_graph_statistics():
            """Get knowledge graph statistics"""
            
            try:
                stats = await self.graph_db.get_database_statistics()
                return {
                    'graph_statistics': stats,
                    'last_updated': datetime.now().isoformat()
                }
            
            except Exception as e:
                self.logger.error("Failed to get graph statistics", error=str(e))
                raise HTTPException(status_code=500, detail=f"Failed to get graph statistics: {str(e)}")
    
    def _setup_batch_routes(self):
        """Setup batch processing routes"""
        
        @self.app.post("/api/v1/batch/process", response_model=BatchProcessingResponse)
        async def start_batch_processing(
            batch_request: BatchProcessingRequest,
            background_tasks: BackgroundTasks
        ):
            """Start batch processing job"""
            
            try:
                # Generate job ID
                job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(batch_request.data)}"
                
                # Create job record
                job_record = {
                    'job_id': job_id,
                    'operation': batch_request.operation,
                    'status': 'queued',
                    'total_items': len(batch_request.data),
                    'processed_items': 0,
                    'created_at': datetime.now().isoformat(),
                    'data': batch_request.data,
                    'options': batch_request.options or {}
                }
                
                # Store job in background tasks
                self.background_tasks[job_id] = job_record
                
                # Start background processing
                background_tasks.add_task(
                    self._process_batch_job,
                    job_id,
                    batch_request.operation,
                    batch_request.data,
                    batch_request.options or {}
                )
                
                return BatchProcessingResponse(
                    job_id=job_id,
                    operation=batch_request.operation,
                    status='queued',
                    total_items=len(batch_request.data),
                    processed_items=0,
                    results=None,
                    errors=[],
                    processing_time_seconds=0
                )
            
            except Exception as e:
                self.logger.error("Failed to start batch processing", error=str(e))
                raise HTTPException(status_code=500, detail=f"Failed to start batch processing: {str(e)}")
        
        @self.app.get("/api/v1/batch/jobs/{job_id}", response_model=BatchProcessingResponse)
        async def get_batch_job_status(job_id: str):
            """Get batch job status"""
            
            if job_id not in self.background_tasks:
                raise HTTPException(status_code=404, detail="Batch job not found")
            
            job_record = self.background_tasks[job_id]
            
            return BatchProcessingResponse(
                job_id=job_record['job_id'],
                operation=job_record['operation'],
                status=job_record['status'],
                total_items=job_record['total_items'],
                processed_items=job_record['processed_items'],
                results=job_record.get('results'),
                errors=job_record.get('errors', []),
                processing_time_seconds=job_record.get('processing_time_seconds', 0)
            )
    
    def _setup_data_routes(self):
        """Setup data management routes"""
        
        @self.app.post("/api/v1/data/scrape")
        async def trigger_data_scraping(
            sources: List[str] = ["openfoodfacts", "usda_fdc"],
            max_records: int = 1000
        ):
            """Trigger data scraping from external sources"""
            
            try:
                # Start scraping job
                scraping_results = await self.data_orchestrator.execute_comprehensive_scraping()
                
                return {
                    'scraping_job_started': True,
                    'sources': sources,
                    'max_records_per_source': max_records,
                    'job_results': scraping_results,
                    'timestamp': datetime.now().isoformat()
                }
            
            except Exception as e:
                self.logger.error("Failed to trigger data scraping", error=str(e))
                raise HTTPException(status_code=500, detail=f"Failed to trigger data scraping: {str(e)}")
        
        @self.app.get("/api/v1/data/stats")
        async def get_data_statistics():
            """Get comprehensive data statistics"""
            
            try:
                scraping_stats = self.data_orchestrator.get_scraping_statistics()
                
                return {
                    'scraping_statistics': scraping_stats,
                    'timestamp': datetime.now().isoformat()
                }
            
            except Exception as e:
                self.logger.error("Failed to get data statistics", error=str(e))
                raise HTTPException(status_code=500, detail=f"Failed to get data statistics: {str(e)}")
    
    # Helper methods
    
    async def _get_cached_profile(self, ingredient_id: str) -> Optional[FlavorProfileResponse]:
        """Get cached flavor profile"""
        try:
            cached_data = self.redis_client.get(f"profile:{ingredient_id}")
            if cached_data:
                profile_dict = json.loads(cached_data)
                return FlavorProfileResponse(**profile_dict)
        except Exception as e:
            self.logger.warning("Cache retrieval failed", error=str(e))
        return None
    
    async def _cache_profile(self, ingredient_id: str, profile: FlavorProfileResponse):
        """Cache flavor profile"""
        try:
            profile_dict = profile.dict()
            self.redis_client.setex(
                f"profile:{ingredient_id}",
                self.config.cache_ttl_seconds,
                json.dumps(profile_dict, default=str)
            )
        except Exception as e:
            self.logger.warning("Cache storage failed", error=str(e))
    
    async def _get_ingredient_profile(self, identifier: str) -> Optional[FlavorProfile]:
        """Get ingredient profile by ID or name"""
        # Implementation would search graph database or local storage
        # This is a mock implementation
        return None
    
    def _convert_profile_to_response(self, profile_data: Union[FlavorProfile, Dict]) -> FlavorProfileResponse:
        """Convert internal profile to API response format"""
        
        if isinstance(profile_data, FlavorProfile):
            return FlavorProfileResponse(
                ingredient_id=profile_data.ingredient_id,
                name=profile_data.name,
                sensory=asdict(profile_data.sensory) if profile_data.sensory else {},
                nutrition=asdict(profile_data.nutrition) if profile_data.nutrition else None,
                category=profile_data.primary_category.value,
                confidence=profile_data.overall_confidence,
                data_completeness=profile_data.data_completeness,
                origin_countries=profile_data.origin_countries,
                created_at=profile_data.last_updated
            )
        else:
            # Handle dictionary format
            return FlavorProfileResponse(**profile_data)
    
    async def _calculate_rule_based_similarity(self, target: FlavorProfile, 
                                             candidates: List[FlavorProfile],
                                             threshold: float) -> List[Dict]:
        """Calculate similarity using rule-based approach"""
        
        similarities = []
        
        for candidate in candidates:
            similarity_score = 0.0
            
            # Category similarity
            if target.primary_category == candidate.primary_category:
                similarity_score += 0.3
            
            # Sensory similarity
            if target.sensory and candidate.sensory:
                sensory_similarity = self._calculate_sensory_similarity(
                    target.sensory, candidate.sensory
                )
                similarity_score += 0.5 * sensory_similarity
            
            # Nutrition similarity
            if target.nutrition and candidate.nutrition:
                nutrition_similarity = self._calculate_nutrition_similarity(
                    target.nutrition, candidate.nutrition
                )
                similarity_score += 0.2 * nutrition_similarity
            
            if similarity_score >= threshold:
                similarities.append({
                    'ingredient': candidate,
                    'similarity_score': similarity_score
                })
        
        return similarities
    
    def _calculate_sensory_similarity(self, sensory1: SensoryProfile, 
                                    sensory2: SensoryProfile) -> float:
        """Calculate sensory profile similarity"""
        
        attributes = ['sweet', 'sour', 'salty', 'bitter', 'umami', 'fatty', 'spicy', 'aromatic']
        
        total_diff = 0.0
        for attr in attributes:
            val1 = getattr(sensory1, attr, 0)
            val2 = getattr(sensory2, attr, 0)
            total_diff += abs(val1 - val2)
        
        # Normalize to 0-1 scale (max diff would be 8*10=80)
        similarity = 1.0 - (total_diff / 80.0)
        return max(0.0, similarity)
    
    def _calculate_nutrition_similarity(self, nutrition1: NutritionData,
                                      nutrition2: NutritionData) -> float:
        """Calculate nutrition profile similarity"""
        
        # Normalize values and calculate similarity
        attrs = ['calories', 'protein', 'fat', 'carbohydrates']
        
        total_similarity = 0.0
        valid_attrs = 0
        
        for attr in attrs:
            val1 = getattr(nutrition1, attr, 0)
            val2 = getattr(nutrition2, attr, 0)
            
            if val1 > 0 and val2 > 0:
                # Calculate relative difference
                rel_diff = abs(val1 - val2) / max(val1, val2)
                similarity = 1.0 - rel_diff
                total_similarity += similarity
                valid_attrs += 1
        
        if valid_attrs == 0:
            return 0.0
        
        return total_similarity / valid_attrs
    
    async def _process_batch_job(self, job_id: str, operation: str, 
                               data: List[Dict], options: Dict):
        """Process batch job in background"""
        
        try:
            job_record = self.background_tasks[job_id]
            job_record['status'] = 'processing'
            job_record['started_at'] = datetime.now().isoformat()
            
            start_time = time.time()
            results = []
            errors = []
            
            for i, item in enumerate(data):
                try:
                    if operation == "analyze_profiles":
                        result = await self._process_flavor_profile_item(item)
                    elif operation == "find_similarities":
                        result = await self._process_similarity_item(item)
                    elif operation == "analyze_recipes":
                        result = await self._process_recipe_item(item)
                    else:
                        raise ValueError(f"Unknown operation: {operation}")
                    
                    results.append(result)
                    job_record['processed_items'] = i + 1
                
                except Exception as e:
                    errors.append(f"Item {i}: {str(e)}")
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
            
            # Update job record
            job_record['status'] = 'completed'
            job_record['results'] = results
            job_record['errors'] = errors
            job_record['processing_time_seconds'] = time.time() - start_time
            job_record['completed_at'] = datetime.now().isoformat()
        
        except Exception as e:
            job_record['status'] = 'failed'
            job_record['errors'] = [str(e)]
            self.logger.error("Batch job failed", job_id=job_id, error=str(e))
    
    async def _cleanup_connections(self):
        """Cleanup database connections"""
        try:
            await self.postgres_engine.dispose()
            await self.graph_db.close()
            self.redis_client.close()
        except Exception as e:
            self.logger.error("Error during cleanup", error=str(e))


# Factory function

def create_flavor_api(config: APIConfig) -> FlavorIntelligenceAPI:
    """Create Flavor Intelligence API instance"""
    return FlavorIntelligenceAPI(config)


# Main entry point for running the API

def run_api(config_path: Optional[str] = None, **kwargs):
    """Run the Flavor Intelligence API server"""
    
    # Load configuration
    if config_path:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = APIConfig(**config_dict)
    else:
        config = APIConfig(**kwargs)
    
    # Create API instance
    api_instance = create_flavor_api(config)
    
    # Run server
    uvicorn.run(
        api_instance.app,
        host=config.host,
        port=config.port,
        reload=config.reload,
        workers=config.workers
    )


if __name__ == "__main__":
    # Example usage
    config = APIConfig(
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="DEBUG"
    )
    
    run_api(
        host=config.host,
        port=config.port,
        reload=config.reload,
        log_level=config.log_level
    )