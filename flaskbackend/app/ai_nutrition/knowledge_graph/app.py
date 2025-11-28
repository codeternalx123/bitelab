"""
FastAPI Integration Setup for Food Knowledge Graph
================================================

Main FastAPI application configuration and setup for the
comprehensive food knowledge graph system.

This module provides:
- FastAPI application factory
- Middleware configuration
- Route registration
- Database and service initialization
- Error handling and logging setup

Author: Wellomex AI Nutrition Team
Version: 1.0.0
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from .routes.api_routes import router as food_knowledge_router
from .services.food_knowledge_service import FoodKnowledgeServiceFactory

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting Food Knowledge Graph API...")
    
    try:
        # Initialize the food knowledge service
        service = await FoodKnowledgeServiceFactory.get_service()
        logger.info("Food Knowledge Graph service initialized successfully")
        
        # Store service in app state for access
        app.state.food_knowledge_service = service
        
        # Perform health check
        analytics = await service.get_system_analytics()
        logger.info(f"System analytics: {analytics.get('service_metrics', {})}")
        
    except Exception as e:
        logger.error(f"Failed to initialize Food Knowledge Graph service: {e}")
        # Don't raise - allow app to start but service will be unavailable
    
    yield
    
    # Shutdown
    logger.info("Shutting down Food Knowledge Graph API...")
    
    try:
        # Cleanup service resources
        if hasattr(app.state, 'food_knowledge_service'):
            service = app.state.food_knowledge_service
            await service.cleanup()
            logger.info("Food Knowledge Graph service cleanup completed")
    except Exception as e:
        logger.error(f"Error during service cleanup: {e}")

def create_food_knowledge_app() -> FastAPI:
    """
    Create and configure the Food Knowledge Graph FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    
    # Create FastAPI app with lifespan management
    app = FastAPI(
        title="Food Knowledge Graph API",
        description="""
        ## Comprehensive Food Knowledge Graph API
        
        This API provides access to a massive food database with AI-powered knowledge graph
        capabilities, supporting millions of foods from around the world.
        
        ### Key Features:
        - **Advanced Food Search**: ML-powered search with natural language processing
        - **Nutritional Analysis**: Comprehensive nutritional profiling and comparison
        - **Food Substitution**: Intelligent food substitute recommendations
        - **Cultural Analysis**: Country-specific food patterns and preferences  
        - **Seasonal Predictions**: ML-based seasonal availability forecasting
        - **Multi-API Integration**: Real-time data from USDA, OpenFoodFacts, and more
        - **Graph Relationships**: Complex food relationships and network analysis
        
        ### Technology Stack:
        - **Database**: Neo4j Graph Database with Redis caching
        - **Machine Learning**: PyTorch, scikit-learn, transformers
        - **APIs**: FastAPI with automatic validation and documentation
        - **Data Sources**: USDA FoodData Central, Open Food Facts, Nutritionix, Spoonacular
        
        ### Usage:
        1. Start with `/search` endpoints to find foods
        2. Use `/foods/{food_id}` to get detailed information
        3. Explore `/substitutes` and `/similar` for recommendations
        4. Analyze cultural patterns with `/cultural-analysis`
        5. Check seasonal availability with `/seasonal-prediction`
        
        All endpoints support both simple GET requests and advanced POST requests
        with detailed parameters for maximum flexibility.
        """,
        version="1.0.0",
        contact={
            "name": "Wellomex AI Nutrition Team",
            "email": "nutrition-ai@wellomex.com",
            "url": "https://wellomex.com/nutrition-ai"
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        },
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = datetime.utcnow()
        
        # Process request
        try:
            response = await call_next(request)
            process_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )
            
            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(
                f"{request.method} {request.url.path} - "
                f"Error: {str(e)} - "
                f"Time: {process_time:.3f}s"
            )
            raise
    
    # Register routes
    app.include_router(food_knowledge_router)
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "Food Knowledge Graph API",
            "version": "1.0.0",
            "description": "Comprehensive food database with AI-powered knowledge graph",
            "documentation": "/docs",
            "alternative_docs": "/redoc",
            "health_check": "/api/v1/food-knowledge/health",
            "search_endpoint": "/api/v1/food-knowledge/search",
            "features": [
                "Advanced food search and discovery",
                "Nutritional analysis and profiling", 
                "Intelligent food substitution",
                "Cultural food pattern analysis",
                "Seasonal availability prediction",
                "Multi-API data integration",
                "Graph relationship analysis"
            ],
            "supported_countries": "200+ countries with specific food data",
            "supported_foods": "Millions of foods with comprehensive data",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            contact=app.contact,
            license_info=app.license_info
        )
        
        # Add custom schema information
        openapi_schema["info"]["x-logo"] = {
            "url": "https://wellomex.com/assets/logo.png"
        }
        
        # Add server information
        openapi_schema["servers"] = [
            {"url": "http://localhost:8000", "description": "Development server"},
            {"url": "https://api.wellomex.com", "description": "Production server"},
        ]
        
        # Add tags
        openapi_schema["tags"] = [
            {
                "name": "Food Knowledge Graph",
                "description": "Comprehensive food data and AI-powered analysis"
            },
            {
                "name": "Root",
                "description": "API information and health checks"
            }
        ]
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    # Global exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "message": exc.detail,
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": "Internal server error",
                "status_code": 500,
                "path": request.url.path,
                "method": request.method,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    return app

# Create the app instance
food_knowledge_app = create_food_knowledge_app()

# Optional: Create a function to run the app
def run_food_knowledge_api(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """
    Run the Food Knowledge Graph API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to  
        debug: Enable debug mode
    """
    import uvicorn
    
    uvicorn.run(
        "app.ai_nutrition.knowledge_graph.app:food_knowledge_app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug",
        access_log=True
    )

if __name__ == "__main__":
    # Run the API server when executed directly
    run_food_knowledge_api(debug=True)