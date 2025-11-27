from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from app.core.config import settings
from app.core.logging import setup_logging
from app.routes import (
    auth, plan, reports, health, analytics, 
    food_scanning, quantum_payments, mpesa_payments, supabase_auth,
    colorimetry, fusion, food_intelligence, risk_integration,
    chemometrics, recipes, chat, ai_routes, family_recipes, food_risk_analysis
)
from app.deps import rate_limit
from app.middleware.logging import RequestMiddleware
from app.middleware.security import SecurityHeadersMiddleware, SQLInjectionMiddleware

logger = setup_logging()

app = FastAPI(
    title=settings.APP_NAME,
    description="""
    # Wellomex AI Nutrition System API
    
    Comprehensive health optimization platform with advanced chemometric analysis and personalized risk assessment.
    
    ## Core Capabilities
    
    ### 🔬 Chemometric Analysis
    - Visual chemometrics for element detection
    - ICP-MS data fusion
    - Computational colorimetry
    - Multi-modal sensor fusion
    
    ### 🍽️ AI Recipe Generation
    - Multi-LLM recipe generation (GPT-4, Claude, Gemini)
    - Cultural recipe knowledge graphs
    - Dietary restriction adaptation
    - Ingredient substitution engine
    - Multi-objective nutritional optimization
    
    ### ⚕️ Risk Integration (NEW)
    - Personalized food safety assessment
    - Medical condition-based risk stratification
    - FDA/WHO/NKF regulatory compliance
    - AI-powered alternative food recommendations
    
    ### 🤖 Conversational AI (NEW)
    - ChatGPT-like nutrition assistant
    - Multi-turn conversations with context awareness
    - Food scanning + instant health assessment
    - Recipe generation from pantry ingredients
    - Automated meal planning and grocery lists
    - Portion estimation for metabolic goals
    - 55+ health goals and all disease management
    - Real-time medication interaction checking
    
    ### 📊 Analytics & Intelligence
    - Advanced food intelligence
    - Nutritional analysis
    - Health reports and tracking
    
    ### 🔐 Security & Payments
    - Quantum-secure payments
    - M-Pesa integration
    - Supabase authentication
    
    ## Risk Integration Layer
    
    The **Dynamic Risk Integration Layer** connects high-resolution atomic element detection 
    to personalized health risk assessment:
    
    1. **Risk Assessment** - 5-step decision process linking chemometric predictions to health outcomes
    2. **Personalized Warnings** - Multi-tier warnings (consumer/clinical/regulatory modes)
    3. **Alternative Finder** - AI-powered search for safer food alternatives
    4. **Health Profiles** - Medical condition tracking with SNOMED CT coding
    5. **Threshold Database** - 500+ regulatory limits from FDA, WHO, NKF, KDIGO
    
    ## Authentication
    
    Most endpoints require authentication via Bearer token:
    ```
    Authorization: Bearer <your_token>
    ```
    
    ## Rate Limiting
    
    API requests are rate-limited to ensure fair usage and system stability.
    
    ## Support
    
    For questions or issues, contact: support@wellomex.com
    """,
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    dependencies=[Depends(rate_limit)],
    contact={
        "name": "Wellomex Support",
        "email": "support@wellomex.com",
        "url": "https://wellomex.com"
    },
    license_info={
        "name": "Proprietary",
        "url": "https://wellomex.com/license"
    }
)

# Security middlewares
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(SQLInjectionMiddleware)
app.add_middleware(RequestMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# API Routes
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(supabase_auth.router, tags=["supabase-auth"])
app.include_router(plan.router, prefix="/api/v1/plan", tags=["plan"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["reports"])
app.include_router(health.router, prefix="/api/v1/health", tags=["health"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(food_scanning.router, prefix="/api/v1/food", tags=["food-scanning"])
app.include_router(quantum_payments.router, prefix="/api/v1/payments", tags=["quantum-secure-payments"])
app.include_router(mpesa_payments.router, tags=["mpesa-payments"])
app.include_router(colorimetry.router, prefix="/api/v1/colorimetry", tags=["computational-colorimetry"])
app.include_router(fusion.router, prefix="/api/v1/fusion", tags=["fused-analysis"])
app.include_router(food_intelligence.router, prefix="/api/v1/food-intelligence", tags=["unified-food-intelligence"])
app.include_router(risk_integration.router, prefix="/api/v1/risk-integration", tags=["risk-integration"])
app.include_router(chemometrics.router, prefix="/api/v1/chemometrics", tags=["chemometrics"])
app.include_router(recipes.router, prefix="/api/v1/recipes", tags=["recipes"])
app.include_router(chat.router, prefix="/api/v1", tags=["conversational-ai"])
app.include_router(ai_routes.router, prefix="/api/v1", tags=["deep-learning"])
app.include_router(family_recipes.router, prefix="/api/v1", tags=["family-recipes"])
app.include_router(food_risk_analysis.router, prefix="/api/v1/food-risk", tags=["food-risk-analysis"])

@app.get('/health')
def health():
    return {'status': 'ok'}
