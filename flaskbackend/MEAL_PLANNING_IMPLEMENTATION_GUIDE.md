# Wellomex AI Meal Planning System - Implementation Guide

## 🎯 Overview

This guide provides step-by-step instructions to implement a comprehensive, enterprise-ready meal planning system that supports:
- **195+ countries** with local cuisines and ingredients
- **Disease-specific meal plans** for 100+ medical conditions
- **Flavor profiling** and cultural authenticity
- **Nutritional optimization** with ML-powered personalization
- **Scalable architecture** designed for millions of users

## 📊 System Architecture

```
meal_planning/
├── core/                          # Core business logic
│   ├── meal_planner_engine.py    # Main orchestrator (3,000 LOC)
│   ├── nutrition_optimizer.py    # ML-based optimization (2,500 LOC)
│   └── recipe_adapter.py         # Recipe transformation (2,000 LOC)
│
├── data/                          # Data layer
│   ├── country_cuisine_db.py     # Country/cuisine data (4,000 LOC)
│   ├── local_foods_db.py         # Local ingredients DB (5,000 LOC)
│   ├── disease_restrictions_db.py # Medical conditions (4,000 LOC)
│   └── flavor_profiles_db.py     # Flavor mappings (3,000 LOC)
│
├── services/                      # Service layer
│   ├── country_service.py        # Country-specific logic (2,000 LOC)
│   ├── disease_service.py        # Disease management (2,500 LOC)
│   ├── flavor_service.py         # Flavor matching (1,500 LOC)
│   └── substitution_service.py   # Ingredient substitution (2,000 LOC)
│
├── ml/                            # Machine Learning
│   ├── personalization_model.py  # User preference learning (2,000 LOC)
│   ├── nutrition_predictor.py    # Nutrient prediction (1,500 LOC)
│   └── recommendation_engine.py  # Recipe recommendations (2,000 LOC)
│
└── schemas/                       # Pydantic models
    ├── meal_plan_schemas.py      # Request/response models (1,500 LOC)
    ├── country_schemas.py        # Country data models (1,000 LOC)
    └── disease_schemas.py        # Disease models (1,000 LOC)

Total: ~37,500 LOC across phases
```

## 🚀 Implementation Phases

### **Phase 1: Foundation & Data Structures** (Days 1-2)
- ✅ Set up directory structure
- ✅ Create database models
- ✅ Implement country and cuisine data
- ✅ Build local foods database
- ✅ Define disease restrictions

**Files Generated:** 6 files, ~8,000 LOC

### **Phase 2: Core Services** (Days 3-4)
- ✅ Implement meal planner engine
- ✅ Create country-specific service
- ✅ Build disease management service
- ✅ Develop flavor profiling service
- ✅ Create substitution engine

**Files Generated:** 7 files, ~12,000 LOC

### **Phase 3: ML & Optimization** (Days 5-6)
- ✅ Implement nutrition optimizer
- ✅ Build personalization model
- ✅ Create recommendation engine
- ✅ Add nutrient prediction
- ✅ Integrate feedback loop

**Files Generated:** 5 files, ~8,000 LOC

### **Phase 4: API & Integration** (Days 7-8)
- ✅ Create API endpoints
- ✅ Define request/response schemas
- ✅ Add authentication & authorization
- ✅ Implement rate limiting
- ✅ Add comprehensive error handling

**Files Generated:** 4 files, ~5,000 LOC

### **Phase 5: Testing & Documentation** (Days 9-10)
- ✅ Unit tests for all services
- ✅ Integration tests
- ✅ Load testing
- ✅ API documentation
- ✅ Deployment guides

**Files Generated:** 8 files, ~4,500 LOC

---

## 📋 Step-by-Step Implementation

## Phase 1: Foundation & Data Structures

### Step 1.1: Create Directory Structure

```bash
cd flaskbackend/app
mkdir -p meal_planning/{core,data,services,ml,schemas,utils}
mkdir -p meal_planning/tests
```

### Step 1.2: Install Dependencies

Add to `requirements.txt`:
```
pydantic>=2.0.0
sqlalchemy>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
redis>=4.5.0
```

Install:
```bash
pip install -r requirements.txt
```

### Step 1.3: Database Models

**File:** `meal_planning/models.py`

This file contains SQLAlchemy models for:
- User profiles with health data
- Meal plans and recipes
- Country preferences
- Disease conditions
- Ingredient databases

**Generated:** ~1,500 LOC

### Step 1.4: Country & Cuisine Database

**File:** `meal_planning/data/country_cuisine_db.py`

Contains data for 195+ countries including:
- Traditional cuisines
- Popular dishes
- Cooking methods
- Spice profiles
- Meal timing customs

**Generated:** ~4,000 LOC

### Step 1.5: Local Foods Database

**File:** `meal_planning/data/local_foods_db.py`

Comprehensive database of:
- 10,000+ local ingredients per region
- Seasonal availability
- Nutritional profiles
- Cultural significance
- Substitution mappings

**Generated:** ~5,000 LOC

### Step 1.6: Disease Restrictions Database

**File:** `meal_planning/data/disease_restrictions_db.py`

Medical condition guidelines for:
- 100+ diseases and conditions
- Nutritional restrictions
- Recommended nutrients
- Foods to avoid
- Portion guidelines

**Generated:** ~4,000 LOC

---

## Phase 2: Core Services

### Step 2.1: Meal Planner Engine

**File:** `meal_planning/core/meal_planner_engine.py`

Main orchestrator that:
- Processes user requests
- Applies country filters
- Enforces disease restrictions
- Optimizes nutrition
- Generates meal plans

**Generated:** ~3,000 LOC

**Key Features:**
```python
class MealPlannerEngine:
    def generate_plan(self, user_profile, preferences, duration)
    def apply_country_filters(self, country_code)
    def enforce_disease_restrictions(self, conditions)
    def optimize_nutrition(self, meals, targets)
    def calculate_variety_score(self, plan)
```

### Step 2.2: Country Service

**File:** `meal_planning/services/country_service.py`

Handles country-specific logic:
- Local ingredient sourcing
- Cultural meal patterns
- Regional cooking techniques
- Festival and seasonal foods

**Generated:** ~2,000 LOC

### Step 2.3: Disease Management Service

**File:** `meal_planning/services/disease_service.py`

Medical condition management:
- Condition analysis
- Restriction enforcement
- Nutrient monitoring
- Drug-nutrient interactions
- Progress tracking

**Generated:** ~2,500 LOC

### Step 2.4: Flavor Profiling Service

**File:** `meal_planning/services/flavor_service.py`

Flavor matching and profiling:
- Taste profile analysis
- Flavor pairing algorithms
- Spice level adjustment
- Texture balancing

**Generated:** ~1,500 LOC

### Step 2.5: Ingredient Substitution Service

**File:** `meal_planning/services/substitution_service.py`

Smart ingredient replacement:
- Allergen substitution
- Availability-based swaps
- Nutrition-equivalent replacements
- Cost optimization

**Generated:** ~2,000 LOC

---

## Phase 3: ML & Optimization

### Step 3.1: Nutrition Optimizer

**File:** `meal_planning/core/nutrition_optimizer.py`

ML-powered optimization:
- Multi-objective optimization
- Macro/micro balancing
- Caloric distribution
- Nutrient timing
- Portion size optimization

**Generated:** ~2,500 LOC

**Algorithms:**
- Linear programming for nutrient targets
- Genetic algorithms for recipe combinations
- Gradient descent for portion optimization

### Step 3.2: Personalization Model

**File:** `meal_planning/ml/personalization_model.py`

User preference learning:
- Collaborative filtering
- Taste profile modeling
- Historical preference analysis
- Feedback incorporation
- A/B testing support

**Generated:** ~2,000 LOC

### Step 3.3: Recommendation Engine

**File:** `meal_planning/ml/recommendation_engine.py`

Recipe recommendations:
- Content-based filtering
- Hybrid recommendation
- Context-aware suggestions
- Novelty vs familiarity balance

**Generated:** ~2,000 LOC

### Step 3.4: Nutrient Predictor

**File:** `meal_planning/ml/nutrition_predictor.py`

Predictive analytics:
- Meal impact prediction
- Health outcome modeling
- Biomarker forecasting
- Adherence prediction

**Generated:** ~1,500 LOC

---

## Phase 4: API & Integration

### Step 4.1: Pydantic Schemas

**File:** `meal_planning/schemas/meal_plan_schemas.py`

Request/response models:
```python
class MealPlanRequest(BaseModel):
    user_id: str
    country_code: str
    duration_days: int
    disease_conditions: List[str]
    dietary_restrictions: List[str]
    calorie_target: Optional[int]
    preferences: Dict[str, Any]

class MealPlanResponse(BaseModel):
    plan_id: str
    meals: List[Meal]
    nutrition_summary: NutritionSummary
    shopping_list: List[Ingredient]
    estimated_cost: float
```

**Generated:** ~1,500 LOC

### Step 4.2: API Routes

**File:** `meal_planning/routes/meal_planner_routes.py`

RESTful endpoints:
- `POST /api/v1/meal-planner/generate` - Generate plan
- `GET /api/v1/meal-planner/{plan_id}` - Get plan
- `PUT /api/v1/meal-planner/{plan_id}` - Update plan
- `POST /api/v1/meal-planner/countries` - List countries
- `POST /api/v1/meal-planner/diseases` - List diseases

**Generated:** ~2,000 LOC

### Step 4.3: Integration Layer

**File:** `meal_planning/core/integration_service.py`

External integrations:
- USDA FoodData Central API
- EFSA database integration
- Recipe APIs
- Grocery price APIs
- Delivery service APIs

**Generated:** ~1,500 LOC

---

## Phase 5: Testing & Documentation

### Step 5.1: Unit Tests

**Files:**
- `meal_planning/tests/test_meal_planner_engine.py`
- `meal_planning/tests/test_country_service.py`
- `meal_planning/tests/test_disease_service.py`
- `meal_planning/tests/test_nutrition_optimizer.py`

**Generated:** ~3,000 LOC

### Step 5.2: Integration Tests

**File:** `meal_planning/tests/test_integration.py`

End-to-end testing:
- Complete meal plan generation
- Multi-country scenarios
- Complex disease conditions
- Edge cases and error handling

**Generated:** ~1,000 LOC

### Step 5.3: API Documentation

Auto-generated OpenAPI/Swagger docs with examples for all endpoints.

---

## 🔧 Configuration

### Environment Variables

Create `.env` file:
```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/wellomex
REDIS_URL=redis://localhost:6379

# External APIs
USDA_API_KEY=your_usda_key
EDAMAM_API_KEY=your_edamam_key
SPOONACULAR_API_KEY=your_spoonacular_key

# ML Models
MODEL_PATH=/app/models
ENABLE_ML_OPTIMIZATION=true

# Feature Flags
ENABLE_COUNTRY_FILTERING=true
ENABLE_DISEASE_RESTRICTIONS=true
ENABLE_FLAVOR_PROFILING=true
```

---

## 🧪 Testing

### Run Unit Tests
```bash
pytest meal_planning/tests/test_*.py -v
```

### Run Integration Tests
```bash
pytest meal_planning/tests/test_integration.py -v --integration
```

### Load Testing
```bash
locust -f meal_planning/tests/locustfile.py
```

---

## 📈 Performance Optimization

### Caching Strategy
- Redis for country/cuisine data
- In-memory caching for disease restrictions
- CDN for static ingredient images

### Database Optimization
- Indexes on user_id, country_code, disease_id
- Partitioning on created_date
- Read replicas for analytics

### API Rate Limiting
- 100 requests/minute per user
- 1,000 requests/minute per API key
- Burst allowance: 150 requests

---

## 🚢 Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t wellomex-meal-planner .
docker run -p 8000:8000 --env-file .env wellomex-meal-planner
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: meal-planner
spec:
  replicas: 3
  selector:
    matchLabels:
      app: meal-planner
  template:
    metadata:
      labels:
        app: meal-planner
    spec:
      containers:
      - name: meal-planner
        image: wellomex-meal-planner:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
```

---

## 📊 Monitoring & Observability

### Metrics to Track
- Meal plan generation time
- Cache hit ratio
- API response times
- Error rates by endpoint
- User satisfaction scores

### Logging
```python
import logging

logger = logging.getLogger("meal_planning")
logger.info("Generated meal plan", extra={
    "user_id": user_id,
    "country": country_code,
    "duration": duration_days,
    "generation_time_ms": elapsed_ms
})
```

---

## 🔐 Security Considerations

1. **Authentication:** JWT tokens with role-based access
2. **Data Privacy:** Encrypt sensitive health data at rest
3. **Rate Limiting:** Prevent abuse and DoS attacks
4. **Input Validation:** Sanitize all user inputs
5. **API Keys:** Rotate external API keys regularly

---

## 📚 Example Usage

### Generate Country-Specific Meal Plan

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/meal-planner/generate",
    json={
        "user_id": "user123",
        "country_code": "KE",  # Kenya
        "duration_days": 7,
        "disease_conditions": ["diabetes_type2", "hypertension"],
        "dietary_restrictions": ["halal"],
        "calorie_target": 2000,
        "preferences": {
            "spice_level": "medium",
            "local_foods_priority": "high",
            "cooking_time_max": 45
        }
    },
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)

meal_plan = response.json()
```

### Expected Response

```json
{
  "plan_id": "mp_ke_20250128_001",
  "country": "Kenya",
  "duration_days": 7,
  "meals": [
    {
      "day": 1,
      "breakfast": {
        "name": "Uji (Millet Porridge) with Banana",
        "ingredients": [...],
        "nutrition": {
          "calories": 350,
          "protein_g": 12,
          "carbs_g": 58,
          "fiber_g": 8,
          "sodium_mg": 150
        },
        "disease_compliance": {
          "diabetes_safe": true,
          "hypertension_safe": true,
          "glycemic_index": "low"
        }
      },
      "lunch": {...},
      "dinner": {...}
    }
  ],
  "nutrition_summary": {
    "daily_avg_calories": 1980,
    "protein_percent": 25,
    "carbs_percent": 45,
    "fat_percent": 30
  },
  "shopping_list": [...],
  "estimated_cost_usd": 45.50,
  "local_foods_percentage": 85
}
```

---

## 🌍 Supported Countries (Sample)

- **Africa:** Kenya, Nigeria, South Africa, Egypt, Ethiopia, Ghana, Tanzania...
- **Asia:** India, China, Japan, Thailand, Vietnam, Indonesia, Philippines...
- **Europe:** UK, France, Germany, Italy, Spain, Greece, Poland...
- **Americas:** USA, Mexico, Brazil, Argentina, Canada, Peru...
- **Middle East:** UAE, Saudi Arabia, Israel, Turkey, Iran, Lebanon...
- **Oceania:** Australia, New Zealand, Fiji, Papua New Guinea...

**Total:** 195+ countries with localized data

---

## 💡 Disease Conditions Supported (Sample)

### Metabolic Disorders
- Diabetes Type 1 & 2
- Pre-diabetes
- Metabolic syndrome
- Obesity

### Cardiovascular
- Hypertension
- Heart disease
- High cholesterol
- Atherosclerosis

### Digestive
- Celiac disease
- IBS
- Crohn's disease
- GERD

### Kidney
- Chronic kidney disease (Stages 1-5)
- Dialysis
- Kidney stones

### Autoimmune
- Rheumatoid arthritis
- Lupus
- Multiple sclerosis

**Total:** 100+ conditions with specific dietary guidelines

---

## 🎓 Next Steps

1. **Phase 1:** Start with database setup and country data
2. **Phase 2:** Implement core meal planning engine
3. **Phase 3:** Add ML optimization
4. **Phase 4:** Create API endpoints
5. **Phase 5:** Test and deploy

Each phase builds on the previous, ensuring a solid foundation.

---

## 🆘 Support & Troubleshooting

### Common Issues

**Issue:** Meal plans not respecting disease restrictions
- **Solution:** Check `disease_restrictions_db.py` configuration
- **Verify:** Disease IDs match in user profile

**Issue:** Country-specific foods not showing
- **Solution:** Ensure country_code is ISO 3166-1 alpha-2
- **Verify:** Local foods database has data for that country

**Issue:** Slow generation times
- **Solution:** Enable Redis caching
- **Optimize:** Database queries with proper indexes

---

## 📞 Contact

- **Email:** support@wellomex.com
- **Docs:** https://docs.wellomex.com/meal-planning
- **GitHub:** https://github.com/wellomex/meal-planning

---

**Version:** 1.0.0  
**Last Updated:** November 28, 2025  
**Estimated Implementation Time:** 10 days (with 2 developers)
