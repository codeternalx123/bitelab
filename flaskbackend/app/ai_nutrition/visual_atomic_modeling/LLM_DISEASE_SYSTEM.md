# LLM-Powered Disease Database System

## Overview

This system generates disease nutritional profiles using GPT-4 or other LLMs instead of hardcoded data. It provides:

- **Dynamic Profile Generation**: LLM creates evidence-based nutritional guidelines on-demand
- **Intelligent Caching**: Profiles cached for 30 days to minimize API costs
- **Fallback Support**: Works without LLM using comprehensive hardcoded database
- **REST API**: Full API access to all features
- **Cost Optimization**: Smart caching and batch generation options

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     LLM Hybrid Database                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐      ┌──────────────────┐            │
│  │  LLM Generator  │      │  Cache (30 days) │            │
│  │   GPT-4/Claude  │◄────►│   JSON Files     │            │
│  └─────────────────┘      └──────────────────┘            │
│          │                                                  │
│          │                                                  │
│          ▼                                                  │
│  ┌─────────────────┐                                       │
│  │  Fallback DB    │                                       │
│  │  175 Diseases   │                                       │
│  └─────────────────┘                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │     REST API Server      │
            │      Port 5003           │
            └──────────────────────────┘
```

---

## Features

### 1. LLM Profile Generation

Generate disease profiles with GPT-4 that include:
- **Nutritional Guidelines**: 10-15 specific, measurable targets with priorities
- **Food Restrictions**: 8-12 restrictions with severity levels and alternatives
- **Recommended Foods**: 15-25 foods tailored to the condition
- **Evidence Sources**: Citations from WHO, ADA, AHA, medical research
- **Meal Timing & Portions**: Practical eating recommendations

### 2. Intelligent Caching

- Profiles cached for 30 days (configurable)
- Automatic cache validation
- Force regeneration option
- Cache statistics and management

### 3. Dual Mode Operation

**LLM Mode** (with API key):
- Generate new profiles on-demand
- Update existing profiles
- Batch generation for all diseases
- Always current medical recommendations

**Fallback Mode** (no API key):
- 175 pre-configured diseases
- No API costs
- Instant responses
- Evidence-based recommendations

---

## Setup

### Prerequisites

```bash
# Install required packages
pip install flask flask-cors openai anthropic
```

### Configuration

#### Option 1: OpenAI GPT-4

```bash
# Get API key from https://platform.openai.com/api-keys

# Windows
set OPENAI_API_KEY=sk-your-key-here

# Linux/Mac
export OPENAI_API_KEY=sk-your-key-here
```

#### Option 2: Azure OpenAI

```bash
set OPENAI_API_KEY=your-azure-key
set AZURE_ENDPOINT=https://your-resource.openai.azure.com
set AZURE_API_VERSION=2024-02-15-preview
```

#### Option 3: Anthropic Claude

```bash
set ANTHROPIC_API_KEY=sk-ant-your-key
```

---

## Usage

### Start the API Server

```bash
cd flaskbackend/app/ai_nutrition/visual_atomic_modeling
python llm_disease_api.py
```

Server runs on: `http://127.0.0.1:5003`

### API Endpoints

#### 1. Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "llm_mode": "enabled",
  "total_diseases": 175
}
```

#### 2. Get Disease Profile

```bash
GET /api/v1/disease/<disease_id>?force_regenerate=false
```

Examples:
```bash
# Get from cache/fallback
curl http://127.0.0.1:5003/api/v1/disease/diabetes_type2

# Force LLM regeneration
curl "http://127.0.0.1:5003/api/v1/disease/diabetes_type2?force_regenerate=true"
```

Response:
```json
{
  "disease_id": "diabetes_type2",
  "name": "Type 2 Diabetes Mellitus",
  "icd10_codes": ["E11"],
  "category": "endocrine",
  "nutritional_guidelines": [
    {
      "nutrient": "carbohydrates",
      "target": "45-60",
      "unit": "%",
      "priority": "high",
      "reasoning": "Balanced intake prevents blood sugar spikes"
    }
  ],
  "food_restrictions": [...],
  "recommended_foods": [...]
}
```

#### 3. List All Diseases

```bash
GET /api/v1/diseases?category=cardiovascular
```

#### 4. Optimize Family Nutrition

```bash
POST /api/v1/optimize
Content-Type: application/json

{
  "family_members": [
    {
      "name": "John",
      "age": 55,
      "diseases": ["diabetes_type2", "hypertension"],
      "dietary_preference": "non-vegetarian"
    }
  ]
}
```

#### 5. Generate LLM Profile

```bash
POST /api/v1/llm/generate
Content-Type: application/json

{
  "disease_name": "Type 2 Diabetes Mellitus",
  "icd10_code": "E11",
  "category": "endocrine",
  "patient_context": {
    "age": 55,
    "severity": "moderate"
  }
}
```

**Cost**: ~$0.10-0.30 per profile

#### 6. Batch Generate Profiles

```bash
POST /api/v1/llm/batch-generate
Content-Type: application/json

{
  "diseases": [
    {"name": "Type 2 Diabetes", "icd10": "E11", "category": "endocrine"},
    {"name": "Hypertension", "icd10": "I10", "category": "cardiovascular"}
  ],
  "delay": 2.0
}
```

**Cost**: ~$0.10-0.30 per disease

#### 7. Cache Management

```bash
# Get cache stats
GET /api/v1/cache/stats

# Clear specific disease
POST /api/v1/cache/clear
Content-Type: application/json
{"disease_id": "diabetes_type2"}

# Clear all cache
POST /api/v1/cache/clear
Content-Type: application/json
{}
```

#### 8. LLM Configuration

```bash
# Get config
GET /api/v1/llm/config

# Update config
POST /api/v1/llm/config
Content-Type: application/json

{
  "api_key": "sk-...",
  "provider": "openai",
  "model": "gpt-4-turbo-preview"
}
```

---

## Testing

### Run Comprehensive Tests

```bash
# Make sure API server is running first
python llm_disease_api.py

# In another terminal, run tests
python test_llm_system.py
```

### Generate Curl Examples

```bash
python test_llm_system.py curl
```

---

## Cost Estimates

### LLM Profile Generation

| Operation | Diseases | Cost Estimate | Time |
|-----------|----------|---------------|------|
| Single profile | 1 | $0.10 - $0.30 | 10-30s |
| Small batch | 10 | $1 - $3 | 2-5 min |
| Medium batch | 50 | $5 - $15 | 10-20 min |
| All diseases | 175 | $35 - $90 | 6-8 hours |

**Note**: 
- Profiles are cached for 30 days
- Regeneration only needed for updates
- Fallback database available at no cost

### Cost Optimization

1. **Use Cache**: Profiles valid for 30 days
2. **Batch Generation**: Generate all at once, use cache thereafter
3. **Selective Updates**: Only regenerate changed diseases
4. **Fallback Mode**: Use hardcoded database (no cost)

---

## Workflow Examples

### Scenario 1: First-Time Setup (LLM Mode)

```bash
# 1. Set API key
set OPENAI_API_KEY=sk-your-key

# 2. Start server
python llm_disease_api.py

# 3. Generate top 10 diseases
curl -X POST http://127.0.0.1:5003/api/v1/llm/batch-generate \
  -H "Content-Type: application/json" \
  -d '{"diseases":[...],"delay":2.0}'

# Cost: ~$2-5
# Time: ~5 minutes
# All profiles cached for 30 days
```

### Scenario 2: Production Use (Fallback Mode)

```bash
# 1. Start server (no API key needed)
python llm_disease_api.py

# 2. Use immediately
curl http://127.0.0.1:5003/api/v1/disease/diabetes_type2

# Cost: $0
# Time: <10ms
# Uses comprehensive pre-built database
```

### Scenario 3: Hybrid Approach

```bash
# 1. Use fallback for common diseases
curl http://127.0.0.1:5003/api/v1/disease/diabetes_type2

# 2. Generate LLM profiles for specific cases
curl -X POST http://127.0.0.1:5003/api/v1/llm/generate \
  -d '{"disease_name":"Rare Disease","patient_context":{...}}'

# 3. Benefits:
#    - Fast responses for common diseases
#    - Custom profiles for special cases
#    - Cost-effective hybrid model
```

---

## Python SDK Usage

```python
from llm_hybrid_disease_db import LLMHybridDiseaseDatabase

# Initialize database
db = LLMHybridDiseaseDatabase(
    use_llm=True,  # Enable LLM
    cache_ttl_days=30
)

# Get disease (uses cache/LLM/fallback automatically)
diabetes = db.get_disease("diabetes_type2")

print(f"Disease: {diabetes.name}")
print(f"Guidelines: {len(diabetes.nutritional_guidelines)}")
print(f"Restrictions: {len(diabetes.food_restrictions)}")

# Force LLM regeneration
diabetes_updated = db.get_disease("diabetes_type2", force_regenerate=True)

# Batch generate all diseases
db.batch_generate_all(max_diseases=10)  # Generate 10 diseases

# Cache management
stats = db.get_cache_stats()
print(f"Cached profiles: {stats['total_cached']}")

db.clear_cache("diabetes_type2")  # Clear specific
db.clear_cache()  # Clear all
```

---

## Comparison: LLM vs Hardcoded

| Feature | LLM Mode | Fallback Mode |
|---------|----------|---------------|
| **Setup** | Requires API key | No setup needed |
| **Cost** | ~$0.20/disease | Free |
| **Speed** | 10-30s (first gen) | <10ms |
| **Updates** | Always current | Manual updates |
| **Customization** | Patient-specific | Standard profiles |
| **Coverage** | Any disease | 175 pre-configured |
| **Evidence** | Latest research | Evidence-based |
| **Caching** | 30 days | N/A |

---

## Best Practices

### 1. Cost Management

- Generate profiles during off-peak hours
- Use batch generation (cheaper than individual calls)
- Set appropriate cache TTL (30 days recommended)
- Monitor API usage via `/api/v1/llm/config`

### 2. Performance

- Use cache for frequently accessed diseases
- Implement rate limiting for batch operations
- Consider async generation for large batches

### 3. Data Quality

- Verify LLM responses match expected schema
- Keep fallback database updated
- Review generated profiles periodically
- Use patient context for personalization

### 4. Production Deployment

- Start with fallback mode (free, fast)
- Generate LLM profiles for top 20 diseases
- Use LLM for rare/custom cases
- Monitor cache hit rates

---

## Troubleshooting

### LLM Not Working

```bash
# Check configuration
curl http://127.0.0.1:5003/api/v1/llm/config

# Verify API key
echo %OPENAI_API_KEY%  # Windows
echo $OPENAI_API_KEY   # Linux/Mac

# Test connection
curl -X POST http://127.0.0.1:5003/api/v1/llm/generate \
  -d '{"disease_name":"Test Disease"}'
```

### Cache Issues

```bash
# Check cache stats
curl http://127.0.0.1:5003/api/v1/cache/stats

# Clear and regenerate
curl -X POST http://127.0.0.1:5003/api/v1/cache/clear
curl "http://127.0.0.1:5003/api/v1/disease/diabetes_type2?force_regenerate=true"
```

### API Errors

- **503 LLM not configured**: Set OPENAI_API_KEY
- **404 Disease not found**: Check disease_id spelling
- **429 Rate limit**: Increase delay in batch generation
- **500 Internal error**: Check server logs

---

## File Structure

```
visual_atomic_modeling/
├── llm_disease_profile_generator.py  # LLM profile generator
├── llm_hybrid_disease_db.py          # Hybrid database system
├── llm_disease_api.py                # Flask API server
├── test_llm_system.py                # Comprehensive tests
├── comprehensive_disease_db.py       # Fallback database (175 diseases)
├── multi_disease_optimizer.py        # Family optimization
└── disease_profiles_cache/           # Cached LLM profiles
    ├── diabetes_type2.json
    ├── hypertension.json
    └── ...
```

---

## Next Steps

1. **Start Small**: Test with 5-10 diseases
2. **Evaluate**: Compare LLM vs hardcoded profiles
3. **Optimize**: Batch generate top diseases
4. **Deploy**: Choose LLM, fallback, or hybrid mode
5. **Monitor**: Track costs and cache performance

---

## Support

- API runs on port 5003
- Fallback database: 175 diseases ready to use
- LLM mode: Optional, requires API key
- Documentation: This file
- Tests: `test_llm_system.py`

**System Status**: ✅ Production Ready

Both LLM and fallback modes are fully functional. Choose based on your needs:
- Need latest medical research? → LLM Mode
- Need fast, free responses? → Fallback Mode
- Need both? → Hybrid Mode (recommended)
