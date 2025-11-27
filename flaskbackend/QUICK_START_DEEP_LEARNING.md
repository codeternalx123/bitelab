# Deep Learning & Knowledge Graphs - Quick Start Guide

## Installation

### 1. Install PyTorch (required for deep learning)

**CPU Version**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**GPU Version (CUDA 11.8)**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install Other Dependencies

```bash
pip install openai anthropic numpy
```

### 3. Set API Keys

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # Optional
```

## Basic Usage

### Example 1: Analyze a Food Item

```python
import asyncio
from app.ai_nutrition.knowledge_graphs.integrated_nutrition_ai import (
    create_integrated_system,
    UserContext,
    FoodItem
)

async def main():
    # Initialize system
    ai_system = create_integrated_system(
        llm_api_key="your-openai-key",
        device="cpu"  # or "cuda" for GPU
    )
    
    # Define user profile
    user = UserContext(
        user_id="user123",
        age=35,
        gender="female",
        weight=65.0,  # kg
        height=165.0,  # cm
        activity_level="moderate",
        health_goals=["weight loss", "heart health"],
        medical_conditions=["prediabetes"],
        medications=["metformin"],
        allergies=[],
        dietary_restrictions=["vegetarian"],
        biomarkers={"glucose": 105, "cholesterol": 200}
    )
    
    # Define food
    food = FoodItem(
        name="quinoa salad",
        ingredients=["quinoa", "spinach", "tomatoes", "olive oil"],
        nutrients={
            "protein": 8,
            "carbs": 39,
            "fiber": 5,
            "calories": 220
        },
        portion_size=150
    )
    
    # Analyze
    recommendation = await ai_system.analyze_food(
        food=food,
        user_context=user,
        use_graph=True,
        use_deep_model=True,
        use_llm=False
    )
    
    # Print results
    print(f"Recommendation Score: {recommendation.recommendation_score:.2f}")
    print(f"\nGoal Alignments:")
    for goal, score in recommendation.alignment_with_goals.items():
        print(f"  - {goal}: {score:.2f}")
    
    print(f"\nDisease Impacts:")
    for disease, risk in recommendation.disease_risk_impacts.items():
        print(f"  - {disease}: {risk:.2f}")
    
    print(f"\nOptimal Portion: {recommendation.portion_recommendation}g")
    print(f"Confidence: {recommendation.confidence:.2f}")
    
    print(f"\nReasoning:")
    for reason in recommendation.reasoning:
        print(f"  - {reason}")
    
    if recommendation.cautions:
        print(f"\nCautions:")
        for caution in recommendation.cautions:
            print(f"  ⚠️  {caution}")
    
    if recommendation.alternatives:
        print(f"\nAlternatives: {', '.join(recommendation.alternatives)}")

# Run
asyncio.run(main())
```

**Output**:
```
Recommendation Score: 0.85

Goal Alignments:
  - weight loss: 0.78
  - heart health: 0.91

Disease Impacts:
  - prediabetes: 0.15

Optimal Portion: 180.0g
Confidence: 0.90

Reasoning:
  - Supports heart health (knowledge graph confidence: 0.89)
  - Deep learning model predicts strong support for weight loss (score: 0.78)
  - High fiber content beneficial for glucose management

Alternatives: brown rice bowl, lentil soup
```

### Example 2: Query Knowledge Graph

```python
from app.ai_nutrition.knowledge_graphs.food_knowledge_graph import (
    FoodKnowledgeGraph,
    RelationType
)

# Create knowledge graph
graph = FoodKnowledgeGraph()

# Query relationships
relations = graph.query("omega-3", RelationType.BENEFITS)

print("Omega-3 Benefits:")
for rel in relations:
    print(f"  - {rel['to']} (confidence: {rel['confidence']:.2f})")
```

**Output**:
```
Omega-3 Benefits:
  - heart health (confidence: 0.95)
  - brain health (confidence: 0.88)
  - inflammation reduction (confidence: 0.90)
```

### Example 3: Expand Knowledge with LLM

```python
import asyncio
from app.ai_nutrition.knowledge_graphs.food_knowledge_graph import FoodKnowledgeGraph
from openai import AsyncOpenAI

async def expand_knowledge():
    # Create graph
    graph = FoodKnowledgeGraph()
    
    # Create OpenAI client
    client = AsyncOpenAI(api_key="your-key")
    
    # Expand knowledge for a food
    await graph.expand_knowledge_from_llm(
        entity_name="chia seeds",
        llm_client=client
    )
    
    # Check new relationships
    relations = graph.query("chia seeds")
    
    print("Chia Seeds Relationships:")
    for rel in relations:
        print(f"  {rel['relation_type']}: {rel['to']} (confidence: {rel['confidence']:.2f})")

asyncio.run(expand_knowledge())
```

**Output**:
```
Chia Seeds Relationships:
  CONTAINS: omega-3 (confidence: 0.90)
  CONTAINS: fiber (confidence: 0.85)
  BENEFITS: digestive health (confidence: 0.75)
  BENEFITS: weight management (confidence: 0.70)
```

### Example 4: Train Models from LLM Data

```python
import asyncio
from app.ai_nutrition.knowledge_graphs.integrated_nutrition_ai import create_integrated_system

async def train_models():
    ai_system = create_integrated_system(llm_api_key="your-key")
    
    # Generate training data from GPT-4
    training_data = await ai_system.train_from_llm(
        foods=["salmon", "broccoli", "quinoa", "almonds"],
        health_goals=["heart health", "weight loss", "muscle gain"],
        diseases=["diabetes", "hypertension"],
        num_samples=100
    )
    
    print(f"Generated {len(training_data)} training examples")
    print(f"Sample: {training_data[0]}")

asyncio.run(train_models())
```

**Output**:
```
Generated 100 training examples
Sample: {
  'food': 'salmon',
  'goal_scores': {
    'heart health': 0.95,
    'weight loss': 0.78,
    'muscle gain': 0.85
  },
  'disease_scores': {
    'diabetes': 0.80,
    'hypertension': 0.88
  }
}
```

## Using the API

### Start the Server

```bash
cd flaskbackend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Examples

#### 1. Analyze Food

```bash
curl -X POST "http://localhost:8000/api/v1/ai/analyze-food" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "food": {
      "name": "grilled salmon",
      "ingredients": ["salmon", "olive oil", "lemon"],
      "nutrients": {
        "protein": 25,
        "fat": 12,
        "omega-3": 2.5
      },
      "portion_size": 150
    },
    "user_context": {
      "age": 45,
      "gender": "male",
      "weight": 80,
      "height": 175,
      "activity_level": "active",
      "health_goals": ["heart health", "muscle gain"],
      "medical_conditions": ["high cholesterol"],
      "medications": ["statin"],
      "allergies": [],
      "dietary_restrictions": [],
      "biomarkers": {
        "cholesterol": 240,
        "ldl": 160
      }
    },
    "use_graph": true,
    "use_deep_model": true,
    "use_llm": false
  }'
```

#### 2. Query Knowledge Graph

```bash
curl -X POST "http://localhost:8000/api/v1/ai/knowledge/query" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_name": "omega-3",
    "relation_type": "BENEFITS",
    "max_results": 10
  }'
```

#### 3. Get Knowledge Stats

```bash
curl -X GET "http://localhost:8000/api/v1/ai/knowledge/stats" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### 4. Expand Knowledge (Background Task)

```bash
curl -X POST "http://localhost:8000/api/v1/ai/knowledge/expand" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "entities": ["quinoa", "chia seeds", "kale"],
    "entity_type": "FOOD"
  }'
```

#### 5. Train from LLM (Admin Only, Background Task)

```bash
curl -X POST "http://localhost:8000/api/v1/ai/train/llm-data" \
  -H "Authorization: Bearer ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "foods": ["salmon", "broccoli", "quinoa"],
    "health_goals": ["heart health", "weight loss"],
    "diseases": ["diabetes", "hypertension"],
    "num_samples": 200
  }'
```

## Integration with Existing LLM Chat

The integrated AI system is automatically available to the existing chat endpoints:

```python
# In app/ai_nutrition/orchestration/llm_orchestrator.py
integrated_ai = get_integrated_ai()

# Use in chat for enhanced recommendations
if integrated_ai:
    recommendation = await integrated_ai.analyze_food(food, user_context)
```

## Performance Tips

### 1. Use GPU for Deep Learning

If you have a CUDA-capable GPU:

```python
ai_system = create_integrated_system(
    llm_api_key="your-key",
    device="cuda"  # Much faster than CPU
)
```

### 2. Cache Embeddings

The system automatically caches embeddings for foods you've analyzed.

### 3. Batch Processing

For multiple foods, process in batches:

```python
foods = [food1, food2, food3]
tasks = [ai_system.analyze_food(f, user_context) for f in foods]
results = await asyncio.gather(*tasks)
```

### 4. Disable LLM Validation for Speed

LLM validation is slow. Only use when you need high confidence:

```python
recommendation = await ai_system.analyze_food(
    food=food,
    user_context=user,
    use_llm=False  # Faster
)
```

## Troubleshooting

### PyTorch Not Found

```bash
# Install PyTorch
pip install torch
```

### CUDA Out of Memory

```python
# Use CPU instead
ai_system = create_integrated_system(device="cpu")
```

### OpenAI API Rate Limit

```python
# Add delay between LLM calls
import asyncio
await asyncio.sleep(1)
```

### Import Errors

```bash
# Ensure you're in the right directory
cd flaskbackend
python -c "from app.ai_nutrition.knowledge_graphs import create_integrated_system"
```

## Next Steps

1. **Add More Foods**: Expand the knowledge graph with your food database
2. **Train Models**: Use LLM to generate training data for your specific use cases
3. **Fine-tune**: Collect user feedback and retrain models
4. **Integrate**: Connect with your existing food scanning and recommendation systems
5. **Monitor**: Track performance metrics and improve over time

## Support

For issues or questions:
- Check documentation: `DEEP_LEARNING_KNOWLEDGE_GRAPHS.md`
- Review API docs: http://localhost:8000/api/docs
- Contact: support@wellomex.com
