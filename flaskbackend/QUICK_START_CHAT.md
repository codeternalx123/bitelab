# Quick Start: Testing the Conversational AI

**Get started with the ChatGPT-like nutrition assistant in 5 minutes**

---

## 🚀 Setup

### 1. Install Dependencies

```bash
cd flaskbackend
pip install openai anthropic fastapi python-multipart
```

### 2. Set API Keys

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-your-openai-key-here"
$env:ANTHROPIC_API_KEY="sk-ant-your-anthropic-key-here"

# Or add to .env file
echo "OPENAI_API_KEY=sk-your-key" >> .env
echo "ANTHROPIC_API_KEY=sk-ant-your-key" >> .env
```

### 3. Run Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit: http://localhost:8000/api/docs

---

## 🧪 Test the API

### Test 1: Create Session

```bash
curl -X POST http://localhost:8000/api/v1/chat/session \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test_token" \
  -d '{
    "mode": "general_nutrition",
    "user_profile": {
      "health_conditions": ["type2_diabetes", "hypertension"],
      "medications": ["metformin", "lisinopril"],
      "allergies": ["peanuts"],
      "health_goals": ["weight_loss", "blood_sugar_control"]
    }
  }'
```

**Expected Response:**
```json
{
  "session_id": "session_test_1732123456",
  "user_id": "test_user",
  "created_at": "2025-11-20T10:30:00",
  "mode": "general_nutrition"
}
```

**Copy the `session_id` for next steps!**

---

### Test 2: Ask a Question

```bash
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test_token" \
  -d '{
    "session_id": "YOUR_SESSION_ID_HERE",
    "message": "I just ate grilled salmon. Was that a good choice for my diabetes?"
  }'
```

**Expected Response:**
```json
{
  "session_id": "session_test_1732123456",
  "assistant_message": "Excellent choice! Grilled salmon is highly beneficial for managing type 2 diabetes. Here's why:\n\n✅ Blood Sugar Benefits:\n- Low glycemic index (won't spike blood sugar)\n- High protein helps stabilize glucose\n...",
  "function_calls": [
    {
      "name": "scan_food",
      "result": {
        "food_name": "Grilled Salmon",
        "nutrition": {...}
      }
    },
    {
      "name": "assess_health_risk",
      "result": {
        "risk_score": 15,
        "risk_level": "very_low"
      }
    }
  ],
  "timestamp": "2025-11-20T10:31:30"
}
```

---

### Test 3: Upload Image (Food Scan)

Save this as `test_scan.sh`:

```bash
#!/bin/bash

SESSION_ID="YOUR_SESSION_ID_HERE"
IMAGE_PATH="path/to/food_image.jpg"

curl -X POST http://localhost:8000/api/v1/chat/scan-and-ask \
  -H "Authorization: Bearer test_token" \
  -F "session_id=$SESSION_ID" \
  -F "question=What food is this and is it safe for me?" \
  -F "image=@$IMAGE_PATH"
```

Run: `bash test_scan.sh`

---

### Test 4: Get Recommendations

```bash
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test_token" \
  -d '{
    "session_id": "YOUR_SESSION_ID_HERE",
    "message": "What are some good foods for breakfast that will help with my weight loss goal?"
  }'
```

---

### Test 5: Generate Recipe

```bash
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test_token" \
  -d '{
    "session_id": "YOUR_SESSION_ID_HERE",
    "message": "Can you create a recipe using chicken, broccoli, and rice? I want something healthy for dinner."
  }'
```

---

### Test 6: Create Meal Plan

```bash
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test_token" \
  -d '{
    "session_id": "YOUR_SESSION_ID_HERE",
    "message": "Create a 7-day meal plan for me. My budget is $75 and I want to lose weight."
  }'
```

---

### Test 7: Get Conversation History

```bash
curl -X GET "http://localhost:8000/api/v1/chat/history/YOUR_SESSION_ID_HERE" \
  -H "Authorization: Bearer test_token"
```

---

### Test 8: Submit Feedback

```bash
curl -X POST http://localhost:8000/api/v1/chat/feedback \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test_token" \
  -d '{
    "session_id": "YOUR_SESSION_ID_HERE",
    "rating": 5.0,
    "outcome_success": true,
    "comments": "Very helpful recommendations!"
  }'
```

---

## 🐍 Python Test Script

Create `test_chat.py`:

```python
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"
TOKEN = "test_token"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

def create_session():
    """Create conversation session"""
    response = requests.post(
        f"{BASE_URL}/chat/session",
        headers=HEADERS,
        json={
            "mode": "general_nutrition",
            "user_profile": {
                "health_conditions": ["type2_diabetes"],
                "medications": ["metformin"],
                "health_goals": ["weight_loss"]
            }
        }
    )
    return response.json()["session_id"]

def send_message(session_id, message):
    """Send message to assistant"""
    response = requests.post(
        f"{BASE_URL}/chat/message",
        headers=HEADERS,
        json={
            "session_id": session_id,
            "message": message
        }
    )
    return response.json()

def main():
    # Create session
    print("Creating session...")
    session_id = create_session()
    print(f"Session ID: {session_id}\n")
    
    # Test conversations
    conversations = [
        "Is grilled salmon safe for my diabetes?",
        "What should I eat for breakfast?",
        "Create a healthy dinner recipe with chicken and vegetables",
        "Plan my meals for this week, budget $75"
    ]
    
    for msg in conversations:
        print(f"\n{'='*60}")
        print(f"USER: {msg}")
        print('='*60)
        
        response = send_message(session_id, msg)
        
        print(f"\nASSISTANT: {response['assistant_message']}")
        
        if response.get('function_calls'):
            print(f"\nFUNCTION CALLS: {len(response['function_calls'])}")
            for fc in response['function_calls']:
                print(f"  - {fc['name']}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
```

Run: `python test_chat.py`

---

## 🔍 Verify Function Calls

Check that functions are being called correctly:

### 1. scan_food
```bash
# Ask about a specific food
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test_token" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "message": "Analyze grilled chicken breast, 200g portion"
  }'

# Look for function_calls in response
# Should see: {"name": "scan_food", "result": {...}}
```

### 2. assess_health_risk
```bash
# Ask about safety
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test_token" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "message": "Is this banana safe with my lisinopril medication?"
  }'

# Should see: {"name": "assess_health_risk", "result": {"risk_score": ...}}
```

### 3. get_recommendations
```bash
# Ask for suggestions
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test_token" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "message": "What are the best foods for heart health?"
  }'

# Should see: {"name": "get_recommendations", "result": {"recommendations": [...]}}
```

---

## 📊 Check Training Data Collection

After submitting feedback (rating ≥4.0), check if data was saved:

```bash
# Windows
dir flaskbackend\data\llm_training\raw

# Linux/Mac
ls -la flaskbackend/data/llm_training/raw/

# Should see JSON files like:
# 20251120_103000_5.0_session_123.json
```

View a training example:
```bash
cat flaskbackend/data/llm_training/raw/20251120_103000_5.0_session_123.json
```

---

## 🎯 Verify Performance Monitoring

Check performance metrics in code:

```python
from app.ai_nutrition.orchestration import PerformanceMonitor

monitor = PerformanceMonitor()

# Record a test metric
monitor.record_metric(
    metric_type=MetricType.USER_SATISFACTION,
    value=5.0,
    health_goal="weight_loss",
    disease="type2_diabetes"
)

# Generate report
report = monitor.generate_performance_report()
print(json.dumps(report, indent=2))
```

---

## 🐛 Troubleshooting

### Issue: "OpenAI client not initialized"
**Solution:** Set OPENAI_API_KEY environment variable

```bash
export OPENAI_API_KEY=sk-your-key-here
```

### Issue: "Session not found"
**Solution:** Session expired (60 min timeout). Create new session.

### Issue: "Function not available"
**Solution:** Check that function_handler.py is imported correctly.

### Issue: "Authentication failed"
**Solution:** Add proper authentication. For testing, update `get_current_user` in `deps.py`:

```python
async def get_current_user():
    return {"user_id": "test_user"}
```

---

## 📈 Monitor System Performance

### Check API Logs
```bash
# Watch logs in real-time
tail -f logs/app.log

# Filter for LLM calls
grep "LLM" logs/app.log

# Filter for function calls
grep "Function execution" logs/app.log
```

### Check Response Times
```bash
# Time a request
time curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test_token" \
  -d '{"session_id": "test", "message": "Test"}'
```

---

## ✅ Success Criteria

You know it's working when:

1. ✅ Session creation returns valid session_id
2. ✅ Messages receive contextual responses
3. ✅ Function calls appear in responses
4. ✅ Food scans return nutrition data
5. ✅ Health risks are assessed correctly
6. ✅ Recommendations align with user goals
7. ✅ Training data files are created
8. ✅ Performance metrics are tracked

---

## 🎉 Next Steps

Once basic tests pass:

1. Test with real OpenAI API key (costs $)
2. Upload actual food images
3. Try complex multi-turn conversations
4. Collect 10+ conversations
5. Submit feedback ratings
6. Check training data quality
7. Review performance metrics

---

## 📞 Getting Help

- API Docs: http://localhost:8000/api/docs
- Full Documentation: `LLM_CONVERSATIONAL_AI.md`
- Implementation Details: `LLM_IMPLEMENTATION_SUMMARY.md`

---

**Happy Testing!** 🚀
