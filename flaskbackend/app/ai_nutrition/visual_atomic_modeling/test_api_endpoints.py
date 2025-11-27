"""
API Testing Script for Knowledge Graph & GPT-4 Endpoints
=========================================================

Test the REST API endpoints with sample requests.
"""

import requests
import json
from time import sleep


# API Base URL
BASE_URL = "http://127.0.0.1:5002"


def test_health_check():
    """Test API health check"""
    print("\n" + "="*80)
    print("TEST 1: Health Check")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_gpt4_config():
    """Test GPT-4 configuration"""
    print("\n" + "="*80)
    print("TEST 2: GPT-4 Configuration")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/v1/gpt4/config")
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    return response.json()


def test_gpt4_connection():
    """Test GPT-4 connection"""
    print("\n" + "="*80)
    print("TEST 3: GPT-4 Connection Test")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/v1/gpt4/test")
    print(f"\nStatus Code: {response.status_code}")
    result = response.json()
    print(f"Response:\n{json.dumps(result, indent=2)}")
    return result.get('status') == 'success'


def test_build_knowledge_graph():
    """Test knowledge graph building"""
    print("\n" + "="*80)
    print("TEST 4: Build Knowledge Graph")
    print("="*80)
    
    payload = {
        "family_members": [
            {
                "name": "John",
                "age": 55,
                "diseases": ["diabetes_type2", "hypertension"],
                "dietary_preferences": "non-vegetarian"
            },
            {
                "name": "Mary",
                "age": 52,
                "diseases": ["osteoporosis", "celiac"],
                "dietary_preferences": "gluten-free"
            }
        ]
    }
    
    print(f"\nRequest Payload:")
    print(json.dumps(payload, indent=2))
    
    response = requests.post(
        f"{BASE_URL}/v1/knowledge-graph/build",
        json=payload
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Knowledge Graph Built Successfully!")
        print(f"\nGraph Statistics:")
        print(f"  - Total Nodes: {result['knowledge_graph']['statistics']['total_nodes']}")
        print(f"  - Total Edges: {result['knowledge_graph']['statistics']['total_edges']}")
        print(f"  - Node Types: {result['knowledge_graph']['statistics']['node_types']}")
        
        print(f"\nOptimization Result:")
        print(f"  - Family Members: {result['optimization_result']['family_members_analyzed']}")
        print(f"  - Total Diseases: {result['optimization_result']['total_diseases']}")
        print(f"  - Nutritional Targets: {len(result['optimization_result']['nutritional_targets'])}")
        print(f"  - Food Restrictions: {len(result['optimization_result']['food_restrictions'])}")
        print(f"  - Recommended Foods: {len(result['optimization_result']['recommended_foods'])}")
        
        # Save LLM export
        with open('api_test_kg_export.txt', 'w') as f:
            f.write(result['llm_export'])
        print(f"\n✅ Saved LLM export to: api_test_kg_export.txt")
        
        return result
    else:
        print(f"\n❌ Error: {response.text}")
        return None


def test_gpt4_query(gpt4_available: bool):
    """Test GPT-4 query"""
    print("\n" + "="*80)
    print("TEST 5: GPT-4 Query")
    print("="*80)
    
    if not gpt4_available:
        print("\n⚠️  Skipping - GPT-4 not available")
        return
    
    payload = {
        "family_members": [
            {
                "name": "John",
                "age": 55,
                "diseases": ["diabetes_type2", "hypertension"],
                "dietary_preferences": "non-vegetarian"
            }
        ],
        "query": "What should John eat for breakfast to manage his diabetes and blood pressure?"
    }
    
    print(f"\nQuery: {payload['query']}")
    print(f"\nSending request... (may take 10-30 seconds)")
    
    response = requests.post(
        f"{BASE_URL}/v1/gpt4/query",
        json=payload
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ GPT-4 Response:")
        print("="*80)
        print(result['answer'])
        print("="*80)
        print(f"\nToken Usage: {result['token_usage']}")
        print(f"Model: {result['model']}")
    else:
        print(f"\n❌ Error: {response.text}")


def test_gpt4_meal_plan(gpt4_available: bool):
    """Test GPT-4 meal plan generation"""
    print("\n" + "="*80)
    print("TEST 6: GPT-4 Meal Plan Generation")
    print("="*80)
    
    if not gpt4_available:
        print("\n⚠️  Skipping - GPT-4 not available")
        return
    
    payload = {
        "family_members": [
            {
                "name": "John",
                "age": 55,
                "diseases": ["diabetes_type2", "hypertension"],
                "dietary_preferences": "non-vegetarian"
            },
            {
                "name": "Mary",
                "age": 52,
                "diseases": ["celiac", "osteoporosis"],
                "dietary_preferences": "gluten-free"
            }
        ],
        "preferences": {
            "cuisine": "Mediterranean",
            "prep_time": "Medium",
            "budget": "Medium",
            "skill_level": "Intermediate"
        }
    }
    
    print(f"\nPreferences: {payload['preferences']}")
    print(f"\nGenerating 7-day meal plan... (may take 30-60 seconds)")
    
    response = requests.post(
        f"{BASE_URL}/v1/gpt4/meal-plan",
        json=payload
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Meal Plan Generated!")
        print("="*80)
        print(result['meal_plan'])
        print("="*80)
        print(f"\nToken Usage: {result['token_usage']}")
        
        # Save meal plan
        with open('api_test_meal_plan.txt', 'w', encoding='utf-8') as f:
            f.write(result['meal_plan'])
        print(f"\n✅ Saved to: api_test_meal_plan.txt")
    else:
        print(f"\n❌ Error: {response.text}")


def generate_curl_examples():
    """Generate curl command examples"""
    print("\n" + "="*80)
    print("CURL COMMAND EXAMPLES")
    print("="*80)
    
    commands = [
        {
            "name": "Health Check",
            "command": f'curl -X GET {BASE_URL}/health'
        },
        {
            "name": "GPT-4 Config",
            "command": f'curl -X GET {BASE_URL}/v1/gpt4/config'
        },
        {
            "name": "Test GPT-4 Connection",
            "command": f'curl -X GET {BASE_URL}/v1/gpt4/test'
        },
        {
            "name": "Build Knowledge Graph",
            "command": f'''curl -X POST {BASE_URL}/v1/knowledge-graph/build \\
  -H "Content-Type: application/json" \\
  -d '{{
    "family_members": [
      {{
        "name": "John",
        "age": 55,
        "diseases": ["diabetes_type2", "hypertension"],
        "dietary_preferences": "non-vegetarian"
      }}
    ]
  }}\''''
        },
        {
            "name": "Query with GPT-4",
            "command": f'''curl -X POST {BASE_URL}/v1/gpt4/query \\
  -H "Content-Type: application/json" \\
  -d '{{
    "family_members": [
      {{
        "name": "John",
        "age": 55,
        "diseases": ["diabetes_type2"],
        "dietary_preferences": "non-vegetarian"
      }}
    ],
    "query": "What should John eat for breakfast?"
  }}\''''
        },
        {
            "name": "Set GPT-4 API Key",
            "command": f'''curl -X POST {BASE_URL}/v1/gpt4/config \\
  -H "Content-Type: application/json" \\
  -d '{{
    "provider": "openai",
    "api_key": "sk-your-key-here",
    "model": "gpt-4-turbo-preview"
  }}\''''
        }
    ]
    
    for cmd in commands:
        print(f"\n{cmd['name']}:")
        print(f"{cmd['command']}\n")


def main():
    """Run all API tests"""
    print("\n" + "="*80)
    print("🧪 KNOWLEDGE GRAPH & GPT-4 API TESTING SUITE")
    print("="*80)
    print(f"\nAPI Server: {BASE_URL}")
    print("Make sure the API server is running on port 5002!")
    print("\nTo start the server:")
    print("  python kg_api.py")
    
    try:
        # Test 1: Health Check
        if not test_health_check():
            print("\n❌ API server not responding. Start it with: python kg_api.py")
            return
        
        # Test 2: GPT-4 Config
        config = test_gpt4_config()
        gpt4_available = config.get('client_initialized', False)
        
        # Test 3: GPT-4 Connection
        if gpt4_available:
            gpt4_working = test_gpt4_connection()
        else:
            print("\n⚠️  GPT-4 client not initialized")
            gpt4_working = False
        
        # Test 4: Build Knowledge Graph (always works)
        kg_result = test_build_knowledge_graph()
        
        # Test 5: GPT-4 Query (requires API key)
        test_gpt4_query(gpt4_working)
        
        # Test 6: GPT-4 Meal Plan (requires API key)
        test_gpt4_meal_plan(gpt4_working)
        
        # Generate curl examples
        generate_curl_examples()
        
        # Final summary
        print("\n" + "="*80)
        print("✅ API TESTING COMPLETE")
        print("="*80)
        print(f"\n📊 Results:")
        print(f"  ✅ API Server: Running")
        print(f"  {'✅' if kg_result else '❌'} Knowledge Graph: {'Working' if kg_result else 'Failed'}")
        print(f"  {'✅' if gpt4_working else '⚠️ '} GPT-4 Integration: {'Working' if gpt4_working else 'Not Configured'}")
        
        if not gpt4_working:
            print(f"\n💡 To enable GPT-4:")
            print(f"  1. pip install openai")
            print(f"  2. Get API key: https://platform.openai.com/api-keys")
            print(f"  3. Set via API: POST {BASE_URL}/v1/gpt4/config")
            print(f"  4. Or set env var: OPENAI_API_KEY=sk-...")
        
        print("\n" + "="*80)
    
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API server!")
        print("   Start the server with: python kg_api.py")
        print("   Default URL: http://127.0.0.1:5002")


if __name__ == "__main__":
    main()
