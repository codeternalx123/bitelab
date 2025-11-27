"""
Test LLM Disease System
========================

Comprehensive testing of LLM-powered disease profile generation
"""

import requests
import json


def test_llm_disease_system():
    """Test the complete LLM disease system"""
    
    BASE_URL = "http://127.0.0.1:5003"
    
    print("\n" + "="*100)
    print("TESTING LLM-POWERED DISEASE SYSTEM")
    print("="*100)
    
    # Test 1: Health Check
    print("\n" + "="*100)
    print("TEST 1: Health Check & System Status")
    print("="*100)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"\nStatus: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))
    
    health = response.json()
    llm_enabled = health.get('llm_mode') == 'enabled'
    
    # Test 2: List All Diseases
    print("\n" + "="*100)
    print("TEST 2: List All Diseases")
    print("="*100)
    
    response = requests.get(f"{BASE_URL}/api/v1/diseases")
    print(f"\nStatus: {response.status_code}")
    data = response.json()
    print(f"Total diseases: {data['total']}")
    print(f"\nFirst 5 diseases:")
    for disease in data['diseases'][:5]:
        print(f"  • {disease['name']} ({disease['disease_id']}) - {disease['category']}")
    
    # Test 3: Get Disease (Fallback Mode)
    print("\n" + "="*100)
    print("TEST 3: Get Disease Profile (Fallback Database)")
    print("="*100)
    
    response = requests.get(f"{BASE_URL}/api/v1/disease/diabetes_type2")
    print(f"\nStatus: {response.status_code}")
    disease = response.json()
    print(f"\nDisease: {disease['name']}")
    print(f"ICD-10: {', '.join(disease['icd10_codes'])}")
    print(f"Category: {disease['category']}")
    print(f"Nutritional Guidelines: {len(disease['nutritional_guidelines'])}")
    print(f"Food Restrictions: {len(disease['food_restrictions'])}")
    print(f"Recommended Foods: {len(disease['recommended_foods'])}")
    
    print(f"\nSample Guidelines:")
    for guideline in disease['nutritional_guidelines'][:3]:
        print(f"  • {guideline['nutrient']}: {guideline['target']} {guideline['unit']} (Priority: {guideline['priority']})")
    
    print(f"\nSample Restrictions:")
    for restriction in disease['food_restrictions'][:3]:
        print(f"  • AVOID {restriction['food_item']} ({restriction['severity']}) - {restriction['reason']}")
    
    # Test 4: LLM Configuration
    print("\n" + "="*100)
    print("TEST 4: LLM Configuration Status")
    print("="*100)
    
    response = requests.get(f"{BASE_URL}/api/v1/llm/config")
    print(f"\nStatus: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))
    
    # Test 5: Cache Statistics
    print("\n" + "="*100)
    print("TEST 5: Cache Statistics")
    print("="*100)
    
    response = requests.get(f"{BASE_URL}/api/v1/cache/stats")
    print(f"\nStatus: {response.status_code}")
    stats = response.json()
    print(f"\nTotal Cached Profiles: {stats['total_cached']}")
    print(f"Cache Directory: {stats['cache_dir']}")
    print(f"Cache TTL: {stats['cache_ttl_days']} days")
    
    if stats['profiles']:
        print(f"\nCached Profiles:")
        for profile in stats['profiles']:
            status = "✅ Valid" if profile['is_valid'] else "⚠️ Expired"
            print(f"  {status} {profile['name']} ({profile['age_days']} days old) - {profile['llm_model']}")
    else:
        print("\n⚠️ No cached profiles found")
    
    # Test 6: Multi-Disease Optimization
    print("\n" + "="*100)
    print("TEST 6: Multi-Disease Family Optimization")
    print("="*100)
    
    family_data = {
        "family_members": [
            {
                "name": "John",
                "age": 55,
                "diseases": ["diabetes_type2", "hypertension"],
                "dietary_preference": "non-vegetarian"
            },
            {
                "name": "Mary",
                "age": 52,
                "diseases": ["osteoporosis"],
                "dietary_preference": "vegetarian"
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/optimize", json=family_data)
    print(f"\nStatus: {response.status_code}")
    result = response.json()
    
    print(f"\nFamily Members: {result['total_members']}")
    print(f"Total Diseases: {result['total_diseases_considered']}")
    print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
    print(f"\nUnified Nutritional Targets: {len(result['unified_nutritional_targets'])}")
    print(f"Food Restrictions: {len(result['food_restrictions'])}")
    print(f"Recommended Foods: {len(result['recommended_foods'])}")
    
    # LLM-specific tests (only if LLM is enabled)
    if llm_enabled:
        print("\n" + "="*100)
        print("LLM-POWERED TESTS")
        print("="*100)
        
        # Test 7: Generate Single LLM Profile
        print("\n" + "="*100)
        print("TEST 7: Generate Disease Profile with LLM")
        print("="*100)
        print("\n⚠️ This will make an API call to GPT-4 (costs ~$0.10-0.30)")
        
        generate_data = {
            "disease_name": "Hypertension",
            "icd10_code": "I10",
            "category": "cardiovascular"
        }
        
        print(f"\nGenerating profile for: {generate_data['disease_name']}")
        print("This may take 10-30 seconds...")
        
        response = requests.post(f"{BASE_URL}/api/v1/llm/generate", json=generate_data)
        print(f"\nStatus: {response.status_code}")
        
        if response.status_code == 200:
            profile = response.json()
            print(f"\n✅ LLM Profile Generated:")
            print(f"   Disease: {profile['name']}")
            print(f"   Model: {profile['llm_model']}")
            print(f"   Last Updated: {profile['last_updated']}")
            print(f"   Guidelines: {len(profile['nutritional_guidelines'])}")
            print(f"   Restrictions: {len(profile['food_restrictions'])}")
            
            print(f"\n   Sample LLM Guidelines:")
            for g in profile['nutritional_guidelines'][:3]:
                print(f"     • {g['nutrient']}: {g['target']} {g['unit']} (Priority: {g['priority']})")
                print(f"       Reasoning: {g['reasoning']}")
            
            print(f"\n   Evidence Sources:")
            for source in profile['evidence_sources']:
                print(f"     • {source}")
        else:
            print(f"❌ Failed: {response.json()}")
        
        # Test 8: Get Disease with LLM Regeneration
        print("\n" + "="*100)
        print("TEST 8: Get Disease with Force Regenerate")
        print("="*100)
        print("\n⚠️ This will regenerate the profile using GPT-4 (costs ~$0.10-0.30)")
        
        response = requests.get(f"{BASE_URL}/api/v1/disease/diabetes_type2?force_regenerate=true")
        print(f"\nStatus: {response.status_code}")
        
        if response.status_code == 200:
            disease_llm = response.json()
            print(f"\n✅ Retrieved (LLM-generated): {disease_llm['name']}")
            print(f"   Guidelines: {len(disease_llm['nutritional_guidelines'])}")
            print(f"   Restrictions: {len(disease_llm['food_restrictions'])}")
        
        # Test 9: Batch Generation
        print("\n" + "="*100)
        print("TEST 9: Batch Generate Profiles (Optional)")
        print("="*100)
        print("\n⚠️ COST WARNING: Batch generation can be expensive!")
        print("\nExample batch generation (3 diseases, ~$0.30-1.50):")
        print("""
POST {BASE_URL}/api/v1/llm/batch-generate
{
    "diseases": [
        {"name": "Type 2 Diabetes", "icd10": "E11", "category": "endocrine"},
        {"name": "Hypertension", "icd10": "I10", "category": "cardiovascular"},
        {"name": "Obesity", "icd10": "E66.9", "category": "endocrine"}
    ],
    "delay": 2.0
}
        """)
        
        print("\n💡 To batch generate all 175 diseases:")
        print("   Cost: $35-90")
        print("   Time: ~6-8 hours (with 2 second delay)")
        print("   Profiles are cached for 30 days")
    
    else:
        print("\n" + "="*100)
        print("LLM NOT CONFIGURED")
        print("="*100)
        print("\nTo enable LLM features:")
        print("  1. Install: pip install openai")
        print("  2. Get API key: https://platform.openai.com/api-keys")
        print("  3. Set environment variable:")
        print("     Windows: set OPENAI_API_KEY=sk-your-key")
        print("     Linux/Mac: export OPENAI_API_KEY=sk-your-key")
        print("  4. Restart the API server")
        print("\nCurrent Status:")
        print("  ✅ API is working with fallback database (175 diseases)")
        print("  ⚠️ LLM generation disabled (will use hardcoded profiles)")
    
    # Test 10: API Documentation
    print("\n" + "="*100)
    print("TEST 10: API Documentation")
    print("="*100)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"\nStatus: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))
    
    print("\n" + "="*100)
    print("TESTING COMPLETE")
    print("="*100)
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"\n✅ API Server: Running on {BASE_URL}")
    print(f"✅ Total Diseases: {health['total_diseases']}")
    print(f"✅ LLM Status: {health['llm_mode']}")
    print(f"✅ Cached Profiles: {stats['total_cached']}")
    
    if llm_enabled:
        print("\n🚀 LLM Features Available:")
        print("   • Generate profiles on-demand")
        print("   • Force regenerate existing profiles")
        print("   • Batch generation")
        print("   • Automatic caching (30 day TTL)")
    else:
        print("\n📦 Using Fallback Database:")
        print("   • 175 pre-configured diseases")
        print("   • No API costs")
        print("   • Fast response times")
        print("   • Evidence-based recommendations")
    
    print("\n" + "="*100)


def generate_curl_examples():
    """Generate curl examples for testing"""
    
    print("\n" + "="*100)
    print("CURL COMMAND EXAMPLES")
    print("="*100)
    
    examples = [
        ("Health Check", 'curl http://127.0.0.1:5003/health'),
        
        ("List All Diseases", 'curl http://127.0.0.1:5003/api/v1/diseases'),
        
        ("Get Disease", 'curl http://127.0.0.1:5003/api/v1/disease/diabetes_type2'),
        
        ("Get Disease (Force LLM Regenerate)", 
         'curl "http://127.0.0.1:5003/api/v1/disease/diabetes_type2?force_regenerate=true"'),
        
        ("Optimize Family Nutrition",
         '''curl -X POST http://127.0.0.1:5003/api/v1/optimize \\
  -H "Content-Type: application/json" \\
  -d '{"family_members":[{"name":"John","age":55,"diseases":["diabetes_type2","hypertension"]}]}'
'''),
        
        ("Generate LLM Profile",
         '''curl -X POST http://127.0.0.1:5003/api/v1/llm/generate \\
  -H "Content-Type: application/json" \\
  -d '{"disease_name":"Hypertension","icd10_code":"I10","category":"cardiovascular"}'
'''),
        
        ("Cache Stats", 'curl http://127.0.0.1:5003/api/v1/cache/stats'),
        
        ("Clear Cache", 
         'curl -X POST http://127.0.0.1:5003/api/v1/cache/clear -H "Content-Type: application/json"'),
    ]
    
    for title, cmd in examples:
        print(f"\n{title}:")
        print("-" * 100)
        print(cmd)
    
    print("\n" + "="*100)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "curl":
        generate_curl_examples()
    else:
        test_llm_disease_system()
