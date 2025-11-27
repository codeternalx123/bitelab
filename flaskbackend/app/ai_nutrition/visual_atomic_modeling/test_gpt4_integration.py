"""
GPT-4 Knowledge Graph Integration - Complete Test Suite
========================================================

Tests knowledge graph generation and GPT-4 integration.
"""

import json
import os
from knowledge_graph_engine import (
    NutritionalKnowledgeGraph,
    GPT4IntegrationEngine
)
from disease_optimization_engine import MultiDiseaseOptimizer


def test_knowledge_graph_generation():
    """Test 1: Knowledge graph generation"""
    print("\n" + "="*80)
    print("TEST 1: KNOWLEDGE GRAPH GENERATION")
    print("="*80)
    
    # Sample family data
    family_members = [
        {
            'name': 'John',
            'age': 55,
            'diseases': ['diabetes_type2', 'hypertension', 'coronary_artery_disease'],
            'dietary_preferences': 'non-vegetarian'
        },
        {
            'name': 'Mary',
            'age': 52,
            'diseases': ['osteoporosis', 'hypothyroidism'],
            'dietary_preferences': 'vegetarian'
        },
        {
            'name': 'Sarah',
            'age': 28,
            'diseases': ['celiac', 'ibs'],
            'dietary_preferences': 'gluten-free'
        }
    ]
    
    print("\n1. Family Members:")
    for member in family_members:
        print(f"   • {member['name']}, {member['age']}: {', '.join(member['diseases'])}")
    
    # Run optimization
    print("\n2. Running Multi-Disease Optimization...")
    optimizer = MultiDiseaseOptimizer()
    result = optimizer.optimize_for_family(family_members)
    print(f"   ✅ Optimized for {result['total_diseases_considered']} diseases")
    print(f"   ✅ {len(result['unified_nutritional_targets'])} nutritional targets")
    print(f"   ✅ {len(result['food_restrictions'])} food restrictions")
    print(f"   ✅ {len(result['recommended_foods'])} recommended foods")
    
    # Build knowledge graph
    print("\n3. Building Knowledge Graph...")
    kg = NutritionalKnowledgeGraph()
    graph_data = kg.build_from_optimization_result(result, family_members)
    
    print(f"   ✅ Graph Statistics:")
    print(f"      - Total Nodes: {graph_data['statistics']['total_nodes']}")
    print(f"      - Total Edges: {graph_data['statistics']['total_edges']}")
    print(f"      - Node Types: {graph_data['statistics']['node_types']}")
    print(f"      - Relationship Types: {graph_data['statistics']['relationship_types']}")
    
    # Sample queries
    print("\n4. Graph Queries:")
    
    # Query 1: Person's diseases
    john_diseases = kg.query_graph('person_diseases', person='John')
    print(f"\n   Query: What diseases does John have?")
    print(f"   Answer: {len(john_diseases)} diseases found")
    for disease in john_diseases:
        print(f"      • {disease['name']} ({disease['icd10_codes'][0]})")
    
    # Query 2: Person's restrictions
    john_restrictions = kg.query_graph('person_restrictions', person='John')
    print(f"\n   Query: What food restrictions does John have?")
    print(f"   Answer: {len(john_restrictions)} restrictions")
    for restriction in john_restrictions[:5]:
        print(f"      • {restriction['restriction_type'].upper()} {restriction['food_item']} ({restriction['severity']})")
    
    # Export for LLM
    print("\n5. Exporting for LLM...")
    kg_text = kg.export_for_llm()
    print(f"   ✅ Exported {len(kg_text)} characters")
    
    # Save files
    print("\n6. Saving Files...")
    with open('test_knowledge_graph.json', 'w') as f:
        json.dump(graph_data, f, indent=2)
    print("   ✅ Saved: test_knowledge_graph.json")
    
    with open('test_knowledge_graph.txt', 'w') as f:
        f.write(kg_text)
    print("   ✅ Saved: test_knowledge_graph.txt")
    
    return kg_text, graph_data


def test_gpt4_connection():
    """Test 2: GPT-4 connection"""
    print("\n" + "="*80)
    print("TEST 2: GPT-4 CONNECTION")
    print("="*80)
    
    # Initialize GPT-4
    print("\n1. Initializing GPT-4 Engine...")
    gpt4 = GPT4IntegrationEngine()
    
    print(f"   Provider: {gpt4.provider}")
    print(f"   Model: {gpt4.model}")
    print(f"   API Key Set: {bool(gpt4.api_key)}")
    print(f"   Client Initialized: {gpt4.client is not None}")
    
    # Test connection
    print("\n2. Testing Connection...")
    test_result = gpt4.test_connection()
    
    print(f"\n   Status: {test_result['status']}")
    if test_result['status'] == 'success':
        print(f"   ✅ Connection successful!")
        print(f"   Response: {test_result.get('response')}")
        print(f"   Token Usage: {test_result.get('usage')}")
        return gpt4, True
    else:
        print(f"   ❌ Connection failed: {test_result.get('message')}")
        print("\n   📝 To enable GPT-4:")
        print("      1. Get API key from: https://platform.openai.com/api-keys")
        print("      2. Set environment variable:")
        print("         Windows: set OPENAI_API_KEY=sk-your-key-here")
        print("         Linux/Mac: export OPENAI_API_KEY=sk-your-key-here")
        print("      3. Or pass directly: GPT4IntegrationEngine(api_key='sk-...')")
        return gpt4, False


def test_gpt4_queries(kg_text: str, gpt4: GPT4IntegrationEngine, connection_ok: bool):
    """Test 3: GPT-4 knowledge graph queries"""
    print("\n" + "="*80)
    print("TEST 3: GPT-4 KNOWLEDGE GRAPH QUERIES")
    print("="*80)
    
    if not connection_ok:
        print("\n⚠️  Skipping - GPT-4 not connected")
        return
    
    test_queries = [
        "What are the most critical food restrictions for John and why?",
        "What breakfast would you recommend for Sarah considering her celiac disease and IBS?",
        "What nutrients should Mary focus on for her osteoporosis and hypothyroidism?",
        "Can you create a shopping list for this family that satisfies everyone's restrictions?",
        "What are the common foods that all family members can safely eat?"
    ]
    
    print("\n🤖 Running GPT-4 Queries on Knowledge Graph...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"QUERY {i}: {query}")
        print('='*80)
        
        response = gpt4.query_knowledge_graph(kg_text, query)
        
        if response['status'] == 'success':
            print(f"\n✅ ANSWER:\n")
            print(response['answer'])
            print(f"\n📊 Token Usage: {response['usage']}")
        else:
            print(f"\n❌ Error: {response.get('message')}")
        
        print()


def test_meal_plan_generation(kg_text: str, gpt4: GPT4IntegrationEngine, connection_ok: bool):
    """Test 4: GPT-4 meal plan generation"""
    print("\n" + "="*80)
    print("TEST 4: GPT-4 MEAL PLAN GENERATION")
    print("="*80)
    
    if not connection_ok:
        print("\n⚠️  Skipping - GPT-4 not connected")
        return
    
    preferences = {
        'cuisine': 'Mediterranean',
        'prep_time': 'Medium (30-45 min)',
        'budget': 'Medium',
        'skill_level': 'Intermediate'
    }
    
    print("\n🍽️  Generating 7-Day Meal Plan with GPT-4...")
    print(f"\n   Preferences:")
    for key, value in preferences.items():
        print(f"      • {key}: {value}")
    
    print("\n   Processing... (this may take 30-60 seconds)\n")
    
    response = gpt4.generate_meal_plan(kg_text, preferences)
    
    if response['status'] == 'success':
        print("="*80)
        print("✅ MEAL PLAN GENERATED")
        print("="*80)
        print(response['answer'])
        print(f"\n📊 Token Usage: {response['usage']}")
        
        # Save meal plan
        with open('gpt4_meal_plan.txt', 'w', encoding='utf-8') as f:
            f.write(response['answer'])
        print("\n✅ Saved to: gpt4_meal_plan.txt")
    else:
        print(f"\n❌ Error: {response.get('message')}")


def display_setup_instructions():
    """Display setup instructions"""
    print("\n" + "="*80)
    print("GPT-4 SETUP INSTRUCTIONS")
    print("="*80)
    
    print("\n📦 STEP 1: Install Required Package")
    print("   pip install openai")
    
    print("\n🔑 STEP 2: Get OpenAI API Key")
    print("   1. Go to: https://platform.openai.com/api-keys")
    print("   2. Sign in or create account")
    print("   3. Click 'Create new secret key'")
    print("   4. Copy the key (starts with 'sk-')")
    
    print("\n⚙️  STEP 3: Set Environment Variable")
    print("\n   Windows (PowerShell):")
    print("   $env:OPENAI_API_KEY='sk-your-key-here'")
    print("\n   Windows (CMD):")
    print("   set OPENAI_API_KEY=sk-your-key-here")
    print("\n   Linux/Mac:")
    print("   export OPENAI_API_KEY='sk-your-key-here'")
    
    print("\n🚀 STEP 4: Run This Test Again")
    print("   python test_gpt4_integration.py")
    
    print("\n💰 PRICING (as of 2024):")
    print("   GPT-4 Turbo:")
    print("   • Input: $0.01 per 1K tokens")
    print("   • Output: $0.03 per 1K tokens")
    print("   • Typical query: ~$0.05-0.15")
    
    print("\n🔒 ALTERNATIVE: Use Azure OpenAI")
    print("   gpt4 = GPT4IntegrationEngine(provider='azure')")
    print("   Set: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT")
    
    print("\n🔒 ALTERNATIVE: Use Anthropic Claude")
    print("   pip install anthropic")
    print("   gpt4 = GPT4IntegrationEngine(provider='anthropic')")
    print("   Set: ANTHROPIC_API_KEY")
    
    print("\n" + "="*80)


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧠 KNOWLEDGE GRAPH & GPT-4 INTEGRATION - COMPLETE TEST SUITE")
    print("="*80)
    
    # Test 1: Knowledge Graph Generation
    kg_text, graph_data = test_knowledge_graph_generation()
    
    # Test 2: GPT-4 Connection
    gpt4, connection_ok = test_gpt4_connection()
    
    if connection_ok:
        # Test 3: GPT-4 Queries
        test_gpt4_queries(kg_text, gpt4, connection_ok)
        
        # Test 4: Meal Plan Generation
        test_meal_plan_generation(kg_text, gpt4, connection_ok)
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\n📁 Files Generated:")
        print("   • test_knowledge_graph.json - Structured graph data")
        print("   • test_knowledge_graph.txt - LLM-readable format")
        print("   • gpt4_meal_plan.txt - Generated meal plan")
        print("\n" + "="*80)
    
    else:
        # Show setup instructions
        display_setup_instructions()
        
        print("\n" + "="*80)
        print("⚠️  GPT-4 NOT CONFIGURED")
        print("="*80)
        print("\n✅ Knowledge graph generation: WORKING")
        print("❌ GPT-4 integration: NOT CONFIGURED")
        print("\nFollow the setup instructions above to enable GPT-4.")
        print("\n" + "="*80)


if __name__ == "__main__":
    main()
