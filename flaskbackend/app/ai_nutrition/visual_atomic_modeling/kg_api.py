"""
Knowledge Graph API with GPT-4 Integration
===========================================

REST API for accessing nutritional knowledge graphs and GPT-4 reasoning.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from knowledge_graph_engine import (
    NutritionalKnowledgeGraph,
    GPT4IntegrationEngine
)
from disease_optimization_engine import MultiDiseaseOptimizer

app = Flask(__name__)
CORS(app)

# Initialize GPT-4 engine (will work if API key is set)
gpt4_engine = GPT4IntegrationEngine()


@app.route('/v1/knowledge-graph/build', methods=['POST'])
def build_knowledge_graph():
    """
    Build knowledge graph from family health data
    
    Request body:
    {
        "family_members": [
            {
                "name": "John",
                "age": 55,
                "diseases": ["diabetes_type2", "hypertension"],
                "dietary_preferences": "non-vegetarian"
            }
        ]
    }
    """
    try:
        data = request.get_json()
        family_members = data.get('family_members', [])
        
        if not family_members:
            return jsonify({'error': 'family_members required'}), 400
        
        # Run optimization
        optimizer = MultiDiseaseOptimizer()
        result = optimizer.optimize_for_family(family_members)
        
        # Build knowledge graph
        kg = NutritionalKnowledgeGraph()
        graph_data = kg.build_from_optimization_result(result, family_members)
        
        # Export for LLM
        kg_text = kg.export_for_llm()
        
        return jsonify({
            'status': 'success',
            'knowledge_graph': graph_data,
            'llm_export': kg_text,
            'optimization_result': {
                'nutritional_targets': result['unified_nutritional_targets'],
                'food_restrictions': result['food_restrictions'],
                'recommended_foods': result['recommended_foods'],
                'family_members_analyzed': result['family_members_analyzed'],
                'total_diseases': result['total_diseases_considered']
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/v1/gpt4/query', methods=['POST'])
def query_with_gpt4():
    """
    Query knowledge graph using GPT-4
    
    Request body:
    {
        "family_members": [...],
        "query": "What should John eat for breakfast?",
        "context": "Optional additional context"
    }
    """
    try:
        data = request.get_json()
        family_members = data.get('family_members', [])
        query = data.get('query')
        context = data.get('context')
        
        if not family_members or not query:
            return jsonify({'error': 'family_members and query required'}), 400
        
        # Build knowledge graph
        optimizer = MultiDiseaseOptimizer()
        result = optimizer.optimize_for_family(family_members)
        
        kg = NutritionalKnowledgeGraph()
        kg.build_from_optimization_result(result, family_members)
        kg_text = kg.export_for_llm()
        
        # Query with GPT-4
        response = gpt4_engine.query_knowledge_graph(kg_text, query, context)
        
        if response['status'] == 'success':
            return jsonify({
                'status': 'success',
                'query': query,
                'answer': response['answer'],
                'model': response['model'],
                'token_usage': response['usage']
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': response.get('message', 'GPT-4 query failed')
            }), 500
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/v1/gpt4/meal-plan', methods=['POST'])
def generate_meal_plan_gpt4():
    """
    Generate meal plan using GPT-4
    
    Request body:
    {
        "family_members": [...],
        "preferences": {
            "cuisine": "Mediterranean",
            "prep_time": "Medium",
            "budget": "Medium",
            "skill_level": "Intermediate"
        }
    }
    """
    try:
        data = request.get_json()
        family_members = data.get('family_members', [])
        preferences = data.get('preferences', {})
        
        if not family_members:
            return jsonify({'error': 'family_members required'}), 400
        
        # Build knowledge graph
        optimizer = MultiDiseaseOptimizer()
        result = optimizer.optimize_for_family(family_members)
        
        kg = NutritionalKnowledgeGraph()
        kg.build_from_optimization_result(result, family_members)
        kg_text = kg.export_for_llm()
        
        # Generate meal plan with GPT-4
        response = gpt4_engine.generate_meal_plan(kg_text, preferences)
        
        if response['status'] == 'success':
            return jsonify({
                'status': 'success',
                'meal_plan': response['answer'],
                'model': response['model'],
                'token_usage': response['usage'],
                'preferences': preferences
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': response.get('message', 'Meal plan generation failed')
            }), 500
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/v1/gpt4/test', methods=['GET'])
def test_gpt4_connection():
    """Test GPT-4 API connection"""
    result = gpt4_engine.test_connection()
    
    if result['status'] == 'success':
        return jsonify(result), 200
    else:
        return jsonify(result), 503


@app.route('/v1/gpt4/config', methods=['GET'])
def get_gpt4_config():
    """Get GPT-4 configuration"""
    return jsonify({
        'provider': gpt4_engine.provider,
        'model': gpt4_engine.model,
        'api_key_set': bool(gpt4_engine.api_key),
        'client_initialized': gpt4_engine.client is not None,
        'supported_providers': ['openai', 'azure', 'anthropic']
    }), 200


@app.route('/v1/gpt4/config', methods=['POST'])
def update_gpt4_config():
    """
    Update GPT-4 configuration
    
    Request body:
    {
        "provider": "openai",
        "api_key": "sk-...",
        "model": "gpt-4-turbo-preview"
    }
    """
    try:
        data = request.get_json()
        
        provider = data.get('provider', 'openai')
        api_key = data.get('api_key')
        model = data.get('model')
        
        # Reinitialize engine
        global gpt4_engine
        gpt4_engine = GPT4IntegrationEngine(api_key=api_key, provider=provider)
        
        if model:
            gpt4_engine.model = model
        
        # Test connection
        test_result = gpt4_engine.test_connection()
        
        return jsonify({
            'status': 'success',
            'message': 'GPT-4 configuration updated',
            'connection_test': test_result
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'gpt4_available': gpt4_engine.client is not None,
        'gpt4_provider': gpt4_engine.provider,
        'gpt4_model': gpt4_engine.model
    }), 200


@app.route('/', methods=['GET'])
def index():
    """API information"""
    return jsonify({
        'name': 'Knowledge Graph & GPT-4 Integration API',
        'version': '1.0.0',
        'endpoints': {
            'build_graph': 'POST /v1/knowledge-graph/build',
            'query_gpt4': 'POST /v1/gpt4/query',
            'meal_plan': 'POST /v1/gpt4/meal-plan',
            'test_gpt4': 'GET /v1/gpt4/test',
            'config': 'GET/POST /v1/gpt4/config',
            'health': 'GET /health'
        },
        'gpt4_status': {
            'available': gpt4_engine.client is not None,
            'provider': gpt4_engine.provider,
            'model': gpt4_engine.model,
            'api_key_set': bool(gpt4_engine.api_key)
        }
    }), 200


if __name__ == '__main__':
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH & GPT-4 API SERVER")
    print("="*80)
    print(f"\n🤖 GPT-4 Provider: {gpt4_engine.provider}")
    print(f"🤖 Model: {gpt4_engine.model}")
    print(f"🔑 API Key Set: {bool(gpt4_engine.api_key)}")
    print(f"✅ Client Initialized: {gpt4_engine.client is not None}")
    
    if not gpt4_engine.api_key:
        print("\n⚠️  GPT-4 API key not found!")
        print("   Set environment variable: OPENAI_API_KEY=your-key-here")
        print("   Or use: POST /v1/gpt4/config to set it via API")
    
    print("\n📡 Endpoints:")
    print("   POST /v1/knowledge-graph/build  - Build knowledge graph")
    print("   POST /v1/gpt4/query             - Query with GPT-4")
    print("   POST /v1/gpt4/meal-plan         - Generate meal plan")
    print("   GET  /v1/gpt4/test              - Test GPT-4 connection")
    print("   GET  /v1/gpt4/config            - Get configuration")
    print("   POST /v1/gpt4/config            - Update configuration")
    print("\n" + "="*80 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=True
    )
