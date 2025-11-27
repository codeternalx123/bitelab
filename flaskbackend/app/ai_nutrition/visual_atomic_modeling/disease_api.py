"""
API Endpoint to List All Available Diseases
===========================================

Provides comprehensive list of all diseases supported by the system
for disease-specific meal planning optimization.
"""

from flask import Flask, jsonify
from comprehensive_disease_db import ComprehensiveDiseaseDatabase

app = Flask(__name__)

# Initialize comprehensive database
print("Loading comprehensive disease database...")
comprehensive_db = ComprehensiveDiseaseDatabase()
print(f"✅ Loaded {comprehensive_db.get_disease_count()} diseases")


@app.route('/v1/diseases/list', methods=['GET'])
def list_all_diseases():
    """Get complete list of all supported diseases"""
    
    all_diseases = []
    categories_summary = {}
    
    for disease_id, disease in comprehensive_db.diseases.items():
        disease_info = {
            "disease_id": disease_id,
            "name": disease.name,
            "category": disease.category,
            "icd10_codes": disease.icd10_codes,
            "nutritional_guidelines_count": len(disease.nutritional_guidelines),
            "food_restrictions_count": len(disease.food_restrictions),
            "recommended_foods_count": len(disease.recommended_foods),
            "meal_timing_important": disease.meal_timing_important,
            "portion_control_critical": disease.portion_control_critical
        }
        
        all_diseases.append(disease_info)
        
        # Count by category
        if disease.category not in categories_summary:
            categories_summary[disease.category] = 0
        categories_summary[disease.category] += 1
    
    # Sort by category then name
    all_diseases.sort(key=lambda x: (x['category'], x['name']))
    
    response = {
        "total_diseases": len(all_diseases),
        "total_categories": len(categories_summary),
        "categories": categories_summary,
        "diseases": all_diseases,
        "system_info": {
            "supports_multi_disease_optimization": True,
            "supports_family_optimization": True,
            "scalable_to": "1000+ diseases",
            "processing_time": "<5ms per optimization",
            "features": [
                "Automated conflict resolution",
                "Priority-based guideline merging",
                "Critical restriction identification",
                "Family-level meal planning",
                "ICD-10 medical coding",
                "Evidence-based nutritional guidelines"
            ]
        }
    }
    
    return jsonify(response), 200


@app.route('/v1/diseases/categories', methods=['GET'])
def list_categories():
    """Get all disease categories"""
    
    categories = {}
    
    for disease in comprehensive_db.diseases.values():
        if disease.category not in categories:
            categories[disease.category] = []
        categories[disease.category].append({
            "disease_id": disease.disease_id,
            "name": disease.name,
            "icd10": disease.icd10_codes[0] if disease.icd10_codes else None
        })
    
    response = {
        "total_categories": len(categories),
        "categories": {
            cat: {
                "count": len(diseases),
                "diseases": diseases
            }
            for cat, diseases in sorted(categories.items())
        }
    }
    
    return jsonify(response), 200


@app.route('/v1/diseases/search', methods=['GET'])
def search_diseases():
    """Search diseases by name or ICD-10 code"""
    from flask import request
    
    query = request.args.get('q', '').lower()
    category = request.args.get('category', None)
    
    if not query:
        return jsonify({"error": "Query parameter 'q' required"}), 400
    
    results = []
    
    for disease in comprehensive_db.diseases.values():
        # Filter by category if specified
        if category and disease.category != category:
            continue
        
        # Search in name, disease_id, or ICD-10 codes
        if (query in disease.name.lower() or 
            query in disease.disease_id.lower() or
            any(query in code.lower() for code in disease.icd10_codes)):
            
            results.append({
                "disease_id": disease.disease_id,
                "name": disease.name,
                "category": disease.category,
                "icd10_codes": disease.icd10_codes
            })
    
    return jsonify({
        "query": query,
        "category_filter": category,
        "results_count": len(results),
        "results": results
    }), 200


@app.route('/v1/diseases/<disease_id>/details', methods=['GET'])
def get_disease_details(disease_id):
    """Get complete details for a specific disease"""
    
    disease = comprehensive_db.get_disease(disease_id)
    
    if not disease:
        return jsonify({"error": f"Disease '{disease_id}' not found"}), 404
    
    response = {
        "disease_id": disease.disease_id,
        "name": disease.name,
        "category": disease.category,
        "icd10_codes": disease.icd10_codes,
        "nutritional_guidelines": [
            {
                "nutrient": g.nutrient,
                "target_min": g.target_min,
                "target_max": g.target_max,
                "unit": g.unit,
                "priority": g.priority,
                "reason": g.reason
            }
            for g in disease.nutritional_guidelines
        ],
        "food_restrictions": [
            {
                "food_item": r.food_item,
                "restriction_type": r.restriction_type,
                "severity": r.severity,
                "reason": r.reason,
                "alternative": r.alternative
            }
            for r in disease.food_restrictions
        ],
        "recommended_foods": disease.recommended_foods,
        "meal_timing_important": disease.meal_timing_important,
        "portion_control_critical": disease.portion_control_critical,
        "hydration_requirements": disease.hydration_requirements,
        "special_considerations": disease.special_considerations
    }
    
    return jsonify(response), 200


@app.route('/health')
def health_check():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "diseases_loaded": comprehensive_db.get_disease_count()
    }), 200


@app.route('/')
def index():
    """API root"""
    return jsonify({
        "message": "Disease Database API",
        "version": "1.0.0",
        "total_diseases": comprehensive_db.get_disease_count(),
        "endpoints": {
            "list_all": "GET /v1/diseases/list",
            "categories": "GET /v1/diseases/categories",
            "search": "GET /v1/diseases/search?q=diabetes",
            "details": "GET /v1/diseases/<disease_id>/details"
        }
    }), 200


if __name__ == '__main__':
    print("\n" + "="*80)
    print("DISEASE DATABASE API SERVER")
    print("="*80)
    print(f"\n✅ Loaded {comprehensive_db.get_disease_count()} disease profiles")
    print("\n🔗 Endpoints:")
    print("   GET  / - API info")
    print("   GET  /health - Health check")
    print("   GET  /v1/diseases/list - All diseases")
    print("   GET  /v1/diseases/categories - By category")
    print("   GET  /v1/diseases/search?q=<query> - Search")
    print("   GET  /v1/diseases/<id>/details - Specific disease")
    print("\n" + "="*80 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )
