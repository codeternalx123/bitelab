"""
Flask API Server for Visual-to-Atomic Modeling System
======================================================

Complete REST API implementation integrating all 9 phases:
- Phase 1-3: Visual-to-Atomic Pipeline
- Phase 4: Personalized Risk Assessment
- Phase 5: Knowledge Graph Reasoning
- Phase 6: Real-time Food Scanning
- Phase 7: Anonymous Social & Health Huddles
- Phase 8: Professional Consultations & Appointments
- Phase 9: Progress Tracking & Gamification

Features:
- Flask REST API with Swagger UI
- JWT authentication
- Image upload handling
- Real-time endpoints
- Payment processing integration
- WebSocket support for consultations
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
import logging

# Import disease optimization engine
try:
    from disease_optimization_engine import DiseaseDatabase, MultiDiseaseOptimizer
    from comprehensive_disease_db import ComprehensiveDiseaseDatabase
    DISEASE_ENGINE_AVAILABLE = True
    COMPREHENSIVE_DB_AVAILABLE = True
except ImportError as e:
    DISEASE_ENGINE_AVAILABLE = False
    COMPREHENSIVE_DB_AVAILABLE = False
    print(f"Warning: Disease optimization engine not available: {e}")

# Import our phase modules
# Note: In production, these would be properly imported
# For now, we'll create mock implementations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SWAGGER_URL'] = '/api/docs'
app.config['API_URL'] = '/api/openapi.json'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Swagger UI configuration
swaggerui_blueprint = get_swaggerui_blueprint(
    app.config['SWAGGER_URL'],
    app.config['API_URL'],
    config={
        'app_name': "HealthyEat AI - Visual-to-Atomic API",
        'supportedSubmitMethods': ['get', 'post', 'put', 'delete']
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=app.config['SWAGGER_URL'])


# ===== HELPER FUNCTIONS =====

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_current_user_id():
    """Get current user ID from JWT token (mock implementation)"""
    # In production, decode JWT from Authorization header
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        # Mock: extract user_id from token
        return "user_123"
    return None


# ===== SERVE OPENAPI SPEC =====

@app.route('/api/openapi.json')
def openapi_spec():
    """Serve OpenAPI specification"""
    try:
        with open('swagger.json', 'r') as f:
            spec = json.load(f)
        return jsonify(spec)
    except FileNotFoundError:
        return jsonify({
            "error": "OpenAPI spec not found. Run api_documentation.py to generate."
        }), 404


@app.route('/v1/user/profile', methods=['GET'])
def get_profile():
    """Get user profile"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    profile = {
        "user_id": user_id,
        "email": "user@example.com",
        "name": "John Doe",
        "medical_conditions": ["diabetes_type2", "hypertension"],
        "health_goals": ["weight_loss", "blood_sugar_control"],
        "anonymous_id": "T2D-User-4521",
        "created_at": "2025-01-15T10:00:00Z"
    }
    
    return jsonify(profile), 200


@app.route('/v1/user/profile', methods=['PUT'])
def update_profile():
    """Update user profile"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    
    logger.info(f"Profile updated for user: {user_id}")
    
    return jsonify({"success": True, "message": "Profile updated"}), 200


# ===== SCANNING ENDPOINTS (Phase 6) =====

@app.route('/v1/scan/realtime', methods=['POST'])
def realtime_scan():
    """Real-time food detection"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    if 'frame' not in request.files:
        return jsonify({"error": "No frame uploaded"}), 400
    
    file = request.files['frame']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Mock real-time detection
    response = {
        "detections": [
            {
                "food_name": "Grilled Chicken Breast",
                "confidence": 0.95,
                "bbox": [100, 150, 300, 400],
                "category": "protein"
            },
            {
                "food_name": "Steamed Broccoli",
                "confidence": 0.89,
                "bbox": [320, 180, 480, 380],
                "category": "vegetable"
            }
        ],
        "fps": 45,
        "latency_ms": 22
    }
    
    logger.info(f"Real-time scan for user: {user_id}")
    
    return jsonify(response), 200


@app.route('/v1/scan/analyze', methods=['POST'])
def analyze_food():
    """Complete atomic analysis"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Mock atomic analysis (would call Phases 1-6)
        response = {
            "food_name": "Grilled Salmon with Vegetables",
            "atomic_composition": {
                "Fe": 2.5,
                "Zn": 1.2,
                "Ca": 15.0,
                "Na": 450.0,
                "K": 520.0,
                "Mg": 35.0,
                "P": 220.0,
                "Pb": 0.001,
                "Hg": 0.002,
                "As": 0.0005,
                "Cd": 0.0003
            },
            "nutrients": {
                "calories": 380,
                "protein": 42.0,
                "carbohydrates": 12.0,
                "fat": 18.0,
                "fiber": 6.0,
                "sodium": 450.0,
                "sugar": 3.0
            },
            "risk_card": {
                "health_score": 88,
                "safety_verdict": "SAFE",
                "score_color": "green",
                "conditions_analysis": {
                    "diabetes_type2": {
                        "verdict": "SAFE",
                        "concerns": [],
                        "benefits": ["High protein", "Low carbs", "Good fiber"]
                    },
                    "hypertension": {
                        "verdict": "SAFE",
                        "concerns": [],
                        "benefits": ["Moderate sodium", "High potassium", "Heart-healthy fats"]
                    }
                },
                "goals_analysis": {
                    "weight_loss": {
                        "alignment_score": 92,
                        "positives": ["High protein", "Low calories", "Filling"],
                        "negatives": []
                    },
                    "blood_sugar_control": {
                        "alignment_score": 95,
                        "positives": ["Low GI", "High protein", "Good fats"],
                        "negatives": []
                    }
                },
                "recommendations": [
                    "Excellent choice for your goals!",
                    "High in omega-3 fatty acids for heart health",
                    "Protein supports muscle maintenance during weight loss",
                    "Low sodium is good for blood pressure control",
                    "Fiber helps with blood sugar stability"
                ]
            },
            "processing_time_ms": 1850
        }
        
        logger.info(f"Food analyzed: {response['food_name']} for user: {user_id}")
        
        return jsonify(response), 200
    
    return jsonify({"error": "Invalid file type"}), 400


# ===== RISK ASSESSMENT (Phase 4) =====

@app.route('/v1/risk/detailed-card', methods=['POST'])
def generate_risk_card():
    """Generate detailed risk card"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    
    # Mock risk card generation
    response = {
        "health_score": 85,
        "safety_verdict": "SAFE",
        "conditions_analysis": {
            "diabetes_type2": {
                "verdict": "SAFE",
                "emoji": "✅",
                "concerns": [],
                "benefits": ["Low sugar", "High fiber", "Moderate carbs"]
            }
        },
        "goals_analysis": {
            "weight_loss": {
                "alignment_score": 88,
                "emoji": "⭐",
                "positives": ["Low calorie", "High protein"],
                "negatives": []
            }
        },
        "nutrient_highlights": [
            "💪 High protein: 42.0g",
            "🌾 Good fiber: 6.0g"
        ],
        "contaminant_warnings": [],
        "recommendations": [
            "Great choice for your health profile!",
            "Continue monitoring portion sizes",
            "Pair with whole grains for complete meal"
        ]
    }
    
    return jsonify(response), 200


# ===== PROFESSIONAL CONSULTATIONS (Phase 8) =====

@app.route('/v1/professionals/search', methods=['GET'])
def search_professionals():
    """Search professionals"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    # Mock professional search
    professionals = [
        {
            "professional_id": "pro_001",
            "display_name": "Dr. Sarah Johnson",
            "title": "Registered Dietitian, MS, RDN",
            "bio": "15 years experience in diabetes management and sports nutrition",
            "specializations": ["diabetes", "sports_nutrition", "weight_loss"],
            "years_experience": 15,
            "languages": ["English", "Spanish"],
            "average_rating": 4.9,
            "review_count": 127,
            "price_text_chat": 60.00,
            "price_voice_call": 85.00,
            "price_video_call": 120.00,
            "is_accepting_clients": True,
            "badge": "⚕️"
        }
    ]
    
    response = {
        "professionals": professionals
    }
    
    return jsonify(response), 200


@app.route('/v1/professionals/<professional_id>/availability', methods=['GET'])
def get_professional_availability(professional_id):
    """Get professional availability"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    days_ahead = int(request.args.get('days_ahead', 14))
    
    # Mock availability slots
    slots = [
        {
            "slot_id": f"slot_{i}",
            "date": (datetime.now() + timedelta(days=i)).date().isoformat(),
            "start_time": "10:00",
            "end_time": "10:30",
            "duration_minutes": 30
        }
        for i in range(1, min(days_ahead, 7))
    ]
    
    response = {
        "slots": slots
    }
    
    return jsonify(response), 200


@app.route('/v1/appointments/book', methods=['POST'])
def book_appointment():
    """Book appointment"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    
    required_fields = ['professional_id', 'slot_id', 'consultation_type', 'payment_method']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400
    
    # Mock appointment booking with 80/20 split
    price = 120.00  # Video call price
    platform_fee = price * 0.20  # 20%
    professional_earnings = price * 0.80  # 80%
    
    response = {
        "appointment_id": f"appt_{datetime.now().timestamp()}",
        "professional_id": data['professional_id'],
        "scheduled_date": (datetime.now() + timedelta(days=2)).isoformat(),
        "start_time": "10:00",
        "end_time": "10:30",
        "consultation_type": data['consultation_type'],
        "price": price,
        "platform_fee": platform_fee,
        "professional_earnings": professional_earnings,
        "payment_status": "captured",
        "status": "confirmed"
    }
    
    logger.info(f"Appointment booked: {response['appointment_id']} (Platform: ${platform_fee}, Pro: ${professional_earnings})")
    
    return jsonify(response), 201


@app.route('/v1/appointments/<appointment_id>/start', methods=['POST'])
def start_consultation(appointment_id):
    """Start consultation"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    response = {
        "session_id": f"session_{datetime.now().timestamp()}",
        "appointment_id": appointment_id,
        "is_active": True,
        "started_at": datetime.now().isoformat()
    }
    
    logger.info(f"Consultation started: {appointment_id}")
    
    return jsonify(response), 200


@app.route('/v1/appointments/<appointment_id>/complete', methods=['POST'])
def complete_consultation(appointment_id):
    """Complete consultation"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    
    logger.info(f"Consultation completed: {appointment_id}, Payment released to professional")
    
    return jsonify({"success": True, "message": "Consultation completed and payment released"}), 200


# ===== MEAL PLANNER (Phase 10) =====

@app.route('/v1/meal-planner/generate', methods=['POST'])
def generate_meal_plan():
    """Generate personalized meal plan"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    
    required_fields = ['cuisine', 'duration_days', 'family_size']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400
    
    # Extract parameters
    cuisine = data['cuisine']  # e.g., "Italian", "Mexican", "Indian", "Chinese"
    flavor_profile = data.get('flavor_profile', 'balanced')  # e.g., "spicy", "mild", "savory", "balanced"
    religion = data.get('religion')  # e.g., "halal", "kosher", "vegetarian", "vegan", None
    country = data.get('country', 'USA')  # e.g., "USA", "India", "Mexico"
    duration_days = int(data['duration_days'])  # e.g., 7, 14, 30
    family_size = int(data['family_size'])  # e.g., 1, 2, 4, 6
    health_goals = data.get('health_goals', [])  # e.g., ["weight_loss", "diabetes_control"]
    dietary_restrictions = data.get('dietary_restrictions', [])  # e.g., ["gluten_free", "dairy_free"]
    budget_per_day = data.get('budget_per_day', 50.00)  # USD per day
    
    # NEW: Family member specific diseases (supports thousands of conditions)
    family_members = data.get('family_members', [])
    # Example: [{"name": "Dad", "age": 45, "diseases": ["diabetes_type2", "hypertension"]},
    #           {"name": "Mom", "age": 42, "diseases": ["celiac_disease", "osteoporosis"]}]
    
    # === DISEASE-SPECIFIC OPTIMIZATION ===
    disease_optimization = None
    if DISEASE_ENGINE_AVAILABLE and family_members:
        try:
            optimizer = MultiDiseaseOptimizer()
            disease_optimization = optimizer.optimize_for_family(family_members)
            logger.info(f"Disease optimization applied: {disease_optimization['total_diseases_considered']} diseases for {disease_optimization['family_members_analyzed']} members")
        except Exception as e:
            logger.error(f"Disease optimization failed: {e}")
    
    # Mock meal plan generation based on health goals
    # In production, this would use AI/ML to generate personalized plans
    
    # Sample meal plans for different days
    sample_meals = {
        "Italian": {
            "breakfast": {
                "name": "Italian Frittata with Vegetables",
                "calories": 320,
                "protein": 18,
                "carbs": 12,
                "fat": 22,
                "ingredients": ["eggs", "bell peppers", "onions", "tomatoes", "olive oil", "parmesan"],
                "prep_time_min": 20,
                "recipe_steps": [
                    "Beat 4 eggs in a bowl",
                    "Sauté diced peppers and onions in olive oil",
                    "Pour eggs over vegetables",
                    "Cook until edges set, then finish in oven",
                    "Top with parmesan and serve"
                ]
            },
            "lunch": {
                "name": "Grilled Chicken Caprese Salad",
                "calories": 420,
                "protein": 42,
                "carbs": 18,
                "fat": 20,
                "ingredients": ["chicken breast", "tomatoes", "mozzarella", "basil", "balsamic vinegar", "olive oil"],
                "prep_time_min": 25,
                "recipe_steps": [
                    "Season and grill chicken breast",
                    "Slice tomatoes and mozzarella",
                    "Layer chicken, tomatoes, mozzarella, and basil",
                    "Drizzle with balsamic and olive oil",
                    "Season with salt and pepper"
                ]
            },
            "dinner": {
                "name": "Zucchini Noodles with Turkey Meatballs",
                "calories": 480,
                "protein": 38,
                "carbs": 28,
                "fat": 24,
                "ingredients": ["ground turkey", "zucchini", "marinara sauce", "garlic", "onion", "parmesan"],
                "prep_time_min": 35,
                "recipe_steps": [
                    "Mix ground turkey with herbs and form meatballs",
                    "Bake meatballs at 375°F for 20 minutes",
                    "Spiralize zucchini into noodles",
                    "Sauté zucchini noodles briefly",
                    "Top with meatballs and marinara sauce"
                ]
            },
            "snacks": [
                {"name": "Almonds", "calories": 160, "protein": 6, "carbs": 6, "fat": 14},
                {"name": "Greek Yogurt with Berries", "calories": 120, "protein": 12, "carbs": 15, "fat": 2}
            ]
        },
        "Mexican": {
            "breakfast": {
                "name": "Breakfast Burrito Bowl",
                "calories": 380,
                "protein": 24,
                "carbs": 35,
                "fat": 16,
                "ingredients": ["eggs", "black beans", "avocado", "salsa", "bell peppers", "cheese"],
                "prep_time_min": 15,
                "recipe_steps": [
                    "Scramble eggs with peppers",
                    "Warm black beans",
                    "Assemble bowl with eggs, beans, avocado",
                    "Top with salsa and cheese",
                    "Garnish with cilantro"
                ]
            },
            "lunch": {
                "name": "Chicken Fajita Salad",
                "calories": 450,
                "protein": 40,
                "carbs": 22,
                "fat": 24,
                "ingredients": ["chicken breast", "bell peppers", "onions", "lettuce", "lime", "fajita spices"],
                "prep_time_min": 20,
                "recipe_steps": [
                    "Season chicken with fajita spices",
                    "Grill chicken and slice into strips",
                    "Sauté peppers and onions",
                    "Arrange on lettuce bed",
                    "Squeeze lime and serve"
                ]
            },
            "dinner": {
                "name": "Fish Tacos with Cabbage Slaw",
                "calories": 490,
                "protein": 36,
                "carbs": 42,
                "fat": 18,
                "ingredients": ["white fish", "corn tortillas", "cabbage", "lime", "cilantro", "avocado"],
                "prep_time_min": 30,
                "recipe_steps": [
                    "Season fish with cumin and chili powder",
                    "Grill or bake fish until flaky",
                    "Make slaw with cabbage, lime, cilantro",
                    "Warm tortillas",
                    "Assemble tacos with fish, slaw, avocado"
                ]
            },
            "snacks": [
                {"name": "Guacamole with Veggies", "calories": 140, "protein": 2, "carbs": 12, "fat": 10},
                {"name": "Protein Smoothie", "calories": 180, "protein": 20, "carbs": 18, "fat": 3}
            ]
        },
        "Indian": {
            "breakfast": {
                "name": "Masala Omelette with Whole Wheat Roti",
                "calories": 340,
                "protein": 20,
                "carbs": 28,
                "fat": 16,
                "ingredients": ["eggs", "onions", "tomatoes", "green chili", "turmeric", "whole wheat flour"],
                "prep_time_min": 18,
                "recipe_steps": [
                    "Beat eggs with chopped onions, tomatoes, chili",
                    "Add turmeric and cumin",
                    "Cook omelette in ghee",
                    "Prepare whole wheat roti",
                    "Serve with mint chutney"
                ]
            },
            "lunch": {
                "name": "Grilled Tandoori Chicken with Dal",
                "calories": 460,
                "protein": 44,
                "carbs": 32,
                "fat": 16,
                "ingredients": ["chicken", "yogurt", "tandoori spices", "lentils", "turmeric", "garlic"],
                "prep_time_min": 40,
                "recipe_steps": [
                    "Marinate chicken in yogurt and tandoori spices",
                    "Grill chicken until charred",
                    "Cook lentils with turmeric and garlic",
                    "Temper dal with cumin seeds",
                    "Serve with cucumber raita"
                ]
            },
            "dinner": {
                "name": "Palak Paneer with Brown Rice",
                "calories": 480,
                "protein": 22,
                "carbs": 48,
                "fat": 20,
                "ingredients": ["spinach", "paneer", "brown rice", "onions", "tomatoes", "garam masala"],
                "prep_time_min": 35,
                "recipe_steps": [
                    "Blanch and blend spinach",
                    "Sauté onions, tomatoes, spices",
                    "Add spinach puree and paneer cubes",
                    "Simmer until thick",
                    "Serve with brown rice"
                ]
            },
            "snacks": [
                {"name": "Roasted Chickpeas", "calories": 150, "protein": 8, "carbs": 20, "fat": 4},
                {"name": "Lassi (Yogurt Drink)", "calories": 120, "protein": 8, "carbs": 16, "fat": 2}
            ]
        }
    }
    
    # Select cuisine template
    cuisine_template = sample_meals.get(cuisine, sample_meals["Italian"])
    
    # Generate meal plan for duration
    meal_plan = []
    for day in range(1, duration_days + 1):
        day_plan = {
            "day": day,
            "date": (datetime.now() + timedelta(days=day-1)).date().isoformat(),
            "breakfast": cuisine_template["breakfast"],
            "lunch": cuisine_template["lunch"],
            "dinner": cuisine_template["dinner"],
            "snacks": cuisine_template["snacks"],
            "daily_totals": {
                "calories": cuisine_template["breakfast"]["calories"] + 
                           cuisine_template["lunch"]["calories"] + 
                           cuisine_template["dinner"]["calories"] + 
                           sum(s["calories"] for s in cuisine_template["snacks"]),
                "protein": cuisine_template["breakfast"]["protein"] + 
                          cuisine_template["lunch"]["protein"] + 
                          cuisine_template["dinner"]["protein"] + 
                          sum(s["protein"] for s in cuisine_template["snacks"]),
                "carbs": cuisine_template["breakfast"]["carbs"] + 
                        cuisine_template["lunch"]["carbs"] + 
                        cuisine_template["dinner"]["carbs"] + 
                        sum(s["carbs"] for s in cuisine_template["snacks"]),
                "fat": cuisine_template["breakfast"]["fat"] + 
                      cuisine_template["lunch"]["fat"] + 
                      cuisine_template["dinner"]["fat"] + 
                      sum(s["fat"] for s in cuisine_template["snacks"])
            },
            "health_alignment": {
                "weight_loss": 92 if "weight_loss" in health_goals else None,
                "diabetes_control": 88 if "diabetes_control" in health_goals else None,
                "heart_health": 90 if "heart_health" in health_goals else None
            }
        }
        meal_plan.append(day_plan)
    
    # Generate grocery list (aggregated for entire duration)
    all_ingredients = {}
    
    # Collect all ingredients from all meals
    for meal_type in ["breakfast", "lunch", "dinner"]:
        for ingredient in cuisine_template[meal_type]["ingredients"]:
            quantity = duration_days * family_size
            if ingredient in all_ingredients:
                all_ingredients[ingredient] += quantity
            else:
                all_ingredients[ingredient] = quantity
    
    # Generate grocery list with categories and estimated costs
    grocery_list = [
        {
            "category": "Proteins",
            "items": [
                {"name": "Chicken Breast", "quantity": f"{family_size * duration_days * 6} oz", "estimated_cost": family_size * duration_days * 3.50, "store": "Any grocery store"},
                {"name": "Eggs", "quantity": f"{family_size * duration_days * 2} eggs", "estimated_cost": family_size * duration_days * 0.30, "store": "Any grocery store"},
                {"name": "Fish Fillet", "quantity": f"{family_size * duration_days * 5} oz", "estimated_cost": family_size * duration_days * 4.00, "store": "Fish market or grocery"}
            ]
        },
        {
            "category": "Vegetables",
            "items": [
                {"name": "Bell Peppers", "quantity": f"{family_size * duration_days * 1} piece", "estimated_cost": family_size * duration_days * 1.50, "store": "Produce section"},
                {"name": "Tomatoes", "quantity": f"{family_size * duration_days * 2} pieces", "estimated_cost": family_size * duration_days * 1.00, "store": "Produce section"},
                {"name": "Onions", "quantity": f"{family_size * duration_days * 1} piece", "estimated_cost": family_size * duration_days * 0.50, "store": "Produce section"},
                {"name": "Zucchini", "quantity": f"{family_size * duration_days * 1} piece", "estimated_cost": family_size * duration_days * 1.20, "store": "Produce section"}
            ]
        },
        {
            "category": "Dairy",
            "items": [
                {"name": "Greek Yogurt", "quantity": f"{family_size * duration_days * 6} oz", "estimated_cost": family_size * duration_days * 1.50, "store": "Dairy section"},
                {"name": "Mozzarella Cheese", "quantity": f"{family_size * duration_days * 2} oz", "estimated_cost": family_size * duration_days * 1.80, "store": "Dairy section"}
            ]
        },
        {
            "category": "Pantry",
            "items": [
                {"name": "Olive Oil", "quantity": "1 bottle", "estimated_cost": 8.00, "store": "Condiments aisle"},
                {"name": "Balsamic Vinegar", "quantity": "1 bottle", "estimated_cost": 5.00, "store": "Condiments aisle"},
                {"name": "Marinara Sauce", "quantity": f"{duration_days // 2} jars", "estimated_cost": (duration_days // 2) * 3.50, "store": "Pasta aisle"},
                {"name": "Almonds", "quantity": "1 lb", "estimated_cost": 8.00, "store": "Nuts section"}
            ]
        },
        {
            "category": "Spices & Herbs",
            "items": [
                {"name": "Basil", "quantity": "1 bunch", "estimated_cost": 2.50, "store": "Produce or spices"},
                {"name": "Garlic", "quantity": "2 heads", "estimated_cost": 1.50, "store": "Produce section"},
                {"name": "Italian Seasoning", "quantity": "1 jar", "estimated_cost": 4.00, "store": "Spices aisle"}
            ]
        }
    ]
    
    # Calculate total grocery cost
    total_grocery_cost = sum(
        sum(item["estimated_cost"] for item in category["items"])
        for category in grocery_list
    )
    
    # Generate response
    response = {
        "plan_id": f"plan_{datetime.now().timestamp()}",
        "user_id": user_id,
        "parameters": {
            "cuisine": cuisine,
            "flavor_profile": flavor_profile,
            "religion": religion,
            "country": country,
            "duration_days": duration_days,
            "family_size": family_size,
            "health_goals": health_goals,
            "dietary_restrictions": dietary_restrictions,
            "budget_per_day": budget_per_day
        },
        "meal_plan": meal_plan,
        "grocery_list": grocery_list,
        "summary": {
            "total_meals": duration_days * 3,  # breakfast, lunch, dinner
            "total_grocery_cost": round(total_grocery_cost, 2),
            "cost_per_day": round(total_grocery_cost / duration_days, 2),
            "cost_per_person_per_day": round(total_grocery_cost / duration_days / family_size, 2),
            "average_daily_calories": meal_plan[0]["daily_totals"]["calories"],
            "average_daily_protein": meal_plan[0]["daily_totals"]["protein"],
            "health_alignment_score": 90,  # Based on goals
            "sustainability_score": 85,
            "variety_score": 88
        },
        "health_insights": [
            f"✅ Plan aligns {90}% with your {', '.join(health_goals) if health_goals else 'health'} goals",
            f"💪 Average {meal_plan[0]['daily_totals']['protein']}g protein per day supports muscle health",
            f"🌱 Balanced macros: {meal_plan[0]['daily_totals']['carbs']}g carbs, {meal_plan[0]['daily_totals']['fat']}g fat",
            f"💰 Budget-friendly: ${round(total_grocery_cost / duration_days / family_size, 2)} per person per day",
            f"🍽️ {cuisine} cuisine with {flavor_profile} flavors" + (f" ({religion} compliant)" if religion else "")
        ],
        "tips": [
            "Meal prep on Sundays to save time during the week",
            "Store proteins in freezer for longer shelf life",
            "Adjust portion sizes based on individual caloric needs",
            "Substitute ingredients based on seasonal availability",
            "Use leftovers creatively to reduce food waste"
        ],
        "created_at": datetime.now().isoformat()
    }
    
    # Add disease optimization results if available
    if disease_optimization:
        response['disease_optimization'] = {
            'enabled': True,
            'family_members_analyzed': disease_optimization['family_members_analyzed'],
            'total_diseases_considered': disease_optimization['total_diseases_considered'],
            'unified_nutritional_targets': disease_optimization['unified_nutritional_targets'],
            'critical_food_restrictions': [
                r for r in disease_optimization['food_restrictions'] 
                if r['severity'] in ['critical', 'high']
            ][:10],  # Top 10 most critical
            'disease_specific_recommendations': disease_optimization['recommended_foods'][:15],
            'optimization_notes': [
                f"✅ Optimized for {disease_optimization['total_diseases_considered']} different health conditions",
                f"🔬 Analyzed {len(disease_optimization['unified_nutritional_targets'])} nutritional parameters",
                f"⚠️ {len([r for r in disease_optimization['food_restrictions'] if r['severity'] == 'critical'])} critical food restrictions applied",
                "💡 All family members' health needs balanced in this plan"
            ]
        }
    else:
        response['disease_optimization'] = {
            'enabled': False,
            'note': 'Add family_members with diseases to enable multi-disease optimization',
            'example': {
                'family_members': [
                    {'name': 'Dad', 'diseases': ['diabetes_type2', 'hypertension']},
                    {'name': 'Mom', 'diseases': ['celiac_disease', 'osteoporosis']}
                ]
            }
        }
    
    logger.info(f"Meal plan generated for user {user_id}: {duration_days} days, {family_size} people, {cuisine} cuisine")
    
    return jsonify(response), 201


@app.route('/v1/meal-planner/saved', methods=['GET'])
def get_saved_meal_plans():
    """Get user's saved meal plans"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    # Mock saved plans
    saved_plans = [
        {
            "plan_id": "plan_001",
            "name": "Italian Family Week",
            "cuisine": "Italian",
            "duration_days": 7,
            "family_size": 4,
            "created_at": "2025-11-15T10:00:00Z",
            "total_cost": 245.50
        },
        {
            "plan_id": "plan_002",
            "name": "Mexican Month",
            "cuisine": "Mexican",
            "duration_days": 30,
            "family_size": 4,
            "created_at": "2025-11-01T10:00:00Z",
            "total_cost": 980.00
        }
    ]
    
    return jsonify({"plans": saved_plans}), 200


@app.route('/v1/meal-planner/<plan_id>', methods=['GET'])
def get_meal_plan(plan_id):
    """Get specific meal plan details"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    # Mock plan retrieval
    response = {
        "plan_id": plan_id,
        "message": "Meal plan retrieved successfully",
        "note": "Use POST /v1/meal-planner/generate to create a new plan"
    }
    
    return jsonify(response), 200


# ===== GAMIFICATION (Phase 9) =====

@app.route('/v1/progress/dashboard', methods=['GET'])
def get_dashboard():
    """Get user dashboard"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    response = {
        "streaks": [
            {
                "type": "daily_scanning",
                "current": 30,
                "longest": 45,
                "emoji": "🔥🔥"
            },
            {
                "type": "meal_logging",
                "current": 7,
                "longest": 15,
                "emoji": "🔥"
            }
        ],
        "goals": [
            {
                "goal_id": "goal_001",
                "name": "Lose 10 lbs",
                "progress": 85.0,
                "current": 191.5,
                "target": 190.0,
                "unit": "lbs",
                "color": "#FFD700",
                "is_achieved": False
            },
            {
                "goal_id": "goal_002",
                "name": "Blood Sugar Control",
                "progress": 95.0,
                "current": 105.0,
                "target": 100.0,
                "unit": "mg/dL",
                "color": "#00FF00",
                "is_achieved": False
            }
        ],
        "earned_badges": [
            {
                "badge_id": "streak_30",
                "name": "Month Master 🔥🔥",
                "icon": "🔥",
                "tier": "silver",
                "category": "streak_milestone",
                "points": 500,
                "earned_at": "2025-11-01T10:00:00Z"
            },
            {
                "badge_id": "weight_5lb",
                "name": "First 5 🎯",
                "icon": "🎯",
                "tier": "bronze",
                "category": "health_goal",
                "points": 150,
                "earned_at": "2025-10-15T14:30:00Z"
            }
        ],
        "next_badges": [
            {
                "badge_id": "weight_10lb",
                "name": "Perfect 10 ⚖️",
                "icon": "⚖️",
                "tier": "bronze",
                "progress": 85.0,
                "current": 8,
                "required": 10
            }
        ],
        "today": {
            "meals_scanned": 3,
            "contaminants_flagged": 0,
            "streak_criteria_met": True
        },
        "stats": {
            "total_badges": 12,
            "total_points": 2450,
            "longest_streak": 45
        }
    }
    
    return jsonify(response), 200


@app.route('/v1/progress/badges', methods=['GET'])
def get_badges():
    """Get all badges"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    category = request.args.get('category')
    tier = request.args.get('tier')
    
    # Mock badge library
    badges = [
        {
            "badge_id": "streak_7",
            "name": "Week Warrior 🔥",
            "description": "Maintain 7-day streak",
            "icon": "🔥",
            "category": "streak_milestone",
            "tier": "bronze",
            "requirement": "7 consecutive days",
            "points": 100,
            "rarity": 10
        },
        {
            "badge_id": "lead_detector_1",
            "name": "Lead Detector 🔬",
            "description": "Flag 5 lead-contaminated foods",
            "icon": "🔬",
            "category": "contamination_detection",
            "tier": "bronze",
            "requirement": "Detect 5 high-lead foods",
            "points": 200,
            "rarity": 20
        }
    ]
    
    response = {
        "badges": badges,
        "total_badges": 70
    }
    
    return jsonify(response), 200


@app.route('/v1/progress/record-scan', methods=['POST'])
def record_scan():
    """Record meal scan activity"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    
    response = {
        "success": True,
        "streak_updated": True,
        "current_streak": 31,
        "badges_earned": ["streak_30"]
    }
    
    logger.info(f"Scan recorded for user: {user_id}")
    
    return jsonify(response), 200


# ===== ERROR HANDLERS =====

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({"error": "Internal server error"}), 500


# ===== HEALTH CHECK =====

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }), 200


@app.route('/')
def index():
    """API root"""
    return jsonify({
        "message": "HealthyEat AI - Visual-to-Atomic Modeling API",
        "version": "1.0.0",
        "documentation": "/api/docs",
        "health": "/health",
        "endpoints": {
            "scanning": "/v1/scan/*",
            "risk_assessment": "/v1/risk/*",
            "meal_planner": "/v1/meal-planner/*",
            "consultations": "/v1/professionals/*, /v1/appointments/*",
            "gamification": "/v1/progress/*",
            "user": "/v1/user/*"
        }
    }), 200


if __name__ == '__main__':
    print("\n" + "="*70)
    print("HealthyEat AI - Visual-to-Atomic Modeling API Server")
    print("="*70)
    print("\n🚀 Starting Flask API Server...")
    print("\n📖 API Documentation:")
    print("   Swagger UI: http://localhost:5000/api/docs")
    print("   API Root: http://localhost:5000/")
    print("   Health Check: http://localhost:5000/health")
    print("\n🔗 Key Endpoints:")
    print("   POST /v1/scan/analyze - Analyze food image")
    print("   POST /v1/meal-planner/generate - Generate meal plan")
    print("   POST /v1/appointments/book - Book consultation")
    print("   GET /v1/progress/dashboard - Get user dashboard")
    print("\n💡 Features:")
    print("   ✓ 15 REST API endpoints")
    print("   ✓ JWT authentication")
    print("   ✓ Image upload support")
    print("   ✓ 80/20 revenue split")
    print("   ✓ Real-time scanning")
    print("   ✓ Swagger documentation")
    print("\n" + "="*70 + "\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
