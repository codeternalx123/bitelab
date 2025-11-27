# Wellomex API Endpoints & Flutter Mobile Architecture

**Complete API Reference and Mobile App Structure**

Version: 2.0.0  
Last Updated: November 21, 2025

---

## 📡 API Endpoints Reference

### Base URL
- **Development**: `http://localhost:8000`
- **Staging**: `https://staging-api.wellomex.com`
- **Production**: `https://api.wellomex.com`

### API Version: `/api/v1`

---

## 🔐 Authentication & User Management

### **Authentication**
**Base Path**: `/api/v1/auth`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/login` | User login with email/password | ❌ |
| POST | `/register` | Register new user account | ❌ |
| POST | `/logout` | Logout current session | ✅ |
| POST | `/refresh` | Refresh access token | ✅ |
| POST | `/forgot-password` | Request password reset | ❌ |
| POST | `/reset-password` | Reset password with token | ❌ |

### **Supabase Authentication**
**Base Path**: `/api/v1/supabase/auth`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/signup` | Sign up via Supabase | ❌ |
| POST | `/login` | Login via Supabase | ❌ |
| POST | `/logout` | Logout Supabase session | ✅ |
| POST | `/oauth/google` | Google OAuth login | ❌ |
| POST | `/oauth/apple` | Apple OAuth login | ❌ |
| POST | `/oauth/facebook` | Facebook OAuth login | ❌ |

---

## 🍽️ Food Scanning & Analysis

### **Food Scanning**
**Base Path**: `/api/v1/food`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/scan` | Scan food from image (base64) | ✅ |
| POST | `/scan/upload` | Upload food image (multipart) | ✅ |
| GET | `/scan/sensor-requirements` | Get sensor specifications | ❌ |
| POST | `/scan/barcode` | Scan food barcode | ✅ |
| GET | `/scan/history` | Get user's scan history | ✅ |
| DELETE | `/scan/{scan_id}` | Delete scan record | ✅ |

**Example Request - Scan Food**:
```json
POST /api/v1/food/scan
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "nir_spectrum": {
    "wavelengths": [760, 970, 1210, 1510, 1930, 2310],
    "intensities": [0.82, 0.74, 0.91, 0.67, 0.89, 0.73]
  },
  "surface_properties": {
    "glossiness_score": 0.76,
    "specular_reflectance": 0.42
  },
  "metadata": {
    "weight_grams": 150,
    "food_name": "Grilled Salmon"
  },
  "user_profile": {
    "age": 35,
    "weight": 70,
    "medical_conditions": ["diabetes"],
    "health_goals": ["weight loss", "blood sugar control"]
  }
}
```

**Response**:
```json
{
  "scan_id": "scan_abc123",
  "food_name": "Grilled Salmon",
  "confidence": 0.94,
  "nutrients": {
    "calories": 206,
    "protein_g": 31.2,
    "fat_g": 8.4,
    "carbs_g": 0.0,
    "omega3_g": 2.1,
    "iron_mg": 1.2
  },
  "chemometric_analysis": {
    "elements": {
      "mercury_ppm": 0.03,
      "lead_ppm": 0.01,
      "calcium_mg": 15,
      "phosphorus_mg": 280
    },
    "safety_status": "safe",
    "confidence_intervals": {
      "mercury_ppm": [0.02, 0.04]
    }
  },
  "health_score": 87,
  "goal_alignment": {
    "weight_loss": 92,
    "blood_sugar_control": 88
  },
  "warnings": [],
  "recommendations": [
    "Excellent protein source for muscle maintenance during weight loss",
    "Omega-3 content supports heart health and inflammation reduction"
  ]
}
```

---

## 🔬 Chemometrics & Element Analysis

### **Chemometrics**
**Base Path**: `/api/v1/chemometrics`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/predict` | Predict element composition | ✅ |
| POST | `/predict/batch` | Batch element prediction (up to 100) | ✅ |
| POST | `/calibrate` | Calibrate model with ICP-MS data | ✅ |
| GET | `/thresholds` | Get FDA/WHO/EFSA safety limits | ❌ |
| GET | `/models/info` | Get model capabilities | ❌ |
| GET | `/health` | Health check | ❌ |

**Example Request - Element Prediction**:
```json
POST /api/v1/chemometrics/predict
{
  "image_base64": "data:image/jpeg;base64,...",
  "food_type": "seafood",
  "elements_of_interest": ["mercury", "lead", "arsenic", "iron", "calcium"],
  "return_confidence_intervals": true,
  "return_explanations": true
}
```

**Response**:
```json
{
  "prediction_id": "pred_xyz789",
  "timestamp": "2025-11-21T10:30:00Z",
  "elements": {
    "mercury_ppm": 0.035,
    "lead_ppm": 0.008,
    "arsenic_ppm": 0.012,
    "iron_mg_per_100g": 1.2,
    "calcium_mg_per_100g": 15
  },
  "confidence_intervals_95": {
    "mercury_ppm": [0.028, 0.042],
    "lead_ppm": [0.005, 0.011]
  },
  "safety_assessment": {
    "overall_status": "safe",
    "mercury": {
      "status": "safe",
      "detected": 0.035,
      "fda_limit": 1.0,
      "who_limit": 0.5,
      "percentage_of_limit": 3.5
    }
  },
  "model_version": "v2.1.0",
  "accuracy_metrics": {
    "r_squared": 0.89,
    "mae": 0.015
  }
}
```

---

## 🧬 Colorimetry & Spectral Analysis (Camera-Connected)

### **Colorimetry**
**Base Path**: `/api/v1/colorimetry`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/analyze` | Comprehensive colorimetric analysis from camera | ✅ |
| POST | `/analyze-stream` | Real-time camera stream analysis | ✅ |
| POST | `/spectral-signature` | Extract spectral signature from image | ✅ |
| POST | `/calibrate-camera` | Calibrate camera color profile | ✅ |
| GET | `/camera-requirements` | Get camera specifications | ❌ |
| GET | `/health` | Health check | ❌ |

**Example Request - Camera-Based Colorimetric Analysis**:
```json
POST /api/v1/colorimetry/analyze
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "camera_metadata": {
    "device_model": "iPhone 15 Pro",
    "camera_type": "rear_wide",
    "iso": 100,
    "exposure_time_ms": 16.67,
    "aperture": "f/1.78",
    "focal_length_mm": 24,
    "white_balance": "auto",
    "flash_used": false,
    "lens_position": 1.0
  },
  "lighting_conditions": {
    "ambient_light_lux": 450,
    "color_temperature_k": 5500,
    "light_source_type": "daylight"
  },
  "roi_coordinates": {
    "x": 100,
    "y": 150,
    "width": 200,
    "height": 200
  },
  "analysis_mode": "food_composition",
  "calibration_reference": "color_checker_24"
}
```

**Response**:
```json
{
  "analysis_id": "color_abc123",
  "timestamp": "2025-11-21T10:45:00Z",
  "color_analysis": {
    "dominant_colors": [
      {
        "rgb": [220, 180, 140],
        "hex": "#DCB48C",
        "percentage": 45.2,
        "lab": [75.3, 8.2, 25.1],
        "hsv": [30, 36.4, 86.3]
      }
    ],
    "color_histogram": {
      "red_channel": [/* 256 values */],
      "green_channel": [/* 256 values */],
      "blue_channel": [/* 256 values */]
    },
    "texture_features": {
      "glossiness": 0.68,
      "roughness": 0.42,
      "surface_uniformity": 0.85
    }
  },
  "spectral_estimation": {
    "estimated_wavelengths_nm": [400, 450, 500, 550, 600, 650, 700],
    "reflectance_values": [0.15, 0.28, 0.42, 0.58, 0.72, 0.68, 0.45],
    "confidence_score": 0.87
  },
  "chemical_indicators": {
    "chlorophyll_index": 0.23,
    "carotenoid_index": 0.56,
    "anthocyanin_index": 0.12,
    "browning_index": 0.34
  },
  "food_composition_estimates": {
    "moisture_content_percent": 65.2,
    "fat_content_indicator": 0.48,
    "protein_content_indicator": 0.62,
    "freshness_score": 82
  },
  "quality_metrics": {
    "image_quality_score": 0.91,
    "lighting_adequacy": 0.88,
    "focus_quality": 0.94,
    "color_accuracy": 0.86
  },
  "recommendations": [
    "Good lighting conditions detected",
    "Camera calibration accurate",
    "Consider capturing from 10cm closer for better detail"
  ]
}
```

**Real-Time Stream Analysis**:
```json
POST /api/v1/colorimetry/analyze-stream
{
  "stream_id": "stream_xyz789",
  "frame_base64": "data:image/jpeg;base64,...",
  "frame_number": 42,
  "timestamp_ms": 1732123456789,
  "camera_metadata": { /* same as above */ },
  "analysis_type": "real_time_preview"
}
```

**Response (Lightweight for streaming)**:
```json
{
  "frame_number": 42,
  "quick_analysis": {
    "dominant_color_rgb": [220, 180, 140],
    "freshness_score": 82,
    "lighting_quality": "good",
    "focus_quality": "excellent",
    "capture_ready": true
  },
  "guidance": {
    "message": "Hold steady - excellent capture conditions",
    "auto_capture_countdown": 3
  }
}
```

---

## 🔀 Multi-Modal Sensor Fusion (Camera-Connected)

### **Fusion**
**Base Path**: `/api/v1/fusion`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/analyze` | Multi-sensor camera fusion analysis | ✅ |
| POST | `/analyze-live` | Real-time camera + sensor fusion | ✅ |
| POST | `/hybrid-prediction` | Visual + NIR camera prediction | ✅ |
| POST | `/camera-nir-fusion` | RGB camera + NIR sensor fusion | ✅ |
| POST | `/validate-sensors` | Validate connected camera sensors | ✅ |
| GET | `/fusion-methods` | Available fusion algorithms | ❌ |
| GET | `/sensor-requirements` | Required camera/sensor specs | ❌ |
| GET | `/health` | Health check | ❌ |

**Example Request - Multi-Modal Camera Fusion**:
```json
POST /api/v1/fusion/analyze
{
  "rgb_image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "camera_data": {
    "rgb_camera": {
      "device_model": "iPhone 15 Pro",
      "sensor_size_mm": [9.8, 7.3],
      "pixel_size_um": 1.22,
      "iso": 100,
      "exposure_ms": 16.67,
      "white_balance_k": 5500
    },
    "nir_sensor": {
      "wavelengths_nm": [760, 850, 940, 1050, 1210, 1450],
      "intensities": [0.82, 0.76, 0.74, 0.68, 0.91, 0.73],
      "integration_time_ms": 50,
      "sensor_temp_c": 28.5
    },
    "depth_sensor": {
      "distance_mm": 150,
      "confidence": 0.96,
      "point_cloud_available": false
    },
    "lidar_data": {
      "surface_reflectance": 0.42,
      "distance_mm": 148,
      "point_density": "high"
    }
  },
  "environmental_sensors": {
    "ambient_light_lux": 450,
    "color_temperature_k": 5500,
    "temperature_c": 22.5,
    "humidity_percent": 45
  },
  "fusion_config": {
    "fusion_method": "deep_neural_fusion",
    "weight_rgb": 0.4,
    "weight_nir": 0.35,
    "weight_texture": 0.15,
    "weight_depth": 0.1,
    "ensemble_strategy": "weighted_average"
  },
  "analysis_targets": [
    "nutrient_composition",
    "element_detection",
    "freshness_assessment",
    "contamination_detection"
  ]
}
```

**Response**:
```json
{
  "fusion_analysis_id": "fusion_abc123",
  "timestamp": "2025-11-21T11:00:00Z",
  "fusion_results": {
    "food_identification": {
      "primary_match": "Grilled Salmon Fillet",
      "confidence": 0.96,
      "rgb_contribution": 0.92,
      "nir_contribution": 0.94,
      "fusion_boost": 0.04
    },
    "nutrient_estimation": {
      "protein_g_per_100g": 31.2,
      "fat_g_per_100g": 8.4,
      "moisture_percent": 65.3,
      "omega3_g_per_100g": 2.1,
      "confidence": 0.89,
      "method": "rgb_nir_fusion"
    },
    "chemical_composition": {
      "elements": {
        "mercury_ppm": 0.035,
        "iron_mg_per_100g": 1.2,
        "calcium_mg_per_100g": 15
      },
      "detection_method": "nir_spectroscopy_ml",
      "confidence_intervals": {
        "mercury_ppm": [0.028, 0.042]
      }
    },
    "freshness_analysis": {
      "freshness_score": 87,
      "estimated_age_hours": 6,
      "storage_conditions": "refrigerated",
      "indicators": {
        "color_degradation": 0.13,
        "texture_change": 0.08,
        "nir_signature_deviation": 0.11
      }
    },
    "quality_assessment": {
      "overall_quality_score": 92,
      "cooking_doneness": "medium",
      "surface_characteristics": {
        "charring_level": 0.15,
        "moisture_retention": 0.88
      }
    }
  },
  "sensor_fusion_metrics": {
    "fusion_confidence": 0.93,
    "sensor_agreement": 0.91,
    "data_quality": {
      "rgb_quality": 0.94,
      "nir_quality": 0.89,
      "depth_quality": 0.96
    },
    "processing_time_ms": 342
  },
  "camera_feedback": {
    "capture_quality": "excellent",
    "lighting_optimal": true,
    "distance_optimal": true,
    "angle_optimal": true,
    "improvements": []
  },
  "recommendations": [
    "Multi-sensor fusion increased accuracy by 12%",
    "NIR data confirms protein and fat content",
    "Excellent capture conditions - no retake needed"
  ]
}
```

**Real-Time Live Fusion (for camera preview)**:
```json
POST /api/v1/fusion/analyze-live
{
  "stream_id": "live_fusion_xyz789",
  "rgb_frame_base64": "data:image/jpeg;base64,...",
  "nir_readings": {
    "wavelengths_nm": [760, 850, 940],
    "intensities": [0.82, 0.76, 0.74]
  },
  "frame_metadata": {
    "frame_number": 156,
    "timestamp_ms": 1732123456789,
    "camera_distance_mm": 152
  },
  "preview_mode": true
}
```

**Response (Optimized for real-time)**:
```json
{
  "frame_number": 156,
  "quick_fusion_result": {
    "food_detected": true,
    "food_name": "Salmon",
    "confidence": 0.91,
    "freshness_indicator": "good",
    "capture_readiness": {
      "ready_to_capture": true,
      "distance_ok": true,
      "lighting_ok": true,
      "nir_signal_ok": true,
      "stability_ok": true
    }
  },
  "ui_guidance": {
    "message": "Perfect alignment - capturing in 3s",
    "overlay_color": "green",
    "auto_capture_enabled": true,
    "countdown": 3
  },
  "processing_ms": 45
}
```

**Camera-NIR Sensor Validation**:
```json
POST /api/v1/fusion/validate-sensors
{
  "device_info": {
    "device_model": "iPhone 15 Pro",
    "os_version": "iOS 17.1",
    "camera_available": true,
    "nir_sensor_attached": true
  },
  "test_capture": {
    "rgb_test_image": "data:image/jpeg;base64,...",
    "nir_test_reading": [0.5, 0.6, 0.7]
  }
}
```

**Response**:
```json
{
  "validation_status": "passed",
  "camera_validation": {
    "resolution_ok": true,
    "color_accuracy_ok": true,
    "focus_capability_ok": true,
    "recommended_settings": {
      "iso": 100,
      "exposure_time_ms": 16.67,
      "white_balance": "daylight"
    }
  },
  "nir_validation": {
    "sensor_detected": true,
    "wavelength_coverage_ok": true,
    "signal_quality_ok": true,
    "calibration_status": "valid",
    "calibration_expires": "2025-12-21T00:00:00Z"
  },
  "fusion_capability": {
    "supported": true,
    "accuracy_level": "high",
    "expected_confidence": 0.92
  }
}
```

---

## ⚕️ Risk Integration & Safety Analysis

### **Risk Assessment**
**Base Path**: `/api/v1/risk`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/assess` | Personalized risk assessment | ✅ |
| POST | `/alternatives` | Find safer food alternatives | ✅ |
| POST | `/batch-assess` | Batch risk assessment | ✅ |
| GET | `/thresholds/{element}` | Get regulatory limits for element | ❌ |
| POST | `/health-profile` | Create/update health profile | ✅ |
| GET | `/health-profile/{user_id}` | Get user health profile | ✅ |
| POST | `/warning-preferences` | Set warning preferences | ✅ |
| GET | `/supported-conditions` | List supported medical conditions | ❌ |

### **Food Risk Analysis**
**Base Path**: `/api/v1/food-risk`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/analyze` | Complete food safety analysis | ✅ |
| POST | `/batch-analyze` | Analyze multiple foods | ✅ |
| POST | `/compare-foods` | Compare safety of multiple foods | ✅ |
| GET | `/contaminant-limits` | Get contaminant safety limits | ❌ |
| GET | `/health` | Health check | ❌ |

---

## 💬 Conversational AI Assistant

### **Chat**
**Base Path**: `/api/v1/chat`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/session` | Create new conversation session | ✅ |
| POST | `/message` | Send message to assistant | ✅ |
| POST | `/scan-and-ask` | Scan food + ask question | ✅ |
| GET | `/history/{session_id}` | Get conversation history | ✅ |
| POST | `/feedback` | Submit user feedback | ✅ |
| GET | `/sessions` | List user's active sessions | ✅ |
| DELETE | `/session/{session_id}` | Delete conversation session | ✅ |

**Example Request - Create Session**:
```json
POST /api/v1/chat/session
{
  "mode": "general_nutrition",
  "user_profile": {
    "health_conditions": ["type2_diabetes", "hypertension"],
    "medications": ["metformin", "lisinopril"],
    "allergies": ["peanuts"],
    "health_goals": ["weight_loss", "blood_sugar_control"]
  }
}
```

**Example Request - Send Message**:
```json
POST /api/v1/chat/message
{
  "session_id": "session_user123_1732123456",
  "message": "I scanned grilled chicken. Is this good for building muscle?"
}
```

**Response**:
```json
{
  "message_id": "msg_abc123",
  "response": "Yes! Grilled chicken is excellent for muscle gain. It provides 31g of protein per 100g with minimal fat (3.6g). This aligns perfectly with your muscle gain goal...",
  "function_calls": [
    {
      "function": "scan_food",
      "result": {
        "food_name": "Grilled Chicken Breast",
        "protein_g": 31,
        "health_score": 92
      }
    }
  ],
  "suggestions": [
    "Would you like a high-protein meal plan?",
    "Need portion size recommendations?"
  ],
  "timestamp": "2025-11-21T10:35:00Z"
}
```

---

## 👨‍🍳 Recipe Generation & Management

### **Recipes**
**Base Path**: `/api/v1/recipes`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/generate` | Generate personalized recipe | ✅ |
| POST | `/generate-from-pantry` | Generate from available ingredients | ✅ |
| POST | `/adapt-cultural` | Adapt recipe to cuisine | ✅ |
| POST | `/substitute` | Get ingredient substitutions | ✅ |
| GET | `/search` | Search recipes by criteria | ✅ |
| GET | `/{recipe_id}` | Get recipe details | ✅ |
| POST | `/save` | Save recipe to favorites | ✅ |
| GET | `/favorites` | Get saved recipes | ✅ |

### **Family Recipes**
**Base Path**: `/api/v1/family-recipes`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/generate` | Generate family-optimized recipes | ✅ |
| POST | `/meal-plan` | Create weekly family meal plan | ✅ |
| POST | `/analyze-family` | Analyze family nutritional needs | ✅ |

**Example Request - Generate Family Recipe**:
```json
POST /api/v1/family-recipes/generate
{
  "family_profile": {
    "members": [
      {
        "name": "Mom",
        "age": 35,
        "health_goals": ["weight_loss"],
        "medical_conditions": ["prediabetes"],
        "taste_preferences": {
          "likes": ["Mediterranean"],
          "dislikes": ["spicy"]
        }
      },
      {
        "name": "Dad",
        "age": 38,
        "health_goals": ["muscle_gain"],
        "medical_conditions": ["high_cholesterol"]
      },
      {
        "name": "Emma",
        "age": 8,
        "health_goals": ["growth"],
        "allergies": ["peanuts"]
      }
    ]
  },
  "meal_type": "dinner",
  "cuisine_preference": "Mediterranean",
  "max_recipes": 3
}
```

---

## 📅 Meal Planning

### **Meal Plans**
**Base Path**: `/api/v1/meal-plans`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/create` | Create weekly meal plan | ✅ |
| GET | `/{plan_id}` | Get meal plan details | ✅ |
| PUT | `/{plan_id}` | Update meal plan | ✅ |
| DELETE | `/{plan_id}` | Delete meal plan | ✅ |
| GET | `/active` | Get active meal plans | ✅ |
| POST | `/generate-grocery-list` | Generate shopping list | ✅ |

---

## 🛒 Grocery & Shopping

### **Grocery Lists**
**Base Path**: `/api/v1/grocery`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/generate` | Auto-generate from meal plan | ✅ |
| POST | `/optimize-local` | Optimize for local stores | ✅ |
| GET | `/{list_id}` | Get grocery list | ✅ |
| PUT | `/{list_id}` | Update grocery list | ✅ |
| POST | `/{list_id}/check-item` | Mark item as purchased | ✅ |

---

## 🧬 AI & Knowledge Graph

### **AI Integration**
**Base Path**: `/api/v1/ai`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/analyze-food` | Deep learning food analysis | ✅ |
| POST | `/knowledge/query` | Query knowledge graph | ✅ |
| GET | `/knowledge/entities` | List graph entities | ✅ |
| POST | `/knowledge/expand` | Expand knowledge with LLM | ✅ |
| POST | `/train/llm-data` | Contribute training data | ✅ |
| GET | `/knowledge/stats` | Get knowledge graph statistics | ❌ |

---

## 💳 Payments

### **Quantum-Secure Payments**
**Base Path**: `/api/v1/payments`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/create` | Create payment intent | ✅ |
| POST | `/confirm` | Confirm payment | ✅ |
| GET | `/{payment_id}` | Get payment status | ✅ |
| POST | `/refund` | Refund payment | ✅ |
| POST | `/subscriptions/create` | Create subscription | ✅ |
| POST | `/webhook/stripe` | Stripe webhook handler | ❌ |
| POST | `/webhook/paypal` | PayPal webhook handler | ❌ |
| GET | `/methods` | Get available payment methods | ✅ |
| GET | `/health` | Health check | ❌ |

### **M-Pesa Payments (Kenya)**
**Base Path**: `/api/v1/payments`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/mpesa/stkpush` | Initiate M-Pesa payment | ✅ |
| POST | `/mpesa/callback` | M-Pesa callback handler | ❌ |

---

## 📊 Analytics & Reporting

### **Analytics**
**Base Path**: `/api/v1/analytics`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | Get user analytics dashboard | ✅ |
| GET | `/nutrition-trends` | Get nutrition trends over time | ✅ |
| GET | `/health-progress` | Get health goal progress | ✅ |
| GET | `/meal-insights` | Get meal pattern insights | ✅ |

### **Reports**
**Base Path**: `/api/v1/reports`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | List available reports | ✅ |
| POST | `/generate` | Generate custom report | ✅ |
| GET | `/{report_id}` | Download report (PDF/CSV) | ✅ |

---

## 🏥 Health & System

### **Health Check**
**Base Path**: `/api/v1/health`

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/status` | System health status | ❌ |
| GET | `/version` | API version info | ❌ |
| GET | `/metrics` | System metrics (admin) | ✅ |

---

## 📱 Flutter Mobile App Architecture

### **Project Structure**

```
flutter_apk/
├── lib/
│   ├── main.dart                          # App entry point
│   ├── app.dart                           # Material App configuration
│   │
│   ├── core/                              # Core utilities
│   │   ├── config/
│   │   │   ├── environment_config.dart    # Environment settings (dev/staging/prod)
│   │   │   ├── network_config.dart        # API endpoints configuration
│   │   │   ├── logger_config.dart         # Logging configuration
│   │   │   └── cache_config.dart          # Caching configuration
│   │   │
│   │   ├── di/                            # Dependency Injection
│   │   │   ├── core_module.dart           # Core dependencies (Firebase, Hive, etc.)
│   │   │   ├── api_module.dart            # API service dependencies
│   │   │   └── services_module.dart       # Business service dependencies
│   │   │
│   │   ├── network/                       # Network layer
│   │   │   ├── api_service.dart           # Base API client (Dio)
│   │   │   ├── network_manager.dart       # Connectivity monitoring
│   │   │   ├── request_queue_manager.dart # Offline request queuing
│   │   │   ├── rate_limiter.dart          # API rate limiting
│   │   │   └── interceptors/
│   │   │       ├── auth_interceptor.dart  # JWT token injection
│   │   │       ├── cache_interceptor.dart # Response caching
│   │   │       ├── error_interceptor.dart # Error handling
│   │   │       ├── retry_interceptor.dart # Retry logic
│   │   │       └── logging_interceptor.dart # Request/response logging
│   │   │
│   │   ├── storage/                       # Local storage
│   │   │   ├── secure_storage_service.dart # Secure storage (flutter_secure_storage)
│   │   │   └── cache_manager.dart         # Hive cache management
│   │   │
│   │   ├── logging/                       # Logging
│   │   │   └── app_logger.dart            # Centralized logging
│   │   │
│   │   └── utils/                         # Utilities
│   │       ├── encryption_service.dart    # Data encryption
│   │       ├── validators.dart            # Input validation
│   │       └── date_formatter.dart        # Date formatting
│   │
│   ├── data/                              # Data layer
│   │   ├── models/                        # Data models
│   │   │   ├── user_model.dart
│   │   │   ├── food_model.dart
│   │   │   ├── scan_model.dart
│   │   │   ├── recipe_model.dart
│   │   │   ├── meal_plan_model.dart
│   │   │   ├── payment_model.dart
│   │   │   └── error_models.dart
│   │   │
│   │   ├── repositories/                  # Data repositories
│   │   │   ├── auth_repository.dart
│   │   │   ├── food_repository.dart
│   │   │   ├── scan_repository.dart
│   │   │   ├── recipe_repository.dart
│   │   │   ├── meal_plan_repository.dart
│   │   │   └── payment_repository.dart
│   │   │
│   │   └── datasources/                   # Data sources
│   │       ├── local/
│   │       │   ├── auth_local_datasource.dart
│   │       │   ├── food_local_datasource.dart
│   │       │   └── cache_local_datasource.dart
│   │       └── remote/
│   │           ├── auth_remote_datasource.dart
│   │           ├── food_remote_datasource.dart
│   │           ├── scan_remote_datasource.dart
│   │           └── payment_remote_datasource.dart
│   │
│   ├── domain/                            # Business logic layer
│   │   ├── entities/                      # Domain entities
│   │   │   ├── user.dart
│   │   │   ├── food.dart
│   │   │   ├── scan_result.dart
│   │   │   └── recipe.dart
│   │   │
│   │   ├── usecases/                      # Use cases
│   │   │   ├── auth/
│   │   │   │   ├── login_usecase.dart
│   │   │   │   ├── register_usecase.dart
│   │   │   │   └── logout_usecase.dart
│   │   │   ├── food/
│   │   │   │   ├── scan_food_usecase.dart
│   │   │   │   ├── get_scan_history_usecase.dart
│   │   │   │   └── analyze_nutrition_usecase.dart
│   │   │   ├── chat/
│   │   │   │   ├── create_session_usecase.dart
│   │   │   │   └── send_message_usecase.dart
│   │   │   └── recipe/
│   │   │       ├── generate_recipe_usecase.dart
│   │   │       └── save_recipe_usecase.dart
│   │   │
│   │   └── repositories/                  # Repository interfaces
│   │       ├── i_auth_repository.dart
│   │       ├── i_food_repository.dart
│   │       └── i_recipe_repository.dart
│   │
│   ├── presentation/                      # Presentation layer
│   │   ├── screens/                       # App screens
│   │   │   ├── auth/
│   │   │   │   ├── login_screen.dart
│   │   │   │   ├── register_screen.dart
│   │   │   │   └── forgot_password_screen.dart
│   │   │   ├── home/
│   │   │   │   ├── home_screen.dart
│   │   │   │   └── dashboard_screen.dart
│   │   │   ├── scanner/
│   │   │   │   ├── food_scanner_screen.dart
│   │   │   │   ├── camera_screen.dart
│   │   │   │   ├── scan_result_screen.dart
│   │   │   │   └── barcode_scanner_screen.dart
│   │   │   ├── chat/
│   │   │   │   ├── chat_screen.dart
│   │   │   │   └── chat_history_screen.dart
│   │   │   ├── recipes/
│   │   │   │   ├── recipe_list_screen.dart
│   │   │   │   ├── recipe_detail_screen.dart
│   │   │   │   ├── generate_recipe_screen.dart
│   │   │   │   └── family_recipe_screen.dart
│   │   │   ├── meal_plan/
│   │   │   │   ├── meal_plan_screen.dart
│   │   │   │   ├── calendar_view_screen.dart
│   │   │   │   └── grocery_list_screen.dart
│   │   │   ├── profile/
│   │   │   │   ├── profile_screen.dart
│   │   │   │   ├── health_profile_screen.dart
│   │   │   │   ├── settings_screen.dart
│   │   │   │   └── subscription_screen.dart
│   │   │   ├── analytics/
│   │   │   │   ├── analytics_screen.dart
│   │   │   │   ├── nutrition_trends_screen.dart
│   │   │   │   └── health_progress_screen.dart
│   │   │   └── payments/
│   │   │       ├── payments_screen.dart
│   │   │       ├── payment_method_screen.dart
│   │   │       └── subscription_plans_screen.dart
│   │   │
│   │   ├── widgets/                       # Reusable widgets
│   │   │   ├── common/
│   │   │   │   ├── custom_button.dart
│   │   │   │   ├── custom_text_field.dart
│   │   │   │   ├── loading_indicator.dart
│   │   │   │   └── error_widget.dart
│   │   │   ├── scanner/
│   │   │   │   ├── camera_preview_widget.dart
│   │   │   │   ├── scan_overlay_widget.dart
│   │   │   │   └── nutrition_card_widget.dart
│   │   │   ├── recipe/
│   │   │   │   ├── recipe_card_widget.dart
│   │   │   │   ├── ingredient_list_widget.dart
│   │   │   │   └── nutrition_info_widget.dart
│   │   │   └── chat/
│   │   │       ├── message_bubble_widget.dart
│   │   │       ├── typing_indicator_widget.dart
│   │   │       └── suggestion_chips_widget.dart
│   │   │
│   │   ├── providers/                     # State management (Riverpod/Provider)
│   │   │   ├── auth_provider.dart
│   │   │   ├── food_scanner_provider.dart
│   │   │   ├── chat_provider.dart
│   │   │   ├── recipe_provider.dart
│   │   │   ├── meal_plan_provider.dart
│   │   │   └── payment_provider.dart
│   │   │
│   │   └── theme/                         # App theming
│   │       ├── app_theme.dart
│   │       ├── app_colors.dart
│   │       └── app_text_styles.dart
│   │
│   ├── services/                          # Business services
│   │   ├── auth_service.dart              # Authentication service
│   │   ├── scanner_service.dart           # Food scanning service
│   │   ├── camera_service.dart            # Camera management
│   │   ├── colorimetry_service.dart       # Colorimetric analysis service
│   │   ├── sensor_fusion_service.dart     # Multi-modal sensor fusion
│   │   ├── nir_sensor_service.dart        # NIR sensor integration
│   │   ├── camera_calibration_service.dart # Camera calibration
│   │   ├── chat_service.dart              # Chat/AI service
│   │   ├── recipe_service.dart            # Recipe service
│   │   ├── payment_service.dart           # Payment service
│   │   ├── analytics_service.dart         # Analytics tracking
│   │   ├── notification_service.dart      # Push notifications
│   │   └── background_sync_service.dart   # Background data sync
│   │
│   ├── routes/                            # Navigation
│   │   ├── app_router.dart                # Route configuration
│   │   └── route_guards.dart              # Auth guards
│   │
│   └── constants/                         # Constants
│       ├── api_endpoints.dart             # API endpoint constants
│       ├── app_constants.dart             # General constants
│       └── error_messages.dart            # Error message constants
│
├── assets/                                # Assets
│   ├── images/
│   ├── icons/
│   ├── fonts/
│   └── animations/
│
├── test/                                  # Tests
│   ├── unit/
│   ├── widget/
│   └── integration/
│
├── android/                               # Android configuration
├── ios/                                   # iOS configuration
├── web/                                   # Web configuration
├── pubspec.yaml                           # Dependencies
└── README.md                              # App documentation
```

---

## 🏗️ Mobile Architecture Patterns

### **1. Clean Architecture**

The app follows Clean Architecture principles with 3 main layers:

```
┌─────────────────────────────────────────────────────────┐
│              PRESENTATION LAYER                         │
│  (Screens, Widgets, Providers/State Management)         │
│  - UI components                                        │
│  - State management (Riverpod/Provider)                 │
│  - User interaction handling                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              DOMAIN LAYER                               │
│  (Entities, Use Cases, Repository Interfaces)           │
│  - Business logic                                       │
│  - Domain entities                                      │
│  - Use case orchestration                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              DATA LAYER                                 │
│  (Models, Repositories, Data Sources)                   │
│  - API communication                                    │
│  - Local storage                                        │
│  - Data transformation                                  │
└─────────────────────────────────────────────────────────┘
```

### **2. State Management: Riverpod**

```dart
// Example: Food Scanner Provider
final foodScannerProvider = StateNotifierProvider<FoodScannerNotifier, FoodScannerState>((ref) {
  final scanUseCase = ref.read(scanFoodUseCaseProvider);
  return FoodScannerNotifier(scanUseCase);
});

class FoodScannerNotifier extends StateNotifier<FoodScannerState> {
  final ScanFoodUseCase _scanUseCase;
  
  FoodScannerNotifier(this._scanUseCase) : super(FoodScannerInitial());
  
  Future<void> scanFood(File imageFile) async {
    state = FoodScannerLoading();
    
    final result = await _scanUseCase(ScanFoodParams(imageFile: imageFile));
    
    result.fold(
      (failure) => state = FoodScannerError(failure.message),
      (scanResult) => state = FoodScannerSuccess(scanResult),
    );
  }
}
```

### **3. Dependency Injection**

Using `get_it` and `injectable`:

```dart
@module
abstract class ServicesModule {
  @singleton
  AuthService getAuthService(ApiService apiService) {
    return AuthService(apiService: apiService);
  }
  
  @singleton
  ScannerService getScannerService(ApiService apiService) {
    return ScannerService(apiService: apiService);
  }
}
```

### **4. Network Layer**

```dart
class ApiService {
  final Dio _dio;
  
  ApiService(this._dio) {
    _dio.interceptors.addAll([
      AuthInterceptor(),
      CacheInterceptor(),
      ErrorInterceptor(),
      RetryInterceptor(),
      LoggingInterceptor(),
    ]);
  }
  
  Future<Response<T>> get<T>(String path, {Map<String, dynamic>? queryParameters});
  Future<Response<T>> post<T>(String path, {dynamic data});
  Future<Response<T>> put<T>(String path, {dynamic data});
  Future<Response<T>> delete<T>(String path);
}
```

---

## 📲 Key Mobile Features

### **1. Food Scanning with Camera Integration**

**Flow**:
1. User opens camera from scanner tab
2. Camera initializes with optimal settings (ISO, exposure, white balance)
3. **Real-time preview with sensor fusion**:
   - RGB camera feed displayed
   - NIR sensor readings overlay (if available)
   - Colorimetric analysis in background
   - Live guidance (distance, lighting, stability)
4. Auto-capture when conditions optimal OR manual capture
5. Multi-modal fusion processing:
   - RGB image analysis
   - NIR spectral analysis
   - Depth/LiDAR data (if available)
   - Colorimetric feature extraction
6. Show loading indicator during analysis
7. Display comprehensive results:
   - Food identification with confidence
   - Nutrition breakdown (fusion-enhanced)
   - Chemical composition from NIR
   - Freshness score from colorimetry
   - Health score and goal alignment
8. Offer to save, chat about food, or rescan

**Camera Integration Features**:
- **Real-time sensor fusion preview**: Live NIR overlay on RGB feed
- **Smart capture guidance**: Visual indicators for optimal distance/lighting
- **Auto-capture mode**: Automatically captures when all sensors aligned
- **Manual controls**: ISO, exposure, focus override
- **Calibration**: Color checker card support for accuracy
- **Multi-camera support**: Wide, ultra-wide, telephoto switching

**Offline Support**:
- Queue scan requests when offline (RGB + NIR data)
- Sync when connection restored
- Cache recent scans locally (Hive)
- Store camera calibration profiles offline
- Basic on-device inference for immediate feedback

### **2. AI Chat Assistant**

**Flow**:
1. Create session with user profile
2. Send/receive messages with function calling
3. Display rich responses (cards, suggestions)
4. Show loading indicators for function execution
5. Maintain conversation history

**Features**:
- Real-time typing indicators
- Suggestion chips for quick actions
- Image attachment support
- Voice input (speech-to-text)

### **3. Recipe Generation**

**Flow**:
1. Select recipe preferences (cuisine, dietary restrictions)
2. Choose family members if generating family recipe
3. AI generates multiple recipe options
4. Display recipe cards with images
5. View detailed instructions and nutrition
6. Save to favorites or add to meal plan

### **4. Meal Planning**

**Flow**:
1. View calendar with current meal plan
2. Add/edit meals for each day
3. Auto-generate grocery list
4. Optimize shopping list by store
5. Track pantry inventory
6. Get substitution suggestions

### **5. Payments & Subscriptions**

**Integration**:
- Stripe (iOS/Android)
- PayPal
- M-Pesa (Kenya only)
- Apple Pay (iOS)
- Google Pay (Android)

**Plans**:
- Free: 10 scans/month
- Basic ($9.99/month): 100 scans/month
- Premium ($19.99/month): Unlimited scans + family features
- Enterprise: Custom pricing

---

## 🔐 Security Implementation

### **Token Management**

```dart
class AuthInterceptor extends Interceptor {
  final SecureStorageService _secureStorage;
  
  @override
  void onRequest(RequestOptions options, RequestInterceptorHandler handler) async {
    final token = await _secureStorage.getAccessToken();
    
    if (token != null) {
      options.headers['Authorization'] = 'Bearer $token';
    }
    
    handler.next(options);
  }
  
  @override
  void onError(DioError err, ErrorInterceptorHandler handler) async {
    if (err.response?.statusCode == 401) {
      // Refresh token logic
      final refreshed = await _refreshToken();
      
      if (refreshed) {
        // Retry request
        handler.resolve(await _retry(err.requestOptions));
      } else {
        // Logout user
        handler.next(err);
      }
    }
  }
}
```

### **Data Encryption**

```dart
class EncryptionService {
  Future<String> encrypt(String plainText) async {
    // AES-256 encryption
  }
  
  Future<String> decrypt(String cipherText) async {
    // AES-256 decryption
  }
}
```

---

## 📊 Performance Optimizations

### **1. Image Optimization**

- Compress images before upload (80% quality)
- Resize to max 1920x1080
- Use progressive JPEG encoding
- Cache processed images

### **2. Caching Strategy**

```dart
class CacheInterceptor extends Interceptor {
  final CacheManager _cacheManager;
  
  @override
  void onRequest(RequestOptions options, RequestInterceptorHandler handler) async {
    if (options.extra['cache'] == true) {
      final cached = await _cacheManager.get(options.uri.toString());
      
      if (cached != null && !_isExpired(cached)) {
        return handler.resolve(Response(
          requestOptions: options,
          data: cached.data,
          statusCode: 200,
        ));
      }
    }
    
    handler.next(options);
  }
}
```

### **3. Lazy Loading**

- Pagination for lists (20 items per page)
- Infinite scroll with load more
- Image lazy loading with placeholders
- Virtual scrolling for long lists

### **4. Background Sync**

```dart
class BackgroundSyncService {
  Future<void> syncPendingScans() async {
    final pendingScans = await _localDb.getPendingScans();
    
    for (final scan in pendingScans) {
      try {
        await _api.uploadScan(scan);
        await _localDb.markSynced(scan.id);
      } catch (e) {
        // Retry later
      }
    }
  }
}
```

---

## 🧪 Testing Strategy

### **Unit Tests**

```dart
void main() {
  group('ScanFoodUseCase', () {
    late ScanFoodUseCase useCase;
    late MockFoodRepository mockRepository;
    
    setUp(() {
      mockRepository = MockFoodRepository();
      useCase = ScanFoodUseCase(mockRepository);
    });
    
    test('should return ScanResult on success', () async {
      // Arrange
      when(mockRepository.scanFood(any))
          .thenAnswer((_) async => Right(tScanResult));
      
      // Act
      final result = await useCase(ScanFoodParams(imageFile: tImageFile));
      
      // Assert
      expect(result, Right(tScanResult));
      verify(mockRepository.scanFood(tImageFile));
    });
  });
}
```

### **Widget Tests**

```dart
void main() {
  testWidgets('FoodScannerScreen displays result correctly', (tester) async {
    // Build widget
    await tester.pumpWidget(
      MaterialApp(home: FoodScannerScreen()),
    );
    
    // Verify initial state
    expect(find.text('Scan Food'), findsOneWidget);
    
    // Tap scan button
    await tester.tap(find.byIcon(Icons.camera));
    await tester.pump();
    
    // Verify loading state
    expect(find.byType(CircularProgressIndicator), findsOneWidget);
  });
}
```

### **Integration Tests**

```dart
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  
  testWidgets('End-to-end food scanning flow', (tester) async {
    await tester.pumpWidget(MyApp());
    
    // Login
    await tester.enterText(find.byKey(Key('email')), 'test@example.com');
    await tester.enterText(find.byKey(Key('password')), 'password');
    await tester.tap(find.text('Login'));
    await tester.pumpAndSettle();
    
    // Navigate to scanner
    await tester.tap(find.byIcon(Icons.camera));
    await tester.pumpAndSettle();
    
    // Scan food
    await tester.tap(find.byKey(Key('capture_button')));
    await tester.pumpAndSettle();
    
    // Verify result
    expect(find.text('Grilled Salmon'), findsOneWidget);
  });
}
```

---

## 📦 Dependencies (pubspec.yaml)

```yaml
dependencies:
  flutter:
    sdk: flutter
  
  # State Management
  flutter_riverpod: ^2.4.0
  
  # Dependency Injection
  get_it: ^7.6.0
  injectable: ^2.3.0
  
  # Network
  dio: ^5.4.0
  retrofit: ^4.0.0
  
  # Storage
  hive: ^2.2.3
  hive_flutter: ^1.1.0
  flutter_secure_storage: ^9.0.0
  
  # Authentication
  firebase_auth: ^4.15.0
  google_sign_in: ^6.1.5
  sign_in_with_apple: ^5.0.0
  
  # Camera & Image
  camera: ^0.10.5
  image_picker: ^1.0.4
  image: ^4.1.3
  
  # Barcode
  mobile_scanner: ^3.5.2
  
  # Payments
  stripe_sdk: ^9.0.0
  pay: ^1.1.2  # Apple Pay & Google Pay
  
  # UI
  cached_network_image: ^3.3.0
  flutter_svg: ^2.0.9
  lottie: ^2.7.0
  
  # Utils
  intl: ^0.18.1
  uuid: ^4.2.0
  path_provider: ^2.1.1
  
  # Analytics
  firebase_analytics: ^10.7.0
  firebase_crashlytics: ^3.4.0
  
  # Notifications
  firebase_messaging: ^14.7.0
  flutter_local_notifications: ^16.1.0

dev_dependencies:
  flutter_test:
    sdk: flutter
  
  # Code Generation
  build_runner: ^2.4.6
  injectable_generator: ^2.4.0
  retrofit_generator: ^8.0.0
  hive_generator: ^2.0.1
  
  # Testing
  mockito: ^5.4.3
  integration_test:
    sdk: flutter
  
  # Linting
  flutter_lints: ^3.0.0
```

---

## 🚀 Build & Deployment

### **Android**

```bash
# Debug build
flutter build apk --debug

# Release build
flutter build apk --release --split-per-abi

# App Bundle (for Play Store)
flutter build appbundle --release
```

### **iOS**

```bash
# Debug build
flutter build ios --debug

# Release build
flutter build ios --release

# Archive for App Store
flutter build ipa --release
```

### **CI/CD Pipeline (GitHub Actions)**

```yaml
name: Build & Deploy

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: subosito/flutter-action@v2
      - run: flutter pub get
      - run: flutter test
      - run: flutter analyze
  
  build-android:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: subosito/flutter-action@v2
      - run: flutter build apk --release
      - uses: actions/upload-artifact@v3
        with:
          name: android-release
          path: build/app/outputs/flutter-apk/
  
  build-ios:
    needs: test
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - uses: subosito/flutter-action@v2
      - run: flutter build ios --release --no-codesign
```

---

## 📞 Support & Resources

- **API Documentation**: https://api.wellomex.com/api/docs
- **Mobile SDK**: https://github.com/wellomex/flutter-sdk
- **Developer Portal**: https://developers.wellomex.com
- **Status Page**: https://status.wellomex.com

---

**Last Updated**: November 21, 2025  
**Version**: 2.0.0  
**Maintained by**: Wellomex Engineering Team
