# Risk Integration API - OpenAPI/Swagger Documentation

## Overview

The **Risk Integration API** provides comprehensive food safety assessment by connecting chemometric element detection to personalized health risk evaluation. This API enables mobile apps, web platforms, and clinical systems to deliver actionable food safety insights based on user health profiles.

## Base URL

```
https://api.wellomex.com/api/v1/risk-integration
```

## Authentication

All endpoints require Bearer token authentication:

```http
Authorization: Bearer <your_access_token>
```

Obtain tokens via `/api/v1/auth/login` endpoint.

## Rate Limits

- **Free Tier**: 100 requests/hour
- **Pro Tier**: 1,000 requests/hour  
- **Enterprise**: Unlimited

---

## Endpoints

### 1. POST /assess - Assess Food Safety Risk

Perform comprehensive food safety risk assessment based on element predictions and user health profile.

**Request:**

```http
POST /api/v1/risk-integration/assess
Content-Type: application/json
Authorization: Bearer <token>

{
  "predictions": [
    {
      "element": "Pb",
      "concentration": 0.45,
      "uncertainty": 0.10,
      "confidence": 0.92,
      "unit": "ppm",
      "measurement_method": "visual_chemometrics"
    },
    {
      "element": "K",
      "concentration": 450.0,
      "confidence": 0.88,
      "unit": "mg/100g",
      "measurement_method": "visual_chemometrics"
    }
  ],
  "user_profile": {
    "user_id": "user_12345",
    "conditions": [
      {
        "snomed_code": "77386006",
        "name": "Pregnancy",
        "severity": "active"
      },
      {
        "snomed_code": "431857002",
        "name": "CKD Stage 4",
        "severity": "severe"
      }
    ],
    "age": 32,
    "weight_kg": 68.0
  },
  "food_item": "Spinach",
  "serving_size": 100.0
}
```

**Response (200 OK):**

```json
{
  "overall_risk_level": "CRITICAL",
  "confidence": 0.92,
  "safety_failures": [
    {
      "element": "Pb",
      "measured_value": 0.45,
      "threshold": 0.1,
      "exceeds_limit": true,
      "exceedance_percentage": 350.0,
      "regulatory_authority": "FDA",
      "severity": "CRITICAL"
    }
  ],
  "nutrient_warnings": [
    {
      "element": "K",
      "measured_value": 450.0,
      "daily_target": 2000.0,
      "contribution_percentage": 22.5,
      "is_restricted": true,
      "recommendation": "CAUTION: This serving provides 22.5% of restricted daily limit"
    }
  ],
  "nutrient_benefits": [
    {
      "element": "Fe",
      "measured_value": 3.5,
      "contribution_percentage": 13.0,
      "is_beneficial": true
    }
  ],
  "uncertainty_flags": []
}
```

**Field Descriptions:**

- `overall_risk_level`: Overall safety determination (`CRITICAL`, `HIGH`, `MODERATE`, `LOW`, `SAFE`)
- `confidence`: Prediction confidence (0.0-1.0)
- `safety_failures`: Toxic elements exceeding regulatory limits
- `nutrient_warnings`: Restricted nutrients for user's conditions
- `nutrient_benefits`: Beneficial nutrients detected
- `uncertainty_flags`: Warnings about prediction uncertainty

**Status Codes:**

- `200 OK` - Successful assessment
- `400 Bad Request` - Invalid input data
- `401 Unauthorized` - Missing/invalid authentication
- `500 Internal Server Error` - Assessment processing error

---

### 2. POST /warnings - Generate Personalized Warnings

Generate user-friendly warnings in three presentation modes: consumer, clinical, or regulatory.

**Request:**

```http
POST /api/v1/risk-integration/warnings
Content-Type: application/json
Authorization: Bearer <token>

{
  "predictions": [...],
  "user_profile": {...},
  "food_item": "Spinach",
  "serving_size": 100.0,
  "message_mode": "consumer",
  "batch_id": "SPX-2024-001"
}
```

**Message Modes:**

- `consumer` - Simple, actionable language for end-users
- `clinical` - Medical terminology for healthcare professionals
- `regulatory` - Formal compliance reports for authorities

**Response (200 OK) - Consumer Mode:**

```json
{
  "banner": {
    "icon": "⛔",
    "message": "DO NOT CONSUME - Unsafe for Pregnancy",
    "risk_level": "CRITICAL"
  },
  "food_item": "Spinach",
  "serving_size": "100g",
  "timestamp": "2024-11-20T10:30:00Z",
  "primary_concerns": [
    {
      "icon": "⛔",
      "color": "#DC2626",
      "element": "Lead",
      "message": "Lead levels are dangerously high (4.5x safe limit for pregnancy). Lead can harm your baby's brain development.",
      "action": "DO NOT CONSUME - Immediate health threat",
      "confidence": "92%"
    }
  ],
  "secondary_concerns": [
    {
      "icon": "⚡",
      "element": "Potassium",
      "message": "High potassium content (22.5% of your daily CKD limit in one serving). This could be dangerous for your kidneys.",
      "action": "Monitor intake - 22.5% of daily limit in this serving"
    }
  ],
  "benefits": [
    {
      "icon": "✓",
      "element": "Iron",
      "message": "Good iron source - provides 13% of your daily pregnancy needs!"
    }
  ],
  "alternatives": [
    "Try kale instead (similar iron, 90% less lead)",
    "Swiss chard (good iron source, lower potassium for CKD)",
    "Beet greens (pregnancy-safe, moderate potassium)"
  ],
  "next_steps": [
    "⛔ DO NOT CONSUME this spinach sample under any circumstances",
    "⚠️ High lead levels can cause serious health problems",
    "👨‍⚕️ Contact your OB-GYN about potential lead exposure",
    "🔬 Consider prenatal lead screening blood test"
  ],
  "confidence_note": "High confidence predictions (>90%). Results are reliable.",
  "see_doctor": true
}
```

**Response (200 OK) - Clinical Mode:**

```json
{
  "report_type": "Clinical Assessment",
  "overall_risk": "CRITICAL",
  "food_item": "Spinach",
  "serving_size": "100g",
  "assessment_time": "2024-11-20T10:30:00Z",
  "patient_conditions": ["Pregnancy", "CKD Stage 4"],
  "critical_findings": [
    {
      "element": "Pb",
      "element_name": "Lead",
      "risk_level": "CRITICAL",
      "measured_value": "0.450 mg/kg",
      "threshold": "0.100 mg/kg",
      "exceedance": "350.0%",
      "message": "Pb concentration: 0.450 mg/kg (350.0% above FDA Defect Action Level threshold 0.100 mg/kg). Patient conditions: Pregnancy, CKD Stage 4.",
      "health_risk": "Can cross placenta and harm fetal brain development; Linked to reduced birth weight and premature birth",
      "recommendation": "Recommend complete avoidance. Consider prenatal lead screening (BLL). Monitor for signs of lead toxicity. Counsel on alternative vegetables. Reference: ACOG Committee Opinion 533.",
      "citation": "FDA",
      "confidence": "92.0%",
      "patient_conditions": "Pregnancy, CKD Stage 4"
    }
  ],
  "additional_findings": [...],
  "nutritional_benefits": [...],
  "clinical_recommendations": [...],
  "medical_consultation_required": true,
  "confidence_note": "High confidence predictions (>90%). Results are reliable."
}
```

**Response (200 OK) - Regulatory Mode:**

```json
{
  "report_header": {
    "report_id": "RPT-SPX-2024-001-20241120",
    "batch_id": "SPX-2024-001",
    "sample_description": "Spinach (100g)",
    "laboratory": "Wellomex Visual Chemometrics Lab",
    "test_date": "2024-11-20T10:30:00Z",
    "report_date": "2024-11-20T10:30:00Z",
    "compliance_status": "FAIL"
  },
  "regulatory_failures": [
    {
      "element": "Pb",
      "measured_value": "0.450 mg/kg",
      "regulatory_limit": "0.100 mg/kg",
      "exceedance": "350.0%",
      "authority": "FDA",
      "regulation": "Defect Action Level (21 CFR 109)",
      "action_required": "Immediate recall per 21 USC 331(k)"
    }
  ],
  "regulatory_compliance": [
    {
      "element": "Cd",
      "status": "COMPLIANT",
      "authority": "FDA/WHO"
    }
  ],
  "methodology": {
    "detection_method": "Visual Chemometrics - Deep Learning CNN",
    "instrument": "Wellomex Portable Food Scanner",
    "accreditation": "ISO/IEC 17025 (pending)",
    "quality_control": "Daily calibration with certified reference materials",
    "measurement_uncertainty": "±10-15% (k=2, 95% confidence)",
    "limit_of_detection": "Element-specific LOD range: 0.001-0.01 mg/kg"
  },
  "target_population": "General population, pregnant women, CKD patients",
  "recommended_actions": [...],
  "regulatory_notification_required": true,
  "certifications": [
    "This report is generated by AI-powered visual chemometrics",
    "Results should be confirmed by accredited laboratory for regulatory purposes"
  ]
}
```

---

### 3. POST /alternatives - Find Safer Alternative Foods

AI-powered search for safer alternatives that minimize problem elements while preserving nutritional benefits.

**Request:**

```http
POST /api/v1/risk-integration/alternatives
Content-Type: application/json
Authorization: Bearer <token>

{
  "original_food_name": "Spinach",
  "search_criteria": {
    "problem_elements": ["Pb", "Cd"],
    "preserve_nutrients": ["Fe", "Ca"],
    "category_preference": "leafy_greens",
    "max_price_increase_pct": 50.0,
    "require_seasonal": false
  },
  "user_profile": {...},
  "top_n": 5
}
```

**Search Criteria Fields:**

- `problem_elements`: Elements to minimize (e.g., toxic metals, restricted nutrients)
- `preserve_nutrients`: Elements to maintain (e.g., iron, calcium)
- `category_preference`: Food category (`leafy_greens`, `grains`, `fruits`, etc.)
- `max_price_increase_pct`: Maximum acceptable price increase (%)
- `require_seasonal`: Only show in-season foods (boolean)

**Response (200 OK):**

```json
{
  "original_food": "Spinach",
  "alternatives": [
    {
      "food_name": "Kale",
      "food_id": "kale_001",
      "total_score": 92.0,
      "risk_reduction_score": 90.0,
      "nutrient_preservation_score": 91.0,
      "availability_score": 100.0,
      "price_score": 80.0,
      "risk_improvement": "CRITICAL → SAFE",
      "element_improvements": {
        "Pb": 88.9,
        "Cd": 75.0
      },
      "element_comparisons": {
        "Pb": {
          "original": 0.45,
          "alternative": 0.05,
          "reduction_pct": 88.9
        },
        "Fe": {
          "original": 35.0,
          "alternative": 32.0,
          "reduction_pct": -8.6
        },
        "Ca": {
          "original": 1050.0,
          "alternative": 1500.0,
          "reduction_pct": -42.9
        }
      },
      "price": "$5.00/kg",
      "seasonality": "In season",
      "preparation_tips": [
        "Massage raw kale to soften",
        "Remove thick stems before cooking",
        "Bake into chips for healthy snack"
      ]
    },
    {
      "food_name": "Swiss Chard",
      "total_score": 88.0,
      "risk_improvement": "CRITICAL → SAFE",
      ...
    }
  ]
}
```

**Scoring Algorithm:**

```
Total Score = 0.5 × Risk_Reduction + 0.3 × Nutrient_Preservation + 0.1 × Availability + 0.1 × Price

Risk Reduction Score: How much problem elements are reduced (0-100)
Nutrient Preservation Score: How well beneficial nutrients maintained (0-100)
Availability Score: Seasonal/local availability (0-100)
Price Score: Cost comparison (0-100)
```

---

### 4. GET /thresholds/{condition} - Get Medical Thresholds

Retrieve regulatory limits and medical thresholds for a specific health condition.

**Request:**

```http
GET /api/v1/risk-integration/thresholds/77386006
Authorization: Bearer <token>
```

**Parameters:**

- `condition` (path): SNOMED CT code or condition name (e.g., "77386006" or "Pregnancy")

**Response (200 OK):**

```json
{
  "condition": "77386006",
  "thresholds": [
    {
      "element": "Pb",
      "limit_value": 0.1,
      "unit": "mg/kg",
      "threshold_type": "maximum",
      "authority": "FDA",
      "regulation": "Defect Action Level for Leafy Vegetables (Pregnancy)",
      "citation": "21 CFR 109.6"
    },
    {
      "element": "K",
      "limit_value": 2000.0,
      "unit": "mg/day",
      "threshold_type": "daily_maximum",
      "authority": "NKF",
      "regulation": "KDOQI CKD Stage 4 Potassium Limit",
      "citation": "KDOQI Clinical Practice Guidelines"
    }
  ]
}
```

**Supported Conditions:**

- Pregnancy (SNOMED: 77386006)
- CKD Stage 3-5 (SNOMED: 431855005, 431856006, 431857002, 433144002, 433146000)
- Diabetes Type 1/2
- Infant (<2 years)
- Hypertension
- Cardiovascular disease

---

### 5. POST /health-profile - Create/Update Health Profile

Create or update user health profile with medical conditions.

**Request:**

```http
POST /api/v1/risk-integration/health-profile
Content-Type: application/json
Authorization: Bearer <token>

{
  "user_id": "user_12345",
  "conditions": [
    {
      "snomed_code": "77386006",
      "name": "Pregnancy",
      "severity": "active",
      "diagnosed_date": "2024-01-15T00:00:00",
      "notes": "Second trimester, uncomplicated"
    },
    {
      "snomed_code": "431857002",
      "name": "CKD Stage 4",
      "severity": "severe",
      "diagnosed_date": "2023-06-01T00:00:00",
      "notes": "eGFR 22 ml/min, managed with diet"
    }
  ],
  "age": 32,
  "weight_kg": 68.0,
  "height_cm": 165.0,
  "medications": ["Prenatal vitamins", "Iron supplement"],
  "allergies": ["Penicillin"]
}
```

**Response (200 OK):**

```json
{
  "user_id": "user_12345",
  "conditions": [
    {
      "condition_id": "cond_0",
      "name": "Pregnancy",
      "snomed_code": "77386006",
      "severity": "active",
      "diagnosed_date": "2024-01-15T00:00:00"
    },
    {
      "condition_id": "cond_1",
      "name": "CKD Stage 4",
      "snomed_code": "431857002",
      "severity": "severe",
      "diagnosed_date": "2023-06-01T00:00:00"
    }
  ],
  "age": 32,
  "weight_kg": 68.0,
  "height_cm": 165.0,
  "risk_score": 85,
  "risk_level": "HIGH",
  "created_at": "2024-11-20T10:30:00Z",
  "updated_at": "2024-11-20T10:30:00Z"
}
```

**Risk Stratification:**

- `HIGH` (score > 80): Critical conditions (pregnancy + CKD, CKD Stage 5)
- `MODERATE` (score 50-80): Significant conditions (CKD Stage 3-4, diabetes)
- `LOW` (score < 50): Minor conditions or general population

---

### 6. GET /health-profile/{user_id} - Retrieve Health Profile

Get user's health profile by ID.

**Request:**

```http
GET /api/v1/risk-integration/health-profile/user_12345
Authorization: Bearer <token>
```

**Response (200 OK):**

```json
{
  "user_id": "user_12345",
  "conditions": [
    {
      "name": "Pregnancy",
      "snomed_code": "77386006",
      "severity": "active"
    }
  ],
  "age": 32,
  "weight_kg": 68.0,
  "height_cm": 165.0
}
```

**Status Codes:**

- `200 OK` - Profile found
- `404 Not Found` - User profile not found
- `401 Unauthorized` - Invalid authentication

---

### 7. GET /food-database/search - Search Food Database

Search food database by name or category.

**Request:**

```http
GET /api/v1/risk-integration/food-database/search?query=kale&category=leafy_greens
Authorization: Bearer <token>
```

**Query Parameters:**

- `query` (optional): Food name search term
- `category` (optional): Food category filter

**Response (200 OK):**

```json
{
  "count": 1,
  "foods": [
    {
      "food_id": "kale_001",
      "name": "Kale",
      "category": "leafy_greens",
      "subcategory": "dark_greens",
      "elements": {
        "Pb": 0.05,
        "Cd": 0.02,
        "As": 0.03,
        "Fe": 32.0,
        "Ca": 1500.0,
        "K": 4910.0
      },
      "nutrients": {
        "protein_g": 4.3,
        "fiber_g": 3.6,
        "calories_kcal": 49
      },
      "price_per_kg": 5.0,
      "in_season": true,
      "preparation_tips": [
        "Massage raw kale to soften",
        "Remove thick stems before cooking"
      ]
    }
  ]
}
```

**Food Categories:**

- `leafy_greens`
- `cruciferous`
- `root_vegetables`
- `legumes`
- `grains`
- `fruits`
- `nuts_seeds`
- `fish_seafood`
- `meat_poultry`
- `dairy`

---

### 8. GET /health - API Health Check

Check if Risk Integration API is operational.

**Request:**

```http
GET /api/v1/risk-integration/health
```

**Response (200 OK):**

```json
{
  "status": "healthy",
  "service": "Risk Integration API",
  "version": "1.0.0",
  "components": {
    "threshold_database": "operational",
    "profile_engine": "operational",
    "risk_engine": "operational",
    "warning_system": "operational",
    "food_database": "operational",
    "alternative_finder": "operational"
  }
}
```

---

## Data Models

### ElementPrediction

Represents a single element prediction from chemometric analysis.

```typescript
{
  element: string;              // Element symbol (e.g., "Pb", "K", "Fe")
  concentration: number;        // Measured value (≥0)
  uncertainty?: number;         // Measurement uncertainty (≥0)
  confidence: number;           // Prediction confidence (0.0-1.0)
  unit: string;                 // Unit (ppm, mg/kg, mg/100g)
  measurement_method: string;   // Detection method
}
```

### MedicalCondition

User medical condition with SNOMED CT coding.

```typescript
{
  snomed_code: string;          // SNOMED CT code
  name: string;                 // Condition name
  severity?: string;            // Severity level
  diagnosed_date?: DateTime;    // Diagnosis date
  notes?: string;               // Additional notes
}
```

### RiskAssessment

Complete food safety risk assessment.

```typescript
{
  overall_risk_level: string;   // CRITICAL, HIGH, MODERATE, LOW, SAFE
  confidence: number;           // 0.0-1.0
  safety_failures: Array<{
    element: string;
    measured_value: number;
    threshold: number;
    exceeds_limit: boolean;
    exceedance_percentage: number;
    severity: string;
  }>;
  nutrient_warnings: Array<{
    element: string;
    contribution_percentage: number;
    is_restricted: boolean;
  }>;
  nutrient_benefits: Array<{
    element: string;
    contribution_percentage: number;
    is_beneficial: boolean;
  }>;
  uncertainty_flags: string[];
}
```

---

## Error Responses

### 400 Bad Request

Invalid input data.

```json
{
  "detail": "Invalid element symbol: 'XX'. Must be valid element from periodic table."
}
```

### 401 Unauthorized

Missing or invalid authentication.

```json
{
  "detail": "Not authenticated"
}
```

### 404 Not Found

Resource not found.

```json
{
  "detail": "Profile not found for user 'user_12345'"
}
```

### 500 Internal Server Error

Server processing error.

```json
{
  "detail": "Risk assessment failed: Internal processing error"
}
```

---

## Code Examples

### Python

```python
import requests

BASE_URL = "https://api.wellomex.com/api/v1/risk-integration"
TOKEN = "your_access_token"

# Assess food risk
response = requests.post(
    f"{BASE_URL}/assess",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json={
        "predictions": [
            {
                "element": "Pb",
                "concentration": 0.45,
                "confidence": 0.92,
                "unit": "ppm"
            }
        ],
        "user_profile": {
            "user_id": "user_12345",
            "conditions": [
                {"snomed_code": "77386006", "name": "Pregnancy"}
            ],
            "age": 32
        },
        "food_item": "Spinach",
        "serving_size": 100.0
    }
)

risk_assessment = response.json()
print(f"Risk Level: {risk_assessment['overall_risk_level']}")
```

### JavaScript/TypeScript

```typescript
const BASE_URL = 'https://api.wellomex.com/api/v1/risk-integration';
const TOKEN = 'your_access_token';

// Generate warnings
const response = await fetch(`${BASE_URL}/warnings`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${TOKEN}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    predictions: [{
      element: 'Pb',
      concentration: 0.45,
      confidence: 0.92,
      unit: 'ppm'
    }],
    user_profile: {
      user_id: 'user_12345',
      conditions: [{
        snomed_code: '77386006',
        name: 'Pregnancy'
      }],
      age: 32
    },
    food_item: 'Spinach',
    serving_size: 100.0,
    message_mode: 'consumer'
  })
});

const warnings = await response.json();
console.log(warnings.banner.message);
```

### cURL

```bash
# Find alternatives
curl -X POST "https://api.wellomex.com/api/v1/risk-integration/alternatives" \
  -H "Authorization: Bearer your_access_token" \
  -H "Content-Type: application/json" \
  -d '{
    "original_food_name": "Spinach",
    "search_criteria": {
      "problem_elements": ["Pb"],
      "preserve_nutrients": ["Fe", "Ca"],
      "max_price_increase_pct": 50.0
    },
    "top_n": 5
  }'
```

---

## Changelog

### Version 1.0.0 (2024-11-20)

- Initial release of Risk Integration API
- 5-step risk assessment engine
- Personalized warning system (3 modes)
- AI-powered alternative food finder
- Health profile management
- Dynamic threshold database (500+ limits)
- Support for pregnancy, CKD, diabetes conditions
- FDA, WHO, NKF, KDIGO regulatory compliance

---

## Support & Resources

- **API Documentation**: https://api.wellomex.com/api/docs
- **Support Email**: support@wellomex.com
- **Developer Portal**: https://developers.wellomex.com
- **Status Page**: https://status.wellomex.com

---

*Generated: November 20, 2024*  
*API Version: 1.0.0*
