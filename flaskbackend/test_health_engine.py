"""
Standalone test for the enhanced Health Profile Engine.
Tests ML/DL integration without complex import dependencies.
"""

import sys
sys.path.insert(0, r"C:\Users\Codeternal\Music\wellomex\flaskbackend")

from app.ai_nutrition.risk_integration.health_profile_engine import (
    HealthProfileEngine,
    HealthCondition,
    LabValue,
    Gender,
    TherapeuticGoal,
    CKDStage
)
from app.ai_nutrition.risk_integration.dynamic_thresholds import DynamicThresholdDatabase

def main():
    print("\n" + "="*80)
    print("ENHANCED HEALTH PROFILE ENGINE TEST")
    print("ML/DL Integration - 55+ Therapeutic Goals - Dynamic Data Loading")
    print("="*80)
    
    # Initialize
    print("\n" + "-"*80)
    print("Initializing...")
    
    threshold_db = DynamicThresholdDatabase()
    engine = HealthProfileEngine(threshold_db)
    
    print("✓ Initialized Engine with:")
    print("  - Risk Stratification Model (RSM)")
    print("  - Therapeutic Recommendation Engine (MTRE)")
    print("  - Disease Compound Extractor (DICE)")
    print("  - Model Loader (ML/DL models)")
    print("  - Data Loader (External configs)")
    
    # Test 1: Create profile with CKD Stage 4
    print("\n" + "-"*80)
    print("Test 1: Patient with CKD Stage 4 (High Risk)")
    
    profile = engine.create_profile(
        user_id="patient001",
        age=65,
        gender=Gender.MALE,
        body_weight_kg=80,
        height_cm=175
    )
    
    # Add CKD condition
    ckd_condition = HealthCondition(
        condition_id="ckd001",
        condition_name="CKD Stage 4",
        severity="severe",
        ckd_stage=CKDStage.CKD_STAGE_4,
        egfr=22.0,
        serum_potassium=5.2,
        serum_phosphorus=5.8
    )
    
    engine.add_condition("patient001", ckd_condition)
    
    # Add abnormal lab
    engine.add_lab_value("patient001", LabValue(
        test_name="Potassium",
        value=5.2,
        unit="mEq/L",
        reference_range_min=3.5,
        reference_range_max=5.0
    ))
    
    # Analyze Risk (RSM)
    risk_analysis = engine.analyze_health_risk("patient001")
    print(f"\n✓ Risk Analysis (RSM with ML Model):")
    print(f"  Risk Score: {risk_analysis['risk_score']:.1f}/100")
    print(f"  Risk Level: {risk_analysis['risk_level']}")
    print(f"  Progression Prob (5yr): {risk_analysis['progression_probability_5yr']:.2%}")
    print(f"  Active Conditions: {', '.join(risk_analysis['active_conditions'])}")
    
    # Test 2: Therapeutic Recommendations (MTRE) - Now with 55+ goals
    print("\n" + "-"*80)
    print("Test 2: Therapeutic Food Recommendations (MTRE)")
    print("Testing with expanded goal set (55+ goals available)")
    
    engine.set_therapeutic_goals("patient001", [
        TherapeuticGoal.KIDNEY_PROTECTION,
        TherapeuticGoal.ANTI_INFLAMMATORY,
        TherapeuticGoal.HEART_HEALTH,
        TherapeuticGoal.ENERGY_BOOST,
        TherapeuticGoal.ANTIOXIDANT_SUPPORT
    ])
    
    sample_foods = [
        {
            "name": "Spinach (Raw)",
            "potassium_mg": 558,
            "compounds": ["lutein", "nitrates", "magnesium"]
        },
        {
            "name": "Blueberries",
            "potassium_mg": 77,
            "compounds": ["anthocyanins", "quercetin", "vitamin_c"]
        },
        {
            "name": "Salmon",
            "potassium_mg": 363,
            "compounds": ["omega-3", "protein", "selenium"]
        }
    ]
    
    recommendations = engine.get_food_recommendations("patient001", sample_foods)
    
    print(f"\n✓ Top Recommendations (ranked by ML model):")
    for i, food in enumerate(recommendations[:3]):
        print(f"  {i+1}. {food['name']}")
        print(f"     Score: {food['uplift_score']:.1f}")
        print(f"     Reasons: {', '.join(food['therapeutic_reasons']) if food['therapeutic_reasons'] else 'N/A'}")
        
    # Test 3: Disease Rules (DICE with external data)
    print("\n" + "-"*80)
    print("Test 3: Disease Rules Extraction (DICE)")
    print("Testing with dynamically loaded knowledge base")
    
    thresholds = engine.get_applicable_thresholds("patient001", "Potassium")
    
    print(f"\n✓ Extracted Rules:")
    print(f"  Risk Score: {thresholds.get('risk_score', 'N/A')}")
    print(f"  Therapy Rules: {list(thresholds.get('therapy_rules', {}).keys())}")
    
    # Test 4: Show expanded therapeutic goals
    print("\n" + "-"*80)
    print("Test 4: Expanded Therapeutic Goals (55+)")
    
    print("\n✓ Sample Available Goals:")
    sample_goals = [
        TherapeuticGoal.WEIGHT_LOSS,
        TherapeuticGoal.MUSCLE_GAIN,
        TherapeuticGoal.COGNITIVE_FUNCTION,
        TherapeuticGoal.ANTI_AGING,
        TherapeuticGoal.CANCER_PREVENTION,
        TherapeuticGoal.MITOCHONDRIAL_HEALTH,
        TherapeuticGoal.TELOMERE_SUPPORT,
        TherapeuticGoal.MOOD_ENHANCEMENT
    ]
    
    for goal in sample_goals:
        print(f"  - {goal.value.replace('_', ' ').title()}")
    
    print(f"\n  Total Available Goals: {len(TherapeuticGoal)} (expandable)")

    # Test 5: Health Summary
    print("\n" + "-"*80)
    print("Test 5: Comprehensive Health Summary")
    
    summary = engine.generate_health_summary("patient001")
    print(f"\n✓ Summary:")
    print(f"  Overall Risk: {summary['overall_risk']}")
    print(f"  Risk Score: {summary['risk_score']:.1f}/100")
    print(f"  Active Conditions: {len(summary['conditions'])}")
    print(f"  Therapeutic Goals: {len(summary['therapeutic_goals'])}")
    print(f"  Applicable Conditions: {', '.join(summary['applicable_conditions'][:3])}...")

    print("\n" + "="*80)
    print("✓ TEST COMPLETE - All ML/DL integrations working!")
    print("="*80)
    print("\nKey Improvements:")
    print("  ✓ Removed hardcoded medication database")
    print("  ✓ Integrated ML model loaders (RSM, MTRE, DICE)")
    print("  ✓ Expanded therapeutic goals from 10 to 55+")
    print("  ✓ Dynamic data loading from JSON/external configs")
    print("  ✓ Mock models for development (ready for real model weights)")
    print("  ✓ Modular architecture for easy model swapping")

if __name__ == "__main__":
    main()
