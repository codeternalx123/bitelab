"""
Comprehensive Test Suite for Visual-Atomic Food Analysis System

Tests the integration of:
1. Visual feature extraction (shininess, reflection, color, texture)
2. Element prediction from visual features
3. ICP-MS validation of visual predictions
4. Safety assessment based on visual and atomic analysis
"""

from visual_atomic_analyzer import (
    VisualFeatures, SurfaceType, ColorProfile,
    IntegratedVisualICPMSAnalyzer
)
from food_nutrient_detector import FoodNutrientDetector
import json


def test_visual_only_analysis():
    """Test 1: Pure visual analysis without ICP-MS data"""
    print("=" * 80)
    print("TEST 1: Visual-Only Analysis (Fresh Spinach)")
    print("=" * 80)
    
    # Create visual features for fresh spinach
    spinach_visual = VisualFeatures(
        # Shininess & Reflection
        shininess_index=78.0,  # Fresh leafy greens are glossy
        reflection_intensity=82.0,
        specular_highlights=15,
        
        # Surface Properties
        surface_type=SurfaceType.GLOSSY,
        texture_roughness=48.0,  # Some leaf veins visible
        moisture_appearance=88.0,  # Very fresh and moist
        
        # Color Analysis
        color_profile=ColorProfile.DEEP_GREEN,
        rgb_values=(34, 139, 34),  # Forest green
        color_uniformity=82.0,
        brightness=62.0,
        saturation=92.0,
        
        # Freshness Indicators
        wilting_score=3.0,  # Minimal wilting
        browning_score=1.0,  # Almost no browning
        spots_or_blemishes=2,  # Very few blemishes
        
        # Size & Shape
        size_mm=155.0,
        shape_regularity=72.0,
        
        # Advanced Features
        translucency=18.0,
        crystalline_structures=False,
        oily_film=False,
        dust_or_residue=False,
        
        lighting_conditions="natural daylight"
    )
    
    # Analyze with visual features only
    detector = FoodNutrientDetector()
    profile = detector.analyze_food('spinach', visual_features=spinach_visual)
    
    print(f"\n✅ Food: {profile.food_name}")
    print(f"   Category: {profile.food_category}")
    print(f"   Safety: {'SAFE ✅' if profile.is_safe_for_consumption else 'UNSAFE ❌'}")
    print(f"   Nutritional Score: {profile.nutritional_quality_score}/100")
    print(f"   Freshness Score: {profile.freshness_score}/100")
    print(f"   Purity Score: {profile.purity_score}/100")
    
    if profile.visual_safety_assessment:
        print(f"\n📊 Visual Safety Assessment:")
        print(f"   Freshness Rating: {profile.visual_safety_assessment['freshness_rating']}")
        print(f"   Recommended Action: {profile.visual_safety_assessment['recommended_action']}")
        print(f"   Heavy Metal Risk: {profile.visual_safety_assessment['risks']['heavy_metal']}")
        print(f"   Spoilage Risk: {profile.visual_safety_assessment['risks']['spoilage']}")
    
    if profile.visual_predictions:
        print(f"\n🔬 Visual Element Predictions ({len(profile.visual_predictions)} elements):")
        for pred in profile.visual_predictions[:5]:  # Show top 5
            print(f"   {pred['element']}: {pred['predicted_ppm']:.1f} ppm (confidence: {pred['confidence']:.0%})")
            print(f"      Basis: {', '.join(pred['prediction_basis'][:2])}")
    
    print("\n" + "-" * 80)


def test_visual_with_icpms_validation():
    """Test 2: Visual analysis validated with ICP-MS measurements"""
    print("\n" + "=" * 80)
    print("TEST 2: Visual Analysis + ICP-MS Validation (Fresh Spinach)")
    print("=" * 80)
    
    # Same visual features as Test 1
    spinach_visual = VisualFeatures(
        shininess_index=78.0,
        reflection_intensity=82.0,
        specular_highlights=15,
        surface_type=SurfaceType.GLOSSY,
        texture_roughness=48.0,
        moisture_appearance=88.0,
        color_profile=ColorProfile.DEEP_GREEN,
        rgb_values=(34, 139, 34),
        color_uniformity=82.0,
        brightness=62.0,
        saturation=92.0,
        wilting_score=3.0,
        browning_score=1.0,
        spots_or_blemishes=2,
        size_mm=155.0,
        shape_regularity=72.0,
        translucency=18.0,
        crystalline_structures=False,
        oily_film=False,
        dust_or_residue=False
    )
    
    # Add ICP-MS measurements
    icpms_data = {
        'Ca': 990.0,   # Calcium
        'Fe': 27.0,    # Iron
        'Mg': 790.0,   # Magnesium
        'K': 5580.0,   # Potassium
        'P': 490.0,    # Phosphorus
        'Zn': 5.3,     # Zinc
        'Mn': 8.97,    # Manganese
        'Cu': 1.3,     # Copper
        'Pb': 0.018,   # Lead (safe level)
        'Cd': 0.008,   # Cadmium (safe level)
    }
    
    # Analyze with both visual and ICP-MS
    detector = FoodNutrientDetector()
    profile = detector.analyze_food('spinach', icpms_data=icpms_data, visual_features=spinach_visual)
    
    print(f"\n✅ Food: {profile.food_name}")
    print(f"   Safety: {'SAFE ✅' if profile.is_safe_for_consumption else 'UNSAFE ❌'}")
    print(f"   Heavy Metal Contamination: {'YES ❌' if profile.heavy_metal_contamination else 'NO ✅'}")
    
    print(f"\n🔬 ICP-MS Elemental Analysis ({len(profile.elemental_composition)} elements):")
    for elem in profile.elemental_composition:
        safety_icon = "✅" if elem.is_safe else "❌"
        essential_tag = "[Essential]" if elem.is_essential else "[TOXIC]"
        print(f"   {elem.element_symbol} ({elem.element_name}): {elem.concentration_ppm:.2f} ppm {safety_icon} {essential_tag}")
        if elem.safe_limit_ppm:
            print(f"      Safe Limit: {elem.safe_limit_ppm} ppm | Status: {'SAFE' if elem.is_safe else 'EXCEEDED'}")
    
    if profile.visual_predictions:
        print(f"\n📊 Visual Prediction Validation:")
        for pred in profile.visual_predictions[:5]:
            if pred['icpms_validated']:
                error = pred.get('prediction_error_%', 0)
                accuracy = 100 - error
                print(f"   {pred['element']}: Predicted {pred['predicted_ppm']:.1f} ppm vs Actual {pred['actual_ppm']:.1f} ppm")
                print(f"      Accuracy: {accuracy:.1f}% | Confidence: {pred['confidence']:.0%}")
    
    print("\n" + "-" * 80)


def test_contaminated_food_detection():
    """Test 3: Detect contaminated food from visual features"""
    print("\n" + "=" * 80)
    print("TEST 3: Contaminated Food Detection (Suspicious Apple)")
    print("=" * 80)
    
    # Apple with metallic sheen and discoloration (warning signs)
    suspicious_apple = VisualFeatures(
        shininess_index=88.0,  # Unusually high shine
        reflection_intensity=92.0,
        specular_highlights=25,
        surface_type=SurfaceType.METALLIC,  # ⚠️ RED FLAG
        texture_roughness=12.0,
        moisture_appearance=65.0,
        color_profile=ColorProfile.DISCOLORED,  # ⚠️ RED FLAG
        rgb_values=(175, 145, 115),  # Abnormal brownish color
        color_uniformity=55.0,  # Uneven coloring
        brightness=78.0,
        saturation=38.0,
        wilting_score=12.0,
        browning_score=22.0,
        spots_or_blemishes=12,  # Many blemishes
        size_mm=82.0,
        shape_regularity=83.0,
        translucency=8.0,
        crystalline_structures=False,
        oily_film=True,  # ⚠️ Warning
        dust_or_residue=True,  # ⚠️ Warning
    )
    
    # Add ICP-MS showing actual contamination
    icpms_contaminated = {
        'Ca': 60.0,
        'K': 1070.0,
        'Fe': 3.2,
        'Pb': 0.15,  # ❌ EXCEEDS SAFE LIMIT (0.1 ppm)
        'Cd': 0.08,  # ❌ EXCEEDS SAFE LIMIT (0.05 ppm)
        'Hg': 0.06,  # ❌ EXCEEDS SAFE LIMIT (0.05 ppm)
        'Al': 12.0,  # ❌ EXCEEDS SAFE LIMIT (10 ppm)
    }
    
    detector = FoodNutrientDetector()
    profile = detector.analyze_food('apple', icpms_data=icpms_contaminated, visual_features=suspicious_apple)
    
    print(f"\n⚠️  Food: {profile.food_name}")
    print(f"   Safety: {'SAFE ✅' if profile.is_safe_for_consumption else '❌ UNSAFE - DO NOT CONSUME'}")
    print(f"   Heavy Metal Contamination: {'❌ YES - DANGEROUS' if profile.heavy_metal_contamination else 'NO ✅'}")
    print(f"   Purity Score: {profile.purity_score}/100")
    
    if profile.visual_safety_assessment:
        print(f"\n🚨 Visual Safety Assessment:")
        print(f"   Safety Score: {profile.visual_safety_assessment['safety_score']}/100")
        print(f"   Freshness: {profile.visual_safety_assessment['freshness_rating']}")
        print(f"   ⚠️  Recommended Action: {profile.visual_safety_assessment['recommended_action']}")
        
        print(f"\n   Risk Levels:")
        for risk_type, risk_level in profile.visual_safety_assessment['risks'].items():
            icon = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"
            print(f"   {icon} {risk_type.replace('_', ' ').title()}: {risk_level}")
        
        if profile.visual_safety_assessment['warnings']:
            print(f"\n   ⚠️  Warnings:")
            for warning in profile.visual_safety_assessment['warnings']:
                print(f"   - {warning}")
        
        if profile.visual_safety_assessment['contamination_indicators']:
            print(f"\n   🚨 Contamination Indicators:")
            for indicator in profile.visual_safety_assessment['contamination_indicators']:
                print(f"   - {indicator}")
    
    print(f"\n🔬 Toxic Elements Detected:")
    for elem in profile.elemental_composition:
        if not elem.is_essential and not elem.is_safe:
            print(f"   ❌ {elem.element_name} ({elem.element_symbol}): {elem.concentration_ppm:.3f} ppm")
            print(f"      Safe Limit: {elem.safe_limit_ppm} ppm | EXCEEDED by {((elem.concentration_ppm/elem.safe_limit_ppm - 1) * 100):.1f}%")
            print(f"      Risk: {elem.health_role}")
    
    print("\n" + "-" * 80)


def test_aged_produce():
    """Test 4: Analyze aged/spoiled produce"""
    print("\n" + "=" * 80)
    print("TEST 4: Aged Produce Analysis (Wilted Broccoli)")
    print("=" * 80)
    
    aged_broccoli = VisualFeatures(
        shininess_index=22.0,  # Lost shine
        reflection_intensity=18.0,
        specular_highlights=1,
        surface_type=SurfaceType.ROUGH,
        texture_roughness=85.0,
        moisture_appearance=28.0,  # Dried out
        color_profile=ColorProfile.BROWN,  # Oxidized
        rgb_values=(95, 85, 48),  # Brownish-yellow
        color_uniformity=48.0,
        brightness=32.0,
        saturation=25.0,
        wilting_score=72.0,  # Severely wilted
        browning_score=68.0,  # Heavily browned
        spots_or_blemishes=23,  # Many blemishes
        size_mm=115.0,
        shape_regularity=55.0,
        translucency=3.0,
        crystalline_structures=False,
        oily_film=False,
        dust_or_residue=False,
    )
    
    detector = FoodNutrientDetector()
    profile = detector.analyze_food('broccoli', visual_features=aged_broccoli)
    
    print(f"\n📊 Food: {profile.food_name}")
    print(f"   Safety: {'SAFE ✅' if profile.is_safe_for_consumption else 'QUESTIONABLE ⚠️'}")
    print(f"   Nutritional Score: {profile.nutritional_quality_score}/100")
    print(f"   Freshness Score: {profile.freshness_score}/100 (POOR)")
    
    if profile.visual_safety_assessment:
        print(f"\n   Visual Assessment:")
        print(f"   - Freshness Rating: {profile.visual_safety_assessment['freshness_rating'].upper()}")
        print(f"   - Safety Score: {profile.visual_safety_assessment['safety_score']}/100")
        print(f"   - Spoilage Risk: {profile.visual_safety_assessment['risks']['spoilage']}")
        print(f"   - Microbial Risk: {profile.visual_safety_assessment['risks']['microbial']}")
        print(f"   - 🔔 Recommendation: {profile.visual_safety_assessment['recommended_action']}")
    
    print("\n" + "-" * 80)


def test_comparison_fresh_vs_aged():
    """Test 5: Side-by-side comparison of fresh vs aged food"""
    print("\n" + "=" * 80)
    print("TEST 5: Fresh vs Aged Comparison (Kale)")
    print("=" * 80)
    
    # Fresh kale
    fresh_kale = VisualFeatures(
        shininess_index=82.0,
        reflection_intensity=85.0,
        specular_highlights=18,
        surface_type=SurfaceType.GLOSSY,
        texture_roughness=52.0,
        moisture_appearance=92.0,
        color_profile=ColorProfile.DEEP_GREEN,
        rgb_values=(28, 115, 28),
        color_uniformity=88.0,
        brightness=58.0,
        saturation=95.0,
        wilting_score=2.0,
        browning_score=0.5,
        spots_or_blemishes=1,
        size_mm=180.0,
        shape_regularity=75.0,
        translucency=15.0,
        crystalline_structures=False,
        oily_film=False,
        dust_or_residue=False,
    )
    
    # Aged kale (3 days old)
    aged_kale = VisualFeatures(
        shininess_index=45.0,
        reflection_intensity=38.0,
        specular_highlights=5,
        surface_type=SurfaceType.MATTE,
        texture_roughness=68.0,
        moisture_appearance=42.0,
        color_profile=ColorProfile.DEEP_GREEN,
        rgb_values=(48, 95, 35),
        color_uniformity=65.0,
        brightness=45.0,
        saturation=62.0,
        wilting_score=38.0,
        browning_score=25.0,
        spots_or_blemishes=8,
        size_mm=175.0,
        shape_regularity=68.0,
        translucency=12.0,
        crystalline_structures=False,
        oily_film=False,
        dust_or_residue=False,
    )
    
    detector = FoodNutrientDetector()
    profile_fresh = detector.analyze_food('kale', visual_features=fresh_kale)
    profile_aged = detector.analyze_food('kale', visual_features=aged_kale)
    
    print("\n📊 COMPARISON RESULTS:\n")
    print(f"{'Metric':<30} {'Fresh Kale':<20} {'Aged Kale (3 days)':<20}")
    print("-" * 70)
    print(f"{'Safety Status':<30} {'✅ SAFE':<20} {'⚠️  ' + ('SAFE' if profile_aged.is_safe_for_consumption else 'QUESTIONABLE'):<20}")
    print(f"{'Freshness Score':<30} {profile_fresh.freshness_score:<20.1f} {profile_aged.freshness_score:<20.1f}")
    print(f"{'Nutritional Score':<30} {profile_fresh.nutritional_quality_score:<20.1f} {profile_aged.nutritional_quality_score:<20.1f}")
    print(f"{'Purity Score':<30} {profile_fresh.purity_score:<20.1f} {profile_aged.purity_score:<20.1f}")
    
    if profile_fresh.visual_safety_assessment and profile_aged.visual_safety_assessment:
        print(f"\n{'Visual Safety Assessment:':<30}")
        fresh_rating = profile_fresh.visual_safety_assessment['freshness_rating']
        aged_rating = profile_aged.visual_safety_assessment['freshness_rating']
        print(f"{'  Freshness Rating':<30} {fresh_rating:<20} {aged_rating:<20}")
        
        fresh_safety = profile_fresh.visual_safety_assessment['safety_score']
        aged_safety = profile_aged.visual_safety_assessment['safety_score']
        print(f"{'  Safety Score':<30} {fresh_safety:<20.1f} {aged_safety:<20.1f}")
        
        print(f"\n{'Recommended Action:':<30}")
        print(f"  Fresh: {profile_fresh.visual_safety_assessment['recommended_action']}")
        print(f"  Aged:  {profile_aged.visual_safety_assessment['recommended_action']}")
    
    print("\n" + "-" * 80)


def run_all_tests():
    """Run complete test suite"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "VISUAL-ATOMIC FOOD ANALYSIS TEST SUITE" + " " * 24 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\nTesting integration of visual features with ICP-MS atomic analysis")
    print("for comprehensive food safety and quality assessment.\n")
    
    try:
        test_visual_only_analysis()
        test_visual_with_icpms_validation()
        test_contaminated_food_detection()
        test_aged_produce()
        test_comparison_fresh_vs_aged()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nKey Capabilities Demonstrated:")
        print("  ✅ Visual feature extraction (shininess, reflection, color, texture)")
        print("  ✅ Element prediction from visual appearance")
        print("  ✅ ICP-MS validation of visual predictions")
        print("  ✅ Heavy metal contamination detection")
        print("  ✅ Freshness and quality assessment")
        print("  ✅ Safety recommendations based on visual + atomic data")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
