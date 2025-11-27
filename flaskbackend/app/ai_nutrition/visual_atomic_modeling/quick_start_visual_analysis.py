"""
Quick Start Guide - Visual-Atomic Food Analysis System
=======================================================

This example demonstrates how to analyze food using visual features
(shininess, reflection, color, texture) combined with ICP-MS data
to predict atomic composition and assess safety.
"""

from visual_atomic_analyzer import (
    VisualFeatures, SurfaceType, ColorProfile,
    IntegratedVisualICPMSAnalyzer
)
from food_nutrient_detector import FoodNutrientDetector
import json


def example_1_fresh_spinach():
    """
    Example 1: Analyze fresh spinach with visual inspection
    
    Use Case: Quick screening at grocery store or farm
    Method: Visual inspection only (no lab equipment needed)
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Fresh Spinach - Visual Inspection Only")
    print("="*80)
    
    print("\n📸 Visual Inspection:")
    print("   - Very glossy leaves (shininess: 78/100)")
    print("   - Deep green color (high chlorophyll)")
    print("   - Firm and crisp (minimal wilting)")
    print("   - Few blemishes (2 small spots)")
    print("   - Moist appearance")
    
    # Create visual features
    spinach = VisualFeatures(
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
    
    # Analyze
    detector = FoodNutrientDetector()
    profile = detector.analyze_food('spinach', visual_features=spinach)
    
    # Display results
    print(f"\n✅ Analysis Results:")
    print(f"   Food: {profile.food_name}")
    print(f"   Safety: {'SAFE ✅' if profile.is_safe_for_consumption else 'UNSAFE ❌'}")
    print(f"   Nutritional Quality: {profile.nutritional_quality_score}/100")
    print(f"   Freshness: {profile.freshness_score}/100")
    print(f"   Purity: {profile.purity_score}/100")
    
    if profile.visual_safety_assessment:
        assessment = profile.visual_safety_assessment
        print(f"\n📊 Safety Assessment:")
        print(f"   Overall Safety Score: {assessment['safety_score']}/100")
        print(f"   Freshness Rating: {assessment['freshness_rating']}")
        print(f"   Recommended Action: {assessment['recommended_action']}")
        print(f"\n   Risk Levels:")
        for risk_type, level in assessment['risks'].items():
            icon = "🔴" if level == "High" else "🟡" if level == "Medium" else "🟢"
            print(f"   {icon} {risk_type.replace('_', ' ').title()}: {level}")
    
    if profile.visual_predictions:
        print(f"\n🔬 Predicted Elements (Top 5):")
        for pred in profile.visual_predictions[:5]:
            print(f"   {pred['element']}: {pred['predicted_ppm']:.1f} ppm "
                  f"(confidence: {pred['confidence']:.0%})")
            print(f"      Based on: {', '.join(pred['prediction_basis'][:2])}")


def example_2_spinach_with_icpms():
    """
    Example 2: Validate visual predictions with ICP-MS measurements
    
    Use Case: Laboratory quality control
    Method: Visual inspection + ICP-MS spectroscopy
    """
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Fresh Spinach - Visual + ICP-MS Validation")
    print("="*80)
    
    print("\n📸 Visual Features: (same as Example 1)")
    print("🔬 ICP-MS Measurements Added:")
    print("   Ca: 990 ppm, Fe: 27 ppm, Mg: 790 ppm, K: 5580 ppm")
    print("   Zn: 5.3 ppm, P: 490 ppm, Pb: 0.018 ppm, Cd: 0.008 ppm")
    
    # Visual features
    spinach = VisualFeatures(
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
    
    # ICP-MS data
    icpms_data = {
        'Ca': 990.0,
        'Fe': 27.0,
        'Mg': 790.0,
        'K': 5580.0,
        'P': 490.0,
        'Zn': 5.3,
        'Mn': 8.97,
        'Cu': 1.3,
        'Pb': 0.018,
        'Cd': 0.008,
    }
    
    # Analyze
    detector = FoodNutrientDetector()
    profile = detector.analyze_food('spinach', icpms_data=icpms_data, 
                                   visual_features=spinach)
    
    # Results
    print(f"\n✅ Analysis Results:")
    print(f"   Safety: {'SAFE ✅' if profile.is_safe_for_consumption else 'UNSAFE ❌'}")
    print(f"   Heavy Metal Contamination: {'YES ❌' if profile.heavy_metal_contamination else 'NO ✅'}")
    
    print(f"\n🔬 ICP-MS Elemental Analysis:")
    for elem in profile.elemental_composition[:8]:
        safety = "✅" if elem.is_safe else "❌"
        elem_type = "[Essential]" if elem.is_essential else "[TOXIC]"
        print(f"   {elem.element_symbol} ({elem.element_name}): "
              f"{elem.concentration_ppm:.2f} ppm {safety} {elem_type}")
    
    if profile.visual_predictions:
        print(f"\n📊 Visual Prediction Accuracy:")
        for pred in profile.visual_predictions[:5]:
            if pred['icpms_validated']:
                accuracy = 100 - pred.get('prediction_error_%', 100)
                print(f"   {pred['element']}: Predicted {pred['predicted_ppm']:.1f} ppm, "
                      f"Actual {pred['actual_ppm']:.1f} ppm "
                      f"(Accuracy: {accuracy:.1f}%)")


def example_3_contaminated_apple():
    """
    Example 3: Detect contaminated food from visual warning signs
    
    Use Case: Food safety screening
    Method: Visual red flags → ICP-MS confirmation
    """
    print("\n\n" + "="*80)
    print("EXAMPLE 3: Contaminated Apple - Warning Detection")
    print("="*80)
    
    print("\n⚠️  Visual Warning Signs Detected:")
    print("   🚨 METALLIC SHEEN on surface")
    print("   🚨 DISCOLORED appearance (brownish tint)")
    print("   ⚠️  Oily film present")
    print("   ⚠️  Surface dust/residue")
    print("   ⚠️  12 spots/blemishes")
    
    # Suspicious visual features
    apple = VisualFeatures(
        shininess_index=88.0,
        reflection_intensity=92.0,
        specular_highlights=25,
        surface_type=SurfaceType.METALLIC,  # RED FLAG
        texture_roughness=12.0,
        moisture_appearance=65.0,
        color_profile=ColorProfile.DISCOLORED,  # RED FLAG
        rgb_values=(175, 145, 115),
        color_uniformity=55.0,
        brightness=78.0,
        saturation=38.0,
        wilting_score=12.0,
        browning_score=22.0,
        spots_or_blemishes=12,
        size_mm=82.0,
        shape_regularity=83.0,
        translucency=8.0,
        crystalline_structures=False,
        oily_film=True,
        dust_or_residue=True,
    )
    
    # ICP-MS confirms contamination
    icpms_contaminated = {
        'Ca': 60.0,
        'K': 1070.0,
        'Fe': 3.2,
        'Pb': 0.15,   # EXCEEDS 0.1 ppm limit
        'Cd': 0.08,   # EXCEEDS 0.05 ppm limit
        'Hg': 0.06,   # EXCEEDS 0.05 ppm limit
        'Al': 12.0,   # EXCEEDS 10 ppm limit
    }
    
    detector = FoodNutrientDetector()
    profile = detector.analyze_food('apple', icpms_data=icpms_contaminated, 
                                   visual_features=apple)
    
    print(f"\n❌ Analysis Results:")
    print(f"   Safety: {'SAFE ✅' if profile.is_safe_for_consumption else '❌ UNSAFE - DO NOT CONSUME'}")
    print(f"   Heavy Metal Contamination: {'❌ YES - DANGEROUS' if profile.heavy_metal_contamination else 'NO ✅'}")
    print(f"   Safety Score: {profile.visual_safety_assessment['safety_score']}/100")
    
    print(f"\n🚨 Contamination Report:")
    if profile.visual_safety_assessment['contamination_indicators']:
        for indicator in profile.visual_safety_assessment['contamination_indicators']:
            print(f"   - {indicator}")
    
    print(f"\n⚠️  Warnings:")
    if profile.visual_safety_assessment['warnings']:
        for warning in profile.visual_safety_assessment['warnings']:
            print(f"   - {warning}")
    
    print(f"\n🔬 Toxic Elements Detected:")
    for elem in profile.elemental_composition:
        if not elem.is_essential and not elem.is_safe:
            exceeded_by = ((elem.concentration_ppm / elem.safe_limit_ppm) - 1) * 100
            print(f"   ❌ {elem.element_name} ({elem.element_symbol}): "
                  f"{elem.concentration_ppm:.3f} ppm")
            print(f"      Safe Limit: {elem.safe_limit_ppm} ppm "
                  f"| EXCEEDED by {exceeded_by:.1f}%")
            print(f"      Health Risk: {elem.health_role}")
    
    print(f"\n🔔 Recommendation: {profile.visual_safety_assessment['recommended_action']}")


def example_4_comparison():
    """
    Example 4: Compare fresh vs aged produce
    
    Use Case: Quality monitoring over time
    Method: Visual comparison
    """
    print("\n\n" + "="*80)
    print("EXAMPLE 4: Fresh vs Aged Kale Comparison")
    print("="*80)
    
    # Fresh kale
    fresh = VisualFeatures(
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
    
    # Aged kale (3 days)
    aged = VisualFeatures(
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
    profile_fresh = detector.analyze_food('kale', visual_features=fresh)
    profile_aged = detector.analyze_food('kale', visual_features=aged)
    
    print("\n📊 COMPARISON TABLE:\n")
    print(f"{'Metric':<35} {'Fresh Kale':<20} {'Aged Kale (3 days)':<20}")
    print("-" * 75)
    
    # Safety
    fresh_safe = "✅ SAFE" if profile_fresh.is_safe_for_consumption else "❌ UNSAFE"
    aged_safe = "✅ SAFE" if profile_aged.is_safe_for_consumption else "⚠️  QUESTIONABLE"
    print(f"{'Safety Status':<35} {fresh_safe:<20} {aged_safe:<20}")
    
    # Scores
    print(f"{'Freshness Score':<35} {profile_fresh.freshness_score:<20.1f} "
          f"{profile_aged.freshness_score:<20.1f}")
    print(f"{'Nutritional Score':<35} {profile_fresh.nutritional_quality_score:<20.1f} "
          f"{profile_aged.nutritional_quality_score:<20.1f}")
    print(f"{'Purity Score':<35} {profile_fresh.purity_score:<20.1f} "
          f"{profile_aged.purity_score:<20.1f}")
    
    if profile_fresh.visual_safety_assessment and profile_aged.visual_safety_assessment:
        print(f"\n{'Visual Assessment:':<35}")
        fresh_rating = profile_fresh.visual_safety_assessment['freshness_rating']
        aged_rating = profile_aged.visual_safety_assessment['freshness_rating']
        print(f"{'  Freshness Rating':<35} {fresh_rating:<20} {aged_rating:<20}")
        
        fresh_safety = profile_fresh.visual_safety_assessment['safety_score']
        aged_safety = profile_aged.visual_safety_assessment['safety_score']
        print(f"{'  Safety Score':<35} {fresh_safety:<20.1f} {aged_safety:<20.1f}")
    
    print(f"\n{'Recommendations:':<35}")
    print(f"  Fresh: {profile_fresh.visual_safety_assessment['recommended_action']}")
    print(f"  Aged:  {profile_aged.visual_safety_assessment['recommended_action']}")
    
    print("\n💡 Insight: Freshness degrades significantly after 3 days.")
    print("   Consider consuming leafy greens within 2-3 days for optimal quality.")


def main():
    """Run all quick start examples"""
    print("\n╔" + "═"*78 + "╗")
    print("║" + " "*20 + "VISUAL-ATOMIC FOOD ANALYSIS" + " "*31 + "║")
    print("║" + " "*28 + "QUICK START GUIDE" + " "*33 + "║")
    print("╚" + "═"*78 + "╝")
    
    print("\nThis guide demonstrates how to use visual features (shininess, reflection,")
    print("color, texture) to predict atomic composition and assess food safety.")
    
    try:
        example_1_fresh_spinach()
        example_2_spinach_with_icpms()
        example_3_contaminated_apple()
        example_4_comparison()
        
        print("\n\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED")
        print("="*80)
        
        print("\n📚 What You Learned:")
        print("   1. Visual-only analysis for quick screening")
        print("   2. ICP-MS validation of visual predictions")
        print("   3. Contamination detection from warning signs")
        print("   4. Freshness comparison over time")
        
        print("\n🚀 Next Steps:")
        print("   • Run full test suite: python test_visual_analysis.py")
        print("   • Read documentation: VISUAL_ATOMIC_ANALYSIS_GUIDE.md")
        print("   • Integrate with your own ICP-MS data")
        print("   • Add camera/image processing for automated feature extraction")
        
        print("\n💡 Pro Tip:")
        print("   Visual analysis is great for SCREENING, but always validate")
        print("   with ICP-MS for heavy metal contamination and precise measurements!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
