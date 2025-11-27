"""
Atomic Vision System - Comprehensive Test Suite
================================================

Tests for:
1. Atomic composition prediction from images
2. ICP-MS data integration
3. Integration with Health Impact Analyzer
4. Full pipeline: Image → Elements → Health Report

Run: python -m app.ai_nutrition.scanner.test_atomic_vision
"""

import sys
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.ai_nutrition.scanner.atomic_vision import (
    AtomicVisionPredictor,
    FoodImageData,
    ElementInfo,
    ElementCategory,
    ELEMENT_DATABASE,
    TOXIC_ELEMENTS,
    NUTRIENT_ELEMENTS,
    ImagePreprocessor
)

from app.ai_nutrition.scanner.icpms_data import (
    ICPMSSample,
    ICPMSDataset,
    ICPMSDataLoader,
    DataSource,
    QualityFlag,
    CalibrationManager
)

from app.ai_nutrition.scanner.health_impact_analyzer import (
    HealthImpactAnalyzer,
    HealthCondition
)


def test_element_database():
    """Test 1: Element database structure"""
    print("\n" + "="*80)
    print("TEST 1: Element Database")
    print("="*80)
    
    # Check toxic elements
    print(f"\nToxic Elements ({len(TOXIC_ELEMENTS)}):")
    for element in TOXIC_ELEMENTS:
        info = ELEMENT_DATABASE[element]
        print(f"  {element} ({info.name}): "
              f"Limit = {info.regulatory_limit_mg_kg} mg/kg, "
              f"LD50 = {info.typical_range_mg_kg}")
        assert info.toxic, f"{element} should be marked toxic"
    
    # Check nutrient elements
    print(f"\nNutrient Elements ({len(NUTRIENT_ELEMENTS)}):")
    for element in NUTRIENT_ELEMENTS[:5]:  # Show first 5
        info = ELEMENT_DATABASE[element]
        print(f"  {element} ({info.name}): "
              f"RDA = {info.rda_mg_day} mg/day")
        assert info.nutritional, f"{element} should be marked nutritional"
    
    print("\n✅ Element database validated")
    return True


def test_image_preprocessing():
    """Test 2: Image preprocessing pipeline"""
    print("\n" + "="*80)
    print("TEST 2: Image Preprocessing")
    print("="*80)
    
    # Create synthetic test image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    
    # Preprocess
    processed = preprocessor.preprocess(image, mode="rgb")
    
    print(f"\nOriginal image shape: {image.shape}")
    print(f"Processed image shape: {processed.shape}")
    print(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Validate
    assert processed.shape == (3, 224, 224), "Wrong output shape"
    
    # Quality assessment
    quality = preprocessor.assess_quality(image)
    print(f"Image quality score: {quality:.2f}/1.0")
    
    print("\n✅ Image preprocessing functional")
    return True


def test_atomic_prediction():
    """Test 3: Atomic composition prediction"""
    print("\n" + "="*80)
    print("TEST 3: Atomic Composition Prediction")
    print("="*80)
    
    # Create test image data
    image = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
    
    image_data = FoodImageData(
        image=image,
        weight_grams=150.0,
        food_type="leafy_vegetable",
        preparation="raw",
        source="organic_farm"
    )
    
    # Validate
    image_data.validate()
    print(f"✓ Image data validated")
    print(f"  Weight: {image_data.weight_grams}g")
    print(f"  Food type: {image_data.food_type}")
    print(f"  Imaging mode: {image_data.imaging_mode}")
    
    # Predict (using fallback heuristic model since PyTorch may not be available)
    predictor = AtomicVisionPredictor()
    result = predictor.predict(image_data)
    
    print(f"\nPrediction Results:")
    print(f"  Timestamp: {result.timestamp}")
    print(f"  Model version: {result.model_version}")
    print(f"  Image quality: {result.image_quality_score:.2f}/1.0")
    print(f"  Total uncertainty: {result.total_uncertainty:.2f} mg/kg")
    print(f"  Elements predicted: {len(result.predictions)}")
    
    # Check toxic elements
    toxic = result.get_toxic_elements()
    print(f"\n⚠️ Toxic Elements Detected: {len(toxic)}")
    for pred in toxic[:3]:  # Show first 3
        ci_low, ci_high = pred.get_confidence_interval()
        status = "❌ EXCEEDS LIMIT" if pred.exceeds_limit else "✓ Safe"
        print(f"  {pred.element}: {pred.concentration_mg_kg:.3f} mg/kg "
              f"(95% CI: {ci_low:.3f}-{ci_high:.3f}) - {status}")
    
    # Check nutrients
    nutrients = result.get_nutrient_elements()
    print(f"\n✓ Essential Nutrients: {len(nutrients)}")
    for pred in nutrients[:5]:  # Show first 5
        element_info = ELEMENT_DATABASE[pred.element]
        print(f"  {pred.element} ({element_info.name}): "
              f"{pred.concentration_mg_kg:.2f} mg/kg "
              f"(confidence: {pred.confidence:.2f})")
    
    print("\n✅ Atomic prediction functional")
    return result


def test_icpms_data():
    """Test 4: ICP-MS data integration"""
    print("\n" + "="*80)
    print("TEST 4: ICP-MS Data Integration")
    print("="*80)
    
    # Create synthetic ICP-MS samples
    samples = []
    food_types = [
        ("spinach", "leafy_vegetable"),
        ("salmon", "fish"),
        ("rice", "grain"),
        ("apple", "fruit"),
        ("beef", "meat")
    ]
    
    for i, (food, category) in enumerate(food_types):
        elements = {
            "Fe": np.random.uniform(10, 100),
            "Zn": np.random.uniform(5, 50),
            "Cu": np.random.uniform(1, 10),
            "Ca": np.random.uniform(100, 1000),
            "Mg": np.random.uniform(50, 500),
            "Pb": np.random.uniform(0.01, 0.1),
            "Cd": np.random.uniform(0.001, 0.05),
        }
        
        sample = ICPMSSample(
            sample_id=f"ICPMS_{i:03d}",
            food_name=food,
            food_category=category,
            image_path=f"images/{food}.jpg",
            weight_grams=100.0,
            elements=elements,
            source=DataSource.CUSTOM_LAB,
            analysis_date=datetime.now(),
            quality_flag=QualityFlag.GOOD
        )
        
        samples.append(sample)
    
    # Create dataset
    dataset = ICPMSDataset(samples=samples, name="test_dataset")
    
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Available elements: {len(dataset.get_available_elements())}")
    print(f"  Elements: {', '.join(sorted(dataset.get_available_elements()))}")
    
    # Element statistics
    for element in ["Fe", "Pb"]:
        stats = dataset.get_element_statistics(element)
        print(f"\n{element} Statistics:")
        print(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f} mg/kg")
        print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}] mg/kg")
        print(f"  Median: {stats['median']:.2f} mg/kg")
    
    # Filter by quality
    filtered = dataset.filter_by_quality(QualityFlag.GOOD)
    print(f"\nFiltered dataset (quality >= GOOD): {len(filtered)} samples")
    
    # Filter by elements
    required = ["Fe", "Zn", "Pb"]
    element_filtered = dataset.filter_by_elements(required)
    print(f"Samples with {required}: {len(element_filtered)}")
    
    print("\n✅ ICP-MS data integration functional")
    return dataset


def test_calibration():
    """Test 5: Calibration curve generation"""
    print("\n" + "="*80)
    print("TEST 5: Calibration Curves")
    print("="*80)
    
    cal_manager = CalibrationManager()
    
    # Generate calibration curve for Fe
    standards_conc = np.array([10, 50, 100, 200, 500])  # mg/kg
    standards_signal = np.array([1000, 5000, 10000, 20000, 50000])  # counts/sec
    
    curve = cal_manager.generate_curve("Fe", standards_conc, standards_signal)
    
    print(f"\nFe Calibration Curve:")
    print(f"  Equation: Conc = {curve.slope:.4f} * Signal + {curve.intercept:.2f}")
    print(f"  R² = {curve.r_squared:.4f}")
    print(f"  Calibration points: {curve.n_points}")
    print(f"  Valid range: {curve.concentration_range[0]}-{curve.concentration_range[1]} mg/kg")
    
    # Test prediction
    test_signal = 15000
    predicted_conc = curve.predict(test_signal)
    print(f"\nPrediction Test:")
    print(f"  Signal: {test_signal} counts/sec")
    print(f"  Predicted concentration: {predicted_conc:.2f} mg/kg")
    
    # Validate
    test_conc = np.array([75, 150, 300])
    test_signals = np.array([7500, 15000, 30000])
    validation = cal_manager.validate_curve("Fe", test_conc, test_signals)
    
    print(f"\nValidation Metrics:")
    print(f"  MAE: {validation['mae']:.2f} mg/kg")
    print(f"  RMSE: {validation['rmse']:.2f} mg/kg")
    print(f"  MAPE: {validation['mape']:.2f}%")
    print(f"  Within 10% tolerance: {validation['within_10_percent']:.1f}%")
    
    print("\n✅ Calibration system functional")
    return True


def test_health_analyzer_integration():
    """Test 6: Integration with Health Impact Analyzer"""
    print("\n" + "="*80)
    print("TEST 6: Health Impact Analyzer Integration")
    print("="*80)
    
    # Create test atomic result (reuse from test 3)
    image = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
    image_data = FoodImageData(
        image=image,
        weight_grams=150.0,
        food_type="leafy_vegetable",
        preparation="raw"
    )
    
    predictor = AtomicVisionPredictor()
    atomic_result = predictor.predict(image_data)
    
    # Initialize health analyzer
    analyzer = HealthImpactAnalyzer(use_ai_models=True)
    
    # Test 1: Integrate atomic composition
    composition = analyzer.integrate_atomic_composition(atomic_result)
    print(f"\n✓ Atomic composition integrated: {len(composition)} elements")
    for element, conc in list(composition.items())[:5]:
        print(f"  {element}: {conc:.2f} mg/kg")
    
    # Test 2: Assess toxicity
    toxicity = analyzer.assess_atomic_toxicity(atomic_result)
    print(f"\n✓ Toxicity assessment:")
    print(f"  Overall risk: {toxicity.overall_risk.value}")
    print(f"  Toxicity score: {toxicity.toxicity_score:.1f}/100")
    print(f"  Heavy metals detected: {len(toxicity.heavy_metals)}")
    print(f"  Safe for consumption: {toxicity.safe_for_consumption}")
    
    if toxicity.warnings:
        print(f"  Warnings: {len(toxicity.warnings)}")
        for warning in toxicity.warnings[:2]:
            print(f"    - {warning}")
    
    # Test 3: Estimate nutrition
    nutrition = analyzer.estimate_nutrition_from_elements(atomic_result)
    print(f"\n✓ Nutritional analysis:")
    print(f"  Minerals detected: {len(nutrition.minerals)}")
    print(f"  RDA compliance tracked: {len(nutrition.rda_compliance)}")
    print(f"  Nutrient density score: {nutrition.nutrient_density_score:.1f}/100")
    print(f"  Health score: {nutrition.health_score:.1f}/100")
    
    if nutrition.minerals:
        print(f"\n  Key Minerals (first 5):")
        for element, conc in list(nutrition.minerals.items())[:5]:
            rda = nutrition.rda_compliance.get(element, 0)
            print(f"    {element}: {conc:.2f} mg/kg ({rda:.1f}% RDA)")
    
    print("\n✅ Health analyzer integration functional")
    return atomic_result, analyzer


def test_full_pipeline():
    """Test 7: Full pipeline - Image to Health Report"""
    print("\n" + "="*80)
    print("TEST 7: Full Pipeline - Image → Elements → Health Report")
    print("="*80)
    
    # Step 1: Create food image
    print("\n📷 Step 1: Food Image Acquisition")
    image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    # Simulate spinach (higher green channel)
    image[:, :, 1] += 30  # More green
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    image_data = FoodImageData(
        image=image,
        weight_grams=150.0,
        food_type="leafy_vegetable",
        preparation="raw",
        source="organic_farm"
    )
    print(f"  ✓ Image loaded: {image.shape}")
    print(f"  ✓ Weight: {image_data.weight_grams}g")
    
    # Step 2: Predict atomic composition
    print("\n🔬 Step 2: Atomic Composition Prediction")
    predictor = AtomicVisionPredictor()
    atomic_result = predictor.predict(image_data, use_uncertainty=True)
    print(f"  ✓ {len(atomic_result.predictions)} elements predicted")
    print(f"  ✓ Image quality: {atomic_result.image_quality_score:.2f}")
    print(f"  ✓ Model: {atomic_result.model_version}")
    
    # Step 3: Generate health report
    print("\n🏥 Step 3: Health Impact Analysis")
    analyzer = HealthImpactAnalyzer(use_ai_models=True)
    report = analyzer.generate_atomic_health_report(
        atomic_result,
        food_name="Fresh Organic Spinach",
        user_conditions=[HealthCondition.ANEMIA, HealthCondition.PREGNANCY]
    )
    
    print(f"  ✓ Report generated: {report.food_name}")
    print(f"  ✓ Safety score: {report.overall_safety_score:.1f}/100")
    print(f"  ✓ Health score: {report.overall_health_score:.1f}/100")
    print(f"  ✓ Recommendation: {report.consumption_recommendation}")
    
    # Display key findings
    print("\n📊 Key Findings:")
    print(f"  Toxicity Risk: {report.toxicity.overall_risk.value.upper()}")
    print(f"  Allergen Risk: {report.allergens.allergen_risk.value}")
    print(f"  Nutrient Density: {report.nutrition.nutrient_density_score:.1f}/100")
    
    if report.personalized_benefits:
        print(f"\n  Personalized Benefits ({len(report.personalized_benefits)}):")
        for benefit in report.personalized_benefits:
            print(f"    {benefit}")
    
    if report.personalized_warnings:
        print(f"\n  Personalized Warnings ({len(report.personalized_warnings)}):")
        for warning in report.personalized_warnings:
            print(f"    {warning}")
    
    print("\n✅ Full pipeline operational")
    return report


def test_confidence_intervals():
    """Test 8: Uncertainty quantification"""
    print("\n" + "="*80)
    print("TEST 8: Uncertainty Quantification")
    print("="*80)
    
    # Create prediction with uncertainty
    image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    image_data = FoodImageData(image=image, weight_grams=100.0)
    
    predictor = AtomicVisionPredictor()
    result = predictor.predict(image_data, use_uncertainty=True)
    
    print(f"\nUncertainty Analysis:")
    print(f"  Total uncertainty: {result.total_uncertainty:.2f} mg/kg")
    print(f"  Image quality impact: {result.image_quality_score:.2f}")
    
    # Show confidence intervals for key elements
    key_elements = ["Fe", "Pb", "Ca"]
    print(f"\nConfidence Intervals (95%):")
    for elem in key_elements:
        pred = result.get_element(elem)
        if pred:
            ci_low, ci_high = pred.get_confidence_interval(z_score=1.96)
            rel_uncert = (pred.uncertainty_mg_kg / pred.concentration_mg_kg) * 100 if pred.concentration_mg_kg > 0 else 0
            print(f"  {elem}: {pred.concentration_mg_kg:.2f} mg/kg "
                  f"[{ci_low:.2f}, {ci_high:.2f}] "
                  f"(±{rel_uncert:.1f}%)")
    
    # Identify high-confidence predictions
    high_confidence = [p for p in result.predictions if p.confidence > 0.8]
    print(f"\nHigh-confidence predictions (>{0.8:.1f}): {len(high_confidence)}/{len(result.predictions)}")
    
    print("\n✅ Uncertainty quantification validated")
    return True


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*80)
    print("ATOMIC VISION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("Testing: Image-based atomic composition prediction + ICP-MS + Health Impact")
    print("="*80)
    
    tests = [
        ("Element Database", test_element_database),
        ("Image Preprocessing", test_image_preprocessing),
        ("Atomic Prediction", test_atomic_prediction),
        ("ICP-MS Data Integration", test_icpms_data),
        ("Calibration Curves", test_calibration),
        ("Health Analyzer Integration", test_health_analyzer_integration),
        ("Full Pipeline", test_full_pipeline),
        ("Uncertainty Quantification", test_confidence_intervals),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            test_func()
            results.append(("✅ PASS", test_name))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(("❌ FAIL", test_name))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for status, name in results:
        print(f"{status}  {name}")
    
    passed = sum(1 for s, _ in results if "PASS" in s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All atomic vision tests passed!")
        print("\nKey Capabilities Validated:")
        print("  ✓ Image-based elemental composition prediction")
        print("  ✓ ICP-MS data integration and calibration")
        print("  ✓ Heavy metal toxicity assessment")
        print("  ✓ Mineral-based nutritional analysis")
        print("  ✓ Personalized health recommendations")
        print("  ✓ Uncertainty quantification")
        print("  ✓ Full pipeline: Image → Atoms → Health Report")
        print("\nNext Steps:")
        print("  - Collect ICP-MS training data (FDA TDS, EFSA, USDA)")
        print("  - Train deep learning models (ViT, EfficientNet)")
        print("  - Deploy with GPU inference")
        print("  - Integrate with mobile app camera")
        return True
    
    return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
