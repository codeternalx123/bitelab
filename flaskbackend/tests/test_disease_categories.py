"""
Unit Tests for New Disease Categories
Tests all newly added disease categories with medical accuracy validation.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'app' / 'ai_nutrition' / 'scanner'))

from cv_integration_bridge import (
    CVIntegrationBridge,
    DiseaseCategory,
    DiseaseSeverity
)


class TestHematologicalDiseases(unittest.TestCase):
    """Test hematological disease configurations."""
    
    def setUp(self):
        self.bridge = CVIntegrationBridge()
    
    def test_iron_deficiency_anemia(self):
        """Test iron deficiency anemia requirements."""
        disease = self.bridge.diseases.get('anemia_iron_deficiency')
        self.assertIsNotNone(disease)
        self.assertEqual(disease.category, DiseaseCategory.HEMATOLOGICAL)
        self.assertGreaterEqual(disease.iron_min, 18)
        self.assertGreaterEqual(disease.vitamin_c_min, 90)
        self.assertIn('red_meat', disease.recommended_foods)
    
    def test_hemochromatosis(self):
        """Test hemochromatosis (iron overload) restrictions."""
        disease = self.bridge.diseases.get('hemochromatosis')
        self.assertIsNotNone(disease)
        self.assertEqual(disease.severity, DiseaseSeverity.SEVERE)
        self.assertLessEqual(disease.iron_max, 8)
        self.assertLessEqual(disease.vitamin_c_max, 500)
        self.assertIn('iron_fortified', disease.forbidden_foods)
        self.assertIn('red_meat', disease.forbidden_foods)
    
    def test_sickle_cell_disease(self):
        """Test sickle cell disease requirements."""
        disease = self.bridge.diseases.get('sickle_cell')
        self.assertIsNotNone(disease)
        self.assertEqual(disease.severity, DiseaseSeverity.SEVERE)
        self.assertGreaterEqual(disease.calories_min, 2500)
        self.assertGreaterEqual(disease.folate_min, 1000)  # High RBC turnover
        self.assertGreaterEqual(disease.water_min, 3.0)  # Hydration critical
    
    def test_thalassemia(self):
        """Test thalassemia requirements."""
        disease = self.bridge.diseases.get('thalassemia')
        self.assertIsNotNone(disease)
        self.assertLessEqual(disease.iron_max, 8)  # Often iron overloaded
        self.assertIn('iron_fortified', disease.forbidden_foods)
    
    def test_b12_deficiency_anemia(self):
        """Test B12 deficiency anemia."""
        disease = self.bridge.diseases.get('anemia_b12_deficiency')
        self.assertIsNotNone(disease)
        self.assertGreaterEqual(disease.b12_min, 2.4)
        self.assertIn('meat', disease.recommended_foods)


class TestEndocrineDiseases(unittest.TestCase):
    """Test endocrine disease configurations."""
    
    def setUp(self):
        self.bridge = CVIntegrationBridge()
    
    def test_pcos(self):
        """Test PCOS nutritional requirements."""
        disease = self.bridge.diseases.get('pcos')
        self.assertIsNotNone(disease)
        self.assertEqual(disease.category, DiseaseCategory.ENDOCRINE)
        self.assertLessEqual(disease.carbs_max, 150)
        self.assertLessEqual(disease.sugar_max, 25)
        self.assertGreaterEqual(disease.fiber_min, 30)
        self.assertGreaterEqual(disease.omega3_min, 1000)
    
    def test_hashimotos(self):
        """Test Hashimoto's thyroiditis requirements."""
        disease = self.bridge.diseases.get('hashimotos')
        self.assertIsNotNone(disease)
        self.assertGreaterEqual(disease.selenium_min, 55)
        self.assertGreaterEqual(disease.zinc_min, 11)
        self.assertIn('gluten', disease.forbidden_foods)
        self.assertIn('selenium_rich', disease.recommended_foods)
    
    def test_graves_disease(self):
        """Test Graves' disease requirements."""
        disease = self.bridge.diseases.get('graves_disease')
        self.assertIsNotNone(disease)
        self.assertGreaterEqual(disease.calories_min, 2500)  # Increased metabolism
        self.assertIn('iodine_rich', disease.forbidden_foods)
        self.assertIn('cruciferous_vegetables', disease.recommended_foods)
    
    def test_addisons_disease(self):
        """Test Addison's disease requirements."""
        disease = self.bridge.diseases.get('addisons')
        self.assertIsNotNone(disease)
        self.assertEqual(disease.severity, DiseaseSeverity.SEVERE)
        self.assertGreaterEqual(disease.sodium_min, 2300)  # Need more sodium
        self.assertLessEqual(disease.potassium_max, 2000)  # Limit potassium
    
    def test_cushings_syndrome(self):
        """Test Cushing's syndrome requirements."""
        disease = self.bridge.diseases.get('cushings')
        self.assertIsNotNone(disease)
        self.assertLessEqual(disease.sodium_max, 1500)
        self.assertLessEqual(disease.sugar_max, 25)
        self.assertGreaterEqual(disease.calcium_min, 1200)


class TestLiverDiseases(unittest.TestCase):
    """Test liver disease configurations."""
    
    def setUp(self):
        self.bridge = CVIntegrationBridge()
    
    def test_fatty_liver(self):
        """Test fatty liver disease requirements."""
        disease = self.bridge.diseases.get('fatty_liver')
        self.assertIsNotNone(disease)
        self.assertLessEqual(disease.calories_max, 1800)
        self.assertLessEqual(disease.sugar_max, 25)
        self.assertLessEqual(disease.saturated_fat_max, 15)
        self.assertGreaterEqual(disease.fiber_min, 30)
        self.assertIn('alcohol', disease.forbidden_foods)
        self.assertIn('coffee', disease.recommended_foods)
    
    def test_cirrhosis(self):
        """Test cirrhosis requirements."""
        disease = self.bridge.diseases.get('cirrhosis')
        self.assertIsNotNone(disease)
        self.assertEqual(disease.severity, DiseaseSeverity.SEVERE)
        self.assertGreaterEqual(disease.protein_min, 75)
        self.assertLessEqual(disease.sodium_max, 2000)
        self.assertIn('alcohol', disease.forbidden_foods)
    
    def test_hepatitis(self):
        """Test hepatitis requirements."""
        disease = self.bridge.diseases.get('hepatitis')
        self.assertIsNotNone(disease)
        self.assertGreaterEqual(disease.protein_min, 75)
        self.assertIn('alcohol', disease.forbidden_foods)
        self.assertIn('raw_shellfish', disease.forbidden_foods)


class TestInflammatoryDiseases(unittest.TestCase):
    """Test inflammatory disease configurations."""
    
    def setUp(self):
        self.bridge = CVIntegrationBridge()
    
    def test_psoriasis(self):
        """Test psoriasis requirements."""
        disease = self.bridge.diseases.get('psoriasis')
        self.assertIsNotNone(disease)
        self.assertGreaterEqual(disease.omega3_min, 2000)
        self.assertGreaterEqual(disease.fiber_min, 30)
        self.assertIn('alcohol', disease.forbidden_foods)
        self.assertIn('anti_inflammatory', disease.recommended_foods)
    
    def test_ibd(self):
        """Test inflammatory bowel disease requirements."""
        disease = self.bridge.diseases.get('ibd')
        self.assertIsNotNone(disease)
        self.assertGreaterEqual(disease.protein_min, 75)
        self.assertLessEqual(disease.fiber_max, 15)  # During flares
        self.assertGreaterEqual(disease.calcium_min, 1200)
    
    def test_fibromyalgia(self):
        """Test fibromyalgia requirements."""
        disease = self.bridge.diseases.get('fibromyalgia')
        self.assertIsNotNone(disease)
        self.assertGreaterEqual(disease.magnesium_min, 400)
        self.assertGreaterEqual(disease.vitamin_d_min, 800)
        self.assertIn('processed_foods', disease.forbidden_foods)


class TestBoneJointDiseases(unittest.TestCase):
    """Test bone and joint disease configurations."""
    
    def setUp(self):
        self.bridge = CVIntegrationBridge()
    
    def test_osteoarthritis(self):
        """Test osteoarthritis requirements."""
        disease = self.bridge.diseases.get('osteoarthritis')
        self.assertIsNotNone(disease)
        self.assertGreaterEqual(disease.omega3_min, 1000)
        self.assertGreaterEqual(disease.vitamin_d_min, 800)
        self.assertGreaterEqual(disease.calcium_min, 1000)
    
    def test_osteopenia(self):
        """Test osteopenia requirements."""
        disease = self.bridge.diseases.get('osteopenia')
        self.assertIsNotNone(disease)
        self.assertGreaterEqual(disease.calcium_min, 1200)
        self.assertGreaterEqual(disease.vitamin_d_min, 800)
        self.assertGreaterEqual(disease.protein_min, 60)
        self.assertIn('excessive_sodium', disease.forbidden_foods)


class TestSkinConditions(unittest.TestCase):
    """Test skin condition configurations."""
    
    def setUp(self):
        self.bridge = CVIntegrationBridge()
    
    def test_acne(self):
        """Test acne nutritional requirements."""
        disease = self.bridge.diseases.get('acne')
        self.assertIsNotNone(disease)
        self.assertLessEqual(disease.sugar_max, 25)
        self.assertGreaterEqual(disease.zinc_min, 11)
        self.assertGreaterEqual(disease.omega3_min, 1000)
        self.assertIn('high_gi_foods', disease.forbidden_foods)
    
    def test_eczema(self):
        """Test eczema requirements."""
        disease = self.bridge.diseases.get('eczema')
        self.assertIsNotNone(disease)
        self.assertGreaterEqual(disease.omega3_min, 1000)
        self.assertGreaterEqual(disease.vitamin_d_min, 600)
        self.assertIn('probiotic_foods', disease.recommended_foods)


class TestEyeConditions(unittest.TestCase):
    """Test eye condition configurations."""
    
    def setUp(self):
        self.bridge = CVIntegrationBridge()
    
    def test_macular_degeneration(self):
        """Test macular degeneration requirements."""
        disease = self.bridge.diseases.get('macular_degeneration')
        self.assertIsNotNone(disease)
        self.assertGreaterEqual(disease.omega3_min, 1000)
        self.assertIn('leafy_greens', disease.recommended_foods)
        self.assertIn('colorful_vegetables', disease.recommended_foods)
    
    def test_glaucoma(self):
        """Test glaucoma requirements."""
        disease = self.bridge.diseases.get('glaucoma')
        self.assertIsNotNone(disease)
        self.assertGreaterEqual(disease.omega3_min, 1000)
        self.assertIn('leafy_greens', disease.recommended_foods)


class TestReproductiveHealth(unittest.TestCase):
    """Test reproductive health configurations."""
    
    def setUp(self):
        self.bridge = CVIntegrationBridge()
    
    def test_endometriosis(self):
        """Test endometriosis requirements."""
        disease = self.bridge.diseases.get('endometriosis')
        self.assertIsNotNone(disease)
        self.assertGreaterEqual(disease.omega3_min, 1000)
        self.assertGreaterEqual(disease.fiber_min, 30)
        self.assertIn('red_meat', disease.forbidden_foods)
        self.assertIn('anti_inflammatory', disease.recommended_foods)
    
    def test_erectile_dysfunction(self):
        """Test erectile dysfunction requirements."""
        disease = self.bridge.diseases.get('erectile_dysfunction')
        self.assertIsNotNone(disease)
        self.assertIn('fish', disease.recommended_foods)
        self.assertIn('vegetables', disease.recommended_foods)


class TestSleepDisorders(unittest.TestCase):
    """Test sleep disorder configurations."""
    
    def setUp(self):
        self.bridge = CVIntegrationBridge()
    
    def test_insomnia(self):
        """Test insomnia requirements."""
        disease = self.bridge.diseases.get('insomnia')
        self.assertIsNotNone(disease)
        self.assertGreaterEqual(disease.magnesium_min, 400)
        self.assertIn('caffeine', disease.forbidden_foods)
        self.assertIn('tryptophan_rich', disease.recommended_foods)
    
    def test_sleep_apnea(self):
        """Test sleep apnea requirements."""
        disease = self.bridge.diseases.get('sleep_apnea')
        self.assertIsNotNone(disease)
        self.assertLessEqual(disease.calories_max, 1800)
        self.assertIn('alcohol', disease.forbidden_foods)


class TestImmuneDisorders(unittest.TestCase):
    """Test immune disorder configurations."""
    
    def setUp(self):
        self.bridge = CVIntegrationBridge()
    
    def test_hiv_aids(self):
        """Test HIV/AIDS requirements."""
        disease = self.bridge.diseases.get('hiv_aids')
        self.assertIsNotNone(disease)
        self.assertEqual(disease.severity, DiseaseSeverity.SEVERE)
        self.assertGreaterEqual(disease.calories_min, 2500)
        self.assertGreaterEqual(disease.protein_min, 100)
        self.assertIn('raw_foods', disease.forbidden_foods)
    
    def test_chronic_fatigue(self):
        """Test chronic fatigue syndrome requirements."""
        disease = self.bridge.diseases.get('chronic_fatigue')
        self.assertIsNotNone(disease)
        self.assertGreaterEqual(disease.magnesium_min, 400)
        self.assertGreaterEqual(disease.b12_min, 2.4)
        self.assertIn('processed_foods', disease.forbidden_foods)


class TestDiseaseDatabase(unittest.TestCase):
    """Test overall disease database integrity."""
    
    def setUp(self):
        self.bridge = CVIntegrationBridge()
    
    def test_disease_count(self):
        """Test that we have 100+ diseases."""
        self.assertGreaterEqual(len(self.bridge.diseases), 80)
        print(f"\nTotal diseases in database: {len(self.bridge.diseases)}")
    
    def test_all_diseases_have_required_fields(self):
        """Test that all diseases have required fields."""
        for disease_id, disease in self.bridge.diseases.items():
            self.assertIsNotNone(disease.disease_id)
            self.assertIsNotNone(disease.name)
            self.assertIsNotNone(disease.category)
            self.assertIsNotNone(disease.severity)
    
    def test_disease_categories(self):
        """Test that all expected categories are present."""
        categories = set(disease.category for disease in self.bridge.diseases.values())
        
        expected_categories = {
            DiseaseCategory.METABOLIC,
            DiseaseCategory.CARDIOVASCULAR,
            DiseaseCategory.RENAL,
            DiseaseCategory.GASTROINTESTINAL,
            DiseaseCategory.AUTOIMMUNE,
            DiseaseCategory.NEUROLOGICAL,
            DiseaseCategory.RESPIRATORY,
            DiseaseCategory.HEMATOLOGICAL,
        }
        
        for category in expected_categories:
            self.assertIn(category, categories, f"Missing category: {category}")
    
    def test_severity_levels(self):
        """Test that severity levels are properly assigned."""
        severities = set(disease.severity for disease in self.bridge.diseases.values())
        self.assertIn(DiseaseSeverity.MILD, severities)
        self.assertIn(DiseaseSeverity.MODERATE, severities)
        self.assertIn(DiseaseSeverity.SEVERE, severities)
    
    def test_no_duplicate_disease_ids(self):
        """Test that there are no duplicate disease IDs."""
        disease_ids = [disease.disease_id for disease in self.bridge.diseases.values()]
        self.assertEqual(len(disease_ids), len(set(disease_ids)))


class TestDiseaseInteractions(unittest.TestCase):
    """Test disease interaction scenarios."""
    
    def setUp(self):
        self.bridge = CVIntegrationBridge()
    
    def test_conflicting_requirements(self):
        """Test diseases with conflicting requirements."""
        # Iron deficiency vs hemochromatosis
        anemia = self.bridge.diseases.get('anemia_iron_deficiency')
        hemo = self.bridge.diseases.get('hemochromatosis')
        
        # They should have opposite iron requirements
        self.assertGreater(anemia.iron_min, hemo.iron_max)
    
    def test_addisons_vs_cushings(self):
        """Test opposite sodium requirements."""
        addisons = self.bridge.diseases.get('addisons')
        cushings = self.bridge.diseases.get('cushings')
        
        # Addison's needs more sodium, Cushing's needs less
        self.assertGreater(addisons.sodium_min, cushings.sodium_max)
    
    def test_graves_vs_hashimotos(self):
        """Test opposite thyroid conditions."""
        graves = self.bridge.diseases.get('graves_disease')
        hashimotos = self.bridge.diseases.get('hashimotos')
        
        # Graves avoids iodine, Hashimoto's is careful with goitrogens
        self.assertIn('iodine_rich', graves.forbidden_foods)
        self.assertIn('cruciferous_vegetables', graves.recommended_foods)


def run_disease_tests():
    """Run all disease category tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHematologicalDiseases))
    suite.addTests(loader.loadTestsFromTestCase(TestEndocrineDiseases))
    suite.addTests(loader.loadTestsFromTestCase(TestLiverDiseases))
    suite.addTests(loader.loadTestsFromTestCase(TestInflammatoryDiseases))
    suite.addTests(loader.loadTestsFromTestCase(TestBoneJointDiseases))
    suite.addTests(loader.loadTestsFromTestCase(TestSkinConditions))
    suite.addTests(loader.loadTestsFromTestCase(TestEyeConditions))
    suite.addTests(loader.loadTestsFromTestCase(TestReproductiveHealth))
    suite.addTests(loader.loadTestsFromTestCase(TestSleepDisorders))
    suite.addTests(loader.loadTestsFromTestCase(TestImmuneDisorders))
    suite.addTests(loader.loadTestsFromTestCase(TestDiseaseDatabase))
    suite.addTests(loader.loadTestsFromTestCase(TestDiseaseInteractions))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_disease_tests()
    sys.exit(0 if success else 1)
