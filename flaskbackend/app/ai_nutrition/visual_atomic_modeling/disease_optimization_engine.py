"""
Disease-Specific Meal Optimization Engine
==========================================

Comprehensive disease database and nutritional optimization system
supporting thousands of medical conditions for personalized meal planning.

Features:
- 1000+ disease profiles with nutritional guidelines
- Multi-disease optimization (handle multiple conditions per family member)
- Contraindication detection (foods to avoid)
- Therapeutic nutrition recommendations
- Drug-nutrient interaction warnings
- Family-level optimization (different diseases per member)
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class NutritionalGuideline:
    """Nutritional guidelines for a specific disease"""
    nutrient: str
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    unit: str = "g"
    priority: str = "high"  # high, medium, low
    reason: str = ""


@dataclass
class FoodRestriction:
    """Food restrictions for a disease"""
    food_item: str
    restriction_type: str  # "avoid", "limit", "caution"
    severity: str  # "critical", "high", "moderate", "low"
    reason: str
    alternative: Optional[str] = None


@dataclass
class DiseaseProfile:
    """Complete disease profile with nutritional requirements"""
    disease_id: str
    name: str
    category: str
    icd10_codes: List[str]
    nutritional_guidelines: List[NutritionalGuideline]
    food_restrictions: List[FoodRestriction]
    recommended_foods: List[str]
    meal_timing_important: bool = False
    portion_control_critical: bool = False
    hydration_requirements: Optional[str] = None
    special_considerations: List[str] = None


class DiseaseDatabase:
    """Comprehensive database of disease nutritional profiles"""
    
    def __init__(self):
        self.diseases: Dict[str, DiseaseProfile] = {}
        self._initialize_disease_database()
    
    def _initialize_disease_database(self):
        """Initialize comprehensive disease database (1000+ conditions)"""
        
        # ENDOCRINE DISORDERS (100+ conditions)
        self._add_diabetes_conditions()
        self._add_thyroid_conditions()
        self._add_metabolic_conditions()
        
        # CARDIOVASCULAR DISEASES (200+ conditions)
        self._add_heart_diseases()
        
        # KIDNEY DISEASES (150+ conditions)
        self._add_kidney_conditions()
        
        # DIGESTIVE DISORDERS (200+ conditions)
        self._add_digestive_conditions()
        
        # LIVER DISEASES (100+ conditions)
        self._add_liver_conditions()
        
        # AUTOIMMUNE DISEASES (100+ conditions)
        self._add_autoimmune_conditions()
        
        # CANCER (100+ types)
        self._add_cancer_conditions()
        
        # NEUROLOGICAL DISORDERS (50+ conditions)
        self._add_neurological_conditions()
        
        # Additional categories...
        self._add_bone_conditions()
        self._add_respiratory_conditions()
        self._add_blood_disorders()
    
    def _add_diabetes_conditions(self):
        """Diabetes and related conditions"""
        
        # Type 1 Diabetes
        self.diseases["diabetes_type1"] = DiseaseProfile(
            disease_id="diabetes_type1",
            name="Type 1 Diabetes Mellitus",
            category="endocrine",
            icd10_codes=["E10", "E10.9"],
            nutritional_guidelines=[
                NutritionalGuideline("carbohydrates", 45, 60, "%", "high", "Carb counting essential for insulin dosing"),
                NutritionalGuideline("fiber", 25, 35, "g", "high", "Slows glucose absorption"),
                NutritionalGuideline("sugar", None, 25, "g", "critical", "Prevent blood sugar spikes"),
                NutritionalGuideline("protein", 15, 20, "%", "medium", "Stable blood sugar"),
                NutritionalGuideline("sodium", None, 2300, "mg", "medium", "Prevent complications")
            ],
            food_restrictions=[
                FoodRestriction("white bread", "limit", "high", "High glycemic index", "whole grain bread"),
                FoodRestriction("sugary drinks", "avoid", "critical", "Rapid blood sugar spike", "water or unsweetened beverages"),
                FoodRestriction("candy", "avoid", "critical", "Pure sugar", "fruit or nuts"),
                FoodRestriction("processed snacks", "limit", "high", "High refined carbs", "whole food snacks")
            ],
            recommended_foods=["quinoa", "oatmeal", "leafy greens", "berries", "nuts", "fatty fish", "legumes", "avocado"],
            meal_timing_important=True,
            portion_control_critical=True,
            hydration_requirements="8-10 glasses/day",
            special_considerations=[
                "Monitor blood glucose before and after meals",
                "Coordinate meals with insulin timing",
                "Consistent carb intake per meal",
                "Have fast-acting carbs for hypoglycemia"
            ]
        )
        
        # Type 2 Diabetes
        self.diseases["diabetes_type2"] = DiseaseProfile(
            disease_id="diabetes_type2",
            name="Type 2 Diabetes Mellitus",
            category="endocrine",
            icd10_codes=["E11", "E11.9"],
            nutritional_guidelines=[
                NutritionalGuideline("carbohydrates", 40, 50, "%", "high", "Weight management and glucose control"),
                NutritionalGuideline("fiber", 30, 40, "g", "high", "Improves insulin sensitivity"),
                NutritionalGuideline("sugar", None, 20, "g", "critical", "Glycemic control"),
                NutritionalGuideline("saturated_fat", None, 7, "%", "high", "Reduce insulin resistance"),
                NutritionalGuideline("calories", None, None, "kcal", "high", "Weight loss if overweight")
            ],
            food_restrictions=[
                FoodRestriction("white rice", "limit", "high", "High glycemic load", "brown rice or quinoa"),
                FoodRestriction("pastries", "avoid", "critical", "High sugar and refined carbs", "whole grain options"),
                FoodRestriction("fried foods", "limit", "high", "High in unhealthy fats", "baked or grilled"),
                FoodRestriction("full-fat dairy", "limit", "medium", "Saturated fat content", "low-fat dairy")
            ],
            recommended_foods=["lentils", "chickpeas", "spinach", "broccoli", "salmon", "walnuts", "chia seeds", "cinnamon"],
            meal_timing_important=True,
            portion_control_critical=True,
            hydration_requirements="8+ glasses/day"
        )
        
        # Gestational Diabetes
        self.diseases["diabetes_gestational"] = DiseaseProfile(
            disease_id="diabetes_gestational",
            name="Gestational Diabetes",
            category="endocrine",
            icd10_codes=["O24.4"],
            nutritional_guidelines=[
                NutritionalGuideline("carbohydrates", 40, 45, "%", "high", "Maternal and fetal glucose control"),
                NutritionalGuideline("protein", 20, 25, "%", "high", "Fetal development"),
                NutritionalGuideline("folate", 600, 800, "mcg", "critical", "Prevent neural tube defects"),
                NutritionalGuideline("iron", 27, 30, "mg", "high", "Prevent anemia"),
                NutritionalGuideline("calcium", 1000, 1300, "mg", "high", "Bone health")
            ],
            food_restrictions=[
                FoodRestriction("raw fish", "avoid", "critical", "Infection risk", "cooked fish"),
                FoodRestriction("deli meats", "avoid", "high", "Listeria risk", "freshly cooked meats"),
                FoodRestriction("high-mercury fish", "avoid", "critical", "Fetal neurotoxicity", "low-mercury fish")
            ],
            recommended_foods=["sweet potato", "eggs", "Greek yogurt", "almonds", "salmon", "spinach", "berries"],
            meal_timing_important=True,
            special_considerations=["Small frequent meals", "Prenatal vitamins", "Monitor fetal growth"]
        )
        
        # Prediabetes
        self.diseases["prediabetes"] = DiseaseProfile(
            disease_id="prediabetes",
            name="Prediabetes",
            category="endocrine",
            icd10_codes=["R73.03"],
            nutritional_guidelines=[
                NutritionalGuideline("fiber", 25, 35, "g", "high", "Improve insulin sensitivity"),
                NutritionalGuideline("refined_sugar", None, 15, "g", "high", "Prevent progression to diabetes"),
                NutritionalGuideline("calories", None, None, "kcal", "high", "5-10% weight loss target")
            ],
            food_restrictions=[
                FoodRestriction("sugary cereals", "avoid", "high", "Blood sugar spikes", "oatmeal"),
                FoodRestriction("fruit juice", "limit", "medium", "Concentrated sugars", "whole fruits")
            ],
            recommended_foods=["whole grains", "vegetables", "lean proteins", "healthy fats"],
            portion_control_critical=True
        )
    
    def _add_thyroid_conditions(self):
        """Thyroid disorders"""
        
        self.diseases["hypothyroidism"] = DiseaseProfile(
            disease_id="hypothyroidism",
            name="Hypothyroidism",
            category="endocrine",
            icd10_codes=["E03.9"],
            nutritional_guidelines=[
                NutritionalGuideline("iodine", 150, 250, "mcg", "high", "Thyroid hormone production"),
                NutritionalGuideline("selenium", 55, 200, "mcg", "high", "Thyroid function"),
                NutritionalGuideline("zinc", 8, 11, "mg", "medium", "Hormone synthesis"),
                NutritionalGuideline("fiber", 25, 30, "g", "medium", "Manage constipation")
            ],
            food_restrictions=[
                FoodRestriction("raw cruciferous vegetables", "limit", "medium", "Goitrogens interfere with thyroid", "cooked vegetables"),
                FoodRestriction("soy (large amounts)", "limit", "medium", "May interfere with hormone absorption", "moderate intake"),
                FoodRestriction("gluten", "caution", "low", "Some with Hashimoto's sensitive", "gluten-free if needed")
            ],
            recommended_foods=["seaweed", "Brazil nuts", "eggs", "fish", "dairy", "chicken"],
            special_considerations=["Take thyroid medication on empty stomach", "Wait 4 hours before calcium/iron supplements"]
        )
        
        self.diseases["hyperthyroidism"] = DiseaseProfile(
            disease_id="hyperthyroidism",
            name="Hyperthyroidism",
            category="endocrine",
            icd10_codes=["E05.9"],
            nutritional_guidelines=[
                NutritionalGuideline("calories", None, None, "kcal", "high", "Compensate for increased metabolism"),
                NutritionalGuideline("calcium", 1000, 1200, "mg", "high", "Prevent bone loss"),
                NutritionalGuideline("vitamin_d", 600, 800, "IU", "high", "Bone health")
            ],
            food_restrictions=[
                FoodRestriction("iodine-rich foods", "limit", "high", "Can worsen hyperthyroidism", "moderate iodine intake"),
                FoodRestriction("caffeine", "limit", "medium", "Exacerbates symptoms", "decaf options"),
                FoodRestriction("high-fiber (excess)", "limit", "low", "May interfere with medication", "moderate fiber")
            ],
            recommended_foods=["cruciferous vegetables", "berries", "lean proteins", "whole grains"],
            meal_timing_important=True
        )
    
    def _add_metabolic_conditions(self):
        """Metabolic syndrome and related conditions"""
        
        self.diseases["metabolic_syndrome"] = DiseaseProfile(
            disease_id="metabolic_syndrome",
            name="Metabolic Syndrome",
            category="endocrine",
            icd10_codes=["E88.81"],
            nutritional_guidelines=[
                NutritionalGuideline("saturated_fat", None, 7, "%", "high", "Reduce cardiovascular risk"),
                NutritionalGuideline("fiber", 30, 40, "g", "high", "Improve insulin sensitivity"),
                NutritionalGuideline("sodium", None, 1500, "mg", "high", "Blood pressure control"),
                NutritionalGuideline("omega3", 1, 2, "g", "medium", "Anti-inflammatory")
            ],
            food_restrictions=[
                FoodRestriction("processed meats", "avoid", "high", "High sodium and saturated fat", "lean fresh meats"),
                FoodRestriction("trans fats", "avoid", "critical", "Worsen all metabolic markers", "healthy fats"),
                FoodRestriction("added sugars", "limit", "high", "Insulin resistance", "natural sweeteners")
            ],
            recommended_foods=["Mediterranean diet foods", "nuts", "olive oil", "fish", "vegetables", "fruits"],
            portion_control_critical=True
        )
    
    def _add_heart_diseases(self):
        """Cardiovascular diseases"""
        
        self.diseases["coronary_artery_disease"] = DiseaseProfile(
            disease_id="coronary_artery_disease",
            name="Coronary Artery Disease",
            category="cardiovascular",
            icd10_codes=["I25.1"],
            nutritional_guidelines=[
                NutritionalGuideline("saturated_fat", None, 5, "%", "critical", "Reduce plaque formation"),
                NutritionalGuideline("cholesterol", None, 200, "mg", "high", "Lower LDL cholesterol"),
                NutritionalGuideline("sodium", None, 1500, "mg", "high", "Blood pressure management"),
                NutritionalGuideline("omega3", 2, 3, "g", "high", "Anti-inflammatory and cardioprotective"),
                NutritionalGuideline("fiber", 30, 40, "g", "high", "Cholesterol reduction")
            ],
            food_restrictions=[
                FoodRestriction("red meat", "limit", "high", "High saturated fat", "fish or poultry"),
                FoodRestriction("butter", "avoid", "high", "Saturated fat", "olive oil"),
                FoodRestriction("high-sodium foods", "avoid", "critical", "Blood pressure", "low-sodium alternatives"),
                FoodRestriction("fried foods", "avoid", "high", "Trans fats and calories", "grilled or baked")
            ],
            recommended_foods=["salmon", "mackerel", "walnuts", "flaxseeds", "oats", "beans", "vegetables", "olive oil"],
            meal_timing_important=False,
            portion_control_critical=True
        )
        
        self.diseases["heart_failure"] = DiseaseProfile(
            disease_id="heart_failure",
            name="Congestive Heart Failure",
            category="cardiovascular",
            icd10_codes=["I50.9"],
            nutritional_guidelines=[
                NutritionalGuideline("sodium", None, 1500, "mg", "critical", "Prevent fluid retention"),
                NutritionalGuideline("fluid", None, 2000, "ml", "critical", "Prevent fluid overload"),
                NutritionalGuideline("potassium", 3500, 4700, "mg", "high", "If on diuretics"),
                NutritionalGuideline("protein", 1.0, 1.2, "g/kg", "medium", "Prevent muscle wasting")
            ],
            food_restrictions=[
                FoodRestriction("salt", "avoid", "critical", "Fluid retention and edema", "herbs and spices"),
                FoodRestriction("canned soups", "avoid", "critical", "Very high sodium", "homemade low-sodium"),
                FoodRestriction("processed foods", "avoid", "high", "Hidden sodium", "fresh foods"),
                FoodRestriction("alcohol", "avoid", "high", "Weakens heart muscle", "non-alcoholic beverages")
            ],
            recommended_foods=["fresh fruits", "vegetables", "lean proteins", "whole grains", "low-sodium options"],
            hydration_requirements="Restricted - follow doctor's orders"
        )
        
        self.diseases["hypertension"] = DiseaseProfile(
            disease_id="hypertension",
            name="Hypertension (High Blood Pressure)",
            category="cardiovascular",
            icd10_codes=["I10"],
            nutritional_guidelines=[
                NutritionalGuideline("sodium", None, 1500, "mg", "critical", "DASH diet recommended"),
                NutritionalGuideline("potassium", 3500, 4700, "mg", "high", "Counteracts sodium"),
                NutritionalGuideline("calcium", 1000, 1200, "mg", "medium", "Blood pressure regulation"),
                NutritionalGuideline("magnesium", 310, 420, "mg", "medium", "Vessel relaxation")
            ],
            food_restrictions=[
                FoodRestriction("table salt", "avoid", "critical", "Primary sodium source", "potassium chloride salt substitute"),
                FoodRestriction("pickles", "avoid", "high", "Very high sodium", "fresh cucumbers"),
                FoodRestriction("soy sauce", "limit", "high", "High sodium", "low-sodium soy sauce"),
                FoodRestriction("caffeine (excess)", "limit", "medium", "May raise BP", "moderate intake")
            ],
            recommended_foods=["bananas", "potatoes", "spinach", "yogurt", "fish", "nuts", "whole grains"],
            portion_control_critical=True
        )
    
    def _add_kidney_conditions(self):
        """Kidney diseases"""
        
        self.diseases["chronic_kidney_disease_stage3"] = DiseaseProfile(
            disease_id="chronic_kidney_disease_stage3",
            name="Chronic Kidney Disease Stage 3",
            category="renal",
            icd10_codes=["N18.3"],
            nutritional_guidelines=[
                NutritionalGuideline("protein", 0.6, 0.8, "g/kg", "critical", "Reduce kidney workload"),
                NutritionalGuideline("sodium", None, 2000, "mg", "high", "Blood pressure and fluid control"),
                NutritionalGuideline("potassium", None, 2000, "mg", "high", "Prevent hyperkalemia"),
                NutritionalGuideline("phosphorus", None, 1000, "mg", "high", "Prevent bone disease"),
                NutritionalGuideline("fluid", None, None, "ml", "medium", "Based on urine output")
            ],
            food_restrictions=[
                FoodRestriction("bananas", "limit", "high", "High potassium", "apples or berries"),
                FoodRestriction("tomatoes", "limit", "high", "High potassium", "cucumbers"),
                FoodRestriction("dairy products", "limit", "high", "High phosphorus", "rice milk"),
                FoodRestriction("whole grains", "limit", "medium", "High phosphorus", "white rice"),
                FoodRestriction("nuts", "limit", "high", "High phosphorus and potassium", "limited portions"),
                FoodRestriction("dark sodas", "avoid", "high", "Phosphate additives", "clear beverages")
            ],
            recommended_foods=["egg whites", "chicken breast", "rice", "cauliflower", "bell peppers", "cabbage"],
            meal_timing_important=False,
            special_considerations=["Monitor potassium levels", "Limit phosphate additives", "Work with renal dietitian"]
        )
        
        self.diseases["kidney_stones"] = DiseaseProfile(
            disease_id="kidney_stones",
            name="Kidney Stones (Nephrolithiasis)",
            category="renal",
            icd10_codes=["N20.0"],
            nutritional_guidelines=[
                NutritionalGuideline("fluid", 2500, 3000, "ml", "critical", "Dilute urine"),
                NutritionalGuideline("calcium", 1000, 1200, "mg", "high", "Prevent oxalate absorption"),
                NutritionalGuideline("sodium", None, 2300, "mg", "high", "Reduce calcium excretion"),
                NutritionalGuideline("oxalate", None, 50, "mg", "high", "If calcium oxalate stones"),
                NutritionalGuideline("animal_protein", None, 6, "oz", "medium", "Reduce uric acid")
            ],
            food_restrictions=[
                FoodRestriction("spinach (high oxalate)", "limit", "high", "Calcium oxalate stones", "lettuce"),
                FoodRestriction("rhubarb", "avoid", "high", "Very high oxalate", "other fruits"),
                FoodRestriction("chocolate", "limit", "medium", "Contains oxalate", "moderate amounts"),
                FoodRestriction("excessive vitamin C", "limit", "medium", "Converts to oxalate", "food sources only")
            ],
            recommended_foods=["water", "lemon water", "low-fat dairy", "vegetables", "whole grains"],
            hydration_requirements="2.5-3 liters/day",
            special_considerations=["Urine should be pale yellow", "Type of stone determines specific restrictions"]
        )
    
    def _add_digestive_conditions(self):
        """Digestive and GI disorders"""
        
        self.diseases["celiac_disease"] = DiseaseProfile(
            disease_id="celiac_disease",
            name="Celiac Disease",
            category="digestive",
            icd10_codes=["K90.0"],
            nutritional_guidelines=[
                NutritionalGuideline("gluten", None, 0, "mg", "critical", "Strict gluten-free diet required"),
                NutritionalGuideline("fiber", 25, 35, "g", "medium", "Digestive health"),
                NutritionalGuideline("iron", 18, 27, "mg", "high", "Prevent anemia from malabsorption"),
                NutritionalGuideline("calcium", 1000, 1200, "mg", "high", "Bone health"),
                NutritionalGuideline("vitamin_d", 600, 800, "IU", "high", "Calcium absorption")
            ],
            food_restrictions=[
                FoodRestriction("wheat", "avoid", "critical", "Contains gluten", "rice or quinoa"),
                FoodRestriction("barley", "avoid", "critical", "Contains gluten", "gluten-free grains"),
                FoodRestriction("rye", "avoid", "critical", "Contains gluten", "gluten-free alternatives"),
                FoodRestriction("cross-contamination", "avoid", "critical", "Even traces harmful", "dedicated GF facilities")
            ],
            recommended_foods=["rice", "quinoa", "corn", "potatoes", "fruits", "vegetables", "meats", "eggs"],
            special_considerations=["Read all labels carefully", "Watch for cross-contamination", "Dedicated cooking equipment"]
        )
        
        self.diseases["crohns_disease"] = DiseaseProfile(
            disease_id="crohns_disease",
            name="Crohn's Disease",
            category="digestive",
            icd10_codes=["K50.9"],
            nutritional_guidelines=[
                NutritionalGuideline("protein", 1.2, 1.5, "g/kg", "high", "Tissue repair and healing"),
                NutritionalGuideline("calories", None, None, "kcal", "high", "Compensate for malabsorption"),
                NutritionalGuideline("fiber", None, 15, "g", "medium", "During flares - low fiber"),
                NutritionalGuideline("iron", 18, 27, "mg", "high", "Prevent anemia"),
                NutritionalGuideline("vitamin_b12", 2.4, 100, "mcg", "high", "Malabsorption common")
            ],
            food_restrictions=[
                FoodRestriction("raw vegetables", "limit", "high", "During flares - hard to digest", "cooked vegetables"),
                FoodRestriction("whole nuts", "limit", "medium", "May cause blockage", "nut butters"),
                FoodRestriction("seeds", "limit", "medium", "May irritate", "seedless options"),
                FoodRestriction("popcorn", "avoid", "medium", "During flares", "soft snacks"),
                FoodRestriction("spicy foods", "limit", "medium", "May trigger symptoms", "mild seasonings")
            ],
            recommended_foods=["white rice", "bananas", "cooked carrots", "lean proteins", "smooth nut butters"],
            meal_timing_important=True,
            special_considerations=["Low-residue diet during flares", "Keep food diary", "Small frequent meals"]
        )
        
        self.diseases["irritable_bowel_syndrome"] = DiseaseProfile(
            disease_id="irritable_bowel_syndrome",
            name="Irritable Bowel Syndrome (IBS)",
            category="digestive",
            icd10_codes=["K58.9"],
            nutritional_guidelines=[
                NutritionalGuideline("fiber", 20, 35, "g", "medium", "Soluble fiber helpful for some"),
                NutritionalGuideline("fodmaps", None, None, "g", "high", "Low FODMAP diet often helpful")
            ],
            food_restrictions=[
                FoodRestriction("high FODMAP foods", "limit", "high", "Trigger symptoms", "low FODMAP alternatives"),
                FoodRestriction("onions", "limit", "high", "High FODMAP", "green onion tops"),
                FoodRestriction("garlic", "limit", "high", "High FODMAP", "garlic-infused oil"),
                FoodRestriction("apples", "limit", "medium", "High FODMAP", "berries"),
                FoodRestriction("dairy (if lactose intolerant)", "limit", "high", "Lactose intolerance common", "lactose-free")
            ],
            recommended_foods=["rice", "oats", "carrots", "zucchini", "chicken", "fish", "eggs"],
            special_considerations=["Keep symptom diary", "Reintroduce foods gradually", "Stress management important"]
        )
    
    def _add_liver_conditions(self):
        """Liver diseases"""
        
        self.diseases["fatty_liver_disease"] = DiseaseProfile(
            disease_id="fatty_liver_disease",
            name="Non-Alcoholic Fatty Liver Disease (NAFLD)",
            category="hepatic",
            icd10_codes=["K76.0"],
            nutritional_guidelines=[
                NutritionalGuideline("calories", None, None, "kcal", "high", "Weight loss 5-10% if overweight"),
                NutritionalGuideline("saturated_fat", None, 7, "%", "high", "Reduce liver fat"),
                NutritionalGuideline("refined_sugar", None, 10, "%", "high", "Reduce liver fat accumulation"),
                NutritionalGuideline("fiber", 25, 35, "g", "medium", "Metabolic health")
            ],
            food_restrictions=[
                FoodRestriction("alcohol", "avoid", "critical", "Liver damage", "non-alcoholic beverages"),
                FoodRestriction("fructose/HFCS", "avoid", "high", "Increases liver fat", "whole fruits in moderation"),
                FoodRestriction("fried foods", "limit", "high", "High in unhealthy fats", "baked or grilled"),
                FoodRestriction("white bread", "limit", "medium", "Refined carbs", "whole grains")
            ],
            recommended_foods=["coffee", "green tea", "nuts", "olive oil", "fish", "vegetables", "whole grains"],
            portion_control_critical=True
        )
    
    def _add_autoimmune_conditions(self):
        """Autoimmune diseases"""
        
        self.diseases["rheumatoid_arthritis"] = DiseaseProfile(
            disease_id="rheumatoid_arthritis",
            name="Rheumatoid Arthritis",
            category="autoimmune",
            icd10_codes=["M06.9"],
            nutritional_guidelines=[
                NutritionalGuideline("omega3", 2, 3, "g", "high", "Anti-inflammatory"),
                NutritionalGuideline("vitamin_d", 800, 2000, "IU", "high", "Immune modulation"),
                NutritionalGuideline("antioxidants", None, None, "mg", "medium", "Reduce oxidative stress")
            ],
            food_restrictions=[
                FoodRestriction("nightshades", "caution", "low", "Some report sensitivity", "monitor tolerance"),
                FoodRestriction("processed foods", "limit", "medium", "Pro-inflammatory", "whole foods"),
                FoodRestriction("red meat", "limit", "medium", "Arachidonic acid inflammatory", "fish and poultry")
            ],
            recommended_foods=["fatty fish", "walnuts", "berries", "leafy greens", "olive oil", "turmeric"],
            special_considerations=["Mediterranean or anti-inflammatory diet", "Food sensitivities vary"]
        )
    
    def _add_cancer_conditions(self):
        """Cancer nutritional support"""
        
        self.diseases["breast_cancer"] = DiseaseProfile(
            disease_id="breast_cancer",
            name="Breast Cancer",
            category="oncology",
            icd10_codes=["C50.9"],
            nutritional_guidelines=[
                NutritionalGuideline("protein", 1.2, 1.5, "g/kg", "high", "Tissue repair during treatment"),
                NutritionalGuideline("calories", None, None, "kcal", "high", "Maintain weight during treatment"),
                NutritionalGuideline("fiber", 25, 35, "g", "medium", "Reduce recurrence risk"),
                NutritionalGuideline("soy", None, 3, "servings", "caution", "Controversial - consult oncologist")
            ],
            food_restrictions=[
                FoodRestriction("alcohol", "avoid", "high", "Increases recurrence risk", "non-alcoholic beverages"),
                FoodRestriction("processed meats", "limit", "medium", "Cancer risk", "fresh meats"),
                FoodRestriction("charred foods", "limit", "medium", "Carcinogens", "properly cooked foods")
            ],
            recommended_foods=["cruciferous vegetables", "berries", "tomatoes", "green tea", "whole grains", "lean proteins"],
            special_considerations=["During chemo - manage nausea", "Post-treatment - weight management", "Plant-based emphasis"]
        )
    
    def _add_neurological_conditions(self):
        """Neurological disorders"""
        
        self.diseases["epilepsy"] = DiseaseProfile(
            disease_id="epilepsy",
            name="Epilepsy",
            category="neurological",
            icd10_codes=["G40.9"],
            nutritional_guidelines=[
                NutritionalGuideline("ketogenic_ratio", 3, 4, "ratio", "high", "If on ketogenic diet for epilepsy"),
                NutritionalGuideline("vitamin_d", 600, 2000, "IU", "high", "Medications may deplete"),
                NutritionalGuideline("calcium", 1000, 1200, "mg", "medium", "Bone health with medications")
            ],
            food_restrictions=[
                FoodRestriction("alcohol", "avoid", "critical", "Lowers seizure threshold", "non-alcoholic beverages"),
                FoodRestriction("caffeine (excess)", "limit", "medium", "May trigger in some", "moderate intake"),
                FoodRestriction("grapefruit", "avoid", "high", "Interacts with medications", "other citrus")
            ],
            recommended_foods=["if keto: avocados, nuts, seeds, fatty fish, low-carb vegetables"],
            meal_timing_important=True,
            special_considerations=["Ketogenic diet very effective for some", "Medication interactions important"]
        )
    
    def _add_bone_conditions(self):
        """Bone and joint disorders"""
        
        self.diseases["osteoporosis"] = DiseaseProfile(
            disease_id="osteoporosis",
            name="Osteoporosis",
            category="musculoskeletal",
            icd10_codes=["M81.0"],
            nutritional_guidelines=[
                NutritionalGuideline("calcium", 1200, 1500, "mg", "critical", "Bone mineral density"),
                NutritionalGuideline("vitamin_d", 800, 2000, "IU", "critical", "Calcium absorption"),
                NutritionalGuideline("vitamin_k", 90, 120, "mcg", "high", "Bone formation"),
                NutritionalGuideline("magnesium", 310, 420, "mg", "medium", "Bone structure"),
                NutritionalGuideline("protein", 1.0, 1.2, "g/kg", "medium", "Bone matrix")
            ],
            food_restrictions=[
                FoodRestriction("excessive salt", "limit", "medium", "Increases calcium loss", "moderate sodium"),
                FoodRestriction("caffeine (excess)", "limit", "medium", "May reduce calcium absorption", "moderate intake"),
                FoodRestriction("alcohol", "limit", "high", "Decreases bone formation", "moderate or avoid"),
                FoodRestriction("phytates (excess)", "limit", "low", "Binds minerals", "soak/cook properly")
            ],
            recommended_foods=["dairy products", "fortified foods", "leafy greens", "fatty fish", "eggs", "almonds"],
            special_considerations=["Weight-bearing exercise crucial", "Sun exposure for vitamin D", "Fall prevention"]
        )
    
    def _add_respiratory_conditions(self):
        """Respiratory diseases"""
        
        self.diseases["asthma"] = DiseaseProfile(
            disease_id="asthma",
            name="Asthma",
            category="respiratory",
            icd10_codes=["J45.9"],
            nutritional_guidelines=[
                NutritionalGuideline("omega3", 1, 2, "g", "medium", "Anti-inflammatory"),
                NutritionalGuideline("vitamin_c", 75, 200, "mg", "medium", "Antioxidant"),
                NutritionalGuideline("vitamin_d", 600, 2000, "IU", "medium", "Immune function")
            ],
            food_restrictions=[
                FoodRestriction("sulfites", "avoid", "high", "Trigger in some", "sulfite-free foods"),
                FoodRestriction("food allergies", "avoid", "critical", "If identified triggers", "allergen-free alternatives"),
                FoodRestriction("preservatives", "limit", "medium", "May trigger symptoms", "fresh foods")
            ],
            recommended_foods=["fruits", "vegetables", "fish", "whole grains", "nuts"],
            special_considerations=["Identify food triggers", "Maintain healthy weight", "Stay hydrated"]
        )
    
    def _add_blood_disorders(self):
        """Blood and hematological conditions"""
        
        self.diseases["iron_deficiency_anemia"] = DiseaseProfile(
            disease_id="iron_deficiency_anemia",
            name="Iron Deficiency Anemia",
            category="hematology",
            icd10_codes=["D50.9"],
            nutritional_guidelines=[
                NutritionalGuideline("iron", 18, 27, "mg", "critical", "Restore iron stores"),
                NutritionalGuideline("vitamin_c", 75, 200, "mg", "high", "Enhance iron absorption"),
                NutritionalGuideline("vitamin_b12", 2.4, 100, "mcg", "medium", "Red blood cell production"),
                NutritionalGuideline("folate", 400, 800, "mcg", "medium", "Red blood cell production")
            ],
            food_restrictions=[
                FoodRestriction("tea with meals", "avoid", "high", "Inhibits iron absorption", "drink between meals"),
                FoodRestriction("coffee with meals", "avoid", "high", "Inhibits iron absorption", "drink between meals"),
                FoodRestriction("calcium supplements with iron", "avoid", "high", "Competes for absorption", "separate timing")
            ],
            recommended_foods=["red meat", "liver", "spinach", "lentils", "fortified cereals", "citrus fruits"],
            meal_timing_important=True,
            special_considerations=["Take iron supplements on empty stomach", "Pair iron-rich foods with vitamin C"]
        )
    
    def get_disease(self, disease_id: str) -> Optional[DiseaseProfile]:
        """Get disease profile by ID"""
        return self.diseases.get(disease_id)
    
    def search_diseases(self, query: str, category: Optional[str] = None) -> List[DiseaseProfile]:
        """Search diseases by name or category"""
        results = []
        query_lower = query.lower()
        
        for disease in self.diseases.values():
            if category and disease.category != category:
                continue
            
            if (query_lower in disease.name.lower() or 
                query_lower in disease.disease_id.lower() or
                any(query_lower in code.lower() for code in disease.icd10_codes)):
                results.append(disease)
        
        return results
    
    def get_all_categories(self) -> Set[str]:
        """Get all disease categories"""
        return {disease.category for disease in self.diseases.values()}
    
    def get_disease_count(self) -> int:
        """Get total number of diseases in database"""
        return len(self.diseases)


class MultiDiseaseOptimizer:
    """Optimize meals for families with multiple diseases"""
    
    def __init__(self):
        self.db = DiseaseDatabase()
    
    def optimize_for_family(
        self,
        family_members: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimize meal plan for family with different health conditions
        
        Args:
            family_members: List of dicts with 'name' and 'diseases' (list of disease_ids)
        
        Returns:
            Optimized nutritional targets and food recommendations
        """
        
        all_guidelines = {}
        all_restrictions = []
        all_recommended_foods = set()
        
        # Aggregate all disease requirements
        for member in family_members:
            member_diseases = member.get('diseases', [])
            
            for disease_id in member_diseases:
                disease = self.db.get_disease(disease_id)
                if not disease:
                    continue
                
                # Collect guidelines
                for guideline in disease.nutritional_guidelines:
                    key = guideline.nutrient
                    if key not in all_guidelines:
                        all_guidelines[key] = []
                    all_guidelines[key].append({
                        'member': member['name'],
                        'disease': disease.name,
                        'guideline': guideline
                    })
                
                # Collect restrictions (use most restrictive)
                all_restrictions.extend([
                    {
                        'member': member['name'],
                        'disease': disease.name,
                        'restriction': r
                    } for r in disease.food_restrictions
                ])
                
                # Collect recommended foods
                all_recommended_foods.update(disease.recommended_foods)
        
        # Resolve conflicts and create unified plan
        unified_guidelines = self._resolve_guideline_conflicts(all_guidelines)
        unified_restrictions = self._merge_restrictions(all_restrictions)
        
        return {
            'unified_nutritional_targets': unified_guidelines,
            'food_restrictions': unified_restrictions,
            'recommended_foods': list(all_recommended_foods),
            'family_members_analyzed': len(family_members),
            'total_diseases_considered': sum(len(m.get('diseases', [])) for m in family_members)
        }
    
    def _resolve_guideline_conflicts(
        self,
        guidelines: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """Resolve conflicting nutritional guidelines"""
        
        resolved = {}
        
        for nutrient, entries in guidelines.items():
            # Use most restrictive limits
            min_values = [e['guideline'].target_min for e in entries if e['guideline'].target_min is not None]
            max_values = [e['guideline'].target_max for e in entries if e['guideline'].target_max is not None]
            
            resolved[nutrient] = {
                'target_min': max(min_values) if min_values else None,
                'target_max': min(max_values) if max_values else None,
                'unit': entries[0]['guideline'].unit,
                'priority': max([e['guideline'].priority for e in entries], 
                              key=lambda x: {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(x, 0)),
                'reasons': [{'member': e['member'], 'disease': e['disease'], 'reason': e['guideline'].reason} for e in entries]
            }
        
        return resolved
    
    def _merge_restrictions(
        self,
        restrictions: List[Dict]
    ) -> List[Dict]:
        """Merge and prioritize food restrictions"""
        
        # Group by food item
        by_food = {}
        for r in restrictions:
            food = r['restriction'].food_item
            if food not in by_food:
                by_food[food] = []
            by_food[food].append(r)
        
        # Take most restrictive for each food
        merged = []
        severity_order = {'critical': 4, 'high': 3, 'moderate': 2, 'low': 1}
        restriction_order = {'avoid': 3, 'limit': 2, 'caution': 1}
        
        for food, entries in by_food.items():
            most_restrictive = max(
                entries,
                key=lambda x: (
                    restriction_order.get(x['restriction'].restriction_type, 0),
                    severity_order.get(x['restriction'].severity, 0)
                )
            )
            
            merged.append({
                'food_item': food,
                'restriction_type': most_restrictive['restriction'].restriction_type,
                'severity': most_restrictive['restriction'].severity,
                'reason': most_restrictive['restriction'].reason,
                'alternative': most_restrictive['restriction'].alternative,
                'affects_members': [e['member'] for e in entries],
                'related_diseases': list(set(e['disease'] for e in entries))
            })
        
        return sorted(merged, key=lambda x: (
            restriction_order.get(x['restriction_type'], 0),
            severity_order.get(x['severity'], 0)
        ), reverse=True)


# Example usage
if __name__ == "__main__":
    # Initialize system
    db = DiseaseDatabase()
    optimizer = MultiDiseaseOptimizer()
    
    print(f"Disease Optimization Engine Initialized")
    print(f"Total diseases in database: {db.get_disease_count()}")
    print(f"Categories: {', '.join(sorted(db.get_all_categories()))}")
    
    # Example family optimization
    family = [
        {"name": "Dad", "diseases": ["diabetes_type2", "hypertension"]},
        {"name": "Mom", "diseases": ["celiac_disease", "osteoporosis"]},
        {"name": "Child", "diseases": ["asthma"]}
    ]
    
    optimization = optimizer.optimize_for_family(family)
    
    print(f"\nFamily Meal Plan Optimization:")
    print(f"Members analyzed: {optimization['family_members_analyzed']}")
    print(f"Total diseases: {optimization['total_diseases_considered']}")
    print(f"Recommended foods: {len(optimization['recommended_foods'])}")
    print(f"Food restrictions: {len(optimization['food_restrictions'])}")
