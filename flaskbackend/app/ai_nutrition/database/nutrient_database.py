"""
AI Nutrition Analysis System - Comprehensive Nutrient Database
Phase 1: Complete database of 150+ nutrients with RDA values, bioavailability, and interactions

This module contains detailed information for all essential and non-essential nutrients
tracked by the system, including vitamins, minerals, amino acids, fatty acids, and phytonutrients.

Author: AI Nutrition System
Version: 1.0.0
"""

from decimal import Decimal
from typing import Dict, List
from app.ai_nutrition.models.core_data_models import (
    Nutrient, NutrientReference, NutrientCategory, NutrientSubcategory,
    MeasurementUnit, LifeStage, BiologicalSex, CookingMethod
)


class NutrientDatabase:
    """Comprehensive database of all nutrients tracked by the system"""
    
    def __init__(self):
        self.nutrients: Dict[str, NutrientReference] = {}
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize all nutrient data"""
        # Macronutrients
        self._add_macronutrients()
        
        # Water-soluble vitamins
        self._add_b_vitamins()
        self._add_vitamin_c()
        
        # Fat-soluble vitamins
        self._add_fat_soluble_vitamins()
        
        # Major minerals
        self._add_major_minerals()
        
        # Trace minerals
        self._add_trace_minerals()
        
        # Essential amino acids
        self._add_essential_amino_acids()
        
        # Non-essential amino acids
        self._add_non_essential_amino_acids()
        
        # Essential fatty acids
        self._add_essential_fatty_acids()
        
        # Omega fatty acids
        self._add_omega_fatty_acids()
        
        # Phytonutrients
        self._add_phytonutrients()
        
        # Antioxidants
        self._add_antioxidants()
        
        # Probiotics and prebiotics
        self._add_probiotics_prebiotics()
    
    # ========================================================================
    # MACRONUTRIENTS
    # ========================================================================
    
    def _add_macronutrients(self):
        """Add macronutrient definitions"""
        
        # Protein
        self.nutrients["protein"] = NutrientReference(
            nutrient_id="protein",
            nutrient_name="Protein",
            category=NutrientCategory.MACRONUTRIENT,
            subcategory=NutrientSubcategory.PROTEIN,
            unit=MeasurementUnit.GRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 56.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 46.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 56.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 46.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.MALE): 56.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.FEMALE): 46.0,
                (LifeStage.SENIOR_71_PLUS_YEARS, BiologicalSex.MALE): 56.0,
                (LifeStage.SENIOR_71_PLUS_YEARS, BiologicalSex.FEMALE): 46.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 71.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 71.0,
                (LifeStage.ADOLESCENT_14_18_YEARS, BiologicalSex.MALE): 52.0,
                (LifeStage.ADOLESCENT_14_18_YEARS, BiologicalSex.FEMALE): 46.0,
            },
            optimal_range=(0.8, 2.2),  # g/kg body weight
            bioavailability_percentage=95.0,
            primary_functions=[
                "Muscle growth and repair",
                "Enzyme and hormone production",
                "Immune function",
                "Nutrient transport",
                "Tissue structure and maintenance"
            ],
            deficiency_symptoms=[
                "Muscle wasting",
                "Weakness",
                "Edema",
                "Impaired immune function",
                "Slow wound healing"
            ],
            top_food_sources=[
                ("Chicken breast", 31.0),
                ("Salmon", 25.0),
                ("Eggs", 13.0),
                ("Greek yogurt", 10.0),
                ("Lentils", 9.0),
                ("Quinoa", 4.4)
            ],
            heat_stable=True,
            water_soluble=False,
            evidence_level="A",
            research_references=["PMID: 28446284", "PMID: 29497353"]
        )
        
        # Carbohydrates
        self.nutrients["carbohydrate"] = NutrientReference(
            nutrient_id="carbohydrate",
            nutrient_name="Carbohydrate (Total)",
            category=NutrientCategory.MACRONUTRIENT,
            subcategory=NutrientSubcategory.CARBOHYDRATE,
            unit=MeasurementUnit.GRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 130.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 130.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 130.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 130.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 175.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 210.0,
            },
            optimal_range=(225.0, 325.0),  # For 2000 kcal diet (45-65% calories)
            bioavailability_percentage=98.0,
            primary_functions=[
                "Primary energy source",
                "Brain fuel (glucose)",
                "Glycogen storage",
                "Fiber for gut health",
                "Protein-sparing effect"
            ],
            top_food_sources=[
                ("White rice", 28.0),
                ("Oats", 66.0),
                ("Sweet potato", 20.0),
                ("Banana", 23.0),
                ("Whole wheat bread", 49.0)
            ],
            heat_stable=True,
            evidence_level="A"
        )
        
        # Dietary Fiber
        self.nutrients["fiber"] = NutrientReference(
            nutrient_id="fiber",
            nutrient_name="Dietary Fiber",
            category=NutrientCategory.FIBER,
            subcategory=NutrientSubcategory.SOLUBLE_FIBER,
            unit=MeasurementUnit.GRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 38.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 25.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 38.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 25.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.MALE): 30.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.FEMALE): 21.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 28.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 29.0,
            },
            optimal_range=(25.0, 40.0),
            primary_functions=[
                "Digestive health",
                "Blood sugar regulation",
                "Cholesterol reduction",
                "Satiety and weight management",
                "Prebiotic for gut microbiome",
                "Colon cancer prevention"
            ],
            deficiency_symptoms=[
                "Constipation",
                "Poor blood sugar control",
                "Elevated cholesterol",
                "Increased appetite",
                "Gut dysbiosis"
            ],
            top_food_sources=[
                ("Chia seeds", 34.4),
                ("Navy beans", 10.5),
                ("Raspberries", 6.5),
                ("Lentils", 7.9),
                ("Avocado", 6.7),
                ("Oats", 10.6)
            ],
            heat_stable=True,
            water_soluble=True,
            evidence_level="A",
            research_references=["PMID: 32483598", "PMID: 31058160"]
        )
        
        # Total Fat
        self.nutrients["fat_total"] = NutrientReference(
            nutrient_id="fat_total",
            nutrient_name="Total Fat",
            category=NutrientCategory.MACRONUTRIENT,
            subcategory=NutrientSubcategory.FAT,
            unit=MeasurementUnit.GRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 44.0,  # 20% of 2000 kcal
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 44.0,
            },
            optimal_range=(44.0, 78.0),  # 20-35% of calories for 2000 kcal diet
            primary_functions=[
                "Energy storage",
                "Hormone production",
                "Cell membrane structure",
                "Fat-soluble vitamin absorption",
                "Brain and nerve function",
                "Insulation and protection"
            ],
            top_food_sources=[
                ("Olive oil", 100.0),
                ("Avocado", 15.0),
                ("Almonds", 49.0),
                ("Salmon", 13.0),
                ("Dark chocolate", 43.0)
            ],
            heat_stable=True,
            evidence_level="A"
        )
        
        # Water
        self.nutrients["water"] = NutrientReference(
            nutrient_id="water",
            nutrient_name="Water",
            category=NutrientCategory.MACRONUTRIENT,
            subcategory=NutrientSubcategory.WATER,
            unit=MeasurementUnit.LITER,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 3.7,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 2.7,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 3.7,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 2.7,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 3.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 3.8,
            },
            primary_functions=[
                "Temperature regulation",
                "Nutrient transport",
                "Waste elimination",
                "Joint lubrication",
                "Cellular processes",
                "Blood volume maintenance"
            ],
            deficiency_symptoms=[
                "Dehydration",
                "Fatigue",
                "Cognitive impairment",
                "Constipation",
                "Kidney stones",
                "Dark urine"
            ],
            evidence_level="A"
        )
    
    # ========================================================================
    # B VITAMINS
    # ========================================================================
    
    def _add_b_vitamins(self):
        """Add B-complex vitamin definitions"""
        
        # Thiamin (B1)
        self.nutrients["thiamin_b1"] = NutrientReference(
            nutrient_id="thiamin_b1",
            nutrient_name="Thiamin (Vitamin B1)",
            category=NutrientCategory.VITAMIN,
            subcategory=NutrientSubcategory.VITAMIN_B_COMPLEX,
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 1.2,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 1.1,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 1.2,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 1.1,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 1.4,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 1.4,
            },
            bioavailability_percentage=85.0,
            water_soluble=True,
            heat_stable=False,
            cooking_loss_rates={
                CookingMethod.BOILED: 30.0,
                CookingMethod.BAKED: 20.0,
                CookingMethod.FRIED: 25.0,
            },
            primary_functions=[
                "Energy metabolism (carbohydrate)",
                "Nerve function",
                "Muscle contraction",
                "Enzyme cofactor"
            ],
            deficiency_symptoms=[
                "Beriberi",
                "Wernicke-Korsakoff syndrome",
                "Fatigue",
                "Nerve damage",
                "Heart problems"
            ],
            top_food_sources=[
                ("Pork chop", 0.87),
                ("Sunflower seeds", 1.48),
                ("Black beans", 0.42),
                ("Trout", 0.4),
                ("Green peas", 0.27)
            ],
            absorption_enhancers=["Adequate protein", "Magnesium"],
            absorption_inhibitors=["Alcohol", "Tannins in tea/coffee"],
            synergistic_nutrients=["vitamin_b2", "vitamin_b3", "magnesium"],
            evidence_level="A",
            research_references=["PMID: 29351415"]
        )
        
        # Riboflavin (B2)
        self.nutrients["riboflavin_b2"] = NutrientReference(
            nutrient_id="riboflavin_b2",
            nutrient_name="Riboflavin (Vitamin B2)",
            category=NutrientCategory.VITAMIN,
            subcategory=NutrientSubcategory.VITAMIN_B_COMPLEX,
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 1.3,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 1.1,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 1.3,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 1.1,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 1.4,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 1.6,
            },
            bioavailability_percentage=90.0,
            water_soluble=True,
            heat_stable=True,
            light_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILED: 20.0,
                CookingMethod.STEAMED: 10.0,
            },
            primary_functions=[
                "Energy production",
                "Antioxidant regeneration",
                "Red blood cell production",
                "Skin and eye health",
                "B6 and folate activation"
            ],
            deficiency_symptoms=[
                "Ariboflavinosis",
                "Cracked lips (cheilosis)",
                "Sore throat",
                "Magenta tongue",
                "Skin rash",
                "Anemia"
            ],
            top_food_sources=[
                ("Beef liver", 2.91),
                ("Yogurt", 0.52),
                ("Milk", 0.45),
                ("Almonds", 1.14),
                ("Spinach", 0.19)
            ],
            absorption_enhancers=["Protein", "Fat"],
            absorption_inhibitors=["Alcohol", "UV light"],
            synergistic_nutrients=["vitamin_b3", "vitamin_b6", "iron"],
            evidence_level="A",
            research_references=["PMID: 28391299"]
        )
        
        # Niacin (B3)
        self.nutrients["niacin_b3"] = NutrientReference(
            nutrient_id="niacin_b3",
            nutrient_name="Niacin (Vitamin B3)",
            category=NutrientCategory.VITAMIN,
            subcategory=NutrientSubcategory.VITAMIN_B_COMPLEX,
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 16.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 14.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 16.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 14.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 18.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 17.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 35.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 35.0,
            },
            therapeutic_range=(500.0, 2000.0),  # For cholesterol management
            bioavailability_percentage=85.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Energy metabolism (NAD/NADP)",
                "DNA repair",
                "Cholesterol management",
                "Hormone synthesis",
                "Cell signaling"
            ],
            deficiency_symptoms=[
                "Pellagra (4 Ds: dermatitis, diarrhea, dementia, death)",
                "Fatigue",
                "Skin lesions",
                "Cognitive decline"
            ],
            toxicity_symptoms=[
                "Flushing (niacin flush)",
                "Liver damage (high doses)",
                "Hyperglycemia",
                "Gout"
            ],
            top_food_sources=[
                ("Tuna", 21.9),
                ("Chicken breast", 13.7),
                ("Turkey", 11.8),
                ("Peanuts", 12.1),
                ("Mushrooms", 3.6)
            ],
            absorption_enhancers=["Tryptophan (converts to niacin)"],
            synergistic_nutrients=["vitamin_b1", "vitamin_b2", "vitamin_b6"],
            evidence_level="A",
            research_references=["PMID: 29145973", "PMID: 25926415"]
        )
        
        # Pantothenic Acid (B5)
        self.nutrients["pantothenic_acid_b5"] = NutrientReference(
            nutrient_id="pantothenic_acid_b5",
            nutrient_name="Pantothenic Acid (Vitamin B5)",
            category=NutrientCategory.VITAMIN,
            subcategory=NutrientSubcategory.VITAMIN_B_COMPLEX,
            unit=MeasurementUnit.MILLIGRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 5.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 5.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 6.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 7.0,
            },
            bioavailability_percentage=85.0,
            water_soluble=True,
            heat_stable=False,
            cooking_loss_rates={
                CookingMethod.BOILED: 35.0,
                CookingMethod.CANNED: 50.0,
            },
            primary_functions=[
                "Coenzyme A synthesis",
                "Energy metabolism",
                "Fatty acid synthesis",
                "Cholesterol production",
                "Neurotransmitter synthesis",
                "Stress hormone production"
            ],
            deficiency_symptoms=[
                "Fatigue",
                "Irritability",
                "Numbness/burning feet",
                "Insomnia",
                "Gastrointestinal problems"
            ],
            top_food_sources=[
                ("Beef liver", 7.17),
                ("Sunflower seeds", 7.04),
                ("Chicken liver", 6.19),
                ("Shiitake mushrooms", 3.59),
                ("Avocado", 1.39)
            ],
            evidence_level="A",
            research_references=["PMID: 21310306"]
        )
        
        # Pyridoxine (B6)
        self.nutrients["pyridoxine_b6"] = NutrientReference(
            nutrient_id="pyridoxine_b6",
            nutrient_name="Pyridoxine (Vitamin B6)",
            category=NutrientCategory.VITAMIN,
            subcategory=NutrientSubcategory.VITAMIN_B_COMPLEX,
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 1.3,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 1.3,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 1.3,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 1.3,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.MALE): 1.7,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.FEMALE): 1.5,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 1.9,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 2.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 100.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 100.0,
            },
            bioavailability_percentage=75.0,
            water_soluble=True,
            heat_stable=False,
            light_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILED: 40.0,
                CookingMethod.FROZEN: 30.0,
            },
            primary_functions=[
                "Amino acid metabolism",
                "Neurotransmitter synthesis (serotonin, dopamine)",
                "Hemoglobin production",
                "Immune function",
                "Gene expression",
                "Homocysteine metabolism"
            ],
            deficiency_symptoms=[
                "Anemia (microcytic)",
                "Depression",
                "Confusion",
                "Weakened immunity",
                "Dermatitis",
                "Elevated homocysteine"
            ],
            toxicity_symptoms=[
                "Peripheral neuropathy (high doses >1000mg)",
                "Skin lesions",
                "Photosensitivity"
            ],
            top_food_sources=[
                ("Chickpeas", 0.54),
                ("Tuna", 1.04),
                ("Chicken breast", 0.64),
                ("Salmon", 0.94),
                ("Banana", 0.37),
                ("Potatoes", 0.3)
            ],
            absorption_enhancers=["Vitamin B2 (converts B6 to active form)"],
            absorption_inhibitors=["Alcohol", "Oral contraceptives"],
            synergistic_nutrients=["vitamin_b12", "folate", "zinc"],
            antagonistic_nutrients=["Levodopa (Parkinson's drug)"],
            evidence_level="A",
            research_references=["PMID: 27928397", "PMID: 29661939"]
        )
        
        # Biotin (B7)
        self.nutrients["biotin_b7"] = NutrientReference(
            nutrient_id="biotin_b7",
            nutrient_name="Biotin (Vitamin B7)",
            category=NutrientCategory.VITAMIN,
            subcategory=NutrientSubcategory.VITAMIN_B_COMPLEX,
            unit=MeasurementUnit.MICROGRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 30.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 30.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 30.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 35.0,
            },
            bioavailability_percentage=90.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Fatty acid synthesis",
                "Amino acid metabolism",
                "Gluconeogenesis",
                "Hair, skin, nail health",
                "Gene regulation"
            ],
            deficiency_symptoms=[
                "Hair loss (alopecia)",
                "Brittle nails",
                "Skin rash (seborrheic dermatitis)",
                "Conjunctivitis",
                "Depression",
                "Seizures (severe)"
            ],
            top_food_sources=[
                ("Beef liver", 30.8),
                ("Egg yolk", 53.0),
                ("Salmon", 5.0),
                ("Almonds", 1.5),
                ("Sweet potato", 2.4),
                ("Spinach", 0.6)
            ],
            absorption_inhibitors=["Raw egg whites (avidin binds biotin)", "Alcohol"],
            synergistic_nutrients=["vitamin_b5", "vitamin_b12"],
            evidence_level="B",
            research_references=["PMID: 28327503"]
        )
        
        # Folate (B9)
        self.nutrients["folate_b9"] = NutrientReference(
            nutrient_id="folate_b9",
            nutrient_name="Folate (Vitamin B9)",
            category=NutrientCategory.VITAMIN,
            subcategory=NutrientSubcategory.VITAMIN_B_COMPLEX,
            unit=MeasurementUnit.MICROGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 400.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 400.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 400.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 400.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 600.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 500.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 1000.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 1000.0,
            },
            bioavailability_percentage=85.0,
            water_soluble=True,
            heat_stable=False,
            light_sensitive=True,
            oxygen_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILED: 50.0,
                CookingMethod.STEAMED: 30.0,
                CookingMethod.MICROWAVED: 20.0,
            },
            primary_functions=[
                "DNA synthesis and repair",
                "Cell division",
                "Amino acid metabolism",
                "Red blood cell formation",
                "Neural tube development (pregnancy)",
                "Homocysteine metabolism"
            ],
            deficiency_symptoms=[
                "Megaloblastic anemia",
                "Neural tube defects (pregnancy)",
                "Fatigue",
                "Mouth sores",
                "Elevated homocysteine",
                "Poor growth"
            ],
            top_food_sources=[
                ("Chicken liver", 560.0),
                ("Lentils", 181.0),
                ("Spinach", 194.0),
                ("Asparagus", 149.0),
                ("Brussels sprouts", 61.0),
                ("Avocado", 81.0)
            ],
            absorption_enhancers=["Vitamin C", "Vitamin B12"],
            absorption_inhibitors=["Alcohol", "Methotrexate", "Sulfasalazine"],
            synergistic_nutrients=["vitamin_b12", "vitamin_b6", "vitamin_c"],
            antagonistic_nutrients=["Methotrexate", "Phenytoin"],
            evidence_level="A",
            research_references=["PMID: 29478324", "PMID: 30982439"]
        )
        
        # Cobalamin (B12)
        self.nutrients["cobalamin_b12"] = NutrientReference(
            nutrient_id="cobalamin_b12",
            nutrient_name="Cobalamin (Vitamin B12)",
            category=NutrientCategory.VITAMIN,
            subcategory=NutrientSubcategory.VITAMIN_B_COMPLEX,
            unit=MeasurementUnit.MICROGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 2.4,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 2.4,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 2.4,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 2.4,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.MALE): 2.4,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.FEMALE): 2.4,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 2.6,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 2.8,
            },
            optimal_range=(2.4, 100.0),  # Higher doses safe for deficiency
            bioavailability_percentage=50.0,  # Requires intrinsic factor
            water_soluble=True,
            heat_stable=True,
            light_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILED: 30.0,
                CookingMethod.MICROWAVED: 20.0,
            },
            primary_functions=[
                "Red blood cell formation",
                "DNA synthesis",
                "Nerve myelination",
                "Fatty acid metabolism",
                "Homocysteine metabolism",
                "Cognitive function"
            ],
            deficiency_symptoms=[
                "Pernicious anemia (lack of intrinsic factor)",
                "Megaloblastic anemia",
                "Peripheral neuropathy",
                "Memory loss",
                "Dementia",
                "Depression",
                "Fatigue"
            ],
            top_food_sources=[
                ("Clams", 84.1),
                ("Beef liver", 70.6),
                ("Trout", 5.4),
                ("Salmon", 4.8),
                ("Eggs", 0.9),
                ("Fortified nutritional yeast", 2.4)
            ],
            absorption_enhancers=["Intrinsic factor", "Adequate stomach acid", "Calcium"],
            absorption_inhibitors=["PPI medications", "Metformin", "H2 blockers", "Atrophic gastritis"],
            synergistic_nutrients=["folate", "vitamin_b6", "calcium"],
            evidence_level="A",
            research_references=["PMID: 28275201", "PMID: 29669537"]
        )
    
    # ========================================================================
    # VITAMIN C
    # ========================================================================
    
    def _add_vitamin_c(self):
        """Add Vitamin C definition"""
        
        self.nutrients["vitamin_c"] = NutrientReference(
            nutrient_id="vitamin_c",
            nutrient_name="Vitamin C (Ascorbic Acid)",
            category=NutrientCategory.VITAMIN,
            subcategory=NutrientSubcategory.WATER_SOLUBLE_VITAMIN,
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 90.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 75.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 90.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 75.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 85.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 120.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 2000.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 2000.0,
            },
            optimal_range=(200.0, 1000.0),  # For immune support
            bioavailability_percentage=80.0,
            water_soluble=True,
            heat_stable=False,
            light_sensitive=True,
            oxygen_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILED: 50.0,
                CookingMethod.STEAMED: 25.0,
                CookingMethod.MICROWAVED: 20.0,
                CookingMethod.RAW: 0.0,
            },
            primary_functions=[
                "Collagen synthesis",
                "Antioxidant protection",
                "Iron absorption enhancement",
                "Immune function",
                "Wound healing",
                "Neurotransmitter synthesis",
                "Carnitine synthesis"
            ],
            deficiency_symptoms=[
                "Scurvy",
                "Poor wound healing",
                "Bleeding gums",
                "Easy bruising",
                "Fatigue",
                "Weakened immunity",
                "Joint pain"
            ],
            toxicity_symptoms=[
                "Diarrhea (high doses)",
                "Kidney stones (in susceptible individuals)",
                "Nausea"
            ],
            top_food_sources=[
                ("Guava", 228.0),
                ("Red bell pepper", 128.0),
                ("Kiwi", 93.0),
                ("Orange", 53.2),
                ("Strawberries", 58.8),
                ("Broccoli", 89.2),
                ("Kale", 120.0)
            ],
            absorption_enhancers=["Bioflavonoids"],
            absorption_inhibitors=["Smoking (depletes vitamin C)", "Stress"],
            synergistic_nutrients=["vitamin_e", "iron", "folate", "zinc"],
            evidence_level="A",
            research_references=["PMID: 29099763", "PMID: 28353648"]
        )
    
    # ========================================================================
    # FAT-SOLUBLE VITAMINS
    # ========================================================================
    
    def _add_fat_soluble_vitamins(self):
        """Add fat-soluble vitamin definitions (A, D, E, K)"""
        
        # Vitamin A (Retinol)
        self.nutrients["vitamin_a"] = NutrientReference(
            nutrient_id="vitamin_a",
            nutrient_name="Vitamin A (Retinol)",
            category=NutrientCategory.VITAMIN,
            subcategory=NutrientSubcategory.FAT_SOLUBLE_VITAMIN,
            unit=MeasurementUnit.MICROGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 900.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 700.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 900.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 700.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 770.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 1300.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 3000.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 3000.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 3000.0,  # Teratogenic at high doses
            },
            bioavailability_percentage=75.0,
            water_soluble=False,
            heat_stable=True,
            light_sensitive=True,
            oxygen_sensitive=True,
            primary_functions=[
                "Vision (rhodopsin formation)",
                "Immune function",
                "Cell differentiation",
                "Skin health",
                "Reproduction",
                "Gene expression"
            ],
            deficiency_symptoms=[
                "Night blindness",
                "Xerophthalmia (dry eyes)",
                "Keratomalacia (corneal damage)",
                "Impaired immunity",
                "Dry skin",
                "Poor growth"
            ],
            toxicity_symptoms=[
                "Hypervitaminosis A",
                "Birth defects (pregnancy)",
                "Liver damage",
                "Bone pain",
                "Skin changes"
            ],
            top_food_sources=[
                ("Beef liver", 9442.0),
                ("Sweet potato", 961.0),
                ("Carrots", 835.0),
                ("Spinach", 469.0),
                ("Kale", 681.0),
                ("Butternut squash", 532.0)
            ],
            absorption_enhancers=["Dietary fat", "Vitamin E", "Zinc"],
            absorption_inhibitors=["Low fat diet", "Bile acid sequestrants"],
            synergistic_nutrients=["vitamin_d", "vitamin_e", "zinc", "iron"],
            evidence_level="A",
            research_references=["PMID: 30200565", "PMID: 29644350"]
        )
        
        # Vitamin D (Calciferol)
        self.nutrients["vitamin_d"] = NutrientReference(
            nutrient_id="vitamin_d",
            nutrient_name="Vitamin D (Calciferol)",
            category=NutrientCategory.VITAMIN,
            subcategory=NutrientSubcategory.FAT_SOLUBLE_VITAMIN,
            unit=MeasurementUnit.MICROGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 15.0,  # 600 IU
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 15.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 15.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 15.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.MALE): 15.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.FEMALE): 15.0,
                (LifeStage.SENIOR_71_PLUS_YEARS, BiologicalSex.MALE): 20.0,  # 800 IU
                (LifeStage.SENIOR_71_PLUS_YEARS, BiologicalSex.FEMALE): 20.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 15.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 15.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 100.0,  # 4000 IU
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 100.0,
            },
            optimal_range=(20.0, 50.0),  # 800-2000 IU for sufficiency
            therapeutic_range=(50.0, 125.0),  # 2000-5000 IU for deficiency
            bioavailability_percentage=80.0,
            water_soluble=False,
            heat_stable=True,
            primary_functions=[
                "Calcium absorption",
                "Bone mineralization",
                "Immune modulation",
                "Cell growth regulation",
                "Inflammation reduction",
                "Muscle function",
                "Cardiovascular health"
            ],
            deficiency_symptoms=[
                "Rickets (children)",
                "Osteomalacia (adults)",
                "Osteoporosis",
                "Muscle weakness",
                "Bone pain",
                "Increased infection risk",
                "Depression"
            ],
            toxicity_symptoms=[
                "Hypercalcemia",
                "Kidney stones",
                "Nausea/vomiting",
                "Weakness",
                "Confusion"
            ],
            top_food_sources=[
                ("Salmon", 14.2),
                ("Sardines", 4.8),
                ("Cod liver oil", 250.0),
                ("Egg yolk", 2.0),
                ("Fortified milk", 1.3),
                ("Mushrooms (UV-exposed)", 10.0)
            ],
            absorption_enhancers=["Dietary fat", "Magnesium", "Vitamin K2"],
            absorption_inhibitors=["Low fat diet", "Kidney/liver disease"],
            synergistic_nutrients=["calcium", "magnesium", "vitamin_k2", "vitamin_a"],
            evidence_level="A",
            research_references=["PMID: 28149654", "PMID: 29699194"]
        )
        
        # Vitamin E (Tocopherol)
        self.nutrients["vitamin_e"] = NutrientReference(
            nutrient_id="vitamin_e",
            nutrient_name="Vitamin E (Alpha-Tocopherol)",
            category=NutrientCategory.VITAMIN,
            subcategory=NutrientSubcategory.FAT_SOLUBLE_VITAMIN,
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 15.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 15.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 15.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 15.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 15.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 19.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 1000.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 1000.0,
            },
            bioavailability_percentage=75.0,
            water_soluble=False,
            heat_stable=False,
            light_sensitive=True,
            oxygen_sensitive=True,
            cooking_loss_rates={
                CookingMethod.FRIED: 30.0,
                CookingMethod.BAKED: 20.0,
            },
            primary_functions=[
                "Antioxidant (protects cell membranes)",
                "Immune function",
                "Platelet aggregation regulation",
                "Gene expression",
                "Skin health",
                "Eye health"
            ],
            deficiency_symptoms=[
                "Peripheral neuropathy",
                "Ataxia",
                "Muscle weakness",
                "Vision problems",
                "Immune impairment",
                "Hemolytic anemia"
            ],
            toxicity_symptoms=[
                "Increased bleeding risk (high doses)",
                "Hemorrhagic stroke (rare)"
            ],
            top_food_sources=[
                ("Sunflower seeds", 35.2),
                ("Almonds", 25.6),
                ("Hazelnuts", 15.0),
                ("Wheat germ oil", 149.4),
                ("Spinach", 2.0),
                ("Avocado", 2.1)
            ],
            absorption_enhancers=["Dietary fat", "Vitamin C (regenerates vitamin E)"],
            absorption_inhibitors=["Mineral oil", "Orlistat"],
            synergistic_nutrients=["vitamin_c", "selenium", "vitamin_a"],
            antagonistic_nutrients=["Iron supplements (separate timing)", "Warfarin"],
            evidence_level="A",
            research_references=["PMID: 24473231", "PMID: 28716455"]
        )
        
        # Vitamin K (Phylloquinone K1 and Menaquinone K2)
        self.nutrients["vitamin_k"] = NutrientReference(
            nutrient_id="vitamin_k",
            nutrient_name="Vitamin K (Phylloquinone)",
            category=NutrientCategory.VITAMIN,
            subcategory=NutrientSubcategory.FAT_SOLUBLE_VITAMIN,
            unit=MeasurementUnit.MICROGRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 120.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 90.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 120.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 90.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 90.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 90.0,
            },
            optimal_range=(200.0, 1000.0),  # K2 for bone/cardiovascular health
            bioavailability_percentage=80.0,
            water_soluble=False,
            heat_stable=True,
            light_sensitive=True,
            primary_functions=[
                "Blood clotting (coagulation factors)",
                "Bone mineralization (osteocalcin activation)",
                "Cardiovascular health (prevents calcification)",
                "Cell growth regulation"
            ],
            deficiency_symptoms=[
                "Easy bleeding/bruising",
                "Heavy menstrual bleeding",
                "Blood in urine/stool",
                "Osteoporosis (K2 deficiency)",
                "Arterial calcification"
            ],
            top_food_sources=[
                ("Kale", 704.0),
                ("Spinach", 483.0),
                ("Collard greens", 623.0),
                ("Brussels sprouts", 194.0),
                ("Broccoli", 102.0),
                ("Natto (K2)", 1100.0),
                ("Cheese (K2)", 50.0)
            ],
            absorption_enhancers=["Dietary fat", "Vitamin D"],
            absorption_inhibitors=["Antibiotics (kill K-producing bacteria)", "Mineral oil"],
            synergistic_nutrients=["vitamin_d", "calcium", "vitamin_a"],
            antagonistic_nutrients=["Warfarin (vitamin K antagonist)"],
            evidence_level="A",
            research_references=["PMID: 27956652", "PMID: 30675873"]
        )
    
    def get_nutrient(self, nutrient_id: str) -> NutrientReference:
        """Retrieve a nutrient by ID"""
        return self.nutrients.get(nutrient_id)
    
    def get_all_nutrients(self) -> Dict[str, NutrientReference]:
        """Get all nutrients in database"""
        return self.nutrients
    
    def get_nutrients_by_category(self, category: NutrientCategory) -> Dict[str, NutrientReference]:
        """Get all nutrients in a specific category"""
        return {
            nid: nutrient for nid, nutrient in self.nutrients.items()
            if nutrient.category == category
        }
    
    def search_nutrients(self, query: str) -> Dict[str, NutrientReference]:
        """Search nutrients by name or synonym"""
        query_lower = query.lower()
        results = {}
        for nid, nutrient in self.nutrients.items():
            if (query_lower in nutrient.nutrient_name.lower() or
                any(query_lower in syn.lower() for syn in nutrient.synonyms)):
                results[nid] = nutrient
        return results
    
    # ========================================================================
    # MAJOR MINERALS
    # ========================================================================
    
    def _add_major_minerals(self):
        """Add major mineral definitions"""
        
        # Calcium
        self.nutrients["calcium"] = NutrientReference(
            nutrient_id="calcium",
            nutrient_name="Calcium",
            category=NutrientCategory.MINERAL,
            subcategory=NutrientSubcategory.MAJOR_MINERAL,
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 1000.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 1000.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 1000.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 1000.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.MALE): 1000.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.FEMALE): 1200.0,
                (LifeStage.SENIOR_71_PLUS_YEARS, BiologicalSex.MALE): 1200.0,
                (LifeStage.SENIOR_71_PLUS_YEARS, BiologicalSex.FEMALE): 1200.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 1000.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 1000.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 2500.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 2500.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.MALE): 2000.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.FEMALE): 2000.0,
            },
            optimal_range=(1000.0, 1500.0),
            bioavailability_percentage=30.0,  # Varies widely
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Bone and teeth mineralization",
                "Muscle contraction",
                "Nerve transmission",
                "Blood clotting",
                "Hormone secretion",
                "Enzyme activation",
                "Blood pressure regulation"
            ],
            deficiency_symptoms=[
                "Osteoporosis",
                "Osteopenia",
                "Rickets (children)",
                "Muscle cramps",
                "Numbness/tingling",
                "Abnormal heart rhythm"
            ],
            toxicity_symptoms=[
                "Kidney stones",
                "Hypercalcemia",
                "Constipation",
                "Impaired absorption of other minerals"
            ],
            top_food_sources=[
                ("Parmesan cheese", 1184.0),
                ("Sardines with bones", 382.0),
                ("Yogurt", 110.0),
                ("Milk", 113.0),
                ("Tofu (calcium-set)", 350.0),
                ("Kale", 150.0),
                ("Chia seeds", 631.0)
            ],
            absorption_enhancers=["Vitamin D", "Lactose", "Adequate protein", "Acidic pH"],
            absorption_inhibitors=["Phytates", "Oxalates", "High sodium", "Caffeine", "Phosphoric acid"],
            synergistic_nutrients=["vitamin_d", "vitamin_k2", "magnesium", "phosphorus"],
            antagonistic_nutrients=["Iron", "Zinc", "Magnesium (compete for absorption)"],
            evidence_level="A",
            research_references=["PMID: 28404575", "PMID: 30909722"]
        )
        
        # Phosphorus
        self.nutrients["phosphorus"] = NutrientReference(
            nutrient_id="phosphorus",
            nutrient_name="Phosphorus",
            category=NutrientCategory.MINERAL,
            subcategory=NutrientSubcategory.MAJOR_MINERAL,
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 700.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 700.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 700.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 700.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 700.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 700.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 4000.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 4000.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.MALE): 3000.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.FEMALE): 3000.0,
            },
            bioavailability_percentage=60.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Bone and teeth structure (hydroxyapatite)",
                "ATP energy production",
                "DNA/RNA structure",
                "Cell membrane phospholipids",
                "pH buffering",
                "Enzyme activation"
            ],
            deficiency_symptoms=[
                "Bone pain and weakness (rare)",
                "Loss of appetite",
                "Fatigue",
                "Rickets/osteomalacia",
                "Impaired growth"
            ],
            toxicity_symptoms=[
                "Hyperphosphatemia",
                "Calcification of soft tissues",
                "Impaired calcium absorption",
                "Kidney damage"
            ],
            top_food_sources=[
                ("Pumpkin seeds", 1233.0),
                ("Sunflower seeds", 660.0),
                ("Salmon", 252.0),
                ("Chicken breast", 220.0),
                ("Lentils", 180.0),
                ("Milk", 93.0)
            ],
            absorption_enhancers=["Vitamin D", "Adequate calcium"],
            absorption_inhibitors=["Aluminum antacids", "Excess calcium"],
            synergistic_nutrients=["calcium", "vitamin_d", "magnesium"],
            antagonistic_nutrients=["Excess calcium (Ca:P ratio important)"],
            evidence_level="A",
            research_references=["PMID: 25076495"]
        )
        
        # Magnesium
        self.nutrients["magnesium"] = NutrientReference(
            nutrient_id="magnesium",
            nutrient_name="Magnesium",
            category=NutrientCategory.MINERAL,
            subcategory=NutrientSubcategory.MAJOR_MINERAL,
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 400.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 310.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 420.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 320.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 350.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 310.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 350.0,  # From supplements only
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 350.0,
            },
            optimal_range=(400.0, 600.0),
            bioavailability_percentage=50.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Enzyme cofactor (300+ reactions)",
                "Energy production (ATP)",
                "Protein synthesis",
                "Muscle contraction/relaxation",
                "Nerve function",
                "Blood pressure regulation",
                "Glucose control",
                "Bone structure"
            ],
            deficiency_symptoms=[
                "Muscle cramps and spasms",
                "Fatigue",
                "Weakness",
                "Irregular heartbeat",
                "Numbness/tingling",
                "Personality changes",
                "Seizures (severe)"
            ],
            toxicity_symptoms=[
                "Diarrhea (from supplements)",
                "Nausea",
                "Hypotension",
                "Cardiac arrest (extreme cases)"
            ],
            top_food_sources=[
                ("Pumpkin seeds", 592.0),
                ("Almonds", 270.0),
                ("Spinach", 79.0),
                ("Cashews", 292.0),
                ("Black beans", 60.0),
                ("Dark chocolate", 228.0),
                ("Avocado", 29.0)
            ],
            absorption_enhancers=["Vitamin D", "Adequate protein", "Inulin"],
            absorption_inhibitors=["Phytates", "High calcium", "Alcohol", "PPIs"],
            synergistic_nutrients=["calcium", "vitamin_d", "vitamin_b6", "potassium"],
            antagonistic_nutrients=["Calcium (high doses compete)", "Zinc"],
            evidence_level="A",
            research_references=["PMID: 29882776", "PMID: 28471731"]
        )
        
        # Sodium
        self.nutrients["sodium"] = NutrientReference(
            nutrient_id="sodium",
            nutrient_name="Sodium",
            category=NutrientCategory.MINERAL,
            subcategory=NutrientSubcategory.MAJOR_MINERAL,
            unit=MeasurementUnit.MILLIGRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 1500.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 1500.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 1500.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 1500.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.MALE): 1300.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.FEMALE): 1300.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 2300.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 2300.0,
            },
            optimal_range=(1500.0, 2300.0),
            bioavailability_percentage=95.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Fluid balance",
                "Nerve impulse transmission",
                "Muscle contraction",
                "Blood pressure regulation",
                "Nutrient absorption",
                "pH balance"
            ],
            deficiency_symptoms=[
                "Hyponatremia",
                "Nausea/vomiting",
                "Headache",
                "Confusion",
                "Seizures",
                "Muscle weakness",
                "Coma (severe)"
            ],
            toxicity_symptoms=[
                "Hypernatremia",
                "Hypertension",
                "Fluid retention/edema",
                "Increased cardiovascular risk",
                "Kidney damage"
            ],
            top_food_sources=[
                ("Table salt", 38758.0),
                ("Soy sauce", 5493.0),
                ("Processed meats", 1000.0),
                ("Canned soups", 700.0),
                ("Bread", 477.0)
            ],
            absorption_enhancers=["Glucose (sodium-glucose cotransport)"],
            synergistic_nutrients=["potassium (balance important)", "chloride"],
            antagonistic_nutrients=["Excess sodium increases calcium excretion"],
            evidence_level="A",
            research_references=["PMID: 28135424", "PMID: 30813447"]
        )
        
        # Potassium
        self.nutrients["potassium"] = NutrientReference(
            nutrient_id="potassium",
            nutrient_name="Potassium",
            category=NutrientCategory.MINERAL,
            subcategory=NutrientSubcategory.MAJOR_MINERAL,
            unit=MeasurementUnit.MILLIGRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 3400.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 2600.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 3400.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 2600.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 2900.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 2800.0,
            },
            optimal_range=(3500.0, 4700.0),
            bioavailability_percentage=90.0,
            water_soluble=True,
            heat_stable=True,
            cooking_loss_rates={
                CookingMethod.BOILED: 50.0,  # Leaches into water
                CookingMethod.STEAMED: 20.0,
            },
            primary_functions=[
                "Nerve impulse transmission",
                "Muscle contraction",
                "Heart rhythm regulation",
                "Blood pressure control",
                "Fluid balance",
                "pH regulation",
                "Protein synthesis"
            ],
            deficiency_symptoms=[
                "Hypokalemia",
                "Muscle weakness/cramps",
                "Fatigue",
                "Constipation",
                "Irregular heartbeat",
                "Paralysis (severe)",
                "Respiratory failure"
            ],
            toxicity_symptoms=[
                "Hyperkalemia",
                "Nausea",
                "Irregular heartbeat",
                "Cardiac arrest (severe)"
            ],
            top_food_sources=[
                ("White beans", 561.0),
                ("Potato with skin", 535.0),
                ("Banana", 358.0),
                ("Spinach", 558.0),
                ("Avocado", 485.0),
                ("Sweet potato", 337.0),
                ("Salmon", 628.0)
            ],
            absorption_enhancers=["Adequate sodium"],
            absorption_inhibitors=["Alcohol", "Diuretics", "Chronic diarrhea"],
            synergistic_nutrients=["sodium (Na-K pump)", "magnesium"],
            antagonistic_nutrients=["Excess sodium depletes potassium"],
            evidence_level="A",
            research_references=["PMID: 31024555", "PMID: 28467925"]
        )
        
        # Chloride
        self.nutrients["chloride"] = NutrientReference(
            nutrient_id="chloride",
            nutrient_name="Chloride",
            category=NutrientCategory.MINERAL,
            subcategory=NutrientSubcategory.MAJOR_MINERAL,
            unit=MeasurementUnit.MILLIGRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 2300.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 2300.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 2300.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 2300.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.MALE): 2000.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.FEMALE): 2000.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 3600.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 3600.0,
            },
            bioavailability_percentage=95.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Fluid balance",
                "Acid-base balance (HCl in stomach)",
                "Nerve transmission",
                "Osmotic pressure maintenance",
                "Digestion (stomach acid)"
            ],
            deficiency_symptoms=[
                "Hypochloremia (rare)",
                "Metabolic alkalosis",
                "Dehydration",
                "Weakness"
            ],
            toxicity_symptoms=[
                "Hyperchloremia",
                "Hypertension",
                "Fluid retention"
            ],
            top_food_sources=[
                ("Table salt (NaCl)", 60270.0),
                ("Seaweed", 9019.0),
                ("Olives", 872.0),
                ("Tomato juice", 200.0)
            ],
            synergistic_nutrients=["sodium (NaCl)", "potassium (KCl)"],
            evidence_level="B",
            research_references=["PMID: 25716273"]
        )
        
        # Sulfur
        self.nutrients["sulfur"] = NutrientReference(
            nutrient_id="sulfur",
            nutrient_name="Sulfur",
            category=NutrientCategory.MINERAL,
            subcategory=NutrientSubcategory.MAJOR_MINERAL,
            unit=MeasurementUnit.MILLIGRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 800.0,  # Estimated
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 800.0,
            },
            bioavailability_percentage=80.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Amino acid structure (methionine, cysteine)",
                "Protein structure (disulfide bonds)",
                "Glutathione synthesis (antioxidant)",
                "Detoxification",
                "Collagen formation",
                "Insulin structure"
            ],
            deficiency_symptoms=[
                "Rare (obtained from protein)",
                "Impaired detoxification",
                "Joint problems"
            ],
            top_food_sources=[
                ("Garlic", 180.0),
                ("Onions", 65.0),
                ("Eggs", 200.0),
                ("Meat", 200.0),
                ("Cruciferous vegetables", 150.0),
                ("Seafood", 250.0)
            ],
            synergistic_nutrients=["Methionine", "Cysteine", "Vitamin B6"],
            primary_functions=[
                "Component of methionine and cysteine",
                "Disulfide bond formation in proteins",
                "Glutathione antioxidant synthesis",
                "Detoxification pathways",
                "Connective tissue structure"
            ],
            evidence_level="B",
            research_references=["PMID: 17854281"]
        )
    
    # ========================================================================
    # TRACE MINERALS
    # ========================================================================
    
    def _add_trace_minerals(self):
        """Add trace mineral definitions"""
        
        # Iron
        self.nutrients["iron"] = NutrientReference(
            nutrient_id="iron",
            nutrient_name="Iron",
            category=NutrientCategory.MINERAL,
            subcategory=NutrientSubcategory.TRACE_MINERAL,
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 8.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 18.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 8.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 18.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.MALE): 8.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.FEMALE): 8.0,  # Post-menopause
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 27.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 9.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 45.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 45.0,
            },
            bioavailability_percentage=18.0,  # Heme iron 25%, non-heme 10-15%
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Hemoglobin oxygen transport",
                "Myoglobin muscle oxygen",
                "Energy production (cytochromes)",
                "DNA synthesis",
                "Immune function",
                "Cognitive development"
            ],
            deficiency_symptoms=[
                "Iron deficiency anemia",
                "Fatigue and weakness",
                "Pale skin",
                "Shortness of breath",
                "Cold hands/feet",
                "Brittle nails",
                "Poor concentration",
                "Restless leg syndrome"
            ],
            toxicity_symptoms=[
                "Hemochromatosis",
                "Liver damage",
                "Diabetes",
                "Heart problems",
                "Joint pain",
                "Oxidative stress"
            ],
            top_food_sources=[
                ("Beef liver", 6.5),
                ("Oysters", 5.7),
                ("White beans", 5.1),
                ("Spinach", 2.7),
                ("Lentils", 3.3),
                ("Dark chocolate", 11.9),
                ("Red meat", 2.6)
            ],
            absorption_enhancers=["Vitamin C (converts Fe3+ to Fe2+)", "Meat factor", "Acidic pH"],
            absorption_inhibitors=["Phytates", "Tannins (tea/coffee)", "Calcium", "Polyphenols", "Zinc"],
            synergistic_nutrients=["vitamin_c", "vitamin_a", "copper", "vitamin_b12"],
            antagonistic_nutrients=["Calcium (competes)", "Zinc (competes)", "Vitamin E (separate timing)"],
            evidence_level="A",
            research_references=["PMID: 26797090", "PMID: 29442945"]
        )
        
        # Zinc
        self.nutrients["zinc"] = NutrientReference(
            nutrient_id="zinc",
            nutrient_name="Zinc",
            category=NutrientCategory.MINERAL,
            subcategory=NutrientSubcategory.TRACE_MINERAL,
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 11.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 8.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 11.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 8.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 11.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 12.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 40.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 40.0,
            },
            optimal_range=(15.0, 30.0),  # For immune support
            bioavailability_percentage=30.0,  # Animal sources 40-60%, plant 15-30%
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Immune function (300+ enzymes)",
                "Protein synthesis",
                "Wound healing",
                "DNA synthesis",
                "Cell division",
                "Taste and smell",
                "Testosterone production",
                "Cognitive function"
            ],
            deficiency_symptoms=[
                "Impaired immunity",
                "Hair loss",
                "Diarrhea",
                "Delayed wound healing",
                "Loss of taste/smell",
                "Skin lesions",
                "Growth retardation",
                "Hypogonadism in males"
            ],
            toxicity_symptoms=[
                "Nausea/vomiting",
                "Loss of appetite",
                "Copper deficiency (competitive)",
                "Lowered immunity",
                "HDL cholesterol reduction"
            ],
            top_food_sources=[
                ("Oysters", 74.0),
                ("Beef", 4.8),
                ("Pumpkin seeds", 7.6),
                ("Cashews", 5.6),
                ("Chickpeas", 1.5),
                ("Oats", 3.6)
            ],
            absorption_enhancers=["Animal protein", "Citric acid"],
            absorption_inhibitors=["Phytates", "Calcium", "Iron", "Copper", "Fiber"],
            synergistic_nutrients=["vitamin_a", "vitamin_b6", "copper"],
            antagonistic_nutrients=["Iron (high doses)", "Copper (compete)", "Calcium"],
            evidence_level="A",
            research_references=["PMID: 29186209", "PMID: 28515951"]
        )
        
        # Copper
        self.nutrients["copper"] = NutrientReference(
            nutrient_id="copper",
            nutrient_name="Copper",
            category=NutrientCategory.MINERAL,
            subcategory=NutrientSubcategory.TRACE_MINERAL,
            unit=MeasurementUnit.MICROGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 900.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 900.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 900.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 900.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 1000.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 1300.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 10000.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 10000.0,
            },
            bioavailability_percentage=50.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Iron metabolism (ceruloplasmin)",
                "Connective tissue formation",
                "Energy production (cytochrome c oxidase)",
                "Antioxidant defense (SOD)",
                "Neurotransmitter synthesis",
                "Melanin pigmentation",
                "Immune function"
            ],
            deficiency_symptoms=[
                "Anemia (impaired iron use)",
                "Neutropenia",
                "Bone abnormalities",
                "Hypopigmentation",
                "Neurological problems",
                "Growth retardation"
            ],
            toxicity_symptoms=[
                "Wilson's disease (genetic)",
                "Liver damage",
                "Nausea/vomiting",
                "Kidney damage"
            ],
            top_food_sources=[
                ("Beef liver", 14307.0),
                ("Oysters", 7620.0),
                ("Shiitake mushrooms", 5165.0),
                ("Cashews", 2195.0),
                ("Lentils", 251.0),
                ("Dark chocolate", 1766.0)
            ],
            absorption_enhancers=["Adequate protein", "Acidic pH"],
            absorption_inhibitors=["Excess zinc", "Vitamin C (high doses)", "Phytates"],
            synergistic_nutrients=["iron", "zinc (balance important)"],
            antagonistic_nutrients=["Zinc (compete for absorption)", "Molybdenum"],
            evidence_level="A",
            research_references=["PMID: 24259556", "PMID: 28299809"]
        )
        
        # Manganese
        self.nutrients["manganese"] = NutrientReference(
            nutrient_id="manganese",
            nutrient_name="Manganese",
            category=NutrientCategory.MINERAL,
            subcategory=NutrientSubcategory.TRACE_MINERAL,
            unit=MeasurementUnit.MILLIGRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 2.3,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 1.8,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 2.3,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 1.8,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 2.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 2.6,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 11.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 11.0,
            },
            bioavailability_percentage=5.0,  # Very low
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Bone formation",
                "Antioxidant defense (MnSOD)",
                "Amino acid metabolism",
                "Carbohydrate metabolism",
                "Cholesterol synthesis",
                "Wound healing"
            ],
            deficiency_symptoms=[
                "Impaired bone growth",
                "Skeletal abnormalities",
                "Impaired glucose tolerance",
                "Altered lipid metabolism",
                "Skin rash"
            ],
            toxicity_symptoms=[
                "Neurotoxicity (manganism - Parkinson's-like)",
                "Psychiatric symptoms",
                "Motor disturbances"
            ],
            top_food_sources=[
                ("Mussels", 6.8),
                ("Hazelnuts", 6.2),
                ("Brown rice", 1.1),
                ("Chickpeas", 1.0),
                ("Spinach", 0.9),
                ("Pineapple", 0.9)
            ],
            absorption_enhancers=["Adequate protein"],
            absorption_inhibitors=["Iron (competes)", "Calcium", "Phytates"],
            synergistic_nutrients=["vitamin_k", "chondroitin"],
            antagonistic_nutrients=["Iron (compete for absorption)"],
            evidence_level="B",
            research_references=["PMID: 23595983"]
        )
        
        # Selenium
        self.nutrients["selenium"] = NutrientReference(
            nutrient_id="selenium",
            nutrient_name="Selenium",
            category=NutrientCategory.MINERAL,
            subcategory=NutrientSubcategory.TRACE_MINERAL,
            unit=MeasurementUnit.MICROGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 55.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 55.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 55.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 55.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 60.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 70.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 400.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 400.0,
            },
            optimal_range=(100.0, 200.0),  # For cancer prevention
            bioavailability_percentage=80.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Antioxidant defense (glutathione peroxidase)",
                "Thyroid hormone metabolism",
                "DNA synthesis",
                "Immune function",
                "Reproduction",
                "Cancer prevention"
            ],
            deficiency_symptoms=[
                "Keshan disease (cardiomyopathy)",
                "Kashin-Beck disease (osteoarthropathy)",
                "Hypothyroidism",
                "Impaired immunity",
                "Infertility",
                "Muscle weakness"
            ],
            toxicity_symptoms=[
                "Selenosis",
                "Garlic breath odor",
                "Hair loss",
                "Nail brittleness",
                "Nausea",
                "Neurological damage"
            ],
            top_food_sources=[
                ("Brazil nuts", 1917.0),
                ("Tuna", 92.0),
                ("Halibut", 47.0),
                ("Sardines", 52.0),
                ("Sunflower seeds", 53.0),
                ("Chicken", 27.0)
            ],
            absorption_enhancers=["Vitamin E (synergy)", "Vitamin C"],
            absorption_inhibitors=["Heavy metals (mercury, cadmium)"],
            synergistic_nutrients=["vitamin_e", "vitamin_c", "iodine"],
            evidence_level="A",
            research_references=["PMID: 28724021", "PMID: 29186209"]
        )
        
        # Iodine
        self.nutrients["iodine"] = NutrientReference(
            nutrient_id="iodine",
            nutrient_name="Iodine",
            category=NutrientCategory.MINERAL,
            subcategory=NutrientSubcategory.TRACE_MINERAL,
            unit=MeasurementUnit.MICROGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 150.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 150.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 150.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 150.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 220.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 290.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 1100.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 1100.0,
            },
            bioavailability_percentage=90.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Thyroid hormone synthesis (T3, T4)",
                "Metabolism regulation",
                "Growth and development",
                "Brain development (fetal/infant)",
                "Body temperature regulation"
            ],
            deficiency_symptoms=[
                "Goiter (enlarged thyroid)",
                "Hypothyroidism",
                "Cretinism (congenital deficiency)",
                "Mental retardation",
                "Weight gain",
                "Fatigue",
                "Cold intolerance"
            ],
            toxicity_symptoms=[
                "Hyperthyroidism or hypothyroidism",
                "Goiter",
                "Thyroid inflammation",
                "Thyroid cancer risk"
            ],
            top_food_sources=[
                ("Seaweed (kelp)", 2984.0),
                ("Cod", 99.0),
                ("Yogurt", 75.0),
                ("Iodized salt", 4733.0),
                ("Shrimp", 35.0),
                ("Eggs", 24.0)
            ],
            absorption_enhancers=["Adequate selenium"],
            absorption_inhibitors=["Goitrogens (cruciferous vegetables)", "Soy", "Cassava"],
            synergistic_nutrients=["selenium", "iron", "vitamin_a"],
            antagonistic_nutrients=["Fluoride", "Chlorine", "Bromine"],
            evidence_level="A",
            research_references=["PMID: 28297385", "PMID: 29951594"]
        )
        
        # Chromium
        self.nutrients["chromium"] = NutrientReference(
            nutrient_id="chromium",
            nutrient_name="Chromium",
            category=NutrientCategory.MINERAL,
            subcategory=NutrientSubcategory.TRACE_MINERAL,
            unit=MeasurementUnit.MICROGRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 35.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 25.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 35.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 25.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.MALE): 30.0,
                (LifeStage.ADULT_51_70_YEARS, BiologicalSex.FEMALE): 20.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 30.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 45.0,
            },
            bioavailability_percentage=2.5,  # Very poor absorption
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Glucose metabolism (insulin potentiation)",
                "Lipid metabolism",
                "Protein metabolism",
                "Carbohydrate metabolism"
            ],
            deficiency_symptoms=[
                "Impaired glucose tolerance",
                "Elevated cholesterol",
                "Weight loss",
                "Peripheral neuropathy",
                "Confusion"
            ],
            top_food_sources=[
                ("Broccoli", 0.4),
                ("Grape juice", 0.8),
                ("Turkey breast", 1.7),
                ("Green beans", 1.1),
                ("Whole wheat bread", 1.0),
                ("Brewer's yeast", 60.0)
            ],
            absorption_enhancers=["Vitamin C", "Niacin"],
            absorption_inhibitors=["Phytates", "Antacids"],
            synergistic_nutrients=["vitamin_c", "vitamin_b3"],
            evidence_level="B",
            research_references=["PMID: 28150351", "PMID: 29882242"]
        )
        
        # Molybdenum
        self.nutrients["molybdenum"] = NutrientReference(
            nutrient_id="molybdenum",
            nutrient_name="Molybdenum",
            category=NutrientCategory.MINERAL,
            subcategory=NutrientSubcategory.TRACE_MINERAL,
            unit=MeasurementUnit.MICROGRAM,
            rda_values={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 45.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 45.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 45.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 45.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 50.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 50.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 2000.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 2000.0,
            },
            bioavailability_percentage=85.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Enzyme cofactor (sulfite oxidase, xanthine oxidase)",
                "Uric acid formation",
                "Sulfur amino acid metabolism",
                "Detoxification"
            ],
            deficiency_symptoms=[
                "Rare in humans",
                "Neurological problems",
                "Mental disturbances",
                "Elevated homocysteine"
            ],
            toxicity_symptoms=[
                "Gout-like symptoms",
                "Copper deficiency"
            ],
            top_food_sources=[
                ("Lima beans", 142.0),
                ("Lentils", 77.0),
                ("Peas", 55.0),
                ("Beef liver", 99.0),
                ("Bananas", 6.0)
            ],
            absorption_inhibitors=["Sulfur (competes)"],
            antagonistic_nutrients=["Copper (compete)"],
            evidence_level="B",
            research_references=["PMID: 24259558"]
        )
        
        # Fluoride
        self.nutrients["fluoride"] = NutrientReference(
            nutrient_id="fluoride",
            nutrient_name="Fluoride",
            category=NutrientCategory.MINERAL,
            subcategory=NutrientSubcategory.TRACE_MINERAL,
            unit=MeasurementUnit.MILLIGRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 4.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 3.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 4.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 3.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 3.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 3.0,
            },
            upper_limit={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 10.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 10.0,
            },
            bioavailability_percentage=90.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Tooth enamel strengthening",
                "Cavity prevention",
                "Bone mineralization"
            ],
            deficiency_symptoms=[
                "Increased dental caries",
                "Weakened tooth enamel"
            ],
            toxicity_symptoms=[
                "Dental fluorosis (children)",
                "Skeletal fluorosis",
                "Bone fractures",
                "Thyroid dysfunction (high doses)"
            ],
            top_food_sources=[
                ("Fluoridated water", 1.0),
                ("Tea", 0.3),
                ("Seafood", 0.3),
                ("Grapes", 0.2)
            ],
            absorption_inhibitors=["Calcium", "Magnesium"],
            antagonistic_nutrients=["Iodine"],
            evidence_level="A",
            research_references=["PMID: 28540924"]
        )
    
    # ========================================================================
    # ESSENTIAL AMINO ACIDS
    # ========================================================================
    
    def _add_essential_amino_acids(self):
        """Add essential amino acid definitions"""
        
        # Leucine
        self.nutrients["leucine"] = NutrientReference(
            nutrient_id="leucine",
            nutrient_name="Leucine",
            category=NutrientCategory.AMINO_ACID,
            subcategory=NutrientSubcategory.ESSENTIAL_AMINO_ACID,
            unit=MeasurementUnit.GRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 2.73,  # Per day for 70kg adult
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 2.1,  # Per day for 57kg adult
            },
            optimal_range=(2.0, 4.0),
            bioavailability_percentage=95.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Muscle protein synthesis (mTOR activation)",
                "Branched-chain amino acid (BCAA)",
                "Energy production",
                "Blood sugar regulation",
                "Wound healing",
                "Growth hormone production"
            ],
            deficiency_symptoms=[
                "Muscle wasting",
                "Fatigue",
                "Poor appetite",
                "Slow wound healing"
            ],
            top_food_sources=[
                ("Soybeans", 3.3),
                ("Beef", 1.7),
                ("Chicken", 1.9),
                ("Salmon", 1.6),
                ("Eggs", 1.1),
                ("Lentils", 1.8)
            ],
            synergistic_nutrients=["isoleucine", "valine", "vitamin_b6"],
            evidence_level="A",
            research_references=["PMID: 28596476", "PMID: 29910305"]
        )
        
        # Isoleucine
        self.nutrients["isoleucine"] = NutrientReference(
            nutrient_id="isoleucine",
            nutrient_name="Isoleucine",
            category=NutrientCategory.AMINO_ACID,
            subcategory=NutrientSubcategory.ESSENTIAL_AMINO_ACID,
            unit=MeasurementUnit.GRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 1.4,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 1.1,
            },
            optimal_range=(1.0, 2.5),
            bioavailability_percentage=95.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Muscle metabolism (BCAA)",
                "Energy regulation",
                "Hemoglobin synthesis",
                "Blood sugar regulation",
                "Immune function",
                "Wound healing"
            ],
            deficiency_symptoms=[
                "Muscle wasting",
                "Fatigue",
                "Dizziness",
                "Headaches",
                "Confusion"
            ],
            top_food_sources=[
                ("Soybeans", 2.0),
                ("Chicken", 1.3),
                ("Eggs", 0.7),
                ("Fish", 1.2),
                ("Lentils", 1.0)
            ],
            synergistic_nutrients=["leucine", "valine", "vitamin_b6"],
            evidence_level="A",
            research_references=["PMID: 28596476"]
        )
        
        # Valine
        self.nutrients["valine"] = NutrientReference(
            nutrient_id="valine",
            nutrient_name="Valine",
            category=NutrientCategory.AMINO_ACID,
            subcategory=NutrientSubcategory.ESSENTIAL_AMINO_ACID,
            unit=MeasurementUnit.GRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 1.82,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 1.4,
            },
            optimal_range=(1.5, 3.0),
            bioavailability_percentage=95.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Muscle growth and repair (BCAA)",
                "Energy production",
                "Nitrogen balance",
                "Cognitive function",
                "Nervous system regulation"
            ],
            deficiency_symptoms=[
                "Muscle wasting",
                "Insomnia",
                "Nervousness",
                "Poor mental function"
            ],
            top_food_sources=[
                ("Soybeans", 2.1),
                ("Beef", 1.4),
                ("Chicken", 1.2),
                ("Fish", 1.3),
                ("Eggs", 0.9)
            ],
            synergistic_nutrients=["leucine", "isoleucine", "vitamin_b6"],
            evidence_level="A",
            research_references=["PMID: 28596476"]
        )
        
        # Lysine
        self.nutrients["lysine"] = NutrientReference(
            nutrient_id="lysine",
            nutrient_name="Lysine",
            category=NutrientCategory.AMINO_ACID,
            subcategory=NutrientSubcategory.ESSENTIAL_AMINO_ACID,
            unit=MeasurementUnit.GRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 2.1,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 1.6,
            },
            optimal_range=(2.0, 3.0),
            therapeutic_range=(1.0, 3.0),  # For herpes suppression
            bioavailability_percentage=90.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Collagen formation",
                "Calcium absorption",
                "Carnitine production",
                "Immune function (antibodies)",
                "Hormone and enzyme synthesis",
                "Herpes virus suppression"
            ],
            deficiency_symptoms=[
                "Fatigue",
                "Nausea",
                "Dizziness",
                "Anemia",
                "Slow growth",
                "Reproductive disorders"
            ],
            top_food_sources=[
                ("Parmesan cheese", 3.3),
                ("Chicken", 2.4),
                ("Beef", 2.2),
                ("Tuna", 2.6),
                ("Lentils", 1.7),
                ("Eggs", 0.9)
            ],
            absorption_enhancers=["Vitamin C (for collagen synthesis)"],
            antagonistic_nutrients=["Arginine (compete for absorption, lysine:arginine ratio important for herpes)"],
            synergistic_nutrients=["vitamin_c", "iron", "vitamin_b6"],
            evidence_level="A",
            research_references=["PMID: 28638350", "PMID: 29151180"]
        )
        
        # Methionine
        self.nutrients["methionine"] = NutrientReference(
            nutrient_id="methionine",
            nutrient_name="Methionine",
            category=NutrientCategory.AMINO_ACID,
            subcategory=NutrientSubcategory.ESSENTIAL_AMINO_ACID,
            unit=MeasurementUnit.GRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 1.05,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 0.8,
            },
            optimal_range=(0.8, 2.0),
            bioavailability_percentage=90.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "SAM-e synthesis (methylation)",
                "Cysteine and taurine production",
                "Detoxification",
                "Antioxidant (glutathione precursor)",
                "Cartilage formation",
                "Fat metabolism"
            ],
            deficiency_symptoms=[
                "Fatty liver",
                "Edema",
                "Hair loss",
                "Skin lesions",
                "Weakness"
            ],
            top_food_sources=[
                ("Brazil nuts", 1.1),
                ("Fish", 0.8),
                ("Chicken", 0.8),
                ("Eggs", 0.4),
                ("Beef", 0.9)
            ],
            synergistic_nutrients=["vitamin_b6", "vitamin_b12", "folate", "choline"],
            evidence_level="A",
            research_references=["PMID: 26887394"]
        )
        
        # Phenylalanine
        self.nutrients["phenylalanine"] = NutrientReference(
            nutrient_id="phenylalanine",
            nutrient_name="Phenylalanine",
            category=NutrientCategory.AMINO_ACID,
            subcategory=NutrientSubcategory.ESSENTIAL_AMINO_ACID,
            unit=MeasurementUnit.GRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 1.4,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 1.1,
            },
            optimal_range=(1.0, 2.5),
            bioavailability_percentage=90.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Tyrosine synthesis",
                "Neurotransmitter production (dopamine, norepinephrine)",
                "Thyroid hormone synthesis",
                "Melanin pigmentation",
                "Pain relief",
                "Mood regulation"
            ],
            deficiency_symptoms=[
                "Depression",
                "Confusion",
                "Memory problems",
                "Decreased alertness",
                "Lack of appetite"
            ],
            top_food_sources=[
                ("Soybeans", 2.1),
                ("Beef", 1.3),
                ("Chicken", 1.2),
                ("Fish", 1.1),
                ("Eggs", 0.7)
            ],
            synergistic_nutrients=["vitamin_b6", "vitamin_c", "iron"],
            evidence_level="A",
            research_references=["PMID: 28249817"]
        )
        
        # Threonine
        self.nutrients["threonine"] = NutrientReference(
            nutrient_id="threonine",
            nutrient_name="Threonine",
            category=NutrientCategory.AMINO_ACID,
            subcategory=NutrientSubcategory.ESSENTIAL_AMINO_ACID,
            unit=MeasurementUnit.GRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 1.05,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 0.8,
            },
            optimal_range=(0.8, 2.0),
            bioavailability_percentage=90.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Collagen and elastin formation",
                "Tooth enamel protein",
                "Fat metabolism (liver)",
                "Immune function (antibodies)",
                "Digestive system health"
            ],
            deficiency_symptoms=[
                "Fatty liver",
                "Poor digestion",
                "Emotional agitation",
                "Confusion"
            ],
            top_food_sources=[
                ("Chicken", 1.2),
                ("Fish", 1.3),
                ("Beef", 1.2),
                ("Lentils", 0.9),
                ("Eggs", 0.6)
            ],
            synergistic_nutrients=["vitamin_b6"],
            evidence_level="B",
            research_references=["PMID: 27117852"]
        )
        
        # Tryptophan
        self.nutrients["tryptophan"] = NutrientReference(
            nutrient_id="tryptophan",
            nutrient_name="Tryptophan",
            category=NutrientCategory.AMINO_ACID,
            subcategory=NutrientSubcategory.ESSENTIAL_AMINO_ACID,
            unit=MeasurementUnit.GRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 0.28,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 0.22,
            },
            optimal_range=(0.25, 1.0),
            therapeutic_range=(0.5, 2.0),  # For sleep/mood
            bioavailability_percentage=85.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Serotonin synthesis (mood regulation)",
                "Melatonin synthesis (sleep)",
                "Niacin (vitamin B3) production",
                "Protein synthesis",
                "Growth hormone release"
            ],
            deficiency_symptoms=[
                "Depression",
                "Anxiety",
                "Insomnia",
                "Irritability",
                "Weight loss",
                "Growth retardation"
            ],
            top_food_sources=[
                ("Pumpkin seeds", 0.58),
                ("Chicken", 0.29),
                ("Tuna", 0.25),
                ("Oats", 0.18),
                ("Eggs", 0.17),
                ("Milk", 0.04)
            ],
            absorption_enhancers=["Carbohydrates (compete with other amino acids for BBB transport)"],
            absorption_inhibitors=["Other large neutral amino acids (compete)"],
            synergistic_nutrients=["vitamin_b6", "magnesium", "zinc"],
            evidence_level="A",
            research_references=["PMID: 26805875", "PMID: 27706093"]
        )
        
        # Histidine
        self.nutrients["histidine"] = NutrientReference(
            nutrient_id="histidine",
            nutrient_name="Histidine",
            category=NutrientCategory.AMINO_ACID,
            subcategory=NutrientSubcategory.ESSENTIAL_AMINO_ACID,
            unit=MeasurementUnit.GRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 0.7,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 0.55,
            },
            optimal_range=(0.5, 1.5),
            bioavailability_percentage=90.0,
            water_soluble=True,
            heat_stable=True,
            primary_functions=[
                "Histamine synthesis (immune response)",
                "Tissue growth and repair",
                "Myelin sheath protection",
                "Heavy metal detoxification",
                "Hemoglobin production",
                "Digestive enzyme production"
            ],
            deficiency_symptoms=[
                "Anemia",
                "Hearing problems",
                "Rheumatoid arthritis worsening",
                "Eczema"
            ],
            top_food_sources=[
                ("Chicken", 1.0),
                ("Beef", 1.1),
                ("Tuna", 0.9),
                ("Soybeans", 1.1),
                ("Eggs", 0.3)
            ],
            synergistic_nutrients=["vitamin_b6", "zinc"],
            evidence_level="B",
            research_references=["PMID: 28275201"]
        )
    
    def _add_non_essential_amino_acids(self):
        """Add non-essential amino acid definitions - placeholder for continuation"""
        # Alanine, Arginine, Asparagine, Aspartic acid, Cysteine, Glutamic acid, Glutamine, Glycine, Proline, Serine, Tyrosine
        pass
    
    # ========================================================================
    # ESSENTIAL FATTY ACIDS & OMEGA FATTY ACIDS
    # ========================================================================
    
    def _add_essential_fatty_acids(self):
        """Add essential fatty acid definitions"""
        
        # Linoleic Acid (Omega-6)
        self.nutrients["linoleic_acid"] = NutrientReference(
            nutrient_id="linoleic_acid",
            nutrient_name="Linoleic Acid (LA, Omega-6)",
            category=NutrientCategory.FATTY_ACID,
            subcategory=NutrientSubcategory.OMEGA6_FATTY_ACID,
            unit=MeasurementUnit.GRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 17.0,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 12.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 17.0,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 12.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 13.0,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 13.0,
            },
            optimal_range=(5.0, 15.0),  # Balance with omega-3
            bioavailability_percentage=95.0,
            water_soluble=False,
            heat_stable=False,
            oxygen_sensitive=True,
            primary_functions=[
                "Cell membrane structure",
                "Skin barrier function",
                "Prostaglandin synthesis",
                "Inflammatory response",
                "Growth and development",
                "Brain function"
            ],
            deficiency_symptoms=[
                "Dry scaly skin",
                "Hair loss",
                "Poor wound healing",
                "Growth retardation",
                "Impaired immunity"
            ],
            top_food_sources=[
                ("Safflower oil", 74.6),
                ("Sunflower oil", 65.7),
                ("Corn oil", 53.5),
                ("Soybean oil", 50.4),
                ("Walnuts", 38.1),
                ("Sunflower seeds", 23.1)
            ],
            synergistic_nutrients=["alpha_linolenic_acid (balance)", "vitamin_e"],
            antagonistic_nutrients=["Excess omega-6 competes with omega-3 conversion"],
            evidence_level="A",
            research_references=["PMID: 28900017", "PMID: 29610056"]
        )
        
        # Alpha-Linolenic Acid (Omega-3)
        self.nutrients["alpha_linolenic_acid"] = NutrientReference(
            nutrient_id="alpha_linolenic_acid",
            nutrient_name="Alpha-Linolenic Acid (ALA, Omega-3)",
            category=NutrientCategory.FATTY_ACID,
            subcategory=NutrientSubcategory.OMEGA3_FATTY_ACID,
            unit=MeasurementUnit.GRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 1.6,
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 1.1,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.MALE): 1.6,
                (LifeStage.ADULT_31_50_YEARS, BiologicalSex.FEMALE): 1.1,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 1.4,
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 1.3,
            },
            optimal_range=(2.0, 4.0),
            bioavailability_percentage=85.0,
            water_soluble=False,
            heat_stable=False,
            oxygen_sensitive=True,
            light_sensitive=True,
            primary_functions=[
                "EPA and DHA precursor (limited conversion 5-10%)",
                "Anti-inflammatory",
                "Heart health",
                "Brain function",
                "Cell membrane fluidity"
            ],
            deficiency_symptoms=[
                "Dry scaly skin",
                "Poor wound healing",
                "Visual problems",
                "Growth retardation",
                "Neurological abnormalities"
            ],
            top_food_sources=[
                ("Flaxseed oil", 53.4),
                ("Chia seeds", 17.8),
                ("Walnuts", 9.1),
                ("Hemp seeds", 8.7),
                ("Flaxseeds", 22.8),
                ("Canola oil", 9.1)
            ],
            absorption_enhancers=["Vitamin E (protects from oxidation)"],
            synergistic_nutrients=["epa", "dha", "vitamin_e"],
            antagonistic_nutrients=["Excess omega-6 impairs conversion to EPA/DHA"],
            evidence_level="A",
            research_references=["PMID: 29610056", "PMID: 28900017"]
        )
    
    def _add_omega_fatty_acids(self):
        """Add omega fatty acid definitions"""
        
        # EPA (Eicosapentaenoic Acid)
        self.nutrients["epa"] = NutrientReference(
            nutrient_id="epa",
            nutrient_name="EPA (Eicosapentaenoic Acid, Omega-3)",
            category=NutrientCategory.FATTY_ACID,
            subcategory=NutrientSubcategory.OMEGA3_FATTY_ACID,
            unit=MeasurementUnit.MILLIGRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 250.0,  # EPA+DHA combined
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 250.0,
            },
            optimal_range=(500.0, 2000.0),  # For cardiovascular health
            therapeutic_range=(1000.0, 4000.0),  # For inflammation/depression
            bioavailability_percentage=90.0,
            water_soluble=False,
            heat_stable=False,
            oxygen_sensitive=True,
            primary_functions=[
                "Anti-inflammatory (competes with arachidonic acid)",
                "Cardiovascular health",
                "Triglyceride reduction",
                "Blood pressure regulation",
                "Mental health (depression, ADHD)",
                "Immune modulation"
            ],
            deficiency_symptoms=[
                "Increased inflammation",
                "Cardiovascular risk",
                "Depression",
                "Cognitive decline",
                "Dry eyes"
            ],
            top_food_sources=[
                ("Salmon (wild)", 2506.0),
                ("Mackerel", 2670.0),
                ("Sardines", 1480.0),
                ("Anchovies", 1478.0),
                ("Herring", 1729.0),
                ("Fish oil supplement", 18000.0)
            ],
            absorption_enhancers=["Taken with meals (fat)", "Vitamin E"],
            synergistic_nutrients=["dha", "vitamin_e", "vitamin_d"],
            evidence_level="A",
            research_references=["PMID: 29610056", "PMID: 28900017", "PMID: 30678300"]
        )
        
        # DHA (Docosahexaenoic Acid)
        self.nutrients["dha"] = NutrientReference(
            nutrient_id="dha",
            nutrient_name="DHA (Docosahexaenoic Acid, Omega-3)",
            category=NutrientCategory.FATTY_ACID,
            subcategory=NutrientSubcategory.OMEGA3_FATTY_ACID,
            unit=MeasurementUnit.MILLIGRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 250.0,  # EPA+DHA combined
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 250.0,
                (LifeStage.PREGNANT, BiologicalSex.FEMALE): 300.0,  # Critical for fetal brain
                (LifeStage.LACTATING, BiologicalSex.FEMALE): 300.0,
            },
            optimal_range=(500.0, 2000.0),
            therapeutic_range=(1000.0, 3000.0),
            bioavailability_percentage=90.0,
            water_soluble=False,
            heat_stable=False,
            oxygen_sensitive=True,
            primary_functions=[
                "Brain structure (60% of brain fat is DHA)",
                "Retinal structure (93% of retina omega-3)",
                "Cognitive function and memory",
                "Fetal brain development",
                "Neuroprotection",
                "Anti-inflammatory",
                "Mental health"
            ],
            deficiency_symptoms=[
                "Cognitive decline",
                "Vision problems",
                "Depression",
                "ADHD symptoms",
                "Poor memory",
                "Developmental delays (infants)"
            ],
            top_food_sources=[
                ("Salmon (wild)", 2147.0),
                ("Mackerel", 1195.0),
                ("Sardines", 1480.0),
                ("Herring", 939.0),
                ("Tuna", 1141.0),
                ("Fish oil supplement", 12000.0),
                ("Algae oil (vegan)", 400.0)
            ],
            absorption_enhancers=["Taken with meals", "Vitamin E", "Phospholipid form"],
            synergistic_nutrients=["epa", "vitamin_e", "vitamin_d", "choline"],
            evidence_level="A",
            research_references=["PMID: 29610056", "PMID: 29145493", "PMID: 28900017"]
        )
        
        # Arachidonic Acid (Omega-6)
        self.nutrients["arachidonic_acid"] = NutrientReference(
            nutrient_id="arachidonic_acid",
            nutrient_name="Arachidonic Acid (AA, Omega-6)",
            category=NutrientCategory.FATTY_ACID,
            subcategory=NutrientSubcategory.OMEGA6_FATTY_ACID,
            unit=MeasurementUnit.MILLIGRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 0.0,  # Can be synthesized from LA
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 0.0,
            },
            optimal_range=(50.0, 300.0),  # Balance with EPA
            bioavailability_percentage=90.0,
            water_soluble=False,
            heat_stable=False,
            oxygen_sensitive=True,
            primary_functions=[
                "Inflammatory signaling (prostaglandins, leukotrienes)",
                "Brain function and learning",
                "Muscle growth",
                "Immune response",
                "Reproductive health"
            ],
            deficiency_symptoms=[
                "Rare (synthesized from linoleic acid)",
                "Impaired wound healing",
                "Growth problems"
            ],
            toxicity_symptoms=[
                "Excessive inflammation",
                "Increased cardiovascular risk",
                "Autoimmune exacerbation"
            ],
            top_food_sources=[
                ("Egg yolk", 297.0),
                ("Beef", 69.0),
                ("Chicken thigh", 99.0),
                ("Pork", 59.0),
                ("Turkey", 61.0)
            ],
            synergistic_nutrients=["epa (balances inflammatory response)"],
            antagonistic_nutrients=["EPA (competes for same enzymes)"],
            evidence_level="B",
            research_references=["PMID: 29145493"]
        )
        
        # GLA (Gamma-Linolenic Acid)
        self.nutrients["gla"] = NutrientReference(
            nutrient_id="gla",
            nutrient_name="GLA (Gamma-Linolenic Acid, Omega-6)",
            category=NutrientCategory.FATTY_ACID,
            subcategory=NutrientSubcategory.OMEGA6_FATTY_ACID,
            unit=MeasurementUnit.MILLIGRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 0.0,  # Not essential
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 0.0,
            },
            optimal_range=(240.0, 960.0),  # Supplemental doses
            therapeutic_range=(500.0, 2800.0),  # For inflammation
            bioavailability_percentage=85.0,
            water_soluble=False,
            heat_stable=False,
            oxygen_sensitive=True,
            primary_functions=[
                "Anti-inflammatory (DGLA pathway)",
                "Skin health (moisture barrier)",
                "Hormone balance (PMS relief)",
                "Nerve function",
                "Immune regulation"
            ],
            deficiency_symptoms=[
                "Dry skin",
                "Eczema",
                "Brittle nails",
                "PMS symptoms"
            ],
            top_food_sources=[
                ("Evening primrose oil", 8000.0),
                ("Borage oil", 20000.0),
                ("Black currant oil", 15000.0),
                ("Hemp seed oil", 2000.0)
            ],
            absorption_enhancers=["Vitamin E", "Magnesium", "Vitamin B6", "Zinc"],
            synergistic_nutrients=["vitamin_e", "magnesium", "vitamin_b6"],
            evidence_level="B",
            research_references=["PMID: 25500434", "PMID: 28716455"]
        )
        
        # Oleic Acid (Omega-9)
        self.nutrients["oleic_acid"] = NutrientReference(
            nutrient_id="oleic_acid",
            nutrient_name="Oleic Acid (Omega-9)",
            category=NutrientCategory.FATTY_ACID,
            subcategory=NutrientSubcategory.OMEGA9_FATTY_ACID,
            unit=MeasurementUnit.GRAM,
            adequate_intake={
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.MALE): 0.0,  # Non-essential
                (LifeStage.ADULT_19_30_YEARS, BiologicalSex.FEMALE): 0.0,
            },
            optimal_range=(10.0, 30.0),  # As part of healthy fat intake
            bioavailability_percentage=95.0,
            water_soluble=False,
            heat_stable=True,
            oxygen_sensitive=False,
            primary_functions=[
                "LDL cholesterol reduction",
                "HDL cholesterol maintenance",
                "Insulin sensitivity",
                "Anti-inflammatory",
                "Cell membrane structure",
                "Energy source"
            ],
            top_food_sources=[
                ("Olive oil", 73.0),
                ("Avocado oil", 70.5),
                ("Almonds", 31.5),
                ("Cashews", 27.3),
                ("Avocado", 9.8),
                ("Pecans", 40.8)
            ],
            synergistic_nutrients=["polyphenols (in olive oil)", "vitamin_e"],
            evidence_level="A",
            research_references=["PMID: 29145493", "PMID: 28716455"]
        )
    
    def _add_non_essential_amino_acids(self):
        """Add non-essential amino acid definitions"""
        # Note: These are synthesized by the body but can be conditionally essential
        
        # L-Arginine - Conditionally essential, critical for immune function, wound healing, nitric oxide production
        self.nutrients["arginine"] = Nutrient(
            nutrient_id="arginine",
            name="L-Arginine",
            category=NutrientCategory.AMINO_ACIDS,
            subcategory="Non-Essential Amino Acids",
            unit=MeasurementUnit.GRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("5.0"),
                    upper_limit=Decimal("20.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Conditionally essential during stress, illness, wound healing"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("5.0"),
                    upper_limit=Decimal("20.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
                (LifeStage.PREGNANCY, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("6.0"),
                    upper_limit=Decimal("20.0"),
                    life_stage=LifeStage.PREGNANCY,
                    biological_sex=BiologicalSex.FEMALE,
                    notes="Increased needs for fetal development"
                ),
            },
            bioavailability_percentage=Decimal("68.0"),
            absorption_enhancers=["Vitamin B6", "Vitamin C", "Magnesium"],
            absorption_inhibitors=["Lysine (competitive)", "High protein meals"],
            synergistic_nutrients=["L-Citrulline", "Nitric oxide cofactors", "B-vitamins"],
            antagonistic_nutrients=["L-Lysine (arginine:lysine ratio important for herpes management)"],
            primary_functions=[
                "Nitric oxide synthesis (vasodilation, blood pressure regulation)",
                "Immune function (T-cell proliferation, macrophage activation)",
                "Wound healing and collagen synthesis",
                "Growth hormone secretion",
                "Insulin secretion regulation",
                "Ammonia detoxification (urea cycle)",
                "Creatine synthesis precursor"
            ],
            deficiency_symptoms=[
                "Impaired wound healing",
                "Hair loss",
                "Skin rashes",
                "Compromised immune function",
                "Muscle weakness",
                "Fatigue"
            ],
            toxicity_symptoms=[
                "Gastrointestinal discomfort >10g/day",
                "Diarrhea",
                "Potential worsening of herpes outbreaks (high arginine:lysine ratio)",
                "Hypotension (excessive vasodilation)"
            ],
            food_sources=[
                ("Turkey breast", Decimal("2.5")),
                ("Pork loin", Decimal("2.2")),
                ("Chicken breast", Decimal("2.0")),
                ("Pumpkin seeds", Decimal("5.4")),
                ("Soybeans", Decimal("2.6")),
                ("Peanuts", Decimal("3.5")),
                ("Spirulina", Decimal("4.3")),
                ("Dairy products", Decimal("1.2"))
            ],
            heat_stable=True,
            water_soluble=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("15.0"),
                CookingMethod.STEAMING: Decimal("8.0"),
                CookingMethod.BAKING: Decimal("5.0"),
                CookingMethod.GRILLING: Decimal("10.0"),
                CookingMethod.FRYING: Decimal("12.0")
            },
            research_references=["PMID: 15531486", "PMID: 11844977", "PMID: 23595206"]
        )
        
        # L-Glutamine - Most abundant amino acid, critical for gut health and immune function
        self.nutrients["glutamine"] = Nutrient(
            nutrient_id="glutamine",
            name="L-Glutamine",
            category=NutrientCategory.AMINO_ACIDS,
            subcategory="Non-Essential Amino Acids",
            unit=MeasurementUnit.GRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("5.0"),
                    upper_limit=Decimal("40.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Conditionally essential during critical illness, intense training"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("5.0"),
                    upper_limit=Decimal("40.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("70.0"),
            absorption_enhancers=["B-vitamins", "Vitamin C", "Zinc"],
            absorption_inhibitors=["Alcohol", "NSAIDs (damage gut lining)"],
            synergistic_nutrients=["N-Acetyl Glucosamine", "Zinc", "Vitamin A", "Probiotics"],
            antagonistic_nutrients=["Excessive glutamate (conversion product)"],
            primary_functions=[
                "Intestinal health (primary fuel for enterocytes)",
                "Immune cell function (lymphocytes, macrophages)",
                "Protein synthesis",
                "Nitrogen transport",
                "Acid-base balance regulation",
                "Neurotransmitter precursor (glutamate, GABA)",
                "Maintains gut barrier integrity (reduces intestinal permeability)"
            ],
            deficiency_symptoms=[
                "Leaky gut syndrome (increased intestinal permeability)",
                "Impaired immune function",
                "Muscle wasting",
                "Slow wound healing",
                "Increased infection susceptibility",
                "Digestive issues"
            ],
            toxicity_symptoms=[
                "Generally well tolerated up to 40g/day",
                "Potential manic episodes in bipolar disorder",
                "Headaches at very high doses",
                "Restlessness or anxiety (conversion to glutamate)"
            ],
            food_sources=[
                ("Beef", Decimal("1.2")),
                ("Chicken", Decimal("1.1")),
                ("Fish", Decimal("0.9")),
                ("Eggs", Decimal("0.6")),
                ("Dairy products", Decimal("0.8")),
                ("Cabbage", Decimal("0.4")),
                ("Bone broth", Decimal("6.0")),
                ("Whey protein", Decimal("4.0"))
            ],
            heat_stable=False,
            water_soluble=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("25.0"),
                CookingMethod.STEAMING: Decimal("15.0"),
                CookingMethod.BAKING: Decimal("10.0"),
                CookingMethod.GRILLING: Decimal("18.0"),
                CookingMethod.FRYING: Decimal("20.0")
            },
            research_references=["PMID: 11157239", "PMID: 11844977", "PMID: 18806480"]
        )
        
        # L-Cysteine - Sulfur-containing amino acid, precursor to glutathione
        self.nutrients["cysteine"] = Nutrient(
            nutrient_id="cysteine",
            name="L-Cysteine",
            category=NutrientCategory.AMINO_ACIDS,
            subcategory="Non-Essential Amino Acids (Sulfur-Containing)",
            unit=MeasurementUnit.GRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("1.5"),
                    upper_limit=Decimal("7.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Can be synthesized from methionine"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("1.5"),
                    upper_limit=Decimal("7.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("85.0"),
            absorption_enhancers=["Vitamin B6", "Selenium", "Vitamin C"],
            absorption_inhibitors=["Heavy metals", "Alcohol"],
            synergistic_nutrients=["Selenium", "Glycine", "Glutamic acid (for glutathione synthesis)"],
            antagonistic_nutrients=["Excessive oxidative stress depletes cysteine reserves"],
            primary_functions=[
                "Glutathione synthesis (master antioxidant)",
                "Detoxification (heavy metals, xenobiotics)",
                "Protein structure (disulfide bonds in keratin, collagen)",
                "Taurine synthesis precursor",
                "Antioxidant protection",
                "Immune function support",
                "Hair, skin, and nail health"
            ],
            deficiency_symptoms=[
                "Weak, brittle hair",
                "Skin problems",
                "Slow wound healing",
                "Weakened immune system",
                "Liver dysfunction",
                "Impaired detoxification"
            ],
            toxicity_symptoms=[
                "Rare with dietary intake",
                "Nausea at high supplemental doses (>7g/day)",
                "Hyperhomocysteinemia if B-vitamin status poor"
            ],
            food_sources=[
                ("Chicken breast", Decimal("0.3")),
                ("Turkey", Decimal("0.3")),
                ("Eggs", Decimal("0.3")),
                ("Dairy products", Decimal("0.2")),
                ("Sunflower seeds", Decimal("0.5")),
                ("Legumes", Decimal("0.2")),
                ("Oats", Decimal("0.4")),
                ("Broccoli", Decimal("0.05"))
            ],
            heat_stable=True,
            water_soluble=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("10.0"),
                CookingMethod.STEAMING: Decimal("5.0"),
                CookingMethod.BAKING: Decimal("8.0"),
                CookingMethod.GRILLING: Decimal("12.0"),
                CookingMethod.FRYING: Decimal("15.0")
            },
            research_references=["PMID: 9230863", "PMID: 17569207"]
        )
        
        # Taurine - Sulfur-containing beta-amino acid
        self.nutrients["taurine"] = Nutrient(
            nutrient_id="taurine",
            name="Taurine",
            category=NutrientCategory.AMINO_ACIDS,
            subcategory="Sulfur-Containing Beta-Amino Acid",
            unit=MeasurementUnit.GRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("0.5"),
                    upper_limit=Decimal("3.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Conditionally essential for vegetarians/vegans, infants"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("0.5"),
                    upper_limit=Decimal("3.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
                (LifeStage.INFANT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("0.04"),
                    upper_limit=Decimal("0.1"),
                    life_stage=LifeStage.INFANT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Essential for infants - limited synthesis capacity"
                ),
            },
            bioavailability_percentage=Decimal("95.0"),
            absorption_enhancers=["Zinc", "Vitamin B6"],
            absorption_inhibitors=["Beta-alanine (competes for absorption)"],
            synergistic_nutrients=["Magnesium", "Potassium", "B-vitamins"],
            antagonistic_nutrients=["Beta-alanine (competitive transporter)"],
            primary_functions=[
                "Cardiovascular function (contractility, blood pressure regulation)",
                "Bile acid conjugation (fat digestion)",
                "Antioxidant and anti-inflammatory effects",
                "Neurological development and function",
                "Retinal health (photoreceptor function)",
                "Calcium homeostasis regulation",
                "Membrane stabilization"
            ],
            deficiency_symptoms=[
                "Retinal degeneration",
                "Cardiomyopathy",
                "Developmental delays in infants",
                "Impaired fat digestion",
                "Increased oxidative stress"
            ],
            toxicity_symptoms=[
                "Very safe, no established toxicity",
                "Doses up to 10g/day well tolerated in studies"
            ],
            food_sources=[
                ("Shellfish (scallops, clams)", Decimal("0.8")),
                ("Fish (tuna, cod)", Decimal("0.16")),
                ("Chicken (dark meat)", Decimal("0.17")),
                ("Beef", Decimal("0.04")),
                ("Seaweed (nori)", Decimal("1.3")),
                ("Energy drinks (added)", Decimal("0.32"))
            ],
            heat_stable=True,
            water_soluble=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("20.0"),
                CookingMethod.STEAMING: Decimal("10.0"),
                CookingMethod.BAKING: Decimal("5.0"),
                CookingMethod.GRILLING: Decimal("8.0"),
                CookingMethod.FRYING: Decimal("15.0")
            },
            research_references=["PMID: 2708042", "PMID: 22051450", "PMID: 28446680"]
        )
        
        # L-Carnitine - Critical for fat metabolism
        self.nutrients["carnitine"] = Nutrient(
            nutrient_id="carnitine",
            name="L-Carnitine",
            category=NutrientCategory.AMINO_ACIDS,
            subcategory="Quaternary Ammonium Compound",
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("15.0"),
                    upper_limit=Decimal("2000.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Synthesized from lysine and methionine"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("15.0"),
                    upper_limit=Decimal("2000.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("75.0"),
            absorption_enhancers=["Vitamin C", "Iron", "Vitamin B6", "Niacin"],
            absorption_inhibitors=["Vegetarian/vegan diet (limited intake)"],
            synergistic_nutrients=["CoQ10", "Alpha-lipoic acid", "Omega-3 fatty acids"],
            antagonistic_nutrients=["D-carnitine (blocks L-carnitine function)"],
            primary_functions=[
                "Fatty acid transport into mitochondria (beta-oxidation)",
                "Energy production from fats",
                "Cardiovascular health",
                "Brain function (acetylcholine synthesis)",
                "Male fertility (sperm motility)",
                "Muscle recovery"
            ],
            deficiency_symptoms=[
                "Muscle weakness and fatigue",
                "Hypoglycemia",
                "Cardiomyopathy",
                "Impaired fat metabolism",
                "Male infertility"
            ],
            toxicity_symptoms=[
                "Fishy body odor (TMAO production)",
                "Nausea and diarrhea at high doses (>3g/day)"
            ],
            food_sources=[
                ("Beef steak", Decimal("95.0")),
                ("Ground beef", Decimal("94.0")),
                ("Pork", Decimal("27.0")),
                ("Chicken breast", Decimal("3.9")),
                ("Whole milk", Decimal("8.0")),
                ("Codfish", Decimal("5.6"))
            ],
            heat_stable=True,
            water_soluble=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("15.0"),
                CookingMethod.STEAMING: Decimal("8.0"),
                CookingMethod.BAKING: Decimal("10.0"),
                CookingMethod.GRILLING: Decimal("20.0"),
                CookingMethod.FRYING: Decimal("18.0")
            },
            research_references=["PMID: 16364791", "PMID: 21519831", "PMID: 23597877"]
        )
        
        # L-Tyrosine - Precursor to catecholamines and thyroid hormones
        self.nutrients["tyrosine"] = Nutrient(
            nutrient_id="tyrosine",
            name="L-Tyrosine",
            category=NutrientCategory.AMINO_ACIDS,
            subcategory="Aromatic Non-Essential Amino Acid",
            unit=MeasurementUnit.GRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("1.5"),
                    upper_limit=Decimal("10.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Combined with phenylalanine requirement"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("1.5"),
                    upper_limit=Decimal("10.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("80.0"),
            absorption_enhancers=["Vitamin B6", "Folate", "Copper", "Vitamin C"],
            absorption_inhibitors=["Other large neutral amino acids (competitive transport)"],
            synergistic_nutrients=["Phenylalanine (precursor)", "Vitamin C", "Folate"],
            antagonistic_nutrients=["Tryptophan (competitive BBB transport)"],
            primary_functions=[
                "Catecholamine synthesis (dopamine, norepinephrine, epinephrine)",
                "Thyroid hormone production (T3, T4)",
                "Melanin synthesis (skin pigmentation)",
                "Cognitive performance under stress",
                "Mood regulation",
                "Alertness and focus"
            ],
            deficiency_symptoms=[
                "Depression and low mood",
                "Cognitive impairment",
                "Low motivation",
                "Hypothyroidism symptoms",
                "Poor stress response"
            ],
            toxicity_symptoms=[
                "Migraine headaches (>150mg/kg)",
                "Nausea",
                "Anxiety or jitteriness",
                "Interaction with MAO inhibitors (hypertensive crisis)"
            ],
            food_sources=[
                ("Parmesan cheese", Decimal("1.9")),
                ("Soybeans", Decimal("1.5")),
                ("Beef", Decimal("1.2")),
                ("Chicken", Decimal("1.1")),
                ("Fish (salmon)", Decimal("1.3")),
                ("Pumpkin seeds", Decimal("1.0")),
                ("Eggs", Decimal("0.5"))
            ],
            heat_stable=True,
            water_soluble=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("12.0"),
                CookingMethod.STEAMING: Decimal("6.0"),
                CookingMethod.BAKING: Decimal("8.0"),
                CookingMethod.GRILLING: Decimal("10.0"),
                CookingMethod.FRYING: Decimal("15.0")
            },
            research_references=["PMID: 2736402", "PMID: 22071706"]
        )
        
        # Glycine - Smallest amino acid, important for collagen
        self.nutrients["glycine"] = Nutrient(
            nutrient_id="glycine",
            name="Glycine",
            category=NutrientCategory.AMINO_ACIDS,
            subcategory="Non-Essential Amino Acid",
            unit=MeasurementUnit.GRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("3.0"),
                    upper_limit=Decimal("15.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Conditionally essential - synthesis often insufficient"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("3.0"),
                    upper_limit=Decimal("15.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("90.0"),
            absorption_enhancers=["Vitamin B6", "Folate"],
            absorption_inhibitors=["Minimal"],
            synergistic_nutrients=["Proline", "Lysine", "Vitamin C (for collagen synthesis)"],
            antagonistic_nutrients=["Excessive methionine"],
            primary_functions=[
                "Collagen synthesis (33% of collagen is glycine)",
                "Inhibitory neurotransmitter in CNS",
                "Glutathione synthesis component",
                "Creatine synthesis precursor",
                "Heme synthesis (for hemoglobin)",
                "Bile acid conjugation",
                "Sleep quality improvement"
            ],
            deficiency_symptoms=[
                "Poor wound healing",
                "Joint pain and stiffness",
                "Skin problems",
                "Sleep disturbances",
                "Impaired immune function"
            ],
            toxicity_symptoms=[
                "Very safe, up to 60g/day studied",
                "Mild GI upset at very high doses"
            ],
            food_sources=[
                ("Gelatin powder", Decimal("27.0")),
                ("Pork skin (cooked)", Decimal("11.9")),
                ("Chicken skin", Decimal("9.0")),
                ("Bone broth", Decimal("3.0")),
                ("Sesame seeds", Decimal("2.0")),
                ("Beef", Decimal("1.8")),
                ("Soybeans", Decimal("1.7"))
            ],
            heat_stable=True,
            water_soluble=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("5.0"),
                CookingMethod.STEAMING: Decimal("3.0"),
                CookingMethod.BAKING: Decimal("2.0"),
                CookingMethod.GRILLING: Decimal("4.0"),
                CookingMethod.FRYING: Decimal("6.0")
            },
            research_references=["PMID: 22855389", "PMID: 23595206"]
        )
        
        # L-Proline - Important for collagen structure
        self.nutrients["proline"] = Nutrient(
            nutrient_id="proline",
            name="L-Proline",
            category=NutrientCategory.AMINO_ACIDS,
            subcategory="Non-Essential Amino Acid",
            unit=MeasurementUnit.GRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("2.0"),
                    upper_limit=Decimal("10.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Conditionally essential for collagen synthesis"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("2.0"),
                    upper_limit=Decimal("10.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("85.0"),
            absorption_enhancers=["Vitamin C", "Glycine"],
            absorption_inhibitors=["Minimal"],
            synergistic_nutrients=["Glycine", "Lysine", "Vitamin C"],
            antagonistic_nutrients=["None significant"],
            primary_functions=[
                "Collagen synthesis (major component)",
                "Wound healing",
                "Skin health and elasticity",
                "Joint health",
                "Cardiovascular health",
                "Cartilage formation"
            ],
            deficiency_symptoms=[
                "Joint pain",
                "Skin issues (reduced elasticity)",
                "Slow wound healing",
                "Weak blood vessels"
            ],
            toxicity_symptoms=[
                "Very low toxicity",
                "No adverse effects at normal dietary levels"
            ],
            food_sources=[
                ("Gelatin", Decimal("16.0")),
                ("Beef (collagen-rich cuts)", Decimal("3.5")),
                ("Chicken skin", Decimal("5.0")),
                ("Bone broth", Decimal("2.0")),
                ("Dairy", Decimal("1.2")),
                ("Wheat germ", Decimal("2.1")),
                ("Soybeans", Decimal("2.4"))
            ],
            heat_stable=True,
            water_soluble=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("8.0"),
                CookingMethod.STEAMING: Decimal("4.0"),
                CookingMethod.BAKING: Decimal("5.0"),
                CookingMethod.GRILLING: Decimal("7.0"),
                CookingMethod.FRYING: Decimal("10.0")
            },
            research_references=["PMID: 19402938", "PMID: 22403395"]
        )
    
    def _add_phytonutrients(self):
        """Add phytonutrient definitions"""
        
        # Resveratrol - Polyphenol with cardiovascular and longevity benefits
        self.nutrients["resveratrol"] = Nutrient(
            nutrient_id="resveratrol",
            name="Resveratrol",
            category=NutrientCategory.PHYTONUTRIENTS,
            subcategory="Polyphenols (Stilbenes)",
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("200.0"),
                    upper_limit=Decimal("1500.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Optimal intake range; no established RDA"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("200.0"),
                    upper_limit=Decimal("1500.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("20.0"),
            absorption_enhancers=["Piperine (black pepper)", "Quercetin", "Healthy fats"],
            absorption_inhibitors=["High fiber meals", "Some gut bacteria degrade it"],
            synergistic_nutrients=["Quercetin", "Curcumin", "Vitamin E", "CoQ10"],
            antagonistic_nutrients=["None significant"],
            primary_functions=[
                "Activates SIRT1 (longevity gene - NAD+ dependent deacetylase)",
                "Cardiovascular protection (anti-atherosclerotic)",
                "Antioxidant and anti-inflammatory effects",
                "Neuroprotection (crosses blood-brain barrier)",
                "Cancer prevention (multiple mechanisms)",
                "Insulin sensitivity improvement",
                "Mimics caloric restriction benefits",
                "Anti-aging effects on mitochondria"
            ],
            deficiency_symptoms=[
                "No classical deficiency (not essential nutrient)",
                "Increased oxidative stress",
                "Reduced cardiovascular protection",
                "Accelerated aging markers"
            ],
            toxicity_symptoms=[
                "Generally safe up to 5g/day in studies",
                "Potential GI upset at very high doses",
                "May interact with blood thinners (antiplatelet effect)",
                "Possible estrogenic effects at high doses"
            ],
            food_sources=[
                ("Red wine", Decimal("1.5")),
                ("Red grapes (skin)", Decimal("0.24")),
                ("Grape juice (red)", Decimal("0.5")),
                ("Peanuts (with skin)", Decimal("0.15")),
                ("Cocoa powder", Decimal("0.28")),
                ("Blueberries", Decimal("0.04")),
                ("Cranberries", Decimal("0.06")),
                ("Pistachios", Decimal("0.09")),
                ("Dark chocolate (70%+)", Decimal("0.35")),
                ("Japanese knotweed extract", Decimal("500.0"))
            ],
            heat_stable=False,
            water_soluble=False,
            light_sensitive=True,
            oxygen_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("50.0"),
                CookingMethod.STEAMING: Decimal("30.0"),
                CookingMethod.BAKING: Decimal("25.0"),
                CookingMethod.GRILLING: Decimal("35.0"),
                CookingMethod.FRYING: Decimal("40.0")
            },
            research_references=["PMID: 15604281", "PMID: 23595206", "PMID: 25720716", "PMID: 27259333"]
        )
        
        # Quercetin - Flavonoid with powerful anti-inflammatory effects
        self.nutrients["quercetin"] = Nutrient(
            nutrient_id="quercetin",
            name="Quercetin",
            category=NutrientCategory.PHYTONUTRIENTS,
            subcategory="Flavonoids (Flavonols)",
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("250.0"),
                    upper_limit=Decimal("1000.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Typical dietary intake 10-100mg/day; therapeutic range higher"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("250.0"),
                    upper_limit=Decimal("1000.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("30.0"),
            absorption_enhancers=["Bromelain (pineapple enzyme)", "Vitamin C", "Piperine"],
            absorption_inhibitors=["High protein meals", "Aluminum"],
            synergistic_nutrients=["Resveratrol", "Catechins", "Vitamin C", "Zinc"],
            antagonistic_nutrients=["Certain antibiotics (fluoroquinolones - may chelate)"],
            primary_functions=[
                "Potent anti-inflammatory (inhibits histamine release)",
                "Antioxidant (scavenges free radicals)",
                "Antihistamine and anti-allergic effects",
                "Cardiovascular protection (reduces blood pressure)",
                "Immune modulation",
                "Senolytic effects (removes senescent cells)",
                "Antiviral properties (inhibits viral replication)",
                "Neuroprotective (reduces neuroinflammation)"
            ],
            deficiency_symptoms=[
                "No classical deficiency",
                "Increased inflammation",
                "Higher allergy susceptibility",
                "Reduced antioxidant capacity"
            ],
            toxicity_symptoms=[
                "Generally safe up to 1g/day",
                "Headache at very high doses",
                "Tingling sensation (rare)",
                "May affect thyroid function at extreme doses",
                "Kidney toxicity only at very high IV doses"
            ],
            food_sources=[
                ("Capers", Decimal("234.0")),
                ("Red onions (raw)", Decimal("32.0")),
                ("Kale", Decimal("23.0")),
                ("Cherry tomatoes", Decimal("12.0")),
                ("Apples (with skin)", Decimal("4.4")),
                ("Cranberries", Decimal("15.0")),
                ("Blueberries", Decimal("7.7")),
                ("Red grapes", Decimal("3.5")),
                ("Green tea (brewed)", Decimal("2.5")),
                ("Broccoli", Decimal("3.2"))
            ],
            heat_stable=True,
            water_soluble=False,
            light_sensitive=False,
            oxygen_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("20.0"),
                CookingMethod.STEAMING: Decimal("10.0"),
                CookingMethod.BAKING: Decimal("15.0"),
                CookingMethod.GRILLING: Decimal("18.0"),
                CookingMethod.FRYING: Decimal("25.0")
            },
            research_references=["PMID: 27187333", "PMID: 31905180", "PMID: 27259333", "PMID: 32408699"]
        )
        
        # Curcumin - Active compound in turmeric with potent anti-inflammatory effects
        self.nutrients["curcumin"] = Nutrient(
            nutrient_id="curcumin",
            name="Curcumin",
            category=NutrientCategory.PHYTONUTRIENTS,
            subcategory="Polyphenols (Curcuminoids)",
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("500.0"),
                    upper_limit=Decimal("8000.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Therapeutic dosing often 500-2000mg/day with bioavailability enhancers"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("500.0"),
                    upper_limit=Decimal("8000.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("5.0"),
            absorption_enhancers=["Piperine (increases 2000%)", "Healthy fats", "Quercetin", "Lecithin"],
            absorption_inhibitors=["Rapid hepatic metabolism", "Low fat meals"],
            synergistic_nutrients=["Resveratrol", "Quercetin", "Omega-3 fatty acids", "Vitamin D"],
            antagonistic_nutrients=["Iron (high doses may chelate)"],
            primary_functions=[
                "Potent anti-inflammatory (inhibits NF-κB, COX-2)",
                "Antioxidant (multiple mechanisms)",
                "Neuroprotection (reduces amyloid plaques in Alzheimer's)",
                "Cancer prevention and treatment support",
                "Joint health (osteoarthritis relief)",
                "Cardiovascular protection",
                "Liver protection and detoxification",
                "Gut health and anti-inflammatory bowel disease",
                "Depression and mood support"
            ],
            deficiency_symptoms=[
                "No classical deficiency",
                "Increased systemic inflammation",
                "Reduced antioxidant protection",
                "Higher oxidative stress"
            ],
            toxicity_symptoms=[
                "Very safe even at high doses",
                "Mild GI upset at >8g/day",
                "Diarrhea or nausea (rare)",
                "May increase bleeding risk at very high doses",
                "Avoid with gallstones (stimulates bile production)"
            ],
            food_sources=[
                ("Turmeric powder", Decimal("3140.0")),
                ("Fresh turmeric root", Decimal("200.0")),
                ("Curry powder", Decimal("100.0")),
                ("Mango ginger", Decimal("35.0")),
                ("Curcumin extract (supplement)", Decimal("95000.0"))
            ],
            heat_stable=True,
            water_soluble=False,
            light_sensitive=True,
            oxygen_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("25.0"),
                CookingMethod.STEAMING: Decimal("12.0"),
                CookingMethod.BAKING: Decimal("15.0"),
                CookingMethod.GRILLING: Decimal("20.0"),
                CookingMethod.FRYING: Decimal("18.0")
            },
            research_references=["PMID: 17569207", "PMID: 23339049", "PMID: 27211847", "PMID: 29065496"]
        )
        
        # Lycopene - Carotenoid with prostate health and cardiovascular benefits
        self.nutrients["lycopene"] = Nutrient(
            nutrient_id="lycopene",
            name="Lycopene",
            category=NutrientCategory.PHYTONUTRIENTS,
            subcategory="Carotenoids",
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("10.0"),
                    upper_limit=Decimal("75.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Especially important for prostate health; typical intake 5-20mg/day"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("10.0"),
                    upper_limit=Decimal("75.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("30.0"),
            absorption_enhancers=["Dietary fats/oils", "Heat processing (cooking)", "Olive oil"],
            absorption_inhibitors=["Plant fibers", "Raw consumption (vs cooked)"],
            synergistic_nutrients=["Vitamin E", "Vitamin C", "Other carotenoids", "Healthy fats"],
            antagonistic_nutrients=["Beta-carotene (competitive absorption at high doses)"],
            primary_functions=[
                "Prostate cancer prevention (strongest evidence)",
                "Cardiovascular protection (reduces LDL oxidation)",
                "Potent antioxidant (singlet oxygen quencher)",
                "Skin protection from UV damage",
                "Bone health support",
                "Male fertility (reduces oxidative damage to sperm)",
                "Anti-inflammatory effects",
                "Neuroprotection"
            ],
            deficiency_symptoms=[
                "No classical deficiency",
                "Increased prostate cancer risk",
                "Higher cardiovascular disease risk",
                "Increased UV skin damage",
                "Reduced antioxidant capacity"
            ],
            toxicity_symptoms=[
                "Very safe, no established toxicity",
                "Lycopenodermia (orange skin) at extremely high intake (reversible)",
                "No serious adverse effects reported"
            ],
            food_sources=[
                ("Tomato paste (concentrated)", Decimal("55.5")),
                ("Sun-dried tomatoes", Decimal("45.9")),
                ("Tomato sauce", Decimal("17.2")),
                ("Watermelon", Decimal("4.5")),
                ("Pink grapefruit", Decimal("3.4")),
                ("Cooked tomatoes", Decimal("5.2")),
                ("Papaya", Decimal("1.8")),
                ("Red bell pepper", Decimal("0.5")),
                ("Fresh tomatoes (raw)", Decimal("3.0")),
                ("Guava (pink)", Decimal("5.2"))
            ],
            heat_stable=True,
            water_soluble=False,
            light_sensitive=True,
            oxygen_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("-10.0"),
                CookingMethod.STEAMING: Decimal("-5.0"),
                CookingMethod.BAKING: Decimal("-8.0"),
                CookingMethod.GRILLING: Decimal("5.0"),
                CookingMethod.FRYING: Decimal("-15.0")
            },
            research_references=["PMID: 15735074", "PMID: 17425952", "PMID: 23595206", "PMID: 28222809"]
        )
        
        # Sulforaphane - Isothiocyanate from cruciferous vegetables with anti-cancer properties
        self.nutrients["sulforaphane"] = Nutrient(
            nutrient_id="sulforaphane",
            name="Sulforaphane",
            category=NutrientCategory.PHYTONUTRIENTS,
            subcategory="Glucosinolates (Isothiocyanates)",
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("30.0"),
                    upper_limit=Decimal("200.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Formed from glucoraphanin by myrosinase enzyme; optimal intake not established"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("30.0"),
                    upper_limit=Decimal("200.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("70.0"),
            absorption_enhancers=["Myrosinase enzyme (raw vegetables, mustard seed)", "Gut bacteria"],
            absorption_inhibitors=["High heat cooking (destroys myrosinase)", "Boiling"],
            synergistic_nutrients=["Vitamin C", "Selenium", "Green tea catechins"],
            antagonistic_nutrients=["Goitrogens at very high doses (thyroid suppression)"],
            primary_functions=[
                "Activates Nrf2 pathway (master antioxidant regulator)",
                "Phase II detoxification enzyme induction",
                "Cancer prevention (multiple mechanisms)",
                "Anti-inflammatory effects",
                "Neuroprotection (autism spectrum, Alzheimer's)",
                "Cardiovascular protection",
                "Antimicrobial properties (H. pylori)",
                "Blood sugar regulation",
                "Autism symptoms improvement in studies"
            ],
            deficiency_symptoms=[
                "No classical deficiency",
                "Reduced detoxification capacity",
                "Increased cancer risk",
                "Lower antioxidant enzyme expression"
            ],
            toxicity_symptoms=[
                "Very safe at dietary levels",
                "Possible thyroid suppression at extremely high doses (hypothyroidism)",
                "GI discomfort if consuming large amounts of raw cruciferous",
                "No serious toxicity reported"
            ],
            food_sources=[
                ("Broccoli sprouts (3-day old)", Decimal("250.0")),
                ("Broccoli (raw)", Decimal("12.0")),
                ("Brussels sprouts", Decimal("10.0")),
                ("Cauliflower (raw)", Decimal("8.0")),
                ("Kale", Decimal("6.0")),
                ("Cabbage (raw)", Decimal("5.5")),
                ("Bok choy", Decimal("4.0")),
                ("Arugula", Decimal("7.0")),
                ("Watercress", Decimal("9.0")),
                ("Mustard greens", Decimal("8.0"))
            ],
            heat_stable=False,
            water_soluble=True,
            light_sensitive=False,
            oxygen_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("90.0"),
                CookingMethod.STEAMING: Decimal("30.0"),
                CookingMethod.BAKING: Decimal("40.0"),
                CookingMethod.GRILLING: Decimal("45.0"),
                CookingMethod.FRYING: Decimal("50.0")
            },
            research_references=["PMID: 18950480", "PMID: 21129940", "PMID: 25324105", "PMID: 29065496"]
        )
        
        # EGCG - Epigallocatechin gallate from green tea
        self.nutrients["egcg"] = Nutrient(
            nutrient_id="egcg",
            name="EGCG (Epigallocatechin Gallate)",
            category=NutrientCategory.PHYTONUTRIENTS,
            subcategory="Polyphenols (Catechins)",
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("200.0"),
                    upper_limit=Decimal("800.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Most abundant catechin in green tea; 3-4 cups green tea provides ~200-300mg"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("200.0"),
                    upper_limit=Decimal("800.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("13.0"),
            absorption_enhancers=["Vitamin C (prevents oxidation)", "Piperine", "Quercetin"],
            absorption_inhibitors=["Iron (forms complexes)", "Milk proteins", "High pH"],
            synergistic_nutrients=["Resveratrol", "Quercetin", "Vitamin C", "Caffeine"],
            antagonistic_nutrients=["Iron supplements (reduce iron absorption)"],
            primary_functions=[
                "Potent antioxidant (most powerful green tea catechin)",
                "Cancer prevention (multiple pathways)",
                "Weight loss and fat oxidation",
                "Cardiovascular protection",
                "Neuroprotection (Alzheimer's, Parkinson's)",
                "Anti-inflammatory effects",
                "Insulin sensitivity improvement",
                "Antimicrobial and antiviral properties",
                "Liver protection"
            ],
            deficiency_symptoms=[
                "No classical deficiency",
                "Reduced metabolic rate",
                "Lower antioxidant capacity",
                "Increased inflammation"
            ],
            toxicity_symptoms=[
                "Generally safe up to 800mg/day with food",
                "Liver toxicity reported with high-dose extracts on empty stomach",
                "Nausea if consumed without food",
                "Insomnia (due to caffeine content in tea)",
                "Iron deficiency with excessive intake"
            ],
            food_sources=[
                ("Green tea (brewed)", Decimal("60.0")),
                ("Matcha green tea powder", Decimal("130.0")),
                ("White tea", Decimal("28.0")),
                ("Oolong tea", Decimal("36.0")),
                ("Black tea", Decimal("9.0")),
                ("Green tea extract (supplement)", Decimal("500.0"))
            ],
            heat_stable=False,
            water_soluble=True,
            light_sensitive=True,
            oxygen_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("50.0"),
                CookingMethod.STEAMING: Decimal("30.0"),
                CookingMethod.BAKING: Decimal("40.0"),
                CookingMethod.GRILLING: Decimal("45.0"),
                CookingMethod.FRYING: Decimal("60.0")
            },
            research_references=["PMID: 16870699", "PMID: 20370896", "PMID: 22052385", "PMID: 28899506"]
        )
        
        # Lutein - Carotenoid critical for eye health
        self.nutrients["lutein"] = Nutrient(
            nutrient_id="lutein",
            name="Lutein",
            category=NutrientCategory.PHYTONUTRIENTS,
            subcategory="Carotenoids (Xanthophylls)",
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("6.0"),
                    upper_limit=Decimal("20.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Critical for macular health; often combined with zeaxanthin"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("6.0"),
                    upper_limit=Decimal("20.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
                (LifeStage.ELDERLY, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("10.0"),
                    upper_limit=Decimal("20.0"),
                    life_stage=LifeStage.ELDERLY,
                    biological_sex=BiologicalSex.MALE,
                    notes="Higher needs for age-related macular degeneration prevention"
                ),
            },
            bioavailability_percentage=Decimal("45.0"),
            absorption_enhancers=["Dietary fats", "Vitamin E", "Cooked vegetables (vs raw)"],
            absorption_inhibitors=["Plant sterols", "Low-fat meals", "Olestra"],
            synergistic_nutrients=["Zeaxanthin", "Vitamin E", "Vitamin C", "Zinc", "DHA"],
            antagonistic_nutrients=["Beta-carotene at high doses (competitive absorption)"],
            primary_functions=[
                "Macular pigment (protects retina from blue light)",
                "Age-related macular degeneration (AMD) prevention",
                "Cataract prevention",
                "Antioxidant (especially in eyes)",
                "Skin health and UV protection",
                "Cognitive function support",
                "Anti-inflammatory effects",
                "Cardiovascular health"
            ],
            deficiency_symptoms=[
                "Increased AMD risk",
                "Cataract development",
                "Reduced visual acuity",
                "Increased photosensitivity",
                "Cognitive decline"
            ],
            toxicity_symptoms=[
                "Extremely safe",
                "No toxicity reported even at high doses",
                "Possible carotenodermia (yellowish skin) at very high intake (harmless)"
            ],
            food_sources=[
                ("Kale (cooked)", Decimal("23.8")),
                ("Spinach (cooked)", Decimal("20.4")),
                ("Collard greens", Decimal("18.5")),
                ("Turnip greens", Decimal("12.2")),
                ("Swiss chard", Decimal("11.0")),
                ("Egg yolks", Decimal("0.5")),
                ("Corn", Decimal("0.6")),
                ("Orange bell pepper", Decimal("0.2")),
                ("Zucchini", Decimal("2.1")),
                ("Avocado", Decimal("0.27"))
            ],
            heat_stable=True,
            water_soluble=False,
            light_sensitive=True,
            oxygen_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("15.0"),
                CookingMethod.STEAMING: Decimal("8.0"),
                CookingMethod.BAKING: Decimal("10.0"),
                CookingMethod.GRILLING: Decimal("12.0"),
                CookingMethod.FRYING: Decimal("20.0")
            },
            research_references=["PMID: 16988130", "PMID: 23571649", "PMID: 25387916", "PMID: 27652134"]
        )
        
        # Anthocyanins - Flavonoids giving berries their color
        self.nutrients["anthocyanins"] = Nutrient(
            nutrient_id="anthocyanins",
            name="Anthocyanins",
            category=NutrientCategory.PHYTONUTRIENTS,
            subcategory="Flavonoids (Anthocyanidins)",
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("50.0"),
                    upper_limit=Decimal("500.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Average intake varies widely; higher intake associated with health benefits"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("50.0"),
                    upper_limit=Decimal("500.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("12.0"),
            absorption_enhancers=["Organic acids", "Vitamin C", "Gut microbiota"],
            absorption_inhibitors=["High pH (alkaline conditions)", "Heat processing"],
            synergistic_nutrients=["Vitamin C", "Quercetin", "Other polyphenols"],
            antagonistic_nutrients=["None significant"],
            primary_functions=[
                "Cardiovascular protection (reduces heart disease risk)",
                "Cognitive function and memory enhancement",
                "Anti-inflammatory effects",
                "Potent antioxidant",
                "Blood sugar regulation",
                "Cancer prevention",
                "Eye health (improves night vision)",
                "Anti-aging effects",
                "Urinary tract health (prevents bacterial adhesion)"
            ],
            deficiency_symptoms=[
                "No classical deficiency",
                "Reduced cardiovascular protection",
                "Increased inflammation",
                "Cognitive decline acceleration"
            ],
            toxicity_symptoms=[
                "Extremely safe",
                "No toxicity reported at dietary levels",
                "Very well tolerated"
            ],
            food_sources=[
                ("Black chokeberry", Decimal("1480.0")),
                ("Elderberries", Decimal("749.0")),
                ("Black currants", Decimal("476.0")),
                ("Blueberries", Decimal("163.0")),
                ("Red cabbage", Decimal("196.0")),
                ("Blackberries", Decimal("245.0")),
                ("Raspberries (red)", Decimal("92.0")),
                ("Strawberries", Decimal("35.0")),
                ("Red grapes", Decimal("27.0")),
                ("Purple sweet potato", Decimal("20.0"))
            ],
            heat_stable=False,
            water_soluble=True,
            light_sensitive=True,
            oxygen_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("45.0"),
                CookingMethod.STEAMING: Decimal("20.0"),
                CookingMethod.BAKING: Decimal("30.0"),
                CookingMethod.GRILLING: Decimal("35.0"),
                CookingMethod.FRYING: Decimal("40.0")
            },
            research_references=["PMID: 20370896", "PMID: 24915350", "PMID: 25694676", "PMID: 27509521"]
        )
    
    def _add_antioxidants(self):
        """Add antioxidant definitions"""
        
        # Coenzyme Q10 (CoQ10) - Essential for mitochondrial energy production
        self.nutrients["coq10"] = Nutrient(
            nutrient_id="coq10",
            name="Coenzyme Q10 (CoQ10)",
            category=NutrientCategory.ANTIOXIDANTS,
            subcategory="Ubiquinone/Ubiquinol",
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("100.0"),
                    upper_limit=Decimal("500.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Body produces but declines with age; supplementation beneficial"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("100.0"),
                    upper_limit=Decimal("500.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
                (LifeStage.ELDERLY, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("200.0"),
                    upper_limit=Decimal("600.0"),
                    life_stage=LifeStage.ELDERLY,
                    biological_sex=BiologicalSex.MALE,
                    notes="Higher needs due to reduced endogenous production"
                ),
            },
            bioavailability_percentage=Decimal("25.0"),
            absorption_enhancers=["Dietary fats", "Ubiquinol form (reduced)", "Piperine"],
            absorption_inhibitors=["Statin drugs (deplete CoQ10)", "Low-fat meals"],
            synergistic_nutrients=["Vitamin E", "L-carnitine", "Alpha-lipoic acid", "PQQ"],
            antagonistic_nutrients=["Statins (block endogenous synthesis)", "Beta-blockers"],
            primary_functions=[
                "Mitochondrial ATP production (electron transport chain)",
                "Powerful antioxidant (regenerates vitamin E)",
                "Cardiovascular health (heart failure treatment)",
                "Statin-induced myopathy prevention",
                "Blood pressure reduction",
                "Male fertility (sperm motility)",
                "Migraine prevention",
                "Exercise performance enhancement",
                "Neuroprotection (Parkinson's disease)"
            ],
            deficiency_symptoms=[
                "Fatigue and muscle weakness",
                "Heart failure",
                "Muscle pain (especially with statin use)",
                "Cognitive impairment",
                "Migraines",
                "Reduced exercise tolerance",
                "Accelerated aging"
            ],
            toxicity_symptoms=[
                "Very safe even at high doses (1200mg/day studied)",
                "Mild GI upset (rare)",
                "Insomnia if taken late in day",
                "May reduce blood clotting (caution with warfarin)"
            ],
            food_sources=[
                ("Organ meats (heart, liver)", Decimal("11.0")),
                ("Beef", Decimal("3.1")),
                ("Herring", Decimal("2.7")),
                ("Rainbow trout", Decimal("1.1")),
                ("Chicken", Decimal("1.4")),
                ("Pork", Decimal("2.4")),
                ("Sesame seeds", Decimal("1.8")),
                ("Pistachios", Decimal("2.1")),
                ("Broccoli", Decimal("0.5")),
                ("Cauliflower", Decimal("0.4"))
            ],
            heat_stable=True,
            water_soluble=False,
            light_sensitive=True,
            oxygen_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("20.0"),
                CookingMethod.STEAMING: Decimal("10.0"),
                CookingMethod.BAKING: Decimal("15.0"),
                CookingMethod.GRILLING: Decimal("25.0"),
                CookingMethod.FRYING: Decimal("30.0")
            },
            research_references=["PMID: 17287847", "PMID: 22530606", "PMID: 25126965", "PMID: 28914794"]
        )
        
        # Glutathione - Master antioxidant
        self.nutrients["glutathione"] = Nutrient(
            nutrient_id="glutathione",
            name="Glutathione (GSH)",
            category=NutrientCategory.ANTIOXIDANTS,
            subcategory="Tripeptide (Cysteine-Glycine-Glutamate)",
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("250.0"),
                    upper_limit=Decimal("1000.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Synthesized from amino acids; oral absorption limited"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("250.0"),
                    upper_limit=Decimal("1000.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("10.0"),
            absorption_enhancers=["Liposomal form", "NAC (precursor)", "Whey protein", "Selenium"],
            absorption_inhibitors=["GI degradation", "Alcohol (depletes)"],
            synergistic_nutrients=["NAC", "Selenium", "Vitamin C", "Vitamin E", "Alpha-lipoic acid"],
            antagonistic_nutrients=["Alcohol", "Acetaminophen (depletes)", "Heavy metals"],
            primary_functions=[
                "Master antioxidant (regenerates vitamins C and E)",
                "Phase II liver detoxification",
                "Heavy metal chelation",
                "Immune system support",
                "DNA synthesis and repair",
                "Protein and prostaglandin synthesis",
                "Amino acid transport",
                "Enzyme activation",
                "Protection against oxidative stress"
            ],
            deficiency_symptoms=[
                "Increased oxidative stress",
                "Impaired detoxification",
                "Compromised immune function",
                "Chronic fatigue",
                "Accelerated aging",
                "Increased disease susceptibility",
                "Neurological dysfunction"
            ],
            toxicity_symptoms=[
                "Very safe even at high doses",
                "Possible zinc depletion at extremely high doses",
                "Rare allergic reactions"
            ],
            food_sources=[
                ("Asparagus (raw)", Decimal("28.3")),
                ("Avocado", Decimal("27.7")),
                ("Spinach (raw)", Decimal("11.4")),
                ("Okra", Decimal("13.0")),
                ("Broccoli (raw)", Decimal("8.0")),
                ("Tomatoes", Decimal("8.0")),
                ("Grapefruit", Decimal("15.0")),
                ("Watermelon", Decimal("7.0")),
                ("Fresh meats (raw)", Decimal("15.0")),
                ("Walnuts", Decimal("4.0"))
            ],
            heat_stable=False,
            water_soluble=True,
            light_sensitive=False,
            oxygen_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("60.0"),
                CookingMethod.STEAMING: Decimal("30.0"),
                CookingMethod.BAKING: Decimal("40.0"),
                CookingMethod.GRILLING: Decimal("50.0"),
                CookingMethod.FRYING: Decimal("55.0")
            },
            research_references=["PMID: 21753067", "PMID: 24791752", "PMID: 28441057", "PMID: 30388871"]
        )
        
        # Alpha-Lipoic Acid - Universal antioxidant
        self.nutrients["alpha_lipoic_acid"] = Nutrient(
            nutrient_id="alpha_lipoic_acid",
            name="Alpha-Lipoic Acid (ALA)",
            category=NutrientCategory.ANTIOXIDANTS,
            subcategory="Organosulfur Compound",
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("300.0"),
                    upper_limit=Decimal("1800.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Synthesized in body but supplementation beneficial for various conditions"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("300.0"),
                    upper_limit=Decimal("1800.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("30.0"),
            absorption_enhancers=["Empty stomach", "R-lipoic acid form (more bioavailable)"],
            absorption_inhibitors=["Food (reduces absorption by 30%)", "High fiber"],
            synergistic_nutrients=["Vitamin C", "Vitamin E", "CoQ10", "Glutathione", "Carnitine"],
            antagonistic_nutrients=["Biotin (competes for absorption at very high ALA doses)"],
            primary_functions=[
                "Universal antioxidant (both water and fat soluble)",
                "Regenerates vitamins C, E, CoQ10, glutathione",
                "Blood sugar regulation (improves insulin sensitivity)",
                "Diabetic neuropathy treatment",
                "Heavy metal chelation",
                "Mitochondrial energy production cofactor",
                "Cognitive function and neuroprotection",
                "Anti-aging effects",
                "Liver health and detoxification"
            ],
            deficiency_symptoms=[
                "No classical deficiency (body synthesizes)",
                "Reduced antioxidant capacity",
                "Impaired glucose metabolism",
                "Accelerated aging"
            ],
            toxicity_symptoms=[
                "Generally safe up to 1800mg/day",
                "Skin rash (rare)",
                "Nausea or GI upset at high doses",
                "Hypoglycemia if diabetic (monitor blood sugar)",
                "Possible thiamine deficiency at extremely high chronic doses"
            ],
            food_sources=[
                ("Organ meats (heart, kidney, liver)", Decimal("5.0")),
                ("Spinach", Decimal("3.2")),
                ("Broccoli", Decimal("0.9")),
                ("Tomatoes", Decimal("0.6")),
                ("Brussels sprouts", Decimal("1.2")),
                ("Peas", Decimal("1.0")),
                ("Red meat", Decimal("2.5")),
                ("Rice bran", Decimal("2.1")),
                ("Brewer's yeast", Decimal("1.5")),
                ("Potatoes", Decimal("0.6"))
            ],
            heat_stable=True,
            water_soluble=True,
            light_sensitive=True,
            oxygen_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("25.0"),
                CookingMethod.STEAMING: Decimal("15.0"),
                CookingMethod.BAKING: Decimal("20.0"),
                CookingMethod.GRILLING: Decimal("30.0"),
                CookingMethod.FRYING: Decimal("35.0")
            },
            research_references=["PMID: 18287346", "PMID: 21569436", "PMID: 25389631", "PMID: 29270170"]
        )
        
        # Astaxanthin - Super antioxidant from algae
        self.nutrients["astaxanthin"] = Nutrient(
            nutrient_id="astaxanthin",
            name="Astaxanthin",
            category=NutrientCategory.ANTIOXIDANTS,
            subcategory="Carotenoids (Xanthophylls)",
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("4.0"),
                    upper_limit=Decimal("12.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Potent antioxidant; 4-8mg/day typical supplementation"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("4.0"),
                    upper_limit=Decimal("12.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("40.0"),
            absorption_enhancers=["Dietary fats", "Phospholipid complex form"],
            absorption_inhibitors=["Low-fat meals", "Plant sterols"],
            synergistic_nutrients=["Vitamin E", "Vitamin C", "Other carotenoids", "Omega-3 fatty acids"],
            antagonistic_nutrients=["Beta-carotene (may compete at high doses)"],
            primary_functions=[
                "Extremely potent antioxidant (6000x stronger than vitamin C)",
                "Crosses blood-brain barrier",
                "Eye health (reduces eye fatigue, improves visual acuity)",
                "Skin protection from UV damage",
                "Anti-inflammatory effects",
                "Cardiovascular protection",
                "Exercise recovery and performance",
                "Immune system support",
                "Neuroprotection",
                "Male fertility improvement"
            ],
            deficiency_symptoms=[
                "No classical deficiency",
                "Increased oxidative stress",
                "Eye strain and fatigue",
                "Reduced skin protection from UV"
            ],
            toxicity_symptoms=[
                "Extremely safe",
                "No toxicity reported even at high doses",
                "Possible orange-tinted skin at very high intake (harmless)",
                "Rare GI discomfort"
            ],
            food_sources=[
                ("Wild Pacific salmon", Decimal("4.5")),
                ("Sockeye salmon", Decimal("4.0")),
                ("Red trout", Decimal("3.5")),
                ("Krill", Decimal("1.5")),
                ("Shrimp", Decimal("1.2")),
                ("Lobster", Decimal("0.8")),
                ("Red crab", Decimal("1.1")),
                ("Salmon roe", Decimal("2.9")),
                ("Haematococcus algae (supplement)", Decimal("50.0")),
                ("Arctic char", Decimal("1.0"))
            ],
            heat_stable=True,
            water_soluble=False,
            light_sensitive=True,
            oxygen_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("10.0"),
                CookingMethod.STEAMING: Decimal("5.0"),
                CookingMethod.BAKING: Decimal("8.0"),
                CookingMethod.GRILLING: Decimal("15.0"),
                CookingMethod.FRYING: Decimal("20.0")
            },
            research_references=["PMID: 21488893", "PMID: 23484470", "PMID: 26018743", "PMID: 29099763"]
        )
        
        # N-Acetyl Cysteine (NAC) - Glutathione precursor
        self.nutrients["nac"] = Nutrient(
            nutrient_id="nac",
            name="N-Acetyl Cysteine (NAC)",
            category=NutrientCategory.ANTIOXIDANTS,
            subcategory="Amino Acid Derivative",
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("600.0"),
                    upper_limit=Decimal("1800.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="More bioavailable than cysteine; powerful glutathione precursor"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("600.0"),
                    upper_limit=Decimal("1800.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("90.0"),
            absorption_enhancers=["Vitamin C", "Selenium", "Empty stomach"],
            absorption_inhibitors=["Food (reduces rate but not extent)"],
            synergistic_nutrients=["Selenium", "Glycine", "Glutamine", "Vitamin C", "Vitamin E"],
            antagonistic_nutrients=["Nitroglycerin (interaction)", "Activated charcoal"],
            primary_functions=[
                "Glutathione synthesis precursor (rate-limiting amino acid)",
                "Mucolytic (breaks down mucus in respiratory conditions)",
                "Acetaminophen overdose antidote",
                "Liver protection and detoxification",
                "Heavy metal chelation",
                "Antioxidant and anti-inflammatory",
                "Mental health support (OCD, addiction, depression)",
                "Fertility enhancement (both male and female)",
                "PCOS treatment support"
            ],
            deficiency_symptoms=[
                "No classical deficiency (derived from cysteine)",
                "Reduced glutathione production",
                "Impaired detoxification",
                "Increased oxidative stress"
            ],
            toxicity_symptoms=[
                "Generally very safe",
                "Nausea or vomiting at high doses (>2400mg/day)",
                "Diarrhea",
                "Skin rash (rare)",
                "May increase homocysteine if B-vitamins inadequate",
                "Avoid during pregnancy (theoretical concerns)"
            ],
            food_sources=[
                ("Not naturally occurring in foods", Decimal("0.0")),
                ("Derived from L-cysteine in foods", Decimal("0.0")),
                ("High-protein foods (contain cysteine)", Decimal("0.0")),
                ("Chicken", Decimal("0.0")),
                ("Turkey", Decimal("0.0")),
                ("Eggs", Decimal("0.0")),
                ("Sunflower seeds", Decimal("0.0")),
                ("Legumes", Decimal("0.0"))
            ],
            heat_stable=True,
            water_soluble=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("0.0"),
                CookingMethod.STEAMING: Decimal("0.0"),
                CookingMethod.BAKING: Decimal("0.0"),
                CookingMethod.GRILLING: Decimal("0.0"),
                CookingMethod.FRYING: Decimal("0.0")
            },
            research_references=["PMID: 19818624", "PMID: 23642096", "PMID: 25803183", "PMID: 30275898"]
        )
        
        # PQQ - Pyrroloquinoline Quinone - Novel antioxidant and mitochondrial biogenesis promoter
        self.nutrients["pqq"] = Nutrient(
            nutrient_id="pqq",
            name="Pyrroloquinoline Quinone (PQQ)",
            category=NutrientCategory.ANTIOXIDANTS,
            subcategory="Quinone Cofactor",
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("10.0"),
                    upper_limit=Decimal("20.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Emerging nutrient; promotes mitochondrial biogenesis"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("10.0"),
                    upper_limit=Decimal("20.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("62.0"),
            absorption_enhancers=["CoQ10 (synergistic)", "Healthy fats"],
            absorption_inhibitors=["Not well established"],
            synergistic_nutrients=["CoQ10", "Alpha-lipoic acid", "Carnitine", "B-vitamins"],
            antagonistic_nutrients=["None identified"],
            primary_functions=[
                "Mitochondrial biogenesis (creates new mitochondria)",
                "Powerful antioxidant (5000x more efficient than vitamin C)",
                "Neuroprotection (nerve growth factor stimulation)",
                "Cognitive enhancement (memory and attention)",
                "Cardiovascular health",
                "Energy production support",
                "Anti-aging effects",
                "Sleep quality improvement",
                "Inflammation reduction"
            ],
            deficiency_symptoms=[
                "No established deficiency syndrome",
                "Reduced mitochondrial function",
                "Cognitive decline",
                "Fatigue"
            ],
            toxicity_symptoms=[
                "Very safe in studies up to 60mg/day",
                "Mild headache (rare)",
                "Insomnia if taken late in day",
                "GI discomfort at very high doses"
            ],
            food_sources=[
                ("Natto (fermented soybeans)", Decimal("0.061")),
                ("Green tea", Decimal("0.029")),
                ("Tofu", Decimal("0.024")),
                ("Green peppers", Decimal("0.028")),
                ("Parsley", Decimal("0.034")),
                ("Kiwi fruit", Decimal("0.027")),
                ("Papaya", Decimal("0.026")),
                ("Spinach", Decimal("0.021")),
                ("Fava beans", Decimal("0.019")),
                ("Cocoa powder", Decimal("0.016"))
            ],
            heat_stable=True,
            water_soluble=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("15.0"),
                CookingMethod.STEAMING: Decimal("8.0"),
                CookingMethod.BAKING: Decimal("10.0"),
                CookingMethod.GRILLING: Decimal("12.0"),
                CookingMethod.FRYING: Decimal("18.0")
            },
            research_references=["PMID: 20804815", "PMID: 23571649", "PMID: 24231099", "PMID: 27450321"]
        )
    
    def _add_probiotics_prebiotics(self):
        """Add probiotic and prebiotic definitions"""
        
        # Lactobacillus acidophilus - Most common probiotic
        self.nutrients["lactobacillus_acidophilus"] = Nutrient(
            nutrient_id="lactobacillus_acidophilus",
            name="Lactobacillus acidophilus",
            category=NutrientCategory.PROBIOTICS,
            subcategory="Lactic Acid Bacteria",
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("1000.0"),
                    upper_limit=Decimal("10000.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Measured in CFU (colony forming units); 1-10 billion CFU/day typical"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("1000.0"),
                    upper_limit=Decimal("10000.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("30.0"),
            absorption_enhancers=["Prebiotics (FOS, inulin)", "Resistant starch", "Low stomach acid"],
            absorption_inhibitors=["Antibiotics", "Chlorinated water", "High heat", "Stomach acid"],
            synergistic_nutrients=["Prebiotics", "Other probiotic strains", "B-vitamins", "Zinc"],
            antagonistic_nutrients=["Antibiotics", "Antifungals", "Chlorine"],
            primary_functions=[
                "Digestive health (lactose digestion, IBS relief)",
                "Immune system modulation",
                "Vaginal health (prevents yeast infections)",
                "Cholesterol reduction",
                "Nutrient absorption enhancement",
                "Pathogen inhibition (competitive exclusion)",
                "Vitamin synthesis (B vitamins, vitamin K)",
                "Anti-inflammatory effects"
            ],
            deficiency_symptoms=[
                "Dysbiosis (gut imbalance)",
                "Digestive issues (bloating, gas, diarrhea)",
                "Weakened immunity",
                "Vaginal infections",
                "Increased inflammation",
                "Nutrient malabsorption"
            ],
            toxicity_symptoms=[
                "Very safe even at high doses",
                "Mild bloating during adjustment period",
                "Rare allergic reactions",
                "Possible immune stimulation in immunocompromised"
            ],
            food_sources=[
                ("Yogurt (live cultures)", Decimal("100.0")),
                ("Kefir", Decimal("300.0")),
                ("Sauerkraut (raw)", Decimal("50.0")),
                ("Kimchi", Decimal("100.0")),
                ("Miso", Decimal("150.0")),
                ("Tempeh", Decimal("80.0")),
                ("Probiotic supplements", Decimal("5000.0")),
                ("Acidophilus milk", Decimal("100.0"))
            ],
            heat_stable=False,
            water_soluble=False,
            light_sensitive=True,
            oxygen_sensitive=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("100.0"),
                CookingMethod.STEAMING: Decimal("90.0"),
                CookingMethod.BAKING: Decimal("100.0"),
                CookingMethod.GRILLING: Decimal("100.0"),
                CookingMethod.FRYING: Decimal("100.0")
            },
            research_references=["PMID: 23609775", "PMID: 24912386", "PMID: 26695080", "PMID: 28914794"]
        )
        
        # Bifidobacterium - Critical infant probiotic
        self.nutrients["bifidobacterium"] = Nutrient(
            nutrient_id="bifidobacterium",
            name="Bifidobacterium spp.",
            category=NutrientCategory.PROBIOTICS,
            subcategory="Bifidobacteria",
            unit=MeasurementUnit.MILLIGRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("1000.0"),
                    upper_limit=Decimal("10000.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Dominant in infant gut; declines with age"
                ),
                (LifeStage.INFANT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("2000.0"),
                    upper_limit=Decimal("10000.0"),
                    life_stage=LifeStage.INFANT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Critical for infant gut development"
                ),
            },
            bioavailability_percentage=Decimal("35.0"),
            absorption_enhancers=["Human milk oligosaccharides", "Prebiotics", "GOS"],
            absorption_inhibitors=["Antibiotics", "Formula feeding (vs breastfeeding)"],
            synergistic_nutrients=["Lactobacillus strains", "Prebiotics", "Iron"],
            antagonistic_nutrients=["Antibiotics", "Excessive hygiene"],
            primary_functions=[
                "Infant gut colonization (from breast milk)",
                "Immune system development",
                "IBS and IBD symptom reduction",
                "Constipation relief",
                "Allergy prevention in infants",
                "Mental health support (gut-brain axis)",
                "Vitamin production (B vitamins, vitamin K2)",
                "Anti-pathogenic activity"
            ],
            deficiency_symptoms=[
                "Digestive disorders",
                "Weakened immunity",
                "Allergies and eczema",
                "Constipation",
                "Mood disorders",
                "Increased inflammation"
            ],
            toxicity_symptoms=[
                "Very safe",
                "Temporary gas/bloating",
                "No serious adverse effects"
            ],
            food_sources=[
                ("Fermented milk products", Decimal("80.0")),
                ("Kefir", Decimal("200.0")),
                ("Yogurt with Bifido cultures", Decimal("100.0")),
                ("Breast milk", Decimal("50.0")),
                ("Aged cheese", Decimal("30.0")),
                ("Probiotic supplements", Decimal("5000.0"))
            ],
            heat_stable=False,
            water_soluble=False,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("100.0"),
                CookingMethod.STEAMING: Decimal("95.0"),
                CookingMethod.BAKING: Decimal("100.0"),
                CookingMethod.GRILLING: Decimal("100.0"),
                CookingMethod.FRYING: Decimal("100.0")
            },
            research_references=["PMID: 24912386", "PMID: 26695080", "PMID: 28473727", "PMID: 29430227"]
        )
        
        # Inulin - Prebiotic fiber
        self.nutrients["inulin"] = Nutrient(
            nutrient_id="inulin",
            name="Inulin",
            category=NutrientCategory.PREBIOTICS,
            subcategory="Fructan Fiber",
            unit=MeasurementUnit.GRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("5.0"),
                    upper_limit=Decimal("20.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Prebiotic fiber that feeds beneficial bacteria"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("5.0"),
                    upper_limit=Decimal("20.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("0.0"),
            absorption_enhancers=["Not absorbed - fermented by gut bacteria"],
            absorption_inhibitors=["Not applicable"],
            synergistic_nutrients=["Probiotics", "Other prebiotics", "Calcium"],
            antagonistic_nutrients=["Antibiotics (kill fermenting bacteria)"],
            primary_functions=[
                "Feeds beneficial gut bacteria (Bifidobacterium)",
                "Improves calcium and magnesium absorption",
                "Promotes regular bowel movements",
                "Blood sugar regulation",
                "Cholesterol reduction",
                "Satiety and weight management",
                "Short-chain fatty acid production (butyrate)",
                "Colon health"
            ],
            deficiency_symptoms=[
                "Dysbiosis",
                "Constipation",
                "Poor mineral absorption",
                "Increased colon cancer risk",
                "Weakened gut barrier"
            ],
            toxicity_symptoms=[
                "Gas and bloating at high doses (>20g/day)",
                "Diarrhea",
                "Abdominal discomfort",
                "FODMAP sensitivity reactions"
            ],
            food_sources=[
                ("Chicory root", Decimal("64.6")),
                ("Jerusalem artichoke", Decimal("31.5")),
                ("Dandelion greens", Decimal("24.3")),
                ("Garlic", Decimal("17.5")),
                ("Leeks", Decimal("11.7")),
                ("Onions", Decimal("8.6")),
                ("Asparagus", Decimal("2.5")),
                ("Bananas (slightly green)", Decimal("0.5")),
                ("Wheat bran", Decimal("2.5")),
                ("Barley", Decimal("3.6"))
            ],
            heat_stable=True,
            water_soluble=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("10.0"),
                CookingMethod.STEAMING: Decimal("5.0"),
                CookingMethod.BAKING: Decimal("8.0"),
                CookingMethod.GRILLING: Decimal("7.0"),
                CookingMethod.FRYING: Decimal("12.0")
            },
            research_references=["PMID: 25599185", "PMID: 26269366", "PMID: 27456347", "PMID: 28914794"]
        )
        
        # FOS - Fructooligosaccharides
        self.nutrients["fos"] = Nutrient(
            nutrient_id="fos",
            name="Fructooligosaccharides (FOS)",
            category=NutrientCategory.PREBIOTICS,
            subcategory="Short-Chain Fructans",
            unit=MeasurementUnit.GRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("3.0"),
                    upper_limit=Decimal("15.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Promotes Bifidobacterium growth"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("3.0"),
                    upper_limit=Decimal("15.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("0.0"),
            absorption_enhancers=["Fermented by beneficial bacteria"],
            absorption_inhibitors=["Not absorbed"],
            synergistic_nutrients=["Probiotics", "Inulin", "GOS"],
            antagonistic_nutrients=["Antibiotics"],
            primary_functions=[
                "Selectively feeds Bifidobacterium",
                "Improves calcium absorption",
                "Immune modulation",
                "Digestive health",
                "Cholesterol reduction",
                "Blood sugar stabilization",
                "Colon health support"
            ],
            deficiency_symptoms=[
                "Reduced beneficial bacteria",
                "Poor mineral absorption",
                "Digestive irregularity"
            ],
            toxicity_symptoms=[
                "Gas and bloating at high doses",
                "Diarrhea (>15g/day)",
                "FODMAP sensitivity"
            ],
            food_sources=[
                ("Onions", Decimal("2.8")),
                ("Garlic", Decimal("1.5")),
                ("Bananas", Decimal("0.5")),
                ("Asparagus", Decimal("0.4")),
                ("Tomatoes", Decimal("0.3")),
                ("Honey", Decimal("1.5")),
                ("Barley", Decimal("0.8")),
                ("Rye", Decimal("1.0"))
            ],
            heat_stable=True,
            water_soluble=True,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("8.0"),
                CookingMethod.STEAMING: Decimal("4.0"),
                CookingMethod.BAKING: Decimal("6.0"),
                CookingMethod.GRILLING: Decimal("5.0"),
                CookingMethod.FRYING: Decimal("10.0")
            },
            research_references=["PMID: 25599185", "PMID: 27456347", "PMID: 28473727"]
        )
        
        # Resistant Starch - Type 2 prebiotic
        self.nutrients["resistant_starch"] = Nutrient(
            nutrient_id="resistant_starch",
            name="Resistant Starch",
            category=NutrientCategory.PREBIOTICS,
            subcategory="Non-Digestible Starch",
            unit=MeasurementUnit.GRAM,
            rda_values={
                (LifeStage.ADULT, BiologicalSex.MALE): NutrientReference(
                    amount=Decimal("15.0"),
                    upper_limit=Decimal("40.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.MALE,
                    notes="Type 2 RS increases with cooling cooked starches"
                ),
                (LifeStage.ADULT, BiologicalSex.FEMALE): NutrientReference(
                    amount=Decimal("15.0"),
                    upper_limit=Decimal("40.0"),
                    life_stage=LifeStage.ADULT,
                    biological_sex=BiologicalSex.FEMALE
                ),
            },
            bioavailability_percentage=Decimal("0.0"),
            absorption_enhancers=["Resistant to digestion - fermented in colon"],
            absorption_inhibitors=["Heat breaks down type 3 RS"],
            synergistic_nutrients=["Probiotics", "Other prebiotics", "Butyrate"],
            antagonistic_nutrients=["None significant"],
            primary_functions=[
                "Butyrate production (primary fuel for colonocytes)",
                "Blood sugar control (low glycemic impact)",
                "Insulin sensitivity improvement",
                "Weight management (increased satiety)",
                "Colon cancer prevention",
                "Gut microbiome diversity",
                "Reduced inflammation",
                "Improved gut barrier function"
            ],
            deficiency_symptoms=[
                "Poor blood sugar control",
                "Reduced butyrate production",
                "Increased colon cancer risk",
                "Gut dysbiosis"
            ],
            toxicity_symptoms=[
                "Gas and bloating (start low and increase slowly)",
                "Temporary digestive discomfort",
                "Generally very safe"
            ],
            food_sources=[
                ("Raw potato starch", Decimal("75.0")),
                ("Green bananas", Decimal("8.5")),
                ("Cooked and cooled potatoes", Decimal("3.2")),
                ("Cooked and cooled rice", Decimal("1.2")),
                ("Cooked and cooled pasta", Decimal("1.1")),
                ("Legumes (cooked)", Decimal("3.4")),
                ("Oats (raw)", Decimal("3.6")),
                ("Cashews", Decimal("13.0")),
                ("Plantains (cooked)", Decimal("4.7")),
                ("Pearl barley", Decimal("3.2"))
            ],
            heat_stable=False,
            water_soluble=False,
            cooking_loss_rates={
                CookingMethod.BOILING: Decimal("-50.0"),
                CookingMethod.STEAMING: Decimal("-30.0"),
                CookingMethod.BAKING: Decimal("-40.0"),
                CookingMethod.GRILLING: Decimal("-35.0"),
                CookingMethod.FRYING: Decimal("-45.0")
            },
            research_references=["PMID: 23631844", "PMID: 26269366", "PMID: 28082216", "PMID: 29430227"]
        )


if __name__ == "__main__":
    db = NutrientDatabase()
    print(f"Nutrient Database initialized with {len(db.nutrients)} nutrients")
    print("Categories covered:")
    for category in NutrientCategory:
        count = len(db.get_nutrients_by_category(category))
        if count > 0:
            print(f"  - {category.value}: {count} nutrients")
