"""
Multi-Condition Recommendation Engine - The "Digital Dietitian"
================================================================

This module implements the intelligent recommendation engine that acts as a personalized
digital dietitian. It takes food's molecular fingerprint and filters it through the user's
complete health profile to provide optimized recommendations.

Core Optimization Problem:
Goal: Maximize "Brain Health" molecules + Minimize "Diabetes" molecules + Minimize "Hypertension" molecules

This is a multi-objective constraint satisfaction problem (CSP) solved using:
- Linear programming for nutrient optimization
- Pareto optimization for conflicting goals
- Disease-specific molecular filtering
- Age-based lifecycle adjustments

Example Decision Logic:
    Food Scan: High C-H bonds (sugar) + High sodium
    User Profile: Age 55, Goal: Brain Health, Diseases: Type 2 Diabetes + Hypertension
    
    AI Analysis:
    - High sugar (C-H bonds) → NEGATIVE for Diabetes goal
    - High sodium → NEGATIVE for Hypertension goal
    - Low polyphenols (O-H bonds) → NEGATIVE for Brain Health goal
    
    Recommendation: "AVOID - This food fails your profile"
    Better Alternative: "Choose avocados or walnuts instead"

Author: Wellomex AI Nutrition Team
Version: 1.0.0
Date: November 7, 2025
"""

import numpy as np
try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum
import logging
from datetime import datetime
from pathlib import Path
import json

# Optimization libraries
try:
    from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # scipy not available - use fallback

# Import from our modules
try:
    from atomic_molecular_profiler import (  # type: ignore
        HealthGoal, DiseaseCondition, NutrientMolecularBreakdown,
        UserHealthProfile, FoodRecommendation, MolecularBondProfile,
        ToxicContaminantProfile, ChemicalBondType
    )
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    logging.warning("atomic_molecular_profiler.py not found in path")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class RecommendationLevel(Enum):
    """Recommendation levels"""
    HIGHLY_RECOMMENDED = "HIGHLY_RECOMMENDED"  # Score 80-100
    RECOMMENDED = "RECOMMENDED"  # Score 60-79
    ACCEPTABLE = "ACCEPTABLE"  # Score 40-59
    NOT_RECOMMENDED = "NOT_RECOMMENDED"  # Score 20-39
    AVOID = "AVOID"  # Score 0-19


class OptimizationStrategy(Enum):
    """Optimization strategies"""
    MAXIMIZE_GOALS = "maximize_goals"  # Prioritize health goals
    MINIMIZE_RISKS = "minimize_risks"  # Prioritize disease management
    BALANCED = "balanced"  # Balance goals and risks
    THERAPEUTIC = "therapeutic"  # Disease treatment focus


# ============================================================================
# DISEASE MOLECULAR PROFILES
# ============================================================================

@dataclass
class DiseaseMolecularProfile:
    """Molecular profile for a specific disease"""
    disease: 'DiseaseCondition'
    
    # Molecules to MAXIMIZE (therapeutic)
    beneficial_molecules: Dict[str, float] = field(default_factory=dict)  # {molecule: importance_weight}
    
    # Molecules to MINIMIZE (harmful)
    harmful_molecules: Dict[str, float] = field(default_factory=dict)  # {molecule: risk_weight}
    
    # Nutrient constraints
    max_values: Dict[str, float] = field(default_factory=dict)  # {nutrient: max_per_day}
    min_values: Dict[str, float] = field(default_factory=dict)  # {nutrient: min_per_day}
    
    # Bond preferences
    beneficial_bonds: List['ChemicalBondType'] = field(default_factory=list)
    harmful_bonds: List['ChemicalBondType'] = field(default_factory=list)
    
    # Severity multiplier (1.0 = normal, 2.0 = strict restrictions)
    severity_multiplier: float = 1.0


class DiseaseProfileDatabase:
    """Database of disease-specific molecular profiles"""
    
    def __init__(self):
        self.profiles: Dict['DiseaseCondition', DiseaseMolecularProfile] = {}
        self._initialize_profiles()
    
    def _initialize_profiles(self):
        """Initialize disease profiles with evidence-based guidelines"""
        
        # Type 2 Diabetes
        if PROFILER_AVAILABLE:
            self.profiles[DiseaseCondition.DIABETES_T2] = DiseaseMolecularProfile(
                disease=DiseaseCondition.DIABETES_T2,
                beneficial_molecules={
                    'fiber': 2.0,  # Slows glucose absorption
                    'polyphenols': 1.5,  # Improves insulin sensitivity
                    'omega_3': 1.3,  # Reduces inflammation
                    'magnesium': 1.4,  # Glucose metabolism
                    'chromium': 1.2  # Insulin function
                },
                harmful_molecules={
                    'simple_sugars': 3.0,  # Spikes blood glucose
                    'saturated_fat': 1.8,  # Insulin resistance
                    'trans_fat': 2.5,  # Inflammation
                    'sodium': 1.5  # Often comorbid with hypertension
                },
                max_values={
                    'carbohydrates': 130,  # g/day (minimum for brain)
                    'simple_sugars': 25,  # g/day (ADA guideline)
                    'saturated_fat': 20,  # g/day
                    'sodium': 2300  # mg/day
                },
                min_values={
                    'fiber': 25,  # g/day
                    'protein': 60  # g/day
                },
                beneficial_bonds=[ChemicalBondType.O_H, ChemicalBondType.N_H],
                harmful_bonds=[ChemicalBondType.C_H],  # High C-H = high sugar/fat
                severity_multiplier=1.5
            )
            
            # Hypertension
            self.profiles[DiseaseCondition.HYPERTENSION] = DiseaseMolecularProfile(
                disease=DiseaseCondition.HYPERTENSION,
                beneficial_molecules={
                    'potassium': 2.0,  # Lowers blood pressure
                    'magnesium': 1.5,  # Vasodilation
                    'calcium': 1.3,  # BP regulation
                    'nitric_oxide_precursors': 1.8,  # Vasodilation
                    'polyphenols': 1.4,  # Endothelial function
                    'omega_3': 1.6  # Anti-inflammatory
                },
                harmful_molecules={
                    'sodium': 3.0,  # Primary culprit
                    'saturated_fat': 1.5,  # Endothelial dysfunction
                    'alcohol': 2.0,  # Raises BP
                    'caffeine': 1.2  # Temporary BP spike
                },
                max_values={
                    'sodium': 1500,  # mg/day (AHA recommendation)
                    'saturated_fat': 15,  # g/day
                    'caffeine': 200  # mg/day
                },
                min_values={
                    'potassium': 3500,  # mg/day
                    'magnesium': 400,  # mg/day
                    'fiber': 30  # g/day
                },
                beneficial_bonds=[ChemicalBondType.O_H],  # Polyphenols
                harmful_bonds=[],
                severity_multiplier=1.8
            )
            
            # Cardiovascular Disease
            self.profiles[DiseaseCondition.CVD] = DiseaseMolecularProfile(
                disease=DiseaseCondition.CVD,
                beneficial_molecules={
                    'omega_3': 2.5,  # Anti-inflammatory, anti-thrombotic
                    'fiber': 2.0,  # Lowers cholesterol
                    'polyphenols': 1.8,  # Antioxidant
                    'plant_sterols': 1.6,  # Blocks cholesterol absorption
                    'nitric_oxide_precursors': 1.5,  # Vasodilation
                    'coq10': 1.4  # Mitochondrial function
                },
                harmful_molecules={
                    'trans_fat': 3.0,  # Raises LDL, lowers HDL
                    'saturated_fat': 2.5,  # Raises LDL
                    'cholesterol': 2.0,  # Dietary cholesterol
                    'sodium': 2.0,  # Fluid retention, BP
                    'refined_carbs': 1.5  # Triglycerides
                },
                max_values={
                    'saturated_fat': 13,  # g/day (<7% calories)
                    'trans_fat': 0,  # g/day (avoid completely)
                    'cholesterol': 200,  # mg/day
                    'sodium': 1500  # mg/day
                },
                min_values={
                    'omega_3': 2,  # g/day (EPA+DHA)
                    'fiber': 30,  # g/day
                    'antioxidants': 5000  # ORAC units
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                harmful_bonds=[ChemicalBondType.C_H],
                severity_multiplier=2.0
            )
            
            # Obesity
            self.profiles[DiseaseCondition.OBESITY] = DiseaseMolecularProfile(
                disease=DiseaseCondition.OBESITY,
                beneficial_molecules={
                    'fiber': 2.5,  # Satiety, slows digestion
                    'protein': 2.0,  # Thermogenesis, satiety
                    'water': 1.8,  # Satiety, metabolism
                    'capsaicin': 1.3,  # Thermogenesis
                    'green_tea_catechins': 1.4,  # Fat oxidation
                    'resistant_starch': 1.5  # Gut health, satiety
                },
                harmful_molecules={
                    'simple_sugars': 2.5,  # Empty calories
                    'saturated_fat': 2.0,  # High calorie density
                    'trans_fat': 2.2,  # Inflammation
                    'alcohol': 2.0,  # Empty calories, inhibits fat burning
                    'fructose': 2.3  # Lipogenesis
                },
                max_values={
                    'calories': 1500,  # Deficit for weight loss
                    'fat': 50,  # g/day
                    'simple_sugars': 20,  # g/day
                    'alcohol': 0  # g/day (avoid)
                },
                min_values={
                    'protein': 80,  # g/day (preserve muscle)
                    'fiber': 35,  # g/day
                    'water': 2000  # ml/day
                },
                beneficial_bonds=[ChemicalBondType.N_H, ChemicalBondType.O_H],
                harmful_bonds=[ChemicalBondType.C_H],
                severity_multiplier=1.5
            )
            
            # Alzheimer's / Brain Health
            self.profiles[DiseaseCondition.ALZHEIMERS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.ALZHEIMERS,
                beneficial_molecules={
                    'dha': 2.5,  # Neuronal membrane
                    'polyphenols': 2.3,  # Neuroprotection
                    'vitamin_e': 1.8,  # Antioxidant
                    'b_vitamins': 1.7,  # Homocysteine reduction
                    'curcumin': 1.6,  # Anti-amyloid
                    'lutein': 1.4,  # Cognitive function
                    'choline': 1.5  # Acetylcholine precursor
                },
                harmful_molecules={
                    'trans_fat': 2.5,  # Neuroinflammation
                    'saturated_fat': 2.0,  # Blood-brain barrier
                    'simple_sugars': 2.2,  # Insulin resistance
                    'aluminum': 1.8,  # Neurotoxic
                    'advanced_glycation_end_products': 2.0  # Oxidative stress
                },
                max_values={
                    'saturated_fat': 15,  # g/day
                    'simple_sugars': 25,  # g/day
                    'trans_fat': 0  # g/day
                },
                min_values={
                    'dha': 1,  # g/day
                    'polyphenols': 500,  # mg/day
                    'b12': 2.4,  # mcg/day
                    'folate': 400  # mcg/day
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H, ChemicalBondType.AROMATIC],
                harmful_bonds=[ChemicalBondType.C_H],
                severity_multiplier=1.8
            )
            
            # Kidney Disease
            self.profiles[DiseaseCondition.KIDNEY_DISEASE] = DiseaseMolecularProfile(
                disease=DiseaseCondition.KIDNEY_DISEASE,
                beneficial_molecules={
                    'omega_3': 1.8,  # Anti-inflammatory
                    'antioxidants': 1.6,  # Oxidative stress
                    'fiber': 1.4  # Gut health
                },
                harmful_molecules={
                    'protein': 2.5,  # Kidney stress (late stage)
                    'phosphorus': 2.8,  # Bone disease
                    'potassium': 2.5,  # Hyperkalemia risk
                    'sodium': 2.3,  # Fluid retention
                    'oxalates': 1.8  # Kidney stones
                },
                max_values={
                    'protein': 50,  # g/day (stage-dependent)
                    'phosphorus': 800,  # mg/day
                    'potassium': 2000,  # mg/day
                    'sodium': 1500,  # mg/day
                    'fluid': 1500  # ml/day
                },
                min_values={
                    'calories': 2000  # Prevent catabolism
                },
                beneficial_bonds=[ChemicalBondType.O_H],
                harmful_bonds=[ChemicalBondType.N_H, ChemicalBondType.P_O],
                severity_multiplier=2.5
            )
            
            # ===================================================================
            # PHASE 1: ADDITIONAL DISEASES (10 more conditions)
            # ===================================================================
            
            # Liver Disease / Fatty Liver (NAFLD)
            self.profiles[DiseaseCondition.FATTY_LIVER] = DiseaseMolecularProfile(
                disease=DiseaseCondition.FATTY_LIVER,
                beneficial_molecules={
                    'omega_3': 2.2,  # Reduces liver fat
                    'vitamin_e': 2.0,  # Antioxidant (proven in trials)
                    'choline': 1.8,  # Fat transport
                    'antioxidants': 1.6,  # Oxidative stress
                    'fiber': 1.5,  # Gut-liver axis
                    'coffee': 1.4  # Hepatoprotective
                },
                harmful_molecules={
                    'fructose': 3.0,  # Primary lipogenesis driver
                    'saturated_fat': 2.5,  # Liver fat accumulation
                    'alcohol': 3.0,  # Hepatotoxic
                    'trans_fat': 2.8,  # Inflammation
                    'refined_carbs': 2.3  # Insulin resistance
                },
                max_values={
                    'fructose': 15,  # g/day (avoid high-fructose corn syrup)
                    'saturated_fat': 15,  # g/day
                    'alcohol': 0,  # g/day (strict abstinence)
                    'calories': 1800  # Weight loss critical
                },
                min_values={
                    'omega_3': 2,  # g/day
                    'vitamin_e': 400,  # IU/day (therapeutic dose)
                    'protein': 70  # g/day (preserve muscle during weight loss)
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                harmful_bonds=[ChemicalBondType.C_H],
                severity_multiplier=2.0
            )
            
            # Cancer Prevention / Active Cancer
            self.profiles[DiseaseCondition.CANCER] = DiseaseMolecularProfile(
                disease=DiseaseCondition.CANCER,
                beneficial_molecules={
                    'cruciferous_compounds': 2.5,  # Sulforaphane (broccoli)
                    'curcumin': 2.3,  # Anti-cancer properties
                    'resveratrol': 2.0,  # Apoptosis induction
                    'quercetin': 1.8,  # Flavonoid
                    'omega_3': 1.9,  # Anti-inflammatory
                    'vitamin_d': 1.7,  # Cell differentiation
                    'green_tea_catechins': 1.8,  # EGCG
                    'fiber': 1.6  # Colon cancer prevention
                },
                harmful_molecules={
                    'processed_meat': 3.0,  # WHO Group 1 carcinogen
                    'red_meat': 2.0,  # WHO Group 2A
                    'alcohol': 2.5,  # Multiple cancer types
                    'acrylamide': 2.3,  # Fried foods
                    'heterocyclic_amines': 2.8,  # Charred meat
                    'nitrites': 2.5,  # Preservatives
                    'simple_sugars': 2.0  # Feeds cancer cells (Warburg effect)
                },
                max_values={
                    'red_meat': 350,  # g/week (not per day!)
                    'processed_meat': 0,  # g/day (avoid)
                    'alcohol': 10,  # g/day (minimal)
                    'simple_sugars': 20  # g/day
                },
                min_values={
                    'cruciferous': 100,  # g/day (broccoli, kale)
                    'fiber': 40,  # g/day
                    'polyphenols': 1000  # mg/day
                },
                beneficial_bonds=[ChemicalBondType.O_H, ChemicalBondType.AROMATIC, ChemicalBondType.S_H],
                harmful_bonds=[ChemicalBondType.N_N],  # Nitrosamines
                severity_multiplier=2.5
            )
            
            # Osteoporosis / Bone Health
            self.profiles[DiseaseCondition.OSTEOPOROSIS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.OSTEOPOROSIS,
                beneficial_molecules={
                    'calcium': 2.5,  # Bone mineral
                    'vitamin_d': 2.5,  # Calcium absorption
                    'vitamin_k2': 2.0,  # Directs calcium to bones
                    'magnesium': 1.8,  # Bone structure
                    'protein': 1.7,  # Bone matrix
                    'boron': 1.3,  # Bone metabolism
                    'vitamin_c': 1.4  # Collagen synthesis
                },
                harmful_molecules={
                    'sodium': 2.0,  # Calcium excretion
                    'caffeine': 1.5,  # Calcium excretion (high doses)
                    'phytates': 1.3,  # Calcium binding
                    'oxalates': 1.3,  # Calcium binding
                    'cola_phosphoric_acid': 1.8  # Bone demineralization
                },
                max_values={
                    'sodium': 1500,  # mg/day
                    'caffeine': 300,  # mg/day
                    'phosphoric_acid': 700  # mg/day
                },
                min_values={
                    'calcium': 1200,  # mg/day
                    'vitamin_d': 800,  # IU/day
                    'vitamin_k2': 90,  # mcg/day
                    'protein': 1.0  # g/kg (not too low!)
                },
                beneficial_bonds=[ChemicalBondType.O_H, ChemicalBondType.N_H],
                harmful_bonds=[ChemicalBondType.P_O],
                severity_multiplier=1.8
            )
            
            # Inflammatory Bowel Disease (IBD) - Crohn's/Ulcerative Colitis
            self.profiles[DiseaseCondition.IBD] = DiseaseMolecularProfile(
                disease=DiseaseCondition.IBD,
                beneficial_molecules={
                    'omega_3': 2.5,  # Anti-inflammatory
                    'curcumin': 2.0,  # Reduces inflammation
                    'glutamine': 1.8,  # Gut lining repair
                    'vitamin_d': 1.7,  # Immune modulation
                    'zinc': 1.6,  # Wound healing
                    'soluble_fiber': 1.5,  # Butyrate (remission only)
                    'polyphenols': 1.4  # Anti-inflammatory
                },
                harmful_molecules={
                    'insoluble_fiber': 2.5,  # Irritation during flare
                    'lactose': 2.0,  # Often intolerant
                    'gluten': 1.8,  # Inflammatory for some
                    'emulsifiers': 2.3,  # Disrupts mucus layer
                    'carrageenan': 2.0,  # Inflammatory
                    'sugar': 1.7  # Feeds harmful bacteria
                },
                max_values={
                    'insoluble_fiber': 10,  # g/day (during flare)
                    'lactose': 5,  # g/day
                    'saturated_fat': 20  # g/day
                },
                min_values={
                    'omega_3': 2,  # g/day
                    'glutamine': 5,  # g/day
                    'vitamin_d': 2000,  # IU/day
                    'calories': 2200  # Prevent malnutrition
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=2.2
            )
            
            # PCOS (Polycystic Ovary Syndrome)
            self.profiles[DiseaseCondition.PCOS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.PCOS,
                beneficial_molecules={
                    'inositol': 2.5,  # Insulin sensitivity (myo + d-chiro)
                    'omega_3': 2.0,  # Reduces androgens
                    'chromium': 1.8,  # Glucose metabolism
                    'magnesium': 1.6,  # Insulin resistance
                    'fiber': 2.0,  # Blood sugar control
                    'spearmint': 1.5,  # Anti-androgen
                    'vitamin_d': 1.7  # Ovulation, fertility
                },
                harmful_molecules={
                    'simple_sugars': 2.8,  # Insulin spikes
                    'refined_carbs': 2.5,  # High glycemic
                    'trans_fat': 2.0,  # Inflammation
                    'dairy': 1.6,  # Insulin-like growth factor
                    'soy_isoflavones': 1.3  # Hormonal (controversial)
                },
                max_values={
                    'carbohydrates': 130,  # g/day (low-carb beneficial)
                    'simple_sugars': 20,  # g/day
                    'glycemic_load': 80,  # per day
                    'saturated_fat': 15  # g/day
                },
                min_values={
                    'inositol': 2000,  # mg/day (myo-inositol)
                    'omega_3': 1.5,  # g/day
                    'fiber': 30,  # g/day
                    'protein': 80  # g/day (satiety)
                },
                beneficial_bonds=[ChemicalBondType.O_H, ChemicalBondType.DOUBLE_BOND],
                harmful_bonds=[ChemicalBondType.C_H],
                severity_multiplier=1.7
            )
            
            # Gout / Hyperuricemia
            self.profiles[DiseaseCondition.GOUT] = DiseaseMolecularProfile(
                disease=DiseaseCondition.GOUT,
                beneficial_molecules={
                    'vitamin_c': 2.0,  # Lowers uric acid
                    'tart_cherry': 2.5,  # Anti-inflammatory, lowers uric acid
                    'coffee': 1.5,  # Lowers uric acid
                    'water': 2.0,  # Flushes uric acid
                    'folate': 1.3  # Xanthine oxidase inhibitor
                },
                harmful_molecules={
                    'purines': 3.0,  # Broken down to uric acid
                    'fructose': 2.8,  # Increases uric acid production
                    'alcohol': 3.0,  # Beer worst (purines + alcohol)
                    'organ_meats': 2.5,  # Very high purines
                    'shellfish': 2.3,  # High purines
                    'red_meat': 2.0  # Moderate purines
                },
                max_values={
                    'purines': 150,  # mg/day
                    'fructose': 25,  # g/day
                    'alcohol': 0,  # g/day (especially beer)
                    'organ_meats': 0  # g/day
                },
                min_values={
                    'vitamin_c': 500,  # mg/day
                    'water': 3000,  # ml/day (critical!)
                    'tart_cherry': 100  # g/day
                },
                beneficial_bonds=[ChemicalBondType.O_H],
                harmful_bonds=[ChemicalBondType.N_H, ChemicalBondType.N_C],  # Purine ring
                severity_multiplier=2.0
            )
            
            # Thyroid Disorders (Hypothyroidism)
            self.profiles[DiseaseCondition.HYPOTHYROID] = DiseaseMolecularProfile(
                disease=DiseaseCondition.HYPOTHYROID,
                beneficial_molecules={
                    'iodine': 2.5,  # T3/T4 synthesis
                    'selenium': 2.3,  # T4 to T3 conversion
                    'zinc': 1.8,  # Thyroid hormone production
                    'iron': 1.7,  # Thyroid peroxidase
                    'vitamin_d': 1.6,  # Immune modulation (Hashimoto's)
                    'tyrosine': 1.9  # Thyroid hormone precursor
                },
                harmful_molecules={
                    'goitrogens': 2.5,  # Blocks iodine (raw cruciferous)
                    'soy_isoflavones': 2.0,  # Interferes with thyroid
                    'gluten': 2.2,  # Hashimoto's trigger
                    'excess_fiber': 1.5,  # Impairs levothyroxine absorption
                    'calcium': 1.3,  # Medication interaction
                    'iron': 1.3  # Medication interaction
                },
                max_values={
                    'goitrogens': 100,  # g/day (raw cruciferous)
                    'soy': 50,  # g/day
                    'fiber': 30  # g/day (too much impairs meds)
                },
                min_values={
                    'iodine': 150,  # mcg/day
                    'selenium': 55,  # mcg/day
                    'zinc': 11,  # mg/day
                    'protein': 75  # g/day (metabolism support)
                },
                beneficial_bonds=[ChemicalBondType.N_H],
                harmful_bonds=[],
                severity_multiplier=1.6
            )
            
            # Depression / Anxiety
            self.profiles[DiseaseCondition.DEPRESSION] = DiseaseMolecularProfile(
                disease=DiseaseCondition.DEPRESSION,
                beneficial_molecules={
                    'omega_3': 2.5,  # EPA especially (1-2g EPA)
                    'vitamin_d': 2.0,  # Serotonin synthesis
                    'b_vitamins': 2.2,  # Neurotransmitter production
                    'magnesium': 1.8,  # NMDA receptor, anxiety
                    'tryptophan': 1.7,  # Serotonin precursor
                    'saffron': 1.6,  # Evidence in trials
                    'probiotics': 1.5,  # Gut-brain axis
                    'zinc': 1.6  # Neurotransmitter function
                },
                harmful_molecules={
                    'trans_fat': 2.0,  # Inflammation
                    'refined_sugar': 2.2,  # Blood sugar swings, inflammation
                    'alcohol': 2.5,  # Depressant
                    'caffeine': 1.5,  # Anxiety (sensitive individuals)
                    'artificial_sweeteners': 1.3  # Microbiome disruption
                },
                max_values={
                    'simple_sugars': 25,  # g/day
                    'trans_fat': 0,  # g/day
                    'alcohol': 10,  # g/day
                    'caffeine': 200  # mg/day
                },
                min_values={
                    'omega_3': 2,  # g/day (EPA 1-2g)
                    'vitamin_d': 2000,  # IU/day
                    'b12': 2.4,  # mcg/day
                    'folate': 400,  # mcg/day
                    'magnesium': 400  # mg/day
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                harmful_bonds=[ChemicalBondType.C_H],
                severity_multiplier=1.5
            )
            
            # Metabolic Syndrome
            self.profiles[DiseaseCondition.METABOLIC_SYNDROME] = DiseaseMolecularProfile(
                disease=DiseaseCondition.METABOLIC_SYNDROME,
                beneficial_molecules={
                    'fiber': 2.5,  # All components
                    'omega_3': 2.2,  # Triglycerides, inflammation
                    'magnesium': 1.8,  # Insulin sensitivity
                    'polyphenols': 1.7,  # Inflammation
                    'resistant_starch': 1.6,  # Glucose control
                    'vinegar': 1.4  # Acetic acid (glucose response)
                },
                harmful_molecules={
                    'fructose': 3.0,  # Central to MetS pathology
                    'trans_fat': 2.8,  # All MetS components worsen
                    'saturated_fat': 2.3,  # Insulin resistance
                    'sodium': 2.0,  # Hypertension component
                    'refined_carbs': 2.5  # Hyperglycemia, triglycerides
                },
                max_values={
                    'fructose': 15,  # g/day
                    'saturated_fat': 15,  # g/day
                    'sodium': 1500,  # mg/day
                    'refined_carbs': 50,  # g/day
                    'calories': 1800  # Deficit for weight loss
                },
                min_values={
                    'fiber': 35,  # g/day
                    'omega_3': 2,  # g/day
                    'protein': 90,  # g/day (30% calories)
                    'vegetables': 400  # g/day
                },
                beneficial_bonds=[ChemicalBondType.O_H, ChemicalBondType.DOUBLE_BOND],
                harmful_bonds=[ChemicalBondType.C_H],
                severity_multiplier=2.0
            )
            
            # Autoimmune Diseases (General)
            self.profiles[DiseaseCondition.AUTOIMMUNE] = DiseaseMolecularProfile(
                disease=DiseaseCondition.AUTOIMMUNE,
                beneficial_molecules={
                    'omega_3': 2.5,  # Immune modulation
                    'vitamin_d': 2.3,  # T-reg cells
                    'curcumin': 2.0,  # NF-kB inhibition
                    'polyphenols': 1.8,  # Anti-inflammatory
                    'glutamine': 1.6,  # Gut barrier
                    'probiotics': 1.7,  # Immune tolerance
                    'vitamin_a': 1.5  # Immune regulation
                },
                harmful_molecules={
                    'gluten': 2.5,  # Leaky gut, molecular mimicry
                    'lectins': 2.0,  # Gut permeability
                    'nightshades': 1.5,  # Inflammatory (some people)
                    'refined_sugar': 2.0,  # Inflammation
                    'trans_fat': 2.2,  # Inflammation
                    'alcohol': 1.8  # Gut barrier damage
                },
                max_values={
                    'gluten': 0,  # g/day (elimination trial)
                    'sugar': 20,  # g/day
                    'omega_6': 10,  # g/day (high omega-6:3 ratio bad)
                    'alcohol': 0  # g/day
                },
                min_values={
                    'omega_3': 3,  # g/day
                    'vitamin_d': 4000,  # IU/day (therapeutic)
                    'antioxidants': 10000,  # ORAC units
                    'glutamine': 5  # g/day
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=2.0
            )
            
            # ===================================================================
            # PHASE 2: ADDITIONAL DISEASES (14 more conditions)
            # ===================================================================
            
            # Diabetes Type 1
            self.profiles[DiseaseCondition.DIABETES_T1] = DiseaseMolecularProfile(
                disease=DiseaseCondition.DIABETES_T1,
                beneficial_molecules={
                    'fiber': 2.0,  # Slows glucose absorption
                    'omega_3': 1.8,  # Reduces inflammation
                    'chromium': 1.5,  # Insulin sensitivity
                    'magnesium': 1.6,  # Glucose metabolism
                    'antioxidants': 1.7  # Oxidative stress
                },
                harmful_molecules={
                    'simple_sugars': 3.0,  # Requires insulin dosing
                    'saturated_fat': 1.8,  # Insulin resistance
                    'trans_fat': 2.5,  # Inflammation
                    'high_glycemic_carbs': 2.3  # Blood sugar spikes
                },
                max_values={
                    'carbohydrates': 200,  # g/day (must count for insulin)
                    'simple_sugars': 30,  # g/day
                    'glycemic_load': 100  # per day
                },
                min_values={
                    'fiber': 30,  # g/day
                    'protein': 70  # g/day
                },
                beneficial_bonds=[ChemicalBondType.O_H, ChemicalBondType.N_H],
                harmful_bonds=[ChemicalBondType.C_H],
                severity_multiplier=2.3
            )
            
            # Celiac Disease
            self.profiles[DiseaseCondition.CELIAC] = DiseaseMolecularProfile(
                disease=DiseaseCondition.CELIAC,
                beneficial_molecules={
                    'fiber': 2.0,  # Gut health (gluten-free sources)
                    'omega_3': 1.8,  # Anti-inflammatory
                    'zinc': 1.9,  # Healing, often deficient
                    'iron': 2.0,  # Often deficient (malabsorption)
                    'b_vitamins': 1.8,  # Often deficient
                    'calcium': 1.7,  # Bone health
                    'vitamin_d': 1.8  # Absorption issues
                },
                harmful_molecules={
                    'gluten': 3.0,  # ZERO TOLERANCE - autoimmune trigger
                    'wheat': 3.0,  # Contains gluten
                    'barley': 3.0,  # Contains gluten
                    'rye': 3.0,  # Contains gluten
                    'cross_contamination': 2.5  # Even trace amounts
                },
                max_values={
                    'gluten': 0,  # Absolute zero (< 20 ppm even)
                    'wheat': 0,
                    'barley': 0,
                    'rye': 0
                },
                min_values={
                    'iron': 18,  # mg/day (replenish stores)
                    'calcium': 1200,  # mg/day
                    'vitamin_d': 800,  # IU/day
                    'fiber': 25  # g/day (from GF sources)
                },
                beneficial_bonds=[ChemicalBondType.O_H, ChemicalBondType.N_H],
                harmful_bonds=[],
                severity_multiplier=2.8  # Strictest diet
            )
            
            # Anemia (Iron Deficiency)
            self.profiles[DiseaseCondition.ANEMIA] = DiseaseMolecularProfile(
                disease=DiseaseCondition.ANEMIA,
                beneficial_molecules={
                    'iron': 3.0,  # Primary deficiency
                    'vitamin_c': 2.5,  # Enhances iron absorption (3-4x)
                    'vitamin_b12': 2.0,  # RBC production
                    'folate': 2.0,  # RBC production
                    'copper': 1.5,  # Iron metabolism
                    'vitamin_a': 1.4  # Iron mobilization
                },
                harmful_molecules={
                    'phytates': 2.5,  # Binds iron (grains, legumes)
                    'calcium': 2.0,  # Competes with iron absorption
                    'tannins': 2.3,  # Binds iron (tea, coffee)
                    'oxalates': 1.8,  # Binds iron
                    'zinc': 1.5  # High doses compete with iron
                },
                max_values={
                    'calcium': 500,  # mg per meal (away from iron)
                    'tea_coffee': 0,  # With iron-rich meals
                    'phytates': 50  # mg per meal
                },
                min_values={
                    'iron': 18,  # mg/day (women), 30+ if severe
                    'vitamin_c': 100,  # mg per iron-rich meal
                    'b12': 2.4,  # mcg/day
                    'folate': 400  # mcg/day
                },
                beneficial_bonds=[ChemicalBondType.N_H, ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=1.8
            )
            
            # Migraines / Chronic Headaches
            self.profiles[DiseaseCondition.MIGRAINES] = DiseaseMolecularProfile(
                disease=DiseaseCondition.MIGRAINES,
                beneficial_molecules={
                    'magnesium': 2.8,  # Vasodilation, NMDA (400-600mg)
                    'riboflavin': 2.5,  # B2 (400mg/day proven)
                    'coq10': 2.3,  # Mitochondrial (300mg/day)
                    'omega_3': 2.0,  # Anti-inflammatory
                    'ginger': 1.8,  # Anti-inflammatory
                    'feverfew': 1.7  # Traditional herbal
                },
                harmful_molecules={
                    'tyramine': 2.8,  # Aged cheese, wine, triggers
                    'msg': 2.5,  # Glutamate trigger
                    'nitrites': 2.7,  # Processed meats trigger
                    'alcohol': 2.5,  # Red wine especially
                    'caffeine': 2.0,  # Withdrawal trigger (paradox)
                    'artificial_sweeteners': 2.2  # Aspartame trigger
                },
                max_values={
                    'tyramine': 6,  # mg/day (very low)
                    'alcohol': 0,  # g/day (especially red wine)
                    'caffeine': 100,  # mg/day (consistent, no withdrawal)
                    'processed_meats': 0  # g/day
                },
                min_values={
                    'magnesium': 400,  # mg/day (therapeutic dose)
                    'riboflavin': 400,  # mg/day (high dose proven)
                    'coq10': 300,  # mg/day
                    'water': 2500  # ml/day (dehydration trigger)
                },
                beneficial_bonds=[ChemicalBondType.O_H],
                harmful_bonds=[ChemicalBondType.N_N],  # Nitrites
                severity_multiplier=2.0
            )
            
            # Asthma / Respiratory Health
            self.profiles[DiseaseCondition.ASTHMA] = DiseaseMolecularProfile(
                disease=DiseaseCondition.ASTHMA,
                beneficial_molecules={
                    'omega_3': 2.5,  # Anti-inflammatory airways
                    'vitamin_c': 2.0,  # Antioxidant, lung function
                    'vitamin_d': 2.2,  # Immune modulation
                    'magnesium': 2.0,  # Bronchodilation
                    'quercetin': 1.8,  # Mast cell stabilizer
                    'vitamin_e': 1.6,  # Antioxidant
                    'selenium': 1.5  # Antioxidant
                },
                harmful_molecules={
                    'sulfites': 2.8,  # Wine, dried fruit - bronchospasm
                    'salicylates': 2.0,  # Aspirin sensitivity
                    'trans_fat': 2.2,  # Inflammation
                    'omega_6_excess': 2.0,  # Pro-inflammatory
                    'food_allergens': 2.5  # Individual triggers
                },
                max_values={
                    'sulfites': 10,  # mg/day (very low)
                    'omega_6': 12,  # g/day
                    'trans_fat': 0  # g/day
                },
                min_values={
                    'omega_3': 2,  # g/day
                    'vitamin_c': 500,  # mg/day
                    'vitamin_d': 2000,  # IU/day
                    'magnesium': 400  # mg/day
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=1.8
            )
            
            # GERD / Acid Reflux
            self.profiles[DiseaseCondition.GERD] = DiseaseMolecularProfile(
                disease=DiseaseCondition.GERD,
                beneficial_molecules={
                    'fiber': 2.0,  # Gastroprotective
                    'ginger': 1.8,  # Prokinetic
                    'aloe_vera': 1.6,  # Soothing
                    'glutamine': 1.5,  # Mucosal healing
                    'melatonin': 1.7  # LES pressure
                },
                harmful_molecules={
                    'fat': 2.5,  # Delays gastric emptying
                    'caffeine': 2.3,  # Relaxes LES
                    'alcohol': 2.5,  # Relaxes LES
                    'chocolate': 2.0,  # Methylxanthines
                    'peppermint': 2.0,  # Relaxes LES
                    'tomatoes': 1.8,  # Acidic
                    'citrus': 1.8,  # Acidic
                    'spicy_foods': 1.7  # Irritating
                },
                max_values={
                    'fat': 50,  # g/day
                    'caffeine': 0,  # mg/day
                    'alcohol': 0,  # g/day
                    'acidic_foods': 100  # g/day
                },
                min_values={
                    'fiber': 25,  # g/day
                    'water': 2000  # ml/day (between meals)
                },
                beneficial_bonds=[ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=1.5
            )
            
            # IBS (Irritable Bowel Syndrome)
            self.profiles[DiseaseCondition.IBS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.IBS,
                beneficial_molecules={
                    'soluble_fiber': 2.5,  # Psyllium, oats
                    'peppermint_oil': 2.0,  # Antispasmodic
                    'probiotics': 2.0,  # Microbiome
                    'ginger': 1.7,  # Prokinetic
                    'turmeric': 1.6,  # Anti-inflammatory
                    'glutamine': 1.5  # Gut lining
                },
                harmful_molecules={
                    'fodmaps': 2.8,  # Fermentable carbs (LOW FODMAP diet)
                    'insoluble_fiber': 2.0,  # Can worsen symptoms
                    'caffeine': 1.8,  # Stimulates motility
                    'alcohol': 2.0,  # Irritating
                    'artificial_sweeteners': 2.2,  # Sorbitol, mannitol
                    'fatty_foods': 1.9  # Triggers cramping
                },
                max_values={
                    'fodmaps': 10,  # g/day (strict LOW FODMAP)
                    'insoluble_fiber': 10,  # g/day
                    'caffeine': 100,  # mg/day
                    'fat': 40  # g/day
                },
                min_values={
                    'soluble_fiber': 15,  # g/day
                    'probiotics': 10_000_000_000,  # CFU/day (10 billion)
                    'water': 2000  # ml/day
                },
                beneficial_bonds=[ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=1.7
            )
            
            # Eczema / Psoriasis (Skin Inflammatory)
            self.profiles[DiseaseCondition.ECZEMA] = DiseaseMolecularProfile(
                disease=DiseaseCondition.ECZEMA,
                beneficial_molecules={
                    'omega_3': 2.8,  # Anti-inflammatory skin
                    'vitamin_d': 2.5,  # Immune modulation
                    'probiotics': 2.2,  # Gut-skin axis
                    'zinc': 2.0,  # Wound healing
                    'vitamin_e': 1.8,  # Antioxidant
                    'quercetin': 1.7,  # Anti-inflammatory
                    'evening_primrose': 1.6  # GLA
                },
                harmful_molecules={
                    'omega_6_excess': 2.5,  # Pro-inflammatory
                    'gluten': 2.0,  # Inflammatory (some)
                    'dairy': 2.0,  # Inflammatory (some)
                    'sugar': 2.2,  # Inflammation
                    'alcohol': 1.8,  # Dehydrating
                    'nightshades': 1.5  # Individual trigger
                },
                max_values={
                    'omega_6': 10,  # g/day
                    'sugar': 25,  # g/day
                    'alcohol': 0  # g/day
                },
                min_values={
                    'omega_3': 3,  # g/day (high dose)
                    'vitamin_d': 2000,  # IU/day
                    'zinc': 15,  # mg/day
                    'water': 3000  # ml/day (hydration critical)
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=1.8
            )
            
            # ADHD / Focus Disorders
            self.profiles[DiseaseCondition.ADHD] = DiseaseMolecularProfile(
                disease=DiseaseCondition.ADHD,
                beneficial_molecules={
                    'omega_3': 2.8,  # Brain structure, EPA+DHA
                    'iron': 2.5,  # Dopamine synthesis
                    'zinc': 2.3,  # Neurotransmitter regulation
                    'magnesium': 2.0,  # ADHD often deficient
                    'protein': 2.2,  # Steady neurotransmitters
                    'b_vitamins': 1.9,  # Energy, focus
                    'l_theanine': 1.7  # Calm focus
                },
                harmful_molecules={
                    'artificial_colors': 2.8,  # Feingold diet (Red 40, Yellow 5)
                    'sugar': 2.5,  # Blood sugar swings
                    'artificial_sweeteners': 2.0,  # Some sensitive
                    'caffeine': 1.5,  # Paradoxical (can help some)
                    'processed_foods': 2.2  # Additives
                },
                max_values={
                    'artificial_colors': 0,  # mg/day
                    'sugar': 20,  # g/day
                    'processed_foods': 50  # g/day
                },
                min_values={
                    'omega_3': 2,  # g/day (1g EPA + 1g DHA)
                    'iron': 18,  # mg/day
                    'zinc': 15,  # mg/day
                    'protein': 1.5,  # g/kg
                    'magnesium': 400  # mg/day
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.N_H],
                harmful_bonds=[],
                severity_multiplier=1.7
            )
            
            # Chronic Fatigue Syndrome (CFS/ME)
            self.profiles[DiseaseCondition.CHRONIC_FATIGUE] = DiseaseMolecularProfile(
                disease=DiseaseCondition.CHRONIC_FATIGUE,
                beneficial_molecules={
                    'coq10': 2.8,  # Mitochondrial energy (200-400mg)
                    'b_vitamins': 2.5,  # Energy metabolism
                    'magnesium': 2.3,  # ATP production
                    'l_carnitine': 2.2,  # Mitochondrial transport
                    'd_ribose': 2.0,  # ATP regeneration
                    'nadh': 1.9,  # Cellular energy
                    'omega_3': 1.8  # Anti-inflammatory
                },
                harmful_molecules={
                    'sugar': 2.5,  # Energy crashes
                    'caffeine': 2.0,  # Adrenal stress (long-term)
                    'alcohol': 2.3,  # Depletes B vitamins
                    'processed_foods': 2.0  # Nutrient-poor
                },
                max_values={
                    'sugar': 25,  # g/day
                    'caffeine': 100,  # mg/day
                    'alcohol': 0  # g/day
                },
                min_values={
                    'coq10': 200,  # mg/day
                    'b_complex': 100,  # mg/day (B1, B2, B3, B6)
                    'b12': 1000,  # mcg/day (high dose)
                    'magnesium': 400,  # mg/day
                    'l_carnitine': 1000  # mg/day
                },
                beneficial_bonds=[ChemicalBondType.P_O, ChemicalBondType.N_H],
                harmful_bonds=[ChemicalBondType.C_H],
                severity_multiplier=2.0
            )
            
            # Fibromyalgia
            self.profiles[DiseaseCondition.FIBROMYALGIA] = DiseaseMolecularProfile(
                disease=DiseaseCondition.FIBROMYALGIA,
                beneficial_molecules={
                    'magnesium': 2.8,  # Muscle relaxation, pain
                    'malic_acid': 2.5,  # Energy, combined with Mg
                    'vitamin_d': 2.3,  # Pain, often deficient
                    'omega_3': 2.2,  # Anti-inflammatory
                    'coq10': 2.0,  # Mitochondrial function
                    '5_htp': 1.8,  # Serotonin, sleep
                    'sam_e': 1.7  # Mood, pain
                },
                harmful_molecules={
                    'msg': 2.8,  # Excitotoxin, pain amplifier
                    'aspartame': 2.7,  # Excitotoxin
                    'sugar': 2.3,  # Inflammation
                    'gluten': 2.0,  # Inflammatory (some)
                    'nightshades': 1.8,  # Pain trigger (some)
                    'caffeine': 1.7  # Sleep disruption
                },
                max_values={
                    'msg': 0,  # mg/day
                    'aspartame': 0,  # mg/day
                    'sugar': 20,  # g/day
                    'caffeine': 50  # mg/day
                },
                min_values={
                    'magnesium': 600,  # mg/day (high dose)
                    'malic_acid': 1200,  # mg/day (with Mg)
                    'vitamin_d': 2000,  # IU/day
                    'omega_3': 2  # g/day
                },
                beneficial_bonds=[ChemicalBondType.O_H, ChemicalBondType.DOUBLE_BOND],
                harmful_bonds=[],
                severity_multiplier=2.0
            )
            
            # Diverticulitis / Diverticulosis
            self.profiles[DiseaseCondition.DIVERTICULITIS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.DIVERTICULITIS,
                beneficial_molecules={
                    'fiber': 2.5,  # Prevention (soluble > insoluble)
                    'probiotics': 2.0,  # Gut health
                    'omega_3': 1.8,  # Anti-inflammatory
                    'glutamine': 1.6,  # Gut lining
                    'vitamin_d': 1.5  # Immune function
                },
                harmful_molecules={
                    'red_meat': 2.5,  # Increases risk
                    'processed_meat': 2.7,  # Inflammation
                    'refined_grains': 2.0,  # Low fiber
                    'nuts_seeds': 1.5,  # OLD MYTH - actually beneficial!
                    'alcohol': 2.0  # Inflammation
                },
                max_values={
                    'red_meat': 100,  # g/day
                    'processed_meat': 0,  # g/day
                    'alcohol': 10  # g/day
                },
                min_values={
                    'fiber': 30,  # g/day (gradually increase)
                    'water': 2500,  # ml/day (with fiber)
                    'probiotics': 10_000_000_000  # CFU/day
                },
                beneficial_bonds=[ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=1.8
            )
            
            # Sleep Apnea
            self.profiles[DiseaseCondition.SLEEP_APNEA] = DiseaseMolecularProfile(
                disease=DiseaseCondition.SLEEP_APNEA,
                beneficial_molecules={
                    'protein': 2.0,  # Weight loss (if obese)
                    'fiber': 2.0,  # Weight loss
                    'omega_3': 1.8,  # Anti-inflammatory airways
                    'vitamin_d': 1.7,  # Often deficient
                    'antioxidants': 1.6  # Oxidative stress
                },
                harmful_molecules={
                    'alcohol': 3.0,  # Relaxes airways (WORST)
                    'sedatives': 2.8,  # Muscle relaxation
                    'calories_excess': 2.5,  # Obesity main cause
                    'saturated_fat': 2.0,  # Weight gain
                    'sugar': 2.2  # Weight gain, inflammation
                },
                max_values={
                    'alcohol': 0,  # g/day (especially before bed)
                    'calories': 1800,  # Deficit if overweight
                    'saturated_fat': 15,  # g/day
                    'sugar': 20  # g/day
                },
                min_values={
                    'protein': 1.2,  # g/kg (preserve muscle)
                    'fiber': 30,  # g/day
                    'water': 2000  # ml/day
                },
                beneficial_bonds=[ChemicalBondType.N_H, ChemicalBondType.O_H],
                harmful_bonds=[ChemicalBondType.C_H],
                severity_multiplier=2.2
            )
            
            # Gastroparesis (Delayed Gastric Emptying)
            self.profiles[DiseaseCondition.GASTROPARESIS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.GASTROPARESIS,
                beneficial_molecules={
                    'ginger': 2.5,  # Prokinetic
                    'vitamin_b6': 2.0,  # Nausea
                    'simple_carbs': 2.0,  # Easy digestion (exception!)
                    'pureed_foods': 1.8,  # Easier emptying
                    'probiotics': 1.6  # Gut health
                },
                harmful_molecules={
                    'fiber': 2.8,  # Delays emptying (opposite of usual!)
                    'fat': 2.8,  # Delays emptying significantly
                    'protein': 2.0,  # Delays emptying (moderate)
                    'alcohol': 2.2,  # Delays emptying
                    'carbonated': 1.8  # Bloating
                },
                max_values={
                    'fiber': 10,  # g/day (very low)
                    'fat': 40,  # g/day (low-fat critical)
                    'protein': 60,  # g/day (moderate)
                    'meal_size': 200  # g per meal (small frequent)
                },
                min_values={
                    'meals_per_day': 5,  # Small frequent meals
                    'liquid_calories': 800,  # kcal/day (easier)
                    'ginger': 1000  # mg/day
                },
                beneficial_bonds=[ChemicalBondType.C_H],  # Simple carbs OK here!
                harmful_bonds=[],
                severity_multiplier=2.2
            )
            
            # ===== PHASE 3: 20 ADVANCED DISEASES =====
            
            # Parkinson's Disease
            self.profiles[DiseaseCondition.PARKINSONS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.PARKINSONS,
                beneficial_molecules={
                    'coq10': 3.0,  # Mitochondrial function (1200mg proven)
                    'vitamin_e': 2.8,  # Neuroprotection (tocopherols)
                    'omega_3': 2.5,  # Neuroinflammation
                    'uric_acid': 2.3,  # Antioxidant (coffee, cherries)
                    'green_tea_egcg': 2.0,  # Neuroprotection
                    'vitamin_d': 1.8,  # Deficiency common
                    'fiber': 2.5  # Constipation (95% of patients)
                },
                harmful_molecules={
                    'dairy': 2.5,  # Increases risk (Harvard study)
                    'saturated_fat': 2.3,  # Neurodegeneration
                    'iron': 2.5,  # Oxidative damage (avoid supplements)
                    'fried_foods': 2.0,  # AGEs damage
                    'pesticides': 3.0  # Rotenone, paraquat exposure
                },
                max_values={
                    'dairy': 100,  # g/day
                    'iron': 8,  # mg/day (no supplementation)
                    'saturated_fat': 15,  # g/day
                    'protein': 50  # g per meal (levodopa competes)
                },
                min_values={
                    'coq10': 1200,  # mg/day (Shults 2002 trial)
                    'vitamin_e': 400,  # IU/day
                    'omega_3': 2,  # g/day
                    'fiber': 30,  # g/day (constipation)
                    'water': 2500  # ml/day
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=2.5
            )
            
            # Multiple Sclerosis
            self.profiles[DiseaseCondition.MULTIPLE_SCLEROSIS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.MULTIPLE_SCLEROSIS,
                beneficial_molecules={
                    'vitamin_d': 3.0,  # CRITICAL (4000-10000 IU, latitude effect)
                    'omega_3': 2.8,  # Myelin protection
                    'vitamin_b12': 2.5,  # Myelin synthesis (1000mcg)
                    'biotin': 2.7,  # High-dose (MD1003, 300mg)
                    'alpha_lipoic_acid': 2.3,  # Neuroprotection (1200mg)
                    'antioxidants': 2.0,  # Oxidative stress
                    'vitamin_a': 1.8  # Immune modulation
                },
                harmful_molecules={
                    'saturated_fat': 2.8,  # Swank diet (<15g/day)
                    'trans_fat': 3.0,  # Myelin damage
                    'salt': 2.5,  # Worsens symptoms (Farez 2015)
                    'gluten': 2.0,  # Inflammatory (some)
                    'dairy': 1.8,  # Molecular mimicry theory
                    'sugar': 2.0  # Inflammation
                },
                max_values={
                    'saturated_fat': 15,  # g/day (Swank diet)
                    'salt': 1500,  # mg/day
                    'sugar': 25,  # g/day
                    'dairy': 100  # g/day
                },
                min_values={
                    'vitamin_d': 4000,  # IU/day (aim 40-60 ng/mL)
                    'omega_3': 3,  # g/day
                    'biotin': 300,  # mg/day (high-dose MD1003)
                    'vitamin_b12': 1000,  # mcg/day
                    'alpha_lipoic_acid': 1200  # mg/day
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.N_H],
                harmful_bonds=[],
                severity_multiplier=2.8
            )
            
            # Epilepsy
            self.profiles[DiseaseCondition.EPILEPSY] = DiseaseMolecularProfile(
                disease=DiseaseCondition.EPILEPSY,
                beneficial_molecules={
                    'magnesium': 2.8,  # Neuronal stability (400-600mg)
                    'vitamin_e': 2.5,  # Antioxidant neuroprotection
                    'vitamin_b6': 2.3,  # GABA synthesis (some forms)
                    'taurine': 2.0,  # Inhibitory neurotransmitter
                    'ketones': 3.0,  # Ketogenic diet (70-80% fat)
                    'mct_oil': 2.5,  # Ketone production
                    'omega_3': 2.0  # Neuronal stability
                },
                harmful_molecules={
                    'carbs': 2.8,  # If ketogenic (limit <50g/day)
                    'aspartame': 2.5,  # Lowers seizure threshold
                    'msg': 2.3,  # Excitotoxin
                    'caffeine': 2.0,  # May lower threshold
                    'alcohol': 3.0,  # Lowers threshold significantly
                    'ginkgo': 2.5  # Can trigger seizures
                },
                max_values={
                    'carbs': 50,  # g/day (if ketogenic)
                    'protein': 70,  # g/day (ketogenic limits)
                    'aspartame': 0,  # mg/day
                    'msg': 0,  # mg/day
                    'caffeine': 100,  # mg/day
                    'alcohol': 0  # g/day
                },
                min_values={
                    'fat': 150,  # g/day (if ketogenic 70-80%)
                    'mct_oil': 30,  # g/day (ketone boost)
                    'magnesium': 400,  # mg/day
                    'vitamin_e': 400,  # IU/day
                    'water': 2500  # ml/day
                },
                beneficial_bonds=[ChemicalBondType.C_H],  # Fats for ketones
                harmful_bonds=[],
                severity_multiplier=2.5
            )
            
            # Lupus (SLE)
            self.profiles[DiseaseCondition.LUPUS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.LUPUS,
                beneficial_molecules={
                    'omega_3': 2.8,  # Anti-inflammatory (3-4g EPA/DHA)
                    'vitamin_d': 2.7,  # Immune modulation (80% deficient)
                    'calcium': 2.5,  # Bone health (steroids deplete)
                    'vitamin_e': 2.3,  # Antioxidant
                    'curcumin': 2.0,  # NF-kB inhibition
                    'dhea': 1.8,  # Hormone modulation (200mg)
                    'probiotics': 2.0  # Gut immune tolerance
                },
                harmful_molecules={
                    'alfalfa': 3.0,  # L-canavanine triggers flares
                    'echinacea': 2.5,  # Immune stimulation (avoid)
                    'salt': 2.3,  # Worsens inflammation
                    'trans_fat': 2.5,  # Inflammation
                    'sugar': 2.0,  # Inflammation
                    'alcohol': 2.2  # Liver, medication interactions
                },
                max_values={
                    'alfalfa': 0,  # g/day (AVOID completely)
                    'salt': 1500,  # mg/day
                    'sugar': 20,  # g/day
                    'alcohol': 0  # g/day (medications)
                },
                min_values={
                    'omega_3': 3,  # g/day
                    'vitamin_d': 2000,  # IU/day (aim 40-60 ng/mL)
                    'calcium': 1200,  # mg/day (steroid bone loss)
                    'vitamin_e': 400,  # IU/day
                    'protein': 80  # g/day (preserve muscle)
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=2.5
            )
            
            # Hashimoto's Thyroiditis
            self.profiles[DiseaseCondition.HASHIMOTOS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.HASHIMOTOS,
                beneficial_molecules={
                    'selenium': 3.0,  # TPO antibody reduction (200mcg)
                    'iodine': 2.0,  # Moderate (150-300mcg, not excess)
                    'vitamin_d': 2.5,  # Autoimmune modulation
                    'iron': 2.3,  # Thyroid peroxidase cofactor
                    'zinc': 2.0,  # T4 to T3 conversion
                    'tyrosine': 1.8,  # Thyroid hormone precursor
                    'omega_3': 2.0  # Anti-inflammatory
                },
                harmful_molecules={
                    'goitrogens': 2.5,  # If raw (cruciferous, soy)
                    'gluten': 2.8,  # Molecular mimicry with TPO
                    'soy': 2.3,  # Goitrogenic, isoflavones
                    'millet': 2.0,  # Goitrogenic
                    'excess_iodine': 2.7,  # >1000mcg worsens (iodine paradox)
                    'fluoride': 2.0  # Thyroid interference
                },
                max_values={
                    'iodine': 500,  # mcg/day (avoid excess)
                    'soy': 50,  # g/day
                    'raw_cruciferous': 100,  # g/day (cooked OK)
                    'fluoride': 1  # mg/day
                },
                min_values={
                    'selenium': 200,  # mcg/day (Gartner 2002)
                    'iodine': 150,  # mcg/day (adequate, not excess)
                    'vitamin_d': 2000,  # IU/day
                    'iron': 18,  # mg/day (if deficient)
                    'zinc': 15  # mg/day
                },
                beneficial_bonds=[ChemicalBondType.N_H, ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=2.3
            )
            
            # Crohn's Disease (specific IBD type)
            self.profiles[DiseaseCondition.CROHNS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.CROHNS,
                beneficial_molecules={
                    'omega_3': 2.8,  # Anti-inflammatory (4-6g EPA/DHA)
                    'glutamine': 2.5,  # Intestinal healing (0.5g/kg)
                    'curcumin': 2.3,  # Remission maintenance
                    'zinc': 2.5,  # Often deficient, healing
                    'vitamin_d': 2.3,  # Immune, bone health
                    'iron': 2.7,  # Anemia common (IV if severe)
                    'soluble_fiber': 2.0,  # Remission phase only
                },
                harmful_molecules={
                    'insoluble_fiber': 2.8,  # Flare phase (mechanical)
                    'lactose': 2.5,  # Intolerance common
                    'fructose': 2.3,  # Malabsorption
                    'sugar_alcohols': 2.5,  # Sorbitol, xylitol
                    'emulsifiers': 2.7,  # Carrageenan, polysorbate
                    'red_meat': 2.0,  # Increases risk
                },
                max_values={
                    'insoluble_fiber': 5,  # g/day (flare)
                    'lactose': 0,  # g/day (if intolerant)
                    'fructose': 10,  # g/day
                    'red_meat': 50  # g/day
                },
                min_values={
                    'omega_3': 4,  # g/day (therapeutic)
                    'glutamine': 20,  # g/day (0.5g/kg for 80kg)
                    'zinc': 30,  # mg/day (healing)
                    'vitamin_d': 2000,  # IU/day
                    'calories': 2500,  # Often malnourished
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.N_H],
                harmful_bonds=[],
                severity_multiplier=2.5
            )
            
            # Ulcerative Colitis (specific IBD type)
            self.profiles[DiseaseCondition.ULCERATIVE_COLITIS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.ULCERATIVE_COLITIS,
                beneficial_molecules={
                    'omega_3': 2.8,  # Anti-inflammatory (4g EPA/DHA)
                    'butyrate': 3.0,  # Colonocyte fuel (enemas, fiber)
                    'curcumin': 2.5,  # Remission (3g/day proven)
                    'probiotics': 2.3,  # VSL#3 strain (450 billion)
                    'vitamin_d': 2.3,  # Immune modulation
                    'zinc': 2.5,  # Healing
                    'folate': 2.0  # Sulfasalazine depletes
                },
                harmful_molecules={
                    'sulfites': 2.8,  # Triggers (wine, dried fruit)
                    'carrageenan': 2.7,  # Emulsifier, inflammation
                    'red_meat': 2.3,  # Heme iron oxidative
                    'alcohol': 2.5,  # Flare trigger
                    'caffeine': 2.0,  # Diarrhea (for some)
                    'dairy': 2.0  # Intolerance common
                },
                max_values={
                    'sulfites': 10,  # ppm
                    'red_meat': 50,  # g/day
                    'alcohol': 0,  # g/day (flares)
                    'caffeine': 100  # mg/day
                },
                min_values={
                    'omega_3': 4,  # g/day (therapeutic)
                    'curcumin': 3000,  # mg/day (Hanai 2006)
                    'probiotics': 450_000_000_000,  # CFU/day (VSL#3)
                    'butyrate': 4,  # g/day (fiber produces)
                    'vitamin_d': 2000  # IU/day
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=2.5
            )
            
            # Rosacea
            self.profiles[DiseaseCondition.ROSACEA] = DiseaseMolecularProfile(
                disease=DiseaseCondition.ROSACEA,
                beneficial_molecules={
                    'omega_3': 2.5,  # Anti-inflammatory skin
                    'azelaic_acid': 2.3,  # Topical/dietary (grains)
                    'niacinamide': 2.0,  # B3, barrier function
                    'zinc': 2.0,  # Anti-inflammatory
                    'probiotics': 2.3,  # Gut-skin axis (SIBO link)
                    'green_tea': 2.0,  # EGCG anti-inflammatory
                    'antioxidants': 1.8  # Reduce redness
                },
                harmful_molecules={
                    'spicy_foods': 2.8,  # Capsaicin vasodilation
                    'hot_drinks': 2.5,  # Temperature trigger
                    'alcohol': 3.0,  # WORST trigger (vasodilation)
                    'histamine': 2.7,  # Aged cheese, wine, fermented
                    'cinnamaldehyde': 2.3,  # Cinnamon, tomatoes
                    'niacin': 2.0  # Flushing (vs niacinamide OK)
                },
                max_values={
                    'alcohol': 0,  # g/day (major trigger)
                    'spicy_foods': 0,  # g/day (capsaicin)
                    'hot_beverages': 0,  # servings/day
                    'histamine_foods': 50  # g/day
                },
                min_values={
                    'omega_3': 2,  # g/day
                    'niacinamide': 500,  # mg/day (NOT niacin)
                    'zinc': 15,  # mg/day
                    'probiotics': 10_000_000_000,  # CFU/day
                    'water': 2500  # ml/day
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=1.8
            )
            
            # Psoriatic Arthritis
            self.profiles[DiseaseCondition.PSORIATIC_ARTHRITIS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.PSORIATIC_ARTHRITIS,
                beneficial_molecules={
                    'omega_3': 2.8,  # Joint + skin inflammation
                    'vitamin_d': 2.7,  # Immune, joint, skin (4000+ IU)
                    'curcumin': 2.5,  # TNF-alpha inhibition
                    'glucosamine': 2.3,  # Cartilage (1500mg)
                    'chondroitin': 2.0,  # With glucosamine
                    'msm': 1.8,  # Sulfur, anti-inflammatory
                    'antioxidants': 2.0  # Oxidative stress
                },
                harmful_molecules={
                    'omega_6': 2.5,  # Pro-inflammatory ratio
                    'nightshades': 2.0,  # Some sensitive (trial)
                    'gluten': 2.3,  # Inflammatory (some)
                    'sugar': 2.3,  # Inflammation, AGEs
                    'alcohol': 2.5,  # Liver (methotrexate), flares
                    'red_meat': 2.0  # Arachidonic acid
                },
                max_values={
                    'omega_6': 12,  # g/day (ratio <4:1)
                    'sugar': 25,  # g/day
                    'alcohol': 0,  # g/day (if on methotrexate)
                    'red_meat': 100  # g/day
                },
                min_values={
                    'omega_3': 3,  # g/day
                    'vitamin_d': 4000,  # IU/day
                    'glucosamine': 1500,  # mg/day
                    'chondroitin': 1200,  # mg/day
                    'curcumin': 1000,  # mg/day
                    'protein': 100  # g/day (muscle preservation)
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=2.3
            )
            
            # Endometriosis
            self.profiles[DiseaseCondition.ENDOMETRIOSIS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.ENDOMETRIOSIS,
                beneficial_molecules={
                    'omega_3': 2.8,  # Prostaglandin balance
                    'fiber': 2.7,  # Estrogen elimination (35g+)
                    'cruciferous': 2.5,  # DIM, estrogen metabolism
                    'antioxidants': 2.3,  # Oxidative stress
                    'vitamin_d': 2.0,  # Anti-inflammatory
                    'magnesium': 2.3,  # Muscle relaxation, pain
                    'curcumin': 2.0  # Anti-inflammatory
                },
                harmful_molecules={
                    'red_meat': 2.8,  # Increases risk 80-100%
                    'trans_fat': 2.7,  # Inflammation, estrogen
                    'alcohol': 2.5,  # Estrogen metabolism
                    'caffeine': 2.3,  # Estrogen levels (>300mg)
                    'dioxins': 3.0,  # Endocrine disruptor (avoid)
                    'dairy': 2.0  # Inflammatory (some)
                },
                max_values={
                    'red_meat': 50,  # g/day
                    'trans_fat': 0,  # g/day
                    'alcohol': 10,  # g/day
                    'caffeine': 200  # mg/day
                },
                min_values={
                    'omega_3': 3,  # g/day
                    'fiber': 35,  # g/day (estrogen elimination)
                    'cruciferous': 3,  # servings/day
                    'vitamin_d': 2000,  # IU/day
                    'magnesium': 400  # mg/day (pain)
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=2.2
            )
            
            # Interstitial Cystitis (IC/BPS)
            self.profiles[DiseaseCondition.INTERSTITIAL_CYSTITIS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.INTERSTITIAL_CYSTITIS,
                beneficial_molecules={
                    'aloe_vera': 2.5,  # Bladder coating (oral)
                    'omega_3': 2.0,  # Anti-inflammatory
                    'quercetin': 2.3,  # Mast cell stabilizer (500mg)
                    'marshmallow_root': 2.0,  # Mucilage, soothing
                    'water': 2.5,  # Dilute urine (2-3L)
                    'calcium_citrate': 1.8  # Alkalinize urine
                },
                harmful_molecules={
                    'acidic_foods': 2.8,  # Citrus, tomatoes, vinegar
                    'caffeine': 3.0,  # WORST bladder irritant
                    'alcohol': 2.8,  # Irritant
                    'spicy_foods': 2.7,  # Capsaicin irritant
                    'artificial_sweeteners': 2.5,  # Aspartame, saccharin
                    'potassium': 2.0,  # Citrate OK, chloride not
                },
                max_values={
                    'caffeine': 0,  # mg/day (major trigger)
                    'alcohol': 0,  # g/day
                    'acidic_foods': 100,  # g/day
                    'spicy_foods': 0,  # g/day
                    'artificial_sweeteners': 0  # mg/day
                },
                min_values={
                    'water': 2500,  # ml/day (dilute urine)
                    'aloe_vera': 200,  # mg/day
                    'quercetin': 500,  # mg/day
                    'omega_3': 2  # g/day
                },
                beneficial_bonds=[ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=2.0
            )
            
            # Restless Leg Syndrome
            self.profiles[DiseaseCondition.RESTLESS_LEG] = DiseaseMolecularProfile(
                disease=DiseaseCondition.RESTLESS_LEG,
                beneficial_molecules={
                    'iron': 3.0,  # CRITICAL (ferritin <75 triggers RLS)
                    'magnesium': 2.8,  # Muscle relaxation (400-800mg)
                    'folate': 2.5,  # Deficiency linked (800mcg+)
                    'vitamin_d': 2.3,  # Deficiency common
                    'vitamin_e': 2.0,  # Circulation
                    'b_vitamins': 1.8  # Nerve function
                },
                harmful_molecules={
                    'caffeine': 2.8,  # Worsens symptoms
                    'alcohol': 2.5,  # Sleep disruption
                    'antihistamines': 2.3,  # Diphenhydramine worsens
                    'sugar': 2.0,  # Blood sugar spikes
                    'tyramine': 1.8  # Dopamine metabolism
                },
                max_values={
                    'caffeine': 100,  # mg/day (avoid PM)
                    'alcohol': 10,  # g/day
                    'sugar': 30  # g/day
                },
                min_values={
                    'iron': 18,  # mg/day (aim ferritin >75 ng/mL)
                    'magnesium': 400,  # mg/day
                    'folate': 800,  # mcg/day
                    'vitamin_d': 2000,  # IU/day
                    'protein': 80  # g/day (tyrosine for dopamine)
                },
                beneficial_bonds=[ChemicalBondType.N_H, ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=1.8
            )
            
            # Peripheral Neuropathy
            self.profiles[DiseaseCondition.PERIPHERAL_NEUROPATHY] = DiseaseMolecularProfile(
                disease=DiseaseCondition.PERIPHERAL_NEUROPATHY,
                beneficial_molecules={
                    'vitamin_b12': 3.0,  # Myelin (1000-2000mcg)
                    'alpha_lipoic_acid': 2.8,  # Nerve regeneration (600mg)
                    'vitamin_b6': 2.5,  # Nerve function (but <200mg)
                    'vitamin_b1': 2.7,  # Thiamine, benfotiamine
                    'acetyl_l_carnitine': 2.3,  # Nerve regeneration
                    'omega_3': 2.0,  # Neuroinflammation
                    'magnesium': 2.0  # Nerve conduction
                },
                harmful_molecules={
                    'alcohol': 3.0,  # Toxic to nerves (major cause)
                    'vitamin_b6': 2.5,  # >200mg CAUSES neuropathy
                    'sugar': 2.5,  # If diabetic (glucose control)
                    'trans_fat': 2.0,  # Inflammation
                    'msg': 1.8  # Neurotoxic
                },
                max_values={
                    'alcohol': 0,  # g/day (neurotoxic)
                    'vitamin_b6': 100,  # mg/day (toxicity >200mg)
                    'sugar': 30  # g/day (if diabetic)
                },
                min_values={
                    'vitamin_b12': 1000,  # mcg/day (sublingual)
                    'alpha_lipoic_acid': 600,  # mg/day (Ziegler 2004)
                    'vitamin_b1': 100,  # mg/day (benfotiamine)
                    'acetyl_l_carnitine': 1500,  # mg/day
                    'omega_3': 2  # g/day
                },
                beneficial_bonds=[ChemicalBondType.N_H, ChemicalBondType.DOUBLE_BOND],
                harmful_bonds=[],
                severity_multiplier=2.2
            )
            
            # Raynaud's Phenomenon
            self.profiles[DiseaseCondition.RAYNAUDS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.RAYNAUDS,
                beneficial_molecules={
                    'niacin': 2.5,  # Vasodilation (flushing form)
                    'l_arginine': 2.8,  # Nitric oxide (3-6g)
                    'ginkgo_biloba': 2.3,  # Circulation (240mg)
                    'omega_3': 2.5,  # Vasodilation
                    'vitamin_e': 2.0,  # Circulation
                    'magnesium': 2.0,  # Vasodilation
                    'ginger': 1.8  # Warming, circulation
                },
                harmful_molecules={
                    'caffeine': 2.8,  # Vasoconstriction
                    'nicotine': 3.0,  # Vasoconstriction (smoking)
                    'pseudoephedrine': 2.5,  # Decongestants
                    'tyramine': 2.0,  # Vasoconstriction
                    'cold_foods': 1.8  # Temperature trigger
                },
                max_values={
                    'caffeine': 100,  # mg/day
                    'nicotine': 0,  # mg/day (quit smoking)
                    'cold_beverages': 500  # ml/day
                },
                min_values={
                    'l_arginine': 3000,  # mg/day
                    'ginkgo': 240,  # mg/day
                    'omega_3': 2,  # g/day
                    'niacin': 100,  # mg/day (flushing form)
                    'warm_fluids': 2000  # ml/day
                },
                beneficial_bonds=[ChemicalBondType.N_H, ChemicalBondType.DOUBLE_BOND],
                harmful_bonds=[],
                severity_multiplier=1.7
            )
            
            # Sjogren's Syndrome
            self.profiles[DiseaseCondition.SJOGRENS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.SJOGRENS,
                beneficial_molecules={
                    'omega_3': 2.8,  # Tear/saliva production
                    'omega_6_gla': 2.5,  # GLA (borage, EPO) for dryness
                    'vitamin_a': 2.3,  # Mucous membranes (10000 IU)
                    'vitamin_d': 2.0,  # Autoimmune
                    'water': 3.0,  # Hydration CRITICAL (3-4L)
                    'electrolytes': 2.0,  # With water
                    'turmeric': 1.8  # Anti-inflammatory
                },
                harmful_molecules={
                    'caffeine': 2.5,  # Dehydrating
                    'alcohol': 2.8,  # Dehydrating, drying
                    'antihistamines': 2.3,  # Drying effect
                    'salt': 2.0,  # Dehydrating
                    'sugar': 2.0  # Inflammation
                },
                max_values={
                    'caffeine': 100,  # mg/day
                    'alcohol': 0,  # g/day (very drying)
                    'salt': 2000  # mg/day
                },
                min_values={
                    'water': 3000,  # ml/day (combat dryness)
                    'omega_3': 3,  # g/day
                    'omega_6_gla': 300,  # mg/day (borage oil)
                    'vitamin_a': 10000,  # IU/day
                    'vitamin_d': 2000  # IU/day
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=2.0
            )
            
            # Scleroderma
            self.profiles[DiseaseCondition.SCLERODERMA] = DiseaseMolecularProfile(
                disease=DiseaseCondition.SCLERODERMA,
                beneficial_molecules={
                    'vitamin_d': 2.8,  # Immune, often deficient
                    'omega_3': 2.5,  # Anti-inflammatory, GERD
                    'vitamin_e': 2.3,  # Antioxidant, skin
                    'l_carnitine': 2.0,  # Muscle weakness
                    'coq10': 2.0,  # Energy, cardiac
                    'antioxidants': 2.0,  # Oxidative stress
                    'probiotics': 1.8  # GI involvement
                },
                harmful_molecules={
                    'l_tryptophan': 3.0,  # Eosinophilia-myalgia link
                    'fat': 2.5,  # GERD (common complication)
                    'caffeine': 2.0,  # GERD, Raynaud's
                    'alcohol': 2.3,  # GERD, liver
                    'large_meals': 2.5  # GERD, motility
                },
                max_values={
                    'l_tryptophan': 0,  # mg/day (avoid supplements)
                    'fat': 40,  # g per meal (GERD)
                    'meal_size': 300,  # g (small frequent)
                    'caffeine': 100,  # mg/day
                    'alcohol': 0  # g/day
                },
                min_values={
                    'vitamin_d': 2000,  # IU/day
                    'omega_3': 3,  # g/day
                    'vitamin_e': 400,  # IU/day
                    'water': 2500,  # ml/day
                    'calories': 2500  # Often malnourished
                },
                beneficial_bonds=[ChemicalBondType.O_H, ChemicalBondType.DOUBLE_BOND],
                harmful_bonds=[],
                severity_multiplier=2.5
            )
            
            # Addison's Disease
            self.profiles[DiseaseCondition.ADDISONS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.ADDISONS,
                beneficial_molecules={
                    'salt': 3.0,  # CRITICAL (sodium loss, 5-10g/day)
                    'water': 2.8,  # Dehydration risk (3-4L)
                    'potassium': 2.0,  # Monitor (can be high)
                    'vitamin_d': 2.3,  # Autoimmune, bone
                    'vitamin_c': 2.5,  # Adrenal support (1000mg)
                    'b_vitamins': 2.0,  # Energy, stress
                    'protein': 2.0  # Muscle preservation
                },
                harmful_molecules={
                    'licorice': 3.0,  # Mimics aldosterone (dangerous)
                    'grapefruit': 2.5,  # Drug interactions
                    'potassium': 2.0,  # Can accumulate (monitor)
                    'alcohol': 2.3,  # Hypoglycemia risk
                    'caffeine': 2.0  # Stress response
                },
                max_values={
                    'licorice': 0,  # g/day (AVOID - glycyrrhizin)
                    'grapefruit': 0,  # g/day (med interactions)
                    'potassium': 3500,  # mg/day (can accumulate)
                    'alcohol': 0  # g/day (hypoglycemia)
                },
                min_values={
                    'salt': 5000,  # mg/day (3-10g depending on losses)
                    'water': 3000,  # ml/day
                    'vitamin_c': 1000,  # mg/day
                    'vitamin_d': 2000,  # IU/day
                    'calories': 2200  # Prevent hypoglycemia
                },
                beneficial_bonds=[ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=2.5
            )
            
            # Cushing's Syndrome
            self.profiles[DiseaseCondition.CUSHINGS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.CUSHINGS,
                beneficial_molecules={
                    'calcium': 2.8,  # Bone loss (1200-1500mg)
                    'vitamin_d': 2.7,  # Bone health
                    'protein': 2.5,  # Muscle wasting (1.2-1.5g/kg)
                    'potassium': 2.5,  # Often low (3500-4700mg)
                    'fiber': 2.3,  # Blood sugar control
                    'chromium': 2.0,  # Insulin sensitivity
                    'antioxidants': 2.0  # Oxidative stress
                },
                harmful_molecules={
                    'salt': 2.8,  # Hypertension, edema
                    'sugar': 2.8,  # Diabetes risk (cortisol)
                    'saturated_fat': 2.5,  # Weight gain, CV risk
                    'refined_carbs': 2.5,  # Blood sugar
                    'alcohol': 2.3,  # Bone, blood sugar
                    'licorice': 2.5  # Worsens symptoms
                },
                max_values={
                    'salt': 1500,  # mg/day
                    'sugar': 20,  # g/day
                    'saturated_fat': 15,  # g/day
                    'alcohol': 0,  # g/day
                    'calories': 1800  # Weight management
                },
                min_values={
                    'calcium': 1500,  # mg/day (bone protection)
                    'vitamin_d': 2000,  # IU/day
                    'protein': 100,  # g/day (muscle preservation)
                    'potassium': 3500,  # mg/day
                    'fiber': 30  # g/day
                },
                beneficial_bonds=[ChemicalBondType.N_H, ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=2.3
            )
            
            # Hemochromatosis (Iron Overload)
            self.profiles[DiseaseCondition.HEMOCHROMATOSIS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.HEMOCHROMATOSIS,
                beneficial_molecules={
                    'tea': 2.8,  # Tannins block iron (black, green)
                    'calcium': 2.5,  # Competes with iron absorption
                    'phytates': 2.3,  # Block iron (grains, legumes)
                    'fiber': 2.0,  # Binds iron
                    'vitamin_e': 2.0,  # Antioxidant (iron oxidative)
                    'dairy': 2.5  # Calcium source, blocks iron
                },
                harmful_molecules={
                    'iron': 3.0,  # AVOID supplements completely
                    'vitamin_c': 2.8,  # Enhances iron (limit with meals)
                    'alcohol': 3.0,  # Liver damage (iron + EtOH)
                    'raw_shellfish': 2.5,  # Vibrio risk (iron overload)
                    'red_meat': 2.8,  # Heme iron (most absorbable)
                    'organ_meats': 3.0  # Highest iron content
                },
                max_values={
                    'iron': 0,  # mg/day supplemental
                    'vitamin_c': 100,  # mg per meal (limit enhancer)
                    'red_meat': 50,  # g/day
                    'organ_meats': 0,  # g/day
                    'alcohol': 0,  # g/day (liver toxicity)
                    'raw_shellfish': 0  # g/day (vibrio risk)
                },
                min_values={
                    'tea': 500,  # ml/day (with meals)
                    'calcium': 1200,  # mg/day (blocks iron)
                    'phytates': 1000,  # mg/day (grains, legumes)
                    'fiber': 30  # g/day
                },
                beneficial_bonds=[ChemicalBondType.O_H],  # Polyphenols block iron
                harmful_bonds=[],
                severity_multiplier=2.5
            )
            
            # Wilson's Disease (Copper Overload)
            self.profiles[DiseaseCondition.WILSONS] = DiseaseMolecularProfile(
                disease=DiseaseCondition.WILSONS,
                beneficial_molecules={
                    'zinc': 3.0,  # Blocks copper absorption (150mg)
                    'molybdenum': 2.5,  # Copper excretion
                    'vitamin_e': 2.3,  # Antioxidant (copper oxidative)
                    'vitamin_c': 2.0,  # Antioxidant
                    'manganese': 2.0,  # Competes with copper
                    'fiber': 1.8  # Binds copper
                },
                harmful_molecules={
                    'copper': 3.0,  # AVOID completely (<1mg/day)
                    'organ_meats': 3.0,  # Highest copper (liver)
                    'shellfish': 3.0,  # High copper (oysters, crab)
                    'mushrooms': 2.8,  # High copper
                    'nuts': 2.8,  # High copper (cashews, brazil)
                    'chocolate': 2.7,  # High copper
                    'legumes': 2.5,  # Moderate copper
                    'whole_grains': 2.0  # Moderate copper
                },
                max_values={
                    'copper': 1,  # mg/day (strict low-copper diet)
                    'organ_meats': 0,  # g/day
                    'shellfish': 0,  # g/day
                    'nuts': 0,  # g/day
                    'chocolate': 0,  # g/day
                    'mushrooms': 0,  # g/day
                    'legumes': 50  # g/day
                },
                min_values={
                    'zinc': 150,  # mg/day (therapeutic, blocks copper)
                    'molybdenum': 150,  # mcg/day
                    'vitamin_e': 400,  # IU/day
                    'protein': 80  # g/day (low-copper sources)
                },
                beneficial_bonds=[],
                harmful_bonds=[],
                severity_multiplier=2.8
            )
        
        # Autoimmune conditions
        self.profiles[DiseaseCondition.AUTOIMMUNE] = DiseaseMolecularProfile(
            disease=DiseaseCondition.AUTOIMMUNE,
                beneficial_molecules={
                    'omega_3': 2.5,  # Immune modulation
                    'vitamin_d': 2.3,  # T-reg cells
                    'curcumin': 2.0,  # NF-kB inhibition
                    'polyphenols': 1.8,  # Anti-inflammatory
                    'glutamine': 1.6,  # Gut barrier
                    'probiotics': 1.7,  # Immune tolerance
                    'vitamin_a': 1.5  # Immune regulation
                },
                harmful_molecules={
                    'gluten': 2.5,  # Leaky gut, molecular mimicry
                    'lectins': 2.0,  # Gut permeability
                    'nightshades': 1.5,  # Inflammatory (some people)
                    'refined_sugar': 2.0,  # Inflammation
                    'trans_fat': 2.2,  # Inflammation
                    'alcohol': 1.8  # Gut barrier damage
                },
                max_values={
                    'gluten': 0,  # g/day (elimination trial)
                    'sugar': 20,  # g/day
                    'omega_6': 10,  # g/day (high omega-6:3 ratio bad)
                    'alcohol': 0  # g/day
                },
                min_values={
                    'omega_3': 3,  # g/day
                    'vitamin_d': 4000,  # IU/day (therapeutic)
                    'antioxidants': 10000,  # ORAC units
                    'glutamine': 5  # g/day
                },
                beneficial_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                harmful_bonds=[],
                severity_multiplier=2.0
            )
        
        logger.info(f"Initialized {len(self.profiles)} disease profiles")
    
    def get_profile(self, disease: 'DiseaseCondition') -> Optional[DiseaseMolecularProfile]:
        """Get molecular profile for a disease"""
        return self.profiles.get(disease)


# ============================================================================
# GOAL MOLECULAR PROFILES
# ============================================================================

@dataclass
class GoalMolecularProfile:
    """Molecular profile for a health goal"""
    goal: 'HealthGoal'
    
    # Molecules to maximize
    target_molecules: Dict[str, float] = field(default_factory=dict)  # {molecule: importance}
    
    # Optimal nutrient ranges
    optimal_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # {nutrient: (min, max)}
    
    # Bond priorities
    priority_bonds: List['ChemicalBondType'] = field(default_factory=list)
    
    # Performance metrics
    key_metrics: List[str] = field(default_factory=list)


class GoalProfileDatabase:
    """Database of goal-specific molecular profiles"""
    
    def __init__(self):
        self.profiles: Dict['HealthGoal', GoalMolecularProfile] = {}
        self._initialize_profiles()
    
    def _initialize_profiles(self):
        """Initialize goal profiles"""
        
        if PROFILER_AVAILABLE:
            # Energy / Fitness / Athletics
            self.profiles[HealthGoal.ENERGY] = GoalMolecularProfile(
                goal=HealthGoal.ENERGY,
                target_molecules={
                    'carbohydrates': 2.0,  # Primary fuel
                    'b_vitamins': 1.5,  # Energy metabolism
                    'iron': 1.4,  # Oxygen transport
                    'magnesium': 1.3,  # ATP production
                    'coq10': 1.2,  # Mitochondrial function
                    'caffeine': 1.1  # CNS stimulation (moderate)
                },
                optimal_ranges={
                    'carbohydrates': (45, 65),  # % of calories
                    'protein': (15, 25),  # % of calories
                    'fat': (20, 35),  # % of calories
                    'iron': (8, 18),  # mg/day
                    'b12': (2.4, 100)  # mcg/day
                },
                priority_bonds=[ChemicalBondType.C_H, ChemicalBondType.P_O],  # Carbs, ATP
                key_metrics=['total_calories', 'carb_timing', 'glycemic_index']
            )
            
            # Muscle Building / Weight Loss
            self.profiles[HealthGoal.MUSCLE] = GoalMolecularProfile(
                goal=HealthGoal.MUSCLE,
                target_molecules={
                    'protein': 2.5,  # Muscle protein synthesis
                    'leucine': 2.0,  # mTOR activation
                    'creatine': 1.5,  # Strength, power
                    'beta_alanine': 1.3,  # Buffering
                    'citrulline': 1.2,  # Blood flow
                    'vitamin_d': 1.4  # Testosterone, strength
                },
                optimal_ranges={
                    'protein': (1.6, 2.2),  # g/kg body weight
                    'leucine': (2.5, 3.0),  # g per meal
                    'calories': (500, 500),  # Surplus for growth
                    'carbohydrates': (3, 5)  # g/kg (training days)
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.N_C],  # Protein
                key_metrics=['protein_timing', 'amino_acid_profile', 'anabolic_window']
            )
            
            # Brain Health / Cognition
            self.profiles[HealthGoal.BRAIN] = GoalMolecularProfile(
                goal=HealthGoal.BRAIN,
                target_molecules={
                    'dha': 2.5,  # Neuronal membranes
                    'polyphenols': 2.3,  # Neuroprotection
                    'choline': 2.0,  # Acetylcholine
                    'b_vitamins': 1.8,  # Neurotransmitters
                    'lutein': 1.5,  # Visual-cognitive
                    'curcumin': 1.4,  # Neuroinflammation
                    'caffeine': 1.3  # Alertness
                },
                optimal_ranges={
                    'dha': (0.5, 2.0),  # g/day
                    'polyphenols': (500, 2000),  # mg/day
                    'choline': (400, 550),  # mg/day
                    'b12': (2.4, 1000),  # mcg/day
                    'folate': (400, 1000)  # mcg/day
                },
                priority_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H, ChemicalBondType.AROMATIC],
                key_metrics=['cognitive_enhancement', 'neuroprotection', 'memory_support']
            )
            
            # Heart / Cardiovascular Health
            self.profiles[HealthGoal.HEART] = GoalMolecularProfile(
                goal=HealthGoal.HEART,
                target_molecules={
                    'omega_3': 2.5,  # Anti-inflammatory
                    'fiber': 2.0,  # Cholesterol lowering
                    'nitric_oxide_precursors': 1.8,  # Vasodilation
                    'potassium': 1.6,  # BP regulation
                    'magnesium': 1.5,  # Rhythm
                    'coq10': 1.4,  # Heart energy
                    'plant_sterols': 1.3  # Cholesterol blocking
                },
                optimal_ranges={
                    'omega_3': (1.5, 3.0),  # g/day
                    'fiber': (30, 40),  # g/day
                    'saturated_fat': (0, 13),  # g/day
                    'sodium': (0, 1500),  # mg/day
                    'potassium': (3500, 4700)  # mg/day
                },
                priority_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                key_metrics=['cholesterol_impact', 'blood_pressure', 'endothelial_function']
            )
            
            # Gut Health
            self.profiles[HealthGoal.GUT] = GoalMolecularProfile(
                goal=HealthGoal.GUT,
                target_molecules={
                    'fiber': 2.5,  # Microbiome fuel
                    'prebiotics': 2.0,  # Feed good bacteria
                    'polyphenols': 1.8,  # Microbiome diversity
                    'resistant_starch': 1.6,  # Butyrate production
                    'glutamine': 1.4,  # Gut lining repair
                    'zinc': 1.3  # Tight junctions
                },
                optimal_ranges={
                    'fiber': (35, 50),  # g/day
                    'prebiotics': (5, 10),  # g/day
                    'polyphenols': (500, 1500),  # mg/day
                    'water': (2000, 3000)  # ml/day
                },
                priority_bonds=[ChemicalBondType.O_H],  # Polyphenols
                key_metrics=['microbiome_diversity', 'scfa_production', 'gut_barrier']
            )
            
            # Immunity
            self.profiles[HealthGoal.IMMUNITY] = GoalMolecularProfile(
                goal=HealthGoal.IMMUNITY,
                target_molecules={
                    'vitamin_c': 2.0,  # Immune cells
                    'vitamin_d': 2.0,  # Immune regulation
                    'zinc': 1.8,  # T-cell function
                    'selenium': 1.5,  # Antioxidant
                    'polyphenols': 1.6,  # Anti-inflammatory
                    'beta_glucans': 1.4,  # Macrophage activation
                    'quercetin': 1.3  # Antiviral
                },
                optimal_ranges={
                    'vitamin_c': (200, 2000),  # mg/day
                    'vitamin_d': (2000, 4000),  # IU/day
                    'zinc': (15, 40),  # mg/day
                    'selenium': (55, 200)  # mcg/day
                },
                priority_bonds=[ChemicalBondType.O_H, ChemicalBondType.AROMATIC],
                key_metrics=['immune_response', 'inflammation', 'antioxidant_capacity']
            )
            
            # Longevity / Anti-Aging
            self.profiles[HealthGoal.LONGEVITY] = GoalMolecularProfile(
                goal=HealthGoal.LONGEVITY,
                target_molecules={
                    'polyphenols': 2.5,  # Sirtuin activation
                    'omega_3': 2.0,  # Telomere protection
                    'resveratrol': 1.8,  # Sirtuin activator
                    'curcumin': 1.6,  # Anti-inflammatory
                    'spermidine': 1.5,  # Autophagy
                    'nad_precursors': 1.4,  # Cellular energy
                    'antioxidants': 2.0  # Oxidative stress
                },
                optimal_ranges={
                    'polyphenols': (1000, 3000),  # mg/day
                    'omega_3': (2, 4),  # g/day
                    'antioxidants': (10000, 20000),  # ORAC units
                    'calories': (-500, 0)  # Caloric restriction
                },
                priority_bonds=[ChemicalBondType.O_H, ChemicalBondType.AROMATIC, ChemicalBondType.DOUBLE_BOND],
                key_metrics=['oxidative_stress', 'inflammation', 'cellular_health']
            )
            
            # ===================================================================
            # PHASE 1: WEIGHT MANAGEMENT GOALS (3)
            # ===================================================================
            
            # Weight Loss / Fat Burning
            self.profiles[HealthGoal.WEIGHT_LOSS] = GoalMolecularProfile(
                goal=HealthGoal.WEIGHT_LOSS,
                target_molecules={
                    'protein': 2.8,  # Satiety, thermogenesis (30% TEF)
                    'fiber': 2.5,  # Volume, satiety
                    'water': 2.0,  # Satiety, metabolism
                    'caffeine': 1.8,  # Thermogenesis, lipolysis
                    'green_tea_catechins': 1.7,  # Fat oxidation
                    'capsaicin': 1.6,  # Thermogenesis
                    'cla': 1.4,  # Conjugated linoleic acid
                    'l_carnitine': 1.3  # Fat transport to mitochondria
                },
                optimal_ranges={
                    'protein': (1.6, 2.4),  # g/kg (preserve muscle)
                    'fiber': (35, 50),  # g/day (satiety)
                    'calories': (-500, -250),  # kcal/day (deficit)
                    'water': (3000, 4000),  # ml/day
                    'carbohydrates': (20, 40)  # % calories (low-carb)
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.O_H],  # Protein, fiber
                key_metrics=['caloric_deficit', 'satiety_index', 'thermogenic_effect', 'muscle_preservation']
            )
            
            # Weight Gain / Bulking
            self.profiles[HealthGoal.WEIGHT_GAIN] = GoalMolecularProfile(
                goal=HealthGoal.WEIGHT_GAIN,
                target_molecules={
                    'protein': 2.5,  # Muscle synthesis
                    'carbohydrates': 2.3,  # Fuel, insulin
                    'creatine': 2.0,  # Strength, cell volume
                    'leucine': 2.2,  # mTOR activation
                    'calories': 2.5,  # Surplus required
                    'healthy_fats': 1.8,  # Calorie dense
                    'beta_alanine': 1.4  # Training capacity
                },
                optimal_ranges={
                    'protein': (1.8, 2.5),  # g/kg
                    'calories': (300, 500),  # kcal/day (surplus)
                    'carbohydrates': (4, 7),  # g/kg
                    'fat': (1.0, 1.5),  # g/kg
                    'meals_per_day': (5, 6)  # Frequency
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.C_H],  # Protein, carbs
                key_metrics=['caloric_surplus', 'muscle_gain_rate', 'lean_mass_ratio']
            )
            
            # Body Recomposition (lose fat + gain muscle simultaneously)
            self.profiles[HealthGoal.BODY_RECOMP] = GoalMolecularProfile(
                goal=HealthGoal.BODY_RECOMP,
                target_molecules={
                    'protein': 3.0,  # HIGHEST priority
                    'leucine': 2.5,  # Muscle synthesis
                    'omega_3': 1.8,  # Insulin sensitivity
                    'fiber': 2.0,  # Satiety
                    'chromium': 1.5,  # Glucose partitioning
                    'caffeine': 1.6,  # Fat oxidation
                    'creatine': 1.7  # Strength, cell volume
                },
                optimal_ranges={
                    'protein': (2.2, 3.0),  # g/kg (very high)
                    'calories': (-100, 100),  # Maintenance or slight deficit
                    'carbohydrates': (2, 4),  # g/kg (moderate)
                    'leucine': (3, 5),  # g per meal
                    'training_frequency': (4, 6)  # days/week
                },
                priority_bonds=[ChemicalBondType.N_H],  # Protein dominant
                key_metrics=['protein_distribution', 'nutrient_timing', 'training_stimulus']
            )
            
            # ===================================================================
            # PHASE 1: ATHLETIC PERFORMANCE GOALS (5)
            # ===================================================================
            
            # Endurance / Cardio Performance
            self.profiles[HealthGoal.ENDURANCE] = GoalMolecularProfile(
                goal=HealthGoal.ENDURANCE,
                target_molecules={
                    'carbohydrates': 2.8,  # Primary fuel
                    'electrolytes': 2.5,  # Sodium, potassium
                    'iron': 2.0,  # Oxygen transport
                    'beetroot_nitrates': 2.3,  # VO2 max improvement
                    'beta_alanine': 1.8,  # Buffering
                    'caffeine': 1.9,  # Performance, fat oxidation
                    'bcaa': 1.5,  # Reduce central fatigue
                    'water': 2.2  # Hydration critical
                },
                optimal_ranges={
                    'carbohydrates': (5, 12),  # g/kg (intensity-dependent)
                    'sodium': (3000, 7000),  # mg/day (sweat losses)
                    'iron': (18, 30),  # mg/day (endurance athletes)
                    'beetroot_juice': (200, 500),  # ml/day (pre-workout)
                    'protein': (1.2, 1.6)  # g/kg (moderate)
                },
                priority_bonds=[ChemicalBondType.C_H, ChemicalBondType.P_O],  # Carbs, ATP
                key_metrics=['glycogen_stores', 'vo2_max', 'lactate_threshold', 'hydration_status']
            )
            
            # Strength / Power Performance
            self.profiles[HealthGoal.STRENGTH] = GoalMolecularProfile(
                goal=HealthGoal.STRENGTH,
                target_molecules={
                    'creatine': 3.0,  # ATP regeneration
                    'protein': 2.5,  # Muscle protein synthesis
                    'leucine': 2.5,  # mTOR pathway
                    'carbohydrates': 2.0,  # Glycogen for power
                    'beta_alanine': 1.7,  # Buffering
                    'caffeine': 1.8,  # CNS activation
                    'citrulline': 1.6,  # Pump, blood flow
                    'vitamin_d': 1.5  # Muscle contraction
                },
                optimal_ranges={
                    'creatine': (3, 5),  # g/day (loading 20g)
                    'protein': (1.6, 2.2),  # g/kg
                    'carbohydrates': (3, 6),  # g/kg
                    'leucine': (2.5, 3),  # g per meal
                    'rest_days': (1, 2)  # per week
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.P_O],  # Protein, phosphocreatine
                key_metrics=['1rm_improvement', 'power_output', 'recovery_rate']
            )
            
            # Speed / Agility / Explosive Performance
            self.profiles[HealthGoal.SPEED] = GoalMolecularProfile(
                goal=HealthGoal.SPEED,
                target_molecules={
                    'creatine': 2.8,  # Phosphocreatine system
                    'carbohydrates': 2.5,  # Quick fuel
                    'caffeine': 2.3,  # Reaction time, CNS
                    'beta_alanine': 1.9,  # Repeated sprints
                    'sodium_bicarbonate': 1.7,  # Buffering
                    'beetroot_nitrates': 1.8,  # Blood flow
                    'tyrosine': 1.5  # Mental focus
                },
                optimal_ranges={
                    'creatine': (5, 5),  # g/day (maintenance)
                    'carbohydrates': (4, 7),  # g/kg
                    'caffeine': (3, 6),  # mg/kg pre-training
                    'protein': (1.4, 1.8),  # g/kg
                    'power_to_weight': (1, 1)  # Ratio critical
                },
                priority_bonds=[ChemicalBondType.P_O, ChemicalBondType.C_H],  # ATP, glucose
                key_metrics=['sprint_speed', 'reaction_time', 'power_to_weight', 'neuromuscular_efficiency']
            )
            
            # Flexibility / Mobility
            self.profiles[HealthGoal.FLEXIBILITY] = GoalMolecularProfile(
                goal=HealthGoal.FLEXIBILITY,
                target_molecules={
                    'collagen': 2.5,  # Connective tissue
                    'vitamin_c': 2.3,  # Collagen synthesis
                    'omega_3': 2.0,  # Joint health, inflammation
                    'magnesium': 1.8,  # Muscle relaxation
                    'glucosamine': 1.7,  # Joint health
                    'hyaluronic_acid': 1.6,  # Joint lubrication
                    'water': 2.0  # Tissue hydration
                },
                optimal_ranges={
                    'collagen': (10, 20),  # g/day
                    'vitamin_c': (200, 1000),  # mg/day
                    'omega_3': (2, 3),  # g/day
                    'magnesium': (400, 500),  # mg/day
                    'water': (2500, 3500)  # ml/day
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.O_H],  # Collagen
                key_metrics=['range_of_motion', 'tissue_hydration', 'collagen_synthesis']
            )
            
            # Athletic Recovery / Post-Workout
            self.profiles[HealthGoal.ATHLETIC_RECOVERY] = GoalMolecularProfile(
                goal=HealthGoal.ATHLETIC_RECOVERY,
                target_molecules={
                    'protein': 2.8,  # Muscle repair
                    'carbohydrates': 2.5,  # Glycogen replenishment
                    'bcaa': 2.0,  # Leucine, isoleucine, valine
                    'tart_cherry': 2.2,  # Anti-inflammatory
                    'omega_3': 2.0,  # Inflammation resolution
                    'glutamine': 1.7,  # Immune support
                    'electrolytes': 2.3,  # Rehydration
                    'antioxidants': 1.8  # Oxidative stress
                },
                optimal_ranges={
                    'protein': (0.25, 0.4),  # g/kg per meal (post-workout)
                    'carbohydrates': (1.0, 1.5),  # g/kg (post-workout)
                    'ratio_carb_protein': (3, 4),  # Optimal 3:1 or 4:1
                    'timing_post_workout': (0, 2),  # hours (anabolic window)
                    'sleep_hours': (8, 10)  # hours
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.C_H],  # Protein, carbs
                key_metrics=['muscle_damage_markers', 'glycogen_resynthesis', 'inflammation_markers']
            )
            
            # ===================================================================
            # PHASE 1: SPECIFIC HEALTH GOALS (7)
            # ===================================================================
            
            # Skin Health / Anti-Aging / Beauty
            self.profiles[HealthGoal.SKIN_HEALTH] = GoalMolecularProfile(
                goal=HealthGoal.SKIN_HEALTH,
                target_molecules={
                    'collagen': 2.8,  # Skin structure
                    'vitamin_c': 2.5,  # Collagen synthesis
                    'vitamin_e': 2.0,  # Antioxidant
                    'vitamin_a': 2.3,  # Retinol, cell turnover
                    'omega_3': 1.9,  # Skin barrier
                    'hyaluronic_acid': 2.0,  # Hydration
                    'astaxanthin': 1.8,  # UV protection
                    'polyphenols': 1.7,  # Antioxidant
                    'zinc': 1.6,  # Wound healing
                    'water': 2.2  # Hydration
                },
                optimal_ranges={
                    'collagen': (10, 20),  # g/day
                    'vitamin_c': (500, 2000),  # mg/day
                    'vitamin_a': (700, 3000),  # mcg RAE/day
                    'omega_3': (2, 3),  # g/day
                    'water': (2500, 4000)  # ml/day
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.O_H],  # Collagen, antioxidants
                key_metrics=['collagen_density', 'elasticity', 'hydration', 'wrinkle_depth']
            )
            
            # Hair & Nails Growth / Health
            self.profiles[HealthGoal.HAIR_NAILS] = GoalMolecularProfile(
                goal=HealthGoal.HAIR_NAILS,
                target_molecules={
                    'biotin': 2.8,  # Keratin synthesis
                    'protein': 2.5,  # Hair/nail structure
                    'iron': 2.3,  # Hair follicles
                    'zinc': 2.0,  # Growth, repair
                    'vitamin_e': 1.8,  # Antioxidant
                    'silica': 1.9,  # Collagen formation
                    'omega_3': 1.7,  # Scalp health
                    'vitamin_d': 1.6,  # Hair follicle cycling
                    'sulfur_amino_acids': 2.2  # Cysteine, methionine
                },
                optimal_ranges={
                    'biotin': (30, 10000),  # mcg/day (hair loss: 5000+)
                    'protein': (1.0, 1.5),  # g/kg
                    'iron': (18, 30),  # mg/day (women higher)
                    'zinc': (15, 30),  # mg/day
                    'silica': (10, 40)  # mg/day
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.S_H],  # Keratin, sulfur
                key_metrics=['hair_growth_rate', 'nail_strength', 'keratin_production']
            )
            
            # Bone Health / Density
            self.profiles[HealthGoal.BONE_HEALTH] = GoalMolecularProfile(
                goal=HealthGoal.BONE_HEALTH,
                target_molecules={
                    'calcium': 2.8,  # Bone mineral
                    'vitamin_d': 2.8,  # Calcium absorption
                    'vitamin_k2': 2.5,  # Directs calcium to bone
                    'magnesium': 2.0,  # Bone structure
                    'protein': 1.9,  # Bone matrix
                    'phosphorus': 1.7,  # Hydroxyapatite
                    'boron': 1.5,  # Bone metabolism
                    'vitamin_c': 1.6  # Collagen
                },
                optimal_ranges={
                    'calcium': (1000, 1300),  # mg/day (age-dependent)
                    'vitamin_d': (800, 2000),  # IU/day
                    'vitamin_k2': (90, 200),  # mcg/day (MK-7)
                    'protein': (1.0, 1.2),  # g/kg
                    'resistance_training': (2, 4)  # days/week
                },
                priority_bonds=[ChemicalBondType.O_H, ChemicalBondType.P_O],  # Minerals
                key_metrics=['bone_mineral_density', 'fracture_risk', 'calcium_balance']
            )
            
            # Joint Health / Cartilage
            self.profiles[HealthGoal.JOINT_HEALTH] = GoalMolecularProfile(
                goal=HealthGoal.JOINT_HEALTH,
                target_molecules={
                    'glucosamine': 2.5,  # Cartilage synthesis
                    'chondroitin': 2.3,  # Cartilage structure
                    'omega_3': 2.5,  # Anti-inflammatory
                    'collagen': 2.2,  # Joint structure
                    'hyaluronic_acid': 2.0,  # Synovial fluid
                    'curcumin': 2.0,  # Anti-inflammatory
                    'msm': 1.8,  # Sulfur, cartilage
                    'vitamin_c': 1.7  # Collagen synthesis
                },
                optimal_ranges={
                    'glucosamine': (1500, 1500),  # mg/day (standard dose)
                    'chondroitin': (1200, 1200),  # mg/day
                    'omega_3': (2, 4),  # g/day (high for inflammation)
                    'collagen': (10, 15),  # g/day
                    'curcumin': (500, 2000)  # mg/day
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.O_H, ChemicalBondType.S_O],
                key_metrics=['joint_pain_score', 'range_of_motion', 'cartilage_thickness']
            )
            
            # Eye Health / Vision
            self.profiles[HealthGoal.EYE_HEALTH] = GoalMolecularProfile(
                goal=HealthGoal.EYE_HEALTH,
                target_molecules={
                    'lutein': 2.8,  # Macular pigment
                    'zeaxanthin': 2.7,  # Macular pigment
                    'omega_3_dha': 2.5,  # Retinal health
                    'vitamin_a': 2.3,  # Rhodopsin
                    'vitamin_c': 1.9,  # Antioxidant
                    'vitamin_e': 1.8,  # Antioxidant
                    'zinc': 1.9,  # Retinal metabolism
                    'astaxanthin': 1.7  # Eye strain, blood flow
                },
                optimal_ranges={
                    'lutein': (10, 20),  # mg/day
                    'zeaxanthin': (2, 4),  # mg/day
                    'dha': (500, 1000),  # mg/day
                    'vitamin_a': (900, 3000),  # mcg RAE/day
                    'zinc': (11, 40)  # mg/day
                },
                priority_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],  # Carotenoids
                key_metrics=['macular_pigment_density', 'night_vision', 'blue_light_protection']
            )
            
            # Sleep Quality / Circadian Rhythm
            self.profiles[HealthGoal.SLEEP_QUALITY] = GoalMolecularProfile(
                goal=HealthGoal.SLEEP_QUALITY,
                target_molecules={
                    'tryptophan': 2.5,  # Serotonin → Melatonin
                    'magnesium': 2.3,  # GABA, relaxation
                    'glycine': 2.0,  # Sleep quality
                    'tart_cherry': 2.2,  # Natural melatonin
                    'gaba': 1.8,  # Inhibitory neurotransmitter
                    'theanine': 1.9,  # Relaxation without sedation
                    'vitamin_b6': 1.7,  # Serotonin synthesis
                    'calcium': 1.6  # Melatonin production
                },
                optimal_ranges={
                    'tryptophan': (250, 500),  # mg/day (evening)
                    'magnesium': (400, 500),  # mg/day (evening)
                    'glycine': (3, 5),  # g/day (before bed)
                    'tart_cherry_juice': (240, 480),  # ml/day
                    'avoid_caffeine_hours': (6, 8)  # hours before bed
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.O_H],
                key_metrics=['sleep_latency', 'sleep_duration', 'sleep_quality', 'rem_cycles']
            )
            
            # Stress Management / Cortisol Regulation
            self.profiles[HealthGoal.STRESS_MANAGEMENT] = GoalMolecularProfile(
                goal=HealthGoal.STRESS_MANAGEMENT,
                target_molecules={
                    'adaptogens': 2.5,  # Ashwagandha, rhodiola
                    'magnesium': 2.3,  # Cortisol regulation
                    'omega_3': 2.0,  # HPA axis
                    'vitamin_c': 1.9,  # Adrenal support
                    'b_vitamins': 2.0,  # Energy, neurotransmitters
                    'theanine': 1.9,  # Calm focus
                    'phosphatidylserine': 1.7,  # Cortisol reduction
                    'holy_basil': 1.6  # Adaptogen
                },
                optimal_ranges={
                    'ashwagandha': (300, 600),  # mg/day
                    'magnesium': (400, 500),  # mg/day
                    'omega_3': (2, 3),  # g/day
                    'vitamin_c': (500, 2000),  # mg/day
                    'b_complex': (1, 2)  # servings/day
                },
                priority_bonds=[ChemicalBondType.O_H, ChemicalBondType.N_H],
                key_metrics=['cortisol_levels', 'perceived_stress', 'hpa_axis_function']
            )
            
            # ===================================================================
            # PHASE 1: LIFE STAGE GOALS (5)
            # ===================================================================
            
            # Pregnancy / Prenatal Nutrition
            self.profiles[HealthGoal.PREGNANCY] = GoalMolecularProfile(
                goal=HealthGoal.PREGNANCY,
                target_molecules={
                    'folate': 3.0,  # Neural tube development (CRITICAL)
                    'iron': 2.8,  # Increased blood volume
                    'dha': 2.7,  # Fetal brain development
                    'calcium': 2.3,  # Fetal bone development
                    'protein': 2.2,  # Fetal growth
                    'choline': 2.5,  # Brain development
                    'iodine': 2.0,  # Thyroid function
                    'vitamin_d': 1.9  # Immune, bone
                },
                optimal_ranges={
                    'folate': (600, 1000),  # mcg DFE/day (prenatal)
                    'iron': (27, 45),  # mg/day (pregnancy)
                    'dha': (200, 300),  # mg/day
                    'calcium': (1000, 1300),  # mg/day
                    'protein': (1.1, 1.5),  # g/kg (increases by trimester)
                    'calories': (300, 500)  # kcal/day surplus (2nd/3rd trimester)
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.DOUBLE_BOND],
                key_metrics=['fetal_development', 'maternal_nutrition_status', 'birth_weight']
            )
            
            # Lactation / Breastfeeding / Postpartum
            self.profiles[HealthGoal.LACTATION] = GoalMolecularProfile(
                goal=HealthGoal.LACTATION,
                target_molecules={
                    'calories': 2.8,  # Milk production (500 kcal/day)
                    'protein': 2.5,  # Milk protein
                    'dha': 2.7,  # Infant brain development
                    'calcium': 2.3,  # Milk calcium, maternal bone
                    'iodine': 2.2,  # Infant thyroid
                    'water': 2.8,  # Hydration critical
                    'galactagogues': 1.8,  # Fenugreek, oats
                    'vitamin_b12': 2.0  # Infant development
                },
                optimal_ranges={
                    'calories': (450, 500),  # kcal/day surplus
                    'protein': (1.3, 1.5),  # g/kg
                    'dha': (200, 300),  # mg/day
                    'water': (3000, 4000),  # ml/day (critical!)
                    'calcium': (1000, 1300),  # mg/day
                    'iodine': (290, 290)  # mcg/day
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.O_H],
                key_metrics=['milk_supply', 'milk_quality', 'infant_growth']
            )
            
            # Menopause / Hormonal Balance
            self.profiles[HealthGoal.MENOPAUSE] = GoalMolecularProfile(
                goal=HealthGoal.MENOPAUSE,
                target_molecules={
                    'phytoestrogens': 2.5,  # Soy isoflavones
                    'calcium': 2.8,  # Bone loss accelerates
                    'vitamin_d': 2.7,  # Bone health
                    'omega_3': 2.0,  # Hot flashes, mood
                    'vitamin_e': 1.9,  # Hot flashes
                    'magnesium': 1.8,  # Mood, sleep
                    'black_cohosh': 1.7,  # Symptom relief
                    'vitamin_k2': 2.0  # Bone health
                },
                optimal_ranges={
                    'soy_isoflavones': (40, 80),  # mg/day
                    'calcium': (1200, 1500),  # mg/day (postmenopause)
                    'vitamin_d': (800, 2000),  # IU/day
                    'omega_3': (2, 3),  # g/day
                    'protein': (1.0, 1.2)  # g/kg (preserve muscle)
                },
                priority_bonds=[ChemicalBondType.O_H, ChemicalBondType.AROMATIC],  # Phytoestrogens
                key_metrics=['hot_flash_frequency', 'bone_density', 'mood_stability']
            )
            
            # Fertility / Reproductive Health
            self.profiles[HealthGoal.FERTILITY] = GoalMolecularProfile(
                goal=HealthGoal.FERTILITY,
                target_molecules={
                    'folate': 2.8,  # Egg/sperm quality
                    'coq10': 2.7,  # Mitochondrial function
                    'omega_3': 2.3,  # Hormone production
                    'vitamin_d': 2.5,  # Fertility, implantation
                    'zinc': 2.4,  # Sperm production
                    'selenium': 2.0,  # Sperm motility
                    'antioxidants': 2.2,  # Oxidative stress
                    'inositol': 2.5  # PCOS, egg quality
                },
                optimal_ranges={
                    'folate': (400, 800),  # mcg/day (preconception)
                    'coq10': (200, 600),  # mg/day
                    'vitamin_d': (2000, 4000),  # IU/day
                    'zinc': (15, 30),  # mg/day (men higher)
                    'omega_3': (2, 3),  # g/day
                    'bmi': (18.5, 24.9)  # Optimal range
                },
                priority_bonds=[ChemicalBondType.O_H, ChemicalBondType.DOUBLE_BOND],
                key_metrics=['egg_quality', 'sperm_quality', 'hormonal_balance', 'implantation_rate']
            )
            
            # Healthy Aging / Vitality (55+)
            self.profiles[HealthGoal.HEALTHY_AGING] = GoalMolecularProfile(
                goal=HealthGoal.HEALTHY_AGING,
                target_molecules={
                    'protein': 2.8,  # Prevent sarcopenia
                    'omega_3': 2.5,  # Brain, heart, inflammation
                    'vitamin_d': 2.7,  # Bone, muscle, immune
                    'calcium': 2.5,  # Bone health
                    'b12': 2.6,  # Absorption declines
                    'polyphenols': 2.3,  # Antioxidants
                    'fiber': 2.0,  # Digestive health
                    'antioxidants': 2.2  # Oxidative stress
                },
                optimal_ranges={
                    'protein': (1.2, 1.5),  # g/kg (higher for aging)
                    'vitamin_d': (800, 2000),  # IU/day
                    'calcium': (1200, 1500),  # mg/day
                    'b12': (2.4, 1000),  # mcg/day (sublingual if low acid)
                    'omega_3': (2, 4),  # g/day
                    'resistance_training': (2, 3)  # days/week
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.O_H, ChemicalBondType.DOUBLE_BOND],
                key_metrics=['muscle_mass', 'bone_density', 'cognitive_function', 'functional_mobility']
            )
            
            # ===================================================================
            # PHASE 2: ADDITIONAL GOALS (12 more goals)
            # ===================================================================
            
            # Detoxification / Cleanse
            self.profiles[HealthGoal.DETOX] = GoalMolecularProfile(
                goal=HealthGoal.DETOX,
                target_molecules={
                    'cruciferous_compounds': 2.8,  # Phase 2 detox (sulforaphane)
                    'glutathione_precursors': 2.5,  # Master antioxidant
                    'fiber': 2.5,  # Bile binding, elimination
                    'milk_thistle': 2.3,  # Liver protection (silymarin)
                    'chlorophyll': 2.0,  # Heavy metal binding
                    'water': 2.8,  # Kidney filtration
                    'antioxidants': 2.2,  # Oxidative stress
                    'magnesium': 1.8  # Bowel movements
                },
                optimal_ranges={
                    'water': (3000, 4000),  # ml/day (critical)
                    'fiber': (35, 50),  # g/day
                    'cruciferous': (200, 400),  # g/day (broccoli, kale)
                    'glutathione': (250, 500),  # mg/day (or NAC precursor)
                    'milk_thistle': (200, 400)  # mg/day (silymarin)
                },
                priority_bonds=[ChemicalBondType.S_H, ChemicalBondType.O_H],  # Sulfur, antioxidants
                key_metrics=['liver_enzymes', 'urine_toxins', 'bowel_regularity']
            )
            
            # Mental Clarity / Focus / Nootropic
            self.profiles[HealthGoal.MENTAL_CLARITY] = GoalMolecularProfile(
                goal=HealthGoal.MENTAL_CLARITY,
                target_molecules={
                    'caffeine': 2.5,  # Alertness, focus
                    'l_theanine': 2.3,  # Calm focus (pairs with caffeine)
                    'omega_3_dha': 2.5,  # Brain structure
                    'b_vitamins': 2.0,  # Neurotransmitters
                    'choline': 2.2,  # Acetylcholine
                    'bacopa': 1.9,  # Memory
                    'ginkgo': 1.8,  # Blood flow
                    'phosphatidylserine': 1.7  # Cognitive function
                },
                optimal_ranges={
                    'caffeine': (100, 400),  # mg/day (optimal range)
                    'l_theanine': (100, 200),  # mg/day (2:1 with caffeine)
                    'dha': (500, 1000),  # mg/day
                    'choline': (400, 550),  # mg/day
                    'b_complex': (100, 100)  # % DV
                },
                priority_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.N_H],
                key_metrics=['focus_duration', 'reaction_time', 'working_memory']
            )
            
            # Memory Enhancement
            self.profiles[HealthGoal.MEMORY] = GoalMolecularProfile(
                goal=HealthGoal.MEMORY,
                target_molecules={
                    'dha': 2.8,  # Hippocampus structure
                    'phosphatidylserine': 2.5,  # Memory formation
                    'choline': 2.5,  # Acetylcholine synthesis
                    'bacopa': 2.3,  # Ayurvedic memory herb
                    'lion_mane': 2.0,  # NGF (nerve growth factor)
                    'ginkgo': 1.9,  # Cerebral blood flow
                    'b_vitamins': 2.0,  # B6, B12, folate
                    'antioxidants': 1.8  # Neuroprotection
                },
                optimal_ranges={
                    'dha': (1000, 2000),  # mg/day (high dose)
                    'phosphatidylserine': (300, 300),  # mg/day (standard)
                    'choline': (400, 550),  # mg/day
                    'bacopa': (300, 450),  # mg/day
                    'b12': (100, 1000)  # mcg/day
                },
                priority_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.P_O],
                key_metrics=['recall_speed', 'short_term_memory', 'long_term_memory']
            )
            
            # Concentration / Deep Work
            self.profiles[HealthGoal.CONCENTRATION] = GoalMolecularProfile(
                goal=HealthGoal.CONCENTRATION,
                target_molecules={
                    'caffeine': 2.8,  # Adenosine blockade
                    'l_tyrosine': 2.5,  # Dopamine precursor (stress)
                    'l_theanine': 2.3,  # Calm alertness
                    'omega_3': 2.0,  # Brain function
                    'b_vitamins': 1.9,  # Energy metabolism
                    'rhodiola': 1.8,  # Adaptogen, focus under stress
                    'citicoline': 1.7  # Acetylcholine, dopamine
                },
                optimal_ranges={
                    'caffeine': (200, 400),  # mg/day (higher for concentration)
                    'l_tyrosine': (500, 2000),  # mg/day (stress-dependent)
                    'l_theanine': (200, 400),  # mg/day
                    'citicoline': (250, 500),  # mg/day
                    'omega_3': (2, 3)  # g/day
                },
                priority_bonds=[ChemicalBondType.N_H],  # Amino acids
                key_metrics=['flow_state_duration', 'distraction_resistance', 'task_completion']
            )
            
            # Injury Rehabilitation
            self.profiles[HealthGoal.INJURY_REHAB] = GoalMolecularProfile(
                goal=HealthGoal.INJURY_REHAB,
                target_molecules={
                    'protein': 2.8,  # Tissue repair (1.6-2.0g/kg)
                    'collagen': 2.7,  # Connective tissue (15-20g/day)
                    'vitamin_c': 2.5,  # Collagen synthesis (1000mg)
                    'omega_3': 2.3,  # Anti-inflammatory (3-4g)
                    'zinc': 2.2,  # Wound healing (15-30mg)
                    'vitamin_a': 2.0,  # Cell differentiation
                    'glutamine': 1.9,  # Immune, healing (10-20g)
                    'curcumin': 1.8  # Anti-inflammatory
                },
                optimal_ranges={
                    'protein': (1.6, 2.0),  # g/kg (elevated)
                    'collagen': (15, 20),  # g/day
                    'vitamin_c': (500, 2000),  # mg/day
                    'omega_3': (3, 4),  # g/day (anti-inflammatory)
                    'zinc': (15, 30),  # mg/day
                    'calories': (250, 500)  # Surplus for healing
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.O_H],  # Protein, collagen
                key_metrics=['healing_rate', 'inflammation_markers', 'pain_score', 'range_of_motion']
            )
            
            # Post-Surgery Recovery
            self.profiles[HealthGoal.POST_SURGERY] = GoalMolecularProfile(
                goal=HealthGoal.POST_SURGERY,
                target_molecules={
                    'protein': 3.0,  # HIGHEST - wound healing (2.0-2.5g/kg)
                    'vitamin_c': 2.8,  # Collagen (2000mg)
                    'zinc': 2.7,  # Wound healing (30-50mg)
                    'vitamin_a': 2.5,  # Epithelialization
                    'arginine': 2.3,  # Immune, wound healing
                    'omega_3': 2.0,  # Inflammation resolution
                    'glutamine': 2.2,  # Immune support (20-30g)
                    'iron': 2.0  # Blood loss replacement
                },
                optimal_ranges={
                    'protein': (2.0, 2.5),  # g/kg (very high)
                    'vitamin_c': (1000, 2000),  # mg/day
                    'zinc': (30, 50),  # mg/day (high dose)
                    'vitamin_a': (10000, 25000),  # IU/day (short-term)
                    'arginine': (15, 30),  # g/day
                    'calories': (500, 1000)  # Surplus for healing
                },
                priority_bonds=[ChemicalBondType.N_H],  # Protein dominant
                key_metrics=['wound_closure_rate', 'infection_risk', 'protein_status']
            )
            
            # Immune Boost (Acute - Cold/Flu Prevention)
            self.profiles[HealthGoal.IMMUNE_BOOST] = GoalMolecularProfile(
                goal=HealthGoal.IMMUNE_BOOST,
                target_molecules={
                    'vitamin_c': 3.0,  # Highest priority (1000-2000mg)
                    'zinc': 2.8,  # Immune cells (30-50mg, short-term)
                    'vitamin_d': 2.7,  # Immune regulation (4000-5000 IU)
                    'elderberry': 2.5,  # Antiviral
                    'echinacea': 2.0,  # Immune stimulation
                    'garlic': 2.2,  # Antimicrobial
                    'quercetin': 2.0,  # Antiviral, zinc ionophore
                    'probiotics': 1.9  # Gut-immune axis
                },
                optimal_ranges={
                    'vitamin_c': (1000, 3000),  # mg/day (high dose)
                    'zinc': (30, 50),  # mg/day (SHORT-TERM only, <2 weeks)
                    'vitamin_d': (4000, 5000),  # IU/day (loading)
                    'elderberry': (300, 600),  # mg/day
                    'garlic': (600, 1200)  # mg/day (aged garlic extract)
                },
                priority_bonds=[ChemicalBondType.O_H, ChemicalBondType.S_H],  # Antioxidants, sulfur
                key_metrics=['white_blood_cell_count', 'illness_duration', 'symptom_severity']
            )
            
            # Allergy Management
            self.profiles[HealthGoal.ALLERGY_MANAGEMENT] = GoalMolecularProfile(
                goal=HealthGoal.ALLERGY_MANAGEMENT,
                target_molecules={
                    'quercetin': 2.8,  # Mast cell stabilizer
                    'vitamin_c': 2.5,  # Antihistamine effect
                    'omega_3': 2.3,  # Anti-inflammatory
                    'probiotics': 2.2,  # Immune tolerance
                    'stinging_nettle': 2.0,  # Natural antihistamine
                    'butterbur': 1.9,  # Allergic rhinitis
                    'vitamin_d': 1.8  # Immune modulation
                },
                harmful_molecules={
                    'histamine_foods': 2.5,  # Aged cheese, wine, fermented
                    'food_allergens': 3.0,  # Individual triggers
                    'sulfites': 2.0,  # Preservatives
                    'msg': 1.8  # Some sensitive
                },
                optimal_ranges={
                    'quercetin': (500, 1000),  # mg/day
                    'vitamin_c': (1000, 2000),  # mg/day
                    'omega_3': (2, 3),  # g/day
                    'probiotics': (25_000_000_000, 50_000_000_000)  # CFU/day (high dose)
                },
                priority_bonds=[ChemicalBondType.O_H, ChemicalBondType.AROMATIC],
                key_metrics=['symptom_frequency', 'symptom_severity', 'medication_use']
            )
            
            # Hormone Optimization (Testosterone - Men)
            self.profiles[HealthGoal.TESTOSTERONE_OPTIMIZATION] = GoalMolecularProfile(
                goal=HealthGoal.TESTOSTERONE_OPTIMIZATION,
                target_molecules={
                    'zinc': 2.8,  # Testosterone synthesis (30mg)
                    'vitamin_d': 2.7,  # Steroid hormone (2000-4000 IU)
                    'magnesium': 2.3,  # Free testosterone
                    'boron': 2.0,  # Increases free T (6-10mg)
                    'healthy_fats': 2.5,  # Cholesterol → hormones
                    'cruciferous': 1.9,  # DIM (estrogen metabolism)
                    'ashwagandha': 2.2,  # Adaptogen, lowers cortisol
                    'vitamin_k2': 1.7  # Testosterone synthesis
                },
                harmful_molecules={
                    'soy_isoflavones': 2.0,  # Phytoestrogens
                    'alcohol': 2.5,  # Lowers testosterone
                    'trans_fat': 2.0,  # Inflammation
                    'licorice': 2.2,  # Lowers testosterone
                    'mint': 1.8  # Some evidence of lowering T
                },
                optimal_ranges={
                    'zinc': (25, 40),  # mg/day
                    'vitamin_d': (2000, 4000),  # IU/day
                    'magnesium': (400, 500),  # mg/day
                    'boron': (6, 10),  # mg/day
                    'healthy_fats': (60, 100),  # g/day (30% calories)
                    'ashwagandha': (300, 600)  # mg/day
                },
                priority_bonds=[ChemicalBondType.O_H],  # Healthy fats
                key_metrics=['total_testosterone', 'free_testosterone', 'shbg', 'estradiol']
            )
            
            # Hormone Balance (Estrogen - Women)
            self.profiles[HealthGoal.ESTROGEN_BALANCE] = GoalMolecularProfile(
                goal=HealthGoal.ESTROGEN_BALANCE,
                target_molecules={
                    'cruciferous_compounds': 2.8,  # DIM, I3C (estrogen metabolism)
                    'fiber': 2.5,  # Estrogen elimination
                    'flax_seeds': 2.3,  # Lignans (modulate estrogen)
                    'b_vitamins': 2.0,  # Methylation, detox
                    'magnesium': 1.9,  # Estrogen balance
                    'probiotics': 2.0,  # Estrobolome (gut-estrogen)
                    'calcium_d_glucarate': 1.8  # Estrogen detox
                },
                harmful_molecules={
                    'xenoestrogens': 2.8,  # Plastics (BPA), pesticides
                    'alcohol': 2.3,  # Increases estrogen
                    'sugar': 2.0,  # Insulin affects estrogen
                    'conventional_dairy': 1.8  # Hormones
                },
                optimal_ranges={
                    'cruciferous': (200, 400),  # g/day
                    'fiber': (35, 50),  # g/day (critical for elimination)
                    'flax_seeds': (1, 2),  # tablespoons/day
                    'probiotics': (25_000_000_000, 25_000_000_000),  # CFU/day
                    'b_complex': (100, 100)  # % DV
                },
                priority_bonds=[ChemicalBondType.S_H, ChemicalBondType.O_H],  # Cruciferous, fiber
                key_metrics=['estradiol', 'estrone', 'estriol', 'progesterone_ratio']
            )
            
            # Hydration / Electrolyte Balance
            self.profiles[HealthGoal.HYDRATION] = GoalMolecularProfile(
                goal=HealthGoal.HYDRATION,
                target_molecules={
                    'water': 3.0,  # HIGHEST priority
                    'sodium': 2.5,  # Electrolyte (2000-5000mg for athletes)
                    'potassium': 2.5,  # Electrolyte (3500-4700mg)
                    'magnesium': 2.0,  # Electrolyte (400mg)
                    'coconut_water': 2.3,  # Natural electrolytes
                    'watermelon': 1.8,  # High water content + citrulline
                    'cucumber': 1.7  # High water content
                },
                harmful_molecules={
                    'caffeine': 1.8,  # Mild diuretic
                    'alcohol': 2.5,  # Strong diuretic
                    'high_protein': 1.5,  # Increases water needs
                    'sugar': 1.6  # High osmolality
                },
                optimal_ranges={
                    'water': (3000, 5000),  # ml/day (activity-dependent)
                    'sodium': (2000, 5000),  # mg/day (sweat losses)
                    'potassium': (3500, 4700),  # mg/day
                    'magnesium': (400, 500),  # mg/day
                    'urine_color': (1, 3)  # Pale yellow (1-3 on chart)
                },
                priority_bonds=[ChemicalBondType.O_H],  # Water
                key_metrics=['urine_specific_gravity', 'urine_color', 'body_weight_change']
            )
            
            # Anti-Inflammatory Diet
            self.profiles[HealthGoal.ANTI_INFLAMMATORY] = GoalMolecularProfile(
                goal=HealthGoal.ANTI_INFLAMMATORY,
                target_molecules={
                    'omega_3': 3.0,  # EPA/DHA (3-4g/day)
                    'curcumin': 2.8,  # Turmeric (1000-2000mg)
                    'ginger': 2.5,  # Gingerols (1-2g)
                    'polyphenols': 2.7,  # Berries, tea, cocoa
                    'vitamin_d': 2.3,  # Immune modulation
                    'quercetin': 2.2,  # Flavonoid
                    'resveratrol': 2.0,  # Sirtuin activation
                    'green_tea': 2.0  # EGCG
                },
                harmful_molecules={
                    'omega_6_excess': 2.8,  # Pro-inflammatory (>10g/day)
                    'trans_fat': 3.0,  # Highly inflammatory
                    'refined_sugar': 2.7,  # AGEs, inflammation
                    'processed_meat': 2.5,  # AGEs, nitrites
                    'fried_foods': 2.5,  # Oxidized fats
                    'alcohol': 2.0  # Pro-inflammatory (excess)
                },
                optimal_ranges={
                    'omega_3': (3, 4),  # g/day (high dose)
                    'omega_6_omega_3_ratio': (1, 4),  # Ratio (ideally 1:1 to 4:1)
                    'curcumin': (1000, 2000),  # mg/day
                    'polyphenols': (1500, 3000),  # mg/day
                    'antioxidants': (15000, 25000)  # ORAC units
                },
                priority_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H, ChemicalBondType.AROMATIC],
                key_metrics=['crp', 'il_6', 'tnf_alpha', 'oxidative_stress']
            )
            
            # ===== PHASE 3: 15 SPECIALIZED GOALS =====
            
            # Ultra-Endurance (100+ mile events, Ironman, ultra-marathon)
            self.profiles[HealthGoal.ULTRA_ENDURANCE] = GoalMolecularProfile(
                goal=HealthGoal.ULTRA_ENDURANCE,
                target_molecules={
                    'carbs': 3.0,  # 8-12g/kg (glycogen supercompensation)
                    'electrolytes': 3.0,  # Massive losses (1000+ mg Na/hr)
                    'mct_oil': 2.8,  # Fat adaptation, ketones
                    'caffeine': 2.5,  # Endurance, fat oxidation (200-400mg)
                    'beetroot': 2.7,  # Nitrates, O2 efficiency
                    'beta_alanine': 2.3,  # Carnosine buffering (3-6g)
                    'sodium': 3.0,  # Critical (3000-7000mg/day training)
                    'bcaa': 2.0  # Muscle preservation ultra-distance
                },
                harmful_molecules={
                    'fiber': 2.5,  # GI distress during event
                    'fat': 2.0,  # Delays gastric emptying (race day)
                    'new_foods': 2.8,  # GI issues (test in training!)
                    'protein': 1.8,  # Moderate (race day, not excessive)
                },
                optimal_ranges={
                    'carbs': (8, 12),  # g/kg body weight
                    'sodium': (5000, 8000),  # mg/day (training)
                    'water': (5, 10),  # L/day (training volume dependent)
                    'calories': (4000, 7000),  # kcal/day (extreme expenditure)
                    'mct_oil': (20, 40)  # g/day (fat adaptation)
                },
                priority_bonds=[ChemicalBondType.C_H, ChemicalBondType.C_C],
                key_metrics=['vo2_max', 'lactate_threshold', 'fat_oxidation_rate', 'sweat_rate']
            )
            
            # Powerlifting (Maximal Strength)
            self.profiles[HealthGoal.POWERLIFTING] = GoalMolecularProfile(
                goal=HealthGoal.POWERLIFTING,
                target_molecules={
                    'creatine': 3.0,  # 5g/day (ATP-PC system)
                    'protein': 2.8,  # 1.6-2.2g/kg (muscle maintenance)
                    'carbs': 2.5,  # Glycogen for high-intensity
                    'leucine': 2.7,  # BCAA, mTOR activation
                    'beta_alanine': 2.3,  # Carnosine, buffering
                    'citrulline': 2.0,  # Pump, blood flow (6-8g)
                    'caffeine': 2.5,  # CNS activation (300-600mg pre-lift)
                    'betaine': 2.3  # Trimethylglycine, power output
                },
                harmful_molecules={
                    'alcohol': 2.8,  # Muscle protein synthesis inhibition
                    'excess_cardio': 2.5,  # Interference effect
                    'low_calories': 2.7,  # Need surplus or maintenance
                },
                optimal_ranges={
                    'protein': (1.6, 2.2),  # g/kg
                    'creatine': (5, 5),  # g/day (maintenance)
                    'carbs': (4, 7),  # g/kg (fuel for intensity)
                    'calories': (3000, 4500),  # kcal/day (depends on weight class)
                    'caffeine': (300, 600)  # mg pre-workout
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.P_O],
                key_metrics=['1rm', 'rate_of_force_development', 'muscle_mass', 'recovery_time']
            )
            
            # CrossFit / HYROX (Mixed Modal)
            self.profiles[HealthGoal.CROSSFIT] = GoalMolecularProfile(
                goal=HealthGoal.CROSSFIT,
                target_molecules={
                    'carbs': 2.8,  # 5-8g/kg (glycolytic + aerobic)
                    'protein': 2.7,  # 1.6-2.0g/kg (recovery)
                    'creatine': 2.5,  # 5g/day (power movements)
                    'electrolytes': 2.5,  # High-intensity sweating
                    'beta_alanine': 2.3,  # Lactate buffering
                    'citrulline': 2.0,  # Work capacity
                    'caffeine': 2.3,  # Multi-domain performance
                    'omega_3': 2.0  # Anti-inflammatory recovery
                },
                harmful_molecules={
                    'low_carb': 2.8,  # Inadequate for intensity
                    'dehydration': 2.7,  # Performance killer
                    'alcohol': 2.5,  # Recovery impairment
                },
                optimal_ranges={
                    'carbs': (5, 8),  # g/kg
                    'protein': (1.6, 2.0),  # g/kg
                    'calories': (2800, 4000),  # kcal/day
                    'water': (3, 5),  # L/day
                    'creatine': (5, 5)  # g/day
                },
                priority_bonds=[ChemicalBondType.C_H, ChemicalBondType.N_H, ChemicalBondType.P_O],
                key_metrics=['work_capacity', 'vo2_max', 'power_output', 'recovery_heart_rate']
            )
            
            # Vegan Optimization (Plant-Based Nutrition)
            self.profiles[HealthGoal.VEGAN_OPTIMIZATION] = GoalMolecularProfile(
                goal=HealthGoal.VEGAN_OPTIMIZATION,
                target_molecules={
                    'vitamin_b12': 3.0,  # CRITICAL (2.4-100mcg supplement)
                    'iron': 2.8,  # Non-heme requires vitamin C
                    'vitamin_c': 2.5,  # Enhances iron absorption
                    'omega_3_ala': 2.5,  # Flax, chia, walnuts (+ algae DHA)
                    'zinc': 2.3,  # Phytates reduce absorption
                    'iodine': 2.5,  # Iodized salt or seaweed (150mcg)
                    'calcium': 2.3,  # Fortified or greens (1000mg)
                    'vitamin_d': 2.5,  # D2 (mushrooms) or D3 (lichen)
                    'protein': 2.5,  # Complete amino acids (legumes + grains)
                },
                harmful_molecules={
                    'phytates': 2.0,  # Mineral absorption (soak/sprout)
                    'oxalates': 1.8,  # Calcium absorption (spinach)
                    'insufficient_calories': 2.5,  # Common pitfall
                },
                optimal_ranges={
                    'vitamin_b12': (250, 1000),  # mcg/day (supplement)
                    'iron': (18, 30),  # mg/day (higher for non-heme)
                    'protein': (1.2, 1.8),  # g/kg (complementary sources)
                    'omega_3_ala': (2, 4),  # g/day ALA (convert to EPA/DHA)
                    'zinc': (15, 25),  # mg/day (higher for phytates)
                    'iodine': (150, 300),  # mcg/day
                    'calcium': (1000, 1300)  # mg/day
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.O_H, ChemicalBondType.DOUBLE_BOND],
                key_metrics=['b12_status', 'iron_ferritin', 'zinc_status', 'protein_adequacy']
            )
            
            # Ketogenic Diet (Therapeutic Keto)
            self.profiles[HealthGoal.KETOGENIC_DIET] = GoalMolecularProfile(
                goal=HealthGoal.KETOGENIC_DIET,
                target_molecules={
                    'fat': 3.0,  # 70-80% calories (130-200g)
                    'mct_oil': 2.8,  # Rapid ketone production (30-60g)
                    'electrolytes': 2.8,  # Keto flu prevention (5000mg Na)
                    'magnesium': 2.5,  # Often deficient (400-600mg)
                    'potassium': 2.5,  # 3500-4700mg
                    'omega_3': 2.3,  # Anti-inflammatory
                    'fiber': 2.3,  # Constipation prevention (25g+)
                    'avocado': 2.5  # Healthy fats + fiber + potassium
                },
                harmful_molecules={
                    'carbs': 3.0,  # <50g/day (strict), <20g (therapeutic)
                    'protein': 2.0,  # Excess converts to glucose (GNG)
                    'sugar': 3.0,  # Breaks ketosis instantly
                    'grains': 2.8,  # High carb
                    'fruit': 2.5  # High fructose (berries OK in moderation)
                },
                optimal_ranges={
                    'fat': (130, 200),  # g/day (70-80% calories)
                    'carbs': (20, 50),  # g/day (net carbs)
                    'protein': (75, 125),  # g/day (0.8-1.2g/kg moderate)
                    'mct_oil': (30, 60),  # g/day
                    'sodium': (5000, 7000),  # mg/day
                    'ketones': (0.5, 3.0)  # mmol/L blood ketones
                },
                priority_bonds=[ChemicalBondType.C_H, ChemicalBondType.C_C],
                key_metrics=['blood_ketones', 'blood_glucose', 'electrolyte_status', 'energy_levels']
            )
            
            # Paleo Diet (Ancestral Nutrition)
            self.profiles[HealthGoal.PALEO_DIET] = GoalMolecularProfile(
                goal=HealthGoal.PALEO_DIET,
                target_molecules={
                    'protein': 2.8,  # Grass-fed, wild-caught (1.2-2.0g/kg)
                    'omega_3': 2.7,  # Wild fish, grass-fed beef
                    'vegetables': 2.8,  # Nutrient density
                    'healthy_fats': 2.5,  # Avocado, nuts, olive oil
                    'antioxidants': 2.3,  # Colorful plants
                    'collagen': 2.0,  # Bone broth, connective tissue
                    'vitamin_d': 2.3  # Sunlight + diet
                },
                harmful_molecules={
                    'grains': 2.8,  # Excluded (gluten, lectins)
                    'legumes': 2.5,  # Excluded (lectins, phytates)
                    'dairy': 2.5,  # Excluded (some include grass-fed)
                    'processed_foods': 3.0,  # Completely avoided
                    'refined_sugar': 2.8,  # Excluded
                    'vegetable_oils': 2.7  # Omega-6 excess
                },
                optimal_ranges={
                    'protein': (100, 180),  # g/day
                    'vegetables': (500, 1000),  # g/day (8-10 servings)
                    'omega_6_omega_3_ratio': (1, 2),  # Ratio (ancestral ~1:1)
                    'fiber': (30, 50),  # g/day (from vegetables)
                    'carbs': (100, 150)  # g/day (moderate from vegetables/fruit)
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                key_metrics=['inflammation_markers', 'gut_health', 'body_composition', 'energy']
            )
            
            # Mediterranean Diet (Heart Health Focus)
            self.profiles[HealthGoal.MEDITERRANEAN_DIET] = GoalMolecularProfile(
                goal=HealthGoal.MEDITERRANEAN_DIET,
                target_molecules={
                    'olive_oil': 3.0,  # EVOO (40-60ml/day monounsaturated)
                    'omega_3': 2.8,  # Fish 2-3x/week (EPA/DHA)
                    'polyphenols': 2.7,  # Wine, olive oil, berries
                    'fiber': 2.5,  # Whole grains, legumes (30-40g)
                    'nuts': 2.5,  # Almonds, walnuts (30g/day)
                    'vegetables': 2.7,  # 5-7 servings/day
                    'fish': 2.8,  # Fatty fish (salmon, sardines)
                    'legumes': 2.3  # Lentils, chickpeas
                },
                harmful_molecules={
                    'red_meat': 2.5,  # Limited (<2x/week)
                    'processed_meat': 2.8,  # Avoided
                    'trans_fat': 3.0,  # Completely avoided
                    'refined_sugar': 2.5,  # Limited
                    'butter': 2.0  # Replace with olive oil
                },
                optimal_ranges={
                    'olive_oil': (40, 60),  # ml/day (PREDIMED study)
                    'fish': (200, 400),  # g/week (2-3 servings)
                    'nuts': (30, 50),  # g/day
                    'vegetables': (400, 600),  # g/day
                    'legumes': (150, 300),  # g/week
                    'wine': (0, 150)  # ml/day (red, optional)
                },
                priority_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H, ChemicalBondType.AROMATIC],
                key_metrics=['cardiovascular_risk', 'cholesterol_ratio', 'inflammation', 'longevity']
            )
            
            # DASH Diet (Hypertension Management)
            self.profiles[HealthGoal.DASH_DIET] = GoalMolecularProfile(
                goal=HealthGoal.DASH_DIET,
                target_molecules={
                    'potassium': 3.0,  # 4700mg (bananas, potatoes, greens)
                    'magnesium': 2.8,  # 400-500mg (nuts, greens)
                    'calcium': 2.7,  # 1200mg (dairy, greens)
                    'fiber': 2.7,  # 30-40g (whole grains, vegetables)
                    'vegetables': 2.8,  # 4-5 servings/day
                    'fruits': 2.5,  # 4-5 servings/day
                    'whole_grains': 2.5,  # 6-8 servings/day
                    'low_fat_dairy': 2.3  # 2-3 servings/day
                },
                harmful_molecules={
                    'sodium': 3.0,  # <1500mg/day (strict)
                    'saturated_fat': 2.5,  # <7% calories
                    'red_meat': 2.3,  # Limited
                    'sweets': 2.5,  # <5 servings/week
                    'alcohol': 2.0  # Limited (1-2 drinks/day max)
                },
                optimal_ranges={
                    'sodium': (1000, 1500),  # mg/day (DASH-sodium study)
                    'potassium': (4000, 4700),  # mg/day
                    'magnesium': (400, 500),  # mg/day
                    'calcium': (1000, 1300),  # mg/day
                    'fiber': (30, 40),  # g/day
                    'vegetables': (400, 600)  # g/day
                },
                priority_bonds=[ChemicalBondType.O_H, ChemicalBondType.C_H],
                key_metrics=['blood_pressure', 'potassium_sodium_ratio', 'cardiovascular_risk']
            )
            
            # Anti-Acne (Hormonal & Inflammation)
            self.profiles[HealthGoal.ANTI_ACNE] = GoalMolecularProfile(
                goal=HealthGoal.ANTI_ACNE,
                target_molecules={
                    'omega_3': 2.8,  # Anti-inflammatory (2-3g EPA/DHA)
                    'zinc': 2.7,  # Sebum regulation (30-40mg)
                    'vitamin_a': 2.5,  # Skin cell turnover (5000-10000 IU)
                    'probiotics': 2.5,  # Gut-skin axis
                    'green_tea': 2.3,  # EGCG, anti-androgenic
                    'spearmint': 2.3,  # Anti-androgenic (for women)
                    'low_glycemic': 2.7,  # Reduces insulin/IGF-1
                    'antioxidants': 2.0  # Reduce inflammation
                },
                harmful_molecules={
                    'dairy': 2.8,  # IGF-1, hormones (especially skim)
                    'high_glycemic': 2.8,  # Insulin spike → androgens
                    'sugar': 2.7,  # Inflammation, insulin
                    'omega_6': 2.5,  # Pro-inflammatory ratio
                    'whey_protein': 2.5,  # Insulin spike (casein OK)
                    'iodine_excess': 2.0  # Kelp, seaweed (>1000mcg)
                },
                optimal_ranges={
                    'omega_3': (2, 3),  # g/day
                    'zinc': (30, 40),  # mg/day
                    'vitamin_a': (5000, 10000),  # IU/day (not if pregnant)
                    'glycemic_load': (0, 50),  # per day (low)
                    'probiotics': (10_000_000_000, 50_000_000_000)  # CFU/day
                },
                priority_bonds=[ChemicalBondType.DOUBLE_BOND, ChemicalBondType.O_H],
                key_metrics=['sebum_production', 'inflammation', 'hormone_levels', 'gut_health']
            )
            
            # Hangover Prevention/Recovery
            self.profiles[HealthGoal.HANGOVER_PREVENTION] = GoalMolecularProfile(
                goal=HealthGoal.HANGOVER_PREVENTION,
                target_molecules={
                    'nac': 2.8,  # N-acetyl cysteine, glutathione (600mg)
                    'vitamin_c': 2.5,  # Antioxidant (1000mg)
                    'b_vitamins': 2.7,  # B1, B6, B12 depleted by alcohol
                    'electrolytes': 2.8,  # Sodium, potassium replenishment
                    'water': 3.0,  # Rehydration CRITICAL (1L per drink)
                    'glucose': 2.0,  # Blood sugar restoration
                    'cysteine': 2.3,  # Acetaldehyde metabolism
                    'magnesium': 2.0  # Depleted by alcohol
                },
                harmful_molecules={
                    'more_alcohol': 2.8,  # "Hair of the dog" delays recovery
                    'caffeine': 2.0,  # Dehydrating (unless with water)
                    'acetaminophen': 3.0,  # Tylenol + alcohol = liver damage
                    'greasy_food': 1.5  # Myth (doesn't help after)
                },
                optimal_ranges={
                    'water': (2, 4),  # L (1L per alcoholic drink)
                    'nac': (600, 1200),  # mg
                    'vitamin_c': (1000, 2000),  # mg
                    'b_complex': (50, 100),  # mg (B1, B6, B12)
                    'electrolytes': (1000, 2000),  # mg sodium
                    'glucose': (20, 40)  # g (honey, juice)
                },
                priority_bonds=[ChemicalBondType.O_H, ChemicalBondType.N_H],
                key_metrics=['hydration_status', 'liver_enzymes', 'acetaldehyde_clearance']
            )
            
            # Jet Lag Recovery
            self.profiles[HealthGoal.JET_LAG_RECOVERY] = GoalMolecularProfile(
                goal=HealthGoal.JET_LAG_RECOVERY,
                target_molecules={
                    'melatonin': 2.8,  # Circadian reset (0.5-5mg)
                    'caffeine': 2.5,  # Strategic timing (AM destination)
                    'protein': 2.3,  # Breakfast at destination time
                    'light_exposure': 2.7,  # Sunlight or blue light
                    'water': 2.5,  # Hydration (flight dehydration)
                    'magnesium': 2.0,  # Sleep quality (400mg)
                    'b_vitamins': 2.0  # Energy metabolism
                },
                harmful_molecules={
                    'alcohol': 2.8,  # Disrupts sleep quality
                    'large_meals': 2.3,  # Digestive burden
                    'caffeine_evening': 2.7,  # Delays circadian shift
                    'blue_light_night': 2.5  # Suppresses melatonin
                },
                optimal_ranges={
                    'melatonin': (0.5, 5),  # mg (destination bedtime)
                    'caffeine': (100, 300),  # mg (AM destination time)
                    'water': (3, 5),  # L/day (flight + destination)
                    'protein_breakfast': (30, 50),  # g (anchor circadian)
                    'light_exposure': (10000, 10000)  # lux (bright light therapy)
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.AROMATIC],
                key_metrics=['sleep_latency', 'sleep_quality', 'cortisol_rhythm', 'alertness']
            )
            
            # Altitude Adaptation (High-Altitude Performance)
            self.profiles[HealthGoal.ALTITUDE_ADAPTATION] = GoalMolecularProfile(
                goal=HealthGoal.ALTITUDE_ADAPTATION,
                target_molecules={
                    'iron': 2.8,  # Hemoglobin production (18-30mg)
                    'vitamin_c': 2.5,  # Iron absorption + antioxidant
                    'beetroot': 2.7,  # Nitrates, O2 efficiency
                    'antioxidants': 2.5,  # Oxidative stress (altitude)
                    'carbs': 2.8,  # Primary fuel (O2 efficient)
                    'water': 3.0,  # Increased fluid loss (3-5L)
                    'vitamin_e': 2.3,  # Antioxidant
                    'rhodiola': 2.0  # Adaptogen
                },
                harmful_molecules={
                    'alcohol': 2.8,  # Exacerbates altitude sickness
                    'caffeine': 2.0,  # Dehydrating (moderate OK)
                    'sleeping_pills': 2.5,  # Respiratory depression
                    'fat': 2.0,  # Less O2 efficient than carbs
                },
                optimal_ranges={
                    'carbs': (6, 10),  # g/kg (O2-efficient fuel)
                    'iron': (18, 30),  # mg/day (if deficient)
                    'water': (4, 6),  # L/day (increased loss)
                    'beetroot': (500, 1000),  # mg nitrates
                    'antioxidants': (15000, 25000),  # ORAC units
                    'sodium': (3000, 5000)  # mg/day (with water)
                },
                priority_bonds=[ChemicalBondType.C_H, ChemicalBondType.O_H, ChemicalBondType.N_O],
                key_metrics=['spo2', 'hemoglobin', 'hematocrit', 'vo2_max_altitude']
            )
            
            # Cold Tolerance (Cold Exposure Optimization)
            self.profiles[HealthGoal.COLD_TOLERANCE] = GoalMolecularProfile(
                goal=HealthGoal.COLD_TOLERANCE,
                target_molecules={
                    'calories': 2.8,  # Thermogenesis (3000-4000 kcal)
                    'fat': 2.7,  # Brown adipose tissue fuel
                    'carbs': 2.5,  # Shivering thermogenesis
                    'iron': 2.3,  # Thyroid function (cold adaptation)
                    'iodine': 2.3,  # Thyroid thermogenesis
                    'caffeine': 2.0,  # Thermogenic
                    'capsaicin': 2.0,  # Thermogenic (hot peppers)
                    'omega_3': 2.0  # Brown fat activation
                },
                harmful_molecules={
                    'low_calories': 2.8,  # Inadequate thermogenesis
                    'low_fat': 2.5,  # Insufficient fuel
                    'alcohol': 2.7,  # Vasodilation (heat loss)
                    'dehydration': 2.3  # Reduces blood volume
                },
                optimal_ranges={
                    'calories': (3000, 4500),  # kcal/day (cold exposure)
                    'fat': (100, 150),  # g/day
                    'carbs': (300, 500),  # g/day (shivering fuel)
                    'iron': (18, 30),  # mg/day (thyroid)
                    'iodine': (150, 300)  # mcg/day (thyroid)
                },
                priority_bonds=[ChemicalBondType.C_H, ChemicalBondType.C_C],
                key_metrics=['body_temperature', 'shivering_threshold', 'brown_fat_activity', 'metabolic_rate']
            )
            
            # Heat Tolerance (Hot Climate Adaptation)
            self.profiles[HealthGoal.HEAT_TOLERANCE] = GoalMolecularProfile(
                goal=HealthGoal.HEAT_TOLERANCE,
                target_molecules={
                    'water': 3.0,  # CRITICAL (5-10L/day if active)
                    'sodium': 3.0,  # Sweat losses (5000-10000mg)
                    'potassium': 2.7,  # Electrolyte balance
                    'magnesium': 2.5,  # Cramp prevention
                    'carbs': 2.3,  # Energy (heat stress metabolic)
                    'antioxidants': 2.0,  # Heat shock proteins
                    'beta_alanine': 1.8  # Heat stress buffering
                },
                harmful_molecules={
                    'alcohol': 3.0,  # Dehydrating, heat illness risk
                    'caffeine': 2.0,  # Dehydrating (moderate OK with water)
                    'high_protein': 2.3,  # Increased metabolic heat
                    'spicy_foods': 1.5  # Increases body temperature
                },
                optimal_ranges={
                    'water': (5, 10),  # L/day (sweat-dependent)
                    'sodium': (5000, 10000),  # mg/day (sweat losses)
                    'potassium': (4000, 5000),  # mg/day
                    'magnesium': (500, 800),  # mg/day
                    'carbs': (5, 8)  # g/kg (energy + glucose for cooling)
                },
                priority_bonds=[ChemicalBondType.O_H],
                key_metrics=['core_temperature', 'sweat_rate', 'hydration_status', 'heart_rate_variability']
            )
            
            # Wound Healing Acceleration
            self.profiles[HealthGoal.WOUND_HEALING] = GoalMolecularProfile(
                goal=HealthGoal.WOUND_HEALING,
                target_molecules={
                    'protein': 3.0,  # CRITICAL (1.5-2.5g/kg)
                    'vitamin_c': 2.8,  # Collagen synthesis (1000-2000mg)
                    'zinc': 2.7,  # Wound healing (30-50mg)
                    'arginine': 2.5,  # Nitric oxide, immune (10-15g)
                    'glutamine': 2.3,  # Immune, gut barrier (20-30g)
                    'vitamin_a': 2.5,  # Epithelialization (10000-25000 IU)
                    'copper': 2.0,  # Collagen cross-linking
                    'iron': 2.0  # Hemoglobin, O2 delivery
                },
                harmful_molecules={
                    'smoking': 3.0,  # Vasoconstriction, hypoxia
                    'alcohol': 2.8,  # Impairs healing
                    'steroids': 2.5,  # Anti-inflammatory (delays healing)
                    'excess_sugar': 2.5,  # Impairs immune function
                },
                optimal_ranges={
                    'protein': (1.5, 2.5),  # g/kg (wound-dependent)
                    'vitamin_c': (1000, 2000),  # mg/day
                    'zinc': (30, 50),  # mg/day (SHORT-TERM)
                    'arginine': (10, 15),  # g/day
                    'glutamine': (20, 30),  # g/day
                    'vitamin_a': (10000, 25000),  # IU/day (SHORT-TERM)
                    'calories': (2500, 3500)  # kcal/day (healing metabolic demand)
                },
                priority_bonds=[ChemicalBondType.N_H, ChemicalBondType.O_H, ChemicalBondType.S_H],
                key_metrics=['wound_closure_rate', 'collagen_deposition', 'infection_risk', 'scar_formation']
            )
            
            # Scar Reduction (Minimize Scarring)
            self.profiles[HealthGoal.SCAR_REDUCTION] = GoalMolecularProfile(
                goal=HealthGoal.SCAR_REDUCTION,
                target_molecules={
                    'vitamin_e': 2.5,  # Topical + oral (400 IU)
                    'vitamin_c': 2.5,  # Collagen remodeling (1000mg)
                    'silica': 2.3,  # Connective tissue (10-30mg)
                    'zinc': 2.0,  # Wound maturation (15-25mg)
                    'copper_peptides': 2.0,  # Collagen remodeling
                    'omega_3': 2.0,  # Anti-inflammatory
                    'allantoin': 1.8,  # Cell proliferation
                    'centella_asiatica': 2.0  # Gotu kola, collagen
                },
                harmful_molecules={
                    'smoking': 3.0,  # Impairs collagen remodeling
                    'sun_exposure': 2.8,  # Hyperpigmentation
                    'alcohol': 2.3,  # Impairs healing
                    'vitamin_a_excess': 2.0  # Can worsen keloids
                },
                optimal_ranges={
                    'vitamin_e': (400, 800),  # IU/day (topical + oral)
                    'vitamin_c': (1000, 2000),  # mg/day
                    'silica': (10, 30),  # mg/day
                    'zinc': (15, 25),  # mg/day
                    'omega_3': (2, 3),  # g/day
                    'protein': (1.0, 1.5)  # g/kg (collagen remodeling)
                },
                priority_bonds=[ChemicalBondType.O_H, ChemicalBondType.N_H],
                key_metrics=['scar_thickness', 'collagen_alignment', 'pigmentation', 'elasticity']
            )
        
        logger.info(f"Initialized {len(self.profiles)} goal profiles")
    
    def get_profile(self, goal: 'HealthGoal') -> Optional[GoalMolecularProfile]:
        """Get molecular profile for a goal"""
        return self.profiles.get(goal)


# ============================================================================
# MULTI-CONDITION OPTIMIZER
# ============================================================================

class MultiConditionOptimizer:
    """
    Multi-Condition Recommendation Engine
    
    Solves the complex optimization problem:
    - Maximize health goal molecules
    - Minimize disease risk molecules
    - Satisfy nutrient constraints
    - Balance conflicting objectives
    
    This is the "Digital Dietitian" brain.
    """
    
    def __init__(self):
        self.disease_db = DiseaseProfileDatabase()
        self.goal_db = GoalProfileDatabase()
        self.strategy = OptimizationStrategy.BALANCED
        
        logger.info("Multi-Condition Optimizer initialized")
    
    def optimize(self, 
                 food_profile: 'NutrientMolecularBreakdown',
                 bonds: List['MolecularBondProfile'],
                 toxics: List['ToxicContaminantProfile'],
                 user_profile: 'UserHealthProfile',
                 food_name: str = "Unknown",
                 scan_id: str = "SCAN_000") -> 'FoodRecommendation':
        """
        Complete multi-condition optimization
        
        Args:
            food_profile: Molecular breakdown of food
            bonds: Chemical bonds detected
            toxics: Toxic contaminants
            user_profile: User's complete health profile
            food_name: Name of food
            scan_id: Scan ID
        
        Returns:
            Personalized food recommendation
        """
        logger.info(f"Optimizing for user {user_profile.user_id}: {food_name}")
        
        # Step 1: Safety Check (highest priority)
        safety_score, safety_warnings = self._safety_check(toxics)
        
        if safety_score < 20:
            # Critical safety issue - immediate AVOID
            return self._create_avoid_recommendation(
                food_name, scan_id, food_profile, bonds, toxics,
                "CRITICAL SAFETY ISSUE - Toxic contaminants detected",
                safety_score
            )
        
        # Step 2: Goal Alignment Score
        goal_score, goal_reasoning = self._calculate_goal_score(
            food_profile, bonds, user_profile
        )
        
        # Step 3: Disease Compatibility Score
        disease_score, disease_reasoning = self._calculate_disease_score(
            food_profile, bonds, user_profile
        )
        
        # Step 4: Calculate Overall Score (weighted combination)
        overall_score = self._calculate_overall_score(
            safety_score, goal_score, disease_score, user_profile
        )
        
        # Step 5: Determine Recommendation Level
        rec_level = self._determine_recommendation_level(overall_score)
        
        # Step 6: Generate Reasoning
        all_reasoning = []
        all_reasoning.extend(safety_warnings)
        all_reasoning.extend(goal_reasoning)
        all_reasoning.extend(disease_reasoning)
        
        # Step 7: Find Better Alternatives (if not highly recommended)
        alternatives = []
        if overall_score < 80:
            alternatives = self._find_better_alternatives(
                food_profile, user_profile
            )
        
        # Step 8: Serving Size Recommendations
        optimal_serving, max_servings = self._calculate_serving_recommendations(
            food_profile, user_profile, overall_score
        )
        
        # Create recommendation
        recommendation = FoodRecommendation(
            food_name=food_name,
            scan_id=scan_id,
            overall_score=overall_score,
            safety_score=safety_score,
            goal_alignment_score=goal_score,
            disease_compatibility_score=disease_score,
            molecular_breakdown=food_profile,
            bond_profile=bonds,
            toxic_profile=toxics,
            recommendation=rec_level.value,
            reasoning=all_reasoning,
            optimal_serving_size=optimal_serving,
            max_daily_servings=max_servings,
            better_alternatives=alternatives
        )
        
        logger.info(f"Recommendation: {rec_level.value} (Score: {overall_score:.1f}/100)")
        
        return recommendation
    
    def _safety_check(self, toxics: List['ToxicContaminantProfile']) -> Tuple[float, List[str]]:
        """
        Safety check - highest priority
        
        Returns:
            (safety_score, warnings)
        """
        if not toxics:
            return 100.0, []
        
        warnings = []
        penalties = []
        
        for toxic in toxics:
            if toxic.risk_level == "CRITICAL":
                penalties.append(80)  # Massive penalty
                warnings.append(f"⚠️ CRITICAL: {toxic.element.value} at {toxic.concentration:.3f} ppm "
                              f"(limit: {toxic.safe_threshold} ppm). {toxic.health_impact}")
            elif toxic.risk_level == "HIGH":
                penalties.append(50)
                warnings.append(f"⚠️ HIGH RISK: {toxic.element.value} exceeds safe limits. Avoid regular consumption.")
            elif toxic.risk_level == "MODERATE":
                penalties.append(20)
                warnings.append(f"⚠️ MODERATE: {toxic.element.value} detected at {toxic.concentration:.3f} ppm. Limit intake.")
            elif toxic.risk_level == "LOW":
                penalties.append(5)
                warnings.append(f"✓ Low levels of {toxic.element.value} detected (within safe limits).")
        
        # Safety score = 100 - total penalties
        safety_score = max(0, 100 - sum(penalties))
        
        return safety_score, warnings
    
    def _calculate_goal_score(self, 
                              food_profile: 'NutrientMolecularBreakdown',
                              bonds: List['MolecularBondProfile'],
                              user_profile: 'UserHealthProfile') -> Tuple[float, List[str]]:
        """
        Calculate how well food aligns with user's health goals
        
        Returns:
            (goal_score, reasoning)
        """
        if not PROFILER_AVAILABLE:
            return 50.0, ["Goal analysis unavailable"]
        
        # Get goal profiles
        primary_goal_profile = self.goal_db.get_profile(user_profile.primary_goal)
        
        if not primary_goal_profile:
            return 50.0, ["Goal profile not found"]
        
        score = 50.0  # Start neutral
        reasoning = []
        
        # Analyze target molecules
        for molecule, importance in primary_goal_profile.target_molecules.items():
            value = self._get_molecule_value(molecule, food_profile)
            
            if value > 0:
                # Calculate contribution (normalized)
                contribution = min(30, value * importance)
                score += contribution
                
                if contribution > 5:
                    reasoning.append(f"✓ {molecule.title()}: {value:.1f} units - "
                                   f"Supports {user_profile.primary_goal.value} (+{contribution:.0f} pts)")
        
        # Check priority bonds
        for bond in bonds:
            if bond.bond_type in primary_goal_profile.priority_bonds:
                bonus = bond.confidence * 10
                score += bonus
                reasoning.append(f"✓ {bond.bond_type.value} bonds detected - "
                               f"Essential for {user_profile.primary_goal.value} (+{bonus:.0f} pts)")
        
        # Cap score at 100
        score = min(100, score)
        
        return score, reasoning
    
    def _calculate_disease_score(self,
                                 food_profile: 'NutrientMolecularBreakdown',
                                 bonds: List['MolecularBondProfile'],
                                 user_profile: 'UserHealthProfile') -> Tuple[float, List[str]]:
        """
        Calculate disease compatibility score
        
        Returns:
            (disease_score, reasoning)
        """
        if not user_profile.diagnosed_conditions:
            return 100.0, ["No disease conditions to manage"]
        
        if not PROFILER_AVAILABLE:
            return 50.0, ["Disease analysis unavailable"]
        
        score = 100.0  # Start perfect, subtract penalties
        reasoning = []
        
        for condition in user_profile.diagnosed_conditions:
            disease_profile = self.disease_db.get_profile(condition)
            
            if not disease_profile:
                continue
            
            # Check harmful molecules
            for molecule, risk_weight in disease_profile.harmful_molecules.items():
                value = self._get_molecule_value(molecule, food_profile)
                
                if value > 0:
                    # Calculate penalty
                    penalty = value * risk_weight * disease_profile.severity_multiplier
                    score -= penalty
                    
                    if penalty > 5:
                        reasoning.append(f"❌ {molecule.title()}: {value:.1f} units - "
                                       f"HARMFUL for {condition.value} (-{penalty:.0f} pts)")
            
            # Check beneficial molecules
            for molecule, benefit_weight in disease_profile.beneficial_molecules.items():
                value = self._get_molecule_value(molecule, food_profile)
                
                if value > 0:
                    bonus = value * benefit_weight * 0.5
                    score += bonus
                    
                    if bonus > 3:
                        reasoning.append(f"✓ {molecule.title()}: {value:.1f} units - "
                                       f"BENEFICIAL for {condition.value} (+{bonus:.0f} pts)")
            
            # Check harmful bonds
            for bond in bonds:
                if bond.bond_type in disease_profile.harmful_bonds:
                    penalty = bond.concentration * 0.1 * disease_profile.severity_multiplier
                    score -= penalty
                    reasoning.append(f"❌ {bond.bond_type.value} bonds - "
                                   f"Problematic for {condition.value} (-{penalty:.0f} pts)")
        
        # Ensure score stays in bounds
        score = max(0, min(100, score))
        
        return score, reasoning
    
    def _get_molecule_value(self, molecule: str, food_profile: 'NutrientMolecularBreakdown') -> float:
        """Extract molecule value from food profile"""
        # Map molecule names to food profile attributes
        mapping = {
            'fiber': food_profile.fiber,
            'protein': food_profile.total_protein,
            'carbohydrates': food_profile.total_carbs,
            'simple_sugars': food_profile.simple_sugars,
            'fat': food_profile.total_fat,
            'saturated_fat': food_profile.saturated_fat,
            'polyphenols': food_profile.polyphenols,
            'omega_3': food_profile.amino_acids.get('omega_3', 0),  # Simplified
            'sodium': 0,  # Would need to add to profile
            'potassium': 0,  # Would need to add to profile
        }
        
        return mapping.get(molecule, 0)
    
    def _calculate_overall_score(self,
                                safety_score: float,
                                goal_score: float,
                                disease_score: float,
                                user_profile: 'UserHealthProfile') -> float:
        """
        Calculate weighted overall score
        
        Weights depend on user's situation:
        - Has diseases → disease score weighted higher
        - Healthy → goal score weighted higher
        - Safety always critical
        """
        # Base weights
        safety_weight = 0.4  # Always 40%
        
        if user_profile.diagnosed_conditions:
            # Has diseases - prioritize disease management
            goal_weight = 0.2
            disease_weight = 0.4
        else:
            # Healthy - prioritize goals
            goal_weight = 0.4
            disease_weight = 0.2
        
        overall = (
            safety_score * safety_weight +
            goal_score * goal_weight +
            disease_score * disease_weight
        )
        
        return overall
    
    def _determine_recommendation_level(self, score: float) -> RecommendationLevel:
        """Determine recommendation level from score"""
        if score >= 80:
            return RecommendationLevel.HIGHLY_RECOMMENDED
        elif score >= 60:
            return RecommendationLevel.RECOMMENDED
        elif score >= 40:
            return RecommendationLevel.ACCEPTABLE
        elif score >= 20:
            return RecommendationLevel.NOT_RECOMMENDED
        else:
            return RecommendationLevel.AVOID
    
    def _find_better_alternatives(self,
                                  food_profile: 'NutrientMolecularBreakdown',
                                  user_profile: 'UserHealthProfile') -> List[str]:
        """Find better food alternatives"""
        alternatives = []
        
        # This would query food database in production
        # For now, return generic suggestions based on goals
        
        if not PROFILER_AVAILABLE:
            return alternatives
        
        goal = user_profile.primary_goal
        
        if goal == HealthGoal.BRAIN:
            alternatives = ["Walnuts (high DHA)", "Blueberries (polyphenols)", 
                          "Avocado (healthy fats)", "Salmon (omega-3)"]
        elif goal == HealthGoal.MUSCLE:
            alternatives = ["Chicken breast (lean protein)", "Greek yogurt (protein + probiotics)",
                          "Eggs (complete protein)", "Quinoa (complete plant protein)"]
        elif goal == HealthGoal.ENERGY:
            alternatives = ["Oats (slow carbs)", "Sweet potato (complex carbs)",
                          "Bananas (quick energy)", "Brown rice (sustained energy)"]
        elif goal == HealthGoal.HEART:
            alternatives = ["Salmon (omega-3)", "Olive oil (MUFA)",
                          "Nuts (healthy fats)", "Leafy greens (nitrates)"]
        
        return alternatives[:3]  # Return top 3
    
    def _calculate_serving_recommendations(self,
                                          food_profile: 'NutrientMolecularBreakdown',
                                          user_profile: 'UserHealthProfile',
                                          score: float) -> Tuple[Optional[float], Optional[int]]:
        """Calculate optimal serving size and max daily servings"""
        # Default serving: 100g
        optimal_serving = 100.0
        
        # Adjust based on score
        if score < 40:
            optimal_serving = 50.0  # Small portion only
            max_servings = 1
        elif score < 60:
            optimal_serving = 75.0
            max_servings = 2
        elif score < 80:
            optimal_serving = 100.0
            max_servings = 3
        else:
            optimal_serving = 150.0
            max_servings = 5
        
        return optimal_serving, max_servings
    
    def _create_avoid_recommendation(self,
                                    food_name: str,
                                    scan_id: str,
                                    food_profile: 'NutrientMolecularBreakdown',
                                    bonds: List['MolecularBondProfile'],
                                    toxics: List['ToxicContaminantProfile'],
                                    reason: str,
                                    safety_score: float) -> 'FoodRecommendation':
        """Create an AVOID recommendation"""
        if not PROFILER_AVAILABLE:
            return None
        
        return FoodRecommendation(
            food_name=food_name,
            scan_id=scan_id,
            overall_score=0.0,
            safety_score=safety_score,
            goal_alignment_score=0.0,
            disease_compatibility_score=0.0,
            molecular_breakdown=food_profile,
            bond_profile=bonds,
            toxic_profile=toxics,
            recommendation=RecommendationLevel.AVOID.value,
            reasoning=[f"🛑 {reason}"],
            optimal_serving_size=0.0,
            max_daily_servings=0,
            better_alternatives=["Choose certified organic alternatives", 
                                "Select foods from trusted sources"]
        )


# ============================================================================
# TESTING & EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-CONDITION OPTIMIZER - Test Suite")
    print("=" * 80)
    
    if not PROFILER_AVAILABLE:
        print("\n⚠️  atomic_molecular_profiler.py not found. Cannot run tests.")
        print("Please ensure atomic_molecular_profiler.py is in the same directory.")
        exit(1)
    
    from atomic_molecular_profiler import (  # type: ignore
        NutrientMolecularBreakdown, UserHealthProfile, MolecularBondProfile,
        ToxicContaminantProfile, HealthGoal, DiseaseCondition, ChemicalBondType,
        ToxicElement
    )
    
    # Initialize optimizer
    optimizer = MultiConditionOptimizer()
    
    # Test 1: High-sugar food + Diabetic user
    print("\n" + "=" * 80)
    print("TEST 1: High-Sugar Food + Type 2 Diabetic User")
    print("=" * 80)
    
    # Create user profile
    diabetic_user = UserHealthProfile(
        user_id="USER001",
        age=55,
        sex="M",
        primary_goal=HealthGoal.BRAIN,
        diagnosed_conditions=[DiseaseCondition.DIABETES_T2, DiseaseCondition.HYPERTENSION],
        weight_kg=85,
        blood_glucose_mg_dl=140
    )
    
    # Create food profile (sugary dessert)
    dessert_profile = NutrientMolecularBreakdown(
        total_carbs=60.0,
        simple_sugars=50.0,  # Very high sugar
        total_protein=2.0,
        total_fat=15.0,
        saturated_fat=10.0,
        fiber=1.0,
        water_content=20.0,
        total_calories=380
    )
    
    # Bonds (high C-H from sugar and fat)
    dessert_bonds = [
        MolecularBondProfile(
            bond_type=ChemicalBondType.C_H,
            concentration=70.0,  # High
            bond_strength=413,
            spectral_signature=np.array([0.8] * 100),
            confidence=0.92
        )
    ]
    
    # No toxics
    dessert_toxics = []
    
    # Optimize
    rec = optimizer.optimize(
        dessert_profile, dessert_bonds, dessert_toxics,
        diabetic_user, "Chocolate Cake", "SCAN001"
    )
    
    print(f"\n📊 Recommendation for {diabetic_user.user_id}:")
    print(f"  Overall Score: {rec.overall_score:.1f}/100")
    print(f"  Recommendation: {rec.recommendation}")
    print(f"\n  Reasoning:")
    for reason in rec.reasoning:
        print(f"    {reason}")
    print(f"\n  Better Alternatives:")
    for alt in rec.better_alternatives:
        print(f"    • {alt}")
    
    # Test 2: Brain-healthy food + Brain health goal
    print("\n" + "=" * 80)
    print("TEST 2: Omega-3 Rich Food + Brain Health Goal")
    print("=" * 80)
    
    brain_user = UserHealthProfile(
        user_id="USER002",
        age=45,
        sex="F",
        primary_goal=HealthGoal.BRAIN,
        diagnosed_conditions=[],
        weight_kg=65
    )
    
    # Salmon profile
    salmon_profile = NutrientMolecularBreakdown(
        total_protein=25.0,
        total_fat=15.0,
        polyunsaturated_fat=8.0,  # High omega-3
        total_carbs=0.0,
        fiber=0.0,
        water_content=60.0,
        polyphenols=5.0,
        total_calories=230
    )
    
    salmon_profile.amino_acids['omega_3'] = 3.5  # 3.5g omega-3
    
    salmon_bonds = [
        MolecularBondProfile(
            bond_type=ChemicalBondType.N_H,
            concentration=25.0,
            bond_strength=391,
            spectral_signature=np.array([0.9] * 100),
            confidence=0.95
        ),
        MolecularBondProfile(
            bond_type=ChemicalBondType.DOUBLE_BOND,
            concentration=8.0,
            bond_strength=614,
            spectral_signature=np.array([0.85] * 100),
            confidence=0.88
        )
    ]
    
    salmon_toxics = []
    
    rec2 = optimizer.optimize(
        salmon_profile, salmon_bonds, salmon_toxics,
        brain_user, "Wild Salmon", "SCAN002"
    )
    
    print(f"\n📊 Recommendation for {brain_user.user_id}:")
    print(f"  Overall Score: {rec2.overall_score:.1f}/100")
    print(f"  Recommendation: {rec2.recommendation}")
    print(f"  Optimal Serving: {rec2.optimal_serving_size:.0f}g")
    print(f"  Max Daily Servings: {rec2.max_daily_servings}")
    print(f"\n  Reasoning:")
    for reason in rec2.reasoning:
        print(f"    {reason}")
    
    # Test 3: Contaminated food
    print("\n" + "=" * 80)
    print("TEST 3: Mercury-Contaminated Fish")
    print("=" * 80)
    
    contaminated_toxics = [
        ToxicContaminantProfile(
            element=ToxicElement.MERCURY,
            concentration=1.2,  # 1.2 ppm (exceeds 0.5 ppm limit)
            detection_limit=0.001,
            safe_threshold=0.5,
            exceeds_limit=True,
            confidence=0.85,
            risk_level="HIGH",
            health_impact="Neurotoxic. Damages brain, kidneys, nervous system."
        )
    ]
    
    rec3 = optimizer.optimize(
        salmon_profile, salmon_bonds, contaminated_toxics,
        brain_user, "Contaminated Tuna", "SCAN003"
    )
    
    print(f"\n📊 Recommendation for {brain_user.user_id}:")
    print(f"  Overall Score: {rec3.overall_score:.1f}/100")
    print(f"  Safety Score: {rec3.safety_score:.1f}/100")
    print(f"  Recommendation: {rec3.recommendation}")
    print(f"\n  Reasoning:")
    for reason in rec3.reasoning:
        print(f"    {reason}")
    
    print("\n" + "=" * 80)
    print("Multi-Condition Optimizer Test Complete!")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Multi-disease optimization (Diabetes + Hypertension)")
    print("  ✓ Goal-based recommendations (Brain health)")
    print("  ✓ Safety checks (Toxic contaminant detection)")
    print("  ✓ Personalized scoring system")
    print("  ✓ Alternative food suggestions")
    print("\nNext: Build lifecycle_modulator.py for age-based optimization")
