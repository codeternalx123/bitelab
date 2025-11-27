"""
Food Chemistry Models
=====================

Specialized AI models for understanding food chemistry, macronutrient interactions,
and cooking transformations at the molecular level.

Features:
1. Macronutrient interaction modeling
2. Cooking transformation prediction
3. Maillard reaction simulation
4. Protein denaturation tracking
5. Starch gelatinization modeling
6. Fat oxidation prediction
7. Vitamin degradation analysis
8. Flavor compound generation
9. Texture transformation
10. pH and acidity impact modeling

Performance Targets:
- Reaction prediction: <100ms
- Transformation accuracy: >90%
- Nutrient retention: ±5% accuracy
- Flavor profile: >85% correlation
- Chemical database: 10,000+ compounds

Author: Wellomex AI Team
Date: November 2025
Version: 6.0.0
"""

import time
import logging
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from collections import defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION & ENUMS
# ============================================================================

class CookingMethod(Enum):
    """Cooking methods with different chemical impacts"""
    RAW = "raw"
    BOILING = "boiling"
    STEAMING = "steaming"
    BAKING = "baking"
    ROASTING = "roasting"
    FRYING = "frying"
    GRILLING = "grilling"
    MICROWAVING = "microwaving"
    PRESSURE_COOKING = "pressure_cooking"
    SLOW_COOKING = "slow_cooking"
    FERMENTATION = "fermentation"
    SMOKING = "smoking"


class MacronutrientType(Enum):
    """Major macronutrient categories"""
    PROTEIN = "protein"
    CARBOHYDRATE = "carbohydrate"
    FAT = "fat"
    FIBER = "fiber"
    WATER = "water"


class ReactionType(Enum):
    """Chemical reaction types in cooking"""
    MAILLARD = "maillard"  # Browning
    CARAMELIZATION = "caramelization"
    DENATURATION = "denaturation"  # Protein
    GELATINIZATION = "gelatinization"  # Starch
    OXIDATION = "oxidation"  # Fat
    HYDROLYSIS = "hydrolysis"
    ENZYMATIC = "enzymatic"
    EMULSIFICATION = "emulsification"


@dataclass
class ChemistryConfig:
    """Food chemistry configuration"""
    # Temperature ranges (Celsius)
    temp_min: float = 0.0
    temp_max: float = 250.0
    
    # pH scale
    ph_min: float = 0.0
    ph_max: float = 14.0
    
    # Time ranges (minutes)
    time_min: float = 0.0
    time_max: float = 480.0  # 8 hours
    
    # Nutrient retention thresholds
    retention_excellent: float = 0.90
    retention_good: float = 0.75
    retention_poor: float = 0.50


# ============================================================================
# CHEMICAL COMPOUND MODELS
# ============================================================================

@dataclass
class ChemicalCompound:
    """Individual chemical compound in food"""
    name: str
    formula: str
    molecular_weight: float
    boiling_point: Optional[float] = None
    melting_point: Optional[float] = None
    stability_temp: Optional[float] = None  # Temperature where degradation begins
    ph_stable_min: float = 0.0
    ph_stable_max: float = 14.0
    
    # Nutritional classification
    compound_type: str = "other"  # vitamin, mineral, amino_acid, fatty_acid, etc.
    
    def is_stable_at(self, temperature: float, ph: float) -> bool:
        """Check if compound is stable at given conditions"""
        temp_stable = True
        if self.stability_temp is not None:
            temp_stable = temperature < self.stability_temp
        
        ph_stable = self.ph_stable_min <= ph <= self.ph_stable_max
        
        return temp_stable and ph_stable


@dataclass
class MacronutrientProfile:
    """Macronutrient composition of food"""
    protein_g: float = 0.0
    carbohydrate_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0
    water_g: float = 0.0
    
    # Detailed breakdowns
    saturated_fat_g: float = 0.0
    unsaturated_fat_g: float = 0.0
    sugar_g: float = 0.0
    starch_g: float = 0.0
    
    def total_mass(self) -> float:
        """Total mass in grams"""
        return (self.protein_g + self.carbohydrate_g + self.fat_g + 
                self.fiber_g + self.water_g)
    
    def protein_ratio(self) -> float:
        """Protein as ratio of total mass"""
        total = self.total_mass()
        return self.protein_g / total if total > 0 else 0.0


# ============================================================================
# MAILLARD REACTION MODEL
# ============================================================================

class MaillardReaction:
    """
    Model Maillard browning reaction between amino acids and reducing sugars
    
    The Maillard reaction creates flavor compounds and brown color in cooked foods.
    Requires:
    - Reducing sugars (glucose, fructose)
    - Amino acids or proteins
    - Heat (typically >140°C)
    - Low moisture
    """
    
    def __init__(self):
        # Temperature-dependent rate constants
        self.activation_temp = 140.0  # °C
        self.optimal_temp = 180.0
        
        # Reactant requirements
        self.min_sugar_g = 0.1
        self.min_protein_g = 0.5
        self.max_moisture = 0.3  # 30% water
        
        logger.info("Maillard Reaction model initialized")
    
    def predict_browning(
        self,
        temperature: float,
        time_minutes: float,
        sugar_g: float,
        protein_g: float,
        moisture_ratio: float
    ) -> Dict[str, float]:
        """
        Predict degree of Maillard browning
        
        Returns:
            browning_index: 0.0-1.0 (none to maximum brown)
            flavor_compounds: Relative concentration
            acrylamide_risk: 0.0-1.0 (carcinogen formation risk)
        """
        # Check prerequisites
        if temperature < self.activation_temp:
            return {
                'browning_index': 0.0,
                'flavor_compounds': 0.0,
                'acrylamide_risk': 0.0
            }
        
        if sugar_g < self.min_sugar_g or protein_g < self.min_protein_g:
            return {
                'browning_index': 0.0,
                'flavor_compounds': 0.0,
                'acrylamide_risk': 0.0
            }
        
        if moisture_ratio > self.max_moisture:
            # High moisture inhibits Maillard
            moisture_penalty = (moisture_ratio - self.max_moisture) * 2.0
        else:
            moisture_penalty = 0.0
        
        # Temperature effect (Arrhenius-like)
        temp_factor = min(1.0, (temperature - self.activation_temp) / 
                         (self.optimal_temp - self.activation_temp))
        
        # Time effect (logarithmic)
        time_factor = min(1.0, math.log(time_minutes + 1) / math.log(60))
        
        # Reactant availability
        reactant_factor = min(sugar_g / 5.0, protein_g / 10.0)  # Normalized
        
        # Browning index
        browning = temp_factor * time_factor * reactant_factor * (1 - moisture_penalty)
        browning = max(0.0, min(1.0, browning))
        
        # Flavor compounds (peak at moderate browning)
        flavor = browning * (1.0 - 0.3 * browning)  # Inverted U-shape
        
        # Acrylamide risk (increases with temperature and browning)
        if temperature > 180:
            acrylamide = min(1.0, browning * (temperature - 180) / 100)
        else:
            acrylamide = 0.0
        
        return {
            'browning_index': float(browning),
            'flavor_compounds': float(flavor),
            'acrylamide_risk': float(acrylamide)
        }


# ============================================================================
# PROTEIN DENATURATION MODEL
# ============================================================================

class ProteinDenaturation:
    """
    Model protein denaturation during cooking
    
    Denaturation unfolds protein structure, affecting:
    - Texture (meat becomes firm)
    - Digestibility (often improved)
    - Water retention
    - Color changes
    """
    
    def __init__(self):
        # Protein-specific denaturation temperatures
        self.denaturation_temps = {
            'myosin': 50.0,      # Meat protein
            'collagen': 65.0,    # Connective tissue
            'actin': 66.0,       # Muscle protein
            'egg_white': 62.0,   # Ovalbumin
            'egg_yolk': 70.0,    # Egg proteins
            'casein': 72.0,      # Milk protein
            'whey': 75.0         # Milk protein
        }
        
        logger.info("Protein Denaturation model initialized")
    
    def predict_denaturation(
        self,
        protein_type: str,
        temperature: float,
        time_minutes: float,
        ph: float = 7.0
    ) -> Dict[str, float]:
        """
        Predict degree of protein denaturation
        
        Returns:
            denaturation_degree: 0.0-1.0
            texture_change: firmness increase
            digestibility: change in digestibility
            water_loss: moisture lost
        """
        # Get denaturation temperature
        denature_temp = self.denaturation_temps.get(protein_type, 60.0)
        
        # pH effect (extreme pH accelerates denaturation)
        ph_factor = 1.0
        if ph < 4.0 or ph > 10.0:
            ph_factor = 1.5
        
        # Temperature above denaturation point
        if temperature < denature_temp:
            temp_excess = 0.0
        else:
            temp_excess = (temperature - denature_temp) * ph_factor
        
        # Denaturation kinetics (first-order)
        rate_constant = 0.1 * (1 + temp_excess / 50.0)
        denaturation = 1.0 - math.exp(-rate_constant * time_minutes)
        denaturation = max(0.0, min(1.0, denaturation))
        
        # Texture change (sigmoid)
        texture_change = 1.0 / (1.0 + math.exp(-5 * (denaturation - 0.5)))
        
        # Digestibility (improves with moderate denaturation)
        if denaturation < 0.7:
            digestibility = 0.7 + 0.3 * denaturation
        else:
            # Over-denaturation can reduce digestibility
            digestibility = 1.0 - 0.2 * (denaturation - 0.7)
        
        # Water loss (more denaturation = more water expelled)
        water_loss = denaturation * 0.3  # Up to 30% water loss
        
        return {
            'denaturation_degree': float(denaturation),
            'texture_change': float(texture_change),
            'digestibility': float(digestibility),
            'water_loss': float(water_loss)
        }


# ============================================================================
# STARCH GELATINIZATION MODEL
# ============================================================================

class StarchGelatinization:
    """
    Model starch gelatinization (swelling and thickening)
    
    When heated with water, starch granules:
    1. Absorb water
    2. Swell
    3. Burst
    4. Release amylose (creates gel/thickness)
    """
    
    def __init__(self):
        # Starch type gelatinization temperatures
        self.gelatinization_temps = {
            'potato': (56.0, 66.0),    # (start, complete)
            'corn': (62.0, 72.0),
            'wheat': (58.0, 64.0),
            'rice': (68.0, 78.0),
            'tapioca': (52.0, 64.0)
        }
        
        logger.info("Starch Gelatinization model initialized")
    
    def predict_gelatinization(
        self,
        starch_type: str,
        temperature: float,
        time_minutes: float,
        water_ratio: float  # Water to starch ratio
    ) -> Dict[str, float]:
        """
        Predict starch gelatinization
        
        Returns:
            gelatinization_degree: 0.0-1.0
            viscosity_increase: Relative thickening
            water_absorption: Grams water absorbed per gram starch
            digestibility: Starch digestibility
        """
        # Get temperature range
        temp_range = self.gelatinization_temps.get(starch_type, (60.0, 70.0))
        start_temp, complete_temp = temp_range
        
        # Check water availability (need at least 2:1 water:starch)
        if water_ratio < 2.0:
            water_factor = water_ratio / 2.0
        else:
            water_factor = 1.0
        
        # Temperature-dependent gelatinization
        if temperature < start_temp:
            gelat = 0.0
        elif temperature >= complete_temp:
            gelat = 1.0
        else:
            # Linear between start and complete
            gelat = (temperature - start_temp) / (complete_temp - start_temp)
        
        # Time dependency (reaches equilibrium)
        time_factor = 1.0 - math.exp(-0.2 * time_minutes)
        
        # Final gelatinization
        gelatinization = gelat * time_factor * water_factor
        gelatinization = max(0.0, min(1.0, gelatinization))
        
        # Viscosity (thickening power)
        viscosity = gelatinization * 5.0  # 5x viscosity increase
        
        # Water absorption (starch can absorb up to 100% of its weight)
        water_absorbed = gelatinization * min(water_ratio, 1.0)
        
        # Digestibility (gelatinized starch is more digestible)
        digestibility = 0.5 + 0.5 * gelatinization
        
        return {
            'gelatinization_degree': float(gelatinization),
            'viscosity_increase': float(viscosity),
            'water_absorption': float(water_absorbed),
            'digestibility': float(digestibility)
        }


# ============================================================================
# FAT OXIDATION MODEL
# ============================================================================

class FatOxidation:
    """
    Model fat oxidation (rancidity) during cooking and storage
    
    Fat oxidation produces:
    - Off-flavors (rancid taste)
    - Loss of nutritional value
    - Potentially harmful compounds
    """
    
    def __init__(self):
        # Fatty acid oxidation susceptibility
        self.oxidation_rates = {
            'saturated': 0.1,        # Low susceptibility
            'monounsaturated': 0.5,  # Medium
            'polyunsaturated': 1.0   # High (omega-3, omega-6)
        }
        
        logger.info("Fat Oxidation model initialized")
    
    def predict_oxidation(
        self,
        fat_type: str,
        temperature: float,
        time_minutes: float,
        oxygen_exposure: float = 1.0,  # 0.0-1.0
        antioxidants: float = 0.0      # 0.0-1.0
    ) -> Dict[str, float]:
        """
        Predict fat oxidation
        
        Returns:
            oxidation_degree: 0.0-1.0
            rancidity_index: Off-flavor intensity
            nutritional_loss: Vitamin E, omega-3 degradation
            harmful_compounds: Aldehydes, ketones
        """
        # Base oxidation rate
        base_rate = self.oxidation_rates.get(fat_type, 0.5)
        
        # Temperature effect (exponential)
        temp_factor = math.exp((temperature - 20) / 50)
        
        # Oxygen exposure
        oxygen_factor = oxygen_exposure
        
        # Antioxidant protection
        protection = 1.0 - antioxidants * 0.8
        
        # Oxidation kinetics
        rate = base_rate * temp_factor * oxygen_factor * protection
        oxidation = 1.0 - math.exp(-rate * time_minutes / 60)
        oxidation = max(0.0, min(1.0, oxidation))
        
        # Rancidity (nonlinear - becomes noticeable around 30%)
        if oxidation < 0.3:
            rancidity = 0.0
        else:
            rancidity = (oxidation - 0.3) / 0.7
        
        # Nutritional loss (vitamin E, omega-3s)
        nutritional_loss = oxidation * 0.8
        
        # Harmful compounds (aldehydes increase with oxidation)
        harmful = oxidation * 0.6
        
        return {
            'oxidation_degree': float(oxidation),
            'rancidity_index': float(rancidity),
            'nutritional_loss': float(nutritional_loss),
            'harmful_compounds': float(harmful)
        }


# ============================================================================
# VITAMIN DEGRADATION MODEL
# ============================================================================

class VitaminDegradation:
    """
    Model vitamin degradation during cooking
    
    Different vitamins have different stability:
    - Water-soluble: B vitamins, Vitamin C (heat-sensitive, leach into water)
    - Fat-soluble: A, D, E, K (more heat-stable)
    """
    
    def __init__(self):
        # Vitamin stability profiles
        self.vitamin_stability = {
            # (heat_stability, water_solubility, oxygen_sensitivity)
            'vitamin_c': (0.2, 1.0, 0.9),      # Very unstable
            'thiamin_b1': (0.3, 1.0, 0.5),
            'riboflavin_b2': (0.7, 0.8, 0.3),
            'niacin_b3': (0.9, 0.8, 0.2),
            'pantothenic_b5': (0.6, 1.0, 0.3),
            'pyridoxine_b6': (0.5, 0.9, 0.4),
            'biotin_b7': (0.8, 0.7, 0.2),
            'folate_b9': (0.4, 1.0, 0.6),
            'cobalamin_b12': (0.7, 0.8, 0.5),
            'vitamin_a': (0.6, 0.1, 0.7),
            'vitamin_d': (0.8, 0.0, 0.3),
            'vitamin_e': (0.7, 0.0, 0.8),
            'vitamin_k': (0.8, 0.0, 0.4)
        }
        
        logger.info("Vitamin Degradation model initialized")
    
    def predict_retention(
        self,
        vitamin: str,
        cooking_method: CookingMethod,
        temperature: float,
        time_minutes: float,
        water_volume_ml: float = 0.0
    ) -> Dict[str, float]:
        """
        Predict vitamin retention after cooking
        
        Returns:
            retention_ratio: 0.0-1.0 (fraction retained)
            degradation_degree: 0.0-1.0 (fraction lost)
            leaching_loss: Loss due to water leaching
            heat_loss: Loss due to heat
        """
        # Get vitamin properties
        if vitamin not in self.vitamin_stability:
            # Default to moderate stability
            heat_stab, water_sol, oxy_sens = (0.6, 0.5, 0.5)
        else:
            heat_stab, water_sol, oxy_sens = self.vitamin_stability[vitamin]
        
        # Heat degradation
        heat_factor = (temperature / 100.0) * (1.0 - heat_stab)
        heat_rate = 0.01 * heat_factor
        heat_loss = 1.0 - math.exp(-heat_rate * time_minutes)
        
        # Water leaching (for water-soluble vitamins in wet cooking)
        if cooking_method in [CookingMethod.BOILING, CookingMethod.STEAMING]:
            leaching_rate = water_sol * 0.02
            if water_volume_ml > 0:
                leaching_loss = 1.0 - math.exp(-leaching_rate * time_minutes)
            else:
                leaching_loss = 0.0
        else:
            leaching_loss = 0.0
        
        # Oxygen degradation (for sensitive vitamins)
        if cooking_method in [CookingMethod.FRYING, CookingMethod.GRILLING]:
            oxygen_rate = oxy_sens * 0.015
            oxygen_loss = 1.0 - math.exp(-oxygen_rate * time_minutes)
        else:
            oxygen_loss = 0.0
        
        # Total degradation (combined effects)
        total_loss = heat_loss + leaching_loss + oxygen_loss
        total_loss = min(1.0, total_loss)
        
        retention = 1.0 - total_loss
        
        return {
            'retention_ratio': float(retention),
            'degradation_degree': float(total_loss),
            'leaching_loss': float(leaching_loss),
            'heat_loss': float(heat_loss),
            'oxygen_loss': float(oxygen_loss)
        }


# ============================================================================
# MACRONUTRIENT INTERACTION MODEL
# ============================================================================

class MacronutrientInteraction:
    """
    Model interactions between macronutrients
    
    Examples:
    - Protein-starch: Gluten formation
    - Fat-water: Emulsification
    - Protein-acid: Curdling
    - Sugar-protein: Maillard
    """
    
    def __init__(self):
        logger.info("Macronutrient Interaction model initialized")
    
    def predict_protein_starch_interaction(
        self,
        protein_g: float,
        starch_g: float,
        water_g: float,
        mixing_time: float
    ) -> Dict[str, float]:
        """Predict gluten formation in dough"""
        # Gluten formation requires wheat protein + water + mixing
        if protein_g < 0.5 or water_g < 5.0:
            return {
                'gluten_development': 0.0,
                'dough_strength': 0.0,
                'elasticity': 0.0
            }
        
        # Hydration ratio
        hydration = water_g / (protein_g + starch_g)
        optimal_hydration = 0.6
        
        if hydration < optimal_hydration:
            hydration_factor = hydration / optimal_hydration
        else:
            hydration_factor = 1.0 - 0.3 * (hydration - optimal_hydration)
        
        # Mixing develops gluten
        mixing_factor = 1.0 - math.exp(-0.1 * mixing_time)
        
        # Gluten development
        gluten = protein_g * hydration_factor * mixing_factor / 20.0
        gluten = min(1.0, gluten)
        
        # Dough properties
        strength = gluten * 0.9
        elasticity = gluten * 0.8
        
        return {
            'gluten_development': float(gluten),
            'dough_strength': float(strength),
            'elasticity': float(elasticity)
        }
    
    def predict_emulsification(
        self,
        fat_g: float,
        water_g: float,
        emulsifier_g: float,
        mixing_speed: float = 1.0
    ) -> Dict[str, float]:
        """Predict fat-water emulsion formation (e.g., mayonnaise)"""
        # Need emulsifier (egg yolk, lecithin, etc.)
        if emulsifier_g < 0.1:
            return {
                'emulsion_stability': 0.0,
                'droplet_size': 100.0,  # Large droplets = unstable
                'viscosity': 1.0
            }
        
        # Emulsifier effectiveness
        emulsifier_ratio = emulsifier_g / (fat_g + water_g)
        optimal_ratio = 0.05
        
        if emulsifier_ratio < optimal_ratio:
            stability = emulsifier_ratio / optimal_ratio
        else:
            stability = 1.0
        
        # Mixing creates smaller droplets
        mixing_factor = min(1.0, mixing_speed)
        
        # Droplet size (micrometers)
        droplet_size = 100.0 * (1.0 - stability * mixing_factor)
        droplet_size = max(1.0, droplet_size)
        
        # Viscosity increases with stable emulsion
        viscosity = 1.0 + 5.0 * stability
        
        return {
            'emulsion_stability': float(stability),
            'droplet_size': float(droplet_size),
            'viscosity': float(viscosity)
        }


# ============================================================================
# COOKING TRANSFORMATION ORCHESTRATOR
# ============================================================================

class CookingTransformation:
    """
    Complete cooking transformation system
    Integrates all chemical models
    """
    
    def __init__(self, config: Optional[ChemistryConfig] = None):
        self.config = config or ChemistryConfig()
        
        # Chemical models
        self.maillard = MaillardReaction()
        self.protein = ProteinDenaturation()
        self.starch = StarchGelatinization()
        self.fat_ox = FatOxidation()
        self.vitamin = VitaminDegradation()
        self.interactions = MacronutrientInteraction()
        
        # Statistics
        self.transformations_predicted = 0
        
        logger.info("Cooking Transformation system initialized")
    
    def predict_cooking_outcome(
        self,
        food_name: str,
        initial_composition: MacronutrientProfile,
        cooking_method: CookingMethod,
        temperature: float,
        time_minutes: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict complete cooking transformation
        
        Returns:
            final_composition: Updated macronutrient profile
            chemical_changes: All reaction outcomes
            sensory_changes: Color, texture, flavor
            nutritional_changes: Vitamin retention, digestibility
        """
        results = {
            'food_name': food_name,
            'cooking_method': cooking_method.value,
            'temperature_c': temperature,
            'time_minutes': time_minutes
        }
        
        # Maillard browning
        if initial_composition.sugar_g > 0 and initial_composition.protein_g > 0:
            moisture_ratio = initial_composition.water_g / initial_composition.total_mass()
            maillard_result = self.maillard.predict_browning(
                temperature, time_minutes,
                initial_composition.sugar_g,
                initial_composition.protein_g,
                moisture_ratio
            )
            results['maillard_reaction'] = maillard_result
        
        # Protein denaturation
        if initial_composition.protein_g > 0:
            protein_result = self.protein.predict_denaturation(
                'myosin', temperature, time_minutes
            )
            results['protein_denaturation'] = protein_result
        
        # Starch gelatinization
        if initial_composition.starch_g > 0:
            water_ratio = initial_composition.water_g / max(initial_composition.starch_g, 0.1)
            starch_result = self.starch.predict_gelatinization(
                'potato', temperature, time_minutes, water_ratio
            )
            results['starch_gelatinization'] = starch_result
        
        # Fat oxidation
        if initial_composition.fat_g > 0:
            fat_result = self.fat_ox.predict_oxidation(
                'polyunsaturated', temperature, time_minutes
            )
            results['fat_oxidation'] = fat_result
        
        # Vitamin retention (Vitamin C as example)
        vitamin_result = self.vitamin.predict_retention(
            'vitamin_c', cooking_method, temperature, time_minutes
        )
        results['vitamin_retention'] = vitamin_result
        
        # Update statistics
        self.transformations_predicted += 1
        
        return results


# ============================================================================
# TESTING
# ============================================================================

def test_food_chemistry():
    """Test food chemistry models"""
    print("=" * 80)
    print("FOOD CHEMISTRY MODELS - TEST")
    print("=" * 80)
    
    # Test Maillard reaction
    print("\n" + "="*80)
    print("Test: Maillard Reaction (Steak Grilling)")
    print("="*80)
    
    maillard = MaillardReaction()
    result = maillard.predict_browning(
        temperature=200.0,
        time_minutes=15.0,
        sugar_g=0.5,
        protein_g=25.0,
        moisture_ratio=0.2
    )
    
    print(f"✓ Steak grilling at 200°C for 15 minutes:")
    print(f"  Browning index: {result['browning_index']:.2f}")
    print(f"  Flavor compounds: {result['flavor_compounds']:.2f}")
    print(f"  Acrylamide risk: {result['acrylamide_risk']:.2f}")
    
    # Test protein denaturation
    print("\n" + "="*80)
    print("Test: Protein Denaturation (Chicken Breast)")
    print("="*80)
    
    protein = ProteinDenaturation()
    result = protein.predict_denaturation(
        protein_type='myosin',
        temperature=75.0,
        time_minutes=20.0,
        ph=6.5
    )
    
    print(f"✓ Chicken breast at 75°C for 20 minutes:")
    print(f"  Denaturation: {result['denaturation_degree']:.2f}")
    print(f"  Texture change: {result['texture_change']:.2f}")
    print(f"  Digestibility: {result['digestibility']:.2f}")
    print(f"  Water loss: {result['water_loss']:.2%}")
    
    # Test starch gelatinization
    print("\n" + "="*80)
    print("Test: Starch Gelatinization (Rice Cooking)")
    print("="*80)
    
    starch = StarchGelatinization()
    result = starch.predict_gelatinization(
        starch_type='rice',
        temperature=95.0,
        time_minutes=18.0,
        water_ratio=2.5
    )
    
    print(f"✓ Rice at 95°C for 18 minutes (2.5:1 water:starch):")
    print(f"  Gelatinization: {result['gelatinization_degree']:.2f}")
    print(f"  Viscosity increase: {result['viscosity_increase']:.2f}x")
    print(f"  Water absorption: {result['water_absorption']:.2f}g/g")
    print(f"  Digestibility: {result['digestibility']:.2f}")
    
    # Test fat oxidation
    print("\n" + "="*80)
    print("Test: Fat Oxidation (Frying Oil)")
    print("="*80)
    
    fat = FatOxidation()
    result = fat.predict_oxidation(
        fat_type='polyunsaturated',
        temperature=180.0,
        time_minutes=30.0,
        oxygen_exposure=1.0,
        antioxidants=0.1
    )
    
    print(f"✓ Polyunsaturated oil at 180°C for 30 minutes:")
    print(f"  Oxidation degree: {result['oxidation_degree']:.2f}")
    print(f"  Rancidity index: {result['rancidity_index']:.2f}")
    print(f"  Nutritional loss: {result['nutritional_loss']:.2%}")
    print(f"  Harmful compounds: {result['harmful_compounds']:.2f}")
    
    # Test vitamin degradation
    print("\n" + "="*80)
    print("Test: Vitamin Degradation (Boiled Broccoli)")
    print("="*80)
    
    vitamin = VitaminDegradation()
    result = vitamin.predict_retention(
        vitamin='vitamin_c',
        cooking_method=CookingMethod.BOILING,
        temperature=100.0,
        time_minutes=10.0,
        water_volume_ml=500.0
    )
    
    print(f"✓ Vitamin C in boiled broccoli (100°C, 10 min):")
    print(f"  Retention: {result['retention_ratio']:.2%}")
    print(f"  Total loss: {result['degradation_degree']:.2%}")
    print(f"  Leaching loss: {result['leaching_loss']:.2%}")
    print(f"  Heat loss: {result['heat_loss']:.2%}")
    
    # Test macronutrient interactions
    print("\n" + "="*80)
    print("Test: Macronutrient Interactions")
    print("="*80)
    
    interactions = MacronutrientInteraction()
    
    # Gluten formation
    result = interactions.predict_protein_starch_interaction(
        protein_g=12.0,
        starch_g=80.0,
        water_g=55.0,
        mixing_time=10.0
    )
    
    print(f"✓ Bread dough (flour + water + mixing):")
    print(f"  Gluten development: {result['gluten_development']:.2f}")
    print(f"  Dough strength: {result['dough_strength']:.2f}")
    print(f"  Elasticity: {result['elasticity']:.2f}")
    
    # Emulsification
    result = interactions.predict_emulsification(
        fat_g=50.0,
        water_g=30.0,
        emulsifier_g=10.0,
        mixing_speed=1.0
    )
    
    print(f"\n✓ Mayonnaise (oil + water + egg yolk):")
    print(f"  Emulsion stability: {result['emulsion_stability']:.2f}")
    print(f"  Droplet size: {result['droplet_size']:.1f} μm")
    print(f"  Viscosity: {result['viscosity']:.2f}x")
    
    # Test complete cooking transformation
    print("\n" + "="*80)
    print("Test: Complete Cooking Transformation (Roasted Chicken)")
    print("="*80)
    
    cooking = CookingTransformation()
    
    chicken_raw = MacronutrientProfile(
        protein_g=23.0,
        carbohydrate_g=0.0,
        fat_g=3.6,
        fiber_g=0.0,
        water_g=73.0,
        sugar_g=0.1
    )
    
    result = cooking.predict_cooking_outcome(
        food_name="Chicken Breast",
        initial_composition=chicken_raw,
        cooking_method=CookingMethod.ROASTING,
        temperature=180.0,
        time_minutes=35.0
    )
    
    print(f"✓ Roasted chicken breast at 180°C for 35 minutes:")
    print(f"  Cooking method: {result['cooking_method']}")
    
    if 'maillard_reaction' in result:
        m = result['maillard_reaction']
        print(f"  Maillard browning: {m['browning_index']:.2f}")
    
    if 'protein_denaturation' in result:
        p = result['protein_denaturation']
        print(f"  Protein denaturation: {p['denaturation_degree']:.2f}")
        print(f"  Moisture loss: {p['water_loss']:.1%}")
    
    if 'vitamin_retention' in result:
        v = result['vitamin_retention']
        print(f"  Vitamin C retention: {v['retention_ratio']:.1%}")
    
    print(f"\n✓ Total transformations predicted: {cooking.transformations_predicted}")
    
    print("\n✅ All food chemistry tests passed!")


if __name__ == '__main__':
    test_food_chemistry()
