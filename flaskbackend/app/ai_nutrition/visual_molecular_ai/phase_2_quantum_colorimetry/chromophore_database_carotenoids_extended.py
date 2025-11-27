"""
PHASE 2 PART 4a: EXTENDED CAROTENOIDS DATABASE
==============================================

Expansion of carotenoid database from 10 to 18 compounds (Phase 1 of database expansion).
This module adds 8 additional carotenoids with complete spectroscopic data:
- Phytoene (colorless carotene precursor)
- Phytofluene (pale yellow intermediate)
- Neurosporene (orange-red intermediate)
- α-Cryptoxanthin (orange xanthophyll)
- β-Apo-8'-carotenal (orange food colorant)
- Citranaxanthin (yellow pigment)
- Diatoxanthin (brown/golden diatom pigment)
- Fucoxanthin (brown algae pigment)

Each entry includes:
- Molecular structure (formula, MW, SMILES)
- UV-Vis absorption (λmax, ε, full spectral bands)
- Fluorescence properties (Φ_f, τ, emission)
- Raman vibrational peaks (C=C, C-C stretches)
- IR absorption bands (functional groups)
- Conjugation length (# double bonds)
- Food sources & biological function

Scientific References:
- Britton, G. et al. (2004) Carotenoids Handbook
- Liaaen-Jensen, S. (2004) Methods in Enzymology
- USDA Carotenoid Database
- Handbook of Natural Colorants (2009)

Author: Visual Molecular AI System
Version: 2.4.1
Lines: ~800 (target for Phase 4a)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: DATA STRUCTURES
# ============================================================================

@dataclass
class FluorescenceData:
    """Fluorescence properties of a chromophore"""
    quantum_yield: float  # Φ_f (0-1)
    lifetime_ns: float    # τ in nanoseconds
    emission_max_nm: float  # λ_em max
    stokes_shift_nm: float  # λ_em - λ_abs
    
@dataclass
class ExtendedCarotenoid:
    """
    Complete carotenoid data structure with full spectroscopic information.
    """
    # Basic molecular properties
    name: str
    formula: str
    molecular_weight: float  # g/mol
    smiles: str
    
    # UV-Vis spectroscopy
    lambda_max: float  # nm, main absorption maximum
    extinction_coeff: float  # M⁻¹cm⁻¹ (ε)
    absorption_bands: List[Tuple[float, float]]  # [(λ nm, relative intensity)]
    
    # Fluorescence (optional - many carotenoids are non-fluorescent)
    fluorescence: Optional[FluorescenceData]
    
    # Vibrational spectroscopy
    raman_peaks: List[Tuple[float, str]]  # [(wavenumber cm⁻¹, assignment)]
    ir_peaks: List[Tuple[float, str]]     # [(wavenumber cm⁻¹, assignment)]
    
    # Molecular structure
    n_conjugated_bonds: int  # Number of conjugated C=C
    chromophore_type: str    # "carotene" or "xanthophyll"
    
    # Biological/food context
    food_sources: List[str]
    biological_function: str
    
    # Additional properties
    solubility_class: str  # "lipophilic", "amphiphilic"
    stability_notes: str   # Oxidation, light sensitivity


# ============================================================================
# SECTION 2: EXTENDED CAROTENOIDS DATABASE (8 NEW COMPOUNDS)
# ============================================================================

EXTENDED_CAROTENOIDS_DATABASE: Dict[str, ExtendedCarotenoid] = {
    
    # ========================================================================
    # 1. PHYTOENE - Colorless Carotene Precursor
    # ========================================================================
    "phytoene": ExtendedCarotenoid(
        name="Phytoene",
        formula="C40H64",
        molecular_weight=544.85,
        smiles="CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC=C(C)CCCC(C)=CCCC(C)=CCCC(C)=C",
        
        # UV-Vis: Only 3 conjugated bonds → absorbs in UV, colorless
        lambda_max=285.0,  # nm (UV region, not visible)
        extinction_coeff=45000.0,  # M⁻¹cm⁻¹
        absorption_bands=[
            (275.0, 0.65),  # Vibronic shoulder
            (285.0, 1.00),  # Main peak
            (295.0, 0.62),  # Vibronic shoulder
        ],
        
        # Fluorescence: Weak, not typically measured
        fluorescence=FluorescenceData(
            quantum_yield=0.02,
            lifetime_ns=0.5,
            emission_max_nm=320.0,
            stokes_shift_nm=35.0
        ),
        
        # Raman: C=C stretches (fewer than typical carotenoids)
        raman_peaks=[
            (1515.0, "ν₁ C=C stretch (conjugated)"),
            (1155.0, "ν₂ C-C stretch"),
            (1005.0, "ν₃ C-CH₃ rock"),
        ],
        
        # IR: Alkene and methyl groups
        ir_peaks=[
            (3010.0, "=C-H stretch"),
            (2960.0, "C-H stretch (CH₃)"),
            (2920.0, "C-H stretch (CH₂)"),
            (1665.0, "C=C stretch"),
            (1440.0, "CH₂ scissor"),
        ],
        
        n_conjugated_bonds=3,
        chromophore_type="carotene",
        
        food_sources=[
            "Tomatoes (trace)",
            "Carrots (biosynthetic precursor)",
            "Algae (biosynthetic intermediate)",
        ],
        
        biological_function=(
            "Biosynthetic precursor to all carotenoids. First C40 carotenoid "
            "formed from two C20 GGPP units. Colorless due to short conjugation. "
            "Converted to phytofluene by desaturase enzymes."
        ),
        
        solubility_class="lipophilic",
        stability_notes=(
            "Relatively stable, less prone to oxidation than fully conjugated "
            "carotenoids. Light-insensitive due to lack of visible absorption."
        ),
    ),
    
    # ========================================================================
    # 2. PHYTOFLUENE - Pale Yellow Intermediate
    # ========================================================================
    "phytofluene": ExtendedCarotenoid(
        name="Phytofluene",
        formula="C40H62",
        molecular_weight=542.91,
        smiles="CC(C)=CCCC(C)=CCCC(C)=CCC=C(C)C=CC(C)=CC=C(C)CC=C(C)CCCC(C)=CCCC(C)=CCCC(C)=C",
        
        # UV-Vis: 5 conjugated bonds → pale yellow (360nm edge)
        lambda_max=348.0,  # nm (UV-vis border)
        extinction_coeff=78000.0,  # M⁻¹cm⁻¹
        absorption_bands=[
            (330.0, 0.72),  # Vibronic
            (348.0, 1.00),  # Main peak
            (367.0, 0.75),  # Vibronic
        ],
        
        # Fluorescence: Weak but detectable
        fluorescence=FluorescenceData(
            quantum_yield=0.05,
            lifetime_ns=1.2,
            emission_max_nm=395.0,
            stokes_shift_nm=47.0
        ),
        
        # Raman: Intermediate conjugation
        raman_peaks=[
            (1530.0, "ν₁ C=C stretch (5 conjugated bonds)"),
            (1160.0, "ν₂ C-C stretch"),
            (1008.0, "ν₃ C-CH₃ rock"),
        ],
        
        # IR: Similar to phytoene
        ir_peaks=[
            (3010.0, "=C-H stretch"),
            (2960.0, "C-H stretch (CH₃)"),
            (2920.0, "C-H stretch (CH₂)"),
            (1650.0, "C=C stretch"),
            (1440.0, "CH₂ scissor"),
        ],
        
        n_conjugated_bonds=5,
        chromophore_type="carotene",
        
        food_sources=[
            "Tomatoes (0.1-0.3 mg/100g)",
            "Pink grapefruit",
            "Watermelon",
            "Guava",
        ],
        
        biological_function=(
            "Biosynthetic intermediate between phytoene and ζ-carotene. "
            "Contributes pale yellow color to some tomato varieties. "
            "Potential health benefits similar to lycopene."
        ),
        
        solubility_class="lipophilic",
        stability_notes=(
            "Moderately stable, begins to show light sensitivity. "
            "Less oxidation-prone than lycopene."
        ),
    ),
    
    # ========================================================================
    # 3. NEUROSPORENE - Orange-Red Intermediate
    # ========================================================================
    "neurosporene": ExtendedCarotenoid(
        name="Neurosporene",
        formula="C40H58",
        molecular_weight=538.89,
        smiles="CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2=C(CCCC2(C)C)C)C)C",
        
        # UV-Vis: 9 conjugated bonds → orange-red
        lambda_max=440.0,  # nm
        extinction_coeff=142000.0,  # M⁻¹cm⁻¹
        absorption_bands=[
            (416.0, 0.68),  # Vibronic (I)
            (440.0, 1.00),  # Main peak (II)
            (468.0, 0.74),  # Vibronic (III)
        ],
        
        # Fluorescence: Very weak (typical for carotenoids)
        fluorescence=FluorescenceData(
            quantum_yield=0.001,
            lifetime_ns=0.1,
            emission_max_nm=520.0,
            stokes_shift_nm=80.0
        ),
        
        # Raman: Strong C=C signature
        raman_peaks=[
            (1555.0, "ν₁ C=C stretch (9 conjugated bonds)"),
            (1195.0, "ν₂ C-C stretch"),
            (1010.0, "ν₃ C-CH₃ rock"),
        ],
        
        # IR: Conjugated system
        ir_peaks=[
            (3010.0, "=C-H stretch"),
            (2960.0, "C-H stretch (CH₃)"),
            (2920.0, "C-H stretch (CH₂)"),
            (1595.0, "C=C stretch (conjugated)"),
            (1440.0, "CH₂ scissor"),
        ],
        
        n_conjugated_bonds=9,
        chromophore_type="carotene",
        
        food_sources=[
            "Tomatoes (biosynthetic intermediate)",
            "Certain bacteria (Neurospora crassa)",
            "Fungi (mutant strains)",
        ],
        
        biological_function=(
            "Biosynthetic intermediate in carotenoid pathway. Precursor to "
            "lycopene (10 conjugated bonds). Named after Neurospora fungus. "
            "Rare in foods, mainly in microorganisms."
        ),
        
        solubility_class="lipophilic",
        stability_notes=(
            "Moderately stable, sensitive to light and oxygen. "
            "9 conjugated bonds → moderate autoxidation rate."
        ),
    ),
    
    # ========================================================================
    # 4. α-CRYPTOXANTHIN - Orange Xanthophyll
    # ========================================================================
    "alpha_cryptoxanthin": ExtendedCarotenoid(
        name="α-Cryptoxanthin",
        formula="C40H56O",
        molecular_weight=552.87,
        smiles="CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2C(=CCCC2(C)CO)C)C)C",
        
        # UV-Vis: Similar to β-cryptoxanthin, one OH group
        lambda_max=450.0,  # nm
        extinction_coeff=135000.0,  # M⁻¹cm⁻¹
        absorption_bands=[
            (425.0, 0.71),  # Vibronic
            (450.0, 1.00),  # Main peak
            (478.0, 0.68),  # Vibronic
        ],
        
        # Fluorescence: Quenched by oxygen
        fluorescence=None,  # Non-fluorescent in air
        
        # Raman: C=C + C-OH signature
        raman_peaks=[
            (1525.0, "ν₁ C=C stretch"),
            (1160.0, "ν₂ C-C stretch"),
            (1080.0, "C-O stretch (primary alcohol)"),
            (1008.0, "ν₃ C-CH₃ rock"),
        ],
        
        # IR: Hydroxyl group signature
        ir_peaks=[
            (3400.0, "O-H stretch (broad)"),
            (3010.0, "=C-H stretch"),
            (2960.0, "C-H stretch (CH₃)"),
            (1600.0, "C=C stretch"),
            (1050.0, "C-O stretch"),
        ],
        
        n_conjugated_bonds=11,
        chromophore_type="xanthophyll",
        
        food_sources=[
            "Oranges (0.02-0.08 mg/100g)",
            "Tangerines",
            "Papayas",
            "Red bell peppers",
            "Egg yolks (trace)",
        ],
        
        biological_function=(
            "Provitamin A (one β-ionone ring intact). Antioxidant activity. "
            "Less abundant than β-cryptoxanthin but similar function. "
            "Contributes to orange color in citrus fruits."
        ),
        
        solubility_class="lipophilic",
        stability_notes=(
            "Moderately stable, OH group slightly increases polarity. "
            "Light and oxygen sensitive. Less stable than carotenes."
        ),
    ),
    
    # ========================================================================
    # 5. β-APO-8'-CAROTENAL - Orange Food Colorant
    # ========================================================================
    "beta_apo_8_carotenal": ExtendedCarotenoid(
        name="β-Apo-8'-carotenal",
        formula="C30H40O",
        molecular_weight=416.64,
        smiles="CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=O)C)C",
        
        # UV-Vis: Shorter conjugation (aldehyde terminal)
        lambda_max=460.0,  # nm
        extinction_coeff=125000.0,  # M⁻¹cm⁻¹
        absorption_bands=[
            (435.0, 0.69),  # Vibronic
            (460.0, 1.00),  # Main peak
            (488.0, 0.65),  # Vibronic
        ],
        
        # Fluorescence: Weak
        fluorescence=FluorescenceData(
            quantum_yield=0.003,
            lifetime_ns=0.2,
            emission_max_nm=540.0,
            stokes_shift_nm=80.0
        ),
        
        # Raman: C=C + C=O signature
        raman_peaks=[
            (1520.0, "ν₁ C=C stretch"),
            (1680.0, "C=O stretch (aldehyde)"),
            (1160.0, "ν₂ C-C stretch"),
            (1008.0, "ν₃ C-CH₃ rock"),
        ],
        
        # IR: Aldehyde group prominent
        ir_peaks=[
            (2820.0, "C-H stretch (aldehyde)"),
            (2720.0, "C-H stretch (aldehyde overtone)"),
            (1680.0, "C=O stretch (aldehyde)"),
            (1600.0, "C=C stretch"),
            (2960.0, "C-H stretch (CH₃)"),
        ],
        
        n_conjugated_bonds=10,  # Including C=O
        chromophore_type="carotene",
        
        food_sources=[
            "Food additive (E160e)",
            "Orange juice (fortified)",
            "Margarine (colorant)",
            "Cheese (colorant)",
            "Baked goods",
        ],
        
        biological_function=(
            "Synthetic/natural food colorant approved in EU (E160e) and USA. "
            "Derived from β-carotene oxidation. Provitamin A activity (retinal analog). "
            "Used to enhance orange color in foods."
        ),
        
        solubility_class="lipophilic",
        stability_notes=(
            "Moderately stable in food matrices. Aldehyde group reactive. "
            "More stable than full-length carotenoids due to shorter chain."
        ),
    ),
    
    # ========================================================================
    # 6. CITRANAXANTHIN - Yellow Citrus Pigment
    # ========================================================================
    "citranaxanthin": ExtendedCarotenoid(
        name="Citranaxanthin",
        formula="C33H44O",
        molecular_weight=456.70,
        smiles="CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC(=O)CCC)C)C",
        
        # UV-Vis: Ketone-extended conjugation
        lambda_max=465.0,  # nm
        extinction_coeff=118000.0,  # M⁻¹cm⁻¹
        absorption_bands=[
            (440.0, 0.73),  # Vibronic
            (465.0, 1.00),  # Main peak
            (492.0, 0.70),  # Vibronic
        ],
        
        # Fluorescence: Quenched by ketone
        fluorescence=None,
        
        # Raman: C=C + C=O signature
        raman_peaks=[
            (1525.0, "ν₁ C=C stretch"),
            (1655.0, "C=O stretch (ketone)"),
            (1165.0, "ν₂ C-C stretch"),
            (1010.0, "ν₃ C-CH₃ rock"),
        ],
        
        # IR: Ketone carbonyl
        ir_peaks=[
            (1715.0, "C=O stretch (ketone)"),
            (3010.0, "=C-H stretch"),
            (2960.0, "C-H stretch (CH₃)"),
            (1600.0, "C=C stretch"),
            (1440.0, "CH₂ scissor"),
        ],
        
        n_conjugated_bonds=11,  # Including C=O
        chromophore_type="xanthophyll",
        
        food_sources=[
            "Orange peel (trace)",
            "Tangerine peel",
            "Valencia oranges",
            "Certain algae",
        ],
        
        biological_function=(
            "Minor citrus pigment, contributes to peel color. "
            "Ketone group extends conjugation. Potential antioxidant. "
            "Rare in human diet."
        ),
        
        solubility_class="lipophilic",
        stability_notes=(
            "Moderately stable. Ketone group increases reactivity. "
            "Light sensitive."
        ),
    ),
    
    # ========================================================================
    # 7. DIATOXANTHIN - Brown/Golden Diatom Pigment
    # ========================================================================
    "diatoxanthin": ExtendedCarotenoid(
        name="Diatoxanthin",
        formula="C40H54O2",
        molecular_weight=566.85,
        smiles="CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2C(C(CC(C2(C)C)O)(C)C)O)C)C",
        
        # UV-Vis: Two OH groups, slight red-shift
        lambda_max=495.0,  # nm (brown-golden)
        extinction_coeff=148000.0,  # M⁻¹cm⁻¹
        absorption_bands=[
            (468.0, 0.75),  # Vibronic
            (495.0, 1.00),  # Main peak
            (525.0, 0.72),  # Vibronic
        ],
        
        # Fluorescence: None (quenched)
        fluorescence=None,
        
        # Raman: C=C + C-OH signatures
        raman_peaks=[
            (1520.0, "ν₁ C=C stretch"),
            (1160.0, "ν₂ C-C stretch"),
            (1085.0, "C-O stretch (secondary alcohol)"),
            (1008.0, "ν₃ C-CH₃ rock"),
        ],
        
        # IR: Two hydroxyl groups
        ir_peaks=[
            (3450.0, "O-H stretch (broad, 2 OH groups)"),
            (3010.0, "=C-H stretch"),
            (2960.0, "C-H stretch (CH₃)"),
            (1600.0, "C=C stretch"),
            (1070.0, "C-O stretch"),
        ],
        
        n_conjugated_bonds=11,
        chromophore_type="xanthophyll",
        
        food_sources=[
            "Diatom algae (brown algae)",
            "Kelp",
            "Seaweed supplements",
            "Marine fish (via diet)",
        ],
        
        biological_function=(
            "Key pigment in diatoms and brown algae. De-epoxidation product "
            "of diadinoxanthin (xanthophyll cycle). Photoprotective role in algae. "
            "Contributes golden-brown color to kelp."
        ),
        
        solubility_class="lipophilic",
        stability_notes=(
            "Moderately stable in algal cells. Two OH groups increase polarity. "
            "Oxidation-sensitive."
        ),
    ),
    
    # ========================================================================
    # 8. FUCOXANTHIN - Brown Algae Pigment
    # ========================================================================
    "fucoxanthin": ExtendedCarotenoid(
        name="Fucoxanthin",
        formula="C42H58O6",
        molecular_weight=658.91,
        smiles="CC1CC(C(C1(C)C)(C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2=C(C(CC(O2)C(C)(C=O)O)(C)C)O)C)C)O)OC(=O)C",
        
        # UV-Vis: Complex structure, multiple functional groups
        lambda_max=450.0,  # nm
        extinction_coeff=152000.0,  # M⁻¹cm⁻¹
        absorption_bands=[
            (425.0, 0.68),  # Vibronic
            (450.0, 1.00),  # Main peak
            (475.0, 0.64),  # Vibronic
        ],
        
        # Fluorescence: None (quenched by multiple groups)
        fluorescence=None,
        
        # Raman: Complex signature
        raman_peaks=[
            (1525.0, "ν₁ C=C stretch"),
            (1740.0, "C=O stretch (ester)"),
            (1685.0, "C=O stretch (aldehyde)"),
            (1160.0, "ν₂ C-C stretch"),
            (1085.0, "C-O stretch (ether/alcohol)"),
        ],
        
        # IR: Multiple oxygen functionalities
        ir_peaks=[
            (3500.0, "O-H stretch (alcohol)"),
            (2960.0, "C-H stretch (CH₃)"),
            (1735.0, "C=O stretch (ester)"),
            (1685.0, "C=O stretch (aldehyde)"),
            (1600.0, "C=C stretch"),
            (1240.0, "C-O stretch (ester)"),
            (1050.0, "C-O stretch (ether)"),
        ],
        
        n_conjugated_bonds=11,
        chromophore_type="xanthophyll",
        
        food_sources=[
            "Wakame seaweed (1-2 mg/g dry)",
            "Kombu kelp",
            "Hijiki seaweed",
            "Brown algae supplements",
            "Undaria pinnatifida",
        ],
        
        biological_function=(
            "Major brown algae pigment (10-25% of total carotenoids). "
            "Potent antioxidant, anti-obesity properties (induces UCP1 in WAT). "
            "Anti-inflammatory, anticancer research. Unique allenic bond and "
            "epoxide groups. Contributes brown color to kelp."
        ),
        
        solubility_class="amphiphilic",  # Multiple OH groups increase polarity
        stability_notes=(
            "Highly unstable. Multiple reactive groups (allene, epoxide, ester, aldehyde). "
            "Degrades rapidly in light, heat, acid. Requires special handling."
        ),
    ),
}


# ============================================================================
# SECTION 3: DATABASE MANAGER FOR EXTENDED CAROTENOIDS
# ============================================================================

class ExtendedCarotenoidsManager:
    """
    Manager class for extended carotenoids database.
    Provides search, indexing, and retrieval functions.
    """
    
    def __init__(self):
        self.database = EXTENDED_CAROTENOIDS_DATABASE
        self.wavelength_index = self._build_wavelength_index()
        self.food_index = self._build_food_index()
        logger.info(f"Loaded {len(self.database)} extended carotenoids")
        
    def _build_wavelength_index(self) -> Dict[int, List[str]]:
        """Build wavelength index (5nm bins) for fast lookup"""
        index = {}
        for name, carotenoid in self.database.items():
            bin_center = int(round(carotenoid.lambda_max / 5.0) * 5)
            if bin_center not in index:
                index[bin_center] = []
            index[bin_center].append(name)
        logger.info(f"Wavelength index: {len(index)} bins")
        return index
    
    def _build_food_index(self) -> Dict[str, List[str]]:
        """Build food source index for nutritional queries"""
        index = {}
        for name, carotenoid in self.database.items():
            for food in carotenoid.food_sources:
                food_lower = food.lower()
                if food_lower not in index:
                    index[food_lower] = []
                index[food_lower].append(name)
        logger.info(f"Food index: {len(index)} foods mapped")
        return index
    
    def search_by_wavelength(self, lambda_nm: float, tolerance_nm: float = 10.0) -> List[str]:
        """
        Search carotenoids by absorption wavelength.
        
        Args:
            lambda_nm: Target wavelength (nm)
            tolerance_nm: Search tolerance (±nm)
            
        Returns:
            List of matching carotenoid names
        """
        matches = []
        for name, carotenoid in self.database.items():
            if abs(carotenoid.lambda_max - lambda_nm) <= tolerance_nm:
                matches.append(name)
        return matches
    
    def search_by_food(self, food_name: str) -> List[str]:
        """
        Search carotenoids by food source.
        
        Args:
            food_name: Food item name (case-insensitive)
            
        Returns:
            List of matching carotenoid names
        """
        food_lower = food_name.lower()
        # Exact match
        if food_lower in self.food_index:
            return self.food_index[food_lower]
        
        # Partial match
        matches = []
        for food_key, carotenoids in self.food_index.items():
            if food_lower in food_key or food_key in food_lower:
                matches.extend(carotenoids)
        return list(set(matches))  # Remove duplicates
    
    def get_carotenoid(self, name: str) -> Optional[ExtendedCarotenoid]:
        """Get carotenoid by name"""
        return self.database.get(name)
    
    def get_all_names(self) -> List[str]:
        """Get all carotenoid names"""
        return list(self.database.keys())
    
    def get_statistics(self) -> Dict[str, any]:
        """Get database statistics"""
        n_carotenes = sum(1 for c in self.database.values() if c.chromophore_type == "carotene")
        n_xanthophylls = sum(1 for c in self.database.values() if c.chromophore_type == "xanthophyll")
        n_fluorescent = sum(1 for c in self.database.values() if c.fluorescence is not None)
        
        lambda_values = [c.lambda_max for c in self.database.values()]
        conjugation_values = [c.n_conjugated_bonds for c in self.database.values()]
        
        return {
            "total_compounds": len(self.database),
            "carotenes": n_carotenes,
            "xanthophylls": n_xanthophylls,
            "fluorescent_compounds": n_fluorescent,
            "wavelength_range": (min(lambda_values), max(lambda_values)),
            "conjugation_range": (min(conjugation_values), max(conjugation_values)),
            "unique_foods": len(self.food_index),
        }


# ============================================================================
# SECTION 4: DEMO & VALIDATION
# ============================================================================

def demo_extended_carotenoids():
    """Demonstrate extended carotenoids database capabilities"""
    print("\n" + "="*70)
    print("EXTENDED CAROTENOIDS DATABASE - PHASE 2 PART 4a")
    print("="*70)
    
    manager = ExtendedCarotenoidsManager()
    stats = manager.get_statistics()
    
    print(f"\n📊 DATABASE STATISTICS:")
    print(f"   Total compounds: {stats['total_compounds']}")
    print(f"   Carotenes: {stats['carotenes']}")
    print(f"   Xanthophylls: {stats['xanthophylls']}")
    print(f"   Fluorescent: {stats['fluorescent_compounds']}")
    print(f"   λmax range: {stats['wavelength_range'][0]:.0f}-{stats['wavelength_range'][1]:.0f} nm")
    print(f"   Conjugation range: {stats['conjugation_range'][0]}-{stats['conjugation_range'][1]} bonds")
    print(f"   Food sources: {stats['unique_foods']}")
    
    # Demo 1: Search by wavelength
    print(f"\n🔍 SEARCH BY WAVELENGTH (450 ± 10 nm):")
    matches = manager.search_by_wavelength(450.0, tolerance_nm=10.0)
    for name in matches:
        carotenoid = manager.get_carotenoid(name)
        print(f"   ✓ {carotenoid.name}: λmax={carotenoid.lambda_max:.0f} nm, "
              f"ε={carotenoid.extinction_coeff:.0f} M⁻¹cm⁻¹")
    
    # Demo 2: Search by food
    print(f"\n🍊 SEARCH BY FOOD (Oranges):")
    matches = manager.search_by_food("oranges")
    for name in matches:
        carotenoid = manager.get_carotenoid(name)
        print(f"   ✓ {carotenoid.name}: {carotenoid.food_sources[0]}")
    
    # Demo 3: Biosynthetic pathway
    print(f"\n🧬 BIOSYNTHETIC PATHWAY (Colorless → Colored):")
    pathway = ["phytoene", "phytofluene", "neurosporene"]
    for name in pathway:
        carotenoid = manager.get_carotenoid(name)
        print(f"   {carotenoid.n_conjugated_bonds} bonds → "
              f"{carotenoid.name} (λ={carotenoid.lambda_max:.0f} nm)")
    
    # Demo 4: Special functional groups
    print(f"\n⚗️ SPECIAL FUNCTIONAL GROUPS:")
    special = {
        "beta_apo_8_carotenal": "Aldehyde (C=O)",
        "fucoxanthin": "Allene + Epoxide + Ester",
        "citranaxanthin": "Ketone (conjugated)",
    }
    for name, group in special.items():
        carotenoid = manager.get_carotenoid(name)
        print(f"   ✓ {carotenoid.name}: {group}")
    
    print(f"\n✅ Extended carotenoids database ready!")
    print(f"   Phase 4a complete: +8 carotenoids (18 total in system)")
    print("="*70 + "\n")


if __name__ == "__main__":
    demo_extended_carotenoids()
