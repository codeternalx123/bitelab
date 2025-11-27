"""
PHASE 2 PART 4d: ADDITIONAL CAROTENOIDS DATABASE (Part 2)
==========================================================

Adding 9 more carotenoids to reach 27 total (10 original + 8 Part 1 + 9 Part 2).
This module includes:

1. Canthaxanthin (4,4'-diketo-β-carotene, salmon/flamingo color)
2. Violaxanthin (5,6:5',6'-diepoxy-β-carotene, xanthophyll cycle)
3. Neoxanthin (allenic xanthophyll, chloroplast pigment)
4. Echinenone (4-keto-β-carotene, sea urchin color)
5. β-Carotene-5,6-epoxide (monoepoxide, heat degradation product)
6. Siphonaxanthin (green algae pigment, rare structure)
7. Rhodoxanthin (red astaxanthin analog, yew trees)
8. Crocetin (di-acid carotenoid, saffron pigment)
9. Bixin (mono-methyl ester, annatto food color E160b)

Each includes complete spectroscopic data, food sources, and biological function.

Scientific References:
- Britton, G. et al. (2004) Carotenoids Handbook
- Liaaen-Jensen, S. (2004) Methods in Enzymology
- Delgado-Vargas et al. (2000) Critical Reviews in Food Science

Author: Visual Molecular AI System
Version: 2.4.4
Lines: ~900 (target for Phase 4d)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: DATA STRUCTURES (REUSE FROM PART 1)
# ============================================================================

@dataclass
class FluorescenceData:
    quantum_yield: float
    lifetime_ns: float
    emission_max_nm: float
    stokes_shift_nm: float

@dataclass
class AdditionalCarotenoid:
    """Complete carotenoid data structure"""
    name: str
    formula: str
    molecular_weight: float
    smiles: str
    lambda_max: float
    extinction_coeff: float
    absorption_bands: List[Tuple[float, float]]
    fluorescence: Optional[FluorescenceData]
    raman_peaks: List[Tuple[float, str]]
    ir_peaks: List[Tuple[float, str]]
    n_conjugated_bonds: int
    chromophore_type: str
    food_sources: List[str]
    biological_function: str
    solubility_class: str
    stability_notes: str


# ============================================================================
# SECTION 2: ADDITIONAL CAROTENOIDS DATABASE (9 NEW COMPOUNDS)
# ============================================================================

ADDITIONAL_CAROTENOIDS_DATABASE: Dict[str, AdditionalCarotenoid] = {
    
    # ========================================================================
    # 1. CANTHAXANTHIN - Salmon/Flamingo Pink Pigment
    # ========================================================================
    "canthaxanthin": AdditionalCarotenoid(
        name="Canthaxanthin",
        formula="C40H52O2",
        molecular_weight=564.84,
        smiles="CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2=C(C(=O)CC(C2(C)C))(C)C)C)C",
        
        # UV-Vis: 4,4'-diketo structure → red-shifted
        lambda_max=468.0,  # nm (orange-red)
        extinction_coeff=142000.0,  # M⁻¹cm⁻¹
        absorption_bands=[
            (440.0, 0.71),
            (468.0, 1.00),
            (497.0, 0.73),
        ],
        
        fluorescence=None,  # Quenched by ketone groups
        
        raman_peaks=[
            (1520.0, "ν₁ C=C stretch"),
            (1665.0, "C=O stretch (4,4'-diketo)"),
            (1160.0, "ν₂ C-C stretch"),
            (1008.0, "ν₃ C-CH₃ rock"),
        ],
        
        ir_peaks=[
            (1715.0, "C=O stretch (conjugated ketone)"),
            (3010.0, "=C-H stretch"),
            (2960.0, "C-H stretch (CH₃)"),
            (1595.0, "C=C stretch"),
            (1440.0, "CH₂ scissor"),
        ],
        
        n_conjugated_bonds=11,  # Including C=O
        chromophore_type="xanthophyll",
        
        food_sources=[
            "Salmon (farmed, feed additive)",
            "Flamingo feathers (via diet)",
            "Mushrooms (Cantharellus)",
            "Algae (synthetic source)",
            "Food colorant (E161g)",
        ],
        
        biological_function=(
            "Produces pink-orange color in salmon, flamingos. Used as food "
            "colorant and tanning agent (controversial). Two ketone groups "
            "extend conjugation. Antioxidant properties. Not provitamin A."
        ),
        
        solubility_class="lipophilic",
        stability_notes=(
            "Moderately stable. Two ketone groups increase reactivity vs pure "
            "carotenes. Light and oxygen sensitive."
        ),
    ),
    
    # ========================================================================
    # 2. VIOLAXANTHIN - Xanthophyll Cycle Pigment
    # ========================================================================
    "violaxanthin": AdditionalCarotenoid(
        name="Violaxanthin",
        formula="C40H56O4",
        molecular_weight=600.87,
        smiles="CC1=C(C2(CCC(O2)(C)C(O)C1)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC3C(O3)(CC=C(C(C3)(C)C)O)C)C)C",
        
        # UV-Vis: Two epoxide groups
        lambda_max=441.0,  # nm
        extinction_coeff=138000.0,  # M⁻¹cm⁻¹
        absorption_bands=[
            (417.0, 0.70),
            (441.0, 1.00),
            (470.0, 0.68),
        ],
        
        fluorescence=None,
        
        raman_peaks=[
            (1520.0, "ν₁ C=C stretch"),
            (1160.0, "ν₂ C-C stretch"),
            (890.0, "Epoxide ring breathing (2 groups)"),
            (1008.0, "ν₃ C-CH₃ rock"),
        ],
        
        ir_peaks=[
            (3450.0, "O-H stretch (2 OH groups)"),
            (3010.0, "=C-H stretch"),
            (2960.0, "C-H stretch (CH₃)"),
            (1600.0, "C=C stretch"),
            (880.0, "Epoxide C-O-C asymmetric stretch"),
        ],
        
        n_conjugated_bonds=11,
        chromophore_type="xanthophyll",
        
        food_sources=[
            "Spinach (0.5-2 mg/100g)",
            "Kale",
            "Broccoli",
            "Lettuce (green varieties)",
            "Avocado",
        ],
        
        biological_function=(
            "Key component of xanthophyll cycle (violaxanthin ↔ zeaxanthin). "
            "Photoprotective role in plants. De-epoxidized under high light "
            "stress. Two 5,6-epoxide groups. Contributes yellow-green color "
            "to leafy vegetables."
        ),
        
        solubility_class="lipophilic",
        stability_notes=(
            "Moderately stable. Epoxide groups sensitive to acid (opens to "
            "diols). Light and heat sensitive."
        ),
    ),
    
    # ========================================================================
    # 3. NEOXANTHIN - Allenic Chloroplast Pigment
    # ========================================================================
    "neoxanthin": AdditionalCarotenoid(
        name="Neoxanthin",
        formula="C40H56O4",
        molecular_weight=600.87,
        smiles="CC1=CC(CC(C1(C)C)(C)O)C=C(C)C=CC=C(C)C=CC=C(C)C=CC2C(=CC(CC2(C)O)(C)C)C",
        
        # UV-Vis: Allenic bond (C=C=C) + epoxide
        lambda_max=437.0,  # nm
        extinction_coeff=125000.0,  # M⁻¹cm⁻¹
        absorption_bands=[
            (413.0, 0.68),
            (437.0, 1.00),
            (466.0, 0.65),
        ],
        
        fluorescence=None,
        
        raman_peaks=[
            (1520.0, "ν₁ C=C stretch"),
            (1950.0, "C=C=C allene stretch"),
            (1160.0, "ν₂ C-C stretch"),
            (885.0, "Epoxide ring"),
            (1008.0, "ν₃ C-CH₃ rock"),
        ],
        
        ir_peaks=[
            (3450.0, "O-H stretch (2 OH)"),
            (1950.0, "Allene C=C=C asymmetric stretch"),
            (3010.0, "=C-H stretch"),
            (2960.0, "C-H stretch"),
            (1600.0, "C=C stretch"),
            (880.0, "Epoxide C-O-C"),
        ],
        
        n_conjugated_bonds=11,  # Including allene
        chromophore_type="xanthophyll",
        
        food_sources=[
            "Spinach (major, 0.3-1.5 mg/100g)",
            "Kale",
            "Lettuce",
            "Green algae",
            "Chloroplasts (all green plants)",
        ],
        
        biological_function=(
            "Unique allenic bond (C=C=C) rare in carotenoids. Found in "
            "chloroplast light-harvesting complexes. Precursor to abscisic "
            "acid (plant hormone). Contributes to green vegetable color."
        ),
        
        solubility_class="lipophilic",
        stability_notes=(
            "Highly unstable. Allene group very reactive. Epoxide acid-labile. "
            "Degrades rapidly in light, heat, acid. Handle with care."
        ),
    ),
    
    # ========================================================================
    # 4. ECHINENONE - Sea Urchin Orange Pigment
    # ========================================================================
    "echinenone": AdditionalCarotenoid(
        name="Echinenone",
        formula="C40H54O",
        molecular_weight=550.85,
        smiles="CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2=C(C(=O)CC(C2(C)C))C)C)C",
        
        # UV-Vis: 4-keto-β-carotene
        lambda_max=458.0,  # nm (orange)
        extinction_coeff=135000.0,  # M⁻¹cm⁻¹
        absorption_bands=[
            (432.0, 0.72),
            (458.0, 1.00),
            (487.0, 0.70),
        ],
        
        fluorescence=FluorescenceData(
            quantum_yield=0.002,
            lifetime_ns=0.15,
            emission_max_nm=535.0,
            stokes_shift_nm=77.0
        ),
        
        raman_peaks=[
            (1522.0, "ν₁ C=C stretch"),
            (1670.0, "C=O stretch (4-keto)"),
            (1162.0, "ν₂ C-C stretch"),
            (1008.0, "ν₃ C-CH₃ rock"),
        ],
        
        ir_peaks=[
            (1715.0, "C=O stretch (ketone)"),
            (3010.0, "=C-H stretch"),
            (2960.0, "C-H stretch"),
            (1598.0, "C=C stretch"),
            (1440.0, "CH₂ scissor"),
        ],
        
        n_conjugated_bonds=11,  # Including C=O
        chromophore_type="xanthophyll",
        
        food_sources=[
            "Sea urchin roe (orange color)",
            "Algae (microalgae)",
            "Shellfish (via diet)",
            "Yeast (Rhodotorula)",
        ],
        
        biological_function=(
            "Produces orange color in sea urchins. Biosynthetic intermediate "
            "to canthaxanthin (4,4'-diketo). Single ketone at C-4 position. "
            "Oxidation product of β-carotene."
        ),
        
        solubility_class="lipophilic",
        stability_notes=(
            "Moderately stable. Ketone group increases reactivity. "
            "Less stable than β-carotene, more stable than canthaxanthin."
        ),
    ),
    
    # ========================================================================
    # 5. β-CAROTENE-5,6-EPOXIDE - Heat Degradation Product
    # ========================================================================
    "beta_carotene_5_6_epoxide": AdditionalCarotenoid(
        name="β-Carotene-5,6-epoxide",
        formula="C40H56O",
        molecular_weight=552.87,
        smiles="CC1=C(C2(CCC(C1)C(C)(C)O2)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC3=C(CCCC3(C)C)C)C)C",
        
        # UV-Vis: Monoepoxide
        lambda_max=444.0,  # nm
        extinction_coeff=140000.0,  # M⁻¹cm⁻¹
        absorption_bands=[
            (420.0, 0.71),
            (444.0, 1.00),
            (472.0, 0.69),
        ],
        
        fluorescence=None,
        
        raman_peaks=[
            (1525.0, "ν₁ C=C stretch"),
            (1158.0, "ν₂ C-C stretch"),
            (888.0, "Epoxide ring (5,6-position)"),
            (1008.0, "ν₃ C-CH₃ rock"),
        ],
        
        ir_peaks=[
            (3010.0, "=C-H stretch"),
            (2960.0, "C-H stretch (CH₃)"),
            (1598.0, "C=C stretch"),
            (1440.0, "CH₂ scissor"),
            (882.0, "Epoxide C-O-C"),
        ],
        
        n_conjugated_bonds=11,
        chromophore_type="xanthophyll",
        
        food_sources=[
            "Carrots (heat-treated, trace)",
            "Sweet potatoes (processed)",
            "Pumpkin (canned)",
            "Tomato products (heated)",
        ],
        
        biological_function=(
            "Thermal degradation/oxidation product of β-carotene. Forms during "
            "cooking, processing. Retains provitamin A activity (one β-ionone "
            "ring intact). Marker of food processing intensity."
        ),
        
        solubility_class="lipophilic",
        stability_notes=(
            "Moderately unstable. Epoxide opens easily under acidic conditions "
            "or heat. Further oxidation leads to apocarotenals."
        ),
    ),
    
    # ========================================================================
    # 6. SIPHONAXANTHIN - Green Algae Pigment (Rare)
    # ========================================================================
    "siphonaxanthin": AdditionalCarotenoid(
        name="Siphonaxanthin",
        formula="C40H58O4",
        molecular_weight=602.89,
        smiles="CC1=C(C(CC(C1(C)C)O)(C)O)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2=C(CCCC2(C)C)C)C)C",
        
        # UV-Vis: Rare allenic structure
        lambda_max=448.0,  # nm
        extinction_coeff=128000.0,  # M⁻¹cm⁻¹
        absorption_bands=[
            (424.0, 0.69),
            (448.0, 1.00),
            (476.0, 0.66),
        ],
        
        fluorescence=None,
        
        raman_peaks=[
            (1520.0, "ν₁ C=C stretch"),
            (1160.0, "ν₂ C-C stretch"),
            (1080.0, "C-O stretch (2 OH groups)"),
            (1008.0, "ν₃ C-CH₃ rock"),
        ],
        
        ir_peaks=[
            (3420.0, "O-H stretch (broad, 2 OH)"),
            (3010.0, "=C-H stretch"),
            (2960.0, "C-H stretch"),
            (1600.0, "C=C stretch"),
            (1070.0, "C-O stretch"),
        ],
        
        n_conjugated_bonds=11,
        chromophore_type="xanthophyll",
        
        food_sources=[
            "Green algae (Codium, Caulerpa)",
            "Sea grapes (Caulerpa lentillifera)",
            "Marine algae supplements",
        ],
        
        biological_function=(
            "Unique to siphonous green algae (Chlorophyta). Rare acyclic structure "
            "with two secondary OH groups. Anti-obesity properties (PPARα activation). "
            "Contributes to green algae color."
        ),
        
        solubility_class="lipophilic",
        stability_notes=(
            "Relatively stable. Two OH groups increase polarity. "
            "Less prone to oxidation than epoxy xanthophylls."
        ),
    ),
    
    # ========================================================================
    # 7. RHODOXANTHIN - Red Yew Tree Pigment
    # ========================================================================
    "rhodoxanthin": AdditionalCarotenoid(
        name="Rhodoxanthin",
        formula="C40H50O2",
        molecular_weight=562.82,
        smiles="CC1=CC(=CC=C1C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2=C(C=CC(=C2)C)C)C)C)O",
        
        # UV-Vis: Extended conjugation, 2 carbonyl groups
        lambda_max=514.0,  # nm (deep red)
        extinction_coeff=148000.0,  # M⁻¹cm⁻¹
        absorption_bands=[
            (486.0, 0.74),
            (514.0, 1.00),
            (545.0, 0.76),
        ],
        
        fluorescence=None,  # Quenched
        
        raman_peaks=[
            (1515.0, "ν₁ C=C stretch (extended)"),
            (1680.0, "C=O stretch (conjugated)"),
            (1165.0, "ν₂ C-C stretch"),
            (1008.0, "ν₃ C-CH₃ rock"),
        ],
        
        ir_peaks=[
            (3400.0, "O-H stretch"),
            (1680.0, "C=O stretch (highly conjugated)"),
            (3010.0, "=C-H stretch"),
            (1585.0, "C=C stretch"),
        ],
        
        n_conjugated_bonds=13,  # Extended conjugation
        chromophore_type="xanthophyll",
        
        food_sources=[
            "Yew tree berries (Taxus, toxic)",
            "Rose hips",
            "Certain fungi",
            "Rarely in foods",
        ],
        
        biological_function=(
            "Deep red pigment in yew tree arils. Rare in human diet. "
            "Extended conjugation (13 double bonds) → red color. "
            "Structural similarity to astaxanthin but aromatic rings."
        ),
        
        solubility_class="lipophilic",
        stability_notes=(
            "Moderately stable. Extended conjugation increases light sensitivity. "
            "Ketone groups increase reactivity."
        ),
    ),
    
    # ========================================================================
    # 8. CROCETIN - Saffron Di-acid Pigment
    # ========================================================================
    "crocetin": AdditionalCarotenoid(
        name="Crocetin",
        formula="C20H24O4",
        molecular_weight=328.40,
        smiles="CC1=C(C(CCC1)(C)C(=O)O)C=CC(=CC=CC(=CC=CC=C(C)C(=O)O)C)C",
        
        # UV-Vis: Short apocarotenoid with 2 COOH groups
        lambda_max=443.0,  # nm (orange-red)
        extinction_coeff=132000.0,  # M⁻¹cm⁻¹
        absorption_bands=[
            (418.0, 0.70),
            (443.0, 1.00),
            (471.0, 0.68),
        ],
        
        fluorescence=FluorescenceData(
            quantum_yield=0.004,
            lifetime_ns=0.25,
            emission_max_nm=520.0,
            stokes_shift_nm=77.0
        ),
        
        raman_peaks=[
            (1520.0, "ν₁ C=C stretch"),
            (1705.0, "C=O stretch (COOH)"),
            (1160.0, "ν₂ C-C stretch"),
            (1290.0, "C-O stretch (acid)"),
        ],
        
        ir_peaks=[
            (3300.0, "O-H stretch (broad, COOH)"),
            (1705.0, "C=O stretch (carboxylic acid)"),
            (3010.0, "=C-H stretch"),
            (1600.0, "C=C stretch"),
            (1290.0, "C-O stretch"),
        ],
        
        n_conjugated_bonds=9,
        chromophore_type="apocarotenoid",
        
        food_sources=[
            "Saffron (Crocus sativus, 1-10% dry weight)",
            "Gardenia fruit (Chinese herb)",
            "Food colorant (natural)",
        ],
        
        biological_function=(
            "Primary pigment in saffron (most expensive spice). Water-soluble "
            "in salt form (glycosides). C20 apocarotenoid (cleaved from zeaxanthin). "
            "Two carboxylic acids → amphiphilic. Antioxidant, neuroprotective."
        ),
        
        solubility_class="amphiphilic",  # COOH groups increase polarity
        stability_notes=(
            "Moderately stable. Carboxylic acids stable. Light sensitive. "
            "More stable than full-length carotenoids due to shorter chain."
        ),
    ),
    
    # ========================================================================
    # 9. BIXIN - Annatto Food Color (E160b)
    # ========================================================================
    "bixin": AdditionalCarotenoid(
        name="Bixin",
        formula="C25H30O4",
        molecular_weight=394.50,
        smiles="COC(=O)C=C(C)C=CC=CC(C)=CC=CC=C(C)C=CC1=C(C(=O)O)CCCC1(C)C",
        
        # UV-Vis: Mono-methyl ester apocarotenoid
        lambda_max=470.0,  # nm (orange-red)
        extinction_coeff=138000.0,  # M⁻¹cm⁻¹
        absorption_bands=[
            (444.0, 0.72),
            (470.0, 1.00),
            (500.0, 0.70),
        ],
        
        fluorescence=FluorescenceData(
            quantum_yield=0.003,
            lifetime_ns=0.18,
            emission_max_nm=545.0,
            stokes_shift_nm=75.0
        ),
        
        raman_peaks=[
            (1518.0, "ν₁ C=C stretch"),
            (1710.0, "C=O stretch (ester + acid)"),
            (1160.0, "ν₂ C-C stretch"),
            (1240.0, "C-O stretch (ester)"),
        ],
        
        ir_peaks=[
            (3300.0, "O-H stretch (COOH)"),
            (2950.0, "C-H stretch (OCH₃)"),
            (1735.0, "C=O stretch (methyl ester)"),
            (1710.0, "C=O stretch (COOH)"),
            (1600.0, "C=C stretch"),
            (1240.0, "C-O stretch (ester)"),
        ],
        
        n_conjugated_bonds=11,
        chromophore_type="apocarotenoid",
        
        food_sources=[
            "Annatto seeds (Bixa orellana, 1-5% dry weight)",
            "Cheese (yellow/orange colorant, E160b)",
            "Butter, margarine (colorant)",
            "Snack foods (Doritos, Cheetos)",
            "Cosmetics (lipstick, blush)",
        ],
        
        biological_function=(
            "Major pigment in annatto seeds. Widely used natural food colorant "
            "(E160b, approved worldwide). C25 apocarotenoid. Mono-methyl ester "
            "(lipophilic) vs norbixin (di-acid, water-soluble). Orange-yellow color."
        ),
        
        solubility_class="lipophilic",  # Methyl ester more lipophilic than crocetin
        stability_notes=(
            "Good stability in foods. Ester group stable. Light and heat tolerant "
            "compared to other carotenoids. Used in processed foods."
        ),
    ),
}


# ============================================================================
# SECTION 3: DATABASE MANAGER
# ============================================================================

class AdditionalCarotenoidsManager:
    """Manager for additional carotenoids database (Part 2)"""
    
    def __init__(self):
        self.database = ADDITIONAL_CAROTENOIDS_DATABASE
        self.wavelength_index = self._build_wavelength_index()
        self.food_index = self._build_food_index()
        logger.info(f"Loaded {len(self.database)} additional carotenoids")
    
    def _build_wavelength_index(self) -> Dict[int, List[str]]:
        index = {}
        for name, carotenoid in self.database.items():
            bin_center = int(round(carotenoid.lambda_max / 5.0) * 5)
            if bin_center not in index:
                index[bin_center] = []
            index[bin_center].append(name)
        return index
    
    def _build_food_index(self) -> Dict[str, List[str]]:
        index = {}
        for name, carotenoid in self.database.items():
            for food in carotenoid.food_sources:
                food_lower = food.lower()
                if food_lower not in index:
                    index[food_lower] = []
                index[food_lower].append(name)
        return index
    
    def search_by_wavelength(self, lambda_nm: float, tolerance_nm: float = 10.0) -> List[str]:
        matches = []
        for name, carotenoid in self.database.items():
            if abs(carotenoid.lambda_max - lambda_nm) <= tolerance_nm:
                matches.append(name)
        return matches
    
    def get_statistics(self) -> Dict[str, any]:
        n_xanthophylls = sum(1 for c in self.database.values() if c.chromophore_type == "xanthophyll")
        n_apocarotenoids = sum(1 for c in self.database.values() if c.chromophore_type == "apocarotenoid")
        
        lambda_values = [c.lambda_max for c in self.database.values()]
        conjugation_values = [c.n_conjugated_bonds for c in self.database.values()]
        
        return {
            "total_compounds": len(self.database),
            "xanthophylls": n_xanthophylls,
            "apocarotenoids": n_apocarotenoids,
            "wavelength_range": (min(lambda_values), max(lambda_values)),
            "conjugation_range": (min(conjugation_values), max(conjugation_values)),
            "unique_foods": len(self.food_index),
        }


# ============================================================================
# SECTION 4: DEMO & VALIDATION
# ============================================================================

def demo_additional_carotenoids():
    print("\n" + "="*70)
    print("ADDITIONAL CAROTENOIDS DATABASE - PHASE 2 PART 4d")
    print("="*70)
    
    manager = AdditionalCarotenoidsManager()
    stats = manager.get_statistics()
    
    print(f"\n📊 DATABASE STATISTICS:")
    print(f"   Total compounds: {stats['total_compounds']}")
    print(f"   Xanthophylls: {stats['xanthophylls']}")
    print(f"   Apocarotenoids: {stats['apocarotenoids']}")
    print(f"   λmax range: {stats['wavelength_range'][0]:.0f}-{stats['wavelength_range'][1]:.0f} nm")
    print(f"   Conjugation range: {stats['conjugation_range'][0]}-{stats['conjugation_range'][1]} bonds")
    print(f"   Food sources: {stats['unique_foods']}")
    
    print(f"\n🍣 FOOD COLORANTS:")
    colorants = ["canthaxanthin", "crocetin", "bixin"]
    for name in colorants:
        carotenoid = manager.database[name]
        print(f"   ✓ {carotenoid.name}: {carotenoid.food_sources[0]}")
    
    print(f"\n🌿 XANTHOPHYLL CYCLE:")
    xanthophyll_cycle = ["violaxanthin", "neoxanthin"]
    for name in xanthophyll_cycle:
        carotenoid = manager.database[name]
        print(f"   ✓ {carotenoid.name}: λ={carotenoid.lambda_max:.0f} nm, "
              f"{carotenoid.biological_function[:50]}...")
    
    print(f"\n🔴 RED-SHIFTED PIGMENTS (>500 nm):")
    for name, carotenoid in manager.database.items():
        if carotenoid.lambda_max >= 500:
            print(f"   ✓ {carotenoid.name}: λ={carotenoid.lambda_max:.0f} nm, "
                  f"{carotenoid.n_conjugated_bonds} conjugated bonds")
    
    print(f"\n⚗️ SPECIAL STRUCTURES:")
    special = {
        "neoxanthin": "Allenic bond (C=C=C)",
        "crocetin": "Di-carboxylic acid (amphiphilic)",
        "bixin": "Mono-methyl ester",
    }
    for name, feature in special.items():
        carotenoid = manager.database[name]
        print(f"   ✓ {carotenoid.name}: {feature}")
    
    print(f"\n✅ Additional carotenoids database ready!")
    print(f"   Phase 4d complete: +9 carotenoids")
    print(f"   Total carotenoid count: 10 (original) + 8 (Part 1) + 9 (Part 2) = 27")
    print("="*70 + "\n")


if __name__ == "__main__":
    demo_additional_carotenoids()
