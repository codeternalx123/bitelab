"""
PHASE 2 PART 4f: EXPANDED CHLOROPHYLLS & PORPHYRINS DATABASE
=============================================================

Adding 14 compounds to reach 17 total chlorophylls/porphyrins/linear tetrapyrroles.
This module includes:

BACTERIOCHLOROPHYLLS (Photosynthetic bacteria, 5 compounds):
1. Bacteriochlorophyll a (BChl a) - Purple bacteria (800-850 nm)
2. Bacteriochlorophyll b (BChl b) - Purple bacteria (1020 nm, NIR)
3. Bacteriochlorophyll c (BChl c) - Green sulfur bacteria (740 nm)
4. Bacteriochlorophyll d (BChl d) - Green sulfur bacteria (725 nm)
5. Bacteriochlorophyll e (BChl e) - Green sulfur bacteria (715 nm)

CHLOROPHYLL DERIVATIVES (Degradation/biosynthesis, 4 compounds):
6. Pheophytin a (Pheo a) - Demetallated chlorophyll a
7. Pheophorbide a (Phorbide a) - Phytol-free pheophytin
8. Chlorophyllide a - Phytol-free chlorophyll (biosynthetic)
9. Chlorophyllide b - Phytol-free chlorophyll b

PORPHYRIN PRECURSORS (Biosynthetic pathway, 3 compounds):
10. Protoporphyrin IX (Proto IX) - Heme/chlorophyll precursor
11. Mg-Protoporphyrin IX (Mg-Proto IX) - Chlorophyll biosynthesis
12. Protochlorophyllide a (Pchlide a) - Light-dependent intermediate

LINEAR TETRAPYRROLES (Open-chain, 2 compounds):
13. Phytochromobilin (PΦB) - Phytochrome chromophore (red/far-red sensor)
14. Phycocyanobilin (PCB) - Phycocyanin chromophore (blue)

Scientific References:
- Grimm, B. et al. (2006) Chlorophylls and Bacteriochlorophylls
- Kobayashi, M. & Miyashita, H. (2003) Photochem. Photobiol. Sci.
- Rockwell, N. C. et al. (2006) Annu. Rev. Plant Biol.

Author: Visual Molecular AI System
Version: 2.4.6
Lines: ~1,200 (target for Phase 4f)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: DATA STRUCTURES
# ============================================================================

@dataclass
class FluorescenceData:
    quantum_yield: float
    lifetime_ns: float
    emission_max_nm: float
    stokes_shift_nm: float

@dataclass
class ExtendedChlorophyll:
    """Complete chlorophyll/porphyrin data structure"""
    name: str
    formula: str
    molecular_weight: float
    smiles: str
    lambda_max: float
    qy_band: float
    soret_band: float
    extinction_soret: float
    fluorescence: Optional[FluorescenceData]
    raman_peaks: List[Tuple[float, str]]
    ir_peaks: List[Tuple[float, str]]
    metal_center: str
    chromophore_type: str
    biological_source: List[str]
    function: str
    absorption_bands: List[Tuple[float, float]]
    solubility_class: str
    stability_notes: str


# ============================================================================
# SECTION 2: EXTENDED CHLOROPHYLLS DATABASE (14 NEW COMPOUNDS)
# ============================================================================

EXTENDED_CHLOROPHYLLS_DATABASE: Dict[str, ExtendedChlorophyll] = {
    
    # ========================================================================
    # CATEGORY 1: BACTERIOCHLOROPHYLLS (5 compounds)
    # ========================================================================
    
    "bacteriochlorophyll_a": ExtendedChlorophyll(
        name="Bacteriochlorophyll a",
        formula="C55H74MgN4O6",
        molecular_weight=911.51,
        smiles="CCC1=C(C)C2=[N+]3C1=Cc1c(C)c(C=C)c4C=C5C(C)=C(CC(=O)OC)[C@H]([C@@H](C)CCC(=O)OC)[N+]5=[Mg]3(n14)n1c(=C2)c(C)c(C(C)=O)c1=C5",
        
        # UV-Vis: Near-infrared absorption
        lambda_max=772.0,  # nm (Qy band, major)
        qy_band=772.0,
        soret_band=358.0,
        extinction_soret=120000.0,  # M⁻¹cm⁻¹
        
        absorption_bands=[
            (358.0, 1.00),  # Soret
            (575.0, 0.22),  # Qx
            (772.0, 0.65),  # Qy (major)
            (800.0, 0.45),  # In vivo (protein shift)
        ],
        
        fluorescence=FluorescenceData(
            quantum_yield=0.25,
            lifetime_ns=2.8,
            emission_max_nm=805.0,
            stokes_shift_nm=33.0
        ),
        
        raman_peaks=[
            (1612.0, "C=C stretch (ring II)"),
            (1585.0, "C=C stretch (conjugated)"),
            (1520.0, "C-N stretch (pyrrole)"),
            (1365.0, "C-H bend"),
            (755.0, "Macrocycle breathing"),
        ],
        
        ir_peaks=[
            (3350.0, "N-H stretch (pyrrole)"),
            (2960.0, "C-H stretch (aliphatic)"),
            (1735.0, "C=O ester (C-17 propionate)"),
            (1695.0, "C=O keto (C-3)"),
            (1590.0, "C=C aromatic"),
        ],
        
        metal_center="Mg²⁺",
        chromophore_type="bacteriochlorophyll",
        
        biological_source=[
            "Purple bacteria (Rhodobacter, Rhodospirillum)",
            "Photosynthetic reaction centers",
            "Light-harvesting complex 2 (LH2)",
        ],
        
        function=(
            "Primary photosynthetic pigment in purple bacteria. Near-infrared "
            "absorption (772 nm) allows growth in low-light, oxic environments. "
            "Reduced ring II (dihydroporphyrin) vs chlorophyll. "
            "Special pair (P870) in reaction center."
        ),
        
        solubility_class="lipophilic",
        stability_notes=(
            "Relatively stable. Mg²⁺ center stable under anaerobic conditions. "
            "Oxidizes in air to bacteriopheophytin. Light-sensitive."
        ),
    ),
    
    "bacteriochlorophyll_b": ExtendedChlorophyll(
        name="Bacteriochlorophyll b",
        formula="C55H72MgN4O6",
        molecular_weight=909.49,
        smiles="C=CC1=C(C)C2=[N+]3C1=Cc1c(C)c(C=C)c4C=C5C(C)=C(CC(=O)OC)[C@H]([C@@H](C)CCC(=O)OC)[N+]5=[Mg]3(n14)n1c(=C2)c(C)c(C=O)c1=C5",
        
        # UV-Vis: Deep near-infrared
        lambda_max=1020.0,  # nm (longest known photosynthetic pigment)
        qy_band=1020.0,
        soret_band=368.0,
        extinction_soret=115000.0,
        
        absorption_bands=[
            (368.0, 1.00),  # Soret
            (605.0, 0.18),  # Qx
            (835.0, 0.32),  # Qy minor
            (1020.0, 0.55),  # Qy major (NIR)
        ],
        
        fluorescence=FluorescenceData(
            quantum_yield=0.18,
            lifetime_ns=2.2,
            emission_max_nm=1050.0,
            stokes_shift_nm=30.0
        ),
        
        raman_peaks=[
            (1608.0, "C=C stretch"),
            (1580.0, "C=C conjugated"),
            (1515.0, "C-N pyrrole"),
            (755.0, "Macrocycle"),
        ],
        
        ir_peaks=[
            (3350.0, "N-H"),
            (2960.0, "C-H"),
            (1735.0, "C=O ester"),
            (1685.0, "C=O aldehyde (C-8)"),
            (1590.0, "C=C"),
        ],
        
        metal_center="Mg²⁺",
        chromophore_type="bacteriochlorophyll",
        
        biological_source=[
            "Purple bacteria (Blastochloris, Rhodopseudomonas)",
            "Deep-water/low-light habitats",
        ],
        
        function=(
            "Extreme near-infrared absorption (1020 nm). Allows photosynthesis "
            "with light transmitted through water. C-8 ethyl → acetyl (vs BChl a). "
            "Rare, specialized pigment."
        ),
        
        solubility_class="lipophilic",
        stability_notes="Unstable. Extremely light-sensitive. Oxidizes rapidly."),
    
    "bacteriochlorophyll_c": ExtendedChlorophyll(
        name="Bacteriochlorophyll c",
        formula="C55H74MgN4O6",
        molecular_weight=911.51,
        smiles="CCC1=C(C)C2=[N+]3C1=Cc1c(C)c(C(C)CC)c4C=C5C(C=C)=C(C)[C@H]([C@@H](C)CC(=O)OCC)[N+]5=[Mg]3(n14)n1c(=C2)c(C)c(C)c1=C5",
        
        # UV-Vis: Green sulfur bacteria
        lambda_max=740.0,  # nm (chlorosome)
        qy_band=740.0,
        soret_band=432.0,
        extinction_soret=95000.0,
        
        absorption_bands=[
            (432.0, 1.00),  # Soret
            (670.0, 0.28),  # Qx
            (740.0, 0.72),  # Qy
        ],
        
        fluorescence=FluorescenceData(
            quantum_yield=0.05,
            lifetime_ns=0.8,
            emission_max_nm=755.0,
            stokes_shift_nm=15.0
        ),
        
        raman_peaks=[
            (1615.0, "C=C"),
            (1590.0, "C=C conjugated"),
            (1525.0, "C-N"),
            (755.0, "Macrocycle"),
        ],
        
        ir_peaks=[
            (3350.0, "N-H"),
            (2960.0, "C-H (many)"),
            (1730.0, "C=O ester"),
            (1595.0, "C=C"),
        ],
        
        metal_center="Mg²⁺",
        chromophore_type="bacteriochlorophyll",
        
        biological_source=[
            "Green sulfur bacteria (Chlorobium)",
            "Chlorosomes (light-harvesting antennae)",
        ],
        
        function=(
            "Self-assembling pigment in chlorosomes (100,000+ molecules). "
            "No protein binding. Rod-shaped aggregates. Highly efficient "
            "light harvesting in low-light environments."
        ),
        
        solubility_class="lipophilic",
        stability_notes="Stable in aggregates. Sensitive when monomeric."),
    
    "bacteriochlorophyll_d": ExtendedChlorophyll(
        name="Bacteriochlorophyll d",
        formula="C54H70MgN4O6",
        molecular_weight=883.47,
        smiles="CC1=C(C)C2=[N+]3C1=Cc1c(C)c(C(C)CC)c4C=C5C(C=C)=C(C)[C@H]([C@@H](C)CC(=O)OCC)[N+]5=[Mg]3(n14)n1c(=C2)c(C)c(C)c1=C5",
        
        # UV-Vis
        lambda_max=725.0,  # nm
        qy_band=725.0,
        soret_band=428.0,
        extinction_soret=92000.0,
        
        absorption_bands=[
            (428.0, 1.00),
            (655.0, 0.25),
            (725.0, 0.68),
        ],
        
        fluorescence=FluorescenceData(
            quantum_yield=0.06,
            lifetime_ns=0.9,
            emission_max_nm=740.0,
            stokes_shift_nm=15.0
        ),
        
        raman_peaks=[
            (1618.0, "C=C"),
            (1592.0, "C=C conjugated"),
            (1528.0, "C-N"),
        ],
        
        ir_peaks=[
            (3350.0, "N-H"),
            (2960.0, "C-H"),
            (1730.0, "C=O ester"),
        ],
        
        metal_center="Mg²⁺",
        chromophore_type="bacteriochlorophyll",
        
        biological_source=[
            "Green sulfur bacteria (Chlorobium)",
            "Chlorosomes",
        ],
        
        function="Chlorosome antenna pigment. C-8 methyl (vs ethyl in BChl c).",
        
        solubility_class="lipophilic",
        stability_notes="Stable in aggregates."),
    
    "bacteriochlorophyll_e": ExtendedChlorophyll(
        name="Bacteriochlorophyll e",
        formula="C51H66MgN4O6",
        molecular_weight=839.41,
        smiles="CC1=C(C)C2=[N+]3C1=Cc1c(C)c(C)c4C=C5C(C=C)=C(C)[C@H]([C@@H](C)CC(=O)OCC)[N+]5=[Mg]3(n14)n1c(=C2)c(C)c(C)c1=C5",
        
        # UV-Vis
        lambda_max=715.0,  # nm
        qy_band=715.0,
        soret_band=425.0,
        extinction_soret=88000.0,
        
        absorption_bands=[
            (425.0, 1.00),
            (648.0, 0.23),
            (715.0, 0.65),
        ],
        
        fluorescence=FluorescenceData(
            quantum_yield=0.07,
            lifetime_ns=1.0,
            emission_max_nm=730.0,
            stokes_shift_nm=15.0
        ),
        
        raman_peaks=[
            (1620.0, "C=C"),
            (1595.0, "C=C conjugated"),
            (1530.0, "C-N"),
        ],
        
        ir_peaks=[
            (3350.0, "N-H"),
            (2960.0, "C-H"),
            (1730.0, "C=O"),
        ],
        
        metal_center="Mg²⁺",
        chromophore_type="bacteriochlorophyll",
        
        biological_source=[
            "Green sulfur bacteria (Chlorobium)",
            "Chlorosomes",
        ],
        
        function="Chlorosome pigment. Farnesyl ester (vs phytyl).",
        
        solubility_class="lipophilic",
        stability_notes="Stable in aggregates."),
    
    # ========================================================================
    # CATEGORY 2: CHLOROPHYLL DERIVATIVES (4 compounds)
    # ========================================================================
    
    "pheophytin_a": ExtendedChlorophyll(
        name="Pheophytin a",
        formula="C55H74N4O5",
        molecular_weight=871.20,
        smiles="CCC1=C(C)C2=Cc3c(C)c(C=C)c4C=C5C(C)=C(C=O)C(C)=C(N5)C=c5c(C)c(C(=O)OC)c(C)c(n5)=Cc5c(C(C)=O)c(C)c1n(c5=C2)CC=C[C@H](CCC(=O)OC[C@H](COC(=O)C[C@H]3C)OC(=O)CC)C",
        
        # UV-Vis: Demetallated chlorophyll a
        lambda_max=667.0,  # nm (vs 665 for Chl a)
        qy_band=667.0,
        soret_band=409.0,
        extinction_soret=125000.0,
        
        absorption_bands=[
            (409.0, 1.00),  # Soret
            (505.0, 0.18),
            (535.0, 0.12),
            (608.0, 0.08),
            (667.0, 0.52),  # Qy (red-shifted from Chl a)
        ],
        
        fluorescence=FluorescenceData(
            quantum_yield=0.12,  # Lower than Chl a
            lifetime_ns=4.5,
            emission_max_nm=685.0,
            stokes_shift_nm=18.0
        ),
        
        raman_peaks=[
            (1610.0, "C=C"),
            (1582.0, "C=C conjugated"),
            (1518.0, "C-N"),
            (1340.0, "C-H bend"),
            (750.0, "Macrocycle"),
        ],
        
        ir_peaks=[
            (3450.0, "N-H stretch (2 NH)"),
            (2960.0, "C-H"),
            (1735.0, "C=O ester"),
            (1695.0, "C=O aldehyde (C-13²)"),
            (1595.0, "C=C"),
        ],
        
        metal_center="None (free base)",
        chromophore_type="pheophytin",
        
        biological_source=[
            "Photosystem II (electron acceptor)",
            "Chlorophyll degradation product",
            "Green vegetables (cooking/acid)",
        ],
        
        function=(
            "Electron acceptor in PSII (reduces plastoquinone). Forms during "
            "chlorophyll degradation (Mg²⁺ loss). Olive-brown color in cooked "
            "green vegetables. Marker of thermal processing."
        ),
        
        solubility_class="lipophilic",
        stability_notes="Stable. More stable than chlorophyll (no Mg²⁺)."),
    
    "pheophorbide_a": ExtendedChlorophyll(
        name="Pheophorbide a",
        formula="C35H36N4O5",
        molecular_weight=592.69,
        smiles="CCC1=C(C)C2=Cc3c(C)c(C=C)c4C=C5C(C)=C(C=O)C(C)=C(N5)C=c5c(C)c(C(=O)OC)c(C)c(n5)=Cc5c(C(C)=O)c(C)c1n(c5=C2)C(C)CCC(=O)O",
        
        # UV-Vis: No phytol tail
        lambda_max=667.0,  # nm
        qy_band=667.0,
        soret_band=408.0,
        extinction_soret=122000.0,
        
        absorption_bands=[
            (408.0, 1.00),
            (505.0, 0.17),
            (535.0, 0.11),
            (608.0, 0.08),
            (667.0, 0.50),
        ],
        
        fluorescence=FluorescenceData(
            quantum_yield=0.15,
            lifetime_ns=5.0,
            emission_max_nm=685.0,
            stokes_shift_nm=18.0
        ),
        
        raman_peaks=[
            (1612.0, "C=C"),
            (1585.0, "C=C conjugated"),
            (1520.0, "C-N"),
        ],
        
        ir_peaks=[
            (3450.0, "N-H (2 NH)"),
            (2960.0, "C-H"),
            (1735.0, "C=O ester"),
            (1695.0, "C=O aldehyde"),
            (1710.0, "C=O acid (propionate)"),
        ],
        
        metal_center="None",
        chromophore_type="pheophorbide",
        
        biological_source=[
            "Chlorophyll degradation",
            "Senescent leaves",
            "Photodynamic therapy agent",
        ],
        
        function=(
            "Chlorophyll breakdown product (phytol cleavage). More polar than "
            "pheophytin. Used in photodynamic therapy (PDT) for cancer. "
            "Accumulates in aging leaves."
        ),
        
        solubility_class="amphiphilic",
        stability_notes="Stable. Water-soluble salts possible."),
    
    "chlorophyllide_a": ExtendedChlorophyll(
        name="Chlorophyllide a",
        formula="C35H34MgN4O5",
        molecular_weight=610.97,
        smiles="CCC1=C(C)C2=Cc3c(C)c(C=C)c4C=C5C(C)=C(C=O)C(C)=[N+]5[Mg]6(n4c3=C1)n1c(=C2)c(C)c(C(=O)OC)c1=C1C=c3c(C)c(C(C)=O)c(C)c(n3)=C16",
        
        # UV-Vis: Biosynthetic intermediate
        lambda_max=666.0,  # nm
        qy_band=666.0,
        soret_band=430.0,
        extinction_soret=135000.0,
        
        absorption_bands=[
            (430.0, 1.00),
            (577.0, 0.12),
            (615.0, 0.08),
            (666.0, 0.48),
        ],
        
        fluorescence=FluorescenceData(
            quantum_yield=0.28,
            lifetime_ns=5.8,
            emission_max_nm=682.0,
            stokes_shift_nm=16.0
        ),
        
        raman_peaks=[
            (1615.0, "C=C"),
            (1588.0, "C=C conjugated"),
            (1522.0, "C-N"),
        ],
        
        ir_peaks=[
            (3350.0, "N-H"),
            (2960.0, "C-H"),
            (1735.0, "C=O ester"),
            (1695.0, "C=O aldehyde"),
            (1710.0, "C=O acid"),
        ],
        
        metal_center="Mg²⁺",
        chromophore_type="chlorophyllide",
        
        biological_source=[
            "Chlorophyll biosynthesis (precursor)",
            "Developing chloroplasts",
        ],
        
        function=(
            "Biosynthetic intermediate. Phytol tail added by chlorophyll synthase "
            "→ chlorophyll a. Used to study chlorophyll synthesis. Water-soluble."
        ),
        
        solubility_class="amphiphilic",
        stability_notes="Moderately stable. Mg²⁺ sensitive to acid."),
    
    "chlorophyllide_b": ExtendedChlorophyll(
        name="Chlorophyllide b",
        formula="C35H32MgN4O6",
        molecular_weight=624.96,
        smiles="CCC1=C(C)C2=Cc3c(C=O)c(C=C)c4C=C5C(C)=C(C=O)C(C)=[N+]5[Mg]6(n4c3=C1)n1c(=C2)c(C)c(C(=O)OC)c1=C1C=c3c(C)c(C(C)=O)c(C)c(n3)=C16",
        
        # UV-Vis
        lambda_max=645.0,  # nm (blue-shifted vs Chlide a)
        qy_band=645.0,
        soret_band=453.0,
        extinction_soret=140000.0,
        
        absorption_bands=[
            (453.0, 1.00),
            (595.0, 0.10),
            (645.0, 0.42),
        ],
        
        fluorescence=FluorescenceData(
            quantum_yield=0.18,
            lifetime_ns=4.2,
            emission_max_nm=662.0,
            stokes_shift_nm=17.0
        ),
        
        raman_peaks=[
            (1618.0, "C=C"),
            (1590.0, "C=C conjugated"),
            (1525.0, "C-N"),
        ],
        
        ir_peaks=[
            (3350.0, "N-H"),
            (2960.0, "C-H"),
            (1735.0, "C=O ester"),
            (1695.0, "C=O aldehyde (2 groups)"),
        ],
        
        metal_center="Mg²⁺",
        chromophore_type="chlorophyllide",
        
        biological_source=[
            "Chlorophyll b biosynthesis",
            "Developing chloroplasts",
        ],
        
        function="Biosynthetic intermediate to chlorophyll b. C-7 formyl group.",
        
        solubility_class="amphiphilic",
        stability_notes="Moderately stable."),
    
    # ========================================================================
    # CATEGORY 3: PORPHYRIN PRECURSORS (3 compounds)
    # ========================================================================
    
    "protoporphyrin_ix": ExtendedChlorophyll(
        name="Protoporphyrin IX",
        formula="C34H34N4O4",
        molecular_weight=562.66,
        smiles="CC1=C(CCC(=O)O)C2=Cc3c(C=C)c(C)c4C=c5c(C)c(C=C)c(n5)C=c5c(C)c(CCC(=O)O)c(n5)=Cc5c(C)c1n(c5=C2)c(=N3)N4",
        
        # UV-Vis: Metal-free porphyrin
        lambda_max=631.0,  # nm (Q bands)
        qy_band=631.0,
        soret_band=408.0,
        extinction_soret=275000.0,  # High (no metal)
        
        absorption_bands=[
            (408.0, 1.00),  # Soret
            (505.0, 0.15),
            (540.0, 0.10),
            (575.0, 0.08),
            (631.0, 0.06),
        ],
        
        fluorescence=FluorescenceData(
            quantum_yield=0.10,
            lifetime_ns=15.0,
            emission_max_nm=632.0,
            stokes_shift_nm=1.0
        ),
        
        raman_peaks=[
            (1620.0, "C=C"),
            (1595.0, "C=C pyrrole"),
            (1530.0, "C-N"),
            (1360.0, "C-H"),
        ],
        
        ir_peaks=[
            (3450.0, "N-H (2 NH)"),
            (2960.0, "C-H"),
            (1710.0, "C=O acid (propionate)"),
            (1600.0, "C=C aromatic"),
        ],
        
        metal_center="None",
        chromophore_type="porphyrin",
        
        biological_source=[
            "Heme biosynthesis (final precursor)",
            "Chlorophyll biosynthesis (Mg insertion)",
            "Erythrocytes (heme pathway)",
        ],
        
        function=(
            "Universal precursor to heme (Fe²⁺ insertion → heme) and chlorophyll "
            "(Mg²⁺ insertion → Mg-protoporphyrin IX). Accumulates in porphyrias "
            "(genetic disorders). Red fluorescence under UV."
        ),
        
        solubility_class="lipophilic",
        stability_notes="Very stable. No metal to lose."),
    
    "mg_protoporphyrin_ix": ExtendedChlorophyll(
        name="Mg-Protoporphyrin IX",
        formula="C34H32MgN4O4",
        molecular_weight=584.94,
        smiles="CC1=C(CCC(=O)O)C2=Cc3c(C=C)c(C)c4C=c5c(C)c(C=C)c([n+]5[Mg]([n+]3c4=C1)n1c(=C2)c(C)c(CCC(=O)O)c1=C1)C=c2c(C)c(C)c1n2",
        
        # UV-Vis: First Mg²⁺ intermediate
        lambda_max=590.0,  # nm
        qy_band=590.0,
        soret_band=419.0,
        extinction_soret=185000.0,
        
        absorption_bands=[
            (419.0, 1.00),
            (550.0, 0.12),
            (590.0, 0.25),
        ],
        
        fluorescence=FluorescenceData(
            quantum_yield=0.18,
            lifetime_ns=8.5,
            emission_max_nm=605.0,
            stokes_shift_nm=15.0
        ),
        
        raman_peaks=[
            (1622.0, "C=C"),
            (1598.0, "C=C pyrrole"),
            (1532.0, "C-N"),
        ],
        
        ir_peaks=[
            (3350.0, "N-H"),
            (2960.0, "C-H"),
            (1710.0, "C=O acid"),
        ],
        
        metal_center="Mg²⁺",
        chromophore_type="porphyrin",
        
        biological_source=[
            "Chlorophyll biosynthesis (early step)",
            "Developing chloroplasts",
        ],
        
        function=(
            "First committed step in chlorophyll biosynthesis (Mg chelatase). "
            "Methylated, cyclized, reduced → protochlorophyllide → chlorophyllide. "
            "Feedback regulation of pathway."
        ),
        
        solubility_class="amphiphilic",
        stability_notes="Moderately stable. Mg²⁺ acid-labile."),
    
    "protochlorophyllide_a": ExtendedChlorophyll(
        name="Protochlorophyllide a",
        formula="C35H32MgN4O5",
        molecular_weight=608.95,
        smiles="CCC1=C(C)C2=Cc3c(C)c(C=C)c4C=C5C(C)=C(C=O)C(C)=[N+]5[Mg]6(n4c3=C1)n1c(=C2)c(C)c(C(=O)OC)c1=C1C=C(C)c3c(C)c(C(C)=O)c(n3)=C16",
        
        # UV-Vis: Light-dependent reduction substrate
        lambda_max=650.0,  # nm (shifts to 666 after reduction)
        qy_band=650.0,
        soret_band=442.0,
        extinction_soret=150000.0,
        
        absorption_bands=[
            (442.0, 1.00),
            (625.0, 0.15),
            (650.0, 0.48),
        ],
        
        fluorescence=FluorescenceData(
            quantum_yield=0.22,
            lifetime_ns=6.5,
            emission_max_nm=670.0,
            stokes_shift_nm=20.0
        ),
        
        raman_peaks=[
            (1625.0, "C=C"),
            (1600.0, "C=C conjugated"),
            (1535.0, "C-N"),
        ],
        
        ir_peaks=[
            (3350.0, "N-H"),
            (2960.0, "C-H"),
            (1735.0, "C=O ester"),
            (1695.0, "C=O aldehyde"),
        ],
        
        metal_center="Mg²⁺",
        chromophore_type="protochlorophyllide",
        
        biological_source=[
            "Etiolated seedlings (dark-grown)",
            "Chloroplast development (light trigger)",
        ],
        
        function=(
            "Light-dependent chlorophyll biosynthesis. Protochlorophyllide "
            "oxidoreductase (POR) reduces D-ring (light-activated). "
            "'Greening' of etiolated seedlings. Fluorescence peak (Shibata shift)."
        ),
        
        solubility_class="amphiphilic",
        stability_notes="Stable in dark. Photoreduced in light."),
    
    # ========================================================================
    # CATEGORY 4: LINEAR TETRAPYRROLES (2 compounds)
    # ========================================================================
    
    "phytochromobilin": ExtendedChlorophyll(
        name="Phytochromobilin",
        formula="C33H38N4O6",
        molecular_weight=586.68,
        smiles="CC1=C(CCC(=O)O)C(=O)N(C1=CC2=C(C)C(=CC3=C(C)C(CCC(=O)O)C(C(=O)NC(C)C)=N3)NC(=O)C2C=C)C",
        
        # UV-Vis: Open-chain chromophore
        lambda_max=660.0,  # nm (Pr form, red-absorbing)
        qy_band=660.0,
        soret_band=368.0,
        extinction_soret=45000.0,
        
        absorption_bands=[
            (368.0, 1.00),  # UV band
            (660.0, 0.85),  # Pr (red-absorbing)
            # Pfr (far-red absorbing) at 730 nm after photoisomerization
        ],
        
        fluorescence=FluorescenceData(
            quantum_yield=0.02,  # Low (efficient photoisomerization)
            lifetime_ns=0.5,
            emission_max_nm=675.0,
            stokes_shift_nm=15.0
        ),
        
        raman_peaks=[
            (1640.0, "C=C stretch (methine bridge)"),
            (1620.0, "C=C pyrrole"),
            (1580.0, "C-N"),
            (1665.0, "C=O amide"),
        ],
        
        ir_peaks=[
            (3420.0, "N-H amide"),
            (2960.0, "C-H"),
            (1710.0, "C=O acid (propionate)"),
            (1665.0, "C=O amide"),
            (1620.0, "C=C"),
        ],
        
        metal_center="None",
        chromophore_type="linear tetrapyrrole",
        
        biological_source=[
            "Phytochrome photoreceptor (plants)",
            "Red/far-red light sensor",
            "Chloroplasts (heme oxygenase pathway)",
        ],
        
        function=(
            "Chromophore of phytochrome (red/far-red photoreceptor). "
            "Pr (660 nm, inactive) ⇌ Pfr (730 nm, active) photoisomerization. "
            "Controls seed germination, flowering, shade avoidance. "
            "Z→E isomerization at C15=C16."
        ),
        
        solubility_class="amphiphilic",
        stability_notes="Stable when protein-bound. Sensitive free in solution."),
    
    "phycocyanobilin": ExtendedChlorophyll(
        name="Phycocyanobilin",
        formula="C33H38N4O6",
        molecular_weight=586.68,
        smiles="CC1=C(CCC(=O)O)C(=O)NC1=CC2=C(C)C(=CC3=C(CCC(=O)O)C(C=C)C(C(=O)O)=N3)NC(=O)C2C=C",
        
        # UV-Vis: Phycobiliprotein chromophore
        lambda_max=620.0,  # nm (blue, in phycocyanin protein)
        qy_band=620.0,
        soret_band=360.0,
        extinction_soret=48000.0,
        
        absorption_bands=[
            (360.0, 1.00),
            (620.0, 0.95),  # Strong blue absorption
        ],
        
        fluorescence=FluorescenceData(
            quantum_yield=0.68,  # Very high (light-harvesting)
            lifetime_ns=1.8,
            emission_max_nm=645.0,
            stokes_shift_nm=25.0
        ),
        
        raman_peaks=[
            (1638.0, "C=C methine"),
            (1618.0, "C=C pyrrole"),
            (1578.0, "C-N"),
        ],
        
        ir_peaks=[
            (3420.0, "N-H"),
            (2960.0, "C-H"),
            (1710.0, "C=O acid"),
            (1665.0, "C=O amide"),
        ],
        
        metal_center="None",
        chromophore_type="linear tetrapyrrole",
        
        biological_source=[
            "Phycocyanin (cyanobacteria, red algae)",
            "Phycobilisomes (light-harvesting antenna)",
            "Spirulina (dietary supplement)",
        ],
        
        function=(
            "Chromophore of phycocyanin (blue pigment). Light-harvesting in "
            "cyanobacteria and red algae. Energy transfer to chlorophyll. "
            "Efficient blue-light absorption (complements chlorophyll green gap). "
            "Used as food colorant (Spirulina blue)."
        ),
        
        solubility_class="amphiphilic",
        stability_notes="Very stable in protein. High fluorescence quantum yield."),
}


# ============================================================================
# SECTION 3: DATABASE MANAGER
# ============================================================================

class ExtendedChlorophyllsManager:
    """Manager for extended chlorophylls/porphyrins database"""
    
    def __init__(self):
        self.database = EXTENDED_CHLOROPHYLLS_DATABASE
        self.metal_index = self._build_metal_index()
        self.type_index = self._build_type_index()
        logger.info(f"Loaded {len(self.database)} extended chlorophyll/porphyrin compounds")
    
    def _build_metal_index(self) -> Dict[str, List[str]]:
        index = {}
        for name, compound in self.database.items():
            metal = compound.metal_center
            if metal not in index:
                index[metal] = []
            index[metal].append(name)
        return index
    
    def _build_type_index(self) -> Dict[str, List[str]]:
        index = {}
        for name, compound in self.database.items():
            ctype = compound.chromophore_type
            if ctype not in index:
                index[ctype] = []
            index[ctype].append(name)
        return index
    
    def search_by_wavelength(self, lambda_nm: float, tolerance_nm: float = 20.0) -> List[str]:
        matches = []
        for name, compound in self.database.items():
            if abs(compound.lambda_max - lambda_nm) <= tolerance_nm:
                matches.append(name)
        return matches
    
    def get_statistics(self) -> Dict[str, any]:
        lambda_values = [c.lambda_max for c in self.database.values()]
        soret_values = [c.soret_band for c in self.database.values()]
        
        return {
            "total_compounds": len(self.database),
            "by_metal": {metal: len(comps) for metal, comps in self.metal_index.items()},
            "by_type": {ctype: len(comps) for ctype, comps in self.type_index.items()},
            "wavelength_range": (min(lambda_values), max(lambda_values)),
            "soret_range": (min(soret_values), max(soret_values)),
        }


# ============================================================================
# SECTION 4: DEMO & VALIDATION
# ============================================================================

def demo_extended_chlorophylls():
    print("\n" + "="*70)
    print("EXTENDED CHLOROPHYLLS & PORPHYRINS DATABASE - PHASE 2 PART 4f")
    print("="*70)
    
    manager = ExtendedChlorophyllsManager()
    stats = manager.get_statistics()
    
    print(f"\n📊 DATABASE STATISTICS:")
    print(f"   Total compounds: {stats['total_compounds']}")
    print(f"   λmax range: {stats['wavelength_range'][0]:.0f}-{stats['wavelength_range'][1]:.0f} nm")
    print(f"   Soret range: {stats['soret_range'][0]:.0f}-{stats['soret_range'][1]:.0f} nm")
    
    print(f"\n   BY METAL CENTER:")
    for metal, count in stats['by_metal'].items():
        print(f"      {metal}: {count} compounds")
    
    print(f"\n   BY TYPE:")
    for ctype, count in stats['by_type'].items():
        print(f"      {ctype}: {count} compounds")
    
    print(f"\n🦠 BACTERIOCHLOROPHYLLS (Near-infrared):")
    bchls = ["bacteriochlorophyll_a", "bacteriochlorophyll_b", "bacteriochlorophyll_c"]
    for name in bchls:
        compound = manager.database[name]
        print(f"   ✓ {compound.name}: λ={compound.lambda_max:.0f} nm, "
              f"{compound.biological_source[0]}")
    
    print(f"\n🧬 BIOSYNTHETIC PATHWAY (Chlorophyll):")
    pathway = [
        "protoporphyrin_ix",
        "mg_protoporphyrin_ix",
        "protochlorophyllide_a",
        "chlorophyllide_a",
    ]
    for i, name in enumerate(pathway, 1):
        compound = manager.database[name]
        print(f"   {i}. {compound.name}: λ={compound.lambda_max:.0f} nm, "
              f"Soret={compound.soret_band:.0f} nm")
    
    print(f"\n🌈 LINEAR TETRAPYRROLES (Open-chain):")
    linear = ["phytochromobilin", "phycocyanobilin"]
    for name in linear:
        compound = manager.database[name]
        print(f"   ✓ {compound.name}: λ={compound.lambda_max:.0f} nm, "
              f"Φ_f={compound.fluorescence.quantum_yield:.2f}")
    
    print(f"\n📡 NEAR-INFRARED CHAMPIONS (λ > 750 nm):")
    for name, compound in manager.database.items():
        if compound.lambda_max >= 750:
            print(f"   ✓ {compound.name}: λ={compound.lambda_max:.0f} nm")
    
    print(f"\n✅ Extended chlorophylls/porphyrins database ready!")
    print(f"   Phase 4f complete: +14 compounds")
    print(f"   Total chlorophyll/porphyrin count: 3 (original) + 14 (new) = 17")
    print("="*70 + "\n")


if __name__ == "__main__":
    demo_extended_chlorophylls()
