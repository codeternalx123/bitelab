"""
Covalent Bond Library - Phase 3A Part 3
========================================

Comprehensive database of covalent bonds with:
- Bond energies (kJ/mol)
- Bond lengths (pm - picometers)
- NIR wavelength mappings
- Functional group associations
- Vibrational frequencies

This library enables molecular structure analysis and NIR spectroscopy
correlation for food composition detection.

Author: AI Nutrition Scanner Team
Date: November 2025
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class BondType(Enum):
    """Types of covalent bonds."""
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    AROMATIC = "aromatic"
    HYDROGEN = "hydrogen"
    COORDINATE = "coordinate"


class BondPolarity(Enum):
    """Bond polarity classification."""
    NONPOLAR = "nonpolar"  # ΔEN < 0.5
    POLAR = "polar"  # 0.5 ≤ ΔEN < 1.7
    IONIC = "ionic"  # ΔEN ≥ 1.7


class FunctionalGroup(Enum):
    """Common functional groups in organic molecules."""
    HYDROXYL = "hydroxyl"  # -OH
    CARBONYL = "carbonyl"  # C=O
    CARBOXYL = "carboxyl"  # -COOH
    AMINE = "amine"  # -NH2
    AMIDE = "amide"  # -CONH-
    ESTER = "ester"  # -COO-
    ETHER = "ether"  # -O-
    THIOL = "thiol"  # -SH
    SULFIDE = "sulfide"  # -S-
    DISULFIDE = "disulfide"  # -S-S-
    ALDEHYDE = "aldehyde"  # -CHO
    KETONE = "ketone"  # R-CO-R
    METHYL = "methyl"  # -CH3
    METHYLENE = "methylene"  # -CH2-
    PHENYL = "phenyl"  # benzene ring
    NITRILE = "nitrile"  # -C≡N
    NITRO = "nitro"  # -NO2
    PHOSPHATE = "phosphate"  # -PO4


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NIRAbsorption:
    """NIR absorption characteristics for a bond."""
    primary_wavelength: float  # nm (fundamental)
    overtone_wavelengths: List[float] = field(default_factory=list)  # nm
    combination_bands: List[Tuple[float, float]] = field(default_factory=list)  # (λ1, λ2)
    molar_absorptivity: float = 1.0  # L/(mol·cm) - Beer-Lambert coefficient
    bandwidth: float = 50.0  # nm (FWHM - Full Width Half Maximum)
    temperature_shift: float = 0.0  # nm/°C
    
    def get_all_wavelengths(self) -> List[float]:
        """Get all absorption wavelengths (primary + overtones)."""
        return [self.primary_wavelength] + self.overtone_wavelengths


@dataclass
class CovalentBond:
    """
    Comprehensive covalent bond data.
    
    Contains all chemical and spectroscopic properties needed for
    molecular identification and quantification.
    """
    # Identity
    name: str  # e.g., "C-H", "C=O", "N-H"
    atoms: Tuple[str, str]  # (atom1, atom2)
    bond_type: BondType
    
    # Thermodynamic Properties
    bond_energy: float  # kJ/mol (dissociation energy)
    bond_length: float  # pm (picometers)
    bond_length_range: Tuple[float, float] = (0.0, 0.0)  # (min, max) pm
    
    # Electronic Properties
    electronegativity_diff: float = 0.0  # ΔEN
    bond_polarity: BondPolarity = BondPolarity.NONPOLAR
    dipole_moment: float = 0.0  # Debye
    
    # Vibrational Spectroscopy
    vibrational_frequency: float = 0.0  # cm⁻¹ (IR fundamental)
    force_constant: float = 0.0  # N/m
    reduced_mass: float = 0.0  # amu
    
    # NIR Spectroscopy
    nir_absorption: Optional[NIRAbsorption] = None
    nir_active: bool = True  # Does it show up in NIR?
    
    # Chemical Context
    functional_groups: List[FunctionalGroup] = field(default_factory=list)
    common_molecules: List[str] = field(default_factory=list)
    
    # Additional Properties
    bond_order: float = 1.0  # 1, 2, 3, or fractional (aromatic)
    hybridization: Tuple[str, str] = ("sp3", "sp3")  # (atom1, atom2)
    rotation_barrier: Optional[float] = None  # kJ/mol (for single bonds)
    
    def __post_init__(self):
        """Calculate derived properties."""
        if self.bond_length_range == (0.0, 0.0):
            # Default range ±5% of bond length
            variation = self.bond_length * 0.05
            self.bond_length_range = (
                self.bond_length - variation,
                self.bond_length + variation
            )
        
        # Calculate force constant if not provided (Hooke's law approximation)
        if self.force_constant == 0.0 and self.vibrational_frequency > 0:
            # k = (2πν)² × μ
            # Convert cm⁻¹ to Hz: ν(Hz) = ν(cm⁻¹) × c (speed of light)
            c = 2.998e10  # cm/s
            freq_hz = self.vibrational_frequency * c
            if self.reduced_mass > 0:
                self.force_constant = (2 * math.pi * freq_hz)**2 * self.reduced_mass * 1.66054e-27
    
    def is_nir_active(self) -> bool:
        """Check if bond is NIR active."""
        return self.nir_active and self.nir_absorption is not None
    
    def get_nir_wavelengths(self) -> List[float]:
        """Get all NIR wavelengths for this bond."""
        if not self.is_nir_active():
            return []
        return self.nir_absorption.get_all_wavelengths()
    
    def calculate_energy_per_photon(self, wavelength_nm: float) -> float:
        """
        Calculate energy of photon at given wavelength.
        
        E = hc/λ
        
        Args:
            wavelength_nm: Wavelength in nanometers
        
        Returns:
            Energy in kJ/mol
        """
        h = 6.626e-34  # J·s (Planck's constant)
        c = 2.998e8  # m/s (speed of light)
        Na = 6.022e23  # Avogadro's number
        
        wavelength_m = wavelength_nm * 1e-9
        energy_J = (h * c) / wavelength_m
        energy_kJ_mol = (energy_J * Na) / 1000
        
        return energy_kJ_mol


@dataclass
class BondCategory:
    """Category of bonds for organization."""
    name: str
    description: str
    bonds: List[str] = field(default_factory=list)  # Bond names


# =============================================================================
# COVALENT BOND DATABASE
# =============================================================================

class CovalentBondLibrary:
    """
    Comprehensive library of covalent bonds.
    
    Provides lookup, analysis, and NIR correlation capabilities for
    molecular structure determination.
    """
    
    def __init__(self):
        """Initialize bond library."""
        self.bonds: Dict[str, CovalentBond] = {}
        self.categories: Dict[str, BondCategory] = {}
        self.functional_group_index: Dict[FunctionalGroup, List[str]] = {}
        self.nir_wavelength_index: Dict[float, List[str]] = {}
        
        self._populate_database()
        self._build_indices()
        
        logger.info(f"Initialized CovalentBondLibrary with {len(self.bonds)} bonds")
    
    def _populate_database(self):
        """Populate database with bond data."""
        
        # =====================================================================
        # CARBON-HYDROGEN BONDS (Most abundant in organic molecules)
        # =====================================================================
        
        self._add_bond(CovalentBond(
            name="C-H (alkane)",
            atoms=("C", "H"),
            bond_type=BondType.SINGLE,
            bond_energy=413,  # kJ/mol
            bond_length=109,  # pm
            bond_length_range=(106, 112),
            electronegativity_diff=0.4,
            bond_polarity=BondPolarity.NONPOLAR,
            vibrational_frequency=2900,  # cm⁻¹
            reduced_mass=0.923,  # amu
            nir_absorption=NIRAbsorption(
                primary_wavelength=1690,  # nm (first overtone of C-H stretch)
                overtone_wavelengths=[2310, 2350],  # nm (combinations)
                molar_absorptivity=0.5,
                bandwidth=60
            ),
            functional_groups=[FunctionalGroup.METHYL, FunctionalGroup.METHYLENE],
            common_molecules=["alkanes", "fatty_acids", "carbohydrates"],
            hybridization=("sp3", "s"),
            rotation_barrier=12.0  # kJ/mol (ethane)
        ))
        
        self._add_bond(CovalentBond(
            name="C-H (aromatic)",
            atoms=("C", "H"),
            bond_type=BondType.SINGLE,
            bond_energy=460,  # kJ/mol (stronger due to sp2)
            bond_length=108,  # pm
            bond_length_range=(106, 110),
            electronegativity_diff=0.4,
            bond_polarity=BondPolarity.NONPOLAR,
            vibrational_frequency=3030,  # cm⁻¹
            reduced_mass=0.923,
            nir_absorption=NIRAbsorption(
                primary_wavelength=1670,  # nm
                overtone_wavelengths=[2280],
                molar_absorptivity=0.4,
                bandwidth=50
            ),
            functional_groups=[FunctionalGroup.PHENYL],
            common_molecules=["benzene", "phenylalanine", "tyrosine", "tryptophan"],
            hybridization=("sp2", "s")
        ))
        
        self._add_bond(CovalentBond(
            name="C-H (aldehyde)",
            atoms=("C", "H"),
            bond_type=BondType.SINGLE,
            bond_energy=365,  # kJ/mol
            bond_length=111,  # pm
            electronegativity_diff=0.4,
            bond_polarity=BondPolarity.POLAR,
            vibrational_frequency=2750,  # cm⁻¹
            reduced_mass=0.923,
            nir_absorption=NIRAbsorption(
                primary_wavelength=1730,  # nm
                overtone_wavelengths=[2380],
                molar_absorptivity=0.6,
                bandwidth=55
            ),
            functional_groups=[FunctionalGroup.ALDEHYDE],
            common_molecules=["formaldehyde", "acetaldehyde", "glucose"],
            hybridization=("sp2", "s")
        ))
        
        # =====================================================================
        # CARBON-CARBON BONDS
        # =====================================================================
        
        self._add_bond(CovalentBond(
            name="C-C (single)",
            atoms=("C", "C"),
            bond_type=BondType.SINGLE,
            bond_energy=348,  # kJ/mol
            bond_length=154,  # pm
            bond_length_range=(150, 158),
            electronegativity_diff=0.0,
            bond_polarity=BondPolarity.NONPOLAR,
            vibrational_frequency=1000,  # cm⁻¹
            reduced_mass=6.0,
            nir_absorption=NIRAbsorption(
                primary_wavelength=2340,  # nm (weak, combination bands)
                overtone_wavelengths=[],
                molar_absorptivity=0.1,  # Very weak
                bandwidth=80
            ),
            nir_active=False,  # C-C single bonds are NIR inactive
            common_molecules=["ethane", "fatty_acids", "amino_acids"],
            hybridization=("sp3", "sp3"),
            rotation_barrier=12.0
        ))
        
        self._add_bond(CovalentBond(
            name="C=C (double)",
            atoms=("C", "C"),
            bond_type=BondType.DOUBLE,
            bond_energy=614,  # kJ/mol
            bond_length=134,  # pm
            bond_length_range=(132, 136),
            electronegativity_diff=0.0,
            bond_polarity=BondPolarity.NONPOLAR,
            vibrational_frequency=1650,  # cm⁻¹
            reduced_mass=6.0,
            nir_absorption=NIRAbsorption(
                primary_wavelength=1650,  # nm (first overtone)
                overtone_wavelengths=[2170],
                molar_absorptivity=0.3,
                bandwidth=70
            ),
            common_molecules=["alkenes", "fatty_acids", "carotenoids"],
            bond_order=2.0,
            hybridization=("sp2", "sp2"),
            rotation_barrier=264.0  # High barrier (restricted rotation)
        ))
        
        self._add_bond(CovalentBond(
            name="C≡C (triple)",
            atoms=("C", "C"),
            bond_type=BondType.TRIPLE,
            bond_energy=839,  # kJ/mol
            bond_length=120,  # pm
            bond_length_range=(118, 122),
            electronegativity_diff=0.0,
            bond_polarity=BondPolarity.NONPOLAR,
            vibrational_frequency=2100,  # cm⁻¹
            reduced_mass=6.0,
            nir_absorption=NIRAbsorption(
                primary_wavelength=2100,  # nm
                overtone_wavelengths=[],
                molar_absorptivity=0.2,
                bandwidth=60
            ),
            common_molecules=["acetylene", "alkynes"],
            bond_order=3.0,
            hybridization=("sp", "sp")
        ))
        
        self._add_bond(CovalentBond(
            name="C-C (aromatic)",
            atoms=("C", "C"),
            bond_type=BondType.AROMATIC,
            bond_energy=518,  # kJ/mol (average of single and double)
            bond_length=140,  # pm (intermediate)
            bond_length_range=(138, 142),
            electronegativity_diff=0.0,
            bond_polarity=BondPolarity.NONPOLAR,
            vibrational_frequency=1500,  # cm⁻¹
            reduced_mass=6.0,
            nir_absorption=NIRAbsorption(
                primary_wavelength=1600,  # nm
                overtone_wavelengths=[2000],
                molar_absorptivity=0.25,
                bandwidth=75
            ),
            functional_groups=[FunctionalGroup.PHENYL],
            common_molecules=["benzene", "phenylalanine", "flavonoids"],
            bond_order=1.5,  # Resonance
            hybridization=("sp2", "sp2")
        ))
        
        # =====================================================================
        # CARBON-OXYGEN BONDS
        # =====================================================================
        
        self._add_bond(CovalentBond(
            name="C-O (alcohol)",
            atoms=("C", "O"),
            bond_type=BondType.SINGLE,
            bond_energy=358,  # kJ/mol
            bond_length=143,  # pm
            bond_length_range=(140, 146),
            electronegativity_diff=1.0,
            bond_polarity=BondPolarity.POLAR,
            dipole_moment=1.7,  # Debye
            vibrational_frequency=1050,  # cm⁻¹
            reduced_mass=5.14,
            nir_absorption=NIRAbsorption(
                primary_wavelength=2100,  # nm (combination)
                overtone_wavelengths=[2270],
                molar_absorptivity=0.4,
                bandwidth=65
            ),
            functional_groups=[FunctionalGroup.HYDROXYL],
            common_molecules=["alcohols", "carbohydrates", "glycerol"],
            hybridization=("sp3", "sp3")
        ))
        
        self._add_bond(CovalentBond(
            name="C=O (carbonyl)",
            atoms=("C", "O"),
            bond_type=BondType.DOUBLE,
            bond_energy=799,  # kJ/mol
            bond_length=120,  # pm (ketone)
            bond_length_range=(118, 123),
            electronegativity_diff=1.0,
            bond_polarity=BondPolarity.POLAR,
            dipole_moment=2.3,
            vibrational_frequency=1715,  # cm⁻¹ (ketone)
            reduced_mass=5.14,
            nir_absorption=NIRAbsorption(
                primary_wavelength=1730,  # nm (first overtone)
                overtone_wavelengths=[2160, 2320],
                molar_absorptivity=0.8,  # Strong absorption
                bandwidth=70
            ),
            functional_groups=[FunctionalGroup.CARBONYL, FunctionalGroup.KETONE],
            common_molecules=["ketones", "quinones", "pyruvate"],
            bond_order=2.0,
            hybridization=("sp2", "sp2")
        ))
        
        self._add_bond(CovalentBond(
            name="C=O (aldehyde)",
            atoms=("C", "O"),
            bond_type=BondType.DOUBLE,
            bond_energy=745,  # kJ/mol
            bond_length=121,  # pm
            electronegativity_diff=1.0,
            bond_polarity=BondPolarity.POLAR,
            dipole_moment=2.7,
            vibrational_frequency=1730,  # cm⁻¹
            reduced_mass=5.14,
            nir_absorption=NIRAbsorption(
                primary_wavelength=1740,  # nm
                overtone_wavelengths=[2180],
                molar_absorptivity=0.7,
                bandwidth=65
            ),
            functional_groups=[FunctionalGroup.ALDEHYDE],
            common_molecules=["formaldehyde", "glucose", "ribose"],
            bond_order=2.0,
            hybridization=("sp2", "sp2")
        ))
        
        self._add_bond(CovalentBond(
            name="C=O (carboxylic acid)",
            atoms=("C", "O"),
            bond_type=BondType.DOUBLE,
            bond_energy=745,  # kJ/mol
            bond_length=123,  # pm
            electronegativity_diff=1.0,
            bond_polarity=BondPolarity.POLAR,
            dipole_moment=1.7,  # Lower due to resonance
            vibrational_frequency=1700,  # cm⁻¹
            reduced_mass=5.14,
            nir_absorption=NIRAbsorption(
                primary_wavelength=1720,  # nm
                overtone_wavelengths=[2140, 2300],
                molar_absorptivity=0.9,
                bandwidth=80
            ),
            functional_groups=[FunctionalGroup.CARBOXYL],
            common_molecules=["fatty_acids", "amino_acids", "citric_acid"],
            bond_order=1.5,  # Resonance with C-O
            hybridization=("sp2", "sp2")
        ))
        
        self._add_bond(CovalentBond(
            name="C=O (ester)",
            atoms=("C", "O"),
            bond_type=BondType.DOUBLE,
            bond_energy=745,  # kJ/mol
            bond_length=123,  # pm
            electronegativity_diff=1.0,
            bond_polarity=BondPolarity.POLAR,
            dipole_moment=1.9,
            vibrational_frequency=1735,  # cm⁻¹
            reduced_mass=5.14,
            nir_absorption=NIRAbsorption(
                primary_wavelength=1745,  # nm
                overtone_wavelengths=[2170],
                molar_absorptivity=0.8,
                bandwidth=70
            ),
            functional_groups=[FunctionalGroup.ESTER],
            common_molecules=["triglycerides", "esters", "waxes"],
            bond_order=2.0,
            hybridization=("sp2", "sp2")
        ))
        
        self._add_bond(CovalentBond(
            name="C=O (amide)",
            atoms=("C", "O"),
            bond_type=BondType.DOUBLE,
            bond_energy=745,  # kJ/mol
            bond_length=124,  # pm
            electronegativity_diff=1.0,
            bond_polarity=BondPolarity.POLAR,
            dipole_moment=3.7,  # High
            vibrational_frequency=1650,  # cm⁻¹ (Amide I)
            reduced_mass=5.14,
            nir_absorption=NIRAbsorption(
                primary_wavelength=1670,  # nm
                overtone_wavelengths=[2100],
                molar_absorptivity=0.9,
                bandwidth=75
            ),
            functional_groups=[FunctionalGroup.AMIDE],
            common_molecules=["proteins", "peptides", "nylon"],
            bond_order=1.5,  # Resonance with C-N
            hybridization=("sp2", "sp2")
        ))
        
        # =====================================================================
        # NITROGEN-HYDROGEN BONDS (Proteins, amino acids, amines)
        # =====================================================================
        
        self._add_bond(CovalentBond(
            name="N-H (amine)",
            atoms=("N", "H"),
            bond_type=BondType.SINGLE,
            bond_energy=391,  # kJ/mol
            bond_length=101,  # pm
            bond_length_range=(99, 103),
            electronegativity_diff=0.9,
            bond_polarity=BondPolarity.POLAR,
            dipole_moment=1.3,
            vibrational_frequency=3300,  # cm⁻¹
            reduced_mass=0.875,
            nir_absorption=NIRAbsorption(
                primary_wavelength=1510,  # nm (first overtone)
                overtone_wavelengths=[2050, 2180],  # nm (combinations)
                molar_absorptivity=0.6,
                bandwidth=55
            ),
            functional_groups=[FunctionalGroup.AMINE],
            common_molecules=["amino_acids", "proteins", "amines"],
            hybridization=("sp3", "s")
        ))
        
        self._add_bond(CovalentBond(
            name="N-H (amide)",
            atoms=("N", "H"),
            bond_type=BondType.SINGLE,
            bond_energy=460,  # kJ/mol (stronger due to resonance)
            bond_length=100,  # pm
            bond_length_range=(98, 102),
            electronegativity_diff=0.9,
            bond_polarity=BondPolarity.POLAR,
            dipole_moment=3.8,  # High due to C=O nearby
            vibrational_frequency=3280,  # cm⁻¹ (Amide A)
            reduced_mass=0.875,
            nir_absorption=NIRAbsorption(
                primary_wavelength=1520,  # nm
                overtone_wavelengths=[2030, 2180],
                molar_absorptivity=0.8,
                bandwidth=60
            ),
            functional_groups=[FunctionalGroup.AMIDE],
            common_molecules=["proteins", "peptides", "amino_acids"],
            hybridization=("sp2", "s")  # sp2 due to resonance
        ))
        
        self._add_bond(CovalentBond(
            name="N-H (aromatic)",
            atoms=("N", "H"),
            bond_type=BondType.SINGLE,
            bond_energy=435,  # kJ/mol
            bond_length=100,  # pm
            electronegativity_diff=0.9,
            bond_polarity=BondPolarity.POLAR,
            vibrational_frequency=3400,  # cm⁻¹
            reduced_mass=0.875,
            nir_absorption=NIRAbsorption(
                primary_wavelength=1490,  # nm
                overtone_wavelengths=[2000],
                molar_absorptivity=0.5,
                bandwidth=50
            ),
            common_molecules=["tryptophan", "histidine", "indole", "imidazole"],
            hybridization=("sp2", "s")
        ))
        
        # =====================================================================
        # OXYGEN-HYDROGEN BONDS (Alcohols, water, acids)
        # =====================================================================
        
        self._add_bond(CovalentBond(
            name="O-H (alcohol)",
            atoms=("O", "H"),
            bond_type=BondType.SINGLE,
            bond_energy=467,  # kJ/mol
            bond_length=96,  # pm
            bond_length_range=(94, 98),
            electronegativity_diff=1.4,
            bond_polarity=BondPolarity.POLAR,
            dipole_moment=1.7,
            vibrational_frequency=3600,  # cm⁻¹ (free OH)
            reduced_mass=0.948,
            nir_absorption=NIRAbsorption(
                primary_wavelength=1450,  # nm (first overtone)
                overtone_wavelengths=[1940, 2270],  # nm
                molar_absorptivity=1.0,  # Very strong
                bandwidth=100,  # Broad due to H-bonding
                temperature_shift=0.2  # H-bonding sensitive
            ),
            functional_groups=[FunctionalGroup.HYDROXYL],
            common_molecules=["alcohols", "carbohydrates", "water", "phenols"],
            hybridization=("sp3", "s")
        ))
        
        self._add_bond(CovalentBond(
            name="O-H (carboxylic acid)",
            atoms=("O", "H"),
            bond_type=BondType.SINGLE,
            bond_energy=467,  # kJ/mol
            bond_length=97,  # pm
            electronegativity_diff=1.4,
            bond_polarity=BondPolarity.POLAR,
            dipole_moment=1.7,
            vibrational_frequency=3000,  # cm⁻¹ (broad, H-bonded)
            reduced_mass=0.948,
            nir_absorption=NIRAbsorption(
                primary_wavelength=1430,  # nm
                overtone_wavelengths=[1930, 2600],
                molar_absorptivity=1.2,  # Very strong
                bandwidth=150  # Very broad
            ),
            functional_groups=[FunctionalGroup.CARBOXYL],
            common_molecules=["fatty_acids", "amino_acids", "citric_acid"],
            hybridization=("sp2", "s")
        ))
        
        self._add_bond(CovalentBond(
            name="O-H (phenol)",
            atoms=("O", "H"),
            bond_type=BondType.SINGLE,
            bond_energy=360,  # kJ/mol (weaker due to resonance)
            bond_length=96,  # pm
            electronegativity_diff=1.4,
            bond_polarity=BondPolarity.POLAR,
            dipole_moment=1.5,
            vibrational_frequency=3610,  # cm⁻¹
            reduced_mass=0.948,
            nir_absorption=NIRAbsorption(
                primary_wavelength=1440,  # nm
                overtone_wavelengths=[1950],
                molar_absorptivity=0.9,
                bandwidth=80
            ),
            functional_groups=[FunctionalGroup.HYDROXYL],
            common_molecules=["tyrosine", "phenols", "flavonoids", "vitamin_E"],
            hybridization=("sp2", "s")
        ))
        
        # =====================================================================
        # SULFUR BONDS (Cysteine, methionine, disulfides)
        # =====================================================================
        
        self._add_bond(CovalentBond(
            name="S-H (thiol)",
            atoms=("S", "H"),
            bond_type=BondType.SINGLE,
            bond_energy=363,  # kJ/mol
            bond_length=134,  # pm
            bond_length_range=(132, 136),
            electronegativity_diff=0.4,
            bond_polarity=BondPolarity.NONPOLAR,
            vibrational_frequency=2570,  # cm⁻¹
            reduced_mass=0.969,
            nir_absorption=NIRAbsorption(
                primary_wavelength=2550,  # nm (characteristic!)
                overtone_wavelengths=[],
                molar_absorptivity=0.4,
                bandwidth=45
            ),
            functional_groups=[FunctionalGroup.THIOL],
            common_molecules=["cysteine", "glutathione", "coenzyme_A"],
            hybridization=("sp3", "s")
        ))
        
        self._add_bond(CovalentBond(
            name="S-S (disulfide)",
            atoms=("S", "S"),
            bond_type=BondType.SINGLE,
            bond_energy=266,  # kJ/mol
            bond_length=204,  # pm
            bond_length_range=(200, 208),
            electronegativity_diff=0.0,
            bond_polarity=BondPolarity.NONPOLAR,
            vibrational_frequency=500,  # cm⁻¹
            reduced_mass=16.0,
            nir_absorption=NIRAbsorption(
                primary_wavelength=2500,  # nm (weak)
                overtone_wavelengths=[],
                molar_absorptivity=0.1,
                bandwidth=70
            ),
            nir_active=False,  # Very weak NIR absorption
            functional_groups=[FunctionalGroup.DISULFIDE],
            common_molecules=["proteins", "cystine", "keratin"],
            hybridization=("sp3", "sp3"),
            rotation_barrier=40.0  # High barrier
        ))
        
        self._add_bond(CovalentBond(
            name="C-S (thioether)",
            atoms=("C", "S"),
            bond_type=BondType.SINGLE,
            bond_energy=272,  # kJ/mol
            bond_length=181,  # pm
            bond_length_range=(178, 184),
            electronegativity_diff=0.0,
            bond_polarity=BondPolarity.NONPOLAR,
            vibrational_frequency=700,  # cm⁻¹
            reduced_mass=7.5,
            nir_absorption=NIRAbsorption(
                primary_wavelength=2400,  # nm (weak)
                overtone_wavelengths=[],
                molar_absorptivity=0.2,
                bandwidth=60
            ),
            nir_active=False,
            functional_groups=[FunctionalGroup.SULFIDE],
            common_molecules=["methionine", "dimethyl_sulfide"],
            hybridization=("sp3", "sp3")
        ))
        
        # =====================================================================
        # CARBON-NITROGEN BONDS
        # =====================================================================
        
        self._add_bond(CovalentBond(
            name="C-N (amine)",
            atoms=("C", "N"),
            bond_type=BondType.SINGLE,
            bond_energy=305,  # kJ/mol
            bond_length=147,  # pm
            bond_length_range=(145, 150),
            electronegativity_diff=0.5,
            bond_polarity=BondPolarity.POLAR,
            vibrational_frequency=1080,  # cm⁻¹
            reduced_mass=5.57,
            nir_absorption=NIRAbsorption(
                primary_wavelength=2200,  # nm (weak)
                overtone_wavelengths=[],
                molar_absorptivity=0.2,
                bandwidth=65
            ),
            nir_active=False,
            functional_groups=[FunctionalGroup.AMINE],
            common_molecules=["amino_acids", "amines", "proteins"],
            hybridization=("sp3", "sp3")
        ))
        
        self._add_bond(CovalentBond(
            name="C-N (amide)",
            atoms=("C", "N"),
            bond_type=BondType.SINGLE,
            bond_energy=305,  # kJ/mol
            bond_length=133,  # pm (partial double bond character)
            bond_length_range=(131, 135),
            electronegativity_diff=0.5,
            bond_polarity=BondPolarity.POLAR,
            vibrational_frequency=1400,  # cm⁻¹
            reduced_mass=5.57,
            nir_absorption=NIRAbsorption(
                primary_wavelength=2150,  # nm
                overtone_wavelengths=[],
                molar_absorptivity=0.3,
                bandwidth=60
            ),
            nir_active=False,
            functional_groups=[FunctionalGroup.AMIDE],
            common_molecules=["proteins", "peptides", "nylon"],
            bond_order=1.5,  # Resonance
            hybridization=("sp2", "sp2"),
            rotation_barrier=75.0  # High barrier (restricted rotation)
        ))
        
        self._add_bond(CovalentBond(
            name="C=N (imine)",
            atoms=("C", "N"),
            bond_type=BondType.DOUBLE,
            bond_energy=615,  # kJ/mol
            bond_length=127,  # pm
            bond_length_range=(125, 129),
            electronegativity_diff=0.5,
            bond_polarity=BondPolarity.POLAR,
            vibrational_frequency=1640,  # cm⁻¹
            reduced_mass=5.57,
            nir_absorption=NIRAbsorption(
                primary_wavelength=1620,  # nm
                overtone_wavelengths=[],
                molar_absorptivity=0.4,
                bandwidth=55
            ),
            common_molecules=["imines", "schiff_bases"],
            bond_order=2.0,
            hybridization=("sp2", "sp2")
        ))
        
        self._add_bond(CovalentBond(
            name="C≡N (nitrile)",
            atoms=("C", "N"),
            bond_type=BondType.TRIPLE,
            bond_energy=891,  # kJ/mol
            bond_length=115,  # pm
            bond_length_range=(113, 117),
            electronegativity_diff=0.5,
            bond_polarity=BondPolarity.POLAR,
            dipole_moment=3.9,
            vibrational_frequency=2220,  # cm⁻¹
            reduced_mass=5.57,
            nir_absorption=NIRAbsorption(
                primary_wavelength=2220,  # nm
                overtone_wavelengths=[],
                molar_absorptivity=0.5,
                bandwidth=50
            ),
            functional_groups=[FunctionalGroup.NITRILE],
            common_molecules=["acetonitrile", "nitriles"],
            bond_order=3.0,
            hybridization=("sp", "sp")
        ))
        
        # =====================================================================
        # NITROGEN-OXYGEN BONDS
        # =====================================================================
        
        self._add_bond(CovalentBond(
            name="N-O (nitro)",
            atoms=("N", "O"),
            bond_type=BondType.DOUBLE,
            bond_energy=607,  # kJ/mol
            bond_length=122,  # pm
            electronegativity_diff=0.5,
            bond_polarity=BondPolarity.POLAR,
            vibrational_frequency=1550,  # cm⁻¹
            reduced_mass=5.60,
            nir_absorption=NIRAbsorption(
                primary_wavelength=1580,  # nm
                overtone_wavelengths=[],
                molar_absorptivity=0.6,
                bandwidth=60
            ),
            functional_groups=[FunctionalGroup.NITRO],
            common_molecules=["nitro_compounds", "explosives"],
            bond_order=1.5,  # Resonance
            hybridization=("sp2", "sp2")
        ))
        
        # =====================================================================
        # PHOSPHORUS BONDS (ATP, DNA, phospholipids)
        # =====================================================================
        
        self._add_bond(CovalentBond(
            name="P-O (phosphate ester)",
            atoms=("P", "O"),
            bond_type=BondType.SINGLE,
            bond_energy=335,  # kJ/mol
            bond_length=163,  # pm
            bond_length_range=(160, 166),
            electronegativity_diff=1.2,
            bond_polarity=BondPolarity.POLAR,
            vibrational_frequency=1080,  # cm⁻¹
            reduced_mass=6.45,
            nir_absorption=NIRAbsorption(
                primary_wavelength=2250,  # nm
                overtone_wavelengths=[],
                molar_absorptivity=0.4,
                bandwidth=70
            ),
            functional_groups=[FunctionalGroup.PHOSPHATE],
            common_molecules=["ATP", "DNA", "RNA", "phospholipids"],
            hybridization=("sp3", "sp3")
        ))
        
        self._add_bond(CovalentBond(
            name="P=O (phosphate)",
            atoms=("P", "O"),
            bond_type=BondType.DOUBLE,
            bond_energy=544,  # kJ/mol
            bond_length=150,  # pm
            electronegativity_diff=1.2,
            bond_polarity=BondPolarity.POLAR,
            dipole_moment=2.7,
            vibrational_frequency=1250,  # cm⁻¹
            reduced_mass=6.45,
            nir_absorption=NIRAbsorption(
                primary_wavelength=2100,  # nm
                overtone_wavelengths=[],
                molar_absorptivity=0.5,
                bandwidth=65
            ),
            functional_groups=[FunctionalGroup.PHOSPHATE],
            common_molecules=["ATP", "DNA", "RNA", "phospholipids"],
            bond_order=2.0,
            hybridization=("sp3", "sp2")
        ))
        
        # =====================================================================
        # HYDROGEN BONDS (Non-covalent but crucial for NIR)
        # =====================================================================
        
        self._add_bond(CovalentBond(
            name="H-bond (O-H···O)",
            atoms=("O", "O"),
            bond_type=BondType.HYDROGEN,
            bond_energy=21,  # kJ/mol (weak)
            bond_length=280,  # pm (H···O distance)
            bond_length_range=(250, 310),
            electronegativity_diff=0.0,
            bond_polarity=BondPolarity.POLAR,
            vibrational_frequency=200,  # cm⁻¹ (very low)
            nir_absorption=NIRAbsorption(
                primary_wavelength=1940,  # nm (shifts O-H absorption)
                overtone_wavelengths=[],
                molar_absorptivity=0.8,
                bandwidth=200,  # Very broad
                temperature_shift=0.5
            ),
            common_molecules=["water", "alcohols", "carboxylic_acids"],
            hybridization=("sp3", "sp3")
        ))
        
        self._add_bond(CovalentBond(
            name="H-bond (N-H···O)",
            atoms=("N", "O"),
            bond_type=BondType.HYDROGEN,
            bond_energy=29,  # kJ/mol
            bond_length=290,  # pm
            bond_length_range=(260, 320),
            electronegativity_diff=0.5,
            bond_polarity=BondPolarity.POLAR,
            vibrational_frequency=180,  # cm⁻¹
            nir_absorption=NIRAbsorption(
                primary_wavelength=2060,  # nm
                overtone_wavelengths=[],
                molar_absorptivity=0.7,
                bandwidth=180
            ),
            functional_groups=[FunctionalGroup.AMIDE],
            common_molecules=["proteins", "peptides", "DNA"],
            hybridization=("sp2", "sp2")
        ))
        
        logger.info("Loaded complete bond database")
    
    def _add_bond(self, bond: CovalentBond):
        """Add bond to database."""
        self.bonds[bond.name] = bond
    
    def _build_indices(self):
        """Build search indices."""
        # Functional group index
        for bond_name, bond in self.bonds.items():
            for fg in bond.functional_groups:
                if fg not in self.functional_group_index:
                    self.functional_group_index[fg] = []
                self.functional_group_index[fg].append(bond_name)
        
        # NIR wavelength index (round to nearest 10nm for lookup)
        for bond_name, bond in self.bonds.items():
            if bond.is_nir_active():
                for wavelength in bond.get_nir_wavelengths():
                    key = round(wavelength / 10) * 10  # Round to nearest 10nm
                    if key not in self.nir_wavelength_index:
                        self.nir_wavelength_index[key] = []
                    self.nir_wavelength_index[key].append(bond_name)
    
    # Query Methods
    
    def get_bond(self, name: str) -> Optional[CovalentBond]:
        """Get bond by name."""
        return self.bonds.get(name)
    
    def get_bonds_by_atoms(self, atom1: str, atom2: str) -> List[CovalentBond]:
        """Get all bonds between two atom types."""
        results = []
        for bond in self.bonds.values():
            if (atom1, atom2) == bond.atoms or (atom2, atom1) == bond.atoms:
                results.append(bond)
        return results
    
    def get_bonds_by_functional_group(self, fg: FunctionalGroup) -> List[CovalentBond]:
        """Get bonds associated with functional group."""
        bond_names = self.functional_group_index.get(fg, [])
        return [self.bonds[name] for name in bond_names]
    
    def search_by_nir_wavelength(self, wavelength: float, tolerance: float = 50) -> List[Tuple[CovalentBond, float]]:
        """
        Search for bonds with NIR absorption near given wavelength.
        
        Args:
            wavelength: Target wavelength in nm
            tolerance: Search tolerance in nm
        
        Returns:
            List of (bond, distance) tuples sorted by distance
        """
        matches = []
        for bond in self.bonds.values():
            if not bond.is_nir_active():
                continue
            
            for bond_wavelength in bond.get_nir_wavelengths():
                distance = abs(bond_wavelength - wavelength)
                if distance <= tolerance:
                    matches.append((bond, distance))
        
        # Sort by distance
        matches.sort(key=lambda x: x[1])
        return matches
    
    def get_strongest_nir_absorbers(self, n: int = 10) -> List[CovalentBond]:
        """Get bonds with strongest NIR absorption."""
        nir_bonds = [b for b in self.bonds.values() if b.is_nir_active()]
        nir_bonds.sort(key=lambda b: b.nir_absorption.molar_absorptivity, reverse=True)
        return nir_bonds[:n]
    
    def get_bond_statistics(self) -> Dict:
        """Get database statistics."""
        nir_active = sum(1 for b in self.bonds.values() if b.is_nir_active())
        
        bond_types = {}
        for bond in self.bonds.values():
            bt = bond.bond_type.value
            bond_types[bt] = bond_types.get(bt, 0) + 1
        
        return {
            "total_bonds": len(self.bonds),
            "nir_active": nir_active,
            "nir_inactive": len(self.bonds) - nir_active,
            "bond_types": bond_types,
            "functional_groups": len(self.functional_group_index)
        }


# =============================================================================
# TEST SUITE
# =============================================================================

def test_bond_database():
    """Test 1: Bond database initialization."""
    print("\n" + "="*80)
    print("TEST 1: Bond Database Initialization")
    print("="*80)
    
    db = CovalentBondLibrary()
    stats = db.get_bond_statistics()
    
    print(f"\n✓ Loaded {stats['total_bonds']} bonds")
    print(f"✓ NIR active: {stats['nir_active']}")
    print(f"✓ NIR inactive: {stats['nir_inactive']}")
    print(f"\n✓ Bond types:")
    for bt, count in stats['bond_types'].items():
        print(f"  {bt:15s}: {count} bonds")
    
    return True


def test_bond_queries():
    """Test 2: Bond queries."""
    print("\n" + "="*80)
    print("TEST 2: Bond Queries")
    print("="*80)
    
    db = CovalentBondLibrary()
    
    # Query specific bond
    bond = db.get_bond("C-H (alkane)")
    print(f"\n✓ Query 'C-H (alkane)':")
    print(f"  Energy: {bond.bond_energy} kJ/mol")
    print(f"  Length: {bond.bond_length} pm")
    print(f"  NIR wavelength: {bond.nir_absorption.primary_wavelength} nm")
    
    # Query by atoms
    co_bonds = db.get_bonds_by_atoms("C", "O")
    print(f"\n✓ C-O bonds: {len(co_bonds)}")
    for bond in co_bonds[:3]:
        print(f"  {bond.name:25s}: {bond.bond_energy} kJ/mol")
    
    # Strongest NIR absorbers
    strong = db.get_strongest_nir_absorbers(5)
    print(f"\n✓ Strongest NIR absorbers:")
    for bond in strong:
        print(f"  {bond.name:25s}: ε = {bond.nir_absorption.molar_absorptivity}")
    
    return True


def test_nir_wavelength_search():
    """Test 3: NIR wavelength search."""
    print("\n" + "="*80)
    print("TEST 3: NIR Wavelength Search")
    print("="*80)
    
    db = CovalentBondLibrary()
    
    # Search for bonds at common NIR wavelengths
    test_wavelengths = [1690, 1730, 2050, 2310]
    
    for wavelength in test_wavelengths:
        matches = db.search_by_nir_wavelength(wavelength, tolerance=50)
        print(f"\n✓ Bonds near {wavelength} nm: {len(matches)}")
        for bond, distance in matches[:3]:
            print(f"  {bond.name:25s}: {wavelength - distance:.0f} nm (Δ={distance:.0f})")
    
    return True


def test_functional_groups():
    """Test 4: Functional group associations."""
    print("\n" + "="*80)
    print("TEST 4: Functional Group Associations")
    print("="*80)
    
    db = CovalentBondLibrary()
    
    # Test key functional groups
    test_groups = [
        FunctionalGroup.HYDROXYL,
        FunctionalGroup.CARBONYL,
        FunctionalGroup.CARBOXYL,
        FunctionalGroup.AMIDE
    ]
    
    for fg in test_groups:
        bonds = db.get_bonds_by_functional_group(fg)
        print(f"\n✓ {fg.value.upper()} group: {len(bonds)} bonds")
        for bond in bonds:
            print(f"  {bond.name}")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("COVALENT BOND LIBRARY - TEST SUITE")
    print("Phase 3A Part 3: Bond Energies & NIR Mapping")
    print("="*80)
    
    tests = [
        ("Bond Database", test_bond_database),
        ("Bond Queries", test_bond_queries),
        ("NIR Wavelength Search", test_nir_wavelength_search),
        ("Functional Groups", test_functional_groups),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append(("✅ PASS", test_name))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(("❌ FAIL", test_name))
            return False
    
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
        print("\n🎉 All tests passed! Bond library functional.")
        print(f"\nCurrent: {len(CovalentBondLibrary().bonds)} bonds")
        print("Next: Add N-H, O-H, S-H, and other bonds")
        return True
    
    return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
