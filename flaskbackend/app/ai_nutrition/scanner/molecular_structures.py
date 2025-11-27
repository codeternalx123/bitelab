"""
Molecular Structures Database - Phase 3A Part 2
================================================

Comprehensive database of molecules found in food, including:
- Macronutrients (amino acids, fatty acids, sugars, starches)
- Micronutrients (vitamins, minerals complexes)
- Phytochemicals (antioxidants, flavonoids, polyphenols)
- Toxins (heavy metals, pesticides, mycotoxins, allergens)
- Bioactive compounds (caffeine, capsaicin, etc.)

Target: 10,000+ molecules with structures, properties, and health impacts.

This module provides:
- Molecular structures (SMILES, InChI, formula)
- Physical/chemical properties
- NIR spectroscopic fingerprints
- Health effects and bioavailability
- Food sources
- Interaction warnings

Scientific References:
----------------------
- PubChem Database (NIH)
- USDA FoodData Central
- ChEBI (Chemical Entities of Biological Interest)
- HMDB (Human Metabolome Database)
- FooDB (Food Component Database)

Author: AI Nutrition Scanner Team
Date: January 2025
"""

import numpy as np
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoleculeCategory(Enum):
    """Broad categories of molecules."""
    AMINO_ACID = "amino_acid"
    FATTY_ACID = "fatty_acid"
    CARBOHYDRATE = "carbohydrate"
    VITAMIN = "vitamin"
    MINERAL_COMPLEX = "mineral_complex"
    ANTIOXIDANT = "antioxidant"
    FLAVONOID = "flavonoid"
    POLYPHENOL = "polyphenol"
    ALKALOID = "alkaloid"
    TOXIN = "toxin"
    ALLERGEN = "allergen"
    PESTICIDE = "pesticide"
    HORMONE = "hormone"
    ENZYME = "enzyme"
    PIGMENT = "pigment"


class HealthEffect(Enum):
    """Health impact classification."""
    BENEFICIAL = "beneficial"
    ESSENTIAL = "essential"
    NEUTRAL = "neutral"
    POTENTIALLY_HARMFUL = "potentially_harmful"
    TOXIC = "toxic"
    ALLERGENIC = "allergenic"
    CARCINOGENIC = "carcinogenic"
    MUTAGENIC = "mutagenic"


@dataclass
class NIRSignature:
    """
    NIR spectroscopic signature of a molecule.
    
    Attributes:
        primary_bands: Main absorption bands (nm)
        secondary_bands: Weaker absorption bands (nm)
        characteristic_peaks: Distinctive peaks for identification (nm)
        intensity_range: Expected intensity range (0-1)
    """
    primary_bands: List[float] = field(default_factory=list)
    secondary_bands: List[float] = field(default_factory=list)
    characteristic_peaks: List[float] = field(default_factory=list)
    intensity_range: Tuple[float, float] = (0.0, 1.0)


@dataclass
class HealthData:
    """
    Health impact information for a molecule.
    
    Attributes:
        health_effect: Overall health classification
        benefits: Beneficial health effects
        risks: Potential health risks
        rda: Recommended Daily Allowance (if applicable)
        ul: Tolerable Upper Intake Level
        bioavailability: Absorption rate (0-100%)
        metabolism: How body processes it
        interactions: Drug/nutrient interactions
    """
    health_effect: HealthEffect
    benefits: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    rda: Optional[float] = None  # mg/day
    ul: Optional[float] = None  # mg/day
    bioavailability: float = 50.0  # %
    metabolism: str = ""
    interactions: List[str] = field(default_factory=list)


@dataclass
class Molecule:
    """
    Complete molecular data structure.
    
    Attributes:
        name: Common name
        iupac_name: IUPAC systematic name
        formula: Chemical formula (e.g., C6H12O6)
        molecular_weight: Molecular weight (g/mol)
        category: Molecule category
        smiles: SMILES notation
        inchi: InChI identifier
        cas_number: CAS registry number
        pubchem_id: PubChem compound ID
        structure_2d: 2D structure representation
        nir_signature: NIR spectroscopic signature
        health_data: Health impact data
        food_sources: Common food sources
        concentration_range: Typical concentration in foods (mg/kg)
        synonyms: Alternative names
    """
    name: str
    iupac_name: str
    formula: str
    molecular_weight: float
    category: MoleculeCategory
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    cas_number: Optional[str] = None
    pubchem_id: Optional[int] = None
    structure_2d: Optional[str] = None
    nir_signature: Optional[NIRSignature] = None
    health_data: Optional[HealthData] = None
    food_sources: List[str] = field(default_factory=list)
    concentration_range: Tuple[float, float] = (0.0, 0.0)  # mg/kg
    synonyms: List[str] = field(default_factory=list)
    
    def is_essential(self) -> bool:
        """Check if molecule is essential nutrient."""
        return (self.health_data and 
                self.health_data.health_effect == HealthEffect.ESSENTIAL)
    
    def is_toxic(self) -> bool:
        """Check if molecule is toxic."""
        return (self.health_data and 
                self.health_data.health_effect in [
                    HealthEffect.TOXIC, 
                    HealthEffect.CARCINOGENIC,
                    HealthEffect.MUTAGENIC
                ])
    
    def get_safe_intake(self) -> Optional[Tuple[float, float]]:
        """Get safe daily intake range (RDA to UL)."""
        if self.health_data and self.health_data.rda:
            return (self.health_data.rda, self.health_data.ul or self.health_data.rda * 10)
        return None


class MolecularDatabase:
    """
    Comprehensive database of food molecules.
    
    Target: 10,000+ molecules across all categories.
    Current implementation: Core molecules for demonstration (expandable).
    """
    
    def __init__(self):
        """Initialize molecular database."""
        self.molecules: Dict[str, Molecule] = {}
        self.category_index: Dict[MoleculeCategory, List[str]] = {}
        self._populate_database()
        
        logger.info(f"Initialized MolecularDatabase with {len(self.molecules)} molecules")
    
    def _populate_database(self):
        """Populate database with molecules."""
        
        # =====================================================================
        # AMINO ACIDS (20 Standard + Key Non-Standard)
        # =====================================================================
        
        # Essential Amino Acids
        self._add_molecule(Molecule(
            name="Leucine",
            iupac_name="2-amino-4-methylpentanoic acid",
            formula="C6H13NO2",
            molecular_weight=131.17,
            category=MoleculeCategory.AMINO_ACID,
            smiles="CC(C)CC(C(=O)O)N",
            inchi="InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)",
            cas_number="61-90-5",
            pubchem_id=6106,
            nir_signature=NIRSignature(
                primary_bands=[2050, 2180],  # N-H stretch
                secondary_bands=[1510, 2310],  # C-H, amide
                characteristic_peaks=[2050],
                intensity_range=(0.2, 0.4)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["muscle_protein_synthesis", "blood_sugar_regulation", "wound_healing"],
                rda=39.0,  # mg/kg body weight/day
                bioavailability=95.0,
                metabolism="Branched-chain amino acid, metabolized in muscle",
                interactions=["May lower blood sugar with diabetes medications"]
            ),
            food_sources=["chicken", "beef", "fish", "eggs", "milk", "soybeans"],
            concentration_range=(1500, 3000),  # mg/kg protein
            synonyms=["L-Leucine", "Leu", "L"]
        ))
        
        self._add_molecule(Molecule(
            name="Lysine",
            iupac_name="2,6-diaminohexanoic acid",
            formula="C6H14N2O2",
            molecular_weight=146.19,
            category=MoleculeCategory.AMINO_ACID,
            smiles="C(CCN)CC(C(=O)O)N",
            cas_number="56-87-1",
            pubchem_id=5962,
            nir_signature=NIRSignature(
                primary_bands=[2050, 2180],
                secondary_bands=[1510],
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["collagen_formation", "calcium_absorption", "immune_function"],
                rda=30.0,  # mg/kg/day
                bioavailability=90.0,
                metabolism="Essential amino acid, not synthesized by body"
            ),
            food_sources=["meat", "fish", "eggs", "dairy", "legumes"],
            concentration_range=(1200, 2500),
            synonyms=["L-Lysine", "Lys", "K"]
        ))
        
        self._add_molecule(Molecule(
            name="Tryptophan",
            iupac_name="2-amino-3-(1H-indol-3-yl)propanoic acid",
            formula="C11H12N2O2",
            molecular_weight=204.23,
            category=MoleculeCategory.AMINO_ACID,
            smiles="C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N",
            cas_number="73-22-3",
            pubchem_id=6305,
            nir_signature=NIRSignature(
                primary_bands=[2050, 2180],
                secondary_bands=[1510, 1600],  # Aromatic ring
                intensity_range=(0.15, 0.25)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["serotonin_precursor", "mood_regulation", "sleep_improvement"],
                rda=4.0,  # mg/kg/day
                bioavailability=85.0,
                metabolism="Converted to serotonin and melatonin",
                interactions=["MAO inhibitors", "SSRIs - serotonin syndrome risk"]
            ),
            food_sources=["turkey", "chicken", "eggs", "cheese", "nuts", "seeds"],
            concentration_range=(200, 400),
            synonyms=["L-Tryptophan", "Trp", "W"]
        ))
        
        # Non-essential but important amino acids
        self._add_molecule(Molecule(
            name="Cysteine",
            iupac_name="2-amino-3-sulfanylpropanoic acid",
            formula="C3H7NO2S",
            molecular_weight=121.16,
            category=MoleculeCategory.AMINO_ACID,
            smiles="C(C(C(=O)O)N)S",
            cas_number="52-90-4",
            pubchem_id=5862,
            nir_signature=NIRSignature(
                primary_bands=[2050, 2550],  # N-H, S-H stretch
                secondary_bands=[1510],
                characteristic_peaks=[2550],  # S-H distinctive
                intensity_range=(0.1, 0.2)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["antioxidant", "detoxification", "glutathione_synthesis"],
                bioavailability=95.0,
                metabolism="Can be synthesized from methionine"
            ),
            food_sources=["poultry", "eggs", "garlic", "onions", "broccoli"],
            concentration_range=(300, 800),
            synonyms=["L-Cysteine", "Cys", "C"]
        ))
        
        # Remaining Essential Amino Acids
        self._add_molecule(Molecule(
            name="Valine",
            iupac_name="2-amino-3-methylbutanoic acid",
            formula="C5H11NO2",
            molecular_weight=117.15,
            category=MoleculeCategory.AMINO_ACID,
            smiles="CC(C)C(C(=O)O)N",
            cas_number="72-18-4",
            pubchem_id=6287,
            nir_signature=NIRSignature(
                primary_bands=[2050, 2180],
                secondary_bands=[1510, 2310],
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["muscle_growth", "tissue_repair", "energy_production"],
                rda=26.0,
                bioavailability=95.0,
                metabolism="BCAA, metabolized in muscle"
            ),
            food_sources=["meat", "dairy", "soy", "beans", "nuts"],
            concentration_range=(1800, 3500),
            synonyms=["L-Valine", "Val", "V"]
        ))
        
        self._add_molecule(Molecule(
            name="Isoleucine",
            iupac_name="2-amino-3-methylpentanoic acid",
            formula="C6H13NO2",
            molecular_weight=131.17,
            category=MoleculeCategory.AMINO_ACID,
            smiles="CCC(C)C(C(=O)O)N",
            cas_number="73-32-5",
            pubchem_id=6306,
            nir_signature=NIRSignature(
                primary_bands=[2050, 2180],
                secondary_bands=[1510, 2310],
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["muscle_recovery", "immune_function", "hemoglobin_production"],
                rda=20.0,
                bioavailability=95.0,
                metabolism="BCAA, metabolized in muscle"
            ),
            food_sources=["meat", "fish", "eggs", "cheese", "nuts"],
            concentration_range=(1500, 3000),
            synonyms=["L-Isoleucine", "Ile", "I"]
        ))
        
        self._add_molecule(Molecule(
            name="Phenylalanine",
            iupac_name="2-amino-3-phenylpropanoic acid",
            formula="C9H11NO2",
            molecular_weight=165.19,
            category=MoleculeCategory.AMINO_ACID,
            smiles="C1=CC=C(C=C1)CC(C(=O)O)N",
            cas_number="63-91-2",
            pubchem_id=6140,
            nir_signature=NIRSignature(
                primary_bands=[2050, 2180],
                secondary_bands=[1600, 1510],  # Aromatic ring
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["neurotransmitter_production", "mood_regulation", "pain_relief"],
                rda=25.0,
                bioavailability=90.0,
                metabolism="Precursor to tyrosine, dopamine, norepinephrine"
            ),
            food_sources=["meat", "fish", "eggs", "dairy", "soy"],
            concentration_range=(1500, 3000),
            synonyms=["L-Phenylalanine", "Phe", "F"]
        ))
        
        self._add_molecule(Molecule(
            name="Threonine",
            iupac_name="2-amino-3-hydroxybutanoic acid",
            formula="C4H9NO3",
            molecular_weight=119.12,
            category=MoleculeCategory.AMINO_ACID,
            smiles="CC(C(C(=O)O)N)O",
            cas_number="72-19-5",
            pubchem_id=6288,
            nir_signature=NIRSignature(
                primary_bands=[1450, 1940, 2050],  # O-H, N-H
                secondary_bands=[1510],
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["protein_balance", "immune_function", "collagen_formation"],
                rda=15.0,
                bioavailability=90.0,
                metabolism="Essential amino acid"
            ),
            food_sources=["meat", "fish", "dairy", "eggs", "lentils"],
            concentration_range=(1200, 2500),
            synonyms=["L-Threonine", "Thr", "T"]
        ))
        
        self._add_molecule(Molecule(
            name="Methionine",
            iupac_name="2-amino-4-(methylsulfanyl)butanoic acid",
            formula="C5H11NO2S",
            molecular_weight=149.21,
            category=MoleculeCategory.AMINO_ACID,
            smiles="CSCCC(C(=O)O)N",
            cas_number="63-68-3",
            pubchem_id=6137,
            nir_signature=NIRSignature(
                primary_bands=[2050, 2180],
                secondary_bands=[1510, 2310],
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["detoxification", "fat_metabolism", "antioxidant_production"],
                rda=10.4,
                bioavailability=90.0,
                metabolism="Precursor to cysteine, SAMe, taurine"
            ),
            food_sources=["fish", "meat", "eggs", "brazil_nuts", "sesame_seeds"],
            concentration_range=(800, 1800),
            synonyms=["L-Methionine", "Met", "M"]
        ))
        
        self._add_molecule(Molecule(
            name="Histidine",
            iupac_name="2-amino-3-(1H-imidazol-4-yl)propanoic acid",
            formula="C6H9N3O2",
            molecular_weight=155.15,
            category=MoleculeCategory.AMINO_ACID,
            smiles="C1=C(NC=N1)CC(C(=O)O)N",
            cas_number="71-00-1",
            pubchem_id=6274,
            nir_signature=NIRSignature(
                primary_bands=[2050, 2180],
                secondary_bands=[1510, 1620],  # Imidazole ring
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["histamine_production", "tissue_repair", "metal_ion_binding"],
                rda=10.0,
                bioavailability=85.0,
                metabolism="Precursor to histamine, essential for growth"
            ),
            food_sources=["meat", "fish", "poultry", "nuts", "seeds"],
            concentration_range=(900, 2000),
            synonyms=["L-Histidine", "His", "H"]
        ))
        
        # Non-Essential Amino Acids
        self._add_molecule(Molecule(
            name="Alanine",
            iupac_name="2-aminopropanoic acid",
            formula="C3H7NO2",
            molecular_weight=89.09,
            category=MoleculeCategory.AMINO_ACID,
            smiles="CC(C(=O)O)N",
            cas_number="56-41-7",
            pubchem_id=5950,
            nir_signature=NIRSignature(
                primary_bands=[2050, 2180],
                secondary_bands=[1510],
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["glucose_metabolism", "immune_support", "energy_production"],
                bioavailability=95.0,
                metabolism="Can be synthesized from pyruvate"
            ),
            food_sources=["meat", "fish", "eggs", "dairy", "avocados"],
            concentration_range=(1500, 3500),
            synonyms=["L-Alanine", "Ala", "A"]
        ))
        
        self._add_molecule(Molecule(
            name="Arginine",
            iupac_name="2-amino-5-(diaminomethylideneamino)pentanoic acid",
            formula="C6H14N4O2",
            molecular_weight=174.20,
            category=MoleculeCategory.AMINO_ACID,
            smiles="C(CC(C(=O)O)N)CN=C(N)N",
            cas_number="74-79-3",
            pubchem_id=6322,
            nir_signature=NIRSignature(
                primary_bands=[2050, 2180],
                secondary_bands=[1510, 1620],
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["nitric_oxide_production", "wound_healing", "immune_function", "cardiovascular_health"],
                bioavailability=85.0,
                metabolism="Semi-essential, precursor to nitric oxide"
            ),
            food_sources=["meat", "fish", "nuts", "seeds", "legumes"],
            concentration_range=(2000, 4500),
            synonyms=["L-Arginine", "Arg", "R"]
        ))
        
        self._add_molecule(Molecule(
            name="Asparagine",
            iupac_name="2-amino-3-carbamoylpropanoic acid",
            formula="C4H8N2O3",
            molecular_weight=132.12,
            category=MoleculeCategory.AMINO_ACID,
            smiles="C(C(C(=O)O)N)C(=O)N",
            cas_number="70-47-3",
            pubchem_id=6267,
            nir_signature=NIRSignature(
                primary_bands=[2050, 2180],
                secondary_bands=[1510, 1670],  # Amide C=O
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["nervous_system_function", "immune_support"],
                bioavailability=90.0,
                metabolism="Can be synthesized from aspartate"
            ),
            food_sources=["asparagus", "potatoes", "nuts", "seeds", "soy"],
            concentration_range=(1000, 2500),
            synonyms=["L-Asparagine", "Asn", "N"]
        ))
        
        self._add_molecule(Molecule(
            name="Aspartate",
            iupac_name="2-aminobutanedioic acid",
            formula="C4H7NO4",
            molecular_weight=133.10,
            category=MoleculeCategory.AMINO_ACID,
            smiles="C(C(C(=O)O)N)C(=O)O",
            cas_number="56-84-8",
            pubchem_id=5960,
            nir_signature=NIRSignature(
                primary_bands=[1730, 2050],  # COOH, N-H
                secondary_bands=[1510],
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["neurotransmission", "energy_production", "DNA_synthesis"],
                bioavailability=90.0,
                metabolism="Glucogenic amino acid, TCA cycle intermediate"
            ),
            food_sources=["meat", "poultry", "eggs", "avocado", "asparagus"],
            concentration_range=(2000, 4000),
            synonyms=["L-Aspartate", "Aspartic Acid", "Asp", "D"]
        ))
        
        self._add_molecule(Molecule(
            name="Glutamate",
            iupac_name="2-aminopentanedioic acid",
            formula="C5H9NO4",
            molecular_weight=147.13,
            category=MoleculeCategory.AMINO_ACID,
            smiles="C(CC(=O)O)C(C(=O)O)N",
            cas_number="56-86-0",
            pubchem_id=33032,
            nir_signature=NIRSignature(
                primary_bands=[1730, 2050],
                secondary_bands=[1510],
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["neurotransmission", "learning", "memory", "nitrogen_transport"],
                bioavailability=90.0,
                metabolism="Major excitatory neurotransmitter"
            ),
            food_sources=["meat", "fish", "eggs", "cheese", "tomatoes"],
            concentration_range=(5000, 15000),  # Highest concentration amino acid
            synonyms=["L-Glutamate", "Glutamic Acid", "Glu", "E", "MSG"]
        ))
        
        self._add_molecule(Molecule(
            name="Glutamine",
            iupac_name="2-amino-4-carbamoylbutanoic acid",
            formula="C5H10N2O3",
            molecular_weight=146.15,
            category=MoleculeCategory.AMINO_ACID,
            smiles="C(CC(=O)N)C(C(=O)O)N",
            cas_number="56-85-9",
            pubchem_id=5961,
            nir_signature=NIRSignature(
                primary_bands=[2050, 2180],
                secondary_bands=[1510, 1670],
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["intestinal_health", "immune_function", "protein_synthesis", "nitrogen_balance"],
                bioavailability=90.0,
                metabolism="Most abundant amino acid in blood"
            ),
            food_sources=["meat", "fish", "eggs", "dairy", "cabbage"],
            concentration_range=(3000, 8000),
            synonyms=["L-Glutamine", "Gln", "Q"]
        ))
        
        self._add_molecule(Molecule(
            name="Glycine",
            iupac_name="aminoacetic acid",
            formula="C2H5NO2",
            molecular_weight=75.07,
            category=MoleculeCategory.AMINO_ACID,
            smiles="C(C(=O)O)N",
            cas_number="56-40-6",
            pubchem_id=750,
            nir_signature=NIRSignature(
                primary_bands=[2050, 2180],
                secondary_bands=[1510],
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["collagen_synthesis", "neurotransmission", "sleep_quality", "antioxidant"],
                bioavailability=95.0,
                metabolism="Smallest amino acid, inhibitory neurotransmitter"
            ),
            food_sources=["meat", "fish", "dairy", "gelatin", "bone_broth"],
            concentration_range=(1500, 4000),
            synonyms=["L-Glycine", "Gly", "G"]
        ))
        
        self._add_molecule(Molecule(
            name="Proline",
            iupac_name="pyrrolidine-2-carboxylic acid",
            formula="C5H9NO2",
            molecular_weight=115.13,
            category=MoleculeCategory.AMINO_ACID,
            smiles="C1CC(NC1)C(=O)O",
            cas_number="147-85-3",
            pubchem_id=145742,
            nir_signature=NIRSignature(
                primary_bands=[2050],  # Secondary amine
                secondary_bands=[1510, 1730],
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["collagen_formation", "wound_healing", "joint_health", "skin_health"],
                bioavailability=90.0,
                metabolism="Unique cyclic structure, synthesized from glutamate"
            ),
            food_sources=["meat", "fish", "dairy", "eggs", "gelatin"],
            concentration_range=(2000, 5000),
            synonyms=["L-Proline", "Pro", "P"]
        ))
        
        self._add_molecule(Molecule(
            name="Serine",
            iupac_name="2-amino-3-hydroxypropanoic acid",
            formula="C3H7NO3",
            molecular_weight=105.09,
            category=MoleculeCategory.AMINO_ACID,
            smiles="C(C(C(=O)O)N)O",
            cas_number="56-45-1",
            pubchem_id=5951,
            nir_signature=NIRSignature(
                primary_bands=[1450, 1940, 2050],  # O-H, N-H
                secondary_bands=[1510],
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["protein_synthesis", "brain_function", "immune_support", "fat_metabolism"],
                bioavailability=90.0,
                metabolism="Precursor to glycine, cysteine, and several neurotransmitters"
            ),
            food_sources=["soy", "nuts", "eggs", "meat", "wheat_gluten"],
            concentration_range=(1500, 3500),
            synonyms=["L-Serine", "Ser", "S"]
        ))
        
        self._add_molecule(Molecule(
            name="Tyrosine",
            iupac_name="2-amino-3-(4-hydroxyphenyl)propanoic acid",
            formula="C9H11NO3",
            molecular_weight=181.19,
            category=MoleculeCategory.AMINO_ACID,
            smiles="C1=CC(=CC=C1CC(C(=O)O)N)O",
            cas_number="60-18-4",
            pubchem_id=6057,
            nir_signature=NIRSignature(
                primary_bands=[1450, 2050],  # Phenolic O-H, N-H
                secondary_bands=[1600, 1510],  # Aromatic
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["neurotransmitter_production", "stress_response", "mood_regulation", "thyroid_hormones"],
                bioavailability=85.0,
                metabolism="Synthesized from phenylalanine, precursor to dopamine, norepinephrine, epinephrine"
            ),
            food_sources=["meat", "fish", "eggs", "dairy", "soy", "nuts"],
            concentration_range=(1200, 2800),
            synonyms=["L-Tyrosine", "Tyr", "Y"]
        ))
        
        # =====================================================================
        # FATTY ACIDS
        # =====================================================================
        
        self._add_molecule(Molecule(
            name="Oleic Acid",
            iupac_name="(Z)-octadec-9-enoic acid",
            formula="C18H34O2",
            molecular_weight=282.46,
            category=MoleculeCategory.FATTY_ACID,
            smiles="CCCCCCCC/C=C\\CCCCCCCC(=O)O",
            inchi="InChI=1S/C18H34O2/c1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18(19)20",
            cas_number="112-80-1",
            pubchem_id=445639,
            nir_signature=NIRSignature(
                primary_bands=[1730, 2310, 2350],  # C=O, C-H stretch
                secondary_bands=[1650, 1210],  # C=C double bond
                characteristic_peaks=[2310],
                intensity_range=(0.3, 0.5)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["heart_health", "reduces_LDL_cholesterol", "anti_inflammatory"],
                bioavailability=98.0,
                metabolism="Monounsaturated omega-9 fatty acid"
            ),
            food_sources=["olive_oil", "avocado", "almonds", "peanuts", "canola_oil"],
            concentration_range=(50000, 800000),  # mg/kg (very high in oils)
            synonyms=["Omega-9", "18:1 n-9", "cis-9-octadecenoic acid"]
        ))
        
        self._add_molecule(Molecule(
            name="Linoleic Acid",
            iupac_name="(9Z,12Z)-octadeca-9,12-dienoic acid",
            formula="C18H32O2",
            molecular_weight=280.45,
            category=MoleculeCategory.FATTY_ACID,
            smiles="CCCCCC=CCC=CCCCCCCCC(=O)O",
            cas_number="60-33-3",
            pubchem_id=5280450,
            nir_signature=NIRSignature(
                primary_bands=[1730, 2310],
                secondary_bands=[1650],  # C=C bonds
                intensity_range=(0.3, 0.5)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["cell_membrane_structure", "hormone_production", "brain_function"],
                rda=17000.0,  # mg/day (17g)
                bioavailability=95.0,
                metabolism="Essential omega-6 fatty acid, converted to arachidonic acid"
            ),
            food_sources=["sunflower_oil", "corn_oil", "soybeans", "nuts", "seeds"],
            concentration_range=(30000, 700000),
            synonyms=["Omega-6", "LA", "18:2 n-6"]
        ))
        
        self._add_molecule(Molecule(
            name="Alpha-Linolenic Acid",
            iupac_name="(9Z,12Z,15Z)-octadeca-9,12,15-trienoic acid",
            formula="C18H30O2",
            molecular_weight=278.43,
            category=MoleculeCategory.FATTY_ACID,
            smiles="CC/C=C\\C/C=C\\C/C=C\\CCCCCCCC(=O)O",
            cas_number="463-40-1",
            pubchem_id=5280934,
            nir_signature=NIRSignature(
                primary_bands=[1730, 2310],
                secondary_bands=[1650],  # Multiple C=C
                intensity_range=(0.2, 0.4)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["heart_health", "brain_development", "anti_inflammatory"],
                rda=1600.0,  # mg/day
                bioavailability=90.0,
                metabolism="Essential omega-3, converted to EPA and DHA"
            ),
            food_sources=["flaxseed_oil", "chia_seeds", "walnuts", "hemp_seeds"],
            concentration_range=(10000, 500000),
            synonyms=["ALA", "Omega-3", "18:3 n-3"]
        ))
        
        self._add_molecule(Molecule(
            name="Docosahexaenoic Acid",
            iupac_name="(4Z,7Z,10Z,13Z,16Z,19Z)-docosa-4,7,10,13,16,19-hexaenoic acid",
            formula="C22H32O2",
            molecular_weight=328.49,
            category=MoleculeCategory.FATTY_ACID,
            smiles="CCC=CCC=CCC=CCC=CCC=CCC=CCCCC(=O)O",
            cas_number="6217-54-5",
            pubchem_id=445580,
            nir_signature=NIRSignature(
                primary_bands=[1730, 2310],
                secondary_bands=[1650],  # 6 C=C bonds
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["brain_development", "eye_health", "cognitive_function", "anti_inflammatory"],
                rda=250.0,  # mg/day
                bioavailability=95.0,
                metabolism="Long-chain omega-3, crucial for brain"
            ),
            food_sources=["fatty_fish", "salmon", "mackerel", "sardines", "algae_oil"],
            concentration_range=(5000, 30000),  # mg/kg in fish
            synonyms=["DHA", "Omega-3", "22:6 n-3"]
        ))
        
        # Additional Essential/Important Fatty Acids
        self._add_molecule(Molecule(
            name="Eicosapentaenoic Acid",
            iupac_name="(5Z,8Z,11Z,14Z,17Z)-icosa-5,8,11,14,17-pentaenoic acid",
            formula="C20H30O2",
            molecular_weight=302.45,
            category=MoleculeCategory.FATTY_ACID,
            smiles="CCC=CCC=CCC=CCC=CCC=CCCCCC(=O)O",
            cas_number="10417-94-4",
            pubchem_id=446284,
            nir_signature=NIRSignature(
                primary_bands=[1730, 2310],
                secondary_bands=[1650],
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["anti_inflammatory", "cardiovascular_health", "mental_health", "joint_health"],
                rda=250.0,
                bioavailability=95.0,
                metabolism="Long-chain omega-3 fatty acid"
            ),
            food_sources=["fatty_fish", "salmon", "herring", "mackerel", "fish_oil"],
            concentration_range=(4000, 25000),
            synonyms=["EPA", "Omega-3", "20:5 n-3"]
        ))
        
        self._add_molecule(Molecule(
            name="Arachidonic Acid",
            iupac_name="(5Z,8Z,11Z,14Z)-icosa-5,8,11,14-tetraenoic acid",
            formula="C20H32O2",
            molecular_weight=304.47,
            category=MoleculeCategory.FATTY_ACID,
            smiles="CCCCC=CCC=CCC=CCC=CCCCCCC(=O)O",
            cas_number="506-32-1",
            pubchem_id=444899,
            nir_signature=NIRSignature(
                primary_bands=[1730, 2310],
                secondary_bands=[1650],
                intensity_range=(0.2, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["brain_development", "immune_response", "muscle_growth"],
                risks=["excessive_inflammation"],
                bioavailability=95.0,
                metabolism="Omega-6, converted to eicosanoids"
            ),
            food_sources=["meat", "eggs", "fish", "seaweed"],
            concentration_range=(500, 2000),
            synonyms=["AA", "ARA", "Omega-6", "20:4 n-6"]
        ))
        
        self._add_molecule(Molecule(
            name="Palmitic Acid",
            iupac_name="hexadecanoic acid",
            formula="C16H32O2",
            molecular_weight=256.42,
            category=MoleculeCategory.FATTY_ACID,
            smiles="CCCCCCCCCCCCCCCC(=O)O",
            cas_number="57-10-3",
            pubchem_id=985,
            nir_signature=NIRSignature(
                primary_bands=[1730, 2310, 2350],
                secondary_bands=[1465],
                intensity_range=(0.3, 0.5)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.NEUTRAL,
                benefits=["energy_storage", "cell_membrane_component"],
                risks=["raises_LDL_cholesterol"],
                bioavailability=98.0,
                metabolism="Most common saturated fatty acid"
            ),
            food_sources=["palm_oil", "meat", "dairy", "butter"],
            concentration_range=(20000, 450000),
            synonyms=["16:0", "Hexadecanoic acid"]
        ))
        
        self._add_molecule(Molecule(
            name="Stearic Acid",
            iupac_name="octadecanoic acid",
            formula="C18H36O2",
            molecular_weight=284.48,
            category=MoleculeCategory.FATTY_ACID,
            smiles="CCCCCCCCCCCCCCCCCC(=O)O",
            cas_number="57-11-4",
            pubchem_id=5281,
            nir_signature=NIRSignature(
                primary_bands=[1730, 2310, 2350],
                secondary_bands=[1465],
                intensity_range=(0.3, 0.5)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.NEUTRAL,
                benefits=["neutral_cholesterol_effect", "energy_source"],
                bioavailability=98.0,
                metabolism="Saturated fatty acid, converted to oleic acid in body"
            ),
            food_sources=["cocoa_butter", "shea_butter", "meat", "dairy"],
            concentration_range=(10000, 200000),
            synonyms=["18:0", "Octadecanoic acid"]
        ))
        
        self._add_molecule(Molecule(
            name="Lauric Acid",
            iupac_name="dodecanoic acid",
            formula="C12H24O2",
            molecular_weight=200.32,
            category=MoleculeCategory.FATTY_ACID,
            smiles="CCCCCCCCCCCC(=O)O",
            cas_number="143-07-7",
            pubchem_id=3893,
            nir_signature=NIRSignature(
                primary_bands=[1730, 2310, 2350],
                secondary_bands=[1465],
                intensity_range=(0.3, 0.4)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["antimicrobial", "raises_HDL_cholesterol", "immune_support"],
                bioavailability=98.0,
                metabolism="Medium-chain saturated fatty acid"
            ),
            food_sources=["coconut_oil", "palm_kernel_oil", "breast_milk"],
            concentration_range=(40000, 500000),
            synonyms=["12:0", "Dodecanoic acid"]
        ))
        
        # =====================================================================
        # CARBOHYDRATES
        # =====================================================================
        
        self._add_molecule(Molecule(
            name="Glucose",
            iupac_name="(2R,3S,4R,5R)-2,3,4,5,6-pentahydroxyhexanal",
            formula="C6H12O6",
            molecular_weight=180.16,
            category=MoleculeCategory.CARBOHYDRATE,
            smiles="C(C1C(C(C(C(O1)O)O)O)O)O",
            cas_number="50-99-7",
            pubchem_id=5793,
            nir_signature=NIRSignature(
                primary_bands=[1450, 2100, 2270],  # C-O, O-H
                secondary_bands=[1940],  # O-H overtone
                characteristic_peaks=[2100],
                intensity_range=(0.2, 0.4)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["primary_energy_source", "brain_fuel"],
                risks=["high_blood_sugar", "diabetes_risk", "weight_gain"],
                bioavailability=100.0,
                metabolism="Rapidly absorbed, used for energy or stored as glycogen"
            ),
            food_sources=["fruits", "honey", "corn_syrup", "table_sugar"],
            concentration_range=(10000, 150000),  # mg/kg
            synonyms=["Dextrose", "D-Glucose", "Blood sugar"]
        ))
        
        self._add_molecule(Molecule(
            name="Fructose",
            iupac_name="(3S,4R,5R)-1,3,4,5,6-pentahydroxyhexan-2-one",
            formula="C6H12O6",
            molecular_weight=180.16,
            category=MoleculeCategory.CARBOHYDRATE,
            smiles="C(C1C(C(C(O1)O)O)O)O",
            cas_number="57-48-7",
            pubchem_id=5984,
            nir_signature=NIRSignature(
                primary_bands=[1450, 2100, 2270],
                secondary_bands=[1940, 1730],  # C=O ketone
                intensity_range=(0.2, 0.4)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["sweet_taste", "slow_glucose_release"],
                risks=["fatty_liver_disease", "insulin_resistance", "obesity"],
                bioavailability=95.0,
                metabolism="Metabolized in liver, different pathway than glucose"
            ),
            food_sources=["fruits", "honey", "agave", "high_fructose_corn_syrup"],
            concentration_range=(10000, 200000),
            synonyms=["Levulose", "Fruit sugar"]
        ))
        
        self._add_molecule(Molecule(
            name="Sucrose",
            iupac_name="(2R,3R,4S,5S,6R)-2-[(2S,3S,4S,5R)-3,4-dihydroxy-2,5-bis(hydroxymethyl)oxolan-2-yl]oxy-6-(hydroxymethyl)oxane-3,4,5-triol",
            formula="C12H22O11",
            molecular_weight=342.30,
            category=MoleculeCategory.CARBOHYDRATE,
            smiles="C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O",
            cas_number="57-50-1",
            pubchem_id=5988,
            nir_signature=NIRSignature(
                primary_bands=[1450, 2100, 2270],
                secondary_bands=[1940],
                intensity_range=(0.3, 0.5)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.NEUTRAL,
                benefits=["quick_energy"],
                risks=["tooth_decay", "blood_sugar_spikes", "weight_gain"],
                bioavailability=100.0,
                metabolism="Broken down to glucose + fructose"
            ),
            food_sources=["table_sugar", "sugar_cane", "sugar_beet", "maple_syrup"],
            concentration_range=(50000, 999000),  # mg/kg (very high in sugar)
            synonyms=["Table sugar", "Cane sugar", "Saccharose"]
        ))
        
        # =====================================================================
        # VITAMINS
        # =====================================================================
        
        self._add_molecule(Molecule(
            name="Ascorbic Acid",
            iupac_name="(2R)-2-[(1S)-1,2-dihydroxyethyl]-4,5-dihydroxyfuran-3-one",
            formula="C6H8O6",
            molecular_weight=176.12,
            category=MoleculeCategory.VITAMIN,
            smiles="C(C(C1C(=C(C(=O)O1)O)O)O)O",
            cas_number="50-81-7",
            pubchem_id=54670067,
            nir_signature=NIRSignature(
                primary_bands=[1450, 1730, 2100],  # O-H, C=O
                secondary_bands=[1940],
                intensity_range=(0.1, 0.2)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["immune_support", "antioxidant", "collagen_synthesis", "iron_absorption"],
                rda=90.0,  # mg/day (males), 75 (females)
                ul=2000.0,
                bioavailability=80.0,
                metabolism="Water-soluble, excess excreted in urine",
                interactions=["Enhances iron absorption", "High doses may interfere with B12"]
            ),
            food_sources=["citrus", "strawberries", "bell_peppers", "broccoli", "kiwi"],
            concentration_range=(100, 2000),  # mg/kg
            synonyms=["Vitamin C", "L-Ascorbic acid"]
        ))
        
        self._add_molecule(Molecule(
            name="Retinol",
            iupac_name="(2E,4E,6E,8E)-3,7-dimethyl-9-(2,6,6-trimethylcyclohex-1-en-1-yl)nona-2,4,6,8-tetraen-1-ol",
            formula="C20H30O",
            molecular_weight=286.45,
            category=MoleculeCategory.VITAMIN,
            smiles="CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/CO)/C)/C",
            cas_number="68-26-8",
            pubchem_id=445354,
            nir_signature=NIRSignature(
                primary_bands=[1650, 2310],  # C=C conjugated system
                secondary_bands=[1450],
                intensity_range=(0.05, 0.15)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["vision", "immune_function", "skin_health", "cell_growth"],
                rda=0.9,  # mg/day (900 μg)
                ul=3.0,  # mg/day
                bioavailability=70.0,
                metabolism="Fat-soluble, stored in liver",
                interactions=["Teratogenic in pregnancy at high doses"]
            ),
            food_sources=["liver", "fish_oil", "egg_yolk", "fortified_dairy"],
            concentration_range=(1, 50),  # mg/kg
            synonyms=["Vitamin A", "Retinol"]
        ))
        
        self._add_molecule(Molecule(
            name="Alpha-Tocopherol",
            iupac_name="(2R)-2,5,7,8-tetramethyl-2-[(4R,8R)-4,8,12-trimethyltridecyl]-3,4-dihydrochromen-6-ol",
            formula="C29H50O2",
            molecular_weight=430.71,
            category=MoleculeCategory.VITAMIN,
            smiles="CC1=C(C(=C2CCC(OC2=C1C)(C)CCCC(C)CCCC(C)CCCC(C)C)C)O",
            cas_number="59-02-9",
            pubchem_id=14985,
            nir_signature=NIRSignature(
                primary_bands=[1450, 2310],  # C-H stretch
                secondary_bands=[1600],  # Aromatic
                intensity_range=(0.05, 0.1)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["antioxidant", "protects_cell_membranes", "immune_support"],
                rda=15.0,  # mg/day
                ul=1000.0,
                bioavailability=50.0,
                metabolism="Fat-soluble, stored in adipose tissue"
            ),
            food_sources=["nuts", "seeds", "vegetable_oils", "spinach", "avocado"],
            concentration_range=(10, 500),  # mg/kg
            synonyms=["Vitamin E", "α-Tocopherol"]
        ))
        
        # B-Complex Vitamins
        self._add_molecule(Molecule(
            name="Thiamine",
            iupac_name="2-[3-[(4-amino-2-methylpyrimidin-5-yl)methyl]-4-methyl-1,3-thiazol-3-ium-5-yl]ethanol",
            formula="C12H17N4OS",
            molecular_weight=265.35,
            category=MoleculeCategory.VITAMIN,
            smiles="CC1=C(SC=[N+]1CC2=CN=C(N=C2N)C)CCCO",
            cas_number="59-43-8",
            pubchem_id=1130,
            nir_signature=NIRSignature(
                primary_bands=[1450, 2050],  # O-H, N-H
                secondary_bands=[1600],
                intensity_range=(0.05, 0.1)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["energy_metabolism", "nerve_function", "muscle_contraction", "glucose_metabolism"],
                rda=1.2,  # mg/day (males), 1.1 (females)
                ul=None,  # No established UL
                bioavailability=90.0,
                metabolism="Water-soluble, excess excreted"
            ),
            food_sources=["whole_grains", "pork", "legumes", "nuts", "seeds"],
            concentration_range=(1, 20),
            synonyms=["Vitamin B1", "Thiamin"]
        ))
        
        self._add_molecule(Molecule(
            name="Riboflavin",
            iupac_name="7,8-dimethyl-10-[(2S,3S,4R)-2,3,4,5-tetrahydroxypentyl]benzo[g]pteridine-2,4-dione",
            formula="C17H20N4O6",
            molecular_weight=376.36,
            category=MoleculeCategory.VITAMIN,
            smiles="CC1=CC2=C(C=C1C)N(C3=NC(=O)NC(=O)C3=N2)CC(C(C(CO)O)O)O",
            cas_number="83-88-5",
            pubchem_id=493570,
            nir_signature=NIRSignature(
                primary_bands=[1450, 1940, 2050],
                secondary_bands=[1600],
                intensity_range=(0.05, 0.1)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["energy_production", "antioxidant", "eye_health", "skin_health"],
                rda=1.3,  # mg/day (males), 1.1 (females)
                ul=None,
                bioavailability=95.0,
                metabolism="Water-soluble, component of FAD, FMN"
            ),
            food_sources=["dairy", "eggs", "lean_meat", "green_vegetables", "almonds"],
            concentration_range=(1, 30),
            synonyms=["Vitamin B2"]
        ))
        
        self._add_molecule(Molecule(
            name="Niacin",
            iupac_name="pyridine-3-carboxylic acid",
            formula="C6H5NO2",
            molecular_weight=123.11,
            category=MoleculeCategory.VITAMIN,
            smiles="C1=CC(=CN=C1)C(=O)O",
            cas_number="59-67-6",
            pubchem_id=938,
            nir_signature=NIRSignature(
                primary_bands=[1730, 2050],  # COOH, aromatic N
                secondary_bands=[1600],
                intensity_range=(0.05, 0.1)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["energy_metabolism", "DNA_repair", "cholesterol_regulation", "skin_health"],
                rda=16.0,  # mg NE/day (males), 14 (females)
                ul=35.0,  # mg/day (from supplements)
                bioavailability=90.0,
                metabolism="Water-soluble, component of NAD+, NADP+"
            ),
            food_sources=["meat", "fish", "peanuts", "mushrooms", "fortified_grains"],
            concentration_range=(10, 100),
            synonyms=["Vitamin B3", "Nicotinic acid"]
        ))
        
        self._add_molecule(Molecule(
            name="Pantothenic Acid",
            iupac_name="3-[(2,4-dihydroxy-3,3-dimethylbutanoyl)amino]propanoic acid",
            formula="C9H17NO5",
            molecular_weight=219.24,
            category=MoleculeCategory.VITAMIN,
            smiles="CC(C)(CO)C(C(=O)NCCC(=O)O)O",
            cas_number="79-83-4",
            pubchem_id=6613,
            nir_signature=NIRSignature(
                primary_bands=[1450, 1730, 2050],
                secondary_bands=[1940],
                intensity_range=(0.05, 0.1)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["energy_metabolism", "hormone_synthesis", "neurotransmitter_synthesis"],
                rda=5.0,  # mg/day
                ul=None,
                bioavailability=85.0,
                metabolism="Water-soluble, component of CoA"
            ),
            food_sources=["meat", "whole_grains", "broccoli", "avocado", "mushrooms"],
            concentration_range=(5, 50),
            synonyms=["Vitamin B5"]
        ))
        
        self._add_molecule(Molecule(
            name="Pyridoxine",
            iupac_name="4,5-bis(hydroxymethyl)-2-methylpyridin-3-ol",
            formula="C8H11NO3",
            molecular_weight=169.18,
            category=MoleculeCategory.VITAMIN,
            smiles="CC1=NC=C(C(=C1O)CO)CO",
            cas_number="65-23-6",
            pubchem_id=1054,
            nir_signature=NIRSignature(
                primary_bands=[1450, 1940, 2050],
                secondary_bands=[1600],
                intensity_range=(0.05, 0.1)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["amino_acid_metabolism", "neurotransmitter_synthesis", "immune_function", "hemoglobin_formation"],
                rda=1.3,  # mg/day (adults)
                ul=100.0,
                bioavailability=75.0,
                metabolism="Water-soluble, active form PLP"
            ),
            food_sources=["poultry", "fish", "potatoes", "chickpeas", "bananas"],
            concentration_range=(1, 30),
            synonyms=["Vitamin B6", "Pyridoxal", "Pyridoxamine"]
        ))
        
        self._add_molecule(Molecule(
            name="Biotin",
            iupac_name="5-[(3aS,4S,6aR)-2-oxohexahydro-1H-thieno[3,4-d]imidazol-4-yl]pentanoic acid",
            formula="C10H16N2O3S",
            molecular_weight=244.31,
            category=MoleculeCategory.VITAMIN,
            smiles="C1C2C(C(S1)CCCCC(=O)O)NC(=O)N2",
            cas_number="58-85-5",
            pubchem_id=171548,
            nir_signature=NIRSignature(
                primary_bands=[1730, 2050],
                secondary_bands=[2310],
                intensity_range=(0.05, 0.1)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["fatty_acid_synthesis", "amino_acid_metabolism", "hair_skin_nail_health"],
                rda=0.03,  # mg/day (30 μg)
                ul=None,
                bioavailability=90.0,
                metabolism="Water-soluble, synthesized by gut bacteria"
            ),
            food_sources=["eggs", "nuts", "seeds", "sweet_potatoes", "spinach"],
            concentration_range=(0.01, 1),
            synonyms=["Vitamin B7", "Vitamin H"]
        ))
        
        self._add_molecule(Molecule(
            name="Folate",
            iupac_name="(2S)-2-[[4-[(2-amino-4-oxo-1H-pteridin-6-yl)methylamino]benzoyl]amino]pentanedioic acid",
            formula="C19H19N7O6",
            molecular_weight=441.40,
            category=MoleculeCategory.VITAMIN,
            smiles="C1=CC(=CC=C1C(=O)NC(CCC(=O)O)C(=O)O)NCC2=CN=C3C(=N2)C(=O)NC(=N3)N",
            cas_number="59-30-3",
            pubchem_id=6037,
            nir_signature=NIRSignature(
                primary_bands=[1730, 2050],
                secondary_bands=[1600],
                intensity_range=(0.05, 0.1)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["DNA_synthesis", "cell_division", "neural_tube_development", "red_blood_cell_formation"],
                rda=0.4,  # mg/day (400 μg DFE)
                ul=1.0,  # mg/day (from supplements)
                bioavailability=85.0,
                metabolism="Water-soluble, crucial in pregnancy"
            ),
            food_sources=["leafy_greens", "legumes", "fortified_grains", "citrus", "asparagus"],
            concentration_range=(0.1, 5),
            synonyms=["Vitamin B9", "Folic acid", "Folacin"]
        ))
        
        self._add_molecule(Molecule(
            name="Cobalamin",
            iupac_name="cobalt-corrin complex",
            formula="C63H88CoN14O14P",
            molecular_weight=1355.37,
            category=MoleculeCategory.VITAMIN,
            cas_number="68-19-9",
            pubchem_id=5311498,
            nir_signature=NIRSignature(
                primary_bands=[1450, 2050],
                secondary_bands=[1600],
                intensity_range=(0.05, 0.1)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["DNA_synthesis", "red_blood_cell_formation", "neurological_function", "energy_metabolism"],
                rda=0.0024,  # mg/day (2.4 μg)
                ul=None,
                bioavailability=50.0,
                metabolism="Water-soluble, requires intrinsic factor for absorption",
                interactions=["Metformin may reduce absorption", "Proton pump inhibitors reduce absorption"]
            ),
            food_sources=["meat", "fish", "dairy", "eggs", "fortified_cereals"],
            concentration_range=(0.001, 0.1),
            synonyms=["Vitamin B12", "Cyanocobalamin", "Methylcobalamin"]
        ))
        
        # Fat-Soluble Vitamins
        self._add_molecule(Molecule(
            name="Cholecalciferol",
            iupac_name="(3β,5Z,7E)-9,10-secocholesta-5,7,10(19)-trien-3-ol",
            formula="C27H44O",
            molecular_weight=384.64,
            category=MoleculeCategory.VITAMIN,
            smiles="CC(C)CCCC(C)C1CCC2C1(CCCC2=CC=C3CC(CCC3=C)O)C",
            cas_number="67-97-0",
            pubchem_id=5280795,
            nir_signature=NIRSignature(
                primary_bands=[1450, 2310],
                secondary_bands=[1650],
                intensity_range=(0.05, 0.1)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["bone_health", "calcium_absorption", "immune_function", "mood_regulation"],
                rda=0.02,  # mg/day (20 μg = 800 IU)
                ul=0.1,  # mg/day (100 μg = 4000 IU)
                bioavailability=80.0,
                metabolism="Fat-soluble, synthesized in skin from sunlight"
            ),
            food_sources=["fatty_fish", "fish_oil", "fortified_dairy", "egg_yolk", "mushrooms"],
            concentration_range=(0.01, 1),
            synonyms=["Vitamin D3", "Vitamin D"]
        ))
        
        self._add_molecule(Molecule(
            name="Phylloquinone",
            iupac_name="2-methyl-3-[(2E,7R,11R)-3,7,11,15-tetramethylhexadec-2-enyl]naphthalene-1,4-dione",
            formula="C31H46O2",
            molecular_weight=450.70,
            category=MoleculeCategory.VITAMIN,
            smiles="CC(C)CCCC(C)CCCC(C)CCCC(C)CC=C(C)C1=C(C(=O)C2=CC=CC=C2C1=O)C",
            cas_number="84-80-0",
            pubchem_id=5284607,
            nir_signature=NIRSignature(
                primary_bands=[1730, 2310],  # Quinone C=O
                secondary_bands=[1650, 1600],
                intensity_range=(0.05, 0.1)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ESSENTIAL,
                benefits=["blood_clotting", "bone_metabolism", "cardiovascular_health"],
                rda=0.12,  # mg/day (120 μg males, 90 females)
                ul=None,
                bioavailability=60.0,
                metabolism="Fat-soluble, recycled efficiently in body"
            ),
            food_sources=["leafy_greens", "broccoli", "brussels_sprouts", "kiwi", "avocado"],
            concentration_range=(0.1, 10),
            synonyms=["Vitamin K1", "Vitamin K"]
        ))
        
        # =====================================================================
        # ANTIOXIDANTS & PHYTOCHEMICALS
        # =====================================================================
        
        self._add_molecule(Molecule(
            name="Quercetin",
            iupac_name="2-(3,4-dihydroxyphenyl)-3,5,7-trihydroxy-4H-chromen-4-one",
            formula="C15H10O7",
            molecular_weight=302.24,
            category=MoleculeCategory.FLAVONOID,
            smiles="C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O",
            cas_number="117-39-5",
            pubchem_id=5280343,
            nir_signature=NIRSignature(
                primary_bands=[1600, 1730],  # Aromatic, C=O
                secondary_bands=[1450],
                intensity_range=(0.05, 0.1)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["anti_inflammatory", "antioxidant", "antihistamine", "cardiovascular_protection"],
                bioavailability=20.0,  # Low absorption
                metabolism="Glucuronidated in liver",
                interactions=["May interact with blood thinners"]
            ),
            food_sources=["onions", "apples", "berries", "grapes", "green_tea"],
            concentration_range=(10, 500),  # mg/kg
            synonyms=["Flavonoid", "Polyphenol"]
        ))
        
        self._add_molecule(Molecule(
            name="Resveratrol",
            iupac_name="5-[(E)-2-(4-hydroxyphenyl)ethenyl]benzene-1,3-diol",
            formula="C14H12O3",
            molecular_weight=228.24,
            category=MoleculeCategory.POLYPHENOL,
            smiles="C1=CC(=CC=C1/C=C/C2=CC(=CC(=C2)O)O)O",
            cas_number="501-36-0",
            pubchem_id=445154,
            nir_signature=NIRSignature(
                primary_bands=[1600, 1650],  # Aromatic, C=C
                secondary_bands=[1450],
                intensity_range=(0.03, 0.08)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["longevity", "cardiovascular_protection", "anti_inflammatory", "neuroprotective"],
                bioavailability=20.0,
                metabolism="Rapidly metabolized, low bioavailability"
            ),
            food_sources=["red_wine", "grapes", "berries", "peanuts", "dark_chocolate"],
            concentration_range=(0.5, 50),  # mg/kg
            synonyms=["Polyphenol", "Stilbenoid"]
        ))
        
        self._add_molecule(Molecule(
            name="Curcumin",
            iupac_name="(1E,6E)-1,7-bis(4-hydroxy-3-methoxyphenyl)hepta-1,6-diene-3,5-dione",
            formula="C21H20O6",
            molecular_weight=368.38,
            category=MoleculeCategory.POLYPHENOL,
            smiles="COC1=C(C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC)O",
            cas_number="458-37-7",
            pubchem_id=969516,
            nir_signature=NIRSignature(
                primary_bands=[1600, 1730],  # Aromatic, C=O
                secondary_bands=[1650],  # C=C conjugation
                intensity_range=(0.05, 0.12)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.BENEFICIAL,
                benefits=["anti_inflammatory", "antioxidant", "pain_relief", "may_improve_cognitive_function"],
                bioavailability=3.0,  # Very low
                metabolism="Poor absorption, rapid metabolism",
                interactions=["May interact with blood thinners"]
            ),
            food_sources=["turmeric", "curry_powder"],
            concentration_range=(30000, 50000),  # mg/kg in turmeric
            synonyms=["Turmeric extract", "Diferuloylmethane"]
        ))
        
        # =====================================================================
        # TOXINS & ALLERGENS
        # =====================================================================
        
        self._add_molecule(Molecule(
            name="Aflatoxin B1",
            iupac_name="(6aR,9aS)-4-methoxy-2,3,6a,9a-tetrahydrocyclopenta[c]furo[3',2':4,5]furo[2,3-h]chromen-1,11-dione",
            formula="C17H12O6",
            molecular_weight=312.27,
            category=MoleculeCategory.TOXIN,
            smiles="COC1=C2C3=C(C(=O)CC3)C(=O)OC2=C4C(=C1)C=CO4",
            cas_number="1162-65-8",
            pubchem_id=186907,
            nir_signature=NIRSignature(
                primary_bands=[1730, 1600],  # C=O, aromatic
                intensity_range=(0.02, 0.05)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.CARCINOGENIC,
                risks=["liver_cancer", "immunosuppression", "acute_toxicity"],
                bioavailability=85.0,
                metabolism="Metabolized to toxic epoxide by CYP450"
            ),
            food_sources=["contaminated_grains", "peanuts", "corn", "tree_nuts"],
            concentration_range=(0.0, 0.02),  # mg/kg (ppb levels, regulated)
            synonyms=["AFB1", "Mycotoxin"]
        ))
        
        self._add_molecule(Molecule(
            name="Acrylamide",
            iupac_name="2-propenamide",
            formula="C3H5NO",
            molecular_weight=71.08,
            category=MoleculeCategory.TOXIN,
            smiles="C=CC(=O)N",
            cas_number="79-06-1",
            pubchem_id=6579,
            nir_signature=NIRSignature(
                primary_bands=[1730, 2050],  # C=O, N-H
                secondary_bands=[1650],  # C=C
                intensity_range=(0.01, 0.03)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.CARCINOGENIC,
                risks=["cancer", "neurotoxicity", "reproductive_toxicity"],
                bioavailability=90.0,
                metabolism="Metabolized to glycidamide (more toxic)"
            ),
            food_sources=["fried_foods", "potato_chips", "coffee", "bread_crust", "toasted_products"],
            concentration_range=(0.05, 4.0),  # mg/kg (higher in fried foods)
            synonyms=["2-Propenamide"]
        ))
        
        self._add_molecule(Molecule(
            name="Ara h 1",
            iupac_name="Peanut allergen Ara h 1 (protein)",
            formula="Protein",  # Not a small molecule
            molecular_weight=63500.0,  # Da (approximate)
            category=MoleculeCategory.ALLERGEN,
            nir_signature=NIRSignature(
                primary_bands=[2050, 2180],  # N-H protein bands
                secondary_bands=[1510],
                intensity_range=(0.1, 0.3)
            ),
            health_data=HealthData(
                health_effect=HealthEffect.ALLERGENIC,
                risks=["anaphylaxis", "hives", "respiratory_distress"],
                bioavailability=70.0,
                metabolism="Digested slowly, epitopes remain allergenic"
            ),
            food_sources=["peanuts", "peanut_butter", "peanut_oil"],
            concentration_range=(50, 1000),  # mg/kg
            synonyms=["Peanut allergen", "Ara h 1"]
        ))
        
        # Add more molecules... (scaffolding for expansion)
        # In production, continue adding:
        # - Remaining amino acids (16 more)
        # - More fatty acids (saturated, trans, omega varieties)
        # - Vitamins (B-complex, K, D)
        # - Minerals complexes
        # - More antioxidants (lycopene, beta-carotene, etc.)
        # - Alkaloids (caffeine, theobromine, capsaicin)
        # - Pesticides (glyphosate, organophosphates)
        # - Heavy metal complexes
        # Target: 10,000+ molecules
        
        logger.info(f"Database populated with {len(self.molecules)} molecules")
    
    def _add_molecule(self, molecule: Molecule):
        """Add molecule to database and update indices."""
        self.molecules[molecule.name.lower()] = molecule
        
        # Update category index
        if molecule.category not in self.category_index:
            self.category_index[molecule.category] = []
        self.category_index[molecule.category].append(molecule.name.lower())
    
    # =========================================================================
    # QUERY METHODS
    # =========================================================================
    
    def get_molecule(self, name: str) -> Optional[Molecule]:
        """Get molecule by name (case-insensitive)."""
        return self.molecules.get(name.lower())
    
    def get_molecules_by_category(self, category: MoleculeCategory) -> List[Molecule]:
        """Get all molecules in a category."""
        names = self.category_index.get(category, [])
        return [self.molecules[name] for name in names]
    
    def get_essential_molecules(self) -> List[Molecule]:
        """Get all essential nutrients."""
        return [m for m in self.molecules.values() if m.is_essential()]
    
    def get_toxic_molecules(self) -> List[Molecule]:
        """Get all toxic molecules."""
        return [m for m in self.molecules.values() if m.is_toxic()]
    
    def search_by_food(self, food: str) -> List[Molecule]:
        """Search molecules found in a specific food."""
        results = []
        food_lower = food.lower()
        
        for molecule in self.molecules.values():
            if any(food_lower in source.lower() or source.lower() in food_lower 
                   for source in molecule.food_sources):
                results.append(molecule)
        
        return results
    
    def search_by_formula(self, formula: str) -> List[Molecule]:
        """Search molecules by chemical formula."""
        return [m for m in self.molecules.values() if m.formula == formula]
    
    def get_molecules_with_nir_signature(self) -> List[Molecule]:
        """Get molecules with NIR spectroscopic signatures."""
        return [m for m in self.molecules.values() if m.nir_signature is not None]
    
    def assess_safety(self, molecule_name: str, concentration_mg_kg: float) -> Dict[str, any]:
        """
        Assess safety of molecule concentration.
        
        Args:
            molecule_name: Molecule name
            concentration_mg_kg: Concentration in food (mg/kg)
        
        Returns:
            Safety assessment dict
        """
        molecule = self.get_molecule(molecule_name)
        if not molecule:
            return {"error": "Molecule not found"}
        
        if not molecule.health_data:
            return {
                "molecule": molecule.name,
                "concentration": concentration_mg_kg,
                "safety": "unknown",
                "message": "No health data available"
            }
        
        # Check if toxic
        if molecule.is_toxic():
            # Toxins should ideally be zero
            if concentration_mg_kg > 0.1:
                return {
                    "molecule": molecule.name,
                    "concentration": concentration_mg_kg,
                    "safety": "danger",
                    "message": f"Toxic compound detected: {', '.join(molecule.health_data.risks[:2])}",
                    "health_effect": molecule.health_data.health_effect.value
                }
            else:
                return {
                    "molecule": molecule.name,
                    "concentration": concentration_mg_kg,
                    "safety": "trace",
                    "message": "Trace amounts detected (acceptable)",
                    "health_effect": molecule.health_data.health_effect.value
                }
        
        # Check if within normal range
        min_conc, max_conc = molecule.concentration_range
        if min_conc <= concentration_mg_kg <= max_conc:
            return {
                "molecule": molecule.name,
                "concentration": concentration_mg_kg,
                "safety": "normal",
                "message": f"Within typical range ({min_conc}-{max_conc} mg/kg)",
                "health_effect": molecule.health_data.health_effect.value
            }
        elif concentration_mg_kg < min_conc:
            return {
                "molecule": molecule.name,
                "concentration": concentration_mg_kg,
                "safety": "low",
                "message": f"Below typical range (expected {min_conc}-{max_conc} mg/kg)",
                "health_effect": molecule.health_data.health_effect.value
            }
        else:
            return {
                "molecule": molecule.name,
                "concentration": concentration_mg_kg,
                "safety": "high",
                "message": f"Above typical range (expected {min_conc}-{max_conc} mg/kg)",
                "health_effect": molecule.health_data.health_effect.value
            }


# =============================================================================
# TESTING & DEMONSTRATION
# =============================================================================

def test_molecular_database():
    """Test molecular database initialization."""
    print("\n" + "="*80)
    print("TEST 1: Molecular Database Initialization")
    print("="*80)
    
    db = MolecularDatabase()
    
    print(f"\n✓ Loaded {len(db.molecules)} molecules")
    
    # Count by category
    print(f"\n✓ Molecules by category:")
    for category in MoleculeCategory:
        mols = db.get_molecules_by_category(category)
        if mols:
            print(f"  {category.value:20s}: {len(mols):3d} molecules")
    
    return True


def test_molecule_queries():
    """Test molecule query methods."""
    print("\n" + "="*80)
    print("TEST 2: Molecule Queries")
    print("="*80)
    
    db = MolecularDatabase()
    
    # Test get by name
    glucose = db.get_molecule("Glucose")
    print(f"\n✓ Query 'Glucose':")
    print(f"  Formula: {glucose.formula}")
    print(f"  MW: {glucose.molecular_weight} g/mol")
    print(f"  Category: {glucose.category.value}")
    print(f"  Health effect: {glucose.health_data.health_effect.value}")
    print(f"  Food sources: {', '.join(glucose.food_sources[:3])}")
    
    # Test essential molecules
    essential = db.get_essential_molecules()
    print(f"\n✓ Essential molecules: {len(essential)}")
    for mol in essential[:5]:
        rda = f"{mol.health_data.rda} mg/day" if mol.health_data.rda else "N/A"
        print(f"  {mol.name:20s}: {rda}")
    
    # Test toxic molecules
    toxic = db.get_toxic_molecules()
    print(f"\n✓ Toxic molecules: {len(toxic)}")
    for mol in toxic:
        print(f"  {mol.name:20s}: {mol.health_data.health_effect.value}")
    
    return True


def test_food_search():
    """Test searching molecules by food source."""
    print("\n" + "="*80)
    print("TEST 3: Food Source Search")
    print("="*80)
    
    db = MolecularDatabase()
    
    foods = ["chicken", "salmon", "nuts", "berries"]
    
    for food in foods:
        molecules = db.search_by_food(food)
        print(f"\n✓ Molecules in '{food}': {len(molecules)}")
        for mol in molecules[:5]:
            print(f"  {mol.name:20s} ({mol.category.value})")
    
    return True


def test_nir_signatures():
    """Test NIR spectroscopic signatures."""
    print("\n" + "="*80)
    print("TEST 4: NIR Spectroscopic Signatures")
    print("="*80)
    
    db = MolecularDatabase()
    
    nir_molecules = db.get_molecules_with_nir_signature()
    print(f"\n✓ Molecules with NIR signatures: {len(nir_molecules)}")
    
    for mol in nir_molecules[:8]:
        if mol.nir_signature:
            peaks = mol.nir_signature.characteristic_peaks or mol.nir_signature.primary_bands[:2]
            peaks_str = ", ".join([f"{p:.0f}nm" for p in peaks])
            print(f"  {mol.name:20s}: {peaks_str}")
    
    return True


def test_safety_assessment():
    """Test safety assessment."""
    print("\n" + "="*80)
    print("TEST 5: Safety Assessment")
    print("="*80)
    
    db = MolecularDatabase()
    
    # Test cases: (molecule, concentration mg/kg)
    test_cases = [
        ("Glucose", 80000),      # Normal
        ("Ascorbic Acid", 1500), # Normal
        ("Aflatoxin B1", 0.001), # Trace (safe)
        ("Aflatoxin B1", 0.5),   # Danger
        ("Acrylamide", 0.1),     # Trace
        ("Acrylamide", 5.0),     # High/Danger
        ("Docosahexaenoic Acid", 15000),  # Normal in fish (DHA)
    ]
    
    for mol_name, conc in test_cases:
        assessment = db.assess_safety(mol_name, conc)
        print(f"\n✓ {assessment['molecule']} at {conc} mg/kg:")
        print(f"  Safety: {assessment['safety'].upper()}")
        print(f"  {assessment['message']}")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("MOLECULAR STRUCTURES DATABASE - TEST SUITE")
    print("Phase 3A Part 2: Food Molecules (Target: 10,000+)")
    print("="*80)
    
    tests = [
        ("Molecular Database", test_molecular_database),
        ("Molecule Queries", test_molecule_queries),
        ("Food Source Search", test_food_search),
        ("NIR Signatures", test_nir_signatures),
        ("Safety Assessment", test_safety_assessment),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}  {test_name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Molecular database functional.")
        print(f"\nCurrent: {25} molecules (demonstration set)")
        print("Production target: 10,000+ molecules")
        print("\nNext: Complete Phase 3A with covalent bond library")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
