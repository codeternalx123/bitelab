"""
Layer B: The Molecular Layer (The "Chemistry") Implementation
===========================================================

This module implements the molecular layer of the Automated Flavor Intelligence Pipeline.
It handles chemical compound identification, molecular structure analysis, and compound-to-flavor
mapping using scientific databases and machine learning.

Key Features:
- Integration with PubChem, FlavorDB, and Wikidata for compound data
- Molecular structure analysis and chemical property calculation
- Volatile and non-volatile compound classification
- Chemical compound similarity analysis using molecular fingerprints
- Automated compound identification from ingredient names
"""

from typing import Dict, List, Optional, Tuple, Set, Union, Any
from dataclasses import dataclass, field
import numpy as np
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
from enum import Enum, auto
import math
from collections import defaultdict, Counter
import hashlib
import pickle
from urllib.parse import quote
import re

from .flavor_data_models import (
    ChemicalCompound, MolecularProfile, MolecularClass, 
    DataSource, SensoryProfile
)


class MolecularAnalysisMethod(Enum):
    """Methods for molecular analysis and compound identification"""
    DATABASE_LOOKUP = "database_lookup"
    STRUCTURE_SIMILARITY = "structure_similarity"
    FUNCTIONAL_GROUP_ANALYSIS = "functional_group_analysis"
    PROPERTY_PREDICTION = "property_prediction"
    ML_CLASSIFICATION = "ml_classification"
    HYBRID_APPROACH = "hybrid_approach"


class CompoundSource(Enum):
    """Sources for chemical compound data"""
    PUBCHEM = "pubchem"
    FLAVORDB = "flavordb"
    WIKIDATA = "wikidata"
    CHEMSPIDER = "chemspider"
    KEGG = "kegg"
    CHEBI = "chebi"
    FOODB = "foodb"
    VOLATILE_COMPOUNDS_IN_FOOD = "vcf"


class MolecularProperty(Enum):
    """Key molecular properties affecting flavor"""
    MOLECULAR_WEIGHT = "molecular_weight"
    BOILING_POINT = "boiling_point" 
    VAPOR_PRESSURE = "vapor_pressure"
    WATER_SOLUBILITY = "water_solubility"
    LOG_P = "log_p"  # Partition coefficient
    POLAR_SURFACE_AREA = "psa"
    HYDROGEN_DONORS = "h_donors"
    HYDROGEN_ACCEPTORS = "h_acceptors"
    ROTATABLE_BONDS = "rotatable_bonds"
    AROMATIC_RINGS = "aromatic_rings"


@dataclass
class MolecularFingerprint:
    """Molecular fingerprint for compound similarity analysis"""
    compound_id: str
    fingerprint_type: str  # ECFP, MACCS, etc.
    fingerprint_bits: np.ndarray
    bit_length: int
    
    def tanimoto_similarity(self, other: 'MolecularFingerprint') -> float:
        """Calculate Tanimoto similarity with another fingerprint"""
        if self.fingerprint_type != other.fingerprint_type:
            return 0.0
        
        intersection = np.logical_and(self.fingerprint_bits, other.fingerprint_bits).sum()
        union = np.logical_or(self.fingerprint_bits, other.fingerprint_bits).sum()
        
        return intersection / union if union > 0 else 0.0


@dataclass  
class CompoundIdentificationResult:
    """Result from compound identification process"""
    ingredient_name: str
    identified_compounds: List[ChemicalCompound]
    confidence_scores: Dict[str, float]
    
    # Analysis metadata
    identification_method: MolecularAnalysisMethod
    data_sources: List[CompoundSource]
    analysis_time_ms: int
    
    # Quality metrics
    completeness_score: float  # How complete is the compound identification
    reliability_score: float   # How reliable are the identifications


@dataclass
class ChemicalSimilarityResult:
    """Result from chemical similarity analysis"""
    compound_a: str
    compound_b: str
    similarity_score: float
    similarity_method: str
    
    # Detailed similarity breakdown
    structural_similarity: float
    property_similarity: float
    functional_group_similarity: float
    
    # Shared features
    common_functional_groups: List[str]
    property_differences: Dict[str, Tuple[float, float]]


@dataclass
class MolecularAnalysisConfig:
    """Configuration for molecular layer analysis"""
    
    # Data source preferences
    primary_sources: List[CompoundSource] = field(default_factory=lambda: [
        CompoundSource.PUBCHEM, CompoundSource.FLAVORDB, CompoundSource.FOODB
    ])
    
    # Analysis parameters
    min_compound_confidence: float = 0.6
    max_compounds_per_ingredient: int = 50
    similarity_threshold: float = 0.7
    
    # Cache and performance
    enable_caching: bool = True
    cache_duration_hours: int = 168  # 1 week
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 30
    
    # Quality control
    validate_molecular_structures: bool = True
    filter_unlikely_compounds: bool = True
    require_food_relevance: bool = True


class ChemicalDatabaseClient:
    """Client for accessing chemical databases (PubChem, FlavorDB, etc.)"""
    
    def __init__(self, config: MolecularAnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # HTTP session for API requests
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Cache for database queries
        self.compound_cache: Dict[str, ChemicalCompound] = {}
        self.search_cache: Dict[str, List[str]] = {}
        
        # API endpoints
        self.endpoints = {
            CompoundSource.PUBCHEM: "https://pubchem.ncbi.nlm.nih.gov/rest/pug",
            CompoundSource.FLAVORDB: "https://cosylab.iiitd.edu.in/flavordb/api",
            CompoundSource.WIKIDATA: "https://query.wikidata.org/sparql",
            CompoundSource.CHEMSPIDER: "https://www.chemspider.com/InChI.asmx",
            CompoundSource.FOODB: "https://foodb.ca/compounds"
        }
        
        # Rate limiting
        self.request_counts: Dict[CompoundSource, int] = defaultdict(int)
        self.last_request_time: Dict[CompoundSource, datetime] = {}
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def search_compounds_by_name(self, ingredient_name: str, 
                                     sources: List[CompoundSource] = None) -> List[str]:
        """Search for compound IDs by ingredient name across multiple databases"""
        sources = sources or self.config.primary_sources
        all_compound_ids = set()
        
        # Check cache first
        cache_key = f"search_{ingredient_name}_{'-'.join([s.value for s in sources])}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        # Search each database
        search_tasks = [
            self._search_single_source(ingredient_name, source) 
            for source in sources
        ]
        
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_compound_ids.update(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"Search failed: {result}")
        
        compound_ids = list(all_compound_ids)
        
        # Cache results
        if self.config.enable_caching:
            self.search_cache[cache_key] = compound_ids
        
        return compound_ids
    
    async def _search_single_source(self, ingredient_name: str, 
                                  source: CompoundSource) -> List[str]:
        """Search for compounds in a single database source"""
        if source == CompoundSource.PUBCHEM:
            return await self._search_pubchem(ingredient_name)
        elif source == CompoundSource.FLAVORDB:
            return await self._search_flavordb(ingredient_name)
        elif source == CompoundSource.WIKIDATA:
            return await self._search_wikidata(ingredient_name)
        elif source == CompoundSource.FOODB:
            return await self._search_foodb(ingredient_name)
        else:
            self.logger.warning(f"Source {source} not implemented")
            return []
    
    async def _search_pubchem(self, ingredient_name: str) -> List[str]:
        """Search PubChem for compounds related to ingredient"""
        try:
            # Search for compound by name
            search_url = f"{self.endpoints[CompoundSource.PUBCHEM]}/compound/name/{quote(ingredient_name)}/cids/JSON"
            
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
                        return [str(cid) for cid in data['IdentifierList']['CID'][:20]]  # Limit results
                
                # Alternative: search for related compounds
                alt_search_url = f"{self.endpoints[CompoundSource.PUBCHEM]}/compound/name/{quote(ingredient_name + ' compound')}/cids/JSON"
                async with self.session.get(alt_search_url) as alt_response:
                    if alt_response.status == 200:
                        alt_data = await alt_response.json()
                        if 'IdentifierList' in alt_data and 'CID' in alt_data['IdentifierList']:
                            return [str(cid) for cid in alt_data['IdentifierList']['CID'][:10]]
            
            return []
            
        except Exception as e:
            self.logger.error(f"PubChem search failed for {ingredient_name}: {e}")
            return []
    
    async def _search_flavordb(self, ingredient_name: str) -> List[str]:
        """Search FlavorDB for flavor compounds"""
        try:
            # FlavorDB has compounds organized by food category
            search_url = f"{self.endpoints[CompoundSource.FLAVORDB]}/search?food={quote(ingredient_name)}"
            
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    compound_ids = []
                    
                    if 'compounds' in data:
                        for compound in data['compounds'][:30]:  # Limit results
                            if 'pubchem_id' in compound:
                                compound_ids.append(str(compound['pubchem_id']))
                    
                    return compound_ids
            
            return []
            
        except Exception as e:
            self.logger.error(f"FlavorDB search failed for {ingredient_name}: {e}")
            return []
    
    async def _search_wikidata(self, ingredient_name: str) -> List[str]:
        """Search Wikidata for chemical compounds in food"""
        try:
            # SPARQL query for compounds found in the ingredient
            sparql_query = f"""
            SELECT DISTINCT ?compound ?pubchemId WHERE {{
                ?food rdfs:label "{ingredient_name}"@en .
                ?food wdt:P527 ?compound .
                ?compound wdt:P662 ?pubchemId .
            }}
            LIMIT 20
            """
            
            params = {
                'query': sparql_query,
                'format': 'json'
            }
            
            async with self.session.get(self.endpoints[CompoundSource.WIKIDATA], params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    compound_ids = []
                    
                    for binding in data.get('results', {}).get('bindings', []):
                        if 'pubchemId' in binding:
                            pubchem_id = binding['pubchemId']['value']
                            compound_ids.append(pubchem_id)
                    
                    return compound_ids
            
            return []
            
        except Exception as e:
            self.logger.error(f"Wikidata search failed for {ingredient_name}: {e}")
            return []
    
    async def _search_foodb(self, ingredient_name: str) -> List[str]:
        """Search FooDB for food compounds"""
        try:
            # Mock implementation - FooDB doesn't have a public API
            # In practice, this would use scraped data or a local FooDB dump
            food_compound_mappings = {
                'tomato': ['5280442', '5280445', '5280804'],  # Lycopene, beta-carotene, etc.
                'garlic': ['5862', '5986', '16127'],          # Allicin, alliin, etc.
                'vanilla': ['1183', '8095'],                   # Vanillin, vanillic acid
                'lemon': ['22311', '323', '6137'],             # Limonene, citric acid, citral
                'mint': ['16666', '1254'],                     # Menthol, menthone
                'ginger': ['442793', '442780'],                # Gingerol, shogaol
                'cinnamon': ['637511', '13706'],               # Cinnamaldehyde, coumarin
                'apple': ['7427', '6137', '7985']              # Malic acid, citral, etc.
            }
            
            ingredient_lower = ingredient_name.lower()
            
            # Find best match
            for food_name, compound_ids in food_compound_mappings.items():
                if food_name in ingredient_lower or ingredient_lower in food_name:
                    return compound_ids
            
            # Generic compounds for unmatched ingredients
            if 'fruit' in ingredient_lower:
                return ['22311', '323', '7427']  # Common fruit compounds
            elif 'herb' in ingredient_lower or 'spice' in ingredient_lower:
                return ['16666', '442793', '637511']  # Common herb/spice compounds
            
            return []
            
        except Exception as e:
            self.logger.error(f"FooDB search failed for {ingredient_name}: {e}")
            return []
    
    async def get_compound_details(self, compound_id: str, 
                                 source: CompoundSource = CompoundSource.PUBCHEM) -> Optional[ChemicalCompound]:
        """Get detailed information about a chemical compound"""
        
        # Check cache first
        cache_key = f"{source.value}_{compound_id}"
        if cache_key in self.compound_cache:
            return self.compound_cache[cache_key]
        
        try:
            if source == CompoundSource.PUBCHEM:
                compound = await self._get_pubchem_compound(compound_id)
            elif source == CompoundSource.FLAVORDB:
                compound = await self._get_flavordb_compound(compound_id)
            else:
                self.logger.warning(f"Compound details not implemented for {source}")
                return None
            
            # Cache result
            if compound and self.config.enable_caching:
                self.compound_cache[cache_key] = compound
            
            return compound
            
        except Exception as e:
            self.logger.error(f"Failed to get compound details for {compound_id}: {e}")
            return None
    
    async def _get_pubchem_compound(self, cid: str) -> Optional[ChemicalCompound]:
        """Get compound details from PubChem"""
        try:
            # Get compound properties
            props_url = f"{self.endpoints[CompoundSource.PUBCHEM]}/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,InChI/JSON"
            
            async with self.session.get(props_url) as response:
                if response.status != 200:
                    return None
                
                props_data = await response.json()
                if 'PropertyTable' not in props_data or 'Properties' not in props_data['PropertyTable']:
                    return None
                
                props = props_data['PropertyTable']['Properties'][0]
                
                # Get synonyms for name
                synonyms_url = f"{self.endpoints[CompoundSource.PUBCHEM]}/compound/cid/{cid}/synonyms/JSON"
                name = f"Compound_{cid}"
                
                try:
                    async with self.session.get(synonyms_url) as syn_response:
                        if syn_response.status == 200:
                            syn_data = await syn_response.json()
                            if 'InformationList' in syn_data and syn_data['InformationList']['Information']:
                                synonyms = syn_data['InformationList']['Information'][0].get('Synonym', [])
                                if synonyms:
                                    name = synonyms[0]  # Use first synonym as primary name
                except:
                    pass  # Use default name if synonyms fail
                
                # Create compound object
                compound = ChemicalCompound(
                    compound_id=cid,
                    name=name,
                    pubchem_cid=cid,
                    molecular_formula=props.get('MolecularFormula', ''),
                    molecular_weight=float(props.get('MolecularWeight', 0)),
                    smiles=props.get('CanonicalSMILES', ''),
                    inchi=props.get('InChI', ''),
                    data_source=DataSource.PUBCHEM,
                    confidence_score=0.9,
                    last_updated=datetime.now()
                )
                
                # Classify compound based on molecular weight and structure
                compound.compound_class = self._classify_compound(compound)
                
                return compound
                
        except Exception as e:
            self.logger.error(f"PubChem compound fetch failed for CID {cid}: {e}")
            return None
    
    async def _get_flavordb_compound(self, compound_id: str) -> Optional[ChemicalCompound]:
        """Get compound details from FlavorDB"""
        try:
            # FlavorDB compound details
            details_url = f"{self.endpoints[CompoundSource.FLAVORDB]}/compound/{compound_id}"
            
            async with self.session.get(details_url) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                
                compound = ChemicalCompound(
                    compound_id=compound_id,
                    name=data.get('name', f'FlavorDB_{compound_id}'),
                    pubchem_cid=data.get('pubchem_id'),
                    molecular_formula=data.get('molecular_formula', ''),
                    molecular_weight=float(data.get('molecular_weight', 0)),
                    smiles=data.get('smiles', ''),
                    odor_descriptors=data.get('odor_descriptors', []),
                    taste_descriptors=data.get('taste_descriptors', []),
                    natural_occurrence=data.get('foods', []),
                    data_source=DataSource.FLAVORDB,
                    confidence_score=0.85,
                    last_updated=datetime.now()
                )
                
                compound.compound_class = self._classify_compound(compound)
                
                return compound
                
        except Exception as e:
            self.logger.error(f"FlavorDB compound fetch failed for ID {compound_id}: {e}")
            return None
    
    def _classify_compound(self, compound: ChemicalCompound) -> MolecularClass:
        """Classify compound based on molecular structure and properties"""
        
        # Check molecular formula patterns
        formula = compound.molecular_formula.upper()
        name = compound.name.lower()
        
        # Terpenes - contain multiple of 5 carbons, often C10, C15, C20
        if re.search(r'C(10|15|20|5)H', formula):
            return MolecularClass.TERPENE
        
        # Esters - contain COO pattern or name contains "ate"
        if 'COO' in formula or any(term in name for term in ['acetate', 'butyrate', 'propionate']):
            return MolecularClass.ESTER
        
        # Aldehydes - often end in "al" or contain CHO
        if 'CHO' in formula or name.endswith('al') or 'aldehyde' in name:
            return MolecularClass.ALDEHYDE
        
        # Ketones - often end in "one" or contain CO
        if name.endswith('one') or 'ketone' in name:
            return MolecularClass.KETONE
        
        # Alcohols - contain OH or end in "ol"
        if 'OH' in formula or name.endswith('ol') or 'alcohol' in name:
            return MolecularClass.ALCOHOL
        
        # Acids - contain COOH or name contains "acid"
        if 'COOH' in formula or 'acid' in name:
            return MolecularClass.ACID
        
        # Phenols - aromatic compounds with OH
        if any(term in name for term in ['phenol', 'vanillin', 'eugenol']):
            return MolecularClass.PHENOL
        
        # Default classification
        return MolecularClass.TERPENE


class MolecularAnalyzer:
    """Advanced analyzer for molecular layer processing"""
    
    def __init__(self, config: MolecularAnalysisConfig = None):
        self.config = config or MolecularAnalysisConfig()
        self.logger = logging.getLogger(__name__)
        
        # Database client
        self.db_client = ChemicalDatabaseClient(self.config)
        
        # Molecular property calculators
        self.property_calculators: Dict[MolecularProperty, Any] = {}
        
        # Fingerprint cache
        self.fingerprint_cache: Dict[str, MolecularFingerprint] = {}
        
        # Analysis statistics
        self.analysis_stats = {
            'total_analyses': 0,
            'successful_identifications': 0,
            'average_compounds_per_ingredient': 0.0,
            'cache_hit_rate': 0.0
        }
    
    async def analyze_ingredient_molecules(self, ingredient_name: str, 
                                        method: MolecularAnalysisMethod = None) -> CompoundIdentificationResult:
        """
        Main entry point for molecular analysis of an ingredient
        """
        method = method or MolecularAnalysisMethod.HYBRID_APPROACH
        start_time = datetime.now()
        
        self.logger.info(f"Starting molecular analysis of '{ingredient_name}' using {method.value}")
        
        try:
            async with self.db_client:
                if method == MolecularAnalysisMethod.DATABASE_LOOKUP:
                    result = await self._analyze_by_database_lookup(ingredient_name)
                elif method == MolecularAnalysisMethod.HYBRID_APPROACH:
                    result = await self._analyze_hybrid_approach(ingredient_name)
                else:
                    result = await self._analyze_by_database_lookup(ingredient_name)
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds() * 1000
            result.analysis_time_ms = int(analysis_time)
            
            # Update statistics
            self.analysis_stats['total_analyses'] += 1
            if result.identified_compounds:
                self.analysis_stats['successful_identifications'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Molecular analysis failed for {ingredient_name}: {e}")
            return CompoundIdentificationResult(
                ingredient_name=ingredient_name,
                identified_compounds=[],
                confidence_scores={},
                identification_method=method,
                data_sources=[],
                analysis_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                completeness_score=0.0,
                reliability_score=0.0
            )
    
    async def _analyze_by_database_lookup(self, ingredient_name: str) -> CompoundIdentificationResult:
        """Analyze ingredient by looking up compounds in databases"""
        
        # Search for compound IDs
        compound_ids = await self.db_client.search_compounds_by_name(ingredient_name)
        
        if not compound_ids:
            return CompoundIdentificationResult(
                ingredient_name=ingredient_name,
                identified_compounds=[],
                confidence_scores={},
                identification_method=MolecularAnalysisMethod.DATABASE_LOOKUP,
                data_sources=self.config.primary_sources,
                analysis_time_ms=0,
                completeness_score=0.0,
                reliability_score=0.0
            )
        
        # Get detailed compound information
        compound_tasks = [
            self.db_client.get_compound_details(cid) 
            for cid in compound_ids[:self.config.max_compounds_per_ingredient]
        ]
        
        compounds_data = await asyncio.gather(*compound_tasks, return_exceptions=True)
        
        # Filter successful results
        identified_compounds = []
        confidence_scores = {}
        
        for compound_data in compounds_data:
            if isinstance(compound_data, ChemicalCompound):
                # Validate compound relevance to food/flavor
                if self._is_food_relevant_compound(compound_data):
                    identified_compounds.append(compound_data)
                    confidence_scores[compound_data.compound_id] = compound_data.confidence_score
        
        # Calculate quality metrics
        completeness_score = min(len(identified_compounds) / 10.0, 1.0)  # Expect ~10 compounds
        reliability_score = np.mean(list(confidence_scores.values())) if confidence_scores else 0.0
        
        return CompoundIdentificationResult(
            ingredient_name=ingredient_name,
            identified_compounds=identified_compounds,
            confidence_scores=confidence_scores,
            identification_method=MolecularAnalysisMethod.DATABASE_LOOKUP,
            data_sources=self.config.primary_sources,
            analysis_time_ms=0,  # Will be set by caller
            completeness_score=completeness_score,
            reliability_score=reliability_score
        )
    
    async def _analyze_hybrid_approach(self, ingredient_name: str) -> CompoundIdentificationResult:
        """
        Hybrid analysis combining multiple methods for best results
        """
        # Start with database lookup
        db_result = await self._analyze_by_database_lookup(ingredient_name)
        
        # Enhance with functional group analysis
        enhanced_compounds = []
        
        for compound in db_result.identified_compounds:
            # Add functional group information
            compound.functional_groups = self._identify_functional_groups(compound)
            
            # Calculate additional properties
            compound = self._calculate_molecular_properties(compound)
            
            # Classify volatility
            compound = self._classify_volatility(compound)
            
            enhanced_compounds.append(compound)
        
        # Update result with enhanced data
        db_result.identified_compounds = enhanced_compounds
        db_result.identification_method = MolecularAnalysisMethod.HYBRID_APPROACH
        
        # Recalculate quality scores
        db_result.completeness_score = self._calculate_completeness_score(enhanced_compounds)
        db_result.reliability_score = self._calculate_reliability_score(enhanced_compounds)
        
        return db_result
    
    def _is_food_relevant_compound(self, compound: ChemicalCompound) -> bool:
        """Check if compound is relevant to food and flavor"""
        
        # Check molecular weight (food compounds typically < 1000 Da)
        if compound.molecular_weight > 1000:
            return False
        
        # Check for food-related keywords in name
        food_keywords = [
            'flavor', 'aroma', 'taste', 'volatile', 'essential', 'natural',
            'extract', 'oil', 'acid', 'alcohol', 'ester', 'aldehyde',
            'terpene', 'phenol', 'vanillin', 'limonene', 'menthol'
        ]
        
        name_lower = compound.name.lower()
        if any(keyword in name_lower for keyword in food_keywords):
            return True
        
        # Check for natural occurrence in foods
        if compound.natural_occurrence:
            return True
        
        # Check odor or taste descriptors
        if compound.odor_descriptors or compound.taste_descriptors:
            return True
        
        # Check molecular class
        food_relevant_classes = [
            MolecularClass.TERPENE, MolecularClass.ESTER, MolecularClass.ALDEHYDE,
            MolecularClass.KETONE, MolecularClass.ALCOHOL, MolecularClass.PHENOL,
            MolecularClass.ACID
        ]
        
        if compound.compound_class in food_relevant_classes:
            return True
        
        return False
    
    def _identify_functional_groups(self, compound: ChemicalCompound) -> List[str]:
        """Identify functional groups from molecular structure"""
        functional_groups = []
        
        if not compound.smiles:
            return functional_groups
        
        smiles = compound.smiles.upper()
        
        # Common functional group patterns in SMILES
        patterns = {
            'hydroxyl': r'O[H]?',
            'carbonyl': r'C=O',
            'carboxyl': r'C\(=O\)O',
            'ester': r'C\(=O\)O[^H]',
            'ether': r'O[^H=]',
            'aldehyde': r'C=O[H]?$',
            'ketone': r'C\(=O\)[^OH]',
            'aromatic': r'c',  # lowercase c indicates aromatic carbon
            'double_bond': r'=C',
            'triple_bond': r'#C'
        }
        
        for group_name, pattern in patterns.items():
            if re.search(pattern, smiles):
                functional_groups.append(group_name)
        
        return functional_groups
    
    def _calculate_molecular_properties(self, compound: ChemicalCompound) -> ChemicalCompound:
        """Calculate additional molecular properties"""
        
        # Estimate boiling point from molecular weight (rough approximation)
        if compound.molecular_weight > 0:
            # Trouton's rule approximation
            compound.boiling_point = 80 + (compound.molecular_weight * 0.5)
        
        # Estimate vapor pressure (very rough approximation)
        if compound.boiling_point:
            # Antoine equation approximation at room temperature
            compound.vapor_pressure = 10 ** (8 - compound.boiling_point / 50)
        
        # Estimate water solubility based on functional groups
        compound.solubility_water = self._estimate_water_solubility(compound)
        
        return compound
    
    def _estimate_water_solubility(self, compound: ChemicalCompound) -> float:
        """Estimate water solubility from molecular structure"""
        
        # Base solubility from molecular weight (smaller = more soluble)
        base_solubility = max(0, 1000 - compound.molecular_weight)
        
        # Adjust based on functional groups
        if 'hydroxyl' in compound.functional_groups:
            base_solubility *= 5  # OH groups increase solubility
        
        if 'carboxyl' in compound.functional_groups:
            base_solubility *= 10  # COOH groups very soluble
        
        if 'aromatic' in compound.functional_groups:
            base_solubility *= 0.1  # Aromatic rings decrease solubility
        
        if len(compound.functional_groups) == 0:
            base_solubility *= 0.01  # Non-polar compounds less soluble
        
        return min(base_solubility, 100000)  # Cap at 100 g/L
    
    def _classify_volatility(self, compound: ChemicalCompound) -> ChemicalCompound:
        """Classify compound as volatile or non-volatile"""
        
        # Compounds with high vapor pressure are volatile
        if compound.vapor_pressure and compound.vapor_pressure > 1.0:  # > 1 mmHg at 25°C
            compound.heat_stable = False  # Volatile compounds may evaporate when heated
        
        # Low molecular weight compounds tend to be volatile
        if compound.molecular_weight < 200:
            if compound.vapor_pressure is None or compound.vapor_pressure > 0.1:
                compound.heat_stable = False
        
        return compound
    
    def _calculate_completeness_score(self, compounds: List[ChemicalCompound]) -> float:
        """Calculate how complete the molecular profile is"""
        if not compounds:
            return 0.0
        
        # Factors for completeness
        num_compounds = len(compounds)
        has_volatile = any(not c.heat_stable for c in compounds)
        has_structural_data = any(c.smiles for c in compounds)
        has_properties = any(c.boiling_point for c in compounds)
        
        # Scoring
        compound_score = min(num_compounds / 15.0, 1.0)  # Expect ~15 compounds
        volatile_score = 1.0 if has_volatile else 0.5
        structure_score = 1.0 if has_structural_data else 0.3
        property_score = 1.0 if has_properties else 0.5
        
        return (compound_score + volatile_score + structure_score + property_score) / 4.0
    
    def _calculate_reliability_score(self, compounds: List[ChemicalCompound]) -> float:
        """Calculate reliability of molecular identification"""
        if not compounds:
            return 0.0
        
        # Average confidence scores
        avg_confidence = np.mean([c.confidence_score for c in compounds])
        
        # Data source reliability
        source_reliability = {
            DataSource.PUBCHEM: 0.9,
            DataSource.FLAVORDB: 0.85,
            DataSource.WIKIDATA: 0.7
        }
        
        avg_source_reliability = np.mean([
            source_reliability.get(c.data_source, 0.5) for c in compounds
        ])
        
        return (avg_confidence + avg_source_reliability) / 2.0
    
    def create_molecular_profile(self, ingredient_id: str, 
                               compound_result: CompoundIdentificationResult) -> MolecularProfile:
        """Create a complete molecular profile from compound identification result"""
        
        profile = MolecularProfile(ingredient_id=ingredient_id)
        
        # Add all identified compounds
        for compound in compound_result.identified_compounds:
            concentration = self._estimate_compound_concentration(compound)
            profile.add_compound(compound, concentration)
        
        # Calculate synergistic pairs
        profile.synergistic_pairs = self._identify_synergistic_compounds(compound_result.identified_compounds)
        
        # Set quality metrics
        profile.completeness_score = compound_result.completeness_score
        profile.confidence_score = compound_result.reliability_score
        profile.last_analysis = datetime.now()
        
        return profile
    
    def _estimate_compound_concentration(self, compound: ChemicalCompound) -> float:
        """Estimate relative concentration of compound in ingredient"""
        
        # Base concentration from compound class
        class_concentrations = {
            MolecularClass.TERPENE: 0.1,      # Often major volatile components
            MolecularClass.ESTER: 0.05,       # Typically minor components
            MolecularClass.ALDEHYDE: 0.02,    # Often trace but potent
            MolecularClass.KETONE: 0.03,
            MolecularClass.ALCOHOL: 0.08,
            MolecularClass.ACID: 0.15,        # Can be major components
            MolecularClass.PHENOL: 0.01       # Usually trace but important
        }
        
        base_concentration = class_concentrations.get(compound.compound_class, 0.01)
        
        # Adjust based on odor/taste descriptors (more descriptors = likely more important)
        if compound.odor_descriptors or compound.taste_descriptors:
            descriptor_count = len(compound.odor_descriptors) + len(compound.taste_descriptors)
            base_concentration *= (1 + descriptor_count * 0.1)
        
        # Adjust based on natural occurrence (more foods = likely higher concentration)
        if compound.natural_occurrence:
            occurrence_factor = min(len(compound.natural_occurrence) / 10.0, 2.0)
            base_concentration *= occurrence_factor
        
        return min(base_concentration, 1.0)  # Cap at 100%
    
    def _identify_synergistic_compounds(self, compounds: List[ChemicalCompound]) -> List[Tuple[str, str, float]]:
        """Identify synergistic compound pairs"""
        synergistic_pairs = []
        
        # Known synergistic combinations
        synergy_patterns = [
            (['aldehyde', 'ester'], 0.8),       # Aldehydes + esters enhance fruitiness
            (['terpene', 'alcohol'], 0.7),      # Terpenes + alcohols enhance floral notes
            (['acid', 'ester'], 0.6),           # Acids + esters balance taste
            (['phenol', 'aldehyde'], 0.9),      # Phenols + aldehydes create complex flavors
            (['ketone', 'alcohol'], 0.5)        # Ketones + alcohols modify intensity
        ]
        
        # Check all compound pairs for synergistic patterns
        for i, compound_a in enumerate(compounds):
            for j, compound_b in enumerate(compounds[i+1:], i+1):
                
                groups_a = set(compound_a.functional_groups)
                groups_b = set(compound_b.functional_groups)
                
                for pattern, synergy_score in synergy_patterns:
                    if any(p in groups_a for p in pattern) and any(p in groups_b for p in pattern):
                        synergistic_pairs.append((
                            compound_a.compound_id, 
                            compound_b.compound_id, 
                            synergy_score
                        ))
                        break
        
        return synergistic_pairs
    
    def calculate_molecular_similarity(self, profile_a: MolecularProfile, 
                                     profile_b: MolecularProfile) -> ChemicalSimilarityResult:
        """Calculate similarity between two molecular profiles"""
        
        compounds_a = set(profile_a.compounds.keys())
        compounds_b = set(profile_b.compounds.keys())
        
        # Structural similarity (Tanimoto coefficient)
        intersection = len(compounds_a.intersection(compounds_b))
        union = len(compounds_a.union(compounds_b))
        structural_similarity = intersection / union if union > 0 else 0.0
        
        # Property similarity (based on molecular classes)
        classes_a = Counter([c.compound_class for c in profile_a.compounds.values()])
        classes_b = Counter([c.compound_class for c in profile_b.compounds.values()])
        
        all_classes = set(classes_a.keys()).union(set(classes_b.keys()))
        property_similarity = 0.0
        
        if all_classes:
            class_similarities = []
            for cls in all_classes:
                count_a = classes_a.get(cls, 0)
                count_b = classes_b.get(cls, 0)
                max_count = max(count_a, count_b)
                min_count = min(count_a, count_b)
                class_similarities.append(min_count / max_count if max_count > 0 else 0.0)
            
            property_similarity = np.mean(class_similarities)
        
        # Functional group similarity
        all_groups_a = set()
        all_groups_b = set()
        
        for compound in profile_a.compounds.values():
            all_groups_a.update(compound.functional_groups)
        
        for compound in profile_b.compounds.values():
            all_groups_b.update(compound.functional_groups)
        
        group_intersection = len(all_groups_a.intersection(all_groups_b))
        group_union = len(all_groups_a.union(all_groups_b))
        functional_group_similarity = group_intersection / group_union if group_union > 0 else 0.0
        
        # Overall similarity
        overall_similarity = (
            structural_similarity * 0.4 +
            property_similarity * 0.3 +
            functional_group_similarity * 0.3
        )
        
        return ChemicalSimilarityResult(
            compound_a=profile_a.ingredient_id,
            compound_b=profile_b.ingredient_id,
            similarity_score=overall_similarity,
            similarity_method="molecular_profile",
            structural_similarity=structural_similarity,
            property_similarity=property_similarity,
            functional_group_similarity=functional_group_similarity,
            common_functional_groups=list(all_groups_a.intersection(all_groups_b)),
            property_differences={}  # Could be calculated for specific properties
        )
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analysis statistics"""
        cache_size = len(self.db_client.compound_cache) + len(self.fingerprint_cache)
        
        return {
            **self.analysis_stats,
            'cache_size': cache_size,
            'supported_sources': [source.value for source in self.config.primary_sources],
            'max_compounds_per_ingredient': self.config.max_compounds_per_ingredient,
            'min_confidence_threshold': self.config.min_compound_confidence
        }


# Factory functions for common use cases

def create_molecular_analyzer(sources: List[CompoundSource] = None) -> MolecularAnalyzer:
    """Create molecular analyzer with specified data sources"""
    config = MolecularAnalysisConfig()
    if sources:
        config.primary_sources = sources
    return MolecularAnalyzer(config)


async def quick_molecular_analysis(ingredient_name: str) -> CompoundIdentificationResult:
    """Quick molecular analysis for single ingredient"""
    analyzer = create_molecular_analyzer()
    return await analyzer.analyze_ingredient_molecules(ingredient_name)


def calculate_compound_similarity(compound_a: ChemicalCompound, 
                                compound_b: ChemicalCompound) -> float:
    """Calculate similarity between two chemical compounds"""
    
    # Molecular weight similarity
    if compound_a.molecular_weight > 0 and compound_b.molecular_weight > 0:
        mw_diff = abs(compound_a.molecular_weight - compound_b.molecular_weight)
        mw_similarity = 1.0 - min(mw_diff / 500.0, 1.0)  # Normalize by 500 Da
    else:
        mw_similarity = 0.0
    
    # Functional group similarity
    groups_a = set(compound_a.functional_groups)
    groups_b = set(compound_b.functional_groups)
    
    if groups_a or groups_b:
        group_intersection = len(groups_a.intersection(groups_b))
        group_union = len(groups_a.union(groups_b))
        group_similarity = group_intersection / group_union
    else:
        group_similarity = 0.0
    
    # Class similarity
    class_similarity = 1.0 if compound_a.compound_class == compound_b.compound_class else 0.0
    
    # Combined similarity
    return (mw_similarity * 0.3 + group_similarity * 0.4 + class_similarity * 0.3)


# Export key classes and functions
__all__ = [
    'MolecularAnalysisMethod', 'CompoundSource', 'MolecularProperty',
    'MolecularFingerprint', 'CompoundIdentificationResult', 'ChemicalSimilarityResult',
    'MolecularAnalysisConfig', 'ChemicalDatabaseClient', 'MolecularAnalyzer',
    'create_molecular_analyzer', 'quick_molecular_analysis', 'calculate_compound_similarity'
]