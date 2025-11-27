"""
Knowledge Graph Engine for Health Impact Analysis
================================================

Replaces hardcoded dictionaries with dynamic AI-powered knowledge graph.
Uses graph database to store and query relationships between:
- Compounds → Molecules → Proteins
- Toxins → Health Effects
- Allergens → Cross-reactivity
- Nutrients → Health Benefits
- Health Conditions → Dietary Restrictions

Author: AI Nutrition Scanner Team
Date: November 2025
"""

import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

# For initial implementation, we'll use in-memory graph
# Production: migrate to Neo4j/TigerGraph
try:
    import networkx as nx
except ImportError:
    nx = None
    logging.warning("NetworkX not installed. Using simplified graph.")

logger = logging.getLogger(__name__)


# =============================================================================
# GRAPH NODE TYPES
# =============================================================================

class NodeType(Enum):
    """Types of nodes in knowledge graph."""
    COMPOUND = "compound"
    MOLECULE = "molecule"
    PROTEIN = "protein"
    TOXIN = "toxin"
    ALLERGEN = "allergen"
    NUTRIENT = "nutrient"
    HEALTH_CONDITION = "health_condition"
    DIETARY_RESTRICTION = "dietary_restriction"
    MEDICATION = "medication"
    INTERACTION = "interaction"


class RelationType(Enum):
    """Types of relationships in knowledge graph."""
    CONTAINS = "contains"
    CAUSES = "causes"
    CROSS_REACTS_WITH = "cross_reacts_with"
    PROVIDES = "provides"
    RESTRICTS = "restricts"
    RECOMMENDS = "recommends"
    INTERACTS_WITH = "interacts_with"
    METABOLIZES_TO = "metabolizes_to"
    INHIBITS = "inhibits"
    INCREASES_RISK_OF = "increases_risk_of"
    BENEFITS = "benefits"


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class GraphNode:
    """Node in knowledge graph."""
    id: str
    type: NodeType
    name: str
    properties: Dict[str, Any]


@dataclass
class GraphRelationship:
    """Relationship between nodes."""
    source_id: str
    target_id: str
    type: RelationType
    properties: Dict[str, Any]


@dataclass
class ToxicityKnowledge:
    """Toxicity information from KG."""
    compound_name: str
    ld50: Optional[float]
    safe_limit_mg_kg: Optional[float]
    hazard_class: str
    carcinogenic: bool
    regulatory_limits: Dict[str, float]
    sources: List[str]


@dataclass
class AllergenKnowledge:
    """Allergen information from KG."""
    allergen_name: str
    protein_family: str
    epitopes: List[str]
    cross_reactive_allergens: List[str]
    severity_distribution: Dict[str, float]
    affected_population_percent: float


@dataclass
class NutrientKnowledge:
    """Nutrient information from KG."""
    nutrient_name: str
    rda_adult_male: float
    rda_adult_female: float
    rda_pregnancy: float
    upper_limit: Optional[float]
    health_benefits: List[str]
    deficiency_symptoms: List[str]


@dataclass
class HealthConditionProfile:
    """Dynamic health condition dietary profile."""
    condition_name: str
    avoid: List[str]
    limit: Dict[str, float]
    increase: List[str]
    monitor: List[str]
    clinical_targets: Dict[str, Any]
    evidence_level: str
    sources: List[str]


# =============================================================================
# KNOWLEDGE GRAPH ENGINE
# =============================================================================

class KnowledgeGraphEngine:
    """
    AI-powered knowledge graph for health impact analysis.
    
    Provides dynamic queries instead of hardcoded dictionaries.
    In production, this will connect to Neo4j/TigerGraph.
    """
    
    def __init__(self, data_dir: Optional[str] = None, use_neo4j: bool = False):
        """
        Initialize knowledge graph.
        
        Args:
            data_dir: Directory containing knowledge graph data
            use_neo4j: Whether to use Neo4j backend (production)
        """
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "kg_data"
        self.use_neo4j = use_neo4j
        
        # Initialize graph backend
        if use_neo4j:
            self._init_neo4j()
        else:
            self._init_memory_graph()
        
        # Load knowledge
        self._load_knowledge_base()
        
        logger.info(f"KnowledgeGraphEngine initialized with {self.node_count()} nodes")
    
    def _init_neo4j(self):
        """Initialize Neo4j connection (production)."""
        try:
            from neo4j import GraphDatabase
            # TODO: Configure Neo4j connection
            self.driver = GraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j", "password")
            )
            logger.info("Connected to Neo4j")
        except ImportError:
            logger.error("Neo4j driver not installed. Falling back to memory graph.")
            self._init_memory_graph()
    
    def _init_memory_graph(self):
        """Initialize in-memory graph (development)."""
        if nx:
            self.graph = nx.MultiDiGraph()
        else:
            # Fallback: simple dict-based graph
            self.graph = {
                'nodes': {},
                'edges': []
            }
        logger.info("Using in-memory graph (development mode)")
    
    def _load_knowledge_base(self):
        """Load knowledge from data files."""
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        self._load_toxicology_db()
        self._load_allergen_db()
        self._load_nutrient_db()
        self._load_health_conditions_db()
        self._load_interactions_db()
    
    def _load_toxicology_db(self):
        """Load toxicology knowledge from sources."""
        # TODO: Integrate TOXNET, EPA IRIS, IARC data
        # For now, create expanded knowledge base
        
        toxins = [
            {
                "name": "aflatoxin_b1",
                "ld50": 0.5,
                "safe_limit": 0.02,
                "hazard": "hepatocarcinogen",
                "carcinogenic": True,
                "iarc_group": "1",
                "sources": ["IARC", "FDA"]
            },
            {
                "name": "acrylamide",
                "ld50": 150,
                "safe_limit": 1.0,
                "hazard": "neurotoxin",
                "carcinogenic": True,
                "iarc_group": "2A",
                "sources": ["IARC", "EPA"]
            },
            {
                "name": "lead",
                "ld50": 450,
                "safe_limit": 0.1,
                "hazard": "neurotoxin",
                "carcinogenic": False,
                "sources": ["WHO", "EPA"]
            },
            {
                "name": "mercury",
                "ld50": 1.4,
                "safe_limit": 0.5,
                "hazard": "neurotoxin",
                "carcinogenic": False,
                "sources": ["WHO", "FDA"]
            },
            {
                "name": "cadmium",
                "ld50": 225,
                "safe_limit": 0.05,
                "hazard": "nephrotoxin",
                "carcinogenic": True,
                "iarc_group": "1",
                "sources": ["IARC", "WHO"]
            },
            {
                "name": "arsenic",
                "ld50": 13,
                "safe_limit": 0.1,
                "hazard": "carcinogen",
                "carcinogenic": True,
                "iarc_group": "1",
                "sources": ["IARC", "EPA"]
            },
            {
                "name": "benzene",
                "ld50": 930,
                "safe_limit": 0.001,
                "hazard": "leukemogen",
                "carcinogenic": True,
                "iarc_group": "1",
                "sources": ["IARC", "OSHA"]
            }
        ]
        
        for toxin in toxins:
            self.add_node(
                id=f"toxin_{toxin['name']}",
                node_type=NodeType.TOXIN,
                name=toxin['name'],
                properties=toxin
            )
    
    def _load_allergen_db(self):
        """Load allergen and cross-reactivity knowledge."""
        # TODO: Integrate AllergenOnline, IUIS data
        
        allergens = [
            {
                "name": "peanut",
                "proteins": ["Ara h 1", "Ara h 2", "Ara h 3"],
                "cross_reactive": ["tree_nuts", "legumes", "soy"],
                "prevalence": 2.0,
                "severity": {"mild": 0.3, "moderate": 0.4, "severe": 0.3}
            },
            {
                "name": "tree_nuts",
                "proteins": ["Various 2S albumins"],
                "cross_reactive": ["peanut", "sesame"],
                "prevalence": 1.5,
                "severity": {"mild": 0.2, "moderate": 0.3, "severe": 0.5}
            },
            {
                "name": "milk",
                "proteins": ["casein", "beta-lactoglobulin", "alpha-lactalbumin"],
                "cross_reactive": ["beef", "goat_milk"],
                "prevalence": 2.5,
                "severity": {"mild": 0.5, "moderate": 0.4, "severe": 0.1}
            },
            {
                "name": "egg",
                "proteins": ["ovalbumin", "ovomucoid", "ovotransferrin"],
                "cross_reactive": ["chicken", "duck_egg"],
                "prevalence": 1.5,
                "severity": {"mild": 0.5, "moderate": 0.3, "severe": 0.2}
            },
            {
                "name": "wheat",
                "proteins": ["gluten", "gliadin", "glutenin"],
                "cross_reactive": ["other_grains", "grass_pollen"],
                "prevalence": 1.0,
                "severity": {"mild": 0.6, "moderate": 0.3, "severe": 0.1}
            },
            {
                "name": "soy",
                "proteins": ["Gly m 4", "Gly m 5", "Gly m 6"],
                "cross_reactive": ["peanut", "other_legumes"],
                "prevalence": 0.5,
                "severity": {"mild": 0.7, "moderate": 0.2, "severe": 0.1}
            },
            {
                "name": "shellfish",
                "proteins": ["tropomyosin"],
                "cross_reactive": ["crustaceans", "mollusks", "dust_mites"],
                "prevalence": 2.5,
                "severity": {"mild": 0.2, "moderate": 0.3, "severe": 0.5}
            },
            {
                "name": "fish",
                "proteins": ["parvalbumin"],
                "cross_reactive": ["other_fish_species"],
                "prevalence": 0.5,
                "severity": {"mild": 0.3, "moderate": 0.4, "severe": 0.3}
            }
        ]
        
        for allergen in allergens:
            node_id = f"allergen_{allergen['name']}"
            self.add_node(
                id=node_id,
                node_type=NodeType.ALLERGEN,
                name=allergen['name'],
                properties=allergen
            )
            
            # Add cross-reactivity relationships
            for cross in allergen['cross_reactive']:
                cross_id = f"allergen_{cross}"
                self.add_relationship(
                    source_id=node_id,
                    target_id=cross_id,
                    rel_type=RelationType.CROSS_REACTS_WITH,
                    properties={"strength": "moderate"}
                )
    
    def _load_nutrient_db(self):
        """Load nutrient RDA and health benefit knowledge."""
        # TODO: Integrate USDA FoodData Central, EFSA databases
        
        nutrients = [
            {
                "name": "vitamin_c",
                "rda_male": 90, "rda_female": 75, "rda_pregnancy": 85,
                "upper_limit": 2000,
                "benefits": ["immune_function", "antioxidant", "collagen_synthesis"],
                "deficiency": ["scurvy", "poor_wound_healing"]
            },
            {
                "name": "vitamin_d",
                "rda_male": 20, "rda_female": 20, "rda_pregnancy": 15,
                "upper_limit": 100,
                "benefits": ["bone_health", "immune_function", "calcium_absorption"],
                "deficiency": ["rickets", "osteomalacia", "poor_bone_density"]
            },
            {
                "name": "calcium",
                "rda_male": 1000, "rda_female": 1000, "rda_pregnancy": 1000,
                "upper_limit": 2500,
                "benefits": ["bone_health", "muscle_function", "nerve_transmission"],
                "deficiency": ["osteoporosis", "muscle_cramps"]
            },
            {
                "name": "iron",
                "rda_male": 8, "rda_female": 18, "rda_pregnancy": 27,
                "upper_limit": 45,
                "benefits": ["oxygen_transport", "energy_metabolism"],
                "deficiency": ["anemia", "fatigue", "poor_cognitive_function"]
            },
            {
                "name": "folate",
                "rda_male": 400, "rda_female": 400, "rda_pregnancy": 600,
                "upper_limit": 1000,
                "benefits": ["DNA_synthesis", "neural_tube_development", "red_blood_cell_formation"],
                "deficiency": ["neural_tube_defects", "megaloblastic_anemia"]
            },
            {
                "name": "omega_3",
                "rda_male": 1600, "rda_female": 1100, "rda_pregnancy": 1400,
                "upper_limit": None,
                "benefits": ["cardiovascular_health", "brain_function", "anti_inflammatory"],
                "deficiency": ["cardiovascular_disease", "cognitive_decline"]
            },
            {
                "name": "protein",
                "rda_male": 56000, "rda_female": 46000, "rda_pregnancy": 71000,
                "upper_limit": None,
                "benefits": ["muscle_maintenance", "enzyme_function", "immune_support"],
                "deficiency": ["muscle_wasting", "poor_wound_healing"]
            }
        ]
        
        for nutrient in nutrients:
            self.add_node(
                id=f"nutrient_{nutrient['name']}",
                node_type=NodeType.NUTRIENT,
                name=nutrient['name'],
                properties=nutrient
            )
    
    def _load_health_conditions_db(self):
        """Load health condition profiles dynamically."""
        # TODO: Integrate clinical guidelines (ADA, AHA, KDIGO)
        
        conditions = [
            {
                "name": "type_2_diabetes",
                "avoid": ["high_sugar", "refined_carbs", "trans_fats"],
                "limit": {"carbohydrates": 45, "saturated_fats": 7},
                "increase": ["fiber", "lean_protein", "omega_3"],
                "monitor": ["blood_glucose", "HbA1c"],
                "targets": {"glycemic_index": 55, "carb_per_meal_g": 45},
                "evidence": "Grade A",
                "sources": ["ADA_2025", "EASD"]
            },
            {
                "name": "hypertension",
                "avoid": ["high_sodium", "processed_foods", "cured_meats"],
                "limit": {"sodium": 1500, "alcohol": 1},
                "increase": ["potassium", "magnesium", "dash_diet"],
                "monitor": ["blood_pressure", "sodium_intake"],
                "targets": {"max_sodium_mg": 1500, "dash_compliance": 0.8},
                "evidence": "Grade A",
                "sources": ["AHA_2025", "ACC"]
            },
            {
                "name": "chronic_kidney_disease",
                "avoid": ["high_potassium", "high_phosphorus", "high_protein"],
                "limit": {"protein": 0.8, "potassium": 2000, "phosphorus": 1000},
                "increase": ["controlled_portions", "low_potassium_foods"],
                "monitor": ["GFR", "creatinine", "electrolytes"],
                "targets": {"max_protein_g_per_kg": 0.8, "max_potassium_mg": 2000},
                "evidence": "Grade A",
                "sources": ["KDIGO_2024", "NKF"]
            }
        ]
        
        for condition in conditions:
            self.add_node(
                id=f"condition_{condition['name']}",
                node_type=NodeType.HEALTH_CONDITION,
                name=condition['name'],
                properties=condition
            )
    
    def _load_interactions_db(self):
        """Load compound-compound and compound-drug interactions."""
        # TODO: Integrate DrugBank, interaction databases
        # Note: Drug interactions removed per user request, but keeping structure
        pass
    
    # =========================================================================
    # GRAPH OPERATIONS
    # =========================================================================
    
    def add_node(self, id: str, node_type: NodeType, name: str, properties: Dict):
        """Add node to graph."""
        if nx and hasattr(self, 'graph') and isinstance(self.graph, nx.MultiDiGraph):
            # NetworkX: merge name into properties to avoid keyword conflict
            all_props = {'type': node_type, 'node_name': name, **properties}
            self.graph.add_node(id, **all_props)
        else:
            if not hasattr(self.graph, '__getitem__'):
                return
            self.graph['nodes'][id] = {
                'type': node_type,
                'name': name,
                'properties': properties
            }
    
    def add_relationship(self, source_id: str, target_id: str, 
                        rel_type: RelationType, properties: Dict):
        """Add relationship to graph."""
        if nx and hasattr(self, 'graph') and isinstance(self.graph, nx.MultiDiGraph):
            self.graph.add_edge(source_id, target_id, type=rel_type, **properties)
        else:
            if not hasattr(self.graph, '__getitem__'):
                return
            self.graph['edges'].append({
                'source': source_id,
                'target': target_id,
                'type': rel_type,
                'properties': properties
            })
    
    def query_toxicity(self, compound_name: str) -> Optional[ToxicityKnowledge]:
        """Query toxicity information for compound."""
        node_id = f"toxin_{compound_name.lower()}"
        
        if nx and isinstance(self.graph, nx.MultiDiGraph):
            if node_id in self.graph:
                props = self.graph.nodes[node_id]
                return ToxicityKnowledge(
                    compound_name=compound_name,
                    ld50=props.get('ld50'),
                    safe_limit_mg_kg=props.get('safe_limit'),
                    hazard_class=props.get('hazard', 'unknown'),
                    carcinogenic=props.get('carcinogenic', False),
                    regulatory_limits={},
                    sources=props.get('sources', [])
                )
        else:
            if node_id in self.graph.get('nodes', {}):
                props = self.graph['nodes'][node_id]['properties']
                return ToxicityKnowledge(
                    compound_name=compound_name,
                    ld50=props.get('ld50'),
                    safe_limit_mg_kg=props.get('safe_limit'),
                    hazard_class=props.get('hazard', 'unknown'),
                    carcinogenic=props.get('carcinogenic', False),
                    regulatory_limits={},
                    sources=props.get('sources', [])
                )
        
        return None
    
    def query_allergen(self, allergen_name: str) -> Optional[AllergenKnowledge]:
        """Query allergen information."""
        node_id = f"allergen_{allergen_name.lower()}"
        
        if nx and isinstance(self.graph, nx.MultiDiGraph):
            if node_id in self.graph:
                props = self.graph.nodes[node_id]
                return AllergenKnowledge(
                    allergen_name=allergen_name,
                    protein_family=props.get('proteins', ['unknown'])[0],
                    epitopes=props.get('proteins', []),
                    cross_reactive_allergens=props.get('cross_reactive', []),
                    severity_distribution=props.get('severity', {}),
                    affected_population_percent=props.get('prevalence', 0)
                )
        else:
            if node_id in self.graph.get('nodes', {}):
                props = self.graph['nodes'][node_id]['properties']
                return AllergenKnowledge(
                    allergen_name=allergen_name,
                    protein_family=props.get('proteins', ['unknown'])[0],
                    epitopes=props.get('proteins', []),
                    cross_reactive_allergens=props.get('cross_reactive', []),
                    severity_distribution=props.get('severity', {}),
                    affected_population_percent=props.get('prevalence', 0)
                )
        
        return None
    
    def query_nutrient_rda(self, nutrient_name: str) -> Optional[NutrientKnowledge]:
        """Query nutrient RDA and benefits."""
        node_id = f"nutrient_{nutrient_name.lower()}"
        
        if nx and isinstance(self.graph, nx.MultiDiGraph):
            if node_id in self.graph:
                props = self.graph.nodes[node_id]
                return NutrientKnowledge(
                    nutrient_name=nutrient_name,
                    rda_adult_male=props.get('rda_male', 0),
                    rda_adult_female=props.get('rda_female', 0),
                    rda_pregnancy=props.get('rda_pregnancy', 0),
                    upper_limit=props.get('upper_limit'),
                    health_benefits=props.get('benefits', []),
                    deficiency_symptoms=props.get('deficiency', [])
                )
        else:
            if node_id in self.graph.get('nodes', {}):
                props = self.graph['nodes'][node_id]['properties']
                return NutrientKnowledge(
                    nutrient_name=nutrient_name,
                    rda_adult_male=props.get('rda_male', 0),
                    rda_adult_female=props.get('rda_female', 0),
                    rda_pregnancy=props.get('rda_pregnancy', 0),
                    upper_limit=props.get('upper_limit'),
                    health_benefits=props.get('benefits', []),
                    deficiency_symptoms=props.get('deficiency', [])
                )
        
        return None
    
    def query_health_condition(self, condition_name: str) -> Optional[HealthConditionProfile]:
        """Query health condition dietary profile."""
        node_id = f"condition_{condition_name.lower()}"
        
        if nx and isinstance(self.graph, nx.MultiDiGraph):
            if node_id in self.graph:
                props = self.graph.nodes[node_id]
                return HealthConditionProfile(
                    condition_name=condition_name,
                    avoid=props.get('avoid', []),
                    limit=props.get('limit', {}),
                    increase=props.get('increase', []),
                    monitor=props.get('monitor', []),
                    clinical_targets=props.get('targets', {}),
                    evidence_level=props.get('evidence', 'Unknown'),
                    sources=props.get('sources', [])
                )
        else:
            if node_id in self.graph.get('nodes', {}):
                props = self.graph['nodes'][node_id]['properties']
                return HealthConditionProfile(
                    condition_name=condition_name,
                    avoid=props.get('avoid', []),
                    limit=props.get('limit', {}),
                    increase=props.get('increase', []),
                    monitor=props.get('monitor', []),
                    clinical_targets=props.get('targets', {}),
                    evidence_level=props.get('evidence', 'Unknown'),
                    sources=props.get('sources', [])
                )
        
        return None
    
    def get_cross_reactive_allergens(self, allergen_name: str) -> List[str]:
        """Get allergens that cross-react with given allergen."""
        allergen_info = self.query_allergen(allergen_name)
        if allergen_info:
            return allergen_info.cross_reactive_allergens
        return []
    
    def node_count(self) -> int:
        """Get total number of nodes in graph."""
        if nx and isinstance(self.graph, nx.MultiDiGraph):
            return self.graph.number_of_nodes()
        else:
            return len(self.graph.get('nodes', {}))
    
    def export_to_neo4j(self):
        """Export graph to Neo4j (production deployment)."""
        # TODO: Implement Neo4j export
        pass
    
    def save_to_disk(self, filepath: str):
        """Save graph to disk for persistence."""
        if nx and isinstance(self.graph, nx.MultiDiGraph):
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.graph, f)
        else:
            with open(filepath, 'w') as f:
                json.dump(self.graph, f, indent=2)
    
    def load_from_disk(self, filepath: str):
        """Load graph from disk."""
        if nx and filepath.endswith('.pkl'):
            import pickle
            with open(filepath, 'rb') as f:
                self.graph = pickle.load(f)
        else:
            with open(filepath, 'r') as f:
                self.graph = json.load(f)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_kg_instance = None

def get_knowledge_graph() -> KnowledgeGraphEngine:
    """Get singleton knowledge graph instance."""
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = KnowledgeGraphEngine()
    return _kg_instance
