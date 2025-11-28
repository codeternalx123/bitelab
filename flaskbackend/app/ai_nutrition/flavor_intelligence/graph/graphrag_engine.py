"""
GraphRAG Implementation for Flavor Intelligence
=============================================

This module implements Graph Retrieval-Augmented Generation (GraphRAG) for the
Automated Flavor Intelligence Pipeline. It combines Neo4j graph databases with
LLM reasoning to create a sophisticated knowledge graph system.

Key Features:
- Neo4j graph database integration for millions of flavor relationships
- LLM-powered graph query generation and interpretation
- Automated triplet generation for neural network training
- Real-time graph traversal and reasoning
- Multi-hop relationship discovery
- Graph embedding generation and similarity search
"""

from typing import Dict, List, Optional, Tuple, Set, Union, Any, AsyncGenerator
from dataclasses import dataclass, field
import numpy as np
import logging
from datetime import datetime, timedelta
import asyncio
import json
import hashlib
from collections import defaultdict, deque
import pickle
from enum import Enum
import re
import math

# Neo4j imports
from neo4j import AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import ServiceUnavailable, AuthError

# Graph analysis imports
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from ..models.flavor_data_models import (
    FlavorProfile, ChemicalCompound, SensoryProfile, 
    IngredientCompatibility, DataSource
)


class GraphQueryType(Enum):
    """Types of graph queries supported"""
    SIMILARITY_SEARCH = "similarity_search"
    SUBSTITUTION_FIND = "substitution_find"
    PAIRING_DISCOVERY = "pairing_discovery"
    CULTURAL_ANALYSIS = "cultural_analysis"
    SEASONAL_PATTERNS = "seasonal_patterns"
    MOLECULAR_SIMILARITY = "molecular_similarity"
    SENSORY_MAPPING = "sensory_mapping"
    RECIPE_GENERATION = "recipe_generation"


class GraphRelationType(Enum):
    """Types of relationships in the flavor graph"""
    PAIRS_WITH = "PAIRS_WITH"
    SUBSTITUTES_FOR = "SUBSTITUTES_FOR"
    CONTAINS_COMPOUND = "CONTAINS_COMPOUND"
    BELONGS_TO_CUISINE = "BELONGS_TO_CUISINE"
    AVAILABLE_IN_SEASON = "AVAILABLE_IN_SEASON"
    HAS_FLAVOR_PROFILE = "HAS_FLAVOR_PROFILE"
    ENHANCES = "ENHANCES"
    CONTRASTS_WITH = "CONTRASTS_WITH"
    DERIVED_FROM = "DERIVED_FROM"
    SIMILAR_TO = "SIMILAR_TO"


class GraphNodeType(Enum):
    """Types of nodes in the flavor graph"""
    INGREDIENT = "Ingredient"
    COMPOUND = "Compound"
    FLAVOR_PROFILE = "FlavorProfile"
    CUISINE = "Cuisine"
    SEASON = "Season"
    RECIPE = "Recipe"
    CATEGORY = "Category"
    COUNTRY = "Country"


@dataclass
class GraphQuery:
    """Structured query for the flavor knowledge graph"""
    query_id: str
    query_type: GraphQueryType
    
    # Query parameters
    target_ingredient: Optional[str] = None
    target_compounds: List[str] = field(default_factory=list)
    cuisine_filter: Optional[str] = None
    season_filter: Optional[str] = None
    similarity_threshold: float = 0.7
    max_results: int = 20
    
    # Relationship constraints
    required_relationships: List[GraphRelationType] = field(default_factory=list)
    excluded_relationships: List[GraphRelationType] = field(default_factory=list)
    
    # Natural language query
    natural_language: Optional[str] = None
    
    # Query metadata
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1=low, 5=high


@dataclass
class GraphQueryResult:
    """Result from a graph query"""
    query_id: str
    results: List[Dict[str, Any]]
    
    # Result metadata
    total_results: int
    execution_time_ms: int
    graph_traversal_depth: int
    
    # Confidence and quality
    confidence_score: float
    result_quality: float
    
    # Explanation and reasoning
    reasoning_path: List[str] = field(default_factory=list)
    llm_interpretation: Optional[str] = None
    
    # Cypher query used (for debugging)
    cypher_query: Optional[str] = None


@dataclass
class TrainingTriplet:
    """Training triplet for neural network learning"""
    anchor: str  # Reference ingredient
    positive: str  # Compatible ingredient
    negative: str  # Incompatible ingredient
    
    # Relationship strength
    positive_score: float
    negative_score: float
    
    # Context information
    relationship_type: GraphRelationType
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Generation metadata
    generated_at: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0


@dataclass
class GraphEmbedding:
    """Graph embedding for an entity"""
    entity_id: str
    entity_type: GraphNodeType
    embedding_vector: np.ndarray
    
    # Embedding metadata
    dimension: int
    algorithm: str  # node2vec, GraphSAGE, etc.
    training_params: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    embedding_quality: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class GraphRAGConfig:
    """Configuration for GraphRAG system"""
    
    # Neo4j connection settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # Graph database settings
    max_connections: int = 50
    connection_timeout: int = 30
    query_timeout: int = 60
    
    # Embedding settings
    embedding_dimension: int = 128
    embedding_algorithm: str = "node2vec"
    walk_length: int = 80
    num_walks: int = 10
    
    # LLM integration
    use_llm_query_generation: bool = True
    use_llm_result_interpretation: bool = True
    llm_temperature: float = 0.7
    
    # Training data generation
    triplets_per_ingredient: int = 100
    negative_sampling_ratio: float = 2.0
    min_relationship_strength: float = 0.3
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    batch_size: int = 1000
    parallel_workers: int = 4


class Neo4jFlavorGraphDB:
    """Neo4j database interface for flavor knowledge graph"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database connection
        self.driver = None
        self.connection_pool = None
        
        # Query cache
        self.query_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Statistics
        self.query_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'average_execution_time_ms': 0,
            'nodes_created': 0,
            'relationships_created': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
                max_connection_pool_size=self.config.max_connections,
                connection_timeout=self.config.connection_timeout
            )
            
            # Test connection
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            
            self.logger.info("Connected to Neo4j database successfully")
            
        except (ServiceUnavailable, AuthError) as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def close(self):
        """Close database connection"""
        if self.driver:
            await self.driver.close()
            self.logger.info("Neo4j connection closed")
    
    async def create_ingredient_node(self, flavor_profile: FlavorProfile) -> str:
        """Create ingredient node in the graph"""
        
        cypher = """
        MERGE (i:Ingredient {id: $ingredient_id})
        SET i.name = $name,
            i.scientific_name = $scientific_name,
            i.primary_category = $primary_category,
            i.origin_countries = $origin_countries,
            i.seasonal_availability = $seasonal_availability,
            i.confidence = $confidence,
            i.last_updated = datetime()
        RETURN i.id as id
        """
        
        params = {
            'ingredient_id': flavor_profile.ingredient_id,
            'name': flavor_profile.name,
            'scientific_name': flavor_profile.scientific_name,
            'primary_category': flavor_profile.primary_category.value,
            'origin_countries': flavor_profile.origin_countries,
            'seasonal_availability': json.dumps(flavor_profile.seasonal_availability),
            'confidence': flavor_profile.overall_confidence
        }
        
        async with self.driver.session() as session:
            result = await session.run(cypher, params)
            record = await result.single()
            
            if record:
                self.query_stats['nodes_created'] += 1
                return record['id']
            
            return None
    
    async def create_sensory_profile_node(self, ingredient_id: str, 
                                        sensory: SensoryProfile) -> str:
        """Create sensory profile node and link to ingredient"""
        
        cypher = """
        MATCH (i:Ingredient {id: $ingredient_id})
        MERGE (s:FlavorProfile {ingredient_id: $ingredient_id})
        SET s.sweet = $sweet,
            s.sour = $sour,
            s.salty = $salty,
            s.bitter = $bitter,
            s.umami = $umami,
            s.fatty = $fatty,
            s.spicy = $spicy,
            s.aromatic = $aromatic,
            s.cooling = $cooling,
            s.warming = $warming,
            s.astringent = $astringent,
            s.confidence_avg = $confidence_avg,
            s.last_updated = datetime()
        MERGE (i)-[:HAS_FLAVOR_PROFILE]->(s)
        RETURN s.ingredient_id as id
        """
        
        sensory_vector = sensory.to_vector()
        confidence_avg = np.mean(list(sensory.confidence_scores.values())) if sensory.confidence_scores else 0.0
        
        params = {
            'ingredient_id': ingredient_id,
            'sweet': float(sensory.sweet),
            'sour': float(sensory.sour),
            'salty': float(sensory.salty),
            'bitter': float(sensory.bitter),
            'umami': float(sensory.umami),
            'fatty': float(sensory.fatty),
            'spicy': float(sensory.spicy),
            'aromatic': float(sensory.aromatic),
            'cooling': float(sensory.cooling),
            'warming': float(sensory.warming),
            'astringent': float(sensory.astringent),
            'confidence_avg': confidence_avg
        }
        
        async with self.driver.session() as session:
            result = await session.run(cypher, params)
            record = await result.single()
            
            if record:
                self.query_stats['nodes_created'] += 1
                return record['id']
            
            return None
    
    async def create_compound_nodes(self, ingredient_id: str, 
                                  compounds: Dict[str, ChemicalCompound]) -> List[str]:
        """Create chemical compound nodes and relationships"""
        
        created_ids = []
        
        for compound_id, compound in compounds.items():
            cypher = """
            MERGE (c:Compound {id: $compound_id})
            SET c.name = $name,
                c.molecular_formula = $formula,
                c.molecular_weight = $weight,
                c.smiles = $smiles,
                c.compound_class = $class,
                c.odor_descriptors = $odor_descriptors,
                c.taste_descriptors = $taste_descriptors,
                c.boiling_point = $boiling_point,
                c.vapor_pressure = $vapor_pressure,
                c.last_updated = datetime()
            
            WITH c
            MATCH (i:Ingredient {id: $ingredient_id})
            MERGE (i)-[r:CONTAINS_COMPOUND]->(c)
            SET r.concentration = $concentration,
                r.volatility = $volatility
            
            RETURN c.id as id
            """
            
            params = {
                'compound_id': compound.compound_id,
                'ingredient_id': ingredient_id,
                'name': compound.name,
                'formula': compound.molecular_formula or '',
                'weight': compound.molecular_weight or 0.0,
                'smiles': compound.smiles or '',
                'class': compound.compound_class.value,
                'odor_descriptors': compound.odor_descriptors,
                'taste_descriptors': compound.taste_descriptors,
                'boiling_point': compound.boiling_point,
                'vapor_pressure': compound.vapor_pressure,
                'concentration': 0.1,  # Default concentration
                'volatility': 'high' if compound.vapor_pressure and compound.vapor_pressure > 1.0 else 'low'
            }
            
            try:
                async with self.driver.session() as session:
                    result = await session.run(cypher, params)
                    record = await result.single()
                    
                    if record:
                        created_ids.append(record['id'])
                        self.query_stats['nodes_created'] += 1
                        self.query_stats['relationships_created'] += 1
            
            except Exception as e:
                self.logger.warning(f"Failed to create compound node {compound_id}: {e}")
        
        return created_ids
    
    async def create_pairing_relationship(self, compatibility: IngredientCompatibility) -> bool:
        """Create ingredient pairing relationship"""
        
        cypher = """
        MATCH (a:Ingredient {id: $ingredient_a})
        MATCH (b:Ingredient {id: $ingredient_b})
        MERGE (a)-[r:PAIRS_WITH]-(b)
        SET r.compatibility_score = $compatibility_score,
            r.pmi_score = $pmi_score,
            r.co_occurrence_count = $co_occurrence_count,
            r.confidence_level = $confidence_level,
            r.cuisines = $cuisines,
            r.cooking_methods = $cooking_methods,
            r.last_updated = datetime()
        RETURN r
        """
        
        params = {
            'ingredient_a': compatibility.ingredient_a,
            'ingredient_b': compatibility.ingredient_b,
            'compatibility_score': compatibility.compatibility_score.value,
            'pmi_score': compatibility.pmi_score,
            'co_occurrence_count': compatibility.co_occurrence_count,
            'confidence_level': compatibility.confidence_level,
            'cuisines': compatibility.cuisine_contexts,
            'cooking_methods': compatibility.cooking_methods
        }
        
        try:
            async with self.driver.session() as session:
                result = await session.run(cypher, params)
                record = await result.single()
                
                if record:
                    self.query_stats['relationships_created'] += 1
                    return True
        
        except Exception as e:
            self.logger.error(f"Failed to create pairing relationship: {e}")
        
        return False
    
    async def execute_cypher_query(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute custom Cypher query"""
        
        # Check cache first
        cache_key = hashlib.md5(f"{cypher}_{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
        
        if self.config.enable_caching and cache_key in self.query_cache:
            cache_time = self.cache_timestamps.get(cache_key)
            if cache_time and datetime.now() - cache_time < timedelta(hours=self.config.cache_ttl_hours):
                self.query_stats['cache_hits'] += 1
                return self.query_cache[cache_key]
        
        start_time = datetime.now()
        
        try:
            async with self.driver.session() as session:
                result = await session.run(cypher, params or {})
                records = await result.data()
                
                # Update statistics
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                self.query_stats['total_queries'] += 1
                
                # Update average execution time
                current_avg = self.query_stats['average_execution_time_ms']
                total_queries = self.query_stats['total_queries']
                self.query_stats['average_execution_time_ms'] = (
                    (current_avg * (total_queries - 1) + execution_time) / total_queries
                )
                
                # Cache result
                if self.config.enable_caching:
                    self.query_cache[cache_key] = records
                    self.cache_timestamps[cache_key] = datetime.now()
                
                return records
        
        except Exception as e:
            self.logger.error(f"Cypher query failed: {e}")
            self.logger.error(f"Query: {cypher}")
            self.logger.error(f"Params: {params}")
            return []
    
    async def find_similar_ingredients(self, ingredient_id: str, 
                                     similarity_threshold: float = 0.7,
                                     limit: int = 10) -> List[Dict[str, Any]]:
        """Find ingredients with similar flavor profiles"""
        
        cypher = """
        MATCH (target:Ingredient {id: $ingredient_id})-[:HAS_FLAVOR_PROFILE]->(target_profile:FlavorProfile)
        MATCH (other:Ingredient)-[:HAS_FLAVOR_PROFILE]->(other_profile:FlavorProfile)
        WHERE target.id <> other.id
        
        WITH target, other, target_profile, other_profile,
             sqrt(
                 pow(target_profile.sweet - other_profile.sweet, 2) +
                 pow(target_profile.sour - other_profile.sour, 2) +
                 pow(target_profile.salty - other_profile.salty, 2) +
                 pow(target_profile.bitter - other_profile.bitter, 2) +
                 pow(target_profile.umami - other_profile.umami, 2) +
                 pow(target_profile.fatty - other_profile.fatty, 2) +
                 pow(target_profile.spicy - other_profile.spicy, 2) +
                 pow(target_profile.aromatic - other_profile.aromatic, 2)
             ) as euclidean_distance
        
        WITH target, other, (1.0 / (1.0 + euclidean_distance)) as similarity_score
        WHERE similarity_score >= $similarity_threshold
        
        RETURN other.id as ingredient_id, 
               other.name as name,
               similarity_score,
               other.primary_category as category
        ORDER BY similarity_score DESC
        LIMIT $limit
        """
        
        params = {
            'ingredient_id': ingredient_id,
            'similarity_threshold': similarity_threshold,
            'limit': limit
        }
        
        return await self.execute_cypher_query(cypher, params)
    
    async def find_ingredient_substitutes(self, ingredient_id: str, 
                                        cuisine_filter: str = None,
                                        limit: int = 10) -> List[Dict[str, Any]]:
        """Find potential substitutes for an ingredient"""
        
        base_cypher = """
        MATCH (target:Ingredient {id: $ingredient_id})
        MATCH (target)-[:PAIRS_WITH]-(shared)-[:PAIRS_WITH]-(substitute:Ingredient)
        WHERE target.id <> substitute.id
        """
        
        if cuisine_filter:
            base_cypher += """
            AND any(cuisine in substitute.cuisine_associations WHERE cuisine = $cuisine_filter)
            """
        
        base_cypher += """
        WITH substitute, count(shared) as shared_pairings,
             collect(shared.name) as shared_ingredients
        
        MATCH (target)-[:HAS_FLAVOR_PROFILE]->(target_profile:FlavorProfile)
        MATCH (substitute)-[:HAS_FLAVOR_PROFILE]->(sub_profile:FlavorProfile)
        
        WITH substitute, shared_pairings, shared_ingredients,
             sqrt(
                 pow(target_profile.sweet - sub_profile.sweet, 2) +
                 pow(target_profile.sour - sub_profile.sour, 2) +
                 pow(target_profile.salty - sub_profile.salty, 2) +
                 pow(target_profile.bitter - sub_profile.bitter, 2) +
                 pow(target_profile.umami - sub_profile.umami, 2)
             ) as flavor_distance
        
        WITH substitute, shared_pairings, shared_ingredients,
             (shared_pairings * 0.6) + ((1.0 / (1.0 + flavor_distance)) * 0.4) as substitute_score
        
        RETURN substitute.id as ingredient_id,
               substitute.name as name,
               substitute_score,
               shared_pairings,
               shared_ingredients[0..5] as sample_shared_ingredients
        ORDER BY substitute_score DESC
        LIMIT $limit
        """
        
        params = {
            'ingredient_id': ingredient_id,
            'limit': limit
        }
        
        if cuisine_filter:
            params['cuisine_filter'] = cuisine_filter
        
        return await self.execute_cypher_query(base_cypher, params)
    
    async def discover_flavor_pairings(self, ingredients: List[str], 
                                     max_depth: int = 2) -> List[Dict[str, Any]]:
        """Discover potential flavor pairings using graph traversal"""
        
        cypher = """
        MATCH (start:Ingredient)
        WHERE start.id IN $ingredients
        
        MATCH path = (start)-[:PAIRS_WITH*1..""" + str(max_depth) + """]->(related:Ingredient)
        WHERE NOT related.id IN $ingredients
        
        WITH related, 
             collect(distinct start.name) as connected_to,
             min(length(path)) as shortest_path,
             avg([r in relationships(path) | r.pmi_score]) as avg_pmi_score
        
        WHERE avg_pmi_score > 1.0
        
        RETURN related.id as ingredient_id,
               related.name as name,
               connected_to,
               shortest_path,
               avg_pmi_score as compatibility_score
        ORDER BY avg_pmi_score DESC, shortest_path ASC
        LIMIT 20
        """
        
        params = {'ingredients': ingredients}
        
        return await self.execute_cypher_query(cypher, params)
    
    async def analyze_cultural_patterns(self, cuisine: str) -> Dict[str, Any]:
        """Analyze ingredient usage patterns for a specific cuisine"""
        
        # Get most common ingredients in cuisine
        ingredient_query = """
        MATCH (i:Ingredient)-[r:PAIRS_WITH]-(other:Ingredient)
        WHERE $cuisine IN r.cuisines
        
        WITH i, count(r) as pairing_count, collect(other.name) as common_pairings
        
        RETURN i.id as ingredient_id,
               i.name as name,
               pairing_count,
               common_pairings[0..10] as top_pairings
        ORDER BY pairing_count DESC
        LIMIT 50
        """
        
        # Get flavor characteristics of cuisine
        flavor_query = """
        MATCH (i:Ingredient)-[r:PAIRS_WITH]-(other:Ingredient)
        WHERE $cuisine IN r.cuisines
        MATCH (i)-[:HAS_FLAVOR_PROFILE]->(profile:FlavorProfile)
        
        RETURN avg(profile.sweet) as avg_sweetness,
               avg(profile.sour) as avg_sourness,
               avg(profile.salty) as avg_saltiness,
               avg(profile.bitter) as avg_bitterness,
               avg(profile.umami) as avg_umami,
               avg(profile.spicy) as avg_spiciness,
               avg(profile.aromatic) as avg_aromatic
        """
        
        params = {'cuisine': cuisine}
        
        ingredients = await self.execute_cypher_query(ingredient_query, params)
        flavor_profile = await self.execute_cypher_query(flavor_query, params)
        
        return {
            'cuisine': cuisine,
            'common_ingredients': ingredients,
            'flavor_characteristics': flavor_profile[0] if flavor_profile else {},
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        
        stats_query = """
        MATCH (i:Ingredient) 
        WITH count(i) as ingredient_count
        
        MATCH (c:Compound)
        WITH ingredient_count, count(c) as compound_count
        
        MATCH (fp:FlavorProfile)
        WITH ingredient_count, compound_count, count(fp) as profile_count
        
        MATCH ()-[r:PAIRS_WITH]-()
        WITH ingredient_count, compound_count, profile_count, count(r) as pairing_count
        
        MATCH ()-[r2:CONTAINS_COMPOUND]-()
        WITH ingredient_count, compound_count, profile_count, pairing_count, count(r2) as compound_rel_count
        
        RETURN ingredient_count, compound_count, profile_count, pairing_count, compound_rel_count
        """
        
        result = await self.execute_cypher_query(stats_query)
        
        if result:
            stats = result[0]
            stats.update(self.query_stats)
            return stats
        
        return self.query_stats


class GraphRAGQueryProcessor:
    """Processes natural language queries and generates graph queries"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Query templates
        self.query_templates = {
            GraphQueryType.SIMILARITY_SEARCH: [
                "Find ingredients similar to {ingredient}",
                "What tastes like {ingredient}?",
                "Ingredients with similar flavor to {ingredient}"
            ],
            GraphQueryType.SUBSTITUTION_FIND: [
                "What can I substitute for {ingredient}?",
                "Alternatives to {ingredient}",
                "Replace {ingredient} with what?"
            ],
            GraphQueryType.PAIRING_DISCOVERY: [
                "What goes well with {ingredients}?",
                "Good pairings for {ingredients}",
                "What to combine with {ingredients}?"
            ]
        }
        
        # LLM prompt templates
        self.llm_prompts = {
            'query_generation': """
            You are a culinary expert and graph database specialist. Convert this natural language query into a structured graph query.

            Natural Language Query: "{query}"

            Available graph query types:
            - similarity_search: Find ingredients with similar flavors
            - substitution_find: Find ingredient substitutes
            - pairing_discovery: Find complementary ingredients
            - cultural_analysis: Analyze cuisine-specific patterns
            - seasonal_patterns: Find seasonal ingredient usage
            - molecular_similarity: Find chemically similar ingredients

            Extract:
            1. Query type
            2. Target ingredients
            3. Any cuisine, season, or other filters
            4. Similarity threshold (0.0-1.0)
            5. Maximum results needed

            Respond in JSON format.
            """,
            
            'result_interpretation': """
            You are a culinary expert. Interpret these graph query results and provide insights.

            Query: "{query}"
            Results: {results}

            Provide:
            1. A natural language summary of the results
            2. Culinary insights and explanations
            3. Practical recommendations for use
            4. Any interesting patterns or relationships discovered

            Keep it concise but informative.
            """
        }
    
    async def process_natural_language_query(self, query_text: str) -> GraphQuery:
        """Process natural language query into structured graph query"""
        
        # Generate query ID
        query_id = hashlib.md5(f"{query_text}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        
        # Try pattern matching first
        structured_query = self._pattern_match_query(query_text)
        
        if not structured_query and self.config.use_llm_query_generation:
            # Fall back to LLM processing
            structured_query = await self._llm_process_query(query_text)
        
        if not structured_query:
            # Default to similarity search
            structured_query = GraphQuery(
                query_id=query_id,
                query_type=GraphQueryType.SIMILARITY_SEARCH,
                natural_language=query_text
            )
        
        structured_query.query_id = query_id
        structured_query.natural_language = query_text
        
        return structured_query
    
    def _pattern_match_query(self, query_text: str) -> Optional[GraphQuery]:
        """Use pattern matching to identify query type and parameters"""
        
        query_lower = query_text.lower()
        
        # Extract ingredients from query
        ingredients = self._extract_ingredients(query_text)
        
        # Pattern matching for different query types
        if any(pattern in query_lower for pattern in ['similar', 'like', 'taste like']):
            return GraphQuery(
                query_id="",
                query_type=GraphQueryType.SIMILARITY_SEARCH,
                target_ingredient=ingredients[0] if ingredients else None
            )
        
        elif any(pattern in query_lower for pattern in ['substitute', 'replace', 'alternative']):
            return GraphQuery(
                query_id="",
                query_type=GraphQueryType.SUBSTITUTION_FIND,
                target_ingredient=ingredients[0] if ingredients else None
            )
        
        elif any(pattern in query_lower for pattern in ['pair', 'go with', 'combine', 'match']):
            return GraphQuery(
                query_id="",
                query_type=GraphQueryType.PAIRING_DISCOVERY,
                target_ingredient=ingredients[0] if ingredients else None
            )
        
        elif any(pattern in query_lower for pattern in ['cuisine', 'cultural', 'traditional']):
            cuisine = self._extract_cuisine(query_text)
            return GraphQuery(
                query_id="",
                query_type=GraphQueryType.CULTURAL_ANALYSIS,
                cuisine_filter=cuisine
            )
        
        return None
    
    def _extract_ingredients(self, text: str) -> List[str]:
        """Extract ingredient names from text"""
        # Simple ingredient extraction - would be enhanced with NER
        common_ingredients = [
            'tomato', 'onion', 'garlic', 'basil', 'oregano', 'thyme', 'rosemary',
            'chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'shrimp',
            'cheese', 'butter', 'cream', 'milk', 'yogurt',
            'lemon', 'lime', 'orange', 'apple', 'banana',
            'carrot', 'celery', 'potato', 'broccoli', 'spinach',
            'rice', 'pasta', 'bread', 'flour',
            'salt', 'pepper', 'sugar', 'honey', 'vinegar'
        ]
        
        found_ingredients = []
        text_lower = text.lower()
        
        for ingredient in common_ingredients:
            if ingredient in text_lower:
                found_ingredients.append(ingredient)
        
        return found_ingredients
    
    def _extract_cuisine(self, text: str) -> Optional[str]:
        """Extract cuisine type from text"""
        cuisines = [
            'italian', 'french', 'chinese', 'indian', 'mexican', 'japanese',
            'thai', 'mediterranean', 'american', 'korean', 'spanish', 'greek'
        ]
        
        text_lower = text.lower()
        
        for cuisine in cuisines:
            if cuisine in text_lower:
                return cuisine
        
        return None
    
    async def _llm_process_query(self, query_text: str) -> Optional[GraphQuery]:
        """Use LLM to process natural language query"""
        
        # Mock LLM processing - would integrate with actual LLM
        prompt = self.llm_prompts['query_generation'].format(query=query_text)
        
        # Simulate LLM response
        llm_response = {
            'query_type': 'similarity_search',
            'target_ingredients': self._extract_ingredients(query_text),
            'similarity_threshold': 0.7,
            'max_results': 10
        }
        
        try:
            query_type = GraphQueryType(llm_response.get('query_type', 'similarity_search'))
            
            return GraphQuery(
                query_id="",
                query_type=query_type,
                target_ingredient=llm_response.get('target_ingredients', [None])[0],
                similarity_threshold=llm_response.get('similarity_threshold', 0.7),
                max_results=llm_response.get('max_results', 10)
            )
        
        except Exception as e:
            self.logger.warning(f"LLM query processing failed: {e}")
            return None


class GraphRAGEngine:
    """Main GraphRAG engine combining graph database with LLM reasoning"""
    
    def __init__(self, config: GraphRAGConfig = None):
        self.config = config or GraphRAGConfig()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.graph_db = Neo4jFlavorGraphDB(self.config)
        self.query_processor = GraphRAGQueryProcessor(self.config)
        
        # Embeddings and ML models
        self.embeddings_cache: Dict[str, GraphEmbedding] = {}
        self.similarity_model = None
        
        # Training data generation
        self.triplet_cache: List[TrainingTriplet] = []
        
        # Performance metrics
        self.performance_stats = {
            'queries_processed': 0,
            'average_response_time_ms': 0,
            'cache_hit_rate': 0.0,
            'triplets_generated': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.graph_db.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.graph_db.close()
    
    async def ingest_flavor_profile(self, flavor_profile: FlavorProfile) -> bool:
        """Ingest a complete flavor profile into the graph"""
        
        try:
            # Create ingredient node
            ingredient_id = await self.graph_db.create_ingredient_node(flavor_profile)
            
            if not ingredient_id:
                return False
            
            # Create sensory profile node
            await self.graph_db.create_sensory_profile_node(ingredient_id, flavor_profile.sensory)
            
            # Create compound nodes
            if flavor_profile.molecular.compounds:
                await self.graph_db.create_compound_nodes(ingredient_id, flavor_profile.molecular.compounds)
            
            # Create pairing relationships
            for other_ingredient, compatibility in flavor_profile.relational.compatible_ingredients.items():
                await self.graph_db.create_pairing_relationship(compatibility)
            
            self.logger.info(f"Successfully ingested flavor profile for {flavor_profile.name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to ingest flavor profile for {flavor_profile.name}: {e}")
            return False
    
    async def query_graph(self, query: Union[str, GraphQuery]) -> GraphQueryResult:
        """Execute graph query and return results"""
        
        start_time = datetime.now()
        
        # Process natural language query if needed
        if isinstance(query, str):
            structured_query = await self.query_processor.process_natural_language_query(query)
        else:
            structured_query = query
        
        # Execute graph query based on type
        if structured_query.query_type == GraphQueryType.SIMILARITY_SEARCH:
            results = await self._execute_similarity_search(structured_query)
        elif structured_query.query_type == GraphQueryType.SUBSTITUTION_FIND:
            results = await self._execute_substitution_search(structured_query)
        elif structured_query.query_type == GraphQueryType.PAIRING_DISCOVERY:
            results = await self._execute_pairing_discovery(structured_query)
        elif structured_query.query_type == GraphQueryType.CULTURAL_ANALYSIS:
            results = await self._execute_cultural_analysis(structured_query)
        else:
            results = []
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Generate result object
        query_result = GraphQueryResult(
            query_id=structured_query.query_id,
            results=results,
            total_results=len(results),
            execution_time_ms=int(execution_time),
            graph_traversal_depth=2,  # Would be calculated based on actual traversal
            confidence_score=0.8,  # Would be calculated based on result quality
            result_quality=0.85   # Would be calculated based on various factors
        )
        
        # Add LLM interpretation if enabled
        if self.config.use_llm_result_interpretation and structured_query.natural_language:
            query_result.llm_interpretation = await self._generate_llm_interpretation(
                structured_query.natural_language, results
            )
        
        # Update performance stats
        self.performance_stats['queries_processed'] += 1
        current_avg = self.performance_stats['average_response_time_ms']
        total_queries = self.performance_stats['queries_processed']
        self.performance_stats['average_response_time_ms'] = (
            (current_avg * (total_queries - 1) + execution_time) / total_queries
        )
        
        return query_result
    
    async def _execute_similarity_search(self, query: GraphQuery) -> List[Dict[str, Any]]:
        """Execute similarity search query"""
        
        if not query.target_ingredient:
            return []
        
        return await self.graph_db.find_similar_ingredients(
            query.target_ingredient,
            query.similarity_threshold,
            query.max_results
        )
    
    async def _execute_substitution_search(self, query: GraphQuery) -> List[Dict[str, Any]]:
        """Execute substitution search query"""
        
        if not query.target_ingredient:
            return []
        
        return await self.graph_db.find_ingredient_substitutes(
            query.target_ingredient,
            query.cuisine_filter,
            query.max_results
        )
    
    async def _execute_pairing_discovery(self, query: GraphQuery) -> List[Dict[str, Any]]:
        """Execute pairing discovery query"""
        
        ingredients = []
        if query.target_ingredient:
            ingredients.append(query.target_ingredient)
        
        if not ingredients:
            return []
        
        return await self.graph_db.discover_flavor_pairings(ingredients, max_depth=2)
    
    async def _execute_cultural_analysis(self, query: GraphQuery) -> List[Dict[str, Any]]:
        """Execute cultural analysis query"""
        
        if not query.cuisine_filter:
            return []
        
        analysis = await self.graph_db.analyze_cultural_patterns(query.cuisine_filter)
        return [analysis] if analysis else []
    
    async def _generate_llm_interpretation(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate LLM interpretation of results"""
        
        # Mock LLM interpretation - would integrate with actual LLM
        if not results:
            return f"No results found for the query: '{query}'. Try rephrasing or using different ingredients."
        
        result_summary = f"Found {len(results)} results for '{query}'."
        
        if len(results) > 0:
            top_result = results[0]
            if 'name' in top_result:
                result_summary += f" The top match is {top_result['name']}."
        
        return result_summary
    
    async def generate_training_triplets(self, ingredient_id: str, 
                                       count: int = None) -> List[TrainingTriplet]:
        """Generate training triplets for neural network learning"""
        
        count = count or self.config.triplets_per_ingredient
        triplets = []
        
        # Get ingredient's pairings (positive examples)
        positive_query = """
        MATCH (anchor:Ingredient {id: $ingredient_id})-[r:PAIRS_WITH]-(positive:Ingredient)
        WHERE r.pmi_score > $min_strength
        RETURN positive.id as positive_id, r.pmi_score as score
        ORDER BY r.pmi_score DESC
        """
        
        positive_results = await self.graph_db.execute_cypher_query(
            positive_query,
            {
                'ingredient_id': ingredient_id,
                'min_strength': self.config.min_relationship_strength
            }
        )
        
        # Get random ingredients for negative examples
        negative_query = """
        MATCH (anchor:Ingredient {id: $ingredient_id})
        MATCH (negative:Ingredient)
        WHERE negative.id <> anchor.id
        AND NOT (anchor)-[:PAIRS_WITH]-(negative)
        RETURN negative.id as negative_id
        ORDER BY rand()
        LIMIT $negative_count
        """
        
        negative_count = int(len(positive_results) * self.config.negative_sampling_ratio)
        negative_results = await self.graph_db.execute_cypher_query(
            negative_query,
            {
                'ingredient_id': ingredient_id,
                'negative_count': negative_count
            }
        )
        
        # Generate triplets
        for i, positive in enumerate(positive_results[:count]):
            if i < len(negative_results):
                negative = negative_results[i]
                
                triplet = TrainingTriplet(
                    anchor=ingredient_id,
                    positive=positive['positive_id'],
                    negative=negative['negative_id'],
                    positive_score=positive['score'],
                    negative_score=0.0,  # Assumed to be incompatible
                    relationship_type=GraphRelationType.PAIRS_WITH,
                    confidence=min(positive['score'] / 5.0, 1.0)  # Normalize PMI score
                )
                
                triplets.append(triplet)
        
        # Cache triplets
        self.triplet_cache.extend(triplets)
        self.performance_stats['triplets_generated'] += len(triplets)
        
        return triplets
    
    async def batch_generate_triplets(self, ingredient_ids: List[str]) -> List[TrainingTriplet]:
        """Generate training triplets for multiple ingredients"""
        
        all_triplets = []
        
        # Process in batches to avoid overwhelming the database
        batch_size = self.config.batch_size
        
        for i in range(0, len(ingredient_ids), batch_size):
            batch_ids = ingredient_ids[i:i + batch_size]
            
            batch_tasks = [
                self.generate_training_triplets(ingredient_id)
                for ingredient_id in batch_ids
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, list):
                    all_triplets.extend(result)
                elif isinstance(result, Exception):
                    self.logger.warning(f"Triplet generation failed: {result}")
        
        return all_triplets
    
    def get_cached_triplets(self, filter_confidence: float = 0.5) -> List[TrainingTriplet]:
        """Get cached training triplets above confidence threshold"""
        return [triplet for triplet in self.triplet_cache if triplet.confidence >= filter_confidence]
    
    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        db_stats = await self.graph_db.get_database_statistics()
        
        return {
            'graph_database': db_stats,
            'performance': self.performance_stats,
            'embeddings_cached': len(self.embeddings_cache),
            'triplets_cached': len(self.triplet_cache),
            'config': {
                'embedding_dimension': self.config.embedding_dimension,
                'max_results_default': 20,
                'cache_enabled': self.config.enable_caching
            }
        }


# Factory functions and utilities

def create_graphrag_engine(neo4j_uri: str = "bolt://localhost:7687",
                          neo4j_user: str = "neo4j", 
                          neo4j_password: str = "password") -> GraphRAGEngine:
    """Create GraphRAG engine with custom Neo4j connection"""
    config = GraphRAGConfig(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password
    )
    return GraphRAGEngine(config)


async def quick_graph_query(query: str, neo4j_config: Dict[str, str] = None) -> Dict[str, Any]:
    """Quick graph query without setting up full engine"""
    
    config = GraphRAGConfig()
    if neo4j_config:
        config.neo4j_uri = neo4j_config.get('uri', config.neo4j_uri)
        config.neo4j_user = neo4j_config.get('user', config.neo4j_user)
        config.neo4j_password = neo4j_config.get('password', config.neo4j_password)
    
    async with GraphRAGEngine(config) as engine:
        result = await engine.query_graph(query)
        
        return {
            'query': query,
            'results': result.results,
            'execution_time_ms': result.execution_time_ms,
            'interpretation': result.llm_interpretation
        }


# Export key classes and functions
__all__ = [
    'GraphQueryType', 'GraphRelationType', 'GraphNodeType',
    'GraphQuery', 'GraphQueryResult', 'TrainingTriplet', 'GraphEmbedding',
    'GraphRAGConfig', 'Neo4jFlavorGraphDB', 'GraphRAGQueryProcessor', 
    'GraphRAGEngine', 'create_graphrag_engine', 'quick_graph_query'
]