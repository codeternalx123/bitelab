"""
Layer C: The Relational Layer (The "Wisdom") Implementation
=========================================================

This module implements the relational layer of the Automated Flavor Intelligence Pipeline.
It processes ingredient compatibility using co-occurrence analysis, graph theory, and 
Pointwise Mutual Information (PMI) calculations from recipe datasets.

Key Features:
- Recipe co-occurrence mining from Recipe1M+ and Food.com datasets
- Graph theory-based compatibility analysis using NetworkX
- Pointwise Mutual Information (PMI) calculation for pairing strength
- Cultural and cuisine-specific compatibility patterns
- Seasonal and contextual pairing analysis
- Automated substitution recommendation system
"""

from typing import Dict, List, Optional, Tuple, Set, Union, Any
from dataclasses import dataclass, field
import numpy as np
import logging
from datetime import datetime, timedelta
import asyncio
import json
import math
import networkx as nx
from collections import defaultdict, Counter
import pickle
import gzip
from pathlib import Path
import csv
import re
from enum import Enum
import statistics

from .flavor_data_models import (
    IngredientCompatibility, RelationalProfile, CompatibilityLevel,
    DataSource, FlavorCategory, FlavorProfile
)


class CompatibilityAnalysisMethod(Enum):
    """Methods for analyzing ingredient compatibility"""
    CO_OCCURRENCE_MINING = "co_occurrence_mining"
    PMI_CALCULATION = "pmi_calculation"
    GRAPH_CENTRALITY = "graph_centrality"
    CULTURAL_ANALYSIS = "cultural_analysis"
    SEASONAL_PATTERNS = "seasonal_patterns"
    MACHINE_LEARNING = "machine_learning"
    HYBRID_APPROACH = "hybrid_approach"


class RecipeDataSource(Enum):
    """Sources for recipe data"""
    RECIPE1M = "recipe1m"
    FOOD_COM = "food_com"
    ALLRECIPES = "allrecipes"
    EPICURIOUS = "epicurious"
    YUMMLY = "yummly"
    CUSTOM_DATASET = "custom_dataset"


class CuisineType(Enum):
    """Major world cuisine types"""
    ITALIAN = "italian"
    FRENCH = "french" 
    CHINESE = "chinese"
    INDIAN = "indian"
    MEXICAN = "mexican"
    JAPANESE = "japanese"
    THAI = "thai"
    MEDITERRANEAN = "mediterranean"
    AMERICAN = "american"
    MIDDLE_EASTERN = "middle_eastern"
    KOREAN = "korean"
    GERMAN = "german"
    SPANISH = "spanish"
    GREEK = "greek"
    TURKISH = "turkish"
    VIETNAMESE = "vietnamese"
    MOROCCAN = "moroccan"
    BRAZILIAN = "brazilian"
    ARGENTINIAN = "argentinian"
    PERUVIAN = "peruvian"


class SeasonalContext(Enum):
    """Seasonal contexts for ingredient pairings"""
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"
    ALL_SEASON = "all_season"


@dataclass
class RecipeIngredient:
    """Individual ingredient within a recipe"""
    name: str
    normalized_name: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    preparation: Optional[str] = None  # chopped, diced, etc.
    
    # Ingredient classification
    category: Optional[FlavorCategory] = None
    is_primary: bool = False  # Main ingredient vs seasoning/garnish
    
    def __hash__(self):
        return hash(self.normalized_name)


@dataclass
class Recipe:
    """Complete recipe data structure"""
    recipe_id: str
    title: str
    ingredients: List[RecipeIngredient]
    
    # Recipe metadata
    cuisine: Optional[CuisineType] = None
    cooking_method: Optional[str] = None
    meal_type: Optional[str] = None  # breakfast, lunch, dinner, etc.
    season: Optional[SeasonalContext] = None
    
    # Quality metrics
    rating: Optional[float] = None
    review_count: int = 0
    source: RecipeDataSource = RecipeDataSource.CUSTOM_DATASET
    
    # Processing metadata
    processed_date: datetime = field(default_factory=datetime.now)
    ingredient_count: int = field(init=False)
    
    def __post_init__(self):
        self.ingredient_count = len(self.ingredients)


@dataclass
class CoOccurrenceMatrix:
    """Co-occurrence matrix for ingredient pairs"""
    ingredients: List[str]
    matrix: np.ndarray
    total_recipes: int
    
    # Individual ingredient counts
    ingredient_counts: Dict[str, int]
    
    # Metadata
    data_source: RecipeDataSource
    last_updated: datetime = field(default_factory=datetime.now)
    cuisine_filter: Optional[CuisineType] = None
    
    def get_co_occurrence(self, ingredient_a: str, ingredient_b: str) -> int:
        """Get co-occurrence count for two ingredients"""
        if ingredient_a not in self.ingredients or ingredient_b not in self.ingredients:
            return 0
        
        idx_a = self.ingredients.index(ingredient_a)
        idx_b = self.ingredients.index(ingredient_b)
        
        return int(self.matrix[idx_a, idx_b])
    
    def calculate_pmi(self, ingredient_a: str, ingredient_b: str) -> float:
        """Calculate Pointwise Mutual Information for ingredient pair"""
        co_count = self.get_co_occurrence(ingredient_a, ingredient_b)
        
        if co_count == 0:
            return float('-inf')  # Never co-occur
        
        count_a = self.ingredient_counts.get(ingredient_a, 0)
        count_b = self.ingredient_counts.get(ingredient_b, 0)
        
        if count_a == 0 or count_b == 0:
            return float('-inf')
        
        # PMI = log2(P(A,B) / (P(A) * P(B)))
        p_ab = co_count / self.total_recipes
        p_a = count_a / self.total_recipes
        p_b = count_b / self.total_recipes
        
        return math.log2(p_ab / (p_a * p_b))
    
    def get_top_pairings(self, ingredient: str, limit: int = 20) -> List[Tuple[str, float]]:
        """Get top PMI pairings for an ingredient"""
        if ingredient not in self.ingredients:
            return []
        
        pairings = []
        
        for other_ingredient in self.ingredients:
            if other_ingredient != ingredient:
                pmi_score = self.calculate_pmi(ingredient, other_ingredient)
                if pmi_score > float('-inf'):
                    pairings.append((other_ingredient, pmi_score))
        
        return sorted(pairings, key=lambda x: x[1], reverse=True)[:limit]


@dataclass
class CompatibilityGraphNode:
    """Node in the ingredient compatibility graph"""
    ingredient_name: str
    total_recipes: int
    cuisines: Set[CuisineType]
    categories: Set[FlavorCategory]
    
    # Centrality measures
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    pagerank: float = 0.0
    
    # Clustering information
    cluster_id: Optional[str] = None
    cluster_coefficient: float = 0.0


@dataclass
class CompatibilityGraphEdge:
    """Edge in the ingredient compatibility graph"""
    ingredient_a: str
    ingredient_b: str
    co_occurrence_count: int
    pmi_score: float
    compatibility_strength: float
    
    # Contextual information
    cuisines: Set[CuisineType]
    cooking_methods: Set[str]
    seasons: Set[SeasonalContext]


@dataclass
class RelationalAnalysisConfig:
    """Configuration for relational layer analysis"""
    
    # Data processing parameters
    min_recipe_count: int = 5  # Minimum recipes for ingredient inclusion
    min_co_occurrence: int = 3  # Minimum co-occurrences for pairing
    pmi_threshold: float = 0.0  # Minimum PMI score for compatibility
    
    # Graph analysis parameters
    max_graph_size: int = 5000  # Maximum nodes in compatibility graph
    centrality_weight: float = 0.3
    clustering_resolution: float = 1.0
    
    # Cultural analysis
    analyze_cuisine_patterns: bool = True
    analyze_seasonal_patterns: bool = True
    min_cuisine_recipes: int = 50  # Minimum recipes per cuisine
    
    # Quality control
    filter_common_ingredients: bool = True  # Filter salt, pepper, etc.
    normalize_ingredient_names: bool = True
    validate_pairings: bool = True
    
    # Performance settings
    use_sparse_matrices: bool = True
    parallel_processing: bool = True
    chunk_size: int = 1000


class IngredientNormalizer:
    """Normalizes ingredient names for consistency"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common ingredient mappings
        self.ingredient_mappings = {
            # Consolidate similar ingredients
            'tomatoes': 'tomato',
            'onions': 'onion', 
            'carrots': 'carrot',
            'potatoes': 'potato',
            'lemons': 'lemon',
            'garlic cloves': 'garlic',
            'olive oil': 'oil',
            'vegetable oil': 'oil',
            'canola oil': 'oil',
            'butter': 'butter',
            'unsalted butter': 'butter',
            'salted butter': 'butter',
            'black pepper': 'pepper',
            'white pepper': 'pepper',
            'ground black pepper': 'pepper',
            'kosher salt': 'salt',
            'sea salt': 'salt',
            'table salt': 'salt',
            'fresh basil': 'basil',
            'dried basil': 'basil',
            'fresh parsley': 'parsley',
            'dried parsley': 'parsley'
        }
        
        # Common words to remove
        self.stop_words = {
            'fresh', 'dried', 'ground', 'chopped', 'diced', 'minced',
            'sliced', 'whole', 'large', 'medium', 'small', 'extra',
            'pure', 'organic', 'raw', 'cooked', 'prepared', 'canned',
            'frozen', 'thawed', 'room', 'temperature', 'cold', 'hot'
        }
        
        # Ingredients to filter out (too common/basic)
        self.common_ingredients = {
            'salt', 'pepper', 'water', 'oil', 'butter', 'flour',
            'sugar', 'egg', 'eggs', 'milk', 'cream'
        }
    
    def normalize(self, ingredient_name: str) -> str:
        """Normalize ingredient name for consistency"""
        if not ingredient_name:
            return ""
        
        # Convert to lowercase and strip
        normalized = ingredient_name.lower().strip()
        
        # Remove measurements and quantities
        normalized = re.sub(r'\d+\.?\d*\s*(cups?|tbsp|tsp|oz|lbs?|kg|g|ml|l)\s*', '', normalized)
        normalized = re.sub(r'\(\d+\.?\d*\s*oz\.?\)', '', normalized)
        
        # Remove parenthetical information
        normalized = re.sub(r'\([^)]*\)', '', normalized)
        
        # Remove stop words
        words = normalized.split()
        filtered_words = [w for w in words if w not in self.stop_words]
        normalized = ' '.join(filtered_words)
        
        # Apply specific mappings
        if normalized in self.ingredient_mappings:
            normalized = self.ingredient_mappings[normalized]
        
        # Final cleanup
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def should_include(self, normalized_name: str) -> bool:
        """Check if ingredient should be included in analysis"""
        if not normalized_name or len(normalized_name) < 3:
            return False
        
        if normalized_name in self.common_ingredients:
            return False
        
        return True


class RecipeDataProcessor:
    """Processes recipe datasets for compatibility analysis"""
    
    def __init__(self, config: RelationalAnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.normalizer = IngredientNormalizer()
        
        # Processing statistics
        self.stats = {
            'total_recipes_processed': 0,
            'valid_recipes': 0,
            'unique_ingredients': 0,
            'total_pairings': 0,
            'processing_time_ms': 0
        }
    
    async def process_recipe_dataset(self, dataset_path: str, 
                                   source: RecipeDataSource = RecipeDataSource.CUSTOM_DATASET) -> CoOccurrenceMatrix:
        """Process a recipe dataset and create co-occurrence matrix"""
        
        start_time = datetime.now()
        self.logger.info(f"Processing recipe dataset: {dataset_path}")
        
        # Load and parse recipes
        recipes = await self._load_recipes(dataset_path, source)
        
        if not recipes:
            raise ValueError(f"No valid recipes found in dataset: {dataset_path}")
        
        # Filter and normalize ingredients
        recipes = self._filter_and_normalize_recipes(recipes)
        
        # Build co-occurrence matrix
        matrix = self._build_cooccurrence_matrix(recipes)
        
        # Update statistics
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self.stats['processing_time_ms'] = int(processing_time)
        
        self.logger.info(f"Processed {len(recipes)} recipes with {len(matrix.ingredients)} unique ingredients")
        
        return matrix
    
    async def _load_recipes(self, dataset_path: str, source: RecipeDataSource) -> List[Recipe]:
        """Load recipes from various dataset formats"""
        
        path = Path(dataset_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        recipes = []
        
        if path.suffix.lower() == '.json':
            recipes = await self._load_json_recipes(path, source)
        elif path.suffix.lower() == '.csv':
            recipes = await self._load_csv_recipes(path, source)
        elif path.suffix.lower() == '.gz':
            recipes = await self._load_compressed_recipes(path, source)
        else:
            raise ValueError(f"Unsupported dataset format: {path.suffix}")
        
        self.stats['total_recipes_processed'] = len(recipes)
        return recipes
    
    async def _load_json_recipes(self, path: Path, source: RecipeDataSource) -> List[Recipe]:
        """Load recipes from JSON format"""
        recipes = []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                recipe_data = data
            elif isinstance(data, dict):
                recipe_data = data.get('recipes', [data])
            else:
                recipe_data = []
            
            for i, recipe_json in enumerate(recipe_data):
                try:
                    recipe = self._parse_recipe_json(recipe_json, source, str(i))
                    if recipe:
                        recipes.append(recipe)
                except Exception as e:
                    self.logger.warning(f"Failed to parse recipe {i}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to load JSON recipes: {e}")
        
        return recipes
    
    async def _load_csv_recipes(self, path: Path, source: RecipeDataSource) -> List[Recipe]:
        """Load recipes from CSV format"""
        recipes = []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    try:
                        recipe = self._parse_recipe_csv(row, source, str(i))
                        if recipe:
                            recipes.append(recipe)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse CSV recipe {i}: {e}")
        
        except Exception as e:
            self.logger.error(f"Failed to load CSV recipes: {e}")
        
        return recipes
    
    async def _load_compressed_recipes(self, path: Path, source: RecipeDataSource) -> List[Recipe]:
        """Load recipes from compressed format"""
        recipes = []
        
        try:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        recipe_json = json.loads(line.strip())
                        recipe = self._parse_recipe_json(recipe_json, source, str(i))
                        if recipe:
                            recipes.append(recipe)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse compressed recipe {i}: {e}")
        
        except Exception as e:
            self.logger.error(f"Failed to load compressed recipes: {e}")
        
        return recipes
    
    def _parse_recipe_json(self, recipe_json: dict, source: RecipeDataSource, recipe_id: str) -> Optional[Recipe]:
        """Parse recipe from JSON format"""
        
        # Extract ingredients
        ingredients_data = recipe_json.get('ingredients', [])
        if not ingredients_data:
            return None
        
        ingredients = []
        for ing_data in ingredients_data:
            if isinstance(ing_data, str):
                ingredient = RecipeIngredient(
                    name=ing_data,
                    normalized_name=self.normalizer.normalize(ing_data)
                )
            elif isinstance(ing_data, dict):
                name = ing_data.get('text', ing_data.get('name', ''))
                ingredient = RecipeIngredient(
                    name=name,
                    normalized_name=self.normalizer.normalize(name),
                    quantity=ing_data.get('quantity'),
                    unit=ing_data.get('unit')
                )
            else:
                continue
            
            if self.normalizer.should_include(ingredient.normalized_name):
                ingredients.append(ingredient)
        
        if len(ingredients) < 2:  # Need at least 2 ingredients for pairings
            return None
        
        # Extract metadata
        cuisine = self._parse_cuisine(recipe_json.get('cuisine', ''))
        
        return Recipe(
            recipe_id=recipe_id,
            title=recipe_json.get('title', recipe_json.get('name', '')),
            ingredients=ingredients,
            cuisine=cuisine,
            cooking_method=recipe_json.get('method'),
            rating=recipe_json.get('rating'),
            review_count=recipe_json.get('reviews', 0),
            source=source
        )
    
    def _parse_recipe_csv(self, row: dict, source: RecipeDataSource, recipe_id: str) -> Optional[Recipe]:
        """Parse recipe from CSV format"""
        
        # Extract ingredients from various possible column names
        ingredients_text = (
            row.get('ingredients', '') or 
            row.get('ingredient_list', '') or
            row.get('recipe_ingredients', '')
        )
        
        if not ingredients_text:
            return None
        
        # Parse ingredients (assume comma or newline separated)
        ingredient_names = re.split(r'[,;\n]', ingredients_text)
        
        ingredients = []
        for name in ingredient_names:
            name = name.strip()
            if name:
                ingredient = RecipeIngredient(
                    name=name,
                    normalized_name=self.normalizer.normalize(name)
                )
                
                if self.normalizer.should_include(ingredient.normalized_name):
                    ingredients.append(ingredient)
        
        if len(ingredients) < 2:
            return None
        
        cuisine = self._parse_cuisine(row.get('cuisine', ''))
        
        return Recipe(
            recipe_id=recipe_id,
            title=row.get('title', row.get('name', '')),
            ingredients=ingredients,
            cuisine=cuisine,
            rating=float(row.get('rating', 0)) if row.get('rating') else None,
            source=source
        )
    
    def _parse_cuisine(self, cuisine_text: str) -> Optional[CuisineType]:
        """Parse cuisine type from text"""
        if not cuisine_text:
            return None
        
        cuisine_lower = cuisine_text.lower()
        
        # Map common cuisine names
        cuisine_mappings = {
            'italian': CuisineType.ITALIAN,
            'french': CuisineType.FRENCH,
            'chinese': CuisineType.CHINESE,
            'indian': CuisineType.INDIAN,
            'mexican': CuisineType.MEXICAN,
            'japanese': CuisineType.JAPANESE,
            'thai': CuisineType.THAI,
            'mediterranean': CuisineType.MEDITERRANEAN,
            'american': CuisineType.AMERICAN,
            'middle eastern': CuisineType.MIDDLE_EASTERN,
            'korean': CuisineType.KOREAN,
            'german': CuisineType.GERMAN,
            'spanish': CuisineType.SPANISH,
            'greek': CuisineType.GREEK,
            'turkish': CuisineType.TURKISH,
            'vietnamese': CuisineType.VIETNAMESE,
            'moroccan': CuisineType.MOROCCAN,
            'brazilian': CuisineType.BRAZILIAN,
            'argentinian': CuisineType.ARGENTINIAN,
            'peruvian': CuisineType.PERUVIAN
        }
        
        for keyword, cuisine_type in cuisine_mappings.items():
            if keyword in cuisine_lower:
                return cuisine_type
        
        return None
    
    def _filter_and_normalize_recipes(self, recipes: List[Recipe]) -> List[Recipe]:
        """Filter and normalize recipes for analysis"""
        
        valid_recipes = []
        ingredient_counts = Counter()
        
        # First pass: count ingredient frequencies
        for recipe in recipes:
            for ingredient in recipe.ingredients:
                ingredient_counts[ingredient.normalized_name] += 1
        
        # Filter ingredients that appear in too few recipes
        frequent_ingredients = {
            ing for ing, count in ingredient_counts.items()
            if count >= self.config.min_recipe_count
        }
        
        # Second pass: filter recipes with frequent ingredients
        for recipe in recipes:
            filtered_ingredients = [
                ing for ing in recipe.ingredients
                if ing.normalized_name in frequent_ingredients
            ]
            
            if len(filtered_ingredients) >= 2:  # Need at least 2 ingredients
                recipe.ingredients = filtered_ingredients
                valid_recipes.append(recipe)
        
        self.stats['valid_recipes'] = len(valid_recipes)
        self.stats['unique_ingredients'] = len(frequent_ingredients)
        
        return valid_recipes
    
    def _build_cooccurrence_matrix(self, recipes: List[Recipe]) -> CoOccurrenceMatrix:
        """Build co-occurrence matrix from recipes"""
        
        # Get all unique ingredients
        all_ingredients = set()
        for recipe in recipes:
            for ingredient in recipe.ingredients:
                all_ingredients.add(ingredient.normalized_name)
        
        ingredients_list = sorted(list(all_ingredients))
        n_ingredients = len(ingredients_list)
        
        # Create mapping for fast lookup
        ingredient_to_idx = {ing: i for i, ing in enumerate(ingredients_list)}
        
        # Initialize matrix
        if self.config.use_sparse_matrices:
            from scipy.sparse import lil_matrix
            matrix = lil_matrix((n_ingredients, n_ingredients), dtype=int)
        else:
            matrix = np.zeros((n_ingredients, n_ingredients), dtype=int)
        
        # Count ingredient occurrences
        ingredient_counts = Counter()
        
        # Process recipes
        for recipe in recipes:
            recipe_ingredients = [ing.normalized_name for ing in recipe.ingredients]
            
            # Count individual ingredients
            for ing in recipe_ingredients:
                ingredient_counts[ing] += 1
            
            # Count pairwise co-occurrences
            for i, ing_a in enumerate(recipe_ingredients):
                for ing_b in recipe_ingredients[i+1:]:  # Avoid duplicates
                    idx_a = ingredient_to_idx[ing_a]
                    idx_b = ingredient_to_idx[ing_b]
                    
                    # Symmetric matrix
                    matrix[idx_a, idx_b] += 1
                    matrix[idx_b, idx_a] += 1
        
        # Convert sparse matrix to dense if needed
        if self.config.use_sparse_matrices:
            matrix = matrix.toarray()
        
        self.stats['total_pairings'] = int(np.sum(matrix) // 2)  # Divide by 2 for symmetric matrix
        
        return CoOccurrenceMatrix(
            ingredients=ingredients_list,
            matrix=matrix,
            total_recipes=len(recipes),
            ingredient_counts=dict(ingredient_counts),
            data_source=recipes[0].source if recipes else RecipeDataSource.CUSTOM_DATASET
        )


class CompatibilityGraphBuilder:
    """Builds and analyzes ingredient compatibility graphs"""
    
    def __init__(self, config: RelationalAnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def build_compatibility_graph(self, matrix: CoOccurrenceMatrix) -> nx.Graph:
        """Build NetworkX graph from co-occurrence matrix"""
        
        G = nx.Graph()
        
        # Add nodes for each ingredient
        for i, ingredient in enumerate(matrix.ingredients):
            count = matrix.ingredient_counts.get(ingredient, 0)
            
            G.add_node(ingredient, 
                      recipe_count=count,
                      frequency=count / matrix.total_recipes)
        
        # Add edges for significant co-occurrences
        for i, ing_a in enumerate(matrix.ingredients):
            for j, ing_b in enumerate(matrix.ingredients[i+1:], i+1):
                co_count = matrix.matrix[i, j]
                
                if co_count >= self.config.min_co_occurrence:
                    pmi_score = matrix.calculate_pmi(ing_a, ing_b)
                    
                    if pmi_score > self.config.pmi_threshold:
                        # Calculate compatibility strength
                        strength = self._calculate_compatibility_strength(
                            co_count, pmi_score, matrix.total_recipes
                        )
                        
                        G.add_edge(ing_a, ing_b,
                                 co_occurrence=co_count,
                                 pmi_score=pmi_score,
                                 strength=strength,
                                 weight=strength)
        
        self.logger.info(f"Built compatibility graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def _calculate_compatibility_strength(self, co_count: int, pmi_score: float, total_recipes: int) -> float:
        """Calculate overall compatibility strength"""
        
        # Normalize co-occurrence count
        co_strength = min(co_count / (total_recipes * 0.1), 1.0)  # Cap at 10% of recipes
        
        # Normalize PMI score (typical range -10 to +10)
        pmi_strength = max(0, min(pmi_score / 10.0, 1.0))
        
        # Combined strength
        return (co_strength * 0.4) + (pmi_strength * 0.6)
    
    def analyze_graph_metrics(self, G: nx.Graph) -> Dict[str, CompatibilityGraphNode]:
        """Calculate centrality metrics and clustering for graph nodes"""
        
        if not G.nodes():
            return {}
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
        closeness_centrality = nx.closeness_centrality(G)
        pagerank = nx.pagerank(G, weight='strength')
        
        # Detect communities/clusters
        try:
            communities = nx.community.greedy_modularity_communities(G, weight='strength')
            node_to_cluster = {}
            for i, community in enumerate(communities):
                for node in community:
                    node_to_cluster[node] = f"cluster_{i}"
        except:
            node_to_cluster = {}
        
        # Calculate clustering coefficients
        clustering = nx.clustering(G, weight='strength')
        
        # Create node objects
        nodes = {}
        for ingredient in G.nodes():
            node_data = G.nodes[ingredient]
            
            nodes[ingredient] = CompatibilityGraphNode(
                ingredient_name=ingredient,
                total_recipes=node_data.get('recipe_count', 0),
                cuisines=set(),  # Would be populated from recipe analysis
                categories=set(),  # Would be populated from ingredient classification
                degree_centrality=degree_centrality.get(ingredient, 0.0),
                betweenness_centrality=betweenness_centrality.get(ingredient, 0.0),
                closeness_centrality=closeness_centrality.get(ingredient, 0.0),
                pagerank=pagerank.get(ingredient, 0.0),
                cluster_id=node_to_cluster.get(ingredient),
                cluster_coefficient=clustering.get(ingredient, 0.0)
            )
        
        return nodes
    
    def find_ingredient_substitutes(self, G: nx.Graph, target_ingredient: str, 
                                  limit: int = 10) -> List[Tuple[str, float]]:
        """Find potential substitutes based on graph structure"""
        
        if target_ingredient not in G:
            return []
        
        # Get neighbors (direct connections)
        neighbors = list(G.neighbors(target_ingredient))
        
        # Calculate substitute scores based on:
        # 1. Shared neighbors (ingredients they both pair with)
        # 2. Similar centrality measures
        # 3. Edge weights to target ingredient
        
        substitute_scores = []
        
        target_neighbors = set(neighbors)
        target_centrality = nx.degree_centrality(G)[target_ingredient]
        
        for candidate in G.nodes():
            if candidate == target_ingredient:
                continue
            
            candidate_neighbors = set(G.neighbors(candidate))
            candidate_centrality = nx.degree_centrality(G)[candidate]
            
            # Jaccard similarity of neighbors
            if target_neighbors or candidate_neighbors:
                shared_neighbors = len(target_neighbors.intersection(candidate_neighbors))
                total_neighbors = len(target_neighbors.union(candidate_neighbors))
                neighbor_similarity = shared_neighbors / total_neighbors if total_neighbors > 0 else 0.0
            else:
                neighbor_similarity = 0.0
            
            # Centrality similarity
            centrality_similarity = 1.0 - abs(target_centrality - candidate_centrality)
            
            # Direct connection strength (if exists)
            if G.has_edge(target_ingredient, candidate):
                direct_strength = G[target_ingredient][candidate]['strength']
            else:
                direct_strength = 0.0
            
            # Combined substitute score
            substitute_score = (
                neighbor_similarity * 0.5 +
                centrality_similarity * 0.2 + 
                direct_strength * 0.3
            )
            
            substitute_scores.append((candidate, substitute_score))
        
        # Sort by score and return top results
        substitute_scores.sort(key=lambda x: x[1], reverse=True)
        return substitute_scores[:limit]


class RelationalAnalyzer:
    """Main analyzer for the relational layer"""
    
    def __init__(self, config: RelationalAnalysisConfig = None):
        self.config = config or RelationalAnalysisConfig()
        self.logger = logging.getLogger(__name__)
        
        # Sub-components
        self.data_processor = RecipeDataProcessor(self.config)
        self.graph_builder = CompatibilityGraphBuilder(self.config)
        
        # Data storage
        self.co_occurrence_matrices: Dict[str, CoOccurrenceMatrix] = {}
        self.compatibility_graphs: Dict[str, nx.Graph] = {}
        self.cuisine_specific_data: Dict[CuisineType, CoOccurrenceMatrix] = {}
        
        # Analysis cache
        self.analysis_cache: Dict[str, Any] = {}
        self.cache_expiry = timedelta(hours=24)
    
    async def analyze_recipe_dataset(self, dataset_path: str, 
                                   dataset_name: str = "default") -> CoOccurrenceMatrix:
        """Analyze a recipe dataset and store results"""
        
        self.logger.info(f"Analyzing recipe dataset: {dataset_name}")
        
        # Process the dataset
        matrix = await self.data_processor.process_recipe_dataset(dataset_path)
        
        # Store matrix
        self.co_occurrence_matrices[dataset_name] = matrix
        
        # Build compatibility graph
        graph = self.graph_builder.build_compatibility_graph(matrix)
        self.compatibility_graphs[dataset_name] = graph
        
        return matrix
    
    def create_ingredient_compatibility(self, ingredient_a: str, ingredient_b: str,
                                     matrix: CoOccurrenceMatrix) -> IngredientCompatibility:
        """Create compatibility relationship between two ingredients"""
        
        co_count = matrix.get_co_occurrence(ingredient_a, ingredient_b)
        pmi_score = matrix.calculate_pmi(ingredient_a, ingredient_b)
        
        count_a = matrix.ingredient_counts.get(ingredient_a, 0)
        count_b = matrix.ingredient_counts.get(ingredient_b, 0)
        
        # Determine compatibility level based on PMI score
        if pmi_score >= 3.0:
            compatibility_level = CompatibilityLevel.EXCELLENT
        elif pmi_score >= 2.0:
            compatibility_level = CompatibilityLevel.VERY_GOOD
        elif pmi_score >= 1.0:
            compatibility_level = CompatibilityLevel.GOOD
        elif pmi_score >= 0.0:
            compatibility_level = CompatibilityLevel.NEUTRAL
        elif pmi_score >= -1.0:
            compatibility_level = CompatibilityLevel.POOR
        else:
            compatibility_level = CompatibilityLevel.INCOMPATIBLE
        
        # Calculate confidence based on sample size
        total_possible = min(count_a, count_b)
        confidence = min(co_count / max(total_possible * 0.1, 1), 1.0) if total_possible > 0 else 0.0
        
        compatibility = IngredientCompatibility(
            ingredient_a=ingredient_a,
            ingredient_b=ingredient_b,
            compatibility_score=compatibility_level,
            co_occurrence_count=co_count,
            total_recipes_a=count_a,
            total_recipes_b=count_b,
            pmi_score=pmi_score,
            confidence_level=confidence,
            sample_size=co_count,
            data_sources=[DataSource.RECIPE1M]
        )
        
        return compatibility
    
    def build_relational_profile(self, ingredient_name: str, 
                               dataset_name: str = "default") -> RelationalProfile:
        """Build complete relational profile for an ingredient"""
        
        if dataset_name not in self.co_occurrence_matrices:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        matrix = self.co_occurrence_matrices[dataset_name]
        graph = self.compatibility_graphs.get(dataset_name)
        
        if ingredient_name not in matrix.ingredients:
            # Return empty profile
            return RelationalProfile(
                ingredient_id=ingredient_name,
                network_completeness=0.0,
                relationship_confidence=0.0
            )
        
        profile = RelationalProfile(ingredient_id=ingredient_name)
        
        # Get top pairings from PMI analysis
        top_pairings = matrix.get_top_pairings(ingredient_name, limit=50)
        
        # Create compatibility relationships
        for other_ingredient, pmi_score in top_pairings:
            if pmi_score > self.config.pmi_threshold:
                compatibility = self.create_ingredient_compatibility(
                    ingredient_name, other_ingredient, matrix
                )
                profile.add_compatibility(other_ingredient, compatibility)
        
        # Add graph-based information if available
        if graph and ingredient_name in graph:
            node_data = self.graph_builder.analyze_graph_metrics(graph)
            node = node_data.get(ingredient_name)
            
            if node:
                profile.centrality_score = node.degree_centrality
                profile.cluster_assignments = [node.cluster_id] if node.cluster_id else []
                
                # Find substitutes
                substitutes = self.graph_builder.find_ingredient_substitutes(
                    graph, ingredient_name, limit=10
                )
                profile.primary_substitutes = [(sub, score) for sub, score in substitutes if score > 0.7]
                profile.secondary_substitutes = [(sub, score) for sub, score in substitutes if 0.4 <= score <= 0.7]
        
        # Calculate quality metrics
        profile.network_completeness = min(len(profile.compatible_ingredients) / 20.0, 1.0)
        
        if profile.compatible_ingredients:
            avg_confidence = statistics.mean([
                compat.confidence_level for compat in profile.compatible_ingredients.values()
            ])
            profile.relationship_confidence = avg_confidence
        
        profile.last_network_update = datetime.now()
        
        return profile
    
    def analyze_cuisine_patterns(self, recipes: List[Recipe]) -> Dict[CuisineType, CoOccurrenceMatrix]:
        """Analyze ingredient patterns by cuisine type"""
        
        cuisine_recipes = defaultdict(list)
        
        # Group recipes by cuisine
        for recipe in recipes:
            if recipe.cuisine:
                cuisine_recipes[recipe.cuisine].append(recipe)
        
        cuisine_matrices = {}
        
        # Build co-occurrence matrix for each cuisine
        for cuisine, cuisine_recipe_list in cuisine_recipes.items():
            if len(cuisine_recipe_list) >= self.config.min_cuisine_recipes:
                
                # Create temporary processor for this cuisine
                temp_processor = RecipeDataProcessor(self.config)
                matrix = temp_processor._build_cooccurrence_matrix(cuisine_recipe_list)
                matrix.cuisine_filter = cuisine
                
                cuisine_matrices[cuisine] = matrix
                self.logger.info(f"Built {cuisine.value} cuisine matrix: {len(matrix.ingredients)} ingredients")
        
        self.cuisine_specific_data = cuisine_matrices
        return cuisine_matrices
    
    def get_cuisine_compatibility(self, ingredient_a: str, ingredient_b: str, 
                                cuisine: CuisineType) -> Optional[float]:
        """Get compatibility score for specific cuisine"""
        
        if cuisine not in self.cuisine_specific_data:
            return None
        
        matrix = self.cuisine_specific_data[cuisine]
        
        if ingredient_a not in matrix.ingredients or ingredient_b not in matrix.ingredients:
            return None
        
        return matrix.calculate_pmi(ingredient_a, ingredient_b)
    
    def find_cultural_substitutes(self, ingredient: str, target_cuisine: CuisineType, 
                                limit: int = 5) -> List[Tuple[str, float]]:
        """Find culturally appropriate substitutes for an ingredient"""
        
        if target_cuisine not in self.cuisine_specific_data:
            return []
        
        cuisine_matrix = self.cuisine_specific_data[target_cuisine]
        
        if ingredient not in cuisine_matrix.ingredients:
            return []
        
        # Get top pairings in the target cuisine
        pairings = cuisine_matrix.get_top_pairings(ingredient, limit=limit * 2)
        
        # Filter for substitutes (ingredients that pair well but are culturally appropriate)
        substitutes = []
        for other_ingredient, pmi_score in pairings:
            if pmi_score > 1.0:  # Good compatibility
                substitutes.append((other_ingredient, pmi_score))
        
        return substitutes[:limit]
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analysis statistics"""
        
        total_ingredients = sum(len(matrix.ingredients) for matrix in self.co_occurrence_matrices.values())
        total_recipes = sum(matrix.total_recipes for matrix in self.co_occurrence_matrices.values())
        total_pairings = sum(
            np.count_nonzero(matrix.matrix) // 2 for matrix in self.co_occurrence_matrices.values()
        )
        
        return {
            'datasets_processed': len(self.co_occurrence_matrices),
            'total_ingredients': total_ingredients,
            'total_recipes': total_recipes,
            'total_pairings': total_pairings,
            'cuisines_analyzed': len(self.cuisine_specific_data),
            'graphs_built': len(self.compatibility_graphs),
            'cache_size': len(self.analysis_cache),
            'processor_stats': self.data_processor.stats
        }


# Factory functions and utilities

def create_relational_analyzer(min_co_occurrence: int = 3, pmi_threshold: float = 0.0) -> RelationalAnalyzer:
    """Create relational analyzer with custom parameters"""
    config = RelationalAnalysisConfig(
        min_co_occurrence=min_co_occurrence,
        pmi_threshold=pmi_threshold
    )
    return RelationalAnalyzer(config)


async def quick_compatibility_analysis(dataset_path: str) -> Dict[str, Any]:
    """Quick compatibility analysis for a dataset"""
    analyzer = create_relational_analyzer()
    matrix = await analyzer.analyze_recipe_dataset(dataset_path, "quick_analysis")
    
    return {
        'unique_ingredients': len(matrix.ingredients),
        'total_recipes': matrix.total_recipes,
        'top_ingredients': sorted(
            matrix.ingredient_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20],
        'sample_pairings': [
            (matrix.ingredients[i], matrix.ingredients[j], matrix.calculate_pmi(matrix.ingredients[i], matrix.ingredients[j]))
            for i in range(min(10, len(matrix.ingredients)))
            for j in range(i+1, min(i+6, len(matrix.ingredients)))
        ][:20]
    }


def calculate_pairing_strength(co_occurrence: int, total_recipes_a: int, 
                             total_recipes_b: int, total_recipes: int) -> Tuple[float, float]:
    """Calculate PMI and confidence for ingredient pairing"""
    
    if co_occurrence == 0 or total_recipes_a == 0 or total_recipes_b == 0:
        return float('-inf'), 0.0
    
    # Calculate PMI
    p_ab = co_occurrence / total_recipes
    p_a = total_recipes_a / total_recipes
    p_b = total_recipes_b / total_recipes
    
    pmi = math.log2(p_ab / (p_a * p_b))
    
    # Calculate confidence (how reliable is this pairing)
    confidence = min(co_occurrence / min(total_recipes_a, total_recipes_b), 1.0)
    
    return pmi, confidence


# Export key classes and functions
__all__ = [
    'CompatibilityAnalysisMethod', 'RecipeDataSource', 'CuisineType', 'SeasonalContext',
    'RecipeIngredient', 'Recipe', 'CoOccurrenceMatrix', 'CompatibilityGraphNode',
    'CompatibilityGraphEdge', 'RelationalAnalysisConfig', 'IngredientNormalizer',
    'RecipeDataProcessor', 'CompatibilityGraphBuilder', 'RelationalAnalyzer',
    'create_relational_analyzer', 'quick_compatibility_analysis', 'calculate_pairing_strength'
]