"""
Knowledge Graph Database Interface
=================================

Neo4j-based graph database implementation for storing and querying
millions of food entities with complex relationships and country-specific data.

Key Features:
- High-performance graph queries for food relationships
- Multi-dimensional indexing for fast searches
- Country-specific data partitioning
- Real-time relationship inference
- Distributed caching for performance
- ACID transactions for data integrity

Author: Wellomex AI Nutrition Team
Version: 1.0.0
"""

import asyncio
import json
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple, Any, AsyncGenerator
import aioredis
from neo4j import AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import Neo4jError
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from ..models.food_knowledge_models import (
    FoodEntity, FoodRelationship, CountrySpecificData, 
    NutrientInteraction, CookingTransformation, FoodCategory,
    MacroNutrient, MicroNutrient, AllergenType, DietaryRestriction,
    serialize_food_entity, deserialize_food_entity
)

logger = logging.getLogger(__name__)

class GraphDatabaseManager:
    """Neo4j graph database manager for food knowledge graph"""
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        redis_url: str = "redis://localhost:6379",
        cache_ttl: int = 3600
    ):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl
        
        self.driver = None
        self.redis_client = None
        self._indexes_created = False
        
    async def initialize(self):
        """Initialize database connections and create indexes"""
        try:
            # Initialize Neo4j
            self.driver = AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
                max_connection_lifetime=30 * 60,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            
            # Initialize Redis
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20
            )
            
            # Create database indexes and constraints
            await self._create_indexes_and_constraints()
            
            logger.info("Graph database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize graph database: {e}")
            raise

    async def close(self):
        """Close database connections"""
        if self.driver:
            await self.driver.close()
        if self.redis_client:
            await self.redis_client.close()

    async def _create_indexes_and_constraints(self):
        """Create database indexes and constraints for optimal performance"""
        if self._indexes_created:
            return
            
        async with self.driver.session() as session:
            # Food entity constraints and indexes
            constraints_and_indexes = [
                # Unique constraints
                "CREATE CONSTRAINT food_id_unique IF NOT EXISTS FOR (f:Food) REQUIRE f.food_id IS UNIQUE",
                "CREATE CONSTRAINT relationship_id_unique IF NOT EXISTS FOR (r:FoodRelationship) REQUIRE r.relationship_id IS UNIQUE",
                
                # Primary indexes
                "CREATE INDEX food_name_idx IF NOT EXISTS FOR (f:Food) ON (f.name)",
                "CREATE INDEX food_category_idx IF NOT EXISTS FOR (f:Food) ON (f.category)",
                "CREATE INDEX food_country_idx IF NOT EXISTS FOR (f:Food) ON (f.country_codes)",
                "CREATE INDEX food_allergens_idx IF NOT EXISTS FOR (f:Food) ON (f.allergens)",
                "CREATE INDEX food_dietary_restrictions_idx IF NOT EXISTS FOR (f:Food) ON (f.dietary_restrictions)",
                
                # Nutritional indexes
                "CREATE INDEX food_calories_idx IF NOT EXISTS FOR (f:Food) ON (f.calories)",
                "CREATE INDEX food_protein_idx IF NOT EXISTS FOR (f:Food) ON (f.protein)",
                "CREATE INDEX food_glycemic_index_idx IF NOT EXISTS FOR (f:Food) ON (f.glycemic_index)",
                
                # Relationship indexes
                "CREATE INDEX relationship_type_idx IF NOT EXISTS FOR (r:FoodRelationship) ON (r.relationship_type)",
                "CREATE INDEX relationship_strength_idx IF NOT EXISTS FOR (r:FoodRelationship) ON (r.strength)",
                "CREATE INDEX relationship_country_idx IF NOT EXISTS FOR (r:FoodRelationship) ON (r.country_specific)",
                
                # Country-specific indexes
                "CREATE INDEX country_data_country_idx IF NOT EXISTS FOR (c:CountryData) ON (c.country_code)",
                "CREATE INDEX country_data_availability_idx IF NOT EXISTS FOR (c:CountryData) ON (c.market_availability)",
                
                # Compound indexes for complex queries
                "CREATE INDEX food_category_country_idx IF NOT EXISTS FOR (f:Food) ON (f.category, f.country_codes)",
                "CREATE INDEX food_nutrition_profile_idx IF NOT EXISTS FOR (f:Food) ON (f.nutritional_profiles)",
                
                # Text indexes for search
                "CREATE FULLTEXT INDEX food_search_idx IF NOT EXISTS FOR (f:Food) ON EACH [f.name, f.common_names, f.scientific_name]",
                
                # Temporal indexes
                "CREATE INDEX food_updated_idx IF NOT EXISTS FOR (f:Food) ON (f.updated_at)",
                "CREATE INDEX food_created_idx IF NOT EXISTS FOR (f:Food) ON (f.created_at)"
            ]
            
            for constraint_or_index in constraints_and_indexes:
                try:
                    await session.run(constraint_or_index)
                    logger.debug(f"Created: {constraint_or_index}")
                except Neo4jError as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Failed to create constraint/index: {e}")
            
        self._indexes_created = True
        logger.info("Database indexes and constraints created successfully")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session with proper error handling"""
        session = self.driver.session()
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()

    async def create_food_entity(self, food: FoodEntity) -> bool:
        """Create a new food entity in the graph database"""
        cache_key = f"food:{food.food_id}"
        
        try:
            async with self.get_session() as session:
                # Prepare food data for Neo4j
                food_props = self._prepare_food_properties(food)
                
                # Create food node
                query = """
                CREATE (f:Food $props)
                SET f.created_at = datetime(),
                    f.updated_at = datetime()
                RETURN f.food_id as food_id
                """
                
                result = await session.run(query, props=food_props)
                record = await result.single()
                
                if record:
                    # Create country-specific data nodes
                    await self._create_country_data_nodes(session, food)
                    
                    # Cache the food entity
                    await self._cache_food_entity(food)
                    
                    logger.info(f"Created food entity: {food.name} ({food.food_id})")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to create food entity {food.name}: {e}")
            return False

    async def get_food_entity(self, food_id: str) -> Optional[FoodEntity]:
        """Retrieve a food entity by ID with caching"""
        cache_key = f"food:{food_id}"
        
        # Try cache first
        cached_data = await self.redis_client.get(cache_key)
        if cached_data:
            try:
                food_dict = json.loads(cached_data)
                return deserialize_food_entity(food_dict)
            except Exception as e:
                logger.warning(f"Failed to deserialize cached food data: {e}")
        
        # Query database
        try:
            async with self.get_session() as session:
                query = """
                MATCH (f:Food {food_id: $food_id})
                OPTIONAL MATCH (f)-[:HAS_COUNTRY_DATA]->(c:CountryData)
                RETURN f, collect(c) as country_data
                """
                
                result = await session.run(query, food_id=food_id)
                record = await result.single()
                
                if record:
                    food_entity = self._construct_food_entity(record)
                    await self._cache_food_entity(food_entity)
                    return food_entity
                    
        except Exception as e:
            logger.error(f"Failed to retrieve food entity {food_id}: {e}")
            
        return None

    async def search_foods(
        self,
        query: str = "",
        category: Optional[FoodCategory] = None,
        country_codes: Optional[List[str]] = None,
        allergen_free: Optional[List[AllergenType]] = None,
        dietary_compatible: Optional[List[DietaryRestriction]] = None,
        nutritional_profiles: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[FoodEntity]:
        """Advanced food search with multiple criteria"""
        cache_key = f"search:{hash(str([query, category, country_codes, allergen_free, dietary_compatible, nutritional_profiles, limit, offset]))}"
        
        # Try cache first
        cached_results = await self.redis_client.get(cache_key)
        if cached_results:
            try:
                results_data = json.loads(cached_results)
                return [deserialize_food_entity(data) for data in results_data]
            except Exception as e:
                logger.warning(f"Failed to deserialize cached search results: {e}")
        
        try:
            async with self.get_session() as session:
                # Build dynamic query
                cypher_query, params = self._build_search_query(
                    query, category, country_codes, allergen_free, 
                    dietary_compatible, nutritional_profiles, limit, offset
                )
                
                result = await session.run(cypher_query, **params)
                
                foods = []
                async for record in result:
                    food_entity = self._construct_food_entity(record)
                    foods.append(food_entity)
                
                # Cache results
                serialized_foods = [serialize_food_entity(food) for food in foods]
                await self.redis_client.setex(
                    cache_key, 
                    self.cache_ttl, 
                    json.dumps(serialized_foods)
                )
                
                return foods
                
        except Exception as e:
            logger.error(f"Failed to search foods: {e}")
            return []

    async def create_food_relationship(self, relationship: FoodRelationship) -> bool:
        """Create a relationship between two food entities"""
        try:
            async with self.get_session() as session:
                query = """
                MATCH (from:Food {food_id: $from_food_id})
                MATCH (to:Food {food_id: $to_food_id})
                CREATE (from)-[r:FOOD_RELATIONSHIP {
                    relationship_id: $relationship_id,
                    relationship_type: $relationship_type,
                    strength: $strength,
                    context: $context,
                    country_specific: $country_specific,
                    confidence: $confidence,
                    created_at: datetime(),
                    data_source: $data_source
                }]->(to)
                RETURN r.relationship_id as relationship_id
                """
                
                params = {
                    'from_food_id': relationship.from_food_id,
                    'to_food_id': relationship.to_food_id,
                    'relationship_id': relationship.relationship_id,
                    'relationship_type': relationship.relationship_type,
                    'strength': float(relationship.strength),
                    'context': relationship.context,
                    'country_specific': relationship.country_specific,
                    'confidence': float(relationship.confidence),
                    'data_source': relationship.data_source
                }
                
                result = await session.run(query, **params)
                record = await result.single()
                
                if record:
                    # Clear related caches
                    await self._invalidate_relationship_caches(relationship)
                    logger.info(f"Created relationship: {relationship.relationship_type} between {relationship.from_food_id} and {relationship.to_food_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to create food relationship: {e}")
            return False

    async def find_food_substitutes(
        self, 
        food_id: str, 
        country_code: Optional[str] = None,
        context: str = "cooking",
        min_strength: float = 0.5,
        limit: int = 10
    ) -> List[Tuple[FoodEntity, float]]:
        """Find substitute foods based on relationships and similarity"""
        cache_key = f"substitutes:{food_id}:{country_code}:{context}:{min_strength}:{limit}"
        
        cached_results = await self.redis_client.get(cache_key)
        if cached_results:
            try:
                results_data = json.loads(cached_results)
                return [(deserialize_food_entity(data[0]), data[1]) for data in results_data]
            except Exception as e:
                logger.warning(f"Failed to deserialize cached substitutes: {e}")
        
        try:
            async with self.get_session() as session:
                # Direct substitution relationships
                query = """
                MATCH (f:Food {food_id: $food_id})-[r:FOOD_RELATIONSHIP]->(s:Food)
                WHERE r.relationship_type = 'substitute' 
                AND r.strength >= $min_strength
                AND ($country_code IS NULL OR r.country_specific IS NULL OR r.country_specific = $country_code)
                AND ($context IS NULL OR r.context IS NULL OR r.context = $context)
                OPTIONAL MATCH (s)-[:HAS_COUNTRY_DATA]->(c:CountryData {country_code: $country_code})
                RETURN s, collect(c) as country_data, r.strength as strength
                ORDER BY r.strength DESC, COALESCE(c.market_availability, 0.5) DESC
                LIMIT $limit
                """
                
                params = {
                    'food_id': food_id,
                    'country_code': country_code,
                    'context': context,
                    'min_strength': min_strength,
                    'limit': limit
                }
                
                result = await session.run(query, **params)
                
                substitutes = []
                async for record in result:
                    food_entity = self._construct_food_entity(record)
                    strength = record['strength']
                    substitutes.append((food_entity, strength))
                
                # Cache results
                serialized_results = [(serialize_food_entity(food), strength) for food, strength in substitutes]
                await self.redis_client.setex(
                    cache_key, 
                    self.cache_ttl, 
                    json.dumps(serialized_results)
                )
                
                return substitutes
                
        except Exception as e:
            logger.error(f"Failed to find food substitutes for {food_id}: {e}")
            return []

    async def get_nutritionally_similar_foods(
        self,
        food_id: str,
        similarity_threshold: float = 0.8,
        country_code: Optional[str] = None,
        limit: int = 20
    ) -> List[Tuple[FoodEntity, float]]:
        """Find nutritionally similar foods using ML similarity"""
        try:
            # Get the target food's nutritional profile
            target_food = await self.get_food_entity(food_id)
            if not target_food or not target_food.macro_nutrients:
                return []
            
            # Get foods from the same category or related categories
            similar_category_foods = await self.search_foods(
                category=target_food.category,
                country_codes=[country_code] if country_code else None,
                limit=1000
            )
            
            if len(similar_category_foods) < 2:
                return []
            
            # Calculate nutritional similarity
            similarity_scores = await self._calculate_nutritional_similarity(
                target_food, similar_category_foods
            )
            
            # Filter and sort by similarity
            similar_foods = [
                (food, score) for food, score in similarity_scores 
                if score >= similarity_threshold and food.food_id != food_id
            ]
            
            similar_foods.sort(key=lambda x: x[1], reverse=True)
            return similar_foods[:limit]
            
        except Exception as e:
            logger.error(f"Failed to find nutritionally similar foods: {e}")
            return []

    async def get_country_food_availability(self, country_code: str) -> Dict[str, Any]:
        """Get comprehensive food availability data for a country"""
        cache_key = f"country_availability:{country_code}"
        
        cached_data = await self.redis_client.get(cache_key)
        if cached_data:
            try:
                return json.loads(cached_data)
            except Exception as e:
                logger.warning(f"Failed to deserialize cached country data: {e}")
        
        try:
            async with self.get_session() as session:
                query = """
                MATCH (f:Food)-[:HAS_COUNTRY_DATA]->(c:CountryData {country_code: $country_code})
                RETURN 
                    f.category as category,
                    count(f) as food_count,
                    avg(c.market_availability) as avg_availability,
                    collect(f.name)[0..10] as sample_foods,
                    avg(f.calories) as avg_calories,
                    avg(f.protein) as avg_protein
                ORDER BY food_count DESC
                """
                
                result = await session.run(query, country_code=country_code)
                
                availability_data = {
                    'country_code': country_code,
                    'categories': {},
                    'total_foods': 0,
                    'avg_availability': 0.0,
                    'generated_at': datetime.utcnow().isoformat()
                }
                
                total_foods = 0
                total_availability = 0.0
                
                async for record in result:
                    category = record['category']
                    food_count = record['food_count']
                    avg_availability = record['avg_availability'] or 0.0
                    
                    availability_data['categories'][category] = {
                        'food_count': food_count,
                        'avg_availability': avg_availability,
                        'sample_foods': record['sample_foods'] or [],
                        'avg_calories': record['avg_calories'],
                        'avg_protein': record['avg_protein']
                    }
                    
                    total_foods += food_count
                    total_availability += avg_availability * food_count
                
                availability_data['total_foods'] = total_foods
                availability_data['avg_availability'] = total_availability / total_foods if total_foods > 0 else 0.0
                
                # Cache results
                await self.redis_client.setex(
                    cache_key, 
                    self.cache_ttl * 4,  # Longer cache for country data
                    json.dumps(availability_data)
                )
                
                return availability_data
                
        except Exception as e:
            logger.error(f"Failed to get country food availability for {country_code}: {e}")
            return {}

    async def update_food_entity(self, food: FoodEntity) -> bool:
        """Update an existing food entity"""
        try:
            async with self.get_session() as session:
                food_props = self._prepare_food_properties(food)
                food_props['updated_at'] = datetime.utcnow().isoformat()
                
                query = """
                MATCH (f:Food {food_id: $food_id})
                SET f += $props
                RETURN f.food_id as food_id
                """
                
                result = await session.run(query, food_id=food.food_id, props=food_props)
                record = await result.single()
                
                if record:
                    # Update country-specific data
                    await self._update_country_data_nodes(session, food)
                    
                    # Clear caches
                    await self._invalidate_food_caches(food.food_id)
                    
                    logger.info(f"Updated food entity: {food.name} ({food.food_id})")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to update food entity {food.food_id}: {e}")
            return False

    async def delete_food_entity(self, food_id: str) -> bool:
        """Delete a food entity and all its relationships"""
        try:
            async with self.get_session() as session:
                query = """
                MATCH (f:Food {food_id: $food_id})
                OPTIONAL MATCH (f)-[r]-()
                DETACH DELETE f
                RETURN count(r) as deleted_relationships
                """
                
                result = await session.run(query, food_id=food_id)
                record = await result.single()
                
                if record is not None:
                    # Clear caches
                    await self._invalidate_food_caches(food_id)
                    
                    logger.info(f"Deleted food entity {food_id} with {record['deleted_relationships']} relationships")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to delete food entity {food_id}: {e}")
            return False

    # Helper methods
    
    def _prepare_food_properties(self, food: FoodEntity) -> Dict[str, Any]:
        """Prepare food properties for Neo4j storage"""
        props = {
            'food_id': food.food_id,
            'name': food.name,
            'scientific_name': food.scientific_name,
            'common_names': food.common_names,
            'category': food.category.value if food.category else None,
            'subcategories': food.subcategories,
            'food_group': food.food_group,
            'taxonomic_family': food.taxonomic_family,
            'allergens': [a.value for a in food.allergens],
            'dietary_restrictions': [d.value for d in food.dietary_restrictions],
            'nutritional_profiles': [p.value for p in food.nutritional_profiles],
            'country_codes': list(food.country_data.keys()),
            'data_sources': food.data_sources,
            'confidence_score': float(food.confidence_score),
            'verification_status': food.verification_status
        }
        
        # Add macro nutrients if available
        if food.macro_nutrients:
            macro = food.macro_nutrients
            props.update({
                'calories': float(macro.calories),
                'carbohydrates': float(macro.carbohydrates),
                'protein': float(macro.protein),
                'fat': float(macro.fat),
                'fiber': float(macro.fiber),
                'sugar': float(macro.sugar),
                'sodium': float(macro.sodium)
            })
        
        # Add key micronutrients
        if food.micro_nutrients:
            micro = food.micro_nutrients
            for nutrient in ['vitamin_c', 'vitamin_d', 'calcium', 'iron', 'potassium']:
                value = getattr(micro, nutrient, None)
                if value is not None:
                    props[nutrient] = float(value)
        
        # Add physical properties
        if food.glycemic_index:
            props['glycemic_index'] = food.glycemic_index
        if food.glycemic_load:
            props['glycemic_load'] = float(food.glycemic_load)
        if food.typical_serving_size:
            props['serving_size'] = float(food.typical_serving_size)
        
        return props

    async def _create_country_data_nodes(self, session: AsyncSession, food: FoodEntity):
        """Create country-specific data nodes"""
        for country_code, country_data in food.country_data.items():
            query = """
            MATCH (f:Food {food_id: $food_id})
            CREATE (c:CountryData {
                country_code: $country_code,
                local_name: $local_name,
                common_varieties: $common_varieties,
                market_availability: $market_availability,
                production_regions: $production_regions,
                import_sources: $import_sources,
                quality_grades: $quality_grades
            })
            CREATE (f)-[:HAS_COUNTRY_DATA]->(c)
            """
            
            params = {
                'food_id': food.food_id,
                'country_code': country_code,
                'local_name': country_data.local_name,
                'common_varieties': country_data.common_varieties,
                'market_availability': float(country_data.market_availability),
                'production_regions': country_data.production_regions,
                'import_sources': country_data.import_sources,
                'quality_grades': country_data.quality_grades
            }
            
            await session.run(query, **params)

    async def _update_country_data_nodes(self, session: AsyncSession, food: FoodEntity):
        """Update country-specific data nodes"""
        # Delete existing country data
        await session.run(
            "MATCH (f:Food {food_id: $food_id})-[:HAS_COUNTRY_DATA]->(c:CountryData) DETACH DELETE c",
            food_id=food.food_id
        )
        
        # Create new country data
        await self._create_country_data_nodes(session, food)

    def _construct_food_entity(self, record) -> FoodEntity:
        """Construct FoodEntity from Neo4j record"""
        food_node = record['f']
        country_data_nodes = record.get('country_data', [])
        
        # Create basic food entity
        food = FoodEntity(
            food_id=food_node['food_id'],
            name=food_node['name'],
            scientific_name=food_node.get('scientific_name'),
            common_names=food_node.get('common_names', [])
        )
        
        # Set category
        if food_node.get('category'):
            try:
                food.category = FoodCategory(food_node['category'])
            except ValueError:
                pass
        
        # Set nutritional data if available
        if food_node.get('calories') is not None:
            food.macro_nutrients = MacroNutrient(
                calories=Decimal(str(food_node.get('calories', 0))),
                carbohydrates=Decimal(str(food_node.get('carbohydrates', 0))),
                protein=Decimal(str(food_node.get('protein', 0))),
                fat=Decimal(str(food_node.get('fat', 0))),
                fiber=Decimal(str(food_node.get('fiber', 0))),
                sugar=Decimal(str(food_node.get('sugar', 0))),
                sodium=Decimal(str(food_node.get('sodium', 0)))
            )
        
        # Add country-specific data
        for country_node in country_data_nodes:
            if country_node:
                country_data = CountrySpecificData(
                    country_code=country_node['country_code'],
                    local_name=country_node['local_name'],
                    common_varieties=country_node.get('common_varieties', []),
                    market_availability=Decimal(str(country_node.get('market_availability', 1.0))),
                    production_regions=country_node.get('production_regions', []),
                    import_sources=country_node.get('import_sources', []),
                    quality_grades=country_node.get('quality_grades', [])
                )
                food.country_data[country_node['country_code']] = country_data
        
        return food

    def _build_search_query(
        self, 
        query: str, 
        category: Optional[FoodCategory], 
        country_codes: Optional[List[str]], 
        allergen_free: Optional[List[AllergenType]], 
        dietary_compatible: Optional[List[DietaryRestriction]], 
        nutritional_profiles: Optional[List[str]], 
        limit: int, 
        offset: int
    ) -> Tuple[str, Dict[str, Any]]:
        """Build dynamic Cypher query for food search"""
        
        conditions = []
        params = {'limit': limit, 'offset': offset}
        
        if query:
            conditions.append("(f.name CONTAINS $query OR ANY(name IN f.common_names WHERE name CONTAINS $query))")
            params['query'] = query
        
        if category:
            conditions.append("f.category = $category")
            params['category'] = category.value
        
        if country_codes:
            conditions.append("ANY(cc IN $country_codes WHERE cc IN f.country_codes)")
            params['country_codes'] = country_codes
        
        if allergen_free:
            allergen_values = [a.value for a in allergen_free]
            conditions.append("NONE(allergen IN $allergen_free WHERE allergen IN f.allergens)")
            params['allergen_free'] = allergen_values
        
        if dietary_compatible:
            diet_values = [d.value for d in dietary_compatible]
            conditions.append("ALL(diet IN $dietary_compatible WHERE diet IN f.dietary_restrictions)")
            params['dietary_compatible'] = diet_values
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        cypher_query = f"""
        MATCH (f:Food)
        {where_clause}
        OPTIONAL MATCH (f)-[:HAS_COUNTRY_DATA]->(c:CountryData)
        RETURN f, collect(c) as country_data
        ORDER BY f.confidence_score DESC, f.name ASC
        SKIP $offset LIMIT $limit
        """
        
        return cypher_query, params

    async def _calculate_nutritional_similarity(
        self, 
        target_food: FoodEntity, 
        candidate_foods: List[FoodEntity]
    ) -> List[Tuple[FoodEntity, float]]:
        """Calculate nutritional similarity using machine learning"""
        
        # Extract nutritional features
        features = []
        valid_foods = []
        
        # Target food features
        target_features = self._extract_nutritional_features(target_food)
        if target_features is None:
            return []
        
        # Candidate food features
        for food in candidate_foods:
            food_features = self._extract_nutritional_features(food)
            if food_features is not None:
                features.append(food_features)
                valid_foods.append(food)
        
        if not features:
            return []
        
        # Normalize features
        scaler = StandardScaler()
        all_features = np.array([target_features] + features)
        normalized_features = scaler.fit_transform(all_features)
        
        # Calculate cosine similarity
        target_normalized = normalized_features[0:1]
        candidates_normalized = normalized_features[1:]
        
        similarities = cosine_similarity(target_normalized, candidates_normalized)[0]
        
        # Return foods with similarity scores
        return list(zip(valid_foods, similarities))

    def _extract_nutritional_features(self, food: FoodEntity) -> Optional[List[float]]:
        """Extract nutritional features for ML similarity calculation"""
        if not food.macro_nutrients:
            return None
        
        macro = food.macro_nutrients
        micro = food.micro_nutrients
        
        features = [
            float(macro.calories),
            float(macro.carbohydrates),
            float(macro.protein),
            float(macro.fat),
            float(macro.fiber),
            float(macro.sugar),
            float(macro.sodium)
        ]
        
        # Add key micronutrients if available
        if micro:
            micronutrient_features = [
                float(micro.vitamin_c or 0),
                float(micro.vitamin_d or 0),
                float(micro.calcium or 0),
                float(micro.iron or 0),
                float(micro.potassium or 0)
            ]
            features.extend(micronutrient_features)
        else:
            features.extend([0.0] * 5)
        
        return features

    async def _cache_food_entity(self, food: FoodEntity):
        """Cache food entity in Redis"""
        cache_key = f"food:{food.food_id}"
        serialized_food = json.dumps(serialize_food_entity(food))
        await self.redis_client.setex(cache_key, self.cache_ttl, serialized_food)

    async def _invalidate_food_caches(self, food_id: str):
        """Invalidate all caches related to a food entity"""
        patterns = [
            f"food:{food_id}",
            f"substitutes:{food_id}:*",
            f"search:*{food_id}*"
        ]
        
        for pattern in patterns:
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)

    async def _invalidate_relationship_caches(self, relationship: FoodRelationship):
        """Invalidate caches related to food relationships"""
        patterns = [
            f"substitutes:{relationship.from_food_id}:*",
            f"substitutes:{relationship.to_food_id}:*"
        ]
        
        for pattern in patterns:
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)

# Graph analytics and network analysis
class FoodNetworkAnalyzer:
    """Advanced network analysis for food relationships"""
    
    def __init__(self, db_manager: GraphDatabaseManager):
        self.db_manager = db_manager
    
    async def build_food_network(self, country_code: Optional[str] = None) -> nx.DiGraph:
        """Build NetworkX graph from Neo4j data"""
        G = nx.DiGraph()
        
        try:
            async with self.db_manager.get_session() as session:
                # Get all foods and relationships
                query = """
                MATCH (f1:Food)-[r:FOOD_RELATIONSHIP]->(f2:Food)
                WHERE ($country_code IS NULL OR r.country_specific IS NULL OR r.country_specific = $country_code)
                RETURN f1.food_id as from_id, f1.name as from_name, f1.category as from_category,
                       f2.food_id as to_id, f2.name as to_name, f2.category as to_category,
                       r.relationship_type as rel_type, r.strength as strength
                """
                
                result = await session.run(query, country_code=country_code)
                
                async for record in result:
                    # Add nodes
                    G.add_node(record['from_id'], 
                              name=record['from_name'], 
                              category=record['from_category'])
                    G.add_node(record['to_id'], 
                              name=record['to_name'], 
                              category=record['to_category'])
                    
                    # Add edge
                    G.add_edge(record['from_id'], record['to_id'], 
                              relationship_type=record['rel_type'],
                              strength=record['strength'])
                
                return G
                
        except Exception as e:
            logger.error(f"Failed to build food network: {e}")
            return nx.DiGraph()
    
    async def find_food_communities(self, country_code: Optional[str] = None) -> Dict[str, List[str]]:
        """Detect food communities using network analysis"""
        G = await self.build_food_network(country_code)
        
        if G.number_of_nodes() == 0:
            return {}
        
        # Convert to undirected for community detection
        G_undirected = G.to_undirected()
        
        # Use Louvain algorithm for community detection
        try:
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.louvain_communities(G_undirected)
            
            community_dict = {}
            for i, community in enumerate(communities):
                community_dict[f"community_{i}"] = list(community)
            
            return community_dict
            
        except ImportError:
            logger.warning("NetworkX community detection not available")
            return {}
    
    async def calculate_food_centrality(self, country_code: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality measures for foods"""
        G = await self.build_food_network(country_code)
        
        if G.number_of_nodes() == 0:
            return {}
        
        centrality_measures = {}
        
        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(G)
            
            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(G)
            
            # Closeness centrality
            closeness_centrality = nx.closeness_centrality(G)
            
            # PageRank
            pagerank = nx.pagerank(G)
            
            # Combine measures
            for node_id in G.nodes():
                centrality_measures[node_id] = {
                    'degree': degree_centrality.get(node_id, 0),
                    'betweenness': betweenness_centrality.get(node_id, 0),
                    'closeness': closeness_centrality.get(node_id, 0),
                    'pagerank': pagerank.get(node_id, 0)
                }
                
            return centrality_measures
            
        except Exception as e:
            logger.error(f"Failed to calculate centrality measures: {e}")
            return {}