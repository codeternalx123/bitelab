"""
Food Knowledge Graph with Deep Learning
========================================

Comprehensive knowledge graph system that models:
- Food entities and their properties
- Health goals and therapeutic outcomes
- Medical conditions and contraindications
- Nutrient interactions and synergies
- Drug-food interactions
- Cultural and dietary contexts

Uses graph neural networks (GNN) to learn relationships and
can be enhanced by querying large LLMs for knowledge expansion.

Author: Wellomex AI Team
Date: November 2025
Version: 1.0.0
"""

import logging
import json
import pickle
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class EntityType(Enum):
    """Knowledge graph entity types"""
    FOOD = "food"
    NUTRIENT = "nutrient"
    HEALTH_GOAL = "health_goal"
    DISEASE = "disease"
    MEDICATION = "medication"
    SYMPTOM = "symptom"
    BIOMARKER = "biomarker"
    CUISINE = "cuisine"
    COOKING_METHOD = "cooking_method"


class RelationType(Enum):
    """Knowledge graph relationship types"""
    CONTAINS = "contains"  # Food contains nutrient
    BENEFITS = "benefits"  # Food/nutrient benefits goal
    MANAGES = "manages"  # Food/nutrient manages disease
    CONTRAINDICATES = "contraindicates"  # Food contraindicates condition
    INTERACTS_WITH = "interacts_with"  # Drug-food interaction
    SYNERGIZES_WITH = "synergizes_with"  # Nutrient synergy
    ANTAGONIZES = "antagonizes"  # Nutrient antagonism
    IMPROVES = "improves"  # Improves biomarker
    WORSENS = "worsens"  # Worsens biomarker
    PART_OF = "part_of"  # Cuisine/cultural context
    PREPARED_BY = "prepared_by"  # Cooking method


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Entity:
    """Knowledge graph entity"""
    id: str
    type: EntityType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[np.ndarray] = None
    
    # LLM-sourced knowledge
    llm_description: Optional[str] = None
    llm_updated: Optional[datetime] = None
    confidence_score: float = 0.0


@dataclass
class Relation:
    """Knowledge graph relationship"""
    source_id: str
    relation_type: RelationType
    target_id: str
    weight: float = 1.0  # Strength of relationship
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Evidence
    source: str = "system"  # system, llm, research, user
    confidence: float = 0.0
    citations: List[str] = field(default_factory=list)


@dataclass
class GraphQuery:
    """Query for knowledge graph"""
    entity_id: Optional[str] = None
    entity_type: Optional[EntityType] = None
    relation_type: Optional[RelationType] = None
    max_hops: int = 2
    min_confidence: float = 0.5


# ============================================================================
# GRAPH NEURAL NETWORK MODELS
# ============================================================================

class GraphConvLayer(nn.Module):
    """Graph Convolutional Layer for learning entity embeddings"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [N, in_features]
            adj: Adjacency matrix [N, N]
        
        Returns:
            Updated node features [N, out_features]
        """
        # Aggregate neighbor features
        aggregated = torch.matmul(adj, x)
        
        # Transform
        out = self.linear(aggregated)
        out = self.activation(out)
        
        return out


class FoodKnowledgeGNN(nn.Module):
    """
    Graph Neural Network for food knowledge graph
    
    Learns embeddings for foods, nutrients, health goals, and diseases
    that capture their relationships and therapeutic effects.
    """
    
    def __init__(
        self,
        num_entities: int,
        entity_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_relation_types: int = 11
    ):
        super().__init__()
        
        self.entity_dim = entity_dim
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, entity_dim)
        
        # Relation embeddings
        self.relation_embeddings = nn.Embedding(num_relation_types, entity_dim)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            GraphConvLayer(entity_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # Output layers for different tasks
        self.goal_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 55)  # 55+ health goals
        )
        
        self.disease_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 100)  # 100+ diseases
        )
        
        self.interaction_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"Initialized FoodKnowledgeGNN with {num_entities} entities")
    
    def forward(
        self,
        entity_ids: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            entity_ids: Entity IDs [batch_size]
            adj_matrix: Adjacency matrix [num_entities, num_entities]
        
        Returns:
            Dictionary with predictions
        """
        # Get initial embeddings
        x = self.entity_embeddings(entity_ids)
        
        # Apply graph convolutions
        for conv in self.conv_layers:
            x = conv(x, adj_matrix)
        
        # Make predictions
        goal_scores = self.goal_predictor(x)
        disease_scores = self.disease_predictor(x)
        
        return {
            "embeddings": x,
            "goal_scores": goal_scores,
            "disease_scores": disease_scores
        }
    
    def predict_interaction(
        self,
        entity1_id: torch.Tensor,
        entity2_id: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Predict interaction strength between two entities"""
        
        # Get embeddings
        emb1 = self.entity_embeddings(entity1_id)
        emb2 = self.entity_embeddings(entity2_id)
        
        # Apply convolutions
        for conv in self.conv_layers:
            emb1 = conv(emb1.unsqueeze(0), adj_matrix).squeeze(0)
            emb2 = conv(emb2.unsqueeze(0), adj_matrix).squeeze(0)
        
        # Predict interaction
        combined = torch.cat([emb1, emb2], dim=-1)
        interaction_score = self.interaction_predictor(combined)
        
        return interaction_score


# ============================================================================
# KNOWLEDGE GRAPH
# ============================================================================

class FoodKnowledgeGraph:
    """
    Comprehensive knowledge graph for food, health, and medicine
    
    Features:
    - Entity and relationship management
    - Graph neural network for learning
    - LLM integration for knowledge expansion
    - Multi-hop reasoning
    - Confidence-weighted relationships
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        
        # Graph storage
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        
        # Index structures
        self.entity_by_type: Dict[EntityType, Set[str]] = defaultdict(set)
        self.relations_by_source: Dict[str, List[Relation]] = defaultdict(list)
        self.relations_by_target: Dict[str, List[Relation]] = defaultdict(list)
        
        # Entity ID mapping
        self.entity_id_to_idx: Dict[str, int] = {}
        self.idx_to_entity_id: Dict[int, str] = {}
        
        # GNN model
        self.gnn_model: Optional[FoodKnowledgeGNN] = None
        
        # Initialize with core knowledge
        self._initialize_core_graph()
        
        logger.info("Initialized FoodKnowledgeGraph")
    
    # ========================================================================
    # ENTITY MANAGEMENT
    # ========================================================================
    
    def add_entity(
        self,
        entity_id: str,
        entity_type: EntityType,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
        llm_description: Optional[str] = None
    ) -> Entity:
        """Add entity to knowledge graph"""
        
        entity = Entity(
            id=entity_id,
            type=entity_type,
            name=name,
            properties=properties or {},
            llm_description=llm_description
        )
        
        self.entities[entity_id] = entity
        self.entity_by_type[entity_type].add(entity_id)
        
        # Update ID mapping
        if entity_id not in self.entity_id_to_idx:
            idx = len(self.entity_id_to_idx)
            self.entity_id_to_idx[entity_id] = idx
            self.idx_to_entity_id[idx] = entity_id
        
        return entity
    
    def add_relation(
        self,
        source_id: str,
        relation_type: RelationType,
        target_id: str,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
        source: str = "system",
        confidence: float = 0.8
    ) -> Relation:
        """Add relationship to knowledge graph"""
        
        relation = Relation(
            source_id=source_id,
            relation_type=relation_type,
            target_id=target_id,
            weight=weight,
            properties=properties or {},
            source=source,
            confidence=confidence
        )
        
        self.relations.append(relation)
        self.relations_by_source[source_id].append(relation)
        self.relations_by_target[target_id].append(relation)
        
        return relation
    
    # ========================================================================
    # GRAPH INITIALIZATION
    # ========================================================================
    
    def _initialize_core_graph(self):
        """Initialize graph with core nutritional knowledge"""
        
        # Add key nutrients
        nutrients = [
            ("protein", "Protein", {"unit": "g", "macro": True}),
            ("carbs", "Carbohydrates", {"unit": "g", "macro": True}),
            ("fat", "Fat", {"unit": "g", "macro": True}),
            ("fiber", "Dietary Fiber", {"unit": "g"}),
            ("omega3", "Omega-3 Fatty Acids", {"unit": "g"}),
            ("vitamin_d", "Vitamin D", {"unit": "mcg"}),
            ("calcium", "Calcium", {"unit": "mg"}),
            ("iron", "Iron", {"unit": "mg"}),
            ("potassium", "Potassium", {"unit": "mg"}),
            ("sodium", "Sodium", {"unit": "mg"})
        ]
        
        for nid, name, props in nutrients:
            self.add_entity(nid, EntityType.NUTRIENT, name, props)
        
        # Add health goals
        goals = [
            "weight_loss", "muscle_gain", "heart_health", "diabetes_control",
            "bone_health", "brain_health", "immune_support", "energy_boost",
            "anti_inflammatory", "digestive_health", "blood_pressure_control"
        ]
        
        for goal in goals:
            self.add_entity(
                goal,
                EntityType.HEALTH_GOAL,
                goal.replace("_", " ").title()
            )
        
        # Add common diseases
        diseases = [
            "type2_diabetes", "hypertension", "hyperlipidemia", "ckd",
            "heart_disease", "obesity", "osteoporosis", "ibs"
        ]
        
        for disease in diseases:
            self.add_entity(
                disease,
                EntityType.DISEASE,
                disease.replace("_", " ").title()
            )
        
        # Add sample foods with relationships
        self._add_sample_foods()
        
        logger.info(f"Initialized graph with {len(self.entities)} entities")
    
    def _add_sample_foods(self):
        """Add sample foods with nutritional relationships"""
        
        # Salmon
        salmon = self.add_entity(
            "salmon",
            EntityType.FOOD,
            "Salmon",
            {"category": "protein", "calories_per_100g": 208}
        )
        
        # Salmon relationships
        self.add_relation("salmon", RelationType.CONTAINS, "protein", weight=20.0)
        self.add_relation("salmon", RelationType.CONTAINS, "omega3", weight=2.3)
        self.add_relation("salmon", RelationType.BENEFITS, "heart_health", confidence=0.95)
        self.add_relation("salmon", RelationType.BENEFITS, "brain_health", confidence=0.90)
        self.add_relation("salmon", RelationType.MANAGES, "type2_diabetes", confidence=0.85)
        
        # Spinach
        spinach = self.add_entity(
            "spinach",
            EntityType.FOOD,
            "Spinach",
            {"category": "vegetable", "calories_per_100g": 23}
        )
        
        self.add_relation("spinach", RelationType.CONTAINS, "iron", weight=2.7)
        self.add_relation("spinach", RelationType.CONTAINS, "calcium", weight=99.0)
        self.add_relation("spinach", RelationType.BENEFITS, "bone_health", confidence=0.88)
        self.add_relation("spinach", RelationType.BENEFITS, "anti_inflammatory", confidence=0.82)
        
        # Blueberries
        blueberries = self.add_entity(
            "blueberries",
            EntityType.FOOD,
            "Blueberries",
            {"category": "fruit", "calories_per_100g": 57}
        )
        
        self.add_relation("blueberries", RelationType.CONTAINS, "fiber", weight=2.4)
        self.add_relation("blueberries", RelationType.BENEFITS, "brain_health", confidence=0.92)
        self.add_relation("blueberries", RelationType.BENEFITS, "anti_inflammatory", confidence=0.90)
        self.add_relation("blueberries", RelationType.IMPROVES, "cognitive_function", confidence=0.87)
    
    # ========================================================================
    # GRAPH NEURAL NETWORK
    # ========================================================================
    
    def build_gnn_model(self, hidden_dim: int = 256, num_layers: int = 3):
        """Build and initialize GNN model"""
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, GNN disabled")
            return
        
        num_entities = len(self.entities)
        num_relation_types = len(RelationType)
        
        self.gnn_model = FoodKnowledgeGNN(
            num_entities=num_entities,
            entity_dim=self.embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_relation_types=num_relation_types
        )
        
        logger.info(f"Built GNN model with {num_entities} entities")
    
    def get_adjacency_matrix(self) -> torch.Tensor:
        """Build adjacency matrix from relations"""
        
        if not TORCH_AVAILABLE:
            return None
        
        num_entities = len(self.entities)
        adj = torch.zeros((num_entities, num_entities))
        
        for relation in self.relations:
            src_idx = self.entity_id_to_idx.get(relation.source_id)
            tgt_idx = self.entity_id_to_idx.get(relation.target_id)
            
            if src_idx is not None and tgt_idx is not None:
                adj[src_idx, tgt_idx] = relation.weight * relation.confidence
                adj[tgt_idx, src_idx] = relation.weight * relation.confidence  # Undirected
        
        # Normalize
        degree = adj.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # Avoid division by zero
        adj = adj / degree
        
        return adj
    
    def train_gnn(
        self,
        num_epochs: int = 100,
        learning_rate: float = 0.001
    ):
        """Train GNN model on graph structure"""
        
        if not TORCH_AVAILABLE or self.gnn_model is None:
            logger.warning("GNN not available")
            return
        
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=learning_rate)
        adj_matrix = self.get_adjacency_matrix()
        
        self.gnn_model.train()
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Sample batch of entities
            entity_ids = torch.randint(0, len(self.entities), (32,))
            
            # Forward pass
            outputs = self.gnn_model(entity_ids, adj_matrix)
            
            # Reconstruction loss (predict neighbors)
            loss = self._compute_reconstruction_loss(outputs, adj_matrix, entity_ids)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        
        logger.info("GNN training complete")
    
    def _compute_reconstruction_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        adj_matrix: torch.Tensor,
        entity_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss for link prediction"""
        
        embeddings = outputs["embeddings"]
        
        # Predict edges from embeddings
        predicted = torch.matmul(embeddings, embeddings.t())
        
        # Get true edges
        true_edges = adj_matrix[entity_ids][:, entity_ids]
        
        # Binary cross entropy
        loss = F.binary_cross_entropy_with_logits(predicted, true_edges)
        
        return loss
    
    # ========================================================================
    # QUERYING
    # ========================================================================
    
    def query(self, query: GraphQuery) -> List[Tuple[Entity, List[Relation]]]:
        """Query knowledge graph"""
        
        results = []
        
        # Filter by entity type
        if query.entity_type:
            entity_ids = self.entity_by_type[query.entity_type]
        elif query.entity_id:
            entity_ids = [query.entity_id]
        else:
            entity_ids = list(self.entities.keys())
        
        # For each entity, get relationships
        for eid in entity_ids:
            entity = self.entities[eid]
            relations = self._get_relations_for_entity(
                eid,
                query.relation_type,
                query.max_hops,
                query.min_confidence
            )
            
            if relations:
                results.append((entity, relations))
        
        return results
    
    def _get_relations_for_entity(
        self,
        entity_id: str,
        relation_type: Optional[RelationType],
        max_hops: int,
        min_confidence: float
    ) -> List[Relation]:
        """Get relations for specific entity"""
        
        relations = self.relations_by_source.get(entity_id, [])
        
        # Filter by type
        if relation_type:
            relations = [r for r in relations if r.relation_type == relation_type]
        
        # Filter by confidence
        relations = [r for r in relations if r.confidence >= min_confidence]
        
        # TODO: Implement multi-hop traversal
        
        return relations
    
    def get_food_for_goal(
        self,
        health_goal: str,
        top_k: int = 10,
        min_confidence: float = 0.7
    ) -> List[Tuple[Entity, float]]:
        """Get foods that benefit a specific health goal"""
        
        results = []
        
        # Find all foods
        food_ids = self.entity_by_type[EntityType.FOOD]
        
        for food_id in food_ids:
            # Check if food benefits this goal
            relations = self.relations_by_source.get(food_id, [])
            
            for rel in relations:
                if (rel.relation_type == RelationType.BENEFITS and
                    rel.target_id == health_goal and
                    rel.confidence >= min_confidence):
                    
                    food = self.entities[food_id]
                    score = rel.confidence * rel.weight
                    results.append((food, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def get_contraindicated_foods(
        self,
        disease: str,
        medication: Optional[str] = None
    ) -> List[Tuple[Entity, str]]:
        """Get foods contraindicated for disease/medication"""
        
        contraindicated = []
        
        # Check disease contraindications
        for rel in self.relations:
            if (rel.relation_type == RelationType.CONTRAINDICATES and
                rel.target_id == disease):
                
                food = self.entities.get(rel.source_id)
                if food and food.type == EntityType.FOOD:
                    contraindicated.append((food, f"Contraindicates {disease}"))
        
        # Check medication interactions
        if medication:
            for rel in self.relations:
                if (rel.relation_type == RelationType.INTERACTS_WITH and
                    (rel.source_id == medication or rel.target_id == medication)):
                    
                    other_id = rel.target_id if rel.source_id == medication else rel.source_id
                    entity = self.entities.get(other_id)
                    
                    if entity and entity.type == EntityType.FOOD:
                        contraindicated.append((entity, f"Interacts with {medication}"))
        
        return contraindicated
    
    # ========================================================================
    # LLM INTEGRATION
    # ========================================================================
    
    async def expand_knowledge_from_llm(
        self,
        entity_id: str,
        llm_client: Any,
        model: str = "gpt-4-turbo-preview"
    ) -> Dict[str, Any]:
        """
        Query LLM for additional knowledge about entity
        
        This can be used to:
        - Get detailed descriptions
        - Discover new relationships
        - Validate existing knowledge
        - Expand cultural/contextual information
        """
        
        entity = self.entities.get(entity_id)
        if not entity:
            return {"error": "Entity not found"}
        
        # Build prompt
        prompt = f"""You are a nutrition and medical knowledge expert. Provide detailed information about: {entity.name}

Please provide:
1. Brief description (2-3 sentences)
2. Key nutritional components
3. Health benefits (list specific health goals)
4. Medical contraindications (diseases or conditions to avoid)
5. Drug interactions (if any)
6. Synergistic foods (foods that work well together)
7. Cultural significance

Format as JSON with keys: description, nutrients, benefits, contraindications, interactions, synergies, cultural_notes"""
        
        try:
            # Call LLM (assuming OpenAI client)
            response = await llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a nutrition expert. Respond in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse response
            knowledge = json.loads(response.choices[0].message.content)
            
            # Update entity
            entity.llm_description = knowledge.get("description")
            entity.llm_updated = datetime.now()
            
            # Add new relationships discovered by LLM
            self._integrate_llm_knowledge(entity_id, knowledge)
            
            logger.info(f"Expanded knowledge for {entity.name} from LLM")
            
            return knowledge
            
        except Exception as e:
            logger.error(f"LLM knowledge expansion error: {e}")
            return {"error": str(e)}
    
    def _integrate_llm_knowledge(self, entity_id: str, knowledge: Dict[str, Any]):
        """Integrate LLM-sourced knowledge into graph"""
        
        # Add health benefits as relationships
        benefits = knowledge.get("benefits", [])
        for benefit in benefits:
            # Create goal entity if doesn't exist
            goal_id = benefit.lower().replace(" ", "_")
            if goal_id not in self.entities:
                self.add_entity(goal_id, EntityType.HEALTH_GOAL, benefit)
            
            # Add relationship
            self.add_relation(
                entity_id,
                RelationType.BENEFITS,
                goal_id,
                source="llm",
                confidence=0.75  # Lower confidence for LLM-sourced
            )
        
        # Add contraindications
        contraindications = knowledge.get("contraindications", [])
        for contra in contraindications:
            disease_id = contra.lower().replace(" ", "_")
            if disease_id not in self.entities:
                self.add_entity(disease_id, EntityType.DISEASE, contra)
            
            self.add_relation(
                entity_id,
                RelationType.CONTRAINDICATES,
                disease_id,
                source="llm",
                confidence=0.70
            )
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def save(self, filepath: str):
        """Save knowledge graph to disk"""
        data = {
            "entities": {eid: {
                "type": e.type.value,
                "name": e.name,
                "properties": e.properties,
                "llm_description": e.llm_description
            } for eid, e in self.entities.items()},
            
            "relations": [{
                "source_id": r.source_id,
                "relation_type": r.relation_type.value,
                "target_id": r.target_id,
                "weight": r.weight,
                "properties": r.properties,
                "source": r.source,
                "confidence": r.confidence
            } for r in self.relations]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved knowledge graph to {filepath}")
    
    def load(self, filepath: str):
        """Load knowledge graph from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load entities
        for eid, edata in data["entities"].items():
            self.add_entity(
                eid,
                EntityType(edata["type"]),
                edata["name"],
                edata.get("properties"),
                edata.get("llm_description")
            )
        
        # Load relations
        for rdata in data["relations"]:
            self.add_relation(
                rdata["source_id"],
                RelationType(rdata["relation_type"]),
                rdata["target_id"],
                rdata.get("weight", 1.0),
                rdata.get("properties"),
                rdata.get("source", "system"),
                rdata.get("confidence", 0.8)
            )
        
        logger.info(f"Loaded knowledge graph from {filepath}")


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Global knowledge graph instance
_knowledge_graph = None

def get_knowledge_graph() -> FoodKnowledgeGraph:
    """Get global knowledge graph instance"""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = FoodKnowledgeGraph()
    return _knowledge_graph
