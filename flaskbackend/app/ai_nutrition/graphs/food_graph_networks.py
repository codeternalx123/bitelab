"""
Graph Neural Networks for Food & Nutrition
===========================================

GNNs for modeling relationships between foods, ingredients, recipes,
and nutritional components.

Features:
1. Food knowledge graph construction
2. Graph Convolutional Networks (GCN)
3. Graph Attention Networks (GAT)
4. Recipe-ingredient-nutrition graphs
5. Food substitution recommendations
6. Ingredient compatibility prediction
7. Nutrient interaction modeling
8. Graph-based food search

Performance Targets:
- Process graphs with 10,000+ nodes
- Inference: <50ms per query
- Recommendation accuracy: >85%
- Support dynamic graph updates
- Multi-hop reasoning (1-5 hops)

Author: Wellomex AI Team
Date: November 2025
Version: 5.0.0
"""

import time
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from datetime import datetime
from collections import defaultdict
import json

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class NodeType(Enum):
    """Graph node types"""
    FOOD = "food"
    INGREDIENT = "ingredient"
    NUTRIENT = "nutrient"
    RECIPE = "recipe"
    CUISINE = "cuisine"
    ALLERGEN = "allergen"


class EdgeType(Enum):
    """Graph edge types"""
    CONTAINS = "contains"  # Recipe contains ingredient
    PROVIDES = "provides"  # Food provides nutrient
    SIMILAR_TO = "similar_to"  # Food similar to food
    SUBSTITUTES = "substitutes"  # Ingredient substitutes ingredient
    CONFLICTS = "conflicts"  # Nutrient conflicts with nutrient
    BELONGS_TO = "belongs_to"  # Food belongs to cuisine


@dataclass
class GraphConfig:
    """Graph neural network configuration"""
    # Node features
    node_feature_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 128
    
    # GNN architecture
    num_layers: int = 3
    num_heads: int = 8  # For GAT
    dropout: float = 0.1
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 100
    
    # Graph construction
    similarity_threshold: float = 0.7
    max_neighbors: int = 50


# ============================================================================
# GRAPH STRUCTURE
# ============================================================================

@dataclass
class Node:
    """Graph node"""
    id: str
    type: NodeType
    features: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """Graph edge"""
    source: str
    target: str
    type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class FoodKnowledgeGraph:
    """
    Food Knowledge Graph
    
    Stores relationships between foods, ingredients, nutrients, etc.
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        
        # Adjacency lists by edge type
        self.adjacency: Dict[EdgeType, Dict[str, List[str]]] = {
            edge_type: defaultdict(list)
            for edge_type in EdgeType
        }
        
        # Reverse adjacency
        self.reverse_adjacency: Dict[EdgeType, Dict[str, List[str]]] = {
            edge_type: defaultdict(list)
            for edge_type in EdgeType
        }
        
        logger.info("Food Knowledge Graph initialized")
    
    def add_node(self, node: Node):
        """Add node to graph"""
        self.nodes[node.id] = node
    
    def add_edge(self, edge: Edge):
        """Add edge to graph"""
        self.edges.append(edge)
        self.adjacency[edge.type][edge.source].append(edge.target)
        self.reverse_adjacency[edge.type][edge.target].append(edge.source)
    
    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        reverse: bool = False
    ) -> List[str]:
        """Get neighbors of a node"""
        if edge_type:
            adj = self.reverse_adjacency if reverse else self.adjacency
            return adj[edge_type].get(node_id, [])
        else:
            # All neighbors regardless of edge type
            neighbors = set()
            adj = self.reverse_adjacency if reverse else self.adjacency
            
            for edge_type in EdgeType:
                neighbors.update(adj[edge_type].get(node_id, []))
            
            return list(neighbors)
    
    def get_subgraph(
        self,
        node_ids: Set[str],
        k_hop: int = 1
    ) -> Tuple[Set[str], List[Edge]]:
        """Extract k-hop subgraph around nodes"""
        current_nodes = set(node_ids)
        all_nodes = set(node_ids)
        
        # Expand k hops
        for _ in range(k_hop):
            next_nodes = set()
            for node_id in current_nodes:
                neighbors = self.get_neighbors(node_id)
                next_nodes.update(neighbors)
            
            current_nodes = next_nodes - all_nodes
            all_nodes.update(next_nodes)
        
        # Get edges within subgraph
        subgraph_edges = [
            edge for edge in self.edges
            if edge.source in all_nodes and edge.target in all_nodes
        ]
        
        return all_nodes, subgraph_edges
    
    def to_adjacency_matrix(
        self,
        node_ids: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """Convert to adjacency matrix"""
        if node_ids is None:
            node_ids = list(self.nodes.keys())
        
        n = len(node_ids)
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        adj_matrix = np.zeros((n, n))
        
        for edge in self.edges:
            if edge.source in node_to_idx and edge.target in node_to_idx:
                i = node_to_idx[edge.source]
                j = node_to_idx[edge.target]
                adj_matrix[i, j] = edge.weight
        
        return adj_matrix, node_to_idx
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        node_types = defaultdict(int)
        edge_types = defaultdict(int)
        
        for node in self.nodes.values():
            node_types[node.type.value] += 1
        
        for edge in self.edges:
            edge_types[edge.type.value] += 1
        
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'node_types': dict(node_types),
            'edge_types': dict(edge_types),
            'avg_degree': len(self.edges) / len(self.nodes) if self.nodes else 0
        }


# ============================================================================
# GRAPH CONVOLUTIONAL NETWORK
# ============================================================================

class GraphConvLayer(nn.Module):
    """Graph Convolutional Layer"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Aggregate neighbor features
        aggregated = torch.matmul(adj, x)
        
        # Transform
        out = self.linear(aggregated)
        
        return out


class GCN(nn.Module):
    """
    Graph Convolutional Network
    
    Multi-layer GCN for node embedding and graph reasoning.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GraphConvLayer(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GraphConvLayer(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(GraphConvLayer(hidden_dim, output_dim))
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass"""
        # Normalize adjacency matrix
        adj_norm = self._normalize_adjacency(adj)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_norm)
            
            # ReLU and dropout for all but last layer
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def _normalize_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        """Normalize adjacency matrix with self-loops"""
        # Add self-loops
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Degree matrix
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        
        # D^{-1/2} A D^{-1/2}
        adj_norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        
        return adj_norm


# ============================================================================
# GRAPH ATTENTION NETWORK
# ============================================================================

class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.out_features = out_features
        self.head_dim = out_features // num_heads
        
        # Linear transformations
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # Attention mechanism
        self.a = nn.Parameter(torch.zeros(1, num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Updated node features [num_nodes, out_features]
        """
        num_nodes = x.size(0)
        
        # Linear transformation
        h = self.W(x)  # [num_nodes, out_features]
        h = h.view(num_nodes, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        
        # Attention coefficients
        # Concatenate features of source and target nodes
        h_i = h.unsqueeze(1).repeat(1, num_nodes, 1, 1)  # [num_nodes, num_nodes, num_heads, head_dim]
        h_j = h.unsqueeze(0).repeat(num_nodes, 1, 1, 1)  # [num_nodes, num_nodes, num_heads, head_dim]
        
        h_cat = torch.cat([h_i, h_j], dim=-1)  # [num_nodes, num_nodes, num_heads, 2*head_dim]
        
        # Compute attention scores
        e = self.leaky_relu((h_cat * self.a).sum(dim=-1))  # [num_nodes, num_nodes, num_heads]
        
        # Mask attention based on adjacency
        mask = (adj == 0)
        e = e.masked_fill(mask.unsqueeze(-1), float('-inf'))
        
        # Softmax
        attention = F.softmax(e, dim=1)  # [num_nodes, num_nodes, num_heads]
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        
        # Aggregate
        h_prime = torch.einsum('ijk,jkl->ikl', attention, h)  # [num_nodes, num_heads, head_dim]
        h_prime = h_prime.reshape(num_nodes, -1)  # [num_nodes, out_features]
        
        return h_prime


class GAT(nn.Module):
    """
    Graph Attention Network
    
    Uses attention mechanism for neighbor aggregation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Layers
        self.gats = nn.ModuleList()
        
        # First layer
        self.gats.append(GraphAttentionLayer(input_dim, hidden_dim, num_heads, dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gats.append(GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout))
        
        # Output layer (single head)
        self.gats.append(GraphAttentionLayer(hidden_dim, output_dim, 1, dropout))
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass"""
        for i, gat in enumerate(self.gats):
            x = gat(x, adj)
            
            # ELU for all but last layer
            if i < self.num_layers - 1:
                x = F.elu(x)
        
        return x


# ============================================================================
# FOOD RELATIONSHIP PREDICTOR
# ============================================================================

class FoodRelationshipPredictor:
    """
    Predict relationships between foods using GNN
    
    Applications: substitution, compatibility, similarity
    """
    
    def __init__(
        self,
        graph: FoodKnowledgeGraph,
        config: GraphConfig,
        use_gat: bool = True
    ):
        self.graph = graph
        self.config = config
        
        # Build GNN model
        if use_gat:
            self.model = GAT(
                config.node_feature_dim,
                config.hidden_dim,
                config.output_dim,
                config.num_layers,
                config.num_heads,
                config.dropout
            )
        else:
            self.model = GCN(
                config.node_feature_dim,
                config.hidden_dim,
                config.output_dim,
                config.num_layers,
                config.dropout
            )
        
        self.optimizer = None
        
        # Node embeddings cache
        self.node_embeddings: Optional[Dict[str, np.ndarray]] = None
        
        logger.info(f"Food Relationship Predictor initialized ({'GAT' if use_gat else 'GCN'})")
    
    def train(
        self,
        train_edges: List[Tuple[str, str, bool]],  # (source, target, is_related)
        val_edges: Optional[List[Tuple[str, str, bool]]] = None
    ):
        """Train relationship predictor"""
        self.model.train()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Prepare graph data
        node_ids = list(self.graph.nodes.keys())
        features = np.stack([self.graph.nodes[nid].features for nid in node_ids])
        adj_matrix, node_to_idx = self.graph.to_adjacency_matrix(node_ids)
        
        features = torch.FloatTensor(features)
        adj = torch.FloatTensor(adj_matrix)
        
        logger.info(f"Training on {len(train_edges)} edges")
        
        for epoch in range(self.config.num_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            node_embeddings = self.model(features, adj)
            
            # Compute loss on training edges
            loss = 0
            for source_id, target_id, is_related in train_edges:
                if source_id not in node_to_idx or target_id not in node_to_idx:
                    continue
                
                i = node_to_idx[source_id]
                j = node_to_idx[target_id]
                
                # Dot product similarity
                similarity = (node_embeddings[i] * node_embeddings[j]).sum()
                
                # Binary cross-entropy
                target = torch.tensor(1.0 if is_related else 0.0)
                loss += F.binary_cross_entropy_with_logits(similarity, target)
            
            loss = loss / len(train_edges)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} - Loss: {loss.item():.4f}")
        
        logger.info("Training complete")
    
    def compute_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute embeddings for all nodes"""
        self.model.eval()
        
        # Prepare graph data
        node_ids = list(self.graph.nodes.keys())
        features = np.stack([self.graph.nodes[nid].features for nid in node_ids])
        adj_matrix, node_to_idx = self.graph.to_adjacency_matrix(node_ids)
        
        features = torch.FloatTensor(features)
        adj = torch.FloatTensor(adj_matrix)
        
        with torch.no_grad():
            embeddings = self.model(features, adj)
        
        # Convert to dict
        self.node_embeddings = {
            node_id: embeddings[idx].numpy()
            for node_id, idx in node_to_idx.items()
        }
        
        return self.node_embeddings
    
    def predict_relationship(
        self,
        source_id: str,
        target_id: str
    ) -> float:
        """Predict relationship strength between two nodes"""
        if self.node_embeddings is None:
            self.compute_embeddings()
        
        if source_id not in self.node_embeddings or target_id not in self.node_embeddings:
            return 0.0
        
        # Cosine similarity
        source_emb = self.node_embeddings[source_id]
        target_emb = self.node_embeddings[target_id]
        
        similarity = np.dot(source_emb, target_emb) / (
            np.linalg.norm(source_emb) * np.linalg.norm(target_emb) + 1e-8
        )
        
        return float(similarity)
    
    def find_similar_foods(
        self,
        food_id: str,
        top_k: int = 10,
        node_type: Optional[NodeType] = None
    ) -> List[Tuple[str, float]]:
        """Find most similar foods"""
        if self.node_embeddings is None:
            self.compute_embeddings()
        
        if food_id not in self.node_embeddings:
            return []
        
        # Compute similarities
        food_emb = self.node_embeddings[food_id]
        similarities = []
        
        for node_id, node_emb in self.node_embeddings.items():
            if node_id == food_id:
                continue
            
            # Filter by node type
            if node_type and self.graph.nodes[node_id].type != node_type:
                continue
            
            similarity = np.dot(food_emb, node_emb) / (
                np.linalg.norm(food_emb) * np.linalg.norm(node_emb) + 1e-8
            )
            
            similarities.append((node_id, float(similarity)))
        
        # Sort and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def recommend_substitutes(
        self,
        ingredient_id: str,
        constraints: Optional[Set[str]] = None,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Recommend ingredient substitutes"""
        # Find similar ingredients
        similar = self.find_similar_foods(
            ingredient_id,
            top_k=top_k * 2,  # Get more candidates
            node_type=NodeType.INGREDIENT
        )
        
        # Filter by constraints (e.g., allergens)
        if constraints:
            filtered = []
            for node_id, score in similar:
                node = self.graph.nodes[node_id]
                allergens = node.metadata.get('allergens', set())
                
                if not allergens & constraints:
                    filtered.append((node_id, score))
                
                if len(filtered) >= top_k:
                    break
            
            return filtered
        
        return similar[:top_k]


# ============================================================================
# TESTING
# ============================================================================

def test_gnn():
    """Test graph neural networks"""
    print("=" * 80)
    print("GRAPH NEURAL NETWORKS - TEST")
    print("=" * 80)
    
    if not TORCH_AVAILABLE or not NUMPY_AVAILABLE:
        print("❌ Required packages not available")
        return
    
    # Create graph
    graph = FoodKnowledgeGraph()
    
    print("\n✓ Knowledge graph created")
    
    # Add nodes
    for i in range(20):
        node = Node(
            id=f"food_{i}",
            type=NodeType.FOOD,
            features=np.random.randn(64),
            metadata={'name': f'Food {i}'}
        )
        graph.add_node(node)
    
    # Add edges
    for i in range(50):
        source = f"food_{np.random.randint(0, 20)}"
        target = f"food_{np.random.randint(0, 20)}"
        
        if source != target:
            edge = Edge(
                source=source,
                target=target,
                type=EdgeType.SIMILAR_TO,
                weight=np.random.rand()
            )
            graph.add_edge(edge)
    
    stats = graph.get_statistics()
    print(f"\n✓ Graph built:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Avg degree: {stats['avg_degree']:.2f}")
    
    # Test GCN
    print("\n" + "="*80)
    print("Test: Graph Convolutional Network")
    print("="*80)
    
    config = GraphConfig(
        node_feature_dim=64,
        hidden_dim=128,
        output_dim=64,
        num_layers=2,
        num_epochs=5
    )
    
    gcn = GCN(
        input_dim=config.node_feature_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        num_layers=config.num_layers
    )
    
    # Prepare data
    node_ids = list(graph.nodes.keys())
    features = np.stack([graph.nodes[nid].features for nid in node_ids])
    adj_matrix, _ = graph.to_adjacency_matrix(node_ids)
    
    features_tensor = torch.FloatTensor(features)
    adj_tensor = torch.FloatTensor(adj_matrix)
    
    # Forward pass
    embeddings = gcn(features_tensor, adj_tensor)
    
    print(f"✓ GCN forward pass:")
    print(f"  Input shape: {features_tensor.shape}")
    print(f"  Output shape: {embeddings.shape}")
    
    # Test GAT
    print("\n" + "="*80)
    print("Test: Graph Attention Network")
    print("="*80)
    
    gat = GAT(
        input_dim=config.node_feature_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        num_layers=2,
        num_heads=4
    )
    
    embeddings = gat(features_tensor, adj_tensor)
    
    print(f"✓ GAT forward pass:")
    print(f"  Input shape: {features_tensor.shape}")
    print(f"  Output shape: {embeddings.shape}")
    
    # Test relationship predictor
    print("\n" + "="*80)
    print("Test: Food Relationship Predictor")
    print("="*80)
    
    predictor = FoodRelationshipPredictor(graph, config, use_gat=True)
    
    # Create training data
    train_edges = [
        (f"food_{i}", f"food_{j}", True)
        for i in range(10) for j in range(i+1, min(i+3, 10))
    ]
    
    predictor.train(train_edges)
    
    # Compute embeddings
    embeddings = predictor.compute_embeddings()
    print(f"\n✓ Embeddings computed for {len(embeddings)} nodes")
    
    # Find similar foods
    similar = predictor.find_similar_foods("food_0", top_k=5)
    print(f"\n✓ Similar foods to 'food_0':")
    for node_id, score in similar:
        print(f"  {node_id}: {score:.3f}")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_gnn()
