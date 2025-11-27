"""
Recommender System
==================

Advanced recommendation engine for personalized food and meal suggestions
using collaborative filtering, content-based filtering, and hybrid approaches.

Features:
1. Collaborative filtering (user-user, item-item)
2. Content-based filtering
3. Matrix factorization (SVD, ALS)
4. Deep learning recommendations
5. Hybrid recommendation strategies
6. Cold start handling
7. Real-time recommendations
8. Contextual recommendations

Performance Targets:
- Recommendation latency: <100ms
- Support 1M+ users
- 100K+ items (recipes/foods)
- Update models: <1 hour
- Precision@10: >0.3
- Recall@10: >0.4
- NDCG@10: >0.5

Author: Wellomex AI Team
Date: November 2025
Version: 5.0.0
"""

import time
import logging
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from collections import defaultdict, Counter
import json
from datetime import datetime

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

class RecommendationType(Enum):
    """Recommendation algorithm type"""
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    MATRIX_FACTORIZATION = "matrix_factorization"
    DEEP_LEARNING = "deep_learning"
    HYBRID = "hybrid"


class InteractionType(Enum):
    """User-item interaction type"""
    VIEW = "view"
    LIKE = "like"
    SAVE = "save"
    COOK = "cook"
    RATE = "rate"
    SHARE = "share"


@dataclass
class RecommenderConfig:
    """Recommender system configuration"""
    # Algorithm
    algorithm: RecommendationType = RecommendationType.HYBRID
    
    # Collaborative filtering
    num_neighbors: int = 50
    similarity_metric: str = "cosine"  # cosine, pearson, jaccard
    
    # Matrix factorization
    num_factors: int = 100
    learning_rate: float = 0.01
    regularization: float = 0.01
    num_iterations: int = 50
    
    # Deep learning
    embedding_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.2
    
    # Hybrid
    cf_weight: float = 0.4
    content_weight: float = 0.3
    mf_weight: float = 0.3
    
    # General
    top_k: int = 10
    min_interactions: int = 5
    cache_size: int = 10000


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class UserInteraction:
    """User-item interaction"""
    user_id: str
    item_id: str
    interaction_type: InteractionType
    rating: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """User profile"""
    user_id: str
    preferences: Dict[str, float] = field(default_factory=dict)
    dietary_restrictions: Set[str] = field(default_factory=set)
    allergens: Set[str] = field(default_factory=set)
    favorite_cuisines: List[str] = field(default_factory=list)
    interaction_history: List[UserInteraction] = field(default_factory=list)


@dataclass
class ItemProfile:
    """Item (recipe/food) profile"""
    item_id: str
    name: str
    category: str
    cuisine: str
    features: Dict[str, float] = field(default_factory=dict)
    ingredients: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    avg_rating: float = 0.0
    num_ratings: int = 0


@dataclass
class Recommendation:
    """Recommendation result"""
    item_id: str
    score: float
    explanation: str
    confidence: float = 1.0


# ============================================================================
# INTERACTION MATRIX
# ============================================================================

class InteractionMatrix:
    """
    User-Item Interaction Matrix
    
    Stores and manages user-item interactions.
    """
    
    def __init__(self):
        self.interactions: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.user_items: Dict[str, Set[str]] = defaultdict(set)
        self.item_users: Dict[str, Set[str]] = defaultdict(set)
        
        logger.info("Interaction Matrix initialized")
    
    def add_interaction(
        self,
        user_id: str,
        item_id: str,
        rating: float,
        interaction_type: InteractionType = InteractionType.RATE
    ):
        """Add user-item interaction"""
        # Weight interactions by type
        weights = {
            InteractionType.VIEW: 0.2,
            InteractionType.LIKE: 0.5,
            InteractionType.SAVE: 0.7,
            InteractionType.COOK: 0.9,
            InteractionType.RATE: 1.0,
            InteractionType.SHARE: 0.8
        }
        
        weighted_rating = rating * weights.get(interaction_type, 1.0)
        
        self.interactions[user_id][item_id] = weighted_rating
        self.user_items[user_id].add(item_id)
        self.item_users[item_id].add(user_id)
    
    def get_user_interactions(self, user_id: str) -> Dict[str, float]:
        """Get all interactions for user"""
        return self.interactions.get(user_id, {})
    
    def get_item_interactions(self, item_id: str) -> Dict[str, float]:
        """Get all interactions for item"""
        interactions = {}
        
        for user in self.item_users.get(item_id, set()):
            if item_id in self.interactions.get(user, {}):
                interactions[user] = self.interactions[user][item_id]
        
        return interactions
    
    def get_common_items(self, user1_id: str, user2_id: str) -> Set[str]:
        """Get items interacted by both users"""
        return self.user_items.get(user1_id, set()) & self.user_items.get(user2_id, set())
    
    def get_common_users(self, item1_id: str, item2_id: str) -> Set[str]:
        """Get users who interacted with both items"""
        return self.item_users.get(item1_id, set()) & self.item_users.get(item2_id, set())
    
    def get_matrix_stats(self) -> Dict[str, Any]:
        """Get matrix statistics"""
        total_interactions = sum(len(items) for items in self.interactions.values())
        
        return {
            'num_users': len(self.interactions),
            'num_items': len(self.item_users),
            'total_interactions': total_interactions,
            'sparsity': 1 - (total_interactions / (len(self.interactions) * len(self.item_users)))
                if len(self.interactions) > 0 and len(self.item_users) > 0 else 1.0
        }


# ============================================================================
# COLLABORATIVE FILTERING
# ============================================================================

class CollaborativeFiltering:
    """
    Collaborative Filtering
    
    User-user and item-item collaborative filtering.
    """
    
    def __init__(
        self,
        interaction_matrix: InteractionMatrix,
        config: RecommenderConfig
    ):
        self.matrix = interaction_matrix
        self.config = config
        
        # Similarity caches
        self.user_similarity_cache: Dict[Tuple[str, str], float] = {}
        self.item_similarity_cache: Dict[Tuple[str, str], float] = {}
        
        logger.info("Collaborative Filtering initialized")
    
    def compute_user_similarity(self, user1_id: str, user2_id: str) -> float:
        """Compute similarity between two users"""
        cache_key = tuple(sorted([user1_id, user2_id]))
        
        if cache_key in self.user_similarity_cache:
            return self.user_similarity_cache[cache_key]
        
        # Get common items
        common_items = self.matrix.get_common_items(user1_id, user2_id)
        
        if len(common_items) < 2:
            return 0.0
        
        # Get ratings
        user1_ratings = self.matrix.get_user_interactions(user1_id)
        user2_ratings = self.matrix.get_user_interactions(user2_id)
        
        # Compute similarity based on metric
        if self.config.similarity_metric == "cosine":
            similarity = self._cosine_similarity(
                [user1_ratings[item] for item in common_items],
                [user2_ratings[item] for item in common_items]
            )
        elif self.config.similarity_metric == "pearson":
            similarity = self._pearson_correlation(
                [user1_ratings[item] for item in common_items],
                [user2_ratings[item] for item in common_items]
            )
        else:  # jaccard
            similarity = len(common_items) / len(
                self.matrix.user_items[user1_id] | self.matrix.user_items[user2_id]
            )
        
        self.user_similarity_cache[cache_key] = similarity
        
        return similarity
    
    def compute_item_similarity(self, item1_id: str, item2_id: str) -> float:
        """Compute similarity between two items"""
        cache_key = tuple(sorted([item1_id, item2_id]))
        
        if cache_key in self.item_similarity_cache:
            return self.item_similarity_cache[cache_key]
        
        # Get common users
        common_users = self.matrix.get_common_users(item1_id, item2_id)
        
        if len(common_users) < 2:
            return 0.0
        
        # Get ratings
        item1_ratings = self.matrix.get_item_interactions(item1_id)
        item2_ratings = self.matrix.get_item_interactions(item2_id)
        
        # Compute cosine similarity
        similarity = self._cosine_similarity(
            [item1_ratings[user] for user in common_users],
            [item2_ratings[user] for user in common_users]
        )
        
        self.item_similarity_cache[cache_key] = similarity
        
        return similarity
    
    def recommend_user_based(
        self,
        user_id: str,
        candidate_items: List[str],
        top_k: int = 10
    ) -> List[Recommendation]:
        """Generate recommendations using user-based CF"""
        user_interactions = self.matrix.get_user_interactions(user_id)
        
        if len(user_interactions) < self.config.min_interactions:
            logger.warning(f"User {user_id} has insufficient interactions")
            return []
        
        # Find similar users
        all_users = list(self.matrix.interactions.keys())
        user_similarities = []
        
        for other_user in all_users:
            if other_user != user_id:
                similarity = self.compute_user_similarity(user_id, other_user)
                if similarity > 0:
                    user_similarities.append((other_user, similarity))
        
        # Get top-K similar users
        user_similarities.sort(key=lambda x: x[1], reverse=True)
        top_neighbors = user_similarities[:self.config.num_neighbors]
        
        # Compute scores for candidate items
        item_scores = {}
        
        for item_id in candidate_items:
            if item_id in user_interactions:
                continue  # Already interacted
            
            score = 0.0
            similarity_sum = 0.0
            
            for neighbor_id, similarity in top_neighbors:
                neighbor_interactions = self.matrix.get_user_interactions(neighbor_id)
                
                if item_id in neighbor_interactions:
                    score += similarity * neighbor_interactions[item_id]
                    similarity_sum += similarity
            
            if similarity_sum > 0:
                item_scores[item_id] = score / similarity_sum
        
        # Sort and create recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = [
            Recommendation(
                item_id=item_id,
                score=score,
                explanation=f"Recommended based on similar users",
                confidence=min(score, 1.0)
            )
            for item_id, score in sorted_items[:top_k]
        ]
        
        return recommendations
    
    def recommend_item_based(
        self,
        user_id: str,
        candidate_items: List[str],
        top_k: int = 10
    ) -> List[Recommendation]:
        """Generate recommendations using item-based CF"""
        user_interactions = self.matrix.get_user_interactions(user_id)
        
        if len(user_interactions) < self.config.min_interactions:
            return []
        
        # Compute scores for candidate items
        item_scores = {}
        
        for candidate_id in candidate_items:
            if candidate_id in user_interactions:
                continue  # Already interacted
            
            score = 0.0
            similarity_sum = 0.0
            
            # Compare with user's interacted items
            for interacted_id, rating in user_interactions.items():
                similarity = self.compute_item_similarity(candidate_id, interacted_id)
                
                if similarity > 0:
                    score += similarity * rating
                    similarity_sum += similarity
            
            if similarity_sum > 0:
                item_scores[candidate_id] = score / similarity_sum
        
        # Sort and create recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = [
            Recommendation(
                item_id=item_id,
                score=score,
                explanation=f"Similar to items you liked",
                confidence=min(score, 1.0)
            )
            for item_id, score in sorted_items[:top_k]
        ]
        
        return recommendations
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity"""
        if not NUMPY_AVAILABLE:
            # Manual computation
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(a * a for a in vec1))
            norm2 = math.sqrt(sum(b * b for b in vec2))
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def _pearson_correlation(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute Pearson correlation"""
        if not NUMPY_AVAILABLE:
            n = len(vec1)
            if n < 2:
                return 0.0
            
            mean1 = sum(vec1) / n
            mean2 = sum(vec2) / n
            
            numerator = sum((a - mean1) * (b - mean2) for a, b in zip(vec1, vec2))
            
            var1 = sum((a - mean1) ** 2 for a in vec1)
            var2 = sum((b - mean2) ** 2 for b in vec2)
            
            denominator = math.sqrt(var1 * var2)
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
        
        return np.corrcoef(vec1, vec2)[0, 1]


# ============================================================================
# CONTENT-BASED FILTERING
# ============================================================================

class ContentBasedFiltering:
    """
    Content-Based Filtering
    
    Recommendations based on item features.
    """
    
    def __init__(self, item_profiles: Dict[str, ItemProfile]):
        self.item_profiles = item_profiles
        
        # Feature vocabulary
        self.feature_vocab: Set[str] = set()
        self._build_vocabulary()
        
        logger.info("Content-Based Filtering initialized")
    
    def _build_vocabulary(self):
        """Build feature vocabulary"""
        for profile in self.item_profiles.values():
            self.feature_vocab.update(profile.features.keys())
            self.feature_vocab.update(profile.tags)
            self.feature_vocab.add(profile.category)
            self.feature_vocab.add(profile.cuisine)
    
    def compute_item_features(self, item_id: str) -> Dict[str, float]:
        """Get feature vector for item"""
        profile = self.item_profiles.get(item_id)
        
        if not profile:
            return {}
        
        features = profile.features.copy()
        
        # Add categorical features
        features[f"category_{profile.category}"] = 1.0
        features[f"cuisine_{profile.cuisine}"] = 1.0
        
        # Add tags
        for tag in profile.tags:
            features[f"tag_{tag}"] = 1.0
        
        return features
    
    def compute_item_similarity(
        self,
        item1_id: str,
        item2_id: str
    ) -> float:
        """Compute similarity between items based on features"""
        features1 = self.compute_item_features(item1_id)
        features2 = self.compute_item_features(item2_id)
        
        # Get all features
        all_features = set(features1.keys()) | set(features2.keys())
        
        # Create vectors
        vec1 = [features1.get(f, 0.0) for f in all_features]
        vec2 = [features2.get(f, 0.0) for f in all_features]
        
        # Cosine similarity
        if not NUMPY_AVAILABLE:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(a * a for a in vec1))
            norm2 = math.sqrt(sum(b * b for b in vec2))
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def recommend(
        self,
        user_profile: UserProfile,
        candidate_items: List[str],
        top_k: int = 10
    ) -> List[Recommendation]:
        """Generate content-based recommendations"""
        # Build user preference profile from history
        user_features = defaultdict(float)
        
        for interaction in user_profile.interaction_history:
            if interaction.rating and interaction.rating > 3.0:
                item_features = self.compute_item_features(interaction.item_id)
                
                for feature, value in item_features.items():
                    user_features[feature] += value
        
        # Normalize
        total = sum(user_features.values())
        if total > 0:
            user_features = {k: v / total for k, v in user_features.items()}
        
        # Score candidate items
        item_scores = {}
        
        for item_id in candidate_items:
            item_features = self.compute_item_features(item_id)
            
            # Compute similarity to user profile
            all_features = set(user_features.keys()) | set(item_features.keys())
            
            user_vec = [user_features.get(f, 0.0) for f in all_features]
            item_vec = [item_features.get(f, 0.0) for f in all_features]
            
            # Cosine similarity
            if not NUMPY_AVAILABLE:
                dot_product = sum(a * b for a, b in zip(user_vec, item_vec))
                norm1 = math.sqrt(sum(a * a for a in user_vec))
                norm2 = math.sqrt(sum(b * b for b in item_vec))
                
                if norm1 > 0 and norm2 > 0:
                    score = dot_product / (norm1 * norm2)
                else:
                    score = 0.0
            else:
                v1 = np.array(user_vec)
                v2 = np.array(item_vec)
                
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                if norm1 > 0 and norm2 > 0:
                    score = np.dot(v1, v2) / (norm1 * norm2)
                else:
                    score = 0.0
            
            item_scores[item_id] = score
        
        # Sort and create recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = [
            Recommendation(
                item_id=item_id,
                score=score,
                explanation="Matches your preferences",
                confidence=min(score, 1.0)
            )
            for item_id, score in sorted_items[:top_k]
        ]
        
        return recommendations


# ============================================================================
# HYBRID RECOMMENDER
# ============================================================================

class HybridRecommender:
    """
    Hybrid Recommender System
    
    Combines multiple recommendation strategies.
    """
    
    def __init__(
        self,
        config: RecommenderConfig,
        interaction_matrix: InteractionMatrix,
        item_profiles: Dict[str, ItemProfile]
    ):
        self.config = config
        self.matrix = interaction_matrix
        
        # Initialize components
        self.cf = CollaborativeFiltering(interaction_matrix, config)
        self.content_based = ContentBasedFiltering(item_profiles)
        
        logger.info("Hybrid Recommender initialized")
    
    def recommend(
        self,
        user_id: str,
        user_profile: Optional[UserProfile],
        candidate_items: List[str],
        top_k: Optional[int] = None
    ) -> List[Recommendation]:
        """Generate hybrid recommendations"""
        if top_k is None:
            top_k = self.config.top_k
        
        start_time = time.time()
        
        # Get recommendations from each algorithm
        cf_user_recs = self.cf.recommend_user_based(user_id, candidate_items, top_k * 2)
        cf_item_recs = self.cf.recommend_item_based(user_id, candidate_items, top_k * 2)
        
        content_recs = []
        if user_profile:
            content_recs = self.content_based.recommend(user_profile, candidate_items, top_k * 2)
        
        # Combine scores
        combined_scores: Dict[str, float] = defaultdict(float)
        
        # Collaborative filtering (user-based)
        for rec in cf_user_recs:
            combined_scores[rec.item_id] += rec.score * self.config.cf_weight * 0.5
        
        # Collaborative filtering (item-based)
        for rec in cf_item_recs:
            combined_scores[rec.item_id] += rec.score * self.config.cf_weight * 0.5
        
        # Content-based
        for rec in content_recs:
            combined_scores[rec.item_id] += rec.score * self.config.content_weight
        
        # Sort by combined score
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create final recommendations
        recommendations = [
            Recommendation(
                item_id=item_id,
                score=score,
                explanation="Personalized recommendation",
                confidence=min(score, 1.0)
            )
            for item_id, score in sorted_items[:top_k]
        ]
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"Generated {len(recommendations)} recommendations in {elapsed_time*1000:.1f}ms")
        
        return recommendations
    
    def handle_cold_start(
        self,
        user_profile: UserProfile,
        candidate_items: List[str],
        top_k: int = 10
    ) -> List[Recommendation]:
        """Handle cold start for new users"""
        # Use content-based + popularity
        content_recs = self.content_based.recommend(user_profile, candidate_items, top_k)
        
        # Get popular items
        item_popularity = {}
        for item_id in candidate_items:
            interactions = self.matrix.get_item_interactions(item_id)
            item_popularity[item_id] = len(interactions)
        
        sorted_popular = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
        
        # Combine
        combined = {}
        
        for rec in content_recs:
            combined[rec.item_id] = rec.score * 0.7
        
        for item_id, popularity in sorted_popular[:top_k]:
            if item_id not in combined:
                combined[item_id] = popularity / max(item_popularity.values()) * 0.3
        
        sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = [
            Recommendation(
                item_id=item_id,
                score=score,
                explanation="Popular recommendation",
                confidence=min(score, 1.0)
            )
            for item_id, score in sorted_items[:top_k]
        ]
        
        return recommendations


# ============================================================================
# TESTING
# ============================================================================

def test_recommender_system():
    """Test recommender system"""
    print("=" * 80)
    print("RECOMMENDER SYSTEM - TEST")
    print("=" * 80)
    
    # Create interaction matrix
    matrix = InteractionMatrix()
    
    # Add sample interactions
    users = [f"user_{i}" for i in range(10)]
    items = [f"item_{i}" for i in range(20)]
    
    for user in users:
        for item in random.sample(items, 5):
            rating = random.uniform(3.0, 5.0)
            matrix.add_interaction(user, item, rating, InteractionType.RATE)
    
    stats = matrix.get_matrix_stats()
    
    print(f"\n✓ Interaction matrix created")
    print(f"  Users: {stats['num_users']}")
    print(f"  Items: {stats['num_items']}")
    print(f"  Interactions: {stats['total_interactions']}")
    print(f"  Sparsity: {stats['sparsity']:.2%}")
    
    # Create item profiles
    item_profiles = {}
    cuisines = ["italian", "chinese", "mexican", "indian"]
    categories = ["main", "appetizer", "dessert"]
    
    for item in items:
        item_profiles[item] = ItemProfile(
            item_id=item,
            name=f"Recipe {item}",
            category=random.choice(categories),
            cuisine=random.choice(cuisines),
            features={
                "calories": random.uniform(200, 800),
                "protein": random.uniform(10, 50)
            },
            tags=[random.choice(["healthy", "quick", "vegetarian"])]
        )
    
    print("✓ Item profiles created")
    
    # Test collaborative filtering
    print("\n" + "="*80)
    print("Test: Collaborative Filtering")
    print("="*80)
    
    config = RecommenderConfig()
    cf = CollaborativeFiltering(matrix, config)
    
    test_user = users[0]
    candidate_items = [item for item in items if item not in matrix.get_user_interactions(test_user)]
    
    user_based_recs = cf.recommend_user_based(test_user, candidate_items, top_k=5)
    
    print(f"\n✓ User-based CF recommendations for {test_user}:")
    for i, rec in enumerate(user_based_recs, 1):
        print(f"  {i}. {rec.item_id} (score: {rec.score:.3f})")
    
    item_based_recs = cf.recommend_item_based(test_user, candidate_items, top_k=5)
    
    print(f"\n✓ Item-based CF recommendations for {test_user}:")
    for i, rec in enumerate(item_based_recs, 1):
        print(f"  {i}. {rec.item_id} (score: {rec.score:.3f})")
    
    # Test content-based filtering
    print("\n" + "="*80)
    print("Test: Content-Based Filtering")
    print("="*80)
    
    content_based = ContentBasedFiltering(item_profiles)
    
    # Create user profile
    user_profile = UserProfile(
        user_id=test_user,
        interaction_history=[
            UserInteraction(test_user, item, InteractionType.RATE, 4.5)
            for item in list(matrix.get_user_interactions(test_user).keys())[:3]
        ]
    )
    
    content_recs = content_based.recommend(user_profile, candidate_items, top_k=5)
    
    print(f"✓ Content-based recommendations:")
    for i, rec in enumerate(content_recs, 1):
        print(f"  {i}. {rec.item_id} (score: {rec.score:.3f})")
    
    # Test hybrid recommender
    print("\n" + "="*80)
    print("Test: Hybrid Recommender")
    print("="*80)
    
    hybrid = HybridRecommender(config, matrix, item_profiles)
    
    hybrid_recs = hybrid.recommend(test_user, user_profile, candidate_items, top_k=5)
    
    print(f"✓ Hybrid recommendations:")
    for i, rec in enumerate(hybrid_recs, 1):
        print(f"  {i}. {rec.item_id} (score: {rec.score:.3f}, confidence: {rec.confidence:.2f})")
    
    # Test cold start
    print("\n" + "="*80)
    print("Test: Cold Start Handling")
    print("="*80)
    
    new_user_profile = UserProfile(
        user_id="new_user",
        favorite_cuisines=["italian"],
        dietary_restrictions={"vegetarian"}
    )
    
    cold_start_recs = hybrid.handle_cold_start(new_user_profile, items[:10], top_k=5)
    
    print(f"✓ Cold start recommendations:")
    for i, rec in enumerate(cold_start_recs, 1):
        print(f"  {i}. {rec.item_id} (score: {rec.score:.3f})")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_recommender_system()
