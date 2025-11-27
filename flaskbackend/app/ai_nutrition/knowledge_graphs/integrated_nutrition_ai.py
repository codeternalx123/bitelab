"""
Integrated Nutrition AI System
===============================

Combines:
1. Knowledge Graphs (GNN-based entity relationships)
2. Deep Learning Models (Transformer + Multi-task learning)
3. LLM Orchestration (GPT-4/Claude for knowledge expansion)

This creates a complete AI system that:
- Understands food-health relationships through graphs
- Makes predictions via deep neural networks
- Continuously learns from large language models
- Provides personalized recommendations

Author: Wellomex AI Team
Date: November 2025
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import json
import asyncio

if TYPE_CHECKING:
    try:
        from openai import AsyncOpenAI  # type: ignore
    except ImportError:
        pass

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .food_knowledge_graph import FoodKnowledgeGraph, EntityType, RelationType
from .deep_learning_models import (
    PersonalizedNutritionModel,
    NutritionInference,
    LLMKnowledgeDistillation,
    TrainingConfig,
    NutritionModelTrainer
)

logger = logging.getLogger(__name__)


# ============================================================================
# INTEGRATION LAYER
# ============================================================================

@dataclass
class UserContext:
    """User context for personalized recommendations"""
    user_id: str
    age: int
    gender: str
    weight: float  # kg
    height: float  # cm
    activity_level: str
    health_goals: List[str]
    medical_conditions: List[str]
    medications: List[str]
    allergies: List[str]
    dietary_restrictions: List[str]
    biomarkers: Dict[str, float]  # e.g., {"glucose": 95, "cholesterol": 180}


@dataclass
class FoodItem:
    """Food item for analysis"""
    name: str
    ingredients: List[str]
    nutrients: Dict[str, float]  # e.g., {"protein": 20, "carbs": 30}
    portion_size: float  # grams
    preparation_method: Optional[str] = None


@dataclass
class NutritionRecommendation:
    """Comprehensive nutrition recommendation"""
    food_name: str
    recommendation_score: float  # 0-1, higher is better
    alignment_with_goals: Dict[str, float]  # goal -> score
    disease_risk_impacts: Dict[str, float]  # disease -> risk change
    portion_recommendation: float  # grams
    reasoning: List[str]
    confidence: float
    alternatives: List[str]
    cautions: List[str]


class IntegratedNutritionAI:
    """
    Main integrated system combining all AI components
    """
    
    def __init__(
        self,
        knowledge_graph: Optional[FoodKnowledgeGraph] = None,
        deep_model: Optional[PersonalizedNutritionModel] = None,
        llm_client: Optional[Any] = None,
        device: str = "cpu"
    ):
        """
        Initialize the integrated AI system
        
        Args:
            knowledge_graph: Pre-initialized knowledge graph
            deep_model: Pre-trained deep learning model
            llm_client: OpenAI/Anthropic client for knowledge expansion
            device: torch device
        """
        self.device = device
        
        # Initialize components
        self.knowledge_graph = knowledge_graph or FoodKnowledgeGraph()
        
        if TORCH_AVAILABLE and deep_model is not None:
            self.deep_model = deep_model.to(device)
            self.inference_engine = NutritionInference(self.deep_model)
        else:
            self.deep_model = None
            self.inference_engine = None
            logger.warning("Deep learning models not available (PyTorch required)")
        
        # LLM knowledge distillation
        if llm_client:
            self.llm_distillation = LLMKnowledgeDistillation(llm_client)
        else:
            self.llm_distillation = None
            logger.warning("LLM client not provided - knowledge expansion disabled")
        
        # Cache for embeddings
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        
    # ========================================================================
    # MAIN RECOMMENDATION ENGINE
    # ========================================================================
    
    async def analyze_food(
        self,
        food: FoodItem,
        user_context: UserContext,
        use_graph: bool = True,
        use_deep_model: bool = True,
        use_llm: bool = False
    ) -> NutritionRecommendation:
        """
        Comprehensive food analysis combining all AI methods
        
        Args:
            food: Food item to analyze
            user_context: User's health profile
            use_graph: Use knowledge graph reasoning
            use_deep_model: Use deep learning predictions
            use_llm: Query LLM for additional insights
        
        Returns:
            Comprehensive nutrition recommendation
        """
        
        reasoning = []
        goal_alignments = {}
        disease_impacts = {}
        cautions = []
        alternatives = []
        
        # ----------------------------------------------------------------
        # 1. KNOWLEDGE GRAPH ANALYSIS
        # ----------------------------------------------------------------
        
        if use_graph:
            logger.info(f"Analyzing {food.name} via knowledge graph")
            
            # Check if food exists in graph
            food_entity = self.knowledge_graph.entities.get(food.name)
            
            if not food_entity:
                # Add food to graph
                food_entity = self.knowledge_graph.add_entity(
                    name=food.name,
                    entity_type=EntityType.FOOD,
                    properties={"nutrients": food.nutrients}
                )
                
                # Add nutrient relationships
                for nutrient, amount in food.nutrients.items():
                    nutrient_entity = self.knowledge_graph.entities.get(nutrient)
                    if nutrient_entity:
                        self.knowledge_graph.add_relation(
                            from_entity=food.name,
                            to_entity=nutrient,
                            relation_type=RelationType.CONTAINS,
                            properties={"amount": amount},
                            confidence=0.9
                        )
            
            # Query graph for health goal alignment
            for goal in user_context.health_goals:
                matching_foods = self.knowledge_graph.get_food_for_goal(goal)
                
                if food.name in matching_foods:
                    # Get relationship confidence
                    relations = self.knowledge_graph.query(food.name, RelationType.BENEFITS)
                    goal_relation = next(
                        (r for r in relations if r["to"] == goal),
                        None
                    )
                    
                    if goal_relation:
                        confidence = goal_relation.get("confidence", 0.5)
                        goal_alignments[goal] = confidence
                        reasoning.append(
                            f"Supports {goal} (knowledge graph confidence: {confidence:.2f})"
                        )
            
            # Check disease contraindications
            for disease in user_context.medical_conditions:
                contraindicated = self.knowledge_graph.get_contraindicated_foods(disease)
                
                if food.name in contraindicated:
                    cautions.append(f"May worsen {disease} - consult healthcare provider")
                    disease_impacts[disease] = 0.8  # High risk
                    reasoning.append(f"Contraindicated for {disease}")
            
            # Check medication interactions
            for medication in user_context.medications:
                med_entity = self.knowledge_graph.entities.get(medication)
                
                if med_entity:
                    interactions = self.knowledge_graph.query(
                        medication,
                        RelationType.INTERACTS_WITH
                    )
                    
                    food_nutrients = set(food.nutrients.keys())
                    
                    for interaction in interactions:
                        if interaction["to"] in food_nutrients:
                            cautions.append(
                                f"Contains {interaction['to']} which may interact with {medication}"
                            )
        
        # ----------------------------------------------------------------
        # 2. DEEP LEARNING PREDICTION
        # ----------------------------------------------------------------
        
        if use_deep_model and self.inference_engine and TORCH_AVAILABLE:
            logger.info(f"Running deep learning prediction for {food.name}")
            
            try:
                # Prepare input tensors
                food_tokens = self._tokenize_food(food)
                user_profile = self._encode_user_context(user_context)
                
                # Run inference
                prediction = self.inference_engine.predict(
                    food_tokens=food_tokens,
                    user_profile=user_profile
                )
                
                recommendation_score = prediction["recommendation_score"]
                
                # Extract goal scores
                goal_scores_array = prediction["goal_scores"]
                
                # Map to user's specific goals
                for i, goal in enumerate(user_context.health_goals):
                    if i < len(goal_scores_array):
                        score = float(goal_scores_array[i])
                        
                        # Combine with graph score if available
                        if goal in goal_alignments:
                            # Weighted average: 60% deep learning, 40% knowledge graph
                            goal_alignments[goal] = 0.6 * score + 0.4 * goal_alignments[goal]
                        else:
                            goal_alignments[goal] = score
                        
                        if score > 0.7:
                            reasoning.append(
                                f"Deep learning model predicts strong support for {goal} (score: {score:.2f})"
                            )
                
                # Extract disease risk scores
                risk_scores_array = prediction["risk_scores"]
                
                for i, disease in enumerate(user_context.medical_conditions):
                    if i < len(risk_scores_array):
                        risk = float(risk_scores_array[i])
                        disease_impacts[disease] = risk
                        
                        if risk > 0.6:
                            cautions.append(
                                f"Model predicts increased risk for {disease} (risk: {risk:.2f})"
                            )
                
                # Get explanation
                explanation = self.inference_engine.explain_prediction(
                    food_tokens=food_tokens,
                    user_profile=user_profile,
                    top_k=3
                )
                
                logger.info(f"Top contributing factors: {explanation['top_goals']}")
                
            except Exception as e:
                logger.error(f"Deep learning prediction failed: {e}")
                recommendation_score = 0.5  # Default
        
        else:
            # Fallback: average goal alignment scores
            if goal_alignments:
                recommendation_score = sum(goal_alignments.values()) / len(goal_alignments)
            else:
                recommendation_score = 0.5
        
        # ----------------------------------------------------------------
        # 3. LLM ENHANCEMENT (Optional)
        # ----------------------------------------------------------------
        
        if use_llm and self.llm_distillation:
            logger.info(f"Querying LLM for additional insights on {food.name}")
            
            try:
                # Validate predictions with LLM
                validation = await self.llm_distillation.validate_prediction(
                    food=food.name,
                    predicted_goal_scores=goal_alignments
                )
                
                if validation.get("accuracy", 0) < 0.7:
                    logger.warning(f"LLM flagged low accuracy for {food.name}")
                    reasoning.append(f"LLM validation: {validation.get('explanation', '')}")
                    
                    # Apply corrections
                    corrections = validation.get("corrections", {})
                    for goal, corrected_score in corrections.items():
                        if goal in goal_alignments:
                            goal_alignments[goal] = corrected_score
            
            except Exception as e:
                logger.error(f"LLM validation failed: {e}")
        
        # ----------------------------------------------------------------
        # 4. PORTION RECOMMENDATION
        # ----------------------------------------------------------------
        
        portion = self._calculate_optimal_portion(
            food=food,
            user_context=user_context,
            goal_alignments=goal_alignments,
            disease_impacts=disease_impacts
        )
        
        # ----------------------------------------------------------------
        # 5. FIND ALTERNATIVES
        # ----------------------------------------------------------------
        
        if use_graph:
            # Find similar foods with better scores
            similar_foods = self._find_similar_foods(
                food=food,
                min_score=recommendation_score + 0.1
            )
            alternatives = similar_foods[:3]
        
        # ----------------------------------------------------------------
        # 6. CALCULATE CONFIDENCE
        # ----------------------------------------------------------------
        
        confidence = self._calculate_confidence(
            has_graph_data=use_graph and bool(goal_alignments),
            has_deep_prediction=use_deep_model and self.inference_engine is not None,
            has_llm_validation=use_llm and self.llm_distillation is not None
        )
        
        return NutritionRecommendation(
            food_name=food.name,
            recommendation_score=recommendation_score,
            alignment_with_goals=goal_alignments,
            disease_risk_impacts=disease_impacts,
            portion_recommendation=portion,
            reasoning=reasoning,
            confidence=confidence,
            alternatives=alternatives,
            cautions=cautions
        )
    
    # ========================================================================
    # KNOWLEDGE EXPANSION
    # ========================================================================
    
    async def expand_knowledge(
        self,
        entities: List[str],
        entity_type: EntityType = EntityType.FOOD
    ):
        """
        Expand knowledge graph using LLM
        
        Queries GPT-4 for relationships and adds to graph
        """
        
        if not self.llm_distillation:
            logger.warning("LLM client not available for knowledge expansion")
            return
        
        logger.info(f"Expanding knowledge for {len(entities)} entities")
        
        for entity in entities:
            try:
                await self.knowledge_graph.expand_knowledge_from_llm(
                    entity_name=entity,
                    llm_client=self.llm_distillation.llm_client
                )
                
            except Exception as e:
                logger.error(f"Knowledge expansion failed for {entity}: {e}")
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    
    async def train_from_llm(
        self,
        foods: List[str],
        health_goals: List[str],
        diseases: List[str],
        num_samples: int = 100
    ):
        """
        Generate training data from LLM and train deep models
        """
        
        if not self.llm_distillation:
            logger.error("LLM client required for training")
            return
        
        if not TORCH_AVAILABLE or not self.deep_model:
            logger.error("Deep learning models not available")
            return
        
        logger.info(f"Generating training data from LLM for {len(foods)} foods")
        
        # Generate training data
        training_data = await self.llm_distillation.generate_training_data(
            foods=foods,
            health_goals=health_goals,
            diseases=diseases,
            samples_per_food=num_samples // len(foods)
        )
        
        logger.info(f"Generated {len(training_data)} training examples")
        
        # TODO: Convert to PyTorch dataset and train
        # This would require implementing a custom Dataset class
        
        return training_data
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _tokenize_food(self, food: FoodItem) -> torch.Tensor:
        """Convert food to token IDs"""
        # Simplified tokenization - in production use proper tokenizer
        tokens = [hash(food.name) % 10000]  # Dummy tokenization
        tokens.extend([hash(ing) % 10000 for ing in food.ingredients[:10]])
        
        # Pad to fixed length
        tokens = tokens[:20] + [0] * max(0, 20 - len(tokens))
        
        return torch.tensor([tokens], dtype=torch.long, device=self.device)
    
    def _encode_user_context(self, user: UserContext) -> torch.Tensor:
        """Encode user profile as feature vector"""
        
        features = [
            user.age / 100.0,
            user.weight / 150.0,
            user.height / 200.0,
            1.0 if user.gender == "male" else 0.0,
            {"sedentary": 0.2, "light": 0.4, "moderate": 0.6, "active": 0.8, "very_active": 1.0}.get(user.activity_level, 0.5)
        ]
        
        # Add biomarker features
        for biomarker in ["glucose", "cholesterol", "blood_pressure", "bmi"]:
            features.append(user.biomarkers.get(biomarker, 0.0) / 200.0)
        
        # Pad to fixed size (128 dims)
        features = features[:128] + [0.0] * max(0, 128 - len(features))
        
        return torch.tensor([features], dtype=torch.float32, device=self.device)
    
    def _calculate_optimal_portion(
        self,
        food: FoodItem,
        user_context: UserContext,
        goal_alignments: Dict[str, float],
        disease_impacts: Dict[str, float]
    ) -> float:
        """Calculate optimal portion size in grams"""
        
        base_portion = food.portion_size
        
        # Adjust based on goals
        if goal_alignments:
            avg_alignment = sum(goal_alignments.values()) / len(goal_alignments)
            
            if avg_alignment > 0.8:
                base_portion *= 1.2  # Increase portion for highly aligned foods
            elif avg_alignment < 0.4:
                base_portion *= 0.7  # Decrease portion for poorly aligned foods
        
        # Adjust based on disease risks
        if disease_impacts:
            max_risk = max(disease_impacts.values())
            
            if max_risk > 0.7:
                base_portion *= 0.5  # Significantly reduce for high-risk foods
        
        # Activity level adjustment
        activity_multipliers = {
            "sedentary": 0.8,
            "light": 0.9,
            "moderate": 1.0,
            "active": 1.2,
            "very_active": 1.4
        }
        
        base_portion *= activity_multipliers.get(user_context.activity_level, 1.0)
        
        return round(base_portion, 1)
    
    def _find_similar_foods(
        self,
        food: FoodItem,
        min_score: float = 0.7
    ) -> List[str]:
        """Find similar foods with better scores"""
        
        # Use knowledge graph to find foods with similar nutrients
        similar = []
        
        for entity_name, entity in self.knowledge_graph.entities.items():
            if entity.entity_type == EntityType.FOOD and entity_name != food.name:
                # Check nutrient overlap
                entity_nutrients = entity.properties.get("nutrients", {})
                
                if entity_nutrients:
                    overlap = len(set(food.nutrients.keys()) & set(entity_nutrients.keys()))
                    
                    if overlap >= 2:  # At least 2 shared nutrients
                        similar.append(entity_name)
        
        return similar[:5]
    
    def _calculate_confidence(
        self,
        has_graph_data: bool,
        has_deep_prediction: bool,
        has_llm_validation: bool
    ) -> float:
        """Calculate overall confidence in recommendation"""
        
        confidence = 0.3  # Base confidence
        
        if has_graph_data:
            confidence += 0.2
        
        if has_deep_prediction:
            confidence += 0.3
        
        if has_llm_validation:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def save(self, directory: str):
        """Save all components"""
        
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save knowledge graph
        self.knowledge_graph.save(os.path.join(directory, "knowledge_graph.json"))
        
        # Save deep model
        if self.deep_model and TORCH_AVAILABLE:
            torch.save(
                self.deep_model.state_dict(),
                os.path.join(directory, "deep_model.pt")
            )
        
        logger.info(f"Saved integrated AI to {directory}")
    
    def load(self, directory: str):
        """Load all components"""
        
        import os
        
        # Load knowledge graph
        graph_path = os.path.join(directory, "knowledge_graph.json")
        if os.path.exists(graph_path):
            self.knowledge_graph = FoodKnowledgeGraph.load(graph_path)
        
        # Load deep model
        model_path = os.path.join(directory, "deep_model.pt")
        if os.path.exists(model_path) and self.deep_model and TORCH_AVAILABLE:
            self.deep_model.load_state_dict(torch.load(model_path))
        
        logger.info(f"Loaded integrated AI from {directory}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_integrated_system(
    llm_api_key: Optional[str] = None,
    device: str = "cpu"
) -> IntegratedNutritionAI:
    """
    Create a fully configured integrated nutrition AI system
    
    Args:
        llm_api_key: OpenAI API key for LLM features
        device: torch device
    
    Returns:
        Configured IntegratedNutritionAI instance
    """
    
    # Initialize knowledge graph
    knowledge_graph = FoodKnowledgeGraph()
    
    # Initialize deep model
    deep_model = None
    if TORCH_AVAILABLE:
        deep_model = PersonalizedNutritionModel(
            food_vocab_size=10000,
            embedding_dim=256,
            num_health_goals=55,
            num_diseases=100,
            user_profile_dim=128
        )
    
    # Initialize LLM client
    llm_client = None
    if llm_api_key:
        try:
            from openai import AsyncOpenAI  # type: ignore
            llm_client = AsyncOpenAI(api_key=llm_api_key)
        except ImportError:
            logger.warning("OpenAI library not available")
    
    return IntegratedNutritionAI(
        knowledge_graph=knowledge_graph,
        deep_model=deep_model,
        llm_client=llm_client,
        device=device
    )
