"""
Knowledge Graphs & Deep Learning Module
========================================

Advanced AI for food-health intelligence:
- Graph Neural Networks (GNN)
- Deep learning models for nutrition
- LLM knowledge distillation
- Integrated AI system

Author: Wellomex AI Team
"""

from .food_knowledge_graph import (
    FoodKnowledgeGraph,
    EntityType,
    RelationType,
    Entity,
    Relation
)

from .deep_learning_models import (
    FoodTransformerEncoder,
    HealthGoalPredictor,
    DiseaseRiskPredictor,
    PersonalizedNutritionModel,
    LLMKnowledgeDistillation,
    TrainingConfig,
    NutritionModelTrainer,
    NutritionInference
)

from .integrated_nutrition_ai import (
    IntegratedNutritionAI,
    UserContext,
    FoodItem,
    NutritionRecommendation,
    create_integrated_system
)

__all__ = [
    # Knowledge Graph
    "FoodKnowledgeGraph",
    "EntityType",
    "RelationType",
    "Entity",
    "Relation",
    
    # Deep Learning Models
    "FoodTransformerEncoder",
    "HealthGoalPredictor",
    "DiseaseRiskPredictor",
    "PersonalizedNutritionModel",
    "LLMKnowledgeDistillation",
    "TrainingConfig",
    "NutritionModelTrainer",
    "NutritionInference",
    
    # Integrated System
    "IntegratedNutritionAI",
    "UserContext",
    "FoodItem",
    "NutritionRecommendation",
    "create_integrated_system"
]
