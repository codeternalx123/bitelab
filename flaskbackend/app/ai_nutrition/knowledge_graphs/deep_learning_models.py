"""
Deep Learning Models for Food-Health Prediction
===============================================

Advanced neural network models for:
- Food embedding learning
- Health goal optimization
- Disease risk prediction
- Personalized nutrition recommendations
- Multi-task learning across 55+ goals

Models can be pre-trained on large datasets and fine-tuned
with data from LLMs or research papers.

Author: Wellomex AI Team
Date: November 2025
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# TRANSFORMER-BASED FOOD ENCODER
# ============================================================================

class FoodTransformerEncoder(nn.Module):
    """
    Transformer-based encoder for food understanding
    
    Learns contextualized embeddings for foods based on:
    - Nutritional composition
    - Ingredient relationships
    - Cooking methods
    - Cultural context
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, embedding_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, food_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode food tokens
        
        Args:
            food_tokens: [batch_size, seq_len] token IDs
        
        Returns:
            Food embeddings: [batch_size, embedding_dim]
        """
        batch_size, seq_len = food_tokens.shape
        
        # Embed tokens
        x = self.embedding(food_tokens)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Pool (mean over sequence)
        x = x.mean(dim=1)
        
        # Project
        x = self.output_proj(x)
        
        return x


# ============================================================================
# MULTI-TASK HEALTH GOAL PREDICTOR
# ============================================================================

class HealthGoalPredictor(nn.Module):
    """
    Multi-task model for predicting alignment with 55+ health goals
    
    Architecture:
    - Shared food encoder
    - Goal-specific prediction heads
    - Cross-goal attention mechanism
    - Uncertainty estimation
    """
    
    def __init__(
        self,
        food_embedding_dim: int = 256,
        num_health_goals: int = 55,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.num_goals = num_health_goals
        
        # Shared encoder
        self.food_encoder = nn.Sequential(
            nn.Linear(food_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Goal embeddings (learnable)
        self.goal_embeddings = nn.Embedding(num_health_goals, hidden_dim)
        
        # Cross-attention between food and goals
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Goal-specific prediction heads
        self.goal_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
            for _ in range(num_health_goals)
        ])
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_health_goals),
            nn.Softplus()  # Positive uncertainty values
        )
        
    def forward(
        self,
        food_embeddings: torch.Tensor,
        goal_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict alignment with health goals
        
        Args:
            food_embeddings: [batch_size, embedding_dim]
            goal_ids: [batch_size, num_goals] optional goal filtering
        
        Returns:
            Dictionary with predictions and uncertainties
        """
        batch_size = food_embeddings.shape[0]
        
        # Encode food
        food_encoded = self.food_encoder(food_embeddings)
        food_encoded = food_encoded.unsqueeze(1)  # [batch, 1, hidden]
        
        # Get all goal embeddings
        goal_ids_all = torch.arange(self.num_goals, device=food_embeddings.device)
        goal_embs = self.goal_embeddings(goal_ids_all)
        goal_embs = goal_embs.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_goals, hidden]
        
        # Cross-attention: food queries goals
        attended, _ = self.cross_attention(
            query=food_encoded,
            key=goal_embs,
            value=goal_embs
        )
        
        attended = attended.squeeze(1)  # [batch, hidden]
        
        # Predict for each goal
        goal_scores = []
        for i, head in enumerate(self.goal_heads):
            # Combine food and attended features
            combined = attended + food_encoded.squeeze(1)
            score = head(combined)
            goal_scores.append(score)
        
        goal_scores = torch.cat(goal_scores, dim=1)  # [batch, num_goals]
        
        # Predict uncertainties
        uncertainties = self.uncertainty_head(attended)
        
        return {
            "goal_scores": goal_scores,
            "uncertainties": uncertainties,
            "food_encoding": food_encoded.squeeze(1)
        }


# ============================================================================
# DISEASE RISK PREDICTOR
# ============================================================================

class DiseaseRiskPredictor(nn.Module):
    """
    Deep model for disease-specific risk assessment
    
    Predicts risk scores for 100+ diseases based on:
    - Food composition
    - User medical history
    - Medication interactions
    - Biomarker levels
    """
    
    def __init__(
        self,
        food_embedding_dim: int = 256,
        user_profile_dim: int = 128,
        num_diseases: int = 100,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        # Food encoder
        self.food_encoder = nn.Sequential(
            nn.Linear(food_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # User profile encoder
        self.user_encoder = nn.Sequential(
            nn.Linear(user_profile_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Interaction modeling
        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Disease-specific risk heads
        self.risk_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_diseases),
            nn.Sigmoid()  # Risk scores 0-1
        )
        
        # Explainability: attention over food components
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
    def forward(
        self,
        food_embeddings: torch.Tensor,
        user_profile: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict disease risks
        
        Args:
            food_embeddings: [batch_size, food_dim]
            user_profile: [batch_size, profile_dim]
        
        Returns:
            Risk scores and attention weights
        """
        # Encode inputs
        food_encoded = self.food_encoder(food_embeddings)
        user_encoded = self.user_encoder(user_profile)
        
        # Model food-user interaction
        combined = torch.cat([food_encoded, user_encoded], dim=1)
        interaction = self.interaction_layer(combined)
        
        # Predict risks
        risk_scores = self.risk_predictor(interaction)
        
        # Compute attention (for explainability)
        interaction_unsq = interaction.unsqueeze(1)
        attended, attention_weights = self.attention(
            query=interaction_unsq,
            key=interaction_unsq,
            value=interaction_unsq
        )
        
        return {
            "risk_scores": risk_scores,  # [batch, num_diseases]
            "attention_weights": attention_weights,
            "interaction_features": interaction
        }


# ============================================================================
# PERSONALIZED NUTRITION RECOMMENDER
# ============================================================================

class PersonalizedNutritionModel(nn.Module):
    """
    End-to-end model for personalized nutrition recommendations
    
    Combines:
    - Food understanding
    - Health goal alignment
    - Disease risk assessment
    - User preferences
    - Medication interactions
    """
    
    def __init__(
        self,
        food_vocab_size: int,
        embedding_dim: int = 256,
        num_health_goals: int = 55,
        num_diseases: int = 100,
        user_profile_dim: int = 128
    ):
        super().__init__()
        
        # Food encoder
        self.food_transformer = FoodTransformerEncoder(
            vocab_size=food_vocab_size,
            embedding_dim=embedding_dim
        )
        
        # Health goal predictor
        self.goal_predictor = HealthGoalPredictor(
            food_embedding_dim=embedding_dim,
            num_health_goals=num_health_goals
        )
        
        # Disease risk predictor
        self.risk_predictor = DiseaseRiskPredictor(
            food_embedding_dim=embedding_dim,
            user_profile_dim=user_profile_dim,
            num_diseases=num_diseases
        )
        
        # Recommendation scorer
        self.recommendation_scorer = nn.Sequential(
            nn.Linear(num_health_goals + num_diseases + embedding_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        food_tokens: torch.Tensor,
        user_profile: torch.Tensor,
        active_goals: Optional[torch.Tensor] = None,
        active_diseases: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass for personalized recommendation
        
        Args:
            food_tokens: [batch_size, seq_len] food token IDs
            user_profile: [batch_size, profile_dim] user features
            active_goals: [batch_size, num_goals] binary mask
            active_diseases: [batch_size, num_diseases] binary mask
        
        Returns:
            Comprehensive predictions
        """
        # Encode food
        food_embeddings = self.food_transformer(food_tokens)
        
        # Predict goal alignment
        goal_outputs = self.goal_predictor(food_embeddings)
        goal_scores = goal_outputs["goal_scores"]
        
        # Predict disease risks
        risk_outputs = self.risk_predictor(food_embeddings, user_profile)
        risk_scores = risk_outputs["risk_scores"]
        
        # Apply masks if provided
        if active_goals is not None:
            goal_scores = goal_scores * active_goals
        if active_diseases is not None:
            risk_scores = risk_scores * active_diseases
        
        # Combine for final recommendation score
        features = torch.cat([
            food_embeddings,
            goal_scores,
            risk_scores
        ], dim=1)
        
        recommendation_score = self.recommendation_scorer(features)
        
        return {
            "recommendation_score": recommendation_score,
            "goal_scores": goal_scores,
            "risk_scores": risk_scores,
            "goal_uncertainties": goal_outputs["uncertainties"],
            "food_embeddings": food_embeddings,
            "attention_weights": risk_outputs["attention_weights"]
        }


# ============================================================================
# LLM-ENHANCED TRAINING
# ============================================================================

class LLMKnowledgeDistillation:
    """
    Use large LLMs to generate training data and improve model accuracy
    
    Strategies:
    1. Query LLM for food-health relationships
    2. Generate synthetic training examples
    3. Validate model predictions against LLM
    4. Active learning with LLM oracle
    """
    
    def __init__(self, llm_client: Any, model: str = "gpt-4-turbo-preview"):
        self.llm_client = llm_client
        self.model = model
        self.training_examples: List[Dict[str, Any]] = []
        
    async def generate_training_data(
        self,
        foods: List[str],
        health_goals: List[str],
        diseases: List[str],
        samples_per_food: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate training data by querying LLM
        
        For each food, ask LLM about:
        - Which health goals it supports (0-1 scores)
        - Which diseases it helps manage
        - Risk levels for various conditions
        """
        
        training_data = []
        
        for food in foods:
            prompt = f"""Rate how well {food} supports each of these health goals (0.0-1.0):

Goals: {', '.join(health_goals)}

Also rate disease management effectiveness (0.0-1.0) for:
Diseases: {', '.join(diseases)}

Provide as JSON with 'goal_scores' and 'disease_scores' dictionaries."""
            
            try:
                response = await self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a nutrition expert. Respond with JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                data = json.loads(response.choices[0].message.content)
                
                training_data.append({
                    "food": food,
                    "goal_scores": data.get("goal_scores", {}),
                    "disease_scores": data.get("disease_scores", {})
                })
                
            except Exception as e:
                logger.error(f"LLM data generation error for {food}: {e}")
        
        self.training_examples.extend(training_data)
        logger.info(f"Generated {len(training_data)} training examples from LLM")
        
        return training_data
    
    async def validate_prediction(
        self,
        food: str,
        predicted_goal_scores: Dict[str, float],
        threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Validate model prediction against LLM knowledge
        
        Returns agreement metrics and corrections
        """
        
        prompt = f"""Review these predicted health goal scores for {food}:

{json.dumps(predicted_goal_scores, indent=2)}

Are these scores accurate? Provide:
1. Overall accuracy (0.0-1.0)
2. Corrections for any scores that are off by >0.3
3. Brief explanation

Respond as JSON with keys: accuracy, corrections, explanation"""
        
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a nutrition expert validator."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            validation = json.loads(response.choices[0].message.content)
            return validation
            
        except Exception as e:
            logger.error(f"LLM validation error: {e}")
            return {"error": str(e)}


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    
    # Multi-task weights
    goal_prediction_weight: float = 1.0
    risk_prediction_weight: float = 1.0
    recommendation_weight: float = 1.0
    
    # LLM enhancement
    use_llm_distillation: bool = True
    llm_validation_frequency: int = 10  # Every N epochs


class NutritionModelTrainer:
    """
    Trainer for nutrition deep learning models
    """
    
    def __init__(
        self,
        model: PersonalizedNutritionModel,
        config: TrainingConfig
    ):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        goal_loss = 0.0
        risk_loss = 0.0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                food_tokens=batch["food_tokens"],
                user_profile=batch["user_profile"],
                active_goals=batch.get("active_goals"),
                active_diseases=batch.get("active_diseases")
            )
            
            # Compute losses
            goal_loss_batch = F.binary_cross_entropy(
                outputs["goal_scores"],
                batch["goal_targets"]
            )
            
            risk_loss_batch = F.binary_cross_entropy(
                outputs["risk_scores"],
                batch["risk_targets"]
            )
            
            rec_loss_batch = F.binary_cross_entropy(
                outputs["recommendation_score"],
                batch["recommendation_target"]
            )
            
            # Combined loss
            loss = (
                self.config.goal_prediction_weight * goal_loss_batch +
                self.config.risk_prediction_weight * risk_loss_batch +
                self.config.recommendation_weight * rec_loss_batch
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            goal_loss += goal_loss_batch.item()
            risk_loss += risk_loss_batch.item()
        
        self.scheduler.step()
        
        return {
            "total_loss": total_loss / len(train_loader),
            "goal_loss": goal_loss / len(train_loader),
            "risk_loss": risk_loss / len(train_loader)
        }


# ============================================================================
# INFERENCE
# ============================================================================

class NutritionInference:
    """
    Inference engine for nutrition models
    """
    
    def __init__(self, model: PersonalizedNutritionModel):
        self.model = model
        self.model.eval()
        
    @torch.no_grad()
    def predict(
        self,
        food_tokens: torch.Tensor,
        user_profile: torch.Tensor
    ) -> Dict[str, Any]:
        """Make prediction for food-user pair"""
        
        outputs = self.model(food_tokens, user_profile)
        
        return {
            "recommendation_score": outputs["recommendation_score"].item(),
            "goal_scores": outputs["goal_scores"].cpu().numpy(),
            "risk_scores": outputs["risk_scores"].cpu().numpy(),
            "uncertainties": outputs["goal_uncertainties"].cpu().numpy()
        }
    
    def explain_prediction(
        self,
        food_tokens: torch.Tensor,
        user_profile: torch.Tensor,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Generate explanation for prediction"""
        
        outputs = self.model(food_tokens, user_profile)
        
        # Get top contributing goals
        goal_scores = outputs["goal_scores"].cpu().numpy()[0]
        top_goal_indices = np.argsort(goal_scores)[-top_k:][::-1]
        
        # Get top risks
        risk_scores = outputs["risk_scores"].cpu().numpy()[0]
        top_risk_indices = np.argsort(risk_scores)[-top_k:][::-1]
        
        # Attention weights for explainability
        attention = outputs["attention_weights"].cpu().numpy()
        
        return {
            "top_goals": top_goal_indices.tolist(),
            "top_goal_scores": goal_scores[top_goal_indices].tolist(),
            "top_risks": top_risk_indices.tolist(),
            "top_risk_scores": risk_scores[top_risk_indices].tolist(),
            "attention_weights": attention
        }
