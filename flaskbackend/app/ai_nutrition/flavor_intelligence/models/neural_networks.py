"""
Neural Network Models for Flavor Intelligence
============================================

This module implements deep learning models for the Automated Flavor Intelligence
Pipeline. It provides PyTorch-based neural networks for flavor embeddings,
similarity learning, substitution recommendations, and pattern recognition.

Key Features:
- Flavor Embedding Networks (multi-modal encoder-decoder)
- Siamese Networks for similarity learning  
- Graph Neural Networks for ingredient relationships
- Transformer models for recipe understanding
- Variational Autoencoders for flavor space modeling
- Reinforcement Learning for substitution optimization
- Multi-task learning for comprehensive flavor analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
import json
import math
from pathlib import Path
import random
from collections import defaultdict, Counter

# Graph neural network imports
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected

# Transformer imports
from transformers import AutoModel, AutoTokenizer, BertModel, RobertaModel
from transformers import get_linear_schedule_with_warmup

# Scientific computing
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import networkx as nx

from ..models.flavor_data_models import (
    FlavorProfile, SensoryProfile, ChemicalCompound, NutritionData,
    FlavorCategory, MolecularClass
)
from ..layers.relational_layer import Recipe, RecipeIngredient, IngredientCompatibility


@dataclass
class ModelConfig:
    """Configuration for neural network models"""
    
    # Model architecture
    embedding_dim: int = 512
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])
    num_attention_heads: int = 8
    num_transformer_layers: int = 6
    dropout_rate: float = 0.1
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 1e-5
    
    # Loss function weights
    sensory_loss_weight: float = 1.0
    similarity_loss_weight: float = 0.5
    classification_loss_weight: float = 0.3
    reconstruction_loss_weight: float = 0.2
    
    # Data processing
    max_sequence_length: int = 128
    num_negative_samples: int = 5
    augmentation_probability: float = 0.3
    
    # Model selection
    use_pretrained_embeddings: bool = True
    pretrained_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    mixed_precision: bool = True
    
    # Checkpointing
    save_top_k: int = 3
    monitor_metric: str = "val_loss"
    patience: int = 15


class FlavorEmbeddingDataset(Dataset):
    """Dataset for training flavor embedding models"""
    
    def __init__(self, flavor_profiles: List[FlavorProfile], 
                 tokenizer=None, config: ModelConfig = None):
        self.flavor_profiles = flavor_profiles
        self.tokenizer = tokenizer
        self.config = config or ModelConfig()
        
        # Create mappings
        self._create_ingredient_mappings()
        self._create_category_mappings()
        
        # Prepare data
        self._prepare_training_samples()
        
    def _create_ingredient_mappings(self):
        """Create ingredient to index mappings"""
        self.ingredient_to_id = {}
        self.id_to_ingredient = {}
        
        for i, profile in enumerate(self.flavor_profiles):
            if profile.ingredient_id not in self.ingredient_to_id:
                idx = len(self.ingredient_to_id)
                self.ingredient_to_id[profile.ingredient_id] = idx
                self.id_to_ingredient[idx] = profile.ingredient_id
    
    def _create_category_mappings(self):
        """Create category mappings"""
        categories = list(set(profile.primary_category for profile in self.flavor_profiles))
        self.category_to_id = {cat: i for i, cat in enumerate(categories)}
        self.id_to_category = {i: cat for cat, i in self.category_to_id.items()}
        self.num_categories = len(categories)
    
    def _prepare_training_samples(self):
        """Prepare training samples with augmentation"""
        self.samples = []
        
        for profile in self.flavor_profiles:
            sample = {
                'ingredient_id': self.ingredient_to_id[profile.ingredient_id],
                'name': profile.name,
                'sensory_vector': self._extract_sensory_vector(profile.sensory),
                'nutrition_vector': self._extract_nutrition_vector(profile.nutrition),
                'category_id': self.category_to_id[profile.primary_category],
                'confidence': profile.overall_confidence,
                'molecular_features': self._extract_molecular_features(profile)
            }
            
            self.samples.append(sample)
    
    def _extract_sensory_vector(self, sensory: SensoryProfile) -> torch.Tensor:
        """Extract sensory profile as tensor"""
        if not sensory:
            return torch.zeros(8)
        
        vector = torch.tensor([
            sensory.sweet, sensory.sour, sensory.salty, sensory.bitter,
            sensory.umami, sensory.fatty, sensory.spicy, sensory.aromatic
        ], dtype=torch.float32)
        
        return vector
    
    def _extract_nutrition_vector(self, nutrition: NutritionData) -> torch.Tensor:
        """Extract nutrition data as tensor"""
        if not nutrition:
            return torch.zeros(10)
        
        vector = torch.tensor([
            nutrition.calories / 1000.0,  # Normalize
            nutrition.protein / 100.0,
            nutrition.fat / 100.0,
            nutrition.carbohydrates / 100.0,
            nutrition.fiber / 50.0,
            nutrition.sugars / 100.0,
            nutrition.sodium / 1000.0,
            nutrition.calcium / 1000.0,
            nutrition.iron / 20.0,
            nutrition.vitamin_c / 100.0
        ], dtype=torch.float32)
        
        return torch.clamp(vector, 0, 5)  # Clamp to reasonable range
    
    def _extract_molecular_features(self, profile: FlavorProfile) -> torch.Tensor:
        """Extract molecular features from profile"""
        # Mock molecular features - would be computed from chemical compounds
        features = torch.randn(64)  # 64-dimensional molecular fingerprint
        return features
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize ingredient name if tokenizer is provided
        if self.tokenizer:
            encoding = self.tokenizer(
                sample['name'],
                truncation=True,
                padding='max_length',
                max_length=self.config.max_sequence_length,
                return_tensors='pt'
            )
            sample['input_ids'] = encoding['input_ids'].squeeze()
            sample['attention_mask'] = encoding['attention_mask'].squeeze()
        
        return sample


class MultiModalFlavorEncoder(nn.Module):
    """Multi-modal encoder for flavor profiles"""
    
    def __init__(self, config: ModelConfig, vocab_size: int = None):
        super().__init__()
        self.config = config
        
        # Text encoder
        if vocab_size:
            self.text_embedding = nn.Embedding(vocab_size, config.embedding_dim)
            self.text_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.embedding_dim,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.hidden_dims[0],
                    dropout=config.dropout_rate
                ),
                num_layers=config.num_transformer_layers
            )
        else:
            # Use pre-trained model
            self.text_encoder = AutoModel.from_pretrained(config.pretrained_model_name)
        
        # Sensory encoder
        self.sensory_encoder = nn.Sequential(
            nn.Linear(8, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[1], config.embedding_dim)
        )
        
        # Nutrition encoder
        self.nutrition_encoder = nn.Sequential(
            nn.Linear(10, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[1], config.embedding_dim)
        )
        
        # Molecular encoder
        self.molecular_encoder = nn.Sequential(
            nn.Linear(64, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[1], config.embedding_dim)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.embedding_dim * 4, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[0], config.embedding_dim),
            nn.Tanh()  # Normalize embeddings
        )
        
        # Attention weights for modality fusion
        self.modality_attention = nn.Linear(config.embedding_dim * 4, 4)
        
    def forward(self, input_ids=None, attention_mask=None, 
                sensory_features=None, nutrition_features=None, 
                molecular_features=None):
        
        embeddings = []
        
        # Text encoding
        if input_ids is not None:
            if hasattr(self.text_encoder, 'pooler_output'):
                # Pre-trained transformer
                text_output = self.text_encoder(input_ids=input_ids, 
                                              attention_mask=attention_mask)
                text_emb = text_output.pooler_output
            else:
                # Custom transformer
                embedded = self.text_embedding(input_ids)
                text_emb = self.text_encoder(embedded.transpose(0, 1)).mean(dim=0)
            
            embeddings.append(text_emb)
        
        # Sensory encoding
        if sensory_features is not None:
            sensory_emb = self.sensory_encoder(sensory_features)
            embeddings.append(sensory_emb)
        
        # Nutrition encoding
        if nutrition_features is not None:
            nutrition_emb = self.nutrition_encoder(nutrition_features)
            embeddings.append(nutrition_emb)
        
        # Molecular encoding
        if molecular_features is not None:
            molecular_emb = self.molecular_encoder(molecular_features)
            embeddings.append(molecular_emb)
        
        # Concatenate all embeddings
        combined_emb = torch.cat(embeddings, dim=-1)
        
        # Apply attention-based fusion
        attention_weights = F.softmax(self.modality_attention(combined_emb), dim=-1)
        
        # Weighted combination of modalities
        weighted_emb = combined_emb.view(-1, len(embeddings), self.config.embedding_dim)
        weighted_emb = (weighted_emb * attention_weights.unsqueeze(-1)).sum(dim=1)
        
        # Final fusion
        fused_emb = self.fusion_layer(combined_emb)
        
        return fused_emb, attention_weights


class FlavorSimilarityNetwork(nn.Module):
    """Siamese network for flavor similarity learning"""
    
    def __init__(self, encoder: MultiModalFlavorEncoder, config: ModelConfig):
        super().__init__()
        self.encoder = encoder
        self.config = config
        
        # Similarity prediction head
        self.similarity_head = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[1], config.hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[2], 1),
            nn.Sigmoid()
        )
        
        # Distance metric learning
        self.distance_metric = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[1], config.embedding_dim)
        )
    
    def forward(self, anchor_data, positive_data, negative_data=None):
        # Encode all inputs
        anchor_emb, _ = self.encoder(**anchor_data)
        positive_emb, _ = self.encoder(**positive_data)
        
        # Transform embeddings for distance learning
        anchor_transformed = self.distance_metric(anchor_emb)
        positive_transformed = self.distance_metric(positive_emb)
        
        # Calculate similarity
        similarity_input = torch.cat([anchor_transformed, positive_transformed], dim=-1)
        similarity_score = self.similarity_head(similarity_input)
        
        results = {
            'anchor_embedding': anchor_emb,
            'positive_embedding': positive_emb,
            'similarity_score': similarity_score,
            'anchor_transformed': anchor_transformed,
            'positive_transformed': positive_transformed
        }
        
        if negative_data is not None:
            negative_emb, _ = self.encoder(**negative_data)
            negative_transformed = self.distance_metric(negative_emb)
            
            negative_similarity_input = torch.cat([anchor_transformed, negative_transformed], dim=-1)
            negative_similarity_score = self.similarity_head(negative_similarity_input)
            
            results.update({
                'negative_embedding': negative_emb,
                'negative_similarity_score': negative_similarity_score,
                'negative_transformed': negative_transformed
            })
        
        return results


class FlavorGraphNeuralNetwork(nn.Module):
    """Graph Neural Network for ingredient relationship modeling"""
    
    def __init__(self, config: ModelConfig, num_ingredients: int):
        super().__init__()
        self.config = config
        self.num_ingredients = num_ingredients
        
        # Initial node embeddings
        self.node_embedding = nn.Embedding(num_ingredients, config.embedding_dim)
        
        # Graph convolution layers
        self.gconv_layers = nn.ModuleList([
            GCNConv(config.embedding_dim, config.hidden_dims[0]),
            GCNConv(config.hidden_dims[0], config.hidden_dims[1]),
            GCNConv(config.hidden_dims[1], config.embedding_dim)
        ])
        
        # Attention mechanism
        self.attention_layers = nn.ModuleList([
            GATConv(config.embedding_dim, config.hidden_dims[0] // config.num_attention_heads, 
                   heads=config.num_attention_heads, dropout=config.dropout_rate),
            GATConv(config.hidden_dims[0], config.embedding_dim, 
                   heads=1, dropout=config.dropout_rate)
        ])
        
        # Prediction heads
        self.compatibility_head = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[1], 1),
            nn.Sigmoid()
        )
        
        # Graph-level prediction
        self.graph_classifier = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[1], config.hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[2], 1)
        )
        
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, node_ids, edge_index, edge_weight=None, batch=None):
        # Get initial node embeddings
        x = self.node_embedding(node_ids)
        
        # Apply graph convolution layers
        for i, gconv in enumerate(self.gconv_layers):
            x = gconv(x, edge_index, edge_weight)
            if i < len(self.gconv_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Apply attention layers
        att_x = x
        for i, att_layer in enumerate(self.attention_layers):
            att_x = att_layer(att_x, edge_index)
            if i < len(self.attention_layers) - 1:
                att_x = F.relu(att_x)
                att_x = self.dropout(att_x)
        
        # Combine GCN and attention outputs
        combined_x = x + att_x
        
        # Graph-level representation
        if batch is not None:
            graph_repr = global_mean_pool(combined_x, batch)
        else:
            graph_repr = combined_x.mean(dim=0, keepdim=True)
        
        return {
            'node_embeddings': combined_x,
            'graph_representation': graph_repr,
            'attention_embeddings': att_x
        }
    
    def predict_compatibility(self, ingredient1_emb, ingredient2_emb):
        """Predict compatibility between two ingredients"""
        combined = torch.cat([ingredient1_emb, ingredient2_emb], dim=-1)
        return self.compatibility_head(combined)


class RecipeTransformer(nn.Module):
    """Transformer model for recipe understanding and generation"""
    
    def __init__(self, config: ModelConfig, vocab_size: int, 
                 max_ingredients: int = 50):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.max_ingredients = max_ingredients
        
        # Ingredient embeddings
        self.ingredient_embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(max_ingredients, config.embedding_dim)
        
        # Transformer encoder for recipes
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_dims[0],
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_transformer_layers
        )
        
        # Recipe classification head
        self.recipe_classifier = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[1], 10)  # 10 cuisine types
        )
        
        # Ingredient prediction head (for recipe completion)
        self.ingredient_predictor = nn.Linear(config.embedding_dim, vocab_size)
        
        # Recipe quality scorer
        self.quality_scorer = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, ingredient_ids, attention_mask=None):
        batch_size, seq_len = ingredient_ids.shape
        
        # Create embeddings
        ingredient_emb = self.ingredient_embedding(ingredient_ids)
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=ingredient_ids.device)
        position_emb = self.position_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine embeddings
        embeddings = ingredient_emb + position_emb
        
        # Apply transformer
        if attention_mask is not None:
            # Convert attention mask to transformer format
            transformer_mask = (attention_mask == 0)
        else:
            transformer_mask = None
        
        encoded = self.transformer_encoder(embeddings, src_key_padding_mask=transformer_mask)
        
        # Pool for recipe-level representation
        if attention_mask is not None:
            # Masked average pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded)
            recipe_repr = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            recipe_repr = encoded.mean(dim=1)
        
        # Predictions
        cuisine_logits = self.recipe_classifier(recipe_repr)
        ingredient_logits = self.ingredient_predictor(encoded)
        quality_score = self.quality_scorer(recipe_repr)
        
        return {
            'recipe_representation': recipe_repr,
            'encoded_ingredients': encoded,
            'cuisine_logits': cuisine_logits,
            'ingredient_logits': ingredient_logits,
            'quality_score': quality_score
        }


class FlavorVariationalAutoEncoder(nn.Module):
    """Variational Autoencoder for flavor space modeling"""
    
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.latent_dim = config.embedding_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[1], config.hidden_dims[2]),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(config.hidden_dims[2], self.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dims[2], self.latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, config.hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[2], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dims[1], config.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[0], input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent parameters"""
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        
        return {
            'reconstruction': recon_x,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def sample(self, num_samples: int, device: str = 'cpu'):
        """Sample new flavor profiles from the latent space"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


class FlavorIntelligenceLightningModule(pl.LightningModule):
    """PyTorch Lightning module for training flavor intelligence models"""
    
    def __init__(self, model_type: str, config: ModelConfig, 
                 dataset_info: Dict, **model_kwargs):
        super().__init__()
        self.config = config
        self.model_type = model_type
        self.dataset_info = dataset_info
        
        # Initialize model based on type
        if model_type == "multimodal_encoder":
            self.model = MultiModalFlavorEncoder(config, **model_kwargs)
        elif model_type == "similarity_network":
            encoder = MultiModalFlavorEncoder(config, **model_kwargs)
            self.model = FlavorSimilarityNetwork(encoder, config)
        elif model_type == "graph_network":
            self.model = FlavorGraphNeuralNetwork(config, **model_kwargs)
        elif model_type == "recipe_transformer":
            self.model = RecipeTransformer(config, **model_kwargs)
        elif model_type == "flavor_vae":
            self.model = FlavorVariationalAutoEncoder(config, **model_kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        
        # Separate learning rates for different components
        param_groups = []
        
        if hasattr(self.model, 'text_encoder'):
            # Lower learning rate for pre-trained components
            param_groups.append({
                'params': self.model.text_encoder.parameters(),
                'lr': self.config.learning_rate * 0.1
            })
        
        # Higher learning rate for newly initialized components
        other_params = []
        for name, param in self.model.named_parameters():
            if not name.startswith('text_encoder'):
                other_params.append(param)
        
        param_groups.append({
            'params': other_params,
            'lr': self.config.learning_rate
        })
        
        optimizer = optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.learning_rate * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
    
    def training_step(self, batch, batch_idx):
        """Training step for different model types"""
        
        if self.model_type == "multimodal_encoder":
            return self._multimodal_training_step(batch, batch_idx)
        elif self.model_type == "similarity_network":
            return self._similarity_training_step(batch, batch_idx)
        elif self.model_type == "graph_network":
            return self._graph_training_step(batch, batch_idx)
        elif self.model_type == "recipe_transformer":
            return self._recipe_training_step(batch, batch_idx)
        elif self.model_type == "flavor_vae":
            return self._vae_training_step(batch, batch_idx)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _multimodal_training_step(self, batch, batch_idx):
        """Training step for multimodal encoder"""
        
        # Forward pass
        embeddings, attention_weights = self.model(
            input_ids=batch.get('input_ids'),
            attention_mask=batch.get('attention_mask'),
            sensory_features=batch.get('sensory_vector'),
            nutrition_features=batch.get('nutrition_vector'),
            molecular_features=batch.get('molecular_features')
        )
        
        # Classification loss
        category_logits = torch.matmul(embeddings, self.model.fusion_layer[-2].weight.t())
        classification_loss = self.ce_loss(category_logits, batch['category_id'])
        
        # Reconstruction losses for each modality
        sensory_recon = torch.matmul(embeddings, self.model.sensory_encoder[-1].weight.t())
        sensory_loss = self.mse_loss(sensory_recon, batch['sensory_vector'])
        
        nutrition_recon = torch.matmul(embeddings, self.model.nutrition_encoder[-1].weight.t())
        nutrition_loss = self.mse_loss(nutrition_recon, batch['nutrition_vector'])
        
        # Total loss
        total_loss = (
            self.config.classification_loss_weight * classification_loss +
            self.config.sensory_loss_weight * sensory_loss +
            self.config.reconstruction_loss_weight * nutrition_loss
        )
        
        # Logging
        self.log('train_loss', total_loss)
        self.log('train_classification_loss', classification_loss)
        self.log('train_sensory_loss', sensory_loss)
        self.log('train_nutrition_loss', nutrition_loss)
        
        return total_loss
    
    def _similarity_training_step(self, batch, batch_idx):
        """Training step for similarity network"""
        
        # Prepare triplet data
        anchor_data = {key: batch[f'anchor_{key}'] for key in ['input_ids', 'sensory_vector', 'nutrition_vector']}
        positive_data = {key: batch[f'positive_{key}'] for key in ['input_ids', 'sensory_vector', 'nutrition_vector']}
        negative_data = {key: batch[f'negative_{key}'] for key in ['input_ids', 'sensory_vector', 'nutrition_vector']}
        
        # Forward pass
        results = self.model(anchor_data, positive_data, negative_data)
        
        # Triplet loss
        triplet_loss = self.triplet_loss(
            results['anchor_transformed'],
            results['positive_transformed'],
            results['negative_transformed']
        )
        
        # Similarity prediction losses
        positive_sim_loss = self.bce_loss(
            results['similarity_score'], 
            torch.ones_like(results['similarity_score'])
        )
        
        negative_sim_loss = self.bce_loss(
            results['negative_similarity_score'], 
            torch.zeros_like(results['negative_similarity_score'])
        )
        
        # Total loss
        total_loss = (
            triplet_loss + 
            self.config.similarity_loss_weight * (positive_sim_loss + negative_sim_loss)
        )
        
        # Logging
        self.log('train_loss', total_loss)
        self.log('train_triplet_loss', triplet_loss)
        self.log('train_similarity_loss', positive_sim_loss + negative_sim_loss)
        
        return total_loss
    
    def _vae_training_step(self, batch, batch_idx):
        """Training step for VAE"""
        
        # Prepare input (concatenated sensory and nutrition features)
        input_features = torch.cat([
            batch['sensory_vector'], 
            batch['nutrition_vector']
        ], dim=-1)
        
        # Forward pass
        results = self.model(input_features)
        
        # Reconstruction loss
        recon_loss = self.mse_loss(results['reconstruction'], input_features)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(
            1 + results['logvar'] - results['mu'].pow(2) - results['logvar'].exp()
        ) / input_features.size(0)
        
        # Total VAE loss
        total_loss = recon_loss + 0.01 * kl_loss  # Beta-VAE with beta=0.01
        
        # Logging
        self.log('train_loss', total_loss)
        self.log('train_recon_loss', recon_loss)
        self.log('train_kl_loss', kl_loss)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        
        # Similar to training step but no gradient computation
        with torch.no_grad():
            if self.model_type == "multimodal_encoder":
                loss = self._multimodal_training_step(batch, batch_idx)
            elif self.model_type == "similarity_network":
                loss = self._similarity_training_step(batch, batch_idx)
            elif self.model_type == "flavor_vae":
                loss = self._vae_training_step(batch, batch_idx)
            else:
                loss = torch.tensor(0.0)
        
        self.log('val_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        """Prediction step for inference"""
        
        if self.model_type == "multimodal_encoder":
            embeddings, attention_weights = self.model(
                input_ids=batch.get('input_ids'),
                attention_mask=batch.get('attention_mask'),
                sensory_features=batch.get('sensory_vector'),
                nutrition_features=batch.get('nutrition_vector'),
                molecular_features=batch.get('molecular_features')
            )
            return {'embeddings': embeddings, 'attention_weights': attention_weights}
        
        elif self.model_type == "flavor_vae":
            input_features = torch.cat([
                batch['sensory_vector'], 
                batch['nutrition_vector']
            ], dim=-1)
            results = self.model(input_features)
            return results
        
        else:
            return self.model(batch)


class FlavorModelTrainer:
    """High-level trainer for flavor intelligence models"""
    
    def __init__(self, config: ModelConfig, output_dir: str = "models"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Training statistics
        self.training_stats = {
            'models_trained': 0,
            'total_training_time_hours': 0.0,
            'best_model_scores': {}
        }
    
    def train_multimodal_encoder(self, dataset: FlavorEmbeddingDataset) -> pl.LightningModule:
        """Train multimodal flavor encoder"""
        
        self.logger.info("Training multimodal flavor encoder")
        
        # Create Lightning module
        model = FlavorIntelligenceLightningModule(
            model_type="multimodal_encoder",
            config=self.config,
            dataset_info={'num_samples': len(dataset)},
            vocab_size=len(dataset.ingredient_to_id)
        )
        
        # Data loaders
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_batch
        )
        
        # Training setup
        trainer = self._create_trainer("multimodal_encoder")
        
        # Train model
        start_time = datetime.now()
        trainer.fit(model, train_loader)
        training_time = (datetime.now() - start_time).total_seconds() / 3600
        
        # Update statistics
        self.training_stats['models_trained'] += 1
        self.training_stats['total_training_time_hours'] += training_time
        
        self.logger.info(f"Multimodal encoder training completed in {training_time:.2f} hours")
        
        return model
    
    def train_similarity_network(self, dataset: FlavorEmbeddingDataset) -> pl.LightningModule:
        """Train flavor similarity network"""
        
        self.logger.info("Training flavor similarity network")
        
        # Create triplet dataset
        triplet_dataset = self._create_triplet_dataset(dataset)
        
        # Create Lightning module
        model = FlavorIntelligenceLightningModule(
            model_type="similarity_network",
            config=self.config,
            dataset_info={'num_samples': len(triplet_dataset)},
            vocab_size=len(dataset.ingredient_to_id)
        )
        
        # Data loader
        train_loader = DataLoader(
            triplet_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_triplet_batch
        )
        
        # Training setup
        trainer = self._create_trainer("similarity_network")
        
        # Train model
        start_time = datetime.now()
        trainer.fit(model, train_loader)
        training_time = (datetime.now() - start_time).total_seconds() / 3600
        
        self.training_stats['models_trained'] += 1
        self.training_stats['total_training_time_hours'] += training_time
        
        self.logger.info(f"Similarity network training completed in {training_time:.2f} hours")
        
        return model
    
    def train_flavor_vae(self, dataset: FlavorEmbeddingDataset) -> pl.LightningModule:
        """Train flavor variational autoencoder"""
        
        self.logger.info("Training flavor VAE")
        
        # Create Lightning module
        model = FlavorIntelligenceLightningModule(
            model_type="flavor_vae",
            config=self.config,
            dataset_info={'num_samples': len(dataset)},
            input_dim=18  # 8 sensory + 10 nutrition features
        )
        
        # Data loader
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_batch
        )
        
        # Training setup
        trainer = self._create_trainer("flavor_vae")
        
        # Train model
        start_time = datetime.now()
        trainer.fit(model, train_loader)
        training_time = (datetime.now() - start_time).total_seconds() / 3600
        
        self.training_stats['models_trained'] += 1
        self.training_stats['total_training_time_hours'] += training_time
        
        self.logger.info(f"Flavor VAE training completed in {training_time:.2f} hours")
        
        return model
    
    def _create_trainer(self, model_name: str) -> pl.Trainer:
        """Create PyTorch Lightning trainer"""
        
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_dir / model_name,
            filename='{epoch}-{val_loss:.3f}',
            monitor=self.config.monitor_metric,
            save_top_k=self.config.save_top_k,
            mode='min'
        )
        
        early_stopping = EarlyStopping(
            monitor=self.config.monitor_metric,
            patience=self.config.patience,
            mode='min'
        )
        
        # Trainer configuration
        trainer_kwargs = {
            'max_epochs': self.config.num_epochs,
            'callbacks': [checkpoint_callback, early_stopping],
            'accelerator': 'gpu' if self.config.device == 'cuda' else 'cpu',
            'precision': 16 if self.config.mixed_precision else 32,
            'log_every_n_steps': 50,
            'enable_progress_bar': True
        }
        
        if self.config.device == 'cuda' and torch.cuda.device_count() > 1:
            trainer_kwargs['strategy'] = 'ddp'
            trainer_kwargs['devices'] = torch.cuda.device_count()
        
        return pl.Trainer(**trainer_kwargs)
    
    def _create_triplet_dataset(self, base_dataset: FlavorEmbeddingDataset) -> Dataset:
        """Create triplet dataset for similarity learning"""
        
        class TripletDataset(Dataset):
            def __init__(self, base_dataset, num_negatives=5):
                self.base_dataset = base_dataset
                self.num_negatives = num_negatives
                
                # Group samples by category for positive sampling
                self.category_groups = defaultdict(list)
                for i, sample in enumerate(base_dataset.samples):
                    self.category_groups[sample['category_id']].append(i)
                
                # Create triplets
                self.triplets = self._create_triplets()
            
            def _create_triplets(self):
                triplets = []
                
                for anchor_idx, anchor_sample in enumerate(self.base_dataset.samples):
                    anchor_category = anchor_sample['category_id']
                    
                    # Positive: same category, different ingredient
                    positive_candidates = [
                        idx for idx in self.category_groups[anchor_category]
                        if idx != anchor_idx
                    ]
                    
                    if positive_candidates:
                        positive_idx = random.choice(positive_candidates)
                        
                        # Negatives: different categories
                        other_categories = [
                            cat for cat in self.category_groups.keys() 
                            if cat != anchor_category
                        ]
                        
                        for _ in range(self.num_negatives):
                            if other_categories:
                                neg_category = random.choice(other_categories)
                                negative_idx = random.choice(self.category_groups[neg_category])
                                
                                triplets.append((anchor_idx, positive_idx, negative_idx))
                
                return triplets
            
            def __len__(self):
                return len(self.triplets)
            
            def __getitem__(self, idx):
                anchor_idx, positive_idx, negative_idx = self.triplets[idx]
                
                anchor = self.base_dataset.samples[anchor_idx]
                positive = self.base_dataset.samples[positive_idx]
                negative = self.base_dataset.samples[negative_idx]
                
                return {
                    'anchor_input_ids': anchor.get('input_ids', torch.zeros(128)),
                    'anchor_sensory_vector': anchor['sensory_vector'],
                    'anchor_nutrition_vector': anchor['nutrition_vector'],
                    
                    'positive_input_ids': positive.get('input_ids', torch.zeros(128)),
                    'positive_sensory_vector': positive['sensory_vector'],
                    'positive_nutrition_vector': positive['nutrition_vector'],
                    
                    'negative_input_ids': negative.get('input_ids', torch.zeros(128)),
                    'negative_sensory_vector': negative['sensory_vector'],
                    'negative_nutrition_vector': negative['nutrition_vector']
                }
        
        return TripletDataset(base_dataset, self.config.num_negative_samples)
    
    def _collate_batch(self, batch):
        """Collate function for regular batches"""
        collated = {}
        
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        
        return collated
    
    def _collate_triplet_batch(self, batch):
        """Collate function for triplet batches"""
        collated = {}
        
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        
        return collated
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        return self.training_stats


# Utility functions for model deployment and inference

class FlavorIntelligenceInference:
    """Inference engine for trained flavor intelligence models"""
    
    def __init__(self, model_paths: Dict[str, str], config: ModelConfig):
        self.config = config
        self.models = {}
        
        # Load trained models
        for model_name, path in model_paths.items():
            self.models[model_name] = self._load_model(path, model_name)
        
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self, model_path: str, model_name: str) -> pl.LightningModule:
        """Load trained model from checkpoint"""
        try:
            model = FlavorIntelligenceLightningModule.load_from_checkpoint(
                model_path,
                model_type=model_name,
                config=self.config,
                dataset_info={}
            )
            model.eval()
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def get_flavor_embedding(self, flavor_profile: FlavorProfile) -> Optional[torch.Tensor]:
        """Get embedding for a flavor profile"""
        if 'multimodal_encoder' not in self.models:
            return None
        
        model = self.models['multimodal_encoder']
        
        # Prepare input data
        input_data = {
            'sensory_features': self._extract_sensory_tensor(flavor_profile.sensory).unsqueeze(0),
            'nutrition_features': self._extract_nutrition_tensor(flavor_profile.nutrition).unsqueeze(0),
            'molecular_features': torch.randn(1, 64)  # Mock molecular features
        }
        
        with torch.no_grad():
            embedding, _ = model.model(
                sensory_features=input_data['sensory_features'],
                nutrition_features=input_data['nutrition_features'],
                molecular_features=input_data['molecular_features']
            )
        
        return embedding.squeeze(0)
    
    def calculate_similarity(self, profile1: FlavorProfile, 
                           profile2: FlavorProfile) -> float:
        """Calculate similarity between two flavor profiles"""
        
        emb1 = self.get_flavor_embedding(profile1)
        emb2 = self.get_flavor_embedding(profile2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return float(similarity.item())
    
    def find_substitutes(self, target_profile: FlavorProfile,
                        candidate_profiles: List[FlavorProfile],
                        top_k: int = 5) -> List[Tuple[FlavorProfile, float]]:
        """Find best substitutes for a target ingredient"""
        
        target_embedding = self.get_flavor_embedding(target_profile)
        if target_embedding is None:
            return []
        
        similarities = []
        for candidate in candidate_profiles:
            candidate_embedding = self.get_flavor_embedding(candidate)
            if candidate_embedding is not None:
                sim = F.cosine_similarity(
                    target_embedding.unsqueeze(0), 
                    candidate_embedding.unsqueeze(0)
                )
                similarities.append((candidate, float(sim.item())))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _extract_sensory_tensor(self, sensory: SensoryProfile) -> torch.Tensor:
        """Extract sensory profile as tensor"""
        if not sensory:
            return torch.zeros(8)
        
        return torch.tensor([
            sensory.sweet, sensory.sour, sensory.salty, sensory.bitter,
            sensory.umami, sensory.fatty, sensory.spicy, sensory.aromatic
        ], dtype=torch.float32)
    
    def _extract_nutrition_tensor(self, nutrition: NutritionData) -> torch.Tensor:
        """Extract nutrition data as tensor"""
        if not nutrition:
            return torch.zeros(10)
        
        vector = torch.tensor([
            nutrition.calories / 1000.0,
            nutrition.protein / 100.0,
            nutrition.fat / 100.0,
            nutrition.carbohydrates / 100.0,
            nutrition.fiber / 50.0,
            nutrition.sugars / 100.0,
            nutrition.sodium / 1000.0,
            nutrition.calcium / 1000.0,
            nutrition.iron / 20.0,
            nutrition.vitamin_c / 100.0
        ], dtype=torch.float32)
        
        return torch.clamp(vector, 0, 5)


# Export key classes and functions
__all__ = [
    'ModelConfig', 'FlavorEmbeddingDataset', 'MultiModalFlavorEncoder',
    'FlavorSimilarityNetwork', 'FlavorGraphNeuralNetwork', 'RecipeTransformer',
    'FlavorVariationalAutoEncoder', 'FlavorIntelligenceLightningModule',
    'FlavorModelTrainer', 'FlavorIntelligenceInference'
]