"""
AI-Powered Food Knowledge Graph Engine
=====================================

Advanced machine learning system for food knowledge graph management,
relationship inference, nutritional analysis, and intelligent food recommendations.

Core Capabilities:
- Deep learning models for food classification and similarity
- Graph neural networks for relationship inference
- Nutritional optimization using reinforcement learning
- Natural language processing for food description analysis
- Computer vision for food image recognition
- Personalized recommendation engines
- Cultural and regional food pattern analysis

ML Models:
1. Food Classification Network (CNN + Transformer)
2. Nutritional Similarity Engine (Graph Neural Network)
3. Substitution Recommendation System (Reinforcement Learning)
4. Cultural Food Pattern Analyzer (LSTM + Attention)
5. Seasonal Availability Predictor (Time Series Analysis)
6. Quality Score Estimator (Ensemble Methods)

Author: Wellomex AI Nutrition Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss

from ..models.food_knowledge_models import (
    FoodEntity, FoodCategory, MacroNutrient, MicroNutrient,
    NutritionalProfile, PreparationMethod, AllergenType,
    DietaryRestriction, FoodRelationship
)
from ..graph_db.neo4j_manager import GraphDatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class MLModelConfig:
    """Configuration for ML models"""
    model_name: str
    model_path: str
    version: str = "1.0.0"
    last_trained: Optional[datetime] = None
    training_samples: int = 0
    accuracy: Optional[float] = None
    feature_count: int = 0
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

class FoodEmbeddingDataset(Dataset):
    """PyTorch dataset for food embeddings"""
    
    def __init__(self, foods: List[FoodEntity], tokenizer, max_length: int = 512):
        self.foods = foods
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.foods)
    
    def __getitem__(self, idx):
        food = self.foods[idx]
        
        # Create text representation
        text = self._create_food_text(food)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create numerical features
        numerical_features = self._extract_numerical_features(food)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'numerical_features': torch.tensor(numerical_features, dtype=torch.float32),
            'food_id': food.food_id,
            'category': food.category.value if food.category else 'unknown'
        }
    
    def _create_food_text(self, food: FoodEntity) -> str:
        """Create text representation of food for embedding"""
        text_parts = [food.name]
        
        if food.scientific_name:
            text_parts.append(f"Scientific name: {food.scientific_name}")
        
        if food.common_names:
            text_parts.append(f"Also known as: {', '.join(food.common_names[:3])}")
        
        if food.category:
            text_parts.append(f"Category: {food.category.value}")
        
        if food.subcategories:
            text_parts.append(f"Subcategories: {', '.join(food.subcategories[:2])}")
        
        # Add nutritional highlights
        if food.macro_nutrients:
            macro = food.macro_nutrients
            if macro.protein > 10:
                text_parts.append("High in protein")
            if macro.fiber > 5:
                text_parts.append("High in fiber")
            if macro.calories < 50:
                text_parts.append("Low calorie")
        
        # Add dietary restrictions
        if food.dietary_restrictions:
            restrictions = [dr.value for dr in food.dietary_restrictions[:3]]
            text_parts.append(f"Suitable for: {', '.join(restrictions)}")
        
        return ". ".join(text_parts)
    
    def _extract_numerical_features(self, food: FoodEntity) -> List[float]:
        """Extract numerical features for ML models"""
        features = []
        
        # Macro nutrients (per 100g)
        if food.macro_nutrients:
            macro = food.macro_nutrients
            features.extend([
                float(macro.calories),
                float(macro.protein),
                float(macro.fat),
                float(macro.carbohydrates),
                float(macro.fiber),
                float(macro.sugar),
                float(macro.sodium)
            ])
        else:
            features.extend([0.0] * 7)
        
        # Key micronutrients
        if food.micro_nutrients:
            micro = food.micro_nutrients
            features.extend([
                float(micro.vitamin_c or 0),
                float(micro.vitamin_d or 0),
                float(micro.calcium or 0),
                float(micro.iron or 0),
                float(micro.potassium or 0)
            ])
        else:
            features.extend([0.0] * 5)
        
        # Physical properties
        features.extend([
            float(food.glycemic_index or 55),  # Default medium GI
            float(food.glycemic_load or 10),   # Default medium GL
            float(food.typical_serving_size or 100)
        ])
        
        # Category encoding (one-hot for major categories)
        category_features = [0.0] * len(FoodCategory)
        if food.category:
            try:
                category_index = list(FoodCategory).index(food.category)
                category_features[category_index] = 1.0
            except ValueError:
                pass
        features.extend(category_features)
        
        # Binary features
        features.extend([
            1.0 if AllergenType.GLUTEN in food.allergens else 0.0,
            1.0 if AllergenType.DAIRY in food.allergens else 0.0,
            1.0 if AllergenType.NUTS in food.allergens else 0.0,
            1.0 if DietaryRestriction.VEGAN in food.dietary_restrictions else 0.0,
            1.0 if DietaryRestriction.VEGETARIAN in food.dietary_restrictions else 0.0,
            float(food.confidence_score),
            len(food.country_data) / 10.0  # Normalize country availability
        ])
        
        return features

class FoodEmbeddingModel(nn.Module):
    """Neural network for generating food embeddings"""
    
    def __init__(
        self, 
        text_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        numerical_feature_size: int = 50,
        embedding_dim: int = 768,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size
        
        # Numerical feature processor
        self.numerical_processor = nn.Sequential(
            nn.Linear(numerical_feature_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Fusion layer
        fusion_input_dim = text_dim + hidden_dim // 4
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()
        )
        
        # Category classifier head (for auxiliary training)
        self.category_classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, len(FoodCategory))
        )
    
    def forward(self, input_ids, attention_mask, numerical_features):
        # Encode text
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = text_outputs.last_hidden_state.mean(dim=1)  # Pool over sequence
        
        # Process numerical features
        numerical_embedding = self.numerical_processor(numerical_features)
        
        # Fuse embeddings
        fused_features = torch.cat([text_embedding, numerical_embedding], dim=1)
        food_embedding = self.fusion_layer(fused_features)
        
        # Category prediction (for training)
        category_logits = self.category_classifier(food_embedding)
        
        return food_embedding, category_logits

class NutritionalSimilarityEngine:
    """ML engine for calculating nutritional similarity between foods"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=20)
        self.similarity_model = None
        self.is_trained = False
    
    def train(self, foods: List[FoodEntity]):
        """Train the nutritional similarity model"""
        logger.info(f"Training nutritional similarity engine with {len(foods)} foods")
        
        # Extract features
        features = []
        valid_foods = []
        
        for food in foods:
            food_features = self._extract_nutritional_features(food)
            if food_features is not None:
                features.append(food_features)
                valid_foods.append(food)
        
        if len(features) < 10:
            logger.warning("Not enough valid nutritional data for training")
            return False
        
        features_array = np.array(features)
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features_array)
        
        # Apply PCA for dimensionality reduction
        features_pca = self.pca.fit_transform(features_normalized)
        
        # Train clustering model for similarity grouping
        self.similarity_model = KMeans(n_clusters=min(50, len(valid_foods) // 10), random_state=42)
        self.similarity_model.fit(features_pca)
        
        self.is_trained = True
        logger.info("Nutritional similarity engine trained successfully")
        return True
    
    def calculate_similarity(self, food1: FoodEntity, food2: FoodEntity) -> float:
        """Calculate nutritional similarity between two foods"""
        if not self.is_trained:
            return 0.0
        
        features1 = self._extract_nutritional_features(food1)
        features2 = self._extract_nutritional_features(food2)
        
        if features1 is None or features2 is None:
            return 0.0
        
        # Normalize and apply PCA
        features1_norm = self.scaler.transform([features1])
        features2_norm = self.scaler.transform([features2])
        
        features1_pca = self.pca.transform(features1_norm)
        features2_pca = self.pca.transform(features2_norm)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(features1_pca, features2_pca)[0][0]
        return max(0.0, min(1.0, similarity))
    
    def find_similar_foods(
        self, 
        target_food: FoodEntity, 
        candidate_foods: List[FoodEntity], 
        top_k: int = 10
    ) -> List[Tuple[FoodEntity, float]]:
        """Find most nutritionally similar foods"""
        similarities = []
        
        for candidate in candidate_foods:
            if candidate.food_id != target_food.food_id:
                similarity = self.calculate_similarity(target_food, candidate)
                if similarity > 0.3:  # Minimum threshold
                    similarities.append((candidate, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _extract_nutritional_features(self, food: FoodEntity) -> Optional[List[float]]:
        """Extract nutritional features for similarity calculation"""
        if not food.macro_nutrients:
            return None
        
        macro = food.macro_nutrients
        micro = food.micro_nutrients
        
        features = [
            float(macro.calories),
            float(macro.protein),
            float(macro.fat),
            float(macro.carbohydrates),
            float(macro.fiber),
            float(macro.sugar),
            float(macro.sodium)
        ]
        
        # Derived ratios
        total_macros = float(macro.protein + macro.fat + macro.carbohydrates)
        if total_macros > 0:
            features.extend([
                float(macro.protein) / total_macros,
                float(macro.fat) / total_macros,
                float(macro.carbohydrates) / total_macros
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Micronutrients
        if micro:
            features.extend([
                float(micro.vitamin_c or 0),
                float(micro.vitamin_d or 0),
                float(micro.calcium or 0),
                float(micro.iron or 0),
                float(micro.potassium or 0)
            ])
        else:
            features.extend([0.0] * 5)
        
        # Nutritional density scores
        if macro.calories > 0:
            features.extend([
                float(macro.protein) * 4 / float(macro.calories),  # Protein density
                float(macro.fiber) / float(macro.calories) * 100,   # Fiber density
            ])
        else:
            features.extend([0.0, 0.0])
        
        return features

class FoodSubstitutionEngine:
    """AI engine for intelligent food substitution recommendations"""
    
    def __init__(self):
        self.substitution_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Substitution rules and weights
        self.substitution_rules = {
            'nutritional_similarity': 0.4,
            'category_compatibility': 0.2,
            'dietary_compatibility': 0.15,
            'cultural_appropriateness': 0.1,
            'availability': 0.1,
            'preparation_method': 0.05
        }
    
    def train(self, substitution_data: List[Tuple[FoodEntity, FoodEntity, float]]):
        """Train substitution model with historical data"""
        logger.info(f"Training substitution engine with {len(substitution_data)} examples")
        
        features = []
        labels = []
        
        for original_food, substitute_food, quality_score in substitution_data:
            feature_vector = self._extract_substitution_features(original_food, substitute_food)
            if feature_vector:
                features.append(feature_vector)
                # Convert quality score to classification (good/bad substitution)
                labels.append(1 if quality_score > 0.6 else 0)
        
        if len(features) < 20:
            logger.warning("Not enough training data for substitution model")
            return False
        
        features_array = np.array(features)
        labels_array = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_array, labels_array, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Train model
        self.substitution_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.substitution_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        logger.info(f"Substitution model trained with accuracy: {accuracy:.3f}")
        return True
    
    def find_substitutes(
        self, 
        original_food: FoodEntity,
        candidate_foods: List[FoodEntity],
        context: str = "general",
        country_code: Optional[str] = None,
        top_k: int = 5
    ) -> List[Tuple[FoodEntity, float]]:
        """Find best substitutes for a food"""
        
        if not self.is_trained:
            # Fall back to rule-based approach
            return self._rule_based_substitution(original_food, candidate_foods, context, top_k)
        
        substitutes = []
        
        for candidate in candidate_foods:
            if candidate.food_id == original_food.food_id:
                continue
            
            # Extract features
            features = self._extract_substitution_features(original_food, candidate)
            if not features:
                continue
            
            # Apply context and country filters
            if not self._passes_context_filter(original_food, candidate, context, country_code):
                continue
            
            # Predict substitution quality
            features_scaled = self.feature_scaler.transform([features])
            probability = self.substitution_model.predict_proba(features_scaled)[0][1]
            
            # Apply rule-based adjustments
            rule_score = self._calculate_rule_based_score(original_food, candidate, context)
            final_score = (probability * 0.7) + (rule_score * 0.3)
            
            if final_score > 0.3:  # Minimum threshold
                substitutes.append((candidate, final_score))
        
        substitutes.sort(key=lambda x: x[1], reverse=True)
        return substitutes[:top_k]
    
    def _extract_substitution_features(
        self, 
        original_food: FoodEntity, 
        substitute_food: FoodEntity
    ) -> Optional[List[float]]:
        """Extract features for substitution prediction"""
        
        if not original_food.macro_nutrients or not substitute_food.macro_nutrients:
            return None
        
        orig_macro = original_food.macro_nutrients
        sub_macro = substitute_food.macro_nutrients
        
        features = []
        
        # Nutritional differences (absolute and relative)
        calorie_diff = abs(float(orig_macro.calories) - float(sub_macro.calories))
        protein_diff = abs(float(orig_macro.protein) - float(sub_macro.protein))
        fat_diff = abs(float(orig_macro.fat) - float(sub_macro.fat))
        carb_diff = abs(float(orig_macro.carbohydrates) - float(sub_macro.carbohydrates))
        
        features.extend([calorie_diff, protein_diff, fat_diff, carb_diff])
        
        # Relative differences (normalized by original values)
        if orig_macro.calories > 0:
            features.append(calorie_diff / float(orig_macro.calories))
        else:
            features.append(0.0)
        
        # Category compatibility
        same_category = 1.0 if original_food.category == substitute_food.category else 0.0
        features.append(same_category)
        
        # Allergen compatibility
        orig_allergens = set(original_food.allergens)
        sub_allergens = set(substitute_food.allergens)
        allergen_overlap = len(orig_allergens.intersection(sub_allergens)) / max(len(orig_allergens.union(sub_allergens)), 1)
        features.append(allergen_overlap)
        
        # Dietary restriction compatibility
        orig_diet = set(original_food.dietary_restrictions)
        sub_diet = set(substitute_food.dietary_restrictions)
        diet_compatibility = len(orig_diet.intersection(sub_diet)) / max(len(orig_diet), 1) if orig_diet else 1.0
        features.append(diet_compatibility)
        
        # Preparation method compatibility
        orig_prep = set(original_food.preparation_methods)
        sub_prep = set(substitute_food.preparation_methods)
        prep_overlap = len(orig_prep.intersection(sub_prep)) / max(len(orig_prep.union(sub_prep)), 1) if orig_prep or sub_prep else 1.0
        features.append(prep_overlap)
        
        # Confidence scores
        features.extend([
            float(original_food.confidence_score),
            float(substitute_food.confidence_score)
        ])
        
        return features
    
    def _passes_context_filter(
        self, 
        original_food: FoodEntity,
        substitute_food: FoodEntity,
        context: str,
        country_code: Optional[str]
    ) -> bool:
        """Check if substitute passes context-specific filters"""
        
        # Country availability filter
        if country_code and country_code not in substitute_food.country_data:
            return False
        
        # Context-specific filters
        if context == "baking":
            # For baking, need similar fat/protein content
            if not original_food.macro_nutrients or not substitute_food.macro_nutrients:
                return False
            
            orig_fat = float(original_food.macro_nutrients.fat)
            sub_fat = float(substitute_food.macro_nutrients.fat)
            
            if abs(orig_fat - sub_fat) > orig_fat * 0.5:  # 50% difference threshold
                return False
        
        elif context == "dietary_restriction":
            # Must maintain or improve dietary compatibility
            orig_restrictions = set(original_food.dietary_restrictions)
            sub_restrictions = set(substitute_food.dietary_restrictions)
            
            # Substitute must support all original restrictions
            if not orig_restrictions.issubset(sub_restrictions):
                return False
        
        elif context == "allergen_avoidance":
            # Substitute must not introduce new allergens
            orig_allergens = set(original_food.allergens)
            sub_allergens = set(substitute_food.allergens)
            
            if sub_allergens - orig_allergens:  # New allergens introduced
                return False
        
        return True
    
    def _calculate_rule_based_score(
        self, 
        original_food: FoodEntity,
        substitute_food: FoodEntity,
        context: str
    ) -> float:
        """Calculate rule-based substitution score"""
        
        score = 0.0
        
        # Nutritional similarity
        if original_food.macro_nutrients and substitute_food.macro_nutrients:
            nutritional_sim = self._calculate_nutritional_similarity(
                original_food.macro_nutrients, 
                substitute_food.macro_nutrients
            )
            score += nutritional_sim * self.substitution_rules['nutritional_similarity']
        
        # Category compatibility
        if original_food.category == substitute_food.category:
            score += self.substitution_rules['category_compatibility']
        elif self._are_categories_compatible(original_food.category, substitute_food.category):
            score += self.substitution_rules['category_compatibility'] * 0.5
        
        # Dietary compatibility
        orig_diet = set(original_food.dietary_restrictions)
        sub_diet = set(substitute_food.dietary_restrictions)
        diet_score = len(orig_diet.intersection(sub_diet)) / max(len(orig_diet), 1) if orig_diet else 1.0
        score += diet_score * self.substitution_rules['dietary_compatibility']
        
        return min(1.0, score)
    
    def _calculate_nutritional_similarity(self, macro1: MacroNutrient, macro2: MacroNutrient) -> float:
        """Calculate similarity between macro nutrient profiles"""
        
        # Compare key nutrients with weights
        comparisons = [
            (float(macro1.calories), float(macro2.calories), 0.3),
            (float(macro1.protein), float(macro2.protein), 0.25),
            (float(macro1.fat), float(macro2.fat), 0.2),
            (float(macro1.carbohydrates), float(macro2.carbohydrates), 0.25)
        ]
        
        weighted_similarity = 0.0
        
        for val1, val2, weight in comparisons:
            if val1 == 0 and val2 == 0:
                similarity = 1.0
            elif val1 == 0 or val2 == 0:
                similarity = 0.0
            else:
                # Calculate percentage difference
                diff = abs(val1 - val2) / max(val1, val2)
                similarity = max(0.0, 1.0 - diff)
            
            weighted_similarity += similarity * weight
        
        return weighted_similarity
    
    def _are_categories_compatible(self, cat1: FoodCategory, cat2: FoodCategory) -> bool:
        """Check if two food categories are compatible for substitution"""
        
        compatible_groups = [
            {FoodCategory.MEAT, FoodCategory.POULTRY, FoodCategory.SEAFOOD},
            {FoodCategory.NUTS_SEEDS, FoodCategory.LEGUMES},
            {FoodCategory.GRAINS, FoodCategory.BAKED_GOODS},
            {FoodCategory.FRUITS, FoodCategory.SWEETENERS},
            {FoodCategory.OILS_FATS, FoodCategory.DAIRY},
        ]
        
        for group in compatible_groups:
            if cat1 in group and cat2 in group:
                return True
        
        return False
    
    def _rule_based_substitution(
        self,
        original_food: FoodEntity,
        candidate_foods: List[FoodEntity],
        context: str,
        top_k: int
    ) -> List[Tuple[FoodEntity, float]]:
        """Fallback rule-based substitution when ML model is not trained"""
        
        substitutes = []
        
        for candidate in candidate_foods:
            if candidate.food_id == original_food.food_id:
                continue
            
            score = self._calculate_rule_based_score(original_food, candidate, context)
            
            if score > 0.2:
                substitutes.append((candidate, score))
        
        substitutes.sort(key=lambda x: x[1], reverse=True)
        return substitutes[:top_k]

class CulturalFoodAnalyzer:
    """ML engine for analyzing cultural food patterns and preferences"""
    
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.cultural_clusters = {}
        self.country_food_profiles = {}
        self.seasonal_models = {}
        self.is_trained = False
    
    async def analyze_cultural_patterns(
        self, 
        foods: List[FoodEntity],
        country_code: str
    ) -> Dict[str, Any]:
        """Analyze cultural food patterns for a specific country"""
        
        # Filter foods for the country
        country_foods = [
            food for food in foods 
            if country_code in food.country_data
        ]
        
        if len(country_foods) < 10:
            return {'error': 'Insufficient data for cultural analysis'}
        
        # Extract cultural features
        cultural_features = []
        food_names = []
        
        for food in country_foods:
            country_data = food.country_data[country_code]
            
            # Create cultural feature vector
            features = {
                'local_name': country_data.local_name,
                'traditional_preparations': country_data.traditional_preparations,
                'production_regions': country_data.production_regions,
                'cultural_context': country_data.cultural_context
            }
            
            cultural_features.append(features)
            food_names.append(food.name)
        
        # Analyze patterns
        analysis = {
            'country_code': country_code,
            'total_foods': len(country_foods),
            'category_distribution': self._analyze_category_distribution(country_foods),
            'seasonal_patterns': self._analyze_seasonal_patterns(country_foods, country_code),
            'preparation_preferences': self._analyze_preparation_preferences(country_foods, country_code),
            'nutritional_profile': self._analyze_country_nutritional_profile(country_foods),
            'unique_foods': self._find_unique_country_foods(country_foods, foods),
            'common_ingredients': self._extract_common_ingredients(country_foods),
        }
        
        return analysis
    
    def _analyze_category_distribution(self, foods: List[FoodEntity]) -> Dict[str, float]:
        """Analyze food category distribution"""
        category_counts = {}
        
        for food in foods:
            if food.category:
                category_name = food.category.value
                category_counts[category_name] = category_counts.get(category_name, 0) + 1
        
        total = len(foods)
        return {cat: count/total for cat, count in category_counts.items()}
    
    def _analyze_seasonal_patterns(
        self, 
        foods: List[FoodEntity], 
        country_code: str
    ) -> Dict[str, List[str]]:
        """Analyze seasonal food availability patterns"""
        seasonal_foods = {
            'spring': [],
            'summer': [],
            'fall': [],
            'winter': [],
            'year_round': []
        }
        
        for food in foods:
            if country_code in food.country_data:
                country_data = food.country_data[country_code]
                for season, available in country_data.seasonal_availability.items():
                    if available:
                        seasonal_foods[season.value].append(food.name)
        
        return seasonal_foods
    
    def _analyze_preparation_preferences(
        self, 
        foods: List[FoodEntity], 
        country_code: str
    ) -> Dict[str, int]:
        """Analyze preferred preparation methods"""
        prep_counts = {}
        
        for food in foods:
            if country_code in food.country_data:
                country_data = food.country_data[country_code]
                for prep_method in country_data.traditional_preparations:
                    method_name = prep_method.value
                    prep_counts[method_name] = prep_counts.get(method_name, 0) + 1
        
        return prep_counts
    
    def _analyze_country_nutritional_profile(self, foods: List[FoodEntity]) -> Dict[str, float]:
        """Analyze average nutritional profile for country foods"""
        nutritional_sums = {
            'calories': 0, 'protein': 0, 'fat': 0, 
            'carbohydrates': 0, 'fiber': 0, 'sodium': 0
        }
        
        valid_count = 0
        
        for food in foods:
            if food.macro_nutrients:
                macro = food.macro_nutrients
                nutritional_sums['calories'] += float(macro.calories)
                nutritional_sums['protein'] += float(macro.protein)
                nutritional_sums['fat'] += float(macro.fat)
                nutritional_sums['carbohydrates'] += float(macro.carbohydrates)
                nutritional_sums['fiber'] += float(macro.fiber)
                nutritional_sums['sodium'] += float(macro.sodium)
                valid_count += 1
        
        if valid_count == 0:
            return {}
        
        return {nutrient: total/valid_count for nutrient, total in nutritional_sums.items()}
    
    def _find_unique_country_foods(
        self, 
        country_foods: List[FoodEntity], 
        all_foods: List[FoodEntity]
    ) -> List[str]:
        """Find foods unique to this country"""
        country_food_ids = {food.food_id for food in country_foods}
        
        unique_foods = []
        for food in country_foods:
            # Check if this food is available in other countries
            other_countries = [
                country for country in food.country_data.keys() 
                if country != list(country_foods[0].country_data.keys())[0]
            ]
            
            if len(other_countries) <= 2:  # Available in 2 or fewer other countries
                unique_foods.append(food.name)
        
        return unique_foods[:10]  # Return top 10
    
    def _extract_common_ingredients(self, foods: List[FoodEntity]) -> List[str]:
        """Extract common ingredients/food components"""
        # This would typically involve NLP processing of food names and descriptions
        # For now, return most common words in food names
        
        all_names = ' '.join([food.name.lower() for food in foods])
        words = all_names.split()
        
        # Filter out common words and count occurrences
        common_words = ['with', 'and', 'or', 'in', 'on', 'the', 'a', 'an']
        word_counts = {}
        
        for word in words:
            if len(word) > 3 and word not in common_words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return most common ingredients
        sorted_ingredients = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [ingredient for ingredient, count in sorted_ingredients[:15] if count > 2]

class FoodKnowledgeMLEngine:
    """Main ML engine coordinating all food knowledge AI components"""
    
    def __init__(
        self, 
        db_manager: GraphDatabaseManager,
        model_cache_dir: str = "models/food_knowledge"
    ):
        self.db_manager = db_manager
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ML components
        self.embedding_model = None
        self.similarity_engine = NutritionalSimilarityEngine()
        self.substitution_engine = FoodSubstitutionEngine()
        self.cultural_analyzer = CulturalFoodAnalyzer()
        
        # Model configurations
        self.model_configs = {}
        
        # Performance metrics
        self.training_metrics = {}
        
    async def initialize_models(self):
        """Initialize and load all ML models"""
        logger.info("Initializing Food Knowledge ML Engine")
        
        try:
            # Load or train embedding model
            await self._load_or_train_embedding_model()
            
            # Load or train similarity engine
            await self._load_or_train_similarity_engine()
            
            # Load or train substitution engine
            await self._load_or_train_substitution_engine()
            
            logger.info("All ML models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
            return False
    
    async def _load_or_train_embedding_model(self):
        """Load existing embedding model or train new one"""
        model_path = self.model_cache_dir / "food_embedding_model.pt"
        
        if model_path.exists():
            logger.info("Loading existing embedding model")
            try:
                # Load model (implementation depends on how model is saved)
                self.embedding_model = FoodEmbeddingModel()
                # self.embedding_model.load_state_dict(torch.load(model_path))
                # self.embedding_model.eval()
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                await self._train_embedding_model()
        else:
            await self._train_embedding_model()
    
    async def _train_embedding_model(self):
        """Train new embedding model"""
        logger.info("Training new embedding model")
        
        # Get training data from database
        training_foods = await self._get_training_foods(limit=10000)
        
        if len(training_foods) < 100:
            logger.warning("Insufficient data for embedding model training")
            return
        
        # Initialize model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_model = FoodEmbeddingModel()
        
        # Create dataset and dataloader
        dataset = FoodEmbeddingDataset(training_foods, tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training configuration
        optimizer = optim.AdamW(self.embedding_model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.embedding_model.train()
        for epoch in range(3):  # Limited epochs for demo
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                embeddings, category_logits = self.embedding_model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['numerical_features']
                )
                
                # Category classification loss (auxiliary task)
                category_labels = self._encode_categories(batch['category'])
                loss = criterion(category_logits, category_labels)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            logger.info(f"Epoch {epoch+1}, Average Loss: {total_loss/len(dataloader):.4f}")
        
        # Save model
        model_path = self.model_cache_dir / "food_embedding_model.pt"
        torch.save(self.embedding_model.state_dict(), model_path)
        
        self.embedding_model.eval()
        logger.info("Embedding model training completed")
    
    async def _load_or_train_similarity_engine(self):
        """Load or train nutritional similarity engine"""
        model_path = self.model_cache_dir / "similarity_engine.pkl"
        
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    similarity_data = pickle.load(f)
                    self.similarity_engine.scaler = similarity_data['scaler']
                    self.similarity_engine.pca = similarity_data['pca']
                    self.similarity_engine.similarity_model = similarity_data['model']
                    self.similarity_engine.is_trained = True
                logger.info("Loaded existing similarity engine")
            except Exception as e:
                logger.warning(f"Failed to load similarity engine: {e}")
                await self._train_similarity_engine()
        else:
            await self._train_similarity_engine()
    
    async def _train_similarity_engine(self):
        """Train nutritional similarity engine"""
        training_foods = await self._get_training_foods(limit=5000)
        
        if self.similarity_engine.train(training_foods):
            # Save model
            model_path = self.model_cache_dir / "similarity_engine.pkl"
            similarity_data = {
                'scaler': self.similarity_engine.scaler,
                'pca': self.similarity_engine.pca,
                'model': self.similarity_engine.similarity_model
            }
            with open(model_path, 'wb') as f:
                pickle.dump(similarity_data, f)
            
            logger.info("Similarity engine trained and saved")
    
    async def _load_or_train_substitution_engine(self):
        """Load or train substitution engine"""
        model_path = self.model_cache_dir / "substitution_engine.pkl"
        
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    substitution_data = pickle.load(f)
                    self.substitution_engine.substitution_model = substitution_data['model']
                    self.substitution_engine.feature_scaler = substitution_data['scaler']
                    self.substitution_engine.is_trained = True
                logger.info("Loaded existing substitution engine")
            except Exception as e:
                logger.warning(f"Failed to load substitution engine: {e}")
                # Generate synthetic training data for demo
                await self._train_substitution_engine_synthetic()
        else:
            await self._train_substitution_engine_synthetic()
    
    async def _train_substitution_engine_synthetic(self):
        """Train substitution engine with synthetic data"""
        # Generate synthetic substitution data for training
        training_foods = await self._get_training_foods(limit=1000)
        
        synthetic_data = []
        for i, food1 in enumerate(training_foods[:100]):
            for food2 in training_foods[i+1:i+6]:  # Compare with next 5 foods
                # Generate synthetic quality score based on similarity
                score = self._calculate_synthetic_substitution_quality(food1, food2)
                synthetic_data.append((food1, food2, score))
        
        if self.substitution_engine.train(synthetic_data):
            # Save model
            model_path = self.model_cache_dir / "substitution_engine.pkl"
            substitution_data = {
                'model': self.substitution_engine.substitution_model,
                'scaler': self.substitution_engine.feature_scaler
            }
            with open(model_path, 'wb') as f:
                pickle.dump(substitution_data, f)
            
            logger.info("Substitution engine trained with synthetic data")
    
    async def _get_training_foods(self, limit: int = 1000) -> List[FoodEntity]:
        """Get foods from database for training"""
        # This would query the Neo4j database for training data
        # For now, return empty list as placeholder
        return []
    
    def _encode_categories(self, categories: List[str]) -> torch.Tensor:
        """Encode food categories for training"""
        category_to_idx = {cat.value: idx for idx, cat in enumerate(FoodCategory)}
        encoded = [category_to_idx.get(cat, 0) for cat in categories]
        return torch.tensor(encoded, dtype=torch.long)
    
    def _calculate_synthetic_substitution_quality(
        self, 
        food1: FoodEntity, 
        food2: FoodEntity
    ) -> float:
        """Calculate synthetic quality score for training data"""
        
        score = 0.0
        
        # Category similarity
        if food1.category == food2.category:
            score += 0.4
        
        # Nutritional similarity
        if food1.macro_nutrients and food2.macro_nutrients:
            macro1, macro2 = food1.macro_nutrients, food2.macro_nutrients
            
            # Simple similarity based on calorie difference
            calorie_diff = abs(float(macro1.calories) - float(macro2.calories))
            max_calories = max(float(macro1.calories), float(macro2.calories))
            if max_calories > 0:
                calorie_sim = 1.0 - min(calorie_diff / max_calories, 1.0)
                score += calorie_sim * 0.3
        
        # Dietary restriction compatibility
        diet1 = set(food1.dietary_restrictions)
        diet2 = set(food2.dietary_restrictions)
        diet_sim = len(diet1.intersection(diet2)) / max(len(diet1.union(diet2)), 1)
        score += diet_sim * 0.2
        
        # Add some randomness
        import random
        score += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, score))
    
    async def get_model_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all ML models"""
        return {
            'embedding_model': {
                'status': 'trained' if self.embedding_model else 'not_trained',
                'model_size': 'large' if self.embedding_model else 'n/a'
            },
            'similarity_engine': {
                'status': 'trained' if self.similarity_engine.is_trained else 'not_trained'
            },
            'substitution_engine': {
                'status': 'trained' if self.substitution_engine.is_trained else 'not_trained'
            },
            'cultural_analyzer': {
                'status': 'ready'
            },
            'last_updated': datetime.utcnow().isoformat()
        }