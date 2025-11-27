"""
ML Model Trainer using FatSecret API Data
Replaces hardcoded dataset with real-time API data
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import logging
from datetime import datetime
from pathlib import Path

from fatsecret_client import FatSecretClient, FoodDataMapper

logger = logging.getLogger(__name__)


class FoodCategoryClassifier:
    """
    Train ML model to classify foods into categories
    Uses FatSecret API data for training
    """
    
    CATEGORIES = [
        'vegetables', 'fruits', 'grains', 'proteins', 'dairy',
        'nuts_seeds', 'legumes', 'oils_fats', 'beverages',
        'seafood', 'poultry', 'meat', 'snacks', 'desserts'
    ]
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False
    
    def fetch_training_data(
        self,
        client: FatSecretClient,
        samples_per_category: int = 100
    ) -> pd.DataFrame:
        """
        Fetch training data from FatSecret API
        
        Args:
            client: FatSecret API client
            samples_per_category: Number of samples to fetch per category
            
        Returns:
            DataFrame with training data
        """
        logger.info("Fetching training data from FatSecret API...")
        
        # Search terms for each category
        category_searches = {
            'vegetables': ['broccoli', 'carrot', 'spinach', 'kale', 'tomato', 'lettuce'],
            'fruits': ['apple', 'banana', 'orange', 'strawberry', 'mango', 'grape'],
            'grains': ['rice', 'bread', 'pasta', 'oats', 'quinoa', 'wheat'],
            'proteins': ['beef', 'pork', 'lamb', 'tofu', 'tempeh'],
            'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream'],
            'nuts_seeds': ['almonds', 'walnuts', 'cashews', 'chia', 'flax'],
            'legumes': ['lentils', 'chickpeas', 'beans', 'peas'],
            'oils_fats': ['olive oil', 'coconut oil', 'avocado', 'butter'],
            'beverages': ['coffee', 'tea', 'juice', 'water', 'soda'],
            'seafood': ['salmon', 'tuna', 'shrimp', 'cod', 'tilapia'],
            'poultry': ['chicken', 'turkey', 'duck'],
            'meat': ['beef', 'pork', 'lamb', 'veal'],
            'snacks': ['chips', 'crackers', 'popcorn', 'pretzels'],
            'desserts': ['cake', 'cookies', 'ice cream', 'chocolate']
        }
        
        all_training_data = []
        mapper = FoodDataMapper()
        
        for category, search_terms in category_searches.items():
            logger.info(f"Fetching {category} samples...")
            category_samples = []
            
            for search_term in search_terms:
                try:
                    # Search for foods
                    results = client.search_foods(
                        search_term,
                        max_results=min(50, samples_per_category // len(search_terms))
                    )
                    
                    foods = results.get('food', [])
                    if not isinstance(foods, list):
                        foods = [foods]
                    
                    # Get detailed info and map
                    for food in foods[:samples_per_category // len(search_terms)]:
                        try:
                            detailed = client.get_food(food.get('food_id'))
                            mapped = mapper.map_food_item(detailed)
                            
                            # Add category label
                            mapped['category'] = category
                            category_samples.append(mapped)
                            
                        except Exception as e:
                            logger.warning(f"Error processing food {food.get('food_id')}: {e}")
                            continue
                    
                except Exception as e:
                    logger.error(f"Error searching for '{search_term}': {e}")
                    continue
            
            logger.info(f"Fetched {len(category_samples)} samples for {category}")
            all_training_data.extend(category_samples)
        
        # Convert to training features
        training_features = mapper.extract_training_data(all_training_data)
        
        # Add categories back
        for i, item in enumerate(all_training_data):
            training_features[i]['category'] = item['category']
        
        df = pd.DataFrame(training_features)
        logger.info(f"Total training samples: {len(df)}")
        
        return df
    
    def train(
        self,
        training_data: pd.DataFrame,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train the classification model
        
        Args:
            training_data: DataFrame with features and category labels
            test_size: Fraction of data to use for testing
            
        Returns:
            Training metrics
        """
        logger.info("Training food category classifier...")
        
        # Separate features and labels
        feature_columns = [
            'calories', 'protein', 'carbs', 'fiber', 'sugar', 'fat',
            'saturated_fat', 'sodium', 'protein_ratio', 'carb_ratio',
            'fat_ratio', 'fiber_density', 'protein_density'
        ]
        
        X = training_data[feature_columns].fillna(0)
        y = training_data['category']
        
        self.feature_names = feature_columns
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5
        )
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': report,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': dict(zip(
                feature_columns,
                self.model.feature_importances_
            ))
        }
        
        logger.info(f"Training complete! Test accuracy: {test_score:.3f}")
        
        return metrics
    
    def predict(self, food_features: Dict) -> Tuple[str, float]:
        """
        Predict food category
        
        Args:
            food_features: Dictionary with food features
            
        Returns:
            Tuple of (predicted_category, confidence)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet!")
        
        # Extract features in correct order
        features = np.array([[
            food_features.get(name, 0) for name in self.feature_names
        ]])
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        category = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]
        
        return category, confidence
    
    def save(self, model_dir: str = "models"):
        """Save trained model"""
        Path(model_dir).mkdir(exist_ok=True)
        
        model_path = Path(model_dir) / "food_classifier.pkl"
        scaler_path = Path(model_dir) / "scaler.pkl"
        encoder_path = Path(model_dir) / "label_encoder.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'categories': self.label_encoder.classes_.tolist(),
            'trained_at': datetime.now().isoformat()
        }
        
        with open(Path(model_dir) / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_dir}/")
    
    def load(self, model_dir: str = "models"):
        """Load trained model"""
        with open(Path(model_dir) / "food_classifier.pkl", 'rb') as f:
            self.model = pickle.load(f)
        with open(Path(model_dir) / "scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        with open(Path(model_dir) / "label_encoder.pkl", 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        with open(Path(model_dir) / "model_metadata.json", 'r') as f:
            metadata = json.load(f)
            self.feature_names = metadata['feature_names']
        
        self.is_trained = True
        logger.info(f"Model loaded from {model_dir}/")


class NutrientPredictor:
    """
    Predict missing nutrients based on available data
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.target_nutrients = [
            'vitamin_a', 'vitamin_c', 'calcium', 'iron',
            'vitamin_d', 'vitamin_e', 'vitamin_k', 'vitamin_b12'
        ]
    
    def train(self, training_data: pd.DataFrame) -> Dict:
        """Train models to predict micronutrients"""
        logger.info("Training nutrient predictor...")
        
        # Base features (macronutrients)
        base_features = [
            'calories', 'protein', 'carbs', 'fiber', 'sugar', 'fat'
        ]
        
        self.feature_names = base_features
        metrics = {}
        
        for nutrient in self.target_nutrients:
            logger.info(f"Training model for {nutrient}...")
            
            # Filter data where nutrient is available
            valid_data = training_data[training_data[nutrient].notna()].copy()
            
            if len(valid_data) < 50:
                logger.warning(f"Insufficient data for {nutrient}, skipping...")
                continue
            
            X = valid_data[base_features].fillna(0)
            y = valid_data[nutrient]
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            self.models[nutrient] = model
            self.scalers[nutrient] = scaler
            
            metrics[nutrient] = {
                'rmse': rmse,
                'r2': r2,
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            logger.info(f"{nutrient}: RMSE={rmse:.2f}, R²={r2:.3f}")
        
        return metrics
    
    def predict(self, food_features: Dict) -> Dict:
        """Predict missing nutrients"""
        features = np.array([[
            food_features.get(name, 0) for name in self.feature_names
        ]])
        
        predictions = {}
        for nutrient, model in self.models.items():
            scaled_features = self.scalers[nutrient].transform(features)
            predictions[nutrient] = max(0, model.predict(scaled_features)[0])
        
        return predictions


# Main training pipeline
def train_models_from_api():
    """Complete training pipeline using FatSecret API"""
    
    print("=" * 80)
    print("AI NUTRITION MODEL TRAINING - FATSECRET API")
    print("=" * 80)
    
    # Check credentials
    if not os.getenv('FATSECRET_CLIENT_ID') or not os.getenv('FATSECRET_CLIENT_SECRET'):
        print("\n❌ FatSecret API credentials not found!")
        print("Set FATSECRET_CLIENT_ID and FATSECRET_CLIENT_SECRET environment variables")
        return
    
    try:
        # Initialize API client
        print("\n📡 Connecting to FatSecret API...")
        client = FatSecretClient()
        
        # Initialize classifier
        print("\n🤖 Initializing ML models...")
        classifier = FoodCategoryClassifier()
        
        # Fetch training data
        print("\n📥 Fetching training data from API...")
        print("This may take several minutes...")
        training_data = classifier.fetch_training_data(client, samples_per_category=50)
        
        print(f"\n✅ Fetched {len(training_data)} food samples")
        print(f"Categories: {training_data['category'].value_counts().to_dict()}")
        
        # Save raw training data
        print("\n💾 Saving training data...")
        training_data.to_csv('training_data.csv', index=False)
        print("Training data saved to training_data.csv")
        
        # Train classifier
        print("\n🎯 Training food category classifier...")
        classifier_metrics = classifier.train(training_data)
        
        print(f"\n✅ Classifier trained!")
        print(f"Training accuracy: {classifier_metrics['train_accuracy']:.3f}")
        print(f"Test accuracy: {classifier_metrics['test_accuracy']:.3f}")
        print(f"Cross-validation: {classifier_metrics['cv_mean']:.3f} ± {classifier_metrics['cv_std']:.3f}")
        
        # Save classifier
        print("\n💾 Saving classifier model...")
        classifier.save('models')
        
        # Train nutrient predictor
        print("\n🎯 Training nutrient predictor...")
        predictor = NutrientPredictor()
        nutrient_metrics = predictor.train(training_data)
        
        print("\n✅ Nutrient predictor trained!")
        for nutrient, metrics in nutrient_metrics.items():
            print(f"  {nutrient}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.2f}")
        
        # Save all metrics
        print("\n💾 Saving training metrics...")
        all_metrics = {
            'classifier': classifier_metrics,
            'nutrient_predictor': nutrient_metrics,
            'training_date': datetime.now().isoformat(),
            'data_source': 'FatSecret API',
            'total_samples': len(training_data)
        }
        
        with open('models/training_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print("\nModels saved to models/ directory:")
        print("  - food_classifier.pkl (category classification)")
        print("  - scaler.pkl (feature scaling)")
        print("  - label_encoder.pkl (category encoding)")
        print("  - model_metadata.json (model info)")
        print("  - training_metrics.json (performance metrics)")
        print("\nYou can now use these models for food scanning!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_models_from_api()
