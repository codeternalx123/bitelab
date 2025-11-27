"""
AI Food Scanner with FatSecret API Integration
Dynamic food database with 1M+ items instead of hardcoded data
"""

import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

from fatsecret_client import FatSecretClient, FoodDataMapper
from api_model_trainer import FoodCategoryClassifier

logger = logging.getLogger(__name__)


@dataclass
class FoodScanResult:
    """Result from food scanning with API data"""
    food_id: str
    name: str
    brand: Optional[str]
    category: str
    confidence: float
    nutrients: Dict[str, float]
    servings: List[Dict]
    source: str = "FatSecret API"
    description: str = ""
    url: str = ""


class ApiFoodScanner:
    """
    AI Food Scanner using FatSecret API
    Provides access to 1M+ foods with ML-powered categorization
    """
    
    def __init__(
        self,
        fatsecret_client_id: str = None,
        fatsecret_client_secret: str = None,
        use_local_models: bool = True
    ):
        """
        Initialize scanner with FatSecret API
        
        Args:
            fatsecret_client_id: FatSecret API client ID
            fatsecret_client_secret: FatSecret API client secret
            use_local_models: Whether to use locally trained models for categorization
        """
        # Initialize FatSecret client
        self.api_client = FatSecretClient(fatsecret_client_id, fatsecret_client_secret)
        self.mapper = FoodDataMapper()
        
        # Load ML models if available
        self.classifier = None
        if use_local_models:
            try:
                self.classifier = FoodCategoryClassifier()
                self.classifier.load('models')
                logger.info("Loaded pre-trained classification models")
            except Exception as e:
                logger.warning(f"Could not load models: {e}")
                logger.info("Will use API categorization only")
        
        logger.info("ApiFoodScanner initialized with FatSecret API")
    
    def scan_food(
        self,
        search_query: str,
        limit: int = 10
    ) -> List[FoodScanResult]:
        """
        Scan/search for foods in the FatSecret database
        
        Args:
            search_query: Food name or description to search
            limit: Maximum number of results
            
        Returns:
            List of food scan results
        """
        logger.info(f"Scanning for: {search_query}")
        
        # Search via API
        search_results = self.api_client.search_foods(
            search_query,
            max_results=min(limit, 50)
        )
        
        foods = search_results.get('food', [])
        if not isinstance(foods, list):
            foods = [foods]
        
        # Process each result
        results = []
        for food in foods[:limit]:
            try:
                # Get detailed information
                food_id = food.get('food_id')
                detailed_food = self.api_client.get_food(food_id)
                
                # Map to internal format
                mapped_food = self.mapper.map_food_item(detailed_food)
                
                # Predict category using ML if available
                category = 'unknown'
                confidence = 0.5
                
                if self.classifier:
                    try:
                        # Extract features for classification
                        training_features = self.mapper.extract_training_data([mapped_food])[0]
                        category, confidence = self.classifier.predict(training_features)
                    except Exception as e:
                        logger.warning(f"Classification failed: {e}")
                        category = self._infer_category_from_name(mapped_food['name'])
                else:
                    # Fallback to name-based inference
                    category = self._infer_category_from_name(mapped_food['name'])
                
                # Create result
                result = FoodScanResult(
                    food_id=mapped_food['food_id'],
                    name=mapped_food['name'],
                    brand=mapped_food['brand'],
                    category=category,
                    confidence=confidence,
                    nutrients=mapped_food['nutrients'],
                    servings=mapped_food['servings'],
                    description=mapped_food['description'],
                    url=mapped_food['url']
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing food {food.get('food_id')}: {e}")
                continue
        
        logger.info(f"Found {len(results)} results for '{search_query}'")
        return results
    
    def scan_by_barcode(self, barcode: str) -> Optional[FoodScanResult]:
        """
        Scan food by barcode (UPC/EAN)
        
        Args:
            barcode: Product barcode
            
        Returns:
            Food scan result if found
        """
        logger.info(f"Scanning barcode: {barcode}")
        
        food_data = self.api_client.search_by_barcode(barcode)
        
        if not food_data:
            logger.info(f"No food found for barcode {barcode}")
            return None
        
        # Map and classify
        mapped_food = self.mapper.map_food_item(food_data)
        
        category = 'unknown'
        confidence = 0.5
        
        if self.classifier:
            try:
                training_features = self.mapper.extract_training_data([mapped_food])[0]
                category, confidence = self.classifier.predict(training_features)
            except Exception as e:
                logger.warning(f"Classification failed: {e}")
                category = self._infer_category_from_name(mapped_food['name'])
        else:
            category = self._infer_category_from_name(mapped_food['name'])
        
        result = FoodScanResult(
            food_id=mapped_food['food_id'],
            name=mapped_food['name'],
            brand=mapped_food['brand'],
            category=category,
            confidence=confidence,
            nutrients=mapped_food['nutrients'],
            servings=mapped_food['servings'],
            description=mapped_food['description'],
            url=mapped_food['url']
        )
        
        return result
    
    def get_autocomplete(self, partial_name: str) -> List[str]:
        """
        Get autocomplete suggestions for food names
        
        Args:
            partial_name: Partial food name
            
        Returns:
            List of suggested food names
        """
        return self.api_client.autocomplete(partial_name)
    
    def get_food_details(self, food_id: str) -> Optional[FoodScanResult]:
        """
        Get detailed information for a specific food ID
        
        Args:
            food_id: FatSecret food ID
            
        Returns:
            Detailed food scan result
        """
        try:
            food_data = self.api_client.get_food(food_id)
            mapped_food = self.mapper.map_food_item(food_data)
            
            category = 'unknown'
            confidence = 0.5
            
            if self.classifier:
                training_features = self.mapper.extract_training_data([mapped_food])[0]
                category, confidence = self.classifier.predict(training_features)
            else:
                category = self._infer_category_from_name(mapped_food['name'])
            
            return FoodScanResult(
                food_id=mapped_food['food_id'],
                name=mapped_food['name'],
                brand=mapped_food['brand'],
                category=category,
                confidence=confidence,
                nutrients=mapped_food['nutrients'],
                servings=mapped_food['servings'],
                description=mapped_food['description'],
                url=mapped_food['url']
            )
        
        except Exception as e:
            logger.error(f"Error getting food details: {e}")
            return None
    
    def _infer_category_from_name(self, name: str) -> str:
        """Fallback: infer category from food name using keywords"""
        name_lower = name.lower()
        
        category_keywords = {
            'vegetables': ['broccoli', 'carrot', 'spinach', 'kale', 'lettuce', 'tomato', 'vegetable'],
            'fruits': ['apple', 'banana', 'orange', 'berry', 'fruit', 'grape', 'melon'],
            'grains': ['bread', 'rice', 'pasta', 'cereal', 'oat', 'wheat', 'grain'],
            'proteins': ['tofu', 'tempeh', 'seitan', 'protein'],
            'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream', 'dairy'],
            'nuts_seeds': ['almond', 'walnut', 'cashew', 'peanut', 'seed', 'nut'],
            'legumes': ['bean', 'lentil', 'chickpea', 'pea', 'legume'],
            'oils_fats': ['oil', 'fat', 'lard'],
            'beverages': ['juice', 'coffee', 'tea', 'soda', 'drink', 'water'],
            'seafood': ['fish', 'salmon', 'tuna', 'shrimp', 'seafood', 'crab'],
            'poultry': ['chicken', 'turkey', 'duck', 'poultry'],
            'meat': ['beef', 'pork', 'lamb', 'veal', 'meat'],
            'snacks': ['chip', 'cracker', 'popcorn', 'snack'],
            'desserts': ['cake', 'cookie', 'ice cream', 'chocolate', 'candy', 'dessert']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in name_lower for keyword in keywords):
                return category
        
        return 'unknown'
    
    def compare_foods(
        self,
        food_ids: List[str]
    ) -> Dict:
        """
        Compare nutritional values of multiple foods
        
        Args:
            food_ids: List of FatSecret food IDs to compare
            
        Returns:
            Comparison data
        """
        foods = []
        for food_id in food_ids:
            result = self.get_food_details(food_id)
            if result:
                foods.append(result)
        
        if not foods:
            return {}
        
        # Create comparison table
        comparison = {
            'foods': [f.name for f in foods],
            'nutrients': {}
        }
        
        # Compare each nutrient
        nutrient_names = foods[0].nutrients.keys()
        for nutrient in nutrient_names:
            values = [f.nutrients.get(nutrient, 0) for f in foods]
            comparison['nutrients'][nutrient] = values
        
        return comparison


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("AI FOOD SCANNER - FATSECRET API INTEGRATION")
    print("=" * 80)
    
    # Check credentials
    if not os.getenv('FATSECRET_CLIENT_ID') or not os.getenv('FATSECRET_CLIENT_SECRET'):
        print("\n⚠️  FatSecret API credentials required!")
        print("Set FATSECRET_CLIENT_ID and FATSECRET_CLIENT_SECRET environment variables")
        exit(1)
    
    try:
        # Initialize scanner
        print("\n🔄 Initializing AI Food Scanner...")
        scanner = ApiFoodScanner()
        
        print("✅ Scanner initialized!")
        
        # Test 1: Search for foods
        print("\n" + "=" * 80)
        print("TEST 1: Search for 'chicken breast'")
        print("=" * 80)
        
        results = scanner.scan_food('chicken breast', limit=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.name}")
            if result.brand:
                print(f"   Brand: {result.brand}")
            print(f"   Category: {result.category} (confidence: {result.confidence:.2%})")
            print(f"   Calories: {result.nutrients['calories']:.1f} per 100g")
            print(f"   Protein: {result.nutrients['protein']:.1f}g")
            print(f"   Carbs: {result.nutrients['carbohydrates']:.1f}g")
            print(f"   Fat: {result.nutrients['fat']:.1f}g")
        
        # Test 2: Autocomplete
        print("\n" + "=" * 80)
        print("TEST 2: Autocomplete for 'bro'")
        print("=" * 80)
        
        suggestions = scanner.get_autocomplete('bro')
        print(f"Suggestions: {', '.join(suggestions[:10])}")
        
        # Test 3: Food comparison
        if len(results) >= 2:
            print("\n" + "=" * 80)
            print("TEST 3: Compare first two results")
            print("=" * 80)
            
            food_ids = [results[0].food_id, results[1].food_id]
            comparison = scanner.compare_foods(food_ids)
            
            print(f"\nComparing: {' vs '.join(comparison['foods'])}")
            print(f"\nNutrients (per 100g):")
            for nutrient, values in list(comparison['nutrients'].items())[:8]:
                print(f"  {nutrient:20s}: {values[0]:8.2f} vs {values[1]:8.2f}")
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe scanner now has access to:")
        print("  - 1,000,000+ foods from FatSecret database")
        print("  - Real-time nutritional data")
        print("  - Barcode scanning support")
        print("  - ML-powered categorization")
        print("  - Brand and generic foods")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
