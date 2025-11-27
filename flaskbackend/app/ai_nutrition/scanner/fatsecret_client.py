"""
FatSecret API Client for Food Database
Provides access to 1M+ foods with comprehensive nutritional data
"""

import os
import time
import hmac
import hashlib
import base64
import urllib.parse
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FatSecretConfig:
    """FatSecret API configuration"""
    client_id: str
    client_secret: str
    api_url: str = "https://platform.fatsecret.com/rest/server.api"
    oauth_url: str = "https://oauth.fatsecret.com/connect/token"


class FatSecretClient:
    """
    Client for FatSecret Platform API
    Provides access to comprehensive food database with 1M+ items
    """
    
    def __init__(self, client_id: str = None, client_secret: str = None):
        """
        Initialize FatSecret API client
        
        Args:
            client_id: FatSecret API client ID (or set FATSECRET_CLIENT_ID env var)
            client_secret: FatSecret API client secret (or set FATSECRET_CLIENT_SECRET env var)
        """
        self.client_id = client_id or os.getenv('FATSECRET_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('FATSECRET_CLIENT_SECRET')
        
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "FatSecret API credentials required. Set FATSECRET_CLIENT_ID and "
                "FATSECRET_CLIENT_SECRET environment variables or pass to constructor."
            )
        
        self.config = FatSecretConfig(
            client_id=self.client_id,
            client_secret=self.client_secret
        )
        self.access_token = None
        self.token_expires_at = 0
        
        logger.info("FatSecret API client initialized")
    
    def _get_access_token(self) -> str:
        """
        Get OAuth 2.0 access token for API calls
        Uses client credentials flow
        """
        # Check if current token is still valid
        if self.access_token and time.time() < self.token_expires_at:
            return self.access_token
        
        # Request new token
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_bytes = auth_string.encode('ascii')
        auth_base64 = base64.b64encode(auth_bytes).decode('ascii')
        
        headers = {
            'Authorization': f'Basic {auth_base64}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'client_credentials',
            'scope': 'basic'
        }
        
        try:
            response = requests.post(
                self.config.oauth_url,
                headers=headers,
                data=data,
                timeout=10
            )
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            # Set expiry to 5 minutes before actual expiry for safety
            expires_in = token_data.get('expires_in', 3600) - 300
            self.token_expires_at = time.time() + expires_in
            
            logger.info("Successfully obtained FatSecret API access token")
            return self.access_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get access token: {e}")
            raise
    
    def _make_request(self, method: str, params: Dict[str, Any]) -> Dict:
        """
        Make authenticated API request
        
        Args:
            method: FatSecret API method name
            params: Additional parameters for the method
            
        Returns:
            API response data
        """
        token = self._get_access_token()
        
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        request_params = {
            'method': method,
            'format': 'json',
            **params
        }
        
        try:
            response = requests.post(
                self.config.api_url,
                headers=headers,
                data=request_params,
                timeout=15
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'error' in data:
                logger.error(f"API error: {data['error']}")
                raise Exception(f"FatSecret API error: {data['error']}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def search_foods(
        self,
        search_expression: str,
        page_number: int = 0,
        max_results: int = 50
    ) -> Dict:
        """
        Search for foods in the FatSecret database
        
        Args:
            search_expression: Food name or description to search
            page_number: Page number for pagination (0-based)
            max_results: Maximum results per page (max 50)
            
        Returns:
            Search results with food items
        """
        params = {
            'search_expression': search_expression,
            'page_number': str(page_number),
            'max_results': str(min(max_results, 50))
        }
        
        result = self._make_request('foods.search', params)
        logger.info(f"Food search for '{search_expression}' returned results")
        
        return result.get('foods', {})
    
    def get_food(self, food_id: str) -> Dict:
        """
        Get detailed information for a specific food
        
        Args:
            food_id: FatSecret food ID
            
        Returns:
            Detailed food information including all servings
        """
        params = {'food_id': str(food_id)}
        
        result = self._make_request('food.get', params)
        logger.info(f"Retrieved food details for ID {food_id}")
        
        return result.get('food', {})
    
    def autocomplete(self, expression: str) -> List[str]:
        """
        Get autocomplete suggestions for food search
        
        Args:
            expression: Partial food name
            
        Returns:
            List of suggested food names
        """
        params = {'expression': expression}
        
        result = self._make_request('foods.autocomplete', params)
        
        suggestions = result.get('suggestions', {}).get('suggestion', [])
        if isinstance(suggestions, str):
            suggestions = [suggestions]
        
        return suggestions
    
    def search_by_barcode(self, barcode: str) -> Optional[Dict]:
        """
        Search for food by barcode (UPC/EAN)
        
        Args:
            barcode: Product barcode
            
        Returns:
            Food data if found, None otherwise
        """
        params = {'barcode': barcode}
        
        try:
            result = self._make_request('food.find_id_for_barcode', params)
            
            if 'food_id' in result:
                food_id = result['food_id']['value']
                return self.get_food(food_id)
            
            return None
            
        except Exception as e:
            logger.warning(f"Barcode lookup failed: {e}")
            return None
    
    def get_popular_foods(
        self,
        category: str = None,
        page_number: int = 0,
        max_results: int = 50
    ) -> Dict:
        """
        Get popular/trending foods
        
        Args:
            category: Optional category filter
            page_number: Page number for pagination
            max_results: Maximum results per page
            
        Returns:
            Popular foods data
        """
        params = {
            'page_number': str(page_number),
            'max_results': str(min(max_results, 50))
        }
        
        if category:
            params['category'] = category
        
        result = self._make_request('foods.get_most_eaten', params)
        
        return result.get('foods', {})


class FoodDataMapper:
    """
    Maps FatSecret API data to our internal food database format
    """
    
    @staticmethod
    def map_food_item(fatsecret_food: Dict) -> Dict:
        """
        Convert FatSecret food data to our internal format
        
        Args:
            fatsecret_food: Food data from FatSecret API
            
        Returns:
            Standardized food data
        """
        # Extract basic info
        food_id = fatsecret_food.get('food_id', '')
        food_name = fatsecret_food.get('food_name', '')
        food_type = fatsecret_food.get('food_type', 'generic')
        brand_name = fatsecret_food.get('brand_name', '')
        
        # Get primary serving
        servings = fatsecret_food.get('servings', {}).get('serving', [])
        if not isinstance(servings, list):
            servings = [servings]
        
        # Use first serving as primary
        primary_serving = servings[0] if servings else {}
        
        # Extract nutrients (per 100g if available, otherwise convert)
        serving_size = float(primary_serving.get('metric_serving_amount', 100))
        serving_unit = primary_serving.get('metric_serving_unit', 'g')
        
        # Calculate per 100g values
        scale_factor = 100.0 / serving_size if serving_size > 0 else 1.0
        
        nutrients = {
            'calories': float(primary_serving.get('calories', 0)) * scale_factor,
            'protein': float(primary_serving.get('protein', 0)) * scale_factor,
            'carbohydrates': float(primary_serving.get('carbohydrate', 0)) * scale_factor,
            'fiber': float(primary_serving.get('fiber', 0)) * scale_factor,
            'sugar': float(primary_serving.get('sugar', 0)) * scale_factor,
            'fat': float(primary_serving.get('fat', 0)) * scale_factor,
            'saturated_fat': float(primary_serving.get('saturated_fat', 0)) * scale_factor,
            'polyunsaturated_fat': float(primary_serving.get('polyunsaturated_fat', 0)) * scale_factor,
            'monounsaturated_fat': float(primary_serving.get('monounsaturated_fat', 0)) * scale_factor,
            'trans_fat': float(primary_serving.get('trans_fat', 0)) * scale_factor,
            'cholesterol': float(primary_serving.get('cholesterol', 0)) * scale_factor,
            'sodium': float(primary_serving.get('sodium', 0)) * scale_factor,
            'potassium': float(primary_serving.get('potassium', 0)) * scale_factor,
            'vitamin_a': float(primary_serving.get('vitamin_a', 0)) * scale_factor,
            'vitamin_c': float(primary_serving.get('vitamin_c', 0)) * scale_factor,
            'calcium': float(primary_serving.get('calcium', 0)) * scale_factor,
            'iron': float(primary_serving.get('iron', 0)) * scale_factor,
        }
        
        return {
            'food_id': food_id,
            'name': food_name,
            'brand': brand_name,
            'food_type': food_type,
            'nutrients': nutrients,
            'servings': servings,
            'description': fatsecret_food.get('food_description', ''),
            'url': fatsecret_food.get('food_url', '')
        }
    
    @staticmethod
    def extract_training_data(food_items: List[Dict]) -> List[Dict]:
        """
        Extract features for ML model training
        
        Args:
            food_items: List of mapped food items
            
        Returns:
            Training data with features
        """
        training_data = []
        
        for item in food_items:
            nutrients = item['nutrients']
            
            # Calculate derived features
            total_macros = nutrients['protein'] + nutrients['carbohydrates'] + nutrients['fat']
            
            features = {
                # Basic nutrients
                'calories': nutrients['calories'],
                'protein': nutrients['protein'],
                'carbs': nutrients['carbohydrates'],
                'fiber': nutrients['fiber'],
                'sugar': nutrients['sugar'],
                'fat': nutrients['fat'],
                'saturated_fat': nutrients['saturated_fat'],
                'sodium': nutrients['sodium'],
                
                # Ratios and derived features
                'protein_ratio': nutrients['protein'] / total_macros if total_macros > 0 else 0,
                'carb_ratio': nutrients['carbohydrates'] / total_macros if total_macros > 0 else 0,
                'fat_ratio': nutrients['fat'] / total_macros if total_macros > 0 else 0,
                'fiber_density': nutrients['fiber'] / nutrients['calories'] * 1000 if nutrients['calories'] > 0 else 0,
                'protein_density': nutrients['protein'] / nutrients['calories'] * 100 if nutrients['calories'] > 0 else 0,
                
                # Metadata
                'food_id': item['food_id'],
                'name': item['name'],
                'brand': item['brand'],
                'food_type': item['food_type']
            }
            
            training_data.append(features)
        
        return training_data


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("FATSECRET API CLIENT - TESTING")
    print("=" * 80)
    
    # Check for credentials
    if not os.getenv('FATSECRET_CLIENT_ID') or not os.getenv('FATSECRET_CLIENT_SECRET'):
        print("\n⚠️  SETUP REQUIRED:")
        print("=" * 80)
        print("1. Sign up for FatSecret Platform API at:")
        print("   https://platform.fatsecret.com/api/")
        print("\n2. Create an application to get your credentials")
        print("\n3. Set environment variables:")
        print("   export FATSECRET_CLIENT_ID='your_client_id'")
        print("   export FATSECRET_CLIENT_SECRET='your_client_secret'")
        print("\n4. On Windows use:")
        print("   set FATSECRET_CLIENT_ID=your_client_id")
        print("   set FATSECRET_CLIENT_SECRET=your_client_secret")
        print("=" * 80)
        exit(1)
    
    try:
        # Initialize client
        client = FatSecretClient()
        mapper = FoodDataMapper()
        
        print("\n✅ Client initialized successfully!")
        
        # Test 1: Search for foods
        print("\n" + "=" * 80)
        print("TEST 1: Search for 'salmon'")
        print("=" * 80)
        
        search_results = client.search_foods('salmon', max_results=5)
        foods = search_results.get('food', [])
        if not isinstance(foods, list):
            foods = [foods]
        
        print(f"\nFound {len(foods)} results:")
        for i, food in enumerate(foods[:3], 1):
            print(f"\n{i}. {food.get('food_name', 'Unknown')}")
            print(f"   ID: {food.get('food_id', 'N/A')}")
            print(f"   Type: {food.get('food_type', 'N/A')}")
            print(f"   Description: {food.get('food_description', 'N/A')[:100]}...")
        
        # Test 2: Get detailed food info
        if foods:
            print("\n" + "=" * 80)
            print("TEST 2: Get detailed info for first result")
            print("=" * 80)
            
            food_id = foods[0].get('food_id')
            detailed_food = client.get_food(food_id)
            
            print(f"\nFood: {detailed_food.get('food_name', 'Unknown')}")
            
            servings = detailed_food.get('servings', {}).get('serving', [])
            if not isinstance(servings, list):
                servings = [servings]
            
            print(f"\nAvailable servings: {len(servings)}")
            for serving in servings[:3]:
                print(f"\n  - {serving.get('serving_description', 'Unknown')}")
                print(f"    Calories: {serving.get('calories', 'N/A')}")
                print(f"    Protein: {serving.get('protein', 'N/A')}g")
                print(f"    Carbs: {serving.get('carbohydrate', 'N/A')}g")
                print(f"    Fat: {serving.get('fat', 'N/A')}g")
        
        # Test 3: Autocomplete
        print("\n" + "=" * 80)
        print("TEST 3: Autocomplete for 'app'")
        print("=" * 80)
        
        suggestions = client.autocomplete('app')
        print(f"\nSuggestions: {', '.join(suggestions[:10])}")
        
        # Test 4: Map to internal format
        print("\n" + "=" * 80)
        print("TEST 4: Map data to internal format")
        print("=" * 80)
        
        if foods:
            detailed = client.get_food(foods[0].get('food_id'))
            mapped = mapper.map_food_item(detailed)
            
            print(f"\nMapped Food: {mapped['name']}")
            print(f"Brand: {mapped['brand'] or 'Generic'}")
            print(f"\nNutrients (per 100g):")
            for key, value in mapped['nutrients'].items():
                print(f"  {key}: {value:.2f}")
        
        # Test 5: Extract training data
        print("\n" + "=" * 80)
        print("TEST 5: Extract training data")
        print("=" * 80)
        
        protein_foods = client.search_foods('chicken breast', max_results=10)
        foods_list = protein_foods.get('food', [])
        if not isinstance(foods_list, list):
            foods_list = [foods_list]
        
        mapped_foods = []
        for food in foods_list[:5]:
            detailed = client.get_food(food.get('food_id'))
            mapped = mapper.map_food_item(detailed)
            mapped_foods.append(mapped)
        
        training_data = mapper.extract_training_data(mapped_foods)
        
        print(f"\nExtracted training data for {len(training_data)} foods")
        if training_data:
            print(f"\nSample features:")
            sample = training_data[0]
            for key, value in list(sample.items())[:10]:
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nYou can now use this client to:")
        print("1. Search 1M+ foods from FatSecret database")
        print("2. Get detailed nutritional information")
        print("3. Look up foods by barcode")
        print("4. Extract training data for ML models")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
