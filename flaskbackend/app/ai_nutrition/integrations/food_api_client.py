"""
Food API Integration Client
Integrates with multiple food databases to fetch 10,000+ foods across regions
"""

import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from decimal import Decimal
import time
from datetime import datetime, timedelta
import os
from enum import Enum

from ..models.core_data_models import (
    FoodItem, FoodCategory, MeasurementUnit, NutrientContent, 
    ChemicalContent, CookingMethod
)


class FoodAPIProvider(Enum):
    """Supported food API providers"""
    USDA_FDC = "usda_fdc"  # USDA FoodData Central - 350k+ foods
    EDAMAM = "edamam"  # Edamam Nutrition API - 800k+ foods
    NUTRITIONIX = "nutritionix"  # Nutritionix API - 800k+ foods
    OPENFOODFACTS = "openfoodfacts"  # Open Food Facts - 2M+ products worldwide
    FAO_INFOODS = "fao_infoods"  # FAO International Food Composition Tables
    LOCAL_USDA = "local_usda"  # Local USDA Standard Reference Database


@dataclass
class APICredentials:
    """API credentials configuration"""
    usda_api_key: Optional[str] = None
    edamam_app_id: Optional[str] = None
    edamam_app_key: Optional[str] = None
    nutritionix_app_id: Optional[str] = None
    nutritionix_app_key: Optional[str] = None


class FoodAPIClient:
    """
    Unified client for multiple food nutrition APIs
    Provides access to 10,000+ foods across global regions
    """
    
    def __init__(self, credentials: Optional[APICredentials] = None):
        self.credentials = credentials or self._load_credentials_from_env()
        self.cache = {}
        self.cache_expiry = timedelta(days=7)
        self.rate_limits = {
            FoodAPIProvider.USDA_FDC: {"calls": 0, "reset": datetime.now()},
            FoodAPIProvider.EDAMAM: {"calls": 0, "reset": datetime.now()},
            FoodAPIProvider.NUTRITIONIX: {"calls": 0, "reset": datetime.now()},
        }
        
    def _load_credentials_from_env(self) -> APICredentials:
        """Load API credentials from environment variables"""
        return APICredentials(
            usda_api_key=os.getenv("USDA_API_KEY"),
            edamam_app_id=os.getenv("EDAMAM_APP_ID"),
            edamam_app_key=os.getenv("EDAMAM_APP_KEY"),
            nutritionix_app_id=os.getenv("NUTRITIONIX_APP_ID"),
            nutritionix_app_key=os.getenv("NUTRITIONIX_APP_KEY"),
        )
    
    def _check_rate_limit(self, provider: FoodAPIProvider, max_calls: int = 100) -> bool:
        """Check if rate limit allows API call"""
        limit_info = self.rate_limits[provider]
        
        # Reset counter if hour has passed
        if datetime.now() > limit_info["reset"]:
            limit_info["calls"] = 0
            limit_info["reset"] = datetime.now() + timedelta(hours=1)
        
        if limit_info["calls"] >= max_calls:
            return False
        
        limit_info["calls"] += 1
        return True
    
    # ==================== USDA FoodData Central API ====================
    
    def search_usda_foods(self, query: str, page_size: int = 50) -> List[Dict]:
        """
        Search USDA FoodData Central
        Free API: 1000 requests/hour
        Database: 350,000+ foods including:
        - Foundation Foods
        - SR Legacy (Standard Reference)
        - FNDDS (Survey Foods)
        - Branded Foods
        """
        if not self.credentials.usda_api_key:
            raise ValueError("USDA API key not configured")
        
        if not self._check_rate_limit(FoodAPIProvider.USDA_FDC, max_calls=900):
            raise Exception("USDA API rate limit exceeded")
        
        url = "https://api.nal.usda.gov/fdc/v1/foods/search"
        params = {
            "api_key": self.credentials.usda_api_key,
            "query": query,
            "pageSize": page_size,
            "dataType": ["Foundation", "SR Legacy", "Survey (FNDDS)"],
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("foods", [])
        except requests.exceptions.RequestException as e:
            print(f"USDA API error: {e}")
            return []
    
    def get_usda_food_details(self, fdc_id: int) -> Optional[Dict]:
        """Get detailed food information from USDA by FDC ID"""
        if not self.credentials.usda_api_key:
            return None
        
        cache_key = f"usda_{fdc_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"
        params = {"api_key": self.credentials.usda_api_key}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            self.cache[cache_key] = data
            return data
        except requests.exceptions.RequestException as e:
            print(f"USDA API error: {e}")
            return None
    
    def convert_usda_to_fooditem(self, usda_data: Dict) -> Optional[FoodItem]:
        """Convert USDA API data to FoodItem"""
        try:
            # Extract basic info
            food_id = f"usda_{usda_data.get('fdcId')}"
            name = usda_data.get("description", "Unknown")
            
            # Map USDA category to our FoodCategory
            category = self._map_usda_category(usda_data.get("foodCategory", {}).get("description", ""))
            
            # Extract nutrients
            nutrients = {}
            nutrient_list = []
            
            for nutrient in usda_data.get("foodNutrients", []):
                nutrient_name = nutrient.get("nutrient", {}).get("name", "")
                nutrient_number = nutrient.get("nutrient", {}).get("number", "")
                amount = nutrient.get("amount", 0)
                unit = nutrient.get("nutrient", {}).get("unitName", "")
                
                # Map to our nutrient IDs
                nutrient_id = self._map_usda_nutrient(nutrient_name, nutrient_number)
                if nutrient_id and amount:
                    unit_enum = self._map_unit(unit)
                    nutrient_list.append(
                        NutrientContent(nutrient_id, Decimal(str(amount)), unit_enum)
                    )
                    
                    # Store for macro calculation
                    if nutrient_number == "203":  # Protein
                        nutrients["protein"] = Decimal(str(amount))
                    elif nutrient_number == "205":  # Carbs
                        nutrients["carbohydrate"] = Decimal(str(amount))
                    elif nutrient_number == "291":  # Fiber
                        nutrients["fiber"] = Decimal(str(amount))
                    elif nutrient_number == "204":  # Total fat
                        nutrients["fat_total"] = Decimal(str(amount))
            
            # Calculate water (100 - sum of macros)
            total_macros = sum([
                nutrients.get("protein", Decimal("0")),
                nutrients.get("carbohydrate", Decimal("0")),
                nutrients.get("fat_total", Decimal("0"))
            ])
            nutrients["water"] = max(Decimal("0"), Decimal("100") - total_macros)
            
            # Create FoodItem
            food_item = FoodItem(
                food_id=food_id,
                name=name,
                category=category,
                subcategory="USDA Database",
                serving_size=Decimal("100.0"),
                serving_unit=MeasurementUnit.GRAM,
                calories_per_100g=Decimal(str(usda_data.get("foodNutrients", [{}])[0].get("amount", 0))),
                macronutrients=nutrients,
                nutrient_content=nutrient_list,
                chemical_content=[],
                allergens=set(),
                glycemic_index=None,
                glycemic_load=None,
                inflammatory_index=None,
                orac_value=None,
                organic_available=True,
                seasonal_availability=["Year-round"],
                storage_conditions="Standard storage",
                shelf_life_days=7,
                preparation_methods=["Various"],
                cooking_compatible_methods=[],
                notes=f"USDA FDC ID: {usda_data.get('fdcId')}"
            )
            
            return food_item
            
        except Exception as e:
            print(f"Error converting USDA data: {e}")
            return None
    
    # ==================== Edamam Nutrition API ====================
    
    def search_edamam_foods(self, query: str) -> List[Dict]:
        """
        Search Edamam Nutrition Database
        Free tier: 5,000 calls/month
        Database: 800,000+ foods and recipes
        """
        if not self.credentials.edamam_app_id or not self.credentials.edamam_app_key:
            raise ValueError("Edamam credentials not configured")
        
        url = "https://api.edamam.com/api/food-database/v2/parser"
        params = {
            "app_id": self.credentials.edamam_app_id,
            "app_key": self.credentials.edamam_app_key,
            "ingr": query,
            "nutrition-type": "logging"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("hints", [])
        except requests.exceptions.RequestException as e:
            print(f"Edamam API error: {e}")
            return []
    
    # ==================== Open Food Facts API ====================
    
    def search_openfoodfacts(self, query: str, country: Optional[str] = None) -> List[Dict]:
        """
        Search Open Food Facts - collaborative database
        Free API, no rate limits
        Database: 2,000,000+ products from 150+ countries
        Strong coverage: Europe, North America, Latin America, Asia
        """
        url = "https://world.openfoodfacts.org/cgi/search.pl"
        params = {
            "search_terms": query,
            "page_size": 50,
            "json": 1,
            "fields": "product_name,brands,categories,countries,nutriments,ingredients_text"
        }
        
        if country:
            params["countries"] = country
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("products", [])
        except requests.exceptions.RequestException as e:
            print(f"Open Food Facts API error: {e}")
            return []
    
    def get_openfoodfacts_by_barcode(self, barcode: str) -> Optional[Dict]:
        """Get product by barcode from Open Food Facts"""
        url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == 1:
                return data.get("product")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Open Food Facts API error: {e}")
            return None
    
    def convert_openfoodfacts_to_fooditem(self, off_data: Dict) -> Optional[FoodItem]:
        """Convert Open Food Facts data to FoodItem"""
        try:
            food_id = f"off_{off_data.get('code', 'unknown')}"
            name = off_data.get("product_name", "Unknown Product")
            brands = off_data.get("brands", "")
            if brands:
                name = f"{name} ({brands})"
            
            nutriments = off_data.get("nutriments", {})
            
            # Extract macros (per 100g)
            macros = {
                "protein": Decimal(str(nutriments.get("proteins_100g", 0))),
                "carbohydrate": Decimal(str(nutriments.get("carbohydrates_100g", 0))),
                "fiber": Decimal(str(nutriments.get("fiber_100g", 0))),
                "fat_total": Decimal(str(nutriments.get("fat_100g", 0))),
            }
            
            total_macros = sum(macros.values())
            macros["water"] = max(Decimal("0"), Decimal("100") - total_macros)
            
            # Build nutrient list
            nutrient_list = []
            nutrient_mapping = {
                "salt_100g": ("sodium", MeasurementUnit.MILLIGRAM, 1000),  # Convert g to mg
                "sugars_100g": ("sugar", MeasurementUnit.GRAM, 1),
                "saturated-fat_100g": ("saturated_fat", MeasurementUnit.GRAM, 1),
                "sodium_100g": ("sodium", MeasurementUnit.MILLIGRAM, 1000),
                "vitamin-c_100g": ("vitamin_c", MeasurementUnit.MILLIGRAM, 1),
                "calcium_100g": ("calcium", MeasurementUnit.MILLIGRAM, 1),
                "iron_100g": ("iron", MeasurementUnit.MILLIGRAM, 1),
            }
            
            for off_key, (nutrient_id, unit, multiplier) in nutrient_mapping.items():
                if off_key in nutriments:
                    amount = Decimal(str(nutriments[off_key])) * multiplier
                    nutrient_list.append(NutrientContent(nutrient_id, amount, unit))
            
            # Determine category
            categories = off_data.get("categories", "").lower()
            category = self._map_category_from_text(categories)
            
            # Get countries
            countries = off_data.get("countries", "").split(",")
            countries = [c.strip() for c in countries if c.strip()]
            
            food_item = FoodItem(
                food_id=food_id,
                name=name,
                category=category,
                subcategory=off_data.get("categories", "").split(",")[0] if "," in off_data.get("categories", "") else "Packaged Food",
                serving_size=Decimal("100.0"),
                serving_unit=MeasurementUnit.GRAM,
                calories_per_100g=Decimal(str(nutriments.get("energy-kcal_100g", 0))),
                macronutrients=macros,
                nutrient_content=nutrient_list,
                chemical_content=[],
                allergens=set(off_data.get("allergens_tags", [])),
                glycemic_index=None,
                glycemic_load=None,
                inflammatory_index=None,
                orac_value=None,
                organic_available="organic" in off_data.get("labels", "").lower(),
                seasonal_availability=["Year-round"],
                storage_conditions="As per package",
                shelf_life_days=180,
                preparation_methods=["As packaged"],
                cooking_compatible_methods=[],
                countries_of_origin=countries[:5] if countries else ["Unknown"],
                notes=f"Open Food Facts barcode: {off_data.get('code')}. Brands: {brands}"
            )
            
            return food_item
            
        except Exception as e:
            print(f"Error converting Open Food Facts data: {e}")
            return None
    
    # ==================== Regional Food Databases ====================
    
    def get_regional_foods(self, region: str) -> List[FoodItem]:
        """
        Get foods specific to a region
        Regions: usa, europe, asia, africa, latin_america, middle_east, oceania
        """
        regional_queries = {
            "usa": ["hamburger", "hot dog", "mac and cheese", "cornbread", "grits", "bbq ribs"],
            "europe": ["bratwurst", "croissant", "paella", "pasta", "pierogi", "haggis"],
            "asia": ["sushi", "dim sum", "curry", "pho", "kimchi", "satay", "biryani"],
            "africa": ["injera", "jollof rice", "fufu", "tagine", "bobotie", "bunny chow"],
            "latin_america": ["tacos", "empanadas", "ceviche", "arepas", "feijoada", "pupusas"],
            "middle_east": ["hummus", "falafel", "shawarma", "tabbouleh", "baklava", "kebab"],
            "oceania": ["pavlova", "lamington", "vegemite", "meat pie", "hangi", "poi"],
        }
        
        foods = []
        queries = regional_queries.get(region.lower(), [])
        
        for query in queries:
            # Try Open Food Facts first (best regional coverage)
            results = self.search_openfoodfacts(query)
            for result in results[:5]:  # Top 5 per query
                food_item = self.convert_openfoodfacts_to_fooditem(result)
                if food_item:
                    foods.append(food_item)
            
            time.sleep(0.5)  # Rate limiting
        
        return foods
    
    # ==================== Bulk Import ====================
    
    def bulk_import_foods(self, categories: List[str], max_per_category: int = 100) -> List[FoodItem]:
        """
        Bulk import foods from multiple sources
        Target: 10,000+ foods across all categories
        """
        all_foods = []
        
        category_queries = {
            "fruits": ["apple", "banana", "orange", "grape", "mango", "berries"],
            "vegetables": ["broccoli", "carrot", "spinach", "tomato", "onion", "lettuce"],
            "grains": ["rice", "wheat", "oats", "quinoa", "barley", "bread"],
            "legumes": ["beans", "lentils", "chickpeas", "peas", "soybeans"],
            "nuts": ["almonds", "walnuts", "cashews", "peanuts", "pistachios"],
            "dairy": ["milk", "cheese", "yogurt", "butter", "cream"],
            "meat": ["beef", "chicken", "pork", "lamb", "turkey"],
            "fish": ["salmon", "tuna", "cod", "shrimp", "tilapia"],
            "oils": ["olive oil", "coconut oil", "butter", "avocado oil"],
            "spices": ["turmeric", "cinnamon", "ginger", "pepper", "cumin"],
        }
        
        for category in categories:
            if category not in category_queries:
                continue
            
            category_foods = []
            queries = category_queries[category]
            
            for query in queries:
                # Try USDA first (most comprehensive)
                if self.credentials.usda_api_key:
                    usda_results = self.search_usda_foods(query, page_size=50)
                    for result in usda_results[:20]:
                        food_item = self.convert_usda_to_fooditem(result)
                        if food_item:
                            category_foods.append(food_item)
                
                # Then Open Food Facts for packaged/regional items
                off_results = self.search_openfoodfacts(query)
                for result in off_results[:20]:
                    food_item = self.convert_openfoodfacts_to_fooditem(result)
                    if food_item:
                        category_foods.append(food_item)
                
                time.sleep(0.5)  # Rate limiting
                
                if len(category_foods) >= max_per_category:
                    break
            
            all_foods.extend(category_foods[:max_per_category])
            print(f"Imported {len(category_foods)} foods for category: {category}")
        
        return all_foods
    
    # ==================== Helper Methods ====================
    
    def _map_usda_category(self, usda_category: str) -> FoodCategory:
        """Map USDA category to our FoodCategory enum"""
        category_lower = usda_category.lower()
        
        if "fruit" in category_lower or "berries" in category_lower:
            return FoodCategory.FRUITS
        elif "vegetable" in category_lower:
            return FoodCategory.VEGETABLES
        elif "grain" in category_lower or "cereal" in category_lower or "bread" in category_lower:
            return FoodCategory.GRAINS
        elif "legume" in category_lower or "bean" in category_lower:
            return FoodCategory.LEGUMES
        elif "nut" in category_lower or "seed" in category_lower:
            return FoodCategory.NUTS_SEEDS
        elif "dairy" in category_lower or "milk" in category_lower or "cheese" in category_lower:
            return FoodCategory.DAIRY_EGGS
        elif "meat" in category_lower or "poultry" in category_lower or "beef" in category_lower:
            return FoodCategory.MEAT_POULTRY
        elif "fish" in category_lower or "seafood" in category_lower:
            return FoodCategory.FISH_SEAFOOD
        elif "oil" in category_lower or "fat" in category_lower:
            return FoodCategory.OILS_FATS
        elif "spice" in category_lower or "herb" in category_lower:
            return FoodCategory.HERBS_SPICES
        else:
            return FoodCategory.VEGETABLES  # Default
    
    def _map_category_from_text(self, text: str) -> FoodCategory:
        """Map category from text description"""
        return self._map_usda_category(text)
    
    def _map_usda_nutrient(self, nutrient_name: str, nutrient_number: str) -> Optional[str]:
        """Map USDA nutrient to our nutrient ID"""
        # Mapping of USDA nutrient numbers to our IDs
        nutrient_map = {
            "203": "protein",
            "204": "fat_total",
            "205": "carbohydrate",
            "208": "energy_kcal",
            "291": "fiber",
            "301": "calcium",
            "303": "iron",
            "304": "magnesium",
            "305": "phosphorus",
            "306": "potassium",
            "307": "sodium",
            "309": "zinc",
            "312": "copper",
            "315": "manganese",
            "317": "selenium",
            "401": "vitamin_c",
            "404": "thiamin_b1",
            "405": "riboflavin_b2",
            "406": "niacin_b3",
            "410": "pantothenic_acid_b5",
            "415": "vitamin_b6",
            "417": "folate_b9",
            "418": "vitamin_b12",
            "320": "vitamin_a",
            "323": "vitamin_e",
            "430": "vitamin_k",
            "324": "vitamin_d",
        }
        
        return nutrient_map.get(nutrient_number)
    
    def _map_unit(self, unit_str: str) -> MeasurementUnit:
        """Map unit string to MeasurementUnit enum"""
        unit_lower = unit_str.lower()
        
        if unit_lower in ["g", "gram", "grams"]:
            return MeasurementUnit.GRAM
        elif unit_lower in ["mg", "milligram", "milligrams"]:
            return MeasurementUnit.MILLIGRAM
        elif unit_lower in ["mcg", "µg", "microgram", "micrograms", "ug"]:
            return MeasurementUnit.MICROGRAM
        elif unit_lower in ["iu", "international unit"]:
            return MeasurementUnit.IU
        elif unit_lower in ["ml", "milliliter"]:
            return MeasurementUnit.MILLILITER
        else:
            return MeasurementUnit.GRAM  # Default


# ==================== Usage Example ====================

def example_usage():
    """Example of how to use the FoodAPIClient"""
    
    # Initialize client with credentials
    credentials = APICredentials(
        usda_api_key="YOUR_USDA_API_KEY",
        edamam_app_id="YOUR_EDAMAM_APP_ID",
        edamam_app_key="YOUR_EDAMAM_APP_KEY"
    )
    
    client = FoodAPIClient(credentials)
    
    # Search for specific food
    print("Searching USDA for 'apple'...")
    usda_results = client.search_usda_foods("apple")
    if usda_results:
        food_item = client.convert_usda_to_fooditem(usda_results[0])
        print(f"Found: {food_item.name}")
    
    # Search Open Food Facts
    print("\nSearching Open Food Facts for 'yogurt'...")
    off_results = client.search_openfoodfacts("yogurt")
    if off_results:
        food_item = client.convert_openfoodfacts_to_fooditem(off_results[0])
        print(f"Found: {food_item.name}")
    
    # Get regional foods
    print("\nGetting Asian foods...")
    asian_foods = client.get_regional_foods("asia")
    print(f"Found {len(asian_foods)} Asian foods")
    
    # Bulk import
    print("\nBulk importing fruits and vegetables...")
    foods = client.bulk_import_foods(["fruits", "vegetables"], max_per_category=100)
    print(f"Imported {len(foods)} total foods")


if __name__ == "__main__":
    example_usage()
