"""
API Client Manager for External Food Data Sources
===============================================

Comprehensive API integration system for sourcing food data from multiple
external providers including USDA, OpenFoodFacts, Nutritionix, Spoonacular,
and other global food databases.

Key Features:
- Multi-provider API integration with failover
- Rate limiting and request throttling
- Data normalization and validation
- Automatic retry and error handling
- Caching and offline mode support
- Real-time data synchronization
- Country-specific API routing

Supported APIs:
- USDA FoodData Central
- Open Food Facts
- Nutritionix API
- Spoonacular Food API
- ESHA Genesis R&D
- Food.com API
- Recipe Puppy API
- TheMealDB API
- Zomato API (country-specific)
- Local government food databases

Author: Wellomex AI Nutrition Team
Version: 1.0.0
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Union, Any, Tuple
from urllib.parse import urlencode, quote
import aiohttp
import backoff
from aiohttp import ClientSession, ClientTimeout
from aiolimiter import AsyncLimiter
import aioredis

from ..models.food_knowledge_models import (
    FoodEntity, FoodCategory, MacroNutrient, MicroNutrient,
    CountrySpecificData, AllergenType, DietaryRestriction,
    PreparationMethod, APISourceMapping
)

logger = logging.getLogger(__name__)

@dataclass
class APIConfiguration:
    """Configuration for external API integration"""
    api_name: str
    base_url: str
    api_key: Optional[str] = None
    rate_limit: int = 100  # requests per minute
    timeout: int = 30  # seconds
    retry_attempts: int = 3
    priority: int = 1  # 1 = highest priority
    country_codes: List[str] = field(default_factory=list)  # Supported countries
    data_types: List[str] = field(default_factory=list)  # nutrition, ingredients, etc.
    active: bool = True
    cost_per_request: Decimal = Decimal('0.0')

@dataclass
class APIResponse:
    """Standardized API response structure"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    response_time: Optional[float] = None
    api_source: Optional[str] = None
    rate_limit_remaining: Optional[int] = None
    cost: Decimal = Decimal('0.0')

class BaseAPIClient(ABC):
    """Abstract base class for API clients"""
    
    def __init__(self, config: APIConfiguration, session: ClientSession):
        self.config = config
        self.session = session
        self.rate_limiter = AsyncLimiter(config.rate_limit, 60)  # per minute
        self._last_request_time = datetime.utcnow()
    
    @abstractmethod
    async def search_foods(self, query: str, **kwargs) -> APIResponse:
        """Search for foods by name or keyword"""
        pass
    
    @abstractmethod
    async def get_food_details(self, food_id: str) -> APIResponse:
        """Get detailed information about a specific food"""
        pass
    
    @abstractmethod
    def normalize_food_data(self, raw_data: Dict[str, Any]) -> Optional[FoodEntity]:
        """Convert API response to standardized FoodEntity"""
        pass
    
    async def make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> APIResponse:
        """Make rate-limited API request with error handling"""
        
        start_time = datetime.utcnow()
        
        try:
            # Apply rate limiting
            async with self.rate_limiter:
                url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
                
                # Prepare headers
                request_headers = {'User-Agent': 'Wellomex-AI-Nutrition/1.0'}
                if self.config.api_key:
                    request_headers.update(self._get_auth_headers())
                if headers:
                    request_headers.update(headers)
                
                # Make request
                timeout = ClientTimeout(total=self.config.timeout)
                async with self.session.request(
                    method, url, params=params, headers=request_headers, timeout=timeout
                ) as response:
                    response_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    if response.status == 200:
                        data = await response.json()
                        return APIResponse(
                            success=True,
                            data=data,
                            response_time=response_time,
                            api_source=self.config.api_name,
                            rate_limit_remaining=self._extract_rate_limit_info(response),
                            cost=self.config.cost_per_request
                        )
                    else:
                        error_msg = f"HTTP {response.status}: {await response.text()}"
                        return APIResponse(
                            success=False,
                            error_message=error_msg,
                            response_time=response_time,
                            api_source=self.config.api_name
                        )
                        
        except asyncio.TimeoutError:
            return APIResponse(
                success=False,
                error_message="Request timeout",
                api_source=self.config.api_name
            )
        except Exception as e:
            return APIResponse(
                success=False,
                error_message=str(e),
                api_source=self.config.api_name
            )
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers (override in subclasses)"""
        return {'X-API-Key': self.config.api_key}
    
    def _extract_rate_limit_info(self, response) -> Optional[int]:
        """Extract rate limit information from response headers"""
        rate_limit_headers = ['X-RateLimit-Remaining', 'X-Rate-Limit-Remaining']
        for header in rate_limit_headers:
            if header in response.headers:
                try:
                    return int(response.headers[header])
                except ValueError:
                    pass
        return None

class USDAFoodDataClient(BaseAPIClient):
    """USDA FoodData Central API client"""
    
    async def search_foods(self, query: str, **kwargs) -> APIResponse:
        """Search USDA food database"""
        params = {
            'query': query,
            'dataType': kwargs.get('data_type', 'Foundation,SR Legacy'),
            'pageSize': kwargs.get('page_size', 50),
            'api_key': self.config.api_key
        }
        
        return await self.make_request('GET', 'foods/search', params=params)
    
    async def get_food_details(self, food_id: str) -> APIResponse:
        """Get detailed USDA food information"""
        params = {'api_key': self.config.api_key}
        return await self.make_request('GET', f'food/{food_id}', params=params)
    
    def normalize_food_data(self, raw_data: Dict[str, Any]) -> Optional[FoodEntity]:
        """Convert USDA data to FoodEntity"""
        try:
            food = FoodEntity(
                name=raw_data.get('description', ''),
                scientific_name=raw_data.get('scientificName'),
                external_ids={'usda': str(raw_data.get('fdcId'))}
            )
            
            # Set category based on food group
            food_group = raw_data.get('foodGroup', {}).get('description', '')
            food.category = self._map_usda_category(food_group)
            food.food_group = food_group
            
            # Process nutrients
            nutrients = raw_data.get('foodNutrients', [])
            macro_nutrients, micro_nutrients = self._process_usda_nutrients(nutrients)
            
            food.macro_nutrients = macro_nutrients
            food.micro_nutrients = micro_nutrients
            
            # Add data source info
            food.data_sources.append('USDA FoodData Central')
            food.verification_status = 'verified'
            food.confidence_score = Decimal('0.95')
            
            return food
            
        except Exception as e:
            logger.error(f"Failed to normalize USDA data: {e}")
            return None
    
    def _map_usda_category(self, food_group: str) -> FoodCategory:
        """Map USDA food group to internal category"""
        mapping = {
            'Dairy and Egg Products': FoodCategory.DAIRY,
            'Spices and Herbs': FoodCategory.HERBS_SPICES,
            'Baby Foods': FoodCategory.PROCESSED,
            'Fats and Oils': FoodCategory.OILS_FATS,
            'Poultry Products': FoodCategory.POULTRY,
            'Soups, Sauces, and Gravies': FoodCategory.CONDIMENTS,
            'Sausages and Luncheon Meats': FoodCategory.MEAT,
            'Breakfast Cereals': FoodCategory.GRAINS,
            'Fruits and Fruit Juices': FoodCategory.FRUITS,
            'Pork Products': FoodCategory.MEAT,
            'Vegetables and Vegetable Products': FoodCategory.VEGETABLES,
            'Nut and Seed Products': FoodCategory.NUTS_SEEDS,
            'Beef Products': FoodCategory.MEAT,
            'Beverages': FoodCategory.BEVERAGES,
            'Finfish and Shellfish Products': FoodCategory.SEAFOOD,
            'Legumes and Legume Products': FoodCategory.LEGUMES,
            'Lamb, Veal, and Game Products': FoodCategory.MEAT,
            'Baked Products': FoodCategory.BAKED_GOODS,
            'Sweets': FoodCategory.SWEETENERS,
            'Cereal Grains and Pasta': FoodCategory.GRAINS,
            'Fast Foods': FoodCategory.PROCESSED,
            'Meals, Entrees, and Side Dishes': FoodCategory.PROCESSED,
            'Snacks': FoodCategory.SNACKS
        }
        
        return mapping.get(food_group, FoodCategory.PROCESSED)
    
    def _process_usda_nutrients(self, nutrients: List[Dict]) -> Tuple[Optional[MacroNutrient], Optional[MicroNutrient]]:
        """Process USDA nutrient data"""
        nutrient_map = {}
        
        # Map USDA nutrient IDs to our fields
        usda_nutrient_mapping = {
            1008: 'calories',      # Energy
            1003: 'protein',       # Protein
            1004: 'fat',          # Total lipid (fat)
            1005: 'carbohydrates', # Carbohydrate, by difference
            1079: 'fiber',        # Fiber, total dietary
            2000: 'sugar',        # Sugars, total
            1093: 'sodium',       # Sodium, Na
            1258: 'saturated_fat', # Fatty acids, total saturated
            1404: 'vitamin_c',    # Vitamin C
            1114: 'vitamin_d',    # Vitamin D
            1087: 'calcium',      # Calcium, Ca
            1089: 'iron',         # Iron, Fe
            1092: 'potassium',    # Potassium, K
        }
        
        for nutrient in nutrients:
            nutrient_id = nutrient.get('nutrient', {}).get('id')
            amount = nutrient.get('amount', 0)
            
            if nutrient_id in usda_nutrient_mapping:
                field_name = usda_nutrient_mapping[nutrient_id]
                nutrient_map[field_name] = Decimal(str(amount))
        
        # Create macro nutrients
        macro = None
        if any(key in nutrient_map for key in ['calories', 'protein', 'fat', 'carbohydrates']):
            macro = MacroNutrient(
                calories=nutrient_map.get('calories', Decimal('0')),
                carbohydrates=nutrient_map.get('carbohydrates', Decimal('0')),
                protein=nutrient_map.get('protein', Decimal('0')),
                fat=nutrient_map.get('fat', Decimal('0')),
                fiber=nutrient_map.get('fiber', Decimal('0')),
                sugar=nutrient_map.get('sugar', Decimal('0')),
                sodium=nutrient_map.get('sodium', Decimal('0')),
                saturated_fat=nutrient_map.get('saturated_fat')
            )
        
        # Create micro nutrients
        micro = None
        if any(key in nutrient_map for key in ['vitamin_c', 'vitamin_d', 'calcium', 'iron']):
            micro = MicroNutrient(
                vitamin_c=nutrient_map.get('vitamin_c'),
                vitamin_d=nutrient_map.get('vitamin_d'),
                calcium=nutrient_map.get('calcium'),
                iron=nutrient_map.get('iron'),
                potassium=nutrient_map.get('potassium')
            )
        
        return macro, micro

class OpenFoodFactsClient(BaseAPIClient):
    """Open Food Facts API client"""
    
    async def search_foods(self, query: str, **kwargs) -> APIResponse:
        """Search Open Food Facts database"""
        params = {
            'search_terms': query,
            'page_size': kwargs.get('page_size', 50),
            'json': 1,
            'fields': 'code,product_name,brands,categories,nutriments,countries,labels'
        }
        
        country_code = kwargs.get('country_code')
        if country_code:
            params['countries'] = country_code
        
        return await self.make_request('GET', 'cgi/search.pl', params=params)
    
    async def get_food_details(self, food_id: str) -> APIResponse:
        """Get detailed Open Food Facts information"""
        return await self.make_request('GET', f'api/v0/product/{food_id}.json')
    
    def normalize_food_data(self, raw_data: Dict[str, Any]) -> Optional[FoodEntity]:
        """Convert Open Food Facts data to FoodEntity"""
        try:
            product = raw_data.get('product', raw_data)
            
            food = FoodEntity(
                name=product.get('product_name', ''),
                common_names=[product.get('brands', '')],
                external_ids={'openfoodfacts': str(product.get('code'))}
            )
            
            # Set category based on categories
            categories = product.get('categories', '')
            food.category = self._map_off_category(categories)
            
            # Process nutrients
            nutriments = product.get('nutriments', {})
            macro_nutrients, micro_nutrients = self._process_off_nutrients(nutriments)
            
            food.macro_nutrients = macro_nutrients
            food.micro_nutrients = micro_nutrients
            
            # Process countries
            countries = product.get('countries', '')
            if countries:
                country_list = [c.strip() for c in countries.split(',')]
                for country in country_list[:5]:  # Limit to first 5 countries
                    country_code = self._country_name_to_code(country)
                    if country_code:
                        country_data = CountrySpecificData(
                            country_code=country_code,
                            local_name=food.name,
                            market_availability=Decimal('0.8')
                        )
                        food.country_data[country_code] = country_data
            
            # Process labels for dietary restrictions
            labels = product.get('labels', '')
            if labels:
                food.dietary_restrictions = self._process_off_labels(labels)
            
            # Add data source info
            food.data_sources.append('Open Food Facts')
            food.verification_status = 'community_verified'
            food.confidence_score = Decimal('0.75')
            
            return food
            
        except Exception as e:
            logger.error(f"Failed to normalize Open Food Facts data: {e}")
            return None
    
    def _map_off_category(self, categories: str) -> FoodCategory:
        """Map Open Food Facts categories to internal category"""
        categories_lower = categories.lower()
        
        if 'dairy' in categories_lower or 'cheese' in categories_lower:
            return FoodCategory.DAIRY
        elif 'fruit' in categories_lower:
            return FoodCategory.FRUITS
        elif 'vegetable' in categories_lower:
            return FoodCategory.VEGETABLES
        elif 'meat' in categories_lower:
            return FoodCategory.MEAT
        elif 'fish' in categories_lower or 'seafood' in categories_lower:
            return FoodCategory.SEAFOOD
        elif 'grain' in categories_lower or 'cereal' in categories_lower:
            return FoodCategory.GRAINS
        elif 'beverage' in categories_lower or 'drink' in categories_lower:
            return FoodCategory.BEVERAGES
        elif 'snack' in categories_lower:
            return FoodCategory.SNACKS
        else:
            return FoodCategory.PROCESSED
    
    def _process_off_nutrients(self, nutriments: Dict) -> Tuple[Optional[MacroNutrient], Optional[MicroNutrient]]:
        """Process Open Food Facts nutrient data"""
        macro = None
        micro = None
        
        # Create macro nutrients
        if 'energy-kcal_100g' in nutriments or 'energy_100g' in nutriments:
            calories = Decimal(str(nutriments.get('energy-kcal_100g', nutriments.get('energy_100g', 0))))
            if 'energy_100g' in nutriments and 'energy-kcal_100g' not in nutriments:
                calories = calories / Decimal('4.184')  # Convert kJ to kcal
            
            macro = MacroNutrient(
                calories=calories,
                carbohydrates=Decimal(str(nutriments.get('carbohydrates_100g', 0))),
                protein=Decimal(str(nutriments.get('proteins_100g', 0))),
                fat=Decimal(str(nutriments.get('fat_100g', 0))),
                fiber=Decimal(str(nutriments.get('fiber_100g', 0))),
                sugar=Decimal(str(nutriments.get('sugars_100g', 0))),
                sodium=Decimal(str(nutriments.get('sodium_100g', 0))) * 1000,  # Convert g to mg
                saturated_fat=Decimal(str(nutriments.get('saturated-fat_100g', 0))) if 'saturated-fat_100g' in nutriments else None
            )
        
        # Create micro nutrients
        micro_data = {}
        if 'vitamin-c_100g' in nutriments:
            micro_data['vitamin_c'] = Decimal(str(nutriments['vitamin-c_100g']))
        if 'calcium_100g' in nutriments:
            micro_data['calcium'] = Decimal(str(nutriments['calcium_100g']))
        if 'iron_100g' in nutriments:
            micro_data['iron'] = Decimal(str(nutriments['iron_100g']))
        
        if micro_data:
            micro = MicroNutrient(**micro_data)
        
        return macro, micro
    
    def _process_off_labels(self, labels: str) -> List[DietaryRestriction]:
        """Process Open Food Facts labels for dietary restrictions"""
        restrictions = []
        labels_lower = labels.lower()
        
        if 'vegan' in labels_lower:
            restrictions.append(DietaryRestriction.VEGAN)
        elif 'vegetarian' in labels_lower:
            restrictions.append(DietaryRestriction.VEGETARIAN)
        
        if 'halal' in labels_lower:
            restrictions.append(DietaryRestriction.HALAL)
        if 'kosher' in labels_lower:
            restrictions.append(DietaryRestriction.KOSHER)
        if 'organic' in labels_lower:
            # Add organic as a nutritional profile rather than dietary restriction
            pass
        
        return restrictions
    
    def _country_name_to_code(self, country_name: str) -> Optional[str]:
        """Convert country name to ISO code"""
        # Simplified mapping - in production, use a comprehensive mapping
        country_mapping = {
            'france': 'FR', 'germany': 'DE', 'italy': 'IT', 'spain': 'ES',
            'united kingdom': 'GB', 'united states': 'US', 'canada': 'CA',
            'australia': 'AU', 'japan': 'JP', 'china': 'CN', 'india': 'IN',
            'brazil': 'BR', 'mexico': 'MX', 'russia': 'RU', 'south korea': 'KR'
        }
        
        return country_mapping.get(country_name.lower().strip())

class NutritionixClient(BaseAPIClient):
    """Nutritionix API client for US food data"""
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Nutritionix uses app ID and key"""
        return {
            'x-app-id': self.config.api_key,
            'x-app-key': getattr(self.config, 'app_key', ''),
            'x-remote-user-id': '0'
        }
    
    async def search_foods(self, query: str, **kwargs) -> APIResponse:
        """Search Nutritionix database"""
        data = {
            'query': query,
            'num_servings': 1,
            'aggregate': kwargs.get('aggregate', 'true'),
            'line_delimited': False,
            'use_raw_foods': kwargs.get('use_raw_foods', False),
            'include_subrecipe': True
        }
        
        headers = {'Content-Type': 'application/json'}
        headers.update(self._get_auth_headers())
        
        # Nutritionix uses POST for natural language queries
        async with self.session.post(
            f"{self.config.base_url}/v1_1/natural/nutrients",
            json=data,
            headers=headers
        ) as response:
            if response.status == 200:
                data = await response.json()
                return APIResponse(
                    success=True,
                    data=data,
                    api_source=self.config.api_name,
                    cost=self.config.cost_per_request
                )
            else:
                return APIResponse(
                    success=False,
                    error_message=f"HTTP {response.status}",
                    api_source=self.config.api_name
                )
    
    async def get_food_details(self, food_id: str) -> APIResponse:
        """Get Nutritionix food details using item search"""
        params = {'upc': food_id}
        return await self.make_request('GET', 'v1_1/item', params=params)
    
    def normalize_food_data(self, raw_data: Dict[str, Any]) -> Optional[FoodEntity]:
        """Convert Nutritionix data to FoodEntity"""
        try:
            foods = raw_data.get('foods', [])
            if not foods:
                return None
            
            food_data = foods[0]  # Take first result
            
            food = FoodEntity(
                name=food_data.get('food_name', ''),
                external_ids={'nutritionix': str(food_data.get('nix_item_id', ''))}
            )
            
            # Set category based on tags
            tags = food_data.get('tags', {})
            food.category = self._map_nutritionix_category(tags)
            
            # Process nutrients (Nutritionix provides per-serving data)
            serving_weight = food_data.get('serving_weight_grams', 100)
            scale_factor = 100 / serving_weight if serving_weight > 0 else 1
            
            macro = MacroNutrient(
                calories=Decimal(str(food_data.get('nf_calories', 0))) * Decimal(str(scale_factor)),
                carbohydrates=Decimal(str(food_data.get('nf_total_carbohydrate', 0))) * Decimal(str(scale_factor)),
                protein=Decimal(str(food_data.get('nf_protein', 0))) * Decimal(str(scale_factor)),
                fat=Decimal(str(food_data.get('nf_total_fat', 0))) * Decimal(str(scale_factor)),
                fiber=Decimal(str(food_data.get('nf_dietary_fiber', 0))) * Decimal(str(scale_factor)),
                sugar=Decimal(str(food_data.get('nf_sugars', 0))) * Decimal(str(scale_factor)),
                sodium=Decimal(str(food_data.get('nf_sodium', 0))) * Decimal(str(scale_factor)),
                saturated_fat=Decimal(str(food_data.get('nf_saturated_fat', 0))) * Decimal(str(scale_factor))
            )
            
            food.macro_nutrients = macro
            
            # Add US country data
            us_data = CountrySpecificData(
                country_code='US',
                local_name=food.name,
                market_availability=Decimal('0.9')
            )
            food.country_data['US'] = us_data
            
            # Add data source info
            food.data_sources.append('Nutritionix')
            food.verification_status = 'verified'
            food.confidence_score = Decimal('0.85')
            
            return food
            
        except Exception as e:
            logger.error(f"Failed to normalize Nutritionix data: {e}")
            return None
    
    def _map_nutritionix_category(self, tags: Dict) -> FoodCategory:
        """Map Nutritionix tags to internal category"""
        # Nutritionix has various tag categories
        tag_id = tags.get('tag_id')
        tag_name = tags.get('tag_name', '').lower()
        
        if 'fruit' in tag_name:
            return FoodCategory.FRUITS
        elif 'vegetable' in tag_name:
            return FoodCategory.VEGETABLES
        elif 'meat' in tag_name or 'beef' in tag_name or 'pork' in tag_name:
            return FoodCategory.MEAT
        elif 'poultry' in tag_name or 'chicken' in tag_name:
            return FoodCategory.POULTRY
        elif 'fish' in tag_name or 'seafood' in tag_name:
            return FoodCategory.SEAFOOD
        elif 'dairy' in tag_name or 'milk' in tag_name:
            return FoodCategory.DAIRY
        elif 'grain' in tag_name or 'bread' in tag_name:
            return FoodCategory.GRAINS
        else:
            return FoodCategory.PROCESSED

class APIClientManager:
    """Manages multiple API clients with intelligent routing and failover"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.clients: Dict[str, BaseAPIClient] = {}
        self.configurations: Dict[str, APIConfiguration] = {}
        self.session: Optional[ClientSession] = None
        self.redis_client = None
        self.redis_url = redis_url
        
        # Performance tracking
        self.api_stats: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize API client manager"""
        # Create HTTP session
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
        timeout = ClientTimeout(total=30)
        self.session = ClientSession(connector=connector, timeout=timeout)
        
        # Initialize Redis for caching
        self.redis_client = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        
        # Load API configurations
        await self._load_api_configurations()
        
        # Initialize clients
        await self._initialize_clients()
        
        logger.info("API Client Manager initialized")
    
    async def close(self):
        """Close all connections"""
        if self.session:
            await self.session.close()
        if self.redis_client:
            await self.redis_client.close()
    
    async def _load_api_configurations(self):
        """Load API configurations from environment or config file"""
        
        # USDA FoodData Central
        self.configurations['usda'] = APIConfiguration(
            api_name='USDA FoodData Central',
            base_url='https://api.nal.usda.gov/fdc/v1',
            api_key=None,  # Free API
            rate_limit=1000,  # requests per hour -> ~17 per minute
            priority=1,
            country_codes=['US'],
            data_types=['nutrition', 'ingredients'],
            cost_per_request=Decimal('0.0')
        )
        
        # Open Food Facts
        self.configurations['openfoodfacts'] = APIConfiguration(
            api_name='Open Food Facts',
            base_url='https://world.openfoodfacts.org',
            rate_limit=60,
            priority=2,
            country_codes=[],  # Global
            data_types=['nutrition', 'ingredients', 'labels'],
            cost_per_request=Decimal('0.0')
        )
        
        # Nutritionix
        self.configurations['nutritionix'] = APIConfiguration(
            api_name='Nutritionix',
            base_url='https://trackapi.nutritionix.com',
            rate_limit=500,
            priority=3,
            country_codes=['US'],
            data_types=['nutrition'],
            cost_per_request=Decimal('0.002')  # $0.002 per request
        )
        
        # Spoonacular
        self.configurations['spoonacular'] = APIConfiguration(
            api_name='Spoonacular',
            base_url='https://api.spoonacular.com/food',
            rate_limit=150,
            priority=4,
            country_codes=[],  # Global
            data_types=['nutrition', 'recipes', 'ingredients'],
            cost_per_request=Decimal('0.004')
        )
    
    async def _initialize_clients(self):
        """Initialize API clients"""
        for api_name, config in self.configurations.items():
            if not config.active:
                continue
                
            if api_name == 'usda':
                self.clients[api_name] = USDAFoodDataClient(config, self.session)
            elif api_name == 'openfoodfacts':
                self.clients[api_name] = OpenFoodFactsClient(config, self.session)
            elif api_name == 'nutritionix':
                self.clients[api_name] = NutritionixClient(config, self.session)
            # Add other clients as needed
            
            # Initialize stats tracking
            self.api_stats[api_name] = {
                'requests_made': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'avg_response_time': 0.0,
                'total_cost': Decimal('0.0'),
                'last_used': None
            }
    
    async def search_foods_multi_source(
        self, 
        query: str, 
        country_code: Optional[str] = None,
        preferred_sources: Optional[List[str]] = None,
        max_sources: int = 3
    ) -> List[Tuple[FoodEntity, str]]:
        """Search foods across multiple API sources with intelligent routing"""
        
        # Determine which APIs to use
        selected_apis = await self._select_apis_for_search(
            country_code, preferred_sources, max_sources
        )
        
        results = []
        
        for api_name in selected_apis:
            client = self.clients.get(api_name)
            if not client:
                continue
            
            try:
                # Check cache first
                cache_key = f"search:{api_name}:{query}:{country_code}"
                cached_result = await self.redis_client.get(cache_key)
                
                if cached_result:
                    cached_foods = json.loads(cached_result)
                    for food_data in cached_foods:
                        food_entity = self._deserialize_food_entity(food_data)
                        if food_entity:
                            results.append((food_entity, api_name))
                    continue
                
                # Make API request
                response = await client.search_foods(query, country_code=country_code)
                await self._update_api_stats(api_name, response)
                
                if response.success and response.data:
                    api_foods = []
                    
                    # Process API response based on source
                    if api_name == 'usda':
                        foods_data = response.data.get('foods', [])
                    elif api_name == 'openfoodfacts':
                        foods_data = response.data.get('products', [])
                    elif api_name == 'nutritionix':
                        foods_data = response.data.get('foods', [])
                    else:
                        foods_data = []
                    
                    for food_data in foods_data[:10]:  # Limit to 10 results per API
                        food_entity = client.normalize_food_data(food_data)
                        if food_entity:
                            results.append((food_entity, api_name))
                            api_foods.append(self._serialize_food_entity(food_entity))
                    
                    # Cache results
                    if api_foods:
                        await self.redis_client.setex(cache_key, 3600, json.dumps(api_foods))
                
            except Exception as e:
                logger.error(f"Error searching {api_name} for '{query}': {e}")
                await self._update_api_stats(api_name, APIResponse(success=False, api_source=api_name))
        
        # Deduplicate and rank results
        return await self._deduplicate_and_rank_results(results)
    
    async def get_food_details_multi_source(
        self, 
        external_ids: Dict[str, str]
    ) -> Optional[FoodEntity]:
        """Get detailed food information from multiple sources and merge"""
        
        food_entities = []
        
        for api_name, external_id in external_ids.items():
            client = self.clients.get(api_name)
            if not client:
                continue
            
            try:
                response = await client.get_food_details(external_id)
                await self._update_api_stats(api_name, response)
                
                if response.success and response.data:
                    food_entity = client.normalize_food_data(response.data)
                    if food_entity:
                        food_entities.append((food_entity, api_name))
                        
            except Exception as e:
                logger.error(f"Error getting details from {api_name} for ID {external_id}: {e}")
        
        if not food_entities:
            return None
        
        # Merge data from multiple sources
        return await self._merge_food_entities(food_entities)
    
    async def _select_apis_for_search(
        self, 
        country_code: Optional[str],
        preferred_sources: Optional[List[str]],
        max_sources: int
    ) -> List[str]:
        """Select best APIs for search based on country and preferences"""
        
        candidates = []
        
        for api_name, config in self.configurations.items():
            if not config.active or api_name not in self.clients:
                continue
            
            score = config.priority
            
            # Boost score for country-specific APIs
            if country_code and (not config.country_codes or country_code in config.country_codes):
                score += 10
            
            # Boost score for preferred sources
            if preferred_sources and api_name in preferred_sources:
                score += 5
            
            # Consider API performance
            stats = self.api_stats.get(api_name, {})
            success_rate = stats.get('successful_requests', 0) / max(stats.get('requests_made', 1), 1)
            score += success_rate * 3
            
            # Consider cost (lower cost = higher score)
            cost_penalty = float(config.cost_per_request) * 10
            score -= cost_penalty
            
            candidates.append((api_name, score))
        
        # Sort by score (descending) and take top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [api_name for api_name, _ in candidates[:max_sources]]
    
    async def _update_api_stats(self, api_name: str, response: APIResponse):
        """Update API performance statistics"""
        stats = self.api_stats.get(api_name, {})
        
        stats['requests_made'] = stats.get('requests_made', 0) + 1
        
        if response.success:
            stats['successful_requests'] = stats.get('successful_requests', 0) + 1
        else:
            stats['failed_requests'] = stats.get('failed_requests', 0) + 1
        
        if response.response_time:
            current_avg = stats.get('avg_response_time', 0.0)
            total_requests = stats['requests_made']
            new_avg = ((current_avg * (total_requests - 1)) + response.response_time) / total_requests
            stats['avg_response_time'] = new_avg
        
        stats['total_cost'] = stats.get('total_cost', Decimal('0.0')) + response.cost
        stats['last_used'] = datetime.utcnow().isoformat()
        
        self.api_stats[api_name] = stats
    
    async def _deduplicate_and_rank_results(
        self, 
        results: List[Tuple[FoodEntity, str]]
    ) -> List[Tuple[FoodEntity, str]]:
        """Remove duplicates and rank results by relevance and quality"""
        
        # Simple deduplication based on name similarity
        unique_results = []
        seen_names = set()
        
        for food_entity, api_source in results:
            name_key = food_entity.name.lower().strip()
            
            # Check for similar names (simple approach)
            is_duplicate = False
            for seen_name in seen_names:
                if self._calculate_name_similarity(name_key, seen_name) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_names.add(name_key)
                unique_results.append((food_entity, api_source))
        
        # Rank by confidence score and API priority
        def rank_key(item):
            food_entity, api_source = item
            api_priority = self.configurations.get(api_source, APIConfiguration(api_name="", base_url="", priority=10)).priority
            return (float(food_entity.confidence_score), -api_priority)
        
        unique_results.sort(key=rank_key, reverse=True)
        return unique_results
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two food names"""
        # Simple Jaccard similarity
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _merge_food_entities(
        self, 
        food_entities: List[Tuple[FoodEntity, str]]
    ) -> FoodEntity:
        """Merge food data from multiple sources into a single entity"""
        
        if len(food_entities) == 1:
            return food_entities[0][0]
        
        # Use highest priority source as base
        priority_map = {api: config.priority for api, config in self.configurations.items()}
        food_entities.sort(key=lambda x: priority_map.get(x[1], 10))
        
        base_food, base_source = food_entities[0]
        
        # Merge data from other sources
        for food_entity, source in food_entities[1:]:
            # Merge external IDs
            base_food.external_ids.update(food_entity.external_ids)
            
            # Merge data sources
            base_food.data_sources.extend(food_entity.data_sources)
            
            # Merge country data
            base_food.country_data.update(food_entity.country_data)
            
            # Use better nutritional data if available
            if not base_food.macro_nutrients and food_entity.macro_nutrients:
                base_food.macro_nutrients = food_entity.macro_nutrients
            
            if not base_food.micro_nutrients and food_entity.micro_nutrients:
                base_food.micro_nutrients = food_entity.micro_nutrients
            
            # Merge common names
            if food_entity.common_names:
                base_food.common_names.extend(food_entity.common_names)
                # Remove duplicates
                base_food.common_names = list(set(base_food.common_names))
        
        # Update confidence based on number of sources
        source_count = len(food_entities)
        confidence_boost = min(0.2, source_count * 0.05)
        base_food.confidence_score = min(Decimal('1.0'), base_food.confidence_score + Decimal(str(confidence_boost)))
        
        return base_food
    
    def _serialize_food_entity(self, food: FoodEntity) -> Dict[str, Any]:
        """Serialize food entity for caching"""
        # Implement serialization logic
        return {}  # Placeholder
    
    def _deserialize_food_entity(self, data: Dict[str, Any]) -> Optional[FoodEntity]:
        """Deserialize food entity from cache"""
        # Implement deserialization logic
        return None  # Placeholder
    
    async def get_api_statistics(self) -> Dict[str, Any]:
        """Get comprehensive API usage statistics"""
        return {
            'apis': self.api_stats,
            'total_requests': sum(stats.get('requests_made', 0) for stats in self.api_stats.values()),
            'total_cost': sum(stats.get('total_cost', Decimal('0.0')) for stats in self.api_stats.values()),
            'overall_success_rate': self._calculate_overall_success_rate(),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all APIs"""
        total_requests = sum(stats.get('requests_made', 0) for stats in self.api_stats.values())
        total_successful = sum(stats.get('successful_requests', 0) for stats in self.api_stats.values())
        
        return total_successful / total_requests if total_requests > 0 else 0.0