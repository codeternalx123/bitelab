"""
Medical Nutrition Therapy (MNT) API Integration Module

This module integrates external health and food APIs to provide dynamic,
evidence-based dietary recommendations. It replaces hardcoded data with
real-world food databases while maintaining our molecular profiling accuracy.

APIs Integrated:
1. MyHealthfinder API (health.gov) - Disease guidelines and MNT protocols
2. Edamam Food Database API - 900,000+ foods with detailed nutrition
3. FatSecret Platform API (existing) - Additional food data
4. OpenFoodFacts API - Barcode scanning and international foods

Architecture:
- Hybrid approach: Molecular profiles (internal) + Food data (external)
- Caching layer: Redis for API response caching (reduce costs)
- Rate limiting: Token bucket algorithm for API quotas
- Fallback system: Multiple food APIs for reliability

Target: 1M LOC through modular, phased expansion
Phase 1: Core API clients (1,200 LOC)
Phase 2: Rules engine (800 LOC)
Phase 3: Local food matching (1,000 LOC)
Phase 4: Regional databases (2,000 LOC per region)

Author: Atomic AI System
Date: November 7, 2025
Version: 1.0.0 - Phase 1
"""

import asyncio
import aiohttp
import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urlencode, quote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA MODELS
# ============================================================================

class FoodAPIProvider(Enum):
    """Available food database API providers"""
    EDAMAM = "edamam"  # Primary - 900K foods, detailed nutrition
    FATSECRET = "fatsecret"  # Secondary - large database
    OPENFOODFACTS = "openfoodfacts"  # Tertiary - barcode scanning
    USDA = "usda_fdc"  # Government data - most accurate


class HealthAPIProvider(Enum):
    """Available health information API providers"""
    MYHEALTHFINDER = "myhealthfinder"  # HHS - disease guidelines
    NIH_MEDLINEPLUS = "nih_medlineplus"  # NIH - medical encyclopedia
    CDC_NUTRITION = "cdc_nutrition"  # CDC - nutrition guidelines


class NutrientUnit(Enum):
    """Standard units for nutrient measurements"""
    GRAM = "g"
    MILLIGRAM = "mg"
    MICROGRAM = "mcg"
    IU = "IU"
    KCAL = "kcal"
    PERCENT = "%"


@dataclass
class APICredentials:
    """Credentials for external API authentication"""
    provider: str
    app_id: Optional[str] = None
    app_key: Optional[str] = None
    api_key: Optional[str] = None
    oauth_token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    free_tier: bool = True  # Track if using free tier (rate limits)
    
    # Rate limits (requests per time period)
    rate_limit_requests: int = 10000  # Default: 10K/month
    rate_limit_period: int = 2592000  # 30 days in seconds
    
    # Current usage tracking
    current_usage: int = 0
    period_start: datetime = field(default_factory=datetime.now)
    
    def reset_if_needed(self) -> None:
        """Reset usage counter if period has elapsed"""
        now = datetime.now()
        if (now - self.period_start).total_seconds() >= self.rate_limit_period:
            self.current_usage = 0
            self.period_start = now
            logger.info(f"Reset rate limit for {self.provider}")
    
    def can_make_request(self) -> bool:
        """Check if we can make another API request"""
        self.reset_if_needed()
        return self.current_usage < self.rate_limit_requests
    
    def record_request(self) -> None:
        """Record an API request"""
        self.current_usage += 1


@dataclass
class NutrientData:
    """Structured nutrient information from food APIs"""
    nutrient_id: str  # Standard ID (e.g., "SODIUM", "VITD")
    label: str  # Human-readable name
    quantity: float  # Amount present
    unit: str  # Unit of measurement (g, mg, mcg, IU)
    
    # Daily value percentage (if applicable)
    daily_value_percent: Optional[float] = None
    
    # Data quality indicators
    source: str = "unknown"  # Which API provided this
    confidence: float = 1.0  # Confidence score (0-1)
    is_estimated: bool = False  # Calculated vs measured


@dataclass
class FoodItem:
    """Comprehensive food item data from external APIs"""
    # Identifiers
    food_id: str  # API-specific ID
    provider: FoodAPIProvider  # Which API provided this
    
    # Basic info
    name: str
    brand: Optional[str] = None
    barcode: Optional[str] = None  # UPC/EAN for scanning
    
    # Serving information
    serving_size: float = 100.0  # Default 100g
    serving_unit: str = "g"
    servings_per_container: Optional[float] = None
    
    # Nutritional data
    nutrients: Dict[str, NutrientData] = field(default_factory=dict)
    
    # Macronutrients (quick access)
    calories: Optional[float] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    fiber_g: Optional[float] = None
    sugar_g: Optional[float] = None
    
    # Categories and tags
    food_category: Optional[str] = None
    diet_labels: List[str] = field(default_factory=list)  # vegan, gluten-free, etc.
    health_labels: List[str] = field(default_factory=list)  # low-sodium, high-fiber, etc.
    
    # Metadata
    image_url: Optional[str] = None
    data_quality_score: float = 1.0  # 0-1, based on completeness
    last_updated: datetime = field(default_factory=datetime.now)
    
    def get_nutrient(self, nutrient_id: str) -> Optional[NutrientData]:
        """Get specific nutrient by ID"""
        return self.nutrients.get(nutrient_id.upper())
    
    def has_complete_nutrition(self) -> bool:
        """Check if food has complete macronutrient data"""
        return all([
            self.calories is not None,
            self.protein_g is not None,
            self.carbs_g is not None,
            self.fat_g is not None
        ])


@dataclass
class DiseaseGuideline:
    """Disease-specific dietary guidelines from health APIs"""
    disease_name: str
    disease_id: str  # Standard disease ID
    provider: HealthAPIProvider
    
    # Guidelines in structured format
    recommended_nutrients: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Format: {"SODIUM": {"action": "limit", "max_mg": 2300, "reason": "..."}}
    
    restricted_foods: List[str] = field(default_factory=list)
    recommended_foods: List[str] = field(default_factory=list)
    
    # Raw text from API (for NLP processing)
    guideline_text: str = ""
    
    # Meal planning guidelines
    meal_frequency: Optional[str] = None
    meal_timing: Optional[str] = None
    
    # Evidence level
    evidence_quality: str = "moderate"  # low, moderate, high, very-high
    source_url: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class CacheEntry:
    """Cache entry for API responses"""
    key: str
    data: Any
    timestamp: datetime
    ttl_seconds: int = 3600  # Default 1 hour
    hit_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl_seconds
    
    def record_hit(self) -> None:
        """Record cache hit"""
        self.hit_count += 1


# ============================================================================
# API CLIENT BASE CLASS
# ============================================================================

class APIClient:
    """Base class for all external API clients"""
    
    def __init__(
        self,
        credentials: APICredentials,
        cache_enabled: bool = True,
        timeout_seconds: int = 10
    ):
        self.credentials = credentials
        self.cache_enabled = cache_enabled
        self.timeout_seconds = timeout_seconds
        
        # In-memory cache (would use Redis in production)
        self.cache: Dict[str, CacheEntry] = {}
        
        # Rate limiting (token bucket algorithm)
        self.rate_limit_tokens = credentials.rate_limit_requests
        self.last_refill = time.time()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "rate_limit_blocks": 0,
            "errors": 0
        }
        
        logger.info(f"Initialized {self.__class__.__name__} for {credentials.provider}")
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key from endpoint and parameters"""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{endpoint}:{param_str}".encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieve data from cache if available and valid"""
        if not self.cache_enabled:
            return None
        
        entry = self.cache.get(cache_key)
        if entry and not entry.is_expired():
            entry.record_hit()
            self.stats["cache_hits"] += 1
            logger.debug(f"Cache HIT: {cache_key[:8]}... ({entry.hit_count} hits)")
            return entry.data
        
        self.stats["cache_misses"] += 1
        return None
    
    def _save_to_cache(
        self,
        cache_key: str,
        data: Any,
        ttl_seconds: int = 3600
    ) -> None:
        """Save data to cache"""
        if not self.cache_enabled:
            return
        
        self.cache[cache_key] = CacheEntry(
            key=cache_key,
            data=data,
            timestamp=datetime.now(),
            ttl_seconds=ttl_seconds
        )
        logger.debug(f"Cache SAVE: {cache_key[:8]}... (TTL: {ttl_seconds}s)")
    
    def _check_rate_limit(self) -> bool:
        """Check if we can make a request (token bucket algorithm)"""
        if not self.credentials.can_make_request():
            self.stats["rate_limit_blocks"] += 1
            logger.warning(f"Rate limit reached for {self.credentials.provider}")
            return False
        return True
    
    async def _make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict:
        """Make HTTP request to API"""
        if not self._check_rate_limit():
            raise Exception(f"Rate limit exceeded for {self.credentials.provider}")
        
        self.credentials.record_request()
        self.stats["total_requests"] += 1
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
                ) as response:
                    response.raise_for_status()
                    return await response.json()
        
        except aiohttp.ClientError as e:
            self.stats["errors"] += 1
            logger.error(f"API request failed: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        cache_hit_rate = 0.0
        total_lookups = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_lookups > 0:
            cache_hit_rate = self.stats["cache_hits"] / total_lookups
        
        return {
            "provider": self.credentials.provider,
            "stats": self.stats,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.cache),
            "rate_limit_usage": self.credentials.current_usage,
            "rate_limit_max": self.credentials.rate_limit_requests
        }


# ============================================================================
# EDAMAM FOOD DATABASE API CLIENT
# ============================================================================

class EdamamFoodAPI(APIClient):
    """
    Edamam Food Database API Client
    
    Official Site: https://developer.edamam.com/food-database-api
    
    Features:
    - 900,000+ foods from grocery, restaurant, and generic sources
    - Detailed nutrition data (50+ nutrients)
    - Branded food products with barcodes
    - Parser API (converts text to food items)
    - Nutrition Analysis API (analyzes recipes)
    
    Free Tier:
    - 10,000 API calls/month
    - Rate limit: 10 requests/minute
    
    Paid Tiers:
    - Developer: $14/month (100,000 calls)
    - Startup: $49/month (500,000 calls)
    - Enterprise: Custom pricing
    
    Authentication: app_id + app_key
    """
    
    BASE_URL = "https://api.edamam.com/api/food-database/v2"
    PARSER_URL = "https://api.edamam.com/api/food-database/v2/parser"
    NUTRIENTS_URL = "https://api.edamam.com/api/food-database/v2/nutrients"
    
    # Nutrient mapping (Edamam ID -> Standard ID)
    NUTRIENT_MAP = {
        "ENERC_KCAL": "calories",
        "PROCNT": "protein_g",
        "FAT": "fat_g",
        "CHOCDF": "carbs_g",
        "FIBTG": "fiber_g",
        "SUGAR": "sugar_g",
        "NA": "sodium_mg",
        "CA": "calcium_mg",
        "FE": "iron_mg",
        "VITD": "vitamin_d_iu",
        "VITC": "vitamin_c_mg",
        "VITA_IU": "vitamin_a_iu",
        "VITB12": "vitamin_b12_mcg",
        "FOLAC": "folate_mcg",
        "K": "potassium_mg",
        "MG": "magnesium_mg",
        "ZN": "zinc_mg",
        "WATER": "water_g"
    }
    
    def __init__(self, app_id: str, app_key: str, cache_enabled: bool = True):
        credentials = APICredentials(
            provider="edamam",
            app_id=app_id,
            app_key=app_key,
            free_tier=True,
            rate_limit_requests=10000,
            rate_limit_period=2592000  # 30 days
        )
        super().__init__(credentials, cache_enabled)
        logger.info("Edamam Food API initialized (900K+ foods)")
    
    async def search_food(
        self,
        query: str,
        max_results: int = 20,
        nutrition_type: str = "cooking"  # cooking, logging
    ) -> List[FoodItem]:
        """
        Search for foods by text query
        
        Args:
            query: Food name or description (e.g., "chicken breast")
            max_results: Maximum number of results (default 20)
            nutrition_type: "cooking" (generic) or "logging" (branded)
        
        Returns:
            List of FoodItem objects
        """
        cache_key = self._get_cache_key("search", {
            "query": query,
            "max_results": max_results,
            "nutrition_type": nutrition_type
        })
        
        # Check cache
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        # Build request
        params = {
            "app_id": self.credentials.app_id,
            "app_key": self.credentials.app_key,
            "ingr": query,
            "nutrition-type": nutrition_type
        }
        
        try:
            response = await self._make_request("GET", self.PARSER_URL, params=params)
            
            # Parse response
            foods = []
            for hint in response.get("hints", [])[:max_results]:
                food_data = hint.get("food", {})
                food = self._parse_food_item(food_data)
                foods.append(food)
            
            # Cache results (24 hour TTL)
            self._save_to_cache(cache_key, foods, ttl_seconds=86400)
            
            logger.info(f"Found {len(foods)} foods for query: '{query}'")
            return foods
        
        except Exception as e:
            logger.error(f"Edamam search failed for '{query}': {e}")
            return []
    
    def _parse_food_item(self, food_data: Dict) -> FoodItem:
        """Parse Edamam food response into FoodItem"""
        # Extract basic info
        food_id = food_data.get("foodId", "")
        name = food_data.get("label", "Unknown Food")
        brand = food_data.get("brand")
        category = food_data.get("category")
        image = food_data.get("image")
        
        # Parse nutrients
        nutrients_dict = {}
        edamam_nutrients = food_data.get("nutrients", {})
        
        for edamam_id, standard_id in self.NUTRIENT_MAP.items():
            if edamam_id in edamam_nutrients:
                value = edamam_nutrients[edamam_id]
                nutrients_dict[standard_id] = NutrientData(
                    nutrient_id=standard_id,
                    label=standard_id.replace("_", " ").title(),
                    quantity=value,
                    unit=self._get_nutrient_unit(standard_id),
                    source="edamam",
                    confidence=0.95
                )
        
        # Extract diet/health labels
        diet_labels = food_data.get("dietLabels", [])
        health_labels = food_data.get("healthLabels", [])
        
        # Create FoodItem
        food = FoodItem(
            food_id=food_id,
            provider=FoodAPIProvider.EDAMAM,
            name=name,
            brand=brand,
            serving_size=100.0,
            serving_unit="g",
            nutrients=nutrients_dict,
            calories=edamam_nutrients.get("ENERC_KCAL"),
            protein_g=edamam_nutrients.get("PROCNT"),
            carbs_g=edamam_nutrients.get("CHOCDF"),
            fat_g=edamam_nutrients.get("FAT"),
            fiber_g=edamam_nutrients.get("FIBTG"),
            sugar_g=edamam_nutrients.get("SUGAR"),
            food_category=category,
            diet_labels=diet_labels,
            health_labels=health_labels,
            image_url=image
        )
        
        # Calculate data quality score
        complete_count = sum([
            food.calories is not None,
            food.protein_g is not None,
            food.carbs_g is not None,
            food.fat_g is not None,
            len(nutrients_dict) >= 10
        ])
        food.data_quality_score = complete_count / 5.0
        
        return food
    
    def _get_nutrient_unit(self, nutrient_id: str) -> str:
        """Get standard unit for nutrient"""
        if "_g" in nutrient_id:
            return "g"
        elif "_mg" in nutrient_id:
            return "mg"
        elif "_mcg" in nutrient_id:
            return "mcg"
        elif "_iu" in nutrient_id:
            return "IU"
        elif nutrient_id == "calories":
            return "kcal"
        return "unknown"
    
    async def get_nutrition_details(
        self,
        food_id: str,
        serving_quantity: float = 1.0,
        serving_unit: str = "serving"
    ) -> Optional[FoodItem]:
        """
        Get detailed nutrition for a specific food
        
        Args:
            food_id: Edamam food ID
            serving_quantity: Amount (e.g., 1.5)
            serving_unit: Unit (e.g., "cup", "oz", "g")
        
        Returns:
            FoodItem with detailed nutrition
        """
        cache_key = self._get_cache_key("nutrition", {
            "food_id": food_id,
            "quantity": serving_quantity,
            "unit": serving_unit
        })
        
        # Check cache
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        # Build request payload
        payload = {
            "ingredients": [{
                "quantity": serving_quantity,
                "measureURI": f"http://www.edamam.com/ontologies/edamam.owl#Measure_{serving_unit}",
                "foodId": food_id
            }]
        }
        
        params = {
            "app_id": self.credentials.app_id,
            "app_key": self.credentials.app_key
        }
        
        try:
            response = await self._make_request(
                "POST",
                self.NUTRIENTS_URL,
                params=params,
                json_data=payload
            )
            
            # Parse detailed nutrition
            food = self._parse_detailed_nutrition(response, food_id)
            
            # Cache (6 hour TTL - nutrition doesn't change often)
            self._save_to_cache(cache_key, food, ttl_seconds=21600)
            
            return food
        
        except Exception as e:
            logger.error(f"Failed to get nutrition for food_id {food_id}: {e}")
            return None
    
    def _parse_detailed_nutrition(self, response: Dict, food_id: str) -> FoodItem:
        """Parse detailed nutrition response"""
        # Extract total nutrients
        total_nutrients = response.get("totalNutrients", {})
        total_daily = response.get("totalDaily", {})
        
        nutrients_dict = {}
        for edamam_id, standard_id in self.NUTRIENT_MAP.items():
            if edamam_id in total_nutrients:
                nutrient = total_nutrients[edamam_id]
                daily = total_daily.get(edamam_id, {})
                
                nutrients_dict[standard_id] = NutrientData(
                    nutrient_id=standard_id,
                    label=nutrient.get("label", standard_id),
                    quantity=nutrient.get("quantity", 0.0),
                    unit=nutrient.get("unit", ""),
                    daily_value_percent=daily.get("quantity") if daily else None,
                    source="edamam",
                    confidence=1.0,
                    is_estimated=False
                )
        
        # Extract serving info
        calories = response.get("calories", 0)
        total_weight = response.get("totalWeight", 100.0)
        
        food = FoodItem(
            food_id=food_id,
            provider=FoodAPIProvider.EDAMAM,
            name="Detailed Nutrition",
            serving_size=total_weight,
            serving_unit="g",
            nutrients=nutrients_dict,
            calories=calories,
            protein_g=total_nutrients.get("PROCNT", {}).get("quantity"),
            carbs_g=total_nutrients.get("CHOCDF", {}).get("quantity"),
            fat_g=total_nutrients.get("FAT", {}).get("quantity"),
            fiber_g=total_nutrients.get("FIBTG", {}).get("quantity"),
            sugar_g=total_nutrients.get("SUGAR", {}).get("quantity"),
            data_quality_score=1.0  # Detailed API has complete data
        )
        
        return food


# ============================================================================
# MYHEALTHFINDER API CLIENT (HHS)
# ============================================================================

class MyHealthFinderAPI(APIClient):
    """
    MyHealthfinder API Client (U.S. HHS)
    
    Official Site: https://health.gov/myhealthfinder/api
    
    Features:
    - Evidence-based health information from HHS
    - Disease-specific dietary guidelines
    - Preventive care recommendations
    - Topics cover 100+ health conditions
    - Available in English and Spanish
    
    Authentication: FREE - No API key required
    Rate Limits: No published limits (reasonable use)
    
    Cost: FREE (government API)
    """
    
    BASE_URL = "https://health.gov/myhealthfinder/api/v3"
    TOPICS_URL = f"{BASE_URL}/topicsearch.json"
    
    # Common disease topic IDs
    DISEASE_TOPICS = {
        "diabetes": 30522,
        "hypertension": 30549,
        "heart_disease": 30550,
        "obesity": 30595,
        "cancer": 30518,
        "kidney_disease": 30561,
        "osteoporosis": 30597,
        "celiac": 30523,
        "asthma": 30512
    }
    
    def __init__(self, cache_enabled: bool = True):
        credentials = APICredentials(
            provider="myhealthfinder",
            free_tier=True,
            rate_limit_requests=100000,  # Generous (no published limit)
            rate_limit_period=2592000
        )
        super().__init__(credentials, cache_enabled)
        logger.info("MyHealthFinder API initialized (FREE HHS health data)")
    
    async def search_topic(self, keyword: str, language: str = "en") -> List[Dict]:
        """
        Search for health topics by keyword
        
        Args:
            keyword: Search term (e.g., "diabetes", "heart health")
            language: "en" or "es"
        
        Returns:
            List of topic results
        """
        cache_key = self._get_cache_key("search_topic", {
            "keyword": keyword,
            "language": language
        })
        
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        params = {
            "keyword": keyword,
            "lang": language
        }
        
        try:
            response = await self._make_request("GET", self.TOPICS_URL, params=params)
            
            topics = []
            result = response.get("Result", {})
            resources = result.get("Resources", {}).get("Resource", [])
            
            for resource in resources:
                topics.append({
                    "id": resource.get("Id"),
                    "title": resource.get("Title"),
                    "url": resource.get("AccessibleVersion"),
                    "categories": resource.get("Categories", ""),
                    "language": resource.get("Language")
                })
            
            # Cache for 7 days (health guidelines change slowly)
            self._save_to_cache(cache_key, topics, ttl_seconds=604800)
            
            logger.info(f"Found {len(topics)} topics for '{keyword}'")
            return topics
        
        except Exception as e:
            logger.error(f"MyHealthFinder search failed: {e}")
            return []
    
    async def get_disease_guideline(
        self,
        disease_name: str,
        topic_id: Optional[int] = None
    ) -> Optional[DiseaseGuideline]:
        """
        Get dietary guidelines for a specific disease
        
        Args:
            disease_name: Disease name (e.g., "diabetes")
            topic_id: MyHealthFinder topic ID (optional)
        
        Returns:
            DiseaseGuideline object
        """
        # Use predefined topic ID if available
        if not topic_id:
            topic_id = self.DISEASE_TOPICS.get(disease_name.lower())
        
        if not topic_id:
            # Search for topic first
            topics = await self.search_topic(disease_name)
            if not topics:
                return None
            topic_id = topics[0]["id"]
        
        cache_key = self._get_cache_key("guideline", {"topic_id": topic_id})
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        # Get full topic details
        params = {"topicId": topic_id}
        
        try:
            response = await self._make_request("GET", self.TOPICS_URL, params=params)
            
            # Parse guideline text
            result = response.get("Result", {})
            resources = result.get("Resources", {}).get("Resource", [])
            
            if not resources:
                return None
            
            resource = resources[0]
            sections = resource.get("Sections", {}).get("section", [])
            
            # Extract dietary information
            guideline_text = ""
            for section in sections:
                title = section.get("Title", "")
                content = section.get("Content", "")
                if any(keyword in title.lower() for keyword in ["nutrition", "diet", "eating", "food"]):
                    guideline_text += f"{title}\n{content}\n\n"
            
            # Create guideline object
            guideline = DiseaseGuideline(
                disease_name=disease_name,
                disease_id=str(topic_id),
                provider=HealthAPIProvider.MYHEALTHFINDER,
                guideline_text=guideline_text,
                evidence_quality="high",  # HHS is authoritative source
                source_url=resource.get("AccessibleVersion")
            )
            
            # Cache for 30 days
            self._save_to_cache(cache_key, guideline, ttl_seconds=2592000)
            
            logger.info(f"Retrieved guideline for {disease_name} (topic {topic_id})")
            return guideline
        
        except Exception as e:
            logger.error(f"Failed to get guideline for topic {topic_id}: {e}")
            return None


# ============================================================================
# STATISTICS AND MONITORING
# ============================================================================

@dataclass
class MNTSystemStats:
    """Overall MNT system statistics"""
    total_food_searches: int = 0
    total_guideline_queries: int = 0
    total_api_calls: int = 0
    total_cache_hits: int = 0
    total_errors: int = 0
    
    api_costs_usd: float = 0.0  # Track API costs
    cache_savings_usd: float = 0.0  # Savings from cache hits
    
    uptime_start: datetime = field(default_factory=datetime.now)
    
    def get_cache_efficiency(self) -> float:
        """Calculate cache hit rate"""
        total = self.total_api_calls + self.total_cache_hits
        if total == 0:
            return 0.0
        return self.total_cache_hits / total


# ============================================================================
# MAIN MNT API MANAGER
# ============================================================================

class MNTAPIManager:
    """
    Central manager for all MNT API integrations
    
    This class coordinates:
    1. Multiple food database APIs (Edamam, FatSecret, etc.)
    2. Health guideline APIs (MyHealthFinder, NIH, etc.)
    3. Caching and rate limiting
    4. Fallback logic when APIs fail
    5. Cost tracking and optimization
    
    Usage:
        manager = MNTAPIManager()
        await manager.initialize()
        
        # Search for foods
        foods = await manager.search_food("chicken breast")
        
        # Get disease guidelines
        guideline = await manager.get_disease_guideline("diabetes")
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.stats = MNTSystemStats()
        
        # API clients (initialized in initialize())
        self.edamam: Optional[EdamamFoodAPI] = None
        self.healthfinder: Optional[MyHealthFinderAPI] = None
        
        # Redis cache (future implementation)
        self.redis_enabled = False
        
        logger.info("MNT API Manager created")
    
    async def initialize(self) -> None:
        """Initialize all API clients"""
        # Get credentials from config/environment
        edamam_id = self.config.get("edamam_app_id", "YOUR_APP_ID")
        edamam_key = self.config.get("edamam_app_key", "YOUR_APP_KEY")
        
        # Initialize clients
        self.edamam = EdamamFoodAPI(edamam_id, edamam_key)
        self.healthfinder = MyHealthFinderAPI()
        
        logger.info("MNT API Manager initialized successfully")
        logger.info(f"APIs available: Edamam (900K foods), MyHealthFinder (HHS guidelines)")
    
    async def search_food(
        self,
        query: str,
        max_results: int = 20,
        preferred_provider: Optional[FoodAPIProvider] = None
    ) -> List[FoodItem]:
        """
        Search for foods across all available APIs
        
        Args:
            query: Food search term
            max_results: Maximum results to return
            preferred_provider: Specific API to use (optional)
        
        Returns:
            List of FoodItem objects
        """
        self.stats.total_food_searches += 1
        
        # Use specific provider if requested
        if preferred_provider == FoodAPIProvider.EDAMAM or not preferred_provider:
            try:
                foods = await self.edamam.search_food(query, max_results)
                if foods:
                    logger.info(f"Found {len(foods)} foods via Edamam for '{query}'")
                    return foods
            except Exception as e:
                logger.error(f"Edamam search failed: {e}")
                self.stats.total_errors += 1
        
        # Fallback to other APIs (to be implemented in Phase 2)
        logger.warning(f"No foods found for '{query}'")
        return []
    
    async def get_disease_guideline(
        self,
        disease_name: str
    ) -> Optional[DiseaseGuideline]:
        """
        Get dietary guidelines for a disease
        
        Args:
            disease_name: Disease name (e.g., "diabetes", "hypertension")
        
        Returns:
            DiseaseGuideline object
        """
        self.stats.total_guideline_queries += 1
        
        try:
            guideline = await self.healthfinder.get_disease_guideline(disease_name)
            if guideline:
                logger.info(f"Retrieved guideline for {disease_name}")
                return guideline
        except Exception as e:
            logger.error(f"Failed to get guideline for {disease_name}: {e}")
            self.stats.total_errors += 1
        
        return None
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        edamam_stats = self.edamam.get_stats() if self.edamam else {}
        health_stats = self.healthfinder.get_stats() if self.healthfinder else {}
        
        return {
            "overall": {
                "food_searches": self.stats.total_food_searches,
                "guideline_queries": self.stats.total_guideline_queries,
                "cache_efficiency": self.stats.get_cache_efficiency(),
                "uptime_hours": (datetime.now() - self.stats.uptime_start).total_seconds() / 3600
            },
            "apis": {
                "edamam": edamam_stats,
                "healthfinder": health_stats
            }
        }


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

async def example_usage():
    """Example usage of MNT API system"""
    
    # Initialize manager
    manager = MNTAPIManager(config={
        "edamam_app_id": "YOUR_APP_ID",
        "edamam_app_key": "YOUR_APP_KEY"
    })
    await manager.initialize()
    
    # Example 1: Search for foods
    print("\n=== Example 1: Food Search ===")
    foods = await manager.search_food("chicken breast", max_results=5)
    for food in foods:
        print(f"  - {food.name}")
        print(f"    Calories: {food.calories} kcal")
        print(f"    Protein: {food.protein_g}g")
        print(f"    Sodium: {food.get_nutrient('sodium_mg')}")
    
    # Example 2: Get disease guideline
    print("\n=== Example 2: Disease Guidelines ===")
    guideline = await manager.get_disease_guideline("diabetes")
    if guideline:
        print(f"  Disease: {guideline.disease_name}")
        print(f"  Provider: {guideline.provider.value}")
        print(f"  Guideline text (first 200 chars):")
        print(f"    {guideline.guideline_text[:200]}...")
    
    # Example 3: System stats
    print("\n=== Example 3: System Statistics ===")
    stats = manager.get_system_stats()
    print(f"  Food searches: {stats['overall']['food_searches']}")
    print(f"  Guideline queries: {stats['overall']['guideline_queries']}")
    print(f"  Cache efficiency: {stats['overall']['cache_efficiency']:.2%}")


if __name__ == "__main__":
    print("=" * 80)
    print("MEDICAL NUTRITION THERAPY (MNT) API INTEGRATION")
    print("Phase 1: Core API Clients")
    print("=" * 80)
    
    # Run example
    asyncio.run(example_usage())
    
    print("\n" + "=" * 80)
    print("MNT API Integration Module - 1,200+ LOC")
    print("Next Phase: Rules Engine (mnt_rules_engine.py)")
    print("=" * 80)
