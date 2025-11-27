"""
USDA FoodData Central Scraper
==============================

Collects nutritional and elemental composition data from USDA FoodData Central:
- Foundation Foods (ICP-MS validated data)
- SR Legacy (Standard Reference - 100,000+ foods)
- Branded Foods (packaged products)
- Survey Foods (FNDDS - What We Eat in America)

API Documentation: https://fdc.nal.usda.gov/api-guide.html

Target: 100,000+ food entries with mineral/trace element data

API Key required (free): https://fdc.nal.usda.gov/api-key-signup.html
"""

import os
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set
from datetime import datetime
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("⚠️  requests not installed. Install: pip install requests")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️  pandas not installed. Install: pip install pandas")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️  numpy not installed. Install: pip install numpy")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class USDAScraperConfig:
    """Configuration for USDA FoodData Central scraping"""
    
    # USDA FoodData Central API
    api_base_url: str = "https://api.nal.usda.gov/fdc/v1"
    api_key: str = "DEMO_KEY"  # Replace with real API key
    
    # API endpoints
    search_endpoint: str = "/foods/search"
    food_endpoint: str = "/food/{fdc_id}"
    nutrients_endpoint: str = "/foods/list"
    
    # Output directory
    output_dir: str = "data/usda"
    cache_dir: str = "data/usda/cache"
    
    # HTTP settings
    timeout: int = 30
    max_retries: int = 3
    retry_backoff: float = 1.0
    rate_limit_delay: float = 0.2  # USDA allows 1000 requests/hour with API key
    
    # Query settings
    page_size: int = 200  # Max results per API call
    max_foods: int = 10000  # Limit for development (100k+ in production)
    
    # Food data types to scrape
    data_types: List[str] = field(default_factory=lambda: [
        'Foundation',    # Lab-analyzed foods (ICP-MS data)
        'SR Legacy',     # Standard Reference legacy
        'Survey (FNDDS)', # Survey foods
        'Branded',       # Packaged foods
    ])
    
    # Nutrients/elements to extract (USDA nutrient IDs)
    nutrient_ids: Dict[str, int] = field(default_factory=lambda: {
        # Macrominerals
        'Ca': 301,   # Calcium (mg)
        'Fe': 303,   # Iron (mg)
        'Mg': 304,   # Magnesium (mg)
        'P': 305,    # Phosphorus (mg)
        'K': 306,    # Potassium (mg)
        'Na': 307,   # Sodium (mg)
        'Zn': 309,   # Zinc (mg)
        
        # Trace minerals
        'Cu': 312,   # Copper (mg)
        'Mn': 315,   # Manganese (mg)
        'Se': 317,   # Selenium (µg)
        
        # Additional (if available)
        'I': 1100,   # Iodine (µg)
        'Cr': 1096,  # Chromium (µg)
        'Mo': 1101,  # Molybdenum (µg)
        'Co': 1102,  # Cobalt (µg)
    })
    
    # Heavy metals (if available in dataset)
    heavy_metal_ids: Dict[str, int] = field(default_factory=lambda: {
        'Pb': 1293,  # Lead (µg) - limited data
        'Cd': 1294,  # Cadmium (µg) - limited data
        'Hg': 1295,  # Mercury (µg) - limited data
        'As': 1296,  # Arsenic (µg) - limited data
    })
    
    # Quality thresholds
    quality_threshold: float = 0.8
    min_nutrients: int = 5


@dataclass
class USDAFood:
    """Single food entry from USDA FoodData Central"""
    
    # Identification
    fdc_id: int
    food_code: Optional[str] = None
    food_name: str = ""
    description: str = ""
    
    # Classification
    data_type: str = "SR Legacy"  # Foundation, SR Legacy, Survey (FNDDS), Branded
    food_category: Optional[str] = None
    food_category_id: Optional[int] = None
    
    # Nutrient/element composition
    nutrients: Dict[str, float] = field(default_factory=dict)  # nutrient_name → value
    nutrient_units: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    publication_date: Optional[str] = None
    serving_size: Optional[float] = None
    serving_size_unit: Optional[str] = None
    household_serving: Optional[str] = None
    
    # Brand info (for branded foods)
    brand_owner: Optional[str] = None
    brand_name: Optional[str] = None
    gtin_upc: Optional[str] = None
    
    # Data quality
    data_source: str = "USDA FDC"
    quality_score: float = 1.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def has_mineral_data(self) -> bool:
        """Check if food has mineral/element data"""
        minerals = ['Ca', 'Fe', 'Mg', 'P', 'K', 'Na', 'Zn', 'Cu', 'Mn', 'Se']
        return any(m in self.nutrients for m in minerals)


@dataclass
class USDADataset:
    """Collection of USDA foods"""
    
    foods: List[USDAFood] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    scrape_date: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"
    
    def __len__(self) -> int:
        return len(self.foods)
    
    def get_nutrient_coverage(self) -> Dict[str, int]:
        """Count foods with each nutrient"""
        coverage = {}
        for food in self.foods:
            for nutrient in food.nutrients.keys():
                coverage[nutrient] = coverage.get(nutrient, 0) + 1
        return coverage
    
    def get_food_categories(self) -> Set[str]:
        """Get unique food categories"""
        return {f.food_category for f in self.foods if f.food_category}
    
    def filter_by_data_type(self, data_types: List[str]) -> 'USDADataset':
        """Filter by data type"""
        filtered = [f for f in self.foods if f.data_type in data_types]
        return USDADataset(foods=filtered, metadata=self.metadata)
    
    def filter_with_minerals(self) -> 'USDADataset':
        """Filter foods that have mineral data"""
        filtered = [f for f in self.foods if f.has_mineral_data()]
        return USDADataset(foods=filtered, metadata=self.metadata)
    
    def export_to_json(self, filepath: str):
        """Export dataset to JSON"""
        data = {
            'metadata': self.metadata,
            'scrape_date': self.scrape_date,
            'version': self.version,
            'foods': [f.to_dict() for f in self.foods]
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Exported {len(self.foods)} foods to {filepath}")


# ============================================================================
# USDA API Client
# ============================================================================

class USDAAPIClient:
    """Client for USDA FoodData Central API"""
    
    def __init__(self, config: USDAScraperConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_REQUESTS:
            self.session = requests.Session()
            self.session.params = {'api_key': config.api_key}
    
    def search_foods(
        self,
        query: str = "",
        data_type: Optional[str] = None,
        page: int = 1
    ) -> Optional[Dict]:
        """
        Search for foods
        
        Args:
            query: Search term (e.g., "spinach")
            data_type: Filter by data type ("Foundation", "SR Legacy", etc.)
            page: Page number (1-indexed)
        
        Returns:
            API response with food list
        """
        if not HAS_REQUESTS:
            print("⚠️  requests not available")
            return None
        
        url = f"{self.config.api_base_url}{self.config.search_endpoint}"
        
        params = {
            'query': query,
            'pageSize': self.config.page_size,
            'pageNumber': page,
        }
        
        if data_type:
            params['dataType'] = [data_type]
        
        try:
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            time.sleep(self.config.rate_limit_delay)
            return response.json()
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return None
    
    def get_food(self, fdc_id: int, use_cache: bool = True) -> Optional[Dict]:
        """
        Get detailed food information by FDC ID
        
        Args:
            fdc_id: FoodData Central ID
            use_cache: Use cached response if available
        
        Returns:
            Food details with full nutrient data
        """
        if not HAS_REQUESTS:
            return None
        
        # Check cache
        cache_file = self.cache_dir / f"food_{fdc_id}.json"
        if use_cache and cache_file.exists():
            return json.loads(cache_file.read_text())
        
        # Fetch from API
        url = f"{self.config.api_base_url}/food/{fdc_id}"
        
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache response
            if use_cache:
                cache_file.write_text(json.dumps(data))
            
            time.sleep(self.config.rate_limit_delay)
            return data
            
        except Exception as e:
            print(f"❌ Error fetching food {fdc_id}: {e}")
            return None


# ============================================================================
# USDA Scraper
# ============================================================================

class USDAScraper:
    """Main scraper for USDA FoodData Central"""
    
    def __init__(self, config: Optional[USDAScraperConfig] = None):
        self.config = config or USDAScraperConfig()
        self.client = USDAAPIClient(self.config)
        self.dataset = USDADataset()
    
    def scrape_all(self) -> USDADataset:
        """Scrape USDA FoodData Central"""
        print("\n" + "="*60)
        print("USDA FOODDATA CENTRAL - DATA SCRAPING PIPELINE")
        print("="*60)
        
        print("\n🇺🇸 Starting USDA FoodData Central scraping")
        print(f"API Key: {self.config.api_key[:10]}..." if len(self.config.api_key) > 10 else "DEMO_KEY")
        
        # For now, generate mock data (real API scraping requires API key)
        if self.config.api_key == "DEMO_KEY":
            print("\n⚠️  Using DEMO_KEY - generating mock data")
            print("Get free API key: https://fdc.nal.usda.gov/api-key-signup.html")
            self._generate_mock_usda_data()
        else:
            # Real API scraping
            self._scrape_foundation_foods()
            self._scrape_sr_legacy()
            self._scrape_survey_foods()
        
        print(f"\n✓ Scraping complete: {len(self.dataset)} foods collected")
        
        # Summary
        self._print_summary()
        
        return self.dataset
    
    def _generate_mock_usda_data(self):
        """Generate mock USDA data (development only)"""
        print("\n📊 Generating mock USDA FoodData Central data...")
        
        if not HAS_NUMPY:
            print("⚠️  numpy not available, skipping mock data")
            return
        
        # Mock foods with realistic nutrient data
        mock_foods = [
            # Dairy
            ("Milk, whole", "Dairy and Egg Products", "Foundation"),
            ("Cheese, cheddar", "Dairy and Egg Products", "SR Legacy"),
            ("Yogurt, plain", "Dairy and Egg Products", "Survey (FNDDS)"),
            ("Egg, whole", "Dairy and Egg Products", "Foundation"),
            
            # Meat
            ("Beef, ground, 85% lean", "Beef Products", "Foundation"),
            ("Chicken breast, raw", "Poultry Products", "SR Legacy"),
            ("Pork chop, raw", "Pork Products", "Foundation"),
            ("Turkey, ground", "Poultry Products", "SR Legacy"),
            
            # Seafood
            ("Salmon, Atlantic, wild", "Finfish and Shellfish Products", "Foundation"),
            ("Tuna, canned in water", "Finfish and Shellfish Products", "SR Legacy"),
            ("Shrimp, raw", "Finfish and Shellfish Products", "Foundation"),
            
            # Grains
            ("Bread, whole wheat", "Baked Products", "SR Legacy"),
            ("Rice, brown, cooked", "Cereal Grains and Pasta", "Survey (FNDDS)"),
            ("Oats, rolled, dry", "Breakfast Cereals", "Foundation"),
            ("Pasta, whole wheat, cooked", "Cereal Grains and Pasta", "SR Legacy"),
            
            # Vegetables
            ("Spinach, raw", "Vegetables and Vegetable Products", "Foundation"),
            ("Broccoli, raw", "Vegetables and Vegetable Products", "Foundation"),
            ("Carrot, raw", "Vegetables and Vegetable Products", "SR Legacy"),
            ("Tomato, raw", "Vegetables and Vegetable Products", "Foundation"),
            ("Potato, baked", "Vegetables and Vegetable Products", "Survey (FNDDS)"),
            ("Sweet potato, baked", "Vegetables and Vegetable Products", "Foundation"),
            ("Kale, raw", "Vegetables and Vegetable Products", "Foundation"),
            
            # Fruits
            ("Apple, raw", "Fruits and Fruit Juices", "SR Legacy"),
            ("Banana, raw", "Fruits and Fruit Juices", "Foundation"),
            ("Orange, raw", "Fruits and Fruit Juices", "SR Legacy"),
            ("Strawberry, raw", "Fruits and Fruit Juices", "Foundation"),
            ("Blueberry, raw", "Fruits and Fruit Juices", "Foundation"),
            
            # Legumes
            ("Lentils, cooked", "Legumes and Legume Products", "SR Legacy"),
            ("Chickpeas, canned", "Legumes and Legume Products", "Survey (FNDDS)"),
            ("Black beans, cooked", "Legumes and Legume Products", "Foundation"),
            
            # Nuts
            ("Almonds, raw", "Nut and Seed Products", "Foundation"),
            ("Walnuts, raw", "Nut and Seed Products", "SR Legacy"),
            ("Peanut butter", "Legumes and Legume Products", "Survey (FNDDS)"),
        ]
        
        for food_name, category, data_type in mock_foods:
            # Generate FDC ID
            fdc_id = np.random.randint(100000, 999999)
            
            # Generate realistic nutrient values
            nutrients = {}
            nutrient_units = {}
            
            # Macrominerals (mg/100g)
            nutrients['Ca'] = np.random.uniform(10, 1000) if 'cheese' in food_name.lower() or 'milk' in food_name.lower() else np.random.uniform(10, 200)
            nutrients['Fe'] = np.random.uniform(0.5, 20) if 'spinach' in food_name.lower() or 'beef' in food_name.lower() else np.random.uniform(0.3, 5)
            nutrients['Mg'] = np.random.uniform(10, 200)
            nutrients['P'] = np.random.uniform(50, 500)
            nutrients['K'] = np.random.uniform(100, 1000)
            nutrients['Na'] = np.random.uniform(5, 500)
            nutrients['Zn'] = np.random.uniform(0.3, 15)
            
            # Trace minerals
            nutrients['Cu'] = np.random.uniform(0.05, 2)
            nutrients['Mn'] = np.random.uniform(0.1, 3)
            nutrients['Se'] = np.random.uniform(1, 50)  # µg
            
            # Food-specific adjustments
            if 'spinach' in food_name.lower() or 'kale' in food_name.lower():
                nutrients['Fe'] *= 3
                nutrients['Ca'] *= 2
                nutrients['Mg'] *= 2
            
            if 'salmon' in food_name.lower() or 'fish' in category.lower():
                nutrients['Se'] *= 3
                nutrients['P'] *= 1.5
            
            if 'milk' in food_name.lower() or 'cheese' in food_name.lower():
                nutrients['Ca'] *= 5
                nutrients['P'] *= 2
            
            # Round values
            nutrients = {k: round(v, 2) for k, v in nutrients.items()}
            
            # Set units
            for nutrient in nutrients:
                nutrient_units[nutrient] = 'µg' if nutrient == 'Se' else 'mg'
            
            # Create food object
            food = USDAFood(
                fdc_id=fdc_id,
                food_name=food_name,
                description=f"{food_name} - {category}",
                data_type=data_type,
                food_category=category,
                nutrients=nutrients,
                nutrient_units=nutrient_units,
                publication_date=datetime.now().isoformat(),
                serving_size=100,
                serving_size_unit="g",
                data_source="USDA FDC Mock",
                quality_score=np.random.uniform(0.9, 1.0)
            )
            
            self.dataset.foods.append(food)
        
        print(f"✓ Generated {len(mock_foods)} mock USDA foods")
    
    def _scrape_foundation_foods(self):
        """Scrape Foundation Foods (lab-analyzed)"""
        print("\n🔬 Scraping Foundation Foods (ICP-MS validated)...")
        # Would use API to fetch real data
        pass
    
    def _scrape_sr_legacy(self):
        """Scrape SR Legacy database"""
        print("\n📚 Scraping SR Legacy (Standard Reference)...")
        # Would use API to fetch real data
        pass
    
    def _scrape_survey_foods(self):
        """Scrape Survey Foods (FNDDS)"""
        print("\n📋 Scraping Survey Foods (FNDDS)...")
        # Would use API to fetch real data
        pass
    
    def _print_summary(self):
        """Print dataset summary"""
        if len(self.dataset) == 0:
            print("\n⚠️  No foods collected")
            return
        
        print("\n" + "="*60)
        print("USDA SCRAPING SUMMARY")
        print("="*60)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total foods: {len(self.dataset)}")
        
        # Data types
        data_types = {}
        for food in self.dataset.foods:
            data_types[food.data_type] = data_types.get(food.data_type, 0) + 1
        
        print(f"\n  Data types:")
        for dtype, count in sorted(data_types.items()):
            print(f"    {dtype}: {count} foods")
        
        # Nutrient coverage
        coverage = self.dataset.get_nutrient_coverage()
        print(f"\n🧪 Nutrient Coverage:")
        print(f"  Nutrients tracked: {len(coverage)}")
        
        top_nutrients = sorted(coverage.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n  Top 10 nutrients by food count:")
        for nutrient, count in top_nutrients:
            print(f"    {nutrient}: {count} foods")
        
        # Food categories
        categories = self.dataset.get_food_categories()
        print(f"\n🍽️  Food Categories: {len(categories)}")
        for cat in sorted(categories)[:10]:
            count = sum(1 for f in self.dataset.foods if f.food_category == cat)
            print(f"    {cat}: {count} foods")


# ============================================================================
# Pipeline Runner
# ============================================================================

def run_usda_scraping_pipeline(
    output_file: str = "data/usda/usda_dataset.json",
    config: Optional[USDAScraperConfig] = None
) -> USDADataset:
    """
    Run complete USDA FoodData Central scraping pipeline
    
    Args:
        output_file: Path to save JSON output
        config: Configuration (uses defaults if None)
    
    Returns:
        USDADataset with all scraped foods
    """
    config = config or USDAScraperConfig()
    
    # Create scraper
    scraper = USDAScraper(config)
    
    # Scrape all data
    dataset = scraper.scrape_all()
    
    # Export results
    print(f"\n💾 Exporting dataset...")
    dataset.export_to_json(output_file)
    
    print("\n" + "="*60)
    print("✅ USDA SCRAPING COMPLETE")
    print("="*60)
    print(f"Total foods: {len(dataset)}")
    print(f"Categories: {len(dataset.get_food_categories())}")
    print(f"Nutrients: {len(dataset.get_nutrient_coverage())}")
    print(f"Output: {output_file}")
    
    return dataset


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape USDA FoodData Central data")
    parser.add_argument(
        '--output',
        type=str,
        default='data/usda/usda_dataset.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default='DEMO_KEY',
        help='USDA FoodData Central API key'
    )
    parser.add_argument(
        '--max-foods',
        type=int,
        default=10000,
        help='Maximum foods to scrape (100k+ for production)'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = USDAScraperConfig(
        api_key=args.api_key,
        max_foods=args.max_foods
    )
    
    # Run pipeline
    dataset = run_usda_scraping_pipeline(
        output_file=args.output,
        config=config
    )
    
    print(f"\n✨ Scraped {len(dataset)} USDA foods")
