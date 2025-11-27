"""
Enhanced Food Database with API Integration
Manages 10,000+ foods from multiple sources including APIs
"""

from typing import Dict, List, Optional, Set
from decimal import Decimal
import json
import pickle
from pathlib import Path

from ..models.core_data_models import FoodItem, FoodCategory
from ..integrations.food_api_client import FoodAPIClient, APICredentials, FoodAPIProvider  # type: ignore


class EnhancedFoodDatabase:
    """
    Enhanced food database with API integration
    Combines manual entries with API-sourced foods for 10,000+ coverage
    """
    
    def __init__(self, api_client: Optional[FoodAPIClient] = None):
        self.foods: Dict[str, FoodItem] = {}
        self.api_client = api_client or FoodAPIClient()
        self.cache_file = Path(__file__).parent.parent / "data" / "food_cache.pkl"
        self.manual_entries_count = 0
        self.api_entries_count = 0
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with manual entries and cached API data"""
        # Load manual entries first (our curated high-quality entries)
        self._add_manual_entries()
        self.manual_entries_count = len(self.foods)
        
        # Load cached API data if available
        if self.cache_file.exists():
            self._load_cache()
        
        print(f"Food Database initialized:")
        print(f"  - Manual entries: {self.manual_entries_count}")
        print(f"  - API entries: {self.api_entries_count}")
        print(f"  - Total foods: {len(self.foods)}")
    
    def _add_manual_entries(self):
        """Add manually curated food entries (from food_database.py)"""
        # Note: Original food_database.py not found, starting with empty database
        # You can add manual entries here if needed
        pass
    
    # ==================== API Integration Methods ====================
    
    def populate_from_usda(self, max_foods: int = 5000):
        """
        Populate database from USDA FoodData Central
        Target: 5,000 foods covering all major categories
        """
        print(f"Populating from USDA (target: {max_foods} foods)...")
        
        # Categories to import
        categories = [
            "fruits", "vegetables", "grains", "legumes", "nuts",
            "dairy", "meat", "fish", "oils", "spices"
        ]
        
        foods_per_category = max_foods // len(categories)
        imported_foods = self.api_client.bulk_import_foods(categories, max_per_category=foods_per_category)
        
        # Add to database
        for food in imported_foods:
            if food.food_id not in self.foods:
                self.foods[food.food_id] = food
                self.api_entries_count += 1
        
        print(f"Added {len(imported_foods)} foods from USDA")
        self._save_cache()
    
    def populate_from_openfoodfacts(self, max_foods: int = 3000):
        """
        Populate database from Open Food Facts
        Target: 3,000 packaged/branded foods from global regions
        """
        print(f"Populating from Open Food Facts (target: {max_foods} foods)...")
        
        # Search queries for diverse food coverage
        queries = [
            # Staples
            "bread", "rice", "pasta", "cereal", "flour",
            # Packaged foods
            "yogurt", "cheese", "milk", "butter", "cream",
            # Snacks
            "chips", "crackers", "cookies", "candy", "chocolate",
            # Beverages
            "juice", "soda", "tea", "coffee", "water",
            # Condiments
            "sauce", "ketchup", "mustard", "mayo", "dressing",
            # International
            "curry", "salsa", "kimchi", "miso", "hummus",
        ]
        
        imported_count = 0
        
        for query in queries:
            if imported_count >= max_foods:
                break
            
            results = self.api_client.search_openfoodfacts(query)
            
            for result in results[:50]:  # Top 50 per query
                if imported_count >= max_foods:
                    break
                
                food_item = self.api_client.convert_openfoodfacts_to_fooditem(result)
                if food_item and food_item.food_id not in self.foods:
                    self.foods[food_item.food_id] = food_item
                    self.api_entries_count += 1
                    imported_count += 1
        
        print(f"Added {imported_count} foods from Open Food Facts")
        self._save_cache()
    
    def populate_regional_foods(self, regions: Optional[List[str]] = None):
        """
        Populate database with regional foods
        Regions: usa, europe, asia, africa, latin_america, middle_east, oceania
        """
        if regions is None:
            regions = ["usa", "europe", "asia", "africa", "latin_america", "middle_east", "oceania"]
        
        print(f"Populating regional foods: {', '.join(regions)}")
        
        for region in regions:
            regional_foods = self.api_client.get_regional_foods(region)
            
            for food in regional_foods:
                if food.food_id not in self.foods:
                    self.foods[food.food_id] = food
                    self.api_entries_count += 1
            
            print(f"  - Added {len(regional_foods)} foods from {region}")
        
        self._save_cache()
    
    def populate_all_sources(self, target_total: int = 10000):
        """
        Populate database from all sources to reach target
        Default target: 10,000+ foods
        """
        print(f"\n{'='*60}")
        print(f"POPULATING FOOD DATABASE - Target: {target_total:,} foods")
        print(f"{'='*60}\n")
        
        current_count = len(self.foods)
        remaining = target_total - current_count
        
        if remaining <= 0:
            print(f"Target already reached! Current: {current_count:,} foods")
            return
        
        # Distribution strategy:
        # 50% USDA (comprehensive US data)
        # 30% Open Food Facts (global packaged foods)
        # 20% Regional specialties
        
        usda_target = int(remaining * 0.5)
        off_target = int(remaining * 0.3)
        
        # Populate from USDA
        if self.api_client.credentials.usda_api_key:
            self.populate_from_usda(max_foods=usda_target)
        else:
            print("⚠️  USDA API key not configured - skipping USDA import")
        
        # Populate from Open Food Facts (no API key needed)
        self.populate_from_openfoodfacts(max_foods=off_target)
        
        # Populate regional foods
        self.populate_regional_foods()
        
        final_count = len(self.foods)
        print(f"\n{'='*60}")
        print(f"DATABASE POPULATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total foods: {final_count:,}")
        print(f"  - Manual entries: {self.manual_entries_count:,}")
        print(f"  - API entries: {self.api_entries_count:,}")
        print(f"Target achievement: {(final_count/target_total*100):.1f}%")
        print(f"{'='*60}\n")
    
    # ==================== Search Methods ====================
    
    def search_by_name(self, query: str, max_results: int = 20) -> List[FoodItem]:
        """Search foods by name"""
        query_lower = query.lower()
        results = [
            food for food in self.foods.values()
            if query_lower in food.name.lower()
        ]
        return results[:max_results]
    
    def search_by_category(self, category: FoodCategory, limit: Optional[int] = None) -> List[FoodItem]:
        """Get foods by category"""
        results = [food for food in self.foods.values() if food.category == category]
        return results[:limit] if limit else results
    
    def search_by_region(self, region: str) -> List[FoodItem]:
        """Search foods by region/country of origin"""
        region_lower = region.lower()
        results = [
            food for food in self.foods.values()
            if any(region_lower in country.lower() for country in food.countries_of_origin)
        ]
        return results
    
    def search_by_nutrient(self, nutrient_id: str, min_amount: Decimal, max_results: int = 50) -> List[FoodItem]:
        """Find foods high in specific nutrient"""
        results = []
        
        for food in self.foods.values():
            for nutrient in food.nutrient_content:
                if nutrient.nutrient_id == nutrient_id and nutrient.amount >= min_amount:
                    results.append((food, nutrient.amount))
                    break
        
        # Sort by nutrient amount (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return [food for food, _ in results[:max_results]]
    
    def get_allergen_free_foods(self, allergens: Set[str]) -> List[FoodItem]:
        """Get foods free of specified allergens"""
        return [
            food for food in self.foods.values()
            if not food.allergens.intersection(allergens)
        ]
    
    def get_low_glycemic_foods(self, max_gi: int = 55) -> List[FoodItem]:
        """Get low glycemic index foods"""
        return [
            food for food in self.foods.values()
            if food.glycemic_index and food.glycemic_index <= max_gi
        ]
    
    def get_high_protein_foods(self, min_protein: Decimal = Decimal("10")) -> List[FoodItem]:
        """Get high protein foods"""
        return [
            food for food in self.foods.values()
            if food.macronutrients.get("protein", Decimal("0")) >= min_protein
        ]
    
    # ==================== Statistics ====================
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        stats = {
            "total_foods": len(self.foods),
            "manual_entries": self.manual_entries_count,
            "api_entries": self.api_entries_count,
            "by_category": {},
            "by_source": {},
            "coverage": {},
        }
        
        # Count by category
        for food in self.foods.values():
            category_name = food.category.name
            stats["by_category"][category_name] = stats["by_category"].get(category_name, 0) + 1
        
        # Count by source
        for food in self.foods.values():
            source = "manual" if not food.food_id.startswith(("usda_", "off_", "edamam_")) else food.food_id.split("_")[0]
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
        
        # Coverage metrics
        stats["coverage"]["has_glycemic_index"] = sum(1 for f in self.foods.values() if f.glycemic_index)
        stats["coverage"]["has_orac"] = sum(1 for f in self.foods.values() if f.orac_value)
        stats["coverage"]["organic_available"] = sum(1 for f in self.foods.values() if f.organic_available)
        
        return stats
    
    def print_statistics(self):
        """Print formatted database statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("FOOD DATABASE STATISTICS")
        print("="*60)
        print(f"\nTotal Foods: {stats['total_foods']:,}")
        print(f"  - Manual entries: {stats['manual_entries']:,}")
        print(f"  - API entries: {stats['api_entries']:,}")
        
        print("\nBy Category:")
        for category, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {category}: {count:,}")
        
        print("\nBy Source:")
        for source, count in sorted(stats['by_source'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {source}: {count:,}")
        
        print("\nData Coverage:")
        for metric, count in stats['coverage'].items():
            percentage = (count / stats['total_foods'] * 100) if stats['total_foods'] > 0 else 0
            print(f"  - {metric}: {count:,} ({percentage:.1f}%)")
        
        print("="*60 + "\n")
    
    # ==================== Cache Management ====================
    
    def _save_cache(self):
        """Save database to cache file"""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            "foods": self.foods,
            "manual_entries_count": self.manual_entries_count,
            "api_entries_count": self.api_entries_count,
        }
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Cache saved: {len(self.foods):,} foods")
    
    def _load_cache(self):
        """Load database from cache file"""
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.foods = cache_data.get("foods", {})
            self.manual_entries_count = cache_data.get("manual_entries_count", 0)
            self.api_entries_count = cache_data.get("api_entries_count", 0)
            
            print(f"Cache loaded: {len(self.foods):,} foods")
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    def clear_cache(self):
        """Clear cached API data (keeps manual entries)"""
        if self.cache_file.exists():
            self.cache_file.unlink()
        
        # Reload only manual entries
        self.foods = {}
        self.api_entries_count = 0
        self._add_manual_entries()
        print("Cache cleared. Manual entries retained.")
    
    # ==================== Export Methods ====================
    
    def export_to_json(self, filepath: Path, include_api: bool = True):
        """Export database to JSON file"""
        foods_to_export = self.foods if include_api else {
            k: v for k, v in self.foods.items() 
            if not k.startswith(("usda_", "off_", "edamam_"))
        }
        
        # Convert to serializable format
        export_data = []
        for food in foods_to_export.values():
            food_dict = {
                "food_id": food.food_id,
                "name": food.name,
                "category": food.category.name,
                "subcategory": food.subcategory,
                "calories_per_100g": float(food.calories_per_100g),
                "macronutrients": {k: float(v) for k, v in food.macronutrients.items()},
                "nutrient_content": [
                    {
                        "nutrient_id": n.nutrient_id,
                        "amount": float(n.amount),
                        "unit": n.unit.name
                    }
                    for n in food.nutrient_content
                ],
                "allergens": list(food.allergens),
                "countries_of_origin": food.countries_of_origin,
            }
            export_data.append(food_dict)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(export_data):,} foods to {filepath}")


# ==================== CLI Interface ====================

def main():
    """Command-line interface for database management"""
    import sys
    
    # Initialize database
    print("Initializing Enhanced Food Database...")
    
    # Load credentials from environment
    credentials = APICredentials()
    api_client = FoodAPIClient(credentials)
    db = EnhancedFoodDatabase(api_client)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "populate":
            target = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
            db.populate_all_sources(target_total=target)
        
        elif command == "stats":
            db.print_statistics()
        
        elif command == "search":
            if len(sys.argv) < 3:
                print("Usage: python enhanced_food_database.py search <query>")
                return
            query = " ".join(sys.argv[2:])
            results = db.search_by_name(query)
            print(f"\nFound {len(results)} results for '{query}':")
            for food in results[:10]:
                print(f"  - {food.name} ({food.category.name})")
        
        elif command == "export":
            filepath = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("food_database_export.json")
            db.export_to_json(filepath)
        
        elif command == "clear":
            db.clear_cache()
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: populate, stats, search, export, clear")
    
    else:
        # Interactive mode
        db.print_statistics()
        print("\nCommands:")
        print("  populate [target]  - Populate database from APIs")
        print("  stats             - Show database statistics")
        print("  search <query>    - Search for foods")
        print("  export [file]     - Export to JSON")
        print("  clear             - Clear API cache")


if __name__ == "__main__":
    main()
