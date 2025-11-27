"""
AI Grocery & Raw Food Recommendation System
===========================================

Integrates ICP-MS analysis with disease nutrition recommendations
to provide personalized grocery shopping lists
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from food_nutrient_detector import FoodNutrientDetector, NutrientProfile
from llm_hybrid_disease_db import LLMHybridDiseaseDatabase
import json


@dataclass
class GroceryItem:
    """Recommended grocery item"""
    food_name: str
    category: str
    quantity: str
    priority: str
    compatibility_score: float
    key_nutrients: List[str]
    price_estimate: Optional[float]
    where_to_buy: List[str]
    preparation_tips: List[str]
    safety_verified: bool


@dataclass
class GroceryList:
    """Complete personalized grocery list"""
    for_person: str
    diseases: List[str]
    items: List[GroceryItem]
    total_estimated_cost: float
    nutritional_coverage: Dict[str, float]
    safety_warnings: List[str]


class GroceryRecommendationEngine:
    """AI-powered grocery recommendation system"""
    
    def __init__(self, use_llm: bool = False):
        """Initialize recommendation engine"""
        self.food_detector = FoodNutrientDetector()
        self.disease_db = LLMHybridDiseaseDatabase(use_llm=use_llm)
        self.raw_food_catalog = self._load_raw_food_catalog()
    
    def _load_raw_food_catalog(self) -> Dict:
        """Load comprehensive raw food catalog with ICP-MS data"""
        return {
            'vegetables': {
                'spinach': {
                    'icpms_data': {'Ca': 990, 'Fe': 27, 'Mg': 790, 'K': 5580, 'Zn': 5},
                    'price_per_100g': 0.50,
                    'stores': ['Walmart', 'Whole Foods', 'Local Market'],
                    'prep_tips': ['Wash thoroughly', 'Steam for 2-3 minutes', 'Add to salads']
                },
                'broccoli': {
                    'icpms_data': {'Ca': 470, 'Fe': 7, 'Mg': 210, 'K': 3160, 'Zn': 4},
                    'price_per_100g': 0.40,
                    'stores': ['Walmart', 'Whole Foods', 'Costco'],
                    'prep_tips': ['Steam for 5 minutes', 'Roast at 425°F', 'Add garlic']
                },
                'kale': {
                    'icpms_data': {'Ca': 1500, 'Fe': 15, 'Mg': 470, 'K': 4470, 'Zn': 6},
                    'price_per_100g': 0.60,
                    'stores': ['Whole Foods', 'Trader Joes', 'Local Market'],
                    'prep_tips': ['Remove stems', 'Massage with olive oil', 'Bake for chips']
                },
            },
            'proteins': {
                'salmon': {
                    'icpms_data': {'Ca': 120, 'Fe': 8, 'Se': 36.5, 'Zn': 6},
                    'price_per_100g': 3.50,
                    'stores': ['Whole Foods', 'Costco', 'Fish Market'],
                    'prep_tips': ['Grill for 12-15 min', 'Bake at 375°F', 'Season with lemon']
                },
                'chicken_breast': {
                    'icpms_data': {'Ca': 150, 'Fe': 10, 'Zn': 10, 'Se': 27},
                    'price_per_100g': 1.50,
                    'stores': ['Walmart', 'Costco', 'Local Butcher'],
                    'prep_tips': ['Marinate 30 min', 'Grill or bake', 'Check internal temp 165°F']
                },
            },
            'grains': {
                'quinoa': {
                    'icpms_data': {'Ca': 470, 'Fe': 46, 'Mg': 1970, 'K': 5630, 'Zn': 31},
                    'price_per_100g': 0.80,
                    'stores': ['Whole Foods', 'Trader Joes', 'Costco'],
                    'prep_tips': ['Rinse before cooking', 'Cook 15 min', 'Fluff with fork']
                },
                'brown_rice': {
                    'icpms_data': {'Ca': 230, 'Fe': 8, 'Mg': 1430, 'K': 2230, 'Zn': 14},
                    'price_per_100g': 0.30,
                    'stores': ['Walmart', 'Costco', 'Asian Market'],
                    'prep_tips': ['Soak 30 min', 'Cook 45 min', 'Let steam 10 min']
                },
            },
            'fruits': {
                'blueberries': {
                    'icpms_data': {'Ca': 60, 'Fe': 3, 'Mg': 60, 'K': 770, 'Zn': 2},
                    'price_per_100g': 1.20,
                    'stores': ['Whole Foods', 'Costco', 'Farmers Market'],
                    'prep_tips': ['Rinse gently', 'Freeze for smoothies', 'Add to oatmeal']
                },
                'avocado': {
                    'icpms_data': {'Ca': 120, 'Fe': 6, 'Mg': 290, 'K': 4850, 'Zn': 6},
                    'price_per_100g': 0.80,
                    'stores': ['Walmart', 'Whole Foods', 'Mexican Market'],
                    'prep_tips': ['Check ripeness', 'Slice and remove pit', 'Add lime juice']
                },
            },
            'nuts_seeds': {
                'almonds': {
                    'icpms_data': {'Ca': 2690, 'Fe': 37, 'Mg': 2700, 'K': 7330, 'Zn': 31},
                    'price_per_100g': 1.50,
                    'stores': ['Costco', 'Whole Foods', 'Trader Joes'],
                    'prep_tips': ['Soak overnight', 'Roast at 350°F', 'Store in airtight container']
                },
                'chia_seeds': {
                    'icpms_data': {'Ca': 6310, 'Fe': 77, 'Mg': 3350, 'K': 4070, 'Zn': 46},
                    'price_per_100g': 2.00,
                    'stores': ['Whole Foods', 'Trader Joes', 'Health Food Store'],
                    'prep_tips': ['Make chia pudding', 'Add to smoothies', 'Sprinkle on yogurt']
                },
            },
        }
    
    def generate_grocery_list(self,
                             person_name: str,
                             diseases: List[str],
                             dietary_preference: str = 'omnivore',
                             budget: Optional[float] = None,
                             verify_icpms: bool = True) -> GroceryList:
        """
        Generate personalized grocery list based on diseases
        
        Args:
            person_name: Person's name
            diseases: List of disease IDs
            dietary_preference: omnivore, vegetarian, vegan, etc.
            budget: Optional budget limit
            verify_icpms: Whether to verify with ICP-MS data
        
        Returns:
            Complete GroceryList
        """
        recommended_items = []
        total_cost = 0.0
        safety_warnings = []
        nutritional_coverage = {}
        
        # Get disease requirements
        all_nutrients_needed = {}
        all_restrictions = set()
        
        for disease_id in diseases:
            disease = self.disease_db.get_disease(disease_id)
            if not disease:
                continue
            
            # Collect nutrient requirements
            for guideline in disease.nutritional_guidelines:
                nutrient = guideline.nutrient
                if nutrient not in all_nutrients_needed:
                    all_nutrients_needed[nutrient] = {
                        'target': guideline.target,
                        'priority': guideline.priority,
                        'unit': guideline.unit
                    }
            
            # Collect restrictions
            for restriction in disease.food_restrictions:
                all_restrictions.add(restriction.food_item.lower())
        
        # Scan raw food catalog
        for category, foods in self.raw_food_catalog.items():
            # Skip meat if vegetarian/vegan
            if dietary_preference in ['vegetarian', 'vegan'] and category == 'proteins':
                continue
            
            for food_name, food_data in foods.items():
                # Check if restricted
                if food_name.lower() in all_restrictions:
                    continue
                
                # Analyze food with ICP-MS
                icpms_data = food_data.get('icpms_data', {})
                food_profile = self.food_detector.analyze_food(food_name, icpms_data=icpms_data)
                
                # Check safety
                if verify_icpms and not food_profile.is_safe_for_consumption:
                    safety_warnings.append(f"{food_name}: Heavy metal contamination detected")
                    continue
                
                # Calculate compatibility with all diseases
                total_compatibility = 0
                compatible_nutrients = []
                
                for disease_id in diseases:
                    compat = self.food_detector.compare_with_disease_requirements(
                        food_profile, disease_id, self.disease_db
                    )
                    total_compatibility += compat['compatibility_score']
                    
                    for nutrient in compat['compatible_nutrients']:
                        if nutrient['nutrient'] not in compatible_nutrients:
                            compatible_nutrients.append(nutrient['nutrient'])
                
                avg_compatibility = total_compatibility / len(diseases) if diseases else 0
                
                # Only add if reasonably compatible
                if avg_compatibility >= 50:
                    # Determine quantity and priority
                    priority = 'high' if avg_compatibility >= 80 else 'medium' if avg_compatibility >= 65 else 'low'
                    quantity = self._determine_quantity(category, priority)
                    
                    # Calculate cost
                    item_cost = food_data['price_per_100g'] * self._quantity_to_grams(quantity) / 100
                    
                    # Check budget
                    if budget and (total_cost + item_cost) > budget:
                        continue
                    
                    recommended_items.append(GroceryItem(
                        food_name=food_name.replace('_', ' ').title(),
                        category=category,
                        quantity=quantity,
                        priority=priority,
                        compatibility_score=avg_compatibility,
                        key_nutrients=compatible_nutrients[:5],
                        price_estimate=item_cost,
                        where_to_buy=food_data['stores'],
                        preparation_tips=food_data['prep_tips'],
                        safety_verified=True
                    ))
                    
                    total_cost += item_cost
                    
                    # Update nutritional coverage
                    for nutrient in compatible_nutrients:
                        if nutrient not in nutritional_coverage:
                            nutritional_coverage[nutrient] = 0
                        nutritional_coverage[nutrient] += avg_compatibility / 100
        
        # Sort by priority and compatibility
        recommended_items.sort(key=lambda x: (
            {'high': 3, 'medium': 2, 'low': 1}[x.priority],
            x.compatibility_score
        ), reverse=True)
        
        return GroceryList(
            for_person=person_name,
            diseases=[self.disease_db.get_disease(d).name for d in diseases if self.disease_db.get_disease(d)],
            items=recommended_items,
            total_estimated_cost=total_cost,
            nutritional_coverage=nutritional_coverage,
            safety_warnings=safety_warnings
        )
    
    def _determine_quantity(self, category: str, priority: str) -> str:
        """Determine recommended quantity"""
        base_quantities = {
            'vegetables': {'high': '500g', 'medium': '300g', 'low': '200g'},
            'proteins': {'high': '400g', 'medium': '300g', 'low': '200g'},
            'grains': {'high': '500g', 'medium': '300g', 'low': '200g'},
            'fruits': {'high': '300g', 'medium': '200g', 'low': '150g'},
            'nuts_seeds': {'high': '150g', 'medium': '100g', 'low': '75g'},
        }
        return base_quantities.get(category, {}).get(priority, '200g')
    
    def _quantity_to_grams(self, quantity: str) -> float:
        """Convert quantity string to grams"""
        return float(quantity.replace('g', ''))
    
    def export_shopping_list(self, grocery_list: GroceryList, format: str = 'text') -> str:
        """
        Export grocery list in various formats
        
        Args:
            grocery_list: GroceryList object
            format: 'text', 'json', 'markdown'
        
        Returns:
            Formatted string
        """
        if format == 'json':
            return json.dumps(asdict(grocery_list), indent=2)
        
        elif format == 'markdown':
            output = f"# Grocery Shopping List for {grocery_list.for_person}\n\n"
            output += f"**Health Conditions:** {', '.join(grocery_list.diseases)}\n"
            output += f"**Total Estimated Cost:** ${grocery_list.total_estimated_cost:.2f}\n\n"
            
            output += "## Shopping List\n\n"
            
            current_category = None
            for item in grocery_list.items:
                if item.category != current_category:
                    current_category = item.category
                    output += f"\n### {current_category.replace('_', ' ').title()}\n\n"
                
                output += f"- **{item.food_name}** ({item.quantity})\n"
                output += f"  - Priority: {item.priority.upper()}\n"
                output += f"  - Compatibility: {item.compatibility_score:.0f}/100\n"
                output += f"  - Price: ${item.price_estimate:.2f}\n"
                output += f"  - Where: {', '.join(item.where_to_buy)}\n"
                output += f"  - Key nutrients: {', '.join(item.key_nutrients)}\n\n"
            
            if grocery_list.safety_warnings:
                output += "\n## ⚠️ Safety Warnings\n\n"
                for warning in grocery_list.safety_warnings:
                    output += f"- {warning}\n"
            
            return output
        
        else:  # text format
            output = f"=" * 100 + "\n"
            output += f"PERSONALIZED GROCERY SHOPPING LIST\n"
            output += f"=" * 100 + "\n\n"
            output += f"For: {grocery_list.for_person}\n"
            output += f"Health Conditions: {', '.join(grocery_list.diseases)}\n"
            output += f"Total Items: {len(grocery_list.items)}\n"
            output += f"Estimated Cost: ${grocery_list.total_estimated_cost:.2f}\n\n"
            
            output += f"=" * 100 + "\n"
            output += f"SHOPPING LIST (Sorted by Priority)\n"
            output += f"=" * 100 + "\n\n"
            
            current_category = None
            for item in grocery_list.items:
                if item.category != current_category:
                    current_category = item.category
                    output += f"\n{current_category.replace('_', ' ').upper()}\n"
                    output += "-" * 100 + "\n"
                
                priority_symbol = "🔴" if item.priority == 'high' else "🟡" if item.priority == 'medium' else "🟢"
                output += f"\n{priority_symbol} {item.food_name} - {item.quantity} (${item.price_estimate:.2f})\n"
                output += f"   Compatibility: {item.compatibility_score:.0f}/100\n"
                output += f"   Key nutrients: {', '.join(item.key_nutrients)}\n"
                output += f"   Where to buy: {', '.join(item.where_to_buy)}\n"
                output += f"   Preparation:\n"
                for tip in item.preparation_tips:
                    output += f"     • {tip}\n"
            
            if grocery_list.safety_warnings:
                output += f"\n\n{'=' * 100}\n"
                output += f"⚠️ SAFETY WARNINGS\n"
                output += f"{'=' * 100}\n\n"
                for warning in grocery_list.safety_warnings:
                    output += f"  • {warning}\n"
            
            output += f"\n\n{'=' * 100}\n"
            output += f"NUTRITIONAL COVERAGE\n"
            output += f"{'=' * 100}\n\n"
            for nutrient, coverage in sorted(grocery_list.nutritional_coverage.items()):
                bars = int(coverage * 20)
                output += f"  {nutrient:20s} {'█' * bars}{'░' * (20-bars)} {coverage*100:.0f}%\n"
            
            return output


def test_grocery_recommendation():
    """Test the grocery recommendation system"""
    
    print("\n" + "="*100)
    print("TESTING AI GROCERY & RAW FOOD RECOMMENDATION SYSTEM")
    print("="*100)
    
    engine = GroceryRecommendationEngine(use_llm=False)
    
    # Test 1: Generate grocery list for diabetes + hypertension
    print("\nTEST 1: Generate Grocery List (Diabetes + Hypertension)")
    print("-" * 100)
    
    grocery_list = engine.generate_grocery_list(
        person_name="John",
        diseases=['diabetes_type2', 'hypertension'],
        dietary_preference='omnivore',
        budget=50.0,
        verify_icpms=True
    )
    
    print(f"\nGenerated {len(grocery_list.items)} items")
    print(f"Total cost: ${grocery_list.total_estimated_cost:.2f}")
    print(f"Diseases: {', '.join(grocery_list.diseases)}")
    
    print(f"\nTop 5 Recommendations:")
    for item in grocery_list.items[:5]:
        print(f"\n  {item.priority.upper()} PRIORITY: {item.food_name}")
        print(f"    Quantity: {item.quantity}")
        print(f"    Compatibility: {item.compatibility_score:.0f}/100")
        print(f"    Price: ${item.price_estimate:.2f}")
        print(f"    Key nutrients: {', '.join(item.key_nutrients)}")
        print(f"    Where: {', '.join(item.where_to_buy[:2])}")
    
    # Test 2: Export shopping list
    print("\n\nTEST 2: Export Shopping List (Text Format)")
    print("-" * 100)
    
    shopping_list_text = engine.export_shopping_list(grocery_list, format='text')
    print(shopping_list_text)
    
    # Save to file
    with open('grocery_list_john.txt', 'w') as f:
        f.write(shopping_list_text)
    print("\n✅ Saved to: grocery_list_john.txt")
    
    # Test 3: Generate markdown format
    print("\n\nTEST 3: Export as Markdown")
    print("-" * 100)
    
    markdown = engine.export_shopping_list(grocery_list, format='markdown')
    with open('grocery_list_john.md', 'w') as f:
        f.write(markdown)
    print("✅ Saved to: grocery_list_john.md")
    print(markdown[:500] + "...")
    
    # Test 4: Vegetarian list for osteoporosis
    print("\n\nTEST 4: Vegetarian List (Osteoporosis)")
    print("-" * 100)
    
    veg_list = engine.generate_grocery_list(
        person_name="Mary",
        diseases=['osteoporosis'],
        dietary_preference='vegetarian',
        budget=40.0
    )
    
    print(f"\nGenerated {len(veg_list.items)} vegetarian items")
    print(f"Total cost: ${veg_list.total_estimated_cost:.2f}")
    print(f"\nHigh Priority Items:")
    for item in [i for i in veg_list.items if i.priority == 'high']:
        print(f"  • {item.food_name} ({item.quantity}) - ${item.price_estimate:.2f}")
    
    print("\n" + "="*100)
    print("TESTING COMPLETE")
    print("="*100)
    
    print("\n📊 SYSTEM CAPABILITIES:")
    print("  ✅ ICP-MS food analysis")
    print("  ✅ Nutrient detection & quantification")
    print("  ✅ Heavy metal contamination screening")
    print("  ✅ Disease-specific recommendations")
    print("  ✅ Personalized grocery lists")
    print("  ✅ Budget-aware shopping")
    print("  ✅ Dietary preference support")
    print("  ✅ Multi-format export (text, JSON, markdown)")
    print("  ✅ Store recommendations")
    print("  ✅ Preparation guidance")


if __name__ == "__main__":
    test_grocery_recommendation()
