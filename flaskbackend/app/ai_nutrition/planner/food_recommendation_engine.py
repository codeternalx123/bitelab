"""
Food Recommendation Engine with Avoidance Checking
Recommends foods based on user profile and health conditions.
Checks for allergens, intolerances, and medical contraindications.

Part of Intelligent Meal Planner System
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum


class AvoidanceReason(Enum):
    """Reasons to avoid foods"""
    ALLERGY = "allergy"  # Life-threatening
    INTOLERANCE = "intolerance"  # Digestive issues
    MEDICAL_CONDITION = "medical_condition"  # Disease contraindication
    MEDICATION_INTERACTION = "medication_interaction"  # Drug-food interaction
    RELIGIOUS = "religious"  # Religious restriction
    DIETARY_PREFERENCE = "dietary_preference"  # Personal choice
    NUTRIENT_CONCERN = "nutrient_concern"  # Too high in problematic nutrient


@dataclass
class FoodAvoidance:
    """Specific food or nutrient to avoid"""
    item: str  # Food name or nutrient ID
    reason: AvoidanceReason
    severity: int  # 1=critical (allergy), 2=important, 3=preference
    explanation: str
    alternatives: List[str] = field(default_factory=list)


@dataclass
class FoodRecommendation:
    """Recommended food with reasoning"""
    food_id: str
    food_name: str
    category: str
    portion_size: Decimal
    portion_unit: str
    
    # Why recommended
    benefits: List[str]
    nutrient_highlights: Dict[str, Decimal]  # Key nutrients it provides
    health_score: float  # 0-100
    
    # Preparation suggestions
    preparation_methods: List[str]
    meal_timing: List[str]  # breakfast, lunch, dinner, snack
    frequency_recommendation: str  # daily, 3x/week, weekly, occasionally


class ConditionFoodDatabase:
    """
    Database of food recommendations and restrictions by health condition.
    In production, this would be loaded from a comprehensive database.
    """
    
    # Foods to AVOID by condition
    CONDITION_AVOID = {
        "diabetes": {
            "high_gi_foods": {
                "foods": ["white bread", "white rice", "potato", "corn flakes", "sugary drinks"],
                "reason": "Rapid blood sugar spikes",
                "severity": 2
            },
            "added_sugars": {
                "foods": ["candy", "cookies", "cake", "ice cream", "soda"],
                "reason": "Direct blood sugar elevation",
                "severity": 1
            },
            "refined_carbs": {
                "foods": ["pastries", "white pasta", "crackers"],
                "reason": "Poor glycemic control",
                "severity": 2
            }
        },
        
        "hypertension": {
            "high_sodium": {
                "foods": ["processed meats", "canned soups", "salty snacks", "fast food", "pickles"],
                "reason": "Increases blood pressure",
                "severity": 1
            },
            "licorice": {
                "foods": ["licorice candy", "licorice tea"],
                "reason": "Raises blood pressure",
                "severity": 1
            },
            "caffeine_excess": {
                "foods": ["energy drinks", "excessive coffee"],
                "reason": "Can temporarily raise blood pressure",
                "severity": 2
            }
        },
        
        "kidney_disease": {
            "high_potassium": {
                "foods": ["bananas", "oranges", "tomatoes", "potatoes", "avocado"],
                "reason": "Kidneys cannot remove excess potassium",
                "severity": 1
            },
            "high_phosphorus": {
                "foods": ["dairy products", "nuts", "whole grains", "beans"],
                "reason": "Can cause bone problems",
                "severity": 1
            },
            "high_protein": {
                "foods": ["red meat", "excessive protein"],
                "reason": "Stresses kidneys",
                "severity": 2
            }
        },
        
        "gout": {
            "high_purine": {
                "foods": ["organ meats", "anchovies", "sardines", "mussels", "beer"],
                "reason": "Increases uric acid",
                "severity": 1
            },
            "alcohol": {
                "foods": ["beer", "spirits", "excessive wine"],
                "reason": "Triggers gout attacks",
                "severity": 1
            }
        },
        
        "celiac_disease": {
            "gluten": {
                "foods": ["wheat", "barley", "rye", "bread", "pasta", "cereals"],
                "reason": "Autoimmune reaction damages intestines",
                "severity": 1
            }
        },
        
        "crohn_disease": {
            "high_fiber_during_flare": {
                "foods": ["raw vegetables", "nuts", "seeds", "popcorn"],
                "reason": "Can worsen symptoms during flare",
                "severity": 2
            },
            "dairy": {
                "foods": ["milk", "cheese", "ice cream"],
                "reason": "Many patients are lactose intolerant",
                "severity": 2
            }
        },
        
        "ibs": {
            "high_fodmap": {
                "foods": ["onions", "garlic", "wheat", "beans", "apples", "milk"],
                "reason": "Fermentable carbs cause bloating and pain",
                "severity": 2
            }
        },
        
        "hypothyroidism": {
            "goitrogens_raw": {
                "foods": ["raw cruciferous vegetables", "soy products"],
                "reason": "Can interfere with thyroid function (OK if cooked)",
                "severity": 3
            }
        },
        
        "osteoporosis": {
            "excessive_sodium": {
                "foods": ["salty processed foods"],
                "reason": "Increases calcium loss",
                "severity": 2
            },
            "excessive_caffeine": {
                "foods": ["excessive coffee"],
                "reason": "Can reduce calcium absorption",
                "severity": 3
            }
        }
    }
    
    # Foods to RECOMMEND by condition
    CONDITION_RECOMMEND = {
        "diabetes": [
            {
                "food": "leafy greens",
                "reason": "Low GI, high fiber, rich in magnesium",
                "frequency": "daily",
                "examples": ["spinach", "kale", "collard greens"]
            },
            {
                "food": "fatty fish",
                "reason": "Omega-3 reduces inflammation, heart disease risk",
                "frequency": "3x per week",
                "examples": ["salmon", "mackerel", "sardines"]
            },
            {
                "food": "nuts",
                "reason": "Improves blood sugar control, healthy fats",
                "frequency": "daily (1 oz)",
                "examples": ["almonds", "walnuts", "pistachios"]
            },
            {
                "food": "legumes",
                "reason": "Low GI, high fiber and protein",
                "frequency": "3-4x per week",
                "examples": ["lentils", "chickpeas", "black beans"]
            },
            {
                "food": "cinnamon",
                "reason": "May improve insulin sensitivity",
                "frequency": "daily (1 tsp)",
                "examples": ["ceylon cinnamon"]
            }
        ],
        
        "hypertension": [
            {
                "food": "leafy greens",
                "reason": "High in potassium, nitrates lower blood pressure",
                "frequency": "daily",
                "examples": ["spinach", "arugula", "swiss chard"]
            },
            {
                "food": "berries",
                "reason": "Anthocyanins improve blood vessel function",
                "frequency": "daily",
                "examples": ["blueberries", "strawberries", "raspberries"]
            },
            {
                "food": "beets",
                "reason": "Dietary nitrates lower blood pressure",
                "frequency": "3x per week",
                "examples": ["beetroot juice", "roasted beets"]
            },
            {
                "food": "fatty fish",
                "reason": "Omega-3 reduces blood pressure",
                "frequency": "2-3x per week",
                "examples": ["salmon", "mackerel", "trout"]
            },
            {
                "food": "garlic",
                "reason": "Allicin relaxes blood vessels",
                "frequency": "daily",
                "examples": ["fresh garlic", "aged garlic extract"]
            }
        ],
        
        "osteoporosis": [
            {
                "food": "dairy products",
                "reason": "High in calcium and vitamin D",
                "frequency": "3 servings daily",
                "examples": ["milk", "yogurt", "cheese"]
            },
            {
                "food": "leafy greens",
                "reason": "Calcium and vitamin K for bone health",
                "frequency": "daily",
                "examples": ["kale", "collard greens", "bok choy"]
            },
            {
                "food": "fatty fish",
                "reason": "Vitamin D and omega-3 for bone density",
                "frequency": "2-3x per week",
                "examples": ["salmon", "sardines with bones", "mackerel"]
            },
            {
                "food": "prunes",
                "reason": "Reduce bone loss",
                "frequency": "5-6 prunes daily",
                "examples": ["dried plums"]
            }
        ],
        
        "anemia": [
            {
                "food": "red meat",
                "reason": "Heme iron (highly absorbable)",
                "frequency": "3-4x per week",
                "examples": ["lean beef", "lamb", "liver"]
            },
            {
                "food": "legumes",
                "reason": "Non-heme iron and folate",
                "frequency": "daily",
                "examples": ["lentils", "kidney beans", "chickpeas"]
            },
            {
                "food": "dark leafy greens",
                "reason": "Iron, folate, vitamin C",
                "frequency": "daily",
                "examples": ["spinach", "kale", "swiss chard"]
            },
            {
                "food": "vitamin_c_foods",
                "reason": "Enhances iron absorption (eat with iron-rich foods)",
                "frequency": "with meals",
                "examples": ["citrus fruits", "bell peppers", "strawberries"]
            }
        ],
        
        "autoimmune_conditions": [
            {
                "food": "fatty fish",
                "reason": "Omega-3 reduces inflammation",
                "frequency": "3-4x per week",
                "examples": ["wild salmon", "sardines", "mackerel"]
            },
            {
                "food": "turmeric",
                "reason": "Curcumin is anti-inflammatory",
                "frequency": "daily (with black pepper)",
                "examples": ["turmeric powder", "golden milk"]
            },
            {
                "food": "berries",
                "reason": "Antioxidants reduce inflammation",
                "frequency": "daily",
                "examples": ["blueberries", "strawberries", "blackberries"]
            },
            {
                "food": "green_tea",
                "reason": "EGCG is anti-inflammatory",
                "frequency": "2-3 cups daily",
                "examples": ["matcha", "sencha"]
            }
        ]
    }
    
    @staticmethod
    def get_avoidances_for_profile(
        medical_conditions: List[str],
        allergies: Set[str],
        intolerances: Set[str],
        medications: List[str]
    ) -> List[FoodAvoidance]:
        """
        Get all foods to avoid for a user profile.
        
        Args:
            medical_conditions: List of diagnosed conditions
            allergies: Set of known allergies
            intolerances: Set of food intolerances
            medications: List of current medications
        
        Returns:
            List of FoodAvoidance objects
        """
        avoidances = []
        
        # Add allergies (highest priority)
        for allergen in allergies:
            avoidances.append(FoodAvoidance(
                item=allergen,
                reason=AvoidanceReason.ALLERGY,
                severity=1,
                explanation=f"Life-threatening allergy to {allergen}",
                alternatives=ConditionFoodDatabase._get_allergen_alternatives(allergen)
            ))
        
        # Add intolerances
        for intolerance in intolerances:
            avoidances.append(FoodAvoidance(
                item=intolerance,
                reason=AvoidanceReason.INTOLERANCE,
                severity=2,
                explanation=f"Food intolerance causes digestive issues",
                alternatives=ConditionFoodDatabase._get_intolerance_alternatives(intolerance)
            ))
        
        # Add condition-specific avoidances
        for condition in medical_conditions:
            condition_key = condition.lower().replace(" ", "_")
            
            if condition_key in ConditionFoodDatabase.CONDITION_AVOID:
                for category, data in ConditionFoodDatabase.CONDITION_AVOID[condition_key].items():
                    for food in data["foods"]:
                        avoidances.append(FoodAvoidance(
                            item=food,
                            reason=AvoidanceReason.MEDICAL_CONDITION,
                            severity=data["severity"],
                            explanation=f"{condition}: {data['reason']}",
                            alternatives=[]
                        ))
        
        # Add medication interactions
        for medication in medications:
            med_avoid = ConditionFoodDatabase._get_medication_interactions(medication)
            avoidances.extend(med_avoid)
        
        return avoidances
    
    @staticmethod
    def _get_allergen_alternatives(allergen: str) -> List[str]:
        """Get safe alternatives for common allergens"""
        alternatives = {
            "dairy": ["almond milk", "oat milk", "coconut milk", "soy milk"],
            "eggs": ["flax eggs", "chia seeds", "applesauce", "commercial egg replacer"],
            "nuts": ["seeds (sunflower, pumpkin)", "nut-free butter", "tahini"],
            "soy": ["pea protein", "hemp protein", "other legumes"],
            "gluten": ["rice", "quinoa", "buckwheat", "certified gluten-free oats"],
            "shellfish": ["fish", "poultry", "plant-based protein"],
            "fish": ["flaxseeds", "chia seeds", "walnuts", "algae oil (for omega-3)"]
        }
        return alternatives.get(allergen.lower(), [])
    
    @staticmethod
    def _get_intolerance_alternatives(intolerance: str) -> List[str]:
        """Get alternatives for intolerances"""
        alternatives = {
            "lactose": ["lactose-free milk", "hard cheeses", "yogurt", "plant milk"],
            "fructose": ["glucose-based sweeteners", "small portions of berries"],
            "histamine": ["fresh foods (avoid aged/fermented)", "low-histamine diet"]
        }
        return alternatives.get(intolerance.lower(), [])
    
    @staticmethod
    def _get_medication_interactions(medication: str) -> List[FoodAvoidance]:
        """Get food-medication interactions"""
        interactions = []
        med_lower = medication.lower()
        
        # Warfarin (blood thinner)
        if 'warfarin' in med_lower or 'coumadin' in med_lower:
            interactions.append(FoodAvoidance(
                item="vitamin_k_foods",
                reason=AvoidanceReason.MEDICATION_INTERACTION,
                severity=1,
                explanation="Vitamin K interferes with warfarin. Keep intake consistent.",
                alternatives=["Maintain stable vitamin K intake, don't eliminate"]
            ))
            interactions.append(FoodAvoidance(
                item="grapefruit",
                reason=AvoidanceReason.MEDICATION_INTERACTION,
                severity=2,
                explanation="May alter warfarin levels",
                alternatives=["oranges", "apples", "berries"]
            ))
        
        # MAO inhibitors (antidepressants)
        if 'maoi' in med_lower or any(x in med_lower for x in ['phenelzine', 'tranylcypromine']):
            interactions.append(FoodAvoidance(
                item="tyramine_foods",
                reason=AvoidanceReason.MEDICATION_INTERACTION,
                severity=1,
                explanation="Tyramine can cause dangerous blood pressure spike",
                alternatives=["Fresh foods instead of aged/fermented"]
            ))
        
        # Statins (cholesterol drugs)
        if 'statin' in med_lower or any(x in med_lower for x in ['atorvastatin', 'simvastatin']):
            interactions.append(FoodAvoidance(
                item="grapefruit",
                reason=AvoidanceReason.MEDICATION_INTERACTION,
                severity=1,
                explanation="Increases statin levels, risk of side effects",
                alternatives=["oranges", "tangerines", "other citrus OK"]
            ))
        
        return interactions
    
    @staticmethod
    def get_recommendations_for_conditions(
        medical_conditions: List[str]
    ) -> List[Dict]:
        """Get recommended foods for medical conditions"""
        recommendations = []
        
        for condition in medical_conditions:
            condition_key = condition.lower().replace(" ", "_")
            
            # Try exact match
            if condition_key in ConditionFoodDatabase.CONDITION_RECOMMEND:
                recommendations.extend(
                    ConditionFoodDatabase.CONDITION_RECOMMEND[condition_key]
                )
            # Try partial matches
            else:
                for key in ConditionFoodDatabase.CONDITION_RECOMMEND:
                    if key in condition_key or condition_key in key:
                        recommendations.extend(
                            ConditionFoodDatabase.CONDITION_RECOMMEND[key]
                        )
                        break
        
        return recommendations


# Test the avoidance system
if __name__ == "__main__":
    print("🚫 Food Avoidance & Recommendation System Test")
    print("=" * 70)
    
    # Test case 1: Diabetes with nut allergy
    print("\n📋 Test Case 1: Person with Diabetes + Nut Allergy")
    avoidances = ConditionFoodDatabase.get_avoidances_for_profile(
        medical_conditions=["type 2 diabetes"],
        allergies={"nuts", "peanuts"},
        intolerances=set(),
        medications=[]
    )
    
    print(f"\nFoods to AVOID ({len(avoidances)} items):")
    for avoid in sorted(avoidances, key=lambda x: x.severity):
        severity = "🔴 CRITICAL" if avoid.severity == 1 else "🟡 IMPORTANT" if avoid.severity == 2 else "🟢 PREFERENCE"
        print(f"  {severity} {avoid.item}")
        print(f"     Reason: {avoid.explanation}")
        if avoid.alternatives:
            print(f"     Alternatives: {', '.join(avoid.alternatives)}")
    
    recommendations = ConditionFoodDatabase.get_recommendations_for_conditions(
        ["type 2 diabetes"]
    )
    
    print(f"\nFoods to EAT ({len(recommendations)} categories):")
    for rec in recommendations:
        print(f"  ✅ {rec['food'].upper()}")
        print(f"     Why: {rec['reason']}")
        print(f"     How often: {rec['frequency']}")
        print(f"     Examples: {', '.join(rec['examples'])}")
    
    # Test case 2: Hypertension on medication
    print("\n\n📋 Test Case 2: Person with Hypertension on Warfarin")
    avoidances2 = ConditionFoodDatabase.get_avoidances_for_profile(
        medical_conditions=["hypertension"],
        allergies=set(),
        intolerances={"lactose"},
        medications=["warfarin"]
    )
    
    print(f"\nFoods to AVOID ({len(avoidances2)} items):")
    for avoid in sorted(avoidances2, key=lambda x: x.severity):
        severity = "🔴 CRITICAL" if avoid.severity == 1 else "🟡 IMPORTANT" if avoid.severity == 2 else "🟢 PREFERENCE"
        print(f"  {severity} {avoid.item}")
        print(f"     Reason: {avoid.explanation}")
        if avoid.alternatives:
            print(f"     Alternatives: {', '.join(avoid.alternatives[:2])}")
    
    print("\n✅ Avoidance system tests complete!")
    print("=" * 70)
