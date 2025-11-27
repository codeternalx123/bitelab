# Family Recipe Generation - Quick Start

## Overview

Generate recipes that work for **entire families** with different ages, health goals, and taste preferences. Perfect for households where members have varying dietary needs!

## Features

✅ **Age-Appropriate**: Portions and modifications for children, teens, adults, seniors  
✅ **Multi-Goal Optimization**: Balances health goals across family members  
✅ **Taste Compromise**: Finds recipes everyone will enjoy  
✅ **Allergy-Safe**: Checks all family members' allergies  
✅ **Health-Conscious**: Considers medical conditions and medications  
✅ **Customizable**: Per-member modifications and portion sizes  

## API Endpoints

### 1. Generate Family Recipes

**Endpoint**: `POST /api/v1/family-recipes/generate`

**Example Request**:
```json
{
  "family_profile": {
    "members": [
      {
        "name": "Mom",
        "age": 35,
        "gender": "female",
        "health_goals": ["weight loss", "heart health"],
        "medical_conditions": ["prediabetes"],
        "medications": [],
        "allergies": [],
        "dietary_restrictions": [],
        "taste_preferences": {
          "likes": ["savory", "Mediterranean"],
          "dislikes": ["too spicy"],
          "favorite_cuisines": ["Italian", "Greek"],
          "favorite_ingredients": ["chicken", "vegetables", "olive oil"]
        },
        "weight": 70.0,
        "height": 165.0,
        "activity_level": "moderate"
      },
      {
        "name": "Dad",
        "age": 38,
        "gender": "male",
        "health_goals": ["muscle gain", "energy"],
        "medical_conditions": ["high cholesterol"],
        "medications": ["statin"],
        "allergies": [],
        "dietary_restrictions": [],
        "taste_preferences": {
          "likes": ["protein-rich", "spicy"],
          "dislikes": ["bland"],
          "favorite_cuisines": ["Mexican", "Asian"],
          "favorite_ingredients": ["beef", "chicken", "beans"]
        },
        "weight": 85.0,
        "height": 180.0,
        "activity_level": "active"
      },
      {
        "name": "Emma",
        "age": 8,
        "gender": "female",
        "health_goals": ["growth", "bone health"],
        "medical_conditions": [],
        "medications": [],
        "allergies": ["peanuts"],
        "dietary_restrictions": [],
        "taste_preferences": {
          "likes": ["mild", "sweet", "creamy"],
          "dislikes": ["spicy", "bitter", "green vegetables"],
          "favorite_ingredients": ["pasta", "cheese", "chicken", "bread"]
        },
        "weight": 25.0,
        "height": 130.0,
        "activity_level": "active"
      }
    ],
    "household_dietary_restrictions": [],
    "budget_level": "moderate",
    "cooking_skill_level": "intermediate",
    "available_cooking_time": 45
  },
  "meal_type": "dinner",
  "cuisine_preference": "Italian",
  "max_recipes": 3
}
```

**Example Response**:
```json
[
  {
    "name": "Family-Style Baked Chicken Parmesan",
    "description": "Classic Italian comfort food with age-appropriate portions and spice levels",
    "cuisine_type": "Italian",
    "ingredients": [
      {"name": "chicken breast", "amount": "600g", "notes": "4 pieces"},
      {"name": "whole wheat breadcrumbs", "amount": "1 cup", "notes": "for coating"},
      {"name": "mozzarella cheese", "amount": "200g", "notes": "shredded"},
      {"name": "marinara sauce", "amount": "2 cups", "notes": "low sodium"},
      {"name": "parmesan cheese", "amount": "1/2 cup", "notes": "grated"}
    ],
    "instructions": [
      "Preheat oven to 375°F",
      "Coat chicken in breadcrumbs",
      "Bake for 20 minutes",
      "Top with sauce and cheese",
      "Bake 10 more minutes until cheese melts"
    ],
    "prep_time": 15,
    "cook_time": 30,
    "servings": 3,
    "difficulty": "easy",
    "nutrition_per_serving": {
      "calories": 420,
      "protein": 45,
      "carbs": 28,
      "fat": 14,
      "fiber": 4
    },
    "member_suitability": {
      "Mom": 0.88,
      "Dad": 0.92,
      "Emma": 0.95
    },
    "goal_alignment": {
      "Mom": {
        "weight loss": 0.82,
        "heart health": 0.89
      },
      "Dad": {
        "muscle gain": 0.91,
        "energy": 0.87
      },
      "Emma": {
        "growth": 0.93,
        "bone health": 0.90
      }
    },
    "taste_match": {
      "Mom": 0.90,
      "Dad": 0.75,
      "Emma": 0.95
    },
    "age_appropriate_portions": {
      "Mom": "1 chicken breast (150g) with side salad",
      "Dad": "1.5 chicken breasts (225g) with extra vegetables",
      "Emma": "3/4 chicken breast (110g) with pasta on the side"
    },
    "modifications_per_member": {
      "Mom": ["Use low-fat mozzarella", "Extra vegetables on the side"],
      "Dad": ["Larger portion for muscle building", "Can add red pepper flakes"],
      "Emma": ["Mild sauce only", "Cut into fun shapes", "Extra cheese if desired"]
    },
    "cost_estimate": "moderate",
    "allergen_warnings": [],
    "contraindications": [
      "Monitor cheese portion for Dad's cholesterol"
    ],
    "why_this_works": [
      "High protein supports Mom's weight loss and Dad's muscle gain",
      "Calcium-rich for Emma's bone development",
      "Everyone loves Italian comfort food",
      "Easily customizable spice levels",
      "One dish, multiple portions - efficient cooking"
    ],
    "tips_for_picky_eaters": [
      "Let Emma help bread the chicken",
      "Serve sauce on the side for control",
      "Make it fun with dinosaur-shaped chicken",
      "Add hidden veggies to the sauce"
    ],
    "family_health_benefits": [
      "Lean protein for all ages",
      "Calcium for bone health",
      "Heart-healthy when using low-fat cheese",
      "Whole grains from breadcrumbs"
    ],
    "overall_family_score": 0.92
  }
]
```

### 2. Generate Meal Plan

**Endpoint**: `POST /api/v1/family-recipes/meal-plan`

Generate 7-day meal plan for entire family.

### 3. Analyze Family Requirements

**Endpoint**: `POST /api/v1/family-recipes/analyze-family`

Get analysis of family constraints before recipe generation.

## Chat Integration

You can also use the **conversational AI** to generate family recipes:

### Example Conversation

**User**: "I need dinner ideas for my family. We have 3 people: me (35yo woman trying to lose weight), my husband (38yo, building muscle, high cholesterol), and our 8-year-old daughter Emma who's allergic to peanuts and doesn't like spicy food. I like Mediterranean food, my husband likes Mexican, and Emma loves pasta and cheese."

**AI**: I'll generate recipes perfect for your family! Let me analyze:
- Mom: weight loss + heart health, likes Mediterranean
- Dad: muscle gain, high cholesterol, likes Mexican 
- Emma: growth, peanut allergy, likes mild pasta

*[AI calls `generate_family_recipe` function]*

**AI Response**: 

I've found the perfect recipe for your family! 🍝

**Baked Chicken Parmesan (Italian-Mexican Fusion)**

This works because:
✅ **For Mom**: High protein, moderate calories (420/serving) - great for weight loss  
✅ **For Dad**: 45g protein per serving - excellent for muscle building  
✅ **For Emma**: Kid-approved pasta + cheese, **completely peanut-free**

**Portions**:
- Mom: 1 chicken breast (150g) + side salad
- Dad: 1.5 chicken breasts (225g) + extra protein
- Emma: 3/4 breast (110g) + pasta on the side

**Customizations**:
- Mom: Low-fat mozzarella for heart health
- Dad: Can add spicy salsa on his portion
- Emma: Mild sauce, cut into fun shapes

**Health Benefits**:
- Lean protein for everyone
- Calcium for Emma's growing bones
- Heart-healthy (important for Dad's cholesterol)

Would you like the full recipe with instructions?

---

### More Examples

**User**: "My family has different health goals. Can you suggest breakfast for: Dad (diabetes, weight loss), Mom (heart health, vegetarian), and two kids (7 and 10 years old)?"

**AI**: *[Generates vegetarian, diabetes-friendly, kid-approved breakfast]*

---

**User**: "We need Asian food for dinner. Family: grandma (85, low sodium needed), me (40, muscle building), wife (38, vegan), son (12, very active athlete). Everyone likes different spice levels."

**AI**: *[Generates Asian stir-fry with customizable portions and spice]*

## Python Code Example

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="your-key")

async def get_family_recipe():
    response = await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": "You are Wellomex AI nutrition assistant with family recipe generation."
            },
            {
                "role": "user",
                "content": """Generate dinner recipe for my family:
                - Mom (35): weight loss, vegetarian
                - Dad (38): muscle gain, likes spicy
                - Kids (6, 10): picky eaters, like pasta
                
                Something Italian that works for everyone!"""
            }
        ],
        functions=[
            {
                "name": "generate_family_recipe",
                "description": "Generate recipes for family with different needs",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "family_members": {
                            "type": "array",
                            "items": {"type": "object"}
                        },
                        "meal_type": {"type": "string"},
                        "cuisine_preference": {"type": "string"}
                    }
                }
            }
        ],
        function_call="auto"
    )
    
    print(response.choices[0].message)

asyncio.run(get_family_recipe())
```

## Key Benefits

### 1. **Saves Time**
- One recipe for everyone
- No need to cook multiple meals
- Pre-calculated portions

### 2. **Healthier Family**
- Each person's goals addressed
- Age-appropriate nutrition
- Medical conditions considered

### 3. **Less Food Waste**
- Shared ingredients
- Right-sized portions
- Smart grocery lists

### 4. **Happier Mealtimes**
- Everyone gets food they like
- Modifications for picky eaters
- Fun family cooking

## Use Cases

### 1. Busy Parents
"I don't have time to cook separate meals for everyone!"
→ One recipe, multiple portions, everyone happy

### 2. Multi-Generational Homes
Grandparents + parents + kids all eating together
→ Age-appropriate portions and textures

### 3. Mixed Dietary Needs
Vegetarian mom + carnivore dad + picky kids
→ Recipes with easy swaps

### 4. Health-Conscious Families
Different medical conditions per person
→ Safe for everyone with modifications

## Tips for Best Results

1. **Be Specific**: Include ages, goals, preferences
2. **Mention Allergies**: Critical for safety
3. **Share Dislikes**: Helps avoid foods kids won't eat
4. **Set Time Limits**: Specify cooking time available
5. **Indicate Skill Level**: Gets appropriate recipes

## Common Questions

**Q: Can it handle vegan + meat-eater in same family?**  
A: Yes! Recipes include protein swaps (tofu for mom, chicken for dad)

**Q: What if one child is extremely picky?**  
A: System provides tips and fun modifications to encourage trying

**Q: Can it accommodate multiple allergies?**  
A: Yes, checks ALL family members' allergies and avoids them

**Q: Does it work for large families (5+ people)?**  
A: Absolutely! Scales portions automatically

**Q: Can I get a weekly meal plan?**  
A: Yes! Use the meal plan endpoint for 7-day planning

## Next Steps

1. Try the API with your family profile
2. Use the chat interface for natural conversation
3. Generate a weekly meal plan
4. Save favorite recipes for repeat use

---

**Need Help?**
- API Docs: http://localhost:8000/api/docs
- Support: support@wellomex.com
