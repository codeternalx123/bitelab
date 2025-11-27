"""
Data Loader
==========

Handles loading of external data configurations for the Risk Integration Layer.
Replaces hardcoded dictionaries with dynamic data loading from JSON/DB.

Manages:
- Disease-Specific Nutritional Rules
- Therapeutic Goal Definitions
- Compound Interaction Data
"""

import os
import json
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Centralized loader for configuration data.
    """
    
    def __init__(self, data_dir: str = "data/risk_integration"):
        self.data_dir = data_dir
        self._ensure_data_dir()
        
        # Cache
        self._disease_rules = None
        self._goal_definitions = None
        
    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            # Create default files if they don't exist (for this demo)
            self._create_default_data()

    def _create_default_data(self):
        """Create default JSON files if missing."""
        rules_path = os.path.join(self.data_dir, "disease_rules.json")
        if not os.path.exists(rules_path):
            default_rules = {
                "ckd": {
                    "avoid": ["starfruit", "high_sodium", "processed_meats"],
                    "limit": ["potassium", "phosphorus"],
                    "beneficial": ["cauliflower", "blueberries", "egg_whites"],
                    "compounds": {
                        "beneficial": ["bicarbonate"],
                        "harmful": ["oxalate", "uric_acid"]
                    }
                },
                "diabetes": {
                    "avoid": ["sugary_drinks", "refined_carbs"],
                    "limit": ["saturated_fats"],
                    "beneficial": ["leafy_greens", "berries", "fatty_fish"],
                    "compounds": {
                        "beneficial": ["fiber", "chromium", "magnesium"],
                        "harmful": ["added_sugars"]
                    }
                },
                "hypertension": {
                    "avoid": ["salty_snacks", "canned_soups"],
                    "limit": ["sodium", "alcohol"],
                    "beneficial": ["beets", "garlic", "pomegranate"],
                    "compounds": {
                        "beneficial": ["nitrates", "potassium", "magnesium"],
                        "harmful": ["sodium"]
                    }
                }
            }
            with open(rules_path, 'w') as f:
                json.dump(default_rules, f, indent=2)

    def load_disease_rules(self) -> Dict[str, Any]:
        """Load disease-specific nutritional rules."""
        if self._disease_rules:
            return self._disease_rules
            
        path = os.path.join(self.data_dir, "disease_rules.json")
        try:
            with open(path, 'r') as f:
                self._disease_rules = json.load(f)
            return self._disease_rules
        except Exception as e:
            logger.error(f"Failed to load disease rules: {e}")
            return {}

    def load_goal_definitions(self) -> Dict[str, Any]:
        """Load therapeutic goal definitions and associated compounds."""
        if self._goal_definitions:
            return self._goal_definitions
            
        # In a real app, this would be a file. Returning dict for now.
        return {
            "reduce_inflammation": ["curcumin", "omega-3", "quercetin", "resveratrol"],
            "kidney_protection": ["bicarbonate", "antioxidants"],
            "manage_blood_sugar": ["fiber", "chromium", "berberine"],
            # ... (would contain all 55+ goals)
        }
