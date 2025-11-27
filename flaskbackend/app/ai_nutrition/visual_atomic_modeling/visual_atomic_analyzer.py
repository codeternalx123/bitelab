"""
Visual-Atomic Food Analyzer
Analyzes food appearance (shininess, reflection, color, texture) to predict atomic composition
and safety based on ICP-MS correlation data.

This system uses computer vision and machine learning to:
1. Extract visual features from food images or descriptions
2. Predict elemental composition from visual characteristics
3. Validate predictions with ICP-MS measurements
4. Assess food safety and quality
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from datetime import datetime


class SurfaceType(Enum):
    """Types of surface appearance for foods"""
    GLOSSY = "glossy"  # High shine, strong reflection (e.g., fresh fruits, oily fish)
    MATTE = "matte"  # Low shine, diffuse reflection (e.g., dry grains, beans)
    WAXY = "waxy"  # Medium shine, protective coating (e.g., apples, peppers)
    METALLIC = "metallic"  # Metallic sheen (can indicate high mineral content or contamination)
    CRYSTALLINE = "crystalline"  # Crystal-like appearance (salt deposits, sugar)
    FIBROUS = "fibrous"  # Textured, fiber-visible (e.g., celery, leafy greens)
    SMOOTH = "smooth"  # Uniform, no texture (e.g., tofu, processed foods)
    ROUGH = "rough"  # Irregular surface (e.g., broccoli, cauliflower)


class ColorProfile(Enum):
    """Color profiles indicating different nutrient/element groups"""
    DEEP_GREEN = "deep_green"  # High chlorophyll, iron, magnesium (spinach, kale)
    YELLOW_ORANGE = "yellow_orange"  # Beta-carotene, vitamin A (carrots, squash)
    RED_PURPLE = "red_purple"  # Anthocyanins, antioxidants (berries, beets)
    WHITE_PALE = "white_pale"  # Low pigment, calcium-rich (dairy, cauliflower)
    BROWN = "brown"  # Oxidation, aging (old produce, whole grains)
    VIBRANT = "vibrant"  # Fresh, high nutrient density
    DULL = "dull"  # Aging, nutrient degradation
    DISCOLORED = "discolored"  # Possible contamination or spoilage


@dataclass
class VisualFeatures:
    """Visual characteristics of food sample"""
    # Shininess & Reflection (0-100 scale)
    shininess_index: float  # 0=completely matte, 100=mirror-like
    reflection_intensity: float  # Light reflection strength
    specular_highlights: int  # Number of bright spots
    
    # Surface Properties
    surface_type: SurfaceType
    texture_roughness: float  # 0=smooth, 100=very rough
    moisture_appearance: float  # 0=dry, 100=wet/glossy
    
    # Color Analysis
    color_profile: ColorProfile
    rgb_values: Tuple[int, int, int]  # Average RGB
    color_uniformity: float  # 0=mixed colors, 100=uniform
    brightness: float  # 0=dark, 100=bright
    saturation: float  # 0=gray, 100=vivid
    
    # Freshness Indicators
    wilting_score: float  # 0=fresh/firm, 100=wilted
    browning_score: float  # 0=no browning, 100=heavily browned
    spots_or_blemishes: int  # Count of visible defects
    
    # Size & Shape
    size_mm: float  # Average dimension
    shape_regularity: float  # 0=irregular, 100=perfect shape
    
    # Advanced Features
    translucency: float  # 0=opaque, 100=transparent
    crystalline_structures: bool  # Visible crystals (salt, sugar, minerals)
    oily_film: bool  # Visible oil layer
    dust_or_residue: bool  # Surface contamination visible
    
    # Metadata
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    lighting_conditions: str = "standard"  # ambient, direct, UV, etc.


@dataclass
class ElementPrediction:
    """Predicted elemental composition from visual features"""
    element_symbol: str
    predicted_ppm: float
    confidence: float  # 0-1
    prediction_basis: List[str]  # Visual features used for prediction
    icpms_validated: bool = False
    icpms_actual_ppm: Optional[float] = None
    prediction_error: Optional[float] = None  # % difference from actual


@dataclass
class VisualSafetyAssessment:
    """Safety assessment based on visual inspection"""
    is_safe: bool
    safety_score: float  # 0-100
    visual_warnings: List[str]
    contamination_indicators: List[str]
    freshness_rating: str  # Excellent/Good/Fair/Poor/Unsafe
    recommended_action: str  # Consume/Wash/Discard
    
    # Specific concerns
    heavy_metal_risk: str  # None/Low/Medium/High
    pesticide_residue_risk: str  # None/Low/Medium/High
    microbial_risk: str  # None/Low/Medium/High
    spoilage_risk: str  # None/Low/Medium/High


class VisualAtomicCorrelations:
    """
    Database of correlations between visual features and elemental composition.
    Based on research in food science and spectroscopy.
    """
    
    @staticmethod
    def get_shininess_correlations() -> Dict:
        """Shininess often correlates with fat content, moisture, and certain minerals"""
        return {
            'high_shine': {  # 70-100 shininess
                'elements': {
                    'P': {'range': (100, 300), 'reason': 'Phospholipids in cell membranes'},
                    'K': {'range': (200, 500), 'reason': 'High water content, cellular integrity'},
                    'Na': {'range': (50, 150), 'reason': 'Surface moisture retention'},
                },
                'foods': ['fresh fish', 'leafy greens', 'fresh fruits', 'oily vegetables']
            },
            'low_shine': {  # 0-30 shininess
                'elements': {
                    'Fe': {'range': (5, 50), 'reason': 'Oxidized iron reduces shine'},
                    'Ca': {'range': (50, 200), 'reason': 'Calcium carbonate deposits (matte)'},
                },
                'foods': ['dried beans', 'grains', 'aged produce', 'powders']
            },
            'waxy_shine': {  # 40-70 with even coating
                'elements': {
                    'Ca': {'range': (100, 300), 'reason': 'Calcium pectate in fruit skins'},
                    'Mg': {'range': (50, 150), 'reason': 'Chlorophyll-related wax'},
                },
                'foods': ['apples', 'peppers', 'cucumbers', 'citrus fruits']
            }
        }
    
    @staticmethod
    def get_color_element_correlations() -> Dict:
        """Color profiles strongly indicate specific elements and compounds"""
        return {
            ColorProfile.DEEP_GREEN: {
                'primary_elements': {
                    'Mg': {'range': (50, 300), 'reason': 'Magnesium in chlorophyll'},
                    'Fe': {'range': (10, 50), 'reason': 'Iron in chloroplasts'},
                    'K': {'range': (200, 600), 'reason': 'Potassium in plant cells'},
                    'Ca': {'range': (100, 500), 'reason': 'Calcium in cell walls'},
                },
                'indicator': 'High chlorophyll content = High Mg, Fe, K'
            },
            ColorProfile.YELLOW_ORANGE: {
                'primary_elements': {
                    'K': {'range': (300, 800), 'reason': 'High in root vegetables'},
                    'P': {'range': (50, 150), 'reason': 'Phosphorus in carotenoids'},
                    'Ca': {'range': (30, 100), 'reason': 'Moderate calcium'},
                },
                'indicator': 'Beta-carotene rich = High K, moderate P'
            },
            ColorProfile.RED_PURPLE: {
                'primary_elements': {
                    'K': {'range': (200, 500), 'reason': 'Potassium in berries'},
                    'Mn': {'range': (5, 30), 'reason': 'Manganese in anthocyanins'},
                    'Fe': {'range': (5, 20), 'reason': 'Iron in red pigments'},
                },
                'indicator': 'Anthocyanins = High K, Mn, moderate Fe'
            },
            ColorProfile.WHITE_PALE: {
                'primary_elements': {
                    'Ca': {'range': (100, 500), 'reason': 'Calcium-rich foods'},
                    'P': {'range': (100, 300), 'reason': 'Phosphorus in proteins'},
                    'S': {'range': (50, 200), 'reason': 'Sulfur in cruciferous'},
                },
                'indicator': 'Low pigment = High Ca, P (cauliflower, dairy)'
            },
            ColorProfile.BROWN: {
                'warning_elements': {
                    'Fe': {'concern': 'Oxidized iron (rust-like)', 'safe_range': (0, 100)},
                    'Cu': {'concern': 'Copper oxidation (brown spots)', 'safe_range': (0, 10)},
                },
                'indicator': 'Oxidation/aging - check for heavy metal accumulation'
            },
            ColorProfile.DISCOLORED: {
                'contamination_risk': {
                    'Pb': {'concern': 'Lead contamination (gray tint)', 'limit': 0.1},
                    'Cd': {'concern': 'Cadmium (yellow-brown)', 'limit': 0.05},
                    'Hg': {'concern': 'Mercury (silver spots)', 'limit': 0.05},
                },
                'indicator': 'Abnormal color = Possible heavy metal contamination'
            }
        }
    
    @staticmethod
    def get_texture_correlations() -> Dict:
        """Surface texture indicates mineral deposits and structure"""
        return {
            'crystalline': {
                'elements': {
                    'Na': {'range': (500, 5000), 'reason': 'Salt crystals'},
                    'Ca': {'range': (100, 1000), 'reason': 'Calcium deposits'},
                    'Mg': {'range': (50, 500), 'reason': 'Magnesium salts'},
                },
                'safety': 'Check if natural or contamination'
            },
            'metallic_sheen': {
                'warning': 'Possible heavy metal contamination or mineral coating',
                'test_for': ['Pb', 'Cd', 'Hg', 'Al', 'As'],
                'safe_ranges': {
                    'Pb': 0.1, 'Cd': 0.05, 'Hg': 0.05, 'Al': 10, 'As': 0.2
                }
            },
            'white_coating': {
                'elements': {
                    'Ca': {'range': (200, 800), 'reason': 'Calcium deposits'},
                    'S': {'range': (100, 400), 'reason': 'Sulfur compounds (bloom)'},
                },
                'foods': ['grapes (bloom)', 'plums', 'cabbage']
            },
            'oily_film': {
                'elements': {
                    'P': {'range': (100, 400), 'reason': 'Phospholipids'},
                    'Se': {'range': (10, 50), 'reason': 'Selenium in oils'},
                    'Zn': {'range': (5, 30), 'reason': 'Zinc in cell membranes'},
                },
                'contamination_check': ['pesticide residues', 'industrial oils']
            }
        }


class VisualAtomicPredictor:
    """
    Machine learning model (simplified) that predicts elemental composition
    from visual features using correlation rules and heuristics.
    """
    
    def __init__(self):
        self.correlations = VisualAtomicCorrelations()
        self.prediction_history = []
    
    def predict_elements_from_visual(
        self, 
        visual_features: VisualFeatures,
        food_name: Optional[str] = None
    ) -> List[ElementPrediction]:
        """
        Predict elemental composition from visual features.
        Returns list of element predictions with confidence scores.
        """
        predictions = []
        
        # 1. Predict from shininess
        shine_predictions = self._predict_from_shininess(visual_features)
        predictions.extend(shine_predictions)
        
        # 2. Predict from color profile
        color_predictions = self._predict_from_color(visual_features)
        predictions.extend(color_predictions)
        
        # 3. Predict from texture/surface
        texture_predictions = self._predict_from_texture(visual_features)
        predictions.extend(texture_predictions)
        
        # 4. Adjust for freshness indicators
        predictions = self._adjust_for_freshness(predictions, visual_features)
        
        # 5. Merge duplicate elements (take highest confidence)
        predictions = self._merge_predictions(predictions)
        
        # 6. Add contamination predictions if visual warnings present
        if visual_features.spots_or_blemishes > 5 or visual_features.color_profile == ColorProfile.DISCOLORED:
            contamination_predictions = self._predict_contamination(visual_features)
            predictions.extend(contamination_predictions)
        
        return sorted(predictions, key=lambda x: x.confidence, reverse=True)
    
    def _predict_from_shininess(self, vf: VisualFeatures) -> List[ElementPrediction]:
        """Predict elements based on surface shininess"""
        predictions = []
        correlations = self.correlations.get_shininess_correlations()
        
        if vf.shininess_index >= 70:
            # High shine
            for elem, data in correlations['high_shine']['elements'].items():
                ppm_range = data['range']
                predicted_ppm = np.mean(ppm_range) * (vf.shininess_index / 100)
                
                predictions.append(ElementPrediction(
                    element_symbol=elem,
                    predicted_ppm=predicted_ppm,
                    confidence=0.7 if vf.moisture_appearance > 60 else 0.5,
                    prediction_basis=[f"High shininess ({vf.shininess_index})", data['reason']]
                ))
        
        elif vf.shininess_index <= 30:
            # Low shine
            for elem, data in correlations['low_shine']['elements'].items():
                ppm_range = data['range']
                predicted_ppm = np.mean(ppm_range) * (1 - vf.shininess_index / 100)
                
                predictions.append(ElementPrediction(
                    element_symbol=elem,
                    predicted_ppm=predicted_ppm,
                    confidence=0.6,
                    prediction_basis=[f"Low shininess ({vf.shininess_index})", data['reason']]
                ))
        
        else:
            # Waxy shine
            for elem, data in correlations['waxy_shine']['elements'].items():
                ppm_range = data['range']
                predicted_ppm = np.mean(ppm_range)
                
                predictions.append(ElementPrediction(
                    element_symbol=elem,
                    predicted_ppm=predicted_ppm,
                    confidence=0.65,
                    prediction_basis=[f"Waxy surface ({vf.shininess_index})", data['reason']]
                ))
        
        return predictions
    
    def _predict_from_color(self, vf: VisualFeatures) -> List[ElementPrediction]:
        """Predict elements based on color profile"""
        predictions = []
        correlations = self.correlations.get_color_element_correlations()
        
        if vf.color_profile not in correlations:
            return predictions
        
        profile_data = correlations[vf.color_profile]
        
        # Check for primary elements
        if 'primary_elements' in profile_data:
            for elem, data in profile_data['primary_elements'].items():
                ppm_range = data['range']
                # Adjust prediction based on color saturation and brightness
                intensity_factor = (vf.saturation + vf.brightness) / 200
                predicted_ppm = np.mean(ppm_range) * intensity_factor
                
                predictions.append(ElementPrediction(
                    element_symbol=elem,
                    predicted_ppm=predicted_ppm,
                    confidence=0.8 if vf.color_uniformity > 70 else 0.6,
                    prediction_basis=[
                        f"Color profile: {vf.color_profile.value}",
                        data['reason'],
                        profile_data.get('indicator', '')
                    ]
                ))
        
        # Check for contamination warnings
        if 'contamination_risk' in profile_data:
            for elem, data in profile_data['contamination_risk'].items():
                predictions.append(ElementPrediction(
                    element_symbol=elem,
                    predicted_ppm=data['limit'] * 0.5,  # Conservative estimate
                    confidence=0.4,  # Lower confidence, needs ICP-MS validation
                    prediction_basis=[
                        f"Visual warning: {vf.color_profile.value}",
                        data['concern'],
                        "⚠️ ICP-MS validation required"
                    ]
                ))
        
        return predictions
    
    def _predict_from_texture(self, vf: VisualFeatures) -> List[ElementPrediction]:
        """Predict elements based on surface texture"""
        predictions = []
        correlations = self.correlations.get_texture_correlations()
        
        # Crystalline structures
        if vf.crystalline_structures:
            for elem, data in correlations['crystalline']['elements'].items():
                ppm_range = data['range']
                predicted_ppm = np.mean(ppm_range)
                
                predictions.append(ElementPrediction(
                    element_symbol=elem,
                    predicted_ppm=predicted_ppm,
                    confidence=0.75,
                    prediction_basis=["Crystalline structures visible", data['reason']]
                ))
        
        # Metallic sheen (contamination warning)
        if vf.surface_type == SurfaceType.METALLIC:
            for elem in correlations['metallic_sheen']['test_for']:
                safe_limit = correlations['metallic_sheen']['safe_ranges'][elem]
                
                predictions.append(ElementPrediction(
                    element_symbol=elem,
                    predicted_ppm=safe_limit * 0.7,  # Conservative
                    confidence=0.3,  # Low confidence, urgent ICP-MS needed
                    prediction_basis=[
                        "⚠️ METALLIC SHEEN DETECTED",
                        correlations['metallic_sheen']['warning'],
                        "URGENT: ICP-MS validation required"
                    ]
                ))
        
        # Oily film
        if vf.oily_film:
            for elem, data in correlations['oily_film']['elements'].items():
                ppm_range = data['range']
                predicted_ppm = np.mean(ppm_range)
                
                predictions.append(ElementPrediction(
                    element_symbol=elem,
                    predicted_ppm=predicted_ppm,
                    confidence=0.65,
                    prediction_basis=["Oily film present", data['reason']]
                ))
        
        return predictions
    
    def _adjust_for_freshness(
        self, 
        predictions: List[ElementPrediction],
        vf: VisualFeatures
    ) -> List[ElementPrediction]:
        """Adjust predictions based on freshness indicators"""
        freshness_factor = 1.0 - (vf.wilting_score + vf.browning_score) / 200
        
        for pred in predictions:
            if pred.element_symbol in ['K', 'Mg', 'Ca']:  # Elements that degrade with age
                pred.predicted_ppm *= max(0.5, freshness_factor)
                if freshness_factor < 0.7:
                    pred.confidence *= 0.8
                    pred.prediction_basis.append(f"Adjusted for freshness (factor: {freshness_factor:.2f})")
        
        return predictions
    
    def _merge_predictions(self, predictions: List[ElementPrediction]) -> List[ElementPrediction]:
        """Merge duplicate element predictions, taking highest confidence"""
        merged = {}
        
        for pred in predictions:
            elem = pred.element_symbol
            if elem not in merged or pred.confidence > merged[elem].confidence:
                if elem in merged:
                    # Combine prediction bases
                    pred.prediction_basis.extend(merged[elem].prediction_basis)
                merged[elem] = pred
        
        return list(merged.values())
    
    def _predict_contamination(self, vf: VisualFeatures) -> List[ElementPrediction]:
        """Predict possible contamination from visual defects"""
        predictions = []
        toxic_elements = ['Pb', 'Cd', 'Hg', 'As', 'Al']
        safe_limits = {'Pb': 0.1, 'Cd': 0.05, 'Hg': 0.05, 'As': 0.2, 'Al': 10}
        
        # Risk based on visual defects
        defect_score = (vf.spots_or_blemishes + vf.browning_score) / 2
        contamination_risk = min(defect_score / 100, 0.5)  # Max 50% of safe limit
        
        for elem in toxic_elements:
            if contamination_risk > 0.2:  # Only predict if significant risk
                predictions.append(ElementPrediction(
                    element_symbol=elem,
                    predicted_ppm=safe_limits[elem] * contamination_risk,
                    confidence=0.3,
                    prediction_basis=[
                        f"Visual defects detected ({vf.spots_or_blemishes} spots)",
                        f"Contamination risk: {contamination_risk*100:.1f}%",
                        "⚠️ ICP-MS validation strongly recommended"
                    ]
                ))
        
        return predictions
    
    def validate_with_icpms(
        self,
        predictions: List[ElementPrediction],
        icpms_measurements: Dict[str, float]
    ) -> List[ElementPrediction]:
        """
        Validate visual predictions with actual ICP-MS measurements.
        Updates predictions with actual values and calculates error.
        """
        for pred in predictions:
            if pred.element_symbol in icpms_measurements:
                actual = icpms_measurements[pred.element_symbol]
                pred.icpms_validated = True
                pred.icpms_actual_ppm = actual
                
                # Calculate prediction error
                if pred.predicted_ppm > 0:
                    error = abs(actual - pred.predicted_ppm) / pred.predicted_ppm * 100
                    pred.prediction_error = error
                    
                    # Adjust confidence based on accuracy
                    if error < 20:  # Within 20%
                        pred.confidence = min(1.0, pred.confidence * 1.2)
                    elif error > 50:  # More than 50% off
                        pred.confidence *= 0.6
        
        return predictions


class VisualSafetyAnalyzer:
    """Analyzes food safety based on visual inspection"""
    
    FRESHNESS_THRESHOLDS = {
        'excellent': {'wilting': (0, 10), 'browning': (0, 5), 'spots': (0, 2)},
        'good': {'wilting': (10, 30), 'browning': (5, 20), 'spots': (2, 5)},
        'fair': {'wilting': (30, 50), 'browning': (20, 40), 'spots': (5, 10)},
        'poor': {'wilting': (50, 70), 'browning': (40, 60), 'spots': (10, 20)},
        'unsafe': {'wilting': (70, 100), 'browning': (60, 100), 'spots': (20, 1000)}
    }
    
    def assess_visual_safety(
        self,
        visual_features: VisualFeatures,
        element_predictions: List[ElementPrediction]
    ) -> VisualSafetyAssessment:
        """Comprehensive visual safety assessment"""
        
        warnings = []
        contamination_indicators = []
        
        # 1. Check freshness
        freshness_rating = self._assess_freshness(visual_features)
        if freshness_rating in ['poor', 'unsafe']:
            warnings.append(f"Poor freshness rating: {freshness_rating}")
        
        # 2. Check for contamination indicators
        if visual_features.surface_type == SurfaceType.METALLIC:
            contamination_indicators.append("Metallic sheen detected - possible heavy metal contamination")
            warnings.append("⚠️ URGENT: Metallic appearance requires ICP-MS testing")
        
        if visual_features.color_profile == ColorProfile.DISCOLORED:
            contamination_indicators.append("Abnormal discoloration detected")
            warnings.append("Discoloration may indicate contamination or spoilage")
        
        if visual_features.dust_or_residue:
            contamination_indicators.append("Visible dust or residue on surface")
            warnings.append("Wash thoroughly before consumption")
        
        # 3. Check predicted heavy metals
        heavy_metal_risk = self._assess_heavy_metal_risk(element_predictions)
        pesticide_risk = self._assess_pesticide_risk(visual_features)
        microbial_risk = self._assess_microbial_risk(visual_features)
        spoilage_risk = self._assess_spoilage_risk(visual_features)
        
        # 4. Calculate overall safety score
        safety_score = self._calculate_safety_score(
            visual_features,
            freshness_rating,
            heavy_metal_risk,
            pesticide_risk,
            microbial_risk,
            spoilage_risk
        )
        
        # 5. Determine if safe and recommended action
        is_safe = safety_score >= 60 and freshness_rating not in ['unsafe']
        recommended_action = self._determine_action(safety_score, freshness_rating, warnings)
        
        return VisualSafetyAssessment(
            is_safe=is_safe,
            safety_score=safety_score,
            visual_warnings=warnings,
            contamination_indicators=contamination_indicators,
            freshness_rating=freshness_rating,
            recommended_action=recommended_action,
            heavy_metal_risk=heavy_metal_risk,
            pesticide_residue_risk=pesticide_risk,
            microbial_risk=microbial_risk,
            spoilage_risk=spoilage_risk
        )
    
    def _assess_freshness(self, vf: VisualFeatures) -> str:
        """Determine freshness rating from visual features"""
        for rating, thresholds in self.FRESHNESS_THRESHOLDS.items():
            if (thresholds['wilting'][0] <= vf.wilting_score <= thresholds['wilting'][1] and
                thresholds['browning'][0] <= vf.browning_score <= thresholds['browning'][1] and
                thresholds['spots'][0] <= vf.spots_or_blemishes <= thresholds['spots'][1]):
                return rating
        return 'poor'
    
    def _assess_heavy_metal_risk(self, predictions: List[ElementPrediction]) -> str:
        """Assess risk of heavy metal contamination"""
        toxic_metals = {'Pb', 'Cd', 'Hg', 'As', 'Al'}
        safe_limits = {'Pb': 0.1, 'Cd': 0.05, 'Hg': 0.05, 'As': 0.2, 'Al': 10}
        
        max_risk = 0
        for pred in predictions:
            if pred.element_symbol in toxic_metals:
                limit = safe_limits[pred.element_symbol]
                risk_ratio = pred.predicted_ppm / limit
                max_risk = max(max_risk, risk_ratio)
        
        if max_risk == 0:
            return "None"
        elif max_risk < 0.5:
            return "Low"
        elif max_risk < 1.0:
            return "Medium"
        else:
            return "High"
    
    def _assess_pesticide_risk(self, vf: VisualFeatures) -> str:
        """Assess risk of pesticide residue"""
        if vf.oily_film and vf.dust_or_residue:
            return "Medium"
        elif vf.dust_or_residue:
            return "Low"
        return "None"
    
    def _assess_microbial_risk(self, vf: VisualFeatures) -> str:
        """Assess risk of microbial contamination"""
        if vf.spots_or_blemishes > 15 or vf.wilting_score > 70:
            return "High"
        elif vf.spots_or_blemishes > 5 or vf.wilting_score > 40:
            return "Medium"
        elif vf.spots_or_blemishes > 0:
            return "Low"
        return "None"
    
    def _assess_spoilage_risk(self, vf: VisualFeatures) -> str:
        """Assess risk of spoilage"""
        if vf.browning_score > 60 or vf.wilting_score > 70:
            return "High"
        elif vf.browning_score > 30 or vf.wilting_score > 40:
            return "Medium"
        elif vf.browning_score > 10 or vf.wilting_score > 20:
            return "Low"
        return "None"
    
    def _calculate_safety_score(
        self,
        vf: VisualFeatures,
        freshness: str,
        heavy_metal: str,
        pesticide: str,
        microbial: str,
        spoilage: str
    ) -> float:
        """Calculate overall safety score (0-100)"""
        score = 100.0
        
        # Freshness penalty
        freshness_penalties = {'excellent': 0, 'good': 5, 'fair': 15, 'poor': 30, 'unsafe': 50}
        score -= freshness_penalties.get(freshness, 30)
        
        # Risk penalties
        risk_penalties = {'None': 0, 'Low': 5, 'Medium': 15, 'High': 30}
        score -= risk_penalties.get(heavy_metal, 15)
        score -= risk_penalties.get(pesticide, 5)
        score -= risk_penalties.get(microbial, 10)
        score -= risk_penalties.get(spoilage, 10)
        
        # Visual defect penalty
        score -= min(vf.spots_or_blemishes * 2, 20)
        
        return max(0, score)
    
    def _determine_action(self, safety_score: float, freshness: str, warnings: List[str]) -> str:
        """Determine recommended action"""
        if safety_score >= 80 and freshness in ['excellent', 'good']:
            return "Safe to consume (wash as normal)"
        elif safety_score >= 60 and freshness in ['good', 'fair']:
            return "Wash thoroughly before consuming"
        elif safety_score >= 40:
            return "Inspect carefully, wash thoroughly, cook if possible"
        else:
            return "⚠️ DISCARD - Safety concerns detected"


class IntegratedVisualICPMSAnalyzer:
    """
    Integrated system combining visual analysis with ICP-MS measurements
    for comprehensive food safety and quality assessment.
    """
    
    def __init__(self):
        self.visual_predictor = VisualAtomicPredictor()
        self.safety_analyzer = VisualSafetyAnalyzer()
    
    def analyze_food_comprehensive(
        self,
        food_name: str,
        visual_features: VisualFeatures,
        icpms_measurements: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Complete analysis combining visual and ICP-MS data.
        
        Returns comprehensive report with:
        - Visual feature analysis
        - Element predictions from visual features
        - ICP-MS validation (if provided)
        - Safety assessment
        - Recommendations
        """
        
        # 1. Predict elements from visual features
        element_predictions = self.visual_predictor.predict_elements_from_visual(
            visual_features, food_name
        )
        
        # 2. Validate with ICP-MS if available
        if icpms_measurements:
            element_predictions = self.visual_predictor.validate_with_icpms(
                element_predictions, icpms_measurements
            )
        
        # 3. Assess safety
        safety_assessment = self.safety_analyzer.assess_visual_safety(
            visual_features, element_predictions
        )
        
        # 4. Generate report
        report = {
            'food_name': food_name,
            'analysis_timestamp': visual_features.analysis_timestamp,
            'visual_features': {
                'shininess_index': visual_features.shininess_index,
                'reflection_intensity': visual_features.reflection_intensity,
                'surface_type': visual_features.surface_type.value,
                'color_profile': visual_features.color_profile.value,
                'texture_roughness': visual_features.texture_roughness,
                'moisture_appearance': visual_features.moisture_appearance,
                'freshness_indicators': {
                    'wilting_score': visual_features.wilting_score,
                    'browning_score': visual_features.browning_score,
                    'spots_or_blemishes': visual_features.spots_or_blemishes
                },
                'warning_flags': {
                    'crystalline_structures': visual_features.crystalline_structures,
                    'oily_film': visual_features.oily_film,
                    'dust_or_residue': visual_features.dust_or_residue
                }
            },
            'element_predictions': [
                {
                    'element': pred.element_symbol,
                    'predicted_ppm': round(pred.predicted_ppm, 2),
                    'confidence': round(pred.confidence, 2),
                    'prediction_basis': pred.prediction_basis,
                    'icpms_validated': pred.icpms_validated,
                    'actual_ppm': round(pred.icpms_actual_ppm, 2) if pred.icpms_actual_ppm else None,
                    'prediction_error_%': round(pred.prediction_error, 1) if pred.prediction_error else None
                }
                for pred in element_predictions
            ],
            'safety_assessment': {
                'is_safe': safety_assessment.is_safe,
                'safety_score': round(safety_assessment.safety_score, 1),
                'freshness_rating': safety_assessment.freshness_rating,
                'recommended_action': safety_assessment.recommended_action,
                'risks': {
                    'heavy_metal': safety_assessment.heavy_metal_risk,
                    'pesticide_residue': safety_assessment.pesticide_residue_risk,
                    'microbial': safety_assessment.microbial_risk,
                    'spoilage': safety_assessment.spoilage_risk
                },
                'warnings': safety_assessment.visual_warnings,
                'contamination_indicators': safety_assessment.contamination_indicators
            },
            'icpms_validation_status': 'validated' if icpms_measurements else 'visual_prediction_only',
            'recommendation': self._generate_recommendation(safety_assessment, element_predictions, icpms_measurements)
        }
        
        return report
    
    def _generate_recommendation(
        self,
        safety: VisualSafetyAssessment,
        predictions: List[ElementPrediction],
        has_icpms: bool
    ) -> str:
        """Generate actionable recommendation"""
        if not has_icpms and safety.heavy_metal_risk in ['Medium', 'High']:
            return "⚠️ URGENT: ICP-MS testing required before consumption due to contamination indicators"
        elif not has_icpms and len(safety.contamination_indicators) > 0:
            return "Recommended: Perform ICP-MS analysis to confirm safety"
        elif safety.is_safe and safety.safety_score >= 80:
            return "✅ Safe to consume - Visual and compositional analysis indicate good quality"
        elif safety.is_safe:
            return "⚠️ Safe to consume with precautions: " + safety.recommended_action
        else:
            return "❌ NOT SAFE: " + safety.recommended_action


# Test functions
def test_visual_atomic_analyzer():
    """Test the visual-atomic analysis system"""
    
    print("=" * 80)
    print("VISUAL-ATOMIC FOOD ANALYZER TEST")
    print("=" * 80)
    
    analyzer = IntegratedVisualICPMSAnalyzer()
    
    # Test 1: Fresh spinach with high shine
    print("\n### TEST 1: Fresh Spinach (Visual + ICP-MS) ###\n")
    
    spinach_visual = VisualFeatures(
        shininess_index=75.0,
        reflection_intensity=80.0,
        specular_highlights=12,
        surface_type=SurfaceType.GLOSSY,
        texture_roughness=45.0,
        moisture_appearance=85.0,
        color_profile=ColorProfile.DEEP_GREEN,
        rgb_values=(34, 139, 34),
        color_uniformity=85.0,
        brightness=60.0,
        saturation=90.0,
        wilting_score=5.0,
        browning_score=2.0,
        spots_or_blemishes=1,
        size_mm=150.0,
        shape_regularity=70.0,
        translucency=20.0,
        crystalline_structures=False,
        oily_film=False,
        dust_or_residue=False
    )
    
    spinach_icpms = {
        'Ca': 990.0, 'Fe': 27.0, 'Mg': 790.0, 'K': 5580.0,
        'Zn': 5.3, 'P': 490.0, 'Pb': 0.02, 'Cd': 0.01
    }
    
    report1 = analyzer.analyze_food_comprehensive('Spinach', spinach_visual, spinach_icpms)
    print(json.dumps(report1, indent=2))
    
    # Test 2: Suspicious apple with metallic sheen (no ICP-MS)
    print("\n\n### TEST 2: Apple with Metallic Sheen (Visual Only - WARNING) ###\n")
    
    apple_visual = VisualFeatures(
        shininess_index=85.0,
        reflection_intensity=90.0,
        specular_highlights=20,
        surface_type=SurfaceType.METALLIC,  # ⚠️ Warning sign
        texture_roughness=15.0,
        moisture_appearance=70.0,
        color_profile=ColorProfile.DISCOLORED,  # ⚠️ Warning sign
        rgb_values=(180, 150, 120),  # Abnormal color
        color_uniformity=60.0,
        brightness=75.0,
        saturation=40.0,
        wilting_score=10.0,
        browning_score=15.0,
        spots_or_blemishes=8,
        size_mm=80.0,
        shape_regularity=85.0,
        translucency=10.0,
        crystalline_structures=False,
        oily_film=True,
        dust_or_residue=True
    )
    
    report2 = analyzer.analyze_food_comprehensive('Apple', apple_visual, None)
    print(json.dumps(report2, indent=2))
    
    # Test 3: Aged broccoli
    print("\n\n### TEST 3: Aged Broccoli (Poor Freshness) ###\n")
    
    broccoli_visual = VisualFeatures(
        shininess_index=30.0,
        reflection_intensity=25.0,
        specular_highlights=2,
        surface_type=SurfaceType.ROUGH,
        texture_roughness=80.0,
        moisture_appearance=25.0,
        color_profile=ColorProfile.BROWN,  # Oxidized
        rgb_values=(100, 90, 50),
        color_uniformity=50.0,
        brightness=35.0,
        saturation=30.0,
        wilting_score=65.0,
        browning_score=55.0,
        spots_or_blemishes=18,
        size_mm=120.0,
        shape_regularity=60.0,
        translucency=5.0,
        crystalline_structures=False,
        oily_film=False,
        dust_or_residue=False
    )
    
    report3 = analyzer.analyze_food_comprehensive('Broccoli', broccoli_visual, None)
    print(json.dumps(report3, indent=2))
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_visual_atomic_analyzer()
