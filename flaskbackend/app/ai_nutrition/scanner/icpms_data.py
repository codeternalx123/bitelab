"""
ICP-MS Data Integration and Management
======================================

This module handles integration with ICP-MS (Inductively Coupled Plasma Mass 
Spectrometry) datasets for training and calibration of atomic composition models.

Data Sources:
- FDA Total Diet Study (TDS)
- EFSA Comprehensive Food Consumption Database
- USDA FoodData Central (minerals/trace elements)
- Custom lab ICP-MS analysis results

Features:
- Data loader for ICP-MS CSV/JSON formats
- Data validation and quality control
- Calibration curve generation per element
- Train/test split with stratification
- Data augmentation for low-sample elements

Author: AI Nutrition Team
Version: 0.1.0-dev
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import logging
from datetime import datetime
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional imports
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("pandas not available - limited data processing")
    PANDAS_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available - limited preprocessing")
    SKLEARN_AVAILABLE = False


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class DataSource(Enum):
    """ICP-MS data source types"""
    FDA_TDS = "fda_total_diet_study"
    EFSA = "efsa_food_database"
    USDA = "usda_fooddata_central"
    CUSTOM_LAB = "custom_laboratory"
    LITERATURE = "scientific_literature"


class QualityFlag(Enum):
    """Data quality indicators"""
    EXCELLENT = "excellent"  # Lab-certified, full metadata
    GOOD = "good"  # Standard ICP-MS, basic metadata
    MODERATE = "moderate"  # Older data, incomplete metadata
    POOR = "poor"  # Questionable or missing critical info
    REJECTED = "rejected"  # Failed validation


@dataclass
class ICPMSSample:
    """Single ICP-MS analysis sample"""
    sample_id: str
    food_name: str
    food_category: str
    image_path: Optional[str]  # path to corresponding food image
    weight_grams: float
    
    # Elemental composition (mg/kg)
    elements: Dict[str, float]
    
    # Metadata
    source: DataSource
    analysis_date: datetime
    lab_name: Optional[str] = None
    instrument_model: Optional[str] = None
    detection_limits: Dict[str, float] = field(default_factory=dict)  # LOD per element
    
    # Food-specific metadata
    preparation_method: Optional[str] = None  # raw, cooked, fried, etc.
    origin: Optional[str] = None  # country/region
    organic: Optional[bool] = None
    brand: Optional[str] = None
    
    # Quality control
    quality_flag: QualityFlag = QualityFlag.GOOD
    validation_notes: str = ""
    
    def __post_init__(self):
        """Validation"""
        if self.weight_grams <= 0:
            raise ValueError(f"Invalid weight: {self.weight_grams}")
        if not self.elements:
            raise ValueError("No elemental data provided")
        
        # Check for negative concentrations
        for element, conc in self.elements.items():
            if conc < 0:
                logger.warning(f"Negative concentration for {element} in {self.sample_id}: {conc}")
                self.quality_flag = QualityFlag.POOR
    
    def get_element(self, symbol: str) -> Optional[float]:
        """Get concentration for specific element"""
        return self.elements.get(symbol)
    
    def is_below_detection_limit(self, symbol: str) -> bool:
        """Check if element is below detection limit"""
        conc = self.get_element(symbol)
        lod = self.detection_limits.get(symbol)
        
        if conc is None or lod is None:
            return False
        
        return conc < lod
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "sample_id": self.sample_id,
            "food_name": self.food_name,
            "food_category": self.food_category,
            "image_path": self.image_path,
            "weight_grams": self.weight_grams,
            "elements": self.elements,
            "source": self.source.value,
            "analysis_date": self.analysis_date.isoformat(),
            "lab_name": self.lab_name,
            "preparation_method": self.preparation_method,
            "origin": self.origin,
            "organic": self.organic,
            "brand": self.brand,
            "quality_flag": self.quality_flag.value,
        }


@dataclass
class ICPMSDataset:
    """Collection of ICP-MS samples"""
    samples: List[ICPMSSample]
    name: str
    version: str = "1.0.0"
    created_date: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def filter_by_quality(self, min_quality: QualityFlag = QualityFlag.MODERATE) -> 'ICPMSDataset':
        """Filter samples by minimum quality"""
        quality_order = {
            QualityFlag.EXCELLENT: 4,
            QualityFlag.GOOD: 3,
            QualityFlag.MODERATE: 2,
            QualityFlag.POOR: 1,
            QualityFlag.REJECTED: 0
        }
        
        min_score = quality_order[min_quality]
        filtered = [s for s in self.samples if quality_order[s.quality_flag] >= min_score]
        
        return ICPMSDataset(
            samples=filtered,
            name=f"{self.name}_filtered",
            version=self.version,
            metadata={**self.metadata, "filter": f"quality>={min_quality.value}"}
        )
    
    def filter_by_elements(self, required_elements: List[str]) -> 'ICPMSDataset':
        """Filter samples that have all required elements"""
        filtered = [
            s for s in self.samples
            if all(elem in s.elements for elem in required_elements)
        ]
        
        return ICPMSDataset(
            samples=filtered,
            name=f"{self.name}_with_elements",
            version=self.version,
            metadata={**self.metadata, "required_elements": required_elements}
        )
    
    def get_element_statistics(self, element: str) -> Dict:
        """Get statistics for specific element across dataset"""
        values = [s.get_element(element) for s in self.samples if s.get_element(element) is not None]
        
        if not values:
            return {"count": 0}
        
        return {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75),
        }
    
    def get_available_elements(self) -> Set[str]:
        """Get set of all elements present in dataset"""
        elements = set()
        for sample in self.samples:
            elements.update(sample.elements.keys())
        return elements
    
    def summary(self) -> str:
        """Generate dataset summary"""
        lines = [
            f"ICP-MS Dataset: {self.name} (v{self.version})",
            f"Total samples: {len(self.samples)}",
            f"Created: {self.created_date.strftime('%Y-%m-%d')}",
            "",
            "Quality Distribution:",
        ]
        
        quality_counts = {}
        for sample in self.samples:
            q = sample.quality_flag.value
            quality_counts[q] = quality_counts.get(q, 0) + 1
        
        for quality, count in sorted(quality_counts.items()):
            lines.append(f"  {quality}: {count}")
        
        lines.append("")
        lines.append(f"Available elements: {len(self.get_available_elements())}")
        
        return "\n".join(lines)


# ============================================================================
# DATA LOADERS
# ============================================================================

class ICPMSDataLoader:
    """Load ICP-MS data from various formats"""
    
    @staticmethod
    def load_from_csv(csv_path: str, source: DataSource = DataSource.CUSTOM_LAB) -> ICPMSDataset:
        """
        Load ICP-MS data from CSV file
        
        Expected columns:
        - sample_id, food_name, food_category, weight_grams
        - Element columns: Fe, Zn, Cu, Pb, Cd, As, etc. (mg/kg)
        - Optional: image_path, preparation_method, origin, brand
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for CSV loading")
        
        df = pd.read_csv(csv_path)
        
        # Identify element columns
        known_elements = ["Fe", "Zn", "Cu", "Se", "Ca", "Mg", "Na", "K", "P",
                         "Mn", "Cr", "Mo", "I", "Pb", "Cd", "As", "Hg",
                         "C", "N", "O", "S"]
        element_columns = [col for col in df.columns if col in known_elements]
        
        samples = []
        for idx, row in df.iterrows():
            # Extract elements
            elements = {}
            for elem in element_columns:
                value = row[elem]
                if pd.notna(value):
                    elements[elem] = float(value)
            
            # Create sample
            sample = ICPMSSample(
                sample_id=str(row.get('sample_id', f"sample_{idx}")),
                food_name=str(row['food_name']),
                food_category=str(row['food_category']),
                image_path=str(row['image_path']) if 'image_path' in row and pd.notna(row['image_path']) else None,
                weight_grams=float(row['weight_grams']),
                elements=elements,
                source=source,
                analysis_date=datetime.now(),  # default
                preparation_method=str(row['preparation_method']) if 'preparation_method' in row and pd.notna(row['preparation_method']) else None,
                origin=str(row['origin']) if 'origin' in row and pd.notna(row['origin']) else None,
                brand=str(row['brand']) if 'brand' in row and pd.notna(row['brand']) else None,
            )
            
            samples.append(sample)
        
        return ICPMSDataset(
            samples=samples,
            name=Path(csv_path).stem,
            metadata={"source_file": csv_path}
        )
    
    @staticmethod
    def load_from_json(json_path: str) -> ICPMSDataset:
        """Load ICP-MS dataset from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        samples = []
        for sample_data in data.get('samples', []):
            sample = ICPMSSample(
                sample_id=sample_data['sample_id'],
                food_name=sample_data['food_name'],
                food_category=sample_data['food_category'],
                image_path=sample_data.get('image_path'),
                weight_grams=sample_data['weight_grams'],
                elements=sample_data['elements'],
                source=DataSource(sample_data.get('source', 'custom_laboratory')),
                analysis_date=datetime.fromisoformat(sample_data.get('analysis_date', datetime.now().isoformat())),
                lab_name=sample_data.get('lab_name'),
                preparation_method=sample_data.get('preparation_method'),
                origin=sample_data.get('origin'),
                organic=sample_data.get('organic'),
                brand=sample_data.get('brand'),
                quality_flag=QualityFlag(sample_data.get('quality_flag', 'good')),
            )
            samples.append(sample)
        
        return ICPMSDataset(
            samples=samples,
            name=data.get('name', Path(json_path).stem),
            version=data.get('version', '1.0.0'),
            metadata=data.get('metadata', {})
        )
    
    @staticmethod
    def save_to_json(dataset: ICPMSDataset, json_path: str):
        """Save dataset to JSON file"""
        data = {
            "name": dataset.name,
            "version": dataset.version,
            "created_date": dataset.created_date.isoformat(),
            "metadata": dataset.metadata,
            "samples": [sample.to_dict() for sample in dataset.samples]
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(dataset.samples)} samples to {json_path}")


# ============================================================================
# CALIBRATION CURVES
# ============================================================================

@dataclass
class CalibrationCurve:
    """Calibration curve for an element"""
    element: str
    slope: float
    intercept: float
    r_squared: float
    n_points: int
    concentration_range: Tuple[float, float]
    
    def predict(self, instrument_signal: float) -> float:
        """Predict concentration from instrument signal"""
        return self.slope * instrument_signal + self.intercept
    
    def is_in_range(self, concentration: float) -> bool:
        """Check if concentration is within calibration range"""
        return self.concentration_range[0] <= concentration <= self.concentration_range[1]


class CalibrationManager:
    """Manage calibration curves for all elements"""
    
    def __init__(self):
        self.curves: Dict[str, CalibrationCurve] = {}
    
    def generate_curve(self, element: str, 
                      concentrations: np.ndarray,
                      signals: np.ndarray) -> CalibrationCurve:
        """
        Generate calibration curve from standard measurements
        
        Args:
            element: Element symbol
            concentrations: Known concentrations (mg/kg)
            signals: Instrument signals (counts/sec or similar)
        
        Returns:
            CalibrationCurve object
        """
        if len(concentrations) != len(signals):
            raise ValueError("Concentrations and signals must have same length")
        
        if len(concentrations) < 3:
            raise ValueError("Need at least 3 calibration points")
        
        # Linear regression
        slope, intercept = np.polyfit(signals, concentrations, 1)
        
        # R-squared
        predictions = slope * signals + intercept
        ss_res = np.sum((concentrations - predictions) ** 2)
        ss_tot = np.sum((concentrations - concentrations.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        curve = CalibrationCurve(
            element=element,
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            n_points=len(concentrations),
            concentration_range=(concentrations.min(), concentrations.max())
        )
        
        self.curves[element] = curve
        
        logger.info(f"Calibration curve for {element}: R² = {r_squared:.4f}")
        
        return curve
    
    def validate_curve(self, element: str, 
                      test_concentrations: np.ndarray,
                      test_signals: np.ndarray) -> Dict:
        """Validate calibration curve with test samples"""
        if element not in self.curves:
            raise ValueError(f"No calibration curve for {element}")
        
        curve = self.curves[element]
        
        # Predict
        predictions = curve.slope * test_signals + curve.intercept
        
        # Metrics
        errors = predictions - test_concentrations
        mae = np.abs(errors).mean()
        rmse = np.sqrt((errors ** 2).mean())
        mape = np.abs(errors / (test_concentrations + 1e-6)).mean() * 100
        
        return {
            "element": element,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "within_10_percent": np.sum(np.abs(errors / test_concentrations) < 0.1) / len(errors) * 100
        }
    
    def save(self, path: str):
        """Save calibration curves to file"""
        data = {
            element: {
                "slope": curve.slope,
                "intercept": curve.intercept,
                "r_squared": curve.r_squared,
                "n_points": curve.n_points,
                "concentration_range": curve.concentration_range,
            }
            for element, curve in self.curves.items()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load calibration curves from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        for element, curve_data in data.items():
            self.curves[element] = CalibrationCurve(
                element=element,
                slope=curve_data['slope'],
                intercept=curve_data['intercept'],
                r_squared=curve_data['r_squared'],
                n_points=curve_data['n_points'],
                concentration_range=tuple(curve_data['concentration_range'])
            )


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class ICPMSAugmenter:
    """Data augmentation for low-sample elements"""
    
    @staticmethod
    def add_gaussian_noise(dataset: ICPMSDataset, 
                          noise_level: float = 0.05) -> ICPMSDataset:
        """
        Add Gaussian noise to create synthetic samples
        
        Args:
            dataset: Original dataset
            noise_level: Relative noise level (e.g., 0.05 = 5%)
        
        Returns:
            Augmented dataset (original + noisy copies)
        """
        augmented_samples = []
        
        for sample in dataset.samples:
            # Original sample
            augmented_samples.append(sample)
            
            # Create noisy copy
            noisy_elements = {}
            for element, conc in sample.elements.items():
                noise = np.random.normal(0, conc * noise_level)
                noisy_conc = max(0, conc + noise)
                noisy_elements[element] = noisy_conc
            
            noisy_sample = ICPMSSample(
                sample_id=f"{sample.sample_id}_aug",
                food_name=sample.food_name,
                food_category=sample.food_category,
                image_path=sample.image_path,
                weight_grams=sample.weight_grams,
                elements=noisy_elements,
                source=sample.source,
                analysis_date=sample.analysis_date,
                lab_name=sample.lab_name,
                preparation_method=sample.preparation_method,
                origin=sample.origin,
                organic=sample.organic,
                brand=sample.brand,
                quality_flag=QualityFlag.MODERATE,  # lower quality for synthetic
                validation_notes="Augmented with Gaussian noise"
            )
            
            augmented_samples.append(noisy_sample)
        
        return ICPMSDataset(
            samples=augmented_samples,
            name=f"{dataset.name}_augmented",
            version=dataset.version,
            metadata={**dataset.metadata, "augmentation": f"gaussian_noise_{noise_level}"}
        )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create synthetic ICP-MS dataset
    
    samples = []
    food_types = ["spinach", "salmon", "rice", "apple", "beef"]
    
    for i, food in enumerate(food_types):
        elements = {
            "Fe": np.random.uniform(10, 100),
            "Zn": np.random.uniform(5, 50),
            "Cu": np.random.uniform(1, 10),
            "Pb": np.random.uniform(0.01, 0.1),
            "Cd": np.random.uniform(0.001, 0.05),
            "Ca": np.random.uniform(100, 1000),
        }
        
        sample = ICPMSSample(
            sample_id=f"TEST_{i:03d}",
            food_name=food,
            food_category="test_category",
            image_path=f"images/{food}.jpg",
            weight_grams=100.0,
            elements=elements,
            source=DataSource.CUSTOM_LAB,
            analysis_date=datetime.now(),
            quality_flag=QualityFlag.GOOD
        )
        
        samples.append(sample)
    
    # Create dataset
    dataset = ICPMSDataset(samples=samples, name="test_dataset")
    
    print(dataset.summary())
    print()
    
    # Element statistics
    for element in ["Fe", "Pb"]:
        stats = dataset.get_element_statistics(element)
        print(f"{element} statistics:")
        print(f"  Mean: {stats['mean']:.2f} mg/kg")
        print(f"  Std: {stats['std']:.2f} mg/kg")
        print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f} mg/kg")
        print()
    
    # Save to JSON
    ICPMSDataLoader.save_to_json(dataset, "test_icpms_dataset.json")
    
    # Calibration example
    cal_manager = CalibrationManager()
    
    # Generate calibration curve for Fe
    standards_conc = np.array([10, 50, 100, 200, 500])  # mg/kg
    standards_signal = np.array([1000, 5000, 10000, 20000, 50000])  # counts/sec
    
    curve = cal_manager.generate_curve("Fe", standards_conc, standards_signal)
    print(f"Fe calibration: {curve.slope:.2f} * signal + {curve.intercept:.2f}")
    print(f"R² = {curve.r_squared:.4f}")
