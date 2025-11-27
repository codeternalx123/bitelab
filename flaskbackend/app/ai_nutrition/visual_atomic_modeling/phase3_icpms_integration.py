"""
PHASE 3: ICP-MS Integration & Ground Truth Database
====================================================

This module implements the ICP-MS (Inductively Coupled Plasma Mass Spectrometry)
database integration layer, providing ground truth atomic composition data for:
- Training deep learning models
- Validating simulated annealing results  
- Calibrating predictions
- Data augmentation

The ICP-MS database contains laboratory-measured elemental concentrations for
50,000+ food samples across different:
- Food types (proteins, grains, vegetables, etc.)
- Preparation methods (raw, cooked, processed)
- Geographic origins
- Storage conditions

This serves as the absolute "ground truth" anchor for the entire system.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path
import json
import sqlite3
import pickle
from scipy.spatial import KDTree
from scipy.stats import norm, t as t_dist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import h5py
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreparationMethod(Enum):
    """Food preparation methods in database"""
    RAW = "raw"
    BOILED = "boiled"
    STEAMED = "steamed"
    FRIED = "fried"
    GRILLED = "grilled"
    BAKED = "baked"
    ROASTED = "roasted"
    DEEP_FRIED = "deep_fried"
    PRESSURE_COOKED = "pressure_cooked"
    FERMENTED = "fermented"
    SMOKED = "smoked"
    DRIED = "dried"
    FROZEN = "frozen"
    CANNED = "canned"


class GeographicOrigin(Enum):
    """Geographic origins affecting elemental composition"""
    NORTH_AMERICA = "north_america"
    SOUTH_AMERICA = "south_america"
    EUROPE = "europe"
    AFRICA = "africa"
    ASIA = "asia"
    OCEANIA = "oceania"
    MIDDLE_EAST = "middle_east"


@dataclass
class ICPMSMeasurement:
    """Single ICP-MS laboratory measurement"""
    sample_id: str
    food_name: str
    food_category: str
    preparation_method: PreparationMethod
    geographic_origin: GeographicOrigin
    
    # Elemental composition (ppm or mg/100g)
    composition: Dict[str, float]
    
    # Measurement uncertainty (95% confidence intervals)
    uncertainties: Dict[str, Tuple[float, float]]
    
    # Metadata
    measurement_date: str
    laboratory: str
    analysis_method: str  # ICP-MS, ICP-OES, AAS, etc.
    sample_preparation: str
    
    # Quality metrics
    detection_limits: Dict[str, float]  # Minimum detectable concentration
    recovery_rates: Dict[str, float]  # Spike recovery %
    rsd_percent: Dict[str, float]  # Relative standard deviation
    
    # Nutritional context
    moisture_content: float  # %
    ash_content: float  # %
    protein_content: float  # g/100g
    fat_content: float  # g/100g
    
    # Additional properties
    cooking_temperature: Optional[float] = None  # °C
    cooking_duration: Optional[int] = None  # minutes
    storage_conditions: str = ""
    harvest_season: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['preparation_method'] = self.preparation_method.value
        data['geographic_origin'] = self.geographic_origin.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ICPMSMeasurement':
        """Create from dictionary"""
        data['preparation_method'] = PreparationMethod(data['preparation_method'])
        data['geographic_origin'] = GeographicOrigin(data['geographic_origin'])
        return cls(**data)


@dataclass
class DatabaseQuery:
    """Query parameters for database search"""
    food_names: Optional[List[str]] = None
    food_categories: Optional[List[str]] = None
    preparation_methods: Optional[List[PreparationMethod]] = None
    geographic_origins: Optional[List[GeographicOrigin]] = None
    elements: Optional[List[str]] = None
    
    # Concentration filters
    element_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    
    # Similarity search
    reference_composition: Optional[Dict[str, float]] = None
    max_distance: Optional[float] = None
    
    # Pagination
    limit: int = 100
    offset: int = 0


class ICPMSDatabaseConnector:
    """
    Connects to and queries the ICP-MS ground truth database.
    
    The database contains 50,000+ entries with laboratory-measured
    elemental concentrations serving as absolute ground truth.
    
    Data Sources:
    - FDA Total Diet Study
    - EFSA Comprehensive Database  
    - USDA FoodData Central
    - Custom laboratory measurements
    - Published research papers
    """
    
    def __init__(
        self,
        database_path: str = "data/icpms_database.db",
        cache_size: int = 1000
    ):
        self.database_path = Path(database_path)
        self.cache_size = cache_size
        
        # Initialize connection
        self.conn = None
        self.cursor = None
        
        # In-memory cache
        self.cache: Dict[str, ICPMSMeasurement] = {}
        self.cache_keys: List[str] = []
        
        # KD-Tree for fast similarity search
        self.kdtree: Optional[KDTree] = None
        self.kdtree_samples: Optional[List[str]] = None
        self.element_order: List[str] = []
        
        # Statistics
        self.total_entries = 0
        self.element_statistics: Dict[str, Dict[str, float]] = {}
        
        # Connect to database
        self._connect()
        self._initialize_kdtree()
        self._compute_statistics()
        
        logger.info(f"Connected to ICP-MS database: {self.total_entries} entries")
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(str(self.database_path))
            self.cursor = self.conn.cursor()
            
            # Get total entries
            self.cursor.execute("SELECT COUNT(*) FROM measurements")
            self.total_entries = self.cursor.fetchone()[0]
            
            logger.info(f"Database connected: {self.total_entries} measurements")
            
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            # Create empty database if doesn't exist
            self._create_database()
    
    def _create_database(self):
        """Create database schema"""
        self.conn = sqlite3.connect(str(self.database_path))
        self.cursor = self.conn.cursor()
        
        # Create measurements table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                sample_id TEXT PRIMARY KEY,
                food_name TEXT NOT NULL,
                food_category TEXT NOT NULL,
                preparation_method TEXT NOT NULL,
                geographic_origin TEXT NOT NULL,
                composition TEXT NOT NULL,
                uncertainties TEXT NOT NULL,
                measurement_date TEXT,
                laboratory TEXT,
                analysis_method TEXT,
                sample_preparation TEXT,
                detection_limits TEXT,
                recovery_rates TEXT,
                rsd_percent TEXT,
                moisture_content REAL,
                ash_content REAL,
                protein_content REAL,
                fat_content REAL,
                cooking_temperature REAL,
                cooking_duration INTEGER,
                storage_conditions TEXT,
                harvest_season TEXT
            )
        """)
        
        # Create indices for faster queries
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_food_name
            ON measurements(food_name)
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_food_category
            ON measurements(food_category)
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_preparation
            ON measurements(preparation_method)
        """)
        
        self.conn.commit()
        logger.info("Database schema created")
    
    def _initialize_kdtree(self):
        """Initialize KD-Tree for fast similarity search"""
        logger.info("Building KD-Tree for similarity search...")
        
        # Get all measurements
        self.cursor.execute("SELECT sample_id, composition FROM measurements LIMIT 10000")
        rows = self.cursor.fetchall()
        
        if not rows:
            logger.warning("No data for KD-Tree initialization")
            return
        
        # Extract compositions
        compositions = []
        sample_ids = []
        
        # Get consistent element order
        first_comp = json.loads(rows[0][1])
        self.element_order = sorted(first_comp.keys())
        
        for sample_id, comp_json in rows:
            comp = json.loads(comp_json)
            vector = [comp.get(e, 0.0) for e in self.element_order]
            compositions.append(vector)
            sample_ids.append(sample_id)
        
        # Build KD-Tree
        self.kdtree = KDTree(np.array(compositions))
        self.kdtree_samples = sample_ids
        
        logger.info(f"KD-Tree built with {len(sample_ids)} samples")
    
    def _compute_statistics(self):
        """Compute database statistics for each element"""
        logger.info("Computing element statistics...")
        
        # Get all compositions
        self.cursor.execute("SELECT composition FROM measurements")
        rows = self.cursor.fetchall()
        
        if not rows:
            return
        
        # Aggregate by element
        element_values: Dict[str, List[float]] = {}
        
        for (comp_json,) in rows:
            comp = json.loads(comp_json)
            for element, value in comp.items():
                if element not in element_values:
                    element_values[element] = []
                element_values[element].append(value)
        
        # Compute statistics
        for element, values in element_values.items():
            values_array = np.array(values)
            
            self.element_statistics[element] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array)),
                'p25': float(np.percentile(values_array, 25)),
                'p75': float(np.percentile(values_array, 75)),
                'p95': float(np.percentile(values_array, 95)),
                'p99': float(np.percentile(values_array, 99)),
            }
        
        logger.info(f"Statistics computed for {len(self.element_statistics)} elements")
    
    def query(self, query: DatabaseQuery) -> List[ICPMSMeasurement]:
        """
        Query database with flexible criteria
        
        Args:
            query: DatabaseQuery object with search parameters
            
        Returns:
            List of matching ICPMSMeasurement objects
        """
        # Build SQL query
        sql = "SELECT * FROM measurements WHERE 1=1"
        params = []
        
        if query.food_names:
            placeholders = ','.join('?' * len(query.food_names))
            sql += f" AND food_name IN ({placeholders})"
            params.extend(query.food_names)
        
        if query.food_categories:
            placeholders = ','.join('?' * len(query.food_categories))
            sql += f" AND food_category IN ({placeholders})"
            params.extend(query.food_categories)
        
        if query.preparation_methods:
            methods = [m.value for m in query.preparation_methods]
            placeholders = ','.join('?' * len(methods))
            sql += f" AND preparation_method IN ({placeholders})"
            params.extend(methods)
        
        if query.geographic_origins:
            origins = [o.value for o in query.geographic_origins]
            placeholders = ','.join('?' * len(origins))
            sql += f" AND geographic_origin IN ({placeholders})"
            params.extend(origins)
        
        # Pagination
        sql += f" LIMIT {query.limit} OFFSET {query.offset}"
        
        # Execute query
        self.cursor.execute(sql, params)
        rows = self.cursor.fetchall()
        
        # Convert to ICPMSMeasurement objects
        measurements = []
        for row in rows:
            measurement = self._row_to_measurement(row)
            measurements.append(measurement)
        
        # Additional filtering (element ranges, similarity)
        if query.element_ranges:
            measurements = self._filter_by_element_ranges(
                measurements,
                query.element_ranges
            )
        
        if query.reference_composition and query.max_distance:
            measurements = self._filter_by_similarity(
                measurements,
                query.reference_composition,
                query.max_distance
            )
        
        return measurements
    
    def query_similar(
        self,
        food_type: str,
        cooking_method: str,
        top_k: int = 5
    ) -> List[ICPMSMeasurement]:
        """
        Query for similar food samples
        
        Args:
            food_type: Food name or category
            cooking_method: Preparation method
            top_k: Number of similar samples to return
            
        Returns:
            List of most similar measurements
        """
        query = DatabaseQuery(
            food_names=[food_type],
            limit=top_k
        )
        
        # Try exact match first
        results = self.query(query)
        
        if results:
            return results
        
        # Try category match
        query.food_names = None
        query.food_categories = [self._get_category(food_type)]
        results = self.query(query)
        
        return results[:top_k]
    
    def query_by_composition(
        self,
        reference_composition: Dict[str, float],
        top_k: int = 5,
        max_distance: Optional[float] = None
    ) -> List[ICPMSMeasurement]:
        """
        Find samples with similar elemental composition using KD-Tree
        
        Args:
            reference_composition: Target composition to match
            top_k: Number of nearest neighbors to return
            max_distance: Maximum distance threshold
            
        Returns:
            List of similar measurements
        """
        if self.kdtree is None:
            logger.warning("KD-Tree not initialized")
            return []
        
        # Convert composition to vector
        query_vector = np.array([
            reference_composition.get(e, 0.0)
            for e in self.element_order
        ])
        
        # Query KD-Tree
        distances, indices = self.kdtree.query(query_vector, k=top_k)
        
        # Filter by max distance if specified
        if max_distance is not None:
            mask = distances <= max_distance
            indices = indices[mask]
            distances = distances[mask]
        
        # Retrieve measurements
        measurements = []
        for idx, dist in zip(indices, distances):
            sample_id = self.kdtree_samples[idx]
            measurement = self.get_by_id(sample_id)
            if measurement:
                measurements.append(measurement)
        
        return measurements
    
    def get_by_id(self, sample_id: str) -> Optional[ICPMSMeasurement]:
        """Get measurement by sample ID"""
        # Check cache first
        if sample_id in self.cache:
            return self.cache[sample_id]
        
        # Query database
        self.cursor.execute(
            "SELECT * FROM measurements WHERE sample_id = ?",
            (sample_id,)
        )
        row = self.cursor.fetchone()
        
        if row is None:
            return None
        
        measurement = self._row_to_measurement(row)
        
        # Add to cache
        self._cache_measurement(sample_id, measurement)
        
        return measurement
    
    def insert(self, measurement: ICPMSMeasurement):
        """Insert new measurement into database"""
        data = measurement.to_dict()
        
        # Convert complex types to JSON
        data['composition'] = json.dumps(data['composition'])
        data['uncertainties'] = json.dumps(data['uncertainties'])
        data['detection_limits'] = json.dumps(data['detection_limits'])
        data['recovery_rates'] = json.dumps(data['recovery_rates'])
        data['rsd_percent'] = json.dumps(data['rsd_percent'])
        
        # Prepare SQL
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        sql = f"INSERT OR REPLACE INTO measurements ({columns}) VALUES ({placeholders})"
        
        # Execute
        self.cursor.execute(sql, list(data.values()))
        self.conn.commit()
        
        self.total_entries += 1
        
        logger.debug(f"Inserted measurement: {measurement.sample_id}")
    
    def batch_insert(self, measurements: List[ICPMSMeasurement]):
        """Batch insert multiple measurements"""
        logger.info(f"Batch inserting {len(measurements)} measurements...")
        
        for i, measurement in enumerate(measurements):
            self.insert(measurement)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Inserted {i + 1}/{len(measurements)} measurements")
        
        logger.info("Batch insert complete")
        
        # Rebuild KD-Tree
        self._initialize_kdtree()
        self._compute_statistics()
    
    def get_element_statistics(self, element: str) -> Optional[Dict[str, float]]:
        """Get statistics for specific element"""
        return self.element_statistics.get(element)
    
    def get_all_food_names(self) -> List[str]:
        """Get list of all food names in database"""
        self.cursor.execute("SELECT DISTINCT food_name FROM measurements")
        return [row[0] for row in self.cursor.fetchall()]
    
    def get_all_categories(self) -> List[str]:
        """Get list of all food categories"""
        self.cursor.execute("SELECT DISTINCT food_category FROM measurements")
        return [row[0] for row in self.cursor.fetchall()]
    
    def _row_to_measurement(self, row: Tuple) -> ICPMSMeasurement:
        """Convert database row to ICPMSMeasurement"""
        return ICPMSMeasurement(
            sample_id=row[0],
            food_name=row[1],
            food_category=row[2],
            preparation_method=PreparationMethod(row[3]),
            geographic_origin=GeographicOrigin(row[4]),
            composition=json.loads(row[5]),
            uncertainties=json.loads(row[6]),
            measurement_date=row[7],
            laboratory=row[8],
            analysis_method=row[9],
            sample_preparation=row[10],
            detection_limits=json.loads(row[11]),
            recovery_rates=json.loads(row[12]),
            rsd_percent=json.loads(row[13]),
            moisture_content=row[14],
            ash_content=row[15],
            protein_content=row[16],
            fat_content=row[17],
            cooking_temperature=row[18],
            cooking_duration=row[19],
            storage_conditions=row[20],
            harvest_season=row[21]
        )
    
    def _cache_measurement(self, sample_id: str, measurement: ICPMSMeasurement):
        """Add measurement to cache"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = self.cache_keys.pop(0)
            del self.cache[oldest_key]
        
        self.cache[sample_id] = measurement
        self.cache_keys.append(sample_id)
    
    def _filter_by_element_ranges(
        self,
        measurements: List[ICPMSMeasurement],
        element_ranges: Dict[str, Tuple[float, float]]
    ) -> List[ICPMSMeasurement]:
        """Filter measurements by element concentration ranges"""
        filtered = []
        
        for measurement in measurements:
            passes = True
            
            for element, (min_val, max_val) in element_ranges.items():
                conc = measurement.composition.get(element, 0.0)
                if not (min_val <= conc <= max_val):
                    passes = False
                    break
            
            if passes:
                filtered.append(measurement)
        
        return filtered
    
    def _filter_by_similarity(
        self,
        measurements: List[ICPMSMeasurement],
        reference: Dict[str, float],
        max_distance: float
    ) -> List[ICPMSMeasurement]:
        """Filter measurements by composition similarity"""
        filtered = []
        
        for measurement in measurements:
            distance = self._compute_composition_distance(
                measurement.composition,
                reference
            )
            
            if distance <= max_distance:
                filtered.append(measurement)
        
        return filtered
    
    def _compute_composition_distance(
        self,
        comp1: Dict[str, float],
        comp2: Dict[str, float]
    ) -> float:
        """Compute Euclidean distance between compositions"""
        elements = set(comp1.keys()) | set(comp2.keys())
        
        distance = 0.0
        for element in elements:
            v1 = comp1.get(element, 0.0)
            v2 = comp2.get(element, 0.0)
            distance += (v1 - v2) ** 2
        
        return np.sqrt(distance)
    
    def _get_category(self, food_name: str) -> str:
        """Determine food category from name"""
        # Simplified category mapping
        categories = {
            'protein': ['chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'egg'],
            'grain': ['rice', 'wheat', 'bread', 'pasta', 'oat'],
            'vegetable': ['broccoli', 'carrot', 'spinach', 'lettuce', 'tomato'],
            'fruit': ['apple', 'banana', 'orange', 'berry', 'grape'],
            'dairy': ['milk', 'cheese', 'yogurt', 'butter'],
        }
        
        food_lower = food_name.lower()
        for category, keywords in categories.items():
            if any(kw in food_lower for kw in keywords):
                return category
        
        return 'other'
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


class GroundTruthValidator:
    """
    Validates predicted atomic compositions against ICP-MS ground truth.
    
    Provides:
    - Accuracy metrics (MAE, RMSE, R²)
    - Uncertainty quantification
    - Bias detection
    - Outlier identification
    """
    
    def __init__(self, database: ICPMSDatabaseConnector):
        self.database = database
        
        # Validation statistics
        self.validation_history: List[Dict] = []
        
    def validate(
        self,
        predicted: Dict[str, float],
        ground_truth: ICPMSMeasurement,
        elements: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate predicted composition against ground truth
        
        Args:
            predicted: Predicted elemental concentrations
            ground_truth: Ground truth ICPMSMeasurement
            elements: Elements to validate (default: all)
            
        Returns:
            Validation metrics dictionary
        """
        if elements is None:
            elements = list(predicted.keys())
        
        # Compute errors
        errors = {}
        relative_errors = {}
        
        for element in elements:
            pred_val = predicted.get(element, 0.0)
            true_val = ground_truth.composition.get(element, 0.0)
            
            # Absolute error
            abs_error = abs(pred_val - true_val)
            errors[element] = abs_error
            
            # Relative error
            if true_val > 0:
                rel_error = abs_error / true_val
                relative_errors[element] = rel_error
        
        # Compute aggregate metrics
        mae = np.mean(list(errors.values()))
        rmse = np.sqrt(np.mean([e**2 for e in errors.values()]))
        
        # R² score
        true_vals = [ground_truth.composition.get(e, 0.0) for e in elements]
        pred_vals = [predicted.get(e, 0.0) for e in elements]
        r2 = self._compute_r2(true_vals, pred_vals)
        
        # Within uncertainty?
        within_uncertainty = {}
        for element in elements:
            pred_val = predicted.get(element, 0.0)
            if element in ground_truth.uncertainties:
                lower, upper = ground_truth.uncertainties[element]
                within = lower <= pred_val <= upper
                within_uncertainty[element] = within
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'errors': errors,
            'relative_errors': relative_errors,
            'within_uncertainty': within_uncertainty,
            'accuracy_percent': 100 * (1 - mae / (np.mean(true_vals) + 1e-8))
        }
        
        # Record validation
        self.validation_history.append(metrics)
        
        return metrics
    
    def batch_validate(
        self,
        predictions: List[Dict[str, float]],
        ground_truths: List[ICPMSMeasurement]
    ) -> Dict[str, Any]:
        """Validate batch of predictions"""
        all_metrics = []
        
        for pred, truth in zip(predictions, ground_truths):
            metrics = self.validate(pred, truth)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregate = {
            'mean_mae': np.mean([m['mae'] for m in all_metrics]),
            'mean_rmse': np.mean([m['rmse'] for m in all_metrics]),
            'mean_r2': np.mean([m['r2'] for m in all_metrics]),
            'mean_accuracy': np.mean([m['accuracy_percent'] for m in all_metrics]),
            'std_mae': np.std([m['mae'] for m in all_metrics]),
            'num_samples': len(predictions)
        }
        
        return aggregate
    
    def _compute_r2(self, y_true: List[float], y_pred: List[float]) -> float:
        """Compute R² score"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def compute_confidence_interval(
        self,
        predicted: float,
        element: str,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for prediction
        
        Uses historical validation errors to estimate uncertainty
        """
        # Get historical errors for this element
        historical_errors = []
        for record in self.validation_history:
            if element in record['errors']:
                historical_errors.append(record['errors'][element])
        
        if not historical_errors:
            # No history, use conservative estimate
            std = predicted * 0.2  # 20% uncertainty
        else:
            std = np.std(historical_errors)
        
        # Compute confidence interval
        z_score = norm.ppf((1 + confidence_level) / 2)
        margin = z_score * std
        
        return (predicted - margin, predicted + margin)


class ElementalCompositionMapper:
    """
    Maps between different elemental concentration formats and units.
    
    Handles:
    - Unit conversions (ppm ↔ mg/100g ↔ μg/g)
    - Dry weight ↔ Fresh weight
    - Missing value imputation
    - Normalization
    """
    
    UNIT_CONVERSIONS = {
        ('ppm', 'mg/100g'): 0.1,  # 1 ppm = 0.1 mg/100g
        ('mg/100g', 'ppm'): 10.0,
        ('ppm', 'μg/g'): 1.0,
        ('μg/g', 'ppm'): 1.0,
        ('mg/100g', 'μg/g'): 10.0,
        ('μg/g', 'mg/100g'): 0.1,
    }
    
    def __init__(self, database: ICPMSDatabaseConnector):
        self.database = database
        
    def convert_units(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """Convert between concentration units"""
        if from_unit == to_unit:
            return value
        
        conversion_key = (from_unit, to_unit)
        if conversion_key in self.UNIT_CONVERSIONS:
            return value * self.UNIT_CONVERSIONS[conversion_key]
        
        raise ValueError(f"Unknown conversion: {from_unit} -> {to_unit}")
    
    def fresh_to_dry_weight(
        self,
        fresh_weight_conc: float,
        moisture_content: float
    ) -> float:
        """
        Convert fresh weight concentration to dry weight
        
        Args:
            fresh_weight_conc: Concentration on fresh weight basis
            moisture_content: Moisture content (fraction 0-1)
            
        Returns:
            Concentration on dry weight basis
        """
        dry_matter = 1.0 - moisture_content
        if dry_matter <= 0:
            return 0.0
        
        return fresh_weight_conc / dry_matter
    
    def dry_to_fresh_weight(
        self,
        dry_weight_conc: float,
        moisture_content: float
    ) -> float:
        """Convert dry weight concentration to fresh weight"""
        dry_matter = 1.0 - moisture_content
        return dry_weight_conc * dry_matter
    
    def impute_missing_elements(
        self,
        composition: Dict[str, float],
        food_category: str,
        method: str = 'category_median'
    ) -> Dict[str, float]:
        """
        Impute missing element concentrations
        
        Args:
            composition: Current composition (may have missing elements)
            food_category: Food category for context
            method: Imputation method ('category_median', 'zero', 'database_mean')
            
        Returns:
            Complete composition with imputed values
        """
        complete_composition = composition.copy()
        
        # Get all elements from database
        all_elements = set()
        for stats in self.database.element_statistics.values():
            all_elements.update(stats.keys())
        
        if method == 'zero':
            # Impute with zeros
            for element in all_elements:
                if element not in complete_composition:
                    complete_composition[element] = 0.0
        
        elif method == 'database_mean':
            # Impute with database mean
            for element in all_elements:
                if element not in complete_composition:
                    stats = self.database.get_element_statistics(element)
                    if stats:
                        complete_composition[element] = stats['mean']
                    else:
                        complete_composition[element] = 0.0
        
        elif method == 'category_median':
            # Impute with category-specific median
            # Query similar foods
            query = DatabaseQuery(food_categories=[food_category], limit=100)
            similar_foods = self.database.query(query)
            
            # Compute median for each element
            element_values: Dict[str, List[float]] = {}
            for food in similar_foods:
                for element, value in food.composition.items():
                    if element not in element_values:
                        element_values[element] = []
                    element_values[element].append(value)
            
            # Impute missing
            for element in all_elements:
                if element not in complete_composition:
                    if element in element_values:
                        complete_composition[element] = float(np.median(element_values[element]))
                    else:
                        complete_composition[element] = 0.0
        
        return complete_composition
    
    def normalize_composition(
        self,
        composition: Dict[str, float],
        method: str = 'zscore'
    ) -> Dict[str, float]:
        """
        Normalize elemental concentrations
        
        Args:
            composition: Raw composition
            method: Normalization method ('zscore', 'minmax', 'robust')
            
        Returns:
            Normalized composition
        """
        normalized = {}
        
        for element, value in composition.items():
            stats = self.database.get_element_statistics(element)
            
            if stats is None:
                normalized[element] = value
                continue
            
            if method == 'zscore':
                # Z-score normalization
                mean = stats['mean']
                std = stats['std']
                normalized[element] = (value - mean) / (std + 1e-8)
            
            elif method == 'minmax':
                # Min-max normalization
                min_val = stats['min']
                max_val = stats['max']
                normalized[element] = (value - min_val) / (max_val - min_val + 1e-8)
            
            elif method == 'robust':
                # Robust scaling using percentiles
                median = stats['median']
                iqr = stats['p75'] - stats['p25']
                normalized[element] = (value - median) / (iqr + 1e-8)
        
        return normalized


class CalibrationManager:
    """
    Manages model calibration using ICP-MS ground truth data.
    
    Provides:
    - Calibration curve fitting
    - Bias correction
    - Uncertainty calibration
    - Model updating
    """
    
    def __init__(self, database: ICPMSDatabaseConnector):
        self.database = database
        
        # Calibration curves (element -> model)
        self.calibration_curves: Dict[str, Any] = {}
        
        # Bias correction factors
        self.bias_corrections: Dict[str, float] = {}
        
    def calibrate_element(
        self,
        element: str,
        predictions: List[float],
        ground_truth_ids: List[str]
    ):
        """
        Calibrate predictions for specific element
        
        Fits a calibration curve: true_value = f(predicted_value)
        """
        # Get ground truth values
        true_values = []
        for sample_id in ground_truth_ids:
            measurement = self.database.get_by_id(sample_id)
            if measurement and element in measurement.composition:
                true_values.append(measurement.composition[element])
        
        if len(true_values) != len(predictions):
            logger.error("Mismatch between predictions and ground truth")
            return
        
        # Fit linear calibration curve
        predictions = np.array(predictions).reshape(-1, 1)
        true_values = np.array(true_values)
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(predictions, true_values)
        
        self.calibration_curves[element] = model
        
        # Compute bias correction
        bias = np.mean(predictions.flatten() - true_values)
        self.bias_corrections[element] = bias
        
        logger.info(f"Calibrated {element}: slope={model.coef_[0]:.4f}, bias={bias:.4f}")
    
    def apply_calibration(
        self,
        predictions: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply calibration to predictions"""
        calibrated = {}
        
        for element, pred_value in predictions.items():
            if element in self.calibration_curves:
                # Apply calibration curve
                model = self.calibration_curves[element]
                calibrated_value = model.predict([[pred_value]])[0]
                calibrated[element] = float(calibrated_value)
            elif element in self.bias_corrections:
                # Apply bias correction
                calibrated[element] = pred_value - self.bias_corrections[element]
            else:
                # No calibration available
                calibrated[element] = pred_value
        
        return calibrated
    
    def save_calibration(self, filepath: str):
        """Save calibration models"""
        data = {
            'calibration_curves': self.calibration_curves,
            'bias_corrections': self.bias_corrections
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Calibration saved to {filepath}")
    
    def load_calibration(self, filepath: str):
        """Load calibration models"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.calibration_curves = data['calibration_curves']
        self.bias_corrections = data['bias_corrections']
        
        logger.info(f"Calibration loaded from {filepath}")


class DataAugmentationEngine:
    """
    Generates augmented training data from ICP-MS database.
    
    Augmentation strategies:
    - Interpolation between samples
    - Cooking transformation simulation
    - Mixing recipes
    - Noise injection
    """
    
    def __init__(self, database: ICPMSDatabaseConnector):
        self.database = database
        
    def augment_cooking_transformation(
        self,
        raw_measurement: ICPMSMeasurement,
        target_cooking_method: PreparationMethod,
        n_samples: int = 10
    ) -> List[ICPMSMeasurement]:
        """
        Generate synthetic cooked samples from raw measurement
        
        Simulates chemical changes during cooking:
        - Moisture loss
        - Nutrient degradation
        - Concentration changes
        - Maillard reactions
        """
        augmented = []
        
        # Get cooking parameters
        cooking_params = self._get_cooking_parameters(target_cooking_method)
        
        for i in range(n_samples):
            # Clone measurement
            new_measurement = ICPMSMeasurement(
                sample_id=f"{raw_measurement.sample_id}_aug_cook_{i}",
                food_name=raw_measurement.food_name,
                food_category=raw_measurement.food_category,
                preparation_method=target_cooking_method,
                geographic_origin=raw_measurement.geographic_origin,
                composition={},
                uncertainties={},
                measurement_date=datetime.now().isoformat(),
                laboratory="synthetic",
                analysis_method="augmentation",
                sample_preparation="simulated_cooking",
                detection_limits=raw_measurement.detection_limits.copy(),
                recovery_rates=raw_measurement.recovery_rates.copy(),
                rsd_percent=raw_measurement.rsd_percent.copy(),
                moisture_content=raw_measurement.moisture_content,
                ash_content=raw_measurement.ash_content,
                protein_content=raw_measurement.protein_content,
                fat_content=raw_measurement.fat_content,
                cooking_temperature=cooking_params['temperature'],
                cooking_duration=cooking_params['duration']
            )
            
            # Simulate composition changes
            for element, conc in raw_measurement.composition.items():
                # Apply moisture loss concentration effect
                moisture_loss = np.random.uniform(
                    cooking_params['min_moisture_loss'],
                    cooking_params['max_moisture_loss']
                )
                concentration_factor = 1.0 / (1.0 - moisture_loss)
                
                # Apply element-specific degradation/volatilization
                retention_factor = self._get_element_retention(
                    element,
                    target_cooking_method
                )
                
                # Add variability
                noise = np.random.normal(1.0, 0.05)
                
                new_conc = conc * concentration_factor * retention_factor * noise
                new_measurement.composition[element] = max(0.0, new_conc)
                
                # Update uncertainty
                if element in raw_measurement.uncertainties:
                    lower, upper = raw_measurement.uncertainties[element]
                    new_measurement.uncertainties[element] = (
                        lower * concentration_factor * retention_factor,
                        upper * concentration_factor * retention_factor
                    )
            
            # Update moisture content
            new_measurement.moisture_content *= (1.0 - moisture_loss)
            
            augmented.append(new_measurement)
        
        return augmented
    
    def augment_mixing(
        self,
        measurements: List[ICPMSMeasurement],
        n_samples: int = 10
    ) -> List[ICPMSMeasurement]:
        """
        Generate synthetic mixed food samples
        
        Simulates combinations/recipes by mixing elemental compositions
        """
        augmented = []
        
        for i in range(n_samples):
            # Select random subset to mix
            n_ingredients = np.random.randint(2, min(5, len(measurements) + 1))
            selected = np.random.choice(measurements, n_ingredients, replace=False)
            
            # Generate random mixing ratios
            ratios = np.random.dirichlet(np.ones(n_ingredients))
            
            # Create mixed measurement
            mixed = ICPMSMeasurement(
                sample_id=f"mixed_aug_{i}",
                food_name="mixed_dish",
                food_category="mixed",
                preparation_method=PreparationMethod.COOKED,
                geographic_origin=selected[0].geographic_origin,
                composition={},
                uncertainties={},
                measurement_date=datetime.now().isoformat(),
                laboratory="synthetic",
                analysis_method="augmentation",
                sample_preparation="mixing",
                detection_limits={},
                recovery_rates={},
                rsd_percent={},
                moisture_content=0.0,
                ash_content=0.0,
                protein_content=0.0,
                fat_content=0.0
            )
            
            # Mix compositions
            all_elements = set()
            for measurement in selected:
                all_elements.update(measurement.composition.keys())
            
            for element in all_elements:
                mixed_conc = sum(
                    measurement.composition.get(element, 0.0) * ratio
                    for measurement, ratio in zip(selected, ratios)
                )
                mixed.composition[element] = mixed_conc
            
            # Mix nutritional properties
            mixed.moisture_content = sum(
                m.moisture_content * r for m, r in zip(selected, ratios)
            )
            mixed.protein_content = sum(
                m.protein_content * r for m, r in zip(selected, ratios)
            )
            mixed.fat_content = sum(
                m.fat_content * r for m, r in zip(selected, ratios)
            )
            
            augmented.append(mixed)
        
        return augmented
    
    def _get_cooking_parameters(
        self,
        method: PreparationMethod
    ) -> Dict[str, Any]:
        """Get typical cooking parameters for method"""
        parameters = {
            PreparationMethod.BOILED: {
                'temperature': 100,
                'duration': 20,
                'min_moisture_loss': 0.0,
                'max_moisture_loss': 0.1
            },
            PreparationMethod.FRIED: {
                'temperature': 180,
                'duration': 10,
                'min_moisture_loss': 0.2,
                'max_moisture_loss': 0.4
            },
            PreparationMethod.GRILLED: {
                'temperature': 200,
                'duration': 15,
                'min_moisture_loss': 0.15,
                'max_moisture_loss': 0.35
            },
            PreparationMethod.BAKED: {
                'temperature': 175,
                'duration': 30,
                'min_moisture_loss': 0.1,
                'max_moisture_loss': 0.3
            },
        }
        
        return parameters.get(method, {
            'temperature': 150,
            'duration': 15,
            'min_moisture_loss': 0.1,
            'max_moisture_loss': 0.2
        })
    
    def _get_element_retention(
        self,
        element: str,
        cooking_method: PreparationMethod
    ) -> float:
        """
        Get retention factor for element during cooking
        
        Some elements are volatile (Hg, I) or water-soluble (Na, K)
        """
        retention_factors = {
            # Volatile elements (loss during cooking)
            'Hg': 0.7,
            'I': 0.8,
            'Se': 0.85,
            
            # Stable elements (no loss)
            'Fe': 1.0,
            'Zn': 0.95,
            'Ca': 1.0,
            'Mg': 0.95,
            
            # Water-soluble (depends on cooking method)
            'Na': 0.9 if cooking_method == PreparationMethod.BOILED else 1.0,
            'K': 0.85 if cooking_method == PreparationMethod.BOILED else 1.0,
        }
        
        return retention_factors.get(element, 0.95)


# Utility functions for database population

def populate_from_fda_tds(
    database: ICPMSDatabaseConnector,
    fda_data_path: str
):
    """Populate database from FDA Total Diet Study data"""
    logger.info(f"Loading FDA TDS data from {fda_data_path}")
    
    # Load FDA data (CSV format)
    df = pd.read_csv(fda_data_path)
    
    measurements = []
    for _, row in df.iterrows():
        measurement = ICPMSMeasurement(
            sample_id=f"FDA_TDS_{row['sample_id']}",
            food_name=row['food_name'],
            food_category=row['food_category'],
            preparation_method=PreparationMethod(row['preparation']),
            geographic_origin=GeographicOrigin.NORTH_AMERICA,
            composition=json.loads(row['composition']),
            uncertainties=json.loads(row['uncertainties']),
            measurement_date=row['measurement_date'],
            laboratory="FDA",
            analysis_method="ICP-MS",
            sample_preparation=row['sample_prep'],
            detection_limits=json.loads(row['detection_limits']),
            recovery_rates=json.loads(row['recovery_rates']),
            rsd_percent=json.loads(row['rsd_percent']),
            moisture_content=row['moisture'],
            ash_content=row['ash'],
            protein_content=row['protein'],
            fat_content=row['fat']
        )
        measurements.append(measurement)
    
    database.batch_insert(measurements)
    logger.info(f"Loaded {len(measurements)} FDA TDS measurements")


if __name__ == "__main__":
    # Test ICP-MS database
    logger.info("Testing Phase 3: ICP-MS Integration")
    
    # Initialize database
    db = ICPMSDatabaseConnector("test_icpms.db")
    
    # Create test measurement
    test_measurement = ICPMSMeasurement(
        sample_id="TEST_001",
        food_name="salmon",
        food_category="protein",
        preparation_method=PreparationMethod.GRILLED,
        geographic_origin=GeographicOrigin.NORTH_AMERICA,
        composition={
            'Fe': 1.2, 'Zn': 0.8, 'Hg': 0.035, 'As': 0.012,
            'Na': 450.0, 'K': 380.0, 'Ca': 25.0
        },
        uncertainties={
            'Fe': (1.1, 1.3), 'Zn': (0.75, 0.85), 'Hg': (0.03, 0.04)
        },
        measurement_date="2024-01-15",
        laboratory="Test Lab",
        analysis_method="ICP-MS",
        sample_preparation="microwave_digestion",
        detection_limits={'Hg': 0.001, 'As': 0.001},
        recovery_rates={'Hg': 95.2, 'As': 98.5},
        rsd_percent={'Hg': 2.3, 'As': 1.8},
        moisture_content=65.0,
        ash_content=1.2,
        protein_content=25.0,
        fat_content=8.0,
        cooking_temperature=200.0,
        cooking_duration=15
    )
    
    # Insert
    db.insert(test_measurement)
    
    # Query
    retrieved = db.get_by_id("TEST_001")
    logger.info(f"Retrieved: {retrieved.food_name}")
    
    # Validate
    validator = GroundTruthValidator(db)
    predicted = {'Fe': 1.25, 'Zn': 0.82, 'Hg': 0.038}
    metrics = validator.validate(predicted, retrieved)
    logger.info(f"Validation MAE: {metrics['mae']:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy_percent']:.1f}%")
    
    # Test augmentation
    augmenter = DataAugmentationEngine(db)
    augmented = augmenter.augment_cooking_transformation(
        test_measurement,
        PreparationMethod.FRIED,
        n_samples=5
    )
    logger.info(f"Generated {len(augmented)} augmented samples")
    
    db.close()
    
    logger.info("Phase 3 test complete!")
