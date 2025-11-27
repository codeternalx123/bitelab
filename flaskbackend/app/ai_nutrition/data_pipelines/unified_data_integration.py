"""
Unified Data Integration Pipeline
==================================

Combines data from multiple sources into a unified training dataset:
- FDA Total Diet Study (US regulatory data)
- EFSA Chemical Contaminants (EU regulatory data)
- USDA FoodData Central (US nutritional database)

Output: Standardized dataset for training atomic vision models

Features:
- Unit conversion (mg/kg, μg/kg, mg/100g harmonization)
- Food taxonomy mapping (FDA → EFSA FoodEx2 → USDA)
- Geographic variability tracking
- Data quality validation
- Train/val/test splitting
- Export to multiple formats (JSON, HDF5, TFRecord)
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path
import hashlib

try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️  numpy not installed. Install: pip install numpy")

try:
    import pandas as pd  # type: ignore
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️  pandas not installed. Install: pip install pandas")

try:
    import h5py  # type: ignore
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("⚠️  h5py not installed. Install: pip install h5py")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class IntegrationConfig:
    """Configuration for data integration"""
    
    # Input paths
    fda_dataset: str = "data/fda_tds/fda_tds_dataset.json"
    efsa_dataset: str = "data/efsa/efsa_dataset.json"
    usda_dataset: str = "data/usda/usda_dataset.json"
    
    # Output paths
    output_dir: str = "data/integrated"
    output_json: str = "data/integrated/unified_dataset.json"
    output_hdf5: str = "data/integrated/unified_dataset.h5"
    output_csv: str = "data/integrated/unified_dataset.csv"
    
    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    
    # Quality filters
    min_elements: int = 5
    quality_threshold: float = 0.7
    
    # Unit conversion targets
    target_unit: str = "mg/kg"  # Standardize to mg/kg
    
    # Element standardization (map all sources to common symbols)
    element_mapping: Dict[str, Set[str]] = field(default_factory=lambda: {
        'Ca': {'Ca', 'Calcium', 'calcium'},
        'Fe': {'Fe', 'Iron', 'iron'},
        'Mg': {'Mg', 'Magnesium', 'magnesium'},
        'P': {'P', 'Phosphorus', 'phosphorus'},
        'K': {'K', 'Potassium', 'potassium'},
        'Na': {'Na', 'Sodium', 'sodium'},
        'Zn': {'Zn', 'Zinc', 'zinc'},
        'Cu': {'Cu', 'Copper', 'copper'},
        'Se': {'Se', 'Selenium', 'selenium'},
        'Mn': {'Mn', 'Manganese', 'manganese'},
        'Cr': {'Cr', 'Chromium', 'chromium'},
        'Mo': {'Mo', 'Molybdenum', 'molybdenum'},
        'I': {'I', 'Iodine', 'iodine'},
        'As': {'As', 'Arsenic', 'arsenic'},
        'Cd': {'Cd', 'Cadmium', 'cadmium'},
        'Pb': {'Pb', 'Lead', 'lead'},
        'Hg': {'Hg', 'Mercury', 'mercury'},
        'Co': {'Co', 'Cobalt', 'cobalt'},
        'Ni': {'Ni', 'Nickel', 'nickel'},
        'Al': {'Al', 'Aluminum', 'Aluminium', 'aluminum'},
        'Sn': {'Sn', 'Tin', 'tin'},
        'Sb': {'Sb', 'Antimony', 'antimony'},
    })


# ============================================================================
# Unified Data Structure
# ============================================================================

@dataclass
class UnifiedSample:
    """Unified food sample combining all data sources"""
    
    # Universal identifier
    sample_id: str
    
    # Food identification
    food_name: str
    food_category: str
    food_taxonomy: Dict[str, str] = field(default_factory=dict)  # FDA code, FoodEx2 code, USDA FDC ID
    
    # Elemental composition (standardized to mg/kg)
    elements: Dict[str, float] = field(default_factory=dict)
    
    # Geographic and preparation
    country: Optional[str] = None  # US, Germany, France, etc.
    region: Optional[str] = None
    preparation: Optional[str] = None  # raw, cooked, boiled, fried, etc.
    cooking_state: str = "unknown"  # raw, cooked
    
    # Data provenance
    data_sources: List[str] = field(default_factory=list)  # ["FDA_TDS", "USDA_FDC"]
    quality_score: float = 1.0
    confidence: Dict[str, float] = field(default_factory=dict)  # Per-element confidence
    
    # Analytical details
    analytical_method: str = "ICP-MS"
    measurement_year: Optional[int] = None
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def element_count(self) -> int:
        """Count available elements"""
        return len(self.elements)


@dataclass
class UnifiedDataset:
    """Integrated dataset from all sources"""
    
    samples: List[UnifiedSample] = field(default_factory=list)
    
    # Dataset splits
    train_samples: List[UnifiedSample] = field(default_factory=list)
    val_samples: List[UnifiedSample] = field(default_factory=list)
    test_samples: List[UnifiedSample] = field(default_factory=list)
    
    # Metadata
    metadata: Dict = field(default_factory=dict)
    integration_date: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def split_data(self, config: IntegrationConfig):
        """Split dataset into train/val/test"""
        if not HAS_NUMPY:
            print("⚠️  numpy required for data splitting")
            return
        
        # Shuffle samples
        np.random.seed(config.random_seed)
        indices = np.random.permutation(len(self.samples))
        
        # Calculate split sizes
        n_train = int(len(self.samples) * config.train_ratio)
        n_val = int(len(self.samples) * config.val_ratio)
        
        # Split
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        self.train_samples = [self.samples[i] for i in train_idx]
        self.val_samples = [self.samples[i] for i in val_idx]
        self.test_samples = [self.samples[i] for i in test_idx]
        
        print(f"✓ Split dataset:")
        print(f"    Train: {len(self.train_samples)} samples ({config.train_ratio*100:.0f}%)")
        print(f"    Val: {len(self.val_samples)} samples ({config.val_ratio*100:.0f}%)")
        print(f"    Test: {len(self.test_samples)} samples ({config.test_ratio*100:.0f}%)")
    
    def get_element_statistics(self) -> Dict[str, Dict]:
        """Compute statistics for each element"""
        if not HAS_NUMPY:
            return {}
        
        stats = {}
        
        # Collect all element values
        element_values = {}
        for sample in self.samples:
            for elem, value in sample.elements.items():
                if elem not in element_values:
                    element_values[elem] = []
                element_values[elem].append(value)
        
        # Compute statistics
        for elem, values in element_values.items():
            values = np.array(values)
            stats[elem] = {
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
            }
        
        return stats
    
    def export_to_json(self, filepath: str):
        """Export to JSON"""
        data = {
            'metadata': self.metadata,
            'integration_date': self.integration_date,
            'version': self.version,
            'statistics': {
                'total_samples': len(self.samples),
                'train_samples': len(self.train_samples),
                'val_samples': len(self.val_samples),
                'test_samples': len(self.test_samples),
            },
            'element_statistics': self.get_element_statistics(),
            'samples': [s.to_dict() for s in self.samples],
            'splits': {
                'train': [s.sample_id for s in self.train_samples],
                'val': [s.sample_id for s in self.val_samples],
                'test': [s.sample_id for s in self.test_samples],
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Exported to JSON: {filepath}")
    
    def export_to_csv(self, filepath: str):
        """Export to CSV (flat format)"""
        if not HAS_PANDAS:
            print("⚠️  pandas required for CSV export")
            return
        
        # Flatten samples to rows
        rows = []
        for sample in self.samples:
            row = {
                'sample_id': sample.sample_id,
                'food_name': sample.food_name,
                'food_category': sample.food_category,
                'country': sample.country or '',
                'preparation': sample.preparation or '',
                'cooking_state': sample.cooking_state,
                'data_sources': ';'.join(sample.data_sources),
                'quality_score': sample.quality_score,
            }
            
            # Add elements as columns
            for elem, value in sample.elements.items():
                row[f'element_{elem}_mg_kg'] = value
                row[f'element_{elem}_confidence'] = sample.confidence.get(elem, 1.0)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        
        print(f"✓ Exported to CSV: {filepath}")
        print(f"    Rows: {len(df)}, Columns: {len(df.columns)}")
    
    def export_to_hdf5(self, filepath: str):
        """Export to HDF5 (efficient binary format for ML)"""
        if not HAS_H5PY or not HAS_NUMPY:
            print("⚠️  h5py and numpy required for HDF5 export")
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Metadata
            f.attrs['version'] = self.version
            f.attrs['integration_date'] = self.integration_date
            f.attrs['total_samples'] = len(self.samples)
            
            # Element names (all unique elements across dataset)
            all_elements = set()
            for sample in self.samples:
                all_elements.update(sample.elements.keys())
            element_list = sorted(all_elements)
            f.create_dataset('element_names', data=np.array(element_list, dtype='S10'))
            
            # Create matrices for train/val/test
            for split_name, split_samples in [
                ('train', self.train_samples),
                ('val', self.val_samples),
                ('test', self.test_samples)
            ]:
                if len(split_samples) == 0:
                    continue
                
                grp = f.create_group(split_name)
                
                # Sample IDs
                sample_ids = [s.sample_id for s in split_samples]
                grp.create_dataset('sample_ids', data=np.array(sample_ids, dtype='S50'))
                
                # Elemental composition matrix (n_samples × n_elements)
                element_matrix = np.zeros((len(split_samples), len(element_list)))
                for i, sample in enumerate(split_samples):
                    for j, elem in enumerate(element_list):
                        if elem in sample.elements:
                            element_matrix[i, j] = sample.elements[elem]
                
                grp.create_dataset('elements', data=element_matrix, compression='gzip')
                
                # Food names
                food_names = [s.food_name for s in split_samples]
                grp.create_dataset('food_names', data=np.array(food_names, dtype='S100'))
                
                # Food categories
                food_categories = [s.food_category for s in split_samples]
                grp.create_dataset('food_categories', data=np.array(food_categories, dtype='S50'))
                
                # Cooking states
                cooking_states = [s.cooking_state for s in split_samples]
                grp.create_dataset('cooking_states', data=np.array(cooking_states, dtype='S20'))
        
        print(f"✓ Exported to HDF5: {filepath}")
        print(f"    Format: n_samples × {len(element_list)} elements")


# ============================================================================
# Data Integration Pipeline
# ============================================================================

class DataIntegrator:
    """Integrate data from FDA, EFSA, USDA"""
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        self.dataset = UnifiedDataset()
    
    def integrate_all(self) -> UnifiedDataset:
        """Integrate all data sources"""
        print("\n" + "="*60)
        print("UNIFIED DATA INTEGRATION PIPELINE")
        print("="*60)
        
        # Load datasets
        print("\n📂 Loading datasets...")
        fda_data = self._load_fda_dataset()
        efsa_data = self._load_efsa_dataset()
        usda_data = self._load_usda_dataset()
        
        print(f"  FDA TDS: {len(fda_data) if fda_data else 0} samples")
        print(f"  EFSA: {len(efsa_data) if efsa_data else 0} samples")
        print(f"  USDA FDC: {len(usda_data) if usda_data else 0} foods")
        
        # Convert to unified format
        print("\n🔄 Converting to unified format...")
        if fda_data:
            self._integrate_fda(fda_data)
        if efsa_data:
            self._integrate_efsa(efsa_data)
        if usda_data:
            self._integrate_usda(usda_data)
        
        print(f"  Total unified samples: {len(self.dataset)}")
        
        # Quality filtering
        print("\n✨ Applying quality filters...")
        self._apply_quality_filters()
        
        # Split data
        print("\n📊 Splitting dataset...")
        self.dataset.split_data(self.config)
        
        # Statistics
        print("\n📈 Computing statistics...")
        stats = self.dataset.get_element_statistics()
        
        # Summary
        self._print_summary(stats)
        
        return self.dataset
    
    def _load_fda_dataset(self) -> Optional[Dict]:
        """Load FDA TDS dataset"""
        path = Path(self.config.fda_dataset)
        if not path.exists():
            print(f"  ⚠️  FDA dataset not found: {path}")
            return None
        
        with open(path) as f:
            return json.load(f)
    
    def _load_efsa_dataset(self) -> Optional[Dict]:
        """Load EFSA dataset"""
        path = Path(self.config.efsa_dataset)
        if not path.exists():
            print(f"  ⚠️  EFSA dataset not found: {path}")
            return None
        
        with open(path) as f:
            return json.load(f)
    
    def _load_usda_dataset(self) -> Optional[Dict]:
        """Load USDA dataset"""
        path = Path(self.config.usda_dataset)
        if not path.exists():
            print(f"  ⚠️  USDA dataset not found: {path}")
            return None
        
        with open(path) as f:
            return json.load(f)
    
    def _integrate_fda(self, fda_data: Dict):
        """Integrate FDA TDS samples"""
        for sample_data in fda_data.get('samples', []):
            # Convert units if needed (FDA uses mg/kg, already standard)
            elements = sample_data.get('elements', {})
            
            unified = UnifiedSample(
                sample_id=f"FDA_{sample_data['sample_id']}",
                food_name=sample_data['food_name'],
                food_category=sample_data.get('category', 'Unknown'),
                food_taxonomy={'fda_code': sample_data.get('food_code', '')},
                elements=elements,
                country='USA',
                preparation=sample_data.get('preparation_method'),
                cooking_state=self._infer_cooking_state(sample_data.get('preparation_method')),
                data_sources=['FDA_TDS'],
                quality_score=sample_data.get('quality_score', 1.0),
                analytical_method=sample_data.get('analytical_method', 'ICP-MS'),
                measurement_year=sample_data.get('collection_year'),
            )
            
            self.dataset.samples.append(unified)
    
    def _integrate_efsa(self, efsa_data: Dict):
        """Integrate EFSA samples"""
        for sample_data in efsa_data.get('samples', []):
            elements = sample_data.get('elements', {})
            
            unified = UnifiedSample(
                sample_id=f"EFSA_{sample_data['sample_id']}",
                food_name=sample_data['food_name'],
                food_category=sample_data.get('category', 'Unknown'),
                food_taxonomy={'foodex2_code': sample_data.get('foodex2_code', '')},
                elements=elements,
                country=sample_data.get('country'),
                region=sample_data.get('region'),
                preparation=sample_data.get('preparation_method'),
                cooking_state=self._infer_cooking_state(sample_data.get('preparation_method')),
                data_sources=['EFSA'],
                quality_score=sample_data.get('quality_score', 1.0),
                analytical_method=sample_data.get('analytical_method', 'ICP-MS'),
                measurement_year=sample_data.get('sampling_year'),
            )
            
            self.dataset.samples.append(unified)
    
    def _integrate_usda(self, usda_data: Dict):
        """Integrate USDA foods"""
        for food_data in usda_data.get('foods', []):
            # Convert nutrients (mg/100g) to elements (mg/kg)
            elements = {}
            for nutrient, value in food_data.get('nutrients', {}).items():
                unit = food_data.get('nutrient_units', {}).get(nutrient, 'mg')
                
                # Convert to mg/kg
                if unit == 'mg':
                    # Assume per 100g, convert to per kg
                    elements[nutrient] = value * 10
                elif unit == 'µg' or unit == 'ug':
                    # Convert µg/100g to mg/kg
                    elements[nutrient] = value * 0.01
                else:
                    elements[nutrient] = value
            
            unified = UnifiedSample(
                sample_id=f"USDA_{food_data['fdc_id']}",
                food_name=food_data['food_name'],
                food_category=food_data.get('food_category', 'Unknown'),
                food_taxonomy={'usda_fdc_id': str(food_data['fdc_id'])},
                elements=elements,
                country='USA',
                preparation=None,
                cooking_state='unknown',
                data_sources=['USDA_FDC'],
                quality_score=food_data.get('quality_score', 0.8),
                analytical_method='Various',
            )
            
            self.dataset.samples.append(unified)
    
    def _infer_cooking_state(self, preparation: Optional[str]) -> str:
        """Infer cooking state from preparation method"""
        if not preparation:
            return 'unknown'
        
        prep_lower = preparation.lower()
        
        if any(x in prep_lower for x in ['raw', 'fresh', 'uncooked']):
            return 'raw'
        elif any(x in prep_lower for x in ['cooked', 'boiled', 'fried', 'baked', 'grilled', 'roasted']):
            return 'cooked'
        else:
            return 'processed'
    
    def _apply_quality_filters(self):
        """Apply quality filters to dataset"""
        original_count = len(self.dataset.samples)
        
        filtered_samples = []
        for sample in self.dataset.samples:
            # Filter by minimum element count
            if sample.element_count() < self.config.min_elements:
                continue
            
            # Filter by quality score
            if sample.quality_score < self.config.quality_threshold:
                continue
            
            filtered_samples.append(sample)
        
        self.dataset.samples = filtered_samples
        removed = original_count - len(filtered_samples)
        
        print(f"  ✓ Filtered {removed} low-quality samples")
        print(f"  ✓ Retained {len(filtered_samples)} high-quality samples")
    
    def _print_summary(self, stats: Dict):
        """Print integration summary"""
        print("\n" + "="*60)
        print("INTEGRATION SUMMARY")
        print("="*60)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total samples: {len(self.dataset)}")
        print(f"  Train: {len(self.dataset.train_samples)}")
        print(f"  Val: {len(self.dataset.val_samples)}")
        print(f"  Test: {len(self.dataset.test_samples)}")
        
        # Data sources
        sources = {}
        for sample in self.dataset.samples:
            for source in sample.data_sources:
                sources[source] = sources.get(source, 0) + 1
        
        print(f"\n📂 Data Sources:")
        for source, count in sorted(sources.items()):
            print(f"    {source}: {count} samples")
        
        # Elements
        print(f"\n🧪 Element Statistics (top 10):")
        top_elements = sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        for elem, stat in top_elements:
            print(f"    {elem}: {stat['count']} samples, mean={stat['mean']:.2f} mg/kg, std={stat['std']:.2f}")
        
        # Geographic distribution
        countries = {}
        for sample in self.dataset.samples:
            if sample.country:
                countries[sample.country] = countries.get(sample.country, 0) + 1
        
        print(f"\n🌍 Geographic Distribution:")
        for country, count in sorted(countries.items(), key=lambda x: x[1], reverse=True):
            print(f"    {country}: {count} samples")


# ============================================================================
# Pipeline Runner
# ============================================================================

def run_integration_pipeline(config: Optional[IntegrationConfig] = None) -> UnifiedDataset:
    """
    Run complete data integration pipeline
    
    Returns:
        UnifiedDataset combining all sources
    """
    config = config or IntegrationConfig()
    
    # Create integrator
    integrator = DataIntegrator(config)
    
    # Integrate all sources
    dataset = integrator.integrate_all()
    
    # Export to multiple formats
    print("\n💾 Exporting integrated dataset...")
    dataset.export_to_json(config.output_json)
    dataset.export_to_csv(config.output_csv)
    dataset.export_to_hdf5(config.output_hdf5)
    
    print("\n" + "="*60)
    print("✅ INTEGRATION COMPLETE")
    print("="*60)
    print(f"Total samples: {len(dataset)}")
    print(f"Elements tracked: {len(dataset.get_element_statistics())}")
    print(f"Outputs:")
    print(f"  JSON: {config.output_json}")
    print(f"  CSV: {config.output_csv}")
    print(f"  HDF5: {config.output_hdf5}")
    
    return dataset


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate FDA, EFSA, USDA datasets")
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/integrated',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = IntegrationConfig(output_dir=args.output_dir)
    
    # Run pipeline
    dataset = run_integration_pipeline(config)
    
    print(f"\n✨ Integrated {len(dataset)} samples from FDA + EFSA + USDA")
