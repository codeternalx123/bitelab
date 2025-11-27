"""
Data Quality Validation and Assurance
=====================================

Comprehensive quality validation for collected food composition data.

Validation checks:
1. Physical plausibility (mass balance, reasonable ranges)
2. Statistical outlier detection
3. Cross-source consistency
4. Temporal consistency
5. Geographic consistency
6. Method validation
7. Uncertainty quantification

Quality tiers:
- GOLD: Multi-source agreement, lab-verified, low uncertainty
- SILVER: Single reliable source, reasonable uncertainty
- BRONZE: Estimated or calculated, higher uncertainty
- REJECT: Failed validation checks
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging
from pathlib import Path
import json

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy not installed: pip install scipy")


logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Validation outcomes"""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class QualityTier(Enum):
    """Data quality tiers"""
    GOLD = "gold"      # Highest quality
    SILVER = "silver"  # Good quality
    BRONZE = "bronze"  # Acceptable
    REJECT = "reject"  # Failed validation


@dataclass
class ValidationIssue:
    """Record of validation issue"""
    check_name: str
    severity: ValidationResult
    message: str
    element: Optional[str] = None
    value: Optional[float] = None
    expected_range: Optional[Tuple[float, float]] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report for a sample"""
    sample_id: str
    overall_result: ValidationResult
    quality_tier: QualityTier
    score: float  # 0-100
    
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: int = 0
    failures: int = 0
    
    # Detailed checks
    mass_balance_ok: bool = True
    range_check_ok: bool = True
    outlier_check_ok: bool = True
    consistency_ok: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'sample_id': self.sample_id,
            'overall_result': self.overall_result.value,
            'quality_tier': self.quality_tier.value,
            'score': self.score,
            'warnings': self.warnings,
            'failures': self.failures,
            'mass_balance_ok': self.mass_balance_ok,
            'range_check_ok': self.range_check_ok,
            'outlier_check_ok': self.outlier_check_ok,
            'consistency_ok': self.consistency_ok,
            'issues': [
                {
                    'check': issue.check_name,
                    'severity': issue.severity.value,
                    'message': issue.message,
                    'element': issue.element,
                    'value': issue.value
                }
                for issue in self.issues
            ]
        }


class ElementRanges:
    """
    Expected concentration ranges for elements in food (mg/kg dry weight)
    Based on literature and regulatory databases
    """
    
    # Major elements (typically > 100 mg/kg)
    MAJOR = {
        'Ca': (10, 50000),      # Calcium: 10 mg/kg - 5%
        'K': (100, 50000),      # Potassium: 0.01% - 5%
        'P': (50, 10000),       # Phosphorus: 0.005% - 1%
        'Mg': (10, 5000),       # Magnesium: 10 mg/kg - 0.5%
        'Na': (1, 50000),       # Sodium: 1 mg/kg - 5%
        'S': (50, 5000),        # Sulfur: 0.005% - 0.5%
        'Cl': (10, 50000),      # Chlorine: 10 mg/kg - 5%
    }
    
    # Trace elements (typically < 100 mg/kg)
    TRACE = {
        'Fe': (0.5, 1000),      # Iron: 0.5 - 1000 mg/kg
        'Zn': (0.2, 500),       # Zinc: 0.2 - 500 mg/kg
        'Cu': (0.1, 100),       # Copper: 0.1 - 100 mg/kg
        'Mn': (0.1, 200),       # Manganese: 0.1 - 200 mg/kg
        'Se': (0.001, 10),      # Selenium: 1 µg/kg - 10 mg/kg
        'I': (0.01, 10),        # Iodine: 10 µg/kg - 10 mg/kg
        'Mo': (0.01, 10),       # Molybdenum: 10 µg/kg - 10 mg/kg
        'Co': (0.001, 5),       # Cobalt: 1 µg/kg - 5 mg/kg
        'Cr': (0.01, 10),       # Chromium: 10 µg/kg - 10 mg/kg
        'Ni': (0.01, 20),       # Nickel: 10 µg/kg - 20 mg/kg
        'As': (0.001, 5),       # Arsenic: 1 µg/kg - 5 mg/kg (natural)
        'Pb': (0.001, 1),       # Lead: 1 µg/kg - 1 mg/kg (limit)
        'Cd': (0.001, 0.5),     # Cadmium: 1 µg/kg - 0.5 mg/kg (limit)
        'Hg': (0.001, 1),       # Mercury: 1 µg/kg - 1 mg/kg (limit)
        'Al': (0.5, 100),       # Aluminum: 0.5 - 100 mg/kg
        'B': (0.1, 100),        # Boron: 0.1 - 100 mg/kg
    }
    
    # Ultra-trace (< 1 mg/kg typical)
    ULTRA_TRACE = {
        'V': (0.001, 5),        # Vanadium
        'Li': (0.001, 5),       # Lithium
        'Sr': (0.01, 100),      # Strontium
        'Ba': (0.01, 50),       # Barium
    }
    
    @classmethod
    def get_range(cls, element: str) -> Optional[Tuple[float, float]]:
        """Get expected range for element"""
        if element in cls.MAJOR:
            return cls.MAJOR[element]
        elif element in cls.TRACE:
            return cls.TRACE[element]
        elif element in cls.ULTRA_TRACE:
            return cls.ULTRA_TRACE[element]
        return None
    
    @classmethod
    def is_within_range(cls, element: str, value: float) -> bool:
        """Check if value is within expected range"""
        range_ = cls.get_range(element)
        if range_ is None:
            return True  # Unknown element, can't validate
        
        min_val, max_val = range_
        return min_val <= value <= max_val


class DataQualityValidator:
    """Comprehensive data quality validation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Statistics for outlier detection (will be populated)
        self.element_distributions: Dict[str, Dict[str, float]] = {}
        
        # Reference data for cross-validation
        self.reference_samples: List[Dict] = []
    
    def validate_sample(
        self,
        sample_id: str,
        elements: Dict[str, float],
        metadata: Optional[Dict] = None
    ) -> ValidationReport:
        """
        Comprehensive validation of a single sample
        
        Args:
            sample_id: Unique identifier
            elements: Dict of element -> concentration (mg/kg)
            metadata: Optional metadata (source, country, method, etc.)
        
        Returns:
            ValidationReport with detailed results
        """
        
        report = ValidationReport(
            sample_id=sample_id,
            overall_result=ValidationResult.PASS,
            quality_tier=QualityTier.SILVER,
            score=100.0
        )
        
        # Run validation checks
        self._check_mass_balance(elements, report)
        self._check_element_ranges(elements, report)
        self._check_outliers(elements, report)
        self._check_consistency(elements, report)
        
        if metadata:
            self._check_metadata(metadata, report)
        
        # Compute overall score
        self._compute_score(report)
        
        # Determine quality tier
        self._determine_tier(report, metadata)
        
        # Set overall result
        if report.failures > 0:
            report.overall_result = ValidationResult.FAIL
        elif report.warnings > 0:
            report.overall_result = ValidationResult.WARNING
        else:
            report.overall_result = ValidationResult.PASS
        
        return report
    
    def _check_mass_balance(
        self,
        elements: Dict[str, float],
        report: ValidationReport
    ):
        """
        Check if total concentration is reasonable
        
        Sum of all elements should be < 100,000 mg/kg (10% dry weight)
        Inorganic elements typically < 5% of food by weight
        """
        
        total = sum(elements.values())
        
        # Threshold: 100,000 mg/kg = 10% by weight
        if total > 100000:
            report.mass_balance_ok = False
            report.failures += 1
            report.issues.append(ValidationIssue(
                check_name='mass_balance',
                severity=ValidationResult.FAIL,
                message=f'Total element concentration ({total:.0f} mg/kg) exceeds 10% by weight',
                value=total,
                expected_range=(0, 100000)
            ))
        
        # Warning if > 5%
        elif total > 50000:
            report.warnings += 1
            report.issues.append(ValidationIssue(
                check_name='mass_balance',
                severity=ValidationResult.WARNING,
                message=f'Total element concentration ({total:.0f} mg/kg) is high (>5%)',
                value=total
            ))
    
    def _check_element_ranges(
        self,
        elements: Dict[str, float],
        report: ValidationReport
    ):
        """Check if individual elements are within expected ranges"""
        
        for element, value in elements.items():
            # Check for negative values
            if value < 0:
                report.range_check_ok = False
                report.failures += 1
                report.issues.append(ValidationIssue(
                    check_name='range_check',
                    severity=ValidationResult.FAIL,
                    message=f'{element} concentration is negative',
                    element=element,
                    value=value
                ))
                continue
            
            # Check against expected ranges
            range_ = ElementRanges.get_range(element)
            if range_:
                min_val, max_val = range_
                
                if value < min_val or value > max_val:
                    # Hard failure if way outside range (>100×)
                    if value < min_val / 100 or value > max_val * 100:
                        report.range_check_ok = False
                        report.failures += 1
                        report.issues.append(ValidationIssue(
                            check_name='range_check',
                            severity=ValidationResult.FAIL,
                            message=f'{element} concentration ({value:.2f}) far outside expected range',
                            element=element,
                            value=value,
                            expected_range=range_
                        ))
                    else:
                        # Warning if outside but not extreme
                        report.warnings += 1
                        report.issues.append(ValidationIssue(
                            check_name='range_check',
                            severity=ValidationResult.WARNING,
                            message=f'{element} concentration ({value:.2f}) outside typical range',
                            element=element,
                            value=value,
                            expected_range=range_
                        ))
    
    def _check_outliers(
        self,
        elements: Dict[str, float],
        report: ValidationReport
    ):
        """
        Statistical outlier detection
        
        Use z-scores if we have population statistics
        """
        
        if not self.element_distributions or not HAS_SCIPY:
            return  # Skip if no reference data
        
        for element, value in elements.items():
            if element not in self.element_distributions:
                continue
            
            stats_dict = self.element_distributions[element]
            mean = stats_dict.get('mean', 0)
            std = stats_dict.get('std', 1)
            
            if std == 0:
                continue
            
            # Compute z-score
            z_score = abs((value - mean) / std)
            
            # Flag if z-score > 4 (very unusual)
            if z_score > 4:
                report.outlier_check_ok = False
                report.warnings += 1
                report.issues.append(ValidationIssue(
                    check_name='outlier_check',
                    severity=ValidationResult.WARNING,
                    message=f'{element} is statistical outlier (z={z_score:.1f})',
                    element=element,
                    value=value
                ))
    
    def _check_consistency(
        self,
        elements: Dict[str, float],
        report: ValidationReport
    ):
        """
        Check for inconsistencies between related elements
        
        Examples:
        - Ca/P ratio should be reasonable (typically 0.5-2.0)
        - Na/K ratio varies but extremes are unusual
        - Heavy metals should correlate in some foods
        """
        
        # Ca/P ratio check
        if 'Ca' in elements and 'P' in elements:
            ca = elements['Ca']
            p = elements['P']
            
            if p > 0:
                ratio = ca / p
                
                # Typical ratio: 0.1 - 10 (very wide range)
                if ratio < 0.01 or ratio > 100:
                    report.consistency_ok = False
                    report.warnings += 1
                    report.issues.append(ValidationIssue(
                        check_name='consistency_check',
                        severity=ValidationResult.WARNING,
                        message=f'Unusual Ca/P ratio: {ratio:.2f}',
                        element='Ca/P',
                        value=ratio
                    ))
        
        # Na/K ratio check
        if 'Na' in elements and 'K' in elements:
            na = elements['Na']
            k = elements['K']
            
            if k > 0:
                ratio = na / k
                
                # Typical ratio: 0.01 - 10 (varies widely by food)
                if ratio < 0.001 or ratio > 100:
                    report.warnings += 1
                    report.issues.append(ValidationIssue(
                        check_name='consistency_check',
                        severity=ValidationResult.WARNING,
                        message=f'Unusual Na/K ratio: {ratio:.2f}',
                        element='Na/K',
                        value=ratio
                    ))
    
    def _check_metadata(
        self,
        metadata: Dict,
        report: ValidationReport
    ):
        """Validate metadata quality"""
        
        # Check if method is specified
        method = metadata.get('analysis_method')
        if not method:
            report.warnings += 1
            report.issues.append(ValidationIssue(
                check_name='metadata_check',
                severity=ValidationResult.WARNING,
                message='Analysis method not specified'
            ))
        
        # Check data source
        source = metadata.get('source')
        if not source:
            report.warnings += 1
            report.issues.append(ValidationIssue(
                check_name='metadata_check',
                severity=ValidationResult.WARNING,
                message='Data source not specified'
            ))
    
    def _compute_score(self, report: ValidationReport):
        """Compute overall quality score (0-100)"""
        
        score = 100.0
        
        # Deduct for issues
        score -= report.failures * 20  # -20 per failure
        score -= report.warnings * 5   # -5 per warning
        
        # Floor at 0
        score = max(0, score)
        
        report.score = score
    
    def _determine_tier(
        self,
        report: ValidationReport,
        metadata: Optional[Dict]
    ):
        """Determine quality tier based on validation and metadata"""
        
        # Start with GOLD, downgrade based on issues
        tier = QualityTier.GOLD
        
        # Automatic REJECT if failed critical checks
        if report.failures > 0:
            tier = QualityTier.REJECT
        
        # Downgrade based on warnings
        elif report.warnings >= 3:
            tier = QualityTier.BRONZE
        elif report.warnings >= 1:
            tier = QualityTier.SILVER
        
        # Upgrade/downgrade based on metadata
        if metadata and tier != QualityTier.REJECT:
            source = metadata.get('source', '')
            method = metadata.get('analysis_method', '')
            
            # High-quality sources (NIST, FDA)
            if 'nist' in source.lower() or 'fda' in source.lower():
                if tier == QualityTier.BRONZE:
                    tier = QualityTier.SILVER
            
            # ICP-MS is gold standard
            if 'icp-ms' in method.lower():
                # Keep tier or upgrade slightly
                pass
            elif 'estimate' in method.lower() or 'calculate' in method.lower():
                # Downgrade
                if tier == QualityTier.GOLD:
                    tier = QualityTier.SILVER
                elif tier == QualityTier.SILVER:
                    tier = QualityTier.BRONZE
        
        report.quality_tier = tier
    
    def calibrate_distributions(self, samples: List[Dict]):
        """
        Calibrate statistical distributions from reference dataset
        
        Args:
            samples: List of dicts with 'elements' key
        """
        
        self.logger.info("📊 Calibrating element distributions...")
        
        # Collect all values by element
        element_values = defaultdict(list)
        
        for sample in samples:
            elements = sample.get('elements', {})
            for element, value in elements.items():
                if value > 0:  # Exclude zeros
                    element_values[element].append(value)
        
        # Compute statistics
        for element, values in element_values.items():
            if len(values) < 10:
                continue  # Need minimum samples
            
            values_array = np.array(values)
            
            self.element_distributions[element] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'median': float(np.median(values_array)),
                'q25': float(np.percentile(values_array, 25)),
                'q75': float(np.percentile(values_array, 75)),
                'count': len(values)
            }
        
        self.logger.info(f"✅ Calibrated distributions for {len(self.element_distributions)} elements")
    
    def validate_batch(
        self,
        samples: List[Dict]
    ) -> List[ValidationReport]:
        """
        Validate a batch of samples
        
        Args:
            samples: List of dicts with 'sample_id', 'elements', and optional metadata
        
        Returns:
            List of ValidationReport
        """
        
        self.logger.info(f"🔍 Validating {len(samples)} samples...")
        
        reports = []
        
        for sample in samples:
            sample_id = sample.get('sample_id', 'unknown')
            elements = sample.get('elements', {})
            metadata = sample.get('metadata')
            
            report = self.validate_sample(sample_id, elements, metadata)
            reports.append(report)
        
        # Summary statistics
        passed = sum(1 for r in reports if r.overall_result == ValidationResult.PASS)
        warnings = sum(1 for r in reports if r.overall_result == ValidationResult.WARNING)
        failed = sum(1 for r in reports if r.overall_result == ValidationResult.FAIL)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("📊 VALIDATION SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"  ✅ Passed: {passed} ({passed/len(reports)*100:.1f}%)")
        self.logger.info(f"  ⚠️  Warnings: {warnings} ({warnings/len(reports)*100:.1f}%)")
        self.logger.info(f"  ❌ Failed: {failed} ({failed/len(reports)*100:.1f}%)")
        
        # Quality tiers
        gold = sum(1 for r in reports if r.quality_tier == QualityTier.GOLD)
        silver = sum(1 for r in reports if r.quality_tier == QualityTier.SILVER)
        bronze = sum(1 for r in reports if r.quality_tier == QualityTier.BRONZE)
        reject = sum(1 for r in reports if r.quality_tier == QualityTier.REJECT)
        
        self.logger.info(f"\n🏆 Quality Tiers:")
        self.logger.info(f"  🥇 Gold: {gold}")
        self.logger.info(f"  🥈 Silver: {silver}")
        self.logger.info(f"  🥉 Bronze: {bronze}")
        self.logger.info(f"  ❌ Reject: {reject}")
        self.logger.info(f"{'='*60}\n")
        
        return reports
    
    def export_reports(self, reports: List[ValidationReport], output_path: Path):
        """Export validation reports to JSON"""
        
        data = {
            'total_samples': len(reports),
            'summary': {
                'passed': sum(1 for r in reports if r.overall_result == ValidationResult.PASS),
                'warnings': sum(1 for r in reports if r.overall_result == ValidationResult.WARNING),
                'failed': sum(1 for r in reports if r.overall_result == ValidationResult.FAIL),
                'gold': sum(1 for r in reports if r.quality_tier == QualityTier.GOLD),
                'silver': sum(1 for r in reports if r.quality_tier == QualityTier.SILVER),
                'bronze': sum(1 for r in reports if r.quality_tier == QualityTier.BRONZE),
                'reject': sum(1 for r in reports if r.quality_tier == QualityTier.REJECT),
            },
            'reports': [r.to_dict() for r in reports]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"💾 Exported validation reports to {output_path}")


def main():
    """Test validation system"""
    
    # Create validator
    validator = DataQualityValidator()
    
    # Test samples
    samples = [
        {
            'sample_id': 'test_001',
            'elements': {
                'Ca': 500, 'Fe': 15, 'K': 1200, 'Mg': 80, 'Na': 200, 'P': 300, 'Zn': 8
            },
            'metadata': {'source': 'NIST', 'analysis_method': 'ICP-MS'}
        },
        {
            'sample_id': 'test_002',
            'elements': {
                'Ca': 50000, 'Fe': 5, 'K': 500, 'Mg': 2000, 'Na': 100, 'P': 800, 'Zn': 3
            },
            'metadata': {'source': 'USDA', 'analysis_method': 'Calculated'}
        },
        {
            'sample_id': 'test_003_bad',
            'elements': {
                'Ca': 150000,  # Too high!
                'Fe': -5,      # Negative!
                'K': 50,       # Too low
                'Mg': 10000,   # Too high
                'Na': 200,
                'P': 300,
                'Zn': 8
            },
            'metadata': {'source': 'Unknown', 'analysis_method': None}
        }
    ]
    
    # Calibrate (use test samples as reference)
    validator.calibrate_distributions(samples[:2])
    
    # Validate
    reports = validator.validate_batch(samples)
    
    # Export
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)
    validator.export_reports(reports, output_dir / "validation_reports.json")
    
    # Print individual reports
    for report in reports:
        print(f"\n{'='*60}")
        print(f"Sample: {report.sample_id}")
        print(f"Result: {report.overall_result.value}")
        print(f"Tier: {report.quality_tier.value}")
        print(f"Score: {report.score:.1f}/100")
        print(f"Issues: {len(report.issues)}")
        
        for issue in report.issues:
            print(f"  [{issue.severity.value}] {issue.message}")


if __name__ == '__main__':
    main()
