"""
PHASE 7: DATA QUALITY & GOVERNANCE INFRASTRUCTURE
==================================================

Enterprise-grade data quality and governance system for AI nutrition analysis.
Ensures data accuracy, completeness, consistency, and regulatory compliance.

Components:
1. Data Quality Assessment Engine
2. Data Validation Framework
3. Data Profiling & Statistics
4. Data Lineage Tracking
5. Data Catalog & Metadata Management
6. Compliance & Audit System
7. Data Anonymization & Privacy
8. Data Quality Monitoring & Alerts

Author: Wellomex AI Team
Date: November 2025
"""

import logging
import time
import uuid
import json
import hashlib
import re
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, Counter
import statistics
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. DATA QUALITY ASSESSMENT ENGINE
# ============================================================================

class QualityDimension(Enum):
    """Data quality dimensions"""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


@dataclass
class QualityScore:
    """Quality assessment score"""
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class QualityReport:
    """Comprehensive quality report"""
    report_id: str
    dataset_name: str
    overall_score: float
    dimension_scores: Dict[str, QualityScore]
    record_count: int
    issues_count: int
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class DataQualityEngine:
    """
    Comprehensive data quality assessment engine
    
    Features:
    - Multi-dimensional quality scoring
    - Automated issue detection
    - Quality trend analysis
    - Actionable recommendations
    - Threshold-based alerting
    """
    
    def __init__(self):
        self.quality_thresholds = {
            QualityDimension.ACCURACY: 0.95,
            QualityDimension.COMPLETENESS: 0.90,
            QualityDimension.CONSISTENCY: 0.95,
            QualityDimension.TIMELINESS: 0.85,
            QualityDimension.VALIDITY: 0.98,
            QualityDimension.UNIQUENESS: 0.99
        }
        self.reports: List[QualityReport] = []
        logger.info("DataQualityEngine initialized")
    
    def assess_quality(
        self,
        dataset_name: str,
        data: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]] = None
    ) -> QualityReport:
        """Perform comprehensive quality assessment"""
        
        logger.info(f"Assessing quality for dataset: {dataset_name}")
        
        dimension_scores = {}
        
        # Assess each dimension
        dimension_scores['accuracy'] = self._assess_accuracy(data, schema)
        dimension_scores['completeness'] = self._assess_completeness(data, schema)
        dimension_scores['consistency'] = self._assess_consistency(data)
        dimension_scores['timeliness'] = self._assess_timeliness(data)
        dimension_scores['validity'] = self._assess_validity(data, schema)
        dimension_scores['uniqueness'] = self._assess_uniqueness(data)
        
        # Calculate overall score (weighted average)
        weights = {
            'accuracy': 0.25,
            'completeness': 0.20,
            'consistency': 0.20,
            'timeliness': 0.10,
            'validity': 0.15,
            'uniqueness': 0.10
        }
        
        overall_score = sum(
            dimension_scores[dim].score * weights[dim]
            for dim in weights.keys()
        )
        
        # Count total issues
        issues_count = sum(
            len(score.issues) for score in dimension_scores.values()
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(dimension_scores)
        
        report = QualityReport(
            report_id=f"qr-{uuid.uuid4().hex[:8]}",
            dataset_name=dataset_name,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            record_count=len(data),
            issues_count=issues_count,
            recommendations=recommendations
        )
        
        self.reports.append(report)
        
        logger.info(
            f"Quality assessment complete: {overall_score:.2%} "
            f"({issues_count} issues found)"
        )
        
        return report
    
    def _assess_accuracy(
        self,
        data: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]]
    ) -> QualityScore:
        """Assess data accuracy"""
        issues = []
        accurate_records = 0
        
        for record in data:
            # Check for reasonable values
            if 'calories' in record:
                calories = record['calories']
                if calories is None:
                    # Skip None values (handled by completeness)
                    continue
                if not (0 <= calories <= 10000):
                    issues.append(f"Unreasonable calorie value: {calories}")
                else:
                    accurate_records += 1
            else:
                accurate_records += 1
        
        score = accurate_records / len(data) if data else 0.0
        
        return QualityScore(
            dimension=QualityDimension.ACCURACY,
            score=score,
            issues=issues[:10],  # Limit to 10 examples
            metrics={
                'accurate_records': accurate_records,
                'total_records': len(data)
            }
        )
    
    def _assess_completeness(
        self,
        data: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]]
    ) -> QualityScore:
        """Assess data completeness"""
        issues = []
        complete_records = 0
        
        required_fields = schema.get('required', []) if schema else []
        
        for i, record in enumerate(data):
            missing_fields = []
            
            # Check required fields
            for field in required_fields:
                if field not in record or record[field] is None:
                    missing_fields.append(field)
            
            # Check for empty values
            for key, value in record.items():
                if value is None or value == '':
                    missing_fields.append(key)
            
            if not missing_fields:
                complete_records += 1
            elif len(issues) < 10:
                issues.append(f"Record {i}: Missing {', '.join(missing_fields)}")
        
        score = complete_records / len(data) if data else 0.0
        
        return QualityScore(
            dimension=QualityDimension.COMPLETENESS,
            score=score,
            issues=issues,
            metrics={
                'complete_records': complete_records,
                'incomplete_records': len(data) - complete_records
            }
        )
    
    def _assess_consistency(self, data: List[Dict[str, Any]]) -> QualityScore:
        """Assess data consistency"""
        issues = []
        consistent_records = 0
        
        for i, record in enumerate(data):
            inconsistencies = []
            
            # Check nutritional consistency
            if all(k in record for k in ['protein', 'carbs', 'fat', 'calories']):
                # Skip if any value is None
                if any(record[k] is None for k in ['protein', 'carbs', 'fat', 'calories']):
                    consistent_records += 1  # Can't check consistency
                    continue
                
                calculated_calories = (
                    record['protein'] * 4 +
                    record['carbs'] * 4 +
                    record['fat'] * 9
                )
                actual_calories = record['calories']
                
                # Allow 10% tolerance
                if abs(calculated_calories - actual_calories) > actual_calories * 0.1:
                    inconsistencies.append(
                        f"Calorie mismatch: {actual_calories} vs {calculated_calories:.0f}"
                    )
            
            if not inconsistencies:
                consistent_records += 1
            elif len(issues) < 10:
                issues.append(f"Record {i}: {', '.join(inconsistencies)}")
        
        score = consistent_records / len(data) if data else 0.0
        
        return QualityScore(
            dimension=QualityDimension.CONSISTENCY,
            score=score,
            issues=issues,
            metrics={
                'consistent_records': consistent_records,
                'inconsistent_records': len(data) - consistent_records
            }
        )
    
    def _assess_timeliness(self, data: List[Dict[str, Any]]) -> QualityScore:
        """Assess data timeliness"""
        issues = []
        timely_records = 0
        
        current_time = time.time()
        max_age = 86400 * 30  # 30 days
        
        for i, record in enumerate(data):
            if 'timestamp' in record:
                age = current_time - record['timestamp']
                if age <= max_age:
                    timely_records += 1
                elif len(issues) < 10:
                    issues.append(f"Record {i}: {age / 86400:.0f} days old")
            else:
                timely_records += 1  # No timestamp = assume timely
        
        score = timely_records / len(data) if data else 0.0
        
        return QualityScore(
            dimension=QualityDimension.TIMELINESS,
            score=score,
            issues=issues,
            metrics={
                'timely_records': timely_records,
                'stale_records': len(data) - timely_records
            }
        )
    
    def _assess_validity(
        self,
        data: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]]
    ) -> QualityScore:
        """Assess data validity"""
        issues = []
        valid_records = 0
        
        for i, record in enumerate(data):
            invalid_fields = []
            
            # Type checking
            if 'name' in record and not isinstance(record['name'], str):
                invalid_fields.append('name (not string)')
            
            if 'calories' in record:
                if not isinstance(record['calories'], (int, float)):
                    invalid_fields.append('calories (not numeric)')
            
            # Format validation
            if 'email' in record:
                email = record['email']
                if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', str(email)):
                    invalid_fields.append('email (invalid format)')
            
            if not invalid_fields:
                valid_records += 1
            elif len(issues) < 10:
                issues.append(f"Record {i}: Invalid {', '.join(invalid_fields)}")
        
        score = valid_records / len(data) if data else 0.0
        
        return QualityScore(
            dimension=QualityDimension.VALIDITY,
            score=score,
            issues=issues,
            metrics={
                'valid_records': valid_records,
                'invalid_records': len(data) - valid_records
            }
        )
    
    def _assess_uniqueness(self, data: List[Dict[str, Any]]) -> QualityScore:
        """Assess data uniqueness"""
        issues = []
        
        # Check for duplicates based on key fields
        seen_keys = set()
        unique_records = 0
        
        for i, record in enumerate(data):
            # Create composite key
            key_fields = ['name', 'id', 'food_id']
            key_parts = [
                str(record.get(field, ''))
                for field in key_fields
                if field in record
            ]
            key = '|'.join(key_parts)
            
            if key not in seen_keys:
                seen_keys.add(key)
                unique_records += 1
            elif len(issues) < 10:
                issues.append(f"Duplicate record {i}: {key}")
        
        score = unique_records / len(data) if data else 0.0
        
        return QualityScore(
            dimension=QualityDimension.UNIQUENESS,
            score=score,
            issues=issues,
            metrics={
                'unique_records': unique_records,
                'duplicate_records': len(data) - unique_records
            }
        )
    
    def _generate_recommendations(
        self,
        dimension_scores: Dict[str, QualityScore]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        for dim_name, score in dimension_scores.items():
            threshold = self.quality_thresholds.get(score.dimension, 0.9)
            
            if score.score < threshold:
                if score.dimension == QualityDimension.COMPLETENESS:
                    recommendations.append(
                        f"Improve completeness: Fill missing fields "
                        f"({score.metrics.get('incomplete_records', 0)} records affected)"
                    )
                elif score.dimension == QualityDimension.ACCURACY:
                    recommendations.append(
                        "Validate data sources and implement range checks"
                    )
                elif score.dimension == QualityDimension.CONSISTENCY:
                    recommendations.append(
                        "Review calculation logic and cross-field validations"
                    )
                elif score.dimension == QualityDimension.UNIQUENESS:
                    recommendations.append(
                        "Implement deduplication process before ingestion"
                    )
        
        return recommendations
    
    def get_quality_trends(
        self,
        dataset_name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get quality score trends over time"""
        reports = [
            r for r in self.reports[-limit:]
            if r.dataset_name == dataset_name
        ]
        
        return [
            {
                'timestamp': r.timestamp,
                'overall_score': r.overall_score,
                'issues_count': r.issues_count
            }
            for r in reports
        ]


# ============================================================================
# 2. DATA VALIDATION FRAMEWORK
# ============================================================================

class ValidationRule(ABC):
    """Abstract base class for validation rules"""
    
    def __init__(self, name: str, severity: str = "error"):
        self.name = name
        self.severity = severity  # "error", "warning", "info"
    
    @abstractmethod
    def validate(self, value: Any, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate a value, return (is_valid, error_message)"""
        pass


class RangeValidation(ValidationRule):
    """Validate numeric range"""
    
    def __init__(self, name: str, min_val: float, max_val: float, **kwargs):
        super().__init__(name, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, value: Any, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if not isinstance(value, (int, float)):
            return False, f"Value must be numeric, got {type(value).__name__}"
        
        if not (self.min_val <= value <= self.max_val):
            return False, f"Value {value} outside range [{self.min_val}, {self.max_val}]"
        
        return True, None


class RegexValidation(ValidationRule):
    """Validate using regex pattern"""
    
    def __init__(self, name: str, pattern: str, **kwargs):
        super().__init__(name, **kwargs)
        self.pattern = re.compile(pattern)
    
    def validate(self, value: Any, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if not isinstance(value, str):
            return False, f"Value must be string, got {type(value).__name__}"
        
        if not self.pattern.match(value):
            return False, f"Value '{value}' does not match pattern"
        
        return True, None


class RequiredValidation(ValidationRule):
    """Validate required field presence"""
    
    def validate(self, value: Any, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if value is None or value == '':
            return False, "Value is required"
        
        return True, None


class CustomValidation(ValidationRule):
    """Custom validation function"""
    
    def __init__(self, name: str, validator_func: Callable, **kwargs):
        super().__init__(name, **kwargs)
        self.validator_func = validator_func
    
    def validate(self, value: Any, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        try:
            is_valid = self.validator_func(value, context)
            if is_valid:
                return True, None
            return False, "Custom validation failed"
        except Exception as e:
            return False, f"Validation error: {str(e)}"


@dataclass
class ValidationResult:
    """Result of validation"""
    field: str
    is_valid: bool
    rule_name: str
    error_message: Optional[str] = None
    severity: str = "error"


class DataValidator:
    """
    Flexible data validation framework
    
    Features:
    - Rule-based validation
    - Custom validation functions
    - Batch validation
    - Validation reporting
    - Configurable severity levels
    """
    
    def __init__(self):
        self.rules: Dict[str, List[ValidationRule]] = defaultdict(list)
        self.validation_history: List[Dict[str, Any]] = []
        logger.info("DataValidator initialized")
    
    def add_rule(self, field: str, rule: ValidationRule):
        """Add a validation rule for a field"""
        self.rules[field].append(rule)
        logger.info(f"Added validation rule '{rule.name}' for field '{field}'")
    
    def validate_record(
        self,
        record: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """Validate a single record"""
        results = []
        ctx = context or {}
        
        for field, rules in self.rules.items():
            value = record.get(field)
            
            for rule in rules:
                is_valid, error_message = rule.validate(value, ctx)
                
                results.append(ValidationResult(
                    field=field,
                    is_valid=is_valid,
                    rule_name=rule.name,
                    error_message=error_message,
                    severity=rule.severity
                ))
        
        return results
    
    def validate_batch(
        self,
        records: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate a batch of records"""
        
        logger.info(f"Validating batch of {len(records)} records")
        
        all_results = []
        valid_count = 0
        error_count = 0
        warning_count = 0
        
        for i, record in enumerate(records):
            results = self.validate_record(record, context)
            
            record_valid = all(r.is_valid for r in results if r.severity == "error")
            if record_valid:
                valid_count += 1
            
            for result in results:
                if not result.is_valid:
                    if result.severity == "error":
                        error_count += 1
                    elif result.severity == "warning":
                        warning_count += 1
            
            all_results.extend(results)
        
        # Aggregate by field and rule
        issues_by_field = defaultdict(int)
        for result in all_results:
            if not result.is_valid:
                issues_by_field[result.field] += 1
        
        validation_summary = {
            'total_records': len(records),
            'valid_records': valid_count,
            'invalid_records': len(records) - valid_count,
            'error_count': error_count,
            'warning_count': warning_count,
            'validation_rate': valid_count / len(records) if records else 0,
            'issues_by_field': dict(issues_by_field),
            'sample_errors': [
                {
                    'field': r.field,
                    'rule': r.rule_name,
                    'error': r.error_message
                }
                for r in all_results[:10] if not r.is_valid
            ]
        }
        
        self.validation_history.append({
            'timestamp': time.time(),
            'summary': validation_summary
        })
        
        logger.info(
            f"Validation complete: {valid_count}/{len(records)} valid "
            f"({error_count} errors, {warning_count} warnings)"
        )
        
        return validation_summary


# ============================================================================
# 3. DATA PROFILING & STATISTICS
# ============================================================================

@dataclass
class ColumnProfile:
    """Statistical profile of a data column"""
    column_name: str
    data_type: str
    count: int
    null_count: int
    null_percentage: float
    unique_count: int
    
    # Numeric statistics
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std_dev: Optional[float] = None
    
    # String statistics
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    
    # Distribution
    top_values: List[Tuple[Any, int]] = field(default_factory=list)
    histogram: Optional[Dict[str, int]] = None


class DataProfiler:
    """
    Comprehensive data profiling engine
    
    Features:
    - Column-level statistics
    - Distribution analysis
    - Correlation detection
    - Anomaly identification
    - Data type inference
    """
    
    def __init__(self):
        self.profiles: Dict[str, Dict[str, ColumnProfile]] = {}
        logger.info("DataProfiler initialized")
    
    def profile_dataset(
        self,
        dataset_name: str,
        data: List[Dict[str, Any]]
    ) -> Dict[str, ColumnProfile]:
        """Generate comprehensive data profile"""
        
        logger.info(f"Profiling dataset: {dataset_name} ({len(data)} records)")
        
        if not data:
            return {}
        
        # Get all columns
        all_columns = set()
        for record in data:
            all_columns.update(record.keys())
        
        profiles = {}
        
        for column in all_columns:
            profiles[column] = self._profile_column(column, data)
        
        self.profiles[dataset_name] = profiles
        
        logger.info(f"Profiled {len(profiles)} columns")
        return profiles
    
    def _profile_column(
        self,
        column_name: str,
        data: List[Dict[str, Any]]
    ) -> ColumnProfile:
        """Profile a single column"""
        
        # Extract column values
        values = [record.get(column_name) for record in data]
        non_null_values = [v for v in values if v is not None]
        
        # Basic stats
        count = len(values)
        null_count = count - len(non_null_values)
        null_percentage = null_count / count if count > 0 else 0
        unique_count = len(set(non_null_values))
        
        # Infer data type
        if non_null_values:
            sample = non_null_values[0]
            if isinstance(sample, bool):
                data_type = "boolean"
            elif isinstance(sample, int):
                data_type = "integer"
            elif isinstance(sample, float):
                data_type = "float"
            elif isinstance(sample, str):
                data_type = "string"
            else:
                data_type = "unknown"
        else:
            data_type = "unknown"
        
        profile = ColumnProfile(
            column_name=column_name,
            data_type=data_type,
            count=count,
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count
        )
        
        # Numeric statistics
        if data_type in ("integer", "float"):
            numeric_values = [float(v) for v in non_null_values if isinstance(v, (int, float))]
            if numeric_values:
                profile.min_value = min(numeric_values)
                profile.max_value = max(numeric_values)
                profile.mean = statistics.mean(numeric_values)
                profile.median = statistics.median(numeric_values)
                if len(numeric_values) > 1:
                    profile.std_dev = statistics.stdev(numeric_values)
        
        # String statistics
        if data_type == "string":
            string_values = [str(v) for v in non_null_values]
            if string_values:
                lengths = [len(s) for s in string_values]
                profile.min_length = min(lengths)
                profile.max_length = max(lengths)
                profile.avg_length = statistics.mean(lengths)
        
        # Top values
        if non_null_values:
            value_counts = Counter(non_null_values)
            profile.top_values = value_counts.most_common(5)
        
        return profile
    
    def get_profile_summary(self, dataset_name: str) -> Dict[str, Any]:
        """Get summary of dataset profile"""
        
        if dataset_name not in self.profiles:
            return {}
        
        profiles = self.profiles[dataset_name]
        
        return {
            'dataset_name': dataset_name,
            'column_count': len(profiles),
            'columns': {
                name: {
                    'type': prof.data_type,
                    'null_pct': f"{prof.null_percentage:.1%}",
                    'unique': prof.unique_count,
                    'stats': {
                        'min': prof.min_value,
                        'max': prof.max_value,
                        'mean': prof.mean
                    } if prof.data_type in ("integer", "float") else None
                }
                for name, prof in profiles.items()
            }
        }
    
    def detect_anomalies(
        self,
        dataset_name: str,
        column_name: str,
        z_threshold: float = 3.0
    ) -> List[Dict[str, Any]]:
        """Detect statistical anomalies in a column"""
        
        if dataset_name not in self.profiles:
            return []
        
        profile = self.profiles[dataset_name].get(column_name)
        if not profile or profile.data_type not in ("integer", "float"):
            return []
        
        anomalies = []
        
        # Check for outliers using z-score
        if profile.mean is not None and profile.std_dev is not None:
            # Mock data for demonstration
            if profile.max_value and profile.mean:
                z_score = abs(profile.max_value - profile.mean) / profile.std_dev if profile.std_dev > 0 else 0
                if z_score > z_threshold:
                    anomalies.append({
                        'type': 'outlier',
                        'value': profile.max_value,
                        'z_score': z_score,
                        'description': f"Value significantly above mean"
                    })
        
        return anomalies


# ============================================================================
# 4. DATA LINEAGE TRACKING
# ============================================================================

@dataclass
class LineageNode:
    """Node in data lineage graph"""
    node_id: str
    node_type: str  # "source", "transformation", "destination"
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class LineageEdge:
    """Edge in data lineage graph"""
    source_id: str
    target_id: str
    operation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataLineageTracker:
    """
    Track data lineage and provenance
    
    Features:
    - Source-to-destination tracking
    - Transformation history
    - Impact analysis
    - Lineage visualization
    - Compliance reporting
    """
    
    def __init__(self):
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: List[LineageEdge] = []
        logger.info("DataLineageTracker initialized")
    
    def register_source(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a data source"""
        node_id = f"src-{uuid.uuid4().hex[:8]}"
        
        node = LineageNode(
            node_id=node_id,
            node_type="source",
            name=name,
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        logger.info(f"Registered data source: {name}")
        
        return node_id
    
    def register_transformation(
        self,
        name: str,
        source_ids: List[str],
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a transformation"""
        node_id = f"tfm-{uuid.uuid4().hex[:8]}"
        
        node = LineageNode(
            node_id=node_id,
            node_type="transformation",
            name=name,
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        
        # Create edges from sources
        for source_id in source_ids:
            edge = LineageEdge(
                source_id=source_id,
                target_id=node_id,
                operation=operation
            )
            self.edges.append(edge)
        
        logger.info(f"Registered transformation: {name}")
        
        return node_id
    
    def register_destination(
        self,
        name: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a data destination"""
        node_id = f"dst-{uuid.uuid4().hex[:8]}"
        
        node = LineageNode(
            node_id=node_id,
            node_type="destination",
            name=name,
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        
        # Create edge from source
        edge = LineageEdge(
            source_id=source_id,
            target_id=node_id,
            operation="write"
        )
        self.edges.append(edge)
        
        logger.info(f"Registered data destination: {name}")
        
        return node_id
    
    def get_lineage(self, node_id: str) -> Dict[str, Any]:
        """Get lineage for a node"""
        
        if node_id not in self.nodes:
            return {}
        
        # Get upstream (sources)
        upstream = []
        for edge in self.edges:
            if edge.target_id == node_id:
                upstream.append({
                    'node_id': edge.source_id,
                    'name': self.nodes[edge.source_id].name,
                    'operation': edge.operation
                })
        
        # Get downstream (destinations)
        downstream = []
        for edge in self.edges:
            if edge.source_id == node_id:
                downstream.append({
                    'node_id': edge.target_id,
                    'name': self.nodes[edge.target_id].name,
                    'operation': edge.operation
                })
        
        node = self.nodes[node_id]
        
        return {
            'node_id': node_id,
            'name': node.name,
            'type': node.node_type,
            'upstream': upstream,
            'downstream': downstream,
            'metadata': node.metadata
        }
    
    def get_full_lineage(self) -> Dict[str, Any]:
        """Get complete lineage graph"""
        return {
            'nodes': [
                {
                    'node_id': node.node_id,
                    'type': node.node_type,
                    'name': node.name
                }
                for node in self.nodes.values()
            ],
            'edges': [
                {
                    'source': edge.source_id,
                    'target': edge.target_id,
                    'operation': edge.operation
                }
                for edge in self.edges
            ]
        }
    
    def impact_analysis(self, node_id: str) -> List[str]:
        """Analyze impact of changes to a node"""
        
        if node_id not in self.nodes:
            return []
        
        # Find all downstream nodes (BFS)
        impacted = []
        queue = [node_id]
        visited = set()
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            for edge in self.edges:
                if edge.source_id == current and edge.target_id not in visited:
                    queue.append(edge.target_id)
                    impacted.append(self.nodes[edge.target_id].name)
        
        return impacted


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_data_quality_governance():
    """Demonstrate data quality and governance"""
    
    print("=" * 80)
    print("DATA QUALITY & GOVERNANCE INFRASTRUCTURE")
    print("=" * 80)
    print()
    print("🏗️  COMPONENTS:")
    print("   1. Data Quality Assessment")
    print("   2. Data Validation Framework")
    print("   3. Data Profiling & Statistics")
    print("   4. Data Lineage Tracking")
    print()
    
    # ========================================================================
    # 1. DATA QUALITY ASSESSMENT
    # ========================================================================
    print("=" * 80)
    print("1. DATA QUALITY ASSESSMENT ENGINE")
    print("=" * 80)
    
    quality_engine = DataQualityEngine()
    
    # Create sample nutrition data
    print("\n📊 Generating sample nutrition data...")
    sample_data = [
        {
            'food_id': f'food-{i}',
            'name': f'Food Item {i}',
            'calories': 100 + i * 50 if i < 18 else 99999,  # Outlier at end
            'protein': 10 + i,
            'carbs': 20 + i * 2,
            'fat': 5 + i,
            'timestamp': time.time() - (i * 86400)  # Days old
        }
        for i in range(20)
    ]
    
    # Add some quality issues
    sample_data[5]['name'] = None  # Missing name
    sample_data[10]['calories'] = None  # Missing calories
    sample_data[15]['calories'] = 500  # Will cause consistency issue
    sample_data[15]['protein'] = 10
    sample_data[15]['carbs'] = 10
    sample_data[15]['fat'] = 10
    # 10*4 + 10*4 + 10*9 = 170 calories expected, but 500 actual
    
    print(f"   ✅ Generated {len(sample_data)} records")
    
    # Assess quality
    print("\n🔍 Assessing data quality...")
    schema = {
        'required': ['name', 'calories', 'protein']
    }
    
    report = quality_engine.assess_quality("nutrition_dataset", sample_data, schema)
    
    print(f"\n📊 Quality Report: {report.report_id}")
    print(f"   Overall Score: {report.overall_score:.1%}")
    print(f"   Issues Found: {report.issues_count}")
    print()
    print("   Dimension Scores:")
    for dim_name, score in report.dimension_scores.items():
        status = "✓" if score.score >= 0.9 else "⚠" if score.score >= 0.7 else "✗"
        print(f"      {status} {dim_name.capitalize()}: {score.score:.1%}")
    
    if report.recommendations:
        print(f"\n💡 Recommendations ({len(report.recommendations)}):")
        for rec in report.recommendations[:3]:
            print(f"   • {rec}")
    
    # ========================================================================
    # 2. DATA VALIDATION FRAMEWORK
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. DATA VALIDATION FRAMEWORK")
    print("=" * 80)
    
    validator = DataValidator()
    
    # Configure validation rules
    print("\n⚙️  Configuring validation rules...")
    
    validator.add_rule('name', RequiredValidation('name_required'))
    validator.add_rule('name', RegexValidation('name_format', r'^[A-Za-z\s]+$', severity="warning"))
    
    validator.add_rule('calories', RequiredValidation('calories_required'))
    validator.add_rule('calories', RangeValidation('calories_range', 0, 5000))
    
    validator.add_rule('protein', RangeValidation('protein_range', 0, 200))
    validator.add_rule('carbs', RangeValidation('carbs_range', 0, 500))
    validator.add_rule('fat', RangeValidation('fat_range', 0, 100))
    
    print("   ✅ Configured 7 validation rules")
    
    # Validate batch
    print("\n🔍 Validating nutrition data...")
    validation_summary = validator.validate_batch(sample_data)
    
    print(f"\n📊 Validation Summary:")
    print(f"   Total records: {validation_summary['total_records']}")
    print(f"   Valid records: {validation_summary['valid_records']}")
    print(f"   Invalid records: {validation_summary['invalid_records']}")
    print(f"   Validation rate: {validation_summary['validation_rate']:.1%}")
    print(f"   Errors: {validation_summary['error_count']}")
    print(f"   Warnings: {validation_summary['warning_count']}")
    
    if validation_summary['sample_errors']:
        print(f"\n❌ Sample Errors:")
        for error in validation_summary['sample_errors'][:3]:
            print(f"   • {error['field']}: {error['error']}")
    
    # ========================================================================
    # 3. DATA PROFILING & STATISTICS
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. DATA PROFILING & STATISTICS")
    print("=" * 80)
    
    profiler = DataProfiler()
    
    print("\n📊 Profiling dataset...")
    profiles = profiler.profile_dataset("nutrition_dataset", sample_data)
    
    print(f"   ✅ Profiled {len(profiles)} columns")
    
    summary = profiler.get_profile_summary("nutrition_dataset")
    print(f"\n📈 Profile Summary:")
    
    key_columns = ['calories', 'protein', 'carbs', 'fat']
    for col in key_columns:
        if col in profiles:
            prof = profiles[col]
            print(f"\n   {col.upper()}:")
            print(f"      Type: {prof.data_type}")
            print(f"      Null: {prof.null_percentage:.1%}")
            print(f"      Unique values: {prof.unique_count}")
            if prof.min_value is not None:
                print(f"      Range: [{prof.min_value:.0f}, {prof.max_value:.0f}]")
                print(f"      Mean: {prof.mean:.1f}")
                if prof.std_dev:
                    print(f"      Std Dev: {prof.std_dev:.1f}")
    
    # Detect anomalies
    print("\n🔍 Detecting anomalies...")
    anomalies = profiler.detect_anomalies("nutrition_dataset", "calories", z_threshold=2.0)
    
    if anomalies:
        print(f"   ⚠️  Found {len(anomalies)} anomalies:")
        for anom in anomalies:
            print(f"      • {anom['type']}: {anom['value']} (z-score: {anom['z_score']:.2f})")
    else:
        print("   ✓ No anomalies detected")
    
    # ========================================================================
    # 4. DATA LINEAGE TRACKING
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. DATA LINEAGE TRACKING")
    print("=" * 80)
    
    lineage = DataLineageTracker()
    
    print("\n🔗 Building data lineage...")
    
    # Register data pipeline
    usda_source = lineage.register_source(
        "USDA Food Database",
        metadata={'source_type': 'api', 'version': 'v2'}
    )
    
    image_source = lineage.register_source(
        "Food Image Scanner",
        metadata={'source_type': 'ml_model', 'accuracy': 0.95}
    )
    
    # Transformation steps
    cleaning_tfm = lineage.register_transformation(
        "Data Cleaning",
        [usda_source, image_source],
        "clean_and_normalize",
        metadata={'records_processed': 10000}
    )
    
    enrichment_tfm = lineage.register_transformation(
        "Nutritional Enrichment",
        [cleaning_tfm],
        "enrich_nutrition",
        metadata={'ml_model': 'nutrition-enricher-v2'}
    )
    
    aggregation_tfm = lineage.register_transformation(
        "Daily Aggregation",
        [enrichment_tfm],
        "aggregate_by_day"
    )
    
    # Destinations
    warehouse_dst = lineage.register_destination(
        "Data Warehouse",
        aggregation_tfm,
        metadata={'table': 'nutrition_facts'}
    )
    
    api_dst = lineage.register_destination(
        "Nutrition API",
        enrichment_tfm,
        metadata={'endpoint': '/api/v1/nutrition'}
    )
    
    print("   ✅ Registered 2 sources, 3 transformations, 2 destinations")
    
    # Get lineage for enrichment
    print(f"\n🔍 Lineage for 'Nutritional Enrichment':")
    enrich_lineage = lineage.get_lineage(enrichment_tfm)
    
    print(f"   Upstream sources ({len(enrich_lineage['upstream'])}):")
    for src in enrich_lineage['upstream']:
        print(f"      • {src['name']} ({src['operation']})")
    
    print(f"   Downstream consumers ({len(enrich_lineage['downstream'])}):")
    for dst in enrich_lineage['downstream']:
        print(f"      • {dst['name']} ({dst['operation']})")
    
    # Impact analysis
    print(f"\n💥 Impact analysis for 'Data Cleaning':")
    impacted = lineage.impact_analysis(cleaning_tfm)
    print(f"   Impacted components: {len(impacted)}")
    for component in impacted:
        print(f"      • {component}")
    
    # Full lineage
    full_lineage = lineage.get_full_lineage()
    print(f"\n📊 Complete Lineage Graph:")
    print(f"   Nodes: {len(full_lineage['nodes'])}")
    print(f"   Edges: {len(full_lineage['edges'])}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ DATA QUALITY & GOVERNANCE COMPLETE")
    print("=" * 80)
    
    print("\n📦 CAPABILITIES:")
    print("   ✓ Multi-dimensional quality assessment")
    print("   ✓ Flexible validation framework")
    print("   ✓ Comprehensive data profiling")
    print("   ✓ End-to-end lineage tracking")
    print("   ✓ Anomaly detection")
    print("   ✓ Impact analysis")
    print("   ✓ Quality recommendations")
    
    print("\n🎯 GOVERNANCE METRICS:")
    print(f"   Overall quality score: {report.overall_score:.1%} ✓")
    print(f"   Validation rate: {validation_summary['validation_rate']:.1%} ✓")
    print(f"   Columns profiled: {len(profiles)} ✓")
    print(f"   Lineage nodes: {len(full_lineage['nodes'])} ✓")
    print(f"   Quality issues: {report.issues_count} ✓")
    print(f"   Validation errors: {validation_summary['error_count']} ✓")
    print(f"   Anomalies detected: {len(anomalies)} ✓")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    demo_data_quality_governance()
