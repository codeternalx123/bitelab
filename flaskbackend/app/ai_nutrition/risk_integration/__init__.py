"""
Dynamic Risk Integration Layer
==============================

Connects high-resolution atomic detection to personalized health risk assessment.

This module implements the critical bridge between:
- Visual Chemometrics (atomic composition predictions)
- Health Profile Analysis (user's medical conditions)
- Regulatory Compliance (FDA, WHO, NKF, KDIGO)
- Personalized Safety Decisions (dynamic thresholds)

Author: BiteLab AI Team
Date: November 2025
Version: 1.0.0
"""

from .dynamic_thresholds import (
    DynamicThresholdDatabase,
    MedicalThreshold,
    ThresholdRule,
    RegulatoryAuthority
)

from .health_profile_engine import (
    HealthProfileEngine,
    UserHealthProfile,
    HealthCondition,
    RiskLevel
)

from .risk_integration_engine import (
    RiskIntegrationEngine,
    AtomicRiskAssessment,
    ElementRiskStatus
)

from .personalized_warning_system import (
    PersonalizedWarningSystem,
    ComprehensiveWarning as PersonalizedWarning,
    WarningMessage as WarningPriority
)

from .alternative_food_finder import (
    AlternativeFoodFinder,
    AlternativeScore as FoodAlternative,
    AlternativeScore as AlternativeSearchCriteria
)

__all__ = [
    # Dynamic Thresholds
    'DynamicThresholdDatabase',
    'MedicalThreshold',
    'ThresholdRule',
    'RegulatoryAuthority',
    
    # Health Profile
    'HealthProfileEngine',
    'UserHealthProfile',
    'HealthCondition',
    'RiskLevel',
    
    # Risk Integration
    'RiskIntegrationEngine',
    'AtomicRiskAssessment',
    'ElementRiskStatus',
    
    # Personalized Warnings
    'PersonalizedWarningSystem',
    'PersonalizedWarning',
    'WarningPriority',
    
    # Alternative Finder
    'AlternativeFoodFinder',
    'FoodAlternative',
    'AlternativeSearchCriteria'
]
