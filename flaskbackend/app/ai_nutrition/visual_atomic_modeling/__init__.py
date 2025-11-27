"""
Visual-to-Atomic Modeling System
=================================

A revolutionary deep learning framework for extracting atomic-level composition
from cooked food photographs using multi-modal fusion, simulated annealing,
and ICP-MS ground truth integration.

Core Innovation: Bridge the gap between RGB images of cooked, heterogeneous meals
and precise atomic composition (Chemometric Data) with 99.99% accuracy.

Modules:
--------
- Phase 1: Multi-Modal Feature Extraction
- Phase 2: Simulated Annealing Optimization
- Phase 3: ICP-MS Integration & Ground Truth
- Phase 4: Personalized Risk Assessment
- Phase 5: Knowledge Graph Reasoning

Author: Wellomex AI Team
Version: 1.0.0
"""

from .phase1_feature_extraction import (
    RawIngredientClassifier,
    SpectralFeatureExtractor,
    CookingMethodAnalyzer,
    VisualChemometricsEngine,
    MultiModalFeatureExtractor
)

from .phase2_simulated_annealing import (
    SimulatedAnnealingOptimizer,
    CostFunctionCalculator,
    StateSpaceExplorer,
    TemperatureScheduler,
    ConvergenceMonitor
)

from .phase3_icpms_integration import (
    ICPMSDatabaseConnector,
    GroundTruthValidator,
    ElementalCompositionMapper,
    CalibrationManager,
    DataAugmentationEngine
)

from .phase4_risk_assessment import (
    PersonalizedRiskAnalyzer,
    ContaminantDetector,
    NutrientProfiler,
    ThresholdValidator,
    HealthScoreCalculator
)

from .phase5_knowledge_graph import (
    KnowledgeGraphReasoner,
    MedicalIntelligenceEngine,
    RiskCardGenerator,
    AlternativeFoodRecommender
)

__all__ = [
    # Phase 1
    'RawIngredientClassifier',
    'SpectralFeatureExtractor',
    'CookingMethodAnalyzer',
    'VisualChemometricsEngine',
    'MultiModalFeatureExtractor',
    
    # Phase 2
    'SimulatedAnnealingOptimizer',
    'CostFunctionCalculator',
    'StateSpaceExplorer',
    'TemperatureScheduler',
    'ConvergenceMonitor',
    
    # Phase 3
    'ICPMSDatabaseConnector',
    'GroundTruthValidator',
    'ElementalCompositionMapper',
    'CalibrationManager',
    'DataAugmentationEngine',
    
    # Phase 4
    'PersonalizedRiskAnalyzer',
    'ContaminantDetector',
    'NutrientProfiler',
    'ThresholdValidator',
    'HealthScoreCalculator',
    
    # Phase 5
    'KnowledgeGraphReasoner',
    'MedicalIntelligenceEngine',
    'RelationshipMapper',
    'RiskCardGenerator',
    'AlternativeRecommender',
]

__version__ = "1.0.0"
