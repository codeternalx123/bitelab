"""
Hyperspectral Classification Algorithms
========================================

Advanced classification methods for hyperspectral image analysis
and atomic composition prediction.

Key Algorithms:
- Spectral Angle Mapper (SAM)
- Spectral Information Divergence (SID)
- Spectral Correlation Mapper (SCM)
- Support Vector Machines (SVM) with RBF kernel
- Random Forest with spectral features
- k-Nearest Neighbors (k-NN) spectral
- Neural Network classifiers
- Ensemble methods

Classifies pixels based on spectral signatures for element identification.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    from scipy.spatial.distance import cosine, euclidean
    from scipy.stats import entropy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.svm import SVC, SVR
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class ClassificationMethod(Enum):
    """Classification algorithms"""
    SAM = "sam"  # Spectral Angle Mapper
    SID = "sid"  # Spectral Information Divergence
    SCM = "scm"  # Spectral Correlation Mapper
    SVM_RBF = "svm_rbf"  # SVM with RBF kernel
    SVM_LINEAR = "svm_linear"  # SVM with linear kernel
    RANDOM_FOREST = "random_forest"
    KNN = "knn"  # k-Nearest Neighbors
    ENSEMBLE = "ensemble"  # Combination of methods
    MLP = "mlp"  # Multi-layer Perceptron


@dataclass
class ClassificationConfig:
    """Configuration for classification"""
    
    method: ClassificationMethod = ClassificationMethod.SAM
    task: str = "regression"  # regression or classification
    
    # SAM parameters
    sam_threshold: float = 0.1  # Radians
    
    # SID parameters
    sid_threshold: float = 0.5
    
    # SVM parameters
    svm_c: float = 1.0
    svm_gamma: str = "scale"  # scale or auto
    svm_kernel: str = "rbf"
    
    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    
    # k-NN parameters
    knn_n_neighbors: int = 5
    knn_weights: str = "distance"  # uniform or distance
    
    # Ensemble parameters
    ensemble_methods: List[ClassificationMethod] = field(default_factory=lambda: [
        ClassificationMethod.SAM,
        ClassificationMethod.RANDOM_FOREST,
        ClassificationMethod.SVM_RBF
    ])
    ensemble_weights: Optional[List[float]] = None
    
    # General
    normalize_spectra: bool = True


@dataclass
class ClassificationResult:
    """Results from classification"""
    
    predictions: np.ndarray  # Predicted values/classes
    confidence: Optional[np.ndarray] = None  # Confidence scores
    probabilities: Optional[np.ndarray] = None  # Class probabilities
    
    # For regression
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    
    # For classification
    accuracy: Optional[float] = None
    
    # Distances/similarities
    distances: Optional[np.ndarray] = None
    
    def reshape_to_image(self, height: int, width: int) -> np.ndarray:
        """Reshape predictions to image format"""
        if self.predictions.ndim == 1:
            return self.predictions.reshape(height, width)
        else:
            return self.predictions.reshape(height, width, -1)


class HyperspectralClassifier:
    """
    Classify hyperspectral pixels for atomic composition prediction
    """
    
    def __init__(self, config: Optional[ClassificationConfig] = None):
        """
        Initialize classifier
        
        Args:
            config: Classification configuration
        """
        self.config = config or ClassificationConfig()
        self.model = None
        self.scaler = None
        self.reference_spectra = None
        
        logger.info(f"Initialized hyperspectral classifier:")
        logger.info(f"  Method: {self.config.method.value}")
        logger.info(f"  Task: {self.config.task}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        reference_spectra: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Train classifier
        
        Args:
            X: Training spectra [num_samples, num_bands]
            y: Training labels/values [num_samples] or [num_samples, num_targets]
            reference_spectra: Reference library for spectral matching methods
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D training data [N, bands], got shape {X.shape}")
        
        logger.info(f"Training classifier on {X.shape[0]} samples")
        
        # Store reference spectra for matching-based methods
        if reference_spectra is not None:
            self.reference_spectra = reference_spectra
        
        # Normalize if requested
        if self.config.normalize_spectra and HAS_SKLEARN:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        method = self.config.method
        
        # Spectral matching methods don't need training
        if method in [ClassificationMethod.SAM, ClassificationMethod.SID, ClassificationMethod.SCM]:
            logger.info("Spectral matching method - no training needed")
            return
        
        # Train supervised models
        if method == ClassificationMethod.SVM_RBF or method == ClassificationMethod.SVM_LINEAR:
            self._train_svm(X, y)
        
        elif method == ClassificationMethod.RANDOM_FOREST:
            self._train_random_forest(X, y)
        
        elif method == ClassificationMethod.KNN:
            self._train_knn(X, y)
        
        elif method == ClassificationMethod.ENSEMBLE:
            self._train_ensemble(X, y)
        
        else:
            raise ValueError(f"Unknown classification method: {method}")
    
    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = False
    ) -> ClassificationResult:
        """
        Predict on new data
        
        Args:
            X: Test spectra [num_samples, num_bands]
            return_confidence: Whether to return confidence scores
            
        Returns:
            ClassificationResult with predictions
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D test data [N, bands], got shape {X.shape}")
        
        logger.info(f"Predicting on {X.shape[0]} samples")
        
        # Normalize if fitted with scaler
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        method = self.config.method
        
        # Spectral matching methods
        if method == ClassificationMethod.SAM:
            return self._predict_sam(X)
        elif method == ClassificationMethod.SID:
            return self._predict_sid(X)
        elif method == ClassificationMethod.SCM:
            return self._predict_scm(X)
        
        # Supervised methods
        elif method in [ClassificationMethod.SVM_RBF, ClassificationMethod.SVM_LINEAR]:
            return self._predict_svm(X, return_confidence)
        elif method == ClassificationMethod.RANDOM_FOREST:
            return self._predict_rf(X, return_confidence)
        elif method == ClassificationMethod.KNN:
            return self._predict_knn(X, return_confidence)
        elif method == ClassificationMethod.ENSEMBLE:
            return self._predict_ensemble(X, return_confidence)
        
        else:
            raise ValueError(f"Unknown classification method: {method}")
    
    def _train_svm(self, X: np.ndarray, y: np.ndarray):
        """Train SVM"""
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required for SVM")
        
        if self.config.task == "classification":
            self.model = SVC(
                C=self.config.svm_c,
                kernel=self.config.svm_kernel,
                gamma=self.config.svm_gamma,
                probability=True,
                random_state=42
            )
        else:
            self.model = SVR(
                C=self.config.svm_c,
                kernel=self.config.svm_kernel,
                gamma=self.config.svm_gamma
            )
        
        # Fit
        if y.ndim > 1 and self.config.task == "regression":
            # Multi-output regression
            from sklearn.multioutput import MultiOutputRegressor
            self.model = MultiOutputRegressor(self.model)
        
        self.model.fit(X, y)
        
        logger.info(f"Trained SVM with {self.config.svm_kernel} kernel")
    
    def _train_random_forest(self, X: np.ndarray, y: np.ndarray):
        """Train Random Forest"""
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required for Random Forest")
        
        if self.config.task == "classification":
            self.model = RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                random_state=42,
                n_jobs=-1
            )
        
        self.model.fit(X, y)
        
        logger.info(f"Trained Random Forest with {self.config.rf_n_estimators} trees")
    
    def _train_knn(self, X: np.ndarray, y: np.ndarray):
        """Train k-NN"""
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required for k-NN")
        
        if self.config.task == "classification":
            self.model = KNeighborsClassifier(
                n_neighbors=self.config.knn_n_neighbors,
                weights=self.config.knn_weights
            )
        else:
            self.model = KNeighborsRegressor(
                n_neighbors=self.config.knn_n_neighbors,
                weights=self.config.knn_weights
            )
        
        self.model.fit(X, y)
        
        logger.info(f"Trained k-NN with k={self.config.knn_n_neighbors}")
    
    def _train_ensemble(self, X: np.ndarray, y: np.ndarray):
        """Train ensemble of classifiers"""
        logger.info("Training ensemble of classifiers")
        
        # This would train multiple models
        # For now, placeholder
        self.model = {"ensemble": True}
    
    def _predict_sam(self, X: np.ndarray) -> ClassificationResult:
        """
        Spectral Angle Mapper
        Computes angle between test and reference spectra
        """
        if self.reference_spectra is None:
            raise ValueError("Reference spectra required for SAM")
        
        num_samples = X.shape[0]
        num_classes = len(self.reference_spectra)
        
        # Compute angles to all references
        angles = np.zeros((num_samples, num_classes))
        
        ref_names = list(self.reference_spectra.keys())
        
        for i, name in enumerate(ref_names):
            reference = self.reference_spectra[name]
            
            # Compute spectral angle for each sample
            for j in range(num_samples):
                test_spectrum = X[j]
                
                # Spectral angle (radians)
                cos_angle = np.dot(test_spectrum, reference) / (
                    np.linalg.norm(test_spectrum) * np.linalg.norm(reference) + 1e-10
                )
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                angles[j, i] = angle
        
        # Predictions: class with minimum angle
        predictions = np.argmin(angles, axis=1)
        
        # Confidence: inverse of minimum angle
        min_angles = np.min(angles, axis=1)
        confidence = 1.0 / (1.0 + min_angles)
        
        return ClassificationResult(
            predictions=predictions,
            confidence=confidence,
            distances=angles
        )
    
    def _predict_sid(self, X: np.ndarray) -> ClassificationResult:
        """
        Spectral Information Divergence
        Measures divergence between probability distributions
        """
        if not HAS_SCIPY:
            logger.warning("scipy not available, falling back to SAM")
            return self._predict_sam(X)
        
        if self.reference_spectra is None:
            raise ValueError("Reference spectra required for SID")
        
        num_samples = X.shape[0]
        num_classes = len(self.reference_spectra)
        
        # Compute divergences
        divergences = np.zeros((num_samples, num_classes))
        
        ref_names = list(self.reference_spectra.keys())
        
        for i, name in enumerate(ref_names):
            reference = self.reference_spectra[name]
            
            # Convert to probability distributions
            ref_prob = reference / (np.sum(reference) + 1e-10)
            
            for j in range(num_samples):
                test_spectrum = X[j]
                test_prob = test_spectrum / (np.sum(test_spectrum) + 1e-10)
                
                # Symmetric KL divergence
                div = entropy(test_prob, ref_prob) + entropy(ref_prob, test_prob)
                
                divergences[j, i] = div
        
        # Predictions: class with minimum divergence
        predictions = np.argmin(divergences, axis=1)
        
        # Confidence
        min_div = np.min(divergences, axis=1)
        confidence = 1.0 / (1.0 + min_div)
        
        return ClassificationResult(
            predictions=predictions,
            confidence=confidence,
            distances=divergences
        )
    
    def _predict_scm(self, X: np.ndarray) -> ClassificationResult:
        """
        Spectral Correlation Mapper
        Computes correlation between spectra
        """
        if self.reference_spectra is None:
            raise ValueError("Reference spectra required for SCM")
        
        num_samples = X.shape[0]
        num_classes = len(self.reference_spectra)
        
        # Compute correlations
        correlations = np.zeros((num_samples, num_classes))
        
        ref_names = list(self.reference_spectra.keys())
        
        for i, name in enumerate(ref_names):
            reference = self.reference_spectra[name]
            
            for j in range(num_samples):
                test_spectrum = X[j]
                
                # Pearson correlation
                corr = np.corrcoef(test_spectrum, reference)[0, 1]
                
                correlations[j, i] = corr
        
        # Predictions: class with maximum correlation
        predictions = np.argmax(correlations, axis=1)
        
        # Confidence: correlation value
        max_corr = np.max(correlations, axis=1)
        confidence = (max_corr + 1) / 2  # Map [-1, 1] to [0, 1]
        
        return ClassificationResult(
            predictions=predictions,
            confidence=confidence,
            distances=1 - correlations  # Distance = 1 - correlation
        )
    
    def _predict_svm(self, X: np.ndarray, return_confidence: bool) -> ClassificationResult:
        """Predict with SVM"""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        predictions = self.model.predict(X)
        
        confidence = None
        probabilities = None
        
        if return_confidence and self.config.task == "classification":
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)
                confidence = np.max(probabilities, axis=1)
        
        return ClassificationResult(
            predictions=predictions,
            confidence=confidence,
            probabilities=probabilities
        )
    
    def _predict_rf(self, X: np.ndarray, return_confidence: bool) -> ClassificationResult:
        """Predict with Random Forest"""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        predictions = self.model.predict(X)
        
        confidence = None
        probabilities = None
        
        if return_confidence:
            if self.config.task == "classification":
                probabilities = self.model.predict_proba(X)
                confidence = np.max(probabilities, axis=1)
            else:
                # For regression: use std of tree predictions as uncertainty
                # Get predictions from all trees
                tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
                std = np.std(tree_predictions, axis=0)
                confidence = 1.0 / (1.0 + std)  # Inverse of uncertainty
        
        return ClassificationResult(
            predictions=predictions,
            confidence=confidence,
            probabilities=probabilities
        )
    
    def _predict_knn(self, X: np.ndarray, return_confidence: bool) -> ClassificationResult:
        """Predict with k-NN"""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        predictions = self.model.predict(X)
        
        confidence = None
        probabilities = None
        
        if return_confidence and self.config.task == "classification":
            probabilities = self.model.predict_proba(X)
            confidence = np.max(probabilities, axis=1)
        
        return ClassificationResult(
            predictions=predictions,
            confidence=confidence,
            probabilities=probabilities
        )
    
    def _predict_ensemble(self, X: np.ndarray, return_confidence: bool) -> ClassificationResult:
        """Predict with ensemble"""
        # Placeholder for ensemble prediction
        # Would combine predictions from multiple models
        raise NotImplementedError("Ensemble prediction not yet implemented")
    
    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate classifier performance
        
        Args:
            X: Test spectra
            y_true: True labels/values
            
        Returns:
            Dictionary of metrics
        """
        result = self.predict(X)
        y_pred = result.predictions
        
        metrics = {}
        
        if self.config.task == "classification":
            if HAS_SKLEARN:
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
        else:
            # Regression metrics
            if HAS_SKLEARN:
                if y_true.ndim == 1:
                    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                    metrics['mae'] = np.mean(np.abs(y_true - y_pred))
                    metrics['r2'] = r2_score(y_true, y_pred)
                else:
                    # Multi-output
                    for i in range(y_true.shape[1]):
                        metrics[f'rmse_target{i}'] = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
                        metrics[f'r2_target{i}'] = r2_score(y_true[:, i], y_pred[:, i])
            else:
                metrics['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
                metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        
        return metrics


if __name__ == '__main__':
    # Example usage
    print("Hyperspectral Classification Example")
    print("=" * 60)
    
    # Create synthetic dataset
    num_train = 500
    num_test = 100
    num_bands = 50
    num_classes = 5
    
    np.random.seed(42)
    
    # Generate class-specific spectra
    class_means = []
    for i in range(num_classes):
        # Random mean spectrum for each class
        mean = np.random.rand(num_bands)
        class_means.append(mean)
    
    # Generate training data
    X_train = []
    y_train = []
    
    for i in range(num_train):
        class_idx = i % num_classes
        spectrum = class_means[class_idx] + np.random.randn(num_bands) * 0.1
        X_train.append(spectrum)
        y_train.append(class_idx)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Generate test data
    X_test = []
    y_test = []
    
    for i in range(num_test):
        class_idx = i % num_classes
        spectrum = class_means[class_idx] + np.random.randn(num_bands) * 0.1
        X_test.append(spectrum)
        y_test.append(class_idx)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\nTraining data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Number of classes: {num_classes}")
    
    # Test different methods
    methods = []
    
    if HAS_SKLEARN:
        methods.extend([
            ClassificationMethod.SVM_RBF,
            ClassificationMethod.RANDOM_FOREST,
            ClassificationMethod.KNN
        ])
    
    # Add spectral matching with reference library
    reference_spectra = {f"class_{i}": class_means[i] for i in range(num_classes)}
    
    methods.extend([
        ClassificationMethod.SAM,
        ClassificationMethod.SCM
    ])
    
    if HAS_SCIPY:
        methods.append(ClassificationMethod.SID)
    
    for method in methods:
        print(f"\n{method.value.upper()}:")
        print("-" * 60)
        
        config = ClassificationConfig(
            method=method,
            task="classification"
        )
        
        classifier = HyperspectralClassifier(config)
        
        try:
            # Train (if supervised)
            if method in [ClassificationMethod.SAM, ClassificationMethod.SID, ClassificationMethod.SCM]:
                classifier.fit(X_train, y_train, reference_spectra=reference_spectra)
            else:
                classifier.fit(X_train, y_train)
            
            # Evaluate
            metrics = classifier.evaluate(X_test, y_test)
            
            print(f"Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
        
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n✅ Classification complete!")
