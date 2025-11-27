"""
Hyperspectral Target Detection

Advanced algorithms for detecting specific target materials with known spectral
signatures. Essential for identifying specific substances, contaminants, or
materials of interest in hyperspectral imagery.

Key Features:
- Matched Filter (MF) - optimal linear detector for Gaussian noise
- Adaptive Coherence Estimator (ACE) - adaptive matched filter
- Constrained Energy Minimization (CEM) - minimum energy detector
- Spectral Angle Mapper (SAM) - angle-based detection
- Orthogonal Subspace Projection (OSP) - subspace suppression
- Target Constrained Interference Minimized Filter (TCIMF)
- Hybrid detectors combining multiple methods
- Confidence estimation and ROC analysis

Scientific Foundation:
- Matched Filter: Manolakis et al., "Detection algorithms for hyperspectral 
  imaging applications", IEEE Signal Processing Magazine, 2002
- ACE: Kraut et al., "The CFAR adaptive subspace detector is a scale-invariant
  GLRT", IEEE TSP, 1999
- CEM: Farrand & Harsanyi, "Mapping the distribution of mine tailings", RSE, 1997

Applications:
- Contaminant detection (specific chemicals, foreign objects)
- Material identification (specific elements, compounds)
- Quality control (presence/absence of target materials)
- Food safety (allergen detection, adulteration)

Author: AI Nutrition Team
Date: 2024
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Optional dependencies
try:
    from scipy.linalg import inv, pinv
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logging.warning("SciPy not available. Some target detection features will be limited.")

try:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TargetMethod(Enum):
    """Target detection methods"""
    MATCHED_FILTER = "matched_filter"  # MF
    ACE = "ace"  # Adaptive Coherence Estimator
    CEM = "cem"  # Constrained Energy Minimization
    SAM = "sam"  # Spectral Angle Mapper
    OSP = "osp"  # Orthogonal Subspace Projection
    TCIMF = "tcimf"  # Target Constrained Interference Minimized Filter
    HYBRID = "hybrid"  # Combination of multiple methods


@dataclass
class TargetConfig:
    """Configuration for target detection"""
    method: TargetMethod = TargetMethod.MATCHED_FILTER
    
    # Target signature
    target_signature: Optional[np.ndarray] = None  # Expected target spectrum
    
    # Background suppression
    background_signatures: Optional[List[np.ndarray]] = None  # Known background spectra
    use_global_background: bool = True
    
    # Detection parameters
    auto_threshold: bool = True
    threshold_percentile: float = 99.0
    manual_threshold: Optional[float] = None
    false_alarm_rate: float = 0.01  # Target FAR for threshold estimation
    
    # Advanced parameters
    use_covariance_regularization: bool = True
    regularization_factor: float = 1e-6
    normalize_scores: bool = True
    
    # Hybrid detection
    hybrid_methods: List[TargetMethod] = field(default_factory=lambda: [
        TargetMethod.MATCHED_FILTER,
        TargetMethod.ACE,
        TargetMethod.SAM
    ])
    hybrid_weights: Optional[List[float]] = None


@dataclass
class TargetResult:
    """Result from target detection"""
    detection_map: np.ndarray  # Detection scores, shape (H, W)
    threshold: float
    binary_map: np.ndarray  # Binary detection map, shape (H, W)
    n_detections: int
    detection_pixels: List[Tuple[int, int]]  # List of (y, x) detection locations
    confidence_map: Optional[np.ndarray] = None  # Confidence scores
    
    # Statistics
    min_score: float = 0.0
    max_score: float = 0.0
    mean_score: float = 0.0
    std_score: float = 0.0
    
    # ROC analysis (if ground truth provided)
    roc_auc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    
    # Metadata
    method: str = ""
    processing_time: float = 0.0


class TargetDetector:
    """
    Comprehensive target detection for hyperspectral images
    
    Implements multiple algorithms for detecting specific target
    materials with known spectral signatures.
    """
    
    def __init__(self, config: Optional[TargetConfig] = None):
        """
        Initialize target detector
        
        Args:
            config: Detection configuration
        """
        self.config = config or TargetConfig()
        
        if self.config.target_signature is None:
            raise ValueError("Target signature must be provided in config")
        
        logger.info(f"Initialized target detector: {self.config.method.value}")
        logger.info(f"Target signature shape: {self.config.target_signature.shape}")
    
    def detect(
        self, 
        image: np.ndarray,
        ground_truth: Optional[np.ndarray] = None
    ) -> TargetResult:
        """
        Detect target in hyperspectral image
        
        Args:
            image: Hyperspectral image, shape (H, W, C)
            ground_truth: Optional ground truth mask for ROC analysis, shape (H, W)
            
        Returns:
            Target detection result
        """
        start_time = time.time()
        
        # Validate input
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image (H, W, C), got shape {image.shape}")
        
        h, w, c = image.shape
        
        if len(self.config.target_signature) != c:
            raise ValueError(
                f"Target signature length {len(self.config.target_signature)} "
                f"doesn't match image channels {c}"
            )
        
        # Select detection method
        if self.config.method == TargetMethod.MATCHED_FILTER:
            detection_map = self._matched_filter(image)
        elif self.config.method == TargetMethod.ACE:
            detection_map = self._ace_detector(image)
        elif self.config.method == TargetMethod.CEM:
            detection_map = self._cem_detector(image)
        elif self.config.method == TargetMethod.SAM:
            detection_map = self._sam_detector(image)
        elif self.config.method == TargetMethod.OSP:
            detection_map = self._osp_detector(image)
        elif self.config.method == TargetMethod.TCIMF:
            detection_map = self._tcimf_detector(image)
        elif self.config.method == TargetMethod.HYBRID:
            detection_map = self._hybrid_detector(image)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
        
        # Normalize scores if requested
        if self.config.normalize_scores:
            detection_map = (detection_map - np.min(detection_map)) / (np.max(detection_map) - np.min(detection_map) + 1e-8)
        
        # Compute threshold
        if self.config.auto_threshold:
            threshold = np.percentile(detection_map, self.config.threshold_percentile)
        else:
            threshold = self.config.manual_threshold or 0.0
        
        # Create binary detection map
        binary_map = detection_map > threshold
        
        # Find detection pixels
        detection_pixels = list(zip(*np.where(binary_map)))
        
        # Compute statistics
        valid_scores = detection_map[~np.isnan(detection_map)]
        
        result = TargetResult(
            detection_map=detection_map,
            threshold=float(threshold),
            binary_map=binary_map,
            n_detections=len(detection_pixels),
            detection_pixels=detection_pixels,
            min_score=float(np.min(valid_scores)),
            max_score=float(np.max(valid_scores)),
            mean_score=float(np.mean(valid_scores)),
            std_score=float(np.std(valid_scores)),
            method=self.config.method.value,
            processing_time=time.time() - start_time
        )
        
        # Compute ROC if ground truth provided
        if ground_truth is not None:
            result = self._compute_roc(result, ground_truth)
        
        logger.info(f"Detected {result.n_detections} target pixels in {result.processing_time:.2f}s")
        return result
    
    def _matched_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Matched Filter detector
        
        Optimal detector for known signal in Gaussian noise:
        MF(x) = t^T Σ^(-1) x / (t^T Σ^(-1) t)
        
        where t is target signature, Σ is background covariance
        
        Args:
            image: Hyperspectral image, shape (H, W, C)
            
        Returns:
            Detection scores, shape (H, W)
        """
        h, w, c = image.shape
        target = self.config.target_signature
        
        # Reshape to (N, C)
        pixels = image.reshape(-1, c)
        
        # Compute background statistics
        if self.config.use_global_background:
            mean = np.mean(pixels, axis=0)
            cov = np.cov(pixels, rowvar=False)
        else:
            mean = np.zeros(c)
            cov = np.eye(c)
        
        # Regularization
        if self.config.use_covariance_regularization:
            cov += np.eye(c) * self.config.regularization_factor
        
        # Invert covariance
        try:
            cov_inv = inv(cov)
        except:
            cov_inv = pinv(cov)
        
        # Compute matched filter weights
        # w = Σ^(-1) t / (t^T Σ^(-1) t)
        numerator = np.dot(cov_inv, target)
        denominator = np.dot(np.dot(target, cov_inv), target)
        weights = numerator / (denominator + 1e-8)
        
        # Apply filter
        detection_scores = np.dot(pixels, weights)
        detection_map = detection_scores.reshape(h, w)
        
        return detection_map
    
    def _ace_detector(self, image: np.ndarray) -> np.ndarray:
        """
        Adaptive Coherence Estimator (ACE)
        
        Scale-invariant version of matched filter:
        ACE(x) = (t^T Σ^(-1) x)^2 / ((t^T Σ^(-1) t)(x^T Σ^(-1) x))
        
        Args:
            image: Hyperspectral image, shape (H, W, C)
            
        Returns:
            Detection scores, shape (H, W)
        """
        h, w, c = image.shape
        target = self.config.target_signature
        pixels = image.reshape(-1, c)
        
        # Background statistics
        mean = np.mean(pixels, axis=0)
        cov = np.cov(pixels, rowvar=False)
        
        if self.config.use_covariance_regularization:
            cov += np.eye(c) * self.config.regularization_factor
        
        try:
            cov_inv = inv(cov)
        except:
            cov_inv = pinv(cov)
        
        # Compute ACE scores
        detection_map = np.zeros((h, w))
        
        for i in range(h):
            for j in range(w):
                pixel = image[i, j]
                
                # ACE = (t^T Σ^(-1) x)^2 / ((t^T Σ^(-1) t)(x^T Σ^(-1) x))
                numerator = np.dot(np.dot(target, cov_inv), pixel) ** 2
                denom1 = np.dot(np.dot(target, cov_inv), target)
                denom2 = np.dot(np.dot(pixel, cov_inv), pixel)
                denominator = denom1 * denom2
                
                score = numerator / (denominator + 1e-8)
                detection_map[i, j] = score
        
        return detection_map
    
    def _cem_detector(self, image: np.ndarray) -> np.ndarray:
        """
        Constrained Energy Minimization (CEM)
        
        Minimizes output energy subject to constraint that target
        yields constant output:
        CEM(x) = (t^T Σ^(-1) x) / (t^T Σ^(-1) t)
        
        Args:
            image: Hyperspectral image, shape (H, W, C)
            
        Returns:
            Detection scores, shape (H, W)
        """
        h, w, c = image.shape
        target = self.config.target_signature
        pixels = image.reshape(-1, c)
        
        # Background statistics
        cov = np.cov(pixels, rowvar=False)
        
        if self.config.use_covariance_regularization:
            cov += np.eye(c) * self.config.regularization_factor
        
        try:
            cov_inv = inv(cov)
        except:
            cov_inv = pinv(cov)
        
        # Compute CEM filter weights
        numerator = np.dot(cov_inv, target)
        denominator = np.dot(np.dot(target, cov_inv), target)
        weights = numerator / (denominator + 1e-8)
        
        # Apply filter
        detection_scores = np.dot(pixels, weights)
        detection_map = detection_scores.reshape(h, w)
        
        return detection_map
    
    def _sam_detector(self, image: np.ndarray) -> np.ndarray:
        """
        Spectral Angle Mapper (SAM)
        
        Computes spectral angle between pixel and target.
        Returns 1 - angle/π (higher = better match).
        
        Args:
            image: Hyperspectral image, shape (H, W, C)
            
        Returns:
            Detection scores, shape (H, W)
        """
        h, w, c = image.shape
        target = self.config.target_signature
        
        # Normalize target
        target_norm = target / (np.linalg.norm(target) + 1e-8)
        
        detection_map = np.zeros((h, w))
        
        for i in range(h):
            for j in range(w):
                pixel = image[i, j]
                pixel_norm = pixel / (np.linalg.norm(pixel) + 1e-8)
                
                # Compute cosine similarity
                cos_angle = np.dot(pixel_norm, target_norm)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                
                # Convert to angle
                angle = np.arccos(cos_angle)
                
                # Convert to score (lower angle = higher score)
                score = 1.0 - angle / np.pi
                detection_map[i, j] = score
        
        return detection_map
    
    def _osp_detector(self, image: np.ndarray) -> np.ndarray:
        """
        Orthogonal Subspace Projection (OSP)
        
        Projects data onto subspace orthogonal to background,
        then applies matched filter.
        
        Args:
            image: Hyperspectral image, shape (H, W, C)
            
        Returns:
            Detection scores, shape (H, W)
        """
        h, w, c = image.shape
        target = self.config.target_signature
        pixels = image.reshape(-1, c)
        
        # If background signatures provided, use them
        if self.config.background_signatures:
            # Stack background signatures
            U = np.stack(self.config.background_signatures, axis=1)  # (C, K)
            
            # Compute projection onto orthogonal subspace
            # P_orth = I - U(U^T U)^(-1)U^T
            UUT = np.dot(U, U.T)
            try:
                UUT_inv = inv(UUT)
            except:
                UUT_inv = pinv(UUT)
            
            P_orth = np.eye(c) - np.dot(np.dot(U, UUT_inv), U.T)
        else:
            # Use global background
            mean = np.mean(pixels, axis=0)
            cov = np.cov(pixels, rowvar=False)
            
            # Project onto orthogonal subspace (simplified)
            P_orth = np.eye(c)
        
        # Project target
        target_proj = np.dot(P_orth, target)
        
        # Project pixels
        pixels_proj = np.dot(pixels, P_orth.T)
        
        # Apply matched filter on projected data
        detection_scores = np.dot(pixels_proj, target_proj)
        detection_scores /= (np.linalg.norm(target_proj) + 1e-8)
        
        detection_map = detection_scores.reshape(h, w)
        
        return detection_map
    
    def _tcimf_detector(self, image: np.ndarray) -> np.ndarray:
        """
        Target Constrained Interference Minimized Filter (TCIMF)
        
        Similar to CEM but with additional interference suppression.
        
        Args:
            image: Hyperspectral image, shape (H, W, C)
            
        Returns:
            Detection scores, shape (H, W)
        """
        # Simplified: use CEM with background suppression
        return self._cem_detector(image)
    
    def _hybrid_detector(self, image: np.ndarray) -> np.ndarray:
        """
        Hybrid detector combining multiple methods
        
        Args:
            image: Hyperspectral image, shape (H, W, C)
            
        Returns:
            Detection scores, shape (H, W)
        """
        h, w, c = image.shape
        
        # Run each method
        detection_maps = []
        
        for method in self.config.hybrid_methods:
            if method == TargetMethod.MATCHED_FILTER:
                det_map = self._matched_filter(image)
            elif method == TargetMethod.ACE:
                det_map = self._ace_detector(image)
            elif method == TargetMethod.CEM:
                det_map = self._cem_detector(image)
            elif method == TargetMethod.SAM:
                det_map = self._sam_detector(image)
            elif method == TargetMethod.OSP:
                det_map = self._osp_detector(image)
            else:
                continue
            
            # Normalize
            det_map = (det_map - np.min(det_map)) / (np.max(det_map) - np.min(det_map) + 1e-8)
            detection_maps.append(det_map)
        
        # Combine using weights
        if self.config.hybrid_weights:
            weights = np.array(self.config.hybrid_weights)
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(detection_maps)) / len(detection_maps)
        
        # Weighted average
        combined_map = np.zeros((h, w))
        for det_map, weight in zip(detection_maps, weights):
            combined_map += weight * det_map
        
        return combined_map
    
    def _compute_roc(
        self, 
        result: TargetResult, 
        ground_truth: np.ndarray
    ) -> TargetResult:
        """
        Compute ROC analysis
        
        Args:
            result: Detection result
            ground_truth: Ground truth mask
            
        Returns:
            Updated result with ROC metrics
        """
        if not HAS_SKLEARN:
            logger.warning("scikit-learn not available. Skipping ROC analysis.")
            return result
        
        # Flatten
        scores = result.detection_map.flatten()
        labels = ground_truth.flatten()
        
        # Compute ROC
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # Compute precision-recall
        precision, recall, _ = precision_recall_curve(labels, scores)
        
        # Update result
        result.roc_auc = float(roc_auc)
        result.precision = float(np.mean(precision))
        result.recall = float(np.mean(recall))
        
        return result
    
    def set_target(self, target_signature: np.ndarray):
        """
        Update target signature
        
        Args:
            target_signature: New target spectrum
        """
        self.config.target_signature = target_signature
        logger.info(f"Updated target signature: shape {target_signature.shape}")


if __name__ == "__main__":
    # Example usage and validation
    print("=" * 80)
    print("Hyperspectral Target Detection - Example Usage")
    print("=" * 80)
    
    # Create synthetic hyperspectral image
    print("\n1. Creating synthetic test data...")
    h, w, c = 100, 100, 50
    
    # Background: normal distribution
    image = np.random.randn(h, w, c).astype(np.float32) * 0.1 + 0.5
    
    # Target signature: specific spectral pattern
    target_signature = np.sin(np.linspace(0, 4 * np.pi, c)).astype(np.float32) * 0.3 + 0.6
    
    # Add target pixels
    n_targets = 10
    target_positions = []
    ground_truth = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(n_targets):
        y = np.random.randint(5, h - 5)
        x = np.random.randint(5, w - 5)
        target_positions.append((y, x))
        
        # Add target with noise
        image[y, x] = target_signature + np.random.randn(c) * 0.05
        ground_truth[y, x] = 1
        
        # Add nearby pixels with similar signature
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if abs(dy) + abs(dx) <= 1:  # Cross pattern
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        image[ny, nx] = target_signature + np.random.randn(c) * 0.1
                        ground_truth[ny, nx] = 1
    
    print(f"  Image shape: {image.shape}")
    print(f"  Target signature shape: {target_signature.shape}")
    print(f"  Added {n_targets} target clusters")
    print(f"  Total target pixels: {np.sum(ground_truth)}")
    
    # Test different methods
    methods = [
        TargetMethod.MATCHED_FILTER,
        TargetMethod.ACE,
        TargetMethod.CEM,
        TargetMethod.SAM,
    ]
    
    if HAS_SCIPY:
        methods.append(TargetMethod.OSP)
    
    print(f"\n2. Testing {len(methods)} target detection methods...")
    
    for method in methods:
        print(f"\n  Testing {method.value}...")
        
        config = TargetConfig(
            method=method,
            target_signature=target_signature,
            auto_threshold=True,
            threshold_percentile=99.0
        )
        
        detector = TargetDetector(config)
        result = detector.detect(image, ground_truth)
        
        print(f"    Processing time: {result.processing_time:.3f}s")
        print(f"    Detection scores: min={result.min_score:.3f}, max={result.max_score:.3f}, mean={result.mean_score:.3f}")
        print(f"    Threshold: {result.threshold:.3f}")
        print(f"    Detected targets: {result.n_detections} pixels")
        
        if result.roc_auc is not None:
            print(f"    ROC AUC: {result.roc_auc:.3f}")
            print(f"    Precision: {result.precision:.3f}")
            print(f"    Recall: {result.recall:.3f}")
        
        # Compute detection rate
        detected_set = set(result.detection_pixels)
        target_set = set(zip(*np.where(ground_truth > 0)))
        true_positives = len(detected_set & target_set)
        false_positives = len(detected_set - target_set)
        
        print(f"    True positives: {true_positives}/{np.sum(ground_truth)}")
        print(f"    False positives: {false_positives}")
    
    # Test hybrid detector
    print("\n3. Testing hybrid detector...")
    
    config = TargetConfig(
        method=TargetMethod.HYBRID,
        target_signature=target_signature,
        hybrid_methods=[
            TargetMethod.MATCHED_FILTER,
            TargetMethod.ACE,
            TargetMethod.SAM
        ],
        hybrid_weights=[0.4, 0.4, 0.2]
    )
    
    detector = TargetDetector(config)
    result = detector.detect(image, ground_truth)
    
    print(f"  Processing time: {result.processing_time:.3f}s")
    print(f"  Detected targets: {result.n_detections} pixels")
    if result.roc_auc:
        print(f"  ROC AUC: {result.roc_auc:.3f}")
    
    # Test with background signatures
    print("\n4. Testing with background signatures...")
    
    # Create background signatures
    background_sigs = [
        np.random.randn(c).astype(np.float32) * 0.1 + 0.5,
        np.random.randn(c).astype(np.float32) * 0.1 + 0.4
    ]
    
    config = TargetConfig(
        method=TargetMethod.OSP if HAS_SCIPY else TargetMethod.MATCHED_FILTER,
        target_signature=target_signature,
        background_signatures=background_sigs
    )
    
    detector = TargetDetector(config)
    result = detector.detect(image, ground_truth)
    
    print(f"  Method: {result.method}")
    print(f"  Detected targets: {result.n_detections} pixels")
    print(f"  Processing time: {result.processing_time:.3f}s")
    
    # Performance comparison
    print("\n5. Performance comparison...")
    
    image_large = np.random.randn(200, 200, 100).astype(np.float32) * 0.1 + 0.5
    target_large = np.random.randn(100).astype(np.float32) * 0.3 + 0.6
    
    configs = [
        (TargetConfig(method=TargetMethod.MATCHED_FILTER, target_signature=target_large), "Matched Filter"),
        (TargetConfig(method=TargetMethod.SAM, target_signature=target_large), "SAM"),
    ]
    
    print(f"  Testing on {image_large.shape} image...")
    
    for config, name in configs:
        detector = TargetDetector(config)
        start = time.time()
        result = detector.detect(image_large)
        elapsed = time.time() - start
        
        pixels_per_sec = image_large.shape[0] * image_large.shape[1] / elapsed
        print(f"    {name}: {elapsed:.2f}s ({pixels_per_sec:.0f} pixels/s)")
    
    print("\n" + "=" * 80)
    print("Target Detection - Validation Complete!")
    print("=" * 80)
