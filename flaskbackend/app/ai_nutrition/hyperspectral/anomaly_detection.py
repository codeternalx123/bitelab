"""
Hyperspectral Anomaly Detection

Advanced algorithms for detecting anomalies, outliers, and unusual materials
in hyperspectral imagery. Essential for quality control, contaminant detection,
and identifying unknown substances.

Key Features:
- RX Detector (Reed-Xiaoli) - global background modeling
- Local RX - adaptive local background estimation
- Isolation Forest - machine learning anomaly detection
- One-Class SVM - boundary-based anomaly detection
- Kernel RX - nonlinear background modeling
- Cluster-Based RX - segmented background modeling
- Dual RX - multi-window adaptive detection
- Mahalanobis Distance - statistical outlier detection

Scientific Foundation:
- RX Detector: Reed & Yu, "Adaptive multiple-band CFAR detection of an optical 
  pattern with unknown spectral distribution", IEEE TAES, 1990
- Local RX: Molero et al., "Analysis and Optimizations of Global and Local Versions
  of the RX Algorithm", JSTSP, 2013
- Isolation Forest: Liu et al., "Isolation Forest", ICDM, 2008

Applications:
- Food contamination detection (foreign objects, spoilage)
- Quality control (defects, anomalies)
- Unknown material identification
- Outlier detection in spectral libraries

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
    from scipy.spatial.distance import mahalanobis
    from scipy.linalg import inv
    from scipy.ndimage import uniform_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logging.warning("SciPy not available. Some anomaly detection features will be limited.")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.covariance import EmpiricalCovariance, MinCovDet
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("scikit-learn not available. ML-based anomaly detection will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyMethod(Enum):
    """Anomaly detection methods"""
    RX = "rx"  # Global RX detector
    LOCAL_RX = "local_rx"  # Local adaptive RX
    KERNEL_RX = "kernel_rx"  # Kernel-based RX
    CLUSTER_RX = "cluster_rx"  # Cluster-based RX
    DUAL_RX = "dual_rx"  # Dual-window RX
    ISOLATION_FOREST = "isolation_forest"  # ML isolation forest
    ONE_CLASS_SVM = "one_class_svm"  # One-class SVM
    MAHALANOBIS = "mahalanobis"  # Mahalanobis distance


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection"""
    method: AnomalyMethod = AnomalyMethod.RX
    
    # RX parameters
    use_covariance_regularization: bool = True
    regularization_factor: float = 1e-6
    
    # Local RX parameters
    inner_window_size: int = 3  # Inner window (test region)
    outer_window_size: int = 21  # Outer window (background)
    
    # Kernel RX parameters
    kernel_type: str = "rbf"  # rbf, polynomial, linear
    kernel_param: float = 1.0  # gamma for RBF, degree for polynomial
    
    # Cluster RX parameters
    n_clusters: int = 5  # Number of background clusters
    
    # Dual RX parameters
    guard_window_size: int = 5  # Guard band between inner and outer
    
    # ML parameters
    contamination: float = 0.1  # Expected fraction of anomalies
    n_estimators: int = 100  # For isolation forest
    
    # Thresholding
    auto_threshold: bool = True
    threshold_percentile: float = 99.0  # Auto-threshold at this percentile
    manual_threshold: Optional[float] = None
    
    # Performance
    use_fast_approximation: bool = False  # Use faster but less accurate methods


@dataclass
class AnomalyResult:
    """Result from anomaly detection"""
    anomaly_map: np.ndarray  # Anomaly scores, shape (H, W)
    threshold: float
    binary_map: np.ndarray  # Binary detection map, shape (H, W)
    n_anomalies: int
    anomaly_pixels: List[Tuple[int, int]]  # List of (y, x) anomaly locations
    
    # Statistics
    min_score: float
    max_score: float
    mean_score: float
    std_score: float
    
    # Metadata
    method: str
    processing_time: float


class AnomalyDetector:
    """
    Comprehensive anomaly detection for hyperspectral images
    
    Implements multiple algorithms for detecting anomalies, outliers,
    and unusual spectral signatures.
    """
    
    def __init__(self, config: Optional[AnomalyConfig] = None):
        """
        Initialize anomaly detector
        
        Args:
            config: Detection configuration
        """
        self.config = config or AnomalyConfig()
        logger.info(f"Initialized anomaly detector: {self.config.method.value}")
    
    def detect(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> AnomalyResult:
        """
        Detect anomalies in hyperspectral image
        
        Args:
            image: Hyperspectral image, shape (H, W, C)
            mask: Optional mask to exclude regions, shape (H, W)
            
        Returns:
            Anomaly detection result
        """
        start_time = time.time()
        
        # Validate input
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image (H, W, C), got shape {image.shape}")
        
        h, w, c = image.shape
        
        # Apply mask if provided
        if mask is not None:
            if mask.shape != (h, w):
                raise ValueError(f"Mask shape {mask.shape} doesn't match image shape {(h, w)}")
        
        # Select detection method
        if self.config.method == AnomalyMethod.RX:
            anomaly_map = self._rx_detector(image, mask)
        elif self.config.method == AnomalyMethod.LOCAL_RX:
            anomaly_map = self._local_rx_detector(image, mask)
        elif self.config.method == AnomalyMethod.KERNEL_RX:
            anomaly_map = self._kernel_rx_detector(image, mask)
        elif self.config.method == AnomalyMethod.CLUSTER_RX:
            anomaly_map = self._cluster_rx_detector(image, mask)
        elif self.config.method == AnomalyMethod.DUAL_RX:
            anomaly_map = self._dual_rx_detector(image, mask)
        elif self.config.method == AnomalyMethod.ISOLATION_FOREST:
            anomaly_map = self._isolation_forest_detector(image, mask)
        elif self.config.method == AnomalyMethod.ONE_CLASS_SVM:
            anomaly_map = self._one_class_svm_detector(image, mask)
        elif self.config.method == AnomalyMethod.MAHALANOBIS:
            anomaly_map = self._mahalanobis_detector(image, mask)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
        
        # Compute threshold
        if self.config.auto_threshold:
            threshold = np.percentile(anomaly_map[~np.isnan(anomaly_map)], 
                                     self.config.threshold_percentile)
        else:
            threshold = self.config.manual_threshold or 0.0
        
        # Create binary detection map
        binary_map = anomaly_map > threshold
        
        # Find anomaly pixels
        anomaly_pixels = list(zip(*np.where(binary_map)))
        
        # Compute statistics
        valid_scores = anomaly_map[~np.isnan(anomaly_map)]
        
        result = AnomalyResult(
            anomaly_map=anomaly_map,
            threshold=float(threshold),
            binary_map=binary_map,
            n_anomalies=len(anomaly_pixels),
            anomaly_pixels=anomaly_pixels,
            min_score=float(np.min(valid_scores)),
            max_score=float(np.max(valid_scores)),
            mean_score=float(np.mean(valid_scores)),
            std_score=float(np.std(valid_scores)),
            method=self.config.method.value,
            processing_time=time.time() - start_time
        )
        
        logger.info(f"Detected {result.n_anomalies} anomalies in {result.processing_time:.2f}s")
        return result
    
    def _rx_detector(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Global RX (Reed-Xiaoli) detector
        
        Computes Mahalanobis distance from global background distribution:
        RX(x) = (x - μ)^T Σ^(-1) (x - μ)
        
        Args:
            image: Hyperspectral image, shape (H, W, C)
            mask: Optional mask
            
        Returns:
            Anomaly scores, shape (H, W)
        """
        h, w, c = image.shape
        
        # Reshape to (N, C)
        pixels = image.reshape(-1, c)
        
        # Apply mask
        if mask is not None:
            mask_flat = mask.reshape(-1)
            background_pixels = pixels[mask_flat == 0]
        else:
            background_pixels = pixels
        
        # Compute background statistics
        mean = np.mean(background_pixels, axis=0)
        
        # Compute covariance with regularization
        cov = np.cov(background_pixels, rowvar=False)
        
        if self.config.use_covariance_regularization:
            # Add regularization to diagonal for numerical stability
            cov += np.eye(c) * self.config.regularization_factor
        
        # Invert covariance
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix singular. Using pseudo-inverse.")
            cov_inv = np.linalg.pinv(cov)
        
        # Compute RX scores for all pixels
        anomaly_map = np.zeros((h, w))
        
        for i in range(h):
            for j in range(w):
                pixel = image[i, j]
                diff = pixel - mean
                # Mahalanobis distance: sqrt(diff^T * Sigma^-1 * diff)
                score = np.sqrt(np.dot(np.dot(diff, cov_inv), diff))
                anomaly_map[i, j] = score
        
        return anomaly_map
    
    def _local_rx_detector(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Local RX detector with adaptive background estimation
        
        For each pixel, estimates background from local neighborhood
        (excluding inner window around test pixel).
        
        Args:
            image: Hyperspectral image, shape (H, W, C)
            mask: Optional mask
            
        Returns:
            Anomaly scores, shape (H, W)
        """
        h, w, c = image.shape
        anomaly_map = np.zeros((h, w))
        
        inner_r = self.config.inner_window_size // 2
        outer_r = self.config.outer_window_size // 2
        
        # Pad image for border handling
        pad_width = outer_r
        padded_image = np.pad(image, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), 
                             mode='reflect')
        
        for i in range(h):
            for j in range(w):
                # Extract outer window (background)
                y_start = i
                y_end = i + 2 * outer_r + 1
                x_start = j
                x_end = j + 2 * outer_r + 1
                
                outer_window = padded_image[y_start:y_end, x_start:x_end]
                
                # Extract inner window (test region)
                iy_start = outer_r - inner_r
                iy_end = outer_r + inner_r + 1
                ix_start = outer_r - inner_r
                ix_end = outer_r + inner_r + 1
                
                # Create mask for background (outer - inner)
                bg_mask = np.ones((2 * outer_r + 1, 2 * outer_r + 1), dtype=bool)
                bg_mask[iy_start:iy_end, ix_start:ix_end] = False
                
                # Get background pixels
                bg_pixels = outer_window[bg_mask].reshape(-1, c)
                
                if len(bg_pixels) < c + 1:  # Need enough samples
                    anomaly_map[i, j] = 0
                    continue
                
                # Compute local background statistics
                mean = np.mean(bg_pixels, axis=0)
                cov = np.cov(bg_pixels, rowvar=False)
                
                # Regularization
                cov += np.eye(c) * self.config.regularization_factor
                
                # Invert
                try:
                    cov_inv = np.linalg.inv(cov)
                except:
                    cov_inv = np.linalg.pinv(cov)
                
                # Compute RX score for center pixel
                pixel = image[i, j]
                diff = pixel - mean
                score = np.sqrt(np.dot(np.dot(diff, cov_inv), diff))
                anomaly_map[i, j] = score
        
        return anomaly_map
    
    def _kernel_rx_detector(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Kernel RX detector using kernel methods for nonlinear backgrounds
        
        Args:
            image: Hyperspectral image, shape (H, W, C)
            mask: Optional mask
            
        Returns:
            Anomaly scores, shape (H, W)
        """
        # Simplified: use RBF kernel with global statistics
        h, w, c = image.shape
        pixels = image.reshape(-1, c)
        
        if mask is not None:
            mask_flat = mask.reshape(-1)
            background_pixels = pixels[mask_flat == 0]
        else:
            background_pixels = pixels
        
        # Use subset for efficiency
        if len(background_pixels) > 1000:
            indices = np.random.choice(len(background_pixels), 1000, replace=False)
            background_pixels = background_pixels[indices]
        
        # Compute RBF kernel
        gamma = self.config.kernel_param
        
        anomaly_map = np.zeros((h, w))
        
        for i in range(h):
            for j in range(w):
                pixel = image[i, j]
                
                # Compute kernel distance to background
                diff = background_pixels - pixel
                distances = np.sum(diff ** 2, axis=1)
                kernel_values = np.exp(-gamma * distances)
                
                # Anomaly score: inverse of mean kernel value
                score = 1.0 / (np.mean(kernel_values) + 1e-8)
                anomaly_map[i, j] = score
        
        return anomaly_map
    
    def _cluster_rx_detector(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Cluster-based RX detector
        
        Segments image into clusters, computes RX within each cluster,
        then combines results.
        
        Args:
            image: Hyperspectral image, shape (H, W, C)
            mask: Optional mask
            
        Returns:
            Anomaly scores, shape (H, W)
        """
        if not HAS_SKLEARN:
            logger.warning("scikit-learn not available. Falling back to global RX.")
            return self._rx_detector(image, mask)
        
        h, w, c = image.shape
        pixels = image.reshape(-1, c)
        
        # Cluster pixels
        kmeans = KMeans(n_clusters=self.config.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        labels = labels.reshape(h, w)
        
        anomaly_map = np.zeros((h, w))
        
        # Compute RX for each cluster
        for cluster_id in range(self.config.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_pixels = pixels[cluster_mask.reshape(-1)]
            
            if len(cluster_pixels) < c + 1:
                continue
            
            # Compute cluster statistics
            mean = np.mean(cluster_pixels, axis=0)
            cov = np.cov(cluster_pixels, rowvar=False)
            cov += np.eye(c) * self.config.regularization_factor
            
            try:
                cov_inv = np.linalg.inv(cov)
            except:
                cov_inv = np.linalg.pinv(cov)
            
            # Compute scores for pixels in this cluster
            for i in range(h):
                for j in range(w):
                    if labels[i, j] == cluster_id:
                        pixel = image[i, j]
                        diff = pixel - mean
                        score = np.sqrt(np.dot(np.dot(diff, cov_inv), diff))
                        anomaly_map[i, j] = score
        
        return anomaly_map
    
    def _dual_rx_detector(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Dual-window RX detector
        
        Uses two windows (inner and outer with guard band) for
        better background estimation.
        
        Args:
            image: Hyperspectral image, shape (H, W, C)
            mask: Optional mask
            
        Returns:
            Anomaly scores, shape (H, W)
        """
        # Similar to local RX but with guard band
        h, w, c = image.shape
        anomaly_map = np.zeros((h, w))
        
        inner_r = self.config.inner_window_size // 2
        guard_r = self.config.guard_window_size // 2
        outer_r = self.config.outer_window_size // 2
        
        pad_width = outer_r
        padded_image = np.pad(image, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), 
                             mode='reflect')
        
        for i in range(h):
            for j in range(w):
                # Extract windows
                y_start = i
                y_end = i + 2 * outer_r + 1
                x_start = j
                x_end = j + 2 * outer_r + 1
                
                outer_window = padded_image[y_start:y_end, x_start:x_end]
                
                # Create mask: outer - (inner + guard)
                bg_mask = np.ones((2 * outer_r + 1, 2 * outer_r + 1), dtype=bool)
                
                # Exclude inner + guard
                exclude_r = inner_r + guard_r
                ey_start = outer_r - exclude_r
                ey_end = outer_r + exclude_r + 1
                ex_start = outer_r - exclude_r
                ex_end = outer_r + exclude_r + 1
                
                bg_mask[ey_start:ey_end, ex_start:ex_end] = False
                
                bg_pixels = outer_window[bg_mask].reshape(-1, c)
                
                if len(bg_pixels) < c + 1:
                    anomaly_map[i, j] = 0
                    continue
                
                # Compute statistics
                mean = np.mean(bg_pixels, axis=0)
                cov = np.cov(bg_pixels, rowvar=False)
                cov += np.eye(c) * self.config.regularization_factor
                
                try:
                    cov_inv = np.linalg.inv(cov)
                except:
                    cov_inv = np.linalg.pinv(cov)
                
                # Compute score
                pixel = image[i, j]
                diff = pixel - mean
                score = np.sqrt(np.dot(np.dot(diff, cov_inv), diff))
                anomaly_map[i, j] = score
        
        return anomaly_map
    
    def _isolation_forest_detector(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Isolation Forest anomaly detector
        
        Uses ensemble of isolation trees to identify anomalies.
        
        Args:
            image: Hyperspectral image, shape (H, W, C)
            mask: Optional mask
            
        Returns:
            Anomaly scores, shape (H, W)
        """
        if not HAS_SKLEARN:
            logger.warning("scikit-learn not available. Falling back to global RX.")
            return self._rx_detector(image, mask)
        
        h, w, c = image.shape
        pixels = image.reshape(-1, c)
        
        # Train isolation forest
        iso_forest = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=42
        )
        
        # Fit and predict
        scores = iso_forest.fit_predict(pixels)
        anomaly_scores = iso_forest.score_samples(pixels)
        
        # Convert to anomaly map (higher = more anomalous)
        anomaly_map = -anomaly_scores.reshape(h, w)
        
        return anomaly_map
    
    def _one_class_svm_detector(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        One-Class SVM anomaly detector
        
        Learns a boundary around normal data.
        
        Args:
            image: Hyperspectral image, shape (H, W, C)
            mask: Optional mask
            
        Returns:
            Anomaly scores, shape (H, W)
        """
        if not HAS_SKLEARN:
            logger.warning("scikit-learn not available. Falling back to global RX.")
            return self._rx_detector(image, mask)
        
        h, w, c = image.shape
        pixels = image.reshape(-1, c)
        
        # Use subset for efficiency
        if len(pixels) > 5000:
            indices = np.random.choice(len(pixels), 5000, replace=False)
            train_pixels = pixels[indices]
        else:
            train_pixels = pixels
        
        # Train One-Class SVM
        svm = OneClassSVM(
            kernel=self.config.kernel_type,
            gamma=self.config.kernel_param if self.config.kernel_type == 'rbf' else 'auto',
            nu=self.config.contamination
        )
        
        svm.fit(train_pixels)
        
        # Compute decision function (higher = more normal)
        decision = svm.decision_function(pixels)
        
        # Convert to anomaly scores (higher = more anomalous)
        anomaly_map = -decision.reshape(h, w)
        
        return anomaly_map
    
    def _mahalanobis_detector(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Mahalanobis distance detector
        
        Similar to RX but explicitly uses Mahalanobis distance.
        
        Args:
            image: Hyperspectral image, shape (H, W, C)
            mask: Optional mask
            
        Returns:
            Anomaly scores, shape (H, W)
        """
        # Essentially the same as RX detector
        return self._rx_detector(image, mask)
    
    def visualize_results(
        self, 
        result: AnomalyResult, 
        original_image: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create visualizations of anomaly detection results
        
        Args:
            result: Anomaly detection result
            original_image: Optional original image for overlay
            
        Returns:
            Dictionary of visualizations
        """
        visualizations = {}
        
        # Normalized anomaly map
        anomaly_norm = (result.anomaly_map - result.min_score) / (result.max_score - result.min_score + 1e-8)
        visualizations['anomaly_map_normalized'] = anomaly_norm
        
        # Binary detection map
        visualizations['binary_map'] = result.binary_map.astype(np.float32)
        
        # Overlay on original image (if provided)
        if original_image is not None:
            # Use first 3 bands as RGB
            if original_image.ndim == 3 and original_image.shape[2] >= 3:
                rgb = original_image[:, :, [0, 1, 2]]
                rgb_norm = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb) + 1e-8)
                
                # Create overlay: red for anomalies
                overlay = rgb_norm.copy()
                overlay[result.binary_map, 0] = 1.0  # Red channel
                overlay[result.binary_map, 1] *= 0.3
                overlay[result.binary_map, 2] *= 0.3
                
                visualizations['overlay'] = overlay
        
        return visualizations


if __name__ == "__main__":
    # Example usage and validation
    print("=" * 80)
    print("Hyperspectral Anomaly Detection - Example Usage")
    print("=" * 80)
    
    # Create synthetic hyperspectral image
    print("\n1. Creating synthetic test data...")
    h, w, c = 100, 100, 50
    
    # Background: normal distribution
    image = np.random.randn(h, w, c).astype(np.float32) * 0.1 + 0.5
    
    # Add anomalies: small regions with different spectral signatures
    n_anomalies = 5
    anomaly_positions = []
    
    for i in range(n_anomalies):
        y = np.random.randint(10, h - 10)
        x = np.random.randint(10, w - 10)
        anomaly_positions.append((y, x))
        
        # Anomaly: different mean and variance
        anomaly_size = np.random.randint(2, 5)
        for dy in range(-anomaly_size, anomaly_size + 1):
            for dx in range(-anomaly_size, anomaly_size + 1):
                if 0 <= y + dy < h and 0 <= x + dx < w:
                    image[y + dy, x + dx] = np.random.randn(c) * 0.3 + 0.8
    
    print(f"  Image shape: {image.shape}")
    print(f"  Added {n_anomalies} anomalies at positions: {anomaly_positions}")
    
    # Test different methods
    methods = [
        AnomalyMethod.RX,
        AnomalyMethod.LOCAL_RX,
        AnomalyMethod.MAHALANOBIS
    ]
    
    if HAS_SKLEARN:
        methods.extend([
            AnomalyMethod.ISOLATION_FOREST,
            AnomalyMethod.CLUSTER_RX
        ])
    
    print(f"\n2. Testing {len(methods)} anomaly detection methods...")
    
    for method in methods:
        print(f"\n  Testing {method.value}...")
        
        config = AnomalyConfig(
            method=method,
            auto_threshold=True,
            threshold_percentile=95.0,
            inner_window_size=3,
            outer_window_size=11,
            n_clusters=3
        )
        
        detector = AnomalyDetector(config)
        result = detector.detect(image)
        
        print(f"    Processing time: {result.processing_time:.3f}s")
        print(f"    Anomaly scores: min={result.min_score:.3f}, max={result.max_score:.3f}, mean={result.mean_score:.3f}")
        print(f"    Threshold: {result.threshold:.3f}")
        print(f"    Detected anomalies: {result.n_anomalies} pixels")
        
        # Check detection accuracy
        detected_positions = set()
        for y, x in result.anomaly_pixels[:20]:  # Check first 20
            detected_positions.add((y, x))
        
        # Count true positives
        true_positives = 0
        for ay, ax in anomaly_positions:
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if (ay + dy, ax + dx) in detected_positions:
                        true_positives += 1
                        break
        
        print(f"    True positives: {true_positives}/{n_anomalies}")
    
    # Test visualization
    print("\n3. Testing visualization...")
    
    config = AnomalyConfig(method=AnomalyMethod.RX)
    detector = AnomalyDetector(config)
    result = detector.detect(image)
    
    visualizations = detector.visualize_results(result, image)
    
    print(f"  Generated visualizations:")
    for name, vis in visualizations.items():
        print(f"    {name}: shape={vis.shape}, dtype={vis.dtype}")
    
    # Test with mask
    print("\n4. Testing with mask...")
    
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:h//2, :] = 1  # Mask top half
    
    result_masked = detector.detect(image, mask)
    print(f"  Detected anomalies (with mask): {result_masked.n_anomalies}")
    print(f"  Processing time: {result_masked.processing_time:.3f}s")
    
    # Performance comparison
    print("\n5. Performance comparison...")
    
    image_large = np.random.randn(200, 200, 100).astype(np.float32) * 0.1 + 0.5
    
    configs = [
        (AnomalyConfig(method=AnomalyMethod.RX), "Global RX"),
        (AnomalyConfig(method=AnomalyMethod.LOCAL_RX, outer_window_size=11), "Local RX (11x11)"),
    ]
    
    print(f"  Testing on {image_large.shape} image...")
    
    for config, name in configs:
        detector = AnomalyDetector(config)
        start = time.time()
        result = detector.detect(image_large)
        elapsed = time.time() - start
        
        print(f"    {name}: {elapsed:.2f}s ({image_large.shape[0] * image_large.shape[1] / elapsed:.0f} pixels/s)")
    
    print("\n" + "=" * 80)
    print("Anomaly Detection - Validation Complete!")
    print("=" * 80)
