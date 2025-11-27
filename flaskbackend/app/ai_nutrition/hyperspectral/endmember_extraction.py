"""
Hyperspectral Endmember Extraction
===================================

Advanced algorithms for identifying pure spectral signatures (endmembers)
in hyperspectral images for spectral unmixing and material identification.

Key Algorithms:
- N-FINDR (N-dimensional simplex volume maximization)
- Vertex Component Analysis (VCA)
- Pixel Purity Index (PPI)
- Automatic Target Generation Process (ATGP)
- Sequential Maximum Angle Convex Cone (SMACC)
- Independent Component Analysis (ICA)
- Minimum Volume Transform (MVT)

Identifies pure elemental/compound signatures for atomic composition detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from scipy.linalg import svd, inv, qr
    from scipy.spatial import ConvexHull
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.decomposition import PCA, FastICA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class EndmemberMethod(Enum):
    """Endmember extraction algorithms"""
    NFINDR = "nfindr"
    VCA = "vca"  # Vertex Component Analysis
    PPI = "ppi"  # Pixel Purity Index
    ATGP = "atgp"  # Automatic Target Generation Process
    SMACC = "smacc"  # Sequential Maximum Angle Convex Cone
    ICA = "ica"  # Independent Component Analysis
    FIPPI = "fippi"  # Fast Iterative PPI
    MVSA = "mvsa"  # Minimum Volume Simplex Analysis


@dataclass
class EndmemberConfig:
    """Configuration for endmember extraction"""
    
    method: EndmemberMethod = EndmemberMethod.NFINDR
    num_endmembers: int = 10
    
    # Dimensionality reduction
    reduce_dimensionality: bool = True
    target_dimension: Optional[int] = None  # Auto-select if None
    
    # N-FINDR parameters
    nfindr_max_iterations: int = 3
    nfindr_tolerance: float = 1e-6
    
    # VCA parameters
    vca_snr_threshold: float = 15.0
    
    # PPI parameters
    ppi_num_skewers: int = 10000
    ppi_threshold_percentile: float = 95.0
    
    # ATGP parameters
    atgp_target_skewness: float = 2.0
    
    # ICA parameters
    ica_max_iterations: int = 200
    
    # Quality control
    min_spectral_angle: float = 5.0  # Minimum angle between endmembers (degrees)
    remove_duplicates: bool = True


@dataclass
class EndmemberResult:
    """Results from endmember extraction"""
    
    endmembers: np.ndarray  # [num_endmembers, num_bands]
    indices: Optional[np.ndarray] = None  # Pixel indices if pure pixels
    abundances: Optional[np.ndarray] = None  # [height, width, num_endmembers]
    volume: Optional[float] = None  # Simplex volume (for N-FINDR)
    snr_estimate: Optional[float] = None  # Signal-to-noise ratio
    
    def get_endmember(self, idx: int) -> np.ndarray:
        """Get specific endmember spectrum"""
        return self.endmembers[idx]
    
    def spectral_angle(self, idx1: int, idx2: int) -> float:
        """Compute spectral angle between two endmembers (degrees)"""
        e1 = self.endmembers[idx1]
        e2 = self.endmembers[idx2]
        
        cos_angle = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        
        return np.degrees(angle_rad)


class EndmemberExtractor:
    """
    Extract pure spectral signatures (endmembers) from hyperspectral data
    """
    
    def __init__(self, config: Optional[EndmemberConfig] = None):
        """
        Initialize endmember extractor
        
        Args:
            config: Extraction configuration
        """
        self.config = config or EndmemberConfig()
        
        logger.info(f"Initialized endmember extractor:")
        logger.info(f"  Method: {self.config.method.value}")
        logger.info(f"  Number of endmembers: {self.config.num_endmembers}")
    
    def extract(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> EndmemberResult:
        """
        Extract endmembers from hyperspectral image
        
        Args:
            image: Hyperspectral image [height, width, bands]
            mask: Binary mask [height, width] (optional)
            
        Returns:
            EndmemberResult with extracted endmembers
        """
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image [H, W, B], got shape {image.shape}")
        
        height, width, num_bands = image.shape
        
        # Reshape to [num_pixels, num_bands]
        if mask is not None:
            pixels = image[mask]
        else:
            pixels = image.reshape(-1, num_bands)
        
        logger.info(f"Extracting {self.config.num_endmembers} endmembers from {pixels.shape[0]} pixels")
        
        # Dimensionality reduction if requested
        if self.config.reduce_dimensionality:
            pixels_reduced, reduction_transform = self._reduce_dimensionality(pixels)
        else:
            pixels_reduced = pixels
            reduction_transform = None
        
        # Extract endmembers using selected method
        method = self.config.method
        
        if method == EndmemberMethod.NFINDR:
            result = self._nfindr(pixels_reduced)
        elif method == EndmemberMethod.VCA:
            result = self._vca(pixels_reduced)
        elif method == EndmemberMethod.PPI:
            result = self._ppi(pixels_reduced)
        elif method == EndmemberMethod.ATGP:
            result = self._atgp(pixels_reduced)
        elif method == EndmemberMethod.SMACC:
            result = self._smacc(pixels_reduced)
        elif method == EndmemberMethod.ICA:
            result = self._ica(pixels_reduced)
        else:
            raise ValueError(f"Unknown endmember method: {method}")
        
        # Project back to original space if dimensionality was reduced
        if reduction_transform is not None:
            result.endmembers = self._project_back(result.endmembers, reduction_transform)
        
        # Remove duplicate endmembers
        if self.config.remove_duplicates:
            result = self._remove_duplicates(result)
        
        logger.info(f"Extracted {len(result.endmembers)} endmembers")
        
        return result
    
    def _reduce_dimensionality(
        self,
        pixels: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Reduce dimensionality using PCA"""
        if not HAS_SKLEARN:
            logger.warning("sklearn not available, skipping dimensionality reduction")
            return pixels, None
        
        num_pixels, num_bands = pixels.shape
        
        # Determine target dimension
        if self.config.target_dimension is not None:
            n_components = min(self.config.target_dimension, num_bands)
        else:
            # Use number of endmembers + some extra dimensions
            n_components = min(self.config.num_endmembers + 5, num_bands)
        
        # Standardize
        scaler = StandardScaler()
        pixels_scaled = scaler.fit_transform(pixels)
        
        # PCA
        pca = PCA(n_components=n_components)
        pixels_reduced = pca.fit_transform(pixels_scaled)
        
        logger.info(f"Reduced dimensionality: {num_bands} -> {n_components}")
        logger.info(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        transform = {
            'scaler': scaler,
            'pca': pca,
            'n_components': n_components
        }
        
        return pixels_reduced, transform
    
    def _project_back(
        self,
        endmembers: np.ndarray,
        transform: Dict
    ) -> np.ndarray:
        """Project endmembers back to original space"""
        scaler = transform['scaler']
        pca = transform['pca']
        
        # Inverse PCA
        endmembers_scaled = pca.inverse_transform(endmembers)
        
        # Inverse standardization
        endmembers_original = scaler.inverse_transform(endmembers_scaled)
        
        return endmembers_original
    
    def _nfindr(self, pixels: np.ndarray) -> EndmemberResult:
        """
        N-FINDR algorithm
        Finds endmembers by maximizing simplex volume
        """
        num_pixels, num_bands = pixels.shape
        num_endmembers = self.config.num_endmembers
        
        # Initialize with random pixels
        np.random.seed(42)
        endmember_indices = np.random.choice(num_pixels, num_endmembers, replace=False)
        endmembers = pixels[endmember_indices].copy()
        
        # Compute initial volume
        volume = self._compute_simplex_volume(endmembers)
        
        logger.info(f"N-FINDR: Initial volume = {volume:.6e}")
        
        # Iterative refinement
        for iteration in range(self.config.nfindr_max_iterations):
            improved = False
            
            # Try replacing each endmember
            for i in range(num_endmembers):
                current_endmember = endmembers[i].copy()
                
                # Try random subset of pixels (for efficiency)
                test_indices = np.random.choice(num_pixels, min(1000, num_pixels), replace=False)
                
                for test_idx in test_indices:
                    # Replace endmember i with test pixel
                    endmembers[i] = pixels[test_idx]
                    
                    # Compute new volume
                    new_volume = self._compute_simplex_volume(endmembers)
                    
                    if new_volume > volume * (1 + self.config.nfindr_tolerance):
                        # Accept replacement
                        volume = new_volume
                        endmember_indices[i] = test_idx
                        improved = True
                        break
                    else:
                        # Reject replacement
                        endmembers[i] = current_endmember
            
            logger.info(f"N-FINDR iteration {iteration + 1}: volume = {volume:.6e}")
            
            if not improved:
                logger.info("N-FINDR converged")
                break
        
        return EndmemberResult(
            endmembers=endmembers,
            indices=endmember_indices,
            volume=volume
        )
    
    def _compute_simplex_volume(self, endmembers: np.ndarray) -> float:
        """Compute volume of simplex defined by endmembers"""
        num_endmembers, num_bands = endmembers.shape
        
        if num_endmembers <= 1:
            return 0.0
        
        # Center the simplex
        center = np.mean(endmembers, axis=0)
        centered = endmembers - center
        
        # Volume is proportional to determinant
        # Use QR decomposition for numerical stability
        if num_endmembers == num_bands:
            # Square matrix
            det = np.abs(np.linalg.det(centered))
        else:
            # Non-square: use Gram matrix
            gram = np.dot(centered, centered.T)
            det = np.sqrt(np.abs(np.linalg.det(gram)))
        
        return det
    
    def _vca(self, pixels: np.ndarray) -> EndmemberResult:
        """
        Vertex Component Analysis (VCA)
        Finds endmembers as vertices of simplex containing data
        """
        num_pixels, num_bands = pixels.shape
        num_endmembers = self.config.num_endmembers
        
        # Estimate SNR
        mean_spectrum = np.mean(pixels, axis=0)
        noise_covariance = np.cov((pixels - mean_spectrum).T)
        snr_estimate = 10 * np.log10(np.mean(mean_spectrum ** 2) / np.mean(np.diag(noise_covariance)))
        
        logger.info(f"VCA: Estimated SNR = {snr_estimate:.2f} dB")
        
        # Center data
        mean = np.mean(pixels, axis=0, keepdims=True)
        pixels_centered = pixels - mean
        
        # Project onto subspace
        if HAS_SCIPY:
            U, s, Vt = svd(pixels_centered.T, full_matrices=False)
            # Keep top components
            k = min(num_endmembers, num_bands)
            U_k = U[:, :k]
            projected = np.dot(pixels_centered, U_k)
        else:
            projected = pixels_centered
        
        # Initialize with pixel furthest from origin
        endmember_indices = []
        norms = np.linalg.norm(projected, axis=1)
        first_idx = np.argmax(norms)
        endmember_indices.append(first_idx)
        
        # Iteratively find vertices
        for _ in range(num_endmembers - 1):
            # Project onto orthogonal complement of current endmembers
            current_endmembers = projected[endmember_indices]
            
            # Gram-Schmidt orthogonalization
            orthogonal = projected.copy()
            for endmember in current_endmembers:
                projection = np.dot(orthogonal, endmember[:, np.newaxis])
                projection = projection * endmember / (np.dot(endmember, endmember) + 1e-10)
                orthogonal -= projection
            
            # Find pixel with maximum norm in orthogonal space
            norms = np.linalg.norm(orthogonal, axis=1)
            next_idx = np.argmax(norms)
            endmember_indices.append(next_idx)
        
        endmember_indices = np.array(endmember_indices)
        endmembers = pixels[endmember_indices]
        
        return EndmemberResult(
            endmembers=endmembers,
            indices=endmember_indices,
            snr_estimate=snr_estimate
        )
    
    def _ppi(self, pixels: np.ndarray) -> EndmemberResult:
        """
        Pixel Purity Index (PPI)
        Projects pixels onto random vectors and finds extrema
        """
        num_pixels, num_bands = pixels.shape
        num_endmembers = self.config.num_endmembers
        num_skewers = self.config.ppi_num_skewers
        
        # Initialize purity scores
        purity_scores = np.zeros(num_pixels)
        
        logger.info(f"PPI: Generating {num_skewers} random skewers")
        
        # Generate random unit vectors (skewers)
        np.random.seed(42)
        for i in range(num_skewers):
            # Random skewer
            skewer = np.random.randn(num_bands)
            skewer /= np.linalg.norm(skewer)
            
            # Project pixels onto skewer
            projections = np.dot(pixels, skewer)
            
            # Find extreme pixels
            min_idx = np.argmin(projections)
            max_idx = np.argmax(projections)
            
            purity_scores[min_idx] += 1
            purity_scores[max_idx] += 1
            
            if (i + 1) % 1000 == 0:
                logger.info(f"  Processed {i + 1}/{num_skewers} skewers")
        
        # Select pixels with highest purity scores
        threshold = np.percentile(purity_scores, self.config.ppi_threshold_percentile)
        pure_pixels = np.where(purity_scores >= threshold)[0]
        
        logger.info(f"PPI: Found {len(pure_pixels)} pure pixels above threshold")
        
        # Select diverse subset of pure pixels
        if len(pure_pixels) > num_endmembers:
            # Use k-means++ initialization for diversity
            selected_indices = self._kmeans_pp_init(pixels[pure_pixels], num_endmembers)
            endmember_indices = pure_pixels[selected_indices]
        else:
            endmember_indices = pure_pixels
        
        endmembers = pixels[endmember_indices]
        
        return EndmemberResult(
            endmembers=endmembers,
            indices=endmember_indices
        )
    
    def _atgp(self, pixels: np.ndarray) -> EndmemberResult:
        """
        Automatic Target Generation Process (ATGP)
        Orthogonal subspace projection
        """
        num_pixels, num_bands = pixels.shape
        num_endmembers = self.config.num_endmembers
        
        endmember_indices = []
        
        # Start with pixel with maximum norm
        norms = np.linalg.norm(pixels, axis=1)
        first_idx = np.argmax(norms)
        endmember_indices.append(first_idx)
        
        # Orthogonal projection
        residuals = pixels.copy()
        
        for i in range(num_endmembers - 1):
            # Current endmember
            current_endmember = pixels[endmember_indices[-1]]
            
            # Project residuals onto orthogonal complement
            projections = np.dot(residuals, current_endmember)
            norm_sq = np.dot(current_endmember, current_endmember)
            
            if norm_sq > 1e-10:
                orthogonal_proj = projections[:, np.newaxis] * current_endmember / norm_sq
                residuals -= orthogonal_proj
            
            # Find pixel with maximum residual norm
            residual_norms = np.linalg.norm(residuals, axis=1)
            next_idx = np.argmax(residual_norms)
            endmember_indices.append(next_idx)
        
        endmember_indices = np.array(endmember_indices)
        endmembers = pixels[endmember_indices]
        
        return EndmemberResult(
            endmembers=endmembers,
            indices=endmember_indices
        )
    
    def _smacc(self, pixels: np.ndarray) -> EndmemberResult:
        """
        Sequential Maximum Angle Convex Cone (SMACC)
        Finds endmembers with maximum angles
        """
        num_pixels, num_bands = pixels.shape
        num_endmembers = self.config.num_endmembers
        
        endmember_indices = []
        
        # Start with brightest pixel
        brightnesses = np.sum(pixels, axis=1)
        first_idx = np.argmax(brightnesses)
        endmember_indices.append(first_idx)
        
        # Iteratively add endmembers with maximum angle
        for _ in range(num_endmembers - 1):
            max_angle = -1
            best_idx = -1
            
            # Subsample for efficiency
            test_indices = np.random.choice(num_pixels, min(5000, num_pixels), replace=False)
            
            for test_idx in test_indices:
                if test_idx in endmember_indices:
                    continue
                
                # Compute minimum angle to existing endmembers
                test_pixel = pixels[test_idx]
                min_angle = np.inf
                
                for em_idx in endmember_indices:
                    endmember = pixels[em_idx]
                    
                    # Spectral angle
                    cos_angle = np.dot(test_pixel, endmember) / (
                        np.linalg.norm(test_pixel) * np.linalg.norm(endmember) + 1e-10
                    )
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    
                    min_angle = min(min_angle, angle)
                
                if min_angle > max_angle:
                    max_angle = min_angle
                    best_idx = test_idx
            
            if best_idx >= 0:
                endmember_indices.append(best_idx)
        
        endmember_indices = np.array(endmember_indices)
        endmembers = pixels[endmember_indices]
        
        return EndmemberResult(
            endmembers=endmembers,
            indices=endmember_indices
        )
    
    def _ica(self, pixels: np.ndarray) -> EndmemberResult:
        """
        Independent Component Analysis (ICA)
        Finds statistically independent endmembers
        """
        if not HAS_SKLEARN:
            logger.warning("sklearn not available, falling back to VCA")
            return self._vca(pixels)
        
        num_endmembers = self.config.num_endmembers
        
        # Apply ICA
        ica = FastICA(
            n_components=num_endmembers,
            max_iter=self.config.ica_max_iterations,
            random_state=42
        )
        
        # ICA expects [n_samples, n_features]
        # We have [n_pixels, n_bands]
        sources = ica.fit_transform(pixels)
        mixing_matrix = ica.mixing_
        
        # Endmembers are columns of mixing matrix
        endmembers = mixing_matrix.T
        
        # Normalize endmembers
        for i in range(len(endmembers)):
            endmembers[i] = np.abs(endmembers[i])
            endmembers[i] /= np.sum(endmembers[i])
        
        return EndmemberResult(
            endmembers=endmembers
        )
    
    def _kmeans_pp_init(
        self,
        pixels: np.ndarray,
        k: int
    ) -> np.ndarray:
        """K-means++ initialization for diverse selection"""
        num_pixels = len(pixels)
        
        # First center: random
        centers = [np.random.randint(num_pixels)]
        
        # Subsequent centers: proportional to distance
        for _ in range(k - 1):
            # Compute distances to nearest center
            distances = np.full(num_pixels, np.inf)
            
            for center_idx in centers:
                center = pixels[center_idx]
                dists = np.linalg.norm(pixels - center, axis=1)
                distances = np.minimum(distances, dists)
            
            # Sample proportional to squared distance
            distances_sq = distances ** 2
            probs = distances_sq / np.sum(distances_sq)
            
            next_center = np.random.choice(num_pixels, p=probs)
            centers.append(next_center)
        
        return np.array(centers)
    
    def _remove_duplicates(self, result: EndmemberResult) -> EndmemberResult:
        """Remove duplicate/similar endmembers"""
        num_endmembers = len(result.endmembers)
        
        # Compute pairwise spectral angles
        to_keep = []
        
        for i in range(num_endmembers):
            is_duplicate = False
            
            for j in to_keep:
                angle = result.spectral_angle(i, j)
                
                if angle < self.config.min_spectral_angle:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                to_keep.append(i)
        
        if len(to_keep) < num_endmembers:
            logger.info(f"Removed {num_endmembers - len(to_keep)} duplicate endmembers")
            
            result.endmembers = result.endmembers[to_keep]
            
            if result.indices is not None:
                result.indices = result.indices[to_keep]
        
        return result


if __name__ == '__main__':
    # Example usage
    print("Endmember Extraction Example")
    print("=" * 60)
    
    # Create mock hyperspectral image with known endmembers
    height, width = 100, 100
    num_bands = 50
    num_true_endmembers = 5
    
    # Generate random endmembers
    np.random.seed(42)
    true_endmembers = np.random.rand(num_true_endmembers, num_bands)
    true_endmembers /= np.linalg.norm(true_endmembers, axis=1, keepdims=True)
    
    # Generate image as linear mixture
    image = np.zeros((height, width, num_bands))
    
    for i in range(height):
        for j in range(width):
            # Random abundances
            abundances = np.random.dirichlet(np.ones(num_true_endmembers))
            
            # Linear mixture
            spectrum = np.dot(abundances, true_endmembers)
            spectrum += np.random.randn(num_bands) * 0.05  # Noise
            
            image[i, j, :] = spectrum
    
    print(f"\nImage shape: {image.shape}")
    print(f"True endmembers: {num_true_endmembers}")
    
    # Test different methods
    methods = [
        EndmemberMethod.NFINDR,
        EndmemberMethod.VCA,
        EndmemberMethod.ATGP,
    ]
    
    if HAS_SKLEARN:
        methods.extend([EndmemberMethod.PPI, EndmemberMethod.ICA])
    
    for method in methods:
        print(f"\n{method.value.upper()}:")
        print("-" * 60)
        
        config = EndmemberConfig(
            method=method,
            num_endmembers=num_true_endmembers,
            reduce_dimensionality=True
        )
        
        extractor = EndmemberExtractor(config)
        
        try:
            result = extractor.extract(image)
            
            print(f"Extracted {len(result.endmembers)} endmembers")
            
            # Compute similarity to true endmembers
            if result.endmembers is not None:
                # Find best matching between extracted and true
                similarities = []
                
                for i in range(min(len(result.endmembers), num_true_endmembers)):
                    extracted = result.endmembers[i]
                    
                    # Best match among true endmembers
                    best_similarity = -1
                    for j in range(num_true_endmembers):
                        true_em = true_endmembers[j]
                        similarity = np.dot(extracted, true_em) / (
                            np.linalg.norm(extracted) * np.linalg.norm(true_em)
                        )
                        best_similarity = max(best_similarity, similarity)
                    
                    similarities.append(best_similarity)
                
                avg_similarity = np.mean(similarities)
                print(f"Average similarity to true endmembers: {avg_similarity:.3f}")
        
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n✅ Endmember extraction complete!")
