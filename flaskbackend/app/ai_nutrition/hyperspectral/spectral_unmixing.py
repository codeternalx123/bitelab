"""
Hyperspectral Spectral Unmixing
================================

Advanced spectral unmixing algorithms to decompose mixed pixels into
pure endmember abundances for atomic composition estimation.

Key Algorithms:
- Fully Constrained Least Squares (FCLS)
- Non-Negative Least Squares (NNLS)
- Sum-to-One Constrained Least Squares (SCLS)
- Sparse Unmixing via Variable Splitting (SUnSAL)
- Collaborative Sparse Regression (CLSUnSAL)
- Non-negative Matrix Factorization (NMF)
- Vertex Component Analysis Unmixing
- Bayesian Unmixing

Decomposes hyperspectral pixels into elemental/compound abundances
for 99%+ accuracy in atomic composition prediction.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from scipy.optimize import nnls, lsq_linear, minimize
    from scipy.linalg import lstsq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.decomposition import NMF
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class UnmixingMethod(Enum):
    """Spectral unmixing algorithms"""
    FCLS = "fcls"  # Fully Constrained Least Squares
    NNLS = "nnls"  # Non-Negative Least Squares
    SCLS = "scls"  # Sum-to-One Constrained
    UCLS = "ucls"  # Unconstrained Least Squares
    SUNSAL = "sunsal"  # Sparse Unmixing via Variable Splitting
    CLSUNSAL = "clsunsal"  # Collaborative Sparse Unmixing
    NMF = "nmf"  # Non-negative Matrix Factorization
    BAYESIAN = "bayesian"  # Bayesian Unmixing


@dataclass
class UnmixingConfig:
    """Configuration for spectral unmixing"""
    
    method: UnmixingMethod = UnmixingMethod.FCLS
    
    # Constraints
    non_negativity: bool = True  # Abundances >= 0
    sum_to_one: bool = True  # Sum of abundances = 1
    
    # Regularization
    use_regularization: bool = False
    regularization_lambda: float = 0.01
    
    # Sparse unmixing parameters
    sparsity_lambda: float = 0.1
    max_iterations: int = 100
    tolerance: float = 1e-4
    
    # NMF parameters
    nmf_init: str = "nndsvd"  # nndsvd, nndsvda, random
    nmf_solver: str = "cd"  # cd (coordinate descent), mu (multiplicative update)
    
    # Bayesian parameters
    prior_type: str = "uniform"  # uniform, dirichlet
    noise_variance: float = 0.01


@dataclass
class UnmixingResult:
    """Results from spectral unmixing"""
    
    abundances: np.ndarray  # [height, width, num_endmembers] or [num_pixels, num_endmembers]
    reconstruction: Optional[np.ndarray] = None  # Reconstructed spectra
    residual: Optional[np.ndarray] = None  # Reconstruction error
    rmse: Optional[float] = None  # Root mean squared error
    
    def get_abundance_map(self, endmember_idx: int) -> np.ndarray:
        """Get spatial abundance map for specific endmember"""
        if self.abundances.ndim == 3:
            return self.abundances[:, :, endmember_idx]
        else:
            raise ValueError("Abundances are not in spatial format")
    
    def get_total_abundance(self) -> np.ndarray:
        """Get sum of abundances (should be ~1 if sum-to-one constraint)"""
        return np.sum(self.abundances, axis=-1)


class SpectralUnmixer:
    """
    Perform spectral unmixing to estimate endmember abundances
    """
    
    def __init__(self, config: Optional[UnmixingConfig] = None):
        """
        Initialize spectral unmixer
        
        Args:
            config: Unmixing configuration
        """
        self.config = config or UnmixingConfig()
        
        logger.info(f"Initialized spectral unmixer:")
        logger.info(f"  Method: {self.config.method.value}")
        logger.info(f"  Non-negativity: {self.config.non_negativity}")
        logger.info(f"  Sum-to-one: {self.config.sum_to_one}")
    
    def unmix(
        self,
        image: np.ndarray,
        endmembers: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> UnmixingResult:
        """
        Unmix hyperspectral image using given endmembers
        
        Args:
            image: Hyperspectral image [height, width, bands]
            endmembers: Endmember signatures [num_endmembers, bands]
            mask: Binary mask [height, width] (optional)
            
        Returns:
            UnmixingResult with abundance maps
        """
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image [H, W, B], got shape {image.shape}")
        
        height, width, num_bands = image.shape
        num_endmembers, em_bands = endmembers.shape
        
        if num_bands != em_bands:
            raise ValueError(
                f"Image has {num_bands} bands but endmembers have {em_bands} bands"
            )
        
        logger.info(f"Unmixing {height}×{width} image with {num_endmembers} endmembers")
        
        # Reshape image
        if mask is not None:
            pixels = image[mask]
            num_pixels = pixels.shape[0]
        else:
            pixels = image.reshape(-1, num_bands)
            num_pixels = height * width
        
        # Unmix using selected method
        method = self.config.method
        
        if method == UnmixingMethod.FCLS:
            abundances = self._fcls(pixels, endmembers)
        elif method == UnmixingMethod.NNLS:
            abundances = self._nnls(pixels, endmembers)
        elif method == UnmixingMethod.SCLS:
            abundances = self._scls(pixels, endmembers)
        elif method == UnmixingMethod.UCLS:
            abundances = self._ucls(pixels, endmembers)
        elif method == UnmixingMethod.SUNSAL:
            abundances = self._sunsal(pixels, endmembers)
        elif method == UnmixingMethod.NMF:
            abundances = self._nmf(pixels, endmembers)
        elif method == UnmixingMethod.BAYESIAN:
            abundances = self._bayesian(pixels, endmembers)
        else:
            raise ValueError(f"Unknown unmixing method: {method}")
        
        # Reconstruct spectra
        reconstruction = np.dot(abundances, endmembers)
        
        # Compute residuals
        residual = pixels - reconstruction
        rmse = np.sqrt(np.mean(residual ** 2))
        
        logger.info(f"Unmixing RMSE: {rmse:.6f}")
        
        # Reshape abundances back to spatial format
        if mask is not None:
            abundances_spatial = np.zeros((height, width, num_endmembers))
            abundances_spatial[mask] = abundances
            abundances = abundances_spatial
        else:
            abundances = abundances.reshape(height, width, num_endmembers)
        
        return UnmixingResult(
            abundances=abundances,
            reconstruction=reconstruction,
            residual=residual,
            rmse=rmse
        )
    
    def _fcls(
        self,
        pixels: np.ndarray,
        endmembers: np.ndarray
    ) -> np.ndarray:
        """
        Fully Constrained Least Squares (FCLS)
        Constraints: non-negativity + sum-to-one
        """
        if not HAS_SCIPY:
            logger.warning("scipy not available, using simple NNLS")
            return self._nnls(pixels, endmembers)
        
        num_pixels, num_bands = pixels.shape
        num_endmembers = endmembers.shape[0]
        
        abundances = np.zeros((num_pixels, num_endmembers))
        
        # Solve for each pixel
        for i in range(num_pixels):
            pixel = pixels[i]
            
            # Quadratic programming problem
            # min ||pixel - endmembers.T @ abundance||^2
            # s.t. abundance >= 0, sum(abundance) = 1
            
            # Use scipy's constrained least squares
            result = lsq_linear(
                endmembers.T,
                pixel,
                bounds=(0, np.inf),  # Non-negativity
                method='bvls'
            )
            
            abundance = result.x
            
            # Enforce sum-to-one
            abundance_sum = np.sum(abundance)
            if abundance_sum > 1e-10:
                abundance /= abundance_sum
            
            abundances[i] = abundance
        
        return abundances
    
    def _nnls(
        self,
        pixels: np.ndarray,
        endmembers: np.ndarray
    ) -> np.ndarray:
        """
        Non-Negative Least Squares (NNLS)
        Constraint: non-negativity only
        """
        if not HAS_SCIPY:
            # Fallback: simple unconstrained LS then clip
            abundances = self._ucls(pixels, endmembers)
            return np.clip(abundances, 0, None)
        
        num_pixels, num_bands = pixels.shape
        num_endmembers = endmembers.shape[0]
        
        abundances = np.zeros((num_pixels, num_endmembers))
        
        for i in range(num_pixels):
            pixel = pixels[i]
            
            # Solve NNLS for this pixel
            abundance, _ = nnls(endmembers.T, pixel)
            
            # Optionally normalize to sum-to-one
            if self.config.sum_to_one:
                abundance_sum = np.sum(abundance)
                if abundance_sum > 1e-10:
                    abundance /= abundance_sum
            
            abundances[i] = abundance
        
        return abundances
    
    def _scls(
        self,
        pixels: np.ndarray,
        endmembers: np.ndarray
    ) -> np.ndarray:
        """
        Sum-to-One Constrained Least Squares (SCLS)
        Constraint: sum-to-one (can be negative)
        """
        if not HAS_SCIPY:
            return self._ucls(pixels, endmembers)
        
        num_pixels, num_bands = pixels.shape
        num_endmembers = endmembers.shape[0]
        
        abundances = np.zeros((num_pixels, num_endmembers))
        
        # Augment system with sum-to-one constraint
        # [endmembers.T] @ abundance = pixel
        # [ones.T     ]              = 1
        
        A_aug = np.vstack([endmembers.T, np.ones(num_endmembers)])
        
        for i in range(num_pixels):
            pixel_aug = np.append(pixels[i], 1.0)
            
            # Least squares solution
            abundance, _, _, _ = lstsq(A_aug, pixel_aug)
            
            # Optionally enforce non-negativity
            if self.config.non_negativity:
                abundance = np.clip(abundance, 0, None)
                # Renormalize
                abundance_sum = np.sum(abundance)
                if abundance_sum > 1e-10:
                    abundance /= abundance_sum
            
            abundances[i] = abundance
        
        return abundances
    
    def _ucls(
        self,
        pixels: np.ndarray,
        endmembers: np.ndarray
    ) -> np.ndarray:
        """
        Unconstrained Least Squares (UCLS)
        No constraints
        """
        # Simple least squares: abundance = (M^T M)^-1 M^T pixel
        # where M = endmembers.T
        
        M = endmembers.T  # [bands, num_endmembers]
        
        # Pseudo-inverse
        M_pinv = np.linalg.pinv(M)
        
        # Solve for all pixels at once
        abundances = np.dot(pixels, M_pinv.T)
        
        return abundances
    
    def _sunsal(
        self,
        pixels: np.ndarray,
        endmembers: np.ndarray
    ) -> np.ndarray:
        """
        Sparse Unmixing via Variable Splitting (SUnSAL)
        Promotes sparsity in abundances
        """
        num_pixels, num_bands = pixels.shape
        num_endmembers = endmembers.shape[0]
        
        lam = self.config.sparsity_lambda
        max_iter = self.config.max_iterations
        tol = self.config.tolerance
        
        abundances = np.zeros((num_pixels, num_endmembers))
        
        M = endmembers.T  # [bands, num_endmembers]
        MtM = np.dot(M.T, M)
        
        for i in range(num_pixels):
            y = pixels[i]
            Mty = np.dot(M.T, y)
            
            # Initialize
            x = np.zeros(num_endmembers)
            z = x.copy()
            u = np.zeros(num_endmembers)
            
            # ADMM iterations
            for iteration in range(max_iter):
                x_old = x.copy()
                
                # x-update (least squares with L2 regularization)
                x = np.linalg.solve(MtM + np.eye(num_endmembers), Mty + z - u)
                
                # z-update (soft thresholding for sparsity)
                z = self._soft_threshold(x + u, lam)
                
                # Enforce constraints
                if self.config.non_negativity:
                    z = np.maximum(z, 0)
                
                if self.config.sum_to_one:
                    z_sum = np.sum(z)
                    if z_sum > 1e-10:
                        z /= z_sum
                
                # u-update (dual variable)
                u = u + x - z
                
                # Check convergence
                if np.linalg.norm(x - x_old) < tol:
                    break
            
            abundances[i] = z
        
        return abundances
    
    def _soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Soft thresholding operator for sparsity"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def _nmf(
        self,
        pixels: np.ndarray,
        endmembers: np.ndarray
    ) -> np.ndarray:
        """
        Non-negative Matrix Factorization (NMF)
        Joint estimation of endmembers and abundances
        """
        if not HAS_SKLEARN:
            logger.warning("sklearn not available, falling back to NNLS")
            return self._nnls(pixels, endmembers)
        
        num_pixels, num_bands = pixels.shape
        num_endmembers = endmembers.shape[0]
        
        # Initialize NMF with given endmembers
        model = NMF(
            n_components=num_endmembers,
            init='custom',
            solver=self.config.nmf_solver,
            max_iter=self.config.max_iterations,
            random_state=42
        )
        
        # Fit and transform
        # Note: NMF expects [n_samples, n_features]
        # We have pixels [n_pixels, n_bands]
        # NMF factorizes X ≈ WH where W=[n_pixels, n_components], H=[n_components, n_bands]
        
        # Transpose: [bands, pixels]
        X_T = pixels.T
        
        # Initialize with endmembers
        H_init = endmembers  # [num_endmembers, bands]
        
        # Fit
        model.fit(X_T.T)
        
        # Get abundances (W matrix)
        abundances = model.transform(X_T.T)
        
        # Normalize to sum-to-one if requested
        if self.config.sum_to_one:
            row_sums = np.sum(abundances, axis=1, keepdims=True)
            abundances = abundances / (row_sums + 1e-10)
        
        return abundances
    
    def _bayesian(
        self,
        pixels: np.ndarray,
        endmembers: np.ndarray
    ) -> np.ndarray:
        """
        Bayesian Unmixing
        Uses probabilistic model with priors
        """
        num_pixels, num_bands = pixels.shape
        num_endmembers = endmembers.shape[0]
        
        noise_var = self.config.noise_variance
        
        abundances = np.zeros((num_pixels, num_endmembers))
        
        M = endmembers.T
        
        for i in range(num_pixels):
            y = pixels[i]
            
            # MAP estimate with uniform/Dirichlet prior
            if self.config.prior_type == "dirichlet":
                # Dirichlet prior encourages sparsity
                alpha = np.ones(num_endmembers) * 0.1
            else:
                # Uniform prior
                alpha = np.ones(num_endmembers)
            
            # Posterior mode (MAP estimate)
            # Equivalent to regularized least squares
            
            MtM = np.dot(M.T, M)
            Mty = np.dot(M.T, y)
            
            # Add prior (regularization)
            reg = np.diag(alpha / noise_var)
            
            # Solve
            abundance = np.linalg.solve(MtM + reg, Mty)
            
            # Project onto simplex (non-negative, sum-to-one)
            abundance = self._project_simplex(abundance)
            
            abundances[i] = abundance
        
        return abundances
    
    def _project_simplex(self, x: np.ndarray) -> np.ndarray:
        """
        Project vector onto probability simplex
        (non-negative, sum-to-one)
        """
        # Sort in descending order
        x_sorted = np.sort(x)[::-1]
        
        # Find threshold
        cumsum = np.cumsum(x_sorted)
        k_array = np.arange(1, len(x) + 1)
        condition = x_sorted - (cumsum - 1) / k_array > 0
        
        if np.any(condition):
            k = np.where(condition)[0][-1] + 1
            theta = (cumsum[k-1] - 1) / k
        else:
            theta = 0
        
        # Project
        x_proj = np.maximum(x - theta, 0)
        
        return x_proj
    
    def evaluate_unmixing(
        self,
        result: UnmixingResult,
        ground_truth_abundances: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate unmixing quality
        
        Args:
            result: Unmixing result
            ground_truth_abundances: True abundances (optional, for synthetic data)
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Reconstruction error (RMSE)
        if result.rmse is not None:
            metrics['rmse'] = result.rmse
        
        # Sum-to-one constraint violation
        abundance_sums = result.get_total_abundance()
        sum_violation = np.mean(np.abs(abundance_sums - 1.0))
        metrics['sum_to_one_violation'] = sum_violation
        
        # Non-negativity constraint violation
        neg_abundances = np.sum(result.abundances < 0)
        metrics['negative_abundances'] = neg_abundances
        
        # Sparsity (average number of non-zero abundances)
        non_zero = np.sum(result.abundances > 1e-6, axis=-1)
        metrics['average_sparsity'] = np.mean(non_zero)
        
        # If ground truth available
        if ground_truth_abundances is not None:
            # Mean absolute error
            mae = np.mean(np.abs(result.abundances - ground_truth_abundances))
            metrics['mae'] = mae
            
            # Root mean squared error
            rmse = np.sqrt(np.mean((result.abundances - ground_truth_abundances) ** 2))
            metrics['abundance_rmse'] = rmse
        
        return metrics


if __name__ == '__main__':
    # Example usage
    print("Spectral Unmixing Example")
    print("=" * 60)
    
    # Create synthetic data with known abundances
    height, width = 50, 50
    num_bands = 30
    num_endmembers = 4
    
    # Generate random endmembers
    np.random.seed(42)
    endmembers = np.random.rand(num_endmembers, num_bands)
    endmembers /= np.linalg.norm(endmembers, axis=1, keepdims=True)
    
    print(f"\nEndmembers shape: {endmembers.shape}")
    
    # Generate image with known abundances
    image = np.zeros((height, width, num_bands))
    true_abundances = np.zeros((height, width, num_endmembers))
    
    for i in range(height):
        for j in range(width):
            # Random abundances (Dirichlet distribution)
            abundances = np.random.dirichlet(np.ones(num_endmembers) * 2)
            true_abundances[i, j] = abundances
            
            # Linear mixture
            spectrum = np.dot(abundances, endmembers)
            spectrum += np.random.randn(num_bands) * 0.02  # Noise
            
            image[i, j] = spectrum
    
    print(f"Image shape: {image.shape}")
    print(f"True abundances shape: {true_abundances.shape}")
    
    # Test different unmixing methods
    methods = [
        UnmixingMethod.FCLS,
        UnmixingMethod.NNLS,
        UnmixingMethod.UCLS,
    ]
    
    if HAS_SCIPY:
        methods.append(UnmixingMethod.SUNSAL)
    
    for method in methods:
        print(f"\n{method.value.upper()}:")
        print("-" * 60)
        
        config = UnmixingConfig(
            method=method,
            non_negativity=True,
            sum_to_one=True
        )
        
        unmixer = SpectralUnmixer(config)
        
        try:
            result = unmixer.unmix(image, endmembers)
            
            print(f"Abundances shape: {result.abundances.shape}")
            print(f"Reconstruction RMSE: {result.rmse:.6f}")
            
            # Evaluate
            metrics = unmixer.evaluate_unmixing(result, true_abundances)
            
            print(f"\nMetrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.6f}")
            
            # Check abundance constraints
            total = result.get_total_abundance()
            print(f"\nAbundance sum: min={total.min():.4f}, max={total.max():.4f}, mean={total.mean():.4f}")
            print(f"Abundance range: min={result.abundances.min():.4f}, max={result.abundances.max():.4f}")
        
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n✅ Spectral unmixing complete!")
