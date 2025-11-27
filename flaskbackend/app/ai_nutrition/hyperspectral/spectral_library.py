"""
Spectral Library Management and Matching System

This module provides comprehensive spectral library management for hyperspectral imaging,
including reference spectra storage, retrieval, and similarity-based matching. Essential
for material identification, validation, and classification tasks.

Key Features:
- Multiple similarity metrics (Euclidean, cosine, spectral angle, correlation, SID)
- Metadata-based filtering (element composition, wavelength range, source)
- Efficient k-NN search with optional pre-filtering
- Library persistence (JSON, HDF5, pickle formats)
- Batch matching and uncertainty quantification
- Spectral resampling for cross-library matching
- Quality scoring and confidence estimation

Scientific Foundation:
- Spectral Angle Mapper (SAM): Kruse et al., 1993
- Spectral Information Divergence (SID): Chang, 2000
- Cross-correlation matching: van der Meer, 2006
- Library spectroscopy: Clark et al., USGS Spectral Library

Author: AI Nutrition Team
Date: 2024
"""

import json
import logging
import pickle
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

# Optional dependencies
try:
    from scipy.interpolate import interp1d
    from scipy.spatial.distance import cdist, cosine, euclidean
    from scipy.stats import entropy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logging.warning("SciPy not available. Some spectral matching features will be limited.")

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logging.warning("h5py not available. HDF5 persistence will be unavailable.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """Spectral similarity metrics"""
    EUCLIDEAN = "euclidean"  # L2 distance
    COSINE = "cosine"  # Cosine similarity
    SAM = "sam"  # Spectral Angle Mapper
    CORRELATION = "correlation"  # Pearson correlation
    SID = "sid"  # Spectral Information Divergence
    MANHATTAN = "manhattan"  # L1 distance
    CHEBYSHEV = "chebyshev"  # L-infinity distance


@dataclass
class SpectrumMetadata:
    """Metadata for a reference spectrum"""
    name: str
    description: str = ""
    source: str = ""  # Lab measurement, synthetic, literature
    date_acquired: str = ""
    
    # Composition information
    elements: Dict[str, float] = field(default_factory=dict)  # Element symbol -> abundance
    compounds: List[str] = field(default_factory=list)  # Chemical compounds present
    
    # Measurement conditions
    instrument: str = ""
    wavelength_range: Tuple[float, float] = (400.0, 1000.0)  # nm
    spectral_resolution: float = 1.0  # nm
    temperature: Optional[float] = None  # Celsius
    
    # Quality metrics
    snr: Optional[float] = None  # Signal-to-noise ratio
    quality_score: float = 1.0  # 0-1, user-defined quality
    
    # Tags for filtering
    tags: List[str] = field(default_factory=list)
    category: str = "general"  # food, mineral, vegetation, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpectrumMetadata':
        """Create from dictionary"""
        # Handle wavelength_range tuple conversion
        if 'wavelength_range' in data and isinstance(data['wavelength_range'], list):
            data['wavelength_range'] = tuple(data['wavelength_range'])
        return cls(**data)


@dataclass
class ReferenceSpectrum:
    """A reference spectrum with metadata"""
    spectrum: np.ndarray  # Shape: (n_bands,)
    wavelengths: np.ndarray  # Shape: (n_bands,)
    metadata: SpectrumMetadata
    spectrum_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate spectrum"""
        if self.spectrum.shape != self.wavelengths.shape:
            raise ValueError(
                f"Spectrum and wavelengths shape mismatch: "
                f"{self.spectrum.shape} vs {self.wavelengths.shape}"
            )
        if self.spectrum_id is None:
            self.spectrum_id = f"{self.metadata.name}_{id(self)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'spectrum': self.spectrum.tolist(),
            'wavelengths': self.wavelengths.tolist(),
            'metadata': self.metadata.to_dict(),
            'spectrum_id': self.spectrum_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReferenceSpectrum':
        """Create from dictionary"""
        return cls(
            spectrum=np.array(data['spectrum']),
            wavelengths=np.array(data['wavelengths']),
            metadata=SpectrumMetadata.from_dict(data['metadata']),
            spectrum_id=data.get('spectrum_id')
        )


@dataclass
class MatchResult:
    """Result of spectral matching"""
    spectrum_id: str
    similarity: float  # Higher = more similar (normalized to 0-1)
    distance: float  # Raw distance value
    metadata: SpectrumMetadata
    confidence: float = 1.0  # Matching confidence (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'spectrum_id': self.spectrum_id,
            'similarity': float(self.similarity),
            'distance': float(self.distance),
            'metadata': self.metadata.to_dict(),
            'confidence': float(self.confidence)
        }


@dataclass
class LibraryStats:
    """Statistics about a spectral library"""
    n_spectra: int
    n_elements: int
    wavelength_range: Tuple[float, float]
    categories: List[str]
    sources: List[str]
    avg_snr: Optional[float] = None
    avg_quality: float = 1.0


class SpectralLibrary:
    """
    Spectral library for reference spectra management and matching
    
    Features:
    - Add/remove/update spectra
    - Metadata-based filtering
    - Multiple similarity metrics
    - k-NN search with confidence
    - Persistence (JSON, HDF5, pickle)
    - Spectral resampling for cross-library matching
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize spectral library
        
        Args:
            name: Library name
        """
        self.name = name
        self.spectra: Dict[str, ReferenceSpectrum] = {}
        self._spectrum_matrix: Optional[np.ndarray] = None
        self._dirty = True  # Flag to rebuild matrix
        
        logger.info(f"Initialized spectral library: {name}")
    
    def add_spectrum(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        metadata: SpectrumMetadata,
        spectrum_id: Optional[str] = None
    ) -> str:
        """
        Add a reference spectrum to the library
        
        Args:
            spectrum: Spectral values, shape (n_bands,)
            wavelengths: Wavelength values, shape (n_bands,)
            metadata: Spectrum metadata
            spectrum_id: Optional unique identifier
            
        Returns:
            Spectrum ID
        """
        ref_spectrum = ReferenceSpectrum(
            spectrum=spectrum,
            wavelengths=wavelengths,
            metadata=metadata,
            spectrum_id=spectrum_id
        )
        
        self.spectra[ref_spectrum.spectrum_id] = ref_spectrum
        self._dirty = True
        
        logger.debug(f"Added spectrum: {ref_spectrum.spectrum_id}")
        return ref_spectrum.spectrum_id
    
    def remove_spectrum(self, spectrum_id: str) -> bool:
        """
        Remove a spectrum from the library
        
        Args:
            spectrum_id: Spectrum identifier
            
        Returns:
            True if removed, False if not found
        """
        if spectrum_id in self.spectra:
            del self.spectra[spectrum_id]
            self._dirty = True
            logger.debug(f"Removed spectrum: {spectrum_id}")
            return True
        return False
    
    def get_spectrum(self, spectrum_id: str) -> Optional[ReferenceSpectrum]:
        """Get a spectrum by ID"""
        return self.spectra.get(spectrum_id)
    
    def filter_spectra(
        self,
        elements: Optional[List[str]] = None,
        category: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_quality: float = 0.0,
        wavelength_range: Optional[Tuple[float, float]] = None
    ) -> List[ReferenceSpectrum]:
        """
        Filter spectra by metadata criteria
        
        Args:
            elements: Filter by elements present (any match)
            category: Filter by category
            source: Filter by source
            tags: Filter by tags (any match)
            min_quality: Minimum quality score
            wavelength_range: Required wavelength coverage
            
        Returns:
            List of matching spectra
        """
        results = []
        
        for ref_spec in self.spectra.values():
            meta = ref_spec.metadata
            
            # Check elements
            if elements is not None:
                if not any(elem in meta.elements for elem in elements):
                    continue
            
            # Check category
            if category is not None and meta.category != category:
                continue
            
            # Check source
            if source is not None and meta.source != source:
                continue
            
            # Check tags
            if tags is not None:
                if not any(tag in meta.tags for tag in tags):
                    continue
            
            # Check quality
            if meta.quality_score < min_quality:
                continue
            
            # Check wavelength range
            if wavelength_range is not None:
                wl_min, wl_max = wavelength_range
                spec_min, spec_max = meta.wavelength_range
                if spec_min > wl_min or spec_max < wl_max:
                    continue
            
            results.append(ref_spec)
        
        logger.debug(f"Filtered {len(results)} spectra from {len(self.spectra)}")
        return results
    
    def _build_spectrum_matrix(self, spectra: Optional[List[ReferenceSpectrum]] = None):
        """Build matrix for efficient batch matching"""
        if spectra is None:
            spectra = list(self.spectra.values())
        
        if len(spectra) == 0:
            self._spectrum_matrix = np.array([])
            return
        
        # Check if all spectra have same wavelengths
        ref_wl = spectra[0].wavelengths
        same_wavelengths = all(
            np.allclose(spec.wavelengths, ref_wl) for spec in spectra
        )
        
        if not same_wavelengths:
            logger.warning(
                "Spectra have different wavelengths. Resampling to common grid."
            )
            # Resample to common wavelength grid (use first spectrum as reference)
            spectra = [self._resample_spectrum(spec, ref_wl) for spec in spectra]
        
        # Stack spectra into matrix
        self._spectrum_matrix = np.vstack([spec.spectrum for spec in spectra])
        self._dirty = False
    
    def _resample_spectrum(
        self,
        spectrum: ReferenceSpectrum,
        target_wavelengths: np.ndarray
    ) -> ReferenceSpectrum:
        """
        Resample spectrum to new wavelength grid
        
        Args:
            spectrum: Original spectrum
            target_wavelengths: Target wavelength values
            
        Returns:
            Resampled spectrum
        """
        if not HAS_SCIPY:
            raise ImportError("SciPy required for spectral resampling")
        
        # Create interpolation function
        interp_func = interp1d(
            spectrum.wavelengths,
            spectrum.spectrum,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        # Resample
        resampled_spectrum = interp_func(target_wavelengths)
        
        # Create new reference spectrum
        return ReferenceSpectrum(
            spectrum=resampled_spectrum,
            wavelengths=target_wavelengths,
            metadata=spectrum.metadata,
            spectrum_id=spectrum.spectrum_id
        )
    
    def match_spectrum(
        self,
        query_spectrum: np.ndarray,
        query_wavelengths: np.ndarray,
        metric: SimilarityMetric = SimilarityMetric.SAM,
        k: int = 5,
        filter_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[MatchResult]:
        """
        Find k most similar spectra in library
        
        Args:
            query_spectrum: Query spectral signature, shape (n_bands,)
            query_wavelengths: Query wavelengths, shape (n_bands,)
            metric: Similarity metric to use
            k: Number of nearest neighbors to return
            filter_kwargs: Optional metadata filters (passed to filter_spectra)
            
        Returns:
            List of k best matches, sorted by similarity (highest first)
        """
        # Filter spectra if requested
        if filter_kwargs is not None:
            candidate_spectra = self.filter_spectra(**filter_kwargs)
        else:
            candidate_spectra = list(self.spectra.values())
        
        if len(candidate_spectra) == 0:
            logger.warning("No candidate spectra found")
            return []
        
        # Ensure all spectra have same wavelengths as query
        ref_wl = candidate_spectra[0].wavelengths
        if not np.allclose(query_wavelengths, ref_wl):
            if HAS_SCIPY:
                logger.debug("Resampling query spectrum to library wavelengths")
                query_spectrum = self._resample_query(
                    query_spectrum, query_wavelengths, ref_wl
                )
            else:
                logger.warning(
                    "Query wavelengths differ from library. "
                    "Install SciPy for automatic resampling."
                )
        
        # Compute similarities
        results = []
        for ref_spec in candidate_spectra:
            # Resample reference if needed
            if not np.allclose(ref_spec.wavelengths, ref_wl):
                if HAS_SCIPY:
                    ref_spec = self._resample_spectrum(ref_spec, ref_wl)
                else:
                    continue  # Skip if can't resample
            
            # Compute similarity
            distance, similarity = self._compute_similarity(
                query_spectrum, ref_spec.spectrum, metric
            )
            
            # Compute confidence (based on distance distribution)
            confidence = self._estimate_confidence(distance, metric)
            
            results.append(MatchResult(
                spectrum_id=ref_spec.spectrum_id,
                similarity=similarity,
                distance=distance,
                metadata=ref_spec.metadata,
                confidence=confidence
            ))
        
        # Sort by similarity (highest first) and return top k
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:k]
    
    def _resample_query(
        self,
        query_spectrum: np.ndarray,
        query_wavelengths: np.ndarray,
        target_wavelengths: np.ndarray
    ) -> np.ndarray:
        """Resample query spectrum to target wavelengths"""
        if not HAS_SCIPY:
            raise ImportError("SciPy required for resampling")
        
        interp_func = interp1d(
            query_wavelengths,
            query_spectrum,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        return interp_func(target_wavelengths)
    
    def _compute_similarity(
        self,
        spectrum1: np.ndarray,
        spectrum2: np.ndarray,
        metric: SimilarityMetric
    ) -> Tuple[float, float]:
        """
        Compute similarity between two spectra
        
        Args:
            spectrum1: First spectrum
            spectrum2: Second spectrum
            metric: Similarity metric
            
        Returns:
            (distance, similarity) where similarity is normalized to 0-1
        """
        if metric == SimilarityMetric.EUCLIDEAN:
            if HAS_SCIPY:
                distance = euclidean(spectrum1, spectrum2)
            else:
                distance = np.linalg.norm(spectrum1 - spectrum2)
            # Convert to similarity (0-1)
            max_distance = np.linalg.norm(spectrum1) + np.linalg.norm(spectrum2)
            similarity = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
        
        elif metric == SimilarityMetric.COSINE:
            if HAS_SCIPY:
                distance = cosine(spectrum1, spectrum2)
            else:
                # Cosine distance = 1 - cosine similarity
                dot = np.dot(spectrum1, spectrum2)
                norm1 = np.linalg.norm(spectrum1)
                norm2 = np.linalg.norm(spectrum2)
                distance = 1.0 - (dot / (norm1 * norm2)) if norm1 * norm2 > 0 else 1.0
            similarity = 1.0 - distance  # Cosine similarity
        
        elif metric == SimilarityMetric.SAM:
            # Spectral Angle Mapper
            dot = np.dot(spectrum1, spectrum2)
            norm1 = np.linalg.norm(spectrum1)
            norm2 = np.linalg.norm(spectrum2)
            
            if norm1 * norm2 > 0:
                cos_angle = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
                angle = np.arccos(cos_angle)  # Radians
                distance = angle
                # Normalize to 0-1 (angle ranges from 0 to π)
                similarity = 1.0 - (angle / np.pi)
            else:
                distance = np.pi
                similarity = 0.0
        
        elif metric == SimilarityMetric.CORRELATION:
            # Pearson correlation
            mean1 = np.mean(spectrum1)
            mean2 = np.mean(spectrum2)
            centered1 = spectrum1 - mean1
            centered2 = spectrum2 - mean2
            
            dot = np.dot(centered1, centered2)
            norm1 = np.linalg.norm(centered1)
            norm2 = np.linalg.norm(centered2)
            
            if norm1 * norm2 > 0:
                corr = dot / (norm1 * norm2)
                distance = 1.0 - corr
                similarity = (corr + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
            else:
                distance = 2.0
                similarity = 0.0
        
        elif metric == SimilarityMetric.SID:
            # Spectral Information Divergence (symmetric KL divergence)
            # Normalize to probability distributions
            eps = 1e-10
            p = spectrum1 / (np.sum(spectrum1) + eps)
            q = spectrum2 / (np.sum(spectrum2) + eps)
            
            if HAS_SCIPY:
                kl_pq = entropy(p, q)
                kl_qp = entropy(q, p)
            else:
                # Manual KL divergence
                kl_pq = np.sum(p * np.log((p + eps) / (q + eps)))
                kl_qp = np.sum(q * np.log((q + eps) / (p + eps)))
            
            distance = kl_pq + kl_qp
            # Normalize to 0-1 (use exponential decay)
            similarity = np.exp(-distance)
        
        elif metric == SimilarityMetric.MANHATTAN:
            distance = np.sum(np.abs(spectrum1 - spectrum2))
            max_distance = np.sum(np.abs(spectrum1)) + np.sum(np.abs(spectrum2))
            similarity = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
        
        elif metric == SimilarityMetric.CHEBYSHEV:
            distance = np.max(np.abs(spectrum1 - spectrum2))
            max_val = max(np.max(np.abs(spectrum1)), np.max(np.abs(spectrum2)))
            similarity = 1.0 - (distance / max_val) if max_val > 0 else 1.0
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return distance, similarity
    
    def _estimate_confidence(
        self,
        distance: float,
        metric: SimilarityMetric,
        threshold_factor: float = 2.0
    ) -> float:
        """
        Estimate matching confidence based on distance
        
        Args:
            distance: Distance value
            metric: Similarity metric used
            threshold_factor: Multiplier for confidence threshold
            
        Returns:
            Confidence score (0-1)
        """
        # Define typical "good" distances for each metric
        thresholds = {
            SimilarityMetric.EUCLIDEAN: 0.1,
            SimilarityMetric.COSINE: 0.1,
            SimilarityMetric.SAM: 0.1,  # ~5.7 degrees
            SimilarityMetric.CORRELATION: 0.1,
            SimilarityMetric.SID: 0.5,
            SimilarityMetric.MANHATTAN: 0.1,
            SimilarityMetric.CHEBYSHEV: 0.1
        }
        
        threshold = thresholds.get(metric, 0.1) * threshold_factor
        
        # Exponential decay confidence
        confidence = np.exp(-distance / threshold)
        return float(confidence)
    
    def batch_match(
        self,
        query_spectra: np.ndarray,
        query_wavelengths: np.ndarray,
        metric: SimilarityMetric = SimilarityMetric.SAM,
        k: int = 5,
        filter_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[List[MatchResult]]:
        """
        Match multiple query spectra in batch
        
        Args:
            query_spectra: Query spectra, shape (n_queries, n_bands)
            query_wavelengths: Wavelengths, shape (n_bands,)
            metric: Similarity metric
            k: Number of matches per query
            filter_kwargs: Optional metadata filters
            
        Returns:
            List of match results for each query
        """
        results = []
        for i in range(query_spectra.shape[0]):
            matches = self.match_spectrum(
                query_spectra[i],
                query_wavelengths,
                metric=metric,
                k=k,
                filter_kwargs=filter_kwargs
            )
            results.append(matches)
        
        logger.info(f"Batch matched {len(results)} query spectra")
        return results
    
    def get_stats(self) -> LibraryStats:
        """Get library statistics"""
        if len(self.spectra) == 0:
            return LibraryStats(
                n_spectra=0,
                n_elements=0,
                wavelength_range=(0.0, 0.0),
                categories=[],
                sources=[]
            )
        
        # Collect statistics
        all_elements = set()
        categories = set()
        sources = set()
        snrs = []
        qualities = []
        wl_mins = []
        wl_maxs = []
        
        for ref_spec in self.spectra.values():
            meta = ref_spec.metadata
            all_elements.update(meta.elements.keys())
            categories.add(meta.category)
            sources.add(meta.source)
            qualities.append(meta.quality_score)
            
            if meta.snr is not None:
                snrs.append(meta.snr)
            
            wl_min, wl_max = meta.wavelength_range
            wl_mins.append(wl_min)
            wl_maxs.append(wl_max)
        
        return LibraryStats(
            n_spectra=len(self.spectra),
            n_elements=len(all_elements),
            wavelength_range=(min(wl_mins), max(wl_maxs)),
            categories=sorted(categories),
            sources=sorted(sources),
            avg_snr=np.mean(snrs) if snrs else None,
            avg_quality=np.mean(qualities)
        )
    
    def save(self, path: Union[str, Path], format: str = "json"):
        """
        Save library to disk
        
        Args:
            path: Save path
            format: File format ('json', 'hdf5', 'pickle')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            self._save_json(path)
        elif format == "hdf5":
            self._save_hdf5(path)
        elif format == "pickle":
            self._save_pickle(path)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved library to {path} ({format} format)")
    
    def _save_json(self, path: Path):
        """Save as JSON"""
        data = {
            'name': self.name,
            'spectra': [spec.to_dict() for spec in self.spectra.values()]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_hdf5(self, path: Path):
        """Save as HDF5"""
        if not HAS_H5PY:
            raise ImportError("h5py required for HDF5 format")
        
        with h5py.File(path, 'w') as f:
            f.attrs['name'] = self.name
            f.attrs['n_spectra'] = len(self.spectra)
            
            for i, (spec_id, ref_spec) in enumerate(self.spectra.items()):
                grp = f.create_group(f'spectrum_{i}')
                grp.attrs['spectrum_id'] = spec_id
                grp.create_dataset('spectrum', data=ref_spec.spectrum)
                grp.create_dataset('wavelengths', data=ref_spec.wavelengths)
                
                # Save metadata as attributes
                meta_grp = grp.create_group('metadata')
                for key, value in ref_spec.metadata.to_dict().items():
                    if value is not None:
                        if isinstance(value, (list, dict)):
                            meta_grp.attrs[key] = json.dumps(value)
                        else:
                            meta_grp.attrs[key] = value
    
    def _save_pickle(self, path: Path):
        """Save as pickle"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Union[str, Path], format: str = "json") -> 'SpectralLibrary':
        """
        Load library from disk
        
        Args:
            path: Load path
            format: File format ('json', 'hdf5', 'pickle')
            
        Returns:
            Loaded library
        """
        path = Path(path)
        
        if format == "json":
            return cls._load_json(path)
        elif format == "hdf5":
            return cls._load_hdf5(path)
        elif format == "pickle":
            return cls._load_pickle(path)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @classmethod
    def _load_json(cls, path: Path) -> 'SpectralLibrary':
        """Load from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        library = cls(name=data['name'])
        
        for spec_data in data['spectra']:
            ref_spec = ReferenceSpectrum.from_dict(spec_data)
            library.spectra[ref_spec.spectrum_id] = ref_spec
        
        logger.info(f"Loaded {len(library.spectra)} spectra from {path}")
        return library
    
    @classmethod
    def _load_hdf5(cls, path: Path) -> 'SpectralLibrary':
        """Load from HDF5"""
        if not HAS_H5PY:
            raise ImportError("h5py required for HDF5 format")
        
        library = None
        
        with h5py.File(path, 'r') as f:
            name = f.attrs['name']
            library = cls(name=name)
            
            n_spectra = f.attrs['n_spectra']
            for i in range(n_spectra):
                grp = f[f'spectrum_{i}']
                spec_id = grp.attrs['spectrum_id']
                spectrum = grp['spectrum'][:]
                wavelengths = grp['wavelengths'][:]
                
                # Load metadata
                meta_grp = grp['metadata']
                meta_dict = {}
                for key in meta_grp.attrs:
                    value = meta_grp.attrs[key]
                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                        meta_dict[key] = json.loads(value)
                    else:
                        meta_dict[key] = value
                
                metadata = SpectrumMetadata.from_dict(meta_dict)
                
                ref_spec = ReferenceSpectrum(
                    spectrum=spectrum,
                    wavelengths=wavelengths,
                    metadata=metadata,
                    spectrum_id=spec_id
                )
                library.spectra[spec_id] = ref_spec
        
        logger.info(f"Loaded {len(library.spectra)} spectra from {path}")
        return library
    
    @classmethod
    def _load_pickle(cls, path: Path) -> 'SpectralLibrary':
        """Load from pickle"""
        with open(path, 'rb') as f:
            library = pickle.load(f)
        
        logger.info(f"Loaded {len(library.spectra)} spectra from {path}")
        return library
    
    def merge(self, other: 'SpectralLibrary', overwrite: bool = False):
        """
        Merge another library into this one
        
        Args:
            other: Library to merge
            overwrite: If True, overwrite existing spectrum IDs
        """
        n_added = 0
        n_skipped = 0
        
        for spec_id, ref_spec in other.spectra.items():
            if spec_id in self.spectra and not overwrite:
                n_skipped += 1
                continue
            
            self.spectra[spec_id] = ref_spec
            n_added += 1
        
        if n_added > 0:
            self._dirty = True
        
        logger.info(
            f"Merged {n_added} spectra from {other.name}. "
            f"Skipped {n_skipped} duplicates."
        )
    
    def __len__(self) -> int:
        """Number of spectra in library"""
        return len(self.spectra)
    
    def __repr__(self) -> str:
        """String representation"""
        stats = self.get_stats()
        return (
            f"SpectralLibrary(name='{self.name}', "
            f"n_spectra={stats.n_spectra}, "
            f"n_elements={stats.n_elements})"
        )


def create_synthetic_library(
    n_spectra: int = 100,
    n_bands: int = 50,
    n_elements: int = 10,
    wavelength_range: Tuple[float, float] = (400.0, 1000.0)
) -> SpectralLibrary:
    """
    Create a synthetic spectral library for testing
    
    Args:
        n_spectra: Number of spectra to generate
        n_bands: Number of spectral bands
        n_elements: Number of unique elements
        wavelength_range: Wavelength range in nm
        
    Returns:
        Synthetic spectral library
    """
    library = SpectralLibrary(name="synthetic")
    
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_bands)
    element_names = [f"Element_{i}" for i in range(n_elements)]
    
    for i in range(n_spectra):
        # Generate random spectrum (Gaussian mixture)
        spectrum = np.zeros(n_bands)
        n_peaks = np.random.randint(1, 4)
        
        for _ in range(n_peaks):
            center = np.random.randint(0, n_bands)
            width = np.random.uniform(3.0, 10.0)
            amplitude = np.random.uniform(0.3, 1.0)
            
            gaussian = amplitude * np.exp(-0.5 * ((np.arange(n_bands) - center) / width) ** 2)
            spectrum += gaussian
        
        # Add noise
        spectrum += np.random.normal(0, 0.05, n_bands)
        spectrum = np.maximum(spectrum, 0)  # Ensure non-negative
        
        # Create metadata
        n_elements_present = np.random.randint(1, min(4, n_elements + 1))
        elements_present = np.random.choice(element_names, n_elements_present, replace=False)
        element_dict = {
            elem: np.random.uniform(0.1, 1.0)
            for elem in elements_present
        }
        
        metadata = SpectrumMetadata(
            name=f"Spectrum_{i}",
            description=f"Synthetic spectrum {i}",
            source="synthetic",
            elements=element_dict,
            wavelength_range=wavelength_range,
            snr=np.random.uniform(10.0, 50.0),
            quality_score=np.random.uniform(0.7, 1.0),
            category=np.random.choice(["food", "mineral", "vegetation"]),
            tags=[f"tag_{j}" for j in range(np.random.randint(1, 4))]
        )
        
        library.add_spectrum(spectrum, wavelengths, metadata)
    
    logger.info(f"Created synthetic library with {len(library)} spectra")
    return library


if __name__ == "__main__":
    # Example usage and validation
    print("=" * 80)
    print("Spectral Library System - Example Usage")
    print("=" * 80)
    
    # Create synthetic library
    print("\n1. Creating synthetic library...")
    library = create_synthetic_library(n_spectra=50, n_bands=100, n_elements=8)
    stats = library.get_stats()
    print(f"Library: {library}")
    print(f"Statistics:")
    print(f"  - Spectra: {stats.n_spectra}")
    print(f"  - Elements: {stats.n_elements}")
    print(f"  - Wavelength range: {stats.wavelength_range[0]:.1f}-{stats.wavelength_range[1]:.1f} nm")
    print(f"  - Categories: {', '.join(stats.categories)}")
    print(f"  - Avg SNR: {stats.avg_snr:.2f}")
    print(f"  - Avg quality: {stats.avg_quality:.3f}")
    
    # Test filtering
    print("\n2. Testing metadata filtering...")
    food_spectra = library.filter_spectra(category="food", min_quality=0.8)
    print(f"Found {len(food_spectra)} high-quality food spectra")
    
    element_spectra = library.filter_spectra(elements=["Element_0", "Element_1"])
    print(f"Found {len(element_spectra)} spectra with Element_0 or Element_1")
    
    # Test matching
    print("\n3. Testing spectral matching...")
    query_wl = np.linspace(400, 1000, 100)
    query_spectrum = np.random.rand(100)  # Random query
    
    # Test different metrics
    metrics = [
        SimilarityMetric.SAM,
        SimilarityMetric.COSINE,
        SimilarityMetric.EUCLIDEAN,
        SimilarityMetric.SID
    ]
    
    for metric in metrics:
        matches = library.match_spectrum(
            query_spectrum,
            query_wl,
            metric=metric,
            k=3
        )
        print(f"\n{metric.value.upper()} - Top 3 matches:")
        for i, match in enumerate(matches, 1):
            print(f"  {i}. {match.spectrum_id}: "
                  f"similarity={match.similarity:.4f}, "
                  f"confidence={match.confidence:.4f}")
    
    # Test batch matching
    print("\n4. Testing batch matching...")
    query_batch = np.random.rand(5, 100)
    batch_results = library.batch_match(
        query_batch,
        query_wl,
        metric=SimilarityMetric.SAM,
        k=3
    )
    print(f"Matched {len(batch_results)} query spectra")
    print(f"First query top match: {batch_results[0][0].spectrum_id} "
          f"(similarity={batch_results[0][0].similarity:.4f})")
    
    # Test persistence
    print("\n5. Testing library persistence...")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # JSON format
        json_path = os.path.join(tmpdir, "library.json")
        library.save(json_path, format="json")
        loaded_json = SpectralLibrary.load(json_path, format="json")
        print(f"JSON: Saved and loaded {len(loaded_json)} spectra")
        
        # Pickle format
        pkl_path = os.path.join(tmpdir, "library.pkl")
        library.save(pkl_path, format="pickle")
        loaded_pkl = SpectralLibrary.load(pkl_path, format="pickle")
        print(f"Pickle: Saved and loaded {len(loaded_pkl)} spectra")
        
        if HAS_H5PY:
            # HDF5 format
            h5_path = os.path.join(tmpdir, "library.h5")
            library.save(h5_path, format="hdf5")
            loaded_h5 = SpectralLibrary.load(h5_path, format="hdf5")
            print(f"HDF5: Saved and loaded {len(loaded_h5)} spectra")
        else:
            print("HDF5: Skipped (h5py not available)")
    
    # Test merging
    print("\n6. Testing library merging...")
    library2 = create_synthetic_library(n_spectra=30, n_bands=100, n_elements=8)
    original_size = len(library)
    library.merge(library2)
    print(f"Merged libraries: {original_size} + {len(library2)} = {len(library)}")
    
    # Test specific spectrum retrieval
    print("\n7. Testing spectrum retrieval...")
    first_id = list(library.spectra.keys())[0]
    spectrum = library.get_spectrum(first_id)
    if spectrum:
        print(f"Retrieved spectrum: {spectrum.spectrum_id}")
        print(f"  - Name: {spectrum.metadata.name}")
        print(f"  - Elements: {list(spectrum.metadata.elements.keys())}")
        print(f"  - Shape: {spectrum.spectrum.shape}")
        print(f"  - Wavelengths: {spectrum.wavelengths.shape}")
    
    print("\n" + "=" * 80)
    print("Spectral Library System - Validation Complete!")
    print("=" * 80)
