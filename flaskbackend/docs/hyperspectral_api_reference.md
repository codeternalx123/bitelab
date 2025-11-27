# Hyperspectral Module - API Reference

**Version**: 1.0  
**Last Updated**: November 2024

Complete API documentation for all 18 hyperspectral components.

---

## Table of Contents

1. [Calibration](#calibration)
2. [Spectral Database](#spectral-database)
3. [Preprocessing](#preprocessing)
4. [Band Selection](#band-selection)
5. [Feature Extraction](#feature-extraction)
6. [Deep Learning](#deep-learning)
7. [Material Analysis](#material-analysis)
8. [Anomaly Detection](#anomaly-detection)
9. [Target Detection](#target-detection)
10. [Change Detection](#change-detection)
11. [Super-Resolution](#super-resolution)
12. [Real-Time Processing](#real-time-processing)

---

## Calibration

**Module**: `app.ai_nutrition.hyperspectral.calibration`

### Classes

#### `HyperspectralCalibrator`

Main calibration engine for hyperspectral cameras.

**Constructor**:
```python
HyperspectralCalibrator(config: Optional[CalibrationConfig] = None)
```

**Parameters**:
- `config`: Calibration configuration (optional)

**Methods**:

##### `calibrate_radiometric(dark_frames, white_frames, linearity_data=None)`

Perform radiometric calibration.

**Parameters**:
- `dark_frames` (List[np.ndarray]): Dark current images (10+ recommended)
- `white_frames` (List[np.ndarray]): White reference images (10+ recommended)
- `linearity_data` (Optional[Dict]): Linearity correction data

**Returns**:
- `RadiometricCalibration`: Calibration parameters (dark, gain, offset, linearity)

**Example**:
```python
calibrator = HyperspectralCalibrator()
rad_cal = calibrator.calibrate_radiometric(dark_frames, white_frames)
```

##### `calibrate_spectral(wavelengths, reference_peaks, reference_wavelengths)`

Perform spectral calibration.

**Parameters**:
- `wavelengths` (np.ndarray): Nominal wavelengths
- `reference_peaks` (np.ndarray): Measured spectral peaks
- `reference_wavelengths` (np.ndarray): Known reference wavelengths

**Returns**:
- `SpectralCalibration`: Wavelength calibration (coefficients, FWHM)

**Example**:
```python
spec_cal = calibrator.calibrate_spectral(
    wavelengths=np.linspace(400, 1000, 100),
    reference_peaks=measured_peaks,
    reference_wavelengths=[435.8, 546.1, 611.9]  # Mercury lines
)
```

##### `calibrate_geometric(checkerboard_images, square_size_mm)`

Perform geometric calibration.

**Parameters**:
- `checkerboard_images` (List[np.ndarray]): Checkerboard calibration images
- `square_size_mm` (float): Physical size of checkerboard squares (mm)

**Returns**:
- `GeometricCalibration`: Distortion maps and spatial resolution

**Example**:
```python
geom_cal = calibrator.calibrate_geometric(
    checkerboard_images=calibration_images,
    square_size_mm=10.0
)
```

##### `apply_radiometric_calibration(image, calibration)`

Apply radiometric calibration to raw image.

**Parameters**:
- `image` (np.ndarray): Raw image, shape (H, W, C)
- `calibration` (RadiometricCalibration): Calibration parameters

**Returns**:
- `np.ndarray`: Calibrated image

**Example**:
```python
calibrated = calibrator.apply_radiometric_calibration(raw_image, rad_cal)
```

##### `validate(image, calibration_data)`

Validate calibration quality.

**Parameters**:
- `image` (np.ndarray): Calibrated image
- `calibration_data` (Dict): Calibration parameters

**Returns**:
- `Dict`: Validation results (SNR, accuracy, uniformity, etc.)

**Example**:
```python
validation = calibrator.validate(calibrated, {
    'radiometric': rad_cal,
    'spectral': spec_cal,
    'geometric': geom_cal
})
print(f"SNR: {validation['snr']:.1f}")
```

##### `generate_certificate(radiometric, spectral, geometric, validation)`

Generate calibration certificate.

**Parameters**:
- `radiometric` (RadiometricCalibration): Radiometric calibration
- `spectral` (SpectralCalibration): Spectral calibration
- `geometric` (GeometricCalibration): Geometric calibration
- `validation` (Dict): Validation results

**Returns**:
- `CalibrationCertificate`: JSON-serializable certificate

**Example**:
```python
cert = calibrator.generate_certificate(rad_cal, spec_cal, geom_cal, validation)
with open("calibration_cert.json", "w") as f:
    json.dump(cert.to_dict(), f, indent=2)
```

---

## Spectral Database

**Module**: `app.ai_nutrition.hyperspectral.spectral_database`

### Classes

#### `SpectralDatabase`

High-performance spectral library with k-NN search.

**Constructor**:
```python
SpectralDatabase(
    db_path: str,
    use_hdf5: bool = True,
    cache_size: int = 1000
)
```

**Parameters**:
- `db_path`: Path to SQLite database file
- `use_hdf5`: Enable HDF5 storage for spectra
- `cache_size`: LRU cache size (number of spectra)

**Methods**:

##### `insert(spectrum, wavelengths, metadata)`

Insert spectrum into database.

**Parameters**:
- `spectrum` (np.ndarray): Spectral data, shape (n_bands,)
- `wavelengths` (np.ndarray): Wavelength values
- `metadata` (SpectrumMetadata): Spectrum metadata

**Returns**:
- `int`: Spectrum ID

**Example**:
```python
db = SpectralDatabase("food_spectra.db")

metadata = SpectrumMetadata(
    category="fruit",
    food_name="apple_red_delicious",
    composition={"Fe": 0.5, "Ca": 10.0, "Mg": 5.0},
    quality_score=0.95,
    source="ICP-MS",
    tags=["organic", "fresh"]
)

spectrum_id = db.insert(spectrum, wavelengths, metadata)
```

##### `get(spectrum_id)`

Retrieve spectrum by ID.

**Parameters**:
- `spectrum_id` (int): Spectrum ID

**Returns**:
- `Tuple[np.ndarray, SpectrumMetadata]`: Spectrum and metadata

**Example**:
```python
spectrum, metadata = db.get(123)
print(f"Retrieved: {metadata.food_name}")
```

##### `search(query_spectrum, config)`

Search for similar spectra.

**Parameters**:
- `query_spectrum` (np.ndarray): Query spectrum
- `config` (SearchConfig): Search configuration

**Returns**:
- `List[SearchResult]`: Ranked search results

**Example**:
```python
config = SearchConfig(
    method="knn",
    k=5,
    min_quality=0.8,
    category_filter="fruit"
)

results = db.search(unknown_spectrum, config)

for i, result in enumerate(results):
    print(f"{i+1}. {result.metadata.food_name} "
          f"(similarity: {result.similarity:.3f})")
```

##### `rebuild_index()`

Rebuild spatial index for fast search.

**Returns**:
- `None`

**Example**:
```python
# Rebuild index after bulk insertions
for spectrum, metadata in large_dataset:
    db.insert(spectrum, wavelengths, metadata)

db.rebuild_index()  # O(N log N) operation
```

##### `filter_by_metadata(**kwargs)`

Filter spectra by metadata criteria.

**Parameters**:
- `kwargs`: Filter criteria (category, source, min_quality, etc.)

**Returns**:
- `List[int]`: Matching spectrum IDs

**Example**:
```python
ids = db.filter_by_metadata(
    category="vegetable",
    min_quality=0.9,
    required_elements=["Fe", "Ca"],
    source="ICP-MS"
)
```

##### `get_stats()`

Get database statistics.

**Returns**:
- `DatabaseStats`: Statistics (count, categories, quality distribution)

**Example**:
```python
stats = db.get_stats()
print(f"Total spectra: {stats.total_count}")
print(f"Categories: {stats.category_counts}")
print(f"Mean quality: {stats.mean_quality:.2f}")
```

---

## Preprocessing

**Module**: `app.ai_nutrition.hyperspectral.spectral_preprocessing`

### Classes

#### `SpectralPreprocessor`

Preprocessing pipeline for hyperspectral images.

**Constructor**:
```python
SpectralPreprocessor(config: Optional[PreprocessConfig] = None)
```

**Methods**:

##### `process(image, dark_current=None, white_reference=None)`

Apply full preprocessing pipeline.

**Parameters**:
- `image` (np.ndarray): Raw image, shape (H, W, C)
- `dark_current` (Optional[np.ndarray]): Dark current image
- `white_reference` (Optional[np.ndarray]): White reference image

**Returns**:
- `np.ndarray`: Preprocessed image

**Example**:
```python
config = PreprocessConfig(
    apply_dark_current=True,
    apply_white_reference=True,
    spatial_filter="gaussian",
    spatial_sigma=1.0,
    spectral_filter="savgol",
    savgol_window=5,
    savgol_polyorder=2,
    remove_bad_bands=True,
    bad_bands=[0, 1, 2, -3, -2, -1],  # First 3 and last 3
    normalization="minmax"
)

preprocessor = SpectralPreprocessor(config)
processed = preprocessor.process(
    raw_image,
    dark_current=dark_frame,
    white_reference=white_frame
)
```

---

## Band Selection

**Module**: `app.ai_nutrition.hyperspectral.band_selection`

### Classes

#### `BandSelector`

Select optimal spectral bands.

**Constructor**:
```python
BandSelector(
    method: SelectionMethod,
    n_bands: int,
    **method_kwargs
)
```

**Methods**:

##### `select(image)`

Select bands from hyperspectral image.

**Parameters**:
- `image` (np.ndarray): Hyperspectral image, shape (H, W, C)

**Returns**:
- `np.ndarray`: Selected band indices

**Example**:
```python
# VCA method
selector = BandSelector(
    method=SelectionMethod.VCA,
    n_bands=20
)
selected = selector.select(image)

# Extract selected bands
reduced = image[:, :, selected]
print(f"Reduced from {image.shape[2]} to {len(selected)} bands")

# Mutual Information method
selector = BandSelector(
    method=SelectionMethod.MUTUAL_INFO,
    n_bands=30
)
selected = selector.select(image)
```

---

## Feature Extraction

**Module**: `app.ai_nutrition.hyperspectral.feature_extraction`

### Classes

#### `FeatureExtractor`

Extract features from hyperspectral data.

**Constructor**:
```python
FeatureExtractor(config: Optional[FeatureConfig] = None)
```

**Methods**:

##### `extract(spectrum, wavelengths=None)`

Extract features from single spectrum.

**Parameters**:
- `spectrum` (np.ndarray): Spectrum, shape (n_bands,)
- `wavelengths` (Optional[np.ndarray]): Wavelength values

**Returns**:
- `Dict[str, float]`: Feature dictionary

**Example**:
```python
extractor = FeatureExtractor(FeatureConfig(
    compute_spectral_shape=True,
    compute_derivatives=True,
    compute_absorption_features=True,
    compute_indices=True
))

features = extractor.extract(spectrum, wavelengths)

print(f"Mean reflectance: {features['mean_reflectance']:.3f}")
print(f"Std reflectance: {features['std_reflectance']:.3f}")
print(f"NDVI: {features['ndvi']:.3f}")
```

##### `extract_spatial(image, wavelengths=None)`

Extract features for each pixel.

**Parameters**:
- `image` (np.ndarray): Hyperspectral image, shape (H, W, C)
- `wavelengths` (Optional[np.ndarray]): Wavelength values

**Returns**:
- `np.ndarray`: Feature image, shape (H, W, n_features)

**Example**:
```python
feature_image = extractor.extract_spatial(hyperspectral_image, wavelengths)
print(f"Extracted {feature_image.shape[2]} features per pixel")
```

---

## Deep Learning

**Module**: `app.ai_nutrition.hyperspectral.spectral_cnn`

### Classes

#### `SpectralCNN`

Deep learning model for hyperspectral analysis.

**Constructor**:
```python
SpectralCNN(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    model_type: ModelType = ModelType.HYBRID,
    dropout_rate: float = 0.5
)
```

**Parameters**:
- `input_shape`: Input shape (H, W, C)
- `num_classes`: Number of output classes
- `model_type`: Architecture type (ONE_D, THREE_D, HYBRID)
- `dropout_rate`: Dropout probability

**Methods**:

##### `forward(x)`

Forward pass.

**Parameters**:
- `x` (torch.Tensor): Input tensor, shape (N, C, H, W)

**Returns**:
- `torch.Tensor`: Predictions, shape (N, num_classes)

**Example**:
```python
import torch

model = SpectralCNN(
    input_shape=(100, 100, 50),
    num_classes=10,
    model_type=ModelType.HYBRID
)

# Input: batch_size=8, channels=50, height=100, width=100
x = torch.randn(8, 50, 100, 100)
output = model(x)  # Shape: (8, 10)
```

##### `predict_with_tta(image, n_augmentations=10)`

Prediction with test-time augmentation.

**Parameters**:
- `image` (np.ndarray): Input image
- `n_augmentations` (int): Number of augmentations

**Returns**:
- `np.ndarray`: Averaged predictions

**Example**:
```python
predictions = model.predict_with_tta(test_image, n_augmentations=10)
```

---

## Material Analysis

### Endmember Extraction

**Module**: `app.ai_nutrition.hyperspectral.endmember_extraction`

#### `EndmemberExtractor`

Extract pure material spectra.

**Constructor**:
```python
EndmemberExtractor(
    method: ExtractionMethod,
    n_endmembers: int
)
```

**Methods**:

##### `extract(image)`

Extract endmembers from image.

**Parameters**:
- `image` (np.ndarray): Hyperspectral image, shape (H, W, C)

**Returns**:
- `np.ndarray`: Endmember spectra, shape (n_endmembers, C)

**Example**:
```python
extractor = EndmemberExtractor(
    method=ExtractionMethod.VCA,
    n_endmembers=5
)

endmembers = extractor.extract(image)
print(f"Extracted {endmembers.shape[0]} endmembers")
```

### Spectral Unmixing

**Module**: `app.ai_nutrition.hyperspectral.spectral_unmixing`

#### `SpectralUnmixer`

Estimate material abundances.

**Constructor**:
```python
SpectralUnmixer(method: UnmixingMethod)
```

**Methods**:

##### `unmix(image, endmembers)`

Unmix image into abundances.

**Parameters**:
- `image` (np.ndarray): Hyperspectral image, shape (H, W, C)
- `endmembers` (np.ndarray): Endmember spectra, shape (n_endmembers, C)

**Returns**:
- `np.ndarray`: Abundance maps, shape (H, W, n_endmembers)

**Example**:
```python
unmixer = SpectralUnmixer(method=UnmixingMethod.FCLS)
abundances = unmixer.unmix(image, endmembers)

# Check abundance of first endmember
print(f"Mean abundance: {abundances[:, :, 0].mean():.2%}")
```

### Classification

**Module**: `app.ai_nutrition.hyperspectral.classification`

#### `HyperspectralClassifier`

Classify hyperspectral pixels.

**Constructor**:
```python
HyperspectralClassifier(
    method: ClassificationMethod,
    n_classes: int
)
```

**Methods**:

##### `train(X, y)`

Train classifier.

**Parameters**:
- `X` (np.ndarray): Training spectra, shape (n_samples, n_bands)
- `y` (np.ndarray): Training labels, shape (n_samples,)

**Returns**:
- `None`

**Example**:
```python
classifier = HyperspectralClassifier(
    method=ClassificationMethod.RANDOM_FOREST,
    n_classes=5
)

# Reshape image to (n_pixels, n_bands)
X_train = train_image.reshape(-1, train_image.shape[2])
y_train = train_labels.reshape(-1)

classifier.train(X_train, y_train)
```

##### `predict(X)`

Predict class labels.

**Parameters**:
- `X` (np.ndarray): Test spectra or image

**Returns**:
- `np.ndarray`: Predicted labels

**Example**:
```python
predictions = classifier.predict(test_image)
# predictions.shape = (H, W)
```

##### `predict_confidence(X)`

Predict with confidence scores.

**Parameters**:
- `X` (np.ndarray): Test spectra or image

**Returns**:
- `Tuple[np.ndarray, np.ndarray]`: Predictions and confidences

**Example**:
```python
predictions, confidence = classifier.predict_confidence(test_image)
print(f"Mean confidence: {confidence.mean():.2%}")
```

---

## Anomaly Detection

**Module**: `app.ai_nutrition.hyperspectral.anomaly_detection`

### Classes

#### `AnomalyDetector`

Detect anomalies in hyperspectral images.

**Constructor**:
```python
AnomalyDetector(config: AnomalyConfig)
```

**Methods**:

##### `detect(image)`

Detect anomalies in image.

**Parameters**:
- `image` (np.ndarray): Hyperspectral image, shape (H, W, C)

**Returns**:
- `AnomalyResult`: Detection results

**Example**:
```python
config = AnomalyConfig(
    method=AnomalyMethod.LOCAL_RX,
    outer_window_size=21,
    inner_window_size=3,
    threshold_percentile=99.0
)

detector = AnomalyDetector(config)
result = detector.detect(image)

print(f"Anomaly percentage: {result.anomaly_percentage:.2f}%")
print(f"Mean anomaly score: {result.mean_score:.3f}")
print(f"Max anomaly score: {result.max_score:.3f}")
```

##### `visualize_results(result, save_path=None)`

Visualize detection results.

**Parameters**:
- `result` (AnomalyResult): Detection results
- `save_path` (Optional[str]): Path to save visualization

**Returns**:
- `None`

**Example**:
```python
detector.visualize_results(result, save_path="anomaly_map.png")
```

---

## Target Detection

**Module**: `app.ai_nutrition.hyperspectral.target_detection`

### Classes

#### `TargetDetector`

Detect specific target materials.

**Constructor**:
```python
TargetDetector(config: TargetConfig)
```

**Methods**:

##### `detect(image, ground_truth=None)`

Detect target in image.

**Parameters**:
- `image` (np.ndarray): Hyperspectral image, shape (H, W, C)
- `ground_truth` (Optional[np.ndarray]): Ground truth for ROC analysis

**Returns**:
- `TargetResult`: Detection results

**Example**:
```python
# Load target signature
target_spectrum = load_reference_spectrum("e_coli")

config = TargetConfig(
    method=DetectionMethod.ACE,
    target_signature=target_spectrum,
    threshold=0.8
)

detector = TargetDetector(config)
result = detector.detect(image, ground_truth=ground_truth_mask)

print(f"Detection percentage: {result.detection_percentage:.2f}%")
print(f"Mean detection score: {result.mean_score:.3f}")
if result.roc_auc:
    print(f"ROC AUC: {result.roc_auc:.3f}")
```

##### `set_target(target_signature)`

Update target signature.

**Parameters**:
- `target_signature` (np.ndarray): New target spectrum

**Returns**:
- `None`

**Example**:
```python
detector.set_target(new_target_spectrum)
```

##### `visualize_detections(result, save_path=None)`

Visualize detection map.

**Parameters**:
- `result` (TargetResult): Detection results
- `save_path` (Optional[str]): Path to save visualization

**Returns**:
- `None`

**Example**:
```python
detector.visualize_detections(result, save_path="detection_map.png")
```

---

## Change Detection

**Module**: `app.ai_nutrition.hyperspectral.change_detection`

### Classes

#### `ChangeDetector`

Detect temporal changes.

**Constructor**:
```python
ChangeDetector(config: ChangeConfig)
```

**Methods**:

##### `detect(image_t1, image_t2)`

Detect changes between two time points.

**Parameters**:
- `image_t1` (np.ndarray): Image at time t1, shape (H, W, C)
- `image_t2` (np.ndarray): Image at time t2, shape (H, W, C)

**Returns**:
- `ChangeResult`: Change detection results

**Example**:
```python
config = ChangeConfig(
    method=ChangeMethod.CVA,
    threshold=0.95,
    normalize_images=True
)

detector = ChangeDetector(config)
result = detector.detect(baseline_image, current_image)

print(f"Change percentage: {result.change_percentage:.2f}%")
print(f"Mean change magnitude: {result.mean_change:.3f}")
```

##### `visualize_changes(result, save_path=None)`

Visualize change map.

**Parameters**:
- `result` (ChangeResult): Change detection results
- `save_path` (Optional[str]): Path to save visualization

**Returns**:
- `None`

**Example**:
```python
detector.visualize_changes(result, save_path="change_map.png")
```

---

## Super-Resolution

**Module**: `app.ai_nutrition.hyperspectral.super_resolution`

### Classes

#### `SuperResolution`

Enhance image resolution.

**Constructor**:
```python
SuperResolution(config: SRConfig)
```

**Methods**:

##### `enhance(image, reference_hr=None)`

Enhance image resolution.

**Parameters**:
- `image` (np.ndarray): Low-resolution image, shape (H, W, C)
- `reference_hr` (Optional[np.ndarray]): High-resolution reference for validation

**Returns**:
- `SRResult`: Super-resolution results

**Example**:
```python
# Spatial super-resolution
config = SRConfig(
    method=SRMethod.BICUBIC,
    mode=SRMode.SPATIAL,
    spatial_scale=2.0
)

sr = SuperResolution(config)
result = sr.enhance(low_res_image, reference_hr=high_res_ground_truth)

print(f"Enhanced {result.original_shape} -> {result.output_shape}")
if result.psnr:
    print(f"PSNR: {result.psnr:.2f} dB")
    print(f"SSIM: {result.ssim:.4f}")

# Spectral super-resolution
config = SRConfig(
    method=SRMethod.BICUBIC,
    mode=SRMode.SPECTRAL,
    spectral_scale=2.0
)

sr = SuperResolution(config)
result = sr.enhance(image)  # Double spectral bands
```

---

## Real-Time Processing

**Module**: `app.ai_nutrition.hyperspectral.realtime_processing`

### Classes

#### `RealtimeProcessor`

Real-time hyperspectral processing.

**Constructor**:
```python
RealtimeProcessor(config: ProcessingConfig)
```

**Methods**:

##### `set_calibration(dark_frame, flat_field)`

Set calibration data.

**Parameters**:
- `dark_frame` (np.ndarray): Dark current frame
- `flat_field` (np.ndarray): Flat field frame

**Returns**:
- `None`

**Example**:
```python
config = ProcessingConfig(
    mode=ProcessingMode.BALANCED,
    target_fps=30,
    max_latency_ms=100
)

processor = RealtimeProcessor(config)
processor.set_calibration(dark_frame, flat_field)
```

##### `process_frame(frame)`

Process single frame.

**Parameters**:
- `frame` (np.ndarray): Input frame, shape (H, W, C)

**Returns**:
- `ProcessingResult`: Processing results

**Example**:
```python
result = processor.process_frame(camera_frame)

if result.success:
    print(f"FPS: {result.fps:.1f}")
    print(f"Latency: {result.latency_ms:.1f}ms")
    print(f"Predictions: {result.predictions}")
```

##### `get_metrics()`

Get performance metrics.

**Returns**:
- `ProcessingMetrics`: Performance statistics

**Example**:
```python
metrics = processor.get_metrics()
print(f"Mean FPS: {metrics.mean_fps:.1f}")
print(f"P95 latency: {metrics.p95_latency_ms:.1f}ms")
print(f"Drop rate: {metrics.drop_rate:.2%}")
```

---

## Data Types

### SpectrumMetadata

```python
@dataclass
class SpectrumMetadata:
    category: str  # Food category
    food_name: str  # Specific food name
    composition: Dict[str, float]  # Element concentrations (ppm)
    quality_score: float  # Quality score (0-1)
    source: str = ""  # Data source (ICP-MS, XRF, etc.)
    acquisition_date: Optional[str] = None
    snr: Optional[float] = None  # Signal-to-noise ratio
    tags: List[str] = field(default_factory=list)
```

### SearchConfig

```python
@dataclass
class SearchConfig:
    method: str = "knn"  # Search method: knn, radius, threshold
    k: int = 5  # Number of neighbors (for knn)
    radius: float = 0.1  # Search radius (for radius)
    threshold: float = 0.9  # Similarity threshold (for threshold)
    
    # Filters
    category_filter: Optional[str] = None
    min_quality: float = 0.0
    required_elements: List[str] = field(default_factory=list)
```

### AnomalyResult

```python
@dataclass
class AnomalyResult:
    anomaly_map: np.ndarray  # Anomaly scores per pixel
    binary_map: np.ndarray  # Binary detection (0/1)
    anomaly_percentage: float  # Percentage of anomalous pixels
    mean_score: float  # Mean anomaly score
    max_score: float  # Maximum anomaly score
    threshold: float  # Threshold used
    method: str  # Method name
    processing_time: float  # Processing time (seconds)
```

### TargetResult

```python
@dataclass
class TargetResult:
    detection_map: np.ndarray  # Detection scores per pixel
    binary_map: np.ndarray  # Binary detection (0/1)
    detection_percentage: float  # Percentage detected
    mean_score: float  # Mean detection score
    confidence_map: Optional[np.ndarray] = None  # Confidence scores
    
    # ROC metrics (if ground truth provided)
    roc_auc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
```

### ChangeResult

```python
@dataclass
class ChangeResult:
    change_map: np.ndarray  # Change magnitude per pixel
    binary_map: np.ndarray  # Binary change detection
    change_percentage: float  # Percentage changed
    mean_change: float  # Mean change magnitude
    max_change: float  # Maximum change
    direction_map: Optional[np.ndarray] = None  # Change direction
    method: str  # Method name
    processing_time: float  # Processing time (seconds)
```

---

## Constants

### Wavelength Ranges

```python
VISIBLE_RANGE = (400, 700)  # nm
NIR_RANGE = (700, 1000)  # nm
SWIR_RANGE = (1000, 2500)  # nm
```

### Typical Band Widths

```python
NARROWBAND = 5  # nm (hyperspectral)
MEDIUMBAND = 20  # nm (multispectral)
BROADBAND = 100  # nm (RGB)
```

### Quality Thresholds

```python
MIN_SNR = 100  # Minimum signal-to-noise ratio
MIN_QUALITY = 0.8  # Minimum quality score
MAX_NOISE_LEVEL = 0.01  # Maximum acceptable noise
```

---

## Error Handling

All functions raise appropriate exceptions:

- `ValueError`: Invalid input parameters
- `RuntimeError`: Processing errors
- `FileNotFoundError`: Missing files
- `MemoryError`: Insufficient memory

**Example**:
```python
try:
    result = detector.detect(image)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Processing error: {e}")
```

---

## Performance Tips

1. **Use band selection**: Reduces computation by 5-10×
2. **Enable caching**: Set `cache_size=1000` in database
3. **Parallel processing**: Use `ThreadPoolExecutor` for batch processing
4. **GPU acceleration**: Move models to GPU with `.cuda()`
5. **Quantization**: Use INT8 quantization for 4× smaller models

---

**API Version**: 1.0  
**Last Updated**: November 2024  
**Maintainer**: AI Nutrition Team
