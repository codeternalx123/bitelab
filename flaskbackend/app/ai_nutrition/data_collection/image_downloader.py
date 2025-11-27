"""
Image Download and Management System
====================================

Download, validate, and manage food images for the collected samples.

Features:
- Async batch downloading
- Image validation (format, size, quality)
- Duplicate detection (perceptual hashing)
- Automatic resizing and normalization
- Error handling and retry logic
- Progress tracking
- Storage optimization

Supports:
- Direct URLs from data sources
- Google Images scraping (fallback)
- Stock photo APIs (Unsplash, Pexels)
- Local file management
"""

import asyncio
import aiohttp
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

try:
    from PIL import Image
    import PIL.Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️  Pillow not installed: pip install Pillow")

try:
    import imagehash  # type: ignore
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("⚠️  imagehash not installed: pip install imagehash")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


logger = logging.getLogger(__name__)


class ImageQuality(Enum):
    """Image quality levels"""
    EXCELLENT = "excellent"  # >1024px, clear, good lighting
    GOOD = "good"           # >512px, acceptable quality
    ACCEPTABLE = "acceptable"  # >256px, usable
    POOR = "poor"           # <256px or quality issues
    INVALID = "invalid"      # Corrupted or wrong format


@dataclass
class ImageMetadata:
    """Metadata for downloaded image"""
    
    sample_id: str
    image_path: Path
    source_url: Optional[str]
    
    # Image properties
    width: int
    height: int
    format: str
    file_size_bytes: int
    
    # Quality assessment
    quality: ImageQuality
    quality_score: float  # 0-100
    
    # Perceptual hash for deduplication
    phash: Optional[str] = None
    
    # Processing info
    downloaded_at: datetime = field(default_factory=datetime.now)
    processed: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'sample_id': self.sample_id,
            'image_path': str(self.image_path),
            'source_url': self.source_url,
            'width': self.width,
            'height': self.height,
            'format': self.format,
            'file_size_mb': self.file_size_bytes / (1024 * 1024),
            'quality': self.quality.value,
            'quality_score': self.quality_score,
            'phash': self.phash,
            'downloaded_at': self.downloaded_at.isoformat(),
            'processed': self.processed,
            'error': self.error
        }


@dataclass
class DownloadConfig:
    """Configuration for image downloading"""
    
    output_dir: Path = Path("data/images/raw")
    processed_dir: Path = Path("data/images/processed")
    
    # Download settings
    batch_size: int = 50
    max_concurrent: int = 10
    timeout_seconds: int = 30
    max_retries: int = 3
    
    # Image requirements
    min_width: int = 224
    min_height: int = 224
    max_file_size_mb: int = 50
    allowed_formats: Set[str] = field(default_factory=lambda: {'JPEG', 'PNG', 'JPG'})
    
    # Processing
    target_size: Tuple[int, int] = (512, 512)
    save_original: bool = True
    save_processed: bool = True
    
    # Quality thresholds
    min_quality_score: float = 30.0  # Reject below this
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


class ImageDownloader:
    """Download and process food images"""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.metadata: List[ImageMetadata] = []
        self.phashes: Set[str] = set()  # For duplicate detection
        
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'duplicate': 0,
            'low_quality': 0
        }
    
    async def download_batch(
        self,
        samples: List[Tuple[str, str]]  # (sample_id, url)
    ) -> List[ImageMetadata]:
        """Download a batch of images"""
        
        self.logger.info(f"📥 Downloading {len(samples)} images...")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for sample_id, url in samples:
                task = self._download_image(session, sample_id, url)
                tasks.append(task)
                
                # Batch processing
                if len(tasks) >= self.config.max_concurrent:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    self._process_results(results)
                    tasks = []
            
            # Process remaining
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                self._process_results(results)
        
        self._print_stats()
        
        return self.metadata
    
    async def _download_image(
        self,
        session: aiohttp.ClientSession,
        sample_id: str,
        url: str
    ) -> Optional[ImageMetadata]:
        """Download single image with retry"""
        
        self.stats['total'] += 1
        
        for attempt in range(self.config.max_retries):
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    
                    if response.status != 200:
                        self.logger.warning(f"HTTP {response.status} for {url}")
                        continue
                    
                    # Check content type
                    content_type = response.headers.get('Content-Type', '')
                    if 'image' not in content_type:
                        self.logger.warning(f"Not an image: {content_type}")
                        return None
                    
                    # Read image data
                    image_data = await response.read()
                    
                    # Check size
                    size_mb = len(image_data) / (1024 * 1024)
                    if size_mb > self.config.max_file_size_mb:
                        self.logger.warning(f"Image too large: {size_mb:.1f}MB")
                        return None
                    
                    # Save and process
                    metadata = self._process_image(sample_id, url, image_data)
                    
                    if metadata:
                        self.stats['success'] += 1
                        return metadata
                    else:
                        self.stats['failed'] += 1
                        return None
            
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout downloading {url} (attempt {attempt + 1})")
            
            except Exception as e:
                self.logger.error(f"Error downloading {url}: {e}")
            
            # Backoff
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        
        self.stats['failed'] += 1
        return None
    
    def _process_image(
        self,
        sample_id: str,
        url: str,
        image_data: bytes
    ) -> Optional[ImageMetadata]:
        """Process and validate downloaded image"""
        
        if not HAS_PIL:
            self.logger.error("PIL not available")
            return None
        
        try:
            # Load image
            from io import BytesIO
            image = Image.open(BytesIO(image_data))
            
            # Basic validation
            if image.format not in self.config.allowed_formats:
                self.logger.warning(f"Unsupported format: {image.format}")
                return None
            
            width, height = image.size
            
            if width < self.config.min_width or height < self.config.min_height:
                self.logger.warning(f"Image too small: {width}x{height}")
                return None
            
            # Quality assessment
            quality_score = self._assess_quality(image)
            
            if quality_score < self.config.min_quality_score:
                self.logger.warning(f"Low quality image: {quality_score:.1f}")
                self.stats['low_quality'] += 1
                return None
            
            # Determine quality level
            if quality_score >= 80 and min(width, height) >= 1024:
                quality = ImageQuality.EXCELLENT
            elif quality_score >= 60 and min(width, height) >= 512:
                quality = ImageQuality.GOOD
            elif quality_score >= 40 and min(width, height) >= 256:
                quality = ImageQuality.ACCEPTABLE
            else:
                quality = ImageQuality.POOR
            
            # Compute perceptual hash for duplicate detection
            phash = None
            if HAS_IMAGEHASH:
                phash = str(imagehash.phash(image))
                
                # Check for duplicates
                if phash in self.phashes:
                    self.logger.info(f"Duplicate image detected: {phash}")
                    self.stats['duplicate'] += 1
                    return None
                
                self.phashes.add(phash)
            
            # Save original
            file_ext = image.format.lower()
            image_filename = f"{sample_id}.{file_ext}"
            image_path = self.config.output_dir / image_filename
            
            if self.config.save_original:
                image.save(image_path, quality=95)
            
            # Process and save resized version
            if self.config.save_processed:
                processed_image = self._resize_image(image)
                processed_path = self.config.processed_dir / image_filename
                processed_image.save(processed_path, quality=90)
            
            # Create metadata
            metadata = ImageMetadata(
                sample_id=sample_id,
                image_path=image_path,
                source_url=url,
                width=width,
                height=height,
                format=image.format,
                file_size_bytes=len(image_data),
                quality=quality,
                quality_score=quality_score,
                phash=phash,
                processed=True
            )
            
            self.metadata.append(metadata)
            
            return metadata
        
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return None
    
    def _assess_quality(self, image: PIL.Image.Image) -> float:
        """Assess image quality (0-100 score)"""
        
        if not HAS_NUMPY:
            return 50.0  # Default score
        
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Check if grayscale
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            # Metrics for quality assessment
            
            # 1. Brightness
            brightness = img_array.mean()
            brightness_score = 100 - abs(brightness - 128) / 1.28  # Penalize very dark/bright
            
            # 2. Contrast
            contrast = img_array.std()
            contrast_score = min(contrast / 0.5, 100)  # Higher is better
            
            # 3. Sharpness (Laplacian variance)
            # Simplified: use gradient
            if img_array.shape[2] >= 3:
                gray = img_array[:, :, 0] * 0.299 + img_array[:, :, 1] * 0.587 + img_array[:, :, 2] * 0.114
            else:
                gray = img_array[:, :, 0]
            
            gradient_x = np.abs(np.diff(gray, axis=1)).mean()
            gradient_y = np.abs(np.diff(gray, axis=0)).mean()
            sharpness = (gradient_x + gradient_y) / 2
            sharpness_score = min(sharpness * 2, 100)
            
            # 4. Color saturation (if RGB)
            if img_array.shape[2] >= 3:
                saturation = np.std(img_array, axis=2).mean()
                saturation_score = min(saturation / 0.3, 100)
            else:
                saturation_score = 50.0
            
            # Combined score (weighted average)
            quality_score = (
                brightness_score * 0.2 +
                contrast_score * 0.3 +
                sharpness_score * 0.3 +
                saturation_score * 0.2
            )
            
            return min(max(quality_score, 0), 100)
        
        except Exception as e:
            self.logger.error(f"Error assessing quality: {e}")
            return 50.0
    
    def _resize_image(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """Resize image to target size with padding"""
        
        # Calculate resize dimensions (maintain aspect ratio)
        width, height = image.size
        target_w, target_h = self.config.target_size
        
        aspect = width / height
        target_aspect = target_w / target_h
        
        if aspect > target_aspect:
            # Width is limiting
            new_width = target_w
            new_height = int(target_w / aspect)
        else:
            # Height is limiting
            new_height = target_h
            new_width = int(target_h * aspect)
        
        # Resize
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create padded image (center the resized image)
        padded = Image.new('RGB', (target_w, target_h), (128, 128, 128))  # Gray padding
        
        paste_x = (target_w - new_width) // 2
        paste_y = (target_h - new_height) // 2
        
        padded.paste(resized, (paste_x, paste_y))
        
        return padded
    
    def _process_results(self, results: List):
        """Process batch results"""
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Download failed: {result}")
            elif isinstance(result, ImageMetadata):
                # Already processed
                pass
    
    def _print_stats(self):
        """Print download statistics"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("📊 DOWNLOAD STATISTICS")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"  Total attempted: {self.stats['total']}")
        self.logger.info(f"  ✅ Success: {self.stats['success']}")
        self.logger.info(f"  ❌ Failed: {self.stats['failed']}")
        self.logger.info(f"  🔁 Duplicates: {self.stats['duplicate']}")
        self.logger.info(f"  ⚠️  Low quality: {self.stats['low_quality']}")
        
        success_rate = self.stats['success'] / max(1, self.stats['total']) * 100
        self.logger.info(f"  📈 Success rate: {success_rate:.1f}%")
        self.logger.info(f"{'='*60}\n")
    
    def export_metadata(self, output_path: Path):
        """Export image metadata to JSON"""
        import json
        
        data = {
            'total_images': len(self.metadata),
            'statistics': self.stats,
            'images': [m.to_dict() for m in self.metadata]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"💾 Exported metadata to {output_path}")


class GoogleImagesScraper:
    """
    Fallback: Scrape Google Images for food items without URLs
    Note: Use sparingly and respect robots.txt and rate limits
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def search(
        self,
        query: str,
        num_results: int = 10
    ) -> List[str]:
        """Search Google Images and return URLs"""
        
        self.logger.warning("Google Images scraping not implemented (requires Selenium)")
        self.logger.warning("Consider using stock photo APIs instead")
        
        # In production, use Selenium with webdriver
        # or use official APIs like:
        # - Google Custom Search API
        # - Unsplash API
        # - Pexels API
        
        return []


class StockPhotoAPI:
    """
    Use free stock photo APIs as fallback
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
    
    async def search_unsplash(
        self,
        query: str,
        num_results: int = 10
    ) -> List[str]:
        """Search Unsplash for food images"""
        
        if not self.api_key:
            self.logger.warning("Unsplash API key not provided")
            return []
        
        base_url = "https://api.unsplash.com/search/photos"
        
        async with aiohttp.ClientSession() as session:
            params = {
                'query': query,
                'per_page': num_results,
                'orientation': 'squarish'
            }
            
            headers = {
                'Authorization': f'Client-ID {self.api_key}'
            }
            
            try:
                async with session.get(base_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        urls = []
                        for result in data.get('results', []):
                            url = result.get('urls', {}).get('regular')
                            if url:
                                urls.append(url)
                        
                        return urls
                    else:
                        self.logger.error(f"Unsplash API error: {response.status}")
                        return []
            
            except Exception as e:
                self.logger.error(f"Error searching Unsplash: {e}")
                return []


async def main():
    """Test image downloader"""
    
    # Configuration
    config = DownloadConfig(
        output_dir=Path("data/images/raw"),
        processed_dir=Path("data/images/processed"),
        batch_size=50,
        max_concurrent=10
    )
    
    # Sample URLs (replace with real food image URLs)
    samples = [
        ("apple_001", "https://example.com/apple.jpg"),
        ("banana_001", "https://example.com/banana.jpg"),
        # Add more...
    ]
    
    # Download
    downloader = ImageDownloader(config)
    metadata = await downloader.download_batch(samples)
    
    # Export metadata
    downloader.export_metadata(Path("data/image_metadata.json"))
    
    print(f"\n✅ Downloaded {len(metadata)} images")


if __name__ == '__main__':
    asyncio.run(main())
