"""
Computer Vision - Image Preprocessing Pipeline
Handles image loading, validation, resizing, normalization, and augmentation
for food recognition AI training and inference.

Part of Phase 2: Computer Vision System
Author: AI Nutrition System
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImageFormat(Enum):
    """Supported image formats"""
    JPEG = "jpg"
    PNG = "png"
    BMP = "bmp"
    WEBP = "webp"
    HEIC = "heic"


class PreprocessMode(Enum):
    """Preprocessing modes for different use cases"""
    TRAINING = "training"  # With augmentation
    VALIDATION = "validation"  # Without augmentation
    INFERENCE = "inference"  # Production mode
    MOBILE = "mobile"  # Optimized for mobile


@dataclass
class ImageConfig:
    """Configuration for image preprocessing"""
    target_size: Tuple[int, int] = (224, 224)  # Standard for most CV models
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet mean
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)  # ImageNet std
    max_file_size_mb: int = 10  # Maximum file size
    min_dimension: int = 32  # Minimum width/height
    max_dimension: int = 4096  # Maximum width/height
    quality: int = 95  # JPEG quality for saving


class ImagePreprocessor:
    """
    Handles all image preprocessing for food recognition CV system.
    Supports training augmentation, validation, and inference modes.
    """
    
    def __init__(self, config: Optional[ImageConfig] = None):
        """
        Initialize image preprocessor with configuration.
        
        Args:
            config: ImageConfig object with preprocessing parameters
        """
        self.config = config or ImageConfig()
        self._setup_augmentation_pipelines()
    
    def _setup_augmentation_pipelines(self):
        """
        Set up albumentations pipelines for different modes.
        Training: Aggressive augmentation for model robustness.
        Validation: Minimal augmentation for fair evaluation.
        Inference: Only essential preprocessing.
        """
        
        # Training pipeline - aggressive augmentation
        self.train_transform = A.Compose([
            # Geometric transformations
            A.Rotate(limit=20, p=0.5),  # Rotate ±20 degrees
            A.HorizontalFlip(p=0.5),  # Mirror images
            A.ShiftScaleRotate(
                shift_limit=0.1,  # Shift image
                scale_limit=0.2,  # Zoom 0.8-1.2x
                rotate_limit=15,
                p=0.5
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),  # Perspective distortion
            
            # Color transformations
            A.RandomBrightnessContrast(
                brightness_limit=0.3,  # ±30% brightness
                contrast_limit=0.3,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,  # Color shift
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
            
            # Noise and blur
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.MotionBlur(blur_limit=5, p=0.3),
                A.MedianBlur(blur_limit=5, p=0.3),
            ], p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            
            # Lighting conditions
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                p=0.3
            ),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
            
            # Image quality
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),
            
            # Normalization and conversion
            A.Resize(height=self.config.target_size[0], width=self.config.target_size[1]),
            A.Normalize(mean=self.config.mean, std=self.config.std),
            ToTensorV2(),
        ])
        
        # Validation pipeline - minimal augmentation
        self.val_transform = A.Compose([
            A.Resize(height=self.config.target_size[0], width=self.config.target_size[1]),
            A.Normalize(mean=self.config.mean, std=self.config.std),
            ToTensorV2(),
        ])
        
        # Inference pipeline - same as validation
        self.inference_transform = self.val_transform
        
        # Mobile pipeline - optimized for mobile devices
        self.mobile_transform = A.Compose([
            A.Resize(height=self.config.target_size[0], width=self.config.target_size[1]),
            A.Normalize(mean=self.config.mean, std=self.config.std),
            # Note: ToTensorV2 not used for mobile (numpy arrays instead)
        ])
    
    def load_image(
        self, 
        image_path: Union[str, Path],
        mode: str = 'RGB'
    ) -> Optional[np.ndarray]:
        """
        Load image from file path with validation.
        
        Args:
            image_path: Path to image file
            mode: Color mode ('RGB', 'BGR', 'GRAY')
        
        Returns:
            numpy array of image or None if invalid
        """
        try:
            image_path = Path(image_path)
            
            # Validate file exists
            if not image_path.exists():
                print(f"❌ Image not found: {image_path}")
                return None
            
            # Validate file size
            file_size_mb = image_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                print(f"❌ Image too large: {file_size_mb:.1f}MB (max: {self.config.max_file_size_mb}MB)")
                return None
            
            # Load image
            if mode == 'BGR':
                image = cv2.imread(str(image_path))
                if image is None:
                    return None
            else:
                image = np.array(Image.open(image_path).convert(mode))
            
            # Validate dimensions
            if len(image.shape) < 2:
                print(f"❌ Invalid image dimensions: {image.shape}")
                return None
            
            height, width = image.shape[:2]
            if height < self.config.min_dimension or width < self.config.min_dimension:
                print(f"❌ Image too small: {width}x{height} (min: {self.config.min_dimension})")
                return None
            
            if height > self.config.max_dimension or width > self.config.max_dimension:
                print(f"⚠️ Image very large: {width}x{height}, resizing...")
                scale = self.config.max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return image
        
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
    
    def load_from_bytes(
        self,
        image_bytes: bytes,
        mode: str = 'RGB'
    ) -> Optional[np.ndarray]:
        """
        Load image from bytes (useful for API uploads).
        
        Args:
            image_bytes: Image data as bytes
            mode: Color mode ('RGB', 'BGR', 'GRAY')
        
        Returns:
            numpy array of image or None if invalid
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image
            if mode == 'BGR':
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif mode == 'GRAY':
                image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            else:  # RGB
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if image is None:
                print("❌ Failed to decode image from bytes")
                return None
            
            return image
        
        except Exception as e:
            print(f"❌ Error loading image from bytes: {e}")
            return None
    
    def preprocess(
        self,
        image: np.ndarray,
        mode: PreprocessMode = PreprocessMode.INFERENCE
    ) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Preprocess image based on mode (training, validation, inference).
        
        Args:
            image: Input image as numpy array
            mode: Preprocessing mode
        
        Returns:
            Preprocessed image (tensor for PyTorch models, array for mobile)
        """
        try:
            # Select appropriate transform pipeline
            if mode == PreprocessMode.TRAINING:
                transform = self.train_transform
            elif mode == PreprocessMode.VALIDATION:
                transform = self.val_transform
            elif mode == PreprocessMode.MOBILE:
                transform = self.mobile_transform
            else:  # INFERENCE
                transform = self.inference_transform
            
            # Apply transformations
            transformed = transform(image=image)
            
            # Return tensor or numpy array
            if mode == PreprocessMode.MOBILE:
                return transformed['image']  # numpy array
            else:
                return transformed['image']  # PyTorch tensor
        
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None
    
    def preprocess_batch(
        self,
        images: List[np.ndarray],
        mode: PreprocessMode = PreprocessMode.INFERENCE
    ) -> List[Union[np.ndarray, 'torch.Tensor']]:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of images as numpy arrays
            mode: Preprocessing mode
        
        Returns:
            List of preprocessed images
        """
        return [self.preprocess(img, mode) for img in images if img is not None]
    
    def denormalize(
        self,
        tensor: 'torch.Tensor'
    ) -> np.ndarray:
        """
        Convert normalized tensor back to displayable image (0-255).
        Useful for visualizing augmented images.
        
        Args:
            tensor: Normalized image tensor (C, H, W)
        
        Returns:
            Image as numpy array (H, W, C) in range 0-255
        """
        try:
            import torch
            
            # Convert tensor to numpy
            if isinstance(tensor, torch.Tensor):
                image = tensor.cpu().numpy()
            else:
                image = tensor
            
            # Transpose from (C, H, W) to (H, W, C)
            if len(image.shape) == 3 and image.shape[0] in [1, 3]:
                image = np.transpose(image, (1, 2, 0))
            
            # Denormalize
            mean = np.array(self.config.mean)
            std = np.array(self.config.std)
            image = image * std + mean
            
            # Clip to valid range and convert to uint8
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            
            return image
        
        except Exception as e:
            print(f"❌ Error denormalizing image: {e}")
            return None
    
    def save_image(
        self,
        image: np.ndarray,
        save_path: Union[str, Path],
        quality: Optional[int] = None
    ) -> bool:
        """
        Save image to file.
        
        Args:
            image: Image as numpy array
            save_path: Path to save image
            quality: JPEG quality (1-100)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            quality = quality or self.config.quality
            
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Save with quality setting
            if save_path.suffix.lower() in ['.jpg', '.jpeg']:
                cv2.imwrite(str(save_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(str(save_path), image)
            
            return True
        
        except Exception as e:
            print(f"❌ Error saving image: {e}")
            return False
    
    def visualize_augmentation(
        self,
        image_path: Union[str, Path],
        num_examples: int = 9,
        save_path: Optional[Union[str, Path]] = None
    ) -> bool:
        """
        Visualize augmentation by showing original + augmented versions.
        
        Args:
            image_path: Path to input image
            num_examples: Number of augmented examples to generate
            save_path: Optional path to save visualization
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import matplotlib.pyplot as plt
            
            # Load original image
            image = self.load_image(image_path)
            if image is None:
                return False
            
            # Create grid
            cols = 3
            rows = (num_examples + 1 + cols - 1) // cols  # +1 for original
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            axes = axes.flatten()
            
            # Show original
            axes[0].imshow(image)
            axes[0].set_title("Original", fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Generate and show augmented versions
            for i in range(1, num_examples + 1):
                augmented = self.preprocess(image, PreprocessMode.TRAINING)
                augmented_np = self.denormalize(augmented)
                
                axes[i].imshow(augmented_np)
                axes[i].set_title(f"Augmented {i}", fontsize=12)
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(num_examples + 1, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"✅ Visualization saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            return True
        
        except Exception as e:
            print(f"❌ Error visualizing augmentation: {e}")
            return False


def test_preprocessing_pipeline():
    """Test the preprocessing pipeline with sample image"""
    print("🧪 Testing Image Preprocessing Pipeline")
    print("=" * 50)
    
    # Initialize preprocessor
    config = ImageConfig(target_size=(224, 224))
    preprocessor = ImagePreprocessor(config)
    
    print("✅ Preprocessor initialized")
    print(f"   Target size: {config.target_size}")
    print(f"   Mean: {config.mean}")
    print(f"   Std: {config.std}")
    print()
    
    # Test with sample data
    print("📝 Test Summary:")
    print("   - Training mode: Aggressive augmentation (rotation, flip, color, blur, etc.)")
    print("   - Validation mode: Resize + normalize only")
    print("   - Inference mode: Same as validation")
    print("   - Mobile mode: Optimized for mobile devices")
    print()
    
    print("🎯 Ready for Phase 2 Computer Vision!")
    print("   Next steps:")
    print("   1. Download Food-101 dataset")
    print("   2. Build YOLOv8 training pipeline")
    print("   3. Train food detection model")
    
    return preprocessor


if __name__ == "__main__":
    # Test the preprocessing pipeline
    preprocessor = test_preprocessing_pipeline()
