"""
Advanced Data Augmentation for Food Images
==========================================

Sophisticated augmentation pipeline specifically designed for food images
while preserving elemental composition properties:

1. Geometric transformations (rotation, flip, crop)
2. Color/lighting adjustments (brightness, contrast, saturation)
3. Cooking simulation (color shifts for raw→cooked)
4. Noise and blur (camera/sensor simulation)
5. Context augmentation (backgrounds, plating)
6. Test-time augmentation (TTA) for robust predictions

Key principle: Augment visual appearance without changing elemental content

References:
- "AutoAugment: Learning Augmentation Strategies" (CVPR 2019)
- "RandAugment: Practical automated data augmentation" (NeurIPS 2020)
- "FoodX-251: Large-Scale Food Classification" (2019)
"""

import random
import math
from typing import Optional, Tuple, List, Dict, Union, Callable
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not installed")

try:
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    HAS_IMAGING = True
except ImportError:
    HAS_IMAGING = False
    print("⚠️  PIL not installed")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ============================================================================
# Configuration
# ============================================================================

class CookingState(str, Enum):
    """Food cooking states"""
    RAW = "raw"
    LIGHTLY_COOKED = "lightly_cooked"
    FULLY_COOKED = "fully_cooked"
    OVERCOOKED = "overcooked"


@dataclass
class AugmentationConfig:
    """Augmentation configuration"""
    
    # Geometric
    rotation_degrees: float = 15
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.1
    scale_range: Tuple[float, float] = (0.8, 1.2)
    crop_scale: Tuple[float, float] = (0.8, 1.0)
    
    # Color/lighting
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    saturation_range: Tuple[float, float] = (0.7, 1.3)
    hue_shift: float = 0.1
    
    # Cooking simulation
    enable_cooking_aug: bool = True
    cooking_intensity: float = 0.3
    
    # Noise/blur
    gaussian_noise_std: float = 0.01
    gaussian_blur_sigma: Tuple[float, float] = (0.1, 2.0)
    motion_blur_prob: float = 0.1
    
    # Advanced
    cutout_prob: float = 0.2
    cutout_size: Tuple[int, int] = (16, 16)
    mixup_alpha: float = 0.2
    
    # TTA
    tta_transforms: int = 10


# ============================================================================
# Cooking Simulation
# ============================================================================

if HAS_IMAGING:
    class CookingSimulator:
        """
        Simulate cooking effects on food images
        
        Simulates visual changes during cooking:
        - Raw → Cooked: Brown/golden color shifts
        - Moisture loss: Increased contrast, darkening
        - Caramelization: Brown tones
        - Charring: Black spots (localized)
        
        Note: This only changes appearance, not elemental composition!
        """
        
        def __init__(self):
            # Color shift matrices for different cooking states
            self.cooking_transforms = {
                CookingState.RAW: {
                    'brightness': 1.0,
                    'contrast': 1.0,
                    'saturation': 1.0,
                    'brown_shift': 0.0
                },
                CookingState.LIGHTLY_COOKED: {
                    'brightness': 0.95,
                    'contrast': 1.05,
                    'saturation': 0.95,
                    'brown_shift': 0.1
                },
                CookingState.FULLY_COOKED: {
                    'brightness': 0.85,
                    'contrast': 1.15,
                    'saturation': 0.85,
                    'brown_shift': 0.25
                },
                CookingState.OVERCOOKED: {
                    'brightness': 0.70,
                    'contrast': 1.25,
                    'saturation': 0.70,
                    'brown_shift': 0.45
                }
            }
        
        def simulate_cooking(
            self,
            image: Image.Image,
            target_state: CookingState,
            intensity: float = 1.0
        ) -> Image.Image:
            """
            Simulate cooking transformation
            
            Args:
                image: Input PIL Image
                target_state: Desired cooking state
                intensity: Effect intensity (0-1)
            
            Returns:
                Transformed image
            """
            params = self.cooking_transforms[target_state]
            
            # Apply transformations with intensity scaling
            result = image
            
            # Brightness
            brightness = 1.0 + (params['brightness'] - 1.0) * intensity
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(brightness)
            
            # Contrast
            contrast = 1.0 + (params['contrast'] - 1.0) * intensity
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(contrast)
            
            # Saturation
            saturation = 1.0 + (params['saturation'] - 1.0) * intensity
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(saturation)
            
            # Brown shift (shift RGB toward brown/golden tones)
            if params['brown_shift'] > 0:
                result = self._apply_brown_shift(result, params['brown_shift'] * intensity)
            
            return result
        
        def _apply_brown_shift(self, image: Image.Image, strength: float) -> Image.Image:
            """Apply brown/golden color shift"""
            # Convert to numpy
            img_array = np.array(image).astype(np.float32)
            
            # Brown color profile (RGB)
            brown_target = np.array([165, 111, 51], dtype=np.float32)
            
            # Shift toward brown
            shifted = img_array * (1 - strength) + brown_target * strength
            shifted = np.clip(shifted, 0, 255).astype(np.uint8)
            
            return Image.fromarray(shifted)
        
        def add_char_marks(
            self,
            image: Image.Image,
            num_marks: int = 3,
            mark_size: Tuple[int, int] = (10, 20)
        ) -> Image.Image:
            """Add random char/burn marks"""
            if not HAS_CV2:
                return image
            
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            
            for _ in range(num_marks):
                # Random position
                x = random.randint(0, w - mark_size[0])
                y = random.randint(0, h - mark_size[1])
                
                # Random size
                mark_w = random.randint(mark_size[0]//2, mark_size[0])
                mark_h = random.randint(mark_size[1]//2, mark_size[1])
                
                # Dark brown/black color
                color = (
                    random.randint(20, 50),
                    random.randint(10, 30),
                    random.randint(5, 15)
                )
                
                # Draw ellipse (char mark)
                cv2.ellipse(
                    img_array,
                    (x + mark_w//2, y + mark_h//2),
                    (mark_w//2, mark_h//2),
                    random.randint(0, 180),
                    0, 360,
                    color,
                    -1
                )
                
                # Blur edges
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(
                    mask,
                    (x + mark_w//2, y + mark_h//2),
                    (mark_w//2 + 2, mark_h//2 + 2),
                    random.randint(0, 180),
                    0, 360,
                    255,
                    -1
                )
                
                blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
                img_array = np.where(mask[..., None] > 0, blurred, img_array)
            
            return Image.fromarray(img_array)


# ============================================================================
# Advanced Augmentation Pipeline
# ============================================================================

if HAS_TORCH:
    class FoodAugmentation(nn.Module):
        """
        Comprehensive food image augmentation pipeline
        
        Combines multiple augmentation techniques while preserving
        elemental composition information
        """
        
        def __init__(self, config: AugmentationConfig, training: bool = True):
            super().__init__()
            self.config = config
            self.training = training
            
            if HAS_IMAGING:
                self.cooking_sim = CookingSimulator()
            
            # Standard transforms
            self.normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        
        def forward(self, image: Union[Image.Image, Tensor]) -> Tensor:
            """
            Apply augmentation pipeline
            
            Args:
                image: PIL Image or Tensor
            
            Returns:
                Augmented tensor
            """
            # Convert to PIL if tensor
            if isinstance(image, Tensor):
                image = TF.to_pil_image(image)
            
            if self.training:
                image = self._apply_training_augmentation(image)
            else:
                image = self._apply_inference_transforms(image)
            
            # Convert to tensor
            tensor = TF.to_tensor(image)
            
            # Normalize
            tensor = self.normalize(tensor)
            
            return tensor
        
        def _apply_training_augmentation(self, image: Image.Image) -> Image.Image:
            """Apply training-time augmentation"""
            # Random resize and crop
            if random.random() < 0.8:
                scale = random.uniform(*self.config.scale_range)
                new_size = (int(image.width * scale), int(image.height * scale))
                image = TF.resize(image, new_size)
            
            # Random crop
            i, j, h, w = T.RandomResizedCrop.get_params(
                image,
                scale=self.config.crop_scale,
                ratio=(0.9, 1.1)
            )
            image = TF.crop(image, i, j, h, w)
            
            # Random flip
            if random.random() < self.config.horizontal_flip_prob:
                image = TF.hflip(image)
            
            if random.random() < self.config.vertical_flip_prob:
                image = TF.vflip(image)
            
            # Random rotation
            if random.random() < 0.5:
                angle = random.uniform(-self.config.rotation_degrees, self.config.rotation_degrees)
                image = TF.rotate(image, angle)
            
            # Color jitter
            if random.random() < 0.8:
                image = self._apply_color_jitter(image)
            
            # Cooking simulation
            if self.config.enable_cooking_aug and random.random() < 0.3:
                image = self._apply_cooking_augmentation(image)
            
            # Blur
            if random.random() < 0.3:
                sigma = random.uniform(*self.config.gaussian_blur_sigma)
                image = image.filter(ImageFilter.GaussianBlur(sigma))
            
            return image
        
        def _apply_inference_transforms(self, image: Image.Image) -> Image.Image:
            """Apply inference-time transforms (center crop, resize)"""
            # Center crop to square
            width, height = image.size
            size = min(width, height)
            left = (width - size) // 2
            top = (height - size) // 2
            image = TF.crop(image, top, left, size, size)
            
            return image
        
        def _apply_color_jitter(self, image: Image.Image) -> Image.Image:
            """Apply random color jitter"""
            # Brightness
            if random.random() < 0.8:
                factor = random.uniform(*self.config.brightness_range)
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(factor)
            
            # Contrast
            if random.random() < 0.8:
                factor = random.uniform(*self.config.contrast_range)
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(factor)
            
            # Saturation
            if random.random() < 0.8:
                factor = random.uniform(*self.config.saturation_range)
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(factor)
            
            # Hue (via HSV)
            if random.random() < 0.5:
                hue_shift = random.uniform(-self.config.hue_shift, self.config.hue_shift)
                image = self._shift_hue(image, hue_shift)
            
            return image
        
        def _shift_hue(self, image: Image.Image, shift: float) -> Image.Image:
            """Shift hue in HSV space"""
            # Convert to HSV
            hsv = image.convert('HSV')
            h, s, v = hsv.split()
            
            # Shift hue
            h_array = np.array(h, dtype=np.int16)
            h_array = (h_array + int(shift * 255)) % 255
            h = Image.fromarray(h_array.astype(np.uint8), mode='L')
            
            # Merge back
            hsv = Image.merge('HSV', (h, s, v))
            rgb = hsv.convert('RGB')
            
            return rgb
        
        def _apply_cooking_augmentation(self, image: Image.Image) -> Image.Image:
            """Simulate cooking transformation"""
            if not HAS_IMAGING:
                return image
            
            # Random cooking state
            state = random.choice(list(CookingState))
            intensity = random.uniform(0.3, self.config.cooking_intensity)
            
            return self.cooking_sim.simulate_cooking(image, state, intensity)
        
        def test_time_augmentation(
            self,
            image: Image.Image,
            num_augmentations: int = 10
        ) -> List[Tensor]:
            """
            Generate multiple augmented versions for TTA
            
            Args:
                image: Input image
                num_augmentations: Number of versions
            
            Returns:
                List of augmented tensors
            """
            augmented = []
            
            # Original (center crop)
            self.training = False
            augmented.append(self.forward(image))
            
            # Augmented versions
            self.training = True
            for _ in range(num_augmentations - 1):
                augmented.append(self.forward(image))
            
            return augmented


    # ============================================================================
    # Cutout / Random Erasing
    # ============================================================================

    class RandomCutout(nn.Module):
        """
        Random cutout augmentation
        
        Randomly masks out rectangular regions to improve robustness
        
        Reference: "Improved Regularization of CNNs with Cutout" (2017)
        """
        
        def __init__(
            self,
            prob: float = 0.5,
            size: Tuple[int, int] = (16, 16),
            fill: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        ):
            super().__init__()
            self.prob = prob
            self.size = size
            self.fill = fill
        
        def forward(self, image: Tensor) -> Tensor:
            """
            Apply cutout
            
            Args:
                image: (C, H, W) tensor
            
            Returns:
                Image with cutout applied
            """
            if random.random() > self.prob:
                return image
            
            c, h, w = image.shape
            
            # Random position
            x = random.randint(0, w - self.size[0])
            y = random.randint(0, h - self.size[1])
            
            # Apply cutout
            image = image.clone()
            for i, fill_value in enumerate(self.fill):
                image[i, y:y+self.size[1], x:x+self.size[0]] = fill_value
            
            return image


    # ============================================================================
    # Auto Augment
    # ============================================================================

    class AutoAugment:
        """
        AutoAugment for food images
        
        Learned augmentation policies optimized for food classification
        
        Reference: "AutoAugment: Learning Augmentation Policies" (CVPR 2019)
        """
        
        def __init__(self):
            # Define augmentation operations
            self.operations = [
                self._rotate,
                self._shear_x,
                self._shear_y,
                self._translate_x,
                self._translate_y,
                self._brightness,
                self._contrast,
                self._saturation,
                self._sharpness
            ]
            
            # Learned policies (example)
            self.policies = [
                # Policy 1
                [('rotate', 0.8, 7), ('brightness', 0.9, 8)],
                # Policy 2
                [('contrast', 0.7, 6), ('sharpness', 0.8, 7)],
                # Policy 3
                [('shear_x', 0.6, 5), ('saturation', 0.7, 6)],
                # Add more policies...
            ]
        
        def __call__(self, image: Image.Image) -> Image.Image:
            """Apply random policy"""
            policy = random.choice(self.policies)
            
            for op_name, prob, magnitude in policy:
                if random.random() < prob:
                    op_func = getattr(self, f'_{op_name}')
                    image = op_func(image, magnitude)
            
            return image
        
        def _rotate(self, image: Image.Image, magnitude: int) -> Image.Image:
            angle = (magnitude / 10) * 30 - 15
            return TF.rotate(image, angle)
        
        def _shear_x(self, image: Image.Image, magnitude: int) -> Image.Image:
            shear = (magnitude / 10) * 0.3 - 0.15
            return TF.affine(image, angle=0, translate=[0, 0], scale=1.0, shear=[shear * 45, 0])
        
        def _shear_y(self, image: Image.Image, magnitude: int) -> Image.Image:
            shear = (magnitude / 10) * 0.3 - 0.15
            return TF.affine(image, angle=0, translate=[0, 0], scale=1.0, shear=[0, shear * 45])
        
        def _translate_x(self, image: Image.Image, magnitude: int) -> Image.Image:
            translate = int((magnitude / 10) * image.width * 0.3)
            return TF.affine(image, angle=0, translate=[translate, 0], scale=1.0, shear=[0, 0])
        
        def _translate_y(self, image: Image.Image, magnitude: int) -> Image.Image:
            translate = int((magnitude / 10) * image.height * 0.3)
            return TF.affine(image, angle=0, translate=[0, translate], scale=1.0, shear=[0, 0])
        
        def _brightness(self, image: Image.Image, magnitude: int) -> Image.Image:
            factor = 1.0 + (magnitude / 10) * 0.4 - 0.2
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)
        
        def _contrast(self, image: Image.Image, magnitude: int) -> Image.Image:
            factor = 1.0 + (magnitude / 10) * 0.4 - 0.2
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
        
        def _saturation(self, image: Image.Image, magnitude: int) -> Image.Image:
            factor = 1.0 + (magnitude / 10) * 0.4 - 0.2
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(factor)
        
        def _sharpness(self, image: Image.Image, magnitude: int) -> Image.Image:
            factor = 1.0 + (magnitude / 10) * 0.4 - 0.2
            enhancer = ImageEnhance.Sharpness(image)
            return enhancer.enhance(factor)


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example usage of augmentation pipeline"""
    if not HAS_TORCH or not HAS_IMAGING:
        print("PyTorch and PIL required")
        return
    
    print("\n" + "="*60)
    print("ADVANCED AUGMENTATION - EXAMPLE")
    print("="*60)
    
    # Create sample image
    image = Image.new('RGB', (256, 256), color=(200, 150, 100))
    
    # Configuration
    config = AugmentationConfig(
        enable_cooking_aug=True,
        cooking_intensity=0.5
    )
    
    # Create augmenter
    augmenter = FoodAugmentation(config, training=True)
    
    print("\n1. Training augmentation...")
    augmented = augmenter(image)
    print(f"   Output shape: {augmented.shape}")
    
    print("\n2. Test-time augmentation...")
    tta_images = augmenter.test_time_augmentation(image, num_augmentations=5)
    print(f"   Generated {len(tta_images)} versions")
    
    print("\n3. Cooking simulation...")
    if HAS_IMAGING:
        cooking_sim = CookingSimulator()
        cooked = cooking_sim.simulate_cooking(image, CookingState.FULLY_COOKED, intensity=0.8)
        print(f"   Simulated cooking: {image.size} -> {cooked.size}")
    
    print("\n✅ Example complete!")


if __name__ == "__main__":
    example_usage()
