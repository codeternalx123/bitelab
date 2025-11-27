"""
Advanced ML Models - Part 4 Phase 4: Semantic Segmentation Models
==================================================================

This module implements semantic segmentation architectures for pixel-level
food ingredient identification and precise portion estimation.

Features:
- U-Net architecture (encoder-decoder with skip connections)
- DeepLab v3+ architecture (atrous convolutions, ASPP)
- PSPNet (Pyramid Scene Parsing Network)
- Attention mechanisms for food segmentation
- Multi-scale segmentation
- Boundary refinement
- Portion volume estimation
- Integration with object detection

Author: Wellomex AI Team
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from pathlib import Path
from enum import Enum

# Optional imports with availability flags
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SegmentationResult:
    """Semantic segmentation result."""
    mask: np.ndarray  # [H, W] or [H, W, C] for multi-class
    class_ids: np.ndarray  # Unique class IDs in mask
    class_names: Optional[List[str]] = None
    confidence: Optional[np.ndarray] = None  # [H, W] confidence map
    
    def get_class_mask(self, class_id: int) -> np.ndarray:
        """Get binary mask for specific class."""
        return (self.mask == class_id).astype(np.uint8)
    
    def compute_areas(self, pixel_size_cm2: float = 1.0) -> Dict[int, float]:
        """Compute area for each class in cm²."""
        areas = {}
        for class_id in self.class_ids:
            mask = self.get_class_mask(class_id)
            pixel_count = np.sum(mask)
            areas[class_id] = pixel_count * pixel_size_cm2
        return areas
    
    def to_color_map(self, colormap: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert segmentation mask to RGB visualization."""
        if colormap is None:
            # Generate default colormap
            np.random.seed(42)
            num_classes = len(self.class_ids)
            colormap = np.random.randint(0, 255, (num_classes + 1, 3), dtype=np.uint8)
            colormap[0] = [0, 0, 0]  # Background
        
        h, w = self.mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id in self.class_ids:
            color_mask[self.mask == class_id] = colormap[class_id]
        
        return color_mask


class SegmentationMetric(Enum):
    """Segmentation evaluation metrics."""
    IOU = "iou"  # Intersection over Union
    DICE = "dice"  # Dice coefficient
    PIXEL_ACCURACY = "pixel_accuracy"
    MEAN_IOU = "mean_iou"
    FREQUENCY_WEIGHTED_IOU = "frequency_weighted_iou"


# ============================================================================
# U-NET ARCHITECTURE
# ============================================================================

if TORCH_AVAILABLE:
    
    class DoubleConv(nn.Module):
        """Double convolution block for U-Net."""
        
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: Optional[int] = None
        ):
            super().__init__()
            
            if mid_channels is None:
                mid_channels = out_channels
            
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x: Tensor) -> Tensor:
            return self.double_conv(x)
    
    
    class DownBlock(nn.Module):
        """Downsampling block for U-Net encoder."""
        
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )
        
        def forward(self, x: Tensor) -> Tensor:
            return self.maxpool_conv(x)
    
    
    class UpBlock(nn.Module):
        """Upsampling block for U-Net decoder."""
        
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bilinear: bool = True
        ):
            super().__init__()
            
            # Upsampling
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            else:
                self.up = nn.ConvTranspose2d(
                    in_channels, in_channels // 2, kernel_size=2, stride=2
                )
                self.conv = DoubleConv(in_channels, out_channels)
        
        def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
            """
            Forward with skip connection.
            
            Args:
                x1: Features from decoder
                x2: Features from encoder (skip connection)
            """
            x1 = self.up(x1)
            
            # Handle size mismatch
            diff_h = x2.size(2) - x1.size(2)
            diff_w = x2.size(3) - x1.size(3)
            
            x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                           diff_h // 2, diff_h - diff_h // 2])
            
            # Concatenate skip connection
            x = torch.cat([x2, x1], dim=1)
            
            return self.conv(x)
    
    
    class FoodUNet(nn.Module):
        """U-Net for food ingredient segmentation."""
        
        def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 100,
            base_channels: int = 64,
            bilinear: bool = True
        ):
            """
            Initialize U-Net.
            
            Args:
                in_channels: Number of input channels (3 for RGB)
                num_classes: Number of segmentation classes
                base_channels: Base number of channels (default 64)
                bilinear: Use bilinear upsampling (True) or transposed conv (False)
            """
            super().__init__()
            
            self.in_channels = in_channels
            self.num_classes = num_classes
            self.bilinear = bilinear
            
            # Encoder
            self.inc = DoubleConv(in_channels, base_channels)
            self.down1 = DownBlock(base_channels, base_channels * 2)
            self.down2 = DownBlock(base_channels * 2, base_channels * 4)
            self.down3 = DownBlock(base_channels * 4, base_channels * 8)
            
            factor = 2 if bilinear else 1
            self.down4 = DownBlock(base_channels * 8, base_channels * 16 // factor)
            
            # Decoder
            self.up1 = UpBlock(base_channels * 16, base_channels * 8 // factor, bilinear)
            self.up2 = UpBlock(base_channels * 8, base_channels * 4 // factor, bilinear)
            self.up3 = UpBlock(base_channels * 4, base_channels * 2 // factor, bilinear)
            self.up4 = UpBlock(base_channels * 2, base_channels, bilinear)
            
            # Output
            self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        
        def forward(self, x: Tensor) -> Tensor:
            """Forward pass."""
            # Encoder
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            # Decoder with skip connections
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            
            # Output
            logits = self.outc(x)
            
            return logits
        
        def extract_features(self, x: Tensor) -> List[Tensor]:
            """Extract multi-scale features."""
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            return [x1, x2, x3, x4, x5]
    
    
    class AttentionUNet(nn.Module):
        """U-Net with attention gates for food segmentation."""
        
        def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 100,
            base_channels: int = 64
        ):
            super().__init__()
            
            self.in_channels = in_channels
            self.num_classes = num_classes
            
            # Encoder
            self.inc = DoubleConv(in_channels, base_channels)
            self.down1 = DownBlock(base_channels, base_channels * 2)
            self.down2 = DownBlock(base_channels * 2, base_channels * 4)
            self.down3 = DownBlock(base_channels * 4, base_channels * 8)
            self.down4 = DownBlock(base_channels * 8, base_channels * 16)
            
            # Attention gates
            self.att1 = AttentionGate(base_channels * 8, base_channels * 16, base_channels * 8)
            self.att2 = AttentionGate(base_channels * 4, base_channels * 8, base_channels * 4)
            self.att3 = AttentionGate(base_channels * 2, base_channels * 4, base_channels * 2)
            self.att4 = AttentionGate(base_channels, base_channels * 2, base_channels)
            
            # Decoder
            self.up1 = UpBlock(base_channels * 16, base_channels * 8, bilinear=False)
            self.up2 = UpBlock(base_channels * 8, base_channels * 4, bilinear=False)
            self.up3 = UpBlock(base_channels * 4, base_channels * 2, bilinear=False)
            self.up4 = UpBlock(base_channels * 2, base_channels, bilinear=False)
            
            # Output
            self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        
        def forward(self, x: Tensor) -> Tensor:
            """Forward pass with attention."""
            # Encoder
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            # Decoder with attention
            x4_att = self.att1(x4, x5)
            x = self.up1(x5, x4_att)
            
            x3_att = self.att2(x3, x)
            x = self.up2(x, x3_att)
            
            x2_att = self.att3(x2, x)
            x = self.up3(x, x2_att)
            
            x1_att = self.att4(x1, x)
            x = self.up4(x, x1_att)
            
            # Output
            logits = self.outc(x)
            
            return logits
    
    
    class AttentionGate(nn.Module):
        """Attention gate for U-Net."""
        
        def __init__(
            self,
            gate_channels: int,
            input_channels: int,
            inter_channels: int
        ):
            super().__init__()
            
            self.W_g = nn.Sequential(
                nn.Conv2d(gate_channels, inter_channels, 1, bias=True),
                nn.BatchNorm2d(inter_channels)
            )
            
            self.W_x = nn.Sequential(
                nn.Conv2d(input_channels, inter_channels, 1, bias=True),
                nn.BatchNorm2d(inter_channels)
            )
            
            self.psi = nn.Sequential(
                nn.Conv2d(inter_channels, 1, 1, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
            
            self.relu = nn.ReLU(inplace=True)
        
        def forward(self, x: Tensor, g: Tensor) -> Tensor:
            """
            Forward pass.
            
            Args:
                x: Skip connection features (from encoder)
                g: Gating signal (from decoder)
            """
            # Resize gating signal if needed
            if g.size()[2:] != x.size()[2:]:
                g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=True)
            
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            
            return x * psi

else:
    # Placeholders when PyTorch not available
    FoodUNet = None
    AttentionUNet = None


# ============================================================================
# DEEPLAB V3+ ARCHITECTURE
# ============================================================================

if TORCH_AVAILABLE:
    
    class AtrousSeparableConv(nn.Module):
        """Atrous (dilated) separable convolution."""
        
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            dilation: int = 1
        ):
            super().__init__()
            
            padding = dilation * (kernel_size - 1) // 2
            
            self.depthwise = nn.Conv2d(
                in_channels, in_channels, kernel_size,
                padding=padding, dilation=dilation,
                groups=in_channels, bias=False
            )
            
            self.pointwise = nn.Conv2d(
                in_channels, out_channels, 1, bias=False
            )
            
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
        
        def forward(self, x: Tensor) -> Tensor:
            x = self.depthwise(x)
            x = self.pointwise(x)
            x = self.bn(x)
            x = self.relu(x)
            return x
    
    
    class ASPP(nn.Module):
        """Atrous Spatial Pyramid Pooling."""
        
        def __init__(
            self,
            in_channels: int,
            out_channels: int = 256,
            output_stride: int = 16
        ):
            super().__init__()
            
            # Atrous rates depend on output stride
            if output_stride == 16:
                dilations = [1, 6, 12, 18]
            elif output_stride == 8:
                dilations = [1, 12, 24, 36]
            else:
                raise ValueError(f"Unsupported output_stride: {output_stride}")
            
            # 1x1 convolution
            self.aspp1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
            # Atrous convolutions
            self.aspp2 = AtrousSeparableConv(in_channels, out_channels, 3, dilations[1])
            self.aspp3 = AtrousSeparableConv(in_channels, out_channels, 3, dilations[2])
            self.aspp4 = AtrousSeparableConv(in_channels, out_channels, 3, dilations[3])
            
            # Image pooling
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
            # Fusion
            self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x: Tensor) -> Tensor:
            x1 = self.aspp1(x)
            x2 = self.aspp2(x)
            x3 = self.aspp3(x)
            x4 = self.aspp4(x)
            
            x5 = self.global_avg_pool(x)
            x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat([x1, x2, x3, x4, x5], dim=1)
            
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.dropout(x)
            
            return x
    
    
    class DeepLabDecoder(nn.Module):
        """DeepLab v3+ decoder."""
        
        def __init__(
            self,
            low_level_channels: int,
            num_classes: int,
            aspp_channels: int = 256
        ):
            super().__init__()
            
            # Low-level feature projection
            self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(48)
            self.relu = nn.ReLU(inplace=True)
            
            # Fusion
            self.last_conv = nn.Sequential(
                nn.Conv2d(aspp_channels + 48, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(256, num_classes, 1)
            )
        
        def forward(self, x: Tensor, low_level_feat: Tensor) -> Tensor:
            # Process low-level features
            low_level_feat = self.conv1(low_level_feat)
            low_level_feat = self.bn1(low_level_feat)
            low_level_feat = self.relu(low_level_feat)
            
            # Upsample ASPP features
            x = F.interpolate(
                x,
                size=low_level_feat.shape[2:],
                mode='bilinear',
                align_corners=True
            )
            
            # Concatenate
            x = torch.cat([x, low_level_feat], dim=1)
            
            # Final convolutions
            x = self.last_conv(x)
            
            return x
    
    
    class FoodDeepLabV3Plus(nn.Module):
        """DeepLab v3+ for food segmentation."""
        
        def __init__(
            self,
            backbone: nn.Module,
            num_classes: int = 100,
            output_stride: int = 16,
            backbone_out_channels: Tuple[int, int] = (256, 2048)
        ):
            """
            Initialize DeepLab v3+.
            
            Args:
                backbone: Feature extraction backbone (e.g., ResNet)
                num_classes: Number of segmentation classes
                output_stride: 8 or 16
                backbone_out_channels: (low_level_channels, high_level_channels)
            """
            super().__init__()
            
            self.backbone = backbone
            self.num_classes = num_classes
            
            low_level_channels, high_level_channels = backbone_out_channels
            
            # ASPP module
            self.aspp = ASPP(high_level_channels, 256, output_stride)
            
            # Decoder
            self.decoder = DeepLabDecoder(low_level_channels, num_classes, 256)
        
        def forward(self, x: Tensor) -> Tensor:
            """Forward pass."""
            input_shape = x.shape[2:]
            
            # Extract features
            if hasattr(self.backbone, 'extract_features'):
                features = self.backbone.extract_features(x)
                low_level_feat = features[1]  # Early layer
                x = features[-1]  # Deep layer
            else:
                # Assume backbone returns list of features
                features = self.backbone(x)
                low_level_feat = features[1]
                x = features[-1]
            
            # ASPP
            x = self.aspp(x)
            
            # Decoder
            x = self.decoder(x, low_level_feat)
            
            # Upsample to input size
            x = F.interpolate(
                x,
                size=input_shape,
                mode='bilinear',
                align_corners=True
            )
            
            return x
    
    
    def create_food_deeplab(
        backbone_name: str = 'resnet50',
        num_classes: int = 100,
        output_stride: int = 16,
        pretrained_backbone: bool = False
    ) -> FoodDeepLabV3Plus:
        """
        Create DeepLab v3+ model.
        
        Args:
            backbone_name: Backbone architecture
            num_classes: Number of segmentation classes
            output_stride: 8 or 16
            pretrained_backbone: Use pretrained backbone
        
        Returns:
            FoodDeepLabV3Plus model
        """
        # Import backbone
        from ml_models_phase3d_part4_phase1 import create_food_resnet
        
        if backbone_name == 'resnet50':
            backbone = create_food_resnet('resnet50', num_classes=1000)
            # Remove final layers
            backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone_out_channels = (256, 2048)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        return FoodDeepLabV3Plus(
            backbone=backbone,
            num_classes=num_classes,
            output_stride=output_stride,
            backbone_out_channels=backbone_out_channels
        )

else:
    # Placeholders
    FoodDeepLabV3Plus = None
    create_food_deeplab = None


# ============================================================================
# PSPNet ARCHITECTURE
# ============================================================================

if TORCH_AVAILABLE:
    
    class PyramidPoolingModule(nn.Module):
        """Pyramid Pooling Module for PSPNet."""
        
        def __init__(
            self,
            in_channels: int,
            pool_sizes: Tuple[int, ...] = (1, 2, 3, 6)
        ):
            super().__init__()
            
            self.stages = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(in_channels, in_channels // len(pool_sizes), 1, bias=False),
                    nn.BatchNorm2d(in_channels // len(pool_sizes)),
                    nn.ReLU(inplace=True)
                )
                for pool_size in pool_sizes
            ])
            
            self.bottleneck = nn.Sequential(
                nn.Conv2d(
                    in_channels + (in_channels // len(pool_sizes)) * len(pool_sizes),
                    in_channels,
                    3,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            )
        
        def forward(self, x: Tensor) -> Tensor:
            h, w = x.size(2), x.size(3)
            
            # Multi-scale pooling
            pyramids = [x]
            for stage in self.stages:
                pooled = stage(x)
                pyramids.append(
                    F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=True)
                )
            
            # Concatenate
            x = torch.cat(pyramids, dim=1)
            
            # Fusion
            x = self.bottleneck(x)
            
            return x
    
    
    class FoodPSPNet(nn.Module):
        """PSPNet for food segmentation."""
        
        def __init__(
            self,
            backbone: nn.Module,
            num_classes: int = 100,
            pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
            backbone_out_channels: int = 2048
        ):
            """
            Initialize PSPNet.
            
            Args:
                backbone: Feature extraction backbone
                num_classes: Number of segmentation classes
                pool_sizes: Pyramid pooling sizes
                backbone_out_channels: Backbone output channels
            """
            super().__init__()
            
            self.backbone = backbone
            self.num_classes = num_classes
            
            # Pyramid Pooling Module
            self.ppm = PyramidPoolingModule(backbone_out_channels, pool_sizes)
            
            # Final classifier
            self.final = nn.Sequential(
                nn.Conv2d(backbone_out_channels, 512, 3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(512, num_classes, 1)
            )
            
            # Auxiliary classifier (for training)
            self.aux = nn.Sequential(
                nn.Conv2d(backbone_out_channels // 2, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(256, num_classes, 1)
            )
        
        def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
            """Forward pass."""
            input_shape = x.shape[2:]
            
            # Extract features
            if hasattr(self.backbone, 'extract_features'):
                features = self.backbone.extract_features(x)
                aux_feat = features[-2]  # Auxiliary features
                x = features[-1]  # Main features
            else:
                features = self.backbone(x)
                aux_feat = features[-2]
                x = features[-1]
            
            # Pyramid pooling
            x = self.ppm(x)
            
            # Main classifier
            x = self.final(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
            
            if self.training:
                # Auxiliary output for training
                aux_out = self.aux(aux_feat)
                aux_out = F.interpolate(
                    aux_out,
                    size=input_shape,
                    mode='bilinear',
                    align_corners=True
                )
                return x, aux_out
            else:
                return x
    
    
    def create_food_pspnet(
        backbone_name: str = 'resnet50',
        num_classes: int = 100,
        pretrained_backbone: bool = False
    ) -> FoodPSPNet:
        """
        Create PSPNet model.
        
        Args:
            backbone_name: Backbone architecture
            num_classes: Number of segmentation classes
            pretrained_backbone: Use pretrained backbone
        
        Returns:
            FoodPSPNet model
        """
        from ml_models_phase3d_part4_phase1 import create_food_resnet
        
        if backbone_name == 'resnet50':
            backbone = create_food_resnet('resnet50', num_classes=1000)
            backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone_out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        return FoodPSPNet(
            backbone=backbone,
            num_classes=num_classes,
            backbone_out_channels=backbone_out_channels
        )

else:
    # Placeholders
    FoodPSPNet = None
    create_food_pspnet = None


# ============================================================================
# SEGMENTATION UTILITIES
# ============================================================================

def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute IoU for each class.
    
    Args:
        pred: Predicted segmentation [H, W]
        target: Ground truth segmentation [H, W]
        num_classes: Number of classes
    
    Returns:
        IoU for each class
    """
    ious = np.zeros(num_classes)
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union > 0:
            ious[cls] = intersection / union
        else:
            ious[cls] = float('nan')
    
    return ious


def compute_dice(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute Dice coefficient for each class.
    
    Args:
        pred: Predicted segmentation [H, W]
        target: Ground truth segmentation [H, W]
        num_classes: Number of classes
    
    Returns:
        Dice coefficient for each class
    """
    dice_scores = np.zeros(num_classes)
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        total = pred_mask.sum() + target_mask.sum()
        
        if total > 0:
            dice_scores[cls] = 2 * intersection / total
        else:
            dice_scores[cls] = float('nan')
    
    return dice_scores


def compute_pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute overall pixel accuracy."""
    correct = (pred == target).sum()
    total = pred.size
    return correct / total


def refine_boundaries(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    iterations: int = 5
) -> np.ndarray:
    """
    Refine segmentation boundaries using morphological operations.
    
    Args:
        mask: Segmentation mask [H, W]
        image: Optional RGB image for guided filtering
        iterations: Number of refinement iterations
    
    Returns:
        Refined mask
    """
    if not CV2_AVAILABLE:
        return mask
    
    refined = mask.copy()
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Close small holes
    refined = cv2.morphologyEx(refined.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    # Remove small noise
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
    
    # Smooth boundaries
    if image is not None and len(image.shape) == 3:
        # Guided filter (edge-preserving smoothing)
        for _ in range(iterations):
            refined_float = refined.astype(np.float32)
            refined_float = cv2.bilateralFilter(refined_float, 9, 75, 75)
            refined = refined_float.astype(np.uint8)
    
    return refined


def estimate_volume_from_mask(
    mask: np.ndarray,
    depth_map: Optional[np.ndarray] = None,
    pixel_size_cm: float = 0.1,
    assumed_height_cm: float = 2.0
) -> Dict[str, float]:
    """
    Estimate volume from segmentation mask.
    
    Args:
        mask: Binary segmentation mask [H, W]
        depth_map: Optional depth map [H, W]
        pixel_size_cm: Physical size of each pixel
        assumed_height_cm: Assumed height if no depth map
    
    Returns:
        Volume estimation in cm³
    """
    area_pixels = np.sum(mask > 0)
    area_cm2 = area_pixels * (pixel_size_cm ** 2)
    
    if depth_map is not None:
        # Use actual depth
        masked_depth = depth_map[mask > 0]
        avg_height_cm = np.mean(masked_depth)
    else:
        # Use assumed height
        avg_height_cm = assumed_height_cm
    
    # Simplified volume (area × height)
    volume_cm3 = area_cm2 * avg_height_cm
    
    return {
        'area_cm2': float(area_cm2),
        'height_cm': float(avg_height_cm),
        'volume_cm3': float(volume_cm3),
        'volume_ml': float(volume_cm3)  # 1 cm³ = 1 ml
    }


def multi_class_volume_estimation(
    segmentation: SegmentationResult,
    depth_map: Optional[np.ndarray] = None,
    pixel_size_cm: float = 0.1,
    density_db: Optional[Dict[int, float]] = None
) -> Dict[int, Dict[str, float]]:
    """
    Estimate volume for multiple ingredient classes.
    
    Args:
        segmentation: Segmentation result
        depth_map: Optional depth map
        pixel_size_cm: Physical pixel size
        density_db: Density database (g/cm³) for weight estimation
    
    Returns:
        Volume estimates per class
    """
    results = {}
    
    for class_id in segmentation.class_ids:
        mask = segmentation.get_class_mask(class_id)
        
        volume_info = estimate_volume_from_mask(
            mask, depth_map, pixel_size_cm
        )
        
        # Add weight estimation if density available
        if density_db and class_id in density_db:
            density = density_db[class_id]
            volume_info['weight_g'] = volume_info['volume_cm3'] * density
        
        # Add class name
        if segmentation.class_names:
            volume_info['class_name'] = segmentation.class_names[class_id]
        
        results[class_id] = volume_info
    
    return results


# ============================================================================
# HIGH-LEVEL SEGMENTATION INTERFACE
# ============================================================================

class FoodSegmentationModel:
    """High-level interface for food segmentation."""
    
    def __init__(
        self,
        model_type: str = 'unet',
        num_classes: int = 100,
        class_names: Optional[List[str]] = None,
        device: str = 'cuda' if TORCH_AVAILABLE else 'cpu'
    ):
        """
        Initialize segmentation model.
        
        Args:
            model_type: 'unet', 'attention_unet', 'deeplab', or 'pspnet'
            num_classes: Number of classes
            class_names: List of class names
            device: Device to run on
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for segmentation")
        
        self.model_type = model_type
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Create model
        if model_type == 'unet':
            self.model = FoodUNet(num_classes=num_classes)
        elif model_type == 'attention_unet':
            self.model = AttentionUNet(num_classes=num_classes)
        elif model_type == 'deeplab':
            self.model = create_food_deeplab(num_classes=num_classes)
        elif model_type == 'pspnet':
            self.model = create_food_pspnet(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def segment(
        self,
        image: np.ndarray,
        refine_boundaries: bool = True
    ) -> SegmentationResult:
        """
        Segment an image.
        
        Args:
            image: Input image [H, W, 3] in RGB
            refine_boundaries: Apply boundary refinement
        
        Returns:
            Segmentation result
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        # Preprocess
        image_tensor = self._preprocess(image)
        
        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            
            if isinstance(logits, tuple):
                logits = logits[0]  # Main output for PSPNet
            
            # Get predictions
            probs = F.softmax(logits, dim=1)
            pred_mask = torch.argmax(probs, dim=1)
            
            # Get confidence
            confidence, _ = torch.max(probs, dim=1)
            
            # Convert to numpy
            pred_mask = pred_mask[0].cpu().numpy().astype(np.uint8)
            confidence = confidence[0].cpu().numpy()
        
        # Resize to original size
        if pred_mask.shape != image.shape[:2]:
            if CV2_AVAILABLE:
                pred_mask = cv2.resize(
                    pred_mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                confidence = cv2.resize(
                    confidence,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
        
        # Refine boundaries
        if refine_boundaries:
            pred_mask = refine_boundaries(pred_mask, image)
        
        # Get unique classes
        class_ids = np.unique(pred_mask)
        class_ids = class_ids[class_ids > 0]  # Exclude background
        
        return SegmentationResult(
            mask=pred_mask,
            class_ids=class_ids,
            class_names=self.class_names,
            confidence=confidence
        )
    
    def _preprocess(self, image: np.ndarray) -> Tensor:
        """Preprocess image for model."""
        # Resize to model input size
        target_size = 512
        h, w = image.shape[:2]
        
        if CV2_AVAILABLE:
            image = cv2.resize(image, (target_size, target_size))
        elif PIL_AVAILABLE:
            image = Image.fromarray(image)
            image = image.resize((target_size, target_size), Image.BILINEAR)
            image = np.array(image)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor [B, C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device)
        
        return image


# ============================================================================
# TESTING
# ============================================================================

def test_segmentation_models():
    """Test segmentation models."""
    print("=" * 80)
    print("TESTING SEGMENTATION MODELS - PART 4 PHASE 4")
    print("=" * 80)
    
    if not TORCH_AVAILABLE:
        print("\n⚠️  PyTorch not available. Skipping tests.")
        return
    
    # Test U-Net
    print("\n" + "=" * 80)
    print("1. Testing U-Net Architecture")
    print("=" * 80)
    
    try:
        unet = FoodUNet(in_channels=3, num_classes=100, base_channels=64)
        
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())
        
        print(f"\nU-Net parameters: {count_parameters(unet):,}")
        
        # Test forward
        x = torch.randn(2, 3, 512, 512)
        output = unet(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output format: [batch, num_classes, H, W]")
        
        # Test feature extraction
        features = unet.extract_features(x)
        print(f"\nMulti-scale features:")
        for i, feat in enumerate(features):
            print(f"  Level {i}: {feat.shape}")
        
        print("✅ U-Net test passed!")
        
    except Exception as e:
        print(f"❌ U-Net test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Attention U-Net
    print("\n" + "=" * 80)
    print("2. Testing Attention U-Net")
    print("=" * 80)
    
    try:
        att_unet = AttentionUNet(in_channels=3, num_classes=100)
        print(f"\nAttention U-Net parameters: {count_parameters(att_unet):,}")
        
        x = torch.randn(2, 3, 512, 512)
        output = att_unet(x)
        
        print(f"Output shape: {output.shape}")
        print("✅ Attention U-Net test passed!")
        
    except Exception as e:
        print(f"❌ Attention U-Net test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test DeepLab components
    print("\n" + "=" * 80)
    print("3. Testing DeepLab v3+ Components")
    print("=" * 80)
    
    try:
        # Test ASPP
        aspp = ASPP(in_channels=2048, out_channels=256)
        x = torch.randn(2, 2048, 32, 32)
        output = aspp(x)
        print(f"ASPP output: {output.shape}")
        
        print("✅ DeepLab components test passed!")
        
    except Exception as e:
        print(f"❌ DeepLab test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test PSPNet
    print("\n" + "=" * 80)
    print("4. Testing PSPNet")
    print("=" * 80)
    
    try:
        # Test PPM
        ppm = PyramidPoolingModule(in_channels=2048)
        x = torch.randn(2, 2048, 32, 32)
        output = ppm(x)
        print(f"PPM output: {output.shape}")
        
        print("✅ PSPNet test passed!")
        
    except Exception as e:
        print(f"❌ PSPNet test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test utilities
    print("\n" + "=" * 80)
    print("5. Testing Segmentation Utilities")
    print("=" * 80)
    
    try:
        # Test IoU
        pred = np.random.randint(0, 5, (256, 256))
        target = np.random.randint(0, 5, (256, 256))
        
        ious = compute_iou(pred, target, num_classes=5)
        print(f"\nIoU per class: {ious}")
        print(f"Mean IoU: {np.nanmean(ious):.4f}")
        
        # Test Dice
        dice = compute_dice(pred, target, num_classes=5)
        print(f"Dice per class: {dice}")
        print(f"Mean Dice: {np.nanmean(dice):.4f}")
        
        # Test pixel accuracy
        acc = compute_pixel_accuracy(pred, target)
        print(f"Pixel accuracy: {acc:.4f}")
        
        # Test volume estimation
        mask = np.random.randint(0, 2, (256, 256))
        volume_info = estimate_volume_from_mask(mask, pixel_size_cm=0.1)
        print(f"\nVolume estimation:")
        print(f"  Area: {volume_info['area_cm2']:.2f} cm²")
        print(f"  Volume: {volume_info['volume_cm3']:.2f} cm³")
        print(f"  Volume: {volume_info['volume_ml']:.2f} ml")
        
        print("✅ Utilities test passed!")
        
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test high-level interface
    print("\n" + "=" * 80)
    print("6. Testing High-Level Interface")
    print("=" * 80)
    
    try:
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print("\nCreating U-Net segmentation model...")
        segmenter = FoodSegmentationModel(
            model_type='unet',
            num_classes=100,
            class_names=[f'ingredient_{i}' for i in range(100)]
        )
        
        print(f"Model type: {segmenter.model_type}")
        print(f"Device: {segmenter.device}")
        print(f"Number of classes: {segmenter.num_classes}")
        
        print("\nRunning segmentation...")
        result = segmenter.segment(dummy_image)
        
        print(f"\nSegmentation result:")
        print(f"  Mask shape: {result.mask.shape}")
        print(f"  Number of classes detected: {len(result.class_ids)}")
        print(f"  Class IDs: {result.class_ids[:10]}...")
        
        # Test area computation
        areas = result.compute_areas(pixel_size_cm2=0.01)
        print(f"\nClass areas (first 5):")
        for class_id in list(areas.keys())[:5]:
            print(f"  Class {class_id}: {areas[class_id]:.2f} cm²")
        
        print("✅ High-level interface test passed!")
        
    except Exception as e:
        print(f"❌ Interface test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("\n✅ All segmentation tests completed!")
    print("\nImplemented architectures:")
    print("  • U-Net with skip connections")
    print("  • Attention U-Net with attention gates")
    print("  • DeepLab v3+ with ASPP")
    print("  • PSPNet with pyramid pooling")
    print("  • Boundary refinement utilities")
    print("  • Volume estimation from masks")
    print("  • High-level segmentation interface")
    
    print("\nNext steps:")
    print("  1. Add depth estimation (Phase 5)")
    print("  2. Optimize for mobile deployment (Phase 6)")
    print("  3. Train on food segmentation datasets")
    print("  4. Integrate with object detection pipeline")


if __name__ == '__main__':
    test_segmentation_models()
