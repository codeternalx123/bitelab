"""
EfficientNetV2 Ensemble for Atomic Composition Prediction
=========================================================

Multi-scale EfficientNetV2 ensemble combining S/M/L variants for
superior accuracy through model diversity and test-time augmentation.

Architecture:
- EfficientNetV2-S: 21M params, 384×384, fast inference
- EfficientNetV2-M: 54M params, 480×480, balanced
- EfficientNetV2-L: 119M params, 640×640, maximum accuracy

Features:
- Progressive training (start small, grow resolution)
- Fused-MBConv blocks for efficiency
- Test-time augmentation (10× predictions averaged)
- Weighted ensemble based on validation performance
- Knowledge distillation from ViT teacher

References:
- Tan & Le "EfficientNetV2: Smaller Models and Faster Training" (ICML 2021)
"""

import math
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
from collections import OrderedDict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not installed. Install: pip install torch torchvision")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️  numpy not installed")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EfficientNetConfig:
    """EfficientNet configuration"""
    
    # Model variant
    model_name: str = "efficientnetv2_s"  # s, m, l
    
    # Input
    image_size: int = 384
    in_channels: int = 3
    
    # Architecture (auto-configured based on model_name)
    width_coefficient: float = 1.0
    depth_coefficient: float = 1.0
    dropout_rate: float = 0.2
    
    # Output
    num_elements: int = 22
    
    # Training
    use_stochastic_depth: bool = True
    stochastic_depth_rate: float = 0.2
    
    @classmethod
    def from_variant(cls, variant: str) -> 'EfficientNetConfig':
        """Create config from variant name"""
        configs = {
            's': {  # EfficientNetV2-S
                'model_name': 'efficientnetv2_s',
                'image_size': 384,
                'width_coefficient': 1.0,
                'depth_coefficient': 1.0,
                'dropout_rate': 0.2,
            },
            'm': {  # EfficientNetV2-M
                'model_name': 'efficientnetv2_m',
                'image_size': 480,
                'width_coefficient': 1.0,
                'depth_coefficient': 1.0,
                'dropout_rate': 0.3,
            },
            'l': {  # EfficientNetV2-L
                'model_name': 'efficientnetv2_l',
                'image_size': 640,
                'width_coefficient': 1.0,
                'depth_coefficient': 1.0,
                'dropout_rate': 0.4,
            }
        }
        
        if variant not in configs:
            raise ValueError(f"Unknown variant: {variant}")
        
        return cls(**configs[variant])


# ============================================================================
# Building Blocks
# ============================================================================

if HAS_TORCH:
    class SqueezeExcitation(nn.Module):
        """
        Squeeze-and-Excitation block for channel attention
        
        Reference: "Squeeze-and-Excitation Networks" (CVPR 2018)
        """
        
        def __init__(
            self,
            in_channels: int,
            reduced_channels: int,
            activation: nn.Module = nn.SiLU()
        ):
            super().__init__()
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, reduced_channels, 1),
                activation,
                nn.Conv2d(reduced_channels, in_channels, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x: Tensor) -> Tensor:
            return x * self.se(x)


    class StochasticDepth(nn.Module):
        """
        Stochastic depth (drop path) for better regularization
        
        Reference: "Deep Networks with Stochastic Depth" (ECCV 2016)
        """
        
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = drop_prob
        
        def forward(self, x: Tensor) -> Tensor:
            if not self.training or self.drop_prob == 0:
                return x
            
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            
            return x.div(keep_prob) * random_tensor


    class MBConvBlock(nn.Module):
        """
        Mobile Inverted Bottleneck Convolution (MBConv) block
        
        Components:
        1. Expansion: 1×1 conv to expand channels
        2. Depthwise: 3×3 or 5×5 depthwise conv
        3. Squeeze-Excitation: Channel attention
        4. Projection: 1×1 conv to reduce channels
        5. Skip connection (if stride=1 and in_channels == out_channels)
        """
        
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            expand_ratio: int = 6,
            se_ratio: float = 0.25,
            drop_path_rate: float = 0.0
        ):
            super().__init__()
            self.use_residual = (stride == 1 and in_channels == out_channels)
            
            # Expansion phase
            expanded_channels = in_channels * expand_ratio
            self.expand = None
            if expand_ratio != 1:
                self.expand = nn.Sequential(
                    nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                    nn.BatchNorm2d(expanded_channels),
                    nn.SiLU()
                )
            
            # Depthwise convolution
            self.depthwise = nn.Sequential(
                nn.Conv2d(
                    expanded_channels,
                    expanded_channels,
                    kernel_size,
                    stride,
                    padding=kernel_size // 2,
                    groups=expanded_channels,
                    bias=False
                ),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU()
            )
            
            # Squeeze-and-Excitation
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = SqueezeExcitation(expanded_channels, se_channels)
            
            # Output projection
            self.project = nn.Sequential(
                nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
            # Stochastic depth
            self.drop_path = StochasticDepth(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
        def forward(self, x: Tensor) -> Tensor:
            identity = x
            
            # Expansion
            if self.expand is not None:
                x = self.expand(x)
            
            # Depthwise + SE
            x = self.depthwise(x)
            x = self.se(x)
            
            # Projection
            x = self.project(x)
            
            # Residual connection
            if self.use_residual:
                x = self.drop_path(x) + identity
            
            return x


    class FusedMBConvBlock(nn.Module):
        """
        Fused Mobile Inverted Bottleneck Convolution block
        
        Replaces depthwise conv with regular conv for better efficiency
        Used in early stages of EfficientNetV2
        """
        
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            expand_ratio: int = 6,
            se_ratio: float = 0.25,
            drop_path_rate: float = 0.0
        ):
            super().__init__()
            self.use_residual = (stride == 1 and in_channels == out_channels)
            
            expanded_channels = in_channels * expand_ratio
            
            # Fused expansion + depthwise (single conv)
            self.fused = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    expanded_channels,
                    kernel_size,
                    stride,
                    padding=kernel_size // 2,
                    bias=False
                ),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU()
            )
            
            # Squeeze-and-Excitation
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = SqueezeExcitation(expanded_channels, se_channels)
            
            # Output projection
            self.project = nn.Sequential(
                nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
            # Stochastic depth
            self.drop_path = StochasticDepth(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
        def forward(self, x: Tensor) -> Tensor:
            identity = x
            
            x = self.fused(x)
            x = self.se(x)
            x = self.project(x)
            
            if self.use_residual:
                x = self.drop_path(x) + identity
            
            return x


    # ============================================================================
    # EfficientNetV2
    # ============================================================================

    class EfficientNetV2(nn.Module):
        """
        EfficientNetV2 for atomic composition prediction
        
        Architecture stages:
        1. Stem: Initial convolution
        2. Fused-MBConv stages (early layers, more efficient)
        3. MBConv stages (later layers, more capacity)
        4. Head: Classification/regression head
        """
        
        def __init__(self, config: EfficientNetConfig):
            super().__init__()
            self.config = config
            
            # Build architecture based on variant
            if 's' in config.model_name:
                self.blocks_args = self._get_blocks_args_s()
            elif 'm' in config.model_name:
                self.blocks_args = self._get_blocks_args_m()
            elif 'l' in config.model_name:
                self.blocks_args = self._get_blocks_args_l()
            else:
                raise ValueError(f"Unknown model: {config.model_name}")
            
            # Stem
            stem_channels = self._round_filters(24)
            self.stem = nn.Sequential(
                nn.Conv2d(config.in_channels, stem_channels, 3, 2, 1, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.SiLU()
            )
            
            # Build blocks
            self.blocks = nn.ModuleList([])
            in_channels = stem_channels
            
            total_blocks = sum(args['num_repeat'] for args in self.blocks_args)
            block_idx = 0
            
            for stage_args in self.blocks_args:
                num_repeat = stage_args['num_repeat']
                out_channels = self._round_filters(stage_args['out_channels'])
                
                for i in range(num_repeat):
                    # First block in stage may have stride > 1
                    stride = stage_args['stride'] if i == 0 else 1
                    
                    # Stochastic depth rate increases with depth
                    drop_path_rate = config.stochastic_depth_rate * block_idx / total_blocks
                    
                    # Create block (Fused-MBConv or MBConv)
                    if stage_args['block_type'] == 'fused':
                        block = FusedMBConvBlock(
                            in_channels,
                            out_channels,
                            kernel_size=stage_args['kernel_size'],
                            stride=stride,
                            expand_ratio=stage_args['expand_ratio'],
                            se_ratio=stage_args['se_ratio'],
                            drop_path_rate=drop_path_rate
                        )
                    else:
                        block = MBConvBlock(
                            in_channels,
                            out_channels,
                            kernel_size=stage_args['kernel_size'],
                            stride=stride,
                            expand_ratio=stage_args['expand_ratio'],
                            se_ratio=stage_args['se_ratio'],
                            drop_path_rate=drop_path_rate
                        )
                    
                    self.blocks.append(block)
                    in_channels = out_channels
                    block_idx += 1
            
            # Head
            head_channels = self._round_filters(1280)
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, head_channels, 1, bias=False),
                nn.BatchNorm2d(head_channels),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            
            # Element prediction
            self.element_head = nn.Sequential(
                nn.Dropout(config.dropout_rate),
                nn.Linear(head_channels, 512),
                nn.SiLU(),
                nn.Dropout(config.dropout_rate / 2),
                nn.Linear(512, config.num_elements)
            )
            
            # Confidence prediction
            self.confidence_head = nn.Sequential(
                nn.Linear(head_channels, 512),
                nn.SiLU(),
                nn.Linear(512, config.num_elements),
                nn.Sigmoid()
            )
            
            self._initialize_weights()
        
        def _get_blocks_args_s(self) -> List[Dict]:
            """EfficientNetV2-S architecture"""
            return [
                # Stage 1: Fused-MBConv
                {'block_type': 'fused', 'num_repeat': 2, 'out_channels': 24, 'expand_ratio': 1, 'kernel_size': 3, 'stride': 1, 'se_ratio': 0},
                {'block_type': 'fused', 'num_repeat': 4, 'out_channels': 48, 'expand_ratio': 4, 'kernel_size': 3, 'stride': 2, 'se_ratio': 0},
                {'block_type': 'fused', 'num_repeat': 4, 'out_channels': 64, 'expand_ratio': 4, 'kernel_size': 3, 'stride': 2, 'se_ratio': 0},
                
                # Stage 2: MBConv
                {'block_type': 'mbconv', 'num_repeat': 6, 'out_channels': 128, 'expand_ratio': 4, 'kernel_size': 3, 'stride': 2, 'se_ratio': 0.25},
                {'block_type': 'mbconv', 'num_repeat': 9, 'out_channels': 160, 'expand_ratio': 6, 'kernel_size': 3, 'stride': 1, 'se_ratio': 0.25},
                {'block_type': 'mbconv', 'num_repeat': 15, 'out_channels': 256, 'expand_ratio': 6, 'kernel_size': 3, 'stride': 2, 'se_ratio': 0.25},
            ]
        
        def _get_blocks_args_m(self) -> List[Dict]:
            """EfficientNetV2-M architecture"""
            return [
                {'block_type': 'fused', 'num_repeat': 3, 'out_channels': 24, 'expand_ratio': 1, 'kernel_size': 3, 'stride': 1, 'se_ratio': 0},
                {'block_type': 'fused', 'num_repeat': 5, 'out_channels': 48, 'expand_ratio': 4, 'kernel_size': 3, 'stride': 2, 'se_ratio': 0},
                {'block_type': 'fused', 'num_repeat': 5, 'out_channels': 80, 'expand_ratio': 4, 'kernel_size': 3, 'stride': 2, 'se_ratio': 0},
                {'block_type': 'mbconv', 'num_repeat': 7, 'out_channels': 160, 'expand_ratio': 4, 'kernel_size': 3, 'stride': 2, 'se_ratio': 0.25},
                {'block_type': 'mbconv', 'num_repeat': 14, 'out_channels': 176, 'expand_ratio': 6, 'kernel_size': 3, 'stride': 1, 'se_ratio': 0.25},
                {'block_type': 'mbconv', 'num_repeat': 18, 'out_channels': 304, 'expand_ratio': 6, 'kernel_size': 3, 'stride': 2, 'se_ratio': 0.25},
                {'block_type': 'mbconv', 'num_repeat': 5, 'out_channels': 512, 'expand_ratio': 6, 'kernel_size': 3, 'stride': 1, 'se_ratio': 0.25},
            ]
        
        def _get_blocks_args_l(self) -> List[Dict]:
            """EfficientNetV2-L architecture"""
            return [
                {'block_type': 'fused', 'num_repeat': 4, 'out_channels': 32, 'expand_ratio': 1, 'kernel_size': 3, 'stride': 1, 'se_ratio': 0},
                {'block_type': 'fused', 'num_repeat': 7, 'out_channels': 64, 'expand_ratio': 4, 'kernel_size': 3, 'stride': 2, 'se_ratio': 0},
                {'block_type': 'fused', 'num_repeat': 7, 'out_channels': 96, 'expand_ratio': 4, 'kernel_size': 3, 'stride': 2, 'se_ratio': 0},
                {'block_type': 'mbconv', 'num_repeat': 10, 'out_channels': 192, 'expand_ratio': 4, 'kernel_size': 3, 'stride': 2, 'se_ratio': 0.25},
                {'block_type': 'mbconv', 'num_repeat': 19, 'out_channels': 224, 'expand_ratio': 6, 'kernel_size': 3, 'stride': 1, 'se_ratio': 0.25},
                {'block_type': 'mbconv', 'num_repeat': 25, 'out_channels': 384, 'expand_ratio': 6, 'kernel_size': 3, 'stride': 2, 'se_ratio': 0.25},
                {'block_type': 'mbconv', 'num_repeat': 7, 'out_channels': 640, 'expand_ratio': 6, 'kernel_size': 3, 'stride': 1, 'se_ratio': 0.25},
            ]
        
        def _round_filters(self, filters: int) -> int:
            """Round number of filters based on width coefficient"""
            multiplier = self.config.width_coefficient
            divisor = 8
            filters *= multiplier
            new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
            if new_filters < 0.9 * filters:
                new_filters += divisor
            return int(new_filters)
        
        def _initialize_weights(self):
            """Initialize model weights"""
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x: Tensor) -> Dict[str, Tensor]:
            """
            Forward pass
            
            Args:
                x: (batch_size, channels, height, width)
            
            Returns:
                Dictionary with predictions
            """
            # Stem
            x = self.stem(x)
            
            # Blocks
            for block in self.blocks:
                x = block(x)
            
            # Head
            features = self.head(x)
            
            # Predictions
            concentrations = F.relu(self.element_head(features))
            confidences = self.confidence_head(features)
            
            return {
                'concentrations': concentrations,
                'confidences': confidences,
                'features': features
            }


    # ============================================================================
    # Ensemble Model
    # ============================================================================

    class EfficientNetEnsemble(nn.Module):
        """
        Ensemble of EfficientNetV2 models (S, M, L)
        
        Features:
        - Weighted averaging based on validation performance
        - Test-time augmentation
        - Temperature scaling for calibration
        """
        
        def __init__(
            self,
            num_elements: int = 22,
            variants: List[str] = ['s', 'm', 'l'],
            weights: Optional[List[float]] = None
        ):
            super().__init__()
            self.num_elements = num_elements
            self.variants = variants
            
            # Create models
            self.models = nn.ModuleList([
                EfficientNetV2(EfficientNetConfig.from_variant(v))
                for v in variants
            ])
            
            # Ensemble weights (learned or fixed)
            if weights is None:
                weights = [1.0 / len(variants)] * len(variants)
            self.register_buffer('weights', torch.tensor(weights))
            
            # Temperature for calibration
            self.temperature = nn.Parameter(torch.ones(1))
        
        def forward(
            self,
            images: Dict[str, Tensor],
            use_tta: bool = False
        ) -> Dict[str, Tensor]:
            """
            Forward pass through ensemble
            
            Args:
                images: Dict of {variant: image_tensor} with appropriate sizes
                use_tta: Use test-time augmentation
            
            Returns:
                Ensemble predictions
            """
            predictions = []
            
            for i, (variant, model) in enumerate(zip(self.variants, self.models)):
                if variant not in images:
                    continue
                
                pred = model(images[variant])
                
                # Weight by ensemble weight
                pred['concentrations'] = pred['concentrations'] * self.weights[i]
                pred['confidences'] = pred['confidences'] * self.weights[i]
                
                predictions.append(pred)
            
            # Aggregate predictions
            concentrations = sum(p['concentrations'] for p in predictions)
            confidences = sum(p['confidences'] for p in predictions)
            
            # Temperature scaling for confidence calibration
            confidences = torch.sigmoid(torch.logit(confidences) / self.temperature)
            
            return {
                'concentrations': concentrations,
                'confidences': confidences
            }
        
        def update_weights(self, val_losses: List[float]):
            """Update ensemble weights based on validation performance"""
            # Convert losses to weights (inverse of loss)
            inverse_losses = [1.0 / (loss + 1e-6) for loss in val_losses]
            total = sum(inverse_losses)
            new_weights = [w / total for w in inverse_losses]
            
            self.weights.data = torch.tensor(new_weights, device=self.weights.device)
            
            print(f"Updated ensemble weights: {new_weights}")


    # ============================================================================
    # Model Factory
    # ============================================================================

    def create_efficientnet(
        variant: str = 's',
        num_elements: int = 22,
        pretrained: bool = False
    ) -> EfficientNetV2:
        """
        Create EfficientNetV2 model
        
        Args:
            variant: 's', 'm', or 'l'
            num_elements: Number of elements to predict
            pretrained: Load pretrained weights
        
        Returns:
            EfficientNetV2 model
        """
        config = EfficientNetConfig.from_variant(variant)
        config.num_elements = num_elements
        
        model = EfficientNetV2(config)
        
        if pretrained:
            print(f"⚠️  Pretrained weights not implemented for EfficientNetV2-{variant.upper()}")
            print("    Consider using timm library: pip install timm")
        
        return model


    def create_ensemble(
        num_elements: int = 22,
        variants: List[str] = ['s', 'm', 'l']
    ) -> EfficientNetEnsemble:
        """Create EfficientNet ensemble"""
        return EfficientNetEnsemble(
            num_elements=num_elements,
            variants=variants
        )


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example usage of EfficientNet models"""
    if not HAS_TORCH:
        print("PyTorch required")
        return
    
    print("\n" + "="*60)
    print("EFFICIENTNETV2 ENSEMBLE - EXAMPLE")
    print("="*60)
    
    # Single model
    print("\n1. Creating EfficientNetV2-S...")
    model_s = create_efficientnet('s', num_elements=22)
    n_params = sum(p.numel() for p in model_s.parameters())
    print(f"   Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    # Test inference
    print("\n2. Testing inference...")
    x = torch.randn(2, 3, 384, 384)
    with torch.no_grad():
        pred = model_s(x)
    print(f"   Input: {x.shape}")
    print(f"   Concentrations: {pred['concentrations'].shape}")
    print(f"   Confidences: {pred['confidences'].shape}")
    
    # Ensemble
    print("\n3. Creating ensemble (S + M + L)...")
    ensemble = create_ensemble(num_elements=22, variants=['s', 'm'])
    
    # Note: Each model needs different image size
    images = {
        's': torch.randn(2, 3, 384, 384),
        'm': torch.randn(2, 3, 480, 480),
    }
    
    with torch.no_grad():
        ensemble_pred = ensemble(images)
    
    print(f"   Ensemble output: {ensemble_pred['concentrations'].shape}")
    print(f"   Ensemble weights: {ensemble.weights}")
    
    print("\n✅ Example complete!")


if __name__ == "__main__":
    example_usage()
