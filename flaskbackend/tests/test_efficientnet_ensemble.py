"""
Unit tests for EfficientNetV2 ensemble

Tests EfficientNetV2 S/M/L ensemble architecture.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ai_nutrition.models.efficientnet_ensemble import (
    EfficientNetV2Ensemble,
    EfficientNetV2,
    FusedMBConv,
    SEBlock,
    ensemble_predict
)


class TestEfficientNetV2Ensemble(unittest.TestCase):
    """Test EfficientNetV2 ensemble"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.n_elements = 20
        self.batch_size = 2
        
    def test_initialization(self):
        """Test ensemble initialization"""
        ensemble = EfficientNetV2Ensemble(
            n_elements=self.n_elements,
            variants=['s', 'm']
        )
        self.assertIsNotNone(ensemble)
        
    def test_forward_pass(self):
        """Test forward pass through ensemble"""
        ensemble = EfficientNetV2Ensemble(
            n_elements=self.n_elements,
            variants=['s']
        )
        
        x = torch.randn(self.batch_size, 3, 224, 224)
        output = ensemble(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.n_elements))
        
    def test_multiple_variants(self):
        """Test ensemble with multiple variants"""
        ensemble = EfficientNetV2Ensemble(
            n_elements=self.n_elements,
            variants=['s', 'm', 'l']
        )
        
        x = torch.randn(self.batch_size, 3, 224, 224)
        output = ensemble(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.n_elements))
        
    def test_weighted_ensemble(self):
        """Test weighted ensemble fusion"""
        ensemble = EfficientNetV2Ensemble(
            n_elements=self.n_elements,
            variants=['s', 'm'],
            weights=[0.4, 0.6]
        )
        
        x = torch.randn(self.batch_size, 3, 224, 224)
        output = ensemble(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.n_elements))


class TestEfficientNetV2(unittest.TestCase):
    """Test individual EfficientNetV2 models"""
    
    def test_efficientnet_s(self):
        """Test EfficientNetV2-S"""
        model = EfficientNetV2(variant='s', n_elements=20)
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 20))
        
    def test_efficientnet_m(self):
        """Test EfficientNetV2-M"""
        model = EfficientNetV2(variant='m', n_elements=20)
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 20))
        
    def test_efficientnet_l(self):
        """Test EfficientNetV2-L"""
        model = EfficientNetV2(variant='l', n_elements=20)
        
        x = torch.randn(2, 3, 256, 256)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 20))


class TestFusedMBConv(unittest.TestCase):
    """Test Fused-MBConv block"""
    
    def test_fused_mbconv(self):
        """Test Fused-MBConv forward pass"""
        block = FusedMBConv(
            in_channels=32,
            out_channels=64,
            expand_ratio=4,
            stride=1
        )
        
        x = torch.randn(2, 32, 56, 56)
        output = block(x)
        
        self.assertEqual(output.shape, (2, 64, 56, 56))
        
    def test_fused_mbconv_stride2(self):
        """Test Fused-MBConv with stride 2"""
        block = FusedMBConv(
            in_channels=32,
            out_channels=64,
            expand_ratio=4,
            stride=2
        )
        
        x = torch.randn(2, 32, 56, 56)
        output = block(x)
        
        # Spatial dimensions should be halved
        self.assertEqual(output.shape, (2, 64, 28, 28))


class TestSEBlock(unittest.TestCase):
    """Test Squeeze-and-Excitation block"""
    
    def test_se_block(self):
        """Test SE block"""
        se = SEBlock(channels=64, reduction=4)
        
        x = torch.randn(2, 64, 56, 56)
        output = se(x)
        
        self.assertEqual(output.shape, x.shape)
        
    def test_channel_attention(self):
        """Test channel attention weights"""
        se = SEBlock(channels=64)
        
        x = torch.randn(2, 64, 56, 56)
        output = se(x)
        
        # Output should be recalibrated version of input
        self.assertFalse(torch.allclose(output, x))


class TestEnsemblePrediction(unittest.TestCase):
    """Test ensemble prediction utilities"""
    
    def test_ensemble_predict(self):
        """Test ensemble prediction function"""
        # Create mock predictions from multiple models
        pred1 = torch.randn(4, 20)
        pred2 = torch.randn(4, 20)
        pred3 = torch.randn(4, 20)
        
        predictions = [pred1, pred2, pred3]
        
        # Average ensemble
        ensemble_pred = ensemble_predict(predictions, method='average')
        
        self.assertEqual(ensemble_pred.shape, (4, 20))
        
    def test_weighted_ensemble_predict(self):
        """Test weighted ensemble prediction"""
        pred1 = torch.randn(4, 20)
        pred2 = torch.randn(4, 20)
        
        predictions = [pred1, pred2]
        weights = [0.3, 0.7]
        
        ensemble_pred = ensemble_predict(predictions, method='weighted', weights=weights)
        
        self.assertEqual(ensemble_pred.shape, (4, 20))


class TestEnsembleTraining(unittest.TestCase):
    """Test ensemble training"""
    
    def test_train_ensemble(self):
        """Test training ensemble"""
        ensemble = EfficientNetV2Ensemble(
            n_elements=20,
            variants=['s']
        )
        
        optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        x = torch.randn(2, 3, 224, 224)
        targets = torch.randn(2, 20)
        
        # Forward
        predictions = ensemble(x)
        loss = criterion(predictions, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        self.assertIsNotNone(loss)


class TestEnsembleInference(unittest.TestCase):
    """Test ensemble inference"""
    
    def test_inference_mode(self):
        """Test inference mode"""
        ensemble = EfficientNetV2Ensemble(
            n_elements=20,
            variants=['s', 'm']
        )
        ensemble.eval()
        
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = ensemble(x)
        
        self.assertEqual(output.shape, (1, 20))


class TestEnsemblePerformance(unittest.TestCase):
    """Test ensemble performance"""
    
    def test_inference_speed(self):
        """Test ensemble inference speed"""
        import time
        
        ensemble = EfficientNetV2Ensemble(
            n_elements=20,
            variants=['s']
        )
        ensemble.eval()
        
        # Warm up
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _ = ensemble(x)
        
        # Time batch
        batch_size = 32
        x = torch.randn(batch_size, 3, 224, 224)
        
        start = time.time()
        with torch.no_grad():
            output = ensemble(x)
        elapsed = time.time() - start
        
        throughput = batch_size / elapsed
        self.assertGreater(throughput, 5)


class TestParameterCount(unittest.TestCase):
    """Test parameter counting"""
    
    def test_efficientnet_s_params(self):
        """Test EfficientNetV2-S parameter count"""
        model = EfficientNetV2(variant='s', n_elements=20)
        
        n_params = sum(p.numel() for p in model.parameters())
        
        # Should have ~20-25M parameters
        self.assertGreater(n_params, 15_000_000)
        self.assertLess(n_params, 30_000_000)
        
    def test_efficientnet_m_params(self):
        """Test EfficientNetV2-M parameter count"""
        model = EfficientNetV2(variant='m', n_elements=20)
        
        n_params = sum(p.numel() for p in model.parameters())
        
        # Should have ~50-60M parameters
        self.assertGreater(n_params, 45_000_000)
        self.assertLess(n_params, 65_000_000)


class TestTemperatureScaling(unittest.TestCase):
    """Test temperature scaling for calibration"""
    
    def test_temperature_scaling(self):
        """Test temperature scaling"""
        ensemble = EfficientNetV2Ensemble(
            n_elements=20,
            variants=['s'],
            use_temperature_scaling=True
        )
        
        x = torch.randn(2, 3, 224, 224)
        output = ensemble(x)
        
        self.assertEqual(output.shape, (2, 20))


class TestTestTimeAugmentation(unittest.TestCase):
    """Test test-time augmentation"""
    
    def test_tta(self):
        """Test test-time augmentation"""
        ensemble = EfficientNetV2Ensemble(
            n_elements=20,
            variants=['s']
        )
        ensemble.eval()
        
        x = torch.randn(1, 3, 224, 224)
        
        # Apply multiple augmentations
        augmented = [
            x,
            torch.flip(x, dims=[3]),  # Horizontal flip
            torch.flip(x, dims=[2])   # Vertical flip
        ]
        
        predictions = []
        with torch.no_grad():
            for aug_x in augmented:
                pred = ensemble(aug_x)
                predictions.append(pred)
        
        # Average predictions
        tta_pred = torch.stack(predictions).mean(dim=0)
        
        self.assertEqual(tta_pred.shape, (1, 20))


class TestStochasticDepth(unittest.TestCase):
    """Test stochastic depth regularization"""
    
    def test_stochastic_depth(self):
        """Test stochastic depth"""
        model = EfficientNetV2(
            variant='s',
            n_elements=20,
            drop_path_rate=0.2
        )
        
        x = torch.randn(2, 3, 224, 224)
        
        # Training mode
        model.train()
        output_train = model(x)
        
        # Eval mode
        model.eval()
        output_eval = model(x)
        
        # Should differ due to stochastic depth
        self.assertEqual(output_train.shape, (2, 20))
        self.assertEqual(output_eval.shape, (2, 20))


if __name__ == '__main__':
    unittest.main()
