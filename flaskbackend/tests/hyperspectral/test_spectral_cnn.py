"""
Unit tests for hyperspectral deep learning module (SpectralCNN)

Tests CNN architectures, training, and inference.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ai_nutrition.hyperspectral.spectral_cnn import (
    SpectralCNN1D,
    SpectralCNN3D,
    HybridSpectralNet,
    SpectralTransformer,
    SpectralAttention,
    SEBlock,
    CBAM
)


class TestSpectralCNN1D(unittest.TestCase):
    """Test 1D CNN for spectral analysis"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.n_bands = 50
        self.n_classes = 10
        self.batch_size = 4
        
    def test_initialization(self):
        """Test 1D CNN initialization"""
        model = SpectralCNN1D(
            n_bands=self.n_bands,
            n_classes=self.n_classes
        )
        self.assertIsNotNone(model)
        
    def test_forward_pass(self):
        """Test forward pass through 1D CNN"""
        model = SpectralCNN1D(
            n_bands=self.n_bands,
            n_classes=self.n_classes
        )
        
        # Create input: (batch, bands)
        x = torch.randn(self.batch_size, self.n_bands)
        
        output = model(x)
        
        # Output should be (batch, n_classes)
        self.assertEqual(output.shape, (self.batch_size, self.n_classes))
        
    def test_different_input_sizes(self):
        """Test with different input sizes"""
        model = SpectralCNN1D(n_bands=100, n_classes=5)
        
        # Test with different batch sizes
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 100)
            output = model(x)
            self.assertEqual(output.shape, (batch_size, 5))
            
    def test_gradient_flow(self):
        """Test gradient flow through 1D CNN"""
        model = SpectralCNN1D(n_bands=self.n_bands, n_classes=self.n_classes)
        
        x = torch.randn(self.batch_size, self.n_bands, requires_grad=True)
        output = model(x)
        
        # Compute loss and backward
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(x.grad)
        
    def test_batch_normalization(self):
        """Test batch normalization layers"""
        model = SpectralCNN1D(
            n_bands=self.n_bands,
            n_classes=self.n_classes,
            use_batch_norm=True
        )
        
        x = torch.randn(self.batch_size, self.n_bands)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.n_classes))
        
    def test_dropout(self):
        """Test dropout functionality"""
        model = SpectralCNN1D(
            n_bands=self.n_bands,
            n_classes=self.n_classes,
            dropout_rate=0.5
        )
        
        x = torch.randn(self.batch_size, self.n_bands)
        
        # Training mode (dropout active)
        model.train()
        output_train = model(x)
        
        # Eval mode (dropout inactive)
        model.eval()
        output_eval = model(x)
        
        # Outputs should be different
        self.assertFalse(torch.allclose(output_train, output_eval))


class TestSpectralCNN3D(unittest.TestCase):
    """Test 3D CNN for spatial-spectral analysis"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.height = 32
        self.width = 32
        self.n_bands = 50
        self.n_classes = 10
        self.batch_size = 2
        
    def test_initialization(self):
        """Test 3D CNN initialization"""
        model = SpectralCNN3D(
            n_bands=self.n_bands,
            n_classes=self.n_classes
        )
        self.assertIsNotNone(model)
        
    def test_forward_pass(self):
        """Test forward pass through 3D CNN"""
        model = SpectralCNN3D(
            n_bands=self.n_bands,
            n_classes=self.n_classes
        )
        
        # Create input: (batch, bands, height, width)
        x = torch.randn(self.batch_size, self.n_bands, self.height, self.width)
        
        output = model(x)
        
        # Output should be (batch, n_classes)
        self.assertEqual(output.shape, (self.batch_size, self.n_classes))
        
    def test_spatial_reduction(self):
        """Test spatial dimension reduction"""
        model = SpectralCNN3D(n_bands=self.n_bands, n_classes=self.n_classes)
        
        # Different spatial sizes
        for size in [16, 32, 64]:
            x = torch.randn(self.batch_size, self.n_bands, size, size)
            output = model(x)
            self.assertEqual(output.shape, (self.batch_size, self.n_classes))
            
    def test_3d_convolution(self):
        """Test 3D convolution operations"""
        model = SpectralCNN3D(
            n_bands=self.n_bands,
            n_classes=self.n_classes,
            kernel_size=(3, 3, 3)
        )
        
        x = torch.randn(self.batch_size, self.n_bands, self.height, self.width)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.n_classes))


class TestHybridSpectralNet(unittest.TestCase):
    """Test hybrid spatial-spectral network"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.height = 32
        self.width = 32
        self.n_bands = 50
        self.n_classes = 10
        self.batch_size = 2
        
    def test_initialization(self):
        """Test hybrid network initialization"""
        model = HybridSpectralNet(
            n_bands=self.n_bands,
            n_classes=self.n_classes
        )
        self.assertIsNotNone(model)
        
    def test_forward_pass(self):
        """Test forward pass through hybrid network"""
        model = HybridSpectralNet(
            n_bands=self.n_bands,
            n_classes=self.n_classes
        )
        
        x = torch.randn(self.batch_size, self.n_bands, self.height, self.width)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.n_classes))
        
    def test_spectral_branch(self):
        """Test spectral processing branch"""
        model = HybridSpectralNet(
            n_bands=self.n_bands,
            n_classes=self.n_classes
        )
        
        x = torch.randn(self.batch_size, self.n_bands, self.height, self.width)
        
        # Extract spectral features
        spectral_features = model.extract_spectral_features(x)
        
        self.assertIsNotNone(spectral_features)
        self.assertEqual(spectral_features.shape[0], self.batch_size)
        
    def test_spatial_branch(self):
        """Test spatial processing branch"""
        model = HybridSpectralNet(
            n_bands=self.n_bands,
            n_classes=self.n_classes
        )
        
        x = torch.randn(self.batch_size, self.n_bands, self.height, self.width)
        
        # Extract spatial features
        spatial_features = model.extract_spatial_features(x)
        
        self.assertIsNotNone(spatial_features)
        self.assertEqual(spatial_features.shape[0], self.batch_size)
        
    def test_feature_fusion(self):
        """Test feature fusion mechanism"""
        model = HybridSpectralNet(
            n_bands=self.n_bands,
            n_classes=self.n_classes,
            fusion_method='concat'
        )
        
        x = torch.randn(self.batch_size, self.n_bands, self.height, self.width)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.n_classes))


class TestSpectralTransformer(unittest.TestCase):
    """Test transformer-based spectral analysis"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.n_bands = 50
        self.n_classes = 10
        self.batch_size = 4
        
    def test_initialization(self):
        """Test transformer initialization"""
        model = SpectralTransformer(
            n_bands=self.n_bands,
            n_classes=self.n_classes
        )
        self.assertIsNotNone(model)
        
    def test_forward_pass(self):
        """Test forward pass through transformer"""
        model = SpectralTransformer(
            n_bands=self.n_bands,
            n_classes=self.n_classes,
            d_model=128,
            n_heads=4,
            n_layers=2
        )
        
        x = torch.randn(self.batch_size, self.n_bands)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.n_classes))
        
    def test_self_attention(self):
        """Test self-attention mechanism"""
        model = SpectralTransformer(
            n_bands=self.n_bands,
            n_classes=self.n_classes,
            d_model=64,
            n_heads=4
        )
        
        x = torch.randn(self.batch_size, self.n_bands)
        
        # Get attention weights
        output, attention_weights = model(x, return_attention=True)
        
        self.assertIsNotNone(attention_weights)
        self.assertEqual(output.shape, (self.batch_size, self.n_classes))
        
    def test_positional_encoding(self):
        """Test positional encoding"""
        model = SpectralTransformer(
            n_bands=self.n_bands,
            n_classes=self.n_classes,
            use_positional_encoding=True
        )
        
        x = torch.randn(self.batch_size, self.n_bands)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.n_classes))
        
    def test_multiple_heads(self):
        """Test different numbers of attention heads"""
        for n_heads in [1, 2, 4, 8]:
            model = SpectralTransformer(
                n_bands=self.n_bands,
                n_classes=self.n_classes,
                d_model=128,
                n_heads=n_heads
            )
            
            x = torch.randn(self.batch_size, self.n_bands)
            output = model(x)
            
            self.assertEqual(output.shape, (self.batch_size, self.n_classes))


class TestSpectralAttention(unittest.TestCase):
    """Test spectral attention mechanisms"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.channels = 64
        self.n_bands = 50
        
    def test_channel_attention(self):
        """Test channel attention"""
        attention = SpectralAttention(channels=self.channels, attention_type='channel')
        
        x = torch.randn(self.batch_size, self.channels, self.n_bands)
        output = attention(x)
        
        self.assertEqual(output.shape, x.shape)
        
    def test_spatial_attention(self):
        """Test spatial attention"""
        attention = SpectralAttention(channels=self.channels, attention_type='spatial')
        
        x = torch.randn(self.batch_size, self.channels, 32, 32)
        output = attention(x)
        
        self.assertEqual(output.shape, x.shape)
        
    def test_attention_weights(self):
        """Test attention weight computation"""
        attention = SpectralAttention(channels=self.channels)
        
        x = torch.randn(self.batch_size, self.channels, self.n_bands)
        
        # Get attention weights
        output, weights = attention(x, return_weights=True)
        
        self.assertIsNotNone(weights)
        # Weights should sum to ~1
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(self.batch_size, self.channels), atol=0.01))


class TestSEBlock(unittest.TestCase):
    """Test Squeeze-and-Excitation block"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.channels = 64
        self.height = 32
        self.width = 32
        
    def test_initialization(self):
        """Test SE block initialization"""
        se = SEBlock(channels=self.channels)
        self.assertIsNotNone(se)
        
    def test_forward_pass(self):
        """Test forward pass through SE block"""
        se = SEBlock(channels=self.channels, reduction=16)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        output = se(x)
        
        self.assertEqual(output.shape, x.shape)
        
    def test_channel_recalibration(self):
        """Test channel recalibration"""
        se = SEBlock(channels=self.channels)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        output = se(x)
        
        # Output should be scaled version of input
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.allclose(output, x))
        
    def test_reduction_ratio(self):
        """Test different reduction ratios"""
        for reduction in [4, 8, 16]:
            se = SEBlock(channels=self.channels, reduction=reduction)
            
            x = torch.randn(self.batch_size, self.channels, self.height, self.width)
            output = se(x)
            
            self.assertEqual(output.shape, x.shape)


class TestCBAM(unittest.TestCase):
    """Test Convolutional Block Attention Module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.channels = 64
        self.height = 32
        self.width = 32
        
    def test_initialization(self):
        """Test CBAM initialization"""
        cbam = CBAM(channels=self.channels)
        self.assertIsNotNone(cbam)
        
    def test_forward_pass(self):
        """Test forward pass through CBAM"""
        cbam = CBAM(channels=self.channels)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        output = cbam(x)
        
        self.assertEqual(output.shape, x.shape)
        
    def test_channel_attention(self):
        """Test channel attention in CBAM"""
        cbam = CBAM(channels=self.channels)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
        # Get channel attention
        channel_att = cbam.channel_attention(x)
        
        self.assertEqual(channel_att.shape, (self.batch_size, self.channels, 1, 1))
        
    def test_spatial_attention(self):
        """Test spatial attention in CBAM"""
        cbam = CBAM(channels=self.channels)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
        # Get spatial attention
        spatial_att = cbam.spatial_attention(x)
        
        self.assertEqual(spatial_att.shape, (self.batch_size, 1, self.height, self.width))


class TestModelTraining(unittest.TestCase):
    """Test model training functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.n_bands = 50
        self.n_classes = 10
        self.batch_size = 4
        
    def test_loss_computation(self):
        """Test loss computation"""
        model = SpectralCNN1D(n_bands=self.n_bands, n_classes=self.n_classes)
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(self.batch_size, self.n_bands)
        targets = torch.randint(0, self.n_classes, (self.batch_size,))
        
        outputs = model(x)
        loss = criterion(outputs, targets)
        
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0)
        
    def test_optimizer_step(self):
        """Test optimizer step"""
        model = SpectralCNN1D(n_bands=self.n_bands, n_classes=self.n_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(self.batch_size, self.n_bands)
        targets = torch.randint(0, self.n_classes, (self.batch_size,))
        
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Loss should be computed
        self.assertIsNotNone(loss)
        
    def test_multiple_training_steps(self):
        """Test multiple training steps"""
        model = SpectralCNN1D(n_bands=self.n_bands, n_classes=self.n_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        for _ in range(10):
            x = torch.randn(self.batch_size, self.n_bands)
            targets = torch.randint(0, self.n_classes, (self.batch_size,))
            
            outputs = model(x)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Should have collected losses
        self.assertEqual(len(losses), 10)


class TestModelInference(unittest.TestCase):
    """Test model inference functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.n_bands = 50
        self.n_classes = 10
        
    def test_single_sample_inference(self):
        """Test inference on single sample"""
        model = SpectralCNN1D(n_bands=self.n_bands, n_classes=self.n_classes)
        model.eval()
        
        x = torch.randn(1, self.n_bands)
        
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, (1, self.n_classes))
        
    def test_batch_inference(self):
        """Test inference on batch"""
        model = SpectralCNN1D(n_bands=self.n_bands, n_classes=self.n_classes)
        model.eval()
        
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, self.n_bands)
            
            with torch.no_grad():
                output = model(x)
            
            self.assertEqual(output.shape, (batch_size, self.n_classes))
            
    def test_probability_output(self):
        """Test probability output"""
        model = SpectralCNN1D(n_bands=self.n_bands, n_classes=self.n_classes)
        model.eval()
        
        x = torch.randn(4, self.n_bands)
        
        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)
        
        # Probabilities should sum to 1
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(4)))


class TestModelSaveLoad(unittest.TestCase):
    """Test model save and load functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.n_bands = 50
        self.n_classes = 10
        
    def test_state_dict_save_load(self):
        """Test state dict save and load"""
        model1 = SpectralCNN1D(n_bands=self.n_bands, n_classes=self.n_classes)
        
        # Save state dict
        state_dict = model1.state_dict()
        
        # Create new model and load
        model2 = SpectralCNN1D(n_bands=self.n_bands, n_classes=self.n_classes)
        model2.load_state_dict(state_dict)
        
        # Models should produce same output
        x = torch.randn(4, self.n_bands)
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model1(x)
            output2 = model2(x)
        
        self.assertTrue(torch.allclose(output1, output2))


class TestModelEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_zero_input(self):
        """Test with zero input"""
        model = SpectralCNN1D(n_bands=50, n_classes=10)
        
        x = torch.zeros(4, 50)
        output = model(x)
        
        self.assertEqual(output.shape, (4, 10))
        self.assertFalse(torch.any(torch.isnan(output)))
        
    def test_single_class(self):
        """Test with single class"""
        model = SpectralCNN1D(n_bands=50, n_classes=1)
        
        x = torch.randn(4, 50)
        output = model(x)
        
        self.assertEqual(output.shape, (4, 1))
        
    def test_large_batch(self):
        """Test with large batch size"""
        model = SpectralCNN1D(n_bands=50, n_classes=10)
        
        x = torch.randn(128, 50)
        output = model(x)
        
        self.assertEqual(output.shape, (128, 10))


class TestModelPerformance(unittest.TestCase):
    """Test model performance"""
    
    def test_inference_speed(self):
        """Test inference speed"""
        import time
        
        model = SpectralCNN1D(n_bands=100, n_classes=20)
        model.eval()
        
        # Warm up
        x = torch.randn(1, 100)
        with torch.no_grad():
            _ = model(x)
        
        # Time inference
        n_samples = 1000
        x = torch.randn(n_samples, 100)
        
        start = time.time()
        with torch.no_grad():
            output = model(x)
        elapsed = time.time() - start
        
        # Should process >100 samples/second
        fps = n_samples / elapsed
        self.assertGreater(fps, 100)
        
    def test_memory_efficiency(self):
        """Test memory efficiency"""
        model = SpectralCNN3D(n_bands=50, n_classes=10)
        
        # Should handle reasonably sized batches
        x = torch.randn(8, 50, 64, 64)
        output = model(x)
        
        self.assertEqual(output.shape, (8, 10))


if __name__ == '__main__':
    unittest.main()
