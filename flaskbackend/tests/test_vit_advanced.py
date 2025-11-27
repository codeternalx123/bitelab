"""
Unit tests for Vision Transformer (ViT) model

Tests ViT architecture, training, and inference.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ai_nutrition.models.vit_advanced import (
    VisionTransformerAdvanced,
    PatchEmbedding,
    TransformerEncoder,
    MultiHeadAttention,
    ElementPredictionHead
)


class TestVisionTransformerAdvanced(unittest.TestCase):
    """Test ViT advanced architecture"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.img_size = 224
        self.patch_size = 16
        self.in_channels = 3
        self.n_elements = 20
        self.batch_size = 2
        
    def test_initialization(self):
        """Test ViT initialization"""
        model = VisionTransformerAdvanced(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            n_elements=self.n_elements
        )
        self.assertIsNotNone(model)
        
    def test_forward_pass(self):
        """Test forward pass through ViT"""
        model = VisionTransformerAdvanced(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            n_elements=self.n_elements
        )
        
        x = torch.randn(self.batch_size, self.in_channels, self.img_size, self.img_size)
        output = model(x)
        
        # Output should be (batch, n_elements)
        self.assertEqual(output.shape, (self.batch_size, self.n_elements))
        
    def test_different_image_sizes(self):
        """Test with different image sizes"""
        for img_size in [224, 256, 384]:
            model = VisionTransformerAdvanced(
                img_size=img_size,
                patch_size=16,
                in_channels=3,
                n_elements=20
            )
            
            x = torch.randn(2, 3, img_size, img_size)
            output = model(x)
            
            self.assertEqual(output.shape, (2, 20))
            
    def test_different_patch_sizes(self):
        """Test with different patch sizes"""
        for patch_size in [8, 16, 32]:
            model = VisionTransformerAdvanced(
                img_size=224,
                patch_size=patch_size,
                in_channels=3,
                n_elements=20
            )
            
            x = torch.randn(2, 3, 224, 224)
            output = model(x)
            
            self.assertEqual(output.shape, (2, 20))


class TestPatchEmbedding(unittest.TestCase):
    """Test patch embedding layer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.img_size = 224
        self.patch_size = 16
        self.in_channels = 3
        self.embed_dim = 768
        
    def test_patch_embedding(self):
        """Test patch embedding"""
        patch_embed = PatchEmbedding(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim
        )
        
        x = torch.randn(2, self.in_channels, self.img_size, self.img_size)
        patches = patch_embed(x)
        
        # Should produce sequence of patch embeddings
        n_patches = (self.img_size // self.patch_size) ** 2
        self.assertEqual(patches.shape, (2, n_patches, self.embed_dim))
        
    def test_positional_encoding(self):
        """Test positional encoding"""
        patch_embed = PatchEmbedding(
            img_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            use_positional_encoding=True
        )
        
        x = torch.randn(2, 3, 224, 224)
        patches = patch_embed(x)
        
        # Should add positional information
        self.assertIsNotNone(patches)


class TestTransformerEncoder(unittest.TestCase):
    """Test transformer encoder"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.embed_dim = 768
        self.n_heads = 12
        self.mlp_ratio = 4
        
    def test_encoder_forward(self):
        """Test encoder forward pass"""
        encoder = TransformerEncoder(
            embed_dim=self.embed_dim,
            n_heads=self.n_heads,
            mlp_ratio=self.mlp_ratio
        )
        
        x = torch.randn(2, 196, self.embed_dim)
        output = encoder(x)
        
        self.assertEqual(output.shape, x.shape)
        
    def test_encoder_attention(self):
        """Test encoder attention mechanism"""
        encoder = TransformerEncoder(
            embed_dim=self.embed_dim,
            n_heads=self.n_heads
        )
        
        x = torch.randn(2, 196, self.embed_dim)
        output, attention = encoder(x, return_attention=True)
        
        self.assertIsNotNone(attention)
        self.assertEqual(output.shape, x.shape)


class TestMultiHeadAttention(unittest.TestCase):
    """Test multi-head attention"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.embed_dim = 768
        self.n_heads = 12
        
    def test_attention_forward(self):
        """Test attention forward pass"""
        attention = MultiHeadAttention(
            embed_dim=self.embed_dim,
            n_heads=self.n_heads
        )
        
        x = torch.randn(2, 196, self.embed_dim)
        output = attention(x)
        
        self.assertEqual(output.shape, x.shape)
        
    def test_attention_weights(self):
        """Test attention weight computation"""
        attention = MultiHeadAttention(
            embed_dim=self.embed_dim,
            n_heads=self.n_heads
        )
        
        x = torch.randn(2, 196, self.embed_dim)
        output, weights = attention(x, return_weights=True)
        
        self.assertIsNotNone(weights)
        # Weights should have shape (batch, n_heads, seq_len, seq_len)
        self.assertEqual(weights.shape[0], 2)
        self.assertEqual(weights.shape[1], self.n_heads)


class TestElementPredictionHead(unittest.TestCase):
    """Test element prediction head"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.embed_dim = 768
        self.n_elements = 20
        
    def test_prediction_head(self):
        """Test prediction head"""
        head = ElementPredictionHead(
            embed_dim=self.embed_dim,
            n_elements=self.n_elements
        )
        
        x = torch.randn(2, self.embed_dim)
        predictions = head(x)
        
        self.assertEqual(predictions.shape, (2, self.n_elements))
        
    def test_uncertainty_quantification(self):
        """Test uncertainty quantification"""
        head = ElementPredictionHead(
            embed_dim=self.embed_dim,
            n_elements=self.n_elements,
            predict_uncertainty=True
        )
        
        x = torch.randn(2, self.embed_dim)
        predictions, uncertainty = head(x)
        
        self.assertEqual(predictions.shape, (2, self.n_elements))
        self.assertEqual(uncertainty.shape, (2, self.n_elements))
        self.assertTrue(torch.all(uncertainty > 0))


class TestViTTraining(unittest.TestCase):
    """Test ViT training functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = VisionTransformerAdvanced(
            img_size=224,
            patch_size=16,
            in_channels=3,
            n_elements=20
        )
        
    def test_loss_computation(self):
        """Test loss computation"""
        criterion = nn.MSELoss()
        
        x = torch.randn(2, 3, 224, 224)
        targets = torch.randn(2, 20)
        
        predictions = self.model(x)
        loss = criterion(predictions, targets)
        
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0)
        
    def test_backward_pass(self):
        """Test backward pass"""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        x = torch.randn(2, 3, 224, 224)
        targets = torch.randn(2, 20)
        
        # Forward
        predictions = self.model(x)
        loss = criterion(predictions, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Loss should be computed
        self.assertIsNotNone(loss)
        
    def test_gradient_flow(self):
        """Test gradient flow through model"""
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        
        output = self.model(x)
        loss = output.sum()
        loss.backward()
        
        # Input should have gradients
        self.assertIsNotNone(x.grad)


class TestViTInference(unittest.TestCase):
    """Test ViT inference"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = VisionTransformerAdvanced(
            img_size=224,
            patch_size=16,
            in_channels=3,
            n_elements=20
        )
        
    def test_inference_mode(self):
        """Test inference mode"""
        self.model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = self.model(x)
        
        self.assertEqual(output.shape, (1, 20))
        
    def test_batch_inference(self):
        """Test batch inference"""
        self.model.eval()
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 224, 224)
            
            with torch.no_grad():
                output = self.model(x)
            
            self.assertEqual(output.shape, (batch_size, 20))


class TestViTAttentionVisualization(unittest.TestCase):
    """Test attention visualization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = VisionTransformerAdvanced(
            img_size=224,
            patch_size=16,
            in_channels=3,
            n_elements=20
        )
        
    def test_extract_attention_maps(self):
        """Test extracting attention maps"""
        self.model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = self.model(x)
            attention_maps = self.model.get_attention_maps()
        
        # Should have attention maps from each layer
        self.assertIsNotNone(attention_maps)


class TestViTMultiScale(unittest.TestCase):
    """Test multi-scale processing"""
    
    def test_multi_scale_patches(self):
        """Test multi-scale patch extraction"""
        model = VisionTransformerAdvanced(
            img_size=224,
            patch_size=16,
            in_channels=3,
            n_elements=20,
            multi_scale=True,
            patch_sizes=[8, 16, 32]
        )
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 20))


class TestViTEdgeCases(unittest.TestCase):
    """Test edge cases"""
    
    def test_single_channel_input(self):
        """Test with single channel input"""
        model = VisionTransformerAdvanced(
            img_size=224,
            patch_size=16,
            in_channels=1,
            n_elements=20
        )
        
        x = torch.randn(2, 1, 224, 224)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 20))
        
    def test_single_element_prediction(self):
        """Test predicting single element"""
        model = VisionTransformerAdvanced(
            img_size=224,
            patch_size=16,
            in_channels=3,
            n_elements=1
        )
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 1))


class TestViTPerformance(unittest.TestCase):
    """Test ViT performance"""
    
    def test_inference_speed(self):
        """Test inference speed"""
        import time
        
        model = VisionTransformerAdvanced(
            img_size=224,
            patch_size=16,
            in_channels=3,
            n_elements=20
        )
        model.eval()
        
        # Warm up
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _ = model(x)
        
        # Time batch inference
        batch_size = 32
        x = torch.randn(batch_size, 3, 224, 224)
        
        start = time.time()
        with torch.no_grad():
            output = model(x)
        elapsed = time.time() - start
        
        # Should process >10 images/second
        throughput = batch_size / elapsed
        self.assertGreater(throughput, 10)


class TestViTSaveLoad(unittest.TestCase):
    """Test model save and load"""
    
    def test_state_dict_save_load(self):
        """Test state dict save/load"""
        model1 = VisionTransformerAdvanced(
            img_size=224,
            patch_size=16,
            in_channels=3,
            n_elements=20
        )
        
        # Save state
        state_dict = model1.state_dict()
        
        # Load into new model
        model2 = VisionTransformerAdvanced(
            img_size=224,
            patch_size=16,
            in_channels=3,
            n_elements=20
        )
        model2.load_state_dict(state_dict)
        
        # Should produce same output
        x = torch.randn(2, 3, 224, 224)
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model1(x)
            output2 = model2(x)
        
        self.assertTrue(torch.allclose(output1, output2))


class TestViTParameterCount(unittest.TestCase):
    """Test parameter counting"""
    
    def test_parameter_count(self):
        """Test model parameter count"""
        model = VisionTransformerAdvanced(
            img_size=224,
            patch_size=16,
            in_channels=3,
            n_elements=20,
            embed_dim=768,
            depth=12,
            n_heads=12
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        
        # ViT-Base should have ~85-90M parameters
        self.assertGreater(n_params, 80_000_000)
        self.assertLess(n_params, 100_000_000)
        
    def test_trainable_parameters(self):
        """Test counting trainable parameters"""
        model = VisionTransformerAdvanced(
            img_size=224,
            patch_size=16,
            in_channels=3,
            n_elements=20
        )
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        # All parameters should be trainable by default
        self.assertEqual(trainable, total)


class TestViTMemoryEfficiency(unittest.TestCase):
    """Test memory efficiency"""
    
    def test_gradient_checkpointing(self):
        """Test gradient checkpointing for memory efficiency"""
        model = VisionTransformerAdvanced(
            img_size=224,
            patch_size=16,
            in_channels=3,
            n_elements=20,
            use_checkpoint=True
        )
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 20))


class TestViTRegularization(unittest.TestCase):
    """Test regularization techniques"""
    
    def test_dropout(self):
        """Test dropout regularization"""
        model = VisionTransformerAdvanced(
            img_size=224,
            patch_size=16,
            in_channels=3,
            n_elements=20,
            dropout=0.1
        )
        
        x = torch.randn(2, 3, 224, 224)
        
        # Training mode
        model.train()
        output_train = model(x)
        
        # Eval mode
        model.eval()
        output_eval = model(x)
        
        # Outputs should differ due to dropout
        self.assertFalse(torch.allclose(output_train, output_eval))
        
    def test_stochastic_depth(self):
        """Test stochastic depth"""
        model = VisionTransformerAdvanced(
            img_size=224,
            patch_size=16,
            in_channels=3,
            n_elements=20,
            drop_path_rate=0.1
        )
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 20))


if __name__ == '__main__':
    unittest.main()
