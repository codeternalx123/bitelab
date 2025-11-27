"""
Unit tests for advanced training infrastructure

Tests training pipeline, knowledge distillation, and optimization.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ai_nutrition.training.advanced_training import (
    AdvancedTrainer,
    KnowledgeDistillationLoss,
    MixUpAugmentation,
    CutMixAugmentation,
    ProgressiveTraining,
    CosineAnnealingWarmup
)


class TestAdvancedTrainer(unittest.TestCase):
    """Test advanced trainer"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Simple model for testing
        self.model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 5)
        )
        
        # Mock data
        self.train_data = [(torch.randn(4, 10), torch.randn(4, 5)) for _ in range(10)]
        self.val_data = [(torch.randn(4, 10), torch.randn(4, 5)) for _ in range(5)]
        
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = AdvancedTrainer(
            model=self.model,
            train_data=self.train_data,
            val_data=self.val_data
        )
        self.assertIsNotNone(trainer)
        
    def test_single_epoch_training(self):
        """Test single epoch training"""
        trainer = AdvancedTrainer(
            model=self.model,
            train_data=self.train_data,
            val_data=self.val_data
        )
        
        loss = trainer.train_epoch()
        
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
        
    def test_validation(self):
        """Test validation"""
        trainer = AdvancedTrainer(
            model=self.model,
            train_data=self.train_data,
            val_data=self.val_data
        )
        
        val_loss = trainer.validate()
        
        self.assertIsInstance(val_loss, float)
        self.assertGreater(val_loss, 0)


class TestKnowledgeDistillation(unittest.TestCase):
    """Test knowledge distillation"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Teacher model (larger)
        self.teacher = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 5)
        )
        
        # Student model (smaller)
        self.student = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
    def test_distillation_loss_initialization(self):
        """Test distillation loss initialization"""
        distill_loss = KnowledgeDistillationLoss(temperature=3.0, alpha=0.5)
        self.assertIsNotNone(distill_loss)
        
    def test_distillation_loss_computation(self):
        """Test distillation loss computation"""
        distill_loss = KnowledgeDistillationLoss(temperature=3.0, alpha=0.5)
        
        x = torch.randn(4, 10)
        targets = torch.randn(4, 5)
        
        # Get teacher predictions (no grad)
        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        
        # Get student predictions
        student_logits = self.student(x)
        
        # Compute distillation loss
        loss = distill_loss(student_logits, teacher_logits, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
        
    def test_temperature_effect(self):
        """Test temperature parameter effect"""
        # Higher temperature should soften distributions
        distill_loss_low = KnowledgeDistillationLoss(temperature=1.0, alpha=0.5)
        distill_loss_high = KnowledgeDistillationLoss(temperature=10.0, alpha=0.5)
        
        x = torch.randn(4, 10)
        targets = torch.randn(4, 5)
        
        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        
        student_logits = self.student(x)
        
        loss_low = distill_loss_low(student_logits, teacher_logits, targets)
        loss_high = distill_loss_high(student_logits, teacher_logits, targets)
        
        # Both should be valid losses
        self.assertGreater(loss_low.item(), 0)
        self.assertGreater(loss_high.item(), 0)


class TestMixUpAugmentation(unittest.TestCase):
    """Test MixUp augmentation"""
    
    def test_mixup_initialization(self):
        """Test MixUp initialization"""
        mixup = MixUpAugmentation(alpha=1.0)
        self.assertIsNotNone(mixup)
        
    def test_mixup_augmentation(self):
        """Test MixUp augmentation"""
        mixup = MixUpAugmentation(alpha=1.0)
        
        x = torch.randn(4, 3, 224, 224)
        y = torch.randn(4, 10)
        
        x_mixed, y_mixed = mixup(x, y)
        
        # Output should have same shape
        self.assertEqual(x_mixed.shape, x.shape)
        self.assertEqual(y_mixed.shape, y.shape)
        
    def test_mixup_interpolation(self):
        """Test MixUp creates interpolations"""
        mixup = MixUpAugmentation(alpha=1.0)
        
        # Create distinct samples
        x = torch.zeros(4, 3, 32, 32)
        x[0] = 1.0  # First sample is all 1s
        
        y = torch.zeros(4, 5)
        y[0, 0] = 1.0  # One-hot for first sample
        
        x_mixed, y_mixed = mixup(x, y)
        
        # Mixed samples should be between originals
        self.assertTrue(torch.any(x_mixed > 0))
        self.assertTrue(torch.any(x_mixed < 1))


class TestCutMixAugmentation(unittest.TestCase):
    """Test CutMix augmentation"""
    
    def test_cutmix_initialization(self):
        """Test CutMix initialization"""
        cutmix = CutMixAugmentation(alpha=1.0)
        self.assertIsNotNone(cutmix)
        
    def test_cutmix_augmentation(self):
        """Test CutMix augmentation"""
        cutmix = CutMixAugmentation(alpha=1.0)
        
        x = torch.randn(4, 3, 224, 224)
        y = torch.randn(4, 10)
        
        x_mixed, y_mixed = cutmix(x, y)
        
        # Output should have same shape
        self.assertEqual(x_mixed.shape, x.shape)
        self.assertEqual(y_mixed.shape, y.shape)
        
    def test_cutmix_creates_patches(self):
        """Test CutMix creates patch mixing"""
        cutmix = CutMixAugmentation(alpha=1.0)
        
        # Create distinct samples
        x = torch.zeros(4, 3, 64, 64)
        x[0] = 1.0  # First sample
        x[1] = 2.0  # Second sample
        
        y = torch.zeros(4, 5)
        
        x_mixed, y_mixed = cutmix(x, y)
        
        # Should have mix of values from different samples
        unique_vals = torch.unique(x_mixed)
        self.assertGreater(len(unique_vals), 2)


class TestProgressiveTraining(unittest.TestCase):
    """Test progressive training"""
    
    def test_progressive_initialization(self):
        """Test progressive training initialization"""
        model = nn.Sequential(
            nn.Linear(10, 50),
            nn.Linear(50, 5)
        )
        
        progressive = ProgressiveTraining(
            model=model,
            stages=[
                {'lr': 0.01, 'epochs': 5},
                {'lr': 0.001, 'epochs': 5}
            ]
        )
        
        self.assertIsNotNone(progressive)
        
    def test_stage_progression(self):
        """Test progression through training stages"""
        model = nn.Sequential(nn.Linear(10, 5))
        
        progressive = ProgressiveTraining(
            model=model,
            stages=[
                {'lr': 0.01, 'epochs': 2},
                {'lr': 0.001, 'epochs': 2}
            ]
        )
        
        # Should start at stage 0
        self.assertEqual(progressive.current_stage, 0)
        
        # Advance stage
        progressive.advance_stage()
        self.assertEqual(progressive.current_stage, 1)


class TestCosineAnnealingWarmup(unittest.TestCase):
    """Test cosine annealing with warmup"""
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = CosineAnnealingWarmup(
            optimizer=optimizer,
            warmup_epochs=5,
            max_epochs=100
        )
        
        self.assertIsNotNone(scheduler)
        
    def test_warmup_phase(self):
        """Test warmup phase increases LR"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = CosineAnnealingWarmup(
            optimizer=optimizer,
            warmup_epochs=5,
            max_epochs=100
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Step through warmup
        for _ in range(5):
            scheduler.step()
        
        warmup_lr = optimizer.param_groups[0]['lr']
        
        # LR should have increased during warmup
        self.assertGreater(warmup_lr, initial_lr)
        
    def test_cosine_annealing_phase(self):
        """Test cosine annealing after warmup"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = CosineAnnealingWarmup(
            optimizer=optimizer,
            warmup_epochs=2,
            max_epochs=10
        )
        
        # Complete warmup
        for _ in range(2):
            scheduler.step()
        
        lr_after_warmup = optimizer.param_groups[0]['lr']
        
        # Continue for several more steps
        for _ in range(5):
            scheduler.step()
        
        lr_after_annealing = optimizer.param_groups[0]['lr']
        
        # LR should decrease during cosine annealing
        self.assertLess(lr_after_annealing, lr_after_warmup)


class TestMixedPrecisionTraining(unittest.TestCase):
    """Test mixed precision training"""
    
    def test_fp16_training(self):
        """Test FP16 training"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler()
        
        x = torch.randn(4, 10)
        targets = torch.randn(4, 5)
        
        # Forward pass with autocast
        with torch.cuda.amp.autocast():
            outputs = model(x)
            loss = nn.MSELoss()(outputs, targets)
        
        # Backward with scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        self.assertIsNotNone(loss)


class TestGradientAccumulation(unittest.TestCase):
    """Test gradient accumulation"""
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation over multiple batches"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        accumulation_steps = 4
        
        # Accumulate gradients
        optimizer.zero_grad()
        for i in range(accumulation_steps):
            x = torch.randn(2, 10)
            targets = torch.randn(2, 5)
            
            outputs = model(x)
            loss = nn.MSELoss()(outputs, targets)
            
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
        
        # Single optimizer step after accumulation
        optimizer.step()
        
        # Gradients should be accumulated
        for param in model.parameters():
            self.assertIsNotNone(param.grad)


class TestEarlyStopping(unittest.TestCase):
    """Test early stopping"""
    
    def test_early_stopping_trigger(self):
        """Test early stopping triggers"""
        class EarlyStopping:
            def __init__(self, patience=5):
                self.patience = patience
                self.counter = 0
                self.best_loss = float('inf')
                
            def __call__(self, val_loss):
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.counter = 0
                    return False
                else:
                    self.counter += 1
                    return self.counter >= self.patience
        
        early_stop = EarlyStopping(patience=3)
        
        # Improving losses
        self.assertFalse(early_stop(1.0))
        self.assertFalse(early_stop(0.8))
        self.assertFalse(early_stop(0.6))
        
        # Non-improving losses
        self.assertFalse(early_stop(0.7))
        self.assertFalse(early_stop(0.7))
        should_stop = early_stop(0.7)
        
        # Should trigger early stopping
        self.assertTrue(should_stop)


class TestCheckpointing(unittest.TestCase):
    """Test model checkpointing"""
    
    def test_save_checkpoint(self):
        """Test saving checkpoint"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint = {
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 0.5
        }
        
        self.assertIn('epoch', checkpoint)
        self.assertIn('model_state_dict', checkpoint)
        
    def test_load_checkpoint(self):
        """Test loading checkpoint"""
        # Save checkpoint
        model1 = nn.Linear(10, 5)
        checkpoint = {
            'model_state_dict': model1.state_dict(),
            'epoch': 10
        }
        
        # Load into new model
        model2 = nn.Linear(10, 5)
        model2.load_state_dict(checkpoint['model_state_dict'])
        
        # Models should produce same output
        x = torch.randn(4, 10)
        output1 = model1(x)
        output2 = model2(x)
        
        self.assertTrue(torch.allclose(output1, output2))


class TestLearningRateScheduling(unittest.TestCase):
    """Test learning rate scheduling"""
    
    def test_step_lr(self):
        """Test step learning rate decay"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Step through epochs
        for _ in range(10):
            scheduler.step()
        
        decayed_lr = optimizer.param_groups[0]['lr']
        
        # LR should have decayed
        self.assertLess(decayed_lr, initial_lr)
        
    def test_reduce_on_plateau(self):
        """Test reduce on plateau scheduler"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=2, factor=0.5
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Simulate plateau
        for _ in range(5):
            scheduler.step(1.0)  # Constant loss
        
        reduced_lr = optimizer.param_groups[0]['lr']
        
        # LR should have reduced
        self.assertLess(reduced_lr, initial_lr)


class TestDataAugmentationPipeline(unittest.TestCase):
    """Test data augmentation pipeline"""
    
    def test_augmentation_composition(self):
        """Test composing multiple augmentations"""
        mixup = MixUpAugmentation(alpha=1.0)
        cutmix = CutMixAugmentation(alpha=1.0)
        
        x = torch.randn(4, 3, 224, 224)
        y = torch.randn(4, 10)
        
        # Apply MixUp
        x_aug1, y_aug1 = mixup(x, y)
        
        # Apply CutMix
        x_aug2, y_aug2 = cutmix(x, y)
        
        # Both should work
        self.assertEqual(x_aug1.shape, x.shape)
        self.assertEqual(x_aug2.shape, x.shape)


class TestTrainingMetrics(unittest.TestCase):
    """Test training metrics computation"""
    
    def test_loss_tracking(self):
        """Test loss tracking over epochs"""
        losses = []
        
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        
        for epoch in range(10):
            x = torch.randn(4, 10)
            targets = torch.randn(4, 5)
            
            outputs = model(x)
            loss = nn.MSELoss()(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Should have 10 loss values
        self.assertEqual(len(losses), 10)
        
    def test_validation_metrics(self):
        """Test validation metrics computation"""
        model = nn.Linear(10, 5)
        model.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for _ in range(5):
                x = torch.randn(4, 10)
                targets = torch.randn(4, 5)
                
                outputs = model(x)
                loss = nn.MSELoss()(outputs, targets)
                val_losses.append(loss.item())
        
        # Compute average validation loss
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        self.assertIsInstance(avg_val_loss, float)
        self.assertGreater(avg_val_loss, 0)


class TestTrainingPerformance(unittest.TestCase):
    """Test training performance"""
    
    def test_training_speed(self):
        """Test training speed"""
        import time
        
        model = nn.Sequential(
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Linear(500, 50)
        )
        optimizer = torch.optim.Adam(model.parameters())
        
        # Time training loop
        start = time.time()
        
        for _ in range(100):
            x = torch.randn(32, 100)
            targets = torch.randn(32, 50)
            
            outputs = model(x)
            loss = nn.MSELoss()(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        elapsed = time.time() - start
        
        # Should process >10 batches/second
        throughput = 100 / elapsed
        self.assertGreater(throughput, 10)


if __name__ == '__main__':
    unittest.main()
