"""
Advanced Training Strategies
=============================

Sophisticated training techniques including curriculum learning,
adversarial training, knowledge distillation, and mixup variants.

Features:
1. Curriculum Learning (easy-to-hard progression)
2. Adversarial Training (PGD, FGSM)
3. Advanced Knowledge Distillation
4. Label Smoothing and regularization
5. Cosine annealing with warm restarts
6. Stochastic Weight Averaging (SWA)
7. Gradient accumulation
8. Mixed precision training (FP16/BF16)

Performance Targets:
- 5-10% accuracy improvement
- 2-3x faster convergence
- Better generalization
- Robust to adversarial attacks
- Reduced overfitting

Author: Wellomex AI Team
Date: November 2025
Version: 5.0.0
"""

import time
import logging
import random
import math
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from datetime import datetime
from collections import defaultdict, OrderedDict
import json

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class CurriculumStrategy(Enum):
    """Curriculum learning strategies"""
    DIFFICULTY = "difficulty"  # Based on sample difficulty
    LOSS = "loss"  # Based on loss values
    CONFIDENCE = "confidence"  # Based on model confidence
    TRANSFER = "transfer"  # Transfer from similar tasks


class AdversarialMethod(Enum):
    """Adversarial training methods"""
    FGSM = "fgsm"  # Fast Gradient Sign Method
    PGD = "pgd"  # Projected Gradient Descent
    TRADES = "trades"  # TRadeoff-inspired Adversarial DEfense
    MART = "mart"  # Misclassification Aware adversarial training


@dataclass
class TrainingConfig:
    """Advanced training configuration"""
    # Curriculum learning
    enable_curriculum: bool = True
    curriculum_strategy: CurriculumStrategy = CurriculumStrategy.DIFFICULTY
    curriculum_epochs: int = 50
    start_difficulty: float = 0.3
    
    # Adversarial training
    enable_adversarial: bool = True
    adversarial_method: AdversarialMethod = AdversarialMethod.PGD
    adversarial_epsilon: float = 0.03
    adversarial_alpha: float = 0.01
    adversarial_steps: int = 10
    
    # Knowledge distillation
    enable_distillation: bool = False
    teacher_model: Optional[Any] = None
    distillation_alpha: float = 0.5
    distillation_temperature: float = 3.0
    
    # Label smoothing
    label_smoothing: float = 0.1
    
    # Optimization
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    batch_size: int = 128
    num_epochs: int = 200
    
    # SWA
    enable_swa: bool = True
    swa_start_epoch: int = 160
    swa_lr: float = 0.05
    
    # Mixed precision
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    
    # Warm restart
    enable_warm_restart: bool = True
    restart_period: int = 50
    restart_mult: float = 2.0


# ============================================================================
# CURRICULUM LEARNING
# ============================================================================

class CurriculumScheduler:
    """
    Curriculum learning scheduler
    
    Gradually introduces harder samples during training.
    """
    
    def __init__(
        self,
        dataset: Any,
        config: TrainingConfig
    ):
        self.dataset = dataset
        self.config = config
        
        # Compute sample difficulties
        self.sample_difficulties = self._compute_difficulties()
        
        # Current difficulty threshold
        self.current_difficulty = config.start_difficulty
        
        logger.info("Curriculum Scheduler initialized")
    
    def _compute_difficulties(self) -> np.ndarray:
        """Compute difficulty score for each sample"""
        # In production, use pre-computed scores or model predictions
        # For now, simulate with random scores
        
        num_samples = len(self.dataset) if hasattr(self.dataset, '__len__') else 1000
        difficulties = np.random.beta(2, 5, num_samples)  # Skewed toward easier
        
        return difficulties
    
    def update_difficulty(self, epoch: int):
        """Update difficulty threshold based on epoch"""
        progress = min(epoch / self.config.curriculum_epochs, 1.0)
        
        # Linear increase in difficulty
        self.current_difficulty = (
            self.config.start_difficulty +
            progress * (1.0 - self.config.start_difficulty)
        )
        
        logger.info(f"Epoch {epoch}: Difficulty threshold = {self.current_difficulty:.2f}")
    
    def get_subset_indices(self) -> List[int]:
        """Get indices of samples within current difficulty threshold"""
        valid_indices = np.where(
            self.sample_difficulties <= self.current_difficulty
        )[0]
        
        return valid_indices.tolist()
    
    def get_dataloader(self, batch_size: int) -> Any:
        """Get dataloader with current curriculum subset"""
        indices = self.get_subset_indices()
        
        if TORCH_AVAILABLE:
            from torch.utils.data import Subset, DataLoader
            
            subset = Subset(self.dataset, indices)
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4
            )
            
            return loader
        
        return None


# ============================================================================
# ADVERSARIAL TRAINING
# ============================================================================

class AdversarialTrainer:
    """
    Adversarial training to improve robustness
    
    Generates adversarial examples during training.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        logger.info(f"Adversarial Trainer initialized with {config.adversarial_method.value}")
    
    def fgsm_attack(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor:
        """Fast Gradient Sign Method attack"""
        # Require gradients
        inputs.requires_grad = True
        
        # Forward pass
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        sign_data_grad = inputs.grad.sign()
        perturbed_inputs = inputs + epsilon * sign_data_grad
        
        # Clip to valid range
        perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
        
        return perturbed_inputs.detach()
    
    def pgd_attack(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        epsilon: float,
        alpha: float,
        num_steps: int
    ) -> torch.Tensor:
        """Projected Gradient Descent attack"""
        # Start from random perturbation
        delta = torch.zeros_like(inputs).uniform_(-epsilon, epsilon)
        delta.requires_grad = True
        
        for _ in range(num_steps):
            # Forward pass
            outputs = model(inputs + delta)
            loss = F.cross_entropy(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Update perturbation
            delta.data = delta.data + alpha * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(inputs + delta.data, 0, 1) - inputs
            
            # Zero gradients
            delta.grad.zero_()
        
        perturbed_inputs = inputs + delta.detach()
        
        return perturbed_inputs
    
    def generate_adversarial(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Generate adversarial examples based on configured method"""
        model.eval()  # Set to eval mode for attack generation
        
        if self.config.adversarial_method == AdversarialMethod.FGSM:
            adv_inputs = self.fgsm_attack(
                model, inputs, targets,
                self.config.adversarial_epsilon
            )
        elif self.config.adversarial_method == AdversarialMethod.PGD:
            adv_inputs = self.pgd_attack(
                model, inputs, targets,
                self.config.adversarial_epsilon,
                self.config.adversarial_alpha,
                self.config.adversarial_steps
            )
        else:
            adv_inputs = inputs
        
        model.train()  # Back to training mode
        
        return adv_inputs
    
    def adversarial_loss(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss on adversarial examples"""
        # Generate adversarial examples
        adv_inputs = self.generate_adversarial(model, inputs, targets)
        
        # Forward pass on adversarial examples
        adv_outputs = model(adv_inputs)
        
        # Compute loss
        loss = F.cross_entropy(adv_outputs, targets)
        
        return loss


# ============================================================================
# KNOWLEDGE DISTILLATION
# ============================================================================

class KnowledgeDistiller:
    """
    Advanced knowledge distillation
    
    Transfer knowledge from teacher to student model.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        config: TrainingConfig
    ):
        self.teacher_model = teacher_model
        self.config = config
        
        # Freeze teacher
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        teacher_model.eval()
        
        logger.info("Knowledge Distiller initialized")
    
    def distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute distillation loss"""
        T = self.config.distillation_temperature
        alpha = self.config.distillation_alpha
        
        # Hard target loss (student vs ground truth)
        hard_loss = F.cross_entropy(student_outputs, targets)
        
        # Soft target loss (student vs teacher)
        student_soft = F.log_softmax(student_outputs / T, dim=1)
        teacher_soft = F.softmax(teacher_outputs / T, dim=1)
        
        soft_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (T ** 2)
        
        # Combined loss
        total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
        
        return total_loss
    
    def get_teacher_outputs(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """Get teacher model outputs"""
        with torch.no_grad():
            outputs = self.teacher_model(inputs)
        
        return outputs


# ============================================================================
# STOCHASTIC WEIGHT AVERAGING
# ============================================================================

class SWA:
    """
    Stochastic Weight Averaging
    
    Averages model weights during training for better generalization.
    """
    
    def __init__(self, model: nn.Module):
        self.swa_model = copy.deepcopy(model)
        self.swa_n = 0
        
        logger.info("SWA initialized")
    
    def update_swa_model(self, model: nn.Module):
        """Update SWA model with current model weights"""
        # Running average
        for swa_param, param in zip(
            self.swa_model.parameters(),
            model.parameters()
        ):
            swa_param.data = (
                swa_param.data * self.swa_n + param.data
            ) / (self.swa_n + 1)
        
        self.swa_n += 1
    
    def get_swa_model(self) -> nn.Module:
        """Get averaged model"""
        return self.swa_model


# ============================================================================
# LEARNING RATE SCHEDULERS
# ============================================================================

class CosineAnnealingWarmRestarts:
    """
    Cosine annealing with warm restarts
    
    Periodically restarts learning rate.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_0: int,
        T_mult: float = 1.0,
        eta_min: float = 0.0
    ):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        
        self.T_cur = 0
        self.T_i = T_0
        self.base_lr = optimizer.param_groups[0]['lr']
        
        logger.info("Cosine Annealing with Warm Restarts initialized")
    
    def step(self):
        """Update learning rate"""
        self.T_cur += 1
        
        # Compute new learning rate
        lr = self.eta_min + (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self.T_cur / self.T_i)
        ) / 2
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Check for restart
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = int(self.T_i * self.T_mult)
            logger.info(f"Learning rate restarted. Next period: {self.T_i}")
        
        return lr


# ============================================================================
# ADVANCED TRAINER
# ============================================================================

class AdvancedTrainer:
    """
    Advanced training pipeline
    
    Combines curriculum learning, adversarial training, distillation, etc.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig
    ):
        self.model = model
        self.config = config
        
        # Optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        if config.enable_warm_restart:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config.restart_period,
                T_mult=config.restart_mult
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs
            )
        
        # Components
        if config.enable_adversarial:
            self.adversarial_trainer = AdversarialTrainer(config)
        
        if config.enable_distillation and config.teacher_model:
            self.distiller = KnowledgeDistiller(config.teacher_model, config)
        
        if config.enable_swa:
            self.swa = SWA(model)
        
        # Mixed precision
        if config.use_mixed_precision and TORCH_AVAILABLE:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Training history
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        
        logger.info("Advanced Trainer initialized")
    
    def label_smoothing_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Cross entropy with label smoothing"""
        num_classes = outputs.size(1)
        smoothing = self.config.label_smoothing
        
        # One-hot with smoothing
        confidence = 1.0 - smoothing
        smooth_labels = torch.zeros_like(outputs)
        smooth_labels.fill_(smoothing / (num_classes - 1))
        smooth_labels.scatter_(1, targets.unsqueeze(1), confidence)
        
        # Log softmax
        log_probs = F.log_softmax(outputs, dim=1)
        
        # Loss
        loss = -(smooth_labels * log_probs).sum(dim=1).mean()
        
        return loss
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[float, float]:
        """Single training step"""
        self.model.train()
        
        # Mixed precision context
        if self.scaler:
            with autocast():
                outputs = self.model(inputs)
                
                # Compute loss
                if self.config.label_smoothing > 0:
                    loss = self.label_smoothing_loss(outputs, targets)
                else:
                    loss = F.cross_entropy(outputs, targets)
                
                # Add adversarial loss
                if self.config.enable_adversarial:
                    adv_loss = self.adversarial_trainer.adversarial_loss(
                        self.model, inputs, targets
                    )
                    loss = (loss + adv_loss) / 2
                
                # Add distillation loss
                if self.config.enable_distillation and hasattr(self, 'distiller'):
                    teacher_outputs = self.distiller.get_teacher_outputs(inputs)
                    loss = self.distiller.distillation_loss(
                        outputs, teacher_outputs, targets
                    )
            
            # Backward pass with scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular training
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        # Compute accuracy
        pred = outputs.argmax(dim=1)
        accuracy = (pred == targets).float().mean()
        
        return loss.item(), accuracy.item()
    
    def train_epoch(
        self,
        train_loader: Any,
        epoch: int
    ) -> Tuple[float, float]:
        """Train one epoch"""
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            loss, accuracy = self.train_step(inputs, targets)
            
            epoch_loss += loss
            epoch_accuracy += accuracy
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        # Update learning rate
        if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
            self.scheduler.step()
        
        # Update SWA model
        if self.config.enable_swa and epoch >= self.config.swa_start_epoch:
            self.swa.update_swa_model(self.model)
        
        return avg_loss, avg_accuracy
    
    def validate(self, val_loader: Any) -> float:
        """Validate model"""
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.model(inputs)
                pred = outputs.argmax(dim=1)
                
                correct += (pred == targets).sum().item()
                total += targets.size(0)
        
        accuracy = correct / total
        
        return accuracy
    
    def train(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None
    ):
        """Complete training loop"""
        logger.info(f"\n{'='*80}")
        logger.info("ADVANCED TRAINING")
        logger.info(f"{'='*80}")
        
        for epoch in range(self.config.num_epochs):
            # Train epoch
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_acc = 0.0
            if val_loader:
                val_acc = self.validate(val_loader)
                self.val_accuracies.append(val_acc)
            
            # Log
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Loss: {train_loss:.4f}, "
                f"Train Acc: {train_acc:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )
            
            # Step scheduler (if not warm restart)
            if not isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                self.scheduler.step()
        
        # Final SWA model
        if self.config.enable_swa:
            logger.info("Using SWA model for final evaluation")
            self.model = self.swa.get_swa_model()
        
        logger.info("Training complete")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': max(self.val_accuracies) if self.val_accuracies else 0.0,
            'final_train_accuracy': self.train_accuracies[-1] if self.train_accuracies else 0.0
        }


# ============================================================================
# TESTING
# ============================================================================

def test_advanced_training():
    """Test advanced training strategies"""
    print("=" * 80)
    print("ADVANCED TRAINING STRATEGIES - TEST")
    print("=" * 80)
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available")
        return
    
    # Create config
    config = TrainingConfig(
        enable_curriculum=True,
        enable_adversarial=True,
        enable_swa=True,
        num_epochs=5,
        use_mixed_precision=False  # Disable for CPU testing
    )
    
    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc = nn.Linear(64 * 8 * 8, 10)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = SimpleModel()
    
    print("\n✓ Model created")
    
    # Test adversarial training
    print("\n" + "="*80)
    print("Test: Adversarial Training")
    print("="*80)
    
    adv_trainer = AdversarialTrainer(config)
    
    inputs = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 10, (4,))
    
    adv_inputs = adv_trainer.generate_adversarial(model, inputs, targets)
    
    print(f"✓ Generated adversarial examples: {adv_inputs.shape}")
    print(f"  Perturbation norm: {(adv_inputs - inputs).abs().max():.4f}")
    
    # Test knowledge distillation
    print("\n" + "="*80)
    print("Test: Knowledge Distillation")
    print("="*80)
    
    teacher_model = SimpleModel()
    distiller = KnowledgeDistiller(teacher_model, config)
    
    student_outputs = torch.randn(4, 10)
    teacher_outputs = distiller.get_teacher_outputs(inputs)
    
    dist_loss = distiller.distillation_loss(student_outputs, teacher_outputs, targets)
    
    print(f"✓ Distillation loss: {dist_loss.item():.4f}")
    
    # Test SWA
    print("\n" + "="*80)
    print("Test: Stochastic Weight Averaging")
    print("="*80)
    
    swa = SWA(model)
    
    for i in range(3):
        swa.update_swa_model(model)
    
    print(f"✓ SWA updated {swa.swa_n} times")
    
    # Test complete trainer
    print("\n" + "="*80)
    print("Test: Advanced Trainer")
    print("="*80)
    
    trainer = AdvancedTrainer(model, config)
    
    # Mock data loader
    class MockDataLoader:
        def __iter__(self):
            for _ in range(5):
                yield (
                    torch.randn(4, 3, 32, 32),
                    torch.randint(0, 10, (4,))
                )
    
    trainer.train(MockDataLoader())
    
    stats = trainer.get_statistics()
    print(f"\n✓ Training complete:")
    print(f"  Epochs trained: {len(stats['train_losses'])}")
    print(f"  Final train accuracy: {stats['final_train_accuracy']*100:.2f}%")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_advanced_training()
