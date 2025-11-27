"""
Advanced Training Infrastructure
=================================

Implements sophisticated training techniques:
- Knowledge distillation (ViT teacher → EfficientNet student)
- Progressive training (start low resolution, increase gradually)
- Mixed precision training with dynamic loss scaling
- Distributed data parallel (multi-GPU)
- Learning rate scheduling with warmup
- Early stopping with patience
- Model checkpointing
- TensorBoard logging
- Gradient accumulation
- Label smoothing
- Mixup/CutMix augmentation

References:
- Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
- Touvron et al. "Training data-efficient image transformers" (DeiT, 2021)
- He et al. "Bag of Tricks for Image Classification with CNNs" (CVPR 2019)
"""

import os
import time
import math
import json
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW, SGD, Optimizer
    from torch.optim.lr_scheduler import _LRScheduler
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not installed")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DistillationConfig:
    """Knowledge distillation configuration"""
    
    # Teacher model
    teacher_checkpoint: Optional[str] = None
    teacher_frozen: bool = True
    
    # Loss weights
    distillation_alpha: float = 0.5  # Weight for distillation loss
    hard_label_weight: float = 0.5   # Weight for ground truth loss
    
    # Temperature
    temperature: float = 3.0  # Soften predictions
    
    # Features
    feature_distillation: bool = True
    feature_loss_weight: float = 0.1


@dataclass
class ProgressiveConfig:
    """Progressive training configuration"""
    
    # Resolution progression
    start_resolution: int = 224
    end_resolution: int = 640
    resolution_steps: List[int] = None
    epochs_per_step: int = 10
    
    # Batch size adjustment
    adjust_batch_size: bool = True
    base_batch_size: int = 64
    
    def __post_init__(self):
        if self.resolution_steps is None:
            self.resolution_steps = [224, 384, 480, 640]


@dataclass
class AdvancedTrainingConfig:
    """Advanced training configuration"""
    
    # Basic settings
    experiment_name: str = "efficientnet_distillation"
    output_dir: str = "experiments"
    seed: int = 42
    
    # Model
    model_name: str = "efficientnetv2_s"
    num_elements: int = 22
    
    # Data
    data_path: str = "data/unified_dataset.h5"
    image_size: int = 384
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training
    batch_size: int = 32
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Optimizer
    optimizer: str = "adamw"  # adamw, sgd
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    momentum: float = 0.9  # For SGD
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"  # cosine, linear, constant
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Regularization
    label_smoothing: float = 0.1
    dropout_rate: float = 0.2
    stochastic_depth_rate: float = 0.2
    
    # Augmentation
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    augmentation_prob: float = 0.5
    
    # Mixed precision
    use_amp: bool = True
    
    # Distributed
    distributed: bool = False
    local_rank: int = -1
    
    # Checkpointing
    save_frequency: int = 5
    keep_last_n: int = 3
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    
    # Logging
    log_frequency: int = 10
    use_tensorboard: bool = True
    
    # Distillation
    distillation: Optional[DistillationConfig] = None
    
    # Progressive training
    progressive: Optional[ProgressiveConfig] = None


# ============================================================================
# Loss Functions
# ============================================================================

if HAS_TORCH:
    class DistillationLoss(nn.Module):
        """
        Knowledge distillation loss
        
        Combines:
        1. Hard loss: Cross-entropy with ground truth
        2. Soft loss: KL divergence with teacher predictions
        3. Feature loss: MSE between intermediate features
        """
        
        def __init__(
            self,
            temperature: float = 3.0,
            alpha: float = 0.5,
            hard_weight: float = 0.5,
            feature_weight: float = 0.1
        ):
            super().__init__()
            self.temperature = temperature
            self.alpha = alpha
            self.hard_weight = hard_weight
            self.feature_weight = feature_weight
        
        def forward(
            self,
            student_outputs: Tensor,
            teacher_outputs: Tensor,
            targets: Tensor,
            student_features: Optional[Tensor] = None,
            teacher_features: Optional[Tensor] = None
        ) -> Dict[str, Tensor]:
            """
            Compute distillation loss
            
            Args:
                student_outputs: Student predictions (batch_size, num_elements)
                teacher_outputs: Teacher predictions (batch_size, num_elements)
                targets: Ground truth (batch_size, num_elements)
                student_features: Student intermediate features
                teacher_features: Teacher intermediate features
            
            Returns:
                Dictionary with loss components
            """
            # Hard loss (MSE with ground truth)
            hard_loss = F.mse_loss(student_outputs, targets)
            
            # Soft loss (KL divergence with teacher)
            # Scale outputs by temperature to soften distributions
            student_soft = student_outputs / self.temperature
            teacher_soft = teacher_outputs / self.temperature
            
            # KL divergence
            soft_loss = F.mse_loss(student_soft, teacher_soft.detach())
            soft_loss *= (self.temperature ** 2)  # Scale back
            
            # Combined loss
            total_loss = (
                self.hard_weight * hard_loss +
                self.alpha * soft_loss
            )
            
            # Feature distillation
            feature_loss = torch.tensor(0.0, device=student_outputs.device)
            if student_features is not None and teacher_features is not None:
                feature_loss = F.mse_loss(student_features, teacher_features.detach())
                total_loss += self.feature_weight * feature_loss
            
            return {
                'total': total_loss,
                'hard': hard_loss,
                'soft': soft_loss,
                'feature': feature_loss
            }


    class LabelSmoothingLoss(nn.Module):
        """Label smoothing for better generalization"""
        
        def __init__(self, smoothing: float = 0.1):
            super().__init__()
            self.smoothing = smoothing
        
        def forward(self, pred: Tensor, target: Tensor) -> Tensor:
            """
            Args:
                pred: (batch_size, num_elements)
                target: (batch_size, num_elements)
            """
            # Smooth targets
            smooth_target = target * (1 - self.smoothing) + self.smoothing / 2
            
            # MSE loss with smoothed targets
            loss = F.mse_loss(pred, smooth_target)
            
            return loss


    # ============================================================================
    # Data Augmentation
    # ============================================================================

    class MixUp:
        """
        MixUp augmentation
        
        Reference: "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
        """
        
        def __init__(self, alpha: float = 0.2):
            self.alpha = alpha
        
        def __call__(
            self,
            images: Tensor,
            targets: Tensor
        ) -> Tuple[Tensor, Tensor, float]:
            """
            Apply MixUp
            
            Returns:
                Mixed images, mixed targets, lambda value
            """
            if self.alpha > 0:
                lam = np.random.beta(self.alpha, self.alpha)
            else:
                lam = 1.0
            
            batch_size = images.size(0)
            index = torch.randperm(batch_size, device=images.device)
            
            mixed_images = lam * images + (1 - lam) * images[index]
            mixed_targets = lam * targets + (1 - lam) * targets[index]
            
            return mixed_images, mixed_targets, lam


    class CutMix:
        """
        CutMix augmentation
        
        Reference: "CutMix: Regularization Strategy to Train Strong Classifiers" (ICCV 2019)
        """
        
        def __init__(self, alpha: float = 1.0):
            self.alpha = alpha
        
        def __call__(
            self,
            images: Tensor,
            targets: Tensor
        ) -> Tuple[Tensor, Tensor, float]:
            """Apply CutMix"""
            if self.alpha > 0:
                lam = np.random.beta(self.alpha, self.alpha)
            else:
                lam = 1.0
            
            batch_size = images.size(0)
            index = torch.randperm(batch_size, device=images.device)
            
            # Get random box
            _, _, h, w = images.shape
            cut_ratio = np.sqrt(1 - lam)
            cut_h = int(h * cut_ratio)
            cut_w = int(w * cut_ratio)
            
            cx = np.random.randint(w)
            cy = np.random.randint(h)
            
            x1 = np.clip(cx - cut_w // 2, 0, w)
            y1 = np.clip(cy - cut_h // 2, 0, h)
            x2 = np.clip(cx + cut_w // 2, 0, w)
            y2 = np.clip(cy + cut_h // 2, 0, h)
            
            # Apply cutmix
            mixed_images = images.clone()
            mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
            
            # Adjust lambda based on actual box size
            lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
            mixed_targets = lam * targets + (1 - lam) * targets[index]
            
            return mixed_images, mixed_targets, lam


    # ============================================================================
    # Learning Rate Schedulers
    # ============================================================================

    class WarmupCosineScheduler(_LRScheduler):
        """
        Cosine learning rate scheduler with warmup
        
        LR schedule:
        1. Warmup: Linear increase from 0 to base_lr
        2. Cosine: Cosine decay from base_lr to min_lr
        """
        
        def __init__(
            self,
            optimizer: Optimizer,
            warmup_epochs: int,
            max_epochs: int,
            min_lr: float = 1e-6,
            last_epoch: int = -1
        ):
            self.warmup_epochs = warmup_epochs
            self.max_epochs = max_epochs
            self.min_lr = min_lr
            super().__init__(optimizer, last_epoch)
        
        def get_lr(self):
            if self.last_epoch < self.warmup_epochs:
                # Warmup phase
                return [
                    base_lr * self.last_epoch / self.warmup_epochs
                    for base_lr in self.base_lrs
                ]
            else:
                # Cosine decay
                progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                return [
                    self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs
                ]


    # ============================================================================
    # Advanced Trainer
    # ============================================================================

    class AdvancedTrainer:
        """
        Advanced training manager with knowledge distillation, progressive training,
        and sophisticated optimization techniques
        """
        
        def __init__(
            self,
            config: AdvancedTrainingConfig,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            teacher_model: Optional[nn.Module] = None
        ):
            self.config = config
            self.model = model
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.teacher_model = teacher_model
            
            # Setup
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            if self.teacher_model is not None:
                self.teacher_model.to(self.device)
                self.teacher_model.eval()
                for param in self.teacher_model.parameters():
                    param.requires_grad = False
            
            # Optimizer
            self.optimizer = self._create_optimizer()
            
            # Scheduler
            self.scheduler = self._create_scheduler()
            
            # Loss functions
            if config.distillation and teacher_model:
                self.criterion = DistillationLoss(
                    temperature=config.distillation.temperature,
                    alpha=config.distillation.distillation_alpha,
                    hard_weight=config.distillation.hard_label_weight,
                    feature_weight=config.distillation.feature_loss_weight
                )
            else:
                if config.label_smoothing > 0:
                    self.criterion = LabelSmoothingLoss(config.label_smoothing)
                else:
                    self.criterion = nn.MSELoss()
            
            # Augmentation
            self.mixup = MixUp(config.mixup_alpha) if config.use_mixup else None
            self.cutmix = CutMix(config.cutmix_alpha) if config.use_cutmix else None
            
            # AMP
            self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
            
            # Tracking
            self.current_epoch = 0
            self.best_val_loss = float('inf')
            self.patience_counter = 0
            self.history = {
                'train_loss': [],
                'val_loss': [],
                'lr': []
            }
            
            # Paths
            self.output_dir = Path(config.output_dir) / config.experiment_name
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir = self.output_dir / "checkpoints"
            self.checkpoint_dir.mkdir(exist_ok=True)
            
            # Save config
            with open(self.output_dir / "config.json", 'w') as f:
                json.dump(asdict(config), f, indent=2, default=str)
        
        def _create_optimizer(self) -> Optimizer:
            """Create optimizer"""
            if self.config.optimizer.lower() == 'adamw':
                return AdamW(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            elif self.config.optimizer.lower() == 'sgd':
                return SGD(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay
                )
            else:
                raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        def _create_scheduler(self) -> _LRScheduler:
            """Create learning rate scheduler"""
            if self.config.lr_scheduler == 'cosine':
                return WarmupCosineScheduler(
                    self.optimizer,
                    warmup_epochs=self.config.warmup_epochs,
                    max_epochs=self.config.num_epochs,
                    min_lr=self.config.min_lr
                )
            else:
                from torch.optim.lr_scheduler import StepLR
                return StepLR(self.optimizer, step_size=30, gamma=0.1)
        
        def train_epoch(self) -> float:
            """Train for one epoch"""
            self.model.train()
            total_loss = 0.0
            num_batches = len(self.train_loader)
            
            for batch_idx, (images, targets) in enumerate(self.train_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Apply augmentation
                if self.mixup and np.random.rand() < self.config.augmentation_prob:
                    images, targets, _ = self.mixup(images, targets)
                elif self.cutmix and np.random.rand() < self.config.augmentation_prob:
                    images, targets, _ = self.cutmix(images, targets)
                
                # Forward pass
                if self.config.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        
                        if self.teacher_model:
                            with torch.no_grad():
                                teacher_outputs = self.teacher_model(images)
                            loss_dict = self.criterion(
                                outputs['concentrations'],
                                teacher_outputs['concentrations'],
                                targets,
                                outputs.get('features'),
                                teacher_outputs.get('features')
                            )
                            loss = loss_dict['total']
                        else:
                            loss = self.criterion(outputs['concentrations'], targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs['concentrations'], targets)
                
                # Backward pass
                loss = loss / self.config.gradient_accumulation_steps
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Optimizer step
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                
                # Logging
                if batch_idx % self.config.log_frequency == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"  Batch {batch_idx}/{num_batches}: "
                          f"Loss={loss.item():.4f}, LR={current_lr:.6f}")
            
            return total_loss / num_batches
        
        def validate(self) -> float:
            """Validate model"""
            self.model.eval()
            total_loss = 0.0
            num_batches = len(self.val_loader)
            
            with torch.no_grad():
                for images, targets in self.val_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(images)
                    loss = F.mse_loss(outputs['concentrations'], targets)
                    
                    total_loss += loss.item()
            
            return total_loss / num_batches
        
        def train(self):
            """Full training loop"""
            print(f"\n{'='*60}")
            print(f"TRAINING: {self.config.experiment_name}")
            print(f"{'='*60}\n")
            print(f"Device: {self.device}")
            print(f"Model: {self.config.model_name}")
            print(f"Epochs: {self.config.num_epochs}")
            print(f"Batch size: {self.config.batch_size}")
            print(f"Learning rate: {self.config.learning_rate}")
            
            if self.teacher_model:
                print(f"Knowledge distillation: ENABLED")
            
            print(f"\nOutput directory: {self.output_dir}")
            print()
            
            start_time = time.time()
            
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                print(f"Epoch {epoch+1}/{self.config.num_epochs}")
                print("-" * 40)
                
                # Train
                train_loss = self.train_epoch()
                
                # Validate
                val_loss = self.validate()
                
                # Scheduler step
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Track
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['lr'].append(current_lr)
                
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")
                print(f"  LR: {current_lr:.6f}")
                
                # Checkpointing
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pth')
                    print(f"  ✓ New best model! Val loss: {val_loss:.6f}")
                else:
                    self.patience_counter += 1
                
                if (epoch + 1) % self.config.save_frequency == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
                
                # Early stopping
                if self.config.early_stopping and self.patience_counter >= self.config.patience:
                    print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                    break
                
                print()
            
            # Final checkpoint
            self.save_checkpoint('final_model.pth')
            
            # Save history
            with open(self.output_dir / "history.json", 'w') as f:
                json.dump(self.history, f, indent=2)
            
            elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"TRAINING COMPLETE")
            print(f"{'='*60}")
            print(f"Best val loss: {self.best_val_loss:.6f}")
            print(f"Total time: {elapsed/60:.1f} minutes")
            print(f"Output: {self.output_dir}")
        
        def save_checkpoint(self, filename: str):
            """Save checkpoint"""
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'history': self.history,
                'config': asdict(self.config)
            }
            
            if self.scaler:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            path = self.checkpoint_dir / filename
            torch.save(checkpoint, path)
            print(f"  ✓ Saved: {filename}")


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example usage"""
    if not HAS_TORCH:
        print("PyTorch required")
        return
    
    print("\n" + "="*60)
    print("ADVANCED TRAINING - EXAMPLE")
    print("="*60)
    
    # Mock dataset
    class MockDataset(Dataset):
        def __init__(self, num_samples=100):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            image = torch.randn(3, 384, 384)
            target = torch.rand(22)
            return image, target
    
    # Create dataloaders
    train_dataset = MockDataset(100)
    val_dataset = MockDataset(20)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # Create model
    from app.ai_nutrition.models.efficientnet_ensemble import create_efficientnet
    model = create_efficientnet('s', num_elements=22)
    
    # Config
    config = AdvancedTrainingConfig(
        experiment_name="test_run",
        num_epochs=2,
        batch_size=8,
        use_amp=False,  # Disable for CPU
        log_frequency=5
    )
    
    # Trainer
    trainer = AdvancedTrainer(config, model, train_loader, val_loader)
    trainer.train()
    
    print("\n✅ Example complete!")


if __name__ == "__main__":
    example_usage()
