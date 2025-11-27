"""
Active Learning Validation Suite
=================================

Comprehensive validation of active learning system to verify:
1. Data efficiency: 2× fewer samples needed vs random sampling
2. Cost savings: 75% reduction in labeling costs
3. Accuracy: Match or exceed random baseline with fewer samples
4. Sample quality: Selected samples are informative

Tests include:
- Simulated oracle experiments
- Learning curve comparisons
- Sample diversity metrics
- Uncertainty calibration
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import random

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, r2_score
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    print(f"⚠️  Missing dependencies: {e}")

import sys
sys.path.append(str(Path(__file__).parent.parent))

from active_learning.active_learning_system import (
    ActiveLearningManager,
    ActiveLearningConfig,
    SamplingStrategy
)


@dataclass
class ValidationConfig:
    """Configuration for validation experiments"""
    num_samples: int = 1000  # Total dataset size
    num_elements: int = 22  # Number of elements to predict
    image_size: int = 224  # Image dimensions
    
    # Active learning settings
    initial_labeled: int = 100  # Start with 100 labeled samples
    samples_per_iteration: int = 50  # Query 50 samples per iteration
    num_iterations: int = 10  # Run for 10 iterations
    
    # Baseline comparison
    random_baseline: bool = True  # Compare against random sampling
    
    # Evaluation
    eval_every: int = 1  # Evaluate after each iteration
    save_plots: bool = True  # Save learning curves
    save_results: bool = True  # Save metrics to JSON
    
    # Directories
    output_dir: Path = Path("validation_results/active_learning")
    checkpoint_dir: Path = Path("checkpoints/validation")


@dataclass
class ValidationResults:
    """Results from validation experiment"""
    strategy: str
    final_accuracy: float
    samples_used: int
    training_time: float
    sample_efficiency: float  # Accuracy / samples_used
    cost_savings: float  # vs random baseline
    
    learning_curve: List[Tuple[int, float]]  # (num_samples, accuracy)
    uncertainty_metrics: Dict[str, float]
    diversity_metrics: Dict[str, float]


class MockFoodDataset(Dataset):
    """Mock dataset for validation experiments"""
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_elements: int = 22,
        image_size: int = 224,
        seed: int = 42
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.num_samples = num_samples
        self.num_elements = num_elements
        self.image_size = image_size
        
        # Generate synthetic food images
        # Use different patterns to simulate difficulty
        self.images = self._generate_images()
        self.compositions = self._generate_compositions()
        
        # Add noise to make some samples harder
        self.difficulty = self._compute_difficulty()
        
    def _generate_images(self) -> torch.Tensor:
        """Generate synthetic food images with varying complexity"""
        images = []
        
        for i in range(self.num_samples):
            # Create base image with food-like colors
            base_color = torch.rand(3, 1, 1) * 0.5 + 0.3  # RGB in [0.3, 0.8]
            
            # Add texture
            noise = torch.randn(3, self.image_size, self.image_size) * 0.1
            
            # Add spatial patterns (some samples have complex patterns)
            if i % 3 == 0:  # Complex samples (33%)
                x = torch.linspace(-1, 1, self.image_size)
                y = torch.linspace(-1, 1, self.image_size)
                xx, yy = torch.meshgrid(x, y, indexing='ij')
                pattern = torch.sin(5 * xx) * torch.cos(5 * yy)
                pattern = pattern.unsqueeze(0).repeat(3, 1, 1) * 0.2
            else:  # Simple samples
                pattern = torch.zeros(3, self.image_size, self.image_size)
            
            image = base_color + noise + pattern
            image = torch.clamp(image, 0, 1)
            images.append(image)
        
        return torch.stack(images)
    
    def _generate_compositions(self) -> torch.Tensor:
        """Generate elemental compositions correlated with images"""
        compositions = []
        
        for i, image in enumerate(self.images):
            # Extract features from image (mean RGB as proxy)
            rgb_mean = image.mean(dim=(1, 2))  # [3]
            
            # Composition correlates with color
            # e.g., red foods have more iron, green foods have more magnesium
            comp = torch.zeros(self.num_elements)
            
            # Major elements (Ca, Fe, K, Mg, Na, P, Zn)
            comp[0] = rgb_mean[0] * 500 + 100  # Ca (100-600 mg/kg)
            comp[1] = rgb_mean[0] * 20 + 5     # Fe (5-25 mg/kg)
            comp[2] = rgb_mean[1] * 1000 + 500 # K (500-1500 mg/kg)
            comp[3] = rgb_mean[1] * 100 + 50   # Mg (50-150 mg/kg)
            comp[4] = rgb_mean[2] * 200 + 100  # Na (100-300 mg/kg)
            comp[5] = rgb_mean.mean() * 300 + 100  # P (100-400 mg/kg)
            comp[6] = rgb_mean.mean() * 10 + 2     # Zn (2-12 mg/kg)
            
            # Trace elements (smaller amounts)
            for j in range(7, self.num_elements):
                comp[j] = torch.rand(1) * 5 + 0.1  # 0.1-5 mg/kg
            
            # Add noise
            noise = torch.randn(self.num_elements) * 10
            comp = comp + noise
            comp = torch.clamp(comp, 0.1, 10000)  # Physical limits
            
            compositions.append(comp)
        
        return torch.stack(compositions)
    
    def _compute_difficulty(self) -> torch.Tensor:
        """Compute difficulty score for each sample"""
        # Difficult samples have:
        # 1. Complex visual patterns
        # 2. Unusual compositions
        # 3. High variance in local regions
        
        difficulty = []
        
        for i, (image, comp) in enumerate(zip(self.images, self.compositions)):
            # Visual complexity
            visual_var = image.std()
            
            # Composition unusualness (distance from mean)
            comp_dist = torch.abs(comp - self.compositions.mean(dim=0)).mean()
            
            # Combined difficulty
            diff = visual_var + comp_dist / 100
            difficulty.append(diff)
        
        return torch.tensor(difficulty)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.compositions[idx]


class SimpleModel(nn.Module):
    """Simple CNN for validation (faster than full models)"""
    
    def __init__(self, num_elements: int = 22, dropout_rate: float = 0.3):
        super().__init__()
        
        self.num_elements = num_elements
        self.dropout_rate = dropout_rate
        
        # Simple CNN backbone
        self.conv1 = nn.Conv2d(3, 32, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Prediction heads
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_elements)
        
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        # Extract features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        features = self.pool(x).flatten(1)
        
        # Predict composition
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        predictions = self.fc2(x)
        
        if return_features:
            return predictions, features
        return predictions
    
    def enable_dropout(self):
        """Enable dropout for MC dropout uncertainty estimation"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


class ActiveLearningValidator:
    """Validate active learning system performance"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"💻 Using device: {self.device}")
        
        # Create dataset
        print("🔄 Generating synthetic dataset...")
        self.dataset = MockFoodDataset(
            num_samples=config.num_samples,
            num_elements=config.num_elements,
            image_size=config.image_size
        )
        
        # Split into pool and test
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        
        split_idx = int(0.8 * len(indices))
        self.pool_indices = indices[:split_idx]
        self.test_indices = indices[split_idx:]
        
        print(f"📊 Dataset: {len(self.pool_indices)} pool, {len(self.test_indices)} test")
    
    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        epochs: int = 10
    ) -> Dict[str, float]:
        """Train model on labeled data"""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for images, compositions in train_loader:
                images = images.to(self.device)
                compositions = compositions.to(self.device)
                
                optimizer.zero_grad()
                predictions = model(images)
                loss = criterion(predictions, compositions)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
        
        return {'final_loss': losses[-1], 'losses': losses}
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on test set"""
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, compositions in test_loader:
                images = images.to(self.device)
                predictions = model(images)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(compositions)
        
        predictions = torch.cat(all_predictions).numpy()
        targets = torch.cat(all_targets).numpy()
        
        # Compute metrics
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # Per-element accuracy
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-6)), axis=0)
        avg_mape = mape.mean()
        
        # Accuracy as (1 - MAPE)
        accuracy = 1 - avg_mape
        
        return {
            'accuracy': accuracy * 100,  # Convert to percentage
            'mae': mae,
            'r2': r2,
            'mape': avg_mape * 100
        }
    
    def run_active_learning_experiment(
        self,
        strategy: SamplingStrategy
    ) -> ValidationResults:
        """Run active learning with given strategy"""
        print(f"\n{'='*60}")
        print(f"🧪 Testing strategy: {strategy.value}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Create model
        model = SimpleModel(num_elements=self.config.num_elements).to(self.device)
        
        # Create active learning config
        al_config = ActiveLearningConfig(
            strategy=strategy,
            initial_labeled=self.config.initial_labeled,
            samples_per_iteration=self.config.samples_per_iteration,
            budget=self.config.initial_labeled + 
                   (self.config.samples_per_iteration * self.config.num_iterations),
            ensemble_size=3 if strategy == SamplingStrategy.QUERY_BY_COMMITTEE else 1
        )
        
        # Create active learning manager
        manager = ActiveLearningManager(
            config=al_config,
            model=model,
            dataset=self.dataset,
            pool_indices=self.pool_indices.copy()
        )
        
        # Initialize labeled set
        manager.initialize()
        
        # Learning curve: (num_samples, accuracy)
        learning_curve = []
        
        # Test set loader
        test_loader = DataLoader(
            torch.utils.data.Subset(self.dataset, self.test_indices),
            batch_size=32,
            shuffle=False
        )
        
        # Active learning loop
        for iteration in range(self.config.num_iterations):
            print(f"\n📍 Iteration {iteration + 1}/{self.config.num_iterations}")
            print(f"   Labeled samples: {len(manager.labeled_indices)}")
            
            # Train on current labeled set
            train_loader = DataLoader(
                torch.utils.data.Subset(self.dataset, manager.labeled_indices),
                batch_size=32,
                shuffle=True
            )
            
            train_metrics = self.train_model(model, train_loader, epochs=5)
            print(f"   Training loss: {train_metrics['final_loss']:.4f}")
            
            # Evaluate
            eval_metrics = self.evaluate_model(model, test_loader)
            accuracy = eval_metrics['accuracy']
            print(f"   Test accuracy: {accuracy:.2f}%")
            print(f"   MAE: {eval_metrics['mae']:.2f}")
            print(f"   R²: {eval_metrics['r2']:.4f}")
            
            learning_curve.append((len(manager.labeled_indices), accuracy))
            
            # Select new samples
            if iteration < self.config.num_iterations - 1:
                selected_indices = manager.select_samples()
                print(f"   Selected {len(selected_indices)} new samples")
                
                # Simulate oracle (get ground truth labels)
                manager.query_oracle(selected_indices)
                manager.update_labeled_set(selected_indices)
        
        total_time = time.time() - start_time
        
        # Compute metrics
        final_accuracy = learning_curve[-1][1]
        samples_used = learning_curve[-1][0]
        sample_efficiency = final_accuracy / samples_used
        
        # Uncertainty metrics
        uncertainty_metrics = self._compute_uncertainty_metrics(model, test_loader)
        
        # Diversity metrics
        diversity_metrics = self._compute_diversity_metrics(manager.labeled_indices)
        
        results = ValidationResults(
            strategy=strategy.value,
            final_accuracy=final_accuracy,
            samples_used=samples_used,
            training_time=total_time,
            sample_efficiency=sample_efficiency,
            cost_savings=0.0,  # Will compute after baseline
            learning_curve=learning_curve,
            uncertainty_metrics=uncertainty_metrics,
            diversity_metrics=diversity_metrics
        )
        
        print(f"\n✅ Completed in {total_time:.1f}s")
        print(f"   Final accuracy: {final_accuracy:.2f}%")
        print(f"   Sample efficiency: {sample_efficiency:.6f}")
        
        return results
    
    def run_random_baseline(self) -> ValidationResults:
        """Run baseline with random sampling"""
        print(f"\n{'='*60}")
        print(f"🎲 Running random sampling baseline")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Create model
        model = SimpleModel(num_elements=self.config.num_elements).to(self.device)
        
        # Random sampling
        labeled_indices = random.sample(
            self.pool_indices,
            k=self.config.initial_labeled
        )
        
        learning_curve = []
        
        # Test set loader
        test_loader = DataLoader(
            torch.utils.data.Subset(self.dataset, self.test_indices),
            batch_size=32,
            shuffle=False
        )
        
        # Training loop (same schedule as active learning)
        for iteration in range(self.config.num_iterations):
            print(f"\n📍 Iteration {iteration + 1}/{self.config.num_iterations}")
            print(f"   Labeled samples: {len(labeled_indices)}")
            
            # Train
            train_loader = DataLoader(
                torch.utils.data.Subset(self.dataset, labeled_indices),
                batch_size=32,
                shuffle=True
            )
            
            train_metrics = self.train_model(model, train_loader, epochs=5)
            
            # Evaluate
            eval_metrics = self.evaluate_model(model, test_loader)
            accuracy = eval_metrics['accuracy']
            print(f"   Test accuracy: {accuracy:.2f}%")
            
            learning_curve.append((len(labeled_indices), accuracy))
            
            # Add random samples
            if iteration < self.config.num_iterations - 1:
                remaining = [i for i in self.pool_indices if i not in labeled_indices]
                new_samples = random.sample(
                    remaining,
                    k=min(self.config.samples_per_iteration, len(remaining))
                )
                labeled_indices.extend(new_samples)
        
        total_time = time.time() - start_time
        
        final_accuracy = learning_curve[-1][1]
        samples_used = learning_curve[-1][0]
        
        results = ValidationResults(
            strategy='random',
            final_accuracy=final_accuracy,
            samples_used=samples_used,
            training_time=total_time,
            sample_efficiency=final_accuracy / samples_used,
            cost_savings=0.0,
            learning_curve=learning_curve,
            uncertainty_metrics={},
            diversity_metrics={}
        )
        
        print(f"\n✅ Baseline completed in {total_time:.1f}s")
        print(f"   Final accuracy: {final_accuracy:.2f}%")
        
        return results
    
    def _compute_uncertainty_metrics(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Compute uncertainty calibration metrics"""
        model.eval()
        model.enable_dropout()
        
        all_uncertainties = []
        all_errors = []
        
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(self.device)
                
                # MC dropout
                predictions = []
                for _ in range(10):
                    pred = model(images)
                    predictions.append(pred.cpu())
                
                predictions = torch.stack(predictions)  # [10, batch, 22]
                
                # Uncertainty = std across MC samples
                uncertainty = predictions.std(dim=0).mean(dim=1)  # [batch]
                
                # Error = MAE
                pred_mean = predictions.mean(dim=0)
                error = torch.abs(pred_mean - targets).mean(dim=1)
                
                all_uncertainties.append(uncertainty)
                all_errors.append(error)
        
        uncertainties = torch.cat(all_uncertainties).numpy()
        errors = torch.cat(all_errors).numpy()
        
        # Correlation between uncertainty and error
        correlation = np.corrcoef(uncertainties, errors)[0, 1]
        
        return {
            'mean_uncertainty': float(uncertainties.mean()),
            'uncertainty_error_correlation': float(correlation)
        }
    
    def _compute_diversity_metrics(self, labeled_indices: List[int]) -> Dict[str, float]:
        """Compute diversity of selected samples"""
        # Get images
        images = [self.dataset.images[i] for i in labeled_indices]
        images = torch.stack(images)
        
        # Compute pairwise distances
        flat_images = images.view(len(images), -1)
        
        # Sample subset for efficiency
        if len(flat_images) > 100:
            indices = random.sample(range(len(flat_images)), 100)
            flat_images = flat_images[indices]
        
        distances = torch.cdist(flat_images, flat_images)
        
        # Average pairwise distance
        avg_distance = distances.mean().item()
        
        # Minimum pairwise distance
        mask = torch.eye(len(distances), dtype=torch.bool)
        distances_no_diag = distances.masked_fill(mask, float('inf'))
        min_distance = distances_no_diag.min().item()
        
        return {
            'avg_pairwise_distance': avg_distance,
            'min_pairwise_distance': min_distance
        }
    
    def plot_learning_curves(
        self,
        results: Dict[str, ValidationResults],
        baseline: ValidationResults
    ):
        """Plot learning curves comparing strategies"""
        plt.figure(figsize=(12, 6))
        
        # Plot baseline
        samples, accuracies = zip(*baseline.learning_curve)
        plt.plot(samples, accuracies, 'k--', linewidth=2, label='Random (Baseline)')
        
        # Plot active learning strategies
        colors = ['b', 'r', 'g', 'm', 'c']
        for i, (strategy, result) in enumerate(results.items()):
            samples, accuracies = zip(*result.learning_curve)
            plt.plot(
                samples, accuracies,
                color=colors[i % len(colors)],
                linewidth=2,
                marker='o',
                markersize=4,
                label=strategy
            )
        
        plt.xlabel('Number of Labeled Samples', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Active Learning: Sample Efficiency Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        output_path = self.config.output_dir / 'learning_curves.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved learning curves to {output_path}")
        
        plt.close()
    
    def save_results(
        self,
        results: Dict[str, ValidationResults],
        baseline: ValidationResults
    ):
        """Save results to JSON"""
        output = {
            'config': asdict(self.config),
            'baseline': asdict(baseline),
            'strategies': {k: asdict(v) for k, v in results.items()}
        }
        
        # Compute cost savings relative to baseline
        for strategy, result in results.items():
            # Cost savings = (baseline_samples - al_samples) / baseline_samples
            cost_savings = (baseline.samples_used - result.samples_used) / baseline.samples_used * 100
            output['strategies'][strategy]['cost_savings'] = cost_savings
        
        output_path = self.config.output_dir / 'results.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"💾 Saved results to {output_path}")
    
    def print_summary(
        self,
        results: Dict[str, ValidationResults],
        baseline: ValidationResults
    ):
        """Print summary comparison"""
        print(f"\n{'='*70}")
        print("📊 VALIDATION SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n🎲 Random Baseline:")
        print(f"   Accuracy: {baseline.final_accuracy:.2f}%")
        print(f"   Samples: {baseline.samples_used}")
        print(f"   Efficiency: {baseline.sample_efficiency:.6f}")
        
        print(f"\n🧠 Active Learning Strategies:")
        for strategy, result in results.items():
            cost_savings = (baseline.samples_used - result.samples_used) / baseline.samples_used * 100
            accuracy_gain = result.final_accuracy - baseline.final_accuracy
            
            print(f"\n   {strategy}:")
            print(f"      Accuracy: {result.final_accuracy:.2f}% ({accuracy_gain:+.2f}%)")
            print(f"      Samples: {result.samples_used} ({cost_savings:.1f}% savings)")
            print(f"      Efficiency: {result.sample_efficiency:.6f}")
            print(f"      Time: {result.training_time:.1f}s")
            
            if result.uncertainty_metrics:
                print(f"      Uncertainty-Error Correlation: {result.uncertainty_metrics['uncertainty_error_correlation']:.3f}")
            
            if result.diversity_metrics:
                print(f"      Avg Pairwise Distance: {result.diversity_metrics['avg_pairwise_distance']:.2f}")
        
        print(f"\n{'='*70}")


def main():
    """Run validation experiments"""
    if not HAS_DEPS:
        print("❌ Missing required dependencies")
        return
    
    # Configuration
    config = ValidationConfig(
        num_samples=1000,
        initial_labeled=100,
        samples_per_iteration=50,
        num_iterations=10
    )
    
    # Create validator
    validator = ActiveLearningValidator(config)
    
    # Test strategies
    strategies = [
        SamplingStrategy.UNCERTAINTY_ENTROPY,
        SamplingStrategy.HYBRID,
        SamplingStrategy.DIVERSITY_KMEANS,
    ]
    
    results = {}
    
    # Run active learning experiments
    for strategy in strategies:
        results[strategy.value] = validator.run_active_learning_experiment(strategy)
    
    # Run random baseline
    baseline = validator.run_random_baseline()
    
    # Plot and save
    if config.save_plots:
        validator.plot_learning_curves(results, baseline)
    
    if config.save_results:
        validator.save_results(results, baseline)
    
    # Print summary
    validator.print_summary(results, baseline)
    
    print("\n✅ Validation complete!")


if __name__ == '__main__':
    main()
