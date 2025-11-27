"""
PHASE 2: Simulated Annealing Optimization for Visual-to-Atomic Modeling
========================================================================

This module implements the Simulated Annealing (SA) optimization framework
for solving the inverse problem: reconstructing atomic composition from
cooked food visual features.

SA is ideal for this non-convex, high-dimensional optimization problem where:
- State Space: Atomic composition vector [Fe, Zn, Pb, Hg, Na, K, ...]
- Cost Function: E(S) = α·E_visual + β·E_database + γ·E_constraint
- Global Optimization: Find S* that minimizes E(S)

The annealing process allows escaping local minima by accepting worse
solutions at high temperatures, then gradually converging to the global
optimum as temperature decreases.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import euclidean, cosine
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoolingSchedule(Enum):
    """Temperature cooling schedules for simulated annealing"""
    EXPONENTIAL = "exponential"  # T_new = α * T_old
    LINEAR = "linear"  # T_new = T_old - α
    LOGARITHMIC = "logarithmic"  # T_new = T_0 / log(1 + k)
    ADAPTIVE = "adaptive"  # Adjust based on acceptance rate
    FAST = "fast"  # T_new = T_0 / (1 + k)


class PerturbationType(Enum):
    """Types of state perturbation strategies"""
    GAUSSIAN = "gaussian"  # Add Gaussian noise
    UNIFORM = "uniform"  # Add uniform noise
    CAUCHY = "cauchy"  # Add Cauchy noise (heavier tails)
    GRADIENT_GUIDED = "gradient_guided"  # Use gradient information
    HYBRID = "hybrid"  # Combine multiple strategies


@dataclass
class AtomicState:
    """
    Represents a state in the atomic composition space
    
    State S = [A_Fe, A_Zn, A_Pb, A_Hg, A_Na, A_K, ...]
    where A_x is the concentration of element x in ppm or mg/100g
    """
    # Elemental concentrations (ppm or mg/100g)
    concentrations: Dict[str, float]
    
    # Associated metadata
    food_type: str = ""
    cooking_method: str = ""
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    # Cost function components
    visual_error: float = float('inf')
    database_error: float = float('inf')
    constraint_penalty: float = float('inf')
    total_cost: float = float('inf')
    
    def to_vector(self, element_order: List[str]) -> np.ndarray:
        """Convert state to numpy vector"""
        return np.array([self.concentrations.get(e, 0.0) for e in element_order])
    
    @classmethod
    def from_vector(
        cls,
        vector: np.ndarray,
        element_order: List[str],
        **kwargs
    ) -> 'AtomicState':
        """Create state from numpy vector"""
        concentrations = {e: float(v) for e, v in zip(element_order, vector)}
        return cls(concentrations=concentrations, **kwargs)
    
    def copy(self) -> 'AtomicState':
        """Create a deep copy of the state"""
        return AtomicState(
            concentrations=self.concentrations.copy(),
            food_type=self.food_type,
            cooking_method=self.cooking_method,
            confidence_scores=self.confidence_scores.copy(),
            visual_error=self.visual_error,
            database_error=self.database_error,
            constraint_penalty=self.constraint_penalty,
            total_cost=self.total_cost
        )


@dataclass
class SAParameters:
    """Simulated Annealing hyperparameters"""
    initial_temperature: float = 1000.0
    final_temperature: float = 0.01
    cooling_rate: float = 0.95
    max_iterations: int = 10000
    iterations_per_temp: int = 100
    cooling_schedule: CoolingSchedule = CoolingSchedule.EXPONENTIAL
    perturbation_type: PerturbationType = PerturbationType.HYBRID
    perturbation_scale: float = 0.1
    acceptance_threshold: float = 1e-6
    adaptive_cooling: bool = True
    reheat_temperature: float = 100.0
    reheat_frequency: int = 1000


@dataclass
class CostFunctionWeights:
    """Weights for cost function components"""
    alpha: float = 0.4  # Weight for visual error
    beta: float = 0.4   # Weight for database error
    gamma: float = 0.2  # Weight for constraint penalty


class CostFunctionCalculator:
    """
    Calculates the cost function E(S) for a given atomic state.
    
    E(S) = α·E_visual(S) + β·E_database(S) + γ·E_constraint(S)
    
    where:
    - E_visual: Error between predicted visual features and actual image
    - E_database: Error between predicted composition and ICP-MS database
    - E_constraint: Penalty for violating physical/chemical constraints
    """
    
    def __init__(
        self,
        weights: Optional[CostFunctionWeights] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.weights = weights or CostFunctionWeights()
        self.device = device
        
        # Initialize neural network for visual prediction
        self.visual_predictor = VisualFeaturePredictor().to(device)
        
        # Database lookup
        self.database = None  # Will be set externally
        
        # Constraint definitions
        self.constraints = self._initialize_constraints()
        
        logger.info("CostFunctionCalculator initialized")
    
    def compute_total_cost(
        self,
        state: AtomicState,
        target_features: Dict[str, Any],
        temperature: float = 1.0
    ) -> float:
        """
        Compute total cost E(S) for the given state
        
        Args:
            state: Current atomic state
            target_features: Target visual features from image
            temperature: Current temperature (for adaptive weighting)
            
        Returns:
            Total cost value
        """
        # Compute individual cost components
        visual_error = self._compute_visual_error(state, target_features)
        database_error = self._compute_database_error(state)
        constraint_penalty = self._compute_constraint_penalty(state)
        
        # Update state with component costs
        state.visual_error = visual_error
        state.database_error = database_error
        state.constraint_penalty = constraint_penalty
        
        # Weighted combination
        total_cost = (
            self.weights.alpha * visual_error +
            self.weights.beta * database_error +
            self.weights.gamma * constraint_penalty
        )
        
        state.total_cost = total_cost
        
        return total_cost
    
    def _compute_visual_error(
        self,
        state: AtomicState,
        target_features: Dict[str, Any]
    ) -> float:
        """
        Compute error between predicted visual features from state S
        and actual visual features from image
        
        E_visual(S) = ||F_predicted(S) - F_actual||²
        """
        # Convert state to tensor
        state_vector = torch.tensor(
            list(state.concentrations.values()),
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        
        # Predict visual features from atomic composition
        with torch.no_grad():
            predicted_features = self.visual_predictor(state_vector)
        
        # Extract target features
        target_spectral = torch.tensor(
            target_features['spectral_features'].simulated_spectrum,
            dtype=torch.float32,
            device=self.device
        )
        
        target_color = torch.tensor(
            np.array([
                target_features['chemometric_features'].glossiness_score,
                target_features['chemometric_features'].surface_roughness,
                target_features['chemometric_features'].moisture_retention,
            ]),
            dtype=torch.float32,
            device=self.device
        )
        
        # Compute errors
        spectral_error = F.mse_loss(
            predicted_features['spectrum'],
            target_spectral
        )
        
        color_error = F.mse_loss(
            predicted_features['color_properties'],
            target_color
        )
        
        # Combine errors
        visual_error = spectral_error + color_error
        
        return visual_error.item()
    
    def _compute_database_error(self, state: AtomicState) -> float:
        """
        Compute error between state S and nearest ICP-MS database entry
        
        E_database(S) = min_i ||S - DB_i||²
        """
        if self.database is None:
            logger.warning("No database available, returning 0 for database error")
            return 0.0
        
        # Find nearest database entry for this food type
        nearest_entries = self.database.query_similar(
            food_type=state.food_type,
            cooking_method=state.cooking_method,
            top_k=5
        )
        
        if not nearest_entries:
            logger.warning(f"No database entries found for {state.food_type}")
            return 1.0  # High penalty
        
        # Compute minimum distance to database entries
        min_distance = float('inf')
        
        for entry in nearest_entries:
            distance = 0.0
            count = 0
            
            for element, concentration in state.concentrations.items():
                if element in entry.composition:
                    # Normalized squared difference
                    db_value = entry.composition[element]
                    diff = (concentration - db_value) / (db_value + 1e-6)
                    distance += diff ** 2
                    count += 1
            
            if count > 0:
                distance = distance / count
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _compute_constraint_penalty(self, state: AtomicState) -> float:
        """
        Compute penalty for violating physical/chemical constraints
        
        E_constraint(S) = Σ penalty_i(S)
        
        Constraints:
        - Non-negativity: All concentrations >= 0
        - Physical limits: Concentrations within reasonable ranges
        - Mass balance: Total mass fractions sum appropriately
        - Chemical relationships: e.g., Na/K ratio in certain range
        """
        penalty = 0.0
        
        for constraint_name, constraint_func in self.constraints.items():
            violation = constraint_func(state)
            if violation > 0:
                penalty += violation
        
        return penalty
    
    def _initialize_constraints(self) -> Dict[str, Callable]:
        """Initialize constraint functions"""
        constraints = {}
        
        # Non-negativity constraint
        def non_negativity(state: AtomicState) -> float:
            penalty = 0.0
            for element, conc in state.concentrations.items():
                if conc < 0:
                    penalty += abs(conc) * 100  # Heavy penalty
            return penalty
        
        constraints['non_negativity'] = non_negativity
        
        # Physical limits constraint
        def physical_limits(state: AtomicState) -> float:
            penalty = 0.0
            
            # Define reasonable upper limits (ppm or mg/100g)
            limits = {
                'Fe': 50.0,    # Iron: 0-50 mg/100g
                'Zn': 20.0,    # Zinc: 0-20 mg/100g
                'Pb': 1.0,     # Lead: 0-1 ppm (safety limit)
                'Hg': 0.5,     # Mercury: 0-0.5 ppm
                'As': 0.3,     # Arsenic: 0-0.3 ppm
                'Cd': 0.1,     # Cadmium: 0-0.1 ppm
                'Na': 5000.0,  # Sodium: 0-5000 mg/100g
                'K': 3000.0,   # Potassium: 0-3000 mg/100g
                'Ca': 2000.0,  # Calcium: 0-2000 mg/100g
            }
            
            for element, limit in limits.items():
                if element in state.concentrations:
                    conc = state.concentrations[element]
                    if conc > limit:
                        penalty += ((conc - limit) / limit) ** 2
            
            return penalty
        
        constraints['physical_limits'] = physical_limits
        
        # Mass balance constraint
        def mass_balance(state: AtomicState) -> float:
            # For major elements, total should be reasonable
            major_elements = ['Na', 'K', 'Ca', 'Mg', 'P']
            total = sum(
                state.concentrations.get(e, 0.0)
                for e in major_elements
            )
            
            # Total major minerals typically 500-3000 mg/100g
            penalty = 0.0
            if total > 5000:
                penalty = (total - 5000) / 5000
            
            return penalty
        
        constraints['mass_balance'] = mass_balance
        
        # Chemical relationship constraints
        def chemical_relationships(state: AtomicState) -> float:
            penalty = 0.0
            
            # Na/K ratio should be reasonable (typically 0.5-5.0)
            if 'Na' in state.concentrations and 'K' in state.concentrations:
                na = state.concentrations['Na']
                k = state.concentrations['K']
                if k > 0:
                    ratio = na / k
                    if ratio < 0.1 or ratio > 10.0:
                        penalty += 0.5
            
            # Ca/Mg ratio (typically 1-4)
            if 'Ca' in state.concentrations and 'Mg' in state.concentrations:
                ca = state.concentrations['Ca']
                mg = state.concentrations['Mg']
                if mg > 0:
                    ratio = ca / mg
                    if ratio < 0.5 or ratio > 8.0:
                        penalty += 0.3
            
            return penalty
        
        constraints['chemical_relationships'] = chemical_relationships
        
        return constraints
    
    def set_database(self, database):
        """Set the ICP-MS database reference"""
        self.database = database


class VisualFeaturePredictor(nn.Module):
    """
    Neural network that predicts visual features from atomic composition.
    
    This is the inverse of the feature extraction: given atomic state S,
    predict what the visual features (color, texture, spectrum) should be.
    """
    
    def __init__(self, num_elements: int = 45):
        super().__init__()
        
        # Atomic composition encoder
        self.composition_encoder = nn.Sequential(
            nn.Linear(num_elements, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Spectrum predictor
        self.spectrum_predictor = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 300),  # 300 wavelengths
            nn.Sigmoid()
        )
        
        # Color properties predictor
        self.color_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # glossiness, roughness, moisture
            nn.Sigmoid()
        )
        
        # Texture predictor
        self.texture_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Sigmoid()
        )
    
    def forward(self, composition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict visual features from atomic composition
        
        Args:
            composition: (B, num_elements) atomic concentrations
            
        Returns:
            Dictionary of predicted visual features
        """
        # Encode composition
        encoded = self.composition_encoder(composition)
        
        # Predict features
        spectrum = self.spectrum_predictor(encoded)
        color_props = self.color_predictor(encoded)
        texture = self.texture_predictor(encoded)
        
        return {
            'spectrum': spectrum,
            'color_properties': color_props,
            'texture': texture
        }


class StateSpaceExplorer:
    """
    Explores the atomic composition state space through perturbations.
    
    Implements various perturbation strategies:
    - Gaussian noise
    - Cauchy noise (for bigger jumps)
    - Gradient-guided perturbations
    - Hybrid strategies
    """
    
    def __init__(
        self,
        element_order: List[str],
        perturbation_type: PerturbationType = PerturbationType.HYBRID
    ):
        self.element_order = element_order
        self.perturbation_type = perturbation_type
        self.num_elements = len(element_order)
        
        # Statistics for adaptive perturbation
        self.acceptance_history = []
        self.perturbation_scales = {e: 0.1 for e in element_order}
    
    def perturb_state(
        self,
        state: AtomicState,
        temperature: float,
        scale: float = 0.1
    ) -> AtomicState:
        """
        Generate a neighboring state by perturbing the current state
        
        Args:
            state: Current atomic state
            temperature: Current temperature (affects perturbation magnitude)
            scale: Perturbation scale factor
            
        Returns:
            New perturbed state
        """
        new_state = state.copy()
        
        # Temperature-dependent perturbation magnitude
        effective_scale = scale * np.sqrt(temperature)
        
        if self.perturbation_type == PerturbationType.GAUSSIAN:
            new_state = self._gaussian_perturbation(new_state, effective_scale)
        
        elif self.perturbation_type == PerturbationType.CAUCHY:
            new_state = self._cauchy_perturbation(new_state, effective_scale)
        
        elif self.perturbation_type == PerturbationType.HYBRID:
            # Randomly choose perturbation type
            if np.random.random() < 0.7:
                new_state = self._gaussian_perturbation(new_state, effective_scale)
            else:
                new_state = self._cauchy_perturbation(new_state, effective_scale * 2)
        
        # Ensure non-negativity
        for element in new_state.concentrations:
            new_state.concentrations[element] = max(
                0.0,
                new_state.concentrations[element]
            )
        
        return new_state
    
    def _gaussian_perturbation(
        self,
        state: AtomicState,
        scale: float
    ) -> AtomicState:
        """Apply Gaussian noise perturbation"""
        for element in state.concentrations:
            current_value = state.concentrations[element]
            
            # Adaptive scale based on current value
            adaptive_scale = scale * (abs(current_value) + 0.1)
            
            # Add Gaussian noise
            noise = np.random.normal(0, adaptive_scale)
            state.concentrations[element] += noise
        
        return state
    
    def _cauchy_perturbation(
        self,
        state: AtomicState,
        scale: float
    ) -> AtomicState:
        """Apply Cauchy noise perturbation (heavier tails for bigger jumps)"""
        for element in state.concentrations:
            current_value = state.concentrations[element]
            
            adaptive_scale = scale * (abs(current_value) + 0.1)
            
            # Add Cauchy noise
            noise = np.random.standard_cauchy() * adaptive_scale
            
            # Clip extreme values
            noise = np.clip(noise, -current_value * 2, current_value * 2)
            
            state.concentrations[element] += noise
        
        return state
    
    def update_perturbation_stats(self, accepted: bool, element: str):
        """Update perturbation statistics for adaptive scaling"""
        self.acceptance_history.append(accepted)
        
        # Keep last 100 acceptances
        if len(self.acceptance_history) > 100:
            self.acceptance_history.pop(0)
        
        # Adjust perturbation scale based on acceptance rate
        acceptance_rate = sum(self.acceptance_history) / len(self.acceptance_history)
        
        # Target acceptance rate: 20-40%
        if acceptance_rate < 0.2:
            # Too few acceptances, reduce perturbation
            self.perturbation_scales[element] *= 0.9
        elif acceptance_rate > 0.4:
            # Too many acceptances, increase perturbation
            self.perturbation_scales[element] *= 1.1


class TemperatureScheduler:
    """
    Manages temperature cooling schedule for simulated annealing.
    
    Implements multiple cooling strategies:
    - Exponential: T_new = α * T_old
    - Linear: T_new = T_old - α
    - Logarithmic: T_new = T_0 / log(1 + k)
    - Adaptive: Adjust based on convergence metrics
    """
    
    def __init__(
        self,
        initial_temp: float,
        final_temp: float,
        schedule: CoolingSchedule = CoolingSchedule.EXPONENTIAL,
        cooling_rate: float = 0.95
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.current_temp = initial_temp
        self.schedule = schedule
        self.cooling_rate = cooling_rate
        self.iteration = 0
        
        # For adaptive cooling
        self.acceptance_history = []
        self.cost_history = []
    
    def update_temperature(self) -> float:
        """Update and return new temperature"""
        self.iteration += 1
        
        if self.schedule == CoolingSchedule.EXPONENTIAL:
            self.current_temp = max(
                self.final_temp,
                self.current_temp * self.cooling_rate
            )
        
        elif self.schedule == CoolingSchedule.LINEAR:
            decrement = (self.initial_temp - self.final_temp) / 1000
            self.current_temp = max(
                self.final_temp,
                self.current_temp - decrement
            )
        
        elif self.schedule == CoolingSchedule.LOGARITHMIC:
            self.current_temp = max(
                self.final_temp,
                self.initial_temp / np.log(1 + self.iteration)
            )
        
        elif self.schedule == CoolingSchedule.FAST:
            self.current_temp = max(
                self.final_temp,
                self.initial_temp / (1 + self.iteration)
            )
        
        elif self.schedule == CoolingSchedule.ADAPTIVE:
            self.current_temp = self._adaptive_cooling()
        
        return self.current_temp
    
    def _adaptive_cooling(self) -> float:
        """Adaptive cooling based on acceptance rate and cost improvement"""
        if len(self.acceptance_history) < 10:
            return self.current_temp * self.cooling_rate
        
        # Calculate recent acceptance rate
        recent_acceptance = sum(self.acceptance_history[-10:]) / 10
        
        # Calculate cost improvement
        if len(self.cost_history) >= 2:
            cost_improvement = (
                self.cost_history[-2] - self.cost_history[-1]
            ) / (abs(self.cost_history[-2]) + 1e-8)
        else:
            cost_improvement = 0.0
        
        # Adjust cooling rate based on metrics
        if recent_acceptance < 0.1 or cost_improvement < 0.001:
            # Stagnating, cool faster
            adjusted_rate = self.cooling_rate * 0.95
        elif recent_acceptance > 0.5:
            # Too many acceptances, cool slower
            adjusted_rate = self.cooling_rate * 1.05
        else:
            adjusted_rate = self.cooling_rate
        
        return max(self.final_temp, self.current_temp * adjusted_rate)
    
    def record_acceptance(self, accepted: bool):
        """Record acceptance for adaptive cooling"""
        self.acceptance_history.append(accepted)
        if len(self.acceptance_history) > 100:
            self.acceptance_history.pop(0)
    
    def record_cost(self, cost: float):
        """Record cost for adaptive cooling"""
        self.cost_history.append(cost)
        if len(self.cost_history) > 100:
            self.cost_history.pop(0)


class ConvergenceMonitor:
    """
    Monitors convergence of the simulated annealing optimization.
    
    Tracks:
    - Best cost over time
    - Acceptance rate
    - Temperature schedule
    - Convergence criteria
    """
    
    def __init__(
        self,
        patience: int = 500,
        min_improvement: float = 1e-6
    ):
        self.patience = patience
        self.min_improvement = min_improvement
        
        # History tracking
        self.cost_history = []
        self.best_cost_history = []
        self.temperature_history = []
        self.acceptance_history = []
        
        # Convergence tracking
        self.best_cost = float('inf')
        self.best_state = None
        self.iterations_without_improvement = 0
        self.converged = False
    
    def update(
        self,
        current_cost: float,
        current_state: AtomicState,
        temperature: float,
        accepted: bool
    ) -> bool:
        """
        Update convergence monitoring
        
        Returns:
            True if converged, False otherwise
        """
        # Record history
        self.cost_history.append(current_cost)
        self.temperature_history.append(temperature)
        self.acceptance_history.append(accepted)
        
        # Update best state
        if current_cost < self.best_cost - self.min_improvement:
            self.best_cost = current_cost
            self.best_state = current_state.copy()
            self.iterations_without_improvement = 0
            logger.info(f"New best cost: {self.best_cost:.6f}")
        else:
            self.iterations_without_improvement += 1
        
        self.best_cost_history.append(self.best_cost)
        
        # Check convergence criteria
        if self.iterations_without_improvement >= self.patience:
            logger.info(f"Converged: No improvement for {self.patience} iterations")
            self.converged = True
            return True
        
        # Check if temperature is too low
        if temperature < 0.001:
            logger.info("Converged: Temperature near zero")
            self.converged = True
            return True
        
        return False
    
    def get_acceptance_rate(self, window: int = 100) -> float:
        """Get recent acceptance rate"""
        if len(self.acceptance_history) == 0:
            return 0.0
        
        recent = self.acceptance_history[-window:]
        return sum(recent) / len(recent)
    
    def plot_convergence(self, save_path: Optional[str] = None):
        """Plot convergence metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cost history
        axes[0, 0].plot(self.cost_history, label='Current Cost', alpha=0.6)
        axes[0, 0].plot(self.best_cost_history, label='Best Cost', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Cost')
        axes[0, 0].set_title('Cost Evolution')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Temperature
        axes[0, 1].plot(self.temperature_history)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Temperature')
        axes[0, 1].set_title('Temperature Schedule')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Acceptance rate (moving average)
        window = 100
        acceptance_ma = np.convolve(
            self.acceptance_history,
            np.ones(window) / window,
            mode='valid'
        )
        axes[1, 0].plot(acceptance_ma)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Acceptance Rate')
        axes[1, 0].set_title(f'Acceptance Rate (MA-{window})')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cost improvement rate
        if len(self.best_cost_history) > 1:
            improvements = -np.diff(self.best_cost_history)
            axes[1, 1].plot(improvements)
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Cost Improvement')
            axes[1, 1].set_title('Cost Improvement per Iteration')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Convergence plot saved to {save_path}")
        
        plt.show()


class SimulatedAnnealingOptimizer:
    """
    Main Simulated Annealing optimizer for visual-to-atomic modeling.
    
    Solves the inverse problem:
        Find S* = argmin E(S)
        where E(S) = α·E_visual(S) + β·E_database(S) + γ·E_constraint(S)
    
    Process:
    1. Initialize state S_0 from ingredient prediction
    2. At each iteration:
        a. Generate neighbor state S' by perturbation
        b. Compute ΔE = E(S') - E(S)
        c. Accept S' with probability exp(-ΔE/T)
        d. Update temperature T
    3. Return best state S* found
    """
    
    def __init__(
        self,
        element_order: List[str],
        params: Optional[SAParameters] = None,
        cost_weights: Optional[CostFunctionWeights] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.element_order = element_order
        self.params = params or SAParameters()
        self.device = device
        
        # Initialize components
        self.cost_calculator = CostFunctionCalculator(cost_weights, device)
        self.state_explorer = StateSpaceExplorer(
            element_order,
            self.params.perturbation_type
        )
        self.temp_scheduler = TemperatureScheduler(
            self.params.initial_temperature,
            self.params.final_temperature,
            self.params.cooling_schedule,
            self.params.cooling_rate
        )
        self.convergence_monitor = ConvergenceMonitor()
        
        logger.info("SimulatedAnnealingOptimizer initialized")
    
    def optimize(
        self,
        initial_state: AtomicState,
        target_features: Dict[str, Any],
        verbose: bool = True
    ) -> Tuple[AtomicState, Dict[str, Any]]:
        """
        Run simulated annealing optimization
        
        Args:
            initial_state: Initial atomic state (from ingredient prediction)
            target_features: Target visual features from image
            verbose: Print progress
            
        Returns:
            Tuple of (best_state, optimization_info)
        """
        logger.info("Starting Simulated Annealing optimization")
        
        # Initialize
        current_state = initial_state.copy()
        current_cost = self.cost_calculator.compute_total_cost(
            current_state,
            target_features,
            self.temp_scheduler.current_temp
        )
        
        best_state = current_state.copy()
        best_cost = current_cost
        
        # Progress tracking
        pbar = tqdm(
            total=self.params.max_iterations,
            desc="SA Optimization",
            disable=not verbose
        )
        
        iteration = 0
        num_acceptances = 0
        num_rejections = 0
        
        while iteration < self.params.max_iterations:
            # Inner loop: iterations at current temperature
            for _ in range(self.params.iterations_per_temp):
                iteration += 1
                
                # Generate neighbor state
                neighbor_state = self.state_explorer.perturb_state(
                    current_state,
                    self.temp_scheduler.current_temp,
                    self.params.perturbation_scale
                )
                
                # Compute cost of neighbor
                neighbor_cost = self.cost_calculator.compute_total_cost(
                    neighbor_state,
                    target_features,
                    self.temp_scheduler.current_temp
                )
                
                # Compute cost difference
                delta_cost = neighbor_cost - current_cost
                
                # Acceptance criterion
                accepted = self._accept_state(
                    delta_cost,
                    self.temp_scheduler.current_temp
                )
                
                if accepted:
                    current_state = neighbor_state
                    current_cost = neighbor_cost
                    num_acceptances += 1
                    
                    # Update best state
                    if current_cost < best_cost:
                        best_state = current_state.copy()
                        best_cost = current_cost
                else:
                    num_rejections += 1
                
                # Update monitoring
                self.convergence_monitor.update(
                    current_cost,
                    current_state,
                    self.temp_scheduler.current_temp,
                    accepted
                )
                self.temp_scheduler.record_acceptance(accepted)
                self.temp_scheduler.record_cost(current_cost)
                
                # Check convergence
                if self.convergence_monitor.converged:
                    logger.info("Optimization converged")
                    break
                
                # Update progress bar
                if iteration % 10 == 0:
                    pbar.update(10)
                    pbar.set_postfix({
                        'Best Cost': f'{best_cost:.4f}',
                        'Temp': f'{self.temp_scheduler.current_temp:.2f}',
                        'Accept%': f'{num_acceptances/(num_acceptances+num_rejections)*100:.1f}'
                    })
            
            if self.convergence_monitor.converged:
                break
            
            # Cool down temperature
            self.temp_scheduler.update_temperature()
            
            # Optional: Reheating
            if (self.params.adaptive_cooling and
                iteration % self.params.reheat_frequency == 0 and
                self.convergence_monitor.iterations_without_improvement > 100):
                logger.info(f"Reheating temperature from {self.temp_scheduler.current_temp} to {self.params.reheat_temperature}")
                self.temp_scheduler.current_temp = self.params.reheat_temperature
        
        pbar.close()
        
        # Prepare optimization info
        optimization_info = {
            'final_cost': best_cost,
            'visual_error': best_state.visual_error,
            'database_error': best_state.database_error,
            'constraint_penalty': best_state.constraint_penalty,
            'total_iterations': iteration,
            'acceptance_rate': num_acceptances / (num_acceptances + num_rejections),
            'convergence_history': {
                'cost': self.convergence_monitor.cost_history,
                'best_cost': self.convergence_monitor.best_cost_history,
                'temperature': self.convergence_monitor.temperature_history,
                'acceptance': self.convergence_monitor.acceptance_history,
            }
        }
        
        logger.info(f"Optimization complete. Best cost: {best_cost:.6f}")
        logger.info(f"Acceptance rate: {optimization_info['acceptance_rate']:.2%}")
        
        return best_state, optimization_info
    
    def _accept_state(self, delta_cost: float, temperature: float) -> bool:
        """
        Metropolis acceptance criterion
        
        Accept if:
        - ΔE < 0 (improvement): Always accept
        - ΔE > 0 (worsening): Accept with probability exp(-ΔE/T)
        """
        if delta_cost < 0:
            return True
        
        # Metropolis criterion
        acceptance_probability = np.exp(-delta_cost / (temperature + 1e-8))
        
        return np.random.random() < acceptance_probability
    
    def set_database(self, database):
        """Set ICP-MS database"""
        self.cost_calculator.set_database(database)
    
    def save_state(self, filepath: str, state: AtomicState, info: Dict):
        """Save optimization state"""
        data = {
            'state': {
                'concentrations': state.concentrations,
                'food_type': state.food_type,
                'cooking_method': state.cooking_method,
            },
            'info': info,
            'params': {
                'initial_temp': self.params.initial_temperature,
                'final_temp': self.params.final_temperature,
                'cooling_rate': self.params.cooling_rate,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"State saved to {filepath}")


# Utility functions

def initialize_state_from_ingredients(
    ingredients: List['RawIngredientPrediction'],
    element_order: List[str]
) -> AtomicState:
    """
    Initialize atomic state from raw ingredient predictions
    
    Combines predicted ingredients with their mass fractions to
    estimate initial atomic composition
    """
    concentrations = {element: 0.0 for element in element_order}
    
    # Weight atomic signatures by mass fractions
    for ingredient in ingredients:
        weight = ingredient.mass_fraction
        for element, conc in ingredient.atomic_signature.items():
            if element in concentrations:
                concentrations[element] += conc * weight
    
    # Determine food type (dominant ingredient)
    if ingredients:
        food_type = ingredients[0].ingredient_name
    else:
        food_type = "unknown"
    
    return AtomicState(
        concentrations=concentrations,
        food_type=food_type,
        cooking_method="unknown"
    )


if __name__ == "__main__":
    # Test simulated annealing
    logger.info("Testing Phase 2: Simulated Annealing Optimization")
    
    # Define elements
    elements = ['Fe', 'Zn', 'Cu', 'Mn', 'Na', 'K', 'Ca', 'Mg', 'P',
                'Pb', 'Hg', 'As', 'Cd']
    
    # Create initial state
    initial_conc = {
        'Fe': 2.5, 'Zn': 1.2, 'Cu': 0.3, 'Mn': 0.5,
        'Na': 500.0, 'K': 400.0, 'Ca': 50.0, 'Mg': 30.0, 'P': 200.0,
        'Pb': 0.01, 'Hg': 0.005, 'As': 0.002, 'Cd': 0.001
    }
    initial_state = AtomicState(
        concentrations=initial_conc,
        food_type="salmon",
        cooking_method="grilled"
    )
    
    # Create dummy target features
    target_features = {
        'spectral_features': type('obj', (object,), {
            'simulated_spectrum': np.random.rand(300)
        })(),
        'chemometric_features': type('obj', (object,), {
            'glossiness_score': 0.6,
            'surface_roughness': 0.4,
            'moisture_retention': 0.7
        })()
    }
    
    # Create optimizer
    params = SAParameters(
        initial_temperature=100.0,
        final_temperature=0.1,
        max_iterations=1000,
        iterations_per_temp=10
    )
    
    optimizer = SimulatedAnnealingOptimizer(elements, params)
    
    # Run optimization
    best_state, info = optimizer.optimize(
        initial_state,
        target_features,
        verbose=True
    )
    
    logger.info("\nOptimization Results:")
    logger.info(f"Final cost: {info['final_cost']:.6f}")
    logger.info(f"Visual error: {info['visual_error']:.6f}")
    logger.info(f"Database error: {info['database_error']:.6f}")
    logger.info(f"Constraint penalty: {info['constraint_penalty']:.6f}")
    logger.info("\nBest atomic composition:")
    for element, conc in sorted(best_state.concentrations.items()):
        logger.info(f"  {element}: {conc:.4f}")
    
    # Plot convergence
    optimizer.convergence_monitor.plot_convergence()
    
    logger.info("Phase 2 test complete!")
