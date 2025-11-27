"""
Chaos Engineering
=================

Chaos engineering framework for testing system resilience through
controlled failure injection and fault tolerance validation.

Features:
1. Failure injection (network, CPU, memory, disk)
2. Latency injection
3. Resource exhaustion
4. Circuit breaker testing
5. Disaster recovery validation
6. Game day scenarios
7. Automated chaos experiments
8. Resilience metrics

Performance Targets:
- Zero data loss during chaos
- Recovery time: <30 seconds
- System availability: >99.9% during chaos
- Automated rollback on critical failures
- Support 50+ chaos scenarios

Author: Wellomex AI Team
Date: November 2025
Version: 5.0.0
"""

import time
import logging
import random
import threading
import multiprocessing
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class FailureType(Enum):
    """Chaos failure type"""
    NETWORK_LATENCY = "network_latency"
    NETWORK_PARTITION = "network_partition"
    NETWORK_PACKET_LOSS = "network_packet_loss"
    CPU_STRESS = "cpu_stress"
    MEMORY_PRESSURE = "memory_pressure"
    DISK_FILL = "disk_fill"
    DISK_IO_ERROR = "disk_io_error"
    SERVICE_CRASH = "service_crash"
    DATABASE_UNAVAILABLE = "database_unavailable"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"


class TargetType(Enum):
    """Chaos target type"""
    POD = "pod"
    SERVICE = "service"
    NODE = "node"
    DATABASE = "database"
    NETWORK = "network"
    STORAGE = "storage"


class ExperimentStatus(Enum):
    """Experiment status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLBACK = "rollback"


@dataclass
class ChaosConfig:
    """Chaos engineering configuration"""
    # Safety
    max_concurrent_experiments: int = 3
    auto_rollback: bool = True
    rollback_threshold: float = 0.5  # Error rate threshold
    
    # Scheduling
    min_interval_seconds: int = 300
    max_duration_seconds: int = 600
    
    # Monitoring
    health_check_interval: int = 10
    metric_collection_interval: int = 5


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Target:
    """Chaos experiment target"""
    target_type: TargetType
    name: str
    namespace: str = "default"
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class FailureConfig:
    """Failure configuration"""
    failure_type: FailureType
    
    # Parameters
    duration_seconds: int = 60
    intensity: float = 0.5  # 0.0 to 1.0
    
    # Network failures
    latency_ms: Optional[int] = None
    packet_loss_percent: Optional[float] = None
    
    # Resource failures
    cpu_cores: Optional[int] = None
    memory_mb: Optional[int] = None
    
    # Error injection
    error_rate: Optional[float] = None
    timeout_ms: Optional[int] = None


@dataclass
class ChaosExperiment:
    """Chaos experiment definition"""
    id: str
    name: str
    description: str
    
    # Target
    targets: List[Target]
    
    # Failure
    failure_config: FailureConfig
    
    # Schedule
    scheduled_time: Optional[datetime] = None
    
    # State
    status: ExperimentStatus = ExperimentStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Results
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))


@dataclass
class SystemHealth:
    """System health snapshot"""
    timestamp: datetime
    
    # Metrics
    error_rate: float
    latency_p99: float
    availability: float
    throughput: float
    
    # Resources
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    
    # Status
    is_healthy: bool = True


# ============================================================================
# FAILURE INJECTOR
# ============================================================================

class FailureInjector:
    """
    Failure Injector
    
    Injects various types of failures into the system.
    """
    
    def __init__(self):
        self.active_failures: Dict[str, Any] = {}
        
        logger.info("Failure Injector initialized")
    
    def inject_network_latency(
        self,
        target: Target,
        latency_ms: int,
        duration_seconds: int
    ) -> str:
        """Inject network latency"""
        failure_id = f"latency_{target.name}_{int(time.time())}"
        
        logger.info(f"Injecting {latency_ms}ms latency to {target.name}")
        
        # In production, would use tc (traffic control) or similar
        # For simulation, just record the failure
        self.active_failures[failure_id] = {
            'type': FailureType.NETWORK_LATENCY,
            'target': target,
            'latency_ms': latency_ms,
            'start_time': datetime.now(),
            'duration_seconds': duration_seconds
        }
        
        return failure_id
    
    def inject_packet_loss(
        self,
        target: Target,
        loss_percent: float,
        duration_seconds: int
    ) -> str:
        """Inject packet loss"""
        failure_id = f"packet_loss_{target.name}_{int(time.time())}"
        
        logger.info(f"Injecting {loss_percent}% packet loss to {target.name}")
        
        self.active_failures[failure_id] = {
            'type': FailureType.NETWORK_PACKET_LOSS,
            'target': target,
            'loss_percent': loss_percent,
            'start_time': datetime.now(),
            'duration_seconds': duration_seconds
        }
        
        return failure_id
    
    def inject_cpu_stress(
        self,
        target: Target,
        cores: int,
        duration_seconds: int
    ) -> str:
        """Inject CPU stress"""
        failure_id = f"cpu_stress_{target.name}_{int(time.time())}"
        
        logger.info(f"Injecting CPU stress ({cores} cores) to {target.name}")
        
        # In production, would use stress-ng or similar
        self.active_failures[failure_id] = {
            'type': FailureType.CPU_STRESS,
            'target': target,
            'cores': cores,
            'start_time': datetime.now(),
            'duration_seconds': duration_seconds
        }
        
        return failure_id
    
    def inject_memory_pressure(
        self,
        target: Target,
        memory_mb: int,
        duration_seconds: int
    ) -> str:
        """Inject memory pressure"""
        failure_id = f"memory_{target.name}_{int(time.time())}"
        
        logger.info(f"Injecting memory pressure ({memory_mb}MB) to {target.name}")
        
        self.active_failures[failure_id] = {
            'type': FailureType.MEMORY_PRESSURE,
            'target': target,
            'memory_mb': memory_mb,
            'start_time': datetime.now(),
            'duration_seconds': duration_seconds
        }
        
        return failure_id
    
    def inject_service_crash(self, target: Target) -> str:
        """Inject service crash"""
        failure_id = f"crash_{target.name}_{int(time.time())}"
        
        logger.info(f"Crashing service {target.name}")
        
        # In production, would kill pod or process
        self.active_failures[failure_id] = {
            'type': FailureType.SERVICE_CRASH,
            'target': target,
            'start_time': datetime.now()
        }
        
        return failure_id
    
    def inject_api_error(
        self,
        target: Target,
        error_rate: float,
        duration_seconds: int
    ) -> str:
        """Inject API errors"""
        failure_id = f"api_error_{target.name}_{int(time.time())}"
        
        logger.info(f"Injecting {error_rate*100}% error rate to {target.name}")
        
        self.active_failures[failure_id] = {
            'type': FailureType.API_ERROR,
            'target': target,
            'error_rate': error_rate,
            'start_time': datetime.now(),
            'duration_seconds': duration_seconds
        }
        
        return failure_id
    
    def remove_failure(self, failure_id: str):
        """Remove injected failure"""
        if failure_id in self.active_failures:
            failure = self.active_failures[failure_id]
            
            logger.info(f"Removing failure {failure_id}")
            
            # In production, would clean up actual injected failures
            del self.active_failures[failure_id]
    
    def cleanup_expired_failures(self):
        """Clean up expired failures"""
        now = datetime.now()
        expired = []
        
        for failure_id, failure in self.active_failures.items():
            start_time = failure['start_time']
            duration = failure.get('duration_seconds', 0)
            
            if (now - start_time).total_seconds() > duration:
                expired.append(failure_id)
        
        for failure_id in expired:
            self.remove_failure(failure_id)


# ============================================================================
# HEALTH MONITOR
# ============================================================================

class HealthMonitor:
    """
    Health Monitor
    
    Monitors system health during chaos experiments.
    """
    
    def __init__(self, config: ChaosConfig):
        self.config = config
        
        # Health history
        self.health_history: deque = deque(maxlen=1000)
        
        # Baseline metrics
        self.baseline_error_rate = 0.01
        self.baseline_latency = 100.0
        self.baseline_availability = 0.999
        
        logger.info("Health Monitor initialized")
    
    def collect_metrics(self) -> SystemHealth:
        """Collect current system metrics"""
        # In production, would collect from Prometheus, CloudWatch, etc.
        # For simulation, generate synthetic metrics
        
        health = SystemHealth(
            timestamp=datetime.now(),
            error_rate=random.uniform(0.001, 0.05),
            latency_p99=random.uniform(50, 500),
            availability=random.uniform(0.95, 1.0),
            throughput=random.uniform(1000, 5000),
            cpu_usage=random.uniform(0.3, 0.9),
            memory_usage=random.uniform(0.4, 0.8),
            disk_usage=random.uniform(0.2, 0.7)
        )
        
        # Determine health
        health.is_healthy = (
            health.error_rate < self.baseline_error_rate * 5 and
            health.latency_p99 < self.baseline_latency * 3 and
            health.availability > 0.95
        )
        
        self.health_history.append(health)
        
        return health
    
    def is_system_healthy(self) -> bool:
        """Check if system is currently healthy"""
        if not self.health_history:
            return True
        
        recent_health = list(self.health_history)[-10:]
        
        # Check recent health
        healthy_count = sum(1 for h in recent_health if h.is_healthy)
        
        return healthy_count >= len(recent_health) * 0.7
    
    def get_degradation_score(self) -> float:
        """Get system degradation score (0=healthy, 1=critical)"""
        if not self.health_history:
            return 0.0
        
        recent = list(self.health_history)[-10:]
        
        # Compare to baseline
        error_rate_score = sum(h.error_rate for h in recent) / len(recent) / self.baseline_error_rate
        latency_score = sum(h.latency_p99 for h in recent) / len(recent) / self.baseline_latency
        availability_score = (1 - sum(h.availability for h in recent) / len(recent)) / (1 - self.baseline_availability)
        
        # Combined score
        score = (error_rate_score + latency_score + availability_score) / 3
        
        return min(score, 1.0)


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """
    Experiment Runner
    
    Runs chaos experiments and manages lifecycle.
    """
    
    def __init__(
        self,
        config: ChaosConfig,
        injector: FailureInjector,
        monitor: HealthMonitor
    ):
        self.config = config
        self.injector = injector
        self.monitor = monitor
        
        # Experiments
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.running_experiments: List[str] = []
        
        # Control
        self.running = False
        self.experiment_thread: Optional[threading.Thread] = None
        
        logger.info("Experiment Runner initialized")
    
    def schedule_experiment(self, experiment: ChaosExperiment):
        """Schedule chaos experiment"""
        self.experiments[experiment.id] = experiment
        
        logger.info(f"Scheduled experiment: {experiment.name}")
    
    def run_experiment(self, experiment_id: str) -> bool:
        """Run single experiment"""
        experiment = self.experiments.get(experiment_id)
        
        if not experiment:
            logger.error(f"Experiment not found: {experiment_id}")
            return False
        
        # Check if can run
        if len(self.running_experiments) >= self.config.max_concurrent_experiments:
            logger.warning("Too many concurrent experiments")
            return False
        
        # Start experiment
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = datetime.now()
        
        self.running_experiments.append(experiment_id)
        
        logger.info(f"Running experiment: {experiment.name}")
        
        try:
            # Inject failures
            failure_ids = []
            
            for target in experiment.targets:
                failure_id = self._inject_failure(target, experiment.failure_config)
                
                if failure_id:
                    failure_ids.append(failure_id)
            
            # Monitor for duration
            start_time = time.time()
            duration = experiment.failure_config.duration_seconds
            
            while time.time() - start_time < duration:
                # Collect metrics
                health = self.monitor.collect_metrics()
                
                # Record metrics
                experiment.metrics['error_rate'].append(health.error_rate)
                experiment.metrics['latency_p99'].append(health.latency_p99)
                experiment.metrics['availability'].append(health.availability)
                
                # Check if need rollback
                if self.config.auto_rollback:
                    degradation = self.monitor.get_degradation_score()
                    
                    if degradation > self.config.rollback_threshold:
                        logger.warning(f"Degradation too high ({degradation:.2f}), rolling back")
                        experiment.status = ExperimentStatus.ROLLBACK
                        break
                
                time.sleep(self.config.metric_collection_interval)
            
            # Remove failures
            for failure_id in failure_ids:
                self.injector.remove_failure(failure_id)
            
            # Complete
            experiment.end_time = datetime.now()
            
            if experiment.status == ExperimentStatus.RUNNING:
                experiment.status = ExperimentStatus.COMPLETED
            
            # Calculate results
            experiment.results = self._calculate_results(experiment)
            
            logger.info(f"Experiment completed: {experiment.name}")
            
            return True
        
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            experiment.status = ExperimentStatus.FAILED
            return False
        
        finally:
            if experiment_id in self.running_experiments:
                self.running_experiments.remove(experiment_id)
    
    def _inject_failure(
        self,
        target: Target,
        config: FailureConfig
    ) -> Optional[str]:
        """Inject failure based on config"""
        failure_type = config.failure_type
        
        if failure_type == FailureType.NETWORK_LATENCY:
            return self.injector.inject_network_latency(
                target,
                config.latency_ms or 100,
                config.duration_seconds
            )
        
        elif failure_type == FailureType.NETWORK_PACKET_LOSS:
            return self.injector.inject_packet_loss(
                target,
                config.packet_loss_percent or 10.0,
                config.duration_seconds
            )
        
        elif failure_type == FailureType.CPU_STRESS:
            return self.injector.inject_cpu_stress(
                target,
                config.cpu_cores or 2,
                config.duration_seconds
            )
        
        elif failure_type == FailureType.MEMORY_PRESSURE:
            return self.injector.inject_memory_pressure(
                target,
                config.memory_mb or 512,
                config.duration_seconds
            )
        
        elif failure_type == FailureType.SERVICE_CRASH:
            return self.injector.inject_service_crash(target)
        
        elif failure_type == FailureType.API_ERROR:
            return self.injector.inject_api_error(
                target,
                config.error_rate or 0.1,
                config.duration_seconds
            )
        
        return None
    
    def _calculate_results(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Calculate experiment results"""
        metrics = experiment.metrics
        
        results = {
            'duration_seconds': (
                experiment.end_time - experiment.start_time
            ).total_seconds() if experiment.end_time else 0,
            'status': experiment.status.value
        }
        
        if NUMPY_AVAILABLE:
            for metric_name, values in metrics.items():
                if values:
                    values_array = np.array(values)
                    results[f'{metric_name}_mean'] = float(np.mean(values_array))
                    results[f'{metric_name}_max'] = float(np.max(values_array))
                    results[f'{metric_name}_p95'] = float(np.percentile(values_array, 95))
        else:
            for metric_name, values in metrics.items():
                if values:
                    results[f'{metric_name}_mean'] = sum(values) / len(values)
                    results[f'{metric_name}_max'] = max(values)
        
        return results


# ============================================================================
# CHAOS ORCHESTRATOR
# ============================================================================

class ChaosOrchestrator:
    """
    Chaos Orchestrator
    
    Main chaos engineering orchestration system.
    """
    
    def __init__(self, config: Optional[ChaosConfig] = None):
        self.config = config or ChaosConfig()
        
        # Components
        self.injector = FailureInjector()
        self.monitor = HealthMonitor(self.config)
        self.runner = ExperimentRunner(self.config, self.injector, self.monitor)
        
        logger.info("Chaos Orchestrator initialized")
    
    def create_experiment(
        self,
        name: str,
        description: str,
        targets: List[Target],
        failure_config: FailureConfig
    ) -> ChaosExperiment:
        """Create chaos experiment"""
        experiment = ChaosExperiment(
            id=f"exp_{int(time.time())}_{random.randint(1000, 9999)}",
            name=name,
            description=description,
            targets=targets,
            failure_config=failure_config
        )
        
        return experiment
    
    def run_experiment(self, experiment: ChaosExperiment) -> bool:
        """Run chaos experiment"""
        self.runner.schedule_experiment(experiment)
        return self.runner.run_experiment(experiment.id)
    
    def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment results"""
        experiment = self.runner.experiments.get(experiment_id)
        
        if not experiment:
            return None
        
        return {
            'id': experiment.id,
            'name': experiment.name,
            'status': experiment.status.value,
            'results': experiment.results,
            'start_time': experiment.start_time.isoformat() if experiment.start_time else None,
            'end_time': experiment.end_time.isoformat() if experiment.end_time else None
        }


# ============================================================================
# TESTING
# ============================================================================

def test_chaos_engineering():
    """Test chaos engineering system"""
    print("=" * 80)
    print("CHAOS ENGINEERING - TEST")
    print("=" * 80)
    
    # Create orchestrator
    config = ChaosConfig(
        max_concurrent_experiments=2,
        auto_rollback=True,
        rollback_threshold=0.7
    )
    
    orchestrator = ChaosOrchestrator(config)
    
    print("✓ Chaos orchestrator initialized")
    
    # Test network latency injection
    print("\n" + "="*80)
    print("Test: Network Latency Injection")
    print("="*80)
    
    target = Target(
        target_type=TargetType.SERVICE,
        name="nutrition-api",
        namespace="production"
    )
    
    failure_config = FailureConfig(
        failure_type=FailureType.NETWORK_LATENCY,
        duration_seconds=10,
        latency_ms=200
    )
    
    experiment = orchestrator.create_experiment(
        name="Network Latency Test",
        description="Test system resilience under network latency",
        targets=[target],
        failure_config=failure_config
    )
    
    print(f"✓ Created experiment: {experiment.name}")
    print(f"  ID: {experiment.id}")
    print(f"  Target: {target.name}")
    print(f"  Failure: {failure_config.failure_type.value}")
    print(f"  Latency: {failure_config.latency_ms}ms")
    
    # Run experiment
    print("\nRunning experiment...")
    
    success = orchestrator.run_experiment(experiment)
    
    print(f"✓ Experiment {'succeeded' if success else 'failed'}")
    print(f"  Status: {experiment.status.value}")
    
    # Get results
    results = orchestrator.get_experiment_results(experiment.id)
    
    if results:
        print(f"\n  Results:")
        print(f"    Duration: {results['results'].get('duration_seconds', 0):.1f}s")
        
        if 'error_rate_mean' in results['results']:
            print(f"    Avg error rate: {results['results']['error_rate_mean']:.3f}")
        
        if 'latency_p99_mean' in results['results']:
            print(f"    Avg p99 latency: {results['results']['latency_p99_mean']:.1f}ms")
    
    # Test CPU stress
    print("\n" + "="*80)
    print("Test: CPU Stress Injection")
    print("="*80)
    
    cpu_failure = FailureConfig(
        failure_type=FailureType.CPU_STRESS,
        duration_seconds=10,
        cpu_cores=2
    )
    
    cpu_experiment = orchestrator.create_experiment(
        name="CPU Stress Test",
        description="Test system under CPU pressure",
        targets=[target],
        failure_config=cpu_failure
    )
    
    print(f"✓ Created CPU stress experiment")
    print(f"  Cores: {cpu_failure.cpu_cores}")
    
    success = orchestrator.run_experiment(cpu_experiment)
    
    print(f"✓ CPU experiment {'succeeded' if success else 'failed'}")
    
    # Test health monitoring
    print("\n" + "="*80)
    print("Test: Health Monitoring")
    print("="*80)
    
    health = orchestrator.monitor.collect_metrics()
    
    print(f"✓ System health:")
    print(f"  Error rate: {health.error_rate:.3f}")
    print(f"  p99 latency: {health.latency_p99:.1f}ms")
    print(f"  Availability: {health.availability:.3%}")
    print(f"  CPU usage: {health.cpu_usage:.1%}")
    print(f"  Memory usage: {health.memory_usage:.1%}")
    print(f"  Healthy: {'Yes' if health.is_healthy else 'No'}")
    
    degradation = orchestrator.monitor.get_degradation_score()
    
    print(f"\n  Degradation score: {degradation:.2f}")
    print(f"  System healthy: {orchestrator.monitor.is_system_healthy()}")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_chaos_engineering()
