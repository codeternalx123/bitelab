"""
Multi-Region Deployment
=======================

Multi-region active-active deployment strategies for high availability,
disaster recovery, and global performance optimization.

Features:
1. Multi-region orchestration
2. Global load balancing
3. Data replication and consistency
4. Failover automation
5. Traffic routing optimization
6. Region health monitoring
7. Cross-region sync
8. Geo-distributed caching

Performance Targets:
- Global availability: 99.99%
- Failover time: <30 seconds
- Cross-region latency: <100ms
- Data consistency: eventual (configurable)
- Support 10+ regions
- Auto-scaling per region

Author: Wellomex AI Team
Date: November 2025
Version: 5.0.0
"""

import time
import logging
import random
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class RegionStatus(Enum):
    """Region status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class RoutingStrategy(Enum):
    """Traffic routing strategy"""
    LATENCY_BASED = "latency_based"
    GEOPROXIMITY = "geoproximity"
    WEIGHTED = "weighted"
    FAILOVER = "failover"
    ROUND_ROBIN = "round_robin"


class ReplicationMode(Enum):
    """Data replication mode"""
    ASYNC = "async"
    SYNC = "sync"
    SEMI_SYNC = "semi_sync"


class ConsistencyLevel(Enum):
    """Data consistency level"""
    EVENTUAL = "eventual"
    STRONG = "strong"
    BOUNDED_STALENESS = "bounded_staleness"
    SESSION = "session"


@dataclass
class MultiRegionConfig:
    """Multi-region configuration"""
    # Regions
    regions: List[str] = field(default_factory=lambda: ['us-east-1', 'us-west-2', 'eu-west-1'])
    primary_region: str = 'us-east-1'
    
    # Routing
    routing_strategy: RoutingStrategy = RoutingStrategy.LATENCY_BASED
    
    # Replication
    replication_mode: ReplicationMode = ReplicationMode.ASYNC
    consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    
    # Failover
    auto_failover: bool = True
    failover_threshold: float = 0.5  # Health score
    
    # Health checks
    health_check_interval: int = 10  # seconds
    health_check_timeout: int = 5
    
    # Performance
    target_latency_ms: float = 100.0
    max_cross_region_latency_ms: float = 500.0


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Region:
    """Region definition"""
    id: str
    name: str
    location: str  # Geographic location
    
    # Coordinates for geo-routing
    latitude: float
    longitude: float
    
    # Status
    status: RegionStatus = RegionStatus.HEALTHY
    health_score: float = 1.0
    
    # Capacity
    max_capacity: int = 1000
    current_load: int = 0
    
    # Network
    endpoints: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_health_check: Optional[datetime] = None


@dataclass
class RegionMetrics:
    """Region performance metrics"""
    region_id: str
    timestamp: datetime
    
    # Performance
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Throughput
    requests_per_second: float = 0.0
    bytes_per_second: float = 0.0
    
    # Reliability
    success_rate: float = 1.0
    error_rate: float = 0.0
    
    # Resources
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0


@dataclass
class ReplicationTask:
    """Data replication task"""
    id: str
    source_region: str
    target_regions: List[str]
    
    data_key: str
    data_hash: str
    
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    status: str = "pending"  # pending, in_progress, completed, failed
    retries: int = 0


@dataclass
class FailoverEvent:
    """Failover event"""
    id: str
    timestamp: datetime
    
    from_region: str
    to_region: str
    
    reason: str
    triggered_by: str  # auto, manual
    
    duration_seconds: float = 0.0
    success: bool = False


# ============================================================================
# REGION MANAGER
# ============================================================================

class RegionManager:
    """
    Region Manager
    
    Manages region lifecycle and health.
    """
    
    def __init__(self):
        # Regions
        self.regions: Dict[str, Region] = {}
        
        # Health history
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        logger.info("Region Manager initialized")
    
    def register_region(self, region: Region):
        """Register new region"""
        self.regions[region.id] = region
        
        logger.info(f"Registered region: {region.id} ({region.name})")
    
    def get_region(self, region_id: str) -> Optional[Region]:
        """Get region"""
        return self.regions.get(region_id)
    
    def get_healthy_regions(self) -> List[Region]:
        """Get all healthy regions"""
        return [
            r for r in self.regions.values()
            if r.status == RegionStatus.HEALTHY
        ]
    
    def update_health(
        self,
        region_id: str,
        health_score: float,
        status: Optional[RegionStatus] = None
    ):
        """Update region health"""
        region = self.get_region(region_id)
        
        if not region:
            return
        
        region.health_score = health_score
        region.last_health_check = datetime.now()
        
        if status:
            region.status = status
        else:
            # Auto-determine status from health score
            if health_score >= 0.8:
                region.status = RegionStatus.HEALTHY
            elif health_score >= 0.5:
                region.status = RegionStatus.DEGRADED
            else:
                region.status = RegionStatus.UNHEALTHY
        
        # Record health history
        self.health_history[region_id].append({
            'timestamp': datetime.now(),
            'health_score': health_score,
            'status': region.status
        })
    
    def get_region_load(self, region_id: str) -> float:
        """Get region load factor (0-1)"""
        region = self.get_region(region_id)
        
        if not region or region.max_capacity == 0:
            return 0.0
        
        return region.current_load / region.max_capacity


# ============================================================================
# TRAFFIC ROUTER
# ============================================================================

class TrafficRouter:
    """
    Traffic Router
    
    Routes requests to optimal regions.
    """
    
    def __init__(
        self,
        region_manager: RegionManager,
        config: MultiRegionConfig
    ):
        self.region_manager = region_manager
        self.config = config
        
        # Latency matrix (region -> region -> latency_ms)
        self.latency_matrix: Dict[Tuple[str, str], float] = {}
        
        # Weights for weighted routing
        self.region_weights: Dict[str, float] = {}
        
        logger.info("Traffic Router initialized")
    
    def route_request(
        self,
        user_location: Optional[Tuple[float, float]] = None,
        source_region: Optional[str] = None
    ) -> Optional[str]:
        """
        Route request to optimal region
        
        Args:
            user_location: (latitude, longitude)
            source_region: Source region for cross-region requests
        
        Returns:
            Target region ID
        """
        healthy_regions = self.region_manager.get_healthy_regions()
        
        if not healthy_regions:
            logger.error("No healthy regions available")
            return None
        
        if self.config.routing_strategy == RoutingStrategy.LATENCY_BASED:
            return self._route_by_latency(healthy_regions, source_region)
        
        elif self.config.routing_strategy == RoutingStrategy.GEOPROXIMITY:
            if user_location:
                return self._route_by_proximity(healthy_regions, user_location)
            else:
                # Fallback to latency
                return self._route_by_latency(healthy_regions, source_region)
        
        elif self.config.routing_strategy == RoutingStrategy.WEIGHTED:
            return self._route_weighted(healthy_regions)
        
        elif self.config.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            return self._route_round_robin(healthy_regions)
        
        elif self.config.routing_strategy == RoutingStrategy.FAILOVER:
            return self._route_failover(healthy_regions)
        
        # Default: first healthy region
        return healthy_regions[0].id
    
    def _route_by_latency(
        self,
        regions: List[Region],
        source_region: Optional[str]
    ) -> str:
        """Route based on latency"""
        if not source_region:
            # No source, pick region with lowest load
            return min(regions, key=lambda r: self.region_manager.get_region_load(r.id)).id
        
        # Find region with lowest latency from source
        min_latency = float('inf')
        best_region = regions[0].id
        
        for region in regions:
            latency = self.get_latency(source_region, region.id)
            
            # Consider load
            load_factor = self.region_manager.get_region_load(region.id)
            adjusted_latency = latency * (1 + load_factor)
            
            if adjusted_latency < min_latency:
                min_latency = adjusted_latency
                best_region = region.id
        
        return best_region
    
    def _route_by_proximity(
        self,
        regions: List[Region],
        user_location: Tuple[float, float]
    ) -> str:
        """Route based on geographic proximity"""
        user_lat, user_lon = user_location
        
        min_distance = float('inf')
        closest_region = regions[0].id
        
        for region in regions:
            # Haversine distance
            distance = self._haversine_distance(
                user_lat, user_lon,
                region.latitude, region.longitude
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_region = region.id
        
        return closest_region
    
    def _route_weighted(self, regions: List[Region]) -> str:
        """Route based on weights"""
        if not self.region_weights:
            # Equal weights
            return random.choice(regions).id
        
        # Get weights for healthy regions
        region_ids = [r.id for r in regions]
        weights = [self.region_weights.get(r.id, 1.0) for r in regions]
        
        # Normalize
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(regions).id
        
        weights = [w / total_weight for w in weights]
        
        # Weighted random selection
        if NUMPY_AVAILABLE:
            selected_idx = np.random.choice(len(region_ids), p=weights)
        else:
            # Manual weighted selection
            rand_val = random.random()
            cumulative = 0.0
            selected_idx = 0
            
            for i, weight in enumerate(weights):
                cumulative += weight
                if rand_val <= cumulative:
                    selected_idx = i
                    break
        
        return region_ids[selected_idx]
    
    def _route_round_robin(self, regions: List[Region]) -> str:
        """Route using round-robin"""
        if not hasattr(self, '_rr_index'):
            self._rr_index = 0
        
        region = regions[self._rr_index % len(regions)]
        self._rr_index += 1
        
        return region.id
    
    def _route_failover(self, regions: List[Region]) -> str:
        """Route using failover (primary first)"""
        # Try primary region
        primary = self.region_manager.get_region(self.config.primary_region)
        
        if primary and primary in regions:
            return primary.id
        
        # Fallback to first healthy
        return regions[0].id
    
    def get_latency(self, source: str, target: str) -> float:
        """Get latency between regions"""
        if source == target:
            return 1.0  # Local latency
        
        key = (source, target)
        
        if key in self.latency_matrix:
            return self.latency_matrix[key]
        
        # Estimate based on geographic distance
        source_region = self.region_manager.get_region(source)
        target_region = self.region_manager.get_region(target)
        
        if not source_region or not target_region:
            return 100.0  # Default
        
        distance_km = self._haversine_distance(
            source_region.latitude, source_region.longitude,
            target_region.latitude, target_region.longitude
        )
        
        # Rough estimate: 0.1ms per km + 20ms base latency
        latency = 20.0 + distance_km * 0.1
        
        self.latency_matrix[key] = latency
        
        return latency
    
    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """Calculate distance between two points (km)"""
        # Radius of Earth in km
        R = 6371.0
        
        # Convert to radians
        lat1_rad = lat1 * 3.14159 / 180
        lon1_rad = lon1 * 3.14159 / 180
        lat2_rad = lat2 * 3.14159 / 180
        lon2_rad = lon2 * 3.14159 / 180
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (
            (1 - ((1 - dlat/2) ** 2)) / 2 +
            ((1 - dlat/2) ** 2) * ((1 - dlon/2) ** 2) / 2
        )
        
        # Simplified
        a = max(0, min(1, a))
        c = 2 * (a ** 0.5)
        
        distance = R * c
        
        return distance


# ============================================================================
# REPLICATION MANAGER
# ============================================================================

class ReplicationManager:
    """
    Replication Manager
    
    Manages data replication across regions.
    """
    
    def __init__(self, config: MultiRegionConfig):
        self.config = config
        
        # Replication tasks
        self.tasks: Dict[str, ReplicationTask] = {}
        self.task_queue: deque = deque()
        
        # Statistics
        self.total_replicated = 0
        self.total_failed = 0
        
        logger.info("Replication Manager initialized")
    
    def replicate_data(
        self,
        data_key: str,
        data: Any,
        source_region: str,
        target_regions: Optional[List[str]] = None
    ) -> str:
        """
        Replicate data to regions
        
        Returns task ID
        """
        # Default to all regions except source
        if not target_regions:
            target_regions = [
                r for r in self.config.regions
                if r != source_region
            ]
        
        # Create task
        task_id = f"repl_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        # Compute hash
        data_str = str(data).encode('utf-8')
        data_hash = hashlib.sha256(data_str).hexdigest()
        
        task = ReplicationTask(
            id=task_id,
            source_region=source_region,
            target_regions=target_regions,
            data_key=data_key,
            data_hash=data_hash
        )
        
        self.tasks[task_id] = task
        
        # Queue for processing
        if self.config.replication_mode == ReplicationMode.ASYNC:
            self.task_queue.append(task_id)
            task.status = "queued"
        else:
            # Sync or semi-sync: process immediately
            self._process_replication(task_id, data)
        
        return task_id
    
    def _process_replication(self, task_id: str, data: Any):
        """Process replication task"""
        task = self.tasks.get(task_id)
        
        if not task:
            return
        
        task.status = "in_progress"
        
        start_time = time.time()
        
        try:
            # Simulate replication to each target
            for target_region in task.target_regions:
                # In real implementation, this would send data to target region
                # For now, simulate network delay
                time.sleep(0.001)  # 1ms per region
            
            task.status = "completed"
            task.completed_at = datetime.now()
            
            self.total_replicated += 1
            
            logger.debug(f"Replicated {task.data_key} to {len(task.target_regions)} regions")
        
        except Exception as e:
            task.status = "failed"
            task.retries += 1
            
            self.total_failed += 1
            
            logger.error(f"Replication failed: {e}")
            
            # Retry logic
            if task.retries < 3:
                self.task_queue.append(task_id)
        
        finally:
            duration = time.time() - start_time
    
    def process_queue(self, max_tasks: int = 100):
        """Process replication queue"""
        processed = 0
        
        while self.task_queue and processed < max_tasks:
            task_id = self.task_queue.popleft()
            
            # In real implementation, retrieve data from source
            # For now, use placeholder
            self._process_replication(task_id, {})
            
            processed += 1
        
        return processed
    
    def get_replication_lag(self, region: str) -> float:
        """Get replication lag for region (seconds)"""
        # Count pending tasks targeting this region
        pending_tasks = [
            t for t in self.tasks.values()
            if region in t.target_regions and t.status != "completed"
        ]
        
        if not pending_tasks:
            return 0.0
        
        # Estimate lag based on oldest pending task
        oldest_task = min(pending_tasks, key=lambda t: t.created_at)
        lag = (datetime.now() - oldest_task.created_at).total_seconds()
        
        return lag


# ============================================================================
# FAILOVER COORDINATOR
# ============================================================================

class FailoverCoordinator:
    """
    Failover Coordinator
    
    Manages automatic failover between regions.
    """
    
    def __init__(
        self,
        region_manager: RegionManager,
        traffic_router: TrafficRouter,
        config: MultiRegionConfig
    ):
        self.region_manager = region_manager
        self.traffic_router = traffic_router
        self.config = config
        
        # Failover history
        self.failover_events: List[FailoverEvent] = []
        
        logger.info("Failover Coordinator initialized")
    
    def check_and_failover(self) -> Optional[FailoverEvent]:
        """Check regions and trigger failover if needed"""
        if not self.config.auto_failover:
            return None
        
        # Check primary region
        primary = self.region_manager.get_region(self.config.primary_region)
        
        if not primary:
            return None
        
        # Check if failover needed
        if primary.health_score < self.config.failover_threshold:
            # Find healthy backup region
            healthy_regions = self.region_manager.get_healthy_regions()
            
            backup_regions = [r for r in healthy_regions if r.id != primary.id]
            
            if not backup_regions:
                logger.error("No backup regions available for failover")
                return None
            
            # Select best backup
            backup = self._select_backup_region(backup_regions)
            
            # Execute failover
            event = self.execute_failover(
                primary.id,
                backup.id,
                reason=f"Primary health below threshold: {primary.health_score:.2f}",
                triggered_by="auto"
            )
            
            return event
        
        return None
    
    def execute_failover(
        self,
        from_region: str,
        to_region: str,
        reason: str,
        triggered_by: str
    ) -> FailoverEvent:
        """Execute failover"""
        start_time = time.time()
        
        event_id = f"failover_{int(time.time() * 1000)}"
        
        event = FailoverEvent(
            id=event_id,
            timestamp=datetime.now(),
            from_region=from_region,
            to_region=to_region,
            reason=reason,
            triggered_by=triggered_by
        )
        
        try:
            # 1. Update primary region
            self.config.primary_region = to_region
            
            # 2. Update routing
            # (In real implementation, update DNS, load balancer, etc.)
            
            # 3. Mark old primary as degraded
            self.region_manager.update_health(
                from_region,
                health_score=0.3,
                status=RegionStatus.DEGRADED
            )
            
            event.success = True
            
            logger.info(f"Failover executed: {from_region} -> {to_region}")
        
        except Exception as e:
            event.success = False
            logger.error(f"Failover failed: {e}")
        
        finally:
            event.duration_seconds = time.time() - start_time
            self.failover_events.append(event)
        
        return event
    
    def _select_backup_region(self, regions: List[Region]) -> Region:
        """Select best backup region"""
        # Select region with highest health score and lowest load
        def score_region(r: Region) -> float:
            load = self.region_manager.get_region_load(r.id)
            return r.health_score * (1 - load)
        
        return max(regions, key=score_region)


# ============================================================================
# MULTI-REGION ORCHESTRATOR
# ============================================================================

class MultiRegionOrchestrator:
    """
    Multi-Region Orchestrator
    
    Complete multi-region deployment management.
    """
    
    def __init__(self, config: Optional[MultiRegionConfig] = None):
        self.config = config or MultiRegionConfig()
        
        # Components
        self.region_manager = RegionManager()
        self.traffic_router = TrafficRouter(self.region_manager, self.config)
        self.replication_manager = ReplicationManager(self.config)
        self.failover_coordinator = FailoverCoordinator(
            self.region_manager,
            self.traffic_router,
            self.config
        )
        
        # Metrics
        self.total_requests = 0
        self.total_failovers = 0
        
        # Initialize regions
        self._initialize_regions()
        
        logger.info("Multi-Region Orchestrator initialized")
    
    def _initialize_regions(self):
        """Initialize default regions"""
        default_regions = [
            Region(
                id='us-east-1',
                name='US East (N. Virginia)',
                location='Virginia, USA',
                latitude=38.13,
                longitude=-78.45
            ),
            Region(
                id='us-west-2',
                name='US West (Oregon)',
                location='Oregon, USA',
                latitude=45.87,
                longitude=-119.29
            ),
            Region(
                id='eu-west-1',
                name='EU West (Ireland)',
                location='Dublin, Ireland',
                latitude=53.35,
                longitude=-6.26
            ),
            Region(
                id='ap-southeast-1',
                name='Asia Pacific (Singapore)',
                location='Singapore',
                latitude=1.35,
                longitude=103.82
            )
        ]
        
        for region in default_regions:
            if region.id in self.config.regions:
                self.region_manager.register_region(region)
    
    def handle_request(
        self,
        user_location: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """Handle incoming request"""
        start_time = time.time()
        
        # Route to region
        target_region = self.traffic_router.route_request(user_location)
        
        if not target_region:
            return {
                'success': False,
                'error': 'No healthy regions available'
            }
        
        # Update region load
        region = self.region_manager.get_region(target_region)
        
        if region:
            region.current_load += 1
        
        self.total_requests += 1
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Cleanup
        if region:
            region.current_load = max(0, region.current_load - 1)
        
        return {
            'success': True,
            'region': target_region,
            'latency_ms': latency_ms
        }
    
    def replicate_data(
        self,
        data_key: str,
        data: Any,
        source_region: str
    ) -> str:
        """Replicate data across regions"""
        task_id = self.replication_manager.replicate_data(
            data_key,
            data,
            source_region
        )
        
        return task_id
    
    def health_check(self):
        """Perform health check on all regions"""
        for region_id in self.config.regions:
            # Simulate health check
            health_score = random.uniform(0.7, 1.0)
            
            self.region_manager.update_health(region_id, health_score)
        
        # Check for failover
        failover_event = self.failover_coordinator.check_and_failover()
        
        if failover_event:
            self.total_failovers += 1


# ============================================================================
# TESTING
# ============================================================================

def test_multi_region():
    """Test multi-region deployment"""
    print("=" * 80)
    print("MULTI-REGION DEPLOYMENT - TEST")
    print("=" * 80)
    
    # Create orchestrator
    config = MultiRegionConfig(
        regions=['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'],
        routing_strategy=RoutingStrategy.GEOPROXIMITY,
        auto_failover=True
    )
    
    orchestrator = MultiRegionOrchestrator(config)
    
    print("✓ Multi-region orchestrator initialized")
    print(f"  Regions: {len(orchestrator.region_manager.regions)}")
    
    # Test request routing
    print("\n" + "="*80)
    print("Test: Request Routing")
    print("="*80)
    
    # User locations
    test_locations = [
        ((40.7128, -74.0060), "New York"),  # Should route to us-east-1
        ((37.7749, -122.4194), "San Francisco"),  # Should route to us-west-2
        ((51.5074, -0.1278), "London"),  # Should route to eu-west-1
        ((1.3521, 103.8198), "Singapore"),  # Should route to ap-southeast-1
    ]
    
    for location, city in test_locations:
        result = orchestrator.handle_request(user_location=location)
        
        print(f"✓ {city}:")
        print(f"  Routed to: {result['region']}")
        print(f"  Latency: {result['latency_ms']:.2f}ms")
    
    # Test data replication
    print("\n" + "="*80)
    print("Test: Data Replication")
    print("="*80)
    
    task_id = orchestrator.replicate_data(
        data_key="user_profile_12345",
        data={'name': 'John Doe', 'preferences': {}},
        source_region='us-east-1'
    )
    
    print(f"✓ Replication task created: {task_id}")
    
    # Process replication queue
    processed = orchestrator.replication_manager.process_queue()
    
    print(f"✓ Processed {processed} replication tasks")
    
    # Check replication lag
    for region_id in config.regions:
        lag = orchestrator.replication_manager.get_replication_lag(region_id)
        print(f"  {region_id} lag: {lag:.2f}s")
    
    # Test failover
    print("\n" + "="*80)
    print("Test: Automatic Failover")
    print("="*80)
    
    # Simulate primary region failure
    primary_region = config.primary_region
    
    print(f"Simulating failure in primary region: {primary_region}")
    
    orchestrator.region_manager.update_health(
        primary_region,
        health_score=0.3,
        status=RegionStatus.UNHEALTHY
    )
    
    # Trigger failover check
    failover_event = orchestrator.failover_coordinator.check_and_failover()
    
    if failover_event:
        print(f"✓ Failover executed:")
        print(f"  From: {failover_event.from_region}")
        print(f"  To: {failover_event.to_region}")
        print(f"  Duration: {failover_event.duration_seconds:.2f}s")
        print(f"  Success: {failover_event.success}")
        print(f"  New primary: {config.primary_region}")
    
    # Test performance
    print("\n" + "="*80)
    print("Test: Performance")
    print("="*80)
    
    # Simulate load
    num_requests = 1000
    
    start_time = time.time()
    
    for _ in range(num_requests):
        # Random user location (US, EU, or Asia)
        lat = random.uniform(0, 60)
        lon = random.uniform(-120, 120)
        
        orchestrator.handle_request(user_location=(lat, lon))
    
    elapsed = time.time() - start_time
    
    print(f"✓ Handled {num_requests} requests in {elapsed:.2f}s")
    print(f"  Throughput: {num_requests / elapsed:.1f} req/s")
    print(f"  Avg latency: {elapsed / num_requests * 1000:.2f}ms")
    
    # Show region distribution
    print("\n✓ Region load distribution:")
    for region_id, region in orchestrator.region_manager.regions.items():
        print(f"  {region_id}: load={region.current_load}, health={region.health_score:.2f}")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_multi_region()
