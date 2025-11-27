"""
BATCH PROCESSING FRAMEWORK
==========================

Enterprise-grade batch processing system with:
- Job Scheduler & Queue Manager
- Parallel Batch Processor
- Chunk-Based Processing
- Job Dependencies & DAG
- Retry & Error Handling
- Job Monitoring & Progress Tracking
- Resource Management
- Batch Job History & Audit
- Incremental Processing

Author: Wellomex AI Team
Created: 2025-11-12
"""

import logging
import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import heapq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. BATCH JOB DEFINITION
# ============================================================================

class JobStatus(Enum):
    """Batch job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(Enum):
    """Job priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BatchJob:
    """Batch job definition"""
    id: str
    name: str
    processor: Callable
    input_data: List[Any]
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    chunk_size: int = 100
    max_retries: int = 3
    retry_count: int = 0
    dependencies: Set[str] = field(default_factory=set)
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For priority queue"""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at
    
    def duration(self) -> Optional[float]:
        """Calculate job duration"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return None


# ============================================================================
# 2. BATCH PROCESSOR
# ============================================================================

class BatchProcessor:
    """Process data in configurable batch chunks"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.processed_count = 0
        self.failed_count = 0
        self.total_processing_time = 0.0
        logger.info(f"BatchProcessor initialized (max_workers={max_workers})")
    
    async def process_batch(self, job: BatchJob) -> Any:
        """Process batch job in chunks"""
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        
        try:
            logger.info(f"Processing batch job: {job.name} ({len(job.input_data)} items)")
            
            # Split into chunks
            chunks = self._create_chunks(job.input_data, job.chunk_size)
            total_chunks = len(chunks)
            
            results = []
            
            # Process chunks in parallel
            for i in range(0, total_chunks, self.max_workers):
                batch_chunks = chunks[i:i + self.max_workers]
                
                # Process chunk batch
                tasks = [self._process_chunk(job.processor, chunk) for chunk in batch_chunks]
                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle results
                for result in chunk_results:
                    if isinstance(result, Exception):
                        raise result
                    results.extend(result)
                
                # Update progress
                processed_chunks = min(i + self.max_workers, total_chunks)
                job.progress = processed_chunks / total_chunks
                
                logger.debug(f"Job {job.name}: {job.progress:.1%} complete")
            
            # Job completed
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.result = results
            job.progress = 1.0
            
            self.processed_count += 1
            self.total_processing_time += job.duration()
            
            logger.info(f"Batch job completed: {job.name} ({job.duration():.2f}s)")
            
            return results
        
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()
            
            self.failed_count += 1
            
            logger.error(f"Batch job failed: {job.name} - {e}")
            raise
    
    def _create_chunks(self, data: List[Any], chunk_size: int) -> List[List[Any]]:
        """Split data into chunks"""
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunks.append(data[i:i + chunk_size])
        return chunks
    
    async def _process_chunk(self, processor: Callable, chunk: List[Any]) -> List[Any]:
        """Process single chunk"""
        if asyncio.iscoroutinefunction(processor):
            return await processor(chunk)
        else:
            return processor(chunk)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        total_jobs = self.processed_count + self.failed_count
        
        return {
            "processed": self.processed_count,
            "failed": self.failed_count,
            "success_rate": self.processed_count / max(1, total_jobs),
            "avg_processing_time": self.total_processing_time / max(1, self.processed_count)
        }


# ============================================================================
# 3. JOB SCHEDULER
# ============================================================================

class JobScheduler:
    """Schedule and manage batch jobs"""
    
    def __init__(self, processor: BatchProcessor):
        self.processor = processor
        self.jobs: Dict[str, BatchJob] = {}
        self.job_queue: List[BatchJob] = []
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)  # job_id -> dependent_job_ids
        self.running_jobs: Set[str] = set()
        self.scheduled_count = 0
        self.completed_count = 0
        logger.info("JobScheduler initialized")
    
    def schedule_job(self, job: BatchJob) -> str:
        """Schedule batch job"""
        self.jobs[job.id] = job
        
        # Track dependencies
        for dep_id in job.dependencies:
            self.dependencies[dep_id].add(job.id)
        
        # Add to queue if no dependencies or dependencies met
        if self._dependencies_met(job):
            heapq.heappush(self.job_queue, job)
        
        self.scheduled_count += 1
        logger.info(f"Scheduled job: {job.name} ({job.id})")
        
        return job.id
    
    def _dependencies_met(self, job: BatchJob) -> bool:
        """Check if job dependencies are satisfied"""
        for dep_id in job.dependencies:
            dep_job = self.jobs.get(dep_id)
            if not dep_job or dep_job.status != JobStatus.COMPLETED:
                return False
        return True
    
    async def run_scheduler(self, max_concurrent: int = 4) -> None:
        """Run job scheduler"""
        logger.info("Job scheduler started")
        
        while self.job_queue or self.running_jobs:
            # Start jobs up to max concurrent
            while len(self.running_jobs) < max_concurrent and self.job_queue:
                job = heapq.heappop(self.job_queue)
                
                if not self._dependencies_met(job):
                    # Re-check dependencies, skip if not met
                    continue
                
                self.running_jobs.add(job.id)
                asyncio.create_task(self._execute_job(job))
            
            # Wait a bit before next cycle
            await asyncio.sleep(0.1)
        
        logger.info("Job scheduler finished")
    
    async def _execute_job(self, job: BatchJob) -> None:
        """Execute single job"""
        try:
            await self.processor.process_batch(job)
            
            # Mark as completed
            self.completed_count += 1
            
            # Check for dependent jobs
            self._trigger_dependent_jobs(job.id)
        
        except Exception as e:
            # Handle retry
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                job.status = JobStatus.RETRYING
                
                # Re-schedule with delay
                await asyncio.sleep(2 ** job.retry_count)
                heapq.heappush(self.job_queue, job)
                
                logger.info(f"Retrying job {job.name} (attempt {job.retry_count})")
            else:
                logger.error(f"Job {job.name} failed after {job.max_retries} retries")
        
        finally:
            self.running_jobs.discard(job.id)
    
    def _trigger_dependent_jobs(self, completed_job_id: str) -> None:
        """Trigger jobs that depend on completed job"""
        dependent_ids = self.dependencies.get(completed_job_id, set())
        
        for dep_id in dependent_ids:
            dep_job = self.jobs.get(dep_id)
            
            if dep_job and self._dependencies_met(dep_job):
                heapq.heappush(self.job_queue, dep_job)
                logger.debug(f"Triggered dependent job: {dep_job.name}")
    
    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel pending job"""
        job = self.jobs.get(job_id)
        
        if job and job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            logger.info(f"Cancelled job: {job.name}")
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        status_counts = defaultdict(int)
        for job in self.jobs.values():
            status_counts[job.status.value] += 1
        
        return {
            "scheduled": self.scheduled_count,
            "completed": self.completed_count,
            "running": len(self.running_jobs),
            "queued": len(self.job_queue),
            "status_distribution": dict(status_counts)
        }


# ============================================================================
# 4. JOB DEPENDENCY GRAPH (DAG)
# ============================================================================

class JobDAG:
    """Directed Acyclic Graph for job dependencies"""
    
    def __init__(self):
        self.graph: Dict[str, Set[str]] = defaultdict(set)  # job -> dependencies
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)  # job -> dependents
        self.execution_order: List[str] = []
        logger.info("JobDAG initialized")
    
    def add_job(self, job_id: str, dependencies: Optional[Set[str]] = None) -> None:
        """Add job to DAG"""
        if dependencies:
            self.graph[job_id] = dependencies
            
            for dep in dependencies:
                self.reverse_graph[dep].add(job_id)
        else:
            self.graph[job_id] = set()
        
        logger.debug(f"Added job to DAG: {job_id}")
    
    def validate(self) -> bool:
        """Validate DAG has no cycles"""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for dep in self.graph.get(node, set()):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for job_id in self.graph:
            if job_id not in visited:
                if has_cycle(job_id):
                    logger.error("Cycle detected in job DAG")
                    return False
        
        return True
    
    def topological_sort(self) -> List[str]:
        """Get execution order using topological sort"""
        in_degree = {job: len(deps) for job, deps in self.graph.items()}
        
        # Jobs with no dependencies
        queue = deque([job for job, degree in in_degree.items() if degree == 0])
        execution_order = []
        
        while queue:
            job = queue.popleft()
            execution_order.append(job)
            
            # Reduce in-degree for dependent jobs
            for dependent in self.reverse_graph.get(job, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(execution_order) != len(self.graph):
            logger.error("DAG contains cycle, cannot determine execution order")
            return []
        
        self.execution_order = execution_order
        logger.info(f"Computed execution order for {len(execution_order)} jobs")
        
        return execution_order
    
    def get_parallel_groups(self) -> List[List[str]]:
        """Get groups of jobs that can run in parallel"""
        execution_order = self.topological_sort()
        
        if not execution_order:
            return []
        
        groups = []
        level_map = {}
        
        # Assign level to each job
        for job in execution_order:
            deps = self.graph[job]
            if not deps:
                level_map[job] = 0
            else:
                max_dep_level = max(level_map[dep] for dep in deps)
                level_map[job] = max_dep_level + 1
        
        # Group by level
        max_level = max(level_map.values())
        for level in range(max_level + 1):
            group = [job for job, l in level_map.items() if l == level]
            if group:
                groups.append(group)
        
        logger.info(f"Identified {len(groups)} parallel execution groups")
        return groups
    
    def get_stats(self) -> Dict[str, Any]:
        """Get DAG statistics"""
        return {
            "total_jobs": len(self.graph),
            "jobs_with_dependencies": sum(1 for deps in self.graph.values() if deps),
            "max_dependencies": max((len(deps) for deps in self.graph.values()), default=0)
        }


# ============================================================================
# 5. JOB MONITORING & PROGRESS TRACKING
# ============================================================================

class JobMonitor:
    """Monitor job execution and track progress"""
    
    def __init__(self):
        self.job_metrics: Dict[str, Dict[str, Any]] = {}
        self.checkpoints: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        logger.info("JobMonitor initialized")
    
    def start_monitoring(self, job: BatchJob) -> None:
        """Start monitoring job"""
        self.job_metrics[job.id] = {
            "name": job.name,
            "start_time": time.time(),
            "samples": [],
            "checkpoints": []
        }
        logger.debug(f"Started monitoring: {job.name}")
    
    def record_progress(self, job_id: str, progress: float, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Record job progress"""
        if job_id not in self.job_metrics:
            return
        
        sample = {
            "timestamp": time.time(),
            "progress": progress,
            "metrics": metrics or {}
        }
        
        self.job_metrics[job_id]["samples"].append(sample)
        logger.debug(f"Recorded progress for {job_id}: {progress:.1%}")
    
    def create_checkpoint(self, job_id: str, checkpoint_data: Dict[str, Any]) -> None:
        """Create checkpoint for job recovery"""
        checkpoint = {
            "id": f"cp-{uuid.uuid4().hex[:8]}",
            "timestamp": time.time(),
            "data": checkpoint_data
        }
        
        self.checkpoints[job_id].append(checkpoint)
        logger.info(f"Created checkpoint for job {job_id}")
    
    def get_latest_checkpoint(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get latest checkpoint for job"""
        checkpoints = self.checkpoints.get(job_id, [])
        return checkpoints[-1] if checkpoints else None
    
    def estimate_completion(self, job_id: str) -> Optional[float]:
        """Estimate time to completion"""
        if job_id not in self.job_metrics:
            return None
        
        metrics = self.job_metrics[job_id]
        samples = metrics["samples"]
        
        if len(samples) < 2:
            return None
        
        # Calculate processing rate
        first_sample = samples[0]
        last_sample = samples[-1]
        
        progress_made = last_sample["progress"] - first_sample["progress"]
        time_elapsed = last_sample["timestamp"] - first_sample["timestamp"]
        
        if progress_made <= 0 or time_elapsed <= 0:
            return None
        
        rate = progress_made / time_elapsed
        remaining_progress = 1.0 - last_sample["progress"]
        
        estimated_time = remaining_progress / rate
        
        return estimated_time
    
    def get_job_report(self, job_id: str) -> Dict[str, Any]:
        """Get comprehensive job report"""
        if job_id not in self.job_metrics:
            return {}
        
        metrics = self.job_metrics[job_id]
        samples = metrics["samples"]
        
        if not samples:
            return {"job_id": job_id, "status": "no_data"}
        
        latest = samples[-1]
        
        return {
            "job_id": job_id,
            "name": metrics["name"],
            "progress": latest["progress"],
            "elapsed_time": time.time() - metrics["start_time"],
            "estimated_completion": self.estimate_completion(job_id),
            "checkpoints": len(self.checkpoints.get(job_id, [])),
            "samples_collected": len(samples)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitor statistics"""
        return {
            "monitored_jobs": len(self.job_metrics),
            "total_checkpoints": sum(len(cp) for cp in self.checkpoints.values()),
            "total_samples": sum(len(m["samples"]) for m in self.job_metrics.values())
        }


# ============================================================================
# 6. INCREMENTAL PROCESSING
# ============================================================================

class IncrementalProcessor:
    """Process data incrementally with state tracking"""
    
    def __init__(self):
        self.state: Dict[str, Dict[str, Any]] = {}
        self.watermarks: Dict[str, float] = {}  # job_id -> timestamp
        self.processed_count = 0
        logger.info("IncrementalProcessor initialized")
    
    def save_state(self, job_id: str, state_data: Dict[str, Any]) -> None:
        """Save processing state"""
        self.state[job_id] = {
            "timestamp": time.time(),
            "data": state_data
        }
        logger.debug(f"Saved state for job: {job_id}")
    
    def load_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load processing state"""
        state = self.state.get(job_id)
        return state["data"] if state else None
    
    def set_watermark(self, job_id: str, timestamp: float) -> None:
        """Set watermark for incremental processing"""
        self.watermarks[job_id] = timestamp
        logger.debug(f"Set watermark for {job_id}: {timestamp}")
    
    def get_watermark(self, job_id: str) -> Optional[float]:
        """Get current watermark"""
        return self.watermarks.get(job_id)
    
    async def process_incremental(self, job_id: str, processor: Callable, 
                                  data: List[Any], timestamp_field: str = "timestamp") -> Any:
        """Process only new data since last watermark"""
        watermark = self.get_watermark(job_id) or 0
        
        # Filter new data
        new_data = [
            item for item in data 
            if item.get(timestamp_field, 0) > watermark
        ]
        
        if not new_data:
            logger.info(f"No new data to process for job: {job_id}")
            return []
        
        logger.info(f"Processing {len(new_data)} new items for job: {job_id}")
        
        # Process new data
        if asyncio.iscoroutinefunction(processor):
            result = await processor(new_data)
        else:
            result = processor(new_data)
        
        # Update watermark
        max_timestamp = max(item.get(timestamp_field, 0) for item in new_data)
        self.set_watermark(job_id, max_timestamp)
        
        self.processed_count += len(new_data)
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            "jobs_with_state": len(self.state),
            "watermarks": len(self.watermarks),
            "items_processed": self.processed_count
        }


# ============================================================================
# 7. RESOURCE MANAGER
# ============================================================================

class ResourceManager:
    """Manage computational resources for batch jobs"""
    
    def __init__(self, max_memory_mb: float = 1024, max_cpu_percent: float = 80):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.current_memory_mb = 0.0
        self.current_cpu_percent = 0.0
        self.allocations: Dict[str, Dict[str, float]] = {}
        logger.info(f"ResourceManager initialized (mem={max_memory_mb}MB, cpu={max_cpu_percent}%)")
    
    def can_allocate(self, job_id: str, memory_mb: float, cpu_percent: float) -> bool:
        """Check if resources can be allocated"""
        available_memory = self.max_memory_mb - self.current_memory_mb
        available_cpu = self.max_cpu_percent - self.current_cpu_percent
        
        return memory_mb <= available_memory and cpu_percent <= available_cpu
    
    def allocate(self, job_id: str, memory_mb: float, cpu_percent: float) -> bool:
        """Allocate resources for job"""
        if not self.can_allocate(job_id, memory_mb, cpu_percent):
            logger.warning(f"Insufficient resources for job: {job_id}")
            return False
        
        self.allocations[job_id] = {
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent,
            "allocated_at": time.time()
        }
        
        self.current_memory_mb += memory_mb
        self.current_cpu_percent += cpu_percent
        
        logger.info(f"Allocated resources for {job_id}: {memory_mb}MB, {cpu_percent}% CPU")
        return True
    
    def release(self, job_id: str) -> None:
        """Release job resources"""
        if job_id not in self.allocations:
            return
        
        allocation = self.allocations[job_id]
        self.current_memory_mb -= allocation["memory_mb"]
        self.current_cpu_percent -= allocation["cpu_percent"]
        
        del self.allocations[job_id]
        
        logger.info(f"Released resources for job: {job_id}")
    
    def get_utilization(self) -> Dict[str, float]:
        """Get current resource utilization"""
        return {
            "memory_percent": (self.current_memory_mb / self.max_memory_mb) * 100,
            "cpu_percent": (self.current_cpu_percent / self.max_cpu_percent) * 100,
            "memory_mb": self.current_memory_mb,
            "cpu_percent": self.current_cpu_percent
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource statistics"""
        util = self.get_utilization()
        
        return {
            "active_allocations": len(self.allocations),
            "memory_utilization": util["memory_percent"],
            "cpu_utilization": util["cpu_percent"],
            "available_memory_mb": self.max_memory_mb - self.current_memory_mb,
            "available_cpu_percent": self.max_cpu_percent - self.current_cpu_percent
        }


# ============================================================================
# DEMONSTRATION
# ============================================================================

async def demo_batch_processing():
    """Comprehensive demonstration of batch processing framework"""
    
    print("=" * 80)
    print("BATCH PROCESSING FRAMEWORK")
    print("=" * 80)
    print()
    
    print("🏗️  COMPONENTS:")
    print("   1. Batch Job Definition")
    print("   2. Parallel Batch Processor")
    print("   3. Job Scheduler with Dependencies")
    print("   4. Job Dependency DAG")
    print("   5. Job Monitoring & Progress Tracking")
    print("   6. Incremental Processing")
    print("   7. Resource Management")
    print()
    
    # ========================================================================
    # 1. Create Batch Jobs
    # ========================================================================
    print("=" * 80)
    print("1. BATCH JOB DEFINITION")
    print("=" * 80)
    
    # Define sample processors
    async def process_nutrition_data(chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process nutrition data chunk"""
        await asyncio.sleep(0.01)  # Simulate processing
        return [{"id": item["id"], "calories_doubled": item["calories"] * 2} for item in chunk]
    
    async def aggregate_results(chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate processed results"""
        await asyncio.sleep(0.01)
        total = sum(item.get("calories_doubled", 0) for item in chunk)
        return [{"chunk_total": total, "count": len(chunk)}]
    
    # Create sample data
    sample_data = [
        {"id": f"food-{i}", "name": f"Food {i}", "calories": 100 + i * 10}
        for i in range(500)
    ]
    
    print(f"\n📦 Created sample dataset: {len(sample_data)} items")
    
    # Create jobs
    job1 = BatchJob(
        id="job-1",
        name="Process Nutrition Data",
        processor=process_nutrition_data,
        input_data=sample_data,
        chunk_size=50,
        priority=JobPriority.HIGH
    )
    
    job2 = BatchJob(
        id="job-2",
        name="Aggregate Results",
        processor=aggregate_results,
        input_data=[],  # Will be set after job1
        chunk_size=100,
        dependencies={"job-1"}  # Depends on job1
    )
    
    print(f"   ✓ Created 2 batch jobs")
    print(f"   Job 1: {job1.name} ({len(job1.input_data)} items, chunk_size={job1.chunk_size})")
    print(f"   Job 2: {job2.name} (depends on Job 1)")
    
    # ========================================================================
    # 2. Batch Processor
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. PARALLEL BATCH PROCESSOR")
    print("=" * 80)
    
    processor = BatchProcessor(max_workers=4)
    
    print(f"\n▶️  Processing job 1...")
    result1 = await processor.process_batch(job1)
    
    print(f"   ✓ Processed {len(result1)} results")
    print(f"   Duration: {job1.duration():.2f}s")
    print(f"   Progress: {job1.progress:.1%}")
    
    proc_stats = processor.get_stats()
    print(f"\n📊 Processor Statistics:")
    print(f"   Processed Jobs: {proc_stats['processed']}")
    print(f"   Success Rate: {proc_stats['success_rate']:.1%}")
    print(f"   Avg Processing Time: {proc_stats['avg_processing_time']:.2f}s")
    
    # ========================================================================
    # 3. Job Scheduler
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. JOB SCHEDULER WITH DEPENDENCIES")
    print("=" * 80)
    
    scheduler = JobScheduler(processor)
    
    print("\n📅 Scheduling jobs...")
    
    # Create more jobs for scheduling
    job3 = BatchJob(
        id="job-3",
        name="Data Validation",
        processor=process_nutrition_data,
        input_data=sample_data[:100],
        chunk_size=25,
        priority=JobPriority.CRITICAL
    )
    
    job4 = BatchJob(
        id="job-4",
        name="Final Report",
        processor=aggregate_results,
        input_data=result1,
        chunk_size=50,
        dependencies={"job-1", "job-3"}  # Depends on both
    )
    
    scheduler.schedule_job(job1)
    scheduler.schedule_job(job2)
    scheduler.schedule_job(job3)
    scheduler.schedule_job(job4)
    
    print(f"   ✓ Scheduled 4 jobs")
    
    sched_stats = scheduler.get_stats()
    print(f"\n📊 Scheduler Statistics:")
    print(f"   Scheduled: {sched_stats['scheduled']}")
    print(f"   Queued: {sched_stats['queued']}")
    print(f"   Status Distribution: {sched_stats['status_distribution']}")
    
    # Note: Not running full scheduler in demo to save time
    
    # ========================================================================
    # 4. Job DAG
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. JOB DEPENDENCY DAG")
    print("=" * 80)
    
    dag = JobDAG()
    
    print("\n🔗 Building dependency graph...")
    
    dag.add_job("job-1", set())
    dag.add_job("job-2", {"job-1"})
    dag.add_job("job-3", set())
    dag.add_job("job-4", {"job-1", "job-3"})
    
    print(f"   ✓ Added 4 jobs to DAG")
    
    # Validate DAG
    is_valid = dag.validate()
    print(f"   DAG is valid (no cycles): {is_valid}")
    
    # Get execution order
    execution_order = dag.topological_sort()
    print(f"\n📋 Execution Order:")
    for i, job_id in enumerate(execution_order, 1):
        print(f"   {i}. {job_id}")
    
    # Get parallel groups
    parallel_groups = dag.get_parallel_groups()
    print(f"\n⚡ Parallel Execution Groups:")
    for i, group in enumerate(parallel_groups, 1):
        print(f"   Group {i}: {', '.join(group)}")
    
    dag_stats = dag.get_stats()
    print(f"\n📊 DAG Statistics:")
    print(f"   Total Jobs: {dag_stats['total_jobs']}")
    print(f"   Jobs with Dependencies: {dag_stats['jobs_with_dependencies']}")
    print(f"   Max Dependencies: {dag_stats['max_dependencies']}")
    
    # ========================================================================
    # 5. Job Monitoring
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. JOB MONITORING & PROGRESS TRACKING")
    print("=" * 80)
    
    monitor = JobMonitor()
    
    print("\n📊 Monitoring job execution...")
    
    # Start monitoring
    monitor.start_monitoring(job1)
    
    # Simulate progress updates
    for progress in [0.2, 0.4, 0.6, 0.8, 1.0]:
        monitor.record_progress(job1.id, progress, {
            "items_processed": int(progress * len(sample_data)),
            "memory_mb": 50 + progress * 20
        })
    
    # Create checkpoints
    monitor.create_checkpoint(job1.id, {
        "processed_items": 250,
        "last_item_id": "food-249"
    })
    
    print(f"   ✓ Recorded 5 progress samples")
    print(f"   ✓ Created 1 checkpoint")
    
    # Get report
    report = monitor.get_job_report(job1.id)
    print(f"\n📄 Job Report:")
    print(f"   Job: {report['name']}")
    print(f"   Progress: {report['progress']:.1%}")
    print(f"   Elapsed Time: {report['elapsed_time']:.2f}s")
    print(f"   Checkpoints: {report['checkpoints']}")
    
    mon_stats = monitor.get_stats()
    print(f"\n📊 Monitor Statistics:")
    print(f"   Monitored Jobs: {mon_stats['monitored_jobs']}")
    print(f"   Total Checkpoints: {mon_stats['total_checkpoints']}")
    print(f"   Total Samples: {mon_stats['total_samples']}")
    
    # ========================================================================
    # 6. Incremental Processing
    # ========================================================================
    print("\n" + "=" * 80)
    print("6. INCREMENTAL PROCESSING")
    print("=" * 80)
    
    incremental = IncrementalProcessor()
    
    print("\n🔄 Processing data incrementally...")
    
    # Create timestamped data
    timestamped_data = [
        {"id": i, "value": i * 10, "timestamp": time.time() - (100 - i)}
        for i in range(100)
    ]
    
    # First run - process all
    async def simple_processor(data):
        return [{"processed": item["id"]} for item in data]
    
    result = await incremental.process_incremental(
        "incremental-job-1",
        simple_processor,
        timestamped_data
    )
    
    print(f"   ✓ First run: processed {len(result)} items")
    
    # Add new data
    new_data = [
        {"id": i, "value": i * 10, "timestamp": time.time() + i}
        for i in range(100, 120)
    ]
    
    all_data = timestamped_data + new_data
    
    # Second run - process only new
    result = await incremental.process_incremental(
        "incremental-job-1",
        simple_processor,
        all_data
    )
    
    print(f"   ✓ Second run: processed {len(result)} new items only")
    
    inc_stats = incremental.get_stats()
    print(f"\n📊 Incremental Processing Statistics:")
    print(f"   Jobs with State: {inc_stats['jobs_with_state']}")
    print(f"   Watermarks: {inc_stats['watermarks']}")
    print(f"   Total Items Processed: {inc_stats['items_processed']}")
    
    # ========================================================================
    # 7. Resource Management
    # ========================================================================
    print("\n" + "=" * 80)
    print("7. RESOURCE MANAGEMENT")
    print("=" * 80)
    
    resource_mgr = ResourceManager(max_memory_mb=2048, max_cpu_percent=100)
    
    print("\n💻 Managing computational resources...")
    
    # Allocate resources for jobs
    allocations = [
        ("job-1", 512, 25),
        ("job-2", 256, 15),
        ("job-3", 1024, 50),
    ]
    
    for job_id, mem, cpu in allocations:
        success = resource_mgr.allocate(job_id, mem, cpu)
        status = "✓" if success else "✗"
        print(f"   {status} {job_id}: {mem}MB, {cpu}% CPU")
    
    # Try to allocate too much
    can_allocate = resource_mgr.can_allocate("job-4", 1000, 30)
    print(f"   Can allocate job-4 (1000MB, 30% CPU): {can_allocate}")
    
    # Get utilization
    util = resource_mgr.get_utilization()
    print(f"\n📊 Resource Utilization:")
    print(f"   Memory: {util['memory_mb']:.0f}MB ({util['memory_percent']:.1f}%)")
    print(f"   CPU: {util['cpu_percent']:.1f}%")
    
    # Release resources
    resource_mgr.release("job-1")
    print(f"\n   ✓ Released resources for job-1")
    
    util = resource_mgr.get_utilization()
    print(f"   Updated Memory: {util['memory_mb']:.0f}MB ({util['memory_percent']:.1f}%)")
    
    res_stats = resource_mgr.get_stats()
    print(f"\n📊 Resource Manager Statistics:")
    print(f"   Active Allocations: {res_stats['active_allocations']}")
    print(f"   Available Memory: {res_stats['available_memory_mb']:.0f}MB")
    print(f"   Available CPU: {res_stats['available_cpu_percent']:.1f}%")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ BATCH PROCESSING FRAMEWORK COMPLETE")
    print("=" * 80)
    
    print("\n📦 CAPABILITIES:")
    print("   ✓ Configurable batch job definition")
    print("   ✓ Parallel chunk-based processing")
    print("   ✓ Job scheduling with dependencies")
    print("   ✓ DAG-based execution planning")
    print("   ✓ Real-time job monitoring & progress tracking")
    print("   ✓ Incremental processing with watermarks")
    print("   ✓ Resource allocation & management")
    
    print("\n🎯 FRAMEWORK METRICS:")
    print(f"   Jobs processed: {proc_stats['processed']} ✓")
    print(f"   Jobs scheduled: {sched_stats['scheduled']} ✓")
    print(f"   DAG jobs: {dag_stats['total_jobs']} ✓")
    print(f"   Monitored jobs: {mon_stats['monitored_jobs']} ✓")
    print(f"   Incremental items: {inc_stats['items_processed']} ✓")
    print(f"   Resource allocations: {len(allocations)} ✓")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(demo_batch_processing())
