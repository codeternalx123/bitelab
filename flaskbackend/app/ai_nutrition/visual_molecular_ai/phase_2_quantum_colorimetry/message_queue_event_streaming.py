"""
MESSAGE QUEUE & EVENT STREAMING INFRASTRUCTURE
==============================================

Enterprise-grade message queue and event streaming system with:
- Message Queue with Priority & Dead Letter Queue
- Event Bus with Topic-Based Routing
- Event Sourcing & CQRS Pattern
- Stream Processing Engine
- Message Broker with Pub/Sub
- Event Store with Snapshots
- Saga Pattern for Distributed Transactions
- Event Replay & Time Travel

Author: Wellomex AI Team
Created: 2025-11-12
"""

import logging
import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime, timedelta
import heapq
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. MESSAGE QUEUE WITH PRIORITY & DLQ
# ============================================================================

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Message:
    """Message structure"""
    id: str
    topic: str
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3
    ttl: Optional[float] = None  # Time to live in seconds
    
    def __lt__(self, other):
        """For priority queue comparison"""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.timestamp < other.timestamp
    
    def is_expired(self) -> bool:
        """Check if message has exceeded TTL"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class MessageQueue:
    """Priority message queue with dead letter queue"""
    
    def __init__(self, name: str, max_size: int = 10000):
        self.name = name
        self.max_size = max_size
        self.queue: List[Message] = []
        self.dlq: List[Message] = []  # Dead letter queue
        self.processed_count = 0
        self.failed_count = 0
        self.dlq_reasons: Dict[str, str] = {}
        logger.info(f"MessageQueue '{name}' initialized")
    
    def enqueue(self, message: Message) -> bool:
        """Add message to queue"""
        if len(self.queue) >= self.max_size:
            logger.warning(f"Queue '{self.name}' is full")
            return False
        
        heapq.heappush(self.queue, message)
        logger.debug(f"Message {message.id} enqueued to '{self.name}'")
        return True
    
    def dequeue(self) -> Optional[Message]:
        """Remove and return highest priority message"""
        while self.queue:
            message = heapq.heappop(self.queue)
            
            # Check if message expired
            if message.is_expired():
                self._move_to_dlq(message, "expired")
                continue
            
            return message
        
        return None
    
    def peek(self) -> Optional[Message]:
        """View highest priority message without removing"""
        while self.queue:
            message = self.queue[0]
            if message.is_expired():
                heapq.heappop(self.queue)
                self._move_to_dlq(message, "expired")
                continue
            return message
        return None
    
    def ack(self, message: Message) -> None:
        """Acknowledge successful processing"""
        self.processed_count += 1
        logger.debug(f"Message {message.id} acknowledged")
    
    def nack(self, message: Message, reason: str = "processing_failed") -> None:
        """Negative acknowledgment - retry or move to DLQ"""
        message.retry_count += 1
        
        if message.retry_count >= message.max_retries:
            self._move_to_dlq(message, reason)
        else:
            # Re-enqueue with exponential backoff
            message.timestamp = time.time() + (2 ** message.retry_count)
            self.enqueue(message)
            logger.debug(f"Message {message.id} requeued (attempt {message.retry_count})")
    
    def _move_to_dlq(self, message: Message, reason: str) -> None:
        """Move message to dead letter queue"""
        self.dlq.append(message)
        self.dlq_reasons[message.id] = reason
        self.failed_count += 1
        logger.warning(f"Message {message.id} moved to DLQ: {reason}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "name": self.name,
            "size": len(self.queue),
            "dlq_size": len(self.dlq),
            "processed": self.processed_count,
            "failed": self.failed_count,
            "success_rate": self.processed_count / max(1, self.processed_count + self.failed_count)
        }


# ============================================================================
# 2. EVENT BUS WITH TOPIC-BASED ROUTING
# ============================================================================

@dataclass
class Event:
    """Event structure"""
    id: str
    type: str
    source: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Topic-based event bus with pattern matching"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_count = 0
        self.topic_stats: Dict[str, int] = defaultdict(int)
        logger.info("EventBus initialized")
    
    def subscribe(self, topic: str, handler: Callable) -> None:
        """Subscribe to events on a topic"""
        self.subscribers[topic].append(handler)
        logger.info(f"Subscribed to topic '{topic}'")
    
    def unsubscribe(self, topic: str, handler: Callable) -> None:
        """Unsubscribe from topic"""
        if topic in self.subscribers:
            self.subscribers[topic].remove(handler)
            logger.info(f"Unsubscribed from topic '{topic}'")
    
    async def publish(self, topic: str, event: Event) -> int:
        """Publish event to topic"""
        self.event_count += 1
        self.topic_stats[topic] += 1
        
        # Find matching subscribers (supports wildcards)
        handlers = self._match_subscribers(topic)
        
        if not handlers:
            logger.debug(f"No subscribers for topic '{topic}'")
            return 0
        
        # Notify all subscribers
        tasks = []
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.debug(f"Event {event.id} published to {len(handlers)} subscribers")
        return len(handlers)
    
    def _match_subscribers(self, topic: str) -> List[Callable]:
        """Match topic to subscribers (supports wildcards)"""
        handlers = []
        
        # Exact match
        if topic in self.subscribers:
            handlers.extend(self.subscribers[topic])
        
        # Wildcard matches (e.g., "user.*" matches "user.created")
        for pattern, subs in self.subscribers.items():
            if self._topic_matches(topic, pattern):
                handlers.extend(subs)
        
        return list(set(handlers))  # Remove duplicates
    
    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern with wildcards"""
        if pattern == topic:
            return True
        
        if '*' not in pattern and '#' not in pattern:
            return False
        
        # Convert pattern to regex-like matching
        # * matches single level, # matches multiple levels
        topic_parts = topic.split('.')
        pattern_parts = pattern.split('.')
        
        i = j = 0
        while i < len(topic_parts) and j < len(pattern_parts):
            if pattern_parts[j] == '#':
                return True  # # matches everything after
            elif pattern_parts[j] == '*' or pattern_parts[j] == topic_parts[i]:
                i += 1
                j += 1
            else:
                return False
        
        return i == len(topic_parts) and j == len(pattern_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            "total_events": self.event_count,
            "topics": len(self.subscribers),
            "subscribers": sum(len(subs) for subs in self.subscribers.values()),
            "topic_stats": dict(self.topic_stats)
        }


# ============================================================================
# 3. EVENT SOURCING & CQRS PATTERN
# ============================================================================

class Command(ABC):
    """Base command for CQRS"""
    @abstractmethod
    def execute(self) -> Any:
        pass


class Query(ABC):
    """Base query for CQRS"""
    @abstractmethod
    def execute(self) -> Any:
        pass


@dataclass
class AggregateRoot:
    """Aggregate root for event sourcing"""
    id: str
    version: int = 0
    uncommitted_events: List[Event] = field(default_factory=list)


class EventStore:
    """Event store with snapshots"""
    
    def __init__(self, snapshot_interval: int = 10):
        self.events: Dict[str, List[Event]] = defaultdict(list)
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.snapshot_interval = snapshot_interval
        self.event_count = 0
        logger.info(f"EventStore initialized (snapshot every {snapshot_interval} events)")
    
    def append_event(self, aggregate_id: str, event: Event) -> None:
        """Append event to stream"""
        self.events[aggregate_id].append(event)
        self.event_count += 1
        
        # Create snapshot if needed
        if len(self.events[aggregate_id]) % self.snapshot_interval == 0:
            self._create_snapshot(aggregate_id)
        
        logger.debug(f"Event {event.id} appended to aggregate {aggregate_id}")
    
    def get_events(self, aggregate_id: str, from_version: int = 0) -> List[Event]:
        """Get events for aggregate from version"""
        return [e for e in self.events.get(aggregate_id, []) 
                if self.events[aggregate_id].index(e) >= from_version]
    
    def _create_snapshot(self, aggregate_id: str) -> None:
        """Create snapshot of aggregate state"""
        events = self.events[aggregate_id]
        if not events:
            return
        
        # Build state from events
        state = self._rebuild_state(events)
        self.snapshots[aggregate_id] = {
            "version": len(events),
            "state": state,
            "timestamp": time.time()
        }
        logger.info(f"Snapshot created for aggregate {aggregate_id} at version {len(events)}")
    
    def _rebuild_state(self, events: List[Event]) -> Dict[str, Any]:
        """Rebuild aggregate state from events"""
        state = {}
        for event in events:
            # Simple state building - extend based on event types
            state.update(event.data)
        return state
    
    def load_aggregate(self, aggregate_id: str) -> Optional[Dict[str, Any]]:
        """Load aggregate from snapshot + events"""
        snapshot = self.snapshots.get(aggregate_id)
        
        if snapshot:
            # Load from snapshot
            state = snapshot["state"].copy()
            version = snapshot["version"]
            
            # Apply events after snapshot
            remaining_events = self.get_events(aggregate_id, version)
            for event in remaining_events:
                state.update(event.data)
            
            logger.debug(f"Loaded aggregate {aggregate_id} from snapshot + {len(remaining_events)} events")
            return state
        else:
            # Load from all events
            events = self.events.get(aggregate_id, [])
            if not events:
                return None
            
            state = self._rebuild_state(events)
            logger.debug(f"Loaded aggregate {aggregate_id} from {len(events)} events")
            return state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event store statistics"""
        return {
            "total_events": self.event_count,
            "aggregates": len(self.events),
            "snapshots": len(self.snapshots),
            "avg_events_per_aggregate": self.event_count / max(1, len(self.events))
        }


# ============================================================================
# 4. STREAM PROCESSING ENGINE
# ============================================================================

class StreamProcessor:
    """Stream processing with windowing and aggregation"""
    
    def __init__(self, window_size: int = 60, slide_size: int = 10):
        self.window_size = window_size  # seconds
        self.slide_size = slide_size    # seconds
        self.streams: Dict[str, deque] = defaultdict(deque)
        self.processors: Dict[str, Callable] = {}
        self.results: Dict[str, List[Any]] = defaultdict(list)
        logger.info(f"StreamProcessor initialized (window={window_size}s, slide={slide_size}s)")
    
    def register_processor(self, stream_name: str, processor: Callable) -> None:
        """Register stream processor"""
        self.processors[stream_name] = processor
        logger.info(f"Registered processor for stream '{stream_name}'")
    
    def add_event(self, stream_name: str, event: Event) -> None:
        """Add event to stream"""
        self.streams[stream_name].append(event)
        
        # Clean old events outside window
        cutoff_time = time.time() - self.window_size
        while self.streams[stream_name] and self.streams[stream_name][0].timestamp < cutoff_time:
            self.streams[stream_name].popleft()
        
        logger.debug(f"Event added to stream '{stream_name}'")
    
    def process_window(self, stream_name: str) -> Optional[Any]:
        """Process current window"""
        if stream_name not in self.processors:
            logger.warning(f"No processor registered for stream '{stream_name}'")
            return None
        
        events = list(self.streams[stream_name])
        if not events:
            return None
        
        # Apply processor to window
        processor = self.processors[stream_name]
        result = processor(events)
        
        self.results[stream_name].append({
            "timestamp": time.time(),
            "window_size": len(events),
            "result": result
        })
        
        logger.debug(f"Processed window for stream '{stream_name}': {len(events)} events")
        return result
    
    def tumbling_window(self, stream_name: str) -> List[List[Event]]:
        """Create tumbling windows (non-overlapping)"""
        events = sorted(self.streams[stream_name], key=lambda e: e.timestamp)
        windows = []
        
        if not events:
            return windows
        
        start_time = events[0].timestamp
        current_window = []
        
        for event in events:
            if event.timestamp - start_time >= self.window_size:
                if current_window:
                    windows.append(current_window)
                current_window = [event]
                start_time = event.timestamp
            else:
                current_window.append(event)
        
        if current_window:
            windows.append(current_window)
        
        return windows
    
    def sliding_window(self, stream_name: str) -> List[List[Event]]:
        """Create sliding windows (overlapping)"""
        events = sorted(self.streams[stream_name], key=lambda e: e.timestamp)
        windows = []
        
        if not events:
            return windows
        
        start_time = events[0].timestamp
        end_time = events[-1].timestamp
        
        current_start = start_time
        while current_start <= end_time:
            window = [e for e in events 
                     if current_start <= e.timestamp < current_start + self.window_size]
            if window:
                windows.append(window)
            current_start += self.slide_size
        
        return windows
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stream processing statistics"""
        return {
            "streams": len(self.streams),
            "processors": len(self.processors),
            "total_events": sum(len(s) for s in self.streams.values()),
            "processed_windows": sum(len(r) for r in self.results.values())
        }


# ============================================================================
# 5. MESSAGE BROKER WITH PUB/SUB
# ============================================================================

class MessageBroker:
    """Message broker with publish/subscribe pattern"""
    
    def __init__(self):
        self.topics: Dict[str, MessageQueue] = {}
        self.subscribers: Dict[str, Set[str]] = defaultdict(set)
        self.consumer_groups: Dict[str, Dict[str, Any]] = {}
        self.message_count = 0
        logger.info("MessageBroker initialized")
    
    def create_topic(self, topic: str, max_size: int = 10000) -> None:
        """Create message topic"""
        if topic not in self.topics:
            self.topics[topic] = MessageQueue(topic, max_size)
            logger.info(f"Topic '{topic}' created")
    
    def publish(self, topic: str, message: Message) -> bool:
        """Publish message to topic"""
        if topic not in self.topics:
            self.create_topic(topic)
        
        success = self.topics[topic].enqueue(message)
        if success:
            self.message_count += 1
        return success
    
    def subscribe(self, topic: str, consumer_id: str) -> None:
        """Subscribe consumer to topic"""
        if topic not in self.topics:
            self.create_topic(topic)
        
        self.subscribers[topic].add(consumer_id)
        logger.info(f"Consumer '{consumer_id}' subscribed to topic '{topic}'")
    
    def unsubscribe(self, topic: str, consumer_id: str) -> None:
        """Unsubscribe consumer from topic"""
        if topic in self.subscribers:
            self.subscribers[topic].discard(consumer_id)
            logger.info(f"Consumer '{consumer_id}' unsubscribed from topic '{topic}'")
    
    def consume(self, topic: str, consumer_id: str) -> Optional[Message]:
        """Consume message from topic"""
        if topic not in self.topics:
            return None
        
        if consumer_id not in self.subscribers[topic]:
            logger.warning(f"Consumer '{consumer_id}' not subscribed to '{topic}'")
            return None
        
        return self.topics[topic].dequeue()
    
    def create_consumer_group(self, group_id: str, topic: str) -> None:
        """Create consumer group for load balancing"""
        self.consumer_groups[group_id] = {
            "topic": topic,
            "consumers": set(),
            "offset": 0
        }
        logger.info(f"Consumer group '{group_id}' created for topic '{topic}'")
    
    def join_consumer_group(self, group_id: str, consumer_id: str) -> None:
        """Join consumer to consumer group"""
        if group_id in self.consumer_groups:
            self.consumer_groups[group_id]["consumers"].add(consumer_id)
            logger.info(f"Consumer '{consumer_id}' joined group '{group_id}'")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get broker statistics"""
        return {
            "topics": len(self.topics),
            "total_messages": self.message_count,
            "total_subscribers": sum(len(subs) for subs in self.subscribers.values()),
            "consumer_groups": len(self.consumer_groups),
            "topic_stats": {name: queue.get_stats() for name, queue in self.topics.items()}
        }


# ============================================================================
# 6. SAGA PATTERN FOR DISTRIBUTED TRANSACTIONS
# ============================================================================

class SagaStep:
    """Single step in saga"""
    
    def __init__(self, name: str, action: Callable, compensation: Callable):
        self.name = name
        self.action = action
        self.compensation = compensation
        self.executed = False
        self.result = None


class Saga:
    """Saga orchestrator for distributed transactions"""
    
    def __init__(self, saga_id: str, name: str):
        self.saga_id = saga_id
        self.name = name
        self.steps: List[SagaStep] = []
        self.executed_steps: List[SagaStep] = []
        self.status = "pending"  # pending, executing, completed, compensating, failed
        self.error: Optional[str] = None
        logger.info(f"Saga '{name}' created: {saga_id}")
    
    def add_step(self, name: str, action: Callable, compensation: Callable) -> None:
        """Add step to saga"""
        step = SagaStep(name, action, compensation)
        self.steps.append(step)
        logger.debug(f"Step '{name}' added to saga '{self.name}'")
    
    async def execute(self) -> bool:
        """Execute saga with automatic compensation on failure"""
        self.status = "executing"
        logger.info(f"Executing saga '{self.name}'...")
        
        try:
            # Execute all steps
            for step in self.steps:
                logger.debug(f"Executing step: {step.name}")
                
                if asyncio.iscoroutinefunction(step.action):
                    step.result = await step.action()
                else:
                    step.result = step.action()
                
                step.executed = True
                self.executed_steps.append(step)
                logger.debug(f"Step '{step.name}' completed")
            
            self.status = "completed"
            logger.info(f"Saga '{self.name}' completed successfully")
            return True
            
        except Exception as e:
            # Compensate executed steps in reverse order
            self.error = str(e)
            self.status = "compensating"
            logger.error(f"Saga '{self.name}' failed: {e}. Starting compensation...")
            
            await self._compensate()
            
            self.status = "failed"
            logger.info(f"Saga '{self.name}' compensation completed")
            return False
    
    async def _compensate(self) -> None:
        """Compensate executed steps in reverse order"""
        for step in reversed(self.executed_steps):
            try:
                logger.debug(f"Compensating step: {step.name}")
                
                if asyncio.iscoroutinefunction(step.compensation):
                    await step.compensation()
                else:
                    step.compensation()
                
                logger.debug(f"Step '{step.name}' compensated")
            except Exception as e:
                logger.error(f"Compensation failed for step '{step.name}': {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get saga status"""
        return {
            "saga_id": self.saga_id,
            "name": self.name,
            "status": self.status,
            "total_steps": len(self.steps),
            "executed_steps": len(self.executed_steps),
            "error": self.error
        }


class SagaOrchestrator:
    """Orchestrator for managing multiple sagas"""
    
    def __init__(self):
        self.sagas: Dict[str, Saga] = {}
        self.completed_count = 0
        self.failed_count = 0
        logger.info("SagaOrchestrator initialized")
    
    def create_saga(self, name: str) -> Saga:
        """Create new saga"""
        saga_id = f"saga-{uuid.uuid4().hex[:8]}"
        saga = Saga(saga_id, name)
        self.sagas[saga_id] = saga
        return saga
    
    async def execute_saga(self, saga: Saga) -> bool:
        """Execute saga and track results"""
        success = await saga.execute()
        
        if success:
            self.completed_count += 1
        else:
            self.failed_count += 1
        
        return success
    
    def get_saga(self, saga_id: str) -> Optional[Saga]:
        """Get saga by ID"""
        return self.sagas.get(saga_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "total_sagas": len(self.sagas),
            "completed": self.completed_count,
            "failed": self.failed_count,
            "success_rate": self.completed_count / max(1, self.completed_count + self.failed_count)
        }


# ============================================================================
# 7. EVENT REPLAY & TIME TRAVEL
# ============================================================================

class EventReplayer:
    """Event replay for debugging and recovery"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.replay_handlers: Dict[str, List[Callable]] = defaultdict(list)
        logger.info("EventReplayer initialized")
    
    def register_replay_handler(self, event_type: str, handler: Callable) -> None:
        """Register handler for event replay"""
        self.replay_handlers[event_type].append(handler)
        logger.info(f"Replay handler registered for event type '{event_type}'")
    
    async def replay_events(self, aggregate_id: str, from_version: int = 0, 
                           to_version: Optional[int] = None) -> int:
        """Replay events for aggregate"""
        events = self.event_store.get_events(aggregate_id, from_version)
        
        if to_version is not None:
            events = events[:to_version - from_version]
        
        logger.info(f"Replaying {len(events)} events for aggregate {aggregate_id}")
        
        replayed = 0
        for event in events:
            handlers = self.replay_handlers.get(event.type, [])
            
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                    replayed += 1
                except Exception as e:
                    logger.error(f"Error replaying event {event.id}: {e}")
        
        logger.info(f"Replayed {replayed} events")
        return replayed
    
    async def time_travel(self, aggregate_id: str, target_timestamp: float) -> Optional[Dict[str, Any]]:
        """Reconstruct aggregate state at specific point in time"""
        all_events = self.event_store.get_events(aggregate_id)
        
        # Filter events up to target time
        past_events = [e for e in all_events if e.timestamp <= target_timestamp]
        
        if not past_events:
            logger.warning(f"No events found before timestamp {target_timestamp}")
            return None
        
        # Rebuild state from filtered events
        state = self.event_store._rebuild_state(past_events)
        
        logger.info(f"Time traveled to {datetime.fromtimestamp(target_timestamp)} "
                   f"using {len(past_events)} events")
        
        return state
    
    def get_event_timeline(self, aggregate_id: str) -> List[Dict[str, Any]]:
        """Get timeline of events for aggregate"""
        events = self.event_store.get_events(aggregate_id)
        
        timeline = []
        for event in events:
            timeline.append({
                "timestamp": datetime.fromtimestamp(event.timestamp).isoformat(),
                "type": event.type,
                "id": event.id,
                "data": event.data
            })
        
        return timeline


# ============================================================================
# DEMONSTRATION
# ============================================================================

async def demo_message_queue_event_streaming():
    """Comprehensive demonstration of all components"""
    
    print("=" * 80)
    print("MESSAGE QUEUE & EVENT STREAMING INFRASTRUCTURE")
    print("=" * 80)
    print()
    
    print("🏗️  COMPONENTS:")
    print("   1. Priority Message Queue with DLQ")
    print("   2. Event Bus with Topic Routing")
    print("   3. Event Sourcing & CQRS")
    print("   4. Stream Processing Engine")
    print("   5. Message Broker with Pub/Sub")
    print("   6. Saga Pattern for Distributed Transactions")
    print("   7. Event Replay & Time Travel")
    print()
    
    # ========================================================================
    # 1. Message Queue with Priority & DLQ
    # ========================================================================
    print("=" * 80)
    print("1. PRIORITY MESSAGE QUEUE WITH DEAD LETTER QUEUE")
    print("=" * 80)
    
    queue = MessageQueue("nutrition-queue")
    
    # Create messages with different priorities
    messages = [
        Message(id="msg-1", topic="food.scan", payload={"food": "apple"}, 
                priority=MessagePriority.LOW),
        Message(id="msg-2", topic="food.scan", payload={"food": "chicken"}, 
                priority=MessagePriority.CRITICAL),
        Message(id="msg-3", topic="food.scan", payload={"food": "rice"}, 
                priority=MessagePriority.HIGH),
        Message(id="msg-4", topic="food.scan", payload={"food": "banana"}, 
                priority=MessagePriority.NORMAL, ttl=0.1),  # Will expire
    ]
    
    print(f"\n📨 Enqueuing {len(messages)} messages...")
    for msg in messages:
        queue.enqueue(msg)
    
    # Wait for TTL message to expire
    await asyncio.sleep(0.2)
    
    print(f"\n🔄 Processing messages (priority order)...")
    processed = []
    while True:
        msg = queue.dequeue()
        if not msg:
            break
        
        print(f"   • {msg.id}: {msg.payload['food']} (priority: {msg.priority.name})")
        
        # Simulate processing failure for one message
        if msg.id == "msg-3":
            queue.nack(msg, "processing_error")
        else:
            queue.ack(msg)
            processed.append(msg)
    
    stats = queue.get_stats()
    print(f"\n📊 Queue Statistics:")
    print(f"   Processed: {stats['processed']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   DLQ Size: {stats['dlq_size']}")
    print(f"   Success Rate: {stats['success_rate']:.1%}")
    
    # ========================================================================
    # 2. Event Bus with Topic-Based Routing
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. EVENT BUS WITH TOPIC-BASED ROUTING")
    print("=" * 80)
    
    event_bus = EventBus()
    
    # Create event handlers
    user_events = []
    food_events = []
    all_events = []
    
    async def user_handler(event: Event):
        user_events.append(event)
    
    async def food_handler(event: Event):
        food_events.append(event)
    
    async def all_handler(event: Event):
        all_events.append(event)
    
    # Subscribe to topics with wildcards
    event_bus.subscribe("user.created", user_handler)
    event_bus.subscribe("user.updated", user_handler)
    event_bus.subscribe("food.*", food_handler)  # Wildcard
    event_bus.subscribe("#", all_handler)  # Catch all
    
    print(f"\n📡 Publishing events...")
    events_to_publish = [
        Event(id="evt-1", type="user.created", source="auth-service", 
              data={"user_id": "user-1", "email": "john@example.com"}),
        Event(id="evt-2", type="food.scanned", source="scan-service", 
              data={"food": "apple", "calories": 95}),
        Event(id="evt-3", type="user.updated", source="profile-service", 
              data={"user_id": "user-1", "name": "John Doe"}),
        Event(id="evt-4", type="food.analyzed", source="ai-service", 
              data={"food": "chicken", "protein": 31}),
    ]
    
    for event in events_to_publish:
        subscribers = await event_bus.publish(event.type, event)
        print(f"   • {event.type}: {subscribers} subscribers notified")
    
    stats = event_bus.get_stats()
    print(f"\n📊 Event Bus Statistics:")
    print(f"   Total Events: {stats['total_events']}")
    print(f"   Topics: {stats['topics']}")
    print(f"   Subscribers: {stats['subscribers']}")
    print(f"   User Events Received: {len(user_events)}")
    print(f"   Food Events Received: {len(food_events)}")
    print(f"   All Events Received: {len(all_events)}")
    
    # ========================================================================
    # 3. Event Sourcing & CQRS
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. EVENT SOURCING & CQRS PATTERN")
    print("=" * 80)
    
    event_store = EventStore(snapshot_interval=3)
    
    # Create events for user aggregate
    user_id = "user-123"
    user_events = [
        Event(id="e1", type="UserCreated", source="auth", 
              data={"user_id": user_id, "email": "jane@example.com"}),
        Event(id="e2", type="ProfileUpdated", source="profile", 
              data={"name": "Jane Doe", "age": 28}),
        Event(id="e3", type="PreferencesSet", source="settings", 
              data={"diet": "vegetarian", "allergies": ["nuts"]}),
        Event(id="e4", type="FoodScanned", source="scan", 
              data={"food": "tofu", "timestamp": time.time()}),
        Event(id="e5", type="MealLogged", source="diary", 
              data={"meal": "breakfast", "calories": 350}),
    ]
    
    print(f"\n📝 Appending {len(user_events)} events...")
    for event in user_events:
        event_store.append_event(user_id, event)
        print(f"   • {event.type} appended")
    
    print(f"\n🔄 Loading aggregate from event store...")
    state = event_store.load_aggregate(user_id)
    print(f"   Aggregate State: {json.dumps(state, indent=2)}")
    
    stats = event_store.get_stats()
    print(f"\n📊 Event Store Statistics:")
    print(f"   Total Events: {stats['total_events']}")
    print(f"   Aggregates: {stats['aggregates']}")
    print(f"   Snapshots: {stats['snapshots']}")
    print(f"   Avg Events/Aggregate: {stats['avg_events_per_aggregate']:.1f}")
    
    # ========================================================================
    # 4. Stream Processing Engine
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. STREAM PROCESSING ENGINE")
    print("=" * 80)
    
    processor = StreamProcessor(window_size=10, slide_size=5)
    
    # Register processor for calorie stream
    def calorie_aggregator(events: List[Event]) -> Dict[str, Any]:
        total_calories = sum(e.data.get("calories", 0) for e in events)
        avg_calories = total_calories / len(events) if events else 0
        return {
            "total": total_calories,
            "average": avg_calories,
            "count": len(events)
        }
    
    processor.register_processor("calories", calorie_aggregator)
    
    # Add events to stream
    print(f"\n📊 Adding events to calorie stream...")
    base_time = time.time()
    calorie_events = [
        Event(id=f"cal-{i}", type="CalorieIntake", source="diary",
              data={"calories": 100 + i * 50, "meal": f"meal-{i}"},
              timestamp=base_time + i)
        for i in range(8)
    ]
    
    for event in calorie_events:
        processor.add_event("calories", event)
    
    print(f"   Added {len(calorie_events)} events")
    
    # Process window
    print(f"\n🔄 Processing current window...")
    result = processor.process_window("calories")
    print(f"   Total Calories: {result['total']}")
    print(f"   Average Calories: {result['average']:.1f}")
    print(f"   Event Count: {result['count']}")
    
    # Create windowing views
    print(f"\n🪟 Creating tumbling windows...")
    tumbling = processor.tumbling_window("calories")
    print(f"   Created {len(tumbling)} non-overlapping windows")
    
    print(f"\n🪟 Creating sliding windows...")
    sliding = processor.sliding_window("calories")
    print(f"   Created {len(sliding)} overlapping windows")
    
    stats = processor.get_stats()
    print(f"\n📊 Stream Processing Statistics:")
    print(f"   Active Streams: {stats['streams']}")
    print(f"   Registered Processors: {stats['processors']}")
    print(f"   Total Events: {stats['total_events']}")
    print(f"   Processed Windows: {stats['processed_windows']}")
    
    # ========================================================================
    # 5. Message Broker with Pub/Sub
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. MESSAGE BROKER WITH PUB/SUB")
    print("=" * 80)
    
    broker = MessageBroker()
    
    # Create topics
    print(f"\n📢 Creating topics...")
    broker.create_topic("food-analysis")
    broker.create_topic("user-activity")
    print(f"   Created 2 topics")
    
    # Subscribe consumers
    print(f"\n👥 Subscribing consumers...")
    broker.subscribe("food-analysis", "consumer-1")
    broker.subscribe("food-analysis", "consumer-2")
    broker.subscribe("user-activity", "consumer-3")
    print(f"   3 consumers subscribed")
    
    # Publish messages
    print(f"\n📨 Publishing messages...")
    food_messages = [
        Message(id=f"food-{i}", topic="food-analysis", 
                payload={"food": f"item-{i}", "calories": 100 + i * 20})
        for i in range(5)
    ]
    
    for msg in food_messages:
        broker.publish("food-analysis", msg)
    
    print(f"   Published {len(food_messages)} messages to 'food-analysis'")
    
    # Consume messages
    print(f"\n🔄 Consuming messages...")
    consumed = 0
    while True:
        msg = broker.consume("food-analysis", "consumer-1")
        if not msg:
            break
        consumed += 1
    
    print(f"   Consumer-1 consumed {consumed} messages")
    
    # Consumer groups
    print(f"\n👥 Creating consumer group...")
    broker.create_consumer_group("analysis-group", "food-analysis")
    broker.join_consumer_group("analysis-group", "consumer-1")
    broker.join_consumer_group("analysis-group", "consumer-2")
    print(f"   Consumer group 'analysis-group' created with 2 consumers")
    
    stats = broker.get_stats()
    print(f"\n📊 Message Broker Statistics:")
    print(f"   Topics: {stats['topics']}")
    print(f"   Total Messages: {stats['total_messages']}")
    print(f"   Total Subscribers: {stats['total_subscribers']}")
    print(f"   Consumer Groups: {stats['consumer_groups']}")
    
    # ========================================================================
    # 6. Saga Pattern for Distributed Transactions
    # ========================================================================
    print("\n" + "=" * 80)
    print("6. SAGA PATTERN FOR DISTRIBUTED TRANSACTIONS")
    print("=" * 80)
    
    orchestrator = SagaOrchestrator()
    
    # Create order processing saga
    print(f"\n🔄 Creating 'Food Order' saga...")
    saga = orchestrator.create_saga("Food Order Processing")
    
    # Saga state
    saga_state = {"order_id": None, "payment_id": None, "delivery_id": None}
    
    # Define saga steps
    def create_order():
        saga_state["order_id"] = f"order-{uuid.uuid4().hex[:8]}"
        print(f"      ✓ Order created: {saga_state['order_id']}")
        return saga_state["order_id"]
    
    def cancel_order():
        print(f"      ↩ Order cancelled: {saga_state['order_id']}")
        saga_state["order_id"] = None
    
    def process_payment():
        saga_state["payment_id"] = f"pay-{uuid.uuid4().hex[:8]}"
        print(f"      ✓ Payment processed: {saga_state['payment_id']}")
        return saga_state["payment_id"]
    
    def refund_payment():
        print(f"      ↩ Payment refunded: {saga_state['payment_id']}")
        saga_state["payment_id"] = None
    
    def schedule_delivery():
        saga_state["delivery_id"] = f"del-{uuid.uuid4().hex[:8]}"
        print(f"      ✓ Delivery scheduled: {saga_state['delivery_id']}")
        return saga_state["delivery_id"]
    
    def cancel_delivery():
        print(f"      ↩ Delivery cancelled: {saga_state['delivery_id']}")
        saga_state["delivery_id"] = None
    
    # Add steps to saga
    saga.add_step("CreateOrder", create_order, cancel_order)
    saga.add_step("ProcessPayment", process_payment, refund_payment)
    saga.add_step("ScheduleDelivery", schedule_delivery, cancel_delivery)
    
    print(f"\n▶️  Executing saga (SUCCESS scenario)...")
    success = await orchestrator.execute_saga(saga)
    print(f"   Result: {'SUCCESS' if success else 'FAILED'}")
    
    # Create failing saga
    print(f"\n🔄 Creating saga with failure...")
    failing_saga = orchestrator.create_saga("Failing Order")
    
    def failing_step():
        raise Exception("Payment gateway timeout")
    
    failing_saga.add_step("CreateOrder", create_order, cancel_order)
    failing_saga.add_step("ProcessPayment", failing_step, refund_payment)
    failing_saga.add_step("ScheduleDelivery", schedule_delivery, cancel_delivery)
    
    print(f"\n▶️  Executing saga (FAILURE scenario)...")
    success = await orchestrator.execute_saga(failing_saga)
    print(f"   Result: {'SUCCESS' if success else 'FAILED (compensated)'}")
    
    stats = orchestrator.get_stats()
    print(f"\n📊 Saga Orchestrator Statistics:")
    print(f"   Total Sagas: {stats['total_sagas']}")
    print(f"   Completed: {stats['completed']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   Success Rate: {stats['success_rate']:.1%}")
    
    # ========================================================================
    # 7. Event Replay & Time Travel
    # ========================================================================
    print("\n" + "=" * 80)
    print("7. EVENT REPLAY & TIME TRAVEL")
    print("=" * 80)
    
    replayer = EventReplayer(event_store)
    
    # Register replay handlers
    replayed_events = []
    
    def replay_handler(event: Event):
        replayed_events.append(event)
        print(f"      ↻ Replayed: {event.type}")
    
    replayer.register_replay_handler("UserCreated", replay_handler)
    replayer.register_replay_handler("ProfileUpdated", replay_handler)
    replayer.register_replay_handler("MealLogged", replay_handler)
    
    print(f"\n🔄 Replaying events for user {user_id}...")
    replayed_count = await replayer.replay_events(user_id)
    print(f"   Replayed {replayed_count} events")
    
    # Time travel
    print(f"\n⏰ Time traveling to past state...")
    target_time = user_events[2].timestamp  # After 3rd event
    past_state = await replayer.time_travel(user_id, target_time)
    print(f"   State at {datetime.fromtimestamp(target_time).strftime('%H:%M:%S')}:")
    print(f"   {json.dumps(past_state, indent=2)}")
    
    # Event timeline
    print(f"\n📅 Event Timeline:")
    timeline = replayer.get_event_timeline(user_id)
    for i, entry in enumerate(timeline, 1):
        print(f"   {i}. [{entry['timestamp']}] {entry['type']}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ MESSAGE QUEUE & EVENT STREAMING COMPLETE")
    print("=" * 80)
    
    print("\n📦 CAPABILITIES:")
    print("   ✓ Priority message queue with dead letter queue")
    print("   ✓ Event bus with wildcard topic routing")
    print("   ✓ Event sourcing with snapshots")
    print("   ✓ Stream processing with windowing")
    print("   ✓ Message broker with pub/sub pattern")
    print("   ✓ Saga pattern for distributed transactions")
    print("   ✓ Event replay and time travel debugging")
    
    print("\n🎯 INFRASTRUCTURE METRICS:")
    print(f"   Messages processed: {queue.processed_count} ✓")
    print(f"   Events published: {event_bus.event_count} ✓")
    print(f"   Event store entries: {event_store.event_count} ✓")
    print(f"   Stream windows processed: {len(processor.results)} ✓")
    print(f"   Broker messages: {broker.message_count} ✓")
    print(f"   Sagas executed: {orchestrator.completed_count + orchestrator.failed_count} ✓")
    print(f"   Events replayed: {replayed_count} ✓")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(demo_message_queue_event_streaming())
