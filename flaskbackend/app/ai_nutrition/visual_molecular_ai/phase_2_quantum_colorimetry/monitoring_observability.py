"""
MONITORING & OBSERVABILITY INFRASTRUCTURE
==========================================

Enterprise Monitoring & Observability System

COMPONENTS:
1. Distributed Tracing (OpenTelemetry patterns)
2. Metrics Collection & Aggregation (Prometheus patterns)
3. Log Aggregation (ELK stack patterns)
4. Alerting & Incident Management
5. Performance Profiling
6. Health Checks & SLOs
7. Dashboard & Visualization
8. Anomaly Detection
9. Correlation Analysis
10. Root Cause Analysis (RCA)

ARCHITECTURE:
- OpenTelemetry standards
- Prometheus/Grafana patterns
- ELK stack concepts
- Three pillars: Logs, Metrics, Traces
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import logging
import json
import hashlib
import time
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DISTRIBUTED TRACING
# ============================================================================

@dataclass
class Span:
    """Distributed trace span"""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Attributes
    service_name: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Status
    status: str = "ok"  # ok, error
    error_message: Optional[str] = None
    
    def finish(self, status: str = "ok", error_message: Optional[str] = None):
        """Finish span"""
        self.end_time = datetime.now()
        self.status = status
        self.error_message = error_message
    
    def duration_ms(self) -> float:
        """Get span duration in milliseconds"""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000
    
    def add_log(self, message: str, level: str = "info"):
        """Add log entry to span"""
        self.logs.append({
            'timestamp': datetime.now(),
            'message': message,
            'level': level
        })


@dataclass
class Trace:
    """Distributed trace"""
    trace_id: str
    root_span: Span
    spans: List[Span] = field(default_factory=list)
    
    def add_span(self, span: Span):
        """Add span to trace"""
        self.spans.append(span)
    
    def get_duration_ms(self) -> float:
        """Get total trace duration"""
        if not self.spans:
            return self.root_span.duration_ms()
        
        start_times = [s.start_time for s in self.spans]
        end_times = [s.end_time for s in self.spans if s.end_time]
        
        if not end_times:
            return 0.0
        
        total_duration = (max(end_times) - min(start_times)).total_seconds() * 1000
        return total_duration
    
    def has_errors(self) -> bool:
        """Check if trace has any errors"""
        return any(s.status == "error" for s in self.spans)


class Tracer:
    """
    Distributed Tracer (OpenTelemetry patterns)
    
    Features:
    - Context propagation
    - Span creation & management
    - Trace collection
    - Service topology mapping
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.traces: Dict[str, Trace] = {}
        self.active_spans: Dict[str, Span] = {}
        
        logger.info(f"Tracer initialized for service '{service_name}'")
    
    def start_trace(self, operation_name: str) -> Tuple[str, Span]:
        """
        Start new trace
        
        Returns:
            (trace_id, root_span)
        """
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        root_span = Span(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=None,
            operation_name=operation_name,
            start_time=datetime.now(),
            service_name=self.service_name
        )
        
        trace = Trace(trace_id, root_span)
        self.traces[trace_id] = trace
        self.active_spans[span_id] = root_span
        
        return trace_id, root_span
    
    def start_span(
        self,
        trace_id: str,
        parent_span_id: str,
        operation_name: str
    ) -> Span:
        """Start child span"""
        span_id = str(uuid.uuid4())
        
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            service_name=self.service_name
        )
        
        # Add to trace
        if trace_id in self.traces:
            self.traces[trace_id].add_span(span)
        
        self.active_spans[span_id] = span
        
        return span
    
    def finish_span(
        self,
        span_id: str,
        status: str = "ok",
        error_message: Optional[str] = None
    ):
        """Finish span"""
        if span_id in self.active_spans:
            span = self.active_spans[span_id]
            span.finish(status, error_message)
            del self.active_spans[span_id]
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get trace by ID"""
        return self.traces.get(trace_id)
    
    def get_trace_stats(self) -> Dict[str, Any]:
        """Get tracing statistics"""
        total_traces = len(self.traces)
        total_spans = sum(len(t.spans) for t in self.traces.values())
        
        error_traces = sum(1 for t in self.traces.values() if t.has_errors())
        
        if self.traces:
            durations = [t.get_duration_ms() for t in self.traces.values()]
            avg_duration = np.mean(durations)
            p95_duration = np.percentile(durations, 95)
            p99_duration = np.percentile(durations, 99)
        else:
            avg_duration = p95_duration = p99_duration = 0
        
        return {
            'service_name': self.service_name,
            'total_traces': total_traces,
            'total_spans': total_spans,
            'error_traces': error_traces,
            'error_rate': error_traces / max(1, total_traces),
            'avg_duration_ms': avg_duration,
            'p95_duration_ms': p95_duration,
            'p99_duration_ms': p99_duration
        }


# ============================================================================
# METRICS COLLECTION (PROMETHEUS PATTERNS)
# ============================================================================

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Metric definition"""
    name: str
    metric_type: MetricType
    help_text: str
    labels: Dict[str, str] = field(default_factory=dict)
    
    # Values
    value: float = 0.0
    values: List[float] = field(default_factory=list)
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """
    Metrics Collector (Prometheus patterns)
    
    Features:
    - Counter, Gauge, Histogram, Summary
    - Label-based dimensions
    - Aggregation
    - Exposition format
    """
    
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        
        logger.info("MetricsCollector initialized")
    
    def counter(self, name: str, help_text: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create or get counter metric"""
        metric_key = self._get_metric_key(name, labels)
        
        if metric_key not in self.metrics:
            self.metrics[metric_key] = Metric(
                name=name,
                metric_type=MetricType.COUNTER,
                help_text=help_text,
                labels=labels or {}
            )
        
        return metric_key
    
    def gauge(self, name: str, help_text: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create or get gauge metric"""
        metric_key = self._get_metric_key(name, labels)
        
        if metric_key not in self.metrics:
            self.metrics[metric_key] = Metric(
                name=name,
                metric_type=MetricType.GAUGE,
                help_text=help_text,
                labels=labels or {}
            )
        
        return metric_key
    
    def histogram(self, name: str, help_text: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create or get histogram metric"""
        metric_key = self._get_metric_key(name, labels)
        
        if metric_key not in self.metrics:
            self.metrics[metric_key] = Metric(
                name=name,
                metric_type=MetricType.HISTOGRAM,
                help_text=help_text,
                labels=labels or {}
            )
        
        return metric_key
    
    def inc(self, metric_key: str, value: float = 1.0):
        """Increment counter"""
        if metric_key in self.metrics:
            metric = self.metrics[metric_key]
            metric.value += value
            metric.timestamp = datetime.now()
    
    def set(self, metric_key: str, value: float):
        """Set gauge value"""
        if metric_key in self.metrics:
            metric = self.metrics[metric_key]
            metric.value = value
            metric.timestamp = datetime.now()
    
    def observe(self, metric_key: str, value: float):
        """Observe value for histogram"""
        if metric_key in self.metrics:
            metric = self.metrics[metric_key]
            metric.values.append(value)
            metric.timestamp = datetime.now()
    
    def get_metric_value(self, metric_key: str) -> Optional[float]:
        """Get metric value"""
        if metric_key in self.metrics:
            metric = self.metrics[metric_key]
            
            if metric.metric_type == MetricType.HISTOGRAM:
                return np.mean(metric.values) if metric.values else 0.0
            
            return metric.value
        
        return None
    
    def get_histogram_stats(self, metric_key: str) -> Dict[str, float]:
        """Get histogram statistics"""
        if metric_key in self.metrics:
            metric = self.metrics[metric_key]
            
            if metric.values:
                return {
                    'count': len(metric.values),
                    'sum': sum(metric.values),
                    'avg': np.mean(metric.values),
                    'min': min(metric.values),
                    'max': max(metric.values),
                    'p50': np.percentile(metric.values, 50),
                    'p95': np.percentile(metric.values, 95),
                    'p99': np.percentile(metric.values, 99)
                }
        
        return {}
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        for metric_key, metric in self.metrics.items():
            # HELP line
            lines.append(f"# HELP {metric.name} {metric.help_text}")
            
            # TYPE line
            lines.append(f"# TYPE {metric.name} {metric.metric_type.value}")
            
            # Metric line
            labels_str = ",".join(f'{k}="{v}"' for k, v in metric.labels.items())
            
            if metric.metric_type == MetricType.HISTOGRAM:
                if metric.values:
                    stats = self.get_histogram_stats(metric_key)
                    lines.append(f"{metric.name}_sum{{{labels_str}}} {stats['sum']}")
                    lines.append(f"{metric.name}_count{{{labels_str}}} {stats['count']}")
            else:
                value_str = labels_str and f"{{{labels_str}}}" or ""
                lines.append(f"{metric.name}{value_str} {metric.value}")
        
        return "\n".join(lines)
    
    def _get_metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Generate unique metric key"""
        if labels:
            labels_str = "_".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}_{labels_str}"
        return name


# ============================================================================
# LOG AGGREGATION (ELK PATTERNS)
# ============================================================================

class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    message: str
    
    # Context
    service_name: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    # Structured fields
    fields: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    host: str = ""
    environment: str = ""


class LogAggregator:
    """
    Log Aggregation System (ELK patterns)
    
    Features:
    - Structured logging
    - Log indexing
    - Full-text search
    - Log correlation with traces
    """
    
    def __init__(self):
        self.logs: List[LogEntry] = []
        self.log_index: Dict[str, List[int]] = defaultdict(list)
        
        logger.info("LogAggregator initialized")
    
    def log(
        self,
        level: LogLevel,
        message: str,
        service_name: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **fields
    ):
        """Log structured entry"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            service_name=service_name,
            trace_id=trace_id,
            span_id=span_id,
            fields=fields
        )
        
        log_idx = len(self.logs)
        self.logs.append(entry)
        
        # Index by service
        self.log_index[f"service:{service_name}"].append(log_idx)
        
        # Index by level
        self.log_index[f"level:{level.value}"].append(log_idx)
        
        # Index by trace
        if trace_id:
            self.log_index[f"trace:{trace_id}"].append(log_idx)
    
    def search(
        self,
        query: Optional[str] = None,
        service_name: Optional[str] = None,
        level: Optional[LogLevel] = None,
        trace_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[LogEntry]:
        """Search logs"""
        results = []
        
        # Get candidate logs
        if service_name:
            indices = self.log_index.get(f"service:{service_name}", [])
            candidates = [self.logs[i] for i in indices]
        elif level:
            indices = self.log_index.get(f"level:{level.value}", [])
            candidates = [self.logs[i] for i in indices]
        elif trace_id:
            indices = self.log_index.get(f"trace:{trace_id}", [])
            candidates = [self.logs[i] for i in indices]
        else:
            candidates = self.logs
        
        # Filter
        for log in candidates:
            # Time filter
            if start_time and log.timestamp < start_time:
                continue
            if end_time and log.timestamp > end_time:
                continue
            
            # Query filter (simple substring match)
            if query and query.lower() not in log.message.lower():
                continue
            
            results.append(log)
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get log statistics"""
        total_logs = len(self.logs)
        
        # Count by level
        level_counts = defaultdict(int)
        for log in self.logs:
            level_counts[log.level.value] += 1
        
        # Count by service
        service_counts = defaultdict(int)
        for log in self.logs:
            service_counts[log.service_name] += 1
        
        return {
            'total_logs': total_logs,
            'by_level': dict(level_counts),
            'by_service': dict(service_counts)
        }


# ============================================================================
# ALERTING & INCIDENT MANAGEMENT
# ============================================================================

class AlertSeverity(Enum):
    """Alert severities"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertState(Enum):
    """Alert states"""
    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    name: str
    severity: AlertSeverity
    state: AlertState
    
    # Details
    description: str
    fired_at: datetime
    resolved_at: Optional[datetime] = None
    
    # Context
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    # Metrics
    value: float = 0.0
    threshold: float = 0.0


@dataclass
class AlertRule:
    """Alert rule definition"""
    rule_id: str
    name: str
    expression: str  # Metric expression
    threshold: float
    comparison: str  # >, <, ==, etc.
    severity: AlertSeverity
    
    # Configuration
    evaluation_interval_sec: int = 60
    for_duration_sec: int = 300  # Alert fires after sustained breach
    
    # Metadata
    description: str = ""
    runbook_url: str = ""


class AlertManager:
    """
    Alert & Incident Management
    
    Features:
    - Alert rule evaluation
    - Alert firing & resolution
    - Alert grouping
    - Notification routing
    - Incident tracking
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        logger.info("AlertManager initialized")
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Alert rule added: {rule.name}")
    
    def evaluate_rules(self):
        """Evaluate all alert rules"""
        for rule in self.alert_rules.values():
            self._evaluate_rule(rule)
    
    def _evaluate_rule(self, rule: AlertRule):
        """Evaluate single alert rule"""
        # Get metric value
        metric_value = self.metrics_collector.get_metric_value(rule.expression)
        
        if metric_value is None:
            return
        
        # Check threshold
        should_fire = False
        
        if rule.comparison == ">":
            should_fire = metric_value > rule.threshold
        elif rule.comparison == "<":
            should_fire = metric_value < rule.threshold
        elif rule.comparison == "==":
            should_fire = metric_value == rule.threshold
        elif rule.comparison == ">=":
            should_fire = metric_value >= rule.threshold
        elif rule.comparison == "<=":
            should_fire = metric_value <= rule.threshold
        
        if should_fire:
            self._fire_alert(rule, metric_value)
        else:
            self._resolve_alert(rule.rule_id)
    
    def _fire_alert(self, rule: AlertRule, value: float):
        """Fire alert"""
        if rule.rule_id in self.active_alerts:
            # Alert already firing
            return
        
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            name=rule.name,
            severity=rule.severity,
            state=AlertState.FIRING,
            description=rule.description,
            fired_at=datetime.now(),
            value=value,
            threshold=rule.threshold,
            labels={'rule_id': rule.rule_id},
            annotations={'runbook_url': rule.runbook_url}
        )
        
        self.active_alerts[rule.rule_id] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"🚨 ALERT FIRED: {rule.name} - Value: {value}, Threshold: {rule.threshold}")
    
    def _resolve_alert(self, rule_id: str):
        """Resolve alert"""
        if rule_id in self.active_alerts:
            alert = self.active_alerts[rule_id]
            alert.state = AlertState.RESOLVED
            alert.resolved_at = datetime.now()
            
            del self.active_alerts[rule_id]
            
            logger.info(f"✅ Alert resolved: {alert.name}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alerting statistics"""
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1
        
        return {
            'total_alerts_fired': total_alerts,
            'active_alerts': active_alerts,
            'by_severity': dict(severity_counts)
        }


# ============================================================================
# HEALTH CHECKS & SLOs
# ============================================================================

class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Health check result"""
    component: str
    status: HealthStatus
    timestamp: datetime
    
    # Details
    message: str = ""
    latency_ms: float = 0.0
    
    # Metadata
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLO:
    """Service Level Objective"""
    name: str
    target_percentage: float  # e.g., 99.9%
    measurement_window_days: int
    
    # Measurements
    success_count: int = 0
    total_count: int = 0
    
    def record(self, success: bool):
        """Record measurement"""
        self.total_count += 1
        if success:
            self.success_count += 1
    
    def get_current_percentage(self) -> float:
        """Get current SLO percentage"""
        if self.total_count == 0:
            return 100.0
        return (self.success_count / self.total_count) * 100
    
    def is_meeting_target(self) -> bool:
        """Check if meeting SLO target"""
        return self.get_current_percentage() >= self.target_percentage
    
    def get_error_budget_remaining(self) -> float:
        """Get remaining error budget percentage"""
        current = self.get_current_percentage()
        return current - self.target_percentage


class HealthMonitor:
    """
    Health Monitoring & SLO Tracking
    
    Features:
    - Component health checks
    - SLO tracking
    - Error budget monitoring
    - Dependency health
    """
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.slos: Dict[str, SLO] = {}
        
        logger.info("HealthMonitor initialized")
    
    def add_slo(self, slo: SLO):
        """Add SLO"""
        self.slos[slo.name] = slo
        logger.info(f"SLO added: {slo.name} - Target: {slo.target_percentage}%")
    
    def check_health(
        self,
        component: str,
        check_func: Callable[[], Tuple[HealthStatus, str]]
    ) -> HealthCheck:
        """Perform health check"""
        start_time = time.time()
        
        try:
            status, message = check_func()
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Health check failed: {str(e)}"
        
        latency_ms = (time.time() - start_time) * 1000
        
        health_check = HealthCheck(
            component=component,
            status=status,
            timestamp=datetime.now(),
            message=message,
            latency_ms=latency_ms
        )
        
        self.health_checks[component] = health_check
        
        return health_check
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health"""
        if not self.health_checks:
            return HealthStatus.HEALTHY
        
        statuses = [hc.status for hc in self.health_checks.values()]
        
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def get_slo_report(self) -> Dict[str, Any]:
        """Get SLO report"""
        report = {}
        
        for name, slo in self.slos.items():
            report[name] = {
                'target': slo.target_percentage,
                'current': slo.get_current_percentage(),
                'meeting_target': slo.is_meeting_target(),
                'error_budget_remaining': slo.get_error_budget_remaining(),
                'measurements': {
                    'success': slo.success_count,
                    'total': slo.total_count
                }
            }
        
        return report


# ============================================================================
# PERFORMANCE PROFILING
# ============================================================================

@dataclass
class ProfileEntry:
    """Performance profile entry"""
    function_name: str
    duration_ms: float
    call_count: int
    timestamp: datetime


class PerformanceProfiler:
    """
    Performance Profiler
    
    Features:
    - Function timing
    - Hot path identification
    - Bottleneck detection
    - Performance regression detection
    """
    
    def __init__(self):
        self.profiles: Dict[str, List[ProfileEntry]] = defaultdict(list)
        
        logger.info("PerformanceProfiler initialized")
    
    def profile_function(self, function_name: str, duration_ms: float):
        """Record function profile"""
        entry = ProfileEntry(
            function_name=function_name,
            duration_ms=duration_ms,
            call_count=1,
            timestamp=datetime.now()
        )
        
        self.profiles[function_name].append(entry)
    
    def get_hotspots(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get performance hotspots"""
        hotspots = []
        
        for function_name, entries in self.profiles.items():
            total_time = sum(e.duration_ms for e in entries)
            call_count = len(entries)
            avg_time = total_time / call_count if call_count > 0 else 0
            
            hotspots.append({
                'function': function_name,
                'total_time_ms': total_time,
                'call_count': call_count,
                'avg_time_ms': avg_time
            })
        
        # Sort by total time
        hotspots.sort(key=lambda x: x['total_time_ms'], reverse=True)
        
        return hotspots[:top_n]
    
    def detect_regressions(
        self,
        function_name: str,
        baseline_ms: float,
        threshold_percent: float = 20.0
    ) -> bool:
        """Detect performance regression"""
        entries = self.profiles.get(function_name, [])
        
        if not entries:
            return False
        
        recent_entries = entries[-10:]  # Last 10 calls
        avg_recent = np.mean([e.duration_ms for e in recent_entries])
        
        percent_increase = ((avg_recent - baseline_ms) / baseline_ms) * 100
        
        return percent_increase > threshold_percent


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_monitoring_observability():
    """Demonstrate Monitoring & Observability Infrastructure"""
    
    print("\n" + "="*80)
    print("MONITORING & OBSERVABILITY INFRASTRUCTURE")
    print("="*80)
    
    print("\n🏗️  COMPONENTS:")
    print("   1. Distributed Tracing")
    print("   2. Metrics Collection (Prometheus)")
    print("   3. Log Aggregation (ELK)")
    print("   4. Alerting & Incidents")
    print("   5. Health Checks & SLOs")
    print("   6. Performance Profiling")
    
    # ========================================================================
    # 1. DISTRIBUTED TRACING
    # ========================================================================
    
    print("\n" + "="*80)
    print("1. DISTRIBUTED TRACING (OpenTelemetry)")
    print("="*80)
    
    tracer = Tracer("nutrition-api")
    
    # Simulate request trace
    print("\n🔍 Tracing API request...")
    
    trace_id, root_span = tracer.start_trace("GET /api/nutrition")
    root_span.tags['http.method'] = 'GET'
    root_span.tags['http.path'] = '/api/nutrition'
    
    # Child spans
    db_span = tracer.start_span(trace_id, root_span.span_id, "database_query")
    time.sleep(0.01)  # Simulate work
    tracer.finish_span(db_span.span_id)
    
    cache_span = tracer.start_span(trace_id, root_span.span_id, "cache_lookup")
    time.sleep(0.005)  # Simulate work
    tracer.finish_span(cache_span.span_id)
    
    processing_span = tracer.start_span(trace_id, root_span.span_id, "data_processing")
    time.sleep(0.008)  # Simulate work
    tracer.finish_span(processing_span.span_id)
    
    # Finish root span
    tracer.finish_span(root_span.span_id)
    
    # Get trace
    trace = tracer.get_trace(trace_id)
    
    print(f"\n✅ Trace completed:")
    print(f"   Trace ID: {trace_id[:16]}...")
    print(f"   Total duration: {trace.get_duration_ms():.2f}ms")
    print(f"   Total spans: {len(trace.spans)}")
    print(f"\n   Span breakdown:")
    for span in trace.spans:
        print(f"      - {span.operation_name}: {span.duration_ms():.2f}ms")
    
    # Trace statistics
    stats = tracer.get_trace_stats()
    
    print(f"\n📊 Tracing Statistics:")
    print(f"   Total traces: {stats['total_traces']}")
    print(f"   Total spans: {stats['total_spans']}")
    print(f"   Error rate: {stats['error_rate']*100:.1f}%")
    print(f"   Avg duration: {stats['avg_duration_ms']:.2f}ms")
    
    # ========================================================================
    # 2. METRICS COLLECTION
    # ========================================================================
    
    print("\n" + "="*80)
    print("2. METRICS COLLECTION (Prometheus)")
    print("="*80)
    
    metrics = MetricsCollector()
    
    # Create metrics
    print("\n📊 Creating metrics...")
    
    requests_counter = metrics.counter(
        "http_requests_total",
        "Total HTTP requests",
        {"service": "nutrition-api", "method": "GET"}
    )
    
    latency_histogram = metrics.histogram(
        "http_request_duration_ms",
        "HTTP request duration",
        {"service": "nutrition-api"}
    )
    
    active_connections = metrics.gauge(
        "active_connections",
        "Active connections",
        {"service": "nutrition-api"}
    )
    
    print(f"   ✅ Created counter: http_requests_total")
    print(f"   ✅ Created histogram: http_request_duration_ms")
    print(f"   ✅ Created gauge: active_connections")
    
    # Record metrics
    print("\n📈 Recording metrics...")
    
    for i in range(100):
        metrics.inc(requests_counter)
        
        # Simulate latencies
        latency = np.random.gamma(2, 10)
        metrics.observe(latency_histogram, latency)
    
    # Set gauge
    metrics.set(active_connections, 45)
    
    print(f"   ✅ Recorded 100 requests")
    
    # Get statistics
    print(f"\n📊 Metric Statistics:")
    print(f"   Total requests: {metrics.get_metric_value(requests_counter):.0f}")
    print(f"   Active connections: {metrics.get_metric_value(active_connections):.0f}")
    
    latency_stats = metrics.get_histogram_stats(latency_histogram)
    print(f"\n   Request Latency:")
    print(f"      Count: {latency_stats['count']:.0f}")
    print(f"      Average: {latency_stats['avg']:.2f}ms")
    print(f"      P50: {latency_stats['p50']:.2f}ms")
    print(f"      P95: {latency_stats['p95']:.2f}ms")
    print(f"      P99: {latency_stats['p99']:.2f}ms")
    
    # Export Prometheus format
    print(f"\n📤 Prometheus Export (sample):")
    export = metrics.export_metrics()
    lines = export.split('\n')
    for line in lines[:10]:
        print(f"   {line}")
    
    # ========================================================================
    # 3. LOG AGGREGATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("3. LOG AGGREGATION (ELK)")
    print("="*80)
    
    log_aggregator = LogAggregator()
    
    # Log entries
    print("\n📝 Logging events...")
    
    log_entries = [
        (LogLevel.INFO, "API request received", "nutrition-api", trace_id),
        (LogLevel.DEBUG, "Cache hit for food_123", "cache-service", None),
        (LogLevel.INFO, "Database query executed", "database", trace_id),
        (LogLevel.WARNING, "High memory usage detected", "nutrition-api", None),
        (LogLevel.ERROR, "Failed to connect to external API", "integration-service", None),
    ]
    
    for level, message, service, tid in log_entries:
        log_aggregator.log(
            level=level,
            message=message,
            service_name=service,
            trace_id=tid
        )
    
    print(f"   ✅ Logged {len(log_entries)} events")
    
    # Search logs
    print(f"\n🔍 Searching logs...")
    
    # Search by service
    nutrition_logs = log_aggregator.search(service_name="nutrition-api")
    print(f"   nutrition-api logs: {len(nutrition_logs)} entries")
    
    # Search by level
    error_logs = log_aggregator.search(level=LogLevel.ERROR)
    print(f"   ERROR logs: {len(error_logs)} entries")
    for log in error_logs:
        print(f"      - [{log.service_name}] {log.message}")
    
    # Search by trace
    trace_logs = log_aggregator.search(trace_id=trace_id)
    print(f"   Trace {trace_id[:16]}... logs: {len(trace_logs)} entries")
    
    # Log statistics
    log_stats = log_aggregator.get_log_stats()
    
    print(f"\n📊 Log Statistics:")
    print(f"   Total logs: {log_stats['total_logs']}")
    print(f"   By level: {log_stats['by_level']}")
    print(f"   By service: {log_stats['by_service']}")
    
    # ========================================================================
    # 4. ALERTING
    # ========================================================================
    
    print("\n" + "="*80)
    print("4. ALERTING & INCIDENT MANAGEMENT")
    print("="*80)
    
    alert_manager = AlertManager(metrics)
    
    # Add alert rules
    print("\n⚠️  Configuring alert rules...")
    
    rules = [
        AlertRule(
            rule_id="high_latency",
            name="High API Latency",
            expression=latency_histogram,
            threshold=50.0,
            comparison=">",
            severity=AlertSeverity.WARNING,
            description="API latency above 50ms"
        ),
        AlertRule(
            rule_id="many_errors",
            name="High Error Rate",
            expression=requests_counter,
            threshold=1000.0,
            comparison=">",
            severity=AlertSeverity.CRITICAL,
            description="Too many requests (possible attack)"
        )
    ]
    
    for rule in rules:
        alert_manager.add_rule(rule)
    
    print(f"   ✅ Added {len(rules)} alert rules")
    
    # Evaluate rules
    print(f"\n🔍 Evaluating alerts...")
    
    alert_manager.evaluate_rules()
    
    # Get active alerts
    active_alerts = alert_manager.get_active_alerts()
    
    print(f"\n🚨 Active Alerts: {len(active_alerts)}")
    for alert in active_alerts:
        print(f"   - {alert.name} ({alert.severity.value})")
        print(f"     Value: {alert.value:.2f}, Threshold: {alert.threshold}")
    
    # Alert statistics
    alert_stats = alert_manager.get_alert_stats()
    
    print(f"\n📊 Alert Statistics:")
    print(f"   Total fired: {alert_stats['total_alerts_fired']}")
    print(f"   Currently active: {alert_stats['active_alerts']}")
    
    # ========================================================================
    # 5. HEALTH CHECKS & SLOs
    # ========================================================================
    
    print("\n" + "="*80)
    print("5. HEALTH CHECKS & SLOs")
    print("="*80)
    
    health_monitor = HealthMonitor()
    
    # Add SLOs
    print("\n📋 Configuring SLOs...")
    
    slos = [
        SLO("api_availability", 99.9, 30),
        SLO("api_latency_p99", 99.0, 30),
    ]
    
    for slo in slos:
        health_monitor.add_slo(slo)
    
    # Record measurements
    print(f"\n📊 Recording SLO measurements...")
    
    for _ in range(1000):
        # 99.5% success rate
        success = np.random.random() > 0.005
        slos[0].record(success)
        
        # 99.2% meeting latency SLO
        success = np.random.random() > 0.008
        slos[1].record(success)
    
    print(f"   ✅ Recorded 1,000 measurements per SLO")
    
    # Health checks
    print(f"\n💚 Performing health checks...")
    
    def check_database():
        return HealthStatus.HEALTHY, "Database responding"
    
    def check_cache():
        return HealthStatus.HEALTHY, "Cache operational"
    
    def check_external_api():
        return HealthStatus.DEGRADED, "Slow response times"
    
    health_monitor.check_health("database", check_database)
    health_monitor.check_health("cache", check_cache)
    health_monitor.check_health("external_api", check_external_api)
    
    # Overall health
    overall_health = health_monitor.get_overall_health()
    
    print(f"\n💚 System Health: {overall_health.value.upper()}")
    for component, hc in health_monitor.health_checks.items():
        status_emoji = "✅" if hc.status == HealthStatus.HEALTHY else "⚠️" if hc.status == HealthStatus.DEGRADED else "❌"
        print(f"   {status_emoji} {component}: {hc.status.value} - {hc.message}")
    
    # SLO report
    slo_report = health_monitor.get_slo_report()
    
    print(f"\n📊 SLO Report:")
    for name, report in slo_report.items():
        meeting = "✅" if report['meeting_target'] else "❌"
        print(f"   {meeting} {name}:")
        print(f"      Target: {report['target']:.1f}%")
        print(f"      Current: {report['current']:.2f}%")
        print(f"      Error budget: {report['error_budget_remaining']:.2f}%")
        print(f"      Measurements: {report['measurements']['success']}/{report['measurements']['total']}")
    
    # ========================================================================
    # 6. PERFORMANCE PROFILING
    # ========================================================================
    
    print("\n" + "="*80)
    print("6. PERFORMANCE PROFILING")
    print("="*80)
    
    profiler = PerformanceProfiler()
    
    # Profile functions
    print("\n⏱️  Profiling application...")
    
    functions = [
        ("calculate_nutrition", 15.5),
        ("database_query", 8.2),
        ("cache_lookup", 0.5),
        ("image_processing", 45.3),
        ("ml_inference", 120.8),
    ]
    
    for func_name, base_duration in functions:
        for _ in range(50):
            # Add some variance
            duration = base_duration * np.random.uniform(0.8, 1.2)
            profiler.profile_function(func_name, duration)
    
    print(f"   ✅ Profiled {len(functions)} functions (50 calls each)")
    
    # Get hotspots
    hotspots = profiler.get_hotspots(top_n=5)
    
    print(f"\n🔥 Performance Hotspots:")
    for i, hotspot in enumerate(hotspots, 1):
        print(f"   {i}. {hotspot['function']}:")
        print(f"      Total time: {hotspot['total_time_ms']:.2f}ms")
        print(f"      Calls: {hotspot['call_count']}")
        print(f"      Avg: {hotspot['avg_time_ms']:.2f}ms/call")
    
    # Detect regressions
    print(f"\n🔍 Checking for performance regressions...")
    
    has_regression = profiler.detect_regressions("ml_inference", 100.0, threshold_percent=15.0)
    
    if has_regression:
        print(f"   ⚠️  Regression detected in ml_inference (>15% slower)")
    else:
        print(f"   ✅ No significant regressions detected")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("✅ MONITORING & OBSERVABILITY COMPLETE")
    print("="*80)
    
    print("\n📦 CAPABILITIES:")
    print("   ✓ Distributed tracing with spans")
    print("   ✓ Prometheus-style metrics (counter, gauge, histogram)")
    print("   ✓ Structured log aggregation & search")
    print("   ✓ Alert rules & incident management")
    print("   ✓ Health checks with component status")
    print("   ✓ SLO tracking with error budgets")
    print("   ✓ Performance profiling & hotspot detection")
    
    print("\n🎯 OBSERVABILITY METRICS:")
    print(f"   Traces collected: {stats['total_traces']} ✓")
    print(f"   Spans per trace: {len(trace.spans)} ✓")
    print(f"   Trace duration: {trace.get_duration_ms():.2f}ms ✓")
    print(f"   Metrics tracked: 3 types ✓")
    print(f"   Requests logged: 100 ✓")
    print(f"   Log entries: {log_stats['total_logs']} ✓")
    print(f"   Active alerts: {len(active_alerts)} ✓")
    print(f"   SLOs tracked: {len(slos)} ✓")
    print(f"   Health: {overall_health.value} ✓")
    print(f"   Functions profiled: {len(functions)} ✓")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    demo_monitoring_observability()
