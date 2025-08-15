"""
Performance Monitoring System

Comprehensive performance monitoring and metrics collection system for the
MCP Academic RAG Server. Provides real-time performance tracking, metrics
aggregation, and alerting capabilities.
"""

import time
import psutil
import threading
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import contextmanager
from enum import Enum
import json
import weakref


class MetricType(Enum):
    """Types of metrics that can be collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = None
    unit: str = "count"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'unit': self.unit
        }


@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_received: int
    open_files: int
    active_threads: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class AlertRule:
    """Performance alert rule configuration"""
    metric_name: str
    condition: str  # "greater_than", "less_than", "equals", "change_rate"
    threshold: Union[int, float]
    level: AlertLevel
    duration_seconds: int = 60  # How long condition must persist
    enabled: bool = True
    description: str = ""
    
    def __post_init__(self):
        if isinstance(self.level, str):
            self.level = AlertLevel(self.level)


@dataclass
class Alert:
    """Performance alert instance"""
    rule: AlertRule
    triggered_at: datetime
    current_value: Union[int, float]
    message: str
    resolved_at: Optional[datetime] = None
    
    @property
    def is_active(self) -> bool:
        return self.resolved_at is None
    
    def resolve(self):
        """Mark alert as resolved"""
        self.resolved_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'rule_name': self.rule.metric_name,
            'level': self.rule.level.value,
            'triggered_at': self.triggered_at.isoformat(),
            'current_value': self.current_value,
            'message': self.message,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'is_active': self.is_active
        }


class MetricsCollector:
    """Thread-safe metrics collection and aggregation"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.current_values: Dict[str, Any] = {}
        self.aggregated_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._lock = threading.Lock()
        self.logger = logging.getLogger("performance.collector")
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        with self._lock:
            self.metrics_history.append(metric)
            self.current_values[metric.name] = metric.value
            
            # Update aggregated metrics
            self._update_aggregations(metric)
            
            self.logger.debug(f"Recorded metric: {metric.name}={metric.value}")
    
    def _update_aggregations(self, metric: PerformanceMetric):
        """Update aggregated metric calculations"""
        metric_name = metric.name
        
        if metric.metric_type in [MetricType.GAUGE, MetricType.TIMER]:
            # For gauges and timers, track min/max/avg
            recent_values = [
                m.value for m in self.metrics_history 
                if m.name == metric_name and 
                m.timestamp > datetime.now() - timedelta(minutes=5)
            ]
            
            if recent_values:
                self.aggregated_metrics[metric_name].update({
                    'min': min(recent_values),
                    'max': max(recent_values),
                    'avg': sum(recent_values) / len(recent_values),
                    'count': len(recent_values)
                })
        
        elif metric.metric_type == MetricType.COUNTER:
            # For counters, track rate of change
            recent_metrics = [
                m for m in self.metrics_history 
                if m.name == metric_name and 
                m.timestamp > datetime.now() - timedelta(minutes=1)
            ]
            
            if len(recent_metrics) > 1:
                time_diff = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds()
                value_diff = recent_metrics[-1].value - recent_metrics[0].value
                rate = value_diff / time_diff if time_diff > 0 else 0
                
                self.aggregated_metrics[metric_name]['rate'] = rate
    
    def get_current_value(self, metric_name: str) -> Optional[Any]:
        """Get current value of a metric"""
        with self._lock:
            return self.current_values.get(metric_name)
    
    def get_aggregated_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get all aggregated metrics"""
        with self._lock:
            return dict(self.aggregated_metrics)
    
    def get_metrics_history(self, metric_name: str = None, 
                           since: datetime = None) -> List[PerformanceMetric]:
        """Get metrics history with optional filtering"""
        with self._lock:
            metrics = list(self.metrics_history)
        
        if metric_name:
            metrics = [m for m in metrics if m.name == metric_name]
        
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        return metrics


class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.logger = logging.getLogger("performance.system_monitor")
        
        # Initialize psutil process for consistent monitoring
        self.process = psutil.Process()
    
    def start(self, metrics_collector: MetricsCollector):
        """Start system monitoring"""
        if self.is_running:
            self.logger.warning("System monitor already running")
            return
        
        self.metrics_collector = metrics_collector
        self.is_running = True
        
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="SystemMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info(f"System monitor started with {self.collection_interval}s interval")
    
    def stop(self):
        """Stop system monitoring"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("System monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                metrics = self._collect_system_metrics()
                timestamp = datetime.now()
                
                # Record individual metrics
                metric_configs = [
                    ('system.cpu.percent', metrics.cpu_percent, MetricType.GAUGE, '%'),
                    ('system.memory.percent', metrics.memory_percent, MetricType.GAUGE, '%'),
                    ('system.memory.used_mb', metrics.memory_used_mb, MetricType.GAUGE, 'MB'),
                    ('system.memory.available_mb', metrics.memory_available_mb, MetricType.GAUGE, 'MB'),
                    ('system.disk.usage_percent', metrics.disk_usage_percent, MetricType.GAUGE, '%'),
                    ('system.disk.free_gb', metrics.disk_free_gb, MetricType.GAUGE, 'GB'),
                    ('system.network.bytes_sent', metrics.network_bytes_sent, MetricType.COUNTER, 'bytes'),
                    ('system.network.bytes_received', metrics.network_bytes_received, MetricType.COUNTER, 'bytes'),
                    ('system.files.open_count', metrics.open_files, MetricType.GAUGE, 'count'),
                    ('system.threads.active_count', metrics.active_threads, MetricType.GAUGE, 'count')
                ]
                
                for name, value, metric_type, unit in metric_configs:
                    metric = PerformanceMetric(
                        name=name,
                        value=value,
                        metric_type=metric_type,
                        timestamp=timestamp,
                        unit=unit,
                        tags={'component': 'system'}
                    )
                    self.metrics_collector.record_metric(metric)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk usage
        disk = psutil.disk_usage('.')
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024 * 1024 * 1024)
        
        # Network stats
        network = psutil.net_io_counters()
        network_bytes_sent = network.bytes_sent
        network_bytes_received = network.bytes_recv
        
        # Process stats
        try:
            open_files = len(self.process.open_files())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            open_files = 0
        
        active_threads = threading.active_count()
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            network_bytes_sent=network_bytes_sent,
            network_bytes_received=network_bytes_received,
            open_files=open_files,
            active_threads=active_threads,
            timestamp=datetime.now()
        )


class AlertManager:
    """Alert management and notification system"""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
        self.logger = logging.getLogger("performance.alerts")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        with self._lock:
            self.alert_rules[rule.metric_name] = rule
            self.logger.info(f"Added alert rule for {rule.metric_name}: {rule.condition} {rule.threshold}")
    
    def remove_alert_rule(self, metric_name: str):
        """Remove an alert rule"""
        with self._lock:
            if metric_name in self.alert_rules:
                del self.alert_rules[metric_name]
                self.logger.info(f"Removed alert rule for {metric_name}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add alert notification callback"""
        self.alert_callbacks.append(callback)
    
    def check_metrics(self, metrics_collector: MetricsCollector):
        """Check metrics against alert rules"""
        with self._lock:
            for rule_name, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                current_value = metrics_collector.get_current_value(rule.metric_name)
                if current_value is None:
                    continue
                
                should_alert = self._evaluate_condition(rule, current_value)
                
                if should_alert and rule_name not in self.active_alerts:
                    # Create new alert
                    alert = Alert(
                        rule=rule,
                        triggered_at=datetime.now(),
                        current_value=current_value,
                        message=f"{rule.metric_name} {rule.condition} {rule.threshold} (current: {current_value})"
                    )
                    
                    self.active_alerts[rule_name] = alert
                    self.alert_history.append(alert)
                    
                    # Notify callbacks
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            self.logger.error(f"Error in alert callback: {e}")
                    
                    self.logger.warning(f"Alert triggered: {alert.message}")
                
                elif not should_alert and rule_name in self.active_alerts:
                    # Resolve alert
                    alert = self.active_alerts[rule_name]
                    alert.resolve()
                    del self.active_alerts[rule_name]
                    
                    self.logger.info(f"Alert resolved: {rule.metric_name}")
    
    def _evaluate_condition(self, rule: AlertRule, current_value: Union[int, float]) -> bool:
        """Evaluate if alert condition is met"""
        if rule.condition == "greater_than":
            return current_value > rule.threshold
        elif rule.condition == "less_than":
            return current_value < rule.threshold
        elif rule.condition == "equals":
            return current_value == rule.threshold
        elif rule.condition == "not_equals":
            return current_value != rule.threshold
        else:
            self.logger.warning(f"Unknown alert condition: {rule.condition}")
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self._lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        with self._lock:
            return self.alert_history[-limit:]


class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = MetricsCollector(
            max_history=self.config.get('max_metrics_history', 10000)
        )
        self.system_monitor = SystemMonitor(
            collection_interval=self.config.get('system_monitor_interval', 1.0)
        )
        self.alert_manager = AlertManager()
        
        self.is_running = False
        self.alert_check_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger("performance.monitor")
        
        # Setup default alert rules
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                metric_name="system.cpu.percent",
                condition="greater_than",
                threshold=90.0,
                level=AlertLevel.WARNING,
                description="High CPU usage"
            ),
            AlertRule(
                metric_name="system.memory.percent",
                condition="greater_than",
                threshold=85.0,
                level=AlertLevel.WARNING,
                description="High memory usage"
            ),
            AlertRule(
                metric_name="system.disk.usage_percent",
                condition="greater_than",
                threshold=90.0,
                level=AlertLevel.ERROR,
                description="High disk usage"
            )
        ]
        
        for rule in default_rules:
            self.alert_manager.add_alert_rule(rule)
    
    def start(self):
        """Start performance monitoring"""
        if self.is_running:
            self.logger.warning("Performance monitor already running")
            return
        
        self.is_running = True
        
        # Start system monitoring
        self.system_monitor.start(self.metrics_collector)
        
        # Start alert checking
        self.alert_check_thread = threading.Thread(
            target=self._alert_check_loop,
            name="AlertChecker",
            daemon=True
        )
        self.alert_check_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop(self):
        """Stop performance monitoring"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop system monitoring
        self.system_monitor.stop()
        
        # Stop alert checking
        if self.alert_check_thread and self.alert_check_thread.is_alive():
            self.alert_check_thread.join(timeout=5.0)
        
        self.logger.info("Performance monitoring stopped")
    
    def _alert_check_loop(self):
        """Alert checking loop"""
        check_interval = self.config.get('alert_check_interval', 5.0)
        
        while self.is_running:
            try:
                self.alert_manager.check_metrics(self.metrics_collector)
                time.sleep(check_interval)
            except Exception as e:
                self.logger.error(f"Error in alert checking loop: {e}")
                time.sleep(check_interval)
    
    def record_metric(self, name: str, value: Union[int, float], 
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Dict[str, str] = None, unit: str = "count"):
        """Record a custom metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=unit
        )
        self.metrics_collector.record_metric(metric)
    
    @contextmanager
    def timer(self, name: str, tags: Dict[str, str] = None):
        """Context manager for timing operations"""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.record_metric(
                name=name,
                value=duration,
                metric_type=MetricType.TIMER,
                tags=tags,
                unit="seconds"
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring system status"""
        return {
            'is_running': self.is_running,
            'metrics_collected': len(self.metrics_collector.metrics_history),
            'active_alerts': len(self.alert_manager.active_alerts),
            'alert_rules': len(self.alert_manager.alert_rules),
            'system_monitor_running': self.system_monitor.is_running,
            'current_metrics': self.metrics_collector.get_aggregated_metrics()
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        if format == "json":
            metrics = []
            for metric in self.metrics_collector.metrics_history:
                metrics.append(metric.to_dict())
            
            return json.dumps({
                'metrics': metrics,
                'aggregated': self.metrics_collector.get_aggregated_metrics(),
                'alerts': [alert.to_dict() for alert in self.alert_manager.get_active_alerts()],
                'exported_at': datetime.now().isoformat()
            }, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(config: Dict[str, Any] = None) -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(config)
    
    return _performance_monitor


def initialize_monitoring(config: Dict[str, Any] = None) -> PerformanceMonitor:
    """Initialize and start performance monitoring"""
    monitor = get_performance_monitor(config)
    monitor.start()
    return monitor


# Convenient decorators for performance tracking
def track_performance(metric_name: str = None, tags: Dict[str, str] = None):
    """Decorator to track function performance"""
    def decorator(func):
        nonlocal metric_name
        if metric_name is None:
            metric_name = f"function.{func.__module__}.{func.__name__}.duration"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                monitor = get_performance_monitor()
                with monitor.timer(metric_name, tags):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                monitor = get_performance_monitor()
                with monitor.timer(metric_name, tags):
                    return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator


def count_calls(metric_name: str = None, tags: Dict[str, str] = None):
    """Decorator to count function calls"""
    def decorator(func):
        nonlocal metric_name
        if metric_name is None:
            metric_name = f"function.{func.__module__}.{func.__name__}.calls"
        
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            monitor.record_metric(
                metric_name,
                1,
                MetricType.COUNTER,
                tags
            )
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator