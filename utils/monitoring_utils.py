"""
Monitoring and observability utilities for the Academic RAG Server

This module provides monitoring features including:
- Prometheus metrics
- OpenTelemetry tracing
- Health checks
- Performance metrics
- Custom dashboards
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import contextmanager
from functools import wraps
import json
import os

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        CollectorRegistry, generate_latest
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc import trace_exporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.flask import FlaskInstrumentor
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Single metric value with timestamp"""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Health check result"""
    name: str
    status: str  # healthy, unhealthy, degraded
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MetricsCollector:
    """Custom metrics collector for non-Prometheus environments"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._lock = threading.RLock()
    
    def record(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        with self._lock:
            metric_value = MetricValue(
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels or {}
            )
            self._metrics[metric_name].append(metric_value)
    
    def get_metric(self, metric_name: str, time_window: Optional[timedelta] = None) -> List[MetricValue]:
        """Get metric values within time window"""
        with self._lock:
            if metric_name not in self._metrics:
                return []
            
            values = list(self._metrics[metric_name])
            
            if time_window:
                cutoff = datetime.utcnow() - time_window
                values = [v for v in values if v.timestamp >= cutoff]
            
            return values
    
    def get_summary(self, metric_name: str, time_window: Optional[timedelta] = None) -> Dict[str, float]:
        """Get statistical summary of metric"""
        values = self.get_metric(metric_name, time_window)
        
        if not values:
            return {
                'count': 0,
                'min': 0,
                'max': 0,
                'avg': 0,
                'sum': 0
            }
        
        nums = [v.value for v in values]
        return {
            'count': len(nums),
            'min': min(nums),
            'max': max(nums),
            'avg': sum(nums) / len(nums),
            'sum': sum(nums)
        }


class PrometheusMetrics:
    """Prometheus metrics wrapper"""
    
    def __init__(self, namespace: str = "academic_rag", registry: Optional[CollectorRegistry] = None):
        if not HAS_PROMETHEUS:
            raise ImportError("prometheus_client not installed")
        
        self.namespace = namespace
        self.registry = registry or CollectorRegistry()
        
        # Define metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            namespace=namespace,
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            namespace=namespace,
            registry=self.registry
        )
        
        self.document_processing_count = Counter(
            'document_processing_total',
            'Total documents processed',
            ['processor', 'status'],
            namespace=namespace,
            registry=self.registry
        )
        
        self.document_processing_duration = Histogram(
            'document_processing_duration_seconds',
            'Document processing duration',
            ['processor'],
            namespace=namespace,
            registry=self.registry
        )
        
        self.rag_query_count = Counter(
            'rag_queries_total',
            'Total RAG queries',
            ['status'],
            namespace=namespace,
            registry=self.registry
        )
        
        self.rag_query_duration = Histogram(
            'rag_query_duration_seconds',
            'RAG query duration',
            namespace=namespace,
            registry=self.registry
        )
        
        self.active_sessions = Gauge(
            'active_sessions',
            'Number of active sessions',
            namespace=namespace,
            registry=self.registry
        )
        
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            namespace=namespace,
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            namespace=namespace,
            registry=self.registry
        )
        
        self.error_count = Counter(
            'errors_total',
            'Total errors',
            ['error_type', 'component'],
            namespace=namespace,
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],
            namespace=namespace,
            registry=self.registry
        )
    
    def get_metrics(self) -> bytes:
        """Generate metrics in Prometheus format"""
        return generate_latest(self.registry)


class OpenTelemetryTracer:
    """OpenTelemetry tracing wrapper"""
    
    def __init__(
        self,
        service_name: str = "academic-rag-server",
        endpoint: Optional[str] = None
    ):
        if not HAS_OPENTELEMETRY:
            raise ImportError("opentelemetry packages not installed")
        
        self.service_name = service_name
        
        # Configure tracer
        if endpoint:
            otlp_exporter = trace_exporter.OTLPSpanExporter(
                endpoint=endpoint,
                insecure=True
            )
            
            provider = TracerProvider()
            processor = BatchSpanProcessor(otlp_exporter)
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
        
        self.tracer = trace.get_tracer(service_name)
    
    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a trace span"""
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            yield span
    
    def instrument_flask(self, app):
        """Instrument Flask application"""
        FlaskInstrumentor().instrument_app(app)


class MonitoringManager:
    """Central monitoring manager"""
    
    def __init__(
        self,
        enable_prometheus: bool = True,
        enable_opentelemetry: bool = False,
        prometheus_namespace: str = "academic_rag",
        service_name: str = "academic-rag-server",
        otlp_endpoint: Optional[str] = None
    ):
        # Initialize collectors
        self.custom_metrics = MetricsCollector()
        
        # Initialize Prometheus if available and enabled
        self.prometheus = None
        if enable_prometheus and HAS_PROMETHEUS:
            try:
                self.prometheus = PrometheusMetrics(namespace=prometheus_namespace)
                logger.info("Prometheus metrics enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Prometheus: {e}")
        
        # Initialize OpenTelemetry if available and enabled
        self.tracer = None
        if enable_opentelemetry and HAS_OPENTELEMETRY:
            try:
                self.tracer = OpenTelemetryTracer(
                    service_name=service_name,
                    endpoint=otlp_endpoint
                )
                logger.info("OpenTelemetry tracing enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenTelemetry: {e}")
        
        # Health checks
        self._health_checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._health_check_results: Dict[str, HealthCheckResult] = {}
        self._health_check_lock = threading.RLock()
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: str = "gauge",
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric value"""
        # Always record in custom collector
        self.custom_metrics.record(name, value, labels)
        
        # Record in Prometheus if available
        if self.prometheus:
            try:
                # Map to appropriate Prometheus metric
                # This is simplified - in practice you'd have proper metric registration
                if hasattr(self.prometheus, name):
                    metric = getattr(self.prometheus, name)
                    if labels:
                        metric.labels(**labels).set(value)
                    else:
                        metric.set(value)
            except Exception as e:
                logger.error(f"Error recording Prometheus metric: {e}")
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ):
        """Record HTTP request metrics"""
        labels = {
            'method': method,
            'endpoint': endpoint,
            'status': str(status)
        }
        
        # Custom metrics
        self.custom_metrics.record('http_requests', 1, labels)
        self.custom_metrics.record('http_duration', duration, labels)
        
        # Prometheus metrics
        if self.prometheus:
            self.prometheus.request_count.labels(
                method=method,
                endpoint=endpoint,
                status=str(status)
            ).inc()
            
            self.prometheus.request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
    
    def record_document_processing(
        self,
        processor: str,
        status: str,
        duration: float
    ):
        """Record document processing metrics"""
        labels = {
            'processor': processor,
            'status': status
        }
        
        self.custom_metrics.record('document_processing', 1, labels)
        self.custom_metrics.record('processing_duration', duration, labels)
        
        if self.prometheus:
            self.prometheus.document_processing_count.labels(
                processor=processor,
                status=status
            ).inc()
            
            self.prometheus.document_processing_duration.labels(
                processor=processor
            ).observe(duration)
    
    def record_error(self, error_type: str, component: str):
        """Record error occurrence"""
        labels = {
            'error_type': error_type,
            'component': component
        }
        
        self.custom_metrics.record('errors', 1, labels)
        
        if self.prometheus:
            self.prometheus.error_count.labels(
                error_type=error_type,
                component=component
            ).inc()
    
    def trace_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a trace span if tracing is enabled"""
        if self.tracer:
            return self.tracer.span(name, attributes)
        else:
            # Return a no-op context manager
            @contextmanager
            def noop():
                yield None
            return noop()
    
    def register_health_check(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult]
    ):
        """Register a health check function"""
        with self._health_check_lock:
            self._health_checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        
        with self._health_check_lock:
            for name, check_func in self._health_checks.items():
                try:
                    result = check_func()
                    results[name] = result
                    self._health_check_results[name] = result
                except Exception as e:
                    results[name] = HealthCheckResult(
                        name=name,
                        status="unhealthy",
                        message=f"Health check failed: {str(e)}",
                        details={"error": str(e)}
                    )
        
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        results = self.run_health_checks()
        
        # Determine overall status
        statuses = [r.status for r in results.values()]
        if all(s == "healthy" for s in statuses):
            overall_status = "healthy"
        elif any(s == "unhealthy" for s in statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                name: {
                    "status": result.status,
                    "message": result.message,
                    "details": result.details,
                    "timestamp": result.timestamp.isoformat()
                }
                for name, result in results.items()
            }
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {}
        }
        
        # Get custom metrics summary
        for metric_name in ['http_requests', 'document_processing', 'errors']:
            summary["metrics"][metric_name] = self.custom_metrics.get_summary(
                metric_name,
                time_window=timedelta(minutes=5)
            )
        
        return summary


# Decorator for monitoring function execution
def monitor_execution(
    monitoring: Optional[MonitoringManager] = None,
    metric_name: Optional[str] = None,
    trace_name: Optional[str] = None
):
    """Decorator to monitor function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            _metric_name = metric_name or f"{func.__module__}.{func.__name__}"
            _trace_name = trace_name or func.__name__
            
            # Start trace span if monitoring available
            span_context = monitoring.trace_span(_trace_name) if monitoring else None
            
            try:
                with span_context or contextlib.nullcontext():
                    result = func(*args, **kwargs)
                    
                    # Record success metrics
                    duration = time.time() - start_time
                    if monitoring:
                        monitoring.record_metric(
                            f"{_metric_name}_duration",
                            duration,
                            labels={"status": "success"}
                        )
                        monitoring.record_metric(
                            f"{_metric_name}_count",
                            1,
                            labels={"status": "success"}
                        )
                    
                    return result
                    
            except Exception as e:
                # Record error metrics
                duration = time.time() - start_time
                if monitoring:
                    monitoring.record_metric(
                        f"{_metric_name}_duration",
                        duration,
                        labels={"status": "error"}
                    )
                    monitoring.record_metric(
                        f"{_metric_name}_count",
                        1,
                        labels={"status": "error"}
                    )
                    monitoring.record_error(
                        error_type=type(e).__name__,
                        component=func.__module__
                    )
                raise
        
        return wrapper
    return decorator


# Global monitoring instance
_monitoring_instance: Optional[MonitoringManager] = None


def get_monitoring() -> Optional[MonitoringManager]:
    """Get global monitoring instance"""
    return _monitoring_instance


def init_monitoring(**kwargs) -> MonitoringManager:
    """Initialize global monitoring instance"""
    global _monitoring_instance
    _monitoring_instance = MonitoringManager(**kwargs)
    return _monitoring_instance


# Example health check functions
def database_health_check() -> HealthCheckResult:
    """Example database health check"""
    try:
        # Check database connection
        # This is just an example - implement actual check
        return HealthCheckResult(
            name="database",
            status="healthy",
            message="Database connection OK"
        )
    except Exception as e:
        return HealthCheckResult(
            name="database",
            status="unhealthy",
            message="Database connection failed",
            details={"error": str(e)}
        )


def memory_health_check(threshold_mb: float = 1000) -> HealthCheckResult:
    """Check memory usage"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        used_mb = (memory.total - memory.available) / 1024 / 1024
        
        if used_mb > threshold_mb:
            return HealthCheckResult(
                name="memory",
                status="degraded",
                message=f"High memory usage: {used_mb:.1f}MB",
                details={"used_mb": used_mb, "threshold_mb": threshold_mb}
            )
        
        return HealthCheckResult(
            name="memory",
            status="healthy",
            message=f"Memory usage OK: {used_mb:.1f}MB",
            details={"used_mb": used_mb}
        )
    except Exception as e:
        return HealthCheckResult(
            name="memory",
            status="unhealthy",
            message="Failed to check memory",
            details={"error": str(e)}
        )


# Example usage
if __name__ == "__main__":
    # Initialize monitoring
    monitoring = init_monitoring(
        enable_prometheus=True,
        enable_opentelemetry=False
    )
    
    # Register health checks
    monitoring.register_health_check("database", database_health_check)
    monitoring.register_health_check("memory", memory_health_check)
    
    # Example monitored function
    @monitor_execution(monitoring)
    def process_document(doc_id: str):
        time.sleep(0.1)  # Simulate processing
        if doc_id == "error":
            raise ValueError("Processing failed")
        return f"Processed {doc_id}"
    
    # Test execution
    try:
        result = process_document("doc123")
        print(f"Result: {result}")
    except:
        pass
    
    # Get metrics summary
    print(json.dumps(monitoring.get_metrics_summary(), indent=2))
    
    # Get health status
    print(json.dumps(monitoring.get_health_status(), indent=2))