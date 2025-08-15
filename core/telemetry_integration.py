"""
OpenTelemetry Integration Module

Provides comprehensive observability integration using OpenTelemetry for
distributed tracing, metrics collection, and logging correlation.
"""

import os
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
from functools import wraps

try:
    # OpenTelemetry core
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    
    # Instrumentation
    from opentelemetry.instrumentation.auto_instrumentation import sitecustomize
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
    
    # Exporters
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    
    # Resources and semantic conventions
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.semconv.trace import SpanAttributes
    
    # Propagation
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.b3 import B3MultiFormat
    from opentelemetry.propagators.composite import CompositeHTTPPropagator
    from opentelemetry.propagators.jaeger import JaegerPropagator
    
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Create mock classes for when OpenTelemetry is not available
    class MockTracer:
        def start_as_current_span(self, *args, **kwargs):
            return MockSpan()
        
        def start_span(self, *args, **kwargs):
            return MockSpan()
    
    class MockSpan:
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def set_attribute(self, *args):
            pass
        
        def add_event(self, *args):
            pass
        
        def record_exception(self, *args):
            pass
        
        def set_status(self, *args):
            pass
    
    class MockMeter:
        def create_counter(self, *args, **kwargs):
            return MockInstrument()
        
        def create_histogram(self, *args, **kwargs):
            return MockInstrument()
        
        def create_gauge(self, *args, **kwargs):
            return MockInstrument()
    
    class MockInstrument:
        def add(self, *args, **kwargs):
            pass
        
        def record(self, *args, **kwargs):
            pass


class TelemetryConfig:
    """Configuration for OpenTelemetry integration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Service information
        self.service_name = self.config.get('service_name', 'mcp-academic-rag-server')
        self.service_version = self.config.get('service_version', '1.0.0')
        self.environment = self.config.get('environment', 'development')
        
        # Tracing configuration
        self.tracing_enabled = self.config.get('tracing', {}).get('enabled', True)
        self.trace_sampling_ratio = self.config.get('tracing', {}).get('sampling_ratio', 1.0)
        
        # Metrics configuration
        self.metrics_enabled = self.config.get('metrics', {}).get('enabled', True)
        self.metrics_export_interval = self.config.get('metrics', {}).get('export_interval', 30)
        
        # Exporters
        self.exporters = self.config.get('exporters', {})
        
        # Instrumentation
        self.auto_instrumentation = self.config.get('auto_instrumentation', {})


class TelemetryIntegration:
    """Main OpenTelemetry integration class"""
    
    def __init__(self, config: TelemetryConfig = None):
        self.config = config or TelemetryConfig()
        self.logger = logging.getLogger("telemetry.integration")
        
        self.tracer_provider: Optional[TracerProvider] = None
        self.meter_provider: Optional[MeterProvider] = None
        self.tracer = None
        self.meter = None
        
        self._initialized = False
        self._span_processors = []
        self._metric_readers = []
        
        if not OTEL_AVAILABLE:
            self.logger.warning(
                "OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk"
            )
            self._setup_mock_telemetry()
    
    def _setup_mock_telemetry(self):
        """Setup mock telemetry when OpenTelemetry is not available"""
        self.tracer = MockTracer()
        self.meter = MockMeter()
        self._initialized = True
    
    def initialize(self):
        """Initialize OpenTelemetry configuration"""
        if not OTEL_AVAILABLE:
            self.logger.warning("OpenTelemetry not available, using mock implementation")
            return
        
        if self._initialized:
            self.logger.warning("Telemetry already initialized")
            return
        
        try:
            # Setup resource
            resource = self._create_resource()
            
            # Initialize tracing
            if self.config.tracing_enabled:
                self._setup_tracing(resource)
            
            # Initialize metrics
            if self.config.metrics_enabled:
                self._setup_metrics(resource)
            
            # Setup auto-instrumentation
            self._setup_auto_instrumentation()
            
            # Setup propagators
            self._setup_propagators()
            
            self._initialized = True
            self.logger.info("OpenTelemetry initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self._setup_mock_telemetry()
    
    def _create_resource(self) -> Resource:
        """Create OpenTelemetry resource"""
        return Resource.create({
            ResourceAttributes.SERVICE_NAME: self.config.service_name,
            ResourceAttributes.SERVICE_VERSION: self.config.service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.environment,
            "component": "mcp-rag-server"
        })
    
    def _setup_tracing(self, resource: Resource):
        """Setup tracing configuration"""
        self.tracer_provider = TracerProvider(
            resource=resource,
            sampler=trace.sampling.TraceIdRatioBased(self.config.trace_sampling_ratio)
        )
        
        # Setup span processors and exporters
        self._setup_trace_exporters()
        
        # Set global tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        self.tracer = trace.get_tracer(__name__)
        
        self.logger.info("Tracing configured successfully")
    
    def _setup_trace_exporters(self):
        """Setup trace exporters"""
        exporters_config = self.config.exporters.get('tracing', {})
        
        # Console exporter (development)
        if exporters_config.get('console', {}).get('enabled', False):
            console_exporter = ConsoleSpanExporter()
            console_processor = BatchSpanProcessor(console_exporter)
            self.tracer_provider.add_span_processor(console_processor)
            self._span_processors.append(console_processor)
        
        # OTLP exporter
        otlp_config = exporters_config.get('otlp', {})
        if otlp_config.get('enabled', False):
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_config.get('endpoint', 'http://localhost:4317'),
                headers=otlp_config.get('headers', {}),
                timeout=otlp_config.get('timeout', 30)
            )
            otlp_processor = BatchSpanProcessor(otlp_exporter)
            self.tracer_provider.add_span_processor(otlp_processor)
            self._span_processors.append(otlp_processor)
        
        # Jaeger exporter
        jaeger_config = exporters_config.get('jaeger', {})
        if jaeger_config.get('enabled', False):
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_config.get('agent_host', 'localhost'),
                agent_port=jaeger_config.get('agent_port', 6831),
                collector_endpoint=jaeger_config.get('collector_endpoint')
            )
            jaeger_processor = BatchSpanProcessor(jaeger_exporter)
            self.tracer_provider.add_span_processor(jaeger_processor)
            self._span_processors.append(jaeger_processor)
    
    def _setup_metrics(self, resource: Resource):
        """Setup metrics configuration"""
        # Setup metric readers
        self._setup_metric_readers()
        
        self.meter_provider = MeterProvider(
            resource=resource,
            metric_readers=self._metric_readers
        )
        
        # Set global meter provider
        metrics.set_meter_provider(self.meter_provider)
        self.meter = metrics.get_meter(__name__)
        
        self.logger.info("Metrics configured successfully")
    
    def _setup_metric_readers(self):
        """Setup metric readers and exporters"""
        exporters_config = self.config.exporters.get('metrics', {})
        
        # Console exporter (development)
        if exporters_config.get('console', {}).get('enabled', False):
            console_reader = PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=self.config.metrics_export_interval * 1000
            )
            self._metric_readers.append(console_reader)
        
        # OTLP exporter
        otlp_config = exporters_config.get('otlp', {})
        if otlp_config.get('enabled', False):
            otlp_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=otlp_config.get('endpoint', 'http://localhost:4317'),
                    headers=otlp_config.get('headers', {}),
                    timeout=otlp_config.get('timeout', 30)
                ),
                export_interval_millis=self.config.metrics_export_interval * 1000
            )
            self._metric_readers.append(otlp_reader)
        
        # Prometheus exporter
        prometheus_config = exporters_config.get('prometheus', {})
        if prometheus_config.get('enabled', False):
            prometheus_reader = PrometheusMetricReader(
                endpoint=prometheus_config.get('endpoint', 'localhost:9464')
            )
            self._metric_readers.append(prometheus_reader)
    
    def _setup_auto_instrumentation(self):
        """Setup automatic instrumentation"""
        instrumentation_config = self.config.auto_instrumentation
        
        # Requests instrumentation
        if instrumentation_config.get('requests', {}).get('enabled', True):
            RequestsInstrumentor().instrument()
        
        # Logging instrumentation
        if instrumentation_config.get('logging', {}).get('enabled', True):
            LoggingInstrumentor().instrument()
        
        # SQLite instrumentation
        if instrumentation_config.get('sqlite', {}).get('enabled', True):
            SQLite3Instrumentor().instrument()
        
        self.logger.info("Auto-instrumentation configured")
    
    def _setup_propagators(self):
        """Setup trace context propagation"""
        propagators = []
        
        # Add configured propagators
        propagator_configs = self.config.config.get('propagators', ['tracecontext', 'baggage'])
        
        if 'b3' in propagator_configs:
            propagators.append(B3MultiFormat())
        
        if 'jaeger' in propagator_configs:
            propagators.append(JaegerPropagator())
        
        if propagators:
            set_global_textmap(CompositeHTTPPropagator(propagators))
    
    def shutdown(self):
        """Shutdown telemetry system"""
        if not self._initialized or not OTEL_AVAILABLE:
            return
        
        try:
            # Shutdown span processors
            for processor in self._span_processors:
                processor.shutdown()
            
            # Shutdown metric readers
            for reader in self._metric_readers:
                reader.shutdown()
            
            self.logger.info("Telemetry shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during telemetry shutdown: {e}")
    
    @contextmanager
    def trace_span(self, name: str, attributes: Dict[str, Any] = None):
        """Create a trace span context manager"""
        if not self._initialized:
            yield
            return
        
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span
    
    def create_counter(self, name: str, description: str = "", unit: str = "1"):
        """Create a metrics counter"""
        if not self._initialized:
            return MockInstrument()
        
        return self.meter.create_counter(
            name=name,
            description=description,
            unit=unit
        )
    
    def create_histogram(self, name: str, description: str = "", unit: str = "1"):
        """Create a metrics histogram"""
        if not self._initialized:
            return MockInstrument()
        
        return self.meter.create_histogram(
            name=name,
            description=description,
            unit=unit
        )
    
    def create_gauge(self, name: str, description: str = "", unit: str = "1"):
        """Create a metrics gauge"""
        if not self._initialized:
            return MockInstrument()
        
        return self.meter.create_observable_gauge(
            name=name,
            description=description,
            unit=unit
        )


class RAGTelemetryInstrumentation:
    """RAG-specific telemetry instrumentation"""
    
    def __init__(self, telemetry: TelemetryIntegration):
        self.telemetry = telemetry
        
        # Create RAG-specific metrics
        self.document_processing_counter = telemetry.create_counter(
            "rag_documents_processed_total",
            "Total number of documents processed",
            "documents"
        )
        
        self.document_processing_duration = telemetry.create_histogram(
            "rag_document_processing_duration_seconds",
            "Document processing duration",
            "seconds"
        )
        
        self.query_counter = telemetry.create_counter(
            "rag_queries_total",
            "Total number of RAG queries",
            "queries"
        )
        
        self.query_duration = telemetry.create_histogram(
            "rag_query_duration_seconds",
            "RAG query duration",
            "seconds"
        )
        
        self.retrieval_accuracy = telemetry.create_histogram(
            "rag_retrieval_accuracy",
            "RAG retrieval accuracy score",
            "score"
        )
        
        self.vector_store_operations = telemetry.create_counter(
            "rag_vector_store_operations_total",
            "Vector store operations",
            "operations"
        )
    
    def trace_document_processing(self, document_id: str):
        """Trace document processing operation"""
        return self.telemetry.trace_span(
            "document_processing",
            attributes={
                "document.id": document_id,
                "operation.type": "document_processing"
            }
        )
    
    def trace_rag_query(self, query: str, user_id: str = None):
        """Trace RAG query operation"""
        attributes = {
            "query.length": len(query),
            "operation.type": "rag_query"
        }
        if user_id:
            attributes["user.id"] = user_id
        
        return self.telemetry.trace_span("rag_query", attributes)
    
    def trace_vector_search(self, query_embedding_size: int, top_k: int):
        """Trace vector similarity search"""
        return self.telemetry.trace_span(
            "vector_search",
            attributes={
                "embedding.size": query_embedding_size,
                "search.top_k": top_k,
                "operation.type": "vector_search"
            }
        )
    
    def record_document_processed(self, processing_time: float, success: bool = True):
        """Record document processing metrics"""
        labels = {"status": "success" if success else "error"}
        self.document_processing_counter.add(1, labels)
        if success:
            self.document_processing_duration.record(processing_time, labels)
    
    def record_query_executed(self, query_time: float, retrieval_count: int, success: bool = True):
        """Record query execution metrics"""
        labels = {"status": "success" if success else "error"}
        self.query_counter.add(1, labels)
        if success:
            self.query_duration.record(query_time, labels)
            self.vector_store_operations.add(1, {"operation": "search"})


# Global telemetry instances
_telemetry_integration: Optional[TelemetryIntegration] = None
_rag_instrumentation: Optional[RAGTelemetryInstrumentation] = None


def initialize_telemetry(config: Dict[str, Any] = None) -> TelemetryIntegration:
    """Initialize global telemetry integration"""
    global _telemetry_integration, _rag_instrumentation
    
    if _telemetry_integration is None:
        telemetry_config = TelemetryConfig(config)
        _telemetry_integration = TelemetryIntegration(telemetry_config)
        _telemetry_integration.initialize()
        
        _rag_instrumentation = RAGTelemetryInstrumentation(_telemetry_integration)
    
    return _telemetry_integration


def get_telemetry() -> TelemetryIntegration:
    """Get global telemetry integration instance"""
    global _telemetry_integration
    
    if _telemetry_integration is None:
        _telemetry_integration = initialize_telemetry()
    
    return _telemetry_integration


def get_rag_instrumentation() -> RAGTelemetryInstrumentation:
    """Get RAG-specific instrumentation"""
    global _rag_instrumentation
    
    if _rag_instrumentation is None:
        telemetry = get_telemetry()
        _rag_instrumentation = RAGTelemetryInstrumentation(telemetry)
    
    return _rag_instrumentation


# Decorators for easy instrumentation
def trace_function(operation_name: str = None, attributes: Dict[str, Any] = None):
    """Decorator to trace function execution"""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            telemetry = get_telemetry()
            with telemetry.trace_span(operation_name, attributes):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def trace_async_function(operation_name: str = None, attributes: Dict[str, Any] = None):
    """Decorator to trace async function execution"""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            telemetry = get_telemetry()
            with telemetry.trace_span(operation_name, attributes):
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def count_function_calls(metric_name: str = None):
    """Decorator to count function calls"""
    def decorator(func):
        nonlocal metric_name
        if metric_name is None:
            metric_name = f"function_calls_{func.__module__.replace('.', '_')}_{func.__name__}_total"
        
        telemetry = get_telemetry()
        counter = telemetry.create_counter(metric_name, f"Calls to {func.__name__}")
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            counter.add(1)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def time_function_execution(metric_name: str = None):
    """Decorator to time function execution"""
    def decorator(func):
        nonlocal metric_name
        if metric_name is None:
            metric_name = f"function_duration_{func.__module__.replace('.', '_')}_{func.__name__}_seconds"
        
        telemetry = get_telemetry()
        histogram = telemetry.create_histogram(metric_name, f"Duration of {func.__name__}", "seconds")
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                histogram.record(duration, {"status": "success"})
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                histogram.record(duration, {"status": "error"})
                raise
        
        return wrapper
    return decorator