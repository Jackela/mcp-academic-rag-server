"""
OpenTelemetry Integration Module

Provides comprehensive observability integration using OpenTelemetry for
distributed tracing, metrics collection, and logging correlation.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from importlib import import_module
from typing import Any, Callable, Coroutine, Dict, Iterator, List, Optional, ParamSpec, TypeVar

try:
    from opentelemetry import metrics, trace
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    from opentelemetry.metrics._internal.instrument import Gauge
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.composite import CompositeHTTPPropagator
    from opentelemetry.propagators.textmap import TextMapPropagator
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, MetricReader, PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanProcessor
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

P = ParamSpec("P")
R = TypeVar("R")


def _optional_component(module: str, name: str) -> Any:
    """Load only an explicitly enabled plugin; never turn missing plugins into success."""
    try:
        return getattr(import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"Enabled telemetry component {module}.{name} is unavailable") from exc


class TelemetryConfig:
    """Configuration for OpenTelemetry integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

        # Service information
        self.service_name = self.config.get("service_name", "mcp-academic-rag-server")
        self.service_version = self.config.get("service_version", "1.0.0")
        self.environment = self.config.get("environment", "development")

        # Tracing configuration
        self.tracing_enabled = self.config.get("tracing", {}).get("enabled", True)
        self.trace_sampling_ratio = self.config.get("tracing", {}).get("sampling_ratio", 1.0)

        # Metrics configuration
        self.metrics_enabled = self.config.get("metrics", {}).get("enabled", True)
        self.metrics_export_interval = self.config.get("metrics", {}).get("export_interval", 30)

        # Exporters
        self.exporters = self.config.get("exporters", {})

        # Instrumentation
        self.auto_instrumentation = self.config.get("auto_instrumentation", {})


class TelemetryIntegration:
    """Main OpenTelemetry integration class"""

    def __init__(self, config: Optional[TelemetryConfig] = None) -> None:
        self.config = config or TelemetryConfig()
        self.logger = logging.getLogger("telemetry.integration")

        self.tracer_provider: Optional[TracerProvider] = None
        self.meter_provider: Optional[MeterProvider] = None
        self.tracer: Optional[trace.Tracer] = None
        self.meter: Optional[metrics.Meter] = None

        self._initialized = False
        self._span_processors: List[SpanProcessor] = []
        self._metric_readers: List[MetricReader] = []
        self._instrumentors: List[Any] = []

    def initialize(self) -> None:
        """Initialize OpenTelemetry configuration"""
        if not OTEL_AVAILABLE:
            raise ImportError("OpenTelemetry SDK unavailable; install mcp-academic-rag-server[monitoring]")

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

            if not self.config.tracing_enabled:
                self.tracer = trace.NoOpTracerProvider().get_tracer(__name__)
            if not self.config.metrics_enabled:
                self.meter = metrics.NoOpMeterProvider().get_meter(__name__)

            # Setup auto-instrumentation
            self._setup_auto_instrumentation()

            # Setup propagators
            self._setup_propagators()

            self._initialized = True
            self.logger.info("OpenTelemetry initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self.shutdown()
            raise

    def _create_resource(self) -> Resource:
        """Create OpenTelemetry resource"""
        return Resource.create(
            {
                "service.name": self.config.service_name,
                "service.version": self.config.service_version,
                "deployment.environment.name": self.config.environment,
                "component": "mcp-rag-server",
            }
        )

    def _setup_tracing(self, resource: Resource) -> None:
        """Setup tracing configuration"""
        self.tracer_provider = TracerProvider(
            resource=resource, sampler=TraceIdRatioBased(self.config.trace_sampling_ratio)
        )

        # Setup span processors and exporters
        self._setup_trace_exporters()

        # Set global tracer provider
        self.tracer = self.tracer_provider.get_tracer(__name__)

        self.logger.info("Tracing configured successfully")

    def _setup_trace_exporters(self) -> None:
        """Setup trace exporters"""
        assert self.tracer_provider is not None
        exporters_config = self.config.exporters.get("tracing", {})

        # Console exporter (development)
        if exporters_config.get("console", {}).get("enabled", False):
            console_exporter = ConsoleSpanExporter()
            console_processor = BatchSpanProcessor(console_exporter)
            self.tracer_provider.add_span_processor(console_processor)
            self._span_processors.append(console_processor)

        # OTLP exporter
        otlp_config = exporters_config.get("otlp", {})
        if otlp_config.get("enabled", False):
            exporter_class = _optional_component(
                "opentelemetry.exporter.otlp.proto.grpc.trace_exporter", "OTLPSpanExporter"
            )
            otlp_exporter = exporter_class(
                endpoint=otlp_config.get("endpoint", "http://localhost:4317"),
                headers=otlp_config.get("headers", {}),
                timeout=otlp_config.get("timeout", 30),
            )
            otlp_processor = BatchSpanProcessor(otlp_exporter)
            self.tracer_provider.add_span_processor(otlp_processor)
            self._span_processors.append(otlp_processor)

        # Jaeger exporter
        jaeger_config = exporters_config.get("jaeger", {})
        if jaeger_config.get("enabled", False):
            exporter_class = _optional_component("opentelemetry.exporter.jaeger.thrift", "JaegerExporter")
            jaeger_exporter = exporter_class(
                agent_host_name=jaeger_config.get("agent_host", "localhost"),
                agent_port=jaeger_config.get("agent_port", 6831),
                collector_endpoint=jaeger_config.get("collector_endpoint"),
            )
            jaeger_processor = BatchSpanProcessor(jaeger_exporter)
            self.tracer_provider.add_span_processor(jaeger_processor)
            self._span_processors.append(jaeger_processor)

    def _setup_metrics(self, resource: Resource) -> None:
        """Setup metrics configuration"""
        # Setup metric readers
        self._setup_metric_readers()

        self.meter_provider = MeterProvider(resource=resource, metric_readers=self._metric_readers)

        # Set global meter provider
        self.meter = self.meter_provider.get_meter(__name__)

        self.logger.info("Metrics configured successfully")

    def _setup_metric_readers(self) -> None:
        """Setup metric readers and exporters"""
        exporters_config = self.config.exporters.get("metrics", {})

        # Console exporter (development)
        if exporters_config.get("console", {}).get("enabled", False):
            console_reader = PeriodicExportingMetricReader(
                ConsoleMetricExporter(), export_interval_millis=self.config.metrics_export_interval * 1000
            )
            self._metric_readers.append(console_reader)

        # OTLP exporter
        otlp_config = exporters_config.get("otlp", {})
        if otlp_config.get("enabled", False):
            exporter_class = _optional_component(
                "opentelemetry.exporter.otlp.proto.grpc.metric_exporter", "OTLPMetricExporter"
            )
            otlp_reader = PeriodicExportingMetricReader(
                exporter_class(
                    endpoint=otlp_config.get("endpoint", "http://localhost:4317"),
                    headers=otlp_config.get("headers", {}),
                    timeout=otlp_config.get("timeout", 30),
                ),
                export_interval_millis=self.config.metrics_export_interval * 1000,
            )
            self._metric_readers.append(otlp_reader)

        # Prometheus exporter
        prometheus_config = exporters_config.get("prometheus", {})
        if prometheus_config.get("enabled", False):
            if "endpoint" in prometheus_config:
                raise ValueError("PrometheusMetricReader does not host HTTP; configure the HTTP server separately")
            reader_class = _optional_component("opentelemetry.exporter.prometheus", "PrometheusMetricReader")
            prometheus_reader = reader_class()
            self._metric_readers.append(prometheus_reader)

    def _setup_auto_instrumentation(self) -> None:
        """Instrument only configured libraries with explicitly available plugins."""
        for library, class_name in (
            ("requests", "RequestsInstrumentor"),
            ("logging", "LoggingInstrumentor"),
            ("sqlite3", "SQLite3Instrumentor"),
        ):
            config_name = "sqlite" if library == "sqlite3" else library
            if self.config.auto_instrumentation.get(config_name, {}).get("enabled", True):
                instrumentor = _optional_component(f"opentelemetry.instrumentation.{library}", class_name)()
                if not instrumentor.is_instrumented_by_opentelemetry:
                    instrumentor.instrument(tracer_provider=self.tracer_provider, meter_provider=self.meter_provider)
                    self._instrumentors.append(instrumentor)

    def _setup_propagators(self) -> None:
        """Configure all requested propagators, including W3C defaults."""
        propagators: List[TextMapPropagator] = []
        for name in self.config.config.get("propagators", ["tracecontext", "baggage"]):
            if name == "tracecontext":
                propagators.append(TraceContextTextMapPropagator())
            elif name == "baggage":
                propagators.append(W3CBaggagePropagator())
            elif name == "b3":
                propagators.append(_optional_component("opentelemetry.propagators.b3", "B3MultiFormat")())
            elif name == "jaeger":
                propagators.append(_optional_component("opentelemetry.propagators.jaeger", "JaegerPropagator")())
            else:
                raise ValueError(f"Unknown telemetry propagator: {name}")
        set_global_textmap(CompositeHTTPPropagator(propagators))

    def shutdown(self) -> None:
        """Release this instance's SDK providers, including partial initialization."""
        for instrumentor in reversed(self._instrumentors):
            instrumentor.uninstrument()
        self._instrumentors.clear()
        if self.tracer_provider is not None:
            self.tracer_provider.shutdown()
            self.tracer_provider = None
        if self.meter_provider is not None:
            self.meter_provider.shutdown()
            self.meter_provider = None
        else:
            for reader in self._metric_readers:
                reader.shutdown()
        self._span_processors.clear()
        self._metric_readers.clear()
        self.tracer = None
        self.meter = None
        self._initialized = False

    @contextmanager
    def trace_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[trace.Span]:
        """Create a trace span context manager"""
        if not self._initialized or self.tracer is None:
            raise RuntimeError("Telemetry must be initialized before tracing")

        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span

    def create_counter(self, name: str, description: str = "", unit: str = "1") -> metrics.Counter:
        """Create a metrics counter"""
        if not self._initialized or self.meter is None:
            raise RuntimeError("Telemetry must be initialized before creating metrics")

        return self.meter.create_counter(name=name, description=description, unit=unit)

    def create_histogram(self, name: str, description: str = "", unit: str = "1") -> metrics.Histogram:
        """Create a metrics histogram"""
        if not self._initialized or self.meter is None:
            raise RuntimeError("Telemetry must be initialized before creating metrics")

        return self.meter.create_histogram(name=name, description=description, unit=unit)

    def create_gauge(self, name: str, description: str = "", unit: str = "1") -> Gauge:
        """Create a metrics gauge"""
        if not self._initialized or self.meter is None:
            raise RuntimeError("Telemetry must be initialized before creating metrics")

        return self.meter.create_gauge(name=name, description=description, unit=unit)


class RAGTelemetryInstrumentation:
    """RAG-specific telemetry instrumentation"""

    def __init__(self, telemetry: TelemetryIntegration) -> None:
        self.telemetry = telemetry

        # Create RAG-specific metrics
        self.document_processing_counter = telemetry.create_counter(
            "rag_documents_processed_total", "Total number of documents processed", "documents"
        )

        self.document_processing_duration = telemetry.create_histogram(
            "rag_document_processing_duration_seconds", "Document processing duration", "seconds"
        )

        self.query_counter = telemetry.create_counter("rag_queries_total", "Total number of RAG queries", "queries")

        self.query_duration = telemetry.create_histogram("rag_query_duration_seconds", "RAG query duration", "seconds")

        self.retrieval_accuracy = telemetry.create_histogram(
            "rag_retrieval_accuracy", "RAG retrieval accuracy score", "score"
        )

        self.vector_store_operations = telemetry.create_counter(
            "rag_vector_store_operations_total", "Vector store operations", "operations"
        )

    def trace_document_processing(self, document_id: str) -> Any:
        """Trace document processing operation"""
        return self.telemetry.trace_span(
            "document_processing", attributes={"document.id": document_id, "operation.type": "document_processing"}
        )

    def trace_rag_query(self, query: str, user_id: Optional[str] = None) -> Any:
        """Trace RAG query operation"""
        attributes: Dict[str, Any] = {"query.length": len(query), "operation.type": "rag_query"}
        if user_id:
            attributes["user.id"] = user_id

        return self.telemetry.trace_span("rag_query", attributes)

    def trace_vector_search(self, query_embedding_size: int, top_k: int) -> Any:
        """Trace vector similarity search"""
        return self.telemetry.trace_span(
            "vector_search",
            attributes={
                "embedding.size": query_embedding_size,
                "search.top_k": top_k,
                "operation.type": "vector_search",
            },
        )

    def record_document_processed(self, processing_time: float, success: bool = True) -> None:
        """Record document processing metrics"""
        labels = {"status": "success" if success else "error"}
        self.document_processing_counter.add(1, labels)
        if success:
            self.document_processing_duration.record(processing_time, labels)

    def record_query_executed(self, query_time: float, retrieval_count: int, success: bool = True) -> None:
        """Record query execution metrics"""
        labels = {"status": "success" if success else "error"}
        self.query_counter.add(1, labels)
        if success:
            self.query_duration.record(query_time, labels)
            self.vector_store_operations.add(1, {"operation": "search"})


# Global telemetry instances
_telemetry_integration: Optional[TelemetryIntegration] = None
_rag_instrumentation: Optional[RAGTelemetryInstrumentation] = None


def initialize_telemetry(config: Optional[Dict[str, Any]] = None) -> TelemetryIntegration:
    """Initialize global telemetry integration"""
    global _telemetry_integration, _rag_instrumentation

    if _telemetry_integration is None:
        telemetry_config = TelemetryConfig(config)
        candidate = TelemetryIntegration(telemetry_config)
        candidate.initialize()
        instrumentation = RAGTelemetryInstrumentation(candidate)
        _telemetry_integration = candidate
        _rag_instrumentation = instrumentation

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
def trace_function(
    operation_name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to trace function execution"""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            telemetry = get_telemetry()
            with telemetry.trace_span(operation_name, attributes):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def trace_async_function(
    operation_name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """Decorator to trace async function execution"""

    def decorator(func: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            telemetry = get_telemetry()
            with telemetry.trace_span(operation_name, attributes):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def count_function_calls(metric_name: Optional[str] = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to count function calls"""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        nonlocal metric_name
        if metric_name is None:
            metric_name = f"function_calls_{func.__module__.replace('.', '_')}_{func.__name__}_total"

        telemetry = get_telemetry()
        counter = telemetry.create_counter(metric_name, f"Calls to {func.__name__}")

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            counter.add(1)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def time_function_execution(metric_name: Optional[str] = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to time function execution"""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        nonlocal metric_name
        if metric_name is None:
            metric_name = f"function_duration_{func.__module__.replace('.', '_')}_{func.__name__}_seconds"

        telemetry = get_telemetry()
        histogram = telemetry.create_histogram(metric_name, f"Duration of {func.__name__}", "seconds")

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                histogram.record(duration, {"status": "success"})
                return result
            except Exception:
                duration = time.perf_counter() - start_time
                histogram.record(duration, {"status": "error"})
                raise

        return wrapper

    return decorator
