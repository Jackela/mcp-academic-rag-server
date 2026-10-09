"""Actual SDK collection and explicit missing-plugin failures, without exporters on the network."""

from unittest.mock import patch

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import core.telemetry_integration as module
from core.telemetry_integration import RAGTelemetryInstrumentation, TelemetryConfig, TelemetryIntegration


def local_config(**values):
    return TelemetryConfig(
        {
            "service_name": "offline-sdk-contract",
            "auto_instrumentation": {name: {"enabled": False} for name in ("requests", "logging", "sqlite")},
            **values,
        }
    )


def test_actual_span_and_metric_collection(monkeypatch):
    exporter = InMemorySpanExporter()
    reader = InMemoryMetricReader()
    telemetry = TelemetryIntegration(local_config(tracing={"enabled": True, "sampling_ratio": 1.0}))

    def trace_exporters():
        telemetry.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    monkeypatch.setattr(telemetry, "_setup_trace_exporters", trace_exporters)
    monkeypatch.setattr(telemetry, "_setup_metric_readers", lambda: telemetry._metric_readers.append(reader))
    with patch("socket.socket.connect", side_effect=AssertionError("No external telemetry service")):
        telemetry.initialize()
        try:
            instrumentation = RAGTelemetryInstrumentation(telemetry)
            with instrumentation.trace_document_processing("document-42"):
                instrumentation.record_document_processed(0.25)
            gauge = telemetry.create_gauge("current_documents")
            gauge.set(3)
            spans = exporter.get_finished_spans()
            assert len(spans) == 1
            assert spans[0].name == "document_processing"
            assert spans[0].attributes["document.id"] == "document-42"
            assert spans[0].resource.attributes["service.name"] == "offline-sdk-contract"
            data = reader.get_metrics_data()
            actual = {
                metric.name: metric.data.data_points
                for resource in data.resource_metrics
                for scope in resource.scope_metrics
                for metric in scope.metrics
            }
            assert actual["rag_documents_processed_total"][0].value == 1
            assert actual["rag_document_processing_duration_seconds"][0].sum == 0.25
            assert actual["current_documents"][0].value == 3
            assert isinstance(telemetry.meter_provider, MeterProvider)
        finally:
            telemetry.shutdown()
        assert not telemetry._initialized
        with pytest.raises(RuntimeError, match="initialized"):
            telemetry.create_counter("after_shutdown")


def test_missing_core_sdk_does_not_initialize():
    telemetry = TelemetryIntegration(local_config())
    with patch.object(module, "OTEL_AVAILABLE", False), pytest.raises(ImportError, match="SDK unavailable"):
        telemetry.initialize()
    assert not telemetry._initialized
    assert telemetry.tracer is None and telemetry.meter is None


def test_missing_enabled_legacy_exporter_fails_and_global_remains_unset(monkeypatch):
    real_import = module.import_module

    def controlled_import(name):
        if name == "opentelemetry.exporter.jaeger.thrift":
            raise ImportError("Controlled absent legacy plugin")
        return real_import(name)

    monkeypatch.setattr(module, "import_module", controlled_import)
    monkeypatch.setattr(module, "_telemetry_integration", None)
    monkeypatch.setattr(module, "_rag_instrumentation", None)
    with pytest.raises(ImportError, match="jaeger.thrift.JaegerExporter"):
        module.initialize_telemetry(local_config(exporters={"tracing": {"jaeger": {"enabled": True}}}).config)
    assert module._telemetry_integration is None
    assert module._rag_instrumentation is None
    # An absent legacy exporter does not disable the installed core SDK.
    telemetry = TelemetryIntegration(local_config())
    telemetry.initialize()
    assert telemetry._initialized and telemetry.tracer is not None
    telemetry.shutdown()


def test_invalid_sampling_and_prometheus_endpoint_fail_explicitly():
    for config, message in [
        (local_config(tracing={"enabled": True, "sampling_ratio": 2}), "Probability"),
        (
            local_config(exporters={"metrics": {"prometheus": {"enabled": True, "endpoint": "localhost:9464"}}}),
            "does not host HTTP",
        ),
    ]:
        telemetry = TelemetryIntegration(config)
        with pytest.raises(ValueError, match=message):
            telemetry.initialize()
        assert not telemetry._initialized
        assert telemetry.tracer_provider is None and telemetry.meter_provider is None
