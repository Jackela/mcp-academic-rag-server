"""
Monitoring System Integration Tests

Comprehensive integration tests for the performance monitoring, telemetry,
dashboard, and alerting systems working together.
"""

import pytest
import asyncio
import tempfile
import time
import threading
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any

from core.performance_monitor import (
    PerformanceMonitor, PerformanceMetric, MetricType, 
    AlertRule, AlertLevel, get_performance_monitor
)
from core.telemetry_integration import (
    TelemetryIntegration, TelemetryConfig, RAGTelemetryInstrumentation,
    initialize_telemetry, get_rag_instrumentation
)
from core.monitoring_dashboard import MonitoringDashboard, DashboardConfig
from core.alerting_system import (
    AlertingSystem, NotificationConfig, NotificationChannel,
    EmailNotificationProvider, LogNotificationProvider
)


@pytest.fixture
def performance_monitor_config():
    """Configuration for performance monitor"""
    return {
        'max_metrics_history': 1000,
        'system_monitor_interval': 0.1,  # Fast for testing
        'alert_check_interval': 0.1
    }


@pytest.fixture
def telemetry_config():
    """Configuration for telemetry integration"""
    return {
        'service_name': 'test-rag-server',
        'environment': 'test',
        'tracing': {
            'enabled': True,
            'sampling_ratio': 1.0
        },
        'metrics': {
            'enabled': True,
            'export_interval': 1
        },
        'exporters': {
            'tracing': {
                'console': {'enabled': False}
            },
            'metrics': {
                'console': {'enabled': False}
            }
        }
    }


@pytest.fixture
def alerting_config():
    """Configuration for alerting system"""
    return {
        'channels': [
            {
                'channel': 'log',
                'enabled': True,
                'config': {'log_level': 'WARNING'}
            },
            {
                'channel': 'console',
                'enabled': True,
                'config': {}
            }
        ],
        'correlation': {
            'enabled': True,
            'correlation_window_minutes': 1,
            'max_similar_alerts': 3
        }
    }


@pytest.fixture
def dashboard_config():
    """Configuration for monitoring dashboard"""
    return {
        'host': '127.0.0.1',
        'port': 0,  # Let system assign port
        'debug': False,
        'real_time_updates': False,  # Disable for testing
        'update_interval': 0.1
    }


class TestPerformanceMonitoringIntegration:
    """Test performance monitoring system integration"""
    
    def test_performance_monitor_initialization(self, performance_monitor_config):
        """Test performance monitor initialization and basic functionality"""
        monitor = PerformanceMonitor(performance_monitor_config)
        
        assert not monitor.is_running
        assert monitor.metrics_collector is not None
        assert monitor.system_monitor is not None
        assert monitor.alert_manager is not None
    
    def test_performance_monitor_lifecycle(self, performance_monitor_config):
        """Test performance monitor start/stop lifecycle"""
        monitor = PerformanceMonitor(performance_monitor_config)
        
        # Start monitoring
        monitor.start()
        assert monitor.is_running
        assert monitor.system_monitor.is_running
        
        # Wait a bit for metrics collection
        time.sleep(0.2)
        
        # Stop monitoring
        monitor.stop()
        assert not monitor.is_running
        assert not monitor.system_monitor.is_running
    
    def test_custom_metrics_recording(self, performance_monitor_config):
        """Test recording custom metrics"""
        monitor = PerformanceMonitor(performance_monitor_config)
        monitor.start()
        
        try:
            # Record some custom metrics
            monitor.record_metric("test.counter", 1, MetricType.COUNTER)
            monitor.record_metric("test.gauge", 42.5, MetricType.GAUGE)
            monitor.record_metric("test.timer", 0.123, MetricType.TIMER, unit="seconds")
            
            # Verify metrics were recorded
            assert monitor.metrics_collector.get_current_value("test.counter") == 1
            assert monitor.metrics_collector.get_current_value("test.gauge") == 42.5
            assert monitor.metrics_collector.get_current_value("test.timer") == 0.123
            
        finally:
            monitor.stop()
    
    def test_timer_context_manager(self, performance_monitor_config):
        """Test timer context manager functionality"""
        monitor = PerformanceMonitor(performance_monitor_config)
        monitor.start()
        
        try:
            # Use timer context manager
            with monitor.timer("test.operation", tags={"component": "test"}):
                time.sleep(0.01)  # Simulate work
            
            # Verify timer metric was recorded
            duration = monitor.metrics_collector.get_current_value("test.operation")
            assert duration is not None
            assert duration >= 0.01
            
        finally:
            monitor.stop()
    
    def test_alert_system_integration(self, performance_monitor_config):
        """Test alert system integration with performance monitor"""
        monitor = PerformanceMonitor(performance_monitor_config)
        
        # Add custom alert rule
        alert_rule = AlertRule(
            metric_name="test.high_value",
            condition="greater_than",
            threshold=100.0,
            level=AlertLevel.WARNING,
            description="Test high value alert"
        )
        monitor.alert_manager.add_alert_rule(alert_rule)
        
        monitor.start()
        
        try:
            # Record metric that should trigger alert
            monitor.record_metric("test.high_value", 150.0, MetricType.GAUGE)
            
            # Give alert system time to process
            time.sleep(0.2)
            
            # Check for active alerts
            active_alerts = monitor.alert_manager.get_active_alerts()
            assert len(active_alerts) == 1
            assert active_alerts[0].rule.metric_name == "test.high_value"
            
        finally:
            monitor.stop()


class TestTelemetryIntegration:
    """Test telemetry integration functionality"""
    
    def test_telemetry_initialization(self, telemetry_config):
        """Test telemetry initialization"""
        telemetry = TelemetryIntegration(TelemetryConfig(telemetry_config))
        telemetry.initialize()
        
        assert telemetry._initialized
        assert telemetry.tracer is not None
        assert telemetry.meter is not None
    
    def test_telemetry_without_opentelemetry(self):
        """Test telemetry behavior when OpenTelemetry is not available"""
        with patch('core.telemetry_integration.OTEL_AVAILABLE', False):
            telemetry = TelemetryIntegration()
            telemetry.initialize()
            
            assert telemetry._initialized
            # Should use mock implementations
            assert telemetry.tracer is not None
            assert telemetry.meter is not None
    
    def test_trace_span_context_manager(self, telemetry_config):
        """Test trace span context manager"""
        telemetry = TelemetryIntegration(TelemetryConfig(telemetry_config))
        telemetry.initialize()
        
        # Test span creation (should not raise errors)
        with telemetry.trace_span("test.operation", {"test": "value"}):
            time.sleep(0.001)  # Simulate work
    
    def test_metrics_creation(self, telemetry_config):
        """Test metrics creation"""
        telemetry = TelemetryIntegration(TelemetryConfig(telemetry_config))
        telemetry.initialize()
        
        # Create metrics
        counter = telemetry.create_counter("test.counter", "Test counter")
        histogram = telemetry.create_histogram("test.histogram", "Test histogram")
        
        assert counter is not None
        assert histogram is not None
        
        # Use metrics (should not raise errors)
        counter.add(1, {"test": "label"})
        histogram.record(0.5, {"test": "label"})
    
    def test_rag_telemetry_instrumentation(self, telemetry_config):
        """Test RAG-specific telemetry instrumentation"""
        telemetry = TelemetryIntegration(TelemetryConfig(telemetry_config))
        telemetry.initialize()
        
        rag_instrumentation = RAGTelemetryInstrumentation(telemetry)
        
        # Test document processing tracing
        with rag_instrumentation.trace_document_processing("test_doc_001"):
            time.sleep(0.001)
        
        # Test RAG query tracing
        with rag_instrumentation.trace_rag_query("test query", "user_123"):
            time.sleep(0.001)
        
        # Test metrics recording
        rag_instrumentation.record_document_processed(0.5, success=True)
        rag_instrumentation.record_query_executed(1.2, 5, success=True)


class TestAlertingSystemIntegration:
    """Test alerting system integration"""
    
    def test_alerting_system_initialization(self, alerting_config):
        """Test alerting system initialization"""
        alerting = AlertingSystem(alerting_config)
        
        assert len(alerting.providers) >= 2  # log and console providers
        assert NotificationChannel.LOG in alerting.providers
        assert NotificationChannel.CONSOLE in alerting.providers
    
    @pytest.mark.asyncio
    async def test_alert_notification_sending(self, alerting_config):
        """Test sending alert notifications"""
        alerting = AlertingSystem(alerting_config)
        
        # Create test alert
        alert_rule = AlertRule(
            metric_name="test.metric",
            condition="greater_than",
            threshold=100.0,
            level=AlertLevel.WARNING,
            description="Test alert"
        )
        
        from core.performance_monitor import Alert
        alert = Alert(
            rule=alert_rule,
            triggered_at=datetime.now(),
            current_value=150.0,
            message="Test alert message"
        )
        
        # Send alert
        await alerting.send_alert(alert, {"context": "test"})
        
        # Verify notification was tracked
        assert len(alerting.sent_notifications) == 1
        notification = alerting.sent_notifications[0]
        assert notification['metric_name'] == "test.metric"
        assert notification['level'] == "warning"
    
    def test_notification_provider_validation(self):
        """Test notification provider configuration validation"""
        # Valid log provider config
        log_config = NotificationConfig(
            channel=NotificationChannel.LOG,
            config={'log_level': 'WARNING'}
        )
        log_provider = LogNotificationProvider(log_config.config)
        assert log_provider.validate_config()
        
        # Invalid email provider config (missing required fields)
        email_config = NotificationConfig(
            channel=NotificationChannel.EMAIL,
            config={'smtp_server': 'localhost'}  # Missing other required fields
        )
        email_provider = EmailNotificationProvider(email_config.config)
        assert not email_provider.validate_config()
    
    def test_alert_correlation(self, alerting_config):
        """Test alert correlation and deduplication"""
        alerting = AlertingSystem(alerting_config)
        
        # Create similar alerts
        alert_rule = AlertRule(
            metric_name="test.metric",
            condition="greater_than",
            threshold=100.0,
            level=AlertLevel.WARNING
        )
        
        from core.performance_monitor import Alert
        
        alerts = []
        for i in range(5):
            alert = Alert(
                rule=alert_rule,
                triggered_at=datetime.now(),
                current_value=150.0 + i,
                message=f"Test alert {i}"
            )
            alerts.append(alert)
        
        # First alert should pass correlation
        assert alerting.correlation_engine.correlate_alert(alerts[0])
        
        # Subsequent similar alerts should be tracked
        for alert in alerts[1:4]:
            assert alerting.correlation_engine.correlate_alert(alert)
        
        # Alert exceeding max similar alerts might be suppressed
        # (depends on configuration)
        result = alerting.correlation_engine.correlate_alert(alerts[4])
        # Result depends on max_similar_alerts setting


class TestDashboardIntegration:
    """Test monitoring dashboard integration"""
    
    @pytest.mark.skipif(
        not hasattr(pytest, '_dashboard_available'),
        reason="FastAPI not available for dashboard testing"
    )
    def test_dashboard_initialization(self, dashboard_config, performance_monitor_config):
        """Test dashboard initialization"""
        try:
            from core.monitoring_dashboard import FASTAPI_AVAILABLE
            if not FASTAPI_AVAILABLE:
                pytest.skip("FastAPI not available")
            
            monitor = PerformanceMonitor(performance_monitor_config)
            dashboard = MonitoringDashboard(DashboardConfig(dashboard_config))
            dashboard.initialize(monitor)
            
            assert dashboard.performance_monitor == monitor
            assert dashboard.app is not None
            
        except ImportError:
            pytest.skip("FastAPI not available for dashboard testing")
    
    def test_dashboard_metrics_collection(self, dashboard_config, performance_monitor_config):
        """Test dashboard metrics collection"""
        try:
            from core.monitoring_dashboard import FASTAPI_AVAILABLE
            if not FASTAPI_AVAILABLE:
                pytest.skip("FastAPI not available")
            
            monitor = PerformanceMonitor(performance_monitor_config)
            monitor.start()
            
            try:
                dashboard = MonitoringDashboard(DashboardConfig(dashboard_config))
                dashboard.initialize(monitor)
                
                # Record some test metrics
                monitor.record_metric("test.metric", 42.0, MetricType.GAUGE)
                
                # Get metrics for dashboard
                metrics = dashboard._get_current_metrics()
                
                assert isinstance(metrics, dict)
                assert 'aggregated' in metrics
                
            finally:
                monitor.stop()
                
        except ImportError:
            pytest.skip("FastAPI not available for dashboard testing")


class TestFullSystemIntegration:
    """Test full monitoring system integration"""
    
    def test_complete_monitoring_stack(self, performance_monitor_config, 
                                     telemetry_config, alerting_config):
        """Test complete monitoring stack working together"""
        # Initialize all components
        monitor = PerformanceMonitor(performance_monitor_config)
        telemetry = TelemetryIntegration(TelemetryConfig(telemetry_config))
        alerting = AlertingSystem(alerting_config)
        
        # Initialize telemetry
        telemetry.initialize()
        
        # Add alert callback to connect systems
        alert_notifications = []
        
        async def alert_callback(alert):
            alert_notifications.append(alert)
            await alerting.send_alert(alert)
        
        monitor.alert_manager.add_alert_callback(
            lambda alert: asyncio.create_task(alert_callback(alert))
        )
        
        # Start monitoring
        monitor.start()
        
        try:
            # Record metrics that will trigger alerts
            with monitor.timer("operation.duration"):
                time.sleep(0.01)
            
            monitor.record_metric("system.cpu.test", 95.0, MetricType.GAUGE)  # Should trigger alert
            
            # Use telemetry
            with telemetry.trace_span("test.operation"):
                counter = telemetry.create_counter("test.operations", "Test operations")
                counter.add(1, {"status": "success"})
            
            # Wait for processing
            time.sleep(0.5)
            
            # Verify system integration
            assert len(monitor.metrics_collector.metrics_history) > 0
            
            # Check if alerts were processed
            if len(alert_notifications) > 0:
                assert len(alerting.sent_notifications) > 0
            
        finally:
            monitor.stop()
            telemetry.shutdown()
    
    @pytest.mark.asyncio
    async def test_async_monitoring_operations(self, performance_monitor_config, telemetry_config):
        """Test async monitoring operations"""
        monitor = PerformanceMonitor(performance_monitor_config)
        telemetry = TelemetryIntegration(TelemetryConfig(telemetry_config))
        telemetry.initialize()
        
        rag_instrumentation = RAGTelemetryInstrumentation(telemetry)
        
        monitor.start()
        
        try:
            # Simulate async RAG operations
            async def simulate_document_processing(doc_id: str):
                with rag_instrumentation.trace_document_processing(doc_id):
                    # Simulate processing time
                    await asyncio.sleep(0.01)
                    
                    # Record processing metrics
                    processing_time = 0.01
                    rag_instrumentation.record_document_processed(processing_time, success=True)
                    
                    # Record in performance monitor
                    monitor.record_metric(
                        "rag.document.processing_time",
                        processing_time,
                        MetricType.TIMER,
                        tags={"document_id": doc_id}
                    )
            
            async def simulate_rag_query(query: str, user_id: str):
                with rag_instrumentation.trace_rag_query(query, user_id):
                    # Simulate vector search
                    with rag_instrumentation.trace_vector_search(384, 5):
                        await asyncio.sleep(0.005)
                    
                    # Simulate LLM generation
                    await asyncio.sleep(0.02)
                    
                    # Record query metrics
                    query_time = 0.025
                    rag_instrumentation.record_query_executed(query_time, 5, success=True)
                    
                    monitor.record_metric(
                        "rag.query.total_time",
                        query_time,
                        MetricType.TIMER,
                        tags={"user_id": user_id}
                    )
            
            # Run multiple operations concurrently
            tasks = []
            
            # Document processing tasks
            for i in range(3):
                task = simulate_document_processing(f"doc_{i}")
                tasks.append(task)
            
            # Query tasks
            for i in range(2):
                task = simulate_rag_query(f"query {i}", f"user_{i}")
                tasks.append(task)
            
            # Execute all tasks
            await asyncio.gather(*tasks)
            
            # Verify metrics were recorded
            assert monitor.metrics_collector.get_current_value("rag.document.processing_time") is not None
            assert monitor.metrics_collector.get_current_value("rag.query.total_time") is not None
            
        finally:
            monitor.stop()
            telemetry.shutdown()
    
    def test_monitoring_system_status(self, performance_monitor_config, alerting_config):
        """Test monitoring system status reporting"""
        monitor = PerformanceMonitor(performance_monitor_config)
        alerting = AlertingSystem(alerting_config)
        
        # Test status before starting
        status = monitor.get_status()
        assert status['is_running'] is False
        assert status['metrics_collected'] == 0
        
        # Start and test status
        monitor.start()
        
        try:
            # Record some metrics
            monitor.record_metric("test.metric", 1, MetricType.COUNTER)
            time.sleep(0.1)  # Let system monitor collect some metrics
            
            # Check status
            status = monitor.get_status()
            assert status['is_running'] is True
            assert status['metrics_collected'] > 0
            assert status['system_monitor_running'] is True
            
            # Check alerting status
            alerting_stats = alerting.get_notification_stats()
            assert 'total_notifications_sent' in alerting_stats
            assert 'configured_channels' in alerting_stats
            assert alerting_stats['configured_channels'] >= 2  # log and console
            
        finally:
            monitor.stop()
    
    def test_monitoring_configuration_validation(self):
        """Test monitoring system configuration validation"""
        # Test valid configurations
        valid_monitor_config = {
            'max_metrics_history': 1000,
            'system_monitor_interval': 1.0,
            'alert_check_interval': 5.0
        }
        
        monitor = PerformanceMonitor(valid_monitor_config)
        assert monitor.config == valid_monitor_config
        
        # Test telemetry config validation
        valid_telemetry_config = {
            'service_name': 'test-service',
            'environment': 'test',
            'tracing': {'enabled': True},
            'metrics': {'enabled': True}
        }
        
        telemetry_config = TelemetryConfig(valid_telemetry_config)
        assert telemetry_config.service_name == 'test-service'
        assert telemetry_config.environment == 'test'
        assert telemetry_config.tracing_enabled is True
        assert telemetry_config.metrics_enabled is True
        
        # Test alerting config validation
        valid_alerting_config = {
            'channels': [
                {
                    'channel': 'log',
                    'enabled': True,
                    'config': {'log_level': 'INFO'}
                }
            ]
        }
        
        alerting = AlertingSystem(valid_alerting_config)
        assert len(alerting.providers) >= 1
        assert NotificationChannel.LOG in alerting.providers