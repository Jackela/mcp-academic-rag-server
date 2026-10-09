"""
Monitoring Dashboard

Web-based monitoring dashboard for the MCP Academic RAG Server providing
real-time visualization of performance metrics, alerts, and system health.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import uvicorn
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.templating import Jinja2Templates

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from core.performance_monitor import Alert, PerformanceMonitor, get_performance_monitor
from core.telemetry_integration import get_rag_instrumentation


class DashboardConfig:
    """Configuration for monitoring dashboard"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

        self.host = self.config.get("host", "127.0.0.1")
        self.port = self.config.get("port", 8080)
        self.debug = self.config.get("debug", False)

        # Authentication (basic implementation)
        self.auth_enabled = self.config.get("auth", {}).get("enabled", False)
        self.auth_token = self.config.get("auth", {}).get("token")

        # Dashboard features
        self.real_time_updates = self.config.get("real_time_updates", True)
        self.update_interval = self.config.get("update_interval", 5)  # seconds
        self.metrics_history_hours = self.config.get("metrics_history_hours", 24)

        # Visualization settings
        self.chart_points_limit = self.config.get("chart_points_limit", 100)
        self.refresh_rate_ms = self.config.get("refresh_rate_ms", 5000)


class WebSocketManager:
    """Manage WebSocket connections for real-time updates"""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []
        self.logger = logging.getLogger("dashboard.websocket")

    async def connect(self, websocket: WebSocket) -> None:
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_to_all(self, data: Dict[str, Any]) -> None:
        """Send data to all connected WebSocket clients"""
        if not self.active_connections:
            return

        message = json.dumps(data, default=str)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                self.logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.append(connection)

        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_metrics(self, metrics: Dict[str, Any]) -> None:
        """Broadcast metrics update to all clients"""
        await self.send_to_all({"type": "metrics_update", "data": metrics, "timestamp": datetime.now().isoformat()})

    async def broadcast_alert(self, alert: Dict[str, Any]) -> None:
        """Broadcast alert to all clients"""
        await self.send_to_all({"type": "alert", "data": alert, "timestamp": datetime.now().isoformat()})


class MonitoringDashboard:
    """Main monitoring dashboard class"""

    def __init__(self, config: Optional[DashboardConfig] = None) -> None:
        self.config = config or DashboardConfig()
        self.logger = logging.getLogger("dashboard.main")

        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.websocket_manager = WebSocketManager()

        self.app: Optional[FastAPI] = None
        self.templates: Optional[Jinja2Templates] = None

        self._update_task: Optional[asyncio.Task[None]] = None
        self._running = False

        if not FASTAPI_AVAILABLE:
            self.logger.error("FastAPI not available. Install with: pip install fastapi uvicorn jinja2")

    def initialize(self, performance_monitor: Optional[PerformanceMonitor] = None) -> None:
        """Initialize dashboard with performance monitor"""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available for dashboard")

        self.performance_monitor = performance_monitor or get_performance_monitor()

        # Create FastAPI app
        self.app = FastAPI(
            title="MCP RAG Server Monitoring Dashboard",
            description="Real-time monitoring and metrics for MCP Academic RAG Server",
            version="1.0.0",
        )

        # Setup templates
        self._setup_templates()

        # Setup routes
        self._setup_routes()

        # Setup alert callbacks
        self._setup_alert_callbacks()

        self.logger.info("Dashboard initialized")

    def _setup_templates(self) -> None:
        """Setup Jinja2 templates"""
        template_dir = Path(__file__).parent / "templates"
        if not (template_dir / "dashboard.html").is_file():
            raise FileNotFoundError("Packaged monitoring template dashboard.html is missing")
        self.templates = Jinja2Templates(directory=str(template_dir))

    def _setup_routes(self) -> None:
        """Setup FastAPI routes"""

        if self.app is None or self.templates is None or self.performance_monitor is None:
            raise RuntimeError("Dashboard dependencies are not initialized")
        app, templates, monitor = self.app, self.templates, self.performance_monitor

        async def dashboard(request: Request) -> HTMLResponse:
            """Main dashboard page"""
            return templates.TemplateResponse(request=request, name="dashboard.html")

        app.get("/", response_class=HTMLResponse)(dashboard)

        async def get_metrics() -> JSONResponse:
            """API endpoint for current metrics"""
            return JSONResponse(self._get_current_metrics())

        app.get("/api/metrics")(get_metrics)

        async def get_alerts() -> JSONResponse:
            """API endpoint for current alerts"""
            alerts = monitor.alert_manager.get_active_alerts()
            return JSONResponse([alert.to_dict() for alert in alerts])

        app.get("/api/alerts")(get_alerts)

        async def health_check() -> JSONResponse:
            """Health check endpoint"""
            return JSONResponse(
                {
                    "status": "healthy" if self._running else "stopped",
                    "timestamp": datetime.now().isoformat(),
                    "performance_monitor": self.performance_monitor.is_running if self.performance_monitor else False,
                    "websocket_connections": len(self.websocket_manager.active_connections),
                }
            )

        app.get("/api/health")(health_check)

        async def websocket_endpoint(websocket: WebSocket) -> None:
            """WebSocket endpoint for real-time updates"""
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)

        app.websocket("/ws")(websocket_endpoint)

    def _setup_alert_callbacks(self) -> None:
        """Setup alert notification callbacks"""
        if self.performance_monitor:
            self.performance_monitor.alert_manager.add_alert_callback(self._handle_alert)

    def _handle_alert(self, alert: Alert) -> None:
        """Handle new alert notification"""
        asyncio.create_task(self.websocket_manager.broadcast_alert(alert.to_dict()))

    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics for dashboard"""
        if not self.performance_monitor:
            return {}

        # Get system metrics
        system_metrics = {}
        if self.performance_monitor.system_monitor.is_running:
            try:
                current_system = self.performance_monitor.system_monitor._collect_system_metrics()
                system_metrics = current_system.to_dict()
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")

        # Get aggregated metrics
        aggregated = self.performance_monitor.metrics_collector.get_aggregated_metrics()

        # Get RAG-specific metrics
        rag_metrics: Dict[str, Any] = {}
        try:
            get_rag_instrumentation()
            # Add RAG-specific metric collection here
        except Exception as e:
            self.logger.debug(f"Error getting RAG metrics: {e}")

        return {
            "system": system_metrics,
            "aggregated": aggregated,
            "rag": rag_metrics,
            "performance": {"avg_response_time": aggregated.get("rag_query_duration_seconds", {}).get("avg", 0) * 1000},
        }

    async def start_real_time_updates(self) -> None:
        """Start real-time metrics updates"""
        if not self.config.real_time_updates:
            return

        self._update_task = asyncio.create_task(self._update_loop())
        self.logger.info("Real-time updates started")

    async def _update_loop(self) -> None:
        """Main update loop for real-time metrics"""
        while self._running:
            try:
                metrics = self._get_current_metrics()
                await self.websocket_manager.broadcast_metrics(metrics)
                await asyncio.sleep(self.config.update_interval)
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(self.config.update_interval)

    async def start(self) -> None:
        """Start the dashboard server"""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available")

        if not self.app:
            raise RuntimeError("Dashboard not initialized")

        self._running = True

        # Start real-time updates
        await self.start_real_time_updates()

        # Start the FastAPI server
        config = uvicorn.Config(
            self.app, host=self.config.host, port=self.config.port, log_level="info" if self.config.debug else "warning"
        )
        server = uvicorn.Server(config)

        self.logger.info(f"Dashboard starting on http://{self.config.host}:{self.config.port}")
        await server.serve()

    def run(self) -> None:
        """Run dashboard in blocking mode"""
        if not FASTAPI_AVAILABLE:
            self.logger.error("Cannot run dashboard: FastAPI not available")
            return

        asyncio.run(self.start())

    def stop(self) -> None:
        """Stop the dashboard"""
        self._running = False

        if self._update_task:
            self._update_task.cancel()

        self.logger.info("Dashboard stopped")


# Convenience functions
def create_dashboard(
    config: Optional[Dict[str, Any]] = None, performance_monitor: Optional[PerformanceMonitor] = None
) -> MonitoringDashboard:
    """Create and initialize monitoring dashboard"""
    dashboard_config = DashboardConfig(config)
    dashboard = MonitoringDashboard(dashboard_config)
    dashboard.initialize(performance_monitor)
    return dashboard


def run_dashboard(
    config: Optional[Dict[str, Any]] = None, performance_monitor: Optional[PerformanceMonitor] = None
) -> None:
    """Create and run monitoring dashboard"""
    dashboard = create_dashboard(config, performance_monitor)
    dashboard.run()
