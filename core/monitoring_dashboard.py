"""
Monitoring Dashboard

Web-based monitoring dashboard for the MCP Academic RAG Server providing
real-time visualization of performance metrics, alerts, and system health.
"""

import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import weakref

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    WebSocket = None
    WebSocketDisconnect = None
    Request = None
    HTMLResponse = None
    JSONResponse = None

from core.performance_monitor import PerformanceMonitor, get_performance_monitor
from core.telemetry_integration import get_telemetry, get_rag_instrumentation


class DashboardConfig:
    """Configuration for monitoring dashboard"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        self.host = self.config.get('host', '127.0.0.1')
        self.port = self.config.get('port', 8080)
        self.debug = self.config.get('debug', False)
        
        # Authentication (basic implementation)
        self.auth_enabled = self.config.get('auth', {}).get('enabled', False)
        self.auth_token = self.config.get('auth', {}).get('token')
        
        # Dashboard features
        self.real_time_updates = self.config.get('real_time_updates', True)
        self.update_interval = self.config.get('update_interval', 5)  # seconds
        self.metrics_history_hours = self.config.get('metrics_history_hours', 24)
        
        # Visualization settings
        self.chart_points_limit = self.config.get('chart_points_limit', 100)
        self.refresh_rate_ms = self.config.get('refresh_rate_ms', 5000)


class WebSocketManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.logger = logging.getLogger("dashboard.websocket")
    
    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_to_all(self, data: Dict[str, Any]):
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
    
    async def broadcast_metrics(self, metrics: Dict[str, Any]):
        """Broadcast metrics update to all clients"""
        await self.send_to_all({
            'type': 'metrics_update',
            'data': metrics,
            'timestamp': datetime.now().isoformat()
        })
    
    async def broadcast_alert(self, alert: Dict[str, Any]):
        """Broadcast alert to all clients"""
        await self.send_to_all({
            'type': 'alert',
            'data': alert,
            'timestamp': datetime.now().isoformat()
        })


class MonitoringDashboard:
    """Main monitoring dashboard class"""
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.logger = logging.getLogger("dashboard.main")
        
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.websocket_manager = WebSocketManager()
        
        self.app: Optional[FastAPI] = None
        self.templates: Optional[Any] = None
        
        self._update_task: Optional[asyncio.Task] = None
        self._running = False
        
        if not FASTAPI_AVAILABLE:
            self.logger.error("FastAPI not available. Install with: pip install fastapi uvicorn jinja2")
    
    def initialize(self, performance_monitor: PerformanceMonitor = None):
        """Initialize dashboard with performance monitor"""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available for dashboard")
        
        self.performance_monitor = performance_monitor or get_performance_monitor()
        
        # Create FastAPI app
        self.app = FastAPI(
            title="MCP RAG Server Monitoring Dashboard",
            description="Real-time monitoring and metrics for MCP Academic RAG Server",
            version="1.0.0"
        )
        
        # Setup templates
        self._setup_templates()
        
        # Setup routes
        self._setup_routes()
        
        # Setup alert callbacks
        self._setup_alert_callbacks()
        
        self.logger.info("Dashboard initialized")
    
    def _setup_templates(self):
        """Setup Jinja2 templates"""
        template_dir = Path(__file__).parent / "templates"
        if not template_dir.exists():
            # Create basic template directory and files
            template_dir.mkdir(exist_ok=True)
            self._create_default_templates(template_dir)
        
        self.templates = Jinja2Templates(directory=str(template_dir))
    
    def _create_default_templates(self, template_dir: Path):
        """Create default HTML templates"""
        # Main dashboard template
        dashboard_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP RAG Server Monitoring</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .header { background: #2196F3; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #2196F3; }
        .metric-label { color: #666; margin-bottom: 10px; }
        .status-ok { color: #4CAF50; }
        .status-warning { color: #FF9800; }
        .status-error { color: #F44336; }
        .alert { padding: 10px; margin: 5px 0; border-radius: 4px; }
        .alert-warning { background: #FFF3CD; border: 1px solid #FFEAA7; }
        .alert-error { background: #F8D7DA; border: 1px solid #F5C6CB; }
        .chart-container { position: relative; height: 300px; }
        #status-indicator { width: 20px; height: 20px; border-radius: 50%; display: inline-block; margin-right: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>MCP Academic RAG Server Monitoring</h1>
        <div>
            <span id="status-indicator" class="status-ok"></span>
            <span id="connection-status">Connected</span>
            <span style="float: right;">Last Update: <span id="last-update">--</span></span>
        </div>
    </div>
    
    <div class="grid">
        <div class="card">
            <h3>System Overview</h3>
            <div class="metric-label">CPU Usage</div>
            <div class="metric-value" id="cpu-usage">--</div>
            <div class="metric-label">Memory Usage</div>
            <div class="metric-value" id="memory-usage">--</div>
        </div>
        
        <div class="card">
            <h3>RAG Performance</h3>
            <div class="metric-label">Documents Processed</div>
            <div class="metric-value" id="docs-processed">--</div>
            <div class="metric-label">Queries Handled</div>
            <div class="metric-value" id="queries-handled">--</div>
        </div>
        
        <div class="card">
            <h3>Active Alerts</h3>
            <div id="alerts-container">No active alerts</div>
        </div>
        
        <div class="card">
            <h3>System Metrics</h3>
            <div class="chart-container">
                <canvas id="system-chart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h3>Performance Metrics</h3>
            <div class="chart-container">
                <canvas id="performance-chart"></canvas>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);
        
        // Chart configurations
        const systemChart = new Chart(document.getElementById('system-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'CPU %',
                    data: [],
                    borderColor: '#FF6384',
                    tension: 0.1
                }, {
                    label: 'Memory %',
                    data: [],
                    borderColor: '#36A2EB',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
        
        const performanceChart = new Chart(document.getElementById('performance-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Response Time (ms)',
                    data: [],
                    borderColor: '#4BC0C0',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        
        // WebSocket event handlers
        ws.onopen = function() {
            document.getElementById('connection-status').textContent = 'Connected';
            document.getElementById('status-indicator').className = 'status-ok';
        };
        
        ws.onclose = function() {
            document.getElementById('connection-status').textContent = 'Disconnected';
            document.getElementById('status-indicator').className = 'status-error';
        };
        
        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            
            if (message.type === 'metrics_update') {
                updateMetrics(message.data);
                updateCharts(message.data);
            } else if (message.type === 'alert') {
                updateAlerts(message.data);
            }
            
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
        };
        
        function updateMetrics(data) {
            // Update system metrics
            if (data.system) {
                document.getElementById('cpu-usage').textContent = `${data.system.cpu_percent?.toFixed(1)}%`;
                document.getElementById('memory-usage').textContent = `${data.system.memory_percent?.toFixed(1)}%`;
            }
            
            // Update RAG metrics
            if (data.rag) {
                document.getElementById('docs-processed').textContent = data.rag.documents_processed || '--';
                document.getElementById('queries-handled').textContent = data.rag.queries_handled || '--';
            }
        }
        
        function updateCharts(data) {
            const now = new Date();
            const timeLabel = now.toLocaleTimeString();
            
            // Update system chart
            if (data.system) {
                systemChart.data.labels.push(timeLabel);
                systemChart.data.datasets[0].data.push(data.system.cpu_percent);
                systemChart.data.datasets[1].data.push(data.system.memory_percent);
                
                // Keep only last 20 points
                if (systemChart.data.labels.length > 20) {
                    systemChart.data.labels.shift();
                    systemChart.data.datasets[0].data.shift();
                    systemChart.data.datasets[1].data.shift();
                }
                
                systemChart.update('none');
            }
            
            // Update performance chart
            if (data.performance && data.performance.avg_response_time) {
                performanceChart.data.labels.push(timeLabel);
                performanceChart.data.datasets[0].data.push(data.performance.avg_response_time);
                
                if (performanceChart.data.labels.length > 20) {
                    performanceChart.data.labels.shift();
                    performanceChart.data.datasets[0].data.shift();
                }
                
                performanceChart.update('none');
            }
        }
        
        function updateAlerts(alerts) {
            const container = document.getElementById('alerts-container');
            
            if (!alerts || alerts.length === 0) {
                container.innerHTML = 'No active alerts';
                return;
            }
            
            container.innerHTML = alerts.map(alert => `
                <div class="alert alert-${alert.level}">
                    <strong>${alert.rule_name}</strong>: ${alert.message}
                    <br><small>Since: ${new Date(alert.triggered_at).toLocaleString()}</small>
                </div>
            `).join('');
        }
    </script>
</body>
</html>
        '''
        
        (template_dir / "dashboard.html").write_text(dashboard_html)
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Main dashboard page"""
            return self.templates.TemplateResponse("dashboard.html", {"request": request})
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """API endpoint for current metrics"""
            return JSONResponse(self._get_current_metrics())
        
        @self.app.get("/api/alerts")
        async def get_alerts():
            """API endpoint for current alerts"""
            alerts = self.performance_monitor.alert_manager.get_active_alerts()
            return JSONResponse([alert.to_dict() for alert in alerts])
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return JSONResponse({
                "status": "healthy" if self._running else "stopped",
                "timestamp": datetime.now().isoformat(),
                "performance_monitor": self.performance_monitor.is_running if self.performance_monitor else False,
                "websocket_connections": len(self.websocket_manager.active_connections)
            })
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
    
    def _setup_alert_callbacks(self):
        """Setup alert notification callbacks"""
        if self.performance_monitor:
            self.performance_monitor.alert_manager.add_alert_callback(self._handle_alert)
    
    def _handle_alert(self, alert):
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
        rag_metrics = {}
        try:
            rag_instrumentation = get_rag_instrumentation()
            # Add RAG-specific metric collection here
        except Exception as e:
            self.logger.debug(f"Error getting RAG metrics: {e}")
        
        return {
            "system": system_metrics,
            "aggregated": aggregated,
            "rag": rag_metrics,
            "performance": {
                "avg_response_time": aggregated.get("rag_query_duration_seconds", {}).get("avg", 0) * 1000
            }
        }
    
    async def start_real_time_updates(self):
        """Start real-time metrics updates"""
        if not self.config.real_time_updates:
            return
        
        self._update_task = asyncio.create_task(self._update_loop())
        self.logger.info("Real-time updates started")
    
    async def _update_loop(self):
        """Main update loop for real-time metrics"""
        while self._running:
            try:
                metrics = self._get_current_metrics()
                await self.websocket_manager.broadcast_metrics(metrics)
                await asyncio.sleep(self.config.update_interval)
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(self.config.update_interval)
    
    async def start(self):
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
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info" if self.config.debug else "warning"
        )
        server = uvicorn.Server(config)
        
        self.logger.info(f"Dashboard starting on http://{self.config.host}:{self.config.port}")
        await server.serve()
    
    def run(self):
        """Run dashboard in blocking mode"""
        if not FASTAPI_AVAILABLE:
            self.logger.error("Cannot run dashboard: FastAPI not available")
            return
        
        asyncio.run(self.start())
    
    def stop(self):
        """Stop the dashboard"""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
        
        self.logger.info("Dashboard stopped")


# Convenience functions
def create_dashboard(config: Dict[str, Any] = None, 
                    performance_monitor: PerformanceMonitor = None) -> MonitoringDashboard:
    """Create and initialize monitoring dashboard"""
    dashboard_config = DashboardConfig(config)
    dashboard = MonitoringDashboard(dashboard_config)
    dashboard.initialize(performance_monitor)
    return dashboard


def run_dashboard(config: Dict[str, Any] = None, 
                 performance_monitor: PerformanceMonitor = None):
    """Create and run monitoring dashboard"""
    dashboard = create_dashboard(config, performance_monitor)
    dashboard.run()