"""
Alerting and Notification System

Comprehensive alerting system for the MCP Academic RAG Server providing
multi-channel notifications, escalation policies, and alert correlation.
"""

import asyncio
import json
import logging
import smtplib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Set
from urllib.parse import urljoin
import threading

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

from core.performance_monitor import Alert, AlertLevel, AlertRule


class NotificationChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"
    LOG = "log"
    CONSOLE = "console"


class EscalationLevel(Enum):
    """Alert escalation levels"""
    IMMEDIATE = "immediate"
    AFTER_5_MIN = "after_5_min"
    AFTER_15_MIN = "after_15_min"
    AFTER_1_HOUR = "after_1_hour"


@dataclass
class NotificationConfig:
    """Configuration for a notification channel"""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}
        
        if isinstance(self.channel, str):
            self.channel = NotificationChannel(self.channel)


@dataclass
class EscalationPolicy:
    """Alert escalation policy configuration"""
    name: str
    levels: List[Dict[str, Any]]  # List of escalation level configs
    enabled: bool = True
    
    def get_escalation_for_duration(self, duration_minutes: int) -> Optional[Dict[str, Any]]:
        """Get escalation configuration for alert duration"""
        for level_config in self.levels:
            trigger_minutes = level_config.get('trigger_after_minutes', 0)
            if duration_minutes >= trigger_minutes:
                return level_config
        return None


@dataclass
class AlertCorrelation:
    """Alert correlation configuration"""
    correlation_window_minutes: int = 5
    max_similar_alerts: int = 10
    correlation_fields: List[str] = None
    
    def __post_init__(self):
        if self.correlation_fields is None:
            self.correlation_fields = ['metric_name', 'level']


class NotificationProvider(ABC):
    """Abstract base class for notification providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"alerting.{self.__class__.__name__.lower()}")
    
    @abstractmethod
    async def send_notification(self, alert: Alert, context: Dict[str, Any] = None) -> bool:
        """Send notification for an alert"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate provider configuration"""
        pass


class EmailNotificationProvider(NotificationProvider):
    """Email notification provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smtp_server = config.get('smtp_server')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.use_tls = config.get('use_tls', True)
        self.from_email = config.get('from_email')
        self.to_emails = config.get('to_emails', [])
    
    def validate_config(self) -> bool:
        """Validate email configuration"""
        required_fields = ['smtp_server', 'username', 'password', 'from_email', 'to_emails']
        return all(self.config.get(field) for field in required_fields)
    
    async def send_notification(self, alert: Alert, context: Dict[str, Any] = None) -> bool:
        """Send email notification"""
        try:
            subject = f"[{alert.rule.level.value.upper()}] Alert: {alert.rule.metric_name}"
            body = self._create_email_body(alert, context)
            
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'html'))
            
            # Send email in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_email, msg)
            
            self.logger.info(f"Email alert sent for {alert.rule.metric_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _send_email(self, msg: MimeMultipart):
        """Send email using SMTP"""
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
    
    def _create_email_body(self, alert: Alert, context: Dict[str, Any] = None) -> str:
        """Create HTML email body"""
        context = context or {}
        
        color = {
            AlertLevel.INFO: "#17a2b8",
            AlertLevel.WARNING: "#ffc107", 
            AlertLevel.ERROR: "#dc3545",
            AlertLevel.CRITICAL: "#6f42c1"
        }.get(alert.rule.level, "#6c757d")
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px;">
            <div style="border-left: 4px solid {color}; padding-left: 20px; margin-bottom: 20px;">
                <h2 style="color: {color}; margin: 0;">
                    {alert.rule.level.value.upper()} Alert
                </h2>
                <p style="margin: 5px 0; font-size: 18px; font-weight: bold;">
                    {alert.rule.metric_name}
                </p>
            </div>
            
            <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <h3 style="margin-top: 0;">Alert Details</h3>
                <p><strong>Message:</strong> {alert.message}</p>
                <p><strong>Current Value:</strong> {alert.current_value}</p>
                <p><strong>Threshold:</strong> {alert.rule.threshold}</p>
                <p><strong>Condition:</strong> {alert.rule.condition}</p>
                <p><strong>Triggered At:</strong> {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                {f'<p><strong>Duration:</strong> {(datetime.now() - alert.triggered_at).total_seconds() / 60:.1f} minutes</p>' if alert.is_active else ''}
            </div>
            
            {self._create_context_section(context) if context else ''}
            
            <div style="margin-top: 30px; padding-top: 15px; border-top: 1px solid #dee2e6; color: #6c757d; font-size: 12px;">
                <p>This alert was generated by the MCP Academic RAG Server monitoring system.</p>
                <p>Alert ID: {id(alert)}</p>
            </div>
        </body>
        </html>
        """
    
    def _create_context_section(self, context: Dict[str, Any]) -> str:
        """Create context section for email"""
        if not context:
            return ""
        
        items = []
        for key, value in context.items():
            items.append(f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>")
        
        return f"""
        <div style="background: #e9ecef; padding: 15px; border-radius: 5px;">
            <h3 style="margin-top: 0;">Additional Context</h3>
            {''.join(items)}
        </div>
        """


class WebhookNotificationProvider(NotificationProvider):
    """Webhook notification provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.url = config.get('url')
        self.method = config.get('method', 'POST')
        self.headers = config.get('headers', {})
        self.timeout = config.get('timeout', 30)
        self.verify_ssl = config.get('verify_ssl', True)
    
    def validate_config(self) -> bool:
        """Validate webhook configuration"""
        return bool(self.url and REQUESTS_AVAILABLE)
    
    async def send_notification(self, alert: Alert, context: Dict[str, Any] = None) -> bool:
        """Send webhook notification"""
        if not REQUESTS_AVAILABLE:
            self.logger.error("Requests library not available for webhook notifications")
            return False
        
        try:
            payload = self._create_webhook_payload(alert, context)
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._send_webhook,
                payload
            )
            
            if 200 <= response.status_code < 300:
                self.logger.info(f"Webhook alert sent for {alert.rule.metric_name}")
                return True
            else:
                self.logger.error(f"Webhook returned status {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    def _send_webhook(self, payload: Dict[str, Any]) -> Any:
        """Send webhook request"""
        return requests.request(
            method=self.method,
            url=self.url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
            verify=self.verify_ssl
        )
    
    def _create_webhook_payload(self, alert: Alert, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create webhook payload"""
        return {
            "alert": {
                "metric_name": alert.rule.metric_name,
                "level": alert.rule.level.value,
                "message": alert.message,
                "current_value": alert.current_value,
                "threshold": alert.rule.threshold,
                "condition": alert.rule.condition,
                "triggered_at": alert.triggered_at.isoformat(),
                "is_active": alert.is_active,
                "description": alert.rule.description
            },
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
            "source": "mcp-academic-rag-server"
        }


class SlackNotificationProvider(WebhookNotificationProvider):
    """Slack notification provider using webhooks"""
    
    def _create_webhook_payload(self, alert: Alert, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create Slack-formatted payload"""
        color = {
            AlertLevel.INFO: "#36a64f",
            AlertLevel.WARNING: "#ff9500", 
            AlertLevel.ERROR: "#ff0000",
            AlertLevel.CRITICAL: "#800080"
        }.get(alert.rule.level, "#cccccc")
        
        fields = [
            {
                "title": "Current Value",
                "value": str(alert.current_value),
                "short": True
            },
            {
                "title": "Threshold",
                "value": f"{alert.rule.condition} {alert.rule.threshold}",
                "short": True
            },
            {
                "title": "Duration",
                "value": f"{(datetime.now() - alert.triggered_at).total_seconds() / 60:.1f} minutes",
                "short": True
            }
        ]
        
        # Add context fields
        if context:
            for key, value in context.items():
                fields.append({
                    "title": key.replace('_', ' ').title(),
                    "value": str(value),
                    "short": True
                })
        
        return {
            "attachments": [
                {
                    "color": color,
                    "title": f"{alert.rule.level.value.upper()} Alert: {alert.rule.metric_name}",
                    "text": alert.message,
                    "fields": fields,
                    "footer": "MCP RAG Server",
                    "ts": int(alert.triggered_at.timestamp())
                }
            ]
        }


class LogNotificationProvider(NotificationProvider):
    """Log-based notification provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.log_level = config.get('log_level', 'WARNING')
        self.alert_logger = logging.getLogger('alerting.notifications')
        
        # Set log level
        level = getattr(logging, self.log_level.upper(), logging.WARNING)
        self.alert_logger.setLevel(level)
    
    def validate_config(self) -> bool:
        """Log provider always valid"""
        return True
    
    async def send_notification(self, alert: Alert, context: Dict[str, Any] = None) -> bool:
        """Log alert notification"""
        try:
            log_message = f"ALERT [{alert.rule.level.value.upper()}] {alert.rule.metric_name}: {alert.message}"
            
            if alert.rule.level == AlertLevel.CRITICAL:
                self.alert_logger.critical(log_message)
            elif alert.rule.level == AlertLevel.ERROR:
                self.alert_logger.error(log_message)
            elif alert.rule.level == AlertLevel.WARNING:
                self.alert_logger.warning(log_message)
            else:
                self.alert_logger.info(log_message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to log alert: {e}")
            return False


class ConsoleNotificationProvider(NotificationProvider):
    """Console notification provider"""
    
    def validate_config(self) -> bool:
        """Console provider always valid"""
        return True
    
    async def send_notification(self, alert: Alert, context: Dict[str, Any] = None) -> bool:
        """Print alert to console"""
        try:
            timestamp = alert.triggered_at.strftime('%H:%M:%S')
            level_colors = {
                AlertLevel.INFO: '\033[36m',      # Cyan
                AlertLevel.WARNING: '\033[33m',   # Yellow
                AlertLevel.ERROR: '\033[31m',     # Red
                AlertLevel.CRITICAL: '\033[35m'   # Magenta
            }
            
            color = level_colors.get(alert.rule.level, '\033[0m')
            reset = '\033[0m'
            
            print(f"{color}[{timestamp}] ALERT [{alert.rule.level.value.upper()}] {alert.rule.metric_name}: {alert.message}{reset}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to print alert to console: {e}")
            return False


class AlertCorrelationEngine:
    """Engine for correlating and deduplicating alerts"""
    
    def __init__(self, config: AlertCorrelation):
        self.config = config
        self.recent_alerts: List[Alert] = []
        self.correlated_alerts: Dict[str, List[Alert]] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger("alerting.correlation")
    
    def correlate_alert(self, alert: Alert) -> bool:
        """Check if alert should be correlated (deduplicated)"""
        with self._lock:
            # Clean old alerts
            self._clean_old_alerts()
            
            # Find similar alerts
            correlation_key = self._get_correlation_key(alert)
            similar_alerts = self._find_similar_alerts(alert)
            
            if similar_alerts:
                # Correlate with existing alerts
                if correlation_key not in self.correlated_alerts:
                    self.correlated_alerts[correlation_key] = []
                
                self.correlated_alerts[correlation_key].append(alert)
                
                # Check if we should suppress this alert
                if len(self.correlated_alerts[correlation_key]) > self.config.max_similar_alerts:
                    self.logger.info(f"Suppressing correlated alert for {alert.rule.metric_name}")
                    return False
            
            # Add to recent alerts
            self.recent_alerts.append(alert)
            return True
    
    def _get_correlation_key(self, alert: Alert) -> str:
        """Generate correlation key for alert"""
        key_parts = []
        for field in self.config.correlation_fields:
            if field == 'metric_name':
                key_parts.append(alert.rule.metric_name)
            elif field == 'level':
                key_parts.append(alert.rule.level.value)
            elif field == 'condition':
                key_parts.append(alert.rule.condition)
        
        return '|'.join(key_parts)
    
    def _find_similar_alerts(self, alert: Alert) -> List[Alert]:
        """Find similar alerts within correlation window"""
        cutoff_time = datetime.now() - timedelta(minutes=self.config.correlation_window_minutes)
        correlation_key = self._get_correlation_key(alert)
        
        return [
            a for a in self.recent_alerts
            if (a.triggered_at > cutoff_time and 
                self._get_correlation_key(a) == correlation_key)
        ]
    
    def _clean_old_alerts(self):
        """Remove old alerts from tracking"""
        cutoff_time = datetime.now() - timedelta(minutes=self.config.correlation_window_minutes * 2)
        
        # Clean recent alerts
        self.recent_alerts = [
            alert for alert in self.recent_alerts
            if alert.triggered_at > cutoff_time
        ]
        
        # Clean correlated alerts
        for key in list(self.correlated_alerts.keys()):
            self.correlated_alerts[key] = [
                alert for alert in self.correlated_alerts[key]
                if alert.triggered_at > cutoff_time
            ]
            
            if not self.correlated_alerts[key]:
                del self.correlated_alerts[key]


class AlertingSystem:
    """Main alerting system coordinator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("alerting.system")
        
        # Notification providers
        self.providers: Dict[NotificationChannel, NotificationProvider] = {}
        self.notification_configs: List[NotificationConfig] = []
        
        # Escalation and correlation
        self.escalation_policies: Dict[str, EscalationPolicy] = {}
        self.correlation_engine: Optional[AlertCorrelationEngine] = None
        
        # Tracking
        self.sent_notifications: List[Dict[str, Any]] = []
        self.escalated_alerts: Dict[str, datetime] = {}
        
        # Background tasks
        self.escalation_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Initialize from config
        self._initialize_from_config()
    
    def _initialize_from_config(self):
        """Initialize alerting system from configuration"""
        # Setup notification channels
        channels_config = self.config.get('channels', [])
        for channel_config in channels_config:
            notification_config = NotificationConfig(**channel_config)
            self.notification_configs.append(notification_config)
            self._setup_provider(notification_config)
        
        # Setup escalation policies
        policies_config = self.config.get('escalation_policies', {})
        for name, policy_config in policies_config.items():
            self.escalation_policies[name] = EscalationPolicy(name=name, **policy_config)
        
        # Setup correlation
        correlation_config = self.config.get('correlation', {})
        if correlation_config.get('enabled', True):
            self.correlation_engine = AlertCorrelationEngine(AlertCorrelation(**correlation_config))
    
    def _setup_provider(self, config: NotificationConfig):
        """Setup notification provider"""
        if not config.enabled:
            return
        
        provider_classes = {
            NotificationChannel.EMAIL: EmailNotificationProvider,
            NotificationChannel.WEBHOOK: WebhookNotificationProvider,
            NotificationChannel.SLACK: SlackNotificationProvider,
            NotificationChannel.LOG: LogNotificationProvider,
            NotificationChannel.CONSOLE: ConsoleNotificationProvider
        }
        
        provider_class = provider_classes.get(config.channel)
        if not provider_class:
            self.logger.warning(f"Unknown notification channel: {config.channel}")
            return
        
        try:
            provider = provider_class(config.config)
            if provider.validate_config():
                self.providers[config.channel] = provider
                self.logger.info(f"Configured {config.channel.value} notification provider")
            else:
                self.logger.error(f"Invalid configuration for {config.channel.value} provider")
        except Exception as e:
            self.logger.error(f"Failed to setup {config.channel.value} provider: {e}")
    
    async def send_alert(self, alert: Alert, context: Dict[str, Any] = None):
        """Send alert through configured notification channels"""
        # Check correlation
        if self.correlation_engine and not self.correlation_engine.correlate_alert(alert):
            self.logger.debug(f"Alert suppressed due to correlation: {alert.rule.metric_name}")
            return
        
        # Send through all enabled providers
        notification_tasks = []
        for provider in self.providers.values():
            task = asyncio.create_task(
                self._send_notification_safe(provider, alert, context)
            )
            notification_tasks.append(task)
        
        if notification_tasks:
            results = await asyncio.gather(*notification_tasks, return_exceptions=True)
            
            # Log results
            success_count = sum(1 for r in results if r is True)
            self.logger.info(f"Alert sent through {success_count}/{len(results)} channels")
            
            # Track notification
            self.sent_notifications.append({
                'alert_id': id(alert),
                'metric_name': alert.rule.metric_name,
                'level': alert.rule.level.value,
                'timestamp': datetime.now().isoformat(),
                'channels_sent': success_count,
                'total_channels': len(results)
            })
    
    async def _send_notification_safe(self, provider: NotificationProvider, 
                                    alert: Alert, context: Dict[str, Any] = None) -> bool:
        """Send notification with error handling"""
        try:
            return await provider.send_notification(alert, context)
        except Exception as e:
            self.logger.error(f"Error in {provider.__class__.__name__}: {e}")
            return False
    
    def start_escalation_monitoring(self):
        """Start escalation monitoring task"""
        if self._running:
            return
        
        self._running = True
        self.escalation_task = asyncio.create_task(self._escalation_loop())
        self.logger.info("Escalation monitoring started")
    
    def stop_escalation_monitoring(self):
        """Stop escalation monitoring"""
        self._running = False
        
        if self.escalation_task:
            self.escalation_task.cancel()
        
        self.logger.info("Escalation monitoring stopped")
    
    async def _escalation_loop(self):
        """Main escalation monitoring loop"""
        while self._running:
            try:
                await self._check_escalations()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in escalation loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_escalations(self):
        """Check for alerts that need escalation"""
        # Implementation would check active alerts and escalate based on policies
        # This is a placeholder for the escalation logic
        pass
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        total_sent = len(self.sent_notifications)
        
        # Count by channel
        channel_stats = {}
        for provider_channel in self.providers.keys():
            channel_stats[provider_channel.value] = {
                'enabled': True,
                'provider_class': self.providers[provider_channel].__class__.__name__
            }
        
        # Recent notifications
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_notifications = [
            notif for notif in self.sent_notifications
            if datetime.fromisoformat(notif['timestamp']) > recent_cutoff
        ]
        
        return {
            'total_notifications_sent': total_sent,
            'recent_24h': len(recent_notifications),
            'configured_channels': len(self.providers),
            'escalation_policies': len(self.escalation_policies),
            'channel_stats': channel_stats,
            'correlation_enabled': self.correlation_engine is not None
        }