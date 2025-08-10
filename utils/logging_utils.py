"""
Enhanced logging utilities for the Academic RAG Server

This module provides structured logging capabilities with support for:
- JSON formatted logs
- Context injection
- Performance tracking
- Error aggregation
- Log rotation
"""

import logging
import sys
import os
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from contextlib import contextmanager
import time
import threading
from functools import wraps

try:
    from loguru import logger as loguru_logger
    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter"""
    
    def __init__(self, service_name: str = "academic-rag", include_context: bool = True):
        super().__init__()
        self.service_name = service_name
        self.include_context = include_context
        self.hostname = os.environ.get('HOSTNAME', 'localhost')
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'service': self.service_name,
            'hostname': self.hostname,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
            
        # Add custom context if available
        if hasattr(record, 'context') and self.include_context:
            log_data['context'] = record.context
            
        # Add performance metrics if available
        if hasattr(record, 'duration'):
            log_data['performance'] = {
                'duration_ms': record.duration
            }
            
        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName', 
                          'levelname', 'levelno', 'lineno', 'module', 'msecs', 
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                          'context', 'duration']:
                log_data[key] = value
                
        return json.dumps(log_data, ensure_ascii=False)


class ContextLogger:
    """Logger with context injection capabilities"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._context = threading.local()
        
    def set_context(self, **kwargs):
        """Set context values for current thread"""
        if not hasattr(self._context, 'data'):
            self._context.data = {}
        self._context.data.update(kwargs)
        
    def clear_context(self):
        """Clear context for current thread"""
        if hasattr(self._context, 'data'):
            self._context.data = {}
            
    def get_context(self) -> Dict[str, Any]:
        """Get current context"""
        return getattr(self._context, 'data', {})
        
    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Log with context injection"""
        extra = kwargs.get('extra', {})
        extra['context'] = self.get_context()
        kwargs['extra'] = extra
        self.logger.log(level, msg, *args, **kwargs)
        
    def debug(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
        
    def info(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
        
    def error(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
        
    def critical(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)
        
    @contextmanager
    def context(self, **kwargs):
        """Context manager for temporary context"""
        old_context = self.get_context().copy()
        self.set_context(**kwargs)
        try:
            yield
        finally:
            self._context.data = old_context


class LoggingConfig:
    """Centralized logging configuration"""
    
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    @staticmethod
    def setup_logging(
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        log_format: Optional[str] = None,
        use_json: bool = False,
        service_name: str = "academic-rag",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_console: bool = True,
        enable_file: bool = True
    ) -> None:
        """
        Setup logging configuration
        
        Args:
            log_level: Logging level
            log_file: Path to log file
            log_format: Log format string
            use_json: Use JSON structured logging
            service_name: Service name for structured logs
            max_bytes: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
            enable_console: Enable console logging
            enable_file: Enable file logging
        """
        # Remove existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # Set log level
        level = getattr(logging, log_level.upper(), logging.INFO)
        root_logger.setLevel(level)
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            if use_json:
                console_formatter = StructuredFormatter(service_name)
            else:
                console_formatter = logging.Formatter(
                    log_format or LoggingConfig.DEFAULT_FORMAT,
                    datefmt=LoggingConfig.DEFAULT_DATE_FORMAT
                )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if enable_file and log_file:
            # Create log directory if needed
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                
            # Use rotating file handler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            
            if use_json:
                file_formatter = StructuredFormatter(service_name)
            else:
                file_formatter = logging.Formatter(
                    log_format or LoggingConfig.DEFAULT_FORMAT,
                    datefmt=LoggingConfig.DEFAULT_DATE_FORMAT
                )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            
        # Configure third-party loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('haystack').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)
        
    @staticmethod
    def get_logger(name: str, with_context: bool = False) -> Union[logging.Logger, ContextLogger]:
        """
        Get a logger instance
        
        Args:
            name: Logger name
            with_context: Return ContextLogger for context injection
            
        Returns:
            Logger instance
        """
        logger = logging.getLogger(name)
        if with_context:
            return ContextLogger(logger)
        return logger


def log_performance(func):
    """Decorator to log function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000  # Convert to ms
            
            logger.info(
                f"Function {func.__name__} completed",
                extra={
                    'duration': duration,
                    'function': func.__name__,
                    'module': func.__module__
                }
            )
            return result
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(
                f"Function {func.__name__} failed",
                exc_info=True,
                extra={
                    'duration': duration,
                    'function': func.__name__,
                    'module': func.__module__,
                    'error_type': type(e).__name__
                }
            )
            raise
            
    return wrapper


def log_async_performance(func):
    """Decorator to log async function performance"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000
            
            logger.info(
                f"Async function {func.__name__} completed",
                extra={
                    'duration': duration,
                    'function': func.__name__,
                    'module': func.__module__,
                    'is_async': True
                }
            )
            return result
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(
                f"Async function {func.__name__} failed",
                exc_info=True,
                extra={
                    'duration': duration,
                    'function': func.__name__,
                    'module': func.__module__,
                    'error_type': type(e).__name__,
                    'is_async': True
                }
            )
            raise
            
    return wrapper


class ErrorAggregator:
    """Aggregate and report errors"""
    
    def __init__(self, max_errors: int = 100):
        self.max_errors = max_errors
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        
    def add_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Add an error to the aggregator"""
        with self._lock:
            error_type = type(error).__name__
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
            error_data = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'type': error_type,
                'message': str(error),
                'traceback': traceback.format_exc(),
                'context': context or {}
            }
            
            self.errors.append(error_data)
            
            # Keep only recent errors
            if len(self.errors) > self.max_errors:
                self.errors = self.errors[-self.max_errors:]
                
    def get_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        with self._lock:
            return {
                'total_errors': sum(self.error_counts.values()),
                'error_types': dict(self.error_counts),
                'recent_errors': self.errors[-10:]  # Last 10 errors
            }
            
    def clear(self):
        """Clear all errors"""
        with self._lock:
            self.errors.clear()
            self.error_counts.clear()


# Global error aggregator instance
error_aggregator = ErrorAggregator()


def setup_default_logging():
    """Setup default logging configuration from environment"""
    LoggingConfig.setup_logging(
        log_level=os.environ.get('LOG_LEVEL', 'INFO'),
        log_file=os.environ.get('LOG_FILE', './logs/academic-rag.log'),
        use_json=os.environ.get('LOG_JSON', 'false').lower() == 'true',
        service_name=os.environ.get('SERVICE_NAME', 'academic-rag'),
        enable_console=os.environ.get('LOG_CONSOLE', 'true').lower() == 'true',
        enable_file=os.environ.get('LOG_FILE_ENABLED', 'true').lower() == 'true'
    )


# Example usage
if __name__ == "__main__":
    # Setup logging
    LoggingConfig.setup_logging(
        log_level="DEBUG",
        log_file="./logs/test.log",
        use_json=True
    )
    
    # Get logger with context
    logger = LoggingConfig.get_logger(__name__, with_context=True)
    
    # Log with context
    logger.set_context(user_id="12345", session_id="abc-def")
    logger.info("Processing document", extra={"document_id": "doc-001"})
    
    # Use context manager
    with logger.context(operation="ocr_processing"):
        logger.info("Starting OCR")
        # Simulate error
        try:
            raise ValueError("OCR failed")
        except Exception as e:
            logger.error("OCR error occurred", exc_info=True)
            error_aggregator.add_error(e, {"operation": "ocr"})
    
    # Performance logging
    @log_performance
    def slow_operation():
        time.sleep(0.1)
        return "done"
    
    result = slow_operation()
    
    # Print error summary
    print(json.dumps(error_aggregator.get_summary(), indent=2))