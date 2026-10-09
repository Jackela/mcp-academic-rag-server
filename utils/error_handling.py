"""
Enhanced error handling utilities for the Academic RAG Server

This module provides comprehensive error handling with:
- Custom exception classes
- Error recovery strategies
- Retry mechanisms
- Circuit breaker pattern
- Error context preservation
"""

import asyncio
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List, Optional, ParamSpec, Protocol, Type, TypeVar

from loguru import logger

T = TypeVar("T")
P = ParamSpec("P")


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""

    NETWORK = "network"
    VALIDATION = "validation"
    PROCESSING = "processing"
    STORAGE = "storage"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    RESOURCE = "resource"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors"""

    timestamp: datetime
    operation: str
    component: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    document_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "component": self.component,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "document_id": self.document_id,
            "additional_data": self.additional_data or {},
        }


class BaseRAGException(Exception):
    """Base exception for RAG system"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.category = category
        self.context = context
        self.cause = cause
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses"""
        return {
            "error_code": self.error_code,
            "message": str(self),
            "severity": self.severity.value,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context.to_dict() if self.context else None,
            "cause": str(self.cause) if self.cause else None,
            "traceback": traceback.format_exc() if self.cause else None,
        }


# Specific exception classes
class ConfigurationError(BaseRAGException):
    """Configuration related errors"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code="CONFIG_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            **kwargs,
        )


class ProcessingError(BaseRAGException):
    """Document processing errors"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code="PROCESSING_ERROR",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING,
            **kwargs,
        )


class StorageError(BaseRAGException):
    """Storage related errors"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message, error_code="STORAGE_ERROR", severity=ErrorSeverity.HIGH, category=ErrorCategory.STORAGE, **kwargs
        )


class NetworkError(BaseRAGException):
    """Network related errors"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message, error_code="NETWORK_ERROR", severity=ErrorSeverity.MEDIUM, category=ErrorCategory.NETWORK, **kwargs
        )


class ValidationError(BaseRAGException):
    """Validation errors"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.VALIDATION,
            **kwargs,
        )


class AuthenticationError(BaseRAGException):
    """Authentication errors"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code="AUTH_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AUTHENTICATION,
            **kwargs,
        )


class RateLimitError(BaseRAGException):
    """Rate limiting errors"""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(
            message,
            error_code="RATE_LIMIT_ERROR",
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.RATE_LIMIT,
            **kwargs,
        )
        self.retry_after = retry_after


class ResourceError(BaseRAGException):
    """Resource exhaustion errors"""

    def __init__(self, message: str, resource_type: str, **kwargs: Any) -> None:
        super().__init__(
            message, error_code="RESOURCE_ERROR", severity=ErrorSeverity.HIGH, category=ErrorCategory.RESOURCE, **kwargs
        )
        self.resource_type = resource_type


class RetryConfig:
    """Configuration for retry behavior"""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_exceptions: Optional[List[Type[Exception]]] = None,
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_exceptions = retry_exceptions or [NetworkError, RateLimitError, TimeoutError, ConnectionError]


def _retry_delay(config: RetryConfig, error: Exception, attempt: int) -> float:
    delay = min(config.initial_delay * config.exponential_base**attempt, config.max_delay)
    if config.jitter:
        import random

        delay *= 0.5 + random.random()
    if isinstance(error, RateLimitError) and error.retry_after:
        delay = max(delay, error.retry_after)
    return delay


def retry_with_backoff(
    config: Optional[RetryConfig] = None, on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Retry configured exceptions while preserving the callable's parameter contract."""
    retry_config = config if config is not None else RetryConfig()
    if retry_config.max_attempts < 1:
        raise ValueError("max_attempts must be at least one")

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            for attempt in range(retry_config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    if (
                        not isinstance(error, tuple(retry_config.retry_exceptions))
                        or attempt == retry_config.max_attempts - 1
                    ):
                        raise
                    delay = _retry_delay(retry_config, error, attempt)
                    if on_retry:
                        on_retry(error, attempt + 1)
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{retry_config.max_attempts} for {func.__name__} after {delay:.2f}s delay. Error: {error}"
                    )
                    time.sleep(delay)
            raise RuntimeError("Retry attempt configuration changed during execution")

        return wrapper

    return decorator


def async_retry_with_backoff(
    config: Optional[RetryConfig] = None, on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Async retry with the same exception and delay contract as synchronous retry."""
    retry_config = config if config is not None else RetryConfig()
    if retry_config.max_attempts < 1:
        raise ValueError("max_attempts must be at least one")

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            for attempt in range(retry_config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as error:
                    if (
                        not isinstance(error, tuple(retry_config.retry_exceptions))
                        or attempt == retry_config.max_attempts - 1
                    ):
                        raise
                    delay = _retry_delay(retry_config, error, attempt)
                    if on_retry:
                        on_retry(error, attempt + 1)
                    logger.warning(
                        f"Async retry attempt {attempt + 1}/{retry_config.max_attempts} for {func.__name__} after {delay:.2f}s delay. Error: {error}"
                    )
                    await asyncio.sleep(delay)
            raise RuntimeError("Retry attempt configuration changed during execution")

        return wrapper

    return decorator


class CircuitBreakerState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker pattern implementation

    Prevents cascading failures by temporarily blocking calls to failing services
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._last_failure_time and datetime.utcnow() - self._last_failure_time > timedelta(
                    seconds=self.recovery_timeout
                ):
                    self._state = CircuitBreakerState.HALF_OPEN
            return self._state

    def call(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Call function through circuit breaker"""
        if self.state == CircuitBreakerState.OPEN:
            raise NetworkError("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception:
            self._on_failure()
            raise

    async def async_call(self, func: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
        """Async call through circuit breaker"""
        if self.state == CircuitBreakerState.OPEN:
            raise NetworkError("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Handle successful call"""
        with self._lock:
            self._failure_count = 0
            self._state = CircuitBreakerState.CLOSED

    def _on_failure(self) -> None:
        """Handle failed call"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.utcnow()

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN
                logger.error(f"Circuit breaker opened after {self._failure_count} failures")


def handle_errors(
    default_return: object = None,
    log_errors: bool = True,
    reraise: bool = False,
    error_handler: Optional[Callable[[Exception], object]] = None,
) -> Callable[[Callable[P, T]], Callable[P, object]]:
    """
    Decorator for graceful error handling

    Args:
        default_return: Default value to return on error
        log_errors: Whether to log errors
        reraise: Whether to re-raise the exception after handling
        error_handler: Custom error handler function
    """

    def decorator(func: Callable[P, T]) -> Callable[P, object]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> object:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        f"Error in {func.__name__}: {str(e)}",
                        exc_info=True,
                        extra={"function": func.__name__, "module": func.__module__, "error_type": type(e).__name__},
                    )

                if error_handler:
                    result = error_handler(e)
                    if result is not None:
                        return result

                if reraise:
                    raise

                return default_return

        return wrapper

    return decorator


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies"""

    def can_recover(self, error: Exception) -> bool:
        """Check if recovery is possible for this error"""
        raise NotImplementedError

    def recover(self, error: Exception, context: Dict[str, Any]) -> object:
        """Attempt to recover from error"""
        raise NotImplementedError


class FallbackRecovery(ErrorRecoveryStrategy):
    """Fallback to alternative service/method"""

    def __init__(self, fallback_func: Callable[..., object]):
        self.fallback_func = fallback_func

    def can_recover(self, error: Exception) -> bool:
        return isinstance(error, (NetworkError, TimeoutError))

    def recover(self, error: Exception, context: Dict[str, Any]) -> object:
        logger.info(f"Using fallback for {error}")
        return self.fallback_func(**context)


class RecoveryCache(Protocol):
    """The cache boundary used by error recovery."""

    def get(self, key: str) -> object: ...


class CacheRecovery(ErrorRecoveryStrategy):
    """Recover using cached data"""

    def __init__(self, cache_provider: RecoveryCache) -> None:
        self.cache = cache_provider

    def can_recover(self, error: Exception) -> bool:
        return isinstance(error, (NetworkError, RateLimitError))

    def recover(self, error: Exception, context: Dict[str, Any]) -> object:
        cache_key = context.get("cache_key")
        if isinstance(cache_key, str) and cache_key:
            cached_value = self.cache.get(cache_key)
            if cached_value is not None:
                logger.info(f"Recovered from cache for key: {cache_key}")
                return cached_value
        raise error


def with_recovery(strategies: List[ErrorRecoveryStrategy]) -> Callable[[Callable[P, T]], Callable[P, object]]:
    """Decorator to apply recovery strategies on error"""

    def decorator(func: Callable[P, T]) -> Callable[P, object]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> object:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Try each recovery strategy
                for strategy in strategies:
                    if strategy.can_recover(e):
                        try:
                            context = {"args": args, "kwargs": kwargs, "function": func.__name__}
                            return strategy.recover(e, context)
                        except Exception as recovery_error:
                            logger.warning(
                                f"Recovery strategy {strategy.__class__.__name__} " f"failed: {recovery_error}"
                            )
                            continue

                # No recovery possible
                raise

        return wrapper

    return decorator


# Example usage
if __name__ == "__main__":
    # Example with retry
    @retry_with_backoff(config=RetryConfig(max_attempts=3, initial_delay=1.0))
    def unreliable_api_call() -> str:
        import random

        if random.random() < 0.7:
            raise NetworkError("API call failed")
        return "Success"

    # Example with circuit breaker
    circuit_breaker = CircuitBreaker(failure_threshold=3)

    def api_call() -> None:
        # Simulate API call
        raise NetworkError("Service unavailable")

    try:
        result = circuit_breaker.call(api_call)
    except NetworkError:
        print(f"Circuit breaker state: {circuit_breaker.state}")

    # Example with error handling
    @handle_errors(default_return=[], log_errors=True)
    def get_documents() -> List[object]:
        raise StorageError("Database connection failed")

    docs = get_documents()  # Returns [] instead of raising
    print(f"Documents: {docs}")
