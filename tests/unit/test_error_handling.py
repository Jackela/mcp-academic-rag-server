"""
Unit tests for error handling utilities
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime

from utils.error_handling import (
    BaseRAGException,
    ConfigurationError,
    ProcessingError,
    NetworkError,
    RateLimitError,
    ErrorContext,
    ErrorSeverity,
    ErrorCategory,
    RetryConfig,
    retry_with_backoff,
    async_retry_with_backoff,
    CircuitBreaker,
    handle_errors,
    FallbackRecovery,
    with_recovery
)


class TestCustomExceptions:
    """Test custom exception classes"""
    
    def test_base_exception_creation(self):
        """Test BaseRAGException creation"""
        context = ErrorContext(
            timestamp=datetime.utcnow(),
            operation="test_operation",
            component="test_component",
            user_id="user123"
        )
        
        error = BaseRAGException(
            message="Test error",
            error_code="TEST001",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.PROCESSING,
            context=context
        )
        
        assert str(error) == "Test error"
        assert error.error_code == "TEST001"
        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.PROCESSING
        assert error.context == context
    
    def test_exception_to_dict(self):
        """Test exception serialization"""
        error = ConfigurationError("Config not found")
        error_dict = error.to_dict()
        
        assert error_dict['error_code'] == "CONFIG_ERROR"
        assert error_dict['message'] == "Config not found"
        assert error_dict['severity'] == ErrorSeverity.HIGH.value
        assert error_dict['category'] == ErrorCategory.CONFIGURATION.value
        assert 'timestamp' in error_dict
    
    def test_specific_exceptions(self):
        """Test specific exception types"""
        # Test each exception type
        exceptions = [
            (ConfigurationError, "CONFIG_ERROR", ErrorCategory.CONFIGURATION),
            (ProcessingError, "PROCESSING_ERROR", ErrorCategory.PROCESSING),
            (NetworkError, "NETWORK_ERROR", ErrorCategory.NETWORK),
        ]
        
        for exc_class, error_code, category in exceptions:
            error = exc_class("Test message")
            assert error.error_code == error_code
            assert error.category == category
    
    def test_rate_limit_error(self):
        """Test RateLimitError with retry_after"""
        error = RateLimitError("Too many requests", retry_after=60)
        assert error.retry_after == 60
        assert error.category == ErrorCategory.RATE_LIMIT


class TestRetryDecorator:
    """Test retry decorator functionality"""
    
    def test_retry_success_on_second_attempt(self):
        """Test function succeeds on retry"""
        mock_func = Mock(side_effect=[NetworkError("Failed"), "Success"])
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        
        @retry_with_backoff(config=config)
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "Success"
        assert mock_func.call_count == 2
    
    def test_retry_exhausted(self):
        """Test retry attempts exhausted"""
        mock_func = Mock(side_effect=NetworkError("Always fails"))
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        
        @retry_with_backoff(config=config)
        def test_func():
            return mock_func()
        
        with pytest.raises(NetworkError):
            test_func()
        
        assert mock_func.call_count == 3
    
    def test_retry_with_non_retryable_exception(self):
        """Test non-retryable exception"""
        mock_func = Mock(side_effect=ValueError("Invalid value"))
        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.01,
            retry_exceptions=[NetworkError]
        )
        
        @retry_with_backoff(config=config)
        def test_func():
            return mock_func()
        
        with pytest.raises(ValueError):
            test_func()
        
        # Should not retry for ValueError
        assert mock_func.call_count == 1
    
    def test_retry_callback(self):
        """Test retry callback is called"""
        callback_mock = Mock()
        mock_func = Mock(side_effect=[NetworkError("Failed"), "Success"])
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        
        @retry_with_backoff(config=config, on_retry=callback_mock)
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "Success"
        assert callback_mock.call_count == 1
        
        # Check callback arguments
        args = callback_mock.call_args[0]
        assert isinstance(args[0], NetworkError)
        assert args[1] == 1  # Attempt number
    
    @pytest.mark.asyncio
    async def test_async_retry(self):
        """Test async retry decorator"""
        mock_func = Mock(side_effect=[NetworkError("Failed"), "Success"])
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        
        @async_retry_with_backoff(config=config)
        async def test_func():
            return mock_func()
        
        result = await test_func()
        assert result == "Success"
        assert mock_func.call_count == 2


class TestCircuitBreaker:
    """Test circuit breaker pattern"""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state"""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        
        # Successful calls should work
        mock_func = Mock(return_value="Success")
        result = breaker.call(mock_func)
        assert result == "Success"
        assert breaker.state.value == "closed"
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold"""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            expected_exception=NetworkError
        )
        
        mock_func = Mock(side_effect=NetworkError("Failed"))
        
        # Fail threshold times
        for _ in range(3):
            with pytest.raises(NetworkError):
                breaker.call(mock_func)
        
        # Circuit should be open
        assert breaker.state.value == "open"
        
        # Further calls should fail immediately
        with pytest.raises(NetworkError, match="Circuit breaker is OPEN"):
            breaker.call(mock_func)
        
        # Function should not be called when circuit is open
        assert mock_func.call_count == 3
    
    def test_circuit_breaker_half_open_state(self):
        """Test circuit breaker transitions to half-open"""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            expected_exception=NetworkError
        )
        
        mock_func = Mock(side_effect=NetworkError("Failed"))
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(NetworkError):
                breaker.call(mock_func)
        
        assert breaker.state.value == "open"
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # Should be half-open now
        assert breaker.state.value == "half_open"
        
        # Successful call should close circuit
        mock_func.side_effect = None
        mock_func.return_value = "Success"
        result = breaker.call(mock_func)
        assert result == "Success"
        assert breaker.state.value == "closed"
    
    @pytest.mark.asyncio
    async def test_async_circuit_breaker(self):
        """Test async circuit breaker"""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        async def async_func():
            return "Success"
        
        result = await breaker.async_call(async_func)
        assert result == "Success"


class TestErrorHandlingDecorator:
    """Test error handling decorator"""
    
    def test_handle_errors_with_default_return(self):
        """Test error handling with default return value"""
        @handle_errors(default_return="Default", log_errors=False)
        def failing_func():
            raise ValueError("Test error")
        
        result = failing_func()
        assert result == "Default"
    
    def test_handle_errors_with_reraise(self):
        """Test error handling with re-raise"""
        @handle_errors(reraise=True, log_errors=False)
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_func()
    
    def test_handle_errors_with_custom_handler(self):
        """Test error handling with custom handler"""
        def custom_handler(error):
            if isinstance(error, ValueError):
                return "Handled ValueError"
            return None
        
        @handle_errors(error_handler=custom_handler, log_errors=False)
        def failing_func():
            raise ValueError("Test error")
        
        result = failing_func()
        assert result == "Handled ValueError"
    
    def test_handle_errors_logging(self):
        """Test error logging"""
        with patch('utils.error_handling.logger') as mock_logger:
            @handle_errors(log_errors=True)
            def failing_func():
                raise ValueError("Test error")
            
            failing_func()
            mock_logger.error.assert_called_once()


class TestRecoveryStrategies:
    """Test error recovery strategies"""
    
    def test_fallback_recovery(self):
        """Test fallback recovery strategy"""
        def fallback_func(**kwargs):
            return "Fallback result"
        
        recovery = FallbackRecovery(fallback_func)
        
        # Should recover from NetworkError
        assert recovery.can_recover(NetworkError("Failed"))
        result = recovery.recover(NetworkError("Failed"), {})
        assert result == "Fallback result"
        
        # Should not recover from other errors
        assert not recovery.can_recover(ValueError("Failed"))
    
    def test_with_recovery_decorator(self):
        """Test recovery decorator"""
        def fallback_func(**kwargs):
            return "Recovered"
        
        strategies = [FallbackRecovery(fallback_func)]
        
        @with_recovery(strategies)
        def failing_func():
            raise NetworkError("Failed")
        
        result = failing_func()
        assert result == "Recovered"
    
    def test_with_recovery_no_strategy_matches(self):
        """Test recovery when no strategy matches"""
        strategies = []
        
        @with_recovery(strategies)
        def failing_func():
            raise ValueError("Failed")
        
        with pytest.raises(ValueError):
            failing_func()


class TestErrorContext:
    """Test error context functionality"""
    
    def test_error_context_creation(self):
        """Test ErrorContext creation and serialization"""
        context = ErrorContext(
            timestamp=datetime.utcnow(),
            operation="test_op",
            component="test_comp",
            user_id="user123",
            session_id="session456",
            document_id="doc789",
            additional_data={"key": "value"}
        )
        
        context_dict = context.to_dict()
        assert context_dict['operation'] == "test_op"
        assert context_dict['component'] == "test_comp"
        assert context_dict['user_id'] == "user123"
        assert context_dict['additional_data']['key'] == "value"
        assert 'timestamp' in context_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])