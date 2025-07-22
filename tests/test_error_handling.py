"""Unit tests for error recovery mechanisms."""

import pytest
import time
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from stock_analysis.utils.error_recovery import (
    RetryConfig, CircuitBreaker, ErrorRecoveryManager, with_error_recovery
)
from stock_analysis.utils.exceptions import (
    StockAnalysisError, DataRetrievalError, NetworkError, RateLimitError, InsufficientDataError
)


class TestRetryConfig:
    """Test cases for RetryConfig."""
    
    def test_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=False,
            backoff_strategy='exponential'
        )
        
        assert config.calculate_delay(0) == 1.0  # 1 * 2^0
        assert config.calculate_delay(1) == 2.0  # 1 * 2^1
        assert config.calculate_delay(2) == 4.0  # 1 * 2^2
        assert config.calculate_delay(3) == 8.0  # 1 * 2^3
        assert config.calculate_delay(4) == 10.0  # Limited by max_delay
    
    def test_linear_backoff(self):
        """Test linear backoff delay calculation."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=10.0,
            jitter=False,
            backoff_strategy='linear'
        )
        
        assert config.calculate_delay(0) == 1.0  # 1 * (0 + 1)
        assert config.calculate_delay(1) == 2.0  # 1 * (1 + 1)
        assert config.calculate_delay(2) == 3.0  # 1 * (2 + 1)
        assert config.calculate_delay(3) == 4.0  # 1 * (3 + 1)
    
    def test_fixed_backoff(self):
        """Test fixed backoff delay calculation."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=10.0,
            jitter=False,
            backoff_strategy='fixed'
        )
        
        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 1.0
        assert config.calculate_delay(2) == 1.0
    
    def test_jitter(self):
        """Test jitter in delay calculation."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=10.0,
            jitter=True,
            backoff_strategy='fixed'
        )
        
        # Test multiple times to ensure jitter is being applied
        delays = [config.calculate_delay(0) for _ in range(10)]
        
        # Verify delays are within expected range (±10% of base delay)
        for delay in delays:
            assert 0.9 <= delay <= 1.1
        
        # Verify not all delays are the same (jitter is working)
        assert len(set(delays)) > 1


class TestCircuitBreaker:
    """Test cases for CircuitBreaker."""
    
    def test_successful_execution(self):
        """Test successful function execution."""
        breaker = CircuitBreaker()
        func = lambda: "success"
        
        result = breaker.call(func)
        assert result == "success"
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0
    
    def test_failure_threshold(self):
        """Test circuit opens after failure threshold."""
        breaker = CircuitBreaker(failure_threshold=2)
        func = lambda: 1/0  # Will raise ZeroDivisionError
        
        # First failure
        with pytest.raises(ZeroDivisionError):
            breaker.call(func)
        assert breaker.state == "CLOSED"
        
        # Second failure - should open circuit
        with pytest.raises(ZeroDivisionError):
            breaker.call(func)
        assert breaker.state == "OPEN"
        
        # Subsequent calls should raise CircuitBreakerError
        with pytest.raises(StockAnalysisError) as exc_info:
            breaker.call(func)
        assert "Circuit breaker is OPEN" in str(exc_info.value)
    
    def test_recovery_timeout(self):
        """Test circuit recovery after timeout."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        success_func = lambda: "success"
        fail_func = lambda: 1/0
        
        # Fail to open circuit
        with pytest.raises(ZeroDivisionError):
            breaker.call(fail_func)
        assert breaker.state == "OPEN"
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should transition to HALF_OPEN and allow attempt
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == "CLOSED"
    
    def test_half_open_state(self):
        """Test half-open state behavior."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        fail_func = lambda: 1/0
        
        # Fail to open circuit
        with pytest.raises(ZeroDivisionError):
            breaker.call(fail_func)
        assert breaker.state == "OPEN"
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should transition to HALF_OPEN but fail again
        with pytest.raises(ZeroDivisionError):
            breaker.call(fail_func)
        assert breaker.state == "OPEN"


class TestErrorRecoveryManager:
    """Test cases for ErrorRecoveryManager."""
    
    @pytest.fixture
    def recovery_manager(self):
        """Create an error recovery manager for testing."""
        return ErrorRecoveryManager()
    
    def test_retry_success(self, recovery_manager):
        """Test successful retry after failure."""
        attempts = 0
        
        def test_func():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise NetworkError("Test network error")
            return "success"
        
        result = recovery_manager.execute_with_recovery(
            test_func,
            "test_operation",
            retry_category="network",
            use_circuit_breaker=False
        )
        
        assert result == "success"
        assert attempts == 2
    
    def test_retry_exhaustion(self, recovery_manager):
        """Test exhausting all retry attempts."""
        attempts = 0
        
        def test_func():
            nonlocal attempts
            attempts += 1
            raise NetworkError("Test network error")
        
        with pytest.raises(NetworkError):
            recovery_manager.execute_with_recovery(
                test_func,
                "test_operation",
                retry_category="network",
                use_circuit_breaker=False
            )
        
        assert attempts == 3  # Default max attempts for network category
    
    def test_fallback_value(self, recovery_manager):
        """Test using fallback value after failures."""
        def test_func():
            raise NetworkError("Test network error")
        
        result = recovery_manager.execute_with_recovery(
            test_func,
            "test_operation",
            retry_category="network",
            use_circuit_breaker=False,
            fallback_value="fallback"
        )
        
        assert result == "fallback"
    
    def test_circuit_breaker_integration(self, recovery_manager):
        """Test integration with circuit breaker."""
        attempts = 0
        
        def test_func():
            nonlocal attempts
            attempts += 1
            raise NetworkError("Test network error")
        
        # First execution - should try max attempts
        with pytest.raises(NetworkError):
            recovery_manager.execute_with_recovery(
                test_func,
                "test_operation",
                retry_category="network",
                use_circuit_breaker=True
            )
        
        initial_attempts = attempts
        
        # Second execution - circuit should be open
        with pytest.raises(StockAnalysisError) as exc_info:
            recovery_manager.execute_with_recovery(
                test_func,
                "test_operation",
                retry_category="network",
                use_circuit_breaker=True
            )
        
        assert "Circuit breaker is OPEN" in str(exc_info.value)
        assert attempts == initial_attempts  # No new attempts made
    
    def test_custom_fallback_handler(self, recovery_manager):
        """Test registering and using custom fallback handler."""
        def custom_handler(exception, operation_name, context):
            return "custom fallback"
        
        recovery_manager.register_fallback_handler(NetworkError, custom_handler)
        
        def test_func():
            raise NetworkError("Test network error")
        
        result = recovery_manager.execute_with_recovery(
            test_func,
            "test_operation",
            retry_category="network",
            use_circuit_breaker=False
        )
        
        assert result == "custom fallback"
    
    def test_data_retrieval_fallback(self, recovery_manager):
        """Test data retrieval fallback handler."""
        # Mock cache manager
        mock_cache = MagicMock()
        mock_cache.get.return_value = "cached_data"
        
        with patch('stock_analysis.utils.error_recovery.get_cache_manager', return_value=mock_cache):
            def test_func():
                raise DataRetrievalError(
                    "Test error",
                    context={'symbol': 'AAPL'}
                )
            
            result = recovery_manager.execute_with_recovery(
                test_func,
                "get_stock_info",
                retry_category="data_retrieval",
                use_circuit_breaker=False
            )
            
            assert result == "cached_data"
            mock_cache.get.assert_called_once_with("stock_info:AAPL")
    
    def test_rate_limit_fallback(self, recovery_manager):
        """Test rate limit fallback handler."""
        attempts = 0
        
        def test_func():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RateLimitError(
                    "Rate limit exceeded",
                    context={'retry_after': 0.1}
                )
            return "success"
        
        result = recovery_manager.execute_with_recovery(
            test_func,
            "test_operation",
            retry_category="network",
            use_circuit_breaker=False
        )
        
        assert result == "success"
        assert attempts == 2
    
    def test_recovery_stats(self, recovery_manager):
        """Test getting recovery statistics."""
        # Create some circuit breakers
        recovery_manager.get_circuit_breaker("operation1")
        recovery_manager.get_circuit_breaker("operation2")
        
        stats = recovery_manager.get_recovery_stats()
        
        assert "circuit_breakers" in stats
        assert "retry_configs" in stats
        assert "fallback_handlers" in stats
        assert len(stats["circuit_breakers"]) == 2
        assert "operation1" in stats["circuit_breakers"]
        assert "operation2" in stats["circuit_breakers"]


def test_error_recovery_decorator():
    """Test the error recovery decorator."""
    attempts = 0
    
    @with_error_recovery("test_operation", retry_category="network")
    def test_func():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise NetworkError("Test network error")
        return "success"
    
    result = test_func()
    assert result == "success"
    assert attempts == 2
