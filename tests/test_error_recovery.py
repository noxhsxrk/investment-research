"""Tests for error recovery mechanisms."""

import unittest
import time
from unittest.mock import patch, MagicMock, call
import requests
from datetime import datetime

from stock_analysis.utils.error_recovery import (
    RetryConfig,
    CircuitBreaker,
    ErrorRecoveryManager,
    TransientErrorDetector,
    ErrorDiagnostics,
    with_error_recovery,
    with_data_validation
)
from stock_analysis.utils.exceptions import (
    StockAnalysisError,
    DataRetrievalError,
    NetworkError,
    RateLimitError,
    ValidationError
)


class TestRetryConfig(unittest.TestCase):
    """Test cases for RetryConfig class."""
    
    def test_calculate_delay_exponential(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=False,
            backoff_strategy='exponential'
        )
        
        self.assertEqual(config.calculate_delay(0), 1.0)
        self.assertEqual(config.calculate_delay(1), 2.0)
        self.assertEqual(config.calculate_delay(2), 4.0)
        self.assertEqual(config.calculate_delay(3), 8.0)
        # Max delay cap
        self.assertEqual(config.calculate_delay(4), 10.0)
    
    def test_calculate_delay_linear(self):
        """Test linear backoff delay calculation."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=10.0,
            jitter=False,
            backoff_strategy='linear'
        )
        
        self.assertEqual(config.calculate_delay(0), 1.0)
        self.assertEqual(config.calculate_delay(1), 2.0)
        self.assertEqual(config.calculate_delay(2), 3.0)
        self.assertEqual(config.calculate_delay(3), 4.0)
    
    def test_calculate_delay_fixed(self):
        """Test fixed delay calculation."""
        config = RetryConfig(
            base_delay=2.0,
            jitter=False,
            backoff_strategy='fixed'
        )
        
        self.assertEqual(config.calculate_delay(0), 2.0)
        self.assertEqual(config.calculate_delay(1), 2.0)
        self.assertEqual(config.calculate_delay(2), 2.0)
    
    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(
            base_delay=1.0,
            jitter=True,
            backoff_strategy='fixed'
        )
        
        # With jitter, we can only check that the delay is within expected range
        delay = config.calculate_delay(0)
        self.assertTrue(0.9 <= delay <= 1.1)  # 10% jitter


class TestCircuitBreaker(unittest.TestCase):
    """Test cases for CircuitBreaker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1  # Short timeout for testing
        )
    
    def test_successful_call(self):
        """Test successful function call."""
        result = self.circuit_breaker.call(lambda: 42)
        self.assertEqual(result, 42)
        self.assertEqual(self.circuit_breaker.state, 'CLOSED')
        self.assertEqual(self.circuit_breaker.failure_count, 0)
    
    def test_circuit_opens_after_failures(self):
        """Test circuit opens after reaching failure threshold."""
        # First failure
        with self.assertRaises(ValueError):
            self.circuit_breaker.call(lambda: int('not_a_number'))
        
        self.assertEqual(self.circuit_breaker.state, 'CLOSED')
        self.assertEqual(self.circuit_breaker.failure_count, 1)
        
        # Second failure - should open the circuit
        with self.assertRaises(ValueError):
            self.circuit_breaker.call(lambda: int('also_not_a_number'))
        
        self.assertEqual(self.circuit_breaker.state, 'OPEN')
        self.assertEqual(self.circuit_breaker.failure_count, 2)
        
        # Subsequent call should fail with circuit breaker error
        with self.assertRaises(StockAnalysisError) as context:
            self.circuit_breaker.call(lambda: 42)
        
        self.assertIn('Circuit breaker is OPEN', str(context.exception))
    
    def test_circuit_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        # Open the circuit
        with self.assertRaises(ValueError):
            self.circuit_breaker.call(lambda: int('not_a_number'))
        
        with self.assertRaises(ValueError):
            self.circuit_breaker.call(lambda: int('also_not_a_number'))
        
        self.assertEqual(self.circuit_breaker.state, 'OPEN')
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Next call should transition to HALF_OPEN
        result = self.circuit_breaker.call(lambda: 42)
        self.assertEqual(result, 42)
        self.assertEqual(self.circuit_breaker.state, 'CLOSED')
        self.assertEqual(self.circuit_breaker.failure_count, 0)
    
    def test_circuit_remains_open_during_timeout(self):
        """Test circuit remains open during timeout period."""
        # Open the circuit
        with self.assertRaises(ValueError):
            self.circuit_breaker.call(lambda: int('not_a_number'))
        
        with self.assertRaises(ValueError):
            self.circuit_breaker.call(lambda: int('also_not_a_number'))
        
        self.assertEqual(self.circuit_breaker.state, 'OPEN')
        
        # Immediate call should fail with circuit breaker error
        with self.assertRaises(StockAnalysisError):
            self.circuit_breaker.call(lambda: 42)


class TestTransientErrorDetector(unittest.TestCase):
    """Test cases for TransientErrorDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = TransientErrorDetector()
    
    def test_detect_transient_error_by_type(self):
        """Test detection of transient errors by exception type."""
        self.assertTrue(self.detector.is_transient_error(NetworkError("Connection failed")))
        self.assertTrue(self.detector.is_transient_error(RateLimitError("Rate limit exceeded")))
        self.assertTrue(self.detector.is_transient_error(ConnectionError("Connection refused")))
        self.assertTrue(self.detector.is_transient_error(TimeoutError("Operation timed out")))
    
    def test_detect_transient_error_by_message(self):
        """Test detection of transient errors by exception message."""
        self.assertTrue(self.detector.is_transient_error(Exception("Connection timed out")))
        self.assertTrue(self.detector.is_transient_error(Exception("Rate limit exceeded")))
        self.assertTrue(self.detector.is_transient_error(Exception("Service unavailable")))
        self.assertTrue(self.detector.is_transient_error(Exception("Please retry later")))
        self.assertTrue(self.detector.is_transient_error(Exception("503 Server Error")))
    
    def test_non_transient_errors(self):
        """Test non-transient errors are correctly identified."""
        self.assertFalse(self.detector.is_transient_error(ValueError("Invalid value")))
        self.assertFalse(self.detector.is_transient_error(TypeError("Invalid type")))
        self.assertFalse(self.detector.is_transient_error(KeyError("Key not found")))
        self.assertFalse(self.detector.is_transient_error(Exception("Unknown error")))


class TestErrorDiagnostics(unittest.TestCase):
    """Test cases for ErrorDiagnostics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.diagnostics = ErrorDiagnostics()
    
    def test_diagnose_data_retrieval_error(self):
        """Test diagnosis of data retrieval error."""
        error = DataRetrievalError(
            "Failed to retrieve data for AAPL",
            symbol="AAPL",
            data_source="yfinance"
        )
        
        diagnosis = self.diagnostics.diagnose_error(error, "get_security_info")
        
        self.assertEqual(diagnosis['exception_type'], "DataRetrievalError")
        self.assertEqual(diagnosis['category'], "data_retrieval")
        self.assertEqual(diagnosis['severity'], "high")
        self.assertEqual(diagnosis['recommended_action'], "investigate")
    
    def test_diagnose_network_error(self):
        """Test diagnosis of network error."""
        error = NetworkError(
            "Connection timed out",
            url="https://api.example.com",
            status_code=None
        )
        
        diagnosis = self.diagnostics.diagnose_error(error, "api_request")
        
        self.assertEqual(diagnosis['exception_type'], "NetworkError")
        self.assertEqual(diagnosis['category'], "network")
        self.assertTrue(diagnosis['is_transient'])
        self.assertEqual(diagnosis['severity'], "medium")
        self.assertEqual(diagnosis['recommended_action'], "retry")
    
    def test_diagnose_validation_error(self):
        """Test diagnosis of validation error."""
        error = ValidationError(
            "Invalid data format",
            field_name="price",
            field_value="not_a_number"
        )
        
        diagnosis = self.diagnostics.diagnose_error(error, "validate_data")
        
        self.assertEqual(diagnosis['exception_type'], "ValidationError")
        self.assertEqual(diagnosis['category'], "validation")
        self.assertFalse(diagnosis['is_transient'])
        self.assertEqual(diagnosis['severity'], "high")
        self.assertEqual(diagnosis['recommended_action'], "fix_data")


class TestErrorRecoveryManager(unittest.TestCase):
    """Test cases for ErrorRecoveryManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ErrorRecoveryManager()
    
    @patch('stock_analysis.utils.error_recovery.get_operation_logger')
    @patch('stock_analysis.utils.error_recovery.get_metrics_collector')
    def test_execute_with_recovery_success(self, mock_get_metrics_collector, mock_get_operation_logger):
        """Test successful execution with recovery."""
        mock_collector = MagicMock()
        mock_get_metrics_collector.return_value = mock_collector
        
        mock_logger = MagicMock()
        mock_get_operation_logger.return_value = mock_logger
        
        # Function that succeeds
        def success_func():
            return 42
        
        result = self.manager.execute_with_recovery(
            success_func,
            "test_operation",
            retry_category="default",
            use_circuit_breaker=False
        )
        
        self.assertEqual(result, 42)
        mock_logger.start.assert_called_once()
        mock_logger.finish.assert_called_once_with(success=True, attempt=1)
    
    @patch('stock_analysis.utils.error_recovery.get_operation_logger')
    @patch('stock_analysis.utils.error_recovery.get_metrics_collector')
    def test_execute_with_recovery_retry_success(self, mock_get_metrics_collector, mock_get_operation_logger):
        """Test successful execution after retry."""
        mock_collector = MagicMock()
        mock_get_metrics_collector.return_value = mock_collector
        
        mock_logger = MagicMock()
        mock_get_operation_logger.return_value = mock_logger
        
        # Function that fails once then succeeds
        attempt = [0]
        def retry_func():
            attempt[0] += 1
            if attempt[0] == 1:
                raise NetworkError("Connection failed", url="https://api.example.com")
            return 42
        
        result = self.manager.execute_with_recovery(
            retry_func,
            "test_operation",
            retry_category="network",
            use_circuit_breaker=False
        )
        
        self.assertEqual(result, 42)
        self.assertEqual(attempt[0], 2)  # Function was called twice
        mock_logger.start.assert_called_once()
        mock_logger.finish.assert_called_once_with(success=True, attempt=2)
    
    @patch('stock_analysis.utils.error_recovery.get_operation_logger')
    @patch('stock_analysis.utils.error_recovery.get_metrics_collector')
    def test_execute_with_recovery_all_attempts_fail(self, mock_get_metrics_collector, mock_get_operation_logger):
        """Test all retry attempts fail."""
        mock_collector = MagicMock()
        mock_get_metrics_collector.return_value = mock_collector
        
        mock_logger = MagicMock()
        mock_get_operation_logger.return_value = mock_logger
        
        # Function that always fails
        def fail_func():
            raise NetworkError("Connection failed", url="https://api.example.com")
        
        # Configure retry with no delay for testing
        retry_config = RetryConfig(max_attempts=3, base_delay=0.01, jitter=False)
        self.manager.retry_configs['network'] = retry_config
        
        with self.assertRaises(NetworkError):
            self.manager.execute_with_recovery(
                fail_func,
                "test_operation",
                retry_category="network",
                use_circuit_breaker=False
            )
        
        mock_logger.start.assert_called_once()
        mock_logger.finish.assert_called_once_with(success=False, exception=unittest.mock.ANY)
    
    @patch('stock_analysis.utils.error_recovery.get_operation_logger')
    @patch('stock_analysis.utils.error_recovery.get_metrics_collector')
    def test_execute_with_recovery_fallback_value(self, mock_get_metrics_collector, mock_get_operation_logger):
        """Test fallback value is returned when all attempts fail."""
        mock_collector = MagicMock()
        mock_get_metrics_collector.return_value = mock_collector
        
        mock_logger = MagicMock()
        mock_get_operation_logger.return_value = mock_logger
        
        # Function that always fails
        def fail_func():
            raise NetworkError("Connection failed", url="https://api.example.com")
        
        # Configure retry with no delay for testing
        retry_config = RetryConfig(max_attempts=2, base_delay=0.01, jitter=False)
        self.manager.retry_configs['network'] = retry_config
        
        result = self.manager.execute_with_recovery(
            fail_func,
            "test_operation",
            retry_category="network",
            use_circuit_breaker=False,
            fallback_value={"status": "error", "data": None}
        )
        
        self.assertEqual(result, {"status": "error", "data": None})
        mock_logger.start.assert_called_once()
        mock_logger.finish.assert_called_once_with(success=True, used_fallback=True)
    
    @patch('stock_analysis.utils.error_recovery.get_operation_logger')
    @patch('stock_analysis.utils.error_recovery.get_metrics_collector')
    def test_execute_with_recovery_custom_fallback(self, mock_get_metrics_collector, mock_get_operation_logger):
        """Test custom fallback handler is used."""
        mock_collector = MagicMock()
        mock_get_metrics_collector.return_value = mock_collector
        
        mock_logger = MagicMock()
        mock_get_operation_logger.return_value = mock_logger
        
        # Custom fallback handler
        def custom_fallback(exception, operation_name, context):
            return {"status": "fallback", "error": str(exception)}
        
        # Register fallback handler
        self.manager.fallback_handlers[NetworkError] = custom_fallback
        
        # Function that always fails
        def fail_func():
            raise NetworkError("Connection failed", url="https://api.example.com")
        
        # Configure retry with no delay for testing
        retry_config = RetryConfig(max_attempts=2, base_delay=0.01, jitter=False)
        self.manager.retry_configs['network'] = retry_config
        
        result = self.manager.execute_with_recovery(
            fail_func,
            "test_operation",
            retry_category="network",
            use_circuit_breaker=False
        )
        
        self.assertEqual(result["status"], "fallback")
        self.assertIn("Connection failed", result["error"])
        mock_logger.start.assert_called_once()
        mock_logger.finish.assert_called_once_with(success=True, used_fallback=True)


class TestErrorRecoveryDecorators(unittest.TestCase):
    """Test cases for error recovery decorators."""
    
    @patch('stock_analysis.utils.error_recovery.get_error_recovery_manager')
    def test_with_error_recovery_decorator(self, mock_get_manager):
        """Test with_error_recovery decorator."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        mock_manager.execute_with_recovery.return_value = 42
        
        @with_error_recovery("test_operation")
        def test_func(arg1, arg2=None):
            return arg1 + (arg2 or 0)
        
        result = test_func(40, 2)
        
        self.assertEqual(result, 42)
        mock_manager.execute_with_recovery.assert_called_once()
        
        # Check that function arguments were passed correctly
        args, kwargs = mock_manager.execute_with_recovery.call_args
        self.assertEqual(args[0].__name__, "test_func")  # Function
        self.assertEqual(args[1], "test_operation")  # Operation name
        self.assertEqual(args[2], "default")  # Retry category
        self.assertTrue(args[3])  # Use circuit breaker
        self.assertIsNone(args[4])  # Fallback value
        self.assertIsInstance(args[5], dict)  # Context
        self.assertEqual(args[6], 40)  # arg1
        self.assertEqual(kwargs["arg2"], 2)  # arg2
    
    @patch('stock_analysis.utils.error_recovery.validate_data_source_response')
    def test_with_data_validation_decorator(self, mock_validate):
        """Test with_data_validation decorator."""
        mock_validate.return_value = (True, {})
        
        @with_data_validation("security_info")
        def get_security_info(symbol):
            return {"symbol": symbol, "name": "Test Company", "current_price": 100.0}
        
        result = get_security_info("AAPL")
        
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["current_price"], 100.0)
        mock_validate.assert_called_once()
        
        # Check that validation was called with correct arguments
        args, kwargs = mock_validate.call_args
        self.assertEqual(args[0], "test_error_recovery")  # Source name (module name)
        self.assertEqual(args[1], "security_info")  # Data type
        self.assertEqual(args[2]["symbol"], "AAPL")  # Result data
        self.assertEqual(args[3], "AAPL")  # Symbol
    
    @patch('stock_analysis.utils.error_recovery.validate_data_source_response')
    def test_with_data_validation_decorator_strict(self, mock_validate):
        """Test with_data_validation decorator in strict mode."""
        mock_validate.return_value = (False, {
            "is_valid": False,
            "missing_fields": ["current_price"],
            "invalid_values": {}
        })
        
        @with_data_validation("security_info", strict=True)
        def get_security_info(symbol):
            return {"symbol": symbol, "name": "Test Company"}  # Missing current_price
        
        with self.assertRaises(ValidationError):
            get_security_info("AAPL")
        
        mock_validate.assert_called_once()


if __name__ == '__main__':
    unittest.main()