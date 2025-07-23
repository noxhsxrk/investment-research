"""Error recovery mechanisms for robust operation handling.

This module provides utilities for handling errors, implementing retry logic,
circuit breakers, and fallback mechanisms for robust operation handling.
"""

import time
import random
import functools
import inspect
import traceback
import sys
from typing import Callable, Any, Optional, List, Dict, Type, Union, Set, Tuple
from datetime import datetime, timedelta
import asyncio

from .exceptions import (
    StockAnalysisError, DataRetrievalError, NetworkError, 
    RateLimitError, InsufficientDataError, ValidationError
)
from .logging import get_logger, get_operation_logger
from .performance_metrics import get_metrics_collector, PerformanceMonitor
from .cache_manager import get_cache_manager

logger = get_logger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        backoff_strategy: str = 'exponential'  # 'exponential', 'linear', 'fixed'
    ):
        """Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
            backoff_strategy: Strategy for calculating delays
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.backoff_strategy = backoff_strategy
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        if self.backoff_strategy == 'exponential':
            delay = self.base_delay * (self.exponential_base ** attempt)
        elif self.backoff_strategy == 'linear':
            delay = self.base_delay * (attempt + 1)
        else:  # fixed
            delay = self.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.max_delay)
        
        # Add jitter if enabled
        if self.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)


class CircuitBreaker:
    """Circuit breaker pattern implementation for preventing cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
                logger.info("Circuit breaker transitioning to HALF_OPEN state")
            else:
                raise StockAnalysisError(
                    f"Circuit breaker is OPEN. Recovery timeout: {self.recovery_timeout}s",
                    error_code="CIRCUIT_BREAKER_OPEN"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.utcnow() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            logger.info("Circuit breaker reset to CLOSED state")
        
        self.failure_count = 0
        self.last_failure_time = None
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class ErrorRecoveryManager:
    """Manages error recovery strategies for different types of operations."""
    
    def __init__(self):
        """Initialize error recovery manager."""
        self.retry_configs: Dict[str, RetryConfig] = {
            'default': RetryConfig(),
            'data_retrieval': RetryConfig(max_attempts=5, base_delay=2.0, max_delay=30.0),
            'network': RetryConfig(max_attempts=3, base_delay=1.0, max_delay=10.0),
            'calculation': RetryConfig(max_attempts=2, base_delay=0.5, max_delay=5.0),
            'export': RetryConfig(max_attempts=3, base_delay=1.0, max_delay=15.0),
            'api': RetryConfig(max_attempts=4, base_delay=1.5, max_delay=20.0),
            'database': RetryConfig(max_attempts=3, base_delay=1.0, max_delay=15.0),
            'validation': RetryConfig(max_attempts=2, base_delay=0.5, max_delay=5.0)
        }
        
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_handlers: Dict[Type[Exception], Callable] = {}
        self.recovery_registry = ErrorRecoveryRegistry()
        self.error_diagnostics = ErrorDiagnostics()
        self.transient_detector = TransientErrorDetector()
        
        # Register default fallback handlers
        self._register_default_fallbacks()
        
        # Register default recovery strategies
        self._register_default_recovery_strategies()
        
        self.logger = get_logger(f"{__name__}.ErrorRecoveryManager")
    
    def _register_default_fallbacks(self):
        """Register default fallback handlers for common exceptions."""
        self.fallback_handlers[DataRetrievalError] = self._handle_data_retrieval_fallback
        self.fallback_handlers[NetworkError] = self._handle_network_fallback
        self.fallback_handlers[RateLimitError] = self._handle_rate_limit_fallback
        self.fallback_handlers[InsufficientDataError] = self._handle_insufficient_data_fallback
        self.fallback_handlers[ValidationError] = self._handle_validation_fallback
    
    def _register_default_recovery_strategies(self):
        """Register default recovery strategies for common operation patterns."""
        # Data retrieval operations
        self.recovery_registry.register_recovery_strategy(
            'get_*',
            retry_config=self.retry_configs['data_retrieval'],
            use_circuit_breaker=True
        )
        
        # API operations
        self.recovery_registry.register_recovery_strategy(
            '*_api_*',
            retry_config=self.retry_configs['api'],
            use_circuit_breaker=True
        )
        
        # Database operations
        self.recovery_registry.register_recovery_strategy(
            '*_db_*',
            retry_config=self.retry_configs['database'],
            use_circuit_breaker=True
        )
        
        # Validation operations
        self.recovery_registry.register_recovery_strategy(
            'validate_*',
            retry_config=self.retry_configs['validation'],
            use_circuit_breaker=False
        )
    
    def get_circuit_breaker(self, operation_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Circuit breaker instance
        """
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreaker()
        return self.circuit_breakers[operation_name]
    
    def execute_with_recovery(
        self,
        func: Callable,
        operation_name: str,
        retry_category: str = 'default',
        use_circuit_breaker: bool = True,
        fallback_value: Any = None,
        context: Optional[Dict[str, Any]] = None,
        validate_result: bool = False,
        args: tuple = (),
        kwargs: dict = None
    ) -> Any:
        """Execute function with comprehensive error recovery.
        
        Args:
            func: Function to execute
            operation_name: Name of the operation for monitoring
            retry_category: Category for retry configuration
            use_circuit_breaker: Whether to use circuit breaker
            fallback_value: Value to return if all recovery attempts fail
            context: Additional context for monitoring
            validate_result: Whether to validate the result
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result or fallback value
        """
        # Get retry configuration
        retry_config = self.retry_configs.get(retry_category, self.retry_configs['default'])
        
        # Get circuit breaker if enabled
        circuit_breaker = self.get_circuit_breaker(operation_name) if use_circuit_breaker else None
        
        # Get metrics collector
        collector = get_metrics_collector()
        
        # Get operation logger
        op_logger = get_operation_logger(__name__, operation_name)
        
        # Initialize context if not provided
        if context is None:
            context = {}
        
        # Add operation metadata to context
        context.update({
            'operation_name': operation_name,
            'retry_category': retry_category,
            'use_circuit_breaker': use_circuit_breaker
        })
        
        # Start operation logging
        op_logger.start(**context)
        
        with PerformanceMonitor(collector, f"{operation_name}_with_recovery", context):
            last_exception = None
            last_diagnostics = None
            
            for attempt in range(retry_config.max_attempts):
                try:
                    # Execute with circuit breaker if enabled
                    if kwargs is None:
                        kwargs = {}
                    
                    if circuit_breaker:
                        result = circuit_breaker.call(func, *args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    if attempt > 0:
                        self.logger.info(f"Operation {operation_name} succeeded on attempt {attempt + 1}")
                    
                    # Validate result if requested
                    if validate_result:
                        self._validate_result(result, operation_name, context)
                    
                    # Log successful operation completion
                    op_logger.finish(success=True, attempt=attempt + 1)
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Diagnose the error
                    diagnostics = self.error_diagnostics.diagnose_error(e, operation_name)
                    last_diagnostics = diagnostics
                    
                    # Log the attempt with diagnostics
                    self.logger.warning(
                        f"Operation {operation_name} failed on attempt {attempt + 1}/{retry_config.max_attempts}: "
                        f"{diagnostics['exception_type']}: {diagnostics['exception_message']} "
                        f"(Category: {diagnostics['category']}, Transient: {diagnostics['is_transient']})"
                    )
                    
                    # Check if we should retry
                    if not self._should_retry(e, attempt, retry_config, diagnostics):
                        break
                    
                    # Calculate and apply delay
                    if attempt < retry_config.max_attempts - 1:
                        delay = retry_config.calculate_delay(attempt)
                        self.logger.info(f"Retrying {operation_name} in {delay:.2f} seconds...")
                        time.sleep(delay)
            
            # All attempts failed, try fallback
            self.logger.error(f"All retry attempts failed for {operation_name}: {last_exception}")
            
            # Log error diagnostics
            if last_diagnostics:
                self.error_diagnostics.log_error_diagnostics(last_diagnostics)
            
            # Try to handle with registered fallback
            fallback_result = self._try_fallback(last_exception, operation_name, context)
            if fallback_result is not None:
                op_logger.finish(success=True, used_fallback=True)
                return fallback_result
            
            # Return fallback value if provided
            if fallback_value is not None:
                self.logger.info(f"Returning fallback value for {operation_name}")
                op_logger.finish(success=True, used_fallback=True)
                return fallback_value
            
            # Log operation failure
            op_logger.finish(success=False, exception=last_exception)
            
            # Re-raise the last exception
            raise last_exception
    
    def _should_retry(
        self, 
        exception: Exception, 
        attempt: int, 
        retry_config: RetryConfig,
        diagnostics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Determine if an operation should be retried.
        
        Args:
            exception: Exception that occurred
            attempt: Current attempt number (0-based)
            retry_config: Retry configuration
            diagnostics: Error diagnostics
            
        Returns:
            True if should retry, False otherwise
        """
        # Don't retry if we've reached max attempts
        if attempt >= retry_config.max_attempts - 1:
            return False
        
        # Don't retry certain types of errors
        non_retryable_errors = (
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            IndexError,
            ValidationError
        )
        
        if isinstance(exception, non_retryable_errors):
            self.logger.debug(f"Not retrying due to non-retryable error type: {type(exception).__name__}")
            return False
        
        # Special handling for rate limit errors
        if isinstance(exception, RateLimitError):
            retry_after = getattr(exception, 'retry_after', None)
            if not retry_after and hasattr(exception, 'context'):
                retry_after = exception.context.get('retry_after')
            
            if retry_after and retry_after > retry_config.max_delay:
                self.logger.warning(f"Rate limit retry delay ({retry_after}s) exceeds max delay ({retry_config.max_delay}s)")
                return False
        
        # Use diagnostics to determine if error is transient
        if diagnostics and not diagnostics.get('is_transient', False):
            self.logger.debug(f"Not retrying due to non-transient error: {diagnostics.get('category', 'unknown')}")
            return False
        
        return True
    
    def _try_fallback(
        self, 
        exception: Exception, 
        operation_name: str, 
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Try to handle exception with registered fallback handler.
        
        Args:
            exception: Exception that occurred
            operation_name: Name of the operation
            context: Additional context
            
        Returns:
            Fallback result or None if no handler available
        """
        # First try operation-specific fallback handler from registry
        registry_handler = self.recovery_registry.get_fallback_handler(operation_name)
        if registry_handler:
            try:
                self.logger.info(f"Using registered fallback handler for {operation_name}")
                return registry_handler(exception, operation_name, context)
            except Exception as fallback_error:
                self.logger.error(f"Registered fallback handler failed: {fallback_error}")
        
        # Then try exception-type handlers
        exception_type = type(exception)
        
        # Look for exact match first
        if exception_type in self.fallback_handlers:
            try:
                self.logger.info(f"Using fallback handler for {exception_type.__name__}")
                return self.fallback_handlers[exception_type](exception, operation_name, context)
            except Exception as fallback_error:
                self.logger.error(f"Fallback handler failed: {fallback_error}")
        
        # Look for parent class matches
        for registered_type, handler in self.fallback_handlers.items():
            if isinstance(exception, registered_type):
                try:
                    self.logger.info(f"Using fallback handler for {registered_type.__name__}")
                    return handler(exception, operation_name, context)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback handler failed: {fallback_error}")
        
        return None
    
    def _validate_result(self, result: Any, operation_name: str, context: Dict[str, Any]) -> None:
        """Validate operation result using data quality validation.
        
        Args:
            result: Operation result
            operation_name: Name of the operation
            context: Operation context
            
        Raises:
            ValidationError: If validation fails
        """
        # Skip validation for None results or primitive types
        if result is None or isinstance(result, (str, int, float, bool)):
            return
        
        try:
            from .data_quality import get_data_quality_validator
            
            # Determine data type from operation name
            data_type = None
            if 'security_info' in operation_name:
                data_type = 'security_info'
            elif 'historical' in operation_name:
                data_type = 'historical_prices'
            elif 'financial_statements' in operation_name or 'financials' in operation_name:
                if 'income' in operation_name:
                    data_type = 'financial_statements_income'
                elif 'balance' in operation_name:
                    data_type = 'financial_statements_balance'
                elif 'cash' in operation_name:
                    data_type = 'financial_statements_cash'
                else:
                    data_type = 'financial_statements_income'  # Default
            elif 'technical' in operation_name:
                data_type = 'technical_indicators'
            elif 'market' in operation_name:
                data_type = 'market_data'
            elif 'news' in operation_name:
                data_type = 'news_items'
            
            # Skip validation if data type couldn't be determined
            if not data_type:
                return
            
            # Get symbol from context if available
            symbol = context.get('symbol')
            
            # Get validator and validate result
            validator = get_data_quality_validator(strict_mode=False)
            
            if data_type == 'security_info':
                validator.validate_security_info(symbol, result)
            elif data_type == 'historical_prices':
                validator.validate_historical_prices(symbol, result)
            elif data_type.startswith('financial_statements_'):
                statement_type = data_type.split('_')[-1]
                validator.validate_financial_statements(symbol, result, statement_type)
            elif data_type == 'technical_indicators':
                validator.validate_technical_indicators(symbol, result)
            elif data_type == 'market_data':
                validator.validate_market_data(result)
            elif data_type == 'news_items' and isinstance(result, list):
                validator.validate_news_items(result)
            
        except Exception as e:
            self.logger.warning(f"Result validation failed for {operation_name}: {str(e)}")
            # Don't raise the exception, just log it
    
    def _handle_validation_fallback(
        self, 
        exception: ValidationError, 
        operation_name: str, 
        context: Optional[Dict[str, Any]]
    ) -> Optional[Any]:
        """Handle validation errors with fallback strategies.
        
        Args:
            exception: Validation error
            operation_name: Name of the operation
            context: Additional context
            
        Returns:
            Fallback result or None
        """
        self.logger.info(f"Attempting validation fallback for {operation_name}")
        
        # For validation errors, we might return partial or default data
        field_name = exception.field_name if hasattr(exception, 'field_name') else None
        
        if 'security_info' in operation_name:
            # Return minimal security info
            symbol = context.get('symbol') if context else None
            if symbol:
                return {
                    'symbol': symbol,
                    'name': symbol,
                    'current_price': None,
                    '_warning': 'Fallback data due to validation error'
                }
        
        elif 'historical_prices' in operation_name:
            # Return empty DataFrame with correct columns
            import pandas as pd
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        elif 'financial_statements' in operation_name:
            # Return empty DataFrame
            import pandas as pd
            return pd.DataFrame()
        
        elif 'technical_indicators' in operation_name:
            # Return empty indicators
            return {'_warning': 'Fallback data due to validation error'}
        
        elif 'market_data' in operation_name:
            # Return minimal market data structure
            return {
                'indices': {},
                'commodities': {},
                'forex': {},
                '_warning': 'Fallback data due to validation error'
            }
        
        elif 'news' in operation_name:
            # Return empty news list
            return []
        
        return None
    
    def _handle_data_retrieval_fallback(
        self, 
        exception: DataRetrievalError, 
        operation_name: str, 
        context: Optional[Dict[str, Any]]
    ) -> Optional[Any]:
        """Handle data retrieval errors with fallback strategies."""
        logger.info(f"Attempting data retrieval fallback for {operation_name}")
        
        # Try to get data from cache
        symbol = exception.context.get('symbol')
        if symbol:
            cache = get_cache_manager()
            
            # Try to determine what type of data we're retrieving
            data_type = None
            if 'stock_info' in operation_name:
                data_type = 'stock_info'
                cache_key = f"stock_info:{symbol}"
            elif 'historical' in operation_name:
                data_type = 'historical_data'
                period = exception.context.get('period', '1y')
                interval = exception.context.get('interval', '1d')
                cache_key = f"historical_data:{symbol}:{period}:{interval}"
            elif 'financial' in operation_name or 'statement' in operation_name:
                data_type = 'financial_statements'
                statement_type = exception.context.get('statement_type', 'income')
                period = exception.context.get('period', 'annual')
                cache_key = f"financial_statements:{symbol}:{statement_type}:{period}"
            elif 'peer' in operation_name:
                data_type = 'peer_data'
                cache_key = f"peer_data:{symbol}"
            else:
                # Generic fallback
                cache_key = f"*:{symbol}:*"
                
            if data_type:
                # Try to get from cache with specific key
                cached_data = cache.get(cache_key)
                if cached_data is not None:
                    logger.info(f"Using cached {data_type} data for {symbol} as fallback")
                    return cached_data
                
            # If we couldn't find with specific key, try pattern matching
            if not data_type or cached_data is None:
                # This is a simplified approach - in a real implementation,
                # you might want to search more intelligently
                logger.info(f"No specific cached data found for {symbol}, checking all cache entries")
                
                # We can't directly search the cache with patterns in this implementation
                # In a real implementation, you might have a method to search by pattern
                # For now, we'll just return None
                
            logger.warning(f"No cached data available for fallback for {symbol}")
        
        return None
    
    def _handle_network_fallback(
        self, 
        exception: NetworkError, 
        operation_name: str, 
        context: Optional[Dict[str, Any]]
    ) -> Optional[Any]:
        """Handle network errors with fallback strategies."""
        logger.info(f"Attempting network fallback for {operation_name}")
        
        # For network errors, try to use cached data similar to data retrieval fallback
        return self._handle_data_retrieval_fallback(exception, operation_name, context)
    
    def _handle_rate_limit_fallback(
        self, 
        exception: RateLimitError, 
        operation_name: str, 
        context: Optional[Dict[str, Any]]
    ) -> Optional[Any]:
        """Handle rate limit errors with fallback strategies."""
        logger.info(f"Attempting rate limit fallback for {operation_name}")
        
        retry_after = exception.context.get('retry_after', 60)
        
        # If retry_after is reasonable, wait and try once more
        if retry_after <= 120:  # Max 2 minutes
            logger.info(f"Rate limited, waiting {retry_after} seconds before final attempt")
            time.sleep(retry_after)
            # Return None to indicate no fallback value, let caller retry
        
        return None
    
    def _handle_insufficient_data_fallback(
        self, 
        exception: InsufficientDataError, 
        operation_name: str, 
        context: Optional[Dict[str, Any]]
    ) -> Optional[Any]:
        """Handle insufficient data errors with fallback strategies."""
        logger.info(f"Attempting insufficient data fallback for {operation_name}")
        
        # Could implement:
        # 1. Use partial data with warnings
        # 2. Use industry averages
        # 3. Skip certain calculations
        
        return None
    
    def register_fallback_handler(
        self, 
        exception_type: Type[Exception], 
        handler: Callable[[Exception, str, Optional[Dict[str, Any]]], Any]
    ):
        """Register a custom fallback handler for an exception type.
        
        Args:
            exception_type: Type of exception to handle
            handler: Fallback handler function
        """
        self.fallback_handlers[exception_type] = handler
        logger.info(f"Registered fallback handler for {exception_type.__name__}")
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get statistics about error recovery operations.
        
        Returns:
            Dictionary with recovery statistics
        """
        stats = {
            'circuit_breakers': {},
            'retry_configs': {name: {
                'max_attempts': config.max_attempts,
                'base_delay': config.base_delay,
                'max_delay': config.max_delay,
                'backoff_strategy': config.backoff_strategy
            } for name, config in self.retry_configs.items()},
            'fallback_handlers': list(self.fallback_handlers.keys())
        }
        
        for name, breaker in self.circuit_breakers.items():
            stats['circuit_breakers'][name] = {
                'state': breaker.state,
                'failure_count': breaker.failure_count,
                'last_failure_time': breaker.last_failure_time.isoformat() if breaker.last_failure_time else None
            }
        
        return stats


# Global error recovery manager instance
_global_recovery_manager: Optional[ErrorRecoveryManager] = None


class TransientErrorDetector:
    """Utility for detecting transient errors that can be retried."""
    
    def __init__(self):
        """Initialize transient error detector."""
        # Common transient error patterns in exception messages
        self.transient_patterns = [
            'timeout', 'timed out',
            'connection reset', 'connection refused', 'connection aborted',
            'too many requests', 'rate limit', 'throttl',
            'service unavailable', '503', '429',
            'temporary failure', 'temporary error',
            'try again', 'retry',
            'socket error', 'network error',
            'server error', 'internal server error', '500',
            'bad gateway', '502',
            'gateway timeout', '504',
            'database is locked', 'deadlock',
            'resource temporarily unavailable'
        ]
        
        # Known transient exception types
        self.transient_exception_types = {
            'ConnectionError', 'ConnectionRefusedError', 'ConnectionResetError',
            'TimeoutError', 'socket.timeout', 'requests.exceptions.Timeout',
            'requests.exceptions.ConnectionError', 'requests.exceptions.ReadTimeout',
            'urllib3.exceptions.ReadTimeoutError', 'urllib3.exceptions.ConnectTimeoutError',
            'concurrent.futures._base.TimeoutError',
            'asyncio.TimeoutError',
            'RateLimitError', 'NetworkError'
        }
    
    def is_transient_error(self, exception: Exception) -> bool:
        """Determine if an exception represents a transient error.
        
        Args:
            exception: Exception to check
            
        Returns:
            True if the exception is likely transient, False otherwise
        """
        # Check exception type name
        exception_type_name = type(exception).__name__
        if exception_type_name in self.transient_exception_types:
            return True
        
        # Check for known transient error subclasses
        if isinstance(exception, (RateLimitError, NetworkError)):
            return True
        
        # Check exception message for transient patterns
        exception_message = str(exception).lower()
        for pattern in self.transient_patterns:
            if pattern in exception_message:
                return True
        
        # Check exception cause chain
        cause = getattr(exception, '__cause__', None) or getattr(exception, '__context__', None)
        if cause is not None and cause is not exception:
            return self.is_transient_error(cause)
        
        return False


class ErrorDiagnostics:
    """Utility for diagnosing and categorizing errors."""
    
    def __init__(self):
        """Initialize error diagnostics."""
        self.logger = get_logger(f"{__name__}.ErrorDiagnostics")
        self.transient_detector = TransientErrorDetector()
    
    def diagnose_error(self, exception: Exception, operation_name: str) -> Dict[str, Any]:
        """Diagnose an error and provide detailed information.
        
        Args:
            exception: Exception to diagnose
            operation_name: Name of the operation that failed
            
        Returns:
            Dictionary with diagnostic information
        """
        # Get exception details
        exception_type = type(exception).__name__
        exception_module = type(exception).__module__
        exception_message = str(exception)
        
        # Get traceback information
        tb_summary = self._get_traceback_summary(exception)
        
        # Determine if error is transient
        is_transient = self.transient_detector.is_transient_error(exception)
        
        # Determine error category
        if isinstance(exception, DataRetrievalError):
            category = "data_retrieval"
        elif isinstance(exception, ValidationError):
            category = "validation"
        elif isinstance(exception, (NetworkError, RateLimitError)):
            category = "network"
        elif "timeout" in exception_message.lower():
            category = "timeout"
        elif any(name in exception_type.lower() for name in ["value", "type", "attribute", "key"]):
            category = "programming"
        else:
            category = "unknown"
        
        # Determine severity
        if category in ["programming", "validation"]:
            severity = "high"  # Requires code changes
        elif is_transient:
            severity = "medium"  # May resolve with retries
        else:
            severity = "high"  # Persistent issue
        
        # Determine recommended action
        if is_transient:
            recommended_action = "retry"
        elif category == "programming":
            recommended_action = "fix_code"
        elif category == "validation":
            recommended_action = "fix_data"
        else:
            recommended_action = "investigate"
        
        # Create diagnostic report
        diagnostics = {
            "operation": operation_name,
            "exception_type": exception_type,
            "exception_module": exception_module,
            "exception_message": exception_message,
            "traceback_summary": tb_summary,
            "category": category,
            "is_transient": is_transient,
            "severity": severity,
            "recommended_action": recommended_action,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add context from StockAnalysisError
        if isinstance(exception, StockAnalysisError):
            diagnostics["error_code"] = exception.error_code
            diagnostics["context"] = exception.context
            diagnostics["original_exception"] = str(exception.original_exception) if exception.original_exception else None
        
        return diagnostics
    
    def _get_traceback_summary(self, exception: Exception) -> List[Dict[str, str]]:
        """Get a summary of the exception traceback.
        
        Args:
            exception: Exception to analyze
            
        Returns:
            List of dictionaries with file, line, function, and code information
        """
        tb_summary = []
        
        try:
            tb = traceback.extract_tb(sys.exc_info()[2] if sys.exc_info()[2] else exception.__traceback__)
            
            for frame in tb:
                tb_summary.append({
                    "file": frame.filename,
                    "line": frame.lineno,
                    "function": frame.name,
                    "code": frame.line
                })
        except Exception as e:
            self.logger.warning(f"Error extracting traceback: {str(e)}")
        
        return tb_summary
    
    def log_error_diagnostics(self, diagnostics: Dict[str, Any]) -> None:
        """Log error diagnostics.
        
        Args:
            diagnostics: Error diagnostic information
        """
        self.logger.error(
            f"Error in operation '{diagnostics['operation']}': "
            f"{diagnostics['exception_type']}: {diagnostics['exception_message']} "
            f"(Category: {diagnostics['category']}, Transient: {diagnostics['is_transient']})"
        )
        
        # Log detailed diagnostics at debug level
        self.logger.debug(f"Error diagnostics: {diagnostics}")


class ErrorRecoveryRegistry:
    """Registry for error recovery strategies."""
    
    def __init__(self):
        """Initialize error recovery registry."""
        self.logger = get_logger(f"{__name__}.ErrorRecoveryRegistry")
        self.recovery_strategies: Dict[str, Dict[str, Any]] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.error_handlers: Dict[str, Callable] = {}
    
    def register_recovery_strategy(
        self, 
        operation_pattern: str, 
        retry_config: Optional[RetryConfig] = None,
        use_circuit_breaker: bool = True,
        fallback_handler: Optional[Callable] = None,
        error_handler: Optional[Callable] = None
    ) -> None:
        """Register a recovery strategy for operations matching a pattern.
        
        Args:
            operation_pattern: Pattern to match operation names
            retry_config: Retry configuration
            use_circuit_breaker: Whether to use circuit breaker
            fallback_handler: Custom fallback handler
            error_handler: Custom error handler
        """
        self.recovery_strategies[operation_pattern] = {
            'retry_config': retry_config,
            'use_circuit_breaker': use_circuit_breaker
        }
        
        if fallback_handler:
            self.fallback_handlers[operation_pattern] = fallback_handler
        
        if error_handler:
            self.error_handlers[operation_pattern] = error_handler
        
        self.logger.info(f"Registered recovery strategy for '{operation_pattern}'")
    
    def get_strategy_for_operation(self, operation_name: str) -> Dict[str, Any]:
        """Get recovery strategy for an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Dictionary with recovery strategy
        """
        # First try exact match
        if operation_name in self.recovery_strategies:
            return self.recovery_strategies[operation_name]
        
        # Then try pattern matching
        for pattern, strategy in self.recovery_strategies.items():
            if self._match_pattern(operation_name, pattern):
                return strategy
        
        # Return default strategy
        return {
            'retry_config': None,
            'use_circuit_breaker': True
        }
    
    def get_fallback_handler(self, operation_name: str) -> Optional[Callable]:
        """Get fallback handler for an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Fallback handler function or None
        """
        # First try exact match
        if operation_name in self.fallback_handlers:
            return self.fallback_handlers[operation_name]
        
        # Then try pattern matching
        for pattern, handler in self.fallback_handlers.items():
            if self._match_pattern(operation_name, pattern):
                return handler
        
        return None
    
    def get_error_handler(self, operation_name: str) -> Optional[Callable]:
        """Get error handler for an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Error handler function or None
        """
        # First try exact match
        if operation_name in self.error_handlers:
            return self.error_handlers[operation_name]
        
        # Then try pattern matching
        for pattern, handler in self.error_handlers.items():
            if self._match_pattern(operation_name, pattern):
                return handler
        
        return None
    
    def _match_pattern(self, operation_name: str, pattern: str) -> bool:
        """Check if operation name matches a pattern.
        
        Args:
            operation_name: Name of the operation
            pattern: Pattern to match
            
        Returns:
            True if operation name matches pattern, False otherwise
        """
        # Simple wildcard matching
        if pattern.endswith('*'):
            return operation_name.startswith(pattern[:-1])
        elif pattern.startswith('*'):
            return operation_name.endswith(pattern[1:])
        elif '*' in pattern:
            prefix, suffix = pattern.split('*', 1)
            return operation_name.startswith(prefix) and operation_name.endswith(suffix)
        else:
            return operation_name == pattern


def get_error_recovery_manager() -> ErrorRecoveryManager:
    """Get the global error recovery manager instance."""
    global _global_recovery_manager
    if _global_recovery_manager is None:
        _global_recovery_manager = ErrorRecoveryManager()
    return _global_recovery_manager


def with_error_recovery(
    operation_name: str,
    retry_category: str = 'default',
    use_circuit_breaker: bool = True,
    fallback_value: Any = None,
    validate_result: bool = False
):
    """Decorator for adding error recovery to functions.
    
    Args:
        operation_name: Name of the operation
        retry_category: Category for retry configuration
        use_circuit_breaker: Whether to use circuit breaker
        fallback_value: Value to return if all recovery attempts fail
        validate_result: Whether to validate the result using data quality validation
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            recovery_manager = get_error_recovery_manager()
            
            # Create context with function arguments
            context = {
                'function_name': func.__name__,
                'module_name': func.__module__
            }
            
            # Add named arguments to context
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            for param_name, param_value in bound_args.arguments.items():
                # Only add simple types to context
                if isinstance(param_value, (str, int, float, bool)) or param_value is None:
                    context[param_name] = param_value
            
            result = recovery_manager.execute_with_recovery(
                func,
                operation_name,
                retry_category=retry_category,
                use_circuit_breaker=use_circuit_breaker,
                fallback_value=fallback_value,
                context=context,
                validate_result=validate_result,
                args=args,
                kwargs=kwargs
            )
            return result
        return wrapper
    return decorator


def with_data_validation(data_type: str, strict: bool = False):
    """Decorator for adding data validation to functions.
    
    Args:
        data_type: Type of data to validate
        strict: Whether to raise exceptions for validation failures
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from .data_quality import validate_data_source_response
            
            # Get the first argument as symbol if available
            symbol = args[0] if len(args) > 0 and isinstance(args[0], str) else None
            
            # Get function module name for source name
            source_name = func.__module__.split('.')[-1]
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Validate the result
            is_valid, validation_details = validate_data_source_response(
                source_name, data_type, result, symbol
            )
            
            # If strict mode and validation failed, raise exception
            if strict and not is_valid:
                missing = validation_details.get('missing_fields', [])
                invalid = validation_details.get('invalid_values', {})
                
                field_name = missing[0] if missing else list(invalid.keys())[0] if invalid else None
                field_value = None if not invalid else list(invalid.values())[0] if invalid else None
                
                raise ValidationError(
                    f"Data validation failed for {data_type}" + (f" ({symbol})" if symbol else ""),
                    field_name=field_name,
                    field_value=field_value,
                    validation_rule=f"{data_type}_validation"
                )
            
            return result
        return wrapper
    return decorator