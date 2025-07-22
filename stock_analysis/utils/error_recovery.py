"""Error recovery mechanisms for robust operation handling."""

import time
import random
import functools
from typing import Callable, Any, Optional, List, Dict, Type, Union
from datetime import datetime, timedelta
import asyncio

from .exceptions import (
    StockAnalysisError, DataRetrievalError, NetworkError, 
    RateLimitError, InsufficientDataError
)
from .logging import get_logger
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
            'export': RetryConfig(max_attempts=3, base_delay=1.0, max_delay=15.0)
        }
        
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_handlers: Dict[Type[Exception], Callable] = {}
        
        # Register default fallback handlers
        self._register_default_fallbacks()
    
    def _register_default_fallbacks(self):
        """Register default fallback handlers for common exceptions."""
        self.fallback_handlers[DataRetrievalError] = self._handle_data_retrieval_fallback
        self.fallback_handlers[NetworkError] = self._handle_network_fallback
        self.fallback_handlers[RateLimitError] = self._handle_rate_limit_fallback
        self.fallback_handlers[InsufficientDataError] = self._handle_insufficient_data_fallback
    
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
        *args,
        **kwargs
    ) -> Any:
        """Execute function with comprehensive error recovery.
        
        Args:
            func: Function to execute
            operation_name: Name of the operation for monitoring
            retry_category: Category for retry configuration
            use_circuit_breaker: Whether to use circuit breaker
            fallback_value: Value to return if all recovery attempts fail
            context: Additional context for monitoring
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result or fallback value
        """
        retry_config = self.retry_configs.get(retry_category, self.retry_configs['default'])
        circuit_breaker = self.get_circuit_breaker(operation_name) if use_circuit_breaker else None
        
        collector = get_metrics_collector()
        
        with PerformanceMonitor(collector, f"{operation_name}_with_recovery", context):
            last_exception = None
            
            for attempt in range(retry_config.max_attempts):
                try:
                    # Execute with circuit breaker if enabled
                    if circuit_breaker:
                        result = circuit_breaker.call(func, *args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    if attempt > 0:
                        logger.info(f"Operation {operation_name} succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Log the attempt
                    logger.warning(
                        f"Operation {operation_name} failed on attempt {attempt + 1}/{retry_config.max_attempts}: {e}"
                    )
                    
                    # Check if we should retry
                    if not self._should_retry(e, attempt, retry_config):
                        break
                    
                    # Calculate and apply delay
                    if attempt < retry_config.max_attempts - 1:
                        delay = retry_config.calculate_delay(attempt)
                        logger.info(f"Retrying {operation_name} in {delay:.2f} seconds...")
                        time.sleep(delay)
            
            # All attempts failed, try fallback
            logger.error(f"All retry attempts failed for {operation_name}: {last_exception}")
            
            # Try to handle with registered fallback
            fallback_result = self._try_fallback(last_exception, operation_name, context)
            if fallback_result is not None:
                return fallback_result
            
            # Return fallback value if provided
            if fallback_value is not None:
                logger.info(f"Returning fallback value for {operation_name}")
                return fallback_value
            
            # Re-raise the last exception
            raise last_exception
    
    def _should_retry(self, exception: Exception, attempt: int, retry_config: RetryConfig) -> bool:
        """Determine if an operation should be retried.
        
        Args:
            exception: Exception that occurred
            attempt: Current attempt number (0-based)
            retry_config: Retry configuration
            
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
        )
        
        if isinstance(exception, non_retryable_errors):
            logger.debug(f"Not retrying due to non-retryable error type: {type(exception).__name__}")
            return False
        
        # Special handling for rate limit errors
        if isinstance(exception, RateLimitError):
            retry_after = exception.context.get('retry_after')
            if retry_after and retry_after > retry_config.max_delay:
                logger.warning(f"Rate limit retry delay ({retry_after}s) exceeds max delay ({retry_config.max_delay}s)")
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
        exception_type = type(exception)
        
        # Look for exact match first
        if exception_type in self.fallback_handlers:
            try:
                return self.fallback_handlers[exception_type](exception, operation_name, context)
            except Exception as fallback_error:
                logger.error(f"Fallback handler failed: {fallback_error}")
        
        # Look for parent class matches
        for registered_type, handler in self.fallback_handlers.items():
            if isinstance(exception, registered_type):
                try:
                    return handler(exception, operation_name, context)
                except Exception as fallback_error:
                    logger.error(f"Fallback handler failed: {fallback_error}")
        
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
    fallback_value: Any = None
):
    """Decorator for adding error recovery to functions.
    
    Args:
        operation_name: Name of the operation
        retry_category: Category for retry configuration
        use_circuit_breaker: Whether to use circuit breaker
        fallback_value: Value to return if all recovery attempts fail
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            recovery_manager = get_error_recovery_manager()
            return recovery_manager.execute_with_recovery(
                func,
                operation_name,
                retry_category,
                use_circuit_breaker,
                fallback_value,
                None,  # context
                *args,
                **kwargs
            )
        return wrapper
    return decorator