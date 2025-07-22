"""Custom exceptions for stock analysis system."""

import traceback
from typing import Optional, Dict, Any
from datetime import datetime, timezone


class StockAnalysisError(Exception):
    """Base exception for stock analysis errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """Initialize the exception with enhanced error information.
        
        Args:
            message: Human-readable error message
            error_code: Unique error code for categorization
            context: Additional context information
            original_exception: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.context = context or {}
        self.original_exception = original_exception
        self.timestamp = datetime.now(timezone.utc)
        self.traceback_str = traceback.format_exc() if original_exception else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'original_exception': str(self.original_exception) if self.original_exception else None,
            'traceback': self.traceback_str
        }
    
    def __str__(self) -> str:
        """String representation with enhanced information."""
        base_msg = f"[{self.error_code}] {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg += f" (Context: {context_str})"
        return base_msg


class DataRetrievalError(StockAnalysisError):
    """Raised when data cannot be retrieved from external sources."""
    
    def __init__(
        self, 
        message: str, 
        symbol: Optional[str] = None,
        data_source: Optional[str] = None,
        retry_count: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if symbol:
            context['symbol'] = symbol
        if data_source:
            context['data_source'] = data_source
        if retry_count is not None:
            context['retry_count'] = retry_count
        
        super().__init__(
            message, 
            error_code="DATA_RETRIEVAL_ERROR",
            context=context,
            **kwargs
        )


class CalculationError(StockAnalysisError):
    """Raised when financial calculations fail."""
    
    def __init__(
        self, 
        message: str, 
        symbol: Optional[str] = None,
        calculation_type: Optional[str] = None,
        missing_data: Optional[list] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if symbol:
            context['symbol'] = symbol
        if calculation_type:
            context['calculation_type'] = calculation_type
        if missing_data:
            context['missing_data'] = missing_data
        
        super().__init__(
            message, 
            error_code="CALCULATION_ERROR",
            context=context,
            **kwargs
        )


class ExportError(StockAnalysisError):
    """Raised when export operations fail."""
    
    def __init__(
        self, 
        message: str, 
        export_format: Optional[str] = None,
        file_path: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if export_format:
            context['export_format'] = export_format
        if file_path:
            context['file_path'] = file_path
        
        super().__init__(
            message, 
            error_code="EXPORT_ERROR",
            context=context,
            **kwargs
        )


class ConfigurationError(StockAnalysisError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(
        self, 
        message: str, 
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if config_key:
            context['config_key'] = config_key
        if config_file:
            context['config_file'] = config_file
        
        super().__init__(
            message, 
            error_code="CONFIGURATION_ERROR",
            context=context,
            **kwargs
        )


class ValidationError(StockAnalysisError):
    """Raised when data validation fails."""
    
    def __init__(
        self, 
        message: str, 
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        validation_rule: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if field_name:
            context['field_name'] = field_name
        if field_value is not None:
            context['field_value'] = str(field_value)
        if validation_rule:
            context['validation_rule'] = validation_rule
        
        super().__init__(
            message, 
            error_code="VALIDATION_ERROR",
            context=context,
            **kwargs
        )


class SchedulingError(StockAnalysisError):
    """Raised when scheduling operations fail."""
    
    def __init__(
        self, 
        message: str, 
        schedule_type: Optional[str] = None,
        schedule_interval: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if schedule_type:
            context['schedule_type'] = schedule_type
        if schedule_interval:
            context['schedule_interval'] = schedule_interval
        
        super().__init__(
            message, 
            error_code="SCHEDULING_ERROR",
            context=context,
            **kwargs
        )


class NetworkError(DataRetrievalError):
    """Raised when network-related errors occur."""
    
    def __init__(
        self, 
        message: str, 
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if url:
            context['url'] = url
        if status_code:
            context['status_code'] = status_code
        
        super().__init__(
            message, 
            error_code="NETWORK_ERROR",
            context=context,
            **kwargs
        )


class RateLimitError(DataRetrievalError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(
        self, 
        message: str, 
        retry_after: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if retry_after:
            context['retry_after'] = retry_after
        
        super().__init__(
            message, 
            error_code="RATE_LIMIT_ERROR",
            context=context,
            **kwargs
        )


class InsufficientDataError(CalculationError):
    """Raised when insufficient data is available for calculations."""
    
    def __init__(
        self, 
        message: str, 
        required_data: Optional[list] = None,
        available_data: Optional[list] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if required_data:
            context['required_data'] = required_data
        if available_data:
            context['available_data'] = available_data
        
        super().__init__(
            message, 
            error_code="INSUFFICIENT_DATA_ERROR",
            context=context,
            **kwargs
        )