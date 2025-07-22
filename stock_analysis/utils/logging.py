"""Enhanced logging configuration for stock analysis system."""

import logging
import logging.handlers
import os
import json
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from datetime import datetime, timezone
import threading

from .config import config


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log message as JSON string
        """
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': threading.current_thread().name,
            'process': os.getpid()
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, default=str)


class ContextualLogger:
    """Logger wrapper that adds contextual information to log messages."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize contextual logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
        self.context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """Set context information for subsequent log messages.
        
        Args:
            **kwargs: Context key-value pairs
        """
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context information."""
        self.context.clear()
    
    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Log message with context information.
        
        Args:
            level: Log level
            msg: Log message
            *args: Message formatting arguments
            **kwargs: Additional keyword arguments
        """
        # Create a copy of context to avoid modifying the original
        extra_fields = self.context.copy()
        
        # Add any extra fields from kwargs
        if 'extra' in kwargs:
            extra_fields.update(kwargs.pop('extra'))
        
        # Create custom LogRecord with extra fields
        if extra_fields:
            kwargs['extra'] = {'extra_fields': extra_fields}
        
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message with context."""
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message with context."""
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message with context."""
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        """Log exception message with context."""
        kwargs['exc_info'] = True
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)


class OperationLogger:
    """Logger for tracking operations with start/end events."""
    
    def __init__(self, logger: ContextualLogger, operation_name: str):
        """Initialize operation logger.
        
        Args:
            logger: Contextual logger instance
            operation_name: Name of the operation
        """
        self.logger = logger
        self.operation_name = operation_name
        self.start_time: Optional[datetime] = None
    
    def start(self, **context):
        """Log operation start.
        
        Args:
            **context: Additional context for the operation
        """
        self.start_time = datetime.now(timezone.utc)
        self.logger.set_context(
            operation=self.operation_name,
            operation_start=self.start_time.isoformat(),
            **context
        )
        self.logger.info(f"Starting operation: {self.operation_name}")
    
    def finish(self, success: bool = True, **context):
        """Log operation completion.
        
        Args:
            success: Whether the operation was successful
            **context: Additional context for the completion
        """
        end_time = datetime.now(timezone.utc)
        duration = (end_time - self.start_time).total_seconds() if self.start_time else None
        
        self.logger.set_context(
            operation_end=end_time.isoformat(),
            operation_duration_seconds=duration,
            operation_success=success,
            **context
        )
        
        if success:
            self.logger.info(f"Completed operation: {self.operation_name} (Duration: {duration:.3f}s)")
        else:
            self.logger.error(f"Failed operation: {self.operation_name} (Duration: {duration:.3f}s)")
    
    def progress(self, message: str, **context):
        """Log operation progress.
        
        Args:
            message: Progress message
            **context: Additional context
        """
        self.logger.set_context(**context)
        self.logger.info(f"[{self.operation_name}] {message}")


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    max_file_size: Optional[str] = None,
    backup_count: Optional[int] = None,
    structured_logging: bool = False,
    enable_performance_logging: bool = True
) -> logging.Logger:
    """Set up enhanced logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        max_file_size: Maximum log file size (e.g., '10MB')
        backup_count: Number of backup log files to keep
        structured_logging: Whether to use structured JSON logging
        enable_performance_logging: Whether to enable performance logging
        
    Returns:
        Configured logger instance
    """
    # Get configuration values
    log_level = log_level or config.get('stock_analysis.logging.level', 'INFO')
    log_file = log_file or config.get('stock_analysis.logging.file_path', './logs/stock_analysis.log')
    max_file_size = max_file_size or config.get('stock_analysis.logging.max_file_size', '10MB')
    backup_count = backup_count or config.get('stock_analysis.logging.backup_count', 5)
    structured_logging = structured_logging or config.get('stock_analysis.logging.structured', False)
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger('stock_analysis')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    if structured_logging:
        formatter = StructuredFormatter()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = formatter
    
    # Console handler (always use simple format for console)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        max_bytes = _parse_file_size(max_file_size)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Performance logging handler (separate file)
    if enable_performance_logging:
        perf_log_file = log_file.replace('.log', '_performance.log') if log_file else './logs/performance.log'
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(StructuredFormatter())
        
        # Create performance logger
        perf_logger = logging.getLogger('stock_analysis.performance')
        perf_logger.setLevel(logging.INFO)
        perf_logger.addHandler(perf_handler)
        perf_logger.propagate = False  # Don't propagate to root logger
    
    # Error logging handler (separate file for errors)
    error_log_file = log_file.replace('.log', '_errors.log') if log_file else './logs/errors.log'
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(StructuredFormatter())
    logger.addHandler(error_handler)
    
    return logger


def get_logger(name: str) -> ContextualLogger:
    """Get a contextual logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Contextual logger instance
    """
    base_logger = logging.getLogger(f'stock_analysis.{name}')
    return ContextualLogger(base_logger)


def get_operation_logger(name: str, operation_name: str) -> OperationLogger:
    """Get an operation logger for tracking operation lifecycle.
    
    Args:
        name: Logger name (typically __name__)
        operation_name: Name of the operation to track
        
    Returns:
        Operation logger instance
    """
    contextual_logger = get_logger(name)
    return OperationLogger(contextual_logger, operation_name)


def get_performance_logger() -> logging.Logger:
    """Get the performance logger instance.
    
    Returns:
        Performance logger instance
    """
    return logging.getLogger('stock_analysis.performance')


def log_api_call(
    logger: ContextualLogger,
    api_name: str,
    endpoint: str,
    parameters: Optional[Dict[str, Any]] = None,
    response_time: Optional[float] = None,
    status_code: Optional[int] = None,
    error: Optional[Exception] = None
):
    """Log API call details.
    
    Args:
        logger: Logger instance
        api_name: Name of the API
        endpoint: API endpoint
        parameters: Request parameters
        response_time: Response time in seconds
        status_code: HTTP status code
        error: Exception if call failed
    """
    context = {
        'api_name': api_name,
        'endpoint': endpoint,
        'parameters': parameters or {},
        'response_time_seconds': response_time,
        'status_code': status_code
    }
    
    if error:
        logger.set_context(**context)
        logger.error(f"API call failed: {api_name} - {endpoint}", exc_info=True)
    else:
        logger.set_context(**context)
        logger.info(f"API call successful: {api_name} - {endpoint}")


def log_data_quality_issue(
    logger: ContextualLogger,
    symbol: str,
    data_type: str,
    issue_description: str,
    severity: str = 'warning',
    missing_fields: Optional[List[str]] = None,
    invalid_values: Optional[Dict[str, Any]] = None
):
    """Log data quality issues.
    
    Args:
        logger: Logger instance
        symbol: Stock symbol
        data_type: Type of data (e.g., 'stock_info', 'financial_statements')
        issue_description: Description of the issue
        severity: Severity level ('info', 'warning', 'error')
        missing_fields: List of missing fields
        invalid_values: Dictionary of invalid values
    """
    context = {
        'symbol': symbol,
        'data_type': data_type,
        'data_quality_issue': True,
        'missing_fields': missing_fields or [],
        'invalid_values': invalid_values or {}
    }
    
    logger.set_context(**context)
    
    if severity == 'error':
        logger.error(f"Data quality error for {symbol} ({data_type}): {issue_description}")
    elif severity == 'warning':
        logger.warning(f"Data quality warning for {symbol} ({data_type}): {issue_description}")
    else:
        logger.info(f"Data quality info for {symbol} ({data_type}): {issue_description}")


def _parse_file_size(size_str: str) -> int:
    """Parse file size string to bytes.
    
    Args:
        size_str: Size string (e.g., '10MB', '1GB')
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper().strip()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        # Assume bytes if no unit specified
        return int(size_str)


# Initialize logging on module import
setup_logging()