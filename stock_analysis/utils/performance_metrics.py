"""Performance metrics collection and reporting system."""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self, success: bool = True, error_message: Optional[str] = None):
        """Mark the operation as finished."""
        self.end_time = datetime.utcnow()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.error_message = error_message
        
        # Capture system metrics if psutil is available
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                self.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                self.cpu_usage_percent = process.cpu_percent()
            except Exception as e:
                logger.warning(f"Failed to capture system metrics: {e}")
        else:
            logger.debug("psutil not available, skipping system metrics collection")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'operation_name': self.operation_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'success': self.success,
            'error_message': self.error_message,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'context': self.context
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for multiple operations."""
    operation_name: str
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_duration_seconds: float = 0.0
    average_duration_seconds: float = 0.0
    min_duration_seconds: Optional[float] = None
    max_duration_seconds: Optional[float] = None
    average_memory_usage_mb: Optional[float] = None
    average_cpu_usage_percent: Optional[float] = None
    error_rate_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert aggregated metrics to dictionary."""
        return {
            'operation_name': self.operation_name,
            'total_operations': self.total_operations,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'total_duration_seconds': self.total_duration_seconds,
            'average_duration_seconds': self.average_duration_seconds,
            'min_duration_seconds': self.min_duration_seconds,
            'max_duration_seconds': self.max_duration_seconds,
            'average_memory_usage_mb': self.average_memory_usage_mb,
            'average_cpu_usage_percent': self.average_cpu_usage_percent,
            'error_rate_percent': self.error_rate_percent
        }


class PerformanceMetricsCollector:
    """Collects and manages performance metrics."""
    
    def __init__(self, max_metrics_per_operation: int = 1000):
        """Initialize the metrics collector.
        
        Args:
            max_metrics_per_operation: Maximum number of metrics to keep per operation type
        """
        self.max_metrics_per_operation = max_metrics_per_operation
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics_per_operation))
        self.active_operations: Dict[str, OperationMetrics] = {}
        self._lock = threading.Lock()
        
        logger.info("Performance metrics collector initialized")
    
    def start_operation(
        self, 
        operation_name: str, 
        operation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start tracking an operation.
        
        Args:
            operation_name: Name of the operation
            operation_id: Unique identifier for this operation instance
            context: Additional context information
            
        Returns:
            Operation ID for tracking
        """
        if operation_id is None:
            operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        metrics = OperationMetrics(
            operation_name=operation_name,
            start_time=datetime.utcnow(),
            context=context or {}
        )
        
        with self._lock:
            self.active_operations[operation_id] = metrics
        
        logger.debug(f"Started tracking operation: {operation_name} (ID: {operation_id})")
        return operation_id
    
    def finish_operation(
        self, 
        operation_id: str, 
        success: bool = True, 
        error_message: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """Finish tracking an operation.
        
        Args:
            operation_id: Operation ID returned by start_operation
            success: Whether the operation was successful
            error_message: Error message if operation failed
            additional_context: Additional context to add
        """
        with self._lock:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation ID not found: {operation_id}")
                return
            
            metrics = self.active_operations.pop(operation_id)
            
            # Add additional context
            if additional_context:
                metrics.context.update(additional_context)
            
            # Finish the operation
            metrics.finish(success=success, error_message=error_message)
            
            # Store the metrics
            self.metrics[metrics.operation_name].append(metrics)
        
        logger.debug(f"Finished tracking operation: {metrics.operation_name} "
                    f"(Duration: {metrics.duration_seconds:.3f}s, Success: {success})")
    
    def get_operation_metrics(self, operation_name: str) -> List[OperationMetrics]:
        """Get all metrics for a specific operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            List of operation metrics
        """
        with self._lock:
            return list(self.metrics.get(operation_name, []))
    
    def get_aggregated_metrics(self, operation_name: str) -> Optional[AggregatedMetrics]:
        """Get aggregated metrics for a specific operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Aggregated metrics or None if no data available
        """
        metrics_list = self.get_operation_metrics(operation_name)
        if not metrics_list:
            return None
        
        # Calculate aggregated metrics
        total_ops = len(metrics_list)
        successful_ops = sum(1 for m in metrics_list if m.success)
        failed_ops = total_ops - successful_ops
        
        durations = [m.duration_seconds for m in metrics_list if m.duration_seconds is not None]
        memory_usages = [m.memory_usage_mb for m in metrics_list if m.memory_usage_mb is not None]
        cpu_usages = [m.cpu_usage_percent for m in metrics_list if m.cpu_usage_percent is not None]
        
        aggregated = AggregatedMetrics(
            operation_name=operation_name,
            total_operations=total_ops,
            successful_operations=successful_ops,
            failed_operations=failed_ops,
            error_rate_percent=(failed_ops / total_ops * 100) if total_ops > 0 else 0.0
        )
        
        if durations:
            aggregated.total_duration_seconds = sum(durations)
            aggregated.average_duration_seconds = sum(durations) / len(durations)
            aggregated.min_duration_seconds = min(durations)
            aggregated.max_duration_seconds = max(durations)
        
        if memory_usages:
            aggregated.average_memory_usage_mb = sum(memory_usages) / len(memory_usages)
        
        if cpu_usages:
            aggregated.average_cpu_usage_percent = sum(cpu_usages) / len(cpu_usages)
        
        return aggregated
    
    def get_all_aggregated_metrics(self) -> Dict[str, AggregatedMetrics]:
        """Get aggregated metrics for all operations.
        
        Returns:
            Dictionary mapping operation names to aggregated metrics
        """
        result = {}
        with self._lock:
            operation_names = list(self.metrics.keys())
        
        for operation_name in operation_names:
            aggregated = self.get_aggregated_metrics(operation_name)
            if aggregated:
                result[operation_name] = aggregated
        
        return result
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report.
        
        Returns:
            Performance report dictionary
        """
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_info': self._get_system_info(),
            'operations': {}
        }
        
        # Add aggregated metrics for each operation
        for operation_name, aggregated in self.get_all_aggregated_metrics().items():
            report['operations'][operation_name] = aggregated.to_dict()
        
        # Add currently active operations
        with self._lock:
            active_ops = []
            for op_id, metrics in self.active_operations.items():
                duration = (datetime.utcnow() - metrics.start_time).total_seconds()
                active_ops.append({
                    'operation_id': op_id,
                    'operation_name': metrics.operation_name,
                    'duration_seconds': duration,
                    'context': metrics.context
                })
            report['active_operations'] = active_ops
        
        return report
    
    def export_metrics(self, file_path: str, operation_name: Optional[str] = None):
        """Export metrics to a JSON file.
        
        Args:
            file_path: Path to export file
            operation_name: Specific operation to export (None for all)
        """
        try:
            if operation_name:
                metrics_list = self.get_operation_metrics(operation_name)
                data = {
                    'operation_name': operation_name,
                    'metrics': [m.to_dict() for m in metrics_list],
                    'aggregated': self.get_aggregated_metrics(operation_name).to_dict()
                }
            else:
                data = self.generate_performance_report()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Metrics exported to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def clear_metrics(self, operation_name: Optional[str] = None):
        """Clear stored metrics.
        
        Args:
            operation_name: Specific operation to clear (None for all)
        """
        with self._lock:
            if operation_name:
                if operation_name in self.metrics:
                    self.metrics[operation_name].clear()
                    logger.info(f"Cleared metrics for operation: {operation_name}")
            else:
                self.metrics.clear()
                logger.info("Cleared all metrics")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        if not PSUTIL_AVAILABLE:
            logger.debug("psutil not available, returning empty system info")
            return {'psutil_available': False}
        
        try:
            return {
                'psutil_available': True,
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                'memory_percent': psutil.virtual_memory().percent,
                'cpu_percent': psutil.cpu_percent(interval=1),
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
        except Exception as e:
            logger.warning(f"Failed to get system info: {e}")
            return {'psutil_available': True, 'error': str(e)}


class PerformanceMonitor:
    """Context manager for monitoring operation performance."""
    
    def __init__(
        self, 
        collector: PerformanceMetricsCollector,
        operation_name: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Initialize the performance monitor.
        
        Args:
            collector: Metrics collector instance
            operation_name: Name of the operation to monitor
            context: Additional context information
        """
        self.collector = collector
        self.operation_name = operation_name
        self.context = context or {}
        self.operation_id: Optional[str] = None
    
    def __enter__(self):
        """Start monitoring the operation."""
        self.operation_id = self.collector.start_operation(
            self.operation_name, 
            context=self.context
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finish monitoring the operation."""
        if self.operation_id:
            success = exc_type is None
            error_message = str(exc_val) if exc_val else None
            
            self.collector.finish_operation(
                self.operation_id,
                success=success,
                error_message=error_message
            )
    
    def add_context(self, key: str, value: Any):
        """Add additional context during operation."""
        if self.operation_id and self.operation_id in self.collector.active_operations:
            self.collector.active_operations[self.operation_id].context[key] = value


# Global metrics collector instance
_global_collector: Optional[PerformanceMetricsCollector] = None


def get_metrics_collector() -> PerformanceMetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = PerformanceMetricsCollector()
    return _global_collector


def monitor_performance(operation_name: str, context: Optional[Dict[str, Any]] = None):
    """Decorator for monitoring function performance.
    
    Args:
        operation_name: Name of the operation
        context: Additional context information
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            with PerformanceMonitor(collector, operation_name, context):
                return func(*args, **kwargs)
        return wrapper
    return decorator