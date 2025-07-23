"""Parallel processing utilities for batch operations.

This module provides utilities for executing batch operations in parallel
to improve performance for data-intensive tasks.
"""

import time
import threading
import multiprocessing
from typing import List, Dict, Any, Callable, TypeVar, Generic, Iterable, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
import functools

from stock_analysis.utils.logging import get_logger
from stock_analysis.utils.performance_metrics import get_metrics_collector, PerformanceMonitor
from stock_analysis.utils.config import config

logger = get_logger(__name__)

# Type variables for generic functions
T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type


class ParallelExecutor:
    """Executor for parallel batch operations."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False,
        timeout: Optional[float] = None
    ):
        """Initialize parallel executor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            use_processes: Whether to use processes instead of threads
            timeout: Default timeout for operations in seconds
        """
        self.use_processes = use_processes
        
        # Determine max workers if not specified
        if max_workers is None:
            if use_processes:
                # Use CPU count for CPU-bound tasks
                max_workers = multiprocessing.cpu_count()
            else:
                # Use 2x CPU count for I/O-bound tasks
                max_workers = multiprocessing.cpu_count() * 2
        
        self.max_workers = max_workers
        self.timeout = timeout
        self.logger = get_logger(f"{__name__}.ParallelExecutor")
        self.metrics_collector = get_metrics_collector()
    
    def map(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        timeout: Optional[float] = None,
        operation_name: str = "parallel_map",
        chunksize: int = 1
    ) -> List[R]:
        """Execute a function on each item in parallel.
        
        Args:
            func: Function to execute on each item
            items: Iterable of items to process
            timeout: Timeout for each operation in seconds
            operation_name: Name for performance monitoring
            chunksize: Number of items per chunk for process pools
            
        Returns:
            List of results in the same order as input items
            
        Raises:
            TimeoutError: If operation times out
            Exception: If any worker raises an exception
        """
        timeout = timeout or self.timeout
        items_list = list(items)  # Convert to list to ensure we can index
        
        if not items_list:
            return []
        
        # Skip parallel execution for small batches
        if len(items_list) == 1:
            with PerformanceMonitor(self.metrics_collector, f"{operation_name}_single"):
                return [func(items_list[0])]
        
        # Use appropriate executor based on configuration
        executor_cls = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with PerformanceMonitor(self.metrics_collector, operation_name, 
                               {"batch_size": len(items_list), "workers": self.max_workers}):
            results = []
            
            with executor_cls(max_workers=self.max_workers) as executor:
                # Submit all tasks
                if self.use_processes:
                    future_to_index = {
                        executor.submit(func, item): i 
                        for i, item in enumerate(items_list)
                    }
                else:
                    future_to_index = {
                        executor.submit(func, item): i 
                        for i, item in enumerate(items_list)
                    }
                
                # Initialize results list with None placeholders
                results = [None] * len(items_list)
                
                # Process results as they complete
                for future in as_completed(future_to_index, timeout=timeout):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        self.logger.error(f"Error in parallel task {index}: {str(e)}")
                        raise
            
            return results
    
    def map_with_progress(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        timeout: Optional[float] = None,
        operation_name: str = "parallel_map_progress"
    ) -> List[R]:
        """Execute a function on each item in parallel with progress reporting.
        
        Args:
            func: Function to execute on each item
            items: Iterable of items to process
            progress_callback: Function to call with progress updates (completed, total)
            timeout: Timeout for each operation in seconds
            operation_name: Name for performance monitoring
            
        Returns:
            List of results in the same order as input items
            
        Raises:
            TimeoutError: If operation times out
            Exception: If any worker raises an exception
        """
        timeout = timeout or self.timeout
        items_list = list(items)
        total = len(items_list)
        
        if not items_list:
            return []
        
        # Skip parallel execution for small batches
        if total == 1:
            with PerformanceMonitor(self.metrics_collector, f"{operation_name}_single"):
                result = [func(items_list[0])]
                if progress_callback:
                    progress_callback(1, 1)
                return result
        
        # Use appropriate executor based on configuration
        executor_cls = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with PerformanceMonitor(self.metrics_collector, operation_name, 
                               {"batch_size": total, "workers": self.max_workers}):
            results = [None] * total
            completed = 0
            
            with executor_cls(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(func, item): i 
                    for i, item in enumerate(items_list)
                }
                
                # Process results as they complete
                for future in as_completed(future_to_index, timeout=timeout):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        self.logger.error(f"Error in parallel task {index}: {str(e)}")
                        raise
                    
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)
            
            return results
    
    def execute_batch(
        self,
        batch_items: List[Tuple[Callable[..., R], List, Dict]],
        timeout: Optional[float] = None,
        operation_name: str = "parallel_batch"
    ) -> List[R]:
        """Execute a batch of different function calls in parallel.
        
        Args:
            batch_items: List of (function, args, kwargs) tuples
            timeout: Timeout for the entire batch in seconds
            operation_name: Name for performance monitoring
            
        Returns:
            List of results in the same order as input batch_items
            
        Raises:
            TimeoutError: If operation times out
            Exception: If any worker raises an exception
        """
        timeout = timeout or self.timeout
        
        if not batch_items:
            return []
        
        # Skip parallel execution for small batches
        if len(batch_items) == 1:
            func, args, kwargs = batch_items[0]
            with PerformanceMonitor(self.metrics_collector, f"{operation_name}_single"):
                return [func(*args, **kwargs)]
        
        # Use appropriate executor based on configuration
        executor_cls = ThreadPoolExecutor  # Always use threads for mixed function calls
        
        with PerformanceMonitor(self.metrics_collector, operation_name, 
                               {"batch_size": len(batch_items), "workers": self.max_workers}):
            results = []
            
            with executor_cls(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_index = {}
                for i, (func, args, kwargs) in enumerate(batch_items):
                    future = executor.submit(func, *args, **kwargs)
                    future_to_index[future] = i
                
                # Initialize results list with None placeholders
                results = [None] * len(batch_items)
                
                # Process results as they complete
                for future in as_completed(future_to_index, timeout=timeout):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        self.logger.error(f"Error in batch task {index}: {str(e)}")
                        raise
            
            return results


# Global parallel executor instance
_global_executor: Optional[ParallelExecutor] = None


def get_parallel_executor() -> ParallelExecutor:
    """Get the global parallel executor instance.
    
    Returns:
        Global ParallelExecutor instance
    """
    global _global_executor
    if _global_executor is None:
        max_workers = config.get('stock_analysis.parallel.max_workers', None)
        use_processes = config.get('stock_analysis.parallel.use_processes', False)
        timeout = config.get('stock_analysis.parallel.timeout', None)
        
        _global_executor = ParallelExecutor(
            max_workers=max_workers,
            use_processes=use_processes,
            timeout=timeout
        )
    
    return _global_executor


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
    operation_name: str = "parallel_map",
    use_processes: bool = False
) -> List[R]:
    """Execute a function on each item in parallel.
    
    Args:
        func: Function to execute on each item
        items: Iterable of items to process
        max_workers: Maximum number of worker threads/processes
        timeout: Timeout for each operation in seconds
        operation_name: Name for performance monitoring
        use_processes: Whether to use processes instead of threads
        
    Returns:
        List of results in the same order as input items
    """
    if max_workers is not None or use_processes:
        # Create a custom executor if parameters differ from global
        executor = ParallelExecutor(
            max_workers=max_workers,
            use_processes=use_processes,
            timeout=timeout
        )
        return executor.map(func, items, timeout, operation_name)
    else:
        # Use global executor with default settings
        return get_parallel_executor().map(func, items, timeout, operation_name)


def parallel_batch(
    batch_items: List[Tuple[Callable[..., R], List, Dict]],
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
    operation_name: str = "parallel_batch"
) -> List[R]:
    """Execute a batch of different function calls in parallel.
    
    Args:
        batch_items: List of (function, args, kwargs) tuples
        max_workers: Maximum number of worker threads
        timeout: Timeout for the entire batch in seconds
        operation_name: Name for performance monitoring
        
    Returns:
        List of results in the same order as input batch_items
    """
    if max_workers is not None:
        # Create a custom executor if parameters differ from global
        executor = ParallelExecutor(
            max_workers=max_workers,
            use_processes=False,  # Always use threads for mixed function calls
            timeout=timeout
        )
        return executor.execute_batch(batch_items, timeout, operation_name)
    else:
        # Use global executor with default settings
        return get_parallel_executor().execute_batch(batch_items, timeout, operation_name)


def parallelize(
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
    operation_name: Optional[str] = None,
    use_processes: bool = False
):
    """Decorator to parallelize a function that processes a batch of items.
    
    The decorated function should take an iterable as its first argument.
    The decorator will replace this with a parallel implementation.
    
    Args:
        max_workers: Maximum number of worker threads/processes
        timeout: Timeout for each operation in seconds
        operation_name: Name for performance monitoring
        use_processes: Whether to use processes instead of threads
    """
    def decorator(func: Callable[[Iterable[T]], List[R]]):
        @functools.wraps(func)
        def wrapper(items: Iterable[T], *args, **kwargs):
            # Extract the item processing logic
            def process_item(item: T) -> R:
                return func([item], *args, **kwargs)[0]
            
            # Use operation name from parameter or function name
            op_name = operation_name or f"parallel_{func.__name__}"
            
            # Execute in parallel
            return parallel_map(
                process_item,
                items,
                max_workers=max_workers,
                timeout=timeout,
                operation_name=op_name,
                use_processes=use_processes
            )
        
        return wrapper
    
    return decorator