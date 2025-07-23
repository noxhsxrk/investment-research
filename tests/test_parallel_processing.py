"""Tests for parallel processing utilities."""

import time
import unittest
from unittest.mock import patch, MagicMock
import multiprocessing

from stock_analysis.utils.parallel_processing import (
    ParallelExecutor, get_parallel_executor, parallel_map, parallel_batch, parallelize
)


class TestParallelProcessing(unittest.TestCase):
    """Test cases for parallel processing utilities."""
    
    def test_parallel_executor_initialization(self):
        """Test ParallelExecutor initialization."""
        # Test with default parameters
        executor = ParallelExecutor()
        self.assertEqual(executor.max_workers, multiprocessing.cpu_count() * 2)
        self.assertFalse(executor.use_processes)
        self.assertIsNone(executor.timeout)
        
        # Test with custom parameters
        executor = ParallelExecutor(max_workers=4, use_processes=True, timeout=30)
        self.assertEqual(executor.max_workers, 4)
        self.assertTrue(executor.use_processes)
        self.assertEqual(executor.timeout, 30)
    
    def test_parallel_map(self):
        """Test parallel map functionality."""
        executor = ParallelExecutor(max_workers=2)
        
        # Test with simple function
        def square(x):
            return x * x
        
        results = executor.map(square, [1, 2, 3, 4, 5])
        self.assertEqual(results, [1, 4, 9, 16, 25])
        
        # Test with empty list
        results = executor.map(square, [])
        self.assertEqual(results, [])
        
        # Test with single item
        results = executor.map(square, [10])
        self.assertEqual(results, [100])
    
    def test_parallel_map_with_progress(self):
        """Test parallel map with progress reporting."""
        executor = ParallelExecutor(max_workers=2)
        
        # Define test function
        def slow_square(x):
            time.sleep(0.01)
            return x * x
        
        # Mock progress callback
        progress_callback = MagicMock()
        
        # Execute with progress reporting
        results = executor.map_with_progress(
            slow_square, 
            [1, 2, 3, 4, 5],
            progress_callback=progress_callback
        )
        
        # Verify results
        self.assertEqual(results, [1, 4, 9, 16, 25])
        
        # Verify progress callback was called
        self.assertEqual(progress_callback.call_count, 5)
        progress_callback.assert_any_call(1, 5)
        progress_callback.assert_any_call(5, 5)
    
    def test_execute_batch(self):
        """Test executing a batch of different function calls."""
        executor = ParallelExecutor(max_workers=2)
        
        # Define test functions
        def add(a, b):
            return a + b
        
        def multiply(a, b):
            return a * b
        
        def square(x):
            return x * x
        
        # Create batch items
        batch_items = [
            (add, [1, 2], {}),
            (multiply, [3, 4], {}),
            (square, [5], {})
        ]
        
        # Execute batch
        results = executor.execute_batch(batch_items)
        
        # Verify results
        self.assertEqual(results, [3, 12, 25])
    
    def test_global_executor(self):
        """Test global executor instance."""
        executor1 = get_parallel_executor()
        executor2 = get_parallel_executor()
        
        # Verify singleton pattern
        self.assertIs(executor1, executor2)
    
    def test_parallel_map_function(self):
        """Test parallel_map function."""
        def square(x):
            return x * x
        
        # Test with default parameters
        results = parallel_map(square, [1, 2, 3, 4, 5])
        self.assertEqual(results, [1, 4, 9, 16, 25])
        
        # Test with custom parameters
        results = parallel_map(
            square, 
            [1, 2, 3, 4, 5],
            max_workers=2,
            timeout=30,
            operation_name="test_map",
            use_processes=False
        )
        self.assertEqual(results, [1, 4, 9, 16, 25])
    
    def test_parallel_batch_function(self):
        """Test parallel_batch function."""
        # Define test functions
        def add(a, b):
            return a + b
        
        def multiply(a, b):
            return a * b
        
        # Create batch items
        batch_items = [
            (add, [1, 2], {}),
            (multiply, [3, 4], {})
        ]
        
        # Test with default parameters
        results = parallel_batch(batch_items)
        self.assertEqual(results, [3, 12])
        
        # Test with custom parameters
        results = parallel_batch(
            batch_items,
            max_workers=2,
            timeout=30,
            operation_name="test_batch"
        )
        self.assertEqual(results, [3, 12])
    
    def test_parallelize_decorator(self):
        """Test parallelize decorator."""
        # Define a function that processes a batch
        @parallelize(max_workers=2, operation_name="test_parallelize")
        def process_batch(items):
            return [item * item for item in items]
        
        # Test the decorated function
        results = process_batch([1, 2, 3, 4, 5])
        self.assertEqual(results, [1, 4, 9, 16, 25])
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_exception_handling(self, mock_executor):
        """Test exception handling in parallel execution."""
        # Setup mock executor to raise an exception
        mock_future = MagicMock()
        mock_future.result.side_effect = ValueError("Test error")
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.submit.return_value = mock_future
        
        mock_executor.return_value = mock_executor_instance
        
        # Create executor and test function
        executor = ParallelExecutor(max_workers=1)
        
        def failing_function(x):
            raise ValueError("Test error")
        
        # Test exception propagation
        with self.assertRaises(ValueError):
            executor.map(failing_function, [1])


if __name__ == '__main__':
    unittest.main()