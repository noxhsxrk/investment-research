"""Tests for error monitoring module."""

import unittest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from stock_analysis.utils.error_monitoring import (
    ErrorMonitor,
    get_error_monitor,
    monitor_error,
    error_monitoring_decorator
)
from stock_analysis.utils.exceptions import (
    DataRetrievalError,
    NetworkError,
    ValidationError
)


class TestErrorMonitor(unittest.TestCase):
    """Test cases for ErrorMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test reports
        self.test_dir = tempfile.mkdtemp()
        self.monitor = ErrorMonitor(max_errors=100, output_dir=self.test_dir)
        
        # Add some test errors
        self.monitor.record_error(
            DataRetrievalError("Failed to retrieve data"),
            "get_security_info"
        )
        self.monitor.record_error(
            NetworkError("Connection timeout", url="https://api.example.com"),
            "api_request"
        )
        self.monitor.record_error(
            ValueError("Invalid parameter"),
            "calculate_metrics"
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_record_error(self):
        """Test recording errors."""
        # Record a new error
        self.monitor.record_error(
            ValidationError("Invalid data format"),
            "validate_data"
        )
        
        # Check that error was recorded
        self.assertEqual(len(self.monitor.errors), 4)
        
        # Check error details
        latest_error = self.monitor.errors[-1]
        self.assertEqual(latest_error['exception_type'], "ValidationError")
        self.assertEqual(latest_error['operation'], "validate_data")
        self.assertEqual(latest_error['category'], "validation")
    
    def test_get_error_count(self):
        """Test getting error count."""
        # Total count
        self.assertEqual(self.monitor.get_error_count(), 3)
        
        # Count with time window
        # All errors should be within the last hour
        count = self.monitor.get_error_count(timedelta(hours=1))
        self.assertEqual(count, 3)
        
        # No errors should be older than a week
        count = self.monitor.get_error_count(timedelta(days=7))
        self.assertEqual(count, 3)
    
    def test_get_error_rate(self):
        """Test getting error rate."""
        # Get error rate for the last hour
        rate = self.monitor.get_error_rate(timedelta(hours=1))
        
        self.assertEqual(rate['error_count'], 3)
        self.assertGreater(rate['error_rate_per_minute'], 0)
        self.assertGreater(rate['error_rate_per_hour'], 0)
    
    def test_get_most_common_errors(self):
        """Test getting most common errors."""
        # Add another network error to make it the most common
        self.monitor.record_error(
            NetworkError("Connection refused", url="https://api.example.com"),
            "api_request"
        )
        
        # Get most common errors
        common_errors = self.monitor.get_most_common_errors(limit=2)
        
        self.assertEqual(len(common_errors), 2)
        self.assertEqual(common_errors[0]['category'], "network")
        self.assertEqual(common_errors[0]['count'], 2)
    
    def test_get_error_trends(self):
        """Test getting error trends."""
        # Get error trends for the last day
        trends = self.monitor.get_error_trends(timedelta(days=1), 60)
        
        self.assertIn('intervals', trends)
        self.assertIn('counts', trends)
        self.assertIn('categories', trends)
    
    def test_generate_error_report(self):
        """Test generating error report."""
        # Generate report without charts for faster testing
        report_file = self.monitor.generate_error_report(include_charts=False)
        
        # Check that report file was created
        self.assertTrue(os.path.exists(report_file))
        
        # Check report content
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        self.assertEqual(report_data['total_errors'], 3)
        self.assertIn('error_rate', report_data)
        self.assertIn('most_common_errors', report_data)
        self.assertIn('error_trends', report_data)
        self.assertIn('recommendations', report_data)
    
    def test_max_errors_limit(self):
        """Test that max_errors limit is enforced."""
        # Create monitor with small limit
        small_monitor = ErrorMonitor(max_errors=2)
        
        # Add 3 errors
        small_monitor.record_error(Exception("Error 1"), "op1")
        small_monitor.record_error(Exception("Error 2"), "op2")
        small_monitor.record_error(Exception("Error 3"), "op3")
        
        # Check that only the latest 2 are kept
        self.assertEqual(len(small_monitor.errors), 2)
        self.assertEqual(small_monitor.errors[0]['exception_message'], "Error 2")
        self.assertEqual(small_monitor.errors[1]['exception_message'], "Error 3")


class TestErrorMonitoringFunctions(unittest.TestCase):
    """Test cases for error monitoring utility functions."""
    
    @patch('stock_analysis.utils.error_monitoring.get_error_monitor')
    def test_monitor_error(self, mock_get_monitor):
        """Test monitor_error function."""
        mock_monitor = MagicMock()
        mock_get_monitor.return_value = mock_monitor
        
        exception = ValueError("Test error")
        monitor_error(exception, "test_operation", {"param": "value"})
        
        mock_monitor.record_error.assert_called_once_with(
            exception, "test_operation", {"param": "value"}
        )
    
    @patch('stock_analysis.utils.error_monitoring.monitor_error')
    def test_error_monitoring_decorator(self, mock_monitor_error):
        """Test error_monitoring_decorator."""
        # Define a function that raises an exception
        @error_monitoring_decorator("test_operation")
        def failing_function(arg1, arg2=None):
            raise ValueError("Test error")
        
        # Call the function and expect it to raise
        with self.assertRaises(ValueError):
            failing_function("test", arg2="value")
        
        # Check that monitor_error was called
        mock_monitor_error.assert_called_once()
        args, kwargs = mock_monitor_error.call_args
        self.assertIsInstance(args[0], ValueError)
        self.assertEqual(args[1], "test_operation")
        self.assertIn("args", args[2])
        self.assertIn("kwargs", args[2])


if __name__ == '__main__':
    unittest.main()