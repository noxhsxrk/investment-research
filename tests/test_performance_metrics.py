"""Tests for performance metrics collection system."""

import unittest
from unittest.mock import patch, Mock
import time
import tempfile
import os
import json

from stock_analysis.utils.performance_metrics import (
    PerformanceMetricsCollector, OperationMetrics, 
    PerformanceMonitor, get_metrics_collector, monitor_performance
)


class TestPerformanceMetrics(unittest.TestCase):
    """Test cases for performance metrics collection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = PerformanceMetricsCollector(max_metrics_per_operation=10)
    
    def test_operation_tracking(self):
        """Test basic operation tracking."""
        operation_id = self.collector.start_operation("test_op")
        
        # Should be in active operations
        self.assertIn(operation_id, self.collector.active_operations)
        
        # Finish the operation
        self.collector.finish_operation(operation_id, success=True)
        
        # Should be moved to metrics
        self.assertNotIn(operation_id, self.collector.active_operations)
        metrics = self.collector.get_operation_metrics("test_op")
        self.assertEqual(len(metrics), 1)
        self.assertTrue(metrics[0].success)
    
    def test_operation_context(self):
        """Test operation context tracking."""
        context = {"symbol": "AAPL", "data_type": "stock_info"}
        operation_id = self.collector.start_operation("test_op", context=context)
        
        self.collector.finish_operation(
            operation_id, 
            success=False, 
            error_message="Test error",
            additional_context={"retry_count": 3}
        )
        
        metrics = self.collector.get_operation_metrics("test_op")
        self.assertEqual(len(metrics), 1)
        
        metric = metrics[0]
        self.assertFalse(metric.success)
        self.assertEqual(metric.error_message, "Test error")
        self.assertEqual(metric.context["symbol"], "AAPL")
        self.assertEqual(metric.context["retry_count"], 3)
    
    def test_aggregated_metrics(self):
        """Test aggregated metrics calculation."""
        # Add multiple operations
        for i in range(5):
            operation_id = self.collector.start_operation("test_op")
            success = i < 3  # First 3 succeed, last 2 fail
            self.collector.finish_operation(operation_id, success=success)
        
        aggregated = self.collector.get_aggregated_metrics("test_op")
        
        self.assertIsNotNone(aggregated)
        self.assertEqual(aggregated.total_operations, 5)
        self.assertEqual(aggregated.successful_operations, 3)
        self.assertEqual(aggregated.failed_operations, 2)
        self.assertEqual(aggregated.error_rate_percent, 40.0)
    
    def test_performance_monitor_context_manager(self):
        """Test PerformanceMonitor context manager."""
        with PerformanceMonitor(self.collector, "test_op") as monitor:
            monitor.add_context("test_key", "test_value")
            time.sleep(0.01)  # Small delay to measure
        
        metrics = self.collector.get_operation_metrics("test_op")
        self.assertEqual(len(metrics), 1)
        
        metric = metrics[0]
        self.assertTrue(metric.success)
        self.assertGreater(metric.duration_seconds, 0)
        self.assertEqual(metric.context["test_key"], "test_value")
    
    def test_performance_monitor_with_exception(self):
        """Test PerformanceMonitor with exception handling."""
        with self.assertRaises(ValueError):
            with PerformanceMonitor(self.collector, "test_op"):
                raise ValueError("Test error")
        
        metrics = self.collector.get_operation_metrics("test_op")
        self.assertEqual(len(metrics), 1)
        
        metric = metrics[0]
        self.assertFalse(metric.success)
        self.assertEqual(metric.error_message, "Test error")
    
    def test_performance_decorator(self):
        """Test performance monitoring decorator."""
        @monitor_performance("decorated_op")
        def test_function(x, y):
            return x + y
        
        result = test_function(2, 3)
        self.assertEqual(result, 5)
        
        # Check metrics were collected
        collector = get_metrics_collector()
        metrics = collector.get_operation_metrics("decorated_op")
        self.assertGreater(len(metrics), 0)
    
    def test_metrics_export(self):
        """Test metrics export functionality."""
        # Add some test data
        operation_id = self.collector.start_operation("export_test")
        self.collector.finish_operation(operation_id, success=True)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.collector.export_metrics(temp_path, "export_test")
            
            # Verify file was created and contains data
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            self.assertEqual(data["operation_name"], "export_test")
            self.assertIn("metrics", data)
            self.assertIn("aggregated", data)
            
        finally:
            os.unlink(temp_path)
    
    def test_performance_report_generation(self):
        """Test comprehensive performance report generation."""
        # Add test operations
        for op_name in ["op1", "op2"]:
            operation_id = self.collector.start_operation(op_name)
            self.collector.finish_operation(operation_id, success=True)
        
        report = self.collector.generate_performance_report()
        
        self.assertIn("timestamp", report)
        self.assertIn("system_info", report)
        self.assertIn("operations", report)
        self.assertIn("active_operations", report)
        
        # Should have data for both operations
        self.assertIn("op1", report["operations"])
        self.assertIn("op2", report["operations"])


if __name__ == '__main__':
    unittest.main()