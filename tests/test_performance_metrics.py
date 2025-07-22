"""Unit tests for performance metrics collection."""

import pytest
import time
import json
import os
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from stock_analysis.utils.performance_metrics import (
    OperationMetrics, AggregatedMetrics, PerformanceMetricsCollector,
    PerformanceMonitor, monitor_performance
)


@pytest.fixture
def metrics_collector():
    """Create a metrics collector for testing."""
    return PerformanceMetricsCollector(max_metrics_per_operation=10)


class TestOperationMetrics:
    """Test cases for OperationMetrics."""
    
    def test_operation_metrics_creation(self):
        """Test creating operation metrics."""
        metrics = OperationMetrics(
            operation_name="test_operation",
            start_time=datetime.utcnow()
        )
        
        assert metrics.operation_name == "test_operation"
        assert metrics.end_time is None
        assert metrics.duration_seconds is None
        assert metrics.success is True
        assert metrics.error_message is None
    
    def test_operation_metrics_finish(self):
        """Test finishing operation metrics."""
        metrics = OperationMetrics(
            operation_name="test_operation",
            start_time=datetime.utcnow()
        )
        
        time.sleep(0.1)  # Small delay to ensure measurable duration
        metrics.finish(success=True)
        
        assert metrics.end_time is not None
        assert metrics.duration_seconds is not None
        assert metrics.duration_seconds > 0
        assert metrics.success is True
    
    def test_operation_metrics_finish_with_error(self):
        """Test finishing operation metrics with error."""
        metrics = OperationMetrics(
            operation_name="test_operation",
            start_time=datetime.utcnow()
        )
        
        metrics.finish(success=False, error_message="Test error")
        
        assert metrics.success is False
        assert metrics.error_message == "Test error"
    
    def test_operation_metrics_to_dict(self):
        """Test converting operation metrics to dictionary."""
        start_time = datetime.utcnow()
        metrics = OperationMetrics(
            operation_name="test_operation",
            start_time=start_time,
            context={'test_key': 'test_value'}
        )
        
        metrics.finish(success=True)
        data = metrics.to_dict()
        
        assert data['operation_name'] == "test_operation"
        assert data['start_time'] == start_time.isoformat()
        assert data['success'] is True
        assert data['context'] == {'test_key': 'test_value'}


class TestAggregatedMetrics:
    """Test cases for AggregatedMetrics."""
    
    def test_aggregated_metrics_creation(self):
        """Test creating aggregated metrics."""
        metrics = AggregatedMetrics(operation_name="test_operation")
        
        assert metrics.operation_name == "test_operation"
        assert metrics.total_operations == 0
        assert metrics.successful_operations == 0
        assert metrics.failed_operations == 0
        assert metrics.error_rate_percent == 0.0
    
    def test_aggregated_metrics_to_dict(self):
        """Test converting aggregated metrics to dictionary."""
        metrics = AggregatedMetrics(
            operation_name="test_operation",
            total_operations=10,
            successful_operations=8,
            failed_operations=2,
            total_duration_seconds=5.0,
            average_duration_seconds=0.5,
            min_duration_seconds=0.1,
            max_duration_seconds=1.0,
            average_memory_usage_mb=100.0,
            average_cpu_usage_percent=50.0,
            error_rate_percent=20.0
        )
        
        data = metrics.to_dict()
        
        assert data['operation_name'] == "test_operation"
        assert data['total_operations'] == 10
        assert data['successful_operations'] == 8
        assert data['failed_operations'] == 2
        assert data['error_rate_percent'] == 20.0


class TestPerformanceMetricsCollector:
    """Test cases for PerformanceMetricsCollector."""
    
    def test_start_operation(self, metrics_collector):
        """Test starting operation tracking."""
        operation_id = metrics_collector.start_operation(
            "test_operation",
            context={'test_key': 'test_value'}
        )
        
        assert operation_id in metrics_collector.active_operations
        metrics = metrics_collector.active_operations[operation_id]
        assert metrics.operation_name == "test_operation"
        assert metrics.context == {'test_key': 'test_value'}
    
    def test_finish_operation(self, metrics_collector):
        """Test finishing operation tracking."""
        operation_id = metrics_collector.start_operation("test_operation")
        metrics_collector.finish_operation(operation_id, success=True)
        
        assert operation_id not in metrics_collector.active_operations
        assert len(metrics_collector.metrics["test_operation"]) == 1
        metrics = metrics_collector.metrics["test_operation"][0]
        assert metrics.success is True
    
    def test_finish_operation_with_error(self, metrics_collector):
        """Test finishing operation with error."""
        operation_id = metrics_collector.start_operation("test_operation")
        metrics_collector.finish_operation(
            operation_id,
            success=False,
            error_message="Test error"
        )
        
        metrics = metrics_collector.metrics["test_operation"][0]
        assert metrics.success is False
        assert metrics.error_message == "Test error"
    
    def test_get_operation_metrics(self, metrics_collector):
        """Test getting operation metrics."""
        # Add some test metrics
        operation_id = metrics_collector.start_operation("test_operation")
        metrics_collector.finish_operation(operation_id, success=True)
        
        metrics_list = metrics_collector.get_operation_metrics("test_operation")
        assert len(metrics_list) == 1
        assert metrics_list[0].operation_name == "test_operation"
        assert metrics_list[0].success is True
    
    def test_get_aggregated_metrics(self, metrics_collector):
        """Test getting aggregated metrics."""
        # Add some test metrics
        for i in range(3):
            operation_id = metrics_collector.start_operation("test_operation")
            metrics_collector.finish_operation(
                operation_id,
                success=(i < 2)  # Make 2 successful, 1 failed
            )
        
        aggregated = metrics_collector.get_aggregated_metrics("test_operation")
        assert aggregated is not None
        assert aggregated.total_operations == 3
        assert aggregated.successful_operations == 2
        assert aggregated.failed_operations == 1
        assert aggregated.error_rate_percent == pytest.approx(33.33, rel=0.01)
    
    def test_max_metrics_limit(self, metrics_collector):
        """Test maximum metrics limit per operation."""
        # Add more metrics than the limit
        for i in range(15):  # Limit is 10
            operation_id = metrics_collector.start_operation("test_operation")
            metrics_collector.finish_operation(operation_id)
        
        metrics_list = metrics_collector.get_operation_metrics("test_operation")
        assert len(metrics_list) == 10  # Should be limited to 10
    
    def test_generate_performance_report(self, metrics_collector):
        """Test generating performance report."""
        # Add some test metrics
        operation_id = metrics_collector.start_operation("test_operation")
        metrics_collector.finish_operation(operation_id)
        
        report = metrics_collector.generate_performance_report()
        assert 'timestamp' in report
        assert 'system_info' in report
        assert 'operations' in report
        assert 'test_operation' in report['operations']
    
    def test_export_metrics(self, metrics_collector, tmp_path):
        """Test exporting metrics to file."""
        # Add some test metrics
        operation_id = metrics_collector.start_operation("test_operation")
        metrics_collector.finish_operation(operation_id)
        
        # Export metrics
        export_file = tmp_path / "metrics.json"
        metrics_collector.export_metrics(str(export_file))
        
        # Verify exported file
        assert export_file.exists()
        with open(export_file) as f:
            data = json.load(f)
            assert 'operations' in data
            assert 'test_operation' in data['operations']
    
    def test_clear_metrics(self, metrics_collector):
        """Test clearing metrics."""
        # Add some test metrics
        operation_id = metrics_collector.start_operation("test_operation")
        metrics_collector.finish_operation(operation_id)
        
        # Clear metrics for specific operation
        metrics_collector.clear_metrics("test_operation")
        assert len(metrics_collector.get_operation_metrics("test_operation")) == 0
        
        # Add more metrics and clear all
        operation_id = metrics_collector.start_operation("test_operation")
        metrics_collector.finish_operation(operation_id)
        metrics_collector.clear_metrics()
        assert len(metrics_collector.metrics) == 0


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""
    
    def test_context_manager(self, metrics_collector):
        """Test using performance monitor as context manager."""
        with PerformanceMonitor(metrics_collector, "test_operation"):
            time.sleep(0.1)  # Simulate some work
        
        metrics_list = metrics_collector.get_operation_metrics("test_operation")
        assert len(metrics_list) == 1
        assert metrics_list[0].success is True
        assert metrics_list[0].duration_seconds > 0
    
    def test_context_manager_with_error(self, metrics_collector):
        """Test context manager with error."""
        with pytest.raises(ValueError):
            with PerformanceMonitor(metrics_collector, "test_operation"):
                raise ValueError("Test error")
        
        metrics_list = metrics_collector.get_operation_metrics("test_operation")
        assert len(metrics_list) == 1
        assert metrics_list[0].success is False
        assert "Test error" in metrics_list[0].error_message
    
    def test_add_context(self, metrics_collector):
        """Test adding context during operation."""
        with PerformanceMonitor(metrics_collector, "test_operation") as monitor:
            monitor.add_context("test_key", "test_value")
        
        metrics_list = metrics_collector.get_operation_metrics("test_operation")
        assert metrics_list[0].context["test_key"] == "test_value"


def test_performance_decorator(metrics_collector):
    """Test performance monitoring decorator."""
    # Create a test function with the decorator
    with patch('stock_analysis.utils.performance_metrics.get_metrics_collector', return_value=metrics_collector):
        @monitor_performance("test_operation")
        def test_function():
            time.sleep(0.1)
            return "success"
        
        # Call the function
        result = test_function()
        
        assert result == "success"
        metrics_list = metrics_collector.get_operation_metrics("test_operation")
        assert len(metrics_list) == 1
        assert metrics_list[0].success is True
        assert metrics_list[0].duration_seconds > 0