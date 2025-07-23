"""Error monitoring and reporting module.

This module provides utilities for monitoring errors, generating error reports,
and analyzing error patterns.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt

from stock_analysis.utils.exceptions import StockAnalysisError
from stock_analysis.utils.logging import get_logger
from stock_analysis.utils.config import config
from stock_analysis.utils.error_recovery import ErrorDiagnostics

logger = get_logger(__name__)


class ErrorMonitor:
    """Monitor and analyze errors in the system."""
    
    def __init__(self, max_errors: int = 1000, output_dir: Optional[str] = None):
        """Initialize error monitor.
        
        Args:
            max_errors: Maximum number of errors to store in memory
            output_dir: Directory for error reports (default: ./error_reports)
        """
        self.max_errors = max_errors
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'error_reports')
        self.errors: List[Dict[str, Any]] = []
        self.error_diagnostics = ErrorDiagnostics()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def record_error(self, exception: Exception, operation_name: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Record an error for monitoring.
        
        Args:
            exception: Exception that occurred
            operation_name: Name of the operation that failed
            context: Additional context
        """
        # Diagnose the error
        diagnostics = self.error_diagnostics.diagnose_error(exception, operation_name)
        
        # Add context if provided
        if context:
            diagnostics['context'] = context
        
        # Add to error list
        self.errors.append(diagnostics)
        
        # Trim list if it exceeds max size
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors:]
    
    def get_error_count(self, time_window: Optional[timedelta] = None) -> int:
        """Get count of errors within a time window.
        
        Args:
            time_window: Time window to count errors (None for all errors)
            
        Returns:
            Number of errors
        """
        if time_window is None:
            return len(self.errors)
        
        cutoff_time = datetime.now() - time_window
        return sum(1 for error in self.errors 
                  if datetime.fromisoformat(error['timestamp']) >= cutoff_time)
    
    def get_error_rate(self, time_window: Optional[timedelta] = None) -> Dict[str, float]:
        """Get error rate statistics.
        
        Args:
            time_window: Time window to calculate rate (None for all time)
            
        Returns:
            Dictionary with error rate statistics
        """
        if not self.errors:
            return {'error_count': 0, 'error_rate_per_minute': 0.0, 'error_rate_per_hour': 0.0}
        
        if time_window is None:
            # Use the entire time range of recorded errors
            start_time = datetime.fromisoformat(self.errors[0]['timestamp'])
            end_time = datetime.fromisoformat(self.errors[-1]['timestamp'])
            error_count = len(self.errors)
        else:
            # Use specified time window
            cutoff_time = datetime.now() - time_window
            filtered_errors = [error for error in self.errors 
                              if datetime.fromisoformat(error['timestamp']) >= cutoff_time]
            
            if not filtered_errors:
                return {'error_count': 0, 'error_rate_per_minute': 0.0, 'error_rate_per_hour': 0.0}
            
            start_time = cutoff_time
            end_time = datetime.now()
            error_count = len(filtered_errors)
        
        # Calculate time difference in minutes and hours
        time_diff_minutes = (end_time - start_time).total_seconds() / 60
        time_diff_hours = time_diff_minutes / 60
        
        # Calculate rates
        rate_per_minute = error_count / time_diff_minutes if time_diff_minutes > 0 else 0
        rate_per_hour = error_count / time_diff_hours if time_diff_hours > 0 else 0
        
        return {
            'error_count': error_count,
            'time_period_minutes': time_diff_minutes,
            'error_rate_per_minute': rate_per_minute,
            'error_rate_per_hour': rate_per_hour
        }
    
    def get_most_common_errors(self, limit: int = 10, time_window: Optional[timedelta] = None) -> List[Dict[str, Any]]:
        """Get most common errors.
        
        Args:
            limit: Maximum number of errors to return
            time_window: Time window to consider (None for all time)
            
        Returns:
            List of error categories with counts
        """
        if not self.errors:
            return []
        
        # Filter by time window if specified
        if time_window is not None:
            cutoff_time = datetime.now() - time_window
            filtered_errors = [error for error in self.errors 
                              if datetime.fromisoformat(error['timestamp']) >= cutoff_time]
        else:
            filtered_errors = self.errors
        
        if not filtered_errors:
            return []
        
        # Count errors by category and exception type
        error_counts = defaultdict(int)
        for error in filtered_errors:
            key = f"{error['category']}:{error['exception_type']}"
            error_counts[key] += 1
        
        # Sort by count (descending) and get top N
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Format results
        result = []
        for key, count in sorted_errors:
            category, exception_type = key.split(':', 1)
            result.append({
                'category': category,
                'exception_type': exception_type,
                'count': count,
                'percentage': (count / len(filtered_errors)) * 100
            })
        
        return result
    
    def get_error_trends(self, time_window: timedelta = timedelta(days=1), interval_minutes: int = 60) -> Dict[str, Any]:
        """Get error trends over time.
        
        Args:
            time_window: Time window to analyze
            interval_minutes: Interval in minutes for trend data points
            
        Returns:
            Dictionary with error trend data
        """
        if not self.errors:
            return {'intervals': [], 'counts': [], 'categories': {}}
        
        # Calculate time intervals
        end_time = datetime.now()
        start_time = end_time - time_window
        
        # Create time intervals
        intervals = []
        current_time = start_time
        while current_time <= end_time:
            intervals.append(current_time)
            current_time += timedelta(minutes=interval_minutes)
        
        if not intervals:
            return {'intervals': [], 'counts': [], 'categories': {}}
        
        # Count errors in each interval
        interval_counts = [0] * len(intervals)
        category_counts = defaultdict(lambda: [0] * len(intervals))
        
        for error in self.errors:
            error_time = datetime.fromisoformat(error['timestamp'])
            if error_time < start_time:
                continue
            
            # Find the interval this error belongs to
            for i, interval_start in enumerate(intervals[:-1]):
                interval_end = intervals[i + 1]
                if interval_start <= error_time < interval_end:
                    interval_counts[i] += 1
                    category_counts[error['category']][i] += 1
                    break
            else:
                # Check if it belongs to the last interval
                if intervals[-1] <= error_time <= end_time:
                    interval_counts[-1] += 1
                    category_counts[error['category']][-1] += 1
        
        # Format intervals as strings
        interval_labels = [interval.strftime('%Y-%m-%d %H:%M') for interval in intervals]
        
        # Format category data
        categories = {}
        for category, counts in category_counts.items():
            categories[category] = counts
        
        return {
            'intervals': interval_labels,
            'counts': interval_counts,
            'categories': categories
        }
    
    def generate_error_report(self, include_charts: bool = True) -> str:
        """Generate a comprehensive error report.
        
        Args:
            include_charts: Whether to include charts in the report
            
        Returns:
            Path to the generated report file
        """
        # Add timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create report file path
        report_file = os.path.join(self.output_dir, f'error_report_{timestamp}.json')
        
        # Generate report data
        report_data = {
            'timestamp': timestamp,
            'total_errors': len(self.errors),
            'error_rate': self.get_error_rate(timedelta(hours=24)),
            'most_common_errors': self.get_most_common_errors(limit=10),
            'error_trends': self.get_error_trends(timedelta(days=1), 60),
            'error_categories': self._get_error_categories(),
            'transient_vs_persistent': self._get_transient_persistent_ratio(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report to file
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate charts if requested
        if include_charts and self.errors:
            charts_dir = os.path.join(self.output_dir, f'charts_{timestamp}')
            os.makedirs(charts_dir, exist_ok=True)
            self._generate_error_charts(report_data, charts_dir)
        
        logger.info(f"Error report generated: {report_file}")
        return report_file
    
    def _get_error_categories(self) -> Dict[str, int]:
        """Get counts of errors by category.
        
        Returns:
            Dictionary mapping categories to counts
        """
        categories = Counter()
        for error in self.errors:
            categories[error['category']] += 1
        return dict(categories)
    
    def _get_transient_persistent_ratio(self) -> Dict[str, Any]:
        """Get ratio of transient to persistent errors.
        
        Returns:
            Dictionary with transient/persistent error statistics
        """
        if not self.errors:
            return {'transient': 0, 'persistent': 0, 'transient_percentage': 0}
        
        transient_count = sum(1 for error in self.errors if error['is_transient'])
        persistent_count = len(self.errors) - transient_count
        
        return {
            'transient': transient_count,
            'persistent': persistent_count,
            'transient_percentage': (transient_count / len(self.errors)) * 100 if self.errors else 0
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on error patterns.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if not self.errors:
            return ["No errors recorded. System appears to be functioning normally."]
        
        # Check for high error rate
        error_rate = self.get_error_rate(timedelta(hours=1))
        if error_rate['error_rate_per_minute'] > 1:
            recommendations.append(
                f"High error rate detected: {error_rate['error_rate_per_minute']:.2f} errors/minute. "
                "Consider investigating system stability issues."
            )
        
        # Check for common error categories
        categories = self._get_error_categories()
        for category, count in categories.items():
            percentage = (count / len(self.errors)) * 100
            
            if category == 'network' and percentage > 30:
                recommendations.append(
                    f"High rate of network errors ({percentage:.1f}%). "
                    "Check network connectivity and API endpoint availability."
                )
            
            elif category == 'data_retrieval' and percentage > 30:
                recommendations.append(
                    f"High rate of data retrieval errors ({percentage:.1f}%). "
                    "Check data source availability and credentials."
                )
            
            elif category == 'validation' and percentage > 20:
                recommendations.append(
                    f"High rate of data validation errors ({percentage:.1f}%). "
                    "Review data quality and validation rules."
                )
        
        # Check for transient vs persistent errors
        transient_stats = self._get_transient_persistent_ratio()
        if transient_stats['persistent'] > 0 and transient_stats['transient_percentage'] < 30:
            recommendations.append(
                f"High rate of persistent errors ({100 - transient_stats['transient_percentage']:.1f}%). "
                "These may require code or configuration changes to resolve."
            )
        
        # Add general recommendations if none specific
        if not recommendations:
            recommendations.append(
                "No critical error patterns detected. Continue monitoring for changes."
            )
        
        return recommendations
    
    def _generate_error_charts(self, report_data: Dict[str, Any], output_dir: str) -> None:
        """Generate error charts from report data.
        
        Args:
            report_data: Error report data
            output_dir: Directory for output charts
        """
        try:
            # 1. Error Trend Chart
            trends = report_data['error_trends']
            if trends['intervals'] and trends['counts']:
                plt.figure(figsize=(12, 6))
                plt.plot(trends['intervals'], trends['counts'], marker='o')
                plt.xlabel('Time')
                plt.ylabel('Error Count')
                plt.title('Error Trend Over Time')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'error_trend.png'))
                plt.close()
            
            # 2. Error Categories Chart
            categories = report_data['error_categories']
            if categories:
                plt.figure(figsize=(10, 6))
                plt.bar(categories.keys(), categories.values(), color='salmon')
                plt.xlabel('Error Category')
                plt.ylabel('Count')
                plt.title('Errors by Category')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'error_categories.png'))
                plt.close()
            
            # 3. Transient vs Persistent Chart
            transient_data = report_data['transient_vs_persistent']
            if transient_data['transient'] > 0 or transient_data['persistent'] > 0:
                plt.figure(figsize=(8, 8))
                plt.pie(
                    [transient_data['transient'], transient_data['persistent']],
                    labels=['Transient', 'Persistent'],
                    autopct='%1.1f%%',
                    colors=['lightgreen', 'salmon']
                )
                plt.title('Transient vs Persistent Errors')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'transient_persistent.png'))
                plt.close()
            
            # 4. Most Common Errors Chart
            common_errors = report_data['most_common_errors']
            if common_errors:
                labels = [f"{error['exception_type']} ({error['category']})" for error in common_errors]
                counts = [error['count'] for error in common_errors]
                
                plt.figure(figsize=(12, 6))
                plt.barh(labels, counts, color='lightblue')
                plt.xlabel('Count')
                plt.ylabel('Error Type')
                plt.title('Most Common Errors')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'common_errors.png'))
                plt.close()
            
            logger.info(f"Error charts generated in {output_dir}")
            
        except Exception as e:
            logger.error(f"Error generating error charts: {str(e)}")


# Global error monitor instance
_global_error_monitor: Optional[ErrorMonitor] = None


def get_error_monitor() -> ErrorMonitor:
    """Get the global error monitor instance.
    
    Returns:
        ErrorMonitor instance
    """
    global _global_error_monitor
    if _global_error_monitor is None:
        output_dir = config.get('stock_analysis.error_monitoring.output_dir', None)
        max_errors = config.get('stock_analysis.error_monitoring.max_errors', 1000)
        _global_error_monitor = ErrorMonitor(max_errors=max_errors, output_dir=output_dir)
    return _global_error_monitor


def monitor_error(exception: Exception, operation_name: str, context: Optional[Dict[str, Any]] = None) -> None:
    """Record an error in the global error monitor.
    
    Args:
        exception: Exception that occurred
        operation_name: Name of the operation that failed
        context: Additional context
    """
    monitor = get_error_monitor()
    monitor.record_error(exception, operation_name, context)


def error_monitoring_decorator(operation_name: Optional[str] = None):
    """Decorator for monitoring errors in functions.
    
    Args:
        operation_name: Name of the operation (defaults to function name)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Record the error
                context = {'args': str(args), 'kwargs': str(kwargs)}
                monitor_error(e, op_name, context)
                # Re-raise the exception
                raise
        return wrapper
    return decorator