"""Performance monitoring and analysis tool.

This module provides utilities for monitoring and analyzing performance
of the stock analysis system, identifying bottlenecks, and suggesting
optimizations.
"""

import time
import os
import json
import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tabulate import tabulate

from stock_analysis.utils.performance_metrics import get_metrics_collector
from stock_analysis.utils.cache_manager import get_cache_manager
from stock_analysis.utils.logging import get_logger
from stock_analysis.utils.config import config

logger = get_logger(__name__)


class PerformanceMonitoringTool:
    """Tool for monitoring and analyzing system performance."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize performance monitoring tool.
        
        Args:
            output_dir: Directory for output files (default: ./performance_reports)
        """
        self.metrics_collector = get_metrics_collector()
        self.cache_manager = get_cache_manager()
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'performance_reports')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_performance_report(self, include_charts: bool = True) -> str:
        """Generate a comprehensive performance report.
        
        Args:
            include_charts: Whether to include charts in the report
            
        Returns:
            Path to the generated report file
        """
        # Get performance data
        report_data = self.metrics_collector.generate_performance_report()
        cache_stats = self.cache_manager.get_stats()
        
        # Add timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create report file path
        report_file = os.path.join(self.output_dir, f'performance_report_{timestamp}.json')
        
        # Combine data
        full_report = {
            'timestamp': timestamp,
            'performance_metrics': report_data,
            'cache_stats': cache_stats
        }
        
        # Save report to file
        with open(report_file, 'w') as f:
            json.dump(full_report, f, indent=2, default=str)
        
        # Generate charts if requested
        if include_charts:
            charts_dir = os.path.join(self.output_dir, f'charts_{timestamp}')
            os.makedirs(charts_dir, exist_ok=True)
            self._generate_performance_charts(full_report, charts_dir)
        
        logger.info(f"Performance report generated: {report_file}")
        return report_file
    
    def _generate_performance_charts(self, report_data: Dict[str, Any], output_dir: str) -> None:
        """Generate performance charts from report data.
        
        Args:
            report_data: Performance report data
            output_dir: Directory for output charts
        """
        try:
            # Extract operation metrics
            operations = report_data['performance_metrics']['operations']
            
            if not operations:
                logger.warning("No operation metrics available for charts")
                return
            
            # Create dataframe for operations
            ops_data = []
            for op_name, metrics in operations.items():
                ops_data.append({
                    'operation': op_name,
                    'avg_duration': metrics.get('average_duration_seconds', 0),
                    'min_duration': metrics.get('min_duration_seconds', 0),
                    'max_duration': metrics.get('max_duration_seconds', 0),
                    'total_operations': metrics.get('total_operations', 0),
                    'error_rate': metrics.get('error_rate_percent', 0),
                    'memory_usage': metrics.get('average_memory_usage_mb', 0),
                    'cpu_usage': metrics.get('average_cpu_usage_percent', 0)
                })
            
            df = pd.DataFrame(ops_data)
            
            # 1. Operation Duration Chart
            plt.figure(figsize=(12, 6))
            plt.barh(df['operation'], df['avg_duration'], color='skyblue')
            plt.xlabel('Average Duration (seconds)')
            plt.ylabel('Operation')
            plt.title('Average Operation Duration')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'operation_duration.png'))
            plt.close()
            
            # 2. Operation Count Chart
            plt.figure(figsize=(12, 6))
            plt.barh(df['operation'], df['total_operations'], color='lightgreen')
            plt.xlabel('Total Operations')
            plt.ylabel('Operation')
            plt.title('Operation Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'operation_count.png'))
            plt.close()
            
            # 3. Error Rate Chart
            plt.figure(figsize=(12, 6))
            plt.barh(df['operation'], df['error_rate'], color='salmon')
            plt.xlabel('Error Rate (%)')
            plt.ylabel('Operation')
            plt.title('Operation Error Rate')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'error_rate.png'))
            plt.close()
            
            # 4. Resource Usage Chart
            if 'memory_usage' in df and df['memory_usage'].sum() > 0:
                plt.figure(figsize=(12, 6))
                plt.barh(df['operation'], df['memory_usage'], color='purple')
                plt.xlabel('Average Memory Usage (MB)')
                plt.ylabel('Operation')
                plt.title('Operation Memory Usage')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'memory_usage.png'))
                plt.close()
            
            # 5. Cache Statistics Chart
            cache_stats = report_data['cache_stats']
            if cache_stats:
                # Memory vs Disk Cache
                cache_sizes = [
                    cache_stats.get('memory_size_mb', 0),
                    cache_stats.get('disk_size_mb', 0),
                    cache_stats.get('hot_cache_size_mb', 0)
                ]
                cache_labels = ['Memory Cache', 'Disk Cache', 'Hot Cache']
                
                plt.figure(figsize=(10, 6))
                plt.bar(cache_labels, cache_sizes, color=['skyblue', 'lightgreen', 'salmon'])
                plt.ylabel('Size (MB)')
                plt.title('Cache Size Distribution')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'cache_size.png'))
                plt.close()
                
                # Cache Entries by Type
                if 'entries_by_type' in cache_stats:
                    entry_types = list(cache_stats['entries_by_type'].keys())
                    entry_counts = list(cache_stats['entries_by_type'].values())
                    
                    plt.figure(figsize=(10, 6))
                    plt.bar(entry_types, entry_counts, color='lightblue')
                    plt.ylabel('Number of Entries')
                    plt.title('Cache Entries by Type')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'cache_entries_by_type.png'))
                    plt.close()
            
            logger.info(f"Performance charts generated in {output_dir}")
            
        except Exception as e:
            logger.error(f"Error generating performance charts: {str(e)}")
    
    def analyze_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """Analyze performance data to identify bottlenecks.
        
        Returns:
            List of identified bottlenecks with recommendations
        """
        # Get performance data
        report_data = self.metrics_collector.generate_performance_report()
        operations = report_data.get('operations', {})
        
        bottlenecks = []
        
        # Analyze operation durations
        for op_name, metrics in operations.items():
            avg_duration = metrics.get('average_duration_seconds', 0)
            error_rate = metrics.get('error_rate_percent', 0)
            
            # Check for slow operations
            if avg_duration > 1.0:  # Operations taking more than 1 second
                bottleneck = {
                    'operation': op_name,
                    'issue': 'Slow operation',
                    'metric': f'{avg_duration:.2f} seconds average duration',
                    'severity': 'high' if avg_duration > 5.0 else 'medium',
                    'recommendations': []
                }
                
                # Add specific recommendations
                if 'data_retrieval' in op_name.lower() or 'api' in op_name.lower():
                    bottleneck['recommendations'].append('Consider implementing more aggressive caching')
                    bottleneck['recommendations'].append('Check for network latency issues')
                    bottleneck['recommendations'].append('Consider implementing parallel requests')
                
                if 'batch' in op_name.lower():
                    bottleneck['recommendations'].append('Increase parallelism for batch operations')
                    bottleneck['recommendations'].append('Consider reducing batch size')
                
                if 'calculation' in op_name.lower() or 'analysis' in op_name.lower():
                    bottleneck['recommendations'].append('Optimize calculation algorithms')
                    bottleneck['recommendations'].append('Consider using vectorized operations')
                    bottleneck['recommendations'].append('Implement memoization for repeated calculations')
                
                bottlenecks.append(bottleneck)
            
            # Check for high error rates
            if error_rate > 5.0:  # Error rate above 5%
                bottleneck = {
                    'operation': op_name,
                    'issue': 'High error rate',
                    'metric': f'{error_rate:.2f}% error rate',
                    'severity': 'high' if error_rate > 20.0 else 'medium',
                    'recommendations': [
                        'Implement more robust error handling',
                        'Add retry logic with exponential backoff',
                        'Check for API rate limiting issues',
                        'Verify data source availability'
                    ]
                }
                bottlenecks.append(bottleneck)
        
        # Analyze cache efficiency
        cache_stats = self.cache_manager.get_stats()
        memory_size_mb = cache_stats.get('memory_size_mb', 0)
        max_memory_size_mb = self.cache_manager._max_memory_size / (1024 * 1024)
        
        if memory_size_mb > 0.9 * max_memory_size_mb:
            bottleneck = {
                'operation': 'Cache Management',
                'issue': 'Memory cache near capacity',
                'metric': f'{memory_size_mb:.2f}MB / {max_memory_size_mb:.2f}MB ({memory_size_mb/max_memory_size_mb*100:.1f}%)',
                'severity': 'medium',
                'recommendations': [
                    'Increase memory cache size limit',
                    'Implement more aggressive cache eviction',
                    'Review cache TTL settings',
                    'Consider using more disk cache for less frequently accessed data'
                ]
            }
            bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def print_performance_summary(self) -> None:
        """Print a summary of performance metrics to the console."""
        # Get performance data
        report_data = self.metrics_collector.generate_performance_report()
        operations = report_data.get('operations', {})
        
        # Prepare data for tabulation
        table_data = []
        for op_name, metrics in operations.items():
            table_data.append([
                op_name,
                metrics.get('total_operations', 0),
                f"{metrics.get('average_duration_seconds', 0):.3f}s",
                f"{metrics.get('min_duration_seconds', 0):.3f}s",
                f"{metrics.get('max_duration_seconds', 0):.3f}s",
                f"{metrics.get('error_rate_percent', 0):.1f}%"
            ])
        
        # Sort by average duration (descending)
        table_data.sort(key=lambda x: float(x[2][:-1]), reverse=True)
        
        # Print table
        headers = ['Operation', 'Count', 'Avg Duration', 'Min Duration', 'Max Duration', 'Error Rate']
        print("\n=== Performance Metrics Summary ===")
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # Print cache stats
        cache_stats = self.cache_manager.get_stats()
        print("\n=== Cache Statistics ===")
        print(f"Memory Cache: {cache_stats.get('memory_entries', 0)} entries, "
              f"{cache_stats.get('memory_size_mb', 0):.2f}MB")
        print(f"Hot Cache: {cache_stats.get('hot_cache_entries', 0)} entries, "
              f"{cache_stats.get('hot_cache_size_mb', 0):.2f}MB")
        print(f"Disk Cache: {cache_stats.get('disk_entries', 0)} entries, "
              f"{cache_stats.get('disk_size_mb', 0):.2f}MB")
        
        # Print bottlenecks
        bottlenecks = self.analyze_performance_bottlenecks()
        if bottlenecks:
            print("\n=== Performance Bottlenecks ===")
            for i, bottleneck in enumerate(bottlenecks, 1):
                print(f"{i}. {bottleneck['operation']} - {bottleneck['issue']} ({bottleneck['metric']})")
                print(f"   Severity: {bottleneck['severity']}")
                print(f"   Recommendations:")
                for rec in bottleneck['recommendations']:
                    print(f"   - {rec}")
                print()
    
    def monitor_operation(self, operation_name: str, duration_seconds: int = 60) -> Dict[str, Any]:
        """Monitor a specific operation for a period of time.
        
        Args:
            operation_name: Name of the operation to monitor
            duration_seconds: Duration of monitoring in seconds
            
        Returns:
            Dictionary with monitoring results
        """
        logger.info(f"Starting performance monitoring for {operation_name} "
                   f"for {duration_seconds} seconds")
        
        # Get initial metrics
        initial_metrics = self.metrics_collector.get_aggregated_metrics(operation_name)
        
        # Wait for specified duration
        time.sleep(duration_seconds)
        
        # Get updated metrics
        final_metrics = self.metrics_collector.get_aggregated_metrics(operation_name)
        
        # Calculate differences
        if initial_metrics and final_metrics:
            operations_delta = final_metrics.total_operations - initial_metrics.total_operations
            success_delta = final_metrics.successful_operations - initial_metrics.successful_operations
            failure_delta = final_metrics.failed_operations - initial_metrics.failed_operations
            
            # Calculate rates
            ops_per_second = operations_delta / duration_seconds if duration_seconds > 0 else 0
            success_rate = success_delta / operations_delta * 100 if operations_delta > 0 else 0
            failure_rate = failure_delta / operations_delta * 100 if operations_delta > 0 else 0
            
            results = {
                'operation_name': operation_name,
                'monitoring_duration_seconds': duration_seconds,
                'operations_total': operations_delta,
                'operations_per_second': ops_per_second,
                'success_rate_percent': success_rate,
                'failure_rate_percent': failure_rate,
                'average_duration_seconds': final_metrics.average_duration_seconds,
                'min_duration_seconds': final_metrics.min_duration_seconds,
                'max_duration_seconds': final_metrics.max_duration_seconds
            }
        else:
            results = {
                'operation_name': operation_name,
                'monitoring_duration_seconds': duration_seconds,
                'operations_total': 0,
                'message': 'No operations recorded during monitoring period'
            }
        
        logger.info(f"Completed performance monitoring for {operation_name}")
        return results


def get_performance_monitoring_tool() -> PerformanceMonitoringTool:
    """Get a performance monitoring tool instance.
    
    Returns:
        PerformanceMonitoringTool instance
    """
    output_dir = config.get('stock_analysis.performance.output_dir', None)
    return PerformanceMonitoringTool(output_dir)


if __name__ == '__main__':
    # When run as a script, generate a performance report
    tool = get_performance_monitoring_tool()
    report_file = tool.generate_performance_report()
    tool.print_performance_summary()
    print(f"\nPerformance report saved to: {report_file}")