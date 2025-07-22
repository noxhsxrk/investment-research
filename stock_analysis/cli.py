"""Command-line interface for the Stock Analysis Dashboard.

This module provides a comprehensive CLI for stock analysis operations including
single stock analysis, batch processing, scheduling management, and configuration.
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from stock_analysis.orchestrator import StockAnalysisOrchestrator, AnalysisProgress
from stock_analysis.services.scheduler_service import SchedulerService
from stock_analysis.services.scheduler_daemon import SchedulerDaemon
from stock_analysis.utils.config import config
from stock_analysis.utils.logging import get_logger
from stock_analysis.utils.exceptions import StockAnalysisError, DataRetrievalError, CalculationError, ExportError
from stock_analysis.models.data_models import (
    AnalysisResult, StockInfo, FinancialRatios, LiquidityRatios, ProfitabilityRatios,
    LeverageRatios, EfficiencyRatios, HealthScore, FairValueResult, SentimentResult
)

logger = get_logger(__name__)


class ProgressIndicator:
    """Simple progress indicator for CLI operations."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.last_progress = 0
    
    def update_progress(self, progress: AnalysisProgress) -> None:
        """Update progress display.
        
        Args:
            progress: Current analysis progress
        """
        if not self.verbose:
            return
        
        percentage = progress.completion_percentage
        
        # Only update if progress changed significantly
        if abs(percentage - self.last_progress) >= 5 or percentage == 100:
            print(f"\rProgress: {percentage:.1f}% ({progress.completed_stocks}/{progress.total_stocks} completed, "
                  f"{progress.failed_stocks} failed)", end="", flush=True)
            self.last_progress = percentage
            
            if progress.current_stock:
                print(f" - Currently analyzing: {progress.current_stock}", end="", flush=True)
        
        if percentage == 100:
            print()  # New line when complete


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog='stock-analysis',
        description='Stock Analysis Dashboard - Comprehensive stock analysis with Power BI integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single stock
  stock-analysis analyze AAPL
  
  # Analyze multiple stocks with verbose output
  stock-analysis analyze AAPL MSFT GOOGL --verbose --export-format excel
  
  # Analyze stocks from a file
  stock-analysis batch stocks.txt --export-format csv
  
  # Schedule daily analysis
  stock-analysis schedule add daily-tech "AAPL,MSFT,GOOGL" --interval daily
  
  # Start the scheduler
  stock-analysis schedule start
  
  # Export existing analysis results
  stock-analysis export results.json --format powerbi --output dashboard_data
  
  # Show configuration
  stock-analysis config show
        """
    )
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output with progress indicators')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze one or more stocks')
    analyze_parser.add_argument('symbols', nargs='+', help='Stock symbols to analyze')
    analyze_parser.add_argument('--export-format', '-f', choices=['csv', 'excel', 'json'],
                               default='excel', help='Export format (default: excel)')
    analyze_parser.add_argument('--output', '-o', type=str,
                               help='Output filename (without extension)')
    analyze_parser.add_argument('--no-export', action='store_true',
                               help='Skip exporting results')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch analyze stocks from file')
    batch_parser.add_argument('file', help='File containing stock symbols (one per line or CSV)')
    batch_parser.add_argument('--export-format', '-f', choices=['csv', 'excel', 'json'],
                             default='excel', help='Export format (default: excel)')
    batch_parser.add_argument('--output', '-o', type=str,
                             help='Output filename (without extension)')
    batch_parser.add_argument('--parallel', action='store_true', default=True,
                             help='Enable parallel processing (default: enabled)')
    batch_parser.add_argument('--max-workers', type=int, default=4,
                             help='Maximum number of worker threads (default: 4)')
    batch_parser.add_argument('--continue-on-error', action='store_true', default=True,
                             help='Continue processing if individual stocks fail (default: enabled)')
    
    # Schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Manage scheduled analysis jobs')
    schedule_subparsers = schedule_parser.add_subparsers(dest='schedule_action', help='Schedule actions')
    
    # Schedule add
    schedule_add_parser = schedule_subparsers.add_parser('add', help='Add a new scheduled job')
    schedule_add_parser.add_argument('job_id', help='Unique job identifier')
    schedule_add_parser.add_argument('symbols', help='Comma-separated list of stock symbols')
    schedule_add_parser.add_argument('--name', help='Human-readable job name')
    schedule_add_parser.add_argument('--interval', choices=['daily', 'weekly', 'hourly'],
                                   default='daily', help='Schedule interval (default: daily)')
    schedule_add_parser.add_argument('--export-format', choices=['csv', 'excel', 'json'],
                                   default='excel', help='Export format (default: excel)')
    schedule_add_parser.add_argument('--no-notifications', action='store_true',
                                   help='Disable notifications for this job')
    
    # Schedule remove
    schedule_remove_parser = schedule_subparsers.add_parser('remove', help='Remove a scheduled job')
    schedule_remove_parser.add_argument('job_id', help='Job identifier to remove')
    
    # Schedule list
    schedule_subparsers.add_parser('list', help='List all scheduled jobs')
    
    # Schedule status
    schedule_status_parser = schedule_subparsers.add_parser('status', help='Show scheduler status')
    schedule_status_parser.add_argument('job_id', nargs='?', help='Show status for specific job')
    
    # Schedule start/stop
    schedule_start_parser = schedule_subparsers.add_parser('start', help='Start the scheduler service')
    schedule_start_parser.add_argument('--daemon', action='store_true', help='Run as daemon in the background')
    schedule_subparsers.add_parser('stop', help='Stop the scheduler service')
    
    # Schedule run
    schedule_run_parser = schedule_subparsers.add_parser('run', help='Run a scheduled job immediately')
    schedule_run_parser.add_argument('job_id', help='Job identifier to run')
    
    # Schedule enable/disable
    schedule_enable_parser = schedule_subparsers.add_parser('enable', help='Enable a scheduled job')
    schedule_enable_parser.add_argument('job_id', help='Job identifier to enable')
    
    schedule_disable_parser = schedule_subparsers.add_parser('disable', help='Disable a scheduled job')
    schedule_disable_parser.add_argument('job_id', help='Job identifier to disable')
    
    # Schedule report
    schedule_report_parser = schedule_subparsers.add_parser('report', help='Generate scheduler report')
    schedule_report_parser.add_argument('--days', type=int, default=7,
                                      help='Number of days to include in report (default: 7)')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export analysis results in various formats')
    export_parser.add_argument('input_file', help='Input file containing analysis results (JSON format)')
    export_parser.add_argument('--format', '-f', choices=['csv', 'excel', 'powerbi', 'json'],
                              default='excel', help='Export format (default: excel)')
    export_parser.add_argument('--output', '-o', type=str,
                              help='Output filename (without extension)')
    export_parser.add_argument('--output-dir', type=str,
                              help='Output directory (default: ./exports)')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_subparsers = config_parser.add_subparsers(dest='config_action', help='Configuration actions')
    
    # Config show
    config_show_parser = config_subparsers.add_parser('show', help='Show current configuration')
    config_show_parser.add_argument('key', nargs='?', help='Specific configuration key to show')
    
    # Config set
    config_set_parser = config_subparsers.add_parser('set', help='Set configuration value')
    config_set_parser.add_argument('key', help='Configuration key (dot notation)')
    config_set_parser.add_argument('value', help='Configuration value')
    
    # Config reset
    config_subparsers.add_parser('reset', help='Reset configuration to defaults')
    
    # Config validate
    config_subparsers.add_parser('validate', help='Validate current configuration')
    
    return parser


def load_symbols_from_file(filepath: str) -> List[str]:
    """Load stock symbols from a file.
    
    Args:
        filepath: Path to file containing symbols
        
    Returns:
        List[str]: List of stock symbols
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    symbols = []
    
    with open(file_path, 'r') as f:
        content = f.read().strip()
        
        # Try to detect format
        if ',' in content:
            # CSV format - could be single line or multiple lines
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    symbols.extend([s.strip().upper() for s in line.split(',') if s.strip()])
        else:
            # One symbol per line
            for line in content.split('\n'):
                line = line.strip().upper()
                if line and not line.startswith('#'):
                    symbols.append(line)
    
    if not symbols:
        raise ValueError(f"No valid symbols found in file: {filepath}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_symbols = []
    for symbol in symbols:
        if symbol not in seen:
            seen.add(symbol)
            unique_symbols.append(symbol)
    
    return unique_symbols


def handle_analyze_command(args) -> int:
    """Handle the analyze command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        print(f"Analyzing {len(args.symbols)} stock(s): {', '.join(args.symbols)}")
        
        # Create orchestrator
        orchestrator = StockAnalysisOrchestrator(
            enable_parallel_processing=len(args.symbols) > 1,
            continue_on_error=True
        )
        
        # Set up progress indicator
        if args.verbose:
            progress_indicator = ProgressIndicator(verbose=True)
            orchestrator.add_progress_callback(progress_indicator.update_progress)
        
        # Perform analysis
        if len(args.symbols) == 1:
            # Single stock analysis
            result = orchestrator.analyze_single_stock(args.symbols[0])
            results = [result]
            
            # Display summary
            print(f"\nAnalysis completed for {result.symbol}")
            print(f"Company: {result.stock_info.company_name}")
            print(f"Current Price: ${result.stock_info.current_price:.2f}")
            print(f"Health Score: {result.health_score.overall_score:.1f}/100")
            print(f"Recommendation: {result.fair_value.recommendation}")
            print(f"Fair Value: ${result.fair_value.average_fair_value:.2f}")
            print(f"Sentiment: {result.sentiment.overall_sentiment:.2f}")
            
            if args.verbose:
                print("\nRecommendations:")
                for i, rec in enumerate(result.recommendations, 1):
                    print(f"  {i}. {rec}")
        else:
            # Multiple stocks analysis
            report = orchestrator.analyze_multiple_stocks(args.symbols)
            results = report.results
            
            # Display summary
            print(f"\nBatch analysis completed:")
            print(f"Total stocks: {report.total_stocks}")
            print(f"Successful: {report.successful_analyses}")
            print(f"Failed: {report.failed_analyses}")
            print(f"Success rate: {report.success_rate:.1f}%")
            print(f"Execution time: {report.execution_time:.2f} seconds")
            
            if report.failed_symbols:
                print(f"Failed symbols: {', '.join(report.failed_symbols)}")
            
            if args.verbose and results:
                print("\nAnalysis Summary:")
                buy_count = sum(1 for r in results if r.fair_value.recommendation == "BUY")
                sell_count = sum(1 for r in results if r.fair_value.recommendation == "SELL")
                hold_count = sum(1 for r in results if r.fair_value.recommendation == "HOLD")
                avg_health = sum(r.health_score.overall_score for r in results) / len(results)
                
                print(f"  Buy recommendations: {buy_count}")
                print(f"  Hold recommendations: {hold_count}")
                print(f"  Sell recommendations: {sell_count}")
                print(f"  Average health score: {avg_health:.1f}/100")
        
        # Export results if requested
        if not args.no_export and results:
            filename = args.output
            export_path = orchestrator.export_results(results, args.export_format, filename)
            print(f"\nResults exported to: {export_path}")
        
        return 0
        
    except (DataRetrievalError, CalculationError, ExportError) as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_batch_command(args) -> int:
    """Handle the batch command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        # Load symbols from file
        symbols = load_symbols_from_file(args.file)
        print(f"Loaded {len(symbols)} symbols from {args.file}")
        
        if args.verbose:
            print(f"Symbols: {', '.join(symbols)}")
        
        # Create orchestrator
        orchestrator = StockAnalysisOrchestrator(
            max_workers=args.max_workers,
            enable_parallel_processing=args.parallel,
            continue_on_error=args.continue_on_error
        )
        
        # Set up progress indicator
        if args.verbose:
            progress_indicator = ProgressIndicator(verbose=True)
            orchestrator.add_progress_callback(progress_indicator.update_progress)
        
        # Perform batch analysis
        report = orchestrator.analyze_multiple_stocks(symbols)
        
        # Display results
        print(f"\nBatch analysis completed:")
        print(f"Total stocks: {report.total_stocks}")
        print(f"Successful: {report.successful_analyses}")
        print(f"Failed: {report.failed_analyses}")
        print(f"Success rate: {report.success_rate:.1f}%")
        print(f"Execution time: {report.execution_time:.2f} seconds")
        
        if report.failed_symbols:
            print(f"Failed symbols: {', '.join(report.failed_symbols)}")
        
        if args.verbose:
            print(f"\nDetailed summary:")
            print(orchestrator.get_analysis_summary(report))
        
        # Export results
        if report.results:
            filename = args.output
            export_path = orchestrator.export_results(report.results, args.export_format, filename)
            print(f"\nResults exported to: {export_path}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_schedule_command(args) -> int:
    """Handle the schedule command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        scheduler = SchedulerService()
        
        if args.schedule_action == 'add':
            # Add new scheduled job
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
            job_name = args.name or f"Analysis for {', '.join(symbols)}"
            
            scheduler.add_job(
                job_id=args.job_id,
                name=job_name,
                symbols=symbols,
                interval=args.interval,
                export_format=args.export_format,
                notification_enabled=not args.no_notifications
            )
            
            print(f"Scheduled job '{args.job_id}' added successfully")
            print(f"Name: {job_name}")
            print(f"Symbols: {', '.join(symbols)}")
            print(f"Interval: {args.interval}")
            print(f"Export format: {args.export_format}")
            print(f"Notifications: {'Disabled' if args.no_notifications else 'Enabled'}")
            
        elif args.schedule_action == 'remove':
            # Remove scheduled job
            scheduler.remove_job(args.job_id)
            print(f"Scheduled job '{args.job_id}' removed successfully")
            
        elif args.schedule_action == 'list':
            # List all scheduled jobs
            if not scheduler.jobs:
                print("No scheduled jobs found")
                return 0
            
            print(f"Scheduled Jobs ({len(scheduler.jobs)} total):")
            print("-" * 80)
            
            for job_id, job in scheduler.jobs.items():
                status = "Enabled" if job.enabled else "Disabled"
                next_run = job.next_run.strftime('%Y-%m-%d %H:%M') if job.next_run else "Not scheduled"
                
                print(f"ID: {job_id}")
                print(f"  Name: {job.name}")
                print(f"  Status: {status}")
                print(f"  Symbols: {', '.join(job.symbols)}")
                print(f"  Interval: {job.interval}")
                print(f"  Next run: {next_run}")
                print(f"  Success/Failure: {job.success_count}/{job.failure_count}")
                print()
                
        elif args.schedule_action == 'status':
            # Show scheduler status
            if args.job_id:
                # Show specific job status
                job_status = scheduler.get_job_status(args.job_id)
                print(f"Job Status: {job_status['job_id']}")
                print(f"Name: {job_status['name']}")
                print(f"Enabled: {job_status['enabled']}")
                print(f"Symbols: {', '.join(job_status['symbols'])}")
                print(f"Interval: {job_status['interval']}")
                print(f"Last run: {job_status['last_run'] or 'Never'}")
                print(f"Next run: {job_status['next_run'] or 'Not scheduled'}")
                print(f"Success count: {job_status['success_count']}")
                print(f"Failure count: {job_status['failure_count']}")
                if job_status['last_error']:
                    print(f"Last error: {job_status['last_error']}")
            else:
                # Show overall scheduler status
                daemon = SchedulerDaemon()
                is_running = daemon.is_running()
                status_report = scheduler.get_scheduler_status()
                
                print(f"Scheduler Status: {'Running' if is_running else 'Stopped'}")
                print(f"Total jobs: {status_report.total_jobs}")
                print(f"Active jobs: {status_report.active_jobs}")
                print(f"Successful runs today: {status_report.successful_runs_today}")
                print(f"Failed runs today: {status_report.failed_runs_today}")
                
                if is_running:
                    daemon_status = daemon.load_status()
                    if daemon_status.get('started_at'):
                        started_at = datetime.fromisoformat(daemon_status['started_at'])
                        print(f"Started at: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    if daemon_status.get('pid'):
                        print(f"Process ID: {daemon_status['pid']}")
                
                if status_report.next_scheduled_run:
                    print(f"Next scheduled run: {status_report.next_scheduled_run.strftime('%Y-%m-%d %H:%M')}")
                
                if status_report.recent_jobs:
                    print("\nRecent job executions:")
                    for job in status_report.recent_jobs:
                        timestamp = datetime.fromisoformat(job['timestamp']).strftime('%Y-%m-%d %H:%M')
                        status = "Success" if job['success'] else "Failed"
                        print(f"  {timestamp} - {job['job_id']}: {status}")
                        
        elif args.schedule_action == 'start':
            # Create scheduler daemon manager
            daemon = SchedulerDaemon()
            
            # Check if scheduler is already running
            if daemon.is_running():
                print("Scheduler is already running")
                return 0
            
            # Start scheduler
            scheduler.start_scheduler()
            print("Scheduler started successfully")
            
            if args.daemon:
                # Run as daemon in the background
                print("Running as daemon in the background")
                
                # Fork a child process
                try:
                    pid = os.fork()
                    if pid > 0:
                        # Parent process exits
                        return 0
                except OSError as e:
                    print(f"Fork failed: {e}", file=sys.stderr)
                    return 1
                
                # Child process continues
                # Detach from parent environment
                os.setsid()
                
                # Fork a second time
                try:
                    pid = os.fork()
                    if pid > 0:
                        # Exit from second parent
                        sys.exit(0)
                except OSError as e:
                    print(f"Fork failed: {e}", file=sys.stderr)
                    sys.exit(1)
                
                # Redirect standard file descriptors
                sys.stdout.flush()
                sys.stderr.flush()
                
                with open('/dev/null', 'r') as f:
                    os.dup2(f.fileno(), sys.stdin.fileno())
                with open('/dev/null', 'a+') as f:
                    os.dup2(f.fileno(), sys.stdout.fileno())
                with open('/dev/null', 'a+') as f:
                    os.dup2(f.fileno(), sys.stderr.fileno())
                
                # Save daemon status
                daemon.save_pid()
                daemon.save_status(True, os.getpid())
                
                # Keep the process alive to maintain the scheduler
                try:
                    while True:
                        time.sleep(1)
                except:
                    scheduler.stop_scheduler()
                    daemon.save_status(False)
                    daemon.remove_pid_file()
                    sys.exit(0)
            else:
                # Run in foreground
                print("The scheduler is running in the foreground")
                print("Press Ctrl+C to stop")
                
                # Save daemon status
                daemon.save_pid()
                daemon.save_status(True, os.getpid())
                
                # Keep the process alive to maintain the scheduler
                try:
                    while scheduler.is_running:
                        import time
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nShutting down scheduler...")
                    scheduler.stop_scheduler()
                    daemon.save_status(False)
                    daemon.remove_pid_file()
                    print("Scheduler stopped")
            
        elif args.schedule_action == 'stop':
            # Create scheduler daemon manager
            daemon = SchedulerDaemon()
            
            # Check if scheduler is running
            if not daemon.is_running():
                print("Scheduler is not running")
                return 0
            
            # Stop scheduler daemon
            if daemon.stop_daemon():
                print("Scheduler stopped successfully")
            else:
                print("Failed to stop scheduler")
                return 1
            
        elif args.schedule_action == 'run':
            # Run job immediately
            print(f"Running job '{args.job_id}' immediately...")
            report = scheduler.run_job_now(args.job_id)
            
            print(f"Job execution completed:")
            print(f"Total stocks: {report.total_stocks}")
            print(f"Successful: {report.successful_analyses}")
            print(f"Failed: {report.failed_analyses}")
            print(f"Success rate: {report.success_rate:.1f}%")
            print(f"Execution time: {report.execution_time:.2f} seconds")
            
        elif args.schedule_action == 'enable':
            # Enable job
            scheduler.enable_job(args.job_id)
            print(f"Job '{args.job_id}' enabled successfully")
            
        elif args.schedule_action == 'disable':
            # Disable job
            scheduler.disable_job(args.job_id)
            print(f"Job '{args.job_id}' disabled successfully")
            
        elif args.schedule_action == 'report':
            # Generate scheduler report
            report = scheduler.generate_summary_report(args.days)
            print(report)
        
        return 0
        
    except Exception as e:
        print(f"Scheduler error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_export_command(args) -> int:
    """Handle the export command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        from stock_analysis.exporters.export_service import ExportService
        
        # Check if input file exists
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
            return 1
        
        # Load analysis results from JSON file
        print(f"Loading analysis results from: {args.input_file}")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse the data structure - handle both single results and lists
        results = []
        if isinstance(data, list):
            # List of results
            for item in data:
                if isinstance(item, dict) and 'symbol' in item:
                    # Convert dict back to AnalysisResult object
                    result = _dict_to_analysis_result(item)
                    results.append(result)
        elif isinstance(data, dict):
            if 'symbol' in data:
                # Single result
                result = _dict_to_analysis_result(data)
                results.append(result)
            elif 'data' in data:
                # Power BI JSON format with data wrapper
                if isinstance(data['data'], list):
                    for item in data['data']:
                        if isinstance(item, dict) and 'symbol' in item:
                            result = _dict_to_analysis_result(item)
                            results.append(result)
                elif isinstance(data['data'], dict) and 'summary' in data['data']:
                    # Handle nested structure with summary data
                    for item in data['data']['summary']:
                        if isinstance(item, dict) and 'Symbol' in item:
                            # Create a simplified result from summary data
                            result = _summary_to_analysis_result(item)
                            results.append(result)
        
        if not results:
            print("Error: No valid analysis results found in input file", file=sys.stderr)
            return 1
        
        print(f"Loaded {len(results)} analysis result(s)")
        
        # Set up export service
        output_dir = args.output_dir or config.get('stock_analysis.export.output_directory', './exports')
        export_service = ExportService(output_directory=output_dir)
        
        # Determine export method based on format
        export_format = args.format
        if export_format == 'powerbi':
            export_format = 'json'  # Power BI uses JSON format
        
        # Export the results
        if export_format == 'csv':
            filepath = export_service.export_to_csv(results, args.output)
        elif export_format == 'excel':
            filepath = export_service.export_to_excel(results, args.output)
        elif export_format == 'json':
            if args.format == 'powerbi':
                filepath = export_service.export_to_powerbi_json(results, args.output)
            else:
                # Regular JSON export - just save the results as-is
                filename = args.output or f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                if not filename.endswith('.json'):
                    filename += '.json'
                filepath = Path(output_dir) / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump([_analysis_result_to_dict(r) for r in results], f, indent=2, default=str)
                filepath = str(filepath)
        else:
            print(f"Error: Unsupported export format: {export_format}", file=sys.stderr)
            return 1
        
        print(f"Successfully exported {len(results)} result(s) to: {filepath}")
        print(f"Format: {args.format.upper()}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in input file: {str(e)}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Export error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def _summary_to_analysis_result(data: dict) -> AnalysisResult:
    """Convert summary data to AnalysisResult object.
    
    Args:
        data: Dictionary containing summary data
        
    Returns:
        AnalysisResult: Converted analysis result object
    """
    # Create a simplified result from summary data
    stock_info = StockInfo(
        symbol=data.get('Symbol', ''),
        company_name=data.get('Company_Name', ''),
        current_price=data.get('Current_Price', 0.0),
        market_cap=data.get('Market_Cap', 0.0),
        sector=data.get('Sector', ''),
        industry=data.get('Industry', ''),
        pe_ratio=0.0,
        pb_ratio=0.0,
        dividend_yield=0.0,
        beta=0.0
    )
    
    # Create minimal health score
    health_score = HealthScore(
        overall_score=data.get('Overall_Health_Score', 0.0),
        financial_strength=0.0,
        profitability_health=0.0,
        liquidity_health=0.0,
        risk_assessment=data.get('Risk_Assessment', 'Medium')
    )
    
    # Create minimal fair value
    fair_value = FairValueResult(
        current_price=data.get('Current_Price', 0.0),
        dcf_value=0.0,
        peer_comparison_value=0.0,
        average_fair_value=data.get('Average_Fair_Value', 0.0),
        recommendation=data.get('Valuation_Recommendation', 'HOLD'),
        confidence_level=0.0
    )
    
    # Create minimal sentiment
    sentiment = SentimentResult(
        overall_sentiment=data.get('Overall_Sentiment', 0.0),
        positive_count=0,
        negative_count=0,
        neutral_count=0,
        key_themes=[],
        sentiment_trend=[]
    )
    
    # Create timestamp from data or use current time
    timestamp = datetime.fromisoformat(data.get('Timestamp')) if data.get('Timestamp') else datetime.now()
    
    # Create minimal financial ratios
    financial_ratios = FinancialRatios(
        liquidity_ratios=LiquidityRatios(current_ratio=0.0, quick_ratio=0.0, cash_ratio=0.0),
        profitability_ratios=ProfitabilityRatios(
            gross_margin=0.0, operating_margin=0.0, net_profit_margin=0.0, 
            return_on_assets=0.0, return_on_equity=0.0
        ),
        leverage_ratios=LeverageRatios(debt_to_equity=0.0, debt_to_assets=0.0, interest_coverage=0.0),
        efficiency_ratios=EfficiencyRatios(
            asset_turnover=0.0, inventory_turnover=0.0, receivables_turnover=0.0
        )
    )
    
    return AnalysisResult(
        symbol=data.get('Symbol', ''),
        timestamp=timestamp,
        stock_info=stock_info,
        financial_ratios=financial_ratios,
        health_score=health_score,
        fair_value=fair_value,
        sentiment=sentiment,
        recommendations=[]
    )


def _dict_to_analysis_result(data: dict) -> AnalysisResult:
    """Convert dictionary to AnalysisResult object.
    
    Args:
        data: Dictionary containing analysis result data
        
    Returns:
        AnalysisResult: Converted analysis result object
    """
    
    # This is a simplified conversion - in a real scenario, you'd want more robust parsing
    # For now, create minimal objects with the available data
    stock_info = StockInfo(
        symbol=data.get('symbol', ''),
        company_name=data.get('company_name', ''),
        current_price=data.get('current_price', 0.0),
        market_cap=data.get('market_cap', 0.0),
        pe_ratio=data.get('pe_ratio'),
        pb_ratio=data.get('pb_ratio'),
        dividend_yield=data.get('dividend_yield'),
        beta=data.get('beta'),
        sector=data.get('sector'),
        industry=data.get('industry')
    )
    
    # Create minimal financial ratios
    financial_ratios = FinancialRatios(
        liquidity_ratios=LiquidityRatios(),
        profitability_ratios=ProfitabilityRatios(),
        leverage_ratios=LeverageRatios(),
        efficiency_ratios=EfficiencyRatios()
    )
    
    # Create minimal health score
    health_score = HealthScore(
        overall_score=data.get('overall_health_score', 50.0),
        financial_strength=50.0,
        profitability_health=50.0,
        liquidity_health=50.0,
        risk_assessment=data.get('risk_assessment', 'Medium')
    )
    
    # Create minimal fair value
    fair_value = FairValueResult(
        current_price=data.get('current_price', 0.0),
        dcf_value=data.get('dcf_value'),
        peer_comparison_value=data.get('peer_comparison_value'),
        average_fair_value=data.get('average_fair_value', 0.0),
        recommendation=data.get('recommendation', 'HOLD'),
        confidence_level=data.get('confidence_level', 0.5)
    )
    
    # Create minimal sentiment
    sentiment = SentimentResult(
        overall_sentiment=data.get('overall_sentiment', 0.0),
        positive_count=data.get('positive_count', 0),
        negative_count=data.get('negative_count', 0),
        neutral_count=data.get('neutral_count', 0),
        key_themes=data.get('key_themes', ['general'])
    )
    
    return AnalysisResult(
        symbol=data.get('symbol', ''),
        timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
        stock_info=stock_info,
        financial_ratios=financial_ratios,
        health_score=health_score,
        fair_value=fair_value,
        sentiment=sentiment,
        recommendations=data.get('recommendations', [])
    )


def _analysis_result_to_dict(result: AnalysisResult) -> dict:
    """Convert AnalysisResult object to dictionary.
    
    Args:
        result: AnalysisResult object to convert
        
    Returns:
        dict: Dictionary representation of the analysis result
    """
    return {
        'symbol': result.symbol,
        'timestamp': result.timestamp.isoformat(),
        'company_name': result.stock_info.company_name,
        'current_price': result.stock_info.current_price,
        'market_cap': result.stock_info.market_cap,
        'pe_ratio': result.stock_info.pe_ratio,
        'pb_ratio': result.stock_info.pb_ratio,
        'dividend_yield': result.stock_info.dividend_yield,
        'beta': result.stock_info.beta,
        'sector': result.stock_info.sector,
        'industry': result.stock_info.industry,
        'overall_health_score': result.health_score.overall_score,
        'risk_assessment': result.health_score.risk_assessment,
        'dcf_value': result.fair_value.dcf_value,
        'peer_comparison_value': result.fair_value.peer_comparison_value,
        'average_fair_value': result.fair_value.average_fair_value,
        'recommendation': result.fair_value.recommendation,
        'confidence_level': result.fair_value.confidence_level,
        'overall_sentiment': result.sentiment.overall_sentiment,
        'positive_count': result.sentiment.positive_count,
        'negative_count': result.sentiment.negative_count,
        'neutral_count': result.sentiment.neutral_count,
        'key_themes': result.sentiment.key_themes,
        'recommendations': result.recommendations
    }


def handle_config_command(args) -> int:
    """Handle the config command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        if args.config_action == 'show':
            # Show configuration
            if args.key:
                # Show specific key
                value = config.get(args.key)
                if value is not None:
                    print(f"{args.key}: {value}")
                else:
                    print(f"Configuration key '{args.key}' not found")
                    return 1
            else:
                # Show all configuration
                print("Current Configuration:")
                print("-" * 50)
                
                # Pretty print the configuration
                import yaml
                print(yaml.dump(config._config, default_flow_style=False, indent=2))
                
        elif args.config_action == 'set':
            # Set configuration value
            # Try to parse value as appropriate type
            value = args.value
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                value = float(value)
            
            config.set(args.key, value)
            config.save_config()
            print(f"Configuration updated: {args.key} = {value}")
            
        elif args.config_action == 'reset':
            # Reset to defaults
            print("Resetting configuration to defaults...")
            config._config = config._get_default_config()
            config.save_config()
            print("Configuration reset successfully")
            
        elif args.config_action == 'validate':
            # Validate configuration
            print("Validating configuration...")
            
            # Check required directories
            output_dir = config.get('stock_analysis.export.output_directory', './exports')
            log_dir = Path(config.get('stock_analysis.logging.file_path', './logs/stock_analysis.log')).parent
            
            try:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                print("✓ Directory structure is valid")
            except Exception as e:
                print(f"✗ Directory validation failed: {e}")
                return 1
            
            # Check numeric values
            timeout = config.get('stock_analysis.data_sources.yfinance.timeout')
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                print("✗ Invalid yfinance timeout value")
                return 1
            print("✓ Timeout configuration is valid")
            
            # Check export format
            export_format = config.get('stock_analysis.export.default_format')
            if export_format not in ['csv', 'excel', 'json']:
                print("✗ Invalid default export format")
                return 1
            print("✓ Export format configuration is valid")
            
            print("Configuration validation completed successfully")
        
        return 0
        
    except Exception as e:
        print(f"Configuration error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point for the CLI.
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle case where no command is provided
    if not args.command:
        parser.print_help()
        return 1
    
    # Set up logging level if specified
    if args.log_level:
        import logging
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load configuration if specified
    if hasattr(args, 'config') and args.config != 'config.yaml':
        config.config_path = args.config
        config.load_config()
    
    # Route to appropriate command handler
    try:
        if args.command == 'analyze':
            return handle_analyze_command(args)
        elif args.command == 'batch':
            return handle_batch_command(args)
        elif args.command == 'schedule':
            return handle_schedule_command(args)
        elif args.command == 'export':
            return handle_export_command(args)
        elif args.command == 'config':
            return handle_config_command(args)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())