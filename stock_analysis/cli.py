"""Command-line interface for the Stock Analysis Dashboard.

This module provides a comprehensive CLI for stock and ETF analysis operations including
single security analysis, batch processing, scheduling management, and configuration.
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union

from stock_analysis.orchestrator import StockAnalysisOrchestrator, AnalysisProgress
from stock_analysis.services.scheduler_service import SchedulerService
from stock_analysis.services.scheduler_daemon import SchedulerDaemon
from stock_analysis.utils.config import config
from stock_analysis.utils.logging import get_logger
from stock_analysis.utils.exceptions import StockAnalysisError, DataRetrievalError, CalculationError, ExportError
from stock_analysis.models.data_models import (
    AnalysisResult, SecurityInfo, StockInfo, ETFInfo, FinancialRatios, LiquidityRatios, 
    ProfitabilityRatios, LeverageRatios, EfficiencyRatios, HealthScore, FairValueResult, 
    SentimentResult
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
            print(f"\rProgress: {percentage:.1f}% ({progress.completed_securities}/{progress.total_securities} completed, "
                  f"{progress.failed_securities} failed)", end="", flush=True)
            self.last_progress = percentage
            
            if progress.current_security:
                print(f" - Currently analyzing: {progress.current_security}", end="", flush=True)
        
        if percentage == 100:
            print()  # New line when complete


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog='stock-analysis',
        description='Stock Analysis Dashboard - Comprehensive stock and ETF analysis with Power BI integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single stock
  stock-analysis analyze AAPL
  
  # Analyze multiple stocks and ETFs
  stock-analysis analyze AAPL MSFT SCHD IVV --verbose --export-format excel
  
  # Analyze securities from a file
  stock-analysis batch securities.txt --export-format csv
  
  # Schedule daily analysis
  stock-analysis schedule add daily-tech "AAPL,MSFT,SCHD" --interval daily
  
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
    analyze_parser = subparsers.add_parser('analyze', help='Analyze one or more securities (stocks or ETFs)')
    analyze_parser.add_argument('symbols', nargs='+', help='Stock or ETF symbols to analyze')
    analyze_parser.add_argument('--export-format', '-f', choices=['csv', 'excel', 'json'],
                               default='excel', help='Export format (default: excel)')
    analyze_parser.add_argument('--output', '-o', type=str,
                               help='Output filename (without extension)')
    analyze_parser.add_argument('--no-export', action='store_true',
                               help='Skip exporting results')
    analyze_parser.add_argument('--include-technicals', action='store_true',
                               help='Include technical indicators in analysis')
    analyze_parser.add_argument('--include-analyst', action='store_true',
                               help='Include analyst recommendations and price targets')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch analyze securities from file')
    batch_parser.add_argument('file', help='File containing stock/ETF symbols (one per line or CSV)')
    batch_parser.add_argument('--export-format', '-f', choices=['csv', 'excel', 'json'],
                             default='excel', help='Export format (default: excel)')
    batch_parser.add_argument('--output', '-o', type=str,
                             help='Output filename (without extension)')
    batch_parser.add_argument('--parallel', action='store_true', default=True,
                             help='Enable parallel processing (default: enabled)')
    batch_parser.add_argument('--max-workers', type=int, default=4,
                             help='Maximum number of worker threads (default: 4)')
    batch_parser.add_argument('--continue-on-error', action='store_true', default=True,
                             help='Continue processing if individual securities fail (default: enabled)')
    
    # Market command
    market_parser = subparsers.add_parser('market', help='Get market data and economic indicators')
    market_parser.add_argument('--indices', action='store_true',
                             help='Show major market indices')
    market_parser.add_argument('--sectors', action='store_true',
                             help='Show sector performance')
    market_parser.add_argument('--commodities', action='store_true',
                             help='Show commodity prices')
    market_parser.add_argument('--forex', action='store_true',
                             help='Show foreign exchange rates')
    market_parser.add_argument('--economic', action='store_true',
                             help='Show economic indicators')
    market_parser.add_argument('--region', type=str,
                             help='Filter economic indicators by region (e.g., US, EU)')
    market_parser.add_argument('--export-format', '-f', choices=['csv', 'excel', 'json'],
                             help='Export format (if exporting results)')
    market_parser.add_argument('--output', '-o', type=str,
                             help='Output filename (without extension)')
    market_parser.add_argument('--no-export', action='store_true',
                             help='Skip exporting results')
    
    # News command
    news_parser = subparsers.add_parser('news', help='Get financial news and economic events')
    news_parser.add_argument('--symbol', type=str,
                           help='Get news for specific symbol')
    news_parser.add_argument('--market', action='store_true',
                           help='Get general market news')
    news_parser.add_argument('--category', type=str,
                           help='Filter news by category (e.g., earnings, economy, technology)')
    news_parser.add_argument('--economic-calendar', action='store_true',
                           help='Show economic calendar')
    news_parser.add_argument('--days', type=int, default=7,
                           help='Number of days to look back/ahead (default: 7)')
    news_parser.add_argument('--limit', type=int, default=10,
                           help='Number of news items to retrieve (default: 10)')
    news_parser.add_argument('--sentiment', action='store_true',
                           help='Include sentiment analysis')
    news_parser.add_argument('--trending', action='store_true',
                           help='Show trending topics')
    news_parser.add_argument('--export-format', '-f', choices=['csv', 'excel', 'json'],
                           help='Export format (if exporting results)')
    news_parser.add_argument('--output', '-o', type=str,
                           help='Output filename (without extension)')
    news_parser.add_argument('--no-export', action='store_true',
                           help='Skip exporting results')
    
    # Financials command
    financials_parser = subparsers.add_parser('financials', help='Get detailed financial data')
    financials_parser.add_argument('symbol', help='Stock symbol')
    financials_parser.add_argument('--statement', choices=['income', 'balance', 'cash', 'all'],
                                 default='all', help='Financial statement type')
    financials_parser.add_argument('--period', choices=['annual', 'quarterly'],
                                 default='annual', help='Reporting period')
    financials_parser.add_argument('--years', type=int, default=5,
                                 help='Number of years of historical data')
    financials_parser.add_argument('--growth', action='store_true',
                                 help='Show growth metrics')
    financials_parser.add_argument('--compare-industry', action='store_true',
                                 help='Compare with industry averages')
    financials_parser.add_argument('--export-format', '-f', choices=['csv', 'excel', 'json'],
                                 help='Export format (if exporting results)')
    financials_parser.add_argument('--output', '-o', type=str,
                                 help='Output filename (without extension)')
    financials_parser.add_argument('--no-export', action='store_true',
                                 help='Skip exporting results')
    
    # Schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Manage scheduled analysis jobs')
    schedule_subparsers = schedule_parser.add_subparsers(dest='schedule_action', help='Schedule actions')
    
    # Schedule add
    schedule_add_parser = schedule_subparsers.add_parser('add', help='Add a new scheduled job')
    schedule_add_parser.add_argument('job_id', help='Unique job identifier')
    schedule_add_parser.add_argument('symbols', help='Comma-separated list of stock/ETF symbols')
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
        print(f"Analyzing {len(args.symbols)} security(s): {', '.join(args.symbols)}")
        
        # Create orchestrator with enhanced options
        orchestrator = StockAnalysisOrchestrator(
            enable_parallel_processing=len(args.symbols) > 1,
            continue_on_error=True,
            include_technicals=args.include_technicals,
            include_analyst=args.include_analyst
        )
        
        # Set up progress indicator
        if args.verbose:
            progress_indicator = ProgressIndicator(verbose=True)
            orchestrator.add_progress_callback(progress_indicator.update_progress)
        
        # Perform analysis
        if len(args.symbols) == 1:
            # Single security analysis
            result = orchestrator.analyze_single_security(args.symbols[0])
            results = [result]
            
            # Display summary
            print(f"\nAnalysis completed for {result.symbol}")
            
            if isinstance(result.stock_info, ETFInfo):
                # ETF summary
                print(f"ETF: {result.stock_info.name}")
                print(f"Current Price: ${result.stock_info.current_price:.2f}")
                print(f"NAV: ${result.stock_info.nav:.2f}" if result.stock_info.nav else "NAV: Not available")
                print(f"Expense Ratio: {result.stock_info.expense_ratio*100:.2f}%" if result.stock_info.expense_ratio else "Expense Ratio: Not available")
                print(f"Assets Under Management: ${result.stock_info.assets_under_management/1e9:.2f}B" if result.stock_info.assets_under_management else "AUM: Not available")
                print(f"Category: {result.stock_info.category}" if result.stock_info.category else "Category: Not available")
                if result.stock_info.holdings:
                    print(f"Number of Holdings: {len(result.stock_info.holdings)}")
                    print("\nTop 5 Holdings:")
                    for holding in result.stock_info.holdings[:5]:
                        print(f"  {holding['symbol']} ({holding['name']}): {holding['weight']*100:.2f}%")
            else:
                # Stock summary
                from stock_analysis.models.enhanced_data_models import EnhancedSecurityInfo
                
                if isinstance(result.stock_info, EnhancedSecurityInfo):
                    # Enhanced security info has 'name' instead of 'company_name'
                    print(f"Company: {result.stock_info.name}")
                else:
                    print(f"Company: {result.stock_info.company_name}")
                    
                print(f"Current Price: ${result.stock_info.current_price:.2f}")
                
                # Handle sector and industry which might not be in EnhancedSecurityInfo
                if hasattr(result.stock_info, 'sector') and result.stock_info.sector:
                    print(f"Sector: {result.stock_info.sector}")
                
                if hasattr(result.stock_info, 'industry') and result.stock_info.industry:
                    print(f"Industry: {result.stock_info.industry}")
                
                # Handle PE and PB ratios which might not be in EnhancedSecurityInfo
                if hasattr(result.stock_info, 'pe_ratio') and result.stock_info.pe_ratio:
                    print(f"P/E Ratio: {result.stock_info.pe_ratio:.2f}")
                else:
                    print("P/E Ratio: Not available")
                    
                if hasattr(result.stock_info, 'pb_ratio') and result.stock_info.pb_ratio:
                    print(f"P/B Ratio: {result.stock_info.pb_ratio:.2f}")
                else:
                    print("P/B Ratio: Not available")
                
                # Display enhanced data if available
                from stock_analysis.models.enhanced_data_models import EnhancedSecurityInfo
                if isinstance(result.stock_info, EnhancedSecurityInfo):
                    print("\nEnhanced Data:")
                    
                    # Display technical indicators if available
                    if args.include_technicals:
                        print("\nTechnical Indicators:")
                        if result.stock_info.rsi_14 is not None:
                            print(f"  RSI (14): {result.stock_info.rsi_14:.2f}")
                        
                        if result.stock_info.moving_averages:
                            print("  Moving Averages:")
                            for ma_name, ma_value in result.stock_info.moving_averages.items():
                                if ma_value is not None:
                                    print(f"    {ma_name}: ${ma_value:.2f}")
                        
                        if result.stock_info.macd:
                            print("  MACD:")
                            macd = result.stock_info.macd
                            print(f"    MACD Line: {macd.get('macd_line', 0):.4f}")
                            print(f"    Signal Line: {macd.get('signal_line', 0):.4f}")
                            print(f"    Histogram: {macd.get('histogram', 0):.4f}")
                    
                    # Display analyst data if available
                    if args.include_analyst:
                        print("\nAnalyst Data:")
                        if result.stock_info.analyst_rating:
                            print(f"  Analyst Rating: {result.stock_info.analyst_rating}")
                        
                        if result.stock_info.price_target:
                            pt = result.stock_info.price_target
                            print(f"  Price Target: ${pt.get('average', 0):.2f} (Low: ${pt.get('low', 0):.2f}, High: ${pt.get('high', 0):.2f})")
                        
                        if result.stock_info.analyst_count:
                            print(f"  Analyst Count: {result.stock_info.analyst_count}")
                    
                    # Display additional metadata
                    if result.stock_info.exchange:
                        print(f"\nExchange: {result.stock_info.exchange}")
                    
                    if result.stock_info.earnings_growth is not None:
                        print(f"Earnings Growth: {result.stock_info.earnings_growth:.2%}")
                    
                    if result.stock_info.revenue_growth is not None:
                        print(f"Revenue Growth: {result.stock_info.revenue_growth:.2%}")
            
            # Common metrics
            print(f"\nHealth Score: {result.health_score.overall_score:.1f}/100")
            print(f"Recommendation: {result.fair_value.recommendation}")
            print(f"Fair Value: ${result.fair_value.average_fair_value:.2f}")
            print(f"Sentiment: {result.sentiment.overall_sentiment:.2f}")
            
            if args.verbose:
                print("\nRecommendations:")
                for i, rec in enumerate(result.recommendations, 1):
                    print(f"  {i}. {rec}")
        else:
            # Multiple securities analysis
            report = orchestrator.analyze_multiple_securities(args.symbols)
            results = report.results
            
            # Display summary
            print(f"\nBatch analysis completed:")
            print(f"Total securities: {report.total_securities}")
            print(f"Successful: {report.successful_analyses}")
            print(f"Failed: {report.failed_analyses}")
            print(f"Success rate: {report.success_rate:.1f}%")
            print(f"Execution time: {report.execution_time:.2f} seconds")
            
            if report.failed_symbols:
                print(f"Failed symbols: {', '.join(report.failed_symbols)}")
            
            if args.verbose and results:
                print("\nAnalysis Summary:")
                
                # Count stocks vs ETFs
                stocks = [r for r in results if isinstance(r.stock_info, StockInfo)]
                etfs = [r for r in results if isinstance(r.stock_info, ETFInfo)]
                
                print(f"  Stocks analyzed: {len(stocks)}")
                print(f"  ETFs analyzed: {len(etfs)}")
                
                # Recommendations
                buy_count = sum(1 for r in results if r.fair_value.recommendation == "BUY")
                sell_count = sum(1 for r in results if r.fair_value.recommendation == "SELL")
                hold_count = sum(1 for r in results if r.fair_value.recommendation == "HOLD")
                avg_health = sum(r.health_score.overall_score for r in results) / len(results)
                
                print(f"\n  Buy recommendations: {buy_count}")
                print(f"  Hold recommendations: {hold_count}")
                print(f"  Sell recommendations: {sell_count}")
                print(f"  Average health score: {avg_health:.1f}/100")
                
                # ETF-specific summary
                if etfs:
                    print("\n  ETF Summary:")
                    avg_expense = sum(e.stock_info.expense_ratio for e in etfs if e.stock_info.expense_ratio) / len([e for e in etfs if e.stock_info.expense_ratio])
                    total_aum = sum(e.stock_info.assets_under_management for e in etfs if e.stock_info.assets_under_management)
                    print(f"    Average expense ratio: {avg_expense*100:.2f}%")
                    print(f"    Total AUM: ${total_aum/1e9:.2f}B")
        
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


def handle_market_command(args) -> int:
    """Handle the market command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        from stock_analysis.services.market_data_service import MarketDataService
        from stock_analysis.exporters.export_service import ExportService
        
        # Initialize services
        market_service = MarketDataService()
        export_service = ExportService()
        
        # Determine what data to retrieve
        show_all = not any([args.indices, args.sectors, args.commodities, args.forex, args.economic])
        
        # Initialize results dictionary for potential export
        results = {}
        
        # Get market overview if no specific options are selected
        if show_all:
            print("Retrieving market overview...")
            market_data = market_service.get_market_overview()
            
            # Display indices
            print("\nMajor Market Indices:")
            print("-" * 60)
            print(f"{'Index':<20} {'Value':<12} {'Change':<10} {'% Change':<10}")
            print("-" * 60)
            for name, data in market_data.indices.items():
                value = data.get('value', 0)
                change = data.get('change', 0)
                pct_change = data.get('change_percent', 0)
                print(f"{name:<20} {value:<12,.2f} {change:<+10,.2f} {pct_change:<+10,.2f}%")
            
            # Display sector performance
            print("\nSector Performance:")
            print("-" * 40)
            print(f"{'Sector':<20} {'Performance':<10}")
            print("-" * 40)
            for sector, performance in market_data.sector_performance.items():
                print(f"{sector:<20} {performance:<+10,.2f}%")
            
            # Display commodities
            print("\nCommodity Prices:")
            print("-" * 60)
            print(f"{'Commodity':<20} {'Price':<12} {'Change':<10} {'% Change':<10}")
            print("-" * 60)
            for name, data in market_data.commodities.items():
                value = data.get('value', 0)
                change = data.get('change', 0)
                pct_change = data.get('change_percent', 0)
                print(f"{name:<20} {value:<12,.2f} {change:<+10,.2f} {pct_change:<+10,.2f}%")
            
            # Display forex
            print("\nForex Rates:")
            print("-" * 60)
            print(f"{'Currency Pair':<20} {'Rate':<12} {'Change':<10} {'% Change':<10}")
            print("-" * 60)
            for name, data in market_data.forex.items():
                value = data.get('value', 0)
                change = data.get('change', 0)
                pct_change = data.get('change_percent', 0)
                print(f"{name:<20} {value:<12,.4f} {change:<+10,.4f} {pct_change:<+10,.2f}%")
            
            # Display economic indicators
            print("\nKey Economic Indicators:")
            print("-" * 80)
            print(f"{'Indicator':<25} {'Value':<10} {'Previous':<10} {'Forecast':<10} {'Unit':<10}")
            print("-" * 80)
            for name, data in market_data.economic_indicators.items():
                value = data.get('value', 'N/A')
                previous = data.get('previous', 'N/A')
                forecast = data.get('forecast', 'N/A')
                unit = data.get('unit', '')
                
                # Format numeric values
                if isinstance(value, (int, float)):
                    value = f"{value:.2f}"
                if isinstance(previous, (int, float)):
                    previous = f"{previous:.2f}"
                if isinstance(forecast, (int, float)):
                    forecast = f"{forecast:.2f}"
                
                print(f"{name:<25} {value:<10} {previous:<10} {forecast:<10} {unit:<10}")
            
            # Store all data for potential export
            results = {
                'indices': market_data.indices,
                'sectors': market_data.sector_performance,
                'commodities': market_data.commodities,
                'forex': market_data.forex,
                'economic_indicators': market_data.economic_indicators
            }
            
        else:
            # Retrieve and display specific data types
            
            # Indices
            if args.indices:
                print("Retrieving market indices...")
                indices = market_service.get_market_indices()
                
                print("\nMajor Market Indices:")
                print("-" * 60)
                print(f"{'Index':<20} {'Value':<12} {'Change':<10} {'% Change':<10}")
                print("-" * 60)
                for name, data in indices.items():
                    value = data.get('value', 0)
                    change = data.get('change', 0)
                    pct_change = data.get('change_percent', 0)
                    print(f"{name:<20} {value:<12,.2f} {change:<+10,.2f} {pct_change:<+10,.2f}%")
                
                results['indices'] = indices
            
            # Sectors
            if args.sectors:
                print("Retrieving sector performance...")
                sectors = market_service.get_sector_performance()
                
                print("\nSector Performance:")
                print("-" * 40)
                print(f"{'Sector':<20} {'Performance':<10}")
                print("-" * 40)
                for sector, performance in sectors.items():
                    print(f"{sector:<20} {performance:<+10,.2f}%")
                
                results['sectors'] = sectors
            
            # Commodities
            if args.commodities:
                print("Retrieving commodity prices...")
                commodities = market_service.get_commodity_prices()
                
                print("\nCommodity Prices:")
                print("-" * 60)
                print(f"{'Commodity':<20} {'Price':<12} {'Change':<10} {'% Change':<10}")
                print("-" * 60)
                for name, data in commodities.items():
                    value = data.get('value', 0)
                    change = data.get('change', 0)
                    pct_change = data.get('change_percent', 0)
                    print(f"{name:<20} {value:<12,.2f} {change:<+10,.2f} {pct_change:<+10,.2f}%")
                
                results['commodities'] = commodities
            
            # Forex
            if args.forex:
                print("Retrieving forex rates...")
                forex = market_service.get_market_data('forex')
                
                print("\nForex Rates:")
                print("-" * 60)
                print(f"{'Currency Pair':<20} {'Rate':<12} {'Change':<10} {'% Change':<10}")
                print("-" * 60)
                for name, data in forex.items():
                    value = data.get('value', 0)
                    change = data.get('change', 0)
                    pct_change = data.get('change_percent', 0)
                    print(f"{name:<20} {value:<12,.4f} {change:<+10,.4f} {pct_change:<+10,.2f}%")
                
                results['forex'] = forex
            
            # Economic indicators
            if args.economic:
                print("Retrieving economic indicators...")
                indicators = market_service.get_economic_indicators(region=args.region)
                
                print("\nKey Economic Indicators:")
                print("-" * 80)
                print(f"{'Indicator':<25} {'Value':<10} {'Previous':<10} {'Forecast':<10} {'Unit':<10}")
                print("-" * 80)
                for name, data in indicators.items():
                    value = data.get('value', 'N/A')
                    previous = data.get('previous', 'N/A')
                    forecast = data.get('forecast', 'N/A')
                    unit = data.get('unit', '')
                    
                    # Format numeric values
                    if isinstance(value, (int, float)):
                        value = f"{value:.2f}"
                    if isinstance(previous, (int, float)):
                        previous = f"{previous:.2f}"
                    if isinstance(forecast, (int, float)):
                        forecast = f"{forecast:.2f}"
                    
                    print(f"{name:<25} {value:<10} {previous:<10} {forecast:<10} {unit:<10}")
                
                results['economic_indicators'] = indicators
        
        # Export results if requested
        if not args.no_export and args.export_format and results:
            export_format = args.export_format
            filename = args.output or f"market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if export_format == 'json':
                export_path = export_service.export_to_json(results, filename)
            elif export_format == 'csv':
                export_path = export_service.export_to_csv(results, filename)
            elif export_format == 'excel':
                export_path = export_service.export_to_excel(results, filename)
            
            print(f"\nResults exported to: {export_path}")
        
        return 0
        
    except DataRetrievalError as e:
        print(f"Error retrieving market data: {str(e)}", file=sys.stderr)
        return 1
    except ExportError as e:
        print(f"Error exporting results: {str(e)}", file=sys.stderr)
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
            report = scheduler.generate_report(days=args.days)
            
            print(f"Scheduler Report (Last {args.days} days):")
            print("-" * 80)
            print(f"Total jobs: {report['total_jobs']}")
            print(f"Active jobs: {report['active_jobs']}")
            print(f"Total executions: {report['total_executions']}")
            print(f"Successful executions: {report['successful_executions']}")
            print(f"Failed executions: {report['failed_executions']}")
            print(f"Success rate: {report['success_rate']:.1f}%")
            
            if report['job_stats']:
                print("\nJob Statistics:")
                for job_id, stats in report['job_stats'].items():
                    print(f"  {job_id}:")
                    print(f"    Executions: {stats['executions']}")
                    print(f"    Success rate: {stats['success_rate']:.1f}%")
                    print(f"    Average execution time: {stats['avg_execution_time']:.2f}s")
            
            if report['recent_failures']:
                print("\nRecent Failures:")
                for failure in report['recent_failures']:
                    print(f"  {failure['timestamp']} - {failure['job_id']}: {failure['error']}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
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
        
        # Initialize export service
        export_service = ExportService()
        
        # Load input file
        print(f"Loading analysis results from {args.input_file}...")
        with open(args.input_file, 'r') as f:
            data = json.load(f)
        
        # Export to specified format
        export_format = args.format
        output_filename = args.output
        output_dir = args.output_dir
        
        print(f"Exporting to {export_format} format...")
        
        if export_format == 'csv':
            output_path = export_service.export_to_csv(data, output_filename, output_dir)
        elif export_format == 'excel':
            output_path = export_service.export_to_excel(data, output_filename, output_dir)
        elif export_format == 'powerbi':
            output_path = export_service.export_to_powerbi_json(data, output_filename, output_dir)
        elif export_format == 'json':
            output_path = export_service.export_to_json(data, output_filename, output_dir)
        
        print(f"Results exported to: {output_path}")
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in input file: {str(e)}", file=sys.stderr)
        return 1
    except ExportError as e:
        print(f"Export error: {str(e)}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


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
                # Show specific configuration key
                value = config.get(args.key)
                if value is None:
                    print(f"Configuration key '{args.key}' not found")
                    return 1
                
                print(f"{args.key}: {value}")
            else:
                # Show all configuration
                print("Current Configuration:")
                for key, value in config.get_all().items():
                    print(f"{key}: {value}")
        
        elif args.config_action == 'set':
            # Set configuration value
            config.set(args.key, args.value)
            config.save()
            print(f"Configuration key '{args.key}' set to '{args.value}'")
            
        elif args.config_action == 'reset':
            # Reset configuration to defaults
            config.reset_to_defaults()
            config.save()
            print("Configuration reset to defaults")
            
        elif args.config_action == 'validate':
            # Validate configuration
            is_valid, issues = config.validate()
            
            if is_valid:
                print("Configuration is valid")
            else:
                print("Configuration validation failed:")
                for issue in issues:
                    print(f"  - {issue}")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point for the CLI.
    
    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging
    if args.log_level:
        logger.setLevel(args.log_level)
    
    # Load configuration
    if args.config:
        config.config_path = args.config
        config.load_config()
    
    # Handle commands
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
    elif args.command == 'market':
        return handle_market_command(args)
    elif args.command == 'news':
        return handle_news_command(args)
    elif args.command == 'financials':
        return handle_financials_command(args)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())


def handle_news_command(args) -> int:
    """Handle the news command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        from stock_analysis.services.news_service import NewsService
        from stock_analysis.exporters.export_service import ExportService
        
        # Initialize services
        news_service = NewsService()
        export_service = ExportService()
        
        # Initialize results dictionary for potential export
        results = {}
        
        # Determine what data to retrieve
        show_all = not any([args.symbol, args.market, args.economic_calendar, args.trending])
        
        # Company news for specific symbol
        if args.symbol:
            print(f"Retrieving news for {args.symbol}...")
            news_items = news_service.get_company_news(
                symbol=args.symbol,
                days=args.days,
                limit=args.limit,
                include_sentiment=args.sentiment
            )
            
            print(f"\nNews for {args.symbol}:")
            print("-" * 80)
            
            if not news_items:
                print("No news found.")
            else:
                for i, item in enumerate(news_items, 1):
                    published_at = item.published_at.strftime('%Y-%m-%d %H:%M') if item.published_at else 'N/A'
                    print(f"{i}. {item.title}")
                    print(f"   Source: {item.source} | Published: {published_at}")
                    
                    if args.sentiment and item.sentiment is not None:
                        sentiment_str = "Positive" if item.sentiment > 0.2 else "Negative" if item.sentiment < -0.2 else "Neutral"
                        print(f"   Sentiment: {sentiment_str} ({item.sentiment:.2f})")
                    
                    if item.impact:
                        print(f"   Impact: {item.impact}")
                    
                    if item.summary:
                        print(f"   {item.summary}")
                    
                    print()
            
            # Get sentiment analysis if requested
            if args.sentiment:
                print("\nSentiment Analysis:")
                sentiment = news_service.get_news_sentiment(args.symbol, args.days)
                print(f"Overall Sentiment: {sentiment.overall_sentiment:.2f} (-1 to 1 scale)")
                print(f"Positive Articles: {sentiment.positive_count}")
                print(f"Negative Articles: {sentiment.negative_count}")
                print(f"Neutral Articles: {sentiment.neutral_count}")
                
                if sentiment.key_themes:
                    print("\nKey Themes:")
                    for theme in sentiment.key_themes:
                        print(f"- {theme}")
            
            # Store results for potential export
            results['company_news'] = [item.__dict__ for item in news_items]
            if args.sentiment:
                results['sentiment'] = {
                    'overall_sentiment': sentiment.overall_sentiment,
                    'positive_count': sentiment.positive_count,
                    'negative_count': sentiment.negative_count,
                    'neutral_count': sentiment.neutral_count,
                    'key_themes': sentiment.key_themes
                }
        
        # Market news
        if args.market or show_all:
            print("Retrieving market news...")
            news_items = news_service.get_market_news(
                category=args.category,
                days=args.days,
                limit=args.limit,
                include_sentiment=args.sentiment
            )
            
            print("\nMarket News:")
            print("-" * 80)
            
            if not news_items:
                print("No news found.")
            else:
                for i, item in enumerate(news_items, 1):
                    published_at = item.published_at.strftime('%Y-%m-%d %H:%M') if item.published_at else 'N/A'
                    print(f"{i}. {item.title}")
                    print(f"   Source: {item.source} | Published: {published_at}")
                    
                    if args.sentiment and item.sentiment is not None:
                        sentiment_str = "Positive" if item.sentiment > 0.2 else "Negative" if item.sentiment < -0.2 else "Neutral"
                        print(f"   Sentiment: {sentiment_str} ({item.sentiment:.2f})")
                    
                    if item.categories:
                        print(f"   Categories: {', '.join(item.categories)}")
                    
                    if item.impact:
                        print(f"   Impact: {item.impact}")
                    
                    if item.summary:
                        print(f"   {item.summary}")
                    
                    print()
            
            # Store results for potential export
            results['market_news'] = [item.__dict__ for item in news_items]
        
        # Economic calendar
        if args.economic_calendar or show_all:
            print("Retrieving economic calendar...")
            calendar_events = news_service.get_economic_calendar(
                days_ahead=args.days,
                days_behind=1
            )
            
            print("\nEconomic Calendar:")
            print("-" * 100)
            print(f"{'Date':<12} {'Time':<8} {'Country':<10} {'Event':<30} {'Importance':<10} {'Actual':<8} {'Forecast':<8} {'Previous':<8}")
            print("-" * 100)
            
            if not calendar_events:
                print("No economic events found.")
            else:
                # Sort events by date
                sorted_events = sorted(calendar_events, key=lambda x: x.get('date', ''))
                
                for event in sorted_events:
                    date = event.get('date', 'N/A')
                    time = event.get('time', 'N/A')
                    country = event.get('country', 'N/A')
                    name = event.get('event', 'N/A')
                    importance = event.get('importance', 'N/A')
                    actual = event.get('actual', 'N/A')
                    forecast = event.get('forecast', 'N/A')
                    previous = event.get('previous', 'N/A')
                    
                    # Truncate long event names
                    if len(name) > 28:
                        name = name[:25] + '...'
                    
                    print(f"{date:<12} {time:<8} {country:<10} {name:<30} {importance:<10} {actual:<8} {forecast:<8} {previous:<8}")
            
            # Store results for potential export
            results['economic_calendar'] = calendar_events
        
        # Trending topics
        if args.trending:
            print("Retrieving trending topics...")
            trending_topics = news_service.get_trending_topics(days=args.days)
            
            print("\nTrending Topics:")
            print("-" * 60)
            print(f"{'Topic':<30} {'Mentions':<10} {'Sentiment':<10}")
            print("-" * 60)
            
            if not trending_topics:
                print("No trending topics found.")
            else:
                for topic in trending_topics:
                    name = topic.get('topic', 'N/A')
                    count = topic.get('count', 0)
                    sentiment = topic.get('sentiment', 0)
                    sentiment_str = f"{sentiment:.2f}"
                    
                    print(f"{name:<30} {count:<10} {sentiment_str:<10}")
            
            # Store results for potential export
            results['trending_topics'] = trending_topics
        
        # Export results if requested
        if not args.no_export and args.export_format and results:
            export_format = args.export_format
            filename = args.output or f"news_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if export_format == 'json':
                export_path = export_service.export_to_json(results, filename)
            elif export_format == 'csv':
                export_path = export_service.export_to_csv(results, filename)
            elif export_format == 'excel':
                export_path = export_service.export_to_excel(results, filename)
            
            print(f"\nResults exported to: {export_path}")
        
        return 0
        
    except DataRetrievalError as e:
        print(f"Error retrieving news data: {str(e)}", file=sys.stderr)
        return 1
    except ExportError as e:
        print(f"Error exporting results: {str(e)}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_financials_command(args) -> int:
    """Handle the financials command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
        from stock_analysis.exporters.export_service import ExportService
        
        # Initialize services
        integration_service = FinancialDataIntegrationService()
        export_service = ExportService()
        
        symbol = args.symbol.upper()
        statement_type = args.statement
        period = args.period
        years = args.years
        show_growth = args.growth
        compare_industry = args.compare_industry
        
        print(f"Retrieving financial data for {symbol}...")
        
        # Initialize results dictionary for potential export
        results = {}
        
        # Get company info
        print("Retrieving company information...")
        company_info = integration_service.get_security_info(symbol)
        
        print(f"\nCompany: {company_info.get('name', symbol)}")
        print(f"Sector: {company_info.get('sector', 'N/A')}")
        print(f"Industry: {company_info.get('industry', 'N/A')}")
        print(f"Exchange: {company_info.get('exchange', 'N/A')}")
        
        # Store company info for potential export
        results['company_info'] = company_info
        
        # Get financial statements
        if statement_type in ['income', 'all']:
            print("\nRetrieving income statement...")
            income_statement = integration_service.get_financial_statements(
                symbol, 
                statement_type='income', 
                period=period,
                years=years
            )
            
            print("\nIncome Statement:")
            print("-" * 100)
            
            if income_statement is None or income_statement.empty:
                print("No income statement data available.")
            else:
                # Print column headers (dates)
                header = f"{'Item':<40}"
                for col in income_statement.columns:
                    header += f"{col:<15}"
                print(header)
                print("-" * 100)
                
                # Print key metrics
                key_metrics = [
                    'Revenue', 'Cost of Revenue', 'Gross Profit', 'Operating Expenses',
                    'Operating Income', 'Net Income', 'EPS', 'EBITDA'
                ]
                
                for metric in key_metrics:
                    if metric in income_statement.index.values:
                        row = f"{metric:<40}"
                        for value in income_statement.loc[metric].values:
                            if isinstance(value, (int, float)):
                                if abs(value) >= 1e9:
                                    row += f"${value/1e9:<14.2f}B"
                                elif abs(value) >= 1e6:
                                    row += f"${value/1e6:<14.2f}M"
                                else:
                                    row += f"${value:<14.2f}"
                            else:
                                row += f"{str(value):<15}"
                        print(row)
                
                # Show growth metrics if requested
                if show_growth and len(income_statement.columns) > 1:
                    print("\nGrowth Metrics:")
                    print("-" * 100)
                    
                    for metric in ['Revenue', 'Net Income', 'EPS']:
                        if metric in income_statement.index.values:
                            values = income_statement.loc[metric].values
                            if len(values) > 1:
                                row = f"{metric} Growth %:<40"
                                for i in range(1, len(values)):
                                    if values[i-1] != 0 and isinstance(values[i], (int, float)) and isinstance(values[i-1], (int, float)):
                                        growth = (values[i] - values[i-1]) / abs(values[i-1]) * 100
                                        row += f"{growth:<14.2f}%"
                                    else:
                                        row += f"{'N/A':<15}"
                                print(row)
            
            # Store income statement for potential export
            results['income_statement'] = income_statement
        
        if statement_type in ['balance', 'all']:
            print("\nRetrieving balance sheet...")
            balance_sheet = integration_service.get_financial_statements(
                symbol, 
                statement_type='balance', 
                period=period,
                years=years
            )
            
            print("\nBalance Sheet:")
            print("-" * 100)
            
            if balance_sheet is None or balance_sheet.empty:
                print("No balance sheet data available.")
            else:
                # Print column headers (dates)
                header = f"{'Item':<40}"
                for col in balance_sheet.columns:
                    header += f"{col:<15}"
                print(header)
                print("-" * 100)
                
                # Print key metrics
                key_metrics = [
                    'Total Assets', 'Current Assets', 'Cash and Cash Equivalents',
                    'Total Liabilities', 'Current Liabilities', 'Long Term Debt',
                    'Total Equity', 'Retained Earnings'
                ]
                
                for metric in key_metrics:
                    if metric in balance_sheet.index.values:
                        values = balance_sheet.loc[metric].values
                        row = f"{metric:<40}"
                        for value in values:
                            if isinstance(value, (int, float)):
                                if abs(value) >= 1e9:
                                    row += f"${value/1e9:<14.2f}B"
                                elif abs(value) >= 1e6:
                                    row += f"${value/1e6:<14.2f}M"
                                else:
                                    row += f"${value:<14.2f}"
                            else:
                                row += f"{str(value):<15}"
                        print(row)
            
            # Store balance sheet for potential export
            results['balance_sheet'] = balance_sheet
        
        if statement_type in ['cash', 'all']:
            print("\nRetrieving cash flow statement...")
            cash_flow = integration_service.get_financial_statements(
                symbol, 
                statement_type='cash', 
                period=period,
                years=years
            )
            
            print("\nCash Flow Statement:")
            print("-" * 100)
            
            if cash_flow is None or cash_flow.empty:
                print("No cash flow statement data available.")
            else:
                # Print column headers (dates)
                header = f"{'Item':<40}"
                for col in cash_flow.columns:
                    header += f"{col:<15}"
                print(header)
                print("-" * 100)
                
                # Print key metrics
                key_metrics = [
                    'Operating Cash Flow', 'Capital Expenditure', 'Free Cash Flow',
                    'Investing Cash Flow', 'Financing Cash Flow', 'Net Cash Flow'
                ]
                
                for metric in key_metrics:
                    if metric in cash_flow.index.values:
                        values = cash_flow.loc[metric].values
                        row = f"{metric:<40}"
                        for value in values:
                            if isinstance(value, (int, float)):
                                if abs(value) >= 1e9:
                                    row += f"${value/1e9:<14.2f}B"
                                elif abs(value) >= 1e6:
                                    row += f"${value/1e6:<14.2f}M"
                                else:
                                    row += f"${value:<14.2f}"
                            else:
                                row += f"{str(value):<15}"
                        print(row)
            
            # Store cash flow statement for potential export
            results['cash_flow'] = cash_flow
        
        # Get financial ratios
        print("\nRetrieving financial ratios...")
        try:
            ratios = integration_service.get_financial_ratios(symbol, period=period, years=years)
            
            print("\nFinancial Ratios:")
            print("-" * 100)
            
            if ratios is None or ratios.empty:
                print("No financial ratio data available.")
            else:
                # Print column headers (dates)
                header = f"{'Ratio':<40}"
                for col in ratios.columns:
                    header += f"{col:<15}"
                print(header)
                print("-" * 100)
        except Exception as e:
            print(f"Error retrieving financial ratios: {str(e)}")
            ratios = pd.DataFrame()  # Use empty DataFrame
            
            # Group ratios by category
            ratio_categories = {
                'Profitability Ratios': [
                    'Gross Margin', 'Operating Margin', 'Net Profit Margin',
                    'Return on Assets', 'Return on Equity', 'Return on Invested Capital'
                ],
                'Liquidity Ratios': [
                    'Current Ratio', 'Quick Ratio', 'Cash Ratio'
                ],
                'Leverage Ratios': [
                    'Debt to Equity', 'Debt to Assets', 'Interest Coverage'
                ],
                'Efficiency Ratios': [
                    'Asset Turnover', 'Inventory Turnover', 'Receivables Turnover'
                ],
                'Valuation Ratios': [
                    'P/E Ratio', 'P/B Ratio', 'EV/EBITDA', 'PEG Ratio'
                ]
            }
            
            for category, category_ratios in ratio_categories.items():
                print(f"\n{category}:")
                
                for ratio in category_ratios:
                    if ratio in ratios.index:
                        values = ratios.loc[ratio]
                        row = f"{ratio:<40}"
                        for value in values:
                            if isinstance(value, (int, float)):
                                if ratio.endswith('Margin') or ratio.startswith('Return on'):
                                    row += f"{value*100:<14.2f}%"
                                else:
                                    row += f"{value:<14.2f}"
                            else:
                                row += f"{str(value):<15}"
                        print(row)
        
        # Store ratios for potential export
        results['financial_ratios'] = ratios
        
        # Compare with industry averages if requested
        if compare_industry:
            print("\nRetrieving industry averages...")
            industry_averages = integration_service.get_industry_averages(
                symbol, 
                ratios=list(ratios.keys()) if ratios else None
            )
            
            print("\nIndustry Comparison:")
            print("-" * 80)
            
            if not industry_averages:
                print("No industry average data available.")
            else:
                print(f"{'Ratio':<30} {'Company':<15} {'Industry Avg':<15} {'Difference':<15}")
                print("-" * 80)
                
                for ratio, industry_value in industry_averages.items():
                    if ratio in ratios and ratios[ratio] and len(ratios[ratio]) > 0:
                        company_value = ratios[ratio][0]  # Most recent value
                        
                        if isinstance(company_value, (int, float)) and isinstance(industry_value, (int, float)):
                            difference = company_value - industry_value
                            
                            if ratio.endswith('Margin') or ratio.startswith('Return on'):
                                company_str = f"{company_value*100:.2f}%"
                                industry_str = f"{industry_value*100:.2f}%"
                                diff_str = f"{difference*100:+.2f}%"
                            else:
                                company_str = f"{company_value:.2f}"
                                industry_str = f"{industry_value:.2f}"
                                diff_str = f"{difference:+.2f}"
                            
                            print(f"{ratio:<30} {company_str:<15} {industry_str:<15} {diff_str:<15}")
            
            # Store industry averages for potential export
            results['industry_averages'] = industry_averages
        
        # Export results if requested
        if not args.no_export and args.export_format and results:
            export_format = args.export_format
            filename = args.output or f"{symbol}_financials_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if export_format == 'json':
                export_path = export_service.export_to_json(results, filename)
            elif export_format == 'csv':
                export_path = export_service.export_to_csv(results, filename)
            elif export_format == 'excel':
                export_path = export_service.export_to_excel(results, filename)
            
            print(f"\nResults exported to: {export_path}")
        
        return 0
        
    except DataRetrievalError as e:
        print(f"Error retrieving financial data: {str(e)}", file=sys.stderr)
        return 1
    except ExportError as e:
        print(f"Error exporting results: {str(e)}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1