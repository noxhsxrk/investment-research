"""Main analysis orchestrator for coordinating all stock analysis services.

This module provides the StockAnalysisOrchestrator class that coordinates all
analysis services, handles batch processing, error handling, and progress tracking.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from stock_analysis.models.data_models import AnalysisResult, StockInfo, FinancialRatios, HealthScore, FairValueResult, SentimentResult
from stock_analysis.services.stock_data_service import StockDataService
from stock_analysis.analyzers.financial_analysis_engine import FinancialAnalysisEngine
from stock_analysis.analyzers.valuation_engine import ValuationEngine
from stock_analysis.analyzers.news_sentiment_analyzer import NewsSentimentAnalyzer
from stock_analysis.exporters.export_service import ExportService
from stock_analysis.utils.exceptions import DataRetrievalError, CalculationError, ExportError
from stock_analysis.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisProgress:
    """Progress tracking for analysis operations."""
    total_stocks: int
    completed_stocks: int
    failed_stocks: int
    current_stock: Optional[str] = None
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_stocks == 0:
            return 0.0
        return (self.completed_stocks / self.total_stocks) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        processed = self.completed_stocks + self.failed_stocks
        if processed == 0:
            return 0.0
        return (self.completed_stocks / processed) * 100


@dataclass
class AnalysisReport:
    """Report of analysis execution results."""
    total_stocks: int
    successful_analyses: int
    failed_analyses: int
    execution_time: float
    success_rate: float
    failed_symbols: List[str]
    error_summary: Dict[str, int]
    results: List[AnalysisResult]


class StockAnalysisOrchestrator:
    """Main orchestrator for coordinating all stock analysis services.
    
    This class coordinates the execution of stock analysis across multiple services,
    handles batch processing, error recovery, and progress tracking.
    """
    
    def __init__(self, 
                 max_workers: int = 4,
                 enable_parallel_processing: bool = True,
                 continue_on_error: bool = True):
        """Initialize the stock analysis orchestrator.
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
            enable_parallel_processing: Whether to enable parallel processing of stocks
            continue_on_error: Whether to continue processing other stocks if one fails
        """
        logger.info("Initializing Stock Analysis Orchestrator")
        
        # Configuration
        self.max_workers = max_workers
        self.enable_parallel_processing = enable_parallel_processing
        self.continue_on_error = continue_on_error
        
        # Initialize services
        self.stock_data_service = StockDataService()
        self.financial_analysis_engine = FinancialAnalysisEngine()
        self.valuation_engine = ValuationEngine()
        self.news_sentiment_analyzer = NewsSentimentAnalyzer()
        self.export_service = ExportService()
        
        # Progress tracking
        self.current_progress: Optional[AnalysisProgress] = None
        self.progress_callbacks: List[Callable[[AnalysisProgress], None]] = []
        
        logger.info(f"Orchestrator initialized with max_workers={max_workers}, "
                   f"parallel_processing={enable_parallel_processing}, "
                   f"continue_on_error={continue_on_error}")
    
    def add_progress_callback(self, callback: Callable[[AnalysisProgress], None]) -> None:
        """Add a callback function to receive progress updates.
        
        Args:
            callback: Function that will be called with AnalysisProgress updates
        """
        self.progress_callbacks.append(callback)
    
    def _update_progress(self, progress: AnalysisProgress) -> None:
        """Update progress and notify callbacks.
        
        Args:
            progress: Current progress state
        """
        self.current_progress = progress
        
        # Notify all callbacks
        for callback in self.progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def analyze_single_stock(self, symbol: str) -> AnalysisResult:
        """Analyze a single stock symbol.
        
        Args:
            symbol: Stock ticker symbol to analyze
            
        Returns:
            AnalysisResult: Complete analysis result for the stock
            
        Raises:
            DataRetrievalError: If stock data cannot be retrieved
            CalculationError: If analysis calculations fail
        """
        logger.info(f"Starting analysis for {symbol}")
        start_time = time.time()
        
        try:
            # Step 1: Get basic stock information
            logger.debug(f"Retrieving stock info for {symbol}")
            stock_info = self.stock_data_service.get_stock_info(symbol)
            
            # Step 2: Get financial statements
            logger.debug(f"Retrieving financial statements for {symbol}")
            income_statement = self.stock_data_service.get_financial_statements(
                symbol, "income", "annual"
            )
            balance_sheet = self.stock_data_service.get_financial_statements(
                symbol, "balance", "annual"
            )
            cash_flow = self.stock_data_service.get_financial_statements(
                symbol, "cash", "annual"
            )
            
            # Step 3: Calculate financial ratios
            logger.debug(f"Calculating financial ratios for {symbol}")
            financial_ratios = self.financial_analysis_engine.calculate_financial_ratios(
                symbol, stock_info, income_statement, balance_sheet, cash_flow
            )
            
            # Step 4: Assess company health
            logger.debug(f"Assessing company health for {symbol}")
            health_score = self.financial_analysis_engine.assess_company_health(
                symbol, financial_ratios
            )
            
            # Step 5: Calculate fair value
            logger.debug(f"Calculating fair value for {symbol}")
            # Get peer data (simplified - in real implementation might get actual peers)
            peer_data = []  # Could be enhanced to get actual peer data
            
            fair_value = self.valuation_engine.calculate_fair_value(
                symbol, stock_info, income_statement, balance_sheet, cash_flow, peer_data
            )
            
            # Step 6: Analyze news sentiment
            logger.debug(f"Analyzing news sentiment for {symbol}")
            news_articles = self.news_sentiment_analyzer.get_news_articles(symbol, days=7)
            sentiment = self.news_sentiment_analyzer.analyze_sentiment(news_articles)
            
            # Step 7: Generate recommendations
            logger.debug(f"Generating recommendations for {symbol}")
            recommendations = self._generate_recommendations(
                symbol, stock_info, financial_ratios, health_score, fair_value, sentiment
            )
            
            # Create final analysis result
            analysis_result = AnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                stock_info=stock_info,
                financial_ratios=financial_ratios,
                health_score=health_score,
                fair_value=fair_value,
                sentiment=sentiment,
                recommendations=recommendations
            )
            
            # Validate the result
            analysis_result.validate()
            
            execution_time = time.time() - start_time
            logger.info(f"Successfully completed analysis for {symbol} in {execution_time:.2f}s")
            
            return analysis_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Analysis failed for {symbol} after {execution_time:.2f}s: {str(e)}")
            raise
    
    def analyze_multiple_stocks(self, symbols: List[str]) -> AnalysisReport:
        """Analyze multiple stocks with batch processing and error handling.
        
        Args:
            symbols: List of stock ticker symbols to analyze
            
        Returns:
            AnalysisReport: Summary report of the batch analysis
        """
        logger.info(f"Starting batch analysis for {len(symbols)} stocks")
        start_time = time.time()
        
        # Initialize progress tracking
        progress = AnalysisProgress(
            total_stocks=len(symbols),
            completed_stocks=0,
            failed_stocks=0,
            start_time=datetime.now()
        )
        self._update_progress(progress)
        
        # Track results and errors
        successful_results: List[AnalysisResult] = []
        failed_symbols: List[str] = []
        error_summary: Dict[str, int] = {}
        
        if self.enable_parallel_processing and len(symbols) > 1:
            # Parallel processing
            logger.info(f"Using parallel processing with {self.max_workers} workers")
            successful_results, failed_symbols, error_summary = self._process_stocks_parallel(
                symbols, progress
            )
        else:
            # Sequential processing
            logger.info("Using sequential processing")
            successful_results, failed_symbols, error_summary = self._process_stocks_sequential(
                symbols, progress
            )
        
        # Calculate final metrics
        execution_time = time.time() - start_time
        success_rate = (len(successful_results) / len(symbols)) * 100 if symbols else 0
        
        # Create analysis report
        report = AnalysisReport(
            total_stocks=len(symbols),
            successful_analyses=len(successful_results),
            failed_analyses=len(failed_symbols),
            execution_time=execution_time,
            success_rate=success_rate,
            failed_symbols=failed_symbols,
            error_summary=error_summary,
            results=successful_results
        )
        
        logger.info(f"Batch analysis completed: {len(successful_results)}/{len(symbols)} successful "
                   f"({success_rate:.1f}%) in {execution_time:.2f}s")
        
        return report
    
    def _process_stocks_sequential(self, 
                                 symbols: List[str], 
                                 progress: AnalysisProgress) -> Tuple[List[AnalysisResult], List[str], Dict[str, int]]:
        """Process stocks sequentially.
        
        Args:
            symbols: List of stock symbols to process
            progress: Progress tracking object
            
        Returns:
            Tuple of (successful_results, failed_symbols, error_summary)
        """
        successful_results: List[AnalysisResult] = []
        failed_symbols: List[str] = []
        error_summary: Dict[str, int] = {}
        
        for i, symbol in enumerate(symbols):
            progress.current_stock = symbol
            self._update_progress(progress)
            
            try:
                result = self.analyze_single_stock(symbol)
                successful_results.append(result)
                progress.completed_stocks += 1
                
            except Exception as e:
                error_type = type(e).__name__
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
                failed_symbols.append(symbol)
                progress.failed_stocks += 1
                
                logger.error(f"Failed to analyze {symbol}: {str(e)}")
                
                if not self.continue_on_error:
                    logger.error("Stopping batch processing due to error")
                    break
            
            # Update progress
            self._update_progress(progress)
            
            # Estimate completion time
            if progress.completed_stocks + progress.failed_stocks > 0:
                elapsed = (datetime.now() - progress.start_time).total_seconds()
                avg_time_per_stock = elapsed / (progress.completed_stocks + progress.failed_stocks)
                remaining_stocks = progress.total_stocks - progress.completed_stocks - progress.failed_stocks
                estimated_remaining = remaining_stocks * avg_time_per_stock
                progress.estimated_completion = datetime.now().replace(
                    microsecond=0
                ) + timedelta(seconds=estimated_remaining)
        
        return successful_results, failed_symbols, error_summary
    
    def _process_stocks_parallel(self, 
                               symbols: List[str], 
                               progress: AnalysisProgress) -> Tuple[List[AnalysisResult], List[str], Dict[str, int]]:
        """Process stocks in parallel using ThreadPoolExecutor.
        
        Args:
            symbols: List of stock symbols to process
            progress: Progress tracking object
            
        Returns:
            Tuple of (successful_results, failed_symbols, error_summary)
        """
        successful_results: List[AnalysisResult] = []
        failed_symbols: List[str] = []
        error_summary: Dict[str, int] = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.analyze_single_stock, symbol): symbol
                for symbol in symbols
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                progress.current_stock = symbol
                
                try:
                    result = future.result()
                    successful_results.append(result)
                    progress.completed_stocks += 1
                    
                except Exception as e:
                    error_type = type(e).__name__
                    error_summary[error_type] = error_summary.get(error_type, 0) + 1
                    failed_symbols.append(symbol)
                    progress.failed_stocks += 1
                    
                    logger.error(f"Failed to analyze {symbol}: {str(e)}")
                
                # Update progress
                self._update_progress(progress)
        
        return successful_results, failed_symbols, error_summary
    
    def _generate_recommendations(self,
                                symbol: str,
                                stock_info: StockInfo,
                                financial_ratios: FinancialRatios,
                                health_score: HealthScore,
                                fair_value: FairValueResult,
                                sentiment: SentimentResult) -> List[str]:
        """Generate investment recommendations based on analysis results.
        
        Args:
            symbol: Stock ticker symbol
            stock_info: Basic stock information
            financial_ratios: Financial ratios analysis
            health_score: Company health assessment
            fair_value: Fair value analysis
            sentiment: News sentiment analysis
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Valuation-based recommendations
        if fair_value.recommendation == "BUY":
            if fair_value.confidence_level > 0.7:
                recommendations.append(f"Strong Buy: Stock appears undervalued with high confidence ({fair_value.confidence_level:.1%})")
            else:
                recommendations.append(f"Buy: Stock appears undervalued (confidence: {fair_value.confidence_level:.1%})")
        elif fair_value.recommendation == "SELL":
            if fair_value.confidence_level > 0.7:
                recommendations.append(f"Strong Sell: Stock appears overvalued with high confidence ({fair_value.confidence_level:.1%})")
            else:
                recommendations.append(f"Sell: Stock appears overvalued (confidence: {fair_value.confidence_level:.1%})")
        else:
            recommendations.append("Hold: Stock appears fairly valued")
        
        # Health-based recommendations
        if health_score.overall_score >= 80:
            recommendations.append("Excellent financial health - suitable for conservative investors")
        elif health_score.overall_score >= 60:
            recommendations.append("Good financial health - suitable for moderate risk investors")
        elif health_score.overall_score >= 40:
            recommendations.append("Fair financial health - requires careful monitoring")
        else:
            recommendations.append("Poor financial health - high risk investment")
        
        # Risk-based recommendations
        if health_score.risk_assessment == "Low":
            recommendations.append("Low risk profile - suitable for income-focused portfolios")
        elif health_score.risk_assessment == "Medium":
            recommendations.append("Medium risk profile - suitable for balanced portfolios")
        else:
            recommendations.append("High risk profile - suitable only for aggressive growth portfolios")
        
        # Liquidity-based recommendations
        if financial_ratios.liquidity_ratios.current_ratio is not None:
            if financial_ratios.liquidity_ratios.current_ratio > 2.0:
                recommendations.append("Strong liquidity position - low short-term financial risk")
            elif financial_ratios.liquidity_ratios.current_ratio < 1.0:
                recommendations.append("Weak liquidity position - monitor short-term obligations closely")
        
        # Profitability-based recommendations
        if financial_ratios.profitability_ratios.return_on_equity is not None:
            if financial_ratios.profitability_ratios.return_on_equity > 0.15:
                recommendations.append("Strong profitability - efficient use of shareholder equity")
            elif financial_ratios.profitability_ratios.return_on_equity < 0.05:
                recommendations.append("Weak profitability - consider management effectiveness")
        
        # Sentiment-based recommendations
        if sentiment.overall_sentiment > 0.3:
            recommendations.append("Positive news sentiment - market perception is favorable")
        elif sentiment.overall_sentiment < -0.3:
            recommendations.append("Negative news sentiment - monitor for potential issues")
        
        # Sector/Industry specific recommendations
        if stock_info.sector:
            if stock_info.sector in ["Technology", "Healthcare"]:
                recommendations.append(f"Growth sector ({stock_info.sector}) - consider growth potential")
            elif stock_info.sector in ["Utilities", "Consumer Staples"]:
                recommendations.append(f"Defensive sector ({stock_info.sector}) - suitable for stability")
        
        # Ensure we have at least one recommendation
        if not recommendations:
            recommendations.append("Neutral outlook - conduct additional research before investing")
        
        return recommendations
    
    def export_results(self, 
                      results: List[AnalysisResult], 
                      export_format: str = "excel",
                      filename: Optional[str] = None) -> str:
        """Export analysis results to specified format.
        
        Args:
            results: List of analysis results to export
            export_format: Export format ("csv", "excel", "json")
            filename: Optional custom filename
            
        Returns:
            str: Path to exported file
            
        Raises:
            ExportError: If export operation fails
        """
        logger.info(f"Exporting {len(results)} results to {export_format} format")
        
        try:
            if export_format.lower() == "csv":
                return self.export_service.export_to_csv(results, filename)
            elif export_format.lower() == "excel":
                return self.export_service.export_to_excel(results, filename)
            elif export_format.lower() == "json":
                return self.export_service.export_to_powerbi_json(results, filename)
            else:
                raise ExportError(f"Unsupported export format: {export_format}")
                
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            raise
    
    def get_analysis_summary(self, report: AnalysisReport) -> str:
        """Generate a human-readable summary of analysis results.
        
        Args:
            report: Analysis report to summarize
            
        Returns:
            str: Human-readable summary
        """
        summary_lines = [
            f"Stock Analysis Summary",
            f"=" * 50,
            f"Total stocks analyzed: {report.total_stocks}",
            f"Successful analyses: {report.successful_analyses}",
            f"Failed analyses: {report.failed_analyses}",
            f"Success rate: {report.success_rate:.1f}%",
            f"Execution time: {report.execution_time:.2f} seconds",
            ""
        ]
        
        if report.failed_symbols:
            summary_lines.extend([
                f"Failed symbols: {', '.join(report.failed_symbols)}",
                ""
            ])
        
        if report.error_summary:
            summary_lines.append("Error summary:")
            for error_type, count in report.error_summary.items():
                summary_lines.append(f"  {error_type}: {count}")
            summary_lines.append("")
        
        if report.results:
            summary_lines.append("Analysis highlights:")
            
            # Calculate some aggregate statistics
            buy_recommendations = sum(1 for r in report.results if r.fair_value.recommendation == "BUY")
            sell_recommendations = sum(1 for r in report.results if r.fair_value.recommendation == "SELL")
            hold_recommendations = sum(1 for r in report.results if r.fair_value.recommendation == "HOLD")
            
            avg_health_score = sum(r.health_score.overall_score for r in report.results) / len(report.results)
            avg_sentiment = sum(r.sentiment.overall_sentiment for r in report.results) / len(report.results)
            
            summary_lines.extend([
                f"  Buy recommendations: {buy_recommendations}",
                f"  Hold recommendations: {hold_recommendations}",
                f"  Sell recommendations: {sell_recommendations}",
                f"  Average health score: {avg_health_score:.1f}/100",
                f"  Average sentiment: {avg_sentiment:.2f} (-1 to 1)",
            ])
        
        return "\n".join(summary_lines)