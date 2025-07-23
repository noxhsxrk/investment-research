"""Main analysis orchestrator for coordinating all stock analysis services.

This module provides the StockAnalysisOrchestrator class that coordinates all
analysis services, handles batch processing, error handling, and progress tracking.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from stock_analysis.models.data_models import (
    AnalysisResult, SecurityInfo, StockInfo, ETFInfo, FinancialRatios, 
    HealthScore, FairValueResult, SentimentResult
)
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
    total_securities: int
    completed_securities: int
    failed_securities: int
    current_security: Optional[str] = None
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_securities == 0:
            return 0.0
        return (self.completed_securities / self.total_securities) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        processed = self.completed_securities + self.failed_securities
        if processed == 0:
            return 0.0
        return (self.completed_securities / processed) * 100


@dataclass
class AnalysisReport:
    """Report of analysis execution results."""
    total_securities: int
    successful_analyses: int
    failed_analyses: int
    execution_time: float
    success_rate: float
    failed_symbols: List[str]
    error_summary: Dict[str, int]
    results: List[AnalysisResult]


class StockAnalysisOrchestrator:
    """Main orchestrator for coordinating all stock and ETF analysis services.
    
    This class coordinates the execution of analysis across multiple services,
    handles batch processing, error recovery, and progress tracking.
    """
    
    def __init__(self, 
                 max_workers: int = 4,
                 enable_parallel_processing: bool = True,
                 continue_on_error: bool = True,
                 include_technicals: bool = False,
                 include_analyst: bool = False):
        """Initialize the analysis orchestrator.
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
            enable_parallel_processing: Whether to enable parallel processing
            continue_on_error: Whether to continue processing if one fails
            include_technicals: Whether to include technical indicators in analysis
            include_analyst: Whether to include analyst recommendations and price targets
        """
        logger.info("Initializing Stock Analysis Orchestrator")
        
        # Configuration
        self.max_workers = max_workers
        self.enable_parallel_processing = enable_parallel_processing
        self.continue_on_error = continue_on_error
        self.include_technicals = include_technicals
        self.include_analyst = include_analyst
        
        # Initialize services
        self.stock_data_service = StockDataService()
        self.financial_analysis_engine = FinancialAnalysisEngine()
        self.valuation_engine = ValuationEngine()
        self.news_sentiment_analyzer = NewsSentimentAnalyzer()
        self.export_service = ExportService()
        
        # Initialize enhanced services if needed
        if include_technicals or include_analyst:
            from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
            from stock_analysis.services.enhanced_stock_data_service import EnhancedStockDataService
            
            self.integration_service = FinancialDataIntegrationService()
            self.enhanced_stock_data_service = EnhancedStockDataService(self.integration_service)
        
        # Progress tracking
        self.current_progress: Optional[AnalysisProgress] = None
        self.progress_callbacks: List[Callable[[AnalysisProgress], None]] = []
        
        logger.info(f"Orchestrator initialized with max_workers={max_workers}, "
                   f"parallel_processing={enable_parallel_processing}, "
                   f"continue_on_error={continue_on_error}, "
                   f"include_technicals={include_technicals}, "
                   f"include_analyst={include_analyst}")
    
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
    
    def analyze_single_security(self, symbol: str) -> AnalysisResult:
        """Analyze a single security (stock or ETF).
        
        Args:
            symbol: Security ticker symbol to analyze
            
        Returns:
            AnalysisResult: Complete analysis result for the security
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
            CalculationError: If analysis calculations fail
        """
        logger.info(f"Starting analysis for {symbol}")
        start_time = time.time()
        
        try:
            # Step 1: Get security information (basic or enhanced)
            logger.debug(f"Retrieving security info for {symbol}")
            
            if hasattr(self, 'enhanced_stock_data_service') and (self.include_technicals or self.include_analyst):
                # Use enhanced stock data service
                security_info = self.enhanced_stock_data_service.get_enhanced_security_info(
                    symbol,
                    include_technicals=self.include_technicals,
                    include_analyst_data=self.include_analyst
                )
            else:
                # Use standard stock data service
                security_info = self.stock_data_service.get_security_info(symbol)
            
            # Different analysis paths for stocks vs ETFs
            if isinstance(security_info, ETFInfo):
                # ETF analysis path
                logger.debug(f"Processing ETF: {symbol}")
                
                # For ETFs, we focus on different metrics
                financial_ratios = FinancialRatios(
                    liquidity_ratios=None,  # ETFs don't have traditional ratios
                    profitability_ratios=None,
                    leverage_ratios=None,
                    efficiency_ratios=None
                )
                
                # Health score for ETFs based on different criteria
                health_score = HealthScore(
                    overall_score=self._calculate_etf_health_score(security_info),
                    financial_strength=100.0,  # ETFs are generally financially stable
                    profitability_health=100.0,
                    liquidity_health=100.0,
                    risk_assessment=self._assess_etf_risk(security_info)
                )
                
                # Fair value analysis for ETFs
                fair_value = self._calculate_etf_fair_value(security_info)
                
            else:
                # Stock analysis path
                logger.debug(f"Processing stock: {symbol}")
                
                # Get financial statements
                income_statement = self.stock_data_service.get_financial_statements(
                    symbol, "income", "annual"
                )
                balance_sheet = self.stock_data_service.get_financial_statements(
                    symbol, "balance", "annual"
                )
                cash_flow = self.stock_data_service.get_financial_statements(
                    symbol, "cash", "annual"
                )
                
                # Calculate financial ratios
                logger.debug(f"Calculating financial ratios for {symbol}")
                financial_ratios = self.financial_analysis_engine.calculate_financial_ratios(
                    symbol, security_info, income_statement, balance_sheet, cash_flow
                )
                
                # Assess company health
                logger.debug(f"Assessing company health for {symbol}")
                health_score = self.financial_analysis_engine.assess_company_health(
                    symbol, financial_ratios
                )
                
                # Calculate fair value
                logger.debug(f"Calculating fair value for {symbol}")
                peer_data = []  # Could be enhanced to get actual peer data
                fair_value = self.valuation_engine.calculate_fair_value(
                    symbol, security_info, income_statement, balance_sheet, cash_flow, peer_data
                )
            
            # Common analysis steps for both stocks and ETFs
            
            # Analyze news sentiment
            logger.debug(f"Analyzing news sentiment for {symbol}")
            news_articles = self.news_sentiment_analyzer.get_news_articles(symbol, days=7)
            sentiment = self.news_sentiment_analyzer.analyze_sentiment(news_articles)
            
            # Generate recommendations
            logger.debug(f"Generating recommendations for {symbol}")
            recommendations = self._generate_recommendations(
                symbol, security_info, financial_ratios, health_score, fair_value, sentiment
            )
            
            # Create final analysis result
            analysis_result = AnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                stock_info=security_info,
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
    
    def _calculate_etf_health_score(self, etf_info: ETFInfo) -> float:
        """Calculate health score for an ETF.
        
        Args:
            etf_info: ETF information
            
        Returns:
            float: Health score (0-100)
        """
        score = 100.0  # Start with perfect score
        
        # Penalize for high expense ratio
        if etf_info.expense_ratio is not None:
            if etf_info.expense_ratio > 0.01:  # More than 1%
                score -= 20
            elif etf_info.expense_ratio > 0.005:  # More than 0.5%
                score -= 10
        
        # Penalize for low assets under management
        if etf_info.assets_under_management is not None:
            if etf_info.assets_under_management < 100_000_000:  # Less than $100M
                score -= 20
            elif etf_info.assets_under_management < 500_000_000:  # Less than $500M
                score -= 10
        
        # Consider diversification if we have holdings data
        if etf_info.holdings:
            num_holdings = len(etf_info.holdings)
            if num_holdings < 20:
                score -= 20
            elif num_holdings < 50:
                score -= 10
        
        return max(0.0, min(100.0, score))
    
    def _assess_etf_risk(self, etf_info: ETFInfo) -> str:
        """Assess risk level for an ETF.
        
        Args:
            etf_info: ETF information
            
        Returns:
            str: Risk assessment ("Low", "Medium", or "High")
        """
        if etf_info.beta is not None:
            if etf_info.beta < 0.8:
                return "Low"
            elif etf_info.beta > 1.2:
                return "High"
        
        # Default to medium risk if we can't determine
        return "Medium"
    
    def _calculate_etf_fair_value(self, etf_info: ETFInfo) -> FairValueResult:
        """Calculate fair value metrics for an ETF.
        
        Args:
            etf_info: ETF information
            
        Returns:
            FairValueResult: Fair value analysis result
        """
        # For ETFs, we typically compare to NAV
        if etf_info.nav is not None and etf_info.current_price > 0:
            premium_discount = (etf_info.current_price - etf_info.nav) / etf_info.nav
            
            # Determine recommendation based on premium/discount
            if premium_discount < -0.02:  # Trading at >2% discount
                recommendation = "BUY"
                confidence = 0.8
            elif premium_discount > 0.02:  # Trading at >2% premium
                recommendation = "SELL"
                confidence = 0.8
            else:
                recommendation = "HOLD"
                confidence = 0.9
            
            return FairValueResult(
                current_price=etf_info.current_price,
                dcf_value=etf_info.nav,  # Use NAV as the fair value
                peer_comparison_value=None,
                average_fair_value=etf_info.nav,
                recommendation=recommendation,
                confidence_level=confidence
            )
        
        # If we don't have NAV, return a neutral result
        return FairValueResult(
            current_price=etf_info.current_price,
            dcf_value=None,
            peer_comparison_value=None,
            average_fair_value=etf_info.current_price,
            recommendation="HOLD",
            confidence_level=0.5
        )
    
    def _generate_recommendations(self,
                                symbol: str,
                                security_info: SecurityInfo,
                                financial_ratios: FinancialRatios,
                                health_score: HealthScore,
                                fair_value: FairValueResult,
                                sentiment: SentimentResult) -> List[str]:
        """Generate investment recommendations based on analysis results.
        
        Args:
            symbol: Security ticker symbol
            security_info: Basic security information
            financial_ratios: Financial ratios analysis
            health_score: Security health assessment
            fair_value: Fair value analysis
            sentiment: News sentiment analysis
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if isinstance(security_info, ETFInfo):
            # ETF-specific recommendations
            recommendations.extend(self._generate_etf_recommendations(security_info))
        else:
            # Stock-specific recommendations
            recommendations.extend(self._generate_stock_recommendations(
                security_info, financial_ratios, health_score
            ))
        
        # Common recommendations for both stocks and ETFs
        
        # Valuation-based recommendations
        if fair_value.recommendation == "BUY":
            if fair_value.confidence_level > 0.7:
                recommendations.append(f"Strong Buy: Security appears undervalued with high confidence ({fair_value.confidence_level:.1%})")
            else:
                recommendations.append(f"Buy: Security appears undervalued (confidence: {fair_value.confidence_level:.1%})")
        elif fair_value.recommendation == "SELL":
            if fair_value.confidence_level > 0.7:
                recommendations.append(f"Strong Sell: Security appears overvalued with high confidence ({fair_value.confidence_level:.1%})")
            else:
                recommendations.append(f"Sell: Security appears overvalued (confidence: {fair_value.confidence_level:.1%})")
        else:
            recommendations.append("Hold: Security appears fairly valued")
        
        # Health-based recommendations
        if health_score.overall_score >= 80:
            recommendations.append("Excellent health - suitable for conservative investors")
        elif health_score.overall_score >= 60:
            recommendations.append("Good health - suitable for moderate risk investors")
        elif health_score.overall_score >= 40:
            recommendations.append("Fair health - requires careful monitoring")
        else:
            recommendations.append("Poor health - high risk investment")
        
        # Sentiment-based recommendations
        if sentiment.overall_sentiment > 0.3:
            recommendations.append("Positive news sentiment - market perception is favorable")
        elif sentiment.overall_sentiment < -0.3:
            recommendations.append("Negative news sentiment - monitor for potential issues")
        
        # Ensure we have at least one recommendation
        if not recommendations:
            recommendations.append("Neutral outlook - conduct additional research before investing")
        
        return recommendations
    
    def _generate_etf_recommendations(self, etf_info: ETFInfo) -> List[str]:
        """Generate ETF-specific recommendations.
        
        Args:
            etf_info: ETF information
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Expense ratio recommendations
        if etf_info.expense_ratio is not None:
            if etf_info.expense_ratio < 0.0020:  # Less than 0.20%
                recommendations.append("Very low expense ratio - excellent cost efficiency")
            elif etf_info.expense_ratio < 0.0050:  # Less than 0.50%
                recommendations.append("Reasonable expense ratio - good cost efficiency")
            else:
                recommendations.append("High expense ratio - consider lower-cost alternatives")
        
        # Asset size recommendations
        if etf_info.assets_under_management is not None:
            if etf_info.assets_under_management > 1_000_000_000:  # >$1B
                recommendations.append("Large fund size - good liquidity and stability")
            elif etf_info.assets_under_management < 100_000_000:  # <$100M
                recommendations.append("Small fund size - monitor for closure risk")
        
        # Diversification recommendations
        if etf_info.holdings:
            num_holdings = len(etf_info.holdings)
            if num_holdings > 100:
                recommendations.append("Well diversified - broad market exposure")
            elif num_holdings > 50:
                recommendations.append("Moderately diversified - sector/theme focused")
            else:
                recommendations.append("Concentrated portfolio - higher specific risk")
        
        # Asset allocation recommendations
        if etf_info.asset_allocation:
            for asset_type, allocation in etf_info.asset_allocation.items():
                if allocation > 0.8:  # >80%
                    recommendations.append(f"High concentration in {asset_type} - consider diversification needs")
        
        return recommendations
    
    def _generate_stock_recommendations(self,
                                      stock_info: StockInfo,
                                      financial_ratios: FinancialRatios,
                                      health_score: HealthScore) -> List[str]:
        """Generate stock-specific recommendations.
        
        Args:
            stock_info: Stock information
            financial_ratios: Financial ratios
            health_score: Health assessment
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Risk-based recommendations
        if health_score.risk_assessment == "Low":
            recommendations.append("Low risk profile - suitable for income-focused portfolios")
        elif health_score.risk_assessment == "Medium":
            recommendations.append("Medium risk profile - suitable for balanced portfolios")
        else:
            recommendations.append("High risk profile - suitable only for aggressive growth portfolios")
        
        # Liquidity-based recommendations
        if financial_ratios.liquidity_ratios and financial_ratios.liquidity_ratios.current_ratio is not None:
            if financial_ratios.liquidity_ratios.current_ratio > 2.0:
                recommendations.append("Strong liquidity position - low short-term financial risk")
            elif financial_ratios.liquidity_ratios.current_ratio < 1.0:
                recommendations.append("Weak liquidity position - monitor short-term obligations closely")
        
        # Profitability-based recommendations
        if (financial_ratios.profitability_ratios and 
            financial_ratios.profitability_ratios.return_on_equity is not None):
            if financial_ratios.profitability_ratios.return_on_equity > 0.15:
                recommendations.append("Strong profitability - efficient use of shareholder equity")
            elif financial_ratios.profitability_ratios.return_on_equity < 0.05:
                recommendations.append("Weak profitability - consider management effectiveness")
        
        # Sector/Industry specific recommendations
        if stock_info.sector:
            if stock_info.sector in ["Technology", "Healthcare"]:
                recommendations.append(f"Growth sector ({stock_info.sector}) - consider growth potential")
            elif stock_info.sector in ["Utilities", "Consumer Staples"]:
                recommendations.append(f"Defensive sector ({stock_info.sector}) - suitable for stability")
        
        return recommendations
    
    def analyze_multiple_securities(self, symbols: List[str]) -> AnalysisReport:
        """Analyze multiple securities with batch processing and error handling.
        
        Args:
            symbols: List of security ticker symbols to analyze
            
        Returns:
            AnalysisReport: Summary report of the batch analysis
        """
        logger.info(f"Starting batch analysis for {len(symbols)} securities")
        start_time = time.time()
        
        # Initialize progress tracking
        progress = AnalysisProgress(
            total_securities=len(symbols),
            completed_securities=0,
            failed_securities=0,
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
            successful_results, failed_symbols, error_summary = self._process_securities_parallel(
                symbols, progress
            )
        else:
            # Sequential processing
            logger.info("Using sequential processing")
            successful_results, failed_symbols, error_summary = self._process_securities_sequential(
                symbols, progress
            )
        
        # Calculate final metrics
        execution_time = time.time() - start_time
        success_rate = (len(successful_results) / len(symbols)) * 100 if symbols else 0
        
        # Create analysis report
        report = AnalysisReport(
            total_securities=len(symbols),
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
    
    def _process_securities_sequential(self, 
                                     symbols: List[str], 
                                     progress: AnalysisProgress) -> Tuple[List[AnalysisResult], List[str], Dict[str, int]]:
        """Process securities sequentially.
        
        Args:
            symbols: List of security symbols to process
            progress: Progress tracking object
            
        Returns:
            Tuple of (successful_results, failed_symbols, error_summary)
        """
        successful_results: List[AnalysisResult] = []
        failed_symbols: List[str] = []
        error_summary: Dict[str, int] = {}
        
        for i, symbol in enumerate(symbols):
            progress.current_security = symbol
            self._update_progress(progress)
            
            try:
                result = self.analyze_single_security(symbol)
                successful_results.append(result)
                progress.completed_securities += 1
                
            except Exception as e:
                error_type = type(e).__name__
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
                failed_symbols.append(symbol)
                progress.failed_securities += 1
                
                logger.error(f"Failed to analyze {symbol}: {str(e)}")
                
                if not self.continue_on_error:
                    logger.error("Stopping batch processing due to error")
                    break
            
            # Update progress
            self._update_progress(progress)
            
            # Estimate completion time
            if progress.completed_securities + progress.failed_securities > 0:
                elapsed = (datetime.now() - progress.start_time).total_seconds()
                avg_time_per_security = elapsed / (progress.completed_securities + progress.failed_securities)
                remaining_securities = progress.total_securities - progress.completed_securities - progress.failed_securities
                estimated_remaining = remaining_securities * avg_time_per_security
                progress.estimated_completion = datetime.now().replace(
                    microsecond=0
                ) + timedelta(seconds=estimated_remaining)
        
        return successful_results, failed_symbols, error_summary
    
    def _process_securities_parallel(self, 
                                   symbols: List[str], 
                                   progress: AnalysisProgress) -> Tuple[List[AnalysisResult], List[str], Dict[str, int]]:
        """Process securities in parallel using ThreadPoolExecutor.
        
        Args:
            symbols: List of security symbols to process
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
                executor.submit(self.analyze_single_security, symbol): symbol
                for symbol in symbols
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                progress.current_security = symbol
                
                try:
                    result = future.result()
                    successful_results.append(result)
                    progress.completed_securities += 1
                    
                except Exception as e:
                    error_type = type(e).__name__
                    error_summary[error_type] = error_summary.get(error_type, 0) + 1
                    failed_symbols.append(symbol)
                    progress.failed_securities += 1
                    
                    logger.error(f"Failed to analyze {symbol}: {str(e)}")
                
                # Update progress
                self._update_progress(progress)
        
        return successful_results, failed_symbols, error_summary
    
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
                return self.export_service.export_to_json(results, filename)
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
            f"Analysis Summary",
            f"=" * 50,
            f"Total securities analyzed: {report.total_securities}",
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
            
            # Count stocks vs ETFs
            stocks = [r for r in report.results if isinstance(r.stock_info, StockInfo)]
            etfs = [r for r in report.results if isinstance(r.stock_info, ETFInfo)]
            
            summary_lines.extend([
                f"  Stocks analyzed: {len(stocks)}",
                f"  ETFs analyzed: {len(etfs)}",
                ""
            ])
            
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