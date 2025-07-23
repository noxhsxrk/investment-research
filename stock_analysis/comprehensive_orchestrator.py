"""Comprehensive analysis orchestrator for coordinating multiple data retrieval operations.

This module provides the ComprehensiveAnalysisOrchestrator class that coordinates
the execution of multiple data retrieval operations including stock analysis,
financial statements, and financial news.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from stock_analysis.orchestrator import StockAnalysisOrchestrator, AnalysisProgress, AnalysisReport
from stock_analysis.services.stock_data_service import StockDataService
from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
from stock_analysis.services.news_service import NewsService
from stock_analysis.exporters.export_service import ExportService
from stock_analysis.models.data_models import AnalysisResult, SentimentResult
from stock_analysis.models.enhanced_data_models import NewsItem
from stock_analysis.models.comprehensive_models import ComprehensiveAnalysisResult, ComprehensiveAnalysisReport
from stock_analysis.utils.exceptions import DataRetrievalError, CalculationError, ExportError
from stock_analysis.utils.logging import get_logger

logger = get_logger(__name__)


class ComprehensiveAnalysisOrchestrator:
    """Orchestrates comprehensive stock analysis operations.
    
    This class coordinates the execution of multiple data retrieval operations
    including stock analysis, financial statements, and financial news.
    """
    
    def __init__(self, 
                 enable_parallel_processing: bool = True, 
                 max_workers: int = 4,
                 continue_on_error: bool = True):
        """Initialize the comprehensive analysis orchestrator.
        
        Args:
            enable_parallel_processing: Whether to enable parallel processing
            max_workers: Maximum number of worker threads for parallel processing
            continue_on_error: Whether to continue processing if one operation fails
        """
        logger.info("Initializing Comprehensive Analysis Orchestrator")
        
        # Configuration
        self.enable_parallel_processing = enable_parallel_processing
        self.max_workers = max_workers
        self.continue_on_error = continue_on_error
        
        # Initialize services
        self.stock_analysis_orchestrator = StockAnalysisOrchestrator(
            enable_parallel_processing=enable_parallel_processing,
            max_workers=max_workers,
            continue_on_error=continue_on_error
        )
        self.stock_data_service = StockDataService()
        self.integration_service = FinancialDataIntegrationService()
        self.news_service = NewsService(integration_service=self.integration_service)
        self.export_service = ExportService()
        
        logger.info(f"Orchestrator initialized with max_workers={max_workers}, "
                   f"parallel_processing={enable_parallel_processing}, "
                   f"continue_on_error={continue_on_error}")
    
    def analyze_comprehensive(self, 
                             symbols: List[str], 
                             options: Dict[str, Any] = None) -> ComprehensiveAnalysisReport:
        """Perform comprehensive analysis for the given symbols.
        
        Args:
            symbols: List of stock symbols to analyze
            options: Dictionary of options to customize the analysis
                - skip_analysis: Whether to skip stock analysis
                - skip_financials: Whether to skip financial statements
                - skip_news: Whether to skip financial news
                - statement: Financial statement type ('income', 'balance', 'cash', 'all')
                - period: Reporting period ('annual', 'quarterly')
                - years: Number of years of historical data
                - news_limit: Number of news items to retrieve
                - news_days: Number of days to look back for news
                - sentiment: Whether to include sentiment analysis for news
            
        Returns:
            ComprehensiveAnalysisReport: Report containing the analysis results
        """
        logger.info(f"Starting comprehensive analysis for {len(symbols)} securities: {', '.join(symbols)}")
        start_time = time.time()
        
        # Default options
        if options is None:
            options = {}
        
        default_options = {
            'skip_analysis': False,
            'skip_financials': False,
            'skip_news': False,
            'statement': 'all',
            'period': 'annual',
            'years': 5,
            'news_limit': 10,
            'news_days': 7,
            'sentiment': True
        }
        
        # Merge with default options
        for key, value in default_options.items():
            if key not in options:
                options[key] = value
        
        # Use parallel processing if enabled and multiple symbols
        if self.enable_parallel_processing and len(symbols) > 1:
            # Import the enhanced parallel processing implementation
            from stock_analysis.comprehensive_orchestrator_parallel import _analyze_multiple_securities_parallel
            
            # Use the enhanced parallel processing implementation
            try:
                # Bind the method to this instance
                analyze_multiple = _analyze_multiple_securities_parallel.__get__(self)
                results = analyze_multiple(symbols, options)
                failed_symbols = [symbol for symbol in symbols if symbol not in [r.symbol for r in results]]
            except Exception as e:
                logger.error(f"Parallel analysis failed: {str(e)}")
                results = []
                failed_symbols = symbols
        else:
            # Process each symbol sequentially
            results = []
            failed_symbols = []
            
            for symbol in symbols:
                try:
                    logger.info(f"Processing comprehensive analysis for {symbol}")
                    result = self._analyze_single_security(symbol, options)
                    results.append(result)
                    logger.info(f"Completed comprehensive analysis for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to analyze {symbol}: {str(e)}")
                    failed_symbols.append(symbol)
                    if not self.continue_on_error:
                        logger.error("Stopping batch processing due to error")
                        break
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Create report
        report = ComprehensiveAnalysisReport(
            results=results,
            total_securities=len(symbols),
            successful_analyses=len(results),
            failed_analyses=len(failed_symbols),
            failed_symbols=failed_symbols,
            execution_time=execution_time
        )
        
        logger.info(f"Comprehensive analysis completed: {len(results)}/{len(symbols)} successful "
                   f"({report.success_rate:.1f}%) in {execution_time:.2f}s")
        
        return report
    
    def _analyze_single_security(self, 
                               symbol: str, 
                               options: Dict[str, Any]) -> ComprehensiveAnalysisResult:
        """Analyze a single security with comprehensive data.
        
        Args:
            symbol: Stock symbol to analyze
            options: Analysis options
            
        Returns:
            ComprehensiveAnalysisResult: Comprehensive analysis result
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.info(f"Starting comprehensive analysis for {symbol}")
        
        # Initialize result
        result = ComprehensiveAnalysisResult(symbol=symbol)
        
        # Use parallel processing if enabled
        if self.enable_parallel_processing:
            self._analyze_single_security_parallel(symbol, options, result)
        else:
            self._analyze_single_security_sequential(symbol, options, result)
        
        # Log a summary of what was retrieved
        self._log_analysis_summary(symbol, result)
        
        return result
        
    def _log_analysis_summary(self, symbol: str, result: ComprehensiveAnalysisResult) -> None:
        """Log a summary of what data was successfully retrieved.
        
        Args:
            symbol: Stock symbol
            result: Analysis result
        """
        components = []
        
        if result.analysis_result:
            components.append("stock analysis")
        
        financial_statements_count = sum(1 for stmt in result.financial_statements.values() if stmt is not None)
        if financial_statements_count > 0:
            components.append(f"{financial_statements_count} financial statements")
        
        if result.news_items:
            components.append(f"{len(result.news_items)} news items")
            
        if result.news_sentiment:
            components.append("news sentiment")
        
        if components:
            logger.info(f"Successfully retrieved {', '.join(components)} for {symbol}")
        else:
            logger.warning(f"No data was successfully retrieved for {symbol}")
            
        # Log any missing components that should have been retrieved
        missing = []
        
        if not result.analysis_result:
            missing.append("stock analysis")
            
        if financial_statements_count == 0:
            missing.append("financial statements")
            
        if not result.news_items:
            missing.append("news items")
            
        if missing:
            logger.warning(f"Failed to retrieve {', '.join(missing)} for {symbol}")
            
        return
    
    def _analyze_single_security_sequential(self,
                                          symbol: str,
                                          options: Dict[str, Any],
                                          result: ComprehensiveAnalysisResult) -> None:
        """Analyze a single security sequentially.
        
        Args:
            symbol: Stock symbol to analyze
            options: Analysis options
            result: Result object to populate
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        # Get stock analysis
        if not options['skip_analysis']:
            try:
                logger.debug(f"Retrieving stock analysis for {symbol}")
                # Pass any relevant options to the stock analysis orchestrator
                include_technicals = options.get('include_technicals', False)
                include_analyst = options.get('include_analyst', False)
                
                # Reuse the existing StockAnalysisOrchestrator to retrieve stock analysis data
                try:
                    # Pass any additional options that might be needed for analysis
                    include_technicals = options.get('include_technicals', False)
                    include_analyst = options.get('include_analyst', False)
                    
                    # Use the existing orchestrator to retrieve stock analysis data
                    # Create a new orchestrator with the required parameters if needed
                    if include_technicals or include_analyst:
                        temp_orchestrator = StockAnalysisOrchestrator(
                            enable_parallel_processing=self.enable_parallel_processing,
                            max_workers=self.max_workers,
                            continue_on_error=self.continue_on_error,
                            include_technicals=include_technicals,
                            include_analyst=include_analyst
                        )
                        result.analysis_result = temp_orchestrator.analyze_single_security(symbol)
                    else:
                        result.analysis_result = self.stock_analysis_orchestrator.analyze_single_security(symbol)
                    logger.debug(f"Successfully retrieved stock analysis for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to retrieve stock analysis for {symbol}: {str(e)}")
                    # Log detailed error information for debugging
                    logger.debug(f"Stock analysis error details: {type(e).__name__}", exc_info=True)
                    if not self.continue_on_error:
                        raise DataRetrievalError(
                            f"Failed to retrieve stock analysis for {symbol}: {str(e)}",
                            symbol=symbol,
                            data_type="analysis"
                        )
                    # If we continue on error, we'll have a partial result with analysis_result as None
            except Exception as e:
                logger.error(f"Failed to retrieve stock analysis for {symbol}: {str(e)}")
                if not self.continue_on_error:
                    raise
                # If we continue on error, we'll have a partial result with analysis_result as None
        
        # Get financial statements
        if not options['skip_financials']:
            try:
                logger.debug(f"Retrieving financial statements for {symbol}")
                self._retrieve_financial_statements(symbol, options, result)
                logger.debug(f"Successfully retrieved financial statements for {symbol}")
            except Exception as e:
                logger.error(f"Failed to retrieve financial statements for {symbol}: {str(e)}")
                if not self.continue_on_error:
                    raise
        
        # Get financial news
        if not options['skip_news']:
            try:
                logger.debug(f"Retrieving financial news for {symbol}")
                self._retrieve_financial_news(symbol, options, result)
                logger.debug(f"Successfully retrieved financial news for {symbol}")
            except Exception as e:
                logger.error(f"Failed to retrieve financial news for {symbol}: {str(e)}")
                if not self.continue_on_error:
                    raise
    
    def _analyze_single_security_parallel(self,
                                        symbol: str,
                                        options: Dict[str, Any],
                                        result: ComprehensiveAnalysisResult) -> None:
        """Analyze a single security with parallel processing.
        
        Args:
            symbol: Stock symbol to analyze
            options: Analysis options
            result: Result object to populate
            
        Raises:
            DataRetrievalError: If data cannot be retrieved and continue_on_error is False
        """
        # Import the enhanced parallel processing implementation
        from stock_analysis.comprehensive_orchestrator_parallel import ParallelTaskManager
        
        # Define tasks to execute in parallel
        tasks = []
        
        if not options['skip_analysis']:
            # Create a task for stock analysis retrieval with improved error handling
            def stock_analysis_task():
                try:
                    # Pass any additional options that might be needed for analysis
                    include_technicals = options.get('include_technicals', False)
                    include_analyst = options.get('include_analyst', False)
                    
                    # Use the existing orchestrator to retrieve stock analysis data
                    return self.stock_analysis_orchestrator.analyze_single_security(
                        symbol,
                        include_technicals=include_technicals,
                        include_analyst=include_analyst
                    )
                except Exception as e:
                    logger.error(f"Failed to retrieve stock analysis for {symbol}: {str(e)}")
                    # Log detailed error information for debugging
                    logger.debug(f"Stock analysis error details: {type(e).__name__}", exc_info=True)
                    if not self.continue_on_error:
                        raise DataRetrievalError(
                            f"Failed to retrieve stock analysis for {symbol}: {str(e)}",
                            symbol=symbol,
                            data_type="analysis"
                        )
                    return None
                    
            tasks.append(('analysis', stock_analysis_task))
        
        if not options['skip_financials']:
            tasks.append(('financials', lambda: self._retrieve_financial_statements_task(symbol, options)))
        
        if not options['skip_news']:
            tasks.append(('news', lambda: self._retrieve_financial_news_task(symbol, options)))
        
        # Create a task manager for improved parallel processing
        task_manager = ParallelTaskManager(
            max_workers=min(len(tasks), self.max_workers),
            continue_on_error=self.continue_on_error
        )
        
        # Execute tasks in parallel with enhanced error handling and resource management
        try:
            results = task_manager.execute_tasks(tasks)
            
            # Store results based on task type
            if 'analysis' in results:
                result.analysis_result = results['analysis']
                logger.debug(f"Successfully retrieved stock analysis for {symbol}")
            
            if 'financials' in results:
                result.financial_statements = results['financials']
                logger.debug(f"Successfully retrieved financial statements for {symbol}")
            
            if 'news' in results:
                result.news_items = results['news']['news_items']
                result.news_sentiment = results['news']['news_sentiment']
                logger.debug(f"Successfully retrieved financial news for {symbol}")
            
            # Log any errors
            if '_errors' in results:
                for task_name, error in results['_errors'].items():
                    logger.error(f"Failed to retrieve {task_name} for {symbol}: {error}")
        
        except Exception as e:
            logger.error(f"Parallel analysis failed for {symbol}: {str(e)}")
            if not self.continue_on_error:
                raise DataRetrievalError(
                    f"Failed to analyze {symbol}: {str(e)}",
                    symbol=symbol,
                    data_type="comprehensive"
                )
    
    def _retrieve_financial_statements(self,
                                     symbol: str,
                                     options: Dict[str, Any],
                                     result: ComprehensiveAnalysisResult) -> None:
        """Retrieve financial statements for a symbol.
        
        Args:
            symbol: Stock symbol
            options: Analysis options
            result: Result object to populate
            
        Raises:
            DataRetrievalError: If data cannot be retrieved and continue_on_error is False
        """
        statement_type = options['statement']
        period = options['period']
        years = options.get('years', 5)  # Default to 5 years if not specified
        
        logger.info(f"Retrieving financial statements for {symbol} (type: {statement_type}, period: {period}, years: {years})")
        
        # Track successful retrievals
        successful_statements = []
        
        # Retrieve all statement types if 'all' is specified
        if statement_type == 'all' or statement_type == 'income':
            try:
                logger.debug(f"Retrieving income statement for {symbol}")
                income_statement = self.stock_data_service.get_financial_statements(
                    symbol, "income", period, use_cache=True
                )
                
                # Filter by years if specified
                if years > 0 and not income_statement.empty and len(income_statement.columns) > years:
                    # Take only the most recent 'years' columns
                    income_statement = income_statement.iloc[:, :years]
                    logger.debug(f"Filtered income statement to {years} years for {symbol}")
                
                result.financial_statements['income_statement'] = income_statement
                successful_statements.append("income statement")
                logger.debug(f"Successfully retrieved income statement for {symbol} with {len(income_statement.columns)} periods")
            except Exception as e:
                logger.error(f"Failed to retrieve income statement for {symbol}: {str(e)}")
                logger.debug(f"Income statement error details: {type(e).__name__}", exc_info=True)
                if not self.continue_on_error:
                    raise DataRetrievalError(
                        f"Failed to retrieve income statement for {symbol}: {str(e)}",
                        symbol=symbol,
                        data_type="income_statement"
                    )
        
        if statement_type == 'all' or statement_type == 'balance':
            try:
                logger.debug(f"Retrieving balance sheet for {symbol}")
                balance_sheet = self.stock_data_service.get_financial_statements(
                    symbol, "balance", period, use_cache=True
                )
                
                # Filter by years if specified
                if years > 0 and not balance_sheet.empty and len(balance_sheet.columns) > years:
                    # Take only the most recent 'years' columns
                    balance_sheet = balance_sheet.iloc[:, :years]
                    logger.debug(f"Filtered balance sheet to {years} years for {symbol}")
                
                result.financial_statements['balance_sheet'] = balance_sheet
                successful_statements.append("balance sheet")
                logger.debug(f"Successfully retrieved balance sheet for {symbol} with {len(balance_sheet.columns)} periods")
            except Exception as e:
                logger.error(f"Failed to retrieve balance sheet for {symbol}: {str(e)}")
                logger.debug(f"Balance sheet error details: {type(e).__name__}", exc_info=True)
                if not self.continue_on_error:
                    raise DataRetrievalError(
                        f"Failed to retrieve balance sheet for {symbol}: {str(e)}",
                        symbol=symbol,
                        data_type="balance_sheet"
                    )
        
        if statement_type == 'all' or statement_type == 'cash':
            try:
                logger.debug(f"Retrieving cash flow statement for {symbol}")
                cash_flow = self.stock_data_service.get_financial_statements(
                    symbol, "cash", period, use_cache=True
                )
                
                # Filter by years if specified
                if years > 0 and not cash_flow.empty and len(cash_flow.columns) > years:
                    # Take only the most recent 'years' columns
                    cash_flow = cash_flow.iloc[:, :years]
                    logger.debug(f"Filtered cash flow statement to {years} years for {symbol}")
                
                result.financial_statements['cash_flow'] = cash_flow
                successful_statements.append("cash flow statement")
                logger.debug(f"Successfully retrieved cash flow statement for {symbol} with {len(cash_flow.columns)} periods")
            except Exception as e:
                logger.error(f"Failed to retrieve cash flow statement for {symbol}: {str(e)}")
                logger.debug(f"Cash flow statement error details: {type(e).__name__}", exc_info=True)
                if not self.continue_on_error:
                    raise DataRetrievalError(
                        f"Failed to retrieve cash flow statement for {symbol}: {str(e)}",
                        symbol=symbol,
                        data_type="cash_flow"
                    )
        
        # Log summary of retrieved statements
        if successful_statements:
            logger.info(f"Successfully retrieved {', '.join(successful_statements)} for {symbol}")
        else:
            logger.warning(f"No financial statements were successfully retrieved for {symbol}")
    
    def _retrieve_financial_statements_task(self,
                                          symbol: str,
                                          options: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve financial statements for a symbol as a parallel task.
        
        Args:
            symbol: Stock symbol
            options: Analysis options
            
        Returns:
            Dictionary containing financial statements and metadata
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        financial_statements = {
            'income_statement': None,
            'balance_sheet': None,
            'cash_flow': None,
            '_metadata': {
                'statement_type': options['statement'],
                'period': options['period'],
                'years': options.get('years', 5),
                'timestamp': datetime.now(),
                'symbol': symbol
            }
        }
        
        statement_type = options['statement']
        period = options['period']
        years = options.get('years', 5)  # Default to 5 years if not specified
        
        logger.debug(f"Financial statements task for {symbol} (type: {statement_type}, period: {period}, years: {years})")
        
        # Retrieve all statement types if 'all' is specified
        if statement_type == 'all' or statement_type == 'income':
            try:
                income_statement = self.stock_data_service.get_financial_statements(
                    symbol, "income", period, use_cache=True
                )
                
                # Filter by years if specified
                if years > 0 and not income_statement.empty and len(income_statement.columns) > years:
                    # Take only the most recent 'years' columns
                    income_statement = income_statement.iloc[:, :years]
                
                financial_statements['income_statement'] = income_statement
                logger.debug(f"Retrieved income statement for {symbol} with {len(income_statement.columns) if not income_statement.empty else 0} periods")
            except Exception as e:
                logger.error(f"Failed to retrieve income statement for {symbol}: {str(e)}")
                if not self.continue_on_error:
                    raise DataRetrievalError(
                        f"Failed to retrieve income statement for {symbol}: {str(e)}",
                        symbol=symbol,
                        data_type="income_statement"
                    )
        
        if statement_type == 'all' or statement_type == 'balance':
            try:
                balance_sheet = self.stock_data_service.get_financial_statements(
                    symbol, "balance", period, use_cache=True
                )
                
                # Filter by years if specified
                if years > 0 and not balance_sheet.empty and len(balance_sheet.columns) > years:
                    # Take only the most recent 'years' columns
                    balance_sheet = balance_sheet.iloc[:, :years]
                
                financial_statements['balance_sheet'] = balance_sheet
                logger.debug(f"Retrieved balance sheet for {symbol} with {len(balance_sheet.columns) if not balance_sheet.empty else 0} periods")
            except Exception as e:
                logger.error(f"Failed to retrieve balance sheet for {symbol}: {str(e)}")
                if not self.continue_on_error:
                    raise DataRetrievalError(
                        f"Failed to retrieve balance sheet for {symbol}: {str(e)}",
                        symbol=symbol,
                        data_type="balance_sheet"
                    )
        
        if statement_type == 'all' or statement_type == 'cash':
            try:
                cash_flow = self.stock_data_service.get_financial_statements(
                    symbol, "cash", period, use_cache=True
                )
                
                # Filter by years if specified
                if years > 0 and not cash_flow.empty and len(cash_flow.columns) > years:
                    # Take only the most recent 'years' columns
                    cash_flow = cash_flow.iloc[:, :years]
                
                financial_statements['cash_flow'] = cash_flow
                logger.debug(f"Retrieved cash flow statement for {symbol} with {len(cash_flow.columns) if not cash_flow.empty else 0} periods")
            except Exception as e:
                logger.error(f"Failed to retrieve cash flow statement for {symbol}: {str(e)}")
                if not self.continue_on_error:
                    raise DataRetrievalError(
                        f"Failed to retrieve cash flow statement for {symbol}: {str(e)}",
                        symbol=symbol,
                        data_type="cash_flow"
                    )
        
        return financial_statements
        
    def _retrieve_financial_news(self,
                               symbol: str,
                               options: Dict[str, Any],
                               result: ComprehensiveAnalysisResult) -> None:
        """Retrieve financial news for a symbol.
        
        Args:
            symbol: Stock symbol
            options: Analysis options
            result: Result object to populate
            
        Raises:
            DataRetrievalError: If data cannot be retrieved and continue_on_error is False
        """
        news_limit = options.get('news_limit', 10)
        news_days = options.get('news_days', 7)
        include_sentiment = options.get('sentiment', True)
        
        logger.info(f"Retrieving financial news for {symbol} (limit: {news_limit}, days: {news_days}, sentiment: {include_sentiment})")
        
        try:
            # Get company news
            logger.debug(f"Retrieving company news for {symbol}")
            news_items = self.news_service.get_company_news(
                symbol=symbol,
                days=news_days,
                limit=news_limit,
                include_sentiment=include_sentiment,
                use_cache=True
            )
            
            # Check if news_items is a list and not empty
            has_news = isinstance(news_items, list) and len(news_items) > 0
            
            if has_news:
                result.news_items = news_items
                logger.debug(f"Successfully retrieved {len(news_items)} news items for {symbol}")
                
                # Get news sentiment if requested and we have news items
                if include_sentiment:
                    try:
                        logger.debug(f"Retrieving news sentiment for {symbol}")
                        news_sentiment = self.news_service.get_news_sentiment(
                            symbol=symbol,
                            days=news_days,
                            use_cache=True
                        )
                        
                        if news_sentiment:
                            result.news_sentiment = news_sentiment
                            logger.debug(f"Successfully retrieved news sentiment for {symbol}: {news_sentiment.overall_sentiment:.2f}")
                        else:
                            logger.warning(f"No sentiment data available for {symbol}")
                            result.news_sentiment = None
                    except Exception as e:
                        logger.error(f"Failed to retrieve news sentiment for {symbol}: {str(e)}")
                        logger.debug(f"News sentiment error details: {type(e).__name__}", exc_info=True)
                        if not self.continue_on_error:
                            raise DataRetrievalError(
                                f"Failed to retrieve news sentiment for {symbol}: {str(e)}",
                                symbol=symbol,
                                data_source="news_sentiment"
                            )
            else:
                logger.warning(f"No news items found for {symbol}")
                result.news_items = []
                # Explicitly set news_sentiment to None when no news items are found
                result.news_sentiment = None
        except Exception as e:
            logger.error(f"Failed to retrieve financial news for {symbol}: {str(e)}")
            logger.debug(f"Financial news error details: {type(e).__name__}", exc_info=True)
            if not self.continue_on_error:
                raise DataRetrievalError(
                    f"Failed to retrieve financial news for {symbol}: {str(e)}",
                    symbol=symbol,
                    data_source="financial_news"
                )
    
    def _retrieve_financial_news_task(self,
                                    symbol: str,
                                    options: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve financial news for a symbol as a parallel task.
        
        Args:
            symbol: Stock symbol
            options: Analysis options
            
        Returns:
            Dictionary containing news items and sentiment
            
        Raises:
            DataRetrievalError: If data cannot be retrieved and continue_on_error is False
        """
        news_limit = options.get('news_limit', 10)
        news_days = options.get('news_days', 7)
        include_sentiment = options.get('sentiment', True)
        
        logger.debug(f"Financial news task for {symbol} (limit: {news_limit}, days: {news_days}, sentiment: {include_sentiment})")
        
        result = {
            'news_items': [],
            'news_sentiment': None,
            '_metadata': {
                'symbol': symbol,
                'news_limit': news_limit,
                'news_days': news_days,
                'include_sentiment': include_sentiment,
                'timestamp': datetime.now()
            }
        }
        
        try:
            # Get company news
            news_items = self.news_service.get_company_news(
                symbol=symbol,
                days=news_days,
                limit=news_limit,
                include_sentiment=include_sentiment,
                use_cache=True
            )
            
            # Check if news_items is a list and not empty
            has_news = isinstance(news_items, list) and len(news_items) > 0
            
            if has_news:
                result['news_items'] = news_items
                logger.debug(f"Retrieved {len(news_items)} news items for {symbol}")
                
                # Get news sentiment if requested and we have news items
                if include_sentiment:
                    try:
                        news_sentiment = self.news_service.get_news_sentiment(
                            symbol=symbol,
                            days=news_days,
                            use_cache=True
                        )
                        
                        if news_sentiment:
                            result['news_sentiment'] = news_sentiment
                            logger.debug(f"Retrieved news sentiment for {symbol}: {news_sentiment.overall_sentiment:.2f}")
                        else:
                            logger.warning(f"No sentiment data available for {symbol}")
                    except Exception as e:
                        logger.error(f"Failed to retrieve news sentiment for {symbol}: {str(e)}")
                        logger.debug(f"News sentiment error details: {type(e).__name__}", exc_info=True)
                        if not self.continue_on_error:
                            raise DataRetrievalError(
                                f"Failed to retrieve news sentiment for {symbol}: {str(e)}",
                                symbol=symbol,
                                data_source="news_sentiment"
                            )
            else:
                logger.warning(f"No news items found for {symbol}")
        except Exception as e:
            logger.error(f"Failed to retrieve financial news for {symbol}: {str(e)}")
            logger.debug(f"Financial news error details: {type(e).__name__}", exc_info=True)
            if not self.continue_on_error:
                raise DataRetrievalError(
                    f"Failed to retrieve financial news for {symbol}: {str(e)}",
                    symbol=symbol,
                    data_source="financial_news"
                )
        
        return result
        # Add metadata about the retrieval
        financial_statements['_metadata'] = {
            'statement_type': statement_type,
            'period': period,
            'years': years,
            'timestamp': datetime.now()
        }
        
        return financial_statements
    
    def _retrieve_financial_news(self,
                               symbol: str,
                               options: Dict[str, Any],
                               result: ComprehensiveAnalysisResult) -> None:
        """Retrieve financial news for a symbol.
        
        Args:
            symbol: Stock symbol
            options: Analysis options
            result: Result object to populate
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        news_limit = options.get('news_limit', 10)
        news_days = options.get('news_days', 7)
        include_sentiment = options.get('sentiment', True)
        
        logger.info(f"Retrieving financial news for {symbol} (limit: {news_limit}, days: {news_days}, sentiment: {include_sentiment})")
        
        try:
            # Get company news
            logger.debug(f"Retrieving company news for {symbol}")
            news_items = self.news_service.get_company_news(
                symbol=symbol,
                days=news_days,
                limit=news_limit,
                include_sentiment=include_sentiment,
                use_cache=True
            )
            
            if news_items:
                result.news_items = news_items
                logger.debug(f"Successfully retrieved {len(news_items)} news items for {symbol}")
                
                # Get news sentiment if requested and we have news items
                if include_sentiment:
                    try:
                        logger.debug(f"Retrieving news sentiment for {symbol}")
                        news_sentiment = self.news_service.get_news_sentiment(
                            symbol=symbol,
                            days=news_days,
                            use_cache=True
                        )
                        
                        if news_sentiment:
                            result.news_sentiment = news_sentiment
                            logger.debug(f"Successfully retrieved news sentiment for {symbol}: {news_sentiment.overall_sentiment:.2f}")
                        else:
                            logger.warning(f"No sentiment data available for {symbol}")
                            result.news_sentiment = None
                    except Exception as e:
                        logger.error(f"Failed to retrieve news sentiment for {symbol}: {str(e)}")
                        logger.debug(f"News sentiment error details: {type(e).__name__}", exc_info=True)
                        if not self.continue_on_error:
                            raise DataRetrievalError(
                                f"Failed to retrieve news sentiment for {symbol}: {str(e)}",
                                symbol=symbol,
                                data_source="news_sentiment"
                            )
            else:
                logger.warning(f"No news items found for {symbol}")
                result.news_items = []
                result.news_sentiment = None
                try:
                    logger.debug(f"Retrieving news sentiment for {symbol}")
                    news_sentiment = self.news_service.get_news_sentiment(
                        symbol=symbol,
                        days=news_days,
                        use_cache=True
                    )
                    
                    if news_sentiment:
                        result.news_sentiment = news_sentiment
                        logger.debug(f"Successfully retrieved news sentiment for {symbol}: {news_sentiment.overall_sentiment:.2f}")
                    else:
                        logger.warning(f"No sentiment data available for {symbol}")
                        result.news_sentiment = None
                except Exception as e:
                    logger.error(f"Failed to retrieve news sentiment for {symbol}: {str(e)}")
                    logger.debug(f"News sentiment error details: {type(e).__name__}", exc_info=True)
                    if not self.continue_on_error:
                        raise DataRetrievalError(
                            f"Failed to retrieve news sentiment for {symbol}: {str(e)}",
                            symbol=symbol,
                            data_source="news_sentiment"
                        )
        except Exception as e:
            logger.error(f"Failed to retrieve financial news for {symbol}: {str(e)}")
            logger.debug(f"Financial news error details: {type(e).__name__}", exc_info=True)
            if not self.continue_on_error:
                raise DataRetrievalError(
                    f"Failed to retrieve financial news for {symbol}: {str(e)}",
                    symbol=symbol,
                    data_source="financial_news"
                )
    
    def _retrieve_financial_news_task(self,
                                    symbol: str,
                                    options: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve financial news for a symbol as a parallel task.
        
        Args:
            symbol: Stock symbol
            options: Analysis options
            
        Returns:
            Dictionary containing news items and sentiment
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        news_limit = options.get('news_limit', 10)
        news_days = options.get('news_days', 7)
        include_sentiment = options.get('sentiment', True)
        
        logger.debug(f"Financial news task for {symbol} (limit: {news_limit}, days: {news_days}, sentiment: {include_sentiment})")
        
        result = {
            'news_items': [],
            'news_sentiment': None,
            '_metadata': {
                'symbol': symbol,
                'news_limit': news_limit,
                'news_days': news_days,
                'include_sentiment': include_sentiment,
                'timestamp': datetime.now()
            }
        }
        
        try:
            # Get company news
            news_items = self.news_service.get_company_news(
                symbol=symbol,
                days=news_days,
                limit=news_limit,
                include_sentiment=include_sentiment,
                use_cache=True
            )
            
            if news_items:
                result['news_items'] = news_items
                logger.debug(f"Retrieved {len(news_items)} news items for {symbol}")
                
                # Get news sentiment if requested and we have news items
                if include_sentiment:
                    try:
                        news_sentiment = self.news_service.get_news_sentiment(
                            symbol=symbol,
                            days=news_days,
                            use_cache=True
                        )
                        
                        if news_sentiment:
                            result['news_sentiment'] = news_sentiment
                            logger.debug(f"Retrieved news sentiment for {symbol}: {news_sentiment.overall_sentiment:.2f}")
                        else:
                            logger.warning(f"No sentiment data available for {symbol}")
                    except Exception as e:
                        logger.error(f"Failed to retrieve news sentiment for {symbol}: {str(e)}")
                        logger.debug(f"News sentiment error details: {type(e).__name__}", exc_info=True)
                        if not self.continue_on_error:
                            raise DataRetrievalError(
                                f"Failed to retrieve news sentiment for {symbol}: {str(e)}",
                                symbol=symbol,
                                data_source="news_sentiment"
                            )
            else:
                logger.warning(f"No news items found for {symbol}")
                try:
                    news_sentiment = self.news_service.get_news_sentiment(
                        symbol=symbol,
                        days=news_days,
                        use_cache=True
                    )
                    
                    if news_sentiment:
                        result['news_sentiment'] = news_sentiment
                        logger.debug(f"Retrieved news sentiment for {symbol}: {news_sentiment.overall_sentiment:.2f}")
                    else:
                        logger.warning(f"No sentiment data available for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to retrieve news sentiment for {symbol}: {str(e)}")
                    logger.debug(f"News sentiment error details: {type(e).__name__}", exc_info=True)
                    if not self.continue_on_error:
                        raise DataRetrievalError(
                            f"Failed to retrieve news sentiment for {symbol}: {str(e)}",
                            symbol=symbol,
                            data_source="news_sentiment"
                        )
        except Exception as e:
            logger.error(f"Failed to retrieve financial news for {symbol}: {str(e)}")
            logger.debug(f"Financial news error details: {type(e).__name__}", exc_info=True)
            if not self.continue_on_error:
                raise DataRetrievalError(
                    f"Failed to retrieve financial news for {symbol}: {str(e)}",
                    symbol=symbol,
                    data_source="financial_news"
                )
        
        return result
    
    def export_results(self,
                      report: ComprehensiveAnalysisReport,
                      export_format: str = "excel",
                      filename: Optional[str] = None) -> str:
        """Export comprehensive analysis results.
        
        Args:
            report: Comprehensive analysis report
            export_format: Export format ('csv', 'excel', 'json')
            filename: Optional filename (without extension)
            
        Returns:
            Path to the exported file
            
        Raises:
            ExportError: If export fails
        """
        logger.info(f"Exporting comprehensive analysis results in {export_format} format")
        
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"comprehensive_analysis_{timestamp}"
            
            # Use the appropriate export method based on format
            if export_format.lower() == "csv":
                return self.export_service.export_to_csv(report.results, filename)
            elif export_format.lower() == "excel":
                return self.export_service.export_to_excel(report.results, filename)
            elif export_format.lower() == "json":
                return self.export_service.export_to_json(report.results, filename)
            else:
                raise ExportError(f"Unsupported export format: {export_format}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {str(e)}")
            raise ExportError(f"Failed to export results: {str(e)}")