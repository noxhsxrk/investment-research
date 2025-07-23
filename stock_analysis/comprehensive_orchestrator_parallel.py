"""Enhanced parallel processing implementation for the ComprehensiveAnalysisOrchestrator.

This module provides improved parallel processing capabilities for the ComprehensiveAnalysisOrchestrator,
allowing efficient coordination of multiple data retrieval operations.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import threading
from dataclasses import dataclass

from stock_analysis.utils.logging import get_logger
from stock_analysis.utils.exceptions import DataRetrievalError

logger = get_logger(__name__)


class ParallelTaskManager:
    """Manages parallel execution of tasks with resource utilization controls.
    
    This class provides enhanced parallel processing capabilities with:
    - Dynamic thread pool management
    - Resource utilization monitoring
    - Task prioritization
    - Error handling and recovery
    """
    
    def __init__(self, max_workers: int = 4, continue_on_error: bool = True):
        """Initialize the parallel task manager.
        
        Args:
            max_workers: Maximum number of worker threads
            continue_on_error: Whether to continue processing if one operation fails
        """
        self.max_workers = max_workers
        self.continue_on_error = continue_on_error
        self.thread_local = threading.local()
        self._lock = threading.RLock()
        self._active_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._start_time = None
        
        logger.info(f"Initialized ParallelTaskManager with max_workers={max_workers}")
    
    def execute_tasks(self, tasks: List[Tuple[str, Callable]], timeout: Optional[float] = None) -> Dict[str, Any]:
        """Execute multiple tasks in parallel.
        
        Args:
            tasks: List of (task_name, task_function) tuples
            timeout: Maximum time to wait for all tasks to complete (None for no timeout)
            
        Returns:
            Dictionary mapping task names to results
            
        Raises:
            TimeoutError: If timeout is reached before all tasks complete
            Exception: If a task fails and continue_on_error is False
        """
        if not tasks:
            logger.warning("No tasks provided to execute_tasks")
            return {}
        
        self._start_time = time.time()
        self._active_tasks = len(tasks)
        self._completed_tasks = 0
        self._failed_tasks = 0
        
        results = {}
        errors = {}
        
        # Use a thread pool with a maximum number of workers
        with ThreadPoolExecutor(max_workers=min(len(tasks), self.max_workers)) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(task_func): task_name for task_name, task_func in tasks}
            
            # Process completed tasks
            for future in as_completed(future_to_task, timeout=timeout):
                task_name = future_to_task[future]
                
                try:
                    task_result = future.result()
                    results[task_name] = task_result
                    self._completed_tasks += 1
                    logger.debug(f"Task '{task_name}' completed successfully")
                except Exception as e:
                    self._failed_tasks += 1
                    errors[task_name] = str(e)
                    logger.error(f"Task '{task_name}' failed: {str(e)}")
                    
                    if not self.continue_on_error:
                        # Cancel all pending tasks
                        for f in future_to_task:
                            if not f.done():
                                f.cancel()
                        
                        # Raise the exception
                        raise
                
                finally:
                    with self._lock:
                        self._active_tasks -= 1
        
        # Log summary
        execution_time = time.time() - self._start_time
        logger.info(f"Parallel execution completed: {self._completed_tasks} succeeded, "
                   f"{self._failed_tasks} failed in {execution_time:.2f}s")
        
        if errors:
            logger.warning(f"Tasks with errors: {', '.join(errors.keys())}")
        
        # Include errors in the results
        if errors:
            results['_errors'] = errors
        
        return results
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        execution_time = 0
        if self._start_time:
            execution_time = time.time() - self._start_time
        
        return {
            'active_tasks': self._active_tasks,
            'completed_tasks': self._completed_tasks,
            'failed_tasks': self._failed_tasks,
            'execution_time': execution_time
        }


def _analyze_multiple_securities_parallel(self, symbols: List[str], options: Dict[str, Any]) -> List[Any]:
    """Analyze multiple securities in parallel.
    
    Args:
        symbols: List of stock symbols to analyze
        options: Analysis options
        
    Returns:
        List of analysis results
        
    Raises:
        DataRetrievalError: If data cannot be retrieved and continue_on_error is False
    """
    logger = self.logger if hasattr(self, 'logger') else get_logger(__name__)
    logger.info(f"Starting parallel analysis for {len(symbols)} securities")
    
    # Create a task manager
    task_manager = ParallelTaskManager(
        max_workers=self.max_workers,
        continue_on_error=self.continue_on_error
    )
    
    # Create tasks for each symbol
    tasks = []
    for symbol in symbols:
        tasks.append((
            f"analyze_{symbol}",
            lambda s=symbol: self._analyze_single_security(s, options)
        ))
    
    # Execute tasks in parallel
    try:
        results_dict = task_manager.execute_tasks(tasks)
        
        # Extract results (excluding _errors key)
        results = [results_dict[task_name] for task_name in results_dict if not task_name.startswith('_')]
        
        # Log any errors
        if '_errors' in results_dict:
            for task_name, error in results_dict['_errors'].items():
                symbol = task_name.split('_')[1]  # Extract symbol from task name
                logger.error(f"Failed to analyze {symbol}: {error}")
        
        return results
    
    except Exception as e:
        logger.error(f"Parallel analysis failed: {str(e)}")
        raise


def _analyze_single_security_parallel(self,
                                    symbol: str,
                                    options: Dict[str, Any],
                                    result: Any) -> None:
    """Analyze a single security with parallel processing.
    
    Args:
        symbol: Stock symbol to analyze
        options: Analysis options
        result: Result object to populate
        
    Raises:
        DataRetrievalError: If data cannot be retrieved and continue_on_error is False
    """
    logger = self.logger if hasattr(self, 'logger') else get_logger(__name__)
    logger.info(f"Starting parallel analysis for {symbol}")
    
    # Define tasks to execute in parallel
    tasks = []
    
    if not options.get('skip_analysis', False):
        # Create a task for stock analysis retrieval
        def stock_analysis_task():
            try:
                # Pass any additional options that might be needed for analysis
                include_technicals = options.get('include_technicals', False)
                include_analyst = options.get('include_analyst', False)
                
                # Use the existing orchestrator to retrieve stock analysis data
                # Create a new orchestrator with the required parameters if needed
                from stock_analysis.orchestrator import StockAnalysisOrchestrator
                
                if include_technicals or include_analyst:
                    temp_orchestrator = StockAnalysisOrchestrator(
                        enable_parallel_processing=self.enable_parallel_processing,
                        max_workers=self.max_workers,
                        continue_on_error=self.continue_on_error,
                        include_technicals=include_technicals,
                        include_analyst=include_analyst
                    )
                    return temp_orchestrator.analyze_single_security(symbol)
                else:
                    return self.stock_analysis_orchestrator.analyze_single_security(symbol)
            except Exception as e:
                logger.error(f"Failed to retrieve stock analysis for {symbol}: {str(e)}")
                logger.debug(f"Stock analysis error details: {type(e).__name__}", exc_info=True)
                if not self.continue_on_error:
                    raise DataRetrievalError(
                        f"Failed to retrieve stock analysis for {symbol}: {str(e)}",
                        symbol=symbol,
                        data_type="analysis"
                    )
                return None
                
        tasks.append(('analysis', stock_analysis_task))
    
    if not options.get('skip_financials', False):
        tasks.append(('financials', lambda: self._retrieve_financial_statements_task(symbol, options)))
    
    if not options.get('skip_news', False):
        tasks.append(('news', lambda: self._retrieve_financial_news_task(symbol, options)))
    
    # Create a task manager
    task_manager = ParallelTaskManager(
        max_workers=min(len(tasks), self.max_workers),
        continue_on_error=self.continue_on_error
    )
    
    # Execute tasks in parallel
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


def analyze_comprehensive_parallel(self, 
                                 symbols: List[str], 
                                 options: Dict[str, Any] = None) -> Any:
    """Perform comprehensive analysis for multiple symbols in parallel.
    
    Args:
        symbols: List of stock symbols to analyze
        options: Dictionary of options to customize the analysis
            
    Returns:
        ComprehensiveAnalysisReport: Report containing the analysis results
    """
    # Import the ComprehensiveAnalysisReport class
    from stock_analysis.comprehensive_orchestrator import ComprehensiveAnalysisReport
    
    logger = self.logger if hasattr(self, 'logger') else get_logger(__name__)
    logger.info(f"Starting comprehensive parallel analysis for {len(symbols)} securities")
    
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
    
    # Use parallel processing for multiple symbols
    if len(symbols) > 1:
        results = self._analyze_multiple_securities_parallel(symbols, options)
        failed_symbols = [symbol for symbol in symbols if symbol not in [r.symbol for r in results]]
    else:
        # For a single symbol, use the single security analysis
        symbol = symbols[0]
        try:
            result = self._analyze_single_security(symbol, options)
            results = [result]
            failed_symbols = []
        except Exception as e:
            logger.error(f"Failed to analyze {symbol}: {str(e)}")
            results = []
            failed_symbols = [symbol]
    
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