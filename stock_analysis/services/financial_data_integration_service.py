"""Financial data integration service.

This module provides a service for integrating financial data from multiple sources,
with source prioritization, fallback logic, and data normalization.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from stock_analysis.adapters.base_adapter import FinancialDataAdapter
from stock_analysis.adapters.yfinance_adapter import YFinanceAdapter
from stock_analysis.adapters.investing_adapter import InvestingAdapter
from stock_analysis.adapters.alpha_vantage_adapter import AlphaVantageAdapter
from stock_analysis.utils.cache_manager import get_cache_manager
from stock_analysis.utils.exceptions import DataRetrievalError, ValidationError
from stock_analysis.utils.logging import get_logger, get_operation_logger, log_data_quality_issue
from stock_analysis.utils.config import config
from stock_analysis.utils.error_recovery import with_error_recovery, with_data_validation
from stock_analysis.utils.data_quality import get_data_quality_validator, validate_data_source_response

logger = get_logger(__name__)


class FinancialDataIntegrationService:
    """Service for integrating data from multiple financial data sources.
    
    This service coordinates data retrieval from multiple sources with
    prioritization, fallback logic, and data normalization.
    """
    
    def __init__(self, adapters: Optional[List[FinancialDataAdapter]] = None):
        """Initialize the financial data integration service.
        
        Args:
            adapters: List of data source adapters (if None, default adapters are used)
        """
        self.logger = get_logger(f"{__name__}.FinancialDataIntegrationService")
        self.cache = get_cache_manager()
        
        # Initialize adapters if not provided
        if adapters is None:
            self.adapters = self._initialize_default_adapters()
        else:
            self.adapters = adapters
        
        # Load source priorities from config
        self.source_priorities = self._load_source_priorities()
        
        # Cache settings
        self.cache_ttl = {
            "security_info": config.get('stock_analysis.integration.cache_ttl.security_info', 3600),
            "historical_prices": config.get('stock_analysis.integration.cache_ttl.historical_prices', 3600),
            "financial_statements": config.get('stock_analysis.integration.cache_ttl.financial_statements', 86400),
            "technical_indicators": config.get('stock_analysis.integration.cache_ttl.technical_indicators', 1800),
            "market_data": config.get('stock_analysis.integration.cache_ttl.market_data', 300),
            "news": config.get('stock_analysis.integration.cache_ttl.news', 300),
            "economic_data": config.get('stock_analysis.integration.cache_ttl.economic_data', 3600),
        }
        
        # Performance optimization settings
        self.max_workers = config.get('stock_analysis.integration.max_workers', 3)
        self.batch_size = config.get('stock_analysis.integration.batch_size', 10)
        self.parallel_enabled = config.get('stock_analysis.integration.parallel_enabled', True)
        
        # Import parallel processing utilities
        from stock_analysis.utils.parallel_processing import get_parallel_executor
        self.parallel_executor = get_parallel_executor()
        
        # Import performance metrics
        from stock_analysis.utils.performance_metrics import get_metrics_collector
        self.metrics_collector = get_metrics_collector()
    
    def _initialize_default_adapters(self) -> List[FinancialDataAdapter]:
        """Initialize default adapters.
        
        Returns:
            List of initialized adapters
        """
        adapters = []
        
        try:
            adapters.append(YFinanceAdapter())
            self.logger.info("Initialized YFinance adapter")
        except Exception as e:
            self.logger.warning(f"Failed to initialize YFinance adapter: {str(e)}")
        
        try:
            adapters.append(InvestingAdapter())
            self.logger.info("Initialized Investing.com adapter")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Investing.com adapter: {str(e)}")
        
        try:
            adapters.append(AlphaVantageAdapter())
            self.logger.info("Initialized Alpha Vantage adapter")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Alpha Vantage adapter: {str(e)}")
        
        if not adapters:
            self.logger.error("No adapters could be initialized")
        
        return adapters
    
    def _load_source_priorities(self) -> Dict[str, Dict[str, int]]:
        """Load source priorities from configuration.
        
        Returns:
            Dictionary mapping data types to source priority dictionaries
        """
        # Default priorities (lower number = higher priority)
        default_priorities = {
            "security_info": {"investing": 1, "alpha_vantage": 2, "yfinance": 3},
            "historical_prices": {"yfinance": 1, "investing": 2, "alpha_vantage": 3},
            "financial_statements": {"investing": 1, "alpha_vantage": 2, "yfinance": 3},
            "technical_indicators": {"investing": 1, "alpha_vantage": 2, "yfinance": 3},
            "market_data": {"investing": 1, "alpha_vantage": 2, "yfinance": 3},
            "news": {"investing": 1, "alpha_vantage": 2, "yfinance": 3},
            "economic_data": {"investing": 1, "alpha_vantage": 2, "yfinance": 3},
        }
        
        # Load priorities from config
        priorities = {}
        for data_type, default_priority in default_priorities.items():
            config_key = f"stock_analysis.integration.priorities.{data_type}"
            config_priority = config.get(config_key, default_priority)
            priorities[data_type] = config_priority
        
        return priorities
    
    def get_adapter_by_name(self, name: str) -> Optional[FinancialDataAdapter]:
        """Get adapter by name.
        
        Args:
            name: Adapter name
            
        Returns:
            Adapter instance or None if not found
        """
        for adapter in self.adapters:
            if adapter.name.lower() == name.lower():
                return adapter
        return None
    
    @with_error_recovery("get_security_info", retry_category="data_retrieval", validate_result=True)
    def get_security_info(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """Get comprehensive security information from optimal sources.
        
        Args:
            symbol: Security ticker symbol
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary containing security information
            
        Raises:
            DataRetrievalError: If data cannot be retrieved from any source
        """
        cache_key = f"integration_security_info_{symbol}"
        data_type = "security_info"
        
        # Create operation logger for detailed logging
        op_logger = get_operation_logger(__name__, f"get_security_info_{symbol}")
        op_logger.start(symbol=symbol, data_type=data_type, use_cache=use_cache)
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                self.logger.debug(f"Using cached security info for {symbol}")
                op_logger.finish(success=True, from_cache=True)
                return cached_data
        
        self.logger.info(f"Retrieving integrated security info for {symbol}")
        
        # Get prioritized adapters for this data type
        prioritized_adapters = self._get_prioritized_adapters(data_type)
        
        # Try each adapter in priority order
        errors = []
        partial_results = {}
        data_quality_issues = {}
        
        for adapter in prioritized_adapters:
            try:
                self.logger.debug(f"Trying {adapter.name} for security info")
                op_logger.progress(f"Trying adapter: {adapter.name}")
                
                # Use error recovery and data validation for adapter call
                result = self._get_security_info_from_adapter(adapter, symbol)
                
                # Validate data quality
                validator = get_data_quality_validator()
                is_valid, missing_fields, invalid_values = validator.validate_security_info(symbol, result)
                
                # If we got a complete result, cache and return it
                if is_valid:
                    self.logger.debug(f"Got valid security info from {adapter.name}")
                    op_logger.progress(f"Got valid data from {adapter.name}")
                    
                    if use_cache:
                        self.cache.set(
                            cache_key, 
                            result, 
                            ttl=self.cache_ttl.get("security_info"),
                            data_type=data_type,
                            tags=[symbol, adapter.name]
                        )
                    
                    op_logger.finish(success=True, adapter=adapter.name)
                    return result
                
                # Otherwise, store partial result and data quality issues
                partial_results[adapter.name] = result
                data_quality_issues[adapter.name] = {
                    'missing_fields': missing_fields,
                    'invalid_values': invalid_values
                }
                
                self.logger.debug(
                    f"Got partial security info from {adapter.name}: "
                    f"{len(missing_fields)} missing fields, {len(invalid_values)} invalid values"
                )
                
            except Exception as e:
                self.logger.warning(f"Error retrieving security info from {adapter.name}: {str(e)}")
                op_logger.progress(f"Error from {adapter.name}: {str(e)}")
                errors.append(f"{adapter.name}: {str(e)}")
        
        # If we have partial results, combine them
        if partial_results:
            self.logger.info(
                f"Combining partial results from {len(partial_results)} sources for {symbol}"
            )
            op_logger.progress(f"Combining partial results from {len(partial_results)} sources")
            
            combined_result = self._combine_security_info(partial_results)
            
            # Add data quality metadata
            combined_result['_data_quality'] = {
                'combined_from': list(partial_results.keys()),
                'issues': data_quality_issues
            }
            
            if use_cache:
                self.cache.set(
                    cache_key, 
                    combined_result, 
                    ttl=self.cache_ttl.get("security_info"),
                    data_type=data_type,
                    tags=[symbol]
                )
            
            op_logger.finish(success=True, combined=True, sources=list(partial_results.keys()))
            return combined_result
        
        # If we get here, all sources failed
        error_msg = f"Failed to retrieve security info for {symbol} from any source: {'; '.join(errors)}"
        self.logger.error(error_msg)
        op_logger.finish(success=False, errors=errors)
        raise DataRetrievalError(error_msg, symbol=symbol, data_type=data_type)
    
    @with_error_recovery("get_security_info_from_adapter", retry_category="data_retrieval")
    @with_data_validation("security_info")
    def _get_security_info_from_adapter(self, adapter: FinancialDataAdapter, symbol: str) -> Dict[str, Any]:
        """Get security info from a specific adapter with error handling.
        
        Args:
            adapter: Data adapter to use
            symbol: Security ticker symbol
            
        Returns:
            Security information dictionary
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        try:
            return adapter.get_security_info(symbol)
        except Exception as e:
            # Convert any exception to DataRetrievalError with context
            raise DataRetrievalError(
                f"Failed to retrieve security info for {symbol} from {adapter.name}: {str(e)}",
                symbol=symbol,
                data_source=adapter.name,
                original_exception=e
            )
        
    def get_enhanced_security_info(self, symbol: str, include_technicals: bool = True, 
                                  include_analyst_data: bool = True, use_cache: bool = True) -> Dict[str, Any]:
        """Get enhanced security information with additional data points.
        
        This method integrates security information with technical indicators,
        analyst data, and additional metadata from multiple sources.
        
        Args:
            symbol: Security ticker symbol
            include_technicals: Whether to include technical indicators
            include_analyst_data: Whether to include analyst recommendations and price targets
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary containing enhanced security information
            
        Raises:
            DataRetrievalError: If data cannot be retrieved from any source
        """
        cache_key = f"integration_enhanced_security_{symbol}_{include_technicals}_{include_analyst_data}"
        data_type = "security_info"
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                self.logger.debug(f"Using cached enhanced security info for {symbol}")
                return cached_data
        
        self.logger.info(f"Retrieving integrated enhanced security info for {symbol}")
        
        try:
            # Get base security info
            base_info = self.get_security_info(symbol, use_cache=use_cache)
            
            # Start with base info
            enhanced_info = base_info.copy()
            
            # Add technical indicators if requested
            if include_technicals:
                try:
                    technical_indicators = self.get_technical_indicators(
                        symbol, 
                        indicators=['sma_20', 'sma_50', 'sma_200', 'rsi_14', 'macd'],
                        use_cache=use_cache
                    )
                    
                    # Add technical indicators to enhanced info
                    for indicator, value in technical_indicators.items():
                        if indicator != '_sources':
                            enhanced_info[indicator] = value
                    
                    # Add technical indicators source info
                    if '_sources' in technical_indicators:
                        if '_sources' not in enhanced_info:
                            enhanced_info['_sources'] = {}
                        enhanced_info['_sources'].update(technical_indicators['_sources'])
                        
                except Exception as e:
                    self.logger.warning(f"Error retrieving technical indicators for {symbol}: {str(e)}")
            
            # Add analyst data if requested
            if include_analyst_data:
                try:
                    analyst_data = self._get_analyst_data(symbol, use_cache=use_cache)
                    
                    # Add analyst data to enhanced info
                    for key, value in analyst_data.items():
                        enhanced_info[key] = value
                        
                except Exception as e:
                    self.logger.warning(f"Error retrieving analyst data for {symbol}: {str(e)}")
            
            # Cache the enhanced info
            if use_cache:
                self.cache.set(
                    cache_key, 
                    enhanced_info, 
                    ttl=self.cache_ttl.get("security_info"),
                    data_type=data_type,
                    tags=[symbol, 'enhanced']
                )
            
            return enhanced_info
            
        except Exception as e:
            error_msg = f"Failed to retrieve enhanced security info for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            raise DataRetrievalError(error_msg)
    
    def _get_analyst_data(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """Get analyst recommendations and price targets.
        
        Args:
            symbol: Security ticker symbol
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary containing analyst data
        """
        cache_key = f"integration_analyst_data_{symbol}"
        data_type = "analyst_data"
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data
        
        self.logger.info(f"Retrieving analyst data for {symbol}")
        
        # Get prioritized adapters for this data type
        prioritized_adapters = self._get_prioritized_adapters("security_info")
        
        # Try each adapter in priority order
        analyst_data = {}
        
        for adapter in prioritized_adapters:
            try:
                # Get security info which might contain analyst data
                security_info = adapter.get_security_info(symbol)
                
                # Extract analyst data if available
                if 'analyst_rating' in security_info:
                    analyst_data['analyst_rating'] = security_info['analyst_rating']
                
                if 'price_target' in security_info:
                    analyst_data['price_target'] = security_info['price_target']
                
                if 'analyst_count' in security_info:
                    analyst_data['analyst_count'] = security_info['analyst_count']
                
                if 'buy_ratings' in security_info:
                    analyst_data['buy_ratings'] = security_info['buy_ratings']
                
                if 'hold_ratings' in security_info:
                    analyst_data['hold_ratings'] = security_info['hold_ratings']
                
                if 'sell_ratings' in security_info:
                    analyst_data['sell_ratings'] = security_info['sell_ratings']
                
                # If we got some analyst data, cache and return it
                if analyst_data:
                    if use_cache:
                        self.cache.set(
                            cache_key, 
                            analyst_data, 
                            ttl=self.cache_ttl.get("security_info"),
                            data_type=data_type,
                            tags=[symbol, adapter.name]
                        )
                    return analyst_data
                
            except Exception as e:
                self.logger.warning(f"Error retrieving analyst data from {adapter.name}: {str(e)}")
        
        # If no analyst data found, return empty dict
        return {}
        
    @with_error_recovery("get_company_news", retry_category="data_retrieval")
    def get_company_news(self, symbol: str, days: int = 7, limit: int = 10, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Get news for a specific company.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back
            limit: Maximum number of news items to return
            use_cache: Whether to use cached data if available
            
        Returns:
            List of news items as dictionaries
            
        Raises:
            DataRetrievalError: If data cannot be retrieved from any source
        """
        cache_key = f"integration_company_news_{symbol}_{days}_{limit}"
        data_type = "news"
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                self.logger.debug(f"Using cached company news for {symbol}")
                return cached_data
        
        self.logger.info(f"Retrieving company news for {symbol} (last {days} days, limit {limit})")
        
        # Get prioritized adapters for this data type
        prioritized_adapters = self._get_prioritized_adapters("news")
        
        # Try each adapter in priority order
        errors = []
        all_news = []
        
        for adapter in prioritized_adapters:
            try:
                self.logger.debug(f"Trying {adapter.name} for company news")
                
                # Check if adapter has get_company_news method
                if hasattr(adapter, 'get_company_news') and callable(getattr(adapter, 'get_company_news')):
                    news_items = adapter.get_company_news(symbol, days, limit)
                    
                    if news_items:
                        # Add source information
                        for item in news_items:
                            if 'source' not in item:
                                item['source'] = adapter.name
                        
                        all_news.extend(news_items)
                        
                        # If we have enough news items, stop
                        if len(all_news) >= limit:
                            break
                else:
                    self.logger.debug(f"Adapter {adapter.name} does not support company news")
                
            except Exception as e:
                self.logger.warning(f"Error retrieving company news from {adapter.name}: {str(e)}")
                errors.append(f"{adapter.name}: {str(e)}")
        
        # Sort by published date (newest first) and limit
        if all_news:
            # Sort by published_at if available
            all_news.sort(
                key=lambda x: x.get('published_at', datetime.now()), 
                reverse=True
            )
            
            # Limit the number of items
            all_news = all_news[:limit]
            
            # Cache the result
            if use_cache:
                self.cache.set(
                    cache_key, 
                    all_news, 
                    ttl=self.cache_ttl.get("news"),
                    data_type=data_type,
                    tags=[symbol, "news"]
                )
            
            return all_news
        
        # If we get here, all sources failed
        error_msg = f"Failed to retrieve company news for {symbol} from any source"
        if errors:
            error_msg += f": {'; '.join(errors)}"
            
        self.logger.error(error_msg)
        context = {'data_type': data_type}
        raise DataRetrievalError(error_msg, symbol=symbol, context=context)
        
    @with_error_recovery("get_market_news", retry_category="data_retrieval")
    def get_market_news(self, category: Optional[str] = None, days: int = 3, limit: int = 10, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Get general market news.
        
        Args:
            category: News category filter (e.g., 'market', 'economy', 'earnings')
            days: Number of days to look back
            limit: Maximum number of news items to return
            use_cache: Whether to use cached data if available
            
        Returns:
            List of news items as dictionaries
            
        Raises:
            DataRetrievalError: If data cannot be retrieved from any source
        """
        cache_key = f"integration_market_news_{category or 'all'}_{days}_{limit}"
        data_type = "news"
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                self.logger.debug(f"Using cached market news")
                return cached_data
        
        self.logger.info(f"Retrieving market news (category={category}, last {days} days, limit {limit})")
        
        # Get prioritized adapters for this data type
        prioritized_adapters = self._get_prioritized_adapters("news")
        
        # Try each adapter in priority order
        errors = []
        all_news = []
        
        for adapter in prioritized_adapters:
            try:
                self.logger.debug(f"Trying {adapter.name} for market news")
                
                # Check if adapter has get_market_news method
                if hasattr(adapter, 'get_market_news') and callable(getattr(adapter, 'get_market_news')):
                    news_items = adapter.get_market_news(category=category, days=days, limit=limit)
                    
                    if news_items:
                        # Add source information
                        for item in news_items:
                            if 'source' not in item:
                                item['source'] = adapter.name
                        
                        all_news.extend(news_items)
                        
                        # If we have enough news items, stop
                        if len(all_news) >= limit:
                            break
                else:
                    self.logger.debug(f"Adapter {adapter.name} does not support market news")
                
            except Exception as e:
                self.logger.warning(f"Error retrieving market news from {adapter.name}: {str(e)}")
                errors.append(f"{adapter.name}: {str(e)}")
        
        # Sort by published date (newest first) and limit
        if all_news:
            # Sort by published_at if available
            all_news.sort(
                key=lambda x: x.get('published_at', datetime.now()), 
                reverse=True
            )
            
            # Limit the number of items
            all_news = all_news[:limit]
            
            # Cache the result
            if use_cache:
                self.cache.set(
                    cache_key, 
                    all_news, 
                    ttl=self.cache_ttl.get("news"),
                    data_type=data_type,
                    tags=["market_news", category or "all"]
                )
            
            return all_news
        
        # If we get here, all sources failed
        error_msg = f"Failed to retrieve market news from any source"
        if errors:
            error_msg += f": {'; '.join(errors)}"
            
        self.logger.error(error_msg)
        context = {'data_type': data_type}
        raise DataRetrievalError(error_msg, context=context)
        
    @with_error_recovery("get_economic_calendar", retry_category="data_retrieval")
    def get_economic_calendar(self, days_ahead: int = 7, days_behind: int = 1, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Get economic calendar events.
        
        Args:
            days_ahead: Number of days to look ahead
            days_behind: Number of days to look behind
            use_cache: Whether to use cached data if available
            
        Returns:
            List of economic events as dictionaries
            
        Raises:
            DataRetrievalError: If data cannot be retrieved from any source
        """
        cache_key = f"integration_economic_calendar_{days_ahead}_{days_behind}"
        data_type = "economic_data"
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                self.logger.debug(f"Using cached economic calendar")
                return cached_data
        
        self.logger.info(f"Retrieving economic calendar (days_ahead={days_ahead}, days_behind={days_behind})")
        
        # Get prioritized adapters for this data type
        prioritized_adapters = self._get_prioritized_adapters("economic_data")
        
        # Try each adapter in priority order
        errors = []
        all_events = []
        
        for adapter in prioritized_adapters:
            try:
                self.logger.debug(f"Trying {adapter.name} for economic calendar")
                
                # Check if adapter has get_economic_calendar method
                if hasattr(adapter, 'get_economic_calendar') and callable(getattr(adapter, 'get_economic_calendar')):
                    events = adapter.get_economic_calendar(days_ahead=days_ahead, days_behind=days_behind)
                    
                    if events:
                        # Add source information
                        for event in events:
                            if 'source' not in event:
                                event['source'] = adapter.name
                        
                        all_events.extend(events)
                else:
                    self.logger.debug(f"Adapter {adapter.name} does not support economic calendar")
                
            except Exception as e:
                self.logger.warning(f"Error retrieving economic calendar from {adapter.name}: {str(e)}")
                errors.append(f"{adapter.name}: {str(e)}")
        
        # Sort by date (upcoming events first)
        if all_events:
            # Sort by date if available
            all_events.sort(
                key=lambda x: x.get('date', datetime.now()), 
                reverse=False
            )
            
            # Cache the result
            if use_cache:
                self.cache.set(
                    cache_key, 
                    all_events, 
                    ttl=self.cache_ttl.get("economic_data"),
                    data_type=data_type,
                    tags=["economic_calendar"]
                )
            
            return all_events
        
        # If we get here, all sources failed
        error_msg = f"Failed to retrieve economic calendar from any source"
        if errors:
            error_msg += f": {'; '.join(errors)}"
            
        self.logger.error(error_msg)
        context = {'data_type': data_type}
        raise DataRetrievalError(error_msg, context=context)
    
    def _validate_security_info(self, data: Dict[str, Any]) -> bool:
        """Validate security information data.
        
        Args:
            data: Security information data
            
        Returns:
            True if data is valid and complete, False otherwise
        """
        required_fields = ['symbol', 'name', 'current_price']
        optional_fields = ['market_cap', 'pe_ratio', 'sector', 'industry']
        
        # Check required fields
        for field in required_fields:
            if field not in data or data[field] is None:
                return False
        
        # Check if we have at least some optional fields
        optional_count = sum(1 for field in optional_fields if field in data and data[field] is not None)
        
        return optional_count >= 2  # At least 2 optional fields should be present
    
    def _combine_security_info(self, partial_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine partial security information from multiple sources.
        
        Args:
            partial_results: Dictionary mapping source names to partial results
            
        Returns:
            Combined security information
        """
        if not partial_results:
            return {}
        
        # Start with the result that has the most fields
        best_source = max(partial_results.keys(), key=lambda k: len(partial_results[k]))
        combined = partial_results[best_source].copy()
        
        # Fill in missing fields from other sources
        for source, result in partial_results.items():
            if source == best_source:
                continue
            
            for key, value in result.items():
                if key not in combined or combined[key] is None:
                    combined[key] = value
        
        # Add metadata about sources
        combined['_sources'] = list(partial_results.keys())
        
        return combined
    
    def get_historical_prices(
        self, 
        symbol: str, 
        period: str = "1y", 
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get historical price data from optimal sources.
        
        Args:
            symbol: Security ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with historical price data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved from any source
        """
        cache_key = f"integration_historical_{symbol}_{period}_{interval}"
        data_type = "historical_prices"
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Using cached historical data for {symbol}")
                return cached_data
        
        self.logger.info(f"Retrieving integrated historical data for {symbol} (period={period}, interval={interval})")
        
        # Get prioritized adapters for this data type
        prioritized_adapters = self._get_prioritized_adapters(data_type)
        
        # Try each adapter in priority order
        errors = []
        
        for adapter in prioritized_adapters:
            try:
                self.logger.debug(f"Trying {adapter.name} for historical prices")
                result = adapter.get_historical_prices(symbol, period, interval)
                
                if not result.empty:
                    # Validate and normalize data
                    result = self._normalize_historical_data(result)
                    
                    if use_cache:
                        self.cache.set(
                            cache_key, 
                            result, 
                            ttl=self.cache_ttl.get("historical_prices"),
                            data_type=data_type,
                            tags=[symbol, adapter.name]
                        )
                    
                    return result
                
            except Exception as e:
                self.logger.warning(f"Error retrieving historical data from {adapter.name}: {str(e)}")
                errors.append(f"{adapter.name}: {str(e)}")
        
        # If we get here, all sources failed
        error_msg = f"Failed to retrieve historical data for {symbol} from any source: {'; '.join(errors)}"
        self.logger.error(error_msg)
        raise DataRetrievalError(error_msg)
    
    def _normalize_historical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize historical price data.
        
        Args:
            df: Historical price DataFrame
            
        Returns:
            Normalized DataFrame
        """
        # Ensure standard column names
        standard_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Create a mapping of current columns to standard columns
        column_mapping = {}
        for std_col in standard_columns:
            for col in df.columns:
                if std_col.lower() == col.lower():
                    column_mapping[col] = std_col
                    break
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Ensure all standard columns exist
        for col in standard_columns:
            if col not in df.columns:
                df[col] = None
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                self.logger.warning(f"Failed to convert index to datetime: {str(e)}")
        
        # Sort by date
        df = df.sort_index()
        
        return df
    
    def get_financial_statements(
        self, 
        symbol: str, 
        statement_type: str = "income", 
        period: str = "annual",
        use_cache: bool = True,
        years: int = 5
    ) -> pd.DataFrame:
        """Get financial statements from optimal sources.
        
        Args:
            symbol: Security ticker symbol
            statement_type: Type of statement ('income', 'balance', 'cash')
            period: Period ('annual' or 'quarterly')
            use_cache: Whether to use cached data
            years: Number of years of historical data to retrieve
            
        Returns:
            DataFrame with financial statement data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved from any source
        """
        cache_key = f"integration_financials_{symbol}_{statement_type}_{period}"
        data_type = "financial_statements"
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Using cached financial statements for {symbol}")
                return cached_data
        
        self.logger.info(f"Retrieving integrated {period} {statement_type} statement for {symbol}")
        
        # Get prioritized adapters for this data type
        prioritized_adapters = self._get_prioritized_adapters(data_type)
        
        # Try each adapter in priority order
        errors = []
        partial_results = {}
        
        for adapter in prioritized_adapters:
            try:
                self.logger.debug(f"Trying {adapter.name} for financial statements")
                # Check if the adapter's method accepts the years parameter
                import inspect
                adapter_method = adapter.get_financial_statements
                adapter_params = inspect.signature(adapter_method).parameters
                
                if 'years' in adapter_params:
                    result = adapter.get_financial_statements(symbol, statement_type, period, years=years)
                else:
                    result = adapter.get_financial_statements(symbol, statement_type, period)
                
                if not result.empty:
                    # Normalize data before validation
                    result = self._normalize_financial_statements(result, statement_type)
                    
                    # Store as partial result
                    partial_results[adapter.name] = result
                
            except Exception as e:
                self.logger.warning(f"Error retrieving financial statements from {adapter.name}: {str(e)}")
                errors.append(f"{adapter.name}: {str(e)}")
        
        # If we have partial results, combine them
        if partial_results:
            combined_result = self._combine_financial_statements(partial_results, statement_type)
            
            if use_cache:
                self.cache.set(
                    cache_key, 
                    combined_result, 
                    ttl=self.cache_ttl.get("financial_statements"),
                    data_type=data_type,
                    tags=[symbol]
                )
            
            return combined_result
        
        # If we get here, all sources failed
        error_msg = f"Failed to retrieve {statement_type} statement for {symbol} from any source: {'; '.join(errors)}"
        self.logger.error(error_msg)
        raise DataRetrievalError(error_msg)
    
    def _validate_financial_statements(self, df: pd.DataFrame, statement_type: str) -> bool:
        """Validate financial statement data.
        
        Args:
            df: Financial statement DataFrame
            statement_type: Type of statement ('income', 'balance', 'cash')
            
        Returns:
            True if data is valid and complete, False otherwise
        """
        if df.empty:
            return False
        
        # Define required fields for each statement type
        required_fields = {
            'income': ['Revenue', 'NetIncome'],
            'balance': ['TotalAssets', 'TotalLiabilities'],
            'cash': ['OperatingCashFlow']
        }
        
        # Check if required fields are present
        for field in required_fields.get(statement_type, []):
            if field not in df.columns:
                return False
        
        # Check if we have at least some data
        if len(df) < 1:
            return False
        
        # Check if we have a reasonable number of columns
        min_columns = {
            'income': 5,
            'balance': 5,
            'cash': 3
        }
        
        if len(df.columns) < min_columns.get(statement_type, 3):
            return False
        
        return True
    
    def _normalize_financial_statements(self, df: pd.DataFrame, statement_type: str) -> pd.DataFrame:
        """Normalize financial statement data.
        
        Args:
            df: Financial statement DataFrame
            statement_type: Type of statement ('income', 'balance', 'cash')
            
        Returns:
            Normalized DataFrame
        """
        if df.empty:
            return df
            
        # Ensure index is datetime if possible
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                # First, directly replace problematic date formats in the index
                new_index = []
                for idx in df.index:
                    if isinstance(idx, str):
                        if '-00-' in idx:
                            # Replace YYYY-00-DD with YYYY-12-31
                            try:
                                year = idx.split('-')[0]
                                new_index.append(f"{year}-12-31")
                            except:
                                new_index.append(idx)
                        elif '-0-' in idx and len(idx.split('-')) >= 3:
                            # Replace YYYY-0-DD with YYYY-12-31
                            try:
                                year = idx.split('-')[0]
                                new_index.append(f"{year}-12-31")
                            except:
                                new_index.append(idx)
                        else:
                            new_index.append(idx)
                    else:
                        new_index.append(idx)
                
                # Use the cleaned index
                df.index = new_index
                
                # Now convert to datetime
                df.index = pd.to_datetime(df.index, errors='coerce')
                
                # Sort by date, most recent first
                df = df.sort_index(ascending=False)
            except Exception as e:
                logger.warning(f"Failed to convert index to datetime: {str(e)}")
                pass  # Keep original index if conversion fails
        
        # Convert numeric columns
        for col in df.columns:
            if col not in ['reportedCurrency', 'currency', 'unit']:
                # Make a copy to avoid pandas warnings about modifying a view
                df = df.copy()
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Standardize column names
        column_mapping = self._get_standard_column_mapping(statement_type)
        
        # Rename columns if they match standard mappings
        for old_name, new_name in column_mapping.items():
            for col in df.columns:
                # Convert column name to string if it's not already
                col_str = str(col) if not isinstance(col, str) else col
                if col_str.lower() == old_name.lower():
                    df = df.rename(columns={col: new_name})
                    break
        
        return df
    
    def _get_standard_column_mapping(self, statement_type: str) -> dict:
        """Get standard column name mapping for financial statements.
        
        Args:
            statement_type: Type of statement ('income', 'balance', 'cash')
            
        Returns:
            Dictionary mapping common column names to standard names
        """
        if statement_type == 'income':
            return {
                'totalRevenue': 'Revenue',
                'grossProfit': 'GrossProfit',
                'operatingIncome': 'OperatingIncome',
                'netIncome': 'NetIncome',
                'earningsPerShare': 'EPS',
                'dilutedEPS': 'DilutedEPS',
                'costOfRevenue': 'CostOfRevenue',
                'researchAndDevelopment': 'ResearchAndDevelopment',
                'sellingGeneralAndAdministrative': 'SGA',
                'operatingExpenses': 'OperatingExpenses',
                'interestExpense': 'InterestExpense',
                'incomeTaxExpense': 'IncomeTaxExpense',
                'ebitda': 'EBITDA'
            }
        elif statement_type == 'balance':
            return {
                'totalAssets': 'TotalAssets',
                'totalCurrentAssets': 'CurrentAssets',
                'cashAndCashEquivalents': 'CashAndCashEquivalents',
                'shortTermInvestments': 'ShortTermInvestments',
                'inventory': 'Inventory',
                'totalLiabilities': 'TotalLiabilities',
                'totalCurrentLiabilities': 'CurrentLiabilities',
                'longTermDebt': 'LongTermDebt',
                'totalShareholderEquity': 'TotalShareholderEquity',
                'retainedEarnings': 'RetainedEarnings',
                'commonStock': 'CommonStock',
                'accountsReceivable': 'AccountsReceivable',
                'accountsPayable': 'AccountsPayable'
            }
        elif statement_type == 'cash':
            return {
                'operatingCashflow': 'OperatingCashFlow',
                'cashflowFromOperations': 'OperatingCashFlow',
                'capitalExpenditures': 'CapitalExpenditures',
                'freeCashFlow': 'FreeCashFlow',
                'dividendsPaid': 'DividendsPaid',
                'netInvestingCashflow': 'NetInvestingCashFlow',
                'netFinancingCashflow': 'NetFinancingCashFlow',
                'netChangeInCash': 'NetChangeInCash',
                'repurchaseOfStock': 'ShareRepurchase',
                'debtRepayment': 'DebtRepayment'
            }
        else:
            return {}
    
    def _combine_financial_statements(self, partial_results: dict, statement_type: str) -> pd.DataFrame:
        """Combine financial statements from multiple sources.
        
        Args:
            partial_results: Dictionary mapping source names to partial results
            statement_type: Type of statement ('income', 'balance', 'cash')
            
        Returns:
            Combined financial statements DataFrame
        """
        # Implementation details...
        
    def get_financial_ratios(
        self, 
        symbol: str, 
        period: str = "annual",
        years: int = 5,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get financial ratios from optimal sources.
        
        Args:
            symbol: Security ticker symbol
            period: Period ('annual' or 'quarterly')
            years: Number of years of historical data to retrieve
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with financial ratios
            
        Raises:
            DataRetrievalError: If data cannot be retrieved from any source
        """
        cache_key = f"integration_ratios_{symbol}_{period}"
        data_type = "financial_ratios"
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Using cached financial ratios for {symbol}")
                return cached_data
        
        self.logger.info(f"Retrieving integrated {period} financial ratios for {symbol}")
        
        # Calculate ratios from financial statements
        try:
            # Get income statement
            income_statement = self.get_financial_statements(
                symbol, 
                statement_type='income', 
                period=period,
                years=years
            )
            
            # Get balance sheet
            balance_sheet = self.get_financial_statements(
                symbol, 
                statement_type='balance', 
                period=period,
                years=years
            )
            
            # Get cash flow statement
            cash_flow = self.get_financial_statements(
                symbol, 
                statement_type='cash', 
                period=period,
                years=years
            )
            
            # Get security info for current price
            security_info = self.get_security_info(symbol)
            current_price = security_info.get('current_price', 0)
            
            # Create empty DataFrame for ratios
            ratios = pd.DataFrame()
            
            # If we have all the necessary data, calculate ratios
            if not income_statement.empty and not balance_sheet.empty and not cash_flow.empty:
                # Get columns (dates) from income statement
                columns = income_statement.columns
                
                # Create empty DataFrame with same columns
                ratios = pd.DataFrame(index=[], columns=columns)
                
                # Calculate profitability ratios
                if 'Revenue' in income_statement.index and 'GrossProfit' in income_statement.index:
                    ratios.loc['Gross Margin'] = income_statement.loc['GrossProfit'] / income_statement.loc['Revenue']
                
                if 'Revenue' in income_statement.index and 'OperatingIncome' in income_statement.index:
                    ratios.loc['Operating Margin'] = income_statement.loc['OperatingIncome'] / income_statement.loc['Revenue']
                
                if 'Revenue' in income_statement.index and 'NetIncome' in income_statement.index:
                    ratios.loc['Net Profit Margin'] = income_statement.loc['NetIncome'] / income_statement.loc['Revenue']
                
                if 'NetIncome' in income_statement.index and 'TotalAssets' in balance_sheet.index:
                    ratios.loc['Return on Assets'] = income_statement.loc['NetIncome'] / balance_sheet.loc['TotalAssets']
                
                if 'NetIncome' in income_statement.index and 'TotalShareholderEquity' in balance_sheet.index:
                    ratios.loc['Return on Equity'] = income_statement.loc['NetIncome'] / balance_sheet.loc['TotalShareholderEquity']
                
                # Calculate liquidity ratios
                if 'CurrentAssets' in balance_sheet.index and 'CurrentLiabilities' in balance_sheet.index:
                    ratios.loc['Current Ratio'] = balance_sheet.loc['CurrentAssets'] / balance_sheet.loc['CurrentLiabilities']
                
                if 'cashAndCashEquivalentsAtCarryingValue' in balance_sheet.index and 'CurrentLiabilities' in balance_sheet.index:
                    ratios.loc['Cash Ratio'] = balance_sheet.loc['cashAndCashEquivalentsAtCarryingValue'] / balance_sheet.loc['CurrentLiabilities']
                
                # Calculate leverage ratios
                if 'TotalLiabilities' in balance_sheet.index and 'TotalShareholderEquity' in balance_sheet.index:
                    ratios.loc['Debt to Equity'] = balance_sheet.loc['TotalLiabilities'] / balance_sheet.loc['TotalShareholderEquity']
                
                if 'TotalLiabilities' in balance_sheet.index and 'TotalAssets' in balance_sheet.index:
                    ratios.loc['Debt to Assets'] = balance_sheet.loc['TotalLiabilities'] / balance_sheet.loc['TotalAssets']
                
                if 'OperatingIncome' in income_statement.index and 'InterestExpense' in income_statement.index:
                    ratios.loc['Interest Coverage'] = income_statement.loc['OperatingIncome'] / income_statement.loc['InterestExpense']
                
                # Calculate valuation ratios
                if current_price > 0 and 'NetIncome' in income_statement.index:
                    # Use the most recent values for P/E ratio
                    latest_net_income = income_statement.loc['NetIncome'].iloc[-1]
                    if latest_net_income > 0:
                        ratios.loc['P/E Ratio'] = current_price / latest_net_income
                
                # Cache the results
                if use_cache:
                    self.cache.set(
                        cache_key, 
                        ratios, 
                        ttl=self.cache_ttl.get("financial_statements"),
                        data_type=data_type,
                        tags=[symbol]
                    )
            
            return ratios
            
        except Exception as e:
            error_msg = f"Failed to calculate financial ratios for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            return pd.DataFrame()  # Return empty DataFrame on error
        
    def get_market_data(
        self, 
        data_type: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Get market data from optimal sources.
        
        Args:
            data_type: Type of market data ('indices', 'commodities', 'forex', 'sectors')
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary containing market data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved from any source
        """
        from stock_analysis.services.market_data_integration import validate_market_data, combine_market_data
        
        cache_key = f"integration_market_{data_type}"
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Using cached market data for {data_type}")
                return cached_data
        
        self.logger.info(f"Retrieving integrated market data: {data_type}")
        
        # Get prioritized adapters for this data type
        prioritized_adapters = self._get_prioritized_adapters("market_data")
        
        # Try each adapter in priority order
        errors = []
        partial_results = {}
        
        for adapter in prioritized_adapters:
            try:
                self.logger.debug(f"Trying {adapter.name} for market data ({data_type})")
                result = adapter.get_market_data(data_type)
                
                # If we got a complete result, validate it
                if result and validate_market_data(result, data_type):
                    if use_cache:
                        self.cache.set(
                            cache_key, 
                            result, 
                            ttl=self.cache_ttl.get("market_data"),
                            data_type="market_data",
                            tags=[data_type, adapter.name]
                        )
                    return result
                
                # Otherwise, store partial result
                if result:
                    partial_results[adapter.name] = result
                
            except Exception as e:
                self.logger.warning(f"Error retrieving market data from {adapter.name}: {str(e)}")
                errors.append(f"{adapter.name}: {str(e)}")
        
        # If we have partial results, combine them
        if partial_results:
            combined_result = combine_market_data(partial_results, data_type)
            
            if use_cache:
                self.cache.set(
                    cache_key, 
                    combined_result, 
                    ttl=self.cache_ttl.get("market_data"),
                    data_type="market_data",
                    tags=[data_type]
                )
            
            return combined_result
        
        # If we get here, all sources failed
        error_msg = f"Failed to retrieve market data ({data_type}) from any source: {'; '.join(errors)}"
        self.logger.error(error_msg)
        raise DataRetrievalError(error_msg)
        
    def _combine_financial_statements(self, partial_results: dict, statement_type: str) -> pd.DataFrame:
        """Combine financial statements from multiple sources.
        
        Args:
            partial_results: Dictionary mapping source names to partial results
            statement_type: Type of statement ('income', 'balance', 'cash')
            
        Returns:
            Combined financial statements DataFrame
        """
        if not partial_results:
            return pd.DataFrame()
        
        # Start with the most complete result
        best_source = max(partial_results.keys(), 
                         key=lambda k: len(partial_results[k].columns) * len(partial_results[k]))
        combined_df = partial_results[best_source].copy()
        
        # Store sources as metadata (using a dictionary attribute)
        combined_df.attrs['_sources'] = list(partial_results.keys())
        
        # If we only have one source, return it
        if len(partial_results) == 1:
            return combined_df
        
        # Combine data from other sources
        for source, df in partial_results.items():
            if source == best_source:
                continue
                
            # For each period in the other source
            for date_idx in df.index:
                if date_idx not in combined_df.index:
                    # Add new periods
                    combined_df.loc[date_idx] = df.loc[date_idx]
                else:
                    # Fill missing values in existing periods
                    for col in df.columns:
                        if col in combined_df.columns and pd.isna(combined_df.loc[date_idx, col]):
                            combined_df.loc[date_idx, col] = df.loc[date_idx, col]
                    
                    # Add new columns
                    for col in df.columns:
                        if col not in combined_df.columns:
                            combined_df[col] = None
                            combined_df.loc[date_idx, col] = df.loc[date_idx, col]
        
        # Sort by date (most recent first)
        combined_df = combined_df.sort_index(ascending=False)
        
        return combined_df
    
    def get_enhanced_financial_statements(
        self,
        symbol: str,
        statement_types: List[str] = None,
        period: str = "annual",
        years: int = 5,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Get enhanced financial statements with data from multiple sources.
        
        This method retrieves multiple financial statement types and combines them
        into a comprehensive financial data package.
        
        Args:
            symbol: Security ticker symbol
            statement_types: List of statement types to retrieve ('income', 'balance', 'cash')
            period: Period ('annual' or 'quarterly')
            years: Number of years of historical data to include
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping statement types to DataFrames
            
        Raises:
            DataRetrievalError: If data cannot be retrieved from any source
        """
        if statement_types is None:
            statement_types = ['income', 'balance', 'cash']
            
        cache_key = f"integration_enhanced_financials_{symbol}_{period}_{','.join(statement_types)}"
        data_type = "financial_statements"
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Using cached enhanced financial statements for {symbol}")
                return cached_data
        
        self.logger.info(f"Retrieving enhanced financial statements for {symbol}")
        
        result = {}
        errors = []
        
        # Retrieve each statement type
        for statement_type in statement_types:
            try:
                df = self.get_financial_statements(symbol, statement_type, period, use_cache)
                
                # Limit to requested number of years if needed
                if years > 0 and len(df) > years:
                    df = df.iloc[:years]
                    
                result[statement_type] = df
                
            except Exception as e:
                self.logger.warning(f"Error retrieving {statement_type} statement for {symbol}: {str(e)}")
                errors.append(f"{statement_type}: {str(e)}")
        
        # Calculate additional metrics if we have the necessary data
        if 'income' in result and 'balance' in result:
            try:
                result['metrics'] = self._calculate_financial_metrics(result['income'], result['balance'])
            except Exception as e:
                self.logger.warning(f"Error calculating financial metrics for {symbol}: {str(e)}")
        
        # Cache the result
        if use_cache and result:
            self.cache.set(
                cache_key, 
                result, 
                ttl=self.cache_ttl.get("financial_statements"),
                data_type=data_type,
                tags=[symbol, 'enhanced']
            )
        
        # If we couldn't retrieve any statements, raise an error
        if not result:
            error_msg = f"Failed to retrieve any financial statements for {symbol}: {'; '.join(errors)}"
            self.logger.error(error_msg)
            raise DataRetrievalError(error_msg)
        
        return result
    
    def _calculate_financial_metrics(self, income_df: pd.DataFrame, balance_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional financial metrics from income and balance sheet data.
        
        Args:
            income_df: Income statement DataFrame
            balance_df: Balance sheet DataFrame
            
        Returns:
            DataFrame with calculated financial metrics
        """
        # Create a new DataFrame for metrics
        metrics = pd.DataFrame(index=income_df.index)
        
        # Calculate profitability ratios
        if 'GrossProfit' in income_df.columns and 'Revenue' in income_df.columns:
            metrics['GrossMargin'] = income_df['GrossProfit'] / income_df['Revenue']
            
        if 'NetIncome' in income_df.columns and 'Revenue' in income_df.columns:
            metrics['NetMargin'] = income_df['NetIncome'] / income_df['Revenue']
            
        if 'OperatingIncome' in income_df.columns and 'Revenue' in income_df.columns:
            metrics['OperatingMargin'] = income_df['OperatingIncome'] / income_df['Revenue']
        
        # Calculate return ratios
        if 'NetIncome' in income_df.columns and 'TotalAssets' in balance_df.columns:
            # Align balance sheet with income statement periods
            aligned_assets = balance_df.reindex(income_df.index, method='nearest')
            metrics['ROA'] = income_df['NetIncome'] / aligned_assets['TotalAssets']
            
        if 'NetIncome' in income_df.columns and 'TotalShareholderEquity' in balance_df.columns:
            # Align balance sheet with income statement periods
            aligned_equity = balance_df.reindex(income_df.index, method='nearest')
            metrics['ROE'] = income_df['NetIncome'] / aligned_equity['TotalShareholderEquity']
        
        # Calculate liquidity ratios
        if 'CurrentAssets' in balance_df.columns and 'CurrentLiabilities' in balance_df.columns:
            metrics['CurrentRatio'] = balance_df['CurrentAssets'] / balance_df['CurrentLiabilities']
            
        # Calculate debt ratios
        if 'TotalLiabilities' in balance_df.columns and 'TotalAssets' in balance_df.columns:
            metrics['DebtToAssets'] = balance_df['TotalLiabilities'] / balance_df['TotalAssets']
            
        if 'TotalLiabilities' in balance_df.columns and 'TotalShareholderEquity' in balance_df.columns:
            metrics['DebtToEquity'] = balance_df['TotalLiabilities'] / balance_df['TotalShareholderEquity']
        
        return metrics
    
    def get_technical_indicators(
        self, 
        symbol: str, 
        indicators: List[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Get technical indicators from optimal sources.
        
        Args:
            symbol: Security ticker symbol
            indicators: List of technical indicators to retrieve
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary containing technical indicator values
            
        Raises:
            DataRetrievalError: If data cannot be retrieved from any source
        """
        if indicators is None:
            indicators = ['sma_20', 'sma_50', 'rsi_14', 'macd']
        
        cache_key = f"integration_technicals_{symbol}_{','.join(sorted(indicators))}"
        data_type = "technical_indicators"
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Using cached technical indicators for {symbol}")
                return cached_data
        
        self.logger.info(f"Retrieving integrated technical indicators for {symbol}: {indicators}")
        
        # Get prioritized adapters for this data type
        prioritized_adapters = self._get_prioritized_adapters(data_type)
        
        # Try to get indicators from multiple sources in parallel
        results = {}
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_adapter = {
                executor.submit(adapter.get_technical_indicators, symbol, indicators): adapter
                for adapter in prioritized_adapters
            }
            
            for future in as_completed(future_to_adapter):
                adapter = future_to_adapter[future]
                try:
                    result = future.result()
                    if result:
                        # Normalize the data
                        normalized_result = self._normalize_technical_indicators(result)
                        
                        # Validate the data
                        if self._validate_technical_indicators(normalized_result, indicators):
                            # If we have a complete result from a single source, use it
                            if use_cache:
                                self.cache.set(
                                    cache_key, 
                                    normalized_result, 
                                    ttl=self.cache_ttl.get("technical_indicators"),
                                    data_type=data_type,
                                    tags=[symbol, adapter.name]
                                )
                            return normalized_result
                        
                        # Otherwise, store for later combination
                        results[adapter.name] = result
                except Exception as e:
                    self.logger.warning(f"Error retrieving technical indicators from {adapter.name}: {str(e)}")
                    errors.append(f"{adapter.name}: {str(e)}")
        
        # Combine results from different sources
        if results:
            combined_result = self._combine_technical_indicators(results, indicators)
            
            if use_cache:
                self.cache.set(
                    cache_key, 
                    combined_result, 
                    ttl=self.cache_ttl.get("technical_indicators"),
                    data_type=data_type,
                    tags=[symbol]
                )
            
            return combined_result
        
        # If we get here, all sources failed
        error_msg = f"Failed to retrieve technical indicators for {symbol} from any source: {'; '.join(errors)}"
        self.logger.error(error_msg)
        raise DataRetrievalError(error_msg)
        self.logger.error(error_msg)
        raise DataRetrievalError(error_msg)
    
    def _validate_technical_indicators(self, data: Dict[str, Any], requested_indicators: List[str]) -> bool:
        """Validate technical indicator data.
        
        Args:
            data: Technical indicator data
            requested_indicators: List of requested indicator names
            
        Returns:
            True if data is valid and complete, False otherwise
        """
        if not data:
            return False
        
        # Check if all requested indicators are present and have valid values
        for indicator in requested_indicators:
            if indicator not in data or data[indicator] is None:
                return False
            
            # For complex indicators like MACD, check if they have the required components
            if indicator == 'macd' and isinstance(data[indicator], dict):
                if not all(key in data[indicator] for key in ['macd', 'signal', 'histogram']):
                    return False
        
        return True
    
    def _normalize_technical_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize technical indicator data.
        
        Args:
            data: Technical indicator data
            
        Returns:
            Normalized technical indicator data
        """
        if not data:
            return {}
        
        normalized = {}
        
        # Standard mappings for indicator names
        name_mappings = {
            'SMA20': 'sma_20',
            'SMA50': 'sma_50',
            'SMA200': 'sma_200',
            'EMA20': 'ema_20',
            'EMA50': 'ema_50',
            'RSI': 'rsi_14',
            'RSI14': 'rsi_14',
            'MACD': 'macd'
        }
        
        # Process each indicator
        for key, value in data.items():
            # Normalize indicator name
            normalized_key = key.lower()
            
            # Apply mappings if available
            if key in name_mappings:
                normalized_key = name_mappings[key]
            elif key.startswith('SMA'):
                # Handle SMA with different period formats
                try:
                    period = int(key[3:])
                    normalized_key = f'sma_{period}'
                except ValueError:
                    pass
            elif key.startswith('EMA'):
                # Handle EMA with different period formats
                try:
                    period = int(key[3:])
                    normalized_key = f'ema_{period}'
                except ValueError:
                    pass
            elif key.startswith('RSI'):
                # Handle RSI with different period formats
                try:
                    period = int(key[3:])
                    normalized_key = f'rsi_{period}'
                except ValueError:
                    normalized_key = 'rsi_14'  # Default to RSI-14 if period not specified
            
            # Normalize MACD values
            if normalized_key == 'macd' and isinstance(value, dict):
                macd_dict = {}
                
                # Map MACD component names
                macd_mappings = {
                    'macd_line': 'macd',
                    'macdLine': 'macd',
                    'signal_line': 'signal',
                    'signalLine': 'signal',
                    'histogram': 'histogram',
                    'hist': 'histogram'
                }
                
                for macd_key, macd_value in value.items():
                    normalized_macd_key = macd_key
                    if macd_key in macd_mappings:
                        normalized_macd_key = macd_mappings[macd_key]
                    macd_dict[normalized_macd_key] = macd_value
                
                # Ensure all required components exist
                if 'macd' not in macd_dict:
                    macd_dict['macd'] = None
                if 'signal' not in macd_dict:
                    macd_dict['signal'] = None
                if 'histogram' not in macd_dict:
                    macd_dict['histogram'] = None
                
                normalized[normalized_key] = macd_dict
            else:
                normalized[normalized_key] = value
        
        return normalized
    
    def _combine_technical_indicators(
        self, 
        results: Dict[str, Dict[str, Any]], 
        requested_indicators: List[str]
    ) -> Dict[str, Any]:
        """Combine technical indicators from multiple sources.
        
        Args:
            results: Dictionary mapping source names to indicator results
            requested_indicators: List of requested indicator names
            
        Returns:
            Combined technical indicators
        """
        combined = {}
        sources = {}
        
        # Normalize data from each source
        normalized_results = {}
        for source, result in results.items():
            normalized_results[source] = self._normalize_technical_indicators(result)
        
        # For each requested indicator, find the best source
        for indicator in requested_indicators:
            for source, result in normalized_results.items():
                if indicator in result and result[indicator] is not None:
                    # Use priority order to determine which source to use
                    if indicator not in combined or self.source_priorities.get("technical_indicators", {}).get(source, 999) < self.source_priorities.get("technical_indicators", {}).get(sources.get(indicator, ""), 999):
                        combined[indicator] = result[indicator]
                        sources[indicator] = source
        
        # Add metadata about sources
        combined['_sources'] = sources
        
        return combined
    
    def get_market_data(
        self, 
        data_type: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Get market data from optimal sources.
        
        Args:
            data_type: Type of market data ('indices', 'commodities', 'forex', 'sectors')
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary containing market data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved from any source
        """
        cache_key = f"integration_market_{data_type}"
        data_category = "market_data"
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Using cached market data for {data_type}")
                return cached_data
        
        self.logger.info(f"Retrieving integrated market data: {data_type}")
        
        # Get prioritized adapters for this data type
        prioritized_adapters = self._get_prioritized_adapters(data_category)
        
        # Try to get data from multiple sources in parallel
        results = {}
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_adapter = {
                executor.submit(adapter.get_market_data, data_type): adapter
                for adapter in prioritized_adapters
            }
            
            for future in as_completed(future_to_adapter):
                adapter = future_to_adapter[future]
                try:
                    result = future.result()
                    if result:
                        results[adapter.name] = result
                except Exception as e:
                    self.logger.warning(f"Error retrieving market data from {adapter.name}: {str(e)}")
                    errors.append(f"{adapter.name}: {str(e)}")
        
        # Combine results from different sources
        if results:
            from stock_analysis.services.market_data_integration import combine_market_data
            combined_result = combine_market_data(results, data_type)
            
            if use_cache:
                self.cache.set(
                    cache_key, 
                    combined_result, 
                    ttl=self.cache_ttl.get("market_data"),
                    data_type="market_data",
                    tags=[data_type]
                )
            
            return combined_result
        
        # If we get here, all sources failed
        error_msg = f"Failed to retrieve market data ({data_type}) from any source: {'; '.join(errors)}"
        self.logger.error(error_msg)
        raise DataRetrievalError(error_msg)
    
    def _validate_market_data(self, data: Dict[str, Any], data_type: str) -> bool:
        """Validate market data.
        
        Args:
            data: Market data dictionary
            data_type: Type of market data ('indices', 'commodities', 'forex', 'sectors')
            
        Returns:
            True if data is valid and complete, False otherwise
        """
        if not data:
            return False
        
        # Define minimum expected items for each data type
        min_items = {
            'indices': 2,  # At least 2 major indices
            'commodities': 2,  # At least 2 commodities
            'forex': 2,  # At least 2 currency pairs
            'sectors': 5,  # At least 5 sectors
        }
        
        # Check if we have the minimum number of items
        if len(data) < min_items.get(data_type, 1):
            return False
        
        # Check if data items have the required fields
        required_fields = ['price']
        optional_fields = ['change', 'change_percent', 'name', 'symbol']
        
        # Check a sample of items (up to 3)
        sample_keys = list(data.keys())[:3]
        for key in sample_keys:
            item = data[key]
            
            # Skip metadata
            if key == '_sources':
                continue
                
            # Check required fields
            for field in required_fields:
                if field not in item or item[field] is None:
                    return False
            
            # Check if we have at least some optional fields
            optional_count = sum(1 for field in optional_fields if field in item and item[field] is not None)
            if optional_count < 1:  # At least 1 optional field should be present
                return False
        
        return True
    
    def _combine_market_data(
        self, 
        results: Dict[str, Dict[str, Any]], 
        data_type: str
    ) -> Dict[str, Any]:
        """Combine market data from multiple sources.
        
        Args:
            results: Dictionary mapping source names to market data results
            data_type: Type of market data
            
        Returns:
            Combined market data
        """
        # Start with the most complete result
        best_source = max(results.keys(), key=lambda k: len(results[k]))
        combined = results[best_source].copy()
        
        # Add data from other sources for missing items
        for source, result in results.items():
            if source == best_source:
                continue
            
            for key, value in result.items():
                if key not in combined:
                    combined[key] = value
        
        # Add metadata about sources
        combined['_sources'] = list(results.keys())
        
        return combined
    
    def get_news(
        self, 
        symbol: str = None, 
        category: str = None, 
        limit: int = 10,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get news articles from optimal sources.
        
        Args:
            symbol: Security ticker symbol (optional)
            category: News category (optional)
            limit: Maximum number of articles to retrieve
            use_cache: Whether to use cached data
            
        Returns:
            List of news articles
            
        Raises:
            DataRetrievalError: If data cannot be retrieved from any source
        """
        cache_key = f"integration_news_{symbol or 'general'}_{category or 'all'}"
        data_type = "news"
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Using cached news for {symbol or 'general'}")
                return cached_data[:limit]  # Respect limit even for cached data
        
        self.logger.info(f"Retrieving integrated news for {symbol or 'general'}")
        
        # Get prioritized adapters for this data type
        prioritized_adapters = self._get_prioritized_adapters(data_type)
        
        # Try to get news from multiple sources in parallel
        results = {}
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_adapter = {
                executor.submit(adapter.get_news, symbol, category, limit): adapter
                for adapter in prioritized_adapters
            }
            
            for future in as_completed(future_to_adapter):
                adapter = future_to_adapter[future]
                try:
                    result = future.result()
                    if result:
                        results[adapter.name] = result
                except Exception as e:
                    self.logger.warning(f"Error retrieving news from {adapter.name}: {str(e)}")
                    errors.append(f"{adapter.name}: {str(e)}")
        
        # Combine results from different sources
        if results:
            combined_result = self._combine_news(results, limit)
            
            if use_cache:
                self.cache.set(
                    cache_key, 
                    combined_result, 
                    ttl=self.cache_ttl.get("news"),
                    data_type=data_type,
                    tags=[symbol or 'general', category or 'all']
                )
            
            return combined_result
        
        # If we get here, all sources failed
        error_msg = f"Failed to retrieve news for {symbol or 'general'} from any source: {'; '.join(errors)}"
        self.logger.error(error_msg)
        raise DataRetrievalError(error_msg)
    
    def _combine_news(
        self, 
        results: Dict[str, List[Dict[str, Any]]], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Combine news articles from multiple sources.
        
        Args:
            results: Dictionary mapping source names to news article lists
            limit: Maximum number of articles to include
            
        Returns:
            Combined list of news articles
        """
        all_articles = []
        
        # Add source information to each article
        for source, articles in results.items():
            for article in articles:
                article['_source'] = source
                all_articles.append(article)
        
        # Sort by publication date if available
        try:
            all_articles.sort(key=lambda x: x.get('published_at', ''), reverse=True)
        except Exception:
            pass
        
        # Remove duplicates (based on title similarity)
        unique_articles = []
        titles = set()
        
        for article in all_articles:
            title = article.get('title', '').lower()
            # Check if we already have a similar title
            if not any(self._is_similar_title(title, existing) for existing in titles):
                titles.add(title)
                unique_articles.append(article)
        
        return unique_articles[:limit]
    
    def _is_similar_title(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """Check if two titles are similar.
        
        Args:
            title1: First title
            title2: Second title
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if titles are similar, False otherwise
        """
        # Simple similarity check based on common words
        if not title1 or not title2:
            return False
        
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return False
        
        common_words = words1.intersection(words2)
        similarity = len(common_words) / max(len(words1), len(words2))
        
        return similarity >= threshold
    
    def get_economic_data(
        self, 
        indicator: str, 
        region: str = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get economic indicator data from optimal sources.
        
        Args:
            indicator: Economic indicator name
            region: Geographic region (optional)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with economic indicator data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved from any source
        """
        cache_key = f"integration_economic_{indicator}_{region or 'all'}"
        data_type = "economic_data"
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Using cached economic data for {indicator}")
                return cached_data
        
        self.logger.info(f"Retrieving integrated economic data: {indicator} ({region or 'all regions'})")
        
        # Get prioritized adapters for this data type
        prioritized_adapters = self._get_prioritized_adapters(data_type)
        
        # Try each adapter in priority order
        errors = []
        
        for adapter in prioritized_adapters:
            try:
                self.logger.debug(f"Trying {adapter.name} for economic data")
                result = adapter.get_economic_data(indicator, region)
                
                if not result.empty:
                    if use_cache:
                        self.cache.set(
                            cache_key, 
                            result, 
                            ttl=self.cache_ttl.get("economic_data"),
                            data_type=data_type,
                            tags=[indicator, region or 'all']
                        )
                    
                    return result
                
            except Exception as e:
                self.logger.warning(f"Error retrieving economic data from {adapter.name}: {str(e)}")
                errors.append(f"{adapter.name}: {str(e)}")
        
        # If we get here, all sources failed
        error_msg = f"Failed to retrieve economic data ({indicator}) from any source: {'; '.join(errors)}"
        self.logger.error(error_msg)
        raise DataRetrievalError(error_msg)
    
    def get_enhanced_financial_data(
        self, 
        symbol: str, 
        data_types: List[str]
    ) -> Dict[str, Any]:
        """Get multiple types of financial data in a single request.
        
        Args:
            symbol: Security ticker symbol
            data_types: List of data types to retrieve ('security_info', 'historical_prices', etc.)
            
        Returns:
            Dictionary containing requested financial data
        """
        self.logger.info(f"Retrieving enhanced financial data for {symbol}: {data_types}")
        
        result = {}
        errors = {}
        
        # Process each data type in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            # Submit tasks for each data type
            for data_type in data_types:
                if data_type == 'security_info':
                    futures[executor.submit(self.get_security_info, symbol)] = data_type
                elif data_type == 'historical_prices':
                    futures[executor.submit(self.get_historical_prices, symbol)] = data_type
                elif data_type == 'financial_statements':
                    # For financial statements, we need to get all types
                    for statement_type in ['income', 'balance', 'cash']:
                        futures[executor.submit(self.get_financial_statements, symbol, statement_type)] = f"{data_type}_{statement_type}"
                elif data_type == 'technical_indicators':
                    futures[executor.submit(self.get_technical_indicators, symbol)] = data_type
                elif data_type == 'news':
                    futures[executor.submit(self.get_news, symbol)] = data_type
            
            # Process results as they complete
            for future in as_completed(futures):
                data_type = futures[future]
                try:
                    data = future.result()
                    result[data_type] = data
                except Exception as e:
                    self.logger.warning(f"Error retrieving {data_type} for {symbol}: {str(e)}")
                    errors[data_type] = str(e)
        
        # Add error information
        if errors:
            result['_errors'] = errors
        
        return result
    
    def _get_prioritized_adapters(self, data_type: str) -> List[FinancialDataAdapter]:
        """Get adapters prioritized for a specific data type.
        
        Args:
            data_type: Type of data
            
        Returns:
            List of adapters sorted by priority
        """
        # Get priority map for this data type
        priority_map = self.source_priorities.get(data_type, {})
        
        # Sort adapters by priority
        return sorted(
            self.adapters,
            key=lambda a: priority_map.get(a.name, 999)
        )
    
    def invalidate_cache(self, pattern: str = None, data_type: str = None, tag: str = None) -> int:
        """Invalidate cache entries.
        
        Args:
            pattern: Pattern to match against keys
            data_type: Data type to invalidate
            tag: Tag to invalidate
            
        Returns:
            Number of entries invalidated
        """
        if pattern:
            return self.cache.invalidate_by_pattern(pattern)
        elif data_type:
            return self.cache.invalidate_by_type(data_type)
        elif tag:
            return self.cache.invalidate_by_tag(tag)
        else:
            self.cache.clear()
            return -1  # Unknown count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_stats()    
def get_batch_security_info(self, symbols: List[str], use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
        """Get security information for multiple symbols in parallel.
        
        Args:
            symbols: List of security ticker symbols
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping symbols to security information
            
        Raises:
            DataRetrievalError: If data cannot be retrieved for any symbol
        """
        if not symbols:
            return {}
        
        # Skip parallel processing if disabled or only one symbol
        if not self.parallel_enabled or len(symbols) == 1:
            return {symbol: self.get_security_info(symbol, use_cache) for symbol in symbols}
        
        from stock_analysis.utils.performance_metrics import PerformanceMonitor
        
        with PerformanceMonitor(self.metrics_collector, "batch_security_info", 
                               {"batch_size": len(symbols)}):
            
            # Define function to get security info for a single symbol
            def get_symbol_info(symbol: str) -> Tuple[str, Dict[str, Any]]:
                try:
                    return symbol, self.get_security_info(symbol, use_cache)
                except Exception as e:
                    self.logger.warning(f"Error retrieving security info for {symbol}: {str(e)}")
                    return symbol, {"error": str(e)}
            
            # Process symbols in parallel
            results = self.parallel_executor.map(
                get_symbol_info,
                symbols,
                timeout=30,
                operation_name="batch_security_info"
            )
            
            # Convert results to dictionary
            return {symbol: info for symbol, info in results}
    
def get_batch_historical_prices(
        self, 
        symbols: List[str], 
        period: str = "1y", 
        interval: str = "1d",
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Get historical price data for multiple symbols in parallel.
        
        Args:
            symbols: List of security ticker symbols
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping symbols to historical price DataFrames
            
        Raises:
            DataRetrievalError: If data cannot be retrieved for any symbol
        """
        if not symbols:
            return {}
        
        # Skip parallel processing if disabled or only one symbol
        if not self.parallel_enabled or len(symbols) == 1:
            return {symbol: self.get_historical_prices(symbol, period, interval, use_cache) 
                   for symbol in symbols}
        
        from stock_analysis.utils.performance_metrics import PerformanceMonitor
        
        with PerformanceMonitor(self.metrics_collector, "batch_historical_prices", 
                               {"batch_size": len(symbols), "period": period, "interval": interval}):
            
            # Define function to get historical prices for a single symbol
            def get_symbol_history(symbol: str) -> Tuple[str, pd.DataFrame]:
                try:
                    return symbol, self.get_historical_prices(symbol, period, interval, use_cache)
                except Exception as e:
                    self.logger.warning(f"Error retrieving historical data for {symbol}: {str(e)}")
                    return symbol, pd.DataFrame()
            
            # Process symbols in parallel
            results = self.parallel_executor.map(
                get_symbol_history,
                symbols,
                timeout=30,
                operation_name="batch_historical_prices"
            )
            
            # Convert results to dictionary
            return {symbol: df for symbol, df in results if not df.empty}
    
def get_batch_financial_statements(
        self, 
        symbols: List[str], 
        statement_type: str = "income", 
        period: str = "annual",
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Get financial statements for multiple symbols in parallel.
        
        Args:
            symbols: List of security ticker symbols
            statement_type: Type of statement ('income', 'balance', 'cash')
            period: Period ('annual' or 'quarterly')
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping symbols to financial statement DataFrames
            
        Raises:
            DataRetrievalError: If data cannot be retrieved for any symbol
        """
        if not symbols:
            return {}
        
        # Skip parallel processing if disabled or only one symbol
        if not self.parallel_enabled or len(symbols) == 1:
            return {symbol: self.get_financial_statements(symbol, statement_type, period, use_cache) 
                   for symbol in symbols}
        
        from stock_analysis.utils.performance_metrics import PerformanceMonitor
        
        with PerformanceMonitor(self.metrics_collector, "batch_financial_statements", 
                               {"batch_size": len(symbols), "statement_type": statement_type, "period": period}):
            
            # Define function to get financial statements for a single symbol
            def get_symbol_financials(symbol: str) -> Tuple[str, pd.DataFrame]:
                try:
                    return symbol, self.get_financial_statements(symbol, statement_type, period, use_cache)
                except Exception as e:
                    self.logger.warning(f"Error retrieving {statement_type} statement for {symbol}: {str(e)}")
                    return symbol, pd.DataFrame()
            
            # Process symbols in parallel
            results = self.parallel_executor.map(
                get_symbol_financials,
                symbols,
                timeout=60,  # Financial statements may take longer
                operation_name="batch_financial_statements"
            )
            
            # Convert results to dictionary
            return {symbol: df for symbol, df in results if not df.empty}
    
def get_batch_technical_indicators(
        self, 
        symbols: List[str], 
        indicators: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Get technical indicators for multiple symbols in parallel.
        
        Args:
            symbols: List of security ticker symbols
            indicators: List of technical indicators to retrieve
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping symbols to technical indicators
            
        Raises:
            DataRetrievalError: If data cannot be retrieved for any symbol
        """
        if not symbols:
            return {}
        
        # Skip parallel processing if disabled or only one symbol
        if not self.parallel_enabled or len(symbols) == 1:
            return {symbol: self.get_technical_indicators(symbol, indicators, use_cache) 
                   for symbol in symbols}
        
        from stock_analysis.utils.performance_metrics import PerformanceMonitor
        
        with PerformanceMonitor(self.metrics_collector, "batch_technical_indicators", 
                               {"batch_size": len(symbols)}):
            
            # Define function to get technical indicators for a single symbol
            def get_symbol_indicators(symbol: str) -> Tuple[str, Dict[str, Any]]:
                try:
                    return symbol, self.get_technical_indicators(symbol, indicators, use_cache)
                except Exception as e:
                    self.logger.warning(f"Error retrieving technical indicators for {symbol}: {str(e)}")
                    return symbol, {}
            
            # Process symbols in parallel
            results = self.parallel_executor.map(
                get_symbol_indicators,
                symbols,
                timeout=30,
                operation_name="batch_technical_indicators"
            )
            
            # Convert results to dictionary
            return {symbol: indicators for symbol, indicators in results if indicators}