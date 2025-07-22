"""Stock data service for retrieving financial data using yfinance.

This module provides a service for retrieving stock data from Yahoo Finance
using the yfinance library, with retry logic, rate limiting, and error handling.
"""

import time
import random
import logging
from typing import Dict, List, Optional, Union, Any, Type
import pandas as pd
import yfinance as yf

from stock_analysis.models.data_models import SecurityInfo, StockInfo, ETFInfo
from stock_analysis.utils.exceptions import DataRetrievalError, NetworkError, RateLimitError
from stock_analysis.utils.config import config
from stock_analysis.utils.logging import get_logger, log_api_call, log_data_quality_issue
from stock_analysis.utils.error_recovery import get_error_recovery_manager, with_error_recovery
from stock_analysis.utils.performance_metrics import get_metrics_collector, monitor_performance
from stock_analysis.utils.cache_manager import get_cache_manager

logger = get_logger(__name__)


class StockDataService:
    """Service for retrieving stock data from Yahoo Finance."""
    
    def __init__(self):
        """Initialize the stock data service with configuration."""
        self.timeout = config.get('stock_analysis.data_sources.yfinance.timeout', 30)
        self.retry_attempts = config.get('stock_analysis.data_sources.yfinance.retry_attempts', 3)
        self.rate_limit_delay = config.get('stock_analysis.data_sources.yfinance.rate_limit_delay', 1.0)
        self.last_request_time = 0
    
    def _is_etf(self, info: dict) -> bool:
        """Determine if a security is an ETF based on its info.
        
        Args:
            info: Security info dictionary from yfinance
            
        Returns:
            bool: True if the security is an ETF, False otherwise
        """
        # Check various indicators that suggest this is an ETF
        etf_indicators = [
            info.get('quoteType', '').upper() == 'ETF',
            info.get('fundFamily') is not None,
            info.get('etf', False) is True,
            'expense ratio' in str(info.get('longBusinessSummary', '')).lower(),
            'fund' in str(info.get('longName', '')).lower()
        ]
        return any(etf_indicators)
    
    def _extract_etf_info(self, symbol: str, info: dict) -> ETFInfo:
        """Extract ETF information from yfinance data.
        
        Args:
            symbol: ETF symbol
            info: Raw info dictionary from yfinance
            
        Returns:
            ETFInfo object with ETF data
        """
        # Extract asset allocation if available
        asset_allocation = {}
        for key, value in info.items():
            if key.endswith('Position') and isinstance(value, (int, float)):
                asset_type = key.replace('Position', '').lower()
                asset_allocation[asset_type] = value / 100  # Convert percentage to decimal
        
        # Extract holdings if available
        holdings = []
        if info.get('holdings'):
            for holding in info['holdings']:
                holdings.append({
                    'symbol': holding.get('symbol', ''),
                    'name': holding.get('holdingName', ''),
                    'weight': holding.get('holdingPercent', 0.0) / 100  # Convert to decimal
                })
        
        return ETFInfo(
            symbol=symbol,
            name=info.get('longName', info.get('shortName', symbol)),
            current_price=info.get('regularMarketPrice', info.get('currentPrice', 0.0)),
            market_cap=info.get('marketCap'),  # Can be None for ETFs
            beta=info.get('beta'),
            expense_ratio=info.get('annualReportExpenseRatio'),
            assets_under_management=info.get('totalAssets'),
            nav=info.get('navPrice'),
            category=info.get('category'),
            asset_allocation=asset_allocation if asset_allocation else None,
            holdings=holdings if holdings else None,
            dividend_yield=info.get('dividendYield')
        )
    
    def _extract_stock_info(self, symbol: str, info: dict) -> StockInfo:
        """Extract stock information from yfinance data.
        
        Args:
            symbol: Stock symbol
            info: Raw info dictionary from yfinance
            
        Returns:
            StockInfo object with stock data
        """
        # Create StockInfo with required fields first
        stock_info = StockInfo(
            company_name=info.get('longName', info.get('shortName', symbol)),
            symbol=symbol,
            name=info.get('longName', info.get('shortName', symbol)),
            current_price=info.get('regularMarketPrice', info.get('currentPrice', 0.0)),
            market_cap=info.get('marketCap', 0.0),
            beta=info.get('beta'),
            pe_ratio=info.get('trailingPE'),
            pb_ratio=info.get('priceToBook'),
            dividend_yield=info.get('dividendYield'),
            sector=info.get('sector'),
            industry=info.get('industry')
        )
        
        return stock_info
    
    @monitor_performance('get_security_info')
    def get_security_info(self, symbol: str, use_cache: bool = True) -> SecurityInfo:
        """Get security (stock or ETF) information.
        
        Args:
            symbol: Security ticker symbol
            use_cache: Whether to use cached data if available
            
        Returns:
            SecurityInfo object (either StockInfo or ETFInfo) with security information
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.set_context(symbol=symbol, operation='get_security_info')
        logger.info(f"Retrieving security info for {symbol}")
        
        # Generate cache key
        cache_key = f"security_info:{symbol}"
        cache = get_cache_manager()
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved security info for {symbol} from cache")
                return cached_data
        
        start_time = time.time()
        
        try:
            # Apply rate limiting before making requests
            self._rate_limit()
            
            # Get security information using yfinance
            ticker = self._retry_with_backoff(yf.Ticker, symbol)
            
            self._rate_limit()
            info = self._retry_with_backoff(lambda: ticker.info)
            
            # Log API call details
            response_time = time.time() - start_time
            log_api_call(
                logger, 
                'yfinance', 
                f'/ticker/{symbol}/info',
                {'symbol': symbol},
                response_time,
                200  # Assume success if no exception
            )
            
            # Determine if this is an ETF and create appropriate object
            is_etf = self._is_etf(info)
            security_info = (
                self._extract_etf_info(symbol, info) if is_etf 
                else self._extract_stock_info(symbol, info)
            )
            
            # Check for data quality issues
            missing_fields = []
            invalid_values = {}
            
            if not security_info.name:
                missing_fields.append('name')
            
            if security_info.current_price <= 0:
                invalid_values['current_price'] = security_info.current_price
            
            if missing_fields or invalid_values:
                log_data_quality_issue(
                    logger, symbol, 'security_info', 
                    'Missing or invalid data fields detected',
                    'warning', missing_fields, invalid_values
                )
            
            # Validate the security info
            security_info.validate()
            
            # Cache the result
            if use_cache:
                cache.set(
                    cache_key, 
                    security_info, 
                    data_type="security_info", 
                    tags=[symbol, "security_info", "etf" if is_etf else "stock"]
                )
            
            logger.info(f"Successfully retrieved {'ETF' if is_etf else 'stock'} info for {symbol}")
            return security_info
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # Log failed API call
            log_api_call(
                logger, 
                'yfinance', 
                f'/ticker/{symbol}/info',
                {'symbol': symbol},
                response_time,
                error=e
            )
            
            logger.error(f"Error retrieving security info for {symbol}: {str(e)}")
            raise DataRetrievalError(
                f"Failed to retrieve security info for {symbol}: {str(e)}",
                symbol=symbol,
                data_source='yfinance',
                original_exception=e
            )
    
    # Alias for backward compatibility
    get_stock_info = get_security_info

    def _rate_limit(self) -> None:
        """Apply rate limiting to avoid API throttling."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.rate_limit_delay:
            # Add a small random delay to avoid synchronized requests
            delay = self.rate_limit_delay - elapsed + random.uniform(0.1, 0.5)
            logger.debug(f"Rate limiting applied, sleeping for {delay:.2f} seconds")
            time.sleep(delay)
        
        self.last_request_time = time.time()
    
    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """Execute a function with exponential backoff retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            DataRetrievalError: If all retry attempts fail
        """
        max_attempts = self.retry_attempts
        attempt = 0
        last_exception = None
        
        while attempt < max_attempts:
            try:
                # Execute the function
                return func(*args, **kwargs)
                
            except Exception as e:
                attempt += 1
                last_exception = e
                
                if attempt < max_attempts:
                    # Calculate exponential backoff with jitter
                    backoff_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Request failed (attempt {attempt}/{max_attempts}): {str(e)}. "
                        f"Retrying in {backoff_time:.2f} seconds..."
                    )
                    time.sleep(backoff_time)
                else:
                    logger.error(f"All {max_attempts} attempts failed: {str(e)}")
        
        # If we get here, all attempts failed
        raise DataRetrievalError(f"Failed to retrieve data after {max_attempts} attempts: {str(last_exception)}")
    
    def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1y", 
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get historical price data for a stock.
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with historical price data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.info(f"Retrieving historical data for {symbol} with period={period}, interval={interval}")
        
        # Generate cache key
        cache_key = f"historical_data:{symbol}:{period}:{interval}"
        cache = get_cache_manager()
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                logger.info(f"Retrieved historical data for {symbol} from cache")
                return cached_data
        
        try:
            # Apply rate limiting before making requests
            self._rate_limit()
            
            # Get historical data using yfinance
            ticker = self._retry_with_backoff(yf.Ticker, symbol)
            
            self._rate_limit()
            history = self._retry_with_backoff(
                lambda: ticker.history(period=period, interval=interval)
            )
            
            if history.empty:
                logger.warning(f"No historical data found for {symbol}")
                raise DataRetrievalError(f"No historical data found for {symbol}")
            
            # Cache the result
            if use_cache:
                cache.set(
                    cache_key, 
                    history, 
                    data_type="historical_data", 
                    tags=[symbol, "historical_data", period, interval]
                )
            
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving historical data for {symbol}: {str(e)}")
            raise DataRetrievalError(f"Failed to retrieve historical data for {symbol}: {str(e)}")
    
    @monitor_performance('get_financial_statements')
    def get_financial_statements(
        self, 
        symbol: str, 
        statement_type: str = "income", 
        period: str = "annual",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get financial statements for a stock.
        
        Args:
            symbol: Stock ticker symbol
            statement_type: Type of statement ('income', 'balance', 'cash')
            period: Period ('annual' or 'quarterly')
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with financial statement data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.set_context(
            symbol=symbol, 
            statement_type=statement_type, 
            period=period,
            operation='get_financial_statements'
        )
        logger.info(f"Retrieving {period} {statement_type} statement for {symbol}")
        
        # Map statement type to yfinance method
        statement_methods = {
            "income": "income_stmt",
            "balance": "balance_sheet",
            "cash": "cashflow"
        }
        
        if statement_type not in statement_methods:
            raise ValueError(f"Invalid statement type: {statement_type}. Must be one of {list(statement_methods.keys())}")
        
        if period not in ["annual", "quarterly"]:
            raise ValueError(f"Invalid period: {period}. Must be 'annual' or 'quarterly'")
        
        # Generate cache key
        cache_key = f"financial_statements:{symbol}:{statement_type}:{period}"
        cache = get_cache_manager()
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                logger.info(f"Retrieved {period} {statement_type} statement for {symbol} from cache")
                return cached_data
        
        start_time = time.time()
        
        try:
            # Apply rate limiting before making requests
            self._rate_limit()
            
            # Get financial statement using yfinance
            ticker = self._retry_with_backoff(yf.Ticker, symbol)
            method_name = statement_methods[statement_type]
            
            # Get the appropriate property (yfinance changed to properties)
            self._rate_limit()
            
            if period == "quarterly":
                # For quarterly data, use the quarterly properties
                quarterly_methods = {
                    "income": "quarterly_income_stmt",
                    "balance": "quarterly_balance_sheet", 
                    "cash": "quarterly_cashflow"
                }
                if statement_type in quarterly_methods:
                    method_name = quarterly_methods[statement_type]
            
            statement = self._retry_with_backoff(
                lambda: getattr(ticker, method_name)
            )
            
            response_time = time.time() - start_time
            
            # Log API call details
            log_api_call(
                logger, 
                'yfinance', 
                f'/ticker/{symbol}/{method_name}',
                {'symbol': symbol, 'statement_type': statement_type, 'period': period},
                response_time,
                200
            )
            
            if statement.empty:
                log_data_quality_issue(
                    logger, symbol, f'{statement_type}_statement',
                    f'No {period} {statement_type} statement data available',
                    'warning'
                )
                raise DataRetrievalError(
                    f"No {statement_type} statement data found for {symbol}",
                    symbol=symbol,
                    data_source='yfinance',
                    context={'statement_type': statement_type, 'period': period}
                )
            
            # Check data quality
            if len(statement.columns) == 0:
                log_data_quality_issue(
                    logger, symbol, f'{statement_type}_statement',
                    'Statement has no time periods',
                    'error'
                )
            elif len(statement.columns) < 2:
                log_data_quality_issue(
                    logger, symbol, f'{statement_type}_statement',
                    'Statement has limited historical data (less than 2 periods)',
                    'warning'
                )
            
            # Cache the result
            if use_cache:
                cache.set(
                    cache_key, 
                    statement, 
                    data_type="financial_statements", 
                    tags=[symbol, "financial_statements", statement_type, period]
                )
            
            logger.info(f"Successfully retrieved {period} {statement_type} statement for {symbol} "
                       f"({len(statement.columns)} periods, {len(statement)} rows)")
            
            return statement
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # Log failed API call
            log_api_call(
                logger, 
                'yfinance', 
                f'/ticker/{symbol}/{method_name}',
                {'symbol': symbol, 'statement_type': statement_type, 'period': period},
                response_time,
                error=e
            )
            
            logger.error(f"Error retrieving {statement_type} statement for {symbol}: {str(e)}")
            raise DataRetrievalError(
                f"Failed to retrieve {statement_type} statement for {symbol}: {str(e)}",
                symbol=symbol,
                data_source='yfinance',
                context={'statement_type': statement_type, 'period': period},
                original_exception=e
            )
    
    def get_stock_peers(self, symbol: str, use_cache: bool = True) -> List[str]:
        """Get peer companies for a stock.
        
        Args:
            symbol: Stock ticker symbol
            use_cache: Whether to use cached data if available
            
        Returns:
            List of peer company ticker symbols
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.info(f"Retrieving peer companies for {symbol}")
        
        # Generate cache key
        cache_key = f"peer_data:{symbol}"
        cache = get_cache_manager()
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                logger.info(f"Retrieved peer companies for {symbol} from cache")
                return cached_data
        
        try:
            # Get stock info to determine sector and industry
            stock_info = self.get_stock_info(symbol, use_cache=use_cache)
            sector = stock_info.sector
            industry = stock_info.industry
            
            if not sector or not industry:
                logger.warning(f"No sector/industry information for {symbol}")
                return []
            
            # This is a simplified approach - in a real implementation,
            # you might want to use a more sophisticated method to find peers
            # For now, we'll just return an empty list as yfinance doesn't directly
            # provide peer information
            
            # In a real implementation, you could:
            # 1. Use a separate API to get peers
            # 2. Query for companies in the same industry
            # 3. Use pre-defined peer groups
            
            logger.warning("Peer company retrieval not fully implemented")
            peers = []
            
            # Cache the result
            if use_cache:
                cache.set(
                    cache_key, 
                    peers, 
                    data_type="peer_data", 
                    tags=[symbol, "peer_data"]
                )
            
            return peers
            
        except Exception as e:
            logger.error(f"Error retrieving peer companies for {symbol}: {str(e)}")
            raise DataRetrievalError(f"Failed to retrieve peer companies for {symbol}: {str(e)}")
    
    def validate_symbol(self, symbol: str, use_cache: bool = True) -> bool:
        """Validate if a stock symbol exists.
        
        Args:
            symbol: Stock ticker symbol
            use_cache: Whether to use cached data if available
            
        Returns:
            True if the symbol is valid, False otherwise
        """
        logger.info(f"Validating symbol: {symbol}")
        
        # Generate cache key
        cache_key = f"symbol_valid:{symbol}"
        cache = get_cache_manager()
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                logger.info(f"Retrieved symbol validation for {symbol} from cache")
                return cached_data
        
        try:
            # Apply rate limiting before making requests
            self._rate_limit()
            
            ticker = self._retry_with_backoff(yf.Ticker, symbol)
            
            self._rate_limit()
            info = self._retry_with_backoff(lambda: ticker.info)
            
            # Check if we got meaningful data
            is_valid = 'regularMarketPrice' in info or 'currentPrice' in info
            
            if not is_valid:
                logger.warning(f"Symbol {symbol} appears to be invalid")
            
            # Cache the result
            if use_cache:
                cache.set(
                    cache_key, 
                    is_valid, 
                    data_type="general", 
                    tags=[symbol, "symbol_validation"]
                )
            
            return is_valid
                
        except Exception as e:
            logger.warning(f"Symbol validation failed for {symbol}: {str(e)}")
            return False