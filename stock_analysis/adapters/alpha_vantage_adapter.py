"""Alpha Vantage adapter implementation.

This module provides an adapter for Alpha Vantage financial data API,
implementing the FinancialDataAdapter interface.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import requests
from datetime import datetime

from .base_adapter import FinancialDataAdapter
from .config import get_adapter_config
from .utils import HTTPClient, RateLimiter, normalize_symbol, validate_data_quality, retry_with_backoff
from stock_analysis.utils.exceptions import DataRetrievalError, NetworkError, ConfigurationError
from stock_analysis.utils.logging import get_logger

logger = get_logger(__name__)


class AlphaVantageAdapter(FinancialDataAdapter):
    """Adapter for Alpha Vantage financial data API.
    
    This adapter provides access to financial data from Alpha Vantage,
    including stock quotes, historical prices, financial statements,
    and technical indicators.
    """
    
    def __init__(self):
        """Initialize Alpha Vantage adapter."""
        config = get_adapter_config('alpha_vantage')
        super().__init__('alpha_vantage', config.settings if config else {})
        
        # Get API key from configuration
        self.api_key = config.get_credential('api_key') if config else None
        if not self.api_key:
            logger.warning("Alpha Vantage API key not found in configuration")
        
        # Initialize HTTP client with rate limiting
        rate_limit = config.rate_limit if config else 0.2  # Default to 0.2 requests per second (5 per minute)
        timeout = config.timeout if config else 30
        max_retries = config.max_retries if config else 3
        
        self.http_client = HTTPClient(
            base_url="https://www.alphavantage.co",
            headers={
                'User-Agent': 'StockAnalysisApp/1.0',
                'Accept': 'application/json',
            },
            timeout=timeout,
            rate_limiter=RateLimiter(rate_limit),
            max_retries=max_retries
        )
        
        # Cache for data
        self._data_cache = {}
        self._cache_ttl = config.get_setting('cache_ttl', 900) if config else 900  # 15 minutes default
    
    def _validate_api_key(self) -> None:
        """Validate that API key is available.
        
        Raises:
            ConfigurationError: If API key is not configured
        """
        # If API key is already set, use it
        if self.api_key:
            return
            
        # Try to get API key from config.yaml
        from stock_analysis.utils.config import config
        api_key = config.get('stock_analysis.data_sources.alpha_vantage.api_key')
        
        if api_key:
            self.api_key = api_key
            return
            
        # If still no API key, raise error
        raise ConfigurationError(
            "Alpha Vantage API key not configured",
            config_key="api_key"
        )
    
    def get_security_info(self, symbol: str) -> Dict[str, Any]:
        """Retrieve basic security information.
        
        Args:
            symbol: Security ticker symbol
            
        Returns:
            Dictionary containing security information
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
            ConfigurationError: If API key is not configured
        """
        try:
            self._validate_api_key()
            symbol = normalize_symbol(symbol)
            self.logger.info(f"Retrieving security info for {symbol}")
            
            # Get global quote data
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            quote_response = self.http_client.get('/query', params=params)
            
            if 'Global Quote' not in quote_response:
                raise DataRetrievalError(f"Failed to retrieve quote data for {symbol}")
            
            quote_data = quote_response['Global Quote']
            
            # Get company overview data
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            overview_response = self.http_client.get('/query', params=params)
            
            # Check if we got valid data
            if not overview_response or 'Symbol' not in overview_response:
                # If overview fails, we'll still return quote data
                self.logger.warning(f"Failed to retrieve company overview for {symbol}")
                overview_data = {}
            else:
                overview_data = overview_response
            
            # Combine data
            result = {
                'symbol': symbol,
                'name': overview_data.get('Name', symbol),
                'current_price': float(quote_data.get('05. price', 0)),
                'change': float(quote_data.get('09. change', 0)),
                'change_percent': float(quote_data.get('10. change percent', '0%').replace('%', '')),
                'volume': int(quote_data.get('06. volume', 0)),
                'market_cap': float(overview_data.get('MarketCapitalization', 0)),
                'pe_ratio': float(overview_data.get('PERatio', 0)) if overview_data.get('PERatio') else None,
                'pb_ratio': float(overview_data.get('PriceToBookRatio', 0)) if overview_data.get('PriceToBookRatio') else None,
                'dividend_yield': float(overview_data.get('DividendYield', 0)) if overview_data.get('DividendYield') else None,
                'eps': float(overview_data.get('EPS', 0)) if overview_data.get('EPS') else None,
                'beta': float(overview_data.get('Beta', 0)) if overview_data.get('Beta') else None,
                'sector': overview_data.get('Sector', ''),
                'industry': overview_data.get('Industry', ''),
                'description': overview_data.get('Description', '')
            }
            
            # Validate data quality
            required_fields = ['symbol', 'name', 'current_price']
            validation = validate_data_quality(result, required_fields)
            
            if not validation['is_valid']:
                self.logger.warning(f"Data quality issues for {symbol}: {validation}")
            
            return result
            
        except ConfigurationError as e:
            raise e
        except Exception as e:
            self.logger.error(f"Error retrieving security info for {symbol}: {str(e)}")
            raise DataRetrievalError(f"Failed to retrieve security info for {symbol}: {str(e)}")
    
    def get_historical_prices(
        self, 
        symbol: str, 
        period: str = "1y", 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Retrieve historical price data.
        
        Args:
            symbol: Security ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)
            
        Returns:
            DataFrame with historical price data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
            ConfigurationError: If API key is not configured
        """
        try:
            self._validate_api_key()
            symbol = normalize_symbol(symbol)
            self.logger.info(f"Retrieving historical data for {symbol} (period={period}, interval={interval})")
            
            # Map interval to Alpha Vantage function and output size
            interval_map = {
                '1m': ('TIME_SERIES_INTRADAY', '1min', 'compact'),
                '5m': ('TIME_SERIES_INTRADAY', '5min', 'compact'),
                '15m': ('TIME_SERIES_INTRADAY', '15min', 'compact'),
                '30m': ('TIME_SERIES_INTRADAY', '30min', 'compact'),
                '60m': ('TIME_SERIES_INTRADAY', '60min', 'compact'),
                '1d': ('TIME_SERIES_DAILY', None, 'compact'),
                '1wk': ('TIME_SERIES_WEEKLY', None, None),
                '1mo': ('TIME_SERIES_MONTHLY', None, None)
            }
            
            # For longer periods, use full output size
            if period in ['1y', '2y', '5y', '10y', 'max']:
                interval_map['1d'] = ('TIME_SERIES_DAILY', None, 'full')
            
            if interval not in interval_map:
                interval = '1d'  # Default to daily
            
            function, time_interval, output_size = interval_map[interval]
            
            # Build request parameters
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            if time_interval:
                params['interval'] = time_interval
            
            if output_size:
                params['outputsize'] = output_size
            
            # Make request
            response = self.http_client.get('/query', params=params)
            
            # Extract data based on function
            time_series_key = None
            if function == 'TIME_SERIES_INTRADAY':
                time_series_key = f"Time Series ({time_interval})"
            elif function == 'TIME_SERIES_DAILY':
                time_series_key = "Time Series (Daily)"
            elif function == 'TIME_SERIES_WEEKLY':
                time_series_key = "Weekly Time Series"
            elif function == 'TIME_SERIES_MONTHLY':
                time_series_key = "Monthly Time Series"
            
            if not time_series_key or time_series_key not in response:
                raise DataRetrievalError(f"Failed to retrieve historical data for {symbol}")
            
            # Convert to DataFrame
            time_series_data = response[time_series_key]
            df = pd.DataFrame.from_dict(time_series_data, orient='index')
            
            # Rename columns
            column_map = {
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            }
            
            df = df.rename(columns=column_map)
            
            # Convert data types
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col])
            
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Sort by date
            df = df.sort_index()
            
            # Filter based on period
            if period != 'max':
                end_date = datetime.now()
                if period == '1d':
                    start_date = end_date - pd.Timedelta(days=1)
                    df = df[df.index >= start_date]
                elif period == '5d':
                    start_date = end_date - pd.Timedelta(days=5)
                    df = df[df.index >= start_date]
                elif period == '1mo':
                    start_date = end_date - pd.Timedelta(days=30)
                    df = df[df.index >= start_date]
                elif period == '3mo':
                    start_date = end_date - pd.Timedelta(days=90)
                    df = df[df.index >= start_date]
                elif period == '6mo':
                    start_date = end_date - pd.Timedelta(days=180)
                    df = df[df.index >= start_date]
                elif period == '1y':
                    start_date = end_date - pd.Timedelta(days=365)
                    df = df[df.index >= start_date]
                elif period == '2y':
                    start_date = end_date - pd.Timedelta(days=730)
                    df = df[df.index >= start_date]
                elif period == '5y':
                    start_date = end_date - pd.Timedelta(days=1825)
                    df = df[df.index >= start_date]
                elif period == '10y':
                    start_date = end_date - pd.Timedelta(days=3650)
                    df = df[df.index >= start_date]
                elif period == 'ytd':
                    start_date = datetime(end_date.year, 1, 1)
                    df = df[df.index >= start_date]
            
            return df
            
        except ConfigurationError as e:
            raise e
        except Exception as e:
            self.logger.error(f"Error retrieving historical data for {symbol}: {str(e)}")
            raise DataRetrievalError(f"Failed to retrieve historical data for {symbol}: {str(e)}")
    
    def get_financial_statements(
        self, 
        symbol: str, 
        statement_type: str = "income", 
        period: str = "annual",
        years: int = 5
    ) -> pd.DataFrame:
        """Retrieve financial statements.
        
        Args:
            symbol: Security ticker symbol
            statement_type: Type of statement ('income', 'balance', 'cash')
            period: Period ('annual' or 'quarterly')
            years: Number of years of historical data to retrieve
            
        Returns:
            DataFrame with financial statement data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
            ConfigurationError: If API key is not configured
        """
        try:
            self._validate_api_key()
            symbol = normalize_symbol(symbol)
            self.logger.info(f"Retrieving {period} {statement_type} statement for {symbol}")
            
            # Map statement type to Alpha Vantage function
            statement_map = {
                'income': 'INCOME_STATEMENT',
                'balance': 'BALANCE_SHEET',
                'cash': 'CASH_FLOW'
            }
            
            function = statement_map.get(statement_type, 'INCOME_STATEMENT')
            
            # Build request parameters
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            # Make request
            response = self.http_client.get('/query', params=params)
            
            # Check for error response
            if 'Error Message' in response:
                raise DataRetrievalError(f"Alpha Vantage API error: {response['Error Message']}")
            
            # Extract data based on period
            reports_key = 'annualReports' if period == 'annual' else 'quarterlyReports'
            
            if reports_key not in response:
                raise DataRetrievalError(f"Failed to retrieve {statement_type} statement for {symbol}")
            
            reports = response[reports_key]
            
            if not reports:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(reports)
            
            # Set fiscal date as index
            if 'fiscalDateEnding' in df.columns:
                df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
                df = df.set_index('fiscalDateEnding')
                df = df.sort_index(ascending=False)  # Most recent first
            
            # Convert numeric columns
            for col in df.columns:
                if col != 'reportedCurrency':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except ConfigurationError as e:
            raise e
        except Exception as e:
            self.logger.error(f"Error retrieving {statement_type} statement for {symbol}: {str(e)}")
            raise DataRetrievalError(f"Failed to retrieve {statement_type} statement for {symbol}: {str(e)}")
    
    def get_technical_indicators(
        self, 
        symbol: str, 
        indicators: List[str] = None
    ) -> Dict[str, Any]:
        """Retrieve technical indicators.
        
        Args:
            symbol: Security ticker symbol
            indicators: List of technical indicators to retrieve
            
        Returns:
            Dictionary containing technical indicator values
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
            ConfigurationError: If API key is not configured
        """
        try:
            self._validate_api_key()
            symbol = normalize_symbol(symbol)
            
            if not indicators:
                indicators = ['sma_20', 'sma_50', 'rsi_14', 'macd']
            
            self.logger.info(f"Retrieving technical indicators for {symbol}: {indicators}")
            
            result = {}
            
            # Process each indicator
            for indicator in indicators:
                try:
                    if indicator.startswith('sma_'):
                        period = int(indicator.split('_')[1])
                        result[indicator] = self._get_sma(symbol, period)
                    elif indicator.startswith('ema_'):
                        period = int(indicator.split('_')[1])
                        result[indicator] = self._get_ema(symbol, period)
                    elif indicator.startswith('rsi_'):
                        period = int(indicator.split('_')[1])
                        result[indicator] = self._get_rsi(symbol, period)
                    elif indicator == 'macd':
                        result[indicator] = self._get_macd(symbol)
                    else:
                        self.logger.warning(f"Unsupported technical indicator: {indicator}")
                except Exception as e:
                    self.logger.warning(f"Error retrieving {indicator} for {symbol}: {str(e)}")
            
            return result
            
        except ConfigurationError as e:
            raise e
        except Exception as e:
            self.logger.error(f"Error retrieving technical indicators for {symbol}: {str(e)}")
            return {}
    
    def _get_sma(self, symbol: str, period: int) -> float:
        """Get Simple Moving Average (SMA) for a symbol.
        
        Args:
            symbol: Security ticker symbol
            period: Time period for SMA
            
        Returns:
            SMA value
        """
        params = {
            'function': 'SMA',
            'symbol': symbol,
            'interval': 'daily',
            'time_period': period,
            'series_type': 'close',
            'apikey': self.api_key
        }
        
        response = self.http_client.get('/query', params=params)
        
        if 'Technical Analysis: SMA' not in response:
            return None
        
        # Get the most recent value
        technical_data = response['Technical Analysis: SMA']
        latest_date = max(technical_data.keys())
        latest_value = technical_data[latest_date]['SMA']
        
        return float(latest_value)
    
    def _get_ema(self, symbol: str, period: int) -> float:
        """Get Exponential Moving Average (EMA) for a symbol.
        
        Args:
            symbol: Security ticker symbol
            period: Time period for EMA
            
        Returns:
            EMA value
        """
        params = {
            'function': 'EMA',
            'symbol': symbol,
            'interval': 'daily',
            'time_period': period,
            'series_type': 'close',
            'apikey': self.api_key
        }
        
        response = self.http_client.get('/query', params=params)
        
        if 'Technical Analysis: EMA' not in response:
            return None
        
        # Get the most recent value
        technical_data = response['Technical Analysis: EMA']
        latest_date = max(technical_data.keys())
        latest_value = technical_data[latest_date]['EMA']
        
        return float(latest_value)
    
    def _get_rsi(self, symbol: str, period: int) -> float:
        """Get Relative Strength Index (RSI) for a symbol.
        
        Args:
            symbol: Security ticker symbol
            period: Time period for RSI
            
        Returns:
            RSI value
        """
        params = {
            'function': 'RSI',
            'symbol': symbol,
            'interval': 'daily',
            'time_period': period,
            'series_type': 'close',
            'apikey': self.api_key
        }
        
        response = self.http_client.get('/query', params=params)
        
        if 'Technical Analysis: RSI' not in response:
            return None
        
        # Get the most recent value
        technical_data = response['Technical Analysis: RSI']
        latest_date = max(technical_data.keys())
        latest_value = technical_data[latest_date]['RSI']
        
        return float(latest_value)
    
    def _get_macd(self, symbol: str) -> Dict[str, float]:
        """Get Moving Average Convergence Divergence (MACD) for a symbol.
        
        Args:
            symbol: Security ticker symbol
            
        Returns:
            Dictionary with MACD values
        """
        params = {
            'function': 'MACD',
            'symbol': symbol,
            'interval': 'daily',
            'series_type': 'close',
            'fastperiod': 12,
            'slowperiod': 26,
            'signalperiod': 9,
            'apikey': self.api_key
        }
        
        response = self.http_client.get('/query', params=params)
        
        if 'Technical Analysis: MACD' not in response:
            return {}
        
        # Get the most recent value
        technical_data = response['Technical Analysis: MACD']
        latest_date = max(technical_data.keys())
        latest_values = technical_data[latest_date]
        
        return {
            'macd': float(latest_values['MACD']),
            'signal': float(latest_values['MACD_Signal']),
            'histogram': float(latest_values['MACD_Hist'])
        }
    
    def get_market_data(self, data_type: str) -> Dict[str, Any]:
        """Retrieve market data like indices, commodities, forex.
        
        Args:
            data_type: Type of market data ('indices', 'commodities', 'forex', 'sectors')
            
        Returns:
            Dictionary containing market data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
            ConfigurationError: If API key is not configured
        """
        try:
            self._validate_api_key()
            self.logger.info(f"Retrieving market data: {data_type}")
            
            if data_type == 'indices':
                return self._get_market_indices()
            elif data_type == 'commodities':
                return self._get_commodities()
            elif data_type == 'forex':
                return self._get_forex()
            elif data_type == 'sectors':
                return self._get_sector_performance()
            else:
                self.logger.warning(f"Unsupported market data type: {data_type}")
                return {}
                
        except ConfigurationError as e:
            raise e
        except Exception as e:
            self.logger.error(f"Error retrieving market data ({data_type}): {str(e)}")
            return {}
    
    def _get_market_indices(self) -> Dict[str, Any]:
        """Get major market indices data."""
        indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ',
            'DIA': 'Dow Jones',
            'IWM': 'Russell 2000',
            'VIX': 'VIX'
        }
        
        result = {}
        for symbol, name in indices.items():
            try:
                # Get global quote data
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': self.api_key
                }
                
                response = self.http_client.get('/query', params=params)
                
                if 'Global Quote' in response:
                    quote_data = response['Global Quote']
                    result[name] = {
                        'symbol': symbol,
                        'name': name,
                        'price': float(quote_data.get('05. price', 0)),
                        'change': float(quote_data.get('09. change', 0)),
                        'change_percent': float(quote_data.get('10. change percent', '0%').replace('%', ''))
                    }
            except Exception as e:
                self.logger.warning(f"Error getting data for {name} ({symbol}): {str(e)}")
        
        return result
    
    def _get_commodities(self) -> Dict[str, Any]:
        """Get commodities data."""
        commodities = {
            'GLD': 'Gold',
            'SLV': 'Silver',
            'USO': 'Crude Oil',
            'UNG': 'Natural Gas'
        }
        
        result = {}
        for symbol, name in commodities.items():
            try:
                # Get global quote data
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': self.api_key
                }
                
                response = self.http_client.get('/query', params=params)
                
                if 'Global Quote' in response:
                    quote_data = response['Global Quote']
                    result[name] = {
                        'symbol': symbol,
                        'name': name,
                        'price': float(quote_data.get('05. price', 0)),
                        'change': float(quote_data.get('09. change', 0)),
                        'change_percent': float(quote_data.get('10. change percent', '0%').replace('%', ''))
                    }
            except Exception as e:
                self.logger.warning(f"Error getting data for {name} ({symbol}): {str(e)}")
        
        return result
    
    def _get_forex(self) -> Dict[str, Any]:
        """Get forex data."""
        forex_pairs = {
            'EUR/USD': 'EUR/USD',
            'GBP/USD': 'GBP/USD',
            'USD/JPY': 'USD/JPY',
            'USD/CAD': 'USD/CAD'
        }
        
        result = {}
        for symbol, name in forex_pairs.items():
            try:
                # Get forex data
                from_currency, to_currency = symbol.split('/')
                
                params = {
                    'function': 'CURRENCY_EXCHANGE_RATE',
                    'from_currency': from_currency,
                    'to_currency': to_currency,
                    'apikey': self.api_key
                }
                
                response = self.http_client.get('/query', params=params)
                
                if 'Realtime Currency Exchange Rate' in response:
                    exchange_data = response['Realtime Currency Exchange Rate']
                    result[name] = {
                        'symbol': symbol,
                        'name': name,
                        'price': float(exchange_data.get('5. Exchange Rate', 0)),
                        'bid': float(exchange_data.get('8. Bid Price', 0)) if '8. Bid Price' in exchange_data else None,
                        'ask': float(exchange_data.get('9. Ask Price', 0)) if '9. Ask Price' in exchange_data else None
                    }
            except Exception as e:
                self.logger.warning(f"Error getting data for {name} ({symbol}): {str(e)}")
        
        return result
    
    def _get_sector_performance(self) -> Dict[str, Any]:
        """Get sector performance data."""
        try:
            params = {
                'function': 'SECTOR',
                'apikey': self.api_key
            }
            
            response = self.http_client.get('/query', params=params)
            
            if not response or 'Rank A: Real-Time Performance' not in response:
                return {}
            
            # Extract real-time performance
            sector_data = response['Rank A: Real-Time Performance']
            
            result = {}
            for sector, performance in sector_data.items():
                # Remove "Information Technology" from sector name
                sector_name = sector.replace(" Sector", "")
                # Convert percentage string to float
                performance_value = float(performance.replace('%', ''))
                result[sector_name] = performance_value
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Error getting sector performance: {str(e)}")
            return {}
    
    def get_news(
        self, 
        symbol: str = None, 
        category: str = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve news articles.
        
        Args:
            symbol: Security ticker symbol (optional)
            category: News category (optional)
            limit: Maximum number of articles to retrieve
            
        Returns:
            List of news articles
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
            ConfigurationError: If API key is not configured
        """
        try:
            self._validate_api_key()
            
            if symbol:
                symbol = normalize_symbol(symbol)
                self.logger.info(f"Retrieving news for {symbol}")
                
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': symbol,
                    'apikey': self.api_key,
                    'limit': min(limit, 50)  # API limit is 50
                }
            else:
                self.logger.info("Retrieving general market news")
                
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'apikey': self.api_key,
                    'limit': min(limit, 50)  # API limit is 50
                }
                
                if category:
                    params['topics'] = category
            
            response = self.http_client.get('/query', params=params)
            
            if 'feed' not in response:
                return []
            
            news_items = response['feed']
            
            result = []
            for item in news_items[:limit]:
                news_item = {
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'source': item.get('source', ''),
                    'published_at': item.get('time_published', ''),
                    'summary': item.get('summary', ''),
                    'sentiment': float(item.get('overall_sentiment_score', 0)) if 'overall_sentiment_score' in item else None,
                    'sentiment_label': item.get('overall_sentiment', '')
                }
                result.append(news_item)
            
            return result
            
        except ConfigurationError as e:
            raise e
        except Exception as e:
            self.logger.error(f"Error retrieving news: {str(e)}")
            return []
    
    def get_economic_data(
        self, 
        indicator: str, 
        region: str = None
    ) -> pd.DataFrame:
        """Retrieve economic indicators data.
        
        Args:
            indicator: Economic indicator name
            region: Geographic region (optional)
            
        Returns:
            DataFrame with economic indicator data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
            ConfigurationError: If API key is not configured
        """
        try:
            self._validate_api_key()
            self.logger.info(f"Retrieving economic data: {indicator}")
            
            # Map common indicator names to Alpha Vantage functions
            indicator_map = {
                'GDP': 'REAL_GDP',
                'INFLATION': 'INFLATION',
                'UNEMPLOYMENT': 'UNEMPLOYMENT',
                'RETAIL_SALES': 'RETAIL_SALES',
                'NONFARM_PAYROLL': 'NONFARM_PAYROLL',
                'CPI': 'CPI',
                'FEDERAL_FUNDS_RATE': 'FEDERAL_FUNDS_RATE'
            }
            
            function = indicator_map.get(indicator.upper(), indicator.upper())
            
            params = {
                'function': function,
                'apikey': self.api_key
            }
            
            if region:
                # Only some indicators support region parameter
                if function in ['REAL_GDP', 'INFLATION', 'UNEMPLOYMENT']:
                    params['interval'] = region
            
            response = self.http_client.get('/query', params=params)
            
            # Check for error response
            if 'Error Message' in response:
                raise DataRetrievalError(f"Alpha Vantage API error: {response['Error Message']}")
            
            # Extract data
            data_key = 'data'
            if data_key not in response:
                return pd.DataFrame()
            
            data = response[data_key]
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Set date as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df = df.sort_index(ascending=False)  # Most recent first
            
            # Convert numeric columns
            for col in df.columns:
                if col != 'date':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except ConfigurationError as e:
            raise e
        except Exception as e:
            self.logger.error(f"Error retrieving economic data for {indicator}: {str(e)}")
            return pd.DataFrame()
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a security symbol exists.
        
        Args:
            symbol: Security ticker symbol
            
        Returns:
            True if the symbol is valid, False otherwise
        """
        try:
            symbol = normalize_symbol(symbol)
            
            # Try to get basic security info to validate symbol
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = self.http_client.get('/query', params=params)
            
            if 'Global Quote' in response and response['Global Quote']:
                return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Symbol validation failed for {symbol}: {str(e)}")
            return False