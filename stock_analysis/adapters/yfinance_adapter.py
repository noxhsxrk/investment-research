"""YFinance adapter implementation.

This module provides an adapter for Yahoo Finance data using the yfinance library,
implementing the FinancialDataAdapter interface.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import yfinance as yf

from .base_adapter import FinancialDataAdapter
from .config import get_adapter_config
from .utils import normalize_symbol, validate_data_quality, parse_financial_data
from stock_analysis.services.stock_data_service import StockDataService
from stock_analysis.utils.exceptions import DataRetrievalError
from stock_analysis.utils.logging import get_logger

logger = get_logger(__name__)


class YFinanceAdapter(FinancialDataAdapter):
    """Adapter for Yahoo Finance data using yfinance library.
    
    This adapter wraps the existing StockDataService to provide
    the standardized FinancialDataAdapter interface.
    """
    
    def __init__(self):
        """Initialize YFinance adapter."""
        config = get_adapter_config('yfinance')
        super().__init__('yfinance', config.settings if config else {})
        
        # Initialize the underlying stock data service
        self.stock_service = StockDataService()
        
        # Cache for technical indicators (since yfinance doesn't provide them directly)
        self._technical_indicators_cache = {}
    
    def get_security_info(self, symbol: str) -> Dict[str, Any]:
        """Retrieve basic security information.
        
        Args:
            symbol: Security ticker symbol
            
        Returns:
            Dictionary containing security information
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        try:
            symbol = normalize_symbol(symbol)
            self.logger.info(f"Retrieving security info for {symbol}")
            
            # Use existing stock data service
            security_info = self.stock_service.get_security_info(symbol)
            
            # Convert to dictionary format
            result = {
                'symbol': security_info.symbol,
                'name': security_info.name,
                'current_price': security_info.current_price,
                'market_cap': security_info.market_cap,
                'beta': security_info.beta
            }
            
            # Add additional fields based on security type
            if hasattr(security_info, 'pe_ratio'):
                result.update({
                    'pe_ratio': security_info.pe_ratio,
                    'pb_ratio': security_info.pb_ratio,
                    'dividend_yield': security_info.dividend_yield,
                    'sector': security_info.sector,
                    'industry': security_info.industry
                })
            
            if hasattr(security_info, 'expense_ratio'):
                result.update({
                    'expense_ratio': security_info.expense_ratio,
                    'assets_under_management': security_info.assets_under_management,
                    'nav': security_info.nav,
                    'category': security_info.category
                })
            
            # Validate data quality
            required_fields = ['symbol', 'name', 'current_price']
            validation = validate_data_quality(result, required_fields)
            
            if not validation['is_valid']:
                self.logger.warning(f"Data quality issues for {symbol}: {validation}")
            
            return result
            
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
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with historical price data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        try:
            symbol = normalize_symbol(symbol)
            self.logger.info(f"Retrieving historical data for {symbol} (period={period}, interval={interval})")
            
            # Use existing stock data service
            historical_data = self.stock_service.get_historical_data(symbol, period, interval)
            
            if historical_data.empty:
                raise DataRetrievalError(f"No historical data found for {symbol}")
            
            return historical_data
            
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
        """
        try:
            symbol = normalize_symbol(symbol)
            self.logger.info(f"Retrieving {period} {statement_type} statement for {symbol}")
            
            # Use existing stock data service
            statement_data = self.stock_service.get_financial_statements(symbol, statement_type, period)
            
            if statement_data.empty:
                raise DataRetrievalError(f"No {statement_type} statement data found for {symbol}")
            
            return statement_data
            
        except Exception as e:
            self.logger.error(f"Error retrieving {statement_type} statement for {symbol}: {str(e)}")
            raise DataRetrievalError(f"Failed to retrieve {statement_type} statement for {symbol}: {str(e)}")
    
    def get_technical_indicators(
        self, 
        symbol: str, 
        indicators: List[str] = None
    ) -> Dict[str, Any]:
        """Retrieve technical indicators.
        
        Note: YFinance doesn't provide technical indicators directly,
        so this method calculates basic indicators from price data.
        
        Args:
            symbol: Security ticker symbol
            indicators: List of technical indicators to retrieve
            
        Returns:
            Dictionary containing technical indicator values
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        try:
            symbol = normalize_symbol(symbol)
            
            if not indicators:
                indicators = ['sma_20', 'sma_50', 'rsi_14', 'macd']
            
            self.logger.info(f"Calculating technical indicators for {symbol}: {indicators}")
            
            # Get historical data for calculations
            historical_data = self.get_historical_prices(symbol, period="1y", interval="1d")
            
            if historical_data.empty:
                return {}
            
            result = {}
            
            # Calculate requested indicators
            for indicator in indicators:
                try:
                    if indicator.startswith('sma_'):
                        # Simple Moving Average
                        period = int(indicator.split('_')[1])
                        result[indicator] = self._calculate_sma(historical_data, period)
                    elif indicator.startswith('ema_'):
                        # Exponential Moving Average
                        period = int(indicator.split('_')[1])
                        result[indicator] = self._calculate_ema(historical_data, period)
                    elif indicator.startswith('rsi_'):
                        # Relative Strength Index
                        period = int(indicator.split('_')[1])
                        result[indicator] = self._calculate_rsi(historical_data, period)
                    elif indicator == 'macd':
                        # MACD
                        result[indicator] = self._calculate_macd(historical_data)
                    else:
                        self.logger.warning(f"Unknown technical indicator: {indicator}")
                except Exception as e:
                    self.logger.warning(f"Error calculating {indicator} for {symbol}: {str(e)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving technical indicators for {symbol}: {str(e)}")
            return {}
    
    def _calculate_sma(self, data: pd.DataFrame, period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(data) < period:
            return None
        return float(data['Close'].tail(period).mean())
    
    def _calculate_ema(self, data: pd.DataFrame, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return None
        return float(data['Close'].ewm(span=period).mean().iloc[-1])
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(data) < period + 1:
            return None
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
    
    def _calculate_macd(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if len(data) < 26:
            return {}
        
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None,
            'signal': float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None,
            'histogram': float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None
        }
    
    def get_market_data(self, data_type: str) -> Dict[str, Any]:
        """Retrieve market data like indices, commodities, forex.
        
        Args:
            data_type: Type of market data ('indices', 'commodities', 'forex', 'sectors')
            
        Returns:
            Dictionary containing market data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        try:
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
                
        except Exception as e:
            self.logger.error(f"Error retrieving market data ({data_type}): {str(e)}")
            return {}
    
    def _get_market_indices(self) -> Dict[str, Any]:
        """Get major market indices data."""
        indices = {
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'Dow Jones',
            '^RUT': 'Russell 2000',
            '^VIX': 'VIX'
        }
        
        result = {}
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                result[name] = {
                    'symbol': symbol,
                    'name': name,
                    'price': info.get('regularMarketPrice', info.get('currentPrice')),
                    'change': info.get('regularMarketChange'),
                    'change_percent': info.get('regularMarketChangePercent')
                }
            except Exception as e:
                self.logger.warning(f"Error getting data for {name} ({symbol}): {str(e)}")
        
        return result
    
    def _get_commodities(self) -> Dict[str, Any]:
        """Get commodities data."""
        commodities = {
            'GC=F': 'Gold',
            'SI=F': 'Silver',
            'CL=F': 'Crude Oil',
            'NG=F': 'Natural Gas'
        }
        
        result = {}
        for symbol, name in commodities.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                result[name] = {
                    'symbol': symbol,
                    'name': name,
                    'price': info.get('regularMarketPrice', info.get('currentPrice')),
                    'change': info.get('regularMarketChange'),
                    'change_percent': info.get('regularMarketChangePercent')
                }
            except Exception as e:
                self.logger.warning(f"Error getting data for {name} ({symbol}): {str(e)}")
        
        return result
    
    def _get_forex(self) -> Dict[str, Any]:
        """Get forex data."""
        forex_pairs = {
            'EURUSD=X': 'EUR/USD',
            'GBPUSD=X': 'GBP/USD',
            'USDJPY=X': 'USD/JPY',
            'USDCAD=X': 'USD/CAD'
        }
        
        result = {}
        for symbol, name in forex_pairs.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                result[name] = {
                    'symbol': symbol,
                    'name': name,
                    'price': info.get('regularMarketPrice', info.get('currentPrice')),
                    'change': info.get('regularMarketChange'),
                    'change_percent': info.get('regularMarketChangePercent')
                }
            except Exception as e:
                self.logger.warning(f"Error getting data for {name} ({symbol}): {str(e)}")
        
        return result
    
    def _get_sector_performance(self) -> Dict[str, Any]:
        """Get sector performance data using sector ETFs."""
        sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financial',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrial',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLU': 'Utilities',
            'XLB': 'Materials',
            'XLRE': 'Real Estate'
        }
        
        result = {}
        for symbol, name in sector_etfs.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                result[name] = {
                    'symbol': symbol,
                    'name': name,
                    'price': info.get('regularMarketPrice', info.get('currentPrice')),
                    'change': info.get('regularMarketChange'),
                    'change_percent': info.get('regularMarketChangePercent')
                }
            except Exception as e:
                self.logger.warning(f"Error getting data for {name} ({symbol}): {str(e)}")
        
        return result
    
    def get_company_news(self, symbol: str, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve news for a specific company.
        
        Args:
            symbol: Security ticker symbol
            days: Number of days to look back
            limit: Maximum number of news items to return
            
        Returns:
            List of news items as dictionaries
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        try:
            symbol = normalize_symbol(symbol)
            self.logger.info(f"Retrieving news for {symbol} (last {days} days, limit {limit})")
            
            # Use yfinance to get news
            ticker = yf.Ticker(symbol)
            news_data = ticker.news
            
            if not news_data:
                self.logger.warning(f"No news found for {symbol}")
                return []
            
            # Process and format news data
            result = []
            for item in news_data[:limit]:
                try:
                    # Handle the nested structure of YFinance news data
                    content = item.get('content', {}) if isinstance(item.get('content'), dict) else {}
                    
                    # Get title from content or item
                    title = content.get('title') if content else item.get('title')
                    if not title:
                        title = f"{symbol} News Update"
                    
                    # Get summary from content or item
                    summary = content.get('summary') if content else item.get('summary')
                    if not summary or not isinstance(summary, str):
                        summary = "No summary available for this news item."
                    
                    # Get URL from content or item
                    url = None
                    if content and 'clickThroughUrl' in content and content['clickThroughUrl']:
                        if isinstance(content['clickThroughUrl'], dict) and 'url' in content['clickThroughUrl']:
                            url = content['clickThroughUrl']['url']
                        elif isinstance(content['clickThroughUrl'], str):
                            url = content['clickThroughUrl']
                    
                    if not url and content and 'canonicalUrl' in content and content['canonicalUrl']:
                        if isinstance(content['canonicalUrl'], dict) and 'url' in content['canonicalUrl']:
                            url = content['canonicalUrl']['url']
                        elif isinstance(content['canonicalUrl'], str):
                            url = content['canonicalUrl']
                    
                    if not url:
                        url = f"https://finance.yahoo.com/quote/{symbol}"
                    
                    # Get source from content or item
                    source = None
                    if content and 'provider' in content and content['provider']:
                        if isinstance(content['provider'], dict) and 'displayName' in content['provider']:
                            source = content['provider']['displayName']
                        elif isinstance(content['provider'], str):
                            source = content['provider']
                    
                    if not source:
                        source = 'Yahoo Finance'
                    
                    # Get published date from content or item
                    published_at = datetime.now()  # Default to current time
                    
                    if content and 'pubDate' in content and content['pubDate']:
                        try:
                            # Try to parse ISO format date string
                            published_at = datetime.fromisoformat(content['pubDate'].replace('Z', '+00:00'))
                        except (ValueError, TypeError):
                            pass
                    
                    # Create news item with all required fields
                    news_item = {
                        'title': title,
                        'summary': summary,
                        'url': url,
                        'source': source,
                        'published_at': published_at,
                        # Optional fields
                        'categories': ['company', 'stock'],
                        'sentiment': None,  # Will be calculated by the news service
                        'impact': None      # Will be calculated by the news service
                    }
                    
                    result.append(news_item)
                except Exception as e:
                    self.logger.warning(f"Error processing news item for {symbol}: {str(e)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving news for {symbol}: {str(e)}")
            return []
    
    def get_market_news(self, category: Optional[str] = None, days: int = 3, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve general market news.
        
        Args:
            category: News category filter (e.g., 'market', 'economy', 'earnings')
            days: Number of days to look back
            limit: Maximum number of news items to return
            
        Returns:
            List of news items as dictionaries
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        try:
            self.logger.info(f"Retrieving market news (category={category}, last {days} days, limit {limit})")
            
            # For market news, we'll use a market index as proxy
            ticker = yf.Ticker('^GSPC')  # S&P 500
            news_data = ticker.news
            
            if not news_data:
                self.logger.warning("No market news found")
                return []
            
            # Process and format news data
            result = []
            for item in news_data[:limit]:
                try:
                    # Handle the nested structure of YFinance news data
                    content = item.get('content', {}) if isinstance(item.get('content'), dict) else {}
                    
                    # Get title from content or item
                    title = content.get('title') if content else item.get('title')
                    if not title:
                        title = "Market News Update"
                    
                    # Get summary from content or item
                    summary = content.get('summary') if content else item.get('summary')
                    if not summary or not isinstance(summary, str):
                        summary = "No summary available for this market news item."
                    
                    # Get URL from content or item
                    url = None
                    if content and 'clickThroughUrl' in content and content['clickThroughUrl']:
                        if isinstance(content['clickThroughUrl'], dict) and 'url' in content['clickThroughUrl']:
                            url = content['clickThroughUrl']['url']
                        elif isinstance(content['clickThroughUrl'], str):
                            url = content['clickThroughUrl']
                    
                    if not url and content and 'canonicalUrl' in content and content['canonicalUrl']:
                        if isinstance(content['canonicalUrl'], dict) and 'url' in content['canonicalUrl']:
                            url = content['canonicalUrl']['url']
                        elif isinstance(content['canonicalUrl'], str):
                            url = content['canonicalUrl']
                    
                    if not url:
                        url = "https://finance.yahoo.com/news/"
                    
                    # Get source from content or item
                    source = None
                    if content and 'provider' in content and content['provider']:
                        if isinstance(content['provider'], dict) and 'displayName' in content['provider']:
                            source = content['provider']['displayName']
                        elif isinstance(content['provider'], str):
                            source = content['provider']
                    
                    if not source:
                        source = 'Yahoo Finance'
                    
                    # Get published date from content or item
                    published_at = datetime.now()  # Default to current time
                    
                    if content and 'pubDate' in content and content['pubDate']:
                        try:
                            # Try to parse ISO format date string
                            published_at = datetime.fromisoformat(content['pubDate'].replace('Z', '+00:00'))
                        except (ValueError, TypeError):
                            pass
                    
                    # Determine categories based on title and summary
                    categories = []
                    title_lower = title.lower()
                    summary_lower = summary.lower()
                    
                    if 'market' in title_lower or 'market' in summary_lower:
                        categories.append('market')
                    if 'economy' in title_lower or 'economy' in summary_lower:
                        categories.append('economy')
                    if 'earnings' in title_lower or 'earnings' in summary_lower:
                        categories.append('earnings')
                    if 'fed' in title_lower or 'interest rate' in summary_lower:
                        categories.append('policy')
                    
                    # Default category if none detected
                    if not categories:
                        categories.append('market')
                    
                    # Filter by category if specified
                    if category and category not in categories:
                        continue
                    
                    # Create news item with all required fields
                    news_item = {
                        'title': title,
                        'summary': summary,
                        'url': url,
                        'source': source,
                        'published_at': published_at,
                        'categories': categories,
                        'sentiment': None,  # Will be calculated by the news service
                        'impact': None      # Will be calculated by the news service
                    }
                    
                    result.append(news_item)
                except Exception as e:
                    self.logger.warning(f"Error processing market news item: {str(e)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving market news: {str(e)}")
            return []
    
    def get_economic_calendar(self, days_ahead: int = 7, days_behind: int = 1) -> List[Dict[str, Any]]:
        """Retrieve economic calendar events.
        
        Args:
            days_ahead: Number of days to look ahead
            days_behind: Number of days to look behind
            
        Returns:
            List of economic events as dictionaries
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        # YFinance doesn't provide economic calendar data
        self.logger.warning("Economic calendar not supported by YFinance adapter")
        return []
        
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
        """
        try:
            if symbol:
                return self.get_company_news(symbol, days=7, limit=limit)
            else:
                return self.get_market_news(category=category, days=3, limit=limit)
        except Exception as e:
            self.logger.error(f"Error retrieving news: {str(e)}")
            return []
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a security symbol exists.
        
        Args:
            symbol: Security ticker symbol
            
        Returns:
            True if the symbol is valid, False otherwise
        """
        try:
            symbol = normalize_symbol(symbol)
            return self.stock_service.validate_symbol(symbol)
        except Exception as e:
            self.logger.debug(f"Symbol validation failed for {symbol}: {str(e)}")
            return False