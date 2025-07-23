# Extending Data Sources Guide

## Introduction

This guide provides detailed instructions for extending the Stock Analysis Dashboard with new data sources. It covers the adapter interface, implementation patterns, and integration with the existing system.

## Data Source Adapter Interface

All data source adapters must implement the `FinancialDataAdapter` abstract base class, which defines the following methods:

```python
class FinancialDataAdapter(ABC):
    """Abstract base class for financial data adapters."""
    
    @abstractmethod
    def get_security_info(self, symbol: str) -> Dict[str, Any]:
        """Retrieve basic security information."""
        pass
    
    @abstractmethod
    def get_historical_prices(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Retrieve historical price data."""
        pass
    
    @abstractmethod
    def get_financial_statements(self, symbol: str, statement_type: str = "income", period: str = "annual") -> pd.DataFrame:
        """Retrieve financial statements."""
        pass
    
    @abstractmethod
    def get_technical_indicators(self, symbol: str, indicators: Optional[List[str]] = None) -> Dict[str, Any]:
        """Retrieve technical indicators."""
        pass
    
    @abstractmethod
    def get_market_data(self, data_type: str) -> Dict[str, Any]:
        """Retrieve market data like indices, commodities, forex."""
        pass
    
    @abstractmethod
    def get_news(self, symbol: Optional[str] = None, category: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve news articles."""
        pass
    
    @abstractmethod
    def get_economic_data(self, indicator: str, region: Optional[str] = None) -> pd.DataFrame:
        """Retrieve economic indicators data."""
        pass
```

## Implementation Example: REST API Adapter

Here's a detailed example of implementing an adapter for a REST API data source:

```python
import requests
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from stock_analysis.adapters.base_adapter import FinancialDataAdapter
from stock_analysis.utils.exceptions import DataRetrievalError
from stock_analysis.utils.logging import get_logger
from stock_analysis.utils.config import config

class RestApiAdapter(FinancialDataAdapter):
    """Adapter for a REST API financial data source."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the REST API adapter.
        
        Args:
            api_key: API key for authentication (optional)
        """
        self.name = "rest_api"
        self.logger = get_logger(f"{__name__}.RestApiAdapter")
        self.api_key = api_key or self._get_api_key_from_config()
        self.base_url = "https://api.example.com/v1"
        self.timeout = config.get('stock_analysis.data_sources.rest_api.timeout', 30)
        
    def _get_api_key_from_config(self) -> str:
        """Get API key from configuration.
        
        Returns:
            API key string
        """
        return config.get('stock_analysis.data_sources.rest_api.api_key', '')
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a request to the API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            DataRetrievalError: If request fails
        """
        url = f"{self.base_url}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            self.logger.debug(f"Making API request to {url}")
            response = requests.get(url, headers=headers, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            self.logger.error(error_msg)
            raise DataRetrievalError(error_msg, data_source=self.name, original_exception=e)
    
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
            self.logger.info(f"Retrieving security info for {symbol}")
            
            # Make API request
            data = self._make_request(f"securities/{symbol}")
            
            # Transform the data to match the expected format
            return {
                'symbol': symbol,
                'name': data.get('companyName'),
                'current_price': data.get('price'),
                'market_cap': data.get('marketCap'),
                'pe_ratio': data.get('peRatio'),
                'pb_ratio': data.get('pbRatio'),
                'dividend_yield': data.get('dividendYield'),
                'sector': data.get('sector'),
                'industry': data.get('industry'),
                'exchange': data.get('exchange'),
                'currency': data.get('currency'),
                
                # Additional fields for enhanced security info
                'earnings_growth': data.get('earningsGrowth'),
                'revenue_growth': data.get('revenueGrowth'),
                'analyst_rating': data.get('analystRating'),
                'price_target': {
                    'low': data.get('priceTargetLow'),
                    'average': data.get('priceTargetAverage'),
                    'high': data.get('priceTargetHigh')
                } if data.get('priceTargetAverage') else None,
                'analyst_count': data.get('analystCount')
            }
            
        except Exception as e:
            error_msg = f"Failed to retrieve security info for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            raise DataRetrievalError(
                error_msg,
                symbol=symbol,
                data_source=self.name,
                original_exception=e
            )
    
    def get_historical_prices(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
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
            self.logger.info(f"Retrieving historical prices for {symbol} (period={period}, interval={interval})")
            
            # Map period and interval to API parameters
            period_map = {
                "1d": "1day", "5d": "5day", "1mo": "1month", "3mo": "3month",
                "6mo": "6month", "1y": "1year", "2y": "2year", "5y": "5year",
                "10y": "10year", "ytd": "ytd", "max": "max"
            }
            
            interval_map = {
                "1m": "1min", "2m": "2min", "5m": "5min", "15m": "15min",
                "30m": "30min", "60m": "60min", "90m": "90min", "1h": "1hour",
                "1d": "1day", "5d": "5day", "1wk": "1week", "1mo": "1month",
                "3mo": "3month"
            }
            
            # Make API request
            data = self._make_request(
                f"securities/{symbol}/history",
                params={
                    "period": period_map.get(period, "1year"),
                    "interval": interval_map.get(interval, "1day")
                }
            )
            
            # Transform the data to a DataFrame
            history = data.get('history', [])
            if not history:
                return pd.DataFrame()
            
            df = pd.DataFrame(history)
            
            # Rename columns to match expected format
            column_map = {
                'date': 'Date',
                'openPrice': 'Open',
                'highPrice': 'High',
                'lowPrice': 'Low',
                'closePrice': 'Close',
                'volume': 'Volume',
                'adjustedClose': 'Adj Close'
            }
            df = df.rename(columns=column_map)
            
            # Convert date column to datetime and set as index
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            # Sort by date
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            error_msg = f"Failed to retrieve historical prices for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            raise DataRetrievalError(
                error_msg,
                symbol=symbol,
                data_source=self.name,
                original_exception=e
            )
    
    def get_financial_statements(self, symbol: str, statement_type: str = "income", period: str = "annual") -> pd.DataFrame:
        """Retrieve financial statements.
        
        Args:
            symbol: Security ticker symbol
            statement_type: Type of statement ('income', 'balance', 'cash')
            period: Period ('annual' or 'quarterly')
            
        Returns:
            DataFrame with financial statement data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        try:
            self.logger.info(f"Retrieving {period} {statement_type} statement for {symbol}")
            
            # Map statement type to API parameter
            statement_map = {
                "income": "income-statement",
                "balance": "balance-sheet",
                "cash": "cash-flow"
            }
            
            # Make API request
            data = self._make_request(
                f"securities/{symbol}/financials/{statement_map.get(statement_type, 'income-statement')}",
                params={"period": period}
            )
            
            # Transform the data to a DataFrame
            statements = data.get('statements', [])
            if not statements:
                return pd.DataFrame()
            
            # Extract dates and create columns
            dates = [stmt.get('date') for stmt in statements]
            
            # Create a dictionary of line items
            line_items = {}
            for stmt in statements:
                date = stmt.get('date')
                for item_name, item_value in stmt.get('items', {}).items():
                    if item_name not in line_items:
                        line_items[item_name] = {}
                    line_items[item_name][date] = item_value
            
            # Create DataFrame
            df = pd.DataFrame(line_items).T
            
            return df
            
        except Exception as e:
            error_msg = f"Failed to retrieve {statement_type} statement for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            raise DataRetrievalError(
                error_msg,
                symbol=symbol,
                data_source=self.name,
                original_exception=e
            )
    
    def get_technical_indicators(self, symbol: str, indicators: Optional[List[str]] = None) -> Dict[str, Any]:
        """Retrieve technical indicators.
        
        Args:
            symbol: Security ticker symbol
            indicators: List of indicators to retrieve (e.g., ['rsi_14', 'sma_50', 'macd'])
            
        Returns:
            Dictionary containing technical indicators
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        try:
            self.logger.info(f"Retrieving technical indicators for {symbol}")
            
            # Default indicators if none specified
            if indicators is None:
                indicators = ['rsi_14', 'sma_20', 'sma_50', 'sma_200', 'macd']
            
            # Make API request
            data = self._make_request(
                f"securities/{symbol}/indicators",
                params={"indicators": ",".join(indicators)}
            )
            
            # Transform the data to match the expected format
            result = {}
            
            # Process RSI
            if 'rsi' in data:
                result['rsi_14'] = data['rsi'].get('value')
            
            # Process moving averages
            moving_averages = {}
            for ma in ['sma_20', 'sma_50', 'sma_200']:
                if ma in data:
                    moving_averages[ma] = data[ma].get('value')
            
            if moving_averages:
                result['moving_averages'] = moving_averages
            
            # Process MACD
            if 'macd' in data:
                result['macd'] = {
                    'macd_line': data['macd'].get('macdLine'),
                    'signal_line': data['macd'].get('signalLine'),
                    'histogram': data['macd'].get('histogram')
                }
            
            # Add source information
            result['_sources'] = {'technical_indicators': self.name}
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to retrieve technical indicators for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            raise DataRetrievalError(
                error_msg,
                symbol=symbol,
                data_source=self.name,
                original_exception=e
            )
    
    def get_market_data(self, data_type: str) -> Dict[str, Any]:
        """Retrieve market data like indices, commodities, forex.
        
        Args:
            data_type: Type of market data ('indices', 'commodities', 'forex', 'sectors', 'economic')
            
        Returns:
            Dictionary containing market data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        try:
            self.logger.info(f"Retrieving market data: {data_type}")
            
            # Make API request
            data = self._make_request(f"market/{data_type}")
            
            # Transform the data based on data type
            if data_type == 'indices':
                result = {}
                for index in data.get('indices', []):
                    result[index['name']] = {
                        'value': index.get('value'),
                        'change': index.get('change'),
                        'change_percent': index.get('changePercent')
                    }
                return result
                
            elif data_type == 'commodities':
                result = {}
                for commodity in data.get('commodities', []):
                    result[commodity['name']] = {
                        'value': commodity.get('value'),
                        'change': commodity.get('change'),
                        'change_percent': commodity.get('changePercent'),
                        'unit': commodity.get('unit')
                    }
                return result
                
            elif data_type == 'forex':
                result = {}
                for pair in data.get('forexPairs', []):
                    result[pair['name']] = {
                        'value': pair.get('value'),
                        'change': pair.get('change'),
                        'change_percent': pair.get('changePercent')
                    }
                return result
                
            elif data_type == 'sectors':
                result = {}
                for sector in data.get('sectors', []):
                    result[sector['name']] = sector.get('performance')
                return result
                
            elif data_type == 'economic':
                result = {}
                for indicator in data.get('indicators', []):
                    result[indicator['name']] = {
                        'value': indicator.get('value'),
                        'previous': indicator.get('previous'),
                        'forecast': indicator.get('forecast'),
                        'unit': indicator.get('unit')
                    }
                return result
                
            else:
                return {}
            
        except Exception as e:
            error_msg = f"Failed to retrieve market data ({data_type}): {str(e)}"
            self.logger.error(error_msg)
            raise DataRetrievalError(
                error_msg,
                data_type=data_type,
                data_source=self.name,
                original_exception=e
            )
    
    def get_news(self, symbol: Optional[str] = None, category: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve news articles.
        
        Args:
            symbol: Security ticker symbol (optional)
            category: News category (optional)
            limit: Maximum number of news items to retrieve
            
        Returns:
            List of dictionaries containing news articles
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        try:
            params = {"limit": limit}
            
            if symbol:
                self.logger.info(f"Retrieving news for {symbol}")
                endpoint = f"securities/{symbol}/news"
            else:
                self.logger.info("Retrieving market news")
                endpoint = "news"
                if category:
                    params["category"] = category
            
            # Make API request
            data = self._make_request(endpoint, params=params)
            
            # Transform the data to match the expected format
            news_items = []
            for item in data.get('news', []):
                news_items.append({
                    'title': item.get('title'),
                    'source': item.get('source'),
                    'url': item.get('url'),
                    'published_at': datetime.fromisoformat(item.get('publishedAt')),
                    'summary': item.get('summary'),
                    'sentiment': item.get('sentiment'),
                    'impact': item.get('impact'),
                    'categories': item.get('categories')
                })
            
            return news_items
            
        except Exception as e:
            error_msg = f"Failed to retrieve news: {str(e)}"
            self.logger.error(error_msg)
            raise DataRetrievalError(
                error_msg,
                symbol=symbol,
                data_source=self.name,
                original_exception=e
            )
    
    def get_economic_data(self, indicator: str, region: Optional[str] = None) -> pd.DataFrame:
        """Retrieve economic indicators data.
        
        Args:
            indicator: Economic indicator name
            region: Region or country code (optional)
            
        Returns:
            DataFrame with economic indicator data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        try:
            self.logger.info(f"Retrieving economic data: {indicator}" + (f" for {region}" if region else ""))
            
            # Prepare parameters
            params = {"indicator": indicator}
            if region:
                params["region"] = region
            
            # Make API request
            data = self._make_request("economic", params=params)
            
            # Transform the data to a DataFrame
            history = data.get('history', [])
            if not history:
                return pd.DataFrame()
            
            df = pd.DataFrame(history)
            
            # Convert date column to datetime and set as index
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Sort by date
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            error_msg = f"Failed to retrieve economic data ({indicator}): {str(e)}"
            self.logger.error(error_msg)
            raise DataRetrievalError(
                error_msg,
                indicator=indicator,
                region=region,
                data_source=self.name,
                original_exception=e
            )
```

## Implementation Example: Web Scraping Adapter

For data sources without an API, you can implement a web scraping adapter:

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from stock_analysis.adapters.base_adapter import FinancialDataAdapter
from stock_analysis.utils.exceptions import DataRetrievalError
from stock_analysis.utils.logging import get_logger

class WebScrapingAdapter(FinancialDataAdapter):
    """Adapter for web scraping financial data."""
    
    def __init__(self):
        """Initialize the web scraping adapter."""
        self.name = "web_scraping"
        self.logger = get_logger(f"{__name__}.WebScrapingAdapter")
        self.base_url = "https://example.com"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def _fetch_page(self, url: str) -> BeautifulSoup:
        """Fetch a web page and parse it with BeautifulSoup.
        
        Args:
            url: URL to fetch
            
        Returns:
            BeautifulSoup object
            
        Raises:
            DataRetrievalError: If page cannot be fetched
        """
        try:
            self.logger.debug(f"Fetching page: {url}")
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            error_msg = f"Failed to fetch page {url}: {str(e)}"
            self.logger.error(error_msg)
            raise DataRetrievalError(error_msg, data_source=self.name, original_exception=e)
    
    def get_security_info(self, symbol: str) -> Dict[str, Any]:
        """Retrieve basic security information by scraping a web page.
        
        Args:
            symbol: Security ticker symbol
            
        Returns:
            Dictionary containing security information
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        try:
            self.logger.info(f"Retrieving security info for {symbol}")
            
            # Fetch the security page
            url = f"{self.base_url}/stocks/{symbol}"
            soup = self._fetch_page(url)
            
            # Extract data using BeautifulSoup selectors
            name = soup.select_one('.company-name').text.strip()
            current_price = float(soup.select_one('.current-price').text.strip().replace('$', ''))
            
            # Extract additional data
            market_cap_elem = soup.select_one('.market-cap')
            market_cap = float(market_cap_elem.text.strip().replace('$', '').replace('B', 'e9').replace('M', 'e6')) if market_cap_elem else None
            
            pe_ratio_elem = soup.select_one('.pe-ratio')
            pe_ratio = float(pe_ratio_elem.text.strip()) if pe_ratio_elem else None
            
            sector_elem = soup.select_one('.sector')
            sector = sector_elem.text.strip() if sector_elem else None
            
            industry_elem = soup.select_one('.industry')
            industry = industry_elem.text.strip() if industry_elem else None
            
            # Return the extracted data
            return {
                'symbol': symbol,
                'name': name,
                'current_price': current_price,
                'market_cap': market_cap,
                'pe_ratio': pe_ratio,
                'sector': sector,
                'industry': industry
            }
            
        except Exception as e:
            error_msg = f"Failed to retrieve security info for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            raise DataRetrievalError(
                error_msg,
                symbol=symbol,
                data_source=self.name,
                original_exception=e
            )
    
    # Implement other methods similarly...
```

## Integration with the System

### 1. Register the Adapter

Add your adapter to the `FinancialDataIntegrationService`:

```python
from stock_analysis.adapters.rest_api_adapter import RestApiAdapter

def _initialize_default_adapters(self) -> List[FinancialDataAdapter]:
    """Initialize default adapters."""
    adapters = []
    
    # Add existing adapters
    try:
        adapters.append(YFinanceAdapter())
        self.logger.info("Initialized YFinance adapter")
    except Exception as e:
        self.logger.warning(f"Failed to initialize YFinance adapter: {str(e)}")
    
    # Add your custom adapter
    try:
        adapters.append(RestApiAdapter())
        self.logger.info("Initialized REST API adapter")
    except Exception as e:
        self.logger.warning(f"Failed to initialize REST API adapter: {str(e)}")
    
    return adapters
```

### 2. Configure Source Priorities

Update the source priorities in the configuration:

```python
def _load_source_priorities(self) -> Dict[str, Dict[str, int]]:
    """Load source priorities from configuration."""
    # Default priorities (lower number = higher priority)
    default_priorities = {
        "security_info": {"investing": 1, "rest_api": 2, "alpha_vantage": 3, "yfinance": 4},
        "historical_prices": {"yfinance": 1, "rest_api": 2, "investing": 3, "alpha_vantage": 4},
        "financial_statements": {"rest_api": 1, "investing": 2, "alpha_vantage": 3, "yfinance": 4},
        # Add other data types...
    }
    
    # Load priorities from config
    priorities = {}
    for data_type, default_priority in default_priorities.items():
        config_key = f"stock_analysis.integration.priorities.{data_type}"
        config_priority = config.get(config_key, default_priority)
        priorities[data_type] = config_priority
    
    return priorities
```

### 3. Update Configuration

Add configuration options for your adapter:

```yaml
stock_analysis:
  data_sources:
    rest_api:
      enabled: true
      api_key: "YOUR_API_KEY"
      timeout: 30
```

## Advanced Adapter Features

### 1. Rate Limiting

Implement rate limiting to avoid hitting API limits:

```python
import time
from functools import wraps

def rate_limit(calls_per_second=1):
    """Decorator to implement rate limiting."""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

class RestApiAdapter(FinancialDataAdapter):
    # ...
    
    @rate_limit(calls_per_second=2)
    def _make_request(self, endpoint, params=None):
        # Implementation
        pass
```

### 2. Caching

Implement caching to improve performance:

```python
from stock_analysis.utils.cache_manager import get_cache_manager

class RestApiAdapter(FinancialDataAdapter):
    def __init__(self):
        # ...
        self.cache = get_cache_manager()
        self.cache_ttl = {
            "security_info": 3600,  # 1 hour
            "historical_prices": 3600,  # 1 hour
            "financial_statements": 86400,  # 24 hours
            "technical_indicators": 1800,  # 30 minutes
            "market_data": 300,  # 5 minutes
            "news": 300,  # 5 minutes
            "economic_data": 3600,  # 1 hour
        }
    
    def get_security_info(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        cache_key = f"{self.name}_security_info_{symbol}"
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                self.logger.debug(f"Using cached security info for {symbol}")
                return cached_data
        
        # Retrieve data
        data = self._retrieve_security_info(symbol)
        
        # Cache data
        if use_cache:
            self.cache.set(
                cache_key, 
                data, 
                ttl=self.cache_ttl.get("security_info"),
                data_type="security_info",
                tags=[symbol, self.name]
            )
        
        return data
```

### 3. Error Recovery

Implement error recovery for transient failures:

```python
from stock_analysis.utils.error_recovery import with_error_recovery, with_data_validation

class RestApiAdapter(FinancialDataAdapter):
    # ...
    
    @with_error_recovery("get_security_info", retry_category="data_retrieval")
    @with_data_validation("security_info")
    def get_security_info(self, symbol: str) -> Dict[str, Any]:
        # Implementation
        pass
```

### 4. Data Validation

Implement data validation to ensure data quality:

```python
from stock_analysis.utils.data_quality import validate_data_source_response

class RestApiAdapter(FinancialDataAdapter):
    # ...
    
    def get_security_info(self, symbol: str) -> Dict[str, Any]:
        # Retrieve data
        data = self._retrieve_security_info(symbol)
        
        # Validate data
        validate_data_source_response(data, "security_info", symbol, self.name)
        
        return data
```

## Testing Your Adapter

### 1. Unit Tests

Create unit tests for your adapter:

```python
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from stock_analysis.adapters.rest_api_adapter import RestApiAdapter
from stock_analysis.utils.exceptions import DataRetrievalError

class TestRestApiAdapter(unittest.TestCase):
    """Test cases for RestApiAdapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = RestApiAdapter(api_key="test_key")
    
    @patch('requests.get')
    def test_get_security_info(self, mock_get):
        """Test retrieving security information."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'companyName': 'Test Company',
            'price': 100.0,
            'marketCap': 1000000000,
            'peRatio': 20.5,
            'sector': 'Technology',
            'industry': 'Software'
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Call the method
        result = self.adapter.get_security_info('TEST')
        
        # Verify the result
        self.assertEqual(result['symbol'], 'TEST')
        self.assertEqual(result['name'], 'Test Company')
        self.assertEqual(result['current_price'], 100.0)
        self.assertEqual(result['market_cap'], 1000000000)
        self.assertEqual(result['pe_ratio'], 20.5)
        self.assertEqual(result['sector'], 'Technology')
        self.assertEqual(result['industry'], 'Software')
```

### 2. Integration Tests

Create integration tests for your adapter:

```python
import unittest
import pandas as pd
from stock_analysis.adapters.rest_api_adapter import RestApiAdapter
from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService

class TestRestApiAdapterIntegration(unittest.TestCase):
    """Integration tests for RestApiAdapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = RestApiAdapter()
        self.integration_service = FinancialDataIntegrationService([self.adapter])
    
    def test_get_security_info_integration(self):
        """Test retrieving security information through the integration service."""
        # Skip if no API key is available
        if not self.adapter.api_key:
            self.skipTest("No API key available for testing")
        
        # Call the method
        result = self.integration_service.get_security_info('AAPL')
        
        # Verify the result
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertIsNotNone(result['name'])
        self.assertIsNotNone(result['current_price'])
        self.assertIsNotNone(result['market_cap'])
```

## Best Practices

### 1. Error Handling

- Always catch and properly handle exceptions
- Convert exceptions to `DataRetrievalError` with context
- Log detailed error information
- Implement retry logic for transient failures

### 2. Data Normalization

- Normalize data to match the expected format
- Handle missing or null values gracefully
- Convert data types as needed (e.g., strings to numbers)
- Ensure consistent date formats

### 3. Performance Optimization

- Use caching to reduce API calls
- Implement rate limiting to avoid hitting API limits
- Use parallel processing for batch operations
- Optimize network requests (e.g., combine multiple requests)

### 4. Security

- Store API keys securely in configuration
- Use HTTPS for all API requests
- Validate and sanitize input data
- Handle sensitive data appropriately

### 5. Documentation

- Document the adapter's capabilities and limitations
- Document the data source's API and rate limits
- Document any special handling or transformations
- Document configuration options