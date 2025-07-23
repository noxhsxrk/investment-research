# Stock Analysis Developer Guide

## Introduction

This guide provides information for developers who want to extend the Stock Analysis Dashboard with new data sources, enhance existing functionality, or contribute to the project. It focuses on the data integration architecture and how to implement new data source adapters.

## Architecture Overview

The Stock Analysis Dashboard follows a multi-layered architecture:

1. **Data Source Layer**: Interfaces with external financial data providers
2. **Data Integration Layer**: Normalizes and combines data from multiple sources
3. **Service Layer**: Provides domain-specific services for different types of financial data
4. **API Layer**: Exposes the enhanced data through the existing application interfaces

```
CLI Interface → Orchestrator → Data Services → Data Integration Layer → Data Source Adapters → External APIs
```

## Implementing a New Data Source Adapter

### Step 1: Create a New Adapter Class

Create a new Python file in the `stock_analysis/adapters` directory:

```python
# stock_analysis/adapters/custom_adapter.py

from typing import Dict, List, Any, Optional
import pandas as pd
from stock_analysis.adapters.base_adapter import FinancialDataAdapter
from stock_analysis.utils.exceptions import DataRetrievalError

class CustomAdapter(FinancialDataAdapter):
    """Custom adapter for a financial data source."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the custom adapter.
        
        Args:
            api_key: API key for authentication (optional)
        """
        super().__init__()
        self.name = "custom"
        self.api_key = api_key or self._get_api_key_from_config()
        self.base_url = "https://api.example.com/v1"
        
    def _get_api_key_from_config(self) -> str:
        """Get API key from configuration.
        
        Returns:
            API key string
        """
        from stock_analysis.utils.config import config
        return config.get('stock_analysis.data_sources.custom.api_key', '')
    
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
            # Implementation for retrieving security info
            # Example:
            import requests
            
            url = f"{self.base_url}/securities/{symbol}"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Transform the data to match the expected format
            return {
                'symbol': symbol,
                'name': data.get('companyName'),
                'current_price': data.get('price'),
                'market_cap': data.get('marketCap'),
                'pe_ratio': data.get('peRatio'),
                'sector': data.get('sector'),
                'industry': data.get('industry'),
                # Add any additional fields available from your data source
            }
            
        except Exception as e:
            raise DataRetrievalError(
                f"Failed to retrieve security info for {symbol} from {self.name}: {str(e)}",
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
        # Implementation for retrieving historical prices
        pass
    
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
        # Implementation for retrieving financial statements
        pass
    
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
        # Implementation for retrieving technical indicators
        pass
    
    def get_market_data(self, data_type: str) -> Dict[str, Any]:
        """Retrieve market data like indices, commodities, forex.
        
        Args:
            data_type: Type of market data ('indices', 'commodities', 'forex', 'sectors', 'economic')
            
        Returns:
            Dictionary containing market data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        # Implementation for retrieving market data
        pass
    
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
        # Implementation for retrieving news
        pass
    
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
        # Implementation for retrieving economic data
        pass
```

### Step 2: Register the Adapter

Update the `FinancialDataIntegrationService` to include your new adapter:

```python
# stock_analysis/services/financial_data_integration_service.py

from stock_analysis.adapters.custom_adapter import CustomAdapter

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
    
    # Add your custom adapter
    try:
        adapters.append(CustomAdapter())
        self.logger.info("Initialized Custom adapter")
    except Exception as e:
        self.logger.warning(f"Failed to initialize Custom adapter: {str(e)}")
    
    if not adapters:
        self.logger.error("No adapters could be initialized")
    
    return adapters
```

### Step 3: Update Configuration

Add configuration options for your adapter in `config.yaml`:

```yaml
stock_analysis:
  data_sources:
    custom:
      enabled: true
      api_key: "YOUR_API_KEY"
      timeout: 30
  integration:
    priorities:
      security_info:
        investing: 1
        alpha_vantage: 2
        yfinance: 3
        custom: 4  # Add your adapter to the priority list
```

### Step 4: Write Tests

Create tests for your adapter in the `tests` directory:

```python
# tests/test_custom_adapter.py

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from stock_analysis.adapters.custom_adapter import CustomAdapter
from stock_analysis.utils.exceptions import DataRetrievalError

class TestCustomAdapter(unittest.TestCase):
    """Test cases for CustomAdapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = CustomAdapter(api_key="test_key")
    
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
        
        # Verify the API call
        mock_get.assert_called_once_with(
            'https://api.example.com/v1/securities/TEST',
            headers={'Authorization': 'Bearer test_key'}
        )
    
    @patch('requests.get')
    def test_get_security_info_error(self, mock_get):
        """Test error handling when retrieving security information."""
        # Mock error response
        mock_get.side_effect = Exception("API error")
        
        # Call the method and verify it raises the expected exception
        with self.assertRaises(DataRetrievalError) as context:
            self.adapter.get_security_info('TEST')
        
        # Verify the error message
        self.assertIn('Failed to retrieve security info for TEST from custom', str(context.exception))
```

## Data Integration Best Practices

### 1. Error Handling

Always use proper error handling to ensure the system can recover from failures:

```python
from stock_analysis.utils.error_recovery import with_error_recovery, with_data_validation

@with_error_recovery("get_security_info", retry_category="data_retrieval")
@with_data_validation("security_info")
def get_security_info(self, symbol: str) -> Dict[str, Any]:
    # Implementation
    pass
```

### 2. Data Validation

Validate all data before returning it to ensure consistency:

```python
from stock_analysis.utils.data_quality import validate_data_source_response

def get_security_info(self, symbol: str) -> Dict[str, Any]:
    # Retrieve data
    data = self._retrieve_data(symbol)
    
    # Validate data
    validate_data_source_response(data, "security_info", symbol, self.name)
    
    return data
```

### 3. Caching

Use caching to improve performance and reduce API calls:

```python
def get_security_info(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
    cache_key = f"custom_security_info_{symbol}"
    
    # Check cache first
    if use_cache:
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data
    
    # Retrieve data
    data = self._retrieve_data(symbol)
    
    # Cache data
    if use_cache:
        self.cache.set(cache_key, data, ttl=3600)
    
    return data
```

### 4. Rate Limiting

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

@rate_limit(calls_per_second=2)
def _make_api_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    # Implementation
    pass
```

### 5. Logging

Use detailed logging to help with debugging:

```python
from stock_analysis.utils.logging import get_logger

def __init__(self):
    self.logger = get_logger(f"{__name__}.CustomAdapter")
    
def get_security_info(self, symbol: str) -> Dict[str, Any]:
    self.logger.info(f"Retrieving security info for {symbol}")
    
    try:
        # Implementation
        pass
    except Exception as e:
        self.logger.error(f"Error retrieving security info for {symbol}: {str(e)}")
        raise
```

## Extending Data Models

### Creating a New Data Model

To create a new data model, define a dataclass with validation:

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from stock_analysis.utils.exceptions import ValidationError

@dataclass
class CustomDataModel:
    """Custom data model for specific data type."""
    
    field1: str
    field2: float
    field3: Dict[str, Any]
    field4: Optional[List[str]] = None
    
    # When extending existing models, make sure to include all required fields
    # For example, when extending SecurityInfo, include company_name, sector and industry
    company_name: str = field(default="")
    sector: Optional[str] = None
    industry: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set company_name to field1 if not provided
        if not self.company_name:
            self.company_name = self.field1
    
    def validate(self) -> None:
        """Validate the data model.
        
        Raises:
            ValidationError: If any validation checks fail.
        """
        if not self.field1:
            raise ValidationError("field1 cannot be empty")
        
        if not isinstance(self.field2, (int, float)):
            raise ValidationError(f"field2 must be a number: {self.field2}")
        
        if not isinstance(self.field3, dict):
            raise ValidationError(f"field3 must be a dictionary: {self.field3}")
        
        if self.field4 is not None:
            if not isinstance(self.field4, list):
                raise ValidationError(f"field4 must be a list: {self.field4}")
            
            for item in self.field4:
                if not isinstance(item, str):
                    raise ValidationError(f"field4 items must be strings: {item}")
```

## Contributing to the Project

### Development Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd stock-analysis-dashboard
```

2. Install development dependencies:

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

3. Run tests:

```bash
pytest
```

### Code Style

Follow these code style guidelines:

1. Use PEP 8 for Python code style
2. Use type hints for function parameters and return values
3. Write docstrings for all classes and functions
4. Use meaningful variable and function names
5. Keep functions small and focused on a single task

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Documentation

Update documentation when adding new features:

1. Update API documentation with new classes and methods
2. Update the user guide with new functionality
3. Update CLI examples with new commands
4. Add developer documentation for complex features