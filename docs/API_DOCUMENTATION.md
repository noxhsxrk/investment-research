# Stock Analysis API Documentation

## Overview

The Stock Analysis API provides programmatic access to comprehensive financial data and analysis capabilities. This documentation covers the enhanced features added through the Investing.com Data Integration, including detailed financial statements, market data, technical indicators, and news sentiment analysis.

## Core Services

### Financial Data Integration Service

The `FinancialDataIntegrationService` coordinates data retrieval from multiple sources with prioritization, fallback logic, and data normalization.

```python
from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService

# Initialize the service
integration_service = FinancialDataIntegrationService()

# Get comprehensive security information
security_info = integration_service.get_security_info("AAPL")

# Get enhanced security information with technical indicators and analyst data
enhanced_info = integration_service.get_enhanced_security_info(
    "AAPL", 
    include_technicals=True, 
    include_analyst_data=True
)

# Get financial statements
income_statement = integration_service.get_financial_statements(
    "AAPL", 
    statement_type="income", 
    period="annual",
    years=5  # Number of years of historical data
)

# Get technical indicators
indicators = integration_service.get_technical_indicators(
    "AAPL", 
    indicators=["rsi_14", "sma_50", "macd"]
)

# Get market data
market_data = integration_service.get_market_data(
    data_type="indices"
)

# Get news for a specific symbol
news = integration_service.get_news("AAPL", limit=10)

# Get economic indicators
economic_data = integration_service.get_economic_data(
    indicator="gdp", 
    region="US"
)
```

### Enhanced Stock Data Service

The `EnhancedStockDataService` extends the base `StockDataService` with additional capabilities for retrieving comprehensive stock data.

```python
from stock_analysis.services.enhanced_stock_data_service import EnhancedStockDataService

# Initialize the service
stock_service = EnhancedStockDataService()

# Get enhanced security information
security_info = stock_service.get_enhanced_security_info("AAPL")

# Get technical indicators
indicators = stock_service.get_technical_indicators("AAPL")

# Get analyst data
analyst_data = stock_service.get_analyst_data("AAPL")
```

### Market Data Service

The `MarketDataService` provides access to market-wide data including indices, commodities, forex, and economic indicators.

```python
from stock_analysis.services.market_data_service import MarketDataService

# Initialize the service
market_service = MarketDataService()

# Get comprehensive market overview
market_overview = market_service.get_market_overview()

# Get economic indicators
economic_indicators = market_service.get_economic_indicators(region="US")

# Get sector performance
sector_performance = market_service.get_sector_performance()

# Get commodity prices
commodities = market_service.get_commodities()

# Get foreign exchange rates
forex = market_service.get_forex_rates()
```

### News Service

The `NewsService` provides access to financial news and economic events.

```python
from stock_analysis.services.news_service import NewsService

# Initialize the service
news_service = NewsService()

# Get news for a specific company
company_news = news_service.get_company_news("AAPL", limit=10)

# Get general market news
market_news = news_service.get_market_news(category="economy", limit=10)

# Get economic calendar
economic_calendar = news_service.get_economic_calendar(days=7)
```

## Data Models

### Enhanced Security Info

The `EnhancedSecurityInfo` class extends the base `SecurityInfo` class with additional data points:

```python
from stock_analysis.models.enhanced_data_models import EnhancedSecurityInfo

# Create an instance
security_info = EnhancedSecurityInfo(
    symbol="AAPL",
    name="Apple Inc.",
    current_price=150.0,
    
    # Company name (if different from name)
    company_name="Apple Inc.",
    
    # Technical indicators
    rsi_14=65.4,
    macd={"macd_line": 2.5, "signal_line": 1.8, "histogram": 0.7},
    moving_averages={"sma_20": 148.5, "sma_50": 145.2, "sma_200": 140.8},
    
    # Analyst data
    analyst_rating="Buy",
    price_target={"low": 140.0, "average": 165.0, "high": 190.0},
    analyst_count=32,
    
    # Stock information
    pe_ratio=28.5,
    pb_ratio=15.2,
    dividend_yield=0.005,
    sector="Technology",
    industry="Consumer Electronics",
    
    # Additional metadata
    exchange="NASDAQ",
    currency="USD",
    company_description="Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
    key_executives=[
        {"name": "Tim Cook", "title": "CEO"},
        {"name": "Luca Maestri", "title": "CFO"}
    ]
)
```

### Enhanced Financial Statements

The `EnhancedFinancialStatements` class provides a comprehensive view of a company's financial statements:

```python
from stock_analysis.models.enhanced_data_models import EnhancedFinancialStatements
import pandas as pd

# Create an instance
financial_statements = EnhancedFinancialStatements(
    income_statements={
        "annual": pd.DataFrame(...),
        "quarterly": pd.DataFrame(...)
    },
    balance_sheets={
        "annual": pd.DataFrame(...),
        "quarterly": pd.DataFrame(...)
    },
    cash_flow_statements={
        "annual": pd.DataFrame(...),
        "quarterly": pd.DataFrame(...)
    },
    key_metrics={
        "revenue_growth": [0.05, 0.08, 0.12, 0.10, 0.07],
        "profit_margin": [0.21, 0.22, 0.24, 0.25, 0.26]
    },
    growth_metrics={
        "revenue": [0.05, 0.08, 0.12, 0.10, 0.07],
        "net_income": [0.06, 0.09, 0.15, 0.12, 0.08]
    },
    industry_averages={
        "revenue_growth": 0.06,
        "profit_margin": 0.20
    },
    sector_averages={
        "revenue_growth": 0.05,
        "profit_margin": 0.18
    }
)
```

### Market Data

The `MarketData` class represents market-wide data:

```python
from stock_analysis.models.enhanced_data_models import MarketData

# Create an instance
market_data = MarketData(
    indices={
        "S&P 500": {"value": 4500.0, "change": 15.0, "change_percent": 0.33},
        "NASDAQ": {"value": 14000.0, "change": 50.0, "change_percent": 0.36},
        "Dow Jones": {"value": 35000.0, "change": 100.0, "change_percent": 0.29}
    },
    commodities={
        "Gold": {"value": 1800.0, "change": 5.0, "change_percent": 0.28, "unit": "USD/oz"},
        "Oil (WTI)": {"value": 75.0, "change": -1.5, "change_percent": -2.0, "unit": "USD/bbl"}
    },
    forex={
        "EUR/USD": {"value": 1.18, "change": 0.002, "change_percent": 0.17},
        "USD/JPY": {"value": 110.5, "change": -0.3, "change_percent": -0.27}
    },
    sector_performance={
        "Technology": 0.5,
        "Healthcare": 0.3,
        "Financials": -0.2,
        "Energy": -0.4
    },
    economic_indicators={
        "GDP Growth": {"value": 2.8, "previous": 2.5, "forecast": 3.0, "unit": "%"},
        "Unemployment": {"value": 5.2, "previous": 5.4, "forecast": 5.0, "unit": "%"},
        "Inflation": {"value": 2.5, "previous": 2.3, "forecast": 2.6, "unit": "%"}
    }
)
```

### News Item

The `NewsItem` class represents financial news articles:

```python
from stock_analysis.models.enhanced_data_models import NewsItem
from datetime import datetime

# Create an instance
news_item = NewsItem(
    title="Apple Reports Record Quarterly Revenue",
    source="Financial Times",
    url="https://example.com/apple-earnings",
    published_at=datetime(2023, 7, 28, 16, 30, 0),
    summary="Apple Inc. reported record quarterly revenue of $90.1 billion, exceeding analyst expectations.",
    sentiment=0.75,  # Positive sentiment (range: -1.0 to 1.0)
    impact="high",   # High impact news
    categories=["earnings", "technology", "stocks"]
)
```

## Data Source Adapters

### Base Adapter Interface

All data source adapters implement the `FinancialDataAdapter` interface:

```python
from stock_analysis.adapters.base_adapter import FinancialDataAdapter

class CustomAdapter(FinancialDataAdapter):
    """Custom adapter for a financial data source."""
    
    def get_security_info(self, symbol):
        # Implementation
        pass
    
    def get_historical_prices(self, symbol, period, interval):
        # Implementation
        pass
    
    def get_financial_statements(self, symbol, statement_type, period, years=5):
        # Implementation
        pass
    
    def get_technical_indicators(self, symbol, indicators):
        # Implementation
        pass
    
    def get_market_data(self, data_type):
        # Implementation
        pass
    
    def get_news(self, symbol=None, category=None, limit=10):
        # Implementation
        pass
    
    def get_economic_data(self, indicator, region=None):
        # Implementation
        pass
```

### Available Adapters

The system includes the following data source adapters:

1. **YFinanceAdapter**: Wrapper around the yfinance library
2. **InvestingAdapter**: Adapter for Investing.com data
3. **AlphaVantageAdapter**: Adapter for Alpha Vantage API

### Configuration

The adapters can be configured in the `config.yaml` file:

```yaml
stock_analysis:
  data_sources:
    yfinance:
      timeout: 30
      retry_attempts: 3
      rate_limit_delay: 1.0
    investing:
      enabled: true
      timeout: 30
    alpha_vantage:
      enabled: true
      api_key: "YOUR_API_KEY"  # Replace with your Alpha Vantage API key
      timeout: 30
```

## Error Handling

The API includes comprehensive error handling:

```python
from stock_analysis.utils.exceptions import DataRetrievalError, ValidationError

try:
    data = integration_service.get_security_info("INVALID")
except DataRetrievalError as e:
    print(f"Data retrieval error: {e}")
    # Handle data retrieval error
except ValidationError as e:
    print(f"Validation error: {e}")
    # Handle validation error
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle other errors
```

## Performance Optimization

The API includes several performance optimization features:

1. **Caching**: Frequently accessed data is cached to reduce API calls
2. **Parallel Processing**: Batch operations are processed in parallel
3. **Source Prioritization**: Data sources are prioritized based on reliability and performance
4. **Fallback Logic**: If a primary data source fails, the system falls back to alternative sources

```python
# Configure cache TTL (time-to-live) in seconds
from stock_analysis.utils.config import config

config.set('stock_analysis.integration.cache_ttl.security_info', 3600)  # 1 hour
config.set('stock_analysis.integration.cache_ttl.market_data', 300)     # 5 minutes
config.set('stock_analysis.integration.cache_ttl.news', 300)            # 5 minutes

# Configure parallel processing
config.set('stock_analysis.integration.max_workers', 4)
config.set('stock_analysis.integration.parallel_enabled', True)

# Configure source priorities (lower number = higher priority)
config.set('stock_analysis.integration.priorities.security_info', {
    "investing": 1, 
    "alpha_vantage": 2, 
    "yfinance": 3
})
```