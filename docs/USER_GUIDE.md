# Stock Analysis User Guide

## Introduction

The Stock Analysis Dashboard is a comprehensive Python-based tool for analyzing stocks and financial markets. This guide covers the enhanced data features added through the Investing.com Data Integration, providing access to detailed financial statements, market data, technical indicators, and news sentiment analysis.

## Installation and Setup

1. Install the package:

```bash
pip install -r requirements.txt
pip install -e .
```

2. Configure your data sources in `config.yaml`:

```yaml
stock_analysis:
  data_sources:
    yfinance:
      enabled: true
      timeout: 30
    investing:
      enabled: true
      timeout: 30
    alpha_vantage:
      enabled: true
      api_key: "YOUR_API_KEY"  # Replace with your Alpha Vantage API key
      timeout: 30
  integration:
    priorities:
      security_info:
        investing: 1
        alpha_vantage: 2
        yfinance: 3
    cache_ttl:
      security_info: 3600        # 1 hour
      historical_prices: 3600    # 1 hour
      financial_statements: 86400 # 24 hours
      market_data: 300           # 5 minutes
      news: 300                  # 5 minutes
    parallel_enabled: true
    max_workers: 4
```

You can obtain an Alpha Vantage API key by signing up at [Alpha Vantage](https://www.alphavantage.co/support/#api-key). The free tier provides 5 API calls per minute and 500 calls per day.

You can specify a custom configuration file when running commands:

```bash
stock-analysis --config my_custom_config.yaml analyze AAPL
```

## Command Line Interface

### Enhanced Stock Analysis

The `analyze` command has been enhanced with additional options for technical indicators and analyst data:

```bash
# Basic stock analysis
stock-analysis analyze AAPL

# Analysis with technical indicators
stock-analysis analyze AAPL --include-technicals

# Analysis with analyst recommendations and price targets
stock-analysis analyze AAPL --include-analyst

# Analysis with both technical indicators and analyst data
stock-analysis analyze AAPL --include-technicals --include-analyst

# Analyze multiple stocks with enhanced data
stock-analysis analyze AAPL MSFT GOOGL --include-technicals --include-analyst
```

### Financial Statements

The new `financials` command provides access to detailed financial statements:

```bash
# Get all financial statements (income, balance, cash flow)
stock-analysis financials AAPL

# Get specific financial statement
stock-analysis financials AAPL --statement income
stock-analysis financials AAPL --statement balance
stock-analysis financials AAPL --statement cash

# Get quarterly financial statements
stock-analysis financials AAPL --period quarterly

# Get financial statements with growth metrics
stock-analysis financials AAPL --growth

# Get financial statements with industry comparison
stock-analysis financials AAPL --compare-industry

# Export financial statements
stock-analysis financials AAPL --export-format excel --output apple_financials
```

### Market Data

The new `market` command provides access to market-wide data:

```bash
# Get comprehensive market overview
stock-analysis market

# Get specific market data
stock-analysis market --indices
stock-analysis market --sectors
stock-analysis market --commodities
stock-analysis market --forex
stock-analysis market --economic

# Get economic indicators for specific region
stock-analysis market --economic --region US

# Export market data
stock-analysis market --indices --sectors --export-format json --output market_overview
```

### Financial News

The new `news` command provides access to financial news and economic events:

```bash
# Get news for a specific stock
stock-analysis news --symbol AAPL

# Get general market news
stock-analysis news --market

# Get news by category
stock-analysis news --market --category earnings
stock-analysis news --market --category economy
stock-analysis news --market --category technology

# Get economic calendar
stock-analysis news --economic-calendar

# Get news with sentiment analysis
stock-analysis news --symbol AAPL --sentiment

# Get trending topics
stock-analysis news --trending

# Control the number of news items
stock-analysis news --symbol AAPL --limit 20

# Export news data
stock-analysis news --symbol AAPL --export-format json --output apple_news
```

## Data Types and Features

### Enhanced Security Information

Enhanced security information includes:

- **Basic Information**: Symbol, name, current price, market cap, etc.
- **Technical Indicators**: RSI, MACD, moving averages
- **Analyst Data**: Analyst ratings, price targets, buy/hold/sell recommendations
- **Additional Metadata**: Exchange, currency, company description, key executives

Example output:

```
Company: Apple Inc.
Current Price: $150.25
Sector: Technology
Industry: Consumer Electronics
P/E Ratio: 28.5
P/B Ratio: 35.2

Enhanced Data:

Technical Indicators:
  RSI (14): 65.4
  Moving Averages:
    SMA 20: $148.50
    SMA 50: $145.20
    SMA 200: $140.80
  MACD:
    MACD Line: 2.5000
    Signal Line: 1.8000
    Histogram: 0.7000

Analyst Data:
  Analyst Rating: Buy
  Price Target: $165.00 (Low: $140.00, High: $190.00)
  Analyst Count: 32

Exchange: NASDAQ
Earnings Growth: 15.5%
Revenue Growth: 12.8%
```

Note: The sector, industry, P/E ratio, and P/B ratio are now included in the EnhancedSecurityInfo model.

### Financial Statements

Enhanced financial statements include:

- **Income Statements**: Revenue, expenses, net income
- **Balance Sheets**: Assets, liabilities, equity
- **Cash Flow Statements**: Operating, investing, financing cash flows
- **Growth Metrics**: Year-over-year growth rates
- **Industry Comparisons**: Comparison with industry and sector averages

### Market Data

Market data includes:

- **Market Indices**: S&P 500, NASDAQ, Dow Jones, etc.
- **Sector Performance**: Performance metrics for different market sectors
- **Commodities**: Gold, oil, natural gas, etc.
- **Forex**: Major currency pairs
- **Economic Indicators**: GDP, inflation, unemployment, interest rates

### Financial News

Financial news includes:

- **Company News**: News articles related to specific companies
- **Market News**: General market news and headlines
- **Economic Calendar**: Upcoming economic events and their expected impact
- **Sentiment Analysis**: Positive/negative sentiment scores for news articles
- **Categorization**: News categorized by topic and impact level

## Advanced Features

### Data Source Prioritization

The system automatically prioritizes data sources based on reliability and performance. You can customize the priorities in the configuration:

```yaml
stock_analysis:
  integration:
    priorities:
      security_info:
        investing: 1
        alpha_vantage: 2
        yfinance: 3
      historical_prices:
        yfinance: 1
        investing: 2
        alpha_vantage: 3
```

### Caching

The system caches frequently accessed data to improve performance. You can configure cache settings:

```yaml
stock_analysis:
  integration:
    cache_ttl:
      security_info: 3600        # 1 hour
      historical_prices: 3600    # 1 hour
      financial_statements: 86400 # 24 hours
      technical_indicators: 1800 # 30 minutes
      market_data: 300           # 5 minutes
      news: 300                  # 5 minutes
```

### Parallel Processing

For batch operations, the system uses parallel processing to improve performance:

```yaml
stock_analysis:
  integration:
    parallel_enabled: true
    max_workers: 4
    batch_size: 10
```

## Troubleshooting

### Common Issues

1. **Data Source Errors**:
   - Check your internet connection
   - Verify API keys in configuration
   - Check if the data source is available

2. **Rate Limiting**:
   - Some data sources have rate limits
   - Use caching to reduce API calls
   - Increase timeout settings in configuration

3. **Missing Data**:
   - Some data may not be available for certain securities
   - Try alternative data sources
   - Check if the security symbol is correct

### Error Messages

- **DataRetrievalError**: Error retrieving data from external sources
- **ValidationError**: Error validating data structure or values
- **CalculationError**: Error calculating derived metrics
- **ExportError**: Error exporting results

### Logging

The system provides detailed logging for troubleshooting:

```bash
# Enable debug logging
stock-analysis --log-level DEBUG analyze AAPL

# Check log files
cat logs/stock_analysis.log
cat logs/stock_analysis_errors.log
```

## Best Practices

1. **Use Caching**: Enable caching to improve performance and reduce API calls
2. **Batch Processing**: Use batch commands for analyzing multiple securities
3. **Export Results**: Export results for further analysis or visualization
4. **Regular Updates**: Keep the package updated for the latest features and bug fixes
5. **Custom Configuration**: Customize configuration for your specific needs
