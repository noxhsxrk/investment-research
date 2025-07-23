# Stock Analysis CLI Usage Examples

## Single Stock Analysis

```bash
# Analyze a single stock with default settings
stock-analysis analyze AAPL

# Analyze with verbose output and custom export format
stock-analysis analyze AAPL --verbose --export-format json --output my_analysis

# Analyze without exporting results
stock-analysis analyze AAPL --no-export

# Analyze with technical indicators
stock-analysis analyze AAPL --include-technicals

# Analyze with analyst recommendations and price targets
stock-analysis analyze AAPL --include-analyst

# Analyze with both technical indicators and analyst data
stock-analysis analyze AAPL --include-technicals --include-analyst
```

## Comprehensive Analysis

```bash
# Comprehensive analysis of a single stock with default settings
stock-analysis comprehensive AAPL

# Comprehensive analysis with sentiment analysis for news
stock-analysis comprehensive AAPL --sentiment

# Comprehensive analysis with custom export format
stock-analysis comprehensive AAPL --export-format json --output apple_comprehensive

# Comprehensive analysis focusing only on specific data categories
stock-analysis comprehensive AAPL --skip-financials
stock-analysis comprehensive AAPL --skip-news
stock-analysis comprehensive AAPL --skip-analysis

# Comprehensive analysis with specific financial statement options
stock-analysis comprehensive AAPL --statement income --period quarterly --years 3

# Comprehensive analysis with custom news options
stock-analysis comprehensive AAPL --news-limit 20 --news-days 14

# Comprehensive analysis with technical indicators and analyst data
stock-analysis comprehensive AAPL --include-technicals --include-analyst

# Comprehensive analysis of multiple stocks with parallel processing
stock-analysis comprehensive AAPL MSFT GOOGL --parallel --max-workers 6
```

## Multiple Stock Analysis

```bash
# Analyze multiple stocks
stock-analysis analyze AAPL MSFT GOOGL --verbose

# Analyze with custom export settings
stock-analysis analyze AAPL MSFT GOOGL --export-format excel --output tech_stocks

# Analyze multiple stocks with enhanced data
stock-analysis analyze AAPL MSFT GOOGL --include-technicals --include-analyst
```

## Batch Analysis

```bash
# Analyze stocks from a file
stock-analysis batch example_stocks.txt

# Batch analysis with custom settings
stock-analysis batch example_stocks.txt --export-format csv --max-workers 8 --verbose

# Batch analysis with custom output
stock-analysis batch example_stocks.txt --output quarterly_analysis --export-format excel
```

## Financial Statements

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

## Market Data

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

## Financial News

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

## Scheduling

```bash
# Add a daily scheduled job
stock-analysis schedule add daily-tech "AAPL,MSFT,GOOGL" --interval daily --name "Daily Tech Analysis"

# Add a weekly job with notifications disabled
stock-analysis schedule add weekly-finance "JPM,BAC,WFC" --interval weekly --no-notifications

# List all scheduled jobs
stock-analysis schedule list

# Show scheduler status
stock-analysis schedule status

# Show specific job status
stock-analysis schedule status daily-tech

# Run a job immediately
stock-analysis schedule run daily-tech

# Enable/disable jobs
stock-analysis schedule enable daily-tech
stock-analysis schedule disable daily-tech

# Remove a job
stock-analysis schedule remove daily-tech

# Start/stop the scheduler
stock-analysis schedule start
stock-analysis schedule stop

# Generate scheduler report
stock-analysis schedule report --days 30
```

## Configuration Management

```bash
# Show all configuration
stock-analysis config show

# Show specific configuration key
stock-analysis config show stock_analysis.export.default_format

# Set configuration values
stock-analysis config set stock_analysis.export.default_format json
stock-analysis config set stock_analysis.data_sources.yfinance.timeout 60

# Validate configuration
stock-analysis config validate

# Reset to defaults
stock-analysis config reset
```

## Global Options

```bash
# Use custom configuration file
stock-analysis --config my_config.yaml analyze AAPL

# Set log level
stock-analysis --log-level DEBUG analyze AAPL --verbose

# Verbose output for any command
stock-analysis --verbose batch example_stocks.txt
```

## File Formats

### Stock Symbol Files

Text file (one symbol per line):

```
AAPL
MSFT
GOOGL
# Comments are supported
AMZN
```

CSV file (comma-separated):

```
AAPL,MSFT,GOOGL
JPM,BAC,WFC
JNJ,PFE,UNH
```

### Export Formats

- **CSV**: Flattened data suitable for spreadsheet analysis
- **Excel**: Multi-sheet workbook with organized data
- **JSON**: Structured data with metadata for Power BI integration

## Error Handling

The CLI includes comprehensive error handling:

- Invalid stock symbols are reported but don't stop batch processing
- Network timeouts are retried automatically
- Configuration validation prevents invalid settings
- Verbose mode shows detailed error information

## Comprehensive Analysis Workflow

The comprehensive command provides a unified workflow for analyzing stocks:

### Basic Usage

```bash
# Single stock comprehensive analysis
stock-analysis comprehensive AAPL

# Multiple stocks comprehensive analysis
stock-analysis comprehensive AAPL MSFT GOOGL
```

### Customizing Data Categories

```bash
# Skip specific data categories
stock-analysis comprehensive AAPL --skip-financials
stock-analysis comprehensive AAPL --skip-news
stock-analysis comprehensive AAPL --skip-analysis

# Focus only on financial statements and news (skip analysis)
stock-analysis comprehensive AAPL --skip-analysis
```

### Analysis Options

```bash
# Include technical indicators (RSI, MACD, moving averages)
stock-analysis comprehensive AAPL --include-technicals

# Include analyst recommendations and price targets
stock-analysis comprehensive AAPL --include-analyst

# Include both technical indicators and analyst data
stock-analysis comprehensive AAPL --include-technicals --include-analyst
```

### Financial Statement Options

```bash
# Get only income statements
stock-analysis comprehensive AAPL --statement income

# Get quarterly financial statements
stock-analysis comprehensive AAPL --period quarterly

# Get 3 years of historical data
stock-analysis comprehensive AAPL --years 3

# Combine financial statement options
stock-analysis comprehensive AAPL --statement income --period quarterly --years 2
```

### News Options

```bash
# Get more news items
stock-analysis comprehensive AAPL --news-limit 20

# Look back further for news
stock-analysis comprehensive AAPL --news-days 14

# Include sentiment analysis
stock-analysis comprehensive AAPL --sentiment

# Combine news options
stock-analysis comprehensive AAPL --news-limit 15 --news-days 10 --sentiment
```

### Export Options

```bash
# Export to different formats
stock-analysis comprehensive AAPL --export-format json
stock-analysis comprehensive AAPL --export-format csv
stock-analysis comprehensive AAPL --export-format excel

# Specify output filename
stock-analysis comprehensive AAPL --output apple_analysis

# Skip exporting results
stock-analysis comprehensive AAPL --no-export
```

### Performance Optimization

```bash
# Enable parallel processing (default)
stock-analysis comprehensive AAPL MSFT GOOGL --parallel

# Increase worker threads for better performance
stock-analysis comprehensive AAPL MSFT GOOGL --max-workers 8

# Disable parallel processing
stock-analysis comprehensive AAPL MSFT GOOGL --no-parallel
```

### Combined Examples

```bash
# Full analysis with all options
stock-analysis comprehensive AAPL MSFT --include-technicals --include-analyst --statement all --period quarterly --years 3 --news-limit 15 --news-days 10 --sentiment --export-format excel --output tech_stocks --parallel --max-workers 6

# Focused financial analysis
stock-analysis comprehensive AAPL MSFT GOOGL --skip-news --statement income --period quarterly --years 5 --export-format excel --output tech_financials

# News-only analysis
stock-analysis comprehensive AAPL MSFT GOOGL --skip-analysis --skip-financials --news-limit 20 --news-days 14 --sentiment --export-format json --output tech_news
```
