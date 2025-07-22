
# Stock Analysis CLI Usage Examples

## Single Stock Analysis
```bash
# Analyze a single stock with default settings
stock-analysis analyze AAPL

# Analyze with verbose output and custom export format
stock-analysis analyze AAPL --verbose --export-format json --output my_analysis

# Analyze without exporting results
stock-analysis analyze AAPL --no-export
```

## Multiple Stock Analysis
```bash
# Analyze multiple stocks
stock-analysis analyze AAPL MSFT GOOGL --verbose

# Analyze with custom export settings
stock-analysis analyze AAPL MSFT GOOGL --export-format excel --output tech_stocks
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
