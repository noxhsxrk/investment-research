# Stock Analysis Dashboard

A comprehensive Python-based stock analysis system that retrieves financial data from multiple sources including yfinance, Investing.com, and Alpha Vantage. The system integrates with Power BI for visualization and analyzes multiple aspects of stocks including news sentiment, financial metrics, company health indicators, and fair value calculations.

## Features

- **Multi-Source Data Integration**: Retrieves data from yfinance, Investing.com, and Alpha Vantage with fallback logic
- **Enhanced Stock Data**: Comprehensive security information including technical indicators and analyst recommendations
- **Detailed Financial Statements**: Income statements, balance sheets, and cash flow statements with growth metrics
- **Market Data**: Major indices, sector performance, commodities, forex, and economic indicators
- **Financial News**: Company news, market news, and economic calendar with sentiment analysis
- **Financial Analysis**: Calculates key financial ratios and health scores
- **Valuation Models**: Implements DCF, peer comparison, and PEG ratio models
- **Sentiment Analysis**: Analyzes news sentiment for market perception insights
- **Power BI Integration**: Exports data in Power BI compatible formats
- **Automated Scheduling**: Supports scheduled analysis runs
- **Comprehensive Logging**: Detailed logging and error handling
- **Performance Optimization**: Caching, parallel processing, and source prioritization

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd stock-analysis-dashboard
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install the package:

```bash
pip install -e .
```

## Configuration

The system uses a YAML configuration file (`config.yaml`) for settings. You can also override settings using environment variables:

- `STOCK_ANALYSIS_YFINANCE_TIMEOUT`: API timeout in seconds
- `STOCK_ANALYSIS_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `STOCK_ANALYSIS_OUTPUT_DIR`: Output directory for exports
- `STOCK_ANALYSIS_NOTIFICATION_EMAIL`: Email for notifications

## Usage

### Command Line Interface

```bash
# Analyze a single stock
stock-analysis analyze AAPL

# Analyze with technical indicators and analyst data
stock-analysis analyze AAPL --include-technicals --include-analyst

# Analyze multiple stocks
stock-analysis analyze AAPL MSFT GOOGL

# Get detailed financial statements
stock-analysis financials AAPL --statement income --period quarterly --growth

# Get market data
stock-analysis market --indices --sectors --commodities

# Get financial news
stock-analysis news --symbol AAPL --sentiment

# Run scheduled analysis
stock-analysis schedule add daily-tech "AAPL,MSFT,GOOGL" --interval daily

# Export results to Power BI format
stock-analysis export --format powerbi
```

### Python API

```python
from stock_analysis import StockAnalysisOrchestrator
from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
from stock_analysis.services.market_data_service import MarketDataService
from stock_analysis.services.news_service import NewsService

# Initialize orchestrator
orchestrator = StockAnalysisOrchestrator(include_technicals=True, include_analyst=True)

# Analyze a stock
result = orchestrator.analyze_single_security('AAPL')

# Export results
orchestrator.export_results(result, format='excel')

# Use integration service for advanced data retrieval
integration_service = FinancialDataIntegrationService()
security_info = integration_service.get_enhanced_security_info('AAPL')
financial_statements = integration_service.get_financial_statements('AAPL', 'income', 'annual')

# Get market data
market_service = MarketDataService()
market_overview = market_service.get_market_overview()

# Get financial news
news_service = NewsService()
company_news = news_service.get_company_news('AAPL')
```

## Project Structure

```
stock_analysis/
├── models/          # Data models and structures
├── services/        # Data retrieval services
├── analyzers/       # Analysis engines
├── exporters/       # Export services
└── utils/           # Utilities and configuration

tests/               # Test suite
exports/             # Output directory
logs/                # Log files
```

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=stock_analysis
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
