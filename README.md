# Stock Analysis Dashboard

A comprehensive Python-based stock analysis system that retrieves financial data using the yfinance library and integrates with Power BI for visualization. The system analyzes multiple aspects of stocks including news sentiment, financial metrics, company health indicators, and fair value calculations.

## Features

- **Stock Data Retrieval**: Uses yfinance to get current and historical stock data
- **Financial Analysis**: Calculates key financial ratios and health scores
- **Valuation Models**: Implements DCF, peer comparison, and PEG ratio models
- **Sentiment Analysis**: Analyzes news sentiment for market perception insights
- **Power BI Integration**: Exports data in Power BI compatible formats
- **Automated Scheduling**: Supports scheduled analysis runs
- **Comprehensive Logging**: Detailed logging and error handling

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

# Analyze multiple stocks
stock-analysis analyze AAPL MSFT GOOGL

# Run scheduled analysis
stock-analysis schedule --interval daily

# Export results to Power BI format
stock-analysis export --format powerbi
```

### Python API

```python
from stock_analysis import StockAnalysisOrchestrator

# Initialize analyzer
analyzer = StockAnalysisOrchestrator()

# Analyze a stock
result = analyzer.analyze_stock('AAPL')

# Export results
analyzer.export_results(result, format='excel')
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
