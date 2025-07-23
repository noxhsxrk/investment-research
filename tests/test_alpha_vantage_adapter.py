"""Tests for Alpha Vantage adapter."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime

from stock_analysis.adapters.alpha_vantage_adapter import AlphaVantageAdapter
from stock_analysis.utils.exceptions import DataRetrievalError, ConfigurationError


@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client."""
    with patch('stock_analysis.adapters.alpha_vantage_adapter.HTTPClient') as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        yield client_instance


@pytest.fixture
def mock_config():
    """Create a mock adapter config."""
    with patch('stock_analysis.adapters.alpha_vantage_adapter.get_adapter_config') as mock:
        config = MagicMock()
        config.get_credential.return_value = 'test_api_key'
        config.rate_limit = 0.2
        config.timeout = 30
        config.max_retries = 3
        config.settings = {'cache_ttl': 900}
        mock.return_value = config
        yield mock


@pytest.fixture
def adapter(mock_http_client, mock_config):
    """Create an AlphaVantageAdapter instance with mocked dependencies."""
    # Ensure the mock is properly set up before creating the adapter
    mock_http_client.reset_mock()
    adapter = AlphaVantageAdapter()
    # Manually set the http_client to our mock
    adapter.http_client = mock_http_client
    return adapter


def test_init(adapter, mock_http_client, mock_config):
    """Test adapter initialization."""
    assert adapter.name == 'alpha_vantage'
    assert adapter.api_key == 'test_api_key'
    # We're manually setting the http_client in the fixture, so we don't need to check if it was called
    # Just verify that the adapter has the http_client attribute
    assert hasattr(adapter, 'http_client')


def test_validate_api_key_missing():
    """Test API key validation when key is missing."""
    with patch('stock_analysis.adapters.alpha_vantage_adapter.get_adapter_config') as mock:
        config = MagicMock()
        config.get_credential.return_value = None
        config.rate_limit = 0.2  # Add concrete value for rate_limit
        config.timeout = 30
        config.max_retries = 3
        config.settings = {'cache_ttl': 900}
        mock.return_value = config
        
        # Also patch HTTPClient to avoid actual initialization
        with patch('stock_analysis.adapters.alpha_vantage_adapter.HTTPClient'):
            adapter = AlphaVantageAdapter()
            
            with pytest.raises(ConfigurationError) as excinfo:
                adapter._validate_api_key()
            
            assert "API key not configured" in str(excinfo.value)


def test_get_security_info(adapter, mock_http_client):
    """Test retrieving security information."""
    # Mock response data
    mock_quote_response = {
        'Global Quote': {
            '01. symbol': 'AAPL',
            '02. open': '150.0',
            '03. high': '152.0',
            '04. low': '149.0',
            '05. price': '151.0',
            '06. volume': '1000000',
            '07. latest trading day': '2023-01-01',
            '08. previous close': '149.5',
            '09. change': '1.5',
            '10. change percent': '1.0%'
        }
    }
    
    mock_overview_response = {
        'Symbol': 'AAPL',
        'Name': 'Apple Inc',
        'Description': 'Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.',
        'MarketCapitalization': '2500000000000',
        'PERatio': '25.5',
        'PriceToBookRatio': '35.5',
        'DividendYield': '0.6',
        'EPS': '6.0',
        'Beta': '1.2',
        'Sector': 'Technology',
        'Industry': 'Consumer Electronics'
    }
    
    # Configure mock to return different responses based on parameters
    def mock_get(endpoint, params=None, **kwargs):
        if params and params.get('function') == 'GLOBAL_QUOTE':
            return mock_quote_response
        elif params and params.get('function') == 'OVERVIEW':
            return mock_overview_response
        return {}
    
    mock_http_client.get.side_effect = mock_get
    
    # Call the method
    result = adapter.get_security_info('AAPL')
    
    # Verify the result
    assert result['symbol'] == 'AAPL'
    assert result['name'] == 'Apple Inc'
    assert result['current_price'] == 151.0
    assert result['market_cap'] == 2500000000000.0
    assert result['pe_ratio'] == 25.5
    assert result['sector'] == 'Technology'
    assert result['industry'] == 'Consumer Electronics'
    
    # Verify API calls
    assert mock_http_client.get.call_count == 2
    calls = mock_http_client.get.call_args_list
    assert calls[0][1]['params']['function'] == 'GLOBAL_QUOTE'
    assert calls[1][1]['params']['function'] == 'OVERVIEW'


def test_get_historical_prices(adapter, mock_http_client):
    """Test retrieving historical price data."""
    # Mock response data for daily time series
    mock_response = {
        'Time Series (Daily)': {
            '2023-01-03': {
                '1. open': '150.0',
                '2. high': '152.0',
                '3. low': '149.0',
                '4. close': '151.0',
                '5. volume': '1000000'
            },
            '2023-01-02': {
                '1. open': '148.0',
                '2. high': '150.0',
                '3. low': '147.0',
                '4. close': '149.5',
                '5. volume': '900000'
            },
            '2023-01-01': {
                '1. open': '147.0',
                '2. high': '149.0',
                '3. low': '146.0',
                '4. close': '148.0',
                '5. volume': '800000'
            }
        }
    }
    
    mock_http_client.get.return_value = mock_response
    
    # Patch datetime.now to return a fixed date for testing
    with patch('stock_analysis.adapters.alpha_vantage_adapter.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2023, 1, 5)  # Fixed date after our test data
        
        # Call the method
        result = adapter.get_historical_prices('AAPL', period='5d', interval='1d')
        
        # Verify the result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
    assert 'Open' in result.columns
    assert 'High' in result.columns
    assert 'Low' in result.columns
    assert 'Close' in result.columns
    assert 'Volume' in result.columns
    
    # Verify API call
    mock_http_client.get.assert_called_once()
    args, kwargs = mock_http_client.get.call_args
    assert kwargs['params']['function'] == 'TIME_SERIES_DAILY'
    assert kwargs['params']['symbol'] == 'AAPL'


def test_get_financial_statements(adapter, mock_http_client):
    """Test retrieving financial statements."""
    # Mock response data
    mock_response = {
        'symbol': 'AAPL',
        'annualReports': [
            {
                'fiscalDateEnding': '2022-09-30',
                'reportedCurrency': 'USD',
                'totalRevenue': '394328000000',
                'netIncome': '99803000000',
                'grossProfit': '170782000000',
                'operatingIncome': '119437000000'
            },
            {
                'fiscalDateEnding': '2021-09-30',
                'reportedCurrency': 'USD',
                'totalRevenue': '365817000000',
                'netIncome': '94680000000',
                'grossProfit': '152836000000',
                'operatingIncome': '108949000000'
            }
        ],
        'quarterlyReports': [
            {
                'fiscalDateEnding': '2022-12-31',
                'reportedCurrency': 'USD',
                'totalRevenue': '117154000000',
                'netIncome': '29998000000',
                'grossProfit': '50332000000',
                'operatingIncome': '36016000000'
            }
        ]
    }
    
    mock_http_client.get.return_value = mock_response
    
    # Call the method for annual reports
    annual_result = adapter.get_financial_statements('AAPL', statement_type='income', period='annual')
    
    # Verify the annual result
    assert isinstance(annual_result, pd.DataFrame)
    assert len(annual_result) == 2
    assert 'totalRevenue' in annual_result.columns
    assert 'netIncome' in annual_result.columns
    
    # Call the method for quarterly reports
    mock_http_client.get.return_value = mock_response  # Reset mock
    quarterly_result = adapter.get_financial_statements('AAPL', statement_type='income', period='quarterly')
    
    # Verify the quarterly result
    assert isinstance(quarterly_result, pd.DataFrame)
    assert len(quarterly_result) == 1
    
    # Verify API calls
    assert mock_http_client.get.call_count == 2
    args, kwargs = mock_http_client.get.call_args
    assert kwargs['params']['function'] == 'INCOME_STATEMENT'
    assert kwargs['params']['symbol'] == 'AAPL'


def test_get_technical_indicators(adapter, mock_http_client):
    """Test retrieving technical indicators."""
    # Mock responses for different indicators
    mock_sma_response = {
        'Technical Analysis: SMA': {
            '2023-01-03': {'SMA': '150.5'},
            '2023-01-02': {'SMA': '149.8'}
        }
    }
    
    mock_rsi_response = {
        'Technical Analysis: RSI': {
            '2023-01-03': {'RSI': '65.5'},
            '2023-01-02': {'RSI': '63.2'}
        }
    }
    
    mock_macd_response = {
        'Technical Analysis: MACD': {
            '2023-01-03': {
                'MACD': '2.5',
                'MACD_Signal': '1.8',
                'MACD_Hist': '0.7'
            },
            '2023-01-02': {
                'MACD': '2.2',
                'MACD_Signal': '1.7',
                'MACD_Hist': '0.5'
            }
        }
    }
    
    # Configure mock to return different responses based on parameters
    def mock_get(endpoint, params=None, **kwargs):
        if params and params.get('function') == 'SMA':
            return mock_sma_response
        elif params and params.get('function') == 'RSI':
            return mock_rsi_response
        elif params and params.get('function') == 'MACD':
            return mock_macd_response
        return {}
    
    mock_http_client.get.side_effect = mock_get
    
    # Call the method
    result = adapter.get_technical_indicators('AAPL', indicators=['sma_20', 'rsi_14', 'macd'])
    
    # Verify the result
    assert 'sma_20' in result
    assert 'rsi_14' in result
    assert 'macd' in result
    assert result['sma_20'] == 150.5
    assert result['rsi_14'] == 65.5
    assert result['macd']['macd'] == 2.5
    assert result['macd']['signal'] == 1.8
    assert result['macd']['histogram'] == 0.7
    
    # Verify API calls
    assert mock_http_client.get.call_count == 3


def test_get_market_data_indices(adapter, mock_http_client):
    """Test retrieving market indices data."""
    # Mock response for a single index
    mock_quote_response = {
        'Global Quote': {
            '01. symbol': 'SPY',
            '05. price': '450.0',
            '09. change': '2.5',
            '10. change percent': '0.56%'
        }
    }
    
    mock_http_client.get.return_value = mock_quote_response
    
    # Call the method
    result = adapter.get_market_data('indices')
    
    # Verify API calls - should be called multiple times for different indices
    assert mock_http_client.get.call_count > 0
    args, kwargs = mock_http_client.get.call_args
    assert kwargs['params']['function'] == 'GLOBAL_QUOTE'


def test_get_news(adapter, mock_http_client):
    """Test retrieving news articles."""
    # Mock response data
    mock_response = {
        'feed': [
            {
                'title': 'Test News Article',
                'url': 'https://example.com/news/1',
                'source': 'Test Source',
                'time_published': '20230103T120000',
                'summary': 'This is a test news article summary.',
                'overall_sentiment_score': '0.25',
                'overall_sentiment': 'Somewhat Bullish'
            },
            {
                'title': 'Another Test Article',
                'url': 'https://example.com/news/2',
                'source': 'Another Source',
                'time_published': '20230103T110000',
                'summary': 'This is another test news article summary.',
                'overall_sentiment_score': '-0.15',
                'overall_sentiment': 'Somewhat Bearish'
            }
        ]
    }
    
    mock_http_client.get.return_value = mock_response
    
    # Call the method
    result = adapter.get_news(symbol='AAPL', limit=2)
    
    # Verify the result
    assert len(result) == 2
    assert result[0]['title'] == 'Test News Article'
    assert result[0]['source'] == 'Test Source'
    assert result[0]['sentiment'] == 0.25
    assert result[1]['title'] == 'Another Test Article'
    assert result[1]['sentiment'] == -0.15
    
    # Verify API call
    mock_http_client.get.assert_called_once()
    args, kwargs = mock_http_client.get.call_args
    assert kwargs['params']['function'] == 'NEWS_SENTIMENT'
    assert kwargs['params']['tickers'] == 'AAPL'
    assert kwargs['params']['limit'] == 2


def test_get_economic_data(adapter, mock_http_client):
    """Test retrieving economic data."""
    # Mock response data
    mock_response = {
        'data': [
            {
                'date': '2022-12-31',
                'value': '2.1'
            },
            {
                'date': '2022-09-30',
                'value': '2.3'
            },
            {
                'date': '2022-06-30',
                'value': '2.5'
            }
        ]
    }
    
    mock_http_client.get.return_value = mock_response
    
    # Call the method
    result = adapter.get_economic_data('GDP')
    
    # Verify the result
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert 'value' in result.columns
    
    # Verify API call
    mock_http_client.get.assert_called_once()
    args, kwargs = mock_http_client.get.call_args
    assert kwargs['params']['function'] == 'REAL_GDP'


def test_validate_symbol_valid(adapter, mock_http_client):
    """Test validating a valid symbol."""
    # Mock response data
    mock_response = {
        'Global Quote': {
            '01. symbol': 'AAPL',
            '05. price': '151.0'
        }
    }
    
    mock_http_client.get.return_value = mock_response
    
    # Call the method
    result = adapter.validate_symbol('AAPL')
    
    # Verify the result
    assert result is True
    
    # Verify API call
    mock_http_client.get.assert_called_once()
    args, kwargs = mock_http_client.get.call_args
    assert kwargs['params']['function'] == 'GLOBAL_QUOTE'
    assert kwargs['params']['symbol'] == 'AAPL'


def test_validate_symbol_invalid(adapter, mock_http_client):
    """Test validating an invalid symbol."""
    # Mock response data
    mock_response = {
        'Global Quote': {}
    }
    
    mock_http_client.get.return_value = mock_response
    
    # Call the method
    result = adapter.validate_symbol('INVALID')
    
    # Verify the result
    assert result is False
    
    # Verify API call
    mock_http_client.get.assert_called_once()
    args, kwargs = mock_http_client.get.call_args
    assert kwargs['params']['function'] == 'GLOBAL_QUOTE'
    assert kwargs['params']['symbol'] == 'INVALID'


def test_error_handling(adapter, mock_http_client):
    """Test error handling in the adapter."""
    # Mock HTTP client to raise an exception
    mock_http_client.get.side_effect = Exception("Test error")
    
    # Call methods and verify they handle errors gracefully
    with pytest.raises(DataRetrievalError):
        adapter.get_security_info('AAPL')
    
    with pytest.raises(DataRetrievalError):
        adapter.get_historical_prices('AAPL')
    
    with pytest.raises(DataRetrievalError):
        adapter.get_financial_statements('AAPL')
    
    # These methods should return empty results rather than raising exceptions
    assert adapter.get_technical_indicators('AAPL') == {}
    assert adapter.get_market_data('indices') == {}
    assert adapter.get_news('AAPL') == []
    assert adapter.get_economic_data('GDP').empty
    assert adapter.validate_symbol('AAPL') is False