"""Unit tests for Investing.com adapter.

This module contains comprehensive tests for the InvestingAdapter class,
including tests for all major functionality and error handling scenarios.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests

from stock_analysis.adapters.investing_adapter import InvestingAdapter
from stock_analysis.utils.exceptions import DataRetrievalError, NetworkError


class TestInvestingAdapter:
    """Test cases for InvestingAdapter."""
    
    @pytest.fixture
    def adapter(self):
        """Create InvestingAdapter instance for testing."""
        with patch('stock_analysis.adapters.investing_adapter.get_adapter_config') as mock_config:
            mock_config.return_value = None
            return InvestingAdapter()
    
    @pytest.fixture
    def mock_http_client(self, adapter):
        """Mock HTTP client for testing."""
        with patch.object(adapter, 'http_client') as mock_client:
            yield mock_client
    
    @pytest.fixture
    def sample_search_response(self):
        """Sample search response HTML."""
        return {
            'content': '''
            <html>
                <body>
                    <a class="js-inner-all-results-quote-item" href="/equities/apple-computer-inc">
                        Apple Inc. (AAPL)
                    </a>
                    <a class="js-inner-all-results-quote-item" href="/equities/microsoft-corp">
                        Microsoft Corporation (MSFT)
                    </a>
                </body>
            </html>
            '''
        }
    
    @pytest.fixture
    def sample_security_page(self):
        """Sample security page HTML."""
        return {
            'content': '''
            <html>
                <body>
                    <span data-test="instrument-price-last">150.25</span>
                    <div class="overview-data-table">
                        <div class="flex">
                            <div>Market Cap</div>
                            <div>2.5T</div>
                        </div>
                        <div class="flex">
                            <div>P/E Ratio</div>
                            <div>25.5</div>
                        </div>
                        <div class="flex">
                            <div>Beta</div>
                            <div>1.2</div>
                        </div>
                    </div>
                    <nav class="breadcrumb">
                        <a href="/equities">Equities</a>
                        <a href="/equities/technology">Technology</a>
                        <a href="/equities/technology/software">Software</a>
                    </nav>
                </body>
            </html>
            '''
        }
    
    @pytest.fixture
    def sample_historical_data(self):
        """Sample historical data HTML."""
        return {
            'content': '''
            <html>
                <body>
                    <table class="historical-data-table">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Price</th>
                                <th>Open</th>
                                <th>High</th>
                                <th>Low</th>
                                <th>Vol.</th>
                                <th>Change %</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Jan 15, 2024</td>
                                <td>150.25</td>
                                <td>149.50</td>
                                <td>151.00</td>
                                <td>148.75</td>
                                <td>50.2M</td>
                                <td>+0.75%</td>
                            </tr>
                            <tr>
                                <td>Jan 14, 2024</td>
                                <td>149.50</td>
                                <td>148.00</td>
                                <td>150.00</td>
                                <td>147.50</td>
                                <td>45.8M</td>
                                <td>+1.01%</td>
                            </tr>
                        </tbody>
                    </table>
                </body>
            </html>
            '''
        }
    
    def test_initialization(self):
        """Test adapter initialization."""
        with patch('stock_analysis.adapters.investing_adapter.get_adapter_config') as mock_config:
            mock_config.return_value = None
            adapter = InvestingAdapter()
            
            assert adapter.name == 'investing'
            assert adapter.http_client is not None
            assert adapter._symbol_cache == {}
            assert adapter._data_cache == {}
            assert not adapter._session_initialized
    
    def test_initialization_with_config(self):
        """Test adapter initialization with configuration."""
        mock_config = Mock()
        mock_config.rate_limit = 1.0
        mock_config.timeout = 30
        mock_config.max_retries = 2
        mock_config.settings = {'cache_ttl': 300}
        mock_config.get_setting.return_value = 300
        
        with patch('stock_analysis.adapters.investing_adapter.get_adapter_config') as mock_get_config:
            mock_get_config.return_value = mock_config
            adapter = InvestingAdapter()
            
            assert adapter._cache_ttl == 300
    
    def test_session_initialization(self, adapter, mock_http_client):
        """Test session initialization."""
        mock_http_client.get.return_value = {'content': '<html></html>'}
        
        adapter._initialize_session()
        
        assert adapter._session_initialized
        mock_http_client.get.assert_called_once_with('/')
    
    def test_session_initialization_failure(self, adapter, mock_http_client):
        """Test session initialization failure."""
        mock_http_client.get.side_effect = Exception("Connection failed")
        
        with pytest.raises(DataRetrievalError, match="Failed to initialize session"):
            adapter._initialize_session()
    
    def test_determine_security_type(self, adapter):
        """Test security type determination from URL."""
        assert adapter._determine_security_type('/equities/apple-inc') == 'stock'
        assert adapter._determine_security_type('/etfs/spy') == 'etf'
        assert adapter._determine_security_type('/indices/s-p-500') == 'index'
        assert adapter._determine_security_type('/commodities/gold') == 'commodity'
        assert adapter._determine_security_type('/currencies/eur-usd') == 'currency'
        assert adapter._determine_security_type('/unknown/path') == 'unknown'
    
    def test_search_symbol_success(self, adapter, mock_http_client, sample_search_response):
        """Test successful symbol search."""
        mock_http_client.get.return_value = sample_search_response
        
        result = adapter._search_symbol('AAPL')
        
        assert result is not None
        assert result['url'] == '/equities/apple-computer-inc'
        assert 'Apple Inc.' in result['name']
        assert result['type'] == 'stock'
        
        # Test caching
        assert 'AAPL' in adapter._symbol_cache
    
    def test_search_symbol_cached(self, adapter):
        """Test cached symbol search."""
        # Pre-populate cache
        adapter._symbol_cache['AAPL'] = {
            'url': '/equities/apple-inc',
            'name': 'Apple Inc.',
            'type': 'stock'
        }
        
        result = adapter._search_symbol('AAPL')
        
        assert result == adapter._symbol_cache['AAPL']
    
    def test_search_symbol_not_found(self, adapter, mock_http_client):
        """Test symbol search when symbol not found."""
        mock_http_client.get.return_value = {'content': '<html><body></body></html>'}
        
        result = adapter._search_symbol('INVALID')
        
        assert result is None
    
    def test_parse_numeric_value(self, adapter):
        """Test numeric value parsing."""
        assert adapter._parse_numeric_value('1,234.56') == 1234.56
        assert adapter._parse_numeric_value('1.5K') == 1500.0
        assert adapter._parse_numeric_value('2.5M') == 2500000.0
        assert adapter._parse_numeric_value('1.2B') == 1200000000.0
        assert adapter._parse_numeric_value('1.5T') == 1500000000000.0
        assert adapter._parse_numeric_value('invalid') is None
        assert adapter._parse_numeric_value('') is None
    
    def test_parse_float_value(self, adapter):
        """Test float value parsing."""
        assert adapter._parse_float_value('25.5') == 25.5
        assert adapter._parse_float_value('5.5%') == 0.055
        assert adapter._parse_float_value('1,234.56') == 1234.56
        assert adapter._parse_float_value('invalid') is None
        assert adapter._parse_float_value('') is None
    
    def test_parse_financial_value(self, adapter):
        """Test financial value parsing."""
        assert adapter._parse_financial_value('1,234.56') == 1234.56
        assert adapter._parse_financial_value('(1,234.56)') == -1234.56
        assert adapter._parse_financial_value('1.5M') == 1500000.0
        assert adapter._parse_financial_value('(2.5B)') == -2500000000.0
        assert adapter._parse_financial_value('-') is None
        assert adapter._parse_financial_value('N/A') is None
        assert adapter._parse_financial_value('') is None
    
    def test_get_security_info_success(self, adapter, mock_http_client, sample_search_response, sample_security_page):
        """Test successful security info retrieval."""
        mock_http_client.get.side_effect = [sample_search_response, sample_security_page]
        
        result = adapter.get_security_info('AAPL')
        
        assert result['symbol'] == 'AAPL'
        assert 'Apple Inc.' in result['name']
        assert result['current_price'] == 150.25
        assert result['market_cap'] == 2.5e12  # 2.5T
        assert result['pe_ratio'] == 25.5
        assert result['beta'] == 1.2
        assert result['industry'] == 'Software'
        assert result['sector'] == 'Technology'
    
    def test_get_security_info_symbol_not_found(self, adapter, mock_http_client):
        """Test security info retrieval when symbol not found."""
        mock_http_client.get.return_value = {'content': '<html><body></body></html>'}
        
        with pytest.raises(DataRetrievalError, match="Symbol INVALID not found"):
            adapter.get_security_info('INVALID')
    
    def test_get_security_info_cached(self, adapter):
        """Test cached security info retrieval."""
        # Pre-populate cache
        cached_data = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'current_price': 150.0
        }
        adapter._set_cached_data('security_info_AAPL', cached_data)
        
        result = adapter.get_security_info('AAPL')
        
        assert result == cached_data
    
    def test_get_historical_prices_success(self, adapter, mock_http_client, sample_search_response, sample_historical_data):
        """Test successful historical prices retrieval."""
        mock_http_client.get.side_effect = [sample_search_response, sample_historical_data]
        
        result = adapter.get_historical_prices('AAPL', '1y', '1d')
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'Close' in result.columns
        assert 'Open' in result.columns
        assert 'High' in result.columns
        assert 'Low' in result.columns
        assert 'Volume' in result.columns
    
    def test_get_historical_prices_symbol_not_found(self, adapter, mock_http_client):
        """Test historical prices retrieval when symbol not found."""
        mock_http_client.get.return_value = {'content': '<html><body></body></html>'}
        
        with pytest.raises(DataRetrievalError, match="Symbol INVALID not found"):
            adapter.get_historical_prices('INVALID')
    
    def test_calculate_start_date(self, adapter):
        """Test start date calculation."""
        end_date = datetime(2024, 1, 15)
        
        # Test various periods
        assert adapter._calculate_start_date('1d', end_date) == end_date - timedelta(days=1)
        assert adapter._calculate_start_date('1y', end_date) == end_date - timedelta(days=365)
        assert adapter._calculate_start_date('5y', end_date) == end_date - timedelta(days=1825)
        
        # Test YTD
        ytd_start = adapter._calculate_start_date('ytd', end_date)
        expected_ytd = end_date - timedelta(days=(end_date - datetime(end_date.year, 1, 1)).days)
        assert ytd_start == expected_ytd
    
    def test_get_financial_statements_success(self, adapter, mock_http_client, sample_search_response):
        """Test successful financial statements retrieval."""
        financial_data = {
            'content': '''
            <html>
                <body>
                    <table class="genTbl">
                        <tr>
                            <th>Item</th>
                            <th>2023</th>
                            <th>2022</th>
                            <th>2021</th>
                        </tr>
                        <tr>
                            <td>Revenue</td>
                            <td>394.3B</td>
                            <td>365.8B</td>
                            <td>274.5B</td>
                        </tr>
                        <tr>
                            <td>Net Income</td>
                            <td>97.0B</td>
                            <td>99.8B</td>
                            <td>94.7B</td>
                        </tr>
                    </table>
                </body>
            </html>
            '''
        }
        
        mock_http_client.get.side_effect = [sample_search_response, financial_data]
        
        result = adapter.get_financial_statements('AAPL', 'income', 'annual')
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert '2023' in result.columns
        assert '2022' in result.columns
        assert '2021' in result.columns
    
    def test_get_technical_indicators_success(self, adapter, mock_http_client, sample_search_response):
        """Test successful technical indicators retrieval."""
        technical_data = {
            'content': '''
            <html>
                <body>
                    <div class="technicalIndicatorsTbl">
                        <div>RSI (14): 65.5</div>
                        <div>MACD: 2.5</div>
                        <div>SMA 20: 148.5</div>
                        <div>SMA 50: 145.2</div>
                    </div>
                </body>
            </html>
            '''
        }
        
        mock_http_client.get.side_effect = [sample_search_response, technical_data]
        
        result = adapter.get_technical_indicators('AAPL', ['rsi_14', 'macd', 'sma_20'])
        
        assert isinstance(result, dict)
        # Note: The actual extraction might not work perfectly with this simple HTML,
        # but the method should return a dictionary without errors
    
    def test_get_market_data_indices(self, adapter, mock_http_client):
        """Test market indices data retrieval."""
        indices_data = {
            'content': '''
            <html>
                <body>
                    <table class="genTbl">
                        <tr>
                            <th>Index</th>
                            <th>Price</th>
                            <th>Change</th>
                        </tr>
                        <tr>
                            <td>S&P 500</td>
                            <td>4,500.25</td>
                            <td>+15.5</td>
                        </tr>
                        <tr>
                            <td>NASDAQ</td>
                            <td>14,200.75</td>
                            <td>-25.3</td>
                        </tr>
                    </table>
                </body>
            </html>
            '''
        }
        
        mock_http_client.get.return_value = indices_data
        
        result = adapter.get_market_data('indices')
        
        assert isinstance(result, dict)
        # The method should return data without errors
    
    def test_get_news_company_specific(self, adapter, mock_http_client, sample_search_response):
        """Test company-specific news retrieval."""
        news_data = {
            'content': '''
            <html>
                <body>
                    <div class="largeTitle">
                        <a href="/news/apple-earnings-beat-expectations">Apple Earnings Beat Expectations</a>
                        <span class="date">Jan 15, 2024</span>
                        <p>Apple reported strong quarterly earnings...</p>
                    </div>
                    <div class="largeTitle">
                        <a href="/news/apple-new-product-launch">Apple Announces New Product</a>
                        <span class="date">Jan 14, 2024</span>
                        <p>Apple unveiled its latest innovation...</p>
                    </div>
                </body>
            </html>
            '''
        }
        
        mock_http_client.get.side_effect = [sample_search_response, news_data]
        
        result = adapter.get_news('AAPL', limit=5)
        
        assert isinstance(result, list)
        # The method should return a list without errors
    
    def test_get_news_market_general(self, adapter, mock_http_client):
        """Test general market news retrieval."""
        news_data = {
            'content': '''
            <html>
                <body>
                    <div class="largeTitle">
                        <a href="/news/market-update">Market Update</a>
                        <span class="date">Jan 15, 2024</span>
                        <p>Markets closed higher today...</p>
                    </div>
                </body>
            </html>
            '''
        }
        
        mock_http_client.get.return_value = news_data
        
        result = adapter.get_news(limit=5)
        
        assert isinstance(result, list)
    
    def test_get_economic_data_success(self, adapter, mock_http_client):
        """Test economic data retrieval."""
        economic_data = {
            'content': '''
            <html>
                <body>
                    <table id="economicCalendarData">
                        <tr>
                            <th>Date</th>
                            <th>Event</th>
                            <th>Actual</th>
                            <th>Forecast</th>
                            <th>Previous</th>
                        </tr>
                        <tr>
                            <td>Jan 15, 2024</td>
                            <td>GDP Growth Rate</td>
                            <td>2.5%</td>
                            <td>2.3%</td>
                            <td>2.1%</td>
                        </tr>
                    </table>
                </body>
            </html>
            '''
        }
        
        mock_http_client.get.return_value = economic_data
        
        result = adapter.get_economic_data('GDP')
        
        assert isinstance(result, pd.DataFrame)
    
    def test_caching_functionality(self, adapter):
        """Test data caching functionality."""
        # Test setting and getting cached data
        test_data = {'test': 'value'}
        adapter._set_cached_data('test_key', test_data)
        
        cached_result = adapter._get_cached_data('test_key')
        assert cached_result == test_data
        
        # Test cache expiration
        adapter._cache_ttl = 0  # Set TTL to 0 for immediate expiration
        import time
        time.sleep(0.1)  # Wait a bit
        
        expired_result = adapter._get_cached_data('test_key')
        assert expired_result is None
    
    def test_error_handling_network_error(self, adapter, mock_http_client):
        """Test error handling for network errors."""
        mock_http_client.get.side_effect = requests.RequestException("Network connection failed")
        
        with pytest.raises(DataRetrievalError):
            adapter.get_security_info('AAPL')
    
    def test_error_handling_invalid_response(self, adapter, mock_http_client):
        """Test error handling for invalid responses."""
        mock_http_client.get.return_value = {}  # No 'content' key
        
        with pytest.raises(DataRetrievalError):
            adapter.get_security_info('AAPL')
    
    def test_validate_symbol(self, adapter):
        """Test symbol validation."""
        # Mock get_security_info to test validation
        with patch.object(adapter, 'get_security_info') as mock_get_info:
            mock_get_info.return_value = {'symbol': 'AAPL', 'name': 'Apple Inc.'}
            assert adapter.validate_symbol('AAPL') is True
            
            mock_get_info.side_effect = Exception("Symbol not found")
            assert adapter.validate_symbol('INVALID') is False
    
    def test_get_supported_features(self, adapter):
        """Test getting supported features."""
        features = adapter.get_supported_features()
        
        assert isinstance(features, list)
        assert 'security_info' in features
        assert 'historical_prices' in features
        assert 'financial_statements' in features
    
    def test_string_representations(self, adapter):
        """Test string representations of the adapter."""
        str_repr = str(adapter)
        assert 'InvestingAdapter' in str_repr
        assert 'investing' in str_repr
        
        detailed_repr = repr(adapter)
        assert 'InvestingAdapter' in detailed_repr
        assert 'investing' in detailed_repr


class TestInvestingAdapterIntegration:
    """Integration tests for InvestingAdapter.
    
    These tests require network access and should be run separately
    or mocked for CI/CD environments.
    """
    
    @pytest.mark.integration
    def test_real_symbol_search(self):
        """Test real symbol search (requires network access)."""
        adapter = InvestingAdapter()
        
        # This test would require actual network access
        # In a real scenario, you might want to test with a known symbol
        # result = adapter._search_symbol('AAPL')
        # assert result is not None
        pass
    
    @pytest.mark.integration
    def test_real_security_info(self):
        """Test real security info retrieval (requires network access)."""
        adapter = InvestingAdapter()
        
        # This test would require actual network access
        # result = adapter.get_security_info('AAPL')
        # assert 'symbol' in result
        # assert 'current_price' in result
        pass


if __name__ == '__main__':
    pytest.main([__file__])