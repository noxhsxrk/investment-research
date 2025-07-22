"""Integration tests for cache manager with stock data service."""

import pytest
import os
import time
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

from stock_analysis.utils.cache_manager import CacheManager, get_cache_manager
from stock_analysis.services.stock_data_service import StockDataService


@pytest.fixture
def cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def config_mock(cache_dir):
    """Create a mock configuration."""
    mock = MagicMock()
    
    def mock_get(key, default=None):
        config_values = {
            'stock_analysis.cache.use_disk_cache': True,
            'stock_analysis.cache.directory': str(cache_dir),
            'stock_analysis.cache.max_memory_size': 1024 * 1024,  # 1MB
            'stock_analysis.cache.max_disk_size': 10 * 1024 * 1024,  # 10MB
            'stock_analysis.cache.cleanup_interval': 1,  # 1 second for faster testing
            'stock_analysis.cache.expiry.stock_info': 60,  # 1 minute
            'stock_analysis.cache.expiry.historical_data': 300,  # 5 minutes
            'stock_analysis.cache.expiry.financial_statements': 600,  # 10 minutes
            'stock_analysis.cache.expiry.news': 30,  # 30 seconds
            'stock_analysis.cache.expiry.peer_data': 600,  # 10 minutes
            'stock_analysis.cache.expiry.general': 60,  # 1 minute
            'stock_analysis.data_sources.yfinance.timeout': 30,
            'stock_analysis.data_sources.yfinance.retry_attempts': 3,
            'stock_analysis.data_sources.yfinance.rate_limit_delay': 0.1,  # Fast for testing
        }
        return config_values.get(key, default)
    
    mock.get.side_effect = mock_get
    return mock


@pytest.fixture
def cache_manager(cache_dir, config_mock):
    """Create a cache manager instance for testing."""
    with patch('stock_analysis.utils.cache_manager.config', config_mock):
        # Reset singleton instance
        CacheManager._instance = None
        manager = CacheManager(cache_dir)
        yield manager
        manager.stop_cleanup_thread()


@pytest.fixture
def stock_data_service(cache_manager, config_mock):
    """Create a stock data service instance for testing."""
    with patch('stock_analysis.services.stock_data_service.config', config_mock):
        return StockDataService()


class TestCacheIntegration:
    """Test integration of cache manager with stock data service."""
    
    @patch('yfinance.Ticker')
    def test_stock_info_caching(self, mock_ticker, stock_data_service):
        """Test caching of stock info in stock data service."""
        # Mock the ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock the info property
        mock_ticker_instance.info = {
            'symbol': 'AAPL',
            'shortName': 'Apple Inc.',
            'longName': 'Apple Inc.',
            'currentPrice': 150.0,
            'marketCap': 2500000000000,
            'trailingPE': 25.0,
            'priceToBook': 30.0,
            'dividendYield': 0.005,
            'beta': 1.2,
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'quoteType': 'EQUITY'
        }
        
        # First call should retrieve from API
        result1 = stock_data_service.get_security_info('AAPL')
        
        # Second call should retrieve from cache
        result2 = stock_data_service.get_security_info('AAPL')
        
        # Verify ticker was only created once
        mock_ticker.assert_called_once_with('AAPL')
        
        # Verify results are the same
        assert result1.symbol == result2.symbol
        assert result1.name == result2.name
        assert result1.current_price == result2.current_price
    
    @patch('yfinance.Ticker')
    def test_historical_data_caching(self, mock_ticker, stock_data_service):
        """Test caching of historical data in stock data service."""
        # Mock the ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock the history method
        mock_data = pd.DataFrame({
            'Open': [150.0, 151.0, 152.0],
            'High': [155.0, 156.0, 157.0],
            'Low': [149.0, 150.0, 151.0],
            'Close': [153.0, 154.0, 155.0],
            'Volume': [1000000, 1100000, 1200000]
        })
        mock_ticker_instance.history = MagicMock(return_value=mock_data)
        
        # First call should retrieve from API
        result1 = stock_data_service.get_historical_data('AAPL', period='1mo')
        
        # Second call should retrieve from cache
        result2 = stock_data_service.get_historical_data('AAPL', period='1mo')
        
        # Verify history was only called once
        mock_ticker_instance.history.assert_called_once()
        
        # Verify results are the same
        pd.testing.assert_frame_equal(result1, result2)
    
    @patch('yfinance.Ticker')
    def test_financial_statements_caching(self, mock_ticker, stock_data_service):
        """Test caching of financial statements in stock data service."""
        # Mock the ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock the income_stmt property
        mock_data = pd.DataFrame({
            '2020-12-31': [100000, 50000, 30000],
            '2021-12-31': [110000, 55000, 33000],
            '2022-12-31': [120000, 60000, 36000]
        }, index=['Revenue', 'GrossProfit', 'NetIncome'])
        mock_ticker_instance.income_stmt = mock_data
        
        # First call should retrieve from API
        result1 = stock_data_service.get_financial_statements('AAPL', statement_type='income')
        
        # Second call should retrieve from cache
        result2 = stock_data_service.get_financial_statements('AAPL', statement_type='income')
        
        # Verify ticker was only created once
        mock_ticker.assert_called_once_with('AAPL')
        
        # Verify results are the same
        pd.testing.assert_frame_equal(result1, result2)
    
    @patch('yfinance.Ticker')
    def test_cache_expiration(self, mock_ticker, stock_data_service, cache_manager):
        """Test cache expiration for stock data."""
        # Mock the ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock the info property
        mock_ticker_instance.info = {
            'symbol': 'AAPL',
            'shortName': 'Apple Inc.',
            'longName': 'Apple Inc.',
            'currentPrice': 150.0,
            'marketCap': 2500000000000,
            'trailingPE': 25.0,
            'priceToBook': 30.0,
            'dividendYield': 0.005,
            'beta': 1.2,
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'quoteType': 'EQUITY'
        }
        
        # Set a very short expiration time for testing
        cache_manager._default_expiry['stock_info'] = 1  # 1 second
        
        # First call should retrieve from API
        stock_data_service.get_security_info('AAPL')
        
        # Reset mock to verify second call
        mock_ticker.reset_mock()
        
        # Wait for cache to expire
        time.sleep(1.1)
        
        # Second call should retrieve from API again due to expiration
        stock_data_service.get_security_info('AAPL')
        
        # Verify ticker was created again
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('yfinance.Ticker')
    def test_cache_invalidation(self, mock_ticker, stock_data_service):
        """Test cache invalidation for stock data."""
        # Mock the ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock the info property
        mock_ticker_instance.info = {
            'symbol': 'AAPL',
            'shortName': 'Apple Inc.',
            'longName': 'Apple Inc.',
            'currentPrice': 150.0,
            'marketCap': 2500000000000,
            'trailingPE': 25.0,
            'priceToBook': 30.0,
            'dividendYield': 0.005,
            'beta': 1.2,
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'quoteType': 'EQUITY'
        }
        
        # First call should retrieve from API
        stock_data_service.get_security_info('AAPL')
        
        # Reset mock to verify second call
        mock_ticker.reset_mock()
        
        # Invalidate the cache
        cache = get_cache_manager()
        cache.invalidate_by_pattern("security_info:*")
        
        # Second call should retrieve from API again due to invalidation
        stock_data_service.get_security_info('AAPL')
        
        # Verify ticker was created again
        mock_ticker.assert_called_once_with('AAPL')