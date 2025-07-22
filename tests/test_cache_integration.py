"""Integration tests for the cache manager with stock data service."""

import os
import time
import shutil
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

from stock_analysis.utils.cache_manager import CacheManager, CacheEntry, get_cache_manager
from stock_analysis.utils.config import ConfigManager
from stock_analysis.services.stock_data_service import StockDataService


class TestCacheIntegration(unittest.TestCase):
    """Test the integration of cache manager with stock data service."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test cache directory
        self.test_cache_dir = "./.test_cache"
        os.makedirs(self.test_cache_dir, exist_ok=True)
        
        # Create a test config
        self.config_mock = MagicMock()
        self.config_mock.get.side_effect = self._mock_config_get
        
        # Create a cache manager with test configuration
        with patch('stock_analysis.utils.cache_manager.config', self.config_mock):
            with patch('stock_analysis.services.stock_data_service.config', self.config_mock):
                # Reset the singleton instance for testing
                CacheManager._instance = None
                self.cache_manager = CacheManager(self.test_cache_dir)
                self.stock_data_service = StockDataService()
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop cleanup thread
        self.cache_manager.stop_cleanup_thread()
        
        # Remove test cache directory
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
    
    def _mock_config_get(self, key, default=None):
        """Mock config.get method."""
        config_values = {
            'stock_analysis.cache.use_disk_cache': True,
            'stock_analysis.cache.directory': self.test_cache_dir,
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
    
    @patch('stock_analysis.services.stock_data_service.yf.Ticker')
    def test_stock_info_caching(self, mock_ticker):
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
            'industry': 'Consumer Electronics'
        }
        
        # First call should retrieve from API
        result1 = self.stock_data_service.get_stock_info('AAPL')
        
        # Second call should retrieve from cache
        result2 = self.stock_data_service.get_stock_info('AAPL')
        
        # Verify ticker was only created once
        mock_ticker.assert_called_once_with('AAPL')
        
        # Verify results are the same
        self.assertEqual(result1.symbol, result2.symbol)
        self.assertEqual(result1.company_name, result2.company_name)
        self.assertEqual(result1.current_price, result2.current_price)
    
    @patch('stock_analysis.services.stock_data_service.yf.Ticker')
    def test_historical_data_caching(self, mock_ticker):
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
        mock_ticker_instance.history.return_value = mock_data
        
        # First call should retrieve from API
        result1 = self.stock_data_service.get_historical_data('AAPL', period='1mo')
        
        # Second call should retrieve from cache
        result2 = self.stock_data_service.get_historical_data('AAPL', period='1mo')
        
        # Verify history was only called once
        mock_ticker_instance.history.assert_called_once()
        
        # Verify results are the same
        pd.testing.assert_frame_equal(result1, result2)
    
    @patch('yfinance.Ticker')
    def test_financial_statements_caching(self, mock_ticker):
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
        result1 = self.stock_data_service.get_financial_statements('AAPL', statement_type='income')
        
        # Second call should retrieve from cache
        result2 = self.stock_data_service.get_financial_statements('AAPL', statement_type='income')
        
        # Verify ticker was only created once
        mock_ticker.assert_called_once_with('AAPL')
        
        # Verify results are the same
        pd.testing.assert_frame_equal(result1, result2)
    
    @patch('yfinance.Ticker')
    def test_cache_expiration(self, mock_ticker):
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
            'industry': 'Consumer Electronics'
        }
        
        # Set a very short expiration time for testing
        with patch.dict(self.cache_manager._default_expiry, {'stock_info': 1}):  # 1 second
            # First call should retrieve from API
            self.stock_data_service.get_stock_info('AAPL')
            
            # Reset mock to verify second call
            mock_ticker.reset_mock()
            
            # Wait for cache to expire
            time.sleep(1.1)
            
            # Second call should retrieve from API again due to expiration
            self.stock_data_service.get_stock_info('AAPL')
            
            # Verify ticker was created again
            mock_ticker.assert_called_once_with('AAPL')
    
    @patch('yfinance.Ticker')
    def test_cache_invalidation(self, mock_ticker):
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
            'industry': 'Consumer Electronics'
        }
        
        # First call should retrieve from API
        self.stock_data_service.get_stock_info('AAPL')
        
        # Reset mock to verify second call
        mock_ticker.reset_mock()
        
        # Invalidate the cache
        cache = get_cache_manager()
        cache.invalidate_by_pattern("stock_info:*")
        
        # Second call should retrieve from API again due to invalidation
        self.stock_data_service.get_stock_info('AAPL')
        
        # Verify ticker was created again
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('yfinance.Ticker')
    def test_different_cache_types(self, mock_ticker):
        """Test different cache types have different expiration times."""
        # Mock the ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock the info property
        mock_ticker_instance.info = {
            'symbol': 'AAPL',
            'shortName': 'Apple Inc.',
            'currentPrice': 150.0,
        }
        
        # Mock the history method
        mock_ticker_instance.history.return_value = pd.DataFrame({
            'Close': [153.0, 154.0, 155.0],
        })
        
        # Get cache manager
        cache = get_cache_manager()
        
        # Check default expiry times for different data types
        self.assertNotEqual(
            cache._default_expiry['stock_info'],
            cache._default_expiry['historical_data']
        )
        
        # Verify stock_info has shorter expiry than historical_data
        self.assertLess(
            cache._default_expiry['stock_info'],
            cache._default_expiry['historical_data']
        )


if __name__ == '__main__':
    unittest.main()