"""Integration tests for data integration service.

This module contains integration tests for the FinancialDataIntegrationService,
testing its interaction with multiple data sources and caching mechanisms.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import pytest

from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
from stock_analysis.adapters.yfinance_adapter import YFinanceAdapter
from stock_analysis.adapters.investing_adapter import InvestingAdapter
from stock_analysis.adapters.alpha_vantage_adapter import AlphaVantageAdapter
from stock_analysis.utils.exceptions import DataRetrievalError
from stock_analysis.utils.cache_manager import get_cache_manager


class TestDataIntegrationServiceIntegration(unittest.TestCase):
    """Integration tests for FinancialDataIntegrationService."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        # Use mock configuration for tests to avoid actual API calls
        with patch('stock_analysis.adapters.config.get_api_config') as mock_config:
            mock_config.return_value = {
                'yfinance': {'use_cache': True},
                'investing': {
                    'username': 'test_user',
                    'password': 'test_password',
                    'use_cache': True
                },
                'alpha_vantage': {
                    'api_key': 'test_api_key',
                    'use_cache': True
                }
            }
            
            # Initialize real adapters with mocked config
            cls.yf_adapter = YFinanceAdapter()
            cls.investing_adapter = InvestingAdapter()
            cls.alpha_vantage_adapter = AlphaVantageAdapter()
    
    def setUp(self):
        """Set up test fixtures for each test."""
        # Create integration service with real adapters
        self.service = FinancialDataIntegrationService(
            adapters=[self.yf_adapter, self.investing_adapter, self.alpha_vantage_adapter]
        )
        
        # Sample test data for mocking responses
        self.sample_security_info = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'current_price': 150.0,
            'market_cap': 2500000000000,
            'pe_ratio': 25.0,
            'sector': 'Technology',
            'industry': 'Consumer Electronics'
        }
        
        self.sample_historical_data = pd.DataFrame({
            'Open': [145.0, 146.0, 147.0],
            'High': [150.0, 151.0, 152.0],
            'Low': [144.0, 145.0, 146.0],
            'Close': [149.0, 150.0, 151.0],
            'Volume': [100000000, 95000000, 105000000]
        }, index=pd.date_range(start='2023-01-01', periods=3))
        
        self.sample_financial_statements = pd.DataFrame({
            'Revenue': [365000000000, 350000000000, 325000000000],
            'Net Income': [95000000000, 90000000000, 85000000000],
            'EPS': [5.5, 5.2, 4.9]
        }, index=pd.date_range(start='2020-01-01', periods=3, freq='Y'))
        
        self.sample_technical_indicators = {
            'sma_20': 148.5,
            'sma_50': 145.2,
            'rsi_14': 65.3,
            'macd': {
                'macd': 2.5,
                'signal': 1.8,
                'histogram': 0.7
            }
        }
        
        self.sample_market_data = {
            'S&P 500': {
                'price': 4500.0,
                'change': 15.0,
                'change_percent': 0.33
            },
            'NASDAQ': {
                'price': 14000.0,
                'change': 50.0,
                'change_percent': 0.36
            }
        }
        
        self.sample_news = [
            {
                'title': 'Apple Announces New iPhone',
                'source': 'Tech News',
                'url': 'https://example.com/news/1',
                'published_at': '2023-01-15T12:30:00Z',
                'summary': 'Apple has announced the new iPhone model.'
            },
            {
                'title': 'Apple Reports Record Earnings',
                'source': 'Financial News',
                'url': 'https://example.com/news/2',
                'published_at': '2023-01-10T15:45:00Z',
                'summary': 'Apple reported record earnings for the quarter.'
            }
        ]
    
    def test_service_initialization(self):
        """Test service initialization with real adapters."""
        # Create service with default adapters
        with patch('stock_analysis.services.financial_data_integration_service.YFinanceAdapter') as mock_yf_class, \
             patch('stock_analysis.services.financial_data_integration_service.InvestingAdapter') as mock_inv_class, \
             patch('stock_analysis.services.financial_data_integration_service.AlphaVantageAdapter') as mock_av_class:
            
            mock_yf_class.return_value = self.yf_adapter
            mock_inv_class.return_value = self.investing_adapter
            mock_av_class.return_value = self.alpha_vantage_adapter
            
            service = FinancialDataIntegrationService()
            
            # Verify adapters were initialized
            self.assertEqual(len(service.adapters), 3)
            self.assertIsInstance(service.adapters[0], YFinanceAdapter)
            self.assertIsInstance(service.adapters[1], InvestingAdapter)
            self.assertIsInstance(service.adapters[2], AlphaVantageAdapter)
    
    def test_source_priorities(self):
        """Test source priorities configuration."""
        # Verify source priorities were loaded
        self.assertIn('security_info', self.service.source_priorities)
        self.assertIn('historical_prices', self.service.source_priorities)
        self.assertIn('financial_statements', self.service.source_priorities)
        
        # Verify adapter names are in priorities
        for data_type, priorities in self.service.source_priorities.items():
            self.assertIn('yfinance', priorities)
            self.assertIn('investing', priorities)
            self.assertIn('alpha_vantage', priorities)
    
    @patch('stock_analysis.adapters.yfinance_adapter.YFinanceAdapter.get_security_info')
    @patch('stock_analysis.adapters.investing_adapter.InvestingAdapter.get_security_info')
    def test_get_security_info_integration(self, mock_inv_get_info, mock_yf_get_info):
        """Test getting security info with multiple adapters."""
        # Set up mocks
        mock_yf_get_info.return_value = self.sample_security_info
        mock_inv_get_info.side_effect = DataRetrievalError("Test error")
        
        # Test method
        result = self.service.get_security_info("AAPL", use_cache=False)
        
        # Verify result
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['current_price'], 150.0)
        
        # Verify mocks were called in correct order based on priorities
        mock_yf_get_info.assert_called_once_with("AAPL")
        mock_inv_get_info.assert_not_called()  # Should not be called if first adapter succeeds
    
    @patch('stock_analysis.adapters.yfinance_adapter.YFinanceAdapter.get_security_info')
    @patch('stock_analysis.adapters.investing_adapter.InvestingAdapter.get_security_info')
    def test_get_security_info_fallback_integration(self, mock_inv_get_info, mock_yf_get_info):
        """Test fallback between adapters."""
        # Set up mocks
        mock_yf_get_info.side_effect = DataRetrievalError("Test error")
        mock_inv_get_info.return_value = self.sample_security_info
        
        # Override priorities to ensure YFinance is tried first
        original_priorities = self.service.source_priorities['security_info'].copy()
        self.service.source_priorities['security_info'] = {'yfinance': 1, 'investing': 2, 'alpha_vantage': 3}
        
        try:
            # Test method
            result = self.service.get_security_info("AAPL", use_cache=False)
            
            # Verify result
            self.assertEqual(result['symbol'], 'AAPL')
            self.assertEqual(result['current_price'], 150.0)
            
            # Verify both mocks were called
            mock_yf_get_info.assert_called_once_with("AAPL")
            mock_inv_get_info.assert_called_once_with("AAPL")
        finally:
            # Restore original priorities
            self.service.source_priorities['security_info'] = original_priorities
    
    @patch('stock_analysis.adapters.yfinance_adapter.YFinanceAdapter.get_financial_statements')
    @patch('stock_analysis.adapters.investing_adapter.InvestingAdapter.get_financial_statements')
    def test_get_financial_statements_integration(self, mock_inv_get_statements, mock_yf_get_statements):
        """Test getting financial statements with multiple adapters."""
        # Set up mocks
        mock_yf_get_statements.return_value = pd.DataFrame()  # Empty result
        mock_inv_get_statements.return_value = self.sample_financial_statements
        
        # Override priorities to ensure Investing is tried first for financial statements
        original_priorities = self.service.source_priorities['financial_statements'].copy()
        self.service.source_priorities['financial_statements'] = {'investing': 1, 'yfinance': 2, 'alpha_vantage': 3}
        
        try:
            # Test method
            result = self.service.get_financial_statements(
                "AAPL", statement_type="income", period="annual", use_cache=False)
            
            # Verify result
            self.assertFalse(result.empty)
            self.assertEqual(len(result), 3)
            self.assertEqual(result['Revenue'].iloc[0], 365000000000)
            
            # Verify mocks were called
            mock_inv_get_statements.assert_called_once_with("AAPL", "income", "annual")
            mock_yf_get_statements.assert_not_called()  # Should not be called if first adapter succeeds
        finally:
            # Restore original priorities
            self.service.source_priorities['financial_statements'] = original_priorities
    
    @patch('stock_analysis.adapters.yfinance_adapter.YFinanceAdapter.get_historical_prices')
    @patch('stock_analysis.adapters.alpha_vantage_adapter.AlphaVantageAdapter.get_historical_prices')
    def test_get_historical_prices_integration(self, mock_av_get_prices, mock_yf_get_prices):
        """Test getting historical prices with multiple adapters."""
        # Set up mocks
        mock_yf_get_prices.return_value = self.sample_historical_data
        mock_av_get_prices.side_effect = DataRetrievalError("Test error")
        
        # Test method
        result = self.service.get_historical_prices(
            "AAPL", period="1y", interval="1d", use_cache=False)
        
        # Verify result
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)
        self.assertEqual(result['Close'].iloc[-1], 151.0)
        
        # Verify mocks were called
        mock_yf_get_prices.assert_called_once_with("AAPL", "1y", "1d")
        mock_av_get_prices.assert_not_called()  # Should not be called if first adapter succeeds
    
    @patch('stock_analysis.adapters.investing_adapter.InvestingAdapter.get_technical_indicators')
    def test_get_technical_indicators_integration(self, mock_inv_get_indicators):
        """Test getting technical indicators."""
        # Set up mock
        mock_inv_get_indicators.return_value = self.sample_technical_indicators
        
        # Test method
        result = self.service.get_technical_indicators(
            "AAPL", indicators=['sma_20', 'rsi_14'], use_cache=False)
        
        # Verify result
        self.assertEqual(result['sma_20'], 148.5)
        self.assertEqual(result['rsi_14'], 65.3)
        
        # Verify mock was called
        mock_inv_get_indicators.assert_called_once_with("AAPL", ['sma_20', 'rsi_14'])
    
    @patch('stock_analysis.adapters.investing_adapter.InvestingAdapter.get_market_data')
    def test_get_market_data_integration(self, mock_inv_get_market):
        """Test getting market data."""
        # Set up mock
        mock_inv_get_market.return_value = self.sample_market_data
        
        # Test method
        result = self.service.get_market_data("indices", use_cache=False)
        
        # Verify result
        self.assertEqual(result['S&P 500']['price'], 4500.0)
        self.assertEqual(result['NASDAQ']['price'], 14000.0)
        
        # Verify mock was called
        mock_inv_get_market.assert_called_once_with("indices")
    
    @patch('stock_analysis.adapters.investing_adapter.InvestingAdapter.get_news')
    def test_get_news_integration(self, mock_inv_get_news):
        """Test getting news."""
        # Set up mock
        mock_inv_get_news.return_value = self.sample_news
        
        # Test method
        result = self.service.get_news("AAPL", limit=2, use_cache=False)
        
        # Verify result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['title'], 'Apple Announces New iPhone')
        
        # Verify mock was called
        mock_inv_get_news.assert_called_once_with("AAPL", None, 2)
    
    @patch('stock_analysis.adapters.yfinance_adapter.YFinanceAdapter.get_security_info')
    def test_cache_integration(self, mock_yf_get_info):
        """Test integration with cache manager."""
        # Set up mock
        mock_yf_get_info.return_value = self.sample_security_info
        
        # Clear cache before test
        cache_manager = get_cache_manager()
        cache_manager.clear()
        
        # First call should use adapter
        result1 = self.service.get_security_info("AAPL", use_cache=True)
        
        # Second call should use cache
        result2 = self.service.get_security_info("AAPL", use_cache=True)
        
        # Verify results are the same
        self.assertEqual(result1['symbol'], result2['symbol'])
        self.assertEqual(result1['current_price'], result2['current_price'])
        
        # Verify adapter was called only once
        mock_yf_get_info.assert_called_once_with("AAPL")
        
        # Test cache invalidation
        count = self.service.invalidate_cache(tag="AAPL")
        self.assertGreater(count, 0)
        
        # After invalidation, adapter should be called again
        mock_yf_get_info.reset_mock()
        result3 = self.service.get_security_info("AAPL", use_cache=True)
        mock_yf_get_info.assert_called_once_with("AAPL")
    
    @patch('stock_analysis.adapters.yfinance_adapter.YFinanceAdapter.get_security_info')
    @patch('stock_analysis.adapters.yfinance_adapter.YFinanceAdapter.get_historical_prices')
    @patch('stock_analysis.adapters.investing_adapter.InvestingAdapter.get_technical_indicators')
    def test_get_enhanced_financial_data_integration(self, mock_inv_get_indicators, 
                                                   mock_yf_get_prices, mock_yf_get_info):
        """Test getting enhanced financial data from multiple sources."""
        # Set up mocks
        mock_yf_get_info.return_value = self.sample_security_info
        mock_yf_get_prices.return_value = self.sample_historical_data
        mock_inv_get_indicators.return_value = self.sample_technical_indicators
        
        # Test method
        result = self.service.get_enhanced_financial_data(
            "AAPL", 
            data_types=['security_info', 'historical_prices', 'technical_indicators']
        )
        
        # Verify result contains all requested data types
        self.assertIn('security_info', result)
        self.assertIn('historical_prices', result)
        self.assertIn('technical_indicators', result)
        
        # Verify data content
        self.assertEqual(result['security_info']['symbol'], 'AAPL')
        self.assertEqual(len(result['historical_prices']), 3)
        self.assertEqual(result['technical_indicators']['sma_20'], 148.5)
        
        # Verify mocks were called
        mock_yf_get_info.assert_called_once_with("AAPL")
        mock_yf_get_prices.assert_called_once_with("AAPL", "1y", "1d")
        mock_inv_get_indicators.assert_called_once()
    
    def test_data_normalization_integration(self):
        """Test data normalization across different sources."""
        # Create test data with different formats
        yf_data = pd.DataFrame({
            'Open': [145.0, 146.0, 147.0],
            'High': [150.0, 151.0, 152.0],
            'Low': [144.0, 145.0, 146.0],
            'Close': [149.0, 150.0, 151.0],
            'Volume': [100000000, 95000000, 105000000]
        }, index=pd.date_range(start='2023-01-01', periods=3))
        
        av_data = pd.DataFrame({
            'open': [145.0, 146.0, 147.0],
            'high': [150.0, 151.0, 152.0],
            'low': [144.0, 145.0, 146.0],
            'close': [149.0, 150.0, 151.0],
            'volume': [100000000, 95000000, 105000000]
        }, index=pd.date_range(start='2023-01-01', periods=3))
        
        # Test normalization
        normalized_yf = self.service._normalize_historical_data(yf_data)
        normalized_av = self.service._normalize_historical_data(av_data)
        
        # Verify both are normalized to the same format
        pd.testing.assert_frame_equal(normalized_yf, normalized_av)
        
        # Verify column names are standardized
        self.assertIn('Open', normalized_yf.columns)
        self.assertIn('High', normalized_yf.columns)
        self.assertIn('Low', normalized_yf.columns)
        self.assertIn('Close', normalized_yf.columns)
        self.assertIn('Volume', normalized_yf.columns)


if __name__ == '__main__':
    unittest.main()