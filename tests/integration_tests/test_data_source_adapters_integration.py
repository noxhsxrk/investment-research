"""Integration tests for data source adapters.

This module contains integration tests for the data source adapters,
testing their interaction with external APIs and data sources.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import pytest

from stock_analysis.adapters.yfinance_adapter import YFinanceAdapter
from stock_analysis.adapters.investing_adapter import InvestingAdapter
from stock_analysis.adapters.alpha_vantage_adapter import AlphaVantageAdapter
from stock_analysis.utils.exceptions import DataRetrievalError


class TestDataSourceAdaptersIntegration(unittest.TestCase):
    """Integration tests for data source adapters."""
    
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
            
            # Initialize adapters with mocked config
            cls.yf_adapter = YFinanceAdapter()
            cls.investing_adapter = InvestingAdapter()
            cls.alpha_vantage_adapter = AlphaVantageAdapter()
    
    def setUp(self):
        """Set up test fixtures for each test."""
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
    
    @patch('yfinance.Ticker')
    def test_yfinance_adapter_security_info(self, mock_ticker_class):
        """Test YFinanceAdapter security info retrieval."""
        # Set up mock
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.info = {
            'symbol': 'AAPL',
            'shortName': 'Apple Inc.',
            'regularMarketPrice': 150.0,
            'marketCap': 2500000000000,
            'trailingPE': 25.0,
            'sector': 'Technology',
            'industry': 'Consumer Electronics'
        }
        
        # Test method
        result = self.yf_adapter.get_security_info('AAPL')
        
        # Verify result
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['name'], 'Apple Inc.')
        self.assertEqual(result['current_price'], 150.0)
        
        # Verify mock was called
        mock_ticker_class.assert_called_once_with('AAPL')
    
    @patch('yfinance.Ticker')
    def test_yfinance_adapter_historical_prices(self, mock_ticker_class):
        """Test YFinanceAdapter historical prices retrieval."""
        # Set up mock
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.return_value = self.sample_historical_data
        
        # Test method
        result = self.yf_adapter.get_historical_prices('AAPL', period='1y', interval='1d')
        
        # Verify result
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)
        self.assertEqual(result['Close'].iloc[-1], 151.0)
        
        # Verify mock was called
        mock_ticker_class.assert_called_once_with('AAPL')
        mock_ticker.history.assert_called_once()
    
    @patch('stock_analysis.adapters.investing_adapter.InvestingAdapter._get_session')
    @patch('stock_analysis.adapters.investing_adapter.InvestingAdapter._make_request')
    def test_investing_adapter_security_info(self, mock_make_request, mock_get_session):
        """Test InvestingAdapter security info retrieval."""
        # Set up mock
        mock_get_session.return_value = MagicMock()
        mock_make_request.return_value = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'price': 150.0,
            'market_cap': 2500000000000,
            'pe_ratio': 25.0,
            'sector': 'Technology',
            'industry': 'Consumer Electronics'
        }
        
        # Test method
        result = self.investing_adapter.get_security_info('AAPL')
        
        # Verify result
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['name'], 'Apple Inc.')
        self.assertEqual(result['current_price'], 150.0)
        
        # Verify mock was called
        mock_make_request.assert_called_once()
    
    @patch('stock_analysis.adapters.investing_adapter.InvestingAdapter._get_session')
    @patch('stock_analysis.adapters.investing_adapter.InvestingAdapter._make_request')
    def test_investing_adapter_financial_statements(self, mock_make_request, mock_get_session):
        """Test InvestingAdapter financial statements retrieval."""
        # Set up mock
        mock_get_session.return_value = MagicMock()
        mock_make_request.return_value = self.sample_financial_statements.to_dict()
        
        # Test method
        result = self.investing_adapter.get_financial_statements('AAPL', statement_type='income', period='annual')
        
        # Verify result
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)
        
        # Verify mock was called
        mock_make_request.assert_called_once()
    
    @patch('requests.get')
    def test_alpha_vantage_adapter_security_info(self, mock_requests_get):
        """Test AlphaVantageAdapter security info retrieval."""
        # Set up mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'Global Quote': {
                '01. symbol': 'AAPL',
                '02. open': '145.0',
                '03. high': '150.0',
                '04. low': '144.0',
                '05. price': '150.0',
                '06. volume': '100000000',
                '07. latest trading day': '2023-01-03',
                '08. previous close': '149.0',
                '09. change': '1.0',
                '10. change percent': '0.67%'
            }
        }
        mock_requests_get.return_value = mock_response
        
        # Test method
        result = self.alpha_vantage_adapter.get_security_info('AAPL')
        
        # Verify result
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['current_price'], 150.0)
        
        # Verify mock was called
        mock_requests_get.assert_called_once()
    
    @patch('requests.get')
    def test_alpha_vantage_adapter_historical_prices(self, mock_requests_get):
        """Test AlphaVantageAdapter historical prices retrieval."""
        # Set up mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'Meta Data': {
                '1. Information': 'Daily Prices',
                '2. Symbol': 'AAPL'
            },
            'Time Series (Daily)': {
                '2023-01-03': {
                    '1. open': '145.0',
                    '2. high': '150.0',
                    '3. low': '144.0',
                    '4. close': '149.0',
                    '5. volume': '100000000'
                },
                '2023-01-02': {
                    '1. open': '146.0',
                    '2. high': '151.0',
                    '3. low': '145.0',
                    '4. close': '150.0',
                    '5. volume': '95000000'
                },
                '2023-01-01': {
                    '1. open': '147.0',
                    '2. high': '152.0',
                    '3. low': '146.0',
                    '4. close': '151.0',
                    '5. volume': '105000000'
                }
            }
        }
        mock_requests_get.return_value = mock_response
        
        # Test method
        result = self.alpha_vantage_adapter.get_historical_prices('AAPL', period='1y', interval='1d')
        
        # Verify result
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)
        
        # Verify mock was called
        mock_requests_get.assert_called_once()
    
    def test_adapter_feature_support(self):
        """Test adapter feature support reporting."""
        # Test YFinanceAdapter
        yf_features = self.yf_adapter.get_supported_features()
        self.assertIn('security_info', yf_features)
        self.assertIn('historical_prices', yf_features)
        
        # Test InvestingAdapter
        investing_features = self.investing_adapter.get_supported_features()
        self.assertIn('security_info', investing_features)
        self.assertIn('financial_statements', investing_features)
        
        # Test AlphaVantageAdapter
        av_features = self.alpha_vantage_adapter.get_supported_features()
        self.assertIn('security_info', av_features)
        self.assertIn('historical_prices', av_features)
    
    def test_adapter_error_handling(self):
        """Test adapter error handling."""
        # Test with invalid symbol
        with patch('yfinance.Ticker') as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker_class.return_value = mock_ticker
            mock_ticker.info = {}  # Empty info indicates invalid symbol
            
            with self.assertRaises(DataRetrievalError):
                self.yf_adapter.get_security_info('INVALID')
    
    def test_adapter_rate_limiting(self):
        """Test adapter rate limiting functionality."""
        # This test verifies that rate limiting is applied
        with patch('time.sleep') as mock_sleep, \
             patch('yfinance.Ticker') as mock_ticker_class:
            
            mock_ticker = MagicMock()
            mock_ticker_class.return_value = mock_ticker
            mock_ticker.info = {'symbol': 'AAPL', 'shortName': 'Apple Inc.'}
            
            # Make multiple requests in quick succession
            for _ in range(5):
                self.yf_adapter.get_security_info('AAPL')
            
            # Verify rate limiting was applied (sleep was called)
            self.assertTrue(mock_sleep.called)


if __name__ == '__main__':
    unittest.main()