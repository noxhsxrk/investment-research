"""Tests for the financial data integration service.

This module contains tests for the FinancialDataIntegrationService class,
which coordinates data retrieval from multiple sources.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime
import pytest

from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
from stock_analysis.adapters.base_adapter import FinancialDataAdapter
from stock_analysis.utils.exceptions import DataRetrievalError


class TestFinancialDataIntegrationService(unittest.TestCase):
    """Test cases for FinancialDataIntegrationService."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock adapters
        self.mock_adapter1 = Mock(spec=FinancialDataAdapter)
        self.mock_adapter1.name = "adapter1"
        self.mock_adapter1.get_supported_features.return_value = [
            'security_info', 'historical_prices', 'financial_statements',
            'technical_indicators', 'market_data', 'news'
        ]
        
        self.mock_adapter2 = Mock(spec=FinancialDataAdapter)
        self.mock_adapter2.name = "adapter2"
        self.mock_adapter2.get_supported_features.return_value = [
            'security_info', 'historical_prices', 'financial_statements'
        ]
        
        # Create service with mock adapters
        self.service = FinancialDataIntegrationService(
            adapters=[self.mock_adapter1, self.mock_adapter2]
        )
        
        # Override source priorities for testing
        self.service.source_priorities = {
            "security_info": {"adapter1": 1, "adapter2": 2},
            "historical_prices": {"adapter2": 1, "adapter1": 2},
            "financial_statements": {"adapter1": 1, "adapter2": 2},
            "technical_indicators": {"adapter1": 1, "adapter2": 2},
            "market_data": {"adapter1": 1, "adapter2": 2},
            "news": {"adapter1": 1, "adapter2": 2},
            "economic_data": {"adapter1": 1, "adapter2": 2},
        }
        
        # Sample test data
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
    
    def test_initialization(self):
        """Test service initialization."""
        # Test with provided adapters
        service = FinancialDataIntegrationService(
            adapters=[self.mock_adapter1, self.mock_adapter2]
        )
        self.assertEqual(len(service.adapters), 2)
        
        # Test with default adapters (requires mocking)
        with patch('stock_analysis.services.financial_data_integration_service.YFinanceAdapter') as mock_yf, \
             patch('stock_analysis.services.financial_data_integration_service.InvestingAdapter') as mock_inv, \
             patch('stock_analysis.services.financial_data_integration_service.AlphaVantageAdapter') as mock_av:
            
            mock_yf.return_value = self.mock_adapter1
            mock_inv.return_value = self.mock_adapter2
            mock_av.side_effect = Exception("Test exception")
            
            service = FinancialDataIntegrationService()
            self.assertEqual(len(service.adapters), 2)
    
    def test_get_adapter_by_name(self):
        """Test getting adapter by name."""
        adapter = self.service.get_adapter_by_name("adapter1")
        self.assertEqual(adapter, self.mock_adapter1)
        
        adapter = self.service.get_adapter_by_name("nonexistent")
        self.assertIsNone(adapter)
    
    def test_get_security_info_success(self):
        """Test getting security info successfully."""
        # Set up mock adapter responses
        self.mock_adapter1.get_security_info.return_value = self.sample_security_info
        
        # Test with cache disabled
        result = self.service.get_security_info("AAPL", use_cache=False)
        
        # Verify result
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['current_price'], 150.0)
        
        # Verify adapter was called
        self.mock_adapter1.get_security_info.assert_called_once_with("AAPL")
        self.mock_adapter2.get_security_info.assert_not_called()
    
    def test_get_security_info_fallback(self):
        """Test fallback to secondary adapter for security info."""
        # Set up mock adapter responses
        self.mock_adapter1.get_security_info.side_effect = DataRetrievalError("Test error")
        self.mock_adapter2.get_security_info.return_value = self.sample_security_info
        
        # Test with cache disabled
        result = self.service.get_security_info("AAPL", use_cache=False)
        
        # Verify result
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['current_price'], 150.0)
        
        # Verify both adapters were called
        self.mock_adapter1.get_security_info.assert_called_once_with("AAPL")
        self.mock_adapter2.get_security_info.assert_called_once_with("AAPL")
    
    def test_get_security_info_all_fail(self):
        """Test handling when all adapters fail for security info."""
        # Set up mock adapter responses
        self.mock_adapter1.get_security_info.side_effect = DataRetrievalError("Test error 1")
        self.mock_adapter2.get_security_info.side_effect = DataRetrievalError("Test error 2")
        
        # Test with cache disabled
        with self.assertRaises(DataRetrievalError):
            self.service.get_security_info("AAPL", use_cache=False)
        
        # Verify both adapters were called
        self.mock_adapter1.get_security_info.assert_called_once_with("AAPL")
        self.mock_adapter2.get_security_info.assert_called_once_with("AAPL")
    
    def test_get_security_info_combine(self):
        """Test combining partial security info from multiple sources."""
        # Set up mock adapter responses with partial data
        partial_info1 = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'current_price': 150.0
        }
        
        partial_info2 = {
            'symbol': 'AAPL',
            'market_cap': 2500000000000,
            'pe_ratio': 25.0,
            'sector': 'Technology'
        }
        
        self.mock_adapter1.get_security_info.return_value = partial_info1
        self.mock_adapter2.get_security_info.return_value = partial_info2
        
        # Override validation to treat partial results as invalid
        with patch.object(self.service, '_validate_security_info', return_value=False):
            # Test with cache disabled
            result = self.service.get_security_info("AAPL", use_cache=False)
            
            # Verify result contains combined data
            self.assertEqual(result['symbol'], 'AAPL')
            self.assertEqual(result['current_price'], 150.0)
            self.assertEqual(result['market_cap'], 2500000000000)
            self.assertEqual(result['sector'], 'Technology')
            
            # Verify both adapters were called
            self.mock_adapter1.get_security_info.assert_called_once_with("AAPL")
            self.mock_adapter2.get_security_info.assert_called_once_with("AAPL")
    
    def test_get_historical_prices_success(self):
        """Test getting historical prices successfully."""
        # Set up mock adapter responses
        self.mock_adapter2.get_historical_prices.return_value = self.sample_historical_data
        
        # Test with cache disabled
        result = self.service.get_historical_prices("AAPL", period="1y", interval="1d", use_cache=False)
        
        # Verify result
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)
        self.assertEqual(result['Close'].iloc[-1], 151.0)
        
        # Verify adapter was called
        self.mock_adapter2.get_historical_prices.assert_called_once_with("AAPL", "1y", "1d")
        self.mock_adapter1.get_historical_prices.assert_not_called()
    
    def test_get_financial_statements_success(self):
        """Test getting financial statements successfully."""
        # Set up mock adapter responses
        self.mock_adapter1.get_financial_statements.return_value = self.sample_financial_statements
        
        # Test with cache disabled
        result = self.service.get_financial_statements("AAPL", statement_type="income", period="annual", use_cache=False)
        
        # Verify result
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)
        self.assertEqual(result['Revenue'].iloc[0], 365000000000)
        
        # Verify adapter was called
        self.mock_adapter1.get_financial_statements.assert_called_once_with("AAPL", "income", "annual")
        self.mock_adapter2.get_financial_statements.assert_not_called()
    
    def test_get_technical_indicators_success(self):
        """Test getting technical indicators successfully."""
        # Set up mock adapter responses
        self.mock_adapter1.get_technical_indicators.return_value = self.sample_technical_indicators
        
        # Test with cache disabled
        result = self.service.get_technical_indicators("AAPL", indicators=['sma_20', 'rsi_14'], use_cache=False)
        
        # Verify result
        self.assertEqual(result['sma_20'], 148.5)
        self.assertEqual(result['rsi_14'], 65.3)
        
        # Verify adapter was called
        self.mock_adapter1.get_technical_indicators.assert_called_once_with("AAPL", ['sma_20', 'rsi_14'])
        self.mock_adapter2.get_technical_indicators.assert_not_called()
    
    def test_get_market_data_success(self):
        """Test getting market data successfully."""
        # Set up mock adapter responses
        self.mock_adapter1.get_market_data.return_value = self.sample_market_data
        
        # Test with cache disabled
        result = self.service.get_market_data("indices", use_cache=False)
        
        # Verify result
        self.assertEqual(result['S&P 500']['price'], 4500.0)
        self.assertEqual(result['NASDAQ']['price'], 14000.0)
        
        # Verify adapter was called
        self.mock_adapter1.get_market_data.assert_called_once_with("indices")
        self.mock_adapter2.get_market_data.assert_not_called()
    
    def test_get_news_success(self):
        """Test getting news successfully."""
        # Set up mock adapter responses
        self.mock_adapter1.get_news.return_value = self.sample_news
        
        # Test with cache disabled
        result = self.service.get_news("AAPL", limit=2, use_cache=False)
        
        # Verify result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['title'], 'Apple Announces New iPhone')
        
        # Verify adapter was called
        self.mock_adapter1.get_news.assert_called_once_with("AAPL", None, 2)
        self.mock_adapter2.get_news.assert_not_called()
    
    def test_get_enhanced_financial_data(self):
        """Test getting enhanced financial data."""
        # Set up mock adapter responses
        self.mock_adapter1.get_security_info.return_value = self.sample_security_info
        self.mock_adapter2.get_historical_prices.return_value = self.sample_historical_data
        self.mock_adapter1.get_financial_statements.return_value = self.sample_financial_statements
        self.mock_adapter1.get_technical_indicators.return_value = self.sample_technical_indicators
        
        # Mock cache to avoid actual caching
        with patch('stock_analysis.services.financial_data_integration_service.get_cache_manager'):
            # Test getting multiple data types
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
    
    def test_cache_usage(self):
        """Test cache usage."""
        # Set up mock adapter responses
        self.mock_adapter1.get_security_info.return_value = self.sample_security_info
        
        # Mock cache manager
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # First call returns None (cache miss)
        
        with patch('stock_analysis.services.financial_data_integration_service.get_cache_manager', return_value=mock_cache):
            # First call should use adapter
            self.service.get_security_info("AAPL", use_cache=True)
            
            # Verify cache was checked and set
            mock_cache.get.assert_called_once()
            mock_cache.set.assert_called_once()
            self.mock_adapter1.get_security_info.assert_called_once()
            
            # Reset mocks
            mock_cache.get.reset_mock()
            mock_cache.set.reset_mock()
            self.mock_adapter1.get_security_info.reset_mock()
            
            # Set up cache hit for second call
            mock_cache.get.return_value = self.sample_security_info
            
            # Second call should use cache
            result = self.service.get_security_info("AAPL", use_cache=True)
            
            # Verify cache was checked but not set, and adapter was not called
            mock_cache.get.assert_called_once()
            mock_cache.set.assert_not_called()
            self.mock_adapter1.get_security_info.assert_not_called()
            
            # Verify result
            self.assertEqual(result['symbol'], 'AAPL')
    
    def test_normalize_historical_data(self):
        """Test normalization of historical data."""
        # Create DataFrame with non-standard column names
        df = pd.DataFrame({
            'open': [145.0, 146.0, 147.0],
            'high': [150.0, 151.0, 152.0],
            'low': [144.0, 145.0, 146.0],
            'close': [149.0, 150.0, 151.0],
            'volume': [100000000, 95000000, 105000000]
        })
        
        # Normalize data
        result = self.service._normalize_historical_data(df)
        
        # Verify column names are standardized
        self.assertIn('Open', result.columns)
        self.assertIn('High', result.columns)
        self.assertIn('Low', result.columns)
        self.assertIn('Close', result.columns)
        self.assertIn('Volume', result.columns)
    
    def test_combine_technical_indicators(self):
        """Test combining technical indicators from multiple sources."""
        # Set up test data
        source1_data = {
            'sma_20': 148.5,
            'rsi_14': 65.3
        }
        
        source2_data = {
            'sma_20': 149.0,  # Different value
            'macd': {
                'macd': 2.5,
                'signal': 1.8,
                'histogram': 0.7
            }
        }
        
        results = {
            'adapter1': source1_data,
            'adapter2': source2_data
        }
        
        # Combine indicators
        combined = self.service._combine_technical_indicators(results, ['sma_20', 'rsi_14', 'macd'])
        
        # Verify result uses values from higher priority source when available
        self.assertEqual(combined['sma_20'], 148.5)  # From adapter1 (higher priority)
        self.assertEqual(combined['rsi_14'], 65.3)   # From adapter1 (only source)
        self.assertEqual(combined['macd']['macd'], 2.5)  # From adapter2 (only source)
        
        # Verify sources metadata
        self.assertIn('_sources', combined)
        self.assertEqual(combined['_sources']['sma_20'], 'adapter1')
        self.assertEqual(combined['_sources']['macd'], 'adapter2')
    
    def test_invalidate_cache(self):
        """Test cache invalidation."""
        # Mock cache manager
        mock_cache = MagicMock()
        mock_cache.invalidate_by_pattern.return_value = 5
        mock_cache.invalidate_by_type.return_value = 3
        mock_cache.invalidate_by_tag.return_value = 2
        
        with patch('stock_analysis.services.financial_data_integration_service.get_cache_manager', return_value=mock_cache):
            # Test invalidation by pattern
            count = self.service.invalidate_cache(pattern="security_*")
            self.assertEqual(count, 5)
            mock_cache.invalidate_by_pattern.assert_called_once_with("security_*")
            
            # Test invalidation by data type
            count = self.service.invalidate_cache(data_type="security_info")
            self.assertEqual(count, 3)
            mock_cache.invalidate_by_type.assert_called_once_with("security_info")
            
            # Test invalidation by tag
            count = self.service.invalidate_cache(tag="AAPL")
            self.assertEqual(count, 2)
            mock_cache.invalidate_by_tag.assert_called_once_with("AAPL")
            
            # Test clear all
            count = self.service.invalidate_cache()
            self.assertEqual(count, -1)
            mock_cache.clear.assert_called_once()


if __name__ == '__main__':
    unittest.main()