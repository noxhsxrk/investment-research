"""Tests for market data integration.

This module contains tests for the market data integration functionality
of the FinancialDataIntegrationService.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime
import pytest

from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
from stock_analysis.adapters.base_adapter import FinancialDataAdapter
from stock_analysis.utils.exceptions import DataRetrievalError
from stock_analysis.services.market_data_integration import validate_market_data, combine_market_data


class TestMarketDataIntegration(unittest.TestCase):
    """Test cases for market data integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock adapters
        self.mock_adapter1 = Mock(spec=FinancialDataAdapter)
        self.mock_adapter1.name = "adapter1"
        
        self.mock_adapter2 = Mock(spec=FinancialDataAdapter)
        self.mock_adapter2.name = "adapter2"
        
        # Create service with mock adapters
        self.service = FinancialDataIntegrationService(
            adapters=[self.mock_adapter1, self.mock_adapter2]
        )
        
        # Override source priorities for testing
        self.service.source_priorities = {
            "market_data": {"adapter1": 1, "adapter2": 2},
        }
        
        # Sample test data
        self.sample_indices = {
            'S&P 500': {
                'symbol': '^GSPC',
                'name': 'S&P 500',
                'price': 4500.0,
                'change': 15.0,
                'change_percent': 0.33
            },
            'NASDAQ': {
                'symbol': '^IXIC',
                'name': 'NASDAQ',
                'price': 14000.0,
                'change': 50.0,
                'change_percent': 0.36
            },
            'Dow Jones': {
                'symbol': '^DJI',
                'name': 'Dow Jones',
                'price': 35000.0,
                'change': 100.0,
                'change_percent': 0.29
            }
        }
        
        self.sample_commodities = {
            'Gold': {
                'symbol': 'GC=F',
                'name': 'Gold',
                'price': 1800.0,
                'change': 5.0,
                'change_percent': 0.28
            },
            'Silver': {
                'symbol': 'SI=F',
                'name': 'Silver',
                'price': 24.0,
                'change': 0.3,
                'change_percent': 1.27
            },
            'Crude Oil': {
                'symbol': 'CL=F',
                'name': 'Crude Oil',
                'price': 75.0,
                'change': -1.5,
                'change_percent': -1.96
            }
        }
        
        self.sample_forex = {
            'EUR/USD': {
                'symbol': 'EURUSD=X',
                'name': 'EUR/USD',
                'price': 1.18,
                'change': 0.002,
                'change_percent': 0.17
            },
            'GBP/USD': {
                'symbol': 'GBPUSD=X',
                'name': 'GBP/USD',
                'price': 1.38,
                'change': 0.005,
                'change_percent': 0.36
            }
        }
        
        self.sample_sectors = {
            'Technology': {
                'symbol': 'XLK',
                'name': 'Technology',
                'price': 150.0,
                'change': 1.5,
                'change_percent': 1.01
            },
            'Financial': {
                'symbol': 'XLF',
                'name': 'Financial',
                'price': 38.0,
                'change': 0.3,
                'change_percent': 0.8
            },
            'Healthcare': {
                'symbol': 'XLV',
                'name': 'Healthcare',
                'price': 130.0,
                'change': -0.5,
                'change_percent': -0.38
            },
            'Energy': {
                'symbol': 'XLE',
                'name': 'Energy',
                'price': 55.0,
                'change': -1.2,
                'change_percent': -2.14
            },
            'Industrial': {
                'symbol': 'XLI',
                'name': 'Industrial',
                'price': 105.0,
                'change': 0.8,
                'change_percent': 0.77
            }
        }
    
    def test_validate_market_data(self):
        """Test validation of market data."""
        # Valid indices data
        valid_indices = self.sample_indices.copy()
        self.assertTrue(validate_market_data(valid_indices, 'indices'))
        
        # Valid commodities data
        valid_commodities = self.sample_commodities.copy()
        self.assertTrue(validate_market_data(valid_commodities, 'commodities'))
        
        # Valid forex data
        valid_forex = self.sample_forex.copy()
        self.assertTrue(validate_market_data(valid_forex, 'forex'))
        
        # Valid sectors data
        valid_sectors = self.sample_sectors.copy()
        self.assertTrue(validate_market_data(valid_sectors, 'sectors'))
        
        # Invalid data - too few items
        invalid_indices = {
            'S&P 500': self.sample_indices['S&P 500']
        }
        self.assertFalse(validate_market_data(invalid_indices, 'indices'))
        
        # Invalid data - missing required field
        invalid_item = {
            'S&P 500': {
                'symbol': '^GSPC',
                'name': 'S&P 500',
                'change': 15.0,
                'change_percent': 0.33
            },
            'NASDAQ': self.sample_indices['NASDAQ']
        }
        self.assertFalse(validate_market_data(invalid_item, 'indices'))
        
        # Invalid data - missing optional fields
        invalid_optional = {
            'S&P 500': {
                'price': 4500.0
            },
            'NASDAQ': {
                'price': 14000.0
            }
        }
        self.assertFalse(validate_market_data(invalid_optional, 'indices'))
        
        # Empty data
        self.assertFalse(validate_market_data({}, 'indices'))
    
    def test_combine_market_data(self):
        """Test combining market data from multiple sources."""
        # Set up test data
        source1_data = {
            'S&P 500': self.sample_indices['S&P 500'],
            'NASDAQ': self.sample_indices['NASDAQ']
        }
        
        source2_data = {
            'S&P 500': {
                'symbol': '^GSPC',
                'name': 'S&P 500',
                'price': 4510.0,  # Different price
                'change': 25.0,   # Different change
                'change_percent': 0.56  # Different percent
            },
            'Dow Jones': self.sample_indices['Dow Jones']
        }
        
        results = {
            'adapter1': source1_data,
            'adapter2': source2_data
        }
        
        # Combine data
        combined = combine_market_data(results, 'indices')
        
        # Verify result contains data from both sources
        self.assertEqual(len(combined), 4)  # 3 indices + _sources
        self.assertIn('S&P 500', combined)
        self.assertIn('NASDAQ', combined)
        self.assertIn('Dow Jones', combined)
        
        # Verify data from first source (higher priority) is used for overlapping items
        self.assertEqual(combined['S&P 500']['price'], 4500.0)
        self.assertEqual(combined['S&P 500']['change'], 15.0)
        
        # Verify sources metadata
        self.assertIn('_sources', combined)
        self.assertListEqual(combined['_sources'], ['adapter1', 'adapter2'])
    
    def test_get_market_data_success(self):
        """Test getting market data successfully."""
        # Set up mock adapter responses
        self.mock_adapter1.get_market_data.return_value = self.sample_indices
        
        # Mock cache to avoid actual caching
        with patch('stock_analysis.services.financial_data_integration_service.get_cache_manager'):
            # Mock validate_market_data to return True for the test data
            with patch('stock_analysis.services.market_data_integration.validate_market_data', return_value=True):
                # Mock combine_market_data to avoid calling it
                with patch('stock_analysis.services.market_data_integration.combine_market_data', return_value=self.sample_indices):
                    # Test getting market data
                    result = self.service.get_market_data('indices', use_cache=False)
                    
                    # Verify result
                    self.assertIn('S&P 500', result)
                    self.assertIn('NASDAQ', result)
                    self.assertIn('Dow Jones', result)
                    self.assertEqual(result['S&P 500']['price'], 4500.0)
                    self.assertEqual(result['NASDAQ']['price'], 14000.0)
                    self.assertEqual(result['Dow Jones']['price'], 35000.0)
                    
                    # Verify adapter was called
                    self.mock_adapter1.get_market_data.assert_called_once_with('indices')
                    self.mock_adapter2.get_market_data.assert_not_called()
    
    def test_get_market_data_fallback(self):
        """Test fallback to secondary adapter for market data."""
        # Set up mock adapter responses
        self.mock_adapter1.get_market_data.side_effect = DataRetrievalError("Test error")
        self.mock_adapter2.get_market_data.return_value = self.sample_commodities
        
        # Mock cache to avoid actual caching
        with patch('stock_analysis.services.financial_data_integration_service.get_cache_manager'):
            # Mock validate_market_data to return True for the test data
            with patch('stock_analysis.services.market_data_integration.validate_market_data', return_value=True):
                # Mock combine_market_data to return the sample data directly
                with patch('stock_analysis.services.market_data_integration.combine_market_data', return_value=self.sample_commodities):
                    # Test getting market data
                    result = self.service.get_market_data('commodities', use_cache=False)
                    
                    # Verify result
                    self.assertIn('Gold', result)
                    self.assertIn('Silver', result)
                    self.assertIn('Crude Oil', result)
                    self.assertEqual(result['Gold']['price'], 1800.0)
                    self.assertEqual(result['Silver']['price'], 24.0)
                    self.assertEqual(result['Crude Oil']['price'], 75.0)
                    
                    # Verify both adapters were called
                    self.mock_adapter1.get_market_data.assert_called_once_with('commodities')
                    self.mock_adapter2.get_market_data.assert_called_once_with('commodities')
    
    def test_get_market_data_combine(self):
        """Test combining market data from multiple sources."""
        # Set up mock adapter responses with partial data
        partial_data1 = {
            'EUR/USD': self.sample_forex['EUR/USD']
        }
        
        partial_data2 = {
            'GBP/USD': self.sample_forex['GBP/USD']
        }
        
        self.mock_adapter1.get_market_data.return_value = partial_data1
        self.mock_adapter2.get_market_data.return_value = partial_data2
        
        # Override validation to treat partial results as invalid
        with patch('stock_analysis.services.market_data_integration.validate_market_data', return_value=False):
            # Mock cache to avoid actual caching
            with patch('stock_analysis.services.financial_data_integration_service.get_cache_manager'):
                # Test getting market data
                result = self.service.get_market_data('forex', use_cache=False)
                
                # Verify result contains combined data
                self.assertEqual(len(result), 3)  # 2 forex pairs + _sources
                self.assertEqual(result['EUR/USD']['price'], 1.18)
                self.assertEqual(result['GBP/USD']['price'], 1.38)
                
                # Verify both adapters were called
                self.mock_adapter1.get_market_data.assert_called_once_with('forex')
                self.mock_adapter2.get_market_data.assert_called_once_with('forex')
    
    def test_get_market_data_all_fail(self):
        """Test handling when all adapters fail for market data."""
        # Set up mock adapter responses
        self.mock_adapter1.get_market_data.side_effect = DataRetrievalError("Test error 1")
        self.mock_adapter2.get_market_data.side_effect = DataRetrievalError("Test error 2")
        
        # Mock cache to avoid actual caching
        with patch('stock_analysis.services.financial_data_integration_service.get_cache_manager'):
            # Test getting market data
            with self.assertRaises(DataRetrievalError):
                self.service.get_market_data('sectors', use_cache=False)
            
            # Verify both adapters were called
            self.mock_adapter1.get_market_data.assert_called_once_with('sectors')
            self.mock_adapter2.get_market_data.assert_called_once_with('sectors')
    
    def test_get_market_data_cache(self):
        """Test cache usage for market data."""
        # Set up mock adapter responses
        self.mock_adapter1.get_market_data.return_value = self.sample_indices
        
        # Mock cache manager
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # First call returns None (cache miss)
        
        with patch('stock_analysis.services.financial_data_integration_service.get_cache_manager', return_value=mock_cache):
            # Mock validate_market_data to return True for the test data
            with patch('stock_analysis.services.market_data_integration.validate_market_data', return_value=True):
                # Mock combine_market_data to avoid calling it
                with patch('stock_analysis.services.market_data_integration.combine_market_data', return_value=self.sample_indices):
                    # First call should use adapter
                    self.service.get_market_data('indices', use_cache=True)
                    
                    # Verify cache was checked and set
                    mock_cache.get.assert_called_once()
                    mock_cache.set.assert_called_once()
                    self.mock_adapter1.get_market_data.assert_called_once()
                    
                    # Reset mocks
                    mock_cache.get.reset_mock()
                    mock_cache.set.reset_mock()
                    self.mock_adapter1.get_market_data.reset_mock()
                    
                    # Set up cache hit for second call
                    mock_cache.get.return_value = self.sample_indices
                    
                    # Second call should use cache
                    result = self.service.get_market_data('indices', use_cache=True)
                    
                    # Verify cache was checked but not set, and adapter was not called
                    mock_cache.get.assert_called_once()
                    mock_cache.set.assert_not_called()
                    self.mock_adapter1.get_market_data.assert_not_called()
                    
                    # Verify result
                    self.assertEqual(result['S&P 500']['price'], 4500.0)


if __name__ == '__main__':
    unittest.main()