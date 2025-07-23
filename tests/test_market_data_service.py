"""Tests for the market data service.

This module contains tests for the MarketDataService class.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime
import pytest

from stock_analysis.services.market_data_service import MarketDataService
from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
from stock_analysis.models.enhanced_data_models import MarketData
from stock_analysis.utils.exceptions import DataRetrievalError


class TestMarketDataService:
    """Test cases for the MarketDataService class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a mock integration service
        self.mock_integration_service = Mock(spec=FinancialDataIntegrationService)
        
        # Create the service with the mock integration service
        self.service = MarketDataService(integration_service=self.mock_integration_service)
        
        # Mock cache manager
        self.mock_cache = MagicMock()
        self.mock_cache.get.return_value = None
        
        # Sample data
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
        
        self.sample_economic_indicators = {
            'GDP Growth Rate': {
                'value': 2.1,
                'previous': 1.8,
                'forecast': 2.0,
                'unit': '%',
                'period': 'Q2 2023',
                'region': 'US'
            },
            'Inflation Rate': {
                'value': 3.2,
                'previous': 3.7,
                'forecast': 3.3,
                'unit': '%',
                'period': 'Sep 2023',
                'region': 'US'
            },
            'Unemployment Rate': {
                'value': 3.8,
                'previous': 3.7,
                'forecast': 3.8,
                'unit': '%',
                'period': 'Sep 2023',
                'region': 'US'
            },
            'Interest Rate': {
                'value': 5.5,
                'previous': 5.5,
                'forecast': 5.5,
                'unit': '%',
                'period': 'Sep 2023',
                'region': 'US'
            }
        }
    
    @patch("stock_analysis.services.market_data_service.get_cache_manager")
    def test_get_market_overview(self, mock_get_cache):
        """Test retrieving market overview."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_market_data.side_effect = lambda data_type, use_cache: {
            'indices': self.sample_indices,
            'commodities': self.sample_commodities,
            'forex': self.sample_forex,
            'sectors': self.sample_sectors
        }[data_type]
        self.mock_integration_service.get_economic_data.return_value = self.sample_economic_indicators
        
        # Execute
        result = self.service.get_market_overview()
        
        # Verify
        assert isinstance(result, MarketData)
        assert len(result.indices) == 3
        assert len(result.commodities) == 3
        assert len(result.forex) == 2
        assert len(result.sector_performance) == 5
        assert len(result.economic_indicators) == 4
        
        # Verify integration service was called correctly
        assert self.mock_integration_service.get_market_data.call_count == 4
        self.mock_integration_service.get_economic_data.assert_called_once()
        
        # Verify cache was checked and set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_called_once()
    
    @patch("stock_analysis.services.market_data_service.get_cache_manager")
    def test_get_market_overview_from_cache(self, mock_get_cache):
        """Test retrieving market overview from cache."""
        # Setup
        cached_result = MarketData(
            indices={'S&P 500': {'value': 4500.0, 'change': 15.0, 'change_percent': 0.33}},
            commodities={'Gold': {'value': 1800.0, 'change': 5.0, 'change_percent': 0.28}},
            forex={'EUR/USD': {'value': 1.18, 'change': 0.002, 'change_percent': 0.17}},
            sector_performance={'Technology': 1.01},
            economic_indicators={'GDP Growth Rate': {'value': 2.1, 'previous': 1.8, 'forecast': 2.0, 'unit': '%'}}
        )
        self.mock_cache.get.return_value = cached_result
        mock_get_cache.return_value = self.mock_cache
        
        # Execute
        result = self.service.get_market_overview()
        
        # Verify
        assert result is cached_result
        
        # Verify integration service was not called
        self.mock_integration_service.get_market_data.assert_not_called()
        self.mock_integration_service.get_economic_data.assert_not_called()
        
        # Verify cache was checked but not set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_not_called()
    
    @patch("stock_analysis.services.market_data_service.get_cache_manager")
    def test_get_market_overview_error(self, mock_get_cache):
        """Test error handling when retrieving market overview."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_market_data.side_effect = Exception("API error")
        
        # Execute and verify
        with pytest.raises(DataRetrievalError) as excinfo:
            self.service.get_market_overview()
        
        # Verify error message
        assert "Failed to retrieve market overview" in str(excinfo.value)
        
        # Verify integration service was called
        self.mock_integration_service.get_market_data.assert_called_once()
    
    @patch("stock_analysis.services.market_data_service.get_cache_manager")
    def test_get_economic_indicators(self, mock_get_cache):
        """Test retrieving economic indicators."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_economic_data.return_value = self.sample_economic_indicators
        
        # Execute
        result = self.service.get_economic_indicators()
        
        # Verify
        assert len(result) == 4
        assert 'GDP Growth Rate' in result
        assert 'Inflation Rate' in result
        assert 'Unemployment Rate' in result
        assert 'Interest Rate' in result
        assert result['GDP Growth Rate']['value'] == 2.1
        assert result['Inflation Rate']['value'] == 3.2
        
        # Verify integration service was called correctly
        self.mock_integration_service.get_economic_data.assert_called_once_with(region=None, use_cache=True)
        
        # Verify cache was checked and set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_called_once()
    
    @patch("stock_analysis.services.market_data_service.get_cache_manager")
    def test_get_economic_indicators_with_region(self, mock_get_cache):
        """Test retrieving economic indicators for a specific region."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_economic_data.return_value = {
            'GDP Growth Rate': {
                'value': 1.5,
                'previous': 1.2,
                'forecast': 1.6,
                'unit': '%',
                'period': 'Q2 2023',
                'region': 'EU'
            },
            'Inflation Rate': {
                'value': 2.9,
                'previous': 3.1,
                'forecast': 2.8,
                'unit': '%',
                'period': 'Sep 2023',
                'region': 'EU'
            }
        }
        
        # Execute
        result = self.service.get_economic_indicators(region='EU')
        
        # Verify
        assert len(result) == 2
        assert 'GDP Growth Rate' in result
        assert 'Inflation Rate' in result
        assert result['GDP Growth Rate']['value'] == 1.5
        assert result['Inflation Rate']['value'] == 2.9
        
        # Verify integration service was called correctly
        self.mock_integration_service.get_economic_data.assert_called_once_with(region='EU', use_cache=True)
    
    @patch("stock_analysis.services.market_data_service.get_cache_manager")
    def test_get_sector_performance(self, mock_get_cache):
        """Test retrieving sector performance."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_market_data.return_value = self.sample_sectors
        
        # Execute
        result = self.service.get_sector_performance()
        
        # Verify
        assert len(result) == 5
        assert 'Technology' in result
        assert 'Financial' in result
        assert 'Healthcare' in result
        assert 'Energy' in result
        assert 'Industrial' in result
        assert result['Technology'] == 1.01
        assert result['Energy'] == -2.14
        
        # Verify integration service was called correctly
        self.mock_integration_service.get_market_data.assert_called_once_with('sectors', use_cache=True)
        
        # Verify cache was checked and set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_called_once()
    
    @patch("stock_analysis.services.market_data_service.get_cache_manager")
    def test_get_commodity_prices(self, mock_get_cache):
        """Test retrieving commodity prices."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_market_data.return_value = self.sample_commodities
        
        # Execute
        result = self.service.get_commodity_prices()
        
        # Verify
        assert len(result) == 3
        assert 'Gold' in result
        assert 'Silver' in result
        assert 'Crude Oil' in result
        assert result['Gold']['value'] == 1800.0
        assert result['Silver']['value'] == 24.0
        assert result['Crude Oil']['value'] == 75.0
        
        # Verify integration service was called correctly
        self.mock_integration_service.get_market_data.assert_called_once_with('commodities', use_cache=True)
        
        # Verify cache was checked and set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_called_once()
    
    @patch("stock_analysis.services.market_data_service.get_cache_manager")
    def test_get_commodity_prices_filtered(self, mock_get_cache):
        """Test retrieving filtered commodity prices."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_market_data.return_value = self.sample_commodities
        
        # Execute
        result = self.service.get_commodity_prices(commodities=['Gold', 'Silver'])
        
        # Verify
        assert len(result) == 2
        assert 'Gold' in result
        assert 'Silver' in result
        assert 'Crude Oil' not in result
        assert result['Gold']['value'] == 1800.0
        assert result['Silver']['value'] == 24.0
        
        # Verify integration service was called correctly
        self.mock_integration_service.get_market_data.assert_called_once_with('commodities', use_cache=True)
    
    @patch("stock_analysis.services.market_data_service.get_cache_manager")
    def test_get_market_indices(self, mock_get_cache):
        """Test retrieving market indices."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_market_data.return_value = self.sample_indices
        
        # Execute
        result = self.service.get_market_indices()
        
        # Verify
        assert len(result) == 3
        assert 'S&P 500' in result
        assert 'NASDAQ' in result
        assert 'Dow Jones' in result
        assert result['S&P 500']['value'] == 4500.0
        assert result['NASDAQ']['value'] == 14000.0
        assert result['Dow Jones']['value'] == 35000.0
        
        # Verify integration service was called correctly
        self.mock_integration_service.get_market_data.assert_called_once_with('indices', use_cache=True)
        
        # Verify cache was checked and set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_called_once()
    
    @patch("stock_analysis.services.market_data_service.get_cache_manager")
    def test_get_market_indices_filtered(self, mock_get_cache):
        """Test retrieving filtered market indices."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_market_data.return_value = self.sample_indices
        
        # Execute
        result = self.service.get_market_indices(indices=['S&P 500', 'NASDAQ'])
        
        # Verify
        assert len(result) == 2
        assert 'S&P 500' in result
        assert 'NASDAQ' in result
        assert 'Dow Jones' not in result
        assert result['S&P 500']['value'] == 4500.0
        assert result['NASDAQ']['value'] == 14000.0
        
        # Verify integration service was called correctly
        self.mock_integration_service.get_market_data.assert_called_once_with('indices', use_cache=True)
    
    def test_normalize_market_data(self):
        """Test normalizing market data."""
        # Setup
        input_data = {
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
                'last': 14000.0,  # Different field name
                'change': 50.0,
                'pct_change': 0.36  # Different field name
            },
            '_sources': ['adapter1', 'adapter2']  # Metadata field
        }
        
        # Execute
        result = self.service._normalize_market_data(input_data)
        
        # Verify
        assert len(result) == 2  # Metadata field should be skipped
        assert 'S&P 500' in result
        assert 'NASDAQ' in result
        assert result['S&P 500']['value'] == 4500.0
        assert result['NASDAQ']['value'] == 14000.0
        assert result['S&P 500']['change'] == 15.0
        assert result['NASDAQ']['change'] == 50.0
        assert result['S&P 500']['change_percent'] == 0.33
        assert result['NASDAQ']['change_percent'] == 0.36
    
    def test_extract_sector_performance(self):
        """Test extracting sector performance."""
        # Setup
        input_data = {
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
                'pct_change': 0.8  # Different field name
            },
            'Energy': {
                'symbol': 'XLE',
                'name': 'Energy',
                'price': 55.0,
                'change': -1.2,
                # No change_percent or pct_change
            },
            '_sources': ['adapter1']  # Metadata field
        }
        
        # Execute
        result = self.service._extract_sector_performance(input_data)
        
        # Verify
        assert len(result) == 3
        assert 'Technology' in result
        assert 'Financial' in result
        assert 'Energy' in result
        assert result['Technology'] == 1.01
        assert result['Financial'] == 0.8
        assert abs(result['Energy'] - (-2.18)) < 0.01  # Calculated from change/price
    
    def test_normalize_economic_indicators(self):
        """Test normalizing economic indicators."""
        # Setup
        input_data = {
            'GDP Growth Rate': {
                'value': 2.1,
                'previous': 1.8,
                'forecast': 2.0,
                'unit': '%'
            },
            'Inflation Rate': {
                'current': 3.2,  # Different field name
                'previous': 3.7,
                'forecast': 3.3,
                # No unit
            },
            '_sources': ['adapter1']  # Metadata field
        }
        
        # Execute
        result = self.service._normalize_economic_indicators(input_data)
        
        # Verify
        assert len(result) == 2
        assert 'GDP Growth Rate' in result
        assert 'Inflation Rate' in result
        assert result['GDP Growth Rate']['value'] == 2.1
        assert result['Inflation Rate']['value'] == 3.2
        assert result['GDP Growth Rate']['unit'] == '%'
        assert result['Inflation Rate']['unit'] == '%'  # Default unit