"""Tests for technical indicators integration."""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime

from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
from stock_analysis.utils.exceptions import DataRetrievalError


class TestTechnicalIndicatorsIntegration(unittest.TestCase):
    """Test cases for technical indicators integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock adapters
        self.mock_adapter1 = MagicMock()
        self.mock_adapter1.name = "adapter1"
        
        self.mock_adapter2 = MagicMock()
        self.mock_adapter2.name = "adapter2"
        
        self.mock_adapter3 = MagicMock()
        self.mock_adapter3.name = "adapter3"
        
        # Create service with mock adapters
        self.service = FinancialDataIntegrationService(
            adapters=[self.mock_adapter1, self.mock_adapter2, self.mock_adapter3]
        )
        
        # Mock source priorities
        self.service.source_priorities = {
            "technical_indicators": {"adapter1": 1, "adapter2": 2, "adapter3": 3}
        }
        
        # Mock cache
        self.mock_cache = MagicMock()
        self.service.cache = self.mock_cache
        
        # Sample technical indicators data
        self.sample_technical_indicators = {
            "sma_20": 148.5,
            "sma_50": 142.3,
            "sma_200": 135.7,
            "rsi_14": 65.3,
            "macd": {
                "macd": 2.5,
                "signal": 1.8,
                "histogram": 0.7
            }
        }
        
        # Sample historical data for calculating indicators
        self.sample_historical_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range(start='2023-01-01', periods=5))

    def test_get_technical_indicators_from_cache(self):
        """Test retrieving technical indicators from cache."""
        # Set up mock cache response
        self.mock_cache.get.return_value = self.sample_technical_indicators
        
        # Call the method
        result = self.service.get_technical_indicators("AAPL", indicators=['sma_20', 'rsi_14'])
        
        # Verify cache was checked
        self.mock_cache.get.assert_called_once()
        
        # Verify adapters were not called
        self.mock_adapter1.get_technical_indicators.assert_not_called()
        self.mock_adapter2.get_technical_indicators.assert_not_called()
        
        # Verify result
        self.assertEqual(result, self.sample_technical_indicators)

    def test_get_technical_indicators_from_primary_source(self):
        """Test retrieving technical indicators from primary source."""
        # Set up mock cache and adapter responses
        self.mock_cache.get.return_value = None
        self.mock_adapter1.get_technical_indicators.return_value = self.sample_technical_indicators
        
        # Call the method
        result = self.service.get_technical_indicators("AAPL", indicators=['sma_20', 'rsi_14'], use_cache=True)
        
        # Verify cache was checked and set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_called_once()
        
        # Verify adapter was called
        self.mock_adapter1.get_technical_indicators.assert_called_once_with("AAPL", ['sma_20', 'rsi_14'])
        
        # Verify result
        self.assertEqual(result['sma_20'], 148.5)
        self.assertEqual(result['rsi_14'], 65.3)

    def test_get_technical_indicators_fallback(self):
        """Test fallback to secondary source when primary fails."""
        # Set up mock cache and adapter responses
        self.mock_cache.get.return_value = None
        self.mock_adapter1.get_technical_indicators.side_effect = Exception("Test error")
        self.mock_adapter2.get_technical_indicators.return_value = {
            "sma_20": 149.0,
            "rsi_14": 66.0
        }
        
        # Call the method
        result = self.service.get_technical_indicators("AAPL", indicators=['sma_20', 'rsi_14'], use_cache=True)
        
        # Verify adapters were called in priority order
        self.mock_adapter1.get_technical_indicators.assert_called_once()
        self.mock_adapter2.get_technical_indicators.assert_called_once()
        
        # Verify result
        self.assertEqual(result['sma_20'], 149.0)
        self.assertEqual(result['rsi_14'], 66.0)

    def test_get_technical_indicators_combine_sources(self):
        """Test combining technical indicators from multiple sources."""
        # Set up mock cache and adapter responses
        self.mock_cache.get.return_value = None
        
        # First adapter provides some indicators
        self.mock_adapter1.get_technical_indicators.return_value = {
            "sma_20": 148.5,
            "rsi_14": 65.3
        }
        
        # Second adapter provides different indicators
        self.mock_adapter2.get_technical_indicators.return_value = {
            "sma_50": 142.3,
            "macd": {
                "macd": 2.5,
                "signal": 1.8,
                "histogram": 0.7
            }
        }
        
        # Call the method
        result = self.service.get_technical_indicators(
            "AAPL", 
            indicators=['sma_20', 'sma_50', 'rsi_14', 'macd'], 
            use_cache=True
        )
        
        # Verify adapters were called
        self.mock_adapter1.get_technical_indicators.assert_called_once()
        self.mock_adapter2.get_technical_indicators.assert_called_once()
        
        # Verify result combines data from both sources
        self.assertEqual(result['sma_20'], 148.5)  # From adapter1
        self.assertEqual(result['rsi_14'], 65.3)   # From adapter1
        self.assertEqual(result['sma_50'], 142.3)  # From adapter2
        self.assertEqual(result['macd']['macd'], 2.5)  # From adapter2
        
        # Verify sources metadata
        self.assertIn('_sources', result)
        self.assertEqual(result['_sources']['sma_20'], 'adapter1')
        self.assertEqual(result['_sources']['sma_50'], 'adapter2')

    def test_get_technical_indicators_all_sources_fail(self):
        """Test error handling when all sources fail."""
        # Set up mock cache and adapter responses
        self.mock_cache.get.return_value = None
        self.mock_adapter1.get_technical_indicators.side_effect = Exception("Error 1")
        self.mock_adapter2.get_technical_indicators.side_effect = Exception("Error 2")
        self.mock_adapter3.get_technical_indicators.side_effect = Exception("Error 3")
        
        # Call the method and expect exception
        with self.assertRaises(DataRetrievalError):
            self.service.get_technical_indicators("AAPL", indicators=['sma_20', 'rsi_14'], use_cache=True)
        
        # Verify all adapters were called
        self.mock_adapter1.get_technical_indicators.assert_called_once()
        self.mock_adapter2.get_technical_indicators.assert_called_once()
        self.mock_adapter3.get_technical_indicators.assert_called_once()

    def test_validate_technical_indicators(self):
        """Test validation of technical indicators."""
        # Test with valid data
        valid_data = {
            "sma_20": 148.5,
            "rsi_14": 65.3
        }
        self.assertTrue(self.service._validate_technical_indicators(valid_data, ['sma_20', 'rsi_14']))
        
        # Test with missing indicators
        invalid_data = {
            "sma_20": 148.5
        }
        self.assertFalse(self.service._validate_technical_indicators(invalid_data, ['sma_20', 'rsi_14']))
        
        # Test with None values
        invalid_data = {
            "sma_20": 148.5,
            "rsi_14": None
        }
        self.assertFalse(self.service._validate_technical_indicators(invalid_data, ['sma_20', 'rsi_14']))

    def test_normalize_technical_indicators(self):
        """Test normalization of technical indicators."""
        # Test with various formats of indicators
        raw_data = {
            "SMA20": 148.5,
            "RSI": 65.3,
            "MACD": {
                "macd_line": 2.5,
                "signal_line": 1.8,
                "histogram": 0.7
            }
        }
        
        normalized = self.service._normalize_technical_indicators(raw_data)
        
        # Verify normalization
        self.assertEqual(normalized["sma_20"], 148.5)
        self.assertEqual(normalized["rsi_14"], 65.3)
        self.assertEqual(normalized["macd"]["macd"], 2.5)
        self.assertEqual(normalized["macd"]["signal"], 1.8)
        self.assertEqual(normalized["macd"]["histogram"], 0.7)

if __name__ == '__main__':
    unittest.main()