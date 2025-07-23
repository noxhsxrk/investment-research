"""Tests for security info integration.

This module contains tests for the security info integration functionality
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


class TestSecurityInfoIntegration(unittest.TestCase):
    """Test cases for security info integration."""
    
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
            "security_info": {"adapter1": 1, "adapter2": 2},
            "technical_indicators": {"adapter1": 1, "adapter2": 2},
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
        
        self.sample_technical_indicators = {
            'sma_20': 148.5,
            'sma_50': 145.2,
            'sma_200': 140.0,
            'rsi_14': 65.3,
            'macd': {
                'macd': 2.5,
                'signal': 1.8,
                'histogram': 0.7
            }
        }
        
        self.sample_analyst_data = {
            'analyst_rating': 'Buy',
            'price_target': {
                'low': 140.0,
                'average': 165.0,
                'high': 190.0
            },
            'analyst_count': 35,
            'buy_ratings': 25,
            'hold_ratings': 8,
            'sell_ratings': 2
        }
    
    def test_validate_security_info(self):
        """Test validation of security info."""
        # Valid complete data
        valid_data = self.sample_security_info.copy()
        self.assertTrue(self.service._validate_security_info(valid_data))
        
        # Missing required field
        invalid_data = self.sample_security_info.copy()
        del invalid_data['current_price']
        self.assertFalse(self.service._validate_security_info(invalid_data))
        
        # Minimal valid data
        minimal_data = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'current_price': 150.0,
            'market_cap': 2500000000000,
            'pe_ratio': 25.0
        }
        self.assertTrue(self.service._validate_security_info(minimal_data))
        
        # Too few optional fields
        sparse_data = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'current_price': 150.0,
            'market_cap': 2500000000000
        }
        self.assertFalse(self.service._validate_security_info(sparse_data))
    
    def test_combine_security_info(self):
        """Test combining security info from multiple sources."""
        # Set up test data
        source1_data = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'current_price': 150.0,
            'market_cap': 2500000000000
        }
        
        source2_data = {
            'symbol': 'AAPL',
            'pe_ratio': 25.0,
            'sector': 'Technology',
            'industry': 'Consumer Electronics'
        }
        
        partial_results = {
            'adapter1': source1_data,
            'adapter2': source2_data
        }
        
        # Combine data
        combined = self.service._combine_security_info(partial_results)
        
        # Verify result contains data from both sources
        self.assertEqual(combined['symbol'], 'AAPL')
        self.assertEqual(combined['current_price'], 150.0)
        self.assertEqual(combined['pe_ratio'], 25.0)
        self.assertEqual(combined['sector'], 'Technology')
        
        # Verify sources metadata
        self.assertIn('_sources', combined)
        self.assertListEqual(combined['_sources'], ['adapter1', 'adapter2'])
    
    def test_get_enhanced_security_info(self):
        """Test getting enhanced security info."""
        # Set up mock adapter responses
        self.mock_adapter1.get_security_info.return_value = self.sample_security_info
        
        # Mock the get_technical_indicators method to return sample data
        with patch.object(self.service, 'get_technical_indicators', return_value=self.sample_technical_indicators):
            # Add analyst data to security info
            security_info_with_analyst = self.sample_security_info.copy()
            security_info_with_analyst.update(self.sample_analyst_data)
            self.mock_adapter2.get_security_info.return_value = security_info_with_analyst
            
            # Mock cache to avoid actual caching
            with patch('stock_analysis.services.financial_data_integration_service.get_cache_manager'):
                # Test getting enhanced security info
                result = self.service.get_enhanced_security_info(
                    "AAPL", 
                    include_technicals=True,
                    include_analyst_data=True,
                    use_cache=False
                )
                
                # Verify result contains base security info
                self.assertEqual(result['symbol'], 'AAPL')
                self.assertEqual(result['current_price'], 150.0)
                
                # Verify result contains technical indicators
                self.assertEqual(result['sma_20'], 148.5)
                self.assertEqual(result['rsi_14'], 65.3)
                self.assertEqual(result['macd']['macd'], 2.5)
                
                # Verify analyst data was extracted
                self.assertEqual(result['analyst_rating'], 'Buy')
                self.assertEqual(result['price_target']['average'], 165.0)
                self.assertEqual(result['analyst_count'], 35)
    
    def test_get_enhanced_security_info_without_technicals(self):
        """Test getting enhanced security info without technical indicators."""
        # Set up mock adapter responses
        self.mock_adapter1.get_security_info.return_value = self.sample_security_info
        
        # Add analyst data to security info
        security_info_with_analyst = self.sample_security_info.copy()
        security_info_with_analyst.update(self.sample_analyst_data)
        self.mock_adapter2.get_security_info.return_value = security_info_with_analyst
        
        # Mock cache to avoid actual caching
        with patch('stock_analysis.services.financial_data_integration_service.get_cache_manager'):
            # Test getting enhanced security info without technicals
            result = self.service.get_enhanced_security_info(
                "AAPL", 
                include_technicals=False,
                include_analyst_data=True,
                use_cache=False
            )
            
            # Verify result contains base security info
            self.assertEqual(result['symbol'], 'AAPL')
            self.assertEqual(result['current_price'], 150.0)
            
            # Verify result does not contain technical indicators
            self.assertNotIn('sma_20', result)
            self.assertNotIn('rsi_14', result)
            
            # Verify analyst data was extracted
            self.assertEqual(result['analyst_rating'], 'Buy')
            self.assertEqual(result['price_target']['average'], 165.0)
    
    def test_get_analyst_data(self):
        """Test getting analyst data."""
        # Set up mock adapter responses
        security_info_with_analyst = self.sample_security_info.copy()
        security_info_with_analyst.update(self.sample_analyst_data)
        self.mock_adapter1.get_security_info.return_value = security_info_with_analyst
        
        # Mock cache to avoid actual caching
        with patch('stock_analysis.services.financial_data_integration_service.get_cache_manager'):
            # Test getting analyst data
            result = self.service._get_analyst_data("AAPL", use_cache=False)
            
            # Verify result contains analyst data
            self.assertEqual(result['analyst_rating'], 'Buy')
            self.assertEqual(result['price_target']['average'], 165.0)
            self.assertEqual(result['analyst_count'], 35)
            self.assertEqual(result['buy_ratings'], 25)
            self.assertEqual(result['hold_ratings'], 8)
            self.assertEqual(result['sell_ratings'], 2)
    
    def test_get_analyst_data_not_available(self):
        """Test getting analyst data when not available."""
        # Set up mock adapter responses without analyst data
        self.mock_adapter1.get_security_info.return_value = self.sample_security_info
        self.mock_adapter2.get_security_info.return_value = self.sample_security_info
        
        # Mock cache to avoid actual caching
        with patch('stock_analysis.services.financial_data_integration_service.get_cache_manager'):
            # Test getting analyst data
            result = self.service._get_analyst_data("AAPL", use_cache=False)
            
            # Verify result is empty
            self.assertEqual(result, {})
    
    def test_get_enhanced_security_info_error_handling(self):
        """Test error handling in enhanced security info."""
        # Set up mock adapter responses
        self.mock_adapter1.get_security_info.return_value = self.sample_security_info
        self.mock_adapter1.get_technical_indicators.side_effect = DataRetrievalError("Test error")
        
        # Mock cache to avoid actual caching
        with patch('stock_analysis.services.financial_data_integration_service.get_cache_manager'):
            # Test getting enhanced security info with technical indicator error
            result = self.service.get_enhanced_security_info(
                "AAPL", 
                include_technicals=True,
                include_analyst_data=True,
                use_cache=False
            )
            
            # Verify result contains base security info despite technical indicator error
            self.assertEqual(result['symbol'], 'AAPL')
            self.assertEqual(result['current_price'], 150.0)
            
            # Verify result does not contain technical indicators
            self.assertNotIn('sma_20', result)
            self.assertNotIn('rsi_14', result)


if __name__ == '__main__':
    unittest.main()