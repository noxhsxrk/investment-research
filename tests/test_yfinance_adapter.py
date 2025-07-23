"""Unit tests for YFinance adapter.

This module contains unit tests for the YFinanceAdapter class.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime

from stock_analysis.adapters.yfinance_adapter import YFinanceAdapter
from stock_analysis.models.data_models import StockInfo, ETFInfo
from stock_analysis.utils.exceptions import DataRetrievalError


class TestYFinanceAdapter(unittest.TestCase):
    """Test cases for YFinanceAdapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = YFinanceAdapter()
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        self.assertEqual(self.adapter.name, 'yfinance')
        self.assertIsNotNone(self.adapter.stock_service)
        self.assertIn('security_info', self.adapter.get_supported_features())
        self.assertIn('historical_prices', self.adapter.get_supported_features())
        self.assertIn('financial_statements', self.adapter.get_supported_features())
    
    @patch('stock_analysis.adapters.yfinance_adapter.StockDataService')
    def test_get_security_info_stock(self, mock_service_class):
        """Test getting security info for a stock."""
        # Mock the stock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Create mock stock info
        mock_stock_info = StockInfo(
            company_name='Apple Inc.',
            symbol='AAPL',
            current_price=150.0,
            market_cap=2500000000000,
            beta=1.2,
            pe_ratio=25.0,
            pb_ratio=5.0,
            dividend_yield=0.02,
            sector='Technology',
            industry='Consumer Electronics'
        )
        
        mock_service.get_security_info.return_value = mock_stock_info
        
        # Create new adapter instance to use mocked service
        adapter = YFinanceAdapter()
        
        # Test getting security info
        result = adapter.get_security_info('AAPL')
        
        # Verify results
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['name'], 'Apple Inc.')
        self.assertEqual(result['current_price'], 150.0)
        self.assertEqual(result['pe_ratio'], 25.0)
        self.assertEqual(result['sector'], 'Technology')
        
        # Verify service was called correctly
        mock_service.get_security_info.assert_called_once_with('AAPL')
    
    @patch('stock_analysis.adapters.yfinance_adapter.StockDataService')
    def test_get_security_info_etf(self, mock_service_class):
        """Test getting security info for an ETF."""
        # Mock the stock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Create mock ETF info
        mock_etf_info = ETFInfo(
            symbol='SPY',
            name='SPDR S&P 500 ETF Trust',
            current_price=400.0,
            market_cap=None,
            beta=1.0,
            expense_ratio=0.0945,
            assets_under_management=350000000000,
            nav=399.5,
            category='Large Blend'
        )
        
        mock_service.get_security_info.return_value = mock_etf_info
        
        # Create new adapter instance to use mocked service
        adapter = YFinanceAdapter()
        
        # Test getting security info
        result = adapter.get_security_info('SPY')
        
        # Verify results
        self.assertEqual(result['symbol'], 'SPY')
        self.assertEqual(result['name'], 'SPDR S&P 500 ETF Trust')
        self.assertEqual(result['current_price'], 400.0)
        self.assertEqual(result['expense_ratio'], 0.0945)
        self.assertEqual(result['nav'], 399.5)
        
        # Verify service was called correctly
        mock_service.get_security_info.assert_called_once_with('SPY')
    
    @patch('stock_analysis.adapters.yfinance_adapter.StockDataService')
    def test_get_historical_prices(self, mock_service_class):
        """Test getting historical price data."""
        # Mock the stock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Create mock historical data
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        mock_service.get_historical_data.return_value = mock_data
        
        # Create new adapter instance to use mocked service
        adapter = YFinanceAdapter()
        
        # Test getting historical data
        result = adapter.get_historical_prices('AAPL', '1y', '1d')
        
        # Verify results
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn('Close', result.columns)
        
        # Verify service was called correctly
        mock_service.get_historical_data.assert_called_once_with('AAPL', '1y', '1d')
    
    @patch('stock_analysis.adapters.yfinance_adapter.StockDataService')
    def test_get_financial_statements(self, mock_service_class):
        """Test getting financial statements."""
        # Mock the stock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Create mock financial statement data
        mock_data = pd.DataFrame({
            '2023-12-31': [1000000, 500000, 300000],
            '2022-12-31': [900000, 450000, 250000],
            '2021-12-31': [800000, 400000, 200000]
        }, index=['Total Revenue', 'Cost of Revenue', 'Gross Profit'])
        
        mock_service.get_financial_statements.return_value = mock_data
        
        # Create new adapter instance to use mocked service
        adapter = YFinanceAdapter()
        
        # Test getting financial statements
        result = adapter.get_financial_statements('AAPL', 'income', 'annual')
        
        # Verify results
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result.columns), 3)
        self.assertIn('Total Revenue', result.index)
        
        # Verify service was called correctly
        mock_service.get_financial_statements.assert_called_once_with('AAPL', 'income', 'annual')
    
    def test_calculate_sma(self):
        """Test SMA calculation."""
        # Create test data
        test_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        })
        
        # Test SMA calculation
        sma_5 = self.adapter._calculate_sma(test_data, 5)
        expected_sma = test_data['Close'].tail(5).mean()
        
        self.assertAlmostEqual(sma_5, expected_sma, places=2)
    
    def test_calculate_rsi(self):
        """Test RSI calculation."""
        # Create test data with some price movement
        test_data = pd.DataFrame({
            'Close': [100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108, 107]
        })
        
        # Test RSI calculation
        rsi = self.adapter._calculate_rsi(test_data, 14)
        
        # RSI should be between 0 and 100
        self.assertIsNotNone(rsi)
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)
    
    def test_calculate_macd(self):
        """Test MACD calculation."""
        # Create test data
        test_data = pd.DataFrame({
            'Close': list(range(100, 130))  # 30 data points
        })
        
        # Test MACD calculation
        macd = self.adapter._calculate_macd(test_data)
        
        # Verify MACD components
        self.assertIn('macd', macd)
        self.assertIn('signal', macd)
        self.assertIn('histogram', macd)
        self.assertIsNotNone(macd['macd'])
    
    @patch('yfinance.Ticker')
    def test_get_market_indices(self, mock_ticker):
        """Test getting market indices data."""
        # Mock ticker info
        mock_info = {
            'regularMarketPrice': 4500.0,
            'regularMarketChange': 25.0,
            'regularMarketChangePercent': 0.56
        }
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance
        
        # Test getting market indices
        result = self.adapter._get_market_indices()
        
        # Verify results
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        
        # Check that at least one index is present
        if 'S&P 500' in result:
            self.assertEqual(result['S&P 500']['price'], 4500.0)
    
    def test_validate_symbol(self):
        """Test symbol validation."""
        # Mock the stock service validate_symbol method
        with patch.object(self.adapter.stock_service, 'validate_symbol') as mock_validate:
            mock_validate.return_value = True
            
            result = self.adapter.validate_symbol('AAPL')
            self.assertTrue(result)
            
            mock_validate.assert_called_once_with('AAPL')
    
    def test_normalize_symbol_input(self):
        """Test that symbols are normalized before processing."""
        with patch.object(self.adapter.stock_service, 'validate_symbol') as mock_validate:
            mock_validate.return_value = True
            
            # Test with lowercase and whitespace
            result = self.adapter.validate_symbol('  aapl  ')
            
            # Verify that the symbol was normalized to uppercase
            mock_validate.assert_called_once_with('AAPL')
    
    def test_error_handling(self):
        """Test error handling in adapter methods."""
        # Mock the stock service to raise an exception
        with patch.object(self.adapter.stock_service, 'get_security_info') as mock_get_info:
            mock_get_info.side_effect = Exception("Test error")
            
            # Test that DataRetrievalError is raised
            with self.assertRaises(DataRetrievalError):
                self.adapter.get_security_info('INVALID')


if __name__ == '__main__':
    unittest.main()