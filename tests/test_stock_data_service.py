"""Tests for the StockDataService class."""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from datetime import datetime

from stock_analysis.services.stock_data_service import StockDataService
from stock_analysis.models.data_models import StockInfo
from stock_analysis.utils.exceptions import DataRetrievalError


class TestStockDataService(unittest.TestCase):
    """Test cases for StockDataService."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.service = StockDataService()
    
    @patch('yfinance.Ticker')
    def test_get_stock_info_success(self, mock_ticker):
        """Test successful retrieval of stock info."""
        # Mock the ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock the info property
        mock_ticker_instance.info = {
            'symbol': 'AAPL',
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
        
        # Call the method
        result = self.service.get_stock_info('AAPL')
        
        # Verify the result
        self.assertIsInstance(result, StockInfo)
        self.assertEqual(result.symbol, 'AAPL')
        self.assertEqual(result.company_name, 'Apple Inc.')
        self.assertEqual(result.current_price, 150.0)
        self.assertEqual(result.market_cap, 2500000000000)
        self.assertEqual(result.pe_ratio, 25.0)
        self.assertEqual(result.pb_ratio, 30.0)
        self.assertEqual(result.dividend_yield, 0.005)
        self.assertEqual(result.beta, 1.2)
        self.assertEqual(result.sector, 'Technology')
        self.assertEqual(result.industry, 'Consumer Electronics')
    
    @patch('yfinance.Ticker')
    def test_get_stock_info_missing_fields(self, mock_ticker):
        """Test stock info retrieval with missing fields."""
        # Mock the ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock the info property with minimal fields
        mock_ticker_instance.info = {
            'symbol': 'XYZ',
            'shortName': 'XYZ Corp',
            'regularMarketPrice': 50.0,
            'marketCap': 1000000000
        }
        
        # Call the method
        result = self.service.get_stock_info('XYZ')
        
        # Verify the result
        self.assertIsInstance(result, StockInfo)
        self.assertEqual(result.symbol, 'XYZ')
        self.assertEqual(result.company_name, 'XYZ Corp')
        self.assertEqual(result.current_price, 50.0)
        self.assertEqual(result.market_cap, 1000000000)
        self.assertIsNone(result.pe_ratio)
        self.assertIsNone(result.pb_ratio)
        self.assertIsNone(result.dividend_yield)
        self.assertIsNone(result.beta)
    
    @patch('yfinance.Ticker')
    def test_get_stock_info_error(self, mock_ticker):
        """Test error handling in stock info retrieval."""
        # Mock the ticker instance to raise an exception
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        mock_ticker_instance.info = MagicMock(side_effect=Exception("API Error"))
        
        # Call the method and check for exception
        with self.assertRaises(DataRetrievalError):
            self.service.get_stock_info('AAPL')
    
    @patch('yfinance.Ticker')
    def test_get_historical_data_success(self, mock_ticker):
        """Test successful retrieval of historical data."""
        # Mock the ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Create a sample DataFrame for historical data
        dates = pd.date_range(start='2023-01-01', periods=5)
        mock_history = pd.DataFrame({
            'Open': [150.0, 151.0, 152.0, 153.0, 154.0],
            'High': [155.0, 156.0, 157.0, 158.0, 159.0],
            'Low': [145.0, 146.0, 147.0, 148.0, 149.0],
            'Close': [153.0, 154.0, 155.0, 156.0, 157.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=dates)
        
        # Mock the history method
        mock_ticker_instance.history = MagicMock(return_value=mock_history)
        
        # Call the method
        result = self.service.get_historical_data('AAPL', period='1mo', interval='1d')
        
        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        self.assertTrue('Open' in result.columns)
        self.assertTrue('Close' in result.columns)
        
        # Verify the method was called with correct parameters
        mock_ticker_instance.history.assert_called_once_with(period='1mo', interval='1d')
    
    @patch('yfinance.Ticker')
    def test_get_historical_data_empty(self, mock_ticker):
        """Test handling of empty historical data."""
        # Mock the ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock the history method to return empty DataFrame
        mock_ticker_instance.history = MagicMock(return_value=pd.DataFrame())
        
        # Call the method and check for exception
        with self.assertRaises(DataRetrievalError):
            self.service.get_historical_data('AAPL')
    
    @patch('yfinance.Ticker')
    def test_get_financial_statements_success(self, mock_ticker):
        """Test successful retrieval of financial statements."""
        # Mock the ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Create a sample DataFrame for financial statements
        dates = [datetime(2020, 12, 31), datetime(2021, 12, 31)]
        mock_statement = pd.DataFrame({
            'Revenue': [100000000, 120000000],
            'CostOfRevenue': [50000000, 60000000],
            'GrossProfit': [50000000, 60000000],
            'NetIncome': [20000000, 25000000]
        }, index=dates)
        
        # Mock the income_stmt method
        mock_ticker_instance.income_stmt = MagicMock(return_value=mock_statement)
        
        # Call the method
        result = self.service.get_financial_statements('AAPL', statement_type='income', period='annual')
        
        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertTrue('Revenue' in result.columns)
        self.assertTrue('NetIncome' in result.columns)
        
        # Verify the method was called with correct parameters
        mock_ticker_instance.income_stmt.assert_called_once_with(period='annual')
    
    @patch('yfinance.Ticker')
    def test_get_financial_statements_invalid_type(self, mock_ticker):
        """Test handling of invalid statement type."""
        # Call the method with invalid statement type
        with self.assertRaises(ValueError):
            self.service.get_financial_statements('AAPL', statement_type='invalid')
    
    @patch('yfinance.Ticker')
    def test_validate_symbol_valid(self, mock_ticker):
        """Test validation of a valid symbol."""
        # Mock the ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock the info property
        mock_ticker_instance.info = {
            'regularMarketPrice': 150.0,
            'longName': 'Apple Inc.'
        }
        
        # Call the method
        result = self.service.validate_symbol('AAPL')
        
        # Verify the result
        self.assertTrue(result)
    
    @patch('yfinance.Ticker')
    def test_validate_symbol_invalid(self, mock_ticker):
        """Test validation of an invalid symbol."""
        # Mock the ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock the info property with minimal data (no price)
        mock_ticker_instance.info = {}
        
        # Call the method
        result = self.service.validate_symbol('INVALID')
        
        # Verify the result
        self.assertFalse(result)
    
    @patch('yfinance.Ticker')
    def test_validate_symbol_error(self, mock_ticker):
        """Test validation when an error occurs."""
        # Mock the ticker instance to raise an exception
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        mock_ticker_instance.info = MagicMock(side_effect=Exception("API Error"))
        
        # Call the method
        result = self.service.validate_symbol('AAPL')
        
        # Verify the result (should return False on error)
        self.assertFalse(result)
    
    @patch('time.sleep')
    def test_rate_limit(self, mock_sleep):
        """Test rate limiting functionality."""
        # Set a short delay for testing
        self.service.rate_limit_delay = 0.1
        
        # Call rate limit twice
        self.service._rate_limit()  # First call should not delay
        self.service._rate_limit()  # Second call should delay
        
        # Verify that sleep was called once (for the second call)
        self.assertEqual(mock_sleep.call_count, 1)
        
        # Verify sleep was called with a value close to our rate limit delay
        # The exact value will include some random jitter
        call_arg = mock_sleep.call_args[0][0]
        self.assertGreaterEqual(call_arg, 0.1)  # Should be at least our delay
    
    @patch('time.sleep')
    def test_retry_with_backoff(self, mock_sleep):
        """Test retry with backoff functionality."""
        # Create a mock function that fails twice then succeeds
        mock_func = MagicMock(side_effect=[Exception("First failure"), 
                                          Exception("Second failure"), 
                                          "Success"])
        
        # Call the retry method
        result = self.service._retry_with_backoff(mock_func)
        
        # Verify the result
        self.assertEqual(result, "Success")
        
        # Verify the function was called 3 times
        self.assertEqual(mock_func.call_count, 3)
        
        # Verify sleep was called twice (after first and second failures)
        self.assertEqual(mock_sleep.call_count, 2)
    
    @patch('time.sleep')
    def test_retry_with_backoff_all_failures(self, mock_sleep):
        """Test retry with backoff when all attempts fail."""
        # Set a small number of retry attempts
        self.service.retry_attempts = 2
        
        # Create a mock function that always fails
        mock_func = MagicMock(side_effect=Exception("Always fails"))
        
        # Call the retry method and check for exception
        with self.assertRaises(DataRetrievalError):
            self.service._retry_with_backoff(mock_func)
        
        # Verify the function was called the expected number of times
        self.assertEqual(mock_func.call_count, 2)
        
        # Verify sleep was called once (after first failure)
        self.assertEqual(mock_sleep.call_count, 1)


if __name__ == '__main__':
    unittest.main()