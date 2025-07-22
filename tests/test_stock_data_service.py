"""Unit tests for stock data service."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime

from stock_analysis.services.stock_data_service import StockDataService
from stock_analysis.models.data_models import StockInfo, ETFInfo
from stock_analysis.utils.exceptions import DataRetrievalError


@pytest.fixture
def mock_ticker():
    """Create a mock yfinance Ticker."""
    mock = MagicMock()
    mock.info = {
        'symbol': 'AAPL',
        'shortName': 'Apple Inc.',
        'longName': 'Apple Inc.',
        'currentPrice': 150.0,
        'marketCap': 2500000000000,
        'trailingPE': 25.0,
        'priceToBook': 15.0,
        'dividendYield': 0.005,
        'beta': 1.2,
        'sector': 'Technology',
        'industry': 'Consumer Electronics',
        'quoteType': 'EQUITY'
    }
    return mock


@pytest.fixture
def service():
    """Create a stock data service instance."""
    return StockDataService()


class TestStockDataService:
    """Test cases for StockDataService."""
    
    def test_get_financial_statements_success(self, service, mock_ticker):
        """Test successful retrieval of financial statements."""
        # Create a sample DataFrame for financial statements
        dates = [datetime(2020, 12, 31), datetime(2021, 12, 31)]
        mock_statement = pd.DataFrame({
            'Revenue': [100000000, 120000000],
            'NetIncome': [20000000, 25000000],
            'OperatingIncome': [30000000, 35000000],
            'GrossProfit': [50000000, 60000000]
        }, index=dates)
        
        # Mock the income_stmt property
        mock_ticker.income_stmt = mock_statement
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            # Call the method
            result = service.get_financial_statements('AAPL', statement_type='income', period='annual', use_cache=False)
            
            # Verify the result
            assert isinstance(result, pd.DataFrame)
            assert result.shape[0] == 2  # 2 time periods
            assert result.shape[1] == 4  # 4 metrics
            assert 'Revenue' in result.columns
            assert 'NetIncome' in result.columns
    
    def test_get_historical_data_empty(self, service, mock_ticker):
        """Test handling of empty historical data."""
        # Mock the history method to return empty DataFrame
        mock_ticker.history = MagicMock(return_value=pd.DataFrame())
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            with pytest.raises(DataRetrievalError) as exc_info:
                service.get_historical_data('AAPL', period='1mo', use_cache=False)
            assert "No historical data available" in str(exc_info.value)
    
    def test_get_security_info_error(self, service, mock_ticker):
        """Test error handling in security info retrieval."""
        # Mock the ticker instance to raise an exception
        mock_ticker.info = MagicMock(side_effect=Exception("API Error"))
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            with pytest.raises(DataRetrievalError) as exc_info:
                service.get_security_info('AAPL', use_cache=False)
            assert "Failed to retrieve security info" in str(exc_info.value)
    
    def test_get_security_info_stock_success(self, service, mock_ticker):
        """Test successful retrieval of stock info."""
        with patch('yfinance.Ticker', return_value=mock_ticker):
            result = service.get_security_info('AAPL', use_cache=False)
            
            assert isinstance(result, StockInfo)
            assert result.symbol == 'AAPL'
            assert result.name == 'Apple Inc.'
            assert result.current_price == 150.0
            assert result.market_cap == 2500000000000
            assert result.beta == 1.2
            assert result.pe_ratio == 25.0
            assert result.pb_ratio == 15.0
            assert result.dividend_yield == 0.005
            assert result.sector == 'Technology'
            assert result.industry == 'Consumer Electronics'
    
    def test_get_security_info_etf_success(self, service, mock_ticker):
        """Test successful retrieval of ETF info."""
        # Update mock info for ETF
        mock_ticker.info = {
            'symbol': 'SPY',
            'shortName': 'SPDR S&P 500 ETF Trust',
            'longName': 'SPDR S&P 500 ETF Trust',
            'currentPrice': 450.0,
            'marketCap': 450000000000,
            'beta': 1.0,
            'dividendYield': 0.015,
            'totalAssets': 450000000000,
            'navPrice': 450.0,
            'category': 'Large Blend',
            'quoteType': 'ETF',
            'fundFamily': 'State Street Global Advisors',
            'annualReportExpenseRatio': 0.0095,
            'holdings': [
                {'symbol': 'AAPL', 'holdingName': 'Apple Inc.', 'holdingPercent': 7.0},
                {'symbol': 'MSFT', 'holdingName': 'Microsoft Corp', 'holdingPercent': 6.0}
            ],
            'assetProfile': {
                'assetAllocation': {
                    'stocks': 0.98,
                    'bonds': 0.0,
                    'cash': 0.02
                }
            }
        }
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            result = service.get_security_info('SPY', use_cache=False)
            
            assert isinstance(result, ETFInfo)
            assert result.symbol == 'SPY'
            assert result.name == 'SPDR S&P 500 ETF Trust'
            assert result.current_price == 450.0
            assert result.market_cap == 450000000000
            assert result.beta == 1.0
            assert result.expense_ratio == 0.0095
            assert result.assets_under_management == 450000000000
            assert result.nav == 450.0
            assert result.category == 'Large Blend'
            assert result.dividend_yield == 0.015
            assert len(result.holdings) == 2
            assert result.holdings[0]['symbol'] == 'AAPL'
            assert result.holdings[0]['weight'] == 0.07
            assert result.asset_allocation['stocks'] == 0.98
    
    def test_is_etf_detection(self, service):
        """Test ETF detection logic."""
        # Test with explicit ETF type
        assert service._is_etf({'quoteType': 'ETF'})
        
        # Test with fund family
        assert service._is_etf({'fundFamily': 'Vanguard'})
        
        # Test with ETF in name
        assert service._is_etf({'shortName': 'Vanguard S&P 500 ETF'})
        
        # Test with non-ETF
        assert not service._is_etf({'quoteType': 'EQUITY'})
    
    @patch('time.sleep')
    def test_retry_with_backoff_success(self, mock_sleep, service):
        """Test retry with backoff success."""
        attempts = 0
        
        def test_func(operation_name):
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise Exception("Temporary error")
            return "success"
        
        result = service._retry_with_backoff(test_func, "test_operation")
        assert result == "success"
        assert attempts == 2
        mock_sleep.assert_called_once()
    
    @patch('time.sleep')
    def test_retry_with_backoff_all_failures(self, mock_sleep, service):
        """Test retry with backoff when all attempts fail."""
        def test_func(operation_name):
            raise Exception("Always fails")
        
        with pytest.raises(DataRetrievalError) as exc_info:
            service._retry_with_backoff(test_func, "test_operation")
        assert "Failed to retrieve data after 3 attempts" in str(exc_info.value)
        
        assert mock_sleep.call_count == 2  # Called for each retry except last attempt