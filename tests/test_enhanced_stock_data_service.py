"""Tests for the enhanced stock data service.

This module contains tests for the EnhancedStockDataService class.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime
import pytest

from stock_analysis.services.enhanced_stock_data_service import EnhancedStockDataService
from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
from stock_analysis.models.enhanced_data_models import EnhancedSecurityInfo
from stock_analysis.utils.exceptions import DataRetrievalError


class TestEnhancedStockDataService:
    """Test cases for the EnhancedStockDataService class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a mock integration service
        self.mock_integration_service = Mock(spec=FinancialDataIntegrationService)
        
        # Create the service with the mock integration service
        self.service = EnhancedStockDataService(integration_service=self.mock_integration_service)
        
        # Mock cache manager
        self.mock_cache = MagicMock()
        self.mock_cache.get.return_value = None
        
        # Sample data
        self.sample_symbol = "AAPL"
        
        # Sample enhanced security info data
        self.sample_enhanced_data = {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "current_price": 150.0,
            "market_cap": 2500000000000,
            "beta": 1.2,
            "earnings_growth": 0.15,
            "revenue_growth": 0.12,
            "profit_margin_trend": [0.21, 0.22, 0.23],
            "rsi_14": 65.5,
            "macd_line": 2.5,
            "signal_line": 1.8,
            "histogram": 0.7,
            "sma_20": 148.5,
            "sma_50": 145.2,
            "sma_200": 140.8,
            "analyst_rating": "Buy",
            "price_target_low": 140.0,
            "price_target_average": 160.0,
            "price_target_high": 180.0,
            "analyst_count": 30,
            "exchange": "NASDAQ",
            "currency": "USD",
            "company_description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
            "key_executives": [
                {"name": "Tim Cook", "title": "CEO"},
                {"name": "Luca Maestri", "title": "CFO"}
            ]
        }
        
        # Sample technical indicators data
        self.sample_technical_indicators = {
            "rsi_14": 65.5,
            "macd_line": 2.5,
            "signal_line": 1.8,
            "histogram": 0.7,
            "sma_20": 148.5,
            "sma_50": 145.2,
            "sma_200": 140.8
        }
        
        # Sample analyst data
        self.sample_analyst_data = {
            "analyst_rating": "Buy",
            "price_target": {
                "low": 140.0,
                "average": 160.0,
                "high": 180.0
            },
            "analyst_count": 30,
            "buy_ratings": 20,
            "hold_ratings": 8,
            "sell_ratings": 2
        }
        
        # Sample historical data
        self.sample_historical_data = pd.DataFrame({
            "Open": [145.0, 146.0, 147.0],
            "High": [150.0, 151.0, 152.0],
            "Low": [144.0, 145.0, 146.0],
            "Close": [149.0, 150.0, 151.0],
            "Volume": [1000000, 1100000, 1200000]
        }, index=pd.date_range(start="2023-01-01", periods=3))
    
    @patch("stock_analysis.services.enhanced_stock_data_service.get_cache_manager")
    def test_get_enhanced_security_info(self, mock_get_cache):
        """Test retrieving enhanced security info."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_enhanced_security_info.return_value = self.sample_enhanced_data
        
        # Execute
        result = self.service.get_enhanced_security_info(self.sample_symbol)
        
        # Verify
        assert isinstance(result, EnhancedSecurityInfo)
        assert result.symbol == self.sample_symbol
        assert result.name == "Apple Inc."
        assert result.current_price == 150.0
        assert result.market_cap == 2500000000000
        assert result.beta == 1.2
        assert result.earnings_growth == 0.15
        assert result.revenue_growth == 0.12
        assert result.profit_margin_trend == [0.21, 0.22, 0.23]
        assert result.rsi_14 == 65.5
        assert result.macd == {"macd_line": 2.5, "signal_line": 1.8, "histogram": 0.7}
        assert result.moving_averages == {"SMA20": 148.5, "SMA50": 145.2, "SMA200": 140.8}
        assert result.analyst_rating == "Buy"
        assert result.price_target == {"low": 140.0, "average": 160.0, "high": 180.0}
        assert result.analyst_count == 30
        assert result.exchange == "NASDAQ"
        assert result.currency == "USD"
        assert "Apple Inc. designs" in result.company_description
        assert len(result.key_executives) == 2
        assert result.key_executives[0]["name"] == "Tim Cook"
        
        # Verify integration service was called correctly
        self.mock_integration_service.get_enhanced_security_info.assert_called_once_with(
            self.sample_symbol, 
            include_technicals=True,
            include_analyst_data=True,
            use_cache=True
        )
        
        # Verify cache was checked and set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_called_once()
    
    @patch("stock_analysis.services.enhanced_stock_data_service.get_cache_manager")
    def test_get_enhanced_security_info_from_cache(self, mock_get_cache):
        """Test retrieving enhanced security info from cache."""
        # Setup
        cached_result = EnhancedSecurityInfo(
            symbol=self.sample_symbol,
            name="Apple Inc.",
            current_price=150.0
        )
        self.mock_cache.get.return_value = cached_result
        mock_get_cache.return_value = self.mock_cache
        
        # Execute
        result = self.service.get_enhanced_security_info(self.sample_symbol)
        
        # Verify
        assert result is cached_result
        
        # Verify integration service was not called
        self.mock_integration_service.get_enhanced_security_info.assert_not_called()
        
        # Verify cache was checked but not set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_not_called()
    
    @patch("stock_analysis.services.enhanced_stock_data_service.get_cache_manager")
    def test_get_enhanced_security_info_error(self, mock_get_cache):
        """Test error handling when retrieving enhanced security info."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_enhanced_security_info.side_effect = Exception("API error")
        
        # Execute and verify
        with pytest.raises(DataRetrievalError) as excinfo:
            self.service.get_enhanced_security_info(self.sample_symbol)
        
        # Verify error message
        assert "Failed to retrieve enhanced security info" in str(excinfo.value)
        assert self.sample_symbol in str(excinfo.value)
        
        # Verify integration service was called
        self.mock_integration_service.get_enhanced_security_info.assert_called_once()
    
    @patch("stock_analysis.services.enhanced_stock_data_service.get_cache_manager")
    def test_get_technical_indicators(self, mock_get_cache):
        """Test retrieving technical indicators."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_technical_indicators.return_value = self.sample_technical_indicators
        
        # Execute
        result = self.service.get_technical_indicators(self.sample_symbol)
        
        # Verify
        assert result == self.sample_technical_indicators
        
        # Verify integration service was called correctly
        self.mock_integration_service.get_technical_indicators.assert_called_once_with(
            self.sample_symbol, 
            indicators=['sma_20', 'sma_50', 'sma_200', 'rsi_14', 'macd'],
            use_cache=True
        )
        
        # Verify cache was checked and set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_called_once()
    
    @patch("stock_analysis.services.enhanced_stock_data_service.get_cache_manager")
    def test_get_technical_indicators_custom(self, mock_get_cache):
        """Test retrieving custom technical indicators."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_technical_indicators.return_value = {
            "rsi_14": 65.5,
            "ema_12": 149.2
        }
        
        # Execute
        result = self.service.get_technical_indicators(
            self.sample_symbol, 
            indicators=["rsi_14", "ema_12"]
        )
        
        # Verify
        assert result == {"rsi_14": 65.5, "ema_12": 149.2}
        
        # Verify integration service was called correctly
        self.mock_integration_service.get_technical_indicators.assert_called_once_with(
            self.sample_symbol, 
            indicators=["rsi_14", "ema_12"],
            use_cache=True
        )
    
    @patch("stock_analysis.services.enhanced_stock_data_service.get_cache_manager")
    def test_get_analyst_data(self, mock_get_cache):
        """Test retrieving analyst data."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service._get_analyst_data.return_value = self.sample_analyst_data
        
        # Execute
        result = self.service.get_analyst_data(self.sample_symbol)
        
        # Verify
        assert result == self.sample_analyst_data
        
        # Verify integration service was called correctly
        self.mock_integration_service._get_analyst_data.assert_called_once_with(
            self.sample_symbol, 
            use_cache=True
        )
        
        # Verify cache was checked and set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_called_once()
    
    @patch("stock_analysis.services.enhanced_stock_data_service.get_cache_manager")
    def test_get_historical_data_with_indicators(self, mock_get_cache):
        """Test retrieving historical data with indicators."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        
        # Mock the get_historical_data method
        self.service.get_historical_data = Mock(return_value=self.sample_historical_data)
        
        # Execute
        result = self.service.get_historical_data_with_indicators(
            self.sample_symbol,
            period="1y",
            interval="1d"
        )
        
        # Verify
        assert isinstance(result, pd.DataFrame)
        assert "SMA_20" in result.columns
        assert "SMA_50" in result.columns
        assert "SMA_200" in result.columns
        assert "RSI_14" in result.columns
        assert "MACD_Line" in result.columns
        assert "Signal_Line" in result.columns
        assert "MACD_Histogram" in result.columns
        
        # Verify get_historical_data was called correctly
        self.service.get_historical_data.assert_called_once_with(
            self.sample_symbol,
            period="1y",
            interval="1d",
            use_cache=True
        )
        
        # Verify cache was checked and set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_called_once()
    
    @patch("stock_analysis.services.enhanced_stock_data_service.get_cache_manager")
    def test_get_historical_data_with_custom_indicators(self, mock_get_cache):
        """Test retrieving historical data with custom indicators."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        
        # Mock the get_historical_data method
        self.service.get_historical_data = Mock(return_value=self.sample_historical_data)
        
        # Execute
        result = self.service.get_historical_data_with_indicators(
            self.sample_symbol,
            period="1y",
            interval="1d",
            indicators=["sma_20", "ema_12"]
        )
        
        # Verify
        assert isinstance(result, pd.DataFrame)
        assert "SMA_20" in result.columns
        assert "EMA_12" in result.columns
        assert "SMA_50" not in result.columns
        assert "RSI_14" not in result.columns
        
        # Verify get_historical_data was called correctly
        self.service.get_historical_data.assert_called_once_with(
            self.sample_symbol,
            period="1y",
            interval="1d",
            use_cache=True
        )
    
    def test_create_enhanced_security_info(self):
        """Test creating EnhancedSecurityInfo from dictionary data."""
        # Execute
        result = self.service._create_enhanced_security_info(
            self.sample_symbol,
            self.sample_enhanced_data
        )
        
        # Verify
        assert isinstance(result, EnhancedSecurityInfo)
        assert result.symbol == self.sample_symbol
        assert result.name == "Apple Inc."
        assert result.current_price == 150.0
        assert result.market_cap == 2500000000000
        assert result.beta == 1.2
        assert result.earnings_growth == 0.15
        assert result.revenue_growth == 0.12
        assert result.profit_margin_trend == [0.21, 0.22, 0.23]
        assert result.rsi_14 == 65.5
        assert result.macd == {"macd_line": 2.5, "signal_line": 1.8, "histogram": 0.7}
        assert result.moving_averages == {"SMA20": 148.5, "SMA50": 145.2, "SMA200": 140.8}
        assert result.analyst_rating == "Buy"
        assert result.price_target == {"low": 140.0, "average": 160.0, "high": 180.0}
        assert result.analyst_count == 30
        assert result.exchange == "NASDAQ"
        assert result.currency == "USD"
        assert "Apple Inc. designs" in result.company_description
        assert len(result.key_executives) == 2
        assert result.key_executives[0]["name"] == "Tim Cook"