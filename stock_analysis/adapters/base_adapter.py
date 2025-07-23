"""Base adapter interface for financial data sources.

This module defines the common interface that all financial data source
adapters must implement, ensuring consistency across different data providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

from stock_analysis.utils.logging import get_logger

logger = get_logger(__name__)


class FinancialDataAdapter(ABC):
    """Base interface for financial data source adapters.
    
    This abstract class defines the common methods that all data source
    adapters must implement to provide a unified interface for data retrieval.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initialize the adapter.
        
        Args:
            name: Name of the data source
            config: Configuration dictionary for the adapter
        """
        self.name = name
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.{name}")
        self._initialize_adapter()
    
    def _initialize_adapter(self) -> None:
        """Initialize adapter-specific configuration.
        
        This method can be overridden by subclasses to perform
        adapter-specific initialization.
        """
        pass
    
    @abstractmethod
    def get_security_info(self, symbol: str) -> Dict[str, Any]:
        """Retrieve basic security information.
        
        Args:
            symbol: Security ticker symbol
            
        Returns:
            Dictionary containing security information
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        pass
    
    @abstractmethod
    def get_historical_prices(
        self, 
        symbol: str, 
        period: str = "1y", 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Retrieve historical price data.
        
        Args:
            symbol: Security ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with historical price data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        pass
    
    @abstractmethod
    def get_financial_statements(
        self, 
        symbol: str, 
        statement_type: str = "income", 
        period: str = "annual",
        years: int = 5
    ) -> pd.DataFrame:
        """Retrieve financial statements.
        
        Args:
            symbol: Security ticker symbol
            statement_type: Type of statement ('income', 'balance', 'cash')
            period: Period ('annual' or 'quarterly')
            years: Number of years of historical data to retrieve
            
        Returns:
            DataFrame with financial statement data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        pass
    
    def get_technical_indicators(
        self, 
        symbol: str, 
        indicators: List[str] = None
    ) -> Dict[str, Any]:
        """Retrieve technical indicators.
        
        Args:
            symbol: Security ticker symbol
            indicators: List of technical indicators to retrieve
            
        Returns:
            Dictionary containing technical indicator values
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        # Default implementation returns empty dict
        # Subclasses should override if they support technical indicators
        self.logger.warning(f"Technical indicators not supported by {self.name} adapter")
        return {}
    
    def get_market_data(self, data_type: str) -> Dict[str, Any]:
        """Retrieve market data like indices, commodities, forex.
        
        Args:
            data_type: Type of market data ('indices', 'commodities', 'forex', 'sectors')
            
        Returns:
            Dictionary containing market data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        # Default implementation returns empty dict
        # Subclasses should override if they support market data
        self.logger.warning(f"Market data not supported by {self.name} adapter")
        return {}
    
    def get_news(
        self, 
        symbol: str = None, 
        category: str = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve news articles.
        
        Args:
            symbol: Security ticker symbol (optional)
            category: News category (optional)
            limit: Maximum number of articles to retrieve
            
        Returns:
            List of news articles
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        # Default implementation returns empty list
        # Subclasses should override if they support news data
        self.logger.warning(f"News data not supported by {self.name} adapter")
        return []
    
    def get_economic_data(
        self, 
        indicator: str, 
        region: str = None
    ) -> pd.DataFrame:
        """Retrieve economic indicators data.
        
        Args:
            indicator: Economic indicator name
            region: Geographic region (optional)
            
        Returns:
            DataFrame with economic indicator data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        # Default implementation returns empty DataFrame
        # Subclasses should override if they support economic data
        self.logger.warning(f"Economic data not supported by {self.name} adapter")
        return pd.DataFrame()
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a security symbol exists.
        
        Args:
            symbol: Security ticker symbol
            
        Returns:
            True if the symbol is valid, False otherwise
        """
        try:
            # Try to get basic security info to validate symbol
            info = self.get_security_info(symbol)
            return bool(info)
        except Exception as e:
            self.logger.debug(f"Symbol validation failed for {symbol}: {str(e)}")
            return False
    
    def get_supported_features(self) -> List[str]:
        """Get list of features supported by this adapter.
        
        Returns:
            List of supported feature names
        """
        features = ['security_info', 'historical_prices', 'financial_statements']
        
        # Check if adapter supports optional features by testing method implementations
        try:
            # Test if technical indicators are supported
            test_result = self.get_technical_indicators('TEST', [])
            if test_result or hasattr(self.__class__.get_technical_indicators, '__isabstractmethod__'):
                features.append('technical_indicators')
        except NotImplementedError:
            pass
        
        try:
            # Test if market data is supported
            test_result = self.get_market_data('indices')
            if test_result or hasattr(self.__class__.get_market_data, '__isabstractmethod__'):
                features.append('market_data')
        except NotImplementedError:
            pass
        
        try:
            # Test if news is supported
            test_result = self.get_news(limit=1)
            if test_result or hasattr(self.__class__.get_news, '__isabstractmethod__'):
                features.append('news')
        except NotImplementedError:
            pass
        
        try:
            # Test if economic data is supported
            test_result = self.get_economic_data('GDP')
            if not test_result.empty or hasattr(self.__class__.get_economic_data, '__isabstractmethod__'):
                features.append('economic_data')
        except NotImplementedError:
            pass
        
        return features
    
    def __str__(self) -> str:
        """String representation of the adapter."""
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the adapter."""
        features = self.get_supported_features()
        return f"{self.__class__.__name__}(name='{self.name}', features={features})"