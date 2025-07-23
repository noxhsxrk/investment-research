"""Enhanced stock data service for retrieving comprehensive financial data.

This module provides an enhanced service for retrieving stock data from multiple sources,
with additional capabilities for technical indicators and analyst data.
"""

import logging
from typing import Dict, List, Optional, Union, Any

import pandas as pd

from stock_analysis.services.stock_data_service import StockDataService
from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
from stock_analysis.models.enhanced_data_models import EnhancedSecurityInfo
from stock_analysis.utils.exceptions import DataRetrievalError
from stock_analysis.utils.logging import get_logger, log_api_call, log_data_quality_issue
from stock_analysis.utils.performance_metrics import monitor_performance
from stock_analysis.utils.cache_manager import get_cache_manager

logger = get_logger(__name__)


class EnhancedStockDataService(StockDataService):
    """Enhanced service for retrieving comprehensive stock data."""
    
    def __init__(self, integration_service: Optional[FinancialDataIntegrationService] = None):
        """Initialize the enhanced stock data service.
        
        Args:
            integration_service: Financial data integration service instance
        """
        super().__init__()
        
        # Initialize integration service if not provided
        if integration_service is None:
            self.integration_service = FinancialDataIntegrationService()
        else:
            self.integration_service = integration_service
    
    @monitor_performance('get_enhanced_security_info')
    def get_enhanced_security_info(
        self, 
        symbol: str, 
        include_technicals: bool = True,
        include_analyst_data: bool = True,
        use_cache: bool = True
    ) -> EnhancedSecurityInfo:
        """Get enhanced security information with additional data points.
        
        Args:
            symbol: Security ticker symbol
            include_technicals: Whether to include technical indicators
            include_analyst_data: Whether to include analyst recommendations and price targets
            use_cache: Whether to use cached data if available
            
        Returns:
            EnhancedSecurityInfo object with enhanced security information
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.set_context(symbol=symbol, operation='get_enhanced_security_info')
        logger.info(f"Retrieving enhanced security info for {symbol}")
        
        # Generate cache key
        cache_key = f"enhanced_security_info:{symbol}:{include_technicals}:{include_analyst_data}"
        cache = get_cache_manager()
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved enhanced security info for {symbol} from cache")
                return cached_data
        
        try:
            # Get enhanced security info from integration service
            enhanced_data = self.integration_service.get_enhanced_security_info(
                symbol, 
                include_technicals=include_technicals,
                include_analyst_data=include_analyst_data,
                use_cache=use_cache
            )
            
            # Convert dictionary to EnhancedSecurityInfo object
            enhanced_security_info = self._create_enhanced_security_info(symbol, enhanced_data)
            
            # Validate the enhanced security info
            enhanced_security_info.validate()
            
            # Cache the result
            if use_cache:
                cache.set(
                    cache_key, 
                    enhanced_security_info, 
                    data_type="enhanced_security_info", 
                    tags=[symbol, "enhanced_security_info"]
                )
            
            logger.info(f"Successfully retrieved enhanced security info for {symbol}")
            return enhanced_security_info
            
        except Exception as e:
            logger.error(f"Error retrieving enhanced security info for {symbol}: {str(e)}")
            raise DataRetrievalError(
                f"Failed to retrieve enhanced security info for {symbol}: {str(e)}",
                symbol=symbol,
                data_source='multiple',
                original_exception=e
            )
    
    def _create_enhanced_security_info(self, symbol: str, data: Dict[str, Any]) -> EnhancedSecurityInfo:
        """Create EnhancedSecurityInfo object from dictionary data.
        
        Args:
            symbol: Security ticker symbol
            data: Dictionary containing security information
            
        Returns:
            EnhancedSecurityInfo object
        """
        # Extract technical indicators
        moving_averages = {}
        if 'sma_20' in data:
            moving_averages['SMA20'] = data.get('sma_20')
        if 'sma_50' in data:
            moving_averages['SMA50'] = data.get('sma_50')
        if 'sma_200' in data:
            moving_averages['SMA200'] = data.get('sma_200')
        if 'ema_12' in data:
            moving_averages['EMA12'] = data.get('ema_12')
        if 'ema_26' in data:
            moving_averages['EMA26'] = data.get('ema_26')
        
        # Extract MACD data
        macd_data = None
        if all(k in data for k in ['macd_line', 'signal_line', 'histogram']):
            macd_data = {
                'macd_line': data.get('macd_line'),
                'signal_line': data.get('signal_line'),
                'histogram': data.get('histogram')
            }
        
        # Extract price target data
        price_target = None
        if all(k in data for k in ['price_target_low', 'price_target_average', 'price_target_high']):
            price_target = {
                'low': data.get('price_target_low'),
                'average': data.get('price_target_average'),
                'high': data.get('price_target_high')
            }
        elif 'price_target' in data and isinstance(data['price_target'], dict):
            price_target = data['price_target']
        
        # Extract key executives
        key_executives = data.get('key_executives')
        if not key_executives and 'executives' in data:
            key_executives = data['executives']
        
        # Create EnhancedSecurityInfo object
        return EnhancedSecurityInfo(
            # Base SecurityInfo fields
            symbol=symbol,
            name=data.get('name', data.get('company_name', symbol)),
            current_price=data.get('current_price', data.get('price', 0.0)),
            market_cap=data.get('market_cap'),
            beta=data.get('beta'),
            
            # Additional fundamental data
            earnings_growth=data.get('earnings_growth'),
            revenue_growth=data.get('revenue_growth'),
            profit_margin_trend=data.get('profit_margin_trend'),
            
            # Technical indicators
            rsi_14=data.get('rsi_14'),
            macd=macd_data,
            moving_averages=moving_averages if moving_averages else None,
            
            # Analyst data
            analyst_rating=data.get('analyst_rating'),
            price_target=price_target,
            analyst_count=data.get('analyst_count'),
            
            # Additional metadata
            exchange=data.get('exchange'),
            currency=data.get('currency'),
            company_description=data.get('company_description', data.get('description')),
            key_executives=key_executives
        )
    
    @monitor_performance('get_technical_indicators')
    def get_technical_indicators(
        self, 
        symbol: str, 
        indicators: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Get technical indicators for a security.
        
        Args:
            symbol: Security ticker symbol
            indicators: List of technical indicators to retrieve (if None, retrieves all available)
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing technical indicators
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.set_context(symbol=symbol, operation='get_technical_indicators')
        logger.info(f"Retrieving technical indicators for {symbol}")
        
        # Default indicators if not specified
        if indicators is None:
            indicators = ['sma_20', 'sma_50', 'sma_200', 'rsi_14', 'macd']
        
        # Generate cache key
        cache_key = f"technical_indicators:{symbol}:{','.join(sorted(indicators))}"
        cache = get_cache_manager()
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved technical indicators for {symbol} from cache")
                return cached_data
        
        try:
            # Get technical indicators from integration service
            technical_indicators = self.integration_service.get_technical_indicators(
                symbol, 
                indicators=indicators,
                use_cache=use_cache
            )
            
            # Cache the result
            if use_cache:
                cache.set(
                    cache_key, 
                    technical_indicators, 
                    data_type="technical_indicators", 
                    tags=[symbol, "technical_indicators"]
                )
            
            logger.info(f"Successfully retrieved technical indicators for {symbol}")
            return technical_indicators
            
        except Exception as e:
            logger.error(f"Error retrieving technical indicators for {symbol}: {str(e)}")
            raise DataRetrievalError(
                f"Failed to retrieve technical indicators for {symbol}: {str(e)}",
                symbol=symbol,
                data_source='multiple',
                original_exception=e
            )
    
    @monitor_performance('get_analyst_data')
    def get_analyst_data(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """Get analyst recommendations and price targets.
        
        Args:
            symbol: Security ticker symbol
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing analyst data
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.set_context(symbol=symbol, operation='get_analyst_data')
        logger.info(f"Retrieving analyst data for {symbol}")
        
        # Generate cache key
        cache_key = f"analyst_data:{symbol}"
        cache = get_cache_manager()
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved analyst data for {symbol} from cache")
                return cached_data
        
        try:
            # Get analyst data from integration service
            analyst_data = self.integration_service._get_analyst_data(
                symbol, 
                use_cache=use_cache
            )
            
            # Cache the result
            if use_cache:
                cache.set(
                    cache_key, 
                    analyst_data, 
                    data_type="analyst_data", 
                    tags=[symbol, "analyst_data"]
                )
            
            logger.info(f"Successfully retrieved analyst data for {symbol}")
            return analyst_data
            
        except Exception as e:
            logger.error(f"Error retrieving analyst data for {symbol}: {str(e)}")
            raise DataRetrievalError(
                f"Failed to retrieve analyst data for {symbol}: {str(e)}",
                symbol=symbol,
                data_source='multiple',
                original_exception=e
            )
    
    @monitor_performance('get_historical_data_with_indicators')
    def get_historical_data_with_indicators(
        self, 
        symbol: str, 
        period: str = "1y", 
        interval: str = "1d",
        indicators: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get historical price data with technical indicators.
        
        Args:
            symbol: Security ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            indicators: List of technical indicators to include (if None, includes common ones)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with historical price data and indicators
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.set_context(symbol=symbol, operation='get_historical_data_with_indicators')
        logger.info(f"Retrieving historical data with indicators for {symbol}")
        
        # Default indicators if not specified
        if indicators is None:
            indicators = ['sma_20', 'sma_50', 'sma_200', 'rsi_14', 'macd']
        
        # Generate cache key
        cache_key = f"historical_with_indicators:{symbol}:{period}:{interval}:{','.join(sorted(indicators))}"
        cache = get_cache_manager()
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                logger.info(f"Retrieved historical data with indicators for {symbol} from cache")
                return cached_data
        
        try:
            # Get historical data
            historical_data = self.get_historical_data(symbol, period, interval, use_cache=use_cache)
            
            # Add technical indicators
            if not historical_data.empty:
                for indicator in indicators:
                    try:
                        if indicator == 'sma_20':
                            historical_data['SMA_20'] = historical_data['Close'].rolling(window=20).mean()
                        elif indicator == 'sma_50':
                            historical_data['SMA_50'] = historical_data['Close'].rolling(window=50).mean()
                        elif indicator == 'sma_200':
                            historical_data['SMA_200'] = historical_data['Close'].rolling(window=200).mean()
                        elif indicator == 'ema_12':
                            historical_data['EMA_12'] = historical_data['Close'].ewm(span=12, adjust=False).mean()
                        elif indicator == 'ema_26':
                            historical_data['EMA_26'] = historical_data['Close'].ewm(span=26, adjust=False).mean()
                        elif indicator == 'rsi_14':
                            delta = historical_data['Close'].diff()
                            gain = delta.where(delta > 0, 0)
                            loss = -delta.where(delta < 0, 0)
                            avg_gain = gain.rolling(window=14).mean()
                            avg_loss = loss.rolling(window=14).mean()
                            rs = avg_gain / avg_loss
                            historical_data['RSI_14'] = 100 - (100 / (1 + rs))
                        elif indicator == 'macd':
                            ema_12 = historical_data['Close'].ewm(span=12, adjust=False).mean()
                            ema_26 = historical_data['Close'].ewm(span=26, adjust=False).mean()
                            historical_data['MACD_Line'] = ema_12 - ema_26
                            historical_data['Signal_Line'] = historical_data['MACD_Line'].ewm(span=9, adjust=False).mean()
                            historical_data['MACD_Histogram'] = historical_data['MACD_Line'] - historical_data['Signal_Line']
                    except Exception as e:
                        logger.warning(f"Error calculating indicator {indicator} for {symbol}: {str(e)}")
            
            # Cache the result
            if use_cache:
                cache.set(
                    cache_key, 
                    historical_data, 
                    data_type="historical_with_indicators", 
                    tags=[symbol, "historical_data", "indicators"]
                )
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error retrieving historical data with indicators for {symbol}: {str(e)}")
            raise DataRetrievalError(
                f"Failed to retrieve historical data with indicators for {symbol}: {str(e)}",
                symbol=symbol,
                data_source='multiple',
                original_exception=e
            )