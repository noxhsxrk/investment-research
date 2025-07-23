"""Market data service for retrieving market-wide data.

This module provides a service for retrieving market data including indices,
commodities, forex, and economic indicators from multiple sources.
"""

import logging
from typing import Dict, List, Optional, Union, Any

import pandas as pd

from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
from stock_analysis.models.enhanced_data_models import MarketData
from stock_analysis.utils.exceptions import DataRetrievalError
from stock_analysis.utils.logging import get_logger, log_api_call, log_data_quality_issue
from stock_analysis.utils.performance_metrics import monitor_performance
from stock_analysis.utils.cache_manager import get_cache_manager

logger = get_logger(__name__)


class MarketDataService:
    """Service for retrieving market-wide data."""
    
    def __init__(self, integration_service: Optional[FinancialDataIntegrationService] = None):
        """Initialize the market data service.
        
        Args:
            integration_service: Financial data integration service instance
        """
        # Initialize integration service if not provided
        if integration_service is None:
            self.integration_service = FinancialDataIntegrationService()
        else:
            self.integration_service = integration_service
        
        self.cache = get_cache_manager()
    
    @monitor_performance('get_market_overview')
    def get_market_overview(self, use_cache: bool = True) -> MarketData:
        """Get comprehensive market overview.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            MarketData object with market overview
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.set_context(operation='get_market_overview')
        logger.info("Retrieving market overview")
        
        # Generate cache key
        cache_key = "market_overview"
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info("Retrieved market overview from cache")
                return cached_data
        
        try:
            # Get indices data
            indices = self.integration_service.get_market_data('indices', use_cache=use_cache)
            
            # Get commodities data
            commodities = self.integration_service.get_market_data('commodities', use_cache=use_cache)
            
            # Get forex data
            forex = self.integration_service.get_market_data('forex', use_cache=use_cache)
            
            # Get sector performance data
            sector_performance = self.integration_service.get_market_data('sectors', use_cache=use_cache)
            
            # Get economic indicators
            economic_indicators = self.get_economic_indicators(use_cache=use_cache)
            
            # Create MarketData object
            market_data = MarketData(
                indices=self._normalize_market_data(indices),
                commodities=self._normalize_market_data(commodities),
                forex=self._normalize_market_data(forex),
                sector_performance=self._extract_sector_performance(sector_performance),
                economic_indicators=economic_indicators
            )
            
            # Validate the market data
            market_data.validate()
            
            # Cache the result
            if use_cache:
                self.cache.set(
                    cache_key, 
                    market_data, 
                    data_type="market_data", 
                    tags=["market_overview"]
                )
            
            logger.info("Successfully retrieved market overview")
            return market_data
            
        except Exception as e:
            logger.error(f"Error retrieving market overview: {str(e)}")
            raise DataRetrievalError(
                f"Failed to retrieve market overview: {str(e)}",
                data_source='multiple',
                original_exception=e
            )
    
    def _normalize_market_data(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Normalize market data to standard format.
        
        Args:
            data: Raw market data dictionary
            
        Returns:
            Normalized market data dictionary
        """
        normalized = {}
        
        for key, value in data.items():
            # Skip metadata fields
            if key.startswith('_'):
                continue
                
            # Normalize field names
            normalized_item = {}
            
            # Map common field names
            field_mapping = {
                'price': 'value',
                'last': 'value',
                'value': 'value',
                'change': 'change',
                'change_percent': 'change_percent',
                'pct_change': 'change_percent',
                'symbol': 'symbol',
                'name': 'name',
                'description': 'description'
            }
            
            for field, normalized_field in field_mapping.items():
                if field in value:
                    normalized_item[normalized_field] = value[field]
            
            # Ensure required fields exist
            if 'value' not in normalized_item and 'price' in value:
                normalized_item['value'] = value['price']
            
            if 'change' not in normalized_item:
                normalized_item['change'] = 0.0
            
            if 'change_percent' not in normalized_item:
                normalized_item['change_percent'] = 0.0
            
            normalized[key] = normalized_item
        
        return normalized
    
    def _extract_sector_performance(self, sector_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract sector performance from sector data.
        
        Args:
            sector_data: Sector data dictionary
            
        Returns:
            Dictionary mapping sector names to performance values
        """
        performance = {}
        
        for sector, data in sector_data.items():
            # Skip metadata fields
            if sector.startswith('_'):
                continue
                
            # Use change_percent as performance metric
            if 'change_percent' in data:
                performance[sector] = data['change_percent']
            elif 'pct_change' in data:
                performance[sector] = data['pct_change']
            elif 'change' in data and 'price' in data and data['price'] != 0:
                # Calculate percent change if not provided
                performance[sector] = (data['change'] / data['price']) * 100
            else:
                # Default to 0 if no performance data available
                performance[sector] = 0.0
        
        return performance
    
    @monitor_performance('get_economic_indicators')
    def get_economic_indicators(self, region: Optional[str] = None, use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
        """Get economic indicators.
        
        Args:
            region: Region for economic indicators (e.g., 'US', 'EU')
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing economic indicators
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.set_context(region=region, operation='get_economic_indicators')
        logger.info(f"Retrieving economic indicators for region: {region or 'all'}")
        
        # Generate cache key
        cache_key = f"economic_indicators:{region or 'all'}"
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved economic indicators for region {region or 'all'} from cache")
                return cached_data
        
        try:
            # Get economic indicators from integration service
            indicators = self.integration_service.get_economic_data(region=region, use_cache=use_cache)
            
            # Normalize indicators
            normalized_indicators = self._normalize_economic_indicators(indicators)
            
            # Cache the result
            if use_cache:
                self.cache.set(
                    cache_key, 
                    normalized_indicators, 
                    data_type="economic_indicators", 
                    tags=["economic_indicators", region or "all"]
                )
            
            logger.info(f"Successfully retrieved economic indicators for region {region or 'all'}")
            return normalized_indicators
            
        except Exception as e:
            logger.error(f"Error retrieving economic indicators for region {region or 'all'}: {str(e)}")
            raise DataRetrievalError(
                f"Failed to retrieve economic indicators for region {region or 'all'}: {str(e)}",
                data_source='multiple',
                original_exception=e
            )
    
    def _normalize_economic_indicators(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Normalize economic indicators to standard format.
        
        Args:
            data: Raw economic indicators dictionary
            
        Returns:
            Normalized economic indicators dictionary
        """
        normalized = {}
        
        for key, value in data.items():
            # Skip metadata fields
            if key.startswith('_'):
                continue
                
            # Normalize field names
            normalized_item = {}
            
            # Map common field names
            field_mapping = {
                'value': 'value',
                'current': 'value',
                'previous': 'previous',
                'forecast': 'forecast',
                'unit': 'unit',
                'period': 'period',
                'region': 'region',
                'date': 'date'
            }
            
            for field, normalized_field in field_mapping.items():
                if field in value:
                    normalized_item[normalized_field] = value[field]
            
            # Ensure required fields exist
            if 'value' not in normalized_item and 'current' in value:
                normalized_item['value'] = value['current']
            
            if 'unit' not in normalized_item:
                normalized_item['unit'] = '%'  # Default unit
            
            if 'previous' not in normalized_item:
                normalized_item['previous'] = None
            
            if 'forecast' not in normalized_item:
                normalized_item['forecast'] = None
            
            normalized[key] = normalized_item
        
        return normalized
    
    @monitor_performance('get_sector_performance')
    def get_sector_performance(self, use_cache: bool = True) -> Dict[str, float]:
        """Get performance metrics for market sectors.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary mapping sector names to performance values
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.set_context(operation='get_sector_performance')
        logger.info("Retrieving sector performance")
        
        # Generate cache key
        cache_key = "sector_performance"
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info("Retrieved sector performance from cache")
                return cached_data
        
        try:
            # Get sector data from integration service
            sector_data = self.integration_service.get_market_data('sectors', use_cache=use_cache)
            
            # Extract performance metrics
            performance = self._extract_sector_performance(sector_data)
            
            # Cache the result
            if use_cache:
                self.cache.set(
                    cache_key, 
                    performance, 
                    data_type="sector_performance", 
                    tags=["sector_performance"]
                )
            
            logger.info("Successfully retrieved sector performance")
            return performance
            
        except Exception as e:
            logger.error(f"Error retrieving sector performance: {str(e)}")
            raise DataRetrievalError(
                f"Failed to retrieve sector performance: {str(e)}",
                data_source='multiple',
                original_exception=e
            )
    
    @monitor_performance('get_commodity_prices')
    def get_commodity_prices(self, commodities: Optional[List[str]] = None, use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
        """Get commodity prices.
        
        Args:
            commodities: List of commodities to retrieve (if None, retrieves all available)
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing commodity prices
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.set_context(operation='get_commodity_prices')
        logger.info(f"Retrieving commodity prices for: {commodities or 'all'}")
        
        # Generate cache key
        cache_key = f"commodity_prices:{','.join(sorted(commodities)) if commodities else 'all'}"
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved commodity prices for {commodities or 'all'} from cache")
                return cached_data
        
        try:
            # Get commodity data from integration service
            commodity_data = self.integration_service.get_market_data('commodities', use_cache=use_cache)
            
            # Filter commodities if specified
            if commodities:
                filtered_data = {}
                for commodity in commodities:
                    if commodity in commodity_data:
                        filtered_data[commodity] = commodity_data[commodity]
                    else:
                        logger.warning(f"Commodity {commodity} not found in available data")
                
                commodity_data = filtered_data
            
            # Normalize commodity data
            normalized_data = self._normalize_market_data(commodity_data)
            
            # Cache the result
            if use_cache:
                self.cache.set(
                    cache_key, 
                    normalized_data, 
                    data_type="commodity_prices", 
                    tags=["commodity_prices"]
                )
            
            logger.info(f"Successfully retrieved commodity prices for {commodities or 'all'}")
            return normalized_data
            
        except Exception as e:
            logger.error(f"Error retrieving commodity prices for {commodities or 'all'}: {str(e)}")
            raise DataRetrievalError(
                f"Failed to retrieve commodity prices for {commodities or 'all'}: {str(e)}",
                data_source='multiple',
                original_exception=e
            )
    
    @monitor_performance('get_market_indices')
    def get_market_indices(self, indices: Optional[List[str]] = None, use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
        """Get market indices.
        
        Args:
            indices: List of indices to retrieve (if None, retrieves all available)
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing market indices
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.set_context(operation='get_market_indices')
        logger.info(f"Retrieving market indices for: {indices or 'all'}")
        
        # Generate cache key
        cache_key = f"market_indices:{','.join(sorted(indices)) if indices else 'all'}"
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved market indices for {indices or 'all'} from cache")
                return cached_data
        
        try:
            # Get indices data from integration service
            indices_data = self.integration_service.get_market_data('indices', use_cache=use_cache)
            
            # Filter indices if specified
            if indices:
                filtered_data = {}
                for index in indices:
                    if index in indices_data:
                        filtered_data[index] = indices_data[index]
                    else:
                        logger.warning(f"Index {index} not found in available data")
                
                indices_data = filtered_data
            
            # Normalize indices data
            normalized_data = self._normalize_market_data(indices_data)
            
            # Cache the result
            if use_cache:
                self.cache.set(
                    cache_key, 
                    normalized_data, 
                    data_type="market_indices", 
                    tags=["market_indices"]
                )
            
            logger.info(f"Successfully retrieved market indices for {indices or 'all'}")
            return normalized_data
            
        except Exception as e:
            logger.error(f"Error retrieving market indices for {indices or 'all'}: {str(e)}")
            raise DataRetrievalError(
                f"Failed to retrieve market indices for {indices or 'all'}: {str(e)}",
                data_source='multiple',
                original_exception=e
            )