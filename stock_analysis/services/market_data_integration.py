"""Market data integration functionality.

This module provides functions for integrating market data from multiple sources.
"""

from typing import Dict, List, Any, Optional
import logging

from stock_analysis.utils.logging import get_logger

logger = get_logger(__name__)


def validate_market_data(data: Dict[str, Any], data_type: str) -> bool:
    """Validate market data.
    
    Args:
        data: Market data dictionary
        data_type: Type of market data ('indices', 'commodities', 'forex', 'sectors')
        
    Returns:
        True if data is valid and complete, False otherwise
    """
    if not data:
        return False
    
    # Minimum number of items required for each data type
    min_items = {
        'indices': 2,
        'commodities': 2,
        'forex': 2,
        'sectors': 3
    }
    
    # Check if we have enough items
    if len(data) < min_items.get(data_type, 2):
        return False
    
    # Check required fields for each item
    required_fields = ['price']
    optional_fields = ['symbol', 'name', 'change', 'change_percent']
    
    for item_name, item_data in data.items():
        # Skip metadata fields
        if item_name.startswith('_'):
            continue
            
        # Check required fields
        for field in required_fields:
            if field not in item_data or item_data[field] is None:
                return False
        
        # Check if we have at least some optional fields
        optional_count = sum(1 for field in optional_fields if field in item_data and item_data[field] is not None)
        if optional_count < 1:  # At least 1 optional field should be present
            return False
    
    return True


def combine_market_data(partial_results: Dict[str, Dict[str, Any]], data_type: str) -> Dict[str, Any]:
    """Combine market data from multiple sources.
    
    Args:
        partial_results: Dictionary mapping source names to partial results
        data_type: Type of market data ('indices', 'commodities', 'forex', 'sectors')
        
    Returns:
        Combined market data dictionary
    """
    if not partial_results:
        return {}
    
    combined = {}
    sources = list(partial_results.keys())
    
    # Process each source in priority order
    for source in sources:
        source_data = partial_results[source]
        
        for item_name, item_data in source_data.items():
            # If item doesn't exist in combined result yet, add it
            if item_name not in combined:
                combined[item_name] = item_data.copy()
            # Otherwise, only fill in missing fields
            else:
                for field, value in item_data.items():
                    if field not in combined[item_name] or combined[item_name][field] is None:
                        combined[item_name][field] = value
    
    # Add metadata about sources
    combined['_sources'] = sources
    
    return combined