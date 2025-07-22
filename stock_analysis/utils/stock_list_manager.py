"""Stock list management for stock analysis system.

This module provides functionality for loading, validating, and managing
lists of stock symbols from various file formats.
"""

import os
import csv
import json
import logging
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import yfinance as yf

from stock_analysis.utils.config import config
from stock_analysis.utils.exceptions import ConfigurationError, ValidationError
from stock_analysis.utils.logging import get_logger
from stock_analysis.services.stock_data_service import StockDataService

logger = get_logger(__name__)


class StockListManager:
    """Manages stock lists from various sources."""
    
    def __init__(self, stock_data_service: Optional[StockDataService] = None):
        """Initialize the stock list manager.
        
        Args:
            stock_data_service: Optional StockDataService instance for symbol validation
        """
        self.stock_data_service = stock_data_service or StockDataService()
        self._stock_lists: Dict[str, List[str]] = {}
        self._industry_map: Dict[str, str] = {}
        self._sector_map: Dict[str, str] = {}
        self._cache_file = os.path.join(os.path.dirname(config.config_path), 'stock_classifications.json')
        
        # Register for configuration changes
        config.register_callback(self._on_config_change)
        
        # Load cached industry and sector data if available
        self._load_classification_cache()
        
        # Load default stock lists if configured
        self._load_default_lists()
        
        # Start watching for configuration changes
        config.start_watching()
    
    def _on_config_change(self) -> None:
        """Handle configuration changes."""
        logger.info("Configuration changed, reloading stock lists")
        self._load_default_lists()
    
    def _load_default_lists(self) -> None:
        """Load default stock lists from configuration."""
        default_lists = config.get('stock_analysis.stock_lists.default_files', [])
        for list_file in default_lists:
            try:
                name = os.path.basename(list_file).split('.')[0]
                self.load_stock_list(list_file, name)
            except Exception as e:
                logger.warning(f"Failed to load default stock list {list_file}: {str(e)}")
        
        # Auto-validate symbols if configured
        if config.get('stock_analysis.stock_lists.auto_validate', False):
            for list_name, symbols in self._stock_lists.items():
                try:
                    valid, invalid = self.validate_symbols(symbols, validate_with_api=True)
                    if invalid:
                        logger.warning(f"Found {len(invalid)} invalid symbols in list '{list_name}': {', '.join(invalid)}")
                    self._stock_lists[list_name] = valid
                except Exception as e:
                    logger.warning(f"Failed to validate stock list '{list_name}': {str(e)}")
    
    def _load_classification_cache(self) -> None:
        """Load cached industry and sector classifications."""
        if not config.get('stock_analysis.stock_lists.cache_industry_data', True):
            return
            
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                    if 'industry_map' in cache_data:
                        self._industry_map.update(cache_data['industry_map'])
                    
                    if 'sector_map' in cache_data:
                        self._sector_map.update(cache_data['sector_map'])
                        
                    logger.info(f"Loaded classifications for {len(self._industry_map)} symbols from cache")
        except Exception as e:
            logger.warning(f"Failed to load classification cache: {str(e)}")
    
    def _save_classification_cache(self) -> None:
        """Save industry and sector classifications to cache."""
        if not config.get('stock_analysis.stock_lists.cache_industry_data', True):
            return
            
        try:
            cache_dir = os.path.dirname(self._cache_file)
            os.makedirs(cache_dir, exist_ok=True)
            
            with open(self._cache_file, 'w') as f:
                json.dump({
                    'industry_map': self._industry_map,
                    'sector_map': self._sector_map
                }, f, indent=2)
                
            logger.info(f"Saved classifications for {len(self._industry_map)} symbols to cache")
        except Exception as e:
            logger.warning(f"Failed to save classification cache: {str(e)}")
    
    def load_stock_list(self, file_path: str, list_name: Optional[str] = None) -> List[str]:
        """Load a stock list from a file.
        
        Supports CSV, TXT, and JSON formats.
        
        Args:
            file_path: Path to the stock list file
            list_name: Name to assign to the stock list (defaults to filename without extension)
            
        Returns:
            List of stock symbols
            
        Raises:
            ConfigurationError: If the file cannot be loaded or has an invalid format
        """
        if not os.path.exists(file_path):
            raise ConfigurationError(f"Stock list file not found: {file_path}")
        
        # Use filename as list name if not provided
        if list_name is None:
            list_name = os.path.basename(file_path).split('.')[0]
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            symbols = []
            
            if file_ext == '.csv':
                symbols = self._load_from_csv(file_path)
            elif file_ext == '.txt':
                symbols = self._load_from_txt(file_path)
            elif file_ext == '.json':
                symbols = self._load_from_json(file_path)
            else:
                raise ConfigurationError(f"Unsupported stock list file format: {file_ext}")
            
            # Remove duplicates and empty strings
            symbols = [s.strip().upper() for s in symbols if s.strip()]
            symbols = list(dict.fromkeys(symbols))  # Remove duplicates while preserving order
            
            if not symbols:
                logger.warning(f"No valid stock symbols found in {file_path}")
            else:
                logger.info(f"Loaded {len(symbols)} stock symbols from {file_path}")
                
            # Store the stock list
            self._stock_lists[list_name] = symbols
            return symbols
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load stock list from {file_path}: {str(e)}")
    
    def _load_from_csv(self, file_path: str) -> List[str]:
        """Load stock symbols from a CSV file.
        
        Supports both comma-separated symbols on a single line and
        symbols in separate rows.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of stock symbols
        """
        symbols = []
        
        with open(file_path, 'r') as file:
            # Try to parse as a CSV file with headers
            reader = csv.reader(file)
            for row in reader:
                symbols.extend([s.strip() for s in row if s.strip()])
        
        return symbols
    
    def _load_from_txt(self, file_path: str) -> List[str]:
        """Load stock symbols from a text file.
        
        Ignores lines starting with # (comments).
        Each symbol should be on a separate line.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of stock symbols
        """
        symbols = []
        
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    symbols.append(line)
        
        return symbols
    
    def _load_from_json(self, file_path: str) -> List[str]:
        """Load stock symbols from a JSON file.
        
        Supports various JSON formats:
        - List of symbols: ["AAPL", "MSFT", ...]
        - Object with lists: {"tech": ["AAPL", "MSFT"], "finance": ["JPM", "BAC"]}
        - List of objects: [{"symbol": "AAPL", "sector": "Technology"}, ...]
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of stock symbols
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        symbols = []
        
        if isinstance(data, list):
            # Check if it's a simple list of symbols or a list of objects
            if data and isinstance(data[0], str):
                # Simple list of symbols
                symbols = data
            elif data and isinstance(data[0], dict):
                # List of objects with symbol property
                for item in data:
                    if 'symbol' in item:
                        symbols.append(item['symbol'])
                        
                        # Store sector and industry information if available
                        if 'sector' in item and item['symbol'] and item['sector']:
                            self._sector_map[item['symbol']] = item['sector']
                        
                        if 'industry' in item and item['symbol'] and item['industry']:
                            self._industry_map[item['symbol']] = item['industry']
        
        elif isinstance(data, dict):
            # Dictionary with category keys and symbol lists
            for category, category_symbols in data.items():
                if isinstance(category_symbols, list):
                    symbols.extend(category_symbols)
        
        return symbols
    
    def get_stock_list(self, list_name: str) -> List[str]:
        """Get a stock list by name.
        
        Args:
            list_name: Name of the stock list
            
        Returns:
            List of stock symbols
            
        Raises:
            KeyError: If the stock list does not exist
        """
        if list_name not in self._stock_lists:
            raise KeyError(f"Stock list not found: {list_name}")
        
        return self._stock_lists[list_name]
    
    def get_all_stock_lists(self) -> Dict[str, List[str]]:
        """Get all loaded stock lists.
        
        Returns:
            Dictionary mapping list names to symbol lists
        """
        return self._stock_lists.copy()
    
    def save_stock_list(self, symbols: List[str], file_path: str, format_type: str = 'txt') -> None:
        """Save a stock list to a file.
        
        Args:
            symbols: List of stock symbols
            file_path: Path to save the file
            format_type: File format ('txt', 'csv', or 'json')
            
        Raises:
            ConfigurationError: If the file cannot be saved or has an invalid format
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            if format_type == 'txt':
                with open(file_path, 'w') as file:
                    file.write("# Stock symbols\n")
                    for symbol in symbols:
                        file.write(f"{symbol}\n")
            
            elif format_type == 'csv':
                with open(file_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(symbols)
            
            elif format_type == 'json':
                with open(file_path, 'w') as file:
                    json.dump(symbols, file, indent=2)
            
            else:
                raise ConfigurationError(f"Unsupported stock list format: {format_type}")
            
            logger.info(f"Saved {len(symbols)} stock symbols to {file_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save stock list to {file_path}: {str(e)}")
    
    def validate_symbols(self, symbols: List[str], validate_with_api: bool = False) -> Tuple[List[str], List[str]]:
        """Validate a list of stock symbols.
        
        Args:
            symbols: List of stock symbols to validate
            validate_with_api: Whether to validate symbols against the API
            
        Returns:
            Tuple of (valid_symbols, invalid_symbols)
        """
        if not symbols:
            return [], []
        
        valid_symbols = []
        invalid_symbols = []
        
        # Basic format validation
        for symbol in symbols:
            if not symbol or not isinstance(symbol, str):
                invalid_symbols.append(symbol)
                continue
                
            # Basic format check (alphanumeric with some special chars)
            if not self._is_valid_symbol_format(symbol):
                logger.warning(f"Invalid symbol format: {symbol}")
                invalid_symbols.append(symbol)
                continue
                
            valid_symbols.append(symbol)
        
        # API validation if requested
        if validate_with_api and valid_symbols:
            api_valid = []
            api_invalid = []
            
            for symbol in valid_symbols:
                try:
                    if self.stock_data_service.validate_symbol(symbol):
                        api_valid.append(symbol)
                    else:
                        logger.warning(f"Symbol not found in API: {symbol}")
                        api_invalid.append(symbol)
                except Exception as e:
                    logger.warning(f"Error validating symbol {symbol}: {str(e)}")
                    api_invalid.append(symbol)
            
            return api_valid, api_invalid + invalid_symbols
        
        return valid_symbols, invalid_symbols
    
    def _is_valid_symbol_format(self, symbol: str) -> bool:
        """Check if a symbol has a valid format.
        
        Args:
            symbol: Stock symbol to check
            
        Returns:
            True if the symbol has a valid format, False otherwise
        """
        import re
        # Allow alphanumeric characters, dots, hyphens, and carets (for indices)
        return bool(re.match(r'^[A-Z0-9.^-]{1,10}$', symbol.upper()))
    
    def get_symbol_industry(self, symbol: str) -> Optional[str]:
        """Get the industry for a stock symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Industry name or None if not available
        """
        return self._industry_map.get(symbol.upper())
    
    def get_symbol_sector(self, symbol: str) -> Optional[str]:
        """Get the sector for a stock symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Sector name or None if not available
        """
        return self._sector_map.get(symbol.upper())
    
    def update_symbol_classification(self, symbol: str, sector: Optional[str] = None, industry: Optional[str] = None) -> None:
        """Update sector and industry classification for a symbol.
        
        Args:
            symbol: Stock symbol
            sector: Sector name
            industry: Industry name
        """
        symbol = symbol.upper()
        
        if sector:
            self._sector_map[symbol] = sector
        
        if industry:
            self._industry_map[symbol] = industry
    
    def fetch_symbol_classifications(self, symbols: List[str]) -> Dict[str, Dict[str, str]]:
        """Fetch sector and industry classifications for symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to their classifications
        """
        result = {}
        updated = False
        
        for symbol in symbols:
            symbol = symbol.upper()
            try:
                stock_info = self.stock_data_service.get_stock_info(symbol)
                
                if stock_info.sector or stock_info.industry:
                    result[symbol] = {
                        'sector': stock_info.sector,
                        'industry': stock_info.industry
                    }
                    
                    # Update internal maps
                    if stock_info.sector:
                        self._sector_map[symbol] = stock_info.sector
                        updated = True
                    
                    if stock_info.industry:
                        self._industry_map[symbol] = stock_info.industry
                        updated = True
                
            except Exception as e:
                logger.warning(f"Failed to fetch classification for {symbol}: {str(e)}")
        
        # Save cache if updated
        if updated:
            self._save_classification_cache()
            
        return result


# Global stock list manager instance
stock_list_manager = StockListManager()