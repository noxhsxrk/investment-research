"""Tests for stock list management."""

import os
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from stock_analysis.utils.stock_list_manager import StockListManager
from stock_analysis.utils.exceptions import ConfigurationError, ValidationError


class TestStockListManager:
    """Test cases for StockListManager."""
    
    @pytest.fixture
    def mock_stock_data_service(self):
        """Create a mock stock data service."""
        mock_service = MagicMock()
        mock_service.validate_symbol.return_value = True
        
        # Mock stock info with sector and industry
        mock_stock_info = MagicMock()
        mock_stock_info.sector = "Technology"
        mock_stock_info.industry = "Software"
        mock_service.get_stock_info.return_value = mock_stock_info
        
        return mock_service
    
    @pytest.fixture
    def stock_list_manager(self, mock_stock_data_service):
        """Create a StockListManager instance with mocked dependencies."""
        with patch('stock_analysis.utils.stock_list_manager.config') as mock_config:
            # Mock configuration
            mock_config.get.return_value = []
            mock_config.config_path = "config.yaml"
            
            manager = StockListManager(stock_data_service=mock_stock_data_service)
            return manager
    
    def test_load_from_csv(self, stock_list_manager):
        """Test loading stock symbols from CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("AAPL,MSFT,GOOG\n")
            f.write("AMZN,TSLA\n")
            csv_path = f.name
        
        try:
            symbols = stock_list_manager.load_stock_list(csv_path, "tech_stocks")
            
            assert len(symbols) == 5
            assert "AAPL" in symbols
            assert "MSFT" in symbols
            assert "GOOG" in symbols
            assert "AMZN" in symbols
            assert "TSLA" in symbols
            
            # Check that the list was stored
            assert "tech_stocks" in stock_list_manager.get_all_stock_lists()
            assert stock_list_manager.get_stock_list("tech_stocks") == symbols
        finally:
            os.unlink(csv_path)
    
    def test_load_from_txt(self, stock_list_manager):
        """Test loading stock symbols from text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# Tech stocks\n")
            f.write("AAPL\n")
            f.write("MSFT\n")
            f.write("# Comment line\n")
            f.write("GOOG\n")
            txt_path = f.name
        
        try:
            symbols = stock_list_manager.load_stock_list(txt_path)
            
            assert len(symbols) == 3
            assert "AAPL" in symbols
            assert "MSFT" in symbols
            assert "GOOG" in symbols
            
            # Comments should be ignored
            assert "# Tech stocks" not in symbols
            assert "# Comment line" not in symbols
        finally:
            os.unlink(txt_path)
    
    def test_load_from_json(self, stock_list_manager):
        """Test loading stock symbols from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([
                {"symbol": "AAPL", "sector": "Technology", "industry": "Consumer Electronics"},
                {"symbol": "MSFT", "sector": "Technology", "industry": "Software"},
                {"symbol": "GOOG", "sector": "Technology", "industry": "Internet Content"}
            ], f)
            json_path = f.name
        
        try:
            symbols = stock_list_manager.load_stock_list(json_path)
            
            assert len(symbols) == 3
            assert "AAPL" in symbols
            assert "MSFT" in symbols
            assert "GOOG" in symbols
            
            # Check that sector and industry were stored
            assert stock_list_manager.get_symbol_sector("AAPL") == "Technology"
            assert stock_list_manager.get_symbol_industry("AAPL") == "Consumer Electronics"
        finally:
            os.unlink(json_path)
    
    def test_load_from_json_categories(self, stock_list_manager):
        """Test loading stock symbols from JSON file with categories."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "tech": ["AAPL", "MSFT", "GOOG"],
                "finance": ["JPM", "BAC", "GS"]
            }, f)
            json_path = f.name
        
        try:
            symbols = stock_list_manager.load_stock_list(json_path)
            
            assert len(symbols) == 6
            assert "AAPL" in symbols
            assert "JPM" in symbols
        finally:
            os.unlink(json_path)
    
    def test_validate_symbols(self, stock_list_manager, mock_stock_data_service):
        """Test symbol validation."""
        # Test basic format validation
        valid, invalid = stock_list_manager.validate_symbols(["AAPL", "MSFT", "123", "A@BC"])
        
        assert "AAPL" in valid
        assert "MSFT" in valid
        assert "123" in valid  # Numbers are valid
        assert "A@BC" in invalid  # Special chars not allowed
        
        # Test API validation
        mock_stock_data_service.validate_symbol.side_effect = lambda s: s in ["AAPL", "MSFT"]
        valid, invalid = stock_list_manager.validate_symbols(["AAPL", "MSFT", "INVALID"], validate_with_api=True)
        
        assert "AAPL" in valid
        assert "MSFT" in valid
        assert "INVALID" in invalid
    
    def test_save_stock_list(self, stock_list_manager):
        """Test saving stock list to file."""
        symbols = ["AAPL", "MSFT", "GOOG"]
        
        # Test saving as TXT
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            txt_path = f.name
        
        try:
            stock_list_manager.save_stock_list(symbols, txt_path, 'txt')
            
            with open(txt_path, 'r') as f:
                content = f.read()
                assert "AAPL" in content
                assert "MSFT" in content
                assert "GOOG" in content
        finally:
            os.unlink(txt_path)
        
        # Test saving as CSV
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            stock_list_manager.save_stock_list(symbols, csv_path, 'csv')
            
            with open(csv_path, 'r') as f:
                content = f.read()
                assert "AAPL,MSFT,GOOG" in content
        finally:
            os.unlink(csv_path)
        
        # Test saving as JSON
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            stock_list_manager.save_stock_list(symbols, json_path, 'json')
            
            with open(json_path, 'r') as f:
                loaded = json.load(f)
                assert "AAPL" in loaded
                assert "MSFT" in loaded
                assert "GOOG" in loaded
        finally:
            os.unlink(json_path)
    
    def test_fetch_symbol_classifications(self, stock_list_manager, mock_stock_data_service):
        """Test fetching symbol classifications."""
        result = stock_list_manager.fetch_symbol_classifications(["AAPL", "MSFT"])
        
        assert "AAPL" in result
        assert "MSFT" in result
        assert result["AAPL"]["sector"] == "Technology"
        assert result["AAPL"]["industry"] == "Software"
        
        # Check that classifications were stored
        assert stock_list_manager.get_symbol_sector("AAPL") == "Technology"
        assert stock_list_manager.get_symbol_industry("AAPL") == "Software"
    
    def test_update_symbol_classification(self, stock_list_manager):
        """Test updating symbol classifications."""
        stock_list_manager.update_symbol_classification("AAPL", "Tech", "Mobile Devices")
        
        assert stock_list_manager.get_symbol_sector("AAPL") == "Tech"
        assert stock_list_manager.get_symbol_industry("AAPL") == "Mobile Devices"
        
        # Update only sector
        stock_list_manager.update_symbol_classification("AAPL", "Technology")
        
        assert stock_list_manager.get_symbol_sector("AAPL") == "Technology"
        assert stock_list_manager.get_symbol_industry("AAPL") == "Mobile Devices"
    
    def test_invalid_file_path(self, stock_list_manager):
        """Test handling of invalid file paths."""
        with pytest.raises(ConfigurationError):
            stock_list_manager.load_stock_list("nonexistent_file.csv")
    
    def test_unsupported_file_format(self, stock_list_manager):
        """Test handling of unsupported file formats."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            invalid_path = f.name
        
        try:
            with pytest.raises(ConfigurationError):
                stock_list_manager.load_stock_list(invalid_path)
        finally:
            os.unlink(invalid_path)