"""Unit tests for stock list manager."""

import pytest
import os
import json
import csv
from unittest.mock import patch, MagicMock
from pathlib import Path

from stock_analysis.utils.stock_list_manager import StockListManager
from stock_analysis.utils.exceptions import ConfigurationError


@pytest.fixture
def stock_list_manager():
    """Create a stock list manager instance."""
    return StockListManager()


@pytest.fixture
def sample_stock_list():
    """Create a sample stock list."""
    return [
        {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
        {"symbol": "MSFT", "name": "Microsoft Corp", "sector": "Technology"},
        {"symbol": "GOOGL", "name": "Alphabet Inc", "sector": "Technology"}
    ]


class TestStockListManager:
    """Test cases for StockListManager."""
    
    def test_load_stock_list_csv(self, stock_list_manager, tmp_path, sample_stock_list):
        """Test loading stock list from CSV file."""
        # Create test CSV file
        csv_file = tmp_path / "test_stocks.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["symbol", "name", "sector"])
            writer.writeheader()
            writer.writerows(sample_stock_list)
        
        # Load stock list
        stocks = stock_list_manager.load_stock_list(str(csv_file))
        
        assert len(stocks) == 3
        assert stocks[0]["symbol"] == "AAPL"
        assert stocks[1]["name"] == "Microsoft Corp"
        assert stocks[2]["sector"] == "Technology"
    
    def test_load_stock_list_json(self, stock_list_manager, tmp_path, sample_stock_list):
        """Test loading stock list from JSON file."""
        # Create test JSON file
        json_file = tmp_path / "test_stocks.json"
        with open(json_file, 'w') as f:
            json.dump(sample_stock_list, f)
        
        # Load stock list
        stocks = stock_list_manager.load_stock_list(str(json_file))
        
        assert len(stocks) == 3
        assert stocks[0]["symbol"] == "AAPL"
        assert stocks[1]["name"] == "Microsoft Corp"
        assert stocks[2]["sector"] == "Technology"
    
    def test_save_stock_list_csv(self, stock_list_manager, tmp_path, sample_stock_list):
        """Test saving stock list to CSV file."""
        # Save stock list
        csv_file = tmp_path / "test_stocks.csv"
        stock_list_manager.save_stock_list(sample_stock_list, str(csv_file))
        
        # Verify file was created
        assert csv_file.exists()
        
        # Read back and verify contents
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            saved_stocks = list(reader)
            assert len(saved_stocks) == 3
            assert saved_stocks[0]["symbol"] == "AAPL"
            assert saved_stocks[1]["name"] == "Microsoft Corp"
            assert saved_stocks[2]["sector"] == "Technology"
    
    def test_save_stock_list_json(self, stock_list_manager, tmp_path, sample_stock_list):
        """Test saving stock list to JSON file."""
        # Save stock list
        json_file = tmp_path / "test_stocks.json"
        stock_list_manager.save_stock_list(sample_stock_list, str(json_file))
        
        # Verify file was created
        assert json_file.exists()
        
        # Read back and verify contents
        with open(json_file, 'r') as f:
            saved_stocks = json.load(f)
            assert len(saved_stocks) == 3
            assert saved_stocks[0]["symbol"] == "AAPL"
            assert saved_stocks[1]["name"] == "Microsoft Corp"
            assert saved_stocks[2]["sector"] == "Technology"
    
    def test_validate_stock_list(self, stock_list_manager, sample_stock_list):
        """Test stock list validation."""
        # Valid stock list should pass validation
        stock_list_manager.validate_stock_list(sample_stock_list)
        
        # Invalid stock list (missing required fields)
        invalid_stocks = [
            {"name": "Apple Inc.", "sector": "Technology"}  # Missing symbol
        ]
        with pytest.raises(ConfigurationError) as exc_info:
            stock_list_manager.validate_stock_list(invalid_stocks)
        assert "Stock list validation failed" in str(exc_info.value)
    
    def test_filter_stocks_by_sector(self, stock_list_manager, sample_stock_list):
        """Test filtering stocks by sector."""
        tech_stocks = stock_list_manager.filter_stocks_by_sector(sample_stock_list, "Technology")
        assert len(tech_stocks) == 3
        
        finance_stocks = stock_list_manager.filter_stocks_by_sector(sample_stock_list, "Finance")
        assert len(finance_stocks) == 0
    
    def test_filter_stocks_by_market_cap(self, stock_list_manager):
        """Test filtering stocks by market cap."""
        stocks = [
            {"symbol": "AAPL", "market_cap": 2500000000000},  # Large cap
            {"symbol": "MID", "market_cap": 8000000000},      # Mid cap
            {"symbol": "SMALL", "market_cap": 500000000}      # Small cap
        ]
        
        large_cap = stock_list_manager.filter_stocks_by_market_cap(stocks, "large")
        assert len(large_cap) == 1
        assert large_cap[0]["symbol"] == "AAPL"
        
        mid_cap = stock_list_manager.filter_stocks_by_market_cap(stocks, "mid")
        assert len(mid_cap) == 1
        assert mid_cap[0]["symbol"] == "MID"
        
        small_cap = stock_list_manager.filter_stocks_by_market_cap(stocks, "small")
        assert len(small_cap) == 1
        assert small_cap[0]["symbol"] == "SMALL"
    
    def test_filter_stocks_by_exchange(self, stock_list_manager):
        """Test filtering stocks by exchange."""
        stocks = [
            {"symbol": "AAPL", "exchange": "NASDAQ"},
            {"symbol": "IBM", "exchange": "NYSE"},
            {"symbol": "GOOGL", "exchange": "NASDAQ"}
        ]
        
        nasdaq_stocks = stock_list_manager.filter_stocks_by_exchange(stocks, "NASDAQ")
        assert len(nasdaq_stocks) == 2
        assert nasdaq_stocks[0]["symbol"] == "AAPL"
        assert nasdaq_stocks[1]["symbol"] == "GOOGL"
        
        nyse_stocks = stock_list_manager.filter_stocks_by_exchange(stocks, "NYSE")
        assert len(nyse_stocks) == 1
        assert nyse_stocks[0]["symbol"] == "IBM"
    
    def test_merge_stock_lists(self, stock_list_manager):
        """Test merging multiple stock lists."""
        list1 = [
            {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
            {"symbol": "MSFT", "name": "Microsoft Corp", "sector": "Technology"}
        ]
        
        list2 = [
            {"symbol": "GOOGL", "name": "Alphabet Inc", "sector": "Technology"},
            {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"}  # Duplicate
        ]
        
        merged = stock_list_manager.merge_stock_lists([list1, list2])
        assert len(merged) == 3  # Duplicates removed
        assert any(stock["symbol"] == "AAPL" for stock in merged)
        assert any(stock["symbol"] == "MSFT" for stock in merged)
        assert any(stock["symbol"] == "GOOGL" for stock in merged)
    
    def test_remove_duplicates(self, stock_list_manager):
        """Test removing duplicate stocks."""
        stocks = [
            {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
            {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},  # Duplicate
            {"symbol": "MSFT", "name": "Microsoft Corp", "sector": "Technology"}
        ]
        
        unique_stocks = stock_list_manager.remove_duplicates(stocks)
        assert len(unique_stocks) == 2
        assert unique_stocks[0]["symbol"] == "AAPL"
        assert unique_stocks[1]["symbol"] == "MSFT"
    
    def test_invalid_file_path(self, stock_list_manager):
        """Test handling of invalid file paths."""
        with pytest.raises(ConfigurationError) as exc_info:
            stock_list_manager.load_stock_list("nonexistent.csv")
        assert "Stock list file not found" in str(exc_info.value)
    
    def test_unsupported_file_format(self, stock_list_manager, tmp_path):
        """Test handling of unsupported file formats."""
        with pytest.raises(ConfigurationError) as exc_info:
            stock_list_manager.load_stock_list(str(tmp_path / "test.xyz"))
        assert "Unsupported stock list file format" in str(exc_info.value)
    
    def test_invalid_market_cap_filter(self, stock_list_manager, sample_stock_list):
        """Test handling of invalid market cap filter."""
        with pytest.raises(ConfigurationError) as exc_info:
            stock_list_manager.filter_stocks_by_market_cap(sample_stock_list, "invalid")
        assert "Invalid market cap category" in str(exc_info.value)
    
    def test_empty_stock_list(self, stock_list_manager):
        """Test handling of empty stock list."""
        with pytest.raises(ConfigurationError) as exc_info:
            stock_list_manager.validate_stock_list([])
        assert "Stock list cannot be empty" in str(exc_info.value)
    
    def test_update_stock_info(self, stock_list_manager):
        """Test updating stock information."""
        stocks = [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corp"}
        ]
        
        updates = {
            "AAPL": {"sector": "Technology", "industry": "Consumer Electronics"},
            "MSFT": {"sector": "Technology", "industry": "Software"}
        }
        
        updated_stocks = stock_list_manager.update_stock_info(stocks, updates)
        assert updated_stocks[0]["sector"] == "Technology"
        assert updated_stocks[0]["industry"] == "Consumer Electronics"
        assert updated_stocks[1]["sector"] == "Technology"
        assert updated_stocks[1]["industry"] == "Software"
    
    def test_get_symbols_list(self, stock_list_manager, sample_stock_list):
        """Test extracting symbols list."""
        symbols = stock_list_manager.get_symbols_list(sample_stock_list)
        assert symbols == ["AAPL", "MSFT", "GOOGL"]
    
    def test_get_sectors_list(self, stock_list_manager, sample_stock_list):
        """Test extracting unique sectors list."""
        sectors = stock_list_manager.get_sectors_list(sample_stock_list)
        assert sectors == ["Technology"]