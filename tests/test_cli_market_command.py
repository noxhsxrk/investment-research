"""Unit tests for market command in CLI.

This module tests the market command for retrieving market data.
"""

import unittest
from unittest.mock import patch, MagicMock
import argparse
import sys
from io import StringIO

from stock_analysis.cli import handle_market_command
from stock_analysis.models.enhanced_data_models import MarketData


class TestMarketCommand(unittest.TestCase):
    """Test cases for market command."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock arguments
        self.args = argparse.Namespace()
        self.args.indices = False
        self.args.sectors = False
        self.args.commodities = False
        self.args.forex = False
        self.args.economic = False
        self.args.region = None
        self.args.export_format = None
        self.args.output = None
        self.args.no_export = True
        self.args.verbose = False
        
        # Create mock market data
        self.indices = {
            'S&P 500': {'value': 4500.0, 'change': 45.0, 'change_percent': 1.0},
            'NASDAQ': {'value': 14000.0, 'change': 120.0, 'change_percent': 0.85},
            'Dow Jones': {'value': 35000.0, 'change': -150.0, 'change_percent': -0.43}
        }
        
        self.sectors = {
            'Technology': 1.2,
            'Healthcare': 0.5,
            'Energy': -0.8,
            'Financials': 0.3,
            'Consumer Discretionary': -0.2
        }
        
        self.commodities = {
            'Gold': {'value': 1850.0, 'change': 15.0, 'change_percent': 0.8},
            'Oil (WTI)': {'value': 75.0, 'change': -2.0, 'change_percent': -2.6},
            'Silver': {'value': 24.0, 'change': 0.5, 'change_percent': 2.1}
        }
        
        self.forex = {
            'EUR/USD': {'value': 1.1850, 'change': 0.0025, 'change_percent': 0.21},
            'USD/JPY': {'value': 110.25, 'change': -0.35, 'change_percent': -0.32},
            'GBP/USD': {'value': 1.3750, 'change': 0.0080, 'change_percent': 0.58}
        }
        
        self.economic_indicators = {
            'US GDP Growth': {'value': 5.7, 'previous': 5.5, 'forecast': 5.8, 'unit': '%'},
            'US Inflation': {'value': 5.4, 'previous': 5.3, 'forecast': 5.2, 'unit': '%'},
            'US Unemployment': {'value': 4.2, 'previous': 4.4, 'forecast': 4.1, 'unit': '%'}
        }
        
        # Create mock market data object
        self.market_data = MarketData(
            indices=self.indices,
            commodities=self.commodities,
            forex=self.forex,
            sector_performance=self.sectors,
            economic_indicators=self.economic_indicators
        )
    
    @patch('stock_analysis.services.market_data_service.MarketDataService')
    @patch('stock_analysis.exporters.export_service.ExportService')
    def test_market_command_overview(self, mock_export_service_class, mock_market_service_class):
        """Test market command with no specific options (overview)."""
        # Set up mocks
        mock_market_service = MagicMock()
        mock_market_service_class.return_value = mock_market_service
        mock_market_service.get_market_overview.return_value = self.market_data
        
        mock_export_service = MagicMock()
        mock_export_service_class.return_value = mock_export_service
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_market_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_market_service.get_market_overview.assert_called_once()
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Retrieving market overview...', output)
        self.assertIn('Major Market Indices:', output)
        self.assertIn('S&P 500', output)
        self.assertIn('Sector Performance:', output)
        self.assertIn('Technology', output)
        self.assertIn('Commodity Prices:', output)
        self.assertIn('Gold', output)
        self.assertIn('Forex Rates:', output)
        self.assertIn('EUR/USD', output)
        self.assertIn('Key Economic Indicators:', output)
        self.assertIn('US GDP Growth', output)
    
    @patch('stock_analysis.services.market_data_service.MarketDataService')
    @patch('stock_analysis.exporters.export_service.ExportService')
    def test_market_command_indices(self, mock_export_service_class, mock_market_service_class):
        """Test market command with indices option."""
        # Set up mocks
        mock_market_service = MagicMock()
        mock_market_service_class.return_value = mock_market_service
        mock_market_service.get_market_indices.return_value = self.indices
        
        mock_export_service = MagicMock()
        mock_export_service_class.return_value = mock_export_service
        
        # Set indices option
        self.args.indices = True
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_market_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_market_service.get_market_indices.assert_called_once()
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Retrieving market indices...', output)
        self.assertIn('Major Market Indices:', output)
        self.assertIn('S&P 500', output)
        self.assertIn('NASDAQ', output)
        self.assertIn('Dow Jones', output)
    
    @patch('stock_analysis.services.market_data_service.MarketDataService')
    @patch('stock_analysis.exporters.export_service.ExportService')
    def test_market_command_sectors(self, mock_export_service_class, mock_market_service_class):
        """Test market command with sectors option."""
        # Set up mocks
        mock_market_service = MagicMock()
        mock_market_service_class.return_value = mock_market_service
        mock_market_service.get_sector_performance.return_value = self.sectors
        
        mock_export_service = MagicMock()
        mock_export_service_class.return_value = mock_export_service
        
        # Set sectors option
        self.args.sectors = True
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_market_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_market_service.get_sector_performance.assert_called_once()
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Retrieving sector performance...', output)
        self.assertIn('Sector Performance:', output)
        self.assertIn('Technology', output)
        self.assertIn('Healthcare', output)
        self.assertIn('Energy', output)
    
    @patch('stock_analysis.services.market_data_service.MarketDataService')
    @patch('stock_analysis.exporters.export_service.ExportService')
    def test_market_command_commodities(self, mock_export_service_class, mock_market_service_class):
        """Test market command with commodities option."""
        # Set up mocks
        mock_market_service = MagicMock()
        mock_market_service_class.return_value = mock_market_service
        mock_market_service.get_commodity_prices.return_value = self.commodities
        
        mock_export_service = MagicMock()
        mock_export_service_class.return_value = mock_export_service
        
        # Set commodities option
        self.args.commodities = True
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_market_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_market_service.get_commodity_prices.assert_called_once()
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Retrieving commodity prices...', output)
        self.assertIn('Commodity Prices:', output)
        self.assertIn('Gold', output)
        self.assertIn('Oil (WTI)', output)
        self.assertIn('Silver', output)
    
    @patch('stock_analysis.services.market_data_service.MarketDataService')
    @patch('stock_analysis.exporters.export_service.ExportService')
    def test_market_command_economic(self, mock_export_service_class, mock_market_service_class):
        """Test market command with economic option."""
        # Set up mocks
        mock_market_service = MagicMock()
        mock_market_service_class.return_value = mock_market_service
        mock_market_service.get_economic_indicators.return_value = self.economic_indicators
        
        mock_export_service = MagicMock()
        mock_export_service_class.return_value = mock_export_service
        
        # Set economic option
        self.args.economic = True
        self.args.region = 'US'
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_market_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_market_service.get_economic_indicators.assert_called_once_with(region='US')
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Retrieving economic indicators...', output)
        self.assertIn('Key Economic Indicators:', output)
        self.assertIn('US GDP Growth', output)
        self.assertIn('US Inflation', output)
        self.assertIn('US Unemployment', output)
    
    @patch('stock_analysis.services.market_data_service.MarketDataService')
    @patch('stock_analysis.exporters.export_service.ExportService')
    def test_market_command_export(self, mock_export_service_class, mock_market_service_class):
        """Test market command with export option."""
        # Set up mocks
        mock_market_service = MagicMock()
        mock_market_service_class.return_value = mock_market_service
        mock_market_service.get_market_overview.return_value = self.market_data
        
        mock_export_service = MagicMock()
        mock_export_service_class.return_value = mock_export_service
        mock_export_service.export_to_json.return_value = 'market_data.json'
        
        # Set export options
        self.args.no_export = False
        self.args.export_format = 'json'
        self.args.output = 'market_data'
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_market_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_market_service.get_market_overview.assert_called_once()
        mock_export_service.export_to_json.assert_called_once()
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Results exported to:', output)


if __name__ == '__main__':
    unittest.main()