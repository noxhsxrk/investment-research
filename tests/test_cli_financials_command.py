"""Unit tests for financials command in CLI.

This module tests the financials command for retrieving detailed financial data.
"""

import unittest
from unittest.mock import patch, MagicMock
import argparse
import sys
from io import StringIO

from stock_analysis.cli import handle_financials_command


class TestFinancialsCommand(unittest.TestCase):
    """Test cases for financials command."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock arguments
        self.args = argparse.Namespace()
        self.args.symbol = 'AAPL'
        self.args.statement = 'all'
        self.args.period = 'annual'
        self.args.years = 5
        self.args.growth = False
        self.args.compare_industry = False
        self.args.export_format = None
        self.args.output = None
        self.args.no_export = True
        self.args.verbose = False
        
        # Create mock company info
        self.company_info = {
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'exchange': 'NASDAQ'
        }
        
        # Create mock income statement
        self.income_statement = {
            'dates': ['2021-12-31', '2022-12-31', '2023-12-31'],
            'Revenue': [365817000000, 394328000000, 383285000000],
            'Cost of Revenue': [212981000000, 223546000000, 214137000000],
            'Gross Profit': [152836000000, 170782000000, 169148000000],
            'Operating Expenses': [43887000000, 48160000000, 50793000000],
            'Operating Income': [108949000000, 122622000000, 118355000000],
            'Net Income': [94680000000, 99803000000, 96995000000],
            'EPS': [5.61, 6.11, 6.14],
            'EBITDA': [123136000000, 135372000000, 130896000000]
        }
        
        # Create mock balance sheet
        self.balance_sheet = {
            'dates': ['2021-12-31', '2022-12-31', '2023-12-31'],
            'Total Assets': [351002000000, 352755000000, 355400000000],
            'Current Assets': [134836000000, 135405000000, 143700000000],
            'Cash and Cash Equivalents': [37115000000, 23646000000, 29965000000],
            'Total Liabilities': [287912000000, 302083000000, 290437000000],
            'Current Liabilities': [125481000000, 153982000000, 145198000000],
            'Long Term Debt': [109106000000, 98959000000, 95281000000],
            'Total Equity': [63090000000, 50672000000, 64963000000],
            'Retained Earnings': [5562000000, -3068000000, 1408000000]
        }
        
        # Create mock cash flow statement
        self.cash_flow = {
            'dates': ['2021-12-31', '2022-12-31', '2023-12-31'],
            'Operating Cash Flow': [104038000000, 122151000000, 113781000000],
            'Capital Expenditure': [-11085000000, [-10708000000], [-10366000000]],
            'Free Cash Flow': [92953000000, 111443000000, 103415000000],
            'Investing Cash Flow': [-14545000000, [-22354000000], [-8896000000]],
            'Financing Cash Flow': [-93353000000, [-110749000000], [-96418000000]],
            'Net Cash Flow': [-3860000000, [-10952000000], [8467000000]]
        }
        
        # Create mock financial ratios
        self.financial_ratios = {
            'dates': ['2021-12-31', '2022-12-31', '2023-12-31'],
            'Gross Margin': [0.418, 0.433, 0.441],
            'Operating Margin': [0.298, 0.311, 0.309],
            'Net Profit Margin': [0.259, 0.253, 0.253],
            'Return on Assets': [0.270, 0.283, 0.273],
            'Return on Equity': [1.501, 1.969, 1.493],
            'Return on Invested Capital': [0.510, 0.516, 0.499],
            'Current Ratio': [1.074, 0.879, 0.990],
            'Quick Ratio': [0.891, 0.721, 0.818],
            'Cash Ratio': [0.296, 0.154, 0.206],
            'Debt to Equity': [1.729, 1.953, 1.467],
            'Debt to Assets': [0.311, 0.281, 0.268],
            'Interest Coverage': [40.352, 33.222, 26.301],
            'Asset Turnover': [1.043, 1.118, 1.079],
            'Inventory Turnover': [40.388, 38.942, 35.689],
            'Receivables Turnover': [14.675, 15.073, 14.742],
            'P/E Ratio': [31.02, 21.63, 30.13],
            'P/B Ratio': [46.55, 42.67, 44.98],
            'EV/EBITDA': [24.31, 18.98, 22.45],
            'PEG Ratio': [2.58, 1.80, 2.51]
        }
        
        # Create mock industry averages
        self.industry_averages = {
            'Gross Margin': 0.325,
            'Operating Margin': 0.185,
            'Net Profit Margin': 0.145,
            'Return on Assets': 0.112,
            'Return on Equity': 0.215,
            'Current Ratio': 1.5,
            'Debt to Equity': 0.8,
            'P/E Ratio': 25.4,
            'P/B Ratio': 3.8
        }
    
    @patch('stock_analysis.services.financial_data_integration_service.FinancialDataIntegrationService')
    @patch('stock_analysis.exporters.export_service.ExportService')
    def test_financials_command_all_statements(self, mock_export_service_class, mock_integration_service_class):
        """Test financials command with all statements."""
        # Set up mocks
        mock_integration_service = MagicMock()
        mock_integration_service_class.return_value = mock_integration_service
        mock_integration_service.get_security_info.return_value = self.company_info
        mock_integration_service.get_financial_statements.side_effect = lambda symbol, statement_type, period, years: {
            'income': self.income_statement,
            'balance': self.balance_sheet,
            'cash': self.cash_flow
        }.get(statement_type, {})
        mock_integration_service.get_financial_ratios.return_value = self.financial_ratios
        
        mock_export_service = MagicMock()
        mock_export_service_class.return_value = mock_export_service
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_financials_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_integration_service.get_security_info.assert_called_once_with('AAPL')
        self.assertEqual(mock_integration_service.get_financial_statements.call_count, 3)
        mock_integration_service.get_financial_ratios.assert_called_once_with('AAPL', period='annual', years=5)
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Retrieving financial data for AAPL...', output)
        self.assertIn('Company: Apple Inc.', output)
        self.assertIn('Income Statement:', output)
        self.assertIn('Balance Sheet:', output)
        self.assertIn('Cash Flow Statement:', output)
        self.assertIn('Financial Ratios:', output)
    
    @patch('stock_analysis.services.financial_data_integration_service.FinancialDataIntegrationService')
    @patch('stock_analysis.exporters.export_service.ExportService')
    def test_financials_command_income_statement(self, mock_export_service_class, mock_integration_service_class):
        """Test financials command with income statement only."""
        # Set up mocks
        mock_integration_service = MagicMock()
        mock_integration_service_class.return_value = mock_integration_service
        mock_integration_service.get_security_info.return_value = self.company_info
        mock_integration_service.get_financial_statements.return_value = self.income_statement
        mock_integration_service.get_financial_ratios.return_value = self.financial_ratios
        
        mock_export_service = MagicMock()
        mock_export_service_class.return_value = mock_export_service
        
        # Set income statement only
        self.args.statement = 'income'
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_financials_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_integration_service.get_security_info.assert_called_once_with('AAPL')
        mock_integration_service.get_financial_statements.assert_called_once_with(
            'AAPL', statement_type='income', period='annual', years=5
        )
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Retrieving financial data for AAPL...', output)
        self.assertIn('Income Statement:', output)
        self.assertNotIn('Balance Sheet:', output)
        self.assertNotIn('Cash Flow Statement:', output)
    
    @patch('stock_analysis.services.financial_data_integration_service.FinancialDataIntegrationService')
    @patch('stock_analysis.exporters.export_service.ExportService')
    def test_financials_command_with_growth(self, mock_export_service_class, mock_integration_service_class):
        """Test financials command with growth metrics."""
        # Set up mocks
        mock_integration_service = MagicMock()
        mock_integration_service_class.return_value = mock_integration_service
        mock_integration_service.get_security_info.return_value = self.company_info
        mock_integration_service.get_financial_statements.return_value = self.income_statement
        mock_integration_service.get_financial_ratios.return_value = self.financial_ratios
        
        mock_export_service = MagicMock()
        mock_export_service_class.return_value = mock_export_service
        
        # Set income statement only and enable growth metrics
        self.args.statement = 'income'
        self.args.growth = True
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_financials_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Growth Metrics:', output)
        self.assertIn('Revenue Growth', output)
        self.assertIn('Net Income Growth', output)
    
    @patch('stock_analysis.services.financial_data_integration_service.FinancialDataIntegrationService')
    @patch('stock_analysis.exporters.export_service.ExportService')
    def test_financials_command_with_industry_comparison(self, mock_export_service_class, mock_integration_service_class):
        """Test financials command with industry comparison."""
        # Set up mocks
        mock_integration_service = MagicMock()
        mock_integration_service_class.return_value = mock_integration_service
        mock_integration_service.get_security_info.return_value = self.company_info
        mock_integration_service.get_financial_statements.return_value = self.income_statement
        mock_integration_service.get_financial_ratios.return_value = self.financial_ratios
        mock_integration_service.get_industry_averages.return_value = self.industry_averages
        
        mock_export_service = MagicMock()
        mock_export_service_class.return_value = mock_export_service
        
        # Enable industry comparison
        self.args.compare_industry = True
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_financials_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_integration_service.get_industry_averages.assert_called_once()
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Industry Comparison:', output)
        self.assertIn('Company', output)
        self.assertIn('Industry Avg', output)
        self.assertIn('Difference', output)
    
    @patch('stock_analysis.services.financial_data_integration_service.FinancialDataIntegrationService')
    @patch('stock_analysis.exporters.export_service.ExportService')
    def test_financials_command_export(self, mock_export_service_class, mock_integration_service_class):
        """Test financials command with export option."""
        # Set up mocks
        mock_integration_service = MagicMock()
        mock_integration_service_class.return_value = mock_integration_service
        mock_integration_service.get_security_info.return_value = self.company_info
        mock_integration_service.get_financial_statements.return_value = self.income_statement
        mock_integration_service.get_financial_ratios.return_value = self.financial_ratios
        
        mock_export_service = MagicMock()
        mock_export_service_class.return_value = mock_export_service
        mock_export_service.export_to_json.return_value = 'AAPL_financials.json'
        
        # Set export options
        self.args.no_export = False
        self.args.export_format = 'json'
        self.args.output = 'AAPL_financials'
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_financials_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_export_service.export_to_json.assert_called_once()
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Results exported to:', output)


if __name__ == '__main__':
    unittest.main()