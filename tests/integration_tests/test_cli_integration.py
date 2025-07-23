"""Integration tests for CLI interface.

This module contains integration tests for the CLI interface,
testing the interaction between CLI commands and the service layer.
"""

import unittest
from unittest.mock import patch, MagicMock
import argparse
import sys
from io import StringIO
import pandas as pd
from datetime import datetime

from stock_analysis.cli import (
    handle_analyze_command,
    handle_market_command,
    handle_news_command,
    handle_financials_command,
    create_parser
)
from stock_analysis.services.enhanced_stock_data_service import EnhancedStockDataService
from stock_analysis.services.market_data_service import MarketDataService
from stock_analysis.services.news_service import NewsService
from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
from stock_analysis.models.enhanced_data_models import EnhancedSecurityInfo, MarketData, NewsItem


class TestCLIIntegration(unittest.TestCase):
    """Integration tests for CLI interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock services
        self.mock_enhanced_stock_service = MagicMock(spec=EnhancedStockDataService)
        self.mock_market_service = MagicMock(spec=MarketDataService)
        self.mock_news_service = MagicMock(spec=NewsService)
        self.mock_integration_service = MagicMock(spec=FinancialDataIntegrationService)
        
        # Sample test data
        self.sample_enhanced_security_info = EnhancedSecurityInfo(
            symbol='AAPL',
            name='Apple Inc.',
            current_price=150.0,
            market_cap=2500000000000,
            pe_ratio=25.0,
            sector='Technology',
            industry='Consumer Electronics',
            exchange='NASDAQ',
            currency='USD',
            company_description='Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.',
            earnings_growth=0.15,
            revenue_growth=0.08,
            profit_margin_trend=[0.21, 0.22, 0.25],
            rsi_14=65.3,
            macd={'macd': 2.5, 'signal': 1.8, 'histogram': 0.7},
            moving_averages={'sma_20': 148.5, 'sma_50': 145.2, 'sma_200': 140.0},
            analyst_rating='Buy',
            price_target={'low': 140.0, 'average': 170.0, 'high': 200.0},
            analyst_count=35
        )
        
        self.sample_market_data = MarketData(
            indices={
                'S&P 500': {
                    'price': 4500.0,
                    'change': 15.0,
                    'change_percent': 0.33
                },
                'NASDAQ': {
                    'price': 14000.0,
                    'change': 50.0,
                    'change_percent': 0.36
                }
            },
            commodities={
                'Crude Oil': {
                    'price': 75.0,
                    'change': -0.5,
                    'change_percent': -0.66
                },
                'Gold': {
                    'price': 1850.0,
                    'change': 10.0,
                    'change_percent': 0.54
                }
            },
            forex={
                'EUR/USD': {
                    'price': 1.08,
                    'change': 0.002,
                    'change_percent': 0.19
                },
                'USD/JPY': {
                    'price': 145.5,
                    'change': -0.3,
                    'change_percent': -0.21
                }
            },
            sector_performance={
                'Technology': 1.2,
                'Healthcare': 0.5,
                'Financials': -0.3,
                'Consumer Discretionary': 0.8,
                'Energy': -1.0
            },
            economic_indicators={
                'US GDP Growth': {
                    'value': 2.1,
                    'previous': 2.0,
                    'change': 0.1
                },
                'US Inflation': {
                    'value': 3.2,
                    'previous': 3.4,
                    'change': -0.2
                },
                'US Unemployment': {
                    'value': 3.8,
                    'previous': 3.7,
                    'change': 0.1
                }
            }
        )
        
        self.sample_news_items = [
            NewsItem(
                title='Apple Announces New iPhone',
                source='Tech News',
                url='https://example.com/news/1',
                published_at=datetime.fromisoformat('2023-01-15T12:30:00'),
                summary='Apple has announced the new iPhone model.',
                sentiment=0.75,
                impact='high',
                categories=['product', 'technology']
            ),
            NewsItem(
                title='Apple Reports Record Earnings',
                source='Financial News',
                url='https://example.com/news/2',
                published_at=datetime.fromisoformat('2023-01-10T15:45:00'),
                summary='Apple reported record earnings for the quarter.',
                sentiment=0.85,
                impact='high',
                categories=['earnings', 'financial']
            )
        ]
        
        self.sample_economic_calendar = [
            {
                'event': 'US Fed Interest Rate Decision',
                'date': datetime.fromisoformat('2023-02-01T14:00:00'),
                'importance': 'high',
                'previous': '5.25%',
                'forecast': '5.25%',
                'actual': None
            },
            {
                'event': 'US Non-Farm Payrolls',
                'date': datetime.fromisoformat('2023-02-03T13:30:00'),
                'importance': 'high',
                'previous': '216K',
                'forecast': '180K',
                'actual': None
            }
        ]
        
        self.sample_financial_statements = {
            'income': pd.DataFrame({
                'Revenue': [365817000000, 394328000000, 383285000000],
                'Cost of Revenue': [212981000000, 223546000000, 214137000000],
                'Gross Profit': [152836000000, 170782000000, 169148000000],
                'Operating Expenses': [43887000000, 48160000000, 50793000000],
                'Operating Income': [108949000000, 122622000000, 118355000000],
                'Net Income': [94680000000, 99803000000, 96995000000],
                'EPS': [5.61, 6.11, 6.14],
                'EBITDA': [123136000000, 135372000000, 130896000000]
            }, index=pd.date_range(start='2021-01-01', periods=3, freq='Y')),
            'balance': pd.DataFrame({
                'Total Assets': [351002000000, 352755000000, 355400000000],
                'Current Assets': [134836000000, 135405000000, 143700000000],
                'Cash and Cash Equivalents': [37115000000, 23646000000, 29965000000],
                'Total Liabilities': [287912000000, 302083000000, 290437000000],
                'Current Liabilities': [125481000000, 153982000000, 145198000000],
                'Long Term Debt': [109106000000, 98959000000, 95281000000],
                'Total Equity': [63090000000, 50672000000, 64963000000]
            }, index=pd.date_range(start='2021-01-01', periods=3, freq='Y')),
            'cash': pd.DataFrame({
                'Operating Cash Flow': [104038000000, 122151000000, 113781000000],
                'Capital Expenditure': [-11085000000, -10708000000, -10366000000],
                'Free Cash Flow': [92953000000, 111443000000, 103415000000],
                'Investing Cash Flow': [-14545000000, -22354000000, -8896000000],
                'Financing Cash Flow': [-93353000000, -110749000000, -96418000000],
                'Net Cash Flow': [-3860000000, -10952000000, 8467000000]
            }, index=pd.date_range(start='2021-01-01', periods=3, freq='Y'))
        }
        
        self.sample_financial_ratios = pd.DataFrame({
            'Gross Margin': [0.418, 0.433, 0.441],
            'Operating Margin': [0.298, 0.311, 0.309],
            'Net Profit Margin': [0.259, 0.253, 0.253],
            'Return on Assets': [0.270, 0.283, 0.273],
            'Return on Equity': [1.501, 1.969, 1.493],
            'Current Ratio': [1.074, 0.879, 0.990],
            'Debt to Equity': [1.729, 1.953, 1.467],
            'P/E Ratio': [31.02, 21.63, 30.13]
        }, index=pd.date_range(start='2021-01-01', periods=3, freq='Y'))
    
    def test_cli_parser_integration(self):
        """Test CLI parser integration."""
        # Create parser
        parser = create_parser()
        
        # Test analyze command
        args = parser.parse_args(['analyze', 'AAPL', '--include-technicals', '--include-analyst'])
        self.assertEqual(args.command, 'analyze')
        self.assertEqual(args.symbol, 'AAPL')
        self.assertTrue(args.include_technicals)
        self.assertTrue(args.include_analyst)
        
        # Test market command
        args = parser.parse_args(['market', '--indices', '--sectors'])
        self.assertEqual(args.command, 'market')
        self.assertTrue(args.indices)
        self.assertTrue(args.sectors)
        
        # Test news command
        args = parser.parse_args(['news', '--symbol', 'AAPL', '--limit', '5'])
        self.assertEqual(args.command, 'news')
        self.assertEqual(args.symbol, 'AAPL')
        self.assertEqual(args.limit, 5)
        
        # Test financials command
        args = parser.parse_args(['financials', 'AAPL', '--statement', 'income', '--period', 'annual'])
        self.assertEqual(args.command, 'financials')
        self.assertEqual(args.symbol, 'AAPL')
        self.assertEqual(args.statement, 'income')
        self.assertEqual(args.period, 'annual')
    
    @patch('stock_analysis.services.enhanced_stock_data_service.EnhancedStockDataService')
    def test_analyze_command_integration(self, mock_enhanced_stock_service_class):
        """Test analyze command integration."""
        # Set up mock
        mock_enhanced_stock_service_class.return_value = self.mock_enhanced_stock_service
        self.mock_enhanced_stock_service.get_enhanced_security_info.return_value = self.sample_enhanced_security_info
        
        # Create args
        args = argparse.Namespace()
        args.symbol = 'AAPL'
        args.include_technicals = True
        args.include_analyst = True
        args.period = '1y'
        args.export_format = None
        args.output = None
        args.no_export = True
        args.verbose = False
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_analyze_command(args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        self.mock_enhanced_stock_service.get_enhanced_security_info.assert_called_once_with(
            'AAPL', include_technicals=True, include_analyst=True
        )
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Apple Inc. (AAPL)', output)
        self.assertIn('Current Price: $150.00', output)
        self.assertIn('Technical Indicators:', output)
        self.assertIn('RSI (14): 65.30', output)
        self.assertIn('Analyst Recommendations:', output)
        self.assertIn('Rating: Buy', output)
    
    @patch('stock_analysis.services.market_data_service.MarketDataService')
    def test_market_command_integration(self, mock_market_service_class):
        """Test market command integration."""
        # Set up mock
        mock_market_service_class.return_value = self.mock_market_service
        self.mock_market_service.get_market_overview.return_value = self.sample_market_data
        self.mock_market_service.get_indices.return_value = self.sample_market_data.indices
        self.mock_market_service.get_sector_performance.return_value = self.sample_market_data.sector_performance
        self.mock_market_service.get_commodities.return_value = self.sample_market_data.commodities
        self.mock_market_service.get_forex.return_value = self.sample_market_data.forex
        self.mock_market_service.get_economic_indicators.return_value = self.sample_market_data.economic_indicators
        
        # Create args
        args = argparse.Namespace()
        args.indices = True
        args.sectors = True
        args.commodities = False
        args.forex = False
        args.economic = False
        args.all = False
        args.export_format = None
        args.output = None
        args.no_export = True
        args.verbose = False
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_market_command(args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        self.mock_market_service.get_indices.assert_called_once()
        self.mock_market_service.get_sector_performance.assert_called_once()
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Market Indices:', output)
        self.assertIn('S&P 500:', output)
        self.assertIn('4500.00', output)
        self.assertIn('Sector Performance:', output)
        self.assertIn('Technology:', output)
        self.assertIn('+1.20%', output)
    
    @patch('stock_analysis.services.news_service.NewsService')
    def test_news_command_integration(self, mock_news_service_class):
        """Test news command integration."""
        # Set up mock
        mock_news_service_class.return_value = self.mock_news_service
        self.mock_news_service.get_company_news.return_value = self.sample_news_items
        self.mock_news_service.get_market_news.return_value = self.sample_news_items
        self.mock_news_service.get_economic_calendar.return_value = self.sample_economic_calendar
        
        # Create args
        args = argparse.Namespace()
        args.symbol = 'AAPL'
        args.market = False
        args.economic_calendar = False
        args.limit = 5
        args.export_format = None
        args.output = None
        args.no_export = True
        args.verbose = False
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_news_command(args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        self.mock_news_service.get_company_news.assert_called_once_with('AAPL', limit=5)
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('News for AAPL:', output)
        self.assertIn('Apple Announces New iPhone', output)
        self.assertIn('Sentiment: Positive (0.75)', output)
    
    @patch('stock_analysis.services.financial_data_integration_service.FinancialDataIntegrationService')
    def test_financials_command_integration(self, mock_integration_service_class):
        """Test financials command integration."""
        # Set up mock
        mock_integration_service_class.return_value = self.mock_integration_service
        self.mock_integration_service.get_security_info.return_value = {
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'exchange': 'NASDAQ'
        }
        self.mock_integration_service.get_financial_statements.side_effect = lambda symbol, statement_type, period, years: {
            'income': self.sample_financial_statements['income'],
            'balance': self.sample_financial_statements['balance'],
            'cash': self.sample_financial_statements['cash']
        }.get(statement_type, pd.DataFrame())
        self.mock_integration_service.get_financial_ratios.return_value = self.sample_financial_ratios
        
        # Create args
        args = argparse.Namespace()
        args.symbol = 'AAPL'
        args.statement = 'all'
        args.period = 'annual'
        args.years = 3
        args.growth = False
        args.compare_industry = False
        args.export_format = None
        args.output = None
        args.no_export = True
        args.verbose = False
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_financials_command(args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        self.mock_integration_service.get_security_info.assert_called_once_with('AAPL')
        self.assertEqual(self.mock_integration_service.get_financial_statements.call_count, 3)
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Retrieving financial data for AAPL...', output)
        self.assertIn('Company: Apple Inc.', output)
        self.assertIn('Income Statement:', output)
        self.assertIn('Balance Sheet:', output)
        self.assertIn('Cash Flow Statement:', output)
    
    def test_cli_error_handling(self):
        """Test CLI error handling."""
        # Set up mock to raise error
        with patch('stock_analysis.services.enhanced_stock_data_service.EnhancedStockDataService') as mock_service_class:
            mock_service = MagicMock()
            mock_service_class.return_value = mock_service
            mock_service.get_enhanced_security_info.side_effect = Exception("Test error")
            
            # Create args
            args = argparse.Namespace()
            args.symbol = 'AAPL'
            args.include_technicals = False
            args.include_analyst = False
            args.period = '1y'
            args.export_format = None
            args.output = None
            args.no_export = True
            args.verbose = True
            
            # Capture stdout
            captured_output = StringIO()
            sys.stdout = captured_output
            
            # Execute command
            result = handle_analyze_command(args)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Verify results
            self.assertEqual(result, 1)  # Error return code
            
            # Check output
            output = captured_output.getvalue()
            self.assertIn('Error:', output)
            self.assertIn('Test error', output)


if __name__ == '__main__':
    unittest.main()