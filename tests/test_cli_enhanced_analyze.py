"""Unit tests for enhanced analyze command in CLI.

This module tests the enhanced analyze command with technical indicators and analyst data.
"""

import unittest
from unittest.mock import patch, MagicMock
import argparse
import sys
from io import StringIO

from stock_analysis.cli import handle_analyze_command
from stock_analysis.models.data_models import AnalysisResult, StockInfo, HealthScore, FairValueResult, SentimentResult
from stock_analysis.models.enhanced_data_models import EnhancedSecurityInfo


class TestEnhancedAnalyzeCommand(unittest.TestCase):
    """Test cases for enhanced analyze command."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock arguments
        self.args = argparse.Namespace()
        self.args.symbols = ['AAPL']
        self.args.export_format = 'json'
        self.args.output = None
        self.args.no_export = True
        self.args.verbose = False
        self.args.include_technicals = False
        self.args.include_analyst = False
        
        # Create mock stock info
        self.stock_info = StockInfo(
            symbol='AAPL',
            company_name='Apple Inc.',
            current_price=150.0,
            market_cap=2500000000000,
            pe_ratio=25.0,
            pb_ratio=15.0,
            dividend_yield=0.005,
            beta=1.2,
            sector='Technology',
            industry='Consumer Electronics'
        )
        
        # Create mock enhanced stock info
        self.enhanced_stock_info = EnhancedSecurityInfo(
            symbol='AAPL',
            name='Apple Inc.',
            current_price=150.0,
            market_cap=2500000000000,
            beta=1.2,
            earnings_growth=0.15,
            revenue_growth=0.12,
            profit_margin_trend=[0.21, 0.22, 0.23],
            rsi_14=65.5,
            macd={'macd_line': 2.5, 'signal_line': 1.8, 'histogram': 0.7},
            moving_averages={'SMA20': 148.5, 'SMA50': 145.2, 'SMA200': 140.0},
            analyst_rating='BUY',
            price_target={'low': 140.0, 'average': 170.0, 'high': 200.0},
            analyst_count=32,
            exchange='NASDAQ',
            currency='USD',
            company_description='Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.',
            key_executives=[{'name': 'Tim Cook', 'title': 'CEO'}]
        )
        
        # Create mock health score
        self.health_score = HealthScore(
            overall_score=85.0,
            financial_strength=90.0,
            profitability_health=85.0,
            liquidity_health=80.0,
            risk_assessment='Low'
        )
        
        # Create mock fair value result
        self.fair_value = FairValueResult(
            current_price=150.0,
            dcf_value=165.0,
            peer_comparison_value=160.0,
            average_fair_value=162.5,
            recommendation='BUY',
            confidence_level=0.8
        )
        
        # Create mock sentiment result
        self.sentiment = SentimentResult(
            overall_sentiment=0.65,
            positive_count=15,
            negative_count=5,
            neutral_count=10,
            key_themes=["Earnings", "Product Launch", "Market Share"],
            sentiment_trend=[0.6, 0.62, 0.65]
        )
        
        # Create mock analysis result
        self.analysis_result = AnalysisResult(
            symbol='AAPL',
            timestamp=None,
            stock_info=self.stock_info,
            financial_ratios=None,
            health_score=self.health_score,
            fair_value=self.fair_value,
            sentiment=self.sentiment,
            recommendations=['Strong Buy: Security appears undervalued with high confidence (80.0%)']
        )
        
        # Create mock enhanced analysis result
        self.enhanced_analysis_result = AnalysisResult(
            symbol='AAPL',
            timestamp=None,
            stock_info=self.enhanced_stock_info,
            financial_ratios=None,
            health_score=self.health_score,
            fair_value=self.fair_value,
            sentiment=self.sentiment,
            recommendations=['Strong Buy: Security appears undervalued with high confidence (80.0%)']
        )
    
    @patch('stock_analysis.cli.StockAnalysisOrchestrator')
    def test_analyze_command_basic(self, mock_orchestrator_class):
        """Test basic analyze command without enhanced options."""
        # Set up mock
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator
        mock_orchestrator.analyze_single_security.return_value = self.analysis_result
        mock_orchestrator.export_results.return_value = 'output.json'
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_analyze_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_orchestrator_class.assert_called_once_with(
            enable_parallel_processing=False,
            continue_on_error=True,
            include_technicals=False,
            include_analyst=False
        )
        mock_orchestrator.analyze_single_security.assert_called_once_with('AAPL')
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Analyzing 1 security(s): AAPL', output)
        
        # Verify no enhanced data is displayed
        self.assertNotIn('Technical Indicators:', output)
        self.assertNotIn('Analyst Data:', output)
    
    @patch('stock_analysis.cli.StockAnalysisOrchestrator')
    def test_analyze_command_with_technicals(self, mock_orchestrator_class):
        """Test analyze command with technical indicators."""
        # Set up mock
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator
        mock_orchestrator.analyze_single_security.return_value = self.enhanced_analysis_result
        mock_orchestrator.export_results.return_value = 'output.json'
        
        # Enable technical indicators
        self.args.include_technicals = True
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_analyze_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_orchestrator_class.assert_called_once_with(
            enable_parallel_processing=False,
            continue_on_error=True,
            include_technicals=True,
            include_analyst=False
        )
        mock_orchestrator.analyze_single_security.assert_called_once_with('AAPL')
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Analyzing 1 security(s): AAPL', output)
        
        # Verify technical indicators are included in the command
        mock_orchestrator_class.assert_called_once_with(
            enable_parallel_processing=False,
            continue_on_error=True,
            include_technicals=True,
            include_analyst=False
        )
    
    @patch('stock_analysis.cli.StockAnalysisOrchestrator')
    def test_analyze_command_with_analyst(self, mock_orchestrator_class):
        """Test analyze command with analyst data."""
        # Set up mock
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator
        mock_orchestrator.analyze_single_security.return_value = self.enhanced_analysis_result
        mock_orchestrator.export_results.return_value = 'output.json'
        
        # Enable analyst data
        self.args.include_analyst = True
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_analyze_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_orchestrator_class.assert_called_once_with(
            enable_parallel_processing=False,
            continue_on_error=True,
            include_technicals=False,
            include_analyst=True
        )
        mock_orchestrator.analyze_single_security.assert_called_once_with('AAPL')
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Analyzing 1 security(s): AAPL', output)
        
        # Verify analyst data is included in the command
        mock_orchestrator_class.assert_called_once_with(
            enable_parallel_processing=False,
            continue_on_error=True,
            include_technicals=False,
            include_analyst=True
        )
    
    @patch('stock_analysis.cli.StockAnalysisOrchestrator')
    def test_analyze_command_with_both_enhancements(self, mock_orchestrator_class):
        """Test analyze command with both technical indicators and analyst data."""
        # Set up mock
        mock_orchestrator = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator
        mock_orchestrator.analyze_single_security.return_value = self.enhanced_analysis_result
        mock_orchestrator.export_results.return_value = 'output.json'
        
        # Enable both enhancements
        self.args.include_technicals = True
        self.args.include_analyst = True
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_analyze_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_orchestrator_class.assert_called_once_with(
            enable_parallel_processing=False,
            continue_on_error=True,
            include_technicals=True,
            include_analyst=True
        )
        mock_orchestrator.analyze_single_security.assert_called_once_with('AAPL')
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Analyzing 1 security(s): AAPL', output)
        
        # Verify both technical indicators and analyst data are included in the command
        mock_orchestrator_class.assert_called_once_with(
            enable_parallel_processing=False,
            continue_on_error=True,
            include_technicals=True,
            include_analyst=True
        )


if __name__ == '__main__':
    unittest.main()