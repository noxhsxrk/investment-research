"""Tests for the financial analysis engine."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from stock_analysis.analyzers.financial_analysis_engine import FinancialAnalysisEngine
from stock_analysis.models.data_models import (
    StockInfo, FinancialRatios, LiquidityRatios, ProfitabilityRatios,
    LeverageRatios, EfficiencyRatios, HealthScore
)
from stock_analysis.utils.exceptions import CalculationError


class TestFinancialAnalysisEngine(unittest.TestCase):
    """Test cases for the FinancialAnalysisEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = FinancialAnalysisEngine()
        self.symbol = "AAPL"
        
        # Create sample stock info
        self.stock_info = StockInfo(
            symbol="AAPL",
            company_name="Apple Inc.",
            current_price=150.0,
            market_cap=2500000000000.0,
            pe_ratio=25.0,
            pb_ratio=15.0,
            dividend_yield=0.005,
            beta=1.2,
            sector="Technology",
            industry="Consumer Electronics"
        )
        
        # Create sample balance sheet data
        self.balance_sheet = pd.DataFrame({
            '2023-09-30': {
                'Total Current Assets': 135000000000,
                'Total Current Liabilities': 125000000000,
                'Cash And Cash Equivalents': 30000000000,
                'Short Term Investments': 25000000000,
                'Inventory': 5000000000,
                'Total Assets': 350000000000,
                'Total Stockholder Equity': 200000000000,
                'Total Debt': 110000000000,
                'Long Term Debt': 100000000000,
                'Short Term Debt': 10000000000,
                'Net Receivables': 45000000000
            },
            '2022-09-30': {
                'Total Current Assets': 125000000000,
                'Total Current Liabilities': 115000000000,
                'Cash And Cash Equivalents': 25000000000,
                'Short Term Investments': 20000000000,
                'Inventory': 4500000000,
                'Total Assets': 325000000000,
                'Total Stockholder Equity': 180000000000,
                'Total Debt': 100000000000,
                'Long Term Debt': 90000000000,
                'Short Term Debt': 10000000000,
                'Net Receivables': 40000000000
            }
        })
        
        # Create sample income statement data
        self.income_statement = pd.DataFrame({
            '2023-09-30': {
                'Total Revenue': 380000000000,
                'Gross Profit': 170000000000,
                'Operating Income': 115000000000,
                'Net Income': 95000000000,
                'Interest Expense': 3000000000,
                'Cost Of Revenue': 210000000000
            },
            '2022-09-30': {
                'Total Revenue': 365000000000,
                'Gross Profit': 160000000000,
                'Operating Income': 110000000000,
                'Net Income': 90000000000,
                'Interest Expense': 2800000000,
                'Cost Of Revenue': 205000000000
            }
        })
        
        # Create sample cash flow data
        self.cash_flow = pd.DataFrame({
            '2023-09-30': {
                'Operating Cash Flow': 120000000000,
                'Capital Expenditure': -10000000000,
                'Free Cash Flow': 110000000000
            },
            '2022-09-30': {
                'Operating Cash Flow': 115000000000,
                'Capital Expenditure': -9000000000,
                'Free Cash Flow': 106000000000
            }
        })
        
        # Create sample historical price data
        dates = [datetime.now() - timedelta(days=i) for i in range(300)]
        prices = [150 * (1 + 0.0002 * i + 0.01 * np.sin(i/10)) for i in range(300)]
        self.historical_data = pd.DataFrame({
            'Open': prices,
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.99 for p in prices],
            'Close': prices,
            'Volume': [1000000 + 500000 * np.sin(i/10) for i in range(300)]
        }, index=dates)
    
    def test_calculate_financial_ratios(self):
        """Test calculation of financial ratios."""
        ratios = self.engine.calculate_financial_ratios(
            self.symbol,
            self.stock_info,
            self.income_statement,
            self.balance_sheet,
            self.cash_flow
        )
        
        # Verify that we got a FinancialRatios object
        self.assertIsInstance(ratios, FinancialRatios)
        
        # Verify that all ratio categories are present
        self.assertIsInstance(ratios.liquidity_ratios, LiquidityRatios)
        self.assertIsInstance(ratios.profitability_ratios, ProfitabilityRatios)
        self.assertIsInstance(ratios.leverage_ratios, LeverageRatios)
        self.assertIsInstance(ratios.efficiency_ratios, EfficiencyRatios)
        
        # Check specific ratio calculations
        # Liquidity ratios
        self.assertAlmostEqual(ratios.liquidity_ratios.current_ratio, 1.08, places=2)
        
        # Profitability ratios
        self.assertAlmostEqual(ratios.profitability_ratios.gross_margin, 0.447, places=3)
        self.assertAlmostEqual(ratios.profitability_ratios.net_profit_margin, 0.25, places=2)
        self.assertAlmostEqual(ratios.profitability_ratios.return_on_equity, 0.475, places=3)
        
        # Leverage ratios
        self.assertAlmostEqual(ratios.leverage_ratios.debt_to_equity, 0.55, places=2)
        self.assertAlmostEqual(ratios.leverage_ratios.interest_coverage, 38.33, places=2)
        
        # Efficiency ratios
        self.assertAlmostEqual(ratios.efficiency_ratios.asset_turnover, 1.086, places=2)
    
    def test_assess_company_health(self):
        """Test company health assessment."""
        # First calculate financial ratios
        ratios = self.engine.calculate_financial_ratios(
            self.symbol,
            self.stock_info,
            self.income_statement,
            self.balance_sheet,
            self.cash_flow
        )
        
        # Then assess company health
        health = self.engine.assess_company_health(self.symbol, ratios)
        
        # Verify that we got a HealthScore object
        self.assertIsInstance(health, HealthScore)
        
        # Check that scores are within expected ranges
        self.assertTrue(0 <= health.overall_score <= 100)
        self.assertTrue(0 <= health.financial_strength <= 100)
        self.assertTrue(0 <= health.profitability_health <= 100)
        self.assertTrue(0 <= health.liquidity_health <= 100)
        
        # Check risk assessment
        self.assertIn(health.risk_assessment, ["Low", "Medium", "High"])
        
        # For this sample data, we expect good health scores
        self.assertGreater(health.overall_score, 70)
        self.assertEqual(health.risk_assessment, "Low")
    
    def test_calculate_growth_metrics(self):
        """Test calculation of growth metrics."""
        growth_metrics = self.engine.calculate_growth_metrics(
            self.symbol,
            self.historical_data,
            self.income_statement,
            self.balance_sheet
        )
        
        # Verify that we got a dictionary
        self.assertIsInstance(growth_metrics, dict)
        
        # Check that expected metrics are present
        self.assertIn('revenue_growth', growth_metrics)
        self.assertIn('earnings_growth', growth_metrics)
        self.assertIn('asset_growth', growth_metrics)
        self.assertIn('equity_growth', growth_metrics)
        self.assertIn('price_cagr', growth_metrics)
        self.assertIn('eps_growth', growth_metrics)
        
        # Check specific growth calculations
        self.assertAlmostEqual(growth_metrics['revenue_growth'], 0.041, places=3)  # 4.1% growth
        self.assertAlmostEqual(growth_metrics['earnings_growth'], 0.056, places=3)  # 5.6% growth
        self.assertAlmostEqual(growth_metrics['asset_growth'], 0.077, places=3)  # 7.7% growth
        self.assertAlmostEqual(growth_metrics['equity_growth'], 0.111, places=3)  # 11.1% growth
        
        # Test with empty DataFrames
        empty_metrics = self.engine.calculate_growth_metrics(
            self.symbol,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame()
        )
        
        self.assertIsInstance(empty_metrics, dict)
        self.assertIn('revenue_growth', empty_metrics)
        self.assertIsNone(empty_metrics['revenue_growth'])
        self.assertIsNone(empty_metrics['earnings_growth'])
        self.assertIsNone(empty_metrics['asset_growth'])
        self.assertIsNone(empty_metrics['equity_growth'])
        self.assertIsNone(empty_metrics['price_cagr'])
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with empty DataFrames
        with self.assertRaises(CalculationError):
            self.engine.calculate_financial_ratios(
                self.symbol,
                self.stock_info,
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame()
            )
        
        # Test with missing key data in balance sheet
        bad_balance_sheet = pd.DataFrame({
            '2023-09-30': {
                # Missing Total Current Assets and Total Current Liabilities
                'Cash And Cash Equivalents': 30000000000,
                'Inventory': 5000000000
            }
        })
        
        # This should not raise an exception, but return ratios with None values
        ratios = self.engine.calculate_financial_ratios(
            self.symbol,
            self.stock_info,
            self.income_statement,
            bad_balance_sheet,
            self.cash_flow
        )
        
        # Verify that liquidity ratios are None
        self.assertIsNone(ratios.liquidity_ratios.current_ratio)
        self.assertIsNone(ratios.liquidity_ratios.quick_ratio)
        
    def test_growth_metrics_error_handling(self):
        """Test error handling for growth metrics calculation."""
        # Create a DataFrame with invalid data
        invalid_income_statement = pd.DataFrame({
            '2023-09-30': {
                'Total Revenue': 'invalid',  # String instead of number
                'Net Income': 95000000000
            },
            '2022-09-30': {
                'Total Revenue': 365000000000,
                'Net Income': 90000000000
            }
        })
        
        # This should not raise an exception, but return metrics with None values
        metrics = self.engine.calculate_growth_metrics(
            self.symbol,
            self.historical_data,
            invalid_income_statement,
            self.balance_sheet
        )
        
        # Verify that revenue growth is None due to invalid data
        self.assertIsNone(metrics['revenue_growth'])
        
        # Other metrics should still be calculated
        self.assertIsNotNone(metrics['earnings_growth'])
        self.assertIsNotNone(metrics['asset_growth'])
        self.assertIsNotNone(metrics['equity_growth'])


if __name__ == '__main__':
    unittest.main()