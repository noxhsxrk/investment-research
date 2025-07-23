"""Tests for financial statements integration.

This module contains tests for the financial statements integration functionality
of the FinancialDataIntegrationService.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime
import pytest

from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
from stock_analysis.adapters.base_adapter import FinancialDataAdapter
from stock_analysis.utils.exceptions import DataRetrievalError


class TestFinancialStatementsIntegration(unittest.TestCase):
    """Test cases for financial statements integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock adapters
        self.mock_adapter1 = Mock(spec=FinancialDataAdapter)
        self.mock_adapter1.name = "adapter1"
        
        self.mock_adapter2 = Mock(spec=FinancialDataAdapter)
        self.mock_adapter2.name = "adapter2"
        
        # Create service with mock adapters
        self.service = FinancialDataIntegrationService(
            adapters=[self.mock_adapter1, self.mock_adapter2]
        )
        
        # Override source priorities for testing
        self.service.source_priorities = {
            "financial_statements": {"adapter1": 1, "adapter2": 2},
        }
        
        # Sample test data - Income Statement
        self.sample_income_annual = pd.DataFrame({
            'Revenue': [365000000000, 350000000000, 325000000000],
            'CostOfRevenue': [190000000000, 185000000000, 175000000000],
            'GrossProfit': [175000000000, 165000000000, 150000000000],
            'OperatingExpenses': [50000000000, 48000000000, 45000000000],
            'OperatingIncome': [125000000000, 117000000000, 105000000000],
            'NetIncome': [95000000000, 90000000000, 85000000000],
            'EPS': [5.5, 5.2, 4.9],
            'DilutedEPS': [5.4, 5.1, 4.8],
            'reportedCurrency': ['USD', 'USD', 'USD']
        }, index=pd.date_range(start='2020-01-01', periods=3, freq='Y'))
        
        # Sample test data - Balance Sheet
        self.sample_balance_annual = pd.DataFrame({
            'TotalAssets': [350000000000, 330000000000, 310000000000],
            'CurrentAssets': [150000000000, 140000000000, 130000000000],
            'CashAndCashEquivalents': [50000000000, 45000000000, 40000000000],
            'TotalLiabilities': [150000000000, 140000000000, 130000000000],
            'CurrentLiabilities': [80000000000, 75000000000, 70000000000],
            'TotalShareholderEquity': [200000000000, 190000000000, 180000000000],
            'reportedCurrency': ['USD', 'USD', 'USD']
        }, index=pd.date_range(start='2020-01-01', periods=3, freq='Y'))
        
        # Sample test data - Cash Flow
        self.sample_cash_flow_annual = pd.DataFrame({
            'OperatingCashFlow': [110000000000, 105000000000, 100000000000],
            'CapitalExpenditures': [-15000000000, -14000000000, -13000000000],
            'FreeCashFlow': [95000000000, 91000000000, 87000000000],
            'DividendsPaid': [-15000000000, -14000000000, -13000000000],
            'NetInvestingCashFlow': [-30000000000, -28000000000, -26000000000],
            'NetFinancingCashFlow': [-40000000000, -38000000000, -36000000000],
            'reportedCurrency': ['USD', 'USD', 'USD']
        }, index=pd.date_range(start='2020-01-01', periods=3, freq='Y'))
        
        # Sample test data - Partial Income Statement
        self.sample_income_partial = pd.DataFrame({
            'Revenue': [365000000000, 350000000000],
            'GrossProfit': [175000000000, 165000000000],
            'NetIncome': [95000000000, 90000000000],
            'EPS': [5.5, 5.2],
            'reportedCurrency': ['USD', 'USD']
        }, index=pd.date_range(start='2021-01-01', periods=2, freq='Y'))
        
        # Sample test data - Quarterly Income Statement
        self.sample_income_quarterly = pd.DataFrame({
            'Revenue': [90000000000, 88000000000, 85000000000, 82000000000],
            'GrossProfit': [45000000000, 44000000000, 42000000000, 40000000000],
            'NetIncome': [24000000000, 23000000000, 22000000000, 21000000000],
            'EPS': [1.4, 1.35, 1.3, 1.25],
            'reportedCurrency': ['USD', 'USD', 'USD', 'USD']
        }, index=pd.date_range(start='2022-01-01', periods=4, freq='3M'))
    
    def test_validate_financial_statements(self):
        """Test validation of financial statements."""
        # Valid income statement
        self.assertTrue(self.service._validate_financial_statements(
            self.sample_income_annual, 'income'))
        
        # Valid balance sheet
        self.assertTrue(self.service._validate_financial_statements(
            self.sample_balance_annual, 'balance'))
        
        # Valid cash flow statement
        self.assertTrue(self.service._validate_financial_statements(
            self.sample_cash_flow_annual, 'cash'))
        
        # Invalid income statement (missing key metrics)
        invalid_income = pd.DataFrame({
            'Revenue': [365000000000],
            'OtherMetric': [10000000000]
        }, index=pd.date_range(start='2020-01-01', periods=1, freq='Y'))
        self.assertFalse(self.service._validate_financial_statements(
            invalid_income, 'income'))
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        self.assertFalse(self.service._validate_financial_statements(
            empty_df, 'income'))
    
    def test_normalize_financial_statements(self):
        """Test normalization of financial statements."""
        # Test with non-datetime index
        df = pd.DataFrame({
            'Revenue': [365000000000, 350000000000, 325000000000],
            'NetIncome': [95000000000, 90000000000, 85000000000],
            'EPS': ['5.5', '5.2', '4.9'],  # String values
            'reportedCurrency': ['USD', 'USD', 'USD']
        }, index=['2022', '2021', '2020'])  # String index
        
        normalized = self.service._normalize_financial_statements(df, 'income')
        
        # Verify index is converted to datetime
        self.assertIsInstance(normalized.index, pd.DatetimeIndex)
        
        # Verify numeric columns are converted to numeric types
        self.assertTrue(pd.api.types.is_numeric_dtype(normalized['Revenue']))
        self.assertIsInstance(normalized['EPS'].iloc[0], (int, float))
        
        # Verify non-numeric columns are preserved
        self.assertIsInstance(normalized['reportedCurrency'].iloc[0], str)
    
    def test_combine_financial_statements(self):
        """Test combining financial statements from multiple sources."""
        # Set up test data with different time ranges
        adapter1_data = self.sample_income_annual  # 2020-2022
        adapter2_data = self.sample_income_quarterly  # Q1-Q4 2022
        
        combined = self.service._combine_financial_statements(
            {'adapter1': adapter1_data, 'adapter2': adapter2_data}, 'income')
        
        # Verify combined data has all periods
        self.assertEqual(len(combined), 7)  # 3 annual + 4 quarterly
        
        # Verify columns are preserved
        for col in adapter1_data.columns:
            self.assertIn(col, combined.columns)
        
        # Verify source information is added
        self.assertTrue('_sources' in combined.attrs)
        self.assertEqual(combined.attrs['_sources'], ['adapter1', 'adapter2'])
    
    def test_get_financial_statements_success(self):
        """Test getting financial statements successfully."""
        # Set up mock adapter responses
        self.mock_adapter1.get_financial_statements.return_value = self.sample_income_annual
        
        # Test with cache disabled
        result = self.service.get_financial_statements(
            "AAPL", statement_type="income", period="annual", use_cache=False)
        
        # Verify result
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)
        self.assertEqual(result['Revenue'].iloc[0], 365000000000)
        
        # Verify adapters were called
        self.mock_adapter1.get_financial_statements.assert_called_once_with(
            "AAPL", "income", "annual")
        # Both adapters are called in our implementation
        self.mock_adapter2.get_financial_statements.assert_called_once_with(
            "AAPL", "income", "annual")
    
    def test_get_financial_statements_fallback(self):
        """Test fallback to secondary adapter for financial statements."""
        # Set up mock adapter responses
        self.mock_adapter1.get_financial_statements.side_effect = DataRetrievalError("Test error")
        self.mock_adapter2.get_financial_statements.return_value = self.sample_income_annual
        
        # Test with cache disabled
        result = self.service.get_financial_statements(
            "AAPL", statement_type="income", period="annual", use_cache=False)
        
        # Verify result
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)
        self.assertEqual(result['Revenue'].iloc[0], 365000000000)
        
        # Verify both adapters were called
        self.mock_adapter1.get_financial_statements.assert_called_once_with(
            "AAPL", "income", "annual")
        self.mock_adapter2.get_financial_statements.assert_called_once_with(
            "AAPL", "income", "annual")
    
    def test_get_financial_statements_combine_partial(self):
        """Test combining partial financial statements from multiple sources."""
        # Set up mock adapter responses with partial data
        self.mock_adapter1.get_financial_statements.return_value = self.sample_income_partial
        self.mock_adapter2.get_financial_statements.return_value = self.sample_income_annual
        
        # Override validation to treat both results as valid but incomplete
        with patch.object(self.service, '_validate_financial_statements', return_value=True):
            # Test with cache disabled
            result = self.service.get_financial_statements(
                "AAPL", statement_type="income", period="annual", use_cache=False)
            
            # Verify result contains data (exact length depends on implementation)
            self.assertFalse(result.empty)
            
            # Verify both adapters were called
            self.mock_adapter1.get_financial_statements.assert_called_once_with(
                "AAPL", "income", "annual")
            self.mock_adapter2.get_financial_statements.assert_called_once_with(
                "AAPL", "income", "annual")
    
    def test_get_financial_statements_all_fail(self):
        """Test handling when all adapters fail for financial statements."""
        # Set up mock adapter responses
        self.mock_adapter1.get_financial_statements.side_effect = DataRetrievalError("Test error 1")
        self.mock_adapter2.get_financial_statements.side_effect = DataRetrievalError("Test error 2")
        
        # Test with cache disabled
        with self.assertRaises(DataRetrievalError):
            self.service.get_financial_statements(
                "AAPL", statement_type="income", period="annual", use_cache=False)
        
        # Verify both adapters were called
        self.mock_adapter1.get_financial_statements.assert_called_once_with(
            "AAPL", "income", "annual")
        self.mock_adapter2.get_financial_statements.assert_called_once_with(
            "AAPL", "income", "annual")
    
    def test_get_enhanced_financial_statements(self):
        """Test getting enhanced financial statements."""
        # Set up mock adapter responses
        self.mock_adapter1.get_financial_statements.return_value = self.sample_income_annual
        
        # Mock cache to avoid actual caching
        with patch('stock_analysis.services.financial_data_integration_service.get_cache_manager'):
            # Test getting enhanced financial statements
            result = self.service.get_enhanced_financial_statements(
                "AAPL", 
                statement_types=["income", "balance", "cash"],
                period="annual",
                use_cache=False
            )
            
            # Verify result structure
            self.assertIn('income', result)
            self.assertIn('balance', result)
            self.assertIn('cash', result)
            
            # Verify income statement data
            self.assertEqual(result['income']['Revenue'].iloc[0], 365000000000)
            
            # Verify adapter calls
            self.assertEqual(self.mock_adapter1.get_financial_statements.call_count, 3)


if __name__ == '__main__':
    unittest.main()