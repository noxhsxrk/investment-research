"""Integration tests for valuation engine with other components.

This module contains integration tests to verify that the ValuationEngine
works correctly with other system components.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from stock_analysis.analyzers.valuation_engine import ValuationEngine
from stock_analysis.services.stock_data_service import StockDataService
from stock_analysis.models.data_models import StockInfo, FairValueResult


class TestValuationIntegration:
    """Integration test cases for ValuationEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.valuation_engine = ValuationEngine()
        
        # Sample stock info that would come from StockDataService
        self.stock_info = StockInfo(
            symbol="AAPL",
            company_name="Apple Inc.",
            current_price=150.0,
            market_cap=2500000000000,
            pe_ratio=25.0,
            pb_ratio=5.0,
            dividend_yield=0.005,
            beta=1.2,
            sector="Technology",
            industry="Consumer Electronics"
        )
        
        # Sample financial statements that would come from StockDataService
        self.income_statement = pd.DataFrame({
            '2023-12-31': [100000, 80000, 20000, 15000],
            '2022-12-31': [90000, 72000, 18000, 13000],
            '2021-12-31': [80000, 64000, 16000, 11000]
        }, index=['Total Revenue', 'Cost Of Revenue', 'Operating Income', 'Net Income'])
        
        self.balance_sheet = pd.DataFrame({
            '2023-12-31': [300000, 150000, 100000, 50000, 16666666667],
            '2022-12-31': [280000, 140000, 90000, 45000, 16000000000],
            '2021-12-31': [260000, 130000, 80000, 40000, 15500000000]
        }, index=['Total Assets', 'Total Current Assets', 'Total Stockholder Equity', 
                 'Total Current Liabilities', 'Ordinary Shares Number'])
        
        self.cash_flow = pd.DataFrame({
            '2023-12-31': [25000, -5000, 20000],
            '2022-12-31': [22000, -4500, 17500],
            '2021-12-31': [20000, -4000, 16000]
        }, index=['Operating Cash Flow', 'Capital Expenditures', 'Free Cash Flow'])
    
    def test_end_to_end_valuation_workflow(self):
        """Test complete valuation workflow from data to recommendation."""
        # This simulates the complete workflow:
        # 1. Get stock data (mocked)
        # 2. Calculate fair value using multiple models
        # 3. Generate recommendation
        
        # Calculate fair value
        result = self.valuation_engine.calculate_fair_value(
            symbol="AAPL",
            stock_info=self.stock_info,
            income_statement=self.income_statement,
            balance_sheet=self.balance_sheet,
            cash_flow=self.cash_flow,
            peer_data=None
        )
        
        # Verify result structure
        assert isinstance(result, FairValueResult)
        assert result.current_price == 150.0
        assert result.recommendation in ["BUY", "HOLD", "SELL"]
        assert 0 <= result.confidence_level <= 1
        
        # Verify that at least one valuation model worked
        assert result.dcf_value is not None or result.peer_comparison_value is not None
        
        # Verify fair value is reasonable
        assert result.average_fair_value > 0
        
        print(f"Valuation Results for {self.stock_info.symbol}:")
        print(f"Current Price: ${result.current_price:.2f}")
        if result.dcf_value:
            print(f"DCF Value: ${result.dcf_value:.2f}")
        if result.peer_comparison_value:
            print(f"Peer Comparison Value: ${result.peer_comparison_value:.2f}")
        print(f"Average Fair Value: ${result.average_fair_value:.2f}")
        print(f"Recommendation: {result.recommendation}")
        print(f"Confidence: {result.confidence_level:.2f}")
    
    def test_valuation_with_peer_comparison(self):
        """Test valuation including peer comparison."""
        # Create peer data
        peer_data = [
            StockInfo("MSFT", "Microsoft", 300.0, 2200000000000, 28.0, 4.5, 0.007, 0.9, "Technology", "Software"),
            StockInfo("GOOGL", "Alphabet", 120.0, 1500000000000, 22.0, 3.8, 0.0, 1.1, "Technology", "Internet")
        ]
        
        result = self.valuation_engine.calculate_fair_value(
            symbol="AAPL",
            stock_info=self.stock_info,
            income_statement=self.income_statement,
            balance_sheet=self.balance_sheet,
            cash_flow=self.cash_flow,
            peer_data=peer_data
        )
        
        # Should have both DCF and peer comparison values
        assert result.dcf_value is not None
        assert result.peer_comparison_value is not None
        
        # Average should be weighted combination
        expected_avg = (result.dcf_value * 0.7 + result.peer_comparison_value * 0.3)
        assert abs(result.average_fair_value - expected_avg) < 1e-6
    
    def test_relative_valuation_metrics_calculation(self):
        """Test calculation of additional relative valuation metrics."""
        metrics = self.valuation_engine.calculate_relative_valuation_metrics(
            symbol="AAPL",
            stock_info=self.stock_info,
            income_statement=self.income_statement,
            balance_sheet=self.balance_sheet
        )
        
        # Should calculate multiple metrics
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        
        # Check specific metrics
        expected_metrics = ['pb_ratio', 'ps_ratio', 'ev_ebitda', 'peg_ratio']
        for metric in expected_metrics:
            assert metric in metrics
        
        # Verify calculated values are reasonable
        if metrics['pb_ratio'] is not None:
            assert metrics['pb_ratio'] > 0
            print(f"P/B Ratio: {metrics['pb_ratio']:.2f}")
        
        if metrics['ps_ratio'] is not None:
            assert metrics['ps_ratio'] > 0
            print(f"P/S Ratio: {metrics['ps_ratio']:.2f}")
        
        if metrics['ev_ebitda'] is not None:
            assert metrics['ev_ebitda'] > 0
            print(f"EV/EBITDA: {metrics['ev_ebitda']:.2f}")
        
        if metrics['peg_ratio'] is not None:
            assert metrics['peg_ratio'] > 0
            print(f"PEG Ratio: {metrics['peg_ratio']:.2f}")
    
    def test_dcf_model_components(self):
        """Test individual components of the DCF model."""
        # Test free cash flow extraction
        free_cash_flows = self.valuation_engine._extract_free_cash_flows(self.cash_flow)
        assert len(free_cash_flows) > 0
        assert all(fcf > 0 for fcf in free_cash_flows)
        
        # Test growth rate calculation
        growth_rate = self.valuation_engine._calculate_cash_flow_growth_rate(free_cash_flows)
        assert growth_rate is not None
        assert isinstance(growth_rate, float)
        
        # Test cash flow projection
        projected_flows = self.valuation_engine._project_cash_flows(
            free_cash_flows[-1], growth_rate, 5
        )
        assert len(projected_flows) == 5
        assert all(flow > 0 for flow in projected_flows)
        
        # Test terminal value calculation
        terminal_value = self.valuation_engine._calculate_terminal_value(
            projected_flows[-1], 0.025
        )
        assert terminal_value > 0
        assert terminal_value > projected_flows[-1]
        
        print(f"DCF Model Components:")
        print(f"Historical FCF: {free_cash_flows}")
        print(f"Growth Rate: {growth_rate:.2%}")
        print(f"Projected FCF: {[f'{flow:.0f}' for flow in projected_flows]}")
        print(f"Terminal Value: {terminal_value:.0f}")
    
    def test_recommendation_logic(self):
        """Test recommendation generation logic with different scenarios."""
        test_cases = [
            # (current_price, fair_value, expected_recommendation)
            (100.0, 130.0, "BUY"),    # 30% undervalued
            (130.0, 100.0, "SELL"),   # 30% overvalued
            (100.0, 105.0, "HOLD"),   # 5% overvalued (within hold range)
            (105.0, 100.0, "HOLD"),   # 5% overvalued (within hold range)
        ]
        
        for current_price, fair_value, expected_rec in test_cases:
            recommendation, confidence = self.valuation_engine._generate_recommendation(
                current_price, fair_value, 1.2, 2
            )
            
            assert recommendation == expected_rec
            assert 0 < confidence <= 1
            
            print(f"Price: ${current_price}, Fair Value: ${fair_value}, "
                  f"Recommendation: {recommendation}, Confidence: {confidence:.2f}")
    
    def test_integration_with_mock_data(self):
        """Test integration workflow with mock data (simulating real service calls)."""
        # This test simulates what would happen in a real application
        # where data comes from StockDataService
        
        # Simulate getting data from service
        stock_info = self.stock_info
        income_stmt = self.income_statement
        balance_sheet = self.balance_sheet
        cash_flow = self.cash_flow
        
        # Calculate valuation (this is the main integration point)
        result = self.valuation_engine.calculate_fair_value(
            "AAPL", stock_info, income_stmt, balance_sheet, cash_flow
        )
        
        # Verify the integration worked
        assert isinstance(result, FairValueResult)
        assert result.current_price == stock_info.current_price
        assert result.recommendation in ["BUY", "HOLD", "SELL"]
        
        # Verify that the valuation engine can work with data in the format
        # that would come from StockDataService
        assert hasattr(result, 'dcf_value')
        assert hasattr(result, 'peer_comparison_value')
        assert hasattr(result, 'average_fair_value')
        assert hasattr(result, 'confidence_level')
        
        print(f"Integration Test Results:")
        print(f"Symbol: {stock_info.symbol}")
        print(f"Current Price: ${result.current_price:.2f}")
        print(f"Fair Value: ${result.average_fair_value:.2f}")
        print(f"Recommendation: {result.recommendation}")
        print(f"Confidence: {result.confidence_level:.2f}")
    
    def test_error_handling_with_missing_data(self):
        """Test error handling when some financial data is missing."""
        # Test with missing income statement
        result = self.valuation_engine.calculate_fair_value(
            "AAPL", self.stock_info, pd.DataFrame(), 
            self.balance_sheet, self.cash_flow
        )
        
        assert isinstance(result, FairValueResult)
        # Should still work with available data
        
        # Test with missing cash flow (affects DCF)
        result = self.valuation_engine.calculate_fair_value(
            "AAPL", self.stock_info, self.income_statement, 
            self.balance_sheet, pd.DataFrame()
        )
        
        assert isinstance(result, FairValueResult)
        assert result.dcf_value is None  # DCF should fail without cash flow data
    
    def test_valuation_consistency(self):
        """Test that valuation results are consistent across multiple runs."""
        results = []
        
        # Run valuation multiple times
        for _ in range(3):
            result = self.valuation_engine.calculate_fair_value(
                "AAPL", self.stock_info, self.income_statement,
                self.balance_sheet, self.cash_flow
            )
            results.append(result)
        
        # Results should be identical (deterministic)
        for i in range(1, len(results)):
            assert results[i].dcf_value == results[0].dcf_value
            assert results[i].average_fair_value == results[0].average_fair_value
            assert results[i].recommendation == results[0].recommendation
            assert results[i].confidence_level == results[0].confidence_level