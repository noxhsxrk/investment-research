"""Tests for the valuation engine module.

This module contains unit tests for the ValuationEngine class and its methods.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from stock_analysis.analyzers.valuation_engine import ValuationEngine
from stock_analysis.models.data_models import StockInfo, FairValueResult
from stock_analysis.utils.exceptions import CalculationError


class TestValuationEngine:
    """Test cases for ValuationEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValuationEngine()
        
        # Sample stock info
        self.stock_info = StockInfo(
            symbol="AAPL",
            company_name="Apple Inc.",
            current_price=150.0,
            market_cap=2500000000000,  # 2.5T
            pe_ratio=25.0,
            pb_ratio=5.0,
            dividend_yield=0.005,
            beta=1.2,
            sector="Technology",
            industry="Consumer Electronics"
        )
        
        # Sample financial data
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
        
        # Sample peer data
        self.peer_data = [
            StockInfo("MSFT", "Microsoft", 300.0, 2200000000000, 28.0, 4.5, 0.007, 0.9, "Technology", "Software"),
            StockInfo("GOOGL", "Alphabet", 120.0, 1500000000000, 22.0, 3.8, 0.0, 1.1, "Technology", "Internet"),
            StockInfo("AMZN", "Amazon", 140.0, 1400000000000, 45.0, 8.2, 0.0, 1.3, "Technology", "E-commerce")
        ]
    
    def test_initialization(self):
        """Test ValuationEngine initialization."""
        engine = ValuationEngine()
        assert engine.default_terminal_growth_rate == 0.025
        assert engine.default_discount_rate == 0.10
        assert engine.default_projection_years == 5
        assert engine.peer_weight == 0.3
        assert engine.dcf_weight == 0.7
    
    def test_calculate_fair_value_success(self):
        """Test successful fair value calculation."""
        result = self.engine.calculate_fair_value(
            "AAPL", self.stock_info, self.income_statement, 
            self.balance_sheet, self.cash_flow, self.peer_data
        )
        
        assert isinstance(result, FairValueResult)
        assert result.current_price == 150.0
        assert result.dcf_value is not None
        assert result.peer_comparison_value is not None
        assert result.average_fair_value > 0
        assert result.recommendation in ["BUY", "HOLD", "SELL"]
        assert 0 <= result.confidence_level <= 1
    
    def test_calculate_fair_value_no_peer_data(self):
        """Test fair value calculation without peer data."""
        result = self.engine.calculate_fair_value(
            "AAPL", self.stock_info, self.income_statement, 
            self.balance_sheet, self.cash_flow, None
        )
        
        assert isinstance(result, FairValueResult)
        assert result.peer_comparison_value is None
        assert result.dcf_value is not None
    
    def test_calculate_dcf_value_success(self):
        """Test successful DCF value calculation."""
        dcf_value = self.engine.calculate_dcf_value(
            "AAPL", self.stock_info, self.income_statement, 
            self.balance_sheet, self.cash_flow
        )
        
        assert dcf_value is not None
        assert dcf_value > 0
        assert isinstance(dcf_value, float)
    
    def test_calculate_dcf_value_insufficient_data(self):
        """Test DCF calculation with insufficient data."""
        # Empty cash flow statement
        empty_cash_flow = pd.DataFrame()
        
        dcf_value = self.engine.calculate_dcf_value(
            "AAPL", self.stock_info, self.income_statement, 
            self.balance_sheet, empty_cash_flow
        )
        
        assert dcf_value is None
    
    def test_calculate_peer_comparison_value_success(self):
        """Test successful peer comparison value calculation."""
        peer_value = self.engine.calculate_peer_comparison_value(
            "AAPL", self.stock_info, self.peer_data
        )
        
        assert peer_value is not None
        assert peer_value > 0
        assert isinstance(peer_value, float)
    
    def test_calculate_peer_comparison_value_no_peers(self):
        """Test peer comparison with no peer data."""
        peer_value = self.engine.calculate_peer_comparison_value(
            "AAPL", self.stock_info, []
        )
        
        assert peer_value is None
    
    def test_calculate_peer_comparison_value_no_pe_ratios(self):
        """Test peer comparison with peers having no P/E ratios."""
        peers_no_pe = [
            StockInfo("TEST", "Test Corp", 100.0, 1000000000, None, 2.0, 0.0, 1.0, "Technology", "Software")
        ]
        
        peer_value = self.engine.calculate_peer_comparison_value(
            "AAPL", self.stock_info, peers_no_pe
        )
        
        assert peer_value is None
    
    def test_calculate_peg_ratio_success(self):
        """Test successful PEG ratio calculation."""
        peg_ratio = self.engine.calculate_peg_ratio(
            "AAPL", self.stock_info, self.income_statement
        )
        
        assert peg_ratio is not None
        assert isinstance(peg_ratio, float)
        assert peg_ratio > 0
    
    def test_calculate_peg_ratio_no_pe(self):
        """Test PEG ratio calculation with no P/E ratio."""
        stock_no_pe = StockInfo(
            "TEST", "Test Corp", 100.0, 1000000000, None, 2.0, 0.0, 1.0, "Technology", "Software"
        )
        
        peg_ratio = self.engine.calculate_peg_ratio(
            "TEST", stock_no_pe, self.income_statement
        )
        
        assert peg_ratio is None
    
    def test_calculate_peg_ratio_no_growth(self):
        """Test PEG ratio calculation with no earnings growth."""
        # Create income statement with flat earnings
        flat_income = pd.DataFrame({
            '2023-12-31': [15000],
            '2022-12-31': [15000],
            '2021-12-31': [15000]
        }, index=['Net Income'])
        
        peg_ratio = self.engine.calculate_peg_ratio(
            "AAPL", self.stock_info, flat_income
        )
        
        assert peg_ratio is None
    
    def test_extract_free_cash_flows_direct(self):
        """Test extracting free cash flows when directly available."""
        free_cash_flows = self.engine._extract_free_cash_flows(self.cash_flow)
        
        assert len(free_cash_flows) == 3
        assert free_cash_flows == [20000.0, 17500.0, 16000.0]
    
    def test_extract_free_cash_flows_calculated(self):
        """Test extracting free cash flows by calculation."""
        # Remove direct free cash flow row
        cash_flow_no_fcf = self.cash_flow.drop('Free Cash Flow')
        
        free_cash_flows = self.engine._extract_free_cash_flows(cash_flow_no_fcf)
        
        assert len(free_cash_flows) == 3
        # Should be Operating CF + Capex (capex is negative)
        expected = [25000 + (-5000), 22000 + (-4500), 20000 + (-4000)]
        assert free_cash_flows == expected
    
    def test_extract_free_cash_flows_empty(self):
        """Test extracting free cash flows from empty DataFrame."""
        empty_df = pd.DataFrame()
        
        free_cash_flows = self.engine._extract_free_cash_flows(empty_df)
        
        assert free_cash_flows == []
    
    def test_calculate_cash_flow_growth_rate(self):
        """Test cash flow growth rate calculation."""
        cash_flows = [20000.0, 18000.0, 16000.0]  # Growing cash flows
        
        growth_rate = self.engine._calculate_cash_flow_growth_rate(cash_flows)
        
        assert growth_rate is not None
        assert isinstance(growth_rate, float)
        assert growth_rate > 0  # Should be positive for growing cash flows
    
    def test_calculate_cash_flow_growth_rate_insufficient_data(self):
        """Test growth rate calculation with insufficient data."""
        cash_flows = [20000.0]  # Only one data point
        
        growth_rate = self.engine._calculate_cash_flow_growth_rate(cash_flows)
        
        assert growth_rate is None
    
    def test_project_cash_flows(self):
        """Test cash flow projection."""
        base_flow = 20000.0
        growth_rate = 0.1  # 10% growth
        years = 5
        
        projected = self.engine._project_cash_flows(base_flow, growth_rate, years)
        
        assert len(projected) == years
        assert all(flow > 0 for flow in projected)
        assert projected[0] > base_flow  # First year should be higher than base
    
    def test_calculate_terminal_value(self):
        """Test terminal value calculation."""
        final_cash_flow = 25000.0
        terminal_growth = 0.025
        
        terminal_value = self.engine._calculate_terminal_value(final_cash_flow, terminal_growth)
        
        assert terminal_value > 0
        assert terminal_value > final_cash_flow  # Should be much larger
    
    def test_discount_cash_flows(self):
        """Test cash flow discounting."""
        cash_flows = [22000.0, 24200.0, 26620.0, 29282.0, 32210.0]
        terminal_value = 500000.0
        discount_rate = 0.10
        
        present_value = self.engine._discount_cash_flows(cash_flows, terminal_value, discount_rate)
        
        assert present_value > 0
        assert present_value < sum(cash_flows) + terminal_value  # Should be less due to discounting
    
    def test_get_shares_outstanding_from_balance_sheet(self):
        """Test getting shares outstanding from balance sheet."""
        shares = self.engine._get_shares_outstanding(self.balance_sheet, self.stock_info)
        
        assert shares is not None
        assert shares > 0
        assert shares == 16666666667.0  # From balance sheet
    
    def test_get_shares_outstanding_from_market_cap(self):
        """Test calculating shares outstanding from market cap."""
        # Balance sheet without shares data
        balance_sheet_no_shares = self.balance_sheet.drop('Ordinary Shares Number')
        
        shares = self.engine._get_shares_outstanding(balance_sheet_no_shares, self.stock_info)
        
        assert shares is not None
        assert shares > 0
        # Should be market_cap / current_price
        expected = self.stock_info.market_cap / self.stock_info.current_price
        assert abs(shares - expected) < 1e-6
    
    def test_calculate_earnings_growth_rate(self):
        """Test earnings growth rate calculation."""
        growth_rate = self.engine._calculate_earnings_growth_rate(self.income_statement)
        
        assert growth_rate is not None
        assert isinstance(growth_rate, float)
        assert growth_rate > 0  # Should be positive for growing earnings
    
    def test_generate_recommendation_buy(self):
        """Test recommendation generation for undervalued stock."""
        current_price = 100.0
        fair_value = 130.0  # 30% undervalued
        peg_ratio = 0.8  # Good PEG ratio
        num_models = 2
        
        recommendation, confidence = self.engine._generate_recommendation(
            current_price, fair_value, peg_ratio, num_models
        )
        
        assert recommendation == "BUY"
        assert 0 < confidence <= 1
    
    def test_generate_recommendation_sell(self):
        """Test recommendation generation for overvalued stock."""
        current_price = 130.0
        fair_value = 100.0  # 30% overvalued
        peg_ratio = 2.5  # High PEG ratio
        num_models = 2
        
        recommendation, confidence = self.engine._generate_recommendation(
            current_price, fair_value, peg_ratio, num_models
        )
        
        assert recommendation == "SELL"
        assert 0 < confidence <= 1
    
    def test_generate_recommendation_hold(self):
        """Test recommendation generation for fairly valued stock."""
        current_price = 100.0
        fair_value = 105.0  # 5% overvalued (within hold range)
        peg_ratio = 1.2  # Fair PEG ratio
        num_models = 1
        
        recommendation, confidence = self.engine._generate_recommendation(
            current_price, fair_value, peg_ratio, num_models
        )
        
        assert recommendation == "HOLD"
        assert 0 < confidence <= 1
    
    def test_calculate_relative_valuation_metrics(self):
        """Test calculation of relative valuation metrics."""
        metrics = self.engine.calculate_relative_valuation_metrics(
            "AAPL", self.stock_info, self.income_statement, self.balance_sheet
        )
        
        assert isinstance(metrics, dict)
        assert 'pb_ratio' in metrics
        assert 'ps_ratio' in metrics
        assert 'ev_ebitda' in metrics
        assert 'peg_ratio' in metrics
        
        # Check that calculated values are reasonable
        if metrics['pb_ratio'] is not None:
            assert metrics['pb_ratio'] > 0
        if metrics['ps_ratio'] is not None:
            assert metrics['ps_ratio'] > 0
        if metrics['ev_ebitda'] is not None:
            assert metrics['ev_ebitda'] > 0
    
    def test_aggregate_fair_value_with_both_models(self):
        """Test fair value aggregation with both DCF and peer comparison."""
        dcf_value = 140.0
        peer_value = 160.0
        peg_ratio = 1.2
        
        result = self.engine._aggregate_fair_value(
            "AAPL", self.stock_info, dcf_value, peer_value, peg_ratio
        )
        
        assert isinstance(result, FairValueResult)
        assert result.dcf_value == dcf_value
        assert result.peer_comparison_value == peer_value
        # Average should be weighted (70% DCF, 30% peer)
        expected_avg = dcf_value * 0.7 + peer_value * 0.3
        assert abs(result.average_fair_value - expected_avg) < 1e-6
    
    def test_aggregate_fair_value_dcf_only(self):
        """Test fair value aggregation with only DCF model."""
        dcf_value = 140.0
        peer_value = None
        peg_ratio = 1.2
        
        result = self.engine._aggregate_fair_value(
            "AAPL", self.stock_info, dcf_value, peer_value, peg_ratio
        )
        
        assert result.dcf_value == dcf_value
        assert result.peer_comparison_value is None
        assert result.average_fair_value == dcf_value
    
    def test_aggregate_fair_value_no_models(self):
        """Test fair value aggregation with no working models."""
        dcf_value = None
        peer_value = None
        peg_ratio = None
        
        result = self.engine._aggregate_fair_value(
            "AAPL", self.stock_info, dcf_value, peer_value, peg_ratio
        )
        
        assert result.dcf_value is None
        assert result.peer_comparison_value is None
        # Should fallback to current price
        assert result.average_fair_value == self.stock_info.current_price
    
    def test_calculate_fair_value_with_invalid_data(self):
        """Test fair value calculation with invalid financial data."""
        # Create invalid data (empty DataFrames)
        empty_income = pd.DataFrame()
        empty_balance = pd.DataFrame()
        empty_cash_flow = pd.DataFrame()
        
        # Should handle empty data gracefully, not raise exception
        result = self.engine.calculate_fair_value(
            "INVALID", self.stock_info, empty_income, 
            empty_balance, empty_cash_flow, None
        )
        
        assert isinstance(result, FairValueResult)
        assert result.dcf_value is None  # DCF should fail with empty data
        assert result.peer_comparison_value is None  # No peer data provided
        # Should fallback to current price
        assert result.average_fair_value == self.stock_info.current_price
    
    def test_valuation_with_extreme_values(self):
        """Test valuation engine with extreme input values."""
        # Create stock with extreme values
        extreme_stock = StockInfo(
            "EXTREME", "Extreme Corp", 1000000.0, 1e15, 1000.0, 100.0, 0.5, 5.0, "Tech", "Software"
        )
        
        # Should handle extreme values gracefully
        result = self.engine.calculate_fair_value(
            "EXTREME", extreme_stock, self.income_statement, 
            self.balance_sheet, self.cash_flow, None
        )
        
        assert isinstance(result, FairValueResult)
        assert result.current_price == 1000000.0
        # Other values may be None due to extreme inputs, which is acceptable