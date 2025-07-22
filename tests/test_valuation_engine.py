"""Unit tests for valuation engine."""

import pytest
import pandas as pd
from datetime import datetime

from stock_analysis.analyzers.valuation_engine import ValuationEngine
from stock_analysis.models.data_models import StockInfo, FairValueResult


@pytest.fixture
def engine():
    """Create a valuation engine instance."""
    return ValuationEngine()


@pytest.fixture
def stock_info():
    """Create sample stock info."""
    return StockInfo(
        symbol="AAPL",
        name="Apple Inc.",
        current_price=150.0,
        market_cap=2500000000000.0,
        beta=1.2,
        company_name="Apple Inc.",
        pe_ratio=25.0,
        pb_ratio=15.0,
        dividend_yield=0.005,
        sector="Technology",
        industry="Consumer Electronics"
    )


@pytest.fixture
def income_statement():
    """Create sample income statement."""
    dates = [datetime(2020, 12, 31), datetime(2021, 12, 31), datetime(2022, 12, 31)]
    return pd.DataFrame({
        'Revenue': [100000000, 120000000, 150000000],
        'NetIncome': [20000000, 25000000, 30000000],
        'OperatingIncome': [30000000, 35000000, 40000000],
        'GrossProfit': [50000000, 60000000, 75000000]
    }, index=dates)


@pytest.fixture
def balance_sheet():
    """Create sample balance sheet."""
    dates = [datetime(2020, 12, 31), datetime(2021, 12, 31), datetime(2022, 12, 31)]
    return pd.DataFrame({
        'TotalAssets': [200000000, 250000000, 300000000],
        'TotalLiabilities': [100000000, 120000000, 140000000],
        'TotalEquity': [100000000, 130000000, 160000000],
        'Cash': [30000000, 40000000, 50000000],
        'CommonStockSharesOutstanding': [1000000, 1000000, 1000000]
    }, index=dates)


@pytest.fixture
def cash_flow():
    """Create sample cash flow statement."""
    dates = [datetime(2020, 12, 31), datetime(2021, 12, 31), datetime(2022, 12, 31)]
    return pd.DataFrame({
        'FreeCashFlow': [15000000, 20000000, 25000000],
        'OperatingCashFlow': [25000000, 30000000, 35000000],
        'CapitalExpenditures': [-13000000, -14000000, -15000000],
        'DividendsPaid': [-6500000, -6750000, -7000000]
    }, index=dates)


class TestValuationEngine:
    """Test cases for ValuationEngine."""
    
    def test_calculate_fair_value_success(self, engine, stock_info, income_statement, balance_sheet, cash_flow):
        """Test successful fair value calculation."""
        result = engine.calculate_fair_value(
            "AAPL", stock_info, income_statement, balance_sheet, cash_flow
        )
        
        assert isinstance(result, FairValueResult)
        assert result.current_price == 150.0
        assert result.dcf_value is not None
        assert result.peer_comparison_value is not None
        assert result.average_fair_value is not None
        assert result.recommendation in ["BUY", "SELL", "HOLD"]
        assert 0.0 <= result.confidence_level <= 1.0
    
    def test_calculate_dcf_value(self, engine, stock_info, income_statement, balance_sheet, cash_flow):
        """Test DCF value calculation."""
        dcf_value = engine.calculate_dcf_value(
            "AAPL", stock_info, income_statement, balance_sheet, cash_flow
        )
        
        assert dcf_value is not None
        assert dcf_value > 0
    
    def test_calculate_peer_comparison_value(self, engine, stock_info):
        """Test peer comparison value calculation."""
        peer_data = [
            {
                'symbol': 'MSFT',
                'pe_ratio': 30.0,
                'pb_ratio': 12.0,
                'market_cap': 2000000000000.0
            },
            {
                'symbol': 'GOOGL',
                'pe_ratio': 28.0,
                'pb_ratio': 10.0,
                'market_cap': 1800000000000.0
            }
        ]
        
        peer_value = engine.calculate_peer_comparison_value(
            "AAPL", stock_info, peer_data
        )
        
        assert peer_value is not None
        assert peer_value > 0
    
    def test_calculate_peg_ratio(self, engine, stock_info, income_statement):
        """Test PEG ratio calculation."""
        peg_ratio = engine.calculate_peg_ratio(
            "AAPL", stock_info, income_statement
        )
        
        assert peg_ratio is not None
        assert peg_ratio > 0
    
    def test_extract_free_cash_flows(self, engine, cash_flow):
        """Test extracting free cash flows."""
        free_cash_flows = engine._extract_free_cash_flows(cash_flow)
        
        assert len(free_cash_flows) == 3
        assert all(fcf > 0 for fcf in free_cash_flows)
        assert free_cash_flows[0] == 15000000
        assert free_cash_flows[1] == 20000000
        assert free_cash_flows[2] == 25000000
    
    def test_calculate_earnings_growth_rate(self, engine, income_statement):
        """Test earnings growth rate calculation."""
        growth_rate = engine._calculate_earnings_growth_rate(income_statement)
        
        assert growth_rate is not None
        assert growth_rate > 0
    
    def test_calculate_relative_valuation_metrics(self, engine, stock_info, income_statement, balance_sheet):
        """Test relative valuation metrics calculation."""
        metrics = engine.calculate_relative_valuation_metrics(
            "AAPL", stock_info, income_statement, balance_sheet
        )
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        assert 'pe_ratio' in metrics
        assert 'pb_ratio' in metrics
        assert 'ev_ebitda' in metrics
        assert 'roe' in metrics
    
    def test_calculate_discount_rate(self, engine, stock_info, balance_sheet):
        """Test discount rate calculation."""
        discount_rate = engine.calculate_discount_rate(stock_info, balance_sheet)
        
        assert discount_rate is not None
        assert 0.05 <= discount_rate <= 0.15  # Typical range for discount rate
    
    def test_calculate_terminal_value(self, engine, cash_flow):
        """Test terminal value calculation."""
        terminal_value = engine.calculate_terminal_value(
            cash_flow['FreeCashFlow'].iloc[-1],
            0.03,  # growth rate
            0.10   # discount rate
        )
        
        assert terminal_value is not None
        assert terminal_value > 0
    
    def test_generate_recommendation(self, engine):
        """Test recommendation generation."""
        recommendation = engine.generate_recommendation(
            current_price=150.0,
            fair_value=180.0,
            upside_threshold=0.15,
            downside_threshold=0.15
        )
        
        assert recommendation in ["BUY", "SELL", "HOLD"]
    
    def test_calculate_confidence_level(self, engine):
        """Test confidence level calculation."""
        confidence = engine.calculate_confidence_level(
            metrics_quality=0.8,
            data_completeness=0.9,
            growth_stability=0.7
        )
        
        assert 0.0 <= confidence <= 1.0