"""Unit tests for data models."""

import pytest
from datetime import datetime
from stock_analysis.models.data_models import (
    StockInfo, LiquidityRatios, ProfitabilityRatios, LeverageRatios,
    EfficiencyRatios, FinancialRatios, HealthScore, FairValueResult,
    SentimentResult, AnalysisResult
)
from stock_analysis.utils.exceptions import ValidationError


class TestStockInfo:
    """Test cases for StockInfo data model."""
    
    def test_valid_stock_info(self):
        """Test that a valid StockInfo object passes validation."""
        stock_info = StockInfo(
            symbol="AAPL",
            company_name="Apple Inc.",
            current_price=150.0,
            market_cap=2500000000000.0,
            pe_ratio=25.0,
            pb_ratio=15.0,
            dividend_yield=0.005,
            beta=1.2
        )
        # Should not raise an exception
        stock_info.validate()
    
    def test_invalid_symbol(self):
        """Test that invalid symbols are rejected."""
        stock_info = StockInfo(
            symbol="",  # Empty symbol
            company_name="Apple Inc.",
            current_price=150.0,
            market_cap=2500000000000.0
        )
        with pytest.raises(ValidationError, match="Stock symbol cannot be empty"):
            stock_info.validate()
        
        stock_info = StockInfo(
            symbol="apple inc",  # Lowercase and space
            company_name="Apple Inc.",
            current_price=150.0,
            market_cap=2500000000000.0
        )
        with pytest.raises(ValidationError, match="Invalid stock symbol format"):
            stock_info.validate()
    
    def test_negative_price(self):
        """Test that negative prices are rejected."""
        stock_info = StockInfo(
            symbol="AAPL",
            company_name="Apple Inc.",
            current_price=-10.0,  # Negative price
            market_cap=2500000000000.0
        )
        with pytest.raises(ValidationError, match="Current price must be positive"):
            stock_info.validate()


class TestFinancialRatios:
    """Test cases for financial ratios data models."""
    
    def test_valid_liquidity_ratios(self):
        """Test that valid liquidity ratios pass validation."""
        ratios = LiquidityRatios(
            current_ratio=1.5,
            quick_ratio=1.2,
            cash_ratio=0.8
        )
        # Should not raise an exception
        ratios.validate()
    
    def test_negative_liquidity_ratios(self):
        """Test that negative liquidity ratios are rejected."""
        ratios = LiquidityRatios(
            current_ratio=-0.5,  # Negative ratio
            quick_ratio=1.2,
            cash_ratio=0.8
        )
        with pytest.raises(ValidationError, match="Current ratio cannot be negative"):
            ratios.validate()
    
    def test_valid_profitability_ratios(self):
        """Test that valid profitability ratios pass validation."""
        ratios = ProfitabilityRatios(
            gross_margin=0.4,
            operating_margin=0.2,
            net_profit_margin=0.15,
            return_on_assets=0.1,
            return_on_equity=0.2
        )
        # Should not raise an exception
        ratios.validate()
    
    def test_out_of_range_profitability_ratios(self):
        """Test that out-of-range profitability ratios are rejected."""
        ratios = ProfitabilityRatios(
            gross_margin=1.5,  # Out of range
            operating_margin=0.2,
            net_profit_margin=0.15
        )
        with pytest.raises(ValidationError, match="Gross margin should be between -1 and 1"):
            ratios.validate()
    
    def test_complete_financial_ratios(self):
        """Test that a complete set of financial ratios passes validation."""
        financial_ratios = FinancialRatios(
            liquidity_ratios=LiquidityRatios(current_ratio=1.5, quick_ratio=1.2),
            profitability_ratios=ProfitabilityRatios(
                gross_margin=0.4, 
                operating_margin=0.2, 
                net_profit_margin=0.15
            ),
            leverage_ratios=LeverageRatios(debt_to_equity=0.5, interest_coverage=5.0),
            efficiency_ratios=EfficiencyRatios(asset_turnover=0.7)
        )
        # Should not raise an exception
        financial_ratios.validate()


class TestHealthScore:
    """Test cases for HealthScore data model."""
    
    def test_valid_health_score(self):
        """Test that a valid health score passes validation."""
        health_score = HealthScore(
            overall_score=75.0,
            financial_strength=80.0,
            profitability_health=70.0,
            liquidity_health=65.0,
            risk_assessment="Low"
        )
        # Should not raise an exception
        health_score.validate()
    
    def test_out_of_range_score(self):
        """Test that out-of-range scores are rejected."""
        health_score = HealthScore(
            overall_score=105.0,  # Out of range
            financial_strength=80.0,
            profitability_health=70.0,
            liquidity_health=65.0,
            risk_assessment="Low"
        )
        with pytest.raises(ValidationError, match="overall_score must be between 0 and 100"):
            health_score.validate()
    
    def test_invalid_risk_assessment(self):
        """Test that invalid risk assessments are rejected."""
        health_score = HealthScore(
            overall_score=75.0,
            financial_strength=80.0,
            profitability_health=70.0,
            liquidity_health=65.0,
            risk_assessment="Very Low"  # Invalid value
        )
        with pytest.raises(ValidationError, match="Risk assessment must be one of"):
            health_score.validate()


class TestFairValueResult:
    """Test cases for FairValueResult data model."""
    
    def test_valid_fair_value(self):
        """Test that a valid fair value result passes validation."""
        fair_value = FairValueResult(
            current_price=150.0,
            dcf_value=165.0,
            peer_comparison_value=155.0,
            average_fair_value=160.0,
            recommendation="BUY",
            confidence_level=0.8
        )
        # Should not raise an exception
        fair_value.validate()
    
    def test_auto_calculate_average(self):
        """Test that average fair value is auto-calculated if not provided."""
        fair_value = FairValueResult(
            current_price=150.0,
            dcf_value=165.0,
            peer_comparison_value=155.0,
            # average_fair_value not provided
            recommendation="BUY",
            confidence_level=0.8
        )
        assert fair_value.average_fair_value == 160.0
    
    def test_invalid_recommendation(self):
        """Test that invalid recommendations are rejected."""
        fair_value = FairValueResult(
            current_price=150.0,
            dcf_value=165.0,
            average_fair_value=165.0,
            recommendation="STRONG BUY",  # Invalid value
            confidence_level=0.8
        )
        with pytest.raises(ValidationError, match="Recommendation must be one of"):
            fair_value.validate()
    
    def test_out_of_range_confidence(self):
        """Test that out-of-range confidence levels are rejected."""
        fair_value = FairValueResult(
            current_price=150.0,
            dcf_value=165.0,
            average_fair_value=165.0,
            recommendation="BUY",
            confidence_level=1.5  # Out of range
        )
        with pytest.raises(ValidationError, match="Confidence level must be between 0 and 1"):
            fair_value.validate()


class TestSentimentResult:
    """Test cases for SentimentResult data model."""
    
    def test_valid_sentiment(self):
        """Test that a valid sentiment result passes validation."""
        sentiment = SentimentResult(
            overall_sentiment=0.6,
            positive_count=15,
            negative_count=5,
            neutral_count=10,
            key_themes=["Earnings", "Product Launch", "Market Expansion"],
            sentiment_trend=[0.2, 0.3, 0.5, 0.6]
        )
        # Should not raise an exception
        sentiment.validate()
    
    def test_out_of_range_sentiment(self):
        """Test that out-of-range sentiment values are rejected."""
        sentiment = SentimentResult(
            overall_sentiment=1.5,  # Out of range
            positive_count=15,
            negative_count=5,
            neutral_count=10,
            key_themes=["Earnings", "Product Launch"]
        )
        with pytest.raises(ValidationError, match="Overall sentiment must be between -1 and 1"):
            sentiment.validate()
    
    def test_empty_themes(self):
        """Test that empty themes list is rejected."""
        sentiment = SentimentResult(
            overall_sentiment=0.6,
            positive_count=15,
            negative_count=5,
            neutral_count=10,
            key_themes=[]  # Empty list
        )
        with pytest.raises(ValidationError, match="Key themes list cannot be empty"):
            sentiment.validate()


class TestAnalysisResult:
    """Test cases for the complete AnalysisResult data model."""
    
    def test_valid_analysis_result(self):
        """Test that a valid complete analysis result passes validation."""
        # Create all required components
        stock_info = StockInfo(
            symbol="AAPL",
            company_name="Apple Inc.",
            current_price=150.0,
            market_cap=2500000000000.0
        )
        
        financial_ratios = FinancialRatios(
            liquidity_ratios=LiquidityRatios(current_ratio=1.5),
            profitability_ratios=ProfitabilityRatios(gross_margin=0.4),
            leverage_ratios=LeverageRatios(debt_to_equity=0.5),
            efficiency_ratios=EfficiencyRatios(asset_turnover=0.7)
        )
        
        health_score = HealthScore(
            overall_score=75.0,
            financial_strength=80.0,
            profitability_health=70.0,
            liquidity_health=65.0,
            risk_assessment="Low"
        )
        
        fair_value = FairValueResult(
            current_price=150.0,
            dcf_value=165.0,
            average_fair_value=165.0,
            recommendation="BUY",
            confidence_level=0.8
        )
        
        sentiment = SentimentResult(
            overall_sentiment=0.6,
            positive_count=15,
            negative_count=5,
            neutral_count=10,
            key_themes=["Earnings", "Product Launch"]
        )
        
        # Create the complete analysis result
        analysis_result = AnalysisResult(
            symbol="AAPL",
            timestamp=datetime.now(),
            stock_info=stock_info,
            financial_ratios=financial_ratios,
            health_score=health_score,
            fair_value=fair_value,
            sentiment=sentiment,
            recommendations=["Consider buying based on undervaluation", "Strong financial health"]
        )
        
        # Should not raise an exception
        analysis_result.validate()
    
    def test_empty_recommendations(self):
        """Test that empty recommendations list is rejected."""
        # Create minimal components for testing
        stock_info = StockInfo(
            symbol="AAPL",
            company_name="Apple Inc.",
            current_price=150.0,
            market_cap=2500000000000.0
        )
        
        financial_ratios = FinancialRatios(
            liquidity_ratios=LiquidityRatios(current_ratio=1.5),
            profitability_ratios=ProfitabilityRatios(gross_margin=0.4),
            leverage_ratios=LeverageRatios(debt_to_equity=0.5),
            efficiency_ratios=EfficiencyRatios(asset_turnover=0.7)
        )
        
        health_score = HealthScore(
            overall_score=75.0,
            financial_strength=80.0,
            profitability_health=70.0,
            liquidity_health=65.0,
            risk_assessment="Low"
        )
        
        fair_value = FairValueResult(
            current_price=150.0,
            dcf_value=165.0,
            average_fair_value=165.0,
            recommendation="BUY",
            confidence_level=0.8
        )
        
        sentiment = SentimentResult(
            overall_sentiment=0.6,
            positive_count=15,
            negative_count=5,
            neutral_count=10,
            key_themes=["Earnings", "Product Launch"]
        )
        
        # Create analysis result with empty recommendations
        analysis_result = AnalysisResult(
            symbol="AAPL",
            timestamp=datetime.now(),
            stock_info=stock_info,
            financial_ratios=financial_ratios,
            health_score=health_score,
            fair_value=fair_value,
            sentiment=sentiment,
            recommendations=[]  # Empty list
        )
        
        with pytest.raises(ValidationError, match="Recommendations list cannot be empty"):
            analysis_result.validate()