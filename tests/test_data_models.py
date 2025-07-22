"""Unit tests for data models."""

import pytest
from datetime import datetime
from stock_analysis.models.data_models import (
    StockInfo, ETFInfo, SecurityInfo, LiquidityRatios, ProfitabilityRatios, LeverageRatios,
    EfficiencyRatios, FinancialRatios, HealthScore, FairValueResult,
    SentimentResult, AnalysisResult
)
from stock_analysis.utils.exceptions import ValidationError


class TestSecurityInfo:
    """Test cases for SecurityInfo base class."""
    
    def test_valid_security_info(self):
        """Test that a valid SecurityInfo object passes validation."""
        security_info = SecurityInfo(
            symbol="AAPL",
            name="Apple Inc.",
            current_price=150.0,
            market_cap=2500000000000.0,
            beta=1.2
        )
        
        assert security_info.symbol == "AAPL"
        assert security_info.name == "Apple Inc."
        assert security_info.current_price == 150.0
        assert security_info.market_cap == 2500000000000.0
        assert security_info.beta == 1.2
    
    def test_invalid_symbol(self):
        """Test validation of invalid symbol."""
        with pytest.raises(ValidationError, match="Symbol must be a non-empty string"):
            SecurityInfo(
                symbol="",  # Invalid: empty string
                name="Apple Inc.",
                current_price=150.0,
                market_cap=2500000000000.0,
                beta=1.2
            )
    
    def test_negative_price(self):
        """Test validation of negative price."""
        with pytest.raises(ValidationError, match="Current price must be positive"):
            SecurityInfo(
                symbol="AAPL",
                name="Apple Inc.",
                current_price=-150.0,  # Invalid: negative price
                market_cap=2500000000000.0,
                beta=1.2
            )


class TestStockInfo:
    """Test cases for StockInfo class."""
    
    def test_valid_stock_info(self):
        """Test that a valid StockInfo object passes validation."""
        stock_info = StockInfo(
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
        
        assert stock_info.symbol == "AAPL"
        assert stock_info.company_name == "Apple Inc."
        assert stock_info.pe_ratio == 25.0
        assert stock_info.pb_ratio == 15.0
        assert stock_info.dividend_yield == 0.005
        assert stock_info.sector == "Technology"
        assert stock_info.industry == "Consumer Electronics"
    
    def test_negative_ratios(self):
        """Test validation of negative ratios."""
        with pytest.raises(ValidationError, match="PE ratio must be positive"):
            StockInfo(
                symbol="AAPL",
                name="Apple Inc.",
                current_price=150.0,
                market_cap=2500000000000.0,
                beta=1.2,
                company_name="Apple Inc.",
                pe_ratio=-25.0,  # Invalid: negative PE ratio
                pb_ratio=15.0,
                dividend_yield=0.005,
                sector="Technology",
                industry="Consumer Electronics"
            )


class TestETFInfo:
    """Test cases for ETFInfo class."""
    
    def test_valid_etf_info(self):
        """Test that a valid ETFInfo object passes validation."""
        etf_info = ETFInfo(
            symbol="SPY",
            name="SPDR S&P 500 ETF Trust",
            current_price=450.0,
            market_cap=450000000000.0,
            beta=1.0,
            expense_ratio=0.0095,
            assets_under_management=450000000000.0,
            nav=450.0,
            category="Large Blend",
            fund_family="State Street Global Advisors",
            dividend_yield=0.015,
            holdings=[
                {"symbol": "AAPL", "name": "Apple Inc.", "weight": 0.07},
                {"symbol": "MSFT", "name": "Microsoft Corp", "weight": 0.06}
            ],
            asset_allocation={
                "stocks": 0.98,
                "bonds": 0.0,
                "cash": 0.02
            }
        )
        
        assert etf_info.symbol == "SPY"
        assert etf_info.expense_ratio == 0.0095
        assert etf_info.assets_under_management == 450000000000.0
        assert etf_info.nav == 450.0
        assert etf_info.category == "Large Blend"
        assert len(etf_info.holdings) == 2
        assert etf_info.holdings[0]["symbol"] == "AAPL"
        assert etf_info.asset_allocation["stocks"] == 0.98
    
    def test_negative_expense_ratio(self):
        """Test validation of negative expense ratio."""
        with pytest.raises(ValidationError, match="Expense ratio must be positive"):
            ETFInfo(
                symbol="SPY",
                name="SPDR S&P 500 ETF Trust",
                current_price=450.0,
                market_cap=450000000000.0,
                beta=1.0,
                expense_ratio=-0.0095,  # Invalid: negative expense ratio
                assets_under_management=450000000000.0,
                nav=450.0,
                category="Large Blend",
                fund_family="State Street Global Advisors",
                dividend_yield=0.015,
                holdings=[],
                asset_allocation={"stocks": 1.0}
            )
    
    def test_invalid_asset_allocation(self):
        """Test validation of invalid asset allocation."""
        with pytest.raises(ValidationError, match="Asset allocation percentages must sum to 1.0"):
            ETFInfo(
                symbol="SPY",
                name="SPDR S&P 500 ETF Trust",
                current_price=450.0,
                market_cap=450000000000.0,
                beta=1.0,
                expense_ratio=0.0095,
                assets_under_management=450000000000.0,
                nav=450.0,
                category="Large Blend",
                fund_family="State Street Global Advisors",
                dividend_yield=0.015,
                holdings=[],
                asset_allocation={  # Invalid: percentages sum to > 1.0
                    "stocks": 0.8,
                    "bonds": 0.3,
                    "cash": 0.2
                }
            )
    
    def test_invalid_holdings_format(self):
        """Test validation of invalid holdings format."""
        with pytest.raises(ValidationError, match="Holdings must contain symbol, name, and weight"):
            ETFInfo(
                symbol="SPY",
                name="SPDR S&P 500 ETF Trust",
                current_price=450.0,
                market_cap=450000000000.0,
                beta=1.0,
                expense_ratio=0.0095,
                assets_under_management=450000000000.0,
                nav=450.0,
                category="Large Blend",
                fund_family="State Street Global Advisors",
                dividend_yield=0.015,
                holdings=[  # Invalid: missing required fields
                    {"symbol": "AAPL"}
                ],
                asset_allocation={"stocks": 1.0}
            )


class TestFinancialRatios:
    """Test cases for financial ratio classes."""
    
    def test_valid_financial_ratios(self):
        """Test that valid financial ratios pass validation."""
        liquidity = LiquidityRatios(
            current_ratio=2.0,
            quick_ratio=1.5,
            cash_ratio=0.8
        )
        
        profitability = ProfitabilityRatios(
            gross_margin=0.4,
            operating_margin=0.2,
            net_margin=0.15,
            roe=0.25,
            roa=0.12
        )
        
        leverage = LeverageRatios(
            debt_to_equity=1.5,
            debt_to_assets=0.6,
            interest_coverage=8.0
        )
        
        efficiency = EfficiencyRatios(
            asset_turnover=1.2,
            inventory_turnover=8.0,
            receivables_turnover=12.0
        )
        
        ratios = FinancialRatios(
            liquidity=liquidity,
            profitability=profitability,
            leverage=leverage,
            efficiency=efficiency
        )
        
        assert ratios.liquidity.current_ratio == 2.0
        assert ratios.profitability.net_margin == 0.15
        assert ratios.leverage.debt_to_equity == 1.5
        assert ratios.efficiency.asset_turnover == 1.2
    
    def test_negative_liquidity_ratios(self):
        """Test validation of negative liquidity ratios."""
        with pytest.raises(ValidationError, match="Current ratio must be positive"):
            LiquidityRatios(
                current_ratio=-2.0,  # Invalid: negative ratio
                quick_ratio=1.5,
                cash_ratio=0.8
            )
    
    def test_out_of_range_profitability_ratios(self):
        """Test validation of out-of-range profitability ratios."""
        with pytest.raises(ValidationError, match="Margin ratios must be between -1 and 1"):
            ProfitabilityRatios(
                gross_margin=1.5,  # Invalid: > 1.0
                operating_margin=0.2,
                net_margin=0.15,
                roe=0.25,
                roa=0.12
            )


class TestHealthScore:
    """Test cases for HealthScore class."""
    
    def test_valid_health_score(self):
        """Test that a valid health score passes validation."""
        score = HealthScore(
            overall_score=85.0,
            liquidity_score=90.0,
            profitability_score=80.0,
            leverage_score=75.0,
            efficiency_score=85.0,
            risk_assessment="LOW",
            risk_factors=["High cash position", "Strong balance sheet"]
        )
        
        assert score.overall_score == 85.0
        assert score.liquidity_score == 90.0
        assert score.risk_assessment == "LOW"
        assert len(score.risk_factors) == 2
    
    def test_out_of_range_score(self):
        """Test validation of out-of-range scores."""
        with pytest.raises(ValidationError, match="Score must be between 0 and 100"):
            HealthScore(
                overall_score=150.0,  # Invalid: > 100
                liquidity_score=90.0,
                profitability_score=80.0,
                leverage_score=75.0,
                efficiency_score=85.0,
                risk_assessment="LOW",
                risk_factors=[]
            )
    
    def test_invalid_risk_assessment(self):
        """Test validation of invalid risk assessment."""
        with pytest.raises(ValidationError, match="Risk assessment must be one of"):
            HealthScore(
                overall_score=85.0,
                liquidity_score=90.0,
                profitability_score=80.0,
                leverage_score=75.0,
                efficiency_score=85.0,
                risk_assessment="INVALID",  # Invalid: not in allowed values
                risk_factors=[]
            )


class TestFairValueResult:
    """Test cases for FairValueResult class."""
    
    def test_valid_fair_value_result(self):
        """Test that a valid fair value result passes validation."""
        result = FairValueResult(
            current_price=150.0,
            dcf_value=180.0,
            peer_comparison_value=170.0,
            average_fair_value=175.0,
            recommendation="BUY",
            confidence_level=0.8
        )
        
        assert result.current_price == 150.0
        assert result.dcf_value == 180.0
        assert result.peer_comparison_value == 170.0
        assert result.average_fair_value == 175.0
        assert result.recommendation == "BUY"
        assert result.confidence_level == 0.8
    
    def test_invalid_recommendation(self):
        """Test validation of invalid recommendation."""
        with pytest.raises(ValidationError, match="Recommendation must be one of"):
            FairValueResult(
                current_price=150.0,
                dcf_value=180.0,
                peer_comparison_value=170.0,
                average_fair_value=175.0,
                recommendation="INVALID",  # Invalid: not in allowed values
                confidence_level=0.8
            )
    
    def test_out_of_range_confidence(self):
        """Test validation of out-of-range confidence level."""
        with pytest.raises(ValidationError, match="Confidence level must be between 0 and 1"):
            FairValueResult(
                current_price=150.0,
                dcf_value=180.0,
                peer_comparison_value=170.0,
                average_fair_value=175.0,
                recommendation="BUY",
                confidence_level=1.5  # Invalid: > 1.0
            )


class TestSentimentResult:
    """Test cases for SentimentResult class."""
    
    def test_valid_sentiment_result(self):
        """Test that a valid sentiment result passes validation."""
        result = SentimentResult(
            overall_sentiment=0.75,
            sentiment_scores={
                "news": 0.8,
                "social_media": 0.7
            },
            sentiment_trends=[
                {"date": datetime.now(), "score": 0.75}
            ],
            key_themes=["Earnings", "Growth"],
            source_breakdown={
                "news": 0.6,
                "social_media": 0.4
            }
        )
        
        assert result.overall_sentiment == 0.75
        assert result.sentiment_scores["news"] == 0.8
        assert len(result.sentiment_trends) == 1
        assert len(result.key_themes) == 2
    
    def test_out_of_range_sentiment(self):
        """Test validation of out-of-range sentiment scores."""
        with pytest.raises(ValidationError, match="Sentiment scores must be between -1 and 1"):
            SentimentResult(
                overall_sentiment=1.5,  # Invalid: > 1.0
                sentiment_scores={},
                sentiment_trends=[],
                key_themes=[],
                source_breakdown={}
            )
    
    def test_empty_themes(self):
        """Test validation of empty themes list."""
        with pytest.raises(ValidationError, match="Must provide at least one key theme"):
            SentimentResult(
                overall_sentiment=0.75,
                sentiment_scores={},
                sentiment_trends=[],
                key_themes=[],  # Invalid: empty list
                source_breakdown={}
            )


class TestAnalysisResult:
    """Test cases for AnalysisResult class."""
    
    def test_valid_analysis_result(self):
        """Test that a valid analysis result passes validation."""
        result = AnalysisResult(
            total_stocks=10,
            successful_analyses=8,
            failed_analyses=2,
            execution_time=120.5,
            success_rate=80.0,
            failed_symbols=["ABC", "XYZ"],
            error_summary={"API_ERROR": 2},
            results=[
                {
                    "symbol": "AAPL",
                    "recommendation": "BUY",
                    "confidence": 0.8
                }
            ]
        )
        
        assert result.total_stocks == 10
        assert result.successful_analyses == 8
        assert result.failed_analyses == 2
        assert result.success_rate == 80.0
        assert len(result.failed_symbols) == 2
        assert len(result.results) == 1
    
    def test_empty_recommendations(self):
        """Test validation of empty recommendations list."""
        with pytest.raises(ValidationError, match="Must provide at least one analysis result"):
            AnalysisResult(
                total_stocks=10,
                successful_analyses=8,
                failed_analyses=2,
                execution_time=120.5,
                success_rate=80.0,
                failed_symbols=["ABC", "XYZ"],
                error_summary={"API_ERROR": 2},
                results=[]  # Invalid: empty list
            )