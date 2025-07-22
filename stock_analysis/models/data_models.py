"""Data models for stock analysis system.

This module contains the core data models used throughout the stock analysis system,
including validation methods and related utilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union
import re

from stock_analysis.utils.exceptions import ValidationError


@dataclass
class StockInfo:
    """Stock information data model.
    
    Contains basic information about a stock including price, ratios, and market data.
    """
    symbol: str
    company_name: str
    current_price: float
    market_cap: float
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    
    def validate(self) -> None:
        """Validate the stock information data.
        
        Raises:
            ValidationError: If any validation checks fail.
        """
        if not self.symbol:
            raise ValidationError("Stock symbol cannot be empty")
        
        if not self.company_name:
            raise ValidationError("Company name cannot be empty")
        
        if not isinstance(self.symbol, str) or not re.match(r'^[A-Z0-9.^-]{1,10}$', self.symbol):
            raise ValidationError(f"Invalid stock symbol format: {self.symbol}")
        
        if self.current_price <= 0:
            raise ValidationError(f"Current price must be positive: {self.current_price}")
        
        if self.market_cap <= 0:
            raise ValidationError(f"Market cap must be positive: {self.market_cap}")
        
        if self.pe_ratio is not None and self.pe_ratio < 0:
            raise ValidationError(f"P/E ratio cannot be negative: {self.pe_ratio}")
        
        if self.pb_ratio is not None and self.pb_ratio < 0:
            raise ValidationError(f"P/B ratio cannot be negative: {self.pb_ratio}")
        
        if self.dividend_yield is not None and self.dividend_yield < 0:
            raise ValidationError(f"Dividend yield cannot be negative: {self.dividend_yield}")


@dataclass
class LiquidityRatios:
    """Liquidity ratios data model.
    
    Contains ratios that measure a company's ability to pay short-term obligations.
    """
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    cash_ratio: Optional[float] = None
    
    def validate(self) -> None:
        """Validate liquidity ratios.
        
        Raises:
            ValidationError: If any validation checks fail.
        """
        if self.current_ratio is not None and self.current_ratio < 0:
            raise ValidationError(f"Current ratio cannot be negative: {self.current_ratio}")
        
        if self.quick_ratio is not None and self.quick_ratio < 0:
            raise ValidationError(f"Quick ratio cannot be negative: {self.quick_ratio}")
        
        if self.cash_ratio is not None and self.cash_ratio < 0:
            raise ValidationError(f"Cash ratio cannot be negative: {self.cash_ratio}")


@dataclass
class ProfitabilityRatios:
    """Profitability ratios data model.
    
    Contains ratios that measure a company's ability to generate earnings.
    """
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_profit_margin: Optional[float] = None
    return_on_assets: Optional[float] = None
    return_on_equity: Optional[float] = None
    
    def validate(self) -> None:
        """Validate profitability ratios.
        
        Raises:
            ValidationError: If any validation checks fail.
        """
        # These ratios can be negative in real-world scenarios
        # We use more realistic bounds to catch obvious data errors while allowing extreme values
        
        if self.gross_margin is not None and (self.gross_margin < -10 or self.gross_margin > 1):
            raise ValidationError(f"Gross margin should be between -10 and 1: {self.gross_margin}")
        
        if self.operating_margin is not None and (self.operating_margin < -10 or self.operating_margin > 1):
            raise ValidationError(f"Operating margin should be between -10 and 1: {self.operating_margin}")
        
        if self.net_profit_margin is not None and (self.net_profit_margin < -10 or self.net_profit_margin > 1):
            raise ValidationError(f"Net profit margin should be between -10 and 1: {self.net_profit_margin}")
        
        if self.return_on_assets is not None and (self.return_on_assets < -5 or self.return_on_assets > 5):
            raise ValidationError(f"Return on assets should be between -5 and 5: {self.return_on_assets}")
        
        if self.return_on_equity is not None and (self.return_on_equity < -10 or self.return_on_equity > 10):
            raise ValidationError(f"Return on equity should be between -10 and 10: {self.return_on_equity}")


@dataclass
class LeverageRatios:
    """Leverage ratios data model.
    
    Contains ratios that measure a company's debt levels and ability to meet financial obligations.
    """
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    interest_coverage: Optional[float] = None
    
    def validate(self) -> None:
        """Validate leverage ratios.
        
        Raises:
            ValidationError: If any validation checks fail.
        """
        if self.debt_to_equity is not None and self.debt_to_equity < 0:
            raise ValidationError(f"Debt to equity ratio cannot be negative: {self.debt_to_equity}")
        
        if self.debt_to_assets is not None and (self.debt_to_assets < 0 or self.debt_to_assets > 1):
            raise ValidationError(f"Debt to assets ratio should be between 0 and 1: {self.debt_to_assets}")
        
        # Interest coverage can be negative if a company is losing money
        # but extremely negative values might indicate data issues
        if self.interest_coverage is not None and self.interest_coverage < -100:
            raise ValidationError(f"Interest coverage ratio is suspiciously low: {self.interest_coverage}")


@dataclass
class EfficiencyRatios:
    """Efficiency ratios data model.
    
    Contains ratios that measure how efficiently a company uses its assets and manages its operations.
    """
    asset_turnover: Optional[float] = None
    inventory_turnover: Optional[float] = None
    receivables_turnover: Optional[float] = None
    
    def validate(self) -> None:
        """Validate efficiency ratios.
        
        Raises:
            ValidationError: If any validation checks fail.
        """
        if self.asset_turnover is not None and self.asset_turnover < 0:
            raise ValidationError(f"Asset turnover cannot be negative: {self.asset_turnover}")
        
        if self.inventory_turnover is not None and self.inventory_turnover < 0:
            raise ValidationError(f"Inventory turnover cannot be negative: {self.inventory_turnover}")
        
        if self.receivables_turnover is not None and self.receivables_turnover < 0:
            raise ValidationError(f"Receivables turnover cannot be negative: {self.receivables_turnover}")


@dataclass
class FinancialRatios:
    """Financial ratios data model.
    
    Contains all financial ratio categories for comprehensive financial analysis.
    """
    liquidity_ratios: LiquidityRatios
    profitability_ratios: ProfitabilityRatios
    leverage_ratios: LeverageRatios
    efficiency_ratios: EfficiencyRatios
    
    def validate(self) -> None:
        """Validate all financial ratios.
        
        Raises:
            ValidationError: If any validation checks fail.
        """
        self.liquidity_ratios.validate()
        self.profitability_ratios.validate()
        self.leverage_ratios.validate()
        self.efficiency_ratios.validate()


@dataclass
class HealthScore:
    """Company health score data model.
    
    Contains scores and assessments of a company's financial health.
    """
    overall_score: float  # 0-100
    financial_strength: float  # 0-100
    profitability_health: float  # 0-100
    liquidity_health: float  # 0-100
    risk_assessment: str  # Low, Medium, High
    
    def validate(self) -> None:
        """Validate health score data.
        
        Raises:
            ValidationError: If any validation checks fail.
        """
        for score_name, score in [
            ("overall_score", self.overall_score),
            ("financial_strength", self.financial_strength),
            ("profitability_health", self.profitability_health),
            ("liquidity_health", self.liquidity_health)
        ]:
            if not 0 <= score <= 100:
                raise ValidationError(f"{score_name} must be between 0 and 100: {score}")
        
        valid_risk_levels = ["Low", "Medium", "High"]
        if self.risk_assessment not in valid_risk_levels:
            raise ValidationError(
                f"Risk assessment must be one of {valid_risk_levels}: {self.risk_assessment}"
            )


@dataclass
class FairValueResult:
    """Fair value analysis result data model.
    
    Contains valuation results from different models and the final recommendation.
    """
    current_price: float
    dcf_value: Optional[float] = None
    peer_comparison_value: Optional[float] = None
    average_fair_value: float = 0.0
    recommendation: str = "HOLD"  # BUY, HOLD, SELL
    confidence_level: float = 0.0  # 0-1
    
    def __post_init__(self):
        """Calculate average fair value if not provided."""
        if self.average_fair_value == 0.0 and (self.dcf_value is not None or self.peer_comparison_value is not None):
            values = [v for v in [self.dcf_value, self.peer_comparison_value] if v is not None]
            if values:
                self.average_fair_value = sum(values) / len(values)
    
    def validate(self) -> None:
        """Validate fair value result data.
        
        Raises:
            ValidationError: If any validation checks fail.
        """
        if self.current_price <= 0:
            raise ValidationError(f"Current price must be positive: {self.current_price}")
        
        # DCF values can be negative for companies with negative cash flows
        # Only validate that they're not extremely unrealistic (e.g., more negative than -1000)
        if self.dcf_value is not None and self.dcf_value < -1000:
            raise ValidationError(f"DCF value is unrealistically negative: {self.dcf_value}")
        
        if self.peer_comparison_value is not None and self.peer_comparison_value <= 0:
            raise ValidationError(f"Peer comparison value must be positive: {self.peer_comparison_value}")
        
        # Average fair value can be negative if all models produce negative values
        # Only validate against extremely unrealistic values
        if self.average_fair_value < -1000:
            raise ValidationError(f"Average fair value is unrealistically negative: {self.average_fair_value}")
        
        valid_recommendations = ["BUY", "HOLD", "SELL"]
        if self.recommendation not in valid_recommendations:
            raise ValidationError(
                f"Recommendation must be one of {valid_recommendations}: {self.recommendation}"
            )
        
        if not 0 <= self.confidence_level <= 1:
            raise ValidationError(f"Confidence level must be between 0 and 1: {self.confidence_level}")


@dataclass
class SentimentResult:
    """News sentiment analysis result data model.
    
    Contains sentiment scores, trends, and key themes from news analysis.
    """
    overall_sentiment: float  # -1 to 1
    positive_count: int
    negative_count: int
    neutral_count: int
    key_themes: List[str]
    sentiment_trend: List[float] = field(default_factory=list)
    
    def validate(self) -> None:
        """Validate sentiment result data.
        
        Raises:
            ValidationError: If any validation checks fail.
        """
        if not -1 <= self.overall_sentiment <= 1:
            raise ValidationError(f"Overall sentiment must be between -1 and 1: {self.overall_sentiment}")
        
        if self.positive_count < 0:
            raise ValidationError(f"Positive count cannot be negative: {self.positive_count}")
        
        if self.negative_count < 0:
            raise ValidationError(f"Negative count cannot be negative: {self.negative_count}")
        
        if self.neutral_count < 0:
            raise ValidationError(f"Neutral count cannot be negative: {self.neutral_count}")
        
        if not self.key_themes:
            raise ValidationError("Key themes list cannot be empty")
        
        for trend in self.sentiment_trend:
            if not -1 <= trend <= 1:
                raise ValidationError(f"Sentiment trend values must be between -1 and 1: {trend}")


@dataclass
class AnalysisResult:
    """Complete stock analysis result data model.
    
    Contains all analysis components for a stock.
    """
    symbol: str
    timestamp: datetime
    stock_info: StockInfo
    financial_ratios: FinancialRatios
    health_score: HealthScore
    fair_value: FairValueResult
    sentiment: SentimentResult
    recommendations: List[str]
    
    def validate(self) -> None:
        """Validate the complete analysis result.
        
        Raises:
            ValidationError: If any validation checks fail.
        """
        if not self.symbol:
            raise ValidationError("Stock symbol cannot be empty")
        
        if not isinstance(self.timestamp, datetime):
            raise ValidationError(f"Timestamp must be a datetime object: {self.timestamp}")
        
        if not self.recommendations:
            raise ValidationError("Recommendations list cannot be empty")
        
        # Validate all component models
        self.stock_info.validate()
        self.financial_ratios.validate()
        self.health_score.validate()
        self.fair_value.validate()
        self.sentiment.validate()