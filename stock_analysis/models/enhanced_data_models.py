"""Enhanced data models for stock analysis system.

This module contains enhanced data models that extend the core data models
to provide more comprehensive financial data and analysis capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import pandas as pd

from stock_analysis.models.data_models import (
    SecurityInfo, FinancialRatios, ValidationError
)


class EnhancedSecurityInfo(SecurityInfo):
    """Enhanced security information with additional data points."""
    
    def __init__(
        self,
        symbol: str,
        name: str,
        current_price: float,
        market_cap: Optional[float] = None,
        beta: Optional[float] = None,
        # Additional fundamental data
        earnings_growth: Optional[float] = None,
        revenue_growth: Optional[float] = None,
        profit_margin_trend: Optional[List[float]] = None,
        # Technical indicators
        rsi_14: Optional[float] = None,
        macd: Optional[Dict[str, float]] = None,
        moving_averages: Optional[Dict[str, float]] = None,
        # Analyst data
        analyst_rating: Optional[str] = None,
        price_target: Optional[Dict[str, float]] = None,  # low, average, high
        analyst_count: Optional[int] = None,
        # Additional metadata
        exchange: Optional[str] = None,
        currency: Optional[str] = None,
        company_description: Optional[str] = None,
        key_executives: Optional[List[Dict[str, str]]] = None,
        # StockInfo attributes
        company_name: str = "",
        pe_ratio: Optional[float] = None,
        pb_ratio: Optional[float] = None,
        dividend_yield: Optional[float] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None
    ):
        """Initialize EnhancedSecurityInfo.
        
        Args:
            symbol: Security ticker symbol
            name: Security name
            current_price: Current price
            market_cap: Market capitalization
            beta: Beta value
            earnings_growth: Earnings growth rate
            revenue_growth: Revenue growth rate
            profit_margin_trend: List of profit margin values over time
            rsi_14: 14-day Relative Strength Index
            macd: Moving Average Convergence Divergence data
            moving_averages: Dictionary of moving averages
            analyst_rating: Analyst consensus rating
            price_target: Dictionary with price target data
            analyst_count: Number of analysts covering the security
            exchange: Exchange where the security is listed
            currency: Currency of the security
            company_description: Description of the company
            key_executives: List of key executives
            company_name: Name of the company
            pe_ratio: Price to earnings ratio
            pb_ratio: Price to book ratio
            dividend_yield: Dividend yield
            sector: Business sector
            industry: Industry classification
        """
        # Initialize the base SecurityInfo class
        super().__init__(
            symbol=symbol,
            name=name,
            current_price=current_price,
            market_cap=market_cap,
            beta=beta
        )
        
        # Additional fundamental data
        self.earnings_growth = earnings_growth
        self.revenue_growth = revenue_growth
        self.profit_margin_trend = profit_margin_trend
        
        # Technical indicators
        self.rsi_14 = rsi_14
        self.macd = macd
        self.moving_averages = moving_averages
        
        # Analyst data
        self.analyst_rating = analyst_rating
        self.price_target = price_target
        self.analyst_count = analyst_count
        
        # Additional metadata
        self.exchange = exchange
        self.currency = currency
        self.company_description = company_description
        self.key_executives = key_executives
        
        # StockInfo attributes
        self.company_name = company_name if company_name else name
        self.pe_ratio = pe_ratio
        self.pb_ratio = pb_ratio
        self.dividend_yield = dividend_yield
        self.sector = sector
        self.industry = industry
    
    def validate(self) -> None:
        """Validate the enhanced security information data.
        
        Raises:
            ValidationError: If any validation checks fail.
        """
        # First validate the base class fields
        super().validate()
        
        # Validate StockInfo fields
        if self.pe_ratio is not None and self.pe_ratio <= 0:
            raise ValidationError(f"PE ratio must be positive: {self.pe_ratio}")
        
        if self.pb_ratio is not None and self.pb_ratio <= 0:
            raise ValidationError(f"PB ratio must be positive: {self.pb_ratio}")
        
        if self.dividend_yield is not None and self.dividend_yield < 0:
            raise ValidationError(f"Dividend yield cannot be negative: {self.dividend_yield}")
        
        if self.sector is not None and not isinstance(self.sector, str):
            raise ValidationError(f"Sector must be a string: {self.sector}")
        
        if self.industry is not None and not isinstance(self.industry, str):
            raise ValidationError(f"Industry must be a string: {self.industry}")
        
        # Validate additional fundamental data
        if self.earnings_growth is not None and not isinstance(self.earnings_growth, (int, float)):
            raise ValidationError(f"Earnings growth must be a number: {self.earnings_growth}")
        
        if self.revenue_growth is not None and not isinstance(self.revenue_growth, (int, float)):
            raise ValidationError(f"Revenue growth must be a number: {self.revenue_growth}")
        
        if self.profit_margin_trend is not None:
            if not isinstance(self.profit_margin_trend, list):
                raise ValidationError(f"Profit margin trend must be a list: {self.profit_margin_trend}")
            for value in self.profit_margin_trend:
                if not isinstance(value, (int, float)):
                    raise ValidationError(f"Profit margin trend values must be numbers: {value}")
        
        # Validate technical indicators
        if self.rsi_14 is not None:
            if not isinstance(self.rsi_14, (int, float)):
                raise ValidationError(f"RSI-14 must be a number: {self.rsi_14}")
            if not 0 <= self.rsi_14 <= 100:
                raise ValidationError(f"RSI-14 must be between 0 and 100: {self.rsi_14}")
        
        if self.macd is not None:
            if not isinstance(self.macd, dict):
                raise ValidationError(f"MACD must be a dictionary: {self.macd}")
            required_keys = ["macd_line", "signal_line", "histogram"]
            for key in required_keys:
                if key not in self.macd:
                    raise ValidationError(f"MACD dictionary missing required key: {key}")
                if not isinstance(self.macd[key], (int, float)):
                    raise ValidationError(f"MACD {key} must be a number: {self.macd[key]}")
        
        if self.moving_averages is not None:
            if not isinstance(self.moving_averages, dict):
                raise ValidationError(f"Moving averages must be a dictionary: {self.moving_averages}")
            for key, value in self.moving_averages.items():
                if not isinstance(value, (int, float)):
                    raise ValidationError(f"Moving average value must be a number: {value}")
        
        # Validate analyst data
        valid_ratings = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell", None]
        if self.analyst_rating not in valid_ratings:
            raise ValidationError(f"Analyst rating must be one of {valid_ratings}: {self.analyst_rating}")
        
        if self.price_target is not None:
            if not isinstance(self.price_target, dict):
                raise ValidationError(f"Price target must be a dictionary: {self.price_target}")
            required_keys = ["low", "average", "high"]
            for key in required_keys:
                if key not in self.price_target:
                    raise ValidationError(f"Price target dictionary missing required key: {key}")
                if not isinstance(self.price_target[key], (int, float)):
                    raise ValidationError(f"Price target {key} must be a number: {self.price_target[key]}")
                if self.price_target[key] < 0:
                    raise ValidationError(f"Price target {key} cannot be negative: {self.price_target[key]}")
        
        if self.analyst_count is not None:
            if not isinstance(self.analyst_count, int):
                raise ValidationError(f"Analyst count must be an integer: {self.analyst_count}")
            if self.analyst_count < 0:
                raise ValidationError(f"Analyst count cannot be negative: {self.analyst_count}")
        
        # Validate additional metadata
        if self.exchange is not None and not isinstance(self.exchange, str):
            raise ValidationError(f"Exchange must be a string: {self.exchange}")
        
        if self.currency is not None and not isinstance(self.currency, str):
            raise ValidationError(f"Currency must be a string: {self.currency}")
        
        if self.company_description is not None and not isinstance(self.company_description, str):
            raise ValidationError(f"Company description must be a string: {self.company_description}")
        
        if self.key_executives is not None:
            if not isinstance(self.key_executives, list):
                raise ValidationError(f"Key executives must be a list: {self.key_executives}")
            for executive in self.key_executives:
                if not isinstance(executive, dict):
                    raise ValidationError(f"Key executive must be a dictionary: {executive}")
                required_keys = ["name", "title"]
                for key in required_keys:
                    if key not in executive:
                        raise ValidationError(f"Key executive dictionary missing required key: {key}")
                    if not isinstance(executive[key], str):
                        raise ValidationError(f"Key executive {key} must be a string: {executive[key]}")


@dataclass
class EnhancedFinancialStatements:
    """Enhanced financial statements with additional metrics and historical data."""
    
    income_statements: Dict[str, pd.DataFrame]  # Key is period (annual/quarterly)
    balance_sheets: Dict[str, pd.DataFrame]
    cash_flow_statements: Dict[str, pd.DataFrame]
    
    # Additional financial metrics
    key_metrics: Dict[str, List[float]]  # Historical values for key metrics
    growth_metrics: Dict[str, List[float]]  # Growth rates for key metrics
    
    # Comparison data
    industry_averages: Optional[Dict[str, float]] = None
    sector_averages: Optional[Dict[str, float]] = None
    
    def validate(self) -> None:
        """Validate the enhanced financial statements data.
        
        Raises:
            ValidationError: If any validation checks fail.
        """
        # Validate financial statements
        required_statements = {
            "income_statements": self.income_statements,
            "balance_sheets": self.balance_sheets,
            "cash_flow_statements": self.cash_flow_statements
        }
        
        for statement_name, statement_dict in required_statements.items():
            if not isinstance(statement_dict, dict):
                raise ValidationError(f"{statement_name} must be a dictionary: {statement_dict}")
            
            if not statement_dict:
                raise ValidationError(f"{statement_name} dictionary cannot be empty")
            
            for period, df in statement_dict.items():
                if not isinstance(period, str):
                    raise ValidationError(f"Period key in {statement_name} must be a string: {period}")
                
                if not isinstance(df, pd.DataFrame):
                    raise ValidationError(f"Statement data in {statement_name} must be a DataFrame: {df}")
                
                if df.empty:
                    raise ValidationError(f"DataFrame in {statement_name} for period {period} cannot be empty")
        
        # Validate key metrics
        if not isinstance(self.key_metrics, dict):
            raise ValidationError(f"Key metrics must be a dictionary: {self.key_metrics}")
        
        if not self.key_metrics:
            raise ValidationError("Key metrics dictionary cannot be empty")
        
        for metric_name, values in self.key_metrics.items():
            if not isinstance(metric_name, str):
                raise ValidationError(f"Metric name must be a string: {metric_name}")
            
            if not isinstance(values, list):
                raise ValidationError(f"Metric values must be a list: {values}")
            
            if not values:
                raise ValidationError(f"Metric values list for {metric_name} cannot be empty")
            
            for value in values:
                if not isinstance(value, (int, float)) and value is not None:
                    raise ValidationError(f"Metric value must be a number or None: {value}")
        
        # Validate growth metrics
        if not isinstance(self.growth_metrics, dict):
            raise ValidationError(f"Growth metrics must be a dictionary: {self.growth_metrics}")
        
        if not self.growth_metrics:
            raise ValidationError("Growth metrics dictionary cannot be empty")
        
        for metric_name, values in self.growth_metrics.items():
            if not isinstance(metric_name, str):
                raise ValidationError(f"Growth metric name must be a string: {metric_name}")
            
            if not isinstance(values, list):
                raise ValidationError(f"Growth metric values must be a list: {values}")
            
            if not values:
                raise ValidationError(f"Growth metric values list for {metric_name} cannot be empty")
            
            for value in values:
                if not isinstance(value, (int, float)) and value is not None:
                    raise ValidationError(f"Growth metric value must be a number or None: {value}")
        
        # Validate comparison data
        if self.industry_averages is not None:
            if not isinstance(self.industry_averages, dict):
                raise ValidationError(f"Industry averages must be a dictionary: {self.industry_averages}")
            
            if not self.industry_averages:
                raise ValidationError("Industry averages dictionary cannot be empty")
            
            for metric_name, value in self.industry_averages.items():
                if not isinstance(metric_name, str):
                    raise ValidationError(f"Industry average metric name must be a string: {metric_name}")
                
                if not isinstance(value, (int, float)) and value is not None:
                    raise ValidationError(f"Industry average value must be a number or None: {value}")
        
        if self.sector_averages is not None:
            if not isinstance(self.sector_averages, dict):
                raise ValidationError(f"Sector averages must be a dictionary: {self.sector_averages}")
            
            if not self.sector_averages:
                raise ValidationError("Sector averages dictionary cannot be empty")
            
            for metric_name, value in self.sector_averages.items():
                if not isinstance(metric_name, str):
                    raise ValidationError(f"Sector average metric name must be a string: {metric_name}")
                
                if not isinstance(value, (int, float)) and value is not None:
                    raise ValidationError(f"Sector average value must be a number or None: {value}")


@dataclass
class MarketData:
    """Market data including indices, commodities, forex, and economic indicators."""
    
    indices: Dict[str, Dict[str, Any]]  # Major market indices
    commodities: Dict[str, Dict[str, Any]]  # Commodity prices
    forex: Dict[str, Dict[str, Any]]  # Foreign exchange rates
    sector_performance: Dict[str, float]  # Sector performance metrics
    economic_indicators: Dict[str, Dict[str, Any]]  # Economic indicators
    
    def validate(self) -> None:
        """Validate the market data.
        
        Raises:
            ValidationError: If any validation checks fail.
        """
        # Validate indices
        if not isinstance(self.indices, dict):
            raise ValidationError(f"Indices must be a dictionary: {self.indices}")
        
        if not self.indices:
            raise ValidationError("Indices dictionary cannot be empty")
        
        for index_name, index_data in self.indices.items():
            if not isinstance(index_name, str):
                raise ValidationError(f"Index name must be a string: {index_name}")
            
            if not isinstance(index_data, dict):
                raise ValidationError(f"Index data must be a dictionary: {index_data}")
            
            if not index_data:
                raise ValidationError(f"Index data dictionary for {index_name} cannot be empty")
            
            required_keys = ["value", "change", "change_percent"]
            for key in required_keys:
                if key not in index_data:
                    raise ValidationError(f"Index data dictionary missing required key: {key}")
                
                if not isinstance(index_data[key], (int, float)) and index_data[key] is not None:
                    raise ValidationError(f"Index data {key} must be a number or None: {index_data[key]}")
        
        # Validate commodities
        if not isinstance(self.commodities, dict):
            raise ValidationError(f"Commodities must be a dictionary: {self.commodities}")
        
        if not self.commodities:
            raise ValidationError("Commodities dictionary cannot be empty")
        
        for commodity_name, commodity_data in self.commodities.items():
            if not isinstance(commodity_name, str):
                raise ValidationError(f"Commodity name must be a string: {commodity_name}")
            
            if not isinstance(commodity_data, dict):
                raise ValidationError(f"Commodity data must be a dictionary: {commodity_data}")
            
            if not commodity_data:
                raise ValidationError(f"Commodity data dictionary for {commodity_name} cannot be empty")
            
            required_keys = ["value", "change", "change_percent", "unit"]
            for key in required_keys:
                if key not in commodity_data:
                    raise ValidationError(f"Commodity data dictionary missing required key: {key}")
                
                if key != "unit" and not isinstance(commodity_data[key], (int, float)) and commodity_data[key] is not None:
                    raise ValidationError(f"Commodity data {key} must be a number or None: {commodity_data[key]}")
                
                if key == "unit" and not isinstance(commodity_data[key], str):
                    raise ValidationError(f"Commodity unit must be a string: {commodity_data[key]}")
        
        # Validate forex
        if not isinstance(self.forex, dict):
            raise ValidationError(f"Forex must be a dictionary: {self.forex}")
        
        if not self.forex:
            raise ValidationError("Forex dictionary cannot be empty")
        
        for pair_name, pair_data in self.forex.items():
            if not isinstance(pair_name, str):
                raise ValidationError(f"Forex pair name must be a string: {pair_name}")
            
            if not isinstance(pair_data, dict):
                raise ValidationError(f"Forex pair data must be a dictionary: {pair_data}")
            
            if not pair_data:
                raise ValidationError(f"Forex pair data dictionary for {pair_name} cannot be empty")
            
            required_keys = ["value", "change", "change_percent"]
            for key in required_keys:
                if key not in pair_data:
                    raise ValidationError(f"Forex pair data dictionary missing required key: {key}")
                
                if not isinstance(pair_data[key], (int, float)) and pair_data[key] is not None:
                    raise ValidationError(f"Forex pair data {key} must be a number or None: {pair_data[key]}")
        
        # Validate sector performance
        if not isinstance(self.sector_performance, dict):
            raise ValidationError(f"Sector performance must be a dictionary: {self.sector_performance}")
        
        if not self.sector_performance:
            raise ValidationError("Sector performance dictionary cannot be empty")
        
        for sector_name, performance in self.sector_performance.items():
            if not isinstance(sector_name, str):
                raise ValidationError(f"Sector name must be a string: {sector_name}")
            
            if not isinstance(performance, (int, float)) and performance is not None:
                raise ValidationError(f"Sector performance must be a number or None: {performance}")
        
        # Validate economic indicators
        if not isinstance(self.economic_indicators, dict):
            raise ValidationError(f"Economic indicators must be a dictionary: {self.economic_indicators}")
        
        if not self.economic_indicators:
            raise ValidationError("Economic indicators dictionary cannot be empty")
        
        for indicator_name, indicator_data in self.economic_indicators.items():
            if not isinstance(indicator_name, str):
                raise ValidationError(f"Economic indicator name must be a string: {indicator_name}")
            
            if not isinstance(indicator_data, dict):
                raise ValidationError(f"Economic indicator data must be a dictionary: {indicator_data}")
            
            if not indicator_data:
                raise ValidationError(f"Economic indicator data dictionary for {indicator_name} cannot be empty")
            
            required_keys = ["value", "previous", "forecast", "unit"]
            for key in required_keys:
                if key not in indicator_data:
                    raise ValidationError(f"Economic indicator data dictionary missing required key: {key}")
                
                if key != "unit" and not isinstance(indicator_data[key], (int, float)) and indicator_data[key] is not None:
                    raise ValidationError(f"Economic indicator data {key} must be a number or None: {indicator_data[key]}")
                
                if key == "unit" and not isinstance(indicator_data[key], str):
                    raise ValidationError(f"Economic indicator unit must be a string: {indicator_data[key]}")


@dataclass
class NewsItem:
    """News article with metadata."""
    
    title: str
    source: str
    url: str
    published_at: datetime
    summary: str
    sentiment: Optional[float] = None  # -1.0 to 1.0
    impact: Optional[str] = None  # high, medium, low
    categories: Optional[List[str]] = None
    
    def validate(self) -> None:
        """Validate the news item data.
        
        Raises:
            ValidationError: If any validation checks fail.
        """
        # Validate required fields
        if not self.title:
            raise ValidationError("News title cannot be empty")
        
        if not isinstance(self.title, str):
            raise ValidationError(f"News title must be a string: {self.title}")
        
        if not self.source:
            raise ValidationError("News source cannot be empty")
        
        if not isinstance(self.source, str):
            raise ValidationError(f"News source must be a string: {self.source}")
        
        if not self.url:
            raise ValidationError("News URL cannot be empty")
        
        if not isinstance(self.url, str):
            raise ValidationError(f"News URL must be a string: {self.url}")
        
        if not isinstance(self.published_at, datetime):
            raise ValidationError(f"Published date must be a datetime object: {self.published_at}")
        
        if not self.summary:
            raise ValidationError("News summary cannot be empty")
        
        if not isinstance(self.summary, str):
            raise ValidationError(f"News summary must be a string: {self.summary}")
        
        # Validate optional fields
        if self.sentiment is not None:
            if not isinstance(self.sentiment, (int, float)):
                raise ValidationError(f"Sentiment must be a number: {self.sentiment}")
            
            if not -1.0 <= self.sentiment <= 1.0:
                raise ValidationError(f"Sentiment must be between -1.0 and 1.0: {self.sentiment}")
        
        valid_impacts = ["high", "medium", "low", None]
        if self.impact not in valid_impacts:
            raise ValidationError(f"Impact must be one of {valid_impacts}: {self.impact}")
        
        if self.categories is not None:
            if not isinstance(self.categories, list):
                raise ValidationError(f"Categories must be a list: {self.categories}")
            
            for category in self.categories:
                if not isinstance(category, str):
                    raise ValidationError(f"Category must be a string: {category}")
                
                if not category:
                    raise ValidationError("Category cannot be empty")