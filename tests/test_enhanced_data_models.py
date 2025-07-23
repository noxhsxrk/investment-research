"""Tests for enhanced data models.

This module contains tests for the enhanced data models used in the stock analysis system.
"""

import pytest
from typing import Dict, List

from stock_analysis.models.enhanced_data_models import EnhancedSecurityInfo
from stock_analysis.utils.exceptions import ValidationError


class TestEnhancedSecurityInfo:
    """Test cases for the EnhancedSecurityInfo class."""
    
    def test_valid_enhanced_security_info(self):
        """Test that a valid EnhancedSecurityInfo object passes validation."""
        # Create a valid EnhancedSecurityInfo object
        security_info = EnhancedSecurityInfo(
            symbol="AAPL",
            name="Apple Inc.",
            current_price=150.0,
            market_cap=2.5e12,
            beta=1.2,
            earnings_growth=0.15,
            revenue_growth=0.12,
            profit_margin_trend=[0.21, 0.22, 0.23],
            rsi_14=65.5,
            macd={"macd_line": 2.5, "signal_line": 1.8, "histogram": 0.7},
            moving_averages={"SMA_50": 148.5, "SMA_200": 142.0, "EMA_20": 151.2},
            analyst_rating="Buy",
            price_target={"low": 140.0, "average": 170.0, "high": 200.0},
            analyst_count=32,
            exchange="NASDAQ",
            currency="USD",
            company_description="Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
            key_executives=[
                {"name": "Tim Cook", "title": "CEO"},
                {"name": "Luca Maestri", "title": "CFO"}
            ]
        )
        
        # Validation should not raise any exceptions
        security_info.validate()
    
    def test_invalid_rsi_value(self):
        """Test that an invalid RSI value fails validation."""
        # Create an EnhancedSecurityInfo with an invalid RSI value
        security_info = EnhancedSecurityInfo(
            symbol="AAPL",
            name="Apple Inc.",
            current_price=150.0,
            rsi_14=105.0  # RSI should be between 0 and 100
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            security_info.validate()
        
        assert "RSI-14 must be between 0 and 100" in str(excinfo.value)
    
    def test_invalid_macd_structure(self):
        """Test that an invalid MACD structure fails validation."""
        # Create an EnhancedSecurityInfo with an invalid MACD structure
        security_info = EnhancedSecurityInfo(
            symbol="AAPL",
            name="Apple Inc.",
            current_price=150.0,
            macd={"macd_line": 2.5}  # Missing required keys
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            security_info.validate()
        
        assert "MACD dictionary missing required key" in str(excinfo.value)
    
    def test_invalid_analyst_rating(self):
        """Test that an invalid analyst rating fails validation."""
        # Create an EnhancedSecurityInfo with an invalid analyst rating
        security_info = EnhancedSecurityInfo(
            symbol="AAPL",
            name="Apple Inc.",
            current_price=150.0,
            analyst_rating="Super Buy"  # Not a valid rating
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            security_info.validate()
        
        assert "Analyst rating must be one of" in str(excinfo.value)
    
    def test_invalid_price_target(self):
        """Test that an invalid price target fails validation."""
        # Create an EnhancedSecurityInfo with an invalid price target
        security_info = EnhancedSecurityInfo(
            symbol="AAPL",
            name="Apple Inc.",
            current_price=150.0,
            price_target={"low": -10.0, "average": 170.0, "high": 200.0}  # Negative price target
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            security_info.validate()
        
        assert "Price target low cannot be negative" in str(excinfo.value)
    
    def test_invalid_analyst_count(self):
        """Test that an invalid analyst count fails validation."""
        # Create an EnhancedSecurityInfo with an invalid analyst count
        security_info = EnhancedSecurityInfo(
            symbol="AAPL",
            name="Apple Inc.",
            current_price=150.0,
            analyst_count=-5  # Negative count
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            security_info.validate()
        
        assert "Analyst count cannot be negative" in str(excinfo.value)
    
    def test_invalid_key_executives(self):
        """Test that invalid key executives data fails validation."""
        # Create an EnhancedSecurityInfo with invalid key executives data
        security_info = EnhancedSecurityInfo(
            symbol="AAPL",
            name="Apple Inc.",
            current_price=150.0,
            key_executives=[
                {"name": "Tim Cook"}  # Missing required key 'title'
            ]
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            security_info.validate()
        
        assert "Key executive dictionary missing required key" in str(excinfo.value)
    
    def test_inheritance_from_security_info(self):
        """Test that EnhancedSecurityInfo inherits validation from SecurityInfo."""
        # Create an EnhancedSecurityInfo with an invalid base class field
        security_info = EnhancedSecurityInfo(
            symbol="",  # Empty symbol should fail base class validation
            name="Apple Inc.",
            current_price=150.0
        )
        
        # Validation should raise a ValidationError from the base class
        with pytest.raises(ValidationError) as excinfo:
            security_info.validate()
        
        assert "Security symbol cannot be empty" in str(excinfo.value)


class TestEnhancedFinancialStatements:
    """Test cases for the EnhancedFinancialStatements class."""
    
    def test_valid_enhanced_financial_statements(self):
        """Test that a valid EnhancedFinancialStatements object passes validation."""
        from stock_analysis.models.enhanced_data_models import EnhancedFinancialStatements
        import pandas as pd
        
        # Create sample DataFrames for financial statements
        income_df = pd.DataFrame({
            'Revenue': [100000, 120000, 150000],
            'Cost of Revenue': [60000, 70000, 85000],
            'Gross Profit': [40000, 50000, 65000],
            'Operating Expenses': [30000, 35000, 40000],
            'Operating Income': [10000, 15000, 25000],
            'Net Income': [8000, 12000, 20000]
        }, index=['2020', '2021', '2022'])
        
        balance_df = pd.DataFrame({
            'Cash': [50000, 60000, 80000],
            'Accounts Receivable': [20000, 25000, 30000],
            'Inventory': [30000, 35000, 40000],
            'Total Assets': [200000, 250000, 300000],
            'Accounts Payable': [15000, 18000, 22000],
            'Long Term Debt': [50000, 45000, 40000],
            'Total Liabilities': [100000, 110000, 120000],
            'Stockholders Equity': [100000, 140000, 180000]
        }, index=['2020', '2021', '2022'])
        
        cash_flow_df = pd.DataFrame({
            'Operating Cash Flow': [15000, 20000, 30000],
            'Capital Expenditures': [-10000, -12000, -15000],
            'Free Cash Flow': [5000, 8000, 15000],
            'Dividends Paid': [-2000, -2500, -3000],
            'Net Borrowings': [-3000, -5000, -5000],
            'Net Cash Flow': [0, 500, 7000]
        }, index=['2020', '2021', '2022'])
        
        # Create a valid EnhancedFinancialStatements object
        financial_statements = EnhancedFinancialStatements(
            income_statements={
                'annual': income_df,
                'quarterly': income_df.copy()  # Just for testing, would be different in real use
            },
            balance_sheets={
                'annual': balance_df,
                'quarterly': balance_df.copy()  # Just for testing, would be different in real use
            },
            cash_flow_statements={
                'annual': cash_flow_df,
                'quarterly': cash_flow_df.copy()  # Just for testing, would be different in real use
            },
            key_metrics={
                'ROE': [0.08, 0.09, 0.11],
                'ROA': [0.04, 0.05, 0.07],
                'Profit Margin': [0.08, 0.10, 0.13],
                'Debt to Equity': [0.5, 0.45, 0.4]
            },
            growth_metrics={
                'Revenue Growth': [None, 0.20, 0.25],
                'EPS Growth': [None, 0.50, 0.67],
                'FCF Growth': [None, 0.60, 0.88]
            },
            industry_averages={
                'ROE': 0.10,
                'ROA': 0.06,
                'Profit Margin': 0.12,
                'Debt to Equity': 0.45
            },
            sector_averages={
                'ROE': 0.09,
                'ROA': 0.05,
                'Profit Margin': 0.11,
                'Debt to Equity': 0.48
            }
        )
        
        # Validation should not raise any exceptions
        financial_statements.validate()
    
    def test_invalid_financial_statements_type(self):
        """Test that invalid financial statements type fails validation."""
        from stock_analysis.models.enhanced_data_models import EnhancedFinancialStatements
        import pandas as pd
        
        # Create a sample DataFrame
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        # Create an EnhancedFinancialStatements with invalid income_statements type
        financial_statements = EnhancedFinancialStatements(
            income_statements="not a dictionary",  # Should be a dictionary
            balance_sheets={'annual': df},
            cash_flow_statements={'annual': df},
            key_metrics={'ROE': [0.08, 0.09, 0.11]},
            growth_metrics={'Revenue Growth': [None, 0.20, 0.25]}
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            financial_statements.validate()
        
        assert "income_statements must be a dictionary" in str(excinfo.value)
    
    def test_empty_financial_statements(self):
        """Test that empty financial statements fail validation."""
        from stock_analysis.models.enhanced_data_models import EnhancedFinancialStatements
        import pandas as pd
        
        # Create a sample DataFrame
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        # Create an EnhancedFinancialStatements with empty income_statements
        financial_statements = EnhancedFinancialStatements(
            income_statements={},  # Empty dictionary
            balance_sheets={'annual': df},
            cash_flow_statements={'annual': df},
            key_metrics={'ROE': [0.08, 0.09, 0.11]},
            growth_metrics={'Revenue Growth': [None, 0.20, 0.25]}
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            financial_statements.validate()
        
        assert "income_statements dictionary cannot be empty" in str(excinfo.value)
    
    def test_invalid_dataframe(self):
        """Test that invalid DataFrame fails validation."""
        from stock_analysis.models.enhanced_data_models import EnhancedFinancialStatements
        import pandas as pd
        
        # Create a sample DataFrame
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        # Create an EnhancedFinancialStatements with non-DataFrame value
        financial_statements = EnhancedFinancialStatements(
            income_statements={'annual': "not a dataframe"},
            balance_sheets={'annual': df},
            cash_flow_statements={'annual': df},
            key_metrics={'ROE': [0.08, 0.09, 0.11]},
            growth_metrics={'Revenue Growth': [None, 0.20, 0.25]}
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            financial_statements.validate()
        
        assert "Statement data in income_statements must be a DataFrame" in str(excinfo.value)
    
    def test_empty_dataframe(self):
        """Test that empty DataFrame fails validation."""
        from stock_analysis.models.enhanced_data_models import EnhancedFinancialStatements
        import pandas as pd
        
        # Create sample DataFrames
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        empty_df = pd.DataFrame()
        
        # Create an EnhancedFinancialStatements with empty DataFrame
        financial_statements = EnhancedFinancialStatements(
            income_statements={'annual': empty_df},
            balance_sheets={'annual': df},
            cash_flow_statements={'annual': df},
            key_metrics={'ROE': [0.08, 0.09, 0.11]},
            growth_metrics={'Revenue Growth': [None, 0.20, 0.25]}
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            financial_statements.validate()
        
        assert "DataFrame in income_statements for period annual cannot be empty" in str(excinfo.value)
    
    def test_invalid_key_metrics(self):
        """Test that invalid key metrics fail validation."""
        from stock_analysis.models.enhanced_data_models import EnhancedFinancialStatements
        import pandas as pd
        
        # Create a sample DataFrame
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        # Create an EnhancedFinancialStatements with invalid key_metrics
        financial_statements = EnhancedFinancialStatements(
            income_statements={'annual': df},
            balance_sheets={'annual': df},
            cash_flow_statements={'annual': df},
            key_metrics={'ROE': "not a list"},  # Should be a list
            growth_metrics={'Revenue Growth': [None, 0.20, 0.25]}
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            financial_statements.validate()
        
        assert "Metric values must be a list" in str(excinfo.value)
    
    def test_invalid_growth_metrics(self):
        """Test that invalid growth metrics fail validation."""
        from stock_analysis.models.enhanced_data_models import EnhancedFinancialStatements
        import pandas as pd
        
        # Create a sample DataFrame
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        # Create an EnhancedFinancialStatements with invalid growth_metrics
        financial_statements = EnhancedFinancialStatements(
            income_statements={'annual': df},
            balance_sheets={'annual': df},
            cash_flow_statements={'annual': df},
            key_metrics={'ROE': [0.08, 0.09, 0.11]},
            growth_metrics={'Revenue Growth': []}  # Empty list
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            financial_statements.validate()
        
        assert "Growth metric values list for Revenue Growth cannot be empty" in str(excinfo.value)
    
    def test_invalid_industry_averages(self):
        """Test that invalid industry averages fail validation."""
        from stock_analysis.models.enhanced_data_models import EnhancedFinancialStatements
        import pandas as pd
        
        # Create a sample DataFrame
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        # Create an EnhancedFinancialStatements with invalid industry_averages
        financial_statements = EnhancedFinancialStatements(
            income_statements={'annual': df},
            balance_sheets={'annual': df},
            cash_flow_statements={'annual': df},
            key_metrics={'ROE': [0.08, 0.09, 0.11]},
            growth_metrics={'Revenue Growth': [None, 0.20, 0.25]},
            industry_averages={'ROE': "not a number"}  # Should be a number
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            financial_statements.validate()
        
        assert "Industry average value must be a number or None" in str(excinfo.value)


class TestMarketData:
    """Test cases for the MarketData class."""
    
    def test_valid_market_data(self):
        """Test that a valid MarketData object passes validation."""
        from stock_analysis.models.enhanced_data_models import MarketData
        
        # Create a valid MarketData object
        market_data = MarketData(
            indices={
                "S&P 500": {"value": 4500.0, "change": 15.5, "change_percent": 0.35},
                "NASDAQ": {"value": 14000.0, "change": 50.2, "change_percent": 0.36},
                "Dow Jones": {"value": 35000.0, "change": -120.5, "change_percent": -0.34}
            },
            commodities={
                "Gold": {"value": 1850.0, "change": 5.5, "change_percent": 0.3, "unit": "USD/oz"},
                "Oil (WTI)": {"value": 75.5, "change": -1.2, "change_percent": -1.56, "unit": "USD/bbl"},
                "Natural Gas": {"value": 3.85, "change": 0.15, "change_percent": 4.05, "unit": "USD/MMBtu"}
            },
            forex={
                "EUR/USD": {"value": 1.18, "change": 0.002, "change_percent": 0.17},
                "USD/JPY": {"value": 110.5, "change": -0.3, "change_percent": -0.27},
                "GBP/USD": {"value": 1.38, "change": 0.005, "change_percent": 0.36}
            },
            sector_performance={
                "Technology": 0.75,
                "Healthcare": -0.25,
                "Financials": 0.45,
                "Energy": -1.2,
                "Consumer Discretionary": 0.35
            },
            economic_indicators={
                "GDP Growth": {"value": 2.8, "previous": 2.2, "forecast": 3.0, "unit": "%"},
                "Unemployment Rate": {"value": 5.2, "previous": 5.4, "forecast": 5.1, "unit": "%"},
                "Inflation Rate": {"value": 2.5, "previous": 2.3, "forecast": 2.6, "unit": "%"},
                "Interest Rate": {"value": 0.25, "previous": 0.25, "forecast": 0.25, "unit": "%"}
            }
        )
        
        # Validation should not raise any exceptions
        market_data.validate()
    
    def test_invalid_indices_type(self):
        """Test that invalid indices type fails validation."""
        from stock_analysis.models.enhanced_data_models import MarketData
        
        # Create a MarketData object with invalid indices type
        market_data = MarketData(
            indices="not a dictionary",  # Should be a dictionary
            commodities={"Gold": {"value": 1850.0, "change": 5.5, "change_percent": 0.3, "unit": "USD/oz"}},
            forex={"EUR/USD": {"value": 1.18, "change": 0.002, "change_percent": 0.17}},
            sector_performance={"Technology": 0.75},
            economic_indicators={"GDP Growth": {"value": 2.8, "previous": 2.2, "forecast": 3.0, "unit": "%"}}
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            market_data.validate()
        
        assert "Indices must be a dictionary" in str(excinfo.value)
    
    def test_empty_indices(self):
        """Test that empty indices fail validation."""
        from stock_analysis.models.enhanced_data_models import MarketData
        
        # Create a MarketData object with empty indices
        market_data = MarketData(
            indices={},  # Empty dictionary
            commodities={"Gold": {"value": 1850.0, "change": 5.5, "change_percent": 0.3, "unit": "USD/oz"}},
            forex={"EUR/USD": {"value": 1.18, "change": 0.002, "change_percent": 0.17}},
            sector_performance={"Technology": 0.75},
            economic_indicators={"GDP Growth": {"value": 2.8, "previous": 2.2, "forecast": 3.0, "unit": "%"}}
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            market_data.validate()
        
        assert "Indices dictionary cannot be empty" in str(excinfo.value)
    
    def test_invalid_index_data(self):
        """Test that invalid index data fails validation."""
        from stock_analysis.models.enhanced_data_models import MarketData
        
        # Create a MarketData object with invalid index data
        market_data = MarketData(
            indices={"S&P 500": "not a dictionary"},  # Should be a dictionary
            commodities={"Gold": {"value": 1850.0, "change": 5.5, "change_percent": 0.3, "unit": "USD/oz"}},
            forex={"EUR/USD": {"value": 1.18, "change": 0.002, "change_percent": 0.17}},
            sector_performance={"Technology": 0.75},
            economic_indicators={"GDP Growth": {"value": 2.8, "previous": 2.2, "forecast": 3.0, "unit": "%"}}
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            market_data.validate()
        
        assert "Index data must be a dictionary" in str(excinfo.value)
    
    def test_missing_required_index_key(self):
        """Test that missing required index key fails validation."""
        from stock_analysis.models.enhanced_data_models import MarketData
        
        # Create a MarketData object with missing required index key
        market_data = MarketData(
            indices={"S&P 500": {"value": 4500.0, "change": 15.5}},  # Missing change_percent
            commodities={"Gold": {"value": 1850.0, "change": 5.5, "change_percent": 0.3, "unit": "USD/oz"}},
            forex={"EUR/USD": {"value": 1.18, "change": 0.002, "change_percent": 0.17}},
            sector_performance={"Technology": 0.75},
            economic_indicators={"GDP Growth": {"value": 2.8, "previous": 2.2, "forecast": 3.0, "unit": "%"}}
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            market_data.validate()
        
        assert "Index data dictionary missing required key" in str(excinfo.value)
    
    def test_invalid_commodity_unit(self):
        """Test that invalid commodity unit fails validation."""
        from stock_analysis.models.enhanced_data_models import MarketData
        
        # Create a MarketData object with invalid commodity unit
        market_data = MarketData(
            indices={"S&P 500": {"value": 4500.0, "change": 15.5, "change_percent": 0.35}},
            commodities={"Gold": {"value": 1850.0, "change": 5.5, "change_percent": 0.3, "unit": 123}},  # Unit should be a string
            forex={"EUR/USD": {"value": 1.18, "change": 0.002, "change_percent": 0.17}},
            sector_performance={"Technology": 0.75},
            economic_indicators={"GDP Growth": {"value": 2.8, "previous": 2.2, "forecast": 3.0, "unit": "%"}}
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            market_data.validate()
        
        assert "Commodity unit must be a string" in str(excinfo.value)
    
    def test_invalid_sector_performance(self):
        """Test that invalid sector performance fails validation."""
        from stock_analysis.models.enhanced_data_models import MarketData
        
        # Create a MarketData object with invalid sector performance
        market_data = MarketData(
            indices={"S&P 500": {"value": 4500.0, "change": 15.5, "change_percent": 0.35}},
            commodities={"Gold": {"value": 1850.0, "change": 5.5, "change_percent": 0.3, "unit": "USD/oz"}},
            forex={"EUR/USD": {"value": 1.18, "change": 0.002, "change_percent": 0.17}},
            sector_performance={"Technology": "not a number"},  # Should be a number
            economic_indicators={"GDP Growth": {"value": 2.8, "previous": 2.2, "forecast": 3.0, "unit": "%"}}
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            market_data.validate()
        
        assert "Sector performance must be a number or None" in str(excinfo.value)
    
    def test_invalid_economic_indicator_data(self):
        """Test that invalid economic indicator data fails validation."""
        from stock_analysis.models.enhanced_data_models import MarketData
        
        # Create a MarketData object with invalid economic indicator data
        market_data = MarketData(
            indices={"S&P 500": {"value": 4500.0, "change": 15.5, "change_percent": 0.35}},
            commodities={"Gold": {"value": 1850.0, "change": 5.5, "change_percent": 0.3, "unit": "USD/oz"}},
            forex={"EUR/USD": {"value": 1.18, "change": 0.002, "change_percent": 0.17}},
            sector_performance={"Technology": 0.75},
            economic_indicators={"GDP Growth": {"value": "not a number", "previous": 2.2, "forecast": 3.0, "unit": "%"}}
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            market_data.validate()
        
        assert "Economic indicator data value must be a number or None" in str(excinfo.value)


class TestNewsItem:
    """Test cases for the NewsItem class."""
    
    def test_valid_news_item(self):
        """Test that a valid NewsItem object passes validation."""
        from stock_analysis.models.enhanced_data_models import NewsItem
        from datetime import datetime
        
        # Create a valid NewsItem object
        news_item = NewsItem(
            title="Apple Reports Record Q3 Earnings",
            source="Financial Times",
            url="https://ft.com/apple-q3-earnings",
            published_at=datetime(2023, 7, 28, 16, 30, 0),
            summary="Apple Inc. reported record third-quarter earnings, beating analyst expectations with strong iPhone sales.",
            sentiment=0.75,
            impact="high",
            categories=["Technology", "Earnings", "Apple"]
        )
        
        # Validation should not raise any exceptions
        news_item.validate()
    
    def test_empty_title(self):
        """Test that empty title fails validation."""
        from stock_analysis.models.enhanced_data_models import NewsItem
        from datetime import datetime
        
        # Create a NewsItem with empty title
        news_item = NewsItem(
            title="",  # Empty title
            source="Financial Times",
            url="https://ft.com/apple-q3-earnings",
            published_at=datetime(2023, 7, 28, 16, 30, 0),
            summary="Apple Inc. reported record third-quarter earnings, beating analyst expectations with strong iPhone sales."
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            news_item.validate()
        
        assert "News title cannot be empty" in str(excinfo.value)
    
    def test_invalid_published_at(self):
        """Test that invalid published_at fails validation."""
        from stock_analysis.models.enhanced_data_models import NewsItem
        
        # Create a NewsItem with invalid published_at
        news_item = NewsItem(
            title="Apple Reports Record Q3 Earnings",
            source="Financial Times",
            url="https://ft.com/apple-q3-earnings",
            published_at="2023-07-28",  # Should be a datetime object
            summary="Apple Inc. reported record third-quarter earnings, beating analyst expectations with strong iPhone sales."
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            news_item.validate()
        
        assert "Published date must be a datetime object" in str(excinfo.value)
    
    def test_invalid_sentiment_range(self):
        """Test that invalid sentiment range fails validation."""
        from stock_analysis.models.enhanced_data_models import NewsItem
        from datetime import datetime
        
        # Create a NewsItem with invalid sentiment range
        news_item = NewsItem(
            title="Apple Reports Record Q3 Earnings",
            source="Financial Times",
            url="https://ft.com/apple-q3-earnings",
            published_at=datetime(2023, 7, 28, 16, 30, 0),
            summary="Apple Inc. reported record third-quarter earnings, beating analyst expectations with strong iPhone sales.",
            sentiment=1.5  # Should be between -1.0 and 1.0
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            news_item.validate()
        
        assert "Sentiment must be between -1.0 and 1.0" in str(excinfo.value)
    
    def test_invalid_impact(self):
        """Test that invalid impact fails validation."""
        from stock_analysis.models.enhanced_data_models import NewsItem
        from datetime import datetime
        
        # Create a NewsItem with invalid impact
        news_item = NewsItem(
            title="Apple Reports Record Q3 Earnings",
            source="Financial Times",
            url="https://ft.com/apple-q3-earnings",
            published_at=datetime(2023, 7, 28, 16, 30, 0),
            summary="Apple Inc. reported record third-quarter earnings, beating analyst expectations with strong iPhone sales.",
            impact="critical"  # Should be high, medium, low, or None
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            news_item.validate()
        
        assert "Impact must be one of" in str(excinfo.value)
    
    def test_invalid_categories(self):
        """Test that invalid categories fails validation."""
        from stock_analysis.models.enhanced_data_models import NewsItem
        from datetime import datetime
        
        # Create a NewsItem with invalid categories
        news_item = NewsItem(
            title="Apple Reports Record Q3 Earnings",
            source="Financial Times",
            url="https://ft.com/apple-q3-earnings",
            published_at=datetime(2023, 7, 28, 16, 30, 0),
            summary="Apple Inc. reported record third-quarter earnings, beating analyst expectations with strong iPhone sales.",
            categories=["Technology", 123, "Apple"]  # All categories should be strings
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            news_item.validate()
        
        assert "Category must be a string" in str(excinfo.value)
    
    def test_empty_category(self):
        """Test that empty category fails validation."""
        from stock_analysis.models.enhanced_data_models import NewsItem
        from datetime import datetime
        
        # Create a NewsItem with empty category
        news_item = NewsItem(
            title="Apple Reports Record Q3 Earnings",
            source="Financial Times",
            url="https://ft.com/apple-q3-earnings",
            published_at=datetime(2023, 7, 28, 16, 30, 0),
            summary="Apple Inc. reported record third-quarter earnings, beating analyst expectations with strong iPhone sales.",
            categories=["Technology", "", "Apple"]  # Empty category
        )
        
        # Validation should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            news_item.validate()
        
        assert "Category cannot be empty" in str(excinfo.value)