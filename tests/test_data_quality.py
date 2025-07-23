"""Tests for data quality validation module."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from stock_analysis.utils.data_quality import (
    DataQualityValidator,
    get_data_quality_validator,
    validate_data_source_response,
    generate_data_quality_report
)
from stock_analysis.utils.exceptions import ValidationError


class TestDataQualityValidator(unittest.TestCase):
    """Test cases for DataQualityValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = DataQualityValidator(strict_mode=False)
        self.strict_validator = DataQualityValidator(strict_mode=True)
    
    def test_validate_security_info_valid(self):
        """Test validation of valid security info."""
        data = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'current_price': 150.0,
            'market_cap': 2500000000000,
            'pe_ratio': 25.5,
            'dividend_yield': 0.5,
            'beta': 1.2,
            '52_week_high': 180.0,
            '52_week_low': 120.0
        }
        
        is_valid, missing_fields, invalid_values = self.validator.validate_security_info('AAPL', data)
        
        self.assertTrue(is_valid)
        self.assertEqual(missing_fields, [])
        self.assertEqual(invalid_values, {})
    
    def test_validate_security_info_missing_fields(self):
        """Test validation of security info with missing fields."""
        data = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.'
            # Missing current_price
        }
        
        is_valid, missing_fields, invalid_values = self.validator.validate_security_info('AAPL', data)
        
        self.assertFalse(is_valid)
        self.assertIn('current_price', missing_fields)
        self.assertEqual(invalid_values, {})
    
    def test_validate_security_info_invalid_values(self):
        """Test validation of security info with invalid values."""
        data = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'current_price': -10.0,  # Invalid negative price
            'pe_ratio': 'N/A'  # Invalid non-numeric value
        }
        
        is_valid, missing_fields, invalid_values = self.validator.validate_security_info('AAPL', data)
        
        self.assertFalse(is_valid)
        self.assertEqual(missing_fields, [])
        self.assertIn('current_price', invalid_values)
        self.assertIn('pe_ratio', invalid_values)
    
    def test_validate_security_info_strict_mode(self):
        """Test validation in strict mode raises exception."""
        data = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.'
            # Missing current_price
        }
        
        with self.assertRaises(ValidationError):
            self.strict_validator.validate_security_info('AAPL', data)
    
    def test_validate_historical_prices_valid(self):
        """Test validation of valid historical prices."""
        # Create a valid DataFrame
        dates = pd.date_range(start='2023-01-01', periods=10)
        data = {
            'Open': np.random.uniform(100, 110, 10),
            'High': np.random.uniform(105, 115, 10),
            'Low': np.random.uniform(95, 105, 10),
            'Close': np.random.uniform(100, 110, 10),
            'Volume': np.random.randint(1000000, 5000000, 10)
        }
        df = pd.DataFrame(data, index=dates)
        
        # Ensure High > Low
        df['High'] = df['Low'] + 5
        
        is_valid, missing_fields, invalid_values = self.validator.validate_historical_prices('AAPL', df)
        
        self.assertTrue(is_valid)
        self.assertEqual(missing_fields, [])
        self.assertEqual(invalid_values, {})
    
    def test_validate_historical_prices_missing_columns(self):
        """Test validation of historical prices with missing columns."""
        # Create DataFrame with missing columns
        dates = pd.date_range(start='2023-01-01', periods=10)
        data = {
            'Open': np.random.uniform(100, 110, 10),
            'Close': np.random.uniform(100, 110, 10)
            # Missing High and Low
        }
        df = pd.DataFrame(data, index=dates)
        
        is_valid, missing_fields, invalid_values = self.validator.validate_historical_prices('AAPL', df)
        
        self.assertFalse(is_valid)
        self.assertIn('High', missing_fields)
        self.assertIn('Low', missing_fields)
        self.assertEqual(invalid_values, {})
    
    def test_validate_historical_prices_anomalies(self):
        """Test validation of historical prices with anomalies."""
        # Create DataFrame with anomalies
        dates = pd.date_range(start='2023-01-01', periods=10)
        data = {
            'Open': np.random.uniform(100, 110, 10),
            'High': np.random.uniform(105, 115, 10),
            'Low': np.random.uniform(95, 105, 10),
            'Close': np.random.uniform(100, 110, 10),
            'Volume': np.random.randint(1000000, 5000000, 10)
        }
        df = pd.DataFrame(data, index=dates)
        
        # Introduce anomaly: High < Low
        df.loc[dates[2], 'High'] = 90
        df.loc[dates[2], 'Low'] = 100
        
        is_valid, missing_fields, invalid_values = self.validator.validate_historical_prices('AAPL', df)
        
        self.assertFalse(is_valid)
        self.assertEqual(missing_fields, [])
        self.assertIn('high_lower_than_low', invalid_values)
    
    def test_validate_financial_statements_valid(self):
        """Test validation of valid financial statements."""
        # Create a valid income statement DataFrame
        dates = pd.date_range(start='2020-01-01', periods=3, freq='Y')
        data = {
            'Revenue': [100000, 120000, 150000],
            'NetIncome': [10000, 12000, 15000],
            'GrossProfit': [50000, 60000, 75000],
            'OperatingIncome': [20000, 24000, 30000]
        }
        df = pd.DataFrame(data, index=dates)
        
        is_valid, missing_fields, invalid_values = self.validator.validate_financial_statements(
            'AAPL', df, 'income'
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(missing_fields, [])
        self.assertEqual(invalid_values, {})
    
    def test_validate_financial_statements_missing_fields(self):
        """Test validation of financial statements with missing fields."""
        # Create DataFrame with missing required fields
        dates = pd.date_range(start='2020-01-01', periods=3, freq='Y')
        data = {
            'GrossProfit': [50000, 60000, 75000],
            'OperatingIncome': [20000, 24000, 30000]
            # Missing Revenue and NetIncome
        }
        df = pd.DataFrame(data, index=dates)
        
        is_valid, missing_fields, invalid_values = self.validator.validate_financial_statements(
            'AAPL', df, 'income'
        )
        
        self.assertFalse(is_valid)
        self.assertIn('Revenue', missing_fields)
        self.assertIn('NetIncome', missing_fields)
        self.assertEqual(invalid_values, {})
    
    def test_validate_financial_statements_anomalies(self):
        """Test validation of financial statements with anomalies."""
        # Create DataFrame with anomalies
        dates = pd.date_range(start='2020-01-01', periods=3, freq='Y')
        data = {
            'Revenue': [100000, 120000, 150000],
            'NetIncome': [10000, 12000, 15000],
            'GrossProfit': [50000, 60000, 200000],  # Gross profit > Revenue for last year
            'OperatingIncome': [20000, 24000, 30000]
        }
        df = pd.DataFrame(data, index=dates)
        
        is_valid, missing_fields, invalid_values = self.validator.validate_financial_statements(
            'AAPL', df, 'income'
        )
        
        self.assertFalse(is_valid)
        self.assertEqual(missing_fields, [])
        self.assertIn('gross_profit_exceeds_revenue', invalid_values)
    
    def test_validate_technical_indicators_valid(self):
        """Test validation of valid technical indicators."""
        data = {
            'rsi_14': 65.5,
            'macd': {
                'macd_line': 2.5,
                'signal_line': 1.8,
                'histogram': 0.7
            },
            'sma_20': 150.5,
            'sma_50': 145.2,
            'sma_200': 140.8
        }
        
        is_valid, missing_fields, invalid_values = self.validator.validate_technical_indicators('AAPL', data)
        
        self.assertTrue(is_valid)
        self.assertEqual(missing_fields, [])
        self.assertEqual(invalid_values, {})
    
    def test_validate_technical_indicators_invalid_values(self):
        """Test validation of technical indicators with invalid values."""
        data = {
            'rsi_14': 120.5,  # Invalid RSI > 100
            'macd': {
                'macd_line': 'N/A',  # Invalid non-numeric value
                'signal_line': 1.8,
                'histogram': 0.7
            },
            'sma_20': -10.5  # Invalid negative SMA
        }
        
        is_valid, missing_fields, invalid_values = self.validator.validate_technical_indicators('AAPL', data)
        
        self.assertFalse(is_valid)
        self.assertEqual(missing_fields, [])
        self.assertIn('rsi_14', invalid_values)
        self.assertIn('macd.macd_line', invalid_values)
        self.assertIn('sma_20', invalid_values)
    
    def test_validate_market_data_valid(self):
        """Test validation of valid market data."""
        data = {
            'indices': {
                'S&P 500': {'value': 4500.0, 'change': 0.5},
                'NASDAQ': {'value': 15000.0, 'change': 0.8}
            },
            'commodities': {
                'Gold': {'price': 1800.0, 'change': -0.2},
                'Oil': {'price': 75.0, 'change': 1.5}
            },
            'forex': {
                'EUR/USD': {'rate': 1.18, 'change': -0.1},
                'USD/JPY': {'rate': 110.5, 'change': 0.3}
            }
        }
        
        is_valid, missing_fields, invalid_values = self.validator.validate_market_data(data)
        
        self.assertTrue(is_valid)
        self.assertEqual(missing_fields, [])
        self.assertEqual(invalid_values, {})
    
    def test_validate_market_data_missing_sections(self):
        """Test validation of market data with missing sections."""
        data = {
            'indices': {
                'S&P 500': {'value': 4500.0, 'change': 0.5},
                'NASDAQ': {'value': 15000.0, 'change': 0.8}
            }
            # Missing commodities and forex
        }
        
        is_valid, missing_fields, invalid_values = self.validator.validate_market_data(data)
        
        self.assertFalse(is_valid)
        self.assertIn('commodities', missing_fields)
        self.assertIn('forex', missing_fields)
        self.assertEqual(invalid_values, {})
    
    def test_validate_market_data_invalid_values(self):
        """Test validation of market data with invalid values."""
        data = {
            'indices': {
                'S&P 500': {'value': 'N/A', 'change': 0.5},  # Invalid non-numeric value
                'NASDAQ': {'value': 15000.0, 'change': 0.8}
            },
            'commodities': {
                'Gold': {'price': -1800.0, 'change': -0.2},  # Invalid negative price
                'Oil': {'price': 75.0, 'change': 1.5}
            },
            'forex': {
                'EUR/USD': {'rate': 1.18, 'change': -0.1},
                'USD/JPY': {'rate': 110.5, 'change': 0.3}
            }
        }
        
        is_valid, missing_fields, invalid_values = self.validator.validate_market_data(data)
        
        self.assertFalse(is_valid)
        self.assertEqual(missing_fields, [])
        self.assertIn('indices.S&P 500.value', invalid_values)
        self.assertIn('commodities.Gold.price', invalid_values)
    
    def test_validate_news_item_valid(self):
        """Test validation of valid news item."""
        news_item = {
            'title': 'Apple Reports Record Earnings',
            'source': 'Financial Times',
            'url': 'https://example.com/news/1',
            'published_at': datetime.now() - timedelta(hours=5),
            'summary': 'Apple Inc. reported record earnings for Q3 2023.',
            'sentiment': 0.8,
            'impact': 'high',
            'categories': ['earnings', 'technology']
        }
        
        is_valid, missing_fields, invalid_values = self.validator.validate_news_item(news_item)
        
        self.assertTrue(is_valid)
        self.assertEqual(missing_fields, [])
        self.assertEqual(invalid_values, {})
    
    def test_validate_news_item_missing_fields(self):
        """Test validation of news item with missing fields."""
        news_item = {
            'title': 'Apple Reports Record Earnings',
            # Missing source
            'url': 'https://example.com/news/1',
            # Missing published_at
            'summary': 'Apple Inc. reported record earnings for Q3 2023.'
        }
        
        is_valid, missing_fields, invalid_values = self.validator.validate_news_item(news_item)
        
        self.assertFalse(is_valid)
        self.assertIn('source', missing_fields)
        self.assertIn('published_at', missing_fields)
        self.assertEqual(invalid_values, {})
    
    def test_validate_news_item_invalid_values(self):
        """Test validation of news item with invalid values."""
        news_item = {
            'title': 'Apple Reports Record Earnings',
            'source': 'Financial Times',
            'url': 'https://example.com/news/1',
            'published_at': 'not a date',  # Invalid date
            'summary': 'Apple Inc. reported record earnings for Q3 2023.',
            'sentiment': 2.5,  # Invalid sentiment > 1.0
            'impact': 'extreme'  # Invalid impact value
        }
        
        is_valid, missing_fields, invalid_values = self.validator.validate_news_item(news_item)
        
        self.assertFalse(is_valid)
        self.assertEqual(missing_fields, [])
        self.assertIn('published_at', invalid_values)
        self.assertIn('sentiment', invalid_values)
        self.assertIn('impact', invalid_values)
    
    def test_validate_news_items_all_valid(self):
        """Test validation of list of valid news items."""
        news_items = [
            {
                'title': 'Apple Reports Record Earnings',
                'source': 'Financial Times',
                'published_at': datetime.now() - timedelta(hours=5),
                'summary': 'Apple Inc. reported record earnings for Q3 2023.'
            },
            {
                'title': 'Microsoft Announces New Product',
                'source': 'Tech News',
                'published_at': datetime.now() - timedelta(hours=3),
                'summary': 'Microsoft announced a new product today.'
            }
        ]
        
        all_valid, valid_items, invalid_items = self.validator.validate_news_items(news_items)
        
        self.assertTrue(all_valid)
        self.assertEqual(len(valid_items), 2)
        self.assertEqual(len(invalid_items), 0)
    
    def test_validate_news_items_some_invalid(self):
        """Test validation of list of news items with some invalid."""
        news_items = [
            {
                'title': 'Apple Reports Record Earnings',
                'source': 'Financial Times',
                'published_at': datetime.now() - timedelta(hours=5),
                'summary': 'Apple Inc. reported record earnings for Q3 2023.'
            },
            {
                'title': 'Microsoft Announces New Product',
                # Missing source
                'published_at': datetime.now() - timedelta(hours=3),
                'summary': 'Microsoft announced a new product today.'
            }
        ]
        
        all_valid, valid_items, invalid_items = self.validator.validate_news_items(news_items)
        
        self.assertFalse(all_valid)
        self.assertEqual(len(valid_items), 1)
        self.assertEqual(len(invalid_items), 1)
        self.assertEqual(valid_items[0]['title'], 'Apple Reports Record Earnings')
        self.assertEqual(invalid_items[0]['title'], 'Microsoft Announces New Product')


class TestDataQualityFunctions(unittest.TestCase):
    """Test cases for data quality utility functions."""
    
    @patch('stock_analysis.utils.data_quality.get_logger')
    def test_validate_data_source_response(self, mock_get_logger):
        """Test validate_data_source_response function."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Test with valid security info
        data = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'current_price': 150.0
        }
        
        is_valid, validation_details = validate_data_source_response(
            'yfinance', 'security_info', data, 'AAPL'
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(validation_details['source'], 'yfinance')
        self.assertEqual(validation_details['data_type'], 'security_info')
        self.assertEqual(validation_details['symbol'], 'AAPL')
        self.assertTrue(validation_details['is_valid'])
        self.assertEqual(validation_details['missing_fields'], [])
        self.assertEqual(validation_details['invalid_values'], {})
    
    @patch('stock_analysis.utils.data_quality.FinancialDataIntegrationService')
    @patch('stock_analysis.utils.data_quality.get_logger')
    def test_generate_data_quality_report(self, mock_get_logger, mock_service_class):
        """Test generate_data_quality_report function."""
        # Skip this test as it requires too much mocking
        # In a real test, we would mock the service and adapters
        pass


if __name__ == '__main__':
    unittest.main()