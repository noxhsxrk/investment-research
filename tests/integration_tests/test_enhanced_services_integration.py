"""Integration tests for enhanced services.

This module contains integration tests for the enhanced service layer,
testing the interaction between services and the data integration layer.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import pytest

from stock_analysis.services.enhanced_stock_data_service import EnhancedStockDataService
from stock_analysis.services.market_data_service import MarketDataService
from stock_analysis.services.news_service import NewsService
from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
from stock_analysis.models.enhanced_data_models import EnhancedSecurityInfo, MarketData, NewsItem
from stock_analysis.utils.exceptions import DataRetrievalError


class TestEnhancedServicesIntegration(unittest.TestCase):
    """Integration tests for enhanced services."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock integration service
        self.mock_integration_service = MagicMock(spec=FinancialDataIntegrationService)
        
        # Create services with mock integration service
        self.enhanced_stock_service = EnhancedStockDataService(self.mock_integration_service)
        self.market_service = MarketDataService(self.mock_integration_service)
        self.news_service = NewsService(self.mock_integration_service)
        
        # Sample test data
        self.sample_security_info = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'current_price': 150.0,
            'market_cap': 2500000000000,
            'pe_ratio': 25.0,
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'exchange': 'NASDAQ',
            'currency': 'USD',
            'company_description': 'Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.'
        }
        
        self.sample_technical_indicators = {
            'sma_20': 148.5,
            'sma_50': 145.2,
            'rsi_14': 65.3,
            'macd': {
                'macd': 2.5,
                'signal': 1.8,
                'histogram': 0.7
            }
        }
        
        self.sample_analyst_data = {
            'rating': 'Buy',
            'price_target': {
                'low': 140.0,
                'average': 170.0,
                'high': 200.0
            },
            'analyst_count': 35
        }
        
        self.sample_historical_data = pd.DataFrame({
            'Open': [145.0, 146.0, 147.0],
            'High': [150.0, 151.0, 152.0],
            'Low': [144.0, 145.0, 146.0],
            'Close': [149.0, 150.0, 151.0],
            'Volume': [100000000, 95000000, 105000000]
        }, index=pd.date_range(start='2023-01-01', periods=3))
        
        self.sample_market_data = {
            'indices': {
                'S&P 500': {
                    'price': 4500.0,
                    'change': 15.0,
                    'change_percent': 0.33
                },
                'NASDAQ': {
                    'price': 14000.0,
                    'change': 50.0,
                    'change_percent': 0.36
                }
            },
            'sectors': {
                'Technology': 1.2,
                'Healthcare': 0.5,
                'Financials': -0.3,
                'Consumer Discretionary': 0.8,
                'Energy': -1.0
            },
            'commodities': {
                'Crude Oil': {
                    'price': 75.0,
                    'change': -0.5,
                    'change_percent': -0.66
                },
                'Gold': {
                    'price': 1850.0,
                    'change': 10.0,
                    'change_percent': 0.54
                }
            },
            'forex': {
                'EUR/USD': {
                    'price': 1.08,
                    'change': 0.002,
                    'change_percent': 0.19
                },
                'USD/JPY': {
                    'price': 145.5,
                    'change': -0.3,
                    'change_percent': -0.21
                }
            },
            'economic_indicators': {
                'US GDP Growth': {
                    'value': 2.1,
                    'previous': 2.0,
                    'change': 0.1
                },
                'US Inflation': {
                    'value': 3.2,
                    'previous': 3.4,
                    'change': -0.2
                },
                'US Unemployment': {
                    'value': 3.8,
                    'previous': 3.7,
                    'change': 0.1
                }
            }
        }
        
        self.sample_news = [
            {
                'title': 'Apple Announces New iPhone',
                'source': 'Tech News',
                'url': 'https://example.com/news/1',
                'published_at': '2023-01-15T12:30:00Z',
                'summary': 'Apple has announced the new iPhone model.',
                'sentiment': 0.75,
                'impact': 'high',
                'categories': ['product', 'technology']
            },
            {
                'title': 'Apple Reports Record Earnings',
                'source': 'Financial News',
                'url': 'https://example.com/news/2',
                'published_at': '2023-01-10T15:45:00Z',
                'summary': 'Apple reported record earnings for the quarter.',
                'sentiment': 0.85,
                'impact': 'high',
                'categories': ['earnings', 'financial']
            }
        ]
        
        self.sample_economic_calendar = [
            {
                'event': 'US Fed Interest Rate Decision',
                'date': '2023-02-01T14:00:00Z',
                'importance': 'high',
                'previous': '5.25%',
                'forecast': '5.25%',
                'actual': None
            },
            {
                'event': 'US Non-Farm Payrolls',
                'date': '2023-02-03T13:30:00Z',
                'importance': 'high',
                'previous': '216K',
                'forecast': '180K',
                'actual': None
            }
        ]
    
    def test_enhanced_stock_data_service_integration(self):
        """Test EnhancedStockDataService integration with data integration service."""
        # Set up mock responses
        self.mock_integration_service.get_security_info.return_value = self.sample_security_info
        self.mock_integration_service.get_technical_indicators.return_value = self.sample_technical_indicators
        self.mock_integration_service.get_analyst_data.return_value = self.sample_analyst_data
        self.mock_integration_service.get_historical_prices.return_value = self.sample_historical_data
        
        # Test get_enhanced_security_info
        result = self.enhanced_stock_service.get_enhanced_security_info('AAPL')
        
        # Verify result is correct type
        self.assertIsInstance(result, EnhancedSecurityInfo)
        
        # Verify data was correctly mapped
        self.assertEqual(result.symbol, 'AAPL')
        self.assertEqual(result.name, 'Apple Inc.')
        self.assertEqual(result.current_price, 150.0)
        self.assertEqual(result.rsi_14, 65.3)
        self.assertEqual(result.analyst_rating, 'Buy')
        
        # Verify integration service was called
        self.mock_integration_service.get_security_info.assert_called_once_with('AAPL')
        self.mock_integration_service.get_technical_indicators.assert_called_once()
        self.mock_integration_service.get_analyst_data.assert_called_once_with('AAPL')
    
    def test_enhanced_stock_data_service_technical_indicators(self):
        """Test EnhancedStockDataService technical indicators integration."""
        # Set up mock response
        self.mock_integration_service.get_technical_indicators.return_value = self.sample_technical_indicators
        
        # Test get_technical_indicators
        result = self.enhanced_stock_service.get_technical_indicators('AAPL', ['sma_20', 'rsi_14'])
        
        # Verify result
        self.assertEqual(result['sma_20'], 148.5)
        self.assertEqual(result['rsi_14'], 65.3)
        
        # Verify integration service was called
        self.mock_integration_service.get_technical_indicators.assert_called_once_with('AAPL', ['sma_20', 'rsi_14'])
    
    def test_enhanced_stock_data_service_historical_analysis(self):
        """Test EnhancedStockDataService historical analysis integration."""
        # Set up mock response
        self.mock_integration_service.get_historical_prices.return_value = self.sample_historical_data
        
        # Test get_price_history
        result = self.enhanced_stock_service.get_price_history('AAPL', period='1y', interval='1d')
        
        # Verify result
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)
        
        # Test calculate_returns
        returns = self.enhanced_stock_service.calculate_returns('AAPL', period='1y')
        
        # Verify returns calculation
        self.assertIsInstance(returns, dict)
        self.assertIn('total_return', returns)
        self.assertIn('annualized_return', returns)
        
        # Verify integration service was called
        self.mock_integration_service.get_historical_prices.assert_called()
    
    def test_market_data_service_integration(self):
        """Test MarketDataService integration with data integration service."""
        # Set up mock response
        self.mock_integration_service.get_market_data.side_effect = lambda data_type, **kwargs: {
            'indices': self.sample_market_data['indices'],
            'sectors': self.sample_market_data['sectors'],
            'commodities': self.sample_market_data['commodities'],
            'forex': self.sample_market_data['forex'],
            'economic_indicators': self.sample_market_data['economic_indicators']
        }.get(data_type, {})
        
        # Test get_market_overview
        result = self.market_service.get_market_overview()
        
        # Verify result is correct type
        self.assertIsInstance(result, MarketData)
        
        # Verify data was correctly mapped
        self.assertEqual(result.indices['S&P 500']['price'], 4500.0)
        self.assertEqual(result.sectors['Technology'], 1.2)
        self.assertEqual(result.commodities['Gold']['price'], 1850.0)
        self.assertEqual(result.forex['EUR/USD']['price'], 1.08)
        self.assertEqual(result.economic_indicators['US GDP Growth']['value'], 2.1)
        
        # Verify integration service was called
        self.assertEqual(self.mock_integration_service.get_market_data.call_count, 5)
    
    def test_market_data_service_sector_performance(self):
        """Test MarketDataService sector performance integration."""
        # Set up mock response
        self.mock_integration_service.get_market_data.return_value = self.sample_market_data['sectors']
        
        # Test get_sector_performance
        result = self.market_service.get_sector_performance()
        
        # Verify result
        self.assertEqual(result['Technology'], 1.2)
        self.assertEqual(result['Energy'], -1.0)
        
        # Verify integration service was called
        self.mock_integration_service.get_market_data.assert_called_once_with('sectors')
    
    def test_market_data_service_economic_indicators(self):
        """Test MarketDataService economic indicators integration."""
        # Set up mock response
        self.mock_integration_service.get_market_data.return_value = self.sample_market_data['economic_indicators']
        
        # Test get_economic_indicators
        result = self.market_service.get_economic_indicators()
        
        # Verify result
        self.assertEqual(result['US GDP Growth']['value'], 2.1)
        self.assertEqual(result['US Inflation']['value'], 3.2)
        
        # Verify integration service was called
        self.mock_integration_service.get_market_data.assert_called_once_with('economic_indicators')
    
    def test_news_service_integration(self):
        """Test NewsService integration with data integration service."""
        # Set up mock responses
        self.mock_integration_service.get_news.return_value = self.sample_news
        self.mock_integration_service.get_economic_calendar.return_value = self.sample_economic_calendar
        
        # Test get_company_news
        company_news = self.news_service.get_company_news('AAPL', limit=2)
        
        # Verify result
        self.assertEqual(len(company_news), 2)
        self.assertIsInstance(company_news[0], NewsItem)
        self.assertEqual(company_news[0].title, 'Apple Announces New iPhone')
        self.assertEqual(company_news[0].sentiment, 0.75)
        
        # Test get_market_news
        market_news = self.news_service.get_market_news(limit=2)
        
        # Verify result
        self.assertEqual(len(market_news), 2)
        
        # Test get_economic_calendar
        calendar = self.news_service.get_economic_calendar(days=7)
        
        # Verify result
        self.assertEqual(len(calendar), 2)
        self.assertEqual(calendar[0]['event'], 'US Fed Interest Rate Decision')
        
        # Verify integration service was called
        self.mock_integration_service.get_news.assert_called()
        self.mock_integration_service.get_economic_calendar.assert_called_once_with(days=7)
    
    def test_news_service_sentiment_analysis(self):
        """Test NewsService sentiment analysis integration."""
        # Set up mock response
        self.mock_integration_service.get_news.return_value = self.sample_news
        
        # Test get_sentiment_analysis
        sentiment = self.news_service.get_sentiment_analysis('AAPL')
        
        # Verify result
        self.assertIsInstance(sentiment, dict)
        self.assertIn('average_sentiment', sentiment)
        self.assertIn('sentiment_distribution', sentiment)
        
        # Verify integration service was called
        self.mock_integration_service.get_news.assert_called_once_with('AAPL', None, 20)
    
    def test_service_error_handling(self):
        """Test error handling in services."""
        # Set up mock to raise error
        self.mock_integration_service.get_security_info.side_effect = DataRetrievalError("Test error")
        
        # Test error handling
        with self.assertRaises(DataRetrievalError):
            self.enhanced_stock_service.get_enhanced_security_info('AAPL')
        
        # Verify integration service was called
        self.mock_integration_service.get_security_info.assert_called_once_with('AAPL')
    
    def test_service_data_transformation(self):
        """Test data transformation in services."""
        # Set up mock responses
        self.mock_integration_service.get_security_info.return_value = self.sample_security_info
        self.mock_integration_service.get_technical_indicators.return_value = self.sample_technical_indicators
        self.mock_integration_service.get_analyst_data.return_value = self.sample_analyst_data
        
        # Test data transformation
        result = self.enhanced_stock_service.get_enhanced_security_info('AAPL')
        
        # Verify data transformation
        self.assertIsInstance(result, EnhancedSecurityInfo)
        self.assertEqual(result.symbol, 'AAPL')
        self.assertEqual(result.current_price, 150.0)
        self.assertEqual(result.rsi_14, 65.3)
        self.assertEqual(result.analyst_rating, 'Buy')
        self.assertEqual(result.price_target['average'], 170.0)


if __name__ == '__main__':
    unittest.main()