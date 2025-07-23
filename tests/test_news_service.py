"""Tests for the news service.

This module contains tests for the NewsService class.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import pytest

from stock_analysis.services.news_service import NewsService
from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
from stock_analysis.models.enhanced_data_models import NewsItem
from stock_analysis.models.data_models import SentimentResult
from stock_analysis.utils.exceptions import DataRetrievalError


class TestNewsService:
    """Test cases for the NewsService class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a mock integration service
        self.mock_integration_service = Mock(spec=FinancialDataIntegrationService)
        
        # Create the service with the mock integration service
        self.service = NewsService(integration_service=self.mock_integration_service)
        
        # Mock cache manager
        self.mock_cache = MagicMock()
        self.mock_cache.get.return_value = None
        
        # Sample data
        self.sample_symbol = "AAPL"
        
        # Sample news data
        self.sample_news_data = [
            {
                'title': 'Company Reports Strong Quarterly Earnings',
                'source': 'test_source',
                'url': 'https://example.com/news1',
                'published_at': datetime.now() - timedelta(days=1),
                'summary': 'The company exceeded expectations with record profits and strong growth.',
                'sentiment': 0.8,
                'impact': 'high',
                'categories': ['earnings', 'market']
            },
            {
                'title': 'Stock Price Drops on Market Concerns',
                'source': 'test_source',
                'url': 'https://example.com/news2',
                'published_at': datetime.now() - timedelta(days=2),
                'summary': 'Shares fell sharply due to broader market volatility and investor concerns.',
                'sentiment': -0.6,
                'impact': 'medium',
                'categories': ['market']
            },
            {
                'title': 'Company Announces New Product Launch',
                'source': 'test_source',
                'url': 'https://example.com/news3',
                'published_at': datetime.now() - timedelta(days=3),
                'summary': 'The company unveiled its latest innovation in a competitive market.',
                'sentiment': 0.5,
                'impact': 'medium',
                'categories': ['product']
            }
        ]
        
        # Sample economic calendar data
        self.sample_economic_calendar = [
            {
                'event_name': 'GDP Growth Rate',
                'date': datetime.now() + timedelta(days=1),
                'country': 'United States',
                'importance': 'high',
                'actual': None,
                'forecast': '2.0%',
                'previous': '1.9%'
            },
            {
                'event_name': 'Unemployment Rate',
                'date': datetime.now() - timedelta(days=1),
                'country': 'United States',
                'importance': 'high',
                'actual': '3.7%',
                'forecast': '3.8%',
                'previous': '3.8%'
            },
            {
                'event_name': 'Interest Rate Decision',
                'date': datetime.now() + timedelta(days=2),
                'country': 'United States',
                'importance': 'high',
                'actual': None,
                'forecast': '5.25%',
                'previous': '5.25%'
            }
        ]
        
        # Sample sentiment result
        self.sample_sentiment_result = SentimentResult(
            overall_sentiment=0.3,
            positive_count=2,
            negative_count=1,
            neutral_count=0,
            key_themes=['earnings', 'growth'],
            sentiment_trend=[0.2, 0.3, 0.4]
        )
    
    @patch("stock_analysis.services.news_service.get_cache_manager")
    def test_get_company_news(self, mock_get_cache):
        """Test retrieving company news."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_company_news.return_value = self.sample_news_data
        
        # Execute
        result = self.service.get_company_news(self.sample_symbol)
        
        # Verify
        assert len(result) == 3
        assert all(isinstance(item, NewsItem) for item in result)
        assert result[0].title == 'Company Reports Strong Quarterly Earnings'
        assert result[0].sentiment == 0.8
        assert result[0].impact == 'high'
        assert 'earnings' in result[0].categories
        
        # Verify integration service was called correctly
        self.mock_integration_service.get_company_news.assert_called_once_with(
            symbol=self.sample_symbol,
            days=7,
            limit=10,
            use_cache=True
        )
        
        # Verify cache was checked and set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_called_once()
    
    @patch("stock_analysis.services.news_service.get_cache_manager")
    def test_get_company_news_from_cache(self, mock_get_cache):
        """Test retrieving company news from cache."""
        # Setup
        cached_result = [
            NewsItem(
                title='Cached News',
                source='cache',
                url='https://example.com/cached',
                published_at=datetime.now(),
                summary='Cached news summary'
            )
        ]
        self.mock_cache.get.return_value = cached_result
        mock_get_cache.return_value = self.mock_cache
        
        # Execute
        result = self.service.get_company_news(self.sample_symbol)
        
        # Verify
        assert result is cached_result
        
        # Verify integration service was not called
        self.mock_integration_service.get_company_news.assert_not_called()
        
        # Verify cache was checked but not set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_not_called()
    
    @patch("stock_analysis.services.news_service.get_cache_manager")
    def test_get_company_news_error(self, mock_get_cache):
        """Test error handling when retrieving company news."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_company_news.side_effect = Exception("API error")
        
        # Execute and verify
        with pytest.raises(DataRetrievalError) as excinfo:
            self.service.get_company_news(self.sample_symbol)
        
        # Verify error message
        assert "Failed to retrieve company news" in str(excinfo.value)
        assert self.sample_symbol in str(excinfo.value)
        
        # Verify integration service was called
        self.mock_integration_service.get_company_news.assert_called_once()
    
    @patch("stock_analysis.services.news_service.get_cache_manager")
    def test_get_market_news(self, mock_get_cache):
        """Test retrieving market news."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_market_news.return_value = self.sample_news_data
        
        # Execute
        result = self.service.get_market_news()
        
        # Verify
        assert len(result) == 3
        assert all(isinstance(item, NewsItem) for item in result)
        
        # Verify integration service was called correctly
        self.mock_integration_service.get_market_news.assert_called_once_with(
            category=None,
            days=3,
            limit=10,
            use_cache=True
        )
        
        # Verify cache was checked and set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_called_once()
    
    @patch("stock_analysis.services.news_service.get_cache_manager")
    def test_get_market_news_with_category(self, mock_get_cache):
        """Test retrieving market news with category filter."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_market_news.return_value = self.sample_news_data
        
        # Execute
        result = self.service.get_market_news(category='market')
        
        # Verify
        assert len(result) == 2  # Only 2 items have 'market' category
        assert all(isinstance(item, NewsItem) for item in result)
        assert all('market' in item.categories for item in result)
        
        # Verify integration service was called correctly
        self.mock_integration_service.get_market_news.assert_called_once_with(
            category='market',
            days=3,
            limit=10,
            use_cache=True
        )
    
    @patch("stock_analysis.services.news_service.get_cache_manager")
    def test_get_economic_calendar(self, mock_get_cache):
        """Test retrieving economic calendar."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_economic_calendar.return_value = self.sample_economic_calendar
        
        # Execute
        result = self.service.get_economic_calendar()
        
        # Verify
        assert len(result) == 3
        assert result[0]['event_name'] == 'GDP Growth Rate'
        assert result[1]['event_name'] == 'Unemployment Rate'
        assert result[2]['event_name'] == 'Interest Rate Decision'
        
        # Verify integration service was called correctly
        self.mock_integration_service.get_economic_calendar.assert_called_once_with(
            days_ahead=7,
            days_behind=1,
            use_cache=True
        )
        
        # Verify cache was checked and set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_called_once()
    
    @patch("stock_analysis.services.news_service.get_cache_manager")
    def test_get_economic_calendar_with_filters(self, mock_get_cache):
        """Test retrieving economic calendar with filters."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        self.mock_integration_service.get_economic_calendar.return_value = self.sample_economic_calendar
        
        # Execute
        result = self.service.get_economic_calendar(
            days_ahead=5,
            days_behind=2,
            country='United States',
            importance='high'
        )
        
        # Verify
        assert len(result) == 3  # All sample events are for US and high importance
        
        # Verify integration service was called correctly
        self.mock_integration_service.get_economic_calendar.assert_called_once_with(
            days_ahead=5,
            days_behind=2,
            use_cache=True
        )
    
    @patch("stock_analysis.services.news_service.get_news_and_sentiment")
    @patch("stock_analysis.services.news_service.get_cache_manager")
    def test_get_news_sentiment(self, mock_get_cache, mock_get_news_and_sentiment):
        """Test retrieving news sentiment."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        mock_get_news_and_sentiment.return_value = (self.sample_news_data, self.sample_sentiment_result)
        
        # Execute
        result = self.service.get_news_sentiment(self.sample_symbol)
        
        # Verify
        assert isinstance(result, SentimentResult)
        assert result.overall_sentiment == 0.3
        assert result.positive_count == 2
        assert result.negative_count == 1
        
        # Verify get_news_and_sentiment was called correctly
        mock_get_news_and_sentiment.assert_called_once_with(
            self.sample_symbol,
            days=7,
            use_cache=True
        )
        
        # Verify cache was checked and set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_called_once()
    
    @patch("stock_analysis.services.news_service.get_cache_manager")
    def test_get_trending_topics(self, mock_get_cache):
        """Test retrieving trending topics."""
        # Setup
        mock_get_cache.return_value = self.mock_cache
        
        # Mock get_market_news to return sample news data
        self.service.get_market_news = Mock(return_value=[
            NewsItem(
                title='News 1',
                source='source1',
                url='https://example.com/1',
                published_at=datetime.now(),
                summary='Summary 1',
                sentiment=0.5,
                impact='medium',
                categories=['market', 'economy']
            ),
            NewsItem(
                title='News 2',
                source='source2',
                url='https://example.com/2',
                published_at=datetime.now(),
                summary='Summary 2',
                sentiment=-0.3,
                impact='low',
                categories=['market', 'earnings']
            ),
            NewsItem(
                title='News 3',
                source='source3',
                url='https://example.com/3',
                published_at=datetime.now(),
                summary='Summary 3',
                sentiment=0.2,
                impact='high',
                categories=['economy', 'policy']
            )
        ])
        
        # Execute
        result = self.service.get_trending_topics()
        
        # Verify
        assert len(result) > 0
        assert isinstance(result[0], dict)
        assert 'topic' in result[0]
        assert 'count' in result[0]
        assert 'sentiment' in result[0]
        
        # Market should be the most frequent topic
        assert result[0]['topic'] == 'market'
        assert result[0]['count'] == 2
        
        # Verify get_market_news was called correctly
        self.service.get_market_news.assert_called_once_with(
            days=3,
            limit=100,
            include_sentiment=True,
            use_cache=True
        )
        
        # Verify cache was checked and set
        self.mock_cache.get.assert_called_once()
        self.mock_cache.set.assert_called_once()
    
    def test_process_news_data(self):
        """Test processing raw news data."""
        # Setup
        raw_news_data = [
            {
                'title': 'Test News',
                'source': 'test_source',
                'url': 'https://example.com',
                'published_at': datetime.now(),
                'summary': 'Test summary'
                # No sentiment, categories, or impact
            }
        ]
        
        # Execute with sentiment analysis
        with patch("stock_analysis.services.news_service.analyze_news_sentiment") as mock_analyze:
            with patch("stock_analysis.services.news_service.categorize_news") as mock_categorize:
                with patch("stock_analysis.services.news_service.assess_news_impact") as mock_assess:
                    result = self.service._process_news_data(raw_news_data, include_sentiment=True)
                    
                    # Verify sentiment analysis, categorization, and impact assessment were called
                    mock_analyze.assert_called_once()
                    mock_categorize.assert_called_once()
                    mock_assess.assert_called_once()
        
        # Verify result
        assert len(result) == 1
        assert isinstance(result[0], NewsItem)
        assert result[0].title == 'Test News'
        assert result[0].source == 'test_source'
        
        # Execute without sentiment analysis
        with patch("stock_analysis.services.news_service.analyze_news_sentiment") as mock_analyze:
            with patch("stock_analysis.services.news_service.categorize_news") as mock_categorize:
                with patch("stock_analysis.services.news_service.assess_news_impact") as mock_assess:
                    result = self.service._process_news_data(raw_news_data, include_sentiment=False)
                    
                    # Verify sentiment analysis was not called
                    mock_analyze.assert_not_called()
                    # Categorization and impact assessment should still be called
                    mock_categorize.assert_called_once()
                    mock_assess.assert_called_once()
    
    def test_process_news_data_with_existing_fields(self):
        """Test processing news data that already has sentiment, categories, and impact."""
        # Setup
        news_data = [
            {
                'title': 'Test News',
                'source': 'test_source',
                'url': 'https://example.com',
                'published_at': datetime.now(),
                'summary': 'Test summary',
                'sentiment': 0.5,
                'categories': ['test'],
                'impact': 'medium'
            }
        ]
        
        # Execute
        with patch("stock_analysis.services.news_service.analyze_news_sentiment") as mock_analyze:
            with patch("stock_analysis.services.news_service.categorize_news") as mock_categorize:
                with patch("stock_analysis.services.news_service.assess_news_impact") as mock_assess:
                    result = self.service._process_news_data(news_data, include_sentiment=True)
                    
                    # Verify none of the processing functions were called
                    mock_analyze.assert_not_called()
                    mock_categorize.assert_not_called()
                    mock_assess.assert_not_called()
        
        # Verify result
        assert len(result) == 1
        assert isinstance(result[0], NewsItem)
        assert result[0].sentiment == 0.5
        assert result[0].categories == ['test']
        assert result[0].impact == 'medium'