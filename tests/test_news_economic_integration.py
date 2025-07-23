"""Tests for news and economic data integration."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd

from stock_analysis.services.news_economic_integration import (
    NewsItem, EconomicEvent, validate_news_data, validate_economic_data,
    combine_news_data, combine_economic_data, analyze_news_sentiment,
    categorize_news, assess_news_impact, get_news_and_sentiment,
    get_economic_calendar
)
from stock_analysis.models.data_models import SentimentResult


class TestNewsEconomicIntegration:
    """Test cases for news and economic data integration."""
    
    @pytest.fixture
    def sample_news_items(self):
        """Create sample news items for testing."""
        return [
            {
                'title': 'Company Reports Strong Quarterly Earnings',
                'source': 'test_source',
                'url': 'https://example.com/news1',
                'published_at': datetime.now() - timedelta(days=1),
                'summary': 'The company exceeded expectations with record profits and strong growth.'
            },
            {
                'title': 'Stock Price Drops on Market Concerns',
                'source': 'test_source',
                'url': 'https://example.com/news2',
                'published_at': datetime.now() - timedelta(days=2),
                'summary': 'Shares fell sharply due to broader market volatility and investor concerns.'
            },
            {
                'title': 'Company Announces New Product Launch',
                'source': 'test_source',
                'url': 'https://example.com/news3',
                'published_at': datetime.now() - timedelta(days=3),
                'summary': 'The company unveiled its latest innovation in a competitive market.'
            }
        ]
    
    @pytest.fixture
    def sample_economic_events(self):
        """Create sample economic events for testing."""
        return [
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
    
    def test_news_item_creation(self):
        """Test NewsItem creation and conversion."""
        news_item = NewsItem(
            title='Test News',
            source='test_source',
            url='https://example.com',
            published_at=datetime.now(),
            summary='Test summary',
            sentiment=0.5,
            impact='medium',
            categories=['earnings', 'market']
        )
        
        assert news_item.title == 'Test News'
        assert news_item.source == 'test_source'
        assert news_item.url == 'https://example.com'
        assert isinstance(news_item.published_at, datetime)
        assert news_item.summary == 'Test summary'
        assert news_item.sentiment == 0.5
        assert news_item.impact == 'medium'
        assert news_item.categories == ['earnings', 'market']
        
        # Test to_dict and from_dict
        news_dict = news_item.to_dict()
        assert isinstance(news_dict, dict)
        
        recreated_item = NewsItem.from_dict(news_dict)
        assert recreated_item.title == news_item.title
        assert recreated_item.sentiment == news_item.sentiment
        assert recreated_item.impact == news_item.impact
    
    def test_economic_event_creation(self):
        """Test EconomicEvent creation and conversion."""
        event = EconomicEvent(
            event_name='GDP Growth Rate',
            date=datetime.now(),
            country='United States',
            importance='high',
            actual='2.1%',
            forecast='2.0%',
            previous='1.9%'
        )
        
        assert event.event_name == 'GDP Growth Rate'
        assert event.country == 'United States'
        assert isinstance(event.date, datetime)
        assert event.importance == 'high'
        assert event.actual == '2.1%'
        assert event.forecast == '2.0%'
        assert event.previous == '1.9%'
        
        # Test to_dict and from_dict
        event_dict = event.to_dict()
        assert isinstance(event_dict, dict)
        
        recreated_event = EconomicEvent.from_dict(event_dict)
        assert recreated_event.event_name == event.event_name
        assert recreated_event.importance == event.importance
        assert recreated_event.actual == event.actual
    
    def test_validate_news_data_valid(self, sample_news_items):
        """Test news data validation with valid data."""
        assert validate_news_data(sample_news_items) is True
    
    def test_validate_news_data_invalid(self):
        """Test news data validation with invalid data."""
        # Empty list
        assert validate_news_data([]) is False
        
        # Missing required fields
        invalid_items = [
            {
                'title': 'Test News',
                # Missing source
                'url': 'https://example.com',
                'published_at': datetime.now()
            }
        ]
        assert validate_news_data(invalid_items) is False
    
    def test_validate_economic_data_valid(self, sample_economic_events):
        """Test economic data validation with valid data."""
        assert validate_economic_data(sample_economic_events) is True
    
    def test_validate_economic_data_invalid(self):
        """Test economic data validation with invalid data."""
        # Empty list
        assert validate_economic_data([]) is False
        
        # Missing required fields
        invalid_events = [
            {
                'event_name': 'GDP Growth Rate',
                # Missing date
                'country': 'United States',
                'importance': 'high'
            }
        ]
        assert validate_economic_data(invalid_events) is False
    
    def test_combine_news_data(self, sample_news_items):
        """Test combining news data from multiple sources."""
        source1_items = sample_news_items[:2]
        source2_items = [sample_news_items[2]]
        
        # Add a duplicate item
        duplicate_item = sample_news_items[0].copy()
        source2_items.append(duplicate_item)
        
        partial_results = {
            'source1': source1_items,
            'source2': source2_items
        }
        
        combined = combine_news_data(partial_results)
        
        # Should have 3 unique items
        assert len(combined) == 3
        
        # Should be sorted by published_at (most recent first)
        assert combined[0]['published_at'] > combined[1]['published_at']
    
    def test_combine_economic_data(self, sample_economic_events):
        """Test combining economic data from multiple sources."""
        source1_events = sample_economic_events[:2]
        source2_events = [sample_economic_events[2]]
        
        # Add an event with the same name and date but different details
        duplicate_event = sample_economic_events[0].copy()
        duplicate_event['actual'] = '2.2%'  # Different actual value
        source2_events.append(duplicate_event)
        
        partial_results = {
            'source1': source1_events,
            'source2': source2_events
        }
        
        combined = combine_economic_data(partial_results)
        
        # Should have 3 unique events
        assert len(combined) == 3
        
        # Should be sorted by date (upcoming events first)
        assert combined[0]['date'] < combined[1]['date'] < combined[2]['date']
    
    @patch('stock_analysis.services.news_economic_integration.NewsSentimentAnalyzer')
    def test_analyze_news_sentiment(self, mock_analyzer, sample_news_items):
        """Test news sentiment analysis."""
        # Mock the sentiment analyzer
        mock_instance = Mock()
        mock_instance.analyze_sentiment.return_value = SentimentResult(
            overall_sentiment=0.3,
            positive_count=2,
            negative_count=1,
            neutral_count=0,
            key_themes=['earnings', 'growth'],
            sentiment_trend=[0.2, 0.3, 0.4]
        )
        mock_instance._analyze_article_sentiment.return_value = 0.5
        mock_analyzer.return_value = mock_instance
        
        result = analyze_news_sentiment(sample_news_items)
        
        assert isinstance(result, SentimentResult)
        assert result.overall_sentiment == 0.3
        assert result.positive_count == 2
        assert result.negative_count == 1
        
        # Should update the original news items with sentiment
        assert 'sentiment' in sample_news_items[0]
        assert sample_news_items[0]['sentiment'] == 0.5
    
    def test_categorize_news(self, sample_news_items):
        """Test news categorization."""
        categorized = categorize_news(sample_news_items)
        
        # All items should have categories
        assert all('categories' in item for item in categorized)
        
        # First item should be categorized as 'earnings'
        assert 'earnings' in categorized[0]['categories']
        
        # Second item should be categorized as 'market'
        assert 'market' in categorized[1]['categories']
    
    def test_assess_news_impact(self, sample_news_items):
        """Test news impact assessment."""
        # Add sentiment scores
        sample_news_items[0]['sentiment'] = 0.8  # Strong positive
        sample_news_items[1]['sentiment'] = -0.7  # Strong negative
        sample_news_items[2]['sentiment'] = 0.1  # Neutral
        
        assessed = assess_news_impact(sample_news_items)
        
        # All items should have impact assessment
        assert all('impact' in item for item in assessed)
        
        # First item should have high impact (contains 'record')
        assert assessed[0]['impact'] == 'high'
        
        # Second item should have high impact (contains 'sharply')
        assert assessed[1]['impact'] in ['high', 'medium']
    
    @patch('stock_analysis.services.news_economic_integration.NewsSentimentAnalyzer')
    def test_get_news_and_sentiment(self, mock_analyzer):
        """Test getting news and sentiment for a symbol."""
        # Mock the news sentiment analyzer
        mock_instance = Mock()
        mock_instance.get_news_articles.return_value = [
            Mock(title='Test News', content='Test content', url='https://example.com',
                 published_date=datetime.now(), source='test')
        ]
        mock_instance.analyze_sentiment.return_value = SentimentResult(
            overall_sentiment=0.3,
            positive_count=1,
            negative_count=0,
            neutral_count=0,
            key_themes=['test'],
            sentiment_trend=[0.3]
        )
        mock_instance._analyze_article_sentiment.return_value = 0.3
        mock_analyzer.return_value = mock_instance
        
        news_items, sentiment = get_news_and_sentiment('AAPL', days=7)
        
        assert isinstance(news_items, list)
        assert len(news_items) == 1
        assert 'sentiment' in news_items[0]
        assert 'categories' in news_items[0]
        assert 'impact' in news_items[0]
        
        assert isinstance(sentiment, SentimentResult)
        assert sentiment.overall_sentiment == 0.3
    
    def test_get_economic_calendar(self):
        """Test getting economic calendar events."""
        events = get_economic_calendar(days_ahead=7, days_behind=1)
        
        assert isinstance(events, list)
        assert len(events) > 0
        
        # Check that we have both past and future events
        past_events = [e for e in events if e['date'] < datetime.now()]
        future_events = [e for e in events if e['date'] >= datetime.now()]
        
        assert len(past_events) > 0
        assert len(future_events) > 0
        
        # Some past events should have actual values (not all might have them)
        past_with_actual = [e for e in past_events if e['actual'] is not None]
        assert len(past_with_actual) > 0
        
        # Future events should not have actual values
        assert all(e['actual'] is None for e in future_events)