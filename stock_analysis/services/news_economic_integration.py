"""News and economic data integration functionality.

This module provides functions for integrating news and economic data from multiple sources,
including sentiment analysis, categorization, and impact assessment.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import pandas as pd

from stock_analysis.analyzers.news_sentiment_analyzer import NewsSentimentAnalyzer, NewsArticle
from stock_analysis.models.data_models import SentimentResult
from stock_analysis.utils.logging import get_logger
from stock_analysis.utils.exceptions import DataRetrievalError

logger = get_logger(__name__)


class NewsItem:
    """Data model for a news article with metadata."""
    
    def __init__(self, title: str, source: str, url: str, published_at: datetime, 
                 summary: str, sentiment: Optional[float] = None, 
                 impact: Optional[str] = None, categories: Optional[List[str]] = None):
        """Initialize a NewsItem.
        
        Args:
            title: News article title
            source: Source of the news article
            url: URL to the news article
            published_at: Publication date and time
            summary: Summary or content of the article
            sentiment: Sentiment score (-1.0 to 1.0)
            impact: Impact assessment (high, medium, low)
            categories: List of categories for the article
        """
        self.title = title
        self.source = source
        self.url = url
        self.published_at = published_at
        self.summary = summary
        self.sentiment = sentiment
        self.impact = impact
        self.categories = categories or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert NewsItem to dictionary.
        
        Returns:
            Dictionary representation of the NewsItem
        """
        return {
            'title': self.title,
            'source': self.source,
            'url': self.url,
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'summary': self.summary,
            'sentiment': self.sentiment,
            'impact': self.impact,
            'categories': self.categories
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsItem':
        """Create NewsItem from dictionary.
        
        Args:
            data: Dictionary representation of a NewsItem
            
        Returns:
            NewsItem instance
        """
        published_at = None
        if data.get('published_at'):
            try:
                if isinstance(data['published_at'], str):
                    published_at = datetime.fromisoformat(data['published_at'])
                else:
                    published_at = data['published_at']
            except (ValueError, TypeError):
                published_at = None
        
        return cls(
            title=data.get('title', ''),
            source=data.get('source', ''),
            url=data.get('url', ''),
            published_at=published_at,
            summary=data.get('summary', ''),
            sentiment=data.get('sentiment'),
            impact=data.get('impact'),
            categories=data.get('categories', [])
        )


class EconomicEvent:
    """Data model for an economic event or indicator."""
    
    def __init__(self, event_name: str, date: datetime, country: str, 
                 importance: str, actual: Optional[str] = None, 
                 forecast: Optional[str] = None, previous: Optional[str] = None):
        """Initialize an EconomicEvent.
        
        Args:
            event_name: Name of the economic event or indicator
            date: Date and time of the event
            country: Country or region the event relates to
            importance: Importance level (high, medium, low)
            actual: Actual value or result
            forecast: Forecasted value
            previous: Previous value
        """
        self.event_name = event_name
        self.date = date
        self.country = country
        self.importance = importance
        self.actual = actual
        self.forecast = forecast
        self.previous = previous
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert EconomicEvent to dictionary.
        
        Returns:
            Dictionary representation of the EconomicEvent
        """
        return {
            'event_name': self.event_name,
            'date': self.date.isoformat() if self.date else None,
            'country': self.country,
            'importance': self.importance,
            'actual': self.actual,
            'forecast': self.forecast,
            'previous': self.previous
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EconomicEvent':
        """Create EconomicEvent from dictionary.
        
        Args:
            data: Dictionary representation of an EconomicEvent
            
        Returns:
            EconomicEvent instance
        """
        date = None
        if data.get('date'):
            try:
                if isinstance(data['date'], str):
                    date = datetime.fromisoformat(data['date'])
                else:
                    date = data['date']
            except (ValueError, TypeError):
                date = None
        
        return cls(
            event_name=data.get('event_name', ''),
            date=date,
            country=data.get('country', ''),
            importance=data.get('importance', 'medium'),
            actual=data.get('actual'),
            forecast=data.get('forecast'),
            previous=data.get('previous')
        )


def validate_news_data(news_items: List[Dict[str, Any]]) -> bool:
    """Validate news data.
    
    Args:
        news_items: List of news items
        
    Returns:
        True if data is valid and complete, False otherwise
    """
    if not news_items:
        return False
    
    # Check if we have enough items
    if len(news_items) < 1:
        return False
    
    # Check required fields for each item
    required_fields = ['title', 'source', 'published_at']
    
    for item in news_items:
        # Check required fields
        for field in required_fields:
            if field not in item or item[field] is None:
                return False
    
    return True


def validate_economic_data(economic_events: List[Dict[str, Any]]) -> bool:
    """Validate economic data.
    
    Args:
        economic_events: List of economic events
        
    Returns:
        True if data is valid and complete, False otherwise
    """
    if not economic_events:
        return False
    
    # Check if we have enough items
    if len(economic_events) < 1:
        return False
    
    # Check required fields for each item
    required_fields = ['event_name', 'date', 'importance']
    
    for event in economic_events:
        # Check required fields
        for field in required_fields:
            if field not in event or event[field] is None:
                return False
    
    return True


def combine_news_data(partial_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Combine news data from multiple sources.
    
    Args:
        partial_results: Dictionary mapping source names to partial results
        
    Returns:
        Combined list of news items
    """
    if not partial_results:
        return []
    
    combined = []
    seen_urls = set()
    
    # Process each source
    for source, items in partial_results.items():
        for item in items:
            # Add source information if not present
            if 'source' not in item or not item['source']:
                item['source'] = source
            
            # Check for duplicates based on URL
            url = item.get('url', '')
            if url and url in seen_urls:
                continue
            
            if url:
                seen_urls.add(url)
            
            combined.append(item)
    
    # Sort by published date (most recent first)
    combined.sort(key=lambda x: x.get('published_at', datetime.min), reverse=True)
    
    return combined


def combine_economic_data(partial_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Combine economic data from multiple sources.
    
    Args:
        partial_results: Dictionary mapping source names to partial results
        
    Returns:
        Combined list of economic events
    """
    if not partial_results:
        return []
    
    combined = []
    seen_events = {}  # Map of event_name+date to index in combined list
    
    # Process each source
    for source, events in partial_results.items():
        for event in events:
            # Create a unique key for the event
            event_date = event.get('date')
            event_name = event.get('event_name', '')
            
            if not event_date or not event_name:
                continue
            
            key = f"{event_name}_{event_date}"
            
            # If we've seen this event before, update with any missing information
            if key in seen_events:
                idx = seen_events[key]
                existing_event = combined[idx]
                
                # Update fields that might be missing in the existing event
                for field in ['actual', 'forecast', 'previous', 'importance', 'country']:
                    if field not in existing_event or existing_event[field] is None:
                        if field in event and event[field] is not None:
                            existing_event[field] = event[field]
            else:
                # Add new event
                combined.append(event)
                seen_events[key] = len(combined) - 1
    
    # Sort by date (upcoming events first)
    combined.sort(key=lambda x: x.get('date', datetime.max))
    
    return combined


def analyze_news_sentiment(news_items: List[Dict[str, Any]]) -> SentimentResult:
    """Analyze sentiment of news items.
    
    Args:
        news_items: List of news items
        
    Returns:
        SentimentResult with sentiment scores and analysis
    """
    logger.info(f"Analyzing sentiment for {len(news_items)} news items")
    
    # Convert news items to NewsArticle objects for sentiment analysis
    articles = []
    for item in news_items:
        article = NewsArticle(
            title=item.get('title', ''),
            content=item.get('summary', ''),
            url=item.get('url', ''),
            published_date=item.get('published_at', datetime.now()),
            source=item.get('source', '')
        )
        articles.append(article)
    
    # Use the existing sentiment analyzer
    analyzer = NewsSentimentAnalyzer()
    result = analyzer.analyze_sentiment(articles)
    
    # Update the original news items with sentiment scores
    for i, article in enumerate(articles):
        if i < len(news_items):
            news_items[i]['sentiment'] = analyzer._analyze_article_sentiment(article)
    
    return result


def categorize_news(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Categorize news items.
    
    Args:
        news_items: List of news items
        
    Returns:
        List of news items with categories added
    """
    logger.info(f"Categorizing {len(news_items)} news items")
    
    # Define category keywords
    categories = {
        'earnings': ['earnings', 'revenue', 'profit', 'loss', 'quarterly', 'financial results'],
        'market': ['market', 'index', 'indices', 'dow', 'nasdaq', 's&p', 'trading'],
        'economy': ['economy', 'economic', 'gdp', 'inflation', 'unemployment', 'fed', 'interest rate'],
        'merger': ['merger', 'acquisition', 'takeover', 'buyout', 'deal'],
        'product': ['product', 'launch', 'release', 'new', 'innovation'],
        'legal': ['lawsuit', 'legal', 'court', 'settlement', 'regulation', 'compliance'],
        'management': ['ceo', 'executive', 'management', 'board', 'director', 'leadership']
    }
    
    for item in news_items:
        item_categories = []
        text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
        
        # Check each category
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    item_categories.append(category)
                    break
        
        # If no categories found, add 'general'
        if not item_categories:
            item_categories.append('general')
        
        # Update the item
        item['categories'] = item_categories
    
    return news_items


def assess_news_impact(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assess impact of news items.
    
    Args:
        news_items: List of news items
        
    Returns:
        List of news items with impact assessment added
    """
    logger.info(f"Assessing impact for {len(news_items)} news items")
    
    # Define impact keywords
    high_impact_keywords = [
        'significant', 'major', 'critical', 'substantial', 'dramatic',
        'breakthrough', 'record', 'massive', 'huge', 'extraordinary',
        'crash', 'surge', 'plunge', 'soar', 'collapse'
    ]
    
    medium_impact_keywords = [
        'important', 'notable', 'considerable', 'moderate', 'meaningful',
        'rise', 'fall', 'increase', 'decrease', 'change',
        'update', 'announce', 'report', 'reveal'
    ]
    
    for item in news_items:
        text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
        sentiment = item.get('sentiment', 0)
        
        # Check for high impact keywords
        for keyword in high_impact_keywords:
            if keyword.lower() in text:
                item['impact'] = 'high'
                break
        
        # If no high impact, check for medium impact
        if 'impact' not in item:
            for keyword in medium_impact_keywords:
                if keyword.lower() in text:
                    item['impact'] = 'medium'
                    break
        
        # If still no impact assigned, use sentiment to determine
        if 'impact' not in item:
            if abs(sentiment) > 0.5:
                item['impact'] = 'medium'
            else:
                item['impact'] = 'low'
    
    return news_items


def get_news_and_sentiment(symbol: str, days: int = 7, use_cache: bool = True) -> Tuple[List[Dict[str, Any]], SentimentResult]:
    """Get news and sentiment analysis for a symbol.
    
    Args:
        symbol: Stock ticker symbol
        days: Number of days to look back
        use_cache: Whether to use cached data
        
    Returns:
        Tuple of (news items, sentiment result)
    """
    logger.info(f"Getting news and sentiment for {symbol} (last {days} days)")
    
    # Get news articles
    analyzer = NewsSentimentAnalyzer()
    articles = analyzer.get_news_articles(symbol, days)
    
    # Convert to news items
    news_items = []
    for article in articles:
        news_item = {
            'title': article.title,
            'source': article.source,
            'url': article.url,
            'published_at': article.published_date,
            'summary': article.content
        }
        news_items.append(news_item)
    
    # Analyze sentiment
    sentiment_result = analyze_news_sentiment(news_items)
    
    # Categorize news
    news_items = categorize_news(news_items)
    
    # Assess impact
    news_items = assess_news_impact(news_items)
    
    return news_items, sentiment_result


def get_economic_calendar(days_ahead: int = 7, days_behind: int = 1, use_cache: bool = True) -> List[Dict[str, Any]]:
    """Get economic calendar events.
    
    Args:
        days_ahead: Number of days to look ahead
        days_behind: Number of days to look behind
        use_cache: Whether to use cached data
        
    Returns:
        List of economic events
    """
    logger.info(f"Getting economic calendar (looking {days_behind} days behind and {days_ahead} days ahead)")
    
    # This would typically call an adapter to get economic calendar data
    # For now, we'll return a placeholder implementation
    
    # Create some sample economic events
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    events = []
    
    # Sample events for demonstration
    sample_events = [
        {
            'event_name': 'GDP Growth Rate',
            'country': 'United States',
            'importance': 'high',
            'actual': '2.1%',
            'forecast': '2.0%',
            'previous': '1.9%'
        },
        {
            'event_name': 'Unemployment Rate',
            'country': 'United States',
            'importance': 'high',
            'actual': '3.7%',
            'forecast': '3.8%',
            'previous': '3.8%'
        },
        {
            'event_name': 'Interest Rate Decision',
            'country': 'United States',
            'importance': 'high',
            'actual': None,
            'forecast': '5.25%',
            'previous': '5.25%'
        },
        {
            'event_name': 'Consumer Price Index (CPI)',
            'country': 'United States',
            'importance': 'high',
            'actual': '3.1%',
            'forecast': '3.2%',
            'previous': '3.3%'
        },
        {
            'event_name': 'Retail Sales',
            'country': 'United States',
            'importance': 'medium',
            'actual': '0.6%',
            'forecast': '0.4%',
            'previous': '0.2%'
        }
    ]
    
    # Create events for past days
    for i in range(days_behind):
        event_date = today - timedelta(days=i+1)
        for j, event in enumerate(sample_events[:3]):  # Use first 3 events for past
            event_copy = event.copy()
            event_copy['date'] = event_date + timedelta(hours=9+j)
            events.append(event_copy)
    
    # Create events for future days
    for i in range(days_ahead):
        event_date = today + timedelta(days=i)
        for j, event in enumerate(sample_events[2:]):  # Use last 3 events for future
            event_copy = event.copy()
            event_copy['date'] = event_date + timedelta(hours=9+j)
            # Remove actual value for future events
            event_copy['actual'] = None
            events.append(event_copy)
    
    return events