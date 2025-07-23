"""News service for retrieving financial news and economic events.

This module provides a service for retrieving financial news, market news,
economic calendar events, and sentiment analysis.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

from stock_analysis.services.financial_data_integration_service import FinancialDataIntegrationService
from stock_analysis.services.news_economic_integration import (
    NewsItem, EconomicEvent, get_news_and_sentiment, get_economic_calendar,
    analyze_news_sentiment, categorize_news, assess_news_impact
)
from stock_analysis.models.data_models import SentimentResult
from stock_analysis.models.enhanced_data_models import NewsItem as EnhancedNewsItem
from stock_analysis.utils.exceptions import DataRetrievalError
from stock_analysis.utils.logging import get_logger, log_api_call, log_data_quality_issue
from stock_analysis.utils.performance_metrics import monitor_performance
from stock_analysis.utils.cache_manager import get_cache_manager

logger = get_logger(__name__)


class NewsService:
    """Service for retrieving and analyzing financial news."""
    
    def __init__(self, integration_service: Optional[FinancialDataIntegrationService] = None):
        """Initialize the news service.
        
        Args:
            integration_service: Financial data integration service instance
        """
        # Initialize integration service if not provided
        if integration_service is None:
            self.integration_service = FinancialDataIntegrationService()
        else:
            self.integration_service = integration_service
        
        self.cache = get_cache_manager()
    
    @monitor_performance('get_company_news')
    def get_company_news(
        self, 
        symbol: str, 
        days: int = 7, 
        limit: int = 10,
        include_sentiment: bool = True,
        use_cache: bool = True
    ) -> List[EnhancedNewsItem]:
        """Get news for a specific company.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back
            limit: Maximum number of news items to return
            include_sentiment: Whether to include sentiment analysis
            use_cache: Whether to use cached data if available
            
        Returns:
            List of NewsItem objects
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.set_context(symbol=symbol, operation='get_company_news')
        logger.info(f"Retrieving company news for {symbol} (last {days} days, limit {limit})")
        
        # Generate cache key
        cache_key = f"company_news:{symbol}:{days}:{limit}:{include_sentiment}"
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved company news for {symbol} from cache")
                return cached_data
        
        try:
            # Get news from integration service
            news_data = self.integration_service.get_company_news(
                symbol=symbol,
                days=days,
                limit=limit,
                use_cache=use_cache
            )
            
            # Process news data
            news_items = self._process_news_data(news_data, include_sentiment)
            
            # Limit the number of items
            news_items = news_items[:limit]
            
            # Cache the result
            if use_cache:
                self.cache.set(
                    cache_key, 
                    news_items, 
                    data_type="company_news", 
                    tags=[symbol, "news"]
                )
            
            logger.info(f"Successfully retrieved {len(news_items)} news items for {symbol}")
            return news_items
            
        except Exception as e:
            logger.error(f"Error retrieving company news for {symbol}: {str(e)}")
            raise DataRetrievalError(
                f"Failed to retrieve company news for {symbol}: {str(e)}",
                symbol=symbol,
                data_source='multiple',
                original_exception=e
            )
    
    @monitor_performance('get_market_news')
    def get_market_news(
        self, 
        category: Optional[str] = None, 
        days: int = 3, 
        limit: int = 10,
        include_sentiment: bool = True,
        use_cache: bool = True
    ) -> List[EnhancedNewsItem]:
        """Get general market news.
        
        Args:
            category: News category filter (e.g., 'market', 'economy', 'earnings')
            days: Number of days to look back
            limit: Maximum number of news items to return
            include_sentiment: Whether to include sentiment analysis
            use_cache: Whether to use cached data if available
            
        Returns:
            List of NewsItem objects
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.set_context(category=category, operation='get_market_news')
        logger.info(f"Retrieving market news (category={category}, last {days} days, limit {limit})")
        
        # Generate cache key
        cache_key = f"market_news:{category or 'all'}:{days}:{limit}:{include_sentiment}"
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved market news from cache")
                return cached_data
        
        try:
            # Get news from integration service
            news_data = self.integration_service.get_market_news(
                category=category,
                days=days,
                limit=limit,
                use_cache=use_cache
            )
            
            # Process news data
            news_items = self._process_news_data(news_data, include_sentiment)
            
            # Filter by category if specified
            if category:
                news_items = [item for item in news_items if category in item.categories] if news_items else []
            
            # Limit the number of items
            news_items = news_items[:limit]
            
            # Cache the result
            if use_cache:
                self.cache.set(
                    cache_key, 
                    news_items, 
                    data_type="market_news", 
                    tags=["market_news", category or "all"]
                )
            
            logger.info(f"Successfully retrieved {len(news_items)} market news items")
            return news_items
            
        except Exception as e:
            logger.error(f"Error retrieving market news: {str(e)}")
            raise DataRetrievalError(
                f"Failed to retrieve market news: {str(e)}",
                data_source='multiple',
                original_exception=e
            )
    
    def _process_news_data(self, news_data: List[Dict[str, Any]], include_sentiment: bool) -> List[EnhancedNewsItem]:
        """Process raw news data into NewsItem objects.
        
        Args:
            news_data: List of news data dictionaries
            include_sentiment: Whether to include sentiment analysis
            
        Returns:
            List of NewsItem objects
        """
        # If no news data, return empty list
        if not news_data:
            return []
        
        # Add sentiment analysis if requested
        if include_sentiment:
            # Check if sentiment is already included
            if not any('sentiment' in item for item in news_data):
                analyze_news_sentiment(news_data)
        
        # Categorize news if not already categorized
        if not any('categories' in item for item in news_data):
            categorize_news(news_data)
        
        # Assess impact if not already assessed
        if not any('impact' in item for item in news_data):
            assess_news_impact(news_data)
        
        # Convert to NewsItem objects
        news_items = []
        for item in news_data:
            try:
                # Convert published_at to datetime if it's a string
                published_at = item.get('published_at')
                if isinstance(published_at, str):
                    try:
                        published_at = datetime.fromisoformat(published_at)
                    except ValueError:
                        published_at = datetime.now()
                elif not isinstance(published_at, datetime):
                    published_at = datetime.now()
                
                # Ensure summary is a string
                summary = item.get('summary', '')
                if not isinstance(summary, str):
                    summary = str(summary) if summary is not None else ''
                
                # Ensure title is a string
                title = item.get('title', '')
                if not isinstance(title, str):
                    title = str(title) if title is not None else ''
                
                # Ensure source is a string
                source = item.get('source', '')
                if not isinstance(source, str):
                    source = str(source) if source is not None else ''
                
                # Ensure url is a string
                url = item.get('url', '')
                if not isinstance(url, str):
                    url = str(url) if url is not None else ''
                
                # Ensure categories is a list
                categories = item.get('categories', [])
                if not isinstance(categories, list):
                    categories = [str(categories)] if categories is not None else []
                
                news_item = EnhancedNewsItem(
                    title=title,
                    source=source,
                    url=url,
                    published_at=published_at,
                    summary=summary,
                    sentiment=item.get('sentiment'),
                    impact=item.get('impact'),
                    categories=categories
                )
                
                # Validate the news item
                news_item.validate()
                news_items.append(news_item)
                
            except Exception as e:
                logger.warning(f"Error processing news item: {str(e)}")
        
        return news_items
    
    @monitor_performance('get_economic_calendar')
    def get_economic_calendar(
        self, 
        days_ahead: int = 7, 
        days_behind: int = 1,
        country: Optional[str] = None,
        importance: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get upcoming economic events.
        
        Args:
            days_ahead: Number of days to look ahead
            days_behind: Number of days to look behind
            country: Filter by country (e.g., 'United States', 'EU')
            importance: Filter by importance ('high', 'medium', 'low')
            use_cache: Whether to use cached data if available
            
        Returns:
            List of economic events
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.set_context(
            days_ahead=days_ahead,
            days_behind=days_behind,
            country=country,
            importance=importance,
            operation='get_economic_calendar'
        )
        logger.info(f"Retrieving economic calendar (ahead={days_ahead}, behind={days_behind}, country={country}, importance={importance})")
        
        # Generate cache key
        cache_key = f"economic_calendar:{days_ahead}:{days_behind}:{country or 'all'}:{importance or 'all'}"
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved economic calendar from cache")
                return cached_data
        
        try:
            # Get economic calendar from integration service
            calendar_data = self.integration_service.get_economic_calendar(
                days_ahead=days_ahead,
                days_behind=days_behind,
                use_cache=use_cache
            )
            
            # Filter by country if specified
            if country:
                calendar_data = [
                    event for event in calendar_data 
                    if event.get('country') == country
                ]
            
            # Filter by importance if specified
            if importance:
                calendar_data = [
                    event for event in calendar_data 
                    if event.get('importance') == importance
                ]
            
            # Cache the result
            if use_cache:
                self.cache.set(
                    cache_key, 
                    calendar_data, 
                    data_type="economic_calendar", 
                    tags=["economic_calendar"]
                )
            
            logger.info(f"Successfully retrieved {len(calendar_data)} economic events")
            return calendar_data
            
        except Exception as e:
            logger.error(f"Error retrieving economic calendar: {str(e)}")
            raise DataRetrievalError(
                f"Failed to retrieve economic calendar: {str(e)}",
                data_source='multiple',
                original_exception=e
            )
    
    @monitor_performance('get_news_sentiment')
    def get_news_sentiment(
        self, 
        symbol: str, 
        days: int = 7,
        use_cache: bool = True
    ) -> SentimentResult:
        """Get sentiment analysis for news related to a symbol.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back
            use_cache: Whether to use cached data if available
            
        Returns:
            SentimentResult object with sentiment analysis
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.set_context(symbol=symbol, days=days, operation='get_news_sentiment')
        logger.info(f"Retrieving news sentiment for {symbol} (last {days} days)")
        
        # Generate cache key
        cache_key = f"news_sentiment:{symbol}:{days}"
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved news sentiment for {symbol} from cache")
                return cached_data
        
        try:
            # Get news and sentiment from integration service
            news_items, sentiment_result = get_news_and_sentiment(symbol, days, use_cache=use_cache)
            
            # Cache the result
            if use_cache:
                self.cache.set(
                    cache_key, 
                    sentiment_result, 
                    data_type="news_sentiment", 
                    tags=[symbol, "sentiment"]
                )
            
            logger.info(f"Successfully retrieved news sentiment for {symbol}")
            return sentiment_result
            
        except Exception as e:
            logger.error(f"Error retrieving news sentiment for {symbol}: {str(e)}")
            raise DataRetrievalError(
                f"Failed to retrieve news sentiment for {symbol}: {str(e)}",
                symbol=symbol,
                data_source='multiple',
                original_exception=e
            )
    
    @monitor_performance('get_trending_topics')
    def get_trending_topics(self, days: int = 3, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Get trending topics in financial news.
        
        Args:
            days: Number of days to look back
            use_cache: Whether to use cached data if available
            
        Returns:
            List of trending topics with frequency and sentiment
            
        Raises:
            DataRetrievalError: If data cannot be retrieved
        """
        logger.set_context(days=days, operation='get_trending_topics')
        logger.info(f"Retrieving trending topics (last {days} days)")
        
        # Generate cache key
        cache_key = f"trending_topics:{days}"
        
        # Try to get from cache first if caching is enabled
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved trending topics from cache")
                return cached_data
        
        try:
            # Get market news
            news_items = self.get_market_news(days=days, limit=100, include_sentiment=True, use_cache=use_cache)
            
            # Extract topics from categories and count frequency
            topic_counts = {}
            topic_sentiments = {}
            
            for item in news_items:
                for category in item.categories:
                    if category not in topic_counts:
                        topic_counts[category] = 0
                        topic_sentiments[category] = []
                    
                    topic_counts[category] += 1
                    if item.sentiment is not None:
                        topic_sentiments[category].append(item.sentiment)
            
            # Calculate average sentiment for each topic
            trending_topics = []
            for topic, count in topic_counts.items():
                avg_sentiment = sum(topic_sentiments[topic]) / len(topic_sentiments[topic]) if topic_sentiments[topic] else 0
                
                trending_topics.append({
                    'topic': topic,
                    'count': count,
                    'sentiment': avg_sentiment
                })
            
            # Sort by count (descending)
            trending_topics.sort(key=lambda x: x['count'], reverse=True)
            
            # Cache the result
            if use_cache:
                self.cache.set(
                    cache_key, 
                    trending_topics, 
                    data_type="trending_topics", 
                    tags=["trending_topics"]
                )
            
            logger.info(f"Successfully retrieved {len(trending_topics)} trending topics")
            return trending_topics
            
        except Exception as e:
            logger.error(f"Error retrieving trending topics: {str(e)}")
            raise DataRetrievalError(
                f"Failed to retrieve trending topics: {str(e)}",
                data_source='multiple',
                original_exception=e
            )