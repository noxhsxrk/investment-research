"""News sentiment analysis engine for stock-related news.

This module provides functionality for retrieving news articles related to stocks
and performing sentiment analysis to understand market perception and potential
impact on stock performance.
"""

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter

import requests
import yfinance as yf
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from stock_analysis.models.data_models import SentimentResult
from stock_analysis.utils.exceptions import DataRetrievalError, CalculationError
from stock_analysis.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NewsArticle:
    """Data model for a news article."""
    title: str
    content: str
    url: str
    published_date: datetime
    source: str
    
    def get_full_text(self) -> str:
        """Get the full text of the article (title + content)."""
        return f"{self.title}. {self.content}"


class NewsSentimentAnalyzer:
    """Analyzer for retrieving and analyzing news sentiment for stocks."""
    
    def __init__(self):
        """Initialize the news sentiment analyzer."""
        logger.info("Initializing News Sentiment Analyzer")
        
        # Initialize sentiment analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Configuration
        self.max_articles_per_source = 20
        self.request_timeout = 10
        self.retry_attempts = 3
        self.retry_delay = 1.0
        
        # Theme extraction keywords
        self.financial_keywords = {
            'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
            'quarterly', 'annual', 'forecast', 'guidance', 'outlook',
            'dividend', 'buyback', 'acquisition', 'merger', 'ipo'
        }
        
        self.market_keywords = {
            'bullish', 'bearish', 'rally', 'crash', 'volatility', 'momentum',
            'resistance', 'support', 'breakout', 'correction', 'bubble'
        }
        
        self.business_keywords = {
            'expansion', 'partnership', 'contract', 'deal', 'launch',
            'innovation', 'technology', 'competition', 'regulation', 'lawsuit'
        }
    
    def _download_nltk_data(self) -> None:
        """Download required NLTK data if not already present."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
    
    def get_news_articles(self, symbol: str, days: int = 7) -> List[NewsArticle]:
        """Retrieve recent news articles for a stock symbol.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back for news
            
        Returns:
            List of NewsArticle objects
            
        Raises:
            DataRetrievalError: If news retrieval fails
        """
        logger.info(f"Retrieving news articles for {symbol} (last {days} days)")
        
        articles = []
        
        # Try multiple news sources
        sources = [
            self._get_yfinance_news,
            self._get_yahoo_finance_news,
        ]
        
        for source_func in sources:
            try:
                source_articles = source_func(symbol, days)
                articles.extend(source_articles)
                source_name = getattr(source_func, '__name__', str(source_func))
                logger.info(f"Retrieved {len(source_articles)} articles from {source_name}")
            except Exception as e:
                source_name = getattr(source_func, '__name__', str(source_func))
                logger.warning(f"Failed to retrieve news from {source_name}: {e}")
                continue
        
        if not articles:
            logger.warning(f"No news articles found for {symbol}")
            # Return a minimal article to avoid empty results
            return [NewsArticle(
                title=f"No recent news found for {symbol}",
                content="No recent news articles were found for this stock symbol.",
                url="",
                published_date=datetime.now(),
                source="system"
            )]
        
        # Remove duplicates based on title similarity
        articles = self._remove_duplicate_articles(articles)
        
        # Sort by date (most recent first)
        articles.sort(key=lambda x: x.published_date, reverse=True)
        
        logger.info(f"Retrieved {len(articles)} unique articles for {symbol}")
        return articles[:self.max_articles_per_source]
    
    def _get_yfinance_news(self, symbol: str, days: int) -> List[NewsArticle]:
        """Get news from yfinance library.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            List of NewsArticle objects
        """
        articles = []
        
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for item in news:
                # Convert timestamp to datetime
                pub_date = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                
                if pub_date < cutoff_date:
                    continue
                
                article = NewsArticle(
                    title=item.get('title', ''),
                    content=item.get('summary', ''),
                    url=item.get('link', ''),
                    published_date=pub_date,
                    source='yfinance'
                )
                articles.append(article)
                
        except Exception as e:
            logger.error(f"Error retrieving yfinance news for {symbol}: {e}")
            raise DataRetrievalError(f"Failed to retrieve yfinance news: {e}")
        
        return articles
    
    def _get_yahoo_finance_news(self, symbol: str, days: int) -> List[NewsArticle]:
        """Get news from Yahoo Finance RSS feed.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            List of NewsArticle objects
        """
        articles = []
        
        try:
            # Yahoo Finance RSS feed URL
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
            
            response = self._make_request_with_retry(url)
            if not response:
                return articles
            
            # Simple RSS parsing (avoiding xml dependency)
            content = response.text
            
            # Extract items using regex (basic RSS parsing)
            item_pattern = r'<item>(.*?)</item>'
            items = re.findall(item_pattern, content, re.DOTALL)
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            for item in items:
                title_match = re.search(r'<title><!\[CDATA\[(.*?)\]\]></title>', item)
                desc_match = re.search(r'<description><!\[CDATA\[(.*?)\]\]></description>', item)
                link_match = re.search(r'<link>(.*?)</link>', item)
                date_match = re.search(r'<pubDate>(.*?)</pubDate>', item)
                
                if not (title_match and desc_match):
                    continue
                
                title = title_match.group(1)
                description = desc_match.group(1)
                link = link_match.group(1) if link_match else ""
                
                # Parse date
                try:
                    if date_match:
                        date_str = date_match.group(1)
                        pub_date = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z').replace(tzinfo=None)
                    else:
                        pub_date = datetime.now()
                except:
                    pub_date = datetime.now()
                
                if pub_date < cutoff_date:
                    continue
                
                article = NewsArticle(
                    title=title,
                    content=description,
                    url=link,
                    published_date=pub_date,
                    source='yahoo_rss'
                )
                articles.append(article)
                
        except Exception as e:
            logger.error(f"Error retrieving Yahoo RSS news for {symbol}: {e}")
            # Don't raise error here, just return empty list
            
        return articles
    
    def _make_request_with_retry(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with retry logic.
        
        Args:
            url: URL to request
            
        Returns:
            Response object or None if failed
        """
        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(
                    url,
                    timeout=self.request_timeout,
                    headers={'User-Agent': 'Mozilla/5.0 (compatible; StockAnalyzer/1.0)'}
                )
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                continue
        
        return None
    
    def _remove_duplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity.
        
        Args:
            articles: List of articles to deduplicate
            
        Returns:
            List of unique articles
        """
        if not articles:
            return articles
        
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Normalize title for comparison
            normalized_title = re.sub(r'[^\w\s]', '', article.title.lower()).strip()
            
            # Check if we've seen a similar title
            is_duplicate = False
            for seen_title in seen_titles:
                # Simple similarity check - if 80% of words match
                title_words = set(normalized_title.split())
                seen_words = set(seen_title.split())
                
                if len(title_words) > 0 and len(seen_words) > 0:
                    intersection = len(title_words.intersection(seen_words))
                    union = len(title_words.union(seen_words))
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > 0.6:  # Lowered threshold for better duplicate detection
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_articles.append(article)
                seen_titles.add(normalized_title)
        
        return unique_articles
    
    def analyze_sentiment(self, articles: List[NewsArticle]) -> SentimentResult:
        """Analyze sentiment of news articles.
        
        Args:
            articles: List of news articles to analyze
            
        Returns:
            SentimentResult with sentiment scores and analysis
            
        Raises:
            CalculationError: If sentiment analysis fails
        """
        logger.info(f"Analyzing sentiment for {len(articles)} articles")
        
        if not articles:
            logger.warning("No articles provided for sentiment analysis")
            return SentimentResult(
                overall_sentiment=0.0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                key_themes=["No news available"],
                sentiment_trend=[]
            )
        
        try:
            sentiment_scores = []
            sentiment_categories = {'positive': 0, 'negative': 0, 'neutral': 0}
            daily_sentiments = {}
            
            for article in articles:
                # Get sentiment score for the article
                score = self._analyze_article_sentiment(article)
                sentiment_scores.append(score)
                
                # Categorize sentiment
                if score > 0.1:
                    sentiment_categories['positive'] += 1
                elif score < -0.1:
                    sentiment_categories['negative'] += 1
                else:
                    sentiment_categories['neutral'] += 1
                
                # Track daily sentiment for trend analysis
                date_key = article.published_date.date()
                if date_key not in daily_sentiments:
                    daily_sentiments[date_key] = []
                daily_sentiments[date_key].append(score)
            
            # Calculate overall sentiment
            overall_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
            
            # Calculate sentiment trend (daily averages)
            sentiment_trend = []
            for date in sorted(daily_sentiments.keys()):
                daily_avg = sum(daily_sentiments[date]) / len(daily_sentiments[date])
                sentiment_trend.append(daily_avg)
            
            # Extract key themes
            key_themes = self.extract_themes(articles)
            
            result = SentimentResult(
                overall_sentiment=overall_sentiment,
                positive_count=sentiment_categories['positive'],
                negative_count=sentiment_categories['negative'],
                neutral_count=sentiment_categories['neutral'],
                key_themes=key_themes,
                sentiment_trend=sentiment_trend
            )
            
            logger.info(f"Sentiment analysis complete: overall={overall_sentiment:.3f}, "
                       f"pos={result.positive_count}, neg={result.negative_count}, "
                       f"neu={result.neutral_count}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            raise CalculationError(f"Sentiment analysis failed: {e}")
    
    def _analyze_article_sentiment(self, article: NewsArticle) -> float:
        """Analyze sentiment of a single article using multiple methods.
        
        Args:
            article: NewsArticle to analyze
            
        Returns:
            Sentiment score between -1 and 1
        """
        text = article.get_full_text()
        
        # Use VADER sentiment analyzer
        vader_scores = self.vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        # Use TextBlob sentiment analyzer
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        
        # Combine scores (weighted average)
        # VADER is generally better for social media/news text
        combined_score = (vader_compound * 0.7) + (textblob_polarity * 0.3)
        
        # Ensure score is within bounds
        return max(-1.0, min(1.0, combined_score))
    
    def extract_themes(self, articles: List[NewsArticle]) -> List[str]:
        """Extract key themes and topics from news articles.
        
        Args:
            articles: List of news articles
            
        Returns:
            List of key themes/topics
        """
        logger.info(f"Extracting themes from {len(articles)} articles")
        
        if not articles:
            return ["No themes available"]
        
        try:
            # Combine all article text
            all_text = " ".join([article.get_full_text() for article in articles])
            
            # Tokenize and clean text
            tokens = word_tokenize(all_text.lower())
            
            # Remove stopwords and non-alphabetic tokens
            filtered_tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token.isalpha() and token not in self.stop_words and len(token) > 2
            ]
            
            # Find themes based on keyword categories
            themes = []
            
            # Check for financial themes
            financial_matches = [token for token in filtered_tokens if token in self.financial_keywords]
            if financial_matches:
                most_common_financial = Counter(financial_matches).most_common(3)
                themes.extend([f"Financial: {word}" for word, _ in most_common_financial])
            
            # Check for market themes
            market_matches = [token for token in filtered_tokens if token in self.market_keywords]
            if market_matches:
                most_common_market = Counter(market_matches).most_common(2)
                themes.extend([f"Market: {word}" for word, _ in most_common_market])
            
            # Check for business themes
            business_matches = [token for token in filtered_tokens if token in self.business_keywords]
            if business_matches:
                most_common_business = Counter(business_matches).most_common(2)
                themes.extend([f"Business: {word}" for word, _ in most_common_business])
            
            # Add most frequent general terms (excluding common words)
            word_freq = Counter(filtered_tokens)
            # Remove very common words that aren't meaningful
            common_words = {'company', 'stock', 'share', 'market', 'price', 'year', 'time', 'new', 'said'}
            meaningful_words = {word: count for word, count in word_freq.items() 
                              if word not in common_words and count >= 2}
            
            if meaningful_words:
                most_common_general = Counter(meaningful_words).most_common(3)
                themes.extend([f"Topic: {word}" for word, _ in most_common_general])
            
            # If no themes found, provide generic themes
            if not themes:
                themes = ["General market news", "Company updates"]
            
            # Limit to top 10 themes
            themes = themes[:10]
            
            logger.info(f"Extracted {len(themes)} themes: {themes}")
            return themes
            
        except Exception as e:
            logger.error(f"Error extracting themes: {e}")
            return ["Theme extraction failed", "General news"]
    
    def get_sentiment_summary(self, sentiment_result: SentimentResult) -> str:
        """Generate a human-readable sentiment summary.
        
        Args:
            sentiment_result: SentimentResult to summarize
            
        Returns:
            Human-readable sentiment summary
        """
        total_articles = (sentiment_result.positive_count + 
                         sentiment_result.negative_count + 
                         sentiment_result.neutral_count)
        
        if total_articles == 0:
            return "No news sentiment data available"
        
        # Determine overall sentiment category
        if sentiment_result.overall_sentiment > 0.2:
            sentiment_category = "Positive"
        elif sentiment_result.overall_sentiment < -0.2:
            sentiment_category = "Negative"
        else:
            sentiment_category = "Neutral"
        
        # Calculate percentages
        pos_pct = (sentiment_result.positive_count / total_articles) * 100
        neg_pct = (sentiment_result.negative_count / total_articles) * 100
        neu_pct = (sentiment_result.neutral_count / total_articles) * 100
        
        summary = (
            f"{sentiment_category} sentiment overall "
            f"(score: {sentiment_result.overall_sentiment:.2f}). "
            f"Analysis of {total_articles} articles: "
            f"{pos_pct:.1f}% positive, {neg_pct:.1f}% negative, {neu_pct:.1f}% neutral. "
            f"Key themes: {', '.join(sentiment_result.key_themes[:3])}"
        )
        
        return summary