"""Unit tests for news command in CLI.

This module tests the news command for retrieving financial news.
"""

import unittest
from unittest.mock import patch, MagicMock
import argparse
import sys
from io import StringIO
from datetime import datetime

from stock_analysis.cli import handle_news_command
from stock_analysis.models.enhanced_data_models import NewsItem
from stock_analysis.models.data_models import SentimentResult


class TestNewsCommand(unittest.TestCase):
    """Test cases for news command."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock arguments
        self.args = argparse.Namespace()
        self.args.symbol = None
        self.args.market = False
        self.args.category = None
        self.args.economic_calendar = False
        self.args.days = 7
        self.args.limit = 10
        self.args.sentiment = False
        self.args.trending = False
        self.args.export_format = None
        self.args.output = None
        self.args.no_export = True
        self.args.verbose = False
        
        # Create mock news items
        self.company_news = [
            NewsItem(
                title="Apple Reports Record Quarterly Revenue",
                source="Financial Times",
                url="https://example.com/news/1",
                published_at=datetime.now(),
                summary="Apple Inc. reported record quarterly revenue of $89.6 billion.",
                sentiment=0.75,
                impact="high",
                categories=["earnings", "technology"]
            ),
            NewsItem(
                title="Apple Announces New Product Line",
                source="CNBC",
                url="https://example.com/news/2",
                published_at=datetime.now(),
                summary="Apple Inc. announced a new product line at their annual event.",
                sentiment=0.65,
                impact="medium",
                categories=["product", "technology"]
            )
        ]
        
        self.market_news = [
            NewsItem(
                title="Fed Raises Interest Rates by 25 Basis Points",
                source="Wall Street Journal",
                url="https://example.com/news/3",
                published_at=datetime.now(),
                summary="The Federal Reserve raised interest rates by 25 basis points.",
                sentiment=-0.2,
                impact="high",
                categories=["economy", "federal reserve"]
            ),
            NewsItem(
                title="S&P 500 Reaches New All-Time High",
                source="Bloomberg",
                url="https://example.com/news/4",
                published_at=datetime.now(),
                summary="The S&P 500 index reached a new all-time high today.",
                sentiment=0.8,
                impact="medium",
                categories=["markets", "stocks"]
            )
        ]
        
        # Create mock sentiment result
        self.sentiment_result = SentimentResult(
            overall_sentiment=0.65,
            positive_count=15,
            negative_count=5,
            neutral_count=10,
            key_themes=["Earnings", "Product Launch", "Market Share"],
            sentiment_trend=[0.6, 0.62, 0.65]
        )
        
        # Create mock economic calendar
        self.economic_calendar = [
            {
                'date': '2025-07-23',
                'time': '08:30',
                'country': 'US',
                'event': 'Initial Jobless Claims',
                'importance': 'medium',
                'actual': '240K',
                'forecast': '245K',
                'previous': '248K'
            },
            {
                'date': '2025-07-24',
                'time': '14:00',
                'country': 'US',
                'event': 'Fed Interest Rate Decision',
                'importance': 'high',
                'actual': 'N/A',
                'forecast': '5.50%',
                'previous': '5.25%'
            }
        ]
        
        # Create mock trending topics
        self.trending_topics = [
            {
                'topic': 'Federal Reserve',
                'count': 45,
                'sentiment': -0.1
            },
            {
                'topic': 'Earnings Season',
                'count': 38,
                'sentiment': 0.3
            },
            {
                'topic': 'Tech Stocks',
                'count': 32,
                'sentiment': 0.5
            }
        ]
    
    @patch('stock_analysis.services.news_service.NewsService')
    @patch('stock_analysis.exporters.export_service.ExportService')
    def test_news_command_company_news(self, mock_export_service_class, mock_news_service_class):
        """Test news command with company news option."""
        # Set up mocks
        mock_news_service = MagicMock()
        mock_news_service_class.return_value = mock_news_service
        mock_news_service.get_company_news.return_value = self.company_news
        
        mock_export_service = MagicMock()
        mock_export_service_class.return_value = mock_export_service
        
        # Set company news option
        self.args.symbol = 'AAPL'
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_news_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_news_service.get_company_news.assert_called_once_with(
            symbol='AAPL',
            days=7,
            limit=10,
            include_sentiment=False
        )
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Retrieving news for AAPL...', output)
        self.assertIn('News for AAPL:', output)
        self.assertIn('Apple Reports Record Quarterly Revenue', output)
        self.assertIn('Apple Announces New Product Line', output)
    
    @patch('stock_analysis.services.news_service.NewsService')
    @patch('stock_analysis.exporters.export_service.ExportService')
    def test_news_command_company_news_with_sentiment(self, mock_export_service_class, mock_news_service_class):
        """Test news command with company news and sentiment option."""
        # Set up mocks
        mock_news_service = MagicMock()
        mock_news_service_class.return_value = mock_news_service
        mock_news_service.get_company_news.return_value = self.company_news
        mock_news_service.get_news_sentiment.return_value = self.sentiment_result
        
        mock_export_service = MagicMock()
        mock_export_service_class.return_value = mock_export_service
        
        # Set company news and sentiment options
        self.args.symbol = 'AAPL'
        self.args.sentiment = True
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_news_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_news_service.get_company_news.assert_called_once_with(
            symbol='AAPL',
            days=7,
            limit=10,
            include_sentiment=True
        )
        mock_news_service.get_news_sentiment.assert_called_once_with('AAPL', 7)
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Retrieving news for AAPL...', output)
        self.assertIn('News for AAPL:', output)
        self.assertIn('Sentiment Analysis:', output)
        self.assertIn('Overall Sentiment:', output)
        self.assertIn('Key Themes:', output)
    
    @patch('stock_analysis.services.news_service.NewsService')
    @patch('stock_analysis.exporters.export_service.ExportService')
    def test_news_command_market_news(self, mock_export_service_class, mock_news_service_class):
        """Test news command with market news option."""
        # Set up mocks
        mock_news_service = MagicMock()
        mock_news_service_class.return_value = mock_news_service
        mock_news_service.get_market_news.return_value = self.market_news
        
        mock_export_service = MagicMock()
        mock_export_service_class.return_value = mock_export_service
        
        # Set market news option
        self.args.market = True
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_news_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_news_service.get_market_news.assert_called_once_with(
            category=None,
            days=7,
            limit=10,
            include_sentiment=False
        )
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Retrieving market news...', output)
        self.assertIn('Market News:', output)
        self.assertIn('Fed Raises Interest Rates', output)
        self.assertIn('S&P 500 Reaches New All-Time High', output)
    
    @patch('stock_analysis.services.news_service.NewsService')
    @patch('stock_analysis.exporters.export_service.ExportService')
    def test_news_command_economic_calendar(self, mock_export_service_class, mock_news_service_class):
        """Test news command with economic calendar option."""
        # Set up mocks
        mock_news_service = MagicMock()
        mock_news_service_class.return_value = mock_news_service
        mock_news_service.get_economic_calendar.return_value = self.economic_calendar
        
        mock_export_service = MagicMock()
        mock_export_service_class.return_value = mock_export_service
        
        # Set economic calendar option
        self.args.economic_calendar = True
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_news_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_news_service.get_economic_calendar.assert_called_once_with(
            days_ahead=7,
            days_behind=1
        )
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Retrieving economic calendar...', output)
        self.assertIn('Economic Calendar:', output)
        self.assertIn('Initial Jobless Claims', output)
        self.assertIn('Fed Interest Rate Decision', output)
    
    @patch('stock_analysis.services.news_service.NewsService')
    @patch('stock_analysis.exporters.export_service.ExportService')
    def test_news_command_trending_topics(self, mock_export_service_class, mock_news_service_class):
        """Test news command with trending topics option."""
        # Set up mocks
        mock_news_service = MagicMock()
        mock_news_service_class.return_value = mock_news_service
        mock_news_service.get_trending_topics.return_value = self.trending_topics
        
        mock_export_service = MagicMock()
        mock_export_service_class.return_value = mock_export_service
        
        # Set trending topics option
        self.args.trending = True
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_news_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_news_service.get_trending_topics.assert_called_once_with(days=7)
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Retrieving trending topics...', output)
        self.assertIn('Trending Topics:', output)
        self.assertIn('Federal Reserve', output)
        self.assertIn('Earnings Season', output)
        self.assertIn('Tech Stocks', output)
    
    @patch('stock_analysis.services.news_service.NewsService')
    @patch('stock_analysis.exporters.export_service.ExportService')
    def test_news_command_export(self, mock_export_service_class, mock_news_service_class):
        """Test news command with export option."""
        # Set up mocks
        mock_news_service = MagicMock()
        mock_news_service_class.return_value = mock_news_service
        mock_news_service.get_market_news.return_value = self.market_news
        
        mock_export_service = MagicMock()
        mock_export_service_class.return_value = mock_export_service
        mock_export_service.export_to_json.return_value = 'news_data.json'
        
        # Set export options
        self.args.market = True
        self.args.no_export = False
        self.args.export_format = 'json'
        self.args.output = 'news_data'
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Execute command
        result = handle_news_command(self.args)
        
        # Reset stdout
        sys.stdout = sys.__stdout__
        
        # Verify results
        self.assertEqual(result, 0)
        mock_news_service.get_market_news.assert_called_once()
        mock_export_service.export_to_json.assert_called_once()
        
        # Check output
        output = captured_output.getvalue()
        self.assertIn('Results exported to:', output)


if __name__ == '__main__':
    unittest.main()