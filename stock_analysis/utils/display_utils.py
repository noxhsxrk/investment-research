"""Display utilities for the stock analysis system.

This module provides functions for displaying analysis results in a user-friendly format.
"""

import sys
from typing import Dict, List, Optional, Any, Union
import pandas as pd

from stock_analysis.models.comprehensive_models import ComprehensiveAnalysisResult, ComprehensiveAnalysisReport
from stock_analysis.models.data_models import StockInfo, ETFInfo


def display_comprehensive_summary(result: Union[ComprehensiveAnalysisResult, ComprehensiveAnalysisReport], 
                                verbose: bool = False) -> None:
    """Display a summary of comprehensive analysis results.
    
    Args:
        result: Either a single ComprehensiveAnalysisResult or a ComprehensiveAnalysisReport
        verbose: Whether to display detailed information
    """
    if isinstance(result, ComprehensiveAnalysisReport):
        _display_comprehensive_report_summary(result, verbose)
    else:
        _display_comprehensive_result_summary(result, verbose)


def _display_comprehensive_result_summary(result: ComprehensiveAnalysisResult, verbose: bool = False) -> None:
    """Display a summary of a single comprehensive analysis result.
    
    Args:
        result: ComprehensiveAnalysisResult to display
        verbose: Whether to display detailed information
    """
    print(f"\nComprehensive Analysis Summary for {result.symbol}")
    print("=" * 50)
    
    # Basic information section
    print("\nBasic Information:")
    print("-" * 20)
    
    if result.analysis_result:
        stock_info = result.analysis_result.stock_info
        
        if isinstance(stock_info, ETFInfo):
            # ETF summary
            print(f"ETF: {stock_info.name}")
            print(f"Current Price: ${stock_info.current_price:.2f}")
            print(f"NAV: ${stock_info.nav:.2f}" if stock_info.nav else "NAV: Not available")
            print(f"Expense Ratio: {stock_info.expense_ratio*100:.2f}%" if stock_info.expense_ratio else "Expense Ratio: Not available")
            print(f"Assets Under Management: ${stock_info.assets_under_management/1e9:.2f}B" if stock_info.assets_under_management else "AUM: Not available")
            print(f"Category: {stock_info.category}" if stock_info.category else "Category: Not available")
            
            if stock_info.holdings and verbose:
                print(f"\nTop 5 Holdings:")
                for holding in stock_info.holdings[:5]:
                    print(f"  {holding['symbol']} ({holding['name']}): {holding['weight']*100:.2f}%")
        else:
            # Stock summary
            if hasattr(stock_info, 'company_name'):
                print(f"Company: {stock_info.company_name}")
            else:
                print(f"Company: {stock_info.name}")
                
            print(f"Current Price: ${stock_info.current_price:.2f}")
            
            if hasattr(stock_info, 'sector') and stock_info.sector:
                print(f"Sector: {stock_info.sector}")
            
            if hasattr(stock_info, 'industry') and stock_info.industry:
                print(f"Industry: {stock_info.industry}")
            
            if hasattr(stock_info, 'pe_ratio') and stock_info.pe_ratio:
                print(f"P/E Ratio: {stock_info.pe_ratio:.2f}")
            else:
                print("P/E Ratio: Not available")
                
            if hasattr(stock_info, 'pb_ratio') and stock_info.pb_ratio:
                print(f"P/B Ratio: {stock_info.pb_ratio:.2f}")
            else:
                print("P/B Ratio: Not available")
            
            if hasattr(stock_info, 'dividend_yield') and stock_info.dividend_yield:
                print(f"Dividend Yield: {stock_info.dividend_yield*100:.2f}%")
            else:
                print("Dividend Yield: Not available")
    else:
        print("Stock analysis data not available")
    
    # Financial indicators section
    print("\nFinancial Indicators:")
    print("-" * 20)
    
    if result.analysis_result:
        print(f"Health Score: {result.analysis_result.health_score.overall_score:.1f}/100")
        print(f"Financial Strength: {result.analysis_result.health_score.financial_strength:.1f}/100")
        print(f"Profitability: {result.analysis_result.health_score.profitability_health:.1f}/100")
        print(f"Liquidity: {result.analysis_result.health_score.liquidity_health:.1f}/100")
        print(f"Risk Assessment: {result.analysis_result.health_score.risk_assessment}")
        
        print(f"\nFair Value: ${result.analysis_result.fair_value.average_fair_value:.2f}")
        print(f"Current Price: ${result.analysis_result.fair_value.current_price:.2f}")
        
        # Calculate price difference and percentage
        price_diff = result.analysis_result.fair_value.average_fair_value - result.analysis_result.fair_value.current_price
        price_pct = (price_diff / result.analysis_result.fair_value.current_price) * 100
        
        if price_diff > 0:
            print(f"Undervalued by: ${abs(price_diff):.2f} ({abs(price_pct):.1f}%)")
        elif price_diff < 0:
            print(f"Overvalued by: ${abs(price_diff):.2f} ({abs(price_pct):.1f}%)")
        else:
            print("Fairly valued")
            
        print(f"Recommendation: {result.analysis_result.fair_value.recommendation}")
    else:
        print("Financial indicators not available")
    
    # Financial statements section
    if result.financial_statements and verbose:
        print("\nFinancial Statements:")
        print("-" * 20)
        
        statements_available = []
        
        if result.financial_statements.get('income_statement') is not None:
            statements_available.append("Income Statement")
        
        if result.financial_statements.get('balance_sheet') is not None:
            statements_available.append("Balance Sheet")
        
        if result.financial_statements.get('cash_flow') is not None:
            statements_available.append("Cash Flow Statement")
        
        if statements_available:
            print(f"Available statements: {', '.join(statements_available)}")
            
            # Show key metrics from income statement if available
            income_stmt = result.financial_statements.get('income_statement')
            if income_stmt is not None and not income_stmt.empty:
                print("\nKey Financial Metrics (most recent year):")
                
                # Get the most recent year's data
                latest_year = income_stmt.columns[0]
                
                # Extract key metrics
                key_metrics = {}
                
                if 'Revenue' in income_stmt.index:
                    key_metrics['Revenue'] = income_stmt.loc['Revenue', latest_year]
                
                if 'Gross Profit' in income_stmt.index:
                    key_metrics['Gross Profit'] = income_stmt.loc['Gross Profit', latest_year]
                
                if 'Operating Income' in income_stmt.index:
                    key_metrics['Operating Income'] = income_stmt.loc['Operating Income', latest_year]
                
                if 'Net Income' in income_stmt.index:
                    key_metrics['Net Income'] = income_stmt.loc['Net Income', latest_year]
                
                # Display key metrics
                for metric, value in key_metrics.items():
                    if isinstance(value, (int, float)):
                        if abs(value) >= 1e9:
                            print(f"  {metric}: ${value/1e9:.2f}B")
                        elif abs(value) >= 1e6:
                            print(f"  {metric}: ${value/1e6:.2f}M")
                        else:
                            print(f"  {metric}: ${value:.2f}")
                    else:
                        print(f"  {metric}: {value}")
        else:
            print("No financial statements available")
    
    # News sentiment section
    print("\nNews Sentiment:")
    print("-" * 20)
    
    if result.news_sentiment:
        sentiment_score = result.news_sentiment.overall_sentiment
        
        # Convert sentiment score to descriptive text
        sentiment_text = "Neutral"
        if sentiment_score >= 0.5:
            sentiment_text = "Very Positive"
        elif sentiment_score >= 0.2:
            sentiment_text = "Positive"
        elif sentiment_score <= -0.5:
            sentiment_text = "Very Negative"
        elif sentiment_score <= -0.2:
            sentiment_text = "Negative"
        
        print(f"Overall Sentiment: {sentiment_text} ({sentiment_score:.2f})")
        print(f"Positive News: {result.news_sentiment.positive_count}")
        print(f"Negative News: {result.news_sentiment.negative_count}")
        print(f"Neutral News: {result.news_sentiment.neutral_count}")
        
        if result.news_sentiment.key_themes:
            print(f"Key Themes: {', '.join(result.news_sentiment.key_themes)}")
    else:
        print("News sentiment data not available")
    
    # Recent news section
    if result.news_items and verbose:
        print("\nRecent News:")
        print("-" * 20)
        
        # Display up to 5 most recent news items
        for i, news in enumerate(result.news_items[:5], 1):
            print(f"{i}. {news.title}")
            print(f"   Source: {news.source} | Date: {news.published_at.strftime('%Y-%m-%d')}")
            if news.sentiment is not None:
                sentiment_text = "Neutral"
                if news.sentiment >= 0.5:
                    sentiment_text = "Very Positive"
                elif news.sentiment >= 0.2:
                    sentiment_text = "Positive"
                elif news.sentiment <= -0.5:
                    sentiment_text = "Very Negative"
                elif news.sentiment <= -0.2:
                    sentiment_text = "Negative"
                print(f"   Sentiment: {sentiment_text} ({news.sentiment:.2f})")
            print()
    
    # Recommendations section
    if result.analysis_result and result.analysis_result.recommendations:
        print("\nRecommendations:")
        print("-" * 20)
        
        for i, rec in enumerate(result.analysis_result.recommendations[:3], 1):
            print(f"{i}. {rec}")
    
    print("\n" + "=" * 50)


def _display_comprehensive_report_summary(report: ComprehensiveAnalysisReport, verbose: bool = False) -> None:
    """Display a summary of a comprehensive analysis report for multiple securities.
    
    Args:
        report: ComprehensiveAnalysisReport to display
        verbose: Whether to display detailed information
    """
    print("\nComprehensive Analysis Report Summary")
    print("=" * 50)
    
    # Basic report statistics
    print(f"\nTotal securities analyzed: {report.total_securities}")
    print(f"Successful analyses: {report.successful_analyses}")
    print(f"Failed analyses: {report.failed_analyses}")
    print(f"Success rate: {report.success_rate:.1f}%")
    print(f"Execution time: {report.execution_time:.2f} seconds")
    
    if report.failed_symbols:
        print(f"Failed symbols: {', '.join(report.failed_symbols)}")
    
    # Count stocks vs ETFs
    stocks = [r for r in report.results if r.analysis_result and isinstance(r.analysis_result.stock_info, StockInfo)]
    etfs = [r for r in report.results if r.analysis_result and isinstance(r.analysis_result.stock_info, ETFInfo)]
    
    print(f"\nStocks analyzed: {len(stocks)}")
    print(f"ETFs analyzed: {len(etfs)}")
    
    # Recommendations summary
    if report.results:
        buy_count = sum(1 for r in report.results if r.analysis_result and 
                        r.analysis_result.fair_value.recommendation == "BUY")
        sell_count = sum(1 for r in report.results if r.analysis_result and 
                         r.analysis_result.fair_value.recommendation == "SELL")
        hold_count = sum(1 for r in report.results if r.analysis_result and 
                         r.analysis_result.fair_value.recommendation == "HOLD")
        
        print("\nRecommendation Summary:")
        print(f"  Buy recommendations: {buy_count}")
        print(f"  Hold recommendations: {hold_count}")
        print(f"  Sell recommendations: {sell_count}")
    
    # Health scores summary
    if report.results:
        health_scores = [r.analysis_result.health_score.overall_score for r in report.results 
                         if r.analysis_result]
        
        if health_scores:
            avg_health = sum(health_scores) / len(health_scores)
            max_health = max(health_scores)
            min_health = min(health_scores)
            
            print("\nHealth Score Summary:")
            print(f"  Average health score: {avg_health:.1f}/100")
            print(f"  Highest health score: {max_health:.1f}/100")
            print(f"  Lowest health score: {min_health:.1f}/100")
    
    # News sentiment summary
    sentiment_results = [r for r in report.results if r.news_sentiment]
    if sentiment_results:
        sentiment_scores = [r.news_sentiment.overall_sentiment for r in sentiment_results]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Convert sentiment score to descriptive text
        sentiment_text = "Neutral"
        if avg_sentiment >= 0.5:
            sentiment_text = "Very Positive"
        elif avg_sentiment >= 0.2:
            sentiment_text = "Positive"
        elif avg_sentiment <= -0.5:
            sentiment_text = "Very Negative"
        elif avg_sentiment <= -0.2:
            sentiment_text = "Negative"
        
        print("\nNews Sentiment Summary:")
        print(f"  Average sentiment: {sentiment_text} ({avg_sentiment:.2f})")
        
        # Most positive and negative stocks
        if len(sentiment_results) > 1:
            most_positive = max(sentiment_results, key=lambda r: r.news_sentiment.overall_sentiment)
            most_negative = min(sentiment_results, key=lambda r: r.news_sentiment.overall_sentiment)
            
            print(f"  Most positive news: {most_positive.symbol} ({most_positive.news_sentiment.overall_sentiment:.2f})")
            print(f"  Most negative news: {most_negative.symbol} ({most_negative.news_sentiment.overall_sentiment:.2f})")
    
    # Display individual security summaries if verbose
    if verbose and report.results:
        print("\nIndividual Security Summaries:")
        print("=" * 50)
        
        for result in report.results:
            _display_comprehensive_result_summary(result, verbose=False)
    
    print("\n" + "=" * 50)


def display_comparative_analysis(report: ComprehensiveAnalysisReport) -> None:
    """Display comparative analysis for multiple stocks.
    
    Args:
        report: ComprehensiveAnalysisReport containing multiple stock results
    """
    if not report.results or len(report.results) < 2:
        print("Comparative analysis requires at least two successfully analyzed securities.")
        return
    
    print("\nComparative Analysis")
    print("=" * 50)
    
    # Get comparative metrics
    comparative_metrics = report.get_comparative_metrics()
    
    # Display price comparison
    if 'price' in comparative_metrics:
        print("\nPrice Comparison:")
        _display_metric_table(comparative_metrics['price'], 'Current Price ($)')
    
    # Display health score comparison
    if 'health_score' in comparative_metrics:
        print("\nHealth Score Comparison:")
        _display_metric_table(comparative_metrics['health_score'], 'Health Score (0-100)')
    
    # Display PE ratio comparison
    if 'pe_ratio' in comparative_metrics:
        print("\nP/E Ratio Comparison:")
        _display_metric_table(comparative_metrics['pe_ratio'], 'P/E Ratio')
    
    # Display PB ratio comparison
    if 'pb_ratio' in comparative_metrics:
        print("\nP/B Ratio Comparison:")
        _display_metric_table(comparative_metrics['pb_ratio'], 'P/B Ratio')
    
    # Display dividend yield comparison
    if 'dividend_yield' in comparative_metrics:
        # Convert to percentage
        for item in comparative_metrics['dividend_yield']:
            item['value'] = item['value'] * 100
        
        print("\nDividend Yield Comparison:")
        _display_metric_table(comparative_metrics['dividend_yield'], 'Dividend Yield (%)')
    
    # Display sentiment comparison
    if 'sentiment' in comparative_metrics:
        print("\nNews Sentiment Comparison:")
        _display_metric_table(comparative_metrics['sentiment'], 'Sentiment (-1 to 1)')
    
    print("\n" + "=" * 50)


def _display_metric_table(metrics: List[Dict[str, Any]], metric_name: str) -> None:
    """Display a table of metrics for multiple securities.
    
    Args:
        metrics: List of dictionaries containing symbol and value
        metric_name: Name of the metric being displayed
    """
    # Sort by value (descending)
    sorted_metrics = sorted(metrics, key=lambda x: x['value'], reverse=True)
    
    # Display header
    print(f"{'Rank':<5} {'Symbol':<10} {metric_name:<15}")
    print("-" * 30)
    
    # Display data
    for i, metric in enumerate(sorted_metrics, 1):
        print(f"{i:<5} {metric['symbol']:<10} {metric['value']:.2f}")
    
    print()