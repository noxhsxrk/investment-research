"""Comprehensive analysis data models.

This module contains data models for comprehensive stock analysis results,
including combined data from stock analysis, financial statements, and news.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd

from stock_analysis.models.data_models import AnalysisResult, SentimentResult, ValidationError
from stock_analysis.models.enhanced_data_models import NewsItem


@dataclass
class ComprehensiveAnalysisResult:
    """Represents the combined results of a comprehensive stock analysis.
    
    This class combines data from multiple sources including stock analysis,
    financial statements, and financial news to provide a complete view of
    a security's performance and outlook.
    """
    
    symbol: str
    timestamp: datetime = None
    analysis_result: Optional[AnalysisResult] = None
    financial_statements: Dict[str, Any] = None
    news_items: List[NewsItem] = None
    news_sentiment: Optional[SentimentResult] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        if self.financial_statements is None:
            self.financial_statements = {
                'income_statement': None,
                'balance_sheet': None,
                'cash_flow': None
            }
        
        if self.news_items is None:
            self.news_items = []
    
    def validate(self) -> None:
        """Validate the comprehensive analysis result data.
        
        Raises:
            ValidationError: If any validation checks fail.
        """
        if not self.symbol:
            raise ValidationError("Symbol cannot be empty")
        
        if not isinstance(self.symbol, str):
            raise ValidationError(f"Symbol must be a string: {self.symbol}")
        
        if not isinstance(self.timestamp, datetime):
            raise ValidationError(f"Timestamp must be a datetime object: {self.timestamp}")
        
        # Validate analysis result if present
        if self.analysis_result is not None:
            if not isinstance(self.analysis_result, AnalysisResult):
                raise ValidationError(f"Analysis result must be an AnalysisResult object: {self.analysis_result}")
            try:
                self.analysis_result.validate()
            except ValidationError as e:
                raise ValidationError(f"Invalid analysis result: {str(e)}")
        
        # Validate financial statements if present
        if self.financial_statements is not None:
            if not isinstance(self.financial_statements, dict):
                raise ValidationError(f"Financial statements must be a dictionary: {self.financial_statements}")
            
            for statement_name, statement in self.financial_statements.items():
                if statement is not None and not isinstance(statement, pd.DataFrame):
                    raise ValidationError(f"{statement_name} must be a DataFrame: {statement}")
        
        # Validate news items if present
        if self.news_items is not None:
            if not isinstance(self.news_items, list):
                raise ValidationError(f"News items must be a list: {self.news_items}")
            
            for i, news_item in enumerate(self.news_items):
                if not isinstance(news_item, NewsItem):
                    raise ValidationError(f"News item at index {i} must be a NewsItem object: {news_item}")
                try:
                    news_item.validate()
                except ValidationError as e:
                    raise ValidationError(f"Invalid news item at index {i}: {str(e)}")
        
        # Validate news sentiment if present
        if self.news_sentiment is not None:
            if not isinstance(self.news_sentiment, SentimentResult):
                raise ValidationError(f"News sentiment must be a SentimentResult object: {self.news_sentiment}")
            try:
                self.news_sentiment.validate()
            except ValidationError as e:
                raise ValidationError(f"Invalid news sentiment: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation of the comprehensive analysis result
        """
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'analysis': self.analysis_result.to_dict() if self.analysis_result else None,
            'financials': {
                'income_statement': self.financial_statements['income_statement'].to_dict() 
                    if self.financial_statements.get('income_statement') is not None else None,
                'balance_sheet': self.financial_statements['balance_sheet'].to_dict() 
                    if self.financial_statements.get('balance_sheet') is not None else None,
                'cash_flow': self.financial_statements['cash_flow'].to_dict() 
                    if self.financial_statements.get('cash_flow') is not None else None
            },
            'news': [news_item.to_dict() for news_item in self.news_items] if self.news_items else [],
            'news_sentiment': self.news_sentiment.to_dict() if self.news_sentiment else None
        }
    
    def get_key_metrics(self) -> Dict[str, Any]:
        """Extract key metrics from the comprehensive analysis result.
        
        Returns:
            Dictionary containing key metrics from the analysis
        """
        metrics = {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
        }
        
        # Extract key metrics from analysis result
        if self.analysis_result:
            metrics['current_price'] = self.analysis_result.stock_info.current_price
            metrics['health_score'] = self.analysis_result.health_score.overall_score
            metrics['recommendation'] = self.analysis_result.fair_value.recommendation
            metrics['fair_value'] = self.analysis_result.fair_value.average_fair_value
            
            # Add stock-specific metrics
            if hasattr(self.analysis_result.stock_info, 'pe_ratio'):
                metrics['pe_ratio'] = self.analysis_result.stock_info.pe_ratio
            
            if hasattr(self.analysis_result.stock_info, 'pb_ratio'):
                metrics['pb_ratio'] = self.analysis_result.stock_info.pb_ratio
            
            if hasattr(self.analysis_result.stock_info, 'dividend_yield'):
                metrics['dividend_yield'] = self.analysis_result.stock_info.dividend_yield
        
        # Extract key financial indicators
        if self.financial_statements:
            income_stmt = self.financial_statements.get('income_statement')
            if income_stmt is not None and not income_stmt.empty:
                # Get the most recent year's data
                latest_year = income_stmt.columns[0]
                if 'Revenue' in income_stmt.index:
                    metrics['revenue'] = income_stmt.loc['Revenue', latest_year]
                if 'Net Income' in income_stmt.index:
                    metrics['net_income'] = income_stmt.loc['Net Income', latest_year]
        
        # Extract news sentiment
        if self.news_sentiment:
            metrics['news_sentiment'] = self.news_sentiment.overall_sentiment
            metrics['positive_news'] = self.news_sentiment.positive_count
            metrics['negative_news'] = self.news_sentiment.negative_count
        
        return metrics


@dataclass
class ComprehensiveAnalysisReport:
    """Represents the results of a comprehensive analysis for multiple securities.
    
    This class aggregates the results of analyzing multiple securities and provides
    summary statistics and comparative analysis capabilities.
    """
    
    results: List[ComprehensiveAnalysisResult]
    total_securities: int
    successful_analyses: int
    failed_analyses: int
    failed_symbols: List[str]
    execution_time: float
    timestamp: datetime = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def validate(self) -> None:
        """Validate the comprehensive analysis report data.
        
        Raises:
            ValidationError: If any validation checks fail.
        """
        if not isinstance(self.results, list):
            raise ValidationError(f"Results must be a list: {self.results}")
        
        for i, result in enumerate(self.results):
            if not isinstance(result, ComprehensiveAnalysisResult):
                raise ValidationError(f"Result at index {i} must be a ComprehensiveAnalysisResult object: {result}")
            try:
                result.validate()
            except ValidationError as e:
                raise ValidationError(f"Invalid result at index {i}: {str(e)}")
        
        if not isinstance(self.total_securities, int):
            raise ValidationError(f"Total securities must be an integer: {self.total_securities}")
        
        if self.total_securities < 0:
            raise ValidationError(f"Total securities cannot be negative: {self.total_securities}")
        
        if not isinstance(self.successful_analyses, int):
            raise ValidationError(f"Successful analyses must be an integer: {self.successful_analyses}")
        
        if self.successful_analyses < 0:
            raise ValidationError(f"Successful analyses cannot be negative: {self.successful_analyses}")
        
        if not isinstance(self.failed_analyses, int):
            raise ValidationError(f"Failed analyses must be an integer: {self.failed_analyses}")
        
        if self.failed_analyses < 0:
            raise ValidationError(f"Failed analyses cannot be negative: {self.failed_analyses}")
        
        if self.successful_analyses + self.failed_analyses != self.total_securities:
            raise ValidationError(
                f"Sum of successful ({self.successful_analyses}) and failed ({self.failed_analyses}) "
                f"analyses must equal total securities ({self.total_securities})"
            )
        
        if not isinstance(self.failed_symbols, list):
            raise ValidationError(f"Failed symbols must be a list: {self.failed_symbols}")
        
        if len(self.failed_symbols) != self.failed_analyses:
            raise ValidationError(
                f"Length of failed symbols list ({len(self.failed_symbols)}) must equal "
                f"failed analyses count ({self.failed_analyses})"
            )
        
        if not isinstance(self.execution_time, (int, float)):
            raise ValidationError(f"Execution time must be a number: {self.execution_time}")
        
        if self.execution_time < 0:
            raise ValidationError(f"Execution time cannot be negative: {self.execution_time}")
        
        if not isinstance(self.timestamp, datetime):
            raise ValidationError(f"Timestamp must be a datetime object: {self.timestamp}")
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage.
        
        Returns:
            Success rate as a percentage
        """
        if self.total_securities == 0:
            return 0.0
        return (self.successful_analyses / self.total_securities) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation of the comprehensive analysis report
        """
        return {
            'results': [result.to_dict() for result in self.results],
            'total_securities': self.total_securities,
            'successful_analyses': self.successful_analyses,
            'failed_analyses': self.failed_analyses,
            'failed_symbols': self.failed_symbols,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp,
            'success_rate': self.success_rate
        }
    
    def get_comparative_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Extract comparative metrics for all securities in the report.
        
        Returns:
            Dictionary containing lists of metrics for comparison
        """
        if not self.results:
            return {}
        
        # Extract key metrics for each security
        metrics = {}
        
        # Financial metrics
        metrics['price'] = []
        metrics['health_score'] = []
        metrics['pe_ratio'] = []
        metrics['pb_ratio'] = []
        metrics['dividend_yield'] = []
        metrics['sentiment'] = []
        
        # Collect metrics from each result
        for result in self.results:
            if result.analysis_result:
                # Basic info
                security_metrics = {
                    'symbol': result.symbol,
                    'value': result.analysis_result.stock_info.current_price
                }
                metrics['price'].append(security_metrics)
                
                # Health score
                security_metrics = {
                    'symbol': result.symbol,
                    'value': result.analysis_result.health_score.overall_score
                }
                metrics['health_score'].append(security_metrics)
                
                # PE ratio
                if hasattr(result.analysis_result.stock_info, 'pe_ratio') and result.analysis_result.stock_info.pe_ratio:
                    security_metrics = {
                        'symbol': result.symbol,
                        'value': result.analysis_result.stock_info.pe_ratio
                    }
                    metrics['pe_ratio'].append(security_metrics)
                
                # PB ratio
                if hasattr(result.analysis_result.stock_info, 'pb_ratio') and result.analysis_result.stock_info.pb_ratio:
                    security_metrics = {
                        'symbol': result.symbol,
                        'value': result.analysis_result.stock_info.pb_ratio
                    }
                    metrics['pb_ratio'].append(security_metrics)
                
                # Dividend yield
                if hasattr(result.analysis_result.stock_info, 'dividend_yield') and result.analysis_result.stock_info.dividend_yield:
                    security_metrics = {
                        'symbol': result.symbol,
                        'value': result.analysis_result.stock_info.dividend_yield
                    }
                    metrics['dividend_yield'].append(security_metrics)
            
            # News sentiment
            if result.news_sentiment:
                security_metrics = {
                    'symbol': result.symbol,
                    'value': result.news_sentiment.overall_sentiment
                }
                metrics['sentiment'].append(security_metrics)
        
        # Remove empty metric categories
        return {k: v for k, v in metrics.items() if v}