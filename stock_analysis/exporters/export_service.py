"""Export service for Power BI integration.

This module provides comprehensive export functionality for stock analysis results
in formats optimized for Power BI consumption including CSV, Excel, and JSON.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from stock_analysis.models.data_models import AnalysisResult
from stock_analysis.utils.exceptions import ExportError
from stock_analysis.utils.logging import get_logger

logger = get_logger(__name__)


class ExportService:
    """Service for exporting stock analysis results in Power BI compatible formats.
    
    Supports CSV, Excel, and JSON exports with proper schemas and metadata
    for seamless Power BI integration.
    """
    
    def __init__(self, output_directory: str = "./exports"):
        """Initialize the export service.
        
        Args:
            output_directory: Directory where exported files will be saved.
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"ExportService initialized with output directory: {self.output_directory}")
    
    def export_to_csv(self, 
                     data: Union[AnalysisResult, List[AnalysisResult]], 
                     filename: Optional[str] = None) -> str:
        """Export analysis results to CSV format optimized for Power BI.
        
        Creates a flattened CSV structure that Power BI can easily consume
        with proper column naming and data types.
        
        Args:
            data: Single analysis result or list of results to export.
            filename: Optional custom filename. If not provided, generates timestamp-based name.
            
        Returns:
            str: Path to the exported CSV file.
            
        Raises:
            ExportError: If export operation fails.
        """
        try:
            # Ensure data is a list
            if isinstance(data, AnalysisResult):
                data = [data]
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"stock_analysis_{timestamp}.csv"
            
            filepath = self.output_directory / filename
            
            # Convert to flattened structure for CSV
            flattened_data = []
            for result in data:
                flattened_row = self._flatten_analysis_result(result)
                flattened_data.append(flattened_row)
            
            # Write to CSV
            if flattened_data:
                fieldnames = flattened_data[0].keys()
                with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(flattened_data)
            
            logger.info(f"Successfully exported {len(data)} records to CSV: {filepath}")
            return str(filepath)
            
        except Exception as e:
            error_msg = f"Failed to export to CSV: {str(e)}"
            logger.error(error_msg)
            raise ExportError(error_msg) from e
    
    def export_to_excel(self, 
                       data: Union[AnalysisResult, List[AnalysisResult]], 
                       filename: Optional[str] = None) -> str:
        """Export analysis results to Excel format with multiple sheets.
        
        Creates a comprehensive Excel workbook with separate sheets for different
        data categories, formatted for Power BI consumption.
        
        Args:
            data: Single analysis result or list of results to export.
            filename: Optional custom filename. If not provided, generates timestamp-based name.
            
        Returns:
            str: Path to the exported Excel file.
            
        Raises:
            ExportError: If export operation fails.
        """
        try:
            # Ensure data is a list
            if isinstance(data, AnalysisResult):
                data = [data]
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"stock_analysis_{timestamp}.xlsx"
            elif not filename.endswith('.xlsx'):
                filename = f"{filename}.xlsx"
            
            filepath = self.output_directory / filename
            
            # Create Excel writer
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Summary sheet with key metrics
                summary_data = self._create_summary_data(data)
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Stock info sheet
                stock_info_data = self._create_stock_info_data(data)
                stock_info_df = pd.DataFrame(stock_info_data)
                stock_info_df.to_excel(writer, sheet_name='Stock_Info', index=False)
                
                # Financial ratios sheet
                ratios_data = self._create_financial_ratios_data(data)
                ratios_df = pd.DataFrame(ratios_data)
                ratios_df.to_excel(writer, sheet_name='Financial_Ratios', index=False)
                
                # Health scores sheet
                health_data = self._create_health_score_data(data)
                health_df = pd.DataFrame(health_data)
                health_df.to_excel(writer, sheet_name='Health_Scores', index=False)
                
                # Valuation sheet
                valuation_data = self._create_valuation_data(data)
                valuation_df = pd.DataFrame(valuation_data)
                valuation_df.to_excel(writer, sheet_name='Valuation', index=False)
                
                # Sentiment sheet
                sentiment_data = self._create_sentiment_data(data)
                sentiment_df = pd.DataFrame(sentiment_data)
                sentiment_df.to_excel(writer, sheet_name='Sentiment', index=False)
                
                # Recommendations sheet
                recommendations_data = self._create_recommendations_data(data)
                recommendations_df = pd.DataFrame(recommendations_data)
                recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            logger.info(f"Successfully exported {len(data)} records to Excel: {filepath}")
            return str(filepath)
            
        except Exception as e:
            error_msg = f"Failed to export to Excel: {str(e)}"
            logger.error(error_msg)
            raise ExportError(error_msg) from e
    
    def export_to_powerbi_json(self, 
                              data: Union[AnalysisResult, List[AnalysisResult]], 
                              filename: Optional[str] = None) -> str:
        """Export analysis results to JSON format with Power BI metadata.
        
        Creates a structured JSON file with data dictionaries and metadata
        optimized for Power BI REST API consumption.
        
        Args:
            data: Single analysis result or list of results to export.
            filename: Optional custom filename. If not provided, generates timestamp-based name.
            
        Returns:
            str: Path to the exported JSON file.
            
        Raises:
            ExportError: If export operation fails.
        """
        try:
            # Ensure data is a list
            if isinstance(data, AnalysisResult):
                data = [data]
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"stock_analysis_{timestamp}.json"
            
            filepath = self.output_directory / filename
            
            # Create Power BI optimized JSON structure
            powerbi_data = {
                "metadata": self._create_metadata(data),
                "schema": self._create_schema_definition(),
                "data": {
                    "summary": self._create_summary_data(data),
                    "stock_info": self._create_stock_info_data(data),
                    "financial_ratios": self._create_financial_ratios_data(data),
                    "health_scores": self._create_health_score_data(data),
                    "valuation": self._create_valuation_data(data),
                    "sentiment": self._create_sentiment_data(data),
                    "recommendations": self._create_recommendations_data(data)
                }
            }
            
            # Write to JSON file
            with open(filepath, 'w', encoding='utf-8') as jsonfile:
                json.dump(powerbi_data, jsonfile, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Successfully exported {len(data)} records to Power BI JSON: {filepath}")
            return str(filepath)
            
        except Exception as e:
            error_msg = f"Failed to export to Power BI JSON: {str(e)}"
            logger.error(error_msg)
            raise ExportError(error_msg) from e    

    def _flatten_analysis_result(self, result: AnalysisResult) -> Dict:
        """Flatten analysis result into a single dictionary for CSV export.
        
        Args:
            result: Analysis result to flatten.
            
        Returns:
            Dict: Flattened dictionary with all data points.
        """
        flattened = {
            # Basic info
            'Symbol': result.symbol,
            'Timestamp': result.timestamp.isoformat(),
            'Company_Name': result.stock_info.company_name,
            'Current_Price': result.stock_info.current_price,
            'Market_Cap': result.stock_info.market_cap,
            'Sector': result.stock_info.sector,
            'Industry': result.stock_info.industry,
            
            # Stock ratios
            'PE_Ratio': result.stock_info.pe_ratio,
            'PB_Ratio': result.stock_info.pb_ratio,
            'Dividend_Yield': result.stock_info.dividend_yield,
            'Beta': result.stock_info.beta,
            
            # Liquidity ratios
            'Current_Ratio': result.financial_ratios.liquidity_ratios.current_ratio,
            'Quick_Ratio': result.financial_ratios.liquidity_ratios.quick_ratio,
            'Cash_Ratio': result.financial_ratios.liquidity_ratios.cash_ratio,
            
            # Profitability ratios
            'Gross_Margin': result.financial_ratios.profitability_ratios.gross_margin,
            'Operating_Margin': result.financial_ratios.profitability_ratios.operating_margin,
            'Net_Profit_Margin': result.financial_ratios.profitability_ratios.net_profit_margin,
            'Return_On_Assets': result.financial_ratios.profitability_ratios.return_on_assets,
            'Return_On_Equity': result.financial_ratios.profitability_ratios.return_on_equity,
            
            # Leverage ratios
            'Debt_To_Equity': result.financial_ratios.leverage_ratios.debt_to_equity,
            'Debt_To_Assets': result.financial_ratios.leverage_ratios.debt_to_assets,
            'Interest_Coverage': result.financial_ratios.leverage_ratios.interest_coverage,
            
            # Efficiency ratios
            'Asset_Turnover': result.financial_ratios.efficiency_ratios.asset_turnover,
            'Inventory_Turnover': result.financial_ratios.efficiency_ratios.inventory_turnover,
            'Receivables_Turnover': result.financial_ratios.efficiency_ratios.receivables_turnover,
            
            # Health scores
            'Overall_Health_Score': result.health_score.overall_score,
            'Financial_Strength': result.health_score.financial_strength,
            'Profitability_Health': result.health_score.profitability_health,
            'Liquidity_Health': result.health_score.liquidity_health,
            'Risk_Assessment': result.health_score.risk_assessment,
            
            # Valuation
            'DCF_Value': result.fair_value.dcf_value,
            'Peer_Comparison_Value': result.fair_value.peer_comparison_value,
            'Average_Fair_Value': result.fair_value.average_fair_value,
            'Valuation_Recommendation': result.fair_value.recommendation,
            'Confidence_Level': result.fair_value.confidence_level,
            
            # Sentiment
            'Overall_Sentiment': result.sentiment.overall_sentiment,
            'Positive_News_Count': result.sentiment.positive_count,
            'Negative_News_Count': result.sentiment.negative_count,
            'Neutral_News_Count': result.sentiment.neutral_count,
            'Key_Themes': '; '.join(result.sentiment.key_themes),
            'Sentiment_Trend': '; '.join(map(str, result.sentiment.sentiment_trend)),
            
            # Recommendations
            'Recommendations': '; '.join(result.recommendations)
        }
        
        return flattened
    
    def _create_summary_data(self, data: List[AnalysisResult]) -> List[Dict]:
        """Create summary data for Power BI dashboard overview.
        
        Args:
            data: List of analysis results.
            
        Returns:
            List[Dict]: Summary data records.
        """
        summary_data = []
        for result in data:
            summary_data.append({
                'Symbol': result.symbol,
                'Company_Name': result.stock_info.company_name,
                'Timestamp': result.timestamp.isoformat(),
                'Current_Price': result.stock_info.current_price,
                'Market_Cap': result.stock_info.market_cap,
                'Overall_Health_Score': result.health_score.overall_score,
                'Risk_Assessment': result.health_score.risk_assessment,
                'Valuation_Recommendation': result.fair_value.recommendation,
                'Average_Fair_Value': result.fair_value.average_fair_value,
                'Overall_Sentiment': result.sentiment.overall_sentiment,
                'Sector': result.stock_info.sector,
                'Industry': result.stock_info.industry
            })
        return summary_data
    
    def _create_stock_info_data(self, data: List[AnalysisResult]) -> List[Dict]:
        """Create stock information data for detailed analysis.
        
        Args:
            data: List of analysis results.
            
        Returns:
            List[Dict]: Stock information records.
        """
        stock_info_data = []
        for result in data:
            stock_info_data.append({
                'Symbol': result.symbol,
                'Company_Name': result.stock_info.company_name,
                'Timestamp': result.timestamp.isoformat(),
                'Current_Price': result.stock_info.current_price,
                'Market_Cap': result.stock_info.market_cap,
                'PE_Ratio': result.stock_info.pe_ratio,
                'PB_Ratio': result.stock_info.pb_ratio,
                'Dividend_Yield': result.stock_info.dividend_yield,
                'Beta': result.stock_info.beta,
                'Sector': result.stock_info.sector,
                'Industry': result.stock_info.industry
            })
        return stock_info_data
    
    def _create_financial_ratios_data(self, data: List[AnalysisResult]) -> List[Dict]:
        """Create financial ratios data for ratio analysis.
        
        Args:
            data: List of analysis results.
            
        Returns:
            List[Dict]: Financial ratios records.
        """
        ratios_data = []
        for result in data:
            ratios_data.append({
                'Symbol': result.symbol,
                'Timestamp': result.timestamp.isoformat(),
                # Liquidity ratios
                'Current_Ratio': result.financial_ratios.liquidity_ratios.current_ratio,
                'Quick_Ratio': result.financial_ratios.liquidity_ratios.quick_ratio,
                'Cash_Ratio': result.financial_ratios.liquidity_ratios.cash_ratio,
                # Profitability ratios
                'Gross_Margin': result.financial_ratios.profitability_ratios.gross_margin,
                'Operating_Margin': result.financial_ratios.profitability_ratios.operating_margin,
                'Net_Profit_Margin': result.financial_ratios.profitability_ratios.net_profit_margin,
                'Return_On_Assets': result.financial_ratios.profitability_ratios.return_on_assets,
                'Return_On_Equity': result.financial_ratios.profitability_ratios.return_on_equity,
                # Leverage ratios
                'Debt_To_Equity': result.financial_ratios.leverage_ratios.debt_to_equity,
                'Debt_To_Assets': result.financial_ratios.leverage_ratios.debt_to_assets,
                'Interest_Coverage': result.financial_ratios.leverage_ratios.interest_coverage,
                # Efficiency ratios
                'Asset_Turnover': result.financial_ratios.efficiency_ratios.asset_turnover,
                'Inventory_Turnover': result.financial_ratios.efficiency_ratios.inventory_turnover,
                'Receivables_Turnover': result.financial_ratios.efficiency_ratios.receivables_turnover
            })
        return ratios_data
    
    def _create_health_score_data(self, data: List[AnalysisResult]) -> List[Dict]:
        """Create health score data for company health analysis.
        
        Args:
            data: List of analysis results.
            
        Returns:
            List[Dict]: Health score records.
        """
        health_data = []
        for result in data:
            health_data.append({
                'Symbol': result.symbol,
                'Timestamp': result.timestamp.isoformat(),
                'Overall_Health_Score': result.health_score.overall_score,
                'Financial_Strength': result.health_score.financial_strength,
                'Profitability_Health': result.health_score.profitability_health,
                'Liquidity_Health': result.health_score.liquidity_health,
                'Risk_Assessment': result.health_score.risk_assessment
            })
        return health_data
    
    def _create_valuation_data(self, data: List[AnalysisResult]) -> List[Dict]:
        """Create valuation data for investment analysis.
        
        Args:
            data: List of analysis results.
            
        Returns:
            List[Dict]: Valuation records.
        """
        valuation_data = []
        for result in data:
            valuation_data.append({
                'Symbol': result.symbol,
                'Timestamp': result.timestamp.isoformat(),
                'Current_Price': result.fair_value.current_price,
                'DCF_Value': result.fair_value.dcf_value,
                'Peer_Comparison_Value': result.fair_value.peer_comparison_value,
                'Average_Fair_Value': result.fair_value.average_fair_value,
                'Recommendation': result.fair_value.recommendation,
                'Confidence_Level': result.fair_value.confidence_level,
                'Price_vs_Fair_Value_Ratio': (
                    result.fair_value.current_price / result.fair_value.average_fair_value
                    if result.fair_value.average_fair_value > 0 else None
                )
            })
        return valuation_data
    
    def _create_sentiment_data(self, data: List[AnalysisResult]) -> List[Dict]:
        """Create sentiment data for news analysis.
        
        Args:
            data: List of analysis results.
            
        Returns:
            List[Dict]: Sentiment records.
        """
        sentiment_data = []
        for result in data:
            sentiment_data.append({
                'Symbol': result.symbol,
                'Timestamp': result.timestamp.isoformat(),
                'Overall_Sentiment': result.sentiment.overall_sentiment,
                'Positive_Count': result.sentiment.positive_count,
                'Negative_Count': result.sentiment.negative_count,
                'Neutral_Count': result.sentiment.neutral_count,
                'Total_Articles': (
                    result.sentiment.positive_count + 
                    result.sentiment.negative_count + 
                    result.sentiment.neutral_count
                ),
                'Key_Themes': '; '.join(result.sentiment.key_themes),
                'Sentiment_Trend': '; '.join(map(str, result.sentiment.sentiment_trend))
            })
        return sentiment_data
    
    def _create_recommendations_data(self, data: List[AnalysisResult]) -> List[Dict]:
        """Create recommendations data for action items.
        
        Args:
            data: List of analysis results.
            
        Returns:
            List[Dict]: Recommendations records.
        """
        recommendations_data = []
        for result in data:
            for i, recommendation in enumerate(result.recommendations, 1):
                recommendations_data.append({
                    'Symbol': result.symbol,
                    'Timestamp': result.timestamp.isoformat(),
                    'Recommendation_Number': i,
                    'Recommendation': recommendation,
                    'Overall_Health_Score': result.health_score.overall_score,
                    'Valuation_Recommendation': result.fair_value.recommendation,
                    'Risk_Assessment': result.health_score.risk_assessment
                })
        return recommendations_data
    
    def _create_metadata(self, data: List[AnalysisResult]) -> Dict:
        """Create metadata for Power BI JSON export.
        
        Args:
            data: List of analysis results.
            
        Returns:
            Dict: Metadata information.
        """
        return {
            'export_timestamp': datetime.now().isoformat(),
            'record_count': len(data),
            'symbols': [result.symbol for result in data],
            'data_range': {
                'earliest': min(result.timestamp for result in data).isoformat() if data else None,
                'latest': max(result.timestamp for result in data).isoformat() if data else None
            },
            'version': '1.0',
            'description': 'Stock analysis data exported for Power BI integration'
        }
    
    def _create_schema_definition(self) -> Dict:
        """Create schema definition for Power BI data types.
        
        Returns:
            Dict: Schema definition with data types and descriptions.
        """
        return {
            'summary': {
                'Symbol': {'type': 'string', 'description': 'Stock ticker symbol'},
                'Company_Name': {'type': 'string', 'description': 'Company name'},
                'Timestamp': {'type': 'datetime', 'description': 'Analysis timestamp'},
                'Current_Price': {'type': 'decimal', 'description': 'Current stock price'},
                'Market_Cap': {'type': 'decimal', 'description': 'Market capitalization'},
                'Overall_Health_Score': {'type': 'decimal', 'description': 'Overall health score (0-100)'},
                'Risk_Assessment': {'type': 'string', 'description': 'Risk level (Low/Medium/High)'},
                'Valuation_Recommendation': {'type': 'string', 'description': 'Buy/Hold/Sell recommendation'},
                'Average_Fair_Value': {'type': 'decimal', 'description': 'Average fair value estimate'},
                'Overall_Sentiment': {'type': 'decimal', 'description': 'News sentiment score (-1 to 1)'},
                'Sector': {'type': 'string', 'description': 'Business sector'},
                'Industry': {'type': 'string', 'description': 'Industry classification'}
            },
            'financial_ratios': {
                'Symbol': {'type': 'string', 'description': 'Stock ticker symbol'},
                'Timestamp': {'type': 'datetime', 'description': 'Analysis timestamp'},
                'Current_Ratio': {'type': 'decimal', 'description': 'Current assets / Current liabilities'},
                'Quick_Ratio': {'type': 'decimal', 'description': 'Quick assets / Current liabilities'},
                'Gross_Margin': {'type': 'decimal', 'description': 'Gross profit margin'},
                'Net_Profit_Margin': {'type': 'decimal', 'description': 'Net profit margin'},
                'Return_On_Equity': {'type': 'decimal', 'description': 'Return on equity ratio'},
                'Debt_To_Equity': {'type': 'decimal', 'description': 'Total debt / Total equity'}
            },
            'health_scores': {
                'Symbol': {'type': 'string', 'description': 'Stock ticker symbol'},
                'Timestamp': {'type': 'datetime', 'description': 'Analysis timestamp'},
                'Overall_Health_Score': {'type': 'decimal', 'description': 'Overall health score (0-100)'},
                'Financial_Strength': {'type': 'decimal', 'description': 'Financial strength score (0-100)'},
                'Profitability_Health': {'type': 'decimal', 'description': 'Profitability health score (0-100)'},
                'Liquidity_Health': {'type': 'decimal', 'description': 'Liquidity health score (0-100)'},
                'Risk_Assessment': {'type': 'string', 'description': 'Risk level (Low/Medium/High)'}
            },
            'valuation': {
                'Symbol': {'type': 'string', 'description': 'Stock ticker symbol'},
                'Timestamp': {'type': 'datetime', 'description': 'Analysis timestamp'},
                'Current_Price': {'type': 'decimal', 'description': 'Current stock price'},
                'DCF_Value': {'type': 'decimal', 'description': 'Discounted cash flow valuation'},
                'Peer_Comparison_Value': {'type': 'decimal', 'description': 'Peer comparison valuation'},
                'Average_Fair_Value': {'type': 'decimal', 'description': 'Average fair value estimate'},
                'Recommendation': {'type': 'string', 'description': 'Investment recommendation'},
                'Confidence_Level': {'type': 'decimal', 'description': 'Confidence in recommendation (0-1)'}
            },
            'sentiment': {
                'Symbol': {'type': 'string', 'description': 'Stock ticker symbol'},
                'Timestamp': {'type': 'datetime', 'description': 'Analysis timestamp'},
                'Overall_Sentiment': {'type': 'decimal', 'description': 'Overall sentiment score (-1 to 1)'},
                'Positive_Count': {'type': 'integer', 'description': 'Number of positive news articles'},
                'Negative_Count': {'type': 'integer', 'description': 'Number of negative news articles'},
                'Neutral_Count': {'type': 'integer', 'description': 'Number of neutral news articles'},
                'Total_Articles': {'type': 'integer', 'description': 'Total number of news articles analyzed'},
                'Key_Themes': {'type': 'string', 'description': 'Key themes identified in news articles'}
            }
        }