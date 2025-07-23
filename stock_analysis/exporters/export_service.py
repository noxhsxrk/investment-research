"""Export service for Power BI integration.

This module provides comprehensive export functionality for analysis results
in formats optimized for Power BI consumption including CSV, Excel, and JSON.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd

from stock_analysis.models.data_models import AnalysisResult, StockInfo, ETFInfo
from stock_analysis.utils.exceptions import ExportError
from stock_analysis.utils.logging import get_logger

logger = get_logger(__name__)


class ExportService:
    """Service for exporting analysis results in Power BI compatible formats.
    
    Supports CSV, Excel, and JSON exports with proper schemas and metadata
    for seamless Power BI integration.
    """
    
    def __init__(self, output_dir: str = "./exports"):
        """Initialize the export service.
        
        Args:
            output_dir: Directory where exported files will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"ExportService initialized with output directory: {output_dir}")
    
    def export_to_json(self, data: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """Export data to JSON format.
        
        Args:
            data: List of dictionaries to export
            filename: Optional filename (without extension)
            
        Returns:
            Path to the exported file
        """
        # This is an alias for export_to_powerbi_json for backward compatibility
        return self.export_to_powerbi_json(data, filename)
    
    def export_to_powerbi_json(self, data: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """Export data to Power BI compatible JSON format.
        
        Args:
            data: List of dictionaries to export
            filename: Optional filename (without extension)
            
        Returns:
            Path to the exported file
        """
        if not data:
            logger.warning("No data to export")
            return ""
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"powerbi_data_{timestamp}"
        
        # Ensure the filename has .json extension
        if not filename.endswith(".json"):
            filename = f"{filename}.json"
        
        # Create full path
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            # Convert data to a serializable format first
            serializable_data = []
            for item in data:
                if hasattr(item, '__dict__'):
                    # This is an object, not a dict
                    serialized_item = self._json_serializer(item)
                    serializable_data.append(serialized_item)
                else:
                    # This is already a dict
                    serializable_data.append(item)
            
            # Convert data to Power BI compatible format
            powerbi_data = self._convert_to_powerbi_format(serializable_data)
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(powerbi_data, f, indent=2, default=self._json_serializer)
            
            logger.info(f"Successfully exported {len(data)} records to Power BI JSON: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting to Power BI JSON: {str(e)}")
            return ""
    
    def _convert_to_powerbi_format(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert data to Power BI compatible format.
        
        Args:
            data: List of dictionaries to convert
        
        Returns:
            Power BI compatible data structure
        """
        return {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "record_count": len(data)
            },
            "data": data
        }
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for handling non-serializable objects.
        
        Args:
            obj: Object to serialize
            
        Returns:
            Serialized representation of the object
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle AnalysisResult objects
        from stock_analysis.models.data_models import AnalysisResult
        if isinstance(obj, AnalysisResult):
            return {
                'symbol': obj.symbol,
                'timestamp': obj.timestamp,
                'stock_info': self._serialize_stock_info(obj.stock_info),
                'financial_ratios': self._serialize_financial_ratios(obj.financial_ratios),
                'health_score': self._serialize_health_score(obj.health_score),
                'fair_value': self._serialize_fair_value(obj.fair_value),
                'sentiment': self._serialize_sentiment(obj.sentiment),
                'recommendations': obj.recommendations
            }
            
        # Handle ComprehensiveAnalysisResult objects
        from stock_analysis.models.comprehensive_models import ComprehensiveAnalysisResult
        if isinstance(obj, ComprehensiveAnalysisResult):
            result = {
                'symbol': obj.symbol,
                'timestamp': obj.timestamp,
                'analysis_result': obj.analysis_result,
                'news_sentiment': obj.news_sentiment
            }
            
            # Handle financial statements separately
            if obj.financial_statements:
                result['financial_statements'] = {}
                for key, value in obj.financial_statements.items():
                    if value is not None and hasattr(value, 'to_dict'):
                        # Convert DataFrame to dict with string keys
                        df_dict = {}
                        for col in value.columns:
                            col_dict = {}
                            for idx in value.index:
                                col_dict[str(idx)] = value.loc[idx, col]
                            df_dict[str(col)] = col_dict
                        result['financial_statements'][key] = df_dict
                    else:
                        result['financial_statements'][key] = None
            else:
                result['financial_statements'] = {}
                
            # Handle news items separately
            if obj.news_items:
                result['news_items'] = [
                    {
                        'title': item.title,
                        'source': item.source,
                        'url': item.url,
                        'published_at': item.published_at,
                        'summary': item.summary,
                        'sentiment': item.sentiment,
                        'impact': item.impact,
                        'categories': item.categories
                    } for item in obj.news_items
                ]
            else:
                result['news_items'] = []
                
            return result
            
        # Handle ComprehensiveAnalysisReport objects
        from stock_analysis.models.comprehensive_models import ComprehensiveAnalysisReport
        if isinstance(obj, ComprehensiveAnalysisReport):
            return {
                'results': obj.results,
                'total_securities': obj.total_securities,
                'successful_analyses': obj.successful_analyses,
                'failed_analyses': obj.failed_analyses,
                'failed_symbols': obj.failed_symbols,
                'execution_time': obj.execution_time,
                'timestamp': obj.timestamp,
                'success_rate': obj.success_rate
            }
            
        # If we get here, we don't know how to serialize this object
        raise TypeError(f"Type {type(obj)} not serializable")
        
    def _serialize_stock_info(self, stock_info):
        """Serialize StockInfo or ETFInfo objects."""
        if stock_info is None:
            return None
            
        # Convert to dictionary with all attributes
        return {key: value for key, value in stock_info.__dict__.items()}
        
    def _serialize_financial_ratios(self, financial_ratios):
        """Serialize FinancialRatios objects."""
        if financial_ratios is None:
            return None
            
        result = {}
        
        # Handle nested ratio objects
        for ratio_type in ['liquidity_ratios', 'profitability_ratios', 'leverage_ratios', 'efficiency_ratios']:
            ratio_obj = getattr(financial_ratios, ratio_type, None)
            if ratio_obj:
                result[ratio_type] = {key: value for key, value in ratio_obj.__dict__.items()}
            else:
                result[ratio_type] = None
                
        return result
        
    def _serialize_health_score(self, health_score):
        """Serialize HealthScore objects."""
        if health_score is None:
            return None
            
        return {key: value for key, value in health_score.__dict__.items()}
        
    def _serialize_fair_value(self, fair_value):
        """Serialize FairValueResult objects."""
        if fair_value is None:
            return None
            
        return {key: value for key, value in fair_value.__dict__.items()}
        
    def _serialize_sentiment(self, sentiment):
        """Serialize SentimentResult objects."""
        if sentiment is None:
            return None
            
        return {key: value for key, value in sentiment.__dict__.items()}
    
    def export_to_csv(self, data: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """Export data to CSV format.
        
        Args:
            data: List of dictionaries to export
            filename: Optional filename (without extension)
            
        Returns:
            Path to the exported file
        """
        if not data:
            logger.warning("No data to export")
            return ""
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_export_{timestamp}.csv"
        elif not filename.endswith(".csv"):
            filename = f"{filename}.csv"
        
        # Create full path
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            # Write to CSV
            with open(filepath, 'w', newline='') as csvfile:
                if data:
                    writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
            
            logger.info(f"Successfully exported {len(data)} records to CSV: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            return ""
    
    def export_to_excel(self, data: Any, filename: Optional[str] = None) -> str:
        """Export data to Excel format.
        
        Args:
            data: Data to export (can be a list of dictionaries, DataFrame, or multi-dimensional array)
            filename: Optional filename (without extension)
            
        Returns:
            Path to the exported file
        """
        if data is None:
            logger.warning("No data to export")
            return ""
    
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_export_{timestamp}.xlsx"
        elif not filename.endswith(".xlsx"):
            filename = f"{filename}.xlsx"
        
        # Create full path
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            # Pre-process data to handle timezone-aware datetimes for list of dictionaries
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                processed_data = []
                for item in data:
                    processed_item = {}
                    for key, value in item.items():
                        # Convert timezone-aware datetimes to strings
                        if isinstance(value, datetime):
                            processed_item[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            processed_item[key] = value
                    processed_data.append(processed_item)
                data = processed_data
            
            # Handle different data types
            if isinstance(data, dict) and "income_statement" in data and "balance_sheet" in data and "cash_flow" in data:
                # Financial statements dictionary - create multi-sheet Excel file
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    # Income Statement
                    if isinstance(data["income_statement"], pd.DataFrame):
                        income_df = data["income_statement"].copy()
                        # Convert index to strings and fix problematic dates
                        income_df.index = [str(idx).replace('-00-', '-12-').replace('-0-', '-12-') for idx in income_df.index]
                        income_df.to_excel(writer, sheet_name='Income_Statement')
                    
                    # Balance Sheet
                    if isinstance(data["balance_sheet"], pd.DataFrame):
                        balance_df = data["balance_sheet"].copy()
                        # Convert index to strings and fix problematic dates
                        balance_df.index = [str(idx).replace('-00-', '-12-').replace('-0-', '-12-') for idx in balance_df.index]
                        balance_df.to_excel(writer, sheet_name='Balance_Sheet')
                    
                    # Cash Flow
                    if isinstance(data["cash_flow"], pd.DataFrame):
                        cash_df = data["cash_flow"].copy()
                        # Convert index to strings and fix problematic dates
                        cash_df.index = [str(idx).replace('-00-', '-12-').replace('-0-', '-12-') for idx in cash_df.index]
                        cash_df.to_excel(writer, sheet_name='Cash_Flow')
                    
                    # Financial Ratios (if available)
                    if "financial_ratios" in data and isinstance(data["financial_ratios"], pd.DataFrame):
                        ratios_df = data["financial_ratios"].copy()
                        # Convert index to strings and fix problematic dates
                        ratios_df.index = [str(idx).replace('-00-', '-12-').replace('-0-', '-12-') for idx in ratios_df.index]
                        ratios_df.to_excel(writer, sheet_name='Financial_Ratios')
                
                logger.info(f"Successfully exported financial statements to Excel: {filepath}")
                return filepath
            elif isinstance(data, dict):
                # Dictionary with multiple data items - create multi-sheet Excel file
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    for sheet_name, sheet_data in data.items():
                        if isinstance(sheet_data, pd.DataFrame):
                            sheet_df = sheet_data.copy()
                            # Convert index to strings and fix problematic dates
                            sheet_df.index = [str(idx).replace('-00-', '-12-').replace('-0-', '-12-') for idx in sheet_df.index]
                            # Convert datetime columns to strings
                            for col in sheet_df.columns:
                                if pd.api.types.is_datetime64_any_dtype(sheet_df[col]):
                                    sheet_df[col] = sheet_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                            sheet_df.to_excel(writer, sheet_name=sheet_name)
                        elif isinstance(sheet_data, (list, dict)):
                            # Pre-process list data to handle timezone-aware datetimes
                            if isinstance(sheet_data, list) and all(isinstance(item, dict) for item in sheet_data):
                                processed_sheet_data = []
                                for item in sheet_data:
                                    processed_item = {}
                                    for key, value in item.items():
                                        if isinstance(value, datetime):
                                            processed_item[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                                        else:
                                            processed_item[key] = value
                                    processed_sheet_data.append(processed_item)
                                sheet_data = processed_sheet_data
                            
                            sheet_df = pd.DataFrame(sheet_data)
                            # Convert datetime columns to strings
                            for col in sheet_df.columns:
                                if pd.api.types.is_datetime64_any_dtype(sheet_df[col]):
                                    sheet_df[col] = sheet_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                            sheet_df.to_excel(writer, sheet_name=sheet_name)
                
                logger.info(f"Successfully exported multi-sheet data to Excel: {filepath}")
                return filepath
            elif isinstance(data, pd.DataFrame):
                # Already a DataFrame
                df = data.copy()
                # Convert index to strings and fix problematic dates
                df.index = [str(idx).replace('-00-', '-12-').replace('-0-', '-12-') for idx in df.index]
                # Convert datetime columns to strings
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                # Export to Excel
                df.to_excel(filepath, index=True)
            elif isinstance(data, list) and all(hasattr(item, '__dict__') and hasattr(item, 'symbol') and hasattr(item, 'analysis_result') for item in data):
                # This is a list of ComprehensiveAnalysisResult objects
                return self._export_comprehensive_results_to_excel(data, filepath)
            elif isinstance(data, list) and all(hasattr(item, '__dict__') and hasattr(item, 'symbol') and hasattr(item, 'stock_info') for item in data):
                # This is a list of AnalysisResult objects
                return self._export_analysis_results_to_excel(data, filepath)
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                # List of dictionaries - already pre-processed above
                df = pd.DataFrame(data)
                # Convert datetime columns to strings
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                # Export to Excel
                df.to_excel(filepath, index=True)
            elif hasattr(data, 'shape') and len(getattr(data, 'shape', ())) > 1:
                # Multi-dimensional array
                df = pd.DataFrame(data)
                df.to_excel(filepath, index=True)
            else:
                # Try to convert to DataFrame
                df = pd.DataFrame(data)
                # Convert datetime columns to strings
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                df.to_excel(filepath, index=True)
            
            logger.info(f"Successfully exported data to Excel: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}")
            return ""
    
    def export_comprehensive_results(self, 
                                    results: List[Any], 
                                    export_format: str = "excel", 
                                    filename: Optional[str] = None) -> str:
        """Export comprehensive analysis results in the specified format.
        
        This method handles the export of comprehensive analysis results,
        ensuring proper organization of data in different export formats.
        
        Args:
            results: List of ComprehensiveAnalysisResult objects
            export_format: Export format ('csv', 'excel', 'json')
            filename: Optional filename (without extension)
            
        Returns:
            Path to the exported file
            
        Raises:
            ExportError: If export fails
        """
        logger.info(f"Exporting comprehensive analysis results in {export_format} format")
        
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"comprehensive_analysis_{timestamp}"
            
            # Export based on format
            if export_format == "json":
                return self.export_comprehensive_to_json(results, filename)
            elif export_format == "excel":
                return self.export_comprehensive_to_excel(results, filename)
            elif export_format == "csv":
                return self.export_comprehensive_to_csv(results, filename)
            else:
                raise ExportError(f"Unsupported export format: {export_format}")
        except Exception as e:
            logger.error(f"Failed to export comprehensive results: {str(e)}")
            raise ExportError(f"Failed to export comprehensive results: {str(e)}")
    
    def _export_analysis_results_to_excel(self, results: List[Any], filepath: str) -> str:
        """Export analysis results to Excel with well-formatted sheets.
        
        Args:
            results: List of AnalysisResult objects
            filepath: Path to save the Excel file
            
        Returns:
            Path to the exported file
        """
        from stock_analysis.models.data_models import AnalysisResult, StockInfo, ETFInfo
        
        try:
            # Create Excel writer with openpyxl engine
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Create summary sheet
                summary_data = []
                for result in results:
                    summary_item = {
                        'Symbol': result.symbol,
                        'Timestamp': result.timestamp,
                        'Current Price': result.stock_info.current_price,
                        'Fair Value': result.fair_value.average_fair_value,
                        'Recommendation': result.fair_value.recommendation,
                        'Health Score': result.health_score.overall_score,
                        'Risk Assessment': result.health_score.risk_assessment,
                        'Sentiment': result.sentiment.overall_sentiment
                    }
                    
                    # Add stock-specific fields
                    if isinstance(result.stock_info, StockInfo):
                        summary_item['Company Name'] = result.stock_info.company_name
                        summary_item['Sector'] = getattr(result.stock_info, 'sector', 'N/A')
                        summary_item['Industry'] = getattr(result.stock_info, 'industry', 'N/A')
                        summary_item['P/E Ratio'] = getattr(result.stock_info, 'pe_ratio', 'N/A')
                        summary_item['P/B Ratio'] = getattr(result.stock_info, 'pb_ratio', 'N/A')
                        summary_item['Dividend Yield'] = getattr(result.stock_info, 'dividend_yield', 'N/A')
                    # Add ETF-specific fields
                    elif isinstance(result.stock_info, ETFInfo):
                        summary_item['ETF Name'] = result.stock_info.name
                        summary_item['NAV'] = getattr(result.stock_info, 'nav', 'N/A')
                        summary_item['Expense Ratio'] = getattr(result.stock_info, 'expense_ratio', 'N/A')
                        summary_item['AUM'] = getattr(result.stock_info, 'assets_under_management', 'N/A')
                        summary_item['Category'] = getattr(result.stock_info, 'category', 'N/A')
                    
                    summary_data.append(summary_item)
                
                # Create summary DataFrame and export to Excel
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Create financial ratios sheet
                ratios_data = []
                for result in results:
                    if result.financial_ratios:
                        ratio_item = {'Symbol': result.symbol}
                        
                        # Add liquidity ratios
                        if result.financial_ratios.liquidity_ratios:
                            for key, value in result.financial_ratios.liquidity_ratios.__dict__.items():
                                if not key.startswith('_'):
                                    ratio_item[f'Liquidity: {key}'] = value
                        
                        # Add profitability ratios
                        if result.financial_ratios.profitability_ratios:
                            for key, value in result.financial_ratios.profitability_ratios.__dict__.items():
                                if not key.startswith('_'):
                                    ratio_item[f'Profitability: {key}'] = value
                        
                        # Add leverage ratios
                        if result.financial_ratios.leverage_ratios:
                            for key, value in result.financial_ratios.leverage_ratios.__dict__.items():
                                if not key.startswith('_'):
                                    ratio_item[f'Leverage: {key}'] = value
                        
                        # Add efficiency ratios
                        if result.financial_ratios.efficiency_ratios:
                            for key, value in result.financial_ratios.efficiency_ratios.__dict__.items():
                                if not key.startswith('_'):
                                    ratio_item[f'Efficiency: {key}'] = value
                        
                        ratios_data.append(ratio_item)
                
                # Create ratios DataFrame and export to Excel if we have data
                if ratios_data:
                    ratios_df = pd.DataFrame(ratios_data)
                    ratios_df.to_excel(writer, sheet_name='Financial_Ratios', index=False)
                
                # Create health scores sheet
                health_data = []
                for result in results:
                    health_item = {
                        'Symbol': result.symbol,
                        'Overall Score': result.health_score.overall_score,
                        'Financial Strength': result.health_score.financial_strength,
                        'Profitability Health': result.health_score.profitability_health,
                        'Liquidity Health': result.health_score.liquidity_health,
                        'Risk Assessment': result.health_score.risk_assessment
                    }
                    health_data.append(health_item)
                
                # Create health DataFrame and export to Excel
                health_df = pd.DataFrame(health_data)
                health_df.to_excel(writer, sheet_name='Health_Scores', index=False)
                
                # Create valuation sheet
                valuation_data = []
                for result in results:
                    valuation_item = {
                        'Symbol': result.symbol,
                        'Current Price': result.fair_value.current_price,
                        'DCF Value': result.fair_value.dcf_value,
                        'Peer Comparison Value': result.fair_value.peer_comparison_value,
                        'Average Fair Value': result.fair_value.average_fair_value,
                        'Recommendation': result.fair_value.recommendation,
                        'Confidence Level': result.fair_value.confidence_level
                    }
                    valuation_data.append(valuation_item)
                
                # Create valuation DataFrame and export to Excel
                valuation_df = pd.DataFrame(valuation_data)
                valuation_df.to_excel(writer, sheet_name='Valuation', index=False)
                
                # Create sentiment sheet
                sentiment_data = []
                for result in results:
                    sentiment_item = {
                        'Symbol': result.symbol,
                        'Overall Sentiment': result.sentiment.overall_sentiment,
                        'Positive Count': result.sentiment.positive_count,
                        'Negative Count': result.sentiment.negative_count,
                        'Neutral Count': result.sentiment.neutral_count
                    }
                    
                    # Add key themes if available
                    if hasattr(result.sentiment, 'key_themes') and result.sentiment.key_themes:
                        sentiment_item['Key Themes'] = ', '.join(result.sentiment.key_themes)
                    
                    sentiment_data.append(sentiment_item)
                
                # Create sentiment DataFrame and export to Excel
                sentiment_df = pd.DataFrame(sentiment_data)
                sentiment_df.to_excel(writer, sheet_name='Sentiment', index=False)
                
                # Create recommendations sheet
                recommendations_data = []
                for result in results:
                    for i, rec in enumerate(result.recommendations, 1):
                        recommendations_data.append({
                            'Symbol': result.symbol,
                            'Recommendation #': i,
                            'Recommendation': rec
                        })
                
                # Create recommendations DataFrame and export to Excel
                recommendations_df = pd.DataFrame(recommendations_data)
                recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
                
                # Format the Excel file
                workbook = writer.book
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    # Auto-adjust column widths
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = (max_length + 2)
                        worksheet.column_dimensions[column_letter].width = min(adjusted_width, 50)
            
            logger.info(f"Successfully exported analysis results to Excel: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting analysis results to Excel: {str(e)}")
            return ""
    
    def export_comprehensive_to_json(self, results: List[Any], filename: Optional[str] = None) -> str:
        """Export comprehensive analysis results to JSON format.
        
        Args:
            results: List of ComprehensiveAnalysisResult objects
            filename: Optional filename (without extension)
            
        Returns:
            Path to the exported file
        """
        # Convert results to dictionaries
        results_dict = [result.to_dict() for result in results]
        
        # Ensure the filename has .json extension
        if filename and not filename.endswith(".json"):
            filename = f"{filename}.json"
        
        # Use the existing JSON export method
        return self.export_to_json(results_dict, filename)
    
    def export_comprehensive_to_csv(self, results: List[Any], filename: Optional[str] = None) -> str:
        """Export comprehensive analysis results to CSV format.
        
        Args:
            results: List of ComprehensiveAnalysisResult objects
            filename: Optional filename (without extension)
            
        Returns:
            Path to the exported file
        """
        # Flatten results for CSV export
        flattened_results = []
        
        for result in results:
            # Create base record with symbol and timestamp
            base_record = {
                'symbol': result.symbol,
                'timestamp': result.timestamp
            }
            
            # Add analysis data if available
            if result.analysis_result:
                analysis_dict = result.analysis_result.to_dict()
                for key, value in analysis_dict.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            base_record[f"analysis_{key}_{subkey}"] = subvalue
                    else:
                        base_record[f"analysis_{key}"] = value
            
            # Add key financial metrics if available
            if result.financial_statements:
                for statement_type, statement in result.financial_statements.items():
                    if statement is not None and isinstance(statement, pd.DataFrame) and not statement.empty:
                        # Get the most recent year's data for key metrics
                        latest_year = statement.columns[0]
                        if statement_type == 'income_statement':
                            if 'Revenue' in statement.index:
                                base_record['revenue'] = statement.loc['Revenue', latest_year]
                            if 'Net Income' in statement.index:
                                base_record['net_income'] = statement.loc['Net Income', latest_year]
                        elif statement_type == 'balance_sheet':
                            if 'Total Assets' in statement.index:
                                base_record['total_assets'] = statement.loc['Total Assets', latest_year]
                            if 'Total Liabilities' in statement.index:
                                base_record['total_liabilities'] = statement.loc['Total Liabilities', latest_year]
            
            # Add news sentiment if available
            if result.news_sentiment:
                sentiment_dict = result.news_sentiment.to_dict()
                for key, value in sentiment_dict.items():
                    base_record[f"news_sentiment_{key}"] = value
            
            flattened_results.append(base_record)
        
        # Ensure the filename has .csv extension
        if filename and not filename.endswith(".csv"):
            filename = f"{filename}.csv"
        
        # Use the existing CSV export method
        return self.export_to_csv(flattened_results, filename)
    
    def _export_comprehensive_results_to_excel(self, results: List[Any], filepath: str) -> str:
        """Export comprehensive analysis results to Excel with well-formatted sheets.
        
        Args:
            results: List of ComprehensiveAnalysisResult objects
            filepath: Path to save the Excel file
            
        Returns:
            Path to the exported file
        """
        try:
            # Create Excel writer with openpyxl engine
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Create summary sheet
                summary_data = []
                for result in results:
                    summary_item = {
                        'Symbol': result.symbol,
                        'Timestamp': result.timestamp.replace(tzinfo=None) if hasattr(result.timestamp, 'tzinfo') else result.timestamp
                    }
                    
                    # Add analysis data if available
                    if result.analysis_result:
                        summary_item['Current Price'] = result.analysis_result.stock_info.current_price
                        summary_item['Fair Value'] = result.analysis_result.fair_value.average_fair_value
                        summary_item['Recommendation'] = result.analysis_result.fair_value.recommendation
                        summary_item['Health Score'] = result.analysis_result.health_score.overall_score
                        summary_item['Risk Assessment'] = result.analysis_result.health_score.risk_assessment
                        
                        # Add stock-specific metrics
                        if hasattr(result.analysis_result.stock_info, 'company_name'):
                            summary_item['Company Name'] = result.analysis_result.stock_info.company_name
                        
                        if hasattr(result.analysis_result.stock_info, 'sector'):
                            summary_item['Sector'] = result.analysis_result.stock_info.sector
                        
                        if hasattr(result.analysis_result.stock_info, 'industry'):
                            summary_item['Industry'] = result.analysis_result.stock_info.industry
                        
                        if hasattr(result.analysis_result.stock_info, 'pe_ratio'):
                            summary_item['P/E Ratio'] = result.analysis_result.stock_info.pe_ratio
                        
                        if hasattr(result.analysis_result.stock_info, 'pb_ratio'):
                            summary_item['P/B Ratio'] = result.analysis_result.stock_info.pb_ratio
                        
                        if hasattr(result.analysis_result.stock_info, 'dividend_yield'):
                            summary_item['Dividend Yield'] = result.analysis_result.stock_info.dividend_yield
                    
                    # Add financial statement data if available
                    if result.financial_statements:
                        income_stmt = result.financial_statements.get('income_statement')
                        if income_stmt is not None and isinstance(income_stmt, pd.DataFrame) and not income_stmt.empty:
                            # Get the most recent year's data
                            latest_year = income_stmt.columns[0]
                            if 'Revenue' in income_stmt.index:
                                summary_item['Revenue'] = income_stmt.loc['Revenue', latest_year]
                            if 'Net Income' in income_stmt.index:
                                summary_item['Net Income'] = income_stmt.loc['Net Income', latest_year]
                        
                        balance_sheet = result.financial_statements.get('balance_sheet')
                        if balance_sheet is not None and isinstance(balance_sheet, pd.DataFrame) and not balance_sheet.empty:
                            latest_year = balance_sheet.columns[0]
                            if 'Total Assets' in balance_sheet.index:
                                summary_item['Total Assets'] = balance_sheet.loc['Total Assets', latest_year]
                            if 'Total Liabilities' in balance_sheet.index:
                                summary_item['Total Liabilities'] = balance_sheet.loc['Total Liabilities', latest_year]
                    
                    # Add news sentiment if available
                    if result.news_sentiment:
                        summary_item['News Sentiment'] = result.news_sentiment.overall_sentiment
                        summary_item['Positive News'] = result.news_sentiment.positive_count
                        summary_item['Negative News'] = result.news_sentiment.negative_count
                        summary_item['Neutral News'] = result.news_sentiment.neutral_count
                    
                    # Add news count if available
                    if result.news_items:
                        summary_item['News Count'] = len(result.news_items)
                    
                    summary_data.append(summary_item)
                
                # Create summary DataFrame and export to Excel
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Create financial statements sheets
                for result in results:
                    if result.financial_statements:
                        for statement_type, statement_data in result.financial_statements.items():
                            if statement_data is not None and isinstance(statement_data, pd.DataFrame) and not statement_data.empty:
                                sheet_name = f"{result.symbol}_{statement_type.replace('_', ' ').title()}"
                                # Limit sheet name length to 31 characters (Excel limitation)
                                sheet_name = sheet_name[:31]
                                statement_data.to_excel(writer, sheet_name=sheet_name)
                
                # Create news sheet
                news_data = []
                for result in results:
                    if result.news_items:
                        for news_item in result.news_items:
                            news_data.append({
                                'Symbol': result.symbol,
                                'Title': news_item.title,
                                'Source': news_item.source,
                                'Published At': news_item.published_at.replace(tzinfo=None) if hasattr(news_item.published_at, 'tzinfo') else news_item.published_at,
                                'Sentiment': getattr(news_item, 'sentiment', None),
                                'URL': news_item.url
                            })
                
                if news_data:
                    news_df = pd.DataFrame(news_data)
                    news_df.to_excel(writer, sheet_name='News', index=False)
                
                # Format the Excel file
                workbook = writer.book
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    # Auto-adjust column widths
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = (max_length + 2)
                        worksheet.column_dimensions[column_letter].width = min(adjusted_width, 50)
            
            logger.info(f"Successfully exported comprehensive results to Excel: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting comprehensive results to Excel: {str(e)}")
            return ""
    
    def export_comprehensive_to_excel(self, results: List[Any], filename: Optional[str] = None) -> str:
        """Export comprehensive analysis results to Excel with multiple worksheets.
        
        Args:
            results: List of ComprehensiveAnalysisResult objects
            filename: Optional filename (without extension)
            
        Returns:
            Path to the exported file
        """
        # Prepare data for multi-sheet Excel export
        excel_data = {}
        
        # Summary sheet with key metrics from all securities
        summary_data = []
        for result in results:
            summary_record = {
                'symbol': result.symbol,
                'timestamp': result.timestamp
            }
            
            # Add analysis summary if available
            if result.analysis_result:
                summary_record['current_price'] = result.analysis_result.stock_info.current_price
                summary_record['fair_value'] = result.analysis_result.fair_value.average_fair_value
                summary_record['recommendation'] = result.analysis_result.fair_value.recommendation
                summary_record['health_score'] = result.analysis_result.health_score.overall_score
                summary_record['risk_assessment'] = result.analysis_result.health_score.risk_assessment
                
                # Add stock-specific metrics
                if hasattr(result.analysis_result.stock_info, 'pe_ratio'):
                    summary_record['pe_ratio'] = result.analysis_result.stock_info.pe_ratio
                
                if hasattr(result.analysis_result.stock_info, 'pb_ratio'):
                    summary_record['pb_ratio'] = result.analysis_result.stock_info.pb_ratio
                
                if hasattr(result.analysis_result.stock_info, 'dividend_yield'):
                    summary_record['dividend_yield'] = result.analysis_result.stock_info.dividend_yield
            
            # Add key financial metrics if available
            if result.financial_statements:
                income_stmt = result.financial_statements.get('income_statement')
                if income_stmt is not None and isinstance(income_stmt, pd.DataFrame) and not income_stmt.empty:
                    # Get the most recent year's data
                    latest_year = income_stmt.columns[0]
                    if 'Revenue' in income_stmt.index:
                        summary_record['revenue'] = income_stmt.loc['Revenue', latest_year]
                    if 'Net Income' in income_stmt.index:
                        summary_record['net_income'] = income_stmt.loc['Net Income', latest_year]
            
            # Add news sentiment if available
            if result.news_sentiment:
                summary_record['news_sentiment'] = result.news_sentiment.overall_sentiment
                summary_record['positive_news'] = result.news_sentiment.positive_count
                summary_record['negative_news'] = result.news_sentiment.negative_count
            
            summary_data.append(summary_record)
        
        excel_data['Summary'] = summary_data
        
        # Add individual sheets for each symbol
        for result in results:
            # Analysis sheet
            if result.analysis_result:
                sheet_name = f"{result.symbol}_Analysis"
                excel_data[sheet_name] = result.analysis_result.to_dict()
            
            # Financial statements sheets
            if result.financial_statements:
                for statement_type, statement_data in result.financial_statements.items():
                    if statement_data is not None and isinstance(statement_data, pd.DataFrame) and not statement_data.empty:
                        sheet_name = f"{result.symbol}_{statement_type.replace('_', ' ').title()}"
                        excel_data[sheet_name] = statement_data
            
            # News sheet
            if result.news_items:
                sheet_name = f"{result.symbol}_News"
                news_data = [news_item.to_dict() for news_item in result.news_items]
                excel_data[sheet_name] = news_data
        
        # Comparative analysis sheet if multiple securities
        if len(results) > 1:
            comparative_data = {}
            
            # Financial metrics
            metrics = ['current_price', 'health_score', 'pe_ratio', 'pb_ratio', 'dividend_yield', 'news_sentiment']
            
            for metric in metrics:
                metric_data = []
                for result in results:
                    if metric == 'current_price' and result.analysis_result:
                        metric_data.append({
                            'symbol': result.symbol,
                            'value': result.analysis_result.stock_info.current_price
                        })
                    elif metric == 'health_score' and result.analysis_result:
                        metric_data.append({
                            'symbol': result.symbol,
                            'value': result.analysis_result.health_score.overall_score
                        })
                    elif metric == 'pe_ratio' and result.analysis_result and hasattr(result.analysis_result.stock_info, 'pe_ratio'):
                        metric_data.append({
                            'symbol': result.symbol,
                            'value': result.analysis_result.stock_info.pe_ratio
                        })
                    elif metric == 'pb_ratio' and result.analysis_result and hasattr(result.analysis_result.stock_info, 'pb_ratio'):
                        metric_data.append({
                            'symbol': result.symbol,
                            'value': result.analysis_result.stock_info.pb_ratio
                        })
                    elif metric == 'dividend_yield' and result.analysis_result and hasattr(result.analysis_result.stock_info, 'dividend_yield'):
                        metric_data.append({
                            'symbol': result.symbol,
                            'value': result.analysis_result.stock_info.dividend_yield
                        })
                    elif metric == 'news_sentiment' and result.news_sentiment:
                        metric_data.append({
                            'symbol': result.symbol,
                            'value': result.news_sentiment.overall_sentiment
                        })
                
                if metric_data:
                    comparative_data[metric] = metric_data
            
            if comparative_data:
                excel_data['Comparative_Analysis'] = comparative_data
        
        # Ensure the filename has .xlsx extension
        if filename and not filename.endswith(".xlsx"):
            filename = f"{filename}.xlsx"
        
        # Use the existing Excel export method
        return self.export_to_excel(excel_data, filename)