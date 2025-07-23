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
            # Convert data to Power BI compatible format
            powerbi_data = self._convert_to_powerbi_format(data)
            
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
        raise TypeError(f"Type {type(obj)} not serializable")
    
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
    
