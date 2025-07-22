"""Tests for the export service module."""

import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from stock_analysis.exporters.export_service import ExportService
from stock_analysis.models.data_models import (
    AnalysisResult, EfficiencyRatios, FairValueResult, FinancialRatios,
    HealthScore, LeverageRatios, LiquidityRatios, ProfitabilityRatios,
    SentimentResult, StockInfo
)
from stock_analysis.utils.exceptions import ExportError


@pytest.fixture
def sample_stock_info():
    """Create sample stock info for testing."""
    return StockInfo(
        symbol="AAPL",
        company_name="Apple Inc.",
        current_price=150.0,
        market_cap=2500000000000,
        pe_ratio=25.5,
        pb_ratio=8.2,
        dividend_yield=0.005,
        beta=1.2,
        sector="Technology",
        industry="Consumer Electronics"
    )


@pytest.fixture
def sample_financial_ratios():
    """Create sample financial ratios for testing."""
    return FinancialRatios(
        liquidity_ratios=LiquidityRatios(
            current_ratio=1.5,
            quick_ratio=1.2,
            cash_ratio=0.8
        ),
        profitability_ratios=ProfitabilityRatios(
            gross_margin=0.38,
            operating_margin=0.25,
            net_profit_margin=0.21,
            return_on_assets=0.15,
            return_on_equity=0.85
        ),
        leverage_ratios=LeverageRatios(
            debt_to_equity=1.8,
            debt_to_assets=0.35,
            interest_coverage=12.5
        ),
        efficiency_ratios=EfficiencyRatios(
            asset_turnover=1.1,
            inventory_turnover=8.5,
            receivables_turnover=15.2
        )
    )


@pytest.fixture
def sample_health_score():
    """Create sample health score for testing."""
    return HealthScore(
        overall_score=85.0,
        financial_strength=88.0,
        profitability_health=82.0,
        liquidity_health=90.0,
        risk_assessment="Low"
    )


@pytest.fixture
def sample_fair_value():
    """Create sample fair value result for testing."""
    return FairValueResult(
        current_price=150.0,
        dcf_value=165.0,
        peer_comparison_value=155.0,
        average_fair_value=160.0,
        recommendation="BUY",
        confidence_level=0.8
    )


@pytest.fixture
def sample_sentiment():
    """Create sample sentiment result for testing."""
    return SentimentResult(
        overall_sentiment=0.3,
        positive_count=15,
        negative_count=5,
        neutral_count=10,
        key_themes=["earnings", "innovation", "market growth"],
        sentiment_trend=[0.2, 0.3, 0.4, 0.3, 0.3]
    )


@pytest.fixture
def sample_analysis_result(sample_stock_info, sample_financial_ratios, 
                          sample_health_score, sample_fair_value, sample_sentiment):
    """Create sample analysis result for testing."""
    return AnalysisResult(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        stock_info=sample_stock_info,
        financial_ratios=sample_financial_ratios,
        health_score=sample_health_score,
        fair_value=sample_fair_value,
        sentiment=sample_sentiment,
        recommendations=["Strong buy based on fundamentals", "Monitor for entry points"]
    )


@pytest.fixture
def export_service():
    """Create export service with temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield ExportService(output_directory=temp_dir)


class TestExportService:
    """Test cases for ExportService class."""
    
    def test_init_creates_output_directory(self):
        """Test that initialization creates output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_exports"
            service = ExportService(str(output_path))
            
            assert output_path.exists()
            assert output_path.is_dir()
    
    def test_export_to_csv_single_result(self, export_service, sample_analysis_result):
        """Test CSV export with single analysis result."""
        filepath = export_service.export_to_csv(sample_analysis_result, "test_single.csv")
        
        assert Path(filepath).exists()
        
        # Verify CSV content
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            assert len(rows) == 1
            row = rows[0]
            assert row['Symbol'] == 'AAPL'
            assert row['Company_Name'] == 'Apple Inc.'
            assert float(row['Current_Price']) == 150.0
            assert row['Valuation_Recommendation'] == 'BUY'
    
    def test_export_to_csv_multiple_results(self, export_service, sample_analysis_result):
        """Test CSV export with multiple analysis results."""
        # Create second result with different symbol (proper copy)
        import copy
        result2 = copy.deepcopy(sample_analysis_result)
        result2.symbol = "MSFT"
        result2.stock_info.symbol = "MSFT"
        result2.stock_info.company_name = "Microsoft Corporation"
        
        results = [sample_analysis_result, result2]
        filepath = export_service.export_to_csv(results, "test_multiple.csv")
        
        assert Path(filepath).exists()
        
        # Verify CSV content
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            assert len(rows) == 2
            symbols = [row['Symbol'] for row in rows]
            assert 'AAPL' in symbols
            assert 'MSFT' in symbols
    
    def test_export_to_csv_auto_filename(self, export_service, sample_analysis_result):
        """Test CSV export with auto-generated filename."""
        filepath = export_service.export_to_csv(sample_analysis_result)
        
        assert Path(filepath).exists()
        filename = Path(filepath).name
        assert filename.startswith('stock_analysis_')
        assert filename.endswith('.csv')
    
    def test_export_to_excel_single_result(self, export_service, sample_analysis_result):
        """Test Excel export with single analysis result."""
        filepath = export_service.export_to_excel(sample_analysis_result, "test_single.xlsx")
        
        assert Path(filepath).exists()
        
        # Verify Excel content
        excel_file = pd.ExcelFile(filepath)
        expected_sheets = ['Summary', 'Stock_Info', 'Financial_Ratios', 
                          'Health_Scores', 'Valuation', 'Sentiment', 'Recommendations']
        
        assert all(sheet in excel_file.sheet_names for sheet in expected_sheets)
        
        # Check summary sheet content
        summary_df = pd.read_excel(filepath, sheet_name='Summary')
        assert len(summary_df) == 1
        assert summary_df.iloc[0]['Symbol'] == 'AAPL'
        assert summary_df.iloc[0]['Company_Name'] == 'Apple Inc.'
    
    def test_export_to_excel_multiple_sheets(self, export_service, sample_analysis_result):
        """Test Excel export creates all expected sheets."""
        filepath = export_service.export_to_excel(sample_analysis_result, "test_sheets.xlsx")
        
        # Verify all sheets have data
        sheets_to_check = {
            'Summary': ['Symbol', 'Company_Name', 'Overall_Health_Score'],
            'Stock_Info': ['Symbol', 'Current_Price', 'Market_Cap'],
            'Financial_Ratios': ['Symbol', 'Current_Ratio', 'Gross_Margin'],
            'Health_Scores': ['Symbol', 'Overall_Health_Score', 'Risk_Assessment'],
            'Valuation': ['Symbol', 'Current_Price', 'Average_Fair_Value'],
            'Sentiment': ['Symbol', 'Overall_Sentiment', 'Positive_Count'],
            'Recommendations': ['Symbol', 'Recommendation', 'Recommendation_Number']
        }
        
        for sheet_name, expected_columns in sheets_to_check.items():
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            assert len(df) > 0
            for col in expected_columns:
                assert col in df.columns
    
    def test_export_to_powerbi_json_structure(self, export_service, sample_analysis_result):
        """Test Power BI JSON export structure."""
        filepath = export_service.export_to_powerbi_json(sample_analysis_result, "test.json")
        
        assert Path(filepath).exists()
        
        # Verify JSON structure
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert 'metadata' in data
        assert 'schema' in data
        assert 'data' in data
        
        # Check metadata
        metadata = data['metadata']
        assert 'export_timestamp' in metadata
        assert 'record_count' in metadata
        assert metadata['record_count'] == 1
        assert 'AAPL' in metadata['symbols']
        
        # Check data sections
        data_section = data['data']
        expected_sections = ['summary', 'stock_info', 'financial_ratios', 
                           'health_scores', 'valuation', 'sentiment', 'recommendations']
        
        for section in expected_sections:
            assert section in data_section
            assert len(data_section[section]) > 0
    
    def test_export_to_powerbi_json_schema(self, export_service, sample_analysis_result):
        """Test Power BI JSON export includes proper schema."""
        filepath = export_service.export_to_powerbi_json(sample_analysis_result, "test_schema.json")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        schema = data['schema']
        assert 'summary' in schema
        assert 'financial_ratios' in schema
        assert 'valuation' in schema
        
        # Check schema structure
        summary_schema = schema['summary']
        assert 'Symbol' in summary_schema
        assert summary_schema['Symbol']['type'] == 'string'
        assert 'description' in summary_schema['Symbol']
    
    def test_flatten_analysis_result(self, export_service, sample_analysis_result):
        """Test flattening of analysis result."""
        flattened = export_service._flatten_analysis_result(sample_analysis_result)
        
        # Check key fields are present
        assert flattened['Symbol'] == 'AAPL'
        assert flattened['Company_Name'] == 'Apple Inc.'
        assert flattened['Current_Price'] == 150.0
        assert flattened['Overall_Health_Score'] == 85.0
        assert flattened['Valuation_Recommendation'] == 'BUY'
        assert flattened['Overall_Sentiment'] == 0.3
        
        # Check that lists are joined with semicolons
        assert 'earnings; innovation; market growth' in flattened['Key_Themes']
        assert 'Strong buy based on fundamentals; Monitor for entry points' in flattened['Recommendations']
    
    def test_create_summary_data(self, export_service, sample_analysis_result):
        """Test creation of summary data."""
        summary_data = export_service._create_summary_data([sample_analysis_result])
        
        assert len(summary_data) == 1
        summary = summary_data[0]
        
        assert summary['Symbol'] == 'AAPL'
        assert summary['Company_Name'] == 'Apple Inc.'
        assert summary['Overall_Health_Score'] == 85.0
        assert summary['Risk_Assessment'] == 'Low'
        assert summary['Valuation_Recommendation'] == 'BUY'
    
    def test_create_financial_ratios_data(self, export_service, sample_analysis_result):
        """Test creation of financial ratios data."""
        ratios_data = export_service._create_financial_ratios_data([sample_analysis_result])
        
        assert len(ratios_data) == 1
        ratios = ratios_data[0]
        
        assert ratios['Symbol'] == 'AAPL'
        assert ratios['Current_Ratio'] == 1.5
        assert ratios['Gross_Margin'] == 0.38
        assert ratios['Debt_To_Equity'] == 1.8
        assert ratios['Asset_Turnover'] == 1.1
    
    def test_create_valuation_data(self, export_service, sample_analysis_result):
        """Test creation of valuation data."""
        valuation_data = export_service._create_valuation_data([sample_analysis_result])
        
        assert len(valuation_data) == 1
        valuation = valuation_data[0]
        
        assert valuation['Symbol'] == 'AAPL'
        assert valuation['Current_Price'] == 150.0
        assert valuation['DCF_Value'] == 165.0
        assert valuation['Average_Fair_Value'] == 160.0
        assert valuation['Recommendation'] == 'BUY'
        assert valuation['Price_vs_Fair_Value_Ratio'] == 150.0 / 160.0
    
    def test_create_sentiment_data(self, export_service, sample_analysis_result):
        """Test creation of sentiment data."""
        sentiment_data = export_service._create_sentiment_data([sample_analysis_result])
        
        assert len(sentiment_data) == 1
        sentiment = sentiment_data[0]
        
        assert sentiment['Symbol'] == 'AAPL'
        assert sentiment['Overall_Sentiment'] == 0.3
        assert sentiment['Positive_Count'] == 15
        assert sentiment['Total_Articles'] == 30  # 15 + 5 + 10
        assert 'earnings; innovation; market growth' in sentiment['Key_Themes']
    
    def test_create_recommendations_data(self, export_service, sample_analysis_result):
        """Test creation of recommendations data."""
        recommendations_data = export_service._create_recommendations_data([sample_analysis_result])
        
        assert len(recommendations_data) == 2  # Two recommendations in sample data
        
        rec1 = recommendations_data[0]
        assert rec1['Symbol'] == 'AAPL'
        assert rec1['Recommendation_Number'] == 1
        assert rec1['Recommendation'] == 'Strong buy based on fundamentals'
        
        rec2 = recommendations_data[1]
        assert rec2['Recommendation_Number'] == 2
        assert rec2['Recommendation'] == 'Monitor for entry points'
    
    def test_create_metadata(self, export_service, sample_analysis_result):
        """Test creation of metadata."""
        metadata = export_service._create_metadata([sample_analysis_result])
        
        assert 'export_timestamp' in metadata
        assert metadata['record_count'] == 1
        assert metadata['symbols'] == ['AAPL']
        assert 'data_range' in metadata
        assert metadata['version'] == '1.0'
    
    def test_create_schema_definition(self, export_service):
        """Test creation of schema definition."""
        schema = export_service._create_schema_definition()
        
        assert 'summary' in schema
        assert 'financial_ratios' in schema
        assert 'valuation' in schema
        
        # Check that each field has type and description
        for section_name, section_schema in schema.items():
            for field_name, field_info in section_schema.items():
                assert 'type' in field_info
                assert 'description' in field_info
    
    def test_export_error_handling(self, export_service):
        """Test error handling in export methods."""
        # Test with invalid data
        with pytest.raises(ExportError):
            export_service.export_to_csv(None)
    
    @patch('stock_analysis.exporters.export_service.open')
    def test_csv_export_file_error(self, mock_open, export_service, sample_analysis_result):
        """Test CSV export handles file errors."""
        mock_open.side_effect = IOError("Permission denied")
        
        with pytest.raises(ExportError) as exc_info:
            export_service.export_to_csv(sample_analysis_result, "test.csv")
        
        assert "Failed to export to CSV" in str(exc_info.value)
    
    @patch('pandas.ExcelWriter')
    def test_excel_export_file_error(self, mock_writer, export_service, sample_analysis_result):
        """Test Excel export handles file errors."""
        mock_writer.side_effect = IOError("Permission denied")
        
        with pytest.raises(ExportError) as exc_info:
            export_service.export_to_excel(sample_analysis_result, "test.xlsx")
        
        assert "Failed to export to Excel" in str(exc_info.value)
    
    @patch('stock_analysis.exporters.export_service.open')
    def test_json_export_file_error(self, mock_open, export_service, sample_analysis_result):
        """Test JSON export handles file errors."""
        mock_open.side_effect = IOError("Permission denied")
        
        with pytest.raises(ExportError) as exc_info:
            export_service.export_to_powerbi_json(sample_analysis_result, "test.json")
        
        assert "Failed to export to Power BI JSON" in str(exc_info.value)