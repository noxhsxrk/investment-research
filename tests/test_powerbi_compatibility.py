"""Tests for Power BI compatibility of exported data formats."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from stock_analysis.exporters.export_service import ExportService
from stock_analysis.models.data_models import (
    AnalysisResult, EfficiencyRatios, FairValueResult, FinancialRatios,
    HealthScore, LeverageRatios, LiquidityRatios, ProfitabilityRatios,
    SentimentResult, StockInfo
)


@pytest.fixture
def sample_analysis_result():
    """Create a sample analysis result for testing."""
    return AnalysisResult(
        symbol="AAPL",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        stock_info=StockInfo(
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
        ),
        financial_ratios=FinancialRatios(
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
        ),
        health_score=HealthScore(
            overall_score=85.0,
            financial_strength=88.0,
            profitability_health=82.0,
            liquidity_health=90.0,
            risk_assessment="Low"
        ),
        fair_value=FairValueResult(
            current_price=150.0,
            dcf_value=165.0,
            peer_comparison_value=155.0,
            average_fair_value=160.0,
            recommendation="BUY",
            confidence_level=0.8
        ),
        sentiment=SentimentResult(
            overall_sentiment=0.3,
            positive_count=15,
            negative_count=5,
            neutral_count=10,
            key_themes=["earnings", "innovation", "market growth"],
            sentiment_trend=[0.2, 0.3, 0.4, 0.3, 0.3]
        ),
        recommendations=["Strong buy based on fundamentals", "Monitor for entry points"]
    )


class TestPowerBICompatibility:
    """Test Power BI compatibility of exported data formats."""
    
    def test_csv_powerbi_compatibility(self, sample_analysis_result):
        """Test CSV export format compatibility with Power BI."""
        with tempfile.TemporaryDirectory() as temp_dir:
            export_service = ExportService(output_directory=temp_dir)
            filepath = export_service.export_to_csv(sample_analysis_result, "test_powerbi.csv")
            
            # Check if file exists
            assert Path(filepath).exists()
            
            # Try to load with pandas (simulating Power BI import)
            df = pd.read_csv(filepath)
            
            # Check essential columns for Power BI visualizations
            essential_columns = [
                'Symbol', 'Company_Name', 'Current_Price', 'Overall_Health_Score',
                'Valuation_Recommendation', 'Overall_Sentiment'
            ]
            for column in essential_columns:
                assert column in df.columns
            
            # Check data types are appropriate for Power BI
            assert pd.api.types.is_string_dtype(df['Symbol'])
            assert pd.api.types.is_numeric_dtype(df['Current_Price'])
            assert pd.api.types.is_numeric_dtype(df['Overall_Health_Score'])
    
    def test_excel_powerbi_compatibility(self, sample_analysis_result):
        """Test Excel export format compatibility with Power BI."""
        with tempfile.TemporaryDirectory() as temp_dir:
            export_service = ExportService(output_directory=temp_dir)
            filepath = export_service.export_to_excel(sample_analysis_result, "test_powerbi.xlsx")
            
            # Check if file exists
            assert Path(filepath).exists()
            
            # Try to load with pandas (simulating Power BI import)
            excel_file = pd.ExcelFile(filepath)
            
            # Check essential sheets for Power BI
            essential_sheets = [
                'Summary', 'Financial_Ratios', 'Health_Scores', 
                'Valuation', 'Sentiment'
            ]
            for sheet in essential_sheets:
                assert sheet in excel_file.sheet_names
                
                # Load each sheet and check for key columns
                df = pd.read_excel(filepath, sheet_name=sheet)
                assert 'Symbol' in df.columns
                assert len(df) > 0
    
    def test_json_powerbi_compatibility(self, sample_analysis_result):
        """Test JSON export format compatibility with Power BI."""
        with tempfile.TemporaryDirectory() as temp_dir:
            export_service = ExportService(output_directory=temp_dir)
            filepath = export_service.export_to_powerbi_json(sample_analysis_result, "test_powerbi.json")
            
            # Check if file exists
            assert Path(filepath).exists()
            
            # Try to load JSON (simulating Power BI import)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check essential sections for Power BI
            assert 'metadata' in data
            assert 'schema' in data
            assert 'data' in data
            
            # Check data sections
            essential_sections = [
                'summary', 'financial_ratios', 'health_scores', 
                'valuation', 'sentiment'
            ]
            for section in essential_sections:
                assert section in data['data']
                assert len(data['data'][section]) > 0
            
            # Check schema definitions - some sections might have different naming in schema
            schema_keys = data['schema'].keys()
            for section in essential_sections:
                # Allow for variations in naming (singular/plural or different prefixes)
                section_base = section.rstrip('s')
                found = any(section in key or section_base in key for key in schema_keys)
                assert found, f"Schema definition for {section} or {section_base} not found"
    
    def test_multiple_stocks_compatibility(self, sample_analysis_result):
        """Test compatibility with multiple stocks data."""
        # Create a second stock result
        import copy
        result2 = copy.deepcopy(sample_analysis_result)
        result2.symbol = "MSFT"
        result2.stock_info.symbol = "MSFT"
        result2.stock_info.company_name = "Microsoft Corporation"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_service = ExportService(output_directory=temp_dir)
            
            # Test CSV with multiple stocks
            csv_path = export_service.export_to_csv([sample_analysis_result, result2], "multi_stock.csv")
            df_csv = pd.read_csv(csv_path)
            assert len(df_csv) == 2
            assert set(df_csv['Symbol']) == {"AAPL", "MSFT"}
            
            # Test Excel with multiple stocks
            excel_path = export_service.export_to_excel([sample_analysis_result, result2], "multi_stock.xlsx")
            df_summary = pd.read_excel(excel_path, sheet_name='Summary')
            assert len(df_summary) == 2
            assert set(df_summary['Symbol']) == {"AAPL", "MSFT"}
            
            # Test JSON with multiple stocks
            json_path = export_service.export_to_powerbi_json([sample_analysis_result, result2], "multi_stock.json")
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            assert len(json_data['data']['summary']) == 2
            symbols = {item['Symbol'] for item in json_data['data']['summary']}
            assert symbols == {"AAPL", "MSFT"}