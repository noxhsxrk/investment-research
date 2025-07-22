"""Tests for the stock analysis orchestrator.

This module contains comprehensive tests for the StockAnalysisOrchestrator class,
including unit tests, integration tests, and error handling scenarios.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from stock_analysis.orchestrator import StockAnalysisOrchestrator, AnalysisProgress, AnalysisReport
from stock_analysis.models.data_models import (
    StockInfo, FinancialRatios, LiquidityRatios, ProfitabilityRatios,
    LeverageRatios, EfficiencyRatios, HealthScore, FairValueResult,
    SentimentResult, AnalysisResult
)
from stock_analysis.utils.exceptions import DataRetrievalError, CalculationError, ExportError


class TestAnalysisProgress:
    """Test cases for AnalysisProgress class."""
    
    def test_completion_percentage_calculation(self):
        """Test completion percentage calculation."""
        progress = AnalysisProgress(
            total_stocks=10,
            completed_stocks=3,
            failed_stocks=2
        )
        
        assert progress.completion_percentage == 30.0
    
    def test_completion_percentage_zero_total(self):
        """Test completion percentage with zero total stocks."""
        progress = AnalysisProgress(
            total_stocks=0,
            completed_stocks=0,
            failed_stocks=0
        )
        
        assert progress.completion_percentage == 0.0
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        progress = AnalysisProgress(
            total_stocks=10,
            completed_stocks=7,
            failed_stocks=3
        )
        
        assert progress.success_rate == 70.0
    
    def test_success_rate_no_processed(self):
        """Test success rate with no processed stocks."""
        progress = AnalysisProgress(
            total_stocks=10,
            completed_stocks=0,
            failed_stocks=0
        )
        
        assert progress.success_rate == 0.0


class TestStockAnalysisOrchestrator:
    """Test cases for StockAnalysisOrchestrator class."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a test orchestrator instance."""
        return StockAnalysisOrchestrator(
            max_workers=2,
            enable_parallel_processing=False,
            continue_on_error=True
        )
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        with patch.multiple(
            'stock_analysis.orchestrator',
            StockDataService=Mock(),
            FinancialAnalysisEngine=Mock(),
            ValuationEngine=Mock(),
            NewsSentimentAnalyzer=Mock(),
            ExportService=Mock()
        ) as mocks:
            yield mocks
    
    @pytest.fixture
    def sample_stock_info(self):
        """Create sample stock info for testing."""
        return StockInfo(
            symbol="AAPL",
            company_name="Apple Inc.",
            current_price=150.0,
            market_cap=2500000000000,
            pe_ratio=25.0,
            pb_ratio=5.0,
            dividend_yield=0.02,
            beta=1.2,
            sector="Technology",
            industry="Consumer Electronics"
        )
    
    @pytest.fixture
    def sample_financial_ratios(self):
        """Create sample financial ratios for testing."""
        return FinancialRatios(
            liquidity_ratios=LiquidityRatios(current_ratio=2.0, quick_ratio=1.5, cash_ratio=0.8),
            profitability_ratios=ProfitabilityRatios(
                gross_margin=0.4, operating_margin=0.25, net_profit_margin=0.2,
                return_on_assets=0.15, return_on_equity=0.25
            ),
            leverage_ratios=LeverageRatios(debt_to_equity=0.5, debt_to_assets=0.3, interest_coverage=8.0),
            efficiency_ratios=EfficiencyRatios(asset_turnover=0.8, inventory_turnover=12.0, receivables_turnover=15.0)
        )
    
    @pytest.fixture
    def sample_health_score(self):
        """Create sample health score for testing."""
        return HealthScore(
            overall_score=75.0,
            financial_strength=80.0,
            profitability_health=70.0,
            liquidity_health=75.0,
            risk_assessment="Medium"
        )
    
    @pytest.fixture
    def sample_fair_value(self):
        """Create sample fair value result for testing."""
        return FairValueResult(
            current_price=150.0,
            dcf_value=160.0,
            peer_comparison_value=155.0,
            average_fair_value=157.5,
            recommendation="BUY",
            confidence_level=0.8
        )
    
    @pytest.fixture
    def sample_sentiment(self):
        """Create sample sentiment result for testing."""
        return SentimentResult(
            overall_sentiment=0.3,
            positive_count=15,
            negative_count=5,
            neutral_count=10,
            key_themes=["earnings", "growth", "innovation"],
            sentiment_trend=[0.2, 0.3, 0.4]
        )
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = StockAnalysisOrchestrator(
            max_workers=4,
            enable_parallel_processing=True,
            continue_on_error=False
        )
        
        assert orchestrator.max_workers == 4
        assert orchestrator.enable_parallel_processing is True
        assert orchestrator.continue_on_error is False
        assert orchestrator.current_progress is None
        assert len(orchestrator.progress_callbacks) == 0
    
    def test_add_progress_callback(self, orchestrator):
        """Test adding progress callbacks."""
        callback1 = Mock()
        callback2 = Mock()
        
        orchestrator.add_progress_callback(callback1)
        orchestrator.add_progress_callback(callback2)
        
        assert len(orchestrator.progress_callbacks) == 2
        assert callback1 in orchestrator.progress_callbacks
        assert callback2 in orchestrator.progress_callbacks
    
    def test_update_progress(self, orchestrator):
        """Test progress update and callback notification."""
        callback1 = Mock()
        callback2 = Mock()
        orchestrator.add_progress_callback(callback1)
        orchestrator.add_progress_callback(callback2)
        
        progress = AnalysisProgress(total_stocks=10, completed_stocks=5, failed_stocks=1)
        orchestrator._update_progress(progress)
        
        assert orchestrator.current_progress == progress
        callback1.assert_called_once_with(progress)
        callback2.assert_called_once_with(progress)
    
    def test_update_progress_callback_error(self, orchestrator):
        """Test progress update with callback error."""
        failing_callback = Mock(side_effect=Exception("Callback error"))
        working_callback = Mock()
        
        orchestrator.add_progress_callback(failing_callback)
        orchestrator.add_progress_callback(working_callback)
        
        progress = AnalysisProgress(total_stocks=10, completed_stocks=5, failed_stocks=1)
        
        # Should not raise exception even if callback fails
        orchestrator._update_progress(progress)
        
        assert orchestrator.current_progress == progress
        failing_callback.assert_called_once_with(progress)
        working_callback.assert_called_once_with(progress)
    
    @patch('stock_analysis.orchestrator.StockDataService')
    @patch('stock_analysis.orchestrator.FinancialAnalysisEngine')
    @patch('stock_analysis.orchestrator.ValuationEngine')
    @patch('stock_analysis.orchestrator.NewsSentimentAnalyzer')
    def test_analyze_single_stock_success(self, mock_sentiment, mock_valuation, mock_financial, mock_stock_data,
                                        sample_stock_info, sample_financial_ratios, sample_health_score,
                                        sample_fair_value, sample_sentiment):
        """Test successful single stock analysis."""
        # Setup mocks
        mock_stock_service = Mock()
        mock_financial_engine = Mock()
        mock_valuation_engine = Mock()
        mock_sentiment_analyzer = Mock()
        
        mock_stock_data.return_value = mock_stock_service
        mock_financial.return_value = mock_financial_engine
        mock_valuation.return_value = mock_valuation_engine
        mock_sentiment.return_value = mock_sentiment_analyzer
        
        # Configure mock returns
        mock_stock_service.get_stock_info.return_value = sample_stock_info
        mock_stock_service.get_financial_statements.return_value = Mock()  # DataFrame mock
        mock_financial_engine.calculate_financial_ratios.return_value = sample_financial_ratios
        mock_financial_engine.assess_company_health.return_value = sample_health_score
        mock_valuation_engine.calculate_fair_value.return_value = sample_fair_value
        mock_sentiment_analyzer.get_news_articles.return_value = []
        mock_sentiment_analyzer.analyze_sentiment.return_value = sample_sentiment
        
        # Create orchestrator and analyze
        orchestrator = StockAnalysisOrchestrator()
        result = orchestrator.analyze_single_stock("AAPL")
        
        # Verify result
        assert isinstance(result, AnalysisResult)
        assert result.symbol == "AAPL"
        assert result.stock_info == sample_stock_info
        assert result.financial_ratios == sample_financial_ratios
        assert result.health_score == sample_health_score
        assert result.fair_value == sample_fair_value
        assert result.sentiment == sample_sentiment
        assert len(result.recommendations) > 0
        
        # Verify service calls
        mock_stock_service.get_stock_info.assert_called_once_with("AAPL")
        assert mock_stock_service.get_financial_statements.call_count == 3
        mock_financial_engine.calculate_financial_ratios.assert_called_once()
        mock_financial_engine.assess_company_health.assert_called_once()
        mock_valuation_engine.calculate_fair_value.assert_called_once()
        mock_sentiment_analyzer.get_news_articles.assert_called_once()
        mock_sentiment_analyzer.analyze_sentiment.assert_called_once()
    
    @patch('stock_analysis.orchestrator.StockDataService')
    def test_analyze_single_stock_data_retrieval_error(self, mock_stock_data):
        """Test single stock analysis with data retrieval error."""
        mock_stock_service = Mock()
        mock_stock_data.return_value = mock_stock_service
        mock_stock_service.get_stock_info.side_effect = DataRetrievalError("Stock not found")
        
        orchestrator = StockAnalysisOrchestrator()
        
        with pytest.raises(DataRetrievalError):
            orchestrator.analyze_single_stock("INVALID")
    
    def test_analyze_multiple_stocks_sequential(self, orchestrator):
        """Test multiple stock analysis in sequential mode."""
        # Mock the analyze_single_stock method
        mock_result1 = Mock(spec=AnalysisResult)
        mock_result1.symbol = "AAPL"
        mock_result2 = Mock(spec=AnalysisResult)
        mock_result2.symbol = "GOOGL"
        
        orchestrator.analyze_single_stock = Mock(side_effect=[mock_result1, mock_result2])
        
        symbols = ["AAPL", "GOOGL"]
        report = orchestrator.analyze_multiple_stocks(symbols)
        
        # Verify report
        assert isinstance(report, AnalysisReport)
        assert report.total_stocks == 2
        assert report.successful_analyses == 2
        assert report.failed_analyses == 0
        assert report.success_rate == 100.0
        assert len(report.results) == 2
        assert len(report.failed_symbols) == 0
        
        # Verify method calls
        assert orchestrator.analyze_single_stock.call_count == 2
        orchestrator.analyze_single_stock.assert_any_call("AAPL")
        orchestrator.analyze_single_stock.assert_any_call("GOOGL")
    
    def test_analyze_multiple_stocks_with_failures(self, orchestrator):
        """Test multiple stock analysis with some failures."""
        mock_result = Mock(spec=AnalysisResult)
        mock_result.symbol = "AAPL"
        
        def mock_analyze(symbol):
            if symbol == "AAPL":
                return mock_result
            elif symbol == "INVALID":
                raise DataRetrievalError("Stock not found")
            else:
                raise CalculationError("Calculation failed")
        
        orchestrator.analyze_single_stock = Mock(side_effect=mock_analyze)
        
        symbols = ["AAPL", "INVALID", "GOOGL"]
        report = orchestrator.analyze_multiple_stocks(symbols)
        
        # Verify report
        assert report.total_stocks == 3
        assert report.successful_analyses == 1
        assert report.failed_analyses == 2
        assert report.success_rate == pytest.approx(33.33, rel=1e-2)
        assert len(report.results) == 1
        assert len(report.failed_symbols) == 2
        assert "INVALID" in report.failed_symbols
        assert "GOOGL" in report.failed_symbols
        assert "DataRetrievalError" in report.error_summary
        assert "CalculationError" in report.error_summary
    
    def test_analyze_multiple_stocks_parallel(self):
        """Test multiple stock analysis in parallel mode."""
        orchestrator = StockAnalysisOrchestrator(
            max_workers=2,
            enable_parallel_processing=True
        )
        
        # Mock the analyze_single_stock method
        mock_result1 = Mock(spec=AnalysisResult)
        mock_result1.symbol = "AAPL"
        mock_result2 = Mock(spec=AnalysisResult)
        mock_result2.symbol = "GOOGL"
        
        orchestrator.analyze_single_stock = Mock(side_effect=[mock_result1, mock_result2])
        
        symbols = ["AAPL", "GOOGL"]
        report = orchestrator.analyze_multiple_stocks(symbols)
        
        # Verify report
        assert report.total_stocks == 2
        assert report.successful_analyses == 2
        assert report.failed_analyses == 0
        assert report.success_rate == 100.0
        assert len(report.results) == 2
    
    def test_analyze_multiple_stocks_stop_on_error(self):
        """Test multiple stock analysis stopping on first error."""
        orchestrator = StockAnalysisOrchestrator(continue_on_error=False)
        
        def mock_analyze(symbol):
            if symbol == "AAPL":
                mock_result = Mock(spec=AnalysisResult)
                mock_result.symbol = "AAPL"
                return mock_result
            else:
                raise DataRetrievalError("Stock not found")
        
        orchestrator.analyze_single_stock = Mock(side_effect=mock_analyze)
        
        symbols = ["AAPL", "INVALID", "GOOGL"]
        report = orchestrator.analyze_multiple_stocks(symbols)
        
        # Should stop after first error
        assert report.successful_analyses == 1
        assert report.failed_analyses == 1
        assert len(report.failed_symbols) == 1
        assert orchestrator.analyze_single_stock.call_count == 2  # AAPL + INVALID
    
    def test_generate_recommendations_buy_scenario(self, sample_stock_info, sample_financial_ratios,
                                                 sample_health_score, sample_sentiment):
        """Test recommendation generation for buy scenario."""
        orchestrator = StockAnalysisOrchestrator()
        
        # Create buy scenario
        fair_value = FairValueResult(
            current_price=150.0,
            dcf_value=180.0,
            average_fair_value=175.0,
            recommendation="BUY",
            confidence_level=0.8
        )
        
        recommendations = orchestrator._generate_recommendations(
            "AAPL", sample_stock_info, sample_financial_ratios,
            sample_health_score, fair_value, sample_sentiment
        )
        
        assert len(recommendations) > 0
        assert any("Buy" in rec for rec in recommendations)
        assert any("confidence" in rec for rec in recommendations)
    
    def test_generate_recommendations_sell_scenario(self, sample_stock_info, sample_financial_ratios,
                                                  sample_health_score, sample_sentiment):
        """Test recommendation generation for sell scenario."""
        orchestrator = StockAnalysisOrchestrator()
        
        # Create sell scenario
        fair_value = FairValueResult(
            current_price=150.0,
            dcf_value=120.0,
            average_fair_value=125.0,
            recommendation="SELL",
            confidence_level=0.9
        )
        
        recommendations = orchestrator._generate_recommendations(
            "AAPL", sample_stock_info, sample_financial_ratios,
            sample_health_score, fair_value, sample_sentiment
        )
        
        assert len(recommendations) > 0
        assert any("Sell" in rec for rec in recommendations)
    
    def test_generate_recommendations_health_based(self, sample_stock_info, sample_financial_ratios,
                                                 sample_fair_value, sample_sentiment):
        """Test health-based recommendations."""
        orchestrator = StockAnalysisOrchestrator()
        
        # Test excellent health
        excellent_health = HealthScore(
            overall_score=85.0,
            financial_strength=90.0,
            profitability_health=80.0,
            liquidity_health=85.0,
            risk_assessment="Low"
        )
        
        recommendations = orchestrator._generate_recommendations(
            "AAPL", sample_stock_info, sample_financial_ratios,
            excellent_health, sample_fair_value, sample_sentiment
        )
        
        assert any("Excellent financial health" in rec for rec in recommendations)
        assert any("Low risk profile" in rec for rec in recommendations)
    
    @patch('stock_analysis.orchestrator.ExportService')
    def test_export_results_csv(self, mock_export_service):
        """Test exporting results to CSV format."""
        mock_service = Mock()
        mock_export_service.return_value = mock_service
        mock_service.export_to_csv.return_value = "/path/to/export.csv"
        
        orchestrator = StockAnalysisOrchestrator()
        results = [Mock(spec=AnalysisResult)]
        
        filepath = orchestrator.export_results(results, "csv", "test.csv")
        
        assert filepath == "/path/to/export.csv"
        mock_service.export_to_csv.assert_called_once_with(results, "test.csv")
    
    @patch('stock_analysis.orchestrator.ExportService')
    def test_export_results_excel(self, mock_export_service):
        """Test exporting results to Excel format."""
        mock_service = Mock()
        mock_export_service.return_value = mock_service
        mock_service.export_to_excel.return_value = "/path/to/export.xlsx"
        
        orchestrator = StockAnalysisOrchestrator()
        results = [Mock(spec=AnalysisResult)]
        
        filepath = orchestrator.export_results(results, "excel")
        
        assert filepath == "/path/to/export.xlsx"
        mock_service.export_to_excel.assert_called_once_with(results, None)
    
    @patch('stock_analysis.orchestrator.ExportService')
    def test_export_results_json(self, mock_export_service):
        """Test exporting results to JSON format."""
        mock_service = Mock()
        mock_export_service.return_value = mock_service
        mock_service.export_to_powerbi_json.return_value = "/path/to/export.json"
        
        orchestrator = StockAnalysisOrchestrator()
        results = [Mock(spec=AnalysisResult)]
        
        filepath = orchestrator.export_results(results, "json")
        
        assert filepath == "/path/to/export.json"
        mock_service.export_to_powerbi_json.assert_called_once_with(results, None)
    
    def test_export_results_unsupported_format(self, orchestrator):
        """Test exporting with unsupported format."""
        results = [Mock(spec=AnalysisResult)]
        
        with pytest.raises(ExportError):
            orchestrator.export_results(results, "unsupported")
    
    def test_get_analysis_summary(self, orchestrator):
        """Test analysis summary generation."""
        # Create mock results
        mock_results = []
        for i, symbol in enumerate(["AAPL", "GOOGL", "MSFT"]):
            result = Mock(spec=AnalysisResult)
            result.symbol = symbol
            result.fair_value = Mock()
            result.fair_value.recommendation = ["BUY", "HOLD", "SELL"][i]
            result.health_score = Mock()
            result.health_score.overall_score = 70.0 + i * 10
            result.sentiment = Mock()
            result.sentiment.overall_sentiment = 0.1 + i * 0.1
            mock_results.append(result)
        
        report = AnalysisReport(
            total_stocks=4,
            successful_analyses=3,
            failed_analyses=1,
            execution_time=45.5,
            success_rate=75.0,
            failed_symbols=["INVALID"],
            error_summary={"DataRetrievalError": 1},
            results=mock_results
        )
        
        summary = orchestrator.get_analysis_summary(report)
        
        assert "Total stocks analyzed: 4" in summary
        assert "Successful analyses: 3" in summary
        assert "Failed analyses: 1" in summary
        assert "Success rate: 75.0%" in summary
        assert "Execution time: 45.50 seconds" in summary
        assert "Failed symbols: INVALID" in summary
        assert "DataRetrievalError: 1" in summary
        assert "Buy recommendations: 1" in summary
        assert "Hold recommendations: 1" in summary
        assert "Sell recommendations: 1" in summary
        assert "Average health score:" in summary
        assert "Average sentiment:" in summary
    
    def test_get_analysis_summary_no_results(self, orchestrator):
        """Test analysis summary with no results."""
        report = AnalysisReport(
            total_stocks=2,
            successful_analyses=0,
            failed_analyses=2,
            execution_time=10.0,
            success_rate=0.0,
            failed_symbols=["AAPL", "GOOGL"],
            error_summary={"DataRetrievalError": 2},
            results=[]
        )
        
        summary = orchestrator.get_analysis_summary(report)
        
        assert "Total stocks analyzed: 2" in summary
        assert "Successful analyses: 0" in summary
        assert "Failed analyses: 2" in summary
        assert "Success rate: 0.0%" in summary
        assert "Failed symbols: AAPL, GOOGL" in summary
        assert "DataRetrievalError: 2" in summary
        # Should not contain analysis highlights section
        assert "Analysis highlights:" not in summary


if __name__ == "__main__":
    pytest.main([__file__])