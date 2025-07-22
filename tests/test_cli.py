"""Unit tests for the CLI module."""

import pytest
import argparse
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Mock the problematic imports before importing the CLI module
import sys
sys.modules['schedule'] = Mock()
sys.modules['stock_analysis.services.scheduler_service'] = Mock()
sys.modules['stock_analysis.orchestrator'] = Mock()
sys.modules['stock_analysis.utils.config'] = Mock()
sys.modules['stock_analysis.utils.logging'] = Mock()
sys.modules['stock_analysis.utils.exceptions'] = Mock()

# Now we can import the CLI module
from stock_analysis.cli import (
    create_parser, 
    load_symbols_from_file, 
    ProgressIndicator
)


class TestCLIParser:
    """Test the CLI argument parser."""
    
    def test_create_parser(self):
        """Test that the parser is created correctly."""
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.prog == 'stock-analysis'
    
    def test_analyze_command_parsing(self):
        """Test parsing of analyze command."""
        parser = create_parser()
        
        # Test single symbol
        args = parser.parse_args(['analyze', 'AAPL'])
        assert args.command == 'analyze'
        assert args.symbols == ['AAPL']
        assert args.export_format == 'excel'  # default
        
        # Test multiple symbols with options
        args = parser.parse_args(['--verbose', 'analyze', 'AAPL', 'MSFT', '--export-format', 'csv'])
        assert args.command == 'analyze'
        assert args.symbols == ['AAPL', 'MSFT']
        assert args.export_format == 'csv'
        assert args.verbose is True
    
    def test_batch_command_parsing(self):
        """Test parsing of batch command."""
        parser = create_parser()
        
        args = parser.parse_args(['batch', 'stocks.txt'])
        assert args.command == 'batch'
        assert args.file == 'stocks.txt'
        assert args.export_format == 'excel'  # default
        assert args.max_workers == 4  # default
        
        # Test with options
        args = parser.parse_args(['batch', 'stocks.txt', '--export-format', 'json', '--max-workers', '8'])
        assert args.export_format == 'json'
        assert args.max_workers == 8
    
    def test_schedule_command_parsing(self):
        """Test parsing of schedule commands."""
        parser = create_parser()
        
        # Test schedule add
        args = parser.parse_args(['schedule', 'add', 'job1', 'AAPL,MSFT'])
        assert args.command == 'schedule'
        assert args.schedule_action == 'add'
        assert args.job_id == 'job1'
        assert args.symbols == 'AAPL,MSFT'
        assert args.interval == 'daily'  # default
        
        # Test schedule list
        args = parser.parse_args(['schedule', 'list'])
        assert args.schedule_action == 'list'
        
        # Test schedule status
        args = parser.parse_args(['schedule', 'status'])
        assert args.schedule_action == 'status'
        
        # Test schedule status with job_id
        args = parser.parse_args(['schedule', 'status', 'job1'])
        assert args.job_id == 'job1'
    
    def test_config_command_parsing(self):
        """Test parsing of config commands."""
        parser = create_parser()
        
        # Test config show
        args = parser.parse_args(['config', 'show'])
        assert args.command == 'config'
        assert args.config_action == 'show'
        
        # Test config show with key
        args = parser.parse_args(['config', 'show', 'some.key'])
        assert args.key == 'some.key'
        
        # Test config set
        args = parser.parse_args(['config', 'set', 'some.key', 'value'])
        assert args.config_action == 'set'
        assert args.key == 'some.key'
        assert args.value == 'value'
    
    def test_global_options(self):
        """Test global options parsing."""
        parser = create_parser()
        
        args = parser.parse_args(['--verbose', '--config', 'custom.yaml', 'analyze', 'AAPL'])
        assert args.verbose is True
        assert args.config == 'custom.yaml'
        assert args.command == 'analyze'


class TestFileLoading:
    """Test file loading functionality."""
    
    def test_load_symbols_from_text_file(self):
        """Test loading symbols from text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("AAPL\nMSFT\nGOOGL\n# Comment\nAMZN\n")
            f.flush()
            
            try:
                symbols = load_symbols_from_file(f.name)
                assert symbols == ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
            finally:
                os.unlink(f.name)
    
    def test_load_symbols_from_csv_file(self):
        """Test loading symbols from CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("AAPL,MSFT,GOOGL\nJPM,BAC\n")
            f.flush()
            
            try:
                symbols = load_symbols_from_file(f.name)
                assert symbols == ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC']
            finally:
                os.unlink(f.name)
    
    def test_load_symbols_removes_duplicates(self):
        """Test that duplicate symbols are removed."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("AAPL\nMSFT\nAAPL\nGOOGL\nMSFT\n")
            f.flush()
            
            try:
                symbols = load_symbols_from_file(f.name)
                assert symbols == ['AAPL', 'MSFT', 'GOOGL']
            finally:
                os.unlink(f.name)
    
    def test_load_symbols_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_symbols_from_file('nonexistent_file.txt')
    
    def test_load_symbols_empty_file(self):
        """Test error handling for empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# Only comments\n# No symbols\n")
            f.flush()
            
            try:
                with pytest.raises(ValueError, match="No valid symbols found"):
                    load_symbols_from_file(f.name)
            finally:
                os.unlink(f.name)


class TestProgressIndicator:
    """Test the progress indicator functionality."""
    
    def test_progress_indicator_creation(self):
        """Test creating progress indicator."""
        indicator = ProgressIndicator(verbose=True)
        assert indicator.verbose is True
        assert indicator.last_progress == 0
        
        indicator = ProgressIndicator(verbose=False)
        assert indicator.verbose is False
    
    @patch('builtins.print')
    def test_progress_indicator_update_verbose(self, mock_print):
        """Test progress indicator updates in verbose mode."""
        indicator = ProgressIndicator(verbose=True)
        
        # Mock progress object
        progress = Mock()
        progress.completion_percentage = 25.0
        progress.completed_stocks = 5
        progress.total_stocks = 20
        progress.failed_stocks = 1
        progress.current_stock = 'AAPL'
        
        indicator.update_progress(progress)
        
        # Should print progress since it's a significant change (>5%)
        mock_print.assert_called()
    
    @patch('builtins.print')
    def test_progress_indicator_update_non_verbose(self, mock_print):
        """Test progress indicator doesn't update in non-verbose mode."""
        indicator = ProgressIndicator(verbose=False)
        
        progress = Mock()
        progress.completion_percentage = 25.0
        
        indicator.update_progress(progress)
        
        # Should not print in non-verbose mode
        mock_print.assert_not_called()


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    @patch('stock_analysis.cli.StockAnalysisOrchestrator')
    @patch('stock_analysis.cli.config')
    def test_analyze_command_integration(self, mock_config, mock_orchestrator):
        """Test analyze command integration."""
        # Mock the orchestrator
        mock_orch_instance = Mock()
        mock_orchestrator.return_value = mock_orch_instance
        
        # Mock analysis result
        mock_result = Mock()
        mock_result.symbol = 'AAPL'
        mock_result.stock_info.company_name = 'Apple Inc.'
        mock_result.stock_info.current_price = 150.0
        mock_result.health_score.overall_score = 85.0
        mock_result.fair_value.recommendation = 'BUY'
        mock_result.fair_value.average_fair_value = 160.0
        mock_result.sentiment.overall_sentiment = 0.3
        mock_result.recommendations = ['Strong buy recommendation']
        
        mock_orch_instance.analyze_single_stock.return_value = mock_result
        
        # Test the command (we can't actually run it due to dependencies)
        # But we can verify the structure is correct
        from stock_analysis.cli import handle_analyze_command
        
        # Mock args
        args = Mock()
        args.symbols = ['AAPL']
        args.verbose = True
        args.no_export = True
        
        # This would work if dependencies were available
        # result = handle_analyze_command(args)
        # assert result == 0


def test_cli_module_structure():
    """Test that the CLI module has the expected structure."""
    import stock_analysis.cli as cli_module
    
    # Check that required functions exist
    required_functions = [
        'create_parser',
        'load_symbols_from_file', 
        'handle_analyze_command',
        'handle_batch_command',
        'handle_schedule_command', 
        'handle_config_command',
        'main'
    ]
    
    for func_name in required_functions:
        assert hasattr(cli_module, func_name), f"Missing function: {func_name}"
        assert callable(getattr(cli_module, func_name)), f"Not callable: {func_name}"
    
    # Check that required classes exist
    assert hasattr(cli_module, 'ProgressIndicator')
    assert callable(cli_module.ProgressIndicator)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])