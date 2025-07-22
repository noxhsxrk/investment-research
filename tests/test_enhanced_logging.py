"""Unit tests for enhanced logging functionality."""

import pytest
import json
import logging
import os
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from stock_analysis.utils.logging import (
    StructuredFormatter, ContextualLogger, OperationLogger,
    setup_logging, get_logger, get_operation_logger,
    log_api_call, log_data_quality_issue
)


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create a temporary directory for log files."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return str(log_dir)


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config_mock = MagicMock()
    config_mock.get.side_effect = lambda key, default=None: {
        'stock_analysis.logging.level': 'INFO',
        'stock_analysis.logging.file_path': None,
        'stock_analysis.logging.max_file_size': '10MB',
        'stock_analysis.logging.backup_count': 5,
        'stock_analysis.logging.structured': False
    }.get(key, default)
    return config_mock


class TestStructuredFormatter:
    """Test cases for StructuredFormatter."""
    
    def test_basic_formatting(self):
        """Test basic log record formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        log_entry = json.loads(formatted)
        
        assert log_entry['level'] == 'INFO'
        assert log_entry['logger'] == 'test_logger'
        assert log_entry['message'] == 'Test message'
        assert log_entry['line'] == 1
        assert 'timestamp' in log_entry
        assert 'thread' in log_entry
        assert 'process' in log_entry
    
    def test_exception_formatting(self):
        """Test formatting log record with exception."""
        formatter = StructuredFormatter()
        try:
            raise ValueError("Test error")
        except ValueError:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=True
            )
        
        formatted = formatter.format(record)
        log_entry = json.loads(formatted)
        
        assert log_entry['level'] == 'ERROR'
        assert log_entry['message'] == 'Error occurred'
        assert 'exception' in log_entry
        assert log_entry['exception']['type'] == 'ValueError'
        assert log_entry['exception']['message'] == 'Test error'
        assert isinstance(log_entry['exception']['traceback'], list)
    
    def test_extra_fields(self):
        """Test formatting log record with extra fields."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.extra_fields = {'user_id': 123, 'action': 'test'}
        
        formatted = formatter.format(record)
        log_entry = json.loads(formatted)
        
        assert log_entry['user_id'] == 123
        assert log_entry['action'] == 'test'


class TestContextualLogger:
    """Test cases for ContextualLogger."""
    
    @pytest.fixture
    def contextual_logger(self):
        """Create a contextual logger for testing."""
        base_logger = logging.getLogger('test')
        return ContextualLogger(base_logger)
    
    def test_set_context(self, contextual_logger):
        """Test setting context information."""
        contextual_logger.set_context(user_id=123, action='test')
        assert contextual_logger.context == {'user_id': 123, 'action': 'test'}
    
    def test_clear_context(self, contextual_logger):
        """Test clearing context information."""
        contextual_logger.set_context(user_id=123)
        contextual_logger.clear_context()
        assert contextual_logger.context == {}
    
    def test_logging_with_context(self, contextual_logger):
        """Test logging message with context."""
        with patch.object(contextual_logger.logger, 'log') as mock_log:
            contextual_logger.set_context(user_id=123)
            contextual_logger.info("Test message")
            
            mock_log.assert_called_once()
            args = mock_log.call_args
            assert args[0][0] == logging.INFO
            assert args[0][1] == "Test message"
            assert args[1]['extra']['extra_fields']['user_id'] == 123
    
    def test_exception_logging(self, contextual_logger):
        """Test logging exception with context."""
        with patch.object(contextual_logger.logger, 'log') as mock_log:
            contextual_logger.set_context(user_id=123)
            try:
                raise ValueError("Test error")
            except ValueError:
                contextual_logger.exception("Error occurred")
            
            mock_log.assert_called_once()
            args = mock_log.call_args
            assert args[0][0] == logging.ERROR
            assert args[0][1] == "Error occurred"
            assert args[1]['exc_info'] is True
            assert args[1]['extra']['extra_fields']['user_id'] == 123


class TestOperationLogger:
    """Test cases for OperationLogger."""
    
    @pytest.fixture
    def operation_logger(self):
        """Create an operation logger for testing."""
        contextual_logger = get_logger('test')
        return OperationLogger(contextual_logger, "test_operation")
    
    def test_operation_lifecycle(self, operation_logger):
        """Test complete operation lifecycle logging."""
        with patch.object(operation_logger.logger.logger, 'log') as mock_log:
            # Start operation
            operation_logger.start(user_id=123)
            
            start_call = mock_log.call_args_list[0]
            assert start_call[0][0] == logging.INFO
            assert "Starting operation: test_operation" in start_call[0][1]
            assert start_call[1]['extra']['extra_fields']['operation'] == "test_operation"
            assert start_call[1]['extra']['extra_fields']['user_id'] == 123
            
            # Log progress
            operation_logger.progress("50% complete", progress=0.5)
            
            progress_call = mock_log.call_args_list[1]
            assert progress_call[0][0] == logging.INFO
            assert "50% complete" in progress_call[0][1]
            assert progress_call[1]['extra']['extra_fields']['progress'] == 0.5
            
            # Finish operation
            operation_logger.finish(success=True, items_processed=100)
            
            finish_call = mock_log.call_args_list[2]
            assert finish_call[0][0] == logging.INFO
            assert "Completed operation: test_operation" in finish_call[0][1]
            assert finish_call[1]['extra']['extra_fields']['operation_success'] is True
            assert finish_call[1]['extra']['extra_fields']['items_processed'] == 100
            assert 'operation_duration_seconds' in finish_call[1]['extra']['extra_fields']
    
    def test_failed_operation(self, operation_logger):
        """Test logging failed operation."""
        with patch.object(operation_logger.logger.logger, 'log') as mock_log:
            operation_logger.start()
            operation_logger.finish(success=False, error="Test error")
            
            finish_call = mock_log.call_args_list[1]
            assert finish_call[0][0] == logging.ERROR
            assert "Failed operation: test_operation" in finish_call[0][1]
            assert finish_call[1]['extra']['extra_fields']['operation_success'] is False
            assert finish_call[1]['extra']['extra_fields']['error'] == "Test error"


def test_setup_logging(temp_log_dir, mock_config):
    """Test logging setup configuration."""
    with patch('stock_analysis.utils.logging.config', mock_config):
        log_file = os.path.join(temp_log_dir, "test.log")
        logger = setup_logging(
            log_level="DEBUG",
            log_file=log_file,
            max_file_size="1MB",
            backup_count=3,
            structured_logging=True
        )
        
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 3  # Console, File, and Error handlers
        
        # Verify log files were created
        assert os.path.exists(log_file)
        assert os.path.exists(log_file.replace('.log', '_errors.log'))


def test_log_api_call():
    """Test API call logging."""
    logger = get_logger('test')
    with patch.object(logger.logger, 'log') as mock_log:
        log_api_call(
            logger,
            api_name="TestAPI",
            endpoint="/test",
            parameters={"param": "value"},
            response_time=0.5,
            status_code=200
        )
        
        args = mock_log.call_args
        assert args[0][0] == logging.INFO
        assert "API call successful" in args[0][1]
        assert args[1]['extra']['extra_fields']['api_name'] == "TestAPI"
        assert args[1]['extra']['extra_fields']['endpoint'] == "/test"
        assert args[1]['extra']['extra_fields']['response_time_seconds'] == 0.5
        assert args[1]['extra']['extra_fields']['status_code'] == 200


def test_log_data_quality_issue():
    """Test data quality issue logging."""
    logger = get_logger('test')
    with patch.object(logger.logger, 'log') as mock_log:
        log_data_quality_issue(
            logger,
            symbol="AAPL",
            data_type="stock_info",
            issue_description="Missing data",
            severity="warning",
            missing_fields=["price"],
            invalid_values={"volume": -1}
        )
        
        args = mock_log.call_args
        assert args[0][0] == logging.WARNING
        assert "Data quality warning" in args[0][1]
        assert args[1]['extra']['extra_fields']['symbol'] == "AAPL"
        assert args[1]['extra']['extra_fields']['data_type'] == "stock_info"
        assert args[1]['extra']['extra_fields']['missing_fields'] == ["price"]
        assert args[1]['extra']['extra_fields']['invalid_values'] == {"volume": -1}
