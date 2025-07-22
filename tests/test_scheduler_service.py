"""Tests for the scheduler service module."""

import pytest
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from stock_analysis.services.scheduler_service import (
    SchedulerService, ScheduledJob, NotificationConfig, SchedulerReport
)
from stock_analysis.orchestrator import AnalysisReport
from stock_analysis.utils.exceptions import SchedulingError


class TestScheduledJob:
    """Test cases for ScheduledJob dataclass."""
    
    def test_scheduled_job_creation(self):
        """Test creating a scheduled job."""
        job = ScheduledJob(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL", "GOOGL"],
            interval="daily"
        )
        
        assert job.job_id == "test_job"
        assert job.name == "Test Job"
        assert job.symbols == ["AAPL", "GOOGL"]
        assert job.interval == "daily"
        assert job.enabled is True
        assert job.success_count == 0
        assert job.failure_count == 0


class TestNotificationConfig:
    """Test cases for NotificationConfig dataclass."""
    
    def test_notification_config_defaults(self):
        """Test notification config with default values."""
        config = NotificationConfig()
        
        assert config.enabled is True
        assert config.email_port == 587
        assert config.email_use_tls is True
        assert config.send_on_success is True
        assert config.send_on_failure is True


class TestSchedulerService:
    """Test cases for SchedulerService class."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        orchestrator = Mock()
        orchestrator.analyze_multiple_stocks.return_value = AnalysisReport(
            total_stocks=2,
            successful_analyses=2,
            failed_analyses=0,
            execution_time=10.5,
            success_rate=100.0,
            failed_symbols=[],
            error_summary={},
            results=[]
        )
        orchestrator.export_results.return_value = "/path/to/export.xlsx"
        return orchestrator
    
    @pytest.fixture
    def scheduler_service(self, mock_orchestrator):
        """Create a scheduler service instance."""
        return SchedulerService(orchestrator=mock_orchestrator)
    
    def test_scheduler_service_initialization(self, scheduler_service):
        """Test scheduler service initialization."""
        assert scheduler_service.orchestrator is not None
        assert scheduler_service.jobs == {}
        assert scheduler_service.is_running is False
        assert scheduler_service.max_retry_attempts == 3
        assert scheduler_service.retry_delay_minutes == 30
    
    def test_add_job(self, scheduler_service):
        """Test adding a scheduled job."""
        scheduler_service.add_job(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL", "GOOGL"],
            interval="daily"
        )
        
        assert "test_job" in scheduler_service.jobs
        job = scheduler_service.jobs["test_job"]
        assert job.name == "Test Job"
        assert job.symbols == ["AAPL", "GOOGL"]
        assert job.interval == "daily"
        assert job.next_run is not None
    
    def test_add_duplicate_job(self, scheduler_service):
        """Test adding a job with duplicate ID."""
        scheduler_service.add_job(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        with pytest.raises(SchedulingError, match="Job with ID 'test_job' already exists"):
            scheduler_service.add_job(
                job_id="test_job",
                name="Another Job",
                symbols=["GOOGL"],
                interval="weekly"
            )
    
    def test_remove_job(self, scheduler_service):
        """Test removing a scheduled job."""
        scheduler_service.add_job(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        assert "test_job" in scheduler_service.jobs
        
        with patch('schedule.cancel_job') as mock_cancel:
            scheduler_service.remove_job("test_job")
            mock_cancel.assert_called_once_with("test_job")
        
        assert "test_job" not in scheduler_service.jobs
    
    def test_remove_nonexistent_job(self, scheduler_service):
        """Test removing a job that doesn't exist."""
        with pytest.raises(SchedulingError, match="Job with ID 'nonexistent' not found"):
            scheduler_service.remove_job("nonexistent")
    
    def test_enable_job(self, scheduler_service):
        """Test enabling a job."""
        scheduler_service.add_job(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        job = scheduler_service.jobs["test_job"]
        job.enabled = False
        
        scheduler_service.enable_job("test_job")
        assert job.enabled is True
    
    def test_disable_job(self, scheduler_service):
        """Test disabling a job."""
        scheduler_service.add_job(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        with patch('schedule.cancel_job') as mock_cancel:
            scheduler_service.disable_job("test_job")
            mock_cancel.assert_called_once_with("test_job")
        
        job = scheduler_service.jobs["test_job"]
        assert job.enabled is False
    
    @patch('threading.Thread')
    @patch('schedule.clear')
    def test_start_scheduler(self, mock_clear, mock_thread, scheduler_service):
        """Test starting the scheduler."""
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        
        scheduler_service.start_scheduler()
        
        assert scheduler_service.is_running is True
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()
    
    def test_start_scheduler_already_running(self, scheduler_service):
        """Test starting scheduler when already running."""
        scheduler_service.is_running = True
        
        with patch('threading.Thread') as mock_thread:
            scheduler_service.start_scheduler()
            mock_thread.assert_not_called()
    
    @patch('schedule.clear')
    def test_stop_scheduler(self, mock_clear, scheduler_service):
        """Test stopping the scheduler."""
        scheduler_service.is_running = True
        mock_thread = Mock()
        mock_thread.is_alive.return_value = False
        scheduler_service.scheduler_thread = mock_thread
        
        scheduler_service.stop_scheduler()
        
        assert scheduler_service.is_running is False
        mock_clear.assert_called_once()
    
    def test_run_job_now(self, scheduler_service, mock_orchestrator):
        """Test running a job immediately."""
        scheduler_service.add_job(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL", "GOOGL"],
            interval="daily"
        )
        
        with patch.object(scheduler_service, '_execute_job') as mock_execute:
            mock_report = Mock()
            mock_execute.return_value = mock_report
            
            result = scheduler_service.run_job_now("test_job")
            
            assert result == mock_report
            mock_execute.assert_called_once()
    
    def test_run_nonexistent_job(self, scheduler_service):
        """Test running a job that doesn't exist."""
        with pytest.raises(SchedulingError, match="Job with ID 'nonexistent' not found"):
            scheduler_service.run_job_now("nonexistent")
    
    def test_get_scheduler_status(self, scheduler_service):
        """Test getting scheduler status."""
        # Add some jobs
        scheduler_service.add_job("job1", "Job 1", ["AAPL"], "daily")
        scheduler_service.add_job("job2", "Job 2", ["GOOGL"], "weekly")
        scheduler_service.disable_job("job2")
        
        # Add some execution history
        scheduler_service.execution_history = [
            {
                'job_id': 'job1',
                'timestamp': datetime.now().isoformat(),
                'success': True
            },
            {
                'job_id': 'job1',
                'timestamp': (datetime.now() - timedelta(days=1)).isoformat(),
                'success': False
            }
        ]
        
        status = scheduler_service.get_scheduler_status()
        
        assert isinstance(status, SchedulerReport)
        assert status.total_jobs == 2
        assert status.active_jobs == 1  # Only job1 is enabled
        assert status.system_status == "Stopped"
    
    def test_get_job_status(self, scheduler_service):
        """Test getting status for a specific job."""
        scheduler_service.add_job(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        status = scheduler_service.get_job_status("test_job")
        
        assert status['job_id'] == "test_job"
        assert status['name'] == "Test Job"
        assert status['symbols'] == ["AAPL"]
        assert status['interval'] == "daily"
        assert status['enabled'] is True
    
    def test_get_nonexistent_job_status(self, scheduler_service):
        """Test getting status for a job that doesn't exist."""
        with pytest.raises(SchedulingError, match="Job with ID 'nonexistent' not found"):
            scheduler_service.get_job_status("nonexistent")
    
    def test_generate_summary_report(self, scheduler_service):
        """Test generating a summary report."""
        # Add some jobs
        scheduler_service.add_job("job1", "Job 1", ["AAPL"], "daily")
        scheduler_service.add_job("job2", "Job 2", ["GOOGL"], "weekly")
        
        # Add execution history
        now = datetime.now()
        scheduler_service.execution_history = [
            {
                'job_id': 'job1',
                'timestamp': now.isoformat(),
                'success': True
            },
            {
                'job_id': 'job1',
                'timestamp': (now - timedelta(hours=1)).isoformat(),
                'success': False,
                'error': 'Test error'
            },
            {
                'job_id': 'job2',
                'timestamp': (now - timedelta(days=2)).isoformat(),
                'success': True
            }
        ]
        
        report = scheduler_service.generate_summary_report(period_days=7)
        
        assert "Stock Analysis Scheduler Report" in report
        assert "Total executions: 3" in report
        assert "Successful executions: 2" in report
        assert "Failed executions: 1" in report
        assert "Success rate: 66.7%" in report
    
    def test_calculate_next_run_daily(self, scheduler_service):
        """Test calculating next run for daily interval."""
        next_run = scheduler_service._calculate_next_run("daily")
        
        assert isinstance(next_run, datetime)
        assert next_run.hour == 9
        assert next_run.minute == 0
        assert next_run > datetime.now()
    
    def test_calculate_next_run_weekly(self, scheduler_service):
        """Test calculating next run for weekly interval."""
        next_run = scheduler_service._calculate_next_run("weekly")
        
        assert isinstance(next_run, datetime)
        assert next_run.hour == 9
        assert next_run.minute == 0
        assert next_run.weekday() == 0  # Monday
    
    def test_calculate_next_run_hourly(self, scheduler_service):
        """Test calculating next run for hourly interval."""
        next_run = scheduler_service._calculate_next_run("hourly")
        
        assert isinstance(next_run, datetime)
        assert next_run.minute == 0
        assert next_run > datetime.now()
    
    def test_calculate_next_run_custom_hours(self, scheduler_service):
        """Test calculating next run for custom hour interval."""
        next_run = scheduler_service._calculate_next_run("every_2_hours")
        
        assert isinstance(next_run, datetime)
        assert next_run > datetime.now()
    
    def test_calculate_next_run_custom_minutes(self, scheduler_service):
        """Test calculating next run for custom minute interval."""
        next_run = scheduler_service._calculate_next_run("every_30_minutes")
        
        assert isinstance(next_run, datetime)
        assert next_run > datetime.now()
    
    def test_calculate_next_run_invalid(self, scheduler_service):
        """Test calculating next run for invalid interval."""
        next_run = scheduler_service._calculate_next_run("invalid_interval")
        
        assert isinstance(next_run, datetime)
        assert next_run > datetime.now()
    
    def test_record_execution_success(self, scheduler_service):
        """Test recording successful execution."""
        mock_report = Mock()
        mock_report.total_stocks = 2
        mock_report.successful_analyses = 2
        mock_report.failed_analyses = 0
        mock_report.execution_time = 10.5
        mock_report.success_rate = 100.0
        
        scheduler_service._record_execution("test_job", True, mock_report, None)
        
        assert len(scheduler_service.execution_history) == 1
        record = scheduler_service.execution_history[0]
        assert record['job_id'] == "test_job"
        assert record['success'] is True
        assert record['total_stocks'] == 2
        assert record['execution_time'] == 10.5
    
    def test_record_execution_failure(self, scheduler_service):
        """Test recording failed execution."""
        scheduler_service._record_execution("test_job", False, None, "Test error")
        
        assert len(scheduler_service.execution_history) == 1
        record = scheduler_service.execution_history[0]
        assert record['job_id'] == "test_job"
        assert record['success'] is False
        assert record['error'] == "Test error"
    
    def test_execute_job_success(self, scheduler_service, mock_orchestrator):
        """Test successful job execution."""
        job = ScheduledJob(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL", "GOOGL"],
            interval="daily"
        )
        
        with patch.object(scheduler_service, '_record_execution') as mock_record:
            with patch.object(scheduler_service, '_send_notification') as mock_notify:
                result = scheduler_service._execute_job(job)
                
                assert isinstance(result, AnalysisReport)
                assert job.success_count == 1
                assert job.last_error is None
                mock_orchestrator.analyze_multiple_stocks.assert_called_once_with(["AAPL", "GOOGL"])
                mock_record.assert_called_once()
                mock_notify.assert_called_once()
    
    def test_execute_job_failure_with_retry(self, scheduler_service, mock_orchestrator):
        """Test job execution failure with retry."""
        job = ScheduledJob(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        # Mock orchestrator to fail first time, succeed second time
        mock_orchestrator.analyze_multiple_stocks.side_effect = [
            Exception("First failure"),
            AnalysisReport(
                total_stocks=1,
                successful_analyses=1,
                failed_analyses=0,
                execution_time=5.0,
                success_rate=100.0,
                failed_symbols=[],
                error_summary={},
                results=[]
            )
        ]
        
        scheduler_service.max_retry_attempts = 1
        
        with patch('time.sleep'):  # Speed up the test
            with patch.object(scheduler_service, '_record_execution') as mock_record:
                result = scheduler_service._execute_job(job)
                
                assert isinstance(result, AnalysisReport)
                assert job.success_count == 1
                assert mock_orchestrator.analyze_multiple_stocks.call_count == 2
    
    def test_execute_job_failure_exhausted_retries(self, scheduler_service, mock_orchestrator):
        """Test job execution failure with exhausted retries."""
        job = ScheduledJob(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        mock_orchestrator.analyze_multiple_stocks.side_effect = Exception("Persistent failure")
        scheduler_service.max_retry_attempts = 1
        
        with patch('time.sleep'):  # Speed up the test
            with patch.object(scheduler_service, '_record_execution') as mock_record:
                with pytest.raises(SchedulingError, match="failed after 1 retries"):
                    scheduler_service._execute_job(job)
                
                assert job.failure_count == 1
                assert job.last_error == "Persistent failure"
    
    @patch('smtplib.SMTP')
    def test_send_notification_success(self, mock_smtp, scheduler_service):
        """Test sending success notification."""
        # Configure notification settings
        scheduler_service.notification_config.enabled = True
        scheduler_service.notification_config.email_host = "smtp.example.com"
        scheduler_service.notification_config.email_username = "test@example.com"
        scheduler_service.notification_config.email_password = "password"
        scheduler_service.notification_config.recipients = ["recipient@example.com"]
        
        job = ScheduledJob(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        job.last_run = datetime.now()
        job.next_run = datetime.now() + timedelta(days=1)
        
        mock_report = Mock()
        mock_report.total_stocks = 1
        mock_report.successful_analyses = 1
        mock_report.failed_analyses = 0
        mock_report.success_rate = 100.0
        mock_report.execution_time = 5.0
        mock_report.failed_symbols = []
        mock_report.results = []
        
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        scheduler_service._send_notification(job, mock_report, True)
        
        mock_smtp.assert_called_once_with("smtp.example.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("test@example.com", "password")
        mock_server.send_message.assert_called_once()
        mock_server.quit.assert_called_once()
    
    @patch('smtplib.SMTP')
    def test_send_notification_failure(self, mock_smtp, scheduler_service):
        """Test sending failure notification."""
        # Configure notification settings
        scheduler_service.notification_config.enabled = True
        scheduler_service.notification_config.email_host = "smtp.example.com"
        scheduler_service.notification_config.recipients = ["recipient@example.com"]
        
        job = ScheduledJob(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        job.last_run = datetime.now()
        job.next_run = datetime.now() + timedelta(days=1)
        
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        scheduler_service._send_notification(job, None, False, "Test error")
        
        mock_server.send_message.assert_called_once()
    
    def test_send_notification_disabled(self, scheduler_service):
        """Test that notification is not sent when disabled."""
        scheduler_service.notification_config.enabled = False
        
        job = ScheduledJob(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        with patch('smtplib.SMTP') as mock_smtp:
            scheduler_service._send_notification(job, None, True)
            mock_smtp.assert_not_called()
    
    def test_create_success_notification_body(self, scheduler_service):
        """Test creating success notification email body."""
        job = ScheduledJob(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        job.last_run = datetime(2023, 1, 1, 9, 0, 0)
        job.next_run = datetime(2023, 1, 2, 9, 0, 0)
        
        mock_report = Mock()
        mock_report.total_stocks = 1
        mock_report.successful_analyses = 1
        mock_report.failed_analyses = 0
        mock_report.success_rate = 100.0
        mock_report.execution_time = 5.0
        mock_report.failed_symbols = []
        mock_report.results = []
        
        body = scheduler_service._create_success_notification_body(job, mock_report)
        
        assert "Stock Analysis Job Completed Successfully" in body
        assert "Test Job" in body
        assert "test_job" in body
        assert "Total Stocks: 1" in body
        assert "Success Rate: 100.0%" in body
    
    def test_create_failure_notification_body(self, scheduler_service):
        """Test creating failure notification email body."""
        job = ScheduledJob(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        job.last_run = datetime(2023, 1, 1, 9, 0, 0)
        job.next_run = datetime(2023, 1, 2, 9, 0, 0)
        job.success_count = 5
        job.failure_count = 2
        
        body = scheduler_service._create_failure_notification_body(job, "Test error message")
        
        assert "Stock Analysis Job Failed" in body
        assert "Test Job" in body
        assert "test_job" in body
        assert "Test error message" in body
        assert "Total Successes: 5" in body
        assert "Total Failures: 2" in body
    
    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_save_jobs_config(self, mock_json_dump, mock_open, scheduler_service):
        """Test saving jobs configuration."""
        scheduler_service.add_job("job1", "Job 1", ["AAPL"], "daily")
        
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        scheduler_service.save_jobs_config("test_config.json")
        
        mock_open.assert_called_once_with("test_config.json", 'w')
        mock_json_dump.assert_called_once()
    
    @patch('os.path.exists')
    @patch('builtins.open', create=True)
    @patch('json.load')
    def test_load_jobs_config(self, mock_json_load, mock_open, mock_exists, scheduler_service):
        """Test loading jobs configuration."""
        mock_exists.return_value = True
        
        mock_jobs_data = {
            "job1": {
                "name": "Job 1",
                "symbols": ["AAPL"],
                "interval": "daily",
                "enabled": True,
                "export_format": "excel",
                "notification_enabled": True,
                "success_count": 5,
                "failure_count": 1,
                "last_run": "2023-01-01T09:00:00",
                "next_run": "2023-01-02T09:00:00",
                "last_error": None
            }
        }
        
        mock_json_load.return_value = mock_jobs_data
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        scheduler_service.load_jobs_config("test_config.json")
        
        assert "job1" in scheduler_service.jobs
        job = scheduler_service.jobs["job1"]
        assert job.name == "Job 1"
        assert job.symbols == ["AAPL"]
        assert job.success_count == 5
        assert job.failure_count == 1
    
    @patch('os.path.exists')
    def test_load_jobs_config_file_not_found(self, mock_exists, scheduler_service):
        """Test loading jobs configuration when file doesn't exist."""
        mock_exists.return_value = False
        
        scheduler_service.load_jobs_config("nonexistent.json")
        
        assert len(scheduler_service.jobs) == 0