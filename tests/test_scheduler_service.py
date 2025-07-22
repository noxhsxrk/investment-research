"""Unit tests for scheduler service."""

import pytest
import json
import os
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from stock_analysis.services.scheduler_service import (
    SchedulerService, ScheduledJob, NotificationConfig, SchedulerReport
)
from stock_analysis.utils.exceptions import SchedulingError


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator."""
    orchestrator = MagicMock()
    orchestrator.analyze_multiple_stocks.return_value = {
        'total_stocks': 2,
        'successful_analyses': 2,
        'failed_analyses': 0,
        'execution_time': 10.5,
        'success_rate': 100.0,
        'failed_symbols': [],
        'error_summary': {},
        'results': []
    }
    orchestrator.export_results.return_value = "/path/to/export.xlsx"
    return orchestrator


@pytest.fixture
def scheduler_service(mock_orchestrator):
    """Create a scheduler service instance."""
    return SchedulerService(orchestrator=mock_orchestrator)


class TestScheduledJob:
    """Test cases for ScheduledJob class."""
    
    def test_scheduled_job_creation(self):
        """Test creating a scheduled job."""
        job = ScheduledJob(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        assert job.job_id == "test_job"
        assert job.name == "Test Job"
        assert job.symbols == ["AAPL"]
        assert job.interval == "daily"
        assert job.enabled is True
        assert job.success_count == 0
        assert job.failure_count == 0
        assert job.last_error is None


class TestNotificationConfig:
    """Test cases for NotificationConfig class."""
    
    def test_notification_config_defaults(self):
        """Test notification config default values."""
        config = NotificationConfig()
        
        assert config.enabled is True
        assert config.email_port == 587
        assert config.email_use_tls is True
        assert config.email_host is None
        assert config.email_username is None
        assert config.email_password is None
        assert config.recipients is None


class TestSchedulerService:
    """Test cases for SchedulerService."""
    
    def test_scheduler_service_initialization(self, scheduler_service):
        """Test scheduler service initialization."""
        assert scheduler_service.jobs == {}
        assert scheduler_service.is_running is False
        assert scheduler_service.scheduler_thread is None
    
    def test_add_job(self, scheduler_service):
        """Test adding a job."""
        scheduler_service.add_job(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        assert "test_job" in scheduler_service.jobs
        job = scheduler_service.jobs["test_job"]
        assert job.name == "Test Job"
        assert job.symbols == ["AAPL"]
        assert job.interval == "daily"
    
    def test_add_duplicate_job(self, scheduler_service):
        """Test adding a duplicate job."""
        scheduler_service.add_job(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        with pytest.raises(SchedulingError, match="Job with ID 'test_job' already exists"):
            scheduler_service.add_job(
                job_id="test_job",
                name="Test Job 2",
                symbols=["MSFT"],
                interval="daily"
            )
    
    def test_remove_job(self, scheduler_service):
        """Test removing a job."""
        scheduler_service.add_job(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        scheduler_service.remove_job("test_job")
        assert "test_job" not in scheduler_service.jobs
    
    def test_remove_nonexistent_job(self, scheduler_service):
        """Test removing a nonexistent job."""
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
        
        scheduler_service.jobs["test_job"].enabled = False
        scheduler_service.enable_job("test_job")
        assert scheduler_service.jobs["test_job"].enabled is True
    
    def test_disable_job(self, scheduler_service):
        """Test disabling a job."""
        scheduler_service.add_job(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        scheduler_service.disable_job("test_job")
        assert scheduler_service.jobs["test_job"].enabled is False
    
    def test_start_scheduler(self, scheduler_service):
        """Test starting the scheduler."""
        scheduler_service.start_scheduler()
        assert scheduler_service.is_running is True
        assert scheduler_service.scheduler_thread is not None
        scheduler_service.stop_scheduler()
    
    def test_stop_scheduler(self, scheduler_service):
        """Test stopping the scheduler."""
        scheduler_service.start_scheduler()
        scheduler_service.stop_scheduler()
        assert scheduler_service.is_running is False
        assert scheduler_service.stop_event.is_set()
    
    def test_run_job_now(self, scheduler_service, mock_orchestrator):
        """Test running a job immediately."""
        # Add a job
        scheduler_service.add_job(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        # Run the job
        result = scheduler_service.run_job_now("test_job")
        
        # Verify the orchestrator was called
        mock_orchestrator.analyze_multiple_stocks.assert_called_once_with(["AAPL"])
        assert result == mock_orchestrator.analyze_multiple_stocks.return_value
    
    def test_run_nonexistent_job(self, scheduler_service):
        """Test running a nonexistent job."""
        with pytest.raises(SchedulingError, match="Job with ID 'nonexistent' not found"):
            scheduler_service.run_job_now("nonexistent")
    
    def test_get_scheduler_status(self, scheduler_service):
        """Test getting scheduler status."""
        status = scheduler_service.get_scheduler_status()
        assert isinstance(status, dict)
        assert "total_jobs" in status
        assert "active_jobs" in status
        assert "system_status" in status
    
    def test_get_job_status(self, scheduler_service):
        """Test getting job status."""
        # Add a job
        scheduler_service.add_job(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        status = scheduler_service.get_job_status("test_job")
        assert status["job_id"] == "test_job"
        assert status["name"] == "Test Job"
        assert status["symbols"] == ["AAPL"]
        assert status["interval"] == "daily"
    
    def test_get_nonexistent_job_status(self, scheduler_service):
        """Test getting status of a nonexistent job."""
        with pytest.raises(SchedulingError, match="Job with ID 'nonexistent' not found"):
            scheduler_service.get_job_status("nonexistent")
    
    def test_generate_summary_report(self, scheduler_service):
        """Test generating a summary report."""
        # Add some jobs and execution history
        scheduler_service.add_job(
            job_id="test_job1",
            name="Test Job 1",
            symbols=["AAPL"],
            interval="daily"
        )
        
        scheduler_service.add_job(
            job_id="test_job2",
            name="Test Job 2",
            symbols=["MSFT"],
            interval="daily"
        )
        
        # Add some execution history
        scheduler_service._record_execution(
            "test_job1",
            True,
            mock_orchestrator.analyze_multiple_stocks.return_value,
            None
        )
        
        report = scheduler_service.generate_summary_report(period_days=7)
        assert isinstance(report, str)
        assert "Stock Analysis Scheduler Report" in report
        assert "test_job1" in report
        assert "test_job2" in report
    
    def test_calculate_next_run_daily(self, scheduler_service):
        """Test calculating next run time for daily interval."""
        next_run = scheduler_service._calculate_next_run("daily")
        assert isinstance(next_run, datetime)
        assert next_run.hour == 9
        assert next_run.minute == 0
        assert next_run > datetime.now()
    
    def test_calculate_next_run_weekly(self, scheduler_service):
        """Test calculating next run time for weekly interval."""
        next_run = scheduler_service._calculate_next_run("weekly")
        assert isinstance(next_run, datetime)
        assert next_run.hour == 9
        assert next_run.minute == 0
        assert next_run.weekday() == 0  # Monday
        assert next_run > datetime.now()
    
    def test_calculate_next_run_hourly(self, scheduler_service):
        """Test calculating next run time for hourly interval."""
        next_run = scheduler_service._calculate_next_run("hourly")
        assert isinstance(next_run, datetime)
        assert next_run.minute == 0
        assert next_run > datetime.now()
    
    def test_calculate_next_run_custom_hours(self, scheduler_service):
        """Test calculating next run time for custom hours interval."""
        next_run = scheduler_service._calculate_next_run("every_2_hours")
        assert isinstance(next_run, datetime)
        assert next_run > datetime.now()
    
    def test_calculate_next_run_custom_minutes(self, scheduler_service):
        """Test calculating next run time for custom minutes interval."""
        next_run = scheduler_service._calculate_next_run("every_30_minutes")
        assert isinstance(next_run, datetime)
        assert next_run > datetime.now()
    
    def test_calculate_next_run_invalid(self, scheduler_service):
        """Test calculating next run time for invalid interval."""
        next_run = scheduler_service._calculate_next_run("invalid")
        assert isinstance(next_run, datetime)
        assert next_run > datetime.now()
    
    def test_record_execution_success(self, scheduler_service):
        """Test recording successful job execution."""
        mock_report = MagicMock()
        mock_report.total_stocks = 1
        mock_report.successful_analyses = 1
        mock_report.failed_analyses = 0
        mock_report.execution_time = 5.0
        mock_report.success_rate = 100.0
        
        scheduler_service._record_execution(
            "test_job",
            True,
            mock_report,
            None
        )
        
        assert len(scheduler_service.execution_history) == 1
        record = scheduler_service.execution_history[0]
        assert record["job_id"] == "test_job"
        assert record["success"] is True
        assert record["total_stocks"] == 1
    
    def test_record_execution_failure(self, scheduler_service):
        """Test recording failed job execution."""
        scheduler_service._record_execution(
            "test_job",
            False,
            None,
            "Test error"
        )
        
        assert len(scheduler_service.execution_history) == 1
        record = scheduler_service.execution_history[0]
        assert record["job_id"] == "test_job"
        assert record["success"] is False
        assert record["error"] == "Test error"
    
    def test_execute_job_success(self, scheduler_service, mock_orchestrator):
        """Test successful job execution."""
        # Add a job
        job = ScheduledJob(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        # Execute the job
        result = scheduler_service._execute_job(job)
        
        assert result == mock_orchestrator.analyze_multiple_stocks.return_value
        assert job.success_count == 1
        assert job.failure_count == 0
        assert job.last_error is None
    
    def test_execute_job_failure_with_retry(self, scheduler_service, mock_orchestrator):
        """Test job execution with retry on failure."""
        # Add a job
        job = ScheduledJob(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        # Mock the orchestrator to fail once then succeed
        mock_orchestrator.analyze_multiple_stocks.side_effect = [
            Exception("First attempt failed"),
            mock_orchestrator.analyze_multiple_stocks.return_value
        ]
        
        # Set short retry delay for testing
        scheduler_service.retry_delay_minutes = 0
        
        # Execute the job
        result = scheduler_service._execute_job(job)
        
        assert result == mock_orchestrator.analyze_multiple_stocks.return_value
        assert job.success_count == 1
        assert job.failure_count == 0
        assert job.last_error is None
    
    def test_execute_job_failure_exhausted_retries(self, scheduler_service, mock_orchestrator):
        """Test job execution with all retries exhausted."""
        # Add a job
        job = ScheduledJob(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        # Mock the orchestrator to always fail
        error_msg = "Test error"
        mock_orchestrator.analyze_multiple_stocks.side_effect = Exception(error_msg)
        
        # Set short retry delay for testing
        scheduler_service.retry_delay_minutes = 0
        
        # Execute the job and expect exception
        with pytest.raises(SchedulingError, match=f"Job 'test_job' failed after .* retries"):
            scheduler_service._execute_job(job)
        
        assert job.success_count == 0
        assert job.failure_count == 1
        assert job.last_error == error_msg
    
    @patch('smtplib.SMTP')
    def test_send_notification_success(self, mock_smtp, scheduler_service):
        """Test sending success notification."""
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
        
        mock_report = MagicMock()
        mock_report.total_stocks = 1
        mock_report.successful_analyses = 1
        mock_report.failed_analyses = 0
        mock_report.success_rate = 100.0
        mock_report.execution_time = 5.0
        mock_report.failed_symbols = []
        mock_report.results = []
        
        mock_server = MagicMock()
        mock_smtp.return_value = mock_server
        
        scheduler_service._send_notification(job, mock_report, True)
        
        mock_server.send_message.assert_called_once()
    
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
        
        mock_server = MagicMock()
        mock_smtp.return_value = mock_server
        
        scheduler_service._send_notification(job, None, False, "Test error")
        
        mock_server.send_message.assert_called_once()
    
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
        
        mock_report = MagicMock()
        mock_report.total_stocks = 1
        mock_report.successful_analyses = 1
        mock_report.failed_analyses = 0
        mock_report.success_rate = 100.0
        mock_report.execution_time = 5.0
        mock_report.failed_symbols = []
        mock_report.results = []
        
        body = scheduler_service._create_success_notification_body(job, mock_report)
        
        assert "Stock Analysis Job Completed Successfully" in body
    
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
    
    def test_save_jobs_config(self, scheduler_service, tmp_path):
        """Test saving jobs configuration."""
        # Add a job
        scheduler_service.add_job(
            job_id="test_job",
            name="Test Job",
            symbols=["AAPL"],
            interval="daily"
        )
        
        # Save configuration
        config_file = tmp_path / "test_config.json"
        scheduler_service.save_jobs_config(str(config_file))
        
        # Verify file was created and contains job data
        assert config_file.exists()
        with open(config_file) as f:
            config_data = json.load(f)
            assert "test_job" in config_data
    
    def test_load_jobs_config(self, scheduler_service, tmp_path):
        """Test loading jobs configuration."""
        # Create a test config file
        config_file = tmp_path / "test_config.json"
        config_data = {
            "test_job": {
                "job_id": "test_job",
                "name": "Test Job",
                "symbols": ["AAPL"],
                "interval": "daily",
                "enabled": True,
                "export_format": "excel",
                "notification_enabled": True
            }
        }
        
        with open(config_file, "w") as f:
            json.dump(config_data, f)
        
        # Load configuration
        scheduler_service.load_jobs_config(str(config_file))
        
        # Verify job was loaded
        assert "test_job" in scheduler_service.jobs
        job = scheduler_service.jobs["test_job"]
        assert job.name == "Test Job"
        assert job.symbols == ["AAPL"]
    
    def test_load_jobs_config_file_not_found(self, scheduler_service, tmp_path):
        """Test loading jobs configuration when file doesn't exist."""
        config_file = tmp_path / "nonexistent.json"
        scheduler_service.load_jobs_config(str(config_file))
        assert len(scheduler_service.jobs) == 0