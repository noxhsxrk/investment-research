"""Scheduling and automation service for stock analysis system.

This module provides the SchedulerService class that handles automated analysis
execution, scheduling, error recovery, and notification management.
"""

import logging
import time
import threading
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import schedule
import json
import os

from stock_analysis.orchestrator import StockAnalysisOrchestrator, AnalysisReport
from stock_analysis.utils.config import config
from stock_analysis.utils.logging import get_logger
from stock_analysis.utils.exceptions import SchedulingError

logger = get_logger(__name__)


@dataclass
class ScheduledJob:
    """Represents a scheduled analysis job."""
    job_id: str
    name: str
    symbols: List[str]
    interval: str  # 'daily', 'weekly', 'hourly', or cron-like expression
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    last_error: Optional[str] = None
    export_format: str = "excel"
    notification_enabled: bool = True


@dataclass
class NotificationConfig:
    """Configuration for notifications."""
    enabled: bool = True
    email_host: Optional[str] = None
    email_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    email_use_tls: bool = True
    recipients: List[str] = None
    send_on_success: bool = True
    send_on_failure: bool = True
    send_summary_reports: bool = True


@dataclass
class SchedulerReport:
    """Report of scheduler execution and status."""
    total_jobs: int
    active_jobs: int
    successful_runs_today: int
    failed_runs_today: int
    next_scheduled_run: Optional[datetime]
    recent_jobs: List[Dict[str, Any]]
    system_status: str


class SchedulerService:
    """Service for managing scheduled stock analysis jobs.
    
    This service handles automated execution of stock analysis, scheduling,
    error recovery, and notification management.
    """
    
    def __init__(self, orchestrator: Optional[StockAnalysisOrchestrator] = None):
        """Initialize the scheduler service.
        
        Args:
            orchestrator: Stock analysis orchestrator instance
        """
        logger.info("Initializing Scheduler Service")
        
        self.orchestrator = orchestrator or StockAnalysisOrchestrator()
        self.jobs: Dict[str, ScheduledJob] = {}
        self.notification_config = self._load_notification_config()
        
        # Scheduler state
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Job execution tracking
        self.execution_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
        # Error recovery settings
        self.max_retry_attempts = 3
        self.retry_delay_minutes = 30
        
        # Job persistence
        self.jobs_config_file = "scheduled_jobs.json"
        self.scheduler_status_file = "scheduler_status.json"
        self.scheduler_pid_file = "scheduler.pid"
        
        # Load existing jobs
        self.load_jobs_config(self.jobs_config_file)
        
        logger.info("Scheduler Service initialized")
    
    def add_job(self, 
                job_id: str,
                name: str,
                symbols: List[str],
                interval: str,
                export_format: str = "excel",
                notification_enabled: bool = True) -> None:
        """Add a new scheduled job.
        
        Args:
            job_id: Unique identifier for the job
            name: Human-readable name for the job
            symbols: List of stock symbols to analyze
            interval: Schedule interval ('daily', 'weekly', 'hourly', or cron expression)
            export_format: Export format for results
            notification_enabled: Whether to send notifications for this job
        """
        logger.info(f"Adding scheduled job: {job_id} ({name})")
        
        if job_id in self.jobs:
            raise SchedulingError(f"Job with ID '{job_id}' already exists")
        
        job = ScheduledJob(
            job_id=job_id,
            name=name,
            symbols=symbols,
            interval=interval,
            export_format=export_format,
            notification_enabled=notification_enabled
        )
        
        # Calculate next run time
        job.next_run = self._calculate_next_run(interval)
        
        self.jobs[job_id] = job
        
        # Save jobs configuration
        self.save_jobs_config(self.jobs_config_file)
        
        # Schedule the job if scheduler is running
        if self.is_running:
            self._schedule_job(job)
        
        logger.info(f"Job '{job_id}' added successfully. Next run: {job.next_run}")
    
    def remove_job(self, job_id: str) -> None:
        """Remove a scheduled job.
        
        Args:
            job_id: ID of the job to remove
        """
        if job_id not in self.jobs:
            raise SchedulingError(f"Job with ID '{job_id}' not found")
        
        # Cancel the scheduled job
        schedule.cancel_job(job_id)
        
        # Remove from jobs dictionary
        del self.jobs[job_id]
        
        # Save jobs configuration
        self.save_jobs_config(self.jobs_config_file)
        
        logger.info(f"Job '{job_id}' removed successfully")
    
    def enable_job(self, job_id: str) -> None:
        """Enable a scheduled job.
        
        Args:
            job_id: ID of the job to enable
        """
        if job_id not in self.jobs:
            raise SchedulingError(f"Job with ID '{job_id}' not found")
        
        job = self.jobs[job_id]
        job.enabled = True
        
        if self.is_running:
            self._schedule_job(job)
        
        logger.info(f"Job '{job_id}' enabled")
    
    def disable_job(self, job_id: str) -> None:
        """Disable a scheduled job.
        
        Args:
            job_id: ID of the job to disable
        """
        if job_id not in self.jobs:
            raise SchedulingError(f"Job with ID '{job_id}' not found")
        
        job = self.jobs[job_id]
        job.enabled = False
        
        # Cancel the scheduled job
        schedule.cancel_job(job_id)
        
        logger.info(f"Job '{job_id}' disabled")
    
    def start_scheduler(self) -> None:
        """Start the scheduler service."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        logger.info("Starting scheduler service")
        
        self.is_running = True
        self.stop_event.clear()
        
        # Schedule all enabled jobs
        for job in self.jobs.values():
            if job.enabled:
                self._schedule_job(job)
        
        # Save PID and status
        self.save_pid_file()
        self.save_scheduler_status()
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Scheduler service started successfully")
    
    def stop_scheduler(self) -> None:
        """Stop the scheduler service."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        logger.info("Stopping scheduler service")
        
        self.is_running = False
        self.stop_event.set()
        
        # Clear all scheduled jobs
        schedule.clear()
        
        # Wait for scheduler thread to finish
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        # Clean up status files
        self.save_scheduler_status()
        self.remove_pid_file()
        
        logger.info("Scheduler service stopped")
    
    def run_job_now(self, job_id: str) -> AnalysisReport:
        """Execute a job immediately.
        
        Args:
            job_id: ID of the job to run
            
        Returns:
            AnalysisReport: Results of the analysis
        """
        if job_id not in self.jobs:
            raise SchedulingError(f"Job with ID '{job_id}' not found")
        
        job = self.jobs[job_id]
        logger.info(f"Running job '{job_id}' immediately")
        
        return self._execute_job(job)
    
    def get_scheduler_status(self) -> SchedulerReport:
        """Get current scheduler status and statistics.
        
        Returns:
            SchedulerReport: Current scheduler status
        """
        active_jobs = sum(1 for job in self.jobs.values() if job.enabled)
        
        # Calculate today's statistics
        today = datetime.now().date()
        today_history = [
            h for h in self.execution_history
            if datetime.fromisoformat(h['timestamp']).date() == today
        ]
        
        successful_runs_today = sum(1 for h in today_history if h['success'])
        failed_runs_today = sum(1 for h in today_history if not h['success'])
        
        # Find next scheduled run
        next_run = None
        if self.jobs:
            next_runs = [job.next_run for job in self.jobs.values() if job.enabled and job.next_run]
            if next_runs:
                next_run = min(next_runs)
        
        # Get recent job history
        recent_jobs = sorted(
            self.execution_history[-10:],
            key=lambda x: x['timestamp'],
            reverse=True
        )
        
        return SchedulerReport(
            total_jobs=len(self.jobs),
            active_jobs=active_jobs,
            successful_runs_today=successful_runs_today,
            failed_runs_today=failed_runs_today,
            next_scheduled_run=next_run,
            recent_jobs=recent_jobs,
            system_status="Running" if self.is_running else "Stopped"
        )
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status information for a specific job.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Dict containing job status information
        """
        if job_id not in self.jobs:
            raise SchedulingError(f"Job with ID '{job_id}' not found")
        
        job = self.jobs[job_id]
        
        # Get recent execution history for this job
        job_history = [
            h for h in self.execution_history
            if h['job_id'] == job_id
        ][-5:]  # Last 5 executions
        
        return {
            'job_id': job.job_id,
            'name': job.name,
            'symbols': job.symbols,
            'interval': job.interval,
            'enabled': job.enabled,
            'last_run': job.last_run.isoformat() if job.last_run else None,
            'next_run': job.next_run.isoformat() if job.next_run else None,
            'success_count': job.success_count,
            'failure_count': job.failure_count,
            'last_error': job.last_error,
            'export_format': job.export_format,
            'notification_enabled': job.notification_enabled,
            'recent_executions': job_history
        }
    
    def generate_summary_report(self, period_days: int = 7) -> str:
        """Generate a summary report of scheduler activity.
        
        Args:
            period_days: Number of days to include in the report
            
        Returns:
            str: Human-readable summary report
        """
        cutoff_date = datetime.now() - timedelta(days=period_days)
        
        # Filter history for the specified period
        period_history = [
            h for h in self.execution_history
            if datetime.fromisoformat(h['timestamp']) >= cutoff_date
        ]
        
        # Calculate statistics
        total_executions = len(period_history)
        successful_executions = sum(1 for h in period_history if h['success'])
        failed_executions = total_executions - successful_executions
        success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
        
        # Group by job
        job_stats = {}
        for history in period_history:
            job_id = history['job_id']
            if job_id not in job_stats:
                job_stats[job_id] = {'success': 0, 'failure': 0}
            
            if history['success']:
                job_stats[job_id]['success'] += 1
            else:
                job_stats[job_id]['failure'] += 1
        
        # Generate report
        report_lines = [
            f"Stock Analysis Scheduler Report",
            f"Period: {period_days} days ({cutoff_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')})",
            f"=" * 60,
            f"",
            f"Overall Statistics:",
            f"  Total executions: {total_executions}",
            f"  Successful executions: {successful_executions}",
            f"  Failed executions: {failed_executions}",
            f"  Success rate: {success_rate:.1f}%",
            f"",
            f"Active Jobs: {sum(1 for job in self.jobs.values() if job.enabled)}",
            f"Total Jobs: {len(self.jobs)}",
            f"",
        ]
        
        if job_stats:
            report_lines.append("Job Performance:")
            for job_id, stats in job_stats.items():
                job_name = self.jobs.get(job_id, {}).get('name', job_id)
                total_job_runs = stats['success'] + stats['failure']
                job_success_rate = (stats['success'] / total_job_runs * 100) if total_job_runs > 0 else 0
                
                report_lines.append(f"  {job_name} ({job_id}):")
                report_lines.append(f"    Runs: {total_job_runs}, Success: {stats['success']}, "
                                  f"Failed: {stats['failure']}, Rate: {job_success_rate:.1f}%")
            report_lines.append("")
        
        # Recent failures
        recent_failures = [h for h in period_history if not h['success']][-5:]
        if recent_failures:
            report_lines.append("Recent Failures:")
            for failure in recent_failures:
                timestamp = datetime.fromisoformat(failure['timestamp']).strftime('%Y-%m-%d %H:%M')
                report_lines.append(f"  {timestamp} - {failure['job_id']}: {failure.get('error', 'Unknown error')}")
            report_lines.append("")
        
        # Next scheduled runs
        upcoming_runs = []
        for job in self.jobs.values():
            if job.enabled and job.next_run:
                upcoming_runs.append((job.next_run, job.job_id, job.name))
        
        upcoming_runs.sort()
        if upcoming_runs:
            report_lines.append("Upcoming Scheduled Runs:")
            for next_run, job_id, job_name in upcoming_runs[:5]:
                report_lines.append(f"  {next_run.strftime('%Y-%m-%d %H:%M')} - {job_name} ({job_id})")
        
        return "\n".join(report_lines)
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop that runs in a separate thread."""
        logger.info("Scheduler loop started")
        
        while not self.stop_event.is_set():
            try:
                # Run pending scheduled jobs
                schedule.run_pending()
                
                # Sleep for a short interval
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(5)  # Wait before retrying
        
        logger.info("Scheduler loop stopped")
    
    def _schedule_job(self, job: ScheduledJob) -> None:
        """Schedule a job using the schedule library.
        
        Args:
            job: Job to schedule
        """
        if not job.enabled:
            return
        
        # Create a wrapper function that includes error handling
        def job_wrapper():
            try:
                self._execute_job(job)
            except Exception as e:
                logger.error(f"Scheduled job '{job.job_id}' failed: {str(e)}")
        
        # Schedule based on interval
        if job.interval == "daily":
            schedule.every().day.at("09:00").do(job_wrapper).tag(job.job_id)
        elif job.interval == "weekly":
            schedule.every().monday.at("09:00").do(job_wrapper).tag(job.job_id)
        elif job.interval == "hourly":
            schedule.every().hour.do(job_wrapper).tag(job.job_id)
        elif job.interval.startswith("every_"):
            # Handle custom intervals like "every_2_hours", "every_30_minutes"
            parts = job.interval.split("_")
            if len(parts) >= 3:
                interval_num = int(parts[1])
                interval_unit = parts[2]
                
                if interval_unit == "hours":
                    schedule.every(interval_num).hours.do(job_wrapper).tag(job.job_id)
                elif interval_unit == "minutes":
                    schedule.every(interval_num).minutes.do(job_wrapper).tag(job.job_id)
                elif interval_unit == "days":
                    schedule.every(interval_num).days.do(job_wrapper).tag(job.job_id)
        else:
            logger.warning(f"Unsupported interval '{job.interval}' for job '{job.job_id}'")
    
    def _execute_job(self, job: ScheduledJob) -> AnalysisReport:
        """Execute a scheduled job with error recovery.
        
        Args:
            job: Job to execute
            
        Returns:
            AnalysisReport: Results of the analysis
        """
        logger.info(f"Executing scheduled job: {job.job_id} ({job.name})")
        start_time = datetime.now()
        
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retry_attempts:
            try:
                # Update job status
                job.last_run = start_time
                
                # Execute analysis
                report = self.orchestrator.analyze_multiple_stocks(job.symbols)
                
                # Export results
                if report.results:
                    export_filename = f"{job.job_id}_{start_time.strftime('%Y%m%d_%H%M%S')}"
                    export_path = self.orchestrator.export_results(
                        report.results, 
                        job.export_format, 
                        export_filename
                    )
                    logger.info(f"Results exported to: {export_path}")
                
                # Update job success statistics
                job.success_count += 1
                job.last_error = None
                job.last_run = start_time
                job.next_run = self._calculate_next_run(job.interval)
                
                # Save job configuration
                self.save_jobs_config(self.jobs_config_file)
                
                # Record execution history
                self._record_execution(job.job_id, True, report, None)
                
                # Send success notification
                if job.notification_enabled and self.notification_config.send_on_success:
                    self._send_notification(job, report, True)
                
                logger.info(f"Job '{job.job_id}' completed successfully")
                return report
                
            except Exception as e:
                last_error = str(e)
                retry_count += 1
                
                logger.error(f"Job '{job.job_id}' failed (attempt {retry_count}/{self.max_retry_attempts + 1}): {last_error}")
                
                if retry_count <= self.max_retry_attempts:
                    logger.info(f"Retrying job '{job.job_id}' in {self.retry_delay_minutes} minutes")
                    time.sleep(self.retry_delay_minutes * 60)
                else:
                    # All retries exhausted
                    job.failure_count += 1
                    job.last_error = last_error
                    job.last_run = start_time
                    job.next_run = self._calculate_next_run(job.interval)
                    
                    # Save job configuration
                    self.save_jobs_config(self.jobs_config_file)
                    
                    # Record execution history
                    self._record_execution(job.job_id, False, None, last_error)
                    
                    # Send failure notification
                    if job.notification_enabled and self.notification_config.send_on_failure:
                        self._send_notification(job, None, False, last_error)
                    
                    raise SchedulingError(f"Job '{job.job_id}' failed after {self.max_retry_attempts} retries: {last_error}")
    
    def _calculate_next_run(self, interval: str) -> datetime:
        """Calculate the next run time for a given interval.
        
        Args:
            interval: Schedule interval
            
        Returns:
            datetime: Next run time
        """
        now = datetime.now()
        
        if interval == "daily":
            next_run = now.replace(hour=9, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
        elif interval == "weekly":
            # Next Monday at 9 AM
            days_ahead = 0 - now.weekday()  # Monday is 0
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(hour=9, minute=0, second=0, microsecond=0)
        elif interval == "hourly":
            next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif interval.startswith("every_"):
            parts = interval.split("_")
            if len(parts) >= 3:
                interval_num = int(parts[1])
                interval_unit = parts[2]
                
                if interval_unit == "hours":
                    next_run = now + timedelta(hours=interval_num)
                elif interval_unit == "minutes":
                    next_run = now + timedelta(minutes=interval_num)
                elif interval_unit == "days":
                    next_run = now + timedelta(days=interval_num)
                else:
                    next_run = now + timedelta(hours=1)  # Default fallback
            else:
                next_run = now + timedelta(hours=1)  # Default fallback
        else:
            next_run = now + timedelta(hours=1)  # Default fallback
        
        return next_run
    
    def _record_execution(self, 
                         job_id: str, 
                         success: bool, 
                         report: Optional[AnalysisReport], 
                         error: Optional[str]) -> None:
        """Record job execution in history.
        
        Args:
            job_id: ID of the executed job
            success: Whether the execution was successful
            report: Analysis report (if successful)
            error: Error message (if failed)
        """
        execution_record = {
            'job_id': job_id,
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'error': error
        }
        
        if report:
            execution_record.update({
                'total_stocks': report.total_stocks,
                'successful_analyses': report.successful_analyses,
                'failed_analyses': report.failed_analyses,
                'execution_time': report.execution_time,
                'success_rate': report.success_rate
            })
        
        self.execution_history.append(execution_record)
        
        # Limit history size
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size:]
    
    def _send_notification(self, 
                          job: ScheduledJob, 
                          report: Optional[AnalysisReport], 
                          success: bool, 
                          error: Optional[str] = None) -> None:
        """Send notification email for job completion.
        
        Args:
            job: The executed job
            report: Analysis report (if successful)
            success: Whether the job was successful
            error: Error message (if failed)
        """
        if not self.notification_config.enabled or not self.notification_config.recipients:
            return
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.notification_config.email_username
            msg['To'] = ', '.join(self.notification_config.recipients)
            
            if success:
                msg['Subject'] = f"Stock Analysis Job Completed: {job.name}"
                body = self._create_success_notification_body(job, report)
            else:
                msg['Subject'] = f"Stock Analysis Job Failed: {job.name}"
                body = self._create_failure_notification_body(job, error)
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.notification_config.email_host, self.notification_config.email_port)
            if self.notification_config.email_use_tls:
                server.starttls()
            
            if self.notification_config.email_username and self.notification_config.email_password:
                server.login(self.notification_config.email_username, self.notification_config.email_password)
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Notification sent for job '{job.job_id}'")
            
        except Exception as e:
            logger.error(f"Failed to send notification for job '{job.job_id}': {str(e)}")
    
    def _create_success_notification_body(self, job: ScheduledJob, report: AnalysisReport) -> str:
        """Create email body for successful job notification.
        
        Args:
            job: The executed job
            report: Analysis report
            
        Returns:
            str: Email body text
        """
        body_lines = [
            f"Stock Analysis Job Completed Successfully",
            f"",
            f"Job Details:",
            f"  Name: {job.name}",
            f"  ID: {job.job_id}",
            f"  Execution Time: {job.last_run.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Next Run: {job.next_run.strftime('%Y-%m-%d %H:%M:%S') if job.next_run else 'Not scheduled'}",
            f"",
            f"Analysis Results:",
            f"  Total Stocks: {report.total_stocks}",
            f"  Successful Analyses: {report.successful_analyses}",
            f"  Failed Analyses: {report.failed_analyses}",
            f"  Success Rate: {report.success_rate:.1f}%",
            f"  Execution Time: {report.execution_time:.2f} seconds",
            f"",
        ]
        
        if report.failed_symbols:
            body_lines.extend([
                f"Failed Symbols: {', '.join(report.failed_symbols)}",
                f""
            ])
        
        if report.results:
            # Add some analysis highlights
            buy_count = sum(1 for r in report.results if r.fair_value.recommendation == "BUY")
            sell_count = sum(1 for r in report.results if r.fair_value.recommendation == "SELL")
            hold_count = sum(1 for r in report.results if r.fair_value.recommendation == "HOLD")
            
            body_lines.extend([
                f"Investment Recommendations:",
                f"  Buy: {buy_count}",
                f"  Hold: {hold_count}",
                f"  Sell: {sell_count}",
                f""
            ])
        
        body_lines.append("This is an automated notification from the Stock Analysis System.")
        
        return "\n".join(body_lines)
    
    def _create_failure_notification_body(self, job: ScheduledJob, error: str) -> str:
        """Create email body for failed job notification.
        
        Args:
            job: The executed job
            error: Error message
            
        Returns:
            str: Email body text
        """
        body_lines = [
            f"Stock Analysis Job Failed",
            f"",
            f"Job Details:",
            f"  Name: {job.name}",
            f"  ID: {job.job_id}",
            f"  Execution Time: {job.last_run.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Next Retry: {job.next_run.strftime('%Y-%m-%d %H:%M:%S') if job.next_run else 'Not scheduled'}",
            f"",
            f"Error Details:",
            f"  {error}",
            f"",
            f"Job Statistics:",
            f"  Total Successes: {job.success_count}",
            f"  Total Failures: {job.failure_count}",
            f"",
            f"Please check the system logs for more detailed error information.",
            f"",
            f"This is an automated notification from the Stock Analysis System."
        ]
        
        return "\n".join(body_lines)
    
    def _load_notification_config(self) -> NotificationConfig:
        """Load notification configuration from config file.
        
        Returns:
            NotificationConfig: Notification configuration
        """
        return NotificationConfig(
            enabled=config.get('stock_analysis.notifications.enabled', True),
            email_host=config.get('stock_analysis.notifications.email_host'),
            email_port=config.get('stock_analysis.notifications.email_port', 587),
            email_username=config.get('stock_analysis.notifications.email_username'),
            email_password=config.get('stock_analysis.notifications.email_password'),
            email_use_tls=config.get('stock_analysis.notifications.email_use_tls', True),
            recipients=config.get('stock_analysis.notifications.recipients', []),
            send_on_success=config.get('stock_analysis.notifications.send_on_success', True),
            send_on_failure=config.get('stock_analysis.notifications.send_on_failure', True),
            send_summary_reports=config.get('stock_analysis.notifications.send_summary_reports', True)
        )
    
    def save_jobs_config(self, filepath: str = "scheduled_jobs.json") -> None:
        """Save current jobs configuration to file.
        
        Args:
            filepath: Path to save the configuration
        """
        jobs_data = {}
        for job_id, job in self.jobs.items():
            jobs_data[job_id] = {
                'job_id': job.job_id,
                'name': job.name,
                'symbols': job.symbols,
                'interval': job.interval,
                'enabled': job.enabled,
                'last_run': job.last_run.isoformat() if job.last_run else None,
                'next_run': job.next_run.isoformat() if job.next_run else None,
                'success_count': job.success_count,
                'failure_count': job.failure_count,
                'last_error': job.last_error,
                'export_format': job.export_format,
                'notification_enabled': job.notification_enabled
            }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(jobs_data, f, indent=2)
            logger.info(f"Jobs configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save jobs configuration: {str(e)}")
            raise SchedulingError(f"Failed to save jobs configuration: {str(e)}")
    
    def load_jobs_config(self, filepath: str = "scheduled_jobs.json") -> None:
        """Load jobs configuration from file.
        
        Args:
            filepath: Path to load the configuration from
        """
        if not os.path.exists(filepath):
            logger.info(f"No jobs configuration file found at {filepath}")
            return
        
        try:
            with open(filepath, 'r') as f:
                jobs_data = json.load(f)
            
            for job_id, job_config in jobs_data.items():
                job = ScheduledJob(
                    job_id=job_id,
                    name=job_config['name'],
                    symbols=job_config['symbols'],
                    interval=job_config['interval'],
                    enabled=job_config.get('enabled', True),
                    success_count=job_config.get('success_count', 0),
                    failure_count=job_config.get('failure_count', 0),
                    last_error=job_config.get('last_error'),
                    export_format=job_config.get('export_format', 'excel'),
                    notification_enabled=job_config.get('notification_enabled', True)
                )
                
                # Parse datetime fields
                if job_config.get('last_run'):
                    job.last_run = datetime.fromisoformat(job_config['last_run'])
                if job_config.get('next_run'):
                    job.next_run = datetime.fromisoformat(job_config['next_run'])
                else:
                    job.next_run = self._calculate_next_run(job.interval)
                
                self.jobs[job_id] = job
            
            logger.info(f"Loaded {len(jobs_data)} jobs from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load jobs configuration: {str(e)}")
            raise SchedulingError(f"Failed to load jobs configuration: {str(e)}")
    
    def save_scheduler_status(self) -> None:
        """Save current scheduler status to file."""
        try:
            status_data = {
                'is_running': self.is_running,
                'pid': os.getpid() if self.is_running else None,
                'started_at': datetime.now().isoformat() if self.is_running else None,
                'total_jobs': len(self.jobs),
                'active_jobs': sum(1 for job in self.jobs.values() if job.enabled)
            }
            
            with open(self.scheduler_status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save scheduler status: {str(e)}")
    
    def load_scheduler_status(self) -> Dict[str, Any]:
        """Load scheduler status from file.
        
        Returns:
            Dict containing scheduler status information
        """
        try:
            if os.path.exists(self.scheduler_status_file):
                with open(self.scheduler_status_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load scheduler status: {str(e)}")
        
        return {
            'is_running': False,
            'pid': None,
            'started_at': None,
            'total_jobs': 0,
            'active_jobs': 0
        }
    
    def is_scheduler_running(self) -> bool:
        """Check if scheduler is currently running.
        
        Returns:
            bool: True if scheduler is running, False otherwise
        """
        # First check if PID file exists
        if not os.path.exists(self.scheduler_pid_file):
            return False
            
        # Read PID from file
        try:
            with open(self.scheduler_pid_file, 'r') as f:
                pid = int(f.read().strip())
        except (OSError, ValueError):
            return False
            
        # Check if the process is actually running
        try:
            os.kill(pid, 0)  # Send signal 0 to check if process exists
            return True
        except OSError:
            # Process doesn't exist, clean up PID file
            self.remove_pid_file()
            return False
    
    def save_pid_file(self) -> None:
        """Save current process PID to file."""
        try:
            with open(self.scheduler_pid_file, 'w') as f:
                f.write(str(os.getpid()))
        except Exception as e:
            logger.warning(f"Failed to save PID file: {str(e)}")
    
    def remove_pid_file(self) -> None:
        """Remove PID file."""
        try:
            if os.path.exists(self.scheduler_pid_file):
                os.remove(self.scheduler_pid_file)
        except Exception as e:
            logger.warning(f"Failed to remove PID file: {str(e)}")
    
    def load_jobs_config(self, filepath: str = "scheduled_jobs.json") -> None:
        """Load jobs configuration from file.
        
        Args:
            filepath: Path to load the configuration from
        """
        if not os.path.exists(filepath):
            logger.info(f"Jobs configuration file {filepath} not found")
            return
        
        try:
            with open(filepath, 'r') as f:
                jobs_data = json.load(f)
            
            for job_id, job_config in jobs_data.items():
                job = ScheduledJob(
                    job_id=job_id,
                    name=job_config['name'],
                    symbols=job_config['symbols'],
                    interval=job_config['interval'],
                    enabled=job_config.get('enabled', True),
                    export_format=job_config.get('export_format', 'excel'),
                    notification_enabled=job_config.get('notification_enabled', True),
                    success_count=job_config.get('success_count', 0),
                    failure_count=job_config.get('failure_count', 0),
                    last_error=job_config.get('last_error')
                )
                
                if job_config.get('last_run'):
                    job.last_run = datetime.fromisoformat(job_config['last_run'])
                
                if job_config.get('next_run'):
                    job.next_run = datetime.fromisoformat(job_config['next_run'])
                else:
                    job.next_run = self._calculate_next_run(job.interval)
                
                self.jobs[job_id] = job
            
            logger.info(f"Loaded {len(jobs_data)} jobs from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load jobs configuration from {filepath}: {str(e)}")
            raise SchedulingError(f"Failed to load jobs configuration from {filepath}: {str(e)}")