"""Scheduler daemon management utilities.

This module provides utilities for managing the scheduler daemon process.
"""

import os
import json
import signal
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from stock_analysis.utils.logging import get_logger

logger = get_logger(__name__)

class SchedulerDaemon:
    """Utilities for managing the scheduler daemon process."""
    
    def __init__(self):
        """Initialize the scheduler daemon manager."""
        self.scheduler_status_file = "scheduler_status.json"
        self.scheduler_pid_file = "scheduler.pid"
    
    def save_status(self, is_running: bool, pid: Optional[int] = None) -> None:
        """Save current scheduler status to file.
        
        Args:
            is_running: Whether the scheduler is running
            pid: Process ID of the scheduler
        """
        try:
            status_data = {
                'is_running': is_running,
                'pid': pid if is_running else None,
                'started_at': datetime.now().isoformat() if is_running else None
            }
            
            with open(self.scheduler_status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save scheduler status: {str(e)}")
    
    def load_status(self) -> Dict[str, Any]:
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
            'started_at': None
        }
    
    def is_running(self) -> bool:
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
    
    def save_pid(self, pid: Optional[int] = None) -> None:
        """Save process PID to file.
        
        Args:
            pid: Process ID to save, or current process if None
        """
        try:
            with open(self.scheduler_pid_file, 'w') as f:
                f.write(str(pid or os.getpid()))
        except Exception as e:
            logger.warning(f"Failed to save PID file: {str(e)}")
    
    def remove_pid_file(self) -> None:
        """Remove PID file."""
        try:
            if os.path.exists(self.scheduler_pid_file):
                os.remove(self.scheduler_pid_file)
        except Exception as e:
            logger.warning(f"Failed to remove PID file: {str(e)}")
    
    def stop_daemon(self) -> bool:
        """Stop the scheduler daemon if running.
        
        Returns:
            bool: True if daemon was stopped, False otherwise
        """
        status = self.load_status()
        if not status.get('is_running') or not status.get('pid'):
            return False
            
        try:
            pid = status['pid']
            os.kill(pid, signal.SIGTERM)
            
            # Update status
            self.save_status(False)
            self.remove_pid_file()
            
            return True
        except OSError:
            # Process doesn't exist
            self.save_status(False)
            self.remove_pid_file()
            return False