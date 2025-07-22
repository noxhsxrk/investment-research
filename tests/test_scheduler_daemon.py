"""Unit tests for scheduler daemon."""

import pytest
import os
import json
import signal
from unittest.mock import patch, MagicMock
from datetime import datetime

from stock_analysis.services.scheduler_daemon import SchedulerDaemon


@pytest.fixture
def scheduler_daemon(tmp_path):
    """Create a scheduler daemon instance for testing."""
    daemon = SchedulerDaemon()
    daemon.scheduler_status_file = str(tmp_path / "scheduler_status.json")
    daemon.scheduler_pid_file = str(tmp_path / "scheduler.pid")
    return daemon


class TestSchedulerDaemon:
    """Test cases for SchedulerDaemon."""
    
    def test_save_status_running(self, scheduler_daemon):
        """Test saving running status."""
        scheduler_daemon.save_status(True, 12345)
        
        with open(scheduler_daemon.scheduler_status_file) as f:
            status = json.load(f)
            assert status['is_running'] is True
            assert status['pid'] == 12345
            assert status['started_at'] is not None
    
    def test_save_status_not_running(self, scheduler_daemon):
        """Test saving not running status."""
        scheduler_daemon.save_status(False)
        
        with open(scheduler_daemon.scheduler_status_file) as f:
            status = json.load(f)
            assert status['is_running'] is False
            assert status['pid'] is None
            assert status['started_at'] is None
    
    def test_load_status_file_exists(self, scheduler_daemon):
        """Test loading status when file exists."""
        # Create test status file
        test_status = {
            'is_running': True,
            'pid': 12345,
            'started_at': datetime.now().isoformat()
        }
        with open(scheduler_daemon.scheduler_status_file, 'w') as f:
            json.dump(test_status, f)
        
        status = scheduler_daemon.load_status()
        assert status['is_running'] is True
        assert status['pid'] == 12345
        assert status['started_at'] is not None
    
    def test_load_status_file_not_exists(self, scheduler_daemon):
        """Test loading status when file doesn't exist."""
        status = scheduler_daemon.load_status()
        assert status['is_running'] is False
        assert status['pid'] is None
        assert status['started_at'] is None
    
    def test_is_running_no_pid_file(self, scheduler_daemon):
        """Test is_running when PID file doesn't exist."""
        assert scheduler_daemon.is_running() is False
    
    @patch('os.kill')
    def test_is_running_process_exists(self, mock_kill, scheduler_daemon):
        """Test is_running when process exists."""
        # Create PID file
        with open(scheduler_daemon.scheduler_pid_file, 'w') as f:
            f.write('12345')
        
        # Mock os.kill to indicate process exists
        mock_kill.return_value = None
        
        assert scheduler_daemon.is_running() is True
        mock_kill.assert_called_once_with(12345, 0)
    
    @patch('os.kill')
    def test_is_running_process_not_exists(self, mock_kill, scheduler_daemon):
        """Test is_running when process doesn't exist."""
        # Create PID file
        with open(scheduler_daemon.scheduler_pid_file, 'w') as f:
            f.write('12345')
        
        # Mock os.kill to indicate process doesn't exist
        mock_kill.side_effect = OSError()
        
        assert scheduler_daemon.is_running() is False
        mock_kill.assert_called_once_with(12345, 0)
        assert not os.path.exists(scheduler_daemon.scheduler_pid_file)
    
    def test_save_pid(self, scheduler_daemon):
        """Test saving PID to file."""
        scheduler_daemon.save_pid(12345)
        
        with open(scheduler_daemon.scheduler_pid_file) as f:
            pid = int(f.read().strip())
            assert pid == 12345
    
    def test_remove_pid_file(self, scheduler_daemon):
        """Test removing PID file."""
        # Create PID file
        with open(scheduler_daemon.scheduler_pid_file, 'w') as f:
            f.write('12345')
        
        scheduler_daemon.remove_pid_file()
        assert not os.path.exists(scheduler_daemon.scheduler_pid_file)
    
    @patch('os.kill')
    def test_stop_daemon_running(self, mock_kill, scheduler_daemon):
        """Test stopping daemon when it's running."""
        # Create status file indicating daemon is running
        test_status = {
            'is_running': True,
            'pid': 12345,
            'started_at': datetime.now().isoformat()
        }
        with open(scheduler_daemon.scheduler_status_file, 'w') as f:
            json.dump(test_status, f)
        
        # Create PID file
        with open(scheduler_daemon.scheduler_pid_file, 'w') as f:
            f.write('12345')
        
        # Mock os.kill
        mock_kill.return_value = None
        
        assert scheduler_daemon.stop_daemon() is True
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
        
        # Verify status was updated
        status = scheduler_daemon.load_status()
        assert status['is_running'] is False
        assert status['pid'] is None
        assert not os.path.exists(scheduler_daemon.scheduler_pid_file)
    
    def test_stop_daemon_not_running(self, scheduler_daemon):
        """Test stopping daemon when it's not running."""
        # Create status file indicating daemon is not running
        test_status = {
            'is_running': False,
            'pid': None,
            'started_at': None
        }
        with open(scheduler_daemon.scheduler_status_file, 'w') as f:
            json.dump(test_status, f)
        
        assert scheduler_daemon.stop_daemon() is False
    
    @patch('os.kill')
    def test_stop_daemon_process_not_exists(self, mock_kill, scheduler_daemon):
        """Test stopping daemon when process doesn't exist."""
        # Create status file indicating daemon is running
        test_status = {
            'is_running': True,
            'pid': 12345,
            'started_at': datetime.now().isoformat()
        }
        with open(scheduler_daemon.scheduler_status_file, 'w') as f:
            json.dump(test_status, f)
        
        # Mock os.kill to indicate process doesn't exist
        mock_kill.side_effect = OSError()
        
        assert scheduler_daemon.stop_daemon() is False
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
        
        # Verify status was updated
        status = scheduler_daemon.load_status()
        assert status['is_running'] is False
        assert status['pid'] is None 