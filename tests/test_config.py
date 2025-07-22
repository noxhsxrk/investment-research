"""Unit tests for configuration manager."""

import pytest
import os
import yaml
from unittest.mock import patch, MagicMock

from stock_analysis.utils.config import ConfigManager
from stock_analysis.utils.exceptions import ConfigurationError


@pytest.fixture
def config_manager(tmp_path):
    """Create a config manager instance for testing."""
    config_file = tmp_path / "test_config.yaml"
    manager = ConfigManager(str(config_file))
    return manager


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    def test_default_config_loaded(self, config_manager):
        """Test that default configuration is loaded."""
        assert config_manager.get('stock_analysis.retry_attempts') == 3
        assert config_manager.get('stock_analysis.retry_delay') == 30
        assert config_manager.get('stock_analysis.logging.level') == 'INFO'
    
    def test_get_with_default(self, config_manager):
        """Test getting configuration with default value."""
        value = config_manager.get('nonexistent.key', default='default')
        assert value == 'default'
    
    def test_set_configuration(self, config_manager):
        """Test setting configuration value."""
        config_manager.set('test.key', 'test_value')
        assert config_manager.get('test.key') == 'test_value'
    
    def test_file_config_override(self, config_manager):
        """Test that file configuration overrides defaults."""
        # Create test config file
        config_data = {
            'stock_analysis': {
                'retry_attempts': 5,
                'retry_delay': 60,
                'logging': {
                    'level': 'DEBUG'
                }
            }
        }
        
        with open(config_manager.config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Reload configuration
        config_manager.load_config()
        
        assert config_manager.get('stock_analysis.retry_attempts') == 5
        assert config_manager.get('stock_analysis.retry_delay') == 60
        assert config_manager.get('stock_analysis.logging.level') == 'DEBUG'
    
    def test_env_variable_override(self, config_manager):
        """Test that environment variables override file configuration."""
        with patch.dict('os.environ', {
            'STOCK_ANALYSIS_RETRY_ATTEMPTS': '10',
            'STOCK_ANALYSIS_RETRY_DELAY': '90',
            'STOCK_ANALYSIS_LOGGING_LEVEL': 'ERROR'
        }):
            config_manager.load_config()
            
            assert config_manager.get('stock_analysis.retry_attempts') == 10
            assert config_manager.get('stock_analysis.retry_delay') == 90
            assert config_manager.get('stock_analysis.logging.level') == 'ERROR'
    
    def test_save_config(self, config_manager):
        """Test saving configuration to file."""
        config_manager.set('test.key', 'test_value')
        config_manager.save_config()
        
        # Verify file was created
        assert os.path.exists(config_manager.config_file)
        
        # Read back and verify
        with open(config_manager.config_file) as f:
            config_data = yaml.safe_load(f)
            assert config_data['test']['key'] == 'test_value'
    
    def test_nonexistent_config_file(self, config_manager):
        """Test loading configuration from nonexistent file."""
        # Should use default values
        assert config_manager.get('stock_analysis.retry_attempts') == 3
        assert config_manager.get('stock_analysis.retry_delay') == 30
        assert config_manager.get('stock_analysis.logging.level') == 'INFO'
    
    def test_config_validation_success(self, config_manager):
        """Test successful configuration validation."""
        config_data = {
            'stock_analysis': {
                'retry_attempts': 5,
                'retry_delay': 60,
                'logging': {
                    'level': 'DEBUG'
                }
            }
        }
        
        # Should not raise any exceptions
        config_manager._validate_config(config_data)
    
    def test_config_validation_failure(self, config_manager):
        """Test configuration validation failure."""
        config_data = {
            'stock_analysis': {
                'retry_attempts': 'invalid',  # Should be integer
                'retry_delay': -30,  # Should be positive
                'logging': {
                    'level': 'INVALID'  # Invalid log level
                }
            }
        }
        
        with pytest.raises(ConfigurationError):
            config_manager._validate_config(config_data)
    
    def test_dynamic_config_updates(self, config_manager):
        """Test dynamic configuration updates."""
        # Set up a callback
        callback_called = False
        def callback(key, value):
            nonlocal callback_called
            callback_called = True
        
        # Register callback
        config_manager.register_callback('test.key', callback)
        
        # Update config
        config_manager.set('test.key', 'new_value')
        
        assert callback_called
        assert config_manager.get('test.key') == 'new_value'
    
    def test_callback_registration(self, config_manager):
        """Test callback registration and unregistration."""
        callback = MagicMock()
        
        # Register callback
        config_manager.register_callback('test.key', callback)
        
        # Update config
        config_manager.set('test.key', 'test_value')
        
        # Verify callback was called
        callback.assert_called_once_with('test.key', 'test_value')
        
        # Unregister callback
        config_manager.unregister_callback('test.key', callback)
        
        # Update config again
        config_manager.set('test.key', 'new_value')
        
        # Verify callback was not called again
        assert callback.call_count == 1