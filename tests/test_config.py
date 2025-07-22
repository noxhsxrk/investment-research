"""Tests for configuration management."""

import os
import time
import tempfile
import pytest
import yaml
import threading
from unittest.mock import patch, MagicMock

from stock_analysis.utils.config import ConfigManager
from stock_analysis.utils.exceptions import ConfigurationError


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    def test_default_config_loaded(self):
        """Test that default configuration is loaded."""
        config = ConfigManager()
        
        assert config.get('stock_analysis.data_sources.yfinance.timeout') == 30
        assert config.get('stock_analysis.logging.level') == 'INFO'
        assert config.get('stock_analysis.export.default_format') == 'excel'
    
    def test_get_with_default(self):
        """Test getting configuration with default value."""
        config = ConfigManager()
        
        # Existing key
        assert config.get('stock_analysis.logging.level') == 'INFO'
        
        # Non-existing key with default
        assert config.get('non.existing.key', 'default_value') == 'default_value'
        
        # Non-existing key without default
        assert config.get('non.existing.key') is None
    
    def test_set_configuration(self):
        """Test setting configuration values."""
        config = ConfigManager()
        
        config.set('stock_analysis.logging.level', 'DEBUG')
        assert config.get('stock_analysis.logging.level') == 'DEBUG'
        
        # Test nested key creation
        config.set('new.nested.key', 'value')
        assert config.get('new.nested.key') == 'value'
    
    def test_file_config_override(self):
        """Test that file configuration overrides defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'stock_analysis': {
                    'logging': {
                        'level': 'DEBUG'
                    },
                    'export': {
                        'default_format': 'csv'
                    }
                }
            }, f)
            config_path = f.name
        
        try:
            config = ConfigManager(config_path)
            
            # Overridden values
            assert config.get('stock_analysis.logging.level') == 'DEBUG'
            assert config.get('stock_analysis.export.default_format') == 'csv'
            
            # Non-overridden values should remain default
            assert config.get('stock_analysis.data_sources.yfinance.timeout') == 30
        finally:
            os.unlink(config_path)
    
    def test_env_variable_override(self):
        """Test that environment variables override configuration."""
        with patch.dict(os.environ, {
            'STOCK_ANALYSIS_LOG_LEVEL': 'ERROR',
            'STOCK_ANALYSIS_YFINANCE_TIMEOUT': '60'
        }):
            config = ConfigManager()
            
            assert config.get('stock_analysis.logging.level') == 'ERROR'
            assert config.get('stock_analysis.data_sources.yfinance.timeout') == 60
    
    def test_save_config(self):
        """Test saving configuration to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            config = ConfigManager()
            config.set('stock_analysis.logging.level', 'WARNING')
            config.save_config(config_path)
            
            # Load saved config
            with open(config_path, 'r') as f:
                saved_config = yaml.safe_load(f)
            
            assert saved_config['stock_analysis']['logging']['level'] == 'WARNING'
        finally:
            os.unlink(config_path)
    
    def test_nonexistent_config_file(self):
        """Test behavior with non-existent config file."""
        config = ConfigManager('nonexistent.yaml')
        
        # Should still load defaults
        assert config.get('stock_analysis.logging.level') == 'INFO'
    
    def test_config_validation_success(self):
        """Test successful configuration validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'stock_analysis': {
                    'data_sources': {
                        'yfinance': {
                            'timeout': 60,
                            'retry_attempts': 5
                        }
                    },
                    'logging': {
                        'level': 'DEBUG'
                    }
                }
            }, f)
            config_path = f.name
        
        try:
            # Should not raise any validation errors
            config = ConfigManager(config_path)
            assert config.get('stock_analysis.data_sources.yfinance.timeout') == 60
        finally:
            os.unlink(config_path)
    
    def test_config_validation_failure(self):
        """Test configuration validation failure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'stock_analysis': {
                    'data_sources': {
                        'yfinance': {
                            'timeout': -10,  # Invalid: below minimum
                            'retry_attempts': 20  # Invalid: above maximum
                        }
                    },
                    'logging': {
                        'level': 'INVALID_LEVEL'  # Invalid: not in allowed values
                    }
                }
            }, f)
            config_path = f.name
        
        try:
            with pytest.raises(ConfigurationError):
                ConfigManager(config_path)
        finally:
            os.unlink(config_path)
    
    def test_dynamic_config_updates(self):
        """Test dynamic configuration updates."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'stock_analysis': {
                    'logging': {
                        'level': 'INFO'
                    }
                }
            }, f)
            config_path = f.name
        
        try:
            config = ConfigManager(config_path)
            
            # Set up a callback to track changes
            callback_called = threading.Event()
            
            def on_config_change():
                callback_called.set()
            
            config.register_callback(on_config_change)
            config.start_watching()
            
            # Update the config file
            time.sleep(0.1)  # Small delay
            with open(config_path, 'w') as f:
                yaml.dump({
                    'stock_analysis': {
                        'logging': {
                            'level': 'DEBUG'
                        }
                    }
                }, f)
            
            # Wait for the callback to be called (with timeout)
            callback_called.wait(timeout=10)
            
            # Stop watching
            config.stop_watching()
            
            # Check that the config was updated
            assert callback_called.is_set()
            assert config.get('stock_analysis.logging.level') == 'DEBUG'
        finally:
            os.unlink(config_path)
    
    def test_callback_registration(self):
        """Test callback registration and unregistration."""
        config = ConfigManager()
        
        callback1 = MagicMock()
        callback2 = MagicMock()
        
        # Register callbacks
        config.register_callback(callback1)
        config.register_callback(callback2)
        
        # Manually trigger callbacks
        config._notify_callbacks()
        
        assert callback1.call_count == 1
        assert callback2.call_count == 1
        
        # Unregister one callback
        config.unregister_callback(callback1)
        
        # Trigger again
        config._notify_callbacks()
        
        assert callback1.call_count == 1  # Should not have been called again
        assert callback2.call_count == 2  # Should have been called again